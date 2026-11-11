"""Top-k versus full-distribution convergence (task 7.12; Requirements 9.1 to 9.4).

A study module of its own, separate from the pre-registered analyses (7.10 and
7.11). It reads only recorded data and sends no request: the frozen plan, the
item results of the 7.7 execution, the replies of the draw recording (the 7.4
draw log of :class:`~scripts.finance_sim.execute.ExecutionDraw`), the
full-distribution evidence of each draw and the arm's calibrators. The
analysis is defined on the arm that records full-distribution evidence (the
local sampling arm, the probe arm of the 2026-11-10 sizing decision) and on the
stages whose items enter the analyses (``full_run`` and ``negative_control``;
pilot items are excluded from every analysis). Every size comes from the plan
and the records: the planned items and renderings are
:func:`~scripts.finance_sim.execute.planned_keys` of the frozen plan, the
ensemble size and minimum are the results', and the kept alternatives are the
evidence's.

Comparisons, for the same responses of the same model and weights (9.1):

- ``host_top_k`` at each k of :data:`CONVERGENCE_KS` up to the kept count: the
  draw's full-distribution evidence truncated to its first k alternatives
  (``truncate_to_top_k``), with the exact actual-token logprob, through the
  full-distribution calibrator. The estimator is held fixed, so only the
  evidence changes, and at k equal to the kept count the row is identical to
  the full distribution by construction. A k above the kept count is reported
  as missing with the reason.
- ``served_top_k`` at the arm's ``top_logprobs``: the generation host's own
  top-k logprobs recorded with the reply, through the top-k calibrator, which
  is the study's top_k estimate source.

The reference is ``features_from_full`` on the same evidence through the
full-distribution calibrator. Before anything is compared, each recorded
estimate must reproduce: the full-distribution estimate from the evidence whose
SHA-256 the result records, and the top-k estimate from the recorded reply.
Evidence is matched to its draw by that SHA-256 only, so a store holding other
evidence cannot be mistaken for it.

Agreement (9.2), per feature (``loss``, ``min_k``, ``min_k_pp``,
``zlib_ratio`` and the estimate) and comparison: the mean and the maximum
absolute difference to the full-distribution value and the share of draws
within :data:`AGREEMENT_TOLERANCE`. Mass: per draw the mean over positions of
the probability the first k alternatives cover (``top_k_mass``), averaged over
draws, with the smallest position value.

Downstream decision (9.3). Per item, rendering and draw mode the decision is
recomputed with ``decide_from_draws`` from ``ItemResult.draws_for``: the full
distribution's draws as recorded, the same draws with each estimate replaced by
the comparison's estimate, and for the served row the top-k draws as recorded.
The full and top-k decisions must equal the stored variants. Reported: the
status, ``p_memorized``, the summary value of the adjusted decision and its
category on both sides, and per comparison and mode the mean and maximum
absolute changes and the shares of items whose adjusted category or status
changed.

Values compare only within one model and one set of weights (9.4): the
evidence of one table must carry one model digest (the arm's pinned GGUF
digest when read from the records) and the table states it. Anything that
cannot be computed is reported as missing with its reason, never substituted.

Outputs are derived statistics only, never a reply: ``convergence-<arm>-<stage>.json``
and its markdown table under ``runs/finance-sim/analysis/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from recall_guard.core.completion import ModelResponse
from recall_guard.core.distribution import RecordedDraw
from recall_guard.core.full_distribution import FullDistributionEvidence
from recall_guard.harness.guarded_decision import (
    DrawMode,
    GuardConfig,
    GuardedDecision,
    GuardIdentity,
    SourceCalibrator,
    decide_from_draws,
)
from recall_guard.mia.estimate_source import features_from_full, features_from_top_k, top_k_mass, truncate_to_top_k
from recall_guard.mia.features import MiaFeatures
from scripts.finance_sim.calibrate import CALIBRATORS_DIR, FEATURE_ORDER, CalibrationRecord
from scripts.finance_sim.execute import (
    DEFINITIONS,
    EXECUTION_STAGES,
    RESULTS_DIR,
    ExecutionDraw,
    ItemResult,
    full_distribution_estimate,
    load_calibrators,
    load_result,
    planned_keys,
    result_path,
)
from scripts.finance_sim.limits import ARMS, RECORDS_DIR, RUNS_ROOT, SAMPLING_ARM
from scripts.finance_sim.plan import (
    FULL_DISTRIBUTION,
    SINGLE_DRAW_INDEX,
    TOP_K,
    ArmPlan,
    StudyPlan,
    ensure_unchanged,
    load_plan,
    require_complete,
)
from scripts.finance_sim.recording import (
    EVIDENCE_SCHEMA,
    draw_log_path,
    evidence_path,
    pinned_digest,
    read_draw_log,
)

#: The frozen k values (design, Analysis inputs); a k above the evidence's kept count is reported missing.
CONVERGENCE_KS: tuple[int, ...] = (1, 5, 10, 20, 50, 100)
#: The calibrators' features, then the estimate they give.
FEATURES: tuple[str, ...] = (*FEATURE_ORDER, "estimate")
#: A value within this absolute difference of the full-distribution value counts as agreeing.
AGREEMENT_TOLERANCE = 1e-6
#: Comparison sources: the full-distribution evidence truncated to k, and the generation host's own top-k.
HOST_TOP_K = "host_top_k"
SERVED_TOP_K = "served_top_k"
EVIDENCE_KINDS: tuple[str, ...] = (HOST_TOP_K, SERVED_TOP_K)
#: Draw modes, the ensemble first.
MODES: tuple[str, ...] = ("ensemble", "single")
#: Stages whose items enter the analyses; pilot items are excluded from every analysis.
CONVERGENCE_STAGES: tuple[str, ...] = tuple(stage for stage in EXECUTION_STAGES if stage != "pilot")
TABLE_SCHEMA = "finance-sim-convergence/1"
#: Analysis outputs (design: ``runs/finance-sim/analysis/*.json``).
CONVERGENCE_DIR = RUNS_ROOT / "analysis"
NO_RESULT = "no stored result: the stage has not executed this item and rendering"
NO_REPLY = "the reply is not in the draw recording"
FULL_ESTIMATE_UNAVAILABLE = "the full-distribution estimate is unavailable for this draw"
ESTIMATE_UNAVAILABLE = "the estimate is unavailable: the calibrator cannot use these features"
NO_DECISION = "no decision: fewer parsed draws than the frozen minimum (the stored result gives the counts)"
DRAWS_LACKING = ("a parsed draw lacks its reply or full-distribution evidence (listed in missing_draws), so the "
                 "decision at this k is not compared")
EXIT_WRITTEN, EXIT_REFUSED = 0, 2
_SHA256_LENGTH = 64


class ConvergenceError(ValueError):
    """Recorded inputs that do not reproduce, or an arm or stage the analysis is not defined on."""


# ---------- statements in the table ----------

def comparability_statement(model: str, model_digest: str) -> str:
    """Requirement 9.4, stated in every table."""
    return (f"Values compare only within one model and one set of weights: {model} with weights {model_digest}. "
            "Values from different builds or model families are not comparable (Requirement 9.4).")


def scope_statement(kept: int) -> str:
    """What "full distribution" means here (design, EstimateSources)."""
    return (f"Full-distribution features use the kept top-{kept} alternatives taken from the full next-token vector "
            "plus the exact actual-token logprob; Min-K%++ uses moments of the kept alternatives, not of the whole "
            f"vocabulary. Agreement at k = {kept} is identical by construction.")


ESTIMATOR_NOTE = (
    "Host top-k rows truncate each draw's full-distribution evidence to its first k alternatives and keep the "
    "full-distribution calibrator, so only the evidence changes. The served row uses the generation host's own "
    "top-k logprobs through the top-k calibrator, the study's top_k estimate source. Agreement is the mean and "
    f"maximum absolute difference to the full-distribution value and the share of draws within {AGREEMENT_TOLERANCE:g}; "
    "mass is the probability the first k alternatives cover, per position. Decisions are recomputed from the "
    "recorded draws with decide_from_draws; no request is sent."
)


# ---------- helpers ----------

def _payload_sha256(payload: object) -> str:
    """SHA-256 of a JSON payload in canonical form: the evidence reference the guarded-decision API records."""
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_or_none(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _mean(values: Sequence[float]) -> float | None:
    return math.fsum(values) / len(values) if values else None


def _estimate(calibrator: SourceCalibrator, features: MiaFeatures) -> float | None:
    """The calibrator's estimate as the guarded-decision API records one: None when unusable or non-finite."""
    try:
        return _finite_or_none(calibrator.estimate(features))
    except (ValueError, RuntimeError):
        return None


def _values(features: MiaFeatures, estimate: float | None) -> dict[str, float | None]:
    return {name: float(getattr(features, name)) for name in FEATURE_ORDER} | {"estimate": estimate}


def _counts(counter: Counter) -> Mapping[str, int]:
    return MappingProxyType(dict(sorted(counter.items())))


def _checked_ks(ks: object) -> tuple[int, ...]:
    if isinstance(ks, str | bytes) or not isinstance(ks, Iterable):
        raise ConvergenceError("ks must be a non-empty, strictly increasing sequence of positive integers")
    values = tuple(ks)
    if not values or not all(_is_int(k) and k >= 1 for k in values) or any(
            later <= earlier for earlier, later in zip(values, values[1:], strict=False)):
        raise ConvergenceError(f"ks must be a non-empty, strictly increasing sequence of positive integers; "
                               f"got {list(values)!r}")
    return values


# ---------- table rows ----------

@dataclass(frozen=True)
class FeatureAgreement:
    """Agreement of one feature between one comparison and the full distribution, over the compared draws.

    ``missing`` counts the draws whose value could not be compared, by reason;
    ``missing_reason`` is set when the whole comparison cannot be computed.
    """

    evidence: str
    k: int
    feature: str
    n: int
    mean_abs_diff: float | None
    max_abs_diff: float | None
    share_within: float | None
    missing: Mapping[str, int] = field(default_factory=lambda: MappingProxyType({}))
    missing_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"evidence": self.evidence, "k": self.k, "feature": self.feature, "n": self.n,
                "mean_abs_diff": self.mean_abs_diff, "max_abs_diff": self.max_abs_diff,
                "share_within": self.share_within, "missing": dict(self.missing),
                "missing_reason": self.missing_reason}


@dataclass(frozen=True)
class MassCoverage:
    """Probability mass the comparison's alternatives cover: mean over draws of the per-draw mean, and the minimum."""

    evidence: str
    k: int
    n_draws: int
    mean_mass: float | None
    min_mass: float | None
    missing: Mapping[str, int] = field(default_factory=lambda: MappingProxyType({}))
    missing_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"evidence": self.evidence, "k": self.k, "n_draws": self.n_draws, "mean_mass": self.mean_mass,
                "min_mass": self.min_mass, "missing": dict(self.missing), "missing_reason": self.missing_reason}


@dataclass(frozen=True)
class ItemDecisionChange:
    """One item and rendering in one draw mode: the full-distribution decision against the comparison's.

    ``adjusted_summary`` is the summary value of the adjusted decision (the
    defensive spread or the outlook) and ``adjusted_category`` its category;
    None where a component is unresolved or the decision failed.
    """

    item_id: str
    rendering: str
    evidence: str
    k: int
    mode: str
    status_full: str | None
    status: str | None
    p_full: float | None
    p: float | None
    adjusted_summary_full: float | None
    adjusted_summary: float | None
    adjusted_category_full: Hashable | None
    adjusted_category: Hashable | None
    missing_reason: str | None = None

    @property
    def abs_p_change(self) -> float | None:
        return None if self.p is None or self.p_full is None else abs(self.p - self.p_full)

    @property
    def abs_adjusted_summary_change(self) -> float | None:
        if self.adjusted_summary is None or self.adjusted_summary_full is None:
            return None
        return abs(self.adjusted_summary - self.adjusted_summary_full)

    @property
    def category_changed(self) -> bool | None:
        if self.adjusted_category is None or self.adjusted_category_full is None:
            return None
        return self.adjusted_category != self.adjusted_category_full

    @property
    def status_changed(self) -> bool | None:
        return None if self.missing_reason is not None else self.status != self.status_full

    def to_dict(self) -> dict[str, Any]:
        return {"item_id": self.item_id, "rendering": self.rendering, "evidence": self.evidence, "k": self.k,
                "mode": self.mode, "status_full": self.status_full, "status": self.status, "p_full": self.p_full,
                "p": self.p, "adjusted_summary_full": self.adjusted_summary_full,
                "adjusted_summary": self.adjusted_summary, "adjusted_category_full": self.adjusted_category_full,
                "adjusted_category": self.adjusted_category, "abs_p_change": self.abs_p_change,
                "abs_adjusted_summary_change": self.abs_adjusted_summary_change,
                "category_changed": self.category_changed, "status_changed": self.status_changed,
                "missing_reason": self.missing_reason}


@dataclass(frozen=True)
class DecisionChange:
    """Change of the adjusted decision between the full distribution and one comparison, over the compared items."""

    evidence: str
    k: int
    mode: str
    n_items: int
    mean_abs_p_change: float | None
    max_abs_p_change: float | None
    mean_abs_adjusted_summary_change: float | None
    max_abs_adjusted_summary_change: float | None
    share_category_changed: float | None
    share_status_changed: float | None
    missing: Mapping[str, int] = field(default_factory=lambda: MappingProxyType({}))
    missing_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"evidence": self.evidence, "k": self.k, "mode": self.mode, "n_items": self.n_items,
                "mean_abs_p_change": self.mean_abs_p_change, "max_abs_p_change": self.max_abs_p_change,
                "mean_abs_adjusted_summary_change": self.mean_abs_adjusted_summary_change,
                "max_abs_adjusted_summary_change": self.max_abs_adjusted_summary_change,
                "share_category_changed": self.share_category_changed,
                "share_status_changed": self.share_status_changed, "missing": dict(self.missing),
                "missing_reason": self.missing_reason}


@dataclass(frozen=True)
class ConvergenceTable:
    """The convergence table of one arm and stage, with the statements it is read under (derived statistics only)."""

    plan_sha256: str
    arm: str
    stage: str
    kind: str
    model: str
    model_digest: str
    n_vocab: int
    kept: int
    ks: tuple[int, ...]
    served_k: int
    tolerance: float
    comparability: str
    scope: str
    estimator: str
    items: tuple[tuple[str, str], ...]
    n_draws: int
    full_mean_mass: float
    full_min_mass: float
    features: tuple[FeatureAgreement, ...]
    mass: tuple[MassCoverage, ...]
    decisions: tuple[DecisionChange, ...]
    item_changes: tuple[ItemDecisionChange, ...]
    missing_items: tuple[Mapping[str, str], ...]
    missing_draws: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible, with lists only, so it reads back from JSON unchanged."""
        return {
            "schema": TABLE_SCHEMA, "plan_sha256": self.plan_sha256, "arm": self.arm, "stage": self.stage,
            "kind": self.kind, "model": self.model, "model_digest": self.model_digest, "n_vocab": self.n_vocab,
            "kept": self.kept, "ks": list(self.ks), "served_k": self.served_k, "tolerance": self.tolerance,
            "comparability": self.comparability, "scope": self.scope, "estimator": self.estimator,
            "items": [list(key) for key in self.items], "n_draws": self.n_draws,
            "full_mean_mass": self.full_mean_mass, "full_min_mass": self.full_min_mass,
            "features": [row.to_dict() for row in self.features], "mass": [row.to_dict() for row in self.mass],
            "decisions": [row.to_dict() for row in self.decisions],
            "item_changes": [row.to_dict() for row in self.item_changes],
            "missing_items": [dict(entry) for entry in self.missing_items],
            "missing_draws": [dict(entry) for entry in self.missing_draws],
        }

    def to_markdown(self) -> str:
        """The table for the report: the statements, then agreement, mass and decision change."""
        lines = [
            "# Top-k versus full-distribution convergence", "",
            f"Arm {self.arm}, stage {self.stage} ({self.kind} items), model {self.model}, weights "
            f"{self.model_digest}, plan {self.plan_sha256}.", "",
            self.comparability, "", self.scope, "", self.estimator, "",
            f"Items compared: {len(self.items)} ({len(self.missing_items)} planned items missing); draws compared: "
            f"{self.n_draws} ({len(self.missing_draws)} draws missing). Kept alternatives: {self.kept} of a "
            f"{self.n_vocab}-token vocabulary, covering a mean mass of {_cell(self.full_mean_mass)} (minimum "
            f"{_cell(self.full_min_mass)}).", "",
            "## Feature agreement", "",
            f"| Evidence | k | Feature | Draws | Mean abs diff | Max abs diff | Share within {self.tolerance:g} | "
            "Missing |",
            "|---|---|---|---|---|---|---|---|",
        ]
        lines += [f"| {row.evidence} | {row.k} | {row.feature} | {row.n} | {_cell(row.mean_abs_diff)} | "
                  f"{_cell(row.max_abs_diff)} | {_cell(row.share_within)} | {_missing(row.missing, row.missing_reason)} |"
                  for row in self.features]
        lines += ["", "## Mass coverage", "", "| Evidence | k | Draws | Mean mass | Min mass | Missing |",
                  "|---|---|---|---|---|---|"]
        lines += [f"| {row.evidence} | {row.k} | {row.n_draws} | {_cell(row.mean_mass)} | {_cell(row.min_mass)} | "
                  f"{_missing(row.missing, row.missing_reason)} |" for row in self.mass]
        lines += ["", "## Adjusted decision change", "",
                  "| Evidence | k | Mode | Items | Mean abs change p_memorized | Max abs change p_memorized | "
                  "Mean abs change adjusted summary | Max abs change adjusted summary | Share category changed | "
                  "Share status changed | Missing |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        lines += [f"| {row.evidence} | {row.k} | {row.mode} | {row.n_items} | {_cell(row.mean_abs_p_change)} | "
                  f"{_cell(row.max_abs_p_change)} | {_cell(row.mean_abs_adjusted_summary_change)} | "
                  f"{_cell(row.max_abs_adjusted_summary_change)} | {_cell(row.share_category_changed)} | "
                  f"{_cell(row.share_status_changed)} | {_missing(row.missing, row.missing_reason)} |"
                  for row in self.decisions]
        if self.missing_items or self.missing_draws:
            lines += ["", "## Missing", ""]
            lines += [f"- {entry['item_id']} {entry['rendering']}: {entry['reason']}" for entry in self.missing_items]
            lines += [f"- {entry['item_id']} {entry['rendering']} draw {entry['draw_index']}: {entry['reason']}"
                      for entry in self.missing_draws]
        return "\n".join(lines) + "\n"


def _cell(value: float | None) -> str:
    return "missing" if value is None else f"{value:.6g}"


def _missing(counts: Mapping[str, int], reason: str | None) -> str:
    if reason is not None:
        return reason
    return "; ".join(f"{text}: {count}" for text, count in counts.items()) or "none"


# ---------- checks ----------

def _check_scope(plan: StudyPlan, arm: str, stage: str) -> ArmPlan:
    require_complete(plan)
    if arm not in ARMS:
        raise ConvergenceError(f"arm must be one of {list(ARMS)}; got {arm!r}")
    arm_plan = plan.arms.arms[arm]
    if FULL_DISTRIBUTION not in arm_plan.estimate_sources:
        reason = arm_plan.missing_variants.get(f"ensemble_{FULL_DISTRIBUTION}", "")
        raise ConvergenceError(f"the {arm} arm records no full_distribution evidence, so its top-k convergence is not "
                               f"defined; {reason}")
    if stage == "pilot":
        raise ConvergenceError("pilot items are excluded from every analysis; the convergence analysis runs on the "
                               f"stages {list(CONVERGENCE_STAGES)}")
    if stage not in CONVERGENCE_STAGES:
        raise ConvergenceError(f"stage must be one of {list(CONVERGENCE_STAGES)}; got {stage!r}")
    return arm_plan


def _checked_calibrators(plan: StudyPlan, arm: str, kind: str, arm_plan: ArmPlan,
                         calibrators: object) -> dict[str, CalibrationRecord]:
    if not isinstance(calibrators, Mapping) or set(calibrators) != set(arm_plan.estimate_sources):
        raise ConvergenceError(f"the {arm} arm needs one calibrator for each of its estimate sources "
                               f"{list(arm_plan.estimate_sources)}")
    for source, record in calibrators.items():
        if not isinstance(record, CalibrationRecord):
            raise TypeError(f"calibrators[{source!r}] must be a CalibrationRecord; got {type(record).__name__}")
        ensure_unchanged(plan, record.plan_sha256)
        if record.kind != kind or record.arm != arm or record.source.value != source:
            raise ConvergenceError(f"calibrators[{source!r}] is the {record.arm} arm's {record.kind} calibrator of the "
                                   f"{record.source.value} source; the {kind} items of the {arm} arm need their own")
    return {source: calibrators[source] for source in arm_plan.estimate_sources}


def _checked_results(plan: StudyPlan, arm: str, stage: str, arm_plan: ArmPlan,
                     records: Mapping[str, CalibrationRecord],
                     results: object) -> tuple[list[ItemResult], list[Mapping[str, str]]]:
    """The results in plan order and the planned items without one; any other result is refused."""
    if isinstance(results, str | bytes) or not isinstance(results, Iterable):
        raise TypeError("results must be a sequence of ItemResult values")
    planned = planned_keys(plan, arm, stage)
    given: dict[tuple[str, str], ItemResult] = {}
    for result in results:
        if not isinstance(result, ItemResult):
            raise TypeError(f"results must be ItemResult values; got {type(result).__name__}")
        ensure_unchanged(plan, result.plan_sha256)
        if (result.arm, result.stage) != (arm, stage):
            raise ConvergenceError(f"the result of {result.key} belongs to {result.arm}/{result.stage}, not "
                                   f"{arm}/{stage}")
        if result.key in given:
            raise ConvergenceError(f"the result of {result.key} is given more than once")
        if result.key not in planned:
            raise ConvergenceError(f"{result.key} is not an item and rendering the plan sizes for {arm}/{stage}")
        if (result.ensemble_draws, result.min_parsed) != (arm_plan.draws, arm_plan.min_parsed):
            raise ConvergenceError(f"the result of {result.key} holds {result.ensemble_draws} draws with minimum "
                                   f"{result.min_parsed}; the frozen arm draws {arm_plan.draws} with minimum "
                                   f"{arm_plan.min_parsed}")
        if result.full_distribution is None or FULL_DISTRIBUTION not in result.identities:
            raise ConvergenceError(f"the result of {result.key} carries no full-distribution estimates")
        for source, record in records.items():
            if result.identities[source].calibrator_fingerprint != record.calibrator.fingerprint:
                raise ConvergenceError(f"the {source} estimates of {result.key} come from another calibrator than "
                                       "the one given")
        given[result.key] = result
    if not given:
        raise ConvergenceError(f"no stored result for {arm}/{stage}: there is nothing to compare")
    models = {result.identities[TOP_K].model for result in given.values()}
    if len(models) != 1:
        raise ConvergenceError(f"the results come from more than one model {sorted(models)}; values compare only "
                               "within one model and one set of weights")
    ordered = [given[key] for key in planned if key in given]
    missing = [MappingProxyType({"item_id": item_id, "rendering": rendering, "reason": NO_RESULT})
               for item_id, rendering in planned if (item_id, rendering) not in given]
    return ordered, missing


def _index_draws(draws: object) -> dict[tuple[str, str, int], ExecutionDraw]:
    """Recorded draws by key, the settled version of a draw recorded twice; conflicting copies are refused."""
    if isinstance(draws, str | bytes) or not isinstance(draws, Iterable):
        raise TypeError("draws must be a sequence of ExecutionDraw values")
    by_key: dict[tuple[str, str, int], ExecutionDraw] = {}
    for draw in draws:
        if not isinstance(draw, ExecutionDraw):
            raise TypeError(f"draws must be ExecutionDraw values; got {type(draw).__name__}")
        earlier = by_key.get(draw.key)
        if earlier is not None and earlier != draw:
            if earlier.full_distribution is None and replace(draw, full_distribution=None) == earlier:
                pass  # The settled version of a pending draw replaces it.
            elif draw.full_distribution is None and replace(earlier, full_distribution=None) == draw:
                continue  # The pending version of a settled draw adds nothing.
            else:
                raise ConvergenceError(f"draw {draw.key} is in the draw recording more than once with different "
                                       "content")
        by_key[draw.key] = draw
    return by_key


def _index_evidence(evidence: object) -> dict[str, FullDistributionEvidence]:
    """Evidence by the SHA-256 of its canonical payload, the reference a result records per draw."""
    if isinstance(evidence, str | bytes) or not isinstance(evidence, Iterable):
        raise TypeError("evidence must be a sequence of FullDistributionEvidence values")
    by_ref: dict[str, FullDistributionEvidence] = {}
    for entry in evidence:
        if not isinstance(entry, FullDistributionEvidence):
            raise TypeError(f"evidence must be FullDistributionEvidence values; got {type(entry).__name__}")
        by_ref[_payload_sha256(entry.to_dict())] = entry
    return by_ref


# ---------- per draw ----------

@dataclass(frozen=True)
class _DrawValues:
    """One compared draw: its feature values per comparison and the mass its alternatives cover per position."""

    key: tuple[str, str]
    index: int
    digest: str
    n_vocab: int
    kept: int
    full: Mapping[str, float | None]
    full_mass: np.ndarray
    host: Mapping[int, Mapping[str, float | None]]
    host_mass: Mapping[int, np.ndarray]
    served: Mapping[str, float | None] | None
    served_mass: np.ndarray | None
    served_reason: str | None


def _served(response: ModelResponse, calibrator: SourceCalibrator) -> tuple[
        Mapping[str, float | None] | None, np.ndarray | None, str | None, float | None]:
    """Features, mass, unavailability reason and estimate of the reply's own top-k logprobs.

    The estimate is computed as the guarded-decision API computes the top-k
    estimate of a draw, so it can be compared with the recorded one.
    """
    if not response.token_evidence:
        return None, None, "the reply carries no top-k logprobs", None
    try:
        logprobs = response.to_legacy().logprobs
        features = features_from_top_k(response.content, logprobs).features
    except ValueError as exc:
        return None, None, f"the reply's top-k logprobs cannot give features: {exc}", None
    estimate = _estimate(calibrator, features)
    mass = np.asarray([float(np.sum(np.exp(np.asarray([alt["logprob"] for alt in entry.top_logprobs],
                                                      dtype=np.float64)))) for entry in logprobs], dtype=np.float64)
    return MappingProxyType(_values(features, estimate)), mass, None, estimate


def _draw_values(result: ItemResult, draw: RecordedDraw, recorded: ExecutionDraw, evidence: FullDistributionEvidence,
                 records: Mapping[str, CalibrationRecord], ks: Sequence[int]) -> _DrawValues | str:
    """The compared values of one parsed draw, or the reason it cannot be compared; a mismatch is refused."""
    where = f"draw {draw.draw_index} of {result.key}"
    stored = result.full_distribution[draw.draw_index]
    if full_distribution_estimate(recorded.response.content, evidence, records[FULL_DISTRIBUTION]) != stored:
        raise ConvergenceError(f"the recorded full-distribution estimate of {where} does not reproduce from its "
                               "evidence and the full-distribution calibrator")
    served, served_mass, served_reason, served_estimate = _served(recorded.response, records[TOP_K].calibrator)
    if served_estimate != draw.estimate:
        raise ConvergenceError(f"the recorded top_k estimate of {where} does not reproduce from its reply and the "
                               "top_k calibrator")
    content = recorded.response.content
    try:
        full = features_from_full(content, evidence)
    except ValueError as exc:
        return f"the full-distribution evidence cannot give features: {exc}"
    kept = full.label.k
    full_calibrator = records[FULL_DISTRIBUTION].calibrator
    host = {}
    host_mass = {}
    for k in ks:
        if k > kept:
            break
        features = features_from_top_k(content, truncate_to_top_k(evidence, k)).features
        host[k] = MappingProxyType(_values(features, _estimate(full_calibrator, features)))
        host_mass[k] = top_k_mass(evidence, k)
    return _DrawValues(
        key=result.key, index=draw.draw_index, digest=evidence.model_digest, n_vocab=evidence.n_vocab, kept=kept,
        full=MappingProxyType(_values(full.features, stored.estimate)), full_mass=top_k_mass(evidence, kept),
        host=MappingProxyType(host), host_mass=MappingProxyType(host_mass), served=served, served_mass=served_mass,
        served_reason=served_reason)


# ---------- aggregation ----------

def _agreement(evidence: str, k: int, feature: str,
               pairs: Sequence[tuple[Mapping[str, float | None], Mapping[str, float | None] | None, str | None]],
               ) -> FeatureAgreement:
    differences = []
    missing: Counter = Counter()
    for full, row, reason in pairs:
        if row is None:
            missing[reason] += 1
        elif full[feature] is None:
            missing[FULL_ESTIMATE_UNAVAILABLE] += 1
        elif row[feature] is None:
            missing[ESTIMATE_UNAVAILABLE] += 1
        else:
            differences.append(abs(row[feature] - full[feature]))
    return FeatureAgreement(
        evidence, k, feature, n=len(differences), mean_abs_diff=_mean(differences),
        max_abs_diff=max(differences) if differences else None,
        share_within=(sum(value <= AGREEMENT_TOLERANCE for value in differences) / len(differences)
                      if differences else None), missing=_counts(missing))


def _coverage(evidence: str, k: int, masses: Sequence[np.ndarray | None], reasons: Sequence[str | None]) -> MassCoverage:
    present = [mass for mass in masses if mass is not None and mass.size]
    missing = Counter(reason for mass, reason in zip(masses, reasons, strict=True) if mass is None)
    return MassCoverage(evidence, k, n_draws=len(present), mean_mass=_mean([float(np.mean(mass)) for mass in present]),
                        min_mass=min(float(np.min(mass)) for mass in present) if present else None,
                        missing=_counts(missing))


def _summary(definition: Any, vector: Mapping[str, float | None] | None) -> tuple[float | None, Hashable | None]:
    if vector is None or any(value is None for value in vector.values()):
        return None, None
    values = dict(vector)
    category = definition.decide(values) if definition.decide is not None else None
    return float(definition.summary(values)), category


def _decide(result: ItemResult, draws: Sequence[RecordedDraw], identity: GuardIdentity, mode: str,
            arm_plan: ArmPlan) -> GuardedDecision:
    """The decision ``decide_from_draws`` gives for these draws, as the 7.7 execution derives each variant."""
    definition = DEFINITIONS[result.kind]
    if mode == "single":
        config = GuardConfig(definition, identity.source, draws=1, min_parsed=1, mode=DrawMode.SINGLE,
                             temperature=arm_plan.temperature, seed_base=arm_plan.seed_base)
        return decide_from_draws((draws[SINGLE_DRAW_INDEX],), config, replace(identity, draws=1))
    config = GuardConfig(definition, identity.source, draws=result.ensemble_draws, min_parsed=result.min_parsed,
                         temperature=arm_plan.temperature, seed_base=arm_plan.seed_base)
    return decide_from_draws(draws, config, identity)


def _check_variant(result: ItemResult, mode: str, source: str, decision: GuardedDecision) -> None:
    stored = result.variant(f"{mode}_{source}")
    adjusted = None if decision.adjusted is None else dict(decision.adjusted)
    if stored.missing is not None or (stored.status, stored.p_memorized, None if stored.adjusted is None else dict(
            stored.adjusted)) != (decision.status, decision.p_memorized, adjusted):
        raise ConvergenceError(f"the stored {mode}_{source} variant of {result.key} does not recompute from its "
                               "recorded draws")


def _change(result: ItemResult, evidence: str, k: int, mode: str, full: GuardedDecision,
            other: GuardedDecision | None, reason: str | None) -> ItemDecisionChange:
    definition = DEFINITIONS[result.kind]
    full_summary, full_category = _summary(definition, full.adjusted)
    if other is None:
        summary = category = None
    else:
        summary, category = _summary(definition, other.adjusted)
    return ItemDecisionChange(
        result.item_id, result.rendering, evidence, k, mode, status_full=full.status,
        status=None if other is None else other.status, p_full=full.p_memorized,
        p=None if other is None else other.p_memorized, adjusted_summary_full=full_summary, adjusted_summary=summary,
        adjusted_category_full=full_category, adjusted_category=category, missing_reason=reason)


def _decision_row(evidence: str, k: int, mode: str, changes: Sequence[ItemDecisionChange]) -> DecisionChange:
    compared = [change for change in changes if change.missing_reason is None]
    p_changes = [change.abs_p_change for change in compared if change.abs_p_change is not None]
    summary_changes = [change.abs_adjusted_summary_change for change in compared
                       if change.abs_adjusted_summary_change is not None]
    categories = [change.category_changed for change in compared if change.category_changed is not None]
    return DecisionChange(
        evidence, k, mode, n_items=len(compared), mean_abs_p_change=_mean(p_changes),
        max_abs_p_change=max(p_changes) if p_changes else None,
        mean_abs_adjusted_summary_change=_mean(summary_changes),
        max_abs_adjusted_summary_change=max(summary_changes) if summary_changes else None,
        share_category_changed=sum(categories) / len(categories) if categories else None,
        share_status_changed=(sum(change.status_changed for change in compared) / len(compared)
                              if compared else None),
        missing=_counts(Counter(change.missing_reason for change in changes if change.missing_reason is not None)))


# ---------- the table ----------

def convergence_table(
    plan: StudyPlan, results: Iterable[ItemResult], draws: Iterable[ExecutionDraw],
    evidence: Iterable[FullDistributionEvidence], *, arm: str, stage: str,
    calibrators: Mapping[str, CalibrationRecord], ks: Sequence[int] = CONVERGENCE_KS,
    expected_digest: str | None = None,
) -> ConvergenceTable:
    """The convergence table of one arm and stage from recorded results, replies and evidence; no request is sent.

    ``results`` are the stored item results of the stage (planned items
    without one are reported missing), ``draws`` the draw recording's latest
    versions (their replies give the features), ``evidence`` the recorded
    full-distribution evidence (matched to each draw by the SHA-256 its
    result records) and ``calibrators`` the arm's calibrators for the stage's
    item kind. ``expected_digest`` is the weights every evidence object must
    carry (the arm's pinned GGUF digest); evidence of more than one digest is
    always refused.
    """
    arm_plan = _check_scope(plan, arm, stage)
    ks = _checked_ks(ks)
    kind = EXECUTION_STAGES[stage]
    records = _checked_calibrators(plan, arm, kind, arm_plan, calibrators)
    ordered, missing_items = _checked_results(plan, arm, stage, arm_plan, records, results)
    replies = _index_draws(draws)
    by_ref = _index_evidence(evidence)

    compared: list[_DrawValues] = []
    missing_draws: list[Mapping[str, Any]] = []
    for result in ordered:
        for draw in result.recorded_draws:
            recorded = replies.get((result.item_id, result.rendering, draw.draw_index))
            stored = result.full_distribution[draw.draw_index]
            if recorded is not None and (recorded.draw != draw or recorded.full_distribution != stored):
                raise ConvergenceError(f"the draw recording of draw {draw.draw_index} of {result.key} differs from the "
                                       "stored result")
            if draw.parsed is None:
                continue  # An unparsed draw has no estimate and no evidence by design.
            reason = None
            if recorded is None or recorded.response is None:
                reason = NO_REPLY
            elif stored is None or stored.evidence_ref is None or stored.evidence_ref not in by_ref:
                reference = None if stored is None else stored.evidence_ref
                reason = f"the full-distribution evidence {reference} is not recorded"
            else:
                values = _draw_values(result, draw, recorded, by_ref[stored.evidence_ref], records, ks)
                if isinstance(values, str):
                    reason = values
                else:
                    compared.append(values)
            if reason is not None:
                missing_draws.append(MappingProxyType({"item_id": result.item_id, "rendering": result.rendering,
                                                       "draw_index": draw.draw_index, "reason": reason}))
    if not compared:
        raise ConvergenceError(f"no recorded draw of {arm}/{stage} carries its reply and full-distribution evidence; "
                               "there is nothing to compare")
    digests = {values.digest for values in compared}
    if len(digests) != 1:
        raise ConvergenceError(f"the evidence comes from more than one set of weights {sorted(digests)}; values compare "
                               "only within one model and one set of weights (Requirement 9.4)")
    (model_digest,) = digests
    if expected_digest is not None and model_digest != expected_digest:
        raise ConvergenceError(f"the evidence comes from the weights {model_digest}, not the arm's pinned weights "
                               f"{expected_digest}; values compare only within one model and one set of weights")
    shapes = {(values.n_vocab, values.kept) for values in compared}
    if len(shapes) != 1:
        raise ConvergenceError(f"the evidence keeps different alternatives or vocabularies {sorted(shapes)}")
    ((n_vocab, kept),) = shapes
    fitted = records[FULL_DISTRIBUTION].evidence_settings.top_k_keep
    if kept != min(fitted, n_vocab):
        raise ConvergenceError(f"the evidence keeps {kept} alternatives; the full-distribution calibrator was fitted "
                               f"on top_k_keep {fitted} of a {n_vocab}-token vocabulary")

    above = {k: f"k {k} is above the {kept} alternatives the evidence keeps" for k in ks if k > kept}
    comparisons = [(HOST_TOP_K, k) for k in ks] + [(SERVED_TOP_K, arm_plan.top_logprobs)]
    features: list[FeatureAgreement] = []
    mass: list[MassCoverage] = []
    for evidence_kind, k in comparisons:
        if evidence_kind == HOST_TOP_K and k in above:
            features += [FeatureAgreement(evidence_kind, k, feature, 0, None, None, None, missing_reason=above[k])
                         for feature in FEATURES]
            mass.append(MassCoverage(evidence_kind, k, 0, None, None, missing_reason=above[k]))
            continue
        if evidence_kind == HOST_TOP_K:
            pairs = [(values.full, values.host[k], None) for values in compared]
            masses, reasons = [values.host_mass[k] for values in compared], [None] * len(compared)
        else:
            pairs = [(values.full, values.served, values.served_reason) for values in compared]
            masses, reasons = [values.served_mass for values in compared], [values.served_reason for values in compared]
        features += [_agreement(evidence_kind, k, feature, pairs) for feature in FEATURES]
        mass.append(_coverage(evidence_kind, k, masses, reasons))

    by_item: dict[tuple[str, str], dict[int, _DrawValues]] = {}
    for values in compared:
        by_item.setdefault(values.key, {})[values.index] = values
    lacking = {(entry["item_id"], entry["rendering"]) for entry in missing_draws}

    changes: dict[tuple[str, int, str], list[ItemDecisionChange]] = {}
    for result in ordered:
        top_k_identity, full_identity = result.identities[TOP_K], result.identities[FULL_DISTRIBUTION]
        full_draws = result.draws_for(FULL_DISTRIBUTION)
        item_values = by_item.get(result.key, {})
        for mode in MODES:
            full = _decide(result, full_draws, full_identity, mode, arm_plan)
            _check_variant(result, mode, FULL_DISTRIBUTION, full)
            served = _decide(result, result.draws_for(TOP_K), top_k_identity, mode, arm_plan)
            _check_variant(result, mode, TOP_K, served)
            failed = NO_DECISION if full.status == "failed" else None
            for evidence_kind, k in comparisons:
                if evidence_kind == HOST_TOP_K and k in above:
                    continue
                if evidence_kind == SERVED_TOP_K:
                    other = served
                elif result.key in lacking:
                    other = None
                else:
                    draws_k = tuple(replace(draw, estimate=item_values[draw.draw_index].host[k]["estimate"])
                                    if draw.parsed is not None else draw for draw in full_draws)
                    other = _decide(result, draws_k, full_identity, mode, arm_plan)
                reason = failed or (DRAWS_LACKING if other is None else None)
                changes.setdefault((evidence_kind, k, mode), []).append(
                    _change(result, evidence_kind, k, mode, full, other, reason))

    decisions: list[DecisionChange] = []
    item_changes: list[ItemDecisionChange] = []
    for evidence_kind, k in comparisons:
        for mode in MODES:
            if evidence_kind == HOST_TOP_K and k in above:
                decisions.append(DecisionChange(evidence_kind, k, mode, 0, None, None, None, None, None, None,
                                                missing_reason=above[k]))
                continue
            rows = changes.get((evidence_kind, k, mode), [])
            decisions.append(_decision_row(evidence_kind, k, mode, rows))
            item_changes += rows

    model = ordered[0].identities[TOP_K].model
    full_masses = [values.full_mass for values in compared]
    return ConvergenceTable(
        plan_sha256=plan.plan_sha256, arm=arm, stage=stage, kind=kind, model=model, model_digest=model_digest,
        n_vocab=n_vocab, kept=kept, ks=ks, served_k=arm_plan.top_logprobs, tolerance=AGREEMENT_TOLERANCE,
        comparability=comparability_statement(model, model_digest), scope=scope_statement(kept),
        estimator=ESTIMATOR_NOTE, items=tuple(result.key for result in ordered), n_draws=len(compared),
        full_mean_mass=_mean([float(np.mean(values)) for values in full_masses]),
        full_min_mass=min(float(np.min(values)) for values in full_masses), features=tuple(features),
        mass=tuple(mass), decisions=tuple(decisions), item_changes=tuple(item_changes),
        missing_items=tuple(missing_items), missing_draws=tuple(missing_draws))


# ---------- from the recorded files ----------

def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        raise ConvergenceError(f"duplicate keys {sorted({key for key in keys if keys.count(key) > 1})}")
    return dict(pairs)


def _no_constant(name: str) -> Any:
    raise ConvergenceError(f"non-finite number {name}")


def _read_json(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConvergenceError(f"{path} cannot be read: {exc}") from None
    try:
        return json.loads(text, object_pairs_hook=_no_duplicate_keys, parse_constant=_no_constant)
    except ConvergenceError as exc:
        raise ConvergenceError(f"{path}: {exc}") from None
    except ValueError as exc:
        raise ConvergenceError(f"{path} is not valid JSON: {exc}") from None


_EVIDENCE_FILE_KEYS = frozenset({"schema", "plan_sha256", "arm", "stage", "item_id", "rendering", "prompt_sha256",
                                 "model_digest", "draw_indices", "evidence"})


def read_evidence_file(path: str | Path, plan: StudyPlan, result: ItemResult) -> tuple[FullDistributionEvidence, ...]:
    """The evidence of one recorded batch of ``result``'s item and rendering (the 7.4 evidence-file format).

    The file must belong to the plan, arm, stage, item, rendering and prompt
    of the result. Each evidence object is still matched to its draw by its
    SHA-256 alone.
    """
    path = Path(path)
    data = _read_json(path)
    if not isinstance(data, Mapping) or not _EVIDENCE_FILE_KEYS <= set(data) or data["schema"] != EVIDENCE_SCHEMA:
        raise ConvergenceError(f"{path} is not a {EVIDENCE_SCHEMA} evidence batch")
    ensure_unchanged(plan, data["plan_sha256"])
    expected = (result.arm, result.stage, result.item_id, result.rendering, result.prompt_sha256)
    if tuple(data[name] for name in ("arm", "stage", "item_id", "rendering", "prompt_sha256")) != expected:
        raise ConvergenceError(f"{path} holds the evidence of another arm, stage, item, rendering or prompt")
    if not isinstance(data["evidence"], list):
        raise ConvergenceError(f"{path}: evidence must be a list")
    try:
        return tuple(FullDistributionEvidence.from_dict(entry) for entry in data["evidence"])
    except (TypeError, ValueError) as exc:
        raise ConvergenceError(f"{path}: the recorded evidence cannot be read ({exc})") from None


def convergence_from_records(
    plan: StudyPlan, *, arm: str = SAMPLING_ARM, stage: str = "full_run", results_dir: str | Path = RESULTS_DIR,
    records_dir: str | Path = RECORDS_DIR, calibrators_dir: str | Path = CALIBRATORS_DIR,
    ks: Sequence[int] = CONVERGENCE_KS,
) -> ConvergenceTable:
    """The convergence table from the private records alone; no client is built and no request is sent.

    Reads the stored result of every planned item and rendering of the stage
    (``execute.result_path``), the stage's draw recording (``draws.jsonl``),
    the evidence batch files beside the request records
    (``recording.evidence_path``) and the arm's calibrators for the stage's
    item kind; the evidence must carry the arm's pinned GGUF digest.

    The draw recording is opened for reading only, through
    :func:`~scripts.finance_sim.recording.read_draw_log`, which never repairs
    a log: an analysis must not be able to modify a private record. A last
    line without its newline is used when it is complete; a log that ends in
    an incomplete line is refused, because the written table has no place to
    state that a line was ignored.
    """
    _check_scope(plan, arm, stage)
    calibrators = load_calibrators(plan, arm=arm, kind=EXECUTION_STAGES[stage], calibrators_dir=calibrators_dir)
    results = []
    for item_id, rendering in planned_keys(plan, arm, stage):
        path = result_path(arm, stage, item_id, rendering, results_dir)
        if path.exists():
            results.append(load_result(path))
    log_path = draw_log_path(arm, stage, records_dir)
    reading = read_draw_log(log_path, plan=plan, loads=ExecutionDraw.from_dict)
    if reading.incomplete_last_lines:
        raise ConvergenceError(f"{log_path} ends in an incomplete last line (a write cut short); the table has no "
                               "place to state an ignored line, so the analysis is refused and the log is left as "
                               "it is")
    draws = reading.draws
    evidence: list[FullDistributionEvidence] = []
    for result in results:
        path = evidence_path(arm, stage, result.item_id, result.rendering, records_dir)
        if path.exists():
            evidence.extend(read_evidence_file(path, plan, result))
    return convergence_table(plan, results, draws, evidence, arm=arm, stage=stage, calibrators=calibrators, ks=ks,
                             expected_digest=pinned_digest(plan, arm))


# ---------- the written table ----------

def table_path(arm: str, stage: str, out_dir: str | Path = CONVERGENCE_DIR) -> Path:
    """``<out_dir>/convergence-<arm>-<stage>.json``; its markdown table sits beside it with ``.md``."""
    if arm not in ARMS or stage not in CONVERGENCE_STAGES:
        raise ConvergenceError(f"unknown arm or stage {arm!r}/{stage!r} for the convergence table")
    return Path(out_dir) / f"convergence-{arm}-{stage}.json"


def _write_text(path: Path, text: str) -> None:
    """Write atomically: a crash leaves the previous file or the new one, never half of one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_table(table: ConvergenceTable, out_dir: str | Path = CONVERGENCE_DIR) -> tuple[Path, Path]:
    """Write the table as JSON and markdown; the same table is a no-op, a different one is refused, never replaced."""
    if not isinstance(table, ConvergenceTable):
        raise TypeError(f"table must be a ConvergenceTable; got {type(table).__name__}")
    json_path = table_path(table.arm, table.stage, out_dir)
    markdown_path = json_path.with_suffix(".md")
    texts = {json_path: json.dumps(table.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
             markdown_path: table.to_markdown()}
    for path, text in texts.items():
        if path.exists() and path.read_text(encoding="utf-8") != text:
            raise ConvergenceError(f"refusing to replace the different convergence table at {path}")
    for path, text in texts.items():
        if not path.exists():
            _write_text(path, text)
    return json_path, markdown_path


def main(argv: Sequence[str] | None = None) -> int:
    """Write the convergence table of one arm and stage from the private records of the frozen plan.

    Exit codes: 0 written (or already written identically), 2 refused (another
    plan, inconsistent records, nothing to compare, or a different table at the
    output path).
    """
    parser = argparse.ArgumentParser(description="Top-k versus full-distribution convergence table (task 7.12), "
                                                 "from recorded results, replies and evidence; sends no request.")
    parser.add_argument("--plan", type=Path, required=True, help="the frozen plan.json")
    parser.add_argument("--plan-sha256", required=True, help="the plan hash the records carry")
    parser.add_argument("--arm", default=SAMPLING_ARM, choices=ARMS)
    parser.add_argument("--stage", default="full_run", choices=CONVERGENCE_STAGES)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--records-dir", type=Path, default=RECORDS_DIR)
    parser.add_argument("--calibrators-dir", type=Path, default=CALIBRATORS_DIR)
    parser.add_argument("--out-dir", type=Path, default=CONVERGENCE_DIR)
    args = parser.parse_args(argv)
    if len(args.plan_sha256) != _SHA256_LENGTH:
        print("convergence: refused: --plan-sha256 must be 64 hex characters", file=sys.stderr)
        return EXIT_REFUSED
    try:
        plan = load_plan(args.plan, expected_sha256=args.plan_sha256)
        table = convergence_from_records(plan, arm=args.arm, stage=args.stage, results_dir=args.results_dir,
                                         records_dir=args.records_dir, calibrators_dir=args.calibrators_dir)
        paths = write_table(table, args.out_dir)
    except (ValueError, TypeError, OSError) as exc:
        print(f"convergence: refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    for path in paths:
        print(path)
    return EXIT_WRITTEN


__all__ = [
    "AGREEMENT_TOLERANCE",
    "CONVERGENCE_DIR",
    "CONVERGENCE_KS",
    "CONVERGENCE_STAGES",
    "ESTIMATOR_NOTE",
    "EVIDENCE_KINDS",
    "FEATURES",
    "HOST_TOP_K",
    "MODES",
    "SERVED_TOP_K",
    "TABLE_SCHEMA",
    "ConvergenceError",
    "ConvergenceTable",
    "DecisionChange",
    "FeatureAgreement",
    "ItemDecisionChange",
    "MassCoverage",
    "comparability_statement",
    "convergence_from_records",
    "convergence_table",
    "main",
    "read_evidence_file",
    "scope_statement",
    "table_path",
    "write_table",
]


if __name__ == "__main__":
    raise SystemExit(main())
