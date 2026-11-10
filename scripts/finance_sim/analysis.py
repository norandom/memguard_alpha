"""Pre-registered analysis: accuracy, losses and multiple testing (task 7.10).

Requirements 6.3, 8.3, 8.4, 8.8, 8.9 and 8.10. No request is sent: the analysis
reads stored item results only.

Inputs. The stored item results of one arm's full run
(:class:`~scripts.finance_sim.execute.ItemResult`, written by the 7.7 execution
and read back with :func:`~scripts.finance_sim.execute.load_result`), passed
through :func:`~scripts.finance_sim.pilot.confirmatory_results`, which keeps the
planned items and renderings of that arm and stage and excludes every pilot item
with the reason (8.9); a result of another plan, arm or stage, or given twice,
is refused there. Every size, threshold and name comes from the frozen plan: the
metric names and their order (``plan.analysis.metrics``), the variants
(``plan.arms.variants``) and the ones the arm cannot compute with the plan's
reason (``ArmPlan.missing_variants``: the serving arm's full-distribution
variants), the model's cutoff class per month, the block length, resample count,
significance level and seed, the pre-registered effect, and the losses of the
scored probability. The serving (primary) arm analyses every evaluation month
with its none and top-k variants; the sampling (probe) arm its frozen probe
evaluation months with all six variants.

Rows (Requirement 6.3; design "Analysis inputs"). One row per planned evaluation
month of the arm, in plan order, holding both renderings (identifying and
anonymised) of one variant. A month enters a statistic of that variant only when
both renderings carry its value (``correct`` as 1 or 0, ``p_memorized``, or the
scored probabilities of the variant and of its reference with the outcome); any
other month is listed with the reason per rendering and never substituted
(8.10). A month is pre-cutoff when the plan lists it on or before the sourced
cutoff of the arm's model (``plan.months.pre_cutoff``).

Statistics per variant:

- ``estimate_difference_pre_cutoff`` and ``estimate_difference_post_cutoff``
  (6.3): the mean over that period's rows of identifying minus anonymised
  ``p_memorized``. The no-adjustment variants carry no estimate by construction
  and are reported as missing with :data:`NO_ESTIMATE`.
- ``accuracy_difference_pre_cutoff`` and ``accuracy_difference_post_cutoff``
  (6.3): the same for ``correct``; on the same rows they are the two cells of the
  DiD gap.
- ``did_accuracy_gap`` (8.3):
  :func:`~recall_guard.core.significance.did_gap` over the rows' items (both
  renderings of every row), equal to the pre-cutoff minus the post-cutoff
  accuracy difference.
- ``brier_difference`` and ``log_loss_difference`` (8.4):
  :func:`~recall_guard.core.significance.paired_loss_difference` of the
  variant's loss against the loss of the no-adjustment variant of the same draw
  mode, rows = months and columns = the two renderings, the loss of the plan's
  adjusted scored probability (``adjusted_scored_probability``) against the
  outcome as the plan's ``scored_probability.losses`` define it. The
  no-adjustment variants are the references and are reported as missing with
  :data:`REFERENCE`.
- ``achieved_power`` (8.9): :func:`~scripts.finance_sim.pilot.achieved_power`
  for the arm, labelled :data:`POWER_APPROXIMATION`.

Intervals. A non-circular moving-block percentile bootstrap over the rows
through :func:`~recall_guard.core.significance.block_bootstrap` (and inside
``paired_loss_difference``) with the plan's ``block_length``, ``n_resamples``,
``seed`` and confidence ``1 - alpha``; the seed is the same for every metric and
variant, so variants with the same rows are resampled alike. A resample whose
period cell is empty (no pre-cutoff or no post-cutoff month among its rows) is
dropped and counted in ``dropped_resamples``, so such an interval is conditional
on non-empty cells and the count shows how often. Each cell is checked up front:
a statistic whose rows are empty, whose needed period has no month, or that has
fewer months than the block length is missing with the reason.

p-values and Holm (8.8). The tests are ``did_accuracy_gap``,
``brier_difference`` and ``log_loss_difference``; the four differences before
and after the cutoff are measurements (6.3) with intervals and no p-value. The
losses take the two-sided p-value of ``paired_loss_difference``. The DiD gap
takes the same two-sided rule, ``min(1, 2 min(1 + #boot <= 0, 1 + #boot >= 0) /
(1 + n_kept))``, over the kept resamples of its own ``block_bootstrap``. Holm
(:func:`~recall_guard.core.significance.holm`) adjusts each tested metric across
the variants that carry a p-value; a missing variant is not a member.

Output. :func:`write_analysis` writes
``runs/finance-sim/analysis/pre-registered-<arm>-<stage>.json`` and a markdown
table of derived statistics beside it (no reply text, no realized returns, no
raw licensed series), once: the same output again is a no-op and a different
one is refused, never replaced. :func:`analysis_from_records` recomputes the
analysis from the stored results alone, so a replay with the network disabled
reproduces the stored output.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from recall_guard.core.significance import Interval, block_bootstrap, did_gap, holm, paired_loss_difference
from scripts.finance_sim.calibrate import ANONYMISED, IDENTIFYING, macro_item_id
from scripts.finance_sim.execute import RESULTS_DIR, STATUSES, ItemResult, VariantResult, load_result, planned_keys
from scripts.finance_sim.execute import result_path as stored_result_path
from scripts.finance_sim.limits import ARMS, RUNS_ROOT
from scripts.finance_sim.pilot import CONFIRMATORY_STAGE, PILOT_STAGE, achieved_power, confirmatory_results
from scripts.finance_sim.plan import NO_ADJUSTMENT, SCORED_LOSSES, ArmPlan, StudyPlan, load_plan, require_complete
from scripts.finance_sim.plan import variant_parts as plan_variant_parts

ANALYSIS_SCHEMA = "finance-sim-pre-registered-analysis/1"
#: Analysis outputs (design: ``runs/finance-sim/analysis/*.json``).
ANALYSIS_DIR = RUNS_ROOT / "analysis"
EXIT_WRITTEN, EXIT_REFUSED = 0, 2
#: The rendering pair of a row; every difference is identifying minus anonymised.
PAIR: tuple[str, str] = (IDENTIFYING, ANONYMISED)

ESTIMATE_PRE, ESTIMATE_POST = "estimate_difference_pre_cutoff", "estimate_difference_post_cutoff"
ACCURACY_PRE, ACCURACY_POST = "accuracy_difference_pre_cutoff", "accuracy_difference_post_cutoff"
DID = "did_accuracy_gap"
BRIER, LOG_LOSS = "brier_difference", "log_loss_difference"
ACHIEVED_POWER = "achieved_power"
#: The frozen metric names this analysis computes, in the order the plan freezes them (``plan.METRICS``).
ANALYSIS_METRICS: tuple[str, ...] = (ESTIMATE_PRE, ESTIMATE_POST, ACCURACY_PRE, ACCURACY_POST, DID, BRIER, LOG_LOSS,
                                     ACHIEVED_POWER)
#: The frozen metric names the ensemble-metrics analysis (task 7.11) computes from the same results.
ENSEMBLE_METRICS: tuple[str, ...] = ("half_ensemble_variance_summary", "half_ensemble_variance_p_memorized",
                                     "first_draw_rate", "random_answerer_null")
#: The metrics that are tests, each a Holm family across the variants (Requirement 8.8).
TESTED_METRICS: tuple[str, ...] = (DID, BRIER, LOG_LOSS)
COMPUTED, DEFERRED, MISSING = "computed", "deferred", "missing"

NO_RESULT = "no stored result: the stage holds no result for this item and rendering, so the run is incomplete"
NO_ESTIMATE = ("not applicable: the no-adjustment variant carries no memorization estimate (its draws pass without "
               "estimates by construction, status estimate_unavailable), so it has no estimate difference")
REFERENCE = ("not applicable: the no-adjustment variant is the reference the adjusted variants of its draw mode are "
             "compared against, so it has no difference against itself")
DEFERRED_REASON = ("an ensemble metric (Requirements 8.5 to 8.7) that the ensemble-metrics analysis (task 7.11) "
                   "computes from the same results; not part of this analysis")
UNKNOWN_METRIC = ("no pre-registered analysis implements this frozen metric name, so it is reported as missing with "
                  "this reason and nothing is substituted")
POWER_APPROXIMATION = (
    "a t-test approximation under independent months: the one-sided pooled two-sample t test of "
    "simulated_power_two_sample at the pre-registered effect, with the realised pre-cutoff and post-cutoff month "
    "counts and the pooled within-period standard deviation of the noise variant's identifying minus anonymised "
    "correctness; the confirmatory test of the DiD gap is the two-sided moving-block bootstrap, which allows "
    "dependence between neighbouring months, so this power approximates the power of that test"
)
_INTERVAL = ("interval: non-circular moving-block percentile bootstrap over the rows in plan order "
             "(recall_guard.core.significance.block_bootstrap) with the plan's block_length, n_resamples, seed and "
             "confidence 1 - alpha; a resample whose needed period has no month is dropped and counted")
_ROWS = ("rows: the arm's planned evaluation months whose identifying and anonymised renderings both carry the "
         "value, pilot items excluded")
ESTIMATE_METHOD = ("identifying minus anonymised p_memorized, mean over the rows of the period; " + _ROWS + "; "
                   + _INTERVAL + "; a measurement (Requirement 6.3), not a test, so no p-value")
ACCURACY_METHOD = ("identifying minus anonymised correctness (1 correct, 0 not), mean over the rows of the period; "
                   + _ROWS + "; " + _INTERVAL + "; a measurement (Requirement 6.3), not a test, so no p-value")
DID_METHOD = ("pre-cutoff minus post-cutoff accuracy difference through "
              "recall_guard.core.significance.did_gap(correct, identifying, pre_cutoff) over both renderings of every "
              "row; " + _ROWS + "; " + _INTERVAL
              + "; two-sided p = min(1, 2 min(1 + #boot <= 0, 1 + #boot >= 0) / (1 + n_kept)) over the kept "
              "resamples (the rule of paired_loss_difference), Holm-adjusted across the variants")
LOSS_METHOD = ("mean loss difference against the no-adjustment variant of the same draw mode through "
               "recall_guard.core.significance.paired_loss_difference(loss of the variant, loss of the reference), "
               "the loss of the plan's adjusted scored probability against the outcome as the plan's "
               "scored_probability.losses define it; rows: the arm's planned evaluation months whose two renderings "
               "both carry the scored probabilities of the variant and its reference and an outcome, one column per "
               "rendering, pilot items excluded; moving-block percentile bootstrap with the plan's block_length, "
               "n_resamples, seed and confidence 1 - alpha; its two-sided p-value, Holm-adjusted across the variants")
HOLM_METHOD = ("Holm step-down adjustment (recall_guard.core.significance.holm) across the variants that carry a "
               "p-value in each tested metric (did_accuracy_gap, brier_difference, log_loss_difference); a missing "
               "variant is not a member")
CONDITIONAL_NOTE = ("A variant with dropped resamples had resamples whose needed period cell was empty (no pre-cutoff "
                    "or no post-cutoff month among the resampled months); its interval and p-value are conditional "
                    "on the kept resamples.")
_REQUIREMENTS: Mapping[str, str] = MappingProxyType({
    ESTIMATE_PRE: "6.3", ESTIMATE_POST: "6.3", ACCURACY_PRE: "6.3", ACCURACY_POST: "6.3", DID: "8.3, 8.8",
    BRIER: "8.4, 8.8", LOG_LOSS: "8.4, 8.8", ACHIEVED_POWER: "8.9"})


class AnalysisError(ValueError):
    """Results, a plan or an output the pre-registered analysis refuses; nothing is written for them."""


# ---------- losses of the scored probability ----------

def _brier(p: float, outcome: int) -> float:
    return (p - outcome) ** 2


def _log_loss(p: float, outcome: int) -> float:
    return -(outcome * math.log(p) + (1 - outcome) * math.log(1.0 - p))


#: The loss of each loss metric, as ``plan.SCORED_LOSSES`` freezes it (``brier`` and ``log_loss``).
_LOSSES: Mapping[str, Callable[[float, int], float]] = MappingProxyType({BRIER: _brier, LOG_LOSS: _log_loss})


# ---------- settings and output records ----------

@dataclass(frozen=True)
class _Settings:
    """What the plan freezes for the bootstraps and the power: block length, resamples, seed, alpha, effect."""

    block_length: int
    n_resamples: int
    seed: int
    alpha: float
    effect_size: float

    @property
    def confidence(self) -> float:
        return 1.0 - self.alpha

    @classmethod
    def of(cls, plan: StudyPlan) -> _Settings:
        analysis = plan.analysis
        return cls(block_length=analysis.block_length, n_resamples=analysis.n_resamples, seed=plan.seed,
                   alpha=analysis.alpha, effect_size=analysis.effect_size)

    def to_dict(self) -> dict[str, Any]:
        return {"block_length": self.block_length, "n_resamples": self.n_resamples, "seed": self.seed,
                "confidence": self.confidence, "alpha": self.alpha, "effect_size": self.effect_size}


def _interval_dict(interval: Interval | None) -> dict[str, Any] | None:
    if interval is None:
        return None
    return {"lo": interval.lo, "hi": interval.hi, "confidence": interval.confidence, "method": interval.method}


@dataclass(frozen=True)
class VariantCell:
    """One variant's value of one metric, or the reason it has none (``missing``).

    ``statistic`` is the effect size on its own scale, ``interval`` its
    bootstrap interval, ``p_value`` the raw two-sided p-value of a tested
    metric and ``p_holm`` its Holm-adjusted value. ``months`` counts the rows
    used (``pre_cutoff_months`` and ``post_cutoff_months`` by period);
    ``missing_months`` lists every other planned month with the reason.
    ``statuses`` counts the variant's guarded statuses over the analysed items,
    ``sign_from_raw_spread`` (accuracy metrics) the items scored on the raw
    spread's sign, and ``reference`` names the no-adjustment variant a loss is
    compared against.
    """

    variant: str
    mode: str
    adjustment: str
    reference: str | None = None
    missing: str | None = None
    statistic: float | None = None
    interval: Interval | None = None
    p_value: float | None = None
    p_holm: float | None = None
    months: int = 0
    pre_cutoff_months: int = 0
    post_cutoff_months: int = 0
    dropped_resamples: int | None = None
    statuses: Mapping[str, int] = field(default_factory=lambda: MappingProxyType({}))
    sign_from_raw_spread: int | None = None
    missing_months: tuple[Mapping[str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"variant": self.variant, "mode": self.mode, "adjustment": self.adjustment,
                "reference": self.reference, "missing": self.missing, "statistic": self.statistic,
                "interval": _interval_dict(self.interval), "p_value": self.p_value, "p_holm": self.p_holm,
                "months": self.months, "pre_cutoff_months": self.pre_cutoff_months,
                "post_cutoff_months": self.post_cutoff_months, "dropped_resamples": self.dropped_resamples,
                "statuses": dict(self.statuses), "sign_from_raw_spread": self.sign_from_raw_spread,
                "missing_months": [dict(entry) for entry in self.missing_months]}


@dataclass(frozen=True)
class MetricEntry:
    """One frozen metric: computed per variant, the achieved power, deferred to task 7.11, or missing."""

    metric: str
    status: str
    reason: str | None = None
    requirement: str | None = None
    method: str | None = None
    tested: bool = False
    holm_members: tuple[str, ...] = ()
    variants: tuple[VariantCell, ...] = ()
    power: Mapping[str, Any] | None = None

    def cell(self, variant: str) -> VariantCell:
        for cell in self.variants:
            if cell.variant == variant:
                return cell
        raise KeyError(variant)

    def to_dict(self) -> dict[str, Any]:
        return {"metric": self.metric, "status": self.status, "reason": self.reason,
                "requirement": self.requirement, "method": self.method, "tested": self.tested,
                "holm_members": list(self.holm_members), "variants": [cell.to_dict() for cell in self.variants],
                "power": None if self.power is None else _plain(self.power)}


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True)
class PreRegisteredAnalysis:
    """The pre-registered accuracy, loss and multiple-testing analysis of one arm and stage (derived values only)."""

    plan_sha256: str
    arm: str
    stage: str
    role: str
    supplies: str
    model: str
    settings: Mapping[str, Any]
    planned_months: int
    pre_cutoff_months: int
    post_cutoff_months: int
    variants: tuple[str, ...]
    missing_variants: Mapping[str, str]
    losses: Mapping[str, str]
    excluded: tuple[Mapping[str, str], ...]
    missing_results: tuple[Mapping[str, str], ...]
    metrics: tuple[MetricEntry, ...]
    holm_method: str = HOLM_METHOD

    def metric(self, name: str) -> MetricEntry:
        for entry in self.metrics:
            if entry.metric == name:
                return entry
        raise KeyError(name)

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible, with lists only, so it reads back from JSON unchanged."""
        return {"schema": ANALYSIS_SCHEMA, "plan_sha256": self.plan_sha256, "arm": self.arm, "stage": self.stage,
                "role": self.role, "supplies": self.supplies, "model": self.model, "settings": dict(self.settings),
                "planned_months": self.planned_months, "pre_cutoff_months": self.pre_cutoff_months,
                "post_cutoff_months": self.post_cutoff_months, "variants": list(self.variants),
                "missing_variants": dict(self.missing_variants), "losses": dict(self.losses),
                "excluded": [dict(entry) for entry in self.excluded],
                "missing_results": [dict(entry) for entry in self.missing_results],
                "holm_method": self.holm_method, "metrics": [entry.to_dict() for entry in self.metrics]}

    def to_markdown(self) -> str:
        """The derived statistics as markdown tables: one per metric, the power, the deferred and missing items."""
        settings = self.settings
        lines = [
            "# Pre-registered analysis: accuracy, losses and multiple testing", "",
            f"Arm {self.arm} ({self.role} arm), stage {self.stage}, model {self.model}, plan {self.plan_sha256}.", "",
            f"This arm supplies {self.supplies}.", "",
            f"Planned months: {self.planned_months} ({self.pre_cutoff_months} on or before the model's cutoff, "
            f"{self.post_cutoff_months} after); pilot items left out: {len(self.excluded)}; planned results missing: "
            f"{len(self.missing_results)}.", "",
            f"Intervals: {_percent(settings['confidence'])} percent non-circular moving-block percentile bootstrap "
            f"over months, block length {settings['block_length']} months, {settings['n_resamples']} resamples, seed "
            f"{settings['seed']}; tested metrics carry two-sided p-values, Holm-adjusted across their variants "
            f"(alpha {settings['alpha']:g}); pre-registered effect {settings['effect_size']:g}.", "",
            "Derived statistics only: no reply text, no realized returns and no raw licensed series.", "",
        ]
        for entry in self.metrics:
            lines += [f"## {entry.metric}", ""]
            if entry.power is not None:
                lines += _power_lines(entry, entry.power)
            elif entry.status != COMPUTED:
                lines += [f"{entry.status.capitalize()}: {entry.reason}", ""]
            else:
                lines += _metric_lines(entry)
        if self.missing_results or self.excluded:
            lines += ["## Missing and excluded items", ""]
            lines += [f"- {entry['item_id']} {entry['rendering']}: {entry['reason']}" for entry in self.missing_results]
            lines += [f"- {entry['item_id']} {entry['rendering']}: {entry['reason']}" for entry in self.excluded]
            lines.append("")
        return "\n".join(lines)


def _percent(confidence: float) -> str:
    return f"{confidence * 100:g}"


def _number(value: float | None) -> str:
    return "missing" if value is None else f"{value:.6g}"


def _interval_text(interval: Interval | None) -> str:
    return "missing" if interval is None else f"[{interval.lo:.6g}, {interval.hi:.6g}]"


def _missing_text(cell: VariantCell) -> str:
    if cell.missing is not None:
        return cell.missing
    count = len(cell.missing_months)
    if count:
        noun = "month" if count == 1 else "months"
        return f"{count} planned {noun} without the value in both renderings (reasons in the JSON)"
    return "none"


def _sentence(text: str) -> str:
    return text[:1].upper() + text[1:]


def _requirements(entry: MetricEntry) -> str:
    requirement = entry.requirement or ""
    return f"Requirements {requirement}" if "," in requirement else f"Requirement {requirement}"


def _metric_lines(entry: MetricEntry) -> list[str]:
    lines = [f"{_requirements(entry)}. {_sentence(entry.method or '')}.", ""]
    if entry.tested:
        lines += ["| Variant | Statistic | Interval | p raw | p Holm | Months (pre, post) | Dropped resamples | "
                  "Missing |", "|---|---|---|---|---|---|---|---|"]
    else:
        lines += ["| Variant | Statistic | Interval | Months (pre, post) | Dropped resamples | Missing |",
                  "|---|---|---|---|---|---|"]
    for cell in entry.variants:
        dropped = "" if cell.dropped_resamples is None else str(cell.dropped_resamples)
        months = f"{cell.months} ({cell.pre_cutoff_months}, {cell.post_cutoff_months})"
        tests = f" {_number(cell.p_value)} | {_number(cell.p_holm)} |" if entry.tested else ""
        lines.append(f"| {cell.variant} | {_number(cell.statistic)} | {_interval_text(cell.interval)} |{tests} "
                     f"{months} | {dropped} | {_missing_text(cell)} |")
    if entry.tested:
        members = ", ".join(entry.holm_members) or "none"
        lines += ["", f"Holm family: {members}."]
    if any(cell.dropped_resamples for cell in entry.variants):
        lines += ["", CONDITIONAL_NOTE]
    lines.append("")
    return lines


def _power_lines(entry: MetricEntry, power: Mapping[str, Any]) -> list[str]:
    if power["power"] is None:
        head = f"Missing: {entry.reason}"
    else:
        head = (f"Achieved power {power['power']:.6g} for the pre-registered effect {power['effect_size']:g} "
                f"(alpha {power['alpha']:g}, {power['pre_cutoff_months']} pre-cutoff and "
                f"{power['post_cutoff_months']} post-cutoff months, noise_sd {_number(power['noise_sd'])}, "
                f"{power['n_sim']} simulations, seed {power['seed']}, noise variant {power['noise_variant']})")
    return [f"{_requirements(entry)}. {head}.", "", f"Read as {power['approximation']}.", "",
            f"Months without the noise statistic: {len(power['missing_months'])}.", ""]


# ---------- checks ----------

def _check_scope(plan: StudyPlan, arm: str, stage: str) -> ArmPlan:
    require_complete(plan)
    if arm not in ARMS:
        raise AnalysisError(f"arm must be one of {list(ARMS)}; got {arm!r}")
    if stage == PILOT_STAGE:
        raise AnalysisError("pilot items size the power gate and enter no later analysis (Requirement 8.9); the "
                            f"pre-registered analysis reads the {CONFIRMATORY_STAGE} stage")
    if stage != CONFIRMATORY_STAGE:
        raise AnalysisError(f"the pre-registered accuracy and loss analysis reads the macro items of the "
                            f"{CONFIRMATORY_STAGE} stage; got {stage!r} (the negative_control stage holds EDGAR "
                            "outlooks without realized outcomes, reported by task 7.9)")
    losses = dict(plan.scored_probability.losses)
    if losses != dict(SCORED_LOSSES):
        raise AnalysisError(f"the plan freezes the scored-probability losses {losses}; this analysis implements "
                            f"{dict(SCORED_LOSSES)} only, so the frozen losses cannot be computed as registered")
    return plan.arms.arms[arm]


def _check_results(plan: StudyPlan, arm: str, arm_plan: ArmPlan, results: Iterable[ItemResult]) -> None:
    for result in results:
        if (result.ensemble_draws, result.min_parsed) != (arm_plan.draws, arm_plan.min_parsed):
            raise AnalysisError(f"the result of {result.key} has an ensemble of {result.ensemble_draws} draws (min "
                                f"{result.min_parsed}); the frozen {arm} arm draws {arm_plan.draws} (min "
                                f"{arm_plan.min_parsed})")
        for name in plan.arms.variants:
            stored, expected = result.variant(name).missing, arm_plan.missing_variants.get(name)
            if stored != expected:
                raise AnalysisError(f"the {name} variant of {result.key} is missing with {stored!r}, but the frozen "
                                    f"{arm} arm gives {expected!r}; the result was not derived under this plan's arm")


# ---------- values per item ----------

@dataclass(frozen=True)
class _Month:
    item_id: str
    pre_cutoff: bool


#: A value of one item (or None) and the reason it has none.
_Value = tuple[Any, str | None]


def _reason(variant: VariantResult) -> str:
    return variant.reason or variant.status or "no reason recorded"


def _correct(name: str) -> Callable[[ItemResult], _Value]:
    def value(result: ItemResult) -> _Value:
        variant = result.variant(name)
        if variant.correct is None:
            return None, variant.missing_reasons.get("correct", f"no correctness ({variant.status})")
        return float(variant.correct), None

    return value


def _estimate(name: str) -> Callable[[ItemResult], _Value]:
    def value(result: ItemResult) -> _Value:
        variant = result.variant(name)
        if variant.p_memorized is None:
            if variant.status == "failed":
                return None, f"no decision: {_reason(variant)}"
            return None, f"no memorization estimate (status {variant.status}): {_reason(variant)}"
        return variant.p_memorized, None

    return value


def _scored(name: str, reference: str) -> Callable[[ItemResult], _Value]:
    def value(result: ItemResult) -> _Value:
        chosen, base = result.variant(name), result.variant(reference)
        if (chosen.scored_probability is not None and base.scored_probability is not None
                and chosen.scored_probability != base.scored_probability):
            raise AnalysisError(f"the {name} and {reference} results of {result.key} hold different raw scored "
                                "probabilities; the variants are not paired on the same draws")
        if chosen.adjusted_scored_probability is None or base.adjusted_scored_probability is None:
            reason = (chosen.missing_reasons.get("scored_probability")
                      or base.missing_reasons.get("scored_probability") or "no scored probability")
            return None, reason
        if result.outcome is None:
            return None, "no outcome: no realized next-month returns for this month"
        return (chosen.adjusted_scored_probability, base.adjusted_scored_probability, result.outcome), None

    return value


def _rows(months: Sequence[_Month], by_key: Mapping[tuple[str, str], ItemResult],
          value: Callable[[ItemResult], _Value]) -> tuple[list[tuple[_Month, Any, Any]], tuple[Mapping[str, str], ...]]:
    """The months whose two renderings both carry a value, and every other month with the reason per rendering."""
    rows: list[tuple[_Month, Any, Any]] = []
    missing: list[Mapping[str, str]] = []
    for month in months:
        found: dict[str, Any] = {}
        reasons: list[str] = []
        for rendering in PAIR:
            result = by_key.get((month.item_id, rendering))
            if result is None:
                reasons.append(f"{rendering}: {NO_RESULT}")
                continue
            got, reason = value(result)
            if got is None:
                reasons.append(f"{rendering}: {reason}")
            else:
                found[rendering] = got
        if reasons:
            missing.append(MappingProxyType({"item_id": month.item_id, "reason": "; ".join(reasons)}))
        else:
            rows.append((month, found[IDENTIFYING], found[ANONYMISED]))
    return rows, tuple(missing)


# ---------- statistics on the rows ----------

def _period_difference(pre: bool) -> Callable[[np.ndarray], float]:
    """Mean identifying minus anonymised value over the rows of one period (column 2 is 1 before the cutoff)."""
    flag, label = (1.0, "pre-cutoff") if pre else (0.0, "post-cutoff")

    def statistic(rows: np.ndarray) -> float:
        chosen = rows[rows[:, 2] == flag]
        if chosen.shape[0] == 0:
            raise ValueError(f"no {label} month among the rows")
        return float(np.mean(chosen[:, 0] - chosen[:, 1]))

    return statistic


def _did(rows: np.ndarray) -> float:
    """``did_gap`` over both renderings of every row: columns identifying, anonymised, pre-cutoff flag."""
    n = rows.shape[0]
    return did_gap(np.concatenate([rows[:, 0], rows[:, 1]]), np.concatenate([np.ones(n, bool), np.zeros(n, bool)]),
                   np.concatenate([rows[:, 2] == 1.0, rows[:, 2] == 1.0]))


def _two_sided_p(kept: np.ndarray) -> float:
    """``min(1, 2 min(1 + #boot <= 0, 1 + #boot >= 0) / (1 + n_kept))``, the rule of ``paired_loss_difference``."""
    at_or_below = 1 + int(np.count_nonzero(kept <= 0.0))
    at_or_above = 1 + int(np.count_nonzero(kept >= 0.0))
    return min(1.0, 2.0 * min(at_or_below, at_or_above) / (1 + kept.size))


def _bootstrap(statistic: Callable[[np.ndarray], float], rows: np.ndarray,
               settings: _Settings) -> tuple[Interval, np.ndarray]:
    """``block_bootstrap`` of ``statistic`` with the plan's settings, and the statistics of its kept resamples."""
    kept: list[float] = []

    def recorded(sample: np.ndarray) -> float:
        value = statistic(sample)
        kept.append(value)
        return value

    interval = block_bootstrap(recorded, rows, block_length=settings.block_length, n_resamples=settings.n_resamples,
                               seed=settings.seed, confidence=settings.confidence)
    return interval, np.asarray(kept, dtype=float)


def _rows_missing(n: int, n_pre: int, n_post: int, *, pre: bool, post: bool, settings: _Settings) -> str | None:
    """The reason a statistic on these rows cannot be computed, checked before any resample, or None."""
    if n == 0:
        return "no planned month carries the value in both renderings"
    for needed, count, label in ((pre, n_pre, "pre-cutoff"), (post, n_post, "post-cutoff")):
        if needed and count == 0:
            return f"no {label} month carries the value in both renderings, so the {label} cell is empty"
    if n < settings.block_length:
        return (f"{n} months carry the value in both renderings, fewer than the frozen block length of "
                f"{settings.block_length} months, so the moving-block bootstrap is not defined")
    return None


def _counts(rows: Sequence[tuple[_Month, Any, Any]]) -> dict[str, int]:
    pre = sum(1 for month, _, _ in rows if month.pre_cutoff)
    return {"months": len(rows), "pre_cutoff_months": pre, "post_cutoff_months": len(rows) - pre}


def _row_cell(base: Mapping[str, Any], rows: Sequence[tuple[_Month, Any, Any]], missing: tuple[Mapping[str, str], ...],
              statistic: Callable[[np.ndarray], float], settings: _Settings, *, pre: bool, post: bool,
              tested: bool) -> VariantCell:
    counts = _counts(rows)
    common = {**base, **counts, "missing_months": missing}
    reason = _rows_missing(counts["months"], counts["pre_cutoff_months"], counts["post_cutoff_months"], pre=pre,
                           post=post, settings=settings)
    if reason is not None:
        return VariantCell(**common, missing=reason)
    array = np.asarray([[a, b, 1.0 if month.pre_cutoff else 0.0] for month, a, b in rows], dtype=float)
    try:
        interval, kept = _bootstrap(statistic, array, settings)
    except ValueError as exc:
        return VariantCell(**common, missing=f"the interval cannot be computed: {exc}")
    return VariantCell(**common, statistic=statistic(array), interval=interval,
                       p_value=_two_sided_p(kept) if tested else None,
                       dropped_resamples=settings.n_resamples - int(kept.size))


def _loss_cell(base: Mapping[str, Any], rows: Sequence[tuple[_Month, Any, Any]],
               missing: tuple[Mapping[str, str], ...], loss: Callable[[float, int], float],
               settings: _Settings) -> VariantCell:
    counts = _counts(rows)
    common = {**base, **counts, "missing_months": missing}
    reason = _rows_missing(counts["months"], counts["pre_cutoff_months"], counts["post_cutoff_months"], pre=False,
                           post=False, settings=settings)
    if reason is not None:
        return VariantCell(**common, missing=reason)
    chosen = np.asarray([[loss(ids[0], ids[2]), loss(anons[0], anons[2])] for _, ids, anons in rows])
    reference = np.asarray([[loss(ids[1], ids[2]), loss(anons[1], anons[2])] for _, ids, anons in rows])
    try:
        test = paired_loss_difference(chosen, reference, block_length=settings.block_length,
                                      n_resamples=settings.n_resamples, seed=settings.seed,
                                      confidence=settings.confidence)
    except ValueError as exc:
        return VariantCell(**common, missing=f"the interval cannot be computed: {exc}")
    return VariantCell(**common, statistic=test.statistic, interval=test.interval, p_value=test.p_value,
                       dropped_resamples=int(test.parameters["dropped_resamples"]))


# ---------- the metrics ----------

@dataclass(frozen=True)
class _Context:
    plan: StudyPlan
    arm_plan: ArmPlan
    months: tuple[_Month, ...]
    by_key: Mapping[tuple[str, str], ItemResult]
    settings: _Settings

    def variant_results(self, name: str) -> list[VariantResult]:
        return [result.variant(name) for result in self.by_key.values()]

    def statuses(self, name: str) -> Mapping[str, int]:
        counts = Counter(variant.status for variant in self.variant_results(name) if variant.status is not None)
        return MappingProxyType({status: counts[status] for status in STATUSES if counts[status]})


def _base(name: str, reference: str | None = None) -> dict[str, Any]:
    mode, adjustment = plan_variant_parts(name)
    return {"variant": name, "mode": mode, "adjustment": adjustment, "reference": reference}


def _period_metric(context: _Context, metric: str, *, estimate: bool, pre: bool | None) -> MetricEntry:
    """An estimate or accuracy difference before (``pre`` True) or after the cutoff, or the DiD gap (None)."""
    cells = []
    for name in context.plan.arms.variants:
        base = _base(name)
        plan_missing = context.arm_plan.missing_variants.get(name)
        if plan_missing is not None:
            cells.append(VariantCell(**base, missing=plan_missing))
            continue
        if estimate and plan_variant_parts(name)[1] == NO_ADJUSTMENT:
            cells.append(VariantCell(**base, missing=NO_ESTIMATE, statuses=context.statuses(name)))
            continue
        rows, missing = _rows(context.months, context.by_key, _estimate(name) if estimate else _correct(name))
        statistic = _did if pre is None else _period_difference(pre)
        cell = _row_cell(base, rows, missing, statistic, context.settings, pre=pre is not False,
                         post=pre is not True, tested=pre is None)
        extra: dict[str, Any] = {"statuses": context.statuses(name)}
        if not estimate:
            extra["sign_from_raw_spread"] = sum(variant.sign_from_raw_spread
                                                for variant in context.variant_results(name))
        cells.append(replace(cell, **extra))
    method = ESTIMATE_METHOD if estimate else (DID_METHOD if pre is None else ACCURACY_METHOD)
    return MetricEntry(metric=metric, status=COMPUTED, requirement=_REQUIREMENTS[metric], method=method,
                       tested=pre is None, variants=tuple(cells))


def _loss_metric(context: _Context, metric: str) -> MetricEntry:
    loss = _LOSSES[metric]
    cells = []
    for name in context.plan.arms.variants:
        mode, adjustment = plan_variant_parts(name)
        reference = f"{mode}_{NO_ADJUSTMENT}"
        plan_missing = context.arm_plan.missing_variants.get(name)
        if plan_missing is not None:
            cells.append(VariantCell(**_base(name, reference), missing=plan_missing))
            continue
        if adjustment == NO_ADJUSTMENT:
            cells.append(VariantCell(**_base(name), missing=REFERENCE, statuses=context.statuses(name)))
            continue
        rows, missing = _rows(context.months, context.by_key, _scored(name, reference))
        cell = _loss_cell(_base(name, reference), rows, missing, loss, context.settings)
        cells.append(replace(cell, statuses=context.statuses(name)))
    return MetricEntry(metric=metric, status=COMPUTED, requirement=_REQUIREMENTS[metric], method=LOSS_METHOD,
                       tested=True, variants=tuple(cells))


def _power_metric(plan: StudyPlan, results: Sequence[ItemResult], arm: str) -> MetricEntry:
    achieved = achieved_power(plan, results, arm=arm)
    power = MappingProxyType({
        "power": achieved.power, "reason": achieved.reason, "approximation": POWER_APPROXIMATION,
        "noise_variant": achieved.noise_variant, "effect_size": achieved.effect_size, "alpha": achieved.alpha,
        "n_sim": achieved.n_sim, "seed": achieved.seed, "pre_cutoff_months": achieved.pre_cutoff_months,
        "post_cutoff_months": achieved.post_cutoff_months, "noise_sd": achieved.noise_sd, "method": achieved.method,
        "missing_months": tuple(MappingProxyType({"item_id": month.item_id, "reason": month.missing_reason})
                                for month in achieved.months if month.difference is None),
    })
    status = COMPUTED if achieved.power is not None else MISSING
    return MetricEntry(metric=ACHIEVED_POWER, status=status, reason=achieved.reason,
                       requirement=_REQUIREMENTS[ACHIEVED_POWER], method=achieved.method, power=power)


def _with_holm(entry: MetricEntry) -> MetricEntry:
    """Holm across the variants of a tested metric that carry a p-value (Requirement 8.8)."""
    raw = {cell.variant: cell.p_value for cell in entry.variants if cell.p_value is not None}
    adjusted = holm(raw)
    cells = tuple(replace(cell, p_holm=adjusted.get(cell.variant)) for cell in entry.variants)
    return replace(entry, variants=cells, holm_members=tuple(raw))


def pre_registered_analysis(plan: StudyPlan, results: Iterable[ItemResult] | Mapping[Any, ItemResult], *, arm: str,
                            stage: str = CONFIRMATORY_STAGE) -> PreRegisteredAnalysis:
    """The pre-registered analysis of ``arm``'s stored ``stage`` results (Requirements 6.3, 8.3, 8.4, 8.8 to 8.10).

    ``results`` are the stored item results (an execution run's ``results`` or
    results read back with :func:`~scripts.finance_sim.execute.load_result`);
    pilot items among them are excluded with the reason. No request is sent.
    Every size, threshold and name comes from the frozen plan (see the module
    docstring for the rows, statistics, intervals, p-values and Holm).
    """
    arm_plan = _check_scope(plan, arm, stage)
    if isinstance(results, Mapping):
        results = results.values()
    if isinstance(results, str | bytes) or not isinstance(results, Iterable):
        raise TypeError("results must be a sequence of ItemResult values or a mapping to them")
    given = tuple(results)
    analysed = confirmatory_results(plan, given, arm=arm, stage=stage)
    _check_results(plan, arm, arm_plan, analysed.results.values())
    model = plan.arms.registry_ids[arm]
    pre = set(plan.months.pre_cutoff[model])
    months_of = {macro_item_id(month): month for month in plan.months.evaluation}
    keys = planned_keys(plan, arm, stage)
    months = tuple(_Month(item_id, months_of[item_id] in pre) for item_id in dict.fromkeys(key for key, _ in keys))
    by_key = MappingProxyType({key: analysed.results[key] for key in keys if key in analysed.results})
    context = _Context(plan=plan, arm_plan=arm_plan, months=months, by_key=by_key, settings=_Settings.of(plan))
    builders: dict[str, Callable[[], MetricEntry]] = {
        ESTIMATE_PRE: lambda: _period_metric(context, ESTIMATE_PRE, estimate=True, pre=True),
        ESTIMATE_POST: lambda: _period_metric(context, ESTIMATE_POST, estimate=True, pre=False),
        ACCURACY_PRE: lambda: _period_metric(context, ACCURACY_PRE, estimate=False, pre=True),
        ACCURACY_POST: lambda: _period_metric(context, ACCURACY_POST, estimate=False, pre=False),
        DID: lambda: _with_holm(_period_metric(context, DID, estimate=False, pre=None)),
        BRIER: lambda: _with_holm(_loss_metric(context, BRIER)),
        LOG_LOSS: lambda: _with_holm(_loss_metric(context, LOG_LOSS)),
        ACHIEVED_POWER: lambda: _power_metric(plan, given, arm),
    }
    entries = []
    for name in plan.analysis.metrics:
        if name in builders:
            entries.append(builders[name]())
        elif name in ENSEMBLE_METRICS:
            entries.append(MetricEntry(metric=name, status=DEFERRED, reason=DEFERRED_REASON))
        else:
            entries.append(MetricEntry(metric=name, status=MISSING, reason=UNKNOWN_METRIC))
    excluded = tuple(MappingProxyType({"item_id": item_id, "rendering": rendering, "reason": reason})
                     for (item_id, rendering), reason in sorted(analysed.excluded.items()))
    missing_results = tuple(MappingProxyType({"item_id": item_id, "rendering": rendering, "reason": NO_RESULT})
                            for item_id, rendering in keys if (item_id, rendering) not in by_key)
    n_pre = sum(1 for month in months if month.pre_cutoff)
    return PreRegisteredAnalysis(
        plan_sha256=plan.plan_sha256, arm=arm, stage=stage, role=arm_plan.role, supplies=arm_plan.supplies,
        model=model, settings=MappingProxyType(context.settings.to_dict()), planned_months=len(months),
        pre_cutoff_months=n_pre, post_cutoff_months=len(months) - n_pre, variants=tuple(plan.arms.variants),
        missing_variants=MappingProxyType(dict(arm_plan.missing_variants)),
        losses=MappingProxyType(dict(plan.scored_probability.losses)), excluded=excluded,
        missing_results=missing_results, metrics=tuple(entries))


def analysis_from_records(plan: StudyPlan, *, arm: str, stage: str = CONFIRMATORY_STAGE,
                          results_dir: str | Path = RESULTS_DIR) -> PreRegisteredAnalysis:
    """The analysis from the stored results of every planned item and rendering alone; nothing is requested.

    A planned result that is not stored is reported as missing with
    :data:`NO_RESULT`; a stored result that is malformed is refused.
    """
    _check_scope(plan, arm, stage)
    results = []
    for item_id, rendering in planned_keys(plan, arm, stage):
        path = stored_result_path(arm, stage, item_id, rendering, results_dir)
        if path.exists():
            results.append(load_result(path))
    return pre_registered_analysis(plan, results, arm=arm, stage=stage)


# ---------- the written output ----------

def analysis_path(arm: str, stage: str, out_dir: str | Path = ANALYSIS_DIR) -> Path:
    """``<out_dir>/pre-registered-<arm>-<stage>.json``; its markdown table sits beside it with ``.md``."""
    if arm not in ARMS or stage != CONFIRMATORY_STAGE:
        raise AnalysisError(f"unknown arm or stage {arm!r}/{stage!r} for the pre-registered analysis")
    return Path(out_dir) / f"pre-registered-{arm}-{stage}.json"


def _write_text(path: Path, text: str) -> None:
    """Write atomically: a crash leaves the previous file or the new one, never half of one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_analysis(analysis: PreRegisteredAnalysis, out_dir: str | Path = ANALYSIS_DIR) -> tuple[Path, Path]:
    """Write the analysis as JSON and markdown once; the same output is a no-op, a different one is refused."""
    if not isinstance(analysis, PreRegisteredAnalysis):
        raise TypeError(f"analysis must be a PreRegisteredAnalysis; got {type(analysis).__name__}")
    json_path = analysis_path(analysis.arm, analysis.stage, out_dir)
    markdown_path = json_path.with_suffix(".md")
    texts = {json_path: json.dumps(analysis.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
             markdown_path: analysis.to_markdown()}
    for path, text in texts.items():
        if path.exists() and path.read_text(encoding="utf-8") != text:
            raise AnalysisError(f"refusing to replace the different pre-registered analysis at {path}")
    for path, text in texts.items():
        if not path.exists():
            _write_text(path, text)
    return json_path, markdown_path


def main(argv: Sequence[str] | None = None) -> int:
    """Write the pre-registered analysis of one arm and stage from the stored results of the frozen plan.

    Exit codes: 0 written (or already written identically), 2 refused (another
    plan, results that do not belong to it, or a different output at the path).
    """
    parser = argparse.ArgumentParser(description="Pre-registered analysis: accuracy, losses and multiple testing "
                                                 "(task 7.10), from stored results; sends no request.")
    parser.add_argument("--plan", type=Path, required=True, help="the frozen plan.json")
    parser.add_argument("--plan-sha256", required=True, help="the plan hash the results carry")
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--stage", default=CONFIRMATORY_STAGE)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=ANALYSIS_DIR)
    args = parser.parse_args(argv)
    try:
        plan = load_plan(args.plan, expected_sha256=args.plan_sha256)
        analysis = analysis_from_records(plan, arm=args.arm, stage=args.stage, results_dir=args.results_dir)
        paths = write_analysis(analysis, args.out_dir)
    except (ValueError, TypeError, OSError) as exc:
        print(f"analysis: refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    for path in paths:
        print(path)
    return EXIT_WRITTEN


__all__ = [
    "ACHIEVED_POWER",
    "ANALYSIS_DIR",
    "ANALYSIS_METRICS",
    "ANALYSIS_SCHEMA",
    "COMPUTED",
    "CONDITIONAL_NOTE",
    "DEFERRED",
    "DEFERRED_REASON",
    "ENSEMBLE_METRICS",
    "HOLM_METHOD",
    "MISSING",
    "NO_ESTIMATE",
    "NO_RESULT",
    "POWER_APPROXIMATION",
    "REFERENCE",
    "TESTED_METRICS",
    "UNKNOWN_METRIC",
    "AnalysisError",
    "MetricEntry",
    "PreRegisteredAnalysis",
    "VariantCell",
    "analysis_from_records",
    "analysis_path",
    "main",
    "pre_registered_analysis",
    "write_analysis",
]


if __name__ == "__main__":
    raise SystemExit(main())
