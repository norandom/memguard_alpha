"""Negative-control decision task (task 7.9; Requirements 7.1, 7.2 and 7.3).

The negative control asks one arm for one outlook per held-out SEC EDGAR
control release (the plan's ``EdgarSplit.negative_control``: post-cutoff
releases the model cannot have memorised, so an honest adjustment should change
nothing) and reports, per variant with an adjustment, how far the adjustment
moves the outlook against no adjustment.

Run (:func:`run_negative_control`, Requirement 7.1). Each frozen release becomes
one :class:`~scripts.finance_sim.execute.ExecutionItem` with the outlook prompt
of :mod:`recall_guard.dataset.edgar_outlook` (the format the EDGAR calibrator
was fitted on), checked against the row and prompt hash the plan froze. The
items go unchanged through the 7.7 entry point
:func:`~scripts.finance_sim.execute.execute_items` on the ``negative_control``
stage: one recorded ensemble per release, every variant of the arm derived from
it, resumable, within the stage's frozen limits. The calibrators are the arm's
EDGAR calibrators (:func:`~scripts.finance_sim.execute.load_calibrators` with
kind ``edgar``); a macro calibrator is refused before any request, so it is never
applied to an EDGAR prompt. Either arm runs it: the serving arm's
full-distribution variants are reported as missing with the plan's reason.

Change (:func:`negative_control_change`, Requirement 7.2). For each variant with
an adjustment and each release, the change is the variant's adjusted outlook
minus its raw outlook (``adjusted_summary_value - summary_value`` of the stored
variant result); the raw outlook is the no-adjustment variant's of the same draw
mode on the same draws, which is checked. The statistic is the mean absolute
change over the releases with both values; the signed mean change and the share
of releases whose sign flips (the outlook definition's nine-decimal sign rule)
are reported beside it. A release without a result, a decision or a resolved
outlook is listed as missing with its reason and never substituted.

Interval. A non-circular moving-block percentile bootstrap over filing quarters
through :func:`recall_guard.core.significance.block_bootstrap`: the rows are the
filing quarters that hold a frozen release, in calendar order; each resample
draws ``ceil(Q / L)`` start quarters uniformly from ``0..Q - L``, takes ``L``
consecutive quarters from each, truncates to ``Q`` quarters and pools their
releases (a quarter drawn twice counts its releases twice). The block length
``L`` is the plan's ``negative_control_block_length``, the resample count its
``n_resamples``, the seed the plan's ``seed`` (the same for every variant, so
the variants are resampled alike) and the confidence ``1 - alpha``. A resample
whose quarters hold no computable release is dropped and counted.

Outcome (Requirement 7.3), against the plan's frozen ``tolerance``:
``no material change`` when the interval's upper bound is below the tolerance;
``discounting honest answers`` (the flag) when its lower bound is above it;
``inconclusive`` otherwise.
"""

from __future__ import annotations

import re
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from recall_guard.core.completion import CompletionClient
from recall_guard.core.full_distribution import FullDistributionClient, FullDistributionEvidence
from recall_guard.core.significance import Interval, block_bootstrap
from recall_guard.dataset.edgar_corpora import Quarter
from recall_guard.dataset.edgar_outlook import OUTLOOK as OUTLOOK_COMPONENT
from recall_guard.dataset.edgar_outlook import outlook_definition, render_outlook_prompt
from scripts.finance_sim.calibrate import CALIBRATORS_DIR, EDGAR, OUTLOOK, CalibrationRecord, EvidenceSettings
from scripts.finance_sim.execute import (
    RESULTS_DIR,
    STATUSES,
    UNRESOLVED_SUMMARY,
    ExecutionDraw,
    ExecutionItem,
    ExecutionRun,
    ItemResult,
    execute_items,
    load_calibrators,
)
from scripts.finance_sim.items import EdgarItem
from scripts.finance_sim.limits import ARMS, BudgetLedger
from scripts.finance_sim.plan import (
    NO_ADJUSTMENT,
    VARIANTS,
    ArmPlan,
    StudyPlan,
    ensure_unchanged,
    require_complete,
    sha256_of,
    variant_parts,
)

#: The ledger and execution stage of the negative control.
STAGE = "negative_control"
NO_MATERIAL_CHANGE = "no material change"
DISCOUNTING_HONEST_ANSWERS = "discounting honest answers"
INCONCLUSIVE = "inconclusive"
OUTCOMES: tuple[str, ...] = (NO_MATERIAL_CHANGE, DISCOUNTING_HONEST_ANSWERS, INCONCLUSIVE)
#: The variants with an adjustment, in plan order; each is reported against the no-adjustment variant of its mode.
ADJUSTED_VARIANTS: tuple[str, ...] = tuple(name for name in VARIANTS if variant_parts(name)[1] != NO_ADJUSTMENT)
REPORT_SCHEMA = "finance-sim-negative-control/1"
OUTCOME_RULE = (
    "per variant with an adjustment, against the frozen tolerance: 'no material change' when the interval's upper "
    "bound is below the tolerance; 'discounting honest answers' (the flag) when its lower bound is above the "
    "tolerance; 'inconclusive' otherwise"
)
METHOD = (
    "per release, the change of the outlook = adjusted minus raw, the raw outlook being the no-adjustment variant's "
    "of the same draw mode on the same draws; statistic: mean absolute change over the releases with both values, "
    "with the signed mean change and the share of releases whose sign flips beside it; interval: non-circular "
    "moving-block percentile bootstrap over filing quarters (rows are the quarters holding a frozen release, in "
    "calendar order; each resample draws ceil(Q / L) start quarters uniformly from 0..Q - L, takes L consecutive "
    "quarters from each, truncates to Q quarters and pools their releases), with the plan's seed, n_resamples, "
    "negative_control_block_length and confidence 1 - alpha; a resample with no computable release is dropped and "
    "counted"
)
NO_RESULT = "no result: the negative-control stage holds no result for this release, so the stage is incomplete"
NO_CHANGE = "no release has a computable change for this variant"
_EXAMPLE = re.search(r"e\.g\. (\{[^}]*\})", render_outlook_prompt("release"))
if _EXAMPLE is None:
    raise ImportError("the outlook prompt of recall_guard.dataset.edgar_outlook no longer shows a JSON example")
#: The anchor statement the design asks the report to carry.
ANCHOR_NOTE = (
    f"The outlook prompt ends with the JSON example {_EXAMPLE.group(1)}, a mild positive anchor that is the same for "
    "every arm and variant, so it cancels in the per-release change against no adjustment."
)
_QUARTER = re.compile(r"(\d{4})Q([1-4])")


class NegativeControlError(ValueError):
    """Negative-control releases, calibrators or results are refused; nothing was sent for them."""


def _interval_dict(interval: Interval | None) -> dict[str, Any] | None:
    if interval is None:
        return None
    return {"lo": interval.lo, "hi": interval.hi, "confidence": interval.confidence, "method": interval.method}


def _quarter_key(label: str) -> tuple[int, int]:
    match = _QUARTER.fullmatch(label)
    if match is None:
        raise NegativeControlError(f"a filing quarter is written YYYYQn; got {label!r}")
    return int(match.group(1)), int(match.group(2))


def _sign(value: float) -> int:
    """The outlook definition's decision: 1 above zero after rounding to nine decimals, else -1."""
    return int(outlook_definition.decide({OUTLOOK_COMPONENT: value}))


# ---------- releases ----------

def _releases(plan: StudyPlan, releases: Iterable[EdgarItem]) -> tuple[EdgarItem, ...]:
    """The frozen negative-control releases in plan order, each equal to the release the plan froze."""
    require_complete(plan)
    if isinstance(releases, str | bytes) or not isinstance(releases, Iterable):
        raise TypeError("releases must be a sequence of EdgarItem values")
    planned = plan.edgar.negative_control
    frozen = set(planned)
    by_id: dict[str, EdgarItem] = {}
    for release in releases:
        if not isinstance(release, EdgarItem):
            raise TypeError(f"releases must be EdgarItem values (scripts.finance_sim.items); got "
                            f"{type(release).__name__}")
        if release.item_id in by_id:
            raise NegativeControlError(f"release {release.item_id} is given more than once; one outlook per release")
        if release.item_id not in frozen or release.split != STAGE:
            raise NegativeControlError(f"{release.item_id} is not a negative-control release of the frozen plan; "
                                       "calibration rows and unused held-out rows never enter the negative control")
        if release.label != 0:
            raise NegativeControlError(f"release {release.item_id} has label {release.label}; the negative-control "
                                       "releases are control releases (label 0)")
        quarter = Quarter.of(release.published_at).label
        if release.quarter != quarter:
            raise NegativeControlError(f"release {release.item_id} carries the filing quarter {release.quarter!r}, "
                                       f"but it was published on {release.published_at.isoformat()} ({quarter})")
        frozen_hash = sha256_of({"prompt_sha256": release.prompt_sha256, "row_sha256": release.row_sha256})
        if plan.items.get(release.item_id) != frozen_hash:
            raise NegativeControlError(f"release {release.item_id} differs from the release the plan froze "
                                       "(row or prompt hash)")
        by_id[release.item_id] = release
    missing = [row_id for row_id in planned if row_id not in by_id]
    if missing:
        raise NegativeControlError(f"the frozen negative-control releases {missing} are missing")
    return tuple(by_id[row_id] for row_id in planned)


def negative_control_items(plan: StudyPlan, releases: Iterable[EdgarItem]) -> tuple[ExecutionItem, ...]:
    """One outlook item per frozen negative-control release, in plan order (Requirement 7.1).

    ``releases`` are the :class:`~scripts.finance_sim.items.EdgarItem` values of
    the corpus (``load_edgar_items(...).negative_control``). Each must be a
    negative-control release of the plan, given once, with the filing quarter of
    its publication date and the row and prompt hash the plan froze; every frozen
    release must be given. Anything else is refused before any request.
    """
    return tuple(ExecutionItem(release.item_id, EDGAR, OUTLOOK, release.prompt)
                 for release in _releases(plan, releases))


# ---------- the report ----------

@dataclass(frozen=True)
class ItemChange:
    """One release's change for one variant (``change = adjusted - raw``), or the reason it has none."""

    item_id: str
    quarter: str
    status: str | None
    raw: float | None = None
    adjusted: float | None = None
    change: float | None = None
    sign_flip: bool | None = None
    missing: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"item_id": self.item_id, "quarter": self.quarter, "status": self.status, "raw": self.raw,
                "adjusted": self.adjusted, "change": self.change, "sign_flip": self.sign_flip,
                "missing": self.missing}


@dataclass(frozen=True)
class VariantChange:
    """The change one variant with an adjustment makes against no adjustment, with its outcome.

    ``missing`` is the plan's reason when the arm cannot compute the variant (then
    nothing else is set), or the reason no statistic could be computed.
    ``interval`` is that of ``mean_absolute_change``, ``signed_interval`` that of
    ``signed_mean_change``, both from the same resamples. ``n_items`` counts the
    releases with a change; ``statuses`` counts the guarded statuses of the
    variant over the releases with a result; ``items`` lists every release with a
    result in plan order, a missing change with its reason.
    """

    variant: str
    mode: str
    adjustment: str
    reference: str
    missing: str | None = None
    outcome: str | None = None
    flagged: bool = False
    mean_absolute_change: float | None = None
    interval: Interval | None = None
    signed_mean_change: float | None = None
    signed_interval: Interval | None = None
    sign_flips: int | None = None
    sign_flip_share: float | None = None
    n_items: int = 0
    dropped_resamples: int | None = None
    statuses: Mapping[str, int] = field(default_factory=lambda: MappingProxyType({}))
    items: tuple[ItemChange, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"variant": self.variant, "mode": self.mode, "adjustment": self.adjustment,
                "reference": self.reference, "missing": self.missing, "outcome": self.outcome,
                "flagged": self.flagged, "mean_absolute_change": self.mean_absolute_change,
                "interval": _interval_dict(self.interval), "signed_mean_change": self.signed_mean_change,
                "signed_interval": _interval_dict(self.signed_interval), "sign_flips": self.sign_flips,
                "sign_flip_share": self.sign_flip_share, "n_items": self.n_items,
                "dropped_resamples": self.dropped_resamples, "statuses": dict(self.statuses),
                "items": [item.to_dict() for item in self.items]}


@dataclass(frozen=True)
class NegativeControlReport:
    """The negative-control report of one arm: the frozen settings and one entry per variant with an adjustment."""

    plan_sha256: str
    arm: str
    tolerance: float
    block_length: int
    n_resamples: int
    seed: int
    confidence: float
    quarters: tuple[str, ...]
    releases_per_quarter: Mapping[str, int]
    planned: int
    missing_results: Mapping[str, str]
    variants: tuple[VariantChange, ...]
    stage: str = STAGE
    outcome_rule: str = OUTCOME_RULE
    method: str = METHOD
    anchor_note: str = ANCHOR_NOTE

    @property
    def complete(self) -> bool:
        """Every frozen release has a result."""
        return not self.missing_results

    @property
    def flagged(self) -> tuple[str, ...]:
        """The variants reported as discounting honest answers (Requirement 7.3)."""
        return tuple(variant.variant for variant in self.variants if variant.flagged)

    def variant(self, name: str) -> VariantChange:
        for variant in self.variants:
            if variant.variant == name:
                return variant
        raise KeyError(name)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": REPORT_SCHEMA, "plan_sha256": self.plan_sha256, "arm": self.arm, "stage": self.stage,
                "tolerance": self.tolerance, "block_length": self.block_length, "n_resamples": self.n_resamples,
                "seed": self.seed, "confidence": self.confidence, "quarters": list(self.quarters),
                "releases_per_quarter": dict(self.releases_per_quarter), "planned": self.planned,
                "complete": self.complete, "missing_results": dict(self.missing_results),
                "flagged": list(self.flagged), "outcome_rule": self.outcome_rule, "method": self.method,
                "anchor_note": self.anchor_note, "variants": [variant.to_dict() for variant in self.variants]}


@dataclass(frozen=True)
class _Settings:
    """What the plan freezes for the negative control: the tolerance and the bootstrap settings."""

    tolerance: float
    block_length: int
    n_resamples: int
    seed: int
    confidence: float

    @classmethod
    def of(cls, plan: StudyPlan) -> _Settings:
        analysis = plan.analysis
        return cls(tolerance=analysis.tolerance, block_length=analysis.negative_control_block_length,
                   n_resamples=analysis.n_resamples, seed=plan.seed, confidence=1.0 - analysis.alpha)


def _results(plan: StudyPlan, results: Iterable[ItemResult] | Mapping[Any, ItemResult], arm: str, arm_plan: ArmPlan,
             releases: Sequence[EdgarItem]) -> dict[str, ItemResult]:
    """The stored outlook results by release id: this plan, arm and stage, one per frozen release prompt."""
    if isinstance(results, Mapping):
        results = results.values()
    if isinstance(results, str | bytes) or not isinstance(results, Iterable):
        raise TypeError("results must be a sequence of ItemResult values")
    prompts = {release.item_id: release.prompt_sha256 for release in releases}
    by_id: dict[str, ItemResult] = {}
    for result in results:
        if not isinstance(result, ItemResult):
            raise TypeError(f"results must be ItemResult values; got {type(result).__name__}")
        ensure_unchanged(plan, result.plan_sha256)
        if result.arm != arm:
            raise NegativeControlError(f"the result of {result.item_id} belongs to the {result.arm} arm, not the "
                                       f"{arm} arm")
        if (result.stage, result.kind, result.rendering) != (STAGE, EDGAR, OUTLOOK):
            raise NegativeControlError(f"the result of {result.item_id} is not an outlook result of the {STAGE} "
                                       f"stage (stage {result.stage!r}, kind {result.kind!r}, rendering "
                                       f"{result.rendering!r})")
        if result.item_id not in prompts:
            raise NegativeControlError(f"{result.item_id} is not a negative-control release of the frozen plan")
        if result.item_id in by_id:
            raise NegativeControlError(f"the result of {result.item_id} is given more than once")
        if result.prompt_sha256 != prompts[result.item_id]:
            raise NegativeControlError(f"the result of {result.item_id} answers another prompt than the release's "
                                       "outlook prompt")
        if (result.ensemble_draws, result.min_parsed) != (arm_plan.draws, arm_plan.min_parsed):
            raise NegativeControlError(f"the result of {result.item_id} has an ensemble of {result.ensemble_draws} "
                                       f"draws (min {result.min_parsed}); the frozen {arm} arm draws "
                                       f"{arm_plan.draws} (min {arm_plan.min_parsed})")
        by_id[result.item_id] = result
    return by_id


def _item_change(result: ItemResult, name: str, reference: str, quarter: str) -> ItemChange:
    chosen = result.variant(name)
    if chosen.missing is not None:
        return ItemChange(result.item_id, quarter, None, missing=chosen.missing)
    if chosen.status == "failed":
        return ItemChange(result.item_id, quarter, chosen.status, missing=f"no decision: {chosen.reason}")
    raw, adjusted = chosen.summary_value, chosen.adjusted_summary_value
    if raw is None or adjusted is None:
        return ItemChange(result.item_id, quarter, chosen.status, missing=UNRESOLVED_SUMMARY)
    if result.variant(reference).summary_value != raw:
        raise NegativeControlError(f"the {name} and {reference} results of {result.item_id} hold different raw "
                                   "outlooks; the variants are not paired on the same draws")
    return ItemChange(result.item_id, quarter, chosen.status, raw=raw, adjusted=adjusted, change=adjusted - raw,
                      sign_flip=_sign(adjusted) != _sign(raw))


def _mean_absolute(values: np.ndarray) -> float:
    return float(np.mean(np.abs(values)))


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values))


def _quarter_interval(groups: Sequence[np.ndarray], statistic: Callable[[np.ndarray], float],
                      settings: _Settings) -> tuple[Interval, int]:
    """The moving-block interval of ``statistic`` over the quarter groups, and the dropped resample count."""
    dropped = 0

    def pooled(sample: np.ndarray) -> float:
        nonlocal dropped
        values = np.concatenate([groups[int(index)] for index in sample])
        if values.size == 0:
            dropped += 1
            raise ValueError("no computable release in the resampled quarters")
        return statistic(values)

    interval = block_bootstrap(pooled, np.arange(len(groups)), block_length=settings.block_length,
                               n_resamples=settings.n_resamples, seed=settings.seed, confidence=settings.confidence)
    return interval, dropped


def _outcome(interval: Interval, tolerance: float) -> str:
    if interval.hi < tolerance:
        return NO_MATERIAL_CHANGE
    if interval.lo > tolerance:
        return DISCOUNTING_HONEST_ANSWERS
    return INCONCLUSIVE


def _variant_change(name: str, arm_plan: ArmPlan, releases: Sequence[EdgarItem], by_id: Mapping[str, ItemResult],
                    quarters: Sequence[str], settings: _Settings) -> VariantChange:
    mode, adjustment = variant_parts(name)
    reference = f"{mode}_{NO_ADJUSTMENT}"
    common = {"variant": name, "mode": mode, "adjustment": adjustment, "reference": reference}
    reason = arm_plan.missing_variants.get(name)
    if reason is not None:
        return VariantChange(**common, missing=reason)
    items = tuple(_item_change(by_id[release.item_id], name, reference, release.quarter)
                  for release in releases if release.item_id in by_id)
    counts = Counter(item.status for item in items if item.status is not None)
    statuses = MappingProxyType({status: counts[status] for status in STATUSES if counts[status]})
    computable = [item for item in items if item.change is not None]
    if not computable:
        return VariantChange(**common, missing=NO_CHANGE, statuses=statuses, items=items)
    groups = [np.asarray([item.change for item in computable if item.quarter == quarter], dtype=float)
              for quarter in quarters]
    try:
        interval, dropped = _quarter_interval(groups, _mean_absolute, settings)
        signed_interval, _ = _quarter_interval(groups, _mean, settings)
    except ValueError as exc:
        return VariantChange(**common, missing=f"the interval cannot be computed: {exc}", statuses=statuses,
                             items=items)
    changes = np.asarray([item.change for item in computable], dtype=float)
    outcome = _outcome(interval, settings.tolerance)
    flips = sum(1 for item in computable if item.sign_flip)
    return VariantChange(**common, outcome=outcome, flagged=outcome == DISCOUNTING_HONEST_ANSWERS,
                         mean_absolute_change=_mean_absolute(changes), interval=interval,
                         signed_mean_change=_mean(changes), signed_interval=signed_interval, sign_flips=flips,
                         sign_flip_share=flips / len(computable), n_items=len(computable),
                         dropped_resamples=dropped, statuses=statuses, items=items)


def negative_control_change(plan: StudyPlan, results: Iterable[ItemResult] | Mapping[Any, ItemResult],
                            releases: Iterable[EdgarItem], *, arm: str) -> NegativeControlReport:
    """The change each variant with an adjustment makes against no adjustment, with its interval and outcome.

    ``results`` are the stored ``negative_control`` item results of ``arm``
    (an :class:`~scripts.finance_sim.execute.ExecutionRun`'s ``results``, or
    results read back with :func:`~scripts.finance_sim.execute.load_result`);
    ``releases`` the frozen releases, which give each release's filing quarter.
    No request is sent. A result of another plan, arm, stage or prompt, or given
    twice, is refused. Every size and threshold comes from the frozen plan (see
    the module docstring for the statistic, interval and outcome rule).
    """
    ordered = _releases(plan, releases)
    if arm not in ARMS:
        raise NegativeControlError(f"arm must be one of {list(ARMS)}; got {arm!r}")
    arm_plan = plan.arms.arms[arm]
    by_id = _results(plan, results, arm, arm_plan, ordered)
    counts = Counter(release.quarter for release in ordered)
    quarters = tuple(sorted(counts, key=_quarter_key))
    settings = _Settings.of(plan)
    variants = tuple(_variant_change(name, arm_plan, ordered, by_id, quarters, settings) for name in ADJUSTED_VARIANTS)
    return NegativeControlReport(
        plan_sha256=plan.plan_sha256, arm=arm, tolerance=settings.tolerance, block_length=settings.block_length,
        n_resamples=settings.n_resamples, seed=settings.seed, confidence=settings.confidence, quarters=quarters,
        releases_per_quarter=MappingProxyType({quarter: counts[quarter] for quarter in quarters}),
        planned=len(ordered),
        missing_results=MappingProxyType({release.item_id: NO_RESULT for release in ordered
                                          if release.item_id not in by_id}),
        variants=variants)


# ---------- the run ----------

@dataclass(frozen=True)
class NegativeControlRun:
    """The negative-control stage as the 7.7 execution ran it, and its report."""

    execution: ExecutionRun
    report: NegativeControlReport


def _check_edgar_calibrators(calibrators: Mapping[str, CalibrationRecord]) -> None:
    if not isinstance(calibrators, Mapping):
        raise TypeError("calibrators must map each estimate source to its EDGAR CalibrationRecord")
    for source, record in calibrators.items():
        if not isinstance(record, CalibrationRecord):
            raise TypeError(f"calibrators[{source!r}] must be a CalibrationRecord; got {type(record).__name__}")
        if record.kind != EDGAR:
            raise NegativeControlError(f"the {source} calibrator is a {record.kind} calibrator; the negative "
                                       "control adjusts with the EDGAR calibrator only, and the macro calibrator is "
                                       "never applied to EDGAR prompts; nothing was sent")


def run_negative_control(
    plan: StudyPlan, releases: Iterable[EdgarItem], *, arm: str, client: CompletionClient, ledger: BudgetLedger,
    calibrators: Mapping[str, CalibrationRecord] | None = None, calibrators_dir: str | Path = CALIBRATORS_DIR,
    full_distribution: FullDistributionClient | None = None, evidence: EvidenceSettings | None = None,
    recorded: Iterable[ExecutionDraw] = (), on_draw: Callable[[ExecutionDraw], None] | None = None,
    on_request: Callable[[ExecutionItem, int, int | None], None] | None = None,
    on_evidence: Callable[[ExecutionItem, Mapping[int, FullDistributionEvidence]], None] | None = None,
    results_dir: str | Path = RESULTS_DIR, sleep: Callable[[float], None] = time.sleep,
) -> NegativeControlRun:
    """Ask ``arm`` for one outlook per frozen negative-control release and report the change (7.1 to 7.3).

    Before any request: the releases must be the plan's (:func:`negative_control_items`),
    and the calibrators the arm's EDGAR calibrators, by default loaded from
    ``calibrators_dir`` with kind ``edgar``; a macro calibrator is refused. The
    items then go unchanged through :func:`~scripts.finance_sim.execute.execute_items`
    on the ``negative_control`` stage with every other argument passed through,
    so its checks, recording seam, limits, evidence pass and resume apply as
    documented there (a frozen limit or unavailable evidence raises from it, with
    the recorded draws and written results kept). The report is computed from
    the results of the run.
    """
    ordered = _releases(plan, releases)
    items = tuple(ExecutionItem(release.item_id, EDGAR, OUTLOOK, release.prompt) for release in ordered)
    if arm not in ARMS:
        raise NegativeControlError(f"arm must be one of {list(ARMS)}; got {arm!r}")
    if calibrators is None:
        calibrators = load_calibrators(plan, arm=arm, kind=EDGAR, calibrators_dir=calibrators_dir)
    _check_edgar_calibrators(calibrators)
    execution = execute_items(plan, items, arm=arm, stage=STAGE, client=client, ledger=ledger,
                              calibrators=calibrators, full_distribution=full_distribution, evidence=evidence,
                              recorded=recorded, on_draw=on_draw, on_request=on_request, on_evidence=on_evidence,
                              results_dir=results_dir, sleep=sleep)
    return NegativeControlRun(execution=execution,
                              report=negative_control_change(plan, execution.results, ordered, arm=arm))


__all__ = [
    "ADJUSTED_VARIANTS",
    "ANCHOR_NOTE",
    "DISCOUNTING_HONEST_ANSWERS",
    "INCONCLUSIVE",
    "METHOD",
    "NO_CHANGE",
    "NO_MATERIAL_CHANGE",
    "NO_RESULT",
    "OUTCOMES",
    "OUTCOME_RULE",
    "REPORT_SCHEMA",
    "STAGE",
    "ItemChange",
    "NegativeControlError",
    "NegativeControlReport",
    "NegativeControlRun",
    "VariantChange",
    "negative_control_change",
    "negative_control_items",
    "run_negative_control",
]
