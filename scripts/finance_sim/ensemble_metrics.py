"""Pre-registered analysis: ensemble metrics (task 7.11; Requirements 8.5, 8.6 and 8.7).

The analysis reads recorded data only and sends no request: the frozen plan and
the stored item results of the 7.7 execution
(:func:`~scripts.finance_sim.execute.load_result`) of one arm and one stage,
``full_run`` (the macro study, the positive control) or ``negative_control``
(the held-out EDGAR releases). The arm is an argument and every result names
it. Pilot items are excluded through
:func:`~scripts.finance_sim.pilot.confirmatory_results`, so the small pilot
ensembles never feed any of these statistics; every ensemble analysed must have
the arm's frozen size and minimum. Every size and name comes from the frozen
plan: the metric names (``plan.analysis.metrics``), the arm's ``draws`` and
``min_parsed``, ``n_resamples``, ``alpha`` (intervals at confidence
``1 - alpha``), ``first_draw_band`` and the plan's ``seed``.

Replay check. Before any statistic, every stored result must recompute from its
recorded draws: the per-draw summary values and categories, and each computed
variant through ``ItemResult.draws_for`` and ``decide_from_draws`` (status,
decision, adjusted decision, ``p_memorized``, category, summary values,
first-draw rank, draw hash, components not assessed). A variant the arm cannot
compute must carry the plan's reason. A result that does not recompute is
refused, so a replay from the recorded files reproduces the stored outputs.

Half-ensemble variance (8.5; ``half_ensemble_variance_summary`` and
``half_ensemble_variance_p_memorized``). With K draws per ensemble, the first
half is draw indices 0 to K/2 - 1 and the second K/2 to K - 1 (0 to 15 and 16
to 31 for 32 draws), each reduced with ``decide_from_draws`` as an ensemble of
K/2 draws that needs ``ceil(min_parsed x (K/2) / K)`` parsed draws, the arm's
parse share; the single draws are indices 0 and K/2, each reduced as a single
draw. All four come from the one recorded ensemble of an item and rendering,
with the adjustment's own per-draw estimates (``draws_for``). The summary
metric is computed per adjustment on the summary value (the defensive spread or
the outlook) of the adjusted decision, which for no adjustment is the consensus
itself; the estimate metric per estimate source on ``p_memorized``. Statistic,
interval and one-sided p-value are those of
:func:`~recall_guard.core.significance.half_ensemble_variance` with the plan's
``n_resamples`` and ``seed``, and ``dropped_resamples``,
``unbounded_resamples`` and ``interval_note`` are reported beside them. Before
the test, the share of items whose two single draws agree, and of items whose
singles and halves both agree, is checked: the probability that an item-level
resample has a zero single-draw variance, and that it is a dropped 0/0
resample, is reported as ``(share) ** n``. When the single draws agree on every
item the ratio is undefined and the cell is missing with the reason. Holm
adjusts the p-values of each metric across its computed cells (8.8). A cell
the arm cannot compute (the serving arm's full-distribution cells) is missing
with the plan's reason.

First-draw rate (8.6, ``first_draw_rate``). Per item and rendering, the first
draw is off when its category differs from the consensus category of the
ensemble without adjustment (``ensemble_none``), or when its mid-rank among the
parsed summary values lies outside the frozen band (inclusive bounds). The rate
is the share of items that are off, with a two-sided Wilson score interval at
confidence ``1 - alpha``, over all items of the stage (the control) and per
rendering. An item whose first draw did not parse, whose ensemble failed, or
whose consensus has no category while its first draw lies inside the band is
missing with the reason.

Random-answerer null (8.7, ``random_answerer_null``). The statistic is the
accuracy of the consensus decision (``ensemble_none``) over the items with a
consensus category and a realized outcome; a decision is correct when its
category is the defensive one exactly when the defensive pair beat the risk
pair (``macro_regime.is_correct``). The random answerer draws each component
uniformly on its grid (the regime loadings on the 0.1 grid) and passes it
through the definition's summary and ``decide``. The test is
:func:`~recall_guard.core.significance.random_answerer_null` with the plan's
``n_resamples`` null samples and ``seed``; its interval is the central range of
the null distribution. EDGAR outlooks have no realized outcome, so on the
negative control the null is missing with that reason.

Every value that cannot be computed is reported as missing with its reason,
never substituted: planned items without a stored result, items without the
values a statistic needs, and whole cells. Outputs: the JSON document and its
markdown table of derived statistics only, ``ensemble-<arm>-<stage>.json`` and
``.md`` under ``runs/finance-sim/analysis/``, written once and never replaced by
a different one.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from recall_guard.core.consensus import Tail, wilson_interval
from recall_guard.core.decision import ComponentSpec, DecisionDefinition
from recall_guard.core.distribution import RecordedDraw
from recall_guard.core.significance import TestResult, half_ensemble_variance, holm, random_answerer_null
from recall_guard.harness.guarded_decision import DrawMode, GuardConfig, GuardedDecision, decide_from_draws
from scripts.finance_sim.calibrate import MACRO
from scripts.finance_sim.execute import (
    DEFINITIONS,
    EDGAR_NO_OUTCOME,
    EXECUTION_STAGES,
    KIND_RENDERINGS,
    REALIZED_MISSING,
    RESULTS_DIR,
    UNRESOLVED_SUMMARY,
    ItemResult,
    VariantResult,
    load_result,
    planned_keys,
    result_path,
)
from scripts.finance_sim.limits import ARMS, RUNS_ROOT
from scripts.finance_sim.pilot import CONFIRMATORY_STAGE, PILOT_STAGE, confirmatory_results
from scripts.finance_sim.plan import (
    ADJUSTMENTS,
    ESTIMATE_SOURCES,
    FULL_DISTRIBUTION,
    NO_ADJUSTMENT,
    SINGLE_DRAW_INDEX,
    TOP_K,
    ArmPlan,
    StudyPlan,
    load_plan,
    require_complete,
)

#: Analysis outputs (design: ``runs/finance-sim/analysis/``).
ANALYSIS_DIR = RUNS_ROOT / "analysis"
METRICS_SCHEMA = "finance-sim-ensemble-metrics/1"
HALF_SUMMARY = "half_ensemble_variance_summary"
HALF_P_MEMORIZED = "half_ensemble_variance_p_memorized"
FIRST_DRAW_RATE = "first_draw_rate"
RANDOM_ANSWERER_NULL = "random_answerer_null"
#: The 7.11 statistics, named as the frozen plan names them (``plan.analysis.metrics``).
ENSEMBLE_METRICS: tuple[str, ...] = (HALF_SUMMARY, HALF_P_MEMORIZED, FIRST_DRAW_RATE, RANDOM_ANSWERER_NULL)
#: The stages whose items enter the analyses; the pilot enters none (8.9).
ANALYSED_STAGES: tuple[str, ...] = (CONFIRMATORY_STAGE, "negative_control")
CONTROLS: Mapping[str, str] = MappingProxyType({
    CONFIRMATORY_STAGE: ("positive control: the macro regime items of the arm's full run, identifying and anonymised "
                         "renderings"),
    "negative_control": "negative control: the held-out EDGAR releases, one outlook rendering each",
})
#: The consensus decision of an ensemble: the ensemble variant without adjustment.
CONSENSUS_VARIANT = "ensemble_" + NO_ADJUSTMENT
#: The category both decisions give a positive summary value (defensive, or a positive outlook).
POSITIVE_CATEGORY = 1

HALF_RULE = (
    "with K draws per ensemble (the arm's frozen draws), the first half is draw indices 0 to K/2 - 1 and the second "
    "K/2 to K - 1, each reduced with decide_from_draws as an ensemble of K/2 draws that needs "
    "ceil(min_parsed x (K/2) / K) parsed draws (the arm's parse share); the single draws are indices 0 and K/2, each "
    "reduced as a single draw; all four come from the one recorded ensemble of the item and rendering, with the "
    "adjustment's own per-draw estimates; run-to-run variance is half the mean squared pair difference over items"
)
SUMMARY_VALUE = ("the summary value (defensive spread or outlook) of the adjusted decision; without adjustment, of "
                 "the consensus itself")
ESTIMATE_VALUE = ("p_memorized: the mean per-draw memorization estimate over the parsed draws that carry one, or "
                  "the single draw's estimate")
FIRST_DRAW_RULE = (
    "per item and rendering, the first draw (index 0) is off when its category differs from the consensus category "
    "of ensemble_none or its mid-rank among the parsed summary values lies outside the frozen first_draw_band "
    "(bounds inclusive); the rate is the share of items that are off, with a two-sided Wilson score interval at "
    "confidence 1 - alpha"
)
NULL_RULE = (
    "statistic: accuracy of the consensus decision (ensemble_none) over the items with a consensus category and a "
    "realized outcome; the random answerer draws each component uniformly on its grid and passes it through the "
    "decision's summary and decide; Monte Carlo null with n_resamples samples and the plan's seed"
)
WILSON_METHOD = "Wilson score interval, two-sided, at confidence 1 - alpha"

NO_RESULT = "no stored result: the stage has not executed this item and rendering"
NO_ITEMS = "no item carries the values this statistic needs (each missing item is listed with its reason)"
ODD_ENSEMBLE = "the arm's ensemble has an odd number of draws, so it has no two disjoint halves of equal size"
ZERO_SINGLE_VARIANCE = ("the two single draws agree on every item, so the single-draw variance is zero and the "
                        "variance ratio is undefined")
FIRST_DRAW_UNPARSED = "the first draw did not parse"
UNDECIDED_FIRST_DRAW = ("the consensus has no category (a component is unresolved) and the first draw lies inside the "
                        "band, so whether it is off cannot be decided")
NO_CATEGORY = "the consensus has no category: a component is unresolved (split into separated clusters across draws)"
EXIT_WRITTEN, EXIT_REFUSED = 0, 2
_SHA256_LENGTH = 64


class EnsembleMetricsError(ValueError):
    """Results or outputs of the ensemble analysis are refused."""


# ---------- the arm and stage analysed ----------

@dataclass(frozen=True)
class _Context:
    """The frozen settings one arm and stage is analysed with."""

    plan: StudyPlan
    arm: str
    stage: str
    arm_plan: ArmPlan
    kind: str
    definition: DecisionDefinition
    confidence: float

    @property
    def half(self) -> int | None:
        draws = self.arm_plan.draws
        return draws // 2 if draws % 2 == 0 else None

    @property
    def half_min_parsed(self) -> int | None:
        if self.half is None:
            return None
        return -(-self.arm_plan.min_parsed * self.half // self.arm_plan.draws)


def _context(plan: StudyPlan, arm: str, stage: str) -> _Context:
    if not isinstance(plan, StudyPlan):
        raise TypeError(f"plan must be a StudyPlan; got {type(plan).__name__}")
    require_complete(plan)
    if arm not in ARMS:
        raise EnsembleMetricsError(f"arm must be one of {list(ARMS)}; got {arm!r}")
    if stage == PILOT_STAGE:
        raise EnsembleMetricsError("the pilot stage enters no later analysis (Requirement 8.9)")
    if stage not in ANALYSED_STAGES:
        raise EnsembleMetricsError(f"the ensemble metrics analyse the stages {list(ANALYSED_STAGES)}; got stage "
                                   f"{stage!r}")
    absent = [name for name in ENSEMBLE_METRICS if name not in plan.analysis.metrics]
    if absent:
        raise EnsembleMetricsError(f"the frozen plan does not pre-register the metrics {absent}")
    kind = EXECUTION_STAGES[stage]
    return _Context(plan=plan, arm=arm, stage=stage, arm_plan=plan.arms.arms[arm], kind=kind,
                    definition=DEFINITIONS[kind], confidence=1.0 - plan.analysis.alpha)


# ---------- reductions of recorded draws, no request ----------

def _source(adjustment: str) -> str:
    """The estimate source whose identity an adjustment's decisions carry (no adjustment uses the top-k one)."""
    return FULL_DISTRIBUTION if adjustment == FULL_DISTRIBUTION else TOP_K


def _summary(definition: DecisionDefinition, vector: Mapping[str, float | None] | None) -> float | None:
    """The summary value of a decision vector, None when it is missing or a component is unresolved."""
    if vector is None or any(value is None for value in vector.values()):
        return None
    return float(definition.summary(dict(vector)))


def _rank(decision: GuardedDecision) -> float | None:
    """The first draw's mid-rank among the parsed summary values, None without a distribution."""
    distribution = decision.distribution
    if distribution is None or distribution.summary_value is None:
        return None
    return distribution.summary_value.first_draw_rank


class _Reducer:
    """Decisions of one stored result recomputed from its recorded draws, cached by adjustment and draw range."""

    def __init__(self, context: _Context, result: ItemResult) -> None:
        self.context = context
        self.result = result
        self._draws: dict[str, tuple[RecordedDraw, ...]] = {}
        self._decisions: dict[tuple[str, int, int], GuardedDecision] = {}

    def decide(self, adjustment: str, start: int, stop: int) -> GuardedDecision:
        """The decision of draw indices ``start`` to ``stop - 1`` with ``adjustment``'s per-draw estimates."""
        key = (adjustment, start, stop)
        if key not in self._decisions:
            if adjustment not in self._draws:
                self._draws[adjustment] = self.result.draws_for(adjustment)
            context, result = self.context, self.result
            identity = result.identities[_source(adjustment)]
            size = stop - start
            arm_plan = context.arm_plan
            if size == 1:
                config = GuardConfig(context.definition, identity.source, draws=1, min_parsed=1, mode=DrawMode.SINGLE,
                                     temperature=arm_plan.temperature, seed_base=arm_plan.seed_base)
            else:
                minimum = result.min_parsed if size == result.ensemble_draws else context.half_min_parsed
                config = GuardConfig(context.definition, identity.source, draws=size, min_parsed=minimum,
                                     temperature=arm_plan.temperature, seed_base=arm_plan.seed_base)
            self._decisions[key] = decide_from_draws(self._draws[adjustment][start:stop], config,
                                                     replace(identity, draws=size))
        return self._decisions[key]


def _plain(vector: Mapping[str, float | None] | None) -> dict[str, float | None] | None:
    return None if vector is None else dict(vector)


def _decision_view(definition: DecisionDefinition, decision: GuardedDecision) -> tuple[Any, ...]:
    return (decision.status, _plain(decision.decision), _plain(decision.adjusted), decision.p_memorized,
            decision.category, decision.draws_sha256, tuple(decision.not_assessed),
            _summary(definition, decision.decision), _summary(definition, decision.adjusted), _rank(decision))


def _variant_view(variant: VariantResult) -> tuple[Any, ...]:
    return (variant.status, _plain(variant.decision), _plain(variant.adjusted), variant.p_memorized, variant.category,
            variant.draws_sha256, tuple(variant.not_assessed), variant.summary_value, variant.adjusted_summary_value,
            variant.first_draw_rank)


def _check_recomputes(context: _Context, result: ItemResult, reducer: _Reducer) -> None:
    """Refuse a stored result whose draw-level arrays or variants do not recompute from its recorded draws."""
    arm_plan = context.arm_plan
    if (result.ensemble_draws, result.min_parsed) != (arm_plan.draws, arm_plan.min_parsed):
        raise EnsembleMetricsError(
            f"the result of {result.key} holds {result.ensemble_draws} draws with minimum {result.min_parsed}; the "
            f"frozen {context.arm} arm draws {arm_plan.draws} with minimum {arm_plan.min_parsed}")
    if result.kind != context.kind:
        raise EnsembleMetricsError(f"the result of {result.key} is a {result.kind} item; the {context.stage} stage "
                                   f"analyses {context.kind} items")
    definition = context.definition
    summaries = tuple(None if draw.parsed is None else float(definition.summary(dict(draw.parsed)))
                      for draw in result.recorded_draws)
    categories = tuple(None if draw.parsed is None else definition.decide(dict(draw.parsed))
                       for draw in result.recorded_draws)
    if summaries != result.summary_values or categories != result.categories:
        raise EnsembleMetricsError(f"the stored result of {result.key} does not recompute from its recorded draws: "
                                   "its per-draw summary values or categories differ; refused")
    for variant in result.variants:
        reason = arm_plan.missing_variants.get(variant.variant)
        if variant.missing is not None or reason is not None:
            if variant.missing != reason:
                raise EnsembleMetricsError(f"the {variant.variant} variant of {result.key} does not recompute: the "
                                           f"frozen arm gives it as {'missing' if reason else 'computed'}; refused")
            continue
        size = 1 if variant.mode == "single" else result.ensemble_draws
        decision = reducer.decide(variant.adjustment, SINGLE_DRAW_INDEX, SINGLE_DRAW_INDEX + size)
        if _variant_view(variant) != _decision_view(definition, decision):
            raise EnsembleMetricsError(f"the stored {variant.variant} variant of {result.key} does not recompute from "
                                       "its recorded draws; refused")


# ---------- output helpers ----------

def _json_value(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _test_dict(result: TestResult) -> dict[str, Any]:
    interval = None if result.interval is None else {
        "lo": result.interval.lo, "hi": result.interval.hi, "confidence": result.interval.confidence,
        "method": result.interval.method}
    return {"statistic": result.statistic, "interval": interval, "p_value": result.p_value, "method": result.method,
            "parameters": {key: _json_value(value) for key, value in result.parameters.items()}}


def _item(key: tuple[str, str], **values: Any) -> dict[str, Any]:
    return {"item_id": key[0], "rendering": key[1], **values}


# ---------- half-ensemble versus single-draw variance (8.5) ----------

def _half_parts(context: _Context) -> tuple[tuple[str, int, int], ...]:
    half, draws = context.half, context.arm_plan.draws
    return ((f"first half (draws 0 to {half - 1})", 0, half), (f"second half (draws {half} to {draws - 1})", half, draws),
            (f"single draw {SINGLE_DRAW_INDEX}", SINGLE_DRAW_INDEX, SINGLE_DRAW_INDEX + 1),
            (f"single draw {half}", half, half + 1))


def _summary_value(context: _Context, decision: GuardedDecision) -> tuple[float | None, str | None]:
    if decision.status == "failed":
        return None, f"no decision: {decision.reason}"
    value = _summary(context.definition, decision.adjusted)
    return (value, None) if value is not None else (None, UNRESOLVED_SUMMARY)


def _estimate_value(context: _Context, decision: GuardedDecision) -> tuple[float | None, str | None]:
    if decision.status == "failed":
        return None, f"no decision: {decision.reason}"
    if decision.p_memorized is None:
        return None, f"no estimate: {decision.reason}"
    return decision.p_memorized, None


def _half_cell(context: _Context, metric: str, key_name: str, key: str, reducers: Sequence[_Reducer],
               value: Callable[[_Context, GuardedDecision], tuple[float | None, str | None]],
               missing: str | None) -> dict[str, Any]:
    """One cell of a half-ensemble metric: its item values, the up-front check and the test, or why it is missing."""
    adjustment = key
    cell: dict[str, Any] = {
        "metric": metric, "arm": context.arm, "stage": context.stage, key_name: key,
        "variants": [f"ensemble_{adjustment}", f"single_{adjustment}"], "missing": missing, "n_items": 0,
        "test": None, "holm_p_value": None, "dropped_resamples": None, "unbounded_resamples": None,
        "interval_note": None, "identical_singles": None, "identical_items": None,
        "zero_single_resample_probability": None, "zero_over_zero_resample_probability": None,
        "items": [], "missing_items": []}
    if missing is not None:
        return cell
    rows: list[tuple[float, float, float, float]] = []
    for reducer in reducers:
        values: list[float] = []
        reasons: list[str] = []
        for label, start, stop in _half_parts(context):
            found, reason = value(context, reducer.decide(adjustment, start, stop))
            if found is None:
                reasons.append(f"{label}: {reason}")
            else:
                values.append(found)
        if reasons:
            cell["missing_items"].append(_item(reducer.result.key, reason="; ".join(reasons)))
            continue
        rows.append((values[0], values[1], values[2], values[3]))
        cell["items"].append(_item(reducer.result.key, half_a=values[0], half_b=values[1], single_a=values[2],
                                   single_b=values[3]))
    n = len(rows)
    cell["n_items"] = n
    if n == 0:
        cell["missing"] = NO_ITEMS
        return cell
    identical_singles = sum(1 for row in rows if row[2] == row[3])
    identical_items = sum(1 for row in rows if row[2] == row[3] and row[0] == row[1])
    cell.update(identical_singles=identical_singles, identical_items=identical_items,
                zero_single_resample_probability=(identical_singles / n) ** n,
                zero_over_zero_resample_probability=(identical_items / n) ** n)
    if identical_singles == n:
        cell["missing"] = ZERO_SINGLE_VARIANCE
        return cell
    analysis = context.plan.analysis
    try:
        result = half_ensemble_variance(np.array([row[0] for row in rows]), np.array([row[1] for row in rows]),
                                        np.array([[row[2], row[3]] for row in rows]),
                                        n_resamples=analysis.n_resamples, seed=context.plan.seed,
                                        confidence=context.confidence)
    except ValueError as exc:
        cell["missing"] = f"the variance ratio cannot be computed: {exc}"
        return cell
    parameters = result.parameters
    cell.update(test=_test_dict(result), dropped_resamples=_json_value(parameters["dropped_resamples"]),
                unbounded_resamples=_json_value(parameters["unbounded_resamples"]),
                interval_note=parameters["interval_note"])
    return cell


def _half_metric(context: _Context, metric: str, reducers: Sequence[_Reducer]) -> dict[str, Any]:
    estimates = metric == HALF_P_MEMORIZED
    keys = ESTIMATE_SOURCES if estimates else ADJUSTMENTS
    value = _estimate_value if estimates else _summary_value
    cells: dict[str, dict[str, Any]] = {}
    for key in keys:
        missing = context.arm_plan.missing_variants.get(f"ensemble_{key}")
        if missing is None and context.half is None:
            missing = ODD_ENSEMBLE
        cells[key] = _half_cell(context, metric, "source" if estimates else "adjustment", key, reducers, value,
                                missing)
    raw = {key: cell["test"]["p_value"] for key, cell in cells.items() if cell["test"] is not None}
    for key, adjusted in holm(raw).items():
        cells[key]["holm_p_value"] = adjusted
    return {"metric": metric, "arm": context.arm, "stage": context.stage,
            "value": ESTIMATE_VALUE if estimates else SUMMARY_VALUE, "rule": HALF_RULE,
            "holm": "Holm step-down across the computed cells of this metric", "results": cells}


# ---------- first draw versus consensus (8.6) ----------

def _rate(context: _Context, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    off = sum(1 for row in rows if row["off"])
    bounds = wilson_interval(off, n, confidence=context.confidence, tail=Tail.TWO_SIDED)
    return {"n": n, "off": off, "rate": off / n if n else None,
            "category_differs": sum(1 for row in rows if row["category_differs"] is True),
            "outside_band": sum(1 for row in rows if row["outside_band"] is True),
            "interval": None if bounds is None else {"lo": bounds[0], "hi": bounds[1],
                                                     "confidence": context.confidence, "method": WILSON_METHOD},
            "missing": None if n else NO_ITEMS}


def _first_draw(context: _Context, reducers: Sequence[_Reducer]) -> dict[str, Any]:
    lower, upper = context.plan.analysis.first_draw_band
    rows: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    for reducer in reducers:
        result = reducer.result
        first = result.recorded_draws[SINGLE_DRAW_INDEX]
        consensus = reducer.decide(NO_ADJUSTMENT, 0, result.ensemble_draws)
        if first.parsed is None:
            missing.append(_item(result.key, reason=f"{FIRST_DRAW_UNPARSED}: {first.failure_code}"))
            continue
        if consensus.status == "failed":
            missing.append(_item(result.key, reason=f"no consensus: {consensus.reason}"))
            continue
        category: Hashable | None = result.categories[SINGLE_DRAW_INDEX]
        rank = _rank(consensus)
        differs = None if consensus.category is None else category != consensus.category
        outside = None if rank is None else not lower <= rank <= upper
        if differs is True or outside is True:
            off: bool | None = True
        elif differs is False and outside is False:
            off = False
        else:
            off = None
        if off is None:
            missing.append(_item(result.key, reason=UNDECIDED_FIRST_DRAW))
            continue
        rows.append(_item(result.key, first_category=category, consensus_category=consensus.category,
                          first_draw_rank=rank, category_differs=differs, outside_band=outside, off=off))
    return {"metric": FIRST_DRAW_RATE, "arm": context.arm, "stage": context.stage, "control": CONTROLS[context.stage],
            "variant": CONSENSUS_VARIANT, "band": [lower, upper], "rule": FIRST_DRAW_RULE,
            "overall": _rate(context, rows),
            "by_rendering": {rendering: _rate(context, [row for row in rows if row["rendering"] == rendering])
                             for rendering in KIND_RENDERINGS[context.kind]},
            "items": rows, "missing_items": missing}


# ---------- random-answerer null (8.7) ----------

def _grid(spec: ComponentSpec) -> tuple[float, ...]:
    """Every grid value of a component from its lower to its upper bound."""
    if spec.grid is None:
        raise EnsembleMetricsError(f"component {spec.name!r} has no grid, so a random answerer cannot draw on it")
    steps = (spec.upper - spec.lower) / spec.grid
    count = round(steps)
    if count < 1 or abs(steps - count) > 1e-9:
        raise EnsembleMetricsError(f"the bounds of component {spec.name!r} are not a whole number of grid steps apart")
    return tuple(round(spec.lower + step * spec.grid, 12) for step in range(count + 1))


def random_loadings(definition: DecisionDefinition, rng: np.random.Generator, size: int) -> np.ndarray:
    """``size`` random answers of ``definition``: each component uniform on its grid, one column per component."""
    if not isinstance(definition, DecisionDefinition):
        raise TypeError(f"definition must be a DecisionDefinition; got {type(definition).__name__}")
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise ValueError(f"size must be a non-negative integer; got {size!r}")
    columns = [np.asarray(grid)[rng.integers(0, len(grid), size=size)]
               for grid in (_grid(spec) for spec in definition.components)]
    return np.column_stack(columns)


def random_answerer_sampler(definition: DecisionDefinition) -> Callable[[np.random.Generator, int], np.ndarray]:
    """The null's sampler: random answers of :func:`random_loadings` passed through the definition's ``decide``."""
    if not isinstance(definition, DecisionDefinition) or definition.decide is None:
        raise EnsembleMetricsError("the random answerer needs a decision definition with a decide mapping")
    names = [spec.name for spec in definition.components]
    decide = definition.decide

    def sampler(rng: np.random.Generator, size: int) -> np.ndarray:
        answers = random_loadings(definition, rng, size).tolist()
        return np.array([decide(dict(zip(names, row, strict=True))) for row in answers])

    return sampler


def _null(context: _Context, reducers: Sequence[_Reducer]) -> dict[str, Any]:
    entry: dict[str, Any] = {"metric": RANDOM_ANSWERER_NULL, "arm": context.arm, "stage": context.stage,
                             "variant": CONSENSUS_VARIANT, "rule": NULL_RULE, "missing": None, "n_items": 0,
                             "test": None, "items": [], "missing_items": []}
    if context.kind != MACRO:
        entry["missing"] = EDGAR_NO_OUTCOME
        return entry
    rows: list[dict[str, Any]] = []
    outcomes: list[int] = []
    for reducer in reducers:
        result = reducer.result
        consensus = reducer.decide(NO_ADJUSTMENT, 0, result.ensemble_draws)
        stored = result.variant(CONSENSUS_VARIANT)
        if consensus.status == "failed":
            entry["missing_items"].append(_item(result.key, reason=f"no consensus: {consensus.reason}"))
        elif consensus.category is None:
            entry["missing_items"].append(_item(result.key, reason=NO_CATEGORY))
        elif result.outcome is None:
            entry["missing_items"].append(_item(result.key,
                                                reason=stored.missing_reasons.get("correct", REALIZED_MISSING)))
        else:
            correct = (consensus.category == POSITIVE_CATEGORY) == (result.outcome == 1)
            if stored.correct is not None and stored.correct != correct:
                raise EnsembleMetricsError(f"the stored correctness of {result.key} does not recompute; refused")
            rows.append(_item(result.key, category=consensus.category, correct=correct))
            outcomes.append(result.outcome)
    entry["items"] = rows
    entry["n_items"] = len(rows)
    if not rows:
        entry["missing"] = NO_ITEMS
        return entry
    realized = np.array(outcomes)

    def accuracy(answers: np.ndarray) -> float:
        return float(np.mean((np.asarray(answers) == POSITIVE_CATEGORY) == (realized == 1)))

    analysis = context.plan.analysis
    null_test = random_answerer_null(np.array([row["category"] for row in rows]),
                                     random_answerer_sampler(context.definition), statistic=accuracy,
                                     n_null=analysis.n_resamples, seed=context.plan.seed, confidence=context.confidence)
    entry["test"] = _test_dict(null_test)
    return entry


# ---------- the analysis ----------

@dataclass(frozen=True)
class EnsembleMetrics:
    """The ensemble metrics of one arm and stage: the canonical JSON document and its markdown table."""

    plan_sha256: str
    arm: str
    stage: str
    document: str

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self.document)

    def to_markdown(self) -> str:
        return _markdown(self.to_dict())


def _keys(rows: Iterable[tuple[str, str]], reasons: Mapping[tuple[str, str], str] | str) -> list[dict[str, Any]]:
    return [_item(key, reason=reasons if isinstance(reasons, str) else reasons[key]) for key in rows]


def ensemble_metrics(plan: StudyPlan, results: Iterable[ItemResult] | Mapping[Any, ItemResult], *, arm: str,
                     stage: str = CONFIRMATORY_STAGE) -> EnsembleMetrics:
    """The pre-registered ensemble metrics of ``arm`` and ``stage`` from stored item results; no request is sent.

    ``results`` may include pilot items, which are excluded with the reason;
    anything else must be a planned result of the arm and stage, given once,
    of the frozen ensemble size, and recompute from its recorded draws.
    """
    context = _context(plan, arm, stage)
    analysed = confirmatory_results(plan, results, arm=arm, stage=stage)
    planned = planned_keys(plan, arm, stage)
    present = [key for key in planned if key in analysed.results]
    reducers = []
    for key in present:
        reducer = _Reducer(context, analysed.results[key])
        _check_recomputes(context, reducer.result, reducer)
        reducers.append(reducer)
    arm_plan = context.arm_plan
    half = context.half
    builders: dict[str, Callable[[], dict[str, Any]]] = {
        HALF_SUMMARY: lambda: _half_metric(context, HALF_SUMMARY, reducers),
        HALF_P_MEMORIZED: lambda: _half_metric(context, HALF_P_MEMORIZED, reducers),
        FIRST_DRAW_RATE: lambda: _first_draw(context, reducers),
        RANDOM_ANSWERER_NULL: lambda: _null(context, reducers),
    }
    names = [name for name in plan.analysis.metrics if name in ENSEMBLE_METRICS]
    document = {
        "schema": METRICS_SCHEMA, "plan_sha256": plan.plan_sha256, "arm": arm, "stage": stage, "kind": context.kind,
        "control": CONTROLS[stage],
        "ensemble_arm": {"arm": arm, "role": arm_plan.role, "measures": arm_plan.measures,
                         "model": plan.arms.registry_ids[arm], "served_model": arm_plan.served_model,
                         "temperature": arm_plan.temperature, "seed_base": arm_plan.seed_base,
                         "draws": arm_plan.draws, "min_parsed": arm_plan.min_parsed},
        "settings": {"draws": arm_plan.draws, "min_parsed": arm_plan.min_parsed, "half_draws": half,
                     "half_min_parsed": context.half_min_parsed,
                     "first_half": None if half is None else [0, half - 1],
                     "second_half": None if half is None else [half, arm_plan.draws - 1],
                     "single_draws": None if half is None else [SINGLE_DRAW_INDEX, half],
                     "first_draw_band": list(plan.analysis.first_draw_band),
                     "n_resamples": plan.analysis.n_resamples, "seed": plan.seed, "alpha": plan.analysis.alpha,
                     "confidence": context.confidence},
        "metric_names": names,
        "items": [_item(key) for key in present],
        "excluded": _keys(sorted(analysed.excluded), analysed.excluded),
        "missing_results": _keys([key for key in planned if key not in analysed.results], NO_RESULT),
        "metrics": {name: builders[name]() for name in names},
    }
    text = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    return EnsembleMetrics(plan_sha256=plan.plan_sha256, arm=arm, stage=stage, document=text)


def ensemble_metrics_from_records(plan: StudyPlan, *, arm: str, stage: str = CONFIRMATORY_STAGE,
                                  results_dir: str | Path = RESULTS_DIR) -> EnsembleMetrics:
    """The ensemble metrics from the stored result of every planned item and rendering of the stage; no request."""
    _context(plan, arm, stage)
    results = []
    for item_id, rendering in planned_keys(plan, arm, stage):
        path = result_path(arm, stage, item_id, rendering, results_dir)
        if path.exists():
            results.append(load_result(path))
    return ensemble_metrics(plan, results, arm=arm, stage=stage)


# ---------- the markdown table ----------

def _number(value: float | None) -> str:
    return "" if value is None else f"{value:.4g}"


def _interval(interval: Mapping[str, Any] | None) -> str:
    if interval is None:
        return "none"
    return f"{interval['lo']:.4g} to {interval['hi']:.4g} ({interval['confidence'] * 100:g} percent)"


def _cell_text(text: str) -> str:
    return text.replace("|", "/").replace("\n", " ")


def _half_rows(name: str, metric: Mapping[str, Any]) -> list[list[str]]:
    rows = []
    estimates = name == HALF_P_MEMORIZED
    for key in ESTIMATE_SOURCES if estimates else ADJUSTMENTS:
        cell = metric["results"][key]
        label = f"{key} estimate" if estimates else f"{key} adjustment"
        if cell["test"] is None:
            rows.append([name, label, str(cell["n_items"]), "missing", "", "", "", cell["missing"]])
            continue
        test = cell["test"]
        note = (f"dropped {cell['dropped_resamples']}, unbounded {cell['unbounded_resamples']} of "
                f"{test['parameters']['n_resamples']} resamples; P(zero single-draw variance per resample) "
                f"{cell['zero_single_resample_probability']:.3g}")
        if cell["interval_note"] is not None:
            note += f"; interval: {cell['interval_note']}"
        rows.append([name, label, str(cell["n_items"]), _number(test["statistic"]), _interval(test["interval"]),
                     _number(test["p_value"]), _number(cell["holm_p_value"]), note])
    return rows


def _rate_row(name: str, label: str, rate: Mapping[str, Any]) -> list[str]:
    if rate["missing"] is not None:
        return [name, label, str(rate["n"]), "missing", "", "", "", rate["missing"]]
    note = (f"{rate['off']} of {rate['n']} off: {rate['category_differs']} with another category, "
            f"{rate['outside_band']} outside the band")
    return [name, label, str(rate["n"]), _number(rate["rate"]), _interval(rate["interval"]), "", "", note]


def _markdown(data: Mapping[str, Any]) -> str:
    arm = data["ensemble_arm"]
    metrics = data["metrics"]
    lines = [
        f"# Ensemble metrics: {data['arm']} arm, {data['stage']} stage",
        "",
        f"Ensemble arm: the {data['arm']} arm ({arm['role']} role), {arm['measures']}. Model {arm['model']}, "
        f"{arm['draws']} draws per ensemble with at least {arm['min_parsed']} parsed, temperature "
        f"{arm['temperature']:g}.",
        "",
        f"Control: {data['control']}. Frozen plan {data['plan_sha256']}. Items analysed: {len(data['items'])}; "
        f"pilot items excluded: {len(data['excluded'])}; planned items without a stored result: "
        f"{len(data['missing_results'])}.",
        "",
        "| Metric | Cell | n | Statistic | Interval | p-value | Holm p-value | Note |",
        "|---|---|---|---|---|---|---|---|",
    ]
    rows: list[list[str]] = []
    for name in data["metric_names"]:
        metric = metrics[name]
        if name in (HALF_SUMMARY, HALF_P_MEMORIZED):
            rows.extend(_half_rows(name, metric))
        elif name == FIRST_DRAW_RATE:
            rows.append(_rate_row(name, "all items", metric["overall"]))
            rows.extend(_rate_row(name, rendering, metric["by_rendering"][rendering])
                        for rendering in KIND_RENDERINGS[data["kind"]])
        elif name == RANDOM_ANSWERER_NULL:
            if metric["test"] is None:
                rows.append([name, "consensus accuracy", str(metric["n_items"]), "missing", "", "", "",
                             metric["missing"]])
            else:
                test = metric["test"]
                rows.append([name, "consensus accuracy", str(metric["n_items"]), _number(test["statistic"]),
                             _interval(test["interval"]), _number(test["p_value"]), "",
                             "interval: central range of the random-answerer null distribution"])
    lines.extend("| " + " | ".join(_cell_text(cell) for cell in row) + " |" for row in rows)
    settings = data["settings"]
    halves = (ODD_ENSEMBLE if settings["half_draws"] is None
              else f"Halves of {settings['half_draws']} draws need {settings['half_min_parsed']} parsed")
    lines += [
        "",
        f"Half-ensemble variance: {HALF_RULE}; statistic: half-ensemble over single-draw variance, below 1 when the "
        "ensemble reduces variance; item-level percentile bootstrap interval and one-sided p-value; Holm across the "
        f"cells of each metric. {halves}.",
        f"First-draw rate: {FIRST_DRAW_RULE}; band {settings['first_draw_band'][0]:g} to "
        f"{settings['first_draw_band'][1]:g}.",
        f"Random-answerer null: {NULL_RULE}.",
        f"Resamples {settings['n_resamples']}, seed {settings['seed']}, confidence {settings['confidence']:g}. "
        "Intervals assume independent items and draws. Derived statistics only: no item, reply or licensed series "
        "is listed.",
        "",
    ]
    return "\n".join(lines)


# ---------- the written outputs ----------

def metrics_path(arm: str, stage: str, out_dir: str | Path = ANALYSIS_DIR) -> Path:
    """``<out_dir>/ensemble-<arm>-<stage>.json``; its markdown table sits beside it with ``.md``."""
    if arm not in ARMS or stage not in ANALYSED_STAGES:
        raise EnsembleMetricsError(f"unknown arm or stage {arm!r}/{stage!r} for the ensemble metrics")
    return Path(out_dir) / f"ensemble-{arm}-{stage}.json"


def _write_text(path: Path, text: str) -> None:
    """Write atomically: a crash leaves the previous file or the new one, never half of one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_metrics(metrics: EnsembleMetrics, out_dir: str | Path = ANALYSIS_DIR) -> tuple[Path, Path]:
    """Write the JSON document and the markdown table once; the same outputs are a no-op, different ones refused."""
    if not isinstance(metrics, EnsembleMetrics):
        raise TypeError(f"metrics must be EnsembleMetrics; got {type(metrics).__name__}")
    json_path = metrics_path(metrics.arm, metrics.stage, out_dir)
    markdown_path = json_path.with_suffix(".md")
    texts = {json_path: metrics.document, markdown_path: metrics.to_markdown()}
    for path, text in texts.items():
        if path.exists() and path.read_text(encoding="utf-8") != text:
            raise EnsembleMetricsError(f"refusing to replace the different ensemble metrics at {path}")
    for path, text in texts.items():
        if not path.exists():
            _write_text(path, text)
    return json_path, markdown_path


def main(argv: Sequence[str] | None = None) -> int:
    """Write the ensemble metrics of one arm and stage from the private results of the frozen plan.

    Exit codes: 0 written (or already written identically), 2 refused (another
    plan, results that do not recompute, or different outputs at the path).
    """
    parser = argparse.ArgumentParser(description="Pre-registered ensemble metrics (task 7.11) from recorded results; "
                                                 "sends no request.")
    parser.add_argument("--plan", type=Path, required=True, help="the frozen plan.json")
    parser.add_argument("--plan-sha256", required=True, help="the plan hash the results carry")
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--stage", default=CONFIRMATORY_STAGE, choices=ANALYSED_STAGES)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=ANALYSIS_DIR)
    args = parser.parse_args(argv)
    if len(args.plan_sha256) != _SHA256_LENGTH:
        print("ensemble metrics: refused: --plan-sha256 must be 64 hex characters", file=sys.stderr)
        return EXIT_REFUSED
    try:
        plan = load_plan(args.plan, expected_sha256=args.plan_sha256)
        metrics = ensemble_metrics_from_records(plan, arm=args.arm, stage=args.stage, results_dir=args.results_dir)
        paths = write_metrics(metrics, args.out_dir)
    except (ValueError, TypeError, OSError) as exc:
        print(f"ensemble metrics: refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    for path in paths:
        print(path)
    return EXIT_WRITTEN


__all__ = [
    "ANALYSED_STAGES",
    "ANALYSIS_DIR",
    "CONSENSUS_VARIANT",
    "CONTROLS",
    "ENSEMBLE_METRICS",
    "FIRST_DRAW_RATE",
    "FIRST_DRAW_RULE",
    "FIRST_DRAW_UNPARSED",
    "HALF_P_MEMORIZED",
    "HALF_RULE",
    "HALF_SUMMARY",
    "METRICS_SCHEMA",
    "NO_CATEGORY",
    "NO_ITEMS",
    "NO_RESULT",
    "NULL_RULE",
    "ODD_ENSEMBLE",
    "RANDOM_ANSWERER_NULL",
    "UNDECIDED_FIRST_DRAW",
    "ZERO_SINGLE_VARIANCE",
    "EnsembleMetrics",
    "EnsembleMetricsError",
    "ensemble_metrics",
    "ensemble_metrics_from_records",
    "main",
    "metrics_path",
    "random_answerer_sampler",
    "random_loadings",
    "write_metrics",
]


if __name__ == "__main__":
    raise SystemExit(main())
