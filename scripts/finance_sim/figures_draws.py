"""First-draw figure (Figure 3) and draw-spread figure (Figure 8) of the worked example.

No request is sent and no stored statistic is recomputed. Both figures read two
stored outputs of one arm and one analysed stage (``full_run``, the positive
control, by default; ``negative_control`` also works):

- the stored item results
  (``runs/finance-sim/results/<arm>/<stage>/<item>.<rendering>.json``, loaded
  with :func:`~scripts.finance_sim.execute.load_result`), for the per-draw
  values of the example item and the per-draw values behind the
  exchangeability baseline;
- the stored ensemble metrics
  (``runs/finance-sim/analysis/ensemble-<arm>-<stage>.json``,
  :func:`~scripts.finance_sim.ensemble_metrics.metrics_path`), for every
  first-draw statistic: each item's first-draw mid-rank, category and verdict,
  the first-draw rate with its Wilson interval, the frozen band, and the items
  whose half-ensemble decisions were unresolved. These are read, never
  recomputed; where both files carry a value (the first draw's mid-rank and
  category) they must agree, or the figures are refused.

Each figure carries one message, in one panel, in the one academic style
(:mod:`scripts.finance_sim.figure_style`): both axes labelled with the quantity
and its unit, two series in one legend, nothing written inside the plot area.

Figure 3 ``draws``: how often the first draw of an ensemble differs from what
the ensemble gives. The stored first-draw rate (the share of items whose first
draw is off: its category differs from the ensemble's or its mid-rank among
the ensemble's draws lies outside the frozen band) with its Wilson interval
bar, for all items and per prompt version, beside the exchangeability
baseline: the share expected off if the first draw were exchangeable with the
other draws of its ensemble (per item, the share of its parsed draws that
would be off in the first draw's place, averaged over the items). With 8
distinct values about half of the mid-ranks fall outside [0.25, 0.75], so the
baseline is not zero.

Figure 8 ``draw-spread``: how much repeated answers to one prompt vary. Every
parsed draw of the example item is one point (its summary value: the defensive
spread, or the outlook on the negative control), tied draws side by side so
that each stays visible, beside the stored ensemble decisions: the ensemble
without adjustment and the adjusted ensembles. The value axis fits the draws,
so their spread is not flattened.

The example item is the first item in plan order with a decided first draw and
a consensus summary value unless one is given (the plan freezes no example
item). The table of Figure 3 (``draws.csv``, ``draws.md``) also keeps what the
earlier four-panel figure tabulated and no figure draws any more: the example
item's recall estimates per draw and the number of items at each first-draw
mid-rank. Outputs and the write-once rule are those of
:mod:`scripts.finance_sim.figures`; one render writes both figures; derived
statistics only. Matplotlib is imported inside the functions that draw.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from recall_guard.core.distribution import summarize_distribution
from recall_guard.core.ensemble import EnsembleSpec
from scripts.finance_sim.calibrate import MACRO
from scripts.finance_sim.ensemble_metrics import (
    ANALYSED_STAGES,
    ANALYSIS_DIR,
    CONSENSUS_VARIANT,
    FIRST_DRAW_RATE,
    HALF_SUMMARY,
    METRICS_SCHEMA,
    WILSON_METHOD,
    metrics_path,
)
from scripts.finance_sim.execute import (
    DEFINITIONS,
    EXECUTION_STAGES,
    KIND_RENDERINGS,
    RESULTS_DIR,
    UNRESOLVED_SUMMARY,
    ItemResult,
    load_result,
    result_path,
)
from scripts.finance_sim.figure_style import (
    FIGURES_DIR,
    FigureError,
    caption,
    figure_rc,
    lay_out,
    series_style,
    write_outputs,
)
from scripts.finance_sim.figures import (
    _HEX64,
    ADJUSTMENT_NAMES,
    DRAW_SPREAD,
    DRAWS,
    EDGAR_SOURCE,
    EVIDENCE_NAMES,
    EXIT_REFUSED,
    EXIT_WRITTEN,
    FIGURE_NUMBERS,
    FMP_STUDY_SOURCE,
    FRED_SOURCE,
    FigureTable,
    _count,
    _flat,
    _number,
    _percent,
    _read_json,
    _real,
    _text,
    figure_texts,
    interval_wording,
    rounded,
    route_label,
    variant_label,
)
from scripts.finance_sim.limits import ARMS, SAMPLING_ARM
from scripts.finance_sim.pilot import CONFIRMATORY_STAGE
from scripts.finance_sim.plan import ESTIMATE_SOURCES, NO_ADJUSTMENT, SINGLE_DRAW_INDEX, StudyPlan, load_plan
from scripts.finance_sim.published_text import published_reason

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

#: The arm both figures draw by default: it samples at a temperature above zero, the other arm's draws agree by
#: design.
DEFAULT_DRAWS_ARM = SAMPLING_ARM
#: Parsed draws the consensus split check needs (``EnsembleSpec.min_cluster_draws``); fewer cannot be unresolved.
SPLIT_CHECK_MIN_DRAWS: int = next(item.default for item in fields(EnsembleSpec) if item.name == "min_cluster_draws")
#: What the summary value of each item kind is, for the caption, and the label of its axis with the unit.
SUMMARY_NAMES: Mapping[str, str] = {
    MACRO: "defensive spread, the gold and cash loadings minus the equity and technology loadings",
    "edgar": "outlook score of the release",
}
SUMMARY_AXES: Mapping[str, str] = {
    MACRO: "defensive spread\n(difference of loadings)",
    "edgar": "outlook (score)",
}
#: The interval method recorded when the stored rate carries no interval.
NO_INTERVAL = "no interval was computed"
#: What the caption calls the interval of the stored first-draw rate, by the method the metrics state on it.
RATE_INTERVALS: Mapping[str, str] = MappingProxyType({WILSON_METHOD: "Wilson score"})
#: The published tables name the serving route and give a variant by its plain label; they carry no arm and no role.
DRAWS_COLUMNS: tuple[str, ...] = (
    "row", "route", "stage", "item_id", "rendering", "quantity", "draw_index", "value", "first_draw",
    "median", "q25", "q75", "variant", "status", "first_category", "consensus_category", "first_draw_rank",
    "outside_band", "category_differs", "off", "halves", "baseline_off", "baseline_outside", "group", "n",
    "count_off", "count_outside", "count_differs", "count_no_category", "rate", "interval_low", "interval_high",
    "reason")
#: The table of Figure 8: the example item's draws, their quartiles and the ensemble decisions.
DRAW_SPREAD_COLUMNS: tuple[str, ...] = (
    "row", "route", "stage", "item_id", "rendering", "quantity", "draw_index", "value", "first_draw",
    "median", "q25", "q75", "n", "variant", "status", "reason")
IN_HALVES = "in the half-ensemble statistic"
NOT_IN_HALVES = "not listed in the half-ensemble statistic"
EXAMPLE_DEFAULT = ("the first item in plan order with a decided first draw and a consensus summary value (the plan "
                   "freezes no example item)")
#: Figure sizes in inches (width, height): one panel each, the legend below it.
_RATE_SIZE_IN, _SPREAD_SIZE_IN = (4.8, 3.5), (5.4, 3.5)
#: In Figure 3 the observed share stands this far left of its place and the expected share this far right of it.
_RATE_DODGE = 0.12
#: In Figure 8 tied draws stand side by side: this far apart, within this width of the place of the single draws;
#: draws closer than this share of the range of the draws count as tied.
_SPREAD_STEP, _SPREAD_WIDTH, _TIE_SHARE = 0.07, 0.6, 0.03
#: The value axis of Figure 8 is at least this wide, so identical draws do not set a degenerate axis.
_SPREAD_MIN_RANGE = 0.1


# ---------- plain data ----------

@dataclass(frozen=True)
class DrawStrip:
    """One quantity of the example item, per draw index (None for an unparsed draw or a missing estimate).

    ``adjustment`` names the quantity (no adjustment for the summary value, the
    estimate source otherwise); ``median``, ``q25`` and ``q75`` summarise the
    parsed draws (linear interpolation, as ``summarize_distribution`` gives
    them), and ``reasons`` names why a draw has no value.
    """

    quantity: str
    label: str
    adjustment: str
    values: tuple[float | None, ...]
    reasons: tuple[str | None, ...]
    median: float | None
    q25: float | None
    q75: float | None

    @property
    def present(self) -> tuple[tuple[int, float], ...]:
        return tuple((index, value) for index, value in enumerate(self.values) if value is not None)

    @property
    def first(self) -> float | None:
        return self.values[SINGLE_DRAW_INDEX]


@dataclass(frozen=True)
class EnsembleMark:
    """One ensemble variant's stored value of a quantity, or why it is not drawn (``reason``)."""

    variant: str
    adjustment: str
    quantity: str
    label: str
    value: float | None
    status: str | None
    reason: str | None


@dataclass(frozen=True)
class ExampleItem:
    """The example item of F3: its draws and the stored ensemble results, per quantity."""

    item_id: str
    rendering: str
    chosen: str
    draws: int
    parsed: int
    failed: int
    first_rank: float | None
    first_category: Hashable | None
    consensus_category: Hashable | None
    summary: DrawStrip
    summary_marks: tuple[EnsembleMark, ...]
    estimates: tuple[DrawStrip, ...]
    estimate_marks: tuple[EnsembleMark, ...]


@dataclass(frozen=True)
class RankItem:
    """One analysed item's first draw as the 7.11 metrics give it, with its exchangeability shares.

    ``baseline_off`` and ``baseline_outside`` are the shares of the item's
    parsed draws that would be off, or outside the band, in the first draw's
    place. ``halves`` says whether the item entered the half-ensemble
    statistic, or why not; ``halves_unresolved`` is True only when 7.11 lists
    its halves as unresolved (otherwise None: not shown as resolved).
    """

    item_id: str
    rendering: str
    first_category: Hashable | None
    consensus_category: Hashable | None
    first_draw_rank: float | None
    outside_band: bool | None
    category_differs: bool | None
    off: bool
    parsed: int
    baseline_off: float
    baseline_outside: float
    halves: str
    halves_unresolved: bool | None

    @property
    def no_category(self) -> bool:
        """The consensus has no category: 7.11 includes the item only when its first draw is outside the band."""
        return self.consensus_category is None


@dataclass(frozen=True)
class RateRow:
    """The stored first-draw rate of one group (all items or one rendering) and its exchangeability baseline."""

    group: str
    label: str
    n: int
    off: int
    rate: float | None
    low: float | None
    high: float | None
    category_differs: int
    outside_band: int
    baseline_off: float | None
    baseline_outside: float | None
    missing: str | None
    no_category: int = 0


@dataclass(frozen=True)
class DrawsFigure:
    """Everything Figures 3 and 8 show, read from the stored results and metrics of one arm and stage.

    ``route`` is the serving route of the arm as data
    (:func:`scripts.finance_sim.figures.route_label`).
    """

    plan_sha256: str
    arm: str
    route: str
    role: str
    stage: str
    kind: str
    model: str
    control: str
    ensemble_draws: int
    half_draws: int | None
    band: tuple[float, float]
    confidence: float
    interval_method: str
    example: ExampleItem
    items: tuple[RankItem, ...]
    missing_items: tuple[tuple[str, str, str], ...]
    rates: tuple[RateRow, ...]
    halves_note: str | None

    @property
    def renderings(self) -> tuple[str, ...]:
        return KIND_RENDERINGS[self.kind]


# ---------- reading ----------

def _group_label(group: str) -> str:
    """An item group as a label: all items, or the items of one prompt version."""
    return "all items" if group == "all" else f"{group} prompt"


def _mid_rank(values: Sequence[float], value: float) -> float:
    """Share strictly below plus half the share tied (``recall_guard.core.distribution``'s mid-rank)."""
    below = sum(1 for other in values if other < value)
    tied = sum(1 for other in values if other == value)
    return (below + 0.5 * tied) / len(values)


def exchangeability_shares(values: Sequence[float], categories: Sequence[Hashable], consensus: Hashable | None,
                           band: tuple[float, float]) -> tuple[float, float]:
    """Expected shares off, and outside ``band``, of an item's first draw under exchangeability, given 7.11's rule.

    With a consensus category, the item enters the first-draw rate whatever
    its first draw does, and a draw is off when its category differs from the
    consensus or its mid-rank among ``values`` lies outside the band (bounds
    inclusive); under exchangeability the first draw is any parsed draw with
    equal chance, so the shares are those of all its parsed draws. Without a
    consensus category the 7.11 rule includes the item only when its first
    draw lies outside the band, and then it is off: given inclusion both
    shares are 1, and an unconditional share would compare the stored rate
    with a selection it never makes.
    """
    if not values or len(values) != len(categories):
        raise FigureError("the exchangeability shares need one category per parsed value and at least one value")
    if consensus is None:
        return 1.0, 1.0
    lower, upper = band
    off = outside = 0
    for value, category in zip(values, categories, strict=True):
        out = not lower <= _mid_rank(values, value) <= upper
        outside += out
        off += out or category != consensus
    return off / len(values), outside / len(values)


def read_ensemble_metrics(path: str | Path, plan: StudyPlan, *, arm: str, stage: str) -> Mapping[str, Any]:
    """The stored 7.11 metrics at ``path``, checked for schema, plan, arm and stage (never recomputed)."""
    data = _read_json(Path(path))
    if not isinstance(data, Mapping) or data.get("schema") != METRICS_SCHEMA:
        raise FigureError(f"{path} is not a {METRICS_SCHEMA} document")
    sha = data.get("plan_sha256")
    if not isinstance(sha, str) or not _HEX64.fullmatch(sha) or sha != plan.plan_sha256:
        raise FigureError(f"the ensemble metrics at {path} were written under plan {str(sha)[:12]}, not the frozen "
                          f"plan {plan.plan_sha256[:12]}")
    if (data.get("arm"), data.get("stage")) != (arm, stage):
        raise FigureError(f"the ensemble metrics at {path} are those of {data.get('arm')}/{data.get('stage')}, not "
                          f"{arm}/{stage}")
    for name in (FIRST_DRAW_RATE, HALF_SUMMARY):
        if name not in data.get("metrics", {}):
            raise FigureError(f"the ensemble metrics at {path} carry no {name}")
    return data


def _check_result(result: ItemResult, plan: StudyPlan, arm: str, stage: str) -> ItemResult:
    if (result.plan_sha256, result.arm, result.stage) != (plan.plan_sha256, arm, stage):
        raise FigureError(f"the stored result of {result.key} belongs to {result.arm}/{result.stage} under plan "
                          f"{result.plan_sha256[:12]}, not {arm}/{stage} under {plan.plan_sha256[:12]}")
    return result


def _strip(result: ItemResult, quantity: str, label: str, adjustment: str) -> DrawStrip:
    definition = DEFINITIONS[result.kind]
    if adjustment == NO_ADJUSTMENT:
        values = tuple(result.summary_values)
        reasons = tuple(None if value is not None else f"draw did not parse: {draw.failure_code}"
                        for draw, value in zip(result.recorded_draws, values, strict=True))
        distribution = summarize_distribution(result.draws_for(NO_ADJUSTMENT), definition)
        quantiles = None if distribution.summary_value is None else distribution.summary_value.quantiles
    else:
        draws = result.draws_for(adjustment)
        values = tuple(None if draw.parsed is None else draw.estimate for draw in draws)
        reasons = tuple(None if value is not None else (
            f"draw did not parse: {draw.failure_code}" if draw.parsed is None else "no estimate for this draw")
            for draw, value in zip(draws, values, strict=True))
        quantiles = summarize_distribution(draws, definition).estimate_quantiles
    if quantiles is None:
        return DrawStrip(quantity, label, adjustment, values, reasons, None, None, None)
    return DrawStrip(quantity, label, adjustment, values, reasons, quantiles[50], quantiles[25], quantiles[75])


def _mark(result: ItemResult, adjustment: str, quantity: str, label: str, value_of: str) -> EnsembleMark:
    variant = result.variant(f"ensemble_{adjustment}")
    if variant.missing is not None:
        return EnsembleMark(variant.variant, adjustment, quantity, label, None, None, variant.missing)
    value = getattr(variant, value_of)
    reason = None
    if value is None:
        if variant.status == "failed":
            reason = f"no decision: {variant.reason}"
        elif value_of in ("summary_value", "adjusted_summary_value"):
            # Not failed, so the decision exists and only an unresolved component leaves it without a summary.
            reason = UNRESOLVED_SUMMARY
        else:
            reason = variant.missing_reasons.get(value_of) or (
                f"no estimate: {variant.reason}" if variant.reason else
                f"the stored {variant.variant} variant carries no {value_of} (status {variant.status})")
    return EnsembleMark(variant.variant, adjustment, quantity, label, value, variant.status, reason)


def example_item(result: ItemResult, *, chosen: str = EXAMPLE_DEFAULT) -> ExampleItem:
    """The per-draw values of one stored result and its stored ensemble results (Figure 8 and the tables)."""
    if not any(draw.parsed is not None for draw in result.recorded_draws):
        raise FigureError(f"the example item {result.key} has no parsed draw to show")
    summary_name = "summary value"
    summary = _strip(result, summary_name, "single draw", NO_ADJUSTMENT)
    consensus = result.variant(CONSENSUS_VARIANT)
    distribution = summarize_distribution(result.draws_for(NO_ADJUSTMENT), DEFINITIONS[result.kind])
    rank = None if distribution.summary_value is None else distribution.summary_value.first_draw_rank
    if consensus.first_draw_rank is not None and rank is not None and not math.isclose(
            consensus.first_draw_rank, rank, rel_tol=0.0, abs_tol=1e-12):
        raise FigureError(f"the stored {CONSENSUS_VARIANT} first-draw rank of {result.key} does not match its draws")
    marks = [_mark(result, NO_ADJUSTMENT, summary_name, f"ensemble, {ADJUSTMENT_NAMES[NO_ADJUSTMENT]}",
                   "summary_value")]
    marks += [_mark(result, source, summary_name, f"ensemble, {ADJUSTMENT_NAMES[source]}",
                    "adjusted_summary_value") for source in ESTIMATE_SOURCES]
    estimates, estimate_marks = [], []
    for source in ESTIMATE_SOURCES:
        name = EVIDENCE_NAMES[source]
        if source in result.identities:
            estimates.append(_strip(result, f"p_memorized {source}", f"{name}, single draw", source))
        estimate_marks.append(_mark(result, source, f"p_memorized {source}", f"{name}, ensemble", "p_memorized"))
    return ExampleItem(
        item_id=result.item_id, rendering=result.rendering, chosen=chosen, draws=result.ensemble_draws,
        parsed=sum(1 for draw in result.recorded_draws if draw.parsed is not None),
        failed=sum(1 for draw in result.recorded_draws if draw.parsed is None), first_rank=rank,
        first_category=result.categories[SINGLE_DRAW_INDEX], consensus_category=consensus.category,
        summary=summary, summary_marks=tuple(marks), estimates=tuple(estimates), estimate_marks=tuple(estimate_marks))


def _halves(data: Mapping[str, Any]) -> tuple[dict[tuple[str, str], tuple[str, bool | None]], str | None]:
    """Per item: whether it entered the half-ensemble statistic of the consensus, or why not (7.11, read)."""
    cell = data["metrics"][HALF_SUMMARY]["results"][NO_ADJUSTMENT]
    found: dict[tuple[str, str], tuple[str, bool | None]] = {}
    for entry in cell["items"]:
        found[(_text(entry["item_id"], "item_id"), _text(entry["rendering"], "rendering"))] = (IN_HALVES, None)
    for entry in cell["missing_items"]:
        reason = _flat(_text(entry["reason"], "reason"))
        found[(_text(entry["item_id"], "item_id"), _text(entry["rendering"], "rendering"))] = (
            f"missing: {reason}", True if UNRESOLVED_SUMMARY in reason else None)
    note = None
    if cell["missing"] is not None and not cell["items"] and not cell["missing_items"]:
        note = f"the half-ensemble statistic of the consensus was not computed: {_flat(cell['missing'])}"
    return found, note


def _rate_row(group: str, entry: Mapping[str, Any], items: Sequence[RankItem]) -> RateRow:
    interval = entry["interval"]
    low = high = None
    if interval is not None:
        low, high = _real(interval["lo"], "interval lo"), _real(interval["hi"], "interval hi")
    chosen = [item for item in items if group == "all" or item.rendering == group]
    n = _count(entry["n"], "n")
    if n != len(chosen):
        raise FigureError(f"the stored first-draw rate of {_group_label(group)} counts {n} items; the stored items "
                          f"are {len(chosen)}")
    return RateRow(group=group, label=_group_label(group), n=n, off=_count(entry["off"], "off"),
                   rate=_real(entry["rate"], "rate"), low=low, high=high,
                   category_differs=_count(entry["category_differs"], "category_differs"),
                   outside_band=_count(entry["outside_band"], "outside_band"),
                   baseline_off=math.fsum(item.baseline_off for item in chosen) / n if n else None,
                   baseline_outside=math.fsum(item.baseline_outside for item in chosen) / n if n else None,
                   missing=None if entry["missing"] is None else _flat(entry["missing"]),
                   no_category=sum(1 for item in chosen if item.no_category))


def draws_figure(plan: StudyPlan, metrics: Mapping[str, Any], results: Iterable[ItemResult], *,
                 example: tuple[str, str] | None = None) -> DrawsFigure:
    """F3's data from the stored 7.11 metrics and the stored item results of the same arm and stage.

    ``results`` must hold the stored result of every item the metrics' first
    draw statistics list, and of the example item (``example``, by default
    the first item in plan order with a decided first draw and a consensus
    summary value). Refuses another
    plan, arm or stage, a result that is not analysed, and a stored first-draw
    value that the results contradict.
    """
    if not isinstance(plan, StudyPlan):
        raise TypeError(f"plan must be a StudyPlan; got {type(plan).__name__}")
    try:
        arm, stage = metrics["arm"], metrics["stage"]
        if arm not in ARMS or stage not in ANALYSED_STAGES:
            raise FigureError(f"the draw figure reads an analysed stage of a study arm; got {arm!r}/{stage!r}")
        if metrics.get("schema") != METRICS_SCHEMA or metrics.get("plan_sha256") != plan.plan_sha256:
            raise FigureError("the ensemble metrics are not this plan's 7.11 document")
        kind = EXECUTION_STAGES[stage]
        by_key: dict[tuple[str, str], ItemResult] = {}
        for result in results:
            if not isinstance(result, ItemResult):
                raise TypeError(f"results must be ItemResult values; got {type(result).__name__}")
            if result.key in by_key:
                raise FigureError(f"the result of {result.key} is given twice")
            by_key[result.key] = _check_result(result, plan, arm, stage)
        analysed = [(_text(entry["item_id"], "item_id"), _text(entry["rendering"], "rendering"))
                    for entry in metrics["items"]]
        settings = metrics["settings"]
        lower, upper = (_real(bound, "first_draw_band") for bound in settings["first_draw_band"])
        band = (float(lower), float(upper))
        if band != tuple(plan.analysis.first_draw_band):
            raise FigureError(f"the stored band {band} is not the frozen first_draw_band "
                              f"{tuple(plan.analysis.first_draw_band)}")
        if example is None:
            decided = [(row["item_id"], row["rendering"]) for row in metrics["metrics"][FIRST_DRAW_RATE]["items"]]
            shown_keys = [key for key in decided if key in by_key
                          and by_key[key].variant(CONSENSUS_VARIANT).summary_value is not None]
            if not shown_keys:
                raise FigureError(f"no item of the {arm}/{stage} ensemble metrics has a decided first draw and a "
                                  "consensus summary value; give the example item explicitly")
            key, chosen = shown_keys[0], EXAMPLE_DEFAULT
        else:
            key, chosen = (str(example[0]), str(example[1])), "given"
            if key not in analysed:
                raise FigureError(f"the example item {key} is not an analysed item of {arm}/{stage}")
        if key not in by_key:
            raise FigureError(f"no stored result of the example item {key}")
        shown = example_item(by_key[key], chosen=chosen)
        halves, halves_note = _halves(metrics)
        first = metrics["metrics"][FIRST_DRAW_RATE]
        items: list[RankItem] = []
        for row in first["items"]:
            item_key = (_text(row["item_id"], "item_id"), _text(row["rendering"], "rendering"))
            if item_key not in by_key:
                raise FigureError(f"no stored result of the analysed item {item_key}")
            result = by_key[item_key]
            pairs = [(value, category) for value, category in zip(result.summary_values, result.categories,
                                                                    strict=True) if value is not None]
            values, categories = [value for value, _ in pairs], [category for _, category in pairs]
            consensus = row["consensus_category"]
            stored_rank = _real(row["first_draw_rank"], "first_draw_rank")
            first_value = result.summary_values[SINGLE_DRAW_INDEX]
            if result.categories[SINGLE_DRAW_INDEX] != row["first_category"] or (
                    stored_rank is not None and (first_value is None or not math.isclose(
                        _mid_rank(values, first_value), stored_rank, rel_tol=0.0, abs_tol=1e-12))):
                raise FigureError(f"the stored result of {item_key} does not match the first draw the ensemble "
                                  "metrics record for it")
            if consensus != result.variant(CONSENSUS_VARIANT).category:
                raise FigureError(f"the stored consensus category of {item_key} does not match the ensemble metrics")
            off_share, outside_share = exchangeability_shares(values, categories, consensus, band)
            status, unresolved = halves.get(item_key, (NOT_IN_HALVES, None))
            items.append(RankItem(
                item_id=item_key[0], rendering=item_key[1], first_category=row["first_category"],
                consensus_category=consensus, first_draw_rank=stored_rank, outside_band=row["outside_band"],
                category_differs=row["category_differs"], off=bool(row["off"]), parsed=len(values),
                baseline_off=off_share, baseline_outside=outside_share, halves=status,
                halves_unresolved=unresolved))
        missing = tuple((_text(entry["item_id"], "item_id"), _text(entry["rendering"], "rendering"),
                         _flat(_text(entry["reason"], "reason"))) for entry in first["missing_items"])
        rates = [_rate_row("all", first["overall"], items)]
        rates += [_rate_row(rendering, first["by_rendering"][rendering], items)
                  for rendering in KIND_RENDERINGS[kind]]
        arm_info = metrics["ensemble_arm"]
        interval = next((row["interval"] for row in (first["overall"], *first["by_rendering"].values())
                         if row["interval"] is not None), None)
        return DrawsFigure(
            plan_sha256=plan.plan_sha256, arm=arm, route=route_label(plan, arm), role=_text(arm_info["role"], "role"),
            stage=stage, kind=kind,
            model=_text(arm_info["model"], "model"), control=_flat(metrics["control"]),
            ensemble_draws=_count(settings["draws"], "draws"),
            half_draws=None if settings["half_draws"] is None else _count(settings["half_draws"], "half_draws"),
            band=band, confidence=float(_real(settings["confidence"], "confidence")),
            interval_method=_flat(interval["method"]) if interval is not None else NO_INTERVAL,
            example=shown, items=tuple(items), missing_items=missing, rates=tuple(rates), halves_note=halves_note)
    except (KeyError, TypeError, AttributeError, ValueError) as exc:
        if isinstance(exc, FigureError):
            raise
        raise FigureError(f"the ensemble metrics or results are malformed: {exc!r}") from None


def draws_figure_from_records(plan: StudyPlan, *, arm: str = DEFAULT_DRAWS_ARM, stage: str = CONFIRMATORY_STAGE,
                              analysis_dir: str | Path = ANALYSIS_DIR, results_dir: str | Path = RESULTS_DIR,
                              example: tuple[str, str] | None = None) -> DrawsFigure:
    """F3's data from the stored metrics file and the stored result files of ``arm`` and ``stage``."""
    if arm not in ARMS or stage not in ANALYSED_STAGES:
        raise FigureError(f"the draw figure reads an analysed stage {list(ANALYSED_STAGES)} of an arm {list(ARMS)}; "
                          f"got {arm!r}/{stage!r}")
    path = metrics_path(arm, stage, analysis_dir)
    metrics = read_ensemble_metrics(path, plan, arm=arm, stage=stage)
    keys = {(entry["item_id"], entry["rendering"]) for entry in metrics["metrics"][FIRST_DRAW_RATE]["items"]}
    if example is not None:
        key = (str(example[0]), str(example[1]))
        if key not in {(entry["item_id"], entry["rendering"]) for entry in metrics["items"]}:
            raise FigureError(f"the example item {key} is not an analysed item of {arm}/{stage}")
        keys.add(key)
    results = []
    for item_id, rendering in sorted(keys):
        file = result_path(arm, stage, item_id, rendering, results_dir)
        if not file.exists():
            raise FigureError(f"no stored result at {arm}/{stage}/{item_id}.{rendering}.json under the results "
                              "directory, though the ensemble metrics list the item")
        results.append(load_result(file))
    return draws_figure(plan, metrics, results, example=example)


# ---------- tables and captions ----------

def _cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool | int | float):
        return _number(value)
    return _flat(value)


def _row(columns: Sequence[str], **values: object) -> tuple[str, ...]:
    return tuple(_cell(values.get(column)) for column in columns)


def _strips(example: ExampleItem) -> tuple[DrawStrip, ...]:
    return (example.summary, *example.estimates)


def _head(data: DrawsFigure) -> dict[str, object]:
    return {"route": data.route, "stage": data.stage}


def _published_halves(status: str) -> str:
    """Column halves of a published table: in the statistic, not listed, or why not in the published wording."""
    return status if status in (IN_HALVES, NOT_IN_HALVES) else published_reason(status)


def _example_rows(data: DrawsFigure, columns: Sequence[str], strips: Sequence[DrawStrip],
                  marks: Sequence[EnsembleMark]) -> list[tuple[str, ...]]:
    """The rows of the example item: every draw of each strip, its quartiles, and the stored ensemble values."""
    example = data.example
    where = {**_head(data), "item_id": example.item_id, "rendering": example.rendering}
    rows = []
    for strip in strips:
        for index, value in enumerate(strip.values):
            rows.append(_row(columns, row="draw", **where, quantity=strip.quantity, draw_index=index, value=value,
                             first_draw=index == SINGLE_DRAW_INDEX, reason=published_reason(strip.reasons[index])))
        rows.append(_row(columns, row="spread", **where, quantity=strip.quantity, median=strip.median, q25=strip.q25,
                         q75=strip.q75, n=len(strip.present),
                         first_draw_rank=example.first_rank if strip is example.summary else None))
    for mark in marks:
        rows.append(_row(columns, row="ensemble", **where, quantity=mark.quantity,
                         variant=variant_label(mark.variant), value=mark.value, status=mark.status,
                         reason=published_reason(mark.reason),
                         first_category=example.first_category if mark.variant == CONSENSUS_VARIANT else None,
                         consensus_category=example.consensus_category if mark.variant == CONSENSUS_VARIANT
                         else None))
    return rows


def _sources(data: DrawsFigure) -> tuple[str, ...]:
    return (FRED_SOURCE if data.kind == MACRO else EDGAR_SOURCE, FMP_STUDY_SOURCE)


def _checked(data: object) -> DrawsFigure:
    if not isinstance(data, DrawsFigure):
        raise TypeError(f"data must be a DrawsFigure; got {type(data).__name__}")
    return data


def draws_table(data: DrawsFigure) -> FigureTable:
    """Figure 3's table and caption: the rates with their baselines, the items behind them, the missing items, and
    what no figure draws: the example item's draws and recall estimates and the number of items at each mid-rank."""
    data = _checked(data)
    head = _head(data)
    example = data.example
    columns = DRAWS_COLUMNS
    rows = _example_rows(data, columns, _strips(example), (*example.summary_marks, *example.estimate_marks))
    for item in data.items:
        rows.append(_row(columns, row="item", **head, item_id=item.item_id, rendering=item.rendering,
                         first_category=item.first_category, consensus_category=item.consensus_category,
                         first_draw_rank=item.first_draw_rank, outside_band=item.outside_band,
                         category_differs=item.category_differs, off=item.off,
                         halves=_published_halves(item.halves),
                         baseline_off=item.baseline_off, baseline_outside=item.baseline_outside, n=item.parsed))
    for rendering, points in rank_counts(data).items():
        for rank, count in points:
            rows.append(_row(columns, row="rank_count", **head, rendering=rendering, first_draw_rank=rank, n=count))
    for item_id, rendering, reason in data.missing_items:
        rows.append(_row(columns, row="missing_item", **head, item_id=item_id, rendering=rendering,
                         reason=published_reason(reason)))
    for rate in data.rates:
        rows.append(_row(columns, row="rate", **head, group=rate.group, n=rate.n, count_off=rate.off,
                         count_outside=rate.outside_band, count_differs=rate.category_differs,
                         count_no_category=rate.no_category, rate=rate.rate,
                         interval_low=rate.low, interval_high=rate.high, baseline_off=rate.baseline_off,
                         baseline_outside=rate.baseline_outside, reason=published_reason(rate.missing)))
    return FigureTable(name=DRAWS, caption=_draws_caption(data), columns=columns, rows=tuple(rows),
                       notes=_draws_notes(data), sources=_sources(data))


def draw_spread_table(data: DrawsFigure) -> FigureTable:
    """Figure 8's table and caption: every draw of the example item, their quartiles and the ensemble decisions."""
    data = _checked(data)
    example = data.example
    rows = _example_rows(data, DRAW_SPREAD_COLUMNS, (example.summary,), example.summary_marks)
    notes = (f"The value is the {SUMMARY_NAMES[data.kind]}.",
             "The example is the first item in plan order whose first draw and ensemble both give a value."
             if example.chosen == EXAMPLE_DEFAULT else "The example was chosen for the figure.",
             "A draw without a value did not parse; the reason is in its row.",
             "The quartiles of the parsed draws are interpolated linearly.")
    return FigureTable(name=DRAW_SPREAD, caption=_spread_caption(data), columns=DRAW_SPREAD_COLUMNS,
                       rows=tuple(rows), notes=notes, sources=_sources(data))


def _rate_interval(data: DrawsFigure) -> str | None:
    """What the caption calls the stored interval of the rate, or None when the metrics carry none."""
    methods = [] if data.interval_method == NO_INTERVAL else [data.interval_method]
    return interval_wording(methods, "the first-draw figure", RATE_INTERVALS)


def _draws_caption(data: DrawsFigure) -> str:
    """Figure 3's caption: the observed share with its interval against the expected one, the items, the interval.

    When a first draw differs and how an item without a category enters are
    definitions: notes below the table (:func:`_draws_notes`).
    """
    overall = data.rates[0]
    kind = _rate_interval(data)
    if overall.rate is None:
        message = "No share of differing first draws is available, because no item has a decided first draw"
    else:
        interval = "no interval" if kind is None or overall.low is None or overall.high is None else (
            f"{rounded(overall.low)} to {rounded(overall.high)}")
        expected = "" if overall.baseline_off is None else (
            f", against {rounded(overall.baseline_off)} expected if the first draw were exchangeable with the "
            "ensemble's other draws")
        message = (f"The first draw of an ensemble differs from the ensemble's decision in {rounded(overall.rate)} "
                   f"of the items ({interval}){expected}")
    by_prompt = "; ".join(f"{rate.label}: {rate.n}" for rate in data.rates[1:])
    size = (f"{len(data.items)} items with a decided first draw on {data.route} ({by_prompt}), "
            f"{data.ensemble_draws} draws per ensemble; {len(data.missing_items)} items without one are listed in "
            "the table")
    interval = "No interval is stored for the observed share." if kind is None else (
        f"Bars and the range in parentheses are {_percent(data.confidence)} percent {kind} intervals; the expected "
        "share has no interval.")
    return caption(FIGURE_NUMBERS[DRAWS], message, n=size, interval=interval)


def _draws_notes(data: DrawsFigure) -> tuple[str, ...]:
    """The lines below Figure 3's table: when a first draw differs, what the expected share is, and how to read
    the rows no figure draws."""
    lower, upper = data.band
    unresolved = sum(1 for item in data.items if item.halves_unresolved)
    notes = [
        "A first draw differs when its decision category is not the ensemble's or its mid-rank among the "
        f"ensemble's draws lies outside {lower:g} to {upper:g}.",
        "An item whose ensemble has no category enters only when its first draw is outside that range, in the "
        "observed and in the expected share alike; it counts 1 in the expected share.",
        "The expected share averages, over the items, the share of an item's parsed draws that would differ in the "
        "first draw's place.",
        "The rows of kind draw, spread and ensemble hold the example item of Figure 8 with its recall estimates; "
        "the rows that count ranks give the number of items at each mid-rank of the first draw.",
        f"Column halves marks an item whose half-ensemble decisions were unresolved ({unresolved} items); an "
        "unmarked item is not shown as resolved.",
    ]
    if data.half_draws is not None and data.half_draws < SPLIT_CHECK_MIN_DRAWS:
        notes.append(f"Halves of {data.half_draws} draws are never assessed: the split check needs at least "
                     f"{SPLIT_CHECK_MIN_DRAWS} parsed draws.")
    if data.halves_note:
        notes.append("The half-ensemble statistic of the unadjusted ensemble was not computed.")
    return tuple(notes)


def _spread_caption(data: DrawsFigure) -> str:
    example = data.example
    values = [value for _, value in example.summary.present]
    consensus = example.summary_marks[0]
    ensemble = "the unadjusted ensemble gives no value" if consensus.value is None else (
        f"the unadjusted ensemble gives {rounded(consensus.value, signed=True)}")
    if len(values) == 1:
        message = (f"One answer to the example prompt parsed: it gives {rounded(values[0], signed=True)} and "
                   f"{ensemble}")
    elif min(values) == max(values):
        # Identical draws are no spread: the caption says so, with their count.
        message = (f"Repeated answers to one prompt do not vary in this example: all {example.parsed} parsed draws "
                   f"equal {rounded(values[0], signed=True)} and {ensemble}")
    else:
        median = "" if example.summary.median is None else (
            f" (median {rounded(example.summary.median, signed=True)})")
        message = ("Repeated answers to one prompt spread around the ensemble decision: the "
                   f"{example.parsed} parsed draws of the example range from {rounded(min(values), signed=True)} to "
                   f"{rounded(max(values), signed=True)}{median} and {ensemble}")
    size = (f"{example.draws} draws of one prompt ({example.item_id}, {example.rendering} prompt) on {data.route}, "
            f"{example.parsed} parsed and {example.failed} failed")
    notes = []
    absent = [mark.label for mark in example.summary_marks if mark.value is None]
    if absent:
        notes.append(f"Not drawn (reason in the table): {'; '.join(absent)}.")
    notes.append("No interval is drawn: every parsed draw is a point, tied draws side by side.")
    return caption(FIGURE_NUMBERS[DRAW_SPREAD], message, n=size, interval=" ".join(notes))


# ---------- drawing ----------

#: Series styles (``figure_style.series_style`` indexes): in Figure 3 the observed share and the expected share, in
#: Figure 8 an ensemble decision and a single draw. A solid black marker is what was decided or observed, an open
#: grey one what it is read against.
_OBSERVED_STYLE, _EXPECTED_STYLE = 0, 1
_ENSEMBLE_STYLE, _DRAW_STYLE = 0, 1


def _marker(index: int) -> dict[str, object]:
    """Series style ``index`` as an unconnected marker."""
    return {**series_style(index), "linestyle": "none"}


def _limits(values: Iterable[float | None], *, minimum: float) -> tuple[float, float]:
    """Axis limits around ``values``: 8 percent padding, at least ``minimum`` wide."""
    found = [value for value in values if value is not None]
    low, high = min(found), max(found)
    pad = 0.08 * (high - low)
    low, high = low - pad, high + pad
    if high - low < minimum:
        centre = (low + high) / 2.0
        low, high = centre - minimum / 2.0, centre + minimum / 2.0
    return low, high


def _places(axes: Axes, labels: Sequence[str]) -> None:
    """The x axis of a panel whose places are named groups: one tick per label, no tick marks."""
    axes.set_xticks(range(len(labels)), labels=list(labels))
    axes.set_xlim(-0.6, max(len(labels), 1) - 0.4)
    axes.tick_params(axis="x", length=0)


def rank_counts(data: DrawsFigure) -> dict[str, tuple[tuple[float, int], ...]]:
    """Per rendering, ``(mid-rank, number of items)`` in rank order: rows of Figure 3's table, not drawn."""
    found = {}
    for rendering in data.renderings:
        counts = Counter(item.first_draw_rank for item in data.items
                         if item.rendering == rendering and item.first_draw_rank is not None)
        found[rendering] = tuple(sorted(counts.items()))
    return found


def _plotted_rates(data: DrawsFigure) -> tuple[RateRow, ...]:
    """The rate rows Figure 3 draws: all items and each prompt version, or all items only for a single version."""
    return data.rates if len(data.renderings) > 1 else data.rates[:1]


def plot_draws(data: DrawsFigure) -> Figure:
    """Figure 3: the share of items whose first draw differs from the ensemble, against the exchangeable share.

    One panel in the academic style: per item group (all items, then each
    prompt version) the stored first-draw rate with its interval bar, and
    beside it the share expected if the first draw were exchangeable with the
    other draws of its ensemble. One legend below the panel names the two
    marks. ``data`` is plain data read from the stored outputs
    (:func:`draws_figure`). Returns a matplotlib Figure (no pyplot, no global
    state).
    """
    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    data = _checked(data)
    rates = _plotted_rates(data)
    with matplotlib.rc_context(figure_rc()):
        figure = Figure(figsize=_RATE_SIZE_IN, layout="constrained")
        axes = figure.subplots()
        observed, expected = _marker(_OBSERVED_STYLE), _marker(_EXPECTED_STYLE)
        for place, rate in enumerate(rates):
            x = place - _RATE_DODGE
            if rate.low is not None and rate.high is not None:
                axes.plot([x, x], [rate.low, rate.high], color=observed["color"], linestyle="-", marker="_",
                          markeredgecolor=observed["color"], zorder=2, gid=f"draws-rate-interval-{rate.group}")
            if rate.rate is not None:
                axes.plot([x], [rate.rate], zorder=3, gid=f"draws-rate-{rate.group}", **observed)
            if rate.baseline_off is not None:
                axes.plot([place + _RATE_DODGE], [rate.baseline_off], zorder=3, gid=f"draws-baseline-{rate.group}",
                          **expected)
        _places(axes, [f"{rate.label}\n(n = {rate.n})" for rate in rates])
        axes.set_ylim(-0.03, 1.03)
        axes.set_xlabel("prompt version")
        axes.set_ylabel("share of items whose first draw\ndiffers from the ensemble (share)")
        figure.legend(handles=[Line2D([], [], label="observed", **observed),
                               Line2D([], [], label="expected if draws were exchangeable", **expected)],
                      loc="outside lower center", ncols=2)
        lay_out(figure)
    return figure


def spread_offsets(values: Sequence[float]) -> list[float]:
    """Where each draw stands across the place of the single draws, so that tied draws are side by side.

    Draws whose values lie within ``_TIE_SHARE`` of the range of the draws
    count as tied; a group of tied draws is centred on the place, its draws
    ``_SPREAD_STEP`` apart or closer when the group would pass
    ``_SPREAD_WIDTH``. A draw without a tie stays at the centre. The value of
    a draw is never moved.
    """
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    if not order:
        return []
    tolerance = _TIE_SHARE * (values[order[-1]] - values[order[0]])
    groups: list[list[int]] = []
    for index in order:
        if groups and values[index] - values[groups[-1][0]] <= tolerance:
            groups[-1].append(index)
        else:
            groups.append([index])
    offsets = [0.0] * len(values)
    for group in groups:
        step = min(_SPREAD_STEP, _SPREAD_WIDTH / len(group))
        for position, index in enumerate(group):
            offsets[index] = (position - (len(group) - 1) / 2.0) * step
    return offsets


def _tick(name: str) -> str:
    """An adjustment as the second line of a tick label, the evidence mode on a line of its own."""
    return name.replace("adjusted with ", "adjusted with\n")


def plot_draw_spread(data: DrawsFigure) -> Figure:
    """Figure 8: every parsed draw of the example item as a point, beside the stored ensemble decisions.

    One panel in the academic style: the first place holds the single draws,
    one point each, tied draws side by side (:func:`spread_offsets`), so the
    reader sees how much repeated answers to one prompt vary; the next places
    hold the ensemble decisions as stored, without adjustment and after each
    adjustment. The value axis fits the values shown. An ensemble without a
    value is not drawn (the caption names it, the table gives the reason).
    One legend below the panel names the two marks. Returns a matplotlib
    Figure (no pyplot, no global state).
    """
    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    data = _checked(data)
    example = data.example
    values = [value for _, value in example.summary.present]
    marks = [mark for mark in example.summary_marks if mark.value is not None]
    with matplotlib.rc_context(figure_rc()):
        figure = Figure(figsize=_SPREAD_SIZE_IN, layout="constrained")
        axes = figure.subplots()
        draw, ensemble = _marker(_DRAW_STYLE), _marker(_ENSEMBLE_STYLE)
        axes.plot(spread_offsets(values), values, zorder=3, gid="spread-draws", **draw)
        for place, mark in enumerate(marks, start=1):
            axes.plot([place], [mark.value], zorder=3, gid=f"spread-ensemble-{mark.variant}", **ensemble)
        _places(axes, [f"single draws\n(n = {len(values)})",
                       *(f"ensemble,\n{_tick(ADJUSTMENT_NAMES[mark.adjustment])}" for mark in marks)])
        axes.set_ylim(*_limits([*values, *(mark.value for mark in marks)], minimum=_SPREAD_MIN_RANGE))
        axes.set_xlabel("decision rule")
        axes.set_ylabel(SUMMARY_AXES[data.kind])
        figure.legend(handles=[Line2D([], [], label="single draw", **draw),
                               Line2D([], [], label="ensemble decision", **ensemble)],
                      loc="outside lower center", ncols=2)
        lay_out(figure)
    return figure


# ---------- outputs ----------

def render_draws(plan: StudyPlan, *, arm: str = DEFAULT_DRAWS_ARM, stage: str = CONFIRMATORY_STAGE,
                 analysis_dir: str | Path = ANALYSIS_DIR, results_dir: str | Path = RESULTS_DIR,
                 example: tuple[str, str] | None = None, out_dir: str | Path = FIGURES_DIR) -> tuple[Path, ...]:
    """Render Figures 3 and 8 from the stored outputs of ``plan`` and write their six files once.

    Both figures are built before any file is written, so a refusal leaves
    the output directory unchanged. Returns the paths of ``draws`` and then of
    ``draw-spread``, each in the order of ``figure_paths``.
    """
    data = draws_figure_from_records(plan, arm=arm, stage=stage, analysis_dir=analysis_dir, results_dir=results_dir,
                                     example=example)
    texts = figure_texts(DRAWS, draws_table(data), lambda: plot_draws(data), out_dir)
    texts.update(figure_texts(DRAW_SPREAD, draw_spread_table(data), lambda: plot_draw_spread(data), out_dir))
    return write_outputs(texts)


def main(argv: Sequence[str] | None = None) -> int:
    """Render the first-draw figure and the draw-spread figure of the frozen plan from its stored outputs.

    Exit codes: 0 written (or already written identically), 2 refused (another
    plan, missing or malformed metrics or results, records that disagree, or a
    different figure at the output path).
    """
    parser = argparse.ArgumentParser(description="First-draw figure and draw-spread figure, from the stored item "
                                                 "results and ensemble metrics; sends no request.")
    parser.add_argument("--plan", type=Path, required=True, help="the frozen plan.json")
    parser.add_argument("--plan-sha256", required=True, help="the plan hash the stored outputs carry")
    parser.add_argument("--arm", default=DEFAULT_DRAWS_ARM, choices=ARMS)
    parser.add_argument("--stage", default=CONFIRMATORY_STAGE, choices=ANALYSED_STAGES)
    parser.add_argument("--example", metavar="ITEM_ID", help="the example item (default: the first item with a decided "
                                                                "first draw and a consensus summary value)")
    parser.add_argument("--rendering", help="the example item's rendering (required with --example)")
    parser.add_argument("--analysis-dir", type=Path, default=ANALYSIS_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args(argv)
    if (args.example is None) != (args.rendering is None):
        print("figures_draws: refused: --example and --rendering go together", file=sys.stderr)
        return EXIT_REFUSED
    try:
        plan = load_plan(args.plan, expected_sha256=args.plan_sha256)
        paths = render_draws(plan, arm=args.arm, stage=args.stage, analysis_dir=args.analysis_dir,
                             results_dir=args.results_dir, out_dir=args.out_dir,
                             example=None if args.example is None else (args.example, args.rendering))
    except (ValueError, TypeError, OSError) as exc:
        print(f"figures_draws: refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    for path in paths:
        print(path)
    return EXIT_WRITTEN


__all__ = [
    "DEFAULT_DRAWS_ARM",
    "DRAWS_COLUMNS",
    "DRAW_SPREAD_COLUMNS",
    "EXAMPLE_DEFAULT",
    "RATE_INTERVALS",
    "SPLIT_CHECK_MIN_DRAWS",
    "SUMMARY_NAMES",
    "DrawStrip",
    "DrawsFigure",
    "EnsembleMark",
    "ExampleItem",
    "RankItem",
    "RateRow",
    "draw_spread_table",
    "draws_figure",
    "draws_figure_from_records",
    "draws_table",
    "example_item",
    "exchangeability_shares",
    "main",
    "plot_draw_spread",
    "plot_draws",
    "rank_counts",
    "read_ensemble_metrics",
    "render_draws",
    "spread_offsets",
]


if __name__ == "__main__":
    raise SystemExit(main())
