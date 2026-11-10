"""Draw-distribution figure F3 of the worked example (task 8.3; Requirements 10.4 and 10.6).

No request is sent and no stored statistic is recomputed. The figure reads two
stored outputs of one arm and one analysed stage (``full_run``, the positive
control, by default; ``negative_control`` also works):

- the item results of the 7.7 execution
  (``runs/finance-sim/results/<arm>/<stage>/<item>.<rendering>.json``, loaded
  with :func:`~scripts.finance_sim.execute.load_result`), for the per-draw
  values of the example item and the per-draw values behind the
  exchangeability baseline;
- the ensemble metrics of task 7.11
  (``runs/finance-sim/analysis/ensemble-<arm>-<stage>.json``,
  :func:`~scripts.finance_sim.ensemble_metrics.metrics_path`), for every
  first-draw statistic: each item's first-draw mid-rank, category and verdict,
  the first-draw rate with its Wilson interval, the frozen band, and the items
  whose half-ensemble decisions were unresolved. These are read, never
  recomputed; where both files carry a value (the first draw's mid-rank and
  category) they must agree, or the figure is refused.

Panels (design: Figures, F3; strip plots, one axis each):

1. The example item: the summary value of each parsed draw (the defensive
   spread, or the outlook on the negative control) with the median and the
   interquartile band of the parsed draws, the first draw marked with a
   distinct marker and a label, and the ensemble results as stored: the
   consensus (``ensemble_none``) and the adjusted ensemble decisions.
2. The same item: ``p_memorized`` of each draw per estimate source the arm
   computes, with median and band, the first draw marked, and each ensemble
   variant's ``p_memorized`` as stored.
3. Every analysed item: the first draw's mid-rank among its ensemble's parsed
   draws against the frozen band (bounds inclusive), one row per rendering,
   marker area by the count printed beside it; items whose half-ensemble
   decisions 7.11 lists as unresolved sit in a row of their own.
4. The stored first-draw rate (the share of items whose first draw is off)
   with its interval, overall and per rendering, against the exchangeability
   baseline: the share expected off if the first draw were exchangeable with
   the other draws of its ensemble (per item, the share of its parsed draws
   that would be off in the first draw's place, averaged over the items). With
   8 distinct values about half of the mid-ranks fall outside [0.25, 0.75], so
   the reference is not zero.

The example item is the first item in plan order with a decided first draw and
a consensus summary value unless one is given (the plan freezes no example
item). Outputs, style and the write-once rule are
those of :mod:`scripts.finance_sim.figures`: ``draws-light.svg``,
``draws-dark.svg``, ``draws.csv`` and ``draws.md``; derived statistics only.
Matplotlib is imported inside the functions that draw.
"""

from __future__ import annotations

import argparse
import math
import sys
import textwrap
from collections import Counter
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
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
from scripts.finance_sim.figures import (
    _HEX64,
    _RC,
    _TITLE_LINE_IN,
    _WIDTH_IN,
    ADJUSTMENT_NAMES,
    DERIVED_ONLY,
    DRAWS,
    EDGAR_SOURCE,
    EXIT_REFUSED,
    EXIT_WRITTEN,
    FIGURES_DIR,
    FMP_STUDY_SOURCE,
    FRED_SOURCE,
    ROLE_PURPOSE,
    FigureError,
    FigureTable,
    _around,
    _count,
    _flat,
    _number,
    _obstacles,
    _percent,
    _place_text,
    _read_json,
    _real,
    _style_axes,
    _text,
    _theme_rc,
    figure_texts,
    write_outputs,
)
from scripts.finance_sim.limits import ARMS, SAMPLING_ARM
from scripts.finance_sim.pilot import CONFIRMATORY_STAGE
from scripts.finance_sim.plan import ESTIMATE_SOURCES, NO_ADJUSTMENT, SINGLE_DRAW_INDEX, StudyPlan, load_plan

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from recall_guard.harness.plots import FigureTheme

#: The arm F3 draws by default: the probe arm samples at temperature 0.7, the serving arm's draws agree by design.
DEFAULT_DRAWS_ARM = SAMPLING_ARM
#: Parsed draws the consensus split check needs (``EnsembleSpec.min_cluster_draws``); fewer cannot be unresolved.
SPLIT_CHECK_MIN_DRAWS: int = next(item.default for item in fields(EnsembleSpec) if item.name == "min_cluster_draws")
#: What the summary value of each item kind is, for the axis and the caption.
SUMMARY_NAMES: Mapping[str, str] = {
    MACRO: "defensive spread ((gold + cash) minus (equity + tech) tilt of the loadings)",
    "edgar": "outlook",
}
DRAWS_COLUMNS: tuple[str, ...] = (
    "row", "arm", "role", "stage", "item_id", "rendering", "quantity", "draw_index", "value", "first_draw", "median",
    "q25", "q75", "variant", "status", "first_category", "consensus_category", "first_draw_rank", "outside_band",
    "category_differs", "off", "halves", "baseline_off", "baseline_outside", "group", "n", "count_off",
    "count_outside", "count_differs", "count_no_category", "rate", "interval_low", "interval_high", "reason")
IN_HALVES = "in the half-ensemble statistic"
EXAMPLE_DEFAULT = ("the first item in plan order with a decided first draw and a consensus summary value (the plan "
                   "freezes no example item)")
_ROW_IN, _PANEL_IN, _FRAME_IN = 0.27, 0.42, 1.0
#: Panel 3 rows are taller: a first draw whose category differs sits above the others of its rank.
_RANK_ROW_IN, _DIFFERS_OFFSET = 0.46, 0.34
_DRAW_PT, _FIRST_PT, _ENSEMBLE_PT, _EDGE_PT = 5.5, 8.0, 7.0, 1.2
_AREA_MIN, _AREA_MAX = 30.0, 150.0
#: A stack keeps within this many rows of its row's centre, its markers at least one diameter apart (10 percent
#: spare); a group that does not fit is one open marker with its count, the first draw ``_APART`` rows above it.
_SPREAD = 0.3
_STACK_STEP = 1.1 * (_DRAW_PT + _EDGE_PT) / (_ROW_IN * 72.0)
_APART = 0.2
#: Draws closer than this share of the axis span form one group (above one marker width on any panel width).
_TIE_FRACTION = 0.026
_LABEL_PT = 6.5
_TITLE_CHARS = 74


# ---------- plain data ----------

@dataclass(frozen=True)
class DrawStrip:
    """One quantity of the example item, per draw index (None for an unparsed draw or a missing estimate).

    ``adjustment`` picks the marker (no adjustment for the summary value, the
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
    """Everything F3 shows, read from the stored results and metrics of one arm and stage."""

    plan_sha256: str
    arm: str
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
    return "all items" if group == "all" else f"{group} rendering"


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
    """The per-draw values of one stored result and its stored ensemble results (panels 1 and 2)."""
    if not any(draw.parsed is not None for draw in result.recorded_draws):
        raise FigureError(f"the example item {result.key} has no parsed draw to show")
    summary_name = "summary value"
    summary = _strip(result, summary_name, "each draw", NO_ADJUSTMENT)
    consensus = result.variant(CONSENSUS_VARIANT)
    distribution = summarize_distribution(result.draws_for(NO_ADJUSTMENT), DEFINITIONS[result.kind])
    rank = None if distribution.summary_value is None else distribution.summary_value.first_draw_rank
    if consensus.first_draw_rank is not None and rank is not None and not math.isclose(
            consensus.first_draw_rank, rank, rel_tol=0.0, abs_tol=1e-12):
        raise FigureError(f"the stored {CONSENSUS_VARIANT} first-draw rank of {result.key} does not match its draws")
    marks = [_mark(result, NO_ADJUSTMENT, summary_name, "ensemble, no adjustment (consensus)", "summary_value")]
    marks += [_mark(result, source, summary_name, f"ensemble, {ADJUSTMENT_NAMES[source]} adjusted",
                    "adjusted_summary_value") for source in ESTIMATE_SOURCES]
    estimates, estimate_marks = [], []
    for source in ESTIMATE_SOURCES:
        name = ADJUSTMENT_NAMES[source]
        if source in result.identities:
            estimates.append(_strip(result, f"p_memorized {source}", f"{name}, each draw", source))
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
            status, unresolved = halves.get(item_key, ("not listed in the half-ensemble statistic", None))
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
            plan_sha256=plan.plan_sha256, arm=arm, role=_text(arm_info["role"], "role"), stage=stage, kind=kind,
            model=_text(arm_info["model"], "model"), control=_flat(metrics["control"]),
            ensemble_draws=_count(settings["draws"], "draws"),
            half_draws=None if settings["half_draws"] is None else _count(settings["half_draws"], "half_draws"),
            band=band, confidence=float(_real(settings["confidence"], "confidence")),
            interval_method=_flat(interval["method"]) if interval is not None else "no interval was computed",
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


# ---------- table and caption ----------

def _cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool | int | float):
        return _number(value)
    return _flat(value)


def _row(**values: object) -> tuple[str, ...]:
    return tuple(_cell(values.get(column)) for column in DRAWS_COLUMNS)


def _strips(example: ExampleItem) -> tuple[DrawStrip, ...]:
    return (example.summary, *example.estimates)


def draws_table(data: DrawsFigure) -> FigureTable:
    """F3's table (draws and band of the example item, ensemble values, items, missing items, rates) and caption."""
    if not isinstance(data, DrawsFigure):
        raise TypeError(f"data must be a DrawsFigure; got {type(data).__name__}")
    head = {"arm": data.arm, "role": data.role, "stage": data.stage}
    example = data.example
    where = {**head, "item_id": example.item_id, "rendering": example.rendering}
    rows = []
    for strip in _strips(example):
        for index, value in enumerate(strip.values):
            rows.append(_row(row="draw", **where, quantity=strip.quantity, draw_index=index, value=value,
                             first_draw=index == SINGLE_DRAW_INDEX, reason=strip.reasons[index]))
        rows.append(_row(row="spread", **where, quantity=strip.quantity, median=strip.median, q25=strip.q25,
                         q75=strip.q75, n=len(strip.present),
                         first_draw_rank=example.first_rank if strip is example.summary else None))
    for mark in (*example.summary_marks, *example.estimate_marks):
        rows.append(_row(row="ensemble", **where, quantity=mark.quantity, variant=mark.variant, value=mark.value,
                         status=mark.status, reason=mark.reason,
                         first_category=example.first_category if mark.variant == CONSENSUS_VARIANT else None,
                         consensus_category=example.consensus_category if mark.variant == CONSENSUS_VARIANT
                         else None))
    for item in data.items:
        rows.append(_row(row="item", **head, item_id=item.item_id, rendering=item.rendering,
                         first_category=item.first_category, consensus_category=item.consensus_category,
                         first_draw_rank=item.first_draw_rank, outside_band=item.outside_band,
                         category_differs=item.category_differs, off=item.off, halves=item.halves,
                         baseline_off=item.baseline_off, baseline_outside=item.baseline_outside, n=item.parsed))
    for item_id, rendering, reason in data.missing_items:
        rows.append(_row(row="missing_item", **head, item_id=item_id, rendering=rendering, reason=reason))
    for rate in data.rates:
        rows.append(_row(row="rate", **head, group=rate.group, n=rate.n, count_off=rate.off,
                         count_outside=rate.outside_band, count_differs=rate.category_differs,
                         count_no_category=rate.no_category, rate=rate.rate,
                         interval_low=rate.low, interval_high=rate.high, baseline_off=rate.baseline_off,
                         baseline_outside=rate.baseline_outside, reason=rate.missing))
    return FigureTable(name=DRAWS, caption=_draws_caption(data), columns=DRAWS_COLUMNS, rows=tuple(rows))


def _not_drawn(marks: Iterable[EnsembleMark]) -> list[tuple[str, str]]:
    return [(mark.variant, _flat(mark.reason)) for mark in marks if mark.value is None and mark.reason]


def _draws_caption(data: DrawsFigure) -> str:
    example = data.example
    lower, upper = data.band
    by_rendering = "; ".join(f"{rate.label}: {rate.n}" for rate in data.rates[1:])
    unresolved = sum(1 for item in data.items if item.halves_unresolved)
    chosen = EXAMPLE_DEFAULT if example.chosen == EXAMPLE_DEFAULT else "chosen for the figure"
    parts = [
        "Figure F3. Per-draw distribution of the decision and of the memorization estimate against the ensemble "
        "result, and where the first draw falls.",
        f"Data: the stored item results (task 7.7) of the {data.stage} stage ({data.control}) of the {data.arm} arm "
        f"({data.role} role, {ROLE_PURPOSE.get(data.role, data.role)}), model {data.model}, plan "
        f"{data.plan_sha256[:12]}, and the stored ensemble metrics of task 7.11 (read, not recomputed); pilot items "
        "excluded.",
        f"Panels 1 and 2 show one example item, {example.item_id} ({example.rendering} rendering; {chosen}): the "
        f"{SUMMARY_NAMES[data.kind]} of each parsed draw and its p_memorized per estimate source, with the ensemble "
        "decisions as stored: the consensus (ensemble_none) and the ensemble decisions after each adjustment, and "
        "each ensemble variant's p_memorized.",
        f"Panels 3 and 4 cover every analysed item: the first draw's mid-rank among its ensemble's parsed summary "
        f"values (the share strictly below plus half the share tied, the first draw included) against the frozen "
        f"band {lower:g} to {upper:g} (bounds inclusive), and the stored first-draw rate: the share of items whose "
        "first draw is off, its category differing from the consensus or its mid-rank outside the band.",
        f"Sample size: the example item has {example.draws} draws, {example.parsed} parsed and {example.failed} "
        f"failed; {len(data.items)} items have a decided first draw ({by_rendering}) and {len(data.missing_items)} "
        "are missing with their reasons in the table.",
        "Interval: none on the per-draw values; the shaded band of panels 1 and 2 is the interquartile range of the "
        "parsed draws (25th to 75th percentile, linear interpolation, as recall_guard.core.distribution summarises "
        f"them) with the median marked; the first-draw rate carries the stored {data.interval_method} at "
        f"confidence {data.confidence:g}.",
        "Reference: the exchangeability baseline is the share of items expected off if the first draw were "
        "exchangeable with the other draws of its ensemble, under the same inclusion rule as the stored rate: per "
        "item with a consensus category, the share of its parsed draws that would be off in the first draw's place "
        "(computed from the stored per-draw values; with eight distinct values about half of the mid-ranks lie "
        "outside the band, so the reference is not zero); an item whose consensus has no category enters the rate "
        "only when its first draw lies outside the band, and is then always off, so it counts 1; the baseline "
        f"averages these over the items. Items without a consensus category: {data.rates[0].no_category} of "
        f"{data.rates[0].n} (included only when outside the band).",
    ]
    halves = (f"Items whose half-ensemble decisions task 7.11 lists as unresolved (missing from the half-ensemble "
              f"variance with that reason) sit in a row of their own: {unresolved} items.")
    if data.half_draws is not None and data.half_draws < SPLIT_CHECK_MIN_DRAWS:
        halves += (f" The consensus split check needs at least {SPLIT_CHECK_MIN_DRAWS} parsed draws, so halves of "
                   f"{data.half_draws} draws are never assessed; an unmarked item is not shown as resolved.")
    else:
        halves += " An unmarked item is not shown as resolved: its halves were not listed as unresolved."
    if data.halves_note:
        halves += f" Note: {data.halves_note}."
    parts.append(halves)
    parts.append("Encoding: an open marker is one draw (circle the summary value, square top-k, triangle full "
                 "distribution), or, with a count such as ×8 beside it, that many draws too close together to "
                 "stack apart (the first draw then sits above it), the filled diamond the first draw, a filled "
                 "marker an ensemble result; rows are "
                 "labelled; in panel 3 a filled marker lies outside the band, a diamond marks a first draw whose "
                 "category differs from the consensus, and marker area grows with the count printed beside it; "
                 "the vertical muted lines mark zero (panel 1), the band bounds (panel 3) and the baseline "
                 "(panel 4).")
    not_drawn = _not_drawn((*example.summary_marks, *example.estimate_marks))
    if not_drawn:
        parts.append("Not drawn: " + "; ".join(f"{variant}: {reason}" for variant, reason in not_drawn) + ".")
    sources = [FRED_SOURCE if data.kind == MACRO else EDGAR_SOURCE, FMP_STUDY_SOURCE]
    parts += [f"Sources: {'; '.join(sources)}.", DERIVED_ONLY]
    return " ".join(parts)


# ---------- drawing ----------

@dataclass(frozen=True)
class _StripLayout:
    """Where one strip's draws go, as (x, row offset): open markers, collapsed stacks with counts, the first draw."""

    opens: tuple[tuple[float, float], ...]
    stacks: tuple[tuple[float, float, int], ...]
    first: tuple[float, float] | None


def _close_groups(present: Sequence[tuple[int, float]], tolerance: float) -> list[list[tuple[int, float]]]:
    """Draws grouped in value order, a draw joining its neighbour's group when the gap is at most ``tolerance``."""
    groups: list[list[tuple[int, float]]] = []
    for index, value in sorted(present, key=lambda pair: (pair[1], pair[0])):
        if groups and value - groups[-1][-1][1] <= tolerance:
            groups[-1].append((index, value))
        else:
            groups.append([(index, value)])
    return [sorted(group) for group in groups]


def strip_layout(present: Sequence[tuple[int, float]], tolerance: float) -> _StripLayout:
    """Marker positions of one strip: no two open markers touch, and the first draw stays distinct.

    Draws closer than ``tolerance`` form a group. A group stacks vertically,
    centred on the row in draw-index order, while the step between markers
    is at least one marker diameter (``_STACK_STEP``); a larger group is
    drawn as one open marker at its lower median value with the count of its
    draws (``stacks``), and its first draw, if it holds it, sits apart above
    it as the distinct first-draw marker.
    """
    opens: list[tuple[float, float]] = []
    stacks: list[tuple[float, float, int]] = []
    first: tuple[float, float] | None = None
    for group in _close_groups(present, tolerance):
        n = len(group)
        if n == 1 or 2 * _SPREAD / (n - 1) >= _STACK_STEP:
            for position, (index, value) in enumerate(group):
                offset = (position - (n - 1) / 2.0) * _STACK_STEP
                if index == SINGLE_DRAW_INDEX:
                    first = (value, offset)
                else:
                    opens.append((value, offset))
            continue
        others = [value for index, value in group if index != SINGLE_DRAW_INDEX]
        held = len(others) < n
        if held:
            first = (group[0][1], -_APART)
        stacks.append((sorted(others)[(len(others) - 1) // 2], _APART if held else 0.0, len(others)))
    return _StripLayout(tuple(opens), tuple(stacks), first)


def _limits(values: Iterable[float | None], *, include: Iterable[float] = (), minimum: float = 1.0,
            clip: tuple[float, float] | None = None) -> tuple[float, float]:
    """Axis limits around ``values`` and ``include``: 8 percent padding, at least ``minimum`` wide, within ``clip``."""
    found = [value for value in values if value is not None] + list(include)
    low, high = min(found), max(found)
    pad = 0.08 * (high - low)
    low, high = low - pad, high + pad
    if high - low < minimum:
        centre = (low + high) / 2.0
        low, high = centre - minimum / 2.0, centre + minimum / 2.0
    if clip is not None:
        if low < clip[0]:
            low, high = clip[0], max(high, clip[0] + minimum)
        if high > clip[1]:
            low, high = min(low, clip[1] - minimum), clip[1]
    return low, high


def _wrapped(lines: Iterable[str]) -> str:
    """Title lines wrapped to the panel width (the row labels take the left of the figure)."""
    return "\n".join(textwrap.fill(line, width=_TITLE_CHARS, break_on_hyphens=False) for line in lines)


def _draw_strip(axes: Axes, strip: DrawStrip, row: float, style: FigureTheme, theme: str,
                key: str) -> list[tuple[float, float, int]]:
    """Draw one strip (band, median, draws, first draw); returns the collapsed stacks to label after layout."""
    from matplotlib.patches import Rectangle

    from recall_guard.harness.plots import variant_style

    variant = variant_style(strip.adjustment, "single", theme=theme)
    if strip.q25 is not None and strip.q75 is not None and strip.q75 > strip.q25:
        axes.add_patch(Rectangle((strip.q25, row - 0.4), strip.q75 - strip.q25, 0.8, facecolor=style.gridline,
                                 edgecolor="none", zorder=0.5, gid=f"draws-iqr-{key}"))
    if strip.median is not None:
        axes.plot([strip.median, strip.median], [row - 0.4, row + 0.4], color=style.secondary_ink, linewidth=1.2,
                  linestyle="-", marker="none", zorder=1.5, gid=f"draws-median-{key}")
    low, high = axes.get_xlim()
    layout = strip_layout(strip.present, _TIE_FRACTION * (high - low))
    marker = {name: value for name, value in variant.marker_kwargs().items() if name != "label"}
    for gid, points in ((f"draws-draw-{key}", layout.opens), (f"draws-stack-{key}", layout.stacks)):
        if points:
            axes.plot([point[0] for point in points], [row + point[1] for point in points], markersize=_DRAW_PT,
                      markeredgewidth=_EDGE_PT, zorder=3, gid=gid, **marker)
    if layout.first is not None:
        axes.plot([layout.first[0]], [row + layout.first[1]], marker="D", linestyle="none",
                  markersize=_FIRST_PT * 0.75, color=style.ink, markeredgecolor=style.surface,
                  markerfacecolor=style.ink, markeredgewidth=0.8, zorder=4, gid=f"draws-first-{key}")
    return [(x, row + offset, count) for x, offset, count in layout.stacks]


def _draw_mark(axes: Axes, mark: EnsembleMark, row: float, theme: str, key: str) -> None:
    from recall_guard.harness.plots import variant_style

    if mark.value is None:
        return
    variant = variant_style(mark.adjustment, "ensemble", theme=theme)
    marker = {name: value for name, value in variant.marker_kwargs().items() if name != "label"}
    axes.plot([mark.value], [row], markersize=_ENSEMBLE_PT, markeredgewidth=_EDGE_PT, zorder=3,
              gid=f"draws-ensemble-{key}-{mark.variant}", **marker)


def _rows(axes: Axes, labels: Sequence[str]) -> None:
    axes.set_yticks(range(len(labels)), labels=list(labels))
    axes.set_ylim(len(labels) - 0.5, -0.5)
    axes.grid(axis="y", visible=False)
    axes.spines["left"].set_visible(False)
    axes.tick_params(axis="y", length=0)


def _example_title(data: DrawsFigure) -> str:
    example = data.example
    strip = example.summary
    if example.summary.first is None:
        first = "the first draw did not parse (no first-draw marker)"
    else:
        rank = "missing" if example.first_rank is None else f"{example.first_rank:.3g}"
        first = f"first-draw mid-rank {rank}"
    lines = [f"Example item {example.item_id}, {example.rendering} rendering ({data.arm} arm, {data.stage}): "
             f"{SUMMARY_NAMES[data.kind].split(' (')[0]} of each draw",
             f"{example.parsed} of {example.draws} draws parsed; median {_number(strip.median) or 'missing'}, "
             f"interquartile range {_number(strip.q25) or 'missing'} to {_number(strip.q75) or 'missing'}; {first}"]
    absent = [mark for mark in example.summary_marks if mark.value is None]
    if absent:
        lines.append(f"Not drawn: {', '.join(mark.variant for mark in absent)} (see the caption)")
    return _wrapped(lines)


def _estimate_title(data: DrawsFigure) -> str:
    example = data.example
    lines = [f"p_memorized of each draw of {example.item_id}, per estimate source"]
    for strip in example.estimates:
        lines.append(f"{ADJUSTMENT_NAMES[strip.adjustment]}: {len(strip.present)} of {example.draws} draws with an "
                     f"estimate, median {_number(strip.median) or 'missing'}")
    absent = [mark for mark in example.estimate_marks if mark.value is None]
    if absent:
        lines.append(f"Not drawn: {', '.join(mark.variant for mark in absent)} (see the caption)")
    return _wrapped(lines)


def _rank_rows(data: DrawsFigure) -> list[tuple[str, str, bool]]:
    rows = []
    for rendering in data.renderings:
        rows.append((rendering, f"{rendering}", False))
        if any(item.halves_unresolved and item.rendering == rendering for item in data.items):
            rows.append((rendering, f"{rendering}, halves unresolved", True))
    return rows


def _rank_groups(data: DrawsFigure) -> list[tuple[int, float, bool, bool, int]]:
    """``(row, mid-rank, category differs, outside, count)`` per marker of panel 3, in a stable order."""
    rows = _rank_rows(data)
    counts: Counter = Counter()
    for item in data.items:
        if item.first_draw_rank is None:
            continue
        unresolved = bool(item.halves_unresolved)
        row = rows.index(next(entry for entry in rows if entry[0] == item.rendering and entry[2] == unresolved))
        counts[(row, item.first_draw_rank, bool(item.category_differs), bool(item.outside_band))] += 1
    return [(*key, count) for key, count in sorted(counts.items())]


def _area(count: int, largest: int) -> float:
    return _AREA_MIN + (_AREA_MAX - _AREA_MIN) * (count - 1) / max(largest - 1, 1)


def _draw_ranks(axes: Axes, data: DrawsFigure, style: FigureTheme, theme: str) -> list[tuple[float, float, int,
                                                                                                float]]:
    from matplotlib.patches import Rectangle

    from recall_guard.harness.plots import figure_theme

    palette = figure_theme(theme).palette
    rows = _rank_rows(data)
    lower, upper = data.band
    top, bottom = -0.5 - _DIFFERS_OFFSET / 2.0, len(rows) - 0.5
    axes.add_patch(Rectangle((lower, top), upper - lower, bottom - top, facecolor=style.gridline, edgecolor="none",
                             zorder=0.5, gid="draws-band"))
    for bound, name in ((lower, "lower"), (upper, "upper")):
        axes.plot([bound, bound], [top, bottom], color=style.muted, linewidth=1.0, linestyle="-",
                  marker="none", zorder=1, gid=f"draws-band-{name}")
    groups = _rank_groups(data)
    largest = max((group[4] for group in groups), default=1)
    placed = []
    for row, rank, differs, outside, count in groups:
        y = row - (_DIFFERS_OFFSET if differs else 0.0)
        axes.scatter([rank], [y], s=_area(count, largest), marker="D" if differs else "o",
                     facecolors=palette[0] if outside else style.surface, edgecolors=palette[0], linewidths=1.3,
                     zorder=3, clip_on=False,
                     gid=f"draws-rank-{row}-{rank:.6f}-{'differs' if differs else 'same'}")
        placed.append((rank, y, count, _area(count, largest)))
    _rows(axes, [label for _, label, _ in rows])
    axes.set_ylim(bottom, top)
    axes.set_xlim(-0.03, 1.03)
    axes.set_xticks([0.0, lower, 0.5, upper, 1.0], labels=["0", f"{lower:g}", "0.5", f"{upper:g}", "1"])
    return placed


def _rank_title(data: DrawsFigure) -> str:
    overall = data.rates[0]
    lower, upper = data.band
    return _wrapped([f"First-draw mid-rank among each ensemble's parsed draws: {len(data.items)} items",
                     f"Shaded: the frozen band {lower:g} to {upper:g} (bounds inclusive); outside it: "
                     f"{overall.outside_band} of {overall.n}; category differs from the consensus: "
                     f"{overall.category_differs}"])


def _plotted_rates(data: DrawsFigure) -> tuple[RateRow, ...]:
    """The rate rows panel 4 draws: overall and per rendering, or overall only for a single rendering."""
    return data.rates if len(data.renderings) > 1 else data.rates[:1]


def _draw_rates(axes: Axes, data: DrawsFigure, style: FigureTheme, theme: str) -> None:
    from recall_guard.harness.plots import figure_theme

    colour = figure_theme(theme).palette[0]
    for row, rate in enumerate(_plotted_rates(data)):
        if rate.low is not None and rate.high is not None:
            axes.plot([rate.low, rate.high], [row, row], color=colour, linewidth=1.8, linestyle="-", marker="none",
                      zorder=2, gid=f"draws-rate-interval-{rate.group}")
        if rate.rate is not None:
            axes.plot([rate.rate], [row], marker="o", linestyle="none", markersize=_ENSEMBLE_PT, color=colour,
                      markerfacecolor=colour, markeredgecolor=colour, markeredgewidth=_EDGE_PT, zorder=3,
                      gid=f"draws-rate-{rate.group}")
        if rate.baseline_off is not None:
            axes.plot([rate.baseline_off, rate.baseline_off], [row - 0.38, row + 0.38], color=style.muted,
                      linewidth=2.0, linestyle="-", marker="none", zorder=1.5, gid=f"draws-baseline-{rate.group}")
    _rows(axes, [f"{rate.label} (n {rate.n})" for rate in _plotted_rates(data)])
    axes.set_xlim(-0.03, 1.03)


def _rate_title(data: DrawsFigure) -> str:
    overall = data.rates[0]
    rate = "missing" if overall.rate is None else f"{overall.rate:.3g}"
    baseline = "missing" if overall.baseline_off is None else f"{overall.baseline_off:.3g}"
    return _wrapped([f"Share of items whose first draw is off: {rate} (all items); exchangeability baseline "
                     f"{baseline}",
                     f"Dot and line: the stored rate with its {_percent(data.confidence)} percent Wilson interval "
                     "(two-sided); muted tick: the baseline under the rate's inclusion rule",
                     f"{overall.no_category} of {overall.n} items have no consensus category: included only when "
                     "outside the band, so always off (counted 1 in the baseline)"])


def _place_draw_labels(axes: Axes, keys: Sequence[tuple[str, float, float]],
                       stacks: Sequence[tuple[str, float, float, int]], style: FigureTheme, renderer: Any) -> None:
    """Label each first-draw marker ``first draw`` and each collapsed stack with its count (``×n``).

    Every label is placed clear of lines, markers, the labels placed before
    it and the frame (:func:`~scripts.finance_sim.figures._place_text`).
    """
    scale = axes.figure.dpi / 72.0
    paths, boxes = _obstacles(axes)

    def rings(anchor: tuple[float, float], offset: float) -> list[Any]:
        # Nearest first; the farther rings clear neighbouring draws in a crowded row.
        return [placement for step in (0.0, 7.0, 14.0) for placement in _around(anchor, offset + step)]

    for key, x, y in keys:
        label = _place_text(axes, "first draw", rings((x, y), _FIRST_PT * 0.75 / 2.0 + 2.5), renderer, paths=paths,
                            boxes=boxes, fontsize=_LABEL_PT, color=style.ink, zorder=3.5,
                            gid=f"draws-first-label-{key}",
                            bbox={"facecolor": style.surface, "edgecolor": "none", "pad": 1.0})
        boxes.append(label.get_window_extent(renderer).padded(scale))
    for index, (key, x, y, count) in enumerate(stacks):
        label = _place_text(axes, f"×{count}", rings((x, y), _DRAW_PT / 2.0 + 2.0), renderer, paths=paths,
                            boxes=boxes, fontsize=_LABEL_PT, color=style.secondary_ink, zorder=3.5,
                            gid=f"draws-stack-label-{key}-{index}",
                            bbox={"facecolor": style.surface, "edgecolor": "none", "pad": 0.8})
        boxes.append(label.get_window_extent(renderer).padded(scale))


def _place_count_labels(axes: Axes, placed: Sequence[tuple[float, float, int, float]], style: FigureTheme,
                        renderer: Any) -> None:
    scale = axes.figure.dpi / 72.0
    paths, boxes = _obstacles(axes)
    for index, (x, y, count, area) in enumerate(placed):
        label = _place_text(axes, str(count), _around((x, y), math.sqrt(area) / 2.0 + 2.0), renderer, paths=paths,
                            boxes=boxes, fontsize=_LABEL_PT, color=style.secondary_ink, zorder=2.5,
                            gid=f"draws-count-{index}",
                            bbox={"facecolor": style.surface, "edgecolor": "none", "pad": 0.8})
        boxes.append(label.get_window_extent(renderer).padded(scale))


def _place_baseline_label(axes: Axes, data: DrawsFigure, style: FigureTheme, renderer: Any) -> None:
    rows = [(row, rate) for row, rate in enumerate(_plotted_rates(data)) if rate.baseline_off is not None]
    if not rows:
        return
    paths, boxes = _obstacles(axes)
    candidates = [placement for row, rate in rows for placement in _around((rate.baseline_off, row), 4.0)]
    # Between two rows, beside the upper tick's end: clear of both rows' intervals when those leave no room.
    candidates += [placement for row, rate in rows[:-1] for placement in (
        ((rate.baseline_off, row + 0.5), 4.0, 0.0, "left", "center"),
        ((rate.baseline_off, row + 0.5), -4.0, 0.0, "right", "center"))]
    _place_text(axes, "exchangeability baseline", candidates, renderer, paths=paths, boxes=boxes, fontsize=_LABEL_PT,
                color=style.secondary_ink, zorder=2.5, gid="draws-baseline-label",
                bbox={"facecolor": style.surface, "edgecolor": "none", "pad": 1.0})


def _legend(figure: Figure, style: FigureTheme, theme: str) -> None:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    from recall_guard.harness.plots import figure_theme, variant_style

    colour = figure_theme(theme).palette[0]
    draw = {name: value for name, value in variant_style(NO_ADJUSTMENT, "single", theme=theme).marker_kwargs().items()
            if name != "label"}
    handles = [
        Line2D([], [], markersize=_DRAW_PT, markeredgewidth=_EDGE_PT, label="one parsed draw", **draw),
        Line2D([], [], marker="D", linestyle="none", markersize=_FIRST_PT * 0.75, color=style.ink,
               markerfacecolor=style.ink, markeredgecolor=style.surface, label="the first draw (index 0)"),
        Patch(facecolor=style.gridline, edgecolor="none", label="interquartile range of the parsed draws"),
        Line2D([], [], color=style.secondary_ink, linewidth=1.2, marker="none", label="median of the parsed draws"),
        Line2D([], [], marker="o", linestyle="none", markersize=6, color=colour, markerfacecolor=style.surface,
               markeredgecolor=colour, markeredgewidth=1.3, label="first draw inside the band"),
        Line2D([], [], marker="o", linestyle="none", markersize=6, color=colour, markerfacecolor=colour,
               markeredgecolor=colour, label="first draw outside the band"),
        Line2D([], [], marker="D", linestyle="none", markersize=5.5, color=colour, markerfacecolor=style.surface,
               markeredgecolor=colour, markeredgewidth=1.3, label="category differs from the consensus"),
        Line2D([], [], color=colour, linewidth=1.8, marker="o", markersize=5, label="stored rate and interval"),
        Line2D([], [], color=style.muted, linewidth=2.0, marker="none", label="exchangeability baseline"),
    ]
    figure.legend(handles=handles, loc="outside lower center", ncols=3, frameon=False)


def plot_draws(data: DrawsFigure, *, theme: str = "light") -> Figure:
    """F3: per-draw strips of the example item, first-draw mid-ranks and the first-draw rate (Requirement 10.4).

    ``data`` is plain data read from the stored outputs (:func:`draws_figure`);
    ``theme`` is ``light`` or ``dark``. Returns a matplotlib Figure (no pyplot,
    no global state); labels are placed after the layout is fixed.
    """
    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from recall_guard.harness.plots import figure_theme

    if not isinstance(data, DrawsFigure):
        raise TypeError(f"data must be a DrawsFigure; got {type(data).__name__}")
    style = figure_theme(theme)
    example = data.example
    summary_marks = [mark for mark in example.summary_marks if mark.value is not None]
    summary_labels = [f"each draw ({example.parsed} of {example.draws} parsed)", *[mark.label for mark in summary_marks]]
    estimate_rows: list[tuple[str, DrawStrip | EnsembleMark]] = []
    for source in ESTIMATE_SOURCES:
        strip = next((strip for strip in example.estimates if strip.adjustment == source), None)
        if strip is not None:
            estimate_rows.append((f"{strip.label} ({len(strip.present)} with an estimate)", strip))
        mark = next(mark for mark in example.estimate_marks if mark.adjustment == source)
        if mark.value is not None:
            estimate_rows.append((mark.label, mark))
    titles = [_example_title(data), _estimate_title(data), _rank_title(data), _rate_title(data)]
    rows = [len(summary_labels) * _ROW_IN, max(len(estimate_rows), 1) * _ROW_IN,
            len(_rank_rows(data)) * _RANK_ROW_IN, len(_plotted_rates(data)) * _ROW_IN]
    spans = [height + _PANEL_IN + _TITLE_LINE_IN * (title.count("\n") + 1)
             for height, title in zip(rows, titles, strict=True)]
    with matplotlib.rc_context(_theme_rc(theme)):
        figure = Figure(figsize=(_WIDTH_IN, sum(spans) + _FRAME_IN), layout="constrained")
        panels = figure.subplots(4, 1, gridspec_kw={"height_ratios": spans})
        for axes in panels:
            _style_axes(axes, style)
        summary_axes, estimate_axes, rank_axes, rate_axes = panels
        firsts: dict[Axes, list[tuple[str, float, float]]] = {summary_axes: [], estimate_axes: []}
        stacks: dict[Axes, list[tuple[str, float, float, int]]] = {summary_axes: [], estimate_axes: []}
        # Panel 1: the summary value of each draw, then the ensemble results.
        summary_axes.set_xlim(*_limits([*example.summary.values, *(mark.value for mark in summary_marks)],
                                       include=(0.0,)))
        summary_axes.axvline(0.0, color=style.muted, linewidth=1.0, zorder=1, gid="draws-zero")
        stacks[summary_axes] += [("summary", *stack)
                                 for stack in _draw_strip(summary_axes, example.summary, 0.0, style, theme, "summary")]
        for row, mark in enumerate(summary_marks, start=1):
            _draw_mark(summary_axes, mark, float(row), theme, "summary")
        _rows(summary_axes, summary_labels)
        summary_axes.set_title(titles[0], loc="left", color=style.ink)
        summary_axes.set_xlabel(SUMMARY_NAMES[data.kind].capitalize(), fontsize=_RC["axes.labelsize"])
        # Panel 2: p_memorized of each draw per estimate source, then each ensemble variant's; the axis spans the
        # values shown (at least 0.1 wide, within [0, 1]) so that close estimates stay apart.
        shown = [value for _, entry in estimate_rows for value in (
            entry.values if isinstance(entry, DrawStrip) else (entry.value,))]
        estimate_axes.set_xlim(*(_limits(shown, minimum=0.1, clip=(-0.02, 1.02)) if any(
            value is not None for value in shown) else (-0.02, 1.02)))
        labels = []
        for row, (label, entry) in enumerate(estimate_rows):
            labels.append(label)
            if isinstance(entry, DrawStrip):
                key = f"estimate-{entry.adjustment}"
                stacks[estimate_axes] += [(key, *stack) for stack in _draw_strip(estimate_axes, entry, float(row),
                                                                                 style, theme, key)]
            else:
                _draw_mark(estimate_axes, entry, float(row), theme, "estimate")
        _rows(estimate_axes, labels or ["no estimate source"])
        estimate_axes.set_title(titles[1], loc="left", color=style.ink)
        estimate_axes.set_xlabel("p_memorized of the draw (top-k or full-distribution calibrator)",
                                 fontsize=_RC["axes.labelsize"])
        # Panels 3 and 4: every analysed item.
        placed = _draw_ranks(rank_axes, data, style, theme)
        rank_axes.set_title(titles[2], loc="left", color=style.ink)
        rank_axes.set_xlabel("Mid-rank of the first draw among the parsed draws", fontsize=_RC["axes.labelsize"])
        _draw_rates(rate_axes, data, style, theme)
        rate_axes.set_title(titles[3], loc="left", color=style.ink)
        rate_axes.set_xlabel("Share of items whose first draw is off", fontsize=_RC["axes.labelsize"])
        _legend(figure, style, theme)
        figure.suptitle("Per-draw distribution against the ensemble result, and where the first draw falls", x=0.01,
                        ha="left")
        renderer = FigureCanvasAgg(figure).get_renderer()
        figure.draw_without_rendering()
        for axes in (summary_axes, estimate_axes):
            for line in axes.lines:
                gid = line.get_gid() or ""
                if gid.startswith("draws-first-"):
                    (x,), (y,) = line.get_xdata(), line.get_ydata()
                    firsts[axes].append((gid.removeprefix("draws-first-"), float(x), float(y)))
            _place_draw_labels(axes, firsts[axes], stacks[axes], style, renderer)
        _place_count_labels(rank_axes, placed, style, renderer)
        _place_baseline_label(rate_axes, data, style, renderer)
        figure.draw_without_rendering()
    return figure


# ---------- outputs ----------

def render_draws(plan: StudyPlan, *, arm: str = DEFAULT_DRAWS_ARM, stage: str = CONFIRMATORY_STAGE,
                 analysis_dir: str | Path = ANALYSIS_DIR, results_dir: str | Path = RESULTS_DIR,
                 example: tuple[str, str] | None = None, out_dir: str | Path = FIGURES_DIR) -> tuple[Path, ...]:
    """Render F3 from the stored outputs of ``plan`` and write its four files once (all built before any write)."""
    data = draws_figure_from_records(plan, arm=arm, stage=stage, analysis_dir=analysis_dir, results_dir=results_dir,
                                     example=example)
    texts = figure_texts(DRAWS, draws_table(data), lambda theme: plot_draws(data, theme=theme), out_dir)
    return write_outputs(texts)


def main(argv: Sequence[str] | None = None) -> int:
    """Render the draw-distribution figure of the frozen plan from its stored outputs.

    Exit codes: 0 written (or already written identically), 2 refused (another
    plan, missing or malformed metrics or results, records that disagree, or a
    different figure at the output path).
    """
    parser = argparse.ArgumentParser(description="Draw-distribution figure F3 (task 8.3), from the stored item "
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
    "EXAMPLE_DEFAULT",
    "SPLIT_CHECK_MIN_DRAWS",
    "SUMMARY_NAMES",
    "DrawStrip",
    "DrawsFigure",
    "EnsembleMark",
    "ExampleItem",
    "RankItem",
    "RateRow",
    "draws_figure",
    "draws_figure_from_records",
    "draws_table",
    "example_item",
    "exchangeability_shares",
    "main",
    "plot_draws",
    "read_ensemble_metrics",
    "render_draws",
]


if __name__ == "__main__":
    raise SystemExit(main())
