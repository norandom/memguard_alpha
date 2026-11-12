"""Accuracy-gap and reliability figures of the worked example (Figures 1 and 2).

No request is sent and nothing is recomputed from item results: every number
comes from a stored output of an earlier step.

Figure 1 ``gap``. Read from the stored pre-registered analysis
(``runs/finance-sim/analysis/pre-registered-<arm>-full_run.json``,
:func:`~scripts.finance_sim.analysis.analysis_path`), never recomputed: per
variant, the pre-cutoff minus post-cutoff difference of identifying minus
anonymised accuracy with its moving-block bootstrap interval, raw and
Holm-adjusted p-values, month counts, dropped resamples and missing months,
beside the two period differences that make it up. Dot and interval bar, one
panel, the variants side by side per serving route. A variant the analysis
reports as missing is listed in the table, named in the caption and not
drawn. The caption states that the adjusted variants cannot differ from the
unadjusted one in accuracy by construction; the achieved power is a note
below the table.

Figure 2 ``reliability``. Read from the stored calibrator files
(``runs/finance-sim/calibrators/<arm>/<source>/<macro|edgar>.json``, loaded
with :func:`~scripts.finance_sim.calibrate.load_calibrator` against the frozen
plan): the held-out probabilities and labels each calibrator retained, binned
by :func:`recall_guard.harness.plots.reliability_curve` with the bins of the
local report's held-out calibration and the same gate
(``CALIBRATION_MIN_PER_CLASS`` held-out draws per class). One panel per
calibration set; each bin is a marker with the Wilson score interval of its
observed share (the bin's draw count is its n), never joined by a line, the
two evidence modes told apart by marker and a small horizontal offset. A
calibrator below the gate, one the route cannot have, or one not fitted yet is
not drawn; the caption says so and the table gives the reason.

Words on a published figure. A route is named as data, by the plan's profile
and temperature (:func:`route_label`); the two evidence modes are "logprobs"
(the top alternatives the route serves per token) and "logits" (the top
alternatives kept from the next-token vector, never the whole vector), each
labelled with its stored count by :func:`evidence_label`; a prompt is the
"identifying prompt" or the "anonymised prompt". The column names of the
tables stay the data keys other code reads.

Outputs. Each figure is written to ``docs/assets/finance-sim/``
(:data:`FIGURES_DIR`) as one ``<figure>.svg`` with ``<figure>.csv``, the table
view of the plotted values, and ``<figure>.md``: the caption, the CSV rendered
as a table, then the notes of the table, one line of sources and the
derived-only line. The caption is the numbered caption of the one figure style
(:func:`scripts.finance_sim.figure_style.caption`): "Figure N.", the one
message with its key number, the sample size, what the interval is, and what
cannot differ by construction, in at most :data:`CAPTION_LIMIT` characters;
definitions and methods are notes below the table, and a fact is stated once
per file. The words for an interval are looked up by the
method the stored output states (:func:`interval_wording`), so a caption
cannot name another interval than the one that was computed. Tables carry
derived statistics only: no reply text, no realized returns, no raw licensed
series; the sources line cites Moody's for BAA10Y and FMP for the prices
behind the outcomes. A figure's files are written once: the same output again
is a no-op and a different one is refused, never replaced (remove the old
files first). Renders are byte-stable (see
:func:`scripts.finance_sim.figure_style.figure_svg`).

Style (Requirement 8; the one style of
:mod:`scripts.finance_sim.figure_style`, which also holds the helpers every
figure module shares and this module re-exports). A figure reads as in a
quantitative journal article: a white page, black axes and text, a serif
typeface, both axes labelled with the quantity and its unit, series told apart
by marker shape and grey level and named in one legend outside the plot area,
interval bars where a value has an interval. Nothing is written inside a plot
area. The sibling modules ``figures_draws`` (Figures 3 and 8) and
``figures_convergence`` (Figure 4) build on the table and the outputs here.

Matplotlib is imported inside the functions that draw, so importing this module
(or ``recall_guard``) never imports it.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import re
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from recall_guard.core.consensus import wilson_interval
from recall_guard.core.significance import BLOCK_BOOTSTRAP_METHOD
from recall_guard.harness.local_reporting import CALIBRATION_BINS, CALIBRATION_MIN_PER_CLASS
from scripts.finance_sim.analysis import (
    ACCURACY_POST,
    ACCURACY_PRE,
    ACHIEVED_POWER,
    ANALYSIS_DIR,
    ANALYSIS_SCHEMA,
    COMPUTED,
    DID,
    analysis_path,
)
from scripts.finance_sim.calibrate import (
    CALIBRATORS_DIR,
    CLUSTERED_METHOD,
    CalibrationRecord,
    calibrator_path,
    load_calibrator,
)
from scripts.finance_sim.figure_style import (
    DERIVED_ONLY,
    FIGURE_WIDTH_IN,
    FIGURES_DIR,
    SVG_HASH_SALT,
    FigureError,
    _default_rc,  # noqa: F401  (tests reach it here)
    caption,
    figure_rc,
    figure_svg,
    freeze_layout,
    label_panel,
    lay_out,
    panel_label,
    series_style,
    write_outputs,
)
from scripts.finance_sim.limits import ARMS, SAMPLING_ARM, SERVING_ARM
from scripts.finance_sim.pilot import CONFIRMATORY_STAGE
from scripts.finance_sim.plan import (
    CALIBRATION_KINDS,
    ESTIMATE_SOURCES,
    FULL_DISTRIBUTION,
    NO_ADJUSTMENT,
    ROLES,
    TOP_K,
    VARIANTS,
    StudyPlan,
    load_plan,
    variant_parts,
)
from scripts.finance_sim.published_text import published_reason

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from recall_guard.harness.plots import ReliabilityBin

GAP, RELIABILITY = "gap", "reliability"
#: The figures this module renders, in render order.
FIGURE_NAMES: tuple[str, ...] = (GAP, RELIABILITY)
#: The figures of the sibling modules: ``figures_draws`` renders the first-draw figure and the draw-spread figure,
#: ``figures_convergence`` the convergence figure, with the helpers here.
DRAWS, CONVERGENCE, DRAW_SPREAD = "draws", "convergence", "draw-spread"
#: Every figure of the worked example, each with its three files in ``FIGURES_DIR``.
PUBLISHED_FIGURES: tuple[str, ...] = (GAP, RELIABILITY, DRAWS, CONVERGENCE, DRAW_SPREAD)
#: The number each caption starts with ("Figure N."); 5, 6 and 7 are the figures of ``figures_catalogue``.
FIGURE_NUMBERS: Mapping[str, int] = MappingProxyType({GAP: 1, RELIABILITY: 2, DRAWS: 3, CONVERGENCE: 4,
                                                      DRAW_SPREAD: 8})
#: A caption is at most this many characters: one message, the sample size, the interval, what cannot differ.
#: Definitions and methods are notes below the table.
CAPTION_LIMIT = 600
#: Order of the routes on a figure: the plan's serving arm first, then its sampling arm.
DEFAULT_ARMS: tuple[str, ...] = (SERVING_ARM, SAMPLING_ARM)
EXIT_WRITTEN, EXIT_REFUSED = 0, 2
#: Statuses of a Figure 2 calibrator: drawn with its bins, listed without bins below the gate, or missing.
DRAWN, INSUFFICIENT, MISSING = "drawn", "insufficient", "missing"

#: The two evidence modes behind a recall estimate, as every label spells them (:func:`evidence_label` adds what
#: each one holds).
EVIDENCE_NAMES: Mapping[str, str] = MappingProxyType({TOP_K: "logprobs", FULL_DISTRIBUTION: "logits"})
#: The Wilson bound of an all-control or all-memorized bin is 0 or 1 up to floating point; within this distance it
#: is written as exactly 0 or 1.
_BOUND_TOLERANCE = 1e-12
#: Text names of the adjustments, of the draw modes and of the calibration sets, as the legends spell them.
ADJUSTMENT_NAMES: Mapping[str, str] = MappingProxyType({
    NO_ADJUSTMENT: "unadjusted", **{source: f"adjusted with {name}" for source, name in EVIDENCE_NAMES.items()}})
MODE_NAMES: Mapping[str, str] = MappingProxyType({"single": "single draw", "ensemble": "ensemble"})
KIND_NAMES: Mapping[str, str] = MappingProxyType({"macro": "macro prompts", "edgar": "EDGAR releases"})
#: What a caption calls an interval, by the method the stored output states on it.
INTERVAL_WORDING: Mapping[str, str] = MappingProxyType({
    BLOCK_BOOTSTRAP_METHOD: "moving-block bootstrap over months",
    CLUSTERED_METHOD: "prompt-clustered bootstrap",
})
#: Confidence of the Wilson score interval on a reliability bin's observed share.
RELIABILITY_CONFIDENCE = 0.95
FRED_SOURCE = ("macro rows from FRED (Federal Reserve Bank of St. Louis) as known on each decision date, including "
               "BAA10Y, Moody's Seasoned Baa Corporate Bond Yield Relative to Yield on 10-Year Treasury Constant "
               "Maturity, copyright Moody's, cited and not reproduced")
FMP_SOURCE = ("realized next-month returns from Financial Modeling Prep (FMP) end-of-day adjusted closes (licensed, "
              "not published) with DTB3 from FRED as the cash leg")
EDGAR_SOURCE = "EDGAR outlook prompts from SEC EDGAR filings"
#: Figures 2, 3, 4 and 8 use no price: FMP is cited for the study's outcomes, as every published table cites it.
FMP_STUDY_SOURCE = ("the study's outcomes come from Financial Modeling Prep (FMP) end-of-day adjusted closes "
                    "(licensed, not published), and no FMP value enters this figure")

#: A published table names the serving route and gives a variant by its plain label (:func:`variant_label`); the
#: arm, its role and the variant keys of the study stay in the private documents.
GAP_COLUMNS: tuple[str, ...] = (
    "route", "variant", "draw_mode", "adjustment", "gap", "interval_low", "interval_high", "p_raw",
    "p_holm", "months", "pre_cutoff_months", "post_cutoff_months", "missing_months", "dropped_resamples",
    "accuracy_difference_pre_cutoff", "accuracy_difference_post_cutoff", "statuses", "missing")
RELIABILITY_COLUMNS: tuple[str, ...] = (
    "route", "kind", "source", "variants", "status", "holdout_prompts", "holdout_draws",
    "memorized_draws", "control_draws", "ece", "holdout_auc", "auc_interval_low", "auc_interval_high", "weak",
    "bin_low", "bin_high", "bin_count", "mean_predicted", "observed_share", "share_interval_low",
    "share_interval_high", "reason")

#: Figure sizes in inches (width, height): Figure 1 with its legend beside the panel; Figure 2 with one square panel
#: per calibration set and the legend below them.
_GAP_SIZE_IN, _RELIABILITY_HEIGHT_IN = (FIGURE_WIDTH_IN, 3.3), 3.9
#: The variants of one route stand side by side within this share of the route's place on the x axis.
_GAP_DODGE = 0.72
#: Distance on the x axis between the markers of the two evidence modes in one reliability bin.
_RELIABILITY_DODGE = 0.02
#: Reference lines (no gap, perfect calibration): thin and grey, behind the data.
_REFERENCE_GREY, _REFERENCE_PT = "0.4", 0.7
_HEX64 = re.compile(r"[0-9a-f]{64}")


# ---------- shared helpers ----------

def _flat(text: object) -> str:
    """One line of text for a table cell or a caption."""
    return str(text).replace("\r", " ").replace("\n", " ")


def _number(value: float | int | bool | None) -> str:
    """A table value: empty when missing, integers as they are, reals to six significant digits."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    number = float(value)
    return f"{0.0 if number == 0 else number:.6g}"


def _markdown_cell(value: str) -> str:
    return _flat(value).replace("|", "\\|")


def rounded(value: float, *, signed: bool = False) -> str:
    """A number in a caption: three decimals; a nonzero value that would round to zero keeps two significant digits."""
    number = float(value)
    if number == 0:
        return "0.000"
    sign = "+" if signed else ""
    if round(abs(number), 3) == 0:
        return f"{number:{sign}.2g}"
    return f"{number:{sign}.3f}"


def route_label(plan: StudyPlan, arm: str) -> str:
    """The serving route of ``arm`` as data: the plan's profile for it and the temperature it draws at."""
    if arm not in plan.arms.arms:
        raise FigureError(f"unknown arm {arm!r}; expected one of {list(plan.arms.arms)}")
    arm_plan = plan.arms.arms[arm]
    return f"{arm_plan.profile_id} at temperature {float(arm_plan.temperature):g}"


def evidence_label(source: str, count: int | None = None) -> str:
    """An evidence mode as a legend, a caption or a note introduces it, with the stored number of alternatives.

    Logprobs are the top ``count`` alternatives the route serves per token.
    Logits are the top ``count`` alternatives kept from the next-token vector:
    the estimate never uses the whole vector, so no label says it does.
    Without a stored count the label says which alternatives without a number.
    """
    name = EVIDENCE_NAMES.get(source, source)
    if source == TOP_K:
        return f"{name} (served alternatives per token)" if count is None else f"{name} (top {count})"
    if source == FULL_DISTRIBUTION:
        return (f"{name} (kept alternatives of the next-token vector)" if count is None
                else f"{name} (top {count} of the next-token vector)")
    return name


def _route_tick(route: str) -> str:
    """A route as a tick label: the profile on one line, the temperature below it."""
    return route.replace(" at temperature ", "\nat temperature ")


def interval_wording(methods: Iterable[str | None], what: str,
                     wording: Mapping[str, str] = INTERVAL_WORDING) -> str | None:
    """What a caption calls the intervals of ``what``, by the methods the stored output states; None without one.

    Raises:
        FigureError: a method without wording here, or intervals that need different words.
    """
    found = sorted({method for method in methods if method is not None})
    unknown = [method for method in found if method not in wording]
    if unknown:
        raise FigureError(f"the caption of {what} has no wording for the interval method {unknown[0]!r}; it "
                          f"describes {sorted(wording)}")
    words = sorted({wording[method] for method in found})
    if len(words) > 1:
        raise FigureError(f"the intervals of {what} are of different kinds ({words}); the caption names one")
    return words[0] if words else None


def _listed(names: Sequence[str]) -> str:
    """``a``, ``a and b`` or ``a, b and c``."""
    names = list(names)
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


@dataclass(frozen=True)
class FigureTable:
    """The table of one figure: derived statistics in rows of text, the caption above it, notes and sources below.

    ``notes`` are the lines a reader of the table needs and the caption does
    not carry; ``sources`` the data citations of the figure, stated once.
    """

    name: str
    caption: str
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    notes: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()

    def to_csv(self) -> str:
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(self.columns)
        writer.writerows(self.rows)
        return buffer.getvalue()

    def to_markdown(self) -> str:
        """The caption, the CSV rendered as a markdown table, then the notes, the sources and the derived-only line."""

        def line(values: Iterable[str]) -> str:
            return "| " + " | ".join(_markdown_cell(value) for value in values) + " |"

        lines = [_flat(self.caption), "", line(self.columns), "|" + "|".join("---" for _ in self.columns) + "|"]
        lines += [line(row) for row in self.rows]
        lines += ["", *(_flat(note) for note in self.notes)]
        if self.sources:
            lines.append(f"Sources: {'; '.join(self.sources)}.")
        lines.append(DERIVED_ONLY)
        return "\n".join(lines) + "\n"


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        raise FigureError(f"duplicate keys {sorted({key for key in keys if keys.count(key) > 1})}")
    return dict(pairs)


def _no_constant(name: str) -> Any:
    raise FigureError(f"non-finite number {name}")


def _read_json(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise FigureError(f"{path} cannot be read: {exc}") from None
    try:
        return json.loads(text, object_pairs_hook=_no_duplicate_keys, parse_constant=_no_constant)
    except FigureError as exc:
        raise FigureError(f"{path}: {exc}") from None
    except ValueError as exc:
        raise FigureError(f"{path} is not valid JSON: {exc}") from None


def _real(value: object, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise FigureError(f"{name} must be a finite number or null; got {value!r}")
    return float(value)


def _count(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FigureError(f"{name} must be a non-negative integer; got {value!r}")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise FigureError(f"{name} must be a non-empty string; got {value!r}")
    return value


def _percent(confidence: float) -> str:
    return f"{confidence * 100:g}"


# ---------- Figure 1: the accuracy gap, from the stored pre-registered analysis ----------

@dataclass(frozen=True)
class GapPoint:
    """One variant's accuracy gap as the stored analysis gives it, or the reason it has none (``missing``).

    ``statistic`` is the pre-cutoff minus post-cutoff difference of identifying
    minus anonymised accuracy, ``low`` and ``high`` its interval by
    ``interval_method``, ``p_value`` and ``p_holm`` its raw and Holm-adjusted
    two-sided p-values. ``months`` counts the months used
    (``pre_cutoff_months`` and ``post_cutoff_months`` by period),
    ``missing_months`` the planned months without the value for both prompts,
    ``dropped_resamples`` the resamples whose period cell was empty.
    ``pre_difference`` and ``post_difference`` are the two cells of the gap.
    ``statuses`` counts the variant's guarded statuses over the analysed items.
    """

    variant: str
    mode: str
    adjustment: str
    statistic: float | None
    low: float | None
    high: float | None
    p_value: float | None
    p_holm: float | None
    months: int
    pre_cutoff_months: int
    post_cutoff_months: int
    missing_months: int
    dropped_resamples: int | None
    pre_difference: float | None
    post_difference: float | None
    statuses: Mapping[str, int] = field(default_factory=lambda: MappingProxyType({}))
    missing: str | None = None
    interval_method: str | None = None

    @property
    def drawn(self) -> bool:
        return self.missing is None


@dataclass(frozen=True)
class GapPanel:
    """One route's part of Figure 1: its months, the analysis settings, its points and its power.

    ``route`` is the serving route the panel is labelled by
    (:func:`route_label`); a panel without it is read but cannot be drawn.
    """

    plan_sha256: str
    arm: str
    role: str
    model: str
    planned_months: int
    pre_cutoff_months: int
    post_cutoff_months: int
    settings: Mapping[str, Any]
    points: tuple[GapPoint, ...]
    power: float | None
    power_reason: str | None
    power_approximation: str | None
    power_n_sim: int | None
    power_effect_size: float | None
    route: str | None = None

    def point(self, variant: str) -> GapPoint:
        for point in self.points:
            if point.variant == variant:
                return point
        raise KeyError(variant)

    @property
    def drawn_points(self) -> tuple[GapPoint, ...]:
        return tuple(point for point in self.points if point.drawn)

    @property
    def max_dropped(self) -> int:
        return max((point.dropped_resamples or 0 for point in self.drawn_points), default=0)

    @property
    def power_monte_carlo_error(self) -> float | None:
        """Binomial standard error of the simulated power, ``sqrt(p (1 - p) / n_sim)``."""
        if self.power is None or not self.power_n_sim:
            return None
        return math.sqrt(self.power * (1.0 - self.power) / self.power_n_sim)


def _metric_entries(data: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    entries = data["metrics"]
    if not isinstance(entries, list):
        raise FigureError("metrics must be a list")
    found: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        name = _text(entry["metric"], "metric")
        if name in found:
            raise FigureError(f"the analysis lists the metric {name!r} twice")
        found[name] = entry
    return found


def _cells(entry: Mapping[str, Any] | None, variants: Sequence[str], name: str) -> dict[str, Mapping[str, Any]]:
    if entry is None or entry["status"] != COMPUTED:
        return {}
    cells = {_text(item["variant"], "variant"): item for item in entry["variants"]}
    if sorted(cells) != sorted(variants):
        raise FigureError(f"the {name} entry lists the variants {sorted(cells)}, not {sorted(variants)}")
    return cells


def _gap_point(variant: str, did: Mapping[str, Any], pre: Mapping[str, Any] | None,
               post: Mapping[str, Any] | None) -> GapPoint:
    mode, adjustment = variant_parts(variant)
    interval = did["interval"]
    low = high = method = None
    if interval is not None:
        low, high = _real(interval["lo"], f"{variant} interval lo"), _real(interval["hi"], f"{variant} interval hi")
        method = _text(interval["method"], f"{variant} interval method")
    statistic = _real(did["statistic"], f"{variant} statistic")
    missing = did["missing"]
    if missing is not None:
        missing = _text(missing, f"{variant} missing")
    elif statistic is None:
        raise FigureError(f"the {variant} gap has neither a statistic nor a reason it is missing")
    dropped = did["dropped_resamples"]
    statuses = did["statuses"]
    if not isinstance(statuses, Mapping):
        raise FigureError(f"the {variant} statuses must be a mapping")
    if not isinstance(did["missing_months"], list):
        raise FigureError(f"the {variant} missing_months must be a list")
    return GapPoint(
        variant=variant, mode=mode, adjustment=adjustment, statistic=statistic, low=low, high=high,
        p_value=_real(did["p_value"], f"{variant} p_value"), p_holm=_real(did["p_holm"], f"{variant} p_holm"),
        months=_count(did["months"], "months"), pre_cutoff_months=_count(did["pre_cutoff_months"], "pre_cutoff_months"),
        post_cutoff_months=_count(did["post_cutoff_months"], "post_cutoff_months"),
        missing_months=len(did["missing_months"]),
        dropped_resamples=None if dropped is None else _count(dropped, "dropped_resamples"),
        pre_difference=None if pre is None else _real(pre["statistic"], f"{variant} pre-cutoff difference"),
        post_difference=None if post is None else _real(post["statistic"], f"{variant} post-cutoff difference"),
        statuses=MappingProxyType({_text(key, "status"): _count(value, "status count")
                                   for key, value in statuses.items()}),
        missing=missing, interval_method=method)


def gap_panel(data: Mapping[str, Any], *, plan_sha256: str | None = None, route: str | None = None) -> GapPanel:
    """One route's stored pre-registered analysis (its JSON payload), read and checked, not recomputed.

    ``route`` is the label of the serving route (:func:`route_label`).
    Refuses another schema, an arm outside the study, a stage other than the
    full run, another plan than ``plan_sha256`` (when given), and an analysis
    without a computed accuracy gap.
    """
    if not isinstance(data, Mapping):
        raise FigureError(f"the analysis must be a JSON object; got {type(data).__name__}")
    if data.get("schema") != ANALYSIS_SCHEMA:
        raise FigureError(f"not a {ANALYSIS_SCHEMA} analysis: schema {data.get('schema')!r}")
    arm = data.get("arm")
    if arm not in ARMS:
        raise FigureError(f"the analysis names the arm {arm!r}; expected one of {list(ARMS)}")
    if data.get("stage") != CONFIRMATORY_STAGE:
        raise FigureError(f"the accuracy-gap figure reads the {CONFIRMATORY_STAGE} stage's analysis; got stage "
                          f"{data.get('stage')!r}")
    sha = data.get("plan_sha256")
    if not isinstance(sha, str) or not _HEX64.fullmatch(sha):
        raise FigureError(f"the analysis plan_sha256 must be 64 lowercase hex characters; got {sha!r}")
    if plan_sha256 is not None and sha != plan_sha256:
        raise FigureError(f"the {arm} arm's analysis was written under plan {sha[:12]}, not the frozen plan "
                          f"{plan_sha256[:12]}")
    try:
        role = data["role"]
        if role not in ROLES:
            raise FigureError(f"the analysis names the role {role!r}; expected one of {list(ROLES)}")
        variants = data["variants"]
        if (not isinstance(variants, list) or len(set(variants)) != len(variants)
                or not set(variants) <= set(VARIANTS)):
            raise FigureError(f"the analysis variants must be distinct frozen variants; got {variants!r}")
        entries = _metric_entries(data)
        did = entries.get(DID)
        if did is None or did["status"] != COMPUTED:
            raise FigureError(f"the {arm} arm's analysis carries no computed {DID}")
        did_cells = _cells(did, variants, DID)
        pre_cells = _cells(entries.get(ACCURACY_PRE), variants, ACCURACY_PRE)
        post_cells = _cells(entries.get(ACCURACY_POST), variants, ACCURACY_POST)
        points = tuple(_gap_point(name, did_cells[name], pre_cells.get(name), post_cells.get(name))
                       for name in variants)
        settings = data["settings"]
        for key in ("block_length", "n_resamples", "seed", "confidence", "alpha", "effect_size"):
            if key not in settings:
                raise FigureError(f"the analysis settings lack {key!r}")
        # The achieved_power entry carries its mapping (value, reason, label, n_sim) even when the value is missing.
        power_entry = entries.get(ACHIEVED_POWER)
        recorded = None if power_entry is None else power_entry["power"]
        if recorded is None:
            reason = "the stored analysis carries no achieved_power entry" if power_entry is None else (
                power_entry["reason"] or "no reason recorded")
            values: Mapping[str, Any] = {"power": None, "reason": reason, "approximation": None, "n_sim": None,
                                         "effect_size": None}
        else:
            values = recorded
        return GapPanel(
            plan_sha256=sha, arm=arm, role=role, model=_text(data["model"], "model"), route=route,
            planned_months=_count(data["planned_months"], "planned_months"),
            pre_cutoff_months=_count(data["pre_cutoff_months"], "pre_cutoff_months"),
            post_cutoff_months=_count(data["post_cutoff_months"], "post_cutoff_months"),
            settings=MappingProxyType(dict(settings)), points=points,
            power=_real(values["power"], "achieved power"),
            power_reason=None if values["reason"] is None else _flat(values["reason"]),
            power_approximation=None if values["approximation"] is None else _flat(values["approximation"]),
            power_n_sim=None if values["n_sim"] is None else _count(values["n_sim"], "n_sim"),
            power_effect_size=_real(values["effect_size"], "effect_size"))
    except (KeyError, TypeError, AttributeError) as exc:
        raise FigureError(f"the {arm} arm's analysis is malformed: {exc!r}") from None


def read_gap_panel(path: str | Path, *, plan_sha256: str | None = None, route: str | None = None) -> GapPanel:
    """The panel of the analysis JSON at ``path`` (duplicate keys and non-finite numbers refused)."""
    return gap_panel(_read_json(Path(path)), plan_sha256=plan_sha256, route=route)


def _checked_gap_panels(panels: Iterable[GapPanel]) -> tuple[GapPanel, ...]:
    panels = tuple(panels)
    if not panels:
        raise FigureError("the accuracy-gap figure needs at least one arm's analysis")
    for panel in panels:
        if not isinstance(panel, GapPanel):
            raise TypeError(f"panels must be GapPanel values; got {type(panel).__name__}")
    if len({panel.plan_sha256 for panel in panels}) != 1:
        raise FigureError("the accuracy-gap panels must come from one plan; got the plans "
                          f"{sorted(panel.plan_sha256[:12] for panel in panels)}")
    arms = [panel.arm for panel in panels]
    for arm in arms:
        if arms.count(arm) > 1:
            raise FigureError(f"the {arm} arm is given twice")
    if len({json.dumps(dict(panel.settings), sort_keys=True) for panel in panels}) != 1:
        raise FigureError("the accuracy-gap panels carry different analysis settings")
    for panel in panels:
        if not panel.route:
            raise FigureError(f"the accuracy-gap figure labels a panel by its serving route; the {panel.arm} panel "
                              "carries none (see route_label)")
    return panels


def _statuses(point: GapPoint) -> str:
    return "; ".join(f"{status} {count}" for status, count in point.statuses.items())


def variant_label(variant: str) -> str:
    """A variant in plain words, the one mapping every legend and every published table uses.

    "single draw, unadjusted", "ensemble, adjusted with logprobs", "ensemble,
    adjusted with logits": the draw mode, then the adjustment by its evidence
    mode. An unknown variant is refused.
    """
    try:
        mode, adjustment = variant_parts(variant)
        return f"{MODE_NAMES[mode]}, {ADJUSTMENT_NAMES[adjustment]}"
    except (KeyError, ValueError) as exc:
        raise FigureError(f"unknown variant {variant!r}: {exc}") from None


_variant_label = variant_label


#: The variant whose gap the caption states: the ensemble without adjustment, the reference of its draw mode.
_REFERENCE_VARIANT = f"ensemble_{NO_ADJUSTMENT}"


def _stated_point(panel: GapPanel) -> GapPoint | None:
    """The variant whose gap the caption states for ``panel``: the reference, else the first drawn; None without."""
    drawn = panel.drawn_points
    return next((point for point in drawn if point.variant == _REFERENCE_VARIANT), drawn[0] if drawn else None)


def _gap_key(panel: GapPanel, name: str) -> tuple[str, str | None]:
    """The gap the caption states for ``panel``, named ``name``, and its variant (None when none has a value)."""
    point = _stated_point(panel)
    if point is None:
        return f"no variant has a value on {name}", None
    interval = "no interval" if point.low is None or point.high is None else (
        f"{rounded(point.low, signed=True)} to {rounded(point.high, signed=True)}")
    return f"{rounded(point.statistic, signed=True)} ({interval}) on {name}", point.variant


def _gap_sizes(panels: Sequence[GapPanel], short: Mapping[str | None, str]) -> str:
    """The months the stated gap of each route rests on, by period; the planned months only where they differ.

    The first route is written in full ("88 of 90 planned months (67 on or
    before the cutoff, 21 after)"), the next ones as the same numbers in the
    same order ("20 (15, 5)"). The counts are those of the stated variant
    (:func:`_stated_point`), never the plan's.
    """
    sizes = []
    for panel in panels:
        point = _stated_point(panel)
        where = f" on {short[panel.route]}" if len(panels) > 1 else ""
        if point is None:
            sizes.append(f"no month with a value{where}")
            continue
        planned = "" if point.months == panel.planned_months else f" of {panel.planned_months} planned"
        if any(" on or before the cutoff" in size for size in sizes):
            sizes.append(f"{point.months}{planned} ({point.pre_cutoff_months}, {point.post_cutoff_months}){where}")
        else:
            sizes.append(f"{point.months}{planned} months ({point.pre_cutoff_months} on or before the cutoff, "
                         f"{point.post_cutoff_months} after){where}")
    return "; ".join(sizes)


def short_routes(routes: Iterable[str | None]) -> dict[str | None, str]:
    """Each route by its profile alone, where that still tells the routes apart; otherwise in full.

    A caption names a route in full once, with its temperature, and by this
    short name after that.
    """
    routes = list(dict.fromkeys(routes))
    short = {route: str(route).split(" at temperature ")[0] for route in routes}
    return short if len(set(short.values())) == len(short) else {route: str(route) for route in routes}


def _absent_variants(panel: GapPanel) -> list[str]:
    """The variants of ``panel`` without a value, an adjustment missing in every draw mode named once."""
    absent = [point for point in panel.points if not point.drawn]
    names = []
    for adjustment in dict.fromkeys(point.adjustment for point in absent):
        group = [point for point in absent if point.adjustment == adjustment]
        modes = {point.mode for point in panel.points if point.adjustment == adjustment}
        if len(group) > 1 and {point.mode for point in group} == modes:
            names.append(f"both variants {ADJUSTMENT_NAMES[adjustment]}")
        else:
            names += [_variant_label(point.variant) for point in group]
    return names


def _gap_caption(panels: Sequence[GapPanel]) -> str:
    """Figure 1's caption: the stated gap per route, the months it rests on, the interval, what cannot differ.

    A route is named by its profile where that tells the routes apart
    (:func:`short_routes`); the x axis and the table carry its temperature.
    """
    settings = panels[0].settings
    short = short_routes(panel.route for panel in panels)
    keys = [_gap_key(panel, short[panel.route]) for panel in panels]
    variants = {variant for _, variant in keys if variant is not None}
    if variants == {_REFERENCE_VARIANT}:
        subject, numbers = ", unadjusted ensemble", _listed([text for text, _ in keys])
    else:
        subject, numbers = "", _listed([text if variant is None else f"{text} ({_variant_label(variant)})"
                                        for text, variant in keys])
    message = ("Accuracy gap (identifying minus anonymised prompt, before minus after the training cutoff)"
               f"{subject}: {numbers}")
    kind = interval_wording((point.interval_method for panel in panels for point in panel.drawn_points),
                            "the accuracy gap")
    notes = ["No interval is stored for the gap." if kind is None else (
        f"Bars and parentheses: {_percent(settings['confidence'])} percent {kind.replace(' over months', '')} "
        "intervals.")]
    notes.append("The adjusted variants of a draw mode cannot differ from its unadjusted one by construction.")
    for panel in panels:
        absent = _absent_variants(panel)
        if absent:
            notes.append(f"Not drawn on {short[panel.route]}: {'; '.join(absent)}.")
    return caption(FIGURE_NUMBERS[GAP], message, n=_gap_sizes(panels, short), interval=" ".join(notes))


def _gap_notes(panels: Sequence[GapPanel]) -> tuple[str, ...]:
    """The lines below Figure 1's table: why the adjusted variants cannot differ, what the evidence modes are, the
    bootstrap and what its interval is conditional on, the p-values of the columns and the achieved power."""
    settings = panels[0].settings
    conditional = (" The intervals are conditional on the resamples with months on both sides of the cutoff."
                   if any(panel.max_dropped for panel in panels) else "")
    powers = []
    for panel in panels:
        if panel.power is None:
            powers.append(f"not available on {panel.route}")
            continue
        error = panel.power_monte_carlo_error
        detail = "" if error is None else (f" (Monte Carlo standard error {error:.3f}, {panel.power_n_sim} "
                                           "simulations)")
        powers.append(f"{panel.power:.3f} on {panel.route}{detail}")
    return (
        "An adjustment scales the size of a call and never changes its direction.",
        f"The adjustments use the recall estimate from {evidence_label(TOP_K)} or from "
        f"{evidence_label(FULL_DISTRIBUTION)}.",
        f"The bootstrap resamples blocks of {settings['block_length']} months, {settings['n_resamples']} times.",
        "The raw p-value is two-sided, from the kept resamples; the Holm p-value adjusts it across the variants of "
        f"one route (alpha {settings['alpha']:g}). The two period differences are measurements without a p-value.",
        f"Achieved power for the pre-registered effect {settings['effect_size']:g}, a t-test approximation under "
        f"independent months: {'; '.join(powers)}.",
        "A month counts for a variant when both prompts carry the value; the other planned months are counted as "
        f"missing. A dropped resample held no month on one side of the cutoff.{conditional}",
    )


def gap_table(panels: Iterable[GapPanel]) -> FigureTable:
    """Figure 1's table (one row per route and variant, missing ones included with the reason) and its caption."""
    panels = _checked_gap_panels(panels)
    rows = []
    for panel in panels:
        for point in panel.points:
            rows.append((
                panel.route, variant_label(point.variant), point.mode, point.adjustment,
                _number(point.statistic),
                _number(point.low), _number(point.high), _number(point.p_value), _number(point.p_holm),
                _number(point.months), _number(point.pre_cutoff_months), _number(point.post_cutoff_months),
                _number(point.missing_months), _number(point.dropped_resamples), _number(point.pre_difference),
                _number(point.post_difference), _statuses(point), published_reason(point.missing)))
    return FigureTable(name=GAP, caption=_gap_caption(panels), columns=GAP_COLUMNS, rows=tuple(rows),
                       notes=_gap_notes(panels), sources=(FRED_SOURCE, FMP_SOURCE))


def _gap_offset(variant: str) -> float:
    """Where a variant stands within its route's place on the x axis: the variants side by side, in plan order."""
    return (VARIANTS.index(variant) - (len(VARIANTS) - 1) / 2.0) * _GAP_DODGE / len(VARIANTS)


def plot_gap(panels: Sequence[GapPanel]) -> Figure:
    """Figure 1: the accuracy gap of each variant with its interval bar, the variants side by side per route.

    One panel in the academic style: the x axis is the serving route, the y
    axis the gap, a variant keeps its marker and its place on every route, and
    a legend beside the panel names the drawn variants. ``panels`` are plain
    data read from the stored analyses (:func:`gap_panel`). A missing variant
    is not drawn and leaves its place empty; nothing is written inside the
    plot (the caption and the table say what is missing). Returns a matplotlib
    Figure (no pyplot, no global state).
    """
    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    panels = _checked_gap_panels(panels)
    with matplotlib.rc_context(figure_rc()):
        figure = Figure(figsize=_GAP_SIZE_IN, layout="constrained")
        axes = figure.subplots()
        axes.axhline(0.0, color=_REFERENCE_GREY, linewidth=_REFERENCE_PT, linestyle="-", zorder=1, gid="gap-zero")
        for place, panel in enumerate(panels):
            for point in panel.drawn_points:
                style = {**series_style(VARIANTS.index(point.variant)), "linestyle": "none"}
                x = place + _gap_offset(point.variant)
                if point.low is not None and point.high is not None:
                    # The interval as stored (a percentile interval need not hold the point), capped at both ends.
                    axes.plot([x, x], [point.low, point.high], color=style["color"], linestyle="-", marker="_",
                              markeredgecolor=style["color"], zorder=2,
                              gid=f"gap-interval-{panel.arm}-{point.variant}")
                axes.plot([x], [point.statistic], zorder=3, gid=f"gap-{panel.arm}-{point.variant}", **style)
        axes.set_xlim(-0.5, len(panels) - 0.5)
        axes.set_xticks(range(len(panels)), labels=[_route_tick(panel.route) for panel in panels])
        axes.tick_params(axis="x", length=0)
        axes.set_xlabel("serving route")
        axes.set_ylabel("accuracy gap, before minus after the cutoff\n(difference in share of months correct)")
        drawn = {point.variant for panel in panels for point in panel.drawn_points}
        handles = [Line2D([], [], label=_variant_label(variant),
                          **{**series_style(index), "linestyle": "none"})
                   for index, variant in enumerate(VARIANTS) if variant in drawn]
        figure.legend(handles=handles, loc="outside right upper")
        lay_out(figure)
    return figure


# ---------- Figure 2: reliability of the recall estimate, from the calibrators ----------

@dataclass(frozen=True)
class HeldOutReliability:
    """A reliability curve of held-out pairs behind the local report's gate, or the reason it is insufficient."""

    status: str
    reason: str | None
    n_memorized: int
    n_control: int
    bins: tuple[ReliabilityBin, ...]
    ece: float | None


def held_out_reliability(probabilities: Sequence[float], labels: Sequence[int], *,
                         min_per_class: int = CALIBRATION_MIN_PER_CLASS) -> HeldOutReliability:
    """The reliability curve of held-out pairs, gated like the local report's held-out calibration.

    With fewer than ``min_per_class`` held-out draws in either class (the
    report's ``CALIBRATION_MIN_PER_CLASS``, where its ECE is null) the status is
    ``insufficient`` with the counts; otherwise the bins and ECE of
    :func:`recall_guard.harness.plots.reliability_curve` over the report's
    ``CALIBRATION_BINS`` equal-width bins.
    """
    probabilities, labels = list(probabilities), list(labels)
    if len(probabilities) != len(labels):
        raise FigureError(f"{len(probabilities)} held-out probabilities but {len(labels)} labels")
    if any(isinstance(label, bool) or label not in (0, 1) for label in labels):
        raise FigureError("every held-out label must be 0 (control) or 1 (memorized)")
    n_memorized = sum(1 for label in labels if label == 1)
    n_control = len(labels) - n_memorized
    if n_memorized < min_per_class or n_control < min_per_class:
        reason = (f"insufficient: {n_memorized} memorized and {n_control} control held-out draws; the local report's "
                  f"held-out calibration needs at least {min_per_class} per class")
        return HeldOutReliability(INSUFFICIENT, reason, n_memorized, n_control, (), None)
    from recall_guard.harness.plots import reliability_curve

    curve = reliability_curve(probabilities, labels, n_bins=CALIBRATION_BINS)
    return HeldOutReliability(DRAWN, None, n_memorized, n_control, curve.bins, curve.ece)


@dataclass(frozen=True)
class ReliabilityPanel:
    """One calibrator of Figure 2: its route, calibration set, evidence mode and its held-out reliability.

    ``status`` is ``drawn`` (with ``bins`` and ``ece``), ``insufficient``
    (shown without bins, the reason gives the counts) or ``missing`` (not
    drawn, with the reason). ``route`` is the serving route
    (:func:`route_label`), ``served_k`` the number of alternatives it
    serves per token and ``kept`` the number of alternatives of the next-token
    vector a calibrator of logits was fitted on (its evidence settings; None
    for logprobs). ``holdout_prompts`` and ``holdout_draws`` give the
    sample, ``n_memorized`` and ``n_control`` its classes; ``auc`` with
    ``auc_low`` and ``auc_high`` is the held-out AUC with the interval the
    calibration reports (``auc_method`` at ``auc_confidence`` over
    ``auc_resamples`` resamples); ``is_weak`` marks a calibrator the guard
    does not apply. ``variants`` are the variants it serves.
    """

    plan_sha256: str
    arm: str
    role: str
    kind: str
    source: str
    variants: tuple[str, ...]
    status: str
    reason: str | None = None
    holdout_prompts: int | None = None
    holdout_draws: int | None = None
    n_memorized: int | None = None
    n_control: int | None = None
    ece: float | None = None
    bins: tuple[ReliabilityBin, ...] = ()
    is_weak: bool | None = None
    auc: float | None = None
    auc_low: float | None = None
    auc_high: float | None = None
    auc_method: str | None = None
    auc_confidence: float | None = None
    auc_resamples: int | None = None
    route: str | None = None
    served_k: int | None = None
    kept: int | None = None

    @property
    def key(self) -> str:
        return f"{self.arm}-{self.kind}-{self.source}"

    @property
    def alternatives(self) -> int | None:
        """The stored number of alternatives behind the estimate: the served k for logprobs, the kept for logits."""
        return self.served_k if self.source == TOP_K else self.kept


def _served_variants(source: str) -> tuple[str, ...]:
    """The variants one estimate source's calibrator serves: both draw modes of that adjustment."""
    return tuple(variant for variant in VARIANTS if variant_parts(variant)[1] == source)


def _route_fields(plan: StudyPlan, arm: str) -> dict[str, Any]:
    arm_plan = plan.arms.arms[arm]
    return {"role": arm_plan.role, "route": route_label(plan, arm), "served_k": arm_plan.top_logprobs}


def reliability_panel(record: CalibrationRecord, plan: StudyPlan) -> ReliabilityPanel:
    """The Figure 2 panel of one loaded calibrator record of ``plan`` (its retained held-out pairs, gated)."""
    if not isinstance(record, CalibrationRecord):
        raise TypeError(f"record must be a CalibrationRecord; got {type(record).__name__}")
    if record.plan_sha256 != plan.plan_sha256:
        raise FigureError(f"the {record.arm} {record.kind} {record.source.value} calibrator belongs to plan "
                          f"{record.plan_sha256[:12]}, not {plan.plan_sha256[:12]}")
    source = record.source.value
    mcs = record.calibrator.mcs
    interval = record.holdout_auc_interval_by_prompt
    base: dict[str, Any] = {
        "plan_sha256": record.plan_sha256, "arm": record.arm, **_route_fields(plan, record.arm),
        "kind": record.kind, "source": source, "variants": _served_variants(source),
        "holdout_prompts": record.counts.get("holdout_prompts"), "is_weak": record.calibrator.is_weak,
        "auc": _real(interval["auc"], "auc"), "auc_low": _real(interval["ci_low"], "ci_low"),
        "auc_high": _real(interval["ci_high"], "ci_high"), "auc_method": _text(interval["method"], "method"),
        "auc_confidence": _real(interval["confidence"], "confidence"),
        "auc_resamples": _count(interval["n_resamples"], "n_resamples"),
        "kept": None if record.evidence_settings is None else _count(record.evidence_settings.top_k_keep,
                                                                     "top_k_keep"),
    }
    if mcs.holdout_probabilities is None or mcs.holdout_labels is None:
        return ReliabilityPanel(**base, status=MISSING, reason="the calibrator carries no retained held-out pairs")
    result = held_out_reliability(mcs.holdout_probabilities, mcs.holdout_labels)
    return ReliabilityPanel(**base, status=result.status, reason=result.reason,
                            holdout_draws=len(mcs.holdout_probabilities), n_memorized=result.n_memorized,
                            n_control=result.n_control, ece=result.ece, bins=result.bins)


def reliability_panels(plan: StudyPlan, *, arms: Sequence[str] = DEFAULT_ARMS,
                       kinds: Sequence[str] = CALIBRATION_KINDS,
                       calibrators_dir: str | Path = CALIBRATORS_DIR) -> tuple[ReliabilityPanel, ...]:
    """Every Figure 2 panel of ``arms`` by kind and estimate source, from the calibrator files of the frozen plan.

    A source the arm does not offer is missing with the plan's reason for its
    variants; a calibrator file not written yet is missing with its path under
    the calibrators directory. A file that is not this plan's calibrator for its
    arm, kind and source is refused (``load_calibrator``).
    """
    arms = _checked_arms(arms)
    panels = []
    for arm in arms:
        arm_plan = plan.arms.arms[arm]
        for kind in kinds:
            if kind not in CALIBRATION_KINDS:
                raise FigureError(f"unknown calibration kind {kind!r}; expected one of {list(CALIBRATION_KINDS)}")
            for source in ESTIMATE_SOURCES:
                variants = _served_variants(source)
                base = {"plan_sha256": plan.plan_sha256, "arm": arm, **_route_fields(plan, arm), "kind": kind,
                        "source": source, "variants": variants}
                if source not in arm_plan.estimate_sources:
                    reasons = dict.fromkeys(arm_plan.missing_variants[variant] for variant in variants)
                    panels.append(ReliabilityPanel(**base, status=MISSING, reason="; ".join(reasons)))
                    continue
                path = calibrator_path(arm, source, kind, calibrators_dir)
                if not path.exists():
                    panels.append(ReliabilityPanel(
                        **base, status=MISSING, reason=f"no calibrator file at {arm}/{source}/{kind}.json under the "
                                                       "calibrators directory: the calibration has not been fitted"))
                    continue
                record = load_calibrator(path, plan, arm=arm, kind=kind, source=source)
                panels.append(reliability_panel(record, plan))
    return tuple(panels)


def _checked_reliability_panels(panels: Iterable[ReliabilityPanel]) -> tuple[ReliabilityPanel, ...]:
    panels = tuple(panels)
    if not panels:
        raise FigureError("the reliability figure needs at least one calibrator panel")
    for panel in panels:
        if not isinstance(panel, ReliabilityPanel):
            raise TypeError(f"panels must be ReliabilityPanel values; got {type(panel).__name__}")
    if len({panel.plan_sha256 for panel in panels}) != 1:
        raise FigureError("the reliability panels must come from one plan")
    keys = [panel.key for panel in panels]
    for key in keys:
        if keys.count(key) > 1:
            raise FigureError(f"the calibrator {key} is given twice")
    return panels


def bin_interval(bin_: ReliabilityBin) -> tuple[float, float]:
    """The Wilson score interval of a bin's observed share at ``RELIABILITY_CONFIDENCE``; the bin's count is n.

    A bound within ``_BOUND_TOLERANCE`` of 0 or 1 is exactly 0 or 1: the lower
    bound of a bin without a memorized draw is 0 and the upper bound of a bin
    of memorized draws only is 1, and floating point leaves a residue there.
    """
    memorized = round(bin_.observed_share * bin_.count)
    interval = wilson_interval(memorized, bin_.count, confidence=RELIABILITY_CONFIDENCE)
    if interval is None:
        raise FigureError("a reliability bin without a draw has no interval")
    low, high = (0.0 if abs(bound) <= _BOUND_TOLERANCE else 1.0 if abs(bound - 1.0) <= _BOUND_TOLERANCE
                 else float(bound) for bound in interval)
    return low, high


def _kinds(panels: Sequence[ReliabilityPanel]) -> tuple[str, ...]:
    """The calibration sets of ``panels`` in plan order: one panel of the figure each."""
    present = {panel.kind for panel in panels}
    return tuple(kind for kind in CALIBRATION_KINDS if kind in present)


def _series_keys(panels: Sequence[ReliabilityPanel]) -> list[tuple[str, str]]:
    """The (arm, estimate source) pairs with a drawn calibrator, in input order: one marker of the legend each."""
    return list(dict.fromkeys((panel.arm, panel.source) for panel in panels if panel.status == DRAWN))


def _reliability_sizes(drawn: Sequence[ReliabilityPanel], letters: Mapping[str, str]) -> str:
    """The held-out sample of the drawn calibrators, per panel of the figure.

    Where the calibrators of every panel hold the same sample, one statement
    per panel ("(a) 24 held-out draws of 12 prompts and (b) 70 of 35, per
    evidence mode"); otherwise each calibrator by its evidence mode (and its
    route when several are drawn). The unit is written once.
    """
    if not drawn:
        return "no held-out draws drawn"
    routes = len({panel.route for panel in drawn})
    groups = [[panel for panel in drawn if panel.kind == kind] for kind in _kinds(drawn)]
    shared = all(len({(panel.holdout_draws, panel.holdout_prompts) for panel in group}) == 1 for group in groups)
    written = False

    def sample(panel: ReliabilityPanel) -> str:
        nonlocal written
        text = (f"{panel.holdout_draws} of {panel.holdout_prompts}" if written else
                f"{panel.holdout_draws} held-out draws of {panel.holdout_prompts} prompts")
        written = True
        return text

    def lead(group: Sequence[ReliabilityPanel]) -> str:
        return f"{letters[group[0].kind]} " if letters else ""

    if shared:
        text = _listed([f"{lead(group)}{sample(group[0])}" for group in groups])
        if any(len(group) > 1 for group in groups):
            text += ", per evidence mode" + (" and route" if routes > 1 else "")
        return text
    return "; ".join(lead(group) + ", ".join(
        f"{EVIDENCE_NAMES.get(panel.source, panel.source)}{f' on {panel.route}' if routes > 1 else ''}: "
        f"{sample(panel)}" for panel in group) for group in groups)


def _where(panel: ReliabilityPanel, letters: Mapping[str, str], routes: int) -> str:
    """A calibrator in a caption: its evidence mode, its panel when there are several, its route when several."""
    text = EVIDENCE_NAMES.get(panel.source, panel.source)
    if letters:
        text += f" in {letters[panel.kind]}"
    return text if routes < 2 else f"{text} on {panel.route}"


def _reliability_caption(panels: Sequence[ReliabilityPanel]) -> str:
    kinds = _kinds(panels)
    letters = {kind: panel_label(index) for index, kind in enumerate(kinds)} if len(kinds) > 1 else {}
    drawn = [panel for panel in panels if panel.status == DRAWN]
    routes = list(dict.fromkeys(panel.route for panel in drawn))
    sets = _listed([f"{letters[kind]} {KIND_NAMES.get(kind, kind)}" if letters else KIND_NAMES.get(kind, kind)
                    for kind in kinds])
    eces = [panel.ece for panel in drawn if panel.ece is not None]
    if not eces:
        key = "no calibrator has enough held-out draws to be drawn"
    elif len(eces) == 1:
        key = f"expected calibration error {rounded(eces[0])}"
    else:
        key = f"expected calibration error {rounded(min(eces))} to {rounded(max(eces))}"
    where = f" on {routes[0]}" if len(routes) == 1 else ""
    message = (f"Observed share of memorized draws against mean predicted recall probability in {CALIBRATION_BINS} "
               f"equal-width bins of held-out draws{where}, for {sets}; {key}")
    notes = [f"Bars are {_percent(RELIABILITY_CONFIDENCE)} percent Wilson score intervals on a bin's observed share "
             "(n is the bin's draw count), treating the draws of one prompt as independent."]
    thin = [_where(panel, letters, len({panel.route for panel in panels})) for panel in panels
            if panel.status == INSUFFICIENT]
    if thin:
        notes.append(f"Not drawn, with fewer than {CALIBRATION_MIN_PER_CLASS} held-out draws in a class: "
                     f"{'; '.join(thin)}.")
    absent = [panel for panel in panels if panel.status == MISSING]
    for route in dict.fromkeys(panel.route for panel in absent):
        if all(panel.status == MISSING for panel in panels if panel.route == route):
            notes.append(f"No calibrator of {route} is drawn (reasons in the table).")
        else:
            names = [_where(panel, letters, 1) for panel in absent if panel.route == route]
            notes.append(f"Not drawn on {route} (reasons in the table): {'; '.join(names)}.")
    return caption(FIGURE_NUMBERS[RELIABILITY], message, n=_reliability_sizes(drawn, letters),
                   interval=" ".join(notes))


def _reliability_notes(panels: Sequence[ReliabilityPanel]) -> tuple[str, ...]:
    """The lines below Figure 2's table: what a row is and what its ECE and AUC columns are."""
    shown = [panel for panel in panels if panel.status != MISSING and panel.auc_method is not None]
    notes = [
        "Each row of a drawn calibrator is one bin with its draw count.",
        "The expected calibration error (column ece) is the count-weighted mean absolute difference between the "
        "mean predicted probability and the observed share over the bins; empty bins are omitted.",
        "One calibrator serves the single-draw and the ensemble variant of its evidence mode; a weak calibrator is "
        "marked in the table and is not applied.",
    ]
    kind = interval_wording((panel.auc_method for panel in shown), "the reliability table")
    if kind is not None:
        settings = list(dict.fromkeys((panel.auc_confidence, panel.auc_resamples) for panel in shown))
        detail = "; ".join(f"{_percent(confidence)} percent {kind} interval ({resamples} resamples)"
                           for confidence, resamples in settings)
        notes.append(f"The held-out AUC carries a {detail}.")
    offset = _reliability_offset_note(panels)
    if offset is not None:
        notes.append(offset)
    return tuple(notes)


def _reliability_shift(index: int, series: int) -> float:
    """How far the markers of series ``index`` of ``series`` stand from their bin's mean predicted probability."""
    return (index - (series - 1) / 2.0) * _RELIABILITY_DODGE


def _reliability_offset_note(panels: Sequence[ReliabilityPanel]) -> str | None:
    """The note that the figure offsets the markers sideways, with the size it uses; None with one drawn series."""
    series = _series_keys(panels)
    if len(series) < 2:
        return None
    names = [EVIDENCE_NAMES.get(source, source) for _, source in series]
    end = "; the table gives the values without the offset."
    if len(series) == len(set(names)) == len(ESTIMATE_SOURCES):
        size = abs(_reliability_shift(0, 2))
        return (f"In the figure the markers of the two evidence modes are offset sideways by {size:g} on the "
                f"probability axis, {names[0]} to the left and {names[1]} to the right, so that both stay visible"
                f"{end}")
    routes = {arm: next(panel.route for panel in panels if panel.arm == arm) for arm, _ in series}
    shifts = "; ".join(f"{name} on {routes[arm]} by {_reliability_shift(index, len(series)):+g}"
                       for index, ((arm, _), name) in enumerate(zip(series, names, strict=True)))
    return (f"In the figure the markers are offset sideways on the probability axis so that all stay visible "
            f"({shifts}){end}")


def _reliability_sources(panels: Sequence[ReliabilityPanel]) -> tuple[str, ...]:
    kinds = {panel.kind for panel in panels}
    return (*((FRED_SOURCE,) if "macro" in kinds else ()), *((EDGAR_SOURCE,) if kinds - {"macro"} else ()),
            FMP_STUDY_SOURCE)


def reliability_table(panels: Iterable[ReliabilityPanel]) -> FigureTable:
    """Figure 2's table (one row per bin of a drawn panel, one row per other panel with its reason) and caption."""
    panels = _checked_reliability_panels(panels)
    rows = []
    for panel in panels:
        ece = "insufficient" if panel.status == INSUFFICIENT else _number(panel.ece)
        head = (panel.route, panel.kind, panel.source, "; ".join(variant_label(name) for name in panel.variants),
                panel.status,
                _number(panel.holdout_prompts), _number(panel.holdout_draws), _number(panel.n_memorized),
                _number(panel.n_control), ece, _number(panel.auc), _number(panel.auc_low), _number(panel.auc_high),
                _number(panel.is_weak))
        reason = published_reason(panel.reason)
        if not panel.bins:
            rows.append((*head, "", "", "", "", "", "", "", reason))
        for bin_ in panel.bins:
            low, high = bin_interval(bin_)
            rows.append((*head, _number(bin_.low), _number(bin_.high), _number(bin_.count),
                         _number(bin_.mean_predicted), _number(bin_.observed_share), _number(low), _number(high),
                         reason))
    return FigureTable(name=RELIABILITY, caption=_reliability_caption(panels), columns=RELIABILITY_COLUMNS,
                       rows=tuple(rows), notes=_reliability_notes(panels), sources=_reliability_sources(panels))


def _draw_reliability(axes: Axes, kind: str, panels: Sequence[ReliabilityPanel],
                      series: Sequence[tuple[str, str]]) -> None:
    """One calibration set: the diagonal, and per drawn calibrator its bins as markers with their interval bars."""
    axes.plot([0.0, 1.0], [0.0, 1.0], color=_REFERENCE_GREY, linewidth=_REFERENCE_PT, linestyle="-", marker="None",
              zorder=1, gid=f"reliability-diagonal-{kind}")
    nan = math.nan
    for panel in panels:
        if panel.kind != kind or panel.status != DRAWN:
            continue
        index = series.index((panel.arm, panel.source))
        style = {**series_style(index), "linestyle": "none"}
        shift = _reliability_shift(index, len(series))
        xs = [bin_.mean_predicted + shift for bin_ in panel.bins]
        intervals = [bin_interval(bin_) for bin_ in panel.bins]
        # One capped bar per bin; the breaks keep a bar from joining the next bin.
        axes.plot([value for x in xs for value in (x, x, nan)],
                  [value for low, high in intervals for value in (low, high, nan)], color=style["color"],
                  linestyle="-", marker="_", markeredgecolor=style["color"], zorder=2, clip_on=False,
                  gid=f"reliability-interval-{panel.key}")
        axes.plot(xs, [bin_.observed_share for bin_ in panel.bins], zorder=3, clip_on=False,
                  gid=f"reliability-{panel.key}", **style)
    ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    axes.set_xlim(-0.04, 1.04)
    axes.set_ylim(-0.04, 1.04)
    axes.set_xticks(ticks, labels=["0", "0.25", "0.5", "0.75", "1"])
    axes.set_yticks(ticks, labels=["0", "0.25", "0.5", "0.75", "1"])
    axes.set_box_aspect(1.0)
    axes.set_xlabel("mean predicted recall probability\nin the bin (probability)")
    axes.set_ylabel("observed share of memorized draws\nin the bin (share)")


def plot_reliability(panels: Sequence[ReliabilityPanel]) -> Figure:
    """Figure 2: one reliability diagram per calibration set, the held-out bins of each calibrator as markers.

    In the academic style: the x axis is the mean predicted recall probability
    in a bin, the y axis the observed share of memorized draws in it with its
    Wilson score interval, the thin grey diagonal is perfect calibration. No
    line joins the bins, so none bridges an empty bin; the evidence modes
    differ by marker and stand a small distance apart on the x axis, so that
    coinciding bins stay visible. With several calibration sets the panels are
    labelled (a), (b) above their plot areas, and one legend below them names
    the markers. ``panels`` are plain data (:func:`reliability_panels`); a
    missing or insufficient calibrator is not drawn, and the caption and the
    table say why. Raises :class:`FigureError` when every panel is missing.
    Returns a matplotlib Figure (no pyplot, no global state).
    """
    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    panels = _checked_reliability_panels(panels)
    if all(panel.status == MISSING for panel in panels):
        raise FigureError("nothing to draw: every reliability panel is missing (see the table)")
    kinds = _kinds(panels)
    series = _series_keys(panels)
    several = len({arm for arm, _ in series}) > 1
    with matplotlib.rc_context(figure_rc()):
        width = FIGURE_WIDTH_IN if len(kinds) > 1 else FIGURE_WIDTH_IN / 2.0
        figure = Figure(figsize=(width, _RELIABILITY_HEIGHT_IN), layout="constrained")
        cells = list(figure.subplots(1, len(kinds), squeeze=False).ravel())
        for index, (axes, kind) in enumerate(zip(cells, kinds, strict=True)):
            _draw_reliability(axes, kind, panels, series)
            if len(kinds) > 1:
                label_panel(axes, index)
        handles = []
        for index, (arm, source) in enumerate(series):
            # The stored count of the drawn calibrators of this series; a series whose calibrators state different
            # counts is labelled without one.
            found = [panel for panel in panels if (panel.arm, panel.source) == (arm, source) and panel.status == DRAWN]
            counts = {panel.alternatives for panel in found}
            name = evidence_label(source, counts.pop() if len(counts) == 1 else None)
            if several:
                name = f"{name}, {found[0].route}"
            handles.append(Line2D([], [], label=name, **{**series_style(index), "linestyle": "none"}))
        handles.append(Line2D([], [], color=_REFERENCE_GREY, linewidth=_REFERENCE_PT, linestyle="-", marker="None",
                              label="perfect calibration"))
        figure.legend(handles=handles, loc="outside lower center", ncols=1 if len(kinds) < 2 else len(handles))
        lay_out(figure)
    return figure


# ---------- outputs ----------

def _checked_arms(arms: Sequence[str]) -> tuple[str, ...]:
    if isinstance(arms, str):
        arms = (arms,)
    arms = tuple(arms)
    if not arms:
        raise FigureError("at least one arm is needed")
    for arm in arms:
        if arm not in ARMS:
            raise FigureError(f"unknown arm {arm!r}; expected one of {list(ARMS)}")
        if arms.count(arm) > 1:
            raise FigureError(f"the {arm} arm is given twice")
    return arms


def figure_paths(name: str, out_dir: str | Path = FIGURES_DIR) -> dict[str, Path]:
    """The three files of figure ``name`` in ``out_dir``: the SVG, the CSV and the markdown, in that order."""
    if name not in PUBLISHED_FIGURES:
        raise FigureError(f"unknown figure {name!r}; expected one of {list(PUBLISHED_FIGURES)}")
    out = Path(out_dir)
    return {"svg": out / f"{name}.svg", "csv": out / f"{name}.csv", "markdown": out / f"{name}.md"}


def figure_texts(name: str, table: FigureTable, draw: Callable[[], Figure],
                 out_dir: str | Path = FIGURES_DIR) -> dict[Path, str]:
    """The three files of one figure as text by path: ``draw()`` rendered as SVG, the table as CSV and markdown."""
    paths = figure_paths(name, out_dir)
    return {paths["svg"]: figure_svg(draw()), paths["csv"]: table.to_csv(), paths["markdown"]: table.to_markdown()}


def render_figures(plan: StudyPlan, *, arms: Sequence[str] = DEFAULT_ARMS, figures: Sequence[str] = FIGURE_NAMES,
                   analysis_dir: str | Path = ANALYSIS_DIR, calibrators_dir: str | Path = CALIBRATORS_DIR,
                   out_dir: str | Path = FIGURES_DIR) -> tuple[Path, ...]:
    """Render ``figures`` for ``arms`` from the stored analyses and calibrators of ``plan`` and write them once.

    Every figure is built before any file is written, so a refusal leaves the
    output directory unchanged. Returns the written (or already identical)
    paths, per figure in the order of :func:`figure_paths`.
    """
    arms = _checked_arms(arms)
    names = tuple(figures)
    for name in names:
        if name not in FIGURE_NAMES or names.count(name) > 1:
            raise FigureError(f"figures must be distinct names of {list(FIGURE_NAMES)}; got {list(names)}")
    texts: dict[Path, str] = {}
    for name in FIGURE_NAMES:
        if name not in names:
            continue
        if name == GAP:
            gap = tuple(read_gap_panel(analysis_path(arm, CONFIRMATORY_STAGE, analysis_dir),
                                       plan_sha256=plan.plan_sha256, route=route_label(plan, arm)) for arm in arms)
            texts.update(figure_texts(GAP, gap_table(gap), lambda panels=gap: plot_gap(panels), out_dir))
        else:
            found = reliability_panels(plan, arms=arms, calibrators_dir=calibrators_dir)
            texts.update(figure_texts(RELIABILITY, reliability_table(found),
                                      lambda panels=found: plot_reliability(panels), out_dir))
    return write_outputs(texts)


def main(argv: Sequence[str] | None = None) -> int:
    """Render the accuracy-gap and reliability figures of the frozen plan from its stored outputs.

    Exit codes: 0 written (or already written identically), 2 refused (another
    plan, a missing or malformed analysis, a foreign calibrator, nothing to
    draw, or a different figure at the output path).
    """
    parser = argparse.ArgumentParser(description="Accuracy-gap and reliability figures, from the stored "
                                                 "pre-registered analyses and calibrators; sends no request.")
    parser.add_argument("--plan", type=Path, required=True, help="the frozen plan.json")
    parser.add_argument("--plan-sha256", required=True, help="the plan hash the analyses and calibrators carry")
    parser.add_argument("--arm", action="append", choices=ARMS, help="an arm of the plan to draw (repeatable; "
                                                                      "default both, serving first)")
    parser.add_argument("--figure", action="append", choices=FIGURE_NAMES, help="a figure to render (repeatable; "
                                                                                 "default all)")
    parser.add_argument("--analysis-dir", type=Path, default=ANALYSIS_DIR)
    parser.add_argument("--calibrators-dir", type=Path, default=CALIBRATORS_DIR)
    parser.add_argument("--out-dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args(argv)
    try:
        plan = load_plan(args.plan, expected_sha256=args.plan_sha256)
        paths = render_figures(plan, arms=args.arm or DEFAULT_ARMS, figures=args.figure or FIGURE_NAMES,
                               analysis_dir=args.analysis_dir, calibrators_dir=args.calibrators_dir,
                               out_dir=args.out_dir)
    except (ValueError, TypeError, OSError) as exc:
        print(f"figures: refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    for path in paths:
        print(path)
    return EXIT_WRITTEN


__all__ = [
    "ADJUSTMENT_NAMES",
    "CAPTION_LIMIT",
    "CONVERGENCE",
    "DEFAULT_ARMS",
    "DERIVED_ONLY",
    "DRAWN",
    "DRAWS",
    "DRAW_SPREAD",
    "EVIDENCE_NAMES",
    "FIGURES_DIR",
    "FIGURE_NAMES",
    "FIGURE_NUMBERS",
    "GAP",
    "GAP_COLUMNS",
    "INSUFFICIENT",
    "INTERVAL_WORDING",
    "MISSING",
    "PUBLISHED_FIGURES",
    "RELIABILITY",
    "RELIABILITY_COLUMNS",
    "RELIABILITY_CONFIDENCE",
    "SVG_HASH_SALT",
    "FigureError",
    "FigureTable",
    "GapPanel",
    "GapPoint",
    "HeldOutReliability",
    "ReliabilityPanel",
    "bin_interval",
    "evidence_label",
    "figure_paths",
    "figure_rc",
    "figure_svg",
    "figure_texts",
    "freeze_layout",
    "gap_panel",
    "gap_table",
    "held_out_reliability",
    "interval_wording",
    "lay_out",
    "main",
    "plot_gap",
    "plot_reliability",
    "read_gap_panel",
    "reliability_panel",
    "reliability_panels",
    "reliability_table",
    "render_figures",
    "rounded",
    "route_label",
    "short_routes",
    "variant_label",
    "write_outputs",
]


if __name__ == "__main__":
    raise SystemExit(main())
