"""Accuracy-gap and reliability figures of the worked example (task 8.2).

Requirements 10.2, 10.3 and 10.6 (design: Figures, F1 and F2). No request is
sent and nothing is recomputed from item results: every number comes from a
stored output of an earlier task.

F1 ``gap`` (Requirement 10.2). Read from the pre-registered analysis JSON of
task 7.10 (``runs/finance-sim/analysis/pre-registered-<arm>-full_run.json``,
:func:`~scripts.finance_sim.analysis.analysis_path`), never recomputed: per
variant, the pre-cutoff minus post-cutoff difference of identifying minus
anonymised accuracy (``did_accuracy_gap``) with its moving-block bootstrap
interval, raw and Holm-adjusted p-values, month counts, ``dropped_resamples``
and ``missing_months``, beside the two period differences that make it up.
Dot and interval, one panel per arm (small multiples, shared x axis): the
serving arm's none and top-k variants are the pre-registered tests, the probe
arm's six variants a small paired check; the arm's achieved power is reported
with its label and its Monte Carlo standard error, and the conditional note of
the analysis stands beside any probe-arm panel or any variant with dropped
resamples. A variant the analysis reports as missing (the serving arm's
full-distribution variants, too few months) is listed in the table and the
caption and not drawn.

F2 ``reliability`` (Requirement 10.3). Read from the calibrator files of task
7.6 (``runs/finance-sim/calibrators/<arm>/<source>/<macro|edgar>.json``, loaded
with :func:`~scripts.finance_sim.calibrate.load_calibrator` against the frozen
plan): the held-out probabilities and labels each calibrator retained, binned by
:func:`recall_guard.harness.plots.reliability_curve` with the bins of the local
report's held-out calibration and the same gate
(``CALIBRATION_MIN_PER_CLASS`` held-out draws per class); below it the panel says
"insufficient", where the report shows null. One calibrator per arm, kind and
estimate source serves both draw modes of its adjustment, so each panel names
the two variants it serves; the no-adjustment variants carry no estimate. A
calibrator the arm cannot have (the serving arm's full distribution) or one not
fitted yet is listed as missing and not drawn.

Outputs (design: Figures layout). Each figure is written to
``docs/assets/finance-sim/`` (:data:`FIGURES_DIR`) as ``<figure>-light.svg``,
``<figure>-dark.svg``, ``<figure>.csv`` and ``<figure>.md``, the markdown being
the CSV rendered below the caption line that states the data, the sample size
and the interval method. Tables carry derived statistics only: no reply text,
no realized returns, no raw licensed series; the captions cite Moody's for
BAA10Y and FMP for the prices behind the outcomes. A figure's files are
written once: the same output again is a no-op and a different one is refused,
never replaced (remove the old files first). Renders are byte-stable: no
timestamp in the SVG metadata, a fixed SVG hash salt, glyphs as paths.

Style (task 8.1, recall_guard.harness.plots). The validated ``STUDY_PALETTE`` and
``STUDY_PALETTE_DARK`` through :func:`~recall_guard.harness.plots.figure_theme`;
identity never by colour alone: marker shape and fill from
:func:`~recall_guard.harness.plots.variant_style`, direct labels (the rows of F1,
the panel titles of F2), at most ``ALL_PAIRS_SLOT_CAP`` colours where marks sit
side by side, and the ``muted`` token applied explicitly to the zero line and the
diagonal. The design places ``plot_gap`` and ``plot_reliability`` in plots.py;
this module holds them instead, beside their data, because plots.py is extended
by task 8.3.

Matplotlib is imported inside the functions that draw, so importing this module
(or ``recall_guard``) never imports it.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
import sys
import textwrap
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from recall_guard.harness.local_reporting import CALIBRATION_BINS, CALIBRATION_MIN_PER_CLASS
from scripts.finance_sim.analysis import (
    ACCURACY_POST,
    ACCURACY_PRE,
    ACHIEVED_POWER,
    ANALYSIS_DIR,
    ANALYSIS_SCHEMA,
    COMPUTED,
    CONDITIONAL_NOTE,
    DID,
    analysis_path,
)
from scripts.finance_sim.calibrate import (
    CALIBRATORS_DIR,
    HOLDOUT_FRACTION,
    CalibrationRecord,
    calibrator_path,
    load_calibrator,
)
from scripts.finance_sim.limits import ARMS, SAMPLING_ARM, SERVING_ARM
from scripts.finance_sim.pilot import CONFIRMATORY_STAGE
from scripts.finance_sim.plan import (
    CALIBRATION_KINDS,
    ESTIMATE_SOURCES,
    PRIMARY_ROLE,
    PROBE_ROLE,
    ROLES,
    VARIANTS,
    StudyPlan,
    load_plan,
    variant_parts,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from recall_guard.harness.plots import FigureTheme, ReliabilityBin

ROOT = Path(__file__).resolve().parents[2]
#: The docs figure folder (design: Figures layout); tests write under tmp_path instead.
FIGURES_DIR = ROOT / "docs" / "assets" / "finance-sim"
GAP, RELIABILITY = "gap", "reliability"
#: The figures this module renders, in render order (task 8.3 adds ``draws`` and ``convergence``).
FIGURE_NAMES: tuple[str, ...] = (GAP, RELIABILITY)
#: Panel order: the primary (serving) arm first, then the probe (sampling) arm.
DEFAULT_ARMS: tuple[str, ...] = (SERVING_ARM, SAMPLING_ARM)
#: Figure themes in render order, as ``recall_guard.harness.plots.THEMES`` (not imported: matplotlib stays lazy).
THEMES: tuple[str, ...] = ("light", "dark")
EXIT_WRITTEN, EXIT_REFUSED = 0, 2
#: Panel statuses of F2: drawn with its bins, shown without bins below the report's gate, or not drawn.
DRAWN, INSUFFICIENT, MISSING = "drawn", "insufficient", "missing"
#: Fixed salt of the SVG element ids, so a rerender is byte-identical.
SVG_HASH_SALT = "memorization-adjusted-finance-sim"

#: What each arm role supplies to the figure (plan: ARM_SUPPLIES, design "Arm sizing decision").
ROLE_PURPOSE: Mapping[str, str] = MappingProxyType({PRIMARY_ROLE: "the pre-registered tests",
                                                    PROBE_ROLE: "a small paired check"})
#: Text names of the adjustments, as the direct labels of ``variant_style`` spell them.
ADJUSTMENT_NAMES: Mapping[str, str] = MappingProxyType({"none": "no adjustment", "top_k": "top-k",
                                                        "full_distribution": "full distribution"})
#: Text names of the calibration kinds and the control each calibrator adjusts.
KIND_NAMES: Mapping[str, str] = MappingProxyType({"macro": "Macro calibrator, positive control",
                                                  "edgar": "EDGAR calibrator, negative control"})
DERIVED_ONLY = "Derived statistics only: no reply text, no realized returns and no raw licensed series."
FRED_SOURCE = ("macro rows from FRED (Federal Reserve Bank of St. Louis) as known on each decision date, including "
               "BAA10Y, Moody's Seasoned Baa Corporate Bond Yield Relative to Yield on 10-Year Treasury Constant "
               "Maturity, copyright Moody's, cited and not reproduced")
FMP_SOURCE = ("realized next-month returns from Financial Modeling Prep (FMP) end-of-day adjusted closes (licensed, "
              "not published) with DTB3 from FRED as the cash leg")
EDGAR_SOURCE = "EDGAR outlook prompts from SEC EDGAR filings"
#: F2 uses no price: FMP is cited for the study's outcomes, as the design cites Moody's and FMP in every caption.
FMP_STUDY_SOURCE = ("the study's outcomes come from Financial Modeling Prep (FMP) end-of-day adjusted closes "
                    "(licensed, not published), and no FMP value enters this figure")

GAP_COLUMNS: tuple[str, ...] = (
    "arm", "role", "variant", "draw_mode", "adjustment", "gap", "interval_low", "interval_high", "p_raw", "p_holm",
    "months", "pre_cutoff_months", "post_cutoff_months", "missing_months", "dropped_resamples",
    "accuracy_difference_pre_cutoff", "accuracy_difference_post_cutoff", "statuses", "missing")
RELIABILITY_COLUMNS: tuple[str, ...] = (
    "arm", "role", "kind", "source", "variants", "status", "holdout_prompts", "holdout_draws", "memorized_draws",
    "control_draws", "ece", "holdout_auc", "auc_interval_low", "auc_interval_high", "weak", "bin_low", "bin_high",
    "bin_count", "mean_predicted", "observed_share", "reason")

#: Chrome shared by both figures, on top of the theme's rcParams (fonts ship with matplotlib, glyphs as paths).
_RC: Mapping[str, Any] = MappingProxyType({
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 8.0, "axes.titlesize": 8.0,
    "axes.labelsize": 8.0, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "figure.titlesize": 9.5, "figure.labelsize": 8.0, "axes.linewidth": 0.8, "svg.fonttype": "path",
    "svg.hashsalt": SVG_HASH_SALT})
_WIDTH_IN = 7.2
_TITLE_LINE_IN = 0.15
_GAP_ROW_IN, _GAP_PANEL_IN, _GAP_FRAME_IN = 0.3, 0.3, 1.25
_GROUP_GAP = 0.5
_RELIABILITY_PANEL_IN, _RELIABILITY_FRAME_IN = 2.95, 0.9
_MARKER_PT, _MARKER_EDGE_PT, _INTERVAL_PT = 7.0, 1.5, 1.8
_AREA_MIN, _AREA_MAX = 36.0, 320.0
#: Count labels: gap between marker edge and label, half the marker's surface ring, and a zorder above the
#: diagonal (1) and below the markers (3); a label keeps this much space from the diagonal, so it reads clear.
_COUNT_GAP_PT, _MARKER_RING_PT, _COUNT_ZORDER = 2.0, 0.75, 2.5
_LINE_CLEARANCE_PT = 3.0
_HEX64 = re.compile(r"[0-9a-f]{64}")


class FigureError(ValueError):
    """An input or output the figures refuse; nothing is written for it."""


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


@dataclass(frozen=True)
class FigureTable:
    """The table of one figure: derived statistics in rows of text, with the caption above it in markdown."""

    name: str
    caption: str
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]

    def to_csv(self) -> str:
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(self.columns)
        writer.writerows(self.rows)
        return buffer.getvalue()

    def to_markdown(self) -> str:
        """The CSV rendered as a markdown table below the caption line."""

        def line(values: Iterable[str]) -> str:
            return "| " + " | ".join(_markdown_cell(value) for value in values) + " |"

        lines = [_flat(self.caption), "", line(self.columns), "|" + "|".join("---" for _ in self.columns) + "|"]
        lines += [line(row) for row in self.rows]
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


def _theme_rc(theme: str) -> dict[str, Any]:
    from recall_guard.harness.plots import figure_theme

    return {**figure_theme(theme).rc_params(), **_RC}


def _style_axes(axes: Axes, style: FigureTheme) -> None:
    """Recessive chrome set on the axes itself, so ticks created later keep it: hairline grid, no box."""
    axes.tick_params(colors=style.baseline, labelcolor=style.secondary_ink, labelsize=_RC["xtick.labelsize"])
    axes.grid(color=style.gridline, linewidth=0.6, linestyle="-")
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(style.baseline)


# ---------- F1: the accuracy gap, from the stored pre-registered analysis ----------

@dataclass(frozen=True)
class GapPoint:
    """One variant's accuracy gap as the stored analysis gives it, or the reason it has none (``missing``).

    ``statistic`` is the pre-cutoff minus post-cutoff difference of identifying
    minus anonymised accuracy, ``low`` and ``high`` its interval, ``p_value``
    and ``p_holm`` its raw and Holm-adjusted two-sided p-values. ``months``
    counts the months used (``pre_cutoff_months`` and ``post_cutoff_months`` by
    period), ``missing_months`` the planned months without the value in both
    renderings, ``dropped_resamples`` the resamples whose period cell was empty.
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

    @property
    def drawn(self) -> bool:
        return self.missing is None


@dataclass(frozen=True)
class GapPanel:
    """One arm's panel of F1: the arm, its role, its months, the analysis settings, its points and its power."""

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
    low = high = None
    if interval is not None:
        low, high = _real(interval["lo"], f"{variant} interval lo"), _real(interval["hi"], f"{variant} interval hi")
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
        missing=missing)


def gap_panel(data: Mapping[str, Any], *, plan_sha256: str | None = None) -> GapPanel:
    """The F1 panel of one stored pre-registered analysis (the 7.10 JSON payload), read and checked, not recomputed.

    Refuses another schema, an arm outside the study, a stage other than the
    full run, another plan than ``plan_sha256`` (when given), and an analysis
    without a computed ``did_accuracy_gap``.
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
            plan_sha256=sha, arm=arm, role=role, model=_text(data["model"], "model"),
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


def read_gap_panel(path: str | Path, *, plan_sha256: str | None = None) -> GapPanel:
    """The F1 panel of the analysis JSON at ``path`` (duplicate keys and non-finite numbers refused)."""
    return gap_panel(_read_json(Path(path)), plan_sha256=plan_sha256)


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
    return panels


def _statuses(point: GapPoint) -> str:
    return "; ".join(f"{status} {count}" for status, count in point.statuses.items())


def _power_text(panel: GapPanel) -> str:
    if panel.power is None:
        return f"achieved power missing: {panel.power_reason}"
    text = f"achieved power {panel.power:.3f}"
    if panel.power_effect_size is not None:
        text += f" for the pre-registered effect {panel.power_effect_size:g}"
    error = panel.power_monte_carlo_error
    if error is not None:
        text += f" (Monte Carlo standard error {error:.3f} over {panel.power_n_sim} simulations)"
    return text


def _grouped(entries: Iterable[tuple[str, str]]) -> str:
    """``key, key: reason; key: reason`` with the keys of one reason together, in order of first appearance."""
    groups: dict[str, list[str]] = {}
    for key, reason in entries:
        groups.setdefault(reason, []).append(key)
    return "; ".join(f"{', '.join(keys)}: {reason}" for reason, keys in groups.items())


def _gap_caption(panels: Sequence[GapPanel]) -> str:
    settings = panels[0].settings
    models = ", ".join(dict.fromkeys(panel.model for panel in panels))
    parts = [
        "Figure F1. Accuracy gap before and after adjustment, positive control (identifying versus anonymised "
        "FRED macro prompts).",
        f"Data: the stored pre-registered analysis (task 7.10; read, not recomputed) of the {CONFIRMATORY_STAGE} "
        f"stage, model {models}, plan {panels[0].plan_sha256[:12]}, pilot items excluded; per variant, the "
        "pre-cutoff minus post-cutoff difference of identifying minus anonymised accuracy (did_accuracy_gap), the "
        "no-adjustment variants beside the top-k and full-distribution adjustments.",
    ]
    for panel in panels:
        parts.append(f"{panel.arm.capitalize()} arm ({panel.role} role, {ROLE_PURPOSE.get(panel.role, panel.role)}): "
                     f"{panel.planned_months} planned months ({panel.pre_cutoff_months} on or before the model's "
                     f"cutoff, {panel.post_cutoff_months} after); {_power_text(panel)}.")
    approximations = list(dict.fromkeys(panel.power_approximation for panel in panels if panel.power_approximation))
    parts += [f"Achieved power is read as {approximation}." for approximation in approximations]
    parts += [
        "Sample size: months per variant in the table (a month counts when both renderings carry the value; the "
        "months without it are counted per variant, with their reasons in the stored analysis).",
        f"Interval: {_percent(settings['confidence'])} percent non-circular moving-block percentile bootstrap over "
        f"months, block length {settings['block_length']} months, {settings['n_resamples']} resamples, seed "
        f"{settings['seed']}; p raw is two-sided from the kept resamples, Holm-adjusted across the variants of each "
        f"arm (alpha {settings['alpha']:g}); the two period differences are measurements without a p-value.",
        "Encoding: marker shape gives the adjustment (circle no adjustment, square top-k, triangle full "
        "distribution), an open marker a single draw and a filled one the ensemble; each row is labelled; the "
        "vertical line marks no gap.",
    ]
    missing = []
    for panel in panels:
        reasons: dict[str, list[str]] = {}
        for point in panel.points:
            if not point.drawn:
                reasons.setdefault(_flat(point.missing), []).append(point.variant)
        missing += [f"{panel.arm} {', '.join(variants)}: {reason}" for reason, variants in reasons.items()]
    if missing:
        parts.append(f"Missing (not drawn): {'; '.join(missing)}.")
    if any(panel.role == PROBE_ROLE or panel.max_dropped for panel in panels):
        dropped = "; ".join(f"{panel.arm}: up to {panel.max_dropped} of {settings['n_resamples']}" for panel in panels)
        parts.append(f"{CONDITIONAL_NOTE} Dropped resamples per variant are in the table ({dropped}).")
    parts += [f"Sources: {FRED_SOURCE}; {FMP_SOURCE}.", DERIVED_ONLY]
    return " ".join(parts)


def gap_table(panels: Iterable[GapPanel]) -> FigureTable:
    """F1's table (one row per arm and variant, missing ones included with the reason) and its caption."""
    panels = _checked_gap_panels(panels)
    rows = []
    for panel in panels:
        for point in panel.points:
            rows.append((
                panel.arm, panel.role, point.variant, point.mode, point.adjustment, _number(point.statistic),
                _number(point.low), _number(point.high), _number(point.p_value), _number(point.p_holm),
                _number(point.months), _number(point.pre_cutoff_months), _number(point.post_cutoff_months),
                _number(point.missing_months), _number(point.dropped_resamples), _number(point.pre_difference),
                _number(point.post_difference), _statuses(point), "" if point.missing is None else _flat(point.missing)))
    return FigureTable(name=GAP, caption=_gap_caption(panels), columns=GAP_COLUMNS, rows=tuple(rows))


def _gap_title(panel: GapPanel, n_resamples: int) -> str:
    lines = [f"{panel.arm.capitalize()} arm, {panel.role} role: {ROLE_PURPOSE.get(panel.role, panel.role)}",
             f"{panel.planned_months} planned months: {panel.pre_cutoff_months} pre-cutoff, "
             f"{panel.post_cutoff_months} post-cutoff",
             "Achieved power missing (see the caption)" if panel.power is None
             else f"Achieved power {panel.power:.2f} (t-test approximation)"]
    absent = [point for point in panel.points if not point.drawn]
    if absent:
        names = ", ".join(dict.fromkeys(ADJUSTMENT_NAMES.get(point.adjustment, point.adjustment) for point in absent))
        noun = "variant" if len(absent) == 1 else "variants"
        lines.append(f"Not drawn: {len(absent)} missing {noun} ({names}); see the table")
    if panel.max_dropped:
        lines.append(f"Intervals conditional on the kept resamples: up to {panel.max_dropped} of {n_resamples} "
                     "dropped; see the caption")
    return "\n".join(lines)


def _gap_positions(points: Sequence[GapPoint]) -> list[float]:
    """Row positions top to bottom, with a little space between the single-draw and the ensemble rows."""
    positions: list[float] = []
    previous: str | None = None
    position = 0.0
    for point in points:
        if previous is not None:
            position += 1.0 + (_GROUP_GAP if point.mode != previous else 0.0)
        positions.append(position)
        previous = point.mode
    return positions


def _gap_limit(panels: Sequence[GapPanel]) -> float:
    values = [abs(value) for panel in panels for point in panel.drawn_points
              for value in (point.statistic, point.low, point.high) if value is not None]
    return max(max(values, default=0.0) * 1.1, 0.05)


def _draw_gap_panel(axes: Axes, panel: GapPanel, style: FigureTheme, theme: str, limit: float,
                    n_resamples: int) -> None:
    from recall_guard.harness.plots import variant_style

    _style_axes(axes, style)
    axes.grid(axis="y", visible=False)
    axes.spines["left"].set_visible(False)
    axes.tick_params(axis="y", length=0)
    axes.axvline(0.0, color=style.muted, linewidth=1.0, zorder=1, gid=f"gap-zero-{panel.arm}")
    points = panel.drawn_points
    positions = _gap_positions(points)
    labels = []
    for position, point in zip(positions, points, strict=True):
        variant = variant_style(point.adjustment, point.mode, theme=theme)
        labels.append(variant.label)
        if point.low is not None and point.high is not None:
            axes.hlines(position, point.low, point.high, colors=variant.color, linewidth=_INTERVAL_PT, zorder=2,
                        gid=f"gap-interval-{panel.arm}-{point.variant}")
        marker = {key: value for key, value in variant.marker_kwargs().items() if key != "label"}
        axes.plot([point.statistic], [position], markersize=_MARKER_PT, markeredgewidth=_MARKER_EDGE_PT, zorder=3,
                  gid=f"gap-{panel.arm}-{point.variant}", **marker)
    if points:
        axes.set_yticks(positions, labels=labels)
        axes.set_ylim(positions[-1] + 0.7, positions[0] - 0.7)
    else:
        axes.set_yticks([])
        axes.set_ylim(1.0, 0.0)
        axes.text(0.5, 0.5, "No variant drawn: every variant is missing (see the table)", transform=axes.transAxes,
                  ha="center", va="center", color=style.secondary_ink)
    axes.set_xlim(-limit, limit)
    axes.set_title(_gap_title(panel, n_resamples), loc="left", color=style.ink)


def plot_gap(panels: Sequence[GapPanel], *, theme: str = "light") -> Figure:
    """F1: dot and interval per variant, one panel per arm on a shared x axis (Requirement 10.2).

    ``panels`` are plain data read from the stored analyses (:func:`gap_panel`);
    ``theme`` is ``light`` or ``dark``. Returns a matplotlib Figure (no pyplot,
    no global state). Missing variants are not drawn.
    """
    import matplotlib
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    from recall_guard.harness.plots import figure_theme, variant_style

    panels = _checked_gap_panels(panels)
    style = figure_theme(theme)
    n_resamples = int(panels[0].settings["n_resamples"])
    spans = [(_gap_positions(panel.drawn_points) or [0.0])[-1] + 1.4 for panel in panels]
    title_lines = [_gap_title(panel, n_resamples).count("\n") + 1 for panel in panels]
    height = sum(span * _GAP_ROW_IN + _GAP_PANEL_IN + _TITLE_LINE_IN * lines
                 for span, lines in zip(spans, title_lines, strict=True)) + _GAP_FRAME_IN
    with matplotlib.rc_context(_theme_rc(theme)):
        figure = Figure(figsize=(_WIDTH_IN, height), layout="constrained")
        grid = figure.subplots(len(panels), 1, sharex=True, squeeze=False, gridspec_kw={"height_ratios": spans})
        limit = _gap_limit(panels)
        for axes, panel in zip(grid[:, 0], panels, strict=True):
            _draw_gap_panel(axes, panel, style, theme, limit, n_resamples)
        bottom = grid[-1, 0]
        bottom.xaxis.set_major_locator(MaxNLocator(nbins=8, symmetric=True))
        bottom.set_xlabel("Pre-cutoff minus post-cutoff gap of identifying minus anonymised accuracy (share correct)")
        drawn = {point.variant for panel in panels for point in panel.drawn_points}
        by_mode: dict[str, list[Line2D]] = {"single": [], "ensemble": []}
        for name in VARIANTS:
            if name in drawn:
                mode, adjustment = variant_parts(name)
                variant = variant_style(adjustment, mode, theme=theme)
                by_mode[mode].append(Line2D([], [], markersize=_MARKER_PT, markeredgewidth=_MARKER_EDGE_PT,
                                            **variant.marker_kwargs()))
        confidence = _percent(float(panels[0].settings["confidence"]))
        key = Line2D([], [], color=style.secondary_ink, linewidth=_INTERVAL_PT, label=f"{confidence} percent interval")
        # Three equal columns, filled column by column: single-draw variants, ensemble variants, the interval key;
        # blank entries pad the shorter columns so each variant stays in its draw mode's column.
        columns = [by_mode["single"], by_mode["ensemble"], [key]]
        rows = max(len(column) for column in columns)
        handles = [handle for column in columns for handle in
                   column + [Line2D([], [], linestyle="none", marker="none", label=" ")
                             for _ in range(rows - len(column))]]
        figure.legend(handles=handles, loc="outside lower center", ncols=3, frameon=False)
        figure.suptitle("Accuracy gap before and after adjustment (positive control)", x=0.01, ha="left")
        figure.draw_without_rendering()
    return figure


# ---------- F2: reliability of the memorization estimate, from the calibrators ----------

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
    """One calibrator's panel of F2: its arm, kind, source, the variants it serves, and its held-out reliability.

    ``status`` is ``drawn`` (with ``bins`` and ``ece``), ``insufficient``
    (shown without bins, the reason gives the counts) or ``missing`` (not
    drawn, with the reason). ``holdout_prompts`` and ``holdout_draws`` give the
    sample, ``n_memorized`` and ``n_control`` its classes; ``auc`` with
    ``auc_low`` and ``auc_high`` is the held-out AUC with the interval the
    calibration reports (``auc_method``); ``is_weak`` marks a calibrator the
    guard does not apply.
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

    @property
    def key(self) -> str:
        return f"{self.arm}-{self.kind}-{self.source}"

    @property
    def label(self) -> str:
        return f"{self.arm} {self.kind} {self.source}"


def _served_variants(source: str) -> tuple[str, ...]:
    """The variants one estimate source's calibrator serves: both draw modes of that adjustment."""
    return tuple(variant for variant in VARIANTS if variant_parts(variant)[1] == source)


def reliability_panel(record: CalibrationRecord, plan: StudyPlan) -> ReliabilityPanel:
    """The F2 panel of one loaded calibrator record of ``plan`` (its retained held-out pairs, gated)."""
    if not isinstance(record, CalibrationRecord):
        raise TypeError(f"record must be a CalibrationRecord; got {type(record).__name__}")
    if record.plan_sha256 != plan.plan_sha256:
        raise FigureError(f"the {record.arm} {record.kind} {record.source.value} calibrator belongs to plan "
                          f"{record.plan_sha256[:12]}, not {plan.plan_sha256[:12]}")
    source = record.source.value
    mcs = record.calibrator.mcs
    interval = record.holdout_auc_interval_by_prompt
    method = (f"{_percent(float(interval['confidence']))} percent {interval['method']}, {interval['n_resamples']} "
              f"resamples, seed {interval['seed']}")
    base: dict[str, Any] = {
        "plan_sha256": record.plan_sha256, "arm": record.arm, "role": plan.arms.arms[record.arm].role,
        "kind": record.kind, "source": source, "variants": _served_variants(source),
        "holdout_prompts": record.counts.get("holdout_prompts"), "is_weak": record.calibrator.is_weak,
        "auc": _real(interval["auc"], "auc"), "auc_low": _real(interval["ci_low"], "ci_low"),
        "auc_high": _real(interval["ci_high"], "ci_high"), "auc_method": method,
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
    """Every F2 panel of ``arms`` by kind and estimate source, from the calibrator files of the frozen plan.

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
                base = {"plan_sha256": plan.plan_sha256, "arm": arm, "role": arm_plan.role, "kind": kind,
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


def _reliability_caption(panels: Sequence[ReliabilityPanel]) -> str:
    shown = [panel for panel in panels if panel.status != MISSING]
    parts = [
        "Figure F2. Reliability of the memorization estimate per variant, on held-out calibration draws.",
        f"Data: the held-out draws each calibrator retained when it was fitted (task 7.6), plan "
        f"{panels[0].plan_sha256[:12]}: p_memorized of the draws of the held-out prompts (a stratified "
        f"{HOLDOUT_FRACTION * 100:g} percent split by prompt) against their labels (1 memorized, 0 control); macro is "
        "the FRED macro calibration set (the calibrator of the positive control), edgar the SEC EDGAR calibration "
        "releases (the calibrator of the negative control).",
        "One calibrator per arm, kind and estimate source serves both draw modes of its adjustment (the single-draw "
        "and the ensemble variant); the no-adjustment variants carry no estimate.",
    ]
    if shown:
        sizes = "; ".join(f"{panel.label}: n {panel.holdout_draws} held-out draws ({panel.n_memorized} memorized, "
                          f"{panel.n_control} control) from {panel.holdout_prompts} prompts" for panel in shown)
        parts.append(f"Sample size: {sizes}; the draws of one prompt are not independent.")
    parts += [
        f"Bins: {CALIBRATION_BINS} equal-width bins over [0, 1], empty bins omitted, marker area growing with the "
        "count printed beside each marker; ECE is the count-weighted mean absolute difference between the mean "
        "predicted p_memorized and the observed memorized share, with the binning of the local report's held-out "
        "calibration (recall_guard.harness.local_reporting.held_out_calibration), computed only with at least "
        f"{CALIBRATION_MIN_PER_CLASS} held-out draws per class, otherwise insufficient; the diagonal marks perfect "
        "calibration.",
    ]
    methods = list(dict.fromkeys(panel.auc_method for panel in shown if panel.auc_method))
    auc = f"the held-out AUC in the table carries its interval: {'; '.join(methods)}" if methods else (
        "no held-out AUC interval is recorded")
    parts.append(f"Interval: none on the bins; {auc}.")
    parts.append("A weak calibrator (held-out AUC below the plan's min_auc) is not applied by the guard; the table "
                 "marks it.")
    insufficient = [(panel.label, _flat(panel.reason)) for panel in panels if panel.status == INSUFFICIENT]
    if insufficient:
        parts.append(f"Insufficient (shown, no bins): {_grouped(insufficient)}.")
    missing = [(panel.label, _flat(panel.reason)) for panel in panels if panel.status == MISSING]
    if missing:
        parts.append(f"Missing (not drawn): {_grouped(missing)}.")
    parts += [f"Sources: {FRED_SOURCE}; {EDGAR_SOURCE}; {FMP_STUDY_SOURCE}.", DERIVED_ONLY]
    return " ".join(parts)


def reliability_table(panels: Iterable[ReliabilityPanel]) -> FigureTable:
    """F2's table (one row per bin of a drawn panel, one row per other panel with its reason) and its caption."""
    panels = _checked_reliability_panels(panels)
    rows = []
    for panel in panels:
        ece = "insufficient" if panel.status == INSUFFICIENT else _number(panel.ece)
        head = (panel.arm, panel.role, panel.kind, panel.source, "; ".join(panel.variants), panel.status,
                _number(panel.holdout_prompts), _number(panel.holdout_draws), _number(panel.n_memorized),
                _number(panel.n_control), ece, _number(panel.auc), _number(panel.auc_low), _number(panel.auc_high),
                _number(panel.is_weak))
        reason = "" if panel.reason is None else _flat(panel.reason)
        if not panel.bins:
            rows.append((*head, "", "", "", "", "", reason))
        for bin_ in panel.bins:
            rows.append((*head, _number(bin_.low), _number(bin_.high), _number(bin_.count),
                         _number(bin_.mean_predicted), _number(bin_.observed_share), reason))
    return FigureTable(name=RELIABILITY, caption=_reliability_caption(panels), columns=RELIABILITY_COLUMNS,
                       rows=tuple(rows))


def _reliability_title(panel: ReliabilityPanel) -> str:
    source = ADJUSTMENT_NAMES.get(panel.source, panel.source)
    lines = [f"{panel.arm.capitalize()} arm ({panel.role}), {source}", KIND_NAMES.get(panel.kind, panel.kind)]
    ece = "ECE insufficient" if panel.ece is None else f"ECE {panel.ece:.3f}"
    lines.append(f"{ece}; n {panel.holdout_draws} draws, {panel.holdout_prompts} prompts")
    if panel.auc is not None:
        auc = f"Held-out AUC {panel.auc:.2f}"
        if panel.auc_low is not None and panel.auc_high is not None:
            auc += f" [{panel.auc_low:.2f}, {panel.auc_high:.2f}]"
        lines.append(auc + ("; weak, not applied" if panel.is_weak else ""))
    return "\n".join(lines)


def _area(count: int, largest: int) -> float:
    return _AREA_MIN + (_AREA_MAX - _AREA_MIN) * count / max(largest, 1)


def _count_candidates(x: float, y: float, offset: float) -> list[tuple[float, float, str, str]]:
    """Label offsets in points with their alignment, in order of preference.

    Straight above (below near the top edge), then the other vertical side,
    then sideways away from the diagonal (left of a bin on or above it, right
    of a bin below it), then diagonally away from it, then the other side.
    """
    up, down = (0.0, offset, "center", "bottom"), (0.0, -offset, "center", "top")
    left, right = (-offset, 0.0, "right", "center"), (offset, 0.0, "left", "center")
    lean = 0.75 * offset
    up_left, down_right = (-lean, lean, "right", "bottom"), (lean, -lean, "left", "top")
    vertical = [up, down] if y <= 0.85 else [down, up]
    away, away_diagonally, other = (left, up_left, right) if y >= x else (right, down_right, left)
    return [*vertical, away, away_diagonally, other]


def _place_counts(axes: Axes, panel: ReliabilityPanel, style: FigureTheme, largest: int, renderer: Any) -> None:
    """Print each bin's count beside its marker, clear of the diagonal, of every marker and of the panel edge.

    Runs after the layout is fixed: each candidate position of
    :func:`_count_candidates` is measured in display coordinates and the first
    one clear of all three, with ``_LINE_CLEARANCE_PT`` of space from the
    diagonal, is kept. Where none is (a bin in the frame's corner), the
    candidate kept is off the diagonal by 1 pt, then inside, then off the
    markers, and only then near the diagonal, in that order. The labels
    are left out of the layout, so placing them moves no panel, and each sits on
    a surface-coloured backing above the diagonal and below the markers.
    """
    from matplotlib.transforms import Bbox

    diagonal = next(line for line in axes.lines if line.get_gid() == f"reliability-diagonal-{panel.key}")
    path = diagonal.get_path().transformed(diagonal.get_transform())
    scale = axes.figure.dpi / 72.0
    points = [(bin_.mean_predicted, bin_.observed_share) for bin_ in panel.bins]
    centres = axes.transData.transform(points)
    halves = [(math.sqrt(_area(bin_.count, largest)) / 2.0 + _MARKER_RING_PT) * scale for bin_ in panel.bins]
    markers = [Bbox.from_extents(cx - half, cy - half, cx + half, cy + half)
               for (cx, cy), half in zip(centres, halves, strict=True)]
    frame = axes.bbox
    for index, (bin_, (x, y)) in enumerate(zip(panel.bins, points, strict=True)):
        offset = math.sqrt(_area(bin_.count, largest)) / 2.0 + _COUNT_GAP_PT
        best: tuple[tuple[bool, bool, bool, bool], Any] | None = None
        for dx, dy, ha, va in _count_candidates(x, y, offset):
            label = axes.annotate(str(bin_.count), (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha,
                                  va=va, fontsize=6.5, color=style.secondary_ink, annotation_clip=False,
                                  in_layout=False, zorder=_COUNT_ZORDER, gid=f"reliability-count-{panel.key}-{index}",
                                  bbox={"facecolor": style.surface, "edgecolor": "none", "pad": 1.0})
            text = label.get_window_extent(renderer)
            extent = text.padded(scale)
            inside = (frame.x0 <= extent.x0 and extent.x1 <= frame.x1 and frame.y0 <= extent.y0
                      and extent.y1 <= frame.y1)
            on_line = bool(path.intersects_bbox(extent, filled=False))
            near_line = bool(path.intersects_bbox(text.padded(_LINE_CLEARANCE_PT * scale), filled=False))
            violations = (on_line, not inside, any(extent.overlaps(marker) for marker in markers), near_line)
            if best is None or violations < best[0]:
                if best is not None:
                    best[1].remove()
                best = (violations, label)
            else:
                label.remove()
            if not any(violations):
                break


def _draw_reliability_panel(axes: Axes, panel: ReliabilityPanel, style: FigureTheme, theme: str,
                            largest: int) -> None:
    from recall_guard.harness.plots import variant_style

    _style_axes(axes, style)
    # Explicit marker and line style: the theme's property cycle would otherwise put markers on the diagonal.
    axes.plot([0.0, 1.0], [0.0, 1.0], color=style.muted, linewidth=1.0, linestyle="-", marker="none", zorder=1,
              gid=f"reliability-diagonal-{panel.key}")
    if panel.status == DRAWN:
        variant = variant_style(panel.source, "ensemble", theme=theme)
        sizes = [_area(bin_.count, largest) for bin_ in panel.bins]
        axes.scatter([bin_.mean_predicted for bin_ in panel.bins], [bin_.observed_share for bin_ in panel.bins],
                     s=sizes, marker=variant.marker, facecolors=variant.color, edgecolors=style.surface,
                     linewidths=1.5, zorder=3, clip_on=False, gid=f"reliability-{panel.key}")
        # The count labels are placed by _place_counts once the layout is fixed.
    else:
        # Wrapped by hand: the note must stay inside its panel (a long line would cross the next one).
        note = textwrap.fill(_flat(panel.reason).removeprefix("insufficient: "), width=34)
        axes.text(0.5, 0.5, f"Insufficient held-out pairs:\n{note}", transform=axes.transAxes, ha="center",
                  va="center", fontsize=7, color=style.secondary_ink, zorder=4,
                  bbox={"facecolor": style.surface, "edgecolor": "none", "pad": 3.0})
    ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    axes.set_xlim(-0.04, 1.04)
    axes.set_ylim(-0.04, 1.04)
    axes.set_xticks(ticks, labels=["0", "0.25", "0.5", "0.75", "1"])
    axes.set_yticks(ticks, labels=["0", "0.25", "0.5", "0.75", "1"])
    axes.set_box_aspect(1.0)
    axes.set_title(_reliability_title(panel), loc="left", color=style.ink)


def plot_reliability(panels: Sequence[ReliabilityPanel], *, theme: str = "light") -> Figure:
    """F2: reliability diagrams, one panel per calibrator shown, with counts and ECE (Requirement 10.3).

    ``panels`` are plain data (:func:`reliability_panels`); missing panels are
    not drawn and insufficient ones are drawn without bins. Raises
    :class:`FigureError` when every panel is missing. Returns a matplotlib
    Figure (no pyplot, no global state).
    """
    import matplotlib
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from recall_guard.harness.plots import figure_theme

    panels = _checked_reliability_panels(panels)
    shown = [panel for panel in panels if panel.status != MISSING]
    if not shown:
        raise FigureError("nothing to draw: every reliability panel is missing (see the table)")
    style = figure_theme(theme)
    columns = 1 if len(shown) == 1 else 2
    rows = math.ceil(len(shown) / columns)
    largest = max((bin_.count for panel in shown for bin_ in panel.bins), default=1)
    width = _WIDTH_IN if columns == 2 else _WIDTH_IN / 2 + 0.4
    lines = max(_reliability_title(panel).count("\n") + 1 for panel in shown)
    panel_height = _RELIABILITY_PANEL_IN + _TITLE_LINE_IN * lines
    title = ("Reliability of the memorization estimate on held-out calibration draws" if columns == 2
             else "Reliability of the memorization estimate")
    with matplotlib.rc_context(_theme_rc(theme)):
        figure = Figure(figsize=(width, rows * panel_height + _RELIABILITY_FRAME_IN), layout="constrained")
        grid = figure.subplots(rows, columns, squeeze=False)
        cells = list(grid.ravel())
        for axes, panel in zip(cells, shown, strict=False):
            _draw_reliability_panel(axes, panel, style, theme, largest)
        for axes in cells[len(shown):]:
            axes.set_axis_off()
        figure.supxlabel("Mean predicted p_memorized in the bin")
        figure.supylabel("Observed share memorized in the bin")
        figure.suptitle(title, x=0.01, ha="left")
        # Fix the layout first, then place the count labels against the drawn geometry (a per-figure Agg canvas
        # measures the text; no pyplot, no backend switch).
        renderer = FigureCanvasAgg(figure).get_renderer()
        figure.draw_without_rendering()
        for axes, panel in zip(cells, shown, strict=False):
            if panel.status == DRAWN:
                _place_counts(axes, panel, style, largest, renderer)
        figure.draw_without_rendering()
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


def figure_svg(figure: Figure) -> str:
    """The SVG text of ``figure``: glyphs as paths, a fixed id salt and no timestamp, so it is byte-stable."""
    import matplotlib

    buffer = io.StringIO()
    with matplotlib.rc_context({"svg.hashsalt": SVG_HASH_SALT, "svg.fonttype": "path"}):
        figure.savefig(buffer, format="svg", metadata={"Date": None}, facecolor=figure.get_facecolor(),
                       edgecolor=figure.get_edgecolor())
    return buffer.getvalue()


def figure_paths(name: str, out_dir: str | Path = FIGURES_DIR) -> dict[str, Path]:
    """The four files of figure ``name`` in ``out_dir``: light and dark SVG, CSV and markdown, in that order."""
    if name not in FIGURE_NAMES:
        raise FigureError(f"unknown figure {name!r}; expected one of {list(FIGURE_NAMES)}")
    out = Path(out_dir)
    return {"light": out / f"{name}-light.svg", "dark": out / f"{name}-dark.svg", "csv": out / f"{name}.csv",
            "markdown": out / f"{name}.md"}


def figure_texts(name: str, table: FigureTable, draw: Callable[[str], Figure],
                 out_dir: str | Path = FIGURES_DIR) -> dict[Path, str]:
    """The four files of one figure as text by path: ``draw(theme)`` rendered for each theme, the table twice."""
    import matplotlib

    paths = figure_paths(name, out_dir)
    texts = {}
    for theme in THEMES:
        with matplotlib.rc_context(_theme_rc(theme)):
            texts[paths[theme]] = figure_svg(draw(theme))
    texts[paths["csv"]] = table.to_csv()
    texts[paths["markdown"]] = table.to_markdown()
    return texts


def _write_text(path: Path, text: str) -> None:
    """Write atomically: a crash leaves the previous file or the new one, never half of one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_outputs(texts: Mapping[Path, str]) -> tuple[Path, ...]:
    """Write every file once: all are checked first; the same text is a no-op, a different one refused."""
    for path, text in texts.items():
        if path.exists() and path.read_text(encoding="utf-8") != text:
            raise FigureError(f"refusing to replace the different figure output at {path}; remove the old figure "
                              "files first")
    for path, text in texts.items():
        if not path.exists():
            _write_text(path, text)
    return tuple(texts)


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
                                       plan_sha256=plan.plan_sha256) for arm in arms)
            texts.update(figure_texts(GAP, gap_table(gap), lambda theme, panels=gap: plot_gap(panels, theme=theme),
                                      out_dir))
        else:
            found = reliability_panels(plan, arms=arms, calibrators_dir=calibrators_dir)
            texts.update(figure_texts(RELIABILITY, reliability_table(found),
                                      lambda theme, panels=found: plot_reliability(panels, theme=theme), out_dir))
    return write_outputs(texts)


def main(argv: Sequence[str] | None = None) -> int:
    """Render the accuracy-gap and reliability figures of the frozen plan from its stored outputs.

    Exit codes: 0 written (or already written identically), 2 refused (another
    plan, a missing or malformed analysis, a foreign calibrator, nothing to
    draw, or a different figure at the output path).
    """
    parser = argparse.ArgumentParser(description="Accuracy-gap and reliability figures (task 8.2), from the stored "
                                                 "pre-registered analyses and calibrators; sends no request.")
    parser.add_argument("--plan", type=Path, required=True, help="the frozen plan.json")
    parser.add_argument("--plan-sha256", required=True, help="the plan hash the analyses and calibrators carry")
    parser.add_argument("--arm", action="append", choices=ARMS, help="an arm to draw (repeatable; default both, "
                                                                      "serving first)")
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
    "DEFAULT_ARMS",
    "DERIVED_ONLY",
    "DRAWN",
    "FIGURES_DIR",
    "FIGURE_NAMES",
    "GAP",
    "GAP_COLUMNS",
    "INSUFFICIENT",
    "MISSING",
    "RELIABILITY",
    "RELIABILITY_COLUMNS",
    "ROLE_PURPOSE",
    "FigureError",
    "FigureTable",
    "GapPanel",
    "GapPoint",
    "HeldOutReliability",
    "ReliabilityPanel",
    "figure_paths",
    "figure_svg",
    "figure_texts",
    "gap_panel",
    "gap_table",
    "held_out_reliability",
    "main",
    "plot_gap",
    "plot_reliability",
    "read_gap_panel",
    "reliability_panel",
    "reliability_panels",
    "reliability_table",
    "render_figures",
    "write_outputs",
]


if __name__ == "__main__":
    raise SystemExit(main())
