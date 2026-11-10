"""Paper-ready matplotlib figure generators for the qualification notebook.

Implements the ``harness.plots`` component of the honest-model-ranking
design (see design.md → Components and Interfaces → harness.plots).
Satisfies Requirements 12.3, 12.4, 12.5:

- 12.3: each notebook step displays at least one figure that visualises
  the underlying statistical process: MIA feature distributions, MCS
  calibration, accuracy with bootstrap CIs, MCS-AUC with bootstrap CIs,
  composite ranking.
- 12.4: figures are paper-ready: vector PDF output, single-column width
  (3.5 inches), font sizes legible at native size, colorblind-safe
  palette, marker cycle that survives black-and-white reproduction.
- 12.5: every ``plot_*`` function consumes a harness/MIA dataclass
  (``Record``, ``ModelEvalResult``, ``CIBound``, ``MCSCalibrator``,
  ``CompositeScore``) and returns a ``matplotlib.figure.Figure`` so the
  notebook can ``fig.savefig(path)`` without writing any plotting
  boilerplate of its own.

It also holds the figure style of the memorization-adjusted-finance-sim
worked example (that design, Figures; Requirement 10.6): the validated
fixed-order ``STUDY_PALETTE`` for light and ``STUDY_PALETTE_DARK`` for dark
surfaces (the legacy ``plot_*`` figures keep ``PAPER_PALETTE``), the theme chrome, the
marker-shape and fill encodings that keep variant identity readable
without colour, and :func:`reliability_curve`.

Pure presentation layer: no I/O, no logging, no global mutable state
beyond ``matplotlib.rcParams``, and the ``rcParams`` write happens only
inside ``configure_paper_style``, which is opt-in. ``FigureTheme.rc_params``
only returns a mapping for ``matplotlib.rc_context``.

This module imports matplotlib when it is imported; the package root and
``recall_guard.harness`` never import it eagerly (the harness resolves the
``plot_*`` names lazily), so ``import recall_guard`` stays matplotlib-free.

Notes
-----
Calibrators fitted by ``train_mcs`` retain the paired held-out
observations, ``holdout_probabilities`` and ``holdout_labels``, so a true
reliability curve is built from them with :func:`reliability_curve`.
Calibrators constructed without them (older or hand-built ones) carry
None, and only the AUC scalar is available. :func:`plot_mcs_calibration`
shows discrimination, not calibration: the held-out AUC as a horizontal
line against the random baseline and the ``min_auc`` gate.
"""

from __future__ import annotations

import numbers
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.figure import Figure

from recall_guard.harness.evaluator import CIBound, ModelEvalResult, Record
from recall_guard.harness.ranker import CompositeScore
from recall_guard.mia.mcs import MCSCalibrator

#: Wong (2011) colorblind-safe palette. Index 0 = IS / surviving model;
#: index 1 = OOS / contrast; remaining entries cycle through additional
#: series. Used by both ``configure_paper_style`` (as the matplotlib
#: prop_cycle) and the individual ``plot_*`` helpers (for explicit
#: hand-picked colors on overlaid series).
#:
#: Kept only for these earlier figures (the honest-model-ranking notebook and
#: its spec pin these values). It fails the dataviz ``validate_palette.js``
#: lightness band (task 8.1, 2026-11-10): in light mode #F0E442 (OKLCH L
#: 0.902, band 0.43-0.77); in dark mode #F0E442 (0.902) and #CC79A7 (0.679,
#: band 0.48-0.67). It also takes WARNs: #CC79A7 and #F0E442 below 3:1 on
#: the light surface, and adjacent CVD Delta E 7.6 (#CC79A7 and #009E73,
#: deutan). The finance-sim figures use ``STUDY_PALETTE`` instead.
PAPER_PALETTE: list[str] = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#F0E442",
]

#: Fixed-order categorical palette of the finance-sim figures, light steps:
#: the first five slots of the dataviz documented palette (blue, orange,
#: aqua, yellow, magenta), assigned in order and never re-ordered. Used by
#: the light ``figure_theme`` and ``variant_style``.
#:
#: Validated (task 8.1, 2026-11-10) with the dataviz ``validate_palette.js``,
#: sha256 a5f3facc62ac64f3872f1674d60126820364ffcd368eddf0a6360f22c49887e2,
#: ``--mode light --surface #fcfcfb``: lightness band, chroma floor, CVD
#: separation (worst adjacent Delta E 9.1) and the normal-vision floor (19.6)
#: pass; contrast is a WARN, since aqua, yellow and magenta sit below 3:1 on
#: the light surface, so those slots need the relief channel (direct labels
#: or the data table every figure ships beside it). The validator output is
#: in the private receipt ``runs/finance-sim/figures/palette/receipt.json``.
STUDY_PALETTE: list[str] = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
]

#: The same five hues stepped for the dark surface (the dark column of the
#: dataviz documented palette), used by the dark ``figure_theme``. Validated
#: with ``--mode dark --surface #1a1a19``: every check passes (worst adjacent
#: CVD Delta E 8.4, normal vision 19.3, every slot at or above 3:1).
STUDY_PALETTE_DARK: list[str] = [
    "#3987e5",
    "#d95926",
    "#199e70",
    "#c98500",
    "#d55181",
]

#: Series cap of ``STUDY_PALETTE`` and ``STUDY_PALETTE_DARK`` for forms where
#: any two marks can sit side by side (dot and interval plots, scatter, small
#: multiples): only their first three slots pass ``--pairs all`` in both modes
#: (worst CVD Delta E 9.2 light / 9.4 dark, normal vision 24.0 / 20.9). Past
#: three, fold or facet; never re-order.
ALL_PAIRS_SLOT_CAP: int = 3

#: Marker cycle that distinguishes lines under black-and-white
#: reproduction (Req 12.4). Length matches ``PAPER_PALETTE`` so the two
#: cyclers compose into a single ``axes.prop_cycle`` (and ``STUDY_PALETTE``,
#: which the figure themes pair it with).
PAPER_MARKERS: list[str] = ["o", "s", "^", "D", "v"]

#: MCS-AUC gate threshold rendered on the AUC plots (mirrors
#: ``harness.ranker.GATES['mcs_auc_min']``). Kept as a module-level
#: constant rather than imported to avoid a circular reference between
#: the ranker (which already imports from evaluator) and the plotting
#: layer.
_MCS_AUC_GATE: float = 0.6
_MCS_AUC_RANDOM: float = 0.5

#: Color for failed-gate bars in the composite-ranking chart (Req 12.4;
#: must reproduce as grey under black-and-white printing).
_FAILED_GATE_COLOR: str = "#999999"


# --- Public API ---------------------------------------------------------------


def configure_paper_style() -> None:
    """Set matplotlib ``rcParams`` for paper-ready single-column figures.

    Idempotent: every call rewrites the same set of keys. Does not
    install fonts or change the matplotlib backend; callers that need a
    headless backend should set ``matplotlib.use("Agg")`` themselves
    before any pyplot import.

    Sets:

    * ``figure.figsize = (3.5, 2.5)``: single column of a two-column
      manuscript at native size.
    * ``font.size = 8``; ``axes.titlesize = 9``; ``axes.labelsize = 8``;
      ``xtick.labelsize = 7``; ``ytick.labelsize = 7``;
      ``legend.fontsize = 7``.
    * ``savefig.dpi = 300``; ``savefig.format = "pdf"``;
      ``savefig.bbox = "tight"``.
    * ``axes.prop_cycle`` to ``PAPER_PALETTE`` × ``PAPER_MARKERS`` so
      that lines drawn without an explicit color/marker still differ
      under B&W reproduction.
    """
    matplotlib.rcParams.update(
        {
            "figure.figsize": (3.5, 2.5),
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "savefig.dpi": 300,
            "savefig.format": "pdf",
            "savefig.bbox": "tight",
            "axes.prop_cycle": (
                cycler(color=PAPER_PALETTE) + cycler(marker=PAPER_MARKERS)
            ),
        }
    )


def plot_mia_feature_distributions(
    is_records: Sequence[Record],
    oos_records: Sequence[Record],
    feature: Literal["loss", "min_k", "min_k_pp", "zlib_ratio", "ref_delta"],
) -> Figure:
    """Overlay IS vs OOS distributions for one MIA feature.

    Records with ``parse_ok=False`` or ``features_raw is None`` are
    filtered out before plotting (the feature value is undefined for
    those rows). For ``feature == "ref_delta"`` records whose
    ``features_raw.ref_delta is None`` are dropped as well (no reference
    logprob run was performed for that record).

    Returns
    -------
    matplotlib.figure.Figure
        Histogram with two overlaid series, ``alpha=0.6`` for visibility
        of overlap.
    """
    is_values = _extract_feature(is_records, feature)
    oos_values = _extract_feature(oos_records, feature)

    fig, ax = plt.subplots()

    bins = _shared_bins(is_values, oos_values, n_bins=20)

    if is_values.size:
        ax.hist(
            is_values,
            bins=bins,
            color=PAPER_PALETTE[0],
            alpha=0.6,
            label="IS (memorized)",
            density=True,
        )
    if oos_values.size:
        ax.hist(
            oos_values,
            bins=bins,
            color=PAPER_PALETTE[1],
            alpha=0.6,
            label="OOS (control)",
            density=True,
        )

    ax.set_title(f"{_pretty_feature_name(feature)}: IS vs OOS")
    ax.set_xlabel(_pretty_feature_name(feature))
    ax.set_ylabel("Density")
    ax.legend(loc="best")
    return fig


def plot_mcs_calibration(mcs: MCSCalibrator) -> Figure:
    """Render the MCS calibrator's held-out AUC against the gate.

    The AUC measures discrimination, not calibration. For a reliability
    curve, pass the calibrator's retained ``holdout_probabilities`` and
    ``holdout_labels`` (set by ``train_mcs``; None on calibrators
    constructed without them) to :func:`reliability_curve`. This figure
    draws three reference lines:

    * ``mcs.holdout_auc`` (the trained AUC, palette[0]),
    * ``0.5`` (random-classifier baseline),
    * ``0.6`` (the ``min_auc`` gate from ``harness.ranker.GATES``).

    The ``is_weak`` flag is annotated when set so the reader can see at
    a glance whether the calibrator passed the gate.
    """
    fig, ax = plt.subplots()

    auc = float(mcs.holdout_auc)
    ax.axhline(_MCS_AUC_RANDOM, color=PAPER_PALETTE[2], linestyle=":", label="Random (0.5)")
    ax.axhline(_MCS_AUC_GATE, color=PAPER_PALETTE[1], linestyle="--", label=f"Gate ({_MCS_AUC_GATE})")
    ax.axhline(auc, color=PAPER_PALETTE[0], linestyle="-", label=f"Holdout AUC ({auc:.3f})")

    ax.set_ylim(0.0, 1.0)
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([])
    ax.set_xlabel("Decision threshold (not retained by MCSCalibrator)")
    ax.set_ylabel("Holdout AUC")

    title = f"MCS calibration: {mcs.model}"
    if mcs.is_weak:
        title += " — weak"
    ax.set_title(title)

    annotation = f"AUC = {auc:.3f}"
    if mcs.is_weak:
        annotation += " (weak)"
    ax.text(
        0.02,
        0.95,
        annotation,
        transform=ax.transAxes,
        ha="left",
        va="top",
    )

    ax.legend(loc="lower right")
    return fig


def plot_accuracy_with_ci(
    results: Sequence[ModelEvalResult],
    majority: CIBound,
) -> Figure:
    """Bar chart of MemGuard accuracy with bootstrap 95% CIs and majority baseline.

    Each bar shows ``result.memguard_accuracy.point`` with asymmetric
    error bars to ``[lo, hi]``. A horizontal dashed line at
    ``majority.point`` plus a shaded band ``[lo, hi]`` represents the
    majority-class baseline (Req 6.2).
    """
    fig, ax = plt.subplots()

    names = [r.model for r in results]
    points = np.asarray([r.memguard_accuracy.point for r in results], dtype=float)
    los = np.asarray([r.memguard_accuracy.lo for r in results], dtype=float)
    his = np.asarray([r.memguard_accuracy.hi for r in results], dtype=float)
    err_lower = np.clip(points - los, a_min=0.0, a_max=None)
    err_upper = np.clip(his - points, a_min=0.0, a_max=None)

    xs = np.arange(len(names))
    ax.bar(
        xs,
        points,
        yerr=[err_lower, err_upper],
        color=PAPER_PALETTE[0],
        capsize=3,
        label="Models",
    )

    ax.axhspan(majority.lo, majority.hi, color=PAPER_PALETTE[1], alpha=0.15)
    ax.axhline(
        majority.point,
        color=PAPER_PALETTE[1],
        linestyle="--",
        label="Majority baseline",
    )

    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("MemGuard Accuracy")
    ax.set_xlabel("Model")
    ax.set_title("MemGuard Accuracy with bootstrap 95% CI")
    ax.legend(loc="best")
    return fig


def plot_mcs_auc_with_ci(results: Sequence[ModelEvalResult]) -> Figure:
    """Bar chart of MCS-AUC with bootstrap 95% CIs and gate references.

    Reference lines at 0.5 (random) and 0.6 (gate). The region below
    0.6 is shaded to indicate the weak-calibration zone.
    """
    fig, ax = plt.subplots()

    names = [r.model for r in results]
    points = np.asarray([r.mcs_auc.point for r in results], dtype=float)
    los = np.asarray([r.mcs_auc.lo for r in results], dtype=float)
    his = np.asarray([r.mcs_auc.hi for r in results], dtype=float)
    err_lower = np.clip(points - los, a_min=0.0, a_max=None)
    err_upper = np.clip(his - points, a_min=0.0, a_max=None)

    xs = np.arange(len(names))
    ax.bar(
        xs,
        points,
        yerr=[err_lower, err_upper],
        color=PAPER_PALETTE[0],
        capsize=3,
    )

    ax.axhspan(0.0, _MCS_AUC_GATE, color=_FAILED_GATE_COLOR, alpha=0.15)
    ax.axhline(_MCS_AUC_RANDOM, color=PAPER_PALETTE[2], linestyle=":", label="Random (0.5)")
    ax.axhline(_MCS_AUC_GATE, color=PAPER_PALETTE[1], linestyle="--", label=f"Gate ({_MCS_AUC_GATE})")

    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("MCS-AUC")
    ax.set_xlabel("Model")
    ax.set_title("MCS-AUC with bootstrap 95% CI")
    ax.legend(loc="best")
    return fig


def plot_composite_ranking(scores: Sequence[CompositeScore]) -> Figure:
    """Horizontal bar chart of composite scores, descending.

    Surviving models render in ``PAPER_PALETTE[0]``; non-survivors render
    in grey (``#999999``) with their first failed gate annotated next to
    the bar. The chart is laid out top-down by descending score so the
    best surviving model appears at the top.
    """
    fig, ax = plt.subplots()

    sorted_scores = sorted(scores, key=lambda s: s.score, reverse=True)

    names = [s.model for s in sorted_scores]
    values = [s.score for s in sorted_scores]
    colors = [
        PAPER_PALETTE[0] if s.survives_gates else _FAILED_GATE_COLOR
        for s in sorted_scores
    ]

    # matplotlib.barh lays out indices bottom-up; reverse so the highest
    # score sits at the top of the figure.
    ys = np.arange(len(sorted_scores))
    bar_values = list(reversed(values))
    bar_colors = list(reversed(colors))
    bar_names = list(reversed(names))
    bar_scores = list(reversed(sorted_scores))

    ax.barh(ys, bar_values, color=bar_colors)
    ax.set_yticks(ys)
    ax.set_yticklabels(bar_names)

    for y, score in zip(ys, bar_scores, strict=True):
        if not score.survives_gates and score.warnings:
            failed = score.warnings[0]
            ax.text(
                max(score.score, 0.0) + 0.01,
                y,
                failed,
                va="center",
                ha="left",
                fontsize=6,
                color=_FAILED_GATE_COLOR,
            )

    ax.set_xlabel("Composite score")
    ax.set_ylabel("Model")
    ax.set_title("Composite ranking")
    return fig


# --- Figure style of the finance-sim worked example -------------------------

Theme = Literal["light", "dark"]

#: Figure themes, in render order: every docs figure has a light and a dark
#: variant (MkDocs Material ``#only-light`` / ``#only-dark``).
THEMES: tuple[str, ...] = ("light", "dark")

#: Adjustment of a study variant, in fixed order; it picks palette slots 1-3.
ADJUSTMENTS: tuple[str, ...] = ("none", "top_k", "full_distribution")

#: Draw mode of a study variant (the ``DrawMode`` values).
DRAW_MODES: tuple[str, ...] = ("single", "ensemble")

#: Equal-width bins of a reliability curve, the count the local report's
#: held-out calibration uses.
RELIABILITY_BINS: int = 10

_ADJUSTMENT_MARKERS = {"none": "o", "top_k": "s", "full_distribution": "^"}
_ADJUSTMENT_LABELS = {"none": "no adjustment", "top_k": "top-k", "full_distribution": "full distribution"}
_DRAW_MODE_LABELS = {"single": "single draw", "ensemble": "ensemble"}
_FILLED_DRAW_MODE = "ensemble"


@dataclass(frozen=True)
class FigureTheme:
    """Colours of one figure theme: the palette, the chart surface and the chrome ink.

    Hex values are the dataviz reference tokens. ``palette`` was validated
    against ``surface`` (``STUDY_PALETTE`` for light, ``STUDY_PALETTE_DARK``
    for dark; see their comments). ``ink`` is for titles and
    labels, ``secondary_ink`` for tick labels, ``muted`` for reference marks
    such as a zero line or the diagonal, ``gridline`` for hairlines and
    ``baseline`` for the axis spines.
    """

    name: str
    palette: tuple[str, ...]
    surface: str
    ink: str
    secondary_ink: str
    muted: str
    gridline: str
    baseline: str

    def rc_params(self) -> dict[str, object]:
        """rcParams for ``matplotlib.rc_context`` that render a figure in this theme.

        The surface fills the figure, the axes and the saved file. The
        property cycle pairs the palette with ``PAPER_MARKERS``, so series
        drawn without an explicit style still differ by marker shape.
        Nothing global changes: use ``with matplotlib.rc_context(...)``.
        """
        return {
            "figure.facecolor": self.surface,
            "figure.edgecolor": self.surface,
            "axes.facecolor": self.surface,
            "savefig.facecolor": self.surface,
            "savefig.edgecolor": self.surface,
            "text.color": self.ink,
            "axes.titlecolor": self.ink,
            "axes.labelcolor": self.ink,
            "axes.edgecolor": self.baseline,
            "xtick.color": self.secondary_ink,
            "ytick.color": self.secondary_ink,
            "grid.color": self.gridline,
            "legend.facecolor": self.surface,
            "legend.edgecolor": self.baseline,
            "legend.labelcolor": self.ink,
            "axes.prop_cycle": cycler(color=list(self.palette)) + cycler(marker=PAPER_MARKERS),
        }


_FIGURE_THEMES = {
    "light": FigureTheme(
        name="light", palette=tuple(STUDY_PALETTE), surface="#fcfcfb", ink="#0b0b0b",
        secondary_ink="#52514e", muted="#898781", gridline="#e1e0d9", baseline="#c3c2b7",
    ),
    "dark": FigureTheme(
        name="dark", palette=tuple(STUDY_PALETTE_DARK), surface="#1a1a19", ink="#f0efec",
        secondary_ink="#c3c2b7", muted="#898781", gridline="#2c2c2a", baseline="#383835",
    ),
}


def figure_theme(theme: Theme) -> FigureTheme:
    """Return the colours of ``theme`` (``"light"`` or ``"dark"``).

    Raises ValueError naming the known themes for any other value.
    """
    try:
        return _FIGURE_THEMES[theme]
    except (KeyError, TypeError):
        raise ValueError(f"unknown figure theme {theme!r}; expected one of: {', '.join(THEMES)}") from None


@dataclass(frozen=True)
class VariantStyle:
    """How one study variant is drawn so that it reads without colour (Requirement 10.6).

    Marker shape encodes the adjustment (circle: none, square: top-k,
    triangle: full distribution) and fill encodes the draw mode (open:
    single draw, filled: ensemble), so the six variants differ in shape and
    fill alone. Colour repeats the adjustment from the first three palette
    slots, within ``ALL_PAIRS_SLOT_CAP``, and is never the only cue. An open
    marker is filled with the theme surface, so it stays hollow over an
    interval line. ``label`` is the text for a direct label or legend entry.
    """

    adjustment: str
    draw_mode: str
    marker: str
    color: str
    facecolor: str
    filled: bool
    label: str

    def marker_kwargs(self) -> dict[str, object]:
        """Keyword arguments for ``Axes.plot`` or ``Axes.errorbar``: markers, no line."""
        return {
            "marker": self.marker,
            "linestyle": "none",
            "color": self.color,
            "markeredgecolor": self.color,
            "markerfacecolor": self.facecolor,
            "label": self.label,
        }

    def scatter_kwargs(self) -> dict[str, object]:
        """Keyword arguments for ``Axes.scatter``."""
        return {
            "marker": self.marker,
            "edgecolors": self.color,
            "facecolors": self.facecolor,
            "label": self.label,
        }


def variant_style(adjustment: str, draw_mode: str, *, theme: Theme) -> VariantStyle:
    """Return the marker encoding of one study variant in ``theme``.

    ``adjustment`` is one of ``ADJUSTMENTS`` (``"none"`` or an
    ``EstimateSource`` value) and ``draw_mode`` one of ``DRAW_MODES`` (a
    ``DrawMode`` value). Raises ValueError naming the allowed values for an
    unknown theme, adjustment or draw mode.
    """
    style = figure_theme(theme)
    if adjustment not in ADJUSTMENTS:
        raise ValueError(f"unknown adjustment {adjustment!r}; expected one of: {', '.join(ADJUSTMENTS)}")
    if draw_mode not in DRAW_MODES:
        raise ValueError(f"unknown draw mode {draw_mode!r}; expected one of: {', '.join(DRAW_MODES)}")
    color = style.palette[ADJUSTMENTS.index(adjustment)]
    filled = draw_mode == _FILLED_DRAW_MODE
    return VariantStyle(
        adjustment=adjustment,
        draw_mode=draw_mode,
        marker=_ADJUSTMENT_MARKERS[adjustment],
        color=color,
        facecolor=color if filled else style.surface,
        filled=filled,
        label=f"{_ADJUSTMENT_LABELS[adjustment]}, {_DRAW_MODE_LABELS[draw_mode]}",
    )


@dataclass(frozen=True)
class ReliabilityBin:
    """One non-empty bin ``[low, high)`` of a reliability curve (the last bin includes 1.0).

    ``count`` is the number of items in the bin, ``mean_predicted`` their
    mean ``p_memorized`` and ``observed_share`` the share of them labelled
    memorized (label 1); all are dimensionless.
    """

    low: float
    high: float
    count: int
    mean_predicted: float
    observed_share: float


@dataclass(frozen=True)
class ReliabilityCurve:
    """Reliability curve of held-out probabilities against their labels.

    ``bins`` lists the non-empty bins in ascending order, ``ece`` is the
    count-weighted mean of ``|mean_predicted - observed_share|`` over them,
    ``n`` is the number of items and ``n_bins`` the number of equal-width
    bins over [0, 1].
    """

    bins: tuple[ReliabilityBin, ...]
    ece: float
    n: int
    n_bins: int


def reliability_curve(
    probabilities: Sequence[float] | None,
    labels: Sequence[int] | None,
    *,
    n_bins: int = RELIABILITY_BINS,
) -> ReliabilityCurve:
    """Bin held-out probabilities into a reliability curve with its ECE.

    ``probabilities`` are ``p_memorized`` values in [0, 1] and ``labels``
    the paired held-out labels (1 = memorized, 0 = control), for example a
    calibrator's ``holdout_probabilities`` and ``holdout_labels``. The bins
    are ``n_bins`` equal-width intervals over [0, 1], with 1.0 in the last
    bin, and empty bins are omitted: the binning of the local report's
    held-out calibration, so a figure and its report agree.

    Raises ValueError before any binning when either sequence is None
    (calibrators constructed without the held-out pairs carry None), when
    the lengths differ or are zero, when a probability is not a finite value
    in [0, 1], when a label is not 0 or 1, or when ``n_bins`` is not a
    positive integer.
    """
    if probabilities is None or labels is None:
        raise ValueError(
            "reliability_curve needs held-out probabilities and labels; "
            "calibrators constructed without them carry None"
        )
    if isinstance(n_bins, bool) or not isinstance(n_bins, numbers.Integral) or n_bins < 1:
        raise ValueError(f"n_bins must be a positive integer, got {n_bins!r}")
    n_bins = int(n_bins)
    predicted = np.asarray(probabilities, dtype=float)
    truth = np.asarray(labels)
    if predicted.ndim != 1 or truth.ndim != 1 or predicted.size != truth.size:
        raise ValueError("probabilities and labels must be flat sequences of the same length")
    if predicted.size == 0:
        raise ValueError("reliability_curve needs at least one held-out item")
    if not np.all(np.isfinite(predicted)) or np.any((predicted < 0.0) | (predicted > 1.0)):
        raise ValueError("every probability must be a finite value in [0, 1]")
    if not np.all(np.isin(truth, (0, 1))):
        raise ValueError("every label must be 0 or 1")
    observed = truth.astype(float)
    index = np.clip(np.floor(predicted * n_bins).astype(int), 0, n_bins - 1)
    bins: list[ReliabilityBin] = []
    weighted_gap = 0.0
    for position in range(n_bins):
        member = index == position
        count = int(member.sum())
        if not count:
            continue
        mean_predicted = float(predicted[member].mean())
        share = float(observed[member].mean())
        bins.append(
            ReliabilityBin(
                low=position / n_bins,
                high=(position + 1) / n_bins,
                count=count,
                mean_predicted=mean_predicted,
                observed_share=share,
            )
        )
        weighted_gap += count * abs(mean_predicted - share)
    return ReliabilityCurve(
        bins=tuple(bins), ece=weighted_gap / predicted.size, n=int(predicted.size), n_bins=n_bins
    )


# --- Internal helpers ---------------------------------------------------------


def _extract_feature(
    records: Sequence[Record],
    feature: str,
) -> np.ndarray:
    """Pull a single MIA feature value from each parse-OK record.

    Drops records where ``parse_ok`` is False, ``features_raw`` is None,
    or (only for ``ref_delta``) the per-record ``ref_delta`` value is
    ``None``. Returns a 1-D float64 array which may be empty.
    """
    values: list[float] = []
    for r in records:
        if not r.parse_ok or r.features_raw is None:
            continue
        raw = getattr(r.features_raw, feature)
        if raw is None:
            continue
        values.append(float(raw))
    return np.asarray(values, dtype=np.float64)


def _shared_bins(
    a: np.ndarray, b: np.ndarray, n_bins: int = 20
) -> np.ndarray | int:
    """Compute a shared histogram bin edge array for two distributions.

    Falls back to an integer bin count when neither array has data, so
    matplotlib will not raise on an empty domain.
    """
    if a.size == 0 and b.size == 0:
        return n_bins
    combined_min = float(min(a.min() if a.size else b.min(), b.min() if b.size else a.min()))
    combined_max = float(max(a.max() if a.size else b.max(), b.max() if b.size else a.max()))
    if combined_min == combined_max:
        # Avoid degenerate single-bin range; pad symmetrically.
        pad = 1.0 if combined_min == 0.0 else abs(combined_min) * 0.1
        combined_min -= pad
        combined_max += pad
    return np.linspace(combined_min, combined_max, n_bins + 1)


_FEATURE_PRETTY_NAMES: dict[str, str] = {
    "loss": "Loss",
    "min_k": "Min-K%",
    "min_k_pp": "Min-K%++",
    "zlib_ratio": "zlib ratio",
    "ref_delta": "Reference delta",
}


def _pretty_feature_name(feature: str) -> str:
    """Map a raw MIA feature key to a human-readable axis/title label."""
    return _FEATURE_PRETTY_NAMES.get(feature, feature)


__all__ = [
    "PAPER_PALETTE",
    "PAPER_MARKERS",
    "STUDY_PALETTE",
    "STUDY_PALETTE_DARK",
    "ALL_PAIRS_SLOT_CAP",
    "THEMES",
    "ADJUSTMENTS",
    "DRAW_MODES",
    "RELIABILITY_BINS",
    "Theme",
    "FigureTheme",
    "VariantStyle",
    "ReliabilityBin",
    "ReliabilityCurve",
    "figure_theme",
    "variant_style",
    "reliability_curve",
    "configure_paper_style",
    "plot_accuracy_with_ci",
    "plot_composite_ranking",
    "plot_mcs_auc_with_ci",
    "plot_mcs_calibration",
    "plot_mia_feature_distributions",
]
