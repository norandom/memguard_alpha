"""Paper-ready matplotlib figure generators for the qualification notebook.

Implements the ``harness.plots`` component of the honest-model-ranking
design (see design.md → Components and Interfaces → harness.plots).
Satisfies Requirements 12.3, 12.4, 12.5:

- 12.3 — each notebook step displays at least one figure that visualises
  the underlying statistical process: MIA feature distributions, MCS
  calibration, accuracy with bootstrap CIs, MCS-AUC with bootstrap CIs,
  composite ranking.
- 12.4 — figures are paper-ready: vector PDF output, single-column width
  (3.5 inches), font sizes legible at native size, colorblind-safe
  palette, marker cycle that survives black-and-white reproduction.
- 12.5 — every ``plot_*`` function consumes a harness/MIA dataclass
  (``Record``, ``ModelEvalResult``, ``CIBound``, ``MCSCalibrator``,
  ``CompositeScore``) and returns a ``matplotlib.figure.Figure`` so the
  notebook can ``fig.savefig(path)`` without writing any plotting
  boilerplate of its own.

Pure presentation layer: no I/O, no logging, no global mutable state
beyond ``matplotlib.rcParams`` — and the ``rcParams`` write happens only
inside ``configure_paper_style``, which is opt-in.

Notes
-----
``MCSCalibrator`` retains only the held-out AUC scalar — not the
per-prompt held-out predictions / labels. A faithful reliability /
calibration curve is therefore not constructible from the dataclass
alone, so :func:`plot_mcs_calibration` renders the held-out AUC as a
horizontal line annotated with the ``min_auc`` gate. This is documented
in the function docstring and is the most honest visualisation of what
the dataclass actually carries.
"""

from __future__ import annotations

from typing import Literal, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.figure import Figure

from src.harness.evaluator import CIBound, ModelEvalResult, Record
from src.harness.ranker import CompositeScore
from src.mia.mcs import MCSCalibrator

#: Wong (2011) colorblind-safe palette. Index 0 = IS / surviving model;
#: index 1 = OOS / contrast; remaining entries cycle through additional
#: series. Used by both ``configure_paper_style`` (as the matplotlib
#: prop_cycle) and the individual ``plot_*`` helpers (for explicit
#: hand-picked colors on overlaid series).
PAPER_PALETTE: list[str] = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#F0E442",
]

#: Marker cycle that distinguishes lines under black-and-white
#: reproduction (Req 12.4). Length matches ``PAPER_PALETTE`` so the two
#: cyclers compose into a single ``axes.prop_cycle``.
PAPER_MARKERS: list[str] = ["o", "s", "^", "D", "v"]

#: MCS-AUC gate threshold rendered on the AUC plots (mirrors
#: ``harness.ranker.GATES['mcs_auc_min']``). Kept as a module-level
#: constant rather than imported to avoid a circular reference between
#: the ranker (which already imports from evaluator) and the plotting
#: layer.
_MCS_AUC_GATE: float = 0.6
_MCS_AUC_RANDOM: float = 0.5

#: Color for failed-gate bars in the composite-ranking chart (Req 12.4 —
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

    * ``figure.figsize = (3.5, 2.5)`` — single column of a two-column
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

    ``MCSCalibrator`` does not retain per-prompt held-out predictions,
    so a true reliability curve cannot be reconstructed from the
    dataclass. This figure draws three honest reference lines:

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

    for y, score in zip(ys, bar_scores):
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
    "configure_paper_style",
    "plot_accuracy_with_ci",
    "plot_composite_ranking",
    "plot_mcs_auc_with_ci",
    "plot_mcs_calibration",
    "plot_mia_feature_distributions",
]
