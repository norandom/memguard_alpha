"""Figure style of the finance-sim worked example (task 8.1, Requirement 10.6).

Covers the validated fixed-order study palette for light and dark surfaces
(the hex values pinned here are the ones the dataviz ``validate_palette.js``
passed for both modes; the validator output is kept in the private receipt
under ``runs/finance-sim/figures/palette/``), the unchanged Wong
``PAPER_PALETTE`` of the earlier figures, the theme chrome and rcParams, the
marker-shape and fill encodings that keep variant identity readable without
colour, and the reliability helper that builds a true reliability curve from
a calibrator's retained held-out pairs.

The Agg backend is selected at import time so the suite can run headless.
"""

from __future__ import annotations

import io
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_hex, to_rgba
from sklearn.linear_model import LogisticRegression

from recall_guard.harness import local_reporting, plots
from recall_guard.harness.evaluator import CIBound, ModelEvalResult
from recall_guard.harness.plots import (
    ADJUSTMENTS,
    ALL_PAIRS_SLOT_CAP,
    DRAW_MODES,
    PAPER_MARKERS,
    PAPER_PALETTE,
    STUDY_PALETTE,
    STUDY_PALETTE_DARK,
    THEMES,
    ReliabilityBin,
    figure_theme,
    plot_accuracy_with_ci,
    plot_mcs_calibration,
    reliability_curve,
    variant_style,
)
from recall_guard.mia.mcs import MCSCalibrator

#: Validated with validate_palette.js (sha256 a5f3facc...e887e2) on 2026-11-10:
#: the first five slots of the dataviz documented palette, light and dark steps.
VALIDATED_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
VALIDATED_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"]
#: Wong (2011) values of the earlier figures, exactly as in HEAD before 8.1.
WONG = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442"]
#: The surfaces the validator ran against (its defaults, passed explicitly).
LIGHT_SURFACE = "#fcfcfb"
DARK_SURFACE = "#1a1a19"
#: Reserved status steps of the dataviz palette; a series never wears them.
STATUS_COLOURS = {"#0ca30c", "#fab219", "#ec835a", "#d03b3b"}


def _calibrator() -> MCSCalibrator:
    """A real fitted calibrator carrying 12 held-out pairs, as ``train_mcs`` retains them."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 4))
    y = (x[:, 0] > 0).astype(int)
    classifier = LogisticRegression(solver="liblinear", random_state=0).fit(x, y)
    return MCSCalibrator(
        model="model-a",
        classifier=classifier,
        feature_order=["loss", "min_k", "min_k_pp", "zlib_ratio"],
        holdout_auc=0.9,
        is_weak=False,
        holdout_labels=tuple(int(v) for v in y[:12]),
        holdout_probabilities=tuple(float(p) for p in classifier.predict_proba(x[:12])[:, 1]),
    )


# --- palette -----------------------------------------------------------------


def test_study_palette_pins_the_validated_light_and_dark_hex_values():
    assert STUDY_PALETTE == VALIDATED_LIGHT
    assert STUDY_PALETTE_DARK == VALIDATED_DARK
    assert figure_theme("light").palette == tuple(VALIDATED_LIGHT)
    assert figure_theme("dark").palette == tuple(VALIDATED_DARK)


def test_paper_palette_keeps_the_wong_values_of_the_earlier_figures():
    assert PAPER_PALETTE == WONG
    assert not set(PAPER_PALETTE) & set(STUDY_PALETTE + STUDY_PALETTE_DARK)


def test_legacy_plot_functions_keep_drawing_with_the_paper_palette():
    results = [
        ModelEvalResult(
            model="model-a",
            raw_accuracy=CIBound(0.7, 0.6, 0.8),
            memguard_accuracy=CIBound(0.7, 0.6, 0.8),
            mcs_auc=CIBound(0.85, 0.75, 0.95),
            parse_success_rate=1.0,
            parse_failures=0,
            warnings=[],
            records=[],
        )
    ]
    accuracy = plot_accuracy_with_ci(results, CIBound(0.5, 0.45, 0.55))
    calibration = plot_mcs_calibration(_calibrator())
    try:
        assert to_hex(accuracy.axes[0].patches[0].get_facecolor()) == PAPER_PALETTE[0].lower()
        line_colours = {to_hex(line.get_color()) for line in calibration.axes[0].lines}
        assert line_colours == {c.lower() for c in PAPER_PALETTE[:3]}
    finally:
        plt.close(accuracy)
        plt.close(calibration)


def test_study_palettes_have_one_marker_per_slot():
    assert len(STUDY_PALETTE) == len(STUDY_PALETTE_DARK) == len(PAPER_MARKERS)
    assert len(set(PAPER_MARKERS)) == len(PAPER_MARKERS)


def test_study_palette_never_reuses_a_status_colour():
    for slots in (STUDY_PALETTE, STUDY_PALETTE_DARK):
        assert not {c.lower() for c in slots} & STATUS_COLOURS


def test_all_pairs_cap_is_the_study_prefix_that_validated_under_all_pairs():
    # Only the first three study slots passed ``--pairs all`` in both modes;
    # dot plots, scatter and small multiples take at most that many colours.
    assert ALL_PAIRS_SLOT_CAP == 3
    assert len(ADJUSTMENTS) <= ALL_PAIRS_SLOT_CAP
    for theme in THEMES:
        adjustment_colours = [variant_style(a, "ensemble", theme=theme).color for a in ADJUSTMENTS]
        assert adjustment_colours == list(figure_theme(theme).palette[:ALL_PAIRS_SLOT_CAP])


# --- theme -------------------------------------------------------------------


def test_figure_theme_uses_the_validated_surfaces_and_chrome():
    light, dark = figure_theme("light"), figure_theme("dark")
    assert THEMES == ("light", "dark")
    assert (light.name, light.surface, light.ink) == ("light", LIGHT_SURFACE, "#0b0b0b")
    assert (dark.name, dark.surface, dark.ink) == ("dark", DARK_SURFACE, "#f0efec")
    assert (light.secondary_ink, light.gridline, light.baseline) == ("#52514e", "#e1e0d9", "#c3c2b7")
    assert (dark.secondary_ink, dark.gridline, dark.baseline) == ("#c3c2b7", "#2c2c2a", "#383835")


def test_figure_theme_rejects_an_unknown_theme():
    with pytest.raises(ValueError, match="light, dark"):
        figure_theme("sepia")


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_rc_params_apply_surface_ink_and_cycle_without_touching_global_state(theme):
    style = figure_theme(theme)
    before = dict(matplotlib.rcParams)
    with matplotlib.rc_context(style.rc_params()):
        fig, ax = plt.subplots()
        try:
            assert fig.get_facecolor() == to_rgba(style.surface)
            assert ax.get_facecolor() == to_rgba(style.surface)
            cycle = matplotlib.rcParams["axes.prop_cycle"].by_key()
            assert [to_hex(c) for c in cycle["color"]] == list(style.palette)
            assert list(cycle["marker"]) == PAPER_MARKERS
            ax.plot([0, 1], [0, 1])
            ax.set_title("t")
            assert to_hex(ax.title.get_color()) == style.ink
            buffer = io.StringIO()
            fig.savefig(buffer, format="svg")
            svg = buffer.getvalue()
            assert style.surface in svg
            assert style.palette[0] in svg
        finally:
            plt.close(fig)
    assert dict(matplotlib.rcParams) == before


# --- marker-shape and fill encodings -----------------------------------------


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_six_variants_stay_distinct_without_colour(theme):
    styles = [variant_style(a, m, theme=theme) for a in ADJUSTMENTS for m in DRAW_MODES]
    assert len(styles) == 6
    assert len({(s.marker, s.filled) for s in styles}) == 6
    for adjustment in ADJUSTMENTS:
        assert len({variant_style(adjustment, m, theme=theme).marker for m in DRAW_MODES}) == 1
    for mode in DRAW_MODES:
        assert len({variant_style(a, mode, theme=theme).filled for a in ADJUSTMENTS}) == 1
    assert len({s.label for s in styles}) == 6


def test_shape_encodes_the_adjustment_and_fill_the_draw_mode():
    assert ADJUSTMENTS == ("none", "top_k", "full_distribution")
    assert DRAW_MODES == ("single", "ensemble")
    assert [variant_style(a, "ensemble", theme="light").marker for a in ADJUSTMENTS] == ["o", "s", "^"]
    assert variant_style("top_k", "single", theme="light").filled is False
    assert variant_style("top_k", "ensemble", theme="light").filled is True


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_colour_repeats_the_adjustment_from_the_all_pairs_prefix(theme):
    style = figure_theme(theme)
    for index, adjustment in enumerate(ADJUSTMENTS):
        filled = variant_style(adjustment, "ensemble", theme=theme)
        hollow = variant_style(adjustment, "single", theme=theme)
        assert index < ALL_PAIRS_SLOT_CAP
        assert filled.color == hollow.color == style.palette[index]
        assert filled.facecolor == filled.color
        assert hollow.facecolor == style.surface


def test_variant_labels_name_adjustment_and_draw_mode():
    assert variant_style("none", "single", theme="light").label == "no adjustment, single draw"
    assert variant_style("top_k", "ensemble", theme="light").label == "top-k, ensemble"
    assert variant_style("full_distribution", "ensemble", theme="dark").label == "full distribution, ensemble"


@pytest.mark.parametrize(
    ("adjustment", "draw_mode", "message"),
    [
        ("full", "ensemble", "none, top_k, full_distribution"),
        ("top_k", "mean", "single, ensemble"),
    ],
)
def test_variant_style_rejects_unknown_adjustment_or_draw_mode(adjustment, draw_mode, message):
    with pytest.raises(ValueError, match=message):
        variant_style(adjustment, draw_mode, theme="light")


def test_variant_style_rejects_an_unknown_theme():
    with pytest.raises(ValueError, match="light, dark"):
        variant_style("none", "single", theme="sepia")


@pytest.mark.parametrize("theme", ["light", "dark"])
def test_marker_and_scatter_kwargs_draw_open_and_filled_markers(theme):
    hollow = variant_style("full_distribution", "single", theme=theme)
    filled = variant_style("full_distribution", "ensemble", theme=theme)
    fig, ax = plt.subplots()
    try:
        container = ax.errorbar([0.1], [0], xerr=[[0.05], [0.05]], **hollow.marker_kwargs())
        (line,) = ax.plot([0.2], [1], **filled.marker_kwargs())
        assert line.get_marker() == "^"
        assert line.get_linestyle() == "None"
        assert to_hex(line.get_markerfacecolor()) == filled.color
        assert to_hex(line.get_markeredgecolor()) == filled.color
        hollow_line = container.lines[0]
        assert hollow_line.get_marker() == "^"
        assert to_hex(hollow_line.get_markerfacecolor()) == figure_theme(theme).surface
        assert to_hex(hollow_line.get_markeredgecolor()) == hollow.color
        points = ax.scatter([0.3], [2], **hollow.scatter_kwargs())
        assert to_hex(points.get_facecolors()[0]) == figure_theme(theme).surface
        assert to_hex(points.get_edgecolors()[0]) == hollow.color
        assert hollow.label in [t for t in ax.get_legend_handles_labels()[1]]
    finally:
        plt.close(fig)


# --- reliability helper ------------------------------------------------------


def test_reliability_curve_reproduces_hand_computed_bins_and_ece():
    curve = reliability_curve([0.05, 0.15, 0.15, 0.95, 1.0], [0, 0, 1, 1, 1], n_bins=10)
    assert curve.n == 5
    assert curve.n_bins == 10
    assert [(b.low, b.high, b.count) for b in curve.bins] == [
        (0.0, 0.1, 1),
        (0.1, 0.2, 2),
        (0.9, 1.0, 2),
    ]
    assert [b.mean_predicted for b in curve.bins] == pytest.approx([0.05, 0.15, 0.975])
    assert [b.observed_share for b in curve.bins] == pytest.approx([0.0, 0.5, 1.0])
    # ECE = (1 * 0.05 + 2 * 0.35 + 2 * 0.025) / 5
    assert curve.ece == pytest.approx(0.16)


def test_reliability_curve_lists_only_non_empty_bins_and_keeps_one_in_the_last():
    curve = reliability_curve([1.0, 1.0, 0.0], [1, 0, 0], n_bins=4)
    assert curve.bins == (
        ReliabilityBin(low=0.0, high=0.25, count=1, mean_predicted=0.0, observed_share=0.0),
        ReliabilityBin(low=0.75, high=1.0, count=2, mean_predicted=1.0, observed_share=0.5),
    )


def test_reliability_curve_matches_the_local_report_calibration():
    rng = np.random.default_rng(7)
    labels = rng.integers(0, 2, size=60)
    probabilities = np.clip(0.35 * labels + rng.uniform(0.0, 0.65, size=60), 0.0, 1.0)
    curve = reliability_curve(probabilities.tolist(), labels.tolist())
    reference = local_reporting.held_out_calibration(probabilities, labels, n_resamples=10)
    assert curve.n_bins == local_reporting.CALIBRATION_BINS
    assert curve.ece == pytest.approx(reference["ece"], rel=1e-12)
    assert [
        (b.low, b.high, b.count, pytest.approx(b.mean_predicted), pytest.approx(b.observed_share))
        for b in curve.bins
    ] == [
        (r["low"], r["high"], r["count"], r["mean_predicted"], r["observed_memorized_fraction"])
        for r in reference["reliability_bins"]
    ]


def test_reliability_curve_from_a_calibrators_retained_holdout_pairs():
    mcs = _calibrator()
    curve = reliability_curve(mcs.holdout_probabilities, mcs.holdout_labels)
    assert curve.n == 12
    assert sum(b.count for b in curve.bins) == 12


@pytest.mark.parametrize(
    ("probabilities", "labels", "n_bins", "message"),
    [
        (None, None, 10, "held-out probabilities and labels"),
        ([0.2, 0.4], [1], 10, "same length"),
        ([], [], 10, "at least one"),
        ([-0.1], [0], 10, r"\[0, 1\]"),
        ([1.5], [1], 10, r"\[0, 1\]"),
        ([float("nan")], [1], 10, r"\[0, 1\]"),
        ([0.5], [2], 10, "0 or 1"),
        ([0.5], [1], 0, "positive integer"),
        ([0.5], [1], True, "positive integer"),
    ],
)
def test_reliability_curve_rejects_invalid_input(probabilities, labels, n_bins, message):
    with pytest.raises(ValueError, match=message):
        reliability_curve(probabilities, labels, n_bins=n_bins)


# --- corrected note and lazy import ------------------------------------------


def test_reliability_note_no_longer_claims_the_curve_cannot_be_built():
    for text in (plots.__doc__, plot_mcs_calibration.__doc__):
        assert "not constructible" not in text
        assert "cannot be reconstructed" not in text
        assert "holdout_probabilities" in text
        assert "reliability_curve" in text


def test_figure_helpers_stay_off_the_package_import_path():
    code = (
        "import sys, recall_guard, recall_guard.harness\n"
        "heavy = sorted(m for m in sys.modules if m.split('.')[0] == 'matplotlib')\n"
        "assert not heavy, heavy\n"
        "names = {'figure_theme', 'variant_style', 'reliability_curve', 'STUDY_PALETTE', 'STUDY_PALETTE_DARK'}\n"
        "assert not names & set(recall_guard.__all__), recall_guard.__all__\n"
        "from recall_guard.harness.plots import figure_theme\n"
        "assert 'matplotlib' in sys.modules\n"
        "print('lazy-ok')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "lazy-ok" in result.stdout
