"""Tests for harness.plots.

Covers Requirements 12.3, 12.4, 12.5 of the honest-model-ranking spec — the
paper-ready matplotlib figure generators that consume harness dataclasses
and produce single-column-width vector figures for the qualification
notebook.

Tests assert structural properties only: returned object is a
``matplotlib.figure.Figure``; figure size matches the paper width
(3.5 inches); axes carry non-empty x/y labels; ``fig.savefig(...pdf)``
round-trips. Pixel content is not asserted.

The Agg backend is selected at import time so the suite can run headless
in CI.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure
from sklearn.linear_model import LogisticRegression

from recall_guard.harness.evaluator import CIBound, ModelEvalResult, Record
from recall_guard.harness.plots import (
    PAPER_PALETTE,
    configure_paper_style,
    plot_accuracy_with_ci,
    plot_composite_ranking,
    plot_mcs_auc_with_ci,
    plot_mcs_calibration,
    plot_mia_feature_distributions,
)
from recall_guard.harness.ranker import CompositeScore
from recall_guard.mia.features import MiaFeatures
from recall_guard.mia.mcs import MCSCalibrator

# --- Synthetic fixtures -------------------------------------------------------


def _synthetic_features(
    *,
    loss: float = 1.0,
    min_k: float = -1.5,
    min_k_pp: float = -0.5,
    zlib_ratio: float = 0.4,
    ref_delta: float | None = -0.2,
) -> MiaFeatures:
    return MiaFeatures(
        loss=loss,
        min_k=min_k,
        min_k_pp=min_k_pp,
        zlib_ratio=zlib_ratio,
        ref_delta=ref_delta,
    )


def _synthetic_record(
    *,
    parse_ok: bool = True,
    features: MiaFeatures | None = None,
    target: int = 1,
    p_memorized: float | None = 0.7,
    model: str = "model-a",
    prompt_hash: str = "abc123def4567890",
) -> Record:
    feats = features if features is not None else _synthetic_features()
    return Record(
        model=model,
        prompt_hash=prompt_hash,
        parse_ok=parse_ok,
        predicted_direction=1 if parse_ok else None,
        raw_confidence=0.8 if parse_ok else None,
        penalized_confidence=0.5 if parse_ok else None,
        target_direction=target,
        features_raw=feats if parse_ok else None,
        features_standardised={
            "loss": 0.1,
            "min_k": -0.2,
            "min_k_pp": 0.0,
            "zlib_ratio": 0.3,
            "ref_delta": 0.05,
        }
        if parse_ok
        else None,
        p_memorized=p_memorized if parse_ok else None,
        fail_reason=None if parse_ok else "parse_failure",
    )


def _synthetic_is_records(n: int = 10) -> list[Record]:
    rng = np.random.default_rng(0)
    out: list[Record] = []
    for i in range(n):
        feats = _synthetic_features(
            loss=float(rng.normal(0.5, 0.2)),
            min_k=float(rng.normal(-2.0, 0.5)),
            min_k_pp=float(rng.normal(-1.0, 0.3)),
            zlib_ratio=float(rng.normal(0.5, 0.1)),
            ref_delta=float(rng.normal(-0.5, 0.2)),
        )
        out.append(_synthetic_record(features=feats, target=1, prompt_hash=f"is{i:04d}{'0'*10}"))
    return out


def _synthetic_oos_records(n: int = 10, with_ref_delta: bool = True) -> list[Record]:
    rng = np.random.default_rng(1)
    out: list[Record] = []
    for i in range(n):
        rd: float | None
        if with_ref_delta:
            rd = float(rng.normal(0.3, 0.2))
        else:
            rd = None
        feats = _synthetic_features(
            loss=float(rng.normal(2.0, 0.5)),
            min_k=float(rng.normal(-1.0, 0.5)),
            min_k_pp=float(rng.normal(0.0, 0.3)),
            zlib_ratio=float(rng.normal(0.7, 0.15)),
            ref_delta=rd,
        )
        out.append(_synthetic_record(features=feats, target=0, prompt_hash=f"oos{i:04d}{'0'*9}"))
    return out


def _synthetic_result(
    name: str = "model-a",
    *,
    raw_acc: tuple[float, float, float] = (0.7, 0.6, 0.8),
    mg_acc: tuple[float, float, float] = (0.7, 0.6, 0.8),
    auc: tuple[float, float, float] = (0.85, 0.75, 0.95),
    parse: float = 1.0,
    warnings: tuple[str, ...] = (),
) -> ModelEvalResult:
    return ModelEvalResult(
        model=name,
        raw_accuracy=CIBound(*raw_acc),
        memguard_accuracy=CIBound(*mg_acc),
        mcs_auc=CIBound(*auc),
        parse_success_rate=parse,
        parse_failures=0,
        warnings=list(warnings),
        records=[],
    )


def _synthetic_mcs(
    *, model: str = "model-a", holdout_auc: float = 0.82, is_weak: bool = False
) -> MCSCalibrator:
    classifier = LogisticRegression(solver="liblinear", random_state=0)
    # Fit on a tiny synthetic dataset so the underlying estimator is real.
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 4))
    y = (x[:, 0] + rng.normal(scale=0.1, size=40) > 0).astype(int)
    classifier.fit(x, y)
    return MCSCalibrator(
        model=model,
        classifier=classifier,
        feature_order=["loss", "min_k", "min_k_pp", "zlib_ratio"],
        holdout_auc=holdout_auc,
        is_weak=is_weak,
    )


def _synthetic_compositescore(
    name: str = "model-a",
    *,
    score: float = 0.42,
    survives: bool = True,
    warnings: tuple[str, ...] = (),
    components: dict[str, float] | None = None,
) -> CompositeScore:
    comps = components or {
        "memguard_acc_lo": 0.6,
        "mcs_auc_point": 0.85,
        "parse_success_rate": 0.95,
    }
    return CompositeScore(
        model=name,
        score=score,
        components=comps,
        survives_gates=survives,
        warnings=list(warnings),
    )


# --- configure_paper_style ----------------------------------------------------


def test_configure_paper_style_sets_rcparams():
    # Mutate first to ensure configure overrides defaults.
    plt.rcParams["figure.figsize"] = (4.0, 3.0)
    plt.rcParams["font.size"] = 12
    plt.rcParams["savefig.dpi"] = 100

    configure_paper_style()

    assert tuple(plt.rcParams["figure.figsize"]) == (3.5, 2.5)
    assert plt.rcParams["font.size"] == 8
    assert plt.rcParams["savefig.dpi"] == 300
    assert plt.rcParams["savefig.format"] == "pdf"
    assert plt.rcParams["savefig.bbox"] == "tight"

    cycle = plt.rcParams["axes.prop_cycle"]
    colors = cycle.by_key().get("color", [])
    assert list(colors) == PAPER_PALETTE
    # Marker cycle for B&W reproduction.
    markers = cycle.by_key().get("marker", [])
    assert len(markers) == len(PAPER_PALETTE)


# --- plot_mia_feature_distributions ------------------------------------------


@pytest.mark.parametrize(
    "feature", ["loss", "min_k", "min_k_pp", "zlib_ratio", "ref_delta"]
)
def test_plot_mia_feature_distributions_returns_figure(feature):
    configure_paper_style()
    is_records = _synthetic_is_records()
    oos_records = _synthetic_oos_records()

    fig = plot_mia_feature_distributions(is_records, oos_records, feature)

    try:
        assert isinstance(fig, Figure)
        assert tuple(fig.get_size_inches()) == pytest.approx((3.5, 2.5))
        ax = fig.axes[0]
        assert ax.get_xlabel() != ""
        assert ax.get_ylabel() != ""
        legend = ax.get_legend()
        assert legend is not None
        labels = [t.get_text() for t in legend.get_texts()]
        assert any("IS" in lab for lab in labels)
        assert any("OOS" in lab for lab in labels)
    finally:
        plt.close(fig)


def test_plot_mia_feature_distributions_handles_ref_delta_with_none():
    """ref_delta entries equal to None must be dropped, not crash the plot."""
    configure_paper_style()
    is_records = _synthetic_is_records(n=8)
    # Mix: half OOS records have ref_delta=None.
    oos_with_ref = _synthetic_oos_records(n=4, with_ref_delta=True)
    oos_without_ref = _synthetic_oos_records(n=4, with_ref_delta=False)
    oos_records = oos_with_ref + oos_without_ref

    fig = plot_mia_feature_distributions(is_records, oos_records, "ref_delta")
    try:
        assert isinstance(fig, Figure)
        ax = fig.axes[0]
        assert ax.get_xlabel() != ""
    finally:
        plt.close(fig)


def test_plot_mia_feature_distributions_skips_failed_parse_records():
    """Records with parse_ok=False or features_raw=None must be filtered out."""
    configure_paper_style()
    is_records = _synthetic_is_records(n=6) + [
        _synthetic_record(parse_ok=False, target=1)
    ]
    oos_records = _synthetic_oos_records(n=6)

    fig = plot_mia_feature_distributions(is_records, oos_records, "loss")
    try:
        assert isinstance(fig, Figure)
    finally:
        plt.close(fig)


# --- plot_mcs_calibration -----------------------------------------------------


def test_plot_mcs_calibration_returns_figure():
    configure_paper_style()
    mcs = _synthetic_mcs(holdout_auc=0.82, is_weak=False)

    fig = plot_mcs_calibration(mcs)

    try:
        assert isinstance(fig, Figure)
        assert tuple(fig.get_size_inches()) == pytest.approx((3.5, 2.5))
        ax = fig.axes[0]
        assert ax.get_xlabel() != ""
        assert ax.get_ylabel() != ""
        # AUC value should be rendered somewhere on the figure (annotation
        # or title).
        rendered = ax.get_title() + " " + " ".join(
            t.get_text() for t in ax.texts
        )
        assert "0.82" in rendered or "0.820" in rendered
    finally:
        plt.close(fig)


def test_plot_mcs_calibration_marks_weak_flag():
    configure_paper_style()
    mcs_weak = _synthetic_mcs(holdout_auc=0.55, is_weak=True)
    fig = plot_mcs_calibration(mcs_weak)
    try:
        ax = fig.axes[0]
        rendered = " ".join(t.get_text() for t in ax.texts).lower()
        title = ax.get_title().lower()
        assert "weak" in (rendered + " " + title)
    finally:
        plt.close(fig)


# --- plot_accuracy_with_ci ----------------------------------------------------


def test_plot_accuracy_with_ci_returns_figure():
    configure_paper_style()
    results = [
        _synthetic_result("model-a", mg_acc=(0.7, 0.6, 0.8)),
        _synthetic_result("model-b", mg_acc=(0.65, 0.55, 0.75)),
        _synthetic_result("model-c", mg_acc=(0.55, 0.45, 0.65)),
    ]
    majority = CIBound(point=0.5, lo=0.45, hi=0.55)

    fig = plot_accuracy_with_ci(results, majority)

    try:
        assert isinstance(fig, Figure)
        assert tuple(fig.get_size_inches()) == pytest.approx((3.5, 2.5))
        ax = fig.axes[0]
        assert ax.get_ylabel() != ""
        # 3 bars + at least one horizontal line for majority baseline.
        # matplotlib.patches.Rectangle is used by ax.bar.
        from matplotlib.patches import Rectangle

        bars = [p for p in ax.patches if isinstance(p, Rectangle)]
        # At least 3 visible bars (the shaded baseline band may add one more).
        assert len(bars) >= 3
        # Tick labels include model names.
        tick_labels = [t.get_text() for t in ax.get_xticklabels()]
        assert "model-a" in tick_labels
        assert "model-b" in tick_labels
        assert "model-c" in tick_labels
    finally:
        plt.close(fig)


# --- plot_mcs_auc_with_ci -----------------------------------------------------


def test_plot_mcs_auc_with_ci_returns_figure():
    configure_paper_style()
    results = [
        _synthetic_result("model-a", auc=(0.85, 0.75, 0.95)),
        _synthetic_result("model-b", auc=(0.65, 0.55, 0.75)),
        _synthetic_result("model-c", auc=(0.55, 0.40, 0.65)),
    ]

    fig = plot_mcs_auc_with_ci(results)

    try:
        assert isinstance(fig, Figure)
        assert tuple(fig.get_size_inches()) == pytest.approx((3.5, 2.5))
        ax = fig.axes[0]
        assert ax.get_ylabel() != ""
        ymin, ymax = ax.get_ylim()
        assert ymin <= 0.0 + 1e-9
        assert ymax >= 1.0 - 1e-9
        # 0.5 (random) and 0.6 (gate) reference y-values must be drawn as
        # horizontal lines.
        line_ys: set[float] = set()
        for line in ax.get_lines():
            ys = line.get_ydata()
            arr = np.asarray(ys, dtype=float).ravel()
            if arr.size and float(arr.min()) == float(arr.max()):
                line_ys.add(round(float(arr[0]), 3))
        assert 0.5 in line_ys
        assert 0.6 in line_ys
    finally:
        plt.close(fig)


# --- plot_composite_ranking ---------------------------------------------------


def test_plot_composite_ranking_returns_figure():
    configure_paper_style()
    scores = [
        _synthetic_compositescore("model-a", score=0.6, survives=True),
        _synthetic_compositescore(
            "model-b",
            score=0.0,
            survives=False,
            warnings=("weak-calibration",),
        ),
        _synthetic_compositescore("model-c", score=0.3, survives=True),
    ]

    fig = plot_composite_ranking(scores)

    try:
        assert isinstance(fig, Figure)
        assert tuple(fig.get_size_inches()) == pytest.approx((3.5, 2.5))
        ax = fig.axes[0]
        assert ax.get_xlabel() != ""
        # Y-tick labels carry model names in descending-score order:
        # [model-a, model-c, model-b].
        ytick_labels = [t.get_text() for t in ax.get_yticklabels()]
        # bar order: top of plot is the last index in barh; matplotlib lays
        # out indices bottom-up. Descending-score model-a should be at the
        # top of the chart, i.e. last in the ytick label sequence.
        assert ytick_labels == ["model-b", "model-c", "model-a"]
    finally:
        plt.close(fig)


# --- Cross-cutting checks -----------------------------------------------------


def test_all_plots_savefig_to_pdf_roundtrip(tmp_path):
    configure_paper_style()
    is_records = _synthetic_is_records()
    oos_records = _synthetic_oos_records()
    results = [
        _synthetic_result("model-a"),
        _synthetic_result("model-b"),
    ]
    majority = CIBound(point=0.5, lo=0.45, hi=0.55)
    mcs = _synthetic_mcs()
    scores = [
        _synthetic_compositescore("model-a", score=0.6),
        _synthetic_compositescore("model-b", score=0.4),
    ]

    figures = [
        ("mia.pdf", plot_mia_feature_distributions(is_records, oos_records, "loss")),
        ("mcs_cal.pdf", plot_mcs_calibration(mcs)),
        ("acc.pdf", plot_accuracy_with_ci(results, majority)),
        ("mcs_auc.pdf", plot_mcs_auc_with_ci(results)),
        ("rank.pdf", plot_composite_ranking(scores)),
    ]

    try:
        for fname, fig in figures:
            out = tmp_path / fname
            fig.savefig(out)
            assert out.exists(), f"{fname} not written"
            assert out.stat().st_size >= 100, f"{fname} too small ({out.stat().st_size}b)"
    finally:
        for _, fig in figures:
            plt.close(fig)


def test_all_plots_have_non_empty_axis_labels():
    configure_paper_style()
    is_records = _synthetic_is_records()
    oos_records = _synthetic_oos_records()
    results = [_synthetic_result("model-a"), _synthetic_result("model-b")]
    majority = CIBound(point=0.5, lo=0.45, hi=0.55)
    mcs = _synthetic_mcs()
    scores = [_synthetic_compositescore("model-a", score=0.4)]

    figs = [
        plot_mia_feature_distributions(is_records, oos_records, "loss"),
        plot_mcs_calibration(mcs),
        plot_accuracy_with_ci(results, majority),
        plot_mcs_auc_with_ci(results),
        plot_composite_ranking(scores),
    ]
    try:
        for fig in figs:
            ax = fig.axes[0]
            assert ax.get_xlabel() != "" or ax.get_ylabel() != ""
            # Specifically: every plot has at least one of x or y labelled,
            # plus a title.
            assert ax.get_title() != ""
    finally:
        for fig in figs:
            plt.close(fig)


def test_figure_size_matches_paper_width():
    configure_paper_style()
    is_records = _synthetic_is_records()
    oos_records = _synthetic_oos_records()
    results = [_synthetic_result("model-a")]
    majority = CIBound(point=0.5, lo=0.45, hi=0.55)
    mcs = _synthetic_mcs()
    scores = [_synthetic_compositescore("model-a", score=0.4)]

    figs = [
        plot_mia_feature_distributions(is_records, oos_records, "loss"),
        plot_mcs_calibration(mcs),
        plot_accuracy_with_ci(results, majority),
        plot_mcs_auc_with_ci(results),
        plot_composite_ranking(scores),
    ]
    try:
        for fig in figs:
            assert tuple(fig.get_size_inches()) == pytest.approx((3.5, 2.5))
    finally:
        for fig in figs:
            plt.close(fig)
