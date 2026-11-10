"""Accuracy-gap and reliability figures (task 8.2).

Requirements 10.2, 10.3 and 10.6.

Fixtures only: the probe (sampling) arm is calibrated and executed through the
real 7.6 and 7.7 path on the generation and full-distribution doubles of the
7.7 tests, and both arms' pre-registered analyses (7.10) are written as JSON
under ``tmp_path``; the figures are read from those files and the calibrator
files the calibration wrote beside them. Sockets are blocked, no model host or
credential is used, and nothing is written under ``runs/`` or ``docs/``. The
Agg backend is selected before anything imports pyplot.
"""

from __future__ import annotations

import copy
import csv
import io
import json
import math
import shutil
import socket
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.markers import MarkerStyle  # noqa: E402
from matplotlib.text import Annotation, Text  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

from recall_guard.dataset.macro_regime import render_anonymised, render_identifying  # noqa: E402
from recall_guard.harness.local_reporting import CALIBRATION_MIN_PER_CLASS, held_out_calibration  # noqa: E402
from recall_guard.harness.plots import (  # noqa: E402
    ALL_PAIRS_SLOT_CAP,
    STUDY_PALETTE,
    STUDY_PALETTE_DARK,
    THEMES,
    figure_theme,
    reliability_curve,
    variant_style,
)
from scripts.finance_sim import analysis as analysis_module  # noqa: E402
from scripts.finance_sim import figures  # noqa: E402
from scripts.finance_sim.analysis import (  # noqa: E402
    CONDITIONAL_NOTE,
    POWER_APPROXIMATION,
    analysis_path,
    pre_registered_analysis,
    write_analysis,
)
from scripts.finance_sim.calibrate import (  # noqa: E402
    DrawSettings,
    FitSettings,
    calibrator_path,
    load_calibrator,
    run_calibration,
)
from scripts.finance_sim.execute import execute_items, load_calibrators, macro_items  # noqa: E402
from scripts.finance_sim.figures import (  # noqa: E402
    DEFAULT_ARMS,
    FIGURE_NAMES,
    FIGURES_DIR,
    FigureError,
    figure_paths,
    figure_svg,
    gap_panel,
    gap_table,
    held_out_reliability,
    main,
    plot_gap,
    plot_reliability,
    read_gap_panel,
    reliability_panel,
    reliability_panels,
    reliability_table,
    render_figures,
)
from scripts.finance_sim.limits import BudgetLedger, LedgerCallBudget, StudyLimits  # noqa: E402
from scripts.finance_sim.plan import SERVING_FULL_DISTRIBUTION_REASON, VARIANTS, write_plan  # noqa: E402
from tests.scripts.finance_sim.test_analysis import FAST, serving_results  # noqa: E402
from tests.scripts.finance_sim.test_execute import (  # noqa: E402
    ALLOWANCE,
    MACRO_ITEMS,
    MISSING_MONTH,
    MONTHS,
    PLAN,
    SEED_BASE,
    TEMPERATURE,
    Clock,
    FakeFull,
    FakeHost,
    edgar_releases,
    macro_row,
    realized,
)

ROOT = Path(__file__).resolve().parents[3]
COMPUTED_SERVING = ("single_none", "single_top_k", "ensemble_none", "ensemble_top_k")
SERVING_MISSING = ("single_full_distribution", "ensemble_full_distribution")
SOURCES = ("top_k", "full_distribution")
KINDS = ("macro", "edgar")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("the figures attempted a network connection")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)


@pytest.fixture(scope="module")
def study(tmp_path_factory) -> SimpleNamespace:
    """Both arms' stored 7.10 analyses and the probe arm's calibrator files, all under tmp_path."""
    root = tmp_path_factory.mktemp("figures-study")
    calibrators_dir, analysis_dir = root / "calibrators", root / "analysis"
    ledger = BudgetLedger.open(root / "calibration" / "budget.json", PLAN,
                               StudyLimits(PLAN.plan_sha256, "sampling", "calibration", 1000, 1_000_000, 1_000_000),
                               ALLOWANCE, clock=Clock())
    run_calibration(PLAN, arm="sampling", macro_rows=[macro_row(month) for month in PLAN.months.calibration],
                    edgar_releases=edgar_releases(), client=FakeHost(LedgerCallBudget(ledger)), ledger=ledger,
                    settings=DrawSettings(draws=2, temperature=TEMPERATURE, seed_base=SEED_BASE),
                    sources=SOURCES, full_distribution=FakeFull(),
                    fit=FitSettings(min_auc=0.6, min_valid=10), calibrators_dir=calibrators_dir,
                    sleep=lambda seconds: None)
    calibrators = load_calibrators(PLAN, arm="sampling", kind="macro", calibrators_dir=calibrators_dir)
    budget = PLAN.limits.stage("sampling", "full_run")
    ledger = BudgetLedger.open(root / "full_run" / "budget.json", PLAN,
                               StudyLimits(PLAN.plan_sha256, "sampling", "full_run", budget.max_requests,
                                           budget.max_output_tokens, budget.max_seconds), ALLOWANCE, clock=Clock())
    run = execute_items(PLAN, macro_items(PLAN, MACRO_ITEMS, stage="full_run", arm="sampling"), arm="sampling",
                        stage="full_run", client=FakeHost(LedgerCallBudget(ledger)), ledger=ledger,
                        calibrators=calibrators, full_distribution=FakeFull(), results_dir=root / "results",
                        sleep=lambda seconds: None)
    assert run.status == "complete"
    write_analysis(pre_registered_analysis(PLAN, run.results, arm="sampling"), analysis_dir)
    write_analysis(pre_registered_analysis(PLAN, serving_results(PLAN), arm="serving"), analysis_dir)
    return SimpleNamespace(root=root, analysis_dir=analysis_dir, calibrators_dir=calibrators_dir)


def stored(study: SimpleNamespace, arm: str) -> dict:
    return json.loads(analysis_path(arm, "full_run", study.analysis_dir).read_text(encoding="utf-8"))


def metric(data: dict, name: str) -> dict:
    return next(entry for entry in data["metrics"] if entry["metric"] == name)


def cell(entry: dict, variant: str) -> dict:
    return next(value for value in entry["variants"] if value["variant"] == variant)


@pytest.fixture(scope="module")
def panels(study) -> tuple:
    return tuple(read_gap_panel(analysis_path(arm, "full_run", study.analysis_dir), plan_sha256=PLAN.plan_sha256)
                 for arm in ("serving", "sampling"))


def csv_rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text)))


def gids(figure, kind=Line2D) -> set[str]:
    return {artist.get_gid() for artist in figure.findobj(kind) if artist.get_gid()}


def by_gid(figure, gid: str):
    return next(artist for artist in figure.findobj() if artist.get_gid() == gid)


# ---------- F1: the accuracy gap, read from the stored analysis ----------

def test_the_gap_figure_reads_every_number_from_the_stored_analysis_and_recomputes_nothing(study, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("the figure recomputed the analysis instead of reading it")

    for name in ("pre_registered_analysis", "analysis_from_records", "block_bootstrap", "did_gap",
                 "paired_loss_difference"):
        monkeypatch.setattr(analysis_module, name, forbidden)
    for arm in ("serving", "sampling"):
        data = stored(study, arm)
        panel = read_gap_panel(analysis_path(arm, "full_run", study.analysis_dir), plan_sha256=PLAN.plan_sha256)
        assert (panel.arm, panel.role, panel.plan_sha256) == (arm, data["role"], PLAN.plan_sha256)
        assert [point.variant for point in panel.points] == data["variants"] == list(VARIANTS)
        did, pre, post = (metric(data, name) for name in ("did_accuracy_gap", "accuracy_difference_pre_cutoff",
                                                          "accuracy_difference_post_cutoff"))
        for point in panel.points:
            source = cell(did, point.variant)
            interval = source["interval"] or {"lo": None, "hi": None}
            assert (point.statistic, point.low, point.high) == (source["statistic"], interval["lo"], interval["hi"])
            assert (point.p_value, point.p_holm) == (source["p_value"], source["p_holm"])
            assert (point.months, point.pre_cutoff_months, point.post_cutoff_months) == (
                source["months"], source["pre_cutoff_months"], source["post_cutoff_months"])
            assert point.dropped_resamples == source["dropped_resamples"]
            assert point.missing_months == len(source["missing_months"])
            assert point.missing == source["missing"]
            assert point.pre_difference == cell(pre, point.variant)["statistic"]
            assert point.post_difference == cell(post, point.variant)["statistic"]
            assert dict(point.statuses) == source["statuses"]
    # The serving fixture has a failed identifying ensemble and a month without realized returns: both months are
    # counted as missing in the table, never substituted.
    serving = read_gap_panel(analysis_path("serving", "full_run", study.analysis_dir))
    expected = len(cell(metric(stored(study, "serving"), "did_accuracy_gap"), "single_top_k")["missing_months"])
    assert serving.point("single_top_k").missing_months == expected == 2
    row = next(row for row in csv_rows(gap_table([serving]).to_csv()) if row["variant"] == "single_top_k")
    assert row["missing_months"] == "2" and row["missing"] == ""
    # A changed stored value is what the table shows: the figure reads the file, it does not recompute it.
    changed = copy.deepcopy(stored(study, "serving"))
    cell(metric(changed, "did_accuracy_gap"), "single_top_k")["statistic"] = 0.123456
    row = next(row for row in csv_rows(gap_table([gap_panel(changed)]).to_csv()) if row["variant"] == "single_top_k")
    assert row["gap"] == "0.123456"


def test_the_serving_arms_missing_variants_are_listed_in_the_table_and_caption_and_not_drawn(panels):
    table = gap_table(panels)
    rows = {(row["arm"], row["variant"]): row for row in csv_rows(table.to_csv())}
    assert len(rows) == 12
    for variant in SERVING_MISSING:
        row = rows[("serving", variant)]
        assert row["missing"] == SERVING_FULL_DISTRIBUTION_REASON
        assert row["gap"] == row["interval_low"] == row["interval_high"] == row["p_raw"] == ""
    for variant in COMPUTED_SERVING:
        assert rows[("serving", variant)]["missing"] == "" and rows[("serving", variant)]["gap"] != ""
    assert SERVING_FULL_DISTRIBUTION_REASON in table.caption
    assert "serving single_full_distribution, ensemble_full_distribution" in table.caption
    figure = plot_gap(panels, theme="light")
    drawn = {f"gap-serving-{variant}" for variant in COMPUTED_SERVING} | {f"gap-sampling-{variant}"
                                                                         for variant in VARIANTS}
    assert {gid for gid in gids(figure) if gid.startswith("gap-") and "interval" not in gid
            and "zero" not in gid} == drawn
    svg = figure_svg(figure)
    assert 'id="gap-serving-single_top_k"' in svg and 'id="gap-sampling-ensemble_full_distribution"' in svg
    for variant in SERVING_MISSING:
        assert f'id="gap-serving-{variant}"' not in svg and f'id="gap-interval-serving-{variant}"' not in svg


def test_the_arm_roles_are_labelled_and_the_conditional_note_sits_beside_the_probe_arm(panels):
    table = gap_table(panels)
    assert "Serving arm (primary role, the pre-registered tests" in table.caption
    assert "Sampling arm (probe role, a small paired check" in table.caption
    assert CONDITIONAL_NOTE in table.caption and CONDITIONAL_NOTE in table.to_markdown()
    rows = {(row["arm"], row["variant"]): row for row in csv_rows(table.to_csv())}
    assert {row["role"] for (arm, _), row in rows.items() if arm == "serving"} == {"primary"}
    assert {row["role"] for (arm, _), row in rows.items() if arm == "sampling"} == {"probe"}
    probe = gap_table([panels[1]])
    assert CONDITIONAL_NOTE in probe.caption  # the probe arm alone still carries the note
    figure = plot_gap(panels, theme="light")
    titles = [axes.get_title(loc="left") for axes in figure.axes if axes.get_title(loc="left")]
    assert len(titles) == 2
    assert "pre-registered tests" in titles[0] and "small paired check" in titles[1]
    assert "conditional" in titles[1]
    dropped = max(point.dropped_resamples for point in panels[1].points if point.dropped_resamples is not None)
    assert dropped > 0 and f"up to {dropped} of {PLAN.analysis.n_resamples}" in titles[1]


def test_achieved_power_is_reported_with_its_label_and_its_monte_carlo_error(study, panels):
    caption = gap_table(panels).caption
    for panel in panels:
        power = metric(stored(study, panel.arm), "achieved_power")["power"]
        assert panel.power == power["power"] and panel.power_n_sim == power["n_sim"]
        assert power["approximation"] == POWER_APPROXIMATION
        if power["power"] is None:
            assert panel.power_reason in caption
            continue
        error = math.sqrt(power["power"] * (1.0 - power["power"]) / power["n_sim"])
        assert panel.power_monte_carlo_error == pytest.approx(error)
        assert f"achieved power {power['power']:.3f}" in caption
        assert f"Monte Carlo standard error {error:.3f} over {power['n_sim']} simulations" in caption
    assert f"read as {POWER_APPROXIMATION}" in caption


def test_the_gap_caption_states_data_sample_size_and_interval_method(panels):
    caption = gap_table(panels).caption
    analysis = PLAN.analysis
    assert caption.startswith("Figure F1.")
    assert "positive control" in caption and "identifying versus anonymised FRED macro prompts" in caption
    assert "google/gemma-3-4b-it" in caption and PLAN.plan_sha256[:12] in caption
    for panel in panels:
        assert (f"{panel.planned_months} planned months ({panel.pre_cutoff_months} on or before the model's cutoff, "
                f"{panel.post_cutoff_months} after)") in caption
    assert "months per variant in the table" in caption
    assert (f"{(1 - analysis.alpha) * 100:g} percent non-circular moving-block percentile bootstrap over months, block "
            f"length {analysis.block_length} months, {analysis.n_resamples} resamples, seed {PLAN.seed}") in caption
    assert "Holm-adjusted" in caption
    assert "Moody's" in caption and "BAA10Y" in caption and "Financial Modeling Prep (FMP)" in caption
    assert figures.DERIVED_ONLY in caption


def test_gap_panels_refuse_another_schema_stage_plan_or_a_mix_of_plans(study):
    data = stored(study, "serving")
    for key, value, message in (("schema", "other/1", "schema"), ("stage", "pilot", "stage"),
                                ("arm", "elsewhere", "arm")):
        with pytest.raises(FigureError, match=message):
            gap_panel({**data, key: value})
    with pytest.raises(FigureError, match="plan"):
        gap_panel(data, plan_sha256=FAST.plan_sha256)
    other = gap_panel({**data, "plan_sha256": FAST.plan_sha256})
    with pytest.raises(FigureError, match="one plan"):
        gap_table([gap_panel(data), other])
    with pytest.raises(FigureError, match="twice"):
        gap_table([gap_panel(data), gap_panel(data)])
    path = study.root / "duplicate.json"
    path.write_text('{"schema": 1, "schema": 2}', encoding="utf-8")
    with pytest.raises(FigureError, match="duplicate"):
        read_gap_panel(path)


# ---------- F2: reliability of the memorization estimate ----------

def test_reliability_panels_come_from_the_calibrators_retained_held_out_pairs(study):
    found = reliability_panels(PLAN, arms=("sampling",), calibrators_dir=study.calibrators_dir)
    assert [(panel.kind, panel.source) for panel in found] == [(kind, source) for kind in KINDS for source in SOURCES]
    for panel in found:
        record = load_calibrator(calibrator_path("sampling", panel.source, panel.kind, study.calibrators_dir), PLAN,
                                 arm="sampling", kind=panel.kind, source=panel.source)
        mcs = record.calibrator.mcs
        probabilities, labels = mcs.holdout_probabilities, mcs.holdout_labels
        curve = reliability_curve(probabilities, labels)
        report = held_out_calibration(probabilities, labels)
        assert panel.status == "drawn" and panel.reason is None
        assert panel.bins == curve.bins and panel.ece == curve.ece
        assert panel.ece == pytest.approx(report["ece"], rel=1e-12, abs=1e-15)
        assert [b.count for b in panel.bins] == [b["count"] for b in report["reliability_bins"]]
        assert sum(b.count for b in panel.bins) == panel.holdout_draws == len(probabilities)
        assert (panel.n_memorized, panel.n_control) == (sum(labels), len(labels) - sum(labels))
        assert panel.holdout_prompts == record.counts["holdout_prompts"]
        assert panel.variants == (f"single_{panel.source}", f"ensemble_{panel.source}")
        interval = record.holdout_auc_interval_by_prompt
        assert (panel.auc, panel.auc_low, panel.auc_high) == (interval["auc"], interval["ci_low"], interval["ci_high"])
        assert panel.is_weak == record.calibrator.is_weak
    table = reliability_table(found)
    rows = csv_rows(table.to_csv())
    assert len(rows) == sum(len(panel.bins) for panel in found)
    first = found[0]
    head = next(row for row in rows if (row["kind"], row["source"]) == (first.kind, first.source))
    assert head["ece"] == f"{first.ece:.6g}" and head["bin_count"] == str(first.bins[0].count)
    assert head["variants"] == "single_top_k; ensemble_top_k"


def test_the_reliability_gate_is_the_reports_minimum_per_class():
    assert CALIBRATION_MIN_PER_CLASS == 2
    for n_memorized in range(4):
        for n_control in range(4):
            labels = [1] * n_memorized + [0] * n_control
            probabilities = [0.15 + 0.2 * i for i in range(n_memorized)] + [0.05 + 0.1 * i for i in range(n_control)]
            result = held_out_reliability(probabilities, labels)
            report = held_out_calibration(probabilities, labels)
            assert (result.n_memorized, result.n_control) == (n_memorized, n_control)
            if report["ece"] is None:
                assert result.status == "insufficient" and result.ece is None and result.bins == ()
                assert result.reason.startswith("insufficient:")
                assert f"{n_memorized} memorized and {n_control} control held-out draws" in result.reason
            else:
                assert result.status == "drawn" and result.reason is None
                assert result.ece == pytest.approx(report["ece"])
                assert [b.count for b in result.bins] == [b["count"] for b in report["reliability_bins"]]
    assert held_out_reliability([0.2, 0.9], [0, 1], min_per_class=1).status == "drawn"


def test_an_insufficient_calibrator_is_shown_as_insufficient_in_the_panel_table_and_caption(study):
    path = calibrator_path("sampling", "top_k", "macro", study.calibrators_dir)
    record = load_calibrator(path, PLAN, arm="sampling", kind="macro", source="top_k")
    mcs = replace(record.calibrator.mcs, holdout_labels=(1, 0, 0, 0), holdout_probabilities=(0.8, 0.3, 0.2, 0.1))
    thin = replace(record, calibrator=replace(record.calibrator, mcs=mcs))
    panel = reliability_panel(thin, PLAN)
    assert panel.status == "insufficient" and panel.ece is None and panel.bins == ()
    assert held_out_calibration(mcs.holdout_probabilities, mcs.holdout_labels)["ece"] is None
    others = reliability_panels(PLAN, arms=("sampling",), calibrators_dir=study.calibrators_dir)[1:]
    table = reliability_table([panel, *others])
    row = csv_rows(table.to_csv())[0]
    assert (row["status"], row["ece"], row["bin_count"]) == ("insufficient", "insufficient", "")
    assert row["reason"] == panel.reason and "1 memorized and 3 control held-out draws" in panel.reason
    assert "Insufficient (shown, no bins): sampling macro top_k" in table.caption
    figure = plot_reliability([panel, *others], theme="dark")
    assert "reliability-sampling-macro-top_k" not in gids(figure, kind=None)
    assert "reliability-diagonal-sampling-macro-top_k" in gids(figure)
    assert "ECE insufficient" in figure.axes[0].get_title(loc="left")


def test_a_serving_arm_without_calibrator_files_or_full_distribution_is_listed_missing_and_not_drawn(study):
    found = reliability_panels(PLAN, arms=DEFAULT_ARMS, calibrators_dir=study.calibrators_dir)
    serving = {(panel.kind, panel.source): panel for panel in found if panel.arm == "serving"}
    assert set(serving) == {(kind, source) for kind in KINDS for source in SOURCES}
    for kind in KINDS:
        assert serving[(kind, "full_distribution")].status == "missing"
        assert serving[(kind, "full_distribution")].reason == SERVING_FULL_DISTRIBUTION_REASON
        assert serving[(kind, "top_k")].status == "missing"
        assert serving[(kind, "top_k")].reason == (f"no calibrator file at serving/top_k/{kind}.json under the "
                                                   "calibrators directory: the calibration has not been fitted")
    table = reliability_table(found)
    rows = csv_rows(table.to_csv())
    missing = [row for row in rows if row["arm"] == "serving"]
    assert len(missing) == 4 and {row["status"] for row in missing} == {"missing"}
    assert SERVING_FULL_DISTRIBUTION_REASON in table.caption and "Missing (not drawn): serving macro top_k" in table.caption
    figure = plot_reliability(found, theme="light")
    assert not [gid for gid in gids(figure, kind=None) if gid.startswith("reliability-") and "-serving-" in gid]
    assert len([axes for axes in figure.axes if axes.axison]) == 4
    with pytest.raises(FigureError, match="nothing to draw"):
        plot_reliability([panel for panel in found if panel.arm == "serving"], theme="light")


def test_a_calibrator_fitted_for_another_arm_is_refused(study, tmp_path):
    target = calibrator_path("serving", "top_k", "macro", tmp_path)
    target.parent.mkdir(parents=True)
    shutil.copyfile(calibrator_path("sampling", "top_k", "macro", study.calibrators_dir), target)
    with pytest.raises(ValueError, match="fitted for arm 'sampling'"):
        reliability_panels(PLAN, arms=("serving",), calibrators_dir=tmp_path)


def test_the_reliability_caption_states_data_sample_size_and_interval_method(study):
    found = reliability_panels(PLAN, arms=("sampling",), calibrators_dir=study.calibrators_dir)
    caption = reliability_table(found).caption
    assert caption.startswith("Figure F2.")
    assert "held-out draws each calibrator retained when it was fitted" in caption
    assert "stratified 25 percent split by prompt" in caption
    assert "both draw modes of its adjustment" in caption and "no-adjustment variants carry no estimate" in caption
    for panel in found:
        assert (f"sampling {panel.kind} {panel.source}: n {panel.holdout_draws} held-out draws "
                f"({panel.n_memorized} memorized, {panel.n_control} control) from {panel.holdout_prompts} prompts"
                ) in caption
    assert "10 equal-width bins over [0, 1]" in caption and "at least 2 held-out draws per class" in caption
    assert "95 percent prompt_clustered_stratified_percentile_bootstrap, 1000 resamples, seed 0" in caption
    assert "Moody's" in caption and "SEC EDGAR" in caption and figures.DERIVED_ONLY in caption
    assert "Financial Modeling Prep (FMP)" in caption  # the design cites Moody's and FMP in every caption


def test_bin_count_labels_stay_clear_of_the_diagonal_the_markers_and_the_panel_edge(study):
    found = reliability_panels(PLAN, arms=("sampling",), calibrators_dir=study.calibrators_dir)
    # The fixture has well-calibrated corner bins next to the diagonal, where a label straight above or below
    # its marker lands on the line.
    corners = [b for panel in found for b in panel.bins if abs(b.observed_share - b.mean_predicted) < 0.05]
    assert corners
    for theme in THEMES:
        figure = plot_reliability(found, theme=theme)
        figure.draw_without_rendering()
        scale = figure.dpi / 72.0
        largest = max(b.count for panel in found for b in panel.bins)
        for axes, panel in zip(figure.axes, found, strict=True):
            diagonal = by_gid(figure, f"reliability-diagonal-{panel.arm}-{panel.kind}-{panel.source}")
            line = diagonal.get_path().transformed(diagonal.get_transform())
            centres = axes.transData.transform([(b.mean_predicted, b.observed_share) for b in panel.bins])
            halves = [(math.sqrt(figures._area(b.count, largest)) / 2.0 + 0.75) * scale for b in panel.bins]
            markers = [Bbox.from_extents(x - h, y - h, x + h, y + h) for (x, y), h in zip(centres, halves,
                                                                                          strict=True)]
            labels = [text for text in axes.texts if isinstance(text, Annotation)]
            assert sorted(label.get_text() for label in labels) == sorted(str(b.count) for b in panel.bins)
            frame = axes.bbox
            top_right = {str(b.count) for b in panel.bins if b.mean_predicted > 0.9 and b.observed_share > 0.9}
            for label in labels:
                extent = label.get_window_extent()
                where = (theme, panel.kind, panel.source, label.get_text())
                # Off the line with a visible gap; at the well-calibrated top-right corner (the 12 and 20 bins)
                # the label sits beside its marker with 3 pt of space.
                gap = 3.0 if label.get_text() in top_right else 1.0
                assert not line.intersects_bbox(extent.padded(gap * scale), filled=False), where
                assert not any(extent.overlaps(marker) for marker in markers), where
                assert (frame.x0 <= extent.x0 and extent.x1 <= frame.x1 and frame.y0 <= extent.y0
                        and extent.y1 <= frame.y1), where
                patch = label.get_bbox_patch()  # a surface backing above the diagonal, below the markers
                assert patch is not None and to_hex(patch.get_facecolor()) == to_hex(figure_theme(theme).surface)
                assert diagonal.get_zorder() < label.get_zorder() < 3


# ---------- encoding: identity never by colour alone ----------

def test_variant_identity_never_depends_on_colour_alone(panels, study):
    found = reliability_panels(PLAN, arms=("sampling",), calibrators_dir=study.calibrators_dir)
    for theme in THEMES:
        style = figure_theme(theme)
        tokens = {to_hex(style.ink), to_hex(style.secondary_ink)}
        figure = plot_gap(panels, theme=theme)
        colours = set()
        for panel in panels:
            for point in panel.points:
                if point.missing is not None:
                    continue
                expected = variant_style(point.adjustment, point.mode, theme=theme)
                line = by_gid(figure, f"gap-{panel.arm}-{point.variant}")
                assert line.get_marker() == expected.marker
                assert to_hex(line.get_markerfacecolor()) == to_hex(expected.facecolor)
                assert (to_hex(line.get_markerfacecolor()) == to_hex(style.surface)) == (point.mode == "single")
                colours.add(to_hex(line.get_markeredgecolor()))
            assert to_hex(by_gid(figure, f"gap-zero-{panel.arm}").get_color()) == to_hex(style.muted)
        assert len(colours) <= ALL_PAIRS_SLOT_CAP
        assert colours <= {to_hex(c) for c in (STUDY_PALETTE if theme == "light" else STUDY_PALETTE_DARK)}
        ticks = [label.get_text() for axes in figure.axes for label in axes.get_yticklabels()]
        for panel in panels:  # direct labels name every drawn row
            for point in panel.points:
                if point.missing is None:
                    assert variant_style(point.adjustment, point.mode, theme=theme).label in ticks
        for text in figure.findobj(Text):
            if text.get_text().strip() and text.get_visible():
                assert to_hex(text.get_color()) in tokens, text.get_text()
        reliability = plot_reliability(found, theme=theme)
        for panel in found:
            key = f"{panel.arm}-{panel.kind}-{panel.source}"
            assert to_hex(by_gid(reliability, f"reliability-diagonal-{key}").get_color()) == to_hex(style.muted)
            marks = by_gid(reliability, f"reliability-{key}")
            expected = variant_style(panel.source, "ensemble", theme=theme)
            shape = MarkerStyle(expected.marker)
            assert np.allclose(marks.get_paths()[0].vertices,
                               shape.get_path().transformed(shape.get_transform()).vertices)
            assert to_hex(marks.get_facecolors()[0]) == to_hex(expected.color)
            assert reliability.axes[found.index(panel)].get_title(loc="left").count(f"ECE {panel.ece:.3f}") == 1
        for text in reliability.findobj(Text):
            if text.get_text().strip() and text.get_visible():
                assert to_hex(text.get_color()) in tokens, text.get_text()


# ---------- outputs ----------

def test_each_figure_is_written_in_light_and_dark_svg_with_a_csv_and_markdown_once(study, tmp_path):
    assert FIGURES_DIR == ROOT / "docs" / "assets" / "finance-sim"
    assert FIGURE_NAMES == ("gap", "reliability")
    out = tmp_path / "assets"
    written = render_figures(PLAN, analysis_dir=study.analysis_dir, calibrators_dir=study.calibrators_dir,
                             out_dir=out)
    expected = [path for name in FIGURE_NAMES for path in figure_paths(name, out).values()]
    assert list(written) == expected
    assert sorted(path.name for path in out.iterdir()) == sorted(
        f"{name}{suffix}" for name in FIGURE_NAMES for suffix in ("-light.svg", "-dark.svg", ".csv", ".md"))
    for name in FIGURE_NAMES:
        paths = figure_paths(name, out)
        light, dark = paths["light"].read_text(encoding="utf-8"), paths["dark"].read_text(encoding="utf-8")
        assert light.lstrip().startswith("<?xml") and "<svg" in light and light != dark
        assert figure_theme("light").surface in light and figure_theme("dark").surface in dark
        assert STUDY_PALETTE[1] in light and STUDY_PALETTE_DARK[1] in dark
        assert "<dc:date>" not in light  # no timestamp: a rerender is byte-identical
    before = {path: path.read_bytes() for path in expected}
    assert render_figures(PLAN, analysis_dir=study.analysis_dir, calibrators_dir=study.calibrators_dir,
                          out_dir=out) == written
    assert {path: path.read_bytes() for path in expected} == before  # deterministic, and the same output is a no-op
    with pytest.raises(FigureError, match="refusing to replace"):
        render_figures(PLAN, arms=("sampling",), analysis_dir=study.analysis_dir,
                       calibrators_dir=study.calibrators_dir, out_dir=out)
    assert {path: path.read_bytes() for path in expected} == before


def test_the_markdown_table_is_the_csv_rendered_below_its_caption(panels, study):
    found = reliability_panels(PLAN, arms=DEFAULT_ARMS, calibrators_dir=study.calibrators_dir)
    for table in (gap_table(panels), reliability_table(found)):
        lines = table.to_markdown().splitlines()
        assert lines[0] == table.caption and lines[1] == ""
        rows = list(csv.reader(io.StringIO(table.to_csv())))
        assert lines[3] == "|" + "|".join("---" for _ in rows[0]) + "|"
        rendered = [[value.strip().replace("\\|", "|") for value in line.strip("|").split(" | ")]
                    for line in [lines[2], *lines[4:]]]
        assert rendered == rows
        assert table.to_markdown().endswith("\n") and table.to_csv().endswith("\n")


def test_published_tables_and_svgs_carry_derived_statistics_only(study, tmp_path):
    out = tmp_path / "assets"
    paths = render_figures(PLAN, analysis_dir=study.analysis_dir, calibrators_dir=study.calibrators_dir,
                           out_dir=out)
    texts = {path: path.read_text(encoding="utf-8") for path in paths}
    returns = [value for month in MONTHS if month != MISSING_MONTH
               for value in (realized(month).world_equity, realized(month).tech_sector,
                             realized(month).gold_commodity)]
    prompts = [render(macro_row(month))[:60] for month in MONTHS[:6] for render in (render_identifying,
                                                                                     render_anonymised)]
    # Raw macro levels whose repr is distinctive (the fixture's round values such as -0.2 also occur as statistics).
    raw = {repr(value) for month in MONTHS for value in macro_row(month).raw.values() if len(repr(value)) > 8}
    assert raw
    for path, text in texts.items():
        assert '"inflation"' not in text and "risk_appetite" not in text and "response_text" not in text
        assert "short_treasury_cash" not in text and "world_equity" not in text
        for value in returns:
            assert repr(value) not in text, path
        for prompt in prompts:
            assert prompt not in text, path
        for value in raw:
            assert value not in text, path
        if path.suffix == ".csv":
            cells = {value for row in csv.reader(io.StringIO(text)) for value in row}
            assert not cells & {f"{value:.6g}" for value in returns}, path


def test_the_module_imports_without_matplotlib():
    code = ("import sys\n"
            "import scripts.finance_sim.figures\n"
            "heavy = sorted(m for m in sys.modules if m.split('.')[0] == 'matplotlib')\n"
            "assert not heavy, heavy\n"
            "print('lean-ok')\n")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stderr
    assert "lean-ok" in result.stdout


def test_the_command_renders_from_the_stored_outputs_and_refuses_another_plan(study, tmp_path):
    plan_path = write_plan(PLAN, tmp_path / "plan.json")
    out = tmp_path / "assets"
    common = ["--plan", str(plan_path), "--analysis-dir", str(study.analysis_dir), "--calibrators-dir",
              str(study.calibrators_dir), "--out-dir", str(out)]
    assert main([*common, "--plan-sha256", PLAN.plan_sha256]) == 0
    assert sorted(path.name for path in out.iterdir()) == sorted(
        f"{name}{suffix}" for name in FIGURE_NAMES for suffix in ("-light.svg", "-dark.svg", ".csv", ".md"))
    assert main([*common, "--plan-sha256", PLAN.plan_sha256]) == 0  # the same output again is a no-op
    assert main([*common, "--plan-sha256", FAST.plan_sha256]) == 2
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(["--plan", str(plan_path), "--plan-sha256", PLAN.plan_sha256, "--analysis-dir", str(empty),
                 "--calibrators-dir", str(study.calibrators_dir), "--out-dir", str(tmp_path / "other"),
                 "--figure", "gap"]) == 2
    assert not (tmp_path / "other").exists()
    only = tmp_path / "only"
    assert main([*common[:-1], str(only), "--plan-sha256", PLAN.plan_sha256, "--figure", "reliability",
                 "--arm", "sampling"]) == 0
    assert sorted(path.name for path in only.iterdir()) == ["reliability-dark.svg", "reliability-light.svg",
                                                            "reliability.csv", "reliability.md"]
