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

import contextlib
import copy
import csv
import hashlib
import io
import json
import math
import re
import shutil
import socket
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.text import Annotation, Text  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

from recall_guard.core.consensus import wilson_interval  # noqa: E402
from recall_guard.dataset.macro_regime import render_anonymised, render_identifying  # noqa: E402
from recall_guard.harness.local_reporting import CALIBRATION_MIN_PER_CLASS, held_out_calibration  # noqa: E402
from recall_guard.harness.plots import configure_paper_style, reliability_curve  # noqa: E402
from scripts.finance_sim import analysis as analysis_module  # noqa: E402
from scripts.finance_sim import figures  # noqa: E402
from scripts.finance_sim.analysis import (  # noqa: E402
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
from scripts.finance_sim.figure_style import series_style  # noqa: E402
from scripts.finance_sim.figures import (  # noqa: E402
    CAPTION_LIMIT,
    DEFAULT_ARMS,
    FIGURE_NAMES,
    FIGURES_DIR,
    GAP_COLUMNS,
    FigureError,
    bin_interval,
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
    route_label,
    variant_label,
)
from scripts.finance_sim.limits import BudgetLedger, LedgerCallBudget, StudyLimits  # noqa: E402
from scripts.finance_sim.plan import SERVING_FULL_DISTRIBUTION_REASON, VARIANTS, write_plan  # noqa: E402
from scripts.finance_sim.published_text import published_reason  # noqa: E402
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
#: Global rcParams a render would pick up if it read them, beyond the paper style of tests/harness/test_plots.py
#: (which sets ``savefig.format`` to pdf, the renderer that lays a figure out, and ``savefig.bbox`` to tight, and
#: never restores them): the figure dpi, the font, the text metrics, the layout padding and the marker defaults.
FOREIGN_RC = {"figure.dpi": 150.0, "font.family": "serif", "text.hinting": "none", "svg.fonttype": "none",
              "svg.hashsalt": "another-salt", "path.simplify": False, "figure.constrained_layout.h_pad": 0.3,
              "figure.constrained_layout.w_pad": 0.3, "lines.markersize": 12.0, "axes.linewidth": 2.0}
COMPUTED_SERVING = ("single_none", "single_top_k", "ensemble_none", "ensemble_top_k")
SERVING_MISSING = ("single_full_distribution", "ensemble_full_distribution")
SOURCES = ("top_k", "full_distribution")
KINDS = ("macro", "edgar")
#: Words of the study's bookkeeping that a published label, caption or note does not use: the arm vocabulary
#: ("serving arm", "sampling arm", "arm", "primary", "probe", "rendering"), "top-k" and "full distribution" as the
#: names of the evidence modes, and a bare "serving" or "sampling" as the name of a group. "serving route" is the
#: catalogue's own term (a model on a serving route) and stays allowed.
BOOKKEEPING_WORDS = (r"\barms?\b", r"\bprimary\b", r"\bprobe\b", r"\brenderings?\b", r"\btop-k\b",
                     r"\bfull distribution\b", r"\bfull next-token distribution\b", r"\bsampling\b",
                     r"\bserving\b(?! routes?\b)")
#: A number standing on its own in a caption: not part of a word, with its sign and an exponent where it has one.
CAPTION_NUMBER = re.compile(r"(?<![\w.])[+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?(?!\w)")
#: Infrastructure wording a published text does not carry (Requirement 7.11).
INFRASTRUCTURE_WORDS = (r"\boperating system\b", r"\bWindows\b", r"\bLinux\b", r"\bmemory\b", r"\bexit codes?\b",
                        r"\bports?\b", r"\bantivirus\b")
ROUTES = {arm: route_label(PLAN, arm) for arm in ("serving", "sampling")}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("the figures attempted a network connection")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)


@contextlib.contextmanager
def foreign_rc_params():
    """The process's global rcParams as an earlier test may leave them; everything is restored on exit."""
    with matplotlib.rc_context():
        configure_paper_style()
        matplotlib.rcParams.update(FOREIGN_RC)
        yield


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
    return tuple(read_gap_panel(analysis_path(arm, "full_run", study.analysis_dir), plan_sha256=PLAN.plan_sha256,
                                route=ROUTES[arm]) for arm in ("serving", "sampling"))


def csv_rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text)))


def published_rows(table) -> dict[tuple[str, str], dict[str, str]]:
    """The rows of a published gap table by (arm, variant): the table itself carries the route and the plain label."""
    arm_of = {route: arm for arm, route in ROUTES.items()}
    variant_of = {variant_label(variant): variant for variant in VARIANTS}
    return {(arm_of[row["route"]], variant_of[row["variant"]]): row for row in csv_rows(table.to_csv())}


def gids(figure, kind=Line2D) -> set[str]:
    return {artist.get_gid() for artist in figure.findobj(kind) if artist.get_gid()}


def by_gid(figure, gid: str):
    return next(artist for artist in figure.findobj() if artist.get_gid() == gid)


def is_grey(colour: object) -> bool:
    red, green, blue = to_rgb(colour)
    return red == green == blue


def assert_plain_vocabulary(text: str) -> None:
    """No bookkeeping word of the study and no infrastructure wording in a label, a legend entry or a caption."""
    for pattern in (*BOOKKEEPING_WORDS, *INFRASTRUCTURE_WORDS):
        assert not re.search(pattern, text), (pattern, text)
    assert "—" not in text


def assert_journal_caption(text: str, number: int) -> None:
    """A caption a journal prints: "Figure N.", one line, the sample size, and nothing of how the work was run.

    No task number, no backtick, no identifier with an underscore, no dotted
    module path, no em dash, none of the study's bookkeeping words, and at
    most ``CAPTION_LIMIT`` (600) characters.
    """
    assert text.startswith(f"Figure {number}. ") and " n = " in text, text
    assert CAPTION_LIMIT == 600 and len(text) <= CAPTION_LIMIT, (len(text), text)
    assert "\n" not in text and "  " not in text and text.endswith(".")
    assert not re.search(r"\btasks? \d", text, flags=re.IGNORECASE), text
    assert "`" not in text
    assert not re.search(r"\w_\w", text), re.findall(r"\S*\w_\w\S*", text)
    assert not re.search(r"\b[A-Za-z_]\w*\.[A-Za-z_]\w*", text), re.findall(r"\b[A-Za-z_]\w*\.[A-Za-z_]\w*", text)
    assert "Sources:" not in text and figures.DERIVED_ONLY not in text
    assert_plain_vocabulary(text)


def number_forms(value: float | int) -> set[str]:
    """Every way a caption writes ``value``: as the table holds it, in short form, and to three decimals."""
    forms = {figures._number(value), f"{value:g}", figures.rounded(value), figures.rounded(value, signed=True)}
    if float(value).is_integer():
        forms |= {str(int(value)), f"{int(value):+d}"}
    return forms


def assert_caption_numbers(table, stored: tuple = ()) -> list[str]:
    """Every number of the caption is a value of the table below it, or one of ``stored``.

    The caption's numbers are parsed from its text (after "Figure N." and with
    the names that are table cells, a route or an item, taken out) and each is
    looked up among the table's cells, as the cell holds it or rounded as a
    caption rounds; ``stored`` are the values of the stored result the caller
    names that are not a cell (a confidence level, a planned count). Returns
    the numbers found.
    """
    caption = table.caption.split(". ", 1)[1]
    cells = {cell for row in table.rows for cell in row if cell}
    known: set[str] = set()
    names: set[str] = set()
    for cell in cells:
        try:
            known |= number_forms(int(cell) if re.fullmatch(r"[+-]?\d+", cell) else float(cell))
        except ValueError:
            if re.search(r"\d", cell):
                names |= {cell, cell.split(" at temperature ")[0]}
    for value in stored:
        known |= number_forms(value)
    for name in sorted(names, key=len, reverse=True):
        caption = caption.replace(name, "")
    found = CAPTION_NUMBER.findall(caption)
    assert found, caption
    for number in found:
        assert number in known, (number, table.caption)
    return found


def assert_notes(table) -> None:
    """The lines below a table: plain wording like the caption, the sources once, the derived-only line last.

    A fact is stated once per file: no sentence of the caption is repeated in a note.
    """
    lines = table.to_markdown().splitlines()
    assert lines[0] == table.caption and lines[-1] == figures.DERIVED_ONLY
    assert lines[-2].startswith("Sources: ") and sum(line.startswith("Sources: ") for line in lines) == 1
    sentences = [sentence for sentence in re.split(r"(?<=[.!?]) ", table.caption) if len(sentence) > 20]
    for note in table.notes:
        assert not [sentence for sentence in sentences if sentence in note], note
        assert note in lines and "`" not in note and "—" not in note
        assert not re.search(r"\btasks? \d", note, flags=re.IGNORECASE) and not re.search(r"\w_\w", note), note
        assert_plain_vocabulary(note)


def assert_academic_figure(figure, *, panels: int) -> None:
    """The figure as a journal prints it (Requirements 8.1, 8.2, 8.4, 8.6 of the qualified-model-catalogue spec).

    Both axes of every drawn panel carry a label, there is no figure title and
    no panel title, no text sits inside a plot area (a multi-panel figure's
    "(a)", "(b)" labels sit above their panel), every text is black, a
    legend names the series, and no label uses the study's bookkeeping words.
    """
    assert figure.get_suptitle() == ""
    # Measured as published: saving freezes the layout the plot function arrived at (no further layout pass here).
    assert_academic_svg(figure_svg(figure))
    with matplotlib.rc_context(figures.figure_rc()):  # measured under the figure's own rcParams
        figure.draw_without_rendering()
    drawn = [axes for axes in figure.axes if axes.axison]
    assert len(drawn) == panels
    for index, axes in enumerate(drawn):
        assert all(axes.get_title(loc=loc) == "" for loc in ("left", "center", "right"))
        assert (axes.get_xlabel() or figure.get_supxlabel()).strip(), f"panel {index} has no x label"
        assert (axes.get_ylabel() or figure.get_supylabel()).strip(), f"panel {index} has no y label"
        frame = axes.bbox
        for text in axes.texts:
            extent = text.get_window_extent()
            assert extent.y0 >= frame.y1 or extent.y1 <= frame.y0 or extent.x1 <= frame.x0 or extent.x0 >= frame.x1, (
                f"{text.get_text()!r} is written inside the plot area of panel {index}")
        letters = [text.get_text() for text in axes.texts]
        assert letters == ([f"({chr(ord('a') + index)})"] if panels > 1 else []), letters
        assert axes.get_legend() is None, "one legend per figure, outside the plot areas"
    assert len(figure.legends) == 1 and not figure.legends[0].get_frame_on()
    # Nothing is cut off at the edge of the page and no two panels, with their labels, overlap each other or the legend.
    page = figure.bbox.padded(1.0)
    boxes = []
    for axes in drawn:
        parts = [axes.get_tightbbox(), axes.xaxis.label.get_window_extent(), axes.yaxis.label.get_window_extent()]
        for axis in (axes.xaxis, axes.yaxis):
            low, high = sorted(axis.get_view_interval())
            ticks = [label.get_window_extent()
                     for location, label in zip(axis.get_majorticklocs(), axis.get_majorticklabels(), strict=True)
                     if low <= location <= high and label.get_text().strip()]
            parts += ticks
            for index, tick in enumerate(ticks):  # neighbouring tick labels do not run into each other
                assert not any(tick.overlaps(other) for other in ticks[index + 1:]), axes.get_ylabel()
        boxes.append(Bbox.union(parts))
    boxes += [legend.get_window_extent() for legend in figure.legends if legend.get_texts()]
    boxes += [label.get_window_extent() for label in (figure._supxlabel, figure._supylabel) if label is not None]
    for index, box in enumerate(boxes):
        assert page.x0 <= box.x0 and box.x1 <= page.x1 and page.y0 <= box.y0 and box.y1 <= page.y1, (index, box)
        for other in boxes[index + 1:]:
            assert not box.overlaps(other), (index, box, other)
    for text in figure.findobj(Text):
        if text.get_text().strip() and text.get_visible():
            assert to_rgb(text.get_color()) == (0.0, 0.0, 0.0), text.get_text()
    for text in figure.findobj(Text):
        if text.get_visible():
            assert_plain_vocabulary(text.get_text())


def assert_academic_svg(svg: str) -> None:
    """Grey levels only on a white page, glyphs as paths, no timestamp."""
    colours = set(re.findall(r"#[0-9a-fA-F]{6}\b", svg))
    assert "#ffffff" in colours and all(is_grey(colour) for colour in colours), sorted(colours)
    assert svg.lstrip().startswith("<?xml") and "<svg" in svg
    assert "<text" not in svg and "<dc:date>" not in svg


def legend_labels(figure) -> list[str]:
    return [text.get_text() for legend in figure.legends for text in legend.get_texts()]


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
    serving = read_gap_panel(analysis_path("serving", "full_run", study.analysis_dir), route=ROUTES["serving"])
    expected = len(cell(metric(stored(study, "serving"), "did_accuracy_gap"), "single_top_k")["missing_months"])
    assert serving.point("single_top_k").missing_months == expected == 2
    row = next(row for row in csv_rows(gap_table([serving]).to_csv())
               if row["variant"] == variant_label("single_top_k"))
    assert row["missing_months"] == "2" and row["missing"] == "" and row["route"] == ROUTES["serving"]
    # A changed stored value is what the table shows: the figure reads the file, it does not recompute it.
    changed = copy.deepcopy(stored(study, "serving"))
    cell(metric(changed, "did_accuracy_gap"), "single_top_k")["statistic"] = 0.123456
    row = next(row for row in csv_rows(gap_table([gap_panel(changed, route=ROUTES["serving"])]).to_csv())
               if row["variant"] == variant_label("single_top_k"))
    assert row["gap"] == "0.123456"


def test_the_first_routes_missing_variants_are_listed_in_the_table_and_caption_and_not_drawn(panels):
    table = gap_table(panels)
    rows = published_rows(table)
    assert len(rows) == 12
    for variant in SERVING_MISSING:
        row = rows[("serving", variant)]
        # The table gives the reason in the published vocabulary; the stored reason stays in the private analysis.
        assert row["missing"] == published_reason(SERVING_FULL_DISTRIBUTION_REASON) == (
            "logits are not available on this serving route: they would come from other weights than the served "
            "model, and values from different weights are not comparable")
        assert row["gap"] == row["interval_low"] == row["interval_high"] == row["p_raw"] == ""
    for variant in COMPUTED_SERVING:
        assert rows[("serving", variant)]["missing"] == "" and rows[("serving", variant)]["gap"] != ""
    # The caption names what is not drawn in plain words; the stored reason stays in the table.
    assert "Not drawn on modal-gemma3-4b: both variants adjusted with logits." in table.caption
    assert SERVING_FULL_DISTRIBUTION_REASON not in table.caption
    figure = plot_gap(panels)
    drawn = {f"gap-serving-{variant}" for variant in COMPUTED_SERVING} | {f"gap-sampling-{variant}"
                                                                         for variant in VARIANTS}
    assert {gid for gid in gids(figure) if gid.startswith("gap-") and "interval" not in gid
            and "zero" not in gid} == drawn
    # Interval bars are kept: every drawn variant with a stored interval carries its bar.
    for panel in panels:
        for point in panel.drawn_points:
            assert (f"gap-interval-{panel.arm}-{point.variant}" in gids(figure, kind=None)) == (
                point.low is not None and point.high is not None)
    assert any(gid.startswith("gap-interval-") for gid in gids(figure, kind=None))
    svg = figure_svg(figure)
    assert 'id="gap-serving-single_top_k"' in svg and 'id="gap-sampling-ensemble_full_distribution"' in svg
    for variant in SERVING_MISSING:
        assert f'id="gap-serving-{variant}"' not in svg and f'id="gap-interval-serving-{variant}"' not in svg


def test_a_panel_is_labelled_by_its_serving_route_as_data_from_the_plan(panels, study):
    """The x axis names the serving route (the plan's profile and temperature), not the study's arm or its role."""
    for arm in ("serving", "sampling"):
        arm_plan = PLAN.arms.arms[arm]
        assert ROUTES[arm] == f"{arm_plan.profile_id} at temperature {arm_plan.temperature:g}"
    assert ROUTES == {"serving": "modal-gemma3-4b at temperature 0", "sampling": "llamacpp-gemma3-4b at temperature 0.7"}
    table = gap_table(panels)
    rows = csv_rows(table.to_csv())
    # A published table names the route and plain labels: no arm, no role and no internal variant key.
    assert "arm" not in GAP_COLUMNS and "role" not in GAP_COLUMNS and GAP_COLUMNS[:2] == ("route", "variant")
    assert {row["route"] for row in rows} == set(ROUTES.values())
    assert {row["variant"] for row in rows} == {variant_label(variant) for variant in VARIANTS}
    assert variant_label("single_none") == "single draw, unadjusted"
    assert variant_label("ensemble_full_distribution") == "ensemble, adjusted with logits"
    cells = {cell for row in rows for cell in row.values()}
    assert not {"serving", "sampling", "primary", "probe", *VARIANTS} & cells
    figure = plot_gap(panels)
    (axes,) = figure.axes
    assert axes.get_xlabel() == "serving route"
    assert [label.get_text() for label in axes.get_xticklabels()] == [
        "modal-gemma3-4b\nat temperature 0", "llamacpp-gemma3-4b\nat temperature 0.7"]
    # The caption names a route by its profile; the temperature is on the axis and in the table beside it.
    for arm in ROUTES:
        assert PLAN.arms.arms[arm].profile_id in table.caption
    # A panel without its route cannot be labelled, so the table and the figure refuse it.
    bare = read_gap_panel(analysis_path("serving", "full_run", study.analysis_dir))
    assert bare.route is None
    for build in (gap_table, plot_gap):
        with pytest.raises(FigureError, match="serving route"):
            build([bare])


def test_dropped_resamples_make_the_interval_conditional_and_the_caption_says_so(panels):
    table = gap_table(panels)
    dropped = max(point.dropped_resamples for point in panels[1].points if point.dropped_resamples is not None)
    assert dropped > 0
    # Methods matter, so it is a note of the table and not caption text; it is stated once.
    sentence = "The intervals are conditional on the resamples with months on both sides of the cutoff."
    assert "A dropped resample held no month on one side of the cutoff." in " ".join(table.notes)
    assert sentence in " ".join(table.notes) and "conditional" not in table.caption
    assert table.to_markdown().count("conditional on the resamples") == 1
    rows = csv_rows(table.to_csv())
    assert max(int(row["dropped_resamples"]) for row in rows if row["dropped_resamples"]) == dropped
    clean = tuple(replace(panel, points=tuple(replace(point, dropped_resamples=0 if point.drawn else None)
                                              for point in panel.points)) for panel in panels)
    assert "conditional" not in gap_table(clean).to_markdown()


def test_achieved_power_is_a_note_below_the_table_with_its_monte_carlo_error(study, panels):
    table = gap_table(panels)
    notes = " ".join(table.notes)
    for panel in panels:
        power = metric(stored(study, panel.arm), "achieved_power")["power"]
        assert panel.power == power["power"] and panel.power_n_sim == power["n_sim"]
        assert power["approximation"] == POWER_APPROXIMATION
        if power["power"] is None:
            assert f"not available on {panel.route}" in notes
            continue
        error = math.sqrt(power["power"] * (1.0 - power["power"]) / power["n_sim"])
        assert panel.power_monte_carlo_error == pytest.approx(error)
        assert (f"{power['power']:.3f} on {panel.route} (Monte Carlo standard error {error:.3f}, {power['n_sim']} "
                "simulations)") in notes
    assert "t-test approximation under independent months" in notes
    assert "power" not in table.caption  # a note of the table, not part of the figure's one message
    assert_notes(table)


def test_the_gap_caption_is_a_journal_caption_with_sample_size_and_interval(panels):
    table = gap_table(panels)
    caption = table.caption
    analysis = PLAN.analysis
    assert_journal_caption(caption, 1)
    # The series cannot differ by construction, and the caption says so (Requirement 8.7); the reason is a note.
    assert "The adjusted variants of a draw mode cannot differ from its unadjusted one by construction." in caption
    assert "An adjustment scales the size of a call and never changes its direction." in table.notes
    assert "direction" not in caption
    assert "identifying minus anonymised prompt" in caption
    # The key number: the unadjusted ensemble's gap with its interval, per route, as the table holds it. A route
    # is named by its profile, which the x axis shows with its temperature.
    profiles = {panel.route: PLAN.arms.arms[panel.arm].profile_id for panel in panels}
    assert figures.short_routes(panel.route for panel in panels) == profiles
    assert figures.short_routes(["a at temperature 0", "a at temperature 1"]) == {
        "a at temperature 0": "a at temperature 0", "a at temperature 1": "a at temperature 1"}
    for panel in panels:
        point = panel.point("ensemble_none")
        signed = [figures.rounded(value, signed=True) for value in (point.statistic, point.low, point.high)]
        assert f"{signed[0]} ({signed[1]} to {signed[2]}) on {profiles[panel.route]}" in caption
    assert "), unadjusted ensemble: " in caption
    # The sample size is the months the stated gap rests on, the first route in full and the next ones in the same
    # order of numbers; the planned number is named only where it differs.
    first, second = (panel.point("ensemble_none") for panel in panels)
    # The first route's fixture has a failed ensemble and a month without returns: 88 of its 90 planned months.
    assert (first.months, first.pre_cutoff_months, first.post_cutoff_months, panels[0].planned_months) == (
        88, 67, 21, 90)
    assert (panels[0].pre_cutoff_months, panels[0].post_cutoff_months) == (68, 22)  # the plan's, not the gap's
    assert second.months == panels[1].planned_months
    sizes = (f"88 of 90 planned months (67 on or before the cutoff, 21 after) on {profiles[panels[0].route]}; "
             f"{second.months} ({second.pre_cutoff_months}, {second.post_cutoff_months}) on "
             f"{profiles[panels[1].route]}")
    assert f" n = {sizes}. " in caption and "68 on or before" not in caption
    # The unit the bootstrap resamples is stated once, in the note with its block length.
    interval = f"Bars and parentheses: {(1 - analysis.alpha) * 100:g} percent moving-block bootstrap intervals."
    assert interval in caption and "over months" not in caption and len(caption) < 570
    assert caption.index(" n = ") < caption.index(interval) < caption.index("by construction")
    # What the two evidence modes are is a note, in the words of the one label function; the gap analysis stores
    # no number of alternatives, so the note names none.
    assert "Logprobs are" not in caption and "next-token" not in caption
    assert ("The adjustments use the recall estimate from logprobs (served alternatives per token) or from logits "
            "(kept alternatives of the next-token vector).") in table.notes
    # Every number of the caption is in the table; the confidence level and the planned months are the stored
    # analysis's own (its settings and its planned_months).
    assert_caption_numbers(table, stored=(round(100 * (1 - analysis.alpha), 6), panels[0].planned_months))
    assert (f"The bootstrap resamples blocks of {analysis.block_length} months, {analysis.n_resamples} times."
            in table.notes)
    # The sources and the derived-only line follow the table, once; the model and the plan are not caption matter.
    assert "Moody's" not in caption and "google/gemma-3-4b-it" not in caption and PLAN.plan_sha256[:12] not in caption
    markdown = table.to_markdown()
    assert markdown.count("Moody's") == 2 and markdown.count("BAA10Y") == 1  # one citation, naming Moody's twice
    assert markdown.count("Financial Modeling Prep (FMP)") == 1 and markdown.count(figures.DERIVED_ONLY) == 1
    assert "Holm" in " ".join(table.notes)
    assert_notes(table)


def test_the_gap_caption_states_the_months_the_gap_rests_on_and_the_planned_months_only_where_they_differ(panels):
    """The sample size is that of the drawn value: a route with missing months is not credited with its plan."""
    serving, sampling = panels

    def short(panel, months: int, pre: int, post: int):
        points = tuple(replace(point, months=months, pre_cutoff_months=pre, post_cutoff_months=post,
                               missing_months=panel.planned_months - months) if point.drawn else point
                       for point in panel.points)
        return replace(panel, points=points)

    # Planned exceeds drawn on both routes: each says "of N planned".
    fewer = (short(serving, 80, 61, 19), short(sampling, 17, 13, 4))
    table = gap_table(fewer)
    assert (" n = 80 of 90 planned months (61 on or before the cutoff, 19 after) on modal-gemma3-4b; 17 of 20 planned "
            "(13, 4) on llamacpp-gemma3-4b. ") in table.caption
    assert_journal_caption(table.caption, 1)
    assert_caption_numbers(table, stored=(95, 90, 20))
    rows = published_rows(table)
    assert (rows[("serving", "ensemble_none")]["months"], rows[("serving", "ensemble_none")]["missing_months"]) == (
        "80", "10")
    # Planned equals drawn everywhere: the planned number is not named.
    full = (short(serving, 90, 68, 22), short(sampling, 20, 15, 5))
    caption = gap_table(full).caption
    assert (" n = 90 months (68 on or before the cutoff, 22 after) on modal-gemma3-4b; 20 (15, 5) on "
            "llamacpp-gemma3-4b. ") in caption and "planned" not in caption
    # One route alone needs no route name in the sample size.
    assert " n = 80 of 90 planned months (61 on or before the cutoff, 19 after). " in gap_table(fewer[:1]).caption


def test_the_published_vocabulary_allows_the_serving_route_and_refuses_the_words_of_the_study(panels):
    """"serving route" is the catalogue's term; the arm vocabulary and a bare group name are not published."""
    for text in ("serving route", "by serving route and temperature", "two serving routes", "logprobs (top 5)",
                 "logits (top 20 of the next-token vector)", "the served alternatives"):
        assert_plain_vocabulary(text)
    for text in ("serving arm", "sampling arm", "the arm", "primary", "the probe route", "rendering", "top-k",
                 "the full distribution", "logits (full next-token distribution)", "serving", "the serving model",
                 "sampling", "sampling route"):
        with pytest.raises(AssertionError):
            assert_plain_vocabulary(text)
    assert plot_gap(panels).axes[0].get_xlabel() == "serving route"


def test_one_function_labels_both_evidence_modes_from_the_stored_counts(study):
    """A label says what the estimate uses: the served top k, or the kept top alternatives of the next-token vector,
    never the whole vector."""
    assert figures.evidence_label("top_k", 5) == "logprobs (top 5)"
    assert figures.evidence_label("full_distribution", 20) == "logits (top 20 of the next-token vector)"
    assert figures.evidence_label("top_k") == "logprobs (served alternatives per token)"
    assert figures.evidence_label("full_distribution") == "logits (kept alternatives of the next-token vector)"
    found = reliability_panels(PLAN, arms=("sampling",), calibrators_dir=study.calibrators_dir)
    served = PLAN.arms.arms["sampling"].top_logprobs
    for panel in found:
        record = load_calibrator(calibrator_path("sampling", panel.source, panel.kind, study.calibrators_dir), PLAN,
                                 arm="sampling", kind=panel.kind, source=panel.source)
        settings = record.evidence_settings
        assert panel.kept == (None if settings is None else settings.top_k_keep)
        assert (panel.kept is None) == (panel.source == "top_k")
    kept = {panel.kept for panel in found if panel.source == "full_distribution"}
    assert len(kept) == 1
    assert legend_labels(plot_reliability(found)) == [
        f"logprobs (top {served})", f"logits (top {kept.pop()} of the next-token vector)", "perfect calibration"]
    # A calibrator that states no kept count is labelled without a number, never as the whole distribution.
    bare = [replace(panel, kept=None) for panel in found]
    assert legend_labels(plot_reliability(bare))[1] == "logits (kept alternatives of the next-token vector)"
    for table in (reliability_table(found), reliability_table(bare)):
        assert "full next-token" not in table.to_markdown() and "served top-" not in table.to_markdown()


def test_the_interval_wording_follows_the_stored_method_and_an_unknown_method_is_refused(study, panels):
    """A caption cannot become untrue silently: its interval words are looked up by the method the analysis stores."""
    data = stored(study, "sampling")
    methods = {cell(metric(data, "did_accuracy_gap"), variant)["interval"]["method"] for variant in VARIANTS}
    assert len(methods) == 1 and methods <= set(figures.INTERVAL_WORDING)
    assert figures.INTERVAL_WORDING[methods.pop()] == "moving-block bootstrap over months"
    assert {point.interval_method for point in panels[1].drawn_points} <= set(figures.INTERVAL_WORDING)
    changed = copy.deepcopy(data)
    for variant in VARIANTS:
        cell(metric(changed, "did_accuracy_gap"), variant)["interval"]["method"] = "another resampling scheme"
    with pytest.raises(FigureError, match="no wording for the interval method 'another resampling scheme'"):
        gap_table([gap_panel(changed, route=ROUTES["sampling"])])


def test_gap_panels_refuse_another_schema_stage_plan_or_a_mix_of_plans(study):
    data = stored(study, "serving")
    for key, value, message in (("schema", "other/1", "schema"), ("stage", "pilot", "stage"),
                                ("arm", "elsewhere", "arm")):
        with pytest.raises(FigureError, match=message):
            gap_panel({**data, key: value})
    with pytest.raises(FigureError, match="plan"):
        gap_panel(data, plan_sha256=FAST.plan_sha256)
    route = ROUTES["serving"]
    other = gap_panel({**data, "plan_sha256": FAST.plan_sha256}, route=route)
    with pytest.raises(FigureError, match="one plan"):
        gap_table([gap_panel(data, route=route), other])
    with pytest.raises(FigureError, match="twice"):
        gap_table([gap_panel(data, route=route), gap_panel(data, route=route)])
    path = study.root / "duplicate.json"
    path.write_text('{"schema": 1, "schema": 2}', encoding="utf-8")
    with pytest.raises(FigureError, match="duplicate"):
        read_gap_panel(path)


# ---------- F2: reliability of the recall estimate ----------

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
        assert (panel.auc_method, panel.auc_confidence, panel.auc_resamples) == (
            interval["method"], interval["confidence"], interval["n_resamples"])
        assert panel.is_weak == record.calibrator.is_weak
        assert (panel.route, panel.served_k) == (ROUTES["sampling"], PLAN.arms.arms["sampling"].top_logprobs)
    table = reliability_table(found)
    rows = csv_rows(table.to_csv())
    assert len(rows) == sum(len(panel.bins) for panel in found)
    first = found[0]
    head = next(row for row in rows if (row["kind"], row["source"]) == (first.kind, first.source))
    assert head["ece"] == f"{first.ece:.6g}" and head["bin_count"] == str(first.bins[0].count)
    assert head["variants"] == "single draw, adjusted with logprobs; ensemble, adjusted with logprobs"
    assert head["route"] == ROUTES["sampling"] and "arm" not in head and "role" not in head


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


def test_every_bin_carries_a_wilson_interval_on_its_observed_share_with_the_bin_count_as_n(study):
    """Requirement 8.5: an observed share is an estimate, so each bin shows its 95 percent Wilson score interval."""
    found = reliability_panels(PLAN, arms=("sampling",), calibrators_dir=study.calibrators_dir)
    figure = plot_reliability(found)
    table = reliability_table(found)
    rows = csv_rows(table.to_csv())
    position = 0
    for panel in found:
        bar = by_gid(figure, f"reliability-interval-{panel.key}")
        lows, highs = [], []
        for bin_ in panel.bins:
            memorized = round(bin_.observed_share * bin_.count)
            assert memorized / bin_.count == pytest.approx(bin_.observed_share)
            raw = wilson_interval(memorized, bin_.count, confidence=0.95)
            # A bound that floating point leaves a hair off 0 or 1 is written as exactly 0 or 1.
            expected = tuple(0.0 if abs(bound) <= 1e-12 else 1.0 if abs(bound - 1.0) <= 1e-12 else bound
                             for bound in raw)
            assert bin_interval(bin_) == expected and expected == pytest.approx(raw, abs=1e-12)
            assert 0.0 <= expected[0] <= bin_.observed_share <= expected[1] <= 1.0
            row = rows[position]
            position += 1
            assert (row["share_interval_low"], row["share_interval_high"]) == (
                figures._number(expected[0]), figures._number(expected[1]))
            lows.append(expected[0])
            highs.append(expected[1])
        # One capped vertical bar per bin, in the bin's place: no bar joins two bins.
        ys = [value for value in bar.get_ydata() if not math.isnan(value)]
        assert ys == [value for pair in zip(lows, highs, strict=True) for value in pair]
        assert sum(math.isnan(value) for value in bar.get_ydata()) == len(panel.bins)
        marker = by_gid(figure, f"reliability-{panel.key}")
        xs = [value for value in bar.get_xdata() if not math.isnan(value)]
        assert xs == [value for value in marker.get_xdata() for _ in range(2)]
    assert figures.RELIABILITY_CONFIDENCE == 0.95
    assert ("Bars are 95 percent Wilson score intervals on a bin's observed share (n is the bin's draw count), "
            "treating the draws of one prompt as independent.") in table.caption
    assert "Each row of a drawn calibrator is one bin with its draw count." in table.notes
    # An all-memorized or all-control bin has a bound at exactly 1 or 0 in the table, not a rounding residue.
    extremes = [bin_ for panel in found for bin_ in panel.bins if bin_.observed_share in (0.0, 1.0)]
    assert extremes
    for bin_ in extremes:
        low, high = bin_interval(bin_)
        assert (low if bin_.observed_share == 0.0 else high) == bin_.observed_share
    for row in rows:
        for column in ("share_interval_low", "share_interval_high"):
            assert "e-" not in row[column] and (row[column] == "" or 0.0 <= float(row[column]) <= 1.0), row
    from recall_guard.harness.plots import ReliabilityBin

    empty = ReliabilityBin(low=0.0, high=0.1, count=7, mean_predicted=0.05, observed_share=0.0)
    assert bin_interval(empty)[0] == 0.0 and 0.0 < bin_interval(empty)[1] < 1.0
    assert bin_interval(replace(empty, observed_share=1.0))[1] == 1.0


def test_bins_are_unconnected_markers_set_apart_by_evidence_mode_one_panel_per_calibration_set(study):
    found = reliability_panels(PLAN, arms=("sampling",), calibrators_dir=study.calibrators_dir)
    figure = plot_reliability(found)
    assert_academic_figure(figure, panels=2)
    macro_axes, edgar_axes = figure.axes
    offsets = {}
    for panel in found:
        line = by_gid(figure, f"reliability-{panel.key}")
        assert line.axes is (macro_axes if panel.kind == "macro" else edgar_axes)
        assert line.get_linestyle() == "None", "no line joins the bins, so none bridges an empty bin"
        assert list(line.get_ydata()) == [bin_.observed_share for bin_ in panel.bins]
        shifts = {round(x - bin_.mean_predicted, 12) for x, bin_ in zip(line.get_xdata(), panel.bins, strict=True)}
        assert len(shifts) == 1
        offsets[(panel.kind, panel.source)] = shifts.pop()
    # A small horizontal offset per evidence mode, the same in both panels, keeps coinciding markers apart.
    for kind in KINDS:
        assert offsets[(kind, "top_k")] < 0 < offsets[(kind, "full_distribution")]
        assert offsets[(kind, "full_distribution")] - offsets[(kind, "top_k")] == pytest.approx(0.02)
    assert offsets[("macro", "top_k")] == offsets[("edgar", "top_k")]
    # The table says so, with the size of the offset the figure uses, so a reader does not take it for data.
    size = abs(offsets[("macro", "top_k")])
    assert size == pytest.approx(0.01)
    note = (f"In the figure the markers of the two evidence modes are offset sideways by {size:g} on the probability "
            "axis, logprobs to the left and logits to the right, so that both stay visible; the table gives the "
            "values without the offset.")
    assert note in reliability_table(found).notes and note in reliability_table(found).to_markdown()
    for kind, axes in zip(KINDS, figure.axes, strict=True):
        diagonal = by_gid(figure, f"reliability-diagonal-{kind}")
        assert diagonal.axes is axes and diagonal.get_marker() == "None" and is_grey(diagonal.get_color())
    caption = reliability_table(found).caption
    assert "(a) macro prompts and (b) EDGAR releases" in caption


def test_an_insufficient_calibrator_is_not_drawn_and_the_caption_says_why(study):
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
    assert "1 memorized and 3 control held-out draws" in panel.reason
    assert row["reason"] == published_reason(panel.reason) == (
        "too few held-out draws: 1 memorized and 3 control, and at least 2 are needed in each class")
    # Not drawn, and the caption says why in a few words: no note is written inside the plot.
    assert "Not drawn, with fewer than 2 held-out draws in a class: logprobs in (a)." in table.caption
    # The sample size counts the drawn calibrators only: (a) now holds one evidence mode.
    assert " n = (a) 24 held-out draws of 12 prompts and (b) 70 of 35, per evidence mode. " in table.caption
    assert_journal_caption(table.caption, 2)
    figure = plot_reliability([panel, *others])
    assert "reliability-sampling-macro-top_k" not in gids(figure, kind=None)
    assert "reliability-interval-sampling-macro-top_k" not in gids(figure, kind=None)
    assert {gid for gid in gids(figure) if "diagonal" not in gid and "interval" not in gid} == {
        f"reliability-{other.key}" for other in others}
    assert not figure.findobj(Annotation) or all(text.get_text() in ("(a)", "(b)")
                                                 for text in figure.findobj(Annotation))
    # Every shown calibrator insufficient: the diagonal alone, an otherwise empty panel.
    alone = plot_reliability([panel])
    assert gids(alone) == {"reliability-diagonal-macro"} and not alone.axes[0].texts
    assert_academic_figure(alone, panels=1)


def test_a_route_without_calibrator_files_or_logits_is_listed_missing_and_not_drawn(study):
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
    missing = [row for row in rows if row["route"] == ROUTES["serving"]]
    assert {row["reason"] for row in missing} == {
        published_reason(SERVING_FULL_DISTRIBUTION_REASON),
        "no logprobs calibrator has been fitted on macro prompts for this serving route",
        "no logprobs calibrator has been fitted on EDGAR releases for this serving route"}
    assert len(missing) == 4 and {row["status"] for row in missing} == {"missing"}
    assert {row["route"] for row in missing} == {ROUTES["serving"]}
    assert f"No calibrator of {ROUTES['serving']} is drawn (reasons in the table)." in table.caption
    assert SERVING_FULL_DISTRIBUTION_REASON not in table.caption
    assert_journal_caption(table.caption, 2)
    figure = plot_reliability(found)
    assert not [gid for gid in gids(figure, kind=None) if gid.startswith("reliability-") and "-serving-" in gid]
    assert len([gid for gid in gids(figure) if gid.startswith("reliability-sampling-")]) == 4
    assert len(figure.axes) == 2
    with pytest.raises(FigureError, match="nothing to draw"):
        plot_reliability([panel for panel in found if panel.arm == "serving"])


def test_a_calibrator_fitted_for_another_arm_is_refused(study, tmp_path):
    target = calibrator_path("serving", "top_k", "macro", tmp_path)
    target.parent.mkdir(parents=True)
    shutil.copyfile(calibrator_path("sampling", "top_k", "macro", study.calibrators_dir), target)
    with pytest.raises(ValueError, match="fitted for arm 'sampling'"):
        reliability_panels(PLAN, arms=("serving",), calibrators_dir=tmp_path)


def test_the_reliability_caption_is_a_journal_caption_with_sample_size_and_interval(study):
    found = reliability_panels(PLAN, arms=("sampling",), calibrators_dir=study.calibrators_dir)
    table = reliability_table(found)
    caption = table.caption
    assert_journal_caption(caption, 2)
    assert f"10 equal-width bins of held-out draws on {ROUTES['sampling']}" in caption
    eces = [panel.ece for panel in found]
    assert f"; expected calibration error {min(eces):.3f} to {max(eces):.3f}. " in caption
    # The sample size is built from the data: one statement per panel where its evidence modes hold the same
    # sample, the modes named where they differ.
    samples = {kind: {(panel.holdout_draws, panel.holdout_prompts) for panel in found if panel.kind == kind}
               for kind in KINDS}
    assert samples == {"macro": {(24, 12)}, "edgar": {(70, 35)}}
    assert " n = (a) 24 held-out draws of 12 prompts and (b) 70 of 35, per evidence mode. " in caption
    uneven = [replace(panel, holdout_draws=22, holdout_prompts=11) if (panel.kind, panel.source) == (
        "macro", "full_distribution") else panel for panel in found]
    assert (" n = (a) logprobs: 24 held-out draws of 12 prompts, logits: 22 of 11; (b) logprobs: 70 of 35, logits: "
            "70 of 35. ") in reliability_table(uneven).caption
    one = [panel for panel in found if panel.kind == "macro"]
    assert " n = 24 held-out draws of 12 prompts, per evidence mode. " in reliability_table(one).caption
    assert " n = 24 held-out draws of 12 prompts. " in reliability_table(one[:1]).caption
    assert "treating the draws of one prompt as independent" in caption
    # What the legend already says and what belongs to the table is not caption text.
    assert "grey line" not in caption and "Bin counts" not in caption
    # Every number of the caption is in the table; the bin count and the confidence are the figure's constants.
    assert_caption_numbers(table, stored=(figures.CALIBRATION_BINS, 100 * figures.RELIABILITY_CONFIDENCE))
    both = reliability_table(reliability_panels(PLAN, arms=DEFAULT_ARMS, calibrators_dir=study.calibrators_dir))
    assert_journal_caption(both.caption, 2)
    assert_caption_numbers(both, stored=(figures.CALIBRATION_BINS, 100 * figures.RELIABILITY_CONFIDENCE))
    # The interval of the held-out AUC is table matter: a note names it by the method the calibration stored.
    notes = " ".join(table.notes)
    assert {panel.auc_method for panel in found} == {"prompt_clustered_stratified_percentile_bootstrap"}
    assert "95 percent prompt-clustered bootstrap interval (1000 resamples)" in notes
    assert "prompt_clustered" not in table.to_markdown().split("\n\n")[-1]
    assert "expected calibration error" in notes
    markdown = table.to_markdown()
    assert "Moody's" in markdown and "SEC EDGAR" in markdown and "Financial Modeling Prep (FMP)" in markdown
    assert_notes(table)
    renamed = replace(found[0], auc_method="another resampling scheme")
    with pytest.raises(FigureError, match="no wording for the interval method"):
        reliability_table([renamed, *found[1:]])


def test_no_value_label_sits_on_a_mark_and_the_bin_counts_are_in_the_table(study, panels):
    """Counts and values are in the table view, not printed beside every mark (Requirement 8.6)."""
    found = reliability_panels(PLAN, arms=("sampling",), calibrators_dir=study.calibrators_dir)
    figure = plot_reliability(found)
    assert sorted(text.get_text() for text in figure.findobj(Annotation)) == ["(a)", "(b)"]
    rows = csv_rows(reliability_table(found).to_csv())
    assert [row["bin_count"] for row in rows] == [str(bin_.count) for panel in found for bin_ in panel.bins]
    assert not plot_gap(panels).findobj(Annotation)


# ---------- the academic style: labelled axes, grey levels, a legend, nothing written inside the plot ----------

def test_both_figures_are_drawn_in_the_academic_style_with_both_axes_labelled(panels, study):
    found = reliability_panels(PLAN, arms=DEFAULT_ARMS, calibrators_dir=study.calibrators_dir)
    gap = plot_gap(panels)
    assert_academic_figure(gap, panels=1)
    assert gap.axes[0].get_xlabel() == "serving route"
    assert gap.axes[0].get_ylabel() == ("accuracy gap, before minus after the cutoff\n"
                                        "(difference in share of months correct)")
    reliability = plot_reliability(found)
    assert_academic_figure(reliability, panels=2)
    for axes in reliability.axes:
        assert axes.get_xlabel() == "mean predicted recall probability\nin the bin (probability)"
        assert axes.get_ylabel() == "observed share of memorized draws\nin the bin (share)"
    for figure in (gap, reliability):
        assert_academic_svg(figure_svg(figure))
    # A route with no variant drawn leaves its place empty; nothing is written there.
    stored = "no planned month carries the value in both renderings"
    absent = replace(panels[0], points=tuple(replace(point, missing=stored) for point in panels[0].points))
    empty = plot_gap([absent])
    assert_academic_figure(empty, panels=1)
    assert not [gid for gid in gids(empty, kind=None) if gid.startswith("gap-serving-")] and not empty.axes[0].texts
    table = gap_table([absent])
    assert ("Not drawn on modal-gemma3-4b: both variants unadjusted; both variants adjusted with logprobs; both "
            "variants adjusted with logits.") in table.caption
    assert "no variant has a value" in table.caption and " n = no month with a value. " in table.caption
    published = "no planned month has the value under both the identifying and the anonymised prompt"
    assert published in table.to_csv() and published not in table.caption and stored not in table.to_markdown()
    # A reason without a published wording is refused: it is never published as stored.
    unknown = replace(panels[0], points=tuple(replace(point, missing="too few months") for point in panels[0].points))
    with pytest.raises(FigureError, match="no published wording"):
        gap_table([unknown])
    assert_journal_caption(table.caption, 1)


def test_series_are_told_apart_by_marker_and_grey_level_and_named_in_a_legend(panels, study):
    found = reliability_panels(PLAN, arms=("sampling",), calibrators_dir=study.calibrators_dir)
    figure = plot_gap(panels)
    for panel in panels:
        for point in panel.drawn_points:
            expected = series_style(VARIANTS.index(point.variant))
            line = by_gid(figure, f"gap-{panel.arm}-{point.variant}")
            assert line.get_marker() == expected["marker"] and line.get_linestyle() == "None"
            assert to_rgb(line.get_markerfacecolor()) == to_rgb(expected["markerfacecolor"])
            assert to_rgb(line.get_markeredgecolor()) == to_rgb(expected["markeredgecolor"])
            assert list(line.get_ydata()) == [point.statistic]
    assert len({series_style(index)["marker"] for index in range(len(VARIANTS))}) == len(VARIANTS)
    assert legend_labels(figure) == [
        "single draw, unadjusted", "single draw, adjusted with logprobs", "single draw, adjusted with logits",
        "ensemble, unadjusted", "ensemble, adjusted with logprobs", "ensemble, adjusted with logits"]
    only_serving = plot_gap(panels[:1])  # the legend names the drawn variants only
    assert legend_labels(only_serving) == ["single draw, unadjusted", "single draw, adjusted with logprobs",
                                           "ensemble, unadjusted", "ensemble, adjusted with logprobs"]
    reliability = plot_reliability(found)
    for panel in found:
        expected = series_style(SOURCES.index(panel.source))
        line = by_gid(reliability, f"reliability-{panel.key}")
        assert line.get_marker() == expected["marker"]
        assert is_grey(line.get_color()) and is_grey(line.get_markerfacecolor())
    assert series_style(0)["marker"] != series_style(1)["marker"]
    kept = next(panel.kept for panel in found if panel.source == "full_distribution")
    assert legend_labels(reliability) == ["logprobs (top 5)", f"logits (top {kept} of the next-token vector)",
                                          "perfect calibration"]
    for drawn in (figure, reliability):
        for artist in drawn.findobj(Line2D):
            assert is_grey(artist.get_color()) and is_grey(artist.get_markeredgecolor()), artist.get_gid()


# ---------- outputs ----------

def test_each_figure_is_written_as_one_svg_with_a_csv_and_markdown_once(study, tmp_path):
    assert FIGURES_DIR == ROOT / "docs" / "assets" / "finance-sim"
    assert FIGURE_NAMES == ("gap", "reliability")
    assert figures.PUBLISHED_FIGURES == ("gap", "reliability", "draws", "convergence", "draw-spread")
    # 5, 6 and 7 are the figures of the catalogue module.
    assert dict(figures.FIGURE_NUMBERS) == {"gap": 1, "reliability": 2, "draws": 3, "convergence": 4,
                                            "draw-spread": 8}
    out = tmp_path / "assets"
    written = render_figures(PLAN, analysis_dir=study.analysis_dir, calibrators_dir=study.calibrators_dir,
                             out_dir=out)
    expected = [path for name in FIGURE_NAMES for path in figure_paths(name, out).values()]
    assert list(written) == expected
    assert [path.name for path in figure_paths("gap", out).values()] == ["gap.svg", "gap.csv", "gap.md"]
    assert sorted(path.name for path in out.iterdir()) == sorted(
        f"{name}{suffix}" for name in FIGURE_NAMES for suffix in (".svg", ".csv", ".md"))
    assert not [path.name for path in out.iterdir() if "light" in path.name or "dark" in path.name]
    for name in FIGURE_NAMES:
        paths = figure_paths(name, out)
        assert_academic_svg(paths["svg"].read_text(encoding="utf-8"))
        assert paths["markdown"].read_text(encoding="utf-8").startswith(f"Figure {figures.FIGURE_NUMBERS[name]}. ")
    before = {path: path.read_bytes() for path in expected}
    assert render_figures(PLAN, analysis_dir=study.analysis_dir, calibrators_dir=study.calibrators_dir,
                          out_dir=out) == written
    assert {path: path.read_bytes() for path in expected} == before  # deterministic, and the same output is a no-op
    with pytest.raises(FigureError, match="refusing to replace"):
        render_figures(PLAN, arms=("sampling",), analysis_dir=study.analysis_dir,
                       calibrators_dir=study.calibrators_dir, out_dir=out)
    assert {path: path.read_bytes() for path in expected} == before


def test_the_render_does_not_depend_on_the_processs_global_rc_params(panels, study):
    """Whatever rcParams an earlier test leaves in the process (the paper style's pdf renderer and tight bounding
    box, another font or dpi), the bytes are those of a clean process: a figure is built, laid out, measured and
    saved under matplotlib's own defaults plus the academic style, and nothing global changes."""
    found = reliability_panels(PLAN, arms=DEFAULT_ARMS, calibrators_dir=study.calibrators_dir)
    draws = {"gap": lambda: plot_gap(panels), "reliability": lambda: plot_reliability(found)}

    def hashes() -> dict[str, str]:
        return {name: hashlib.sha256(figure_svg(draw()).encode("utf-8")).hexdigest() for name, draw in draws.items()}

    before = dict(matplotlib.rcParams)
    clean = hashes()
    with foreign_rc_params():
        assert matplotlib.rcParams["savefig.format"] == "pdf" and matplotlib.rcParams["savefig.bbox"] == "tight"
        assert hashes() == clean
    assert dict(matplotlib.rcParams) == before
    defaults = figures._default_rc()
    assert "backend" not in defaults and defaults["savefig.format"] == "png" and defaults["savefig.bbox"] is None
    style = figures.figure_rc()
    assert style["font.family"] == "serif" and style["savefig.format"] == "png" and style["axes.grid"] is False


def test_the_shared_helpers_live_in_the_style_module_and_are_re_exported_here():
    """The catalogue figures import these names from figures; both modules hold the same objects."""
    from scripts.finance_sim import figure_style

    for name in ("figure_rc", "figure_svg", "lay_out", "write_outputs", "FigureError", "DERIVED_ONLY", "FIGURES_DIR",
                 "freeze_layout", "SVG_HASH_SALT"):
        assert getattr(figures, name) is getattr(figure_style, name), name
        assert name in figures.__all__ and name in figure_style.__all__, name


def test_the_markdown_is_the_caption_the_csv_as_a_table_then_the_notes_the_sources_and_the_derived_only_line(panels,
                                                                                                              study):
    found = reliability_panels(PLAN, arms=DEFAULT_ARMS, calibrators_dir=study.calibrators_dir)
    for table in (gap_table(panels), reliability_table(found)):
        lines = table.to_markdown().splitlines()
        assert lines[0] == table.caption and lines[1] == ""
        rows = list(csv.reader(io.StringIO(table.to_csv())))
        assert lines[3] == "|" + "|".join("---" for _ in rows[0]) + "|"
        end = 3 + len(rows)
        rendered = [[value.strip().replace("\\|", "|") for value in line.strip("|").split(" | ")]
                    for line in [lines[2], *lines[4:end]]]
        assert rendered == rows
        assert lines[end] == "" and lines[end + 1:] == [*table.notes, f"Sources: {'; '.join(table.sources)}.",
                                                        figures.DERIVED_ONLY]
        assert table.notes and table.sources
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
        f"{name}{suffix}" for name in FIGURE_NAMES for suffix in (".svg", ".csv", ".md"))
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
    assert sorted(path.name for path in only.iterdir()) == ["reliability.csv", "reliability.md", "reliability.svg"]
