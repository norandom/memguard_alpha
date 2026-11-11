"""Draw-distribution figure F3 (task 8.3; Requirements 10.4 and 10.6).

Fixtures only. The probe (sampling) arm is calibrated and executed through the
real 7.6 calibration and 7.7 execution on the generation and full-distribution
doubles of the 7.7 tests (the full run, its first draw answered with prose, and
four negative-control releases), its 7.11 ensemble metrics are written as JSON
under ``tmp_path``, and a synthetic serving-arm stage (32-draw results derived
with the 7.7 derivation, no request) adds first draws that flip or fall outside
the band and items whose halves are unresolved. The figure is read from those
files. Sockets are blocked and nothing is written under ``runs/`` or ``docs/``.
The Agg backend is selected before anything imports pyplot.
"""

from __future__ import annotations

import copy
import csv
import io
import json
import socket
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.text import Annotation  # noqa: E402

from recall_guard.core.distribution import summarize_distribution  # noqa: E402
from recall_guard.dataset.edgar_outlook import render_outlook_prompt  # noqa: E402
from recall_guard.dataset.macro_regime import render_anonymised, render_identifying  # noqa: E402
from scripts.finance_sim import ensemble_metrics as metrics_module  # noqa: E402
from scripts.finance_sim import figures  # noqa: E402
from scripts.finance_sim.calibrate import DrawSettings, FitSettings, run_calibration  # noqa: E402
from scripts.finance_sim.ensemble_metrics import (  # noqa: E402
    FIRST_DRAW_RATE,
    WILSON_METHOD,
    ensemble_metrics,
    metrics_path,
    write_metrics,
)
from scripts.finance_sim.execute import (  # noqa: E402
    DEFINITIONS,
    UNRESOLVED_SUMMARY,
    ExecutionItem,
    load_calibrators,
    load_result,
    macro_items,
    result_path,
    write_result,
)
from scripts.finance_sim.figure_style import series_style  # noqa: E402
from scripts.finance_sim.figures import (  # noqa: E402
    FIGURES_DIR,
    FigureError,
    figure_paths,
    figure_svg,
    route_label,
)
from scripts.finance_sim.figures_draws import (  # noqa: E402
    DRAW_SPREAD_COLUMNS,
    EXAMPLE_DEFAULT,
    SPLIT_CHECK_MIN_DRAWS,
    draw_spread_table,
    draws_figure,
    draws_figure_from_records,
    draws_table,
    exchangeability_shares,
    main,
    plot_draw_spread,
    plot_draws,
    read_ensemble_metrics,
    render_draws,
)
from scripts.finance_sim.limits import BudgetLedger, LedgerCallBudget, StudyLimits  # noqa: E402
from scripts.finance_sim.plan import write_plan  # noqa: E402
from tests.scripts.finance_sim.test_ensemble_metrics import (  # noqa: E402
    DEF,
    FIRST_TYPES,
    PLAN,
    edgar_releases,
    first_vectors,
    run,
    serving_items,
    serving_result,
    shifted,
)
from tests.scripts.finance_sim.test_execute import (  # noqa: E402
    ALLOWANCE,
    MACRO_ITEMS,
    MONTHS,
    SEED_BASE,
    TEMPERATURE,
    Clock,
    FakeFull,
    FakeHost,
    macro_row,
    realized,
    release_text,
)
from tests.scripts.finance_sim.test_figures import (  # noqa: E402
    assert_academic_figure,
    assert_academic_svg,
    assert_caption_numbers,
    assert_journal_caption,
    assert_notes,
    foreign_rc_params,
    is_grey,
    legend_labels,
)

ROOT = Path(__file__).resolve().parents[3]
ROUTES = {arm: route_label(PLAN, arm) for arm in ("serving", "sampling")}
#: The six files one render writes: the first-draw figure and the draw-spread figure, three files each.
WRITTEN = ["draw-spread.csv", "draw-spread.md", "draw-spread.svg", "draws.csv", "draws.md", "draws.svg"]
LOW = shifted(DEF, inflation=-1.0)
#: Ten defensive draws, six with the opposite inflation loading, then sixteen defensive: the first half (draws 0 to
#: 15) splits into separated clusters while the whole ensemble of 32 does not (6 of 32 is below the cluster mass).
SPLIT_HALF = [*[DEF] * 10, *[LOW] * 6, *[DEF] * 16]
SERVING_KINDS = (*FIRST_TYPES, "split_half", "split_half")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("the draw figure attempted a network connection")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)


@pytest.fixture(scope="module")
def study(tmp_path_factory) -> SimpleNamespace:
    """The probe arm's full run and negative control, and a synthetic serving full run, stored under tmp_path."""
    root = tmp_path_factory.mktemp("figures-draws")
    ledger = BudgetLedger.open(root / "calibration" / "budget.json", PLAN,
                               StudyLimits(PLAN.plan_sha256, "sampling", "calibration", 1000, 1_000_000, 1_000_000),
                               ALLOWANCE, clock=Clock())
    run_calibration(PLAN, arm="sampling", macro_rows=[macro_row(month) for month in PLAN.months.calibration],
                    edgar_releases=edgar_releases(), client=FakeHost(LedgerCallBudget(ledger)), ledger=ledger,
                    settings=DrawSettings(draws=2, temperature=TEMPERATURE, seed_base=SEED_BASE),
                    sources=("top_k", "full_distribution"), full_distribution=FakeFull(),
                    fit=FitSettings(min_auc=0.6, min_valid=10), calibrators_dir=root / "calibrators",
                    sleep=lambda seconds: None)
    macro = load_calibrators(PLAN, arm="sampling", kind="macro", calibrators_dir=root / "calibrators")
    edgar = load_calibrators(PLAN, arm="sampling", kind="edgar", calibrators_dir=root / "calibrators")
    full = run(root, macro_items(PLAN, MACRO_ITEMS, stage="full_run", arm="sampling"), macro, "full_run", prose=(0,))
    releases = tuple(ExecutionItem(row_id, "edgar", "outlook", render_outlook_prompt(release_text(row_id, 0)))
                     for row_id in PLAN.edgar.negative_control[:4])
    negative = run(root, releases, edgar, "negative_control")
    analysis_dir, results_dir = root / "analysis", root / "results"
    write_metrics(ensemble_metrics(PLAN, full.results.values(), arm="sampling"), analysis_dir)
    write_metrics(ensemble_metrics(PLAN, negative.results.values(), arm="sampling", stage="negative_control"),
                  analysis_dir)
    serving = [serving_result(item, SPLIT_HALF if kind == "split_half" else first_vectors(kind))
               for item, kind in zip(serving_items(len(SERVING_KINDS)), SERVING_KINDS, strict=True)]
    for result in serving:
        write_result(result, results_dir)
    write_metrics(ensemble_metrics(PLAN, serving, arm="serving"), analysis_dir)
    return SimpleNamespace(root=root, analysis_dir=analysis_dir, results_dir=results_dir)


def stored(study: SimpleNamespace, arm: str = "sampling", stage: str = "full_run") -> dict:
    return json.loads(metrics_path(arm, stage, study.analysis_dir).read_text(encoding="utf-8"))


def stored_result(study: SimpleNamespace, item_id: str, rendering: str, arm: str = "sampling",
                  stage: str = "full_run"):
    return load_result(result_path(arm, stage, item_id, rendering, study.results_dir))


def figure_data(study: SimpleNamespace, arm: str = "sampling", stage: str = "full_run", **kwargs):
    return draws_figure_from_records(PLAN, arm=arm, stage=stage, analysis_dir=study.analysis_dir,
                                     results_dir=study.results_dir, **kwargs)


@pytest.fixture(scope="module")
def sampled(study):
    return figure_data(study)


@pytest.fixture(scope="module")
def serving(study):
    return figure_data(study, arm="serving")


def csv_rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text)))


def by_gid(figure, gid: str):
    return next(artist for artist in figure.findobj() if artist.get_gid() == gid)


def gids(figure) -> set[str]:
    return {artist.get_gid() for artist in figure.findobj() if artist.get_gid()}


# ---------- read, never recomputed ----------

def test_every_first_draw_statistic_is_read_from_the_stored_metrics_and_recomputes_nothing(study, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("the figure recomputed the ensemble metrics instead of reading them")

    for name in ("ensemble_metrics", "ensemble_metrics_from_records", "half_ensemble_variance", "wilson_interval",
                 "random_answerer_null", "decide_from_draws"):
        monkeypatch.setattr(metrics_module, name, forbidden)
    data = figure_data(study)
    first = stored(study)["metrics"][FIRST_DRAW_RATE]
    assert [(item.item_id, item.rendering) for item in data.items] == [
        (row["item_id"], row["rendering"]) for row in first["items"]]
    for item, row in zip(data.items, first["items"], strict=True):
        assert (item.first_draw_rank, item.outside_band, item.category_differs, item.off) == (
            row["first_draw_rank"], row["outside_band"], row["category_differs"], row["off"])
        assert (item.first_category, item.consensus_category) == (row["first_category"], row["consensus_category"])
    assert [(entry[0], entry[1], entry[2]) for entry in data.missing_items] == [
        (row["item_id"], row["rendering"], row["reason"]) for row in first["missing_items"]]
    groups = [first["overall"], first["by_rendering"]["identifying"], first["by_rendering"]["anonymised"]]
    assert [rate.group for rate in data.rates] == ["all", "identifying", "anonymised"]
    for rate, entry in zip(data.rates, groups, strict=True):
        assert (rate.n, rate.off, rate.rate, rate.low, rate.high) == (
            entry["n"], entry["off"], entry["rate"], entry["interval"]["lo"], entry["interval"]["hi"])
        assert (rate.outside_band, rate.category_differs) == (entry["outside_band"], entry["category_differs"])
    assert data.band == tuple(stored(study)["settings"]["first_draw_band"]) == PLAN.analysis.first_draw_band
    # A changed stored rate is what the table shows: the figure reads the file.
    changed = copy.deepcopy(stored(study))
    changed["metrics"][FIRST_DRAW_RATE]["overall"]["rate"] = 0.123456
    results = [stored_result(study, row["item_id"], row["rendering"]) for row in first["items"]]
    row = next(row for row in csv_rows(draws_table(draws_figure(PLAN, changed, results)).to_csv())
               if row["row"] == "rate" and row["group"] == "all")
    assert row["rate"] == "0.123456"


def test_the_example_items_draws_are_its_stored_per_draw_values_with_the_librarys_quartiles(study, sampled):
    example = sampled.example
    result = stored_result(study, example.item_id, example.rendering)
    definition = DEFINITIONS[result.kind]
    assert example.summary.values == result.summary_values
    summary = summarize_distribution(result.draws_for("none"), definition).summary_value
    assert (example.summary.q25, example.summary.median, example.summary.q75) == (
        summary.quantiles[25], summary.quantiles[50], summary.quantiles[75])
    assert example.first_rank == summary.first_draw_rank == result.variant("ensemble_none").first_draw_rank
    strips = {strip.adjustment: strip for strip in example.estimates}
    assert set(strips) == {"top_k", "full_distribution"}
    assert strips["top_k"].values == tuple(None if draw.parsed is None else draw.estimate
                                           for draw in result.recorded_draws)
    assert strips["full_distribution"].values == tuple(None if entry is None else entry.estimate
                                                       for entry in result.full_distribution)
    quantiles = summarize_distribution(result.draws_for("top_k"), definition).estimate_quantiles
    assert (strips["top_k"].q25, strips["top_k"].median, strips["top_k"].q75) == (
        quantiles[25], quantiles[50], quantiles[75])
    marks = {mark.variant: mark for mark in (*example.summary_marks, *example.estimate_marks)
             if mark.quantity == "summary value"}
    assert marks["ensemble_none"].value == result.variant("ensemble_none").summary_value
    assert marks["ensemble_top_k"].value == result.variant("ensemble_top_k").adjusted_summary_value
    estimates = {mark.variant: mark.value for mark in example.estimate_marks}
    assert estimates == {variant: result.variant(variant).p_memorized
                         for variant in ("ensemble_top_k", "ensemble_full_distribution")}
    table = csv_rows(draws_table(sampled).to_csv())
    draws = [row for row in table if row["row"] == "draw" and row["quantity"] == "summary value"]
    assert [row["value"] for row in draws] == [figures._number(value) for value in result.summary_values]
    assert [row["first_draw"] for row in draws] == ["yes"] + ["no"] * (result.ensemble_draws - 1)


# ---------- Figure 8: the single draws of one prompt as points beside the ensemble decisions ----------

def test_every_parsed_draw_of_the_example_item_is_a_visible_point_beside_the_ensemble_decisions(sampled, serving,
                                                                                               study):
    """One point per draw, tied draws side by side, so the reader sees how much repeated answers vary."""
    negative = figure_data(study, stage="negative_control")
    for data in (sampled, serving, negative):
        figure = plot_draw_spread(data)
        example = data.example
        values = [value for _, value in example.summary.present]
        points = by_gid(figure, "spread-draws")
        assert points.get_linestyle() == "None" and points.get_marker() == series_style(1)["marker"]
        assert list(points.get_ydata()) == values and len(values) == example.parsed
        xs = list(points.get_xdata())
        assert all(abs(x) <= 0.36 for x in xs)
        assert len(set(zip(xs, values, strict=True))) == len(values), "no two draws share a place"
        marks = [mark for mark in example.summary_marks if mark.value is not None]
        for place, mark in enumerate(marks, start=1):
            line = by_gid(figure, f"spread-ensemble-{mark.variant}")
            assert list(line.get_xdata()) == [place] and list(line.get_ydata()) == [mark.value]
            assert line.get_marker() == series_style(0)["marker"] != points.get_marker()
        for mark in example.summary_marks:
            if mark.value is None:
                assert f"spread-ensemble-{mark.variant}" not in gids(figure)
        # No box hides the points, no label sits beside a mark, and there is one panel.
        assert not figure.findobj(Annotation) and len(figure.axes) == 1
        assert {gid for gid in gids(figure)} == {"spread-draws", *(f"spread-ensemble-{mark.variant}"
                                                                    for mark in marks)}
    # The other route's 32 identical draws: 32 points side by side at one value, not one point.
    assert len(set(serving.example.summary.values)) == 1 and serving.example.draws == 32
    points = by_gid(plot_draw_spread(serving), "spread-draws")
    assert len(set(points.get_xdata())) == 32 and set(points.get_ydata()) == {serving.example.summary.values[1]}
    # Draws that differ are spread along the value axis as they are, none moved.
    assert len(set(sampled.example.summary.values) - {None}) > 1


def test_the_draw_spread_table_lists_every_draw_and_the_ensemble_decisions(study, sampled):
    example = sampled.example
    result = stored_result(study, example.item_id, example.rendering)
    table = draw_spread_table(sampled)
    assert table.name == "draw-spread" and table.columns == DRAW_SPREAD_COLUMNS
    rows = csv_rows(table.to_csv())
    draws = [row for row in rows if row["row"] == "draw"]
    assert [row["value"] for row in draws] == [figures._number(value) for value in result.summary_values]
    assert [row["first_draw"] for row in draws] == ["yes"] + ["no"] * (result.ensemble_draws - 1)
    assert {row["route"] for row in rows} == {ROUTES["sampling"]}
    assert {(row["item_id"], row["rendering"]) for row in rows} == {(example.item_id, example.rendering)}
    (spread,) = [row for row in rows if row["row"] == "spread"]
    assert (spread["q25"], spread["median"], spread["q75"]) == tuple(
        figures._number(value) for value in (example.summary.q25, example.summary.median, example.summary.q75))
    ensemble = {row["variant"]: row for row in rows if row["row"] == "ensemble"}
    assert set(ensemble) == {"ensemble_none", "ensemble_top_k", "ensemble_full_distribution"}
    assert ensemble["ensemble_none"]["value"] == figures._number(result.variant("ensemble_none").summary_value)
    # The table of the first-draw figure keeps everything it held: the example item's rows are still there.
    kinds = {row["row"] for row in csv_rows(draws_table(sampled).to_csv())}
    assert kinds == {"draw", "spread", "ensemble", "item", "rank_count", "missing_item", "rate"}
    quantities = {row["quantity"] for row in csv_rows(draws_table(sampled).to_csv()) if row["row"] == "draw"}
    assert quantities == {"summary value", "p_memorized top_k", "p_memorized full_distribution"}


def test_the_default_example_is_the_first_item_with_a_decided_first_draw_and_another_can_be_chosen(study, sampled):
    first = stored(study)["metrics"][FIRST_DRAW_RATE]
    # The fixture's first ensemble has an unparsed first draw, so the default skips it, and so it skips every
    # decided item whose consensus has no summary value (an unresolved component).
    analysed = stored(study)["items"][0]
    assert stored_result(study, analysed["item_id"], analysed["rendering"]).recorded_draws[0].parsed is None
    decided = [(row["item_id"], row["rendering"]) for row in first["items"]]
    shown = [key for key in decided if stored_result(study, *key).variant("ensemble_none").summary_value is not None]
    assert shown and shown[0] != decided[0]  # the fixture exercises the summary-value condition
    assert (sampled.example.item_id, sampled.example.rendering) == shown[0]
    assert sampled.example.summary_marks[0].value is not None
    assert sampled.example.chosen == EXAMPLE_DEFAULT
    # How the example was picked is a note below the table, not caption text.
    rule = "The example is the first item in plan order whose first draw and ensemble both give a value."
    assert rule in draw_spread_table(sampled).notes and "plan order" not in draw_spread_table(sampled).caption
    chosen = figure_data(study, example=(analysed["item_id"], analysed["rendering"]))
    assert chosen.example.summary.first is None
    assert "The example was chosen for the figure." in draw_spread_table(chosen).notes
    assert rule not in draw_spread_table(chosen).notes and "chosen" not in draw_spread_table(chosen).caption
    assert chosen.example.parsed == chosen.example.draws - 1
    assert len(by_gid(plot_draw_spread(chosen), "spread-draws").get_ydata()) == chosen.example.parsed
    pilot = next(month for month in PLAN.months.pilot)
    with pytest.raises(FigureError, match="not an analysed item"):
        figure_data(study, example=(f"macro-{pilot}", "identifying"))


def _draw_shares(values: list[float], categories: list, consensus, band) -> tuple[float, float]:
    """Brute force: each parsed draw in the first draw's place, with the 7.11 off rule."""
    lower, upper = band
    off = outside = 0
    for value, category in zip(values, categories, strict=True):
        rank = (sum(other < value for other in values) + 0.5 * sum(other == value for other in values)) / len(values)
        out = not lower <= rank <= upper
        outside += out
        off += out or category != consensus
    return off / len(values), outside / len(values)


def test_the_baseline_follows_the_rates_inclusion_rule_for_items_without_a_consensus_category(study, sampled):
    """7.11 includes a no-category item only when its first draw is outside the band, so it is always off."""
    band = PLAN.analysis.first_draw_band
    assert exchangeability_shares([0.1 * i for i in range(8)], [1] * 8, None, band) == (1.0, 1.0)
    rows = stored(study)["metrics"][FIRST_DRAW_RATE]["items"]
    count = sum(1 for row in rows if row["consensus_category"] is None)
    without = [item for item in sampled.items if item.consensus_category is None]
    with_category = [item for item in sampled.items if item.consensus_category is not None]
    # The probe fixture: most consensuses are split, so most included items are no-category items.
    assert len(without) == count > len(rows) / 2 and with_category and len(rows) == 26
    for item in without:
        assert item.no_category and item.off and item.outside_band  # included only outside the band
        assert (item.baseline_off, item.baseline_outside) == (1.0, 1.0)
    expected = []
    for item in with_category:
        result = stored_result(study, item.item_id, item.rendering)
        pairs = [(value, category) for value, category in zip(result.summary_values, result.categories, strict=True)
                 if value is not None]
        shares = _draw_shares([value for value, _ in pairs], [category for _, category in pairs],
                              item.consensus_category, band)
        assert (item.baseline_off, item.baseline_outside) == pytest.approx(shares)
        expected.append(shares[0])
    overall = sampled.rates[0]
    assert overall.no_category == count and overall.n == 26
    assert overall.baseline_off == pytest.approx((count + sum(expected)) / 26)
    unconditional = (sum(_draw_shares(*_parsed(study, item), None, band)[1] for item in without) + sum(expected)) / 26
    assert overall.baseline_off > unconditional  # the selection the unconditional share would miss
    for rate in sampled.rates[1:]:
        chosen = [item for item in sampled.items if item.rendering == rate.group]
        assert rate.no_category == sum(item.no_category for item in chosen)
        assert rate.baseline_off == pytest.approx(sum(item.baseline_off for item in chosen) / len(chosen))
    table = draws_table(sampled)
    # The inclusion rule is methods text: one note below the table, not part of the caption.
    assert ("An item whose ensemble has no category enters only when its first draw is outside that range, in the "
            "observed and in the expected share alike; it counts 1 in the expected share.") in table.notes
    assert "no category" not in table.caption and " ".join(table.notes).count("has no category") == 1
    row = next(row for row in csv_rows(table.to_csv()) if row["row"] == "rate" and row["group"] == "all")
    assert row["count_no_category"] == str(count)


def _parsed(study, item) -> tuple[list[float], list]:
    result = stored_result(study, item.item_id, item.rendering)
    pairs = [(value, category) for value, category in zip(result.summary_values, result.categories, strict=True)
             if value is not None]
    return [value for value, _ in pairs], [category for _, category in pairs]


def test_an_unresolved_example_ensemble_is_not_drawn_and_the_table_gives_the_reason(study):
    first = stored(study)["metrics"][FIRST_DRAW_RATE]["items"]
    key = next((row["item_id"], row["rendering"]) for row in first if row["consensus_category"] is None)
    result = stored_result(study, *key)
    consensus = result.variant("ensemble_none")
    assert consensus.status != "failed" and consensus.summary_value is None
    data = figure_data(study, example=key)
    marks = {mark.variant: mark for mark in data.example.summary_marks}
    assert marks["ensemble_none"].reason == UNRESOLVED_SUMMARY
    for variant in ("ensemble_top_k", "ensemble_full_distribution"):
        stored_variant = result.variant(variant)
        if stored_variant.status != "failed" and stored_variant.adjusted_summary_value is None:
            assert marks[variant].reason == UNRESOLVED_SUMMARY
    table = draw_spread_table(data)
    assert "the unadjusted ensemble gives no value" in table.caption
    assert "Not drawn (reason in the table): ensemble, unadjusted" in table.caption
    assert UNRESOLVED_SUMMARY not in table.caption
    row = next(row for row in csv_rows(table.to_csv()) if row["variant"] == "ensemble_none")
    assert row["reason"] == UNRESOLVED_SUMMARY and row["value"] == ""
    assert "spread-ensemble-ensemble_none" not in gids(plot_draw_spread(data))
    assert_journal_caption(table.caption, 8)


def test_the_exchangeability_baseline_is_about_half_outside_the_band_for_eight_distinct_values(study, sampled):
    band = PLAN.analysis.first_draw_band
    distinct = [0.1 * i for i in range(8)]
    assert exchangeability_shares(distinct, [1] * 8, 1, band) == (0.5, 0.5)
    assert exchangeability_shares([0.3] * 8, [1] * 8, 1, band) == (0.0, 0.0)  # all tied: mid-rank 0.5
    flips = exchangeability_shares(distinct, [1] * 7 + [-1], 1, band)
    assert flips == (0.5, 0.5)  # the flipped draw is the highest, already outside
    assert exchangeability_shares(distinct, [-1] + [1] * 7, 1, band)[0] == 0.5
    assert exchangeability_shares(distinct, [1, 1, 1, -1, 1, 1, 1, 1], 1, band) == (5 / 8, 0.5)
    for item in sampled.items:
        result = stored_result(study, item.item_id, item.rendering)
        values = [value for value in result.summary_values if value is not None]
        categories = [category for value, category in zip(result.summary_values, result.categories, strict=True)
                      if value is not None]
        assert (item.baseline_off, item.baseline_outside) == exchangeability_shares(
            values, categories, item.consensus_category, band)
        if item.consensus_category is not None and len(set(values)) == len(values) == 8:
            assert item.baseline_outside == 0.5
    assert any(len(set(stored_result(study, item.item_id, item.rendering).summary_values)) == 8
               for item in sampled.items if item.consensus_category is not None)
    overall = sampled.rates[0]
    assert overall.baseline_off == pytest.approx(sum(item.baseline_off for item in sampled.items) / overall.n)
    assert overall.baseline_off > 0.3  # the reference is not zero
    figure = plot_draws(sampled)
    for rate in sampled.rates:
        assert list(by_gid(figure, f"draws-baseline-{rate.group}").get_ydata()) == [rate.baseline_off]
        assert list(by_gid(figure, f"draws-rate-{rate.group}").get_ydata()) == [rate.rate]
        bar = by_gid(figure, f"draws-rate-interval-{rate.group}")  # the interval bar is kept
        assert list(bar.get_ydata()) == [rate.low, rate.high] and bar.get_linestyle() == "-"
    assert legend_labels(figure) == ["observed", "expected if draws were exchangeable"]
    assert f"against {figures.rounded(overall.baseline_off)} expected if the first draw were exchangeable" in (
        draws_table(sampled).caption)


def test_items_whose_halves_are_listed_as_unresolved_are_marked_in_the_table(study, serving):
    cell = stored(study, "serving")["metrics"]["half_ensemble_variance_summary"]["results"]["none"]
    unresolved = {(row["item_id"], row["rendering"]) for row in cell["missing_items"]
                  if UNRESOLVED_SUMMARY in row["reason"]}
    assert len(unresolved) == 2
    marked = {(item.item_id, item.rendering) for item in serving.items if item.halves_unresolved}
    assert marked == unresolved
    for item in serving.items:
        if not item.halves_unresolved:
            assert item.halves_unresolved is None  # never shown as resolved
    table = draws_table(serving)
    rows = [row for row in csv_rows(table.to_csv()) if row["row"] == "item"]
    assert {(row["item_id"], row["rendering"]) for row in rows if UNRESOLVED_SUMMARY in row["halves"]} == unresolved
    assert ("Column halves marks an item whose half-ensemble decisions were unresolved (2 items); an unmarked item "
            "is not shown as resolved.") in table.notes
    assert not any("resolved" == row["halves"] for row in rows)
    assert "halves" not in table.caption  # table matter, not part of the figure's one message
    assert_notes(table)


def test_four_draw_halves_are_stated_as_never_assessed(sampled):
    assert sampled.half_draws == 4 < SPLIT_CHECK_MIN_DRAWS == 8
    assert not any(item.halves_unresolved for item in sampled.items)
    assert (f"Halves of 4 draws are never assessed: the split check needs at least {SPLIT_CHECK_MIN_DRAWS} parsed "
            "draws.") in draws_table(sampled).notes


def test_the_negative_control_stage_draws_the_outlook(study):
    data = figure_data(study, stage="negative_control")
    assert data.kind == "edgar" and data.renderings == ("outlook",) and data.example.rendering == "outlook"
    assert [rate.group for rate in data.rates] == ["all", "outlook"]
    for table, number in ((draws_table(data), 3), (draw_spread_table(data), 8)):
        markdown = table.to_markdown()
        assert "SEC EDGAR" in markdown and "Moody's" not in markdown
        assert "SEC EDGAR" not in table.caption
        assert_journal_caption(table.caption, number)
        assert_notes(table)
    assert "(outlook prompt: " in draws_table(data).caption
    assert "The value is the outlook score of the release." in draw_spread_table(data).notes
    assert "The value is" not in draw_spread_table(data).caption
    assert plot_draw_spread(data).axes[0].get_ylabel() == "outlook (score)"
    figure = plot_draws(data)
    assert [label.get_text() for label in figure.axes[0].get_xticklabels()] == [f"all items\n(n = {data.rates[0].n})"]


def test_records_that_disagree_or_belong_elsewhere_are_refused(study, tmp_path):
    data = stored(study)
    first = data["metrics"][FIRST_DRAW_RATE]["items"][0]
    results = [stored_result(study, row["item_id"], row["rendering"])
               for row in data["metrics"][FIRST_DRAW_RATE]["items"]]
    changed = copy.deepcopy(data)
    changed["metrics"][FIRST_DRAW_RATE]["items"][0]["first_draw_rank"] = (first["first_draw_rank"] + 0.25) % 1.0
    with pytest.raises(FigureError, match="does not match"):
        draws_figure(PLAN, changed, results)
    with pytest.raises(FigureError, match="no stored result"):
        draws_figure(PLAN, data, results[1:])
    other = copy.deepcopy(data)
    other["plan_sha256"] = "0" * 64
    with pytest.raises(FigureError, match="this plan"):
        draws_figure(PLAN, other, results)
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({**data, "arm": "serving"}), encoding="utf-8")
    with pytest.raises(FigureError, match="not sampling/full_run"):
        read_ensemble_metrics(path, PLAN, arm="sampling", stage="full_run")
    path.write_text('{"schema": 1, "schema": 2}', encoding="utf-8")
    with pytest.raises(FigureError, match="duplicate"):
        read_ensemble_metrics(path, PLAN, arm="sampling", stage="full_run")
    with pytest.raises(FigureError, match="analysed stage"):
        figure_data(study, stage="pilot")
    empty = tmp_path / "results"
    empty.mkdir()
    with pytest.raises(FigureError, match="no stored result"):
        draws_figure_from_records(PLAN, analysis_dir=study.analysis_dir, results_dir=empty)


def test_both_figures_are_drawn_in_the_academic_style_with_both_axes_labelled(sampled, serving, study):
    negative = figure_data(study, stage="negative_control")
    for data in (sampled, serving, negative):
        figure = plot_draws(data)
        assert_academic_figure(figure, panels=1)
        (axes,) = figure.axes
        assert axes.get_xlabel() == "prompt version"
        assert axes.get_ylabel() == "share of items whose first draw\ndiffers from the ensemble (share)"
        assert [label.get_text() for label in axes.get_xticklabels()] == [
            f"{'all items' if rate.group == 'all' else f'{rate.group} prompt'}\n(n = {rate.n})"
            for rate in (data.rates if len(data.renderings) > 1 else data.rates[:1])]
        assert_academic_svg(figure_svg(figure))
        spread = plot_draw_spread(data)
        assert_academic_figure(spread, panels=1)
        (axes,) = spread.axes
        assert axes.get_xlabel() == "decision rule"
        assert axes.get_ylabel() == ("outlook (score)" if data.kind == "edgar" else
                                     "defensive spread\n(difference of loadings)")
        names = {"none": "unadjusted", "top_k": "adjusted with\nlogprobs", "full_distribution": "adjusted with\nlogits"}
        assert [label.get_text() for label in axes.get_xticklabels()] == [
            f"single draws\n(n = {data.example.parsed})",
            *(f"ensemble,\n{names[mark.adjustment]}" for mark in data.example.summary_marks if mark.value is not None)]
        assert_academic_svg(figure_svg(spread))


def test_each_figure_carries_one_message_with_two_series_named_in_one_legend(sampled, serving):
    """Requirement 8.2: the first-draw figure shows the rate against its baseline and nothing else; the spread
    figure shows the draws against the ensemble decisions and nothing else."""
    for data in (sampled, serving):
        figure = plot_draws(data)
        for rate in data.rates:
            observed, expected = (by_gid(figure, f"draws-{name}-{rate.group}") for name in ("rate", "baseline"))
            assert observed.get_marker() == series_style(0)["marker"]
            assert expected.get_marker() == series_style(1)["marker"] != observed.get_marker()
            assert observed.get_xdata()[0] < expected.get_xdata()[0]  # side by side in the group's place
        assert {gid.rsplit("-", 1)[0] for gid in gids(figure)} == {"draws-rate", "draws-rate-interval",
                                                                    "draws-baseline"}
        assert legend_labels(figure) == ["observed", "expected if draws were exchangeable"]
        spread = plot_draw_spread(data)
        assert legend_labels(spread) == ["single draw", "ensemble decision"]
        for drawn in (figure, spread):
            for artist in drawn.findobj(Line2D):
                assert is_grey(artist.get_color()) and is_grey(artist.get_markeredgecolor()), artist.get_gid()
                assert is_grey(artist.get_markerfacecolor()), artist.get_gid()


def test_the_counts_of_items_at_each_mid_rank_stay_in_the_table_and_are_not_drawn(sampled, serving):
    for data in (sampled, serving):
        for rendering in data.renderings:
            ranks = [item.first_draw_rank for item in data.items
                     if item.rendering == rendering and item.first_draw_rank is not None]
            rows = [row for row in csv_rows(draws_table(data).to_csv())
                    if row["row"] == "rank_count" and row["rendering"] == rendering]
            assert [(row["first_draw_rank"], row["n"]) for row in rows] == [
                (figures._number(rank), str(ranks.count(rank))) for rank in sorted(set(ranks))]
        assert not [gid for gid in gids(plot_draws(data)) if "rank" in gid or "band" in gid or "box" in gid]
        overall = data.rates[0]
        rate = next(row for row in csv_rows(draws_table(data).to_csv()) if row["row"] == "rate"
                    and row["group"] == "all")
        assert (rate["count_outside"], rate["count_differs"]) == (str(overall.outside_band),
                                                                  str(overall.category_differs))


# ---------- caption, table and outputs ----------

def test_the_first_draw_caption_is_a_journal_caption_with_the_rate_the_sample_size_and_the_interval(sampled, serving):
    for data, arm in ((sampled, "sampling"), (serving, "serving")):
        table = draws_table(data)
        caption = table.caption
        assert_journal_caption(caption, 3)
        overall = data.rates[0]
        shares = [figures.rounded(value) for value in (overall.rate, overall.low, overall.high,
                                                       overall.baseline_off)]
        assert (f"in {shares[0]} of the items ({shares[1]} to {shares[2]}), against {shares[3]} expected if the "
                "first draw were exchangeable") in caption
        assert (f" n = {len(data.items)} items with a decided first draw on {ROUTES[arm]} (identifying prompt: "
                f"{data.rates[1].n}; anonymised prompt: {data.rates[2].n}), {data.ensemble_draws} draws per ensemble; "
                f"{len(data.missing_items)} items without one are listed in the table. ") in caption
        assert ("Bars and the range in parentheses are 95 percent Wilson score intervals; the expected share has no "
                "interval.") in caption
        # When a first draw differs is a definition: one note below the table, with the stored range.
        lower, upper = data.band
        assert (f"A first draw differs when its decision category is not the ensemble's or its mid-rank among the "
                f"ensemble's draws lies outside {lower:g} to {upper:g}.") in table.notes
        assert "mid-rank" not in caption and "outside" not in caption and len(caption) < 560
        assert "(a)" not in caption  # one panel
        # Every number of the caption is in the table: the rate row, its counts, and the rows of the missing items;
        # the confidence and the draws per ensemble are the stored metrics' own.
        rows = csv_rows(table.to_csv())
        assert len([row for row in rows if row["row"] == "missing_item"]) == len(data.missing_items)
        assert_caption_numbers(table, stored=(100 * data.confidence, data.ensemble_draws, len(data.missing_items)))
        markdown = table.to_markdown()
        assert "Moody's" in markdown and "BAA10Y" in markdown and "Financial Modeling Prep (FMP)" in markdown
        assert_notes(table)
    # The interval words follow the method the metrics store, so the caption cannot name another interval.
    assert sampled.interval_method == WILSON_METHOD
    with pytest.raises(FigureError, match="no wording for the interval method"):
        draws_table(replace(sampled, interval_method="another interval"))
    assert "at confidence" not in draws_table(sampled).caption


def test_the_draw_spread_caption_is_a_journal_caption_with_the_range_and_the_ensemble_value(sampled, serving):
    for data, arm in ((sampled, "sampling"), (serving, "serving")):
        table = draw_spread_table(data)
        caption = table.caption
        assert_journal_caption(caption, 8)
        example = data.example
        values = [value for _, value in example.summary.present]
        signed = [figures.rounded(value, signed=True) for value in (
            min(values), max(values), example.summary.median, example.summary_marks[0].value)]
        if min(values) == max(values):
            lead = (f"Repeated answers to one prompt do not vary in this example: all {example.parsed} parsed draws "
                    f"equal {signed[0]} and")
        else:
            lead = (f"Repeated answers to one prompt spread around the ensemble decision: the {example.parsed} parsed "
                    f"draws of the example range from {signed[0]} to {signed[1]} (median {signed[2]}) and")
        assert f"Figure 8. {lead} the unadjusted ensemble gives {signed[3]}. " in caption
        assert (f" n = {example.draws} draws of one prompt ({example.item_id}, {example.rendering} prompt) on "
                f"{ROUTES[arm]}, {example.parsed} parsed and {example.failed} failed. ") in caption
        assert caption.endswith("No interval is drawn: every parsed draw is a point, tied draws side by side.")
        # What the value is (the y axis names it) and how the example was picked are notes, stated once.
        assert "The value is the defensive spread, the gold and cash loadings minus the equity and technology " \
               "loadings." in table.notes
        assert "loadings" not in caption and len(caption) < 500
        assert table.to_markdown().count("gold and cash loadings") == 1
        # Every number of the caption is a value of the table; the failed draws are its rows without a value.
        rows = [row for row in csv_rows(table.to_csv()) if row["row"] == "draw"]
        assert len(rows) == example.draws and sum(row["value"] == "" for row in rows) == example.failed
        assert_caption_numbers(table, stored=(example.draws, example.failed))
        assert_notes(table)
    # Draws that differ are a range; identical draws are said to be identical, with their count: one of each here.
    assert "range from" in draw_spread_table(sampled).caption
    assert "do not vary in this example: all 32 parsed draws equal" in draw_spread_table(serving).caption
    assert "spread around" not in draw_spread_table(serving).caption


def test_the_markdown_is_the_caption_the_csv_as_a_table_then_the_notes_the_sources_and_the_derived_only_line(sampled):
    for table in (draws_table(sampled), draw_spread_table(sampled)):
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


def test_both_figures_are_written_as_one_svg_with_a_csv_and_markdown_once(study, tmp_path):
    assert FIGURES_DIR == ROOT / "docs" / "assets" / "finance-sim"
    out = tmp_path / "assets"
    written = render_draws(PLAN, analysis_dir=study.analysis_dir, results_dir=study.results_dir, out_dir=out)
    assert list(written) == [*figure_paths("draws", out).values(), *figure_paths("draw-spread", out).values()]
    assert sorted(path.name for path in out.iterdir()) == WRITTEN
    for name, number in (("draws", 3), ("draw-spread", 8)):
        paths = figure_paths(name, out)
        assert_academic_svg(paths["svg"].read_text(encoding="utf-8"))
        assert paths["markdown"].read_text(encoding="utf-8").startswith(f"Figure {number}. ")
    before = {path: path.read_bytes() for path in written}
    assert render_draws(PLAN, analysis_dir=study.analysis_dir, results_dir=study.results_dir, out_dir=out) == written
    assert {path: path.read_bytes() for path in written} == before  # deterministic; the same output is a no-op
    first = stored(study)["metrics"][FIRST_DRAW_RATE]["items"][1]
    with pytest.raises(FigureError, match="refusing to replace"):
        render_draws(PLAN, analysis_dir=study.analysis_dir, results_dir=study.results_dir, out_dir=out,
                     example=(first["item_id"], first["rendering"]))
    assert {path: path.read_bytes() for path in written} == before
    data = figure_data(study)
    assert figure_svg(plot_draws(data)) == figure_svg(plot_draws(data))
    assert figure_svg(plot_draw_spread(data)) == figure_svg(plot_draw_spread(data))


#: Renders a pickled figure's data in a fresh process and prints the SHA-256 of its SVG.
RENDER_IN_PROCESS = (
    "import hashlib, pickle, sys\n"
    "import matplotlib\n"
    "matplotlib.use('Agg')\n"
    "from scripts.finance_sim.figures import figure_svg\n"
    "from scripts.finance_sim.{module} import {function}\n"
    "with open(sys.argv[1], 'rb') as stream:\n"
    "    data = pickle.load(stream)\n"
    "print(hashlib.sha256(figure_svg({function}(data)).encode('utf-8')).hexdigest())\n")


def svg_hashes_in_processes(data, module: str, function: str, path: Path, runs: int) -> list[list[str]]:
    import pickle

    with path.open("wb") as stream:
        pickle.dump(data, stream)
    code = RENDER_IN_PROCESS.format(module=module, function=function)
    outputs = []
    for _ in range(runs):
        result = subprocess.run([sys.executable, "-c", code, str(path)], capture_output=True, text=True, cwd=ROOT)
        assert result.returncode == 0, result.stderr
        outputs.append(result.stdout.split())
    return outputs


def test_the_svg_bytes_do_not_depend_on_the_process_that_renders_them(study, tmp_path):
    """Two fresh processes render the same bytes: the constrained layout's last bits vary between processes and the
    frozen layout makes them identical (the render under this process's rcParams is checked below)."""
    data = figure_data(study, stage="negative_control")
    for function in ("plot_draws", "plot_draw_spread"):
        first, second = svg_hashes_in_processes(data, "figures_draws", function, tmp_path / f"{function}.pickle",
                                                runs=2)
        assert first == second and len(first) == 1 and len(first[0]) == 64, (function, first, second)
    from matplotlib.layout_engine import ConstrainedLayoutEngine, PlaceHolderLayoutEngine

    figure = plot_draws(data)
    assert isinstance(figure.get_layout_engine(), ConstrainedLayoutEngine)
    figure_svg(figure)  # freezes the layout: positions rounded, the layout engine switched off
    assert isinstance(figure.get_layout_engine(), PlaceHolderLayoutEngine)
    # The corner is stored as rounded; the width and height are stored as the rounded corner plus the rounded size.
    assert all(round(value, 9) == value for axes in figure.axes for value in axes.get_position().p0)


def test_the_render_does_not_depend_on_the_processs_global_rc_params(sampled, serving, study):
    """The paper style's pdf renderer and tight bounding box, or another font, dpi or marker default left in the
    process by an earlier test, change nothing: the figure renders under matplotlib's defaults plus the academic
    style."""
    import hashlib

    negative = figure_data(study, stage="negative_control")

    def hashes() -> list[str]:
        return [hashlib.sha256(figure_svg(plot(data)).encode("utf-8")).hexdigest()
                for data in (sampled, serving, negative) for plot in (plot_draws, plot_draw_spread)]

    clean = hashes()
    with foreign_rc_params():
        assert hashes() == clean


def test_published_tables_and_svgs_carry_derived_statistics_only(study, tmp_path):
    out = tmp_path / "assets"
    texts = {path: path.read_text(encoding="utf-8") for path in
             render_draws(PLAN, analysis_dir=study.analysis_dir, results_dir=study.results_dir, out_dir=out)}
    returns = [value for month in MONTHS for value in (realized(month).world_equity, realized(month).tech_sector,
                                                       realized(month).gold_commodity)]
    prompts = [render(macro_row(month))[:60] for month in MONTHS[:6] for render in (render_identifying,
                                                                                     render_anonymised)]
    raw = {repr(value) for month in MONTHS for value in macro_row(month).raw.values() if len(repr(value)) > 8}
    header = csv_rows(texts[out / "draws.csv"])[0]
    assert not {"correct", "outcome", "realized", "response", "response_text"} & set(header)
    for path, text in texts.items():
        assert "response_text" not in text, path
        assert '"response"' not in text and '"outcome"' not in text, path
        for value in returns:
            assert repr(value) not in text, path
        for prompt in prompts:
            assert prompt not in text, path
        for value in raw:
            assert value not in text, path


def test_the_module_imports_without_matplotlib():
    code = ("import sys\n"
            "import scripts.finance_sim.figures_draws\n"
            "import scripts.finance_sim.figures_convergence\n"
            "heavy = sorted(m for m in sys.modules if m.split('.')[0] == 'matplotlib')\n"
            "assert not heavy, heavy\n"
            "print('lean-ok')\n")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stderr
    assert "lean-ok" in result.stdout


def test_the_command_renders_from_the_stored_outputs_and_refuses_another_plan(study, tmp_path):
    plan_path = write_plan(PLAN, tmp_path / "plan.json")
    out = tmp_path / "assets"
    common = ["--plan", str(plan_path), "--analysis-dir", str(study.analysis_dir), "--results-dir",
              str(study.results_dir), "--out-dir", str(out)]
    assert main([*common, "--plan-sha256", PLAN.plan_sha256]) == 0
    assert sorted(path.name for path in out.iterdir()) == WRITTEN
    assert main([*common, "--plan-sha256", PLAN.plan_sha256]) == 0  # the same output again is a no-op
    assert main([*common, "--plan-sha256", "0" * 64]) == 2
    assert main([*common, "--plan-sha256", PLAN.plan_sha256, "--example", "macro-2016-05"]) == 2
    other = tmp_path / "serving"
    assert main([*common[:-1], str(other), "--plan-sha256", PLAN.plan_sha256, "--arm", "serving"]) == 0
    assert sorted(path.name for path in other.iterdir()) == WRITTEN
