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
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402
from matplotlib.collections import PathCollection  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.text import Annotation, Text  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

from recall_guard.core.distribution import summarize_distribution  # noqa: E402
from recall_guard.dataset.edgar_outlook import render_outlook_prompt  # noqa: E402
from recall_guard.dataset.macro_regime import render_anonymised, render_identifying  # noqa: E402
from recall_guard.harness.plots import (  # noqa: E402
    ALL_PAIRS_SLOT_CAP,
    STUDY_PALETTE,
    STUDY_PALETTE_DARK,
    THEMES,
    figure_theme,
    variant_style,
)
from scripts.finance_sim import ensemble_metrics as metrics_module  # noqa: E402
from scripts.finance_sim import figures  # noqa: E402
from scripts.finance_sim.calibrate import DrawSettings, FitSettings, run_calibration  # noqa: E402
from scripts.finance_sim.ensemble_metrics import (  # noqa: E402
    FIRST_DRAW_RATE,
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
from scripts.finance_sim.figures import FIGURES_DIR, FigureError, figure_paths, figure_svg  # noqa: E402
from scripts.finance_sim.figures_draws import (  # noqa: E402
    EXAMPLE_DEFAULT,
    SPLIT_CHECK_MIN_DRAWS,
    draws_figure,
    draws_figure_from_records,
    draws_table,
    exchangeability_shares,
    main,
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
from tests.scripts.finance_sim.test_figures import foreign_rc_params  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
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


def test_the_first_draw_is_marked_with_a_distinct_marker_and_label(sampled):
    strips = {"summary": sampled.example.summary,
              **{f"estimate-{strip.adjustment}": strip for strip in sampled.example.estimates}}
    assert set(strips) == {"summary", "estimate-top_k", "estimate-full_distribution"}
    for theme in THEMES:
        figure = plot_draws(sampled, theme=theme)
        style = figure_theme(theme)
        for key, strip in strips.items():
            first = by_gid(figure, f"draws-first-{key}")
            assert first.get_marker() == "D" and to_hex(first.get_markerfacecolor()) == to_hex(style.ink)
            assert list(first.get_xdata()) == [strip.first]
            assert by_gid(figure, f"draws-first-label-{key}").get_text() == "first draw"
            assert drawn_count(figure, key) == sum(1 for value in strip.values[1:] if value is not None)


def drawn_count(figure, key: str) -> int:
    """Draws other than the first in a strip: single open markers plus the counts printed beside stacks."""
    present = gids(figure)
    singles = len(by_gid(figure, f"draws-draw-{key}").get_xdata()) if f"draws-draw-{key}" in present else 0
    counts = [int(text.get_text().removeprefix("×")) for text in figure.findobj(Annotation)
              if (text.get_gid() or "").startswith(f"draws-stack-label-{key}-")]
    stacks = len(by_gid(figure, f"draws-stack-{key}").get_xdata()) if f"draws-stack-{key}" in present else 0
    assert stacks == len(counts)
    return singles + sum(counts)


def test_open_draws_stay_distinguishable_and_large_ties_collapse_into_one_counted_marker(sampled, serving, study):
    negative = figure_data(study, stage="negative_control")
    for data in (sampled, serving, negative):
        for theme in THEMES:
            figure = plot_draws(data, theme=theme)
            with matplotlib.rc_context(figures._theme_rc(theme)):  # measured under the figure's own rcParams
                figure.draw_without_rendering()
            scale = figure.dpi / 72.0
            for axes in figure.axes[:2]:
                boxes = []
                for line in axes.lines:
                    if not (line.get_gid() or "").startswith(("draws-draw-", "draws-stack-", "draws-first-")):
                        continue
                    half = (line.get_markersize() + line.get_markeredgewidth()) / 2.0 * scale - 0.3 * scale
                    boxes += [(line.get_gid(), Bbox.from_extents(x - half, y - half, x + half, y + half))
                              for x, y in line.get_transform().transform(line.get_xydata())]
                for index, (gid, box) in enumerate(boxes):
                    for other_gid, other in boxes[index + 1:]:
                        assert not box.overlaps(other), (theme, gid, other_gid)
    # The serving item's 32 identical draws: one open marker counting the other 31, the first draw apart above it.
    figure = plot_draws(serving, theme="light")
    assert len(set(serving.example.summary.values)) == 1 and serving.example.draws == 32
    stack = by_gid(figure, "draws-stack-summary")
    assert list(stack.get_xdata()) == [serving.example.summary.values[1]]
    assert [text.get_text() for text in figure.findobj(Annotation)
            if (text.get_gid() or "").startswith("draws-stack-label-summary-")] == ["×31"]
    assert by_gid(figure, "draws-first-summary").get_ydata()[0] < stack.get_ydata()[0]  # above (rows run down)
    assert "draws-draw-summary" not in gids(figure)
    assert "×8 beside it, that many draws too close together to stack apart" in draws_table(serving).caption


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
    assert sampled.example.chosen == EXAMPLE_DEFAULT and EXAMPLE_DEFAULT in draws_table(sampled).caption
    chosen = figure_data(study, example=(analysed["item_id"], analysed["rendering"]))
    assert chosen.example.summary.first is None and "chosen for the figure" in draws_table(chosen).caption
    assert "did not parse" in plot_draws(chosen).axes[0].get_title(loc="left")
    assert "draws-first-summary" not in gids(plot_draws(chosen))
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
    caption = draws_table(sampled).caption
    assert "under the same inclusion rule as the stored rate" in caption
    assert f"Items without a consensus category: {count} of 26 (included only when outside the band)." in caption
    title = plot_draws(sampled).axes[3].get_title(loc="left").replace("\n", " ")
    assert f"{count} of 26 items have no consensus category: included only when outside the band" in title
    row = next(row for row in csv_rows(draws_table(sampled).to_csv()) if row["row"] == "rate" and row["group"] == "all")
    assert row["count_no_category"] == str(count)


def _parsed(study, item) -> tuple[list[float], list]:
    result = stored_result(study, item.item_id, item.rendering)
    pairs = [(value, category) for value, category in zip(result.summary_values, result.categories, strict=True)
             if value is not None]
    return [value for value, _ in pairs], [category for _, category in pairs]


def test_an_unresolved_example_ensemble_is_reported_as_unresolved_not_with_the_estimate_reason(study):
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
    caption = draws_table(data).caption
    assert f"Not drawn: ensemble_none: {UNRESOLVED_SUMMARY}" in caption
    if consensus.reason:
        assert f"ensemble_none: {consensus.reason}" not in caption  # not the guard's estimate reason


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
    figure = plot_draws(sampled, theme="light")
    tick = by_gid(figure, "draws-baseline-all")
    assert set(tick.get_xdata()) == {overall.baseline_off}
    assert "exchangeability baseline" in draws_table(sampled).caption


def test_items_whose_halves_7_11_lists_as_unresolved_sit_in_a_row_of_their_own(study, serving):
    cell = stored(study, "serving")["metrics"]["half_ensemble_variance_summary"]["results"]["none"]
    unresolved = {(row["item_id"], row["rendering"]) for row in cell["missing_items"]
                  if UNRESOLVED_SUMMARY in row["reason"]}
    assert len(unresolved) == 2
    marked = {(item.item_id, item.rendering) for item in serving.items if item.halves_unresolved}
    assert marked == unresolved
    for item in serving.items:
        if not item.halves_unresolved:
            assert item.halves_unresolved is None  # never shown as resolved
    figure = plot_draws(serving, theme="light")
    labels = [label.get_text() for label in figure.axes[2].get_yticklabels()]
    renderings = {rendering for _, rendering in unresolved}
    assert all(f"{rendering}, halves unresolved" in labels for rendering in renderings)
    rows = [row for row in csv_rows(draws_table(serving).to_csv()) if row["row"] == "item"]
    assert {(row["item_id"], row["rendering"]) for row in rows if UNRESOLVED_SUMMARY in row["halves"]} == unresolved
    caption = draws_table(serving).caption
    assert "(missing from the half-ensemble variance with that reason) sit in a row of their own: 2 items." in caption
    assert "An unmarked item is not shown as resolved" in caption
    assert not any("resolved" == row["halves"] for row in rows)


def test_the_probe_arms_four_draw_halves_are_stated_as_never_assessed(sampled):
    assert sampled.half_draws == 4 < SPLIT_CHECK_MIN_DRAWS == 8
    assert not any(item.halves_unresolved for item in sampled.items)
    caption = draws_table(sampled).caption
    assert (f"needs at least {SPLIT_CHECK_MIN_DRAWS} parsed draws, so halves of 4 draws are never assessed; an "
            "unmarked item is not shown as resolved") in caption


def test_the_negative_control_stage_draws_the_outlook(study):
    data = figure_data(study, stage="negative_control")
    assert data.kind == "edgar" and data.renderings == ("outlook",) and data.example.rendering == "outlook"
    assert [rate.group for rate in data.rates] == ["all", "outlook"]
    caption = draws_table(data).caption
    assert "SEC EDGAR" in caption and "Moody's" not in caption
    figure = plot_draws(data, theme="dark")
    assert figure.axes[0].get_xlabel() == "Outlook"


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


# ---------- encoding and labels ----------

def test_identity_never_depends_on_colour_alone_and_reference_lines_are_muted(sampled, serving):
    for data in (sampled, serving):
        for theme in THEMES:
            style = figure_theme(theme)
            figure = plot_draws(data, theme=theme)
            for key, adjustment in (("summary", "none"), ("estimate-top_k", "top_k"),
                                    ("estimate-full_distribution", "full_distribution")):
                for prefix in ("draws-draw-", "draws-stack-"):
                    if f"{prefix}{key}" not in gids(figure):
                        continue
                    expected = variant_style(adjustment, "single", theme=theme)
                    line = by_gid(figure, f"{prefix}{key}")
                    assert line.get_marker() == expected.marker
                    assert to_hex(line.get_markerfacecolor()) == to_hex(style.surface)  # open: draws
            for mark in (*data.example.summary_marks, *data.example.estimate_marks):
                gid = (f"draws-ensemble-summary-{mark.variant}" if mark.quantity == "summary value"
                       else f"draws-ensemble-estimate-{mark.variant}")
                if mark.value is None:
                    assert gid not in gids(figure)
                    continue
                expected = variant_style(mark.adjustment, "ensemble", theme=theme)
                line = by_gid(figure, gid)
                assert line.get_marker() == expected.marker
                assert to_hex(line.get_markerfacecolor()) == to_hex(expected.color)  # filled: an ensemble
            for gid in ("draws-zero", "draws-band-lower", "draws-band-upper", "draws-baseline-all"):
                assert to_hex(by_gid(figure, gid).get_color()) == to_hex(style.muted), gid
            palette = {to_hex(colour) for colour in (STUDY_PALETTE if theme == "light" else STUDY_PALETTE_DARK)}
            colours = {to_hex(line.get_markeredgecolor()) for line in figure.findobj(Line2D)
                       if (line.get_gid() or "").startswith(("draws-draw-", "draws-stack-", "draws-ensemble-"))}
            colours |= {to_hex(collection.get_edgecolor()[0]) for collection in figure.findobj(PathCollection)
                        if (collection.get_gid() or "").startswith("draws-rank-")}
            assert colours <= palette and len(colours) <= ALL_PAIRS_SLOT_CAP
            ranks = [collection for collection in figure.findobj(PathCollection)
                     if (collection.get_gid() or "").startswith("draws-rank-")]
            assert ranks
            for collection in ranks:  # filled exactly when outside the band; a diamond when the category differs
                rank = float(collection.get_offsets()[0][0])
                lower, upper = data.band
                filled = to_hex(collection.get_facecolor()[0]) != to_hex(style.surface)
                assert filled == (not lower <= rank <= upper)
                assert collection.get_gid().endswith("differs") == (len(collection.get_paths()[0].vertices) == 5)
            tokens = {to_hex(style.ink), to_hex(style.secondary_ink)}
            for text in figure.findobj(Text):
                if text.get_text().strip() and text.get_visible():
                    assert to_hex(text.get_color()) in tokens, text.get_text()


def _marker_boxes(axes, scale: float) -> list[Bbox]:
    boxes = []
    for line in axes.lines:
        if line.get_marker() in (None, "None", "none", ""):
            continue
        half = (line.get_markersize() + line.get_markeredgewidth()) / 2.0 * scale
        boxes += [Bbox.from_extents(x - half, y - half, x + half, y + half)
                  for x, y in line.get_transform().transform(line.get_xydata())]
    for collection in axes.collections:
        if isinstance(collection, PathCollection):
            half = (float(collection.get_sizes()[0]) ** 0.5 / 2.0 + 0.75) * scale
            boxes += [Bbox.from_extents(x - half, y - half, x + half, y + half)
                      for x, y in collection.get_offset_transform().transform(collection.get_offsets())]
    return boxes


def test_labels_stay_clear_of_lines_markers_each_other_and_the_panel_frame(sampled, serving):
    for data in (sampled, serving):
        for theme in THEMES:
            figure = plot_draws(data, theme=theme)
            with matplotlib.rc_context(figures._theme_rc(theme)):  # measured under the figure's own rcParams
                figure.draw_without_rendering()
            scale = figure.dpi / 72.0
            for axes in figure.axes:
                labels = [text for text in axes.texts if isinstance(text, Annotation)]
                lines = [line.get_path().transformed(line.get_transform()) for line in axes.lines
                         if line.get_linestyle() not in ("None", "none", "")]
                markers = _marker_boxes(axes, scale)
                extents = [label.get_window_extent() for label in labels]
                frame = axes.bbox
                for index, (label, extent) in enumerate(zip(labels, extents, strict=True)):
                    where = (theme, label.get_gid(), label.get_text())
                    assert (frame.x0 <= extent.x0 and extent.x1 <= frame.x1 and frame.y0 <= extent.y0
                            and extent.y1 <= frame.y1), where
                    assert not any(extent.overlaps(box) for box in markers), where
                    assert not any(extent.overlaps(other) for other in extents[index + 1:]), where
                    # Every label that is not the by-design fallback keeps 3 pt from every line; whether a fallback
                    # is needed can depend on the platform's font metrics, and a fallback's box masks the line.
                    if not figures.placement_fallback(label):
                        assert not any(path.intersects_bbox(extent.padded(3.0 * scale), filled=False)
                                       for path in lines), where
                    patch = label.get_bbox_patch()
                    assert patch is not None and to_hex(patch.get_facecolor()) == to_hex(figure_theme(theme).surface)


def test_rank_count_labels_add_up_to_the_items_with_a_mid_rank(sampled, serving):
    for data in (sampled, serving):
        figure = plot_draws(data, theme="light")
        counts = [int(text.get_text()) for text in figure.axes[2].texts if isinstance(text, Annotation)]
        assert sum(counts) == sum(1 for item in data.items if item.first_draw_rank is not None)


# ---------- caption, table and outputs ----------

def test_the_caption_states_data_sample_size_interval_method_and_sources(sampled):
    table = draws_table(sampled)
    caption = table.caption
    assert caption.startswith("Figure F3.")
    assert "stored item results (task 7.7)" in caption and "ensemble metrics of task 7.11 (read, not recomputed)" in (
        caption)
    assert "google/gemma-3-4b-it" in caption and PLAN.plan_sha256[:12] in caption
    example = sampled.example
    assert (f"the example item has {example.draws} draws, {example.parsed} parsed and {example.failed} failed; "
            f"{len(sampled.items)} items have a decided first draw") in caption
    assert "interquartile range of the parsed draws (25th to 75th percentile, linear interpolation" in caption
    assert "Wilson score interval, two-sided, at confidence 1 - alpha at confidence 0.95" in caption
    assert "band 0.25 to 0.75 (bounds inclusive)" in caption
    assert "Moody's" in caption and "BAA10Y" in caption and "Financial Modeling Prep (FMP)" in caption
    assert figures.DERIVED_ONLY in caption and "pilot items excluded" in caption


def test_the_markdown_table_is_the_csv_rendered_below_its_caption(sampled):
    table = draws_table(sampled)
    lines = table.to_markdown().splitlines()
    assert lines[0] == table.caption and lines[1] == ""
    rows = list(csv.reader(io.StringIO(table.to_csv())))
    assert lines[3] == "|" + "|".join("---" for _ in rows[0]) + "|"
    rendered = [[value.strip().replace("\\|", "|") for value in line.strip("|").split(" | ")]
                for line in [lines[2], *lines[4:]]]
    assert rendered == rows


def test_the_figure_is_written_in_light_and_dark_svg_with_a_csv_and_markdown_once(study, tmp_path):
    assert FIGURES_DIR == ROOT / "docs" / "assets" / "finance-sim"
    out = tmp_path / "assets"
    written = render_draws(PLAN, analysis_dir=study.analysis_dir, results_dir=study.results_dir, out_dir=out)
    paths = figure_paths("draws", out)
    assert list(written) == list(paths.values())
    assert sorted(path.name for path in out.iterdir()) == ["draws-dark.svg", "draws-light.svg", "draws.csv",
                                                           "draws.md"]
    light, dark = paths["light"].read_text(encoding="utf-8"), paths["dark"].read_text(encoding="utf-8")
    assert light.lstrip().startswith("<?xml") and light != dark and "<dc:date>" not in light
    assert figure_theme("light").surface in light and figure_theme("dark").surface in dark
    before = {path: path.read_bytes() for path in written}
    assert render_draws(PLAN, analysis_dir=study.analysis_dir, results_dir=study.results_dir, out_dir=out) == written
    assert {path: path.read_bytes() for path in written} == before  # deterministic; the same output is a no-op
    first = stored(study)["metrics"][FIRST_DRAW_RATE]["items"][1]
    with pytest.raises(FigureError, match="refusing to replace"):
        render_draws(PLAN, analysis_dir=study.analysis_dir, results_dir=study.results_dir, out_dir=out,
                     example=(first["item_id"], first["rendering"]))
    assert {path: path.read_bytes() for path in written} == before
    data = figure_data(study)
    assert figure_svg(plot_draws(data, theme="light")) == figure_svg(plot_draws(data, theme="light"))


#: Renders a pickled figure's data in a fresh process and prints the SHA-256 of its light and dark SVG.
RENDER_IN_PROCESS = (
    "import hashlib, pickle, sys\n"
    "import matplotlib\n"
    "matplotlib.use('Agg')\n"
    "from scripts.finance_sim.figures import figure_svg\n"
    "from scripts.finance_sim.{module} import {function}\n"
    "with open(sys.argv[1], 'rb') as stream:\n"
    "    data = pickle.load(stream)\n"
    "for theme in ('light', 'dark'):\n"
    "    print(hashlib.sha256(figure_svg({function}(data, theme=theme)).encode('utf-8')).hexdigest())\n")


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
    first, second = svg_hashes_in_processes(data, "figures_draws", "plot_draws", tmp_path / "draws.pickle", runs=2)
    assert first == second and len(first) == 2 and first[0] != first[1], (first, second)
    from matplotlib.layout_engine import ConstrainedLayoutEngine, PlaceHolderLayoutEngine

    figure = plot_draws(data, theme="light")
    assert isinstance(figure.get_layout_engine(), ConstrainedLayoutEngine)
    figure_svg(figure)  # freezes the layout: positions rounded, the layout engine switched off
    assert isinstance(figure.get_layout_engine(), PlaceHolderLayoutEngine)
    # The corner is stored as rounded; the width and height are stored as the rounded corner plus the rounded size.
    assert all(round(value, 9) == value for axes in figure.axes for value in axes.get_position().p0)


def test_the_render_does_not_depend_on_the_processs_global_rc_params(sampled, serving, study):
    """The paper style's pdf renderer and tight bounding box, or another font, dpi or marker default left in the
    process by an earlier test, change nothing: the figure renders under matplotlib's defaults plus the theme."""
    import hashlib

    negative = figure_data(study, stage="negative_control")

    def hashes() -> list[str]:
        return [hashlib.sha256(figure_svg(plot_draws(data, theme=theme)).encode("utf-8")).hexdigest()
                for data in (sampled, serving, negative) for theme in THEMES]

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
    assert sorted(path.name for path in out.iterdir()) == ["draws-dark.svg", "draws-light.svg", "draws.csv",
                                                           "draws.md"]
    assert main([*common, "--plan-sha256", PLAN.plan_sha256]) == 0  # the same output again is a no-op
    assert main([*common, "--plan-sha256", "0" * 64]) == 2
    assert main([*common, "--plan-sha256", PLAN.plan_sha256, "--example", "macro-2016-05"]) == 2
    other = tmp_path / "serving"
    assert main([*common[:-1], str(other), "--plan-sha256", PLAN.plan_sha256, "--arm", "serving"]) == 0
    assert (other / "draws-light.svg").exists()
