"""Convergence figure F4 (task 8.3; Requirements 10.5 and 10.6).

Fixtures only. The 7.12 fixture is rebuilt here: a small frozen plan is
calibrated and three evaluation months are executed once on the generation and
full-distribution doubles of the 7.12 tests, through the real 7.6 calibration
and 7.7 execution, with the draws in the 7.4 draw log and the evidence batches
in evidence files; the convergence table is then computed from those records
and written as JSON under ``tmp_path`` (the evidence keeps 20 alternatives, so
k 50 and 100 are missing). A second, synthetic negative-control table with the
same schema and weights is derived from it. The figure reads only the JSON
files. Sockets are blocked and nothing is written under ``runs/`` or ``docs/``.
"""

from __future__ import annotations

import copy
import csv
import io
import json
import socket
from dataclasses import replace
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.text import Annotation  # noqa: E402

from scripts.finance_sim import convergence as convergence_module  # noqa: E402
from scripts.finance_sim import figures  # noqa: E402
from scripts.finance_sim.calibrate import DrawSettings, EvidenceSettings, FitSettings, run_calibration  # noqa: E402
from scripts.finance_sim.convergence import CONVERGENCE_KS, convergence_table, table_path, write_table  # noqa: E402
from scripts.finance_sim.execute import (  # noqa: E402
    ExecutionDraw,
    execute_items,
    load_calibrators,
    macro_items,
    planned_keys,
)
from scripts.finance_sim.figure_style import series_style  # noqa: E402
from scripts.finance_sim.figures import FigureError, figure_paths, route_label  # noqa: E402
from scripts.finance_sim.figures_convergence import (  # noqa: E402
    FEATURE_NAMES,
    MASS,
    MAX_LINES,
    PLOTTED,
    convergence_stage,
    convergence_stages,
    main,
    plot_convergence,
    read_convergence_stage,
    render_convergence,
)
from scripts.finance_sim.figures_convergence import convergence_table as figure_table  # noqa: E402
from scripts.finance_sim.limits import BudgetLedger, LedgerCallBudget, StudyLimits, budget_path  # noqa: E402
from scripts.finance_sim.plan import SERVING_FULL_DISTRIBUTION_REASON, write_plan  # noqa: E402
from scripts.finance_sim.recording import DrawLog, draw_log_path  # noqa: E402
from tests.scripts.finance_sim.test_convergence import (  # noqa: E402
    ALLOWANCE,
    DIGEST,
    EXECUTED_MONTHS,
    KEEP,
    MACRO_ITEMS,
    PLAN,
    PROSE_CALL,
    SEED_BASE,
    TEMPERATURE,
    Clock,
    FakeFull,
    FakeHost,
    edgar_releases,
    macro_row,
    write_evidence_file,
)
from tests.scripts.finance_sim.test_figures import (  # noqa: E402
    assert_academic_figure,
    assert_academic_svg,
    assert_caption_numbers,
    assert_journal_caption,
    assert_notes,
    is_grey,
    legend_labels,
)

FEATURES = ("loss", "min_k", "min_k_pp", "zlib_ratio", "estimate")
STAGES = ("full_run", "negative_control")
ROUTE = route_label(PLAN, "sampling")
#: The statistic each plotted quantity joins by a line: the mean over draws and the worst draw.
STATISTICS = {"min_k_pp": ("mean_abs_diff", "max_abs_diff"), "estimate": ("mean_abs_diff", "max_abs_diff"),
              "mass": ("mean_mass", "min_mass")}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("the convergence figure attempted a network connection")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)


def synthetic_negative_control(data: dict) -> dict:
    """The full run's table relabelled as the negative control, its differences and masses moved a little."""
    other = copy.deepcopy(data)
    other.update(stage="negative_control", kind="edgar")
    for row in other["features"]:
        for key in ("mean_abs_diff", "max_abs_diff"):
            if row[key] is not None:
                row[key] *= 0.6
    for row in other["mass"]:
        for key in ("mean_mass", "min_mass"):
            if row[key] is not None:
                row[key] = min(1.0, row[key] * 1.05)
    return other


@pytest.fixture(scope="module")
def study(tmp_path_factory) -> SimpleNamespace:
    """The 7.12 records on the fakes, the full run's stored convergence table and a synthetic negative control."""
    root = tmp_path_factory.mktemp("figures-convergence")
    records_dir, results_dir, calibrators_dir = root / "records", root / "results", root / "calibrators"
    ledger = BudgetLedger.open(root / "calibration" / "budget.json", PLAN,
                               StudyLimits(PLAN.plan_sha256, "sampling", "calibration", 1000, 1_000_000, 1_000_000),
                               ALLOWANCE, clock=Clock())
    run_calibration(PLAN, arm="sampling", macro_rows=[macro_row(month) for month in PLAN.months.calibration],
                    edgar_releases=edgar_releases(), client=FakeHost(LedgerCallBudget(ledger)), ledger=ledger,
                    settings=DrawSettings(draws=2, temperature=TEMPERATURE, seed_base=SEED_BASE),
                    sources=("top_k", "full_distribution"), full_distribution=FakeFull(),
                    fit=FitSettings(min_auc=0.6, min_valid=10), evidence=EvidenceSettings(top_k_keep=KEEP),
                    calibrators_dir=calibrators_dir, sleep=lambda seconds: None)
    calibrators = load_calibrators(PLAN, arm="sampling", kind="macro", calibrators_dir=calibrators_dir)
    planned_ids = list(dict.fromkeys(item_id for item_id, _ in planned_keys(PLAN, "sampling", "full_run")))
    chosen = set(planned_ids[:EXECUTED_MONTHS])
    items = tuple(item for item in macro_items(PLAN, MACRO_ITEMS, stage="full_run") if item.item_id in chosen)
    budget = PLAN.limits.stage("sampling", "full_run")
    stage_ledger = BudgetLedger.open(
        budget_path("sampling", "full_run", records_dir), PLAN,
        StudyLimits(PLAN.plan_sha256, "sampling", "full_run", budget.max_requests, budget.max_output_tokens,
                    budget.max_seconds), ALLOWANCE, clock=Clock())
    log = DrawLog(draw_log_path("sampling", "full_run", records_dir), plan=PLAN, loads=ExecutionDraw.from_dict)
    evidence = []

    def on_evidence(item, found) -> None:
        evidence.extend(found.values())
        write_evidence_file(records_dir, item, found)

    run = execute_items(PLAN, items, arm="sampling", stage="full_run",
                        client=FakeHost(LedgerCallBudget(stage_ledger), prose=(PROSE_CALL,)), ledger=stage_ledger,
                        calibrators=calibrators, full_distribution=FakeFull(), on_draw=log, on_evidence=on_evidence,
                        results_dir=results_dir, sleep=lambda seconds: None)
    table = convergence_table(PLAN, tuple(run.results[item.key] for item in items), tuple(log.latest()), evidence,
                              arm="sampling", stage="full_run", calibrators=calibrators, expected_digest=DIGEST)
    analysis_dir = root / "analysis"
    write_table(table, analysis_dir)
    data = json.loads(table_path("sampling", "full_run", analysis_dir).read_text(encoding="utf-8"))
    table_path("sampling", "negative_control", analysis_dir).write_text(
        json.dumps(synthetic_negative_control(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    only = root / "only-full-run"
    write_table(table, only)
    return SimpleNamespace(root=root, analysis_dir=analysis_dir, only=only, data=data)


@pytest.fixture(scope="module")
def stages(study):
    return convergence_stages(PLAN, analysis_dir=study.analysis_dir)


def stored(study: SimpleNamespace, stage: str = "full_run") -> dict:
    return json.loads(table_path("sampling", stage, study.analysis_dir).read_text(encoding="utf-8"))


def csv_rows(text: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(text)))


def by_gid(figure, gid: str):
    return next(artist for artist in figure.findobj() if artist.get_gid() == gid)


def gids(figure) -> set[str]:
    return {artist.get_gid() for artist in figure.findobj() if artist.get_gid()}


def row(data: dict, evidence: str, k: int, feature: str) -> dict:
    return next(entry for entry in data["features"] if (entry["evidence"], entry["k"], entry["feature"]) == (
        evidence, k, feature))



def plotted(feature: str, k: int) -> bool:
    """Min-K%++ is undefined at k = 1, so that point is not plotted; every other stored point is."""
    return not (feature == "min_k_pp" and k == 1)


# ---------- read, never recomputed ----------

def test_every_number_is_read_from_the_stored_tables_and_nothing_is_recomputed(study, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("the figure recomputed the convergence analysis instead of reading it")

    for name in ("convergence_table", "convergence_from_records", "truncate_to_top_k", "features_from_full",
                 "features_from_top_k", "top_k_mass", "decide_from_draws"):
        monkeypatch.setattr(convergence_module, name, forbidden)
    found = convergence_stages(PLAN, analysis_dir=study.analysis_dir)
    assert [stage.stage for stage in found] == list(STAGES) and {stage.status for stage in found} == {"drawn"}
    for stage in found:
        data = stored(study, stage.stage)
        assert (stage.model, stage.model_digest, stage.kept, stage.n_vocab, stage.served_k) == (
            data["model"], data["model_digest"], data["kept"], data["n_vocab"], data["served_k"])
        assert [(entry.evidence, entry.k, entry.feature, entry.n, entry.mean_abs_diff, entry.max_abs_diff,
                 entry.share_within) for entry in stage.features] == [
            (entry["evidence"], entry["k"], entry["feature"], entry["n"], entry["mean_abs_diff"],
             entry["max_abs_diff"], entry["share_within"]) for entry in data["features"]]
        assert [(entry.evidence, entry.k, entry.mean_mass, entry.min_mass) for entry in stage.mass] == [
            (entry["evidence"], entry["k"], entry["mean_mass"], entry["min_mass"]) for entry in data["mass"]]
    figure = plot_convergence(found)
    assert PLOTTED == ("min_k_pp", "estimate", MASS)
    for feature in ("min_k_pp", "estimate"):
        line = by_gid(figure, f"convergence-{feature}-full_run-mean_abs_diff")
        drawn = [entry for entry in study.data["features"] if entry["evidence"] == "host_top_k"
                 and entry["feature"] == feature and entry["missing_reason"] is None and plotted(feature, entry["k"])]
        assert list(line.get_xdata()) == [entry["k"] for entry in drawn]
        assert list(line.get_ydata()) == [entry["mean_abs_diff"] for entry in drawn]
    mass = [entry for entry in study.data["mass"] if entry["evidence"] == "host_top_k"
            and entry["missing_reason"] is None]
    assert list(by_gid(figure, "convergence-mass-full_run-mean_mass").get_ydata()) == [
        entry["mean_mass"] for entry in mass]
    changed = copy.deepcopy(study.data)
    row(changed, "host_top_k", 1, "min_k_pp")["mean_abs_diff"] = 0.123456
    rows = csv_rows(figure_table([convergence_stage(changed)]).to_csv())
    assert next(entry for entry in rows if (entry["evidence"], entry["k"], entry["quantity"]) == (
        "host_top_k", "1", "min_k_pp"))["mean_abs_diff"] == "0.123456"


def test_k_above_the_kept_count_is_listed_in_the_table_not_drawn(study, stages):
    assert KEEP == 20 and CONVERGENCE_KS == (1, 5, 10, 20, 50, 100)
    figure = plot_convergence(stages)
    drawn = [line for line in figure.findobj(Line2D) if (line.get_gid() or "").startswith("convergence-")]
    assert len(drawn) == 3 * 6  # three panels, four lines and two markers of the served values each
    for line in drawn:
        assert max(line.get_xdata()) <= KEEP, line.get_gid()
    caption = figure_table(stages, PLAN).caption
    assert "By construction the differences are zero at k = 20 and, at every k, for " in caption
    assert "exceed" not in caption and "k = 50" not in caption and " 100" not in caption  # the table has those rows
    rows = csv_rows(figure_table(stages, PLAN).to_csv())
    above = [entry for entry in rows if entry["evidence"] == "host_top_k" and entry["k"] in ("50", "100")]
    assert above and {entry["drawn"] for entry in above} == {"no"}
    assert {(entry["k"], entry["missing"]) for entry in above} == {
        ("50", "k 50 is above the 20 alternatives kept"), ("100", "k 100 is above the 20 alternatives kept")}
    # A published table names the route: no arm and no role column.
    assert not {"arm", "role"} & set(rows[0]) and {entry["route"] for entry in rows} == {
        "llamacpp-gemma3-4b at temperature 0.7"}


def test_the_table_states_the_prompts_and_draws_the_caption_counts(stages):
    """The caption states how many prompts and draws were compared; a note under the table gives the same counts,
    so a reader can check the caption against the table."""
    table = figure_table(stages, PLAN)
    prompts = {"full_run": "macro prompts", "negative_control": "EDGAR releases"}
    counted = " and ".join(f"{stage.n_draws} draws of {stage.n_items} {prompts[stage.stage]}" for stage in stages)
    note = f"Compared: {counted}; n in a row is the number of draws compared at that k."
    assert note in table.notes and note in table.to_markdown()
    for stage in stages:
        assert f"{stage.n_draws} draws of {stage.n_items} " in table.caption


def test_the_served_values_are_one_marker_of_the_mean_at_the_served_k_never_joined_to_the_lines(study, stages):
    """The mean and the worst draw of the served values coincided as two identical markers; the figure plots the
    mean only, says so, and leaves the worst draw to the table."""
    served_k = stages[0].served_k
    assert served_k == 5
    figure = plot_convergence(stages)
    present = gids(figure)
    rows = csv_rows(figure_table(stages, PLAN).to_csv())
    for feature in PLOTTED:
        mean, worst = STATISTICS[feature]
        for index, stage in enumerate(STAGES):
            marker = by_gid(figure, f"convergence-served-{feature}-{stage}")
            assert list(marker.get_xdata()) == [served_k] and marker.get_linestyle() == "None"
            assert marker.get_marker() == series_style(4 + index)["marker"]
            for statistic in (mean, worst):
                line = by_gid(figure, f"convergence-{feature}-{stage}-{statistic}")
                assert marker.get_marker() != line.get_marker()  # its own shape: not a point of the line
                assert f"convergence-served-{feature}-{stage}-{statistic}" not in present
            entry = next(entry for entry in rows if (entry["stage"], entry["evidence"], entry["quantity"]) == (
                stage, "served_top_k", feature))
            assert [float(value) for value in marker.get_ydata()] == [pytest.approx(float(entry[mean]), rel=1e-5)]
            assert entry[worst] != "" and entry["drawn"] == "yes"  # the worst draw stays in the table
            if feature != MASS:
                assert list(marker.get_ydata()) == [row(stored(study, stage), "served_top_k", served_k, feature)[mean]]
    # The legend names the marker; that the worst draw is in the table is a note, not caption text.
    table = figure_table(stages, PLAN)
    assert "lone marker" not in table.caption and "worst draw is in the table" not in table.caption
    assert ("The figure shows the mean of the served row as a separate marker at k = 5; its worst draw is in the "
            "table.") in table.notes


def test_at_most_four_lines_per_panel_told_apart_by_marker_and_line_style_and_named_in_one_legend(stages):
    figure = plot_convergence(stages)
    for feature in PLOTTED:
        keys = [f"{stage}-{statistic}" for stage in STAGES for statistic in STATISTICS[feature]]
        lines = [by_gid(figure, f"convergence-{feature}-{key}") for key in keys]
        assert len(lines) == MAX_LINES == 4
        for slot, line in enumerate(lines):
            expected = series_style(slot)
            assert (line.get_marker(), line.get_linestyle()) == (expected["marker"], expected["linestyle"])
            assert is_grey(line.get_color()) and is_grey(line.get_markerfacecolor())
        # Identity without colour: marker shape and line style differ for every pair of lines.
        assert len({line.get_marker() for line in lines}) == len({line.get_linestyle() for line in lines}) == 4
    assert legend_labels(figure) == [
        "macro prompts, mean over draws", "macro prompts, worst draw",
        "macro prompts, logprobs (top 5), mean",
        "EDGAR releases, mean over draws", "EDGAR releases, worst draw",
        "EDGAR releases, logprobs (top 5), mean"]
    # No direct label on a line or a marker: the only texts in the panels are the panel letters.
    assert sorted(text.get_text() for text in figure.findobj(Annotation)) == ["(a)", "(b)", "(c)"]
    for artist in figure.findobj(Line2D):
        assert is_grey(artist.get_color()) and is_grey(artist.get_markeredgecolor()), artist.get_gid()


def test_the_figure_has_the_three_panels_that_carry_information_in_the_academic_style(stages, study):
    for found in (stages, convergence_stages(PLAN, stages=("full_run",), analysis_dir=study.only)):
        figure = plot_convergence(found)
        assert_academic_figure(figure, panels=3)
        assert figure.get_supxlabel() == "alternatives kept per token, k (count, log scale)"
        assert [axes.get_ylabel() for axes in figure.axes] == [
            "absolute difference to logits,\nMin-K%++ (z-score units)",
            "absolute difference to logits,\nrecall estimate (probability)",
            "probability mass covered\n(share)"]
        for axes in figure.axes:
            assert axes.get_xscale() == "log"
            assert axes.yaxis.get_offset_text().get_text() == ""  # no exponent text above the axis
            labels = [label.get_text() for label in axes.get_yticklabels()]
            assert labels and not any("e" in label.lower() for label in labels), labels  # no scientific notation
        assert_academic_svg(figures.figure_svg(figure))


def test_min_k_plus_plus_is_not_plotted_at_k_1_and_the_table_keeps_the_stored_value_with_a_note(study, stages):
    """One alternative has no spread, so the stored k = 1 value only reflects the floor on the standard deviation."""
    figure = plot_convergence(stages)
    for stage in STAGES:
        stored_row = row(stored(study, stage), "host_top_k", 1, "min_k_pp")
        assert stored_row["missing_reason"] is None and stored_row["mean_abs_diff"] is not None
        for statistic in ("mean_abs_diff", "max_abs_diff"):
            line = by_gid(figure, f"convergence-min_k_pp-{stage}-{statistic}")
            assert 1 not in list(line.get_xdata()) and list(line.get_xdata()) == [5, 10, 20]
        assert 1 in list(by_gid(figure, f"convergence-estimate-{stage}-mean_abs_diff").get_xdata())
    # The axis is scaled to the plotted points, so the unplotted value does not stretch it.
    plotted_top = max(entry[statistic] for stage in STAGES for entry in stored(study, stage)["features"]
                      for statistic in ("mean_abs_diff", "max_abs_diff")
                      if (entry["evidence"], entry["feature"]) == ("host_top_k", "min_k_pp") and entry["k"] in (5, 10, 20))
    assert figure.axes[0].get_ylim()[1] <= max(plotted_top * 1.2, 1.0)
    table = figure_table(stages, PLAN)
    rows = csv_rows(table.to_csv())
    at_one = [entry for entry in rows if (entry["evidence"], entry["k"], entry["quantity"]) == (
        "host_top_k", "1", "min_k_pp")]
    assert len(at_one) == 2
    for entry in at_one:
        expected = row(stored(study, entry["stage"]), "host_top_k", 1, "min_k_pp")["mean_abs_diff"]
        assert entry["mean_abs_diff"] == figures._number(expected) and entry["drawn"] == "no"
        assert entry["note"] == ("undefined at k = 1: one alternative has no spread, so the stored value reflects only "
                                 "the floor on the standard deviation; not plotted")
    assert "Min-K%++ is not plotted at k = 1 (see the table)." in table.caption
    assert "undefined" not in table.caption and "spread" not in table.caption  # the reason is the table's note


def test_features_that_are_zero_by_construction_are_not_plotted_and_stay_in_the_table(study, stages):
    figure = plot_convergence(stages)
    table = figure_table(stages, PLAN)
    rows = csv_rows(table.to_csv())
    for feature in ("loss", "min_k", "zlib_ratio"):
        assert not [gid for gid in gids(figure) if f"-{feature}-" in gid]
        kept = [entry for entry in rows if entry["quantity"] == feature and entry["evidence"] == "host_top_k"
                and entry["k"] in ("1", "5", "10", "20")]
        assert len(kept) == 8 and {entry["drawn"] for entry in kept} == {"no"}
        assert {entry["note"] for entry in kept} == {"zero at every k by construction; not plotted"}
        assert {float(entry["mean_abs_diff"]) for entry in kept} == {0.0}
    assert len(rows) == sum(len(stored(study, stage)["features"]) + len(stored(study, stage)["mass"])
                            for stage in STAGES)


def test_one_stage_draws_two_lines_per_panel(study):
    one = convergence_stages(PLAN, stages=("full_run",), analysis_dir=study.analysis_dir)
    figure = plot_convergence(one)
    found = {gid for gid in gids(figure) if gid.startswith("convergence-min_k_pp-")}
    assert found == {"convergence-min_k_pp-full_run-mean_abs_diff", "convergence-min_k_pp-full_run-max_abs_diff"}


# ---------- statements, refusals, missing stages ----------

def test_the_figure_states_what_is_zero_by_construction_and_that_values_compare_within_one_model(study, stages):
    table = figure_table(stages, PLAN)
    caption = table.caption
    assert (f"Values compare only within one model and one set of weights ({study.data['model']}, weights "
            f"{DIGEST}).") in table.notes
    assert plot_convergence(stages).get_suptitle() == ""  # the statement is the table's, not a title
    # Series that cannot differ by construction are named as such (Requirement 8.7), in one sentence; why they
    # cannot is a note below the table.
    assert ("By construction the differences are zero at k = 20 and, at every k, for loss, Min-K% and the zlib ratio "
            "(not plotted).") in caption
    assert "truncation" not in caption
    assert ("Loss, Min-K% and the zlib ratio use only the logprob of each generated token, which truncation keeps "
            "exact; at k = 20 every kept alternative is used.") in table.notes
    # The statement follows the stored rows: a table whose loss differs somewhere does not carry it for the loss.
    changed = copy.deepcopy(study.data)
    row(changed, "host_top_k", 1, "loss")["max_abs_diff"] = 0.5
    other_caption = figure_table([convergence_stage(changed)]).caption
    assert ("By construction the differences are zero at k = 20 and, at every k, for Min-K% and the zlib ratio (not "
            "plotted). Loss is not plotted either (see the table).") in other_caption
    changed_rows = csv_rows(figure_table([convergence_stage(changed)]).to_csv())
    assert {entry["note"] for entry in changed_rows if entry["quantity"] == "loss"
            and entry["evidence"] == "host_top_k" and entry["drawn"] == "no" and entry["k"] != "50"
            and entry["k"] != "100"} == {"not plotted"}
    other = copy.deepcopy(synthetic_negative_control(study.data))
    other["model_digest"] = "sha256:" + "0" * 64
    with pytest.raises(FigureError, match="different models or weights"):
        plot_convergence([stages[0], convergence_stage(other)])
    with pytest.raises(FigureError, match="different models or weights"):
        figure_table([stages[0], convergence_stage(other)])


def test_the_caption_is_a_journal_caption_with_the_key_numbers_and_the_sample_size(study, stages):
    table = figure_table(stages, PLAN)
    caption = table.caption
    data = study.data
    assert_journal_caption(caption, 4)
    at_served = [(row(stored(study, stage), "host_top_k", 5, "estimate")["mean_abs_diff"],
                  next(entry["mean_mass"] for entry in stored(study, stage)["mass"]
                       if (entry["evidence"], entry["k"]) == ("host_top_k", 5))) for stage in STAGES]
    difference, mass = max(value for value, _ in at_served), min(value for _, value in at_served)
    # The label of logits is built from the kept count the analysis stores: the estimate uses the kept top
    # alternatives of the next-token vector, never the whole vector.
    assert {stage.kept for stage in stages} == {data["kept"]} == {KEEP}
    assert figures.evidence_label("full_distribution", KEEP) == f"logits (top {KEEP} of the next-token vector)"
    assert (f"Figure 4. The recall estimate from the first k alternatives per token is within "
            f"{figures.rounded(difference)} on average of that from logits (top {KEEP} of the next-token vector) at "
            f"k = 5, where the mean mass covered is at least {figures.rounded(mass)}. ") in caption
    assert len(caption) < 575
    assert "full next-token" not in table.to_markdown() and "served top-" not in table.to_markdown()
    assert figure_table([replace(stage, kept=None) for stage in stages]).caption.count(
        "logits (kept alternatives of the next-token vector)") == 1
    assert (f" n = {data['n_draws']} draws of {len(data['items'])} macro prompts and {data['n_draws']} draws of "
            f"{len(data['items'])} EDGAR releases on {ROUTE}. ") in caption
    assert "(a) Min-K%++, (b) the recall estimate and (c) the mass covered; no interval is drawn." in caption
    assert caption.index(" n = ") < caption.index("no interval is drawn")
    # Every number of the caption is a value of the table below it; the number of prompts is the length of the
    # stored table's items.
    assert_caption_numbers(table, stored=(len(data["items"]),))
    # What the earlier caption carried from the analysis's own notes is not caption matter.
    assert data["estimator"] not in table.to_markdown() and data["scope"] not in table.to_markdown()
    markdown = table.to_markdown()
    assert "Moody's" in markdown and "BAA10Y" in markdown and "SEC EDGAR" in markdown
    assert "Financial Modeling Prep (FMP)" in markdown
    assert_notes(table)
    # Without the plan there is no route to name; the caption is still a caption.
    bare = figure_table(stages).caption
    assert ROUTE not in bare and f" n = {data['n_draws']} draws of {len(data['items'])} macro prompts and " in bare
    assert_journal_caption(bare, 4)


def test_a_missing_stage_or_a_route_without_logits_is_listed_not_drawn(study):
    found = convergence_stages(PLAN, analysis_dir=study.only)
    assert [stage.status for stage in found] == ["drawn", "missing"]
    assert "no convergence table at convergence-sampling-negative_control.json" in found[1].reason
    table = figure_table(found, PLAN)
    assert "EDGAR releases are not drawn (see the table)." in table.caption
    assert_journal_caption(table.caption, 4)
    rows = csv_rows(table.to_csv())
    assert [entry for entry in rows if entry["stage"] == "negative_control"] == [
        {**{column: "" for column in rows[0]}, "route": ROUTE, "stage": "negative_control", "drawn": "no",
         "missing": "the convergence analysis has not been run for these prompts"}]
    assert "arm" not in rows[0] and "role" not in rows[0] and ".json" not in table.to_csv()
    figure = plot_convergence(found)
    assert not [gid for gid in gids(figure) if "negative_control" in gid]
    assert legend_labels(figure) == ["macro prompts, mean over draws", "macro prompts, worst draw",
                                     "macro prompts, logprobs (top 5), mean"]
    serving = convergence_stages(PLAN, arm="serving", analysis_dir=study.analysis_dir)
    assert {stage.status for stage in serving} == {"missing"}
    assert {stage.reason for stage in serving} == {SERVING_FULL_DISTRIBUTION_REASON}
    with pytest.raises(FigureError, match="nothing to draw"):
        plot_convergence(serving)


def test_tables_of_another_schema_plan_or_shape_are_refused(study, tmp_path):
    data = study.data
    for key, value, message in (("schema", "other/1", "convergence table"), ("stage", "pilot", "stage"),
                                ("arm", "elsewhere", "arm")):
        with pytest.raises(FigureError, match=message):
            convergence_stage({**data, key: value})
    with pytest.raises(FigureError, match="plan"):
        convergence_stage(data, plan_sha256="0" * 64)
    broken = copy.deepcopy(data)
    broken["features"][0]["feature"] = "perplexity"
    with pytest.raises(FigureError, match="unknown convergence row"):
        convergence_stage(broken)
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema": 1, "schema": 2}', encoding="utf-8")
    with pytest.raises(FigureError, match="duplicate"):
        read_convergence_stage(path)
    with pytest.raises(FigureError, match="distinct stages"):
        convergence_stages(PLAN, stages=("full_run", "full_run"), analysis_dir=study.analysis_dir)


# ---------- outputs ----------

def test_the_figure_is_written_as_one_svg_with_a_csv_and_markdown_once(study, tmp_path):
    out = tmp_path / "assets"
    written = render_convergence(PLAN, analysis_dir=study.analysis_dir, out_dir=out)
    assert list(written) == list(figure_paths("convergence", out).values())
    assert sorted(path.name for path in out.iterdir()) == ["convergence.csv", "convergence.md", "convergence.svg"]
    assert_academic_svg(figure_paths("convergence", out)["svg"].read_text(encoding="utf-8"))
    before = {path: path.read_bytes() for path in written}
    assert render_convergence(PLAN, analysis_dir=study.analysis_dir, out_dir=out) == written
    assert {path: path.read_bytes() for path in written} == before
    with pytest.raises(FigureError, match="refusing to replace"):
        render_convergence(PLAN, stages=("full_run",), analysis_dir=study.analysis_dir, out_dir=out)
    assert {path: path.read_bytes() for path in written} == before
    markdown = figure_paths("convergence", out)["markdown"].read_text(encoding="utf-8").splitlines()
    assert markdown[0].startswith("Figure 4. ")
    rows = list(csv.reader(io.StringIO(figure_paths("convergence", out)["csv"].read_text(encoding="utf-8"))))
    assert markdown[1] == "" and markdown[3] == "|" + "|".join("---" for _ in rows[0]) + "|"
    end = 3 + len(rows)
    assert [[value.strip() for value in line.strip("|").split(" | ")] for line in [markdown[2], *markdown[4:end]]] == rows
    assert markdown[end] == "" and markdown[-1] == figures.DERIVED_ONLY and markdown[-2].startswith("Sources: ")


def test_the_svg_bytes_do_not_depend_on_the_process_that_renders_them(stages, tmp_path):
    """Two fresh processes render the same bytes (the render under this process's rcParams is checked below)."""
    from tests.scripts.finance_sim.test_figures_draws import svg_hashes_in_processes

    first, second = svg_hashes_in_processes(stages, "figures_convergence", "plot_convergence",
                                            tmp_path / "convergence.pickle", runs=2)
    assert first == second and len(first) == 1 and len(first[0]) == 64, (first, second)


def test_the_render_does_not_depend_on_the_processs_global_rc_params(stages, study):
    """The paper style's pdf renderer and tight bounding box, or another font, dpi or marker default left in the
    process by an earlier test, change nothing: the figure renders under matplotlib's defaults plus the academic
    style."""
    import hashlib

    from tests.scripts.finance_sim.test_figures import foreign_rc_params

    only = convergence_stages(PLAN, stages=("full_run",), analysis_dir=study.only)

    def hashes() -> list[str]:
        return [hashlib.sha256(figures.figure_svg(plot_convergence(found)).encode("utf-8")).hexdigest()
                for found in (stages, only)]

    clean = hashes()
    with foreign_rc_params():
        assert hashes() == clean


def test_published_outputs_carry_derived_statistics_only(study, tmp_path):
    texts = [path.read_text(encoding="utf-8") for path in
             render_convergence(PLAN, analysis_dir=study.analysis_dir, out_dir=tmp_path / "assets")]
    for text in texts:
        assert "response" not in text and "top_logprobs" not in text and "Real asset identities" not in text
        assert '"regime"' not in text and "item_changes" not in text


def test_feature_names_cover_every_feature_of_the_table():
    assert set(FEATURE_NAMES) == set(FEATURES) == set(convergence_module.FEATURES)


def test_the_command_renders_from_the_stored_tables_and_refuses_another_plan(study, tmp_path):
    plan_path = write_plan(PLAN, tmp_path / "plan.json")
    out = tmp_path / "assets"
    common = ["--plan", str(plan_path), "--analysis-dir", str(study.analysis_dir), "--out-dir", str(out)]
    assert main([*common, "--plan-sha256", PLAN.plan_sha256]) == 0
    assert sorted(path.name for path in out.iterdir()) == ["convergence.csv", "convergence.md", "convergence.svg"]
    assert main([*common, "--plan-sha256", PLAN.plan_sha256]) == 0
    assert main([*common, "--plan-sha256", "0" * 64]) == 2
    assert main([*common[:-1], str(tmp_path / "serving"), "--plan-sha256", PLAN.plan_sha256, "--arm",
                 "serving"]) == 2
    assert not (tmp_path / "serving").exists()
    only = tmp_path / "only"
    assert main([*common[:-1], str(only), "--plan-sha256", PLAN.plan_sha256, "--stage", "full_run"]) == 0
    assert sorted(path.name for path in only.iterdir()) == ["convergence.csv", "convergence.md", "convergence.svg"]
