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
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402
from matplotlib.colors import to_hex  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.text import Annotation, Text  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

from recall_guard.harness.plots import (  # noqa: E402
    PAPER_MARKERS,
    STUDY_PALETTE,
    STUDY_PALETTE_DARK,
    THEMES,
    figure_theme,
)
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
from scripts.finance_sim.figures import FigureError, figure_paths  # noqa: E402
from scripts.finance_sim.figures_convergence import (  # noqa: E402
    FEATURE_NAMES,
    MASS,
    MAX_LINES,
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

FEATURES = ("loss", "min_k", "min_k_pp", "zlib_ratio", "estimate")
STAGES = ("full_run", "negative_control")


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
    figure = plot_convergence(found, theme="light")
    for feature in FEATURES:
        line = by_gid(figure, f"convergence-{feature}-full_run-mean_abs_diff")
        drawn = [entry for entry in study.data["features"] if entry["evidence"] == "host_top_k"
                 and entry["feature"] == feature and entry["missing_reason"] is None]
        assert list(line.get_xdata()) == [entry["k"] for entry in drawn]
        assert list(line.get_ydata()) == [entry["mean_abs_diff"] for entry in drawn]
    changed = copy.deepcopy(study.data)
    row(changed, "host_top_k", 1, "min_k_pp")["mean_abs_diff"] = 0.123456
    rows = csv_rows(figure_table([convergence_stage(changed)]).to_csv())
    assert next(entry for entry in rows if (entry["evidence"], entry["k"], entry["quantity"]) == (
        "host_top_k", "1", "min_k_pp"))["mean_abs_diff"] == "0.123456"


def test_k_above_the_kept_count_is_named_in_the_caption_not_drawn(study, stages):
    assert KEEP == 20 and CONVERGENCE_KS == (1, 5, 10, 20, 50, 100)
    figure = plot_convergence(stages, theme="light")
    for line in figure.findobj(Line2D):
        if (line.get_gid() or "").startswith("convergence-") and "reference" not in line.get_gid() and (
                "leader" not in line.get_gid()):
            assert max(line.get_xdata()) <= KEEP, line.get_gid()
    caption = figure_table(stages, PLAN).caption
    assert "Not drawn: full run, negative control: k 50, 100 above the 20 alternatives the evidence keeps" in caption
    rows = csv_rows(figure_table(stages, PLAN).to_csv())
    above = [entry for entry in rows if entry["evidence"] == "host_top_k" and entry["k"] in ("50", "100")]
    assert above and {entry["drawn"] for entry in above} == {"no"}
    assert all("above the 20 alternatives" in entry["missing"] for entry in above)


def test_the_served_row_is_a_separate_open_marker_at_the_served_k(study, stages):
    served_k = stages[0].served_k
    assert served_k == 5
    for theme in THEMES:
        style = figure_theme(theme)
        figure = plot_convergence(stages, theme=theme)
        for feature in (*FEATURES, MASS):
            for stage in STAGES:
                for statistic in (("mean_mass", "min_mass") if feature == MASS else ("mean_abs_diff",
                                                                                      "max_abs_diff")):
                    marker = by_gid(figure, f"convergence-served-{feature}-{stage}-{statistic}")
                    assert list(marker.get_xdata()) == [served_k] and marker.get_linestyle() == "None"
                    assert to_hex(marker.get_markerfacecolor()) == to_hex(style.surface)  # open
                    line = by_gid(figure, f"convergence-{feature}-{stage}-{statistic}")
                    assert marker.get_marker() == line.get_marker()
                    assert to_hex(marker.get_markeredgecolor()) == to_hex(line.get_color())
                    if feature != MASS:
                        source = stored(study, stage)
                        expected = row(source, "served_top_k", served_k, feature)[statistic]
                        assert list(marker.get_ydata()) == [expected]
            # Only the estimate goes through a calibrator, so only its label names one.
            expected_label = f"served top-{served_k}" + ("\n(top-k calibrator)" if feature == "estimate" else "")
            assert by_gid(figure, f"convergence-served-label-{feature}").get_text() == expected_label
    caption = figure_table(stages, PLAN).caption
    assert "a different calibrator) is a separate open marker at k = 5, not joined to the lines" in caption


def test_at_most_four_lines_per_panel_each_with_its_own_slot_marker_and_direct_label(stages):
    for theme in THEMES:
        figure = plot_convergence(stages, theme=theme)
        palette = STUDY_PALETTE if theme == "light" else STUDY_PALETTE_DARK
        for feature in (*FEATURES, MASS):
            statistics = ("mean_mass", "min_mass") if feature == MASS else ("mean_abs_diff", "max_abs_diff")
            keys = [f"{stage}-{statistic}" for stage in STAGES for statistic in statistics]
            lines = [by_gid(figure, f"convergence-{feature}-{key}") for key in keys]
            assert len(lines) == MAX_LINES == 4
            assert [to_hex(line.get_color()) for line in lines] == [to_hex(colour) for colour in palette[:4]]
            assert [line.get_marker() for line in lines] == PAPER_MARKERS[:4]
            assert [line.get_linestyle() for line in lines] == ["-", "--", "-", "--"]
            # Identity without colour: marker shape and line style differ for every pair of lines.
            assert len({(line.get_marker(), line.get_linestyle()) for line in lines}) == 4
            axes = lines[0].axes
            labels = " ".join(text.get_text() for text in axes.texts if isinstance(text, Annotation)
                              and (text.get_gid() or "").startswith(f"convergence-label-{feature}-"))
            flat = labels.replace("\n", " ")
            assert "all lines" in flat or all(name in flat for name in (
                ("macro, mean", "macro, min", "EDGAR, mean", "EDGAR, min") if feature == MASS
                else ("macro, mean", "macro, max", "EDGAR, mean", "EDGAR, max")))
            assert to_hex(by_gid(figure, f"convergence-reference-{feature}").get_color()) == to_hex(
                figure_theme(theme).muted)
        # Truncation leaves the actual-token features unchanged: one label names all four identical lines.
        for feature in ("loss", "min_k", "zlib_ratio"):
            texts = [text.get_text().replace("\n", " ") for text in figure.findobj(Annotation)
                     if (text.get_gid() or "").startswith(f"convergence-label-{feature}-")]
            assert texts == ["all lines: 0 at every k"]
        style = figure_theme(theme)
        tokens = {to_hex(style.ink), to_hex(style.secondary_ink)}
        for text in figure.findobj(Text):
            if text.get_text().strip() and text.get_visible():
                assert to_hex(text.get_color()) in tokens, text.get_text()


def test_one_stage_draws_two_lines_per_panel(study):
    one = convergence_stages(PLAN, stages=("full_run",), analysis_dir=study.analysis_dir)
    figure = plot_convergence(one, theme="light")
    found = {gid for gid in gids(figure) if gid.startswith("convergence-min_k_pp-")}
    assert found == {"convergence-min_k_pp-full_run-mean_abs_diff", "convergence-min_k_pp-full_run-max_abs_diff"}


def _marker_boxes(axes, scale: float) -> list[Bbox]:
    boxes = []
    for line in axes.lines:
        if line.get_marker() in (None, "None", "none", ""):
            continue
        half = (line.get_markersize() + line.get_markeredgewidth()) / 2.0 * scale
        boxes += [Bbox.from_extents(x - half, y - half, x + half, y + half)
                  for x, y in line.get_transform().transform(line.get_xydata())]
    return boxes


def test_labels_stay_clear_of_lines_markers_each_other_and_the_panel_frame(stages, study):
    for found in (stages, convergence_stages(PLAN, stages=("full_run",), analysis_dir=study.only)):
        for theme in THEMES:
            figure = plot_convergence(found, theme=theme)
            with matplotlib.rc_context(figures._theme_rc(theme)):  # measured under the figure's own rcParams
                figure.draw_without_rendering()
            scale = figure.dpi / 72.0
            for axes in figure.axes:
                labels = [text for text in axes.texts if isinstance(text, Annotation)]
                lines = [line.get_path().transformed(line.get_transform()) for line in axes.lines
                         if line.get_linestyle() not in ("None", "none", "")
                         and "leader" not in (line.get_gid() or "")]
                markers = _marker_boxes(axes, scale)
                extents = [label.get_window_extent() for label in labels]
                frame = axes.bbox
                for index, (label, extent) in enumerate(zip(labels, extents, strict=True)):
                    where = (theme, label.get_gid(), label.get_text())
                    assert (frame.x0 <= extent.x0 and extent.x1 <= frame.x1 and frame.y0 <= extent.y0
                            and extent.y1 <= frame.y1), where
                    assert not any(extent.overlaps(box) for box in markers), where
                    assert not any(extent.overlaps(other) for other in extents[index + 1:]), where
                    # Line labels sit in a column left of every line; the served and reference labels keep 3 pt from
                    # every line unless they are the by-design fallback (whose box masks the line).
                    if not figures.placement_fallback(label):
                        assert not any(path.intersects_bbox(extent.padded(3.0 * scale), filled=False)
                                       for path in lines), where


def high_mass(data: dict) -> dict:
    """The full run's table with the probe arm's shape of mass: the first alternative carries most of it.

    The mean lines, the served marker and the whole-vocabulary line meet within
    a hundredth from k 5 on (the figure published from the real tables), which
    leaves the labels there no room unless they are placed by their boxes.
    """
    other = copy.deepcopy(data)
    mean = {1: 0.95, 5: 0.99, 10: 0.999, 20: 0.9999, 50: 0.99999, 100: 0.999999}
    low = {1: 0.1, 5: 0.5, 10: 0.9, 20: 0.98, 50: 0.99, 100: 0.995}
    for row in other["mass"]:
        if row["mean_mass"] is not None:
            row["mean_mass"], row["min_mass"] = mean[row["k"]], low[row["k"]]
    other["full_mean_mass"], other["full_min_mass"] = 0.99999, 0.99
    return other


def _drawn_marker_boxes(axes, scale: float) -> list[tuple[float, Bbox]]:
    """Every marker as drawn, with its k: the marker's path at the marker size plus half the edge width.

    A diamond reaches 0.71 of the marker size from its centre, a circle or a
    square 0.5, so the half-size boxes of :func:`_marker_boxes` understate it.
    """
    from matplotlib.markers import MarkerStyle

    boxes = []
    for line in axes.lines:
        marker = line.get_marker()
        if marker in (None, "None", "none", ""):
            continue
        style = MarkerStyle(marker)
        reach = style.get_path().transformed(style.get_transform()).get_extents()
        size, edge = line.get_markersize() * scale, line.get_markeredgewidth() * scale / 2.0
        points = line.get_xydata()
        for (k, _), (cx, cy) in zip(points, line.get_transform().transform(points), strict=True):
            boxes.append((float(k), Bbox.from_extents(cx + reach.x0 * size - edge, cy + reach.y0 * size - edge,
                                                      cx + reach.x1 * size + edge, cy + reach.y1 * size + edge)))
    return boxes


def test_direct_labels_never_collide_with_each_other_or_with_the_markers(stages, study, tmp_path):
    """Every direct label, as drawn with its surface box, keeps 2 pt from every other label's box and from every
    marker as drawn, the k 1 stack of every line included. Checked on the fixture and on the probe arm's shape of
    mass, where the published figure printed the whole-vocabulary label over the served label."""
    shaped = tmp_path / "high-mass"
    shaped.mkdir()
    full_run = high_mass(stored(study))
    for stage, data in (("full_run", full_run), ("negative_control", synthetic_negative_control(full_run))):
        table_path("sampling", stage, shaped).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n",
                                                         encoding="utf-8")
    for found in (stages, convergence_stages(PLAN, analysis_dir=shaped)):
        assert [stage.status for stage in found] == ["drawn", "drawn"]
        for theme in THEMES:
            figure = plot_convergence(found, theme=theme)
            with matplotlib.rc_context(figures._theme_rc(theme)):  # measured under the figure's own rcParams
                figure.draw_without_rendering()
            scale = figure.dpi / 72.0
            for axes in figure.axes:
                labels = [text for text in axes.texts if isinstance(text, Annotation)]
                boxes = [label.get_bbox_patch().get_window_extent() for label in labels]
                markers = _drawn_marker_boxes(axes, scale)
                assert any(k == 1.0 for k, _ in markers) or not labels
                for index, (label, box) in enumerate(zip(labels, boxes, strict=True)):
                    where = (theme, label.get_gid(), label.get_text())
                    for other, other_box in zip(labels[index + 1:], boxes[index + 1:], strict=True):
                        assert not box.padded(scale).overlaps(other_box.padded(scale)), (where, other.get_text())
                    touched = [k for k, marker in markers if box.padded(2.0 * scale).overlaps(marker)]
                    assert not touched, (where, touched)


# ---------- statements, refusals, missing stages ----------

def test_the_figure_states_that_values_compare_only_within_one_model_and_set_of_weights(study, stages):
    caption = figure_table(stages, PLAN).caption
    assert study.data["comparability"] in caption and DIGEST in caption
    assert "Values compare only within one model and one set of weights" in caption
    figure = plot_convergence(stages, theme="light")
    assert "Values compare only within one model and one set of weights" in figure.get_suptitle()
    other = copy.deepcopy(synthetic_negative_control(study.data))
    other["model_digest"] = "sha256:" + "0" * 64
    with pytest.raises(FigureError, match="different models or weights"):
        plot_convergence([stages[0], convergence_stage(other)])
    with pytest.raises(FigureError, match="different models or weights"):
        figure_table([stages[0], convergence_stage(other)])


def test_the_caption_states_data_sample_size_interval_method_and_sources(study, stages):
    caption = figure_table(stages, PLAN).caption
    data = study.data
    assert caption.startswith("Figure F4.")
    assert "stored convergence tables (task 7.12; read, not recomputed) of the sampling arm (probe role" in caption
    assert (f"full run: {data['n_draws']} draws of {len(data['items'])} items ({len(data['missing_draws'])} draws and "
            f"{len(data['missing_items'])} planned items missing), {KEEP} kept alternatives of a {data['n_vocab']}-token "
            "vocabulary") in caption
    assert "Interval: none" in caption and data["estimator"] in caption and data["scope"] in caption
    assert "Moody's" in caption and "BAA10Y" in caption and "SEC EDGAR" in caption
    assert "Financial Modeling Prep (FMP)" in caption and figures.DERIVED_ONLY in caption
    assert "macro is the full run" in caption and "EDGAR the negative control" in caption


def test_a_missing_stage_or_an_arm_without_full_distribution_evidence_is_listed_not_drawn(study):
    found = convergence_stages(PLAN, analysis_dir=study.only)
    assert [stage.status for stage in found] == ["drawn", "missing"]
    assert "no convergence table at convergence-sampling-negative_control.json" in found[1].reason
    table = figure_table(found, PLAN)
    assert "Missing (not drawn): negative control: no convergence table" in table.caption
    rows = csv_rows(table.to_csv())
    assert [entry for entry in rows if entry["stage"] == "negative_control"] == [
        {**{column: "" for column in rows[0]}, "arm": "sampling", "role": "probe", "stage": "negative_control",
         "drawn": "no", "missing": found[1].reason}]
    figure = plot_convergence(found, theme="dark")
    assert not [gid for gid in gids(figure) if "negative_control" in gid]
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

def test_the_figure_is_written_in_light_and_dark_svg_with_a_csv_and_markdown_once(study, tmp_path):
    out = tmp_path / "assets"
    written = render_convergence(PLAN, analysis_dir=study.analysis_dir, out_dir=out)
    assert list(written) == list(figure_paths("convergence", out).values())
    assert sorted(path.name for path in out.iterdir()) == ["convergence-dark.svg", "convergence-light.svg",
                                                           "convergence.csv", "convergence.md"]
    light = figure_paths("convergence", out)["light"].read_text(encoding="utf-8")
    dark = figure_paths("convergence", out)["dark"].read_text(encoding="utf-8")
    assert light != dark and "<dc:date>" not in light and STUDY_PALETTE[3] in light and STUDY_PALETTE_DARK[3] in dark
    before = {path: path.read_bytes() for path in written}
    assert render_convergence(PLAN, analysis_dir=study.analysis_dir, out_dir=out) == written
    assert {path: path.read_bytes() for path in written} == before
    with pytest.raises(FigureError, match="refusing to replace"):
        render_convergence(PLAN, stages=("full_run",), analysis_dir=study.analysis_dir, out_dir=out)
    assert {path: path.read_bytes() for path in written} == before
    markdown = figure_paths("convergence", out)["markdown"].read_text(encoding="utf-8").splitlines()
    rows = list(csv.reader(io.StringIO(figure_paths("convergence", out)["csv"].read_text(encoding="utf-8"))))
    assert markdown[1] == "" and markdown[3] == "|" + "|".join("---" for _ in rows[0]) + "|"
    assert [[value.strip() for value in line.strip("|").split(" | ")] for line in [markdown[2], *markdown[4:]]] == rows


def test_the_svg_bytes_do_not_depend_on_the_process_that_renders_them(stages, tmp_path):
    """Two fresh processes render the same bytes (the render under this process's rcParams is checked below)."""
    from tests.scripts.finance_sim.test_figures_draws import svg_hashes_in_processes

    first, second = svg_hashes_in_processes(stages, "figures_convergence", "plot_convergence",
                                            tmp_path / "convergence.pickle", runs=2)
    assert first == second and len(first) == 2 and first[0] != first[1], (first, second)


def test_the_render_does_not_depend_on_the_processs_global_rc_params(stages, study):
    """The paper style's pdf renderer and tight bounding box, or another font, dpi or marker default left in the
    process by an earlier test, change nothing: the figure renders under matplotlib's defaults plus the theme."""
    import hashlib

    from tests.scripts.finance_sim.test_figures import foreign_rc_params

    only = convergence_stages(PLAN, stages=("full_run",), analysis_dir=study.only)

    def hashes() -> list[str]:
        return [hashlib.sha256(figures.figure_svg(plot_convergence(found, theme=theme)).encode("utf-8")).hexdigest()
                for found in (stages, only) for theme in THEMES]

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
    assert sorted(path.name for path in out.iterdir()) == ["convergence-dark.svg", "convergence-light.svg",
                                                           "convergence.csv", "convergence.md"]
    assert main([*common, "--plan-sha256", PLAN.plan_sha256]) == 0
    assert main([*common, "--plan-sha256", "0" * 64]) == 2
    assert main([*common[:-1], str(tmp_path / "serving"), "--plan-sha256", PLAN.plan_sha256, "--arm",
                 "serving"]) == 2
    assert not (tmp_path / "serving").exists()
    only = tmp_path / "only"
    assert main([*common[:-1], str(only), "--plan-sha256", PLAN.plan_sha256, "--stage", "full_run"]) == 0
    assert (only / "convergence-light.svg").exists()
