"""Pre-registered analysis: accuracy, losses and multiple testing (task 7.10).

Requirements 6.3, 8.3, 8.4, 8.8, 8.9 and 8.10.

Pure logic on fakes and recorded fixtures: serving-arm results are derived from
synthetic temperature-0 draws without a request, so every correctness, estimate
and scored probability is known; the sampling arm's results come from the real
7.6 calibration and 7.7 execution on the generation and full-distribution
doubles of the 7.7 tests, recorded under ``tmp_path``. Sockets are blocked, no
credentials or model host are used, and nothing is written under ``runs/``.
"""

from __future__ import annotations

import json
import math
import socket
from collections.abc import Callable, Sequence
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from recall_guard.core.distribution import RecordedDraw
from recall_guard.core.significance import block_bootstrap, did_gap, holm, paired_loss_difference
from scripts.finance_sim import analysis as analysis_module
from scripts.finance_sim.analysis import (
    ANALYSIS_DIR,
    ANALYSIS_METRICS,
    ANALYSIS_SCHEMA,
    COMPUTED,
    CONDITIONAL_NOTE,
    DEFERRED,
    DEFERRED_REASON,
    ENSEMBLE_METRICS,
    MISSING,
    NO_ESTIMATE,
    NO_RESULT,
    POWER_APPROXIMATION,
    REFERENCE,
    TESTED_METRICS,
    AnalysisError,
    analysis_from_records,
    analysis_path,
    main,
    pre_registered_analysis,
    write_analysis,
)
from scripts.finance_sim.calibrate import DrawSettings, FitSettings, run_calibration
from scripts.finance_sim.execute import (
    ExecutionItem,
    ItemResult,
    VariantResult,
    derive_result,
    execute_items,
    load_calibrators,
    macro_items,
    planned_keys,
)
from scripts.finance_sim.limits import BudgetLedger, LedgerCallBudget, StudyLimits
from scripts.finance_sim.pilot import PILOT_EXCLUDED, TOO_FEW_PERIOD_MONTHS, PilotError, achieved_power
from scripts.finance_sim.plan import (
    EDGAR_FILES,
    SCORED_LOSSES,
    SERVING_FULL_DISTRIBUTION_REASON,
    VARIANTS,
    AnalysisSettings,
    MonthClass,
    PilotSettings,
    PlanChangedError,
    ScoredProbability,
    StudyPlan,
    freeze_plan,
    load_serving_manifest,
    split_months,
    write_plan,
)
from tests.scripts.finance_sim.test_execute import (
    ALLOWANCE,
    CREATED_AT,
    EDGAR_ROWS,
    GEMMA,
    MACRO_ITEMS,
    MISSING_MONTH,
    MISSING_REASON,
    MONTHS,
    PILOT_DRAWS,
    PLAN,
    SEED,
    SEED_BASE,
    TEMPERATURE,
    Clock,
    FakeFull,
    FakeHost,
    digest,
    edgar_releases,
    item_hash,
    macro_row,
    realized,
    serving_identity,
    small_arms,
)

#: Loadings with a positive defensive spread, and their mirror with a negative one.
DEFENSIVE = {"inflation": 0.5, "growth": -0.5, "credit_stress": 0.5, "policy": 0.0, "risk_appetite": -0.5}
RISK = {"inflation": -0.5, "growth": 0.5, "credit_stress": -0.5, "policy": 0.0, "risk_appetite": 0.5}
#: A pre-cutoff evaluation month whose identifying ensemble fails (two of three draws unparsed, draw 0 among them).
FAILED_MONTH = next(month for month in PLAN.months.evaluation if month <= "2024-08")
ACCURACY_METRICS = ("accuracy_difference_pre_cutoff", "accuracy_difference_post_cutoff", "did_accuracy_gap")
ESTIMATE_METRICS = ("estimate_difference_pre_cutoff", "estimate_difference_post_cutoff")
LOSS_METRICS = ("brier_difference", "log_loss_difference")
COMPUTED_SERVING = ("single_none", "single_top_k", "ensemble_none", "ensemble_top_k")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("the analysis attempted a network connection")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)


def plan_with(analysis: AnalysisSettings | None = None, scored: ScoredProbability | None = None) -> StudyPlan:
    """The 7.7 tests' frozen plan with other analysis or scored-probability settings (so another plan_sha256)."""
    months = [MonthClass(month, {GEMMA: month <= "2024-08"}) for month in MONTHS]
    items = {item.item_id: item_hash(item) for item in MACRO_ITEMS} | {row.row_id: digest(row.row_id)
                                                                      for row in EDGAR_ROWS}
    arms = small_arms()
    pilot = replace(PilotSettings.for_split(split_months(months, seed=SEED), arms), draws=PILOT_DRAWS,
                    min_parsed=PILOT_DRAWS)
    return freeze_plan(months, EDGAR_ROWS, seed=SEED, created_at=CREATED_AT, items=items,
                       edgar_files={name: digest(name) for name in EDGAR_FILES}, arms=arms,
                       serving_manifest=load_serving_manifest(), pilot=pilot, analysis=analysis,
                       scored_probability=scored)


#: Same items and split as PLAN with fewer resamples, for the variations of the input.
FAST = plan_with(AnalysisSettings(n_resamples=200))


# ---------- serving-arm results with known values ----------

def serving_correct(number: int) -> tuple[bool, bool]:
    """(identifying correct, anonymised correct) of the serving arm's planned month ``number``."""
    return number % 3 != 0, number % 2 == 0


def serving_estimate(number: int, identifying: bool) -> float:
    """The per-draw top-k estimate of a month's rendering: higher and more varied for the identifying prompt."""
    return 0.2 + 0.1 * (number % 4) if identifying else 0.05 * (number % 3)


def answer(plan: StudyPlan, item: ExecutionItem, correct: bool) -> str:
    """A reply whose spread sign is correct, or not, for the item's realized returns."""
    if item.realized is None:
        return json.dumps(DEFENSIVE)
    defensive_wins = plan.scored_probability.outcome_of(item.realized) == 1
    return json.dumps(DEFENSIVE if correct == defensive_wins else RISK)


def serving_result(plan: StudyPlan, item: ExecutionItem, correct: bool, estimate: float, *,
                   failed: bool = False) -> ItemResult:
    """A serving-arm full-run result from three temperature-0 draws, derived without a request."""
    text = answer(plan, item, correct)
    draws = []
    for index in range(plan.arms.arms["serving"].draws):
        if failed and index < 2:
            draws.append(RecordedDraw(draw_index=index, temperature=0.0, seed=None, response_sha256=digest("prose"),
                                      parsed=None, failure_code="parse_error", estimate=None, evidence_ref=None))
        else:
            draws.append(RecordedDraw(draw_index=index, temperature=0.0, seed=None, response_sha256=digest(text),
                                      parsed=json.loads(text), failure_code=None, estimate=estimate,
                                      evidence_ref="e" * 64))
    return derive_result(item, draws, None, plan=plan, arm="serving", stage="full_run",
                         identities={"top_k": serving_identity(item)})


def serving_months(plan: StudyPlan) -> list[str]:
    """The serving arm's planned full-run month ids, in plan order."""
    return list(dict.fromkeys(item_id for item_id, _ in planned_keys(plan, "serving", "full_run")))


def serving_results(plan: StudyPlan, *, keep: Callable[[str], bool] = lambda item_id: True) -> list[ItemResult]:
    order = {item_id: number for number, item_id in enumerate(serving_months(plan))}
    results = []
    for item in macro_items(plan, MACRO_ITEMS, stage="full_run", arm="serving"):
        if not keep(item.item_id):
            continue
        number, identifying = order[item.item_id], item.rendering == "identifying"
        correct = serving_correct(number)[0 if identifying else 1]
        results.append(serving_result(plan, item, correct, serving_estimate(number, identifying),
                                      failed=identifying and item.item_id == f"macro-{FAILED_MONTH}"))
    return results


@pytest.fixture(scope="module")
def serving() -> SimpleNamespace:
    results = serving_results(PLAN)
    return SimpleNamespace(results=results, analysis=pre_registered_analysis(PLAN, results, arm="serving"))


@pytest.fixture(scope="module")
def fast() -> SimpleNamespace:
    return SimpleNamespace(results=serving_results(FAST))


# ---------- independent recomputation in the tests ----------

def complete_rows(plan: StudyPlan, results: Sequence[ItemResult], arm: str,
                  value: Callable[[ItemResult], object]) -> list[tuple[str, bool, object, object]]:
    """(month id, pre-cutoff, identifying value, anonymised value) of every month whose renderings both have one."""
    keyed = {result.key: result for result in results}
    pre = {f"macro-{month}" for month in plan.months.pre_cutoff[GEMMA]}
    rows = []
    for item_id in dict.fromkeys(item_id for item_id, _ in planned_keys(plan, arm, "full_run")):
        pair = keyed.get((item_id, "identifying")), keyed.get((item_id, "anonymised"))
        if None in pair:
            continue
        values = [value(result) for result in pair]
        if None in values:
            continue
        rows.append((item_id, item_id in pre, values[0], values[1]))
    return rows


def correctness(variant: str) -> Callable[[ItemResult], float | None]:
    def value(result: ItemResult) -> float | None:
        correct = result.variant(variant).correct
        return None if correct is None else float(correct)

    return value


def estimate_of(variant: str) -> Callable[[ItemResult], float | None]:
    return lambda result: result.variant(variant).p_memorized


def as_array(rows) -> np.ndarray:
    return np.asarray([[a, b, 1.0 if pre else 0.0] for _, pre, a, b in rows], dtype=float)


def did_of(array: np.ndarray) -> float:
    n = array.shape[0]
    return did_gap(np.concatenate([array[:, 0], array[:, 1]]), np.concatenate([np.ones(n, bool), np.zeros(n, bool)]),
                   np.concatenate([array[:, 2] == 1.0, array[:, 2] == 1.0]))


def period_of(pre: bool) -> Callable[[np.ndarray], float]:
    def statistic(array: np.ndarray) -> float:
        chosen = array[array[:, 2] == (1.0 if pre else 0.0)]
        if chosen.shape[0] == 0:
            raise ValueError("empty period")
        return float(np.mean(chosen[:, 0] - chosen[:, 1]))

    return statistic


def replica_resamples(array: np.ndarray, statistic, plan: StudyPlan) -> np.ndarray:
    """The design's non-circular moving-block resampling, written out here, keeping the resamples that compute."""
    rng = np.random.default_rng(plan.seed)
    n, length = array.shape[0], plan.analysis.block_length
    kept = []
    for _ in range(plan.analysis.n_resamples):
        starts = rng.integers(0, n - length + 1, size=math.ceil(n / length))
        index = (starts[:, None] + np.arange(length)).ravel()[:n]
        try:
            kept.append(statistic(array[index]))
        except ValueError:
            continue
    return np.asarray(kept, dtype=float)


def settings(plan: StudyPlan) -> dict:
    return {"block_length": plan.analysis.block_length, "n_resamples": plan.analysis.n_resamples, "seed": plan.seed,
            "confidence": 1.0 - plan.analysis.alpha}


def brier(p: float, outcome: int) -> float:
    return (p - outcome) ** 2


def log_loss(p: float, outcome: int) -> float:
    return -(outcome * math.log(p) + (1 - outcome) * math.log(1.0 - p))


# ---------- the frozen metric names ----------

def test_the_output_lists_exactly_the_plans_frozen_metric_names_in_order(serving):
    analysis = serving.analysis
    assert [entry.metric for entry in analysis.metrics] == list(PLAN.analysis.metrics)
    assert set(ANALYSIS_METRICS) | set(ENSEMBLE_METRICS) == set(PLAN.analysis.metrics)
    for entry in analysis.metrics:
        if entry.metric in ENSEMBLE_METRICS:
            assert (entry.status, entry.reason, entry.variants) == (DEFERRED, DEFERRED_REASON, ())
        else:
            assert entry.status == COMPUTED
    data = analysis.to_dict()
    assert data["schema"] == ANALYSIS_SCHEMA and [entry["metric"] for entry in data["metrics"]] == list(
        PLAN.analysis.metrics)
    assert (data["arm"], data["stage"], data["plan_sha256"], data["model"]) == (
        "serving", "full_run", PLAN.plan_sha256, GEMMA)
    for name in (*ESTIMATE_METRICS, *ACCURACY_METRICS, *LOSS_METRICS):
        assert [cell.variant for cell in analysis.metric(name).variants] == list(VARIANTS)


def test_a_frozen_metric_name_no_analysis_implements_is_reported_missing():
    odd = plan_with(AnalysisSettings(n_resamples=200, metrics=("did_accuracy_gap", "an_unregistered_statistic")))
    analysis = pre_registered_analysis(odd, serving_results(odd), arm="serving")
    assert [entry.metric for entry in analysis.metrics] == ["did_accuracy_gap", "an_unregistered_statistic"]
    unknown = analysis.metric("an_unregistered_statistic")
    assert unknown.status == MISSING and "nothing is substituted" in unknown.reason and unknown.variants == ()


# ---------- accuracy differences and the DiD gap (6.3, 8.3) ----------

def test_the_accuracy_differences_and_the_did_gap_use_did_gap_over_the_months_with_both_renderings(serving):
    analysis = serving.analysis
    for variant in COMPUTED_SERVING:
        rows = complete_rows(PLAN, serving.results, "serving", correctness(variant))
        array = as_array(rows)
        pre = [a - b for _, is_pre, a, b in rows if is_pre]
        post = [a - b for _, is_pre, a, b in rows if not is_pre]
        cells = [analysis.metric(name).cell(variant) for name in ACCURACY_METRICS]
        assert [cell.statistic for cell in cells] == pytest.approx([np.mean(pre), np.mean(post), did_of(array)])
        assert cells[2].statistic == pytest.approx(cells[0].statistic - cells[1].statistic)
        for cell in cells:
            assert cell.missing is None
            assert (cell.months, cell.pre_cutoff_months, cell.post_cutoff_months) == (len(rows), len(pre), len(post))
            missing = {entry["item_id"]: entry["reason"] for entry in cell.missing_months}
            assert set(missing) == {f"macro-{MISSING_MONTH}", f"macro-{FAILED_MONTH}"}
            assert MISSING_REASON in missing[f"macro-{MISSING_MONTH}"]
            assert missing[f"macro-{FAILED_MONTH}"].startswith("identifying: no decision")
    assert len(complete_rows(PLAN, serving.results, "serving", correctness("ensemble_top_k"))) == 88


def test_the_did_interval_and_p_value_are_the_moving_block_bootstrap_with_the_plans_settings(serving):
    for variant in COMPUTED_SERVING:
        array = as_array(complete_rows(PLAN, serving.results, "serving", correctness(variant)))
        cell = serving.analysis.metric("did_accuracy_gap").cell(variant)
        expected = block_bootstrap(did_of, array, **settings(PLAN))
        assert cell.interval == expected
        kept = replica_resamples(array, did_of, PLAN)
        lo, hi = np.percentile(kept, [2.5, 97.5])
        assert (cell.interval.lo, cell.interval.hi) == pytest.approx((lo, hi))
        p_value = min(1.0, 2.0 * min(1 + np.count_nonzero(kept <= 0), 1 + np.count_nonzero(kept >= 0)) / (
            1 + kept.size))
        assert cell.p_value == pytest.approx(p_value)
        assert cell.dropped_resamples == PLAN.analysis.n_resamples - kept.size
        for name, pre in (("accuracy_difference_pre_cutoff", True), ("accuracy_difference_post_cutoff", False)):
            period = serving.analysis.metric(name).cell(variant)
            assert period.interval == block_bootstrap(period_of(pre), array, **settings(PLAN))
            assert period.p_value is None and period.p_holm is None  # a measurement (6.3), not a test


# ---------- the estimate differences (6.3) ----------

def test_the_estimate_differences_use_p_memorized_and_the_no_adjustment_variants_have_none(serving):
    analysis = serving.analysis
    for variant in ("single_top_k", "ensemble_top_k"):
        rows = complete_rows(PLAN, serving.results, "serving", estimate_of(variant))
        array = as_array(rows)
        for name, pre in (("estimate_difference_pre_cutoff", True), ("estimate_difference_post_cutoff", False)):
            cell = analysis.metric(name).cell(variant)
            assert cell.statistic == pytest.approx(np.mean([a - b for _, is_pre, a, b in rows if is_pre == pre]))
            assert cell.interval == block_bootstrap(period_of(pre), array, **settings(PLAN))
            assert cell.p_value is None and cell.missing is None
            # The month without realized returns still carries an estimate; the failed rendering does not.
            assert {entry["item_id"] for entry in cell.missing_months} == {f"macro-{FAILED_MONTH}"}
    for name in ESTIMATE_METRICS:
        for variant in ("single_none", "ensemble_none"):
            cell = analysis.metric(name).cell(variant)
            assert (cell.missing, cell.statistic, cell.interval) == (NO_ESTIMATE, None, None)


def test_the_serving_arms_full_distribution_variants_are_missing_with_the_plans_reason_in_every_metric(serving):
    for name in (*ESTIMATE_METRICS, *ACCURACY_METRICS, *LOSS_METRICS):
        for variant in ("single_full_distribution", "ensemble_full_distribution"):
            cell = serving.analysis.metric(name).cell(variant)
            assert cell.missing == SERVING_FULL_DISTRIBUTION_REASON
            assert (cell.statistic, cell.interval, cell.p_value, cell.p_holm) == (None, None, None, None)
    assert serving.analysis.missing_variants == dict(PLAN.arms.arms["serving"].missing_variants)


# ---------- Brier and log-loss differences (8.4) ----------

def test_brier_and_log_loss_differences_are_paired_against_no_adjustment_of_the_same_draw_mode(serving):
    for name, loss in (("brier_difference", brier), ("log_loss_difference", log_loss)):
        entry = serving.analysis.metric(name)
        for variant, reference in (("single_top_k", "single_none"), ("ensemble_top_k", "ensemble_none")):
            def value(result, variant=variant, reference=reference):
                chosen, base = result.variant(variant), result.variant(reference)
                if chosen.adjusted_scored_probability is None:
                    return None
                assert chosen.scored_probability == base.scored_probability  # paired on the same draws
                return chosen.adjusted_scored_probability, base.adjusted_scored_probability, result.outcome

            rows = complete_rows(PLAN, serving.results, "serving", value)
            a = np.asarray([[loss(ids[0], ids[2]), loss(anons[0], anons[2])] for _, _, ids, anons in rows])
            b = np.asarray([[loss(ids[1], ids[2]), loss(anons[1], anons[2])] for _, _, ids, anons in rows])
            expected = paired_loss_difference(a, b, **settings(PLAN))
            cell = entry.cell(variant)
            assert cell.reference == reference and cell.missing is None
            assert (cell.statistic, cell.interval, cell.p_value) == (expected.statistic, expected.interval,
                                                                     expected.p_value)
            assert cell.statistic == pytest.approx(a.mean() - b.mean()) and cell.statistic < 0  # shrinking helps here
            assert cell.dropped_resamples == expected.parameters["dropped_resamples"] == 0
            assert cell.months == len(rows) == 88
        for variant in ("single_none", "ensemble_none"):
            assert entry.cell(variant).missing == REFERENCE and entry.cell(variant).statistic is None
    assert serving.analysis.losses == dict(SCORED_LOSSES)


# ---------- Holm across variants (8.8) ----------

def test_holm_adjusts_each_tested_metric_across_the_variants_with_a_p_value(serving):
    assert TESTED_METRICS == ("did_accuracy_gap", "brier_difference", "log_loss_difference")
    for name in TESTED_METRICS:
        entry = serving.analysis.metric(name)
        raw = {cell.variant: cell.p_value for cell in entry.variants if cell.p_value is not None}
        assert entry.holm_members == tuple(raw)
        adjusted = holm(raw)
        for cell in entry.variants:
            assert cell.p_holm == adjusted.get(cell.variant)
            if cell.p_value is not None:
                assert cell.p_value <= cell.p_holm <= 1.0
    assert serving.analysis.metric("did_accuracy_gap").holm_members == COMPUTED_SERVING
    assert serving.analysis.metric("brier_difference").holm_members == ("single_top_k", "ensemble_top_k")
    for name in (*ESTIMATE_METRICS, "accuracy_difference_pre_cutoff", "accuracy_difference_post_cutoff"):
        assert serving.analysis.metric(name).holm_members == ()


# ---------- achieved power (8.9) ----------

def test_achieved_power_is_the_pilot_modules_t_test_approximation_under_independent_months(serving):
    entry = serving.analysis.metric("achieved_power")
    expected = achieved_power(PLAN, serving.results, arm="serving")
    assert entry.status == COMPUTED and entry.variants == ()
    power = entry.power
    assert power["power"] == expected.power and power["approximation"] == POWER_APPROXIMATION
    assert "t-test approximation under independent months" in POWER_APPROXIMATION
    assert (power["pre_cutoff_months"], power["post_cutoff_months"], power["noise_sd"]) == (
        expected.pre_cutoff_months, expected.post_cutoff_months, expected.noise_sd)
    assert (power["effect_size"], power["alpha"], power["n_sim"], power["seed"], power["noise_variant"]) == (
        PLAN.analysis.effect_size, PLAN.analysis.alpha, PLAN.analysis.n_resamples, PLAN.seed, "ensemble_top_k")
    assert {entry["item_id"] for entry in power["missing_months"]} == {f"macro-{MISSING_MONTH}",
                                                                        f"macro-{FAILED_MONTH}"}


# ---------- pilot items excluded (8.9) ----------

def test_pilot_items_are_excluded_with_the_reason_and_change_no_statistic(serving):
    pilot_items = macro_items(PLAN, MACRO_ITEMS, stage="pilot")[:4]
    pilot_results = [serving_result(PLAN, item, True, 0.5) for item in pilot_items]
    mixed = pre_registered_analysis(PLAN, [*pilot_results, *serving.results], arm="serving")
    assert [(entry["item_id"], entry["rendering"]) for entry in mixed.excluded] == sorted(
        item.key for item in pilot_items)
    assert {entry["reason"] for entry in mixed.excluded} == {PILOT_EXCLUDED}
    assert serving.analysis.excluded == ()
    assert mixed.to_dict()["metrics"] == serving.analysis.to_dict()["metrics"]


# ---------- missing never substituted (8.10) ----------

def test_an_incomplete_run_reports_missing_results_and_metrics_and_never_substitutes(fast):
    post = {f"macro-{month}" for month in FAST.months.evaluation if month > "2024-08"}
    results = [result for result in fast.results if result.item_id not in post]
    analysis = pre_registered_analysis(FAST, results, arm="serving")
    assert {(entry["item_id"], entry["rendering"]) for entry in analysis.missing_results} == {
        (item_id, rendering) for item_id in post for rendering in ("identifying", "anonymised")}
    assert {entry["reason"] for entry in analysis.missing_results} == {NO_RESULT}
    for variant in COMPUTED_SERVING:
        assert analysis.metric("accuracy_difference_pre_cutoff").cell(variant).statistic is not None
        for name in ("accuracy_difference_post_cutoff", "did_accuracy_gap"):
            cell = analysis.metric(name).cell(variant)
            assert "no post-cutoff month" in cell.missing
            assert (cell.statistic, cell.interval, cell.p_value, cell.p_holm, cell.post_cutoff_months) == (
                None, None, None, None, 0)
    assert analysis.metric("did_accuracy_gap").holm_members == ()
    brier_cell = analysis.metric("brier_difference").cell("ensemble_top_k")
    assert brier_cell.missing is None and brier_cell.post_cutoff_months == 0
    power = analysis.metric("achieved_power")
    assert (power.status, power.reason, power.power["power"]) == (MISSING, TOO_FEW_PERIOD_MONTHS, None)
    text = analysis.to_markdown()
    assert "no post-cutoff month" in text and NO_RESULT in text


def test_fewer_months_than_the_block_length_leave_every_bootstrap_metric_missing(fast):
    months = serving_months(FAST)
    kept = {*months[2:4], *months[-3:-1]}  # two pre-cutoff and two post-cutoff months, all with returns
    analysis = pre_registered_analysis(FAST, [result for result in fast.results if result.item_id in kept],
                                       arm="serving")
    for name in (*ESTIMATE_METRICS, *ACCURACY_METRICS, *LOSS_METRICS):
        for cell in analysis.metric(name).variants:
            if cell.variant in ("single_top_k", "ensemble_top_k") or name in ACCURACY_METRICS and cell.variant in (
                    "single_none", "ensemble_none"):
                assert "fewer than the frozen block length of 6 months" in cell.missing
                assert (cell.statistic, cell.interval, cell.p_value) == (None, None, None)
                assert cell.months == 4


def test_every_size_and_threshold_comes_from_the_frozen_plan(monkeypatch):
    odd = plan_with(AnalysisSettings(block_length=3, n_resamples=321, alpha=0.1, effect_size=0.25))
    calls = []
    real_bootstrap, real_loss = analysis_module.block_bootstrap, analysis_module.paired_loss_difference

    def bootstrap(statistic, blocks, **kwargs):
        calls.append(kwargs)
        return real_bootstrap(statistic, blocks, **kwargs)

    def loss(a, b, **kwargs):
        calls.append(kwargs)
        return real_loss(a, b, **kwargs)

    monkeypatch.setattr(analysis_module, "block_bootstrap", bootstrap)
    monkeypatch.setattr(analysis_module, "paired_loss_difference", loss)
    analysis = pre_registered_analysis(odd, serving_results(odd), arm="serving")
    assert len(calls) == 2 * 2 + 4 * 3 + 2 * 2  # estimates (top-k), accuracy (four variants), losses (top-k)
    assert all(call == settings(odd) for call in calls)
    assert analysis.settings == {**settings(odd), "alpha": 0.1, "effect_size": 0.25}
    assert analysis.metric("did_accuracy_gap").cell("ensemble_top_k").interval.confidence == pytest.approx(0.9)
    assert analysis.metric("achieved_power").power["effect_size"] == 0.25


def test_the_plans_frozen_losses_must_be_the_ones_this_analysis_implements():
    other = ScoredProbability(losses={"brier": "abs(p_adj - outcome)", "log_loss": SCORED_LOSSES["log_loss"]})
    changed = plan_with(scored=other)
    with pytest.raises(AnalysisError, match="losses"):
        pre_registered_analysis(changed, [], arm="serving")


# ---------- refusals ----------

def test_the_analysis_refuses_results_and_stages_it_cannot_read(serving, fast):
    results = serving.results
    with pytest.raises(PlanChangedError):
        pre_registered_analysis(PLAN, [*results[:4], fast.results[0]], arm="serving")
    with pytest.raises(PilotError, match="more than once"):
        pre_registered_analysis(PLAN, [*results[:4], results[0]], arm="serving")
    with pytest.raises(PilotError, match="serving-arm result"):
        pre_registered_analysis(PLAN, results[:4], arm="sampling")
    with pytest.raises(AnalysisError, match="pilot"):
        pre_registered_analysis(PLAN, results, arm="serving", stage="pilot")
    with pytest.raises(AnalysisError, match="negative_control"):
        pre_registered_analysis(PLAN, [], arm="serving", stage="negative_control")
    with pytest.raises(AnalysisError, match="arm"):
        pre_registered_analysis(PLAN, [], arm="local")
    with pytest.raises(AnalysisError, match="ensemble"):
        pre_registered_analysis(PLAN, [replace(results[0], min_parsed=1), *results[1:]], arm="serving")
    other = VariantResult("ensemble_full_distribution", "ensemble", "full_distribution", missing="another reason")
    variants = tuple(other if variant.variant == other.variant else variant for variant in results[0].variants)
    with pytest.raises(AnalysisError, match="missing"):
        pre_registered_analysis(PLAN, [replace(results[0], variants=variants), *results[1:]], arm="serving")
    with pytest.raises(TypeError):
        pre_registered_analysis(PLAN, ["not a result"], arm="serving")


# ---------- the sampling arm through the real 7.6 and 7.7 path ----------

@pytest.fixture(scope="module")
def sampled(tmp_path_factory) -> SimpleNamespace:
    """The probe arm's whole full run, calibrated and executed on the 7.7 doubles and recorded under tmp_path."""
    root = tmp_path_factory.mktemp("analysis-sampling")
    ledger = BudgetLedger.open(root / "calibration" / "budget.json", PLAN,
                               StudyLimits(PLAN.plan_sha256, "sampling", "calibration", 1000, 1_000_000, 1_000_000),
                               ALLOWANCE, clock=Clock())
    run_calibration(PLAN, arm="sampling", macro_rows=[macro_row(month) for month in PLAN.months.calibration],
                    edgar_releases=edgar_releases(), client=FakeHost(LedgerCallBudget(ledger)), ledger=ledger,
                    settings=DrawSettings(draws=2, temperature=TEMPERATURE, seed_base=SEED_BASE),
                    sources=("top_k", "full_distribution"), full_distribution=FakeFull(),
                    fit=FitSettings(min_auc=0.6, min_valid=10), calibrators_dir=root / "calibrators",
                    sleep=lambda seconds: None)
    calibrators = load_calibrators(PLAN, arm="sampling", kind="macro", calibrators_dir=root / "calibrators")
    budget = PLAN.limits.stage("sampling", "full_run")
    ledger = BudgetLedger.open(root / "full_run" / "budget.json", PLAN,
                               StudyLimits(PLAN.plan_sha256, "sampling", "full_run", budget.max_requests,
                                           budget.max_output_tokens, budget.max_seconds), ALLOWANCE, clock=Clock())
    run = execute_items(PLAN, macro_items(PLAN, MACRO_ITEMS, stage="full_run", arm="sampling"), arm="sampling",
                        stage="full_run", client=FakeHost(LedgerCallBudget(ledger)), ledger=ledger,
                        calibrators=calibrators, full_distribution=FakeFull(), results_dir=root / "results",
                        sleep=lambda seconds: None)
    assert run.status == "complete"
    return SimpleNamespace(root=root, run=run, results_dir=root / "results",
                           analysis=pre_registered_analysis(PLAN, run.results, arm="sampling"))


def test_the_probe_arm_carries_all_six_variants_on_its_frozen_probe_months(sampled):
    analysis = sampled.analysis
    assert analysis.planned_months == len(PLAN.months.probe_evaluation) == 20
    assert (analysis.pre_cutoff_months, analysis.post_cutoff_months) == (15, 5)
    assert analysis.missing_variants == {} and analysis.missing_results == ()
    for name in ACCURACY_METRICS:
        for cell in analysis.metric(name).variants:
            assert cell.missing is None and cell.months >= PLAN.analysis.block_length
            assert cell.dropped_resamples is not None and 0 <= cell.dropped_resamples < PLAN.analysis.n_resamples
    for name in ESTIMATE_METRICS:
        for variant in ("single_full_distribution", "ensemble_full_distribution", "ensemble_top_k"):
            assert analysis.metric(name).cell(variant).statistic is not None
    for name in LOSS_METRICS:
        assert analysis.metric(name).holm_members == ("single_top_k", "single_full_distribution", "ensemble_top_k",
                                                      "ensemble_full_distribution")
    assert analysis.metric("did_accuracy_gap").holm_members == tuple(VARIANTS)
    # Five post-cutoff months and blocks of six: some resamples hold no post-cutoff month; counted and stated.
    assert all(cell.dropped_resamples > 0 for cell in analysis.metric("did_accuracy_gap").variants)
    assert CONDITIONAL_NOTE in analysis.to_markdown()


def test_replay_from_the_stored_results_with_the_network_disabled_reproduces_the_stored_output(sampled, tmp_path):
    out = tmp_path / "analysis"
    json_path, markdown_path = write_analysis(sampled.analysis, out)
    assert json_path == analysis_path("sampling", "full_run", out) == out / "pre-registered-sampling-full_run.json"
    assert markdown_path == out / "pre-registered-sampling-full_run.md"
    assert not json_path.is_relative_to(ANALYSIS_DIR)
    stored = json_path.read_text(encoding="utf-8"), markdown_path.read_text(encoding="utf-8")
    replayed = analysis_from_records(PLAN, arm="sampling", results_dir=sampled.results_dir)
    assert replayed.to_dict() == sampled.analysis.to_dict() == json.loads(stored[0])
    assert replayed.to_markdown() == stored[1]
    assert write_analysis(replayed, out) == (json_path, markdown_path)  # the same output again is a no-op
    plan_path = write_plan(PLAN, tmp_path / "plan.json")
    again = tmp_path / "again"
    assert main(["--plan", str(plan_path), "--plan-sha256", PLAN.plan_sha256, "--arm", "sampling", "--results-dir",
                 str(sampled.results_dir), "--out-dir", str(again)]) == 0
    assert (again / json_path.name).read_text(encoding="utf-8") == stored[0]
    assert (again / markdown_path.name).read_text(encoding="utf-8") == stored[1]
    assert main(["--plan", str(plan_path), "--plan-sha256", FAST.plan_sha256, "--arm", "sampling", "--results-dir",
                 str(sampled.results_dir), "--out-dir", str(again)]) == 2


def test_an_output_is_written_once_and_never_replaced_by_a_different_one(serving, fast, tmp_path):
    json_path, markdown_path = write_analysis(serving.analysis, tmp_path)
    before = json_path.read_text(encoding="utf-8"), markdown_path.read_text(encoding="utf-8")
    different = pre_registered_analysis(FAST, fast.results, arm="serving")
    with pytest.raises(AnalysisError, match="refusing to replace"):
        write_analysis(different, tmp_path)
    assert (json_path.read_text(encoding="utf-8"), markdown_path.read_text(encoding="utf-8")) == before


def test_the_outputs_carry_derived_statistics_only(serving, tmp_path):
    json_path, markdown_path = write_analysis(serving.analysis, tmp_path)
    texts = [json_path.read_text(encoding="utf-8"), markdown_path.read_text(encoding="utf-8")]
    for text in texts:
        assert "inflation" not in text and "risk_appetite" not in text and "response" not in text  # no reply
        assert "short_treasury_cash" not in text and "world_equity" not in text
        for month in MONTHS:
            if month != MISSING_MONTH:
                returns = realized(month)
                for value in (returns.world_equity, returns.tech_sector, returns.gold_commodity):
                    assert repr(value) not in text  # no realized return
    markdown = texts[1]
    assert markdown.startswith("# Pre-registered analysis: accuracy, losses and multiple testing")
    for name in PLAN.analysis.metrics:
        assert f"## {name}" in markdown
    assert "| Variant | Statistic | Interval | p raw | p Holm |" in markdown
    assert SERVING_FULL_DISTRIBUTION_REASON in markdown and DEFERRED_REASON in markdown
