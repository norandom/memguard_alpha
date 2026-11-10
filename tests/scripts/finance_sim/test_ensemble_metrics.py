"""Pre-registered analysis: ensemble metrics (task 7.11; Requirements 8.5, 8.6 and 8.7).

Pure logic on recorded fixtures: the sampling arm's results come from the real
7.6 calibration and 7.7 execution on the 7.7 test fakes (generation and
full-distribution host doubles), the serving arm's from synthetic temperature-0
draws through ``derive_result`` without a request, and every file is under
``tmp_path``. Sockets are blocked; no network, credentials or model host is
used, and nothing is written under ``runs/``.
"""

from __future__ import annotations

import json
import shutil
import socket
import statistics
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from recall_guard.core.consensus import Tail, wilson_interval
from recall_guard.core.decision import adjust
from recall_guard.core.distribution import RecordedDraw
from recall_guard.core.significance import half_ensemble_variance, holm, random_answerer_null
from recall_guard.dataset.edgar_outlook import outlook_definition, render_outlook_prompt
from recall_guard.dataset.macro_regime import MACRO_AXES, defensive_spread, regime_definition
from recall_guard.harness.guarded_decision import DrawMode, GuardConfig, GuardIdentity, decide_from_draws
from recall_guard.mia.estimate_source import EstimateSource, SourceLabel
from scripts.finance_sim import ensemble_metrics as module
from scripts.finance_sim.calibrate import DrawSettings, EdgarRelease, FitSettings, run_calibration
from scripts.finance_sim.ensemble_metrics import (
    ENSEMBLE_METRICS,
    FIRST_DRAW_RATE,
    HALF_P_MEMORIZED,
    HALF_SUMMARY,
    NO_RESULT,
    RANDOM_ANSWERER_NULL,
    UNDECIDED_FIRST_DRAW,
    ZERO_SINGLE_VARIANCE,
    EnsembleMetricsError,
    ensemble_metrics,
    ensemble_metrics_from_records,
    main,
    metrics_path,
    random_answerer_sampler,
    random_loadings,
    write_metrics,
)
from scripts.finance_sim.execute import (
    EDGAR_NO_OUTCOME,
    ExecutionItem,
    derive_result,
    execute_items,
    load_calibrators,
    load_result,
    macro_items,
    planned_keys,
    result_path,
)
from scripts.finance_sim.limits import BudgetLedger, LedgerCallBudget, StudyLimits
from scripts.finance_sim.pilot import PILOT_EXCLUDED, PilotError
from scripts.finance_sim.plan import (
    ARM_PROFILE_FACTS,
    ARM_REGISTRY_IDS,
    EDGAR_FILES,
    SERVING_FULL_DISTRIBUTION_REASON,
    ArmPlan,
    MonthClass,
    PilotSettings,
    StudyArms,
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
    MONTHS,
    PILOT_DRAWS,
    SEED,
    SEED_BASE,
    TEMPERATURE,
    Clock,
    FakeFull,
    FakeHost,
    digest,
    item_hash,
    macro_row,
    release_text,
)


def frozen_plan() -> StudyPlan:
    """The 7.7 tests' items and split with the arms as frozen (probe 8 draws, primary 32) and a small pilot."""
    months = [MonthClass(month, {GEMMA: month <= "2024-08"}) for month in MONTHS]
    items = {item.item_id: item_hash(item) for item in MACRO_ITEMS} | {row.row_id: digest(row.row_id)
                                                                      for row in EDGAR_ROWS}
    arms = StudyArms(registry_ids=ARM_REGISTRY_IDS,
                     arms={arm: ArmPlan.for_arm(arm, **dict(ARM_PROFILE_FACTS[arm])) for arm in ("sampling", "serving")})
    pilot = replace(PilotSettings.for_split(split_months(months, seed=SEED), arms), draws=PILOT_DRAWS,
                    min_parsed=PILOT_DRAWS)
    return freeze_plan(months, EDGAR_ROWS, seed=SEED, created_at=CREATED_AT, items=items,
                       edgar_files={name: digest(name) for name in EDGAR_FILES}, arms=arms,
                       serving_manifest=load_serving_manifest(), pilot=pilot)


PLAN = frozen_plan()
SAMPLING = PLAN.arms.arms["sampling"]
SERVING = PLAN.arms.arms["serving"]
K = SERVING.draws
H = K // 2
CONFIDENCE = 1 - PLAN.analysis.alpha
#: Loadings with a positive defensive spread (category 1), their mirror (category -1) and a more defensive one.
DEF = {"inflation": 0.5, "growth": -0.5, "credit_stress": 0.5, "policy": 0.0, "risk_appetite": -0.5}
RISK = {axis: -value for axis, value in DEF.items()}
BASE = {"inflation": 0.2, "growth": -0.1, "credit_stress": 0.1, "policy": 0.0, "risk_appetite": -0.2}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("the ensemble analysis attempted a network connection")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)


def shifted(vector: dict, **shifts: float) -> dict:
    return {axis: round(vector[axis] + shifts.get(axis, 0.0), 1) for axis in MACRO_AXES}


def stage_limits(stage: str) -> StudyLimits:
    budget = PLAN.limits.stage("sampling", stage)
    return StudyLimits(plan_sha256=PLAN.plan_sha256, arm="sampling", stage=stage, max_requests=budget.max_requests,
                       max_output_tokens=budget.max_output_tokens, max_seconds=budget.max_seconds)


def ledger_for(root: Path, stage: str) -> BudgetLedger:
    return BudgetLedger.open(root / "records" / stage / "budget.json", PLAN, stage_limits(stage), ALLOWANCE,
                             clock=Clock())


def edgar_releases() -> list[EdgarRelease]:
    releases = [EdgarRelease(row_id, 1, release_text(row_id, 1)) for row_id in PLAN.edgar.calibration_memorized]
    return releases + [EdgarRelease(row_id, 0, release_text(row_id, 0)) for row_id in PLAN.edgar.calibration_control]


def run(root: Path, items, calibrators, stage: str, prose=()):
    ledger = ledger_for(root, stage)
    return execute_items(PLAN, items, arm="sampling", stage=stage, client=FakeHost(LedgerCallBudget(ledger), prose=prose),
                         ledger=ledger, calibrators=calibrators, full_distribution=FakeFull(),
                         results_dir=root / "results", sleep=lambda seconds: None)


@pytest.fixture(scope="module")
def recorded(tmp_path_factory) -> SimpleNamespace:
    """The sampling arm through the real 7.6 calibration and 7.7 execution on the fakes, recorded under tmp_path.

    The whole full run (the probe arm's 20 frozen months, both renderings, 8
    draws each, the first draw of the first ensemble unparsed), the whole pilot
    and 4 of the 40 negative-control releases.
    """
    root = tmp_path_factory.mktemp("ensemble-metrics")
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
    pilot = run(root, macro_items(PLAN, MACRO_ITEMS, stage="pilot"), macro, "pilot")
    releases = tuple(ExecutionItem(row_id, "edgar", "outlook", render_outlook_prompt(release_text(row_id, 0)))
                     for row_id in PLAN.edgar.negative_control[:4])
    negative = run(root, releases, edgar, "negative_control")
    return SimpleNamespace(root=root, results_dir=root / "results", full=full, pilot=pilot, negative=negative)


@pytest.fixture(scope="module")
def sampled(recorded):
    return ensemble_metrics(PLAN, recorded.full.results.values(), arm="sampling")


# ---------- synthetic serving-arm results (temperature 0, 32 draws), derived without a request ----------

def serving_items(count: int) -> tuple[ExecutionItem, ...]:
    return macro_items(PLAN, MACRO_ITEMS, stage="full_run", arm="serving")[:count]


def serving_identity(item: ExecutionItem) -> GuardIdentity:
    return GuardIdentity(model=GEMMA, protocol="openai_compatible", endpoint="https://example.invalid/v1",
                         response_mapping="generic", source=SourceLabel(EstimateSource.TOP_K, 5),
                         settings_json='{"logprobs":true,"max_tokens":256,"temperature":0.0,"top_logprobs":5}',
                         temperature=0.0, draws=K, calibrator_fingerprint="c" * 64, calibrator_is_weak=False,
                         prompt_sha256=item.prompt_sha256)


def draw(index: int, vector: dict | None, estimate: float | None = 0.25) -> RecordedDraw:
    if vector is None:
        return RecordedDraw(index, 0.0, None, digest(""), None, "parse_error", None, None)
    text = json.dumps(vector, sort_keys=True)
    return RecordedDraw(index, 0.0, None, digest(text), dict(vector), None, estimate, "e" * 64)


def serving_result(item: ExecutionItem, vectors, estimates=None):
    estimates = [0.25] * K if estimates is None else estimates
    draws = [draw(index, vector, estimate) for index, (vector, estimate) in enumerate(zip(vectors, estimates,
                                                                                          strict=True))]
    return derive_result(item, draws, None, plan=PLAN, arm="serving", stage="full_run",
                         identities={"top_k": serving_identity(item)})


def half_case(j: int) -> SimpleNamespace:
    """Half a is draw 0 plus 15 copies of m, half b draw 16 plus 15 copies of m2; the singles are draws 0 and 16."""
    m = shifted(BASE, inflation=0.1 * (j % 3))
    m2 = shifted(m, growth=0.1 * (j % 3))
    s0 = shifted(m, credit_stress=0.1 * (1 + j % 2))
    s16 = shifted(m2, policy=-0.1 * (j % 2))
    e0, ea, e16, eb = round(0.10 + 0.01 * j, 4), 0.30, round(0.20 + 0.02 * j, 4), 0.35
    return SimpleNamespace(m=m, m2=m2, s0=s0, s16=s16, e0=e0, e16=e16,
                           vectors=[s0, *[m] * (H - 1), s16, *[m2] * (H - 1)],
                           estimates=[e0, *[ea] * (H - 1), e16, *[eb] * (H - 1)],
                           pa=statistics.fmean([e0, *[ea] * (H - 1)]), pb=statistics.fmean([e16, *[eb] * (H - 1)]))


HALF_ITEMS = serving_items(12)
HALF_CASES = [half_case(j) for j in range(len(HALF_ITEMS))]


@pytest.fixture(scope="module")
def halves():
    results = [serving_result(item, case.vectors, case.estimates) for item, case in zip(HALF_ITEMS, HALF_CASES,
                                                                                         strict=True)]
    return ensemble_metrics(PLAN, results, arm="serving")


#: First-draw constructions over 12 items (6 months x 2 renderings, identifying first).
FIRST_TYPES = ("inside", "flip", "edge", "inside", "first_unparsed", "inside", "failed", "split_inside",
               "split_outside", "inside", "flip", "inside")


def first_vectors(kind: str) -> list:
    if kind == "inside":
        return [DEF] * K
    if kind == "flip":
        return [RISK, *[DEF] * (K - 1)]
    if kind == "edge":
        return [shifted(DEF, inflation=0.3), *[DEF] * (K - 1)]
    if kind == "first_unparsed":
        return [None, *[DEF] * (K - 1)]
    if kind == "failed":
        return [DEF, *[None] * 9, *[DEF] * (K - 10)]
    low = shifted(DEF, inflation=-1.0)
    if kind == "split_inside":
        return [*[DEF] * H, *[low] * H]
    return [shifted(DEF, inflation=0.3), *[DEF] * (H - 1), *[low] * H]  # split_outside


FIRST_ITEMS = serving_items(len(FIRST_TYPES))


@pytest.fixture(scope="module")
def first_draws():
    results = [serving_result(item, first_vectors(kind)) for item, kind in zip(FIRST_ITEMS, FIRST_TYPES, strict=True)]
    return SimpleNamespace(results=results, metrics=ensemble_metrics(PLAN, results, arm="serving"))


def zero_case_vectors(j: int) -> list:
    """Ten items with identical draws, one whose halves differ while its singles agree, one the other way round."""
    if j == 10:
        return [*[BASE] * (H + 1), *[shifted(BASE, growth=0.2)] * (H - 1)]
    if j == 11:
        return [shifted(BASE, credit_stress=0.1), *[BASE] * (K - 1)]
    return [BASE] * K


ZERO_ITEMS = serving_items(12)


@pytest.fixture(scope="module")
def mostly_identical():
    results = [serving_result(item, zero_case_vectors(j)) for j, item in enumerate(ZERO_ITEMS)]
    return ensemble_metrics(PLAN, results, arm="serving")


def entry(metrics, name: str, key: str | None = None) -> dict:
    found = metrics.to_dict()["metrics"][name]
    return found if key is None else found["results"][key]


def expected_test(arrays) -> dict:
    """The significance module's result on explicit arrays, with the plan's resamples, seed and confidence."""
    a, b, singles = (np.asarray(values, dtype=float) for values in arrays)
    result = half_ensemble_variance(a, b, singles, n_resamples=PLAN.analysis.n_resamples, seed=PLAN.seed,
                                    confidence=CONFIDENCE)
    interval = None if result.interval is None else {"lo": result.interval.lo, "hi": result.interval.hi,
                                                     "confidence": result.interval.confidence,
                                                     "method": result.interval.method}
    return {"statistic": result.statistic, "interval": interval, "p_value": result.p_value, "method": result.method,
            "parameters": dict(result.parameters)}


# ---------- names, sizes and the ensemble arm come from the frozen plan ----------

def test_every_metric_name_and_size_comes_from_the_frozen_plan(sampled):
    data = sampled.to_dict()
    assert data["metric_names"] == [name for name in PLAN.analysis.metrics if name in ENSEMBLE_METRICS]
    assert set(data["metrics"]) == set(ENSEMBLE_METRICS)
    assert set(ENSEMBLE_METRICS) == {HALF_SUMMARY, HALF_P_MEMORIZED, FIRST_DRAW_RATE, RANDOM_ANSWERER_NULL}
    assert set(ENSEMBLE_METRICS) <= set(PLAN.analysis.metrics)
    settings = data["settings"]
    assert (settings["draws"], settings["min_parsed"]) == (SAMPLING.draws, SAMPLING.min_parsed) == (8, 6)
    assert (settings["half_draws"], settings["half_min_parsed"]) == (4, 3)  # ceil(6 x 4 / 8)
    assert (settings["first_half"], settings["second_half"], settings["single_draws"]) == ([0, 3], [4, 7], [0, 4])
    assert settings["first_draw_band"] == list(PLAN.analysis.first_draw_band) == [0.25, 0.75]
    assert (settings["n_resamples"], settings["seed"], settings["confidence"]) == (
        PLAN.analysis.n_resamples, PLAN.seed, CONFIDENCE)
    serving = entry(ensemble_metrics(PLAN, [], arm="serving"), HALF_SUMMARY)
    assert serving["arm"] == "serving"
    assert ensemble_metrics(PLAN, [], arm="serving").to_dict()["settings"]["half_draws"] == 16


def test_each_result_names_its_ensemble_arm(sampled, halves):
    for metrics, arm in ((sampled, "sampling"), (halves, "serving")):
        data = metrics.to_dict()
        frozen = PLAN.arms.arms[arm]
        assert data["arm"] == arm and data["plan_sha256"] == PLAN.plan_sha256
        assert data["ensemble_arm"] == {
            "arm": arm, "role": frozen.role, "measures": frozen.measures, "model": PLAN.arms.registry_ids[arm],
            "served_model": frozen.served_model, "temperature": frozen.temperature, "seed_base": frozen.seed_base,
            "draws": frozen.draws, "min_parsed": frozen.min_parsed}
        for name, found in data["metrics"].items():
            assert (found["metric"], found["arm"], found["stage"]) == (name, arm, "full_run")
            for nested in found.get("results", {}).values():
                assert (nested["metric"], nested["arm"]) == (name, arm)
        assert f"{arm} arm" in metrics.to_markdown()


# ---------- half-ensemble versus single-draw variance (8.5) ----------

def test_the_halves_and_singles_are_re_derived_from_the_recorded_draws_of_one_ensemble(halves):
    none = entry(halves, HALF_SUMMARY, "none")
    expected = ([defensive_spread(case.m) for case in HALF_CASES], [defensive_spread(case.m2) for case in HALF_CASES],
                [[defensive_spread(case.s0), defensive_spread(case.s16)] for case in HALF_CASES])
    assert [[row["half_a"], row["half_b"], row["single_a"], row["single_b"]] for row in none["items"]] == [
        [a, b, *singles] for a, b, singles in zip(*expected, strict=True)]
    assert none["test"] == expected_test(expected)
    assert none["variants"] == ["ensemble_none", "single_none"] and none["missing"] is None
    assert none["n_items"] == len(HALF_ITEMS) and none["missing_items"] == []
    top_k = entry(halves, HALF_SUMMARY, "top_k")
    expected_top_k = (
        [defensive_spread(adjust(case.m, case.pa)) for case in HALF_CASES],
        [defensive_spread(adjust(case.m2, case.pb)) for case in HALF_CASES],
        [[defensive_spread(adjust(case.s0, case.e0)), defensive_spread(adjust(case.s16, case.e16))]
         for case in HALF_CASES])
    assert top_k["test"] == expected_test(expected_top_k)
    estimates = entry(halves, HALF_P_MEMORIZED, "top_k")
    expected_estimates = ([case.pa for case in HALF_CASES], [case.pb for case in HALF_CASES],
                          [[case.e0, case.e16] for case in HALF_CASES])
    assert estimates["test"] == expected_test(expected_estimates)
    assert estimates["variants"] == ["ensemble_top_k", "single_top_k"]


def test_the_serving_arms_full_distribution_cells_are_missing_with_the_plans_reason(halves):
    for name in (HALF_SUMMARY, HALF_P_MEMORIZED):
        cell = entry(halves, name, "full_distribution")
        assert cell["missing"] == SERVING_FULL_DISTRIBUTION_REASON
        assert (cell["test"], cell["holm_p_value"], cell["items"]) == (None, None, [])
    assert set(entry(halves, HALF_SUMMARY)["results"]) == {"none", "top_k", "full_distribution"}
    assert set(entry(halves, HALF_P_MEMORIZED)["results"]) == {"top_k", "full_distribution"}


def test_holm_adjusts_each_family_across_its_computed_cells(sampled, halves):
    for metrics in (sampled, halves):
        for name in (HALF_SUMMARY, HALF_P_MEMORIZED):
            cells = entry(metrics, name)["results"]
            raw = {key: cell["test"]["p_value"] for key, cell in cells.items() if cell["test"] is not None}
            assert raw
            adjusted = holm(raw)
            assert {key: cell["holm_p_value"] for key, cell in cells.items() if cell["test"] is not None} == adjusted


def test_mostly_identical_items_are_checked_up_front_and_unbounded_resamples_are_reported(mostly_identical):
    none = entry(mostly_identical, HALF_SUMMARY, "none")
    n = len(ZERO_ITEMS)
    assert (none["identical_singles"], none["identical_items"]) == (11, 10)
    assert none["zero_single_resample_probability"] == (11 / n) ** n
    assert none["zero_over_zero_resample_probability"] == (10 / n) ** n
    spreads = [[defensive_spread(vectors[1]), defensive_spread(vectors[K - 1]), defensive_spread(vectors[0]),
                defensive_spread(vectors[H])] for vectors in map(zero_case_vectors, range(n))]
    expected = expected_test(([row[0] for row in spreads], [row[1] for row in spreads], [row[2:] for row in spreads]))
    assert none["test"] == expected
    assert expected["interval"] is None and expected["parameters"]["unbounded_resamples"] > 0
    assert expected["parameters"]["dropped_resamples"] > 0
    assert (none["interval_note"], none["unbounded_resamples"], none["dropped_resamples"]) == (
        expected["parameters"]["interval_note"], expected["parameters"]["unbounded_resamples"],
        expected["parameters"]["dropped_resamples"])
    assert none["interval_note"] == "upper bound unbounded"
    assert "upper bound unbounded" in mostly_identical.to_markdown()


def test_a_zero_single_draw_variance_is_reported_missing_with_the_reason_never_raised(mostly_identical):
    estimates = entry(mostly_identical, HALF_P_MEMORIZED, "top_k")  # every draw carries the estimate 0.25
    assert estimates["missing"] == ZERO_SINGLE_VARIANCE
    assert (estimates["test"], estimates["holm_p_value"]) == (None, None)
    assert (estimates["identical_singles"], estimates["identical_items"]) == (12, 12)
    assert estimates["zero_single_resample_probability"] == 1.0
    assert ZERO_SINGLE_VARIANCE in mostly_identical.to_markdown()


def test_the_sampling_arm_reduces_halves_of_four_draws_with_the_scaled_minimum(recorded, sampled):
    """Spot check on the real execution path: one item's halves and singles through decide_from_draws."""
    key = next(key for key in planned_keys(PLAN, "sampling", "full_run")
               if recorded.full.results[key].recorded_draws[0].parsed is not None)
    result = recorded.full.results[key]
    identity = result.identities["full_distribution"]

    def reduce(start: int, stop: int):
        size = stop - start
        config = (GuardConfig(regime_definition, identity.source, draws=1, min_parsed=1, mode=DrawMode.SINGLE)
                  if size == 1 else GuardConfig(regime_definition, identity.source, draws=size, min_parsed=3))
        decision = decide_from_draws(result.draws_for("full_distribution")[start:stop], config,
                                     replace(identity, draws=size))
        return decision.p_memorized

    cell = entry(sampled, HALF_P_MEMORIZED, "full_distribution")
    row = next(row for row in cell["items"] if (row["item_id"], row["rendering"]) == key)
    assert [row["half_a"], row["half_b"], row["single_a"], row["single_b"]] == [
        reduce(0, 4), reduce(4, 8), reduce(0, 1), reduce(4, 5)]
    for name, keys in ((HALF_SUMMARY, ("none", "top_k", "full_distribution")),
                       (HALF_P_MEMORIZED, ("top_k", "full_distribution"))):
        for adjustment in keys:
            cell = entry(sampled, name, adjustment)
            rows = cell["items"]
            assert cell["test"] == expected_test(([row["half_a"] for row in rows], [row["half_b"] for row in rows],
                                                [[row["single_a"], row["single_b"]] for row in rows]))
            first = next(item for item in cell["missing_items"]
                         if (item["item_id"], item["rendering"]) == planned_keys(PLAN, "sampling", "full_run")[0])
            assert "single draw 0" in first["reason"]  # its first draw did not parse: missing, never substituted
            assert cell["n_items"] + len(cell["missing_items"]) == len(recorded.full.results)


# ---------- first draw versus consensus (8.6) ----------

def test_the_first_draw_rate_counts_category_flips_and_ranks_outside_the_frozen_band(first_draws):
    rate = entry(first_draws.metrics, FIRST_DRAW_RATE)
    assert rate["band"] == [0.25, 0.75] and rate["variant"] == "ensemble_none"
    overall = rate["overall"]
    assert (overall["n"], overall["off"], overall["category_differs"], overall["outside_band"]) == (9, 4, 2, 4)
    assert overall["rate"] == 4 / 9
    lo, hi = wilson_interval(4, 9, confidence=CONFIDENCE, tail=Tail.TWO_SIDED)
    assert (overall["interval"]["lo"], overall["interval"]["hi"], overall["interval"]["confidence"]) == (lo, hi,
                                                                                                         CONFIDENCE)
    assert "Wilson" in overall["interval"]["method"]
    by_rendering = rate["by_rendering"]
    assert (by_rendering["identifying"]["n"], by_rendering["identifying"]["off"]) == (4, 3)
    assert (by_rendering["anonymised"]["n"], by_rendering["anonymised"]["off"]) == (5, 1)
    rows = {(row["item_id"], row["rendering"]): row for row in rate["items"]}
    for item, kind in zip(FIRST_ITEMS, FIRST_TYPES, strict=True):
        if kind == "flip":
            assert rows[item.key]["category_differs"] is True and rows[item.key]["first_draw_rank"] == 0.5 / K
        if kind == "edge":
            assert rows[item.key]["category_differs"] is False and rows[item.key]["outside_band"] is True
        if kind == "inside":
            assert rows[item.key]["off"] is False and rows[item.key]["first_draw_rank"] == 0.5
        if kind == "split_outside":
            assert rows[item.key]["consensus_category"] is None and rows[item.key]["off"] is True


def test_first_draws_without_a_decidable_answer_are_missing_with_their_reasons(first_draws):
    rate = entry(first_draws.metrics, FIRST_DRAW_RATE)
    missing = {(row["item_id"], row["rendering"]): row["reason"] for row in rate["missing_items"]}
    by_kind = dict(zip(FIRST_TYPES, FIRST_ITEMS, strict=True))
    assert set(missing) == {by_kind[kind].key for kind in ("first_unparsed", "failed", "split_inside")}
    assert "first draw did not parse" in missing[by_kind["first_unparsed"].key]
    assert "parse_error" in missing[by_kind["first_unparsed"].key]
    assert missing[by_kind["failed"].key].startswith("no consensus: 23 of 32 draws parsed")
    assert missing[by_kind["split_inside"].key] == UNDECIDED_FIRST_DRAW


def test_the_first_draw_rate_of_the_sampling_arm_is_per_control_and_rendering(sampled, recorded):
    rate = entry(sampled, FIRST_DRAW_RATE)
    assert rate["control"] == sampled.to_dict()["control"] and "positive control" in rate["control"]
    assert set(rate["by_rendering"]) == {"identifying", "anonymised"}
    assert rate["overall"]["n"] == sum(group["n"] for group in rate["by_rendering"].values())
    assert rate["overall"]["n"] + len(rate["missing_items"]) == len(recorded.full.results)
    first = planned_keys(PLAN, "sampling", "full_run")[0]
    assert any((row["item_id"], row["rendering"]) == first for row in rate["missing_items"])
    for row in rate["items"]:
        result = recorded.full.results[(row["item_id"], row["rendering"])]
        assert row["first_category"] == result.categories[0]
        assert row["consensus_category"] == result.variant("ensemble_none").category
        assert row["first_draw_rank"] == result.variant("ensemble_none").first_draw_rank


# ---------- random-answerer null (8.7) ----------

def test_random_loadings_are_uniform_on_the_definitions_grid_and_pass_through_its_decision():
    rng = np.random.default_rng(7)
    loadings = random_loadings(regime_definition, rng, 4000)
    assert loadings.shape == (4000, len(MACRO_AXES))
    grid = [round(-1.0 + 0.1 * step, 1) for step in range(21)]
    assert set(np.unique(loadings).tolist()) == set(grid)
    counts = np.array([np.count_nonzero(loadings == value) for value in grid])
    assert counts.min() > 0.7 * counts.mean() and counts.max() < 1.3 * counts.mean()
    decisions = random_answerer_sampler(regime_definition)(np.random.default_rng(7), 4000)
    expected = [regime_definition.decide(dict(zip(MACRO_AXES, row, strict=True))) for row in loadings.tolist()]
    assert decisions.tolist() == expected and set(expected) == {-1, 1}
    outlooks = random_loadings(outlook_definition, np.random.default_rng(3), 500)
    assert outlooks.shape == (500, 1) and set(np.unique(outlooks).tolist()) <= set(grid)


def test_the_random_answerer_null_tests_the_consensus_accuracy_with_the_frozen_settings(sampled, recorded):
    null = entry(sampled, RANDOM_ANSWERER_NULL)
    assert null["missing"] is None and null["variant"] == "ensemble_none"
    rows = null["items"]
    stored = {key: result.variant("ensemble_none") for key, result in recorded.full.results.items()}
    for row in rows:
        assert row["correct"] == stored[(row["item_id"], row["rendering"])].correct
    assert null["n_items"] == len(rows) and null["n_items"] + len(null["missing_items"]) == len(stored)
    observed = np.array([row["category"] for row in rows])
    outcomes = np.array([recorded.full.results[(row["item_id"], row["rendering"])].outcome for row in rows])

    def accuracy(answers: np.ndarray) -> float:
        return float(np.mean((np.asarray(answers) == 1) == (outcomes == 1)))

    expected = random_answerer_null(observed, random_answerer_sampler(regime_definition), statistic=accuracy,
                                    n_null=PLAN.analysis.n_resamples, seed=PLAN.seed, confidence=CONFIDENCE)
    assert null["test"]["statistic"] == expected.statistic == statistics.fmean(row["correct"] for row in rows)
    assert (null["test"]["p_value"], null["test"]["parameters"]) == (expected.p_value, dict(expected.parameters))
    assert (null["test"]["interval"]["lo"], null["test"]["interval"]["hi"]) == (expected.interval.lo,
                                                                                expected.interval.hi)
    missing = {(row["item_id"], row["rendering"]): row["reason"] for row in null["missing_items"]}
    for key, reason in missing.items():
        variant = stored[key]
        assert variant.correct is None or variant.category is None
        assert reason  # with its reason, never substituted


def test_the_null_lists_items_without_a_consensus_category_or_outcome_as_missing(first_draws):
    null = entry(first_draws.metrics, RANDOM_ANSWERER_NULL)
    missing = {(row["item_id"], row["rendering"]): row["reason"] for row in null["missing_items"]}
    by_kind = dict(zip(FIRST_TYPES, FIRST_ITEMS, strict=True))
    assert missing[by_kind["failed"].key].startswith("no consensus:")
    assert "unresolved" in missing[by_kind["split_inside"].key]
    assert by_kind["first_unparsed"].key not in missing  # the consensus does not need the first draw
    assert null["n_items"] == len(FIRST_ITEMS) - 3


def test_the_negative_control_has_no_outcome_so_its_null_is_missing_with_the_reason(recorded):
    metrics = ensemble_metrics(PLAN, recorded.negative.results.values(), arm="sampling", stage="negative_control")
    null = entry(metrics, RANDOM_ANSWERER_NULL)
    assert (null["missing"], null["test"], null["stage"]) == (EDGAR_NO_OUTCOME, None, "negative_control")
    data = metrics.to_dict()
    assert data["kind"] == "edgar" and "negative control" in data["control"]
    rate = entry(metrics, FIRST_DRAW_RATE)
    assert set(rate["by_rendering"]) == {"outlook"} and rate["overall"]["n"] + len(rate["missing_items"]) == 4


# ---------- pilot items, missing results and refusals ----------

def test_pilot_items_are_excluded_and_never_feed_the_half_ensemble_statistics(recorded, sampled):
    mixed = [*recorded.full.results.values(), *recorded.pilot.results.values()]
    with_pilot = ensemble_metrics(PLAN, mixed, arm="sampling").to_dict()
    excluded = {(row["item_id"], row["rendering"]): row["reason"] for row in with_pilot["excluded"]}
    assert set(excluded) == set(recorded.pilot.results) and set(excluded.values()) == {PILOT_EXCLUDED}
    without = sampled.to_dict()
    assert without["excluded"] == []
    assert with_pilot["metrics"] == without["metrics"]
    pilot_ids = {f"macro-{month}" for month in PLAN.months.pilot}
    for name in (HALF_SUMMARY, HALF_P_MEMORIZED):
        for cell in with_pilot["metrics"][name]["results"].values():
            assert not any(row["item_id"] in pilot_ids for row in [*cell["items"], *cell["missing_items"]])
    with pytest.raises(EnsembleMetricsError, match="enters no later analysis"):
        ensemble_metrics(PLAN, recorded.pilot.results.values(), arm="sampling", stage="pilot")


def test_planned_items_without_a_stored_result_are_missing_with_the_reason(recorded):
    metrics = ensemble_metrics_from_records(PLAN, arm="sampling", stage="negative_control",
                                            results_dir=recorded.results_dir).to_dict()
    planned = planned_keys(PLAN, "sampling", "negative_control")
    missing = {(row["item_id"], row["rendering"]): row["reason"] for row in metrics["missing_results"]}
    assert set(missing) == set(planned) - set(recorded.negative.results) and len(missing) == 36
    assert set(missing.values()) == {NO_RESULT}
    assert [tuple(row.values()) for row in metrics["items"]] == [key for key in planned
                                                                 if key in recorded.negative.results]


def test_results_of_another_arm_or_a_stored_result_that_does_not_recompute_are_refused(recorded, tmp_path, halves):
    serving = serving_result(HALF_ITEMS[0], HALF_CASES[0].vectors, HALF_CASES[0].estimates)
    with pytest.raises(PilotError, match="serving-arm result"):
        ensemble_metrics(PLAN, [*recorded.full.results.values(), serving], arm="sampling")
    key = planned_keys(PLAN, "sampling", "full_run")[1]
    source = result_path("sampling", "full_run", *key, recorded.results_dir)
    for edit in (lambda data: data["variants"][4].update(p_memorized=data["variants"][4]["p_memorized"] + 0.01),
                 lambda data: data["variants"][1].update(category=-data["variants"][1]["category"]),
                 lambda data: data["summary_values"].__setitem__(2, data["summary_values"][2] + 0.1)):
        data = json.loads(source.read_text(encoding="utf-8"))
        edit(data)
        target = tmp_path / "tampered.json"
        target.write_text(json.dumps(data), encoding="utf-8")
        tampered = load_result(target)
        with pytest.raises(EnsembleMetricsError, match="does not recompute"):
            ensemble_metrics(PLAN, [tampered], arm="sampling")
    with pytest.raises(EnsembleMetricsError, match="stage"):
        ensemble_metrics(PLAN, [], arm="sampling", stage="calibration")


# ---------- replay, written outputs and the command ----------

def test_replayed_from_the_recorded_results_with_network_disabled_it_reproduces_the_stored_outputs(
        recorded, sampled, tmp_path):
    out = tmp_path / "analysis"
    replayed = ensemble_metrics_from_records(PLAN, arm="sampling", stage="full_run", results_dir=recorded.results_dir)
    assert replayed.to_dict() == sampled.to_dict()
    json_path, markdown_path = write_metrics(replayed, out)
    assert (json_path, markdown_path) == (metrics_path("sampling", "full_run", out),
                                          metrics_path("sampling", "full_run", out).with_suffix(".md"))
    assert json_path.name == "ensemble-sampling-full_run.json"
    texts = (json_path.read_text(encoding="utf-8"), markdown_path.read_text(encoding="utf-8"))
    assert json.loads(texts[0]) == sampled.to_dict()
    again = ensemble_metrics_from_records(PLAN, arm="sampling", stage="full_run", results_dir=recorded.results_dir)
    assert write_metrics(again, out) == (json_path, markdown_path)  # the same outputs: a no-op
    assert (json_path.read_text(encoding="utf-8"), markdown_path.read_text(encoding="utf-8")) == texts
    copy = tmp_path / "copy"
    shutil.copytree(recorded.results_dir / "sampling" / "full_run", copy / "sampling" / "full_run")
    (copy / "sampling" / "full_run" / f"{planned_keys(PLAN, 'sampling', 'full_run')[-1][0]}.anonymised.json").unlink()
    different = ensemble_metrics_from_records(PLAN, arm="sampling", stage="full_run", results_dir=copy)
    with pytest.raises(EnsembleMetricsError, match="refusing to replace"):
        write_metrics(different, out)
    assert (json_path.read_text(encoding="utf-8"), markdown_path.read_text(encoding="utf-8")) == texts


def test_the_markdown_table_carries_derived_statistics_only(sampled, recorded):
    text = sampled.to_markdown()
    for name in ENSEMBLE_METRICS:
        assert name in text
    assert "|" in text and PLAN.plan_sha256 in text
    assert "macro-20" not in text and '"inflation"' not in text  # no item ids, no replies
    assert "\u2014" not in text
    data = json.dumps(sampled.to_dict())
    assert "regime looks benign" not in data and '"inflation": ' not in data and "Press release" not in data


def test_the_command_writes_the_outputs_of_the_frozen_plan_once_and_refuses_another_plan(recorded, sampled, tmp_path):
    plan_file = write_plan(PLAN, tmp_path / "plan.json")
    out = tmp_path / "analysis"
    common = ["--plan", str(plan_file), "--arm", "sampling", "--stage", "full_run",
              "--results-dir", str(recorded.results_dir), "--out-dir", str(out)]
    assert main([*common, "--plan-sha256", "0" * 64]) == 2
    assert not out.exists()
    assert main([*common, "--plan-sha256", PLAN.plan_sha256]) == 0
    written = json.loads(metrics_path("sampling", "full_run", out).read_text(encoding="utf-8"))
    assert written == sampled.to_dict()
    assert main([*common, "--plan-sha256", PLAN.plan_sha256]) == 0  # written identically: a no-op
    assert module.EXIT_WRITTEN == 0 and module.EXIT_REFUSED == 2
