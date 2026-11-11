"""Logprobs against logits on the same weights and prompts (qualified-model-catalogue task 3.1).

Requirements 4.3 to 4.7 and 6.1, 6.2, 6.3, 6.6, 6.9, 6.11. Recorded fixtures
only: a small frozen plan is calibrated in both estimate sources and a few
evaluation months are executed once, on the fakes of the convergence tests
(a generation-host double that declares its weights digest and a
full-distribution double). The calibration draws, the execution draws, the
evidence batches, the results and the convergence table are written under
``tmp_path``; the comparison then reads only those records. Sockets are
blocked and nothing under ``runs/`` is used.
"""

from __future__ import annotations

import inspect
import json
import socket
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.finance_sim.mode_comparison as mode_comparison_module
from recall_guard.core.local_evidence import LocalModelIdentity, LocalModelProvenance
from recall_guard.core.request_control import RequestControl
from recall_guard.core.significance import (
    AGREEMENT_STATISTICS,
    Statistic,
    agreement,
    clustered_auc_interval,
    paired_auc_difference,
    permutation_auc_test,
    rank_auc,
)
from scripts.finance_sim.calibrate import (
    CalibrationDraw,
    DrawSettings,
    EvidenceSettings,
    FitSettings,
    run_calibration,
)
from scripts.finance_sim.convergence import HOST_TOP_K, convergence_from_records, write_table
from scripts.finance_sim.execute import (
    ExecutionDraw,
    VariantResult,
    execute_items,
    load_calibrators,
    macro_items,
    planned_keys,
)
from scripts.finance_sim.limits import BudgetLedger, LedgerCallBudget, StudyLimits, budget_path
from scripts.finance_sim.mode_comparison import (
    AGREEMENT_MARGIN,
    AUC_MARGIN,
    ENSEMBLE_VARIANTS,
    IMPROVE,
    LOGITS,
    LOGPROBS,
    MATCH,
    UNDECIDED,
    ModeComparison,
    ModeComparisonError,
    comparison_path,
    mode_comparison,
    mode_comparison_from_records,
    write_comparison,
)
from scripts.finance_sim.recording import DrawLog, draw_log_path
from tests.scripts.finance_sim.test_convergence import (
    ALLOWANCE,
    DIGEST,
    EXECUTED_MONTHS,
    KEEP,
    MACRO_ITEMS,
    PLAN,
    SEED_BASE,
    TEMPERATURE,
    Clock,
    FakeFull,
    FakeHost,
    edgar_releases,
    macro_row,
    write_evidence_file,
)
from tests.scripts.finance_sim.test_recording_draws import CUT, aged, unchanged, writing_calls

OTHER_DIGEST = "sha256:" + "0" * 64
CALIBRATION_DRAWS = 2
#: Calibration calls answered with prose: the second draw of every fourth macro prompt, so the held-out rows
#: are not the same number per prompt and the per-prompt membership must come from the calibration draws.
CALIBRATION_PROSE = tuple(range(3, 200, 8))


class StampedHost(FakeHost):
    """The convergence tests' generation-host double, declaring the weights it serves."""

    def __init__(self, budget, *, prose=(), weights: str = DIGEST) -> None:
        super().__init__(budget, prose=prose)
        identity = LocalModelIdentity(declared=LocalModelProvenance(artifact_digest=weights))
        self.config = replace(self.config, local_identity=identity)
        self._control = RequestControl(self.config, budget, clock=Clock())


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("the mode comparison attempted a network connection")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)


@pytest.fixture(scope="module")
def recorded(tmp_path_factory) -> SimpleNamespace:
    """Calibrate in both sources, execute a few evaluation months and write the convergence table."""
    root = tmp_path_factory.mktemp("mode-comparison")
    records_dir, results_dir = root / "records", root / "results"
    calibrators_dir, analysis_dir = root / "calibrators", root / "analysis"
    ledger = BudgetLedger.open(root / "calibration" / "budget.json", PLAN,
                               StudyLimits(PLAN.plan_sha256, "sampling", "calibration", 1000, 1_000_000, 1_000_000),
                               ALLOWANCE, clock=Clock())
    calibration_log = DrawLog(draw_log_path("sampling", "calibration", records_dir), plan=PLAN,
                              loads=CalibrationDraw.from_dict)
    run_calibration(PLAN, arm="sampling", macro_rows=[macro_row(month) for month in PLAN.months.calibration],
                    edgar_releases=edgar_releases(),
                    client=StampedHost(LedgerCallBudget(ledger), prose=CALIBRATION_PROSE), ledger=ledger,
                    settings=DrawSettings(draws=CALIBRATION_DRAWS, temperature=TEMPERATURE, seed_base=SEED_BASE),
                    sources=("top_k", "full_distribution"), full_distribution=FakeFull(),
                    fit=FitSettings(min_auc=0.6, min_valid=10), evidence=EvidenceSettings(top_k_keep=KEEP),
                    on_draw=calibration_log, calibrators_dir=calibrators_dir, sleep=lambda seconds: None)
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
    run = execute_items(PLAN, items, arm="sampling", stage="full_run",
                        client=StampedHost(LedgerCallBudget(stage_ledger)), ledger=stage_ledger,
                        calibrators=calibrators, full_distribution=FakeFull(), on_draw=log,
                        on_evidence=lambda item, found: write_evidence_file(records_dir, item, found),
                        results_dir=results_dir, sleep=lambda seconds: None)
    table = convergence_from_records(PLAN, arm="sampling", stage="full_run", results_dir=results_dir,
                                     records_dir=records_dir, calibrators_dir=calibrators_dir)
    write_table(table, analysis_dir)
    edgar = load_calibrators(PLAN, arm="sampling", kind="edgar", calibrators_dir=calibrators_dir)
    return SimpleNamespace(
        edgar_logprobs=edgar["top_k"], edgar_logits=edgar["full_distribution"],
        root=root, records_dir=records_dir, results_dir=results_dir, calibrators_dir=calibrators_dir,
        analysis_dir=analysis_dir, logprobs=calibrators["top_k"], logits=calibrators["full_distribution"],
        results=tuple(run.results[item.key] for item in items), draws=tuple(log.latest()),
        calibration_draws=tuple(calibration_log.latest()), convergence=table.to_dict())


def compare(recorded, **overrides) -> ModeComparison:
    values = {"arm": "sampling", "logprobs": recorded.logprobs, "logits": recorded.logits,
              "results": recorded.results, "calibration_draws": recorded.calibration_draws,
              "convergence": recorded.convergence}
    values |= overrides
    return mode_comparison(PLAN, **values)


def by_name(comparison: ModeComparison) -> dict[str, Statistic]:
    return {statistic.name: statistic for statistic in comparison.statistics}


# ---------- brute force on the same records ----------

def held_out(record, draws) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scores, labels and prompt index of the held-out rows: each held-out prompt with its usable draws."""
    usable: dict[str, int] = {}
    for draw in draws:
        if (draw.kind == record.kind and draw.failure_code is None
                and draw.features.get(record.source.value) is not None):
            usable[draw.prompt_sha256] = usable.get(draw.prompt_sha256, 0) + 1
    clusters = [index for index, prompt in enumerate(record.holdout_prompts) for _ in range(usable[prompt])]
    mcs = record.calibrator.mcs
    return np.asarray(mcs.holdout_probabilities), np.asarray(mcs.holdout_labels), np.asarray(clusters)


def with_scores(record, scores):
    mcs = replace(record.calibrator.mcs, holdout_probabilities=tuple(float(value) for value in scores))
    return replace(record, calibrator=replace(record.calibrator, mcs=mcs))


def with_weights(record, weights: str | None):
    local = None if weights is None else LocalModelIdentity(declared=LocalModelProvenance(artifact_digest=weights))
    identity = replace(record.calibrator.identity, local_identity=local)
    return replace(record, calibrator=replace(record.calibrator, identity=identity))


def labels_of(record) -> np.ndarray:
    return np.asarray(record.calibrator.mcs.holdout_labels)


def separating(record) -> np.ndarray:
    labels = labels_of(record)
    return 0.25 + 0.5 * labels + 0.001 * np.arange(labels.size) / labels.size


def inverted(record) -> np.ndarray:
    labels = labels_of(record)
    return 0.6 - 0.3 * labels + np.random.default_rng(7).uniform(-0.05, 0.05, labels.size)


def noise(record, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).uniform(0.05, 0.95, labels_of(record).size)


def near_results(results, shift: float = 0.01):
    """The stored results with every full-distribution estimate ``shift`` above the top-k estimate of its draw."""
    out = []
    for result in results:
        full = tuple(entry if entry is None or draw.estimate is None
                     else replace(entry, estimate=min(1.0, draw.estimate + shift))
                     for draw, entry in zip(result.recorded_draws, result.full_distribution, strict=True))
        out.append(replace(result, full_distribution=full))
    return tuple(out)


def month_shifted(results, shifts):
    """The stored results with the full-distribution estimates of the i-th month ``shifts[i]`` above the top-k ones."""
    months = list(dict.fromkeys(result.item_id for result in results))
    assert len(months) == len(shifts)
    return tuple(near_results([result], shifts[months.index(result.item_id)])[0] for result in results)


def without_shared_draws(result):
    """``result`` with no draw that carries both estimates: its full-distribution estimates are absent."""
    return replace(result, full_distribution=tuple(None for _ in result.recorded_draws))


def prompt_rows(results) -> tuple[list[float], list[float], list[str], int]:
    """Per prompt (item and rendering): the mean estimate of each mode over the draws that carry both, its month,
    and the number of such draws over all prompts."""
    full, top, months, n_draws = [], [], [], 0
    for result in results:
        pairs = [(entry.estimate, draw.estimate)
                 for draw, entry in zip(result.recorded_draws, result.full_distribution, strict=True)
                 if draw.estimate is not None and entry is not None and entry.estimate is not None]
        if pairs:
            full.append(float(np.mean([pair[0] for pair in pairs])))
            top.append(float(np.mean([pair[1] for pair in pairs])))
            months.append(result.item_id)
            n_draws += len(pairs)
    return full, top, months, n_draws


def with_ensemble(result, mode: str, **changes):
    """``result`` with fields of the ensemble decision of ``mode`` replaced."""
    name = ENSEMBLE_VARIANTS[mode]
    return replace(result, variants=tuple(replace(variant, **changes) if variant.variant == name else variant
                                          for variant in result.variants))


def without_ensemble_decision(result, mode: str):
    """``result`` whose ensemble variant of ``mode`` was not computed: it carries only the reason."""
    name = ENSEMBLE_VARIANTS[mode]
    return replace(result, variants=tuple(
        VariantResult(name, variant.mode, variant.adjustment, missing="not computed for this test")
        if variant.variant == name else variant for variant in result.variants))


def with_ensemble_estimates(results):
    """The stored results with ensemble estimates that are not the per-prompt means of the per-draw estimates."""
    out = []
    for index, result in enumerate(results):
        result = with_ensemble(result, LOGITS, p_memorized=0.20 + 0.05 * index)
        out.append(with_ensemble(result, LOGPROBS, p_memorized=0.55 - 0.03 * index))
    return tuple(out)


def ensemble_rows(results) -> tuple[list[float], list[float], list[str]]:
    """Per item with an applied ensemble estimate in both modes: 1 - estimate of each mode, and its month."""
    logits, logprobs, months = [], [], []
    for result in results:
        a = result.variant(ENSEMBLE_VARIANTS[LOGITS])
        b = result.variant(ENSEMBLE_VARIANTS[LOGPROBS])
        applied = "adjustment_skipped_weak_calibrator" not in (a.status, b.status)
        if (a.missing is None and b.missing is None and a.p_memorized is not None and b.p_memorized is not None
                and applied):
            logits.append(1.0 - a.p_memorized)
            logprobs.append(1.0 - b.p_memorized)
            months.append(result.item_id)
    return logits, logprobs, months


CORRECTION_NAMES = ("correction_factor_difference", f"correction_factor_mean_{LOGITS}",
                    f"correction_factor_mean_{LOGPROBS}")


# ---------- the fixture is what the tests assume ----------

def test_the_fixture_holds_out_the_same_prompts_and_draws_with_uneven_rows_per_prompt(recorded):
    assert recorded.logprobs.holdout_prompts == recorded.logits.holdout_prompts
    rows = len(recorded.logprobs.calibrator.mcs.holdout_labels)
    assert rows == len(recorded.logits.calibrator.mcs.holdout_labels)
    assert rows != len(recorded.logprobs.holdout_prompts) * CALIBRATION_DRAWS
    full, _, months, n_draws = prompt_rows(recorded.results)
    # Two renderings per month, and more draws than prompts: a prompt is not a draw and not a month.
    assert len(full) == len(recorded.results) == 2 * len(set(months)) and n_draws > len(full)
    # Every stored item carries an ensemble decision with an estimate in both modes.
    assert ENSEMBLE_VARIANTS == {LOGPROBS: "ensemble_top_k", LOGITS: "ensemble_full_distribution"}
    assert len(ensemble_rows(recorded.results)[0]) == len(recorded.results)


# ---------- held-out quality per mode (4.3, 6.1 to 6.3) ----------

@pytest.mark.parametrize("mode", [LOGPROBS, LOGITS])
def test_each_mode_reports_its_held_out_auc_with_the_cluster_level_permutation_p(recorded, mode):
    record = recorded.logprobs if mode == LOGPROBS else recorded.logits
    scores, labels, clusters = held_out(record, recorded.calibration_draws)
    statistic = by_name(compare(recorded))[f"holdout_auc_{mode}"]

    assert statistic.value == rank_auc(scores[labels == 1], scores[labels == 0])
    assert statistic.interval == clustered_auc_interval(scores, labels, clusters, n_resamples=1000, seed=0)
    expected = permutation_auc_test(scores, labels, clusters, n_permutations=2000, seed=0)
    assert statistic.test.p_value == expected.p_value
    assert statistic.test.parameters["n_clusters"] == len(record.holdout_prompts)
    assert statistic.unit == "prompt" and statistic.n == len(record.holdout_prompts)
    assert statistic.independent_units is False  # several draws share a prompt
    assert statistic.inputs["n_rows"] == scores.size
    assert statistic.inputs["n_memorized_rows"] + statistic.inputs["n_control_rows"] == scores.size
    assert statistic.effect_size == pytest.approx(statistic.value - 0.5)
    assert statistic.estimand and statistic.formula and statistic.missing is None


def test_the_stated_inputs_recompute_the_auc(recorded):
    statistic = by_name(compare(recorded))[f"holdout_auc_{LOGPROBS}"]
    inputs = statistic.inputs
    n_pos, n_neg = inputs["n_memorized_rows"], inputs["n_control_rows"]
    assert (inputs["rank_sum_memorized"] - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg) == pytest.approx(statistic.value)


def test_uneven_held_out_rows_without_the_calibration_draws_are_missing_never_guessed(recorded):
    statistics = by_name(compare(recorded, calibration_draws=None))
    for name in (f"holdout_auc_{LOGPROBS}", f"holdout_auc_{LOGITS}", "paired_auc_difference"):
        assert statistics[name].value is None and statistics[name].interval is None
        assert "calibration draws" in statistics[name].missing


def test_calibration_draws_that_do_not_give_the_held_out_rows_leave_the_rest_of_the_analysis_standing(recorded):
    """6.9: the calibrator statistics are missing with the reason; what does not need the mapping is computed."""
    comparison = compare(recorded,
                         calibration_draws=[draw for draw in recorded.calibration_draws if draw.draw_index == 0])
    statistics, full = by_name(comparison), by_name(compare(recorded))
    names = (f"holdout_auc_{LOGPROBS}", f"holdout_auc_{LOGITS}", "paired_auc_difference")
    for name in names:
        assert statistics[name].value is None and statistics[name].interval is None and statistics[name].test is None
        assert "calibration draws do not give the held-out rows" in statistics[name].missing
    for name, statistic in full.items():
        if name not in names:
            assert statistics[name] == statistic
    assert statistics["estimate_agreement_mean_difference"].value is not None
    assert all(statistics[name].value is not None for name in CORRECTION_NAMES)
    assert any(name.startswith("convergence") and statistic.value is not None
               for name, statistic in statistics.items())
    assert comparison.answer == UNDECIDED and comparison.detectable_effect is None
    assert all(name in comparison.reason for name in names) and "held-out rows" in comparison.reason


# ---------- the paired difference (4.3, 6.6, 6.11) ----------

def test_the_paired_difference_is_logits_minus_logprobs_with_interval_test_and_detectable_effect(recorded):
    logits, labels, clusters = held_out(recorded.logits, recorded.calibration_draws)
    logprobs, _, _ = held_out(recorded.logprobs, recorded.calibration_draws)
    value, interval, test = paired_auc_difference(logits, logprobs, labels, clusters, n_resamples=2000, seed=0)
    statistic = by_name(compare(recorded))["paired_auc_difference"]

    assert statistic.value == value and statistic.interval == interval
    assert statistic.test.p_value == test.p_value
    assert statistic.effect_size == value
    assert statistic.detectable_effect == test.parameters["detectable_effect"]
    assert statistic.inputs["auc_logits"] - statistic.inputs["auc_logprobs"] == pytest.approx(value)
    assert statistic.unit == "prompt" and statistic.n == len(recorded.logits.holdout_prompts)
    assert "logits" in statistic.estimand and "minus" in statistic.estimand


def test_calibrators_that_hold_out_other_prompts_are_not_paired_and_say_why(recorded):
    scores, labels, clusters = held_out(recorded.logits, recorded.calibration_draws)
    keep = clusters != clusters.max()
    mcs = replace(recorded.logits.calibrator.mcs, holdout_labels=tuple(int(v) for v in labels[keep]),
                  holdout_probabilities=tuple(float(v) for v in scores[keep]))
    fewer = replace(recorded.logits, holdout_prompts=recorded.logits.holdout_prompts[:-1],
                    calibrator=replace(recorded.logits.calibrator, mcs=mcs))
    comparison = compare(recorded, logits=fewer)
    statistics = by_name(comparison)

    assert statistics["paired_auc_difference"].value is None
    assert "same prompts and draws" in statistics["paired_auc_difference"].missing
    assert statistics[f"holdout_auc_{LOGITS}"].value is not None
    assert statistics[f"holdout_auc_{LOGPROBS}"].value is not None
    assert comparison.answer == UNDECIDED and comparison.detectable_effect is None


# ---------- agreement and the correction factor (4.3, 4.4) ----------

def test_the_agreement_is_prompt_by_prompt_over_the_draws_with_both_estimates_resampled_by_month(recorded):
    """4.3: one row per prompt (item and rendering), each mode's estimate the mean over the same draws."""
    full, top, months, n_draws = prompt_rows(recorded.results)
    expected = agreement(full, top, months, n_resamples=2000, seed=0, unit="month")
    statistics = by_name(compare(recorded))

    for name in AGREEMENT_STATISTICS:
        statistic = statistics[f"estimate_agreement_{name}"]
        assert statistic.value == expected[name].value and statistic.interval == expected[name].interval
        assert statistic.unit == "prompt" and statistic.n == len(full)
        assert statistic.independent_units is False  # the two renderings of a month share the month
        assert statistic.inputs["n_prompts"] == statistic.inputs["n_rows"] == len(full)
        assert statistic.inputs["n_months"] == statistic.inputs["n_clusters"] == len(set(months))
        assert statistic.inputs["n_draws"] == n_draws
        assert statistic.inputs["prompts_without_a_draw_with_both_estimates"] == 0
        assert "logits" in statistic.estimand and "logprobs" in statistic.estimand
        assert "prompt" in statistic.estimand and "mean" in statistic.estimand and "month" in statistic.estimand
    assert statistics["estimate_agreement_mean_difference"].test is not None
    assert statistics["estimate_agreement_mean_difference"].value == pytest.approx(
        float(np.mean(np.asarray(full) - np.asarray(top))))


def test_a_prompt_without_a_draw_that_carries_both_estimates_is_excluded_and_counted(recorded):
    results = (without_shared_draws(recorded.results[0]), *recorded.results[1:])
    full, top, months, n_draws = prompt_rows(results)
    assert len(full) == len(recorded.results) - 1
    expected = agreement(full, top, months, n_resamples=2000, seed=0, unit="month")
    statistics = by_name(compare(recorded, results=results))

    for name in (f"estimate_agreement_{name}" for name in AGREEMENT_STATISTICS):
        assert statistics[name].n == len(full) and statistics[name].inputs["n_draws"] == n_draws
        assert statistics[name].inputs["prompts_without_a_draw_with_both_estimates"] == 1
        assert statistics[name].inputs["n_stored_prompts"] == len(recorded.results)
    assert statistics["estimate_agreement_mean_difference"].value == expected["mean_difference"].value
    none = by_name(compare(recorded, results=[without_shared_draws(result) for result in recorded.results]))
    assert none["estimate_agreement_mean_difference"].value is None
    assert "both estimates" in none["estimate_agreement_mean_difference"].missing
    # The correction factor reads the stored ensemble decisions, not the per-draw estimates: it is unaffected.
    for name in CORRECTION_NAMES:
        assert none[name] == by_name(compare(recorded))[name] and none[name].value is not None


def test_the_correction_factor_difference_is_from_the_ensemble_decisions_resampled_by_month(recorded):
    """4.4: the factor is 1 - p_memorized of the ensemble decision, the value the consumer applies to the tilt."""
    results = with_ensemble_estimates(recorded.results)
    logits, logprobs, months = ensemble_rows(results)
    assert len(logits) == len(results)
    expected = agreement(logits, logprobs, months, n_resamples=2000, seed=0, unit="month")["mean_difference"]
    statistics = by_name(compare(recorded, results=results))
    statistic = statistics["correction_factor_difference"]

    assert statistic.value == pytest.approx(float(np.mean(np.asarray(logits) - np.asarray(logprobs))))
    assert statistic.value == expected.value and statistic.interval == expected.interval
    assert statistic.interval is not None and statistic.interval.lo <= statistic.value <= statistic.interval.hi
    assert statistic.test.p_value == expected.test.p_value
    assert statistic.detectable_effect == expected.detectable_effect and statistic.detectable_effect is not None
    assert statistic.unit == "prompt" and statistic.n == len(logits) and statistic.independent_units is False
    assert statistic.inputs["n_items"] == len(logits) and statistic.inputs["n_months"] == len(set(months))
    assert statistic.inputs["n_stored_items"] == len(results) and statistic.inputs["items_excluded"] == 0
    assert "n_draws" not in statistic.inputs and "n_prompts" not in statistic.inputs
    for phrase in ("difference between the modes in the correction factor the consumer applies to the tilt, "
                   "1 minus the ensemble recall estimate", "logits minus logprobs",
                   "a positive value means logits soften the tilt less", "month"):
        assert phrase in statistic.estimand
    assert "p_memorized" in statistic.formula
    # Not the agreement's mean difference with the opposite sign: the agreement keeps the per-draw estimates.
    agreement_mean = statistics["estimate_agreement_mean_difference"]
    assert agreement_mean == by_name(compare(recorded))["estimate_agreement_mean_difference"]
    assert statistic.value != pytest.approx(-agreement_mean.value)


def test_the_mean_correction_factor_of_each_mode_is_reported_over_the_same_items(recorded):
    """The level beside the difference: two plain statistics on the items of the difference."""
    results = with_ensemble_estimates(recorded.results)
    logits, logprobs, months = ensemble_rows(results)
    statistics = by_name(compare(recorded, results=results))
    difference = statistics["correction_factor_difference"]

    for mode, factors in ((LOGITS, logits), (LOGPROBS, logprobs)):
        statistic = statistics[f"correction_factor_mean_{mode}"]
        assert statistic.value == pytest.approx(float(np.mean(factors)))
        assert statistic.interval is None and statistic.test is None and statistic.detectable_effect is None
        assert statistic.unit == "prompt" and statistic.n == difference.n == len(factors)
        assert statistic.inputs["n_items"] == len(factors) and statistic.inputs["n_months"] == len(set(months))
        assert statistic.inputs["sum_of_factors"] / statistic.inputs["n_items"] == pytest.approx(statistic.value)
        assert mode in statistic.estimand and "1 minus the ensemble recall estimate" in statistic.estimand
        assert "same items" in statistic.estimand and statistic.missing is None
    assert (statistics[f"correction_factor_mean_{LOGITS}"].value
            - statistics[f"correction_factor_mean_{LOGPROBS}"].value) == pytest.approx(difference.value)
    names = [statistic.name for statistic in compare(recorded, results=results).statistics]
    start = names.index("correction_factor_difference")
    assert tuple(names[start:start + 3]) == CORRECTION_NAMES


def test_an_item_without_an_ensemble_decision_or_estimate_in_a_mode_is_excluded_and_counted(recorded):
    results = with_ensemble_estimates(recorded.results)
    results = (without_ensemble_decision(results[0], LOGITS),
               with_ensemble(results[1], LOGPROBS, p_memorized=None, status="estimate_unavailable"), *results[2:])
    logits, logprobs, months = ensemble_rows(results)
    assert len(logits) == len(results) - 2
    expected = agreement(logits, logprobs, months, n_resamples=2000, seed=0, unit="month")["mean_difference"]
    statistics = by_name(compare(recorded, results=results))

    for name in CORRECTION_NAMES:
        inputs = statistics[name].inputs
        assert statistics[name].n == inputs["n_items"] == len(logits)
        assert inputs["n_stored_items"] == len(results) and inputs["items_excluded"] == 2
        assert inputs["items_without_an_ensemble_decision_in_a_mode"] == 1
        assert inputs["items_without_an_ensemble_estimate_in_a_mode"] == 1
        assert inputs["items_excluded_whose_adjustment_was_skipped_in_a_mode"] == 0
        assert "items_included_whose_adjustment_was_skipped_in_a_mode" not in inputs
    assert statistics["correction_factor_difference"].value == expected.value
    assert statistics["correction_factor_difference"].interval == expected.interval
    # The agreement has another n: it does not depend on the ensemble decision.
    assert statistics["estimate_agreement_mean_difference"].n == len(results) != len(logits)


@pytest.mark.parametrize("mode", [LOGITS, LOGPROBS])
def test_an_item_whose_adjustment_was_skipped_in_a_mode_is_excluded_and_counted(recorded, mode):
    """4.4: the consumer applied no factor to that item, so it has no factor to compare."""
    results = with_ensemble_estimates(recorded.results)
    skipped = (with_ensemble(results[0], mode, status="adjustment_skipped_weak_calibrator"), *results[1:])
    assert skipped[0].variant(ENSEMBLE_VARIANTS[mode]).p_memorized is not None  # the estimate is there, unapplied
    logits, logprobs, months = ensemble_rows(skipped)
    assert (logits, logprobs, months) == ensemble_rows(results[1:]) and len(logits) == len(results) - 1
    expected = agreement(logits, logprobs, months, n_resamples=2000, seed=0, unit="month")["mean_difference"]
    statistics = by_name(compare(recorded, results=skipped))

    for name in CORRECTION_NAMES:
        inputs = statistics[name].inputs
        assert statistics[name].n == inputs["n_items"] == len(logits)
        assert inputs["n_stored_items"] == len(results) and inputs["items_excluded"] == 1
        assert inputs["items_excluded_whose_adjustment_was_skipped_in_a_mode"] == 1
        assert inputs["items_without_an_ensemble_decision_in_a_mode"] == 0
        assert inputs["items_without_an_ensemble_estimate_in_a_mode"] == 0
    difference = statistics["correction_factor_difference"]
    assert difference.value == expected.value and difference.interval == expected.interval
    assert difference.value == pytest.approx(float(np.mean(np.asarray(logits) - np.asarray(logprobs))))
    assert difference.value != by_name(compare(recorded, results=results))["correction_factor_difference"].value
    assert "the correction factor the consumer applies" in difference.estimand
    assert "adjustment" in difference.estimand and "excluded and counted" in difference.estimand
    for mode_name, factors in ((LOGITS, logits), (LOGPROBS, logprobs)):
        assert statistics[f"correction_factor_mean_{mode_name}"].value == pytest.approx(float(np.mean(factors)))

    all_skipped = [with_ensemble(result, mode, status="adjustment_skipped_weak_calibrator") for result in results]
    none = by_name(compare(recorded, results=all_skipped))["correction_factor_difference"]
    assert none.value is None and none.n == 0
    assert none.inputs["items_excluded_whose_adjustment_was_skipped_in_a_mode"] == len(results)
    assert "adjustment" in none.missing


@pytest.mark.parametrize("name", ["estimate_agreement_mean_difference", "correction_factor_difference"])
def test_the_effect_size_of_a_mean_difference_is_recomputed_from_the_inputs_of_its_row(recorded, name):
    """6.2: the row carries the standard deviation of the differences and states the formula of the effect size."""
    results = with_ensemble_estimates(recorded.results)
    statistic = by_name(compare(recorded, results=results))[name]
    if name == "correction_factor_difference":
        a, b, _ = ensemble_rows(results)
    else:
        a, b, _, _ = prompt_rows(results)
    differences = np.asarray(a) - np.asarray(b)
    inputs = statistic.inputs

    assert inputs["sd_of_differences"] == float(differences.std(ddof=1)) > 0
    assert statistic.effect_size is not None
    assert statistic.effect_size == pytest.approx(
        (inputs["sum_of_differences"] / inputs["n_rows"]) / inputs["sd_of_differences"], rel=1e-12)
    assert statistic.effect_size == pytest.approx(statistic.value / inputs["sd_of_differences"], rel=1e-12)
    for phrase in ("effect size", "(sum_of_differences / n_rows) / sd_of_differences", "n_rows - 1"):
        assert phrase in statistic.formula
    written = {row["name"]: row for row in compare(recorded, results=results).to_dict()["statistics"]}[name]
    assert written["inputs"]["sd_of_differences"] == inputs["sd_of_differences"]
    assert written["effect_size"] == statistic.effect_size and written["formula"] == statistic.formula


def test_without_an_ensemble_decision_in_both_modes_the_correction_statistics_are_missing(recorded):
    """6.9: no value, and the reason names the input."""
    results = [without_ensemble_decision(result, LOGPROBS) for result in recorded.results]
    statistics = by_name(compare(recorded, results=results))
    for name in CORRECTION_NAMES:
        statistic = statistics[name]
        assert statistic.value is None and statistic.interval is None and statistic.test is None
        assert statistic.n == 0 and statistic.inputs["items_excluded"] == len(results)
        assert "ensemble decision" in statistic.missing and "both modes" in statistic.missing
    assert statistics["estimate_agreement_mean_difference"].value is not None


def one_prompt(results):
    """The first stored prompt alone: one row, so no difference between rows has a standard deviation."""
    return tuple(results[:1])


def two_prompts_of_one_month(results):
    """Two stored prompts (renderings) of the same month: two rows in one cluster."""
    for month in dict.fromkeys(result.item_id for result in results):
        found = tuple(result for result in results if result.item_id == month)
        if len(found) >= 2:
            return found[:2]
    pytest.fail("the fixture holds no month with two renderings")


TOO_FEW_MONTHS = pytest.mark.parametrize("chosen, n_rows", [(one_prompt, 1), (two_prompts_of_one_month, 2)])


def non_finite_numbers(comparison: ModeComparison) -> list[str]:
    """Every number of the statistics of ``comparison`` that is nan or infinite, by where it stands."""
    def walk(where: str, value) -> list[str]:
        if isinstance(value, dict):
            return [found for key, inner in value.items() for found in walk(f"{where}.{key}", inner)]
        if isinstance(value, list | tuple):
            return [found for index, inner in enumerate(value) for found in walk(f"{where}[{index}]", inner)]
        if isinstance(value, bool) or not isinstance(value, int | float | np.number):
            return []
        return [] if np.isfinite(value) else [f"{where} = {value}"]

    return [found for entry in comparison.to_dict()["statistics"] for found in walk(entry["name"], entry)]


@TOO_FEW_MONTHS
def test_a_correction_difference_on_fewer_than_two_months_is_missing_and_still_written(
        recorded, tmp_path, chosen, n_rows):
    """6.9: a missing statistic names its reason and carries no derived input it does not use; 6.2: it is written."""
    results = chosen(with_ensemble_estimates(recorded.results))
    assert len(results) == n_rows and len({result.item_id for result in results}) == 1
    comparison = compare(recorded, results=results)
    difference = by_name(comparison)["correction_factor_difference"]

    assert difference.value is None and difference.interval is None and difference.effect_size is None
    assert difference.missing and "two" in difference.missing
    assert "sd_of_differences" not in difference.inputs
    assert difference.n == difference.inputs["n_items"] == n_rows and difference.inputs["n_months"] == 1
    assert non_finite_numbers(comparison) == []
    written = json.loads(comparison.document)  # json.dumps(allow_nan=False): a nan input would raise here
    row = {entry["name"]: entry for entry in written["statistics"]}["correction_factor_difference"]
    assert row["value"] is None and row["missing"] == difference.missing and "sd_of_differences" not in row["inputs"]
    json_path, markdown_path = write_comparison(comparison, tmp_path)
    assert json.loads(json_path.read_text(encoding="utf-8")) == comparison.to_dict()
    assert "| correction_factor_difference |" in markdown_path.read_text(encoding="utf-8")
    # The level of each mode over the same items is still a plain mean of what is there.
    logits, logprobs, _ = ensemble_rows(results)
    for mode, factors in ((LOGITS, logits), (LOGPROBS, logprobs)):
        assert by_name(comparison)[f"correction_factor_mean_{mode}"].value == pytest.approx(float(np.mean(factors)))


@TOO_FEW_MONTHS
def test_the_agreement_on_fewer_than_two_months_is_missing_and_still_written(recorded, tmp_path, chosen, n_rows):
    """6.9 for the agreement rows: missing with the reason, no sd of the differences, and the files are written."""
    results = chosen(recorded.results)
    assert len(results) == n_rows and len({result.item_id for result in results}) == 1
    comparison = compare(recorded, results=results)
    statistics = by_name(comparison)

    for name in (f"estimate_agreement_{name}" for name in AGREEMENT_STATISTICS):
        statistic = statistics[name]
        assert statistic.value is None and statistic.interval is None and statistic.effect_size is None
        assert statistic.missing and "two" in statistic.missing
        assert "sd_of_differences" not in statistic.inputs
        assert statistic.n == statistic.inputs["n_prompts"] == n_rows and statistic.inputs["n_months"] == 1
    assert non_finite_numbers(comparison) == []
    assert json.loads(comparison.document) == comparison.to_dict()
    json_path, _ = write_comparison(comparison, tmp_path)
    assert json.loads(json_path.read_text(encoding="utf-8")) == comparison.to_dict()


@pytest.mark.parametrize("chosen", [one_prompt, two_prompts_of_one_month, lambda results: tuple(results[:3]),
                                    lambda results: tuple(results)],
                         ids=["one prompt", "two prompts of one month", "three prompts", "all prompts"])
def test_no_number_of_any_statistic_is_nan_or_infinite(recorded, chosen):
    """6.2, 6.9: every input, value, interval bound, test number and effect size is finite, or the row is missing."""
    for results in (chosen(recorded.results), chosen(with_ensemble_estimates(recorded.results))):
        comparison = compare(recorded, results=results)
        assert comparison.statistics and non_finite_numbers(comparison) == []
        for statistic in comparison.statistics:
            assert (statistic.value is None) == (statistic.missing is not None), statistic.name
            if statistic.value is None:
                assert statistic.interval is None and statistic.effect_size is None, statistic.name
                assert "sd_of_differences" not in statistic.inputs, statistic.name
        assert json.loads(comparison.document) == comparison.to_dict()


def test_the_definitions_of_the_compared_estimates_are_stated_in_the_json_and_the_markdown(recorded):
    comparison = compare(recorded)
    definitions = comparison.to_dict()["definitions"]
    definition = definitions["prompt_estimate"]
    for phrase in ("item", "rendering", "mean", "both modes", "month"):
        assert phrase in definition
    assert definition in comparison.to_markdown()
    assert "cutoff check" in definition
    compared = definitions["estimates_compared"]
    for phrase in ("agreement", "per-prompt mean", "more prompts", "minimum of parsed draws", "correction factor",
                   "ensemble estimate", "the value the consumer applies", "different n"):
        assert phrase in compared
    assert comparison.to_markdown().count(compared) == 1


def test_results_scored_with_another_calibrator_are_refused(recorded):
    other = replace(recorded.logits, calibrator=replace(recorded.logits.calibrator, fingerprint="f" * 64))
    with pytest.raises(ModeComparisonError, match="calibrator"):
        compare(recorded, logits=other)


# ---------- convergence, read from the convergence analysis (4.5) ----------

def test_convergence_by_alternatives_kept_is_read_from_the_convergence_table(recorded):
    statistics = by_name(compare(recorded))
    rows = [row for row in recorded.convergence["features"] if row["feature"] == "estimate"]
    assert rows
    for row in rows:
        statistic = statistics[f"convergence_{row['evidence']}_k{row['k']}"]
        if row["mean_abs_diff"] is None:
            assert statistic.value is None and statistic.missing
            continue
        assert statistic.value == row["mean_abs_diff"]
        assert statistic.unit == "draw" and statistic.n == row["n"]
        assert statistic.inputs["k"] == row["k"] and statistic.inputs["n_draws"] == row["n"]
        assert statistic.by_construction == (row["evidence"] == HOST_TOP_K and row["k"] == recorded.convergence["kept"])
        assert statistic.test is None


def test_without_the_convergence_table_the_convergence_is_missing(recorded):
    comparison = compare(recorded, convergence=None)
    (statistic,) = [s for s in comparison.statistics if s.name.startswith("convergence")]
    assert statistic.value is None and "convergence analysis" in statistic.missing


def test_a_convergence_table_of_other_weights_is_not_read(recorded):
    comparison = compare(recorded, convergence=recorded.convergence | {"model_digest": OTHER_DIGEST})
    (statistic,) = [s for s in comparison.statistics if s.name.startswith("convergence")]
    assert statistic.value is None and "weights" in statistic.missing


# ---------- the three answers (4.6) ----------

def improving(recorded, **overrides) -> ModeComparison:
    return compare(recorded, logits=with_scores(recorded.logits, separating(recorded.logits)),
                   logprobs=with_scores(recorded.logprobs, inverted(recorded.logprobs)), **overrides)


def test_improve_needs_an_interval_above_zero_and_a_passed_cutoff_check(recorded):
    comparison = improving(recorded, cutoff_check=True)
    paired = by_name(comparison)["paired_auc_difference"]
    assert paired.interval.lo > 0
    assert comparison.answer == IMPROVE and comparison.cutoff_check is True
    assert comparison.detectable_effect == paired.detectable_effect


@pytest.mark.parametrize(("cutoff", "phrase"), [(None, "not supplied"), (False, "did not pass")])
def test_improve_is_not_concluded_without_a_passed_cutoff_check(recorded, cutoff, phrase):
    comparison = improving(recorded, cutoff_check=cutoff)
    assert comparison.answer == UNDECIDED
    assert "cutoff check" in comparison.reason and phrase in comparison.reason
    assert comparison.detectable_effect is not None


@pytest.mark.parametrize(("cutoff", "phrase"), [(None, "not supplied"), (False, "did not pass")])
def test_the_reason_of_an_undecided_answer_states_the_interval_and_each_fact_once(recorded, cutoff, phrase):
    """Neither improve (no passed cutoff check) nor match (outside the margin): one interval, each fact once."""
    comparison = improving(recorded, cutoff_check=cutoff)
    interval = by_name(comparison)["paired_auc_difference"].interval
    reason = comparison.reason

    assert comparison.answer == UNDECIDED and interval.hi > AUC_MARGIN
    assert reason.count(f"{interval.lo:.4g} to {interval.hi:.4g}") == 1
    for fact in ("paired AUC difference", "excludes 0 in favour of logits", "cutoff check", phrase,
                 f"not within {AUC_MARGIN:g} either side of 0", "improve", "match"):
        assert reason.count(fact) == 1, fact
    assert reason.count("interval") == 1


def test_the_reason_of_an_undecided_answer_without_a_difference_states_the_interval_once(recorded):
    comparison = compare(recorded, logprobs=with_scores(recorded.logprobs, noise(recorded.logprobs, 1)),
                         logits=with_scores(recorded.logits, noise(recorded.logits, 2)), cutoff_check=True)
    interval = by_name(comparison)["paired_auc_difference"].interval
    assert comparison.answer == UNDECIDED
    assert comparison.reason.count(f"{interval.lo:.4g} to {interval.hi:.4g}") == 1
    assert comparison.reason.count("paired AUC difference") == 1
    assert comparison.reason.count("does not exclude 0 in favour of logits") == 1


def test_match_needs_the_difference_within_its_margin_and_the_agreement_limits_within_theirs(recorded):
    scores = noise(recorded.logprobs, 3)
    logprobs = with_scores(recorded.logprobs, scores)
    logits = with_scores(recorded.logits, 0.999 * scores + 0.0005)
    comparison = compare(recorded, logprobs=logprobs, logits=logits, results=near_results(recorded.results))
    statistics = by_name(comparison)

    interval = statistics["paired_auc_difference"].interval
    assert -AUC_MARGIN <= interval.lo and interval.hi <= AUC_MARGIN
    lower = statistics["estimate_agreement_lower_limit_of_agreement"]
    upper = statistics["estimate_agreement_upper_limit_of_agreement"]
    assert lower.interval.lo >= -AGREEMENT_MARGIN and upper.interval.hi <= AGREEMENT_MARGIN
    assert comparison.answer == MATCH
    assert "interval" in comparison.reason and f"{lower.interval.lo:.4g}" in comparison.reason
    assert f"{upper.interval.hi:.4g}" in comparison.reason
    assert (AUC_MARGIN, AGREEMENT_MARGIN) == (0.05, 0.10)
    assert "outer interval bounds" in comparison.to_markdown()

    wide = compare(recorded, logprobs=logprobs, logits=logits, results=near_results(recorded.results, 0.2))
    assert wide.answer == UNDECIDED and "limits of agreement" in wide.reason


def test_match_is_decided_on_the_interval_bounds_of_the_limits_of_agreement_not_their_point_values(recorded):
    """Point limits inside the margin do not make a match when the interval of a limit reaches outside it."""
    scores = noise(recorded.logprobs, 3)
    logprobs = with_scores(recorded.logprobs, scores)
    logits = with_scores(recorded.logits, 0.999 * scores + 0.0005)
    results = month_shifted(recorded.results, (0.0, 0.0, 0.0675))
    comparison = compare(recorded, logprobs=logprobs, logits=logits, results=results)
    statistics = by_name(comparison)

    interval = statistics["paired_auc_difference"].interval
    assert -AUC_MARGIN <= interval.lo and interval.hi <= AUC_MARGIN
    lower = statistics["estimate_agreement_lower_limit_of_agreement"]
    upper = statistics["estimate_agreement_upper_limit_of_agreement"]
    assert -AGREEMENT_MARGIN <= lower.value and upper.value <= AGREEMENT_MARGIN  # the point limits are inside
    assert upper.interval.hi > AGREEMENT_MARGIN                                  # the interval of one is not
    assert comparison.answer == UNDECIDED
    assert "limits of agreement" in comparison.reason and f"{upper.interval.hi:.4g}" in comparison.reason


def test_improve_is_not_concluded_on_a_corpus_other_than_the_financial_time_series(recorded):
    """The paired interval favours logits on the EDGAR calibrators: that is not the macro boundary (4.6)."""
    logprobs = with_scores(recorded.edgar_logprobs, inverted(recorded.edgar_logprobs))
    logits = with_scores(recorded.edgar_logits, separating(recorded.edgar_logits))
    comparison = mode_comparison(PLAN, arm="sampling", kind="edgar", logprobs=logprobs, logits=logits,
                                 results=None, calibration_draws=recorded.calibration_draws, cutoff_check=True)
    paired = by_name(comparison)["paired_auc_difference"]

    assert comparison.kind == "edgar" and paired.interval.lo > 0
    assert comparison.answer == UNDECIDED and comparison.answer != IMPROVE
    assert "excludes 0 in favour of logits" in comparison.reason and "macro" in comparison.reason
    assert comparison.detectable_effect == paired.detectable_effect


def test_undecided_states_the_detectable_effect(recorded):
    comparison = compare(recorded, logprobs=with_scores(recorded.logprobs, noise(recorded.logprobs, 1)),
                         logits=with_scores(recorded.logits, noise(recorded.logits, 2)), cutoff_check=True)
    paired = by_name(comparison)["paired_auc_difference"]
    assert paired.interval.lo <= 0 and paired.interval.hi - paired.interval.lo > 2 * AUC_MARGIN
    assert comparison.answer == UNDECIDED
    assert comparison.detectable_effect == paired.detectable_effect and comparison.detectable_effect > 0
    assert "interval" in comparison.reason


# ---------- refusal across weights (4.7) ----------

def test_modes_with_different_weights_are_refused_and_nothing_is_computed(recorded):
    comparison = compare(recorded, logits=with_weights(recorded.logits, OTHER_DIGEST))
    assert comparison.answer is None and comparison.statistics == ()
    assert DIGEST in comparison.refusal and OTHER_DIGEST in comparison.refusal
    data = comparison.to_dict()
    assert data["statistics"] == [] and data["answer"] is None and data["refusal"] == comparison.refusal
    assert "Refused" in comparison.to_markdown() and "| Statistic |" not in comparison.to_markdown()


def test_a_mode_without_a_weights_identity_is_refused(recorded):
    comparison = compare(recorded, logprobs=with_weights(recorded.logprobs, None))
    assert comparison.answer is None and comparison.statistics == ()
    assert "weights" in comparison.refusal


# ---------- missing inputs (6.9) ----------

def test_a_missing_calibrator_is_reported_without_a_value(recorded):
    comparison = compare(recorded, logits=None)
    statistics = by_name(comparison)
    assert comparison.refusal is None and comparison.answer == UNDECIDED
    for name in (f"holdout_auc_{LOGITS}", "paired_auc_difference"):
        assert statistics[name].value is None and statistics[name].interval is None and statistics[name].test is None
        assert "logits" in statistics[name].missing and "calibrator" in statistics[name].missing
    assert statistics[f"holdout_auc_{LOGPROBS}"].value is not None
    assert "paired_auc_difference" in comparison.reason


def test_missing_item_results_are_reported_without_a_value(recorded):
    statistics = by_name(compare(recorded, results=None))
    names = [f"estimate_agreement_{name}" for name in AGREEMENT_STATISTICS] + list(CORRECTION_NAMES)
    for name in names:
        assert statistics[name].value is None and "item results" in statistics[name].missing
    assert statistics["paired_auc_difference"].value is not None


# ---------- the written analysis ----------

def from_records(recorded, **overrides) -> ModeComparison:
    values = {"arm": "sampling", "calibrators_dir": recorded.calibrators_dir, "results_dir": recorded.results_dir,
              "records_dir": recorded.records_dir, "analysis_dir": recorded.analysis_dir}
    values |= overrides
    return mode_comparison_from_records(PLAN, **values)


def test_the_comparison_is_produced_from_the_recorded_files_alone(recorded):
    assert from_records(recorded).to_dict() == compare(recorded).to_dict()
    assert from_records(recorded, cutoff_check=True).cutoff_check is True


@pytest.mark.parametrize(("ending", "ignored"), [("no_newline", 0), ("cut_short", 1)])
def test_reading_the_records_never_modifies_the_draw_log(recorded, tmp_path, ending, ignored):
    """A last line without its newline is used, one cut short is ignored and stated; the log keeps its bytes."""
    source = draw_log_path("sampling", "calibration", recorded.records_dir)
    path = draw_log_path("sampling", "calibration", tmp_path)
    path.parent.mkdir(parents=True)
    data = source.read_bytes()
    path.write_bytes(data.rstrip(b"\n") if ending == "no_newline" else data + CUT)
    before = aged(path)
    assert not before[0].endswith(b"\n")

    comparison = from_records(recorded, records_dir=tmp_path)

    assert unchanged(path, before)
    assert [found.name for found in tmp_path.rglob("*") if found.is_file()] == ["draws.jsonl"]
    assert comparison.to_dict()["calibration_draw_log"] == {"incomplete_last_lines_ignored": ignored}
    assert ("incomplete last line" in comparison.to_markdown()) == bool(ignored)
    assert comparison.statistics == compare(recorded).statistics and comparison.answer == compare(recorded).answer


def test_the_analysis_holds_no_write_other_than_its_own_outputs():
    source = inspect.getsource(mode_comparison_module)
    assert "DrawLog" not in source and ".latest(" not in source and "_repaired_lines" not in source
    assert not hasattr(mode_comparison_module, "DrawLog")
    assert writing_calls(mode_comparison_module) == {
        ("_write_text", "mkdir"), ("_write_text", "open"), ("_write_text", "os.replace"),
        ("write_comparison", "_write_text")}


def test_an_arm_without_the_logits_mode_reports_it_missing(recorded, tmp_path):
    comparison = mode_comparison_from_records(PLAN, arm="serving", calibrators_dir=tmp_path, results_dir=tmp_path,
                                              records_dir=tmp_path, analysis_dir=tmp_path)
    assert comparison.answer == UNDECIDED
    assert all(statistic.value is None and statistic.missing for statistic in comparison.statistics)


def test_the_written_files_are_reproduced_byte_for_byte_and_never_replaced_by_different_ones(recorded, tmp_path):
    comparison = from_records(recorded)
    json_path, markdown_path = write_comparison(comparison, tmp_path)
    assert json_path == comparison_path("sampling", tmp_path) == tmp_path / "mode-comparison-sampling.json"
    assert markdown_path == tmp_path / "mode-comparison-sampling.md"
    first = json_path.read_bytes(), markdown_path.read_bytes()
    assert b"\r" not in first[0] + first[1]
    assert json.loads(first[0]) == comparison.to_dict()
    assert first[0].decode("utf-8") == json.dumps(comparison.to_dict(), indent=2, sort_keys=True) + "\n"

    assert write_comparison(from_records(recorded), tmp_path) == (json_path, markdown_path)
    assert (json_path.read_bytes(), markdown_path.read_bytes()) == first
    with pytest.raises(ModeComparisonError, match="refusing to replace"):
        write_comparison(from_records(recorded, cutoff_check=True), tmp_path)


def test_every_statistic_is_written_with_its_record(recorded):
    comparison = compare(recorded)
    data = comparison.to_dict()
    assert data["answer"] == comparison.answer and data["weights"] == DIGEST
    assert data["margins"] == {"auc_difference": AUC_MARGIN, "agreement_limits": AGREEMENT_MARGIN}
    assert [entry["name"] for entry in data["statistics"]] == [s.name for s in comparison.statistics]
    keys = {"name", "estimand", "formula", "inputs", "unit", "n", "independent_units", "value", "interval", "test",
            "effect_size", "detectable_effect", "by_construction", "missing"}
    assert all(set(entry) == keys for entry in data["statistics"])
    paired = next(entry for entry in data["statistics"] if entry["name"] == "paired_auc_difference")
    assert set(paired["interval"]) == {"lo", "hi", "confidence", "method"}
    assert {"statistic", "p_value", "method", "parameters"} <= set(paired["test"])


def test_the_markdown_table_gives_value_interval_n_and_the_calculation(recorded):
    comparison = compare(recorded, convergence=None)
    text = comparison.to_markdown()
    header = next(line for line in text.splitlines() if line.startswith("| Statistic |"))
    for column in ("Value", "Interval", "n (unit)", "Calculation", "Estimand", "Test", "Detectable effect"):
        assert column in header
    width = header.count("|")
    rows = [line for line in text.splitlines() if line.startswith("| ") and line != header]
    assert len(rows) == len(comparison.statistics)
    assert all(row.replace("\\|", "").count("|") == width for row in rows)  # a pipe inside a cell is escaped
    paired = by_name(comparison)["paired_auc_difference"]
    row = next(line for line in rows if line.startswith("| paired_auc_difference |"))
    assert f"{paired.value:.4g}" in row and f"{paired.interval.lo:.4g} to {paired.interval.hi:.4g}" in row
    assert f"{paired.n} (prompt)" in row and "auc_logits=" in row and paired.formula in row
    missing = next(line for line in rows if line.startswith("| convergence"))
    assert "missing: " in missing
    assert f"Answer: {comparison.answer}" in text and DIGEST in text


def test_the_outputs_carry_derived_statistics_only(recorded):
    comparison = compare(recorded)
    text = comparison.to_markdown() + json.dumps(comparison.to_dict())
    for draw in recorded.draws:
        if draw.response is not None:
            assert draw.response.content not in text
    for result in recorded.results:
        assert result.item_id not in text and result.prompt_sha256 not in text
    for prompt in recorded.logits.holdout_prompts:
        assert prompt not in text
