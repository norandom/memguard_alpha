"""Pilot and power gate (task 7.8; Requirements 8.9 and 11.3).

Pure logic on fakes: the generation-host double of the 7.7 tests answers every
pilot prompt with a reply whose defensive spread makes the frozen noise variant
correct or not as the test prescribes, the full-distribution double answers
batches, calibrators are fitted by the 7.6 calibration on those fakes, serving-arm
results are derived from synthetic draws without a request, and every file is
under ``tmp_path``. No network, credentials or model host is used, and nothing is
written under ``runs/``.
"""

from __future__ import annotations

import json
import math
import shutil
import socket
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from recall_guard.core.completion import ModelResponse
from recall_guard.core.distribution import RecordedDraw
from recall_guard.core.significance import simulated_power_two_sample
from scripts.finance_sim import pilot as pilot_module
from scripts.finance_sim.calibrate import DrawSettings, FitSettings, run_calibration
from scripts.finance_sim.execute import (
    INTERRUPTED,
    RESULTS_DIR,
    ExecutionDraw,
    ExecutionItem,
    derive_result,
    execute_items,
    load_calibrators,
    macro_items,
    planned_keys,
)
from scripts.finance_sim.limits import AdmissionClosedError, BudgetLedger, LedgerCallBudget, StudyLimits
from scripts.finance_sim.pilot import (
    CONTINUE,
    GATES_DIR,
    NO_VARIATION,
    PILOT_EXCLUDED,
    STOP_FOR_DECISION,
    STOP_OPTIONS,
    TOO_FEW_MONTHS,
    TOO_FEW_PERIOD_MONTHS,
    GateStoppedError,
    MonthNoise,
    PilotError,
    PilotNoise,
    PowerGate,
    achieved_power,
    confirmatory_results,
    estimate_noise,
    gate_path,
    load_gate,
    power_gate,
    require_gate_passed,
    run_pilot,
    write_gate,
)
from scripts.finance_sim.plan import (
    EDGAR_FILES,
    AnalysisSettings,
    MonthClass,
    PilotSettings,
    PlanChangedError,
    StudyPlan,
    freeze_plan,
    load_serving_manifest,
    split_months,
)
from tests.scripts.finance_sim.test_execute import (
    ALLOWANCE,
    COMPLETION_TOKENS,
    CREATED_AT,
    EDGAR_ROWS,
    GEMMA,
    MACRO_ITEMS,
    MISSING_MONTH,
    MONTHS,
    PILOT_DRAWS,
    PLAN,
    SEED,
    SEED_BASE,
    TEMPERATURE,
    Clock,
    FakeFull,
    FakeHost,
    base_logprob,
    digest,
    edgar_releases,
    item_hash,
    macro_row,
    serving_identity,
    small_arms,
    token_evidence,
)

#: Loadings with a positive defensive spread, and their mirror with a negative one.
DEFENSIVE = {"inflation": 0.5, "growth": -0.5, "credit_stress": 0.5, "policy": 0.0, "risk_appetite": -0.5}
RISK = {"inflation": -0.5, "growth": 0.5, "credit_stress": -0.5, "policy": 0.0, "risk_appetite": 0.5}
#: (identifying correct, anonymised correct) per pilot month, cycled: differences 1, 0, 0, -1, 0, 1, 0, 0, 1, 0, 0, -1.
PATTERN: tuple[tuple[bool, bool], ...] = ((True, False), (True, True), (False, False), (False, True), (True, True),
                                          (True, False), (False, False), (True, True), (True, False), (False, False),
                                          (True, True), (False, True))
FLAT: tuple[tuple[bool, bool], ...] = ((True, True), (False, False))


def plan_with(analysis: AnalysisSettings) -> StudyPlan:
    """The 7.7 tests' frozen plan with other analysis settings (so another plan_sha256)."""
    months = [MonthClass(month, {GEMMA: month <= "2024-08"}) for month in MONTHS]
    items = {item.item_id: item_hash(item) for item in MACRO_ITEMS} | {row.row_id: digest(row.row_id)
                                                                      for row in EDGAR_ROWS}
    arms = small_arms()
    pilot = replace(PilotSettings.for_split(split_months(months, seed=SEED), arms), draws=PILOT_DRAWS,
                    min_parsed=PILOT_DRAWS)
    return freeze_plan(months, EDGAR_ROWS, seed=SEED, created_at=CREATED_AT, items=items,
                       edgar_files={name: digest(name) for name in EDGAR_FILES}, arms=arms,
                       serving_manifest=load_serving_manifest(), pilot=pilot, analysis=analysis)


#: Same items, split and pilot as PLAN, with a pre-registered effect the pilot noise can detect.
LARGE = plan_with(AnalysisSettings(effect_size=1.0))


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("the pilot attempted a network connection")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)


class ControlledHost(FakeHost):
    """The 7.7 generation double, answering each prompt with the reply ``answers`` gives it (prose for ``prose``)."""

    def __init__(self, budget, answers: Mapping[str, str], **kwargs) -> None:
        super().__init__(budget, **kwargs)
        self.answers = dict(answers)

    def generate(self, prompt: str, **parameters) -> ModelResponse:
        index = len(self.calls)
        self.calls.append((prompt, dict(parameters)))

        def operation() -> ModelResponse:
            seed = parameters.get("seed")
            content = "The regime looks benign to me." if index in self.prose else self.answers[prompt]
            return ModelResponse(content, token_evidence(base_logprob(prompt, seed)),
                                 usage={"completion_tokens": COMPLETION_TOKENS},
                                 requested_settings=dict(self.config.parameters) | parameters)

        return self._control.run(operation, stage="study")


def answer(plan: StudyPlan, item: ExecutionItem, correct: bool) -> str:
    """A reply whose spread sign is correct, or not, for the item's realized returns."""
    defensive_wins = plan.scored_probability.outcome_of(item.realized) == 1
    return json.dumps(DEFENSIVE if correct == defensive_wins else RISK)


def pilot_answers(plan: StudyPlan, pattern: Sequence[tuple[bool, bool]]) -> dict[str, str]:
    items = macro_items(plan, MACRO_ITEMS, stage="pilot")
    wanted = {}
    for number, month in enumerate(plan.months.pilot):
        wanted[f"macro-{month}"] = dict(zip(("identifying", "anonymised"), pattern[number % len(pattern)],
                                            strict=True))
    return {item.prompt: answer(plan, item, wanted[item.item_id][item.rendering]) for item in items}


def stage_items(plan: StudyPlan, arm: str, stage: str = "full_run") -> tuple[ExecutionItem, ...]:
    """Both renderings of every month the plan sizes for ``arm`` and ``stage``, from the frozen macro items."""
    by_id = {item.item_id: item for item in MACRO_ITEMS}
    items = []
    for item_id, rendering in planned_keys(plan, arm, stage):
        macro = by_id[item_id]
        realized = macro.outcome.realized
        items.append(ExecutionItem(item_id, "macro", rendering, macro.prompts()[rendering], realized=realized,
                                   realized_missing_reason=None if realized is not None else macro.outcome.reason))
    return tuple(items)


def expected_differences(plan: StudyPlan, pattern: Sequence[tuple[bool, bool]]) -> list[int]:
    return [int(pattern[n % len(pattern)][0]) - int(pattern[n % len(pattern)][1]) for n in range(len(plan.months.pilot))]


def stage_limits(plan: StudyPlan, stage: str, **overrides) -> StudyLimits:
    budget = plan.limits.stage("sampling", stage)
    values = {"plan_sha256": plan.plan_sha256, "arm": "sampling", "stage": stage,
              "max_requests": budget.max_requests, "max_output_tokens": budget.max_output_tokens,
              "max_seconds": budget.max_seconds} | overrides
    return StudyLimits(**values)


def open_ledger(root: Path, plan: StudyPlan, stage: str = "pilot", **overrides) -> BudgetLedger:
    return BudgetLedger.open(root / "records" / stage / "budget.json", plan, stage_limits(plan, stage, **overrides),
                             ALLOWANCE, clock=Clock())


@pytest.fixture(scope="module")
def calibrated(tmp_path_factory):
    """Per plan, the sampling arm's macro calibrators, fitted once by the 7.6 calibration on the fakes."""
    cache: dict[str, dict] = {}

    def calibrators(plan: StudyPlan) -> dict:
        if plan.plan_sha256 not in cache:
            root = tmp_path_factory.mktemp("calibration")
            ledger = BudgetLedger.open(root / "budget.json", plan,
                                       StudyLimits(plan.plan_sha256, "sampling", "calibration", 1000, 1_000_000,
                                                   1_000_000), ALLOWANCE, clock=Clock())
            run_calibration(plan, arm="sampling", macro_rows=[macro_row(month) for month in plan.months.calibration],
                            edgar_releases=edgar_releases(), client=FakeHost(LedgerCallBudget(ledger)), ledger=ledger,
                            settings=DrawSettings(draws=2, temperature=TEMPERATURE, seed_base=SEED_BASE),
                            sources=("top_k", "full_distribution"), full_distribution=FakeFull(),
                            fit=FitSettings(min_auc=0.6, min_valid=10), calibrators_dir=root / "calibrators",
                            sleep=lambda seconds: None)
            cache[plan.plan_sha256] = load_calibrators(plan, arm="sampling", kind="macro",
                                                       calibrators_dir=root / "calibrators")
        return cache[plan.plan_sha256]

    return calibrators


def pilot_run(root: Path, plan: StudyPlan, calibrators, *, pattern=PATTERN, prose=(), recorded=(),
              ledger: BudgetLedger | None = None) -> SimpleNamespace:
    """The whole pilot of ``plan`` on the fakes, through run_pilot."""
    ledger = ledger or open_ledger(root, plan)
    host = ControlledHost(LedgerCallBudget(ledger), pilot_answers(plan, pattern), prose=prose)
    full = FakeFull()
    outcome = run_pilot(plan, MACRO_ITEMS, client=host, ledger=ledger, calibrators=calibrators,
                        full_distribution=full, recorded=recorded, results_dir=root / "results",
                        gates_dir=root / "gates", sleep=lambda seconds: None)
    return SimpleNamespace(pilot=outcome, host=host, full=full, ledger=ledger, root=root,
                           results_dir=root / "results", gates_dir=root / "gates")


@pytest.fixture(scope="module")
def stopped(tmp_path_factory, calibrated) -> SimpleNamespace:
    """The pilot of the frozen plan (effect 0.10): the gate stops."""
    return pilot_run(tmp_path_factory.mktemp("pilot-frozen"), PLAN, calibrated(PLAN))


@pytest.fixture(scope="module")
def passed(tmp_path_factory, calibrated) -> SimpleNamespace:
    """The same pilot under a plan with a pre-registered effect of 1.0: the gate continues."""
    return pilot_run(tmp_path_factory.mktemp("pilot-large"), LARGE, calibrated(LARGE))


@pytest.fixture(scope="module")
def full_run(tmp_path_factory, calibrated) -> SimpleNamespace:
    """A few planned full-run months of the sampling arm (both periods), on the 7.7 fakes."""
    root = tmp_path_factory.mktemp("full-run")
    planned = list(dict.fromkeys(item_id for item_id, _ in planned_keys(PLAN, "sampling", "full_run")))
    months = {*planned[:3], *planned[-4:-1]}
    items = tuple(item for item in stage_items(PLAN, "sampling") if item.item_id in months)
    ledger = open_ledger(root, PLAN, "full_run")
    run = execute_items(PLAN, items, arm="sampling", stage="full_run", client=FakeHost(LedgerCallBudget(ledger)),
                        ledger=ledger, calibrators=calibrated(PLAN), full_distribution=FakeFull(),
                        results_dir=root / "results", sleep=lambda seconds: None)
    return SimpleNamespace(run=run, items=items)


def expected_power(plan: StudyPlan, noise_sd: float) -> float:
    return simulated_power_two_sample(plan.analysis.effect_size, noise_sd, plan.pilot.pre_cutoff_evaluation_months,
                                      plan.pilot.post_cutoff_evaluation_months, alpha=plan.analysis.alpha,
                                      n_sim=plan.analysis.n_resamples, seed=plan.seed)


# ---------- the pilot runs through the guarded execution ----------

def test_the_pilot_runs_every_frozen_pilot_month_once_through_the_guarded_execution_on_the_gated_arm(stopped):
    keys = planned_keys(PLAN, PLAN.pilot.arm, "pilot")
    assert PLAN.pilot.arm == "sampling" and len(keys) == len(PLAN.months.pilot) * 2
    calls = stopped.host.calls
    assert len(calls) == len(keys) * PLAN.pilot.draws == PLAN.limits.stage("sampling", "pilot").max_requests
    by_prompt: dict[str, list[dict]] = {}
    for prompt, parameters in calls:
        by_prompt.setdefault(prompt, []).append(parameters)
    assert len(by_prompt) == len(keys)
    for sent in by_prompt.values():
        assert [parameters["seed"] for parameters in sent] == [SEED_BASE + index for index in range(PLAN.pilot.draws)]
        assert {parameters["temperature"] for parameters in sent} == {TEMPERATURE}
    run = stopped.pilot.run
    assert set(run.results) == set(keys) and run.status == stopped.ledger.status == "complete"
    for result in run.results.values():
        assert (result.arm, result.stage, result.ensemble_draws, result.min_parsed) == (
            "sampling", "pilot", PLAN.pilot.draws, PLAN.pilot.min_parsed)
    assert len(stopped.full.batch_calls) == len(keys)  # the evidence pass of the 7.7 execution, one batch each


def test_the_noise_is_the_identifying_minus_anonymised_correctness_of_the_frozen_noise_variant(stopped):
    noise = stopped.pilot.noise
    assert noise.noise_variant == PLAN.pilot.noise_variant == "ensemble_top_k"
    assert [month.item_id for month in noise.months] == [f"macro-{month}" for month in PLAN.months.pilot]
    results = stopped.pilot.run.results
    for month, wanted in zip(noise.months, PATTERN, strict=False):
        assert (month.identifying, month.anonymised) == wanted
        assert month.identifying == results[(month.item_id, "identifying")].variant("ensemble_top_k").correct
        assert month.anonymised == results[(month.item_id, "anonymised")].variant("ensemble_top_k").correct
        assert month.pre_cutoff and month.missing_reason is None and month.interrupted_draws == 0
    differences = expected_differences(PLAN, PATTERN)
    assert list(noise.differences) == differences
    assert (noise.months_used, noise.months_missing, noise.months_lost_to_interruption) == (len(differences), 0, 0)
    assert noise.mean_difference == pytest.approx(statistics.fmean(differences))
    assert noise.noise_variance == pytest.approx(statistics.variance(differences))
    assert noise.noise_sd == pytest.approx(statistics.stdev(differences))


# ---------- the gate: stop below, continue at or above the threshold ----------

def test_the_gate_stops_below_the_threshold_and_reports_the_expected_power_and_the_options(stopped):
    gate, noise = stopped.pilot.gate, stopped.pilot.noise
    assert (gate.arm, gate.plan_sha256) == ("sampling", PLAN.plan_sha256)
    assert gate.expected_power == expected_power(PLAN, noise.noise_sd)
    assert gate.expected_power < PLAN.analysis.power_threshold
    assert gate.decision == STOP_FOR_DECISION and gate.options == STOP_OPTIONS
    assert "below the threshold" in gate.reason
    summary = gate.summary()
    assert f"expected power {gate.expected_power:.3f}" in summary and "stop_for_decision" in summary
    assert "waits for a decision" in summary and all(option in summary for option in STOP_OPTIONS)
    assert (gate.effect_size, gate.alpha, gate.n_sim, gate.seed, gate.power_threshold) == (
        PLAN.analysis.effect_size, PLAN.analysis.alpha, PLAN.analysis.n_resamples, PLAN.seed,
        PLAN.analysis.power_threshold)
    assert (gate.pre_cutoff_months, gate.post_cutoff_months) == (
        PLAN.pilot.pre_cutoff_evaluation_months, PLAN.pilot.post_cutoff_evaluation_months)


def test_the_gate_continues_at_a_power_above_the_threshold(passed, stopped):
    gate = passed.pilot.gate
    assert passed.pilot.noise.months == stopped.pilot.noise.months  # same replies, same noise
    assert gate.expected_power == expected_power(LARGE, passed.pilot.noise.noise_sd)
    assert gate.expected_power >= LARGE.analysis.power_threshold
    assert (gate.decision, gate.reason, gate.options) == (CONTINUE, None, ())
    assert require_gate_passed(LARGE, "sampling", results_dir=passed.results_dir,
                               gates_dir=passed.gates_dir) == gate


def test_every_gate_size_comes_from_the_frozen_plan_and_the_threshold_is_inclusive(stopped, monkeypatch):
    odd = plan_with(AnalysisSettings(effect_size=0.37, alpha=0.1, power_threshold=0.65, n_resamples=321))
    noise = PilotNoise(plan_sha256=odd.plan_sha256, arm="sampling", noise_variant=odd.pilot.noise_variant,
                       months=stopped.pilot.noise.months)
    calls = []
    for returned, decision in ((0.65, CONTINUE), (math.nextafter(0.65, 0.0), STOP_FOR_DECISION), (0.9, CONTINUE)):
        def simulate(*args, returned=returned, **kwargs):
            calls.append((args, kwargs))
            return returned

        monkeypatch.setattr(pilot_module, "simulated_power_two_sample", simulate)
        gate = power_gate(odd, noise)
        assert (gate.expected_power, gate.decision) == (returned, decision)
    assert {repr(call) for call in calls} == {repr(((0.37, noise.noise_sd, odd.pilot.pre_cutoff_evaluation_months,
                                                     odd.pilot.post_cutoff_evaluation_months),
                                                    {"alpha": 0.1, "n_sim": 321, "seed": odd.seed}))}


def test_the_gate_refuses_noise_of_another_plan_or_variant(stopped):
    noise = stopped.pilot.noise
    with pytest.raises(PlanChangedError):
        power_gate(LARGE, noise)
    with pytest.raises(PilotError, match="noise variant"):
        power_gate(PLAN, replace(noise, noise_variant="single_top_k"))
    with pytest.raises(PilotError, match="pilot months in plan order"):
        power_gate(PLAN, replace(noise, months=tuple(reversed(noise.months))))


# ---------- missing months: interrupted and unparsed draws, no variation ----------

def test_a_month_lost_to_an_interrupted_draw_is_reported_missing_and_counted(tmp_path, calibrated):
    calibrators = calibrated(PLAN)
    assert PLAN.pilot.min_parsed == PLAN.pilot.draws  # one lost draw loses its ensemble
    items = macro_items(PLAN, MACRO_ITEMS, stage="pilot")
    target = items[2]  # the second pilot month, identifying
    assert (target.item_id, target.rendering) == (f"macro-{PLAN.months.pilot[1]}", "identifying")
    lost = ExecutionDraw.interrupted(target, 1, seed=SEED_BASE + 1, temperature=TEMPERATURE,
                                     calibrator_fingerprint=calibrators["top_k"].calibrator.fingerprint)
    outcome = pilot_run(tmp_path, PLAN, calibrators, recorded=[lost])
    assert len(outcome.host.calls) == len(items) * PLAN.pilot.draws - 1  # the interrupted request is never resent
    noise = outcome.pilot.noise
    month = noise.months[1]
    assert month.difference is None and month.identifying is None and month.interrupted_draws == 1
    assert month.lost_to_interruption and month.missing_reason.startswith("identifying: no decision")
    result = outcome.pilot.run.results[target.key]
    assert result.recorded_draws[1].failure_code == INTERRUPTED
    assert (noise.months_used, noise.months_missing, noise.months_lost_to_interruption) == (
        len(PLAN.months.pilot) - 1, 1, 1)
    remaining = [d for n, d in enumerate(expected_differences(PLAN, PATTERN)) if n != 1]
    assert noise.noise_sd == pytest.approx(statistics.stdev(remaining))
    assert outcome.pilot.gate.expected_power == expected_power(PLAN, noise.noise_sd)
    assert "1 missing, 1 lost to an interrupted draw" in outcome.pilot.gate.summary()


def test_an_unparsed_draw_loses_its_month_without_counting_as_an_interruption(tmp_path, calibrated):
    first_of_second_month = 2 * PLAN.pilot.draws  # call order: month, rendering, draw
    outcome = pilot_run(tmp_path, PLAN, calibrated(PLAN), prose={first_of_second_month})
    noise = outcome.pilot.noise
    month = noise.months[1]
    assert month.difference is None and month.interrupted_draws == 0 and not month.lost_to_interruption
    assert month.missing_reason.startswith("identifying: no decision")
    assert (noise.months_used, noise.months_missing, noise.months_lost_to_interruption) == (
        len(PLAN.months.pilot) - 1, 1, 0)


def test_a_pilot_without_variation_stops_without_an_expected_power(tmp_path, calibrated):
    outcome = pilot_run(tmp_path, LARGE, calibrated(LARGE), pattern=FLAT)
    noise, gate = outcome.pilot.noise, outcome.pilot.gate
    assert set(noise.differences) == {0} and noise.noise_sd == 0.0
    assert (gate.expected_power, gate.decision, gate.reason, gate.options) == (
        None, STOP_FOR_DECISION, NO_VARIATION, STOP_OPTIONS)
    assert "expected power not computable" in gate.summary()


def test_fewer_than_two_months_with_the_statistic_stop_without_an_expected_power(stopped):
    months = list(stopped.pilot.noise.months)
    months[1:] = [replace(month, identifying=None, missing_reason="identifying: no decision") for month in months[1:]]
    noise = PilotNoise(plan_sha256=PLAN.plan_sha256, arm="sampling", noise_variant="ensemble_top_k",
                       months=tuple(months))
    gate = power_gate(PLAN, noise)
    assert noise.months_used == 1 and noise.noise_sd is None
    assert (gate.expected_power, gate.decision, gate.reason) == (None, STOP_FOR_DECISION, TOO_FEW_MONTHS)


# ---------- the noise reads the complete pilot of the plan only ----------

def test_the_noise_refuses_an_incomplete_pilot_and_results_of_another_plan_stage_or_arm(stopped, passed, full_run):
    results = list(stopped.pilot.run.results.values())
    with pytest.raises(PilotError, match="the pilot is not complete"):
        estimate_noise(PLAN, results[:-1])
    with pytest.raises(PilotError, match="more than once"):
        estimate_noise(PLAN, [*results, results[0]])
    with pytest.raises(PlanChangedError):
        estimate_noise(PLAN, list(passed.pilot.run.results.values()))
    with pytest.raises(PilotError, match="pilot results only"):
        estimate_noise(PLAN, [*results[1:], next(iter(full_run.run.results.values()))])
    assert estimate_noise(PLAN, results) == stopped.pilot.noise


# ---------- the record and the check before the full run ----------

def test_the_gate_is_recorded_privately_and_reads_back(stopped):
    path = stopped.pilot.gate_path
    assert path == gate_path(PLAN.plan_sha256, stopped.gates_dir) == stopped.gates_dir / f"{PLAN.plan_sha256}.json"
    assert path.is_relative_to(stopped.root) and not path.is_relative_to(GATES_DIR)
    assert not stopped.pilot.run.paths[next(iter(stopped.pilot.run.paths))].is_relative_to(RESULTS_DIR)
    assert load_gate(path) == stopped.pilot.gate
    assert PowerGate.from_dict(json.loads(json.dumps(stopped.pilot.gate.to_dict()))) == stopped.pilot.gate
    text = path.read_text(encoding="utf-8")
    assert "inflation" not in text and "risk_appetite" not in text  # derived values only, no reply text
    assert write_gate(stopped.pilot.gate, stopped.gates_dir) == path  # the same gate again is a no-op


def test_a_rerun_of_the_pilot_reuses_its_results_and_its_gate(stopped, calibrated):
    again = pilot_run(stopped.root, PLAN, calibrated(PLAN), ledger=open_ledger(stopped.root, PLAN))
    assert again.host.calls == [] and again.full.batch_calls == []
    assert set(again.pilot.run.reused) == set(stopped.pilot.run.results)
    assert again.pilot.gate == stopped.pilot.gate and again.pilot.gate_path == stopped.pilot.gate_path


def copy_pilot(source: SimpleNamespace, target: Path) -> SimpleNamespace:
    shutil.copytree(source.results_dir, target / "results")
    shutil.copytree(source.gates_dir, target / "gates")
    return SimpleNamespace(results_dir=target / "results", gates_dir=target / "gates",
                           path=target / "gates" / source.pilot.gate_path.name)


def test_a_stopped_gate_refuses_the_gated_arms_full_run_and_waits_for_a_decision(stopped, tmp_path):
    copy = copy_pilot(stopped, tmp_path)
    with pytest.raises(GateStoppedError) as raised:
        require_gate_passed(PLAN, "sampling", results_dir=copy.results_dir, gates_dir=copy.gates_dir)
    assert raised.value.gate == stopped.pilot.gate
    assert "waits for a decision" in str(raised.value) and STOP_OPTIONS[0] in str(raised.value)


def test_a_missing_or_changed_gate_record_or_an_incomplete_pilot_is_refused(stopped, tmp_path):
    copy = copy_pilot(stopped, tmp_path)
    record = json.loads(copy.path.read_text(encoding="utf-8"))
    forged = record | {"expected_power": 0.95, "decision": CONTINUE, "reason": None, "options": []}
    copy.path.write_text(json.dumps(forged), encoding="utf-8")
    assert load_gate(copy.path).decision == CONTINUE  # a consistent record, but not this pilot's gate
    with pytest.raises(PilotError, match="does not recompute"):
        require_gate_passed(PLAN, "sampling", results_dir=copy.results_dir, gates_dir=copy.gates_dir)
    with pytest.raises(PilotError, match="refusing to replace"):
        write_gate(stopped.pilot.gate, copy.gates_dir)
    with pytest.raises(PilotError, match="decision must be"):
        PowerGate.from_dict(record | {"decision": CONTINUE, "reason": None, "options": []})
    copy.path.unlink()
    with pytest.raises(PilotError, match="no power gate is recorded"):
        require_gate_passed(PLAN, "sampling", results_dir=copy.results_dir, gates_dir=copy.gates_dir)
    next((copy.results_dir / "sampling" / "pilot").iterdir()).unlink()
    with pytest.raises(PilotError, match="the pilot is not complete"):
        require_gate_passed(PLAN, "sampling", results_dir=copy.results_dir, gates_dir=copy.gates_dir)


def test_a_frozen_limit_stops_the_pilot_as_incomplete_before_any_gate(tmp_path, calibrated):
    ledger = open_ledger(tmp_path, PLAN, max_requests=5)
    with pytest.raises(AdmissionClosedError):
        pilot_run(tmp_path, PLAN, calibrated(PLAN), ledger=ledger)
    assert (ledger.status, ledger.stop_reason) == ("incomplete", "max_requests")
    assert not (tmp_path / "gates").exists()


# ---------- the serving arm: not gated, achieved power ----------

def serving_result(item: ExecutionItem, correct: bool):
    """A serving-arm full-run result derived from three identical temperature-0 draws, without a request."""
    text = answer(PLAN, item, correct) if item.realized is not None else json.dumps(DEFENSIVE)
    draws = [RecordedDraw(draw_index=index, temperature=0.0, seed=None, response_sha256=digest(text),
                          parsed=json.loads(text), failure_code=None, estimate=0.25, evidence_ref="e" * 64)
             for index in range(PLAN.arms.arms["serving"].draws)]
    return derive_result(item, draws, None, plan=PLAN, arm="serving", stage="full_run",
                         identities={"top_k": serving_identity(item)})


def serving_correct(number: int) -> tuple[bool, bool]:
    """(identifying correct, anonymised correct) of the serving arm's planned month ``number``."""
    return number % 3 != 0, number % 2 == 0


def serving_results() -> list:
    results = []
    for number, item in enumerate(stage_items(PLAN, "serving")):
        identifying, anonymised = serving_correct(number // 2)
        results.append(serving_result(item, identifying if item.rendering == "identifying" else anonymised))
    return results


def serving_months() -> list[str]:
    return [item_id[len("macro-"):] for item_id in dict.fromkeys(key for key, _ in planned_keys(PLAN, "serving",
                                                                                                 "full_run"))]


def test_the_serving_arm_is_not_gated_and_needs_no_pilot_or_gate_record(tmp_path):
    assert not PLAN.arms.arms["serving"].gated and PLAN.pilot.arm == "sampling"
    assert require_gate_passed(PLAN, "serving", results_dir=tmp_path / "none", gates_dir=tmp_path / "none") is None
    with pytest.raises(PilotError, match="pilot results only"):
        estimate_noise(PLAN, serving_results())


def test_the_serving_arm_reports_its_achieved_power_from_its_realised_counts():
    achieved = achieved_power(PLAN, serving_results(), arm="serving")
    pre = set(PLAN.months.pre_cutoff[GEMMA])
    months = serving_months()
    assert set(months) == set(PLAN.months.evaluation)  # the serving arm runs every evaluation month
    before, after = [], []
    for number, month in enumerate(months):
        if month == MISSING_MONTH:
            continue  # no realized returns: missing, never substituted
        identifying, anonymised = serving_correct(number)
        (before if month in pre else after).append(int(identifying) - int(anonymised))
    pooled = ((len(before) - 1) * statistics.variance(before) + (len(after) - 1) * statistics.variance(after)) / (
        len(before) + len(after) - 2)
    assert (achieved.arm, achieved.noise_variant) == ("serving", "ensemble_top_k")
    assert (achieved.pre_cutoff_months, achieved.post_cutoff_months) == (len(before), len(after)) == (68, 21)
    assert achieved.noise_sd == pytest.approx(math.sqrt(pooled))
    assert achieved.power == simulated_power_two_sample(PLAN.analysis.effect_size, achieved.noise_sd, 68, 21,
                                                        alpha=PLAN.analysis.alpha, n_sim=PLAN.analysis.n_resamples,
                                                        seed=PLAN.seed)
    assert achieved.reason is None and achieved.excluded == {}
    assert [month.item_id for month in achieved.months] == [f"macro-{month}" for month in months]
    gap = next(month for month in achieved.months if month.item_id == f"macro-{MISSING_MONTH}")
    assert gap.difference is None and "not complete" in gap.missing_reason
    assert json.loads(json.dumps(achieved.to_dict()))["power"] == achieved.power


def test_achieved_power_is_missing_with_the_reason_when_a_period_has_fewer_than_two_months():
    pre = set(PLAN.months.pre_cutoff[GEMMA])
    dropped = {f"macro-{month}" for month in serving_months() if month not in pre and month != MISSING_MONTH}
    results = [result for result in serving_results() if result.item_id not in dropped]
    achieved = achieved_power(PLAN, results, arm="serving")
    assert (achieved.post_cutoff_months, achieved.power, achieved.reason) == (0, None, TOO_FEW_PERIOD_MONTHS)
    absent = next(month for month in achieved.months if month.item_id in dropped)
    assert absent.missing_reason == ("identifying: no result for this rendering; anonymised: no result for this "
                                     "rendering")


# ---------- pilot items are excluded from every later analysis ----------

def test_pilot_items_are_excluded_from_every_later_analysis(stopped, full_run):
    pilot = list(stopped.pilot.run.results.values())
    mixed = [*full_run.run.results.values(), *pilot]
    analysed = confirmatory_results(PLAN, mixed, arm="sampling")
    pilot_ids = {f"macro-{month}" for month in PLAN.months.pilot}
    assert set(analysed.results) == set(full_run.run.results)
    assert not any(item_id in pilot_ids for item_id, _ in analysed.results)
    assert set(analysed.excluded) == set(stopped.pilot.run.results)
    assert set(analysed.excluded.values()) == {PILOT_EXCLUDED}
    with_pilot = achieved_power(PLAN, mixed, arm="sampling")
    without = achieved_power(PLAN, list(full_run.run.results.values()), arm="sampling")
    assert with_pilot.months == without.months and (with_pilot.power, with_pilot.noise_sd) == (
        without.power, without.noise_sd)
    assert set(with_pilot.excluded) == set(stopped.pilot.run.results) and without.excluded == {}
    assert not any(month.item_id in pilot_ids for month in with_pilot.months)


def test_the_later_analyses_refuse_results_they_cannot_read(stopped, passed, full_run):
    results = list(full_run.run.results.values())
    with pytest.raises(PilotError, match="enters no later analysis"):
        confirmatory_results(PLAN, results, arm="sampling", stage="pilot")
    with pytest.raises(PlanChangedError):
        confirmatory_results(PLAN, [*results, *passed.pilot.run.results.values()], arm="sampling")
    with pytest.raises(PilotError, match="serving-arm result"):
        confirmatory_results(PLAN, [*results, serving_results()[0]], arm="sampling")
    with pytest.raises(PilotError, match="more than once"):
        confirmatory_results(PLAN, [*results, results[0]], arm="sampling")


def test_month_noise_and_pilot_noise_refuse_inconsistent_values(stopped):
    month = stopped.pilot.noise.months[0]
    with pytest.raises(PilotError, match="missing_reason exactly when"):
        replace(month, identifying=None)
    with pytest.raises(PilotError, match="missing_reason exactly when"):
        replace(month, missing_reason="no reason to be missing")
    data = month.to_dict() | {"difference": 7}
    with pytest.raises(PilotError, match="recorded difference"):
        MonthNoise.from_dict(data)
    noise = stopped.pilot.noise.to_dict() | {"noise_sd": 0.1}
    with pytest.raises(PilotError, match="does not recompute"):
        PilotNoise.from_dict(noise)
    assert PilotNoise.from_dict(json.loads(json.dumps(stopped.pilot.noise.to_dict()))) == stopped.pilot.noise
