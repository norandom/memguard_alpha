"""Guarded execution of study items (task 7.7; Requirements 5.8 and 8.1).

Pure logic on fakes: a generation-host double that answers through
``RequestControl`` with the stage's ``LedgerCallBudget`` (so every request is
admitted by the 7.5 ledger), a full-distribution host double that answers
batches only, calibrators fitted by the 7.6 calibration on those fakes, and
files under ``tmp_path`` only. No network, credentials or model host is used,
and nothing is written under ``runs/``.
"""

from __future__ import annotations

import calendar
import hashlib
import json
import random
import socket
import statistics
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from recall_guard.core.completion import ModelConfig, ModelResponse, TokenAlternative, TokenEvidence
from recall_guard.core.distribution import RecordedDraw
from recall_guard.core.full_distribution import FullDistributionEvidence
from recall_guard.core.request_control import CallBudget, RequestControl, RequestError
from recall_guard.dataset.edgar_outlook import render_outlook_prompt
from recall_guard.dataset.macro_regime import (
    MACRO_AXES,
    MacroRow,
    RealizedReturns,
    defensive_spread,
    is_correct,
    regime_definition,
    render_anonymised,
    render_identifying,
)
from recall_guard.harness.guarded_decision import GuardConfig, GuardIdentity, guarded_run
from recall_guard.harness.scorer_client import CalibrationMismatchError
from recall_guard.mia.estimate_source import EstimateSource, SourceLabel
from scripts.finance_sim import execute as execute_module
from scripts.finance_sim.calibrate import (
    HOST_BUSY_RETRIES,
    HOST_BUSY_WAIT_S,
    DrawSettings,
    EdgarRelease,
    EvidenceSettings,
    EvidenceUnavailableError,
    FitSettings,
    run_calibration,
)
from scripts.finance_sim.execute import (
    EDGAR_NO_OUTCOME,
    INTERRUPTED,
    RESULTS_DIR,
    SIGN_FROM_RAW_REASON,
    ExecutionDraw,
    ExecutionError,
    ExecutionItem,
    ItemResult,
    derive_result,
    execute_items,
    full_distribution_estimate,
    generate_draws,
    load_calibrators,
    load_result,
    macro_items,
    planned_keys,
    write_result,
)
from scripts.finance_sim.items import MacroItem, RealizedOutcome, row_to_dict
from scripts.finance_sim.limits import (
    AdmissionClosedError,
    BudgetLedger,
    LedgerCallBudget,
    RequestAllowance,
    StudyLimits,
)
from scripts.finance_sim.plan import (
    ARM_PROFILE_FACTS,
    ARM_REGISTRY_IDS,
    EDGAR_FILES,
    SERVING_FULL_DISTRIBUTION_REASON,
    VARIANTS,
    ArmPlan,
    EdgarRow,
    MonthClass,
    PilotSettings,
    PlanChangedError,
    StudyArms,
    StudyPlan,
    freeze_plan,
    load_serving_manifest,
    split_months,
)

GEMMA = "google/gemma-3-4b-it"
CREATED_AT = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)
SEED = 20261110
DIGEST = "sha256:" + "f" * 64
#: Small ensembles keep the fakes fast: the probe arm's full run is 20 probe months x 2 renderings x 3 draws.
DRAWS, MIN_PARSED = 3, 2
#: The pilot's own, smaller ensemble (it may not exceed the arm's draws).
PILOT_DRAWS = 2
SEED_BASE, TEMPERATURE = 1000, 0.7
PARAMETERS = {"temperature": 0.0, "top_p": 1, "max_tokens": 512, "logprobs": True, "top_logprobs": 5}
ALLOWANCE = RequestAllowance(output_tokens=512, timeout_s=30)
COMPLETION_TOKENS = 7
#: The last month has no complete next month, so its accuracy and scored probability are missing.
MISSING_MONTH = "2026-06"
MISSING_REASON = "the next month is not complete yet"


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_sha256(value) -> str:
    return digest(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")))


def month_range(first: str, last: str) -> list[str]:
    year, month = int(first[:4]), int(first[5:])
    out = []
    while f"{year:04d}-{month:02d}" <= last:
        out.append(f"{year:04d}-{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return out


MONTHS = month_range("2016-01", "2026-06")


def macro_row(month: str) -> MacroRow:
    year, number = int(month[:4]), int(month[5:])
    k = MONTHS.index(month)
    return MacroRow(decision_date=date(year, number, calendar.monthrange(year, number)[1]),
                    raw={"baa10y": 2.0 + 0.01 * k, "cpi_yoy": 1.5 + 0.02 * k, "t10y2y": 0.5 - 0.01 * k},
                    zscores={"baa10y_z": 0.1 * (k % 7) - 0.3, "cpi_yoy_z": 0.05 * (k % 11) - 0.2,
                             "t10y2y_z": -0.04 * (k % 5) + 0.1},
                    revised={"BAA10Y": False, "CPIAUCSL": False, "T10Y2Y": False})


def realized(month: str) -> RealizedReturns:
    rng = random.Random(f"realized:{month}")
    return RealizedReturns(world_equity=rng.uniform(-0.05, 0.05), tech_sector=rng.uniform(-0.07, 0.07),
                           gold_commodity=rng.uniform(-0.04, 0.04), short_treasury_cash=0.003)


def macro_item(month: str) -> MacroItem:
    row = macro_row(month)
    if month == MISSING_MONTH:
        outcome = RealizedOutcome(row.decision_date, None, MISSING_REASON, {})
    else:
        outcome = RealizedOutcome(row.decision_date, realized(month), None, {})
    return MacroItem(f"macro-{month}", row, {}, outcome)


MACRO_ITEMS = [macro_item(month) for month in MONTHS]


def item_hash(item: MacroItem) -> str:
    """The plan's item hash, recomputed independently: both prompt hashes and the row hash."""
    prompts = {"anonymised": digest(render_anonymised(item.row)), "identifying": digest(render_identifying(item.row))}
    return canonical_sha256({"prompt_sha256": prompts, "row_sha256": canonical_sha256(row_to_dict(item.row))})


EDGAR_ROWS = ([EdgarRow(f"cal-mem-{i:03d}", 1, "calibration") for i in range(40)]
              + [EdgarRow(f"cal-ctl-{i:03d}", 0, "calibration") for i in range(100)]
              + [EdgarRow(f"held-ctl-{i:03d}", 0, "held_out") for i in range(40)]
              + [EdgarRow(f"held-mem-{i:03d}", 1, "held_out") for i in range(40)])


def small_arms() -> StudyArms:
    arms = {arm: replace(ArmPlan.for_arm(arm, **dict(ARM_PROFILE_FACTS[arm])), draws=DRAWS, min_parsed=MIN_PARSED)
            for arm in ("sampling", "serving")}
    return StudyArms(registry_ids=ARM_REGISTRY_IDS, arms=arms)


def frozen_plan(seed: int = SEED) -> StudyPlan:
    months = [MonthClass(month, {GEMMA: month <= "2024-08"}) for month in MONTHS]
    items = {item.item_id: item_hash(item) for item in MACRO_ITEMS} | {row.row_id: digest(row.row_id)
                                                                      for row in EDGAR_ROWS}
    arms = small_arms()
    pilot = replace(PilotSettings.for_split(split_months(months, seed=seed), arms), draws=PILOT_DRAWS,
                    min_parsed=PILOT_DRAWS)
    return freeze_plan(months, EDGAR_ROWS, seed=seed, created_at=CREATED_AT, items=items,
                       edgar_files={name: digest(name) for name in EDGAR_FILES}, arms=arms,
                       serving_manifest=load_serving_manifest(), pilot=pilot)


PLAN = frozen_plan()


def release_text(row_id: str, label: int) -> str:
    marker = " (recalled)" if label == 1 else ""
    return f"Company {row_id} reports quarterly results{marker}. Revenue rose and the outlook is unchanged."


def edgar_releases() -> list[EdgarRelease]:
    releases = [EdgarRelease(row_id, 1, release_text(row_id, 1)) for row_id in PLAN.edgar.calibration_memorized]
    releases += [EdgarRelease(row_id, 0, release_text(row_id, 0)) for row_id in PLAN.edgar.calibration_control]
    return releases


def memorized(prompt: str) -> bool:
    """The fakes' ground truth: identifying macro prompts and memorized EDGAR releases."""
    return "Real asset identities:" in prompt or "(recalled)" in prompt


class Clock:
    """A fake monotonic clock in seconds."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def model_config(**overrides) -> ModelConfig:
    return ModelConfig(protocol="callback", endpoint=None, model="gemma3:4b", parameters={**PARAMETERS, **overrides},
                       timeout_s=30, max_retries=0, retry_backoff_s=0)


def token_evidence(base: float) -> tuple[TokenEvidence, ...]:
    """Six realized tokens with five alternatives each, as a top-k host returns them."""
    values = [min(0.0, base - 0.02 * i) for i in range(6)]
    return tuple(TokenEvidence("t", value, tuple(TokenAlternative(token, value - step)
                                                 for token, step in zip("tuvwx", (0.0, 1.0, 2.0, 3.0, 4.0),
                                                                        strict=True)))
                 for value in values)


def base_logprob(prompt: str, seed: object) -> float:
    """Memorized prompts are likelier; noise depends on the prompt and the draw's seed."""
    noise = random.Random(f"{digest(prompt)}:{seed}").uniform(-0.3, 0.3)
    return (-0.15 if memorized(prompt) else -1.2) + noise


def reply(prompt: str, seed: object) -> str:
    """A reply that varies with the draw, so a first draw can differ from the consensus."""
    rng = random.Random(f"reply:{digest(prompt)}:{seed}")
    if "Press release:" in prompt:
        return json.dumps({"outlook": rng.choice([-0.3, 0.1, 0.2, 0.4])})
    return json.dumps({axis: rng.choice([-0.4, -0.2, 0.0, 0.1, 0.3, 0.5]) for axis in MACRO_AXES})


class FakeHost:
    """Generation-host double; every reply runs through RequestControl with the client's budget.

    ``prose`` lists call indexes answered with unparseable text and ``failures``
    maps call indexes to the error raised instead of a reply.
    """

    def __init__(self, budget: CallBudget, *, configuration: ModelConfig | None = None, prose=(), failures=None,
                 events: list | None = None, crash_at: int | None = None) -> None:
        self.config = configuration or model_config()
        self.crash_at = crash_at
        self.budget = budget
        self._control = RequestControl(self.config, budget, clock=Clock())
        self.prose = set(prose)
        self.failures = dict(failures or {})
        self.events = events
        self.calls: list[tuple[str, dict]] = []

    def generate(self, prompt: str, **parameters) -> ModelResponse:
        index = len(self.calls)
        self.calls.append((prompt, dict(parameters)))
        if self.events is not None:
            self.events.append("generate")

        def operation() -> ModelResponse:
            if index == self.crash_at:
                raise ProcessDied  # after on_request and after the ledger admitted the request
            if index in self.failures:
                raise self.failures[index]
            seed = parameters.get("seed")
            content = "The regime looks benign to me." if index in self.prose else reply(prompt, seed)
            return ModelResponse(content, token_evidence(base_logprob(prompt, seed)),
                                 usage={"completion_tokens": COMPLETION_TOKENS},
                                 requested_settings=dict(self.config.parameters) | parameters)

        return self._control.run(operation, stage="study")


def fake_evidence(prompt: str, response_text: str, position: int) -> FullDistributionEvidence:
    """The fake host's evidence for the reply at ``position`` of a batch: four kept alternatives per position."""
    base = base_logprob(prompt, f"full:{position}:{digest(response_text)}")
    values = [min(0.0, base - 0.01 * i) for i in range(6)]
    return FullDistributionEvidence(
        model_digest=DIGEST, n_vocab=50, response_token_ids=tuple(range(6)), logprob=tuple(values),
        entropy=(1.0,) * 6, rank=(1,) * 6,
        top_logprobs=tuple(((i, value), (40, value - 1.0), (41, value - 2.0), (42, value - 3.0))
                           for i, value in enumerate(values)),
    )


def host_busy() -> RequestError:
    return RequestError("HTTP 503: full-distribution request rejected (host_busy)", status_code=503, code="host_busy",
                        retryable=True)


class FakeFull:
    """Full-distribution host double that answers batches only.

    ``busy`` requests are refused with ``host_busy`` before it answers;
    ``failure`` is raised instead of answering once ``answers`` batches were
    answered (a host that dies part-way).
    """

    def __init__(self, *, busy: int = 0, failure: Exception | None = None, answers: int = 0,
                 events: list | None = None) -> None:
        self.busy = busy
        self.failure = failure
        self.answers = answers
        self.events = events
        self.batch_calls: list[tuple[str, tuple[str, ...], int, float | None]] = []

    def evidence(self, prompt_text: str, response_text: str, *, top_k_keep: int = 100) -> FullDistributionEvidence:
        pytest.fail("the execution requested evidence per draw; it must send one batch per item and rendering")

    def evidence_batch(self, prompt_text: str, response_texts, *, top_k_keep: int = 100,
                       timeout_s: float | None = None) -> tuple[FullDistributionEvidence, ...]:
        self.batch_calls.append((prompt_text, tuple(response_texts), top_k_keep, timeout_s))
        if self.events is not None:
            self.events.append("evidence")
        if self.busy:
            self.busy -= 1
            raise host_busy()
        if self.failure is not None:
            if self.answers == 0:
                raise self.failure
            self.answers -= 1
        return tuple(fake_evidence(prompt_text, text, position) for position, text in enumerate(response_texts))


class Interrupted(Exception):
    """A simulated interruption of the runner after a draw was recorded."""


class ProcessDied(BaseException):
    """A simulated death of the runner process while a request is in flight (nothing catches it)."""


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("the execution attempted a network connection")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)


def stage_limits(stage: str = "full_run", **overrides) -> StudyLimits:
    """The frozen plan's limits of a sampling-arm stage."""
    budget = PLAN.limits.stage("sampling", stage)
    values = {"plan_sha256": PLAN.plan_sha256, "arm": "sampling", "stage": stage,
              "max_requests": budget.max_requests, "max_output_tokens": budget.max_output_tokens,
              "max_seconds": budget.max_seconds} | overrides
    return StudyLimits(**values)


def open_ledger(root: Path, stage: str = "full_run", limits: StudyLimits | None = None) -> BudgetLedger:
    return BudgetLedger.open(root / stage / "budget.json", PLAN, limits or stage_limits(stage), ALLOWANCE,
                             clock=Clock())


@pytest.fixture(scope="module")
def module_root(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("execution")


@pytest.fixture(scope="module")
def calibrators_dir(module_root) -> Path:
    """Macro and EDGAR calibrators of the sampling arm, fitted by the 7.6 calibration on the fakes."""
    ledger = BudgetLedger.open(module_root / "calibration" / "budget.json", PLAN,
                               StudyLimits(PLAN.plan_sha256, "sampling", "calibration", 1000, 1_000_000, 1_000_000),
                               ALLOWANCE, clock=Clock())
    run_calibration(PLAN, arm="sampling", macro_rows=[macro_row(month) for month in PLAN.months.calibration],
                    edgar_releases=edgar_releases(), client=FakeHost(LedgerCallBudget(ledger)), ledger=ledger,
                    settings=DrawSettings(draws=2, temperature=TEMPERATURE, seed_base=SEED_BASE),
                    sources=("top_k", "full_distribution"), full_distribution=FakeFull(),
                    fit=FitSettings(min_auc=0.6, min_valid=10), calibrators_dir=module_root / "calibrators",
                    sleep=lambda seconds: None)
    return module_root / "calibrators"


@pytest.fixture(scope="module")
def macro_calibrators(calibrators_dir):
    return load_calibrators(PLAN, arm="sampling", kind="macro", calibrators_dir=calibrators_dir)


def evaluation_items() -> tuple[ExecutionItem, ...]:
    """The probe arm's full-run items: both renderings of its frozen probe evaluation months."""
    return macro_items(PLAN, MACRO_ITEMS, stage="full_run", arm="sampling")


def subset(months: int = 3) -> tuple[ExecutionItem, ...]:
    chosen = {f"macro-{month}" for month in PLAN.months.probe_evaluation[:months]}
    return tuple(item for item in evaluation_items() if item.item_id in chosen)


def run_stage(root: Path, items, calibrators, *, stage: str = "full_run", host: FakeHost | None = None,
              ledger: BudgetLedger | None = None, full: FakeFull | None = None, recorded=(), on_draw=None,
              on_request=None,
              draws: list | None = None, batches: list | None = None, sleeps: list | None = None,
              events: list | None = None, results_dir: Path | None = None) -> SimpleNamespace:
    """One execution on fakes; the recording (``draws``, ``batches``) can be inspected after it raised."""
    ledger = ledger or open_ledger(root, stage)
    host = host or FakeHost(LedgerCallBudget(ledger), events=events)
    full = full or FakeFull(events=events)
    draws = [] if draws is None else draws
    batches = [] if batches is None else batches
    sleeps = [] if sleeps is None else sleeps

    def record(draw) -> None:
        draws.append(draw)
        if on_draw is not None:
            on_draw(draw)

    outcome = SimpleNamespace(ledger=ledger, host=host, full=full, draws=draws, batches=batches, sleeps=sleeps,
                              results_dir=results_dir or root / "results", run=None)
    outcome.run = execute_items(PLAN, items, arm="sampling", stage=stage, client=host, ledger=ledger,
                                calibrators=calibrators, full_distribution=full, recorded=recorded, on_draw=record,
                                on_request=on_request,
                                on_evidence=lambda item, found: batches.append((item.key, dict(found))),
                                results_dir=outcome.results_dir, sleep=sleeps.append)
    return outcome


def latest(draws) -> list:
    """The recording as a store keeps it: the latest version of each draw."""
    return list({draw.key: draw for draw in draws}.values())


@pytest.fixture(scope="module")
def executed(module_root, macro_calibrators) -> SimpleNamespace:
    """The whole full-run stage of the sampling arm, executed once."""
    return run_stage(module_root / "full", evaluation_items(), macro_calibrators, events=[])


# ---------- one ensemble per item and rendering (5.8, 8.1) ----------

def test_the_full_run_issues_exactly_one_ensemble_per_item_and_rendering_and_completes_the_stage(executed):
    keys = planned_keys(PLAN, "sampling", "full_run")
    assert len(keys) == len(PLAN.months.probe_evaluation) * 2 == 40  # the probe arm's frozen probe months
    calls = executed.host.calls
    assert len(calls) == len(keys) * DRAWS == PLAN.limits.stage("sampling", "full_run").max_requests
    by_prompt: dict[str, list[dict]] = {}
    for prompt, parameters in calls:
        by_prompt.setdefault(prompt, []).append(parameters)
    items = {item.prompt: item for item in evaluation_items()}
    assert set(by_prompt) == set(items) and len(by_prompt) == len(keys)
    for sent in by_prompt.values():
        assert [parameters["seed"] for parameters in sent] == [SEED_BASE + index for index in range(DRAWS)]
        assert {parameters["temperature"] for parameters in sent} == {TEMPERATURE}
    assert executed.ledger.counters["requests"] == len(calls)
    assert executed.ledger.status == executed.run.status == "complete"
    assert set(executed.run.results) == set(keys) and executed.run.reused == ()
    assert len(executed.draws) == len(calls) + len(calls)  # each draw once when made and once when it settles


def test_evidence_is_one_batch_per_item_and_rendering_over_the_parsed_replies_in_draw_order(executed):
    calls = executed.full.batch_calls
    assert len(calls) == 40
    assert len({prompt for prompt, *_ in calls}) == 40
    for prompt, texts, top_k_keep, timeout_s in calls:
        assert texts == tuple(reply(prompt, SEED_BASE + index) for index in range(DRAWS))
        assert (top_k_keep, timeout_s) == (100, 600.0)
    assert [key for key, _ in executed.batches] == list(planned_keys(PLAN, "sampling", "full_run"))
    assert all(sorted(found) == list(range(DRAWS)) for _, found in executed.batches)


def test_generation_finishes_before_the_first_evidence_batch(module_root, macro_calibrators):
    events: list[str] = []
    run_stage(module_root / "order", subset(2), macro_calibrators, events=events)
    first = events.index("evidence")
    assert set(events[:first]) == {"generate"} and "generate" not in events[first:]
    assert events.count("generate") == 2 * 2 * DRAWS and events.count("evidence") == 2 * 2


def test_the_pilot_stage_draws_the_pilots_own_ensemble_with_its_minimum(tmp_path, macro_calibrators):
    month = PLAN.months.pilot[0]
    items = tuple(item for item in macro_items(PLAN, MACRO_ITEMS, stage="pilot") if item.item_id == f"macro-{month}")
    outcome = run_stage(tmp_path, items, macro_calibrators, stage="pilot")
    assert len(outcome.host.calls) == len(items) * PILOT_DRAWS
    assert [parameters["seed"] for _, parameters in outcome.host.calls] == [SEED_BASE, SEED_BASE + 1] * len(items)
    for result in outcome.run.results.values():
        assert (result.stage, result.ensemble_draws, result.min_parsed) == ("pilot", PILOT_DRAWS, PILOT_DRAWS)
        assert {identity.draws for identity in result.identities.values()} == {PILOT_DRAWS}


def test_every_item_result_is_written_privately_and_reads_back(executed, module_root):
    for key, path in executed.run.paths.items():
        assert path.is_relative_to(module_root) and not path.is_relative_to(RESULTS_DIR)
        assert path == executed.results_dir / "sampling" / "full_run" / f"{key[0]}.{key[1]}.json"
        assert load_result(path) == executed.run.results[key]


# ---------- every variant from the one ensemble (8.1) ----------

def test_the_ensemble_top_k_variant_is_what_guarded_run_returns_for_the_same_draws(executed, macro_calibrators):
    items = {item.key: item for item in evaluation_items()}
    config = GuardConfig(regime_definition, SourceLabel(EstimateSource.TOP_K, 5), draws=DRAWS, min_parsed=MIN_PARSED,
                         temperature=TEMPERATURE, seed_base=SEED_BASE)
    for key in list(executed.run.results)[:8]:
        result = executed.run.results[key]
        reference = guarded_run(items[key].prompt, config, client=FakeHost(CallBudget(DRAWS)),
                                calibrator=macro_calibrators["top_k"].calibrator).decision
        variant = result.variant("ensemble_top_k")
        assert result.recorded_draws == reference.draws
        assert result.identities["top_k"] == reference.identity
        assert (variant.status, variant.decision, variant.adjusted, variant.p_memorized, variant.category,
                variant.draws_sha256, variant.not_assessed) == (
            reference.status, reference.decision, reference.adjusted, reference.p_memorized, reference.category,
            reference.draws_sha256, reference.not_assessed)


def test_single_variants_use_draw_index_0_and_adjustments_use_their_sources_estimates_on_the_same_draws(
        executed, macro_calibrators):
    items = {item.key: item for item in evaluation_items()}
    full_record = macro_calibrators["full_distribution"]
    for key in list(executed.run.results)[:8]:
        result = executed.run.results[key]
        prompt = items[key].prompt
        first = result.recorded_draws[0]
        assert first.seed == SEED_BASE and first.parsed == json.loads(reply(prompt, SEED_BASE))
        for name in ("single_none", "single_top_k", "single_full_distribution"):
            assert result.variant(name).decision == dict(first.parsed)
        assert result.variant("single_top_k").p_memorized == first.estimate
        assert result.variant("single_full_distribution").p_memorized == result.full_distribution[0].estimate
        for index, (draw, estimate) in enumerate(zip(result.recorded_draws, result.full_distribution, strict=True)):
            text = reply(prompt, SEED_BASE + index)
            assert estimate == full_distribution_estimate(text, fake_evidence(prompt, text, index), full_record)
            assert draw.estimate is not None and draw.estimate != estimate.estimate
        assert result.variant("ensemble_top_k").p_memorized == pytest.approx(
            statistics.fmean(draw.estimate for draw in result.recorded_draws))
        assert result.variant("ensemble_full_distribution").p_memorized == pytest.approx(
            statistics.fmean(entry.estimate for entry in result.full_distribution))
        for mode in ("single", "ensemble"):
            variants = [result.variant(f"{mode}_{adjustment}") for adjustment in ("none", "top_k", "full_distribution")]
            assert len({variant.draws_sha256 for variant in variants}) == 1  # paired on identical replies
            assert len({json.dumps(dict(variant.decision), sort_keys=True) for variant in variants}) == 1
            none = variants[0]
            assert (none.status, none.p_memorized, none.adjusted) == ("estimate_unavailable", None, none.decision)


def test_every_result_holds_all_six_variants_and_the_draw_level_arrays(executed):
    for result in executed.run.results.values():
        assert [variant.variant for variant in result.variants] == list(VARIANTS)
        assert all(variant.missing is None for variant in result.variants)
        assert result.summary_values == tuple(defensive_spread(draw.parsed) for draw in result.recorded_draws)
        assert result.categories == tuple(regime_definition.decide(draw.parsed) for draw in result.recorded_draws)
        assert result.draws_for("none") == tuple(replace(draw, estimate=None, evidence_ref=None)
                                                 for draw in result.recorded_draws)


def test_scored_probability_and_accuracy_follow_the_frozen_plan(executed):
    items = {item.key: item for item in evaluation_items()}
    clip = PLAN.scored_probability.clip
    for key, result in executed.run.results.items():
        item = items[key]
        if item.realized is None:
            assert result.outcome is None
            for variant in result.variants:
                assert (variant.correct, variant.scored_probability, variant.adjusted_scored_probability) == (
                    None, None, None)
                assert dict(variant.missing_reasons) == {"correct": MISSING_REASON,
                                                         "scored_probability": MISSING_REASON}
            continue
        r = item.realized
        assert result.outcome == int(round((r.gold_commodity + r.short_treasury_cash)
                                           - (r.world_equity + r.tech_sector), 9) > 0)
        for variant in result.variants:
            draws = result.recorded_draws[:1] if variant.mode == "single" else result.recorded_draws
            spreads = [defensive_spread(draw.parsed) for draw in draws if draw.parsed is not None]
            share = sum(round(spread, 9) > 0 for spread in spreads) / len(spreads)
            raw = min(max(share, clip[0]), clip[1])
            assert variant.scored_probability == raw
            p = variant.p_memorized if variant.status == "ok" else None
            expected = raw if p is None else 0.5 + (1 - min(max(p, 0.0), 1.0)) * (raw - 0.5)
            assert variant.adjusted_scored_probability == pytest.approx(expected, abs=1e-12)
            if variant.adjusted_summary_value is None:
                assert variant.correct is None and "correct" in variant.missing_reasons
            else:
                assert variant.correct == is_correct(variant.adjusted_summary_value, r)
    # The month without realized returns is a primary-arm evaluation month outside the seeded probe subset.
    missing = next(item for item in macro_items(PLAN, MACRO_ITEMS, stage="full_run", arm="serving")
                   if item.realized is None)
    result = derive_result(missing, serving_draws(missing), None, plan=PLAN, arm="serving", stage="full_run",
                           identities={"top_k": serving_identity(missing)})
    assert result.outcome is None
    for variant in result.variants:
        if variant.missing is None:
            assert (variant.correct, variant.scored_probability, variant.adjusted_scored_probability) == (
                None, None, None)
            assert dict(variant.missing_reasons) == {"correct": MISSING_REASON, "scored_probability": MISSING_REASON}


# ---------- resume without a duplicate request (5.8) ----------

def test_an_interrupted_run_resumes_from_its_recorded_draws_without_repeating_a_request(
        tmp_path, executed, macro_calibrators):
    items = subset(3)
    recorded: list = []

    def interrupt(draw) -> None:
        if len(recorded) == 5:
            raise Interrupted

    with pytest.raises(Interrupted):
        run_stage(tmp_path, items, macro_calibrators, on_draw=interrupt, draws=recorded)
    ledger = open_ledger(tmp_path)
    assert ledger.status == "open" and ledger.counters["requests"] == 5
    assert not (tmp_path / "results").exists()
    sent_before = [(draw.prompt_sha256, draw.draw.seed) for draw in recorded]
    # The recording stores each draw as JSON and passes it back, as the 7.4 recorder will.
    stored = [ExecutionDraw.from_dict(json.loads(json.dumps(draw.to_dict(), allow_nan=False)))
              for draw in latest(recorded)]
    second = run_stage(tmp_path, items, macro_calibrators, ledger=ledger, recorded=stored)
    total = len(items) * DRAWS
    assert len(second.host.calls) == total - 5
    sent_after = [(digest(prompt), parameters["seed"]) for prompt, parameters in second.host.calls]
    assert len(set(sent_before) | set(sent_after)) == total and not set(sent_before) & set(sent_after)
    assert ledger.counters["requests"] == total and second.run.status == "open"  # some of the stage only
    for key, result in second.run.results.items():
        assert result.to_dict() == executed.run.results[key].to_dict()


def test_a_request_in_flight_when_the_process_dies_is_passed_back_interrupted_and_the_stage_still_completes(
        tmp_path, macro_calibrators):
    items = macro_items(PLAN, MACRO_ITEMS, stage="pilot")  # the whole pilot stage, sized to one ensemble each
    planned = len(items) * PILOT_DRAWS
    assert planned == PLAN.limits.stage("sampling", "pilot").max_requests
    requests: list = []   # the recorder's request lines, written before each send
    outcomes: dict = {}   # its outcome lines, the latest version of each draw

    def on_request(item, index, seed) -> None:
        requests.append((item, index, seed))

    def on_draw(draw) -> None:
        outcomes[draw.key] = draw

    crash = 5
    ledger = open_ledger(tmp_path, "pilot")
    host = FakeHost(LedgerCallBudget(ledger), crash_at=crash)
    with pytest.raises(ProcessDied):
        run_stage(tmp_path, items, macro_calibrators, stage="pilot", ledger=ledger, host=host,
                  on_request=on_request, on_draw=on_draw)
    assert (len(host.calls), len(requests), len(outcomes)) == (crash + 1, crash + 1, crash)
    lost = [(item, index, seed) for item, index, seed in requests if (*item.key, index) not in outcomes]
    assert len(lost) == 1
    interrupted = [ExecutionDraw.interrupted(item, index, seed=seed,
                                             temperature=PLAN.arms.arms["sampling"].temperature,
                                             calibrator_fingerprint=macro_calibrators["top_k"].calibrator.fingerprint)
                   for item, index, seed in lost]
    resumed = open_ledger(tmp_path, "pilot")
    second = run_stage(tmp_path, items, macro_calibrators, stage="pilot", ledger=resumed,
                       recorded=[*outcomes.values(), *interrupted], on_request=on_request, on_draw=on_draw)
    sent = [(prompt, parameters["seed"]) for prompt, parameters in host.calls + second.host.calls]
    assert len(sent) == len(set(sent)) == planned
    assert resumed.counters["requests"] == planned
    assert second.run.status == resumed.status == "complete"
    item, index, seed = lost[0]
    result = second.run.results[item.key]
    draw = result.recorded_draws[index]
    assert (draw.failure_code, draw.seed, draw.temperature, draw.parsed, draw.response_sha256) == (
        INTERRUPTED, seed, TEMPERATURE, None, digest(""))
    assert result.variant("ensemble_top_k").status == "failed"  # the pilot needs both of its two draws
    assert result.full_distribution[index] is None


def test_an_interrupted_draw_is_refused_unless_it_carries_the_draws_seed_temperature_and_calibrator(
        tmp_path, macro_calibrators):
    item = subset(1)[0]
    fingerprint = macro_calibrators["top_k"].calibrator.fingerprint
    for draw, message in (
            (ExecutionDraw.interrupted(item, 0, seed=SEED_BASE + 1, temperature=TEMPERATURE,
                                       calibrator_fingerprint=fingerprint), "seed or temperature"),
            (ExecutionDraw.interrupted(item, 0, seed=SEED_BASE, temperature=0.5, calibrator_fingerprint=fingerprint),
             "seed or temperature"),
            (ExecutionDraw.interrupted(item, 0, seed=SEED_BASE, temperature=TEMPERATURE,
                                       calibrator_fingerprint="0" * 64), "another calibrator")):
        ledger = open_ledger(tmp_path / message.replace(" ", "-"))
        host = FakeHost(LedgerCallBudget(ledger))
        with pytest.raises(ExecutionError, match=message):
            run_stage(tmp_path, (item,), macro_calibrators, ledger=ledger, host=host, recorded=[draw])
        assert host.calls == []


def test_each_ensemble_is_one_guarded_run_prefilled_with_its_recorded_draws(
        tmp_path, executed, macro_calibrators, monkeypatch):
    calls = []
    original = execute_module.guarded_run

    def counting(prompt, config, **kwargs):
        calls.append((prompt, config, tuple(kwargs["recorded"])))
        return original(prompt, config, **kwargs)

    monkeypatch.setattr(execute_module, "guarded_run", counting)
    items = subset(1)
    # As the generation pass recorded them: the evidence pass then batches all three replies again.
    first = [replace(draw, full_distribution=None) for draw in recorded_subset(executed, items)
             if draw.rendering == items[0].rendering and draw.draw.draw_index < 2]
    outcome = run_stage(tmp_path, items, macro_calibrators, recorded=first)
    assert [prompt for prompt, *_ in calls] == [item.prompt for item in items]
    for _, config, _ in calls:
        assert (config.draws, config.min_parsed, config.mode.value, config.source) == (
            DRAWS, MIN_PARSED, "ensemble", SourceLabel(EstimateSource.TOP_K, 5))
        assert (config.temperature, config.seed_base) == (TEMPERATURE, SEED_BASE)
    assert [recorded for *_, recorded in calls] == [tuple(draw.draw for draw in first), ()]
    assert [parameters["seed"] for _, parameters in outcome.host.calls] == [SEED_BASE + 2, *range(SEED_BASE,
                                                                                             SEED_BASE + DRAWS)]
    for key, result in outcome.run.results.items():
        assert result.to_dict() == executed.run.results[key].to_dict()


def test_a_stored_result_is_reused_and_nothing_is_sent_for_it(tmp_path, executed, macro_calibrators):
    items = subset(2)
    first = run_stage(tmp_path, items, macro_calibrators)
    again = run_stage(tmp_path, items, macro_calibrators, ledger=open_ledger(tmp_path))
    assert (len(again.host.calls), len(again.full.batch_calls)) == (0, 0)
    assert again.run.reused == tuple(item.key for item in items)
    assert dict(again.run.results) == dict(first.run.results)
    for key, result in again.run.results.items():
        assert result.to_dict() == executed.run.results[key].to_dict()


def test_the_generation_pass_can_run_alone_and_the_evidence_pass_after_it_sends_no_generation_request(
        tmp_path, executed, macro_calibrators):
    items = subset(2)
    ledger = open_ledger(tmp_path)
    host = FakeHost(LedgerCallBudget(ledger))
    with ledger:
        draws = generate_draws(items, plan=PLAN, arm="sampling", stage="full_run", client=host, ledger=ledger,
                               calibrators=macro_calibrators)
    assert len(host.calls) == len(draws) == len(items) * DRAWS
    assert all(draw.awaiting_evidence and draw.response is not None for draw in draws)
    evidence = run_stage(tmp_path, items, macro_calibrators, ledger=open_ledger(tmp_path), recorded=draws)
    assert len(evidence.host.calls) == 0 and len(evidence.full.batch_calls) == len(items)
    for key, result in evidence.run.results.items():
        assert result.to_dict() == executed.run.results[key].to_dict()


# ---------- the evidence pass: busy, dead host, open stage ----------

def test_host_busy_is_retried_then_the_stage_stays_open_and_a_rerun_sends_only_the_missing_evidence(
        tmp_path, executed, macro_calibrators):
    items = subset(1)
    draws: list = []
    sleeps: list = []
    with pytest.raises(EvidenceUnavailableError) as raised:
        run_stage(tmp_path, items, macro_calibrators, full=FakeFull(busy=HOST_BUSY_RETRIES + 1), draws=draws,
                  sleeps=sleeps)
    assert raised.value.code == "host_busy" and raised.value.prompt_sha256 == items[0].prompt_sha256
    assert sleeps == [HOST_BUSY_WAIT_S] * HOST_BUSY_RETRIES
    ledger = open_ledger(tmp_path)
    assert ledger.status == "open" and ledger.stop_reason is None
    assert len(draws) == len(items) * DRAWS and not (tmp_path / "results").exists()
    rerun = run_stage(tmp_path, items, macro_calibrators, ledger=ledger, recorded=latest(draws))
    assert len(rerun.host.calls) == 0 and len(rerun.full.batch_calls) == len(items)
    for key, result in rerun.run.results.items():
        assert result.to_dict() == executed.run.results[key].to_dict()


def test_a_host_that_dies_part_way_leaves_settled_items_written_and_a_rerun_repeats_only_the_missing_batches(
        tmp_path, executed, macro_calibrators):
    items = subset(2)
    draws: list = []
    dead = RequestError("connection refused", code="transport_error", retryable=True)
    with pytest.raises(EvidenceUnavailableError) as raised:
        run_stage(tmp_path, items, macro_calibrators, full=FakeFull(failure=dead, answers=1), draws=draws)
    assert raised.value.code == "transport_error"
    written = sorted(path.name for path in (tmp_path / "results" / "sampling" / "full_run").iterdir())
    assert written == [f"{items[0].item_id}.{items[0].rendering}.json"]
    rerun = run_stage(tmp_path, items, macro_calibrators, ledger=open_ledger(tmp_path), recorded=latest(draws))
    assert len(rerun.host.calls) == 0
    assert [prompt for prompt, *_ in rerun.full.batch_calls] == [item.prompt for item in items[1:]]
    assert rerun.run.reused == (items[0].key,)
    for key, result in rerun.run.results.items():
        assert result.to_dict() == executed.run.results[key].to_dict()


def test_a_timed_out_batch_is_not_resent_in_the_same_run(tmp_path, macro_calibrators):
    full = FakeFull(failure=TimeoutError("read timed out"))
    with pytest.raises(EvidenceUnavailableError) as raised:
        run_stage(tmp_path, subset(1), macro_calibrators, full=full)
    assert raised.value.code == "timeout" and len(full.batch_calls) == 1


# ---------- limits ----------

def test_a_frozen_limit_closes_admission_marks_the_stage_incomplete_and_keeps_the_recorded_draws(
        tmp_path, macro_calibrators):
    draws: list = []
    ledger = open_ledger(tmp_path, limits=stage_limits(max_requests=4))
    with pytest.raises(AdmissionClosedError):
        run_stage(tmp_path, subset(2), macro_calibrators, ledger=ledger, draws=draws)
    assert (ledger.status, ledger.stop_reason) == ("incomplete", "max_requests")
    assert len(draws) == 4 and not (tmp_path / "results").exists()
    again = open_ledger(tmp_path, limits=stage_limits(max_requests=4))
    host = FakeHost(LedgerCallBudget(again))
    with pytest.raises(AdmissionClosedError):
        run_stage(tmp_path, subset(2), macro_calibrators, ledger=again, host=host, recorded=draws)
    assert host.calls == []


# ---------- unparsed and failed draws ----------

def test_unparsed_and_failed_draws_stay_in_the_ensemble_get_no_evidence_and_fail_their_single_variants(
        tmp_path, macro_calibrators):
    items = subset(1)
    ledger = open_ledger(tmp_path)
    failure = RequestError("HTTP 500: server error", status_code=500, code="http_error", retryable=False)
    host = FakeHost(LedgerCallBudget(ledger), prose={0}, failures={DRAWS + 1: failure})
    outcome = run_stage(tmp_path, items, macro_calibrators, ledger=ledger, host=host)
    identifying, anonymised = (outcome.run.results[item.key] for item in items)
    assert identifying.recorded_draws[0].failure_code == "parse_error"
    assert anonymised.recorded_draws[1].failure_code == "request_error:http_error"
    assert identifying.full_distribution[0] is None and anonymised.full_distribution[1] is None
    texts = [texts for _, texts, *_ in outcome.full.batch_calls]
    assert texts == [tuple(reply(items[0].prompt, SEED_BASE + i) for i in (1, 2)),
                     tuple(reply(items[1].prompt, SEED_BASE + i) for i in (0, 2))]
    for name in ("single_none", "single_top_k", "single_full_distribution"):
        variant = identifying.variant(name)
        assert variant.status == "failed" and variant.decision is None and variant.scored_probability is None
        assert variant.missing_reasons["scored_probability"].startswith("no decision: ")
        assert anonymised.variant(name).status != "failed"
    assert all(identifying.variant(f"ensemble_{a}").status != "failed" for a in ("none", "top_k", "full_distribution"))
    assert identifying.summary_values[0] is None and identifying.categories[0] is None


# ---------- calibrators: plan, arm, kind and source, before any request (4.1, 4.2, 6.3) ----------

def test_calibrators_load_per_source_for_the_plan_arm_and_kind(calibrators_dir, macro_calibrators):
    assert set(macro_calibrators) == {"top_k", "full_distribution"}
    for source, record in macro_calibrators.items():
        assert (record.arm, record.kind, record.source.value, record.plan_sha256) == (
            "sampling", "macro", source, PLAN.plan_sha256)
    with pytest.raises(PlanChangedError):
        load_calibrators(frozen_plan(seed=SEED + 1), arm="sampling", kind="macro", calibrators_dir=calibrators_dir)


def refused_before_any_request(tmp_path, items, calibrators, error, match=None, *, stage="full_run", host=None,
                               ledger=None, results_written: bool = False) -> pytest.ExceptionInfo:
    ledger = ledger or open_ledger(tmp_path, stage)
    host = host or FakeHost(LedgerCallBudget(ledger))
    full = FakeFull()
    with pytest.raises(error, match=match) as raised:
        run_stage(tmp_path, items, calibrators, stage=stage, ledger=ledger, host=host, full=full)
    assert host.calls == [] and full.batch_calls == []
    assert ledger.counters["requests"] == 0 and (tmp_path / "results").exists() == results_written
    return raised


def test_the_edgar_calibrator_is_refused_for_macro_items_and_the_macro_calibrator_for_edgar_items(
        tmp_path, calibrators_dir, macro_calibrators):
    edgar = load_calibrators(PLAN, arm="sampling", kind="edgar", calibrators_dir=calibrators_dir)
    refused_before_any_request(tmp_path / "macro", subset(1), edgar, ExecutionError,
                               "edgar calibrator is refused for macro items")
    refused_before_any_request(tmp_path / "edgar", edgar_items(1), macro_calibrators, ExecutionError,
                               "macro calibrator is refused for edgar items", stage="negative_control")


def test_a_missing_or_swapped_calibrator_is_refused_before_any_request(tmp_path, macro_calibrators):
    refused_before_any_request(tmp_path / "missing", subset(1), {"top_k": macro_calibrators["top_k"]},
                               ExecutionError, "one calibrator for each of its estimate sources")
    swapped = {"top_k": macro_calibrators["full_distribution"], "full_distribution": macro_calibrators["top_k"]}
    refused_before_any_request(tmp_path / "swapped", subset(1), swapped, ExecutionError, "was fitted for the")


def test_a_client_the_calibrators_were_not_fitted_for_is_refused_by_the_guard_before_any_request(
        tmp_path, macro_calibrators):
    ledger = open_ledger(tmp_path)
    host = FakeHost(LedgerCallBudget(ledger), configuration=model_config(top_p=0.9))
    raised = refused_before_any_request(tmp_path, subset(1), macro_calibrators, CalibrationMismatchError,
                                        host=host, ledger=ledger)
    assert raised.value.fields == ("settings",)


@pytest.mark.parametrize(("overrides", "message"), [
    ({"top_logprobs": 3}, "top_logprobs"),
    ({"logprobs": False}, "logprobs"),
])
def test_a_client_off_the_frozen_arm_settings_is_refused_before_any_request(
        tmp_path, macro_calibrators, overrides, message):
    ledger = open_ledger(tmp_path)
    host = FakeHost(LedgerCallBudget(ledger), configuration=model_config(**overrides))
    refused_before_any_request(tmp_path, subset(1), macro_calibrators, ExecutionError, message, host=host,
                               ledger=ledger)


def test_another_served_model_is_refused_before_any_request(tmp_path, macro_calibrators):
    ledger = open_ledger(tmp_path)
    host = FakeHost(LedgerCallBudget(ledger), configuration=replace(model_config(), model="gemma3:12b"))
    refused_before_any_request(tmp_path, subset(1), macro_calibrators, ExecutionError, "the arm serves", host=host,
                               ledger=ledger)


# ---------- stage, items, ledger and budget, before any request ----------

def test_items_outside_the_stage_plan_repeated_or_of_another_kind_are_refused_before_any_request(
        tmp_path, macro_calibrators):
    calibration_month = PLAN.months.calibration[0]
    unplanned = ExecutionItem(f"macro-{calibration_month}", "macro", "identifying",
                              render_identifying(macro_row(calibration_month)))
    refused_before_any_request(tmp_path / "unplanned", (*subset(1), unplanned), macro_calibrators, ExecutionError,
                               "not an item and rendering the plan sizes")
    refused_before_any_request(tmp_path / "repeated", (*subset(1), subset(1)[0]), macro_calibrators,
                               ExecutionError, "more than once")
    refused_before_any_request(tmp_path / "kind", edgar_items(1), macro_calibrators, ExecutionError,
                               "executes macro items")
    refused_before_any_request(tmp_path / "empty", (), macro_calibrators, ExecutionError, "no study items")


def test_stages_without_study_items_and_the_serving_arm_pilot_are_refused():
    with pytest.raises(ExecutionError, match="executes no study items"):
        planned_keys(PLAN, "sampling", "calibration")
    with pytest.raises(ExecutionError, match="gated sampling arm only"):
        planned_keys(PLAN, "serving", "pilot")
    pilot = planned_keys(PLAN, "sampling", "pilot")
    assert pilot == tuple((f"macro-{month}", rendering) for month in PLAN.months.pilot
                          for rendering in ("identifying", "anonymised"))
    assert planned_keys(PLAN, "sampling", "negative_control") == tuple(
        (row_id, "outlook") for row_id in PLAN.edgar.negative_control)


def test_a_ledger_of_another_stage_or_a_client_outside_the_ledger_is_refused_before_any_request(
        tmp_path, macro_calibrators):
    pilot = open_ledger(tmp_path / "pilot", "pilot")
    host = FakeHost(LedgerCallBudget(pilot))
    with pytest.raises(ExecutionError, match="belongs to sampling/pilot"):
        execute_items(PLAN, subset(1), arm="sampling", stage="full_run", client=host, ledger=pilot,
                      calibrators=macro_calibrators, full_distribution=FakeFull(), results_dir=tmp_path / "results")
    ledger = open_ledger(tmp_path / "own")
    stray = FakeHost(CallBudget(100))
    with pytest.raises(ExecutionError, match="admitted by the stage's ledger"):
        execute_items(PLAN, subset(1), arm="sampling", stage="full_run", client=stray, ledger=ledger,
                      calibrators=macro_calibrators, full_distribution=FakeFull(), results_dir=tmp_path / "results")
    assert host.calls == stray.calls == []
    wide = BudgetLedger.open(tmp_path / "wide" / "budget.json", PLAN, stage_limits(),
                             RequestAllowance(output_tokens=1024, timeout_s=30), clock=Clock())
    with pytest.raises(ExecutionError, match="request allowance"):
        execute_items(PLAN, subset(1), arm="sampling", stage="full_run", client=FakeHost(LedgerCallBudget(wide)),
                      ledger=wide, calibrators=macro_calibrators, full_distribution=FakeFull(),
                      results_dir=tmp_path / "results")


def test_the_full_distribution_source_needs_a_batch_client_before_any_request(tmp_path, macro_calibrators):
    ledger = open_ledger(tmp_path)
    host = FakeHost(LedgerCallBudget(ledger))
    with pytest.raises(ExecutionError, match="full-distribution client"):
        execute_items(PLAN, subset(1), arm="sampling", stage="full_run", client=host, ledger=ledger,
                      calibrators=macro_calibrators, full_distribution=None, results_dir=tmp_path / "results")
    assert host.calls == []


def test_evidence_settings_must_keep_the_full_calibrators_alternatives(tmp_path, macro_calibrators):
    ledger = open_ledger(tmp_path)
    host = FakeHost(LedgerCallBudget(ledger))
    with pytest.raises(ExecutionError, match="keep the 100 alternatives"):
        execute_items(PLAN, subset(1), arm="sampling", stage="full_run", client=host, ledger=ledger,
                      calibrators=macro_calibrators, full_distribution=FakeFull(),
                      evidence=EvidenceSettings(top_k_keep=20), results_dir=tmp_path / "results")
    assert host.calls == []


def test_the_evidence_pass_takes_its_timeout_and_busy_retries_from_the_frozen_plan(tmp_path, macro_calibrators):
    frozen = PLAN.arms.arms["sampling"].evidence
    assert (frozen.timeout_s, frozen.top_k_keep, frozen.host_busy_retries, frozen.host_busy_wait_s) == (
        600.0, 100, 5, 60.0)
    # A calibrator recorded with other operational settings: the plan's still apply; top_k_keep is the calibrator's.
    full = macro_calibrators["full_distribution"]
    other = replace(full, evidence_settings=EvidenceSettings(timeout_s=900.0, host_busy_retries=1, host_busy_wait_s=5.0))
    sleeps: list = []
    outcome = run_stage(tmp_path, subset(1), {**macro_calibrators, "full_distribution": other},
                        full=FakeFull(busy=2), sleeps=sleeps)
    assert {(top_k_keep, timeout_s) for _, _, top_k_keep, timeout_s in outcome.full.batch_calls} == {(100, 600.0)}
    assert sleeps == [60.0, 60.0]  # two busy refusals retried at the plan's wait, within its five retries


# ---------- recorded draws and stored results are refused, never merged ----------

def recorded_subset(executed, items) -> list:
    keys = {item.key for item in items}
    return [draw for draw in latest(executed.draws) if (draw.item_id, draw.rendering) in keys]


@pytest.mark.parametrize(("change", "message"), [
    (lambda draw: replace(draw, draw=replace(draw.draw, seed=draw.draw.seed + 1)), "seed or temperature"),
    (lambda draw: replace(draw, calibrator_fingerprint="0" * 64), "top_k estimate comes from another calibrator"),
    (lambda draw: replace(draw, full_distribution=replace(draw.full_distribution, calibrator_fingerprint="0" * 64)),
     "full_distribution estimate comes from another calibrator"),
    (lambda draw: replace(draw, item_id="macro-2016-01"), "not an item and rendering of this run"),
    (lambda draw: replace(draw, draw=replace(draw.draw, draw_index=DRAWS)), "outside the 3 draws"),
])
def test_a_recorded_draw_that_does_not_fit_the_run_is_refused_before_any_request(
        tmp_path, executed, macro_calibrators, change, message):
    items = subset(1)
    recorded = recorded_subset(executed, items)
    recorded[0] = change(recorded[0])
    ledger = open_ledger(tmp_path)
    host = FakeHost(LedgerCallBudget(ledger))
    with pytest.raises(ExecutionError, match=message):
        run_stage(tmp_path, items, macro_calibrators, ledger=ledger, host=host, recorded=recorded)
    assert host.calls == []


def test_a_draw_recorded_twice_keeps_its_settled_version_and_conflicting_copies_are_refused(
        tmp_path, executed, macro_calibrators):
    items = subset(1)
    settled = recorded_subset(executed, items)
    pending = [replace(draw, full_distribution=None) for draw in settled]
    outcome = run_stage(tmp_path / "merge", items, macro_calibrators, recorded=pending + settled)
    assert (len(outcome.host.calls), len(outcome.full.batch_calls)) == (0, 0)
    conflicting = replace(settled[0], full_distribution=replace(settled[0].full_distribution, estimate=0.123))
    ledger = open_ledger(tmp_path / "conflict")
    with pytest.raises(ExecutionError, match="more than once with different content"):
        run_stage(tmp_path / "conflict", items, macro_calibrators, ledger=ledger, recorded=[*settled, conflicting])


def copy_result(executed, item: ExecutionItem, root: Path, edit=None) -> Path:
    source = executed.run.paths[item.key]
    data = json.loads(source.read_text(encoding="utf-8"))
    if edit is not None:
        edit(data)
    target = root / "results" / "sampling" / "full_run" / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data), encoding="utf-8")
    return target


@pytest.mark.parametrize(("edit", "error", "message"), [
    (lambda data: data["variants"][4].update(p_memorized=0.5), ExecutionError, "does not recompute"),
    (lambda data: data.update(plan_sha256="0" * 64), PlanChangedError, None),
    (lambda data: data["identities"]["top_k"].update(calibrator_fingerprint="0" * 64), ExecutionError,
     "another client setting or calibrator"),
    (lambda data: data.update(min_parsed=1), ExecutionError, "another stage, item, prompt or ensemble size"),
    (lambda data: data.update(extra=1), ExecutionError, "fields differ"),
])
def test_a_stored_result_that_differs_is_refused_never_merged_and_nothing_is_sent(
        tmp_path, executed, macro_calibrators, edit, error, message):
    items = subset(1)
    copy_result(executed, items[0], tmp_path, edit)
    refused_before_any_request(tmp_path, items, macro_calibrators, error, message, results_written=True)


def test_a_result_is_never_replaced_by_a_different_one(tmp_path, executed):
    result = executed.run.results[subset(1)[0].key]
    path = write_result(result, tmp_path)
    assert write_result(result, tmp_path) == path
    changed = replace(result, variants=(*result.variants[:4], replace(result.variants[4], p_memorized=0.5),
                                        *result.variants[5:]))
    with pytest.raises(ExecutionError, match="refusing to replace"):
        write_result(changed, tmp_path)


# ---------- serialisation of the recording seam and the results ----------

def test_execution_draws_and_item_results_round_trip_through_json(executed):
    for draw in latest(executed.draws)[:12]:
        assert ExecutionDraw.from_dict(json.loads(json.dumps(draw.to_dict(), allow_nan=False))) == draw
    for result in list(executed.run.results.values())[:6]:
        assert ItemResult.from_dict(json.loads(json.dumps(result.to_dict(), allow_nan=False))) == result
    with pytest.raises(ExecutionError, match="fields differ"):
        ExecutionDraw.from_dict({**latest(executed.draws)[0].to_dict(), "extra": 1})


def test_an_execution_draw_keeps_its_reply_only_when_one_arrived(executed):
    draw = latest(executed.draws)[0]
    with pytest.raises(ExecutionError, match="only a failed request has no reply"):
        replace(draw, response=None)
    with pytest.raises(ExecutionError, match="whose SHA-256 the draw records"):
        replace(draw, response=replace(draw.response, content=draw.response.content + " "))


# ---------- macro items from the frozen plan ----------

def test_macro_items_are_both_renderings_of_every_stage_month_in_plan_order_with_their_returns():
    items = evaluation_items()
    assert [item.key for item in items] == list(planned_keys(PLAN, "sampling", "full_run"))
    by_month = {item.item_id: item for item in MACRO_ITEMS}
    for item in items:
        source = by_month[item.item_id]
        expected = render_identifying(source.row) if item.rendering == "identifying" else render_anonymised(source.row)
        assert (item.kind, item.prompt, item.realized) == ("macro", expected, source.outcome.realized)
    missing = [item for item in macro_items(PLAN, MACRO_ITEMS, stage="full_run", arm="serving")
               if item.item_id == f"macro-{MISSING_MONTH}"]
    assert [item.realized_missing_reason for item in missing] == [MISSING_REASON] * 2
    assert [item.key for item in macro_items(PLAN, MACRO_ITEMS, stage="pilot")] == list(
        planned_keys(PLAN, "sampling", "pilot"))


def test_macro_items_refuse_a_changed_or_missing_item_and_a_stage_without_macro_items():
    month = PLAN.months.probe_evaluation[0]
    changed = [replace(item, row=macro_row(PLAN.months.probe_evaluation[1]) if item.item_id == f"macro-{month}"
                       else item.row) for item in MACRO_ITEMS]
    with pytest.raises(ExecutionError, match="holds the row of"):
        macro_items(PLAN, changed, stage="full_run", arm="sampling")
    edited = [replace(item, row=replace(item.row, raw={**item.row.raw, "baa10y": 9.9}))
              if item.item_id == f"macro-{month}" else item for item in MACRO_ITEMS]
    with pytest.raises(ExecutionError, match="differs from the item the plan froze"):
        macro_items(PLAN, edited, stage="full_run", arm="sampling")
    with pytest.raises(ExecutionError, match="is missing"):
        macro_items(PLAN, [item for item in MACRO_ITEMS if item.item_id != f"macro-{month}"], stage="full_run",
                    arm="sampling")
    with pytest.raises(ExecutionError, match="full_run and pilot"):
        macro_items(PLAN, MACRO_ITEMS, stage="negative_control")


def test_the_full_run_months_depend_on_the_arm_the_probe_arm_runs_its_frozen_subset():
    serving = macro_items(PLAN, MACRO_ITEMS, stage="full_run", arm="serving")
    # Without an arm, the full run's items are every evaluation month (the primary arm's, a superset of the probe's).
    assert macro_items(PLAN, MACRO_ITEMS, stage="full_run") == serving
    assert [item.key for item in serving] == list(planned_keys(PLAN, "serving", "full_run"))
    assert {item.item_id for item in serving} == {f"macro-{month}" for month in PLAN.months.evaluation}
    assert len(serving) == 180 and len(evaluation_items()) == 40
    assert {item.item_id for item in evaluation_items()} == {f"macro-{month}" for month in PLAN.months.probe_evaluation}
    with pytest.raises(ExecutionError, match="gated sampling arm only"):
        macro_items(PLAN, MACRO_ITEMS, stage="pilot", arm="serving")
    # The probe arm refuses an evaluation month outside its frozen subset before any request.
    outside = next(month for month in PLAN.months.evaluation if month not in PLAN.months.probe_evaluation)
    item = next(item for item in serving if item.item_id == f"macro-{outside}")
    with pytest.raises(ExecutionError, match="not an item and rendering the plan sizes"):
        execute_items(PLAN, (item,), arm="sampling", stage="full_run", client=None, ledger=None, calibrators={})


# ---------- the serving arm and a weak calibrator (8.1, 8.10) ----------

def serving_identity(item: ExecutionItem, *, weak: bool = False) -> GuardIdentity:
    return GuardIdentity(model=GEMMA, protocol="openai_compatible", endpoint="https://example.invalid/v1",
                         response_mapping="generic", source=SourceLabel(EstimateSource.TOP_K, 5),
                         settings_json='{"logprobs":true,"max_tokens":256,"temperature":0.0,"top_logprobs":5}',
                         temperature=0.0, draws=DRAWS, calibrator_fingerprint="c" * 64, calibrator_is_weak=weak,
                         prompt_sha256=item.prompt_sha256)


def serving_draws(item: ExecutionItem) -> list[RecordedDraw]:
    text = reply(item.prompt, None)  # temperature 0 without a seed: identical draws
    return [RecordedDraw(draw_index=index, temperature=0.0, seed=None, response_sha256=digest(text),
                         parsed=json.loads(text), failure_code=None, estimate=0.25, evidence_ref="e" * 64)
            for index in range(DRAWS)]


def test_the_serving_arm_reports_its_full_distribution_variants_as_missing_with_the_plans_reason():
    item = subset(1)[0]
    result = derive_result(item, serving_draws(item), None, plan=PLAN, arm="serving", stage="full_run",
                           identities={"top_k": serving_identity(item)})
    reason = PLAN.arms.arms["serving"].missing_variants
    assert set(reason) == {"single_full_distribution", "ensemble_full_distribution"}
    for variant in result.variants:
        if variant.adjustment == "full_distribution":
            assert variant.missing == reason[variant.variant] == SERVING_FULL_DISTRIBUTION_REASON
            assert (variant.status, variant.decision, variant.scored_probability) == (None, None, None)
        else:
            assert variant.missing is None and variant.status in ("ok", "estimate_unavailable")
    assert result.full_distribution is None and set(result.identities) == {"top_k"}
    assert result.variant("ensemble_top_k").p_memorized == pytest.approx(0.25)
    with pytest.raises(ExecutionError, match="computes no full-distribution estimate"):
        derive_result(item, serving_draws(item), [None] * DRAWS, plan=PLAN, arm="serving", stage="full_run",
                      identities={"top_k": serving_identity(item)})


def test_a_weak_calibrator_skips_the_adjustment_and_leaves_the_scored_probability_unadjusted():
    item = subset(1)[0]
    result = derive_result(item, serving_draws(item), None, plan=PLAN, arm="serving", stage="full_run",
                           identities={"top_k": serving_identity(item, weak=True)})
    for name in ("single_top_k", "ensemble_top_k"):
        variant = result.variant(name)
        assert variant.status == "adjustment_skipped_weak_calibrator" and variant.p_memorized == pytest.approx(0.25)
        assert variant.adjusted == variant.decision
        assert variant.adjusted_scored_probability == variant.scored_probability


def test_a_spread_shrunk_to_zero_is_scored_on_the_raw_sign_and_flagged_and_its_forecast_moves_to_one_half():
    item = next(item for item in subset(3) if item.realized is not None
                and round(defensive_spread(json.loads(reply(item.prompt, None))), 9) != 0)
    certain = [replace(draw, estimate=1.0) for draw in serving_draws(item)]  # p_memorized 1: every component to 0
    result = derive_result(item, certain, None, plan=PLAN, arm="serving", stage="full_run",
                           identities={"top_k": serving_identity(item)})
    for name in ("single_top_k", "ensemble_top_k"):
        variant = result.variant(name)
        assert (variant.status, variant.p_memorized) == ("ok", 1.0)
        assert round(variant.adjusted_summary_value, 9) == 0 and round(variant.summary_value, 9) != 0
        assert variant.correct == is_correct(variant.summary_value, item.realized)
        assert (variant.sign_from_raw_spread, variant.sign_reason) == (True, SIGN_FROM_RAW_REASON)
        assert variant.adjusted_scored_probability == 0.5
        assert "correct" not in variant.missing_reasons
    for name in ("single_none", "ensemble_none"):
        variant = result.variant(name)
        assert (variant.sign_from_raw_spread, variant.sign_reason) == (False, None)
        assert variant.correct == is_correct(variant.summary_value, item.realized)
    ordinary = derive_result(item, serving_draws(item), None, plan=PLAN, arm="serving", stage="full_run",
                             identities={"top_k": serving_identity(item)})
    assert not any(variant.sign_from_raw_spread for variant in ordinary.variants)
    assert ItemResult.from_dict(json.loads(json.dumps(result.to_dict()))) == result


def test_derive_refuses_draws_that_are_not_one_complete_settled_ensemble(executed):
    item = subset(1)[0]
    result = executed.run.results[item.key]
    common = {"plan": PLAN, "arm": "sampling", "stage": "full_run", "identities": result.identities}
    with pytest.raises(ExecutionError, match="exactly one draw per index"):
        derive_result(item, result.recorded_draws[:-1], result.full_distribution[:-1], **common)
    with pytest.raises(ExecutionError, match="await their full-distribution evidence"):
        derive_result(item, result.recorded_draws, (None,) * DRAWS, **common)
    with pytest.raises(ExecutionError, match="identity is not that of"):
        derive_result(subset(1)[1], result.recorded_draws, result.full_distribution, **common)
    assert derive_result(item, result.recorded_draws, result.full_distribution, **common) == result


# ---------- EDGAR items through the same entry point ----------

def edgar_items(count: int = 2) -> tuple[ExecutionItem, ...]:
    return tuple(ExecutionItem(row_id, "edgar", "outlook", render_outlook_prompt(release_text(row_id, 0)))
                 for row_id in PLAN.edgar.negative_control[:count])


def test_edgar_items_run_through_the_same_entry_point_with_the_edgar_calibrator(tmp_path, calibrators_dir):
    edgar = load_calibrators(PLAN, arm="sampling", kind="edgar", calibrators_dir=calibrators_dir)
    items = edgar_items(2)
    outcome = run_stage(tmp_path, items, edgar, stage="negative_control")
    assert len(outcome.host.calls) == len(items) * DRAWS and len(outcome.full.batch_calls) == len(items)
    assert outcome.run.status == "open"  # two of the forty negative-control items
    for item in items:
        result = outcome.run.results[item.key]
        assert (result.kind, result.stage, result.outcome) == ("edgar", "negative_control", None)
        for variant in result.variants:
            assert variant.correct is None and variant.scored_probability is None
            assert dict(variant.missing_reasons) == {"correct": EDGAR_NO_OUTCOME, "scored_probability": EDGAR_NO_OUTCOME}
            assert variant.summary_value == variant.decision["outlook"]
        assert result.identities["top_k"].calibrator_fingerprint == edgar["top_k"].calibrator.fingerprint
