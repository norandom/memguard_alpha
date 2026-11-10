"""Request recording and resume of the study runner (task 7.4; Requirements 5.8, 11.1).

Pure logic on fakes under ``tmp_path``: a llama-server double that answers by
seed through ``RequestControl`` with the stage's ``LedgerCallBudget`` (so every
request is admitted by the 7.5 ledger), a gateway double for the host checks,
and fake clocks. No network, credentials or model host is used, and nothing is
written under ``runs/``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import socket
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from recall_guard.core.completion import ModelConfig, ModelResponse
from recall_guard.core.distribution import RecordedDraw
from recall_guard.core.request_control import CallBudget, RequestControl, RequestError
from recall_guard.harness.guarded_decision import GuardConfig, GuardedRun, SourceCalibrator
from recall_guard.mia.estimate_source import EstimateSource, SourceLabel
from scripts.finance_sim.limits import (
    EXIT_COMPLETE,
    EXIT_INCOMPLETE,
    EXIT_REFUSED,
    EXIT_UNFINISHED,
    AdmissionClosedError,
    BudgetLedger,
    LedgerCallBudget,
    RequestAllowance,
    StudyLimits,
)
from scripts.finance_sim.plan import PlanChangedError, StudyPlan
from scripts.finance_sim.recording import (
    HOST_UNAVAILABLE,
    INTERRUPTED_FAILURE,
    UNREACHABLE,
    HostCheck,
    HostDigestError,
    HostFailure,
    HttpReply,
    HttpTransportError,
    RecordingError,
    StageSession,
    check_llama_server,
    pinned_digest,
    read_records,
    receipt_path,
    request_allowance,
    requests_path,
    result_path,
)
from tests.harness.test_guarded_decision import definition, reply, token_evidence
from tests.harness.test_scorer_client import Client, calibrate
from tests.scripts.finance_sim.test_plan_arms import full_plan

PLAN = full_plan()
OTHER_PLAN = full_plan(seed=20261111)
PINNED = pinned_digest(PLAN)
BLOB = "sha256-" + PINNED.removeprefix("sha256:")
GATEWAY = "http://127.0.0.1:11571"
ENDPOINT = f"{GATEWAY}/v1/chat/completions"
TOKEN = "local-gateway-token-0123456789abcdef"
ENVIRONMENT = {"LOCAL_MODEL_API_KEY": TOKEN}
PROMPT = "Macro row for 2019-03 (identifying). Reply with JSON loadings for equity and gold."
OTHER_PROMPT = "Macro row for 2019-03 (anonymised). Reply with JSON loadings for equity and gold."
ITEM, RENDERING = "macro-2019-03", "identifying"
STAGE = "full_run"
DRAWS = 6
SEED_BASE = PLAN.arms.arms["sampling"].seed_base
PARAMETERS = {"temperature": 0.7, "top_p": 1, "max_tokens": 512, "logprobs": True, "top_logprobs": 5}
ALLOWANCE = RequestAllowance(output_tokens=512, timeout_s=300)
COMPLETION_TOKENS = 9
NOW = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)


class Clock:
    """A fake monotonic clock; ``sleep`` advances it."""

    def __init__(self, now: float = 1000.0) -> None:
        self.now = now
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class Interrupted(BaseException):
    """The runner process dies while a request is in flight."""


def sampling_config(**overrides) -> ModelConfig:
    values = {"protocol": "openai_chat", "endpoint": ENDPOINT, "model": "gemma3:4b",
              "credential_ref": "env:LOCAL_MODEL_API_KEY", "parameters": PARAMETERS, "timeout_s": 300,
              "max_retries": 0} | overrides
    return ModelConfig(**values)


def sampling_calibrator() -> SourceCalibrator:
    """A top-k calibrator fitted on the sampling arm's client configuration (endpoint and settings)."""
    return SourceCalibrator.from_scorer(calibrate(Client(sampling_config())), source=EstimateSource.TOP_K,
                                        corpus_sha256=None)


@pytest.fixture(scope="module")
def calibrator() -> SourceCalibrator:
    return sampling_calibrator()


CONFIG = GuardConfig(definition(), SourceLabel(EstimateSource.TOP_K, 5), draws=DRAWS, min_parsed=4,
                     temperature=0.7, seed_base=SEED_BASE)


def content_for(seed: int) -> str:
    step = seed % 3
    return reply(round(0.1 + 0.1 * step, 1), round(0.4 + 0.1 * step, 1))


class FakeLlamaServer:
    """llama-server double behind its gateway: answers by seed, every request admitted by the client's budget.

    ``failures`` maps the index of a sent request to the error the host raises;
    ``interrupt_at`` makes the runner die (``Interrupted``) while that request is in flight.
    """

    def __init__(self, budget: CallBudget, *, configuration: ModelConfig | None = None, failures=None,
                 interrupt_at: int | None = None) -> None:
        self.config = configuration or sampling_config()
        self._control = RequestControl(self.config, budget, clock=Clock())
        self.failures = dict(failures or {})
        self.interrupt_at = interrupt_at
        self.calls: list[dict] = []

    @property
    def budget(self) -> CallBudget:
        return self._control.budget

    def generate(self, prompt: str, **parameters) -> ModelResponse:
        def operation() -> ModelResponse:
            index = len(self.calls)
            self.calls.append({"prompt": prompt, **parameters})
            if index == self.interrupt_at:
                raise Interrupted
            if index in self.failures:
                raise self.failures[index]
            seed = parameters["seed"]
            return ModelResponse(content_for(seed), token_evidence(-0.2 - 0.05 * (seed % 7)),
                                 usage={"completion_tokens": COMPLETION_TOKENS},
                                 requested_settings=dict(self.config.parameters) | parameters)

        return self._control.run(operation, stage=STAGE)


class FakeGateway:
    """The llama-server gateway's GET routes: /v1/models, /props and /health."""

    def __init__(self, *, model_path: str | None = None, models_status: int = 200, unreachable: bool = False):
        self.model_path = model_path or f"D:\\Source\\memguard_alpha\\models\\blobs\\{BLOB}"
        self.models_status = models_status
        self.unreachable = unreachable
        self.calls: list[tuple[str, dict, float]] = []

    def __call__(self, url: str, headers, timeout_s: float) -> HttpReply:
        self.calls.append((url, dict(headers), timeout_s))
        if self.unreachable:
            raise HttpTransportError("unreachable")
        if url.endswith("/v1/models"):
            return HttpReply(self.models_status, {"object": "list", "data": [{"id": self.model_path}]})
        if url.endswith("/props"):
            return HttpReply(200, {"model_path": self.model_path, "total_slots": 1})
        return HttpReply(404, None)


def sampling_limits(plan: StudyPlan = PLAN, **overrides) -> StudyLimits:
    values = {"plan_sha256": plan.plan_sha256, "arm": "sampling", "stage": STAGE, "max_requests": 100,
              "max_output_tokens": 100 * 512, "max_seconds": 100_000} | overrides
    return StudyLimits(**values)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("the recording attempted a network connection")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)


def open_session(root: Path, *, plan: StudyPlan = PLAN, limits: StudyLimits | None = None,
                 allowance: RequestAllowance = ALLOWANCE, clock: Clock | None = None) -> StageSession:
    clock = clock or Clock()
    return StageSession.open(plan, arm="sampling", stage=STAGE, limits=limits or sampling_limits(plan),
                             allowance=allowance, records_dir=root, clock=clock, sleep=clock.sleep, now=lambda: NOW)


def llama_check(gateway: FakeGateway | None = None, *, expected: str = PINNED) -> HostCheck:
    return check_llama_server(sampling_config(), expected_digest=expected, environment=ENVIRONMENT,
                              get=gateway or FakeGateway(), now=lambda: NOW)


def admitted_host(session: StageSession, **options) -> FakeLlamaServer:
    session.require_host(llama_check())
    return FakeLlamaServer(LedgerCallBudget(session.ledger), **options)


def run_item(session, host, calibrator, *, prompt: str = PROMPT, config: GuardConfig = CONFIG, item: str = ITEM,
             rendering: str = RENDERING, probe=None) -> GuardedRun:
    return session.run_ensemble(item, rendering, prompt, config, client=host, calibrator=calibrator, probe=probe)


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def receipt(root: Path) -> dict:
    return json.loads(receipt_path("sampling", STAGE, root).read_text(encoding="utf-8"))


# ---------- recorded requests ----------


def test_every_request_and_its_outcome_are_recorded_privately_per_item_with_plan_hash_and_host_digest(
        tmp_path, calibrator):
    with open_session(tmp_path) as session:
        host = admitted_host(session)
        run = run_item(session, host, calibrator)
        session.complete()
    assert [call["seed"] for call in host.calls] == [SEED_BASE + i for i in range(DRAWS)]
    assert requests_path("sampling", STAGE, ITEM, tmp_path) == tmp_path / "sampling" / STAGE / f"{ITEM}.jsonl"
    lines = read_records("sampling", STAGE, ITEM, tmp_path)
    assert [line["kind"] for line in lines] == ["request", "outcome"] * DRAWS
    for index, (request, outcome) in enumerate(zip(lines[::2], lines[1::2], strict=True)):
        for line in (request, outcome):
            assert (line["plan_sha256"], line["arm"], line["stage"], line["item_id"], line["rendering"],
                    line["draw_index"]) == (PLAN.plan_sha256, "sampling", STAGE, ITEM, RENDERING, index)
        assert request["seed"] == SEED_BASE + index
        assert request["prompt_sha256"] == sha256(PROMPT) == run.decision.identity.prompt_sha256
        assert request["settings_json"] == run.decision.identity.settings_json
        assert request["host_digest"] == PINNED
        assert request["requested_at"] == NOW.isoformat(timespec="seconds")
        draw = run.decision.draws[index]
        assert outcome["sent"] is True and outcome["failure_code"] is None
        assert outcome["response_sha256"] == sha256(content_for(SEED_BASE + index)) == draw.response_sha256
        assert outcome["usage"] == {"completion_tokens": COMPLETION_TOKENS}
        assert outcome["latency_s"] >= 0 and outcome["evidence_ref"] == draw.evidence_ref
        assert RecordedDraw.from_dict(outcome["draw"]) == draw
        assert ModelResponse.from_dict(outcome["response"]) == run.responses[index]
    stored = json.loads(result_path("sampling", STAGE, ITEM, RENDERING, tmp_path).read_text(encoding="utf-8"))
    assert stored["plan_sha256"] == PLAN.plan_sha256 and stored["host_digest"] == PINNED
    assert GuardedRun.from_dict(stored["run"]) == run
    budget = json.loads((tmp_path / "sampling" / STAGE / "budget.json").read_text(encoding="utf-8"))
    assert budget["counters"]["requests"] == DRAWS
    assert budget["counters"]["output_tokens"] == DRAWS * COMPLETION_TOKENS
    stage = receipt(tmp_path)
    assert (stage["status"], stage["exit_code"], stage["plan_sha256"]) == ("complete", EXIT_COMPLETE,
                                                                           PLAN.plan_sha256)
    assert stage["host_checks"][0]["status"] == "ok" and TOKEN not in json.dumps(stage)
    assert TOKEN not in (tmp_path / "sampling" / STAGE / f"{ITEM}.jsonl").read_text(encoding="utf-8")


def test_a_complete_result_is_reused_and_issues_nothing(tmp_path, calibrator):
    with open_session(tmp_path) as session:
        first = run_item(session, admitted_host(session), calibrator)
    with open_session(tmp_path) as session:
        host = admitted_host(session)
        again = run_item(session, host, calibrator)
        assert session.ledger.counters["requests"] == DRAWS
    assert host.calls == [] and again == first
    assert len(read_records("sampling", STAGE, ITEM, tmp_path)) == 2 * DRAWS


def test_renderings_of_one_item_share_its_record_file_and_resume_separately(tmp_path, calibrator):
    with open_session(tmp_path) as session:
        host = admitted_host(session)
        run_item(session, host, calibrator)
        run_item(session, host, calibrator, prompt=OTHER_PROMPT, rendering="anonymised")
    lines = read_records("sampling", STAGE, ITEM, tmp_path)
    assert {line["rendering"] for line in lines} == {"identifying", "anonymised"}
    assert result_path("sampling", STAGE, ITEM, "anonymised", tmp_path).is_file()
    assert len(host.calls) == 2 * DRAWS


# ---------- resume without duplicates ----------


def test_an_interrupted_ensemble_resumes_draw_by_draw_and_never_reissues_a_recorded_request(tmp_path, calibrator):
    with pytest.raises(Interrupted), open_session(tmp_path) as session:
        first = admitted_host(session, interrupt_at=3)
        run_item(session, first, calibrator)
    stage = receipt(tmp_path)
    assert (stage["status"], stage["exit_code"]) == ("open", EXIT_UNFINISHED)
    lines = read_records("sampling", STAGE, ITEM, tmp_path)
    assert [line["kind"] for line in lines] == ["request", "outcome"] * 3 + ["request"]
    assert not result_path("sampling", STAGE, ITEM, RENDERING, tmp_path).exists()

    with open_session(tmp_path) as session:
        second = admitted_host(session)
        run = run_item(session, second, calibrator)
        counters = session.ledger.snapshot()
    sent_first = [call["seed"] for call in first.calls]
    sent_second = [call["seed"] for call in second.calls]
    assert sent_first == [SEED_BASE + i for i in range(4)]
    assert sent_second == [SEED_BASE + 4, SEED_BASE + 5]
    assert not set(sent_first) & set(sent_second)
    lost = run.decision.draws[3]
    assert lost.failure_code == INTERRUPTED_FAILURE and lost.parsed is None
    closing = read_records("sampling", STAGE, ITEM, tmp_path)[7]
    assert (closing["kind"], closing["draw_index"], closing["closed_by"]) == ("outcome", 3, "resume")
    assert counters["counters"]["requests"] == DRAWS and counters["unknown_outcomes"] == 1
    assert run.responses[0].content == content_for(SEED_BASE) and run.responses[3] is None
    assert run.decision.status == "ok"


def test_a_cut_off_last_line_is_dropped_and_the_record_stays_line_aligned(tmp_path, calibrator):
    with pytest.raises(Interrupted), open_session(tmp_path) as session:
        run_item(session, admitted_host(session, interrupt_at=2), calibrator)
    path = requests_path("sampling", STAGE, ITEM, tmp_path)
    with path.open("ab") as stream:
        stream.write(b'{"schema": "finance-sim-request/1", "kind": "outc')
    with open_session(tmp_path) as session:
        run = run_item(session, admitted_host(session), calibrator)
    assert run.decision.draws[2].failure_code == INTERRUPTED_FAILURE
    assert all(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())


def test_a_record_under_another_plan_is_refused_never_merged(tmp_path, calibrator):
    with pytest.raises(Interrupted), open_session(tmp_path / "a") as session:
        run_item(session, admitted_host(session, interrupt_at=2), calibrator)
    source = requests_path("sampling", STAGE, ITEM, tmp_path / "a")
    target = requests_path("sampling", STAGE, ITEM, tmp_path / "b")
    target.parent.mkdir(parents=True)
    shutil.copyfile(source, target)
    with open_session(tmp_path / "b", plan=OTHER_PLAN) as session:
        session.require_host(llama_check())
        host = FakeLlamaServer(LedgerCallBudget(session.ledger))
        with pytest.raises(PlanChangedError, match="never merged"):
            run_item(session, host, calibrator)
    assert host.calls == []
    assert target.read_bytes() == source.read_bytes()


def test_a_record_under_another_prompt_settings_calibrator_or_host_digest_is_refused(tmp_path, calibrator):
    with pytest.raises(Interrupted), open_session(tmp_path) as session:
        run_item(session, admitted_host(session, interrupt_at=2), calibrator)
    path = requests_path("sampling", STAGE, ITEM, tmp_path)
    original = path.read_bytes()
    other_calibrator = replace(calibrator, fingerprint="f" * 64)
    with open_session(tmp_path) as session:
        host = admitted_host(session)
        with pytest.raises(RecordingError, match="prompt"):
            run_item(session, host, calibrator, prompt=PROMPT + " ")
        with pytest.raises(RecordingError, match="settings"):
            run_item(session, host, calibrator, config=GuardConfig(
                definition(), SourceLabel(EstimateSource.TOP_K, 5), draws=DRAWS, min_parsed=3, temperature=0.7,
                seed_base=SEED_BASE))
        with pytest.raises(RecordingError, match="calibrator"):
            run_item(session, host, other_calibrator)
    assert host.calls == [] and path.read_bytes() == original
    lines = [json.loads(line) for line in original.decode("utf-8").splitlines()]
    lines[0]["host_digest"] = "sha256:" + "0" * 64
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    with open_session(tmp_path) as session:
        host = admitted_host(session)
        with pytest.raises(RecordingError, match="host digest"):
            run_item(session, host, calibrator)
    assert host.calls == []


def test_a_stored_result_that_does_not_recompute_is_refused(tmp_path, calibrator):
    with open_session(tmp_path) as session:
        run_item(session, admitted_host(session), calibrator)
    path = result_path("sampling", STAGE, ITEM, RENDERING, tmp_path)
    stored = json.loads(path.read_text(encoding="utf-8"))
    stored["run"]["decision"]["p_memorized"] = 0.123
    path.write_text(json.dumps(stored), encoding="utf-8")
    with open_session(tmp_path) as session:
        host = admitted_host(session)
        with pytest.raises(RecordingError, match="recompute"):
            run_item(session, host, calibrator)
    assert host.calls == []


# ---------- admission through the ledger ----------


def test_admission_runs_only_inside_the_stage_session_through_its_ledger(tmp_path, calibrator):
    session = open_session(tmp_path)
    with pytest.raises(RuntimeError, match="with"):
        run_item(session, FakeLlamaServer(LedgerCallBudget(session.ledger)), calibrator)
    with open_session(tmp_path) as session:
        with pytest.raises(RuntimeError, match="host check"):
            run_item(session, FakeLlamaServer(LedgerCallBudget(session.ledger)), calibrator)
        session.require_host(llama_check())
        cases = {
            "ledger": FakeLlamaServer(CallBudget(100)),
            "max_retries": FakeLlamaServer(LedgerCallBudget(session.ledger), configuration=sampling_config(
                max_retries=2)),
            "max_tokens": FakeLlamaServer(LedgerCallBudget(session.ledger), configuration=sampling_config(
                parameters=PARAMETERS | {"max_tokens": 256})),
            "timeout": FakeLlamaServer(LedgerCallBudget(session.ledger), configuration=sampling_config(
                timeout_s=60)),
            "endpoint": FakeLlamaServer(LedgerCallBudget(session.ledger), configuration=sampling_config(
                endpoint="http://127.0.0.1:11999/v1/chat/completions")),
        }
        for message, host in cases.items():
            with pytest.raises(ValueError, match=message):
                run_item(session, host, calibrator)
            assert host.calls == []
        with pytest.raises(ValueError, match="temperature"):
            run_item(session, FakeLlamaServer(LedgerCallBudget(session.ledger)), calibrator, config=GuardConfig(
                definition(), CONFIG.source, draws=DRAWS, min_parsed=4, temperature=0.2, seed_base=SEED_BASE))
        assert session.ledger.counters["requests"] == 0
    assert not requests_path("sampling", STAGE, ITEM, tmp_path).exists()


def test_the_allowance_is_built_from_the_arm_max_tokens_and_the_client_timeout():
    assert request_allowance(PLAN, "sampling", sampling_config()) == ALLOWANCE
    with pytest.raises(ValueError, match="max_tokens"):
        request_allowance(PLAN, "sampling", sampling_config(parameters=PARAMETERS | {"max_tokens": 1024}))
    with pytest.raises(ValueError, match="max_retries"):
        request_allowance(PLAN, "sampling", sampling_config(max_retries=1))
    with pytest.raises(ValueError, match="max_tokens"):
        StageSession.open(PLAN, arm="sampling", stage=STAGE, limits=sampling_limits(),
                          allowance=RequestAllowance(output_tokens=256, timeout_s=300), records_dir=Path("unused"))


def test_the_pinned_digest_is_the_plans_frozen_gguf_digest_not_the_profiles(monkeypatch):
    from recall_guard.core import profiles
    from scripts.finance_sim.plan import ARM_PROFILE_FACTS, study_arms

    assert PINNED == PLAN.arms.arms["sampling"].artifact_digest
    other = "sha256:" + "1" * 64
    facts = {arm: dict(value) for arm, value in ARM_PROFILE_FACTS.items()}
    facts["sampling"]["artifact_digest"] = other
    frozen = full_plan(arms=study_arms(facts))

    def no_profile(*args, **kwargs):
        raise AssertionError("the pinned digest must come from the frozen plan, not from the provider profile")

    monkeypatch.setattr(profiles, "load_profile", no_profile)
    assert pinned_digest(frozen) == other != pinned_digest(PLAN)
    with pytest.raises(ValueError, match="only the sampling arm"):
        pinned_digest(PLAN, "serving")
    # A plan without the digest (only reachable around the plan's own validation) is refused, never filled in.
    bare = full_plan()
    object.__setattr__(bare.arms.arms["sampling"], "artifact_digest", None)
    with pytest.raises(ValueError, match="no GGUF artifact digest"):
        pinned_digest(bare)


def test_a_frozen_limit_closes_admission_mid_ensemble_and_keeps_the_partial_record(tmp_path, calibrator):
    limits = sampling_limits(max_requests=4)
    with pytest.raises(AdmissionClosedError) as closed, open_session(tmp_path, limits=limits) as session:
        host = admitted_host(session)
        run_item(session, host, calibrator)
    assert closed.value.reason == "max_requests" and len(host.calls) == 4
    lines = read_records("sampling", STAGE, ITEM, tmp_path)
    assert [line["kind"] for line in lines] == ["request", "outcome"] * 5
    assert (lines[-1]["sent"], lines[-1]["failure_code"]) == (False, "not_sent:admission_closed")
    assert not result_path("sampling", STAGE, ITEM, RENDERING, tmp_path).exists()
    stage = receipt(tmp_path)
    assert (stage["status"], stage["stop_reason"], stage["exit_code"]) == ("incomplete", "max_requests",
                                                                           EXIT_INCOMPLETE)
    ledger = BudgetLedger.open(tmp_path / "sampling" / STAGE / "budget.json", PLAN, limits, ALLOWANCE)
    assert (ledger.status, ledger.counters["requests"]) == ("incomplete", 4)

    with pytest.raises(AdmissionClosedError), open_session(tmp_path, limits=limits) as session:
        again = admitted_host(session)
        run_item(session, again, calibrator)
    assert again.calls == [] and len(read_records("sampling", STAGE, ITEM, tmp_path)) == len(lines)


# ---------- a dead host ----------


TRANSPORT = RequestError("Generation request failed in transport", code="transport_error", retryable=True)


def test_a_dead_generation_host_leaves_the_stage_open_without_restart_and_resumable_at_the_next_draw(
        tmp_path, calibrator):
    probes = []

    def probe() -> HostCheck:
        check = llama_check(FakeGateway(unreachable=True))
        probes.append(check)
        return check

    with pytest.raises(HostFailure) as failure, open_session(tmp_path) as session:
        host = admitted_host(session, failures={2: TRANSPORT})
        run_item(session, host, calibrator, probe=probe)
    assert failure.value.reason == HOST_UNAVAILABLE and failure.value.exit_code == EXIT_UNFINISHED
    assert len(host.calls) == 3 and probes[0].status == UNREACHABLE
    stage = receipt(tmp_path)
    assert (stage["status"], stage["stop_reason"], stage["exit_code"]) == ("open", HOST_UNAVAILABLE,
                                                                           EXIT_UNFINISHED)
    assert stage["host_checks"][-1]["status"] == UNREACHABLE
    assert read_records("sampling", STAGE, ITEM, tmp_path)[-1]["failure_code"] == "request_error:transport_error"

    with open_session(tmp_path) as session:
        assert session.ledger.status == "open"
        resumed = admitted_host(session)
        run = run_item(session, resumed, calibrator)
    assert [call["seed"] for call in resumed.calls] == [SEED_BASE + 3, SEED_BASE + 4, SEED_BASE + 5]
    assert run.decision.draws[2].failure_code == "request_error:transport_error"


def test_a_failed_request_on_a_live_host_is_recorded_and_the_ensemble_continues(tmp_path, calibrator):
    with open_session(tmp_path) as session:
        host = admitted_host(session, failures={1: TRANSPORT})
        run = run_item(session, host, calibrator, probe=llama_check)
    assert len(host.calls) == DRAWS
    assert run.decision.draws[1].failure_code == "request_error:transport_error"
    assert run.decision.status == "ok"


def test_a_host_serving_other_weights_is_refused_before_any_generation_request(tmp_path, calibrator):
    other = FakeGateway(model_path="D:\\models\\blobs\\sha256-" + "e" * 64)
    with pytest.raises(HostDigestError), open_session(tmp_path) as session:
        check = llama_check(other)
        assert check.status == "digest_mismatch" and check.observed == "sha256-" + "e" * 64
        session.require_host(check)
    stage = receipt(tmp_path)
    assert (stage["status"], stage["stop_reason"], stage["exit_code"]) == ("refused", "digest_mismatch",
                                                                           EXIT_REFUSED)
    assert not requests_path("sampling", STAGE, ITEM, tmp_path).exists()
    with pytest.raises(HostDigestError, match="pins"), open_session(tmp_path) as session:
        session.require_host(llama_check(FakeGateway(model_path="sha256-" + "e" * 64),
                                         expected="sha256:" + "e" * 64))


def test_a_stage_session_refuses_another_arm_or_a_mismatched_plan(tmp_path):
    with pytest.raises(PlanChangedError):
        StageSession.open(OTHER_PLAN, arm="sampling", stage=STAGE, limits=sampling_limits(PLAN),
                          allowance=ALLOWANCE, records_dir=tmp_path)
    with pytest.raises(ValueError, match="arm"):
        StageSession.open(PLAN, arm="serving", stage=STAGE, limits=sampling_limits(), allowance=ALLOWANCE,
                          records_dir=tmp_path)
    with open_session(tmp_path) as session:
        assert session.directory == tmp_path / "sampling" / STAGE
    stored = receipt(tmp_path)
    stored["plan_sha256"] = OTHER_PLAN.plan_sha256
    receipt_path("sampling", STAGE, tmp_path).write_text(json.dumps(stored), encoding="utf-8")
    with pytest.raises(PlanChangedError):
        open_session(tmp_path)
