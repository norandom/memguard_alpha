"""Host checks, the serve lease, recorded evidence batches and teardown evidence (task 7.4; Requirement 11.1).

Pure logic on fakes under ``tmp_path``: gateway doubles answer the GET routes,
a full-distribution client double answers evidence batches, and the serving arm
runs on a copy of the reviewed Gemma 3 4B manifest with an approval record
written by the tests. No network, credentials, Modal SDK or model host is used.
"""

from __future__ import annotations

import hashlib
import json
import socket
from pathlib import Path

import pytest

from recall_guard.core.completion import ModelConfig, ModelResponse
from recall_guard.core.full_distribution import FullDistributionEvidence
from recall_guard.core.request_control import RequestControl, RequestError
from recall_guard.harness.guarded_decision import GuardConfig, SourceCalibrator
from recall_guard.mia.estimate_source import EstimateSource, SourceLabel
from scripts.finance_sim.limits import (
    EXIT_UNFINISHED,
    AdmissionClosedError,
    ApprovalError,
    LedgerCallBudget,
    RequestAllowance,
    approval_path,
    approval_template,
    present_serving_arm,
    require_serving_approval,
)
from scripts.finance_sim.plan import PlanChangedError, stage_limits_files
from scripts.finance_sim.recording import (
    BUSY,
    DIGEST_MISMATCH,
    EVIDENCE_UNRESOLVED,
    HOST_BUSY,
    HOST_UNAVAILABLE,
    OK,
    UNAUTHORIZED,
    UNREACHABLE,
    EvidencePolicy,
    EvidenceSession,
    HostDigestError,
    HostFailure,
    HttpReply,
    HttpTransportError,
    ServeLease,
    StageSession,
    check_full_distribution_host,
    check_serving_host,
    evidence_path,
    read_records,
    receipt_path,
    record_teardown,
    teardown_evidence,
)
from scripts.probe_modal_model import APPROVAL_ASSERTIONS
from tests.harness.test_guarded_decision import definition, token_evidence
from tests.harness.test_scorer_client import Client, calibrate
from tests.scripts.finance_sim.test_recording import (
    ITEM,
    NOW,
    PINNED,
    PLAN,
    PROMPT,
    RENDERING,
    STAGE,
    Clock,
    Interrupted,
    admitted_host,
    content_for,
    open_session,
    run_item,
    sampling_calibrator,
)

ROOT = Path(__file__).resolve().parents[3]
GEMMA_MANIFEST = ROOT / "configs" / "modal" / "gemma-3-4b-measurement.json"
GEMMA = "google/gemma-3-4b-it"
MODAL_ENDPOINT = "https://wishinet--recall-guard-gemma-3-4b.modal.run/v1"
MODAL_TOKEN = "modal-proxy-bearer-0123456789abcdef"
FD_BASE = "http://127.0.0.1:11581"
OTHER = "sha256:" + "e" * 64


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("the recording attempted a network connection")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)


@pytest.fixture(scope="module")
def calibrator():
    return sampling_calibrator()


# ---------- the full-distribution host ----------


def evidence_for(text: str, digest: str = PINNED) -> FullDistributionEvidence:
    positions = range(4)
    base = -0.1 - 0.01 * (len(text) % 5)
    return FullDistributionEvidence(
        model_digest=digest, n_vocab=50, response_token_ids=tuple(positions),
        logprob=tuple(base - 0.01 * i for i in positions), entropy=(1.0,) * 4, rank=(1,) * 4,
        top_logprobs=tuple(((i, base - 0.01 * i), (40, base - 1.0)) for i in positions))


class FakeFullDistribution:
    """Full-distribution client double: ``script`` lists, per batch call, an error to raise or None to answer."""

    def __init__(self, script=(), *, model_digest: str = PINNED) -> None:
        self.base_url = FD_BASE
        self.model_digest = model_digest
        self.script = list(script)
        self.calls: list[dict] = []

    def evidence_batch(self, prompt_text, response_texts, *, top_k_keep=100, timeout_s=None):
        self.calls.append({"prompt": prompt_text, "responses": list(response_texts), "top_k_keep": top_k_keep,
                           "timeout_s": timeout_s})
        outcome = self.script.pop(0) if self.script else None
        if outcome is not None:
            raise outcome
        return tuple(evidence_for(text, self.model_digest) for text in response_texts)


def host_error(code: str, status: int | None = None) -> RequestError:
    return RequestError(f"full-distribution request rejected ({code})", status_code=status, code=code,
                        retryable=code in {"host_busy", "transport_error"})


def fd_check(client=None, *, connect=None, expected: str = PINNED):
    return check_full_distribution_host(client or FakeFullDistribution(), expected_digest=expected,
                                        connect=connect or (lambda address, timeout: Connection()), now=lambda: NOW)


class Connection:
    def close(self) -> None:
        pass


def refused(address, timeout):
    raise ConnectionRefusedError("[WinError 10061] No connection could be made")


def test_the_full_distribution_check_probes_with_the_pinned_digest():
    client = FakeFullDistribution()
    check = fd_check(client)
    assert (check.host, check.status, check.observed, check.url) == ("full_distribution", OK, PINNED, FD_BASE)
    assert len(client.calls) == 1 and client.calls[0]["top_k_keep"] == 1


@pytest.mark.parametrize(("client", "connect", "status"), [
    (FakeFullDistribution([host_error("digest_mismatch", 409)]), None, DIGEST_MISMATCH),
    (FakeFullDistribution([host_error("auth_failed", 401)]), None, UNAUTHORIZED),
    (FakeFullDistribution([host_error("host_busy", 503)]), None, BUSY),
    (FakeFullDistribution([host_error("transport_error")]), None, BUSY),
    (FakeFullDistribution(), refused, UNREACHABLE),
])
def test_the_full_distribution_check_classifies_the_host(client, connect, status):
    assert fd_check(client, connect=connect).status == status


def test_a_full_distribution_client_for_other_weights_is_refused_without_a_request():
    client = FakeFullDistribution(model_digest=OTHER)
    check = fd_check(client)
    assert (check.status, check.observed) == (DIGEST_MISMATCH, OTHER)
    assert client.calls == []


# ---------- recorded evidence batches ----------


def generated(tmp_path, calibrator):
    """One recorded ensemble of the sampling arm; the generation host is then stopped."""
    with open_session(tmp_path) as session:
        run = run_item(session, admitted_host(session), calibrator)
    return run


def evidence_session(tmp_path, clock=None):
    clock = clock or Clock()
    session = EvidenceSession.open(PLAN, stage=STAGE, records_dir=tmp_path, clock=clock, sleep=clock.sleep,
                                   now=lambda: NOW)
    return session, clock


def stage_receipt(tmp_path) -> dict:
    return json.loads(receipt_path("sampling", STAGE, tmp_path).read_text(encoding="utf-8"))


def evidence_outcomes(tmp_path) -> list[dict]:
    return [line for line in read_records("sampling", STAGE, ITEM, tmp_path) if line["kind"] == "evidence_outcome"]


def evidence_ref(evidence: FullDistributionEvidence) -> str:
    """The RecordedDraw.evidence_ref convention: SHA-256 of the canonical JSON of the evidence payload."""
    payload = json.dumps(evidence.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


POLICY = EvidencePolicy(timeout_s=600, top_k_keep=100, max_busy_retries=2, busy_wait_s=60, max_sends=2,
                        idle_wait_s=300, idle_poll_s=30)


def test_the_default_evidence_policy_is_the_frozen_batch_rule():
    policy = EvidencePolicy()
    assert (policy.timeout_s, policy.max_busy_retries, policy.busy_wait_s, policy.top_k_keep) == (600, 5, 60, 100)


def test_an_evidence_pass_never_opens_the_ledger_records_refs_per_draw_and_is_reused_on_rerun(tmp_path, calibrator):
    run = generated(tmp_path, calibrator)
    budget = tmp_path / "sampling" / STAGE / "budget.json"
    before = budget.read_bytes()
    session, clock = evidence_session(tmp_path)
    client = FakeFullDistribution()
    with session:
        session.require_host(fd_check())
        clock.advance(5000)
        evidence = session.record_evidence(ITEM, RENDERING, PROMPT, run, client=client, probe=fd_check,
                                           policy=POLICY)
    assert budget.read_bytes() == before and not hasattr(session, "ledger")
    assert len(client.calls) == 1 and client.calls[0]["timeout_s"] == 600
    assert client.calls[0]["responses"] == [response.content for response in run.responses]
    assert [item.model_digest for item in evidence] == [PINNED] * len(run.responses)
    lines = [line for line in read_records("sampling", STAGE, ITEM, tmp_path) if line["kind"].startswith("evidence")]
    assert [line["kind"] for line in lines] == ["evidence_request", "evidence_outcome"]
    assert lines[0]["host_digest"] == PINNED and lines[0]["draw_indices"] == list(range(len(run.responses)))
    assert lines[1]["failure_code"] is None
    assert lines[1]["evidence_refs"] == {str(index): evidence_ref(item) for index, item in enumerate(evidence)}
    assert evidence_path("sampling", STAGE, ITEM, RENDERING, tmp_path).is_file()
    assert stage_receipt(tmp_path)["last_session"]["kind"] == "evidence"
    assert stage_receipt(tmp_path)["status"] == "open"  # Generation was not marked complete.

    session, _ = evidence_session(tmp_path)
    again = FakeFullDistribution()
    with session:
        session.require_host(fd_check())
        reused = session.record_evidence(ITEM, RENDERING, PROMPT, run, client=again, probe=fd_check, policy=POLICY)
    assert again.calls == [] and reused == evidence
    assert stage_receipt(tmp_path)["last_session"]["evidence_reused"] == 1


def test_an_evidence_pass_after_complete_generation_leaves_the_stage_complete(tmp_path, calibrator):
    with open_session(tmp_path) as session:
        run = run_item(session, admitted_host(session), calibrator)
        session.complete()
    session, _ = evidence_session(tmp_path)
    with session:
        session.require_host(fd_check())
        session.record_evidence(ITEM, RENDERING, PROMPT, run, client=FakeFullDistribution(), probe=fd_check,
                                policy=POLICY)
    assert (stage_receipt(tmp_path)["status"], session.exit_code) == ("complete", 0)


def test_host_busy_is_retried_a_bounded_number_of_times_then_the_stage_stays_open(tmp_path, calibrator):
    run = generated(tmp_path, calibrator)
    session, clock = evidence_session(tmp_path)
    client = FakeFullDistribution([host_error("host_busy", 503)] * 5)
    with pytest.raises(HostFailure) as failure, session:
        session.require_host(fd_check())
        session.record_evidence(ITEM, RENDERING, PROMPT, run, client=client, probe=fd_check, policy=POLICY)
    assert failure.value.reason == HOST_BUSY and failure.value.exit_code == EXIT_UNFINISHED
    assert len(client.calls) == POLICY.max_busy_retries + 1
    assert clock.sleeps == [POLICY.busy_wait_s] * POLICY.max_busy_retries
    stage = stage_receipt(tmp_path)
    assert (stage["status"], stage["stop_reason"], stage["exit_code"]) == ("open", HOST_BUSY, EXIT_UNFINISHED)
    assert len(read_records("sampling", STAGE, ITEM, tmp_path)) == 2 * len(run.responses) + 2 * len(client.calls)

    session, _ = evidence_session(tmp_path)
    rerun = FakeFullDistribution()
    with session:
        session.require_host(fd_check())
        evidence = session.record_evidence(ITEM, RENDERING, PROMPT, run, client=rerun, probe=fd_check, policy=POLICY)
    assert len(rerun.calls) == 1 and all(evidence)


def test_a_batch_timeout_is_resent_only_after_the_host_is_alive_and_idle(tmp_path, calibrator):
    run = generated(tmp_path, calibrator)
    session, clock = evidence_session(tmp_path)
    client = FakeFullDistribution([host_error("transport_error")])
    probes = iter([fd_check(FakeFullDistribution([host_error("transport_error")])), fd_check()])
    with session:
        session.require_host(fd_check())
        evidence = session.record_evidence(ITEM, RENDERING, PROMPT, run, client=client, probe=lambda: next(probes),
                                           policy=POLICY)
    assert len(client.calls) == 2 and clock.sleeps == [POLICY.idle_poll_s]
    assert all(item is not None for item in evidence)
    outcomes = [line for line in read_records("sampling", STAGE, ITEM, tmp_path) if line["kind"] == "evidence_outcome"]
    assert [line["failure_code"] for line in outcomes] == ["transport_error", None]


def test_a_dead_full_distribution_host_leaves_the_stage_open_without_a_resend(tmp_path, calibrator):
    run = generated(tmp_path, calibrator)
    session, _ = evidence_session(tmp_path)
    client = FakeFullDistribution([host_error("transport_error")])
    with pytest.raises(HostFailure) as failure, session:
        session.require_host(fd_check())
        session.record_evidence(ITEM, RENDERING, PROMPT, run, client=client,
                                probe=lambda: fd_check(connect=refused), policy=POLICY)
    assert failure.value.reason == HOST_UNAVAILABLE and len(client.calls) == 1
    stage = stage_receipt(tmp_path)
    assert (stage["status"], stage["stop_reason"], stage["exit_code"]) == ("open", HOST_UNAVAILABLE,
                                                                           EXIT_UNFINISHED)


def test_a_host_that_stays_busy_after_a_timeout_ends_the_stage_unresolved(tmp_path, calibrator):
    run = generated(tmp_path, calibrator)
    session, clock = evidence_session(tmp_path)
    client = FakeFullDistribution([host_error("transport_error")])
    with pytest.raises(HostFailure) as failure, session:
        session.require_host(fd_check())
        session.record_evidence(ITEM, RENDERING, PROMPT, run, client=client,
                                probe=lambda: fd_check(FakeFullDistribution([host_error("transport_error")])),
                                policy=POLICY)
    assert failure.value.reason == EVIDENCE_UNRESOLVED and len(client.calls) == 1
    assert sum(clock.sleeps) <= POLICY.idle_wait_s


def test_a_refused_batch_is_missing_for_the_session_and_only_that_batch_is_repeated_by_a_rerun(tmp_path, calibrator):
    run = generated(tmp_path, calibrator)
    with open_session(tmp_path) as session:
        assert session.ledger.counters["requests"] == len(run.responses)
    session, _ = evidence_session(tmp_path)
    refusing = FakeFullDistribution([host_error("tokenisation_failed", 422)])
    with session:
        session.require_host(fd_check())
        for _ in range(2):
            evidence = session.record_evidence(ITEM, RENDERING, PROMPT, run, client=refusing, probe=fd_check,
                                               policy=POLICY)
            assert evidence == (None,) * len(run.responses)
    assert len(refusing.calls) == 1
    stage = stage_receipt(tmp_path)
    assert (stage["status"], stage["stop_reason"], stage["exit_code"]) == ("open", "evidence_missing",
                                                                           EXIT_UNFINISHED)
    assert [line["failure_code"] for line in evidence_outcomes(tmp_path)] == ["tokenisation_failed"]

    session, _ = evidence_session(tmp_path)
    rerun = FakeFullDistribution()
    with session:
        session.require_host(fd_check())
        evidence = session.record_evidence(ITEM, RENDERING, PROMPT, run, client=rerun, probe=fd_check, policy=POLICY)
    assert len(rerun.calls) == 1 and all(evidence)
    assert [line["failure_code"] for line in evidence_outcomes(tmp_path)] == ["tokenisation_failed", None]
    generation = [line for line in read_records("sampling", STAGE, ITEM, tmp_path) if line["kind"] == "request"]
    assert len(generation) == len(run.responses)  # A rerun never repeats a generation request.


def test_the_evidence_pass_belongs_to_the_sampling_arm_only():
    with pytest.raises(ValueError, match="sampling arm"):
        EvidenceSession.open(PLAN, arm="serving", stage="full_run", records_dir=Path("unused"))


def test_a_lost_evidence_request_is_resent_only_after_an_idle_check(tmp_path, calibrator):
    run = generated(tmp_path, calibrator)
    session, _ = evidence_session(tmp_path)
    with pytest.raises(Interrupted), session:
        session.require_host(fd_check())
        session.record_evidence(ITEM, RENDERING, PROMPT, run, client=FakeFullDistribution([Interrupted()]),
                                probe=fd_check, policy=POLICY)
    probes = []

    def probe():
        probes.append(fd_check())
        return probes[-1]

    session, _ = evidence_session(tmp_path)
    client = FakeFullDistribution()
    with session:
        session.require_host(fd_check())
        evidence = session.record_evidence(ITEM, RENDERING, PROMPT, run, client=client, probe=probe, policy=POLICY)
    assert len(probes) == 1 and len(client.calls) == 1 and all(evidence)
    outcomes = [line for line in read_records("sampling", STAGE, ITEM, tmp_path) if line["kind"] == "evidence_outcome"]
    assert [line["failure_code"] for line in outcomes] == ["interrupted", None]


def test_evidence_from_a_client_for_other_weights_is_refused_before_a_request(tmp_path, calibrator):
    run = generated(tmp_path, calibrator)
    session, _ = evidence_session(tmp_path)
    client = FakeFullDistribution(model_digest=OTHER)
    with session:
        session.require_host(fd_check())
        with pytest.raises(HostDigestError):
            session.record_evidence(ITEM, RENDERING, PROMPT, run, client=client, probe=fd_check, policy=POLICY)
    assert client.calls == []


def test_an_evidence_policy_beyond_the_gateway_timeout_is_refused():
    with pytest.raises(ValueError, match="timeout_s"):
        EvidencePolicy(timeout_s=1800)
    with pytest.raises(ValueError, match="max_sends"):
        EvidencePolicy(max_sends=0)


# ---------- the serving arm: its host check and the serve lease ----------


SERVING_PARAMETERS = {"temperature": 0.0, "top_p": 1, "max_tokens": 256, "logprobs": True, "top_logprobs": 5}


def serving_config(**overrides) -> ModelConfig:
    values = {"protocol": "openai_chat", "endpoint": MODAL_ENDPOINT + "/chat/completions", "model": GEMMA,
              "credential_ref": "env:MODAL_PROXY_BEARER", "parameters": SERVING_PARAMETERS, "timeout_s": 120,
              "max_retries": 0} | overrides
    return ModelConfig(**values)


class ModelsRoute:
    def __init__(self, ids=(GEMMA,), status=200):
        self.ids, self.status, self.calls = list(ids), status, []

    def __call__(self, url, headers, timeout_s):
        self.calls.append((url, dict(headers)))
        return HttpReply(self.status, {"object": "list", "data": [{"id": model} for model in self.ids]})


def serving_check(route=None):
    return check_serving_host(serving_config(), expected_model=GEMMA, environment={"MODAL_PROXY_BEARER": MODAL_TOKEN},
                              get=route or ModelsRoute(), now=lambda: NOW)


def test_the_serving_check_lists_the_frozen_registry_model_through_the_proxy_bearer():
    route = ModelsRoute()
    check = serving_check(route)
    assert (check.host, check.status, check.observed) == ("serving", OK, GEMMA)
    assert route.calls == [(MODAL_ENDPOINT + "/models", {"Authorization": f"Bearer {MODAL_TOKEN}"})]
    assert MODAL_TOKEN not in json.dumps(check.to_dict())
    assert serving_check(ModelsRoute(ids=("google/gemma-3-12b-it",))).status == DIGEST_MISMATCH
    assert serving_check(ModelsRoute(status=401)).status == UNAUTHORIZED


def test_an_unreachable_gateway_is_reported_not_raised():
    def down(url, headers, timeout_s):
        raise HttpTransportError("unreachable")

    assert serving_check(down).status == UNREACHABLE


@pytest.fixture
def serving(tmp_path):
    plan = PLAN
    raw = json.loads(GEMMA_MANIFEST.read_text(encoding="utf-8"))
    raw["target"]["endpoint"] = MODAL_ENDPOINT
    manifest = tmp_path / "gemma-3-4b-measurement.json"
    manifest.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    stages = []
    for stage, data in stage_limits_files(plan):
        path = tmp_path / "stages" / f"{stage}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        stages.append(path)
    presentation = present_serving_arm(plan, manifest, stages)
    record = approval_template(presentation) | dict.fromkeys(APPROVAL_ASSERTIONS, True) | {
        "provider_budget_reference": "Modal workspace budget reviewed 2026-11-10 (operator note R-7.4)",
        "approved_by": "operator", "approved_at": "2026-11-10T12:30:00+00:00"}
    approvals = tmp_path / "approvals"
    path = approval_path(plan.plan_sha256, approvals)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(record), encoding="utf-8")
    return require_serving_approval(plan, manifest, stages, approvals_dir=approvals)


GPU_STARTED = 1_800_000_000.0


def write_serve_receipt(path: Path, approval, stage: str = "smoke_run", **changes) -> Path:
    limits = approval.limits_for(stage)
    stage_sha = next(item.sha256 for item in approval.presentation.stages if item.stage == stage)
    receipt = {
        "mode": "serve", "status": "serving", "manifest": approval.presentation.manifest,
        "manifest_sha256": approval.presentation.manifest_sha256, "plan_sha256": PLAN.plan_sha256,
        "stage_limits": {"plan_sha256": PLAN.plan_sha256, "stage": stage, "max_requests": limits.max_requests,
                         "max_output_tokens": limits.max_output_tokens, "max_seconds": limits.max_seconds,
                         "max_usd": limits.max_usd},
        "stage_limits_sha256": stage_sha, "volume": "recall-guard-gemma-3-4b-weights",
        "lifetime_seconds": limits.max_seconds, "lifecycle": {}, "endpoint_env": "MODAL_ENDPOINT_URL",
        "credential_env": "MODAL_PROXY_BEARER", "approval_gate": "study_plan_record_checked_by_runner",
        "gpu_started_at_unix_s": GPU_STARTED, "app_id": "ap-0123456789",
        "deadline_unix_s": GPU_STARTED + limits.max_seconds, "endpoint_url": MODAL_ENDPOINT,
        "chat_completions_url": MODAL_ENDPOINT + "/chat/completions",
    } | changes
    path.write_text(json.dumps(receipt), encoding="utf-8")
    return path


class Wall:
    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class FakeModal:
    """vLLM double behind the Modal proxy: temperature 0, no seed; each request advances the wall clock."""

    def __init__(self, budget, wall: Wall, step_s: float) -> None:
        self.config = serving_config()
        self._control = RequestControl(self.config, budget, clock=Clock())
        self.wall, self.step_s, self.calls = wall, step_s, []

    @property
    def budget(self):
        return self._control.budget

    def generate(self, prompt, **parameters):
        def operation():
            self.calls.append(dict(parameters))
            self.wall.now += self.step_s
            return ModelResponse(content_for(len(self.calls)), token_evidence(-0.3), usage={"completion_tokens": 7},
                                 requested_settings=dict(self.config.parameters) | parameters)

        return self._control.run(operation, stage="smoke_run")


def serving_session(tmp_path, approval, receipt: Path, wall: Wall, stage: str = "smoke_run") -> StageSession:
    limits = approval.limits_for(stage)
    allowance = RequestAllowance(output_tokens=256, timeout_s=120, usd_per_second=approval.presentation.usd_per_second)
    return StageSession.open(PLAN, arm="serving", stage=stage, limits=limits, allowance=allowance,
                             approval=approval, serve_receipt=receipt, records_dir=tmp_path / "records",
                             clock=Clock(), wall=wall, now=lambda: NOW)


def test_serving_admission_is_gated_on_the_serve_receipt_deadline(tmp_path, serving):
    receipt = write_serve_receipt(tmp_path / "serve-smoke.json", serving)
    lease = ServeLease.from_receipt(receipt, plan=PLAN, approval=serving, stage="smoke_run")
    deadline = lease.deadline_unix_s
    wall = Wall(GPU_STARTED + 10)
    step = (deadline - 120 - wall.now) / 2 + 1
    calibrator = SourceCalibrator.from_scorer(calibrate(Client(serving_config())), source=EstimateSource.TOP_K,
                                              corpus_sha256=None)
    config = GuardConfig(definition(), SourceLabel(EstimateSource.TOP_K, 5), draws=4, min_parsed=2, temperature=0.0)
    with pytest.raises(AdmissionClosedError) as closed, serving_session(tmp_path, serving, receipt, wall) as session:
        session.require_host(serving_check())
        host = FakeModal(LedgerCallBudget(session.ledger), wall, step)
        session.run_ensemble(ITEM, RENDERING, PROMPT, config, client=host, calibrator=calibrator)
    assert closed.value.reason == "max_seconds" and len(host.calls) == 2
    assert all("seed" not in call for call in host.calls)
    lines = read_records("serving", "smoke_run", ITEM, tmp_path / "records")
    assert [line["kind"] for line in lines] == ["request", "outcome"] * 2
    assert lines[0]["host_digest"] == GEMMA
    stage = json.loads(receipt_path("serving", "smoke_run", tmp_path / "records").read_text(encoding="utf-8"))
    assert (stage["status"], stage["stop_reason"]) == ("incomplete", "max_seconds")


def test_one_serve_per_stage_per_approval(tmp_path, serving):
    first = write_serve_receipt(tmp_path / "serve-1.json", serving)
    with serving_session(tmp_path, serving, first, Wall(GPU_STARTED)):
        pass
    with serving_session(tmp_path, serving, first, Wall(GPU_STARTED + 5)):
        pass
    second = write_serve_receipt(tmp_path / "serve-2.json", serving, app_id="ap-9876543210",
                                 gpu_started_at_unix_s=GPU_STARTED + 4000,
                                 deadline_unix_s=GPU_STARTED + 4000 + serving.limits_for("smoke_run").max_seconds)
    with pytest.raises(ApprovalError, match="one serve"):
        serving_session(tmp_path, serving, second, Wall(GPU_STARTED + 4010))


@pytest.mark.parametrize(("changes", "message"), [
    ({"status": "stopped"}, "serving"),
    ({"mode": "stage"}, "serve receipt"),
    ({"plan_sha256": "0" * 64}, "plan"),
    ({"stage_limits_sha256": "0" * 64}, "stage limits"),
    ({"deadline_unix_s": GPU_STARTED + 10**6}, "deadline"),
    ({"endpoint_url": "https://other--app.modal.run/v1"}, "endpoint"),
])
def test_a_serve_receipt_that_does_not_match_the_approval_is_refused(tmp_path, serving, changes, message):
    receipt = write_serve_receipt(tmp_path / "serve.json", serving, **changes)
    with pytest.raises((ApprovalError, PlanChangedError), match=message):
        ServeLease.from_receipt(receipt, plan=PLAN, approval=serving, stage="smoke_run")


def test_the_serving_arm_needs_its_serve_receipt(tmp_path, serving):
    with pytest.raises(ApprovalError, match="serve receipt"):
        serving_session(tmp_path, serving, None, Wall(GPU_STARTED))


# ---------- teardown evidence ----------


def test_teardown_evidence_reads_the_scripts_stop_results(tmp_path):
    modal = tmp_path / "serve-smoke.json"
    modal.write_text(json.dumps({"mode": "serve", "status": "stopped", "cleanup": "verified_stopped"}),
                     encoding="utf-8")
    unverified = tmp_path / "serve-full.json"
    unverified.write_text(json.dumps({"mode": "serve", "status": "unverified", "cleanup": "unverified",
                                      "recovery_command": "modal app stop ap-1"}), encoding="utf-8")
    stopped = tmp_path / "llama-server-stop.txt"
    stopped.write_text("Stopped: no host or gateway process remains; nothing listens on 11570 or 11571.\n",
                       encoding="utf-8")
    remaining = tmp_path / "full-distribution-stop.txt"
    remaining.write_text("STILL LISTENING 127.0.0.1:11580 pid 4242\n", encoding="utf-8")
    gateway = tmp_path / "gateway.log"
    gateway.write_text("2026-11-10T12:00:00.0000000Z POST /v1/full_distribution_batch 200\n"
                       "2026-11-10T12:30:00.0000000Z leftover check: STILL RUNNING host pid 7\n"
                       "2026-11-10T13:00:00.0000000Z leftover check: no host or gateway process remains; nothing "
                       "listens on 11580 or 11581\n", encoding="utf-8")
    results = {path.name: teardown_evidence(path) for path in (modal, unverified, stopped, remaining, gateway)}
    assert {name: entry["verified_stopped"] for name, entry in results.items()} == {
        "serve-smoke.json": True, "serve-full.json": False, "llama-server-stop.txt": True,
        "full-distribution-stop.txt": False, "gateway.log": True}
    assert results["gateway.log"]["kind"] == "gateway_log"
    assert results["serve-smoke.json"]["kind"] == "modal_serve_receipt"
    assert results["llama-server-stop.txt"]["kind"] == "stop_output"
    assert len(results["gateway.log"]["sha256"]) == 64


def test_teardown_is_recorded_in_the_stage_receipt_of_the_frozen_plan(tmp_path, calibrator):
    with open_session(tmp_path) as session:
        run_item(session, admitted_host(session), calibrator)
    stopped = tmp_path / "stop.txt"
    stopped.write_text("Stopped: no host or gateway process remains; nothing listens on 11570 or 11571.\n",
                       encoding="utf-8")
    entries = record_teardown(PLAN, "sampling", STAGE, [stopped], records_dir=tmp_path, now=lambda: NOW)
    stage = json.loads(receipt_path("sampling", STAGE, tmp_path).read_text(encoding="utf-8"))
    assert stage["teardown"] == entries and stage["teardown_verified"] is True
    assert stage["teardown"][0]["path"] == str(stopped)
    from tests.scripts.finance_sim.test_recording import OTHER_PLAN

    with pytest.raises(PlanChangedError):
        record_teardown(OTHER_PLAN, "sampling", STAGE, [stopped], records_dir=tmp_path)
