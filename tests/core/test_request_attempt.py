"""One physical exchange: no logical retry/poll policy belongs in this seam."""

import json
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor

import pytest

from recall_guard.core import request_control as r
from recall_guard.core.completion import ModelConfig, ModelResponse, TokenEvidence
from recall_guard.core.readiness_evidence import PartialResponse
from recall_guard.core.readiness_types import PendingReceipt, ReadinessObservation

REQUEST_ID = "21a340bd-0628-492a-bf54-9b243e9bb1d9"
TIMESTAMP = "2026-11-01T00:00:00+00:00"


class Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def control(budget, *, clock=None, secrets=(), retry_if=r.default_retryable, **settings):
    clock = clock or Clock()
    config = ModelConfig(protocol="callback", endpoint=None, model="test-model", max_retries=3, **settings)
    return r.RequestControl(config, budget, clock=clock, sleep=clock.sleep, secrets=secrets, retry_if=retry_if)


def pending(**overrides):
    return PendingReceipt(**({"operation_id": "operation-1", "model": "test-model", "request_id": REQUEST_ID,
                              "timestamp": TIMESTAMP, "provider_status": "pending"} | overrides))


@pytest.mark.parametrize("error", [TimeoutError("unknown timeout"), ConnectionError("unknown disconnect"),
                                  r.RequestError("gateway", status_code=504), r.RequestError("busy", status_code=503)])
def test_single_attempt_never_hides_retries_or_backoff(error):
    clock, budget = Clock(), r.CallBudget(10)
    calls = []

    def operation():
        calls.append(True)
        raise error

    with pytest.raises(r.RequestError):
        control(budget, clock=clock).run_attempt(operation, stage="probe", purpose="probe", operation_id="operation-1")
    assert len(calls) == budget.used == 1
    assert clock.sleeps == []
    event = budget.events[0]
    assert (event.purpose, event.operation_id, event.status) == ("probe", "operation-1", "failed")


def test_pending_exchange_succeeds_physically_without_completed_answer_or_poll():
    clock, budget = Clock(), r.CallBudget(10)
    receipt, calls = pending(), []

    def operation():
        calls.append(True)
        assert budget.events[0].purpose == "inference"
        clock.now += 0.25
        return receipt

    result = control(budget, clock=clock).run_attempt(
        operation, stage="target", purpose="inference", operation_id="operation-1")
    assert result == receipt and result.complete is False
    assert not isinstance(result, ModelResponse)
    assert len(calls) == budget.used == 1 and clock.sleeps == []
    event = budget.events[0]
    assert (event.status, event.status_code, event.request_id, event.latency_s) == ("succeeded", 202, REQUEST_ID, 0.25)
    assert event.usage is None and event.cost is None


def test_completed_status_exchange_preserves_operation_and_original_request_identity():
    budget = r.CallBudget(1)
    result = control(budget).run_attempt(
        lambda: ModelResponse("ready", usage={"output_tokens": 3}), stage="target", attempt=2,
        purpose="status", operation_id="operation-1", request_id=REQUEST_ID)
    assert result.content == "ready"
    event = budget.events[0]
    assert (event.call_id, event.attempt, event.purpose, event.request_id) == (1, 2, "status", REQUEST_ID)
    assert event.operation_id == "operation-1" and event.usage == {"output_tokens": 3}


def test_inference_preserves_reported_response_identity():
    budget = r.CallBudget(1)
    control(budget).run_attempt(lambda: ModelResponse("ok", request_id="completion-id"), stage="target", purpose="inference")
    assert budget.events[0].request_id == "completion-id"


@pytest.mark.parametrize("overrides", [{"operation_id": "another-operation"}, {"model": "another-model"},
                                      {"request_id": "5120c66a-4132-496d-8d65-75a900b7741d"}])
def test_pending_identity_mismatch_is_failed_without_retry(overrides):
    budget = r.CallBudget(10)
    with pytest.raises(r.RequestError, match="identity"):
        control(budget).run_attempt(lambda: pending(**overrides), stage="target", purpose="status",
                                    operation_id="operation-1", request_id=REQUEST_ID)
    assert budget.used == 1 and budget.events[0].status == "failed"


@pytest.mark.parametrize("metadata", [{"purpose": "poll"}, {"operation_id": ""}, {"request_id": "bad\nvalue"},
                                     {"attempt": 0}, {"attempt": True}, {"attempt": 1.5}, {"stage": ""}])
def test_invalid_metadata_fails_before_admission(metadata):
    budget = r.CallBudget(1)
    with pytest.raises((ValueError, TypeError)):
        control(budget).run_attempt(lambda: pytest.fail("invalid metadata was admitted"), **({"stage": "target"} | metadata))
    assert budget.used == 0


@pytest.mark.parametrize("status,code", [(400, None), (401, None), (403, None), (422, None),
                                        (429, "insufficient_quota"), (503, "billing_hard_limit_reached")])
def test_permanent_failure_cannot_be_made_retryable(status, code):
    budget = r.CallBudget(10)

    def operation():
        raise r.RequestError("rejected", status_code=status, code=code)

    with pytest.raises(r.RequestError) as caught:
        control(budget, retry_if=lambda _: True).run_attempt(operation, stage="probe")
    assert not caught.value.retryable and budget.used == 1


@pytest.mark.parametrize("hint", ["120", "Sun, 01 Nov 2026 01:00:00 GMT", "invalid", ""])
def test_typed_retry_hint_and_receipt_time_survive_detached_failure(hint):
    budget = r.CallBudget(1)

    def operation():
        raise r.RequestError("busy", status_code=429, request_id=REQUEST_ID,
                             retry_after=r.RetryAfterHint(hint, TIMESTAMP))

    with pytest.raises(r.RequestError) as caught:
        control(budget).run_attempt(operation, stage="probe", purpose="probe")
    error = caught.value
    assert error.retry_after == r.RetryAfterHint(hint, TIMESTAMP)
    assert error.request_id == budget.events[0].request_id == REQUEST_ID
    assert error.__cause__ is None and error.__context__ is None


@pytest.mark.parametrize("partial", [False, True])
def test_failure_retains_typed_response_evidence_and_redacts_all_strings(partial):
    secret, budget = "synthetic-private-key", r.CallBudget(1)
    response_type = PartialResponse if partial else ModelResponse
    response = response_type(f"answer-{secret}", token_evidence=(TokenEvidence(f"token-{secret}", -0.1),),
                             usage={f"tokens-{secret}": 2}, requested_settings={"nested": {"value": [secret]}},
                             request_id=f"id-{secret}")

    def operation():
        raise r.RequestError(secret, code="interrupted" if partial else "missing_token_evidence", response_evidence=response,
                             request_id=f"id-{secret}", retry_after=r.RetryAfterHint(secret, TIMESTAMP))

    with pytest.raises(r.RequestError) as caught:
        control(budget, secrets=(secret,)).run_attempt(operation, stage="probe", purpose="probe", operation_id=secret)
    error = caught.value
    assert isinstance(error.response_evidence, response_type)
    assert error.response_evidence.content == "answer-[REDACTED]"
    assert error.response_evidence.token_evidence[0].token == "token-[REDACTED]"
    assert error.response_evidence.requested_settings["nested"]["value"] == ("[REDACTED]",)
    assert error.response_evidence.usage == {"tokens-[REDACTED]": 2}
    assert error.retry_after.value == "[REDACTED]"
    assert secret not in json.dumps(error.response_evidence.to_dict())
    assert secret not in json.dumps([event.to_dict() for event in budget.events])
    assert secret not in "".join(traceback.format_exception(error))
    assert error.__context__ is None and error.__cause__ is None
    assert budget.used == 1 and budget.events[0].status == "failed"
    assert response.content == f"answer-{secret}"


def test_missing_token_response_is_retained_as_failed_capability_check():
    budget = r.CallBudget(1)
    response = ModelResponse("complete answer", usage={"output_tokens": 4}, request_id="completion-id")

    def operation():
        raise r.RequestError("missing tokens", code="missing_token_evidence", response_evidence=response)

    with pytest.raises(r.RequestError) as caught:
        control(budget).run_attempt(operation, stage="probe", purpose="probe")
    assert caught.value.response_evidence == response
    assert caught.value.response_evidence.mia_unavailable_reason is not None
    assert budget.events[0].usage == response.usage
    assert budget.events[0].request_id == response.request_id
    assert budget.events[0].status == "failed"


def test_partial_result_cannot_be_recorded_as_physical_success():
    budget = r.CallBudget(1)
    with pytest.raises(r.RequestError, match="ModelResponse or PendingReceipt"):
        control(budget).run_attempt(lambda: PartialResponse("unfinished"), stage="target")
    assert budget.events[0].status == "failed"


def test_concurrent_attempts_share_cap_and_do_not_hold_lock_during_operations():
    budget = r.CallBudget(3)
    client = control(budget)
    start = threading.Barrier(12)
    network = threading.Barrier(3)

    def invoke(index):
        start.wait(timeout=5)

        def operation():
            network.wait(timeout=5)
            return ModelResponse("ok")

        try:
            return client.run_attempt(operation, stage="target", purpose="inference", operation_id=f"op-{index}")
        except r.BudgetExhaustedError:
            return None

    with ThreadPoolExecutor(max_workers=12) as executor:
        results = list(executor.map(invoke, range(12)))
    assert sum(result is not None for result in results) == budget.used == 3
    assert [event.call_id for event in budget.events] == [1, 2, 3]
    assert all(event.status == "succeeded" for event in budget.events)


def test_attempts_reuse_actual_start_pacing():
    clock, budget = Clock(), r.CallBudget(3)
    client = control(budget, clock=clock, min_call_interval_s=2)
    for _ in range(3):
        client.run_attempt(lambda: ModelResponse("ok"), stage="target", purpose="inference")
    assert [event.started_at for event in budget.events] == [0, 2, 4]
    assert clock.sleeps == [2, 2]


def test_legacy_run_still_retries_and_does_not_add_event_fields():
    clock, budget = Clock(), r.CallBudget(4)
    calls = []

    def operation():
        calls.append(True)
        if len(calls) < 3:
            raise TimeoutError("unknown outcome")
        return ModelResponse("ok", request_id="old-response-id")

    result = control(budget, clock=clock, retry_backoff_s=0.5).run(operation, stage="target")
    assert result.content == "ok" and budget.used == 3
    assert clock.sleeps == [0.5, 1.0]
    assert [event.attempt for event in budget.events] == [1, 2, 3]
    assert all(not {"purpose", "request_id", "operation_id"}.intersection(event.to_dict()) for event in budget.events)


def test_legacy_run_rejects_pending_instead_of_returning_it_as_an_answer():
    budget = r.CallBudget(4)
    with pytest.raises(r.RequestError, match="ModelResponse"):
        control(budget).run(pending, stage="target")
    assert budget.used == 1 and budget.events[0].status == "failed"


def test_oversized_retry_hint_is_bounded_without_turning_its_prefix_into_a_valid_delay():
    hint = r.RetryAfterHint("1" * 1025, TIMESTAMP)
    assert hint.value == "[oversized Retry-After ignored]"


@pytest.mark.parametrize("value,timestamp", [(True, TIMESTAMP), (None, TIMESTAMP), ("1", "not-a-date"),
                                           ("1", "2026-11-01T00:00:00")])
def test_retry_hint_rejects_untyped_value_or_missing_utc_receipt_time(value, timestamp):
    with pytest.raises((TypeError, ValueError)):
        r.RetryAfterHint(value, timestamp)


@pytest.mark.parametrize("secret", ["token", "content", "missing_token_evidence"])
def test_evidence_redaction_preserves_schema_when_secret_matches_a_field_or_diagnostic_code(secret):
    budget = r.CallBudget(1)

    def operation():
        raise r.RequestError("unavailable", code="missing_token_evidence",
                             response_evidence=ModelResponse(secret, token_evidence=(TokenEvidence(secret, -0.1),)))

    with pytest.raises(r.RequestError) as caught:
        control(budget, secrets=(secret,)).run_attempt(operation, stage="probe")
    assert caught.value.response_evidence.content == "[REDACTED]"
    assert caught.value.response_evidence.token_evidence[0].token == "[REDACTED]"
    assert caught.value.__context__ is None and caught.value.__cause__ is None
    assert budget.events[0].status == "failed"


def test_unrepresentable_redacted_pending_trace_does_not_leak_exception_or_invent_identity():
    budget = r.CallBudget(1)
    trace = (ReadinessObservation("operation-1", "probe", "test-model", TIMESTAMP, 0, "pending", request_id=REQUEST_ID),)

    def operation():
        raise r.RequestError("missing evidence", code="missing_token_evidence",
                             response_evidence=ModelResponse("answer", readiness_trace=trace))

    with pytest.raises(r.RequestError) as caught:
        control(budget, secrets=(REQUEST_ID,)).run_attempt(operation, stage="probe")
    assert caught.value.response_evidence.content == "answer"
    assert caught.value.response_evidence.readiness_trace is None
    assert REQUEST_ID not in json.dumps(caught.value.response_evidence.to_dict())
    assert caught.value.__context__ is None and caught.value.__cause__ is None
    assert budget.events[0].status == "failed"


def test_partial_redaction_cannot_exceed_original_retention_limit():
    budget = r.CallBudget(1)
    response = PartialResponse("X" * 500, max_retained_bytes=1000)

    def operation():
        raise r.RequestError("interrupted", response_evidence=response)

    with pytest.raises(r.RequestError) as caught:
        control(budget, secrets=("X",)).run_attempt(operation, stage="probe")
    retained = caught.value.response_evidence
    assert isinstance(retained, PartialResponse) and not retained.usable_answer
    assert retained.content == "" and retained.reason == response.reason
    assert retained.max_retained_bytes == 1000
    assert len(json.dumps(retained.to_dict(), ensure_ascii=False).encode("utf-8")) <= 1000
    assert caught.value.__context__ is None and budget.events[0].status == "failed"


def test_invalid_returned_request_id_finishes_as_failure_instead_of_abandoning_started_event():
    budget = r.CallBudget(1)
    with pytest.raises(r.RequestError, match="request_id") as caught:
        control(budget).run_attempt(lambda: ModelResponse("ok", request_id="invalid\nidentity"), stage="target")
    assert budget.events[0].status == "failed"
    assert caught.value.__context__ is None
