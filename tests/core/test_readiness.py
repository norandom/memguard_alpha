"""Deterministic readiness decisions, physical admission and elapsed snapshots."""

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from datetime import datetime

import pytest

from recall_guard.core.completion import FeatureChoices, ModelConfig, ModelResponse, TokenAlternative, TokenEvidence
from recall_guard.core.readiness import ReadinessControl, retry_after_seconds
from recall_guard.core.readiness_types import CancellationToken, PendingReceipt, ReadinessPolicy
from recall_guard.core.request_control import CallBudget, RequestControl, RequestError, RetryAfterHint

STAMP = "2026-11-01T12:00:00+00:00"
REQUEST_ID = "12345678-1234-1234-1234-123456789abc"


class Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []
        self.on_sleep = None

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds
        if self.on_sleep:
            self.on_sleep()


def setup(*, policy=None, retries=2, budget=10, interval=0, secrets=(), **kwargs):
    clock = Clock()
    config = ModelConfig("callback", None, "model", max_retries=retries,
                         min_call_interval_s=interval, readiness=policy or ReadinessPolicy())
    control = RequestControl(config, CallBudget(budget), clock=clock, sleep=clock.sleep, secrets=secrets)
    readiness = ReadinessControl(control, utcnow=lambda: datetime.fromisoformat(STAMP), **kwargs)
    return clock, control, readiness


def scripted(clock, outcomes, starts):
    def exchange(context):
        starts.append((clock(), context))
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome
    return exchange


@pytest.mark.parametrize("hint", ["17", "Sun, 01 Nov 2026 12:00:17 GMT"])
def test_server_delay_is_not_shortened_by_backoff_cap(hint):
    clock, control, readiness = setup(policy=ReadinessPolicy(backoff_max_s=2, max_wait_s=40))
    starts = []
    operation = readiness.operation(stage="probe")
    result = operation.run(scripted(clock, [RequestError("busy", status_code=429,
                           retry_after=RetryAfterHint(hint, STAMP)), ModelResponse("ok")], starts))
    assert result.ready
    assert [start for start, _ in starts] == [0, 17]
    assert [row.purpose for row in control.budget.events] == ["probe", "retry"]
    assert any(row.delay_s == 17 and row.delay_source == "retry_after" for row in result.observations)
    assert result.response.readiness_trace == result.observations


def test_excessive_server_delay_defers_and_survives_for_other_callers():
    clock, control, readiness = setup(policy=ReadinessPolicy(max_wait_s=5))
    result = readiness.operation(stage="target").run(lambda ctx: (_ for _ in ()).throw(
        RequestError("busy", status_code=503, retry_after=RetryAfterHint("20", STAMP))))
    assert result.observations[-1].state == "deferred"
    assert control.budget.used == 1 and clock.sleeps == []
    second = readiness.operation(stage="target").run(lambda ctx: pytest.fail("cooldown bypassed"))
    assert second.observations[-1].state == "deferred"
    assert control.budget.used == 1


@pytest.mark.parametrize("hint", ["-1", "1.5", "NaN", "garbage"])
def test_invalid_delay_has_visible_evidence_and_bounded_jitter_backoff(hint):
    clock, _, readiness = setup(policy=ReadinessPolicy(backoff_initial_s=2, backoff_max_s=3,
                                                     jitter_fraction=1), random=lambda: 1.0)
    starts = []
    result = readiness.operation(stage="target").run(scripted(clock, [
        RequestError("busy", status_code=503, retry_after=RetryAfterHint(hint, STAMP)),
        RequestError("busy", status_code=429), ModelResponse("ok")], starts))
    assert result.ready
    assert [time for time, _ in starts] == [0, 3, 6]
    assert any(row.reason == "retry_after_invalid" for row in result.observations)


@pytest.mark.parametrize("failure", [TimeoutError("timeout"), ConnectionError("lost"),
                                    RequestError("gateway", status_code=504),
                                    RequestError("detached", retryable=True)])
def test_ambiguous_outcome_is_not_replayed_by_default(failure):
    clock, control, readiness = setup()
    result = readiness.operation(stage="target").run(scripted(clock, [failure], []))
    assert result.observations[-1].state == "unknown_outcome"
    assert result.error is not None and control.budget.used == 1
    assert result.observations[-1].credential_eligibility == "unverified"


def test_explicit_ambiguous_replay_is_counted_and_preserves_unknown_transition():
    clock, control, readiness = setup(policy=ReadinessPolicy(allow_ambiguous_replay=True))
    result = readiness.operation(stage="target").run(scripted(clock, [TimeoutError(), ModelResponse("ok")], []))
    assert result.ready and control.budget.used == 2
    assert any(row.state == "unknown_outcome" for row in result.observations)


@pytest.mark.parametrize("status,code,reason", [(401, None, "authentication_rejected"),
    (403, None, "access_rejected"), (400, None, "parameter_rejected"),
    (429, "insufficient_quota", "quota_rejected")])
def test_permanent_failure_stops_even_if_custom_predicate_accepts(status, code, reason):
    _, control, readiness = setup()
    control._retry_if = lambda error: True
    result = readiness.operation(stage="target").run(lambda ctx: (_ for _ in ()).throw(
        RequestError("rejected", status_code=status, code=code, retryable=True)))
    assert result.observations[-1].state == "rejected"
    assert result.observations[-1].reason == reason and control.budget.used == 1


def test_accepted_pending_is_never_reposted_and_retains_delay():
    _, control, readiness = setup()
    operation = readiness.operation(stage="target")
    receipt = PendingReceipt(operation.operation_id, "model", REQUEST_ID, STAMP,
                             retry_after=RetryAfterHint("7", STAMP))
    result = operation.run(lambda ctx: receipt)
    assert not result.ready and result.pending == receipt
    assert [row.state for row in result.observations][-2:] == ["pending", "deferred"]
    assert result.observations[-1].reason == "pending_route_unavailable"
    assert control.budget.used == 1
    assert PendingReceipt.from_dict(receipt.to_dict()) == receipt


def test_cancellation_during_wait_stops_future_admission():
    clock, control, readiness = setup()
    token = CancellationToken()
    clock.on_sleep = token.cancel
    result = readiness.operation(stage="target", cancellation=token).run(lambda ctx: (_ for _ in ()).throw(
        RequestError("busy", status_code=503)))
    assert result.observations[-1].state == "cancelled" and control.budget.used == 1


def test_pacing_rechecks_cancellation_before_charging_budget():
    clock, control, readiness = setup(interval=3)
    readiness.operation(stage="target").run(lambda ctx: ModelResponse("first"))
    token = CancellationToken()
    clock.on_sleep = token.cancel
    result = readiness.operation(stage="target", cancellation=token).run(lambda ctx: pytest.fail("cancel bypassed"))
    assert result.observations[-1].state == "cancelled" and control.budget.used == 1


def test_pacing_cannot_admit_after_wait_deadline():
    _, control, readiness = setup(interval=5, policy=ReadinessPolicy(max_wait_s=2))
    readiness.operation(stage="target").run(lambda ctx: ModelResponse("first"))
    result = readiness.operation(stage="target").run(lambda ctx: pytest.fail("deadline bypassed"))
    assert result.observations[-1].state == "deferred" and control.budget.used == 1


def test_snapshot_elapsed_is_live_only_while_running_without_fabricated_progress():
    clock, _, readiness = setup()
    notified = []
    operation = readiness.operation(stage="target", on_observation=notified.append)
    def exchange(context):
        clock.now += 3
        snapshot = operation.readiness_snapshot()
        assert snapshot.elapsed_s == 3 and snapshot.state == "waiting"
        assert snapshot.answer_characters == snapshot.reasoning_characters == 0
        assert len(notified) == 1
        with pytest.raises(FrozenInstanceError):
            snapshot.elapsed_s = 0
        assert context.connect_timeout_s <= context.remaining_s
        assert context.socket_idle_timeout_s <= context.remaining_s
        return ModelResponse("ok")
    result = operation.run(exchange)
    clock.now += 20
    assert operation.readiness_snapshot().elapsed_s == result.observations[-1].elapsed_s == 3


def test_callback_failure_is_static_sanitized_and_stops_before_any_call():
    _, control, readiness = setup(secrets=("sensitive",))
    def callback(observation):
        raise RuntimeError("sensitive callback data")
    result = readiness.operation(stage="target", on_observation=callback).run(lambda ctx: pytest.fail("callback failed"))
    assert result.observations[-1].reason == "callback_failed" and control.budget.used == 0
    assert "sensitive" not in str(result.error)


@pytest.mark.parametrize("budget,retries,reason", [(0, 2, "request_budget"), (10, 0, "retry_limit")])
def test_existing_caps_are_the_only_physical_and_retry_caps(budget, retries, reason):
    _, control, readiness = setup(budget=budget, retries=retries)
    result = readiness.operation(stage="target").run(lambda ctx: (_ for _ in ()).throw(RequestError("busy", status_code=503)))
    assert result.observations[-1].state == "exhausted"
    assert result.observations[-1].reason == reason and control.budget.used == min(budget, 1)


def test_complete_text_does_not_establish_required_token_capability():
    _, _, readiness = setup()
    result = readiness.operation(stage="probe", features=FeatureChoices(mia_scoring=True)).run(lambda ctx: ModelResponse("text"))
    assert not result.ready and result.response.content == "text"
    assert result.observations[-1].reason == "missing_token_evidence"


def test_concurrent_network_operations_overlap_without_an_admission_lock():
    _, control, readiness = setup()
    barrier = threading.Barrier(2)
    def invoke(_):
        def exchange(context):
            barrier.wait(timeout=5)
            return ModelResponse("ok")
        return readiness.operation(stage="target").run(exchange)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(invoke, range(2)))
    assert all(result.ready for result in results) and control.budget.used == 2


def test_queued_caller_rechecks_cooldown_published_by_terminal_failure():
    clock, control, readiness = setup(interval=1, retries=0, policy=ReadinessPolicy(max_wait_s=50))
    first_started, waiter_asleep, first_finished = (threading.Event() for _ in range(3))
    def first():
        def exchange(context):
            first_started.set()
            assert waiter_asleep.wait(5)
            raise RequestError("busy", status_code=503, retry_after=RetryAfterHint("20", STAMP))
        result = readiness.operation(stage="target").run(exchange)
        first_finished.set()
        return result
    def sleeper():
        waiter_asleep.set()
        assert first_finished.wait(5)
    clock.on_sleep = sleeper
    starts = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        failed = pool.submit(first)
        assert first_started.wait(5)
        success = pool.submit(lambda: readiness.operation(stage="target").run(
            scripted(clock, [ModelResponse("ok")], starts)))
        assert failed.result(timeout=5).observations[-1].reason == "retry_limit"
        assert success.result(timeout=5).ready
    assert starts[0][0] >= 20 and control.budget.used == 2


def test_cooldown_is_shared_across_wrappers_but_not_other_clients():
    clock, control, readiness = setup(retries=0, policy=ReadinessPolicy(max_wait_s=2))
    readiness.operation(stage="target").run(lambda ctx: (_ for _ in ()).throw(
        RequestError("busy", status_code=429, retry_after=RetryAfterHint("10", STAMP))))
    other_wrapper = ReadinessControl(control)
    blocked = other_wrapper.operation(stage="target").run(lambda ctx: pytest.fail("wrapper bypassed cooldown"))
    assert blocked.observations[-1].state == "deferred"
    other_client = RequestControl(control.config, control.budget, clock=clock, sleep=clock.sleep)
    assert ReadinessControl(other_client).operation(stage="target").run(lambda ctx: ModelResponse("ok")).ready
    assert control.budget.used == 2


def test_actual_overslept_pacing_does_not_compress_starts():
    clock, control, readiness = setup(interval=1)
    starts = []
    readiness.operation(stage="target").run(scripted(clock, [ModelResponse("ok")], starts))
    def late():
        clock.now += 5
        clock.on_sleep = None
    clock.on_sleep = late
    for _ in range(2):
        readiness.operation(stage="target").run(scripted(clock, [ModelResponse("ok")], starts))
    assert starts[1][0] >= 5 and starts[2][0] - starts[1][0] >= 1
    assert [row.started_at for row in control.budget.events] == [start for start, _ in starts]


def test_callback_failure_after_provider_evidence_preserves_failure_and_stops_retry():
    _, control, readiness = setup()
    def observe(row):
        if row.state == "temporarily_unavailable":
            raise RuntimeError("raw callback detail")
    result = readiness.operation(stage="target", on_observation=observe).run(lambda ctx: (_ for _ in ()).throw(
        RequestError("busy", status_code=503)))
    assert result.observations[-1].reason == "callback_failed" and control.budget.used == 1
    assert control.budget.events[0].status_code == 503
    assert "raw callback detail" not in str(result.error)


def test_cancellation_while_io_is_blocked_retains_response_without_claiming_abort():
    clock, control, readiness = setup()
    token = CancellationToken()
    def exchange(context):
        clock.now += 5
        token.cancel()
        return ModelResponse("completed remotely")
    result = readiness.operation(stage="target", cancellation=token).run(exchange)
    assert result.response.content == "completed remotely" and not result.ready
    assert result.observations[-1].state == "cancelled" and control.budget.used == 1


def test_terminal_operation_cannot_be_reused_for_more_admission():
    _, control, readiness = setup()
    operation = readiness.operation(stage="target")
    operation.run(lambda ctx: ModelResponse("ok"))
    with pytest.raises(RuntimeError, match="single-use"):
        operation.run(lambda ctx: pytest.fail("reused operation"))
    assert control.budget.used == 1


def test_capability_error_retains_complete_evidence_without_replay():
    _, control, readiness = setup()
    result = readiness.operation(stage="probe", features=FeatureChoices(mia_scoring=True)).run(
        lambda ctx: (_ for _ in ()).throw(RequestError("tokens missing", response_evidence=ModelResponse("text"))))
    assert result.response.content == "text"
    assert result.observations[-1].reason == "missing_token_evidence" and control.budget.used == 1


def test_huge_valid_integer_hint_defers_instead_of_falling_back_to_short_retry():
    clock, control, readiness = setup()
    result = readiness.operation(stage="target").run(lambda ctx: (_ for _ in ()).throw(
        RequestError("busy", status_code=429, retry_after=RetryAfterHint("9" * 1000, STAMP))))
    assert result.observations[-1].state == "deferred"
    assert control.budget.used == 1 and clock.sleeps == []
    assert result.observations[-1].reason == "retry_after_too_large"
    assert all(row.delay_s is None for row in result.observations)
    assert result.error.retry_after.value == "9" * 1000


@pytest.mark.parametrize("value,expected", [
    ("Sunday, 01-Nov-75 12:00:17 GMT", (datetime(2075, 11, 1, 12, 0, 17) - datetime(2026, 11, 1, 12)).total_seconds()),
    ("Sunday, 01-Nov-77 12:00:17 GMT", 0),
    ("Sun Nov  1 12:00:17 2026", 17),
    ("Sun, 01 Nov 2026 12:00:60 GMT", 60),
])
def test_http_date_forms_follow_receipt_time_and_preserve_leap_second(value, expected):
    assert retry_after_seconds(RetryAfterHint(value, STAMP)) == expected


def test_numeric_delay_rounding_never_shortens_a_valid_integer():
    value = 2**53 + 1
    assert retry_after_seconds(RetryAfterHint(str(value), STAMP)) >= value


def test_timeouts_are_bounded_by_allowance_remaining_after_queued_pacing():
    clock, _, readiness = setup(interval=2, policy=ReadinessPolicy(max_wait_s=5))
    readiness.operation(stage="target").run(lambda ctx: ModelResponse("first"))
    starts = []
    readiness.operation(stage="target").run(scripted(clock, [ModelResponse("ok")], starts))
    assert starts[0][0] == 2
    assert starts[0][1].remaining_s == starts[0][1].connect_timeout_s == starts[0][1].socket_idle_timeout_s == 3


def test_pending_secrets_are_sanitized_without_inventing_a_replacement_identity():
    _, control, readiness = setup(secrets=(REQUEST_ID, "secret-value"))
    operation = readiness.operation(stage="target")
    result = operation.run(lambda ctx: PendingReceipt(operation.operation_id, "model", REQUEST_ID, STAMP,
                                                     provider_status="secret-value"))
    assert result.pending is None and result.observations[-1].reason == "observation_unavailable"
    assert REQUEST_ID not in str(result) and "secret-value" not in str(result)
    assert control.budget.used == 1


def test_partial_data_and_unknown_usage_survive_a_disconnect():
    from recall_guard.core.readiness_evidence import PartialResponse
    _, control, readiness = setup()
    partial = PartialResponse("unfinished", reasoning_characters=20)
    result = readiness.operation(stage="target").run(lambda ctx: (_ for _ in ()).throw(
        RequestError("lost stream", response_evidence=partial)))
    assert result.response == partial and result.response.usable_answer is False
    assert result.observations[-1].state == "unknown_outcome"
    assert control.budget.events[0].usage is None and control.budget.used == 1


def test_keyboard_interrupt_ends_snapshot_without_claiming_remote_completion():
    clock, control, readiness = setup()
    operation = readiness.operation(stage="target")
    with pytest.raises(KeyboardInterrupt):
        operation.run(lambda ctx: (_ for _ in ()).throw(KeyboardInterrupt()))
    snapshot = operation.readiness_snapshot()
    assert snapshot.state == "cancelled" and snapshot.reason == "caller_cancelled"
    clock.now += 10
    assert operation.readiness_snapshot().elapsed_s == snapshot.elapsed_s
    assert control.budget.used == 1 and control.budget.events[0].status == "started"


@pytest.mark.parametrize("failure,reason", [
    ("missing_tokens", "missing_token_evidence"),
    ("mismatched_settings", "malformed_response"),
    ("empty_answer", "malformed_response"),
    ("cancelled", "caller_cancelled"),
    ("callback_failed", "callback_failed"),
    ("deadline", "waiting_allowance"),
])
def test_locally_failed_complete_evidence_is_sanitized_without_losing_nonsecret_data(failure, reason):
    secret = "synthetic-known-secret"
    clock = Clock()
    parameters = {"metadata": {"nested": [{secret: "value " + secret}, "safe value"]}}
    config = ModelConfig("callback", None, "model", parameters=parameters, readiness=ReadinessPolicy())
    control = RequestControl(config, CallBudget(2), secrets=(secret,), clock=clock, sleep=clock.sleep)
    readiness = ReadinessControl(control)
    cancellation = CancellationToken()
    response = ModelResponse(
        " " if failure == "empty_answer" else "useful answer " + secret,
        token_evidence=(TokenEvidence("token " + secret, -0.1,
                                      (TokenAlternative("alternative " + secret, -9999.0),)),),
        usage={secret: 3, "output_tokens": 5}, finish_reason="stop " + secret,
        requested_settings=parameters if failure != "mismatched_settings" else {"metadata": secret},
        confirmed_settings={"nested": [{"value": secret}, "confirmed"]}, request_id=secret,
    )
    def observe(row):
        if failure == "callback_failed" and row.state == "ready":
            raise RuntimeError(secret)
    def exchange(context):
        if failure == "cancelled":
            cancellation.cancel()
        elif failure == "deadline":
            clock.now += config.readiness.max_wait_s + 1
        return response
    result = readiness.operation(stage="probe", features=FeatureChoices(mia_scoring=failure == "missing_tokens"),
                                 cancellation=cancellation, on_observation=observe).run(exchange)
    assert not result.ready and result.observations[-1].reason == reason
    assert secret not in str(result.response.to_dict())
    assert result.response.content == response.content.replace(secret, "[REDACTED]")
    assert result.response.request_id == "[REDACTED]"
    assert result.response.finish_reason == "stop [REDACTED]"
    assert dict(result.response.usage) == {"[REDACTED]": 3, "output_tokens": 5}
    assert result.response.token_evidence == (
        TokenEvidence("token [REDACTED]", -0.1, (TokenAlternative("alternative [REDACTED]", -9999.0),)),)
    assert result.response.to_dict()["confirmed_settings"] == {"nested": [{"value": "[REDACTED]"}, "confirmed"]}
    if failure != "mismatched_settings":
        assert result.response.to_dict()["requested_settings"] == {
            "metadata": {"nested": [{"[REDACTED]": "value [REDACTED]"}, "safe value"]}}
    assert secret not in str(result.observations) and secret not in str(control.budget.events)
    if result.error is not None:
        assert result.error.__cause__ is result.error.__context__ is None
        assert secret not in str(result.error)
        if result.error.response_evidence is not None:
            assert secret not in str(result.error.response_evidence.to_dict())
            assert result.error.response_evidence.content == result.response.content
            assert result.error.response_evidence.token_evidence == result.response.token_evidence
    assert control.budget.used == 1


def test_successful_answers_keep_existing_content_semantics_with_registered_secrets():
    secret = "synthetic-known-secret"
    _, control, readiness = setup(secrets=(secret,))
    response = ModelResponse("requested text " + secret, request_id=secret)
    result = readiness.operation(stage="target").run(lambda context: response)
    assert result.ready and result.error is None
    assert result.response.content == response.content and result.response.request_id == secret
    legacy_response = control.run(lambda: response, stage="target")
    assert legacy_response is response
