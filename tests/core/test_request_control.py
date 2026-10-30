"""Physical request admission, partial accounting, retries and actual-start pacing."""

import json
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError

import pytest

from recall_guard.core import request_control as r
from recall_guard.core.completion import ModelConfig, ModelResponse


class Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []
        self.extra_delay = 0.0
        self.short_wake = False

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds / 2 if self.short_wake and len(self.sleeps) == 1 else seconds + self.extra_delay


def config(**overrides):
    return ModelConfig(protocol="callback", endpoint=None, model="test-model", **overrides)


def control(budget, clock=None, **overrides):
    clock = clock or Clock()
    settings = overrides.pop("config", config(max_retries=0))
    return r.RequestControl(settings, budget, clock=clock, sleep=clock.sleep, **overrides)


def test_all_roles_share_a_physical_budget_and_preserve_partial_events():
    budget = r.CallBudget(4)
    first, second = control(budget), control(budget)
    for stage in ("probe", "rewrite", "verification", "target"):
        (first if stage in {"probe", "target"} else second).run(lambda: ModelResponse("ok"), stage=stage)
    with pytest.raises(r.BudgetExhaustedError, match="budget"):
        first.run(lambda: pytest.fail("over-budget operation ran"), stage="target")
    assert budget.used == 4
    assert budget.remaining == 0
    assert [event.stage for event in budget.events] == ["probe", "rewrite", "verification", "target"]
    assert [event.call_id for event in budget.events] == [1, 2, 3, 4]
    assert all(event.status == "succeeded" for event in budget.events)


@pytest.mark.parametrize("limit", [-1, True, 1.5, None])
def test_invalid_budget_limit_is_rejected(limit):
    with pytest.raises((TypeError, ValueError)):
        r.CallBudget(limit)


def test_zero_budget_prevents_the_first_physical_attempt():
    budget = r.CallBudget(0)
    with pytest.raises(r.BudgetExhaustedError):
        control(budget).run(lambda: pytest.fail("zero budget operation ran"), stage="probe")
    assert budget.events == ()


def test_shared_budget_is_atomic_across_clients_and_threads():
    budget = r.CallBudget(7)
    controls = [control(budget) for _ in range(32)]
    barrier = threading.Barrier(len(controls))
    called = []
    calls_lock = threading.Lock()

    def invoke(client):
        barrier.wait(timeout=5)

        def operation():
            with calls_lock:
                called.append(True)
            return ModelResponse("ok")

        try:
            return client.run(operation, stage="target")
        except r.BudgetExhaustedError:
            return None

    with ThreadPoolExecutor(max_workers=len(controls)) as executor:
        results = list(executor.map(invoke, controls))
    assert sum(result is not None for result in results) == len(called) == budget.used == 7
    assert len(budget.events) == 7
    assert all(event.status == "succeeded" for event in budget.events)


def test_usage_latency_and_events_are_immutable_detached_snapshots():
    clock = Clock()
    budget = r.CallBudget(1)
    usage = {"input_tokens": 10, "output_tokens": 4}

    def operation():
        assert budget.events[0].status == "started"
        clock.now += 0.75
        return ModelResponse("ok", usage=usage)

    result = control(budget, clock).run(operation, stage="target")
    usage["input_tokens"] = 100
    event = budget.events[0]
    assert event.model == "test-model" and event.attempt == 1
    assert event.latency_s == 0.75
    assert event.started_at == 0
    assert event.timestamp.endswith("+00:00")
    assert dict(event.usage) == {"input_tokens": 10, "output_tokens": 4}
    assert result.usage["input_tokens"] == 10
    assert event.cost is None
    with pytest.raises(FrozenInstanceError):
        event.status = "changed"
    with pytest.raises(TypeError):
        event.usage["output_tokens"] = 0
    exported = event.to_dict()
    exported["usage"]["output_tokens"] = 99
    assert event.usage["output_tokens"] == 4
    assert json.loads(json.dumps(event.to_dict()))["cost"] is None


def test_unreported_usage_is_unknown_instead_of_zero():
    budget = r.CallBudget(1)
    control(budget).run(lambda: ModelResponse("text only"), stage="target")
    assert budget.events[0].usage is None
    assert budget.events[0].cost is None


def test_actual_start_spacing_is_independent_of_network_latency():
    clock = Clock()
    budget = r.CallBudget(3)
    client = control(budget, clock, config=config(max_retries=0, min_call_interval_s=2))
    starts = []

    def operation():
        starts.append(clock())
        clock.now += 0.5
        return ModelResponse("ok")

    for _ in range(3):
        client.run(operation, stage="target")
    assert starts == [0, 2, 4]
    assert clock.sleeps == [1.5, 1.5]
    assert [event.latency_s for event in budget.events] == [0.5, 0.5, 0.5]


def test_delayed_wakeups_do_not_compress_subsequent_request_starts():
    clock = Clock()
    clock.extra_delay = 3
    budget = r.CallBudget(3)
    client = control(budget, clock, config=config(max_retries=0, min_call_interval_s=2))
    starts = []
    for _ in range(3):
        client.run(lambda: (starts.append(clock()), ModelResponse("ok"))[1], stage="target")
    assert starts == [0, 5, 10]
    assert clock.sleeps == [2, 2]


def test_early_wakeups_wait_until_the_actual_start_interval():
    clock = Clock()
    clock.short_wake = True
    budget = r.CallBudget(2)
    client = control(budget, clock, config=config(max_retries=0, min_call_interval_s=2))
    for _ in range(2):
        client.run(lambda: ModelResponse("ok"), stage="target")
    assert clock.sleeps == [2, 1]
    assert [event.started_at for event in budget.events] == [0, 2]


@pytest.mark.parametrize("interval", [0, 0.01])
def test_network_operation_runs_outside_admission_lock(interval):
    budget = r.CallBudget(2)
    client = r.RequestControl(config(max_retries=0, min_call_interval_s=interval), budget)
    simultaneous_requests = threading.Barrier(2)

    def operation():
        simultaneous_requests.wait(timeout=2)
        return ModelResponse("ok")

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(client.run, operation, stage="target") for _ in range(2)]
        assert [future.result(timeout=4).content for future in futures] == ["ok", "ok"]


def test_retry_attempts_consume_budget_and_stop_before_an_unfunded_retry():
    clock = Clock()
    budget = r.CallBudget(2)
    client = control(budget, clock, config=config(max_retries=5, retry_backoff_s=1))
    called = []

    def operation():
        called.append(True)
        raise TimeoutError("upstream timeout")

    with pytest.raises(r.BudgetExhaustedError):
        client.run(operation, stage="probe")
    assert len(called) == budget.used == 2
    assert [event.attempt for event in budget.events] == [1, 2]
    assert all(event.status == "failed" and event.error == "upstream timeout" for event in budget.events)


def test_bounded_retries_have_backoff_and_each_retry_is_paced():
    clock = Clock()
    budget = r.CallBudget(4)
    client = control(budget, clock, config=config(max_retries=2, retry_backoff_s=0.5, min_call_interval_s=2))
    starts = []

    def operation():
        starts.append(clock())
        raise TimeoutError("temporary failure")

    with pytest.raises(r.RequestError, match="temporary failure") as raised:
        client.run(operation, stage="target")
    assert raised.value.retryable
    assert starts == [0, 2, 4]
    assert budget.used == 3
    assert clock.sleeps == [0.5, 1.5, 1.0, 1.0]


def test_transient_failure_can_recover_and_preserve_failed_attempt():
    clock = Clock()
    budget = r.CallBudget(3)
    client = control(budget, clock, config=config(max_retries=2, retry_backoff_s=1))
    attempts = 0

    def operation():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise r.RequestError("busy", status_code=503)
        return ModelResponse("recovered", usage={"output_tokens": 2})

    assert client.run(operation, stage="target").content == "recovered"
    assert clock.sleeps == [1]
    assert [event.status for event in budget.events] == ["failed", "succeeded"]
    assert budget.events[0].status_code == 503
    assert budget.events[0].usage is None
    assert dict(budget.events[1].usage) == {"output_tokens": 2}


@pytest.mark.parametrize("status_code", [400, 401, 402, 403, 404, 422])
def test_permanent_http_failures_are_not_retried_even_with_a_permissive_predicate(status_code):
    budget = r.CallBudget(10)
    client = control(budget, config=config(max_retries=5), retry_if=lambda _: True)

    def operation():
        raise r.RequestError("provider rejected request", status_code=status_code, parameter="temperature")

    with pytest.raises(r.RequestError) as raised:
        client.run(operation, stage="probe")
    assert raised.value.status_code == status_code
    assert raised.value.parameter == "temperature"
    assert not raised.value.retryable
    assert budget.used == 1


@pytest.mark.parametrize("code", ["insufficient_quota", "billing_hard_limit_reached", "credit_balance_exhausted"])
def test_nontransient_quota_is_distinct_from_invalid_key_and_not_retried(code):
    budget = r.CallBudget(3)

    def operation():
        raise r.RequestError("account quota exhausted", status_code=429, code=code)

    with pytest.raises(r.RequestError) as raised:
        control(budget, config=config(max_retries=2), retry_if=lambda _: True).run(operation, stage="probe")
    assert raised.value.status_code == 429
    assert raised.value.code == code
    assert budget.used == 1


def test_unknown_runtime_errors_are_not_retried_by_default():
    budget = r.CallBudget(3)

    def operation():
        raise ValueError("malformed completion")

    with pytest.raises(r.RequestError, match="malformed completion"):
        control(budget, config=config(max_retries=2)).run(operation, stage="target")
    assert budget.used == 1


def test_custom_predicate_can_retry_a_known_non_http_transient_error():
    budget = r.CallBudget(3)

    def operation():
        raise LookupError("custom transient")

    with pytest.raises(r.RequestError):
        control(budget, config=config(max_retries=1, retry_backoff_s=0),
                retry_if=lambda error: isinstance(error, LookupError)).run(operation, stage="target")
    assert budget.used == 2


def test_redaction_covers_messages_metadata_and_events_without_exception_chaining():
    secret = "synthetic-secret-value"
    budget = r.CallBudget(1)
    client = control(budget, secrets=(secret, "synthetic-secret"))

    def operation():
        raise r.RequestError(f"Authorization: Bearer {secret}", status_code=400,
                             parameter=f"parameter-{secret}", detail=f"detail-{secret}", code=f"code-{secret}",
                             usage={"input_tokens": 2})

    with pytest.raises(r.RequestError) as raised:
        client.run(operation, stage=f"probe-{secret}")
    error = raised.value
    assert error.status_code == 400
    assert error.__cause__ is None
    assert error.__context__ is None
    assert secret not in "".join(traceback.format_exception(error))
    assert "[REDACTED]" in str(error)
    assert error.parameter == "parameter-[REDACTED]"
    assert error.detail == "detail-[REDACTED]"
    assert error.code == "code-[REDACTED]"
    exported = json.dumps([event.to_dict() for event in budget.events])
    assert secret not in exported
    assert budget.events[0].stage == "probe-[REDACTED]"
    assert dict(budget.events[0].usage) == {"input_tokens": 2}
    assert budget.events[0].detail == "detail-[REDACTED]"


def test_saved_started_event_stays_immutable_after_request_finishes():
    budget = r.CallBudget(1)
    observed = []

    def operation():
        observed.append(budget.events)
        return ModelResponse("ok")

    control(budget).run(operation, stage="target")
    assert observed[0][0].status == "started"
    assert observed[0][0].latency_s is None
    assert budget.events[0].status == "succeeded"


def test_stage_validation_happens_before_budget_or_operation():
    budget = r.CallBudget(1)
    with pytest.raises((ValueError, TypeError), match="stage"):
        control(budget).run(lambda: pytest.fail("invalid stage ran"), stage="")
    assert budget.used == 0


def test_failure_latency_measures_the_operation_before_retry_policy_runs():
    clock = Clock()
    budget = r.CallBudget(1)

    def policy(_):
        clock.now += 5
        return False

    def operation():
        clock.now += 0.25
        raise TimeoutError("timeout")

    with pytest.raises(r.RequestError):
        control(budget, clock, retry_if=policy).run(operation, stage="target")
    assert budget.events[0].latency_s == 0.25


def test_unusual_callback_error_metadata_cannot_leak_the_original_exception():
    secret = "synthetic-private-credential"
    budget = r.CallBudget(1)

    def operation():
        error = RuntimeError(f"callback failed with {secret}")
        error.code = {"provider": "unstructured"}
        error.usage = {"input_tokens": "unavailable"}
        raise error

    with pytest.raises(r.RequestError) as raised:
        control(budget, secrets=(secret,)).run(operation, stage="target")
    assert raised.value.__context__ is None
    assert secret not in str(raised.value)
    assert budget.events[0].status == "failed"
    assert budget.events[0].usage is None


@pytest.mark.parametrize("status_code", [408, 429, 503, 529])
def test_default_policy_retries_recognized_transient_provider_statuses(status_code):
    budget = r.CallBudget(2)

    def operation():
        raise r.RequestError("temporary provider failure", status_code=status_code)

    with pytest.raises(r.RequestError) as raised:
        control(budget, config=config(max_retries=1, retry_backoff_s=0)).run(operation, stage="target")
    assert raised.value.status_code == status_code
    assert budget.used == 2
