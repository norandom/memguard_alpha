"""Shared physical-request accounting and synchronous per-client admission.

One operation must perform exactly one physical request; transport adapters must
disable hidden retries. Monotonic times and durations are in seconds. UTC event
timestamps support provenance without treating missing usage or price as zero.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from threading import Lock
from time import monotonic, sleep
from types import MappingProxyType

from recall_guard.core.completion import ModelConfig, ModelResponse
from recall_guard.core.readiness_evidence import PartialResponse
from recall_guard.core.readiness_types import PendingReceipt, RetryAfterHint, _fields, _integer, _text

_TRANSIENT_HTTP_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504, 529})
_NONTRANSIENT_CODES = frozenset({
    "insufficient_quota", "billing_hard_limit_reached", "billing_not_active",
    "credit_balance_exhausted", "insufficient_credits", "usage_limit_reached",
})


def _usage_snapshot(usage: Mapping[str, int] | None) -> Mapping[str, int] | None:
    if usage is None:
        return None
    if not isinstance(usage, Mapping) or any(
        not isinstance(name, str) or not name.strip()
        or isinstance(count, bool) or not isinstance(count, int) or count < 0
        for name, count in usage.items()
    ):
        raise ValueError("usage must contain reported nonnegative integer token counters")
    return MappingProxyType(dict(usage))


class BudgetExhaustedError(RuntimeError):
    """The shared physical-request limit has been reached."""


class RequestError(RuntimeError):
    """Provider diagnostics; controllers create a sanitized detached failure.

    Adapters should supply status/code/parameter rather than ask callers to
    classify text. Nontransient quota errors retain their own status and code.
    Usage may be reported even when a physical request fails validation.
    ``response_evidence`` retains complete missing-token evidence or incomplete stream
    evidence; neither turns this failed capability check into a usable answer.
    Retry hints carry receipt time so HTTP dates need not be interpreted later
    against a shifted clock. These runtime fields do not alter legacy events.
    """

    def __init__(
        self, message: str, *, status_code: int | None = None, parameter: str | None = None,
        detail: str | None = None, code: str | None = None, retryable: bool = False,
        usage: Mapping[str, int] | None = None,
        request_id: str | None = None, retry_after: RetryAfterHint | None = None,
        response_evidence: ModelResponse | PartialResponse | None = None,
    ) -> None:
        if retry_after is not None and not isinstance(retry_after, RetryAfterHint):
            raise TypeError("retry_after must be RetryAfterHint or None")
        if response_evidence is not None and not isinstance(response_evidence, (ModelResponse, PartialResponse)):
            raise TypeError("response_evidence must be complete/partial typed evidence or None")
        request_id = request_id if request_id is not None else getattr(response_evidence, "request_id", None)
        if request_id is not None:
            _text(request_id, "request_id")
        super().__init__(message)
        self.status_code = status_code
        self.parameter = parameter
        self.detail = detail
        self.code = code
        self.retryable = retryable
        self.usage = _usage_snapshot(response_evidence.usage if usage is None and response_evidence is not None else usage)
        self.request_id = request_id
        self.retry_after = retry_after
        self.response_evidence = response_evidence


@dataclass(frozen=True)
class RequestEvent:
    """Immutable snapshot of one physical attempt, including in-flight calls.

    ``call_id`` is one-based within its shared budget, ``attempt`` within a
    logical operation. ``started_at`` uses the controller's monotonic clock.
    ``usage`` contains only reported token counters, never estimated counters.
    No prices are supplied at this boundary, so ``cost`` remains unknown.
    """

    call_id: int
    stage: str
    model: str
    attempt: int
    started_at: float
    timestamp: str
    status: str = "started"
    latency_s: float | None = None
    usage: Mapping[str, int] | None = None
    cost: float | None = None
    status_code: int | None = None
    error: str | None = None
    parameter: str | None = None
    detail: str | None = None
    code: str | None = None
    purpose: str | None = None
    operation_id: str | None = None
    request_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "usage", _usage_snapshot(self.usage))
        if self.purpose is not None and self.purpose not in {"inference", "probe", "retry", "status"}:
            raise ValueError("purpose must be inference, probe, retry or status")
        for name in ("operation_id", "request_id"):
            if getattr(self, name) is not None:
                _text(getattr(self, name), name)

    def to_dict(self) -> dict:
        """Return detached JSON-compatible provenance without request bodies."""
        result = {
            "call_id": self.call_id, "stage": self.stage, "model": self.model, "attempt": self.attempt,
            "started_at": self.started_at, "timestamp": self.timestamp, "status": self.status,
            "latency_s": self.latency_s, "usage": None if self.usage is None else dict(self.usage), "cost": self.cost,
            "status_code": self.status_code, "error": self.error,
            "parameter": self.parameter, "detail": self.detail, "code": self.code,
        }
        for name in ("purpose", "operation_id", "request_id"):
            if getattr(self, name) is not None:
                result[name] = getattr(self, name)
        return result

    @classmethod
    def from_dict(cls, data):
        return cls(**_fields(data, {"call_id", "stage", "model", "attempt", "started_at", "timestamp", "status",
                                   "latency_s", "usage", "cost", "status_code", "error", "parameter", "detail", "code",
                                   "purpose", "operation_id", "request_id"}))


class CallBudget:
    """Atomic shared cap and attempt ledger across clients, roles and retries."""

    def __init__(self, limit: int) -> None:
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError("budget limit must be an integer")
        if limit < 0:
            raise ValueError("budget limit must be nonnegative")
        self._limit = limit
        self._events: list[RequestEvent] = []
        self._lock = Lock()

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def used(self) -> int:
        with self._lock:
            return len(self._events)

    @property
    def remaining(self) -> int:
        with self._lock:
            return self._limit - len(self._events)

    @property
    def events(self) -> tuple[RequestEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def _admit(
        self, *, stage: str, model: str, attempt: int, started_at: float,
        purpose: str | None = None, operation_id: str | None = None, request_id: str | None = None,
    ) -> RequestEvent:
        with self._lock:
            if len(self._events) >= self._limit:
                raise BudgetExhaustedError("Physical request budget exhausted")
            event = RequestEvent(len(self._events) + 1, stage, model, attempt, started_at,
                                 datetime.now(UTC).isoformat(), purpose=purpose,
                                 operation_id=operation_id, request_id=request_id)
            self._events.append(event)
            return event

    def _finish(self, event: RequestEvent, **changes) -> None:
        with self._lock:
            self._events[event.call_id - 1] = replace(event, **changes)


def _status_code(error: Exception) -> int | None:
    status = getattr(error, "status_code", None)
    if status is None:
        status = getattr(getattr(error, "response", None), "status_code", None)
    return status if isinstance(status, int) and not isinstance(status, bool) else None


def _permanent(error: Exception) -> bool:
    status = _status_code(error)
    code = getattr(error, "code", None)
    return (
        status is not None and 400 <= status < 500 and status not in _TRANSIENT_HTTP_STATUS
    ) or isinstance(code, str) and code in _NONTRANSIENT_CODES


def default_retryable(error: Exception) -> bool:
    """Retry recognized transient failures, retaining explicit rejection errors."""
    return not _permanent(error) and (
        _status_code(error) in _TRANSIENT_HTTP_STATUS
        or isinstance(error, (TimeoutError, ConnectionError))
        or isinstance(error, RequestError) and error.retryable
    )


class RequestControl:
    """Run exactly one physical request per attempt after pacing and admission.

    Share a ``CallBudget`` across every participant in a run. Share one
    controller among threads using the same client. Network operations and
    retry backoff run outside both admission locks. A callback hiding internal
    requests cannot establish strict physical-call accounting.
    """

    def __init__(
        self, config: ModelConfig, budget: CallBudget, *, secrets: Sequence[str] = (),
        clock: Callable[[], float] = monotonic, sleep: Callable[[float], None] = sleep,
        retry_if: Callable[[Exception], bool] = default_retryable,
    ) -> None:
        if not isinstance(config, ModelConfig) or not isinstance(budget, CallBudget):
            raise TypeError("config and budget must be ModelConfig and CallBudget")
        if isinstance(secrets, str) or not all(isinstance(secret, str) for secret in secrets):
            raise TypeError("secrets must be a sequence of strings")
        self.config = config
        self.budget = budget
        self._secrets = tuple(sorted({secret for secret in secrets if secret}, key=len, reverse=True))
        self._clock = clock
        self._sleep = sleep
        self._retry_if = retry_if
        self._admission_lock = Lock()
        self._last_request_start: float | None = None
        self._readiness_cooldown_until = 0.0

    def extend_readiness_cooldown(self, until: float) -> None:
        """Extend this client/model's monotonic cooldown without holding network I/O."""
        with self._admission_lock:
            self._readiness_cooldown_until = max(self._readiness_cooldown_until, until)

    def _redact(self, value: str | None) -> str | None:
        if value is None:
            return None
        clean = str(value)
        for secret in self._secrets:
            clean = clean.replace(secret, "[REDACTED]")
        return clean

    def _usage(self, usage: Mapping[str, int] | None) -> Mapping[str, int] | None:
        return None if usage is None else {self._redact(name): count for name, count in usage.items()}

    def _admit(
        self, stage: str, attempt: int, *, purpose: str | None = None,
        operation_id: str | None = None, request_id: str | None = None,
        admission_guard: Callable[[], None] | None = None,
        wait: Callable[[float], None] | None = None,
    ) -> RequestEvent:
        if admission_guard is not None:
            return self._admit_guarded(stage, attempt, purpose=purpose, operation_id=operation_id,
                                       request_id=request_id, guard=admission_guard, wait=wait or self._sleep)
        with self._admission_lock:
            if self._last_request_start is not None and self.config.min_call_interval_s > 0:
                next_start = self._last_request_start + self.config.min_call_interval_s
                remaining_s = next_start - self._clock()
                while remaining_s > 0:
                    self._sleep(remaining_s)
                    remaining_s = next_start - self._clock()
            actual_start = self._clock()
            event = self.budget._admit(stage=self._redact(stage), model=self._redact(self.config.model),
                                       attempt=attempt, started_at=actual_start, purpose=purpose,
                                       operation_id=self._redact(operation_id), request_id=self._redact(request_id))
            self._last_request_start = actual_start
            return event

    def _admit_guarded(self, stage, attempt, *, purpose, operation_id, request_id, guard, wait):
        # Recompute actual start spacing after every wake. All waits release the
        # admission lock; guard checks must be fast and must not call user code.
        while True:
            with self._admission_lock:
                guard()
                now = self._clock()
                paced_start = now if self._last_request_start is None else (
                    self._last_request_start + self.config.min_call_interval_s)
                delay_s = max(paced_start, self._readiness_cooldown_until) - now
                if self.budget.remaining == 0:
                    raise BudgetExhaustedError("Physical request budget exhausted")
                if delay_s <= 0:
                    event = self.budget._admit(
                        stage=self._redact(stage), model=self._redact(self.config.model), attempt=attempt,
                        started_at=now, purpose=purpose, operation_id=self._redact(operation_id),
                        request_id=self._redact(request_id))
                    self._last_request_start = now
                    return event
            wait(delay_s)

    def _redact_evidence(
        self, response: ModelResponse | PartialResponse | None,
    ) -> ModelResponse | PartialResponse | None:
        if response is None:
            return None

        def clean(value):
            if isinstance(value, str):
                return self._redact(value)
            if isinstance(value, dict):
                return {self._redact(key): clean(item) for key, item in value.items()}
            if isinstance(value, list):
                return [clean(item) for item in value]
            return value

        # Preserve schema keys and fixed enums. A caller's secret may contain
        # words such as "token"; replacing those in the schema corrupts evidence
        # and can raise a secondary exception inside the provider error handler.
        values = response.to_dict()
        for name in ("content", "finish_reason", "request_id"):
            values[name] = self._redact(values[name])
        for name in ("usage", "requested_settings", "confirmed_settings"):
            values[name] = clean(values[name])
        if values.get("local_identity") is not None:
            for kind in ("declared", "observed"):
                provenance = values["local_identity"][kind]
                for fields in (provenance, provenance["score_semantics"]):
                    for name, value in fields.items():
                        if isinstance(value, str) and self._redact(value) != value:
                            # Redacted identity is unknown, not a new verified
                            # identifier; preserve schema keys and enum validity.
                            fields[name] = None
        for token in values["token_evidence"] or ():
            token["token"] = self._redact(token["token"])
            for alternative in token["top_logprobs"]:
                alternative["token"] = self._redact(alternative["token"])
        trace_identity_redacted = False
        for observation in values.get("readiness_trace") or ():
            original_request_id = observation["request_id"]
            for name in ("operation_id", "stage", "model", "request_id"):
                observation[name] = self._redact(observation[name])
            if observation["state"] == "pending" and observation["request_id"] != original_request_id:
                trace_identity_redacted = True
        if trace_identity_redacted:
            # A pending trace requires a genuine UUID. Omit a trace containing
            # credential material instead of inventing an identity or leaking it
            # through a secondary validation exception. Keep answer evidence.
            values.pop("readiness_trace")
        if isinstance(response, PartialResponse):
            try:
                return PartialResponse.from_dict(values)
            except ValueError:
                # Redaction can expand short secrets beyond the original byte
                # bound. Keep the incomplete reason without leaking raw data.
                return PartialResponse("", reason=response.reason, max_retained_bytes=response.max_retained_bytes)
        return ModelResponse.from_dict(values)

    def _failure(self, error: Exception, *, preserve_transport_kind: bool = False) -> RequestError:
        retryable = False
        if not _permanent(error):
            try:
                retryable = bool(self._retry_if(error))
            except Exception:
                # A faulty policy must not leak an unsanitized secondary error.
                retryable = False
        try:
            usage = _usage_snapshot(getattr(error, "usage", None))
        except ValueError:
            # Arbitrary callback exception metadata is not verified token usage.
            usage = None
        # Arbitrary callback attributes are not trusted readiness evidence.
        typed_error = error if isinstance(error, RequestError) else None
        hint = None if typed_error is None else typed_error.retry_after
        return RequestError(
            self._redact(str(error)), status_code=_status_code(error), retryable=retryable,
            parameter=self._redact(getattr(error, "parameter", None)), detail=self._redact(getattr(error, "detail", None)),
            code=self._redact(getattr(error, "code", None) or (
                "timeout" if preserve_transport_kind and isinstance(error, TimeoutError) else
                "disconnect" if preserve_transport_kind and isinstance(error, ConnectionError) else None)), usage=self._usage(usage),
            request_id=None if typed_error is None else self._redact(typed_error.request_id),
            retry_after=None if hint is None else RetryAfterHint(self._redact(hint.value), hint.received_at),
            response_evidence=None if typed_error is None else self._redact_evidence(typed_error.response_evidence),
        )

    def sanitize_failure(self, error: Exception, *, response=None) -> RequestError:
        """Detach a locally rejected success through the owning failure sanitizer.

        No admission or event is invented. Successful responses are unchanged;
        callers use this only after an execution operation has failed/stopped.
        """
        failure = self._failure(error)
        if response is not None:
            failure.response_evidence = self._redact_evidence(response)
            if failure.usage is None:
                failure.usage = self._usage(response.usage)
        if trace := getattr(error, "readiness_trace", None):
            evidence = self._redact_evidence(ModelResponse("", readiness_trace=trace))
            failure.readiness_trace = evidence.readiness_trace or ()
        return failure

    def run_attempt(
        self, operation: Callable[[], ModelResponse | PendingReceipt], *, stage: str, attempt: int = 1,
        purpose: str | None = None, operation_id: str | None = None, request_id: str | None = None,
        admission_guard: Callable[[], None] | None = None,
        wait: Callable[[float], None] | None = None,
    ) -> ModelResponse | PendingReceipt:
        """Admit one exchange, with no hidden retry or status poll.

        Pacing and the shared physical cap apply to every invocation. A pending
        receipt succeeds physically but does not complete the logical answer.
        Caller-supplied IDs/purpose are recorded even while the call is in flight;
        a response identity is retained on completion. Retry policy decisions
        and any follow-up exchange belong to the caller of this seam.
        The optional readiness guard runs under the admission lock immediately
        before budget admission; it must only inspect local state. Its paired
        wait function runs outside locks and handles cooperative pacing/cooldown.
        Without a guard the exact legacy pacing path remains unchanged.
        """
        return self._attempt(operation, stage=stage, attempt=attempt, purpose=purpose,
                             operation_id=operation_id, request_id=request_id, allow_pending=True,
                             admission_guard=admission_guard, wait=wait)

    def _attempt(
        self, operation: Callable[[], ModelResponse | PendingReceipt], *, stage: str, attempt: int,
        purpose: str | None = None, operation_id: str | None = None, request_id: str | None = None,
        allow_pending: bool = False,
        admission_guard: Callable[[], None] | None = None,
        wait: Callable[[float], None] | None = None,
    ) -> ModelResponse | PendingReceipt:
        if not isinstance(stage, str) or not stage.strip():
            raise ValueError("stage must be a nonempty string")
        if not callable(operation):
            raise TypeError("operation must be callable")
        _integer(attempt, "attempt", minimum=1)
        if purpose is not None and purpose not in {"inference", "probe", "retry", "status"}:
            raise ValueError("purpose must be inference, probe, retry or status")
        for name, value in (("operation_id", operation_id), ("request_id", request_id)):
            if value is not None:
                _text(value, name)
        event = self._admit(stage, attempt, purpose=purpose, operation_id=operation_id, request_id=request_id,
                            admission_guard=admission_guard, wait=wait)
        failure = None
        try:
            response = operation()
            if allow_pending and isinstance(response, PendingReceipt):
                if (response.model != self.config.model
                        or operation_id is not None and response.operation_id != operation_id
                        or request_id is not None and response.request_id != request_id):
                    raise ValueError("Pending receipt identity does not match the admitted operation")
            elif not isinstance(response, ModelResponse):
                expected = "ModelResponse or PendingReceipt" if allow_pending else "ModelResponse"
                raise TypeError(f"operation must return {expected}")
            if allow_pending and response.request_id is not None:
                _text(response.request_id, "request_id")
        except Exception as error:
            latency_s = self._clock() - event.started_at
            failure = self._failure(error, preserve_transport_kind=allow_pending)
            self.budget._finish(
                event, status="failed", latency_s=latency_s,
                usage=failure.usage, error=str(failure), status_code=failure.status_code,
                parameter=failure.parameter, detail=failure.detail, code=failure.code,
                request_id=event.request_id or (failure.request_id if allow_pending else None),
            )
        else:
            pending = isinstance(response, PendingReceipt)
            self.budget._finish(
                event, status="succeeded", latency_s=self._clock() - event.started_at,
                usage=None if pending else self._usage(response.usage),
                status_code=202 if pending else None,
                request_id=event.request_id or (self._redact(response.request_id) if allow_pending else None),
                operation_id=event.operation_id or (self._redact(response.operation_id) if pending else None),
            )
            return response
        # Outside the original handler: do not retain provider exception chains.
        raise failure

    def run(self, operation: Callable[[], ModelResponse], *, stage: str,
            purpose: str | None = None, admission_guard: Callable[[], None] | None = None,
            wait: Callable[[float], None] | None = None) -> ModelResponse:
        """Execute a physical operation with bounded retries; retain every attempt.

        Raise sanitized failures outside the original exception handler so both
        ``__cause__`` and ``__context__`` exclude the original provider exception.
        Authentication/parameter rejections and hard quota failures never retry,
        even when a custom predicate accepts other application-specific errors.
        Optional runtime guard/wait controls reuse guarded admission for explicit
        legacy probes. Without them the original admission and sleep paths remain.
        """
        for attempt in range(1, self.config.max_retries + 2):
            failure = None
            try:
                return self._attempt(operation, stage=stage, attempt=attempt,
                                     purpose="retry" if purpose is not None and attempt > 1 else purpose,
                                     admission_guard=admission_guard, wait=wait)
            except RequestError as error:
                failure = error
            if not failure.retryable or attempt > self.config.max_retries:
                raise failure
            if admission_guard is not None:
                admission_guard()
            (wait or self._sleep)(self.config.retry_backoff_s * 2 ** (attempt - 1))
