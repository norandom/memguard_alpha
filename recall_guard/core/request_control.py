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
    """

    def __init__(
        self, message: str, *, status_code: int | None = None, parameter: str | None = None,
        detail: str | None = None, code: str | None = None, retryable: bool = False,
        usage: Mapping[str, int] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.parameter = parameter
        self.detail = detail
        self.code = code
        self.retryable = retryable
        self.usage = _usage_snapshot(usage)


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

    def __post_init__(self) -> None:
        object.__setattr__(self, "usage", _usage_snapshot(self.usage))

    def to_dict(self) -> dict:
        """Return detached JSON-compatible provenance without request bodies."""
        return {
            "call_id": self.call_id, "stage": self.stage, "model": self.model, "attempt": self.attempt,
            "started_at": self.started_at, "timestamp": self.timestamp, "status": self.status,
            "latency_s": self.latency_s, "usage": None if self.usage is None else dict(self.usage), "cost": self.cost,
            "status_code": self.status_code, "error": self.error,
            "parameter": self.parameter, "detail": self.detail, "code": self.code,
        }


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

    def _admit(self, *, stage: str, model: str, attempt: int, started_at: float) -> RequestEvent:
        with self._lock:
            if len(self._events) >= self._limit:
                raise BudgetExhaustedError("Physical request budget exhausted")
            event = RequestEvent(len(self._events) + 1, stage, model, attempt, started_at,
                                 datetime.now(UTC).isoformat())
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

    def _redact(self, value: str | None) -> str | None:
        if value is None:
            return None
        clean = str(value)
        for secret in self._secrets:
            clean = clean.replace(secret, "[REDACTED]")
        return clean

    def _usage(self, usage: Mapping[str, int] | None) -> Mapping[str, int] | None:
        return None if usage is None else {self._redact(name): count for name, count in usage.items()}

    def _admit(self, stage: str, attempt: int) -> RequestEvent:
        with self._admission_lock:
            if self._last_request_start is not None and self.config.min_call_interval_s > 0:
                next_start = self._last_request_start + self.config.min_call_interval_s
                remaining_s = next_start - self._clock()
                while remaining_s > 0:
                    self._sleep(remaining_s)
                    remaining_s = next_start - self._clock()
            actual_start = self._clock()
            event = self.budget._admit(stage=self._redact(stage), model=self._redact(self.config.model),
                                       attempt=attempt, started_at=actual_start)
            self._last_request_start = actual_start
            return event

    def _failure(self, error: Exception) -> RequestError:
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
        return RequestError(
            self._redact(str(error)), status_code=_status_code(error), retryable=retryable,
            parameter=self._redact(getattr(error, "parameter", None)), detail=self._redact(getattr(error, "detail", None)),
            code=self._redact(getattr(error, "code", None)), usage=self._usage(usage),
        )

    def run(self, operation: Callable[[], ModelResponse], *, stage: str) -> ModelResponse:
        """Execute a physical operation with bounded retries; retain every attempt.

        Raise sanitized failures outside the original exception handler so both
        ``__cause__`` and ``__context__`` exclude the original provider exception.
        Authentication/parameter rejections and hard quota failures never retry,
        even when a custom predicate accepts other application-specific errors.
        """
        if not isinstance(stage, str) or not stage.strip():
            raise ValueError("stage must be a nonempty string")
        if not callable(operation):
            raise TypeError("operation must be callable")
        for attempt in range(1, self.config.max_retries + 2):
            event = self._admit(stage, attempt)
            failure = None
            try:
                response = operation()
                if not isinstance(response, ModelResponse):
                    raise TypeError("operation must return ModelResponse")
            except Exception as error:
                latency_s = self._clock() - event.started_at
                failure = self._failure(error)
                self.budget._finish(
                    event, status="failed", latency_s=latency_s,
                    usage=failure.usage, error=str(failure), status_code=failure.status_code,
                    parameter=failure.parameter, detail=failure.detail, code=failure.code,
                )
            else:
                self.budget._finish(event, status="succeeded", latency_s=self._clock() - event.started_at,
                                    usage=self._usage(response.usage))
                return response
            if not failure.retryable or attempt > self.config.max_retries:
                raise failure
            self._sleep(self.config.retry_backoff_s * 2 ** (attempt - 1))
