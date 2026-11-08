"""Bounded cooperative readiness decisions above physical request accounting.

Create one control per configured client and one operation per logical request.
No network implementation or monitoring thread lives here. Blocked synchronous
I/O can return after the allowance: the deadline prevents later admission, not
remote inference or an exact OS-level interruption. Transport must honor the
supplied connect/read bounds and check cancellation while consuming bytes.
"""

import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from random import random
from threading import Lock
from time import sleep
from uuid import uuid4

from recall_guard.core.completion import FeatureChoices, ModelResponse, _plain, validate_features
from recall_guard.core.readiness_evidence import PartialResponse
from recall_guard.core.readiness_types import (
    CancellationToken,
    CredentialEligibility,
    DelaySource,
    ObservationCallback,
    PendingReceipt,
    ReadinessObservation,
    ReadinessReason,
    ReadinessState,
    RetryAfterHint,
    _text,
)
from recall_guard.core.request_control import (
    _NONTRANSIENT_CODES,
    BudgetExhaustedError,
    RequestControl,
    RequestError,
    _permanent,
)


def retry_after_seconds(hint: RetryAfterHint | None) -> float | None:
    """Parse receipt-relative seconds; None is invalid, OverflowError is valid but too large.

    A valid integer beyond the numeric range must force deferral, never a short
    fallback or an invented finite exact delay in the observation trace.
    """
    if hint is None:
        return None
    value = hint.value.strip()
    try:
        if re.fullmatch(r"[0-9]+", value):
            delay_s = float(value)
            if math.isfinite(delay_s) and int(delay_s) < int(value):
                delay_s = math.nextafter(delay_s, math.inf)
            if not math.isfinite(delay_s):
                raise OverflowError("Valid Retry-After exceeds the numeric range")
        else:
            weekday = r"(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)"
            month = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
            formats = (
                rf"{weekday}, [0-9]{{2}} {month} [0-9]{{4}} [0-9]{{2}}:[0-9]{{2}}:[0-9]{{2}} GMT",
                rf"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday), [0-9]{{2}}-{month}-[0-9]{{2}} [0-9]{{2}}:[0-9]{{2}}:[0-9]{{2}} GMT",
                rf"{weekday} {month} [ 0-9][0-9] [0-9]{{2}}:[0-9]{{2}}:[0-9]{{2}} [0-9]{{4}}",
            )
            if not any(re.fullmatch(pattern, value) for pattern in formats):
                return None
            leap_second = re.search(r":60(?= GMT| [0-9]{4}$)", value) is not None
            parsed = parsedate_to_datetime(re.sub(r":60(?= GMT| [0-9]{4}$)", ":59", value))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            received = datetime.fromisoformat(hint.received_at)
            if re.fullmatch(formats[1], value):
                # RFC 9110's rolling 50-year rule is receipt-relative; the
                # email parser's fixed two-digit-year pivot is not suitable.
                year = ((received.year + 50) // 100) * 100 + parsed.year % 100
                candidate = (year, parsed.month, parsed.day, parsed.hour, parsed.minute, parsed.second)
                cutoff = (received.year + 50, received.month, received.day, received.hour, received.minute, received.second)
                parsed = parsed.replace(year=year - 100 if candidate > cutoff else year)
            if leap_second:
                parsed += timedelta(seconds=1)
            delay_s = max(0.0, (parsed - received).total_seconds())
        return delay_s if math.isfinite(delay_s) else None
    except (ValueError, TypeError):
        return None


@dataclass(frozen=True)
class ExchangeContext:
    """Admission-time bounds in seconds for exactly one transport exchange."""

    operation_id: str
    remaining_s: float
    connect_timeout_s: float
    socket_idle_timeout_s: float
    cancellation: CancellationToken
    check: Callable[[], None] | None = None
    on_progress: Callable | None = None


@dataclass(frozen=True)
class ReadinessResult:
    """A terminal operation and all retained evidence, including failed attempts.

    An unsuccessful result never implies a usable response. Complete text can
    remain available when a required token check fails; partial text cannot.
    """

    response: ModelResponse | PartialResponse | None
    pending: PendingReceipt | None
    error: RequestError | None
    observations: tuple[ReadinessObservation, ...]

    @property
    def ready(self) -> bool:
        return self.observations[-1].state == ReadinessState.READY


class _Stop(Exception):
    def __init__(self, state: ReadinessState, reason: ReadinessReason):
        self.state, self.reason = state, reason


class ReadinessControl:
    """Shared client cooldown; existing RequestControl owns every physical cap.

    Multiple readiness controls over the same RequestControl share its cooldown.
    A different client has independent cooldown even when its budget is shared.
    Clock and sleep come from RequestControl, keeping all admission timestamps
    in one monotonic domain. Inject those plus randomness for offline scenarios.
    """

    def __init__(self, control: RequestControl, *, random: Callable[[], float] = random,
                 utcnow: Callable[[], datetime] = lambda: datetime.now(UTC)) -> None:
        if not isinstance(control, RequestControl) or control.config.readiness is None:
            raise ValueError("ReadinessControl requires an explicitly configured readiness policy")
        self.control = control
        self.policy = control.config.readiness
        self.random = random
        self.utcnow = utcnow

    def operation(self, *, stage: str, features: FeatureChoices | None = None,
                  cancellation: CancellationToken | None = None,
                  on_observation: ObservationCallback | None = None,
                  parameters=None, purpose: str | None = None) -> "ReadinessOperation":
        return ReadinessOperation(self, stage, features or FeatureChoices(),
                                  cancellation or CancellationToken(), on_observation, parameters, purpose)


class ReadinessOperation:
    """Single-use logical operation; snapshots are safe during blocked I/O.

    Callbacks run outside admission locks on transitions/wait decisions only.
    Snapshot inspection changes no state and emits no provider-progress events.
    Completed snapshots freeze at the actual terminal observation time.
    """

    def __init__(self, owner, stage, features, cancellation, callback, parameters=None, purpose=None):
        _text(stage, "stage")
        validate_features(owner.control.config, features)
        if not isinstance(cancellation, CancellationToken):
            raise TypeError("cancellation must be CancellationToken")
        if callback is not None and not callable(callback):
            raise TypeError("on_observation must be callable")
        self.owner, self.control, self.policy = owner, owner.control, owner.policy
        self.stage, self.features, self.cancellation = stage, features, cancellation
        self.parameters = owner.control.config.parameters if parameters is None else parameters
        self.purpose = purpose or ("probe" if stage == "probe" else "inference")
        self.operation_id = str(uuid4())
        self._callback = callback
        self._lock = Lock()
        self._started_at = None
        self._ended = False
        self._observations = []
        self._waiting_s = 0.0
        self._response = self._pending = self._error = None
        self._call_id = self._request_id = self._status_code = None
        self._progress = {}
        self._progress_before_exchange = {}

    def _elapsed(self):
        return max(0.0, self.control._clock() - self._started_at)

    def _remaining(self):
        return max(0.0, self.policy.max_wait_s - self._elapsed())

    def readiness_snapshot(self) -> ReadinessObservation | None:
        with self._lock:
            if not self._observations:
                return None
            previous = self._observations[-1]
            return previous if self._ended else replace(previous, elapsed_s=self._elapsed())

    def _emit(self, state, *, reason=None, delay_s=None, delay_source=None, notify=True):
        eligibility = CredentialEligibility.UNVERIFIED
        if state == ReadinessState.READY:
            eligibility = CredentialEligibility.ACCEPTED
        elif state == ReadinessState.REJECTED and reason in {
            ReadinessReason.AUTHENTICATION_REJECTED, ReadinessReason.ACCESS_REJECTED,
        }:
            eligibility = CredentialEligibility.REJECTED
        observation = ReadinessObservation(
            operation_id=self.operation_id, stage=self.control._redact(self.stage),
            model=self.control._redact(self.control.config.model),
            timestamp=self.owner.utcnow().astimezone(UTC).isoformat(), elapsed_s=self._elapsed(),
            state=state, call_id=self._call_id, request_id=self._request_id, status_code=self._status_code,
            waiting_s=min(self._waiting_s, self._elapsed()), reason=reason, delay_s=delay_s,
            delay_source=delay_source, credential_eligibility=eligibility, **self._progress)
        with self._lock:
            if state == ReadinessState.PROGRESS and self._observations and self._observations[-1].state == state:
                self._observations[-1] = observation
            else:
                self._observations.append(observation)
        if notify and self._callback is not None:
            try:
                self._callback(observation)
            except Exception:
                self._callback = None
                self._error = RequestError("Readiness observation callback failed", code="callback_failed")
                raise _Stop(ReadinessState.DEFERRED, ReadinessReason.CALLBACK_FAILED) from None

    def _guard(self):
        # Called while the admission lock is held: no callback, wait or network.
        if self.cancellation.cancelled:
            raise _Stop(ReadinessState.CANCELLED, ReadinessReason.CALLER_CANCELLED)
        if self._remaining() <= 0:
            raise _Stop(ReadinessState.DEFERRED, ReadinessReason.WAITING_ALLOWANCE)

    def _wait(self, delay_s, source=DelaySource.COOLDOWN, reason=None):
        self._guard()
        if delay_s is None or not math.isfinite(delay_s):
            raise _Stop(ReadinessState.DEFERRED, ReadinessReason.RETRY_AFTER_TOO_LARGE)
        if delay_s >= self._remaining():
            self._emit(ReadinessState.DEFERRED, reason=ReadinessReason.WAITING_ALLOWANCE,
                       delay_s=delay_s, delay_source=source)
            raise _Stop(ReadinessState.DEFERRED, ReadinessReason.WAITING_ALLOWANCE)
        self._emit(ReadinessState.WAITING, reason=reason, delay_s=delay_s, delay_source=source)
        until = self.control._clock() + delay_s
        while (remaining_s := until - self.control._clock()) > 0:
            self._guard()
            before = self.control._clock()
            # Native token wait is interruptible. An injected clock uses the
            # injected sleeper so fake-clock tests never sleep in real time.
            if self.control._sleep is sleep:
                self.cancellation.wait(min(remaining_s, 0.1))
            else:
                self.control._sleep(min(remaining_s, 0.1))
            self._waiting_s += max(0.0, self.control._clock() - before)
        self._guard()

    def _context(self):
        remaining_s = self._remaining()
        return ExchangeContext(self.operation_id, remaining_s,
                               min(self.policy.connect_timeout_s, remaining_s),
                               min(self.policy.socket_idle_timeout_s, remaining_s), self.cancellation,
                               self._guard, self._observe_progress)

    def _observe_progress(self, progress):
        """Coalesce immutable cumulative snapshots; bytes alone never imply progress."""
        previous = self._progress
        self._progress = {name: self._progress_before_exchange.get(name, 0) + getattr(progress, name) for name in (
            "answer_characters", "reasoning_characters", "answer_deltas", "reasoning_deltas",
            "transport_bytes", "keepalive_events")}
        semantic = any(self._progress[name] > previous.get(name, 0)
                       for name in ("answer_characters", "reasoning_characters"))
        if semantic:
            self._emit(ReadinessState.PROGRESS)

    def _exchange_record(self):
        rows = [row for row in self.control.budget.events if row.operation_id == self.operation_id]
        if rows:
            row = rows[-1]
            self._call_id, self._request_id, self._status_code = row.call_id, row.request_id, row.status_code

    def _classify(self, error):
        explicit = {
            "invalid_api_key": ReadinessReason.AUTHENTICATION_REJECTED,
            "authentication_error": ReadinessReason.AUTHENTICATION_REJECTED,
            "permission_denied": ReadinessReason.ACCESS_REJECTED,
            "invalid_parameter": ReadinessReason.PARAMETER_REJECTED,
            "invalid_request_error": ReadinessReason.PARAMETER_REJECTED,
        }
        if error.code in explicit:
            return ReadinessState.REJECTED, explicit[error.code]
        if isinstance(error.response_evidence, ModelResponse):
            return ReadinessState.REJECTED, self._validate_response(error.response_evidence) or ReadinessReason.MALFORMED_RESPONSE
        if _permanent(error):
            reason = ReadinessReason.PARAMETER_REJECTED
            if error.code in _NONTRANSIENT_CODES:
                reason = ReadinessReason.QUOTA_REJECTED
            elif error.status_code in {401, 403}:
                reason = (ReadinessReason.AUTHENTICATION_REJECTED if error.status_code == 401
                          else ReadinessReason.ACCESS_REJECTED)
            return ReadinessState.REJECTED, reason
        if isinstance(error.response_evidence, PartialResponse) or error.status_code in {None, 408, 504}:
            reason = ReadinessReason.DISCONNECT
            if error.status_code == 504:
                reason = ReadinessReason.GATEWAY_TIMEOUT
            elif error.status_code == 408 or error.code == "timeout":
                reason = ReadinessReason.TIMEOUT
            return ReadinessState.UNKNOWN_OUTCOME, reason
        if error.status_code in {425, 429, 500, 502, 503, 529}:
            return ReadinessState.TEMPORARILY_UNAVAILABLE, ReadinessReason.PROVIDER_UNAVAILABLE
        return ReadinessState.REJECTED, ReadinessReason.MALFORMED_RESPONSE

    def _delay(self, hint, attempt):
        try:
            delay_s = retry_after_seconds(hint)
        except OverflowError:
            return None, None, ReadinessReason.RETRY_AFTER_TOO_LARGE
        if delay_s is not None:
            return delay_s, DelaySource.RETRY_AFTER, None
        # Saturate before exponentiation to avoid an unbounded integer at large
        # caller retry caps. Jitter never exceeds the configured local cap.
        initial, cap = self.policy.backoff_initial_s, self.policy.backoff_max_s
        exponent = min(attempt - 1, 1023)
        delay_s = min(cap, initial * (2.0 ** exponent))
        try:
            sample = self.owner.random() if self.policy.jitter_fraction else 0.5
            if isinstance(sample, bool) or not isinstance(sample, (int, float)) or not math.isfinite(sample) or not 0 <= sample <= 1:
                raise ValueError
        except Exception:
            self._error = RequestError("Readiness backoff randomness failed", code="callback_failed")
            raise _Stop(ReadinessState.DEFERRED, ReadinessReason.CALLBACK_FAILED) from None
        delay_s = min(cap, delay_s * (1 + self.policy.jitter_fraction * (2 * sample - 1)))
        return delay_s, DelaySource.BACKOFF, ReadinessReason.RETRY_AFTER_INVALID if hint is not None else None

    def _publish_delay(self, delay_s):
        # Infinity is an internal stop marker only. Serialized observations keep
        # the original bounded hint on the error/receipt, never a fabricated delay.
        until = math.inf if delay_s is None else self.control._clock() + delay_s
        self.control.extend_readiness_cooldown(until)

    def _validate_response(self, response):
        if not response.content.strip() or json.dumps(_plain(response.requested_settings), sort_keys=True) != json.dumps(
            _plain(self.parameters), sort_keys=True
        ):
            return ReadinessReason.MALFORMED_RESPONSE
        if (self.features.mia_scoring or self.parameters.get("logprobs") is True) and response.logprobs_unavailable_reason:
            return ReadinessReason.MISSING_TOKEN_EVIDENCE
        if (self.features.mia_scoring or (self.parameters.get("top_logprobs") or 0) > 0) and response.mia_unavailable_reason:
            return ReadinessReason.MISSING_TOKEN_EVIDENCE
        return None

    def _complete(self, response):
        """Retain complete output, sanitizing it only if readiness fails locally.

        A physical success can still fail capability checks or observe a local
        stop. Apply the same schema-aware failure boundary as transport errors;
        successful model answers retain their existing content semantics.
        """
        self._response = response
        try:
            self._guard()
            reason = self._validate_response(response)
            if reason is not None:
                self._error = RequestError("Completed response failed readiness checks", response_evidence=response,
                    code="missing_token_evidence" if reason == ReadinessReason.MISSING_TOKEN_EVIDENCE else "invalid_response")
                raise _Stop(ReadinessState.REJECTED, reason)
            self._error = None
            self._emit(ReadinessState.READY)
        except _Stop:
            self._response = self.control._redact_evidence(response)
            if self._error is not None and self._error.response_evidence is response:
                self._error = self.control._failure(self._error)
            raise

    def _run(self, exchange):
        self._emit(ReadinessState.WAITING)
        attempt = 1
        while True:
            self._guard()
            failure = None
            try:
                self._progress_before_exchange = self._progress.copy()
                response = self.control.run_attempt(
                    lambda: exchange(self._context()), stage=self.stage, attempt=attempt,
                    purpose=self.purpose if attempt == 1 else "retry",
                    operation_id=self.operation_id, admission_guard=self._guard, wait=self._wait)
            except RequestError as error:
                failure = error
            self._exchange_record()
            if failure is None:
                if isinstance(response, PendingReceipt):
                    # The typed receipt is identity-checked by RequestControl.
                    if self.control._redact(response.request_id) != response.request_id:
                        self._request_id = None
                        raise _Stop(ReadinessState.DEFERRED, ReadinessReason.OBSERVATION_UNAVAILABLE)
                    hint = response.retry_after
                    hint = None if hint is None else replace(hint, value=self.control._redact(hint.value))
                    self._pending = replace(response, model=self.control._redact(response.model),
                                            provider_status=self.control._redact(response.provider_status), retry_after=hint)
                    if response.retry_after is not None:
                        delay_s, source, reason = self._delay(hint, attempt)
                        self._publish_delay(delay_s)
                        self._emit(ReadinessState.PENDING, reason=reason, delay_s=delay_s, delay_source=source)
                    else:
                        self._emit(ReadinessState.PENDING)
                    self._guard()
                    # Accepted work is never re-POSTed and no status route is polled.
                    raise _Stop(ReadinessState.DEFERRED, ReadinessReason.PENDING_ROUTE_UNAVAILABLE)
                self._complete(response)
                return
            self._error = failure
            self._response = failure.response_evidence
            self._guard()
            if failure.code == "pending_identity_unavailable":
                raise _Stop(ReadinessState.DEFERRED, ReadinessReason.PENDING_ROUTE_UNAVAILABLE)
            if failure.code == "callback_failed":
                raise _Stop(ReadinessState.DEFERRED, ReadinessReason.CALLBACK_FAILED)
            state, reason = self._classify(failure)
            delay = None
            if state != ReadinessState.REJECTED:
                delay = self._delay(failure.retry_after, attempt)
                # Publish before callbacks and terminal-cap checks. Other
                # callers must honor a server hint even if this one cannot retry.
                self._publish_delay(delay[0])
            self._emit(state, reason=(delay[2] or reason) if delay is not None and state == ReadinessState.TEMPORARILY_UNAVAILABLE else reason,
                       delay_s=None if delay is None else delay[0],
                       delay_source=None if delay is None else delay[1])
            self._guard()
            if state == ReadinessState.REJECTED:
                return
            replay = state != ReadinessState.UNKNOWN_OUTCOME or self.policy.allow_ambiguous_replay
            if not replay:
                return
            if attempt > self.control.config.max_retries:
                raise _Stop(ReadinessState.EXHAUSTED, ReadinessReason.RETRY_LIMIT)
            self._guard()
            if self.control.budget.remaining == 0:
                raise _Stop(ReadinessState.EXHAUSTED, ReadinessReason.REQUEST_BUDGET)
            self._wait(*delay)
            attempt += 1

    def run(self, exchange: Callable[[ExchangeContext], ModelResponse | PendingReceipt]) -> ReadinessResult:
        """Execute once; failed/deferred results retain a terminal trace and evidence."""
        if not callable(exchange):
            raise TypeError("exchange must be callable")
        with self._lock:
            if self._started_at is not None:
                raise RuntimeError("Readiness operations are single-use")
            self._started_at = self.control._clock()
        stopped = None
        interrupted = None
        try:
            self._run(exchange)
        except BudgetExhaustedError:
            stopped = _Stop(ReadinessState.EXHAUSTED, ReadinessReason.REQUEST_BUDGET)
        except _Stop as stop:
            stopped = stop
        except KeyboardInterrupt as error:
            # Transport owns its context cleanup. Keep the physical ledger's
            # unresolved intent and let the caller's interruption propagate.
            self._exchange_record()
            self.cancellation.cancel()
            self._callback = None
            interrupted = error
            stopped = _Stop(ReadinessState.CANCELLED, ReadinessReason.CALLER_CANCELLED)
        if stopped is not None:
            try:
                previous = self.readiness_snapshot()
                if previous is None or (previous.state, previous.reason) != (stopped.state, stopped.reason):
                    self._emit(stopped.state, reason=stopped.reason)
            except _Stop:
                self._emit(ReadinessState.DEFERRED, reason=ReadinessReason.CALLBACK_FAILED, notify=False)
        with self._lock:
            self._ended = True
            observations = tuple(self._observations)
        if interrupted is not None:
            raise interrupted
        response = self._response
        if isinstance(response, ModelResponse):
            response = replace(response, readiness_trace=observations)
        return ReadinessResult(response, self._pending, self._error, observations)
