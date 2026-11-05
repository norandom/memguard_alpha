"""Stdlib-only policy, observation and cooperative cancellation contracts.

No policy admits a request or infers provider support. Durations use seconds;
byte limits use UTF-8 bytes. Observations have one row per operation event,
whereas physical call IDs refer to the existing shared request ledger.
"""

import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from threading import Event
from urllib.parse import urlsplit
from uuid import UUID


def _fields(data, allowed, *, required=()):
    """Copy an explicit persisted-field whitelist; never inspect live objects."""
    if not isinstance(data, Mapping) or data.keys() - set(allowed) or set(required) - data.keys():
        raise ValueError("Invalid persisted contract fields")
    return dict(data)


def _text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip() or any(ord(char) < 32 for char in value):
        raise ValueError(f"{name} must contain nonempty text without control characters")


def _number(value: float, name: str, *, positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    try:
        valid = math.isfinite(value) and (value > 0 if positive else value >= 0)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")


def _integer(value: int, name: str, *, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


def _timestamp(value: str) -> None:
    _text(value, "timestamp")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        raise ValueError("timestamp must be an ISO-8601 UTC timestamp") from None
    if parsed.utcoffset() != timedelta(0):
        raise ValueError("timestamp must include the UTC offset")


def _uuid(value: str) -> str:
    try:
        if not isinstance(value, str):
            raise ValueError
        parsed = UUID(value)
        if str(parsed) != value.lower():
            raise ValueError
    except (ValueError, AttributeError):
        raise ValueError("request_id must be a canonical hyphenated UUID") from None
    return str(parsed)


def _origin(value: str) -> tuple[str, str, int]:
    try:
        _text(value, "endpoint/status URL")
        parsed = urlsplit(value)
        valid = (parsed.scheme in {"http", "https"} and parsed.hostname
                 and parsed.username is None and parsed.password is None
                 and not parsed.query and not parsed.fragment
                 and not any(char.isspace() for char in value) and "\\" not in value)
        if valid:
            port = parsed.port
            if port is None:
                port = 443 if parsed.scheme == "https" else 80
            return parsed.scheme, parsed.hostname.lower(), port
    except ValueError:
        pass
    raise ValueError("endpoint/status URL must be credential-free HTTP(S) without query or fragment")


@dataclass(frozen=True)
class PendingStatusPolicy:
    """Caller-declared verified route bound to one exact model and endpoint.

    ``verified`` is an explicit assertion, not a discovery mechanism. Adapters
    must additionally recognize ``profile`` before using it. The initial NIM
    profile has no verified polling route and must not construct this policy.
    Only the UUID placeholder in the path may vary; response locations are
    never accepted as routing input. Redirect handling belongs to transport.
    """

    profile: str
    model: str
    endpoint: str
    status_url_template: str
    request_id_header: str
    verified: bool

    def __post_init__(self) -> None:
        for name in ("profile", "model", "request_id_header"):
            _text(getattr(self, name), name)
        if self.verified is not True:
            raise ValueError("verified must be explicitly true for a pending status route")
        if (not re.fullmatch(r"[A-Za-z0-9-]+", self.request_id_header)
                or self.request_id_header.lower() in {"authorization", "proxy-authorization", "cookie", "x-api-key"}):
            raise ValueError("request_id_header must name a non-credential identity header")
        template = self.status_url_template
        _text(template, "status_url_template")
        if template.count("{request_id}") != 1 or "{" in template.replace("{request_id}", "") or "}" in template.replace("{request_id}", ""):
            raise ValueError("status_url_template must contain exactly one request_id placeholder")
        if "{request_id}" not in urlsplit(template).path:
            raise ValueError("status_url_template placeholder must be in the path")
        if _origin(self.endpoint) != _origin(template):
            raise ValueError("status_url_template must share the selected endpoint origin")

    def status_url(self, request_id: str, *, model: str, endpoint: str) -> str:
        """Construct only this verified route for the bound configuration."""
        if (model, endpoint) != (self.model, self.endpoint):
            raise ValueError("Pending route does not match the selected configuration")
        return self.status_url_template.replace("{request_id}", _uuid(request_id))

    def to_dict(self) -> dict:
        return {"profile": self.profile, "model": self.model, "endpoint": self.endpoint,
                "status_url_template": self.status_url_template, "request_id_header": self.request_id_header,
                "verified": self.verified}

    @classmethod
    def from_dict(cls, data):
        return cls(**_fields(data, {"profile", "model", "endpoint", "status_url_template", "request_id_header", "verified"}))


@dataclass(frozen=True)
class ReadinessPolicy:
    """Opt-in finite transport/wait limits; existing retry and budget caps win.

    ``max_wait_s`` bounds controlled waiting and subsequent admission, not an
    exact interrupt of blocked OS I/O. Connect/read bounds are capped at runtime
    by the remaining allowance. Jitter is a fraction in [0, 1] applied to local
    backoff only; provider delays must never be shortened. Polls default to zero.
    Response/event limits bound transport bytes; one event must fit a response.
    """

    connect_timeout_s: float = 10.0
    socket_idle_timeout_s: float = 30.0
    max_wait_s: float = 300.0
    max_poll_requests: int = 0
    backoff_initial_s: float = 1.0
    backoff_max_s: float = 30.0
    jitter_fraction: float = 0.0
    allow_ambiguous_replay: bool = False
    pending_status: PendingStatusPolicy | None = None
    max_event_bytes: int = 1024 * 1024
    max_response_bytes: int = 8 * 1024 * 1024

    def __post_init__(self) -> None:
        for name in ("connect_timeout_s", "socket_idle_timeout_s", "max_wait_s"):
            _number(getattr(self, name), name, positive=True)
        for name in ("backoff_initial_s", "backoff_max_s", "jitter_fraction"):
            _number(getattr(self, name), name)
        _integer(self.max_poll_requests, "max_poll_requests")
        for name in ("max_event_bytes", "max_response_bytes"):
            _integer(getattr(self, name), name, minimum=1)
        if not isinstance(self.allow_ambiguous_replay, bool):
            raise TypeError("allow_ambiguous_replay must be boolean")
        if self.pending_status is not None and not isinstance(self.pending_status, PendingStatusPolicy):
            raise TypeError("pending_status must be PendingStatusPolicy or None")
        if self.backoff_initial_s > self.backoff_max_s:
            raise ValueError("backoff_initial_s must not exceed backoff_max_s")
        if self.jitter_fraction > 1:
            raise ValueError("jitter_fraction must be at most one")
        if self.max_event_bytes > self.max_response_bytes:
            raise ValueError("max_event_bytes must not exceed max_response_bytes")
        if self.max_poll_requests and self.pending_status is None:
            raise ValueError("max_poll_requests requires a verified pending_status policy")

    def to_dict(self) -> dict:
        return {"connect_timeout_s": self.connect_timeout_s, "socket_idle_timeout_s": self.socket_idle_timeout_s,
                "max_wait_s": self.max_wait_s, "max_poll_requests": self.max_poll_requests,
                "backoff_initial_s": self.backoff_initial_s, "backoff_max_s": self.backoff_max_s,
                "jitter_fraction": self.jitter_fraction, "allow_ambiguous_replay": self.allow_ambiguous_replay,
                "pending_status": None if self.pending_status is None else self.pending_status.to_dict(),
                "max_event_bytes": self.max_event_bytes, "max_response_bytes": self.max_response_bytes}

    @classmethod
    def from_dict(cls, data):
        values = _fields(data, {"connect_timeout_s", "socket_idle_timeout_s", "max_wait_s", "max_poll_requests",
                               "backoff_initial_s", "backoff_max_s", "jitter_fraction", "allow_ambiguous_replay",
                               "pending_status", "max_event_bytes", "max_response_bytes"})
        if values.get("pending_status") is not None:
            values["pending_status"] = PendingStatusPolicy.from_dict(values["pending_status"])
        return cls(**values)


class ReadinessState(StrEnum):
    WAITING = "waiting"
    PENDING = "pending"
    PROGRESS = "progress"
    READY = "ready"
    TEMPORARILY_UNAVAILABLE = "temporarily_unavailable"
    REJECTED = "rejected"
    UNKNOWN_OUTCOME = "unknown_outcome"
    DEFERRED = "deferred"
    CANCELLED = "cancelled"
    EXHAUSTED = "exhausted"


class ReadinessReason(StrEnum):
    REQUEST_BUDGET = "request_budget"
    RETRY_LIMIT = "retry_limit"
    POLL_LIMIT = "poll_limit"
    WAITING_ALLOWANCE = "waiting_allowance"
    PENDING_ROUTE_UNAVAILABLE = "pending_route_unavailable"
    OBSERVATION_UNAVAILABLE = "observation_unavailable"
    CALLER_CANCELLED = "caller_cancelled"
    TIMEOUT = "timeout"
    DISCONNECT = "disconnect"
    GATEWAY_TIMEOUT = "gateway_timeout"
    AUTHENTICATION_REJECTED = "authentication_rejected"
    ACCESS_REJECTED = "access_rejected"
    PARAMETER_REJECTED = "parameter_rejected"
    QUOTA_REJECTED = "quota_rejected"
    MALFORMED_RESPONSE = "malformed_response"
    MISSING_TOKEN_EVIDENCE = "missing_token_evidence"
    CONFIGURATION_CHANGED = "configuration_changed"
    RESPONSE_SETTINGS_MISMATCH = "response_settings_mismatch"
    CALLBACK_FAILED = "callback_failed"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    RETRY_AFTER_INVALID = "retry_after_invalid"
    RETRY_AFTER_TOO_LARGE = "retry_after_too_large"


class DelaySource(StrEnum):
    RETRY_AFTER = "retry_after"
    BACKOFF = "backoff"
    COOLDOWN = "cooldown"


class CredentialEligibility(StrEnum):
    UNVERIFIED = "unverified"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


@dataclass(frozen=True)
class ReadinessObservation:
    """One immutable event; counters are cumulative within its operation.

    ``elapsed_s`` is monotonic operation age; ``waiting_s`` counts only local
    controlled waits. Answer/reasoning counters count observed semantic deltas
    and Unicode characters separately. Transport bytes/keepalives are not
    semantic progress. Producers redact identifiers before construction; this
    whitelist accepts no raw headers, arbitrary diagnostics or runtime controls.
    A ready event is a producer assertion; ProbeEvidence validates its response.
    """

    operation_id: str
    stage: str
    model: str
    timestamp: str
    elapsed_s: float
    state: ReadinessState
    call_id: int | None = None
    request_id: str | None = None
    status_code: int | None = None
    delay_s: float | None = None
    delay_source: DelaySource | None = None
    waiting_s: float = 0.0
    answer_characters: int = 0
    reasoning_characters: int = 0
    answer_deltas: int = 0
    reasoning_deltas: int = 0
    transport_bytes: int = 0
    keepalive_events: int = 0
    credential_eligibility: CredentialEligibility = CredentialEligibility.UNVERIFIED
    reason: ReadinessReason | None = None

    def __post_init__(self) -> None:
        for name in ("operation_id", "stage", "model"):
            _text(getattr(self, name), name)
        _timestamp(self.timestamp)
        for name in ("elapsed_s", "waiting_s"):
            _number(getattr(self, name), name)
        if self.waiting_s > self.elapsed_s:
            raise ValueError("waiting_s must not exceed elapsed_s")
        object.__setattr__(self, "state", ReadinessState(self.state))
        object.__setattr__(self, "credential_eligibility", CredentialEligibility(self.credential_eligibility))
        if self.reason is not None:
            object.__setattr__(self, "reason", ReadinessReason(self.reason))
        for name in ("answer_characters", "reasoning_characters", "answer_deltas", "reasoning_deltas",
                     "transport_bytes", "keepalive_events"):
            _integer(getattr(self, name), name)
        if self.call_id is not None:
            _integer(self.call_id, "call_id", minimum=1)
        if self.request_id is not None:
            _text(self.request_id, "request_id")
        if self.status_code is not None:
            _integer(self.status_code, "status_code", minimum=100)
            if self.status_code > 599:
                raise ValueError("status_code must be a valid HTTP status")
        if (self.delay_s is None) != (self.delay_source is None):
            raise ValueError("delay_s and delay_source must be supplied together")
        if self.delay_s is not None:
            _number(self.delay_s, "delay_s")
            object.__setattr__(self, "delay_source", DelaySource(self.delay_source))
        if self.state == ReadinessState.PENDING:
            _uuid(self.request_id)
        if self.state == ReadinessState.PROGRESS and not (self.answer_characters or self.reasoning_characters):
            raise ValueError("progress requires observed answer or reasoning content")
        if self.state == ReadinessState.READY and self.status_code is not None and (
            not 200 <= self.status_code < 300 or self.status_code == 202
        ):
            raise ValueError("ready requires a completed successful exchange")
        if self.credential_eligibility == CredentialEligibility.REJECTED and (
            self.state != ReadinessState.REJECTED
            or self.reason not in {ReadinessReason.AUTHENTICATION_REJECTED, ReadinessReason.ACCESS_REJECTED}
        ):
            raise ValueError("credential rejection requires explicit authentication/access evidence")

    def to_dict(self) -> dict:
        return {"operation_id": self.operation_id, "stage": self.stage, "model": self.model,
                "timestamp": self.timestamp, "elapsed_s": self.elapsed_s, "state": self.state.value,
                "call_id": self.call_id, "request_id": self.request_id, "status_code": self.status_code,
                "delay_s": self.delay_s, "delay_source": None if self.delay_source is None else self.delay_source.value,
                "waiting_s": self.waiting_s, "answer_characters": self.answer_characters,
                "reasoning_characters": self.reasoning_characters, "answer_deltas": self.answer_deltas,
                "reasoning_deltas": self.reasoning_deltas, "transport_bytes": self.transport_bytes,
                "keepalive_events": self.keepalive_events, "credential_eligibility": self.credential_eligibility.value,
                "reason": None if self.reason is None else self.reason.value}

    @classmethod
    def from_dict(cls, data):
        return cls(**_fields(data, {"operation_id", "stage", "model", "timestamp", "elapsed_s", "state", "call_id",
                                   "request_id", "status_code", "delay_s", "delay_source", "waiting_s", "answer_characters",
                                   "reasoning_characters", "answer_deltas", "reasoning_deltas", "transport_bytes",
                                   "keepalive_events", "credential_eligibility", "reason"}))


@dataclass(frozen=True)
class RetryAfterHint:
    """One bounded Retry-After value and its UTC receipt time, before parsing.

    Invalid values remain evidence for the readiness policy to ignore. Values
    exceeding 1024 characters become an invalid marker, never a valid prefix.
    The controller redacts known secrets before exposing this transport hint.
    No raw response-header mapping or delay decision is retained here.
    """

    value: str
    received_at: str

    def __post_init__(self) -> None:
        if not isinstance(self.value, str):
            raise TypeError("Retry-After value must be a string")
        _timestamp(self.received_at)
        if len(self.value) > 1024:
            object.__setattr__(self, "value", "[oversized Retry-After ignored]")


@dataclass(frozen=True)
class PendingReceipt:
    """Accepted HTTP 202 work, never a completed logical generation."""

    operation_id: str
    model: str
    request_id: str
    timestamp: str
    provider_status: str | None = None
    retry_after: RetryAfterHint | None = None
    status_code: int = field(default=202, init=False)
    complete: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        for name in ("operation_id", "model"):
            _text(getattr(self, name), name)
        object.__setattr__(self, "request_id", _uuid(self.request_id))
        _timestamp(self.timestamp)
        if self.retry_after is not None and not isinstance(self.retry_after, RetryAfterHint):
            raise TypeError("retry_after must be RetryAfterHint or None")
        if self.provider_status is not None:
            _text(self.provider_status, "provider_status")

    def to_dict(self) -> dict:
        result = {"operation_id": self.operation_id, "model": self.model, "request_id": self.request_id,
                  "timestamp": self.timestamp, "provider_status": self.provider_status,
                  "status_code": self.status_code, "complete": self.complete}
        if self.retry_after is not None:
            result["retry_after"] = {"value": self.retry_after.value, "received_at": self.retry_after.received_at}
        return result

    @classmethod
    def from_dict(cls, data):
        values = _fields(data, {"operation_id", "model", "request_id", "timestamp", "provider_status", "status_code", "complete", "retry_after"})
        if values.get("retry_after") is not None:
            values["retry_after"] = RetryAfterHint(**_fields(values["retry_after"], {"value", "received_at"}))
        complete, status = values.pop("complete", False), values.pop("status_code", 202)
        if complete is not False or type(status) is not int or status != 202:
            raise ValueError("Pending receipt must remain incomplete with status_code 202")
        return cls(**values)


class CancellationToken:
    """Runtime-only, thread-safe cooperative signal; never a remote abort claim.

    ``cancel`` is idempotent. ``wait`` supports bounded interruptible local
    waiting. It does not interrupt already-blocked synchronous network I/O.
    No callback, thread, or background monitoring service is started here.
    """

    def __init__(self) -> None:
        self._event = Event()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        self._event.set()

    def wait(self, timeout_s: float) -> bool:
        """Wait at most timeout_s seconds; return whether cancellation occurred."""
        _number(timeout_s, "timeout_s")
        return self._event.wait(timeout_s)


type ObservationCallback = Callable[[ReadinessObservation], None]
