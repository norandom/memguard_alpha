"""Explicit, synchronous completion contracts with optional token evidence.

This module performs local validation only. Capability declarations and schema
evidence never establish that a provider honored an unechoed generation control.
"""

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Protocol, runtime_checkable
from urllib.parse import urlsplit

from recall_guard.core.local_evidence import LocalModelIdentity
from recall_guard.core.nvidia_lm import CompletionResult, TokenLogprob
from recall_guard.core.readiness_types import ReadinessObservation, ReadinessPolicy, _fields

type ParameterValue = str | int | float | bool | None | list[ParameterValue] | tuple[ParameterValue, ...] | Mapping[str, ParameterValue]

_CREDENTIAL_FIELDS = frozenset({
    "apikey", "authorization", "accesstoken", "password", "credential", "credentials", "credentialref",
    "xapikey", "bearertoken", "secret", "clientsecret", "headers",
})
_TRANSPORT_FIELDS = frozenset({
    "model", "messages", "prompt", "endpoint", "url", "protocol", "provider", "timeout", "timeouts",
    "maxretries", "retrybackoffs", "mincallintervals", "localidentity", "recallguardlocal",
})
_CENSORED_LOGPROB = -9999.0


def _text(value: str, name: str, *, allow_empty: bool = False) -> None:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValueError(f"{name} must be a string" + ("" if allow_empty else " with nonempty content"))


def _number(value: float, name: str, *, minimum: float = 0.0, positive: bool = False) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    try:
        valid = math.isfinite(value) and (value > minimum if positive else value >= minimum)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError(f"{name} must be finite and {'greater than' if positive else 'at least'} {minimum}")


def _integer(value: int, name: str, *, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


def _freeze(value: ParameterValue) -> ParameterValue:
    """Copy JSON-shaped values into immutable containers; reject credential fields."""
    if isinstance(value, Mapping):
        copied = {}
        for key, child in value.items():
            _text(key, "parameter name")
            if key.lower().replace("_", "").replace("-", "") in _CREDENTIAL_FIELDS:
                raise ValueError("Credentials and authorization fields are not generation parameters")
            copied[key] = _freeze(child)
        return MappingProxyType(copied)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(child) for child in value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Generation parameter numbers must be finite")
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError("Generation parameters must contain JSON-compatible values")


def _parameters(values: Mapping[str, ParameterValue]) -> Mapping[str, ParameterValue]:
    if not isinstance(values, Mapping):
        raise TypeError("Generation parameters must be a mapping")
    for name, value in values.items():
        _text(name, "parameter name")
        if name.lower().replace("_", "").replace("-", "") == "stream" and (name != "stream" or not isinstance(value, bool)):
            raise ValueError("stream must be an explicit boolean generation parameter")
        if name.lower().replace("_", "").replace("-", "") in _TRANSPORT_FIELDS:
            raise ValueError("Generation parameters cannot override transport or input fields")
        if value is None:
            continue
        if name in {"max_tokens", "max_completion_tokens"}:
            _integer(value, name, minimum=1)
        elif name in {"seed", "random_seed"}:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        elif name == "top_logprobs":
            _integer(value, name)
        elif name in {"temperature", "top_p"}:
            _number(value, name)
            if name == "top_p" and value > 1:
                raise ValueError("top_p must be at most 1")
        elif name == "logprobs" and not isinstance(value, bool):
            raise TypeError("logprobs must be boolean")
    return _freeze(values)


def _plain(value):
    """Return detached JSON-compatible containers for explicit serialization."""
    if isinstance(value, Mapping):
        return {key: _plain(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_plain(child) for child in value]
    return value


class CompletionProtocol(StrEnum):
    OPENAI_CHAT = "openai_chat"
    ANTHROPIC_MESSAGES = "anthropic_messages"
    CALLBACK = "callback"


class CapabilityStatus(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNVERIFIED = "unverified"


@dataclass(frozen=True)
class CapabilityEvidence:
    """Availability evidence; ``observed`` requires an actual generation probe.

    Observing that a parameter is accepted still does not confirm its effective
    value. Only response ``confirmed_settings`` records echoed effective values.
    """

    status: CapabilityStatus
    source: str
    detail: str = ""
    observed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", CapabilityStatus(self.status))
        _text(self.source, "capability evidence source")
        _text(self.detail, "capability evidence detail", allow_empty=True)
        if not isinstance(self.observed, bool):
            raise TypeError("observed must be boolean")

    def to_dict(self) -> dict:
        return {"status": self.status.value, "source": self.source, "detail": self.detail, "observed": self.observed}


@dataclass(frozen=True)
class FeatureChoices:
    """Explicit local choices; text ensembles and rewriting do not require MIA."""

    mia_scoring: bool = False
    ensemble_draws: int = 1
    rewriting: bool = False

    def __post_init__(self) -> None:
        _integer(self.ensemble_draws, "ensemble_draws", minimum=1)
        if not isinstance(self.mia_scoring, bool) or not isinstance(self.rewriting, bool):
            raise TypeError("mia_scoring and rewriting must be boolean")


def _capabilities(values: Mapping[str, CapabilityEvidence]) -> Mapping[str, CapabilityEvidence]:
    if not isinstance(values, Mapping):
        raise TypeError("capabilities must be a mapping")
    for name, evidence in values.items():
        _text(name, "capability name")
        if not isinstance(evidence, CapabilityEvidence):
            raise TypeError("capabilities must contain CapabilityEvidence values")
    return MappingProxyType(dict(values))


@dataclass(frozen=True)
class CapabilityReport:
    """Required features only, preserving unavailable and unobserved evidence."""

    capabilities: Mapping[str, CapabilityEvidence]
    features: FeatureChoices = field(default_factory=FeatureChoices)

    def __post_init__(self) -> None:
        object.__setattr__(self, "capabilities", _capabilities(self.capabilities))
        if not isinstance(self.features, FeatureChoices):
            raise TypeError("features must be FeatureChoices")

    @property
    def unsupported(self) -> tuple[str, ...]:
        return tuple(name for name, evidence in self.capabilities.items() if evidence.status == CapabilityStatus.UNSUPPORTED)

    @property
    def requires_probe(self) -> tuple[str, ...]:
        return tuple(name for name, evidence in self.capabilities.items()
                     if evidence.status == CapabilityStatus.UNVERIFIED
                     or (evidence.status == CapabilityStatus.SUPPORTED and not evidence.observed))

    @property
    def ready_for_batch(self) -> bool:
        return not self.unsupported and not self.requires_probe

    @property
    def mia_unavailable_reason(self) -> str | None:
        if not self.features.mia_scoring:
            return "MIA/MCS scoring was not selected."
        missing = [name for name in ("logprobs", "top_logprobs")
                   if name not in self.capabilities or self.capabilities[name].status != CapabilityStatus.SUPPORTED
                   or not self.capabilities[name].observed]
        return "MIA/MCS requires verified token evidence: " + ", ".join(missing) if missing else None


@dataclass(frozen=True)
class ModelConfig:
    """Immutable caller configuration, with no resolved credentials.

    ``credential_ref`` is an environment-variable name (optionally ``env:``
    prefixed), resolved by adapters at runtime and omitted from repr/export.
    HTTP endpoints must be credential-free URLs without queries or fragments.
    Missing parameter keys remain omitted; explicit ``None`` remains explicit.
    """

    protocol: CompletionProtocol
    endpoint: str | None
    model: str
    credential_ref: str | None = field(default=None, repr=False)
    parameters: Mapping[str, ParameterValue] = field(default_factory=dict)
    capabilities: Mapping[str, CapabilityEvidence] = field(default_factory=dict)
    provider: str | None = None
    timeout_s: float = 15.0
    max_retries: int = 2
    retry_backoff_s: float = 2.0
    min_call_interval_s: float = 0.0
    readiness: ReadinessPolicy | None = None
    local_identity: LocalModelIdentity | None = None

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "protocol", CompletionProtocol(self.protocol))
        except ValueError:
            raise ValueError("protocol must be openai_chat, anthropic_messages or callback") from None
        _text(self.model, "model")
        if self.local_identity is not None and not isinstance(self.local_identity, LocalModelIdentity):
            raise TypeError("local_identity must be LocalModelIdentity or None")
        if self.provider is not None:
            _text(self.provider, "provider")
        if self.endpoint is not None or self.protocol != CompletionProtocol.CALLBACK:
            self._validate_endpoint()
        if self.credential_ref is not None and (
            not isinstance(self.credential_ref, str)
            or re.fullmatch(r"(?:env:)?[A-Za-z_][A-Za-z0-9_]*", self.credential_ref) is None
        ):
            raise ValueError("credential_ref must name an environment variable, not contain a credential")
        _number(self.timeout_s, "timeout_s", positive=True)
        _integer(self.max_retries, "max_retries")
        _number(self.retry_backoff_s, "retry_backoff_s")
        _number(self.min_call_interval_s, "min_call_interval_s")
        if self.readiness is not None and not isinstance(self.readiness, ReadinessPolicy):
            raise TypeError("readiness must be ReadinessPolicy or None")
        if "stream" in self.parameters and (self.readiness is None or self.protocol != CompletionProtocol.OPENAI_CHAT):
            raise ValueError("stream requires opt-in OpenAI-chat readiness")
        object.__setattr__(self, "parameters", _parameters(self.parameters))
        object.__setattr__(self, "capabilities", _capabilities(self.capabilities))

    def _validate_endpoint(self) -> None:
        try:
            parsed = urlsplit(self.endpoint) if isinstance(self.endpoint, str) else None
            valid = (parsed is not None and parsed.scheme in {"http", "https"} and parsed.hostname
                     and parsed.username is None and parsed.password is None and not parsed.query and not parsed.fragment)
            if valid:
                _ = parsed.port  # Validate a supplied port without echoing malformed input.
        except ValueError:
            valid = False
        if not valid:
            raise ValueError("endpoint must be a credential-free HTTP(S) URL without query or fragment")

    def inspect(self, features: FeatureChoices | None = None) -> CapabilityReport:
        """Inspect declarations without issuing calls or promoting unknown values."""
        features = features if features is not None else FeatureChoices()
        if not isinstance(features, FeatureChoices):
            raise TypeError("features must be FeatureChoices")
        required = {"text_generation", *self.parameters}
        if features.mia_scoring:
            required.update({"logprobs", "top_logprobs"})
        unknown = CapabilityEvidence(CapabilityStatus.UNVERIFIED, "No capability evidence supplied")
        return CapabilityReport({name: self.capabilities.get(name, unknown) for name in sorted(required)}, features)

    def to_dict(self) -> dict:
        """Export provenance without resolving or serializing credentials."""
        result = {
            "protocol": self.protocol.value, "endpoint": self.endpoint, "model": self.model, "provider": self.provider,
            "parameters": _plain(self.parameters),
            "capabilities": {name: evidence.to_dict() for name, evidence in self.capabilities.items()},
            "timeout_s": self.timeout_s, "max_retries": self.max_retries,
            "retry_backoff_s": self.retry_backoff_s, "min_call_interval_s": self.min_call_interval_s,
        }
        if self.readiness is not None:
            result["readiness"] = self.readiness.to_dict()
        if self.local_identity is not None:
            result["local_identity"] = self.local_identity.to_dict()
        return result

    @classmethod
    def from_dict(cls, data):
        values = _fields(data, {"protocol", "endpoint", "model", "provider", "parameters", "capabilities", "timeout_s",
                               "max_retries", "retry_backoff_s", "min_call_interval_s", "readiness", "local_identity"})
        values["capabilities"] = {name: CapabilityEvidence(**_fields(entry, {"status", "source", "detail", "observed"}))
                                  for name, entry in values.get("capabilities", {}).items()}
        if values.get("readiness") is not None:
            values["readiness"] = ReadinessPolicy.from_dict(values["readiness"])
        if values.get("local_identity") is not None:
            values["local_identity"] = LocalModelIdentity.from_dict(values["local_identity"])
        return cls(**values)


def validate_features(
    config: ModelConfig, features: FeatureChoices | None = None, *, require_verified: bool = False,
) -> CapabilityReport:
    """Reject unsupported choices locally; require probes before batch admission."""
    report = config.inspect(features)
    if report.features.mia_scoring:
        for name in ("logprobs", "top_logprobs"):
            if name in config.parameters and config.parameters[name] is None:
                raise ValueError(f"MIA/MCS conflicts with {name} set to explicit null; change the configuration")
        if config.parameters.get("logprobs") is False or config.parameters.get("top_logprobs") == 0:
            raise ValueError("MIA/MCS conflicts with explicitly disabled logprobs or top_logprobs")
    if report.unsupported:
        raise ValueError("Unsupported requested capabilities: " + ", ".join(report.unsupported))
    if require_verified and report.requires_probe:
        raise ValueError("Generation probe required for unverified capabilities: " + ", ".join(report.requires_probe))
    return report


@dataclass(frozen=True)
class TokenAlternative:
    """One raw provider-reported logprob, including wire censoring markers.

    Values use natural log units except the reserved ``-9999.0`` marker, which
    denotes a censored probability. ModelResponse keeps it for audit and text
    generation but excludes it from genuine evidence for MIA/MCS.
    """

    token: str
    logprob: float

    def __post_init__(self) -> None:
        _text(self.token, "token", allow_empty=True)
        _number(self.logprob, "logprob", minimum=-math.inf)
        if self.logprob > 0:
            raise ValueError("logprob must be at most zero")


@dataclass(frozen=True)
class TokenEvidence(TokenAlternative):
    """One realized output token; alternatives may be unavailable for MIA."""

    top_logprobs: tuple[TokenAlternative, ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        entries = tuple(self.top_logprobs)
        if not all(isinstance(entry, TokenAlternative) for entry in entries):
            raise TypeError("top_logprobs must contain TokenAlternative values")
        object.__setattr__(self, "top_logprobs", entries)


@dataclass(frozen=True)
class ModelResponse:
    """Provider output with optional evidence and explicitly unknown metadata.

    ``token_evidence`` is ordered by output position. ``usage`` retains only
    reported token counters; absent counters and usage remain unknown. Requested
    settings do not become confirmed settings merely because a call succeeded.
    Raw token values retain censoring markers; inspect availability before
    treating them as measured log probabilities.
    """

    content: str
    token_evidence: tuple[TokenEvidence, ...] | None = None
    usage: Mapping[str, int] | None = None
    finish_reason: str | None = None
    requested_settings: Mapping[str, ParameterValue] = field(default_factory=dict)
    confirmed_settings: Mapping[str, ParameterValue] = field(default_factory=dict)
    request_id: str | None = None
    readiness_trace: tuple[ReadinessObservation, ...] | None = None
    local_identity: LocalModelIdentity | None = None

    def __post_init__(self) -> None:
        _text(self.content, "content", allow_empty=True)
        if self.local_identity is not None and not isinstance(self.local_identity, LocalModelIdentity):
            raise TypeError("local_identity must be LocalModelIdentity or None")
        if self.readiness_trace is not None:
            trace = tuple(self.readiness_trace)
            if not all(isinstance(item, ReadinessObservation) for item in trace):
                raise TypeError("readiness_trace must contain ReadinessObservation values")
            object.__setattr__(self, "readiness_trace", trace)
        for name in ("finish_reason", "request_id"):
            if getattr(self, name) is not None:
                _text(getattr(self, name), name)
        if self.token_evidence is not None:
            entries = tuple(self.token_evidence)
            if not all(isinstance(entry, TokenEvidence) for entry in entries):
                raise TypeError("token_evidence must contain TokenEvidence values")
            object.__setattr__(self, "token_evidence", entries)
        if self.usage is not None:
            if not isinstance(self.usage, Mapping):
                raise TypeError("usage must be a mapping of reported token counters")
            for name, count in self.usage.items():
                _text(name, "usage counter name")
                _integer(count, "usage token count")
            object.__setattr__(self, "usage", MappingProxyType(dict(self.usage)))
        object.__setattr__(self, "requested_settings", _parameters(self.requested_settings))
        object.__setattr__(self, "confirmed_settings", _parameters(self.confirmed_settings))

    @property
    def logprobs_unavailable_reason(self) -> str | None:
        """Why complete measured realized-token logprobs are unavailable."""
        if not self.token_evidence:
            return "MIA/MCS token evidence is unavailable or empty."
        if any(entry.logprob == _CENSORED_LOGPROB for entry in self.token_evidence):
            return "MIA/MCS token evidence contains a censored realized-token logprob (-9999.0)."
        return None

    @property
    def mia_unavailable_reason(self) -> str | None:
        if reason := self.logprobs_unavailable_reason:
            return reason
        if any(not entry.top_logprobs for entry in self.token_evidence):
            return "MIA/MCS token evidence lacks top_logprobs alternatives."
        if any(
            alt.logprob == _CENSORED_LOGPROB for entry in self.token_evidence for alt in entry.top_logprobs
        ):
            return "MIA/MCS token evidence contains a censored top_logprobs alternative (-9999.0)."
        return None

    @classmethod
    def from_legacy(
        cls, result: CompletionResult, *, requested_settings: Mapping[str, ParameterValue] | None = None,
    ) -> "ModelResponse":
        """Snapshot legacy output; only an echoed temperature is confirmed."""
        if not isinstance(result, CompletionResult):
            raise TypeError("result must be the legacy CompletionResult")
        entries = tuple(TokenEvidence(
            token=entry.token, logprob=entry.logprob,
            top_logprobs=tuple(TokenAlternative(token=alt["token"], logprob=alt["logprob"])
                               for alt in entry.top_logprobs),
        ) for entry in result.logprobs)
        confirmed = {} if result.raw_temperature_observed is None else {"temperature": result.raw_temperature_observed}
        response = cls(result.content, entries, requested_settings=requested_settings or {}, confirmed_settings=confirmed)
        if reason := response.mia_unavailable_reason:
            raise ValueError(reason)
        return response

    def to_legacy(self) -> CompletionResult:
        """Convert only complete genuine evidence, producing fresh legacy lists."""
        if reason := self.mia_unavailable_reason:
            raise ValueError(reason)
        return CompletionResult(
            content=self.content,
            logprobs=[TokenLogprob(entry.token, entry.logprob,
                                  [{"token": alt.token, "logprob": alt.logprob} for alt in entry.top_logprobs])
                      for entry in self.token_evidence],
            raw_temperature_observed=self.confirmed_settings.get("temperature"),
        )

    def to_dict(self) -> dict:
        result = {
            "content": self.content,
            "token_evidence": None if self.token_evidence is None else [
                {"token": entry.token, "logprob": entry.logprob,
                 "top_logprobs": [{"token": alt.token, "logprob": alt.logprob} for alt in entry.top_logprobs]}
                for entry in self.token_evidence],
            "usage": None if self.usage is None else dict(self.usage), "finish_reason": self.finish_reason,
            "requested_settings": _plain(self.requested_settings), "confirmed_settings": _plain(self.confirmed_settings),
            "request_id": self.request_id,
        }
        if self.readiness_trace is not None:
            result["readiness_trace"] = [item.to_dict() for item in self.readiness_trace]
        if self.local_identity is not None:
            result["local_identity"] = self.local_identity.to_dict()
        return result

    @classmethod
    def from_dict(cls, data):
        values = _fields(data, {"content", "token_evidence", "usage", "finish_reason", "requested_settings",
                               "confirmed_settings", "request_id", "readiness_trace", "local_identity"})
        if values.get("token_evidence") is not None:
            tokens = []
            for entry in values["token_evidence"]:
                token = _fields(entry, {"token", "logprob", "top_logprobs"})
                token["top_logprobs"] = tuple(TokenAlternative(**_fields(alt, {"token", "logprob"}))
                                             for alt in token.get("top_logprobs", ()))
                tokens.append(TokenEvidence(**token))
            values["token_evidence"] = tuple(tokens)
        if "readiness_trace" in values:
            values["readiness_trace"] = tuple(ReadinessObservation.from_dict(item) for item in values["readiness_trace"])
        if values.get("local_identity") is not None:
            values["local_identity"] = LocalModelIdentity.from_dict(values["local_identity"])
        return cls(**values)


@runtime_checkable
class CompletionClient(Protocol):
    """Synchronous boundary usable directly inside a notebook's running loop.

    Per-call generation controls support the existing ensemble's explicit
    overrides. Adapters validate them without mutating the configuration.
    """

    config: ModelConfig

    def generate(self, prompt: str, **parameters: ParameterValue) -> ModelResponse: ...
