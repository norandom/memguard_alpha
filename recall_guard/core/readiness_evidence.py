"""Response-dependent readiness evidence, above the pure readiness types.

Keeping this module separate avoids reciprocal imports when ModelConfig gains
an optional ReadinessPolicy. Existing token and capability rules remain owned
by completion.py; partial output can never masquerade as ModelResponse.
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum

from recall_guard.core.completion import (
    CapabilityEvidence,
    CapabilityReport,
    FeatureChoices,
    ModelConfig,
    ModelResponse,
    ParameterValue,
    TokenEvidence,
    _integer,
    _plain,
)
from recall_guard.core.readiness_types import ReadinessObservation, ReadinessState, _fields


class PartialResponseReason(StrEnum):
    PREMATURE_EOF = "premature_eof"
    INTERRUPTED = "interrupted"
    MALFORMED = "malformed"
    EVENT_LIMIT = "event_limit"
    RESPONSE_LIMIT = "response_limit"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class PartialResponse:
    """Bounded retained answer/token evidence from an incomplete response.

    Tokens preserve output position and reuse ModelResponse validation. Reasoning
    progress is counted in Unicode characters, never appended to answer content.
    ``max_retained_bytes`` caps the UTF-8 JSON evidence payload, including token
    alternatives; assemblers must truncate retained data to fit before creation.
    Missing usage remains unknown. Even fully populated partial token rows do
    not establish complete MCS evidence or a usable answer.
    """

    content: str
    token_evidence: tuple[TokenEvidence, ...] | None = None
    usage: Mapping[str, int] | None = None
    finish_reason: str | None = None
    requested_settings: Mapping[str, ParameterValue] = field(default_factory=dict)
    confirmed_settings: Mapping[str, ParameterValue] = field(default_factory=dict)
    request_id: str | None = None
    reason: PartialResponseReason = PartialResponseReason.PREMATURE_EOF
    reasoning_characters: int = 0
    max_retained_bytes: int = 8 * 1024 * 1024
    complete: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason", PartialResponseReason(self.reason))
        _integer(self.reasoning_characters, "reasoning_characters")
        _integer(self.max_retained_bytes, "max_retained_bytes", minimum=1)
        validated = ModelResponse(self.content, self.token_evidence, self.usage, self.finish_reason,
                                  self.requested_settings, self.confirmed_settings, self.request_id)
        for name in ("token_evidence", "usage", "requested_settings", "confirmed_settings"):
            object.__setattr__(self, name, getattr(validated, name))
        if len(json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False).encode("utf-8")) > self.max_retained_bytes:
            raise ValueError("Partial response exceeds max_retained_bytes")

    @property
    def usable_answer(self) -> bool:
        return False

    @property
    def mia_unavailable_reason(self) -> str:
        return "MIA/MCS is unavailable because the provider response is incomplete."

    def to_dict(self) -> dict:
        # Reuse the existing explicit token/settings whitelist, not dataclass asdict.
        response = ModelResponse(self.content, self.token_evidence, self.usage, self.finish_reason,
                                 self.requested_settings, self.confirmed_settings, self.request_id)
        return response.to_dict() | {"complete": False, "reason": self.reason.value,
                                     "reasoning_characters": self.reasoning_characters,
                                     "max_retained_bytes": self.max_retained_bytes}

    @classmethod
    def from_dict(cls, data):
        values = _fields(data, {"content", "token_evidence", "usage", "finish_reason", "requested_settings",
                               "confirmed_settings", "request_id", "complete", "reason", "reasoning_characters",
                               "max_retained_bytes"}, required={"complete"})
        if values.pop("complete") is not False:
            raise ValueError("Partial response must remain incomplete")
        partial = {name: values.pop(name) for name in ("reason", "reasoning_characters", "max_retained_bytes") if name in values}
        response = ModelResponse.from_dict(values)
        return cls(response.content, response.token_evidence, response.usage, response.finish_reason,
                   response.requested_settings, response.confirmed_settings, response.request_id, **partial)


@dataclass(frozen=True)
class ProbeEvidence:
    """One configuration-specific capability probe and its operation trace.

    A complete response may fail required capability checks and is retained for
    explicit fallback preparation. ``ready`` means the final observation passed
    those checks for matching requested settings at its timestamp; it promises
    no future availability. Requested settings never imply confirmed settings. Runtime
    cancellation/callbacks and credential references are not retained or exported.
    """

    config: ModelConfig
    report: CapabilityReport
    response: ModelResponse | PartialResponse | None
    observations: tuple[ReadinessObservation, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.config, ModelConfig) or not isinstance(self.report, CapabilityReport):
            raise TypeError("config and report must be ModelConfig and CapabilityReport")
        object.__setattr__(self, "config", replace(self.config, credential_ref=None))
        if self.response is not None and not isinstance(self.response, (ModelResponse, PartialResponse)):
            raise TypeError("response must be complete/partial typed evidence or None")
        observations = tuple(self.observations)
        if not observations or not all(isinstance(item, ReadinessObservation) for item in observations):
            raise ValueError("observations must contain a nonempty typed operation trace")
        operation_id = observations[0].operation_id
        previous_elapsed_s = 0.0
        for item in observations:
            if item.model != self.config.model or item.operation_id != operation_id:
                raise ValueError("observations must belong to one operation and the selected model")
            if item.elapsed_s < previous_elapsed_s:
                raise ValueError("observations must preserve monotonic elapsed_s order")
            previous_elapsed_s = item.elapsed_s
            if item.state == ReadinessState.READY and not self._has_ready_evidence():
                raise ValueError("ready requires a complete response and observed required capabilities")
        object.__setattr__(self, "observations", observations)

    def _has_ready_evidence(self) -> bool:
        if not isinstance(self.response, ModelResponse) or not self.response.content.strip():
            return False
        # Match configuration identity semantics, including true versus 1 in
        # nested JSON controls; object insertion order is not significant.
        requested_json = json.dumps(_plain(self.response.requested_settings), sort_keys=True, separators=(",", ":"))
        selected_json = json.dumps(_plain(self.config.parameters), sort_keys=True, separators=(",", ":"))
        if requested_json != selected_json:
            return False
        required = self.config.inspect(self.report.features).capabilities
        if not required.keys() <= self.report.capabilities.keys() or not self.report.ready_for_batch:
            return False
        requires_logprobs = self.report.features.mia_scoring or self.config.parameters.get("logprobs") is True
        requires_alternatives = self.report.features.mia_scoring or (self.config.parameters.get("top_logprobs") or 0) > 0
        if requires_logprobs and self.response.logprobs_unavailable_reason is not None:
            return False
        if requires_alternatives and self.response.mia_unavailable_reason is not None:
            return False
        return True

    @property
    def complete(self) -> bool:
        return isinstance(self.response, ModelResponse)

    @property
    def ready(self) -> bool:
        return self.observations[-1].state == ReadinessState.READY

    def to_dict(self) -> dict:
        features = self.report.features
        response = None if self.response is None else self.response.to_dict() | {"complete": self.complete}
        return {"config": self.config.to_dict(),
                "report": {"capabilities": {name: value.to_dict() for name, value in self.report.capabilities.items()},
                           "features": {"mia_scoring": features.mia_scoring, "ensemble_draws": features.ensemble_draws,
                                        "rewriting": features.rewriting}},
                "response": response, "observations": [item.to_dict() for item in self.observations]}

    @classmethod
    def from_dict(cls, data):
        values = _fields(data, {"config", "report", "response", "observations"},
                         required={"config", "report", "response", "observations"})
        report = _fields(values["report"], {"capabilities", "features"}, required={"capabilities", "features"})
        report = CapabilityReport(
            {name: CapabilityEvidence(**_fields(entry, {"status", "source", "detail", "observed"}))
             for name, entry in report["capabilities"].items()},
            FeatureChoices(**_fields(report["features"], {"mia_scoring", "ensemble_draws", "rewriting"})),
        )
        response = values["response"]
        if response is not None:
            if response.get("complete") is False:
                response = PartialResponse.from_dict(response)
            elif response.get("complete") is True:
                response = ModelResponse.from_dict({key: value for key, value in response.items() if key != "complete"})
            else:
                raise ValueError("Probe response requires an explicit complete marker")
        return cls(ModelConfig.from_dict(values["config"]), report, response,
                   tuple(ReadinessObservation.from_dict(item) for item in values["observations"]))
