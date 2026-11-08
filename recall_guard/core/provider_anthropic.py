"""Synchronous native Anthropic Messages transport with text-only evidence.

The audited first-party API version is fixed, not inferred from the model or URL.
Typed thinking/tool blocks are never presented as answer text or token logprobs.
"""

import os
from collections.abc import Mapping
from dataclasses import replace

import requests

from recall_guard.core.completion import (
    CapabilityEvidence,
    CapabilityReport,
    CapabilityStatus,
    CompletionProtocol,
    FeatureChoices,
    ModelConfig,
    ModelResponse,
    ParameterValue,
    validate_features,
)
from recall_guard.core.providers import _http_failure, _reported_usage, _string, _transport_failure
from recall_guard.core.request_control import CallBudget, RequestControl, RequestError

ANTHROPIC_API_VERSION = "2026-10-21"
_UNSUPPORTED_PARAMETERS = frozenset({
    "temperature", "top_p", "seed", "random_seed", "logprobs", "top_logprobs", "max_completion_tokens",
    "reasoning_effort",
})


def validate_anthropic_config(config: ModelConfig, features: FeatureChoices | None = None) -> None:
    """Validate native settings without resolving credentials or issuing calls."""
    if not isinstance(config, ModelConfig):
        raise TypeError("config must be ModelConfig")
    if config.protocol != CompletionProtocol.ANTHROPIC_MESSAGES:
        raise ValueError("AnthropicClient requires the explicit anthropic_messages protocol")
    if config.readiness is not None:
        raise ValueError("Anthropic observation profile is unsupported")
    features = FeatureChoices() if features is None else features
    validate_features(config, features)
    if features.mia_scoring:
        raise ValueError(
            f"Anthropic Messages API {ANTHROPIC_API_VERSION} provides no logprobs/top_logprobs for MIA/MCS; "
            "set mia_scoring=False or explicitly select a logprob-capable client"
        )
    if unsupported := sorted(config.parameters.keys() & _UNSUPPORTED_PARAMETERS):
        raise ValueError(
            f"Unsupported Anthropic Messages API {ANTHROPIC_API_VERSION} parameters: {', '.join(unsupported)}. "
            "Remove these explicit settings from the configuration; use native max_tokens for the output budget."
        )
    if config.parameters.get("max_tokens") is None:
        raise ValueError("Anthropic Messages requires an explicit positive max_tokens in the configuration")


def _messages_response(response: requests.Response, body: object, parameters: dict) -> ModelResponse:
    usage = _reported_usage(body)
    try:
        if not isinstance(body, dict) or not isinstance(body.get("content"), list):
            raise ValueError("response must contain a native content block list")
        text = []
        for block in body["content"]:
            if not isinstance(block, dict) or _string(block.get("type")) is None:
                raise ValueError("content must contain typed block objects")
            if block["type"] == "text":
                if not isinstance(block.get("text"), str):
                    raise ValueError("text blocks must contain a text string")
                text.append(block["text"])
        return ModelResponse(
            content="".join(text), usage=usage, finish_reason=body.get("stop_reason"),
            requested_settings=parameters,
            request_id=_string(response.headers.get("request-id")) or _string(body.get("id")),
        )
    except (TypeError, ValueError) as error:
        raise RequestError(f"Invalid Anthropic Messages response: {error}", code="invalid_response", usage=usage) from None


class AnthropicClient:
    """Native first-party Messages client for a single-workspace API key.

    The fixed API version is ``2026-10-21``; authentication uses ``x-api-key``.
    Keys requiring explicit workspace selection are outside this adapter.
    Resolve the credential reference from the supplied mapping or ``os.environ``
    at construction; recreate the client after rotation. No env file is read.

    Supply native ``max_tokens`` explicitly and optional top-level ``system``
    through config parameters or per-call overrides. No generation defaults are
    inserted. Only actual text blocks form the answer; genuine flat usage and
    stop metadata are retained, while MIA/MCS token evidence stays unavailable.

    Every physical POST, including probes/retries, uses shared RequestControl
    admission. Fresh sessions disable hidden retries, redirects and ambient
    netrc/proxy authentication. Synchronous calls work inside a running kernel.
    """

    def __init__(
        self, config: ModelConfig, budget: CallBudget, *, features: FeatureChoices | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> None:
        self.config = config
        self.features = FeatureChoices() if features is None else features
        validate_anthropic_config(config, self.features)
        environment = os.environ if environment is None else environment
        if not isinstance(environment, Mapping):
            raise TypeError("environment must be a mapping")
        if config.credential_ref is None:
            raise ValueError("A credential reference is required for Anthropic Messages")
        credential = environment.get(config.credential_ref.removeprefix("env:"))
        if not isinstance(credential, str) or not credential.strip():
            raise ValueError("Referenced credential is missing or empty")
        self._credential = credential
        self._control = RequestControl(config, budget, secrets=(credential,))

    @property
    def budget(self) -> CallBudget:
        """The actual shared physical-request budget, for workflow validation."""
        return self._control.budget

    def generate(self, prompt: str, **parameters: ParameterValue) -> ModelResponse:
        """Generate with native configured settings and explicit per-call overrides."""
        return self._generate(prompt, parameters, stage="target")

    def generate_for_stage(self, prompt: str, *, stage: str, **parameters: ParameterValue) -> ModelResponse:
        """Name a budget-ledger stage without reserving a generation control."""
        return self._generate(prompt, parameters, stage=stage)

    def generate_for_execution(self, prompt, *, stage, cancellation=None, on_observation=None, **parameters):
        """Opt-in runtime controls with the existing synchronous transport."""
        from recall_guard.core.provider_probe import legacy_generation

        return legacy_generation(self.config, self.features, self.budget,
            lambda guard, wait: self._generate(prompt, parameters, stage=stage, purpose="inference",
                                               admission_guard=guard, wait=wait),
            self._probe_report, stage=stage, cancellation=cancellation, on_observation=on_observation,
            sanitize_failure=self.sanitize_execution_failure)

    def sanitize_execution_failure(self, error, *, response=None):
        """Retain failed evidence through the owning registered-secret sanitizer."""
        return self._control.sanitize_failure(error, response=response)

    def _generate(self, prompt: str, overrides: dict, *, stage: str, purpose=None,
                  admission_guard=None, wait=None) -> ModelResponse:
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        settings = replace(self.config, parameters=dict(self.config.parameters) | overrides)
        validate_anthropic_config(settings, self.features)
        parameters = settings.to_dict()["parameters"]
        payload = {"model": settings.model, "messages": [{"role": "user", "content": prompt}], **parameters}

        def operation() -> ModelResponse:
            response = self._post(payload)
            try:
                body = response.json()
            except ValueError:
                body = None
            if not 200 <= response.status_code < 300:
                raise _http_failure(response, body)
            return _messages_response(response, body, parameters)

        return self._control.run(operation, stage=stage, purpose=purpose,
                                         admission_guard=admission_guard, wait=wait)

    def _post(self, payload: dict) -> requests.Response:
        headers = {"Content-Type": "application/json", "anthropic-version": ANTHROPIC_API_VERSION,
                   "x-api-key": self._credential}
        try:
            with requests.Session() as session:
                session.trust_env = False
                session.mount("http://", requests.adapters.HTTPAdapter(max_retries=0))
                session.mount("https://", requests.adapters.HTTPAdapter(max_retries=0))
                return session.post(self.config.endpoint, headers=headers, json=payload,
                                    timeout=self.config.timeout_s, allow_redirects=False)
        except requests.RequestException as error:
            raise _transport_failure(error) from None

    def probe(self, prompt: str = "Reply with the word OK.", *, stage: str = "probe") -> CapabilityReport:
        """Probe actual inference access for this exact endpoint/model/settings.

        HTTP authentication, billing and parameter failures remain RequestError;
        success confirms parameter acceptance, not unechoed effective values.
        Empty, thinking-only or whitespace output leaves generation unverified.
        Inspect ``ready_for_batch`` before scheduling corpus work. Nonempty
        truncated output establishes observed text support and retains its stop
        reason during ordinary generation.
        """
        response = self._generate(prompt, {}, stage=stage)
        return self._probe_report(response)

    def _probe_report(self, response):
        source = f"Anthropic Messages API {ANTHROPIC_API_VERSION} authenticated generation probe"
        capabilities = {}
        for name in self.config.inspect(self.features).capabilities:
            status, observed = CapabilityStatus.SUPPORTED, True
            detail = "Text response observed."
            if name == "text_generation" and not response.content.strip():
                status, observed = CapabilityStatus.UNVERIFIED, False
                detail = (
                    f"No non-whitespace text observed; stop_reason={response.finish_reason!r}, "
                    f"requested max_tokens={response.requested_settings['max_tokens']}. "
                    "Inspect the response, adjust max_tokens if needed, and repeat the probe before batch execution."
                )
            elif name != "text_generation":
                detail = "Parameter accepted; effective value remains unconfirmed (not echoed)."
            capabilities[name] = CapabilityEvidence(status, source, detail, observed=observed)
        return CapabilityReport(capabilities, self.features)

    def probe_with_evidence(self, prompt: str = "Reply with the word OK.", *, stage: str = "probe",
                            cancellation=None, on_observation=None):
        """Probe once through the legacy transport, retaining actual response evidence."""
        from recall_guard.core.provider_probe import legacy_probe

        return legacy_probe(self.config, self.features, self.budget,
                            lambda guard, wait: self._generate(prompt, {}, stage=stage, purpose="probe",
                                    admission_guard=guard, wait=wait), self._probe_report, stage=stage,
                            cancellation=cancellation, on_observation=on_observation)
