"""Synchronous OpenAI-chat transport with optional provider-reported token data.

OpenAI, Mistral and NIM use this explicitly selected wire protocol. This adapter
does not translate native parameter names or claim model-specific support from
an endpoint URL. A probe verifies acceptance, not unechoed effective settings.
"""

import json
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
    TokenAlternative,
    TokenEvidence,
    validate_features,
)
from recall_guard.core.request_control import CallBudget, RequestControl, RequestError

_ECHOED_CONTROLS = frozenset({"temperature", "top_p", "seed", "random_seed", "max_tokens", "max_completion_tokens"})
_PROBE_PROMPT = "Reply with the word OK."


def _reported_usage(body: object) -> dict[str, int] | None:
    """Keep flat provider-reported counters; never estimate nested/absent ones."""
    usage = body.get("usage") if isinstance(body, dict) else None
    if not isinstance(usage, dict):
        return None
    return {name: count for name, count in usage.items()
            if isinstance(name, str) and name.strip()
            and isinstance(count, int) and not isinstance(count, bool) and count >= 0}


def _string(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _http_failure(response: requests.Response, body: object) -> RequestError:
    """Preserve heterogeneous error envelopes, including unknown diagnostics."""
    envelope = body if isinstance(body, dict) else {}
    error = envelope.get("error")
    error = error if isinstance(error, dict) else envelope
    code = _string(error.get("code")) or _string(error.get("type"))
    parameter = _string(error.get("param")) or _string(error.get("parameter"))
    message = _string(error.get("message")) or _string(error.get("title"))
    details = error.get("detail")
    if isinstance(details, list) and details and isinstance(details[0], dict):
        diagnostic = details[0]
        location = diagnostic.get("loc")
        if isinstance(location, list) and len(location) > 1:
            parameter = parameter or _string(location[-1])
        code = code or _string(diagnostic.get("type"))
        message = message or _string(diagnostic.get("msg"))
    detail = response.text if body is None else json.dumps(body, ensure_ascii=False)
    return RequestError(
        f"HTTP {response.status_code}: {message or 'completion request rejected'}",
        status_code=response.status_code, parameter=parameter, detail=detail, code=code,
        usage=_reported_usage(body),
    )


def _transport_failure(error: requests.RequestException) -> RequestError:
    """Retain any provider response attached to a transport-level exception."""
    retryable = isinstance(error, (requests.Timeout, requests.ConnectionError))
    code = ("timeout" if isinstance(error, requests.Timeout) else
            "connection_error" if isinstance(error, requests.ConnectionError) else "transport_error")
    if error.response is None:
        return RequestError(str(error), code=code, retryable=retryable)
    try:
        body = error.response.json()
    except ValueError:
        body = None
    failure = _http_failure(error.response, body)
    return RequestError(str(error), status_code=failure.status_code, parameter=failure.parameter,
                        detail=failure.detail, code=failure.code or code, usage=failure.usage, retryable=retryable)


def _token_evidence(choice: dict) -> tuple[TokenEvidence, ...] | None:
    section = choice.get("logprobs")
    if section is None:
        return None
    if not isinstance(section, dict):
        raise ValueError("logprobs must be an object or null")
    entries = section.get("content")
    if entries is None:
        return None
    if not isinstance(entries, list):
        raise ValueError("logprobs.content must be a list or null")
    parsed = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("logprobs.content must contain token objects")
        alternatives = entry.get("top_logprobs")
        alternatives = [] if alternatives is None else alternatives
        if not isinstance(alternatives, list) or not all(isinstance(alt, dict) for alt in alternatives):
            raise ValueError("top_logprobs must contain token objects")
        parsed.append(TokenEvidence(
            token=entry["token"], logprob=entry["logprob"],
            top_logprobs=tuple(TokenAlternative(token=alt["token"], logprob=alt["logprob"]) for alt in alternatives),
        ))
    return tuple(parsed)


def _chat_response(response: requests.Response, body: object, parameters: dict) -> ModelResponse:
    usage = _reported_usage(body)
    try:
        if not isinstance(body, dict):
            raise ValueError("response must be a chat-completion object")
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ValueError("response must contain choices[0]")
        choice = choices[0]
        message = choice.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise ValueError("choices[0].message.content must be a text string")
        confirmed = {name: choice.get(name, body.get(name)) for name in parameters.keys() & _ECHOED_CONTROLS
                     if name in choice or name in body}
        return ModelResponse(
            content=message["content"], token_evidence=_token_evidence(choice), usage=usage,
            finish_reason=choice.get("finish_reason"), requested_settings=parameters,
            confirmed_settings=confirmed,
            request_id=_string(response.headers.get("x-request-id")) or _string(body.get("id")),
        )
    except (KeyError, ValueError, TypeError) as error:
        raise RequestError(f"Invalid OpenAI-chat response: {error}", code="invalid_response", usage=usage) from None


class OpenAICompatibleClient:
    """Configured text/logprob client; every physical POST shares one budget.

    Resolve the credential reference when constructing this client, from the
    supplied environment or ``os.environ``. Recreate it to rotate credentials.
    No environment file is read. Anonymous local endpoints require explicit
    ``allow_unauthenticated=True`` and an omitted credential reference.

    Each attempt uses a fresh requests session with transport retries disabled.
    Redirects are rejected and ambient netrc/proxy authentication is disabled;
    only the explicitly resolved credential is sent to the configured endpoint.
    Sessions are not shared across concurrent calls. RequestControl alone owns
    admission, pacing and retries, including probes and explicitly named stages.
    """

    def __init__(
        self, config: ModelConfig, budget: CallBudget, *, features: FeatureChoices | None = None,
        environment: Mapping[str, str] | None = None, allow_unauthenticated: bool = False,
    ) -> None:
        if not isinstance(config, ModelConfig):
            raise TypeError("config must be ModelConfig")
        if config.protocol != CompletionProtocol.OPENAI_CHAT:
            raise ValueError("OpenAICompatibleClient requires the explicit openai_chat protocol")
        self.config = config
        self.features = FeatureChoices() if features is None else features
        validate_features(config, self.features)
        if not isinstance(allow_unauthenticated, bool):
            raise TypeError("allow_unauthenticated must be boolean")
        environment = os.environ if environment is None else environment
        if not isinstance(environment, Mapping):
            raise TypeError("environment must be a mapping")
        credential = None
        if config.credential_ref is not None:
            credential = environment.get(config.credential_ref.removeprefix("env:"))
            if not isinstance(credential, str) or not credential.strip():
                raise ValueError("Referenced credential is missing or empty")
        elif not allow_unauthenticated:
            raise ValueError("A credential reference is required unless unauthenticated access is explicitly enabled")
        self._credential = credential
        self._control = RequestControl(config, budget, secrets=() if credential is None else (credential,))

    @property
    def budget(self) -> CallBudget:
        """The actual shared physical-request budget, for workflow validation."""
        return self._control.budget

    def generate(self, prompt: str, **parameters: ParameterValue) -> ModelResponse:
        """Generate using configured settings plus explicit per-call overrides."""
        return self._generate(prompt, parameters, stage="target", enforce_mia=self.features.mia_scoring)

    def generate_for_stage(self, prompt: str, *, stage: str, **parameters: ParameterValue) -> ModelResponse:
        """Name the budget-ledger stage without reserving a generation control."""
        return self._generate(prompt, parameters, stage=stage, enforce_mia=self.features.mia_scoring)

    def _generate(self, prompt: str, overrides: dict, *, stage: str, enforce_mia: bool) -> ModelResponse:
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        settings = replace(self.config, parameters=dict(self.config.parameters) | overrides)
        validate_features(settings, self.features)
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
            result = _chat_response(response, body, parameters)
            if enforce_mia and (reason := result.mia_unavailable_reason):
                raise RequestError(reason, code="missing_token_evidence", usage=result.usage)
            return result

        return self._control.run(operation, stage=stage)

    def _post(self, payload: dict) -> requests.Response:
        headers = {"Content-Type": "application/json"}
        if self._credential is not None:
            headers["Authorization"] = f"Bearer {self._credential}"
        try:
            with requests.Session() as session:
                session.trust_env = False
                session.mount("http://", requests.adapters.HTTPAdapter(max_retries=0))
                session.mount("https://", requests.adapters.HTTPAdapter(max_retries=0))
                return session.post(self.config.endpoint, headers=headers, json=payload,
                                    timeout=self.config.timeout_s, allow_redirects=False)
        except requests.RequestException as error:
            raise _transport_failure(error) from None

    def probe(self, prompt: str = _PROBE_PROMPT) -> CapabilityReport:
        """Perform real generation; distinguish acceptance from effective values.

        HTTP/authentication/billing/parameter failures propagate as RequestError.
        A successful text-only probe reports missing requested token capabilities
        explicitly instead of converting missing probabilities to a call error.
        Empty or whitespace-only output leaves text generation unverified;
        ordinary generation still retains those valid responses and metadata.
        The report concerns this exact configured parameter set, endpoint/model
        and feature selection. It is not a guarantee for changed overrides.
        """
        response = self._generate(prompt, {}, stage="probe", enforce_mia=False)
        source = "Authenticated generation probe" if self._credential is not None else "Explicit unauthenticated generation probe"
        capabilities = {}
        for name in self.config.inspect(self.features).capabilities:
            status = CapabilityStatus.SUPPORTED
            observed = True
            detail = "Text response observed."
            if name == "text_generation" and not response.content.strip():
                status = CapabilityStatus.UNVERIFIED
                observed = False
                token_budget = {key: value for key, value in response.requested_settings.items()
                                if key in {"max_tokens", "max_completion_tokens"}}
                detail = (f"No non-whitespace text observed; finish_reason={response.finish_reason!r}, "
                          f"requested token budget={token_budget or 'omitted'}. "
                          "Inspect the response, adjust the token budget if needed, and repeat the probe before batch execution.")
            elif name == "logprobs" and (self.features.mia_scoring or self.config.parameters.get(name) is True):
                reason = response.logprobs_unavailable_reason
                status = CapabilityStatus.SUPPORTED if reason is None else CapabilityStatus.UNSUPPORTED
                detail = reason or "Genuine token logprobs observed."
            elif name == "top_logprobs" and (self.features.mia_scoring or self.config.parameters.get(name)):
                status = CapabilityStatus.SUPPORTED if response.mia_unavailable_reason is None else CapabilityStatus.UNSUPPORTED
                detail = response.mia_unavailable_reason or "Genuine alternatives observed for every token."
            elif name != "text_generation":
                detail = ("Parameter accepted; effective value echoed in response." if name in response.confirmed_settings
                          else "Parameter accepted; effective value remains unconfirmed (not echoed).")
            capabilities[name] = CapabilityEvidence(status, source, detail, observed=observed)
        return CapabilityReport(capabilities, self.features)
