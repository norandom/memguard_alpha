"""Bind genuine configured-client evidence to the existing logprob scorer."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from threading import Lock
from types import MappingProxyType

from recall_guard.core.completion import (
    CapabilityReport,
    CompletionClient,
    CompletionProtocol,
    FeatureChoices,
    ModelConfig,
    ModelResponse,
    ParameterValue,
    validate_features,
)
from recall_guard.core.nvidia_lm import CompletionResult
from recall_guard.core.request_control import BudgetExhaustedError


@dataclass(frozen=True)
class CalibrationIdentity:
    """Immutable fit provenance; omitted and explicitly null settings differ.

    ``parameters_json`` is canonical JSON containing only generation settings.
    Credential references, resolved secrets and transport retries are excluded.
    An intervention ID names the pipeline/template used on both fit corpora.
    """

    protocol: CompletionProtocol
    endpoint: str | None
    model: str
    parameters_json: str
    intervention_id: str

    @classmethod
    def from_config(cls, config: ModelConfig, intervention_id: str) -> CalibrationIdentity:
        if not isinstance(intervention_id, str) or not intervention_id.strip():
            raise ValueError("intervention_id must identify the calibration intervention")
        return cls(config.protocol, config.endpoint, config.model,
                   json.dumps(config.to_dict()["parameters"], sort_keys=True, separators=(",", ":")),
                   intervention_id)


class LegacyLogprobAdapter:
    """Expose complete ModelResponses to the unchanged MIA fitting functions.

    Legacy batch helpers return exceptions as skippable rows. A sticky fatal
    failure stops queued calls at this boundary and must be checked after each
    fit stage, even when fitting itself raises. Already admitted concurrent
    requests may finish. Ordinary timeouts/transport failures remain skippable.
    """

    def __init__(
        self, client: CompletionClient, intervention_id: str,
        is_auth_error: Callable[[BaseException], bool],
    ) -> None:
        self.client = client
        self.identity = CalibrationIdentity.from_config(client.config, intervention_id)
        self._is_auth_error = is_auth_error
        self._fatal: Exception | None = None
        self._confirmed_settings = {}
        self._fitting = True
        self._lock = Lock()

    @property
    def model(self) -> str:
        return self.identity.model

    def validate_client(
        self, client: CompletionClient, *, intervention_id: str, **parameters: ParameterValue,
    ) -> None:
        if CalibrationIdentity.from_config(client.config, intervention_id) != self.identity:
            raise ValueError("Client model/endpoint/protocol/settings/intervention identity differs from calibration")
        config = replace(client.config, parameters=dict(client.config.parameters) | parameters)
        if CalibrationIdentity.from_config(config, intervention_id) != self.identity:
            raise ValueError("Client model/endpoint/protocol/settings/intervention identity differs from calibration")

    def validate_response(self, response: ModelResponse) -> CompletionResult:
        if not isinstance(response, ModelResponse):
            raise ValueError("Configured MIA scoring requires a ModelResponse with genuine token evidence")
        settings = ModelConfig(self.identity.protocol, self.identity.endpoint, self.identity.model,
                               parameters=response.requested_settings)
        if CalibrationIdentity.from_config(settings, self.identity.intervention_id) != self.identity:
            raise ValueError("Response requested settings differ from calibration identity")
        result = response.to_legacy()
        with self._lock:
            expected = dict(response.requested_settings) | self._confirmed_settings
            for name, value in response.confirmed_settings.items():
                if name in expected and value != expected[name]:
                    raise ValueError(f"Response confirmed setting {name!r} differs from calibration settings")
            if self._fitting:
                self._confirmed_settings.update(response.confirmed_settings)
        return result

    def finish_calibration(self) -> None:
        """Freeze observed fit settings; inference cannot change fit provenance."""
        with self._lock:
            self._confirmed_settings = MappingProxyType(dict(self._confirmed_settings))
            self._fitting = False

    def probe(self) -> None:
        features = FeatureChoices(mia_scoring=True)
        required = validate_features(self.client.config, features).capabilities
        probe = getattr(self.client, "probe", None)
        if not callable(probe):
            raise ValueError("An actual generation probe is required before MIA calibration")
        report = probe()
        self.validate_client(self.client, intervention_id=self.identity.intervention_id)
        if not isinstance(report, CapabilityReport) or set(required) - report.capabilities.keys():
            raise ValueError("Generation probe lacks required MIA capabilities")
        selected = CapabilityReport({name: report.capabilities[name] for name in required}, features)
        if not selected.ready_for_batch:
            raise ValueError("MIA generation probe is not ready: complete observed token/control capabilities required")

    def raise_if_fatal(self) -> None:
        with self._lock:
            failure = self._fatal
        if failure is not None:
            raise failure

    def generate(self, prompt: str, **parameters: ParameterValue) -> CompletionResult:
        self.raise_if_fatal()
        try:
            self.validate_client(self.client, intervention_id=self.identity.intervention_id, **parameters)
            response = self.client.generate(prompt, **parameters)
            self.validate_client(self.client, intervention_id=self.identity.intervention_id)
            return self.validate_response(response)
        except Exception as error:
            if getattr(error, "code", None) == "missing_token_evidence":
                error = ValueError("MIA/MCS genuine token evidence is unavailable or incomplete")
            if isinstance(error, (BudgetExhaustedError, ValueError)) or self._is_auth_error(error):
                with self._lock:
                    if self._fatal is None:
                        self._fatal = error
            raise error
