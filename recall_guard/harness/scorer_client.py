"""Bind genuine configured-client evidence to the existing logprob scorer."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
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
from recall_guard.core.local_evidence import LocalModelIdentity, validate_local_evidence
from recall_guard.core.nvidia_lm import CompletionResult
from recall_guard.core.profiles import ResponseMapping
from recall_guard.core.providers import HarmonyOpenAICompatibleClient
from recall_guard.core.request_control import BudgetExhaustedError
from recall_guard.harness.request_failures import FATAL_FAILURES, failure_kind


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
    local_identity: LocalModelIdentity | None = None
    #: Evidence profile (v2): calibrators never cross response mappings.
    response_mapping: str = ResponseMapping.GENERIC.value
    #: Pinned mapping identity, e.g. the Harmony tokenizer SHA-256.
    mapping_revision: str | None = None

    def __post_init__(self) -> None:
        try:
            object.__setattr__(self, "response_mapping", ResponseMapping(self.response_mapping).value)
        except ValueError:
            raise ValueError("response_mapping must be generic or harmony_vllm") from None
        if self.mapping_revision is not None and (
                not isinstance(self.mapping_revision, str) or not self.mapping_revision.strip()):
            raise ValueError("mapping_revision must be nonempty text or null")
        if self.response_mapping != ResponseMapping.GENERIC.value and self.mapping_revision is None:
            raise ValueError("A non-generic response mapping requires its pinned mapping_revision")

    @property
    def reuse_scope(self) -> str:
        """Unknown observations allow exploration on the original client only."""
        if self.local_identity is None:
            return "legacy"
        observed = asdict(self.local_identity.observed)
        semantics = observed.pop("score_semantics")
        complete = all(value is not None for value in (*observed.values(), *semantics.values()))
        return "observed_identity" if complete else "within_run"

    def to_dict(self) -> dict:
        """Preserve the exact legacy fingerprint payload when provenance is absent."""
        result = asdict(self)
        if self.local_identity is None:
            result.pop("local_identity")
        if self.response_mapping == ResponseMapping.GENERIC.value and self.mapping_revision is None:
            result.pop("response_mapping")
            result.pop("mapping_revision")
        return result

    @classmethod
    def from_dict(cls, data: Mapping) -> CalibrationIdentity:
        """Load v1 (legacy) and v2 identity payloads; absent mapping means generic."""
        required = {"protocol", "endpoint", "model", "parameters_json", "intervention_id"}
        optional = {"local_identity", "response_mapping", "mapping_revision"}
        if not isinstance(data, Mapping) or required - data.keys() or data.keys() - required - optional:
            raise ValueError("Invalid calibration identity fields")
        values = dict(data)
        local = values.get("local_identity")
        if local is not None:
            local = dict(local)
            local.setdefault("version", 1)  # dataclasses.asdict exports omit the version tag.
            values["local_identity"] = LocalModelIdentity.from_dict(local)
        values["protocol"] = CompletionProtocol(values["protocol"])
        values.setdefault("response_mapping", ResponseMapping.GENERIC.value)
        return cls(**values)

    @classmethod
    def from_config(
        cls, config: ModelConfig, intervention_id: str, *,
        response_mapping: str = ResponseMapping.GENERIC.value, mapping_revision: str | None = None,
    ) -> CalibrationIdentity:
        if not isinstance(intervention_id, str) or not intervention_id.strip():
            raise ValueError("intervention_id must identify the calibration intervention")
        return cls(config.protocol, config.endpoint, config.model,
                   json.dumps(config.to_dict()["parameters"], sort_keys=True, separators=(",", ":")),
                   intervention_id, config.local_identity, response_mapping, mapping_revision)

    @classmethod
    def from_client(cls, client: CompletionClient, intervention_id: str) -> CalibrationIdentity:
        """Configured identity plus the client's response mapping (v2)."""
        mapping, revision = client_mapping(client)
        return cls.from_config(client.config, intervention_id, response_mapping=mapping, mapping_revision=revision)


def client_mapping(client: object) -> tuple[str, str | None]:
    """Return ``(response_mapping, mapping_revision)`` for a configured client."""
    if isinstance(client, HarmonyOpenAICompatibleClient):
        return ResponseMapping.HARMONY_VLLM.value, client.token_map.tokenizer_sha256
    return ResponseMapping.GENERIC.value, None


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
        self.identity = CalibrationIdentity.from_client(client, intervention_id)
        self._client_identity = self.identity
        self._local_response_seen = False
        self._is_auth_error = is_auth_error
        self._fatal: Exception | None = None
        self._confirmed_settings = {}
        self._fitting = True
        self._lock = Lock()

    @property
    def model(self) -> str:
        return self.identity.model

    @property
    def observed_fit_settings(self) -> Mapping[str, ParameterValue]:
        """Detached, deeply immutable controls echoed during calibration only.

        Missing controls remain unobserved. Reuse the response contract's
        recursive validation/copying rather than expose the adapter's state.
        """
        with self._lock:
            return ModelResponse("", confirmed_settings=self._confirmed_settings).confirmed_settings

    def validate_client(
        self, client: CompletionClient, *, intervention_id: str, **parameters: ParameterValue,
    ) -> None:
        expected = self._client_identity if client is self.client else self.identity
        if client is not self.client and self.identity.reuse_scope == "within_run":
            raise ValueError("Unknown local provenance permits within-run calibration reuse on the original client only")
        current = CalibrationIdentity.from_client(client, intervention_id)
        if (current.response_mapping, current.mapping_revision) != (expected.response_mapping, expected.mapping_revision):
            raise ValueError("Client response mapping differs from calibration; calibrators are not reused "
                             "across evidence profiles")
        if current != expected:
            raise ValueError("Client model/endpoint/protocol/settings/intervention identity differs from calibration")
        config = replace(client.config, parameters=dict(client.config.parameters) | parameters)
        mapping = {"response_mapping": current.response_mapping, "mapping_revision": current.mapping_revision}
        if CalibrationIdentity.from_config(config, intervention_id, **mapping) != expected:
            raise ValueError("Client model/endpoint/protocol/settings/intervention identity differs from calibration")

    def validate_response(self, response: ModelResponse) -> CompletionResult:
        if not isinstance(response, ModelResponse):
            raise ValueError("Configured MIA scoring requires a ModelResponse with genuine token evidence")
        settings = ModelConfig(self.identity.protocol, self.identity.endpoint, self.identity.model,
                               parameters=response.requested_settings, local_identity=response.local_identity)
        candidate = CalibrationIdentity.from_config(settings, self.identity.intervention_id,
                                                    response_mapping=self.identity.response_mapping,
                                                    mapping_revision=self.identity.mapping_revision)
        if replace(candidate, local_identity=None) != replace(self.identity, local_identity=None):
            raise ValueError("Response requested settings differ from calibration identity")
        if self.identity.local_identity is not None or response.local_identity is not None:
            self._validate_local_response(response)
        result = response.to_legacy()
        with self._lock:
            expected = dict(response.requested_settings) | self._confirmed_settings
            for name, value in response.confirmed_settings.items():
                if name in expected and value != expected[name]:
                    raise ValueError(f"Response confirmed setting {name!r} differs from calibration settings")
            if self._fitting:
                self._confirmed_settings.update(response.confirmed_settings)
        return result

    def _validate_local_response(self, response: ModelResponse) -> None:
        local = response.local_identity
        if local is None:
            raise ValueError("Response lost local calibration identity")
        validate_local_evidence(response, required_alternatives=1)
        with self._lock:
            expected = self.identity.local_identity
            if not self._fitting and not self._local_response_seen:
                raise ValueError("Local response identity was not observed during calibration")
            if self._local_response_seen:
                if local != expected:
                    raise ValueError("Response local identity differs from calibration")
            elif expected is not None:
                # The first response may supply observations absent from configuration.
                # Only null observations can be filled; declarations stay declarations.
                if local.declared != expected.declared:
                    raise ValueError("Response declared local identity differs from calibration")
                before, after = asdict(expected.observed), asdict(local.observed)
                before.update(before.pop("score_semantics"))
                after.update(after.pop("score_semantics"))
                if any(value is not None and after[name] != value for name, value in before.items()):
                    raise ValueError("Response observed local identity differs from calibration")
            self.identity = replace(self.identity, local_identity=local)
            self._local_response_seen = True

    def finish_calibration(self) -> None:
        """Freeze observed fit settings; inference cannot change fit provenance."""
        with self._lock:
            self._confirmed_settings = MappingProxyType(dict(self._confirmed_settings))
            self._fitting = False

    def probe(self) -> None:
        features = FeatureChoices(mia_scoring=True)
        required = validate_features(self.client.config, features).capabilities
        local = self.identity.local_identity is not None
        probe = getattr(self.client, "probe_with_evidence" if local else "probe", None)
        if not callable(probe):
            raise ValueError("An actual generation probe is required before MIA calibration")
        evidence = probe()
        report = evidence.report if local else evidence
        if local:
            self.validate_response(evidence.response)
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
            if (isinstance(error, (BudgetExhaustedError, ValueError)) or self._is_auth_error(error)
                    or failure_kind(error) in FATAL_FAILURES):
                with self._lock:
                    if self._fatal is None:
                        self._fatal = error
            raise error
