"""Bind genuine configured-client evidence to the existing logprob scorer."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, fields, replace
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
from recall_guard.core.local_evidence import LocalEvidenceError, LocalModelIdentity, validate_local_evidence
from recall_guard.core.nvidia_lm import CompletionResult
from recall_guard.core.profiles import ResponseMapping
from recall_guard.core.providers import HarmonyOpenAICompatibleClient
from recall_guard.core.request_control import BudgetExhaustedError
from recall_guard.harness.request_failures import FATAL_FAILURES, failure_kind
from recall_guard.mia.estimate_source import EstimateSource

#: Mismatch names that differ from the field name: settings live in ``parameters_json``.
_MISMATCH_NAMES = MappingProxyType({"parameters_json": "settings"})
_SHA256_HEX = re.compile(r"[0-9a-f]{64}")
#: Generation settings that select one sample rather than the calibrated distribution.
SEED_SETTINGS = frozenset({"seed", "random_seed"})

#: Local evidence defects confined to one received response. Such a row is excluded
#: and counted; ``min_valid`` then decides whether calibration may proceed.
#: Missing token evidence and text/token misalignment remain fatal: they indicate
#: a lost capability or an unmapped channel, not one unusable answer.
ITEM_EVIDENCE_CODES = frozenset({"empty_output", "missing_alternatives", "duplicate_alternatives",
                                 "censored_logprob", "invalid_logprob"})


class ItemEvidenceError(RuntimeError):
    """One response's evidence is unusable; skippable like a missing-logprob row.

    The message is the stable non-content reason code from local admission.
    """

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.evidence_code = code


class CalibrationMismatchError(ValueError):
    """Evidence identity differs from the calibrator's; ``fields`` names the differences."""

    def __init__(self, fields: tuple[str, ...]) -> None:
        super().__init__("Evidence differs from the calibration identity in: " + ", ".join(fields))
        self.fields = fields


@dataclass(frozen=True)
class CalibrationIdentity:
    """Immutable fit provenance; omitted and explicitly null settings differ.

    ``parameters_json`` is canonical JSON containing only generation settings,
    including the configured top-logprobs count; a per-response k is never bound.
    Credential references, resolved secrets and transport retries are excluded.
    An intervention ID names the pipeline/template used on both fit corpora.
    ``estimate_source`` and ``corpus_sha256`` bind the evidence source and the
    calibration corpus (v3); at their defaults they are omitted from ``to_dict``.
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
    #: Evidence source of the fit: an ``EstimateSource`` value.
    estimate_source: str = EstimateSource.TOP_K.value
    #: SHA-256 hex digest of the calibration corpus, or None when not recorded.
    corpus_sha256: str | None = None

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
        try:
            object.__setattr__(self, "estimate_source", EstimateSource(self.estimate_source).value)
        except (TypeError, ValueError):
            raise ValueError("estimate_source must be top_k or full_distribution") from None
        if self.corpus_sha256 is not None and (
                not isinstance(self.corpus_sha256, str) or not _SHA256_HEX.fullmatch(self.corpus_sha256)):
            raise ValueError("corpus_sha256 must be a lowercase SHA-256 hex digest or null")

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
        if self.estimate_source == EstimateSource.TOP_K.value:
            result.pop("estimate_source")
        if self.corpus_sha256 is None:
            result.pop("corpus_sha256")
        return result

    @classmethod
    def from_dict(cls, data: Mapping) -> CalibrationIdentity:
        """Load v1 (legacy), v2 and v3 payloads; absent keys take their defaults.

        An absent mapping means generic, an absent source ``top_k`` and an
        absent corpus hash null.
        """
        required = {"protocol", "endpoint", "model", "parameters_json", "intervention_id"}
        optional = {"local_identity", "response_mapping", "mapping_revision", "estimate_source", "corpus_sha256"}
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

    def mismatches(self, other: CalibrationIdentity) -> tuple[str, ...]:
        """Name the fields in which ``other`` differs, in declaration order.

        ``settings`` stands for ``parameters_json``. An empty tuple means the
        identities are equal.
        """
        if not isinstance(other, CalibrationIdentity):
            raise TypeError("other must be a CalibrationIdentity")
        return tuple(_MISMATCH_NAMES.get(field.name, field.name) for field in fields(self)
                     if getattr(self, field.name) != getattr(other, field.name))

    def require_match(self, other: CalibrationIdentity) -> None:
        """Refuse evidence of another identity, naming every differing field.

        Raises
        ------
        CalibrationMismatchError
            If any field differs; ``fields`` lists them as :meth:`mismatches` does.
        """
        differing = self.mismatches(other)
        if differing:
            raise CalibrationMismatchError(differing)

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


def _plain_settings(value: ParameterValue) -> ParameterValue:
    """Detached JSON-compatible containers: mappings become dicts, sequences lists."""
    if isinstance(value, Mapping):
        return {name: _plain_settings(child) for name, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_settings(child) for child in value]
    return value


def identity_settings(parameters: Mapping[str, ParameterValue]) -> str:
    """Canonical JSON of generation settings for identity comparison, without seed keys.

    ``seed`` and ``random_seed`` (:data:`SEED_SETTINGS`) are left out: a seed
    selects one sample, it does not change the distribution a calibrator was
    fitted on. The encoding is the one ``parameters_json`` uses (sorted keys,
    compact separators), so settings without a seed key give the same text.
    """
    if not isinstance(parameters, Mapping):
        raise TypeError("Generation parameters must be a mapping")
    kept = {name: _plain_settings(value) for name, value in parameters.items() if name not in SEED_SETTINGS}
    return json.dumps(kept, sort_keys=True, separators=(",", ":"))


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
    requests may finish. Ordinary timeouts/transport failures remain skippable,
    and so do per-item evidence defects (:data:`ITEM_EVIDENCE_CODES`), raised as
    :class:`ItemEvidenceError` after identity checks pass. A client exposing
    ``reject_item_evidence(code)`` is told which received row was excluded.
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
        # A client carries no estimate source or corpus; both are the calibrator's own.
        fit = {"estimate_source": expected.estimate_source, "corpus_sha256": expected.corpus_sha256}
        current = replace(CalibrationIdentity.from_client(client, intervention_id), **fit)
        if (current.response_mapping, current.mapping_revision) != (expected.response_mapping, expected.mapping_revision):
            raise ValueError("Client response mapping differs from calibration; calibrators are not reused "
                             "across evidence profiles")
        if current != expected:
            raise ValueError("Client model/endpoint/protocol/settings/intervention identity differs from calibration")
        config = replace(client.config, parameters=dict(client.config.parameters) | parameters)
        mapping = {"response_mapping": current.response_mapping, "mapping_revision": current.mapping_revision}
        if replace(CalibrationIdentity.from_config(config, intervention_id, **mapping), **fit) != expected:
            raise ValueError("Client model/endpoint/protocol/settings/intervention identity differs from calibration")

    def validate_response(self, response: ModelResponse) -> CompletionResult:
        if not isinstance(response, ModelResponse):
            raise ValueError("Configured MIA scoring requires a ModelResponse with genuine token evidence")
        settings = ModelConfig(self.identity.protocol, self.identity.endpoint, self.identity.model,
                               parameters=response.requested_settings, local_identity=response.local_identity)
        # The adapter admits top-k token evidence only (to_legacy), so the response's
        # source is top_k; the corpus hash is calibrator provenance.
        candidate = replace(
            CalibrationIdentity.from_config(settings, self.identity.intervention_id,
                                            response_mapping=self.identity.response_mapping,
                                            mapping_revision=self.identity.mapping_revision),
            corpus_sha256=self.identity.corpus_sha256)
        differing = replace(self.identity, local_identity=None).mismatches(replace(candidate, local_identity=None))
        if "estimate_source" in differing:
            raise CalibrationMismatchError(differing)
        if differing:
            raise ValueError("Response requested settings differ from calibration identity")
        local = self.identity.local_identity is not None or response.local_identity is not None
        if local:
            self._validate_local_response(response)
        with self._lock:
            expected = dict(response.requested_settings) | self._confirmed_settings
            for name, value in response.confirmed_settings.items():
                if name in expected and value != expected[name]:
                    raise ValueError(f"Response confirmed setting {name!r} differs from calibration settings")
        # Every identity/settings check precedes evidence admission, so an item-level
        # evidence defect (LocalEvidenceError) never masks drift.
        if local:
            validate_local_evidence(response, required_alternatives=1)
        result = response.to_legacy()
        with self._lock:
            if self._fitting:
                self._confirmed_settings.update(response.confirmed_settings)
        return result

    def _validate_local_response(self, response: ModelResponse) -> None:
        local = response.local_identity
        if local is None:
            raise ValueError("Response lost local calibration identity")
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
        except LocalEvidenceError as error:
            if str(error) not in ITEM_EVIDENCE_CODES:
                self._record_fatal(error)
                raise
            # Excluded and counted by the caller; identity checks already passed.
            reject = getattr(self.client, "reject_item_evidence", None)
            if callable(reject):
                reject(str(error))
            raise ItemEvidenceError(str(error)) from error
        except Exception as error:
            if getattr(error, "code", None) == "missing_token_evidence":
                error = ValueError("MIA/MCS genuine token evidence is unavailable or incomplete")
            if (isinstance(error, (BudgetExhaustedError, ValueError)) or self._is_auth_error(error)
                    or failure_kind(error) in FATAL_FAILURES):
                self._record_fatal(error)
            raise error

    def _record_fatal(self, error: Exception) -> None:
        with self._lock:
            if self._fatal is None:
                self._fatal = error
