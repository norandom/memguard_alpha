"""Immutable, JSON-safe experiment inputs and partial execution snapshots.

Draw grain is one item/evidence arm/zero-based draw. Draw 0 is predesignated as
the single-response condition; all draws also feed that arm's ensemble. Request
records wrap the physical ledger, whose call IDs are local to an execution.
This module performs no inference, credential resolution, scoring or file I/O.
"""

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum
from types import MappingProxyType
from uuid import uuid4

from recall_guard.core.completion import (
    ModelConfig,
    ModelResponse,
    ParameterValue,
)
from recall_guard.core.consensus import Tail
from recall_guard.core.ensemble import EnsembleSpec, LocationMode, MultimodalAction, ReferenceMode
from recall_guard.core.readiness_evidence import PartialResponse, ProbeEvidence
from recall_guard.core.readiness_types import _fields
from recall_guard.core.request_control import RequestEvent
from recall_guard.harness._execution_metadata import check_execution_metadata
from recall_guard.harness.workflow_plan import WorkflowPlan, canonical_json, contract_hash

SCHEMA_VERSION = 1


def _text(value, name, *, empty=False):
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ValueError(f"{name} must be a string" + ("" if empty else " with nonempty content"))


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")


def _digest(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _freeze(value: ParameterValue) -> ParameterValue:
    if isinstance(value, Mapping):
        for key in value:
            _text(key, "JSON object key")
        return MappingProxyType({key: _freeze(child) for key, child in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(child) for child in value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("JSON numbers must be finite")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError("Values must be JSON-compatible")


def _plain(value):
    if isinstance(value, Mapping):
        return {key: _plain(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_plain(child) for child in value]
    return value


def _mapping(value, name):
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return _freeze(value)


def text_hash(text: str) -> str:
    """Hash exact UTF-8 text; whitespace and Unicode are not normalized."""
    _text(text, "text", empty=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _identity(data: dict) -> str:
    return text_hash(json.dumps(data, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")))


def _version(value):
    if type(value) is not int or value not in {1, 2}:
        raise ValueError("Unsupported experiment schema_version")


def _model_from_dict(data, *, schema_version=1):
    if "credential_ref" in data:
        raise ValueError("Persisted model provenance must exclude credential_ref")
    if schema_version == 1 and "readiness" in data:
        raise ValueError("Readiness requires schema version 2")
    return ModelConfig.from_dict(data)


def _response_from_dict(data):
    return None if data is None else ModelResponse.from_dict(data)


class RecordStatus(StrEnum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


class EvidenceArm(StrEnum):
    ORIGINAL = "original"
    REWRITTEN = "rewritten"


@dataclass(frozen=True)
class ExperimentProtocol:
    """Predeclared experiment settings; identity excludes credential references.

    Cohort definitions retain matching rules and any sourced exact-model cutoff
    metadata. Audit selection retains deterministic selection method/seed,
    sample sizes and strata; selected items/judgments live in rewrite checks.
    Reduction settings must not override the shared target settings.
    ``utility_tolerance`` is an absolute accuracy difference in probability
    units (0.01 means one percentage point), not a relative percent change.
    """

    target_model: ModelConfig
    rewriter_model: ModelConfig
    verifier_model: ModelConfig
    data_hashes: Mapping[str, str]
    cohort_definitions: Mapping[str, ParameterValue]
    draws: int
    bootstrap_seed: int
    bootstrap_count: int
    task_metric: str
    utility_tolerance: float
    audit_selection: Mapping[str, ParameterValue]
    request_budget: int
    rewrite_method_version: str
    ensemble_settings: Mapping[str, ParameterValue] = field(default_factory=dict)
    rewrite_scope: str = "shared_per_item"
    schema_version: int = SCHEMA_VERSION
    workflow_plan: WorkflowPlan | None = None
    preparation: "PreparationBinding | None" = None

    def __post_init__(self):
        _version(self.schema_version)
        for name in ("target_model", "rewriter_model", "verifier_model"):
            if not isinstance(getattr(self, name), ModelConfig):
                raise TypeError(f"{name} must be ModelConfig")
        _integer(self.draws, "draws", 1)
        _integer(self.bootstrap_seed, "bootstrap_seed")
        _integer(self.bootstrap_count, "bootstrap_count", 1)
        _integer(self.request_budget, "request_budget")
        _number(self.utility_tolerance, "utility_tolerance")
        _text(self.task_metric, "task_metric")
        _text(self.rewrite_method_version, "rewrite_method_version")
        if self.rewrite_scope != "shared_per_item":
            raise ValueError("rewrite_scope must be shared_per_item")
        for name in ("data_hashes", "cohort_definitions", "audit_selection", "ensemble_settings"):
            object.__setattr__(self, name, _mapping(getattr(self, name), name))
        if not self.data_hashes or not self.cohort_definitions or not self.audit_selection:
            raise ValueError("data_hashes, cohort_definitions and audit_selection must be predeclared")
        for digest in self.data_hashes.values():
            _digest(digest, "data hash")
        self._validate_preparation()
        check_execution_metadata(self.to_dict())

    def _validate_preparation(self):
        if self.schema_version == 1:
            if self.workflow_plan is not None or self.preparation is not None or any(
                config.readiness is not None for config in (self.target_model, self.rewriter_model, self.verifier_model)
            ):
                raise ValueError("Readiness and plans require schema version 2")
            return
        if (self.workflow_plan is None) != (self.preparation is None):
            raise ValueError("A frozen plan requires its preparation binding")
        if self.workflow_plan is None:
            return
        if not isinstance(self.workflow_plan, WorkflowPlan) or not isinstance(self.preparation, PreparationBinding):
            raise TypeError("workflow_plan and preparation require typed contracts")
        selected = self.workflow_plan.selected
        if set(selected.roles) != {"target", "rewriter", "verifier"} or not selected.features.rewriting:
            raise ValueError("Paired experiment plan requires all three roles and selected rewriting")
        if selected.features.ensemble_draws != self.draws or selected.ensemble != self.build_ensemble_spec():
            raise ValueError("Paired experiment plan must match protocol draw count and ensemble settings")
        for role, config in selected.roles.items():
            if canonical_json(config.to_dict()) != canonical_json(getattr(self, role + "_model").to_dict()):
                raise ValueError("Selected role configuration differs from protocol")
        if self.preparation.plan_id != self.workflow_plan.plan_id:
            raise ValueError("Preparation binding identifies a different plan")
        if self.preparation.total_request_cap != self.request_budget:
            raise ValueError("Preparation total cap differs from protocol request_budget")

    @property
    def protocol_id(self) -> str:
        return _identity(self.to_dict())

    def build_ensemble_spec(self) -> EnsembleSpec:
        """Validate executable options and restore enum types without calls.

        Stored settings remain unchanged for hashing and historical inspection.
        Only this construction boundary translates their JSON representation;
        core EnsembleSpec retains its existing Python and reduction contracts.
        Explicit generation controls must equal the shared target settings.
        """
        settings = dict(self.ensemble_settings)
        if "draws" in settings:
            _integer(settings["draws"], "ensemble draws", 1)
            if settings.pop("draws") != self.draws:
                raise ValueError("Ensemble draw count differs from protocol draws")
        integer_fields = {"max_workers", "min_parsed", "trough_steps", "min_cluster_draws",
                          "max_total_requests", "max_tokens"}
        number_fields = {"grid", "confidence", "agreement_target", "trim", "mass_min", "density_ratio",
                         "min_cluster_density", "max_transport_failure_ratio", "temperature"}
        nullable_fields = {"grid", "agreement_target", "max_total_requests", "max_tokens", "temperature"}
        enum_fields = {"tail": Tail, "location_mode": LocationMode,
                       "multimodal_action": MultimodalAction, "reference_mode": ReferenceMode}
        unknown = settings.keys() - integer_fields - number_fields - enum_fields.keys() - {"retain_draws"}
        if unknown:
            raise ValueError(f"Unknown ensemble settings: {', '.join(sorted(unknown))}")
        for name, value in settings.items():
            if name in enum_fields:
                try:
                    settings[name] = enum_fields[name](value)
                except (TypeError, ValueError):
                    raise ValueError(f"Invalid ensemble {name}: {value!r}") from None
            elif name == "retain_draws":
                if type(value) is not bool:
                    raise ValueError("ensemble retain_draws must be boolean")
            elif value is None and name in nullable_fields:
                continue
            elif name in integer_fields:
                _integer(value, "ensemble " + name)
            else:
                _number(value, "ensemble " + name)
        spec = EnsembleSpec(draws=self.draws, **settings)
        for name in ("max_tokens", "temperature"):
            value = getattr(spec, name)
            if value is not None and (name not in self.target_model.parameters or value != self.target_model.parameters[name]):
                raise ValueError("Ensemble generation controls must match the shared target model settings")
        return spec

    def to_dict(self) -> dict:
        result = {
            "schema_version": self.schema_version,
            "target_model": self.target_model.to_dict(), "rewriter_model": self.rewriter_model.to_dict(),
            "verifier_model": self.verifier_model.to_dict(), "data_hashes": _plain(self.data_hashes),
            "cohort_definitions": _plain(self.cohort_definitions), "draws": self.draws,
            "bootstrap_seed": self.bootstrap_seed, "bootstrap_count": self.bootstrap_count,
            "task_metric": self.task_metric, "utility_tolerance": self.utility_tolerance,
            "audit_selection": _plain(self.audit_selection), "request_budget": self.request_budget,
            "rewrite_method_version": self.rewrite_method_version, "ensemble_settings": _plain(self.ensemble_settings),
            "rewrite_scope": self.rewrite_scope,
        }
        if self.schema_version == 2:
            result["workflow_plan"] = None if self.workflow_plan is None else self.workflow_plan.to_dict()
            result["preparation"] = None if self.preparation is None else self.preparation.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Mapping) -> "ExperimentProtocol":
        values = _fields(data, {"target_model", "rewriter_model", "verifier_model", "data_hashes", "cohort_definitions",
                               "draws", "bootstrap_seed", "bootstrap_count", "task_metric", "utility_tolerance",
                               "audit_selection", "request_budget", "rewrite_method_version", "ensemble_settings",
                               "rewrite_scope", "schema_version", "workflow_plan", "preparation"}, required={"schema_version"})
        _version(values["schema_version"])
        if values["schema_version"] == 1 and {"workflow_plan", "preparation", "fallback_policy", "readiness"} & values.keys():
            raise ValueError("Readiness and plans require schema version 2")
        for name in ("target_model", "rewriter_model", "verifier_model"):
            values[name] = _model_from_dict(values[name], schema_version=values["schema_version"])
        if values.get("workflow_plan") is not None:
            values["workflow_plan"] = WorkflowPlan.from_dict(values["workflow_plan"])
        if values.get("preparation") is not None:
            values["preparation"] = PreparationBinding.from_dict(values["preparation"])
        return cls(**values)


@dataclass(frozen=True)
class ExperimentItem:
    """One paired evaluation unit; the label is never part of rewrite input."""

    item_id: str
    instructions: str
    evidence: str
    as_of: str
    label: ParameterValue
    cohort: str

    def __post_init__(self):
        for name in ("item_id", "instructions", "evidence", "as_of", "cohort"):
            _text(getattr(self, name), name)
        if date.fromisoformat(self.as_of).isoformat() != self.as_of:
            raise ValueError("as_of must be an ISO YYYY-MM-DD date")
        object.__setattr__(self, "label", _freeze(self.label))

    @property
    def input_hash(self) -> str:
        return _identity(self.to_dict())

    def to_dict(self) -> dict:
        return {"item_id": self.item_id, "instructions": self.instructions, "evidence": self.evidence,
                "as_of": self.as_of, "label": _plain(self.label), "cohort": self.cohort}

    @classmethod
    def from_dict(cls, data: Mapping) -> "ExperimentItem":
        return cls(**data)


def _outcome(status, error, response=None):
    status = RecordStatus(status)
    if error is not None:
        _text(error, "error")
    if status == RecordStatus.FAILED and error is None:
        raise ValueError("failed records require an error")
    if status != RecordStatus.FAILED and error is not None:
        raise ValueError("Only failed records may contain an error")
    if response is not None and not isinstance(response, ModelResponse):
        raise TypeError("response must be ModelResponse or None")
    if status == RecordStatus.PENDING and response is not None:
        raise ValueError("pending records cannot contain a response")
    return status


@dataclass(frozen=True)
class DrawRecord:
    """One logical target draw, including its exact prompt and partial output.

    Optional JSON metadata retains actual projection/scoring outcomes and
    diagnostics without live scorer/client state. ``attempted`` there denotes
    a logical workflow invocation, distinct from physical RequestRecords.
    """

    item_id: str
    arm: EvidenceArm
    draw_index: int
    status: RecordStatus = RecordStatus.PENDING
    prompt: str | None = None
    response: ModelResponse | None = None
    stage: str = "target"
    error: str | None = None
    metadata: Mapping[str, ParameterValue] = field(default_factory=dict)
    partial_response: PartialResponse | None = None

    def __post_init__(self):
        _text(self.item_id, "item_id")
        _integer(self.draw_index, "draw_index")
        _text(self.stage, "stage")
        if self.prompt is not None:
            _text(self.prompt, "prompt", empty=True)
        object.__setattr__(self, "arm", EvidenceArm(self.arm))
        object.__setattr__(self, "status", _outcome(self.status, self.error, self.response))
        object.__setattr__(self, "metadata", _mapping(self.metadata, "draw metadata"))
        if self.partial_response is not None and (
            not isinstance(self.partial_response, PartialResponse) or self.status != RecordStatus.FAILED or self.response is not None
        ):
            raise ValueError("Partial evidence belongs only to failed draws without a complete response")
        if self.status == RecordStatus.COMPLETED and (self.response is None or self.prompt is None):
            raise ValueError("completed draws require prompt and response")

    @property
    def identity(self) -> tuple[str, EvidenceArm, int]:
        return self.item_id, self.arm, self.draw_index

    @property
    def prompt_hash(self) -> str | None:
        return None if self.prompt is None else text_hash(self.prompt)

    def to_dict(self) -> dict:
        result = {"item_id": self.item_id, "arm": self.arm.value, "draw_index": self.draw_index,
                "status": self.status.value, "prompt": self.prompt,
                "response": None if self.response is None else self.response.to_dict(),
                "stage": self.stage, "error": self.error, "metadata": _plain(self.metadata)}
        if self.partial_response is not None:
            result["partial_response"] = self.partial_response.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Mapping) -> "DrawRecord":
        values = _fields(data, {"item_id", "arm", "draw_index", "status", "prompt", "response", "stage", "error",
                               "metadata", "partial_response"})
        values["response"] = _response_from_dict(values["response"])
        if values.get("partial_response") is not None:
            values["partial_response"] = PartialResponse.from_dict(values["partial_response"])
        return cls(**values)


@dataclass(frozen=True)
class RewriteRecord:
    """One shared rewrite and fidelity evidence, never a semantic guarantee.

    ``completed`` means an accepted rewrite; rejection or malformed verification
    is ``failed`` with stage/error retained. Checks may retain audit judgments.
    """

    item_id: str
    original_evidence: str
    status: RecordStatus = RecordStatus.PENDING
    rewritten_evidence: str | None = None
    rewrite_prompt: str | None = None
    verification_prompt: str | None = None
    rewrite_response: ModelResponse | None = None
    verification_response: ModelResponse | None = None
    fidelity_checks: Mapping[str, ParameterValue] = field(default_factory=dict)
    stage: str = "rewrite"
    error: str | None = None

    def __post_init__(self):
        _text(self.item_id, "item_id")
        _text(self.original_evidence, "original_evidence")
        _text(self.stage, "stage")
        object.__setattr__(self, "status", _outcome(self.status, self.error))
        for name in ("rewritten_evidence", "rewrite_prompt", "verification_prompt"):
            if getattr(self, name) is not None:
                _text(getattr(self, name), name, empty=True)
        for response in (self.rewrite_response, self.verification_response):
            if response is not None and not isinstance(response, ModelResponse):
                raise TypeError("rewrite and verification responses must be ModelResponse or None")
        if self.status == RecordStatus.COMPLETED and not self.rewritten_evidence:
            raise ValueError("completed rewrites require rewritten_evidence")
        object.__setattr__(self, "fidelity_checks", _mapping(self.fidelity_checks, "fidelity_checks"))

    def to_dict(self) -> dict:
        return {"item_id": self.item_id, "original_evidence": self.original_evidence, "status": self.status.value,
                "rewritten_evidence": self.rewritten_evidence, "rewrite_prompt": self.rewrite_prompt,
                "verification_prompt": self.verification_prompt,
                "rewrite_response": None if self.rewrite_response is None else self.rewrite_response.to_dict(),
                "verification_response": None if self.verification_response is None else self.verification_response.to_dict(),
                "fidelity_checks": _plain(self.fidelity_checks), "stage": self.stage, "error": self.error}

    @classmethod
    def from_dict(cls, data: Mapping) -> "RewriteRecord":
        return cls(**(dict(data) | {name: _response_from_dict(data[name])
                                   for name in ("rewrite_response", "verification_response")}))


@dataclass(frozen=True)
class RequestRecord:
    """Logical association around an unchanged physical ``RequestEvent``.

    ``execution_id`` namespaces budget-local call IDs after restart. Run-level
    probes/calibration may omit item identity; item target draws require arm and
    index. Hashes bind exact prompt text retained by stage records. Diagnostics
    must already be sanitized at the request-control boundary.
    """

    execution_id: str
    event: RequestEvent
    model_role: str
    prompt_hash: str
    item_id: str | None = None
    arm: EvidenceArm | None = None
    draw_index: int | None = None
    phase: str | None = None

    def __post_init__(self):
        _text(self.execution_id, "execution_id")
        if self.phase is not None and self.phase not in {"preparation", "evaluation"}:
            raise ValueError("phase must be preparation or evaluation")
        if not isinstance(self.event, RequestEvent):
            raise TypeError("event must be RequestEvent")
        if self.model_role not in {"target", "rewriter", "verifier"}:
            raise ValueError("model_role must be target, rewriter or verifier")
        _digest(self.prompt_hash, "prompt_hash")
        if self.item_id is not None:
            _text(self.item_id, "item_id")
        if self.arm is not None:
            object.__setattr__(self, "arm", EvidenceArm(self.arm))
        if self.draw_index is not None:
            _integer(self.draw_index, "draw_index")
        if self.model_role == "target" and self.item_id is not None:
            if self.arm is None or self.draw_index is None:
                raise ValueError("Item target requests require arm and draw_index")
        elif self.arm is not None or self.draw_index is not None:
            raise ValueError("Only item target requests may contain arm and draw_index")
        event = self.event
        _integer(event.call_id, "call_id", 1)
        _integer(event.attempt, "attempt", 1)
        _number(event.started_at, "started_at")
        _text(event.stage, "request stage")
        _text(event.model, "request model")
        timestamp = datetime.fromisoformat(event.timestamp)
        if timestamp.tzinfo is None:
            raise ValueError("Request timestamp must include a timezone")
        if event.status not in {"started", "succeeded", "failed"}:
            raise ValueError("Unknown RequestEvent status")
        if event.latency_s is not None:
            _number(event.latency_s, "latency_s")
        if event.cost is not None:
            _number(event.cost, "cost")
        if event.status_code is not None:
            _integer(event.status_code, "status_code", 100)
        for name in ("error", "parameter", "detail", "code"):
            if getattr(event, name) is not None:
                _text(getattr(event, name), name, empty=True)

    @property
    def identity(self) -> tuple[str, int] | tuple[str, str, int]:
        if self.phase is None:
            return self.execution_id, self.event.call_id
        return self.phase, self.execution_id, self.event.call_id

    def to_dict(self) -> dict:
        result = {"execution_id": self.execution_id, "event": self.event.to_dict(), "model_role": self.model_role,
                "prompt_hash": self.prompt_hash, "item_id": self.item_id,
                "arm": None if self.arm is None else self.arm.value, "draw_index": self.draw_index}
        if self.phase is not None:
            result["phase"] = self.phase
        return result

    @classmethod
    def from_dict(cls, data: Mapping) -> "RequestRecord":
        values = _fields(data, {"execution_id", "event", "model_role", "prompt_hash", "item_id", "arm", "draw_index", "phase"})
        values["event"] = RequestEvent.from_dict(values["event"])
        return cls(**values)


@dataclass(frozen=True)
class PreparationBinding:
    """Portable immutable-artifact hashes and physical-request allowance.

    The protocol's request_budget is the original *combined* cap. Evaluation
    receives only remaining_evaluation_requests; historical preparation calls
    are never copied into a fresh client's probe cache or local budget ledger.
    Hashes cover canonical UTF-8 contract JSON, not filesystem paths or clients.
    """

    execution_id: str
    plan_id: str
    artifact_hash: str
    ledger_hash: str
    total_request_cap: int
    preparation_used_requests: int
    remaining_evaluation_requests: int

    def __post_init__(self):
        _text(self.execution_id, "preparation execution_id")
        for name in ("plan_id", "artifact_hash", "ledger_hash"):
            _digest(getattr(self, name), name)
        for name in ("total_request_cap", "preparation_used_requests", "remaining_evaluation_requests"):
            _integer(getattr(self, name), name)
        if self.preparation_used_requests + self.remaining_evaluation_requests != self.total_request_cap:
            raise ValueError("Preparation and evaluation allowances must reconcile with the total cap")

    def validate_artifact(self, artifact):
        if not isinstance(artifact, PreparationArtifact):
            raise TypeError("artifact must be PreparationArtifact")
        if self != artifact.bind(total_request_cap=self.total_request_cap):
            raise ValueError("Preparation artifact differs from the frozen binding")

    def to_dict(self):
        return {"execution_id": self.execution_id, "plan_id": self.plan_id, "artifact_hash": self.artifact_hash,
                "ledger_hash": self.ledger_hash, "total_request_cap": self.total_request_cap,
                "preparation_used_requests": self.preparation_used_requests,
                "remaining_evaluation_requests": self.remaining_evaluation_requests}

    @classmethod
    def from_dict(cls, data):
        return cls(**_fields(data, {"execution_id", "plan_id", "artifact_hash", "ledger_hash", "total_request_cap",
                                   "preparation_used_requests", "remaining_evaluation_requests"}))


@dataclass(frozen=True)
class PreparationArtifact:
    """Historical preparation records, at physical-request and role/probe grain.

    Failed/deferred and partial probes remain evidence; retaining this object
    never authorizes execution. A later runtime preparation gate owns readiness.
    No prepared workflow, calibration fit, callback or credential is serialized.
    """

    execution_id: str
    plan: WorkflowPlan
    requests: tuple[RequestRecord, ...]
    evidence: Mapping[str, tuple[ProbeEvidence, ...]]
    stopping_reason: str | None = None
    schema_version: int = 2

    def __post_init__(self):
        if type(self.schema_version) is not int or self.schema_version != 2:
            raise ValueError("Preparation artifacts require schema version 2")
        _text(self.execution_id, "preparation execution_id")
        if not isinstance(self.plan, WorkflowPlan):
            raise TypeError("plan must be WorkflowPlan")
        if self.stopping_reason is not None:
            _text(self.stopping_reason, "stopping_reason")
        requests = tuple(self.requests)
        for request in requests:
            if not isinstance(request, RequestRecord):
                raise TypeError("requests must contain RequestRecord")
            if request.phase != "preparation" or request.execution_id != self.execution_id or request.item_id is not None:
                raise ValueError("Preparation requests must use their own phase/execution namespace without evaluation items")
            if request.model_role not in self.plan.requested.roles or request.event.model != self.plan.requested.roles[request.model_role].model:
                raise ValueError("Preparation request model/role differs from plan")
        if sorted(request.event.call_id for request in requests) != list(range(1, len(requests) + 1)):
            raise ValueError("Preparation ledger requires unique contiguous physical call IDs")
        object.__setattr__(self, "requests", requests)
        by_call_id = {request.event.call_id: request for request in requests}
        if not isinstance(self.evidence, Mapping) or self.evidence.keys() - self.plan.requested.roles.keys():
            raise ValueError("Preparation evidence must use configured roles")
        evidence = {}
        for role, probes in self.evidence.items():
            probes = tuple(probes)
            for probe in probes:
                if not isinstance(probe, ProbeEvidence):
                    raise TypeError("evidence must contain ProbeEvidence values")
                configs = (self.plan.requested.roles[role], self.plan.selected.roles[role])
                if canonical_json(probe.config.to_dict()) not in {canonical_json(config.to_dict()) for config in configs}:
                    raise ValueError("Probe configuration differs from requested/selected role")
                for observation in probe.observations:
                    if observation.call_id is not None:
                        request = by_call_id.get(observation.call_id)
                        if request is None or request.model_role != role:
                            raise ValueError("Probe observation references an unknown physical role request")
                        operation_id = request.event.operation_id
                        if operation_id is not None and operation_id != observation.operation_id:
                            raise ValueError("Probe operation differs from its physical request")
            evidence[role] = probes
        object.__setattr__(self, "evidence", MappingProxyType(evidence))
        check_execution_metadata(self.to_dict())

    @property
    def ledger_hash(self):
        return contract_hash([request.to_dict() for request in self.requests])

    @property
    def artifact_hash(self):
        return contract_hash(self.to_dict())

    def bind(self, *, total_request_cap):
        _integer(total_request_cap, "total_request_cap")
        if len(self.requests) > total_request_cap:
            raise ValueError("Preparation ledger exceeds total request cap")
        return PreparationBinding(self.execution_id, self.plan.plan_id, self.artifact_hash, self.ledger_hash,
                                  total_request_cap, len(self.requests), total_request_cap - len(self.requests))

    def to_dict(self):
        return {"schema_version": self.schema_version, "execution_id": self.execution_id, "plan": self.plan.to_dict(),
                "requests": [request.to_dict() for request in self.requests],
                "evidence": {role: [probe.to_dict() for probe in probes] for role, probes in self.evidence.items()},
                "stopping_reason": self.stopping_reason}

    @classmethod
    def from_dict(cls, data):
        values = _fields(data, {"schema_version", "execution_id", "plan", "requests", "evidence", "stopping_reason"},
                         required={"schema_version", "execution_id", "plan", "requests", "evidence"})
        values["plan"] = WorkflowPlan.from_dict(values["plan"])
        values["requests"] = tuple(RequestRecord.from_dict(request) for request in values["requests"])
        values["evidence"] = {role: tuple(ProbeEvidence.from_dict(probe) for probe in probes)
                              for role, probes in values["evidence"].items()}
        return cls(**values)


@dataclass(frozen=True)
class ResumeState:
    """Construct CallBudget(remaining_requests), namespaced by execution_id.

    Keep the prior ledger in the checkpoint; never reconstruct a full-limit
    budget at restart or merge unnamespaced budget-local call IDs.
    """

    prior_used_requests: int
    remaining_requests: int
    execution_id: str


@dataclass(frozen=True)
class ExperimentCheckpoint:
    """Inspect partial state without calls; explicitly validate before restart.

    ``pending`` marks unscheduled/unfinished work, ``failed`` a stopped run, and
    ``completed`` all planned records terminal (some may have failed). Ordered
    items freeze counterbalancing. Logical failures without physical requests
    are valid, e.g. fidelity rejection or budget admission refusal.
    """

    protocol: ExperimentProtocol
    items: tuple[ExperimentItem, ...]
    status: RecordStatus = RecordStatus.PENDING
    draws: tuple[DrawRecord, ...] = ()
    rewrites: tuple[RewriteRecord, ...] = ()
    requests: tuple[RequestRecord, ...] = ()
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self):
        _version(self.schema_version)
        if not isinstance(self.protocol, ExperimentProtocol):
            raise TypeError("protocol must be ExperimentProtocol")
        if self.schema_version != self.protocol.schema_version:
            raise ValueError("Checkpoint and protocol schema versions must match")
        object.__setattr__(self, "status", RecordStatus(self.status))
        for name, kind in (("items", ExperimentItem), ("draws", DrawRecord),
                           ("rewrites", RewriteRecord), ("requests", RequestRecord)):
            values = tuple(getattr(self, name))
            if not all(isinstance(value, kind) for value in values):
                raise TypeError(f"{name} must contain {kind.__name__} values")
            object.__setattr__(self, name, values)
        self._validate_records()
        if self.schema_version == 1:
            _legacy_checkpoint_fields(self.to_dict())

    def _validate_records(self):
        items = {item.item_id: item for item in self.items}
        draws = {draw.identity: draw for draw in self.draws}
        rewrites = {rewrite.item_id: rewrite for rewrite in self.rewrites}
        requests = {request.identity: request for request in self.requests}
        if (len(items) != len(self.items) or len(draws) != len(self.draws)
                or len(rewrites) != len(self.rewrites) or len(requests) != len(self.requests)):
            raise ValueError("Duplicate item, draw, rewrite or physical request identity")
        allowance = (self.protocol.preparation.remaining_evaluation_requests if self.protocol.preparation is not None
                     else self.protocol.request_budget)
        if len(self.requests) > allowance:
            raise ValueError("Physical request ledger exceeds protocol request_budget")
        if any(item.cohort not in self.protocol.cohort_definitions for item in self.items):
            raise ValueError("Item cohort is absent from protocol cohort_definitions")
        executions = {}
        for draw in self.draws:
            if draw.item_id not in items or draw.draw_index >= self.protocol.draws:
                raise ValueError("Draw references an unknown item or out-of-range draw_index")
            if draw.arm == EvidenceArm.REWRITTEN and draw.response is not None:
                rewrite = rewrites.get(draw.item_id)
                if rewrite is None or rewrite.status != RecordStatus.COMPLETED:
                    raise ValueError("Rewritten target output requires an accepted completed rewrite")
        for rewrite in self.rewrites:
            if rewrite.item_id not in items or rewrite.original_evidence != items[rewrite.item_id].evidence:
                raise ValueError("Rewrite original_evidence does not match its item")
        for request in self.requests:
            if self.schema_version == 2 and (request.phase != "evaluation" or (
                self.protocol.preparation is not None and request.execution_id == self.protocol.preparation.execution_id
            )):
                raise ValueError("Evaluation requests require a separate phase/execution namespace")
            executions.setdefault(request.execution_id, []).append(request.event.call_id)
            config = getattr(self.protocol, request.model_role + "_model")
            if request.event.model != config.model:
                raise ValueError("Request model differs from protocol model identity")
            if request.item_id is None:
                continue
            if request.item_id not in items:
                raise ValueError("Request references unknown item")
            if request.model_role == "target":
                if request.draw_index >= self.protocol.draws:
                    raise ValueError("Request draw_index is outside the protocol")
                draw = draws.get((request.item_id, request.arm, request.draw_index))
                if draw is not None and draw.prompt_hash is not None and draw.prompt_hash != request.prompt_hash:
                    raise ValueError("Request prompt_hash differs from saved draw prompt")
            elif rewrite := rewrites.get(request.item_id):
                prompt = rewrite.rewrite_prompt if request.model_role == "rewriter" else rewrite.verification_prompt
                if prompt is not None and text_hash(prompt) != request.prompt_hash:
                    raise ValueError("Request prompt_hash differs from saved rewrite stage prompt")
        for call_ids in executions.values():
            if sorted(call_ids) != list(range(1, len(call_ids) + 1)):
                raise ValueError("Physical request ledger must preserve contiguous call IDs within each execution")
        if self.status == RecordStatus.COMPLETED:
            planned = {(item_id, arm, index) for item_id in items for arm in EvidenceArm
                       for index in range(self.protocol.draws)}
            if (set(draws) != planned or set(rewrites) != set(items)
                    or any(record.status == RecordStatus.PENDING for record in (*self.draws, *self.rewrites))
                    or any(request.event.status == "started" for request in self.requests)):
                raise ValueError("completed checkpoints require all planned arm/draw and rewrite records terminal")

    def validate_restart(self, protocol: ExperimentProtocol, items: Sequence[ExperimentItem]) -> ResumeState:
        """Reject incompatible or ambiguous state before constructing clients.

        A terminal request with a pending/missing logical result is ambiguous:
        provider work may have completed without its output durably saved.
        Reconcile that record explicitly instead of issuing an automatic copy.
        """
        if self.schema_version == 2:
            raise ValueError("Version-2 runtime resume requires preparation-aware accounting")
        if protocol.protocol_id != self.protocol.protocol_id:
            raise ValueError("Restart protocol differs from persisted protocol")
        if tuple(item.input_hash for item in items) != tuple(item.input_hash for item in self.items):
            raise ValueError("Restart inputs differ from persisted ordered inputs")
        draws = {draw.identity: draw for draw in self.draws}
        rewrites = {rewrite.item_id: rewrite for rewrite in self.rewrites}
        successful_draws = set()
        successful_rewrite_stages = set()
        for request in self.requests:
            if request.event.status == "started":
                raise ValueError("Ambiguous in-flight physical request prevents automatic restart")
            if request.item_id is None:
                continue
            record = (draws.get((request.item_id, request.arm, request.draw_index))
                      if request.model_role == "target" else rewrites.get(request.item_id))
            if record is None or record.status == RecordStatus.PENDING:
                raise ValueError("Ambiguous physical request without a terminal logical result prevents restart")
            if request.event.status == "succeeded":
                if request.model_role == "target":
                    successful_draws.add(record.identity)
                else:
                    successful_rewrite_stages.add((request.item_id, request.model_role))
        for draw in self.draws:
            if draw.response is not None and draw.identity not in successful_draws:
                raise ValueError("Ambiguous target output without a successful physical request in the ledger")
        for rewrite in self.rewrites:
            if rewrite.status == RecordStatus.PENDING and any(value is not None for value in (
                rewrite.rewritten_evidence, rewrite.rewrite_response, rewrite.verification_response,
            )):
                raise ValueError("Ambiguous pending rewrite contains partial output without a completion marker")
            for role, response in (("rewriter", rewrite.rewrite_response), ("verifier", rewrite.verification_response)):
                if response is not None and (rewrite.item_id, role) not in successful_rewrite_stages:
                    raise ValueError("Ambiguous rewrite output without a successful physical request in the ledger")
        execution_id = uuid4().hex
        while execution_id in {request.execution_id for request in self.requests}:
            execution_id = uuid4().hex
        return ResumeState(len(self.requests), self.protocol.request_budget - len(self.requests), execution_id)

    def to_dict(self) -> dict:
        return {"schema_version": self.schema_version, "protocol": self.protocol.to_dict(),
                "protocol_id": self.protocol.protocol_id, "status": self.status.value,
                "items": [item.to_dict() for item in self.items],
                "input_hashes": [item.input_hash for item in self.items],
                "draws": [draw.to_dict() for draw in self.draws],
                "rewrites": [rewrite.to_dict() for rewrite in self.rewrites],
                "requests": [request.to_dict() for request in self.requests]}

    @classmethod
    def from_dict(cls, data: Mapping) -> "ExperimentCheckpoint":
        values = _fields(data, {"schema_version", "protocol", "protocol_id", "status", "items", "input_hashes",
                               "draws", "rewrites", "requests"}, required={"schema_version"})
        _version(values["schema_version"])
        if values["schema_version"] == 1:
            _legacy_checkpoint_fields(values)
        protocol_id = values.pop("protocol_id")
        input_hashes = values.pop("input_hashes")
        values["protocol"] = ExperimentProtocol.from_dict(values["protocol"])
        for name, kind in (("items", ExperimentItem), ("draws", DrawRecord),
                           ("rewrites", RewriteRecord), ("requests", RequestRecord)):
            values[name] = tuple(kind.from_dict(entry) for entry in values[name])
        checkpoint = cls(**values)
        if checkpoint.protocol.protocol_id != protocol_id:
            raise ValueError("Persisted protocol hash does not match protocol")
        if [item.input_hash for item in checkpoint.items] != list(input_hashes):
            raise ValueError("Persisted input hashes do not match inputs")
        return checkpoint


def _legacy_checkpoint_fields(data):
    """Reject v2 fields only in schema-owned slots, preserving arbitrary task data."""
    responses = []
    for draw in data["draws"]:
        if "partial_response" in draw:
            raise ValueError("Partial response evidence requires schema version 2")
        responses.append(draw.get("response"))
    for rewrite in data["rewrites"]:
        responses.extend((rewrite.get("rewrite_response"), rewrite.get("verification_response")))
    if any(response is not None and "readiness_trace" in response for response in responses):
        raise ValueError("Readiness trace requires schema version 2")
    for request in data["requests"]:
        if "phase" in request or {"purpose", "operation_id", "request_id"} & request["event"].keys():
            raise ValueError("Readiness request fields require schema version 2")
