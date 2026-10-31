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
    CapabilityEvidence,
    ModelConfig,
    ModelResponse,
    ParameterValue,
    TokenAlternative,
    TokenEvidence,
)
from recall_guard.core.request_control import RequestEvent

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
    if type(value) is not int or value != SCHEMA_VERSION:
        raise ValueError("Unsupported experiment schema_version")


def _model_from_dict(data):
    values = dict(data)
    if "credential_ref" in values:
        raise ValueError("Persisted model provenance must exclude credential_ref")
    values["capabilities"] = {name: CapabilityEvidence(**entry) for name, entry in values["capabilities"].items()}
    return ModelConfig(**values)


def _response_from_dict(data):
    if data is None:
        return None
    values = dict(data)
    if values["token_evidence"] is not None:
        values["token_evidence"] = tuple(TokenEvidence(
            token=entry["token"], logprob=entry["logprob"],
            top_logprobs=tuple(TokenAlternative(**alternative) for alternative in entry["top_logprobs"]),
        ) for entry in values["token_evidence"])
    return ModelResponse(**values)


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

    @property
    def protocol_id(self) -> str:
        return _identity(self.to_dict())

    def to_dict(self) -> dict:
        return {
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

    @classmethod
    def from_dict(cls, data: Mapping) -> "ExperimentProtocol":
        values = dict(data)
        _version(values["schema_version"])
        for name in ("target_model", "rewriter_model", "verifier_model"):
            values[name] = _model_from_dict(values[name])
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

    def __post_init__(self):
        _text(self.item_id, "item_id")
        _integer(self.draw_index, "draw_index")
        _text(self.stage, "stage")
        if self.prompt is not None:
            _text(self.prompt, "prompt", empty=True)
        object.__setattr__(self, "arm", EvidenceArm(self.arm))
        object.__setattr__(self, "status", _outcome(self.status, self.error, self.response))
        object.__setattr__(self, "metadata", _mapping(self.metadata, "draw metadata"))
        if self.status == RecordStatus.COMPLETED and (self.response is None or self.prompt is None):
            raise ValueError("completed draws require prompt and response")

    @property
    def identity(self) -> tuple[str, EvidenceArm, int]:
        return self.item_id, self.arm, self.draw_index

    @property
    def prompt_hash(self) -> str | None:
        return None if self.prompt is None else text_hash(self.prompt)

    def to_dict(self) -> dict:
        return {"item_id": self.item_id, "arm": self.arm.value, "draw_index": self.draw_index,
                "status": self.status.value, "prompt": self.prompt,
                "response": None if self.response is None else self.response.to_dict(),
                "stage": self.stage, "error": self.error, "metadata": _plain(self.metadata)}

    @classmethod
    def from_dict(cls, data: Mapping) -> "DrawRecord":
        return cls(**(dict(data) | {"response": _response_from_dict(data["response"])}))


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

    def __post_init__(self):
        _text(self.execution_id, "execution_id")
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
    def identity(self) -> tuple[str, int]:
        return self.execution_id, self.event.call_id

    def to_dict(self) -> dict:
        return {"execution_id": self.execution_id, "event": self.event.to_dict(), "model_role": self.model_role,
                "prompt_hash": self.prompt_hash, "item_id": self.item_id,
                "arm": None if self.arm is None else self.arm.value, "draw_index": self.draw_index}

    @classmethod
    def from_dict(cls, data: Mapping) -> "RequestRecord":
        return cls(**(dict(data) | {"event": RequestEvent(**data["event"])}))


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
        object.__setattr__(self, "status", RecordStatus(self.status))
        for name, kind in (("items", ExperimentItem), ("draws", DrawRecord),
                           ("rewrites", RewriteRecord), ("requests", RequestRecord)):
            values = tuple(getattr(self, name))
            if not all(isinstance(value, kind) for value in values):
                raise TypeError(f"{name} must contain {kind.__name__} values")
            object.__setattr__(self, name, values)
        self._validate_records()

    def _validate_records(self):
        items = {item.item_id: item for item in self.items}
        draws = {draw.identity: draw for draw in self.draws}
        rewrites = {rewrite.item_id: rewrite for rewrite in self.rewrites}
        requests = {request.identity: request for request in self.requests}
        if (len(items) != len(self.items) or len(draws) != len(self.draws)
                or len(rewrites) != len(self.rewrites) or len(requests) != len(self.requests)):
            raise ValueError("Duplicate item, draw, rewrite or physical request identity")
        if len(self.requests) > self.protocol.request_budget:
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
        values = dict(data)
        _version(values["schema_version"])
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
