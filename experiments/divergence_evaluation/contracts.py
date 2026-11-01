"""Versioned portable records. No model runtime imports or artifact access."""

import json
import math
import re
from dataclasses import asdict, dataclass, fields
from enum import StrEnum
from types import UnionType
from typing import Self, get_args, get_origin, get_type_hints

SCHEMA_VERSION = 1
MAIN_MODEL_ID = "openai/gpt-oss-20b"
MCS_UNAVAILABLE = "Condition-matched independent calibration and valid token evidence unavailable"


class Phase(StrEnum):
    FITTING = "fitting"
    RUNTIME = "runtime"
    SCREENING = "screening"
    DEVELOPMENT = "development"
    PARITY = "parity"
    CONTROLS = "controls"
    EVALUATION = "evaluation"


class Cohort(StrEnum):
    HISTORICAL = "historical"
    POST_CUTOFF = "post_cutoff"


class Split(StrEnum):
    FITTING = "fitting"
    SCREENING = "screening"
    DEVELOPMENT = "development"
    EVALUATION = "evaluation"


class Arm(StrEnum):
    BASELINE = "baseline"
    ADJUSTED = "adjusted"
    SHAM = "sham"
    CONTROL = "control"


class OperationStatus(StrEnum):
    INTENT = "intent"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INCOMPLETE = "incomplete"
    UNPERFORMED = "unperformed"


def _number(name: str, value: float, *, integer: bool = False, minimum: float = 0) -> None:
    valid_type = type(value) is int if integer else type(value) in (int, float)
    if not valid_type or not math.isfinite(value) or value < minimum:
        raise ValueError(f"{name} must be a finite {'integer' if integer else 'number'} >= {minimum}")


def _hash(name: str, value: str | None, length: int = 64) -> None:
    if value is not None and (not isinstance(value, str) or not re.fullmatch(f"[0-9a-f]{{{length}}}", value)):
        raise ValueError(f"{name} must be an immutable {length}-character lowercase hexadecimal digest")


def _decode(annotation, value):
    if value is None and type(None) in get_args(annotation):
        return None
    if get_origin(annotation) is UnionType:
        return _decode(next(arg for arg in get_args(annotation) if arg is not type(None)), value)
    if get_origin(annotation) is tuple:
        args = get_args(annotation)
        if not isinstance(value, (list, tuple)):
            raise ValueError("expected JSON array")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_decode(args[0], item) for item in value)
        if len(args) != len(value):
            raise ValueError("incorrect tuple length")
        return tuple(_decode(arg, item) for arg, item in zip(args, value, strict=True))
    if isinstance(annotation, type) and issubclass(annotation, JsonContract):
        if isinstance(value, annotation):
            return value
        return annotation.from_dict(value)
    if isinstance(annotation, type) and issubclass(annotation, StrEnum):
        return annotation(value)
    if annotation is float and type(value) in (int, float):
        return value
    if annotation in (str, int, bool, float) and type(value) is not annotation:
        raise ValueError(f"expected {annotation.__name__}")
    return value


class JsonContract:
    """Strict fields and schema versions; immutable nested tuples survive JSON replay."""

    def __post_init__(self):
        for field in fields(self):
            if field.name == "schema_version" and (
                type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION
            ):
                raise ValueError("unsupported schema_version")
        # Constructors receive the same type checks as JSON input, without coercion.
        for name, annotation in get_type_hints(type(self)).items():
            value = getattr(self, name)
            decoded = _decode(annotation, value)
            if type(decoded) is not type(value):
                raise ValueError(f"{name} has incorrect contract type")

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, allow_nan=False)

    @classmethod
    def from_dict(cls, payload: dict) -> Self:
        if not isinstance(payload, dict):
            raise ValueError("contract must be a JSON object")
        annotations = get_type_hints(cls)
        if set(payload) - set(annotations):
            raise ValueError("unknown fields in contract")
        try:
            return cls(**{name: _decode(annotations[name], value) for name, value in payload.items()})
        except TypeError as error:
            raise ValueError("missing or invalid contract fields") from error

    @classmethod
    def from_json(cls, text: str) -> Self:
        return cls.from_dict(json.loads(text))


@dataclass(frozen=True)
class GenerationSettings(JsonContract):
    max_context_tokens: int = 4096
    max_output_tokens: int = 1024
    reasoning_effort: str = "low"
    decoding: str = "greedy"

    def __post_init__(self):
        super().__post_init__()
        _number("max_context_tokens", self.max_context_tokens, integer=True, minimum=1)
        _number("max_output_tokens", self.max_output_tokens, integer=True, minimum=1)
        if self.max_output_tokens > self.max_context_tokens:
            raise ValueError("max_output_tokens exceeds total context tokens")
        if self.decoding != "greedy" or self.reasoning_effort != "low":
            raise ValueError("experiment requires greedy decoding and low reasoning effort")


@dataclass(frozen=True)
class FittingLimits(JsonContract):
    max_documents: int
    max_tokens: int
    max_memory_bytes: int

    def __post_init__(self):
        super().__post_init__()
        for field in fields(self):
            _number(field.name, getattr(self, field.name), integer=True, minimum=1)


@dataclass(frozen=True)
class PhaseBudget(JsonContract):
    phase: Phase
    max_operations: int
    max_generated_tokens: int
    wall_time_seconds: float
    peak_host_memory_bytes: int
    peak_device_memory_bytes: int
    gpu_hours: float
    monetary_cap_usd: float | None = None
    compute_rate_usd_per_gpu_hour: float | None = None

    def __post_init__(self):
        super().__post_init__()
        for name in ("max_operations", "max_generated_tokens", "peak_host_memory_bytes", "peak_device_memory_bytes"):
            _number(name, getattr(self, name), integer=True)
        for name in ("wall_time_seconds", "gpu_hours", "monetary_cap_usd", "compute_rate_usd_per_gpu_hour"):
            if (value := getattr(self, name)) is not None:
                _number(name, value)
        if self.wall_time_seconds == 0:
            raise ValueError("wall_time_seconds must be positive")
        if (self.monetary_cap_usd is None) != (self.compute_rate_usd_per_gpu_hour is None):
            raise ValueError("paid compute requires both monetary cap and recorded rate")


@dataclass(frozen=True)
class CorpusDocument(JsonContract):
    document_id: str
    source_id: str
    content_hash: str
    rights_reference: str
    published_at: str
    revised_at: str
    cohort: Cohort
    split: Split
    fact_exposure_ids: tuple[str, ...] = ()
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self):
        super().__post_init__()
        _hash("content_hash", self.content_hash)


@dataclass(frozen=True)
class AuxiliaryManifest(JsonContract):
    role: str
    tokenizer_mapping_hash: str
    source_hashes: tuple[str, ...]
    # Row grain: source ID, exposure window, document count, token count.
    counts_by_source_window: tuple[tuple[str, str, int, int], ...]
    fitting_limits: FittingLimits
    estimator_version: str = "local-sparse-trigram-v1"
    smoothing_beta: float = 1.0
    document_resets: bool = True
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self):
        super().__post_init__()
        if self.role not in ("older", "newer", "matched_size_control"):
            raise ValueError("unknown auxiliary role")
        if self.estimator_version != "local-sparse-trigram-v1" or self.smoothing_beta != 1 or not self.document_resets:
            raise ValueError("unsupported locally fitted trigram policy")
        _hash("tokenizer_mapping_hash", self.tokenizer_mapping_hash)
        for source_hash in self.source_hashes:
            _hash("source_hash", source_hash)
        for _, _, documents, tokens in self.counts_by_source_window:
            _number("document_count", documents, integer=True)
            _number("token_count", tokens, integer=True)


@dataclass(frozen=True)
class EvaluationPlan(JsonContract):
    main_model_id: str = MAIN_MODEL_ID
    main_revision: str | None = None
    tokenizer_revision: str | None = None
    tokenizer_mapping_hash: str | None = None
    harmony_template_hash: str | None = None
    image_digest: str | None = None
    software_revision: str | None = None
    precision: str | None = None
    hardware: str | None = None
    runtime_identity: str | None = None
    parser_version: str | None = None
    corpus_hashes: tuple[str, ...] = ()
    auxiliary_artifact_hashes: tuple[str, ...] = ()
    split_manifest_hash: str | None = None
    dataset_document_count: int | None = None
    dataset_token_count: int | None = None
    fitting_limits: FittingLimits | None = None
    generation: GenerationSettings = GenerationSettings()
    phase_budgets: tuple[PhaseBudget, ...] = ()
    execution_phase: Phase | None = None
    alpha_grid: tuple[float, ...] = (0.25, 0.5, 1.0)
    alpha_selection_rule: str = "smallest-recall-reducing-alpha-with-utility-loss-at-most-0.05"
    screening_criteria: str | None = None
    worker_grace_seconds: float | None = None
    memory_margin_bytes: int | None = None
    condition: str = "self_hosted"
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self):
        super().__post_init__()
        if self.main_model_id != MAIN_MODEL_ID:
            raise ValueError(f"main_model_id must be {MAIN_MODEL_ID}")
        if self.condition != "self_hosted":
            raise ValueError("this plan describes self_hosted conditions")
        for name in ("main_revision", "tokenizer_revision", "software_revision"):
            _hash(name, getattr(self, name), 40)
        for name in ("tokenizer_mapping_hash", "harmony_template_hash", "split_manifest_hash"):
            _hash(name, getattr(self, name))
        if self.image_digest is not None:
            if not self.image_digest.startswith("sha256:"):
                raise ValueError("image_digest requires sha256 digest")
            _hash("image_digest", self.image_digest[7:])
        for digest in (*self.corpus_hashes, *self.auxiliary_artifact_hashes):
            _hash("artifact_hash", digest)
        for name in ("dataset_document_count", "dataset_token_count", "memory_margin_bytes"):
            if (value := getattr(self, name)) is not None:
                _number(name, value, integer=True, minimum=1)
        if self.worker_grace_seconds is not None:
            _number("worker_grace_seconds", self.worker_grace_seconds, minimum=0.001)
        if self.alpha_grid != (0.25, 0.5, 1.0):
            raise ValueError("alpha_grid must be the frozen development grid (0.25, 0.5, 1.0)")
        phases = [budget.phase for budget in self.phase_budgets]
        if len(phases) != len(set(phases)):
            raise ValueError("duplicate phase budgets")
        for budget in self.phase_budgets:
            if budget.phase != Phase.FITTING and (
                budget.max_generated_tokens < budget.max_operations * self.generation.max_output_tokens
            ):
                raise ValueError("phase generated-token budget cannot cover requested operations")


@dataclass(frozen=True)
class OperationRecord(JsonContract):
    operation_id: str
    phase: Phase
    item_id: str | None
    arm: Arm | None
    intent_at: str
    status: OperationStatus
    elapsed_seconds: float | None = None
    generated_tokens: int | None = None
    peak_host_memory_bytes: int | None = None
    peak_device_memory_bytes: int | None = None
    monetary_cost_usd: float | None = None
    cost_unknown_reason: str | None = "No billing evidence"
    failure_reason: str | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self):
        super().__post_init__()
        for name in ("elapsed_seconds", "generated_tokens", "peak_host_memory_bytes",
                     "peak_device_memory_bytes", "monetary_cost_usd"):
            if (value := getattr(self, name)) is not None:
                _number(name, value, integer=name.endswith(("tokens", "bytes")))


@dataclass(frozen=True)
class ItemResult(JsonContract):
    item_id: str
    cohort: Cohort
    arm: Arm
    answer: str | None
    parse_state: str
    correct: bool | None
    same_prefix_changed_tokens: int
    finish_reason: str
    stated_confidence: float | None = None
    generated_token_probabilities: tuple[float, ...] | None = None
    probability_distribution: str | None = None
    mcs: float | None = None
    mcs_unavailable_reason: str = MCS_UNAVAILABLE
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self):
        super().__post_init__()
        _number("same_prefix_changed_tokens", self.same_prefix_changed_tokens, integer=True)
        if self.mcs is not None:
            raise ValueError("MCS unavailable in the initial experiment")
        for probability in self.generated_token_probabilities or ():
            _number("generated_token_probability", probability)
            if probability > 1:
                raise ValueError("generated_token_probability exceeds 1")
        if self.generated_token_probabilities is not None and self.probability_distribution not in ("original", "adjusted"):
            raise ValueError("token evidence must identify original or adjusted distribution")
