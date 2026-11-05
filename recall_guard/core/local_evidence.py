"""Pure local evidence and provenance; no host or accelerator dependencies.

Token evidence is ordered by generated output position, never prompt position.
Log probabilities use natural-log units. Alternative lists remain partial top-k
observations; they are never normalized or presented as full distributions.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from recall_guard.core.completion import ModelResponse

class LocalEvidenceError(ValueError):
    """Local admission failure; messages are stable, non-content reason codes."""


@dataclass(frozen=True)
class ScoreSemantics:
    """Nullable score provenance, distinct from the partial top-k transport.

    ``normalization`` describes the source distribution (for example
    ``full_vocabulary_log_softmax``); ``log_base`` is ``e`` for natural logs.
    ``sampling_stage`` distinguishes ``base_model`` from
    ``sampling_transformed``. Null means unknown, never base-model evidence.
    """

    normalization: str | None = None
    log_base: str | None = None
    sampling_stage: str | None = None

    def __post_init__(self) -> None:
        _nullable_strings(asdict(self))
        if self.sampling_stage not in {None, "base_model", "sampling_transformed"}:
            raise ValueError("unknown sampling_stage")


def _nullable_strings(values: dict) -> None:
    for name, value in values.items():
        if value is not None and (not isinstance(value, str) or not value.strip()):
            raise ValueError(f"{name} must be a nonempty string or null")


@dataclass(frozen=True)
class LocalModelProvenance:
    """One declared or observed snapshot; absent details remain explicitly null."""

    artifact_digest: str | None = None
    artifact_revision: str | None = None
    quantization: str | None = None
    tokenizer: str | None = None
    template: str | None = None
    runtime: str | None = None
    runtime_version: str | None = None
    backend: str | None = None
    score_semantics: ScoreSemantics = field(default_factory=ScoreSemantics)

    def __post_init__(self) -> None:
        if not isinstance(self.score_semantics, ScoreSemantics):
            raise TypeError("score_semantics must be ScoreSemantics")
        _nullable_strings({key: value for key, value in asdict(self).items() if key != "score_semantics"})

    @classmethod
    def from_dict(cls, data: Mapping) -> LocalModelProvenance:
        if not isinstance(data, Mapping):
            raise ValueError("local provenance must be an object")
        values = dict(data)
        if "score_semantics" in values:
            if not isinstance(values["score_semantics"], Mapping):
                raise ValueError("score_semantics must be an object")
            values["score_semantics"] = ScoreSemantics(**values["score_semantics"])
        return cls(**values)


@dataclass(frozen=True)
class LocalModelIdentity:
    """Version-1 ``recall_guard_local`` extension and canonical identity.

    Declared backend selection does not establish observed execution placement.
    Equal unknowns do not establish cross-run calibration compatibility.
    """

    declared: LocalModelProvenance = field(default_factory=LocalModelProvenance)
    observed: LocalModelProvenance = field(default_factory=LocalModelProvenance)

    def __post_init__(self) -> None:
        if not all(isinstance(value, LocalModelProvenance) for value in (self.declared, self.observed)):
            raise TypeError("declared and observed must be LocalModelProvenance")

    def to_dict(self) -> dict:
        return {"version": 1, **asdict(self)}

    @classmethod
    def from_dict(cls, data: Mapping) -> LocalModelIdentity:
        if not isinstance(data, Mapping) or set(data) != {"version", "declared", "observed"}:
            raise ValueError("invalid local identity extension fields")
        if type(data["version"]) is not int or data["version"] != 1:
            raise ValueError("unsupported local identity extension version")
        return cls(*(LocalModelProvenance.from_dict(data[name]) for name in ("declared", "observed")))

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)

    @property
    def identity_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


def validate_local_evidence(response: ModelResponse, *, required_alternatives: int = 0) -> None:
    """Require exact output text alignment and uncensored natural-log evidence.

    The current contract has token text only. Hidden/reasoning/special channels
    without an explicit mapping cannot be reconstructed and fail alignment.
    No stripping, retokenization or inferred channel mapping is performed.
    ``required_alternatives`` is the selected feature's minimum per position.
    """
    if type(required_alternatives) is not int or required_alternatives < 0:
        raise ValueError("required_alternatives must be a nonnegative integer")
    if not response.content:
        raise LocalEvidenceError("empty_output")
    if not response.token_evidence:
        raise LocalEvidenceError("missing_token_evidence")
    if "".join(token.token for token in response.token_evidence) != response.content:
        raise LocalEvidenceError("token_text_alignment_failed")
    for token in response.token_evidence:
        if not token.token:
            raise LocalEvidenceError("empty_token_alignment_unsupported")
        if len(token.top_logprobs) < required_alternatives:
            raise LocalEvidenceError("missing_alternatives")
        if len({alt.token for alt in token.top_logprobs}) != len(token.top_logprobs):
            raise LocalEvidenceError("duplicate_alternatives")
        for entry in (token, *token.top_logprobs):
            if entry.logprob == -9999:
                raise LocalEvidenceError("censored_logprob")
            if (isinstance(entry.logprob, bool) or not isinstance(entry.logprob, (int, float))
                    or not math.isfinite(entry.logprob) or entry.logprob > 0):
                raise LocalEvidenceError("invalid_logprob")


def full_vocabulary_logprobs(logits, *, vocabulary_size: int) -> np.ndarray:
    """Stable log-softmax of one prefix's complete ``[vocabulary]`` vector.

    The caller must establish completeness against the model vocabulary size,
    capture scores for the prefix that selected this generated token, and record
    normalization/log-base/sampling stage in provenance. This helper cannot
    establish those facts. Batched ``[batch, position, vocabulary]`` arrays and
    pruned/top-k vectors are rejected. Return values are natural-log units.
    """
    if type(vocabulary_size) is not int or vocabulary_size < 1:
        raise LocalEvidenceError("invalid_vocabulary_size")
    scores = np.asarray(logits, dtype=np.float64)
    if scores.shape != (vocabulary_size,):
        raise LocalEvidenceError("full_vocabulary_vector_required")
    if not np.all(np.isfinite(scores)):
        raise LocalEvidenceError("invalid_logits")
    shifted_scores = scores - np.max(scores)
    return shifted_scores - np.log(np.exp(shifted_scores).sum())
