"""Local admission preserves token positions and partial probability evidence."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from recall_guard.core.completion import ModelResponse, TokenAlternative, TokenEvidence
from recall_guard.core.local_evidence import (
    LocalEvidenceError,
    LocalModelIdentity,
    LocalModelProvenance,
    ScoreSemantics,
    full_vocabulary_logprobs,
    validate_local_evidence,
)


@pytest.mark.parametrize("context_length", [True, 0, -1, 1.5, "2048"])
def test_context_length_is_positive_token_count(context_length):
    with pytest.raises(ValueError, match="context_length"):
        LocalModelProvenance(context_length=context_length)


def test_sidecar_native_merge_preserves_known_fields_and_semantic_conflicts():
    from recall_guard.core.local_evidence import merge_local_observations

    sidecar = LocalModelProvenance(artifact_digest="a", score_semantics=ScoreSemantics(log_base="e"))
    native = LocalModelProvenance(backend="cpu", score_semantics=ScoreSemantics(sampling_stage="base_model"))
    merged, conflicts = merge_local_observations(sidecar, native)
    assert conflicts == ()
    assert merged.artifact_digest == "a" and merged.backend == "cpu"
    assert merged.template is None
    assert merged.score_semantics == ScoreSemantics(log_base="e", sampling_stage="base_model")
    _, conflicts = merge_local_observations(sidecar, replace(native, score_semantics=ScoreSemantics(log_base="2")))
    assert conflicts == ("score_semantics.log_base",)


def response(texts=(" café", "\n", "🙂")):
    return ModelResponse("".join(texts), tuple(TokenEvidence(text, -0.5, (
        TokenAlternative("a", -1.0), TokenAlternative("b", -2.0),
    )) for text in texts))


def test_exact_unicode_whitespace_and_partial_alternatives_are_preserved():
    value = response()
    before = value.to_dict()
    validate_local_evidence(value, required_alternatives=2)
    assert value.to_dict() == before
    with pytest.raises(LocalEvidenceError, match="missing_alternatives"):
        validate_local_evidence(value, required_alternatives=3)
    validate_local_evidence(ModelResponse("x", (TokenEvidence("x", -0.5),)))


@pytest.mark.parametrize(("value", "reason"), [
    (ModelResponse(""), "empty_output"),
    (ModelResponse("x"), "missing_token_evidence"),
    (replace(response(), content="café\n🙂"), "token_text_alignment_failed"),
    (replace(response(), content=" café\n🙂<reasoning>"), "token_text_alignment_failed"),
    (response(("", "x")), "empty_token_alignment_unsupported"),
    (ModelResponse("x", (TokenEvidence("x", -9999),)), "censored_logprob"),
    (ModelResponse("x", (TokenEvidence("x", -1, (TokenAlternative("a", -9999),)),)), "censored_logprob"),
    (ModelResponse("x", (TokenEvidence("x", -1, (TokenAlternative("a", -1), TokenAlternative("a", -2))),)), "duplicate_alternatives"),
])
def test_rejects_ineligible_evidence(value, reason):
    with pytest.raises(LocalEvidenceError, match=reason):
        validate_local_evidence(value)


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf"), 0.1, True, None])
def test_defensive_invalid_score_rejection(score):
    value = response(("x",))
    object.__setattr__(value.token_evidence[0], "logprob", score)
    with pytest.raises(LocalEvidenceError, match="invalid_logprob"):
        validate_local_evidence(value)


@pytest.mark.parametrize("count", [-1, True, 1.5])
def test_invalid_alternative_requirement(count):
    with pytest.raises(ValueError, match="required_alternatives"):
        validate_local_evidence(response(), required_alternatives=count)


def test_identity_roundtrip_unknowns_immutability_and_drift():
    declared = LocalModelProvenance(backend="npu", artifact_revision="rev1")
    identity = LocalModelIdentity(declared=declared)
    assert identity.observed.backend is None
    assert identity.to_dict()["observed"]["artifact_digest"] is None
    assert LocalModelIdentity.from_dict(identity.to_dict()) == identity
    assert LocalModelIdentity.from_dict(dict(reversed(list(identity.to_dict().items())))).identity_hash == identity.identity_hash
    with pytest.raises(FrozenInstanceError):
        declared.backend = "cpu"
    assert replace(identity, observed=declared).identity_hash != identity.identity_hash
    for field in ("artifact_digest", "artifact_revision", "quantization", "tokenizer", "template", "runtime", "runtime_version", "backend"):
        assert replace(identity, observed=LocalModelProvenance(**{field: "changed"})).identity_hash != identity.identity_hash
    semantics = ScoreSemantics("full_vocabulary_log_softmax", "e", "base_model")
    assert replace(identity, observed=LocalModelProvenance(score_semantics=semantics)).identity_hash != identity.identity_hash


@pytest.mark.parametrize("change", [
    {"version": 2}, {"version": True}, {"extra": "ignored?"}, {"observed": {"backend": 1}},
    {"observed": []}, {"observed": {"score_semantics": []}},
])
def test_identity_rejects_invalid_wire_data(change):
    with pytest.raises((ValueError, TypeError)):
        LocalModelIdentity.from_dict(LocalModelIdentity().to_dict() | change)


def test_full_vocabulary_normalization_is_stable_and_position_local():
    logits = np.array([1000.0, 1001.0, 999.0])
    logprobs = full_vocabulary_logprobs(logits, vocabulary_size=3)
    np.testing.assert_allclose(np.exp(logprobs).sum(), 1.0)
    np.testing.assert_allclose(logprobs[1] - logprobs[0], 1.0)
    np.testing.assert_array_equal(logits, [1000, 1001, 999])


@pytest.mark.parametrize(("scores", "size"), [([1, 2], 3), ([[1, 2]], 2), ([], 0), ([float("nan")], 1), ([float("inf")], 1), ([1], True)])
def test_partial_or_invalid_logits_are_not_normalized(scores, size):
    with pytest.raises(LocalEvidenceError):
        full_vocabulary_logprobs(scores, vocabulary_size=size)
