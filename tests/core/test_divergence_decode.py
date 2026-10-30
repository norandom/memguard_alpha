"""Callback metadata and request validation for divergence decoding."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from recall_guard.core.divergence_decode import (
    DecodeError,
    DecodeResult,
    ModelSource,
    _validate_decode_request,
)


def make_source(callback=None):
    return ModelSource(
        next_logits=callback or (lambda prefix_token_ids: [0.0, 1.0, 2.0]),
        vocabulary_id="shared-vocabulary-v1",
        vocabulary_size=3,
        eos_token_ids=frozenset({2}),
        context_limit=4,
    )


def validate_request(prompt_token_ids=(0,), **overrides):
    sources = {"main_source": make_source(), "older_source": make_source(), "newer_source": make_source()}
    options = {**sources, "alpha": 0.5, "max_new_tokens": 2, **overrides}
    return _validate_decode_request(prompt_token_ids, **options)


def test_accepts_exact_context_boundary_without_invoking_callbacks():
    calls = []
    source = make_source(lambda prefix_token_ids: calls.append(prefix_token_ids))
    prompt_token_ids = [np.int64(0), 1]
    validated_prompt = validate_request(
        prompt_token_ids, main_source=source, older_source=source, newer_source=source,
    )
    assert validated_prompt == (0, 1)
    assert isinstance(validated_prompt, tuple)
    assert calls == []
    assert prompt_token_ids == [0, 1]


def test_empty_eos_set_allows_budget_only_requests():
    source = replace(make_source(), eos_token_ids=frozenset())
    assert validate_request(main_source=source, older_source=source, newer_source=source) == (0,)


@pytest.mark.parametrize("role", ["main_source", "older_source", "newer_source"])
@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("next_logits", None),
        ("vocabulary_id", ""),
        ("vocabulary_id", 5),
        ("vocabulary_size", 0),
        ("vocabulary_size", True),
        ("vocabulary_size", 3.0),
        ("context_limit", 0),
        ("context_limit", False),
        ("context_limit", 4.0),
        ("eos_token_ids", frozenset({-1})),
        ("eos_token_ids", frozenset({3})),
        ("eos_token_ids", frozenset({True})),
        ("eos_token_ids", frozenset({1.0})),
        ("eos_token_ids", {2}),
    ],
)
def test_rejects_invalid_source_metadata_before_callback(role, field, invalid_value):
    calls = []
    source = make_source(lambda prefix_token_ids: calls.append(prefix_token_ids))
    sources = {name: source for name in ("main_source", "older_source", "newer_source")}
    sources[role] = replace(source, **{field: invalid_value})
    with pytest.raises((TypeError, ValueError), match=f"{role}.*{field}"):
        validate_request(**sources)
    assert calls == []


@pytest.mark.parametrize("role", ["main_source", "older_source", "newer_source"])
def test_rejects_non_source_objects_with_role_context(role):
    with pytest.raises(TypeError, match=role):
        validate_request(**{role: object()})


@pytest.mark.parametrize(
    ("field", "incompatible_value"),
    [("vocabulary_id", "another-mapping"), ("vocabulary_size", 4), ("eos_token_ids", frozenset({1}))],
)
def test_rejects_incompatible_declared_sources(field, incompatible_value):
    with pytest.raises(ValueError, match=field):
        validate_request(older_source=replace(make_source(), **{field: incompatible_value}))


@pytest.mark.parametrize("prompt_token_ids", [[], [-1], [3], [True], [np.bool_(False)], [1.0], ["1"], None])
def test_rejects_invalid_prompt_before_callback(prompt_token_ids):
    calls = []
    source = make_source(lambda prefix_token_ids: calls.append(prefix_token_ids))
    with pytest.raises((TypeError, ValueError), match="prompt_token_ids"):
        validate_request(prompt_token_ids, main_source=source, older_source=source, newer_source=source)
    assert calls == []


@pytest.mark.parametrize("max_new_tokens", [0, -1, True, np.bool_(True), 1.0])
def test_rejects_invalid_output_budget(max_new_tokens):
    with pytest.raises((TypeError, ValueError), match="max_new_tokens"):
        validate_request(max_new_tokens=max_new_tokens)


@pytest.mark.parametrize("alpha", [-1, np.nan, np.inf, True, np.bool_(False), "1", 1j])
def test_rejects_invalid_alpha_before_callback(alpha):
    calls = []
    source = make_source(lambda prefix_token_ids: calls.append(prefix_token_ids))
    with pytest.raises((TypeError, ValueError), match="alpha"):
        validate_request(alpha=alpha, main_source=source, older_source=source, newer_source=source)
    assert calls == []


def test_rejects_request_exceeding_any_context_limit_without_callbacks():
    calls = []
    source = make_source(lambda prefix_token_ids: calls.append(prefix_token_ids))
    with pytest.raises(ValueError, match="context_limit"):
        validate_request(
            (0, 1), main_source=source, older_source=source,
            newer_source=replace(source, context_limit=3),
        )
    assert calls == []


def test_numpy_maximum_budget_cannot_wrap_context_check():
    calls = []
    source = make_source(lambda prefix_token_ids: calls.append(prefix_token_ids))
    with pytest.raises(ValueError, match="context_limit"):
        validate_request(
            max_new_tokens=np.int64(np.iinfo(np.int64).max),
            main_source=source, older_source=source, newer_source=source,
        )
    assert calls == []


def test_source_and_result_values_are_frozen():
    with pytest.raises(FrozenInstanceError):
        make_source().vocabulary_size = 4
    result = DecodeResult(token_ids=(1, 2), finish_reason="eos")
    with pytest.raises(FrozenInstanceError):
        result.finish_reason = "length"


def test_decoder_error_carries_role_and_zero_based_step():
    error = DecodeError("callback failed", role="older", step=0)
    assert error.role == "older"
    assert error.step == 0
    assert "callback failed" in str(error)
