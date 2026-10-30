"""Callback metadata and request validation for divergence decoding."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from recall_guard.core.divergence_decode import (
    DecodeError,
    DecodeResult,
    ModelSource,
    _validate_decode_request,
    decode_greedy,
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


def decode_with_sources(main_callback, older_callback=None, newer_callback=None, **overrides):
    sources = {
        "main_source": replace(make_source(main_callback), context_limit=20),
        "older_source": replace(make_source(older_callback or main_callback), context_limit=20),
        "newer_source": replace(make_source(newer_callback or main_callback), context_limit=20),
    }
    options = {**sources, "alpha": 0.5, "max_new_tokens": 3, **overrides}
    return decode_greedy((0,), **options)


def test_greedy_uses_adjustment_and_shares_every_immutable_prefix():
    calls = []

    def callback_for(role, first_logits):
        def next_logits(prefix_token_ids):
            assert isinstance(prefix_token_ids, tuple)
            calls.append((role, prefix_token_ids))
            return first_logits if len(prefix_token_ids) == 1 else [0.0, 0.0, 10.0]
        return next_logits

    result = decode_with_sources(
        callback_for("main", [2.0, 1.0, 0.0]),
        callback_for("older", [0.0, 4.0, 0.0]),
        callback_for("newer", [0.0, 0.0, 0.0]),
    )
    assert result == DecodeResult(token_ids=(1, 2), finish_reason="eos")
    assert calls == [(role, prefix) for prefix in ((0,), (0, 1)) for role in ("main", "older", "newer")]


def test_greedy_breaks_ties_by_lowest_token_id_and_stops_at_numpy_budget():
    result = decode_with_sources(lambda prefix_token_ids: [4.0, 4.0, 0.0], max_new_tokens=np.int64(2))
    assert result == DecodeResult(token_ids=(0, 0), finish_reason="length")


def test_greedy_includes_first_terminal_eos():
    result = decode_with_sources(lambda prefix_token_ids: [0.0, 0.0, 9.0])
    assert result == DecodeResult(token_ids=(2,), finish_reason="eos")


def test_greedy_runs_all_callbacks_at_zero_alpha():
    calls = []

    def callback_for(role):
        def next_logits(prefix_token_ids):
            calls.append(role)
            return [0.0, 2.0, 0.0]
        return next_logits

    result = decode_with_sources(*(callback_for(role) for role in ("main", "older", "newer")), alpha=0.0,
                                 max_new_tokens=1)
    assert result.token_ids == (1,)
    assert calls == ["main", "older", "newer"]


@pytest.mark.parametrize("failing_role", ["main", "older", "newer"])
def test_greedy_reports_callback_failure_and_calls_no_later_source(failing_role):
    calls = []
    original_error = RuntimeError("source unavailable")

    def callback_for(role):
        def next_logits(prefix_token_ids):
            calls.append(role)
            if role == failing_role:
                raise original_error
            return [0.0, 1.0, 0.0]
        return next_logits

    with pytest.raises(DecodeError) as caught:
        decode_with_sources(*(callback_for(role) for role in ("main", "older", "newer")))
    assert caught.value.role == failing_role
    assert caught.value.step == 0
    assert caught.value.__cause__ is original_error
    assert calls == ["main", "older", "newer"][:["main", "older", "newer"].index(failing_role) + 1]


@pytest.mark.parametrize("invalid_logits", [[1.0, 2.0], [[1.0, 2.0, 3.0]], [1.0, np.nan, 3.0],
                                           [True, False, True], [1j, 2j, 3j]])
@pytest.mark.parametrize("failing_role", ["main", "older", "newer"])
def test_greedy_reports_invalid_returned_logits_with_model_role(failing_role, invalid_logits):
    callbacks = {
        role: (lambda prefix_token_ids: [0.0, 1.0, 0.0])
        for role in ("main", "older", "newer")
    }
    callbacks[failing_role] = lambda prefix_token_ids: invalid_logits
    with pytest.raises(DecodeError) as caught:
        decode_with_sources(*(callbacks[role] for role in ("main", "older", "newer")))
    assert caught.value.role == failing_role
    assert caught.value.step == 0
    assert isinstance(caught.value.__cause__, (ValueError, TypeError))


def test_greedy_failure_after_one_token_has_zero_based_step():
    def next_logits(prefix_token_ids):
        if len(prefix_token_ids) > 1:
            raise OSError("second step failed")
        return [0.0, 1.0, 0.0]

    with pytest.raises(DecodeError) as caught:
        decode_with_sources(next_logits)
    assert caught.value.role == "main"
    assert caught.value.step == 1


def test_greedy_reports_combination_overflow():
    with pytest.raises(DecodeError) as caught:
        decode_with_sources(
            lambda prefix_token_ids: [0.0, 0.0, 0.0],
            lambda prefix_token_ids: [1e308, 0.0, 0.0],
            lambda prefix_token_ids: [-1e308, 0.0, 0.0],
        )
    assert caught.value.role == "combination"
    assert caught.value.step == 0
    assert isinstance(caught.value.__cause__, ValueError)


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
def test_greedy_propagates_process_interrupts(interrupt):
    def next_logits(prefix_token_ids):
        raise interrupt()

    with pytest.raises(interrupt):
        decode_with_sources(next_logits)


def test_greedy_preflights_before_any_callback():
    calls = []
    with pytest.raises(ValueError, match="context_limit"):
        decode_with_sources(lambda prefix_token_ids: calls.append(prefix_token_ids), max_new_tokens=20)
    assert calls == []
