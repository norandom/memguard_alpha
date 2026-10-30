"""Correctness and validation of vocabulary-indexed divergence logits."""

import numpy as np
import pytest

from recall_guard.core.divergence import combine_logits


def test_combines_hand_calculated_logits_without_mutating_inputs():
    main_logits = np.array([1.0, -2.0, 0.0])
    older_logits = np.array([3.0, 2.0, -1.0])
    newer_logits = np.array([1.0, 4.0, 2.0])
    original_logits = [logits.copy() for logits in (main_logits, older_logits, newer_logits)]

    adjusted_logits = combine_logits(main_logits, older_logits, newer_logits, alpha=0.5)

    np.testing.assert_array_equal(adjusted_logits, [2.0, -3.0, -1.5])
    assert adjusted_logits.dtype == np.float64
    for logits, original in zip((main_logits, older_logits, newer_logits), original_logits, strict=True):
        np.testing.assert_array_equal(logits, original)


def test_accepts_real_integer_sequences_and_numpy_scalar_alpha():
    adjusted_logits = combine_logits([1, 2], [2, 1], [1, 2], alpha=np.float32(0.25))
    np.testing.assert_array_equal(adjusted_logits, [1.25, 1.75])
    assert adjusted_logits.dtype == np.float64


def test_zero_alpha_returns_independent_main_copy_without_auxiliary_subtraction():
    main_logits = np.array([1.0, -0.0])
    older_logits = np.array([1e308, -1e308])
    newer_logits = -older_logits

    with np.errstate(over="raise", invalid="raise"):
        adjusted_logits = combine_logits(main_logits, older_logits, newer_logits, alpha=0.0)

    np.testing.assert_array_equal(adjusted_logits, main_logits)
    np.testing.assert_array_equal(np.signbit(adjusted_logits), np.signbit(main_logits))
    assert not np.shares_memory(adjusted_logits, main_logits)
    adjusted_logits[0] = 99.0
    assert main_logits[0] == 1.0


@pytest.mark.parametrize("role", ["main_logits", "older_logits", "newer_logits"])
@pytest.mark.parametrize(
    "invalid_logits",
    [[], 1.0, [[1.0, 2.0]], [np.nan, 2.0], [np.inf, 2.0], [-np.inf, 2.0],
     [True, False], [1j, 2j], ["1", "2"], np.array([1.0, 2.0], dtype=object)],
)
def test_rejects_invalid_vectors_with_role_context(role, invalid_logits):
    model_logits = {"main_logits": [1.0, 2.0], "older_logits": [1.0, 2.0], "newer_logits": [1.0, 2.0]}
    model_logits[role] = invalid_logits
    with pytest.raises((TypeError, ValueError), match=role):
        combine_logits(**model_logits, alpha=0.0)


@pytest.mark.parametrize("role", ["older_logits", "newer_logits"])
def test_rejects_mismatched_vocabulary_lengths(role):
    model_logits = {"main_logits": [1.0, 2.0], "older_logits": [1.0, 2.0], "newer_logits": [1.0, 2.0]}
    model_logits[role] = [1.0]
    with pytest.raises(ValueError, match=role):
        combine_logits(**model_logits, alpha=0.5)


@pytest.mark.parametrize("alpha", [-1.0, np.nan, np.inf, -np.inf, True, np.bool_(False), "1", 1j, [1.0]])
def test_rejects_invalid_alpha(alpha):
    with pytest.raises((TypeError, ValueError), match="alpha"):
        combine_logits([1.0], [1.0], [1.0], alpha=alpha)


@pytest.mark.parametrize(
    ("main_logits", "older_logits", "newer_logits", "alpha"),
    [([0.0], [1e308], [-1e308], 1.0), ([1e308], [1e308], [0.0], 2.0)],
)
def test_rejects_overflowed_combination(main_logits, older_logits, newer_logits, alpha):
    with pytest.raises(ValueError, match="adjusted_logits"):
        combine_logits(main_logits, older_logits, newer_logits, alpha=alpha)
