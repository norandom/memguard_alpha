"""Behavioral checks for the isolated dated-source signature probe."""

import numpy as np
import pytest

from scripts.probe_temporal_signature import extract_statement, fit_counts, sequence_score, words


def test_extracts_implicit_paragraph_end_without_navigation():
    html = (
        "<p>Navigation"
        "<p>Information received since yesterday. <strong>Keep this.</strong>"
        "<p>The Committee decided to maintain rates."
        "<p>Voting for the policy were A and B.</p><p>Footer</p>"
    )
    text = extract_statement(html)
    assert text.startswith("Information received")
    assert "Keep this." in text
    assert "Navigation" not in text
    assert "Footer" not in text


def test_rate_fractions_remain_distinct_tokens():
    assert words("at 2-1/4 to 2-1/2 percent") == ["at", "2-1/4", "to", "2-1/2", "percent"]
    assert words("at 0 to 1/4 percent") == ["at", "0", "to", "1/4", "percent"]


def test_fitting_does_not_create_transitions_between_documents():
    model = fit_counts([[0, 1, 0], [1]], "older", 3, "a" * 64)
    unigram = np.array([3, 3, 1]) / 7
    bigram = (np.array([1, 0, 0]) + unigram) / 2
    expected = (np.array([1, 0, 0]) + bigram) / 2
    np.testing.assert_allclose(np.exp(model.log_probabilities([0, 1])), expected)
    assert model.receipt.tokens == 4


def test_later_exposure_moves_conditional_preference_and_sham_cancels():
    older = fit_counts([[0, 1, 2]] * 4, "older", 4, "a" * 64)
    newer = fit_counts([[0, 1, 2]] * 4 + [[0, 1, 3]] * 4, "newer", 4, "a" * 64)
    old_scores = older.log_probabilities([0, 1])
    new_scores = newer.log_probabilities([0, 1])
    assert new_scores[3] - new_scores[2] > old_scores[3] - old_scores[2]
    assert np.exp(new_scores).sum() == pytest.approx(1)
    np.testing.assert_array_equal(old_scores - older.log_probabilities([0, 1]), np.zeros(4))


def test_sequence_score_uses_candidate_prefix_at_each_step():
    model = fit_counts([[0, 1, 2, 3]], "older", 4, "a" * 64)
    expected = model.log_probabilities([0, 1])[2] + model.log_probabilities([0, 1, 2])[3]
    assert sequence_score(model, [0, 1], [2, 3]) == expected
