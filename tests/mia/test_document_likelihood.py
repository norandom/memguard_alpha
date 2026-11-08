"""Offline tests for recall_guard.mia.document_likelihood (synthetic per-token arrays)."""
from __future__ import annotations

import math
import zlib

import numpy as np
import pytest

from recall_guard.mia.document_likelihood import (
    FEATURES,
    cross_validated_auc,
    document_features,
    feature_matrix,
    feature_signs,
    fit_calibrator,
    min_k_prob,
)


def test_feature_math_on_hand_values():
    logprob = [-0.5, -2.0, -0.1, -4.0, -1.0, -0.2, -3.0, -0.3, -0.7, -1.5]
    entropy = [1.0, 2.0, 0.5, 3.0, 1.5, 0.2, 2.5, 0.4, 0.9, 1.2]
    rank = [1, 3, 1, 20, 2, 1, 7, 1, 1, 4]
    text = 'Example press release text.'
    f = document_features(logprob, entropy, rank, text)
    assert list(f) == list(FEATURES)
    assert f['mean_nll'] == pytest.approx(13.3 / 10)
    assert f['min_k_10'] == pytest.approx(-4.0)            # floor(0.1 * 10) = 1 token
    assert f['min_k_20'] == pytest.approx((-4.0 - 3.0) / 2)  # 2 tokens
    assert f['zlib_ratio'] == pytest.approx(13.3 / len(zlib.compress(text.encode('utf-8'), 9)))
    assert f['mean_entropy'] == pytest.approx(np.mean(entropy))
    assert f['mean_log_rank'] == pytest.approx(np.mean([math.log(r) for r in rank]))


def test_min_k_edge_cases():
    assert min_k_prob([-2.5], 0.1) == -2.5                 # single token: at least one is kept
    assert min_k_prob([-1.0, -3.0, -2.0], 0.2) == -3.0     # floor(0.6) = 0 -> 1
    assert min_k_prob([-1.0, -1.0, -1.0, -1.0], 0.5) == -1.0  # ties
    assert min_k_prob([-1.0, -2.0], 1.0) == -1.5
    values = list(-np.arange(1, 20, dtype=float))          # 19 tokens: 20 % -> 3 lowest
    assert min_k_prob(values, 0.2) == pytest.approx(np.mean([-19, -18, -17]))
    for bad in ([], [[-1.0]]):
        with pytest.raises(ValueError):
            min_k_prob(bad, 0.1)
    for fraction in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            min_k_prob([-1.0], fraction)


def test_document_features_rejects_invalid_arrays():
    with pytest.raises(ValueError):
        document_features([], [], [], 'x')
    with pytest.raises(ValueError):
        document_features([-1.0, -2.0], [1.0], [1, 1], 'x')
    with pytest.raises(ValueError):
        document_features([0.5], [1.0], [1], 'x')            # positive logprob
    with pytest.raises(ValueError):
        document_features([-np.inf], [1.0], [1], 'x')
    with pytest.raises(ValueError):
        document_features([-1.0], [1.0], [0], 'x')            # rank below 1


def _synthetic(seed=1, n=60):
    rng = np.random.default_rng(seed)
    labels = np.r_[np.ones(n // 3, dtype=int), np.zeros(n - n // 3, dtype=int)]
    rows = []
    for label in labels:
        k = int(rng.integers(50, 120))
        shift = 0.6 if label else 0.0
        logprob = -np.abs(rng.normal(2.0 - shift, 1.0, k))
        rows.append(document_features(logprob, np.abs(rng.normal(1.5 - shift, 0.5, k)),
                                      rng.integers(1, 50, k), 'x' * int(rng.integers(200, 400))))
    return feature_matrix(rows), labels


def test_calibrator_is_deterministic_and_separates():
    matrix, labels = _synthetic()
    a, b = fit_calibrator(matrix, labels), fit_calibrator(matrix, labels)
    assert np.array_equal(a.predict_memorized(matrix), b.predict_memorized(matrix))
    assert a.features == FEATURES and a.seed == 0
    assert cross_validated_auc(matrix, labels) == cross_validated_auc(matrix, labels)
    cv = cross_validated_auc(matrix, labels)
    assert len(cv['fold_auc']) == 5 and cv['mean_auc'] > 0.7
    signs = feature_signs(matrix, labels)
    assert signs['mean_nll'] == -1 and signs['min_k_20'] == 1   # memorized: lower NLL, higher Min-K


def test_calibrator_requires_both_classes_and_shape():
    matrix, labels = _synthetic()
    with pytest.raises(ValueError):
        fit_calibrator(matrix, np.ones_like(labels))
    with pytest.raises(ValueError):
        fit_calibrator(matrix[:, :3], labels)
