"""Document-likelihood features and calibrator (release-1.0, register R37).

Scores the candidate text itself: the inputs are teacher-forced per-token
arrays for one document, computed from the model's FULL next-token
distribution at every position t-1 for the actual token t (see
``scripts/score_documents_llamacpp.py``):

- ``logprob[i]``: log p(token_{i+1} | tokens_{<=i}) in nats,
- ``entropy[i]``: entropy of that distribution in nats,
- ``rank[i]``:    rank of the actual token (1 = argmax).

Features (one value per document; higher is not assumed to mean memorized --
the direction of each single feature is chosen on the calibration split):

- ``mean_nll``:     mean negative logprob.
- ``min_k_10`` / ``min_k_20``: mean logprob of the lowest ``k`` fraction of
  tokens, ``max(1, floor(k * n))`` tokens (Min-K% Prob, Shi et al. 2024).
- ``zlib_ratio``:   NLL sum divided by ``len(zlib.compress(text_utf8, 9))``.
- ``mean_entropy``: mean entropy.
- ``mean_log_rank``: mean of ``log(rank)``.

The calibrator is a standardised logistic regression with balanced class
weights and a fixed seed, fitted on calibration documents only. Feature math
is numpy only; the calibrator uses scikit-learn.
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass

import numpy as np

FEATURES: tuple[str, ...] = ('mean_nll', 'min_k_10', 'min_k_20', 'zlib_ratio', 'mean_entropy', 'mean_log_rank')
MIN_K_FRACTIONS: dict[str, float] = {'min_k_10': 0.10, 'min_k_20': 0.20}
SEED = 0


def min_k_prob(logprob, fraction: float) -> float:
    """Mean of the lowest ``max(1, floor(fraction * n))`` logprobs."""
    values = np.asarray(logprob, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError('min_k_prob needs a non-empty 1-D array')
    if not 0.0 < fraction <= 1.0:
        raise ValueError('fraction must be in (0, 1]')
    count = max(1, math.floor(fraction * values.size))
    return float(np.sort(values)[:count].mean())


def document_features(logprob, entropy, rank, text: str) -> dict[str, float]:
    """The FEATURES of one document from its per-token arrays and raw text."""
    logprob = np.asarray(logprob, dtype=np.float64)
    entropy = np.asarray(entropy, dtype=np.float64)
    rank = np.asarray(rank, dtype=np.float64)
    if logprob.ndim != 1 or logprob.size == 0 or entropy.shape != logprob.shape or rank.shape != logprob.shape:
        raise ValueError('logprob, entropy and rank must be non-empty 1-D arrays of equal length')
    if not (np.isfinite(logprob).all() and np.isfinite(entropy).all()) or (logprob > 0).any():
        raise ValueError('logprob must be finite and <= 0; entropy finite')
    if (rank < 1).any():
        raise ValueError('rank must be >= 1')
    compressed = len(zlib.compress(text.encode('utf-8'), 9))
    features = {'mean_nll': float(-logprob.mean())}
    features |= {name: min_k_prob(logprob, k) for name, k in MIN_K_FRACTIONS.items()}
    features |= {'zlib_ratio': float(-logprob.sum() / compressed), 'mean_entropy': float(entropy.mean()),
                 'mean_log_rank': float(np.log(rank).mean())}
    return features


def feature_matrix(rows, features=FEATURES) -> np.ndarray:
    """Stack feature dicts into an (n, len(features)) float array in a fixed order."""
    return np.array([[row[name] for name in features] for row in rows], dtype=np.float64)


@dataclass(frozen=True)
class DocumentCalibrator:
    """Fitted pipeline plus the frozen feature order and seed."""
    pipeline: object
    features: tuple[str, ...]
    seed: int

    def predict_memorized(self, matrix) -> np.ndarray:
        return self.pipeline.predict_proba(np.asarray(matrix, dtype=np.float64))[:, 1]


def _pipeline(seed: int):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(class_weight='balanced', random_state=seed, max_iter=1000))


def fit_calibrator(matrix, labels, *, features=FEATURES, seed: int = SEED) -> DocumentCalibrator:
    """Fit on calibration documents only (label 1 = memorized)."""
    matrix, labels = np.asarray(matrix, dtype=np.float64), np.asarray(labels, dtype=int)
    if matrix.shape != (labels.size, len(features)) or set(np.unique(labels)) != {0, 1}:
        raise ValueError('need one row per label, one column per feature, and both classes')
    return DocumentCalibrator(_pipeline(seed).fit(matrix, labels), tuple(features), seed)


def cross_validated_auc(matrix, labels, *, folds: int = 5, seed: int = SEED) -> dict:
    """Stratified k-fold ROC-AUC of the calibrator on the calibration split."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    matrix, labels = np.asarray(matrix, dtype=np.float64), np.asarray(labels, dtype=int)
    pooled = np.empty(labels.size)
    per_fold = []
    for train, test in StratifiedKFold(folds, shuffle=True, random_state=seed).split(matrix, labels):
        scores = _pipeline(seed).fit(matrix[train], labels[train]).predict_proba(matrix[test])[:, 1]
        pooled[test] = scores
        per_fold.append(float(roc_auc_score(labels[test], scores)))
    return {'folds': folds, 'seed': seed, 'fold_auc': per_fold, 'mean_auc': float(np.mean(per_fold)),
            'std_auc': float(np.std(per_fold)), 'pooled_auc': float(roc_auc_score(labels, pooled))}


def feature_signs(matrix, labels, *, features=FEATURES) -> dict[str, int]:
    """+1 if larger values indicate memorized on calibration (AUC >= 0.5), else -1."""
    from sklearn.metrics import roc_auc_score

    matrix, labels = np.asarray(matrix, dtype=np.float64), np.asarray(labels, dtype=int)
    return {name: (1 if roc_auc_score(labels, matrix[:, j]) >= 0.5 else -1) for j, name in enumerate(features)}
