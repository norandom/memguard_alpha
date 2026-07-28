"""Per-model MCS (Memorization Contamination Score) logistic-regression calibrator.

Implements the ``mia.mcs`` component from the honest-model-ranking design.
Satisfies Requirements 5.1, 5.2, 5.3, 5.4:

- ``MCSCalibrator``: frozen dataclass holding the trained
  ``LogisticRegression`` estimator, the canonical ``feature_order`` used
  during training (so ``predict_proba`` cannot accidentally feed the
  classifier a permuted vector), the held-out AUC, and an ``is_weak``
  flag set when ``holdout_auc < min_auc``.
- ``train(model_lm, is_memorized, oos_control, baseline, ref_lm,
  min_auc=0.6, seed=0)``: runs the model (and optional reference) on
  every row, computes MIA features, standardises them against the
  per-model control baseline, splits a held-out portion via
  ``sklearn.model_selection.train_test_split`` (``test_size=0.25``,
  ``stratify=y``, ``random_state=seed``), fits
  ``LogisticRegression(class_weight="balanced", solver="liblinear",
  random_state=seed)`` on the training half, scores
  ``roc_auc_score`` on the holdout half.
- ``MCSCalibrator.predict_proba(features, baseline) -> float``,
  pure: standardises ``features`` against ``baseline``, builds the
  classifier input vector in ``feature_order``, and returns the
  ``predict_proba(...)[:, 1]`` value.

Per-row LM failures (``TimeoutError``, ``RuntimeError``,
``ValueError``) are skipped with a single WARNING per skip; every
other code path is pure. The MemGuard penalty rule consumed downstream
is ``penalized_confidence = raw_confidence * (1 - p_memorized)``
(Req 5.4: continuous, not threshold-based).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from recall_guard.core.loader import EvalRow
from recall_guard.core.nvidia_lm import NvidiaLM, TokenLogprob, generate_many
from recall_guard.mia.control import ControlBaseline, standardise
from recall_guard.mia.features import MiaFeatures, compute_mia_features

logger = logging.getLogger(__name__)


_BASE_FEATURE_ORDER: tuple[str, ...] = (
    "loss",
    "min_k",
    "min_k_pp",
    "zlib_ratio",
)

_HOLDOUT_FRACTION: float = 0.25


@dataclass(frozen=True)
class MCSCalibrator:
    """Per-model logistic-regression calibrator for ``p(memorized | features)``.

    Attributes
    ----------
    model:
        The NVIDIA model ID this calibrator was trained for.
    classifier:
        The fitted ``sklearn.linear_model.LogisticRegression`` instance.
        sklearn estimators are mutable; ``frozen=True`` only prevents
        reassignment of the field reference, which is the design intent.
    feature_order:
        Canonical order used to flatten the standardised feature dict
        into the classifier's input vector. Populated at train time and
        consumed verbatim by :meth:`predict_proba` so the classifier is
        never fed a permuted row.
    holdout_auc:
        ROC-AUC score of the trained classifier on the 25% held-out
        portion of the labelled IS/OOS corpus. Reported in the manifest
        and the per-model evaluation result (Req 5.2).
    is_weak:
        ``True`` iff ``holdout_auc < min_auc`` at train time. Surfaced
        as the ``weak-calibration`` warning in ``top3.md`` (Req 5.3).
    """

    model: str
    classifier: LogisticRegression
    feature_order: list[str]
    holdout_auc: float
    is_weak: bool

    def predict_proba(
        self, features: MiaFeatures, baseline: ControlBaseline
    ) -> float:
        """Return the calibrated probability of "memorized" for one record.

        Standardises ``features`` against the model's ``baseline`` and
        feeds the resulting vector to the trained classifier in
        ``self.feature_order``.

        Returns
        -------
        float
            ``p(memorized | features) ∈ [0.0, 1.0]``.

        Raises
        ------
        ValueError
            If any of the four core features standardises to ``None``
            (uncalibrated baseline). A missing ``ref_delta`` does NOT
            raise: the reference feature is optional by contract, so it
            is imputed at the control-baseline mean (standardised 0.0),
            which contributes no memorization evidence either way.
        """
        standardised = standardise(features, baseline)
        if "ref_delta" in self.feature_order and standardised.get("ref_delta") is None:
            standardised = {**standardised, "ref_delta": 0.0}
        row = _row_vector(standardised, self.feature_order)
        # Estimator was trained on a 2-D matrix; predict on a 1-row matrix.
        proba = float(self.classifier.predict_proba(row.reshape(1, -1))[0, 1])
        # sklearn returns values strictly in [0, 1] for LR; clamp defensively
        # against fp64 round-off so the float postcondition holds exactly.
        if proba < 0.0:
            proba = 0.0
        elif proba > 1.0:
            proba = 1.0
        return proba


def _row_vector(standardised: dict[str, float | None], order: list[str]) -> np.ndarray:
    """Flatten a standardised feature dict into a 1-D float64 array.

    Raises ``ValueError`` if any value in ``order`` is ``None``; that
    means the caller asked for a feature the classifier expects but
    cannot supply (e.g., asking for ``ref_delta`` when ref_logprobs
    were never recorded for this row).
    """
    out = np.empty(len(order), dtype=np.float64)
    for i, key in enumerate(order):
        value = standardised.get(key)
        if value is None:
            raise ValueError(
                f"MCSCalibrator: standardised feature {key!r} is None; "
                "cannot build classifier input vector."
            )
        out[i] = float(value)
    return out


def _resolve_feature_order(baseline: ControlBaseline) -> list[str]:
    """Pick the column order based on whether the baseline supports ref_delta."""
    order = list(_BASE_FEATURE_ORDER)
    if baseline.feature_means.get("ref_delta") is not None:
        order.append("ref_delta")
    return order


def _collect_features(
    model_lm: NvidiaLM,
    rows: list[EvalRow],
    ref_lm: NvidiaLM | None,
    baseline: ControlBaseline,
    feature_order: list[str],
    label: int,
    corpus_name: str,
    max_workers: int = 1,
) -> tuple[list[list[float]], list[int]]:
    """Run the LM (and optional reference) over ``rows`` and return (X-rows, y).

    With ``max_workers > 1`` the per-row LM calls fan out via
    ``concurrent.futures.ThreadPoolExecutor`` while preserving input order.
    Skips rows where the model call fails, where the reference run
    fails *and* ``ref_delta`` is in ``feature_order``, or where MIA
    feature computation raises. Each skip emits exactly one
    ``logging.WARNING`` record.
    """
    x_rows: list[list[float]] = []
    y_rows: list[int] = []
    needs_ref = "ref_delta" in feature_order

    prompts = [row.prompt for row in rows]
    primary_results = generate_many(model_lm, prompts, max_workers=max_workers)
    ref_results: list = (
        generate_many(ref_lm, prompts, max_workers=max_workers)
        if ref_lm is not None else [None] * len(prompts)
    )

    for idx, (primary, ref_res) in enumerate(zip(primary_results, ref_results, strict=True)):
        if isinstance(primary, Exception) or primary is None:
            logger.warning(
                "mcs.train: skipping row %d in %s for model %s "
                "(timeout or missing logprobs)",
                idx,
                corpus_name,
                model_lm.model,
            )
            continue
        content, logprobs = primary.content, primary.logprobs

        ref_logprobs: list[TokenLogprob] | None = None
        if ref_lm is not None:
            if isinstance(ref_res, Exception) or ref_res is None:
                if needs_ref:
                    logger.warning(
                        "mcs.train: skipping row %d in %s for model %s "
                        "(reference model %s failed; ref_delta required)",
                        idx,
                        corpus_name,
                        model_lm.model,
                        ref_lm.model,
                    )
                    continue
            else:
                ref_logprobs = ref_res.logprobs
        elif needs_ref:
            # The baseline reports a ref_delta calibration but the caller
            # did not pass a reference LM; that is a configuration error.
            raise ValueError(
                "mcs.train: baseline supports ref_delta but ref_lm is None; "
                "feature_order would be inconsistent between train and predict."
            )

        try:
            features = compute_mia_features(content, logprobs, ref_logprobs)
        except (ValueError, RuntimeError):
            logger.warning(
                "mcs.train: skipping row %d in %s for model %s "
                "(MIA feature computation failed)",
                idx,
                corpus_name,
                model_lm.model,
            )
            continue

        standardised = standardise(features, baseline)
        try:
            vec = _row_vector(standardised, feature_order)
        except ValueError:
            logger.warning(
                "mcs.train: skipping row %d in %s for model %s "
                "(standardised feature missing — uncalibrated baseline)",
                idx,
                corpus_name,
                model_lm.model,
            )
            continue
        x_rows.append([float(v) for v in vec.tolist()])
        y_rows.append(label)

    return x_rows, y_rows


def _gather_train_xy(
    model_lm: NvidiaLM,
    is_memorized: list[EvalRow],
    oos_control: list[EvalRow],
    baseline: ControlBaseline,
    ref_lm: NvidiaLM | None,
    feature_order: list[str],
    max_workers: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Collect labelled feature vectors for both corpora and stack them.

    Returns ``(x, y, n_valid_is, n_valid_oos)`` where ``x`` is shape
    ``(n_valid_is + n_valid_oos, len(feature_order))`` and ``y`` is the
    matching label vector. Raises ``ValueError`` when either class ends
    up with fewer than 2 valid rows after per-row skips, since logistic
    regression cannot train on single-class data.
    """
    is_x, is_y = _collect_features(
        model_lm=model_lm,
        rows=is_memorized,
        ref_lm=ref_lm,
        baseline=baseline,
        feature_order=feature_order,
        label=1,
        corpus_name="is_memorized",
        max_workers=max_workers,
    )
    oos_x, oos_y = _collect_features(
        model_lm=model_lm,
        rows=oos_control,
        ref_lm=ref_lm,
        baseline=baseline,
        feature_order=feature_order,
        label=0,
        corpus_name="oos_control",
        max_workers=max_workers,
    )

    n_valid_is = len(is_x)
    n_valid_oos = len(oos_x)
    if n_valid_is < 2 or n_valid_oos < 2:
        raise ValueError(
            "mcs.train: both classes must be present with at least 2 valid rows "
            f"(got n_valid_is={n_valid_is}, n_valid_oos={n_valid_oos}). "
            "Cannot train a stratified holdout split with one-class data."
        )
    x = np.asarray(is_x + oos_x, dtype=np.float64)
    y = np.asarray(is_y + oos_y, dtype=np.int64)
    return x, y, n_valid_is, n_valid_oos


def train(
    model_lm: NvidiaLM,
    is_memorized: list[EvalRow],
    oos_control: list[EvalRow],
    baseline: ControlBaseline,
    ref_lm: NvidiaLM | None,
    min_auc: float = 0.6,
    seed: int = 0,
    max_workers: int = 1,
) -> MCSCalibrator:
    """Train the MCS classifier for one model.

    Drives the LM over both labelled corpora (in parallel when
    ``max_workers > 1``), fits a logistic regression on the standardised
    features, and reports a held-out AUC. Raises ``ValueError`` if either
    class ends up empty after per-row skips.
    """
    feature_order = _resolve_feature_order(baseline)
    x, y, n_valid_is, n_valid_oos = _gather_train_xy(
        model_lm=model_lm,
        is_memorized=is_memorized,
        oos_control=oos_control,
        baseline=baseline,
        ref_lm=ref_lm,
        feature_order=feature_order,
        max_workers=max_workers,
    )

    # The stratified holdout needs at least one row per class in BOTH the
    # train and holdout halves. Check up front so tiny corpora fail with a
    # clear message instead of an opaque sklearn split error.
    n_total = n_valid_is + n_valid_oos
    n_holdout = math.ceil(_HOLDOUT_FRACTION * n_total)
    if n_holdout < 2 or (n_total - n_holdout) < 2:
        raise ValueError(
            f"mcs.train: {n_total} valid rows "
            f"(n_valid_is={n_valid_is}, n_valid_oos={n_valid_oos}) cannot "
            f"support the stratified {_HOLDOUT_FRACTION:.0%} holdout split "
            f"(holdout would hold {n_holdout} row(s), need >= 2 with both "
            "classes). Provide more calibration rows."
        )

    x_train, x_holdout, y_train, y_holdout = train_test_split(
        x, y,
        test_size=_HOLDOUT_FRACTION,
        random_state=seed,
        stratify=y,
    )

    classifier = LogisticRegression(
        class_weight="balanced",
        solver="liblinear",
        random_state=seed,
    )
    classifier.fit(x_train, y_train)

    holdout_scores = classifier.predict_proba(x_holdout)[:, 1]
    holdout_auc = float(roc_auc_score(y_holdout, holdout_scores))
    is_weak = holdout_auc < float(min_auc)

    logger.info(
        "mcs.train: model=%s n_valid_is=%d n_valid_oos=%d "
        "holdout_auc=%.4f is_weak=%s",
        model_lm.model, n_valid_is, n_valid_oos, holdout_auc, is_weak,
    )

    return MCSCalibrator(
        model=model_lm.model,
        classifier=classifier,
        feature_order=feature_order,
        holdout_auc=holdout_auc,
        is_weak=is_weak,
    )


__all__ = ["MCSCalibrator", "train"]
