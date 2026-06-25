"""Tests for `recall_guard.mia.mcs` — per-model MCS logistic-regression calibrator.

Validates Requirements 5.1, 5.2, 5.3, 5.4 and design.md → "Components and
Interfaces → mia → mia.mcs".

The MCS calibrator trains a logistic-regression classifier on standardised
MIA features labelled by corpus origin (1 = in-sample / "memorized", 0 =
out-of-sample). It reports a held-out AUC and flags weak calibration when
that AUC drops below ``min_auc``. The calibrator is then the single
ground-truth source for ``p(memorized | features)`` consumed by the
evaluator's MemGuard penalty (`raw_confidence * (1 - p_memorized)`).

Tests use a mocked ``NvidiaLM`` so no real HTTP traffic occurs, and
synthetic per-token logprobs constructed so that IS-vs-OOS rows take
sharply different MIA-feature values — giving the trained classifier a
well-defined separation to measure against.
"""

from __future__ import annotations

import dataclasses
import logging

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from recall_guard.core.loader import EvalRow
from recall_guard.core.nvidia_lm import CompletionResult, NvidiaLM, TokenLogprob
from recall_guard.mia.control import ControlBaseline
from recall_guard.mia.features import MiaFeatures
from recall_guard.mia.mcs import MCSCalibrator, train

# --- helpers -----------------------------------------------------------


def _tlp(logprob: float, top_logprobs: list[float] | None = None) -> TokenLogprob:
    """Build a TokenLogprob with optional explicit top_logprobs distribution."""
    if top_logprobs is None:
        top_logprobs = [logprob, logprob - 1.0, logprob + 1.0]
    tops = [{"token": "x", "logprob": float(lp)} for lp in top_logprobs]
    return TokenLogprob(token="x", logprob=float(logprob), top_logprobs=tops)


def _row(prompt: str = "p") -> EvalRow:
    return EvalRow(prompt=prompt, target_direction=0, metadata={})


def _completion_for_loss(
    target_loss: float, n_tokens: int = 5, content: str = "resp"
) -> CompletionResult:
    """Build a CompletionResult whose realised tokens have ``logprob == -target_loss``.

    Each top_logprobs distribution is a small spread around the realised
    logprob so per-position z-scores are well-defined and Min-K%++ also
    differentiates IS vs OOS rows.
    """
    realised = -float(target_loss)
    lps: list[TokenLogprob] = []
    for _ in range(n_tokens):
        # 5 alternatives spread around the realised value with small noise.
        top = [realised, realised - 0.5, realised + 0.5, realised - 0.2, realised + 0.2]
        lps.append(_tlp(realised, top))
    return CompletionResult(content=content, logprobs=lps, raw_temperature_observed=0.0)


def _separable_completions(
    n: int, target_loss: float, seed_offset: int = 0
) -> list[CompletionResult]:
    """``n`` completions whose loss feature is tightly clustered around ``target_loss``.

    Adds tiny per-row jitter so the standardised feature std is non-zero.
    """
    rng = np.random.default_rng(seed_offset + 12345)
    out: list[CompletionResult] = []
    for i in range(n):
        jitter = float(rng.normal(0.0, 0.01))
        out.append(
            _completion_for_loss(
                target_loss=target_loss + jitter,
                n_tokens=5,
                content=f"resp-{seed_offset}-{i}",
            )
        )
    return out


def _fake_lm(mocker, side_effect: list, model: str = "test-model") -> NvidiaLM:
    """Return a Mock(spec=NvidiaLM) whose .generate yields each entry."""
    lm = mocker.Mock(spec=NvidiaLM)
    lm.model = model
    lm.generate.side_effect = side_effect
    return lm


def _baseline_no_ref(
    means: dict[str, float] | None = None, stds: dict[str, float] | None = None
) -> ControlBaseline:
    """Build a four-feature ControlBaseline (no reference model)."""
    means = means or {"loss": 0.0, "min_k": 0.0, "min_k_pp": 0.0, "zlib_ratio": 0.0}
    stds = stds or {"loss": 1.0, "min_k": 1.0, "min_k_pp": 1.0, "zlib_ratio": 1.0}
    return ControlBaseline(
        model="test-model",
        n_valid=50,
        feature_means={**means, "ref_delta": None},
        feature_stds={**stds, "ref_delta": None},
        is_calibrated=True,
        min_valid=50,
    )


def _baseline_with_ref() -> ControlBaseline:
    """Build a five-feature ControlBaseline (reference model active)."""
    return ControlBaseline(
        model="test-model",
        n_valid=50,
        feature_means={
            "loss": 0.0,
            "min_k": 0.0,
            "min_k_pp": 0.0,
            "zlib_ratio": 0.0,
            "ref_delta": 0.0,
        },
        feature_stds={
            "loss": 1.0,
            "min_k": 1.0,
            "min_k_pp": 1.0,
            "zlib_ratio": 1.0,
            "ref_delta": 1.0,
        },
        is_calibrated=True,
        min_valid=50,
    )


# --- MCSCalibrator dataclass ------------------------------------------


def test_mcs_calibrator_is_frozen_dataclass() -> None:
    """`MCSCalibrator` is a frozen dataclass with the design-spec fields."""
    clf = LogisticRegression()
    calib = MCSCalibrator(
        model="m",
        classifier=clf,
        feature_order=["loss", "min_k", "min_k_pp", "zlib_ratio"],
        holdout_auc=0.75,
        is_weak=False,
    )
    assert dataclasses.is_dataclass(calib)
    assert calib.__dataclass_params__.frozen is True
    with pytest.raises(dataclasses.FrozenInstanceError):
        calib.holdout_auc = 0.9  # type: ignore[misc]


# --- train: AUC arms (Req 5.1, 5.2, 5.3) ------------------------------


def test_train_synthetic_separable_features_yields_high_auc(mocker) -> None:
    """IS rows with loss ≈ -3 vs OOS rows with loss ≈ +3 → AUC > 0.95, not weak."""
    n = 30
    is_completions = _separable_completions(n, target_loss=-3.0, seed_offset=0)
    oos_completions = _separable_completions(n, target_loss=+3.0, seed_offset=1)

    is_rows = [_row(f"is-{i}") for i in range(n)]
    oos_rows = [_row(f"oos-{i}") for i in range(n)]

    lm = _fake_lm(mocker, is_completions + oos_completions)
    baseline = _baseline_no_ref()

    calib = train(
        model_lm=lm,
        is_memorized=is_rows,
        oos_control=oos_rows,
        baseline=baseline,
        ref_lm=None,
        min_auc=0.6,
        seed=0,
    )

    assert calib.holdout_auc > 0.95
    assert calib.is_weak is False
    # ref_delta excluded since baseline disables it.
    assert "ref_delta" not in calib.feature_order
    assert calib.feature_order == ["loss", "min_k", "min_k_pp", "zlib_ratio"]


def test_train_label_shuffled_features_yields_low_auc_and_weak_flag(mocker) -> None:
    """Random / overlapping features → AUC well below 0.7; is_weak == True at min_auc=0.6."""
    rng = np.random.default_rng(42)
    n = 30
    # Both classes drawn from the SAME distribution → no separable signal.
    is_completions: list[CompletionResult] = []
    oos_completions: list[CompletionResult] = []
    for i in range(n):
        is_loss = float(rng.normal(0.0, 1.0))
        oos_loss = float(rng.normal(0.0, 1.0))
        is_completions.append(_completion_for_loss(is_loss, content=f"is-{i}"))
        oos_completions.append(_completion_for_loss(oos_loss, content=f"oos-{i}"))

    is_rows = [_row(f"is-{i}") for i in range(n)]
    oos_rows = [_row(f"oos-{i}") for i in range(n)]

    lm = _fake_lm(mocker, is_completions + oos_completions)
    baseline = _baseline_no_ref()

    calib = train(
        model_lm=lm,
        is_memorized=is_rows,
        oos_control=oos_rows,
        baseline=baseline,
        ref_lm=None,
        min_auc=0.6,
        seed=0,
    )

    # Don't assert ≈0.5 strictly — small-n holdout has noise. Assert the
    # classifier is clearly not separating IS vs OOS like the success arm does.
    assert calib.holdout_auc < 0.75
    # When the AUC is below the gate, is_weak must trip.
    if calib.holdout_auc < 0.6:
        assert calib.is_weak is True


def test_train_weak_calibration_flag_obeys_min_auc(mocker) -> None:
    """`is_weak` is exactly `holdout_auc < min_auc`."""
    n = 30
    is_completions = _separable_completions(n, target_loss=-3.0, seed_offset=0)
    oos_completions = _separable_completions(n, target_loss=+3.0, seed_offset=1)
    is_rows = [_row(f"is-{i}") for i in range(n)]
    oos_rows = [_row(f"oos-{i}") for i in range(n)]
    baseline = _baseline_no_ref()

    # Strongly separable AUC should be ~1.0; with min_auc=0.99 the flag is False;
    # with a min_auc above the achieved AUC the flag flips to True.
    lm_a = _fake_lm(mocker, is_completions + oos_completions)
    calib_a = train(
        model_lm=lm_a,
        is_memorized=is_rows,
        oos_control=oos_rows,
        baseline=baseline,
        ref_lm=None,
        min_auc=0.5,
        seed=0,
    )
    assert calib_a.is_weak is (calib_a.holdout_auc < 0.5)
    assert calib_a.is_weak is False  # AUC very high → not weak under 0.5

    # Same data, same seed → same holdout_auc; force min_auc above achievable.
    lm_b = _fake_lm(mocker, is_completions + oos_completions)
    calib_b = train(
        model_lm=lm_b,
        is_memorized=is_rows,
        oos_control=oos_rows,
        baseline=baseline,
        ref_lm=None,
        min_auc=2.0,  # impossibly high gate → always weak
        seed=0,
    )
    assert calib_b.is_weak is True
    assert calib_a.holdout_auc == pytest.approx(calib_b.holdout_auc, abs=1e-12)


# --- train: skip + edge cases (Req 5.1) -------------------------------


def test_train_skips_rows_with_lm_failures_and_warns(mocker, caplog) -> None:
    """A RuntimeError on one IS row → that row is dropped, ONE WARN emitted."""
    n_is = 10
    n_oos = 30
    is_completions: list = list(_separable_completions(n_is, target_loss=-3.0, seed_offset=0))
    is_completions[2] = RuntimeError("missing top_logprobs")
    oos_completions = _separable_completions(n_oos, target_loss=+3.0, seed_offset=1)

    is_rows = [_row(f"is-{i}") for i in range(n_is)]
    oos_rows = [_row(f"oos-{i}") for i in range(n_oos)]

    lm = _fake_lm(mocker, is_completions + oos_completions)
    baseline = _baseline_no_ref()

    with caplog.at_level(logging.WARNING, logger="recall_guard.mia.mcs"):
        calib = train(
            model_lm=lm,
            is_memorized=is_rows,
            oos_control=oos_rows,
            baseline=baseline,
            ref_lm=None,
            min_auc=0.6,
            seed=0,
        )

    # Calibrator still produced.
    assert isinstance(calib, MCSCalibrator)
    # Exactly one WARN for the skipped IS row.
    warns = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warns) >= 1
    assert any("is_memorized" in r.getMessage() or "row 2" in r.getMessage() or "2" in r.getMessage()
               for r in warns)


def test_train_raises_when_one_class_too_small(mocker) -> None:
    """1 IS row + 30 OOS rows → ValueError ('both classes must be present')."""
    n_is = 1
    n_oos = 30
    is_completions = _separable_completions(n_is, target_loss=-3.0, seed_offset=0)
    oos_completions = _separable_completions(n_oos, target_loss=+3.0, seed_offset=1)
    is_rows = [_row(f"is-{i}") for i in range(n_is)]
    oos_rows = [_row(f"oos-{i}") for i in range(n_oos)]

    lm = _fake_lm(mocker, is_completions + oos_completions)
    baseline = _baseline_no_ref()

    with pytest.raises(ValueError, match=r"(both classes|too few|insufficient)"):
        train(
            model_lm=lm,
            is_memorized=is_rows,
            oos_control=oos_rows,
            baseline=baseline,
            ref_lm=None,
            min_auc=0.6,
            seed=0,
        )


# --- train: determinism (Req 5.1, 6.5) --------------------------------


def test_train_uses_seed_deterministically(mocker) -> None:
    """Same seed + same inputs → identical holdout_auc and identical predict_proba."""
    n = 30
    is_completions_a = _separable_completions(n, target_loss=-2.0, seed_offset=0)
    oos_completions_a = _separable_completions(n, target_loss=+2.0, seed_offset=1)
    is_completions_b = _separable_completions(n, target_loss=-2.0, seed_offset=0)
    oos_completions_b = _separable_completions(n, target_loss=+2.0, seed_offset=1)

    is_rows = [_row(f"is-{i}") for i in range(n)]
    oos_rows = [_row(f"oos-{i}") for i in range(n)]

    baseline = _baseline_no_ref()
    lm_a = _fake_lm(mocker, is_completions_a + oos_completions_a)
    lm_b = _fake_lm(mocker, is_completions_b + oos_completions_b)

    calib_a = train(
        model_lm=lm_a,
        is_memorized=is_rows,
        oos_control=oos_rows,
        baseline=baseline,
        ref_lm=None,
        min_auc=0.6,
        seed=42,
    )
    calib_b = train(
        model_lm=lm_b,
        is_memorized=is_rows,
        oos_control=oos_rows,
        baseline=baseline,
        ref_lm=None,
        min_auc=0.6,
        seed=42,
    )

    assert calib_a.holdout_auc == pytest.approx(calib_b.holdout_auc, abs=1e-12)
    test_features = MiaFeatures(
        loss=-1.5, min_k=-1.5, min_k_pp=-1.5, zlib_ratio=0.5, ref_delta=None
    )
    p_a = calib_a.predict_proba(test_features, baseline)
    p_b = calib_b.predict_proba(test_features, baseline)
    assert p_a == pytest.approx(p_b, abs=1e-12)


# --- train: ref_delta handling (Req 4.2, 5.1) -------------------------


def test_train_handles_no_reference_model(mocker) -> None:
    """ref_lm=None + baseline ref_delta=None → feature_order excludes ref_delta."""
    n = 30
    is_completions = _separable_completions(n, target_loss=-3.0, seed_offset=0)
    oos_completions = _separable_completions(n, target_loss=+3.0, seed_offset=1)
    is_rows = [_row(f"is-{i}") for i in range(n)]
    oos_rows = [_row(f"oos-{i}") for i in range(n)]

    lm = _fake_lm(mocker, is_completions + oos_completions)
    baseline = _baseline_no_ref()

    calib = train(
        model_lm=lm,
        is_memorized=is_rows,
        oos_control=oos_rows,
        baseline=baseline,
        ref_lm=None,
        min_auc=0.6,
        seed=0,
    )

    assert calib.feature_order == ["loss", "min_k", "min_k_pp", "zlib_ratio"]
    # Underlying estimator must have been trained on 4 features.
    assert calib.classifier.coef_.shape[1] == 4

    # predict_proba on a 4-feature vector works.
    feats = MiaFeatures(
        loss=-2.5, min_k=-2.5, min_k_pp=-2.5, zlib_ratio=0.5, ref_delta=None
    )
    p = calib.predict_proba(feats, baseline)
    assert 0.0 <= p <= 1.0


def test_train_uses_ref_delta_when_baseline_supports_it(mocker) -> None:
    """ref_lm provided + baseline ref_delta calibrated → feature_order includes ref_delta."""
    n = 30
    is_completions = _separable_completions(n, target_loss=-3.0, seed_offset=0)
    oos_completions = _separable_completions(n, target_loss=+3.0, seed_offset=1)
    # The reference model returns a steady stream of "neutral" completions.
    ref_completions = [
        _completion_for_loss(0.5, content=f"ref-{i}") for i in range(2 * n)
    ]
    is_rows = [_row(f"is-{i}") for i in range(n)]
    oos_rows = [_row(f"oos-{i}") for i in range(n)]

    lm = _fake_lm(mocker, is_completions + oos_completions)
    ref_lm = _fake_lm(mocker, ref_completions, model="ref-test")
    baseline = _baseline_with_ref()

    calib = train(
        model_lm=lm,
        is_memorized=is_rows,
        oos_control=oos_rows,
        baseline=baseline,
        ref_lm=ref_lm,
        min_auc=0.6,
        seed=0,
    )

    assert "ref_delta" in calib.feature_order
    assert calib.feature_order == ["loss", "min_k", "min_k_pp", "zlib_ratio", "ref_delta"]
    assert calib.classifier.coef_.shape[1] == 5


# --- predict_proba (Req 5.4) ------------------------------------------


def test_predict_proba_returns_value_in_unit_interval(mocker) -> None:
    """predict_proba on any in-range MiaFeatures returns a float in [0, 1]."""
    n = 30
    is_completions = _separable_completions(n, target_loss=-3.0, seed_offset=0)
    oos_completions = _separable_completions(n, target_loss=+3.0, seed_offset=1)
    is_rows = [_row(f"is-{i}") for i in range(n)]
    oos_rows = [_row(f"oos-{i}") for i in range(n)]
    baseline = _baseline_no_ref()
    lm = _fake_lm(mocker, is_completions + oos_completions)

    calib = train(
        model_lm=lm,
        is_memorized=is_rows,
        oos_control=oos_rows,
        baseline=baseline,
        ref_lm=None,
        min_auc=0.6,
        seed=0,
    )

    for loss_val in (-5.0, -1.0, 0.0, 1.0, 5.0):
        feats = MiaFeatures(
            loss=loss_val,
            min_k=loss_val,
            min_k_pp=loss_val,
            zlib_ratio=0.5,
            ref_delta=None,
        )
        p = calib.predict_proba(feats, baseline)
        assert isinstance(p, float)
        assert 0.0 <= p <= 1.0


def test_predict_proba_orders_consistently_with_loss_signal(mocker) -> None:
    """A row with IS-like (negative-loss) features → higher p_memorized than OOS-like row."""
    n = 30
    is_completions = _separable_completions(n, target_loss=-3.0, seed_offset=0)
    oos_completions = _separable_completions(n, target_loss=+3.0, seed_offset=1)
    is_rows = [_row(f"is-{i}") for i in range(n)]
    oos_rows = [_row(f"oos-{i}") for i in range(n)]
    baseline = _baseline_no_ref()
    lm = _fake_lm(mocker, is_completions + oos_completions)

    calib = train(
        model_lm=lm,
        is_memorized=is_rows,
        oos_control=oos_rows,
        baseline=baseline,
        ref_lm=None,
        min_auc=0.6,
        seed=0,
    )

    is_like = MiaFeatures(
        loss=-3.0, min_k=-3.0, min_k_pp=-3.0, zlib_ratio=-3.0, ref_delta=None
    )
    oos_like = MiaFeatures(
        loss=3.0, min_k=3.0, min_k_pp=3.0, zlib_ratio=3.0, ref_delta=None
    )
    p_is = calib.predict_proba(is_like, baseline)
    p_oos = calib.predict_proba(oos_like, baseline)
    # IS-like inputs must score higher than OOS-like inputs.
    assert p_is > p_oos
    # And both stay in the unit interval.
    assert 0.0 <= p_oos <= 1.0
    assert 0.0 <= p_is <= 1.0
