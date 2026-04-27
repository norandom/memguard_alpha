"""Tests for `src.mia.control` — per-model control-corpus baseline.

Validates Requirements 3.1, 3.2, 3.3, 3.4 and design.md → "Components and
Interfaces → mia → mia.control".

The control baseline collects each MIA feature's mean and std on a model's
out-of-sample control corpus. Evaluation features are then standardised
against the model's own baseline so a fluent large model is not mistaken
for a "memorizer" simply because English is easy to predict.

These tests use a mocked ``NvidiaLM`` (no real HTTP) and synthetic logprobs
to exercise the calibration boundary, the no-reference path, and the
zero-mean / unit-std invariant on the control set itself.
"""

from __future__ import annotations

import dataclasses
import logging
import math

import numpy as np
import pytest

from src.core.loader import EvalRow
from src.core.nvidia_lm import CompletionResult, NvidiaLM, TokenLogprob
from src.mia.control import (
    _STD_FLOOR,
    ControlBaseline,
    build_baseline,
    standardise,
)
from src.mia.features import MiaFeatures, compute_mia_features


# --- helpers -----------------------------------------------------------


def _tlp(logprob: float, top_logprobs: list[float] | None = None) -> TokenLogprob:
    """Build a TokenLogprob with optional explicit top_logprobs distribution."""
    if top_logprobs is None:
        top_logprobs = [logprob, logprob - 1.0, logprob + 1.0]
    tops = [{"token": "x", "logprob": float(lp)} for lp in top_logprobs]
    return TokenLogprob(token="x", logprob=float(logprob), top_logprobs=tops)


def _row(prompt: str = "p") -> EvalRow:
    return EvalRow(prompt=prompt, target_direction=0, metadata={})


def _completion(content: str, lps: list[TokenLogprob]) -> CompletionResult:
    return CompletionResult(content=content, logprobs=lps, raw_temperature_observed=0.0)


def _fake_lm(mocker, side_effect: list) -> NvidiaLM:
    """Return a Mock(spec=NvidiaLM) whose .generate yields each entry in side_effect."""
    lm = mocker.Mock(spec=NvidiaLM)
    lm.model = "test-model"
    lm.generate.side_effect = side_effect
    return lm


def _make_completion_for_seed(seed: int) -> CompletionResult:
    """Build a deterministic CompletionResult that varies per seed.

    Each call returns 5 tokens whose logprobs are seed-derived so that ALL
    five MIA features take distinct values across the synthetic control set,
    including ``min_k_pp``. We achieve per-seed variation in the per-position
    z-score by drawing a fresh, non-linear ``top_logprobs`` distribution per
    token: 5 alternatives sampled iid from ``Normal(0, 1)`` and shifted to
    floor at ``base - 0.1`` so the realised token's z-score depends on the
    sampled shape rather than collapsing to a constant.
    """
    rng = np.random.default_rng(seed)
    base = rng.uniform(-1.5, -0.05, size=5)
    lps = []
    for b in base:
        raw_top = rng.normal(loc=0.0, scale=1.0, size=5)
        # Anchor the top distribution so the realised logprob sits within range.
        shifted = (raw_top - float(np.max(raw_top))) + float(b) + 0.05
        top = [float(b)] + [float(x) for x in shifted]
        lps.append(_tlp(float(b), top))
    return _completion(content=f"resp-{seed}-{base[0]:.4f}", lps=lps)


# --- ControlBaseline dataclass ----------------------------------------


def test_control_baseline_is_frozen_dataclass() -> None:
    """`ControlBaseline` is a frozen dataclass."""
    baseline = ControlBaseline(
        model="m",
        n_valid=10,
        feature_means={"loss": 0.0},
        feature_stds={"loss": 1.0},
        is_calibrated=False,
        min_valid=50,
    )
    assert dataclasses.is_dataclass(baseline)
    assert baseline.__dataclass_params__.frozen is True
    with pytest.raises(dataclasses.FrozenInstanceError):
        baseline.n_valid = 0  # type: ignore[misc]


# --- build_baseline ----------------------------------------------------


def test_build_baseline_computes_per_feature_mean_and_std(mocker) -> None:
    """Per-feature mean/std match numpy.mean / numpy.std(ddof=0) on the valid rows."""
    n = 10
    completions = [_make_completion_for_seed(s) for s in range(n)]
    rows = [_row(f"prompt-{i}") for i in range(n)]
    lm = _fake_lm(mocker, completions)

    baseline = build_baseline(lm, rows, ref_lm=None, min_valid=5)

    # Recompute the expected per-row features the same way the implementation does.
    expected_features = [
        compute_mia_features(c.content, c.logprobs, None) for c in completions
    ]
    for key in ("loss", "min_k", "min_k_pp", "zlib_ratio"):
        values = np.asarray([getattr(f, key) for f in expected_features])
        assert baseline.feature_means[key] == pytest.approx(
            float(np.mean(values)), abs=1e-9
        )
        assert baseline.feature_stds[key] == pytest.approx(
            max(float(np.std(values)), _STD_FLOOR), abs=1e-9
        )

    # ref_delta disabled → None mean/std.
    assert baseline.feature_means["ref_delta"] is None
    assert baseline.feature_stds["ref_delta"] is None
    assert baseline.n_valid == n
    assert baseline.is_calibrated is True


def test_build_baseline_skips_rows_with_runtime_error_and_warns(
    mocker, caplog
) -> None:
    """A row whose `generate` raises RuntimeError is dropped and a WARN is logged."""
    completions: list = [_make_completion_for_seed(s) for s in range(10)]
    completions[3] = RuntimeError("missing top_logprobs")
    rows = [_row(f"p-{i}") for i in range(10)]
    lm = _fake_lm(mocker, completions)

    with caplog.at_level(logging.WARNING, logger="src.mia.control"):
        baseline = build_baseline(lm, rows, ref_lm=None, min_valid=5)

    assert baseline.n_valid == 9
    # At least one WARN that mentions the row index that was dropped.
    warns = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("3" in rec.getMessage() for rec in warns)


def test_build_baseline_skips_rows_with_timeout_and_warns(mocker, caplog) -> None:
    """A row whose `generate` raises TimeoutError is dropped and a WARN is logged."""
    completions: list = [_make_completion_for_seed(s) for s in range(10)]
    completions[5] = TimeoutError("model timed out")
    rows = [_row(f"p-{i}") for i in range(10)]
    lm = _fake_lm(mocker, completions)

    with caplog.at_level(logging.WARNING, logger="src.mia.control"):
        baseline = build_baseline(lm, rows, ref_lm=None, min_valid=5)

    assert baseline.n_valid == 9
    warns = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("5" in rec.getMessage() for rec in warns)


def test_build_baseline_is_calibrated_at_boundary(mocker) -> None:
    """`is_calibrated` flips at exactly `n_valid == min_valid`."""
    # Boundary: n_valid == min_valid → True
    n = 4
    completions = [_make_completion_for_seed(s) for s in range(n)]
    rows = [_row(f"p-{i}") for i in range(n)]
    lm_ok = _fake_lm(mocker, completions)
    baseline_ok = build_baseline(lm_ok, rows, ref_lm=None, min_valid=4)
    assert baseline_ok.n_valid == 4
    assert baseline_ok.is_calibrated is True
    assert baseline_ok.min_valid == 4

    # Below boundary: n_valid == min_valid - 1 → False
    completions_below = [_make_completion_for_seed(s) for s in range(3)]
    rows_below = [_row(f"p-{i}") for i in range(3)]
    lm_low = _fake_lm(mocker, completions_below)
    baseline_low = build_baseline(lm_low, rows_below, ref_lm=None, min_valid=4)
    assert baseline_low.n_valid == 3
    assert baseline_low.is_calibrated is False


def test_build_baseline_handles_no_reference_model(mocker) -> None:
    """`ref_lm=None` → ref_delta mean/std are None, other features unaffected."""
    n = 6
    completions = [_make_completion_for_seed(s) for s in range(n)]
    rows = [_row(f"p-{i}") for i in range(n)]
    lm = _fake_lm(mocker, completions)

    baseline = build_baseline(lm, rows, ref_lm=None, min_valid=3)

    assert baseline.feature_means["ref_delta"] is None
    assert baseline.feature_stds["ref_delta"] is None
    for key in ("loss", "min_k", "min_k_pp", "zlib_ratio"):
        assert isinstance(baseline.feature_means[key], float)
        assert math.isfinite(baseline.feature_means[key])
        assert isinstance(baseline.feature_stds[key], float)
        assert baseline.feature_stds[key] >= _STD_FLOOR


def test_build_baseline_handles_failing_reference_model(mocker) -> None:
    """If ref_lm raises on every row → ref_delta features fall back to None."""
    n = 5
    main_completions = [_make_completion_for_seed(s) for s in range(n)]
    rows = [_row(f"p-{i}") for i in range(n)]
    lm = _fake_lm(mocker, main_completions)

    ref_lm = mocker.Mock(spec=NvidiaLM)
    ref_lm.model = "ref-model"
    ref_lm.generate.side_effect = [RuntimeError("no logprobs") for _ in range(n)]

    baseline = build_baseline(lm, rows, ref_lm=ref_lm, min_valid=3)

    # All four other features remain valid.
    assert baseline.n_valid == n
    for key in ("loss", "min_k", "min_k_pp", "zlib_ratio"):
        assert math.isfinite(baseline.feature_means[key])
    # ref_delta degrades cleanly to None when every valid row had ref_logprobs=None.
    assert baseline.feature_means["ref_delta"] is None
    assert baseline.feature_stds["ref_delta"] is None


# --- standardise -------------------------------------------------------


def test_standardise_zero_mean_one_std_on_control_set(mocker) -> None:
    """Standardising the control set against its own baseline yields mean ≈ 0, std ≈ 1."""
    n = 12
    completions = [_make_completion_for_seed(s) for s in range(n)]
    rows = [_row(f"p-{i}") for i in range(n)]
    lm = _fake_lm(mocker, completions)
    baseline = build_baseline(lm, rows, ref_lm=None, min_valid=5)

    per_row_features = [
        compute_mia_features(c.content, c.logprobs, None) for c in completions
    ]
    standardised = [standardise(f, baseline) for f in per_row_features]

    for key in ("loss", "min_k", "min_k_pp", "zlib_ratio"):
        values = np.asarray([s[key] for s in standardised], dtype=np.float64)
        # Std floor only kicks in when raw std is < 1e-6 (constant feature). For a
        # synthetic varying control set the std is well above the floor, so the
        # standardised mean/std must be exactly 0 / 1 within numeric tolerance.
        assert float(np.mean(values)) == pytest.approx(0.0, abs=1e-9)
        assert float(np.std(values)) == pytest.approx(1.0, abs=1e-9)


def test_standardise_passes_through_none_ref_delta(mocker) -> None:
    """When the baseline has no ref_delta calibration, standardise() returns None for ref_delta."""
    n = 5
    completions = [_make_completion_for_seed(s) for s in range(n)]
    rows = [_row(f"p-{i}") for i in range(n)]
    lm = _fake_lm(mocker, completions)
    baseline = build_baseline(lm, rows, ref_lm=None, min_valid=3)

    # Construct an eval-time MiaFeatures that *does* carry a ref_delta value.
    feats_with_ref = MiaFeatures(
        loss=0.5, min_k=-0.4, min_k_pp=-0.2, zlib_ratio=1.0, ref_delta=-0.3
    )
    out = standardise(feats_with_ref, baseline)
    assert out["ref_delta"] is None
    # Other four still standardise normally.
    for key in ("loss", "min_k", "min_k_pp", "zlib_ratio"):
        assert isinstance(out[key], float)
        assert math.isfinite(out[key])


def test_standardise_passes_through_none_when_features_ref_delta_is_none() -> None:
    """If the eval row has ref_delta=None (ref call failed) → standardised ref_delta is None,
    even when the baseline has a calibrated ref_delta mean/std."""
    baseline = ControlBaseline(
        model="m",
        n_valid=10,
        feature_means={
            "loss": 1.0,
            "min_k": -0.5,
            "min_k_pp": -0.2,
            "zlib_ratio": 0.5,
            "ref_delta": -0.1,
        },
        feature_stds={
            "loss": 0.2,
            "min_k": 0.1,
            "min_k_pp": 0.1,
            "zlib_ratio": 0.05,
            "ref_delta": 0.1,
        },
        is_calibrated=True,
        min_valid=5,
    )
    feats = MiaFeatures(
        loss=1.1, min_k=-0.4, min_k_pp=-0.1, zlib_ratio=0.55, ref_delta=None
    )
    out = standardise(feats, baseline)
    assert out["ref_delta"] is None
    # The other four should be (raw - mean) / std.
    assert out["loss"] == pytest.approx((1.1 - 1.0) / 0.2, abs=1e-9)
    assert out["min_k"] == pytest.approx((-0.4 - -0.5) / 0.1, abs=1e-9)
    assert out["min_k_pp"] == pytest.approx((-0.1 - -0.2) / 0.1, abs=1e-9)
    assert out["zlib_ratio"] == pytest.approx((0.55 - 0.5) / 0.05, abs=1e-9)


def test_standardise_floors_std_at_1e_minus_6() -> None:
    """A baseline with std=0 uses the 1e-6 floor — output is finite, no div-by-zero."""
    baseline = ControlBaseline(
        model="m",
        n_valid=10,
        feature_means={
            "loss": 1.0,
            "min_k": -0.5,
            "min_k_pp": -0.2,
            "zlib_ratio": 0.5,
            "ref_delta": None,
        },
        feature_stds={
            "loss": 0.0,
            "min_k": 0.0,
            "min_k_pp": 0.0,
            "zlib_ratio": 0.0,
            "ref_delta": None,
        },
        is_calibrated=True,
        min_valid=5,
    )
    feats = MiaFeatures(
        loss=1.0 + 1e-9,
        min_k=-0.5,
        min_k_pp=-0.2,
        zlib_ratio=0.5,
        ref_delta=None,
    )
    out = standardise(feats, baseline)
    for key in ("loss", "min_k", "min_k_pp", "zlib_ratio"):
        assert math.isfinite(out[key])
    # 1e-9 / 1e-6 == 1e-3 (loose tol since 1e-9 is at fp64 round-off)
    assert out["loss"] == pytest.approx(1e-3, abs=1e-6)
    assert out["ref_delta"] is None
