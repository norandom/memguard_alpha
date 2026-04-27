"""Tests for `src.mia.features.compute_mia_features`.

Validates Requirements 4.1, 4.2, 4.3 and design.md → "Components and
Interfaces → mia → mia.features".

These tests assert exact numerics (within 1e-9) on hand-computable fixtures
to keep the feature math frozen against accidental drift.
"""

from __future__ import annotations

import dataclasses
import math
import zlib

import numpy as np
import pytest

from src.core.nvidia_lm import TokenLogprob
from src.mia.features import LOGPROB_FLOOR, MiaFeatures, compute_mia_features


def _tlp(logprob: float, top_logprobs: list[float] | None = None) -> TokenLogprob:
    """Build a TokenLogprob with a list of top-K logprob floats.

    The actual NVIDIA shape is `list[dict]` with `token` and `logprob` keys.
    For Min-K%++ we only consume the `logprob` floats from each candidate.
    """
    if top_logprobs is None:
        # Default: a singleton distribution that puts mass on the realised token.
        top_logprobs = [logprob]
    tops = [{"token": "x", "logprob": float(lp)} for lp in top_logprobs]
    return TokenLogprob(token="x", logprob=float(logprob), top_logprobs=tops)


# --- Loss --------------------------------------------------------------


def test_mia_features_loss_is_mean_neg_logprob() -> None:
    """`loss` is the mean negative logprob of the realised tokens."""
    logprobs = [_tlp(-0.1), _tlp(-0.2), _tlp(-0.5), _tlp(-0.1), _tlp(-0.1)]
    feats = compute_mia_features(
        response="response", logprobs=logprobs, ref_logprobs=None, k=0.4
    )
    # mean of [-0.1, -0.2, -0.5, -0.1, -0.1] = -0.2; loss = -mean = 0.2
    assert feats.loss == pytest.approx(0.2, abs=1e-9)


# --- Min-K% ------------------------------------------------------------


def test_mia_features_min_k_is_mean_of_bottom_k() -> None:
    """`min_k` is the mean of the lowest `int(len*k)` clipped logprobs."""
    logprobs = [_tlp(-0.1), _tlp(-0.2), _tlp(-0.5), _tlp(-0.1), _tlp(-0.1)]
    feats = compute_mia_features(
        response="r", logprobs=logprobs, ref_logprobs=None, k=0.4
    )
    # bottom int(5 * 0.4) = 2 logprobs are [-0.5, -0.2] → mean = -0.35
    assert feats.min_k == pytest.approx(-0.35, abs=1e-9)


def test_mia_features_min_k_uses_at_least_one_token() -> None:
    """`min_k` floors the bottom-K count at 1, even when len*k rounds to 0."""
    logprobs = [_tlp(-0.4), _tlp(-0.1)]
    # int(2 * 0.2) == 0 → must floor to 1
    feats = compute_mia_features(
        response="r", logprobs=logprobs, ref_logprobs=None, k=0.2
    )
    assert feats.min_k == pytest.approx(-0.4, abs=1e-9)


# --- Min-K%++ ----------------------------------------------------------


def test_mia_features_min_k_pp_z_scores() -> None:
    """`min_k_pp` averages bottom-K z-scores derived from `top_logprobs`."""
    # Three positions with deterministic top_logprob distributions.
    # ddof=0 std (numpy default).
    pos0_top = [-1.0, 0.0, 1.0]    # mean=0, std=sqrt(2/3)
    pos1_top = [-2.0, 0.0, 2.0]    # mean=0, std=sqrt(8/3)
    pos2_top = [-3.0, 0.0, 3.0]    # mean=0, std=sqrt(6)
    logprobs = [
        _tlp(-1.0, pos0_top),  # z = -1.0 / sqrt(2/3)
        _tlp(0.0, pos1_top),   # z = 0.0
        _tlp(1.0, pos2_top),   # z =  1.0 / sqrt(6)
    ]
    z0 = -1.0 / float(np.std(pos0_top))
    z1 = 0.0
    z2 = 1.0 / float(np.std(pos2_top))

    # k=0.7 → bottom int(3*0.7)=2 → smallest two z values
    bottom_two = sorted([z0, z1, z2])[:2]
    expected = float(np.mean(bottom_two))

    feats = compute_mia_features(
        response="abc", logprobs=logprobs, ref_logprobs=None, k=0.7
    )
    assert feats.min_k_pp == pytest.approx(expected, abs=1e-9)


def test_mia_features_min_k_pp_floors_zero_std() -> None:
    """Std == 0 (degenerate top_logprobs) is floored at 1e-6 to avoid div-by-zero."""
    # All top_logprobs identical → std = 0 → floor to 1e-6.
    logprobs = [
        _tlp(-0.5, [-0.5, -0.5, -0.5]),
        _tlp(-0.5, [-0.5, -0.5, -0.5]),
    ]
    feats = compute_mia_features(
        response="ok", logprobs=logprobs, ref_logprobs=None, k=0.5
    )
    # numerator = clipped_logprob - mean = -0.5 - (-0.5) = 0 → z = 0
    assert feats.min_k_pp == pytest.approx(0.0, abs=1e-9)
    assert math.isfinite(feats.min_k_pp)


# --- zlib ratio --------------------------------------------------------


def test_mia_features_zlib_ratio_uses_zlib_compress() -> None:
    """`zlib_ratio = -sum(clipped_logprobs) / len(zlib.compress(response, 9))`."""
    response = "the quick brown fox jumps over the lazy dog"
    logprobs = [_tlp(-0.1), _tlp(-0.2), _tlp(-0.5), _tlp(-0.1), _tlp(-0.1)]
    feats = compute_mia_features(
        response=response, logprobs=logprobs, ref_logprobs=None, k=0.4
    )
    expected = -(-0.1 + -0.2 + -0.5 + -0.1 + -0.1) / max(
        len(zlib.compress(response.encode("utf-8"), 9)), 1
    )
    assert feats.zlib_ratio == pytest.approx(expected, abs=1e-12)


def test_mia_features_zlib_ratio_zero_on_empty_response() -> None:
    """Empty response → `zlib_ratio == 0.0` (design tie-break)."""
    logprobs = [_tlp(-0.1), _tlp(-0.2)]
    feats = compute_mia_features(
        response="", logprobs=logprobs, ref_logprobs=None, k=0.5
    )
    assert feats.zlib_ratio == 0.0


# --- ref_delta ---------------------------------------------------------


def test_mia_features_ref_delta_none_when_ref_logprobs_none() -> None:
    """`ref_logprobs is None` → `ref_delta is None`."""
    logprobs = [_tlp(-0.1), _tlp(-0.2)]
    feats = compute_mia_features(
        response="r", logprobs=logprobs, ref_logprobs=None, k=0.5
    )
    assert feats.ref_delta is None


def test_mia_features_ref_delta_is_loss_self_minus_loss_ref() -> None:
    """`ref_delta = loss(self) - loss(ref)` on clipped logprobs."""
    self_lp = [_tlp(-0.1), _tlp(-0.3)]   # loss_self = 0.2
    ref_lp = [_tlp(-0.5), _tlp(-1.5)]    # loss_ref  = 1.0
    feats = compute_mia_features(
        response="r", logprobs=self_lp, ref_logprobs=ref_lp, k=0.5
    )
    assert feats.ref_delta == pytest.approx(0.2 - 1.0, abs=1e-9)


# --- clipping ----------------------------------------------------------


def test_mia_features_clips_to_floor() -> None:
    """A logprob below `LOGPROB_FLOOR=-30.0` is clipped before averaging."""
    assert LOGPROB_FLOOR == -30.0
    # logprob = -100.0 should be clipped to -30.0 before mean.
    logprobs = [_tlp(-100.0), _tlp(-10.0)]
    feats = compute_mia_features(
        response="r", logprobs=logprobs, ref_logprobs=None, k=0.5
    )
    # mean of [-30.0, -10.0] = -20.0 ; loss = 20.0
    assert feats.loss == pytest.approx(20.0, abs=1e-9)


# --- preconditions / errors -------------------------------------------


def test_mia_features_raises_on_empty_logprobs() -> None:
    """Empty logprobs is a precondition violation."""
    with pytest.raises(ValueError, match="empty"):
        compute_mia_features(
            response="r", logprobs=[], ref_logprobs=None, k=0.2
        )


def test_mia_features_raises_on_missing_top_logprobs() -> None:
    """A token with empty `top_logprobs` raises with the position index."""
    bad = TokenLogprob(token="x", logprob=-0.1, top_logprobs=[])
    logprobs = [_tlp(-0.1), bad]
    with pytest.raises(ValueError, match="1"):  # position index 1
        compute_mia_features(
            response="r", logprobs=logprobs, ref_logprobs=None, k=0.5
        )


# --- frozen dataclass --------------------------------------------------


def test_mia_features_returns_frozen_dataclass() -> None:
    """The return type is a frozen dataclass."""
    logprobs = [_tlp(-0.1), _tlp(-0.2)]
    feats = compute_mia_features(
        response="r", logprobs=logprobs, ref_logprobs=None, k=0.5
    )
    assert dataclasses.is_dataclass(feats)
    assert feats.__dataclass_params__.frozen is True
    with pytest.raises(dataclasses.FrozenInstanceError):
        feats.loss = 0.0  # type: ignore[misc]
