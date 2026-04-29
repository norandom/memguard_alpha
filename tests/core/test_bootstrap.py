"""Tests for src.core.bootstrap: bootstrap CI helper.

Covers requirements 6.1, 6.3, 6.5 from the honest-model-ranking spec:
- Bootstrap 95% CIs on accuracy and AUC (>= 1000 resamples).
- Fixed seed -> deterministic output.
- Degenerate-resample handling (e.g., AUC with one class collapsed).
"""

from __future__ import annotations

import logging
import math

import pytest

from src.core.bootstrap import bootstrap_ci


def test_bootstrap_ci_determinism() -> None:
    """Same seed -> identical (point, lo, hi) across two calls (Req 6.5)."""
    samples = list(range(50))
    statistic = sum

    a = bootstrap_ci(samples, statistic, n_resamples=1000, confidence=0.95, seed=0)
    b = bootstrap_ci(samples, statistic, n_resamples=1000, confidence=0.95, seed=0)

    assert a == b
    point_a, lo_a, hi_a = a
    # Sanity: bounds bracket the point (Req 6.1).
    assert lo_a <= point_a <= hi_a
    # The original sum is fixed by definition.
    assert point_a == sum(samples)


def test_bootstrap_ci_brackets_known_mean() -> None:
    """50/50 binary samples -> mean CI brackets 0.5 (Req 6.1)."""
    samples = [0] * 50 + [1] * 50

    def mean(s):
        return sum(s) / len(s)

    point, lo, hi = bootstrap_ci(
        samples, mean, n_resamples=1000, confidence=0.95, seed=0
    )

    assert math.isclose(point, 0.5, abs_tol=1e-12)
    assert lo < 0.5 < hi
    assert lo <= point <= hi


def test_bootstrap_ci_single_sample() -> None:
    """Single sample -> degenerate (point, point, point) (Req 6.1)."""
    samples = [42.0]
    def statistic(s):
        return float(s[0])

    point, lo, hi = bootstrap_ci(samples, statistic, n_resamples=1000, seed=0)

    assert point == 42.0
    assert lo == 42.0
    assert hi == 42.0


def test_bootstrap_ci_empty_samples_raises() -> None:
    """Empty input -> ValueError (Req 6.1 precondition)."""
    with pytest.raises(ValueError):
        bootstrap_ci([], sum, n_resamples=1000, seed=0)


def test_bootstrap_ci_drops_failing_resamples(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Resamples that raise ValueError are dropped with one WARNING (Req 6.3).

    Mimics the AUC-with-one-class case: the statistic raises whenever the
    resample contains only one class. We assert (a) the call still returns a
    valid CI bracket, and (b) exactly one WARNING is emitted per call,
    summarising the dropped count.
    """
    samples = [0] * 5 + [1] * 5

    call_count = {"n": 0}

    def picky_statistic(s):
        # Deterministically fail every other call to force drops.
        call_count["n"] += 1
        if len(set(s)) < 2:
            raise ValueError("only one class in resample")
        # Synthetic AUC-like score: fraction of 1s.
        return sum(s) / len(s)

    with caplog.at_level(logging.WARNING, logger="src.core.bootstrap"):
        point, lo, hi = bootstrap_ci(
            samples,
            picky_statistic,
            n_resamples=200,
            confidence=0.95,
            seed=0,
        )

    assert lo <= point <= hi
    # Exactly one WARNING per bootstrap_ci call about dropped resamples.
    drop_warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING
        and ("drop" in r.getMessage().lower() or "skipped" in r.getMessage().lower())
    ]
    assert len(drop_warnings) == 1, (
        f"expected exactly one drop warning, got {len(drop_warnings)}: "
        f"{[r.getMessage() for r in caplog.records]}"
    )


def test_bootstrap_ci_invariant_lo_le_point_le_hi() -> None:
    """Skewed sample -> lo <= point <= hi must always hold (Req 6.1)."""
    samples = [0] * 99 + [10]

    def mean(s):
        return sum(s) / len(s)

    point, lo, hi = bootstrap_ci(
        samples, mean, n_resamples=1000, confidence=0.95, seed=0
    )
    assert lo <= point <= hi
