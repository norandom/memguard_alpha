"""Bootstrap confidence-interval helper for the honest-model-ranking harness.

Implements the design's ``core.bootstrap`` interface (Req 6.1, 6.3, 6.5):

- ``bootstrap_ci(samples, statistic, n_resamples=1000, confidence=0.95, seed=0)``
  returns ``(point, lo, hi)`` where ``point = statistic(samples)`` is computed
  once on the original sample and ``lo`` / ``hi`` come from the percentile
  bootstrap over ``n_resamples`` resamples drawn with replacement.
- Determinism is guaranteed by ``numpy.random.default_rng(seed)``; resamples are
  drawn via index sampling so that ``samples`` may contain arbitrary Python
  objects (the design allows ``Sequence[T]`` for any T).
- Degenerate cases:
    * ``len(samples) == 0`` -> ``ValueError``.
    * ``len(samples) == 1`` -> ``(point, point, point)``.
    * Resamples on which ``statistic`` raises ``ValueError`` or returns a
      non-finite value (``NaN`` / ``inf``) are dropped; a single
      ``logging.WARNING`` per ``bootstrap_ci`` call summarises the count.
    * If every resample is dropped -> ``(point, point, point)`` plus a
      WARNING describing the degenerate result.

The implementation is hand-rolled with ``numpy``; ``scipy`` is intentionally
avoided per the "Build vs Adopt" design entry on bootstrap.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from typing import TypeVar

import numpy as np

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _is_finite(value: float) -> bool:
    """Return True iff ``value`` is a finite real number (not NaN, not inf)."""
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def bootstrap_ci(
    samples: Sequence[T],
    statistic: Callable[[Sequence[T]], float],
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Compute a percentile bootstrap CI for ``statistic`` over ``samples``.

    Args:
        samples: Sequence of arbitrary objects (may be ints, floats, tuples,
            dicts, dataclasses, etc.). Resampling uses index sampling, so the
            element type does not need to be numpy-friendly.
        statistic: Callable taking a resampled sequence and returning a float.
        n_resamples: Number of bootstrap resamples (>=1). Default 1000 matches
            the harness's Req 6.1 minimum.
        confidence: Two-sided confidence level in (0, 1). Default 0.95.
        seed: Seed for ``numpy.random.default_rng``; same seed -> same output.

    Returns:
        Tuple ``(point, lo, hi)``:

        - ``point = statistic(samples)`` (computed once on the original).
        - ``lo``, ``hi`` are the lower / upper percentile bounds.
        - Postcondition: ``lo <= point <= hi`` (clamped on tiny float drift).

    Raises:
        ValueError: if ``samples`` is empty, ``n_resamples < 1``, or
            ``confidence`` is outside ``(0, 1)``.
    """
    n = len(samples)
    if n == 0:
        raise ValueError("bootstrap_ci: 'samples' must contain at least one element.")
    if n_resamples < 1:
        raise ValueError(
            f"bootstrap_ci: 'n_resamples' must be >= 1, got {n_resamples}."
        )
    if not (0.0 < confidence < 1.0):
        raise ValueError(
            f"bootstrap_ci: 'confidence' must be in (0, 1), got {confidence}."
        )

    point = float(statistic(samples))

    # Single-sample short-circuit: every resample is identical, so the CI is
    # degenerate by construction. Skip work and return immediately (Req 6.1
    # precondition note).
    if n == 1:
        return point, point, point

    rng = np.random.default_rng(seed)

    stats: list[float] = []
    dropped = 0
    for _ in range(n_resamples):
        # Index sampling keeps the element type unconstrained: T may be any
        # arbitrary Python object, not necessarily a numpy scalar.
        idx = rng.integers(low=0, high=n, size=n)
        resample = [samples[int(i)] for i in idx]
        try:
            value = statistic(resample)
        except ValueError:
            # Statistic refused this resample (e.g., AUC with a single class).
            # Drop and continue rather than crashing the whole CI computation.
            dropped += 1
            continue
        if not _is_finite(value):
            dropped += 1
            continue
        stats.append(float(value))

    if dropped > 0:
        # Exactly one warning per call, regardless of how many resamples we
        # dropped, so logs do not flood under heavy degeneracy.
        logger.warning(
            "bootstrap_ci: dropped %d / %d resamples where the statistic "
            "raised or returned a non-finite value.",
            dropped,
            n_resamples,
        )

    if not stats:
        # All resamples failed: the CI is undefined. Surface the degeneracy
        # via a separate warning and collapse to the point estimate so callers
        # do not get NaN bounds.
        logger.warning(
            "bootstrap_ci: every resample failed; collapsing CI to the point "
            "estimate. Statistic likely undefined for this sample distribution."
        )
        return point, point, point

    arr = np.asarray(stats, dtype=float)
    alpha = 1.0 - confidence
    lo_pct = (alpha / 2.0) * 100.0
    hi_pct = (1.0 - alpha / 2.0) * 100.0
    lo = float(np.percentile(arr, lo_pct))
    hi = float(np.percentile(arr, hi_pct))

    # Clamp tiny floating-point inversions so the documented postcondition
    # (lo <= point <= hi) always holds without surprising the caller.
    if lo > point:
        lo = point
    if hi < point:
        hi = point

    return point, lo, hi
