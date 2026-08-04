"""Pure statistics for reducing a set of repeated model draws.

Everything here is a deterministic function of its arguments: no I/O, no
randomness, no global state, and no domain knowledge about what a draw means.
That is what lets a stored draw set be replayed into a bit-identical result
without re-querying the model.

The implementation is hand-rolled with the standard library and ``numpy``;
``scipy`` is intentionally avoided, matching :mod:`recall_guard.core.bootstrap`.
The normal quantile the Wilson interval needs comes from
:class:`statistics.NormalDist`, which agrees with the usual reference to within
5e-16 -- far inside anything that matters here.

Three choices are load-bearing and deliberately not the textbook ones, because
the textbook ones were measured against real draws and failed:

* **Location is never snapped to a lattice.** Snapping degrades accuracy on
  every measured component and biases the best-estimated one systematically,
  because its estimate sits near a bin edge and always rounds the same way.
  Snapping is a reporting convention, not an estimator.
* **Summation is exact.** Pairwise summation makes a mean depend on array order
  in the last bit, and a last-bit difference changes a persisted artifact hash.
* **An even-count median takes the lower order statistic**, never the midpoint
  of two draws -- a midpoint is a value the model never emitted.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum
from statistics import NormalDist

#: Scale factor making the median absolute deviation consistent with the
#: standard deviation of a normal sample.
MAD_TO_SIGMA = 1.4826

#: Divisor turning a lattice step into the smallest dispersion distinguishable
#: from zero at that resolution. See :func:`scale_floor`.
_QUANTIZATION_DIVISOR = math.sqrt(12.0)


class Tail(StrEnum):
    """Which tail an interval's confidence level refers to.

    This must be declared rather than assumed: the draw count needed to certify
    an agreement target differs substantially between the two conventions, so a
    feasibility check evaluated against the wrong one is meaningless.
    """

    ONE_SIDED = "one_sided"
    TWO_SIDED = "two_sided"


def _z(confidence: float, tail: Tail) -> float:
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1); got {confidence!r}")
    alpha = 1.0 - confidence
    quantile = 1.0 - (alpha / 2.0 if tail is Tail.TWO_SIDED else alpha)
    return NormalDist().inv_cdf(quantile)


def _exact_mean(values: Sequence[float]) -> float:
    """Order-independent mean.

    ``math.fsum`` is exactly rounded, so this is invariant under permutation by
    construction -- unlike ``sum(sorted(x))``, which merely happens to be stable
    for a given input.
    """
    return math.fsum(values) / len(values)


def wilson_interval(
    k: int,
    n: int,
    *,
    confidence: float = 0.95,
    tail: Tail = Tail.TWO_SIDED,
    continuity: bool = False,
) -> tuple[float, float] | None:
    """Score interval for a binomial proportion, or ``None`` when ``n`` is zero.

    Inverting the score test rather than the Wald test keeps the interval inside
    ``[0, 1]`` and, critically, non-degenerate at ``k == n``. The Wald interval
    collapses to zero width exactly there, which at high agreement is the
    typical case -- it would report certainty from a couple of dozen draws.

    Parameters
    ----------
    k:
        Successes observed. Must satisfy ``0 <= k <= n``.
    n:
        Draws observed.
    continuity:
        Apply the Newcombe continuity correction, widening the interval.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative; got {n}")
    if not 0 <= k <= n:
        raise ValueError(f"k must satisfy 0 <= k <= n; got k={k}, n={n}")
    if n == 0:
        return None

    z = _z(confidence, tail)
    p = k / n
    z2 = z * z
    denominator = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denominator
    spread = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denominator
    lo, hi = centre - spread, centre + spread

    if continuity:
        correction = 1.0 / (2 * n) / denominator
        lo, hi = lo - correction, hi + correction

    return (max(0.0, lo), min(1.0, hi))


def smallest_certifiable_n(
    target: float,
    *,
    confidence: float = 0.95,
    tail: Tail = Tail.TWO_SIDED,
    limit: int = 100_000,
) -> int:
    """Fewest unanimous draws whose interval's lower bound reaches ``target``.

    Unanimity is the best case, so this is a hard floor: below it no observed
    agreement can certify the target, and a configuration requesting fewer draws
    can never succeed no matter what the model returns. Surfacing it at
    construction turns a silently-unreachable setting into an error.
    """
    if not 0.0 < target < 1.0:
        raise ValueError(f"target must be in (0, 1); got {target!r}")
    for n in range(1, limit + 1):
        bounds = wilson_interval(n, n, confidence=confidence, tail=tail)
        if bounds is not None and bounds[0] >= target:
            return n
    raise ValueError(
        f"target {target} is not certifiable within {limit} draws at "
        f"confidence={confidence} ({tail.value})"
    )


def snap_to_grid(value: float, grid: float) -> float:
    """Round ``value`` onto a lattice of step ``grid``, half away from zero.

    Deliberately does **not** divide by ``grid``. That division is inexact in
    binary in a value-dependent way -- ``0.85 / 0.1`` is exactly ``8.5`` while
    ``0.95 / 0.1`` is ``9.499999999999998`` -- so the tie direction ends up
    depending on the value rather than on the rule, and the obvious spellings
    disagree with one another. Working in integer lattice units avoids it.
    """
    if grid <= 0:
        raise ValueError(f"grid must be positive; got {grid!r}")
    units = (Decimal(repr(abs(value))) / Decimal(repr(grid))).quantize(
        Decimal(1), rounding=ROUND_HALF_UP
    )
    return math.copysign(float(units * Decimal(repr(grid))), value)


def grid_adherence(values: Sequence[float], grid: float, *, tolerance: float = 1e-9) -> float:
    """Fraction of ``values`` that actually lie on the declared lattice.

    Reported so a caller who declares a lattice the data does not follow finds
    out, instead of silently receiving mis-snapped results. A continuous
    quantity scores near zero here at any lattice.
    """
    if grid <= 0:
        raise ValueError(f"grid must be positive; got {grid!r}")
    if not values:
        raise ValueError("values must be non-empty")
    on = sum(1 for v in values if abs(v - snap_to_grid(v, grid)) <= tolerance)
    return on / len(values)


def _median(sorted_values: Sequence[float]) -> float:
    """Lower-order-statistic median: always an emitted value."""
    return sorted_values[(len(sorted_values) - 1) // 2]


def scale_floor(values: Sequence[float], *, grid: float | None = None) -> float:
    """Robust scale estimate, floored at the lattice's resolution limit.

    The floor exists because a concentrated lattice-valued component can drive
    the median absolute deviation to exactly zero, leaving the usual robust
    scale undefined. It is an *identifiability* floor, not an estimate of
    quantization noise: any true dispersion far below one lattice step produces
    observations on one or two lattice points and is indistinguishable from
    zero, so an estimate below that level carries no information.

    Two caveats worth knowing before relying on it. It is inert wherever the
    estimate was already well-defined -- it binds only when the deviation is
    zero. And it only helps because the declared lattice is coarser than the
    emitted one; declare the true lattice and the undefined case returns.
    """
    if not values:
        raise ValueError("values must be non-empty")
    ordered = sorted(values)
    centre = _median(ordered)
    mad = _median(sorted(abs(v - centre) for v in values))
    sigma = MAD_TO_SIGMA * mad
    if grid is None:
        return sigma
    if grid <= 0:
        raise ValueError(f"grid must be positive; got {grid!r}")
    return max(sigma, grid / _QUANTIZATION_DIVISOR)


@dataclass(frozen=True)
class MultimodalVerdict:
    """Outcome of the separated-cluster check for one component.

    ``separated`` is the gate: when it is true the component holds two clusters
    with a genuine gap between them, there is no single location to estimate,
    and a location estimator must not be run at all.
    """

    separated: bool
    lower_mass: float
    upper_mass: float
    trough_mass: float
    gap: tuple[float, float] | None


def _occupancy(values: Sequence[float], grid: float) -> tuple[list[float], list[float]]:
    """Lattice positions present in ``values`` and their mass fractions."""
    counts: dict[float, int] = {}
    for value in values:
        snapped = snap_to_grid(value, grid)
        counts[snapped] = counts.get(snapped, 0) + 1
    positions = sorted(counts)
    total = len(values)
    return positions, [counts[p] / total for p in positions]


def detect_multimodal(
    values: Sequence[float],
    *,
    grid: float | None,
    mass_min: float = 0.25,
    trough_steps: int = 3,
    density_ratio: float = 10.0,
) -> MultimodalVerdict | None:
    """Detect two clusters separated by a sparse gap.

    Returns ``None`` when no lattice is declared -- an explicit "did not run",
    never a silent "found nothing". A continuous quantity has no lattice, so a
    caller must be able to tell the two apart.

    The rule is defined directly on the lattice rather than by a classical
    unimodality test. Those tests assume a continuous distribution, and on
    heavily-tied lattice data they measure tie mass instead of modality -- badly
    enough that on the measured corpus the most sharply converged component
    scores as *more* multimodal than the genuinely split one.

    Every threshold is a parameter because all three were tuned against a single
    measurement date. Detects separated clusters only: two overlapping modes
    with no gap between them are invisible to it, which is a known and accepted
    false-negative class.

    Parameters
    ----------
    mass_min:
        Minimum share of draws each cluster must hold.
    trough_steps:
        Minimum width of the gap, in lattice steps.
    density_ratio:
        How much denser the taller cluster peak must be than the busiest bin
        inside the gap.
    """
    if not values:
        raise ValueError("values must be non-empty")
    if grid is None:
        return None
    if grid <= 0:
        raise ValueError(f"grid must be positive; got {grid!r}")
    if not 0.0 < mass_min <= 0.5:
        raise ValueError(f"mass_min must be in (0, 0.5]; got {mass_min!r}")
    if trough_steps < 1:
        raise ValueError(f"trough_steps must be >= 1; got {trough_steps!r}")

    positions, masses = _occupancy(values, grid)
    best: tuple[float, float, int, int] | None = None
    best_verdict: MultimodalVerdict | None = None

    for i in range(len(positions) - 1):
        lower_mass = math.fsum(masses[: i + 1])
        if lower_mass < mass_min:
            continue
        for j in range(i + 1, len(positions)):
            # Gap width is measured in lattice steps, not in occupied bins, so a
            # sparsely-sampled gap still counts as wide.
            steps = round((positions[j] - positions[i]) / grid) - 1
            if steps < trough_steps:
                continue
            upper_mass = math.fsum(masses[j:])
            if upper_mass < mass_min:
                continue
            inner = masses[i + 1 : j]
            trough_peak = max(inner) if inner else 0.0
            peak = max(max(masses[: i + 1]), max(masses[j:]))
            if peak < density_ratio * trough_peak:
                continue
            trough_mass = math.fsum(inner)
            # Deterministic selection: sparsest gap, then the sharpest density
            # contrast, then the lowest indices. Every key is a function of the
            # values alone, so a reordered input selects the same split.
            key = (trough_mass, -(peak - density_ratio * trough_peak), i, j)
            if best is None or key < best:
                best = key
                best_verdict = MultimodalVerdict(
                    separated=True,
                    lower_mass=lower_mass,
                    upper_mass=upper_mass,
                    trough_mass=trough_mass,
                    gap=(positions[i], positions[j]),
                )

    if best_verdict is not None:
        return best_verdict
    return MultimodalVerdict(
        separated=False,
        lower_mass=0.0,
        upper_mass=0.0,
        trough_mass=0.0,
        gap=None,
    )


def robust_location(
    values: Sequence[float],
    *,
    mode: str = "median",
    trim: float = 0.25,
) -> float:
    """Reduce ``values`` to one location, independent of their order.

    ``mode`` is one of ``"mean"``, ``"median"``, or ``"trimmed"``.

    Note what this cannot do: on a component whose draws form two separated
    clusters there is no single location to estimate, and no symmetric trim
    fraction escapes the gap between them -- trimming converges toward the
    median, not toward a mode. Callers must run the multimodality check first
    and skip this entirely for a flagged component.
    """
    if not values:
        raise ValueError("values must be non-empty")

    if mode == "mean":
        return _exact_mean(sorted(values))
    if mode == "median":
        return _median(sorted(values))
    if mode != "trimmed":
        raise ValueError(f"unknown mode {mode!r}; expected mean, median, or trimmed")

    if not 0.0 <= trim < 0.5:
        raise ValueError(f"trim must be in [0, 0.5); got {trim!r}")
    ordered = sorted(values)
    cut = math.floor(len(ordered) * trim)
    core = ordered[cut : len(ordered) - cut] or ordered
    return _exact_mean(core)


__all__ = [
    "MAD_TO_SIGMA",
    "MultimodalVerdict",
    "detect_multimodal",
    "Tail",
    "grid_adherence",
    "robust_location",
    "scale_floor",
    "smallest_certifiable_n",
    "snap_to_grid",
    "wilson_interval",
]
