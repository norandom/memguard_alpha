"""Pure statistics for reducing a set of repeated model draws.

Everything here is a deterministic function of its arguments: no I/O, no
randomness, no global state, and no domain knowledge about what a draw means.
That is what lets a stored draw set be replayed into a bit-identical result
without re-querying the model.

The implementation is hand-rolled with the standard library alone; ``scipy`` is
intentionally avoided, matching :mod:`recall_guard.core.bootstrap`. Numpy
scalars and arrays are accepted -- every numeric argument is coerced with
``float()`` before use, because ``repr(np.float64(0.15))`` is
``'np.float64(0.15)'`` under numpy 2 and would otherwise reach ``Decimal``.
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
        # Newcombe's correction adjusts the pivot *before* inversion, which
        # changes the radicand. Applying it as a constant shift of the already
        # inverted bound is a different, anti-conservative interval: it sits
        # inside the exact one across most of the parameter space, and its true
        # coverage dips below nominal -- failing the one job a continuity
        # correction exists to do.
        span = 2.0 * (n + z2)
        if k == 0:
            lo = 0.0
        else:
            radicand = z2 - 2 - 1.0 / n + 4 * p * (n * (1 - p) + 1)
            lo = (2 * n * p + z2 - 1 - z * math.sqrt(max(radicand, 0.0))) / span
        if k == n:
            hi = 1.0
        else:
            radicand = z2 + 2 - 1.0 / n + 4 * p * (n * (1 - p) - 1)
            hi = (2 * n * p + z2 + 1 + z * math.sqrt(max(radicand, 0.0))) / span

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


def smallest_detectable_split_n(
    *,
    cluster_positions: int,
    min_cluster_draws: int = 8,
    min_cluster_density: float = 1.5,
) -> int:
    """Fewest draws at which :func:`detect_multimodal` *can* flag a split.

    The companion to :func:`smallest_certifiable_n`, and the reason both exist:
    **agreement precision and split detection need different sample sizes**, and
    sizing for the first silently under-sizes for the second. Agreement is where
    the reported confidence lives; component splits are where a silently wrong
    answer lives.

    This is a **necessary condition, not a sufficient one** -- exactly as its
    companion is a floor under unanimity rather than a promise. Below the value
    returned here the density guard cannot be satisfied at all, so a split of
    that shape is undetectable no matter how clean the data. Above it, detection
    becomes *possible*; whether it fires still depends on sampling noise in the
    trough and in the cluster masses. Measured on a corpus whose split is
    unambiguous at full size, detection still missed roughly 3% of 64-draw
    subsamples.

    Parameters
    ----------
    cluster_positions:
        How many lattice positions the two clusters together occupy. Wider
        clusters need more draws to reach the same density.
    """
    if cluster_positions < 2:
        raise ValueError(
            f"cluster_positions must be >= 2 for a split; got {cluster_positions}"
        )
    if min_cluster_draws < 2:
        raise ValueError(f"min_cluster_draws must be >= 2; got {min_cluster_draws}")
    if min_cluster_density < 1.0:
        raise ValueError(
            f"min_cluster_density must be >= 1; got {min_cluster_density}"
        )
    return max(min_cluster_draws, math.ceil(min_cluster_density * cluster_positions))


def _checked_grid(grid: float) -> float:
    """Coerce and validate a lattice step.

    ``not (grid > 0)`` rather than ``grid <= 0`` because every comparison
    against NaN is false, so the naive guard lets NaN through and silently
    poisons every downstream result.
    """
    grid = float(grid)
    if not (grid > 0) or not math.isfinite(grid):
        raise ValueError(f"grid must be a positive finite number; got {grid!r}")
    return grid


def snap_to_grid(value: float, grid: float) -> float:
    """Round ``value`` onto a lattice of step ``grid``, half away from zero.

    Deliberately does **not** divide by ``grid``. That division is inexact in
    binary in a value-dependent way -- ``0.85 / 0.1`` is exactly ``8.5`` while
    ``0.95 / 0.1`` is ``9.499999999999998`` -- so the tie direction ends up
    depending on the value rather than on the rule, and the obvious spellings
    disagree with one another. Working in integer lattice units avoids it.
    """
    grid = _checked_grid(grid)
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"value must be finite; got {value!r}")
    units = (Decimal(repr(abs(value))) / Decimal(repr(grid))).quantize(
        Decimal(1), rounding=ROUND_HALF_UP
    )
    return math.copysign(float(units * Decimal(repr(grid))), value)


def grid_adherence(values: Sequence[float], grid: float, *, tolerance: float = 1e-9) -> float:
    """Fraction of ``values`` that actually lie on the declared lattice.

    Reported so a caller who declares a lattice the data does not follow finds
    out, instead of silently receiving mis-snapped results. A continuous
    quantity scores near zero here at any lattice.

    A non-finite draw counts as off-lattice rather than raising: this reports on
    data quality, so it has to survive the bad data it exists to describe.
    """
    grid = _checked_grid(grid)
    if not (tolerance >= 0) or not math.isfinite(tolerance):
        raise ValueError(f"tolerance must be non-negative and finite; got {tolerance!r}")
    if len(values) == 0:
        raise ValueError("values must be non-empty")
    on = 0
    for raw in values:
        v = float(raw)
        if not math.isfinite(v):
            continue
        if abs(v - snap_to_grid(v, grid)) <= tolerance:
            on += 1
    return on / len(values)


def _finite_floats(values: Sequence[float], *, what: str) -> list[float]:
    """Coerce to plain finite floats, normalising signed zero.

    Estimators **reject** non-finite input rather than dropping it: a NaN makes
    ``sorted`` order-dependent, which would silently break the order-independence
    the replay contract rests on. (Reporting functions such as
    :func:`grid_adherence` drop instead, because their job is to survive the bad
    data they describe. Estimators have no such excuse.)

    ``+ 0.0`` maps ``-0.0`` to ``0.0``. The two compare equal, so a stable sort
    emits whichever arrived first -- and they serialise differently, which is
    exactly the last-bit difference that changes a persisted artifact hash.
    """
    out: list[float] = []
    for raw in values:
        v = float(raw)
        if not math.isfinite(v):
            raise ValueError(f"{what} must contain only finite values; got {v!r}")
        out.append(v + 0.0)
    return out


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

    Two caveats worth knowing before relying on it. It binds whenever
    ``1.4826 * MAD < grid / sqrt(12)`` -- that is, for every ``MAD`` below
    roughly ``0.195 * grid``, not only when the deviation is exactly zero. So on
    a sharply concentrated component it can inflate a small but perfectly
    well-defined estimate. And it only helps at all because the declared lattice
    is coarser than the emitted one; declare the true lattice and the undefined
    case returns.
    """
    if len(values) == 0:
        raise ValueError("values must be non-empty")
    values = _finite_floats(values, what="values")
    ordered = sorted(values)
    centre = _median(ordered)
    mad = _median(sorted(abs(v - centre) for v in values))
    sigma = MAD_TO_SIGMA * mad
    if grid is None:
        return sigma
    return max(sigma, _checked_grid(grid) / _QUANTIZATION_DIVISOR)


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


def _occupancy(values: Sequence[float], grid: float) -> tuple[list[float], list[int]]:
    """Lattice positions present in ``values`` and their integer counts.

    Counts rather than mass fractions: integer sums are exact, so the search
    below makes identical decisions regardless of accumulation order. Fractions
    are formed once, at the end, for reporting only.

    ``+ 0.0`` normalises ``-0.0`` to ``0.0``. The two are ``==`` and hash-equal
    so they already collapse into one bin, but a dict keeps whichever key
    arrived first -- and the two serialise differently, which would make the
    reported gap order-dependent in its persisted form.
    """
    counts: dict[float, int] = {}
    for value in values:
        snapped = snap_to_grid(value, grid) + 0.0
        counts[snapped] = counts.get(snapped, 0) + 1
    positions = sorted(counts)
    return positions, [counts[p] for p in positions]


def detect_multimodal(
    values: Sequence[float],
    *,
    grid: float | None,
    mass_min: float = 0.25,
    trough_steps: int = 3,
    density_ratio: float = 10.0,
    min_draws: int = 8,
    min_cluster_density: float = 1.5,
) -> MultimodalVerdict | None:
    """Detect two clusters separated by a sparse gap.

    Returns ``None`` when the check **could not run** -- never a silent "found
    nothing". There are three such cases, and a caller who needs to tell them
    apart should read the lattice adherence alongside this:

    * no lattice was declared (a continuous quantity has none);
    * fewer than ``min_draws`` draws, which cannot evidence two clusters;
    * the clusters are too thinly populated for a gap between them to mean
      anything (see below).

    The rule is defined directly on the lattice rather than by a classical
    unimodality test. Those assume a continuous distribution, and on heavily
    tied lattice data they measure tie mass instead of modality -- badly enough
    that on the measured corpus the most sharply converged component scores as
    *more* multimodal than the genuinely split one.

    **The sparsity guard matters.** ``trough_steps`` counts lattice steps, not
    draws, so on a lattice much finer than the sampling, empty runs occur
    everywhere by chance; the density test is vacuous there because an empty gap
    has no peak to compare against. Measured, a unimodal normal at a 0.001
    lattice flagged on every single subsample.

    What separates the two regimes is not how wide the gap is but how *dense the
    clusters are*: a real cluster stacks many draws onto few lattice positions,
    while a spurious one is a scatter of singletons whose gaps are ordinary
    spacing. So each side of the split must average at least
    ``min_cluster_density`` draws per occupied position.

    Every threshold is a parameter because all of them were tuned against a
    single measurement date. Detects separated clusters only: two overlapping
    modes with no gap between them are invisible to it, a known and accepted
    false-negative class.

    Parameters
    ----------
    mass_min:
        Minimum share of draws each cluster must hold.
    trough_steps:
        Minimum width of the gap, in lattice steps.
    density_ratio:
        How much denser the taller cluster peak must be than the busiest bin
        inside the gap. Note this binds only when the gap is non-empty; at
        realistic draw counts most detections win on a completely empty gap.
    min_draws:
        Below this many draws the check does not run.
    min_cluster_density:
        Minimum draws per occupied lattice position within each cluster.
    """
    if len(values) == 0:
        raise ValueError("values must be non-empty")
    if grid is None:
        return None
    grid = _checked_grid(grid)
    if not 0.0 < mass_min <= 0.5:
        raise ValueError(f"mass_min must be in (0, 0.5]; got {mass_min!r}")
    if trough_steps < 1:
        raise ValueError(f"trough_steps must be >= 1; got {trough_steps!r}")
    if density_ratio < 0:
        raise ValueError(f"density_ratio must be >= 0; got {density_ratio!r}")
    if min_draws < 2:
        raise ValueError(f"min_draws must be >= 2; got {min_draws!r}")
    if min_cluster_density < 1.0:
        raise ValueError(
            f"min_cluster_density must be >= 1; got {min_cluster_density!r}"
        )

    total = len(values)
    if total < min_draws:
        return None

    positions, counts = _occupancy(_finite_floats(values, what="values"), grid)
    if len(positions) < 2:
        return _unseparated()
    span_steps = round((positions[-1] - positions[0]) / grid)
    if span_steps < 1:
        return _unseparated()

    return _search_split(
        positions,
        counts,
        grid=grid,
        need=mass_min * total,
        trough_steps=trough_steps,
        density_ratio=density_ratio,
        min_cluster_density=min_cluster_density,
    )


def _prefix_tables(counts: Sequence[int]) -> tuple[list[int], list[int], list[int]]:
    """Exact integer prefix sums plus prefix/suffix maxima.

    Integers because the search must make identical decisions regardless of
    accumulation order; prefix tables because the obvious slice-per-pair
    spelling is cubic, which on a fine lattice over a bounded score runs to
    seconds per component.
    """
    prefix = [0] * (len(counts) + 1)
    for idx, count in enumerate(counts):
        prefix[idx + 1] = prefix[idx] + count
    prefix_max, running = [0] * len(counts), 0
    for idx, count in enumerate(counts):
        running = max(running, count)
        prefix_max[idx] = running
    suffix_max = [0] * (len(counts) + 1)
    for idx in range(len(counts) - 1, -1, -1):
        suffix_max[idx] = max(suffix_max[idx + 1], counts[idx])
    return prefix, prefix_max, suffix_max


def _search_split(
    positions: Sequence[float],
    counts: Sequence[int],
    *,
    grid: float,
    need: float,
    trough_steps: int,
    density_ratio: float,
    min_cluster_density: float,
) -> MultimodalVerdict:
    """Sparsest qualifying gap, or an unseparated verdict."""
    total = sum(counts)
    prefix, prefix_max, suffix_max = _prefix_tables(counts)
    best: tuple[int, float, int, int] | None = None
    best_verdict: MultimodalVerdict | None = None

    for i in range(len(positions) - 1):
        lower = prefix[i + 1]
        if lower < need:
            continue
        trough_peak = 0
        for j in range(i + 1, len(positions)):
            if j > i + 1:
                trough_peak = max(trough_peak, counts[j - 1])
            steps = round((positions[j] - positions[i]) / grid) - 1
            if steps < trough_steps:
                continue
            upper = total - prefix[j]
            if upper < need:
                continue
            # Emptiness is evidence only between dense clusters. A scatter of
            # singletons on a too-fine lattice has wide empty runs everywhere.
            cluster_positions = (i + 1) + (len(positions) - j)
            if (lower + upper) / cluster_positions < min_cluster_density:
                continue
            peak = max(prefix_max[i], suffix_max[j])
            if peak < density_ratio * trough_peak:
                continue
            inner = prefix[j] - prefix[i + 1]
            # Deterministic selection: sparsest gap, then sharpest density
            # contrast, then lowest indices. Every key is a function of the
            # values alone, so a reordered input selects the same split.
            key = (inner, -(peak - density_ratio * trough_peak), i, j)
            if best is None or key < best:
                best = key
                best_verdict = MultimodalVerdict(
                    separated=True,
                    lower_mass=lower / total,
                    upper_mass=upper / total,
                    trough_mass=inner / total,
                    gap=(positions[i], positions[j]),
                )

    return best_verdict if best_verdict is not None else _unseparated()


def _unseparated() -> MultimodalVerdict:
    return MultimodalVerdict(
        separated=False, lower_mass=0.0, upper_mass=0.0, trough_mass=0.0, gap=None
    )


def lag_dependence(
    labels: Sequence[object],
    groups: Sequence[int],
) -> float | None:
    """How much more alike draws are within a collection group than across them.

    Returns ``None`` when there are fewer than two groups, or too few pairs to
    compare -- never a misleading zero.

    The statistic is the probability that two draws from the *same* group carry
    the same label, minus the probability for two draws from *different*
    groups. Zero means the grouping carries no information, which is what
    independence looks like; positive means draws collected together agree more
    than draws collected apart.

    This exists because the reported agreement interval assumes independent
    draws, and that is precisely the assumption a serving stack violates --
    batching, cache reuse, and node affinity all couple requests issued
    together. Positive dependence makes every interval narrower than its label,
    in the one direction that matters. Measuring it does not correct the
    interval; it makes the assumption falsifiable instead of merely disclaimed.

    Depends only on the stored labels and group tags, never on arrival order, so
    it replays identically from a persisted draw set.
    """
    if len(labels) != len(groups):
        raise ValueError(
            f"labels and groups must be the same length; got {len(labels)} and {len(groups)}"
        )
    if len(set(groups)) < 2:
        return None

    same_group_pairs = same_group_matches = 0
    diff_group_pairs = diff_group_matches = 0
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            match = labels[i] == labels[j]
            if groups[i] == groups[j]:
                same_group_pairs += 1
                same_group_matches += match
            else:
                diff_group_pairs += 1
                diff_group_matches += match

    if same_group_pairs == 0 or diff_group_pairs == 0:
        return None
    return same_group_matches / same_group_pairs - diff_group_matches / diff_group_pairs


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
    if len(values) == 0:
        raise ValueError("values must be non-empty")
    values = _finite_floats(values, what="values")

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
    "lag_dependence",
    "robust_location",
    "scale_floor",
    "smallest_certifiable_n",
    "smallest_detectable_split_n",
    "snap_to_grid",
    "wilson_interval",
]
