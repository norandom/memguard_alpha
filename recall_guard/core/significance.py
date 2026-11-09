"""Pre-registered significance statistics for the study, without scipy.

The study freezes its tests before it sends a request, so every helper here is
a pure function of its inputs and, where it resamples, of an explicit seed.
Resampling uses ``numpy.random.default_rng(seed)``, so the same seed gives the
same interval on every platform numpy supports.

- :func:`holm` adjusts a family of raw p-values with the Holm step-down method
  (Req 8.8).
- :func:`exact_binomial_greater` is the exact one-sided binomial tail, for a
  count of successes against a chance rate.
- :func:`block_bootstrap` gives a percentile interval from a non-circular
  moving-block bootstrap over ordered units (decision months), keeping every
  row of a unit together, so the identifying/anonymised and variant pairing
  within a month survives resampling (Bertrand, Duflo and Mullainathan 2004).
- :func:`did_gap` is the difference-in-differences accuracy gap whose change
  per variant Req 8.3 tests.
- :func:`random_answerer_null` is the Monte Carlo test of a model's decisions
  against a random answerer over the same decision space (Req 8.7).
- :func:`paired_loss_difference` tests the paired difference in mean Brier
  score or log loss against no adjustment (Req 8.4), and the negative-control
  change against no adjustment (Req 7.2), with a moving-block interval.
- :func:`half_ensemble_variance` compares the run-to-run variance of two
  disjoint half-ensembles with that of two single draws on shared items
  (Req 8.5).
- :func:`simulated_power` is the power of a one-sided t test for a
  pre-registered effect, by simulation and without a t table (Req 8.9).

Intervals use the percentile convention of :mod:`recall_guard.core.bootstrap`:
two-sided, with ``confidence`` the coverage, bounds at the ``(1 - confidence) / 2``
and ``1 - (1 - confidence) / 2`` percentiles under numpy's linear rule.

A statistic may return a float, a numpy scalar or a 0-d real array; the last is
coerced with ``float()``. Any other value that is not a finite real is a
dropped resample in the bootstrap and an error in the random-answerer null.
Every :class:`TestResult` built on resampling records ``dropped_resamples`` in
its parameters, so a conditional interval shows in reports, not only in logs.
"""

from __future__ import annotations

import logging
import math
import numbers
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

logger = logging.getLogger(__name__)

#: Method label of :func:`block_bootstrap` intervals.
BLOCK_BOOTSTRAP_METHOD = "moving-block percentile bootstrap"

#: Method label of the item-level intervals of :func:`half_ensemble_variance`.
ITEM_BOOTSTRAP_METHOD = "item-level percentile bootstrap"


def _is_int(value: object) -> bool:
    # bool is an int subclass; True as a count or seed is a caller bug.
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _is_finite_real(value: object) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(value)


def _finite_float(value: object) -> float | None:
    """Return a statistic value as a float, or None when it is not a finite real.

    A 0-d integer or floating array, for example ``np.asarray(np.mean(x))``,
    is coerced with ``float()`` first. Boolean, complex, string and object 0-d
    arrays are not coerced, so they read as not real, as ``bool`` does in
    :func:`_is_finite_real`.
    """
    if isinstance(value, np.ndarray) and value.ndim == 0 and value.dtype.kind in "iuf":
        value = float(value)
    if not _is_finite_real(value):
        return None
    return float(value)


def _real_array(caller: str, name: str, values: object) -> np.ndarray:
    """Return ``values`` as a float array, refusing non-real or non-finite input."""
    array = np.asarray(values)
    if array.dtype != np.bool_ and (
        not np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.complexfloating)
    ):
        raise ValueError(f"{caller}: '{name}' must be boolean or real, got dtype {array.dtype}.")
    array = array.astype(float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{caller}: '{name}' values must be finite.")
    return array


def _check_confidence(name: str, confidence: object) -> float:
    if not _is_finite_real(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError(f"{name}: 'confidence' must be in (0, 1), got {confidence!r}.")
    return float(confidence)


def _percentile_bounds(values: np.ndarray, confidence: float) -> tuple[float, float]:
    alpha = 1.0 - confidence
    lo = float(np.percentile(values, (alpha / 2.0) * 100.0))
    hi = float(np.percentile(values, (1.0 - alpha / 2.0) * 100.0))
    return lo, hi


@dataclass(frozen=True)
class Interval:
    """A two-sided interval and how it was made.

    Attributes:
        lo: Lower bound.
        hi: Upper bound, at least ``lo``.
        confidence: Two-sided coverage in (0, 1), for example 0.95.
        method: How the bounds were computed, for example
            ``"moving-block percentile bootstrap"``.
    """

    lo: float
    hi: float
    confidence: float
    method: str

    def __post_init__(self) -> None:
        if not (_is_finite_real(self.lo) and _is_finite_real(self.hi)):
            raise ValueError(f"Interval: bounds must be finite reals, got ({self.lo!r}, {self.hi!r}).")
        if self.lo > self.hi:
            raise ValueError(f"Interval: 'lo' {self.lo!r} exceeds 'hi' {self.hi!r}.")
        _check_confidence("Interval", self.confidence)
        if not isinstance(self.method, str) or not self.method:
            raise ValueError("Interval: 'method' must be a non-empty string.")


@dataclass(frozen=True)
class TestResult:
    """One statistical test, with everything needed to report and re-run it.

    Attributes:
        statistic: The observed test statistic.
        interval: Its interval, or None when the test has none.
        p_value: The raw p-value before any Holm adjustment, or None.
        method: What the test is and what the interval means.
        parameters: Sample size, seed, resample counts, block length and the
            like. Stored as a read-only copy.
    """

    __test__ = False  # not a pytest test class

    statistic: float
    interval: Interval | None
    p_value: float | None
    method: str
    parameters: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.p_value is not None and (not _is_finite_real(self.p_value) or not 0.0 <= self.p_value <= 1.0):
            raise ValueError(f"TestResult: 'p_value' must be in [0, 1] or None, got {self.p_value!r}.")
        if not isinstance(self.method, str) or not self.method:
            raise ValueError("TestResult: 'method' must be a non-empty string.")
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))


def holm(p_values: Mapping[str, float]) -> dict[str, float]:
    """Adjust a family of raw p-values with the Holm step-down method.

    The raw values are sorted ascending, ties in a stable order by key. The
    i-th smallest (1-based) of m is multiplied by ``m - i + 1``, a running
    maximum keeps the adjusted values in the raw order, and each is capped at 1.

    Args:
        p_values: Raw p-value per test name, each a finite real in [0, 1].

    Returns:
        The adjusted p-value per test name, with the same keys in the same
        order. An empty family gives an empty dict.

    Raises:
        ValueError: if any p-value is not a real number in [0, 1].
    """
    for key, value in p_values.items():
        if not _is_finite_real(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"holm: p-value for {key!r} must be a real number in [0, 1], got {value!r}.")
    m = len(p_values)
    ordered = sorted(p_values, key=lambda key: (float(p_values[key]), key))
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, key in enumerate(ordered, start=1):
        running = max(running, min(1.0, (m - rank + 1) * float(p_values[key])))
        adjusted[key] = running
    return {key: adjusted[key] for key in p_values}


def exact_binomial_greater(k: int, n: int, p: float = 0.5) -> float:
    """Return P(X >= k) for X ~ Binomial(n, p), summed exactly.

    Each term ``C(n, j) p^j (1 - p)^(n - j)`` for ``j = k..n`` is formed in log
    space with :func:`math.lgamma` and summed with :func:`math.fsum` relative to
    the largest term, so large n neither overflows nor loses the small terms.
    A tail below the smallest double underflows to 0.0.

    Args:
        k: Observed number of successes, an integer with ``0 <= k <= n``.
        n: Number of trials, a non-negative integer.
        p: Success probability under the null, in (0, 1). Default 0.5.

    Returns:
        The one-sided (greater) p-value in [0, 1]; exactly 1.0 when ``k == 0``.

    Raises:
        ValueError: if ``k`` or ``n`` is not an integer, ``k`` is outside
            ``0..n``, or ``p`` is not in (0, 1).
    """
    if not (_is_int(k) and _is_int(n)):
        raise ValueError(f"exact_binomial_greater: 'k' and 'n' must be integers, got {k!r} and {n!r}.")
    if not 0 <= k <= n:
        raise ValueError(f"exact_binomial_greater: need 0 <= k <= n, got k={k}, n={n}.")
    if not _is_finite_real(p) or not 0.0 < p < 1.0:
        raise ValueError(f"exact_binomial_greater: 'p' must be in (0, 1), got {p!r}.")
    if k == 0:
        return 1.0
    log_p = math.log(p)
    log_q = math.log1p(-p)
    log_n_factorial = math.lgamma(n + 1)
    terms = [
        log_n_factorial - math.lgamma(j + 1) - math.lgamma(n - j + 1) + j * log_p + (n - j) * log_q
        for j in range(k, n + 1)
    ]
    peak = max(terms)
    tail = math.exp(peak) * math.fsum(math.exp(term - peak) for term in terms)
    return min(1.0, tail)


def _moving_block_statistics(
    caller: str,
    stat: Callable[[np.ndarray], object],
    blocks: np.ndarray,
    *,
    block_length: int,
    n_resamples: int,
    seed: int,
) -> tuple[np.ndarray, int]:
    """Validate, resample and return the kept statistics and the dropped count.

    This is the resampling behind :func:`block_bootstrap` and the tests built
    on it; see :func:`block_bootstrap` for the scheme. With ``block_length``
    1 the starts are ``n`` uniform draws from ``0..n - 1``, which is the plain
    item-level bootstrap. Error and warning messages start with ``caller``.

    Returns:
        The kept resample statistics in draw order, as a float array of length
        ``n_resamples - dropped``, and the number of dropped resamples.

    Raises:
        ValueError: on an empty or zero-dimensional ``blocks``, a
            ``block_length`` outside ``1..n``, ``n_resamples < 1``, a non-integer
            seed, or when every resample is dropped.
    """
    rows = np.asarray(blocks)
    if rows.ndim == 0 or rows.shape[0] == 0:
        raise ValueError(f"{caller}: 'blocks' must have at least one row along axis 0.")
    n = rows.shape[0]
    if not _is_int(block_length) or not 1 <= block_length <= n:
        raise ValueError(f"{caller}: 'block_length' must be an integer in 1..{n}, got {block_length!r}.")
    if not _is_int(n_resamples) or n_resamples < 1:
        raise ValueError(f"{caller}: 'n_resamples' must be an integer >= 1, got {n_resamples!r}.")
    if not _is_int(seed):
        raise ValueError(f"{caller}: 'seed' must be an integer, got {seed!r}.")

    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(n / block_length)
    offsets = np.arange(block_length)
    kept: list[float] = []
    dropped = 0
    for _ in range(n_resamples):
        starts = rng.integers(0, n - block_length + 1, size=n_blocks)
        index = (starts[:, None] + offsets).ravel()[:n]
        try:
            value = _finite_float(stat(rows[index]))
        except ValueError:
            value = None
        if value is None:
            dropped += 1
            continue
        kept.append(value)

    if dropped:
        logger.warning(
            "%s: dropped %d / %d resamples where the statistic raised or returned a non-finite value.",
            caller,
            dropped,
            n_resamples,
        )
    if not kept:
        raise ValueError(f"{caller}: every resample was dropped; the statistic is undefined on these blocks.")
    return np.asarray(kept, dtype=float), dropped


def block_bootstrap(
    stat: Callable[[np.ndarray], float],
    blocks: np.ndarray,
    *,
    block_length: int,
    n_resamples: int,
    seed: int,
    confidence: float = 0.95,
) -> Interval:
    """Percentile interval for ``stat`` from a non-circular moving-block bootstrap.

    The first axis of ``blocks`` is the ordered unit (decision months, n rows);
    each row holds everything paired within that month, so resampling whole
    rows keeps the pairing. Each resample draws ``ceil(n / block_length)`` start
    indices uniformly from ``0..n - block_length``, concatenates the
    ``block_length`` consecutive rows from each start, truncates to n rows and
    applies ``stat``.

    Resamples on which ``stat`` raises ``ValueError`` (for example an empty
    cell) or returns a value that is not a finite real are dropped, with one
    ``logging.WARNING`` per call giving the count, as in
    :func:`recall_guard.core.bootstrap.bootstrap_ci`. A 0-d real array is
    coerced with ``float()`` and kept.

    Args:
        stat: Statistic of an array shaped like ``blocks``; returns a float
            (or a numpy scalar or 0-d real array).
        blocks: Array with at least one axis; axis 0 is the ordered unit.
        block_length: Rows per block, an integer in ``1..n``. With n the whole
            series is one block and the interval collapses to ``stat(blocks)``.
        n_resamples: Number of resamples, at least 1.
        seed: Seed for ``numpy.random.default_rng``; same seed, same interval.
        confidence: Two-sided coverage in (0, 1). Default 0.95.

    Returns:
        An :class:`Interval` with ``method`` ``"moving-block percentile bootstrap"``.

    Raises:
        ValueError: on an empty or zero-dimensional ``blocks``, a
            ``block_length`` outside ``1..n``, ``n_resamples < 1``, a non-integer
            seed, ``confidence`` outside (0, 1), or when every resample is
            dropped.
    """
    confidence = _check_confidence("block_bootstrap", confidence)
    kept, _ = _moving_block_statistics(
        "block_bootstrap", stat, blocks, block_length=block_length, n_resamples=n_resamples, seed=seed
    )
    lo, hi = _percentile_bounds(kept, confidence)
    return Interval(lo=lo, hi=hi, confidence=confidence, method=BLOCK_BOOTSTRAP_METHOD)


def did_gap(correct: np.ndarray, identifying: np.ndarray, pre_cutoff: np.ndarray) -> float:
    """Difference-in-differences accuracy gap over aligned per-item arrays.

    The gap is (accuracy of identifying minus anonymised prompts before the
    cutoff) minus (the same after the cutoff). Memorization shows as a positive
    gap: identifying prompts help only where the model may have seen the answer.

    Args:
        correct: Per-item correctness, boolean or real in [0, 1], shape (n,).
        identifying: Boolean, shape (n,): True for the identifying prompt,
            False for the anonymised one.
        pre_cutoff: Boolean, shape (n,): True when the decision date is before
            the model's training cutoff.

    Returns:
        The gap as a float in [-2, 2].

    Raises:
        ValueError: if the arrays are not aligned one-dimensional arrays of the
            stated types, or any of the four cells is empty.
    """
    accuracy = np.asarray(correct)
    is_identifying = np.asarray(identifying)
    is_pre = np.asarray(pre_cutoff)
    if accuracy.ndim != 1 or is_identifying.shape != accuracy.shape or is_pre.shape != accuracy.shape:
        raise ValueError(
            "did_gap: 'correct', 'identifying' and 'pre_cutoff' must be aligned one-dimensional arrays, got shapes "
            f"{accuracy.shape}, {is_identifying.shape} and {is_pre.shape}."
        )
    if is_identifying.dtype != np.bool_ or is_pre.dtype != np.bool_:
        raise ValueError("did_gap: 'identifying' and 'pre_cutoff' must be boolean arrays.")
    if accuracy.dtype != np.bool_:
        if not np.issubdtype(accuracy.dtype, np.number) or np.issubdtype(accuracy.dtype, np.complexfloating):
            raise ValueError(f"did_gap: 'correct' must be boolean or real, got dtype {accuracy.dtype}.")
        if not np.all(np.isfinite(accuracy)) or np.any(accuracy < 0) or np.any(accuracy > 1):
            raise ValueError("did_gap: 'correct' values must be finite and in [0, 1].")
    accuracy = accuracy.astype(float)

    def cell(identifying_value: bool, pre_value: bool) -> float:
        mask = (is_identifying == identifying_value) & (is_pre == pre_value)
        if not mask.any():
            prompt = "identifying" if identifying_value else "anonymised"
            period = "before" if pre_value else "after"
            raise ValueError(f"did_gap: the {prompt} cell {period} the cutoff is empty.")
        return float(accuracy[mask].mean())

    before = cell(True, True) - cell(False, True)
    after = cell(True, False) - cell(False, False)
    return before - after


def random_answerer_null(
    observed: np.ndarray,
    sampler: Callable[[np.random.Generator, int], np.ndarray],
    *,
    statistic: Callable[[np.ndarray], float],
    n_null: int,
    seed: int,
    confidence: float = 0.95,
) -> TestResult:
    """Monte Carlo test of observed decisions against a random answerer.

    ``statistic`` is evaluated on ``observed`` and on ``n_null`` samples of size
    ``len(observed)``, each drawn by ``sampler`` from the supplied decision
    space with one generator seeded once from ``seed``. The one-sided (greater)
    p-value is ``(1 + #{null >= observed}) / (1 + n_null)``, so it is never 0.

    Args:
        observed: The model's decisions; axis 0 is the item, length at least 1.
        sampler: ``sampler(rng, size)`` returns ``size`` random answers from the
            decision space along axis 0.
        statistic: Maps answers to a float, larger meaning further from chance.
        n_null: Number of null samples, at least 1.
        seed: Seed for ``numpy.random.default_rng``.
        confidence: Coverage of the reported central range of the null
            distribution, in (0, 1). Default 0.95.

    Returns:
        A :class:`TestResult` whose interval is the central ``confidence`` range
        of the null distribution, not an interval for the observed statistic.
        Its ``dropped_resamples`` parameter is always 0: a null sample whose
        statistic is not a finite real is an error here, never dropped.

    Raises:
        ValueError: on an empty or zero-dimensional ``observed``, ``n_null < 1``,
            a non-integer seed, ``confidence`` outside (0, 1), a sampler that
            returns the wrong number of answers, or a statistic that is not a
            finite real (a 0-d real array is coerced with ``float()`` first).
    """
    answers = np.asarray(observed)
    if answers.ndim == 0 or answers.shape[0] == 0:
        raise ValueError("random_answerer_null: 'observed' must have at least one item along axis 0.")
    if not _is_int(n_null) or n_null < 1:
        raise ValueError(f"random_answerer_null: 'n_null' must be an integer >= 1, got {n_null!r}.")
    if not _is_int(seed):
        raise ValueError(f"random_answerer_null: 'seed' must be an integer, got {seed!r}.")
    confidence = _check_confidence("random_answerer_null", confidence)
    n = answers.shape[0]

    def evaluate(sample: np.ndarray, what: str) -> float:
        raw = statistic(sample)
        value = _finite_float(raw)
        if value is None:
            raise ValueError(f"random_answerer_null: the statistic of the {what} is not a finite real: {raw!r}.")
        return value

    observed_stat = evaluate(answers, "observed decisions")
    rng = np.random.default_rng(seed)
    null = np.empty(n_null, dtype=float)
    for i in range(n_null):
        sample = np.asarray(sampler(rng, n))
        if sample.ndim == 0 or sample.shape[0] != n:
            raise ValueError(
                f"random_answerer_null: the sampler must return {n} answers along axis 0, got size "
                f"{sample.shape[0] if sample.ndim else 0}."
            )
        null[i] = evaluate(sample, "null sample")

    p_value = (1 + int(np.count_nonzero(null >= observed_stat))) / (1 + n_null)
    lo, hi = _percentile_bounds(null, confidence)
    percent = f"{confidence * 100:g} percent"
    interval = Interval(
        lo=lo,
        hi=hi,
        confidence=confidence,
        method=f"central {percent} range of the Monte Carlo random-answerer null distribution",
    )
    return TestResult(
        statistic=observed_stat,
        interval=interval,
        p_value=p_value,
        method=(
            "Monte Carlo random-answerer null, one-sided (greater), p = (1 + #null >= observed) / (1 + n_null); "
            f"the interval is the central {percent} range of the null distribution"
        ),
        parameters={
            "n": n,
            "n_null": n_null,
            "seed": seed,
            "confidence": confidence,
            "alternative": "greater",
            "dropped_resamples": 0,
        },
    )


def paired_loss_difference(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    *,
    block_length: int,
    n_resamples: int,
    seed: int,
    confidence: float = 0.95,
) -> TestResult:
    """Paired difference in mean loss, with a moving-block bootstrap interval.

    ``loss_a`` and ``loss_b`` are aligned: axis 0 is the ordered month unit,
    and a 2-D array holds one row per month (for example the per-item Brier
    scores or log losses of that month's decisions). The statistic is
    ``mean(loss_a) - mean(loss_b)`` over all elements, so a negative value
    means ``loss_a`` is lower. The interval is the moving-block bootstrap of
    that difference (see :func:`block_bootstrap`), resampling the rows of both
    arrays together so the pairing within a month survives. The two-sided
    p-value is ``min(1, 2 min(1 + #{boot <= 0}, 1 + #{boot >= 0}) / (1 + n_kept))``.

    The same test serves the negative-control change against no adjustment
    (Req 7.2), with the outlook values in place of losses.

    Args:
        loss_a: Losses (or outlook values) of the variant, shape (n,) or (n, m).
        loss_b: The paired values of the reference, the same shape.
        block_length: Months per block, an integer in ``1..n``.
        n_resamples: Number of resamples, at least 1.
        seed: Seed for ``numpy.random.default_rng``; same seed, same result.
        confidence: Two-sided coverage in (0, 1). Default 0.95.

    Returns:
        A :class:`TestResult` whose parameters record ``n`` (months),
        ``n_values`` (elements per array), ``block_length``, ``n_resamples``,
        ``seed``, ``confidence``, ``alternative`` and ``dropped_resamples``.

    Raises:
        ValueError: if the arrays are not aligned 1-D or 2-D arrays of finite
            booleans or reals with at least one element, on an invalid
            ``block_length``, ``n_resamples``, seed or ``confidence``, or when
            every resample is dropped.
    """
    caller = "paired_loss_difference"
    a = _real_array(caller, "loss_a", loss_a)
    b = _real_array(caller, "loss_b", loss_b)
    if a.ndim not in (1, 2) or a.shape != b.shape or a.size == 0:
        raise ValueError(
            f"{caller}: 'loss_a' and 'loss_b' must be aligned non-empty 1-D or 2-D arrays, got shapes "
            f"{a.shape} and {b.shape}."
        )
    confidence = _check_confidence(caller, confidence)
    # Axis 1 of rows is (a, b), so a resampled month keeps both of its arrays.
    rows = np.stack([a, b], axis=1)

    def difference(sample: np.ndarray) -> float:
        return float(sample[:, 0].mean() - sample[:, 1].mean())

    observed = difference(rows)
    kept, dropped = _moving_block_statistics(
        caller, difference, rows, block_length=block_length, n_resamples=n_resamples, seed=seed
    )
    n_kept = kept.size
    at_or_below = 1 + int(np.count_nonzero(kept <= 0.0))
    at_or_above = 1 + int(np.count_nonzero(kept >= 0.0))
    p_value = min(1.0, 2.0 * min(at_or_below, at_or_above) / (1 + n_kept))
    lo, hi = _percentile_bounds(kept, confidence)
    return TestResult(
        statistic=observed,
        interval=Interval(lo=lo, hi=hi, confidence=confidence, method=BLOCK_BOOTSTRAP_METHOD),
        p_value=p_value,
        method=(
            "paired difference in mean loss (a minus b) with a moving-block percentile bootstrap interval; "
            "two-sided p = min(1, 2 min(1 + #boot <= 0, 1 + #boot >= 0) / (1 + n_kept))"
        ),
        parameters={
            "n": a.shape[0],
            "n_values": a.size,
            "block_length": block_length,
            "n_resamples": n_resamples,
            "seed": seed,
            "confidence": confidence,
            "alternative": "two-sided",
            "dropped_resamples": dropped,
        },
    )


def half_ensemble_variance(
    halves_a: np.ndarray,
    halves_b: np.ndarray,
    singles: np.ndarray,
    *,
    n_resamples: int,
    seed: int,
    confidence: float = 0.95,
) -> TestResult:
    """Ratio of half-ensemble to single-draw run-to-run variance on shared items.

    Per item, ``halves_a`` and ``halves_b`` are the results (decision or
    memorization estimate) of two disjoint half-ensembles and ``singles`` holds
    two single draws, all from the same recorded ensemble. The run-to-run
    variance of a pair is half the mean squared difference over items. The
    statistic is the half-ensemble variance divided by the single-draw
    variance: below 1 means the ensemble reduces variance, and with
    independent draws it is about ``2 / K`` for halves of ``K / 2`` draws.

    The interval is an item-level percentile bootstrap of the ratio (the
    moving-block scheme with blocks of one item). A resample whose two single
    draws agree on every drawn item has no ratio and is dropped. The one-sided
    p-value for a ratio below 1 is ``(1 + #{boot >= 1}) / (1 + n_kept)``.

    Args:
        halves_a: Result of the first half-ensemble per item, shape (n_items,).
        halves_b: Result of the second half-ensemble per item, shape (n_items,).
        singles: Two single-draw results per item, shape (n_items, 2).
        n_resamples: Number of resamples, at least 1.
        seed: Seed for ``numpy.random.default_rng``; same seed, same result.
        confidence: Two-sided coverage in (0, 1). Default 0.95.

    Returns:
        A :class:`TestResult` whose parameters record ``n_items``,
        ``half_ensemble_variance``, ``single_draw_variance``, ``n_resamples``,
        ``seed``, ``confidence``, ``alternative`` and ``dropped_resamples``.

    Raises:
        ValueError: if the arrays are not finite booleans or reals of the stated
            aligned shapes with at least one item, if the single draws agree on
            every item, on an invalid ``n_resamples``, seed or ``confidence``,
            or when every resample is dropped.
    """
    caller = "half_ensemble_variance"
    a = _real_array(caller, "halves_a", halves_a)
    b = _real_array(caller, "halves_b", halves_b)
    pairs = _real_array(caller, "singles", singles)
    if a.ndim != 1 or a.shape[0] == 0 or b.shape != a.shape or pairs.shape != (a.shape[0], 2):
        raise ValueError(
            f"{caller}: need 'halves_a' and 'halves_b' of shape (n_items,) and 'singles' of shape (n_items, 2) "
            f"with n_items >= 1, got {a.shape}, {b.shape} and {pairs.shape}."
        )
    confidence = _check_confidence(caller, confidence)
    # Columns: half a, half b, first single draw, second single draw.
    rows = np.column_stack([a, b, pairs])

    def variances(sample: np.ndarray) -> tuple[float, float]:
        half = 0.5 * float(np.mean((sample[:, 0] - sample[:, 1]) ** 2))
        single = 0.5 * float(np.mean((sample[:, 2] - sample[:, 3]) ** 2))
        return half, single

    def ratio(sample: np.ndarray) -> float:
        half, single = variances(sample)
        if single == 0.0:
            raise ValueError("the single draws agree on every item")
        return half / single

    half_variance, single_variance = variances(rows)
    if single_variance == 0.0:
        raise ValueError(
            f"{caller}: the single draws agree on every item, so the single-draw variance is zero and the "
            "ratio is undefined."
        )
    observed = _finite_float(half_variance / single_variance)
    if observed is None:
        raise ValueError(f"{caller}: the variance ratio is not finite.")
    kept, dropped = _moving_block_statistics(caller, ratio, rows, block_length=1, n_resamples=n_resamples, seed=seed)
    p_value = (1 + int(np.count_nonzero(kept >= 1.0))) / (1 + kept.size)
    lo, hi = _percentile_bounds(kept, confidence)
    return TestResult(
        statistic=observed,
        interval=Interval(lo=lo, hi=hi, confidence=confidence, method=ITEM_BOOTSTRAP_METHOD),
        p_value=p_value,
        method=(
            "ratio of half-ensemble to single-draw run-to-run variance (half the mean squared pair difference) "
            "with an item-level percentile bootstrap interval; one-sided (ratio below 1) "
            "p = (1 + #boot >= 1) / (1 + n_kept)"
        ),
        parameters={
            "n_items": a.shape[0],
            "half_ensemble_variance": half_variance,
            "single_draw_variance": single_variance,
            "n_resamples": n_resamples,
            "seed": seed,
            "confidence": confidence,
            "alternative": "less",
            "dropped_resamples": dropped,
        },
    )


def _t_statistics(values: np.ndarray) -> np.ndarray:
    """One-sample t statistic of each row of ``values`` (simulations x items)."""
    n_items = values.shape[1]
    return values.mean(axis=1) / (values.std(axis=1, ddof=1) / math.sqrt(n_items))


def simulated_power(effect: float, noise_sd: float, n_items: int, *, alpha: float, n_sim: int, seed: int) -> float:
    """Power of a one-sided t test of a positive mean, by simulation.

    Each simulation draws ``n_items`` independent normal(``effect``,
    ``noise_sd``) values and forms the t statistic ``mean / (sd / sqrt(n))``
    with the sample standard deviation. The critical value is the
    ``1 - alpha`` quantile (numpy's linear rule) of ``n_sim`` such statistics
    simulated under effect 0, so no t table is needed. Power is the share of
    ``n_sim`` simulations under ``effect`` whose statistic exceeds it.

    One generator seeded once from ``seed`` draws the null set first and then
    the standard normal noise of the alternative set, which is scaled and
    shifted by ``effect``. The noise therefore does not depend on ``effect``,
    so for one seed the power never decreases as the effect grows.

    Args:
        effect: Pre-registered mean effect, a finite real (0 gives about alpha).
        noise_sd: Standard deviation of each value, a finite real above 0.
        n_items: Values per simulated study, an integer of at least 2.
        alpha: One-sided significance level, in (0, 1).
        n_sim: Simulations under each of the null and the effect, at least 1.
        seed: Seed for ``numpy.random.default_rng``; same seed, same power.

    Returns:
        The simulated power in [0, 1].

    Raises:
        ValueError: on a non-finite ``effect``, a ``noise_sd`` that is not a
            positive finite real, ``n_items < 2``, ``alpha`` outside (0, 1),
            ``n_sim < 1`` or a non-integer seed or count.
    """
    caller = "simulated_power"
    if not _is_finite_real(effect):
        raise ValueError(f"{caller}: 'effect' must be a finite real, got {effect!r}.")
    if not _is_finite_real(noise_sd) or noise_sd <= 0.0:
        raise ValueError(f"{caller}: 'noise_sd' must be a finite real above 0, got {noise_sd!r}.")
    if not _is_int(n_items) or n_items < 2:
        raise ValueError(f"{caller}: 'n_items' must be an integer >= 2, got {n_items!r}.")
    if not _is_finite_real(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError(f"{caller}: 'alpha' must be in (0, 1), got {alpha!r}.")
    if not _is_int(n_sim) or n_sim < 1:
        raise ValueError(f"{caller}: 'n_sim' must be an integer >= 1, got {n_sim!r}.")
    if not _is_int(seed):
        raise ValueError(f"{caller}: 'seed' must be an integer, got {seed!r}.")

    rng = np.random.default_rng(seed)
    null = _t_statistics(float(noise_sd) * rng.standard_normal((n_sim, n_items)))
    critical = float(np.quantile(null, 1.0 - float(alpha)))
    alternative = _t_statistics(float(effect) + float(noise_sd) * rng.standard_normal((n_sim, n_items)))
    return float(np.count_nonzero(alternative > critical) / n_sim)
