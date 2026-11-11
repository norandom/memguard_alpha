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
- :func:`simulated_power_two_sample` is the same for the difference of two
  independent sample means with the pooled standard error, the power of the
  pre-cutoff minus post-cutoff gap the pilot gates on (Req 8.9).
- :class:`Statistic` is one reported statistic with its calculation exposed,
  and :func:`clustered_auc_interval`, :func:`permutation_auc_test`,
  :func:`paired_auc_difference`, :func:`agreement`,
  :func:`permutation_mean_difference` and :func:`detectable_effect` are the
  statistics of the qualified-model catalogue (its requirements 4.3, 5.3 and 6).

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
from statistics import NormalDist
from types import MappingProxyType

import numpy as np

logger = logging.getLogger(__name__)

#: Method label of :func:`block_bootstrap` intervals.
BLOCK_BOOTSTRAP_METHOD = "moving-block percentile bootstrap"

#: Method label of the item-level intervals of :func:`half_ensemble_variance`.
ITEM_BOOTSTRAP_METHOD = "item-level percentile bootstrap"

#: ``interval_note`` of :func:`half_ensemble_variance` when the upper
#: percentile falls on unbounded resamples and no interval is reported.
UNBOUNDED_UPPER_NOTE = "upper bound unbounded"


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


def _percentile_bounds_ranking_unbounded_highest(values: np.ndarray, confidence: float) -> tuple[float, float] | None:
    """Percentile bounds with ``+inf`` values ranked highest, or None when the upper one is not finite.

    The bounds follow :func:`_percentile_bounds` over all of ``values``. The
    upper bound is not finite when the unbounded share exceeds
    ``(1 - confidence) / 2``, or when numpy's linear rule would interpolate
    towards an unbounded rank at that percentile. Otherwise only finite ranks
    are read, so replacing each ``+inf`` by the largest finite value gives the
    same bounds as ranking it highest.
    """
    unbounded = np.isposinf(values)
    n_unbounded = int(np.count_nonzero(unbounded))
    if n_unbounded == 0:
        return _percentile_bounds(values, confidence)
    n = values.size
    alpha = 1.0 - confidence
    if n_unbounded / n > alpha / 2.0:
        return None
    # numpy's linear rule reads ranks floor(h) and ceil(h), h = (n - 1) * q.
    upper_index = (n - 1) * (((1.0 - alpha / 2.0) * 100.0) / 100.0)
    if math.ceil(upper_index) > n - n_unbounded - 1:
        return None
    finite_max = float(values[~unbounded].max())
    return _percentile_bounds(np.where(unbounded, finite_max, values), confidence)


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
    keep_positive_infinity: bool = False,
) -> tuple[np.ndarray, int]:
    """Validate, resample and return the kept statistics and the dropped count.

    This is the resampling behind :func:`block_bootstrap` and the tests built
    on it; see :func:`block_bootstrap` for the scheme. With ``block_length``
    1 the starts are ``n`` uniform draws from ``0..n - 1``, which is the plain
    item-level bootstrap. Error and warning messages start with ``caller``.

    With ``keep_positive_infinity``, a statistic that returns the float
    ``math.inf`` (an unbounded value, such as a ratio over a zero denominator)
    is kept as ``inf`` instead of dropped. Every other value that is not a
    finite real is still dropped.

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
            raw = stat(rows[index])
        except ValueError:
            raw = None
        value = _finite_float(raw)
        if value is None and keep_positive_infinity and isinstance(raw, float) and raw == math.inf:
            value = math.inf
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
    moving-block scheme with blocks of one item). The one-sided p-value for a
    ratio below 1 is ``(1 + #{boot >= 1}) / (1 + n_kept)``.

    A resample whose two single draws agree on every drawn item has a zero
    single-draw variance. When its half-ensemble variance is positive, the
    ratio is unbounded: the resample is kept, counts as ``>= 1`` in the
    p-value (the conservative direction) and is never dropped. Only a 0/0
    resample is dropped. The interval ranks unbounded resamples highest; when
    the upper percentile falls on them (their share of the kept resamples
    exceeds ``(1 - confidence) / 2``, or the linear rule would interpolate
    towards one) the interval is None and ``interval_note`` says why.

    On the original data a zero single-draw variance raises ``ValueError``,
    whatever the half-ensemble variance, so ``statistic`` is always finite.

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
        ``seed``, ``confidence``, ``alternative``, ``dropped_resamples`` (0/0
        resamples), ``unbounded_resamples`` (kept resamples with an unbounded
        ratio) and ``interval_note`` (None, or ``"upper bound unbounded"``
        when ``interval`` is None).

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
            if half > 0.0:
                return math.inf
            raise ValueError("0/0: the single draws and the half-ensembles agree on every item")
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
    kept, dropped = _moving_block_statistics(
        caller,
        ratio,
        rows,
        block_length=1,
        n_resamples=n_resamples,
        seed=seed,
        keep_positive_infinity=True,
    )
    # Unbounded (inf) resamples compare >= 1, so they count against the ensemble.
    p_value = (1 + int(np.count_nonzero(kept >= 1.0))) / (1 + kept.size)
    bounds = _percentile_bounds_ranking_unbounded_highest(kept, confidence)
    interval = (
        None
        if bounds is None
        else Interval(lo=bounds[0], hi=bounds[1], confidence=confidence, method=ITEM_BOOTSTRAP_METHOD)
    )
    return TestResult(
        statistic=observed,
        interval=interval,
        p_value=p_value,
        method=(
            "ratio of half-ensemble to single-draw run-to-run variance (half the mean squared pair difference) "
            "with an item-level percentile bootstrap interval; one-sided (ratio below 1) "
            "p = (1 + #boot >= 1) / (1 + n_kept); a resample with zero single-draw and positive half-ensemble "
            "variance is unbounded, kept, counted >= 1 and ranked highest, and only 0/0 resamples are dropped"
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
            "unbounded_resamples": int(np.count_nonzero(np.isposinf(kept))),
            "interval_note": UNBOUNDED_UPPER_NOTE if interval is None else None,
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


def _pooled_t_statistics(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pooled two-sample t statistic of each row pair of ``a`` and ``b`` (simulations x items)."""
    n_a, n_b = a.shape[1], b.shape[1]
    pooled = ((n_a - 1) * a.var(axis=1, ddof=1) + (n_b - 1) * b.var(axis=1, ddof=1)) / (n_a + n_b - 2)
    return (a.mean(axis=1) - b.mean(axis=1)) / np.sqrt(pooled * (1.0 / n_a + 1.0 / n_b))


def simulated_power_two_sample(effect: float, noise_sd: float, n_a: int, n_b: int, *, alpha: float, n_sim: int,
                               seed: int) -> float:
    """Power of a one-sided pooled two-sample t test that the first mean exceeds the second, by simulation.

    The conventions of :func:`simulated_power`, for two independent samples.
    Each simulation draws ``n_a`` values from normal(``effect``, ``noise_sd``)
    and ``n_b`` values from normal(0, ``noise_sd``) and forms the statistic
    ``(mean_a - mean_b) / (s_p * sqrt(1 / n_a + 1 / n_b))``, where ``s_p`` is
    the pooled sample standard deviation (each sample with ddof 1). The
    critical value is the ``1 - alpha`` quantile (numpy's linear rule) of
    ``n_sim`` such statistics simulated under effect 0, so no t table is
    needed. Power is the share of ``n_sim`` simulations under ``effect`` whose
    statistic exceeds it.

    One generator seeded once from ``seed`` draws, in this order, the null
    samples ``a`` and ``b`` and then the standard normal noise of the
    alternative samples ``a`` and ``b``; only the alternative ``a`` is shifted
    by ``effect``. The noise therefore does not depend on ``effect``, so for one
    seed the power never decreases as the effect grows. The statistic is
    invariant to the scale of the noise, so the power depends on
    ``effect / noise_sd`` only.

    Args:
        effect: Pre-registered difference of the means, a finite real (0 gives about alpha).
        noise_sd: Standard deviation of each value in both samples, a finite real above 0.
        n_a: Values in the first sample, an integer of at least 2.
        n_b: Values in the second sample, an integer of at least 2.
        alpha: One-sided significance level, in (0, 1).
        n_sim: Simulations under each of the null and the effect, at least 1.
        seed: Seed for ``numpy.random.default_rng``; same seed, same power.

    Returns:
        The simulated power in [0, 1].

    Raises:
        ValueError: on a non-finite ``effect``, a ``noise_sd`` that is not a
            positive finite real, ``n_a < 2`` or ``n_b < 2``, ``alpha`` outside
            (0, 1), ``n_sim < 1`` or a non-integer seed or count.
    """
    caller = "simulated_power_two_sample"
    if not _is_finite_real(effect):
        raise ValueError(f"{caller}: 'effect' must be a finite real, got {effect!r}.")
    if not _is_finite_real(noise_sd) or noise_sd <= 0.0:
        raise ValueError(f"{caller}: 'noise_sd' must be a finite real above 0, got {noise_sd!r}.")
    for name, count in (("n_a", n_a), ("n_b", n_b)):
        if not _is_int(count) or count < 2:
            raise ValueError(f"{caller}: '{name}' must be an integer >= 2, got {count!r}.")
    if not _is_finite_real(alpha) or not 0.0 < alpha < 1.0:
        raise ValueError(f"{caller}: 'alpha' must be in (0, 1), got {alpha!r}.")
    if not _is_int(n_sim) or n_sim < 1:
        raise ValueError(f"{caller}: 'n_sim' must be an integer >= 1, got {n_sim!r}.")
    if not _is_int(seed):
        raise ValueError(f"{caller}: 'seed' must be an integer, got {seed!r}.")

    rng = np.random.default_rng(seed)
    sd = float(noise_sd)
    null = _pooled_t_statistics(sd * rng.standard_normal((n_sim, n_a)), sd * rng.standard_normal((n_sim, n_b)))
    critical = float(np.quantile(null, 1.0 - float(alpha)))
    alternative = _pooled_t_statistics(float(effect) + sd * rng.standard_normal((n_sim, n_a)),
                                       sd * rng.standard_normal((n_sim, n_b)))
    return float(np.count_nonzero(alternative > critical) / n_sim)


# ---------------------------------------------------------------------------
# Statistics with their calculation exposed (qualified-model-catalogue 4.3, 5.3, 6)
# ---------------------------------------------------------------------------

#: Units of analysis a :class:`Statistic` may name.
STATISTIC_UNITS: tuple[str, ...] = ("prompt", "draw", "month", "filing")

#: Method label of :func:`clustered_auc_interval`, the one the finance simulation records.
CLUSTERED_AUC_METHOD = "prompt_clustered_stratified_percentile_bootstrap"

#: Method label of the intervals of :func:`paired_auc_difference`.
PAIRED_AUC_METHOD = "cluster-stratified percentile bootstrap of the paired AUC difference"

#: Method label of the intervals of :func:`agreement`.
CLUSTER_BOOTSTRAP_METHOD = "cluster percentile bootstrap"

#: Power and two-sided significance level of the reported detectable effect.
DETECTABLE_POWER = 0.8
DETECTABLE_ALPHA = 0.05

#: Half-width multiplier of the limits of agreement (Bland and Altman 1986).
LIMITS_OF_AGREEMENT_Z = 1.96

#: Names of the statistics :func:`agreement` returns, in order.
AGREEMENT_STATISTICS: tuple[str, ...] = (
    "mean_difference",
    "lower_limit_of_agreement",
    "upper_limit_of_agreement",
    "mean_absolute_difference",
    "rank_correlation",
)

# Two labellings with the same statistic may differ in the last bits; they count as ties.
_TIE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class Statistic:
    """One reported statistic with everything a reader needs to recompute and judge it.

    Attributes:
        name: Short name of the statistic.
        estimand: What quantity, over which units, under which condition, in words.
        formula: How the value is calculated from ``inputs``.
        inputs: The inputs of the formula with their counts. Stored as a read-only copy.
        unit: Unit of analysis, one of :data:`STATISTIC_UNITS`.
        n: Number of units of analysis.
        independent_units: False when the rows behind the value are not independent
            (several rows share a unit).
        value: The result, or None when it cannot be computed.
        interval: Its interval, or None.
        test: Its test, or None.
        effect_size: Its effect size, or None.
        detectable_effect: The smallest effect detectable at power 0.8 and two-sided
            alpha 0.05 (see :func:`detectable_effect`), or None.
        by_construction: True when the compared values are equal by construction;
            such a statistic carries no test.
        missing: The input that is missing when ``value`` is None, otherwise None.

    A statistic without a value names its missing input and carries no interval,
    test, effect size or detectable effect; it never invents a value.
    """

    name: str
    estimand: str
    formula: str
    inputs: Mapping[str, float | int]
    unit: str
    n: int
    independent_units: bool
    value: float | None
    interval: Interval | None
    test: TestResult | None
    effect_size: float | None
    detectable_effect: float | None
    by_construction: bool
    missing: str | None

    def __post_init__(self) -> None:
        for field in ("name", "estimand", "formula"):
            text = getattr(self, field)
            if not isinstance(text, str) or not text:
                raise ValueError(f"Statistic: '{field}' must be a non-empty string.")
        if self.unit not in STATISTIC_UNITS:
            raise ValueError(f"Statistic: 'unit' must be one of {STATISTIC_UNITS}, got {self.unit!r}.")
        if not _is_int(self.n) or self.n < 0:
            raise ValueError(f"Statistic: 'n' must be a non-negative integer, got {self.n!r}.")
        for field in ("independent_units", "by_construction"):
            if not isinstance(getattr(self, field), bool):
                raise ValueError(f"Statistic: '{field}' must be a bool.")
        for field in ("value", "effect_size", "detectable_effect"):
            number = getattr(self, field)
            if number is not None and not _is_finite_real(number):
                raise ValueError(f"Statistic: '{field}' must be a finite real or None, got {number!r}.")
        if self.value is None:
            if not isinstance(self.missing, str) or not self.missing:
                raise ValueError("Statistic: a statistic without a value must name its missing input in 'missing'.")
            if any(part is not None for part in (self.interval, self.test, self.effect_size, self.detectable_effect)):
                raise ValueError(
                    "Statistic: a statistic with no value carries no interval, test, effect size or detectable effect."
                )
        elif self.missing is not None:
            raise ValueError("Statistic: 'missing' must be None when the statistic has a value.")
        if self.by_construction and self.test is not None:
            raise ValueError("Statistic: values equal by construction are not presented as a test result.")
        object.__setattr__(self, "inputs", MappingProxyType(dict(self.inputs)))


def detectable_effect(standard_error: float, *, power: float = DETECTABLE_POWER, alpha: float = DETECTABLE_ALPHA
                      ) -> float:
    """Smallest effect a two-sided z test at ``alpha`` detects with ``power``, from a standard error.

    ``(z_{1 - alpha / 2} + z_{power}) * standard_error``; with the defaults
    ``(z_0.975 + z_0.80) * standard_error``, about ``2.80 * standard_error``.
    The standard error is the bootstrap one: the standard deviation of the
    resampled statistic.

    Raises:
        ValueError: on a negative or non-finite ``standard_error``, or ``power``
            or ``alpha`` outside (0, 1).
    """
    caller = "detectable_effect"
    if not _is_finite_real(standard_error) or standard_error < 0.0:
        raise ValueError(f"{caller}: 'standard_error' must be a finite real >= 0, got {standard_error!r}.")
    for name, level in (("power", power), ("alpha", alpha)):
        if not _is_finite_real(level) or not 0.0 < level < 1.0:
            raise ValueError(f"{caller}: '{name}' must be in (0, 1), got {level!r}.")
    normal = NormalDist()
    return float((normal.inv_cdf(1.0 - alpha / 2.0) + normal.inv_cdf(power)) * standard_error)


def _standard_error(resampled: np.ndarray) -> float:
    """Bootstrap standard error: the sample standard deviation of the resampled statistic."""
    return float(np.std(resampled, ddof=1)) if resampled.size > 1 else 0.0


def _two_sided_bootstrap_p(resampled: np.ndarray) -> float:
    """``min(1, 2 min(1 + #boot <= 0, 1 + #boot >= 0) / (1 + n))``, as :func:`paired_loss_difference`."""
    at_or_below = 1 + int(np.count_nonzero(resampled <= 0.0))
    at_or_above = 1 + int(np.count_nonzero(resampled >= 0.0))
    return min(1.0, 2.0 * min(at_or_below, at_or_above) / (1 + resampled.size))


def _check_count_and_seed(caller: str, name: str, count: object, seed: object) -> None:
    if not _is_int(count) or count < 1:
        raise ValueError(f"{caller}: '{name}' must be an integer >= 1, got {count!r}.")
    if not _is_int(seed) or seed < 0:
        raise ValueError(f"{caller}: 'seed' must be a non-negative integer, got {seed!r}.")


def _rows_by_unit(caller: str, name: str, units: object, n_rows: int) -> list[np.ndarray]:
    """Row indices of each unit (cluster or block), units in order of first appearance."""
    keys = np.asarray(units)
    if keys.ndim != 1 or keys.size != n_rows:
        raise ValueError(f"{caller}: '{name}' must be 1-D and aligned with the rows, got shape {keys.shape}.")
    rows: dict[object, list[int]] = {}
    for index, key in enumerate(keys.tolist()):
        rows.setdefault(key, []).append(index)
    return [np.asarray(indices, dtype=np.intp) for indices in rows.values()]


def _unit_classes(caller: str, name: str, classes: object, rows: list[np.ndarray], n_rows: int, *, unit: str,
                  kind: str, both: str) -> np.ndarray:
    """The 0/1 class of each unit, refusing a unit with two classes or an absent class."""
    values = np.asarray(classes)
    if values.ndim != 1 or values.size != n_rows:
        raise ValueError(f"{caller}: '{name}' must be 1-D and aligned with the rows, got shape {values.shape}.")
    if values.dtype != np.bool_ and not np.issubdtype(values.dtype, np.number):
        raise ValueError(f"{caller}: '{name}' values must be 0 or 1, got dtype {values.dtype}.")
    if not np.all((values == 0) | (values == 1)):
        raise ValueError(f"{caller}: '{name}' values must be 0 or 1.")
    values = values.astype(int)
    per_unit = np.asarray([values[indices[0]] for indices in rows], dtype=int)
    if any(np.any(values[indices] != values[indices[0]]) for indices in rows):
        raise ValueError(f"{caller}: every {unit} must keep one {kind}; a {unit} has rows of both.")
    if not 0 < int(per_unit.sum()) < per_unit.size:
        raise ValueError(f"{caller}: need both {both}, at least one {unit} of each.")
    return per_unit


def _aligned_scores(caller: str, name: str, scores: object, n_rows: int | None = None) -> np.ndarray:
    array = _real_array(caller, name, scores)
    if array.ndim != 1 or array.size == 0 or (n_rows is not None and array.size != n_rows):
        raise ValueError(f"{caller}: '{name}' must be a non-empty 1-D array aligned with the rows, got shape "
                         f"{array.shape}.")
    return array


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks from 1, with tied values sharing their average rank."""
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    return ((ends - counts + 1 + ends) / 2.0)[inverse]


def rank_auc(positive: np.ndarray, negative: np.ndarray) -> float:
    """ROC-AUC as the Mann-Whitney statistic with average ranks for ties (equal to ``roc_auc_score``).

    ``(R_pos - n_pos (n_pos + 1) / 2) / (n_pos n_neg)``, where ``R_pos`` is the
    sum of the ranks of the positive scores among all scores.
    """
    values = np.concatenate([positive, negative])
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    ranks = ((ends - counts + 1 + ends) / 2.0)[inverse]
    n_positive, n_negative = positive.size, negative.size
    return float((ranks[:n_positive].sum() - n_positive * (n_positive + 1) / 2) / (n_positive * n_negative))


def _class_scores(scores: np.ndarray, rows: list[np.ndarray], classes: np.ndarray
                  ) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """The scores of each positive and each negative cluster, clusters in order of first appearance."""
    positives = [scores[indices] for indices, label in zip(rows, classes, strict=True) if label == 1]
    negatives = [scores[indices] for indices, label in zip(rows, classes, strict=True) if label == 0]
    return positives, negatives


def _pooled_auc(positives: list[np.ndarray], negatives: list[np.ndarray]) -> float:
    return rank_auc(np.concatenate(positives), np.concatenate(negatives))


def clustered_auc_interval(scores: object, labels: object, clusters: object, *, n_resamples: int = 1000,
                           confidence: float = 0.95, seed: int = 0) -> Interval:
    """Percentile interval of the AUC from a stratified bootstrap over clusters (prompts).

    The AUC is :func:`rank_auc` over all rows (draws). Each resample draws, with
    replacement, as many positive clusters as there are and then as many
    negative clusters as there are, each with all its rows, and recomputes the
    AUC. The bounds are the ``(1 - confidence) / 2`` and
    ``1 - (1 - confidence) / 2`` quantiles under numpy's linear rule. Clusters
    are taken in order of first appearance, and one generator seeded from
    ``seed`` draws the positive indices before the negative ones in every
    resample, so the defaults reproduce the held-out interval of the finance
    simulation's calibration number for number.

    Args:
        scores: One score per row, finite reals.
        labels: 1 (positive) or 0 (negative) per row; a cluster keeps one label.
        clusters: Cluster id per row (hashable scalars).
        n_resamples: Number of resamples, at least 1. Default 1000.
        confidence: Two-sided coverage in (0, 1). Default 0.95.
        seed: Seed for ``numpy.random.default_rng``; same seed, same interval.

    Raises:
        ValueError: if the inputs are not aligned 1-D arrays, a label is not 0
            or 1, a cluster has both labels, a class is absent, or on an invalid
            ``n_resamples``, ``confidence`` or seed.
    """
    caller = "clustered_auc_interval"
    values = _aligned_scores(caller, "scores", scores)
    rows = _rows_by_unit(caller, "clusters", clusters, values.size)
    classes = _unit_classes(caller, "labels", labels, rows, values.size, unit="cluster", kind="label",
                            both="classes")
    _check_count_and_seed(caller, "n_resamples", n_resamples, seed)
    confidence = _check_confidence(caller, confidence)
    positives, negatives = _class_scores(values, rows, classes)

    rng = np.random.default_rng(seed)
    resampled = [_pooled_auc([positives[i] for i in rng.integers(len(positives), size=len(positives))],
                             [negatives[i] for i in rng.integers(len(negatives), size=len(negatives))])
                 for _ in range(n_resamples)]
    tail = (1 - confidence) / 2
    low, high = np.quantile(resampled, [tail, 1 - tail])
    return Interval(lo=float(low), hi=float(high), confidence=confidence, method=CLUSTERED_AUC_METHOD)


def permutation_auc_test(scores: object, labels: object, clusters: object, *, n_permutations: int = 2000,
                         seed: int = 0) -> TestResult:
    """Label permutation test of the AUC, permuting at cluster level.

    Null hypothesis: the scores carry no information about the label. The
    statistic is :func:`rank_auc` over all rows. Each permutation shuffles the
    labels over the clusters, so a cluster (prompt) keeps one label for all its
    rows and the number of positive clusters is kept, and recomputes the AUC.
    The one-sided p-value is ``(1 + #{permuted >= observed}) / (1 + n_permutations)``.

    Args:
        scores: One score per row, finite reals.
        labels: 1 (positive) or 0 (negative) per row; a cluster keeps one label.
        clusters: Cluster id per row.
        n_permutations: Number of permutations, at least 1. Default 2000.
        seed: Seed for ``numpy.random.default_rng``; same seed, same result.

    Returns:
        A :class:`TestResult` without an interval, whose parameters record ``n``
        (rows), ``n_clusters``, ``n_positive_clusters``, ``n_permutations``,
        ``seed`` and ``alternative``.

    Raises:
        ValueError: as :func:`clustered_auc_interval`, with ``n_permutations``
            in place of ``n_resamples``.
    """
    caller = "permutation_auc_test"
    values = _aligned_scores(caller, "scores", scores)
    rows = _rows_by_unit(caller, "clusters", clusters, values.size)
    classes = _unit_classes(caller, "labels", labels, rows, values.size, unit="cluster", kind="label",
                            both="classes")
    _check_count_and_seed(caller, "n_permutations", n_permutations, seed)

    # The ranks of the scores do not depend on the labels, so a labelling needs
    # only the rank sum and the row count of its positive clusters.
    ranks = _average_ranks(values)
    rank_sums = np.asarray([ranks[indices].sum() for indices in rows])
    sizes = np.asarray([indices.size for indices in rows], dtype=float)

    def auc(labelling: np.ndarray) -> np.ndarray:
        n_positive = labelling @ sizes
        return (labelling @ rank_sums - n_positive * (n_positive + 1) / 2) / (n_positive * (values.size - n_positive))

    observed = float(auc(classes.astype(float)))
    rng = np.random.default_rng(seed)
    permuted = auc(rng.permuted(np.tile(classes.astype(float), (n_permutations, 1)), axis=1))
    p_value = (1 + int(np.count_nonzero(permuted >= observed - _TIE_TOLERANCE))) / (1 + n_permutations)
    return TestResult(
        statistic=observed,
        interval=None,
        p_value=p_value,
        method=(
            "label permutation test of the AUC at cluster level (a cluster keeps one label); null: the scores "
            "carry no information about the label; one-sided p = (1 + #permuted >= observed) / (1 + n_permutations)"
        ),
        parameters={
            "n": int(values.size),
            "n_clusters": len(rows),
            "n_positive_clusters": int(classes.sum()),
            "n_permutations": n_permutations,
            "seed": seed,
            "alternative": "greater",
        },
    )


def paired_auc_difference(scores_a: object, scores_b: object, labels: object, clusters: object, *,
                          n_resamples: int = 2000, confidence: float = 0.95, seed: int = 0
                          ) -> tuple[float, Interval, TestResult]:
    """Difference of two AUCs on the same rows, with both recomputed on shared cluster resamples.

    The effect size is ``AUC(scores_a) - AUC(scores_b)``, each :func:`rank_auc`
    over all rows. The resampling is that of :func:`clustered_auc_interval`
    (stratified by label, whole clusters, the same use of the generator); both
    AUCs are recomputed on every resample, so the pairing survives. The interval
    is the percentile interval of the resampled difference. Null hypothesis:
    equal AUC; the two-sided p-value is
    ``min(1, 2 min(1 + #{boot <= 0}, 1 + #{boot >= 0}) / (1 + n_resamples))``.

    Args:
        scores_a: Scores of the first estimate, one per row.
        scores_b: Scores of the second estimate on the same rows.
        labels: 1 (positive) or 0 (negative) per row; a cluster keeps one label.
        clusters: Cluster id per row.
        n_resamples: Number of resamples, at least 1. Default 2000.
        confidence: Two-sided coverage in (0, 1). Default 0.95.
        seed: Seed for ``numpy.random.default_rng``; same seed, same result.

    Returns:
        The difference, its interval and the test. The test's parameters record
        ``n`` (rows), ``n_clusters``, ``n_positive_clusters``, ``auc_a``,
        ``auc_b``, ``n_resamples``, ``seed``, ``confidence``, ``alternative``,
        the bootstrap ``standard_error`` and the ``detectable_effect`` at
        ``power`` 0.8 and ``alpha`` 0.05 (see :func:`detectable_effect`).

    Raises:
        ValueError: as :func:`clustered_auc_interval`, also when the two score
            arrays are not aligned.
    """
    caller = "paired_auc_difference"
    a = _aligned_scores(caller, "scores_a", scores_a)
    b = _aligned_scores(caller, "scores_b", scores_b, a.size)
    rows = _rows_by_unit(caller, "clusters", clusters, a.size)
    classes = _unit_classes(caller, "labels", labels, rows, a.size, unit="cluster", kind="label", both="classes")
    _check_count_and_seed(caller, "n_resamples", n_resamples, seed)
    confidence = _check_confidence(caller, confidence)
    positives_a, negatives_a = _class_scores(a, rows, classes)
    positives_b, negatives_b = _class_scores(b, rows, classes)
    auc_a, auc_b = _pooled_auc(positives_a, negatives_a), _pooled_auc(positives_b, negatives_b)
    observed = auc_a - auc_b

    rng = np.random.default_rng(seed)
    resampled = np.empty(n_resamples)
    for index in range(n_resamples):
        drawn_positive = rng.integers(len(positives_a), size=len(positives_a))
        drawn_negative = rng.integers(len(negatives_a), size=len(negatives_a))
        resampled[index] = (
            _pooled_auc([positives_a[i] for i in drawn_positive], [negatives_a[i] for i in drawn_negative])
            - _pooled_auc([positives_b[i] for i in drawn_positive], [negatives_b[i] for i in drawn_negative])
        )
    tail = (1 - confidence) / 2
    low, high = np.quantile(resampled, [tail, 1 - tail])
    interval = Interval(lo=float(low), hi=float(high), confidence=confidence, method=PAIRED_AUC_METHOD)
    standard_error = _standard_error(resampled)
    test = TestResult(
        statistic=observed,
        interval=interval,
        p_value=_two_sided_bootstrap_p(resampled),
        method=(
            "paired AUC difference (a minus b) with both AUCs recomputed on the same cluster-stratified "
            "resamples; null: equal AUC; two-sided p = min(1, 2 min(1 + #boot <= 0, 1 + #boot >= 0) / "
            "(1 + n_resamples))"
        ),
        parameters={
            "n": int(a.size),
            "n_clusters": len(rows),
            "n_positive_clusters": int(classes.sum()),
            "auc_a": auc_a,
            "auc_b": auc_b,
            "n_resamples": n_resamples,
            "seed": seed,
            "confidence": confidence,
            "alternative": "two-sided",
            "standard_error": standard_error,
            "detectable_effect": detectable_effect(standard_error),
            "power": DETECTABLE_POWER,
            "alpha": DETECTABLE_ALPHA,
        },
    )
    return observed, interval, test


_AGREEMENT_ESTIMANDS: Mapping[str, tuple[str, str]] = MappingProxyType({
    "mean_difference": (
        "mean of the row differences a minus b, over all rows",
        "sum(a_i - b_i) / n_rows",
    ),
    "lower_limit_of_agreement": (
        "lower limit of agreement: the value above which about 95 percent of the row differences a minus b lie",
        "mean(d) - 1.96 * sd(d), d_i = a_i - b_i, sd with n_rows - 1 in the denominator",
    ),
    "upper_limit_of_agreement": (
        "upper limit of agreement: the value below which about 95 percent of the row differences a minus b lie",
        "mean(d) + 1.96 * sd(d), d_i = a_i - b_i, sd with n_rows - 1 in the denominator",
    ),
    "mean_absolute_difference": (
        "mean of the absolute row differences between a and b, over all rows",
        "sum(|a_i - b_i|) / n_rows",
    ),
    "rank_correlation": (
        "Spearman rank correlation of a and b over all rows",
        "Pearson correlation of the average ranks of a and the average ranks of b",
    ),
})


def _agreement_values(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, float, float]:
    """Mean difference, both limits of agreement, mean absolute difference and rank correlation (nan if undefined)."""
    difference = a - b
    mean = float(difference.mean())
    spread = float(difference.std(ddof=1)) if difference.size > 1 else math.nan
    ranks_a, ranks_b = _average_ranks(a), _average_ranks(b)
    centred_a, centred_b = ranks_a - ranks_a.mean(), ranks_b - ranks_b.mean()
    scale = math.sqrt(float(centred_a @ centred_a) * float(centred_b @ centred_b))
    correlation = float(centred_a @ centred_b) / scale if scale > 0.0 else math.nan
    return (mean, mean - LIMITS_OF_AGREEMENT_Z * spread, mean + LIMITS_OF_AGREEMENT_Z * spread,
            float(np.abs(difference).mean()), correlation)


def agreement(a: object, b: object, clusters: object, *, n_resamples: int = 2000, confidence: float = 0.95,
              seed: int = 0, unit: str = "prompt") -> Mapping[str, Statistic]:
    """Agreement of two estimates of the same rows, each statistic with its calculation.

    Returns, under the names of :data:`AGREEMENT_STATISTICS`: the mean
    difference ``a - b``, the lower and upper limits of agreement
    ``mean(d) -/+ 1.96 sd(d)`` (Bland and Altman), the mean absolute difference
    and the Spearman rank correlation. Each interval is the percentile interval
    of a bootstrap that draws whole clusters with replacement, as many as there
    are, so rows that share a cluster stay together; a resample on which a
    statistic is undefined is dropped for that statistic and counted in its
    ``inputs``. The mean difference carries the test of the null hypothesis
    that the mean difference is zero (two-sided bootstrap p-value), the
    standardised effect size ``mean(d) / sd(d)`` and the detectable effect from
    the bootstrap standard error; the rank correlation is its own effect size.

    A statistic that cannot be computed has no value and names what is missing:
    ``"a"``, ``"b"`` or ``"clusters"`` when that input is None or empty, fewer
    than two clusters, or no variation for the rank correlation.

    Args:
        a: First estimate, one finite real per row, or None.
        b: Second estimate on the same rows, or None.
        clusters: Cluster id per row, or None.
        n_resamples: Number of resamples, at least 1. Default 2000.
        confidence: Two-sided coverage in (0, 1). Default 0.95.
        seed: Seed for ``numpy.random.default_rng``; same seed, same result.
        unit: What a cluster is, one of :data:`STATISTIC_UNITS`. Default ``"prompt"``.

    Raises:
        ValueError: if present inputs are not aligned 1-D arrays of finite
            reals, or on an invalid ``n_resamples``, ``confidence``, seed or unit.
    """
    caller = "agreement"
    _check_count_and_seed(caller, "n_resamples", n_resamples, seed)
    confidence = _check_confidence(caller, confidence)
    if unit not in STATISTIC_UNITS:
        raise ValueError(f"{caller}: 'unit' must be one of {STATISTIC_UNITS}, got {unit!r}.")

    def absent(missing: str, n_rows: int = 0, n_clusters: int = 0, only: tuple[str, ...] = AGREEMENT_STATISTICS
               ) -> dict[str, Statistic]:
        return {
            name: Statistic(
                name=name, estimand=_AGREEMENT_ESTIMANDS[name][0], formula=_AGREEMENT_ESTIMANDS[name][1],
                inputs={"n_rows": n_rows, "n_clusters": n_clusters}, unit=unit, n=n_clusters,
                independent_units=n_rows == n_clusters, value=None, interval=None, test=None, effect_size=None,
                detectable_effect=None, by_construction=False, missing=missing,
            )
            for name in only
        }

    for name, values in (("a", a), ("b", b), ("clusters", clusters)):
        if values is None or np.asarray(values).size == 0:
            return MappingProxyType(absent(name))
    first = _aligned_scores(caller, "a", a)
    second = _aligned_scores(caller, "b", b, first.size)
    rows = _rows_by_unit(caller, "clusters", clusters, first.size)
    n_rows, n_clusters = int(first.size), len(rows)
    if n_clusters < 2:
        return MappingProxyType(absent("at least two clusters", n_rows, n_clusters))

    observed = _agreement_values(first, second)
    rng = np.random.default_rng(seed)
    resampled = np.empty((n_resamples, len(AGREEMENT_STATISTICS)))
    for index in range(n_resamples):
        drawn = np.concatenate([rows[i] for i in rng.integers(n_clusters, size=n_clusters)])
        resampled[index] = _agreement_values(first[drawn], second[drawn])

    difference = first - second
    spread = float(difference.std(ddof=1))
    shared = {"n_rows": n_rows, "n_clusters": n_clusters, "n_resamples": n_resamples}
    extra: dict[str, dict[str, float | int]] = {
        "mean_difference": {"sum_of_differences": float(difference.sum())},
        "lower_limit_of_agreement": {"mean_difference": observed[0], "sd_of_differences": spread},
        "upper_limit_of_agreement": {"mean_difference": observed[0], "sd_of_differences": spread},
        "mean_absolute_difference": {"sum_of_absolute_differences": float(np.abs(difference).sum())},
        "rank_correlation": {},
    }
    result: dict[str, Statistic] = {}
    for column, name in enumerate(AGREEMENT_STATISTICS):
        value = observed[column]
        if not math.isfinite(value):
            result |= absent("variation in a and in b (the rank correlation of a constant is undefined)",
                             n_rows, n_clusters, only=(name,))
            continue
        kept = resampled[:, column][np.isfinite(resampled[:, column])]
        inputs = shared | extra[name] | {"dropped_resamples": int(n_resamples - kept.size)}
        interval = None
        if kept.size:
            tail = (1 - confidence) / 2
            low, high = np.quantile(kept, [tail, 1 - tail])
            interval = Interval(lo=float(low), hi=float(high), confidence=confidence, method=CLUSTER_BOOTSTRAP_METHOD)
        test, effect_size, detectable = None, None, None
        if name == "mean_difference" and kept.size:
            standard_error = _standard_error(kept)
            detectable = detectable_effect(standard_error)
            effect_size = value / spread if spread > 0.0 else None
            test = TestResult(
                statistic=value,
                interval=interval,
                p_value=_two_sided_bootstrap_p(kept),
                method=(
                    "cluster bootstrap test of the mean difference (a minus b); null: the mean difference is zero; "
                    "two-sided p = min(1, 2 min(1 + #boot <= 0, 1 + #boot >= 0) / (1 + n_kept))"
                ),
                parameters={
                    "n": n_rows, "n_clusters": n_clusters, "n_resamples": n_resamples, "seed": seed,
                    "confidence": confidence, "alternative": "two-sided", "standard_error": standard_error,
                    "detectable_effect": detectable, "power": DETECTABLE_POWER, "alpha": DETECTABLE_ALPHA,
                    "dropped_resamples": int(n_resamples - kept.size),
                },
            )
        elif name == "rank_correlation":
            effect_size = value
        result[name] = Statistic(
            name=name, estimand=_AGREEMENT_ESTIMANDS[name][0], formula=_AGREEMENT_ESTIMANDS[name][1], inputs=inputs,
            unit=unit, n=n_clusters, independent_units=n_rows == n_clusters, value=value, interval=interval,
            test=test, effect_size=effect_size, detectable_effect=detectable, by_construction=False, missing=None,
        )
    return MappingProxyType(result)


def permutation_mean_difference(values: object, groups: object, blocks: object, *, alternative: str = "greater",
                                n_permutations: int = 2000, seed: int = 0) -> TestResult:
    """Permutation test of a difference of two group means, permuting at block level.

    The statistic is ``mean(values in group 1) - mean(values in group 0)`` over
    rows. Null hypothesis: the two groups have the same mean. Each permutation
    shuffles the group label over the blocks, so a block (month) keeps one
    label for all its rows and the number of blocks in group 1 is kept, and
    recomputes the difference. With ``m`` permutations the p-value is
    ``(1 + #{permuted >= observed}) / (1 + m)`` for ``"greater"``,
    ``(1 + #{permuted <= observed}) / (1 + m)`` for ``"less"`` and
    ``(1 + #{|permuted| >= |observed|}) / (1 + m)`` for ``"two-sided"``.

    Args:
        values: One finite real per row.
        groups: 1 or 0 per row; a block keeps one group.
        blocks: Block id per row.
        alternative: ``"greater"`` (default: group 1 has the higher mean),
            ``"less"`` or ``"two-sided"``.
        n_permutations: Number of permutations, at least 1. Default 2000.
        seed: Seed for ``numpy.random.default_rng``; same seed, same result.

    Returns:
        A :class:`TestResult` without an interval, whose parameters record ``n``
        (rows), ``n_blocks``, ``n_blocks_group_1``, ``mean_group_1``,
        ``mean_group_0``, ``n_permutations``, ``seed`` and ``alternative``.

    Raises:
        ValueError: if the inputs are not aligned 1-D arrays, a value is not
            finite, a group is not 0 or 1, a block has both groups, a group is
            absent, or on an invalid ``alternative``, ``n_permutations`` or seed.
    """
    caller = "permutation_mean_difference"
    if alternative not in ("greater", "less", "two-sided"):
        raise ValueError(f"{caller}: 'alternative' must be 'greater', 'less' or 'two-sided', got {alternative!r}.")
    data = _aligned_scores(caller, "values", values)
    rows = _rows_by_unit(caller, "blocks", blocks, data.size)
    classes = _unit_classes(caller, "groups", groups, rows, data.size, unit="block", kind="group", both="groups")
    _check_count_and_seed(caller, "n_permutations", n_permutations, seed)
    sums = np.asarray([data[indices].sum() for indices in rows])
    sizes = np.asarray([indices.size for indices in rows], dtype=float)
    total, n_rows = float(sums.sum()), float(data.size)

    def means(labelling: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        in_first, n_first = labelling @ sums, labelling @ sizes
        return in_first / n_first, (total - in_first) / (n_rows - n_first)

    mean_first, mean_other = means(classes.astype(float))
    observed = float(mean_first - mean_other)
    rng = np.random.default_rng(seed)
    permuted_first, permuted_other = means(rng.permuted(np.tile(classes.astype(float), (n_permutations, 1)), axis=1))
    permuted = permuted_first - permuted_other
    tolerance = _TIE_TOLERANCE * max(1.0, abs(observed))
    if alternative == "greater":
        extreme = permuted >= observed - tolerance
    elif alternative == "less":
        extreme = permuted <= observed + tolerance
    else:
        extreme = np.abs(permuted) >= abs(observed) - tolerance
    return TestResult(
        statistic=observed,
        interval=None,
        p_value=(1 + int(np.count_nonzero(extreme))) / (1 + n_permutations),
        method=(
            "permutation test of the mean difference (group 1 minus group 0) at block level (a block keeps one "
            f"group); null: the two groups have the same mean; alternative: {alternative}; "
            "p = (1 + #permuted at least as extreme as observed) / (1 + n_permutations)"
        ),
        parameters={
            "n": int(data.size),
            "n_blocks": len(rows),
            "n_blocks_group_1": int(classes.sum()),
            "mean_group_1": float(mean_first),
            "mean_group_0": float(mean_other),
            "n_permutations": n_permutations,
            "seed": seed,
            "alternative": alternative,
        },
    )
