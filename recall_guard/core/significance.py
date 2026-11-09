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

Intervals use the percentile convention of :mod:`recall_guard.core.bootstrap`:
two-sided, with ``confidence`` the coverage, bounds at the ``(1 - confidence) / 2``
and ``1 - (1 - confidence) / 2`` percentiles under numpy's linear rule.
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


def _is_int(value: object) -> bool:
    # bool is an int subclass; True as a count or seed is a caller bug.
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _is_finite_real(value: object) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(value)


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
    cell) or returns a non-finite value are dropped, with one
    ``logging.WARNING`` per call giving the count, as in
    :func:`recall_guard.core.bootstrap.bootstrap_ci`.

    Args:
        stat: Statistic of an array shaped like ``blocks``; returns a float.
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
    rows = np.asarray(blocks)
    if rows.ndim == 0 or rows.shape[0] == 0:
        raise ValueError("block_bootstrap: 'blocks' must have at least one row along axis 0.")
    n = rows.shape[0]
    if not _is_int(block_length) or not 1 <= block_length <= n:
        raise ValueError(f"block_bootstrap: 'block_length' must be an integer in 1..{n}, got {block_length!r}.")
    if not _is_int(n_resamples) or n_resamples < 1:
        raise ValueError(f"block_bootstrap: 'n_resamples' must be an integer >= 1, got {n_resamples!r}.")
    if not _is_int(seed):
        raise ValueError(f"block_bootstrap: 'seed' must be an integer, got {seed!r}.")
    confidence = _check_confidence("block_bootstrap", confidence)

    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(n / block_length)
    offsets = np.arange(block_length)
    stats: list[float] = []
    dropped = 0
    for _ in range(n_resamples):
        starts = rng.integers(0, n - block_length + 1, size=n_blocks)
        index = (starts[:, None] + offsets).ravel()[:n]
        try:
            value = stat(rows[index])
        except ValueError:
            dropped += 1
            continue
        if not _is_finite_real(value):
            dropped += 1
            continue
        stats.append(float(value))

    if dropped:
        logger.warning(
            "block_bootstrap: dropped %d / %d resamples where the statistic raised or returned a non-finite value.",
            dropped,
            n_resamples,
        )
    if not stats:
        raise ValueError("block_bootstrap: every resample was dropped; the statistic is undefined on these blocks.")
    lo, hi = _percentile_bounds(np.asarray(stats, dtype=float), confidence)
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

    Raises:
        ValueError: on an empty or zero-dimensional ``observed``, ``n_null < 1``,
            a non-integer seed, ``confidence`` outside (0, 1), a sampler that
            returns the wrong number of answers, or a non-finite statistic.
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
        value = statistic(sample)
        if not _is_finite_real(value):
            raise ValueError(f"random_answerer_null: the statistic of the {what} is not finite: {value!r}.")
        return float(value)

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
        parameters={"n": n, "n_null": n_null, "seed": seed, "confidence": confidence, "alternative": "greater"},
    )
