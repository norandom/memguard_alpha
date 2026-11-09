"""Significance helpers, checked against hand-computed values.

Covers requirements 8.3, 8.7 and 8.8 of the memorization-adjusted-finance-sim
spec: the Holm step-down adjustment, the exact one-sided binomial tail, the
moving-block bootstrap over ordered decision months, the difference-in-differences
accuracy gap and the Monte Carlo random-answerer null.
"""

from __future__ import annotations

import logging
import math
from fractions import Fraction

import numpy as np
import pytest

from recall_guard.core.significance import (
    Interval,
    TestResult,
    block_bootstrap,
    did_gap,
    exact_binomial_greater,
    holm,
    random_answerer_null,
)

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


def test_interval_holds_its_fields() -> None:
    interval = Interval(lo=-0.5, hi=1.5, confidence=0.95, method="moving-block percentile bootstrap")
    assert (interval.lo, interval.hi, interval.confidence) == (-0.5, 1.5, 0.95)
    assert interval.method == "moving-block percentile bootstrap"


@pytest.mark.parametrize(
    ("lo", "hi", "confidence", "method"),
    [
        (1.0, 0.0, 0.95, "m"),
        (math.nan, 1.0, 0.95, "m"),
        (0.0, math.inf, 0.95, "m"),
        (0.0, 1.0, 0.0, "m"),
        (0.0, 1.0, 1.0, "m"),
        (0.0, 1.0, 0.95, ""),
    ],
)
def test_invalid_intervals_are_refused(lo, hi, confidence, method) -> None:
    with pytest.raises(ValueError):
        Interval(lo=lo, hi=hi, confidence=confidence, method=method)


def test_test_result_parameters_are_read_only_copies() -> None:
    parameters = {"n": 3, "seed": 7}
    result = TestResult(statistic=1.0, interval=None, p_value=0.5, method="m", parameters=parameters)
    parameters["n"] = 99
    assert result.parameters["n"] == 3
    with pytest.raises(TypeError):
        result.parameters["n"] = 4  # type: ignore[index]


@pytest.mark.parametrize("p_value", [-0.1, 1.1, math.nan])
def test_test_result_refuses_an_invalid_p_value(p_value) -> None:
    with pytest.raises(ValueError):
        TestResult(statistic=1.0, interval=None, p_value=p_value, method="m", parameters={})


def test_test_result_refuses_a_missing_method() -> None:
    with pytest.raises(ValueError):
        TestResult(statistic=1.0, interval=None, p_value=None, method="", parameters={})


# ---------------------------------------------------------------------------
# Holm
# ---------------------------------------------------------------------------


def test_holm_matches_a_hand_computed_example() -> None:
    # m = 4. Ascending: d 0.005*4 = 0.02; a 0.01*3 = 0.03; c 0.03*2 = 0.06;
    # b 0.04*1 = 0.04, raised to the running maximum 0.06.
    adjusted = holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.005})
    assert adjusted == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06, "d": 0.02})


def test_holm_keeps_the_keys_and_their_order() -> None:
    raw = {"z": 0.2, "a": 0.01, "m": 0.04}
    assert list(holm(raw)) == list(raw)


def test_holm_breaks_ties_by_key_and_stays_monotone() -> None:
    # Ties sort by key: x is first (0.02*4 = 0.08), y second (0.02*3 = 0.06,
    # raised to 0.08), then w (0.03*2 = 0.06, raised to 0.08), then v (0.5).
    adjusted = holm({"y": 0.02, "x": 0.02, "w": 0.03, "v": 0.5})
    assert adjusted == pytest.approx({"y": 0.08, "x": 0.08, "w": 0.08, "v": 0.5})


def test_holm_adjusted_order_follows_raw_order() -> None:
    raw = {"a": 0.001, "b": 0.012, "c": 0.013, "d": 0.2, "e": 0.03, "f": 0.031}
    adjusted = holm(raw)
    ordered = sorted(raw, key=lambda key: (raw[key], key))
    values = [adjusted[key] for key in ordered]
    assert values == sorted(values)
    assert all(adjusted[key] >= raw[key] for key in raw)


def test_holm_caps_at_one() -> None:
    # 0.6*2 = 1.2 caps at 1; 0.7 is raised to the running maximum 1.
    assert holm({"a": 0.6, "b": 0.7}) == {"a": 1.0, "b": 1.0}


def test_holm_with_one_test_is_the_raw_value() -> None:
    assert holm({"only": 0.037}) == {"only": 0.037}


def test_holm_of_no_tests_is_empty() -> None:
    assert holm({}) == {}


@pytest.mark.parametrize("bad", [-0.01, 1.01, math.nan, math.inf, True, "0.1"])
def test_holm_refuses_invalid_p_values(bad) -> None:
    with pytest.raises(ValueError):
        holm({"a": 0.01, "b": bad})


# ---------------------------------------------------------------------------
# Exact binomial
# ---------------------------------------------------------------------------


def test_binomial_tail_matches_the_hand_value() -> None:
    # P(X >= 8 | n = 10, p = 0.5) = (C(10,8) + C(10,9) + C(10,10)) / 2^10 = 56/1024.
    assert exact_binomial_greater(8, 10) == pytest.approx(56 / 1024, rel=1e-12)


def test_binomial_tail_with_an_uneven_probability() -> None:
    # 10*0.3^3*0.7^2 + 5*0.3^4*0.7 + 0.3^5 = 0.1323 + 0.02835 + 0.00243.
    assert exact_binomial_greater(3, 5, 0.3) == pytest.approx(0.16308, rel=1e-12)


def test_binomial_tail_edges() -> None:
    assert exact_binomial_greater(0, 10) == 1.0
    assert exact_binomial_greater(10, 10) == pytest.approx(1 / 1024, rel=1e-12)
    assert exact_binomial_greater(0, 0) == 1.0


@pytest.mark.parametrize(("k", "n", "p"), [(1000, 2000, 0.5), (60, 100, 0.5), (5, 400, 0.01), (1500, 2000, 0.7)])
def test_binomial_tail_matches_exact_fractions_for_large_n(k, n, p) -> None:
    q = Fraction(p).limit_denominator(1000)
    exact = sum(math.comb(n, j) * q**j * (1 - q) ** (n - j) for j in range(k, n + 1))
    assert exact_binomial_greater(k, n, p) == pytest.approx(float(exact), rel=1e-9)


def test_binomial_tail_stays_finite_far_in_the_tail() -> None:
    # 2^-2000 is below the smallest double, so the value underflows to zero
    # without overflow or NaN on the way; 2^-1000 is still representable.
    assert exact_binomial_greater(2000, 2000, 0.5) == 0.0
    assert exact_binomial_greater(1000, 1000, 0.5) == pytest.approx(2.0**-1000, rel=1e-9)


@pytest.mark.parametrize(
    ("k", "n", "p"),
    [(-1, 10, 0.5), (11, 10, 0.5), (3, 10, 0.0), (3, 10, 1.0), (3, 10, math.nan), (True, 10, 0.5), (3.0, 10, 0.5), (0, -1, 0.5)],
)
def test_binomial_tail_validates_its_arguments(k, n, p) -> None:
    with pytest.raises(ValueError):
        exact_binomial_greater(k, n, p)


# ---------------------------------------------------------------------------
# Moving-block bootstrap
# ---------------------------------------------------------------------------


def _mean(rows: np.ndarray) -> float:
    return float(np.mean(rows))


def test_block_bootstrap_is_identical_for_the_same_seed() -> None:
    blocks = np.sin(np.arange(40, dtype=float))
    a = block_bootstrap(_mean, blocks, block_length=4, n_resamples=500, seed=11)
    b = block_bootstrap(_mean, blocks, block_length=4, n_resamples=500, seed=11)
    assert a == b
    assert a.lo < a.hi
    assert a.confidence == 0.95
    assert a.method == "moving-block percentile bootstrap"


def test_block_bootstrap_differs_for_different_seeds() -> None:
    blocks = np.sin(np.arange(40, dtype=float))
    a = block_bootstrap(_mean, blocks, block_length=4, n_resamples=500, seed=11)
    b = block_bootstrap(_mean, blocks, block_length=4, n_resamples=500, seed=12)
    assert (a.lo, a.hi) != (b.lo, b.hi)


def test_block_bootstrap_draws_non_circular_moving_blocks() -> None:
    n, length, seed = 10, 3, 5
    seen: list[np.ndarray] = []

    def record(rows: np.ndarray) -> float:
        seen.append(rows.copy())
        return float(rows.sum())

    block_bootstrap(record, np.arange(n), block_length=length, n_resamples=4, seed=seed)

    rng = np.random.default_rng(seed)
    for rows in seen:
        starts = rng.integers(0, n - length + 1, size=math.ceil(n / length))
        expected = np.concatenate([np.arange(start, start + length) for start in starts])[:n]
        np.testing.assert_array_equal(rows, expected)
        assert len(rows) == n
        assert rows.max() <= n - 1


def test_block_bootstrap_keeps_each_month_row_together() -> None:
    months = np.arange(12)
    blocks = np.column_stack([months, months * 10, months * 100])

    def paired(rows: np.ndarray) -> float:
        assert rows.shape == (12, 3)
        np.testing.assert_array_equal(rows[:, 1], rows[:, 0] * 10)
        np.testing.assert_array_equal(rows[:, 2], rows[:, 0] * 100)
        return float(rows[:, 0].mean())

    block_bootstrap(paired, blocks, block_length=5, n_resamples=50, seed=0)


def test_block_bootstrap_with_one_full_block_is_degenerate() -> None:
    blocks = np.array([1.0, 4.0, 2.0, 8.0])
    interval = block_bootstrap(_mean, blocks, block_length=4, n_resamples=20, seed=3)
    assert interval.lo == interval.hi == pytest.approx(3.75)


def test_block_bootstrap_uses_percentile_bounds() -> None:
    blocks = np.arange(20, dtype=float) ** 2
    interval = block_bootstrap(_mean, blocks, block_length=2, n_resamples=300, seed=9, confidence=0.9)

    rng = np.random.default_rng(9)
    stats = []
    for _ in range(300):
        starts = rng.integers(0, 19, size=10)
        idx = (starts[:, None] + np.arange(2)).ravel()[:20]
        stats.append(float(np.mean(blocks[idx])))
    assert interval.lo == pytest.approx(float(np.percentile(stats, 5.0)))
    assert interval.hi == pytest.approx(float(np.percentile(stats, 95.0)))
    assert interval.confidence == 0.9


def test_block_bootstrap_drops_undefined_resamples_with_one_warning(caplog) -> None:
    calls = {"n": 0}

    def flaky(rows: np.ndarray) -> float:
        calls["n"] += 1
        if calls["n"] % 3 == 0:
            raise ValueError("empty cell")
        if calls["n"] % 3 == 1:
            return float(rows.mean())
        return math.nan

    with caplog.at_level(logging.WARNING, logger="recall_guard.core.significance"):
        interval = block_bootstrap(flaky, np.arange(12, dtype=float), block_length=3, n_resamples=30, seed=1)
    assert interval.lo <= interval.hi
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "20 / 30" in warnings[0].getMessage()


def test_block_bootstrap_refuses_when_every_resample_is_undefined() -> None:
    def never(rows: np.ndarray) -> float:
        raise ValueError("empty cell")

    with pytest.raises(ValueError, match="every resample"):
        block_bootstrap(never, np.arange(6), block_length=2, n_resamples=5, seed=0)


@pytest.mark.parametrize(
    ("blocks", "kwargs"),
    [
        (np.array([]), {}),
        (np.array(3.0), {}),
        (np.arange(6), {"block_length": 0}),
        (np.arange(6), {"block_length": 7}),
        (np.arange(6), {"block_length": True}),
        (np.arange(6), {"n_resamples": 0}),
        (np.arange(6), {"confidence": 0.0}),
        (np.arange(6), {"confidence": 1.0}),
        (np.arange(6), {"seed": 1.5}),
    ],
)
def test_block_bootstrap_validates_its_arguments(blocks, kwargs) -> None:
    arguments = {"block_length": 2, "n_resamples": 10, "seed": 0} | kwargs
    with pytest.raises(ValueError):
        block_bootstrap(_mean, blocks, **arguments)


# ---------------------------------------------------------------------------
# Difference-in-differences accuracy gap
# ---------------------------------------------------------------------------


def _did_table():
    # Pre identifying 3/4, pre anonymised 1/4: +0.5.
    # Post identifying 1/2, post anonymised 3/4: -0.25.  Gap 0.5 - (-0.25) = 0.75.
    rows = [
        (1, True, True), (1, True, True), (1, True, True), (0, True, True),
        (1, False, True), (0, False, True), (0, False, True), (0, False, True),
        (1, True, False), (0, True, False),
        (1, False, False), (0, False, False), (1, False, False), (1, False, False),
    ]  # fmt: skip
    correct = np.array([row[0] for row in rows], dtype=bool)
    identifying = np.array([row[1] for row in rows], dtype=bool)
    pre_cutoff = np.array([row[2] for row in rows], dtype=bool)
    return correct, identifying, pre_cutoff


def test_did_gap_matches_a_hand_built_table() -> None:
    assert did_gap(*_did_table()) == pytest.approx(0.75)


def test_did_gap_accepts_zero_one_accuracy() -> None:
    correct, identifying, pre_cutoff = _did_table()
    assert did_gap(correct.astype(float), identifying, pre_cutoff) == pytest.approx(0.75)


def test_did_gap_does_not_depend_on_item_order() -> None:
    correct, identifying, pre_cutoff = _did_table()
    order = np.random.default_rng(0).permutation(len(correct))
    assert did_gap(correct[order], identifying[order], pre_cutoff[order]) == pytest.approx(0.75)


@pytest.mark.parametrize(("identifying_value", "pre_value"), [(True, True), (False, True), (True, False), (False, False)])
def test_did_gap_refuses_an_empty_cell(identifying_value, pre_value) -> None:
    correct, identifying, pre_cutoff = _did_table()
    keep = ~((identifying == identifying_value) & (pre_cutoff == pre_value))
    with pytest.raises(ValueError, match="empty"):
        did_gap(correct[keep], identifying[keep], pre_cutoff[keep])


def test_did_gap_validates_its_arrays() -> None:
    correct, identifying, pre_cutoff = _did_table()
    with pytest.raises(ValueError):
        did_gap(correct[:-1], identifying, pre_cutoff)
    with pytest.raises(ValueError):
        did_gap(correct, identifying.astype(int), pre_cutoff)
    with pytest.raises(ValueError):
        did_gap(correct.reshape(2, -1), identifying.reshape(2, -1), pre_cutoff.reshape(2, -1))
    bad = correct.astype(float)
    bad[0] = 1.5
    with pytest.raises(ValueError):
        did_gap(bad, identifying, pre_cutoff)
    bad[0] = math.nan
    with pytest.raises(ValueError):
        did_gap(bad, identifying, pre_cutoff)


# ---------------------------------------------------------------------------
# Random-answerer null
# ---------------------------------------------------------------------------


def _cycling_sampler(values):
    """A deterministic sampler: the i-th null sample is all ``values[i]``."""
    state = {"i": 0}
    calls: list[tuple[object, int]] = []

    def sampler(rng: np.random.Generator, size: int) -> np.ndarray:
        calls.append((rng, size))
        value = values[state["i"] % len(values)]
        state["i"] += 1
        return np.full(size, value)

    return sampler, calls


def _largest(sample: np.ndarray) -> float:
    return float(np.max(sample))


def test_random_null_p_value_formula_on_a_deterministic_sampler() -> None:
    sampler, calls = _cycling_sampler(list(range(1, 11)))
    observed = np.array([0, 7, 3])
    result = random_answerer_null(observed, sampler, statistic=_largest, n_null=10, seed=4)

    # Null statistics 1..10; four of them (7, 8, 9, 10) are >= 7: p = (1 + 4) / (1 + 10).
    assert result.statistic == 7.0
    assert result.p_value == pytest.approx(5 / 11)
    assert len(calls) == 10
    assert all(isinstance(rng, np.random.Generator) and size == 3 for rng, size in calls)
    # Central 95 percent range of 1..10 by linear percentiles: 1 + 0.025*9, 1 + 0.975*9.
    assert result.interval is not None
    assert result.interval.lo == pytest.approx(1.225)
    assert result.interval.hi == pytest.approx(9.775)
    assert result.interval.confidence == 0.95
    assert "null" in result.interval.method
    assert "null" in result.method and "95" in result.method
    assert dict(result.parameters) == {
        "n": 3,
        "n_null": 10,
        "seed": 4,
        "confidence": 0.95,
        "alternative": "greater",
    }


def test_random_null_p_value_is_never_zero() -> None:
    sampler, _ = _cycling_sampler([0])
    result = random_answerer_null(np.array([5]), sampler, statistic=_largest, n_null=99, seed=0)
    assert result.p_value == pytest.approx(1 / 100)


def test_random_null_detects_a_perfect_answerer_over_a_decision_space() -> None:
    truth = np.array([1, -1, 1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, 1])

    def coin(rng: np.random.Generator, size: int) -> np.ndarray:
        return rng.choice(np.array([-1, 1]), size=size)

    def agreement(answers: np.ndarray) -> float:
        return float(np.mean(answers == truth))

    perfect = random_answerer_null(truth.copy(), coin, statistic=agreement, n_null=2000, seed=7)
    assert perfect.statistic == 1.0
    assert perfect.p_value == pytest.approx(1 / 2001)

    again = random_answerer_null(truth.copy(), coin, statistic=agreement, n_null=2000, seed=7)
    assert again == perfect

    coin_flip = random_answerer_null(coin(np.random.default_rng(1), 20), coin, statistic=agreement, n_null=2000, seed=7)
    assert coin_flip.p_value > 0.01


def test_random_null_refuses_a_wrong_sized_sample() -> None:
    def short(rng: np.random.Generator, size: int) -> np.ndarray:
        return np.zeros(size - 1)

    with pytest.raises(ValueError, match="size"):
        random_answerer_null(np.zeros(4), short, statistic=_largest, n_null=5, seed=0)


def test_random_null_refuses_an_undefined_statistic() -> None:
    sampler, _ = _cycling_sampler([1])
    with pytest.raises(ValueError):
        random_answerer_null(np.zeros(3), sampler, statistic=lambda sample: math.nan, n_null=5, seed=0)


@pytest.mark.parametrize(
    ("observed", "kwargs"),
    [
        (np.array([]), {}),
        (np.array(1.0), {}),
        (np.zeros(3), {"n_null": 0}),
        (np.zeros(3), {"n_null": True}),
        (np.zeros(3), {"seed": "0"}),
        (np.zeros(3), {"confidence": 1.5}),
    ],
)
def test_random_null_validates_its_arguments(observed, kwargs) -> None:
    sampler, _ = _cycling_sampler([1])
    arguments = {"statistic": _largest, "n_null": 5, "seed": 0} | kwargs
    with pytest.raises(ValueError):
        random_answerer_null(observed, sampler, **arguments)
