"""Significance helpers, checked against hand-computed values.

Covers requirements 7.2, 8.3, 8.4, 8.5, 8.7, 8.8 and 8.9 of the
memorization-adjusted-finance-sim spec: the Holm step-down adjustment, the exact
one-sided binomial tail, the moving-block bootstrap over ordered decision months,
the difference-in-differences accuracy gap, the Monte Carlo random-answerer null,
the paired loss difference, the half-ensemble variance comparison and the
simulated power.
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
    half_ensemble_variance,
    holm,
    paired_loss_difference,
    random_answerer_null,
    simulated_power,
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
        "dropped_resamples": 0,
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


# ---------------------------------------------------------------------------
# 0-d statistic values
# ---------------------------------------------------------------------------


def _mean_as_0d_array(rows: np.ndarray) -> np.ndarray:
    return np.asarray(np.mean(rows))


def test_block_bootstrap_coerces_a_0d_array_statistic(caplog) -> None:
    blocks = np.sin(np.arange(30, dtype=float))
    with caplog.at_level(logging.WARNING, logger="recall_guard.core.significance"):
        coerced = block_bootstrap(_mean_as_0d_array, blocks, block_length=3, n_resamples=200, seed=2)
    assert coerced == block_bootstrap(_mean, blocks, block_length=3, n_resamples=200, seed=2)
    assert not [record for record in caplog.records if record.levelno == logging.WARNING]


@pytest.mark.parametrize("value", [np.array(1 + 0j), np.array(True), np.array("0.5"), np.array(math.nan)])
def test_block_bootstrap_still_drops_a_0d_value_that_is_not_a_finite_real(value) -> None:
    with pytest.raises(ValueError, match="every resample"):
        block_bootstrap(lambda rows: value, np.arange(6.0), block_length=2, n_resamples=5, seed=0)


def test_random_null_coerces_a_0d_array_statistic() -> None:
    sampler, _ = _cycling_sampler(list(range(1, 11)))
    coerced = random_answerer_null(
        np.array([0, 7, 3]), sampler, statistic=lambda sample: np.asarray(np.max(sample)), n_null=10, seed=4
    )
    assert coerced.statistic == 7.0
    assert isinstance(coerced.statistic, float)
    assert coerced.p_value == pytest.approx(5 / 11)
    assert coerced.parameters["dropped_resamples"] == 0


@pytest.mark.parametrize("value", [np.array(1 + 0j), np.array(True), np.array(math.inf)])
def test_random_null_refuses_a_0d_value_that_is_not_a_finite_real(value) -> None:
    sampler, _ = _cycling_sampler([1])
    with pytest.raises(ValueError, match="finite real"):
        random_answerer_null(np.zeros(3), sampler, statistic=lambda sample: value, n_null=5, seed=0)


# ---------------------------------------------------------------------------
# Paired loss difference
# ---------------------------------------------------------------------------


def _paired_rows_difference(rows: np.ndarray) -> float:
    return float(rows[:, 0].mean() - rows[:, 1].mean())


@pytest.mark.parametrize("shape", [(24,), (24, 5)])
def test_paired_loss_difference_is_zero_on_identical_inputs(shape) -> None:
    losses = np.random.default_rng(0).uniform(0.0, 1.0, size=shape)
    result = paired_loss_difference(losses, losses.copy(), block_length=3, n_resamples=300, seed=1)
    assert result.statistic == 0.0
    assert result.interval is not None
    assert (result.interval.lo, result.interval.hi) == (0.0, 0.0)
    assert result.p_value == 1.0
    assert result.parameters["dropped_resamples"] == 0


def test_paired_loss_difference_matches_a_hand_computed_mean_difference() -> None:
    # mean(a) = 2.0 / 4 = 0.5, mean(b) = 1.0 / 4 = 0.25.
    loss_a = np.array([0.2, 0.4, 0.6, 0.8])
    loss_b = np.array([0.1, 0.1, 0.3, 0.5])
    result = paired_loss_difference(loss_a, loss_b, block_length=2, n_resamples=100, seed=0)
    assert result.statistic == pytest.approx(0.25)


def test_paired_loss_difference_of_hand_computed_brier_scores() -> None:
    # Outcomes (1, 0). Variant forecasts (0.8, 0.3): Brier 0.04 and 0.09.
    # Reference forecasts (0.6, 0.5): Brier 0.16 and 0.25. Two identical months.
    # Mean 0.065 minus mean 0.205 is -0.14.
    outcomes = np.array([[1.0, 0.0], [1.0, 0.0]])
    brier_a = (np.array([[0.8, 0.3], [0.8, 0.3]]) - outcomes) ** 2
    brier_b = (np.array([[0.6, 0.5], [0.6, 0.5]]) - outcomes) ** 2
    result = paired_loss_difference(brier_a, brier_b, block_length=1, n_resamples=49, seed=0)
    assert result.statistic == pytest.approx(-0.14)
    assert result.interval is not None
    assert result.interval.lo == pytest.approx(-0.14)
    assert result.interval.hi == pytest.approx(-0.14)
    # Every resample is below 0: p = 2 min(1 + 49, 1 + 0) / (1 + 49).
    assert result.p_value == pytest.approx(2 / 50)
    assert result.parameters["n"] == 2
    assert result.parameters["n_values"] == 4


def test_paired_loss_difference_p_value_on_a_constant_shift() -> None:
    loss_b = np.random.default_rng(3).uniform(0.0, 1.0, size=30)
    loss_a = loss_b + 0.1
    result = paired_loss_difference(loss_a, loss_b, block_length=5, n_resamples=99, seed=8)
    # Every resample is about +0.1: p = 2 min(1 + 0, 1 + 99) / (1 + 99).
    assert result.statistic == pytest.approx(0.1)
    assert result.p_value == pytest.approx(0.02)
    assert result.interval is not None
    assert result.interval.lo == pytest.approx(0.1)
    assert result.interval.hi == pytest.approx(0.1)


def test_paired_loss_difference_interval_is_the_block_bootstrap_of_the_difference() -> None:
    rng = np.random.default_rng(5)
    loss_a = rng.uniform(0.0, 1.0, size=40)
    loss_b = rng.uniform(0.0, 1.0, size=40)
    result = paired_loss_difference(loss_a, loss_b, block_length=4, n_resamples=400, seed=13, confidence=0.9)
    expected = block_bootstrap(
        _paired_rows_difference,
        np.column_stack([loss_a, loss_b]),
        block_length=4,
        n_resamples=400,
        seed=13,
        confidence=0.9,
    )
    assert result.interval is not None
    assert (result.interval.lo, result.interval.hi) == (expected.lo, expected.hi)
    assert result.interval.confidence == 0.9
    assert result.interval.method == "moving-block percentile bootstrap"


def test_paired_loss_difference_resamples_whole_month_rows_of_both_arrays() -> None:
    # 2-D: rows are months. mean(a) = 21 / 6 = 3.5, mean(b) = 6 / 6 = 1.
    loss_a = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    loss_b = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    result = paired_loss_difference(loss_a, loss_b, block_length=2, n_resamples=200, seed=4)
    assert result.statistic == pytest.approx(2.5)
    expected = block_bootstrap(
        _paired_rows_difference, np.stack([loss_a, loss_b], axis=1), block_length=2, n_resamples=200, seed=4
    )
    assert result.interval is not None
    assert (result.interval.lo, result.interval.hi) == (expected.lo, expected.hi)


def test_paired_loss_difference_is_identical_for_the_same_seed() -> None:
    rng = np.random.default_rng(6)
    loss_a = rng.uniform(0.0, 1.0, size=36)
    loss_b = rng.uniform(0.0, 1.0, size=36)
    first = paired_loss_difference(loss_a, loss_b, block_length=3, n_resamples=300, seed=21)
    again = paired_loss_difference(loss_a, loss_b, block_length=3, n_resamples=300, seed=21)
    other = paired_loss_difference(loss_a, loss_b, block_length=3, n_resamples=300, seed=22)
    assert first == again
    assert first.interval is not None and other.interval is not None
    assert (first.interval.lo, first.interval.hi) != (other.interval.lo, other.interval.hi)


def test_paired_loss_difference_is_antisymmetric() -> None:
    rng = np.random.default_rng(7)
    loss_a = rng.uniform(0.0, 1.0, size=36)
    loss_b = rng.uniform(0.0, 1.0, size=36)
    forward = paired_loss_difference(loss_a, loss_b, block_length=3, n_resamples=300, seed=2)
    backward = paired_loss_difference(loss_b, loss_a, block_length=3, n_resamples=300, seed=2)
    assert backward.statistic == pytest.approx(-forward.statistic)
    assert backward.p_value == forward.p_value
    assert forward.interval is not None and backward.interval is not None
    assert backward.interval.lo == pytest.approx(-forward.interval.hi)
    assert backward.interval.hi == pytest.approx(-forward.interval.lo)


def test_paired_loss_difference_records_its_parameters() -> None:
    result = paired_loss_difference(np.arange(6.0), np.zeros(6), block_length=2, n_resamples=10, seed=3)
    assert dict(result.parameters) == {
        "n": 6,
        "n_values": 6,
        "block_length": 2,
        "n_resamples": 10,
        "seed": 3,
        "confidence": 0.95,
        "alternative": "two-sided",
        "dropped_resamples": 0,
    }
    assert "two-sided" in result.method


@pytest.mark.parametrize(
    ("loss_a", "loss_b", "kwargs"),
    [
        (np.zeros(5), np.zeros(4), {}),
        (np.zeros((4, 2)), np.zeros(4), {}),
        (np.zeros((2, 2, 2)), np.zeros((2, 2, 2)), {}),
        (np.array([]), np.array([]), {}),
        (np.zeros((4, 0)), np.zeros((4, 0)), {}),
        (np.array([0.1, math.nan, 0.2, 0.3]), np.zeros(4), {}),
        (np.array(["a", "b", "c", "d"]), np.zeros(4), {}),
        (np.zeros(4, dtype=complex), np.zeros(4), {}),
        (np.zeros(4), np.zeros(4), {"block_length": 0}),
        (np.zeros(4), np.zeros(4), {"block_length": 5}),
        (np.zeros(4), np.zeros(4), {"n_resamples": 0}),
        (np.zeros(4), np.zeros(4), {"seed": 1.5}),
        (np.zeros(4), np.zeros(4), {"confidence": 1.0}),
    ],
)
def test_paired_loss_difference_validates_its_arguments(loss_a, loss_b, kwargs) -> None:
    arguments = {"block_length": 2, "n_resamples": 10, "seed": 0} | kwargs
    with pytest.raises(ValueError):
        paired_loss_difference(loss_a, loss_b, **arguments)


# ---------------------------------------------------------------------------
# Half-ensemble versus single-draw variance
# ---------------------------------------------------------------------------


def _independent_draws(n_items: int, n_draws: int, seed: int) -> np.ndarray:
    """Independent unit-variance draws around item-specific means, (items, draws)."""
    rng = np.random.default_rng(seed)
    item_means = rng.normal(0.0, 3.0, size=(n_items, 1))
    return item_means + rng.normal(0.0, 1.0, size=(n_items, n_draws))


@pytest.mark.parametrize("k", [4, 8, 16])
def test_half_ensembles_of_k_over_2_draws_give_a_ratio_near_2_over_k(k) -> None:
    draws = _independent_draws(4000, k, seed=k)
    half = k // 2
    result = half_ensemble_variance(
        draws[:, :half].mean(axis=1),
        draws[:, half:].mean(axis=1),
        draws[:, [0, half]],
        n_resamples=200,
        seed=1,
    )
    assert result.statistic == pytest.approx(2 / k, rel=0.12)
    assert result.interval is not None
    assert result.interval.hi < 1.0
    assert result.p_value == pytest.approx(1 / 201)
    assert result.parameters["dropped_resamples"] == 0


@pytest.mark.parametrize("k", [2, 5, 10])
def test_the_variance_of_a_mean_of_k_draws_is_near_one_kth_of_a_single_draw(k) -> None:
    # Two disjoint ensembles of K draws each, so each half is a mean of K draws.
    draws = _independent_draws(4000, 2 * k, seed=100 + k)
    result = half_ensemble_variance(
        draws[:, :k].mean(axis=1),
        draws[:, k:].mean(axis=1),
        draws[:, [0, k]],
        n_resamples=200,
        seed=2,
    )
    assert result.statistic == pytest.approx(1 / k, rel=0.12)


def test_half_ensemble_variance_matches_a_hand_computed_ratio() -> None:
    # Half pairs (1, 3) and (2, 2): 0.5 * mean(4, 0) = 1.
    # Single pairs (0, 4) and (1, 1): 0.5 * mean(16, 0) = 4. Ratio 1 / 4.
    result = half_ensemble_variance(
        np.array([1.0, 2.0]), np.array([3.0, 2.0]), np.array([[0.0, 4.0], [1.0, 1.0]]), n_resamples=50, seed=0
    )
    assert result.statistic == pytest.approx(0.25)
    assert result.parameters["half_ensemble_variance"] == pytest.approx(1.0)
    assert result.parameters["single_draw_variance"] == pytest.approx(4.0)
    assert result.interval is not None
    assert result.interval.method == "item-level percentile bootstrap"
    assert result.parameters["alternative"] == "less"


def test_half_ensemble_variance_of_agreeing_halves_is_zero() -> None:
    draws = _independent_draws(50, 2, seed=9)
    halves = np.full(50, 0.3)
    result = half_ensemble_variance(halves, halves.copy(), draws, n_resamples=99, seed=0)
    assert result.statistic == 0.0
    assert result.interval is not None
    assert (result.interval.lo, result.interval.hi) == (0.0, 0.0)
    assert result.p_value == pytest.approx(1 / 100)


def test_half_ensemble_variance_p_value_is_one_when_ensembles_vary_more() -> None:
    rng = np.random.default_rng(10)
    halves_a = rng.normal(0.0, 2.0, size=200)
    halves_b = rng.normal(0.0, 2.0, size=200)
    singles = rng.normal(0.0, 0.5, size=(200, 2))
    result = half_ensemble_variance(halves_a, halves_b, singles, n_resamples=99, seed=0)
    assert result.statistic > 1.0
    assert result.p_value == 1.0


def test_half_ensemble_variance_drops_and_records_resamples_without_a_ratio(caplog) -> None:
    # The single draws differ only on item 0, so a resample without item 0 has
    # a zero single-draw variance and is dropped.
    halves_a = np.array([0.0, 1.0, 2.0])
    halves_b = np.array([0.5, 1.0, 2.0])
    singles = np.array([[0.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    with caplog.at_level(logging.WARNING, logger="recall_guard.core.significance"):
        result = half_ensemble_variance(halves_a, halves_b, singles, n_resamples=200, seed=5)

    rng = np.random.default_rng(5)
    expected = sum(1 for _ in range(200) if 0 not in rng.integers(0, 3, size=3))
    assert expected > 0
    assert result.parameters["dropped_resamples"] == expected
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert f"{expected} / 200" in warnings[0].getMessage()


def test_half_ensemble_variance_is_identical_for_the_same_seed() -> None:
    draws = _independent_draws(300, 8, seed=4)
    arguments = (draws[:, :4].mean(axis=1), draws[:, 4:].mean(axis=1), draws[:, [0, 4]])
    first = half_ensemble_variance(*arguments, n_resamples=300, seed=31)
    again = half_ensemble_variance(*arguments, n_resamples=300, seed=31)
    other = half_ensemble_variance(*arguments, n_resamples=300, seed=32)
    assert first == again
    assert first.interval is not None and other.interval is not None
    assert (first.interval.lo, first.interval.hi) != (other.interval.lo, other.interval.hi)


def test_half_ensemble_variance_refuses_single_draws_that_always_agree() -> None:
    with pytest.raises(ValueError, match="single-draw variance is zero"):
        half_ensemble_variance(np.zeros(3), np.ones(3), np.ones((3, 2)), n_resamples=10, seed=0)


@pytest.mark.parametrize(
    ("halves_a", "halves_b", "singles", "kwargs"),
    [
        (np.zeros(3), np.zeros(4), np.arange(6.0).reshape(3, 2), {}),
        (np.zeros(3), np.zeros(3), np.arange(9.0).reshape(3, 3), {}),
        (np.zeros(3), np.zeros(3), np.arange(3.0), {}),
        (np.zeros((3, 1)), np.zeros((3, 1)), np.arange(6.0).reshape(3, 2), {}),
        (np.array([]), np.array([]), np.zeros((0, 2)), {}),
        (np.array([0.0, math.inf, 0.0]), np.zeros(3), np.arange(6.0).reshape(3, 2), {}),
        (np.array(["a", "b", "c"]), np.zeros(3), np.arange(6.0).reshape(3, 2), {}),
        (np.zeros(3), np.zeros(3), np.arange(6.0).reshape(3, 2), {"n_resamples": 0}),
        (np.zeros(3), np.zeros(3), np.arange(6.0).reshape(3, 2), {"seed": "1"}),
        (np.zeros(3), np.zeros(3), np.arange(6.0).reshape(3, 2), {"confidence": 0.0}),
    ],
)
def test_half_ensemble_variance_validates_its_arguments(halves_a, halves_b, singles, kwargs) -> None:
    arguments = {"n_resamples": 10, "seed": 0} | kwargs
    with pytest.raises(ValueError):
        half_ensemble_variance(halves_a, halves_b, singles, **arguments)


def _replayed_ratios(halves_a, halves_b, singles, *, n_resamples: int, seed: int) -> list[float | None]:
    """Replay the item-level resamples: the ratio, inf for x/0, or None for 0/0."""
    n = len(halves_a)
    rng = np.random.default_rng(seed)
    ratios: list[float | None] = []
    for _ in range(n_resamples):
        index = rng.integers(0, n, size=n)
        half = 0.5 * float(np.mean((halves_a[index] - halves_b[index]) ** 2))
        single = 0.5 * float(np.mean((singles[index, 0] - singles[index, 1]) ** 2))
        if single == 0.0:
            ratios.append(math.inf if half > 0.0 else None)
        else:
            ratios.append(half / single)
    return ratios


def test_half_ensemble_variance_counts_unbounded_resamples_against_the_ensemble() -> None:
    # Single draws differ only on item 0; the halves differ on items 0 and 1.
    # A resample without item 0 but with item 1 has ratio x/0 (unbounded); one
    # with only item 2 is 0/0 (dropped).
    halves_a = np.array([0.0, 1.0, 2.0])
    halves_b = np.array([0.5, 2.0, 2.0])
    singles = np.array([[0.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    result = half_ensemble_variance(halves_a, halves_b, singles, n_resamples=400, seed=5)

    ratios = _replayed_ratios(halves_a, halves_b, singles, n_resamples=400, seed=5)
    finite = [r for r in ratios if r is not None and math.isfinite(r)]
    unbounded = sum(1 for r in ratios if r == math.inf)
    dropped = sum(1 for r in ratios if r is None)
    assert unbounded > 0 and dropped > 0

    finite_at_least_one = sum(1 for r in finite if r >= 1.0)
    counted = (1 + finite_at_least_one + unbounded) / (1 + len(finite) + unbounded)
    dropped_unbounded = (1 + finite_at_least_one) / (1 + len(finite))
    assert result.p_value > dropped_unbounded
    assert result.p_value == pytest.approx(counted)

    # About 7/27 of the kept resamples are unbounded, above (1 - 0.95) / 2.
    assert result.interval is None
    assert result.parameters["unbounded_resamples"] == unbounded
    assert result.parameters["dropped_resamples"] == dropped
    assert result.parameters["interval_note"] == "upper bound unbounded"


def test_half_ensemble_variance_interval_ranks_a_few_unbounded_resamples_highest() -> None:
    # 40 items, single draws differ on 4 of them: about 0.9^40 = 1.5 percent of
    # resamples miss all 4 and are unbounded, below the 2.5 percent tail.
    rng = np.random.default_rng(20)
    halves_a = rng.normal(0.0, 1.0, size=40)
    halves_b = rng.normal(0.0, 1.0, size=40)
    first = rng.normal(0.0, 1.0, size=40)
    second = first.copy()
    second[:4] += rng.normal(0.0, 3.0, size=4)
    singles = np.column_stack([first, second])
    result = half_ensemble_variance(halves_a, halves_b, singles, n_resamples=2000, seed=3)

    ratios = np.array(_replayed_ratios(halves_a, halves_b, singles, n_resamples=2000, seed=3), dtype=float)
    unbounded = np.isposinf(ratios)
    assert 0 < int(unbounded.sum()) < 0.025 * ratios.size

    assert result.interval is not None
    assert math.isfinite(result.interval.lo) and math.isfinite(result.interval.hi)
    # Dropping the unbounded resamples would pull the upper bound down.
    assert result.interval.hi > float(np.percentile(ratios[~unbounded], 97.5))
    ranked_highest = np.where(unbounded, ratios[~unbounded].max(), ratios)
    assert result.interval.lo == pytest.approx(float(np.percentile(ranked_highest, 2.5)))
    assert result.interval.hi == pytest.approx(float(np.percentile(ranked_highest, 97.5)))
    expected_p = (1 + int(np.count_nonzero(ratios >= 1.0))) / (1 + ratios.size)
    assert result.p_value == pytest.approx(expected_p)

    assert result.parameters["unbounded_resamples"] == int(unbounded.sum())
    assert result.parameters["dropped_resamples"] == 0
    assert result.parameters["interval_note"] is None


def test_half_ensemble_variance_still_drops_and_counts_0_over_0_resamples() -> None:
    # Only item 0 has any spread, in both pairs, so a resample without it is 0/0.
    halves_a = np.array([0.0, 1.0, 2.0])
    halves_b = np.array([0.5, 1.0, 2.0])
    singles = np.array([[0.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    result = half_ensemble_variance(halves_a, halves_b, singles, n_resamples=200, seed=5)
    ratios = _replayed_ratios(halves_a, halves_b, singles, n_resamples=200, seed=5)
    assert result.parameters["dropped_resamples"] == sum(1 for r in ratios if r is None) > 0
    assert result.parameters["unbounded_resamples"] == 0
    assert result.parameters["interval_note"] is None
    assert result.interval is not None


def test_half_ensemble_variance_records_no_unbounded_resamples_on_ordinary_data() -> None:
    draws = _independent_draws(200, 8, seed=6)
    result = half_ensemble_variance(
        draws[:, :4].mean(axis=1), draws[:, 4:].mean(axis=1), draws[:, [0, 4]], n_resamples=100, seed=0
    )
    assert result.parameters["unbounded_resamples"] == 0
    assert result.parameters["interval_note"] is None
    assert result.interval is not None


@pytest.mark.parametrize("halves_b", [np.zeros(3), np.ones(3)])
def test_half_ensemble_variance_keeps_the_observed_statistic_finite(halves_b) -> None:
    # Zero single-draw variance on the original data raises for x/0 and 0/0 alike.
    with pytest.raises(ValueError, match="single-draw variance is zero"):
        half_ensemble_variance(np.zeros(3), halves_b, np.ones((3, 2)), n_resamples=10, seed=0)


def test_unbounded_percentile_bounds_follow_the_linear_rule() -> None:
    from recall_guard.core.significance import _percentile_bounds_ranking_unbounded_highest as bounds

    finite = np.arange(1.0, 80.0)
    # No unbounded values: the ordinary percentile bounds.
    assert bounds(finite, 0.95) == pytest.approx((float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))))
    # 1 of 80 unbounded: h = 79 * 0.975 = 77.025 reads ranks 77 and 78, both finite.
    eighty = np.append(finite, math.inf)
    proxy = np.append(finite, finite.max())
    assert bounds(eighty, 0.95) == pytest.approx((float(np.percentile(proxy, 2.5)), float(np.percentile(proxy, 97.5))))
    # 1 of 40 unbounded: the share 0.025 is not above 0.025, but h = 38.025
    # reads rank 39, the unbounded one, so the upper bound is not finite.
    assert bounds(np.append(np.arange(1.0, 40.0), math.inf), 0.95) is None
    # 3 of 40 unbounded: the share 0.075 is above 0.025.
    assert bounds(np.append(np.arange(1.0, 38.0), [math.inf] * 3), 0.95) is None


# ---------------------------------------------------------------------------
# Simulated power
# ---------------------------------------------------------------------------


def test_simulated_power_increases_with_effect_size() -> None:
    effects = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8]
    powers = [simulated_power(effect, 1.0, 20, alpha=0.05, n_sim=2000, seed=3) for effect in effects]
    assert powers == sorted(powers)
    assert powers[-1] - powers[0] > 0.5


@pytest.mark.parametrize("alpha", [0.05, 0.1])
def test_simulated_power_at_effect_zero_is_about_alpha(alpha) -> None:
    assert simulated_power(0.0, 1.0, 20, alpha=alpha, n_sim=8000, seed=11) == pytest.approx(alpha, abs=0.015)


def test_simulated_power_is_near_one_at_a_large_effect() -> None:
    assert simulated_power(2.0, 1.0, 30, alpha=0.05, n_sim=2000, seed=12) >= 0.99


def test_simulated_power_is_near_zero_at_a_large_negative_effect() -> None:
    assert simulated_power(-2.0, 1.0, 30, alpha=0.05, n_sim=2000, seed=12) <= 0.01


def test_simulated_power_matches_the_noncentral_t_value() -> None:
    # One-sided one-sample t test, d = 0.5, n = 20, alpha = 0.05: power is about 0.695.
    assert simulated_power(0.5, 1.0, 20, alpha=0.05, n_sim=8000, seed=13) == pytest.approx(0.695, abs=0.03)


def test_simulated_power_depends_on_effect_over_noise_only() -> None:
    small = simulated_power(0.25, 0.5, 15, alpha=0.05, n_sim=3000, seed=14)
    large = simulated_power(1.0, 2.0, 15, alpha=0.05, n_sim=3000, seed=14)
    assert small == pytest.approx(large, abs=2 / 3000)


def test_simulated_power_is_deterministic_for_a_seed() -> None:
    first = simulated_power(0.3, 1.0, 25, alpha=0.05, n_sim=500, seed=15)
    assert simulated_power(0.3, 1.0, 25, alpha=0.05, n_sim=500, seed=15) == first
    assert isinstance(first, float)
    assert 0.0 <= first <= 1.0
    assert simulated_power(0.3, 1.0, 25, alpha=0.05, n_sim=500, seed=16) != first


@pytest.mark.parametrize(
    ("effect", "noise_sd", "n_items", "kwargs"),
    [
        (math.nan, 1.0, 20, {}),
        (0.5, 0.0, 20, {}),
        (0.5, -1.0, 20, {}),
        (0.5, math.inf, 20, {}),
        (0.5, 1.0, 1, {}),
        (0.5, 1.0, 20.0, {}),
        (0.5, 1.0, True, {}),
        (0.5, 1.0, 20, {"alpha": 0.0}),
        (0.5, 1.0, 20, {"alpha": 1.0}),
        (0.5, 1.0, 20, {"n_sim": 0}),
        (0.5, 1.0, 20, {"seed": 2.5}),
    ],
)
def test_simulated_power_validates_its_arguments(effect, noise_sd, n_items, kwargs) -> None:
    arguments = {"alpha": 0.05, "n_sim": 100, "seed": 0} | kwargs
    with pytest.raises(ValueError):
        simulated_power(effect, noise_sd, n_items, **arguments)
