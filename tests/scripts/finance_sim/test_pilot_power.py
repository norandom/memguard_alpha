"""Two-sample simulated power for the pilot and power gate (task 7.8; Requirement 8.9).

``simulated_power_two_sample`` lives in ``recall_guard/core/significance.py``
(the design's "Pilot and power gate (task 7.8)" paragraph adds it there) and
follows the conventions of ``simulated_power``: the critical value is simulated
under effect 0, the statistic is the difference of the two sample means over its
pooled standard error, and the power is the share of simulations under the effect
that exceed it. Reference values come from the noncentral t distribution
(scipy.stats.nct, computed once offline; scipy is not a runtime dependency).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from recall_guard.core.significance import simulated_power, simulated_power_two_sample


def test_power_increases_with_the_effect_for_one_seed() -> None:
    effects = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2]
    powers = [simulated_power_two_sample(effect, 1.0, 20, 20, alpha=0.05, n_sim=2000, seed=3) for effect in effects]
    assert powers == sorted(powers)
    assert powers[-1] - powers[0] > 0.8


@pytest.mark.parametrize("alpha", [0.05, 0.1])
def test_power_at_effect_zero_is_about_alpha(alpha) -> None:
    assert simulated_power_two_sample(0.0, 1.0, 68, 22, alpha=alpha, n_sim=8000, seed=11) == pytest.approx(
        alpha, abs=0.015)


def test_power_matches_the_noncentral_t_value_for_equal_groups() -> None:
    # One-sided pooled two-sample t test, d = 0.8, 20 and 20, alpha 0.05: nct power 0.7994.
    assert simulated_power_two_sample(0.8, 1.0, 20, 20, alpha=0.05, n_sim=8000, seed=13) == pytest.approx(
        0.7994, abs=0.03)


def test_power_matches_the_noncentral_t_value_for_the_frozen_month_counts() -> None:
    # 68 pre-cutoff and 22 post-cutoff months, d = 0.5: nct power 0.6473; d = 0.2: 0.2017.
    assert simulated_power_two_sample(0.5, 1.0, 68, 22, alpha=0.05, n_sim=8000, seed=17) == pytest.approx(
        0.6473, abs=0.03)
    assert simulated_power_two_sample(0.2, 1.0, 68, 22, alpha=0.05, n_sim=8000, seed=17) == pytest.approx(
        0.2017, abs=0.03)


def test_power_depends_on_effect_over_noise_only() -> None:
    small = simulated_power_two_sample(0.25, 0.5, 15, 9, alpha=0.05, n_sim=3000, seed=14)
    large = simulated_power_two_sample(1.0, 2.0, 15, 9, alpha=0.05, n_sim=3000, seed=14)
    assert small == pytest.approx(large, abs=2 / 3000)


def test_power_is_near_one_at_a_large_effect_and_near_zero_at_a_large_negative_one() -> None:
    assert simulated_power_two_sample(2.0, 1.0, 30, 30, alpha=0.05, n_sim=2000, seed=12) >= 0.99
    assert simulated_power_two_sample(-2.0, 1.0, 30, 30, alpha=0.05, n_sim=2000, seed=12) <= 0.01


def test_power_is_deterministic_for_a_seed() -> None:
    first = simulated_power_two_sample(0.3, 1.0, 25, 12, alpha=0.05, n_sim=500, seed=15)
    assert simulated_power_two_sample(0.3, 1.0, 25, 12, alpha=0.05, n_sim=500, seed=15) == first
    assert isinstance(first, float) and 0.0 <= first <= 1.0
    assert simulated_power_two_sample(0.3, 1.0, 25, 12, alpha=0.05, n_sim=500, seed=16) != first


def test_the_statistic_is_the_mean_difference_over_its_pooled_standard_error() -> None:
    # Rebuild the simulation by hand with the documented draw order and compare.
    effect, sd, n_a, n_b, alpha, n_sim, seed = 0.4, 0.7, 9, 5, 0.05, 400, 21
    rng = np.random.default_rng(seed)

    def statistic(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        pooled = ((n_a - 1) * a.var(axis=1, ddof=1) + (n_b - 1) * b.var(axis=1, ddof=1)) / (n_a + n_b - 2)
        return (a.mean(axis=1) - b.mean(axis=1)) / np.sqrt(pooled * (1 / n_a + 1 / n_b))

    null = statistic(sd * rng.standard_normal((n_sim, n_a)), sd * rng.standard_normal((n_sim, n_b)))
    critical = float(np.quantile(null, 1 - alpha))
    alternative = statistic(effect + sd * rng.standard_normal((n_sim, n_a)), sd * rng.standard_normal((n_sim, n_b)))
    expected = float(np.count_nonzero(alternative > critical) / n_sim)
    assert simulated_power_two_sample(effect, sd, n_a, n_b, alpha=alpha, n_sim=n_sim, seed=seed) == expected


def test_the_one_sample_helper_is_unchanged() -> None:
    assert simulated_power(0.5, 1.0, 20, alpha=0.05, n_sim=8000, seed=13) == pytest.approx(0.695, abs=0.03)


@pytest.mark.parametrize(
    ("effect", "noise_sd", "n_a", "n_b", "kwargs"),
    [
        (math.nan, 1.0, 20, 20, {}),
        (math.inf, 1.0, 20, 20, {}),
        (0.5, 0.0, 20, 20, {}),
        (0.5, -1.0, 20, 20, {}),
        (0.5, math.nan, 20, 20, {}),
        (0.5, 1.0, 1, 20, {}),
        (0.5, 1.0, 20, 1, {}),
        (0.5, 1.0, 20.0, 20, {}),
        (0.5, 1.0, 20, True, {}),
        (0.5, 1.0, 20, 20, {"alpha": 0.0}),
        (0.5, 1.0, 20, 20, {"alpha": 1.0}),
        (0.5, 1.0, 20, 20, {"n_sim": 0}),
        (0.5, 1.0, 20, 20, {"n_sim": 10.0}),
        (0.5, 1.0, 20, 20, {"seed": 2.5}),
        (True, 1.0, 20, 20, {}),
    ],
)
def test_arguments_are_validated(effect, noise_sd, n_a, n_b, kwargs) -> None:
    arguments = {"alpha": 0.05, "n_sim": 100, "seed": 0} | kwargs
    with pytest.raises(ValueError):
        simulated_power_two_sample(effect, noise_sd, n_a, n_b, **arguments)
