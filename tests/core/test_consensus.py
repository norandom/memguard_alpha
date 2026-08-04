"""Pure consensus statistics, tested against the measured pathologies.

The corpus these tests load is the point: every assertion here is a case the
source design proposal's machinery got wrong, so a passing suite means the
implementation survives contact with real data rather than with a textbook.
"""

from __future__ import annotations

import math
import random
import statistics

import pytest

from recall_guard.core.consensus import (
    Tail,
    detect_multimodal,
    grid_adherence,
    robust_location,
    scale_floor,
    smallest_certifiable_n,
    snap_to_grid,
    wilson_interval,
)
from tests.fixtures.dispersion import load_axis, load_guard_scores

# --- 2.2 agreement measurement and intervals ---------------------------------


def test_wilson_is_not_zero_width_at_unanimity() -> None:
    """The case that disqualifies the naive normal approximation.

    At the measured agreement level unanimity is the typical outcome, and the
    Wald interval collapses to [1.0, 1.0] there -- reporting certainty from two
    dozen draws and firing any stopping rule instantly and always.
    """
    for n in (1, 24, 64, 128, 977):
        lo, hi = wilson_interval(n, n)
        assert 0.0 <= lo < hi <= 1.0, (n, lo, hi)
        assert hi == pytest.approx(1.0)
        assert lo < 1.0, f"n={n} produced a zero-width interval at unanimity"


def test_wilson_bounds_stay_inside_the_unit_interval() -> None:
    for k, n in [(0, 1), (0, 30), (1, 2), (13, 17), (126, 128), (963, 977)]:
        lo, hi = wilson_interval(k, n)
        assert 0.0 <= lo <= hi <= 1.0, (k, n, lo, hi)


def test_wilson_matches_published_values() -> None:
    """Pinned against the closed form, two-sided at 95%."""
    assert wilson_interval(24, 24)[0] == pytest.approx(0.8620, abs=5e-4)
    assert wilson_interval(128, 128)[0] == pytest.approx(0.9709, abs=5e-4)


def test_continuity_correction_is_wider_than_the_plain_interval() -> None:
    lo, hi = wilson_interval(126, 128)
    lo_cc, hi_cc = wilson_interval(126, 128, continuity=True)
    assert lo_cc < lo and hi_cc > hi


def test_wilson_rejects_impossible_counts() -> None:
    with pytest.raises(ValueError):
        wilson_interval(5, 4)
    with pytest.raises(ValueError):
        wilson_interval(-1, 4)


def test_zero_draws_yields_no_interval() -> None:
    assert wilson_interval(0, 0) is None


def test_certification_feasibility_floor() -> None:
    """The proposal's stated minimum cannot certify its target under either tail.

    This is what makes an unreachable configuration detectable at construction
    instead of after a full draw budget has been spent.
    """
    assert smallest_certifiable_n(0.95, tail=Tail.TWO_SIDED) == 73
    assert smallest_certifiable_n(0.90, tail=Tail.TWO_SIDED) == 35
    assert smallest_certifiable_n(0.95, tail=Tail.ONE_SIDED) == 52
    for tail in Tail:
        assert smallest_certifiable_n(0.95, tail=tail) > 24


def test_certification_floor_is_consistent_with_the_interval() -> None:
    for target in (0.80, 0.90, 0.95, 0.99):
        n = smallest_certifiable_n(target, tail=Tail.TWO_SIDED)
        assert wilson_interval(n, n)[0] >= target
        assert wilson_interval(n - 1, n - 1)[0] < target


# --- 2.3 lattice handling and adherence --------------------------------------


def test_snap_is_half_away_from_zero_on_every_tie() -> None:
    """All 20 exact half-steps in [-1, 1] at g=0.1.

    Dividing by the lattice step makes the tie direction value-dependent rather
    than rule-dependent: 0.85/0.1 is exactly 8.5 while 0.95/0.1 is
    9.499999999999998, so the obvious spellings disagree with each other and
    with themselves.
    """
    for units in range(-9, 10):
        if units == 0:
            continue
        value = units / 10 + math.copysign(0.05, units)
        snapped = snap_to_grid(value, 0.1)
        expected = (abs(units) / 10 + 0.1) * math.copysign(1, units)
        assert snapped == pytest.approx(expected, abs=1e-12), (value, snapped, expected)


def test_snap_is_symmetric_about_zero() -> None:
    for v in (0.85, 0.35, 0.15, 0.95, 0.65, 0.05):
        assert snap_to_grid(-v, 0.1) == pytest.approx(-snap_to_grid(v, 0.1))


def test_snap_rejects_a_non_positive_lattice() -> None:
    for bad in (0.0, -0.1):
        with pytest.raises(ValueError):
            snap_to_grid(0.5, bad)


def test_grid_adherence_detects_a_wrongly_declared_lattice() -> None:
    """A caller who declares the wrong lattice must find out."""
    inflation = load_axis("inflation")
    assert grid_adherence(inflation, 0.1) < 0.60
    assert grid_adherence(inflation, 0.01) == pytest.approx(1.0)
    assert grid_adherence(load_guard_scores(), 0.1) < 0.10


def test_scale_floor_rescues_the_undefined_estimate() -> None:
    """The pinned component's MAD is exactly zero; the floor keeps it usable."""
    inflation = load_axis("inflation")
    assert scale_floor(inflation, grid=0.1) > 0.0
    # ...but the floor is inert wherever the estimate was already well-defined.
    policy = load_axis("policy")
    assert scale_floor(policy, grid=0.1) == pytest.approx(
        1.4826 * statistics.median([abs(v - statistics.median(policy)) for v in policy])
    )


# --- 2.4 robust location -----------------------------------------------------


def test_location_is_independent_of_input_order() -> None:
    """Bit-identical under shuffling -- the replay contract depends on it.

    Pairwise summation makes a plain mean order-dependent in the last bit, and a
    last-bit difference changes a persisted artifact hash.
    """
    values = load_axis("policy")
    rng = random.Random(0)
    for mode in ("mean", "median", "trimmed"):
        baseline = robust_location(values, mode=mode)
        for _ in range(8):
            shuffled = values[:]
            rng.shuffle(shuffled)
            assert robust_location(shuffled, mode=mode) == baseline, mode


def test_median_at_even_count_uses_the_lower_order_statistic() -> None:
    """Never the midpoint of two draws -- that is a value nobody emitted."""
    assert robust_location([0.7, 0.8], mode="median") == 0.7
    assert robust_location([0.8, 0.7], mode="median") == 0.7
    assert robust_location([0.1, 0.2, 0.3, 0.4], mode="median") == 0.2


def test_trimmed_mean_does_not_escape_the_measured_trough() -> None:
    """Why the multimodality gate must run first.

    Symmetric trimming converges toward the median, not toward a mode. On the
    genuinely disagreeing component every fraction lands inside the gap, so the
    location estimator is the wrong instrument there at any tuning.
    """
    policy = load_axis("policy")
    for fraction in (0.0, 0.05, 0.10, 0.20, 0.25):
        located = robust_location(policy, mode="trimmed", trim=fraction)
        assert -0.4 <= located <= 0.3, (fraction, located)


def test_trim_fraction_is_validated() -> None:
    for bad in (-0.01, 0.5, 0.9):
        with pytest.raises(ValueError):
            robust_location([1.0, 2.0, 3.0], mode="trimmed", trim=bad)


def test_location_rejects_an_empty_sample() -> None:
    with pytest.raises(ValueError):
        robust_location([], mode="median")


def test_mean_is_above_median_on_the_guard_scores() -> None:
    """The estimator choice that decides how much exposure is withheld.

    Attenuation is linear in the score, so the mean is unbiased for expected
    attenuation; the median of this right-skewed sample withholds materially
    less, in the risk-increasing direction.
    """
    scores = load_guard_scores()
    mean = robust_location(scores, mode="mean")
    median = robust_location(scores, mode="median")
    assert mean > median
    assert (1 - median) - (1 - mean) > 0.05


# --- 2.5 separated-cluster detection -----------------------------------------


def test_detects_the_measured_separated_clusters() -> None:
    """The component that motivated the whole gate."""
    verdict = detect_multimodal(load_axis("policy"), grid=0.1)
    assert verdict is not None
    assert verdict.separated is True
    assert verdict.lower_mass > 0.25 and verdict.upper_mass > 0.25
    assert verdict.trough_mass < 0.06


def test_does_not_flag_converged_components() -> None:
    """False positives here would suppress a location estimate that is fine.

    ``growth`` is the near-miss: its minority-sign mass is around 12%, close
    enough to a lower threshold that sampling noise crosses it.
    """
    for axis in ("inflation", "growth", "credit_stress", "risk_appetite"):
        verdict = detect_multimodal(load_axis(axis), grid=0.1)
        assert verdict is not None
        assert verdict.separated is False, axis


def test_detection_holds_at_realistic_draw_counts() -> None:
    values = load_axis("policy")
    rng = random.Random(7)
    for n in (64, 128):
        hits = 0
        for _ in range(40):
            verdict = detect_multimodal(rng.sample(values, n), grid=0.1)
            hits += verdict.separated
        assert hits >= 34, f"n={n} detected only {hits}/40"


def test_thresholds_are_caller_configurable() -> None:
    """Every constant is tuned on one date, so none may be a literal."""
    growth = load_axis("growth")
    assert detect_multimodal(growth, grid=0.1).separated is False
    relaxed = detect_multimodal(growth, grid=0.1, mass_min=0.10, density_ratio=2.0)
    assert relaxed.separated is True


def test_no_declared_lattice_reports_the_check_did_not_run() -> None:
    """Not a silent 'no clusters found' -- the guard scores are continuous."""
    assert detect_multimodal(load_guard_scores(), grid=None) is None


def test_split_selection_is_order_independent() -> None:
    values = load_axis("policy")
    rng = random.Random(3)
    baseline = detect_multimodal(values, grid=0.1)
    for _ in range(8):
        shuffled = values[:]
        rng.shuffle(shuffled)
        assert detect_multimodal(shuffled, grid=0.1) == baseline


def test_detection_validates_its_inputs() -> None:
    with pytest.raises(ValueError):
        detect_multimodal([], grid=0.1)
    with pytest.raises(ValueError):
        detect_multimodal([0.1, 0.2], grid=0.0)
