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


def test_wilson_pins_an_interior_bound() -> None:
    """Both published pins are at k == n, where the variance term vanishes.

    With only those, a mutant using p(1-p)/(n-1) instead of p(1-p)/n passes the
    entire suite while being wrong by up to 0.064. An interior cell is the only
    thing that exercises that term at all.
    """
    lo, hi = wilson_interval(13, 17)
    assert lo == pytest.approx(0.5273820188043501, abs=1e-12)
    assert hi == pytest.approx(0.9044495567791988, abs=1e-12)


def test_continuity_correction_is_wider_than_the_plain_interval() -> None:
    lo, hi = wilson_interval(126, 128)
    lo_cc, hi_cc = wilson_interval(126, 128, continuity=True)
    assert lo_cc < lo and hi_cc > hi


def test_continuity_correction_matches_newcombe() -> None:
    """Pinned against the published closed form.

    "Wider than plain" is too weak an assertion to catch this: applying the
    correction as a constant shift of the already-inverted bound is also wider,
    but is anti-conservative against the exact interval over most of the
    parameter space -- which is the direction that matters for a gate.
    """
    assert wilson_interval(24, 24, continuity=True)[0] == pytest.approx(
        0.8282849931831465, abs=1e-12
    )
    assert wilson_interval(128, 128, continuity=True)[0] == pytest.approx(
        0.963686124793, abs=1e-9
    )
    assert wilson_interval(2, 2, continuity=True)[0] == pytest.approx(
        0.197867455762, abs=1e-9
    )
    assert wilson_interval(1, 24, continuity=True)[0] == pytest.approx(
        0.002178855996, abs=1e-9
    )
    assert wilson_interval(0, 30, continuity=True)[0] == 0.0
    assert wilson_interval(30, 30, continuity=True)[1] == 1.0


def test_continuity_correction_is_never_narrower_than_plain() -> None:
    for n in (2, 5, 10, 24, 64, 128):
        for k in range(n + 1):
            lo, hi = wilson_interval(k, n)
            lo_cc, hi_cc = wilson_interval(k, n, continuity=True)
            assert lo_cc <= lo + 1e-12 and hi_cc >= hi - 1e-12, (k, n)


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


# --- input hardening (from adversarial verification) -------------------------


def test_nan_grid_is_rejected_rather_than_silently_poisoning() -> None:
    """`grid <= 0` lets NaN through, because every NaN comparison is false."""
    for bad in (float("nan"), float("inf"), 0.0, -0.1):
        with pytest.raises(ValueError):
            snap_to_grid(0.5, bad)
        with pytest.raises(ValueError):
            grid_adherence([0.1, 0.2], bad)


def test_non_finite_value_raises_rather_than_leaking_a_decimal_error() -> None:
    """A documented ValueError contract must not surface decimal.InvalidOperation."""
    for bad in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError):
            snap_to_grid(bad, 0.1)


def test_adherence_survives_the_bad_data_it_exists_to_describe() -> None:
    """A reporting function must not crash on a non-finite draw."""
    assert grid_adherence([0.1, 0.2, float("nan")], 0.1) == pytest.approx(2 / 3)
    assert grid_adherence([0.15, float("inf")], 0.1) == 0.0


def test_adherence_validates_its_tolerance() -> None:
    for bad in (-1.0, float("nan")):
        with pytest.raises(ValueError):
            grid_adherence([0.1], 0.1, tolerance=bad)


def test_numpy_scalars_and_arrays_are_accepted() -> None:
    """repr(np.float64(0.15)) is 'np.float64(0.15)' under numpy 2."""
    np = pytest.importorskip("numpy")
    assert snap_to_grid(np.float64(0.15), 0.1) == pytest.approx(0.2)
    assert snap_to_grid(0.15, np.float64(0.1)) == pytest.approx(0.2)
    assert grid_adherence(np.asarray([0.1, 0.2, 0.25]), 0.1) == pytest.approx(2 / 3)
    assert robust_location(np.asarray([0.1, 0.3, 0.2]), mode="median") == pytest.approx(0.2)
    assert scale_floor(np.asarray([0.8, 0.8, 0.8]), grid=0.1) > 0.0
    verdict = detect_multimodal(
        np.asarray([0.1, 0.2] * 5 + [0.9, 1.0] * 5), grid=0.1, mass_min=0.4
    )
    assert verdict is not None and verdict.separated


def test_estimators_reject_non_finite_rather_than_reordering() -> None:
    """NaN makes `sorted` order-dependent, which would break the replay contract.

    Estimators reject; reporting functions like grid_adherence drop. The rule is
    consistent within the module, which it briefly was not.
    """
    for bad in (float("nan"), float("inf"), -float("inf")):
        for mode in ("mean", "median", "trimmed"):
            with pytest.raises(ValueError, match="finite"):
                robust_location([0.1, bad, 0.3], mode=mode)
        with pytest.raises(ValueError, match="finite"):
            scale_floor([0.1, bad, 0.3])


def test_signed_zero_does_not_leak_into_the_result() -> None:
    """-0.0 and 0.0 compare equal but serialise differently.

    A stable sort would emit whichever arrived first, and that last-bit
    difference changes a persisted artifact hash -- the exact failure the module
    exists to prevent. The measured policy axis contains six exact zeros.
    """
    import struct

    a = robust_location([-0.1, -0.0, 0.0, 0.1], mode="median")
    b = robust_location([-0.1, 0.0, -0.0, 0.1], mode="median")
    assert struct.pack("<d", a) == struct.pack("<d", b)
    assert math.copysign(1.0, robust_location([-0.0, -0.0], mode="mean")) == 1.0


def test_scale_floor_binds_below_a_fifth_of_the_lattice_step() -> None:
    """Not only at exactly zero -- the docstring used to claim otherwise."""
    concentrated = [0.80, 0.80, 0.80, 0.81, 0.81, 0.79, 0.79]
    unfloored = scale_floor(concentrated, grid=None)
    floored = scale_floor(concentrated, grid=0.1)
    assert unfloored > 0.0, "MAD is well defined here, not zero"
    assert floored > unfloored


def test_too_few_draws_cannot_certify_two_clusters() -> None:
    """Two draws once "certified" a separated split, which under RAISE threw."""
    assert detect_multimodal([0.0, 0.5], grid=0.1) is None
    assert detect_multimodal([0.0] * 3 + [0.9] * 3, grid=0.1) is None
    assert detect_multimodal([0.0] * 4 + [0.9] * 4, grid=0.1).separated is True


def test_a_lattice_too_fine_for_the_sample_does_not_flag() -> None:
    """Empty runs occur everywhere by chance when the lattice outruns sampling.

    The density test cannot save it: an empty gap has no peak to compare
    against, so it passes vacuously. Measured before the guard, a unimodal
    normal at a 0.001 lattice flagged on every single subsample.
    """
    rng = random.Random(11)
    unimodal = [min(1.0, max(0.0, rng.gauss(0.5, 0.1))) for _ in range(24)]
    assert detect_multimodal(unimodal, grid=0.1) is not None
    for fine in (0.001, 0.0005):
        verdict = detect_multimodal(unimodal, grid=fine)
        assert verdict is None or not verdict.separated, fine

    # The project's own continuous scores are exactly this case.
    guard = load_guard_scores()
    fine_verdict = detect_multimodal(guard, grid=0.001)
    assert fine_verdict is None or not fine_verdict.separated


def test_real_split_still_flags_after_the_sparsity_guard() -> None:
    """The guard must not cost the detection it exists alongside."""
    verdict = detect_multimodal(load_axis("policy"), grid=0.1)
    assert verdict is not None and verdict.separated
    assert verdict.gap == (-0.2, 0.2)


def test_selected_split_is_the_sparsest_gap() -> None:
    """Pins the selection rule, not merely that some split was admissible."""
    verdict = detect_multimodal(load_axis("policy"), grid=0.1)
    assert verdict.trough_mass == pytest.approx(12 / 977)
    assert verdict.lower_mass == pytest.approx(352 / 977)
    assert verdict.upper_mass == pytest.approx(613 / 977)


def test_signed_zero_does_not_move_the_reported_gap() -> None:
    """`-0.0` and `0.0` bin together, but a dict keeps the first key seen."""
    a = [-0.04, 0.04] + [-1.0] * 5 + [1.0] * 7
    b = [0.04, -0.04] + [-1.0] * 5 + [1.0] * 7
    va, vb = detect_multimodal(a, grid=0.1), detect_multimodal(b, grid=0.1)
    assert repr(va) == repr(vb)


def test_detection_stays_quadratic_on_a_fine_lattice() -> None:
    """A cubic search runs to seconds per component on a bounded score."""
    import time

    rng = random.Random(5)
    values = [rng.random() for _ in range(1000)]
    started = time.monotonic()
    detect_multimodal(values, grid=0.001)
    assert time.monotonic() - started < 1.0
