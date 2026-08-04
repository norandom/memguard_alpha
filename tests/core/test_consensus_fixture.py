"""The measured-pathology corpus must load from this repo alone.

Every interesting property of the consensus reduction is a property of measured
data: a component whose median absolute deviation is exactly zero, a component
that splits into two separated clusters, and a contamination score that is
continuous rather than lattice-valued. Those cases were measured in a sibling
project, which this package cannot import -- so the numeric columns are
vendored here and these assertions pin that the pathologies survived the copy.

If one of these fails, the fixture is wrong, not the statistics.
"""

from __future__ import annotations

import statistics

from tests.fixtures.dispersion import (
    AXES,
    load_dispersion_draws,
    load_guard_scores,
)


def test_draws_fixture_loads_without_sibling_project() -> None:
    draws = load_dispersion_draws()
    assert len(draws) == 977, "expected the parsed subset of the 1000-draw study"
    assert set(draws[0]) == set(AXES)
    assert all(len(row) == len(AXES) for row in draws)


def test_pinned_axis_has_zero_median_absolute_deviation() -> None:
    """The case that makes the textbook robust scale estimate undefined."""
    values = [row["inflation"] for row in load_dispersion_draws()]
    median = statistics.median(values)
    mad = statistics.median([abs(v - median) for v in values])
    assert mad == 0.0, f"inflation MAD is {mad}, expected exactly 0.0"


def test_disagreeing_axis_is_split_into_two_separated_clusters() -> None:
    """The case no symmetric trim fraction can reduce to one location."""
    values = [row["policy"] for row in load_dispersion_draws()]
    n = len(values)
    below = sum(1 for v in values if v < 0) / n
    above = sum(1 for v in values if v > 0) / n
    trough = sum(1 for v in values if -0.4 <= v <= 0.2) / n

    assert 0.34 < below < 0.38, below
    assert 0.61 < above < 0.65, above
    assert trough < 0.06, f"trough holds {trough:.3f}; the split is not separated"
    # The arithmetic mean lands in the trough -- the reason averaging is wrong here.
    assert -0.4 <= statistics.fmean(values) <= 0.2


def test_emitted_values_do_not_all_lie_on_the_tenth_lattice() -> None:
    """The declared-lattice premise is false; the reduction must not assume it."""
    values = [v for row in load_dispersion_draws() for v in row.values()]
    off = sum(1 for v in values if abs(v * 10 - round(v * 10)) > 1e-9)
    assert off / len(values) > 0.10, "expected a materially off-lattice population"


def test_guard_scores_are_continuous() -> None:
    """The contamination score has no lattice, so lattice paths must be optional."""
    scores = load_guard_scores()
    assert len(scores) == 100
    assert len(set(scores)) > 80, "expected near-distinct values, not lattice-valued"
    on_grid = sum(1 for s in scores if abs(s * 10 - round(s * 10)) < 1e-9)
    assert on_grid / len(scores) < 0.10
    assert statistics.fmean(scores) > statistics.median(scores), "expected right skew"
