"""Unit tests for ``src.portfolio.cmmd.apply_cmmd_filter``.

Covers Requirements 6.1, 6.2, and 6.4 of the cmmd-backtest spec:

- 6.1: 80th-percentile cut on ``p_memorized`` over parse-OK rows.
- 6.2: empirical threshold returned for manifest provenance.
- 6.4: filter never reorders surviving rows and never mutates them.

The tests use ``types.SimpleNamespace`` as a structural stand-in for
``harness.evaluator.Record``. Importing the real ``Record`` here would
violate the ``portfolio`` ↔ ``harness`` layer rule (portfolio is order=1,
harness is order=0; lower order cannot pull in higher order).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.portfolio.cmmd import apply_cmmd_filter


def _mk(p_memorized: float | None, parse_ok: bool = True, tag: str | None = None) -> SimpleNamespace:
    """Build a Record-shaped object with just the attributes the filter reads."""
    return SimpleNamespace(
        parse_ok=parse_ok,
        p_memorized=p_memorized,
        predicted_direction=1,
        tag=tag,  # opaque marker the filter must preserve verbatim
    )


def test_uniform_100_records_quantile_080_keeps_80_survivors() -> None:
    """100 evenly spaced p_memorized values drop the top 20% at quantile=0.80.

    With ``np.linspace(0, 1, 100)`` the empirical 80th percentile is ~0.80
    and the inclusive ``<= threshold`` rule keeps exactly 80 rows.
    """
    p_values = np.linspace(0.0, 1.0, 100)
    records = [_mk(float(p)) for p in p_values]

    kept, threshold = apply_cmmd_filter(records, quantile=0.80)

    assert len(kept) == 80
    # Cross-check threshold against numpy's own quantile (linear interp).
    assert threshold == pytest.approx(float(np.quantile(p_values, 0.80)), abs=1e-12)


def test_threshold_matches_80th_percentile_exactly() -> None:
    """Threshold returned must equal ``np.quantile(values, 0.80)`` to 1e-9."""
    rng = np.random.default_rng(42)
    p_values = rng.uniform(0.0, 1.0, size=500)
    records = [_mk(float(p)) for p in p_values]

    _, threshold = apply_cmmd_filter(records, quantile=0.80)

    expected = float(np.quantile(p_values, 0.80))
    assert threshold == pytest.approx(expected, abs=1e-9)


def test_parse_ok_false_rows_dropped_before_percentile() -> None:
    """``parse_ok=False`` rows are filtered first, so the percentile is
    computed on the surviving distribution only.

    The interloper rows carry ``p_memorized=0.99``; if they leaked into
    the percentile computation the threshold would be pulled upward and
    survivor count would shift. Asserting both no-leakage AND that the
    threshold matches the percentile of the surviving distribution proves
    the ordering of the two filtering steps.
    """
    surviving_p = np.linspace(0.0, 1.0, 100).tolist()
    good_records = [_mk(p, parse_ok=True, tag=f"good-{i}") for i, p in enumerate(surviving_p)]
    bad_records = [_mk(0.99, parse_ok=False, tag=f"bad-{i}") for i in range(50)]

    interleaved: list[SimpleNamespace] = []
    for i, good in enumerate(good_records):
        interleaved.append(good)
        if i % 2 == 0 and i // 2 < len(bad_records):
            interleaved.append(bad_records[i // 2])

    kept, threshold = apply_cmmd_filter(interleaved, quantile=0.80)

    # No parse_ok=False row may survive.
    assert all(r.parse_ok for r in kept)
    # Threshold must equal the percentile of the *surviving* distribution.
    expected_threshold = float(np.quantile(np.array(surviving_p), 0.80))
    assert threshold == pytest.approx(expected_threshold, abs=1e-12)
    # And exactly 80 of the 100 good rows survive.
    assert len(kept) == 80


def test_input_order_preserved_in_output() -> None:
    """Surviving rows appear in the original input order, not sorted by p."""
    # Deliberately non-monotonic p_memorized stream.
    raw = [0.55, 0.10, 0.92, 0.30, 0.71, 0.05, 0.40, 0.99, 0.20, 0.60]
    records = [_mk(p, tag=f"pos-{i}") for i, p in enumerate(raw)]

    kept, threshold = apply_cmmd_filter(records, quantile=0.80)

    # Threshold = 80th percentile of the full surviving (parse_ok) set.
    expected = float(np.quantile(np.array(raw), 0.80))
    assert threshold == pytest.approx(expected, abs=1e-12)

    # All survivors must have p_memorized <= threshold AND appear in the
    # same relative order their tag indices were inserted.
    kept_tags = [r.tag for r in kept]
    expected_tags = [f"pos-{i}" for i, p in enumerate(raw) if p <= threshold]
    assert kept_tags == expected_tags


def test_p_memorized_none_rows_dropped_alongside_parse_failures() -> None:
    """``p_memorized=None`` rows are dropped before percentile computation,
    same as ``parse_ok=False`` rows."""
    good = [_mk(p, tag=f"g-{i}") for i, p in enumerate(np.linspace(0.0, 1.0, 100))]
    none_rows = [_mk(None, parse_ok=True, tag=f"none-{i}") for i in range(20)]
    bad_rows = [_mk(0.5, parse_ok=False, tag=f"bad-{i}") for i in range(20)]

    kept, threshold = apply_cmmd_filter(good + none_rows + bad_rows, quantile=0.80)

    assert all(r.p_memorized is not None for r in kept)
    assert all(r.parse_ok for r in kept)
    assert len(kept) == 80
    # Threshold computed only on the 100 good rows.
    expected = float(np.quantile(np.linspace(0.0, 1.0, 100), 0.80))
    assert threshold == pytest.approx(expected, abs=1e-12)


def test_invalid_quantile_raises() -> None:
    """Quantiles outside the open interval (0, 1) are rejected."""
    records = [_mk(0.5)]
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            apply_cmmd_filter(records, quantile=bad)


def test_predicted_direction_never_modified() -> None:
    """The filter is read-only on surviving rows (Req 6.4)."""
    records = [_mk(p) for p in (0.1, 0.4, 0.95)]
    for r in records:
        r.predicted_direction = 1

    kept, _ = apply_cmmd_filter(records, quantile=0.80)
    for r in kept:
        assert r.predicted_direction == 1
