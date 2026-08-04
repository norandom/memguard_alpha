"""Loader for the vendored measured-dispersion corpus.

See ``dispersion_PROVENANCE.md`` in this directory for where the data came from
and what it does and does not license you to conclude.
"""

from __future__ import annotations

import csv
from pathlib import Path

#: Component names, in the order the study emitted them.
AXES: tuple[str, ...] = (
    "inflation",
    "growth",
    "credit_stress",
    "policy",
    "risk_appetite",
)

_HERE = Path(__file__).parent


def load_dispersion_draws() -> list[dict[str, float]]:
    """Return the 977 parsed component vectors, in recorded draw order.

    Recorded order is *not* meaningful for a reduction -- the whole point of the
    canonical-ordering rule is that arrival order must not influence a result.
    It is preserved here only so a test can deliberately shuffle it and assert
    the reduction is unchanged.
    """
    with (_HERE / "dispersion_draws.csv").open(newline="") as fh:
        return [{axis: float(row[axis]) for axis in AXES} for row in csv.DictReader(fh)]


def load_guard_scores() -> list[float]:
    """Return the 100 measured contamination scores for one identical prompt."""
    with (_HERE / "dispersion_guard.csv").open(newline="") as fh:
        return [float(row["p_memorized"]) for row in csv.DictReader(fh)]


def load_axis(axis: str) -> list[float]:
    """Return one component's values across all parsed draws."""
    if axis not in AXES:
        raise ValueError(f"unknown axis {axis!r}; expected one of {AXES}")
    return [row[axis] for row in load_dispersion_draws()]
