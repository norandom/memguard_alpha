"""Top-quintile ``p_memorized`` filter for the cmmd-backtest pipeline.

Covers Reqs 6.1, 6.2, and 6.4:

- 6.1: drop the top ``(1 - quantile)`` slice of the parse-OK
  ``p_memorized`` distribution (default = top 20%).
- 6.2: return the empirical threshold so the orchestrator can record
  it in the manifest's ``backtest`` block.
- 6.4: filter only. Never mutate ``predicted_direction`` (or any
  other attribute) of a surviving row, and never reorder.

Layer rules (`.sentrux/rules.toml`)
-----------------------------------

``portfolio`` is order=1 and ``harness`` is order=0, so this module
cannot import ``harness.evaluator.Record`` directly. Instead the
public function accepts any object exposing ``parse_ok: bool`` and
``p_memorized: float | None`` (structural typing via
:class:`typing.Protocol`). The orchestrator passes real ``Record``
instances; the unit tests pass ``types.SimpleNamespace`` stand-ins.

Surviving rows come back in input order, so callers can join the
output back to their original ``(date, ticker)`` indexing without
re-sorting.
"""

from __future__ import annotations

import math
from typing import Protocol, TypeVar, runtime_checkable

import numpy as np


@runtime_checkable
class _RecordLike(Protocol):
    """Minimal structural interface this module reads off each record.

    Any object exposing these two attributes works, including
    :class:`harness.evaluator.Record` (the real production type) and
    ad-hoc :class:`types.SimpleNamespace` stand-ins used in unit tests.
    """

    parse_ok: bool
    p_memorized: float | None


R = TypeVar("R", bound=_RecordLike)


def apply_cmmd_filter(
    records: list[R],
    quantile: float = 0.80,
) -> tuple[list[R], float]:
    """Drop the top ``(1 - quantile)`` slice of parse-OK ``p_memorized``.

    Two-stage filter:

    1. Drop rows where ``parse_ok`` is False or ``p_memorized`` is None
       or non-finite (NaN/inf cannot rank, and a single NaN would poison
       the quantile into dropping everything). The percentile is computed
       over the surviving distribution only, so failed rows cannot bias
       the cutoff.
    2. Drop the ``floor((1 - quantile) * n)`` highest-``p_memorized``
       survivors **by rank**. Ties at the cut boundary are broken
       deterministically: among equal scores, later input positions are
       dropped first, so the earliest rows survive. Rank-based dropping
       guarantees the intended slice is removed even when many rows tie
       at the cutoff value (an inclusive ``<= threshold`` rule would keep
       the entire tied block and filter nothing).

    The returned list preserves the input order of surviving rows
    (Req 6.4) and the function never mutates a row.

    Parameters
    ----------
    records:
        Iterable of record-shaped objects. Anything matching the
        :class:`_RecordLike` protocol is accepted; the production caller
        passes ``list[harness.evaluator.Record]``.
    quantile:
        Cut-point in ``(0, 1)``. The default ``0.80`` reproduces the
        paper's "drop top quintile" rule.

    Returns
    -------
    tuple[list[R], float]
        The surviving records (original order) and the empirical
        ``quantile``-th percentile of the surviving ``p_memorized``
        distribution. The percentile is provenance for the manifest;
        the drop rule itself is rank-based, so with heavy ties a kept
        row's score may equal (or exceed) this reported value.

    Raises
    ------
    ValueError
        When ``quantile`` is not strictly inside ``(0, 1)``.
    """
    if not (0.0 < quantile < 1.0):
        raise ValueError(f"quantile must be in (0, 1), got {quantile!r}")

    # Stage 1: drop parse failures and rows with no usable p_memorized.
    def _usable(r: R) -> bool:
        if not getattr(r, "parse_ok", False):
            return False
        p = getattr(r, "p_memorized", None)
        return p is not None and math.isfinite(float(p))

    surviving = [r for r in records if _usable(r)]
    if not surviving:
        return [], 0.0

    n = len(surviving)
    p_values = np.array([float(r.p_memorized) for r in surviving], dtype=np.float64)
    threshold = float(np.quantile(p_values, quantile))

    # Stage 2: rank-based top-slice. Round to 9 decimals before flooring so
    # binary-float artifacts ((1 - 0.8) * 100 == 19.999...96) cannot shave a
    # row off the intended slice.
    n_drop = math.floor(round((1.0 - quantile) * n, 9))
    if n_drop == 0:
        return list(surviving), threshold

    # Stable ascending sort by (score, input position): the last n_drop
    # entries are the highest scores, and among ties the latest input
    # positions — so the earliest tied rows survive, deterministically.
    order = sorted(range(n), key=lambda i: (p_values[i], i))
    kept_idx = sorted(order[: n - n_drop])
    return [surviving[i] for i in kept_idx], threshold
