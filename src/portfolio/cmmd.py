"""Top-quintile ``p_memorized`` filter for the cmmd-backtest spec.

Implements Requirements 6.1, 6.2, and 6.4 of the cmmd-backtest spec:

- 6.1: drop the top ``(1 - quantile)`` slice of the parse-OK
  ``p_memorized`` distribution (default = top 20%).
- 6.2: return the empirical threshold so the orchestrator can record it
  in the manifest's ``backtest`` block for provenance.
- 6.4: filter only — never mutate ``predicted_direction`` or any other
  attribute of a surviving row, and never reorder them.

Layer rules (`.sentrux/rules.toml`):

The ``portfolio`` layer is order=1 and the ``harness`` layer is order=0
(top of the stack). Order=1 cannot import from order=0, so this module
must NOT import ``harness.evaluator.Record`` directly. Instead the
public function takes any record-shaped object that exposes
``parse_ok: bool`` and ``p_memorized: float | None`` attributes —
structural typing via :class:`typing.Protocol`. This keeps the function
trivially callable from the orchestrator (which holds real ``Record``
instances) while staying inside the layer rules and remaining easy to
unit-test with ``types.SimpleNamespace`` stand-ins.

Order stability: surviving rows are returned in the order they appeared
in the input, so callers can join the output back to their original
``(date, ticker)`` indexing without an extra sort pass.
"""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

import numpy as np


@runtime_checkable
class _RecordLike(Protocol):
    """Minimal structural interface this module reads off each record.

    Any object exposing these two attributes works — including
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
       — the percentile is computed over the surviving distribution
       only, so failed rows cannot bias the cutoff.
    2. Compute the empirical ``quantile``-th percentile of
       ``p_memorized`` on the survivors and keep rows where
       ``p_memorized <= threshold``.

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
        ``p_memorized`` threshold value used.

    Raises
    ------
    ValueError
        When ``quantile`` is not strictly inside ``(0, 1)``.
    """
    if not (0.0 < quantile < 1.0):
        raise ValueError(f"quantile must be in (0, 1), got {quantile!r}")

    # Stage 1: drop parse failures and rows with no p_memorized.
    surviving = [
        r
        for r in records
        if getattr(r, "parse_ok", False) and getattr(r, "p_memorized", None) is not None
    ]
    if not surviving:
        return [], 0.0

    # Stage 2: empirical percentile on the surviving distribution only.
    p_values = np.array([float(r.p_memorized) for r in surviving], dtype=np.float64)
    threshold = float(np.quantile(p_values, quantile))

    kept = [r for r in surviving if float(r.p_memorized) <= threshold]
    return kept, threshold
