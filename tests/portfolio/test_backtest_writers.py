"""Unit tests for ``recall_guard.portfolio.backtest.write_backtest_artifacts``.

Covers Requirements 7.1, 7.2, 7.3, 7.4, and 7.6 of the cmmd-backtest spec.

The writer must build every artifact in memory first, then write all
five to disk atomically. On any IO failure during the write phase the
function raises :class:`BacktestArtifactError` and rolls back any files
it has already written, so the run directory is left as it was before
the call (Req 7.6).

Three tests cover the contract:

1. **Happy path** — five artifacts land on disk, the returned dict has
   the documented keys, the CSVs parse, and the PNG starts with the
   PNG magic bytes ``b"\\x89PNG"`` (Req 7.1, 7.2, 7.3, 7.4).
2. **MD content** — ``backtest_summary.md`` contains rows for both
   ``raw_alpha`` and ``cmmd`` so the README rendering can grep them.
3. **Atomic failure** — patch ``Path.write_bytes`` so that writing the
   PNG raises ``PermissionError``; assert :class:`BacktestArtifactError`
   bubbles out and the run directory contains zero of this call's
   artifacts after rollback (Req 7.6).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from recall_guard.portfolio.backtest import (
    BacktestArtifactError,
    BacktestMetrics,
    BacktestResult,
    write_backtest_artifacts,
)

# --------------------------------------------------------------------- helpers


def _build_fixture_result() -> BacktestResult:
    """Construct a minimal but realistic :class:`BacktestResult`.

    The numbers are chosen so the markdown table is human-readable and
    the equity curves have an unambiguous max-drawdown date — that
    drawdown is what the PNG annotation pins.
    """
    raw = BacktestMetrics(
        label="raw_alpha",
        sharpe_annualised=(0.42, 0.10, 0.74),
        mean_daily_return_bps=(3.5, 1.0, 6.0),
        max_drawdown_pct=-4.2,
        total_return_pct=12.3,
        n_trading_days=5,
        n_signals_used=15,
        cmmd_threshold=None,
    )
    cmmd = BacktestMetrics(
        label="cmmd",
        sharpe_annualised=(0.61, 0.30, 0.92),
        mean_daily_return_bps=(4.7, 2.0, 7.4),
        max_drawdown_pct=-2.1,
        total_return_pct=15.8,
        n_trading_days=5,
        n_signals_used=12,
        cmmd_threshold=0.873,
    )
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    equity_curves = pd.DataFrame(
        {
            "raw_alpha": [1.0, 1.01, 0.97, 1.02, 1.06],
            "cmmd": [1.0, 1.02, 1.00, 1.04, 1.09],
            "buy_and_hold_swda": [1.0, 1.005, 1.01, 1.02, 1.03],
        },
        index=idx,
    )
    daily_returns_bps = pd.DataFrame(
        {
            "raw_alpha": [0.0, 100.0, -395.0, 515.0, 392.0],
            "cmmd": [0.0, 200.0, -196.0, 400.0, 480.0],
        },
        index=idx,
    )
    return BacktestResult(
        raw=raw,
        cmmd=cmmd,
        relative_sharpe_improvement=(0.61 - 0.42) / 0.42,
        equity_curves=equity_curves,
        daily_returns_bps=daily_returns_bps,
        warnings=["low-row-count"],
    )


# ----------------------------------------------------------------------- tests


def test_happy_path_writes_five_files(tmp_path: Path) -> None:
    """All five artifacts land on disk with the documented dict keys.

    CSVs must parse cleanly; PNG must start with the PNG magic bytes.
    """
    result = _build_fixture_result()
    paths = write_backtest_artifacts(result, tmp_path)

    expected_keys = {
        "backtest_summary_csv",
        "backtest_summary_md",
        "equity_curves_csv",
        "equity_curves_png",
        "daily_returns_csv",
    }
    assert set(paths.keys()) == expected_keys

    for key, path in paths.items():
        assert path.exists(), f"missing artifact {key} at {path}"
        assert path.parent == tmp_path

    # CSVs parse without error and have the right top-level columns.
    summary_df = pd.read_csv(paths["backtest_summary_csv"])
    assert set(summary_df["label"].tolist()) == {"raw_alpha", "cmmd"}
    assert "sharpe_point" in summary_df.columns
    assert "cmmd_threshold" in summary_df.columns

    eq_df = pd.read_csv(paths["equity_curves_csv"], index_col=0)
    assert list(eq_df.columns) == ["raw_alpha", "cmmd", "buy_and_hold_swda"]

    dr_df = pd.read_csv(paths["daily_returns_csv"], index_col=0)
    assert list(dr_df.columns) == ["raw_alpha", "cmmd"]

    # PNG header check — first 4 bytes are the PNG signature.
    png_bytes = paths["equity_curves_png"].read_bytes()
    assert png_bytes.startswith(b"\x89PNG"), "equity_curves.png is not a PNG"
    assert len(png_bytes) > 200, "PNG suspiciously small"


def test_summary_md_lists_both_variants(tmp_path: Path) -> None:
    """``backtest_summary.md`` contains rows for both ``raw_alpha`` and
    ``cmmd`` so README ingestion can find them."""
    result = _build_fixture_result()
    paths = write_backtest_artifacts(result, tmp_path)
    md_text = paths["backtest_summary_md"].read_text(encoding="utf-8")
    assert "raw_alpha" in md_text
    assert "cmmd" in md_text
    # Warning surfaced when present.
    assert "low-row-count" in md_text


def test_atomic_failure_leaves_no_partial_files(
    tmp_path: Path, mocker: object
) -> None:
    """If the PNG write fails, :class:`BacktestArtifactError` is raised
    AND no artifact from this call survives in ``tmp_path``.

    Strategy: patch :meth:`pathlib.Path.write_bytes` so that any call
    targeting ``equity_curves.png`` raises ``PermissionError``. The
    writer should detect the failure mid-write, unlink any files it has
    already written this call, and re-raise as
    :class:`BacktestArtifactError`.
    """
    result = _build_fixture_result()
    real_write_bytes = Path.write_bytes

    def _failing_write_bytes(self: Path, data: bytes) -> int:
        if self.name == "equity_curves.png":
            raise PermissionError(f"simulated permission denied: {self}")
        return real_write_bytes(self, data)

    mocker.patch.object(Path, "write_bytes", _failing_write_bytes)  # type: ignore[attr-defined]

    with pytest.raises(BacktestArtifactError) as exc_info:
        write_backtest_artifacts(result, tmp_path)

    msg = str(exc_info.value)
    assert "equity_curves.png" in msg

    # No artifact from this call survives. Iterate the dir and assert
    # none of the documented filenames are present.
    leftovers = {p.name for p in tmp_path.iterdir()}
    documented = {
        "backtest_summary.csv",
        "backtest_summary.md",
        "equity_curves.csv",
        "equity_curves.png",
        "daily_returns.csv",
    }
    assert documented.isdisjoint(leftovers), (
        f"partial artifacts left behind: {leftovers & documented}"
    )
