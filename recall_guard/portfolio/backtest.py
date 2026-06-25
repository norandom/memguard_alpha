"""Long-short cross-sectional backtest engine for the cmmd-backtest spec.

Implements Requirements 5.1–5.8, 6.3, 6.5, and 9.4 of the
``cmmd-backtest`` spec: take a stream of harness ``Record`` objects plus
a ``(date × ticker)`` close-price matrix, build a daily-rebalanced
target-weight matrix where ``weight[d, t] = direction[d, t] *
confidence[d, t]``, and run two backtests through ``vectorbt`` —
``raw_alpha`` (no filter) and ``cmmd`` (top-quintile ``p_memorized``
rows removed). The two variants share one price matrix; the only
difference is the surviving record stream.

This module is the "engine core" of task 2.4. It owns the dataclasses
:class:`BacktestMetrics` and :class:`BacktestResult`, plus the
:func:`run_backtest` entry point. Task 2.5 will add an artifact writer
in this same file.

Layer rules and design deviation
--------------------------------

The ``portfolio`` layer is order=1 in ``.sentrux/rules.toml`` and the
``harness`` layer is order=0 (top of stack). Order=1 cannot import from
order=0, so this module never imports ``harness.evaluator.Record``.
Instead :func:`run_backtest` accepts any record-shaped object exposing
the attributes ``parse_ok``, ``predicted_direction``, ``raw_confidence``,
``p_memorized``, and ``prompt_hash`` — structural typing via
:class:`typing.Protocol`. This matches the strategy already used by
``portfolio.cmmd``.

The ``Record`` dataclass produced by ``harness.evaluator`` does not
carry ``metadata.date`` or ``metadata.ticker`` (the harness flattens
metadata into ``prompt_hash`` indexes and stores the metadata on the
eval-set side). To build a ``(date × ticker)`` weight matrix, this
function therefore takes an additional ``prompt_metadata`` argument
mapping each record's ``prompt_hash`` to ``{"ticker": str, "date": str
(ISO-8601)}``. The orchestrator script (task 3.2) is responsible for
joining the records to the eval-set rows and constructing this dict —
the engine itself stays pure-compute.

Determinism
-----------

The engine is fully deterministic given identical ``records``,
``prices``, ``prompt_metadata``, and ``seed``. Vectorbt's portfolio
construction is itself deterministic; the only stochastic step is the
bootstrap CI on Sharpe and mean daily return, which threads ``seed``
through ``core.bootstrap.bootstrap_ci``.

Key contracts
-------------

- The returned ``BacktestResult.equity_curves`` DataFrame has columns
  ``["raw_alpha", "cmmd", "buy_and_hold_swda"]`` in that exact order,
  with the first row equal to ``[1.0, 1.0, 1.0]`` (Req 7.2).
- ``BacktestResult.daily_returns_bps`` has columns ``["raw_alpha",
  "cmmd"]`` and is denominated in basis points (×10⁴).
- ``BacktestMetrics.max_drawdown_pct`` is signed: a negative number
  reports a drawdown (e.g., -3.4 means -3.4%). The spec is ambiguous
  on the sign so we pick signed-percent and document.

vectorbt 0.28 conventions
-------------------------

- ``size_type='targetpercent'`` rebalances to the target weight on every
  bar. Combined with ``cash_sharing=True, group_by=True`` this gives
  one portfolio across all tickers; vectorbt deducts trading fees on
  the trade notional, which equals ``|Δw_t|`` × portfolio value at the
  rebalance bar.
- ``freq='1D'`` sets the annualisation factor (252 trading days/year)
  for ``Portfolio.sharpe_ratio()``.
- The ``size`` matrix passed to ``from_orders`` already contains BIL's
  residual allocation, so vectorbt charges the BIL purchase as a real
  trade — that matches Req 5.4's "BIL holds the residual" semantics
  and Req 5.6's per-position cost.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import matplotlib

# Headless render path: harness runs in CI / containerised environments
# where no DISPLAY is set. ``Agg`` is the standard non-interactive
# backend and produces deterministic PNG bytes.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  -- after backend pin
import numpy as np
import pandas as pd
import vectorbt as vbt

from recall_guard.core.bootstrap import bootstrap_ci
from recall_guard.portfolio.cmmd import apply_cmmd_filter

logger = logging.getLogger(__name__)


# -------------------------------------------------------------- error classes


class BacktestArtifactError(RuntimeError):
    """Raised when writing the backtest artifacts to disk fails.

    The writer builds every artifact in memory before touching disk and
    rolls back any partially-written files on failure (Req 7.6), so by
    the time this exception surfaces the run directory is in the same
    state it was in before the call.
    """


# ---------------------------------------------------------------- public API


@runtime_checkable
class _RecordLike(Protocol):
    """Structural interface this module reads off each record.

    Any object exposing these attributes works — including
    ``harness.evaluator.Record`` (the production type) and
    ``types.SimpleNamespace`` stand-ins used in unit tests.
    """

    parse_ok: bool
    predicted_direction: int | None
    raw_confidence: float | None
    p_memorized: float | None
    prompt_hash: str


@dataclass(frozen=True)
class BacktestMetrics:
    """Per-variant summary statistics for one backtest run.

    All fields are JSON-friendly so downstream artifact writers (task
    2.5) can dump the dataclass directly. The ``sharpe_annualised`` and
    ``mean_daily_return_bps`` tuples are ``(point, lo, hi)`` from
    ``core.bootstrap.bootstrap_ci`` — point estimate first, then the
    95% percentile bounds.
    """

    label: str
    sharpe_annualised: tuple[float, float, float]
    mean_daily_return_bps: tuple[float, float, float]
    max_drawdown_pct: float
    total_return_pct: float
    n_trading_days: int
    n_signals_used: int
    cmmd_threshold: float | None


@dataclass(frozen=True)
class BacktestResult:
    """Bundle of both variants and the curves needed for plotting.

    ``equity_curves`` is normalised to start at ``1.0`` on the first
    trading day; ``daily_returns_bps`` is in basis points (×10⁴).
    """

    raw: BacktestMetrics
    cmmd: BacktestMetrics
    relative_sharpe_improvement: float
    equity_curves: pd.DataFrame
    daily_returns_bps: pd.DataFrame
    warnings: list[str]


def run_backtest(
    records: list[Any],
    prices: pd.DataFrame,
    prompt_metadata: dict[str, dict[str, str]],
    *,
    cmmd_quantile: float = 0.80,
    fees_one_way: float = 0.00075,
    init_cash: float = 1.0,
    seed: int = 0,
    bootstrap_n: int = 1000,
) -> BacktestResult:
    """Run the long-short backtest twice (raw + cmmd) on one price matrix.

    Args:
        records: List of record-shaped objects (see :class:`_RecordLike`).
            Records with ``parse_ok=False``, ``predicted_direction is
            None``, ``raw_confidence is None``, or no entry in
            ``prompt_metadata`` are dropped from BOTH variants
            (Req 9.1).
        prices: ``(date × ticker)`` close-price matrix. Must contain the
            ``BIL`` column plus at least one risk asset. The DataFrame's
            index is treated as the trading-day calendar; signals dated
            outside the index are dropped (Req 9.2).
        prompt_metadata: Maps each ``prompt_hash`` to a dict with
            ``"ticker"`` and ``"date"`` (ISO-8601) keys. Required because
            the harness ``Record`` schema does not carry date/ticker
            inline; the orchestrator builds this dict from the eval-set
            rows.
        cmmd_quantile: Quantile cut for the cmmd filter (default 0.80,
            i.e., drop top quintile by ``p_memorized``).
        fees_one_way: One-way trading cost in fractional notional
            (default 0.00075 = 7.5 bps; round-trip = 15 bps per the
            paper).
        init_cash: Initial portfolio value passed to vectorbt (default
            1.0 so equity curves start at 1.0).
        seed: Threaded through bootstrap CIs for determinism.
        bootstrap_n: Resamples for ``bootstrap_ci`` (default 1000).

    Returns:
        A :class:`BacktestResult` with both variants populated.

    Raises:
        ValueError: ``prices`` is empty or missing the BIL column.
        ValueError: ``cmmd_quantile`` outside ``(0, 1)``.
    """
    if prices.empty:
        raise ValueError("prices DataFrame must be non-empty.")
    if "BIL" not in prices.columns:
        raise ValueError("prices DataFrame must contain a 'BIL' column.")
    if not (0.0 < cmmd_quantile < 1.0):
        raise ValueError(
            f"cmmd_quantile must be in (0, 1), got {cmmd_quantile!r}"
        )

    # vectorbt's annualised Sharpe relies on the global frequency
    # setting. Pinning '1D' once per call keeps each invocation
    # self-contained and avoids cross-test bleed.
    vbt.settings.array_wrapper["freq"] = "1D"

    # -------- Stage 1: prepare records (drop parse failures) --------
    parse_ok_records = [
        r
        for r in records
        if getattr(r, "parse_ok", False)
        and getattr(r, "predicted_direction", None) is not None
        and getattr(r, "raw_confidence", None) is not None
        and getattr(r, "prompt_hash", None) in prompt_metadata
    ]

    warnings: list[str] = []
    # Req 9.4: < 30 surviving rows → low-row-count warning.
    if len(parse_ok_records) < 30:
        warnings.append("low-row-count")

    # -------- Stage 2: cmmd-filtered stream + threshold --------
    cmmd_records, cmmd_threshold = apply_cmmd_filter(
        parse_ok_records, quantile=cmmd_quantile
    )

    # -------- Stage 3: run both variants on the same price matrix --------
    raw_metrics, raw_returns, raw_equity = _run_one_variant(
        records=parse_ok_records,
        prices=prices,
        prompt_metadata=prompt_metadata,
        label="raw_alpha",
        cmmd_threshold=None,
        fees_one_way=fees_one_way,
        init_cash=init_cash,
        seed=seed,
        bootstrap_n=bootstrap_n,
    )

    cmmd_metrics, cmmd_returns, cmmd_equity = _run_one_variant(
        records=cmmd_records,
        prices=prices,
        prompt_metadata=prompt_metadata,
        label="cmmd",
        cmmd_threshold=cmmd_threshold,
        fees_one_way=fees_one_way,
        init_cash=init_cash,
        seed=seed,
        bootstrap_n=bootstrap_n,
    )

    # -------- Stage 4: buy-and-hold benchmark + bundle --------
    return _assemble_result(
        raw_metrics=raw_metrics, raw_returns=raw_returns, raw_equity=raw_equity,
        cmmd_metrics=cmmd_metrics, cmmd_returns=cmmd_returns, cmmd_equity=cmmd_equity,
        prices=prices, warnings=warnings,
    )


def _relative_sharpe_improvement(raw_point: float, cmmd_point: float) -> float:
    if raw_point != 0.0 and np.isfinite(raw_point):
        return float((cmmd_point - raw_point) / raw_point)
    return float("nan")


def _assemble_result(
    *,
    raw_metrics: BacktestMetrics, raw_returns: pd.Series, raw_equity: pd.Series,
    cmmd_metrics: BacktestMetrics, cmmd_returns: pd.Series, cmmd_equity: pd.Series,
    prices: pd.DataFrame, warnings: list[str],
) -> BacktestResult:
    bh_swda = _buy_and_hold_swda(prices)
    equity_curves = pd.DataFrame({
        "raw_alpha": raw_equity,
        "cmmd": cmmd_equity,
        "buy_and_hold_swda": bh_swda,
    })
    daily_returns_bps = pd.DataFrame({
        "raw_alpha": raw_returns * 1e4,
        "cmmd": cmmd_returns * 1e4,
    })
    rel = _relative_sharpe_improvement(
        raw_metrics.sharpe_annualised[0], cmmd_metrics.sharpe_annualised[0]
    )
    return BacktestResult(
        raw=raw_metrics,
        cmmd=cmmd_metrics,
        relative_sharpe_improvement=rel,
        equity_curves=equity_curves,
        daily_returns_bps=daily_returns_bps,
        warnings=warnings,
    )


# --------------------------------------------------------------- internals


def _run_one_variant(
    *,
    records: list[Any],
    prices: pd.DataFrame,
    prompt_metadata: dict[str, dict[str, str]],
    label: str,
    cmmd_threshold: float | None,
    fees_one_way: float,
    init_cash: float,
    seed: int,
    bootstrap_n: int,
) -> tuple[BacktestMetrics, pd.Series, pd.Series]:
    """Run one backtest variant on ``records``; return metrics + curves.

    Returns a triple ``(metrics, daily_returns, equity_curve)`` where
    the two series share the price matrix's DatetimeIndex.
    """
    weights = _build_weight_matrix(records, prices, prompt_metadata)
    pf = _run_portfolio(weights, prices, fees_one_way, init_cash)

    daily_returns = pf.returns()
    equity = pf.value() / init_cash  # normalise to start at ~1.0
    # Force first-row equity to exactly 1.0 in case vectorbt deducts the
    # entry fee on the first bar (returns are still recorded; equity
    # curve readers expect index[0] == 1.0 by Req 7.2).
    if not equity.empty:
        equity = equity / equity.iloc[0] if equity.iloc[0] != 0 else equity
        equity.iloc[0] = 1.0

    # ---- Point statistics --------------------------------------------------
    sharpe_point = float(pf.sharpe_ratio())
    if not np.isfinite(sharpe_point):
        sharpe_point = 0.0
    mean_daily_bps_point = float(daily_returns.mean() * 1e4)
    if not np.isfinite(mean_daily_bps_point):
        mean_daily_bps_point = 0.0

    # ---- Bootstrap CIs -----------------------------------------------------
    daily_returns_array = daily_returns.fillna(0.0).to_numpy()

    # Sharpe statistic mirrors vectorbt's annualisation (sqrt(252) *
    # mean / std with ddof=1) so the bootstrap CI is consistent with
    # ``pf.sharpe_ratio()``. We re-derive on each resample because
    # vectorbt does not expose a "given returns -> Sharpe" cheap path
    # without re-running a Portfolio.
    def _sharpe_stat(samples: list[float]) -> float:
        arr = np.asarray(samples, dtype=float)
        if arr.size < 2:
            raise ValueError("need >=2 samples for Sharpe")
        std = float(np.std(arr, ddof=1))
        if std == 0.0:
            raise ValueError("zero std")
        return float(np.sqrt(252.0) * np.mean(arr) / std)

    def _mean_bps_stat(samples: list[float]) -> float:
        return float(np.mean(samples) * 1e4)

    # The bootstrap point recomputes the statistic on the original
    # samples; for Sharpe we want vectorbt's value, so we override
    # point afterwards. The CI bounds are still drawn from the same
    # stat on the resamples, which is the right thing.
    if daily_returns_array.size >= 2:
        _, sharpe_lo, sharpe_hi = bootstrap_ci(
            list(daily_returns_array.tolist()),
            _sharpe_stat,
            n_resamples=bootstrap_n,
            confidence=0.95,
            seed=seed,
        )
        _, mean_lo, mean_hi = bootstrap_ci(
            list(daily_returns_array.tolist()),
            _mean_bps_stat,
            n_resamples=bootstrap_n,
            confidence=0.95,
            seed=seed,
        )
    else:
        # Degenerate single-day backtest: collapse CIs to the point.
        sharpe_lo = sharpe_hi = sharpe_point
        mean_lo = mean_hi = mean_daily_bps_point

    # Clamp documented postcondition lo <= point <= hi after the
    # vectorbt point overrides.
    sharpe_lo = min(sharpe_lo, sharpe_point)
    sharpe_hi = max(sharpe_hi, sharpe_point)
    mean_lo = min(mean_lo, mean_daily_bps_point)
    mean_hi = max(mean_hi, mean_daily_bps_point)

    # ---- Drawdown + total return -------------------------------------------
    max_dd = float(pf.max_drawdown())
    if not np.isfinite(max_dd):
        max_dd = 0.0
    # vectorbt returns drawdown as a non-positive fraction; convert to
    # signed percent (negative = drawdown) per module docstring.
    max_dd_pct = max_dd * 100.0

    total_ret = float(pf.total_return())
    if not np.isfinite(total_ret):
        total_ret = 0.0
    total_ret_pct = total_ret * 100.0

    metrics = BacktestMetrics(
        label=label,
        sharpe_annualised=(sharpe_point, float(sharpe_lo), float(sharpe_hi)),
        mean_daily_return_bps=(
            mean_daily_bps_point,
            float(mean_lo),
            float(mean_hi),
        ),
        max_drawdown_pct=max_dd_pct,
        total_return_pct=total_ret_pct,
        n_trading_days=int(len(prices)),
        n_signals_used=int(len(records)),
        cmmd_threshold=cmmd_threshold,
    )

    return metrics, daily_returns, equity


def _build_weight_matrix(
    records: list[Any],
    prices: pd.DataFrame,
    prompt_metadata: dict[str, dict[str, str]],
) -> pd.DataFrame:
    """Construct the ``(date × ticker)`` target-weight matrix.

    Steps (Req 5.3, 5.4, 5.5, 5.8):

    1. Initialise a zero matrix indexed by ``prices.index`` and columns
       ``prices.columns``.
    2. For each record, set ``weight_raw[date, ticker] = direction *
       confidence``. If multiple records map to the same (date,
       ticker) the later record wins (last-write-wins by record list
       position) — a contradiction in the eval set.
    3. For each row, compute ``s = sum(|weights|)`` over risk assets
       (everything except BIL). If ``s > 1`` scale the row by ``1/s``
       so risk-asset absolute sum equals 1 and BIL = 0 (Req 5.5).
       Else BIL = ``1 - s`` (Req 5.4 — long cash, never short).
    4. Days with no records map to ``BIL = 1`` (Req 5.8).
    """
    weights = pd.DataFrame(
        0.0,
        index=prices.index,
        columns=prices.columns,
        dtype=float,
    )

    # Map each record onto the price calendar. Records whose date isn't
    # in the index (e.g., LSE holiday) or whose ticker isn't a column
    # are silently dropped (Req 9.2 — single missing price → drop).
    index_dates = {ts.date(): ts for ts in prices.index}

    for record in records:
        meta = prompt_metadata.get(record.prompt_hash, {})
        ticker = meta.get("ticker")
        date_str = meta.get("date")
        if ticker is None or date_str is None:
            continue
        if ticker not in weights.columns:
            continue
        try:
            d = date.fromisoformat(date_str[:10])
        except ValueError:
            continue
        ts = index_dates.get(d)
        if ts is None:
            continue
        direction = float(record.predicted_direction)
        confidence = float(record.raw_confidence)
        weights.at[ts, ticker] = direction * confidence

    risk_cols = [c for c in weights.columns if c != "BIL"]
    abs_risk_sum = weights[risk_cols].abs().sum(axis=1)

    # Cap rows where leverage > 1: scale every risk weight by 1/sum.
    over_one = abs_risk_sum > 1.0
    if over_one.any():
        scale = 1.0 / abs_risk_sum[over_one]
        weights.loc[over_one, risk_cols] = weights.loc[over_one, risk_cols].mul(
            scale, axis=0
        )
        weights.loc[over_one, "BIL"] = 0.0

    # For rows with leverage <= 1, BIL gets the residual (so the row
    # sums to 1.0 in absolute terms, and is long cash whenever risk
    # weight is short or partial).
    under_or_eq = ~over_one
    if under_or_eq.any():
        weights.loc[under_or_eq, "BIL"] = 1.0 - abs_risk_sum[under_or_eq]

    return weights


def _run_portfolio(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    fees_one_way: float,
    init_cash: float,
) -> vbt.Portfolio:
    """Run vectorbt with target-percent rebalancing on the weight matrix.

    Uses ``size_type='targetpercent'``; vectorbt then handles rebalance
    deltas, fees on trade notional, and equity-curve construction. The
    portfolio is grouped + cash-shared across all tickers so we get a
    single combined value/return series.
    """
    return vbt.Portfolio.from_orders(
        close=prices,
        size=weights,
        size_type="targetpercent",
        fees=fees_one_way,
        init_cash=init_cash,
        cash_sharing=True,
        group_by=True,
        freq="1D",
    )


def _buy_and_hold_swda(prices: pd.DataFrame) -> pd.Series:
    """Buy-and-hold SWDA.L equity curve normalised to 1.0 at day 0.

    Falls back to a flat-1.0 curve when SWDA.L is missing from the
    price matrix — e.g., a smoke test that drops the ETF column.
    """
    if "SWDA.L" not in prices.columns:
        return pd.Series(1.0, index=prices.index, name="buy_and_hold_swda")
    swda = prices["SWDA.L"].astype(float)
    base = float(swda.iloc[0])
    if base == 0.0:
        return pd.Series(1.0, index=prices.index, name="buy_and_hold_swda")
    return (swda / base).rename("buy_and_hold_swda")


# ----------------------------------------------------------- artifact writers
#
# Task 2.5: build all five artifacts in memory, then write atomically.
# Any failure during the write phase rolls back files this call wrote
# and re-raises as :class:`BacktestArtifactError` (Req 7.6).


_SUMMARY_COLUMNS = (
    "label",
    "sharpe_point",
    "sharpe_lo",
    "sharpe_hi",
    "mean_bps_point",
    "mean_bps_lo",
    "mean_bps_hi",
    "max_drawdown_pct",
    "total_return_pct",
    "n_trading_days",
    "n_signals_used",
    "cmmd_threshold",
)


def _summary_row(m: BacktestMetrics) -> dict[str, Any]:
    """Flatten one ``BacktestMetrics`` into the summary CSV's row dict."""
    sharpe_p, sharpe_lo, sharpe_hi = m.sharpe_annualised
    mean_p, mean_lo, mean_hi = m.mean_daily_return_bps
    return {
        "label": m.label,
        "sharpe_point": sharpe_p,
        "sharpe_lo": sharpe_lo,
        "sharpe_hi": sharpe_hi,
        "mean_bps_point": mean_p,
        "mean_bps_lo": mean_lo,
        "mean_bps_hi": mean_hi,
        "max_drawdown_pct": m.max_drawdown_pct,
        "total_return_pct": m.total_return_pct,
        "n_trading_days": m.n_trading_days,
        "n_signals_used": m.n_signals_used,
        # ``cmmd_threshold`` is None for raw_alpha → empty string in CSV.
        "cmmd_threshold": (
            "" if m.cmmd_threshold is None else m.cmmd_threshold
        ),
    }


def _build_summary_csv(result: BacktestResult) -> str:
    """Render ``backtest_summary.csv`` text from the result."""
    df = pd.DataFrame(
        [_summary_row(result.raw), _summary_row(result.cmmd)],
        columns=list(_SUMMARY_COLUMNS),
    )
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def _fmt_metric_md(m: BacktestMetrics) -> str:
    """Format one ``BacktestMetrics`` as a markdown table row."""
    sharpe_p, sharpe_lo, sharpe_hi = m.sharpe_annualised
    mean_p, mean_lo, mean_hi = m.mean_daily_return_bps
    threshold = "—" if m.cmmd_threshold is None else f"{m.cmmd_threshold:.4f}"
    return (
        f"| {m.label} "
        f"| {sharpe_p:.3f} ({sharpe_lo:.3f}, {sharpe_hi:.3f}) "
        f"| {mean_p:.2f} ({mean_lo:.2f}, {mean_hi:.2f}) "
        f"| {m.max_drawdown_pct:.2f} "
        f"| {m.total_return_pct:.2f} "
        f"| {m.n_trading_days} "
        f"| {m.n_signals_used} "
        f"| {threshold} |"
    )


def _build_summary_md(result: BacktestResult) -> str:
    """Render ``backtest_summary.md`` text from the result.

    Includes both ``raw_alpha`` and ``cmmd`` rows (so README ingestion
    can grep them), the relative Sharpe improvement, and any warnings
    (e.g., ``low-row-count``).
    """
    rel = result.relative_sharpe_improvement
    rel_str = "n/a" if not np.isfinite(rel) else f"{rel * 100:+.2f}%"
    warnings_section = ""
    if result.warnings:
        bullets = "\n".join(f"- {w}" for w in result.warnings)
        warnings_section = f"\n## Warnings\n\n{bullets}\n"
    header = (
        "# Backtest summary\n\n"
        "| variant | sharpe (95% CI) | mean bps/day (95% CI) "
        "| max drawdown % | total return % | n_days | n_signals "
        "| cmmd_threshold |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    rows = (
        _fmt_metric_md(result.raw)
        + "\n"
        + _fmt_metric_md(result.cmmd)
        + "\n"
    )
    rel_section = (
        f"\n**Relative Sharpe improvement** "
        f"`(cmmd - raw) / raw` = {rel_str}\n"
    )
    return header + rows + rel_section + warnings_section


def _build_equity_csv(result: BacktestResult) -> str:
    """Render ``equity_curves.csv`` text (date index + 3 series)."""
    buf = io.StringIO()
    result.equity_curves.to_csv(buf, index=True, index_label="date")
    return buf.getvalue()


def _build_returns_csv(result: BacktestResult) -> str:
    """Render ``daily_returns.csv`` text (date index + 2 series)."""
    buf = io.StringIO()
    result.daily_returns_bps.to_csv(buf, index=True, index_label="date")
    return buf.getvalue()


def _annotate_drawdown(ax: Any, series: pd.Series, label: str) -> None:
    """Annotate the max-drawdown trough on ``ax`` for one equity series.

    Computes the running max, finds the (date, value) where drawdown
    is largest in absolute terms, and pins an ``ax.annotate`` arrow at
    that point.
    """
    if series.empty:
        return
    running_max = series.cummax()
    drawdown = series / running_max - 1.0
    if not drawdown.notna().any():
        return
    trough_idx = drawdown.idxmin()
    trough_val = float(series.loc[trough_idx])
    dd_pct = float(drawdown.loc[trough_idx]) * 100.0
    ax.annotate(
        f"{label} max DD {dd_pct:.1f}%",
        xy=(trough_idx, trough_val),
        xytext=(10, -20),
        textcoords="offset points",
        fontsize=8,
        arrowprops={"arrowstyle": "->", "alpha": 0.6},
    )


def _build_equity_png(result: BacktestResult) -> bytes:
    """Render ``equity_curves.png`` to PNG bytes via matplotlib."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    try:
        eq = result.equity_curves
        for column in ("raw_alpha", "cmmd", "buy_and_hold_swda"):
            if column in eq.columns:
                ax.plot(eq.index, eq[column].to_numpy(), label=column)
        # Drawdown annotations on the two strategy curves only.
        for column in ("raw_alpha", "cmmd"):
            if column in eq.columns:
                _annotate_drawdown(ax, eq[column], column)
        ax.set_xlabel("date")
        ax.set_ylabel("equity (start = 1.0)")
        ax.set_title("Backtest equity curves")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        return buf.getvalue()
    finally:
        plt.close(fig)


def write_backtest_artifacts(
    result: BacktestResult,
    run_dir: Path | str,
) -> dict[str, Path]:
    """Write the five backtest artifacts to ``run_dir`` atomically.

    Builds every artifact in memory first; only after every artifact
    has been successfully built does the writer touch disk. If any
    write raises ``OSError`` (disk full, permission denied, etc.) the
    function unlinks any files it has already written this call and
    re-raises as :class:`BacktestArtifactError` so the run directory
    is left in the state it was in before the call (Req 7.6).

    Args:
        result: The :class:`BacktestResult` to serialise.
        run_dir: Directory the artifacts should land in. Must exist.

    Returns:
        ``{artifact_name: Path}`` for the five files written. Keys:
        ``backtest_summary_csv``, ``backtest_summary_md``,
        ``equity_curves_csv``, ``equity_curves_png``,
        ``daily_returns_csv`` — matching the manifest's
        ``backtest.artifacts`` block in ``design.md``.

    Raises:
        BacktestArtifactError: any IO failure during the write phase.
            The run directory is rolled back to its pre-call state.
    """
    run_path = Path(run_dir)

    # ----- in-memory build phase (any error here surfaces as a normal
    # exception; nothing has been written yet so there's no rollback).
    payloads: dict[str, tuple[Path, bytes, bool]] = {
        # key -> (path, bytes, is_text_for_logging)
        "backtest_summary_csv": (
            run_path / "backtest_summary.csv",
            _build_summary_csv(result).encode("utf-8"),
            True,
        ),
        "backtest_summary_md": (
            run_path / "backtest_summary.md",
            _build_summary_md(result).encode("utf-8"),
            True,
        ),
        "equity_curves_csv": (
            run_path / "equity_curves.csv",
            _build_equity_csv(result).encode("utf-8"),
            True,
        ),
        "equity_curves_png": (
            run_path / "equity_curves.png",
            _build_equity_png(result),
            False,
        ),
        "daily_returns_csv": (
            run_path / "daily_returns.csv",
            _build_returns_csv(result).encode("utf-8"),
            True,
        ),
    }

    # ----- atomic write phase -------------------------------------------------
    written: list[Path] = []
    try:
        for _key, (path, blob, _is_text) in payloads.items():
            path.write_bytes(blob)
            written.append(path)
    except OSError as exc:
        # Roll back everything this call has written so the directory
        # is "untouched" relative to before the call.
        for p in written:
            try:
                p.unlink()
            except OSError:
                # Best-effort rollback; the original error is still the
                # one that matters.
                logger.warning("rollback unlink failed for %s", p)
        target = getattr(exc, "filename", None) or "unknown path"
        raise BacktestArtifactError(
            f"failed to write backtest artifact {target}: {exc}"
        ) from exc

    logger.info("write_backtest_artifacts done -> %s", run_path)
    return {key: payloads[key][0] for key in payloads}
