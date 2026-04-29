"""Unit tests for ``src.portfolio.backtest.run_backtest``.

Covers Requirements 5.1–5.8, 6.3, 6.5, and 9.4 of the cmmd-backtest spec
on a deterministic 5-trading-day toy universe so every metric is
hand-verifiable. The tests exercise:

- BIL residual routing when ``sum(|weights|) < 1`` (Req 5.4).
- Leverage cap when ``sum(|weights|) > 1`` (Req 5.5).
- 15 bps round-trip cost on a ``|Δw| = 1`` trade — 7.5 bps each side
  (Req 5.6).
- Total return on an analytically simple price path (Req 5.7).
- Both ``raw`` and ``cmmd`` ``BacktestMetrics`` populated; equity curves
  start at 1.0 with the documented column set (Req 6.5).
- ``low-row-count`` warning emitted when fewer than 30 parse-OK rows
  survive across the horizon (Req 9.4).

Records are constructed as :class:`types.SimpleNamespace` instances —
the ``portfolio`` layer (order=1) cannot import from ``harness`` (order=0),
so the engine consumes any record-shaped object and the tests do
likewise.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from src.portfolio.backtest import (
    BacktestMetrics,
    BacktestResult,
    run_backtest,
)

# --------------------------------------------------------------------- helpers


def _toy_prices(n_days: int = 5) -> pd.DataFrame:
    """Build a 5-trading-day price matrix on the four-ticker universe.

    The price path is engineered so individual day returns are easy to
    verify by hand:

    - ``SWDA.L``: +1% per day every day.
    - ``XLK``: flat at 100.0 (zero return).
    - ``IAU``: -1% per day every day.
    - ``BIL``: flat at 100.0 (cash leg, zero return — same as XLK).

    The first row is the entry-day close; all weight changes happen
    relative to a zero-position prior day.
    """
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    swda = [100.0 * (1.01 ** i) for i in range(n_days)]
    xlk = [100.0] * n_days
    iau = [100.0 * (0.99 ** i) for i in range(n_days)]
    bil = [100.0] * n_days
    return pd.DataFrame(
        {"SWDA.L": swda, "XLK": xlk, "IAU": iau, "BIL": bil},
        index=dates,
    )


def _mk_record(
    *,
    prompt_hash: str,
    direction: int,
    confidence: float,
    p_memorized: float | None = 0.5,
    parse_ok: bool = True,
    target_direction: int = 0,
) -> SimpleNamespace:
    """Build a Record-shaped object exposing only the attributes the
    engine reads. The ``model`` field is a constant placeholder."""
    return SimpleNamespace(
        model="openai/gpt-oss-20b",
        prompt_hash=prompt_hash,
        parse_ok=parse_ok,
        predicted_direction=direction,
        raw_confidence=confidence,
        p_memorized=p_memorized,
        target_direction=target_direction,
    )


def _meta(ticker: str, day: pd.Timestamp) -> dict[str, str]:
    """Build a single ``prompt_metadata`` entry for a given (ticker, date)."""
    return {"ticker": ticker, "date": day.date().isoformat()}


# ----------------------------------------------------------------------- tests


def test_total_return_and_metrics_on_full_swda_long() -> None:
    """100% long SWDA.L for all 5 days produces a hand-computable total
    return.

    With ``SWDA.L`` rising by 1% each day, holding 100% from day 0 to
    day 4 yields a gross total return of ``1.01**4 - 1`` (the day-0
    return is zero because rebalancing happens at close on day 0). After
    fees: 7.5 bps deducted on the day-0 entry trade only (no exits, no
    later rebalances), so the net total return is approximately
    ``(1 - 0.00075) * 1.01**4 - 1``.
    """
    prices = _toy_prices(5)
    days = list(prices.index)

    # One record per (date, SWDA.L) — direction=+1, confidence=1.0.
    records = []
    prompt_metadata: dict[str, dict[str, str]] = {}
    for i, day in enumerate(days):
        ph = f"swda-{i}"
        records.append(_mk_record(prompt_hash=ph, direction=1, confidence=1.0))
        prompt_metadata[ph] = _meta("SWDA.L", day)

    result = run_backtest(
        records,
        prices,
        prompt_metadata,
        cmmd_quantile=0.80,
        fees_one_way=0.00075,
        init_cash=1.0,
        seed=0,
        bootstrap_n=64,
    )

    # Hand-computed: pure gross compounding then a 7.5 bps haircut on the
    # day-0 entry. (Vectorbt charges fees on the trade notional which
    # equals 100% of init_cash on entry, so the net is the gross *
    # (1 - fees_one_way).)
    gross_total = (1.01 ** 4) - 1.0
    net_total_expected = (1.0 - 0.00075) * (1.0 + gross_total) - 1.0
    assert result.raw.total_return_pct == pytest.approx(
        net_total_expected * 100.0, abs=0.05
    )

    # Equity curve has the documented columns and starts at 1.0.
    assert list(result.equity_curves.columns) == ["raw_alpha", "cmmd", "buy_and_hold_swda"]
    assert result.equity_curves.iloc[0, :].to_list() == pytest.approx([1.0, 1.0, 1.0])

    # Sharpe is finite and positive on a strictly-rising equity curve.
    point, lo, hi = result.raw.sharpe_annualised
    assert point > 0
    assert lo <= point <= hi

    # n_signals_used reflects the records that actually populated weights.
    assert result.raw.n_signals_used == 5
    assert result.raw.n_trading_days == 5


def test_15bps_round_trip_cost_on_unit_weight_change() -> None:
    """A ``|Δw| = 1`` entry on day 0 deducts ~7.5 bps one-way.

    The strategy enters 100% XLK (flat-priced) on day 0 only. On day 1+
    weights revert to fully-cash (BIL, also flat). Daily returns:

    - day 0: -7.5 bps (entry fee on the XLK leg)
    - day 1: ~-15 bps (sell XLK + buy BIL = two unit-weight trades)
    - day 2-4: 0 bps (everything flat, no further trades)

    The test asserts the day-0 deduction matches 7.5 bps within
    floating-point tolerance, exercising Req 5.6's one-way half of the
    15 bps round-trip cost.
    """
    prices = _toy_prices(5)
    days = list(prices.index)

    # Only one signal: 100% long XLK on day 0. Day 1+ the absence of
    # records routes capital back to BIL.
    records = [_mk_record(prompt_hash="xlk-0", direction=1, confidence=1.0)]
    prompt_metadata = {"xlk-0": _meta("XLK", days[0])}

    result = run_backtest(
        records,
        prices,
        prompt_metadata,
        cmmd_quantile=0.80,
        fees_one_way=0.00075,
        init_cash=1.0,
        seed=0,
        bootstrap_n=64,
    )

    daily_bps = result.daily_returns_bps["raw_alpha"].to_list()

    # Day 0 entry: 7.5 bps deduction (the one-way fee).
    assert daily_bps[0] == pytest.approx(-7.5, abs=0.05)


def test_leverage_cap_scales_row_to_unit_norm() -> None:
    """When ``sum(|weights|) > 1``, every weight scales by ``1 / sum``.

    Construct day-0 records that demand ``|w_SWDA| + |w_XLK| = 1.5``.
    After the leverage cap each should equal ``1.0 / 1.5 ≈ 0.667``;
    BIL should be 0. The test does not verify the exact post-fee return
    (that's covered by the entry-fee test); it asserts the *ratio* of
    day-0 fee deduction matches the post-cap notional traded — i.e.,
    7.5 bps × 1.0 (capped sum) rather than 7.5 bps × 1.5 (uncapped).
    """
    prices = _toy_prices(5)
    days = list(prices.index)

    records = [
        _mk_record(prompt_hash="swda-0", direction=1, confidence=0.75),
        _mk_record(prompt_hash="xlk-0", direction=1, confidence=0.75),
    ]
    prompt_metadata = {
        "swda-0": _meta("SWDA.L", days[0]),
        "xlk-0": _meta("XLK", days[0]),
    }

    result = run_backtest(
        records,
        prices,
        prompt_metadata,
        cmmd_quantile=0.80,
        fees_one_way=0.00075,
        init_cash=1.0,
        seed=0,
        bootstrap_n=64,
    )

    daily_bps = result.daily_returns_bps["raw_alpha"].to_list()

    # Day-0 fees: capped sum=1.0 -> 7.5 bps total trading cost.
    # Day-0 P&L on 0.667 SWDA + 0.333 XLK: SWDA's day-0 return is 0
    # (price hasn't moved yet from its open), so the *only* effect on
    # day 0 is the entry fee. Therefore daily_bps[0] ≈ -7.5.
    assert daily_bps[0] == pytest.approx(-7.5, abs=0.5)


def test_bil_residual_routing_on_partial_invested_row() -> None:
    """``sum(|weights|) = 0.4`` → BIL gets 0.6 of capital.

    Day 0 holds 0.4 SWDA.L and 0.6 BIL. Both flat-priced on entry day,
    so the day-0 return is ``-0.4 * 7.5 bps - 0.6 * 7.5 bps = -7.5 bps``
    on entry — wait: BIL purchase IS a trade and so DOES incur fees.
    Confirms that the residual routing keeps the row sum-to-1 property
    while still incurring the documented one-way fee.

    A subsequent flat-day return (day 1, after entry trades have
    settled) should be exactly ``0.4 * SWDA_ret + 0.6 * BIL_ret``.
    With SWDA at +1%/day and BIL at 0, the day-1 return is
    ``0.4 * 0.01 + 0.6 * 0 = 0.004`` (40 bps).
    """
    prices = _toy_prices(5)
    days = list(prices.index)

    # Records exist for day 0 only, 0.4 long SWDA. No XLK / IAU records
    # so those columns remain zero. BIL gets the residual 0.6.
    records = [
        _mk_record(prompt_hash="swda-0", direction=1, confidence=0.4),
    ]
    prompt_metadata = {"swda-0": _meta("SWDA.L", days[0])}
    # And we also keep this allocation for subsequent days (so day 1
    # fees are zero) — replicate the same 0.4 long SWDA on every day.
    for i, day in enumerate(days[1:], start=1):
        ph = f"swda-{i}"
        records.append(_mk_record(prompt_hash=ph, direction=1, confidence=0.4))
        prompt_metadata[ph] = _meta("SWDA.L", day)

    result = run_backtest(
        records,
        prices,
        prompt_metadata,
        cmmd_quantile=0.80,
        fees_one_way=0.00075,
        init_cash=1.0,
        seed=0,
        bootstrap_n=64,
    )

    daily_bps = result.daily_returns_bps["raw_alpha"].to_list()

    # Day 1 (no rebalance — same target weights as day 0): pure P&L
    # 0.4 * SWDA_ret(1%) + 0.6 * BIL_ret(0%) = 40 bps.
    assert daily_bps[1] == pytest.approx(40.0, abs=0.5)


def test_both_variants_populated_and_curves_start_at_one() -> None:
    """Result has both ``raw`` and ``cmmd`` ``BacktestMetrics`` and the
    equity_curves DataFrame has the documented columns (raw_alpha, cmmd,
    buy_and_hold_swda) starting at 1.0."""
    prices = _toy_prices(5)
    days = list(prices.index)

    records = []
    prompt_metadata: dict[str, dict[str, str]] = {}
    # Sprinkle p_memorized so cmmd has something to filter (top quintile
    # = highest p_memorized rows get dropped).
    p_values = [0.1, 0.2, 0.3, 0.95, 0.05]
    for i, day in enumerate(days):
        ph = f"swda-{i}"
        records.append(
            _mk_record(
                prompt_hash=ph,
                direction=1,
                confidence=0.5,
                p_memorized=p_values[i],
            )
        )
        prompt_metadata[ph] = _meta("SWDA.L", day)

    result = run_backtest(
        records,
        prices,
        prompt_metadata,
        cmmd_quantile=0.80,
        fees_one_way=0.00075,
        init_cash=1.0,
        seed=0,
        bootstrap_n=64,
    )

    assert isinstance(result, BacktestResult)
    assert isinstance(result.raw, BacktestMetrics)
    assert isinstance(result.cmmd, BacktestMetrics)
    assert result.raw.label == "raw_alpha"
    assert result.cmmd.label == "cmmd"
    assert result.cmmd.cmmd_threshold is not None
    assert result.raw.cmmd_threshold is None

    assert list(result.equity_curves.columns) == [
        "raw_alpha",
        "cmmd",
        "buy_and_hold_swda",
    ]
    first_row = result.equity_curves.iloc[0, :].to_list()
    assert first_row == pytest.approx([1.0, 1.0, 1.0])

    assert list(result.daily_returns_bps.columns) == ["raw_alpha", "cmmd"]


def test_low_row_count_warning_when_fewer_than_30_signals() -> None:
    """Fewer than 30 parse-OK rows in the horizon → ``low-row-count``
    warning appended to ``result.warnings`` (Req 9.4)."""
    prices = _toy_prices(5)
    days = list(prices.index)

    records = [
        _mk_record(prompt_hash="swda-0", direction=1, confidence=0.5),
    ]
    prompt_metadata = {"swda-0": _meta("SWDA.L", days[0])}

    result = run_backtest(
        records,
        prices,
        prompt_metadata,
        cmmd_quantile=0.80,
        fees_one_way=0.00075,
        init_cash=1.0,
        seed=0,
        bootstrap_n=64,
    )

    assert "low-row-count" in result.warnings


def test_parse_failed_rows_dropped_from_both_streams() -> None:
    """Rows with ``parse_ok=False`` are dropped from the weight matrix
    used by both the raw_alpha and cmmd backtests (Req 9.1, 6.5)."""
    prices = _toy_prices(5)
    days = list(prices.index)

    records = [
        _mk_record(prompt_hash="swda-0", direction=1, confidence=1.0),
        _mk_record(
            prompt_hash="swda-1",
            direction=1,
            confidence=1.0,
            parse_ok=False,
        ),
    ]
    prompt_metadata = {
        "swda-0": _meta("SWDA.L", days[0]),
        "swda-1": _meta("SWDA.L", days[1]),
    }

    result = run_backtest(
        records,
        prices,
        prompt_metadata,
        cmmd_quantile=0.80,
        fees_one_way=0.00075,
        init_cash=1.0,
        seed=0,
        bootstrap_n=64,
    )

    # Only one parse-OK row → n_signals_used == 1 in raw_alpha.
    assert result.raw.n_signals_used == 1
