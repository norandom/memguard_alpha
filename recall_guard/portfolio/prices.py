"""FMP-backed EOD price fetcher for the cmmd-backtest universe.

Pulls end-of-day close prices for SWDA.L, XLK, IAU, and BIL from FMP's
``historical-price-eod/light`` endpoint, aligns them with an inner-join
on date, and returns a single ``pandas.DataFrame`` with one column per
ticker. Covers Reqs 5.1, 5.7, 7.2, and 9.2.

Public surface:

- ``PriceFetchError`` is raised on HTTP failure or when the inner-join
  has fewer than 30 aligned trading days.
- ``fetch_universe_prices`` is the only entry point.

This is the only place in the ``portfolio`` layer that performs HTTP
I/O; tests mock ``requests.get``. The retry / API-key resolution
pattern is the same one used by ``recall_guard.dataset.fmp_corpora.fetch_articles``,
duplicated locally because the sentrux ``portfolio ↔ dataset`` boundary
forbids the import.
"""

from __future__ import annotations

import os
from datetime import date

import pandas as pd
import requests

# Below 30 aligned trading days, the joined sample is too small for the
# backtest layer to produce meaningful Sharpe / drawdown numbers. Fail
# here rather than letting downstream artifacts look authoritative.
_MIN_OVERLAP_DAYS = 30


class PriceFetchError(RuntimeError):
    """Raised when an FMP price fetch fails or returns insufficient data.

    Carries either the offending ticker plus HTTP status code (transport
    failure) or the offending ticker plus aligned-day count (overlap
    failure). The orchestrator script presents this directly to stderr.
    """


def _resolve_api_key(api_key: str | None) -> str:
    """Resolve an FMP API key from the explicit arg or the environment.

    Same logic as ``recall_guard.dataset.fmp_corpora._resolve_api_key``,
    duplicated because the sentrux ``portfolio ↔ dataset`` boundary
    forbids the import.
    """
    if api_key:
        return api_key
    env_key = os.environ.get("FMP_API_KEY")
    if not env_key:
        raise PriceFetchError(
            "FMP_API_KEY is not set; pass api_key= or export FMP_API_KEY."
        )
    return env_key


def _fetch_one_ticker(
    ticker: str,
    api_key: str,
    start: date,
    end: date,
) -> pd.Series:
    """Fetch one ticker's EOD close series, filtered to ``[start, end]``.

    Returns a ``pandas.Series`` indexed by ``DatetimeIndex`` with the
    close prices as float values; the series is named after the ticker
    so downstream ``concat`` produces a clean column.
    """
    # FMP's stable endpoints cap history at five years unless ``from`` /
    # ``to`` are supplied; we pass the explicit window so the 10-year
    # cmmd-backtest universe gets the full price history.
    url = (
        "https://financialmodelingprep.com/stable/historical-price-eod/light"
        f"?symbol={ticker}"
        f"&from={start.isoformat()}&to={end.isoformat()}"
        f"&apikey={api_key}"
    )
    try:
        response = requests.get(url, timeout=30)
    except requests.RequestException as exc:
        raise PriceFetchError(
            f"FMP price fetch for {ticker!r} failed: {exc}"
        ) from exc
    if response.status_code != 200:
        raise PriceFetchError(
            f"FMP price fetch for {ticker!r} returned HTTP {response.status_code}"
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise PriceFetchError(
            f"FMP price fetch for {ticker!r} returned invalid JSON"
        ) from exc
    if not isinstance(payload, list):
        # FMP's documented shape is a top-level list. Anything else
        # means the response is not what we expect, so fail loudly
        # instead of returning an empty series.
        raise PriceFetchError(
            f"FMP price fetch for {ticker!r} returned non-list payload"
        )

    rows: list[tuple[date, float]] = []
    for entry in payload:
        if not isinstance(entry, dict):
            raise PriceFetchError(
                f"FMP price fetch for {ticker!r} returned non-object row payload"
            )
        raw_date = entry.get("date")
        raw_price = entry.get("price")
        if not isinstance(raw_date, str) or raw_price is None:
            continue
        try:
            d = date.fromisoformat(raw_date[:10])
        except ValueError:
            continue
        if d < start or d > end:
            continue
        try:
            price = float(raw_price)
        except (TypeError, ValueError):
            continue
        rows.append((d, price))

    if not rows:
        # Return an empty named series. The inner-join will collapse
        # the combined frame to zero rows and the caller's
        # min-overlap check then names this ticker.
        return pd.Series(dtype=float, name=ticker)

    # Deduplicate by date (last write wins) and sort ascending.
    by_date: dict[date, float] = {}
    for d, p in rows:
        by_date[d] = p
    sorted_dates = sorted(by_date.keys())
    series = pd.Series(
        data=[by_date[d] for d in sorted_dates],
        index=pd.to_datetime(sorted_dates),
        name=ticker,
        dtype=float,
    )
    return series


def fetch_universe_prices(
    tickers: list[str],
    start: date,
    end: date,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Return aligned (date × ticker) close-price matrix for the universe.

    Fetches each ticker's EOD close series from FMP's
    ``historical-price-eod/light`` endpoint, filters each series to the
    ``[start, end]`` window, and inner-joins on date so any day where any
    ticker is missing (e.g. LSE holiday vs NYSE) is dropped uniformly.

    Args:
        tickers: Ordered list of FMP symbols. Output column order
            matches this list.
        start: Inclusive lower bound for retained dates.
        end: Inclusive upper bound for retained dates.
        api_key: Explicit FMP key; falls back to the ``FMP_API_KEY``
            environment variable.

    Returns:
        A ``DataFrame`` with a monotonic ``DatetimeIndex``, one column
        per ticker (in input order), and no NaN cells.

    Raises:
        RuntimeError: ``FMP_API_KEY`` is not set and no ``api_key`` was
            passed.
        PriceFetchError: any individual ticker request fails, or the
            inner-joined frame has fewer than 30 aligned trading days.
        ValueError: ``tickers`` is empty or ``start > end``.
    """
    if not tickers:
        raise ValueError("tickers must be a non-empty list of FMP symbols.")
    if start > end:
        raise ValueError(
            f"start ({start.isoformat()}) must be <= end ({end.isoformat()})."
        )

    api_key = _resolve_api_key(api_key)

    series_by_ticker: dict[str, pd.Series] = {}
    for ticker in tickers:
        series_by_ticker[ticker] = _fetch_one_ticker(ticker, api_key, start, end)

    # Inner-join across tickers: any date missing from any ticker is
    # dropped from the combined frame.
    joined = pd.concat(
        [series_by_ticker[t] for t in tickers],
        axis=1,
        join="inner",
    )
    # Re-assert input column order. ``concat`` already preserves it,
    # but reassigning is cheap insurance against an empty series.
    joined.columns = list(tickers)

    # Drop residual NaN rows. ``concat(..., join='inner')`` only
    # filters by the index, so a date present everywhere but with one
    # NaN value would still survive without this.
    joined = joined.dropna(how="any")

    if len(joined) < _MIN_OVERLAP_DAYS:
        # Name the ticker that contributed the fewest raw rows so the
        # caller knows where to look.
        per_ticker_counts = {
            t: len(series_by_ticker[t]) for t in tickers
        }
        worst_ticker = min(per_ticker_counts, key=per_ticker_counts.get)
        worst_n = per_ticker_counts[worst_ticker]
        raise PriceFetchError(
            f"ticker {worst_ticker!r} has only {worst_n} raw rows "
            f"({len(joined)} aligned trading days after inner-join) "
            f"in window {start.isoformat()}..{end.isoformat()} "
            f"(need >= {_MIN_OVERLAP_DAYS})"
        )

    return joined
