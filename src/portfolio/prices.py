"""FMP-backed EOD price fetcher for the cmmd-backtest universe.

Implements requirements 5.1, 5.7, 7.2, 9.2 of the cmmd-backtest spec:
fetch end-of-day close prices for the four-ticker universe (SWDA.L, XLK,
IAU, BIL) from the Financial Modeling Prep ``historical-price-eod/light``
endpoint, align them with an inner-join on `date`, and surface a single
``pandas.DataFrame`` indexed by date with one column per ticker.

The module exposes two public symbols:

- ``PriceFetchError`` -- raised on HTTP failure or insufficient overlap.
- ``fetch_universe_prices`` -- the single service entry point.

This is the only place in the ``portfolio`` layer that performs HTTP I/O
(``requests.get``); tests mock that call. The retry / API-key resolution
pattern matches ``src.dataset.fmp_corpora.fetch_articles``, but the helper
is duplicated locally rather than imported because the sentrux portfolio
↔ dataset boundary forbids that import direction.
"""

from __future__ import annotations

import os
from datetime import date

import pandas as pd
import requests

# Minimum number of aligned trading days the inner-joined frame must
# contain. Below this, the joined sample is too small for the backtest
# layer to do anything meaningful with, so we surface the failure here.
_MIN_OVERLAP_DAYS = 30


class PriceFetchError(RuntimeError):
    """Raised when an FMP price fetch fails or returns insufficient data.

    Carries either the offending ticker plus HTTP status code (transport
    failure) or the offending ticker plus aligned-day count (overlap
    failure). The orchestrator script presents this directly to stderr.
    """


def _resolve_api_key(api_key: str | None) -> str:
    """Resolve an FMP API key from the explicit arg or the environment.

    Mirrors ``src.dataset.fmp_corpora._resolve_api_key`` byte-for-byte —
    duplicated rather than imported because the sentrux portfolio ↔
    dataset boundary forbids that import direction.
    """
    if api_key:
        return api_key
    env_key = os.environ.get("FMP_API_KEY")
    if not env_key:
        raise RuntimeError(
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
    url = (
        "https://financialmodelingprep.com/stable/historical-price-eod/light"
        f"?symbol={ticker}&apikey={api_key}"
    )
    response = requests.get(url, timeout=30)
    if response.status_code != 200:
        raise PriceFetchError(
            f"FMP price fetch for {ticker!r} returned HTTP {response.status_code}"
        )
    payload = response.json()
    if not isinstance(payload, list):
        # Defensive: FMP's documented shape is a top-level list. An
        # unexpected shape means we can't trust the response, so fail
        # loudly rather than silently emitting an empty series.
        raise PriceFetchError(
            f"FMP price fetch for {ticker!r} returned non-list payload"
        )

    rows: list[tuple[date, float]] = []
    for entry in payload:
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
        # Empty series with the right name; the inner-join will then
        # collapse the joined frame to zero rows and the caller's
        # min-overlap check will name this ticker.
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

    # Inner-join all per-ticker series so any date missing from any
    # ticker is dropped from the combined frame.
    joined = pd.concat(
        [series_by_ticker[t] for t in tickers],
        axis=1,
        join="inner",
    )
    # Preserve input column order explicitly (concat already does this
    # but we re-assign defensively in case any series was empty).
    joined.columns = list(tickers)

    # Drop any residual NaN rows: a date where one ticker was reported
    # but the price parsed to NaN would still survive concat's "inner"
    # on the index. Inner-join uses the index, not values, so an
    # explicit dropna is the simplest way to honour the "no NaN cells"
    # postcondition.
    joined = joined.dropna(how="any")

    if len(joined) < _MIN_OVERLAP_DAYS:
        # Identify the ticker(s) that contributed too few raw observations
        # so the error message is actionable.
        per_ticker_counts = {
            t: len(series_by_ticker[t]) for t in tickers
        }
        # Pick the worst offender for the message: the ticker with the
        # fewest raw rows is the one most likely responsible for the
        # joined frame being too small.
        worst_ticker = min(per_ticker_counts, key=per_ticker_counts.get)
        worst_n = per_ticker_counts[worst_ticker]
        raise PriceFetchError(
            f"ticker {worst_ticker!r} has only {worst_n} raw rows "
            f"({len(joined)} aligned trading days after inner-join) "
            f"in window {start.isoformat()}..{end.isoformat()} "
            f"(need >= {_MIN_OVERLAP_DAYS})"
        )

    return joined
