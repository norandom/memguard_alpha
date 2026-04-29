"""Tests for src.portfolio.prices: FMP EOD price fetch for the universe.

Covers requirements 5.1, 5.7, 7.2, 9.2 of cmmd-backtest, task 2.1.

All HTTP traffic is mocked via ``pytest-mock`` patching
``src.portfolio.prices.requests.get``; no test in this module should make
a real outbound call.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd
import pytest

from src.portfolio.prices import PriceFetchError, fetch_universe_prices

# ---------- helpers ----------------------------------------------------------


def _trading_days(start: date, n: int) -> list[date]:
    """Return ``n`` consecutive calendar days starting at ``start``.

    Calendar (not business) days are fine: the inner-join logic only cares
    that all tickers report the same date strings, so a clean run of N
    consecutive ISO dates is the simplest reproducible fixture.
    """
    return [start + timedelta(days=i) for i in range(n)]


def _eod_payload(days: list[date], start_price: float = 100.0) -> list[dict[str, Any]]:
    """Synthetic FMP `historical-price-eod/light` payload.

    Each entry has the documented ``date`` (ISO `YYYY-MM-DD`) and
    ``price`` keys; we walk the price up by 1.0 per day so every column
    of the resulting frame is distinguishable.
    """
    return [
        {"date": d.isoformat(), "price": start_price + float(i)}
        for i, d in enumerate(days)
    ]


def _mock_get_factory(mocker, ticker_to_payload: dict[str, list[dict[str, Any]]]):
    """Patch the module-level ``requests.get`` so each ticker URL gets its payload.

    We patch ``src.portfolio.prices.requests.get`` (NOT the global
    ``requests`` package) so the mock applies inside the module under
    test. The mock parses the `?symbol=` query argument out of the URL
    and returns the matching payload as a 200 response.
    """

    class _FakeResponse:
        def __init__(self, payload: list[dict[str, Any]], status_code: int = 200):
            self._payload = payload
            self.status_code = status_code

        def json(self) -> list[dict[str, Any]]:
            return self._payload

    def _fake_get(url: str, *args: Any, **kwargs: Any) -> _FakeResponse:
        # Crudely parse symbol=... out of the URL.
        # FMP URLs look like: https://...stable/historical-price-eod/light?symbol=XLK&apikey=...
        symbol = ""
        for part in url.split("?", 1)[-1].split("&"):
            if part.startswith("symbol="):
                symbol = part[len("symbol="):]
                break
        if symbol not in ticker_to_payload:
            return _FakeResponse([], status_code=404)
        return _FakeResponse(ticker_to_payload[symbol], status_code=200)

    return mocker.patch("src.portfolio.prices.requests.get", side_effect=_fake_get)


# ---------- happy-path -------------------------------------------------------


def test_happy_path_four_ticker_fetch(mocker):
    """4 tickers, identical 60-day windows → 60×4 frame, no NaN, in input order."""
    tickers = ["SWDA.L", "XLK", "IAU", "BIL"]
    start = date(2024, 1, 1)
    days = _trading_days(start, 60)
    payloads = {t: _eod_payload(days, start_price=100.0 + idx)
                for idx, t in enumerate(tickers)}

    _mock_get_factory(mocker, payloads)

    df = fetch_universe_prices(
        tickers=tickers,
        start=start,
        end=start + timedelta(days=59),
        api_key="fake-key",
    )

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == tickers
    assert len(df) == 60
    assert df.isna().sum().sum() == 0
    # Index should be monotonic and align with input dates.
    assert df.index.is_monotonic_increasing


# ---------- inner-join asymmetry --------------------------------------------


def test_inner_join_drops_missing_date(mocker):
    """Three tickers report 60 days; one ticker is missing exactly one day."""
    tickers = ["SWDA.L", "XLK", "IAU", "BIL"]
    start = date(2024, 1, 1)
    full_days = _trading_days(start, 60)
    # IAU is missing day index 30 (one calendar holiday).
    iau_days = [d for i, d in enumerate(full_days) if i != 30]

    payloads = {
        "SWDA.L": _eod_payload(full_days, start_price=200.0),
        "XLK":    _eod_payload(full_days, start_price=300.0),
        "IAU":    _eod_payload(iau_days, start_price=400.0),
        "BIL":    _eod_payload(full_days, start_price=100.0),
    }
    _mock_get_factory(mocker, payloads)

    df = fetch_universe_prices(
        tickers=tickers,
        start=start,
        end=start + timedelta(days=59),
        api_key="fake-key",
    )

    assert list(df.columns) == tickers
    assert len(df) == 59  # one day dropped by inner-join
    assert df.isna().sum().sum() == 0
    # The missing date should not be in the index.
    missing_date = pd.Timestamp(full_days[30])
    assert missing_date not in df.index


# ---------- under-30-days error ---------------------------------------------


def test_under_30_days_raises(mocker):
    """A ticker with only 25 days should trigger PriceFetchError naming the ticker."""
    tickers = ["SWDA.L", "XLK", "IAU", "BIL"]
    start = date(2024, 1, 1)
    full_days = _trading_days(start, 60)
    short_days = full_days[:25]  # IAU only has 25 days

    payloads = {
        "SWDA.L": _eod_payload(full_days),
        "XLK":    _eod_payload(full_days),
        "IAU":    _eod_payload(short_days),
        "BIL":    _eod_payload(full_days),
    }
    _mock_get_factory(mocker, payloads)

    with pytest.raises(PriceFetchError) as excinfo:
        fetch_universe_prices(
            tickers=tickers,
            start=start,
            end=start + timedelta(days=59),
            api_key="fake-key",
        )
    msg = str(excinfo.value)
    assert "IAU" in msg
    assert "30" in msg


# ---------- API-key-missing error -------------------------------------------


def test_api_key_missing_raises(monkeypatch, mocker):
    """No api_key argument and no FMP_API_KEY in env → RuntimeError."""
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    # Mock so we'd notice if it ever called out. It must not.
    mocked = mocker.patch("src.portfolio.prices.requests.get")

    with pytest.raises(RuntimeError) as excinfo:
        fetch_universe_prices(
            tickers=["SWDA.L", "XLK", "IAU", "BIL"],
            start=date(2024, 1, 1),
            end=date(2024, 3, 1),
        )
    assert "FMP_API_KEY" in str(excinfo.value)
    mocked.assert_not_called()


# ---------- HTTP error path -------------------------------------------------


def test_non_200_raises(mocker):
    """A non-200 response surfaces as PriceFetchError with the ticker + status."""

    class _FakeResponse:
        def __init__(self, status_code: int):
            self.status_code = status_code

        def json(self) -> list[dict[str, Any]]:
            return []

    def _fake_get(url: str, *args: Any, **kwargs: Any) -> _FakeResponse:
        # Force a 503 on the first call (any ticker).
        return _FakeResponse(503)

    mocker.patch("src.portfolio.prices.requests.get", side_effect=_fake_get)

    with pytest.raises(PriceFetchError) as excinfo:
        fetch_universe_prices(
            tickers=["SWDA.L"],
            start=date(2024, 1, 1),
            end=date(2024, 3, 1),
            api_key="fake-key",
        )
    assert "503" in str(excinfo.value)
    assert "SWDA.L" in str(excinfo.value)
