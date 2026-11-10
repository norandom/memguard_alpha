"""The pure rules of study item assembly (task 7.1; Requirements 6.5, 6.6).

Macro rows and realized returns from hand-built inputs, FMP and DTB3 body
validation through the injectable transport, the default transport's request
settings and key handling. No test reaches the network.
"""

import json
import logging
import statistics
from dataclasses import astuple
from datetime import date

import pytest
import requests

from recall_guard.dataset.fred import Observation
from recall_guard.dataset.macro_regime import RealizedReturns, is_correct
from scripts.finance_sim import items

AS_OF = date(2016, 1, 31)
FMP_KEY = "Zq8Wm2Lp5Rt7Yx1Nc4Vb6Hj9Kd3Gf0Sa"
FRED_KEY = "f3d7e9a1c2b4d6f8a0e1c3b5d7f9a2c4"


def index_of(year, month):
    return year * 12 + month - 1


def month_date(index, day=1):
    return date(index // 12, index % 12 + 1, day)


def observation(series, day, value, *, revised=False):
    return Observation(series, day, value, AS_OF, None if revised else AS_OF, revised)


def cpi_level(index):
    k = index - index_of(2009, 1)
    return 200.0 + 0.5 * k + 0.3 * (k % 4)


GAP = index_of(2015, 10)  # A month FRED left without a value; the loader skips it.
CPI = [observation("CPIAUCSL", month_date(index), cpi_level(index))
       for index in range(index_of(2009, 1), index_of(2015, 12) + 1) if index != GAP]
# T10Y2Y: a mid-month value that is never a month-end value, and the month-end value on the 28th.
T10Y2Y = [observation("T10Y2Y", month_date(index, day), 9.0 if day == 10 else 0.5 + 0.1 * (index % 7), revised=True)
          for index in range(index_of(2011, 1), index_of(2016, 1) + 1) for day in (10, 28)]
BAA10Y = [observation("BAA10Y", month_date(index, 28), 2.0 + 0.2 * (index % 5))
          for index in range(index_of(2013, 8), index_of(2016, 1) + 1)]


def vectors(**overrides):
    return {"CPIAUCSL": CPI, "T10Y2Y": T10Y2Y, "BAA10Y": BAA10Y} | overrides


# ---------- decision months ----------

def test_decision_months_are_month_ends_with_leap_days():
    months = items.decision_months((2016, 1), (2016, 3))
    assert months == (date(2016, 1, 31), date(2016, 2, 29), date(2016, 3, 31))
    assert items.price_window(months) == (date(2015, 12, 1), date(2016, 4, 30))
    assert items.price_window(items.decision_months()) == (date(2015, 12, 1), date(2026, 7, 31))
    with pytest.raises(ValueError, match="last calendar day"):
        items.classify_cutoffs((date(2016, 1, 30),), ("google/gemma-3-4b-it",))


# ---------- macro rows ----------

def test_macro_row_follows_the_design_rule_on_hand_built_vectors():
    row = items.build_macro_row(AS_OF, vectors())
    # CPI: December 2015 over December 2014. The October 2015 gap must not shift the base to November 2014.
    december_2015, december_2014 = index_of(2015, 12), index_of(2014, 12)
    assert row.raw["cpi_yoy"] == pytest.approx((cpi_level(december_2015) / cpi_level(december_2014) - 1) * 100,
                                               rel=1e-12)
    yoy = {index: (cpi_level(index) / cpi_level(index - 12) - 1) * 100
           for index in range(december_2015 - 59, december_2015 + 1) if GAP not in (index, index - 12)}
    assert len(yoy) == 59
    window = list(yoy.values())
    assert row.zscores["cpi_yoy_z"] == pytest.approx(
        (row.raw["cpi_yoy"] - statistics.fmean(window)) / statistics.stdev(window), rel=1e-12)

    # Daily series: the latest value; the window is the last value of each of the 60 months, the 10th ignored.
    t10y2y_window = [0.5 + 0.1 * (index % 7) for index in range(index_of(2011, 2), index_of(2016, 1) + 1)]
    assert row.raw["t10y2y"] == t10y2y_window[-1]
    assert row.zscores["t10y2y_z"] == pytest.approx(
        (t10y2y_window[-1] - statistics.fmean(t10y2y_window)) / statistics.stdev(t10y2y_window), rel=1e-12)
    baa_window = [2.0 + 0.2 * (index % 5) for index in range(index_of(2013, 8), index_of(2016, 1) + 1)]
    assert len(baa_window) == 30
    assert row.zscores["baa10y_z"] == pytest.approx(
        (baa_window[-1] - statistics.fmean(baa_window)) / statistics.stdev(baa_window), rel=1e-12)
    assert row.revised == {"BAA10Y": False, "CPIAUCSL": False, "T10Y2Y": True}


@pytest.mark.parametrize(("override", "reason"), [
    ({"BAA10Y": BAA10Y[1:]}, "BAA10Y has 29 monthly values in the 60 months to 2016-01; its z-score needs at least 30"),
    ({"BAA10Y": [observation("BAA10Y", item.observation_date, 2.5) for item in BAA10Y]}, "BAA10Y does not vary"),
    ({"T10Y2Y": []}, "T10Y2Y has no observation available on 2016-01-31"),
    ({"CPIAUCSL": [item for item in CPI if item.observation_date != month_date(index_of(2014, 12))]},
     "CPIAUCSL has no level for 2014-12, twelve months before its latest month 2015-12"),
])
def test_an_uncomputable_month_raises_excluded_month_with_the_reason(override, reason):
    with pytest.raises(items.ExcludedMonth) as raised:
        items.build_macro_row(AS_OF, vectors(**override))
    assert reason in raised.value.reason


def test_vectors_from_another_decision_date_are_refused():
    other = [Observation("BAA10Y", item.observation_date, item.value, date(2016, 2, 29), date(2016, 2, 29), False)
             for item in BAA10Y]
    with pytest.raises(ValueError, match="as of 2016-01-31"):
        items.build_macro_row(AS_OF, vectors(BAA10Y=other))


# ---------- realized returns ----------

CLOSES = {"world_equity": ((date(2016, 1, 29), 100.0), (date(2016, 2, 29), 110.0)),
          "tech_sector": ((date(2016, 1, 29), 200.0), (date(2016, 2, 26), 190.0)),
          "gold_commodity": ((date(2016, 1, 28), 50.0), (date(2016, 2, 29), 51.0))}
CASH = ((date(2016, 1, 29), 1.2),)


def test_realized_returns_follow_the_design_formula():
    (outcome,) = items.realized_returns((AS_OF,), CLOSES, CASH, complete_through=date(2016, 3, 1))
    assert isinstance(outcome.realized, RealizedReturns)
    assert astuple(outcome.realized) == pytest.approx((0.10, -0.05, 0.02, 0.001), rel=1e-12)
    assert outcome.reason is None
    # Defensive minus risk is (0.02 + 0.001) - (0.10 - 0.05) < 0, so a risk-on spread is correct.
    assert is_correct(-0.3, outcome.realized) and not is_correct(0.3, outcome.realized)


@pytest.mark.parametrize(("closes", "cash", "complete_through", "reason"), [
    (CLOSES, CASH, date(2016, 2, 28), "the next month 2016-02 is not complete: data through 2016-02-28"),
    (CLOSES | {"gold_commodity": ((date(2016, 1, 20), 50.0), (date(2016, 2, 29), 51.0))}, CASH, date(2016, 3, 1),
     "IAU has no adjusted close in the last 7 days of 2016-01"),
    (CLOSES | {"tech_sector": ((date(2016, 1, 29), 200.0),)}, CASH, date(2016, 3, 1),
     "XLK has no adjusted close in the last 7 days of 2016-02"),
    (CLOSES, ((date(2015, 12, 31), 0.2),), date(2016, 3, 1), "DTB3 has no value in the last 7 days of 2016-01"),
])
def test_missing_outcome_data_keeps_realized_none_with_the_reason(closes, cash, complete_through, reason):
    (outcome,) = items.realized_returns((AS_OF,), closes, cash, complete_through=complete_through)
    assert outcome.realized is None and outcome.reason == reason
    assert outcome.to_dict()["realized"] is None


def test_realized_returns_need_every_risky_category():
    with pytest.raises(ValueError, match="exactly the categories"):
        items.realized_returns((AS_OF,), {"world_equity": CLOSES["world_equity"]}, CASH,
                               complete_through=date(2016, 3, 1))


# ---------- FMP and DTB3 bodies ----------

class Transport:
    def __init__(self, status, body):
        self.status, self.body = status, body
        self.calls = 0

    def __call__(self, url, params, timeout_s):
        self.calls += 1
        return self.status, self.body if isinstance(self.body, bytes) else json.dumps(self.body).encode()


def fetch_xlk(tmp_path, transport, **overrides):
    arguments = {"start": date(2016, 1, 1), "end": date(2016, 2, 29), "api_key": FMP_KEY,
                 "cache_dir": tmp_path / "fmp", "transport": transport}
    return items.fetch_adjusted_closes("XLK", **(arguments | overrides))


def row(day, close, symbol="XLK"):
    return {"symbol": symbol, "date": day, "adjOpen": close, "adjHigh": close, "adjLow": close, "adjClose": close,
            "volume": 10}


def test_adjusted_closes_are_sorted_and_cached(tmp_path):
    transport = Transport(200, [row("2016-02-29", 41.82), row("2016-01-29", 41.0)])
    closes = fetch_xlk(tmp_path, transport)
    assert closes == ((date(2016, 1, 29), 41.0), (date(2016, 2, 29), 41.82))
    assert fetch_xlk(tmp_path, transport) == closes and transport.calls == 1
    assert [path.name for path in (tmp_path / "fmp").iterdir()] == ["XLK_2016-01-01_2016-02-29.json"]


@pytest.mark.parametrize(("body", "message"), [
    ({"Error Message": "Invalid API KEY. Please retry."}, "must be a list of rows: Invalid API KEY"),
    ([], "no adjusted closes for XLK"),
    ([row("2016-01-29", 41.0), row("2016-01-29", 41.5)], "duplicate XLK closes for 2016-01-29"),
    ([row("2016-03-01", 41.0)], "outside 2016-01-01 to 2016-02-29"),
    ([row("2016-01-29", 0)], "adjClose for 2016-01-29 must be a positive finite number"),
    ([row("2016-01-29", None)], "adjClose for 2016-01-29 must be a positive finite number"),
    ([row("2016-01-29", 41.0, symbol="IAU")], "another symbol"),
    ([{"symbol": "XLK", "date": "29/01/2016", "adjClose": 41.0}], "must be an ISO date"),
])
def test_an_invalid_fmp_body_is_refused_and_never_cached(tmp_path, body, message):
    with pytest.raises(ValueError, match=message):
        fetch_xlk(tmp_path, Transport(200, body))
    assert not (tmp_path / "fmp").exists()


def test_a_corrupt_cached_body_is_refused_instead_of_refetched(tmp_path):
    cache = tmp_path / "fmp"
    cache.mkdir()
    (cache / "XLK_2016-01-01_2016-02-29.json").write_text("{truncated", encoding="utf-8")
    transport = Transport(200, [row("2016-01-29", 41.0)])
    with pytest.raises(ValueError, match="cached FMP adjusted closes for XLK body .* is not JSON"):
        fetch_xlk(tmp_path, transport)
    assert transport.calls == 0


def test_a_malformed_key_is_refused_before_any_request_without_echoing_it(tmp_path):
    transport = Transport(200, [])
    with pytest.raises(ValueError) as raised:
        fetch_xlk(tmp_path, transport, api_key="not a key!")
    assert "not a key!" not in str(raised.value) and transport.calls == 0


def test_cash_rates_skip_missing_values_and_refuse_a_truncated_list(tmp_path):
    body = {"count": 3, "observations": [{"date": "2016-01-28", "value": "0.31"}, {"date": "2016-01-29", "value": "."},
                                         {"date": "2016-02-01", "value": "0.30"}]}
    rates = items.fetch_cash_rates(start=date(2016, 1, 1), end=date(2016, 2, 29), api_key=FRED_KEY,
                                   cache_dir=tmp_path / "fred", transport=Transport(200, body))
    assert rates == ((date(2016, 1, 28), 0.31), (date(2016, 2, 1), 0.30))
    with pytest.raises(ValueError, match="truncated DTB3"):
        items.fetch_cash_rates(start=date(2016, 1, 1), end=date(2016, 2, 29), api_key=FRED_KEY,
                               cache_dir=tmp_path / "other", transport=Transport(200, body | {"count": 4}))


# ---------- default transport and keys ----------

def test_the_default_transport_sends_one_get_without_retries_redirects_or_ambient_proxies(monkeypatch):
    seen = []

    def send(adapter, prepared, **kwargs):
        seen.append({"method": prepared.method, "url": prepared.url, "retries": adapter.max_retries.total,
                     "proxies": kwargs.get("proxies"), "timeout": kwargs.get("timeout")})
        response = requests.Response()
        response.status_code = 302
        response._content = b"moved"
        response.headers["Location"] = "https://elsewhere.example/"
        response.request = prepared
        response.url = prepared.url
        return response

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    assert items.requests_transport(items.FMP_URL, {"symbol": "XLK", "apikey": FMP_KEY}, 12.5) == (302, b"moved")
    assert len(seen) == 1
    assert (seen[0]["method"], seen[0]["retries"], seen[0]["timeout"]) == ("GET", 0, 12.5)
    assert "127.0.0.1:9" not in repr(seen[0]["proxies"])
    assert seen[0]["url"].startswith(items.FMP_URL + "?symbol=XLK&apikey=")


def test_urllib3_request_lines_never_carry_the_fmp_key():
    logger = logging.getLogger("urllib3.connectionpool")
    record = logger.makeRecord(logger.name, logging.DEBUG, __file__, 1, '%s://%s:%s "%s %s %s" %s %s',
                               ("https", "financialmodelingprep.com", 443, "GET",
                                f"/stable/historical-price-eod/dividend-adjusted?symbol=XLK&apikey={FMP_KEY}",
                                "HTTP/1.1", 200, 10), None)
    assert all(log_filter.filter(record) for log_filter in logger.filters)
    assert FMP_KEY not in record.getMessage() and "apikey=[redacted]" in record.getMessage()


def test_keys_come_from_the_process_environment_by_name_only():
    assert items.api_key_from_environment("FMP_API_TOKEN", {"FMP_API_TOKEN": FMP_KEY}) == FMP_KEY
    with pytest.raises(ValueError, match="FRED_API_KEY is not set in the process environment"):
        items.api_key_from_environment("FRED_API_KEY", {"FMP_API_TOKEN": FMP_KEY})
    with pytest.raises(ValueError, match="must be FRED_API_KEY or FMP_API_TOKEN"):
        items.api_key_from_environment("FMP_API_KEY", {"FMP_API_KEY": FMP_KEY})
