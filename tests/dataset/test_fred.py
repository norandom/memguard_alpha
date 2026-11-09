"""Point-in-time FRED loader (Requirement 6.5).

Fixture bodies are recorded in the FRED JSON API shape (``series/observations``
and ``series/vintagedates``); the values are illustrative. They are served by a
fake transport that replaces ``requests.adapters.HTTPAdapter.send``, so no test
reaches the network. The log test alone uses a loopback server on 127.0.0.1,
because it needs urllib3 to emit its own request line.
"""

import hashlib
import json
import logging
import threading
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from datetime import date, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

import pytest
import requests

from recall_guard.core.request_control import RequestError
from recall_guard.dataset import fred
from recall_guard.dataset.fred import (
    DEFAULT_CACHE_DIR,
    MAX_REQUESTS_PER_MINUTE,
    Observation,
    RollingRateLimiter,
    fetch_point_in_time,
    fetch_series_as_of,
    vintage_available,
)

KEY = "f3d7e9a1c2b4d6f8a0e1c3b5d7f9a2c4"
MODULE_LIMITER = fred.RATE_LIMITER
CPI_AS_OF = date(2016, 2, 1)
T10Y2Y_AS_OF = date(2013, 6, 3)
FIRST_VINTAGE = {"CPIAUCSL": "1972-07-21", "T10Y2Y": "2014-01-27", "BAA10Y": "2014-01-27", "DTB3": "2005-06-28"}
TODAY = "2026-11-09"

# CPIAUCSL as published on 2016-02-01: December 2015 is the latest month.
CPI_VINTAGE_ROWS = [("2015-10-01", "237.733"), ("2015-11-01", "238.017"), ("2015-12-01", "237.761")]
# The current vintage: revised seasonal factors, plus January 2016, published on 2016-02-19.
CPI_CURRENT_ROWS = [("2015-10-01", "237.838"), ("2015-11-01", "237.336"), ("2015-12-01", "236.525"),
                    ("2016-01-01", "236.916")]
T10Y2Y_CURRENT_ROWS = [("2013-05-29", "1.81"), ("2013-05-30", "1.79"), ("2013-05-31", "1.86"),
                       ("2013-06-03", "1.84")]


def observations_body(rows, *, realtime, observation_end):
    """A ``series/observations`` body; every row carries the requested real-time period."""
    return {
        "realtime_start": realtime, "realtime_end": realtime,
        "observation_start": "1600-01-01", "observation_end": observation_end,
        "units": "lin", "output_type": 1, "file_type": "json",
        "order_by": "observation_date", "sort_order": "asc",
        "count": len(rows), "offset": 0, "limit": 100000,
        "observations": [{"realtime_start": realtime, "realtime_end": realtime, "date": day, "value": value}
                         for day, value in rows],
    }


def vintage_dates_body(*dates):
    return {"realtime_start": "1776-07-04", "realtime_end": "9999-12-31", "order_by": "vintage_date",
            "sort_order": "asc", "count": 670, "offset": 0, "limit": 1, "vintage_dates": list(dates)}


def recorded_fred(endpoint, params):
    """Route a request to its recorded body, as FRED answers it."""
    series = params["series_id"]
    if endpoint == "vintagedates":
        return 200, vintage_dates_body(FIRST_VINTAGE[series])
    point_in_time = "realtime_start" in params
    rows = {("CPIAUCSL", True): CPI_VINTAGE_ROWS, ("CPIAUCSL", False): CPI_CURRENT_ROWS,
            ("T10Y2Y", False): T10Y2Y_CURRENT_ROWS}[series, point_in_time]
    end = params["observation_end"]
    return 200, observations_body([row for row in rows if row[0] <= end],
                                  realtime=params["realtime_start"] if point_in_time else TODAY,
                                  observation_end=end)


class FakeFred:
    """Answer each prepared request from ``routes`` and record what was sent."""

    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def answer(self, adapter, prepared, kwargs):
        url = urlsplit(prepared.url)
        params = dict(parse_qsl(url.query, keep_blank_values=True))
        endpoint = url.path.rsplit("/", 1)[-1]
        self.calls.append({"method": prepared.method, "host": url.hostname, "path": url.path, "endpoint": endpoint,
                           "params": params, "timeout": kwargs.get("timeout"), "retries": adapter.max_retries.total})
        status, body, *headers = self.routes(endpoint, params)
        response = requests.Response()
        response.status_code = status
        response._content = body if isinstance(body, bytes) else json.dumps(body).encode()
        response._content_consumed = True
        response.headers.update(headers[0] if headers else {})
        response.request = prepared
        response.url = prepared.url
        return response


@pytest.fixture
def serve(monkeypatch):
    def install(routes=recorded_fred):
        fake = FakeFred(routes)

        def send(adapter, prepared, **kwargs):
            return fake.answer(adapter, prepared, kwargs)

        monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)
        return fake

    return install


class FakeClock:
    """Monotonic seconds that only move when the limiter sleeps or a test advances them."""

    def __init__(self):
        self.now = 1000.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    """Every test runs with an injected clock and sleep, so nothing ever waits."""
    fake = FakeClock()
    monkeypatch.setattr(fred, "RATE_LIMITER", RollingRateLimiter(clock=fake, sleep=fake.sleep))
    return fake


def cache_name(series_id, slot, params):
    canonical = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return f"{series_id}_{slot}_{hashlib.sha256(canonical.encode('utf-8')).hexdigest()[:16]}.json"


def assert_key_absent(cache_dir: Path):
    for path in cache_dir.rglob("*"):
        assert KEY not in str(path)
        if path.is_file():
            assert KEY.encode() not in path.read_bytes()


def assert_no_key(error):
    for text in (str(error), repr(error), error.detail or "", repr(error.detail), repr(error.args)):
        assert KEY not in text
    assert error.__cause__ is None and error.__context__ is None


def test_revised_cpi_date_returns_the_vintage_value_as_of_the_decision_date(serve, tmp_path):
    fake = serve()
    observation = fetch_point_in_time("CPIAUCSL", CPI_AS_OF, api_key=KEY, cache_dir=tmp_path)
    # The revised 2015-12 value is 236.525 and January 2016 was published later; neither may leak.
    assert observation == Observation(series_id="CPIAUCSL", observation_date=date(2015, 12, 1), value=237.761,
                                      as_of=CPI_AS_OF, vintage_start=CPI_AS_OF, revised_data=False)
    assert [call["endpoint"] for call in fake.calls] == ["vintagedates", "observations"]
    assert {call["host"] for call in fake.calls} == {"api.stlouisfed.org"}
    assert [call["path"] for call in fake.calls] == ["/fred/series/vintagedates", "/fred/series/observations"]
    assert fake.calls[0]["params"] == {"series_id": "CPIAUCSL", "file_type": "json", "sort_order": "asc",
                                       "limit": "1", "api_key": KEY}
    assert fake.calls[1]["params"] == {"series_id": "CPIAUCSL", "file_type": "json", "sort_order": "asc",
                                       "realtime_start": "2016-02-01", "realtime_end": "2016-02-01",
                                       "observation_end": "2016-02-01", "api_key": KEY}
    assert all(call["method"] == "GET" for call in fake.calls)


def test_series_as_of_returns_every_vintage_row_on_or_before_the_decision_date(serve, tmp_path):
    serve()
    observations = fetch_series_as_of("CPIAUCSL", CPI_AS_OF, api_key=KEY, cache_dir=tmp_path)
    assert [(o.observation_date.isoformat(), o.value) for o in observations] == [
        (day, float(value)) for day, value in CPI_VINTAGE_ROWS]
    assert all(not o.revised_data and o.vintage_start == CPI_AS_OF and o.as_of == CPI_AS_OF for o in observations)
    with pytest.raises(FrozenInstanceError):
        observations[0].value = 1.0


def test_series_whose_earliest_vintage_is_after_the_decision_date_is_flagged_revised(serve, tmp_path):
    fake = serve()
    assert vintage_available("T10Y2Y", date(2014, 1, 27), api_key=KEY, cache_dir=tmp_path)
    assert not vintage_available("T10Y2Y", T10Y2Y_AS_OF, api_key=KEY, cache_dir=tmp_path)
    observations = fetch_series_as_of("T10Y2Y", T10Y2Y_AS_OF, api_key=KEY, cache_dir=tmp_path)
    assert [o.value for o in observations] == [1.81, 1.79, 1.86, 1.84]
    assert all(o.revised_data and o.vintage_start is None for o in observations)
    # One vintagedates request serves every decision date; the current vintage carries no real-time period.
    assert [call["endpoint"] for call in fake.calls] == ["vintagedates", "observations"]
    assert fake.calls[1]["params"] == {"series_id": "T10Y2Y", "file_type": "json", "sort_order": "asc",
                                       "observation_end": "2013-06-03", "api_key": KEY}


def test_series_without_any_vintage_is_flagged_revised(serve, tmp_path):
    def routes(endpoint, params):
        if endpoint == "vintagedates":
            return 200, vintage_dates_body() | {"count": 0}
        return recorded_fred(endpoint, params)

    serve(routes)
    assert not vintage_available("T10Y2Y", date(2026, 1, 2), api_key=KEY, cache_dir=tmp_path)
    observation = fetch_point_in_time("T10Y2Y", T10Y2Y_AS_OF, api_key=KEY, cache_dir=tmp_path)
    assert observation.revised_data and observation.vintage_start is None


def test_cache_files_land_under_cache_dir_with_the_designed_names_and_a_second_call_makes_no_request(
        serve, tmp_path, monkeypatch):
    fake = serve()
    working = tmp_path / "working"
    working.mkdir()
    monkeypatch.chdir(working)
    cache_dir = tmp_path / "runs" / "finance-sim" / "fred"
    first = fetch_series_as_of("CPIAUCSL", CPI_AS_OF, api_key=KEY, cache_dir=cache_dir)
    vintage_params = {"series_id": "CPIAUCSL", "file_type": "json", "sort_order": "asc", "limit": "1"}
    observation_params = {"series_id": "CPIAUCSL", "file_type": "json", "sort_order": "asc",
                          "realtime_start": "2016-02-01", "realtime_end": "2016-02-01",
                          "observation_end": "2016-02-01"}
    assert sorted(path.name for path in cache_dir.iterdir()) == sorted([
        cache_name("CPIAUCSL", "vintagedates", vintage_params),
        cache_name("CPIAUCSL", "2016-02-01", observation_params)])
    stored = json.loads((cache_dir / cache_name("CPIAUCSL", "2016-02-01", observation_params)).read_text("utf-8"))
    assert stored == recorded_fred("observations", observation_params)[1]
    assert_key_absent(tmp_path)
    assert list(working.iterdir()) == []

    requests_made = len(fake.calls)
    assert fetch_series_as_of("CPIAUCSL", CPI_AS_OF, api_key=KEY, cache_dir=cache_dir) == first
    assert vintage_available("CPIAUCSL", date(2020, 1, 1), api_key=KEY, cache_dir=cache_dir)
    assert len(fake.calls) == requests_made == 2
    # Another cache location is another cache.
    fetch_series_as_of("CPIAUCSL", CPI_AS_OF, api_key=KEY, cache_dir=tmp_path / "elsewhere")
    assert len(fake.calls) == 4


def test_a_key_echoed_in_a_response_body_is_not_cached(serve, tmp_path):
    def routes(endpoint, params):
        status, body = recorded_fred(endpoint, params)
        return status, body | {"note": f"request api_key={params['api_key']}"}

    serve(routes)
    fetch_series_as_of("CPIAUCSL", CPI_AS_OF, api_key=KEY, cache_dir=tmp_path)
    assert len(list(tmp_path.iterdir())) == 2
    assert_key_absent(tmp_path)


def test_default_cache_dir_is_the_private_runs_folder():
    assert DEFAULT_CACHE_DIR == Path("runs", "finance-sim", "fred")
    assert not DEFAULT_CACHE_DIR.is_absolute()


def test_missing_values_are_skipped(serve, tmp_path):
    rows = [("2013-05-29", "1.81"), ("2013-05-30", "."), ("2013-05-31", "1.86"), ("2013-06-03", ".")]

    def routes(endpoint, params):
        if endpoint == "vintagedates":
            return recorded_fred(endpoint, params)
        return 200, observations_body(rows, realtime=TODAY, observation_end=params["observation_end"])

    serve(routes)
    observations = fetch_series_as_of("T10Y2Y", T10Y2Y_AS_OF, api_key=KEY, cache_dir=tmp_path)
    assert [(o.observation_date, o.value) for o in observations] == [(date(2013, 5, 29), 1.81),
                                                                     (date(2013, 5, 31), 1.86)]
    assert fetch_point_in_time("T10Y2Y", T10Y2Y_AS_OF, api_key=KEY, cache_dir=tmp_path).observation_date == date(
        2013, 5, 31)


def test_point_in_time_is_the_last_observation_or_none(serve, tmp_path):
    def routes(endpoint, params):
        if endpoint == "vintagedates":
            return recorded_fred(endpoint, params)
        rows = {"DTB3": [], "T10Y2Y": [("2013-05-31", ".")]}[params["series_id"]]
        return 200, observations_body(rows, realtime=params["realtime_start"], observation_end=params["observation_end"])

    serve(routes)
    assert fetch_series_as_of("DTB3", date(2006, 1, 2), api_key=KEY, cache_dir=tmp_path) == ()
    assert fetch_point_in_time("DTB3", date(2006, 1, 2), api_key=KEY, cache_dir=tmp_path) is None
    assert fetch_point_in_time("T10Y2Y", date(2015, 1, 2), api_key=KEY, cache_dir=tmp_path) is None


@pytest.mark.parametrize("value", ["nan", "inf", "-Infinity", "1e999", "n/a", ""])
def test_non_finite_or_malformed_values_are_refused_and_not_cached(serve, tmp_path, value):
    def routes(endpoint, params):
        if endpoint == "vintagedates":
            return recorded_fred(endpoint, params)
        return 200, observations_body([("2015-12-01", value)], realtime=params["realtime_start"],
                                      observation_end=params["observation_end"])

    serve(routes)
    with pytest.raises(ValueError, match="finite"):
        fetch_series_as_of("CPIAUCSL", CPI_AS_OF, api_key=KEY, cache_dir=tmp_path)
    assert [path.name.split("_")[1] for path in tmp_path.iterdir()] == ["vintagedates"]


@pytest.mark.parametrize("row", [
    {"realtime_start": "2016-02-01", "realtime_end": "2016-02-01", "date": "2016-02-02", "value": "1.0"},
    {"realtime_start": "2016-02-02", "realtime_end": "2016-02-29", "date": "2015-12-01", "value": "1.0"},
    {"realtime_start": "2016-01-20", "realtime_end": "2016-01-31", "date": "2015-12-01", "value": "1.0"},
    {"realtime_start": "2016-02-01", "realtime_end": "2016-02-01", "date": "20151201", "value": "1.0"},
])
def test_rows_outside_the_decision_date_are_refused(serve, tmp_path, row):
    def routes(endpoint, params):
        if endpoint == "vintagedates":
            return recorded_fred(endpoint, params)
        return 200, observations_body([], realtime=CPI_AS_OF.isoformat(), observation_end="2016-02-01") | {
            "count": 1, "observations": [row]}

    serve(routes)
    with pytest.raises(ValueError):
        fetch_series_as_of("CPIAUCSL", CPI_AS_OF, api_key=KEY, cache_dir=tmp_path)


def test_truncated_or_duplicated_observations_are_refused(serve, tmp_path):
    bodies = [observations_body(CPI_VINTAGE_ROWS, realtime="2016-02-01", observation_end="2016-02-01") | {"count": 9},
              observations_body(CPI_VINTAGE_ROWS + CPI_VINTAGE_ROWS[-1:], realtime="2016-02-01",
                                observation_end="2016-02-01")]

    def routes(endpoint, params):
        if endpoint == "vintagedates":
            return recorded_fred(endpoint, params)
        return 200, bodies.pop(0)

    serve(routes)
    with pytest.raises(ValueError, match="truncated"):
        fetch_series_as_of("CPIAUCSL", CPI_AS_OF, api_key=KEY, cache_dir=tmp_path)
    with pytest.raises(ValueError, match="duplicate"):
        fetch_series_as_of("CPIAUCSL", CPI_AS_OF, api_key=KEY, cache_dir=tmp_path)


def test_rolling_limiter_blocks_the_121st_request_within_a_minute(clock):
    limiter = RollingRateLimiter(clock=clock, sleep=clock.sleep)
    for _ in range(MAX_REQUESTS_PER_MINUTE):
        limiter.acquire()
        clock.now += 0.25  # 120 starts over 30 seconds
    assert clock.sleeps == []
    assert limiter.acquire() == pytest.approx(30.0)  # the first start leaves the window at 1060.0
    assert clock.sleeps == [pytest.approx(30.0)] and clock.now == pytest.approx(1060.0)
    clock.now += 0.25
    assert limiter.acquire() == pytest.approx(0.0)  # the second start (1000.25) has just left the window
    assert len(clock.sleeps) == 1


def test_rolling_limiter_admits_after_the_window_without_sleeping(clock):
    limiter = RollingRateLimiter(3, 60.0, clock=clock, sleep=clock.sleep)
    for _ in range(3):
        limiter.acquire()
    clock.now += 60.0
    assert limiter.acquire() == 0.0
    assert clock.sleeps == []


def test_rolling_limiter_records_the_actual_wakeup_so_a_late_wakeup_does_not_compress_later_starts(clock):
    def late_sleep(seconds):
        clock.sleeps.append(seconds)
        clock.now += seconds + 5.0

    limiter = RollingRateLimiter(2, 60.0, clock=clock, sleep=late_sleep)
    limiter.acquire()
    limiter.acquire()
    assert limiter.acquire() == pytest.approx(60.0)  # wakes at 1065.0, recorded at 1065.0
    assert limiter.acquire() == 0.0
    assert limiter.acquire() == pytest.approx(60.0)  # both 1065.0 starts are in the window until 1125.0
    assert clock.now == pytest.approx(1130.0)


def test_rolling_limiter_never_records_a_start_before_its_turn(clock):
    """A sleep that returns early (clock not advanced) is recorded at the window edge."""
    limiter = RollingRateLimiter(1, 60.0, clock=clock, sleep=clock.sleeps.append)
    limiter.acquire()
    assert limiter.acquire() == pytest.approx(60.0)  # recorded at 1060.0, not 1000.0
    assert limiter.acquire() == pytest.approx(120.0)  # so the next turn is 1120.0
    assert clock.sleeps == [pytest.approx(60.0), pytest.approx(120.0)]


@pytest.mark.parametrize("args", [(0, 60.0), (True, 60.0), (120, 0.0), (120, float("nan")), (1.5, 60.0)])
def test_rolling_limiter_configuration_is_validated(args):
    with pytest.raises((TypeError, ValueError)):
        RollingRateLimiter(*args)


def test_module_limiter_admits_120_requests_per_rolling_minute():
    assert MODULE_LIMITER.max_requests == MAX_REQUESTS_PER_MINUTE == 120
    assert MODULE_LIMITER.period_s == 60.0


def test_the_loader_waits_for_the_limiter_before_the_121st_request(serve, tmp_path, clock):
    def routes(endpoint, params):
        return 200, vintage_dates_body("2000-01-03")

    fake = serve(routes)
    for number in range(MAX_REQUESTS_PER_MINUTE + 1):
        vintage_available(f"S{number}", date(2010, 1, 4), api_key=KEY, cache_dir=tmp_path)
    assert len(fake.calls) == 121
    assert clock.sleeps == [pytest.approx(60.0)]
    # Cache hits are not requests and never wait.
    vintage_available("S0", date(2010, 1, 4), api_key=KEY, cache_dir=tmp_path)
    assert len(fake.calls) == 121 and len(clock.sleeps) == 1


def test_requests_disable_retries_and_use_an_explicit_timeout(serve, tmp_path):
    fake = serve()
    fetch_series_as_of("CPIAUCSL", CPI_AS_OF, api_key=KEY, cache_dir=tmp_path)
    assert [call["retries"] for call in fake.calls] == [0, 0]
    assert [call["timeout"] for call in fake.calls] == [fred.TIMEOUT_S, fred.TIMEOUT_S]
    assert fred.TIMEOUT_S > 0


@pytest.mark.parametrize(("status", "body", "headers"), [
    (400, {"error_code": 400, "error_message": f"Bad Request.  The value for variable api_key is not registered. {KEY}"},
     {}),
    (429, {"error_code": 429, "error_message": "Too Many Requests.  Exceeded Rate Limit"}, {}),
    (500, b"<html>Internal Server Error</html>", {}),
    (302, b"", {"Location": f"https://example.invalid/fred?api_key={KEY}"}),
])
def test_http_errors_map_to_fred_http_error_without_the_key(serve, tmp_path, status, body, headers):
    fake = serve(lambda endpoint, params: (status, body, headers))
    with pytest.raises(RequestError) as caught:
        fetch_series_as_of("CPIAUCSL", CPI_AS_OF, api_key=KEY, cache_dir=tmp_path)
    error = caught.value
    assert error.code == "fred_http_error" and error.status_code == status
    assert f"HTTP {status}" in str(error)
    assert_no_key(error)
    assert len(fake.calls) == 1
    assert not tmp_path.exists() or list(tmp_path.iterdir()) == []


def test_http_error_detail_keeps_the_fred_message_redacted(serve, tmp_path):
    body = {"error_code": 400, "error_message": f"Bad Request.  Variable api_key={KEY} is not registered."}
    serve(lambda endpoint, params: (400, body))
    with pytest.raises(RequestError) as caught:
        vintage_available("CPIAUCSL", CPI_AS_OF, api_key=KEY, cache_dir=tmp_path)
    assert "is not registered" in caught.value.detail and "[redacted]" in caught.value.detail


def test_transport_failure_maps_to_transport_error_without_the_key(serve, tmp_path):
    def routes(endpoint, params):
        raise requests.ConnectionError(
            "HTTPSConnectionPool(host='api.stlouisfed.org', port=443): Max retries exceeded with url: "
            f"/fred/series/vintagedates?series_id=CPIAUCSL&api_key={params['api_key']} (Caused by NewConnectionError)")

    serve(routes)
    with pytest.raises(RequestError) as caught:
        fetch_point_in_time("CPIAUCSL", CPI_AS_OF, api_key=KEY, cache_dir=tmp_path)
    error = caught.value
    assert error.code == "transport_error" and error.retryable and error.status_code is None
    assert "Max retries exceeded" in error.detail and "[redacted]" in error.detail
    assert_no_key(error)


@pytest.mark.parametrize("api_key", ["", " " + KEY, KEY + "\n", KEY[:8] + "+/=" + KEY[8:], KEY[:8] + "é", None, 7])
def test_unusable_key_is_refused_before_any_request_without_echo(serve, tmp_path, api_key):
    fake = serve()
    with pytest.raises((TypeError, ValueError)) as caught:
        fetch_series_as_of("CPIAUCSL", CPI_AS_OF, api_key=api_key, cache_dir=tmp_path)
    assert fake.calls == []
    if isinstance(api_key, str) and api_key.strip():
        assert api_key.strip() not in str(caught.value)


@pytest.mark.parametrize(("series_id", "as_of"), [
    ("../CPIAUCSL", CPI_AS_OF), ("CPI AUCSL", CPI_AS_OF), ("", CPI_AS_OF), (None, CPI_AS_OF),
    ("CPIAUCSL", datetime(2016, 2, 1)), ("CPIAUCSL", "2016-02-01"),
])
def test_series_id_and_decision_date_are_validated_before_any_request(serve, tmp_path, series_id, as_of):
    fake = serve()
    with pytest.raises((TypeError, ValueError)):
        fetch_series_as_of(series_id, as_of, api_key=KEY, cache_dir=tmp_path)
    assert fake.calls == []


@pytest.mark.parametrize("changes", [
    {"vintage_start": None},
    {"revised_data": True},
    {"value": float("nan")},
    {"value": 10 ** 400},
    {"value": "237.761"},
    {"observation_date": date(2016, 2, 2)},
    {"vintage_start": date(2016, 2, 2)},
    {"as_of": datetime(2016, 2, 1)},
])
def test_observation_fields_are_consistent(changes):
    values = {"series_id": "CPIAUCSL", "observation_date": date(2015, 12, 1), "value": 237.761,
              "as_of": CPI_AS_OF, "vintage_start": CPI_AS_OF, "revised_data": False}
    Observation(**values)
    with pytest.raises((TypeError, ValueError)):
        Observation(**(values | changes))


@contextmanager
def loopback_fred(replies):
    """Serve canned JSON bodies on 127.0.0.1 in order."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            raw = json.dumps(replies.pop(0)).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/fred/series"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(3)


def test_urllib3_debug_request_lines_never_carry_the_key(tmp_path, monkeypatch, caplog):
    replies = [vintage_dates_body("1972-07-21"),
               observations_body(CPI_VINTAGE_ROWS, realtime="2016-02-01", observation_end="2016-02-01")]
    caplog.set_level(logging.DEBUG, logger="urllib3")
    with loopback_fred(replies) as base_url:
        monkeypatch.setattr(fred, "BASE_URL", base_url)
        observation = fetch_point_in_time("CPIAUCSL", CPI_AS_OF, api_key=KEY, cache_dir=tmp_path)
    assert observation.value == 237.761
    request_lines = [record.getMessage() for record in caplog.records if "/fred/series/" in record.getMessage()]
    assert len(request_lines) == 2
    assert all("api_key=[redacted]" in line for line in request_lines)
    assert KEY not in caplog.text


def test_a_malformed_urllib3_record_is_redacted_without_raising(caplog):
    caplog.set_level(logging.DEBUG, logger="urllib3")
    logging.getLogger("urllib3.connectionpool").debug("GET %s %s", f"/fred/series/observations?api_key={KEY}")
    assert KEY not in caplog.text and "api_key=[redacted]" in caplog.text
