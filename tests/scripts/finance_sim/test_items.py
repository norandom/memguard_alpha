"""Study item assembly on recorded responses (task 7.1; Requirements 6.2, 6.4, 6.6).

FRED bodies follow the FRED JSON API (``series/vintagedates`` and
``series/observations``) and are served by a fake that replaces
``requests.adapters.HTTPAdapter.send``, so the real FRED loader runs without
network. FMP and DTB3 go through the injectable transport with small synthetic
bodies in the documented shapes (FMP: a list of ``{symbol, date, adjClose, ...}``
rows); no licensed price data is used. Every private output goes to ``tmp_path``.
"""

import hashlib
import json
import shutil
import statistics
from dataclasses import astuple
from datetime import date
from types import SimpleNamespace
from urllib.parse import parse_qsl, urlsplit

import pytest
import requests

from recall_guard.dataset import fred
from recall_guard.dataset.edgar_outlook import prompt_sha256, render_outlook_prompt
from recall_guard.dataset.fred import RollingRateLimiter
from recall_guard.dataset.macro_regime import RealizedReturns, is_correct, render_anonymised, render_identifying
from scripts.finance_sim import items

FRED_KEY = "f3d7e9a1c2b4d6f8a0e1c3b5d7f9a2c4"
FMP_KEY = "Zq8Wm2Lp5Rt7Yx1Nc4Vb6Hj9Kd3Gf0Sa"
GEMMA = "google/gemma-3-4b-it"
NO_CUTOFF = "acme/no-cutoff-7b"
MONTHS = (date(2016, 1, 31), date(2016, 2, 29), date(2016, 3, 31))
# April 2016 is not complete on this date, so March keeps realized = None.
TODAY = date(2016, 4, 20)
# CPIAUCSL has vintages throughout, T10Y2Y has none, BAA10Y has them from 2016-02-15 on.
FIRST_VINTAGE = {"CPIAUCSL": ["1972-07-21"], "T10Y2Y": [], "BAA10Y": ["2016-02-15"]}
HISTORY_START = 2009 * 12  # January 2009, as a month index


def month_index(day):
    return day.year * 12 + day.month - 1


def month_start(index):
    return date(index // 12, index % 12 + 1, 1)


def cpi_text(index, *, vintage):
    """CPI level as FRED prints it; the current vintage is revised up by 0.25."""
    k = index - HISTORY_START
    base = 210.0 + 0.35 * k + 0.6 * ((k * 7) % 5)
    return f"{base if vintage else base + 0.25:.3f}"


def daily_text(series, day, *, vintage):
    """Daily value on the 10th and the 26th of each month; the 10th is never a month-end value."""
    k = month_index(day) - HISTORY_START
    base = {"T10Y2Y": 1.0 + 0.1 * ((k * 3) % 7), "BAA10Y": 2.5 + 0.15 * ((k * 5) % 6)}[series]
    if day.day == 10:
        base += 0.05
    return f"{base if vintage else base + 0.01:.4f}"


def daily_days(as_of, first_index=HISTORY_START):
    return [day for index in range(first_index, month_index(as_of) + 1)
            for day in (date(index // 12, index % 12 + 1, 10), date(index // 12, index % 12 + 1, 26))
            if day <= as_of]


def observations_body(rows, realtime):
    return {"realtime_start": realtime, "realtime_end": realtime, "observation_start": "1600-01-01",
            "units": "lin", "output_type": 1, "file_type": "json", "order_by": "observation_date",
            "sort_order": "asc", "count": len(rows), "offset": 0, "limit": 100000,
            "observations": [{"realtime_start": realtime, "realtime_end": realtime, "date": day, "value": value}
                             for day, value in rows]}


def fred_routes(endpoint, params, *, baa_first_index=HISTORY_START):
    """Answer a FRED request as FRED does: a vintage on as_of omits CPI for as_of's own month."""
    series = params["series_id"]
    if endpoint == "vintagedates":
        dates = FIRST_VINTAGE[series]
        return 200, {"realtime_start": "1776-07-04", "realtime_end": "9999-12-31", "order_by": "vintage_date",
                     "sort_order": "asc", "count": len(dates), "offset": 0, "limit": 1, "vintage_dates": dates}
    as_of = date.fromisoformat(params["observation_end"])
    point_in_time = "realtime_start" in params
    if series == "CPIAUCSL":
        last = month_index(as_of) - (1 if point_in_time else 0)
        rows = [(month_start(index).isoformat(), cpi_text(index, vintage=point_in_time))
                for index in range(HISTORY_START, last + 1)]
    else:
        first = baa_first_index if series == "BAA10Y" else HISTORY_START
        rows = [(day.isoformat(), daily_text(series, day, vintage=point_in_time)) for day in daily_days(as_of, first)]
    return 200, observations_body(rows, params["realtime_start"] if point_in_time else "2026-11-09")


class FakeFred:
    """Serve FRED requests from ``routes`` at the HTTP adapter; any other host fails the test."""

    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def send(self, adapter, prepared, **kwargs):
        url = urlsplit(prepared.url)
        assert url.hostname == "api.stlouisfed.org", f"unexpected request to {url.hostname}"
        params = dict(parse_qsl(url.query, keep_blank_values=True))
        endpoint = url.path.rsplit("/", 1)[-1]
        self.calls.append({"endpoint": endpoint, "params": params})
        status, body = self.routes(endpoint, params)
        response = requests.Response()
        response.status_code = status
        response._content = body if isinstance(body, bytes) else json.dumps(body).encode()
        response._content_consumed = True
        response.request = prepared
        response.url = prepared.url
        return response


# Synthetic dividend-adjusted closes in FMP's shape (newest first); not real prices.
CLOSES = {
    "SWDA.L": {"2015-12-31": 38.10, "2016-01-15": 36.00, "2016-01-29": 36.20, "2016-02-29": 35.475,
               "2016-03-31": 37.60, "2016-04-15": 38.00},
    "XLK": {"2015-12-31": 42.83, "2016-01-15": 40.00, "2016-01-29": 41.00, "2016-02-29": 41.82,
            "2016-03-31": 44.30, "2016-04-15": 44.90},
    "IAU": {"2015-12-31": 10.14, "2016-01-15": 10.50, "2016-01-29": 10.80, "2016-02-29": 11.88,
            "2016-03-31": 11.85, "2016-04-15": 12.00},
}
DTB3_ROWS = [("2015-12-31", "0.16"), ("2016-01-28", "0.31"), ("2016-01-29", "0.32"), ("2016-02-29", "0.33"),
             ("2016-03-25", "."), ("2016-03-31", "0.21"), ("2016-04-15", "0.22")]


def fmp_body(symbol):
    return [{"symbol": symbol, "date": day, "adjOpen": close, "adjHigh": close, "adjLow": close, "adjClose": close,
             "volume": 1000} for day, close in sorted(CLOSES[symbol].items(), reverse=True)]


def market_routes(url, params):
    if url == items.FMP_URL:
        return 200, json.dumps(fmp_body(params["symbol"])).encode()
    if url == f"{fred.BASE_URL}/observations" and params.get("series_id") == "DTB3":
        return 200, json.dumps(observations_body(DTB3_ROWS, "2026-11-09")).encode()
    raise AssertionError(f"unexpected transport request to {url}")


class RecordedTransport:
    """The injectable transport: records each call and answers from recorded bodies."""

    def __init__(self, routes=market_routes):
        self.routes = routes
        self.calls = []

    def __call__(self, url, params, timeout_s):
        self.calls.append({"url": url, "params": dict(params), "timeout_s": timeout_s})
        return self.routes(url, dict(params))


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


@pytest.fixture
def world(monkeypatch, tmp_path):
    clock = FakeClock()
    monkeypatch.setattr(fred, "RATE_LIMITER", RollingRateLimiter(clock=clock, sleep=clock.sleep))
    fake = FakeFred(fred_routes)
    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send",
                        lambda adapter, prepared, **kwargs: fake.send(adapter, prepared, **kwargs))
    working = tmp_path / "working"
    working.mkdir()
    monkeypatch.chdir(working)
    return SimpleNamespace(fred=fake, transport=RecordedTransport(), root=tmp_path / "private", working=working,
                           tmp=tmp_path)


def assemble(world, **overrides):
    arguments = {"fred_api_key": FRED_KEY, "fmp_api_key": FMP_KEY, "model_ids": (GEMMA,),
                 "private_root": world.root, "months": MONTHS, "today": TODAY, "transport": world.transport}
    return items.assemble_items(**(arguments | overrides))


def assert_no_key(root):
    for path in root.rglob("*"):
        for key in (FRED_KEY, FMP_KEY):
            assert key not in str(path)
            if path.is_file():
                assert key.encode() not in path.read_bytes()


def month_end_window(series, decision_date, *, vintage):
    """The 60 month-end values (the 26th) up to the decision month, as the design's z-score uses them."""
    last = month_index(decision_date)
    return [float(daily_text(series, date(index // 12, index % 12 + 1, 26), vintage=vintage))
            for index in range(last - 59, last + 1)]


# ---------- macro items ----------

def test_macro_items_are_point_in_time_rows_with_vintage_flags(world):
    assembly = assemble(world)
    assert [item.item_id for item in assembly.macro] == ["macro-2016-01", "macro-2016-02", "macro-2016-03"]
    january, february, march = assembly.macro

    # Revised flags follow each series' vintages on the decision date (6.5).
    assert january.row.revised == {"BAA10Y": True, "CPIAUCSL": False, "T10Y2Y": True}
    assert february.row.revised == march.row.revised == {"BAA10Y": False, "CPIAUCSL": False, "T10Y2Y": True}

    # CPI as published on 2016-01-31: December 2015 over December 2014, vintage values, not the revised ones.
    december_2015, december_2014 = month_index(date(2015, 12, 1)), month_index(date(2014, 12, 1))
    vintage_yoy = (float(cpi_text(december_2015, vintage=True)) / float(cpi_text(december_2014, vintage=True)) - 1) * 100
    revised_yoy = (float(cpi_text(december_2015, vintage=False)) / float(cpi_text(december_2014, vintage=False)) - 1) * 100
    assert january.row.raw["cpi_yoy"] == pytest.approx(vintage_yoy, rel=1e-12)
    assert january.row.raw["cpi_yoy"] != pytest.approx(revised_yoy, rel=1e-9)
    assert january.observed_through == {"BAA10Y": "2016-01-26", "CPIAUCSL": "2015-12-01", "T10Y2Y": "2016-01-26"}

    # BAA10Y switches from the current vintage to the point-in-time view once its vintages start.
    assert january.row.raw["baa10y"] == float(daily_text("BAA10Y", date(2016, 1, 26), vintage=False))
    assert february.row.raw["baa10y"] == float(daily_text("BAA10Y", date(2016, 2, 26), vintage=True))

    # z-score: the latest value against the 60 month-end values to the decision month, ddof 1.
    window = month_end_window("T10Y2Y", january.decision_date, vintage=False)
    expected = (window[-1] - statistics.fmean(window)) / statistics.stdev(window)
    assert january.row.zscores["t10y2y_z"] == pytest.approx(expected, rel=1e-12)

    # One vintage list per series, one observation request per series and month.
    endpoints = [call["endpoint"] for call in world.fred.calls]
    assert endpoints.count("vintagedates") == 3 and endpoints.count("observations") == 9
    assert {call["params"]["observation_end"] for call in world.fred.calls if call["endpoint"] == "observations"} \
        == {day.isoformat() for day in MONTHS}


def test_item_records_hash_the_row_and_both_renderings_and_classify_each_month(world):
    assembly = assemble(world)
    record = assembly.to_dict()["macro_items"][0]
    row = items.row_to_dict(assembly.macro[0].row)
    canonical = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert record["row"] == row
    assert record["row_sha256"] == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert record["prompt_sha256"] == {"anonymised": prompt_sha256(render_anonymised(assembly.macro[0].row)),
                                       "identifying": prompt_sha256(render_identifying(assembly.macro[0].row))}
    assert record["row"]["revised"] == {"BAA10Y": True, "CPIAUCSL": False, "T10Y2Y": True}
    assert record["pre_cutoff"] == {GEMMA: True}


def test_a_month_without_enough_z_score_history_is_excluded_with_the_reason(world):
    # BAA10Y starts in October 2013: 28 and 29 monthly values for January and February 2016, 30 for March.
    world.fred.routes = lambda endpoint, params: fred_routes(endpoint, params,
                                                             baa_first_index=month_index(date(2013, 10, 1)))
    assembly = assemble(world)
    assert [item.item_id for item in assembly.macro] == ["macro-2016-03"]
    assert set(assembly.excluded_months) == {date(2016, 1, 31), date(2016, 2, 29)}
    assert "BAA10Y has 28 monthly values" in assembly.excluded_months[date(2016, 1, 31)]
    assert "at least 30" in assembly.excluded_months[date(2016, 2, 29)]
    record = assembly.to_dict()
    assert record["excluded_months"][0] == {"decision_date": "2016-01-31",
                                            "reason": assembly.excluded_months[date(2016, 1, 31)]}
    assert record["summary"]["excluded_months"] == 2


# ---------- realized returns and private files ----------

def test_realized_returns_come_from_adjusted_closes_and_dtb3_and_stay_private(world):
    assembly = assemble(world)
    january, february, march = assembly.macro
    assert isinstance(january.outcome.realized, RealizedReturns)
    assert astuple(january.outcome.realized) == pytest.approx(
        (35.475 / 36.20 - 1, 41.82 / 41.00 - 1, 11.88 / 10.80 - 1, 0.32 / 100 / 12), rel=1e-12)
    assert february.outcome.realized.short_treasury_cash == pytest.approx(0.33 / 100 / 12)
    assert february.outcome.realized.world_equity == pytest.approx(37.60 / 35.475 - 1)
    assert march.outcome.realized is None
    assert "2016-04 is not complete" in march.outcome.reason
    # The returns feed the frozen accuracy rule unchanged (6.6): defensive (gold plus cash) beat risk in January.
    assert is_correct(0.5, january.outcome.realized) and not is_correct(-0.5, january.outcome.realized)

    returns = json.loads((world.root / "data" / "realized_returns.json").read_text("utf-8"))
    assert returns["schema"] == items.RETURNS_SCHEMA and "licensed" in returns["licensing"]
    assert [month["decision_date"] for month in returns["months"]] == [day.isoformat() for day in MONTHS]
    assert returns["months"][0]["dates"] == {"gold_commodity": ["2016-01-29", "2016-02-29"],
                                             "short_treasury_cash": ["2016-01-29"],
                                             "tech_sector": ["2016-01-29", "2016-02-29"],
                                             "world_equity": ["2016-01-29", "2016-02-29"]}
    assert returns["months"][2]["realized"] is None and returns["months"][2]["reason"] == march.outcome.reason
    assert returns["complete_through"] == "2016-04-19"

    # Raw FMP bodies live in the private FMP folder, DTB3 beside the FRED loader cache.
    assert sorted(path.name for path in (world.root / "data" / "fmp").iterdir()) == [
        "IAU_2015-12-01_2016-04-30.json", "SWDA.L_2015-12-01_2016-04-30.json", "XLK_2015-12-01_2016-04-30.json"]
    assert [path.name.startswith("DTB3_current_") for path in (world.root / "fred").glob("DTB3_*")] == [True]
    assert assembly.to_dict()["private_files"] == {"fmp_cache": "data/fmp", "fred_cache": "fred",
                                                   "realized_returns": "data/realized_returns.json"}

    # FMP dividend-adjusted closes over the design window; DTB3 from the current vintage (no real-time period).
    fmp_calls = [call for call in world.transport.calls if call["url"] == items.FMP_URL]
    assert [call["params"] for call in fmp_calls] == [
        {"symbol": symbol, "from": "2015-12-01", "to": "2016-04-30", "apikey": FMP_KEY}
        for symbol in ("SWDA.L", "XLK", "IAU")]
    (dtb3_call,) = [call for call in world.transport.calls if call["url"] != items.FMP_URL]
    assert dtb3_call["params"] == {"series_id": "DTB3", "file_type": "json", "sort_order": "asc",
                                   "observation_start": "2015-12-01", "observation_end": "2016-04-30",
                                   "api_key": FRED_KEY}

    # Nothing outside the private root, and no key in any file or file name.
    assert sorted(path.name for path in world.tmp.iterdir()) == ["private", "working"]
    assert list(world.working.iterdir()) == []
    assert_no_key(world.tmp)


def test_a_repeated_assembly_replays_from_the_private_caches_and_writes_identical_items(world):
    first = assemble(world)
    fred_requests, transport_requests = len(world.fred.calls), len(world.transport.calls)
    second = assemble(world)
    assert (len(world.fred.calls), len(world.transport.calls)) == (fred_requests, transport_requests)
    assert second.to_dict() == first.to_dict()

    digest = items.write_items(first)
    path = world.root / "plan" / "items.json"
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert items.write_items(second) == digest
    stored = json.loads(path.read_text("utf-8"))
    assert stored["schema"] == items.ITEMS_SCHEMA
    assert stored["summary"]["macro_items"] == 3 and stored["summary"]["realized_missing"] == 1
    assert_no_key(world.tmp)


def test_the_default_private_root_is_the_runs_folder_from_the_repository_root():
    assert items.PRIVATE_ROOT == items.ROOT / "runs" / "finance-sim"
    paths = items.PrivatePaths(items.PRIVATE_ROOT)
    assert paths.fred_cache == items.ROOT / fred.DEFAULT_CACHE_DIR
    assert paths.items == items.ROOT / "runs" / "finance-sim" / "plan" / "items.json"


# ---------- cutoff classes ----------

def test_a_model_without_a_sourced_cutoff_is_excluded_with_the_reason(world):
    assembly = assemble(world, model_ids=(GEMMA, NO_CUTOFF))
    assert set(assembly.cutoffs.included) == {GEMMA}
    assert "has no training cutoff in the sourced cutoff registry cutoffs.yaml" in assembly.cutoffs.excluded[NO_CUTOFF]
    record = assembly.to_dict()
    assert record["models"]["excluded"] == {NO_CUTOFF: assembly.cutoffs.excluded[NO_CUTOFF]}
    assert record["models"]["included"][GEMMA]["cutoff"] == "2024-08-31"
    assert record["models"]["included"][GEMMA]["source"] == "https://ai.google.dev/gemma/docs/core/model_card_3"
    assert all(item["pre_cutoff"] == {GEMMA: True} for item in record["macro_items"])
    assert record["summary"]["models_excluded"] == [NO_CUTOFF]


def test_decision_months_split_at_the_gemma_cutoff_into_104_before_and_22_after():
    months = items.decision_months()
    assert (len(months), months[0], months[-1]) == (126, date(2016, 1, 31), date(2026, 6, 30))
    classes = items.classify_cutoffs(months, (GEMMA,))
    gemma = classes.included[GEMMA]
    assert (gemma.pre_count, gemma.post_count) == (104, 22)
    assert gemma.pre_cutoff[date(2024, 8, 31)] is True and gemma.pre_cutoff[date(2024, 9, 30)] is False
    assert classes.excluded == {}


def test_a_registry_entry_without_a_source_is_not_a_sourced_cutoff(tmp_path):
    registry = tmp_path / "cutoffs.yaml"
    registry.write_text("models:\n"
                        "  vendor/sourced-1b: 2024-01-31  # \"Cutoff January 2024\" -- https://example.org/card\n"
                        "  vendor/unsourced-1b: 2024-01-31  # stated in a blog post\n", encoding="utf-8")
    classes = items.classify_cutoffs(items.decision_months(), ("vendor/sourced-1b", "vendor/unsourced-1b"),
                                     registry=registry)
    assert classes.included["vendor/sourced-1b"].source == "https://example.org/card"
    assert "cites no source" in classes.excluded["vendor/unsourced-1b"]


# ---------- EDGAR items ----------

def test_edgar_items_keep_their_split_label_quarter_and_prompt_hash():
    edgar = items.load_edgar_items()
    manifest = json.loads((items.DEFAULT_EDGAR_DIR / "manifest.json").read_text("utf-8"))
    assert [item.label for item in edgar.calibration].count(1) == 40
    assert [item.label for item in edgar.calibration].count(0) == 100
    assert {item.split for item in edgar.calibration} == {"calibration"}
    assert len(edgar.negative_control) == 40
    assert {(item.split, item.label, item.source_file) for item in edgar.negative_control} == {
        ("negative_control", 0, "held_out/oos_control.jsonl")}
    assert all(item.published_at > date(2024, 8, 31) for item in edgar.negative_control)
    assert edgar.unused == {"held_out/is_memorized.jsonl": items.EDGAR_UNUSED_REASON}
    assert {name: entry["sha256"] for name, entry in edgar.files.items()} == {
        name: manifest["files"][name]["sha256"] for name in items.EDGAR_FILES}
    first = edgar.negative_control[0]
    assert first.prompt_sha256 == prompt_sha256(render_outlook_prompt(first.release_text))
    assert first.quarter == f"{first.published_at.year}Q{(first.published_at.month - 1) // 3 + 1}"
    assert first.item_id == f"edgar-negative_control-{first.accession}"
    record = first.to_dict()
    assert record["split"] == "negative_control" and record["prompt_sha256"] == first.prompt_sha256


def test_the_assembly_records_edgar_items_with_their_split(world):
    record = assemble(world).to_dict()["edgar"]
    assert (len(record["calibration"]), len(record["negative_control"])) == (140, 40)
    assert {entry["split"] for entry in record["calibration"]} == {"calibration"}
    assert {entry["split"] for entry in record["negative_control"]} == {"negative_control"}
    assert record["files"]["held_out/is_memorized.jsonl"]["role"] == "unused"


def test_a_changed_edgar_row_set_is_refused(tmp_path):
    copy = tmp_path / "edgar"
    shutil.copytree(items.DEFAULT_EDGAR_DIR, copy)
    target = copy / "held_out" / "oos_control.jsonl"
    target.write_bytes(target.read_bytes().replace(b"Ryan Specialty", b"Ryan Specialtx", 1))
    with pytest.raises(ValueError, match="held_out/oos_control.jsonl does not match its manifest sha256"):
        items.load_edgar_items(copy)


def test_an_edgar_row_whose_label_disagrees_with_the_manifest_is_refused(tmp_path):
    copy = tmp_path / "edgar"
    shutil.copytree(items.DEFAULT_EDGAR_DIR, copy)
    target = copy / "calibration" / "oos_control.jsonl"
    lines = target.read_bytes().split(b"\n")
    row = json.loads(lines[0])
    row["label"] = 1
    lines[0] = json.dumps(row, ensure_ascii=False).encode("utf-8")
    target.write_bytes(b"\n".join(lines))
    manifest = json.loads((copy / "manifest.json").read_text("utf-8"))
    manifest["files"]["calibration/oos_control.jsonl"]["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    (copy / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="line 1 has label 1, but the manifest gives the file label 0"):
        items.load_edgar_items(copy)


# ---------- source failures ----------

def test_an_fmp_http_error_is_classified_retryable_without_the_key_or_a_cache_file(world):
    world.transport.routes = lambda url, params: (503, f'{{"Error Message": "busy for {FMP_KEY}"}}'.encode())
    with pytest.raises(items.ItemSourceError) as raised:
        assemble(world)
    error = raised.value
    assert (error.source, error.code, error.retryable, error.status_code) == ("FMP SWDA.L", "fmp_http_error", True, 503)
    for text in (str(error), repr(error), error.detail or "", repr(error.args)):
        assert FMP_KEY not in text
    assert "[redacted]" in error.detail
    assert error.__cause__ is None and error.__context__ is None
    assert not (world.root / "data" / "fmp").exists()


def test_a_transport_failure_is_retryable_and_redacted(world):
    def failing(url, params):
        raise requests.ConnectionError(f"cannot reach {url}?symbol=XLK&apikey={FMP_KEY}")

    world.transport.routes = failing
    with pytest.raises(items.ItemSourceError) as raised:
        assemble(world)
    error = raised.value
    assert (error.code, error.retryable) == ("transport_error", True)
    assert FMP_KEY not in str(error) and FMP_KEY not in (error.detail or "") and "[redacted]" in error.detail
    assert error.__context__ is None


def test_a_malformed_fred_body_is_classified_as_invalid_and_not_retryable(world):
    def routes(endpoint, params):
        if endpoint == "observations" and params["series_id"] == "CPIAUCSL":
            return 200, {"observations": "not a list"}
        return fred_routes(endpoint, params)

    world.fred.routes = routes
    with pytest.raises(items.ItemSourceError) as raised:
        assemble(world)
    assert (raised.value.source, raised.value.code, raised.value.retryable) == ("FRED CPIAUCSL", "invalid_body", False)
    assert raised.value.__context__ is None
