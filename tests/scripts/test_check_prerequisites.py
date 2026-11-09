"""Offline checks for the finance-simulation prerequisite receipt (task 1.2).

Every test runs against recorded or fake responses; no request leaves the
process. The live run happens once through ``dagger call run-with-secrets``.
"""
import json
from datetime import date

import pytest

from scripts.finance_sim import check_prerequisites as cp

START = date(2016, 1, 1)
END = date(2026, 7, 31)
SECRET = "s3cr3t-value-never-in-receipts"


def _weekdays(first: date, last: date) -> list[str]:
    out = []
    day = first
    while day <= last:
        if day.weekday() < 5:
            out.append(day.isoformat())
        day = date.fromordinal(day.toordinal() + 1)
    return out


# ---------- coverage span ----------

def test_span_covers_full_window_with_trading_day_edges():
    dates = [date.fromisoformat(d) for d in _weekdays(date(2016, 1, 4), date(2026, 7, 31))]
    span = cp.evaluate_span(dates, START, END)
    assert span["covers"] is True
    assert span["first"] == "2016-01-04"
    assert span["last"] == "2026-07-31"
    assert span["count"] == len(dates)
    assert span["max_gap_days"] == 3


def test_span_fails_when_history_starts_late():
    dates = [date.fromisoformat(d) for d in _weekdays(date(2017, 3, 1), date(2026, 7, 31))]
    span = cp.evaluate_span(dates, START, END)
    assert span["covers"] is False
    assert span["covers_start"] is False
    assert span["covers_end"] is True


def test_span_fails_when_history_ends_early():
    dates = [date.fromisoformat(d) for d in _weekdays(date(2016, 1, 4), date(2026, 6, 30))]
    span = cp.evaluate_span(dates, START, END)
    assert span["covers_end"] is False
    assert span["covers"] is False


def test_span_fails_on_internal_gap_and_ignores_dates_outside_window():
    dates = [date.fromisoformat(d) for d in _weekdays(date(2015, 6, 1), date(2020, 1, 31))]
    dates += [date.fromisoformat(d) for d in _weekdays(date(2020, 3, 2), date(2026, 9, 30))]
    span = cp.evaluate_span(dates, START, END)
    assert span["first"] == "2016-01-01"
    assert span["last"] == "2026-07-31"
    assert span["max_gap_days"] > cp.MAX_GAP_DAYS
    assert span["gap_ok"] is False
    assert span["covers"] is False


def test_span_of_no_dates_does_not_cover():
    span = cp.evaluate_span([], START, END)
    assert span == {"first": None, "last": None, "count": 0, "max_gap_days": None,
                    "covers_start": False, "covers_end": False, "gap_ok": False, "covers": False}


# ---------- payload parsing ----------

def test_fmp_adjusted_rows_need_a_numeric_adjusted_close():
    payload = [
        {"symbol": "XLK", "date": "2016-01-05", "adjClose": 40.1},
        {"symbol": "XLK", "date": "2016-01-04", "adjClose": 39.9},
        {"symbol": "XLK", "date": "2016-01-06", "adjClose": None},
        {"symbol": "XLK", "date": "bad", "adjClose": 1.0},
        {"symbol": "XLK", "date": "2016-01-07", "close": 41.0},
    ]
    assert cp.fmp_dates(payload, "adjClose") == [date(2016, 1, 4), date(2016, 1, 5)]


def test_fmp_non_list_payload_is_rejected():
    with pytest.raises(ValueError):
        cp.fmp_dates({"Error Message": "Restricted Endpoint"}, "adjClose")


def test_fred_missing_values_are_excluded():
    payload = {"observations": [
        {"date": "2016-01-01", "value": "."},
        {"date": "2016-01-04", "value": "0.22"},
        {"date": "2016-01-05", "value": "0.21"},
    ]}
    assert cp.fred_observation_dates(payload) == [date(2016, 1, 4), date(2016, 1, 5)]


# ---------- vintage detection ----------

def test_vintages_before_window_are_point_in_time():
    status = cp.classify_vintages({"count": 812, "vintage_dates": ["1996-12-03"]}, START)
    assert status == {"vintage_count": 812, "first_vintage": "1996-12-03",
                      "status": "point_in_time", "revised_data": False}


def test_vintages_starting_inside_window_are_partial():
    status = cp.classify_vintages({"count": 40, "vintage_dates": ["2019-05-01"]}, START)
    assert status["status"] == "partial"
    assert status["revised_data"] is True


def test_no_vintages_means_revised_data():
    status = cp.classify_vintages({"count": 0, "vintage_dates": []}, START)
    assert status["status"] == "none"
    assert status["revised_data"] is True
    assert status["first_vintage"] is None


def test_point_in_time_probe_accepts_only_values_known_on_the_date():
    as_of = date(2016, 2, 1)
    known = {"observations": [
        {"date": "2015-11-01", "value": "237.3", "realtime_start": "2016-01-20"},
        {"date": "2015-12-01", "value": "237.8", "realtime_start": "2016-01-20"},
    ]}
    result = cp.evaluate_point_in_time(known, as_of)
    assert result == {"as_of": "2016-02-01", "count": 2, "last_observation": "2015-12-01", "ok": True}
    leaked = {"observations": [{"date": "2016-01-01", "value": "1", "realtime_start": "2016-02-19"}]}
    assert cp.evaluate_point_in_time(leaked, as_of)["ok"] is False
    assert cp.evaluate_point_in_time({"observations": []}, as_of)["ok"] is False


# ---------- registry ----------

REGISTRY = """\
models:
  # Card, retrieved 2026-11-08.
  google/gemma-3-4b-it: 2024-08-31  # https://ai.google.dev/gemma/docs/core/model_card_3
  some/unsourced-model: 2024-01-31
"""


def test_registry_entry_requires_a_source_url(tmp_path):
    path = tmp_path / "cutoffs.yaml"
    path.write_text(REGISTRY, encoding="utf-8")
    ok = cp.check_registry(path, "google/gemma-3-4b-it")
    assert ok["status"] == "pass"
    assert ok["detail"]["cutoff"] == "2024-08-31"
    assert ok["detail"]["source"] == "https://ai.google.dev/gemma/docs/core/model_card_3"
    assert cp.check_registry(path, "some/unsourced-model")["status"] == "fail"
    assert cp.check_registry(path, "absent/model")["status"] == "fail"


def test_project_registry_has_a_sourced_gemma_3_4b_cutoff():
    result = cp.check_registry(cp.DEFAULT_CUTOFFS, cp.CUTOFF_MODEL_ID)
    assert result["status"] == "pass", result


# ---------- aggregation and the stop rule ----------

def test_all_required_passes_proceed():
    checks = [cp.make_check("a", "data", True), cp.make_check("b", "tooling", True)]
    assert cp.aggregate(checks) == {"status": "pass", "failed": [], "decision": "proceed"}


def test_any_required_failure_stops_for_a_decision_and_names_it():
    checks = [cp.make_check("a", "data", True), cp.make_check("fred_key_works", "data", False),
              cp.make_check("info", "data", False, required=False)]
    outcome = cp.aggregate(checks)
    assert outcome["status"] == "fail"
    assert outcome["failed"] == ["fred_key_works"]
    assert outcome["decision"] == "stop_for_decision"


# ---------- secrets never reach receipts ----------

def test_redact_and_leak_detection():
    text = f"Invalid key {SECRET} for request"
    assert SECRET not in cp.redact(text, {"FRED_API_KEY": SECRET})
    assert cp.find_leaks(text, {"FRED_API_KEY": SECRET, "EMPTY": ""}) == ["FRED_API_KEY"]
    assert cp.find_leaks("clean", {"FRED_API_KEY": SECRET}) == []


# ---------- full run against a fake HTTP session ----------

class FakeResponse:
    def __init__(self, status, payload=None, headers=None):
        self.status_code = status
        self._payload = payload
        self.headers = headers or {}

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeHttp:
    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def _answer(self, method, url, params):
        self.calls.append((method, url, dict(params or {})))
        for key, factory in self.routes.items():
            if key in url:
                return factory(url, params or {})
        raise AssertionError(f"unexpected request {method} {url}")

    def get(self, url, params=None, headers=None, timeout=None, allow_redirects=True):
        return self._answer("GET", url, params)

    def head(self, url, params=None, headers=None, timeout=None, allow_redirects=True):
        return self._answer("HEAD", url, params)


def _fmp(url, params):
    rows = [{"date": d, "adjClose": 10.0} for d in _weekdays(date(2016, 1, 4), date(2026, 7, 31))]
    if params["symbol"] == "SWDA.L" and "dividend-adjusted" in url:
        return FakeResponse(402, {"Error Message": f"Restricted Endpoint for {SECRET}"})
    if "light" in url:
        return FakeResponse(200, [{"date": r["date"], "price": 10.0} for r in rows])
    return FakeResponse(200, rows)


def _fred(url, params):
    series = params["series_id"]
    if url.endswith("vintagedates"):
        first = {"CPIAUCSL": "1996-12-03", "T10Y2Y": "2009-03-06", "BAA10Y": "2009-03-06",
                 "DTB3": "2019-01-01"}[series]
        return FakeResponse(200, {"count": 100, "vintage_dates": [first]})
    if params.get("realtime_start") == "2016-02-01":
        return FakeResponse(200, {"observations": [
            {"date": "2015-12-01", "value": "237.8", "realtime_start": "2016-01-20"}]})
    obs = [{"date": d, "value": "0.3"} for d in _weekdays(date(2016, 1, 4), date(2026, 7, 31))]
    return FakeResponse(200, {"observations": obs})


def _hf(url, params):
    if "/api/models/" in url:
        return FakeResponse(200, {"sha": "a" * 40, "gated": "manual"})
    return FakeResponse(200, None)


ROUTES = {"financialmodelingprep.com": _fmp, "api.stlouisfed.org": _fred, "huggingface.co": _hf}


def _host_facts(tmp_path, *, failing=()):
    ids = ["llama_cpp_b12199", "gemma3_4b_gguf", "node", "palette_validator", "env_template_names"]
    checks = [{"id": i, "group": "host", "status": "fail" if i in failing else "pass",
               "required": True, "detail": {"path": f"C:/x/{i}"}} for i in ids]
    path = tmp_path / "host_facts.json"
    path.write_text(json.dumps({"schema": cp.HOST_FACTS_SCHEMA, "checks": checks}), encoding="utf-8")
    return path


FULL_ENV = {name: SECRET for name in cp.SECRET_NAMES}


def test_full_run_records_every_check_and_never_a_secret(tmp_path):
    http = FakeHttp(ROUTES)
    receipt = cp.run_checks(FULL_ENV, http, _host_facts(tmp_path), cp.DEFAULT_CUTOFFS)
    by_id = {c["id"]: c for c in receipt["checks"]}
    for expected in ("fmp_adjusted_world_equity", "fmp_adjusted_tech", "fmp_adjusted_gold",
                     "fred_key_works", "cpi_point_in_time", "vintages_T10Y2Y", "vintages_BAA10Y",
                     "vintages_DTB3", "dtb3_coverage", "registry_gemma3_4b", "hf_gated_revision",
                     "modal_credentials", "local_model_api_key", "llama_cpp_b12199", "palette_validator"):
        assert expected in by_id, expected
    assert by_id["fmp_adjusted_tech"]["status"] == "pass"
    assert by_id["fmp_adjusted_world_equity"]["status"] == "fail"
    assert by_id["fmp_adjusted_world_equity"]["detail"]["http_status"] == 402
    assert by_id["fmp_adjusted_world_equity"]["detail"]["unadjusted_fallback"]["covers"] is True
    assert by_id["vintages_DTB3"]["status"] == "pass"
    assert by_id["vintages_DTB3"]["detail"]["status"] == "partial"
    assert by_id["cpi_point_in_time"]["status"] == "pass"
    assert by_id["hf_gated_revision"]["detail"]["revision"] == "a" * 40
    assert receipt["outcome"]["failed"] == ["fmp_adjusted_world_equity"]
    assert receipt["outcome"]["decision"] == "stop_for_decision"
    assert receipt["credentials"] == {name: True for name in cp.SECRET_NAMES}
    assert SECRET not in json.dumps(receipt)
    assert SECRET not in cp.render_summary(receipt)
    assert receipt["requests"]["fred"] <= 6
    for _method, url, params in http.calls:
        assert "apikey" not in url and "api_key" not in url and SECRET not in url
        if "financialmodelingprep" in url:
            assert params["apikey"] == SECRET


def test_missing_fred_key_fails_without_any_fred_request(tmp_path):
    env = dict(FULL_ENV, FRED_API_KEY="")
    http = FakeHttp(ROUTES)
    receipt = cp.run_checks(env, http, _host_facts(tmp_path), cp.DEFAULT_CUTOFFS)
    assert not [c for c in http.calls if "stlouisfed" in c[1]]
    failed = receipt["outcome"]["failed"]
    for fred_check in ("fred_key_works", "cpi_point_in_time", "vintages_T10Y2Y", "vintages_BAA10Y",
                       "vintages_DTB3", "dtb3_coverage"):
        assert fred_check in failed
    by_id = {c["id"]: c for c in receipt["checks"]}
    assert "FRED_API_KEY is not set" in by_id["fred_key_works"]["detail"]["reason"]
    assert receipt["credentials"]["FRED_API_KEY"] is False


def test_failing_host_fact_and_missing_host_facts_stop(tmp_path):
    receipt = cp.run_checks(FULL_ENV, FakeHttp(ROUTES), _host_facts(tmp_path, failing=("node",)),
                            cp.DEFAULT_CUTOFFS)
    assert "node" in receipt["outcome"]["failed"]
    missing = cp.run_checks(FULL_ENV, FakeHttp(ROUTES), tmp_path / "absent.json", cp.DEFAULT_CUTOFFS)
    assert "host_facts" in missing["outcome"]["failed"]


def test_gated_repository_without_access_fails(tmp_path):
    routes = dict(ROUTES)
    routes["huggingface.co"] = lambda url, params: (
        FakeResponse(200, {"sha": "b" * 40, "gated": "manual"}) if "/api/models/" in url
        else FakeResponse(403, None, {"x-error-code": "GatedRepo"}))
    receipt = cp.run_checks(FULL_ENV, FakeHttp(routes), _host_facts(tmp_path), cp.DEFAULT_CUTOFFS)
    hf = {c["id"]: c for c in receipt["checks"]}["hf_gated_revision"]
    assert hf["status"] == "fail"
    assert hf["detail"]["file_status"] == 403
    assert hf["detail"]["error_code"] == "GatedRepo"


def test_main_writes_receipt_and_summary_and_exits_zero_on_failed_prerequisites(tmp_path, monkeypatch):
    for name in cp.SECRET_NAMES:
        monkeypatch.setenv(name, SECRET)
    monkeypatch.setenv("FRED_API_KEY", "")
    monkeypatch.setattr(cp, "make_http", lambda: FakeHttp(ROUTES))
    out = tmp_path / "out"
    code = cp.main(["--host-facts", str(_host_facts(tmp_path)), "--out", str(out)])
    assert code == 0
    receipt = json.loads((out / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["outcome"]["decision"] == "stop_for_decision"
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "fred_key_works" in summary
    assert SECRET not in summary and SECRET not in json.dumps(receipt)
