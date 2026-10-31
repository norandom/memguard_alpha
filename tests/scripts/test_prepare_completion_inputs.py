"""Adverse input-preparation fixtures; all HTTP is mocked."""

import copy
import hashlib
import json
import shutil
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from recall_guard.core.completion import ModelConfig
from recall_guard.core.loader import CutoffViolation, assert_cutoff_safe, load_cutoffs, load_eval_set
from recall_guard.harness.intervention_config import load_experiment_config
from recall_guard.harness.intervention_types import ExperimentItem, ExperimentProtocol
from recall_guard.harness.rewriting import REWRITE_METHOD_VERSION, render_rewrite_prompt, render_target_prompt
from scripts import prepare_completion_inputs as prep


def chart(days=80, start="2026-07-01", ticker="SPY"):
    dates = []
    day = date.fromisoformat(start)
    while len(dates) < days:
        if day.weekday() < 5 and day.isoformat() not in {"2026-07-03", "2026-09-07"}:
            dates.append(day)
        day += timedelta(days=1)
    timestamps = [int(datetime(d.year, d.month, d.day, 9, 30, tzinfo=ZoneInfo("America/New_York")).timestamp())
                  for d in dates]
    close = [100 + i // 2 + (2 if i % 2 else 0) for i in range(days)]
    result = {"meta": {"symbol": ticker, "currency": "USD", "exchangeTimezoneName": "America/New_York",
                       "instrumentType": "ETF"}, "timestamp": timestamps,
              "indicators": {"quote": [{"close": close[:]}], "adjclose": [{"adjclose": close[:]}]}, "events": {}}
    return {"chart": {"result": [result], "error": None}}


def parsed(payload=None):
    return prep.parse_chart(json.dumps(payload or chart()).encode(), "SPY", completed_through="2026-10-30")


def test_parse_keeps_exact_decimal_closes_and_exchange_dates():
    payload = chart()
    payload["chart"]["result"][0]["indicators"]["quote"][0]["close"][0] = 100.125
    series = parsed(payload)
    assert series["bars"][0]["date"] == "2026-07-01"
    assert series["bars"][0]["close"] == Decimal("100.125")
    assert series["timezone"] == "America/New_York"


@pytest.mark.parametrize("change,match", [
    (lambda r: r["timestamp"].reverse(), "ordered"),
    (lambda r: r["timestamp"].__setitem__(1, r["timestamp"][0]), "unique"),
    (lambda r: r["timestamp"].__setitem__(1, r["timestamp"][1] + 86400), "session"),
    (lambda r: r["indicators"]["quote"][0]["close"].pop(), "length"),
    (lambda r: r["meta"].update(symbol="QQQ"), "symbol"),
    (lambda r: r["meta"].update(currency="EUR"), "currency"),
    (lambda r: r["meta"].update(exchangeTimezoneName="UTC"), "timezone"),
])
def test_invalid_source_alignment_fails_closed(change, match):
    payload = chart()
    change(payload["chart"]["result"][0])
    with pytest.raises(ValueError, match=match):
        parsed(payload)


def test_missing_and_invalid_sessions_are_excluded_without_bridging():
    payload = chart()
    result = payload["chart"]["result"][0]
    result["timestamp"].pop(12)
    result["indicators"]["quote"][0]["close"].pop(12)
    result["indicators"]["adjclose"][0]["adjclose"].pop(12)
    result["indicators"]["quote"][0]["close"][20] = None
    result["indicators"]["quote"][0]["close"][30] = -1
    series = parsed(payload)
    rows, audit = prep.candidates(series, "post", "2026-07-01", "2026-10-30", cutoff="2026-06-30", period="post")
    assert series["missing_sessions"][0] == "2026-07-20"
    assert len(series["invalid_sessions"]) == 2
    assert audit["missing_session"] > 0 and audit["invalid_close"] > 0
    bad = set(series["missing_sessions"]) | set(series["invalid_sessions"])
    assert all(not bad.intersection(row["window_dates"]) for row in rows)
    assert all(len(row["window_dates"]) == 6 for row in rows)


def test_corporate_actions_and_flat_outcomes_are_reported_and_excluded():
    payload = chart()
    result = payload["chart"]["result"][0]
    result["events"] = {"dividends": {"a": {"date": result["timestamp"][12], "amount": 1}},
                        "splits": {"b": {"date": result["timestamp"][40], "numerator": 2, "denominator": 1}}}
    result["indicators"]["quote"][0]["close"][60] = result["indicators"]["quote"][0]["close"][59]
    rows, audit = prep.candidates(parsed(payload), "post", "2026-07-01", "2026-10-30", cutoff="2026-06-30", period="post")
    action_dates = {"2026-07-20", "2026-08-27"}
    assert audit["corporate_action"] == 12 and audit["flat"] == 1
    assert all(not action_dates.intersection(row["window_dates"]) for row in rows)


def test_no_future_date_price_label_or_membership_leaks_into_inputs():
    rows, _ = prep.candidates(parsed(), "post", "2026-07-01", "2026-10-30", cutoff="2026-06-30", period="post")
    row = rows[0]
    item = row["item"]
    input_value = prep.label_free_input(item)
    target = render_target_prompt(input_value, evidence=input_value.evidence)
    rewrite = render_rewrite_prompt(input_value)
    assert len(row["evidence_dates"]) == 5
    assert max(row["evidence_dates"]) == item.as_of < row["outcome_date"]
    assert min(row["evidence_dates"]) > "2026-06-30"
    for text in (target, rewrite):
        assert row["outcome_date"] not in text
        assert "post" not in text and "cohort" not in text and "label" not in text
    changed = copy.deepcopy(chart())
    changed["chart"]["result"][0]["indicators"]["quote"][0]["close"][5] = 900.9375
    changed_row = prep.candidates(parsed(changed), "post", "2026-07-01", "2026-10-30",
                                  cutoff="2026-06-30", period="post")[0][0]
    assert changed_row["item"].evidence == item.evidence
    assert "900.9375" not in target
    assert '{"direction": <-1|0|1>, "confidence": <0..1>}' in item.instructions
    assert "confidence" not in item.evidence and "predict" not in item.evidence


def test_post_window_rejects_any_evidence_on_or_before_cutoff():
    series = parsed()
    rows, audit = prep.candidates(series, "post", "2026-07-01", "2026-10-30", cutoff="2026-07-10", period="post")
    assert audit["cutoff"] > 0
    assert all(min(row["window_dates"]) > "2026-07-10" for row in rows)
    before, _ = prep.candidates(series, "pre", "2026-07-01", "2026-10-30", cutoff="2026-08-01", period="pre")
    assert all(row["outcome_date"] <= "2026-08-01" for row in before)


def pool():
    result = []
    for cohort in ("pre", "post"):
        for ticker in ("IAU", "SPY"):
            payload = chart(ticker=ticker)
            series = prep.parse_chart(json.dumps(payload).encode(), ticker, completed_through="2026-10-30")
            rows, _ = prep.candidates(series, cohort, "2026-07-01", "2026-10-30", cutoff=None, period=cohort)
            result.extend(rows)
    return result


def test_sha_selection_is_deterministic_balanced_and_windows_disjoint():
    rows = pool()
    chosen, audit = prep.select_balanced(rows, per_stratum=2, salt="fixture")
    again, _ = prep.select_balanced(list(reversed(rows)), per_stratum=2, salt="fixture")
    assert [r["item"].item_id for r in chosen] == [r["item"].item_id for r in again]
    assert len(chosen) == 16 and audit["overlap"] > 0
    seen = {}
    for row in chosen:
        key = row["item"].cohort, row["ticker"], row["item"].label
        seen[key] = seen.get(key, 0) + 1
        other = [r for r in chosen if r is not row and r["ticker"] == row["ticker"]
                 and r["item"].cohort == row["item"].cohort]
        assert all(not set(row["window_dates"]).intersection(r["window_dates"]) for r in other)
    assert set(seen.values()) == {2}
    ordered = prep.evaluation_order(chosen)
    assert [r["item"].cohort for r in ordered[:4]] == ["pre", "post", "post", "pre"]
    for cohort in ("pre", "post"):
        assert sum(r["item"].cohort == cohort for r in ordered[::2]) == 4
        assert sum(r["item"].cohort == cohort for r in ordered[1::2]) == 4
    for block in range(0, 16, 4):
        assert len({(r["ticker"], r["item"].label) for r in ordered[block:block + 4]}) == 1


def test_selection_does_not_silently_shrink_underfilled_strata():
    with pytest.raises(ValueError, match="Insufficient"):
        prep.select_balanced(pool(), per_stratum=100, salt="fixture")


def test_selection_rejects_completely_missing_declared_cohort():
    rows = [row for row in pool() if row["item"].cohort == "pre"]
    with pytest.raises(ValueError, match="Insufficient"):
        prep.select_balanced(rows, per_stratum=2, salt="fixture", cohorts=("pre", "post"))


def test_public_fetch_ignores_ambient_auth_and_preserves_raw_bytes(tmp_path, monkeypatch):
    raw = b'{"chart": "raw bytes"}\r\n'
    calls = []

    class Response:
        content = raw
        url = "https://query1.finance.yahoo.com/v8/finance/chart/SPY?range=5y&interval=1d"
        status_code = 200

        def raise_for_status(self):
            pass

    class Session:
        trust_env = True
        headers = {}

        def get(self, url, **kwargs):
            calls.append((url, kwargs, self.trust_env, dict(self.headers)))
            return Response()

        def close(self):
            pass

    monkeypatch.setattr("requests.Session", Session)
    metadata = prep.fetch_chart("SPY", tmp_path)
    assert (tmp_path / "SPY.chart.json").read_bytes() == raw
    assert metadata["sha256"] == hashlib.sha256(raw).hexdigest()
    assert calls[0][2] is False
    assert "auth" not in calls[0][1] and "Authorization" not in calls[0][3]
    with pytest.raises(FileExistsError):
        prep.fetch_chart("SPY", tmp_path)


DATA = Path(__file__).resolve().parents[2] / "data" / "experiments" / "2026-10-31-completion-api"


def test_frozen_artifacts_recompute_balanced_counts_and_raw_byte_hashes():
    result = prep.validate(DATA)
    assert result["inclusion_counts"] == {"evaluation": 16, "calibration_is": 20, "calibration_oos": 20}
    assert result["artifact_sha256"]["evaluation.json"] == hashlib.sha256((DATA / "evaluation.json").read_bytes()).hexdigest()
    for name in ("evaluation", "calibration-is", "calibration-oos"):
        items = json.loads((DATA / (name + ".json")).read_bytes())["items"]
        for raw in items:
            item = ExperimentItem.from_dict(raw)
            value = prep.label_free_input(item)
            assert not hasattr(value, "label") and not hasattr(value, "cohort")
        assert {i["label"] for i in items} == {-1, 1}
        assert sum(i["label"] == 1 for i in items) == len(items) // 2


@pytest.mark.parametrize("filename", ["raw/SPY.chart.json", "evaluation.json", "selected-windows.json"])
def test_offline_validation_rejects_changed_raw_or_input_bytes(tmp_path, filename):
    destination = tmp_path / "inputs"
    shutil.copytree(DATA, destination)
    path = destination / filename
    path.write_bytes(path.read_bytes() + b"\r\n")
    with pytest.raises(ValueError, match="mismatch|differs"):
        prep.validate(destination)


def test_model_specific_cohorts_pass_current_cli_contract_without_calls(tmp_path):
    definitions = json.loads((DATA / "cohort-definitions.json").read_bytes())["models"]
    for model, cohorts in definitions.items():
        anthropic = model == "claude-opus-5-5"
        target = ModelConfig("anthropic_messages" if anthropic else "openai_chat",
                             "https://api.anthropic.com/v1/messages" if anthropic else "https://integrate.api.nvidia.com/v1/chat/completions",
                             model, max_retries=0, parameters={"max_tokens": 64} if anthropic else {})
        # This temporary zero-budget construction proves schema/date compatibility
        # only. It is not saved as a study protocol or used for any model calls.
        protocol = ExperimentProtocol(target, target, target,
            {"evaluation": hashlib.sha256((DATA / "evaluation.json").read_bytes()).hexdigest()},
            cohorts, draws=2, bootstrap_seed=0, bootstrap_count=1, task_metric="accuracy", utility_tolerance=0,
            audit_selection={"method": "temporary offline contract check only"}, request_budget=0,
            rewrite_method_version=REWRITE_METHOD_VERSION, ensemble_settings={"min_parsed": 1})
        path = tmp_path / "temporary-zero-budget-schema-check.json"
        path.write_text(json.dumps(protocol.to_dict()), encoding="utf-8")
        config = load_experiment_config(path, DATA / "evaluation.json", cutoff_registry=DATA / "cutoff-registry.json")
        assert config.historical["cutoff_comparison_eligible"] is True
        assert "proxies" in config.historical["interpretation"]


def test_exact_opus_cutoff_does_not_relax_legacy_post_cutoff_guard():
    root = DATA.parents[2]
    cutoffs = load_cutoffs(root / "data" / "cutoffs.yaml")
    assert cutoffs["claude-opus-5-5"] == date(2026, 6, 30)
    assert cutoffs["openai/gpt-oss-20b"] == date(2024, 6, 30)
    old_eval = load_eval_set(root / "tests" / "fixtures" / "tiny_eval.jsonl")
    with pytest.raises(CutoffViolation):
        assert_cutoff_safe(old_eval, ["claude-opus-5-5"], cutoffs)


def test_rounding_tie_excluded_and_tail_missing_session_reported():
    payload = chart()
    close = payload["chart"]["result"][0]["indicators"]["quote"][0]["close"]
    close[5], close[4] = 100.004, 100.001
    series = parsed(payload)
    rows, audit = prep.candidates(series, "post", "2026-07-01", "2026-10-30", cutoff=None, period="post")
    assert audit["flat"] >= 1
    assert all(row["as_of_close"] != row["outcome_close"] for row in rows)
    assert "2026-10-30" in series["missing_sessions"]
