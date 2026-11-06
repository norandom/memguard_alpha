"""Offline accounting fixtures: no billing SDK or provider policy mutations."""

import copy
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from deploy.modal.manifest import load_plan
from scripts import modal_cost_report as report

MANIFEST = Path(__file__).resolve().parents[2] / "configs/modal/gpt-oss-20b-pilot.json"


def budget():
    return dict(scope="environment_compute", environment="recall-guard-pilot", workspace="test",
                period_month=datetime.now(UTC).strftime("%Y-%m"), checked_at=datetime.now(UTC).isoformat(), source="operator_budget_review",
                monthly_limit_usd=20, prior_usage_usd=16, requested_allowance_usd=3,
                accounting_available=True, operator_confirmed=True)


def test_budget_checks_prior_usage_and_scope():
    plan = load_plan(MANIFEST)
    from dataclasses import replace
    plan = replace(plan, target=replace(plan.target, workspace="test"))
    assert report.validate_budget(budget(), plan)["remaining_usd"] == 4
    for changes in ({"prior_usage_usd": 18}, {"monthly_limit_usd": float("nan")},
                    {"scope": "per_run"}, {"environment": "other"}, {"period_month": "2026-10"},
                    {"accounting_available": False, "operator_confirmed": False}):
        with pytest.raises(ValueError):
            report.validate_budget(budget() | changes, plan)
    confirmed = budget() | {"accounting_available": False, "prior_usage_usd": None}
    assert report.validate_budget(confirmed, plan)["remaining_usd"] is None


def test_report_separates_http_usable_and_unknown_total_cost():
    receipt = {"inference_posts": 3, "attempts": [
        {"status": "eligible", "stage": "eligibility", "http_status": 200, "elapsed_seconds": 2},
        {"status": "eligible", "stage": "measurement", "http_status": 200, "elapsed_seconds": 3},
        {"status": "ineligible", "stage": "measurement", "http_status": 200, "elapsed_seconds": 4}],
        "gpu_started_at_unix_s": 100, "ready_at_unix_s": 110, "finished_at_unix_s": 120,
        "lifecycle": {"cleanup": {"cleanup": "unverified"}}}
    cost = json.loads(MANIFEST.read_text())["cost"]
    result = report.cost_report(receipt, cost, load_plan(MANIFEST).resources)
    assert result["http_success_count"] == 3
    assert result["usable_measurement_count"] == 1
    assert result["cold_startup_seconds"] == 10
    assert result["generation_seconds"] == 9
    assert result["resource_estimate_gross_usd"] == pytest.approx(20 * (.000583 + 4*.0000131 + 32*.00000222))
    assert result["observed_gross_usd"] is result["credits_usd"] is None
    assert result["observed_usd_per_usable_measurement"] is None
    assert result["estimated_total_gross_usd"] is None
    assert result["cleanup_state"] == "unverified"
    assert result["hard_ceiling"] is False
    assert result["pricing_source"] == cost["pricing_source"]


@pytest.mark.parametrize("observed,usable,expected", [(None, True, None), (0, True, 0), (2, True, 2), (2, False, None)])
def test_cost_per_usable_uses_only_known_whole_pilot_cost(observed, usable, expected):
    receipt = {"attempts": [{"stage": "measurement", "status": "eligible" if usable else "ineligible"}]}
    report.append_billing(receipt, dict(observed_gross_usd=observed, credits_usd=None,
                         source="invoice_review", checked_at="2026-11-06T11:00:00Z", coverage="whole_pilot"))
    result = report.cost_report(receipt, json.loads(MANIFEST.read_text())["cost"], load_plan(MANIFEST).resources)
    assert result["observed_usd_per_usable_measurement"] == expected


def test_billing_appends_without_mutating_attempts_or_exporting_extra_fields():
    receipt = {"attempts": [{"status": "eligible"}]}
    original = copy.deepcopy(receipt["attempts"])
    entry = dict(observed_gross_usd=2, credits_usd=1, source="invoice_review",
                 checked_at="2026-11-06T11:00:00Z", coverage="whole_pilot",
                 credential="secret", generated_text="private answer")
    report.append_billing(receipt, entry)
    report.append_billing(receipt, entry | {"observed_gross_usd": 3})
    assert receipt["attempts"] == original
    assert len(receipt["billing_reconciliations"]) == 2
    assert "secret" not in json.dumps(receipt) and "private answer" not in json.dumps(receipt)
    result = report.cost_report(receipt, json.loads(MANIFEST.read_text())["cost"], load_plan(MANIFEST).resources)
    assert result["observed_gross_usd"] == 3
    assert result["observed_usd_per_usable_measurement"] is None
