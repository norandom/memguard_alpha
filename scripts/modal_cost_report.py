"""Local pilot accounting. Rows are attempts, control events or billing snapshots.

No token arrays cross this interface. Attempt position is the zero-based frozen
prompt index, not a generated-token position. All money is gross USD; credits
are a separate nonnegative amount. No cloud/account policy access occurs here.
"""

import math
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

from deploy.modal.manifest import Plan, Resources


def timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timezone_required")
    return parsed.astimezone(UTC)


def money(value: object) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("finite_nonnegative_usd_required")
    return float(value)


def source_reference(value: str) -> str:
    # References, never arbitrary provider messages or invoice contents.
    if not isinstance(value, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,100}", value):
        raise ValueError("safe_source_reference_required")
    return value


def validate_budget(record: dict, plan: Plan) -> dict:
    """Validate a reviewed monthly snapshot before any cloud mutation.

    An inaccessible ledger cannot establish remaining monthly headroom. In that
    case a dated operator-confirmed allowance is required and stays labeled.
    Environment compute budgets do not cover staging/storage/workspace charges.
    """
    if not isinstance(record, dict):
        raise ValueError("monthly_budget_record_required")
    checked = timestamp(record["checked_at"])
    month = datetime.now(UTC).strftime("%Y-%m")
    if record["period_month"] != month or checked.strftime("%Y-%m") != month or checked > datetime.now(UTC):
        raise ValueError("current_month_budget_required")
    if record["scope"] not in {"environment_compute", "workspace"}:
        raise ValueError("monthly_budget_scope_required")
    if record["workspace"] != plan.target.workspace:
        raise ValueError("budget_workspace_mismatch")
    if record["scope"] == "environment_compute" and record["environment"] != plan.target.environment:
        raise ValueError("budget_environment_mismatch")
    limit = money(record["monthly_limit_usd"])
    allowance = money(record["requested_allowance_usd"])
    if allowance <= 0 or allowance > limit:
        raise ValueError("insufficient_monthly_allowance")
    available = record["accounting_available"]
    if type(available) is not bool or type(record["operator_confirmed"]) is not bool:
        raise ValueError("explicit_accounting_status_required")
    prior = record.get("prior_usage_usd")
    remaining = None if prior is None else limit - money(prior)
    if remaining is not None and remaining < allowance:
        raise ValueError("insufficient_remaining_allowance")
    if (available and prior is None) or (not available and record["operator_confirmed"] is not True):
        raise ValueError("operator_confirmed_allowance_required")
    return {key: record[key] for key in (
        "scope", "workspace", "period_month", "checked_at", "monthly_limit_usd", "prior_usage_usd",
        "requested_allowance_usd", "accounting_available", "operator_confirmed") if key in record} | {
        "source": source_reference(record["source"]), "remaining_usd": remaining,
        "coverage": "compute_only" if record["scope"] == "environment_compute" else "workspace_wide",
        "enforcement": "monthly_snapshot_not_per_run_ceiling"}


@dataclass(frozen=True)
class BillingSnapshot:
    """One later accounting snapshot; snapshots replace totals, never sum them."""

    observed_gross_usd: float | None
    credits_usd: float | None
    source: str
    checked_at: str
    coverage: str


def append_billing(receipt: dict, entry: dict) -> None:
    """Append an allowlisted snapshot without touching attempt or control history."""
    timestamp(entry["checked_at"])
    if entry["coverage"] not in {"whole_pilot", "partial"}:
        raise ValueError("billing_coverage_required")
    snapshot = BillingSnapshot(
        None if entry["observed_gross_usd"] is None else money(entry["observed_gross_usd"]),
        None if entry["credits_usd"] is None else money(entry["credits_usd"]),
        source_reference(entry["source"]), entry["checked_at"], entry["coverage"])
    receipt.setdefault("billing_reconciliations", []).append(asdict(snapshot))


def cost_report(receipt: dict, cost: dict, resources: Resources) -> dict:
    """One summary row per run; estimates cannot substitute for observed billing."""
    attempts = receipt.get("attempts", [])
    usable = sum(row.get("stage") == "measurement" and row.get("status") == "eligible" for row in attempts)
    started, ready, finished = (receipt.get(key) for key in (
        "gpu_started_at_unix_s", "ready_at_unix_s", "finished_at_unix_s"))
    elapsed = None if started is None or finished is None else max(0, finished - started)
    rate = (cost["gpu_usd_per_second"] * resources.gpu_count
            + cost["physical_core_usd_per_second"] * resources.cpu_physical_cores
            + cost["memory_gib_usd_per_second"] * resources.memory_gib)
    billing = receipt.get("billing_reconciliations", [])
    latest = billing[-1] if billing else {}
    observed = latest.get("observed_gross_usd") if latest.get("coverage") == "whole_pilot" else None
    cleanup = receipt.get("lifecycle", {}).get("cleanup", {}).get("cleanup", "unobserved")
    return {
        "reported_at": datetime.now(UTC).isoformat(), "timing_source": "runner_wall_clock",
        "physical_inference_attempt_count": receipt.get("inference_posts", 0),
        "http_success_count": sum(200 <= row.get("http_status", 0) < 300 for row in attempts if row.get("http_status") is not None),
        "usable_measurement_count": usable,
        "failed_or_unknown_attempt_count": sum(row.get("status") != "eligible" for row in attempts),
        "cold_startup_seconds": None if started is None or ready is None else max(0, ready - started),
        "generation_seconds": sum(row.get("elapsed_seconds", 0) for row in attempts),
        "resource_estimate_gross_usd": None if elapsed is None else elapsed * rate,
        "estimated_total_gross_usd": None, "observed_gross_usd": observed,
        "credits_usd": latest.get("credits_usd"),
        "observed_usd_per_usable_measurement": None if observed is None or not usable else observed / usable,
        "cleanup_state": cleanup, "target_usd": cost["target_usd"], "hard_ceiling": False,
        "pricing_source": cost["pricing_source"], "pricing_checked_at": cost["pricing_checked_at"],
        "caveats": cost["caveats"],
        "estimate_basis": "configured_resource_rates_times_runner_elapsed_seconds_not_invoice",
        "excluded_cost_categories": ["staging", "build", "retained_storage", "expiry_controller", "egress", "unverified_cleanup_tail"],
    }
