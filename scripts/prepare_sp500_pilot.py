"""Freeze or verify the local S&P 500 mechanics pilot; no inference capability."""

import argparse
import hashlib
import json
import random
from datetime import date
from pathlib import Path

from scripts.prepare_completion_inputs import write_json
from scripts.sp500_inputs import (
    ARMS,
    HISTORY_LENGTH,
    SEED,
    VOLATILITY_FLOOR,
    cohort_ranges,
    load_snapshot,
    select_matched_dates,
    window_record,
    windows,
)

TARGETS = {
    "openai/gpt-oss-20b": {
        "cutoff": "2024-06-30", "protocol": "openai_chat",
        "endpoint": "https://integrate.api.nvidia.com/v1/chat/completions",
        "parameters": {"max_tokens": 2048, "temperature": 0},
        "credential_ref": "NVIDIA_API_KEY",
        "connect_timeout_s": 10, "socket_idle_timeout_s": 600,
    },
    "claude-opus-5-5": {
        "cutoff": "2026-06-30", "protocol": "anthropic_messages",
        "endpoint": "https://api.anthropic.com/v1/messages",
        "parameters": {"max_tokens": 2048},
        "credential_ref": "ANTHROPIC_API_KEY", "api_version": "2026-10-21",
        "connect_timeout_s": 120, "socket_idle_timeout_s": 120,
    },
}


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(source_directory: Path, completed_through: date):
    observations, source = load_snapshot(source_directory, completed_through)
    last_available = max(observation.day for observation in observations if observation.close is not None)
    items = []
    selection = {}
    for model, settings in TARGETS.items():
        ranges = cohort_ranges(date.fromisoformat(settings["cutoff"]), last_available)
        candidates = {}
        excluded = {}
        for cohort, (start, end) in ranges.items():
            candidates[cohort], excluded[cohort] = windows(observations, start, end)
        pairs = select_matched_dates(candidates["pre"], candidates["post"])
        model_items = [window_record(model, cohort, quartile, window)
                       for quartile, before, after in pairs
                       for cohort, window in (("pre", before), ("post", after))]
        items.extend(model_items)
        selection[model] = {
            "ranges": {cohort: [start.isoformat(), end.isoformat()] for cohort, (start, end) in ranges.items()},
            "candidate_counts": {cohort: len(rows) for cohort, rows in candidates.items()},
            "excluded_missing_sessions": excluded,
            "selected_dates": {cohort: sorted(item["target_date"] for item in model_items if item["cohort"] == cohort)
                               for cohort in ranges},
            "volatility_pairs": [
                {"quartile": quartile, "pre_date": before.target.day.isoformat(),
                 "post_date": after.target.day.isoformat(),
                 "pre_volatility": before.volatility, "post_volatility": after.volatility,
                 "post_to_pre_scale_ratio": max(after.volatility, VOLATILITY_FLOOR)
                 / max(before.volatility, VOLATILITY_FLOOR)}
                for quartile, before, after in pairs],
        }
    rng = random.Random(SEED)
    rng.shuffle(items)
    requests = []
    for item in items:
        arm_order = list(ARMS)
        rng.shuffle(arm_order)
        for arm in arm_order:
            prompt = item["prompts"][arm]
            requests.append({
                "operation_id": f"{item['item_id']}:{arm}", "item_id": item["item_id"],
                "model": item["model"], "arm": arm, "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            })
    overlaps = []
    for index, item in enumerate(items):
        dates = {row["date"] for row in item["history"]} | {item["target_date"]}
        for other in items[index + 1:]:
            common = sorted(dates & ({row["date"] for row in other["history"]} | {other["target_date"]}))
            if common:
                overlaps.append({"item_a": item["item_id"], "item_b": other["item_id"], "shared_dates": common})
    code_paths = [Path(__file__), Path(__file__).with_name("sp500_inputs.py"),
                  Path(__file__).with_name("sp500_scoring.py"),
                  Path(__file__).with_name("prepare_completion_inputs.py")]
    plan = {
        "schema": "recall_guard.sp500_mechanics.v1", "status": "prepared_not_executed",
        "completed_through": completed_through.isoformat(), "last_available": last_available.isoformat(),
        "source": source, "source_rows": len(observations),
        "missing_source_rows": sum(observation.close is None for observation in observations),
        "models": TARGETS, "seed": SEED, "history_length": HISTORY_LENGTH,
        "volatility": {"sample_standard_deviation_ddof": 1, "floor": VOLATILITY_FLOOR, "units": "daily_log_return"},
        "selection": selection, "split": "development_mechanics_only",
        "holdout_exclusion": "Exclude these full context/target date blocks from any later independent holdout.",
        "overlap_pairs": overlaps, "analysis": {
            "primary": "mean(pre anonymous-minus-identified normalized absolute error) minus mean(post)",
            "denominator": "common-usable pairs, accompanied by every arm's planned/parsed/abstained/failed counts",
            "confidence_interval": "not estimated for this tiny mechanics pilot",
            "live_progression_gate": "at least 3/4 common-usable continuation pairs in each model/cohort",
            "direct_recall_tolerances": [0.001, 0.01], "exact_precision_decimal_places": 2,
            "interpretation": "exploratory screening only; no efficacy or training-membership claim",
        },
        "budget": {
            "maximum_physical_requests": 50, "eligibility": 2, "measurement": len(requests),
            "per_model": 25, "max_retries": 0, "max_workers": 1, "min_call_interval_s": 2,
            "admission_window_s_per_provider": 1800,
            "in_flight_policy": "Existing bounded transport; admission deadline is not a hard I/O interrupt",
            "status_poll_requests": 0, "stop_on": ["credential", "billing", "quota", "transport_failure", "deadline"],
            "failure_scope": "Stop affected provider; the other may proceed within its own cap. No budget transfer.",
            "max_output_tokens_per_request": 2048, "authorization": "pending_concrete_pilot_approval",
            "anthropic_planning_input_tokens_per_request": 4096,
            "anthropic_estimate_usd": 1.4336,
            "pricing_assumption_usd_per_million": {"input": 4, "output": 20},
            "estimate_is_hard_billing_cap": False,
            "nim_cost": None, "nim_cost_note": "Existing free-service access; no contractual price guarantee assumed",
            "pricing_source": "https://platform.claude.com/docs/en/models/opus-5-5/overview",
        },
        "eligibility_prompt": 'Return only {"value": 1}.',
        "prompt_bytes_max": max(len(request["prompt"].encode()) for request in requests),
        "code_sha256": {path.name: sha256_file(path) for path in code_paths},
        "model_requests_executed": 0,
    }
    canonical = json.dumps({"plan": plan, "items": items, "requests": requests}, sort_keys=True).encode()
    plan["bundle_sha256"] = hashlib.sha256(canonical).hexdigest()
    return plan, items, requests


def freeze(source_directory, output_directory, completed_through):
    plan, items, requests = build(source_directory, completed_through)
    output_directory.mkdir(parents=True, exist_ok=False)
    for name, value in (("plan.json", plan), ("items.json", items), ("requests.json", requests)):
        write_json(output_directory / name, value)
    print(json.dumps({"bundle_sha256": plan["bundle_sha256"], "selection": plan["selection"],
                      "request_cap": plan["budget"]["maximum_physical_requests"],
                      "latest_available": plan["last_available"], "prompt_bytes_max": plan["prompt_bytes_max"]}))


def validate(source_directory, output_directory, completed_through):
    expected = build(source_directory, completed_through)
    for name, value in zip(("plan.json", "items.json", "requests.json"), expected, strict=True):
        actual = json.loads((output_directory / name).read_text(encoding="utf-8"))
        # JSON round trip normalizes tuple-valued rebased history.
        if actual != json.loads(json.dumps(value)):
            raise ValueError(f"Frozen artifact mismatch: {name}")
    print("Frozen inputs, source hashes, code identities and request plan verified; no inference.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "validate"))
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--completed-through", type=date.fromisoformat, required=True)
    args = parser.parse_args()
    {"freeze": freeze, "validate": validate}[args.action](args.source, args.output, args.completed_through)
