"""Execute an explicitly approved frozen pilot; never resume ambiguous calls."""

import argparse
import json
import os
import time
from datetime import date
from decimal import Decimal
from pathlib import Path

from dotenv import dotenv_values

from recall_guard.core.completion import ModelConfig
from recall_guard.core.provider_anthropic import ANTHROPIC_API_VERSION, AnthropicClient
from recall_guard.core.providers import OpenAICompatibleClient
from recall_guard.core.readiness_types import ReadinessPolicy
from recall_guard.core.request_control import CallBudget
from scripts.prepare_completion_inputs import write_json
from scripts.prepare_sp500_pilot import sha256_file, validate
from scripts.sp500_scoring import parse_answer, score_answer, summarize_scores


def make_client(model, settings, environment):
    readiness = None
    if settings["protocol"] == "openai_chat":
        readiness = ReadinessPolicy(
            connect_timeout_s=settings["connect_timeout_s"],
            socket_idle_timeout_s=settings["socket_idle_timeout_s"],
            max_wait_s=600, allow_ambiguous_replay=False,
        )
    elif settings["api_version"] != ANTHROPIC_API_VERSION:
        raise ValueError("Frozen Anthropic API version differs from adapter")
    config = ModelConfig(
        protocol=settings["protocol"], endpoint=settings["endpoint"], model=model,
        provider="nvidia_nim" if readiness is not None else "anthropic",
        credential_ref=settings["credential_ref"], parameters=settings["parameters"],
        timeout_s=settings["socket_idle_timeout_s"], max_retries=0,
        min_call_interval_s=2, readiness=readiness,
    )
    client_type = AnthropicClient if settings["protocol"] == "anthropic_messages" else OpenAICompatibleClient
    return client_type(config, CallBudget(25), environment=environment)


def append_event(path, event):
    """Flush intent before transport; an interrupted intent remains ambiguous."""
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def execute(plan, items, requests, output, environment, *, client_factory=make_client,
            clock=time.monotonic, sleep=time.sleep, only_model=None):
    if (plan["budget"]["maximum_physical_requests"] != 50
            or len(requests) != 48 or len(plan["models"]) != 2
            or any(sum(row["model"] == model for row in requests) != 24 for model in plan["models"])):
        raise ValueError("Expected the frozen two-model 50-request pilot")
    if only_model is not None and only_model not in plan["models"]:
        raise ValueError("Selected model is absent from the frozen plan")
    selected_models = [only_model] if only_model is not None else list(plan["models"])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "execution.json", {
        "bundle_sha256": plan["bundle_sha256"], "runner_sha256": sha256_file(Path(__file__)),
        "approved_max_requests": 25 * len(selected_models), "resume_allowed": False,
        "selected_models": selected_models,
        "in_flight_policy": plan["budget"]["in_flight_policy"],
    })
    items_by_id = {item["item_id"]: item for item in items}
    scores, provider_results = [], {}
    last_start = None
    for model, settings in plan["models"].items():
        if model not in selected_models:
            provider_results[model] = {"status": "not_selected", "physical_requests": 0}
            continue
        result = {"status": "starting", "physical_requests": 0}
        provider_results[model] = result
        try:
            client = client_factory(model, settings, environment)
        except (ValueError, TypeError):
            result["status"] = "configuration_or_credential_missing"
            continue
        deadline = clock() + plan["budget"]["admission_window_s_per_provider"]
        scheduled = [{"operation_id": f"{model}:eligibility", "prompt": plan["eligibility_prompt"]}]
        scheduled.extend(row for row in requests if row["model"] == model)
        for operation in scheduled:
            if last_start is not None:
                sleep(max(0, 2 - (clock() - last_start)))
            if clock() >= deadline:
                result["status"] = "admission_deadline"
                break
            operation_id = operation["operation_id"]
            append_event(output / "events.jsonl", {"operation_id": operation_id, "state": "intent"})
            last_start = clock()
            print(json.dumps({"model": model, "operation": operation_id, "state": "starting"}), flush=True)
            try:
                response = client.generate_for_stage(operation["prompt"], stage=operation_id)
            except Exception:
                # Provider event ledger holds sanitized failure details. Never export arbitrary exception text.
                append_event(output / "events.jsonl", {"operation_id": operation_id, "state": "failed_or_unknown"})
                result["status"] = "provider_failure"
                if "item_id" in operation:
                    scores.append({"item_id": operation["item_id"], "arm": operation["arm"], "status": "failed"})
                break
            finally:
                result["physical_requests"] = client.budget.used
                write_json(output / f"ledger-{list(plan['models']).index(model)}.json",
                           [event.to_dict() for event in client.budget.events])
            complete = response.finish_reason in {"stop", "end_turn"}
            append_event(output / "events.jsonl", {
                "operation_id": operation_id, "state": "response", "content": response.content,
                "finish_reason": response.finish_reason, "request_id": response.request_id,
                "usage": dict(response.usage) if response.usage is not None else None,
                "requested_settings": dict(response.requested_settings),
                "confirmed_settings": dict(response.confirmed_settings),
            })
            if "item_id" not in operation:
                answer = parse_answer(response.content)
                if not complete or answer["status"] != "parsed" or Decimal(answer["value"]) != 1:
                    result["status"] = "eligibility_rejected"
                    break
                result["status"] = "eligible"
            else:
                score = score_answer(items_by_id[operation["item_id"]], operation["arm"],
                                     response.content if complete else "")
                scores.append(score)
                write_json(output / "scores.json", scores)
        else:
            result["status"] = "completed"
    summary = {"providers": provider_results, "models": summarize_scores(items, scores),
               "physical_requests": sum(result["physical_requests"] for result in provider_results.values()),
               "cost_usd": None, "cost_note": "Unknown until reconciled with reported usage/billing"}
    write_json(output / "scores.json", scores)
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--credentials", type=Path, required=True)
    parser.add_argument("--model", help="Run only this frozen model in a fresh output directory")
    parser.add_argument("--approved-max-requests", type=int, choices=[25, 50], required=True)
    args = parser.parse_args()
    if args.approved_max_requests != (25 if args.model is not None else 50):
        parser.error("Use cap 25 with --model, or cap 50 for the complete pilot")
    plan = json.loads((args.bundle / "plan.json").read_text(encoding="utf-8"))
    validate(args.source, args.bundle, date.fromisoformat(plan["completed_through"]))
    items = json.loads((args.bundle / "items.json").read_text(encoding="utf-8"))
    requests = json.loads((args.bundle / "requests.json").read_text(encoding="utf-8"))
    environment = dotenv_values(args.credentials, interpolate=False)
    execute(plan, items, requests, args.output, environment, only_model=args.model)


if __name__ == "__main__":
    main()
