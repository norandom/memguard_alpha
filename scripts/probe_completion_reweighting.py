"""Bounded completion-sample feasibility spike; never claims token-level decoding."""

import argparse
import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path

from dotenv import dotenv_values

from recall_guard.core.completion import FeatureChoices, ModelConfig
from recall_guard.core.provider_anthropic import AnthropicClient
from recall_guard.core.request_control import CallBudget, RequestError

REQUEST_CAP = 17
DRAWS_PER_ITEM = 8
# Explicit synthetic categorical auxiliary counts; no temporal corpus was fitted.
OLDER_COUNTS = {"A": 80, "B": 20}
NEWER_COUNTS = {"A": 20, "B": 80}
PROMPTS = {
    "named": (
        "Consider a hypothetical forecast made before the 1998 football World Cup final. "
        "A is Brazil and B is France. For this hypothetical scenario, the supplied "
        "pre-match evidence assigns A an 80% chance and B a 20% chance. "
        "Ignore the actual historical result. Draw one simulated outcome from these "
        "probabilities. Return only A or B, without explanation."
    ),
    "anonymous": (
        "Consider a hypothetical forecast made before a football final. "
        "A is Team Cedar and B is Team Birch. For this hypothetical scenario, the supplied "
        "pre-match evidence assigns A an 80% chance and B a 20% chance. "
        "Draw one simulated outcome from these probabilities. "
        "Return only A or B, without explanation."
    ),
}


def summarize(answers, alpha):
    """Each observation is one independent API completion; categories are A/B."""
    if not answers or any(answer not in OLDER_COUNTS for answer in answers):
        raise ValueError("Expected nonempty A/B answers")
    weights = [(OLDER_COUNTS[answer] / NEWER_COUNTS[answer]) ** alpha for answer in answers]
    weight_sum = sum(weights)
    probability_a = sum(weight for answer, weight in zip(answers, weights, strict=True) if answer == "A") / weight_sum
    return {
        "alpha": alpha,
        "probability_a": probability_a,
        "effective_sample_size": weight_sum**2 / sum(weight**2 for weight in weights),
        "largest_normalized_weight": max(weights) / weight_sum,
    }


def persist(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run(output_directory):
    output_directory.mkdir(parents=True, exist_ok=False)
    budget = CallBudget(REQUEST_CAP)
    config = ModelConfig(
        "anthropic_messages", "https://api.anthropic.com/v1/messages",
        "claude-opus-5-5", provider="anthropic", credential_ref="ANTHROPIC_API_KEY",
        parameters={"max_tokens": 256}, timeout_s=60, max_retries=0, min_call_interval_s=2,
    )
    plan = {
        "created_at": datetime.now(UTC).isoformat(), "config": config.to_dict(),
        "request_cap": REQUEST_CAP, "draws_per_item": DRAWS_PER_ITEM,
        "output_token_cap_per_request": 256, "admission_deadline_seconds": 300,
        "auxiliary_provenance": "synthetic categorical counts; not measured temporal exposure",
        "older_counts": OLDER_COUNTS, "newer_counts": NEWER_COUNTS,
        "prompts": PROMPTS, "alpha_grid": [0, 0.5, 1],
        "support_gate": "at least two observed samples in each category per item",
        "weight_gate": "alpha=1 ESS >=4 and probability change >=0.10 per item",
        "interpretation": "mechanics only; no target logprobs, calibration fit, or efficacy claim",
        "cost": None,
    }
    persist(output_directory / "plan.json", plan)
    records = []
    results = {}
    status = "not_started"
    failure = None
    deadline = time.monotonic() + 300

    def call(prompt, stage):
        if time.monotonic() >= deadline:
            raise TimeoutError("admission deadline reached")
        # Flush intent first; an interruption must not invite a silent replay.
        records.append({"stage": stage, "state": "started"})
        persist(output_directory / "responses.json", records)
        response = client.generate_for_stage(prompt, stage=stage)
        records[-1].update(
            state="completed", answer=response.content.strip(),
            finish_reason=response.finish_reason, usage=dict(response.usage or {}),
            request_id=response.request_id,
        )
        persist(output_directory / "responses.json", records)
        persist(output_directory / "requests.json", [event.to_dict() for event in budget.events])
        print(json.dumps({"completed": len(records), "stage": stage}), flush=True)
        if response.finish_reason != "end_turn":
            raise ValueError("incomplete response")
        return response.content.strip()

    try:
        client = AnthropicClient(config, budget, features=FeatureChoices(), environment=dotenv_values("/credentials/.env"))
        if call("Return only OK.", "eligibility") != "OK":
            raise ValueError("eligibility output not usable")
        for item, prompt in PROMPTS.items():
            answers = [call(prompt, item) for _ in range(DRAWS_PER_ITEM)]
            if any(answer not in OLDER_COUNTS for answer in answers):
                raise ValueError("noncategorical answer")
            summaries = [summarize(answers, alpha) for alpha in plan["alpha_grid"]]
            counts = {category: answers.count(category) for category in OLDER_COUNTS}
            results[item] = {"counts": counts, "summaries": summaries,
                             "baseline_probability_a_se": math.sqrt((counts["A"] / 8) * (counts["B"] / 8) / 8)}
            if min(counts.values()) < 2:
                status = "stop_insufficient_sample_support"
                break
            if summaries[-1]["effective_sample_size"] < 4 or abs(
                summaries[-1]["probability_a"] - summaries[0]["probability_a"]
            ) < 0.10:
                status = "stop_weight_gate"
                break
        else:
            status = "mechanics_gate_passed_not_efficacy"
    except (RequestError, ValueError, TimeoutError) as error:
        status = "stopped_error"
        # Do not serialize arbitrary exception strings or request headers.
        failure = {"type": type(error).__name__, "code": getattr(error, "code", None),
                   "http_status": getattr(error, "status_code", None)}
    finally:
        receipt = {"status": status, "failure": failure, "physical_requests": len(budget.events),
                   "request_cap": REQUEST_CAP, "results": results,
                   "unperformed_requests": REQUEST_CAP - len(budget.events),
                   "cost": None, "mcs": None, "mitigation_efficacy": "not_tested"}
        persist(output_directory / "requests.json", [event.to_dict() for event in budget.events])
        persist(output_directory / "receipt.json", receipt)
        print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output)
