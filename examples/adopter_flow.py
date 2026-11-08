"""Adopter flow: choose a profile, preflight it, accept a fallback, run, report.

PowerShell: uv run --no-sync python examples/adopter_flow.py --fixture

``--fixture`` needs no keys and sends nothing over the network: the host memory
reading (13.4 GiB free), the read-only Ollama metadata and every chat response
are fixed, synthetic stand-ins. The provider client, preparation, RecallWorkflow
execution, fidelity checks and request accounting are the real ones. The output
shows wiring only, never model quality or mitigation effect. Without
``--fixture`` the script only prints the chosen profile and its declared host
requirements; it reads no host memory and contacts nothing.
"""

import argparse
import json
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich.console import Console

from recall_guard.core.completion import CapabilityEvidence, FeatureChoices
from recall_guard.core.profiles import GIB, PreflightError, connect, load_profile, preflight_host
from recall_guard.core.provider_probe import TOKEN_CONTROLS
from recall_guard.core.providers import OpenAICompatibleClient
from recall_guard.core.request_control import CallBudget
from recall_guard.harness.request_failures import describe_failure, render_failure_panel
from recall_guard.harness.rewrite_fidelity import VERIFICATION_TEMPLATE
from recall_guard.harness.rewriting import REWRITE_TEMPLATE, EvidenceRewriter, RewriteInput, RewritePolicy
from recall_guard.harness.workflow_plan import CompletionFallbackPolicy, WorkflowRequest
from recall_guard.harness.workflow_preparation import prepare_workflow

REQUESTED_PROFILE = "ollama-gpt-oss-20b"
FIXTURE_FREE_RAM = int(13.4 * GIB)
FIXTURE_TOTAL_RAM = 32 * GIB
_REAL_GEMMA = ("Real gemma3:4b on Ollama returned aligned logprobs "
               "(docs/results/2026-11-06-local-model-smoke.md).")
#: One truthful line per scenario: every token-evidence outcome here is simulated.
SCENARIO_NOTES = {
    "observed_missing_tokens": "Simulated: the fixture endpoint omits logprobs to demonstrate the authorized "
                               "completion-only fallback. " + _REAL_GEMMA,
    "declared_missing_tokens": "Simulated: the adopter declares logprobs unsupported, so the fallback is chosen "
                               "before any target probe. " + _REAL_GEMMA,
    "no_fallback_authorized": "Simulated: the fixture omits logprobs and no fallback is authorized, so preparation "
                              "stops before evaluation. " + _REAL_GEMMA,
    "token_evidence_without_fit": "Simulated: the fixture returns logprobs, but no matching fitted calibration is "
                                  "supplied, so MCS is refused rather than downgraded.",
}
SCENARIOS = tuple(SCENARIO_NOTES)
ACCEPTED = RewriteInput("item-accepted", "Return only up or down.",
                        "Acme rose 2% on 2024-01-02; estimates remain uncertain.", "2024-01-02")
REJECTED = RewriteInput("item-rejected", "Return only up or down.",
                        "Globex fell 4% on 2024-01-03 after guidance.", "2024-01-03")
#: Synthetic rewriter output per source: the second changes a quantity (4% -> 5%).
REWRITES = {ACCEPTED.evidence: "Let's review: " + ACCEPTED.evidence,
            REJECTED.evidence: "Let's review: Globex fell 5% on 2024-01-03 after guidance."}
VERDICT = json.dumps({"equivalent": True, "checks": dict.fromkeys((
    "claims", "entities", "dates", "quantities", "units", "uncertainty", "qualifications", "attribution",
    "instruction_injection"), True), "discrepancies": []})


class FixtureHost:
    """Synthetic Ollama host: fixed free RAM, read-only metadata, fixed chat replies."""

    def __init__(self, *, token_evidence=False):
        self.token_evidence = token_evidence
        self.failed_once = False

    def memory(self):
        return FIXTURE_FREE_RAM, FIXTURE_TOTAL_RAM

    def metadata(self, url):
        if url.endswith("/api/ps"):
            return {"models": []}  # Nothing already loaded, so the free-RAM gate applies.
        if url.endswith("/api/version"):
            return {"version": "0.40.11"}
        return {"models": [{"name": "gemma3:4b", "digest": "0" * 64}]}

    def post(self, payload):
        prompt = payload["messages"][-1]["content"]
        if prompt.startswith(REWRITE_TEMPLATE):
            if not self.failed_once:  # One transient 503 shows a retry counted once.
                self.failed_once = True
                return _reply(503, {"error": {"message": "fixture overloaded", "code": "service_unavailable"}})
            content = next(text for source, text in REWRITES.items() if json.dumps(source)[1:-1] in prompt)
        elif prompt.startswith(VERIFICATION_TEMPLATE):
            content = VERDICT
        else:
            content = "OK" if prompt == "Reply with the word OK." else "up"
        choice = {"message": {"content": content}, "finish_reason": "stop"}
        if self.token_evidence and payload.get("logprobs"):
            choice["logprobs"] = {"content": [{"token": content, "logprob": -0.1, "top_logprobs": [
                {"token": content, "logprob": -0.1}, {"token": "x", "logprob": -2.3}]}]}
        return _reply(200, {"choices": [choice], "usage": {"prompt_tokens": len(prompt) // 4,
                                                             "completion_tokens": 1}})


def _reply(status, body):
    return SimpleNamespace(status_code=status, headers={}, json=lambda: body, text=json.dumps(body))


def preflight_with_fallback(requested, host, console):
    """Preflight the requested profile; on a refusal, take the first proposal that passes."""
    attempts = []

    def check(profile_id):
        report = preflight_host(profile_id, environment={}, memory_reader=host.memory, http_get=host.metadata)
        attempts.append({"profile": profile_id, "passed": report.passed, "reason": report.reason,
                         "free_ram_gib": None if report.measured_free_ram_bytes is None
                         else round(report.measured_free_ram_bytes / GIB, 1),
                         "required_ram_gib": None if report.required_free_ram_bytes is None
                         else round(report.required_free_ram_bytes / GIB, 1)})
        return report

    first = check(requested)
    if first.passed:
        return load_profile(requested), attempts
    console.print(render_failure_panel(describe_failure(PreflightError(first), stage="host preflight")))
    for candidate in first.fallback_profile_ids:
        if check(candidate).passed:
            return load_profile(candidate), attempts
    return None, attempts


def build_request(profile, scenario):
    """Request MCS scoring with rewriting; the fallback policy is an explicit choice."""
    target = profile.model_config
    if scenario == "declared_missing_tokens":
        target = replace(target, capabilities=dict(target.capabilities) | {
            "logprobs": CapabilityEvidence("unsupported", "Adopter declaration for this host")})
    auxiliary = replace(profile.model_config, max_retries=1, retry_backoff_s=0.0, parameters={
        name: value for name, value in profile.model_config.parameters.items() if name not in TOKEN_CONTROLS})

    def factory(config, budget, *, features):
        # Same connect() factory as live use; preflight already ran above.
        return connect(replace(profile, model_config=config), budget, features=features, environment={},
                       enforce_preflight=False)

    template = factory(auxiliary, CallBudget(0), features=FeatureChoices())
    policy = RewritePolicy(EvidenceRewriter(template), factory(auxiliary, CallBudget(0), features=FeatureChoices()))
    request = WorkflowRequest({"target": target, "rewriter": auxiliary, "verifier": auxiliary},
                              FeatureChoices(mia_scoring=True, rewriting=True), policy.intervention_id)
    fallback = (None if scenario == "no_fallback_authorized"
                else CompletionFallbackPolicy(FeatureChoices(rewriting=True), omit_logprob_controls=True))
    return request, fallback, factory, policy


def _stages(result):
    rewrite = result.rewrite
    checks = {} if rewrite is None else rewrite.fidelity_checks
    generation = checks.get("generation", {})
    deterministic = checks.get("deterministic", {})
    semantic = checks.get("semantic", {})
    target_ran = result.stage == "target"
    return {"preflight": "ready" if result.preflight and result.preflight.ready_for_batch else "failed",
            "rewrite": generation.get("status", "skipped"),
            "fidelity_deterministic": deterministic.get("status", "skipped"),
            "fidelity_semantic": semantic.get("status", "skipped"),
            "target": result.status.value if target_ran else "skipped"}


def _requests(budget, preparation):
    prepared = {record.event.call_id for record in preparation.artifact.requests}
    rows = []
    for event in budget.events:
        kind = ("probe" if event.stage.split("/")[-1] == "probe" or "/probe/" in event.stage
                else "retry" if event.attempt > 1 else "generation")
        rows.append((event, "preparation" if event.call_id in prepared else "evaluation", kind))
    usage = Counter()
    for event, _, _ in rows:
        usage.update(event.usage or {})
    unknown_usage = sum(event.usage is None for event, _, _ in rows)
    return {
        "physical_requests": len(rows), "unique_call_ids": len({event.call_id for event, _, _ in rows}),
        "budget_used": budget.used, "budget_limit": budget.limit,
        "by_phase_and_kind": {f"{phase}/{kind}": count for (phase, kind), count in sorted(Counter(
            (phase, kind) for _, phase, kind in rows).items())},
        "by_stage": dict(sorted(Counter(event.stage.removeprefix(preparation.artifact.execution_id + "/")
                                        for event, _, _ in rows).items())),
        "statuses": dict(sorted(Counter(event.status for event, _, _ in rows).items())),
        "usage": {"reported_counters": dict(sorted(usage.items())), "requests_without_usage": unknown_usage,
                  "complete": unknown_usage == 0},
        "cost": {"total": None, "status": "unknown", "recorded_requests": sum(e.cost is not None for e, _, _ in rows),
                 "reason": "No price is recorded at the request boundary; nothing is estimated."},
    }


def run_fixture(scenario="observed_missing_tokens", *, console=None):
    """Run the complete offline flow and return one compact, JSON-ready run record."""
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {SCENARIOS}")
    console = console or Console(file=sys.stdout, highlight=False)
    host = FixtureHost(token_evidence=scenario == "token_evidence_without_fit")
    with patch.object(OpenAICompatibleClient, "_post", lambda client, payload: host.post(payload)):
        profile, attempts = preflight_with_fallback(REQUESTED_PROFILE, host, console)
        record = {"record": "adopter_flow", "evidence": "synthetic fixture; no network, no model quality claim",
                  "scenario": scenario, "scenario_note": SCENARIO_NOTES[scenario], "requested_profile": REQUESTED_PROFILE, "preflight": attempts,
                  "selected_profile": None if profile is None else profile.profile_id}
        if profile is None:
            return record | {"status": "no_profile_passed_preflight"}
        request, fallback, factory, policy = build_request(profile, scenario)
        budget = CallBudget(16)
        preparation = prepare_workflow(request, fallback, factory, budget, rewrite_policy=policy)
        plan = preparation.plan
        selected = plan.selected.roles["target"]
        record |= {
            "model": selected.model, "endpoint": selected.endpoint, "hosting_mode": profile.hosting_mode.value,
            "settings": {"requested": dict(plan.requested.roles["target"].parameters),
                         "selected": dict(selected.parameters)},
            "features": {"requested": plan.requested.to_dict()["features"],
                         "selected": plan.selected.to_dict()["features"]},
            "fallback_policy": None if plan.fallback_policy is None else plan.fallback_policy.to_dict(),
            "policy_id": plan.policy_id,
            "method": {"requested": plan.requested.method, "selected": plan.selected.method},
            "omission": {"reason": plan.reason, "omitted_controls": list(plan.omitted_controls),
                         "mia_unavailable_reason": plan.mia_unavailable_reason},
            "preparation": {"ready": preparation.ready, "stopping_reason": preparation.artifact.stopping_reason,
                            "probes": {role: len(items) for role, items in preparation.artifact.evidence.items()}},
        }
        items = []
        if preparation.ready:
            for item in (ACCEPTED, REJECTED):
                result = preparation.workflow.run(item)
                # Observed: the method that actually reached the target; None when it was skipped.
                observed = result.method if result.stage == "target" else None
                items.append({"item_id": item.item_id, "observed_method": observed,
                              "status": result.status.value, "stage": result.stage, "answer": result.answer,
                              "failure_kind": result.failure_kind, "stages": _stages(result),
                              "mcs": result.unavailable_reasons.get("p_memorized")})
        record["items"] = items
        record["requests"] = _requests(budget, preparation)
    return record | {"status": "completed" if preparation.ready else "stopped_before_evaluation"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fixture", action="store_true", help="Run the full flow against synthetic fixtures")
    parser.add_argument("--scenario", choices=SCENARIOS, default=SCENARIOS[0], help="Fixture scenario: " + " ".join(
        f"{name}: {note}" for name, note in SCENARIO_NOTES.items()))
    args = parser.parse_args(argv)
    if not args.fixture:
        profile = load_profile(REQUESTED_PROFILE)
        print(json.dumps({"status": "dry_run", "network_requests": 0, "profile": profile.profile_id,
                          "model": profile.model_config.model, "endpoint": profile.model_config.endpoint,
                          "host_requirements": profile.host_requirements.to_dict(),
                          "next": "add --fixture for the offline end-to-end flow"}, indent=2))
        return 0
    record = run_fixture(args.scenario)  # The friendly preflight panel prints first.
    print(json.dumps(record, indent=2))
    return 0 if record["status"] in {"completed", "stopped_before_evaluation"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
