"""Freeze the dated pilot and explicitly probe eligibility; no batch execution.

``freeze``, ``validate`` and ``verify-receipt`` are offline and environment-free.
Only explicit ``probe``/``recover`` resolve the two named environment variables.
This helper never reads an env file, automatically replays a probe, or fits/runs
the study. The sole operator recovery retains and counts the failed history.
"""

import argparse
import json
import os
import re
from dataclasses import fields, replace
from datetime import UTC, datetime
from pathlib import Path

from recall_guard.core.completion import CapabilityEvidence, CapabilityReport, FeatureChoices, ModelConfig
from recall_guard.core.ensemble import EnsembleSpec
from recall_guard.core.provider_anthropic import ANTHROPIC_API_VERSION, AnthropicClient
from recall_guard.core.providers import OpenAICompatibleClient
from recall_guard.core.request_control import BudgetExhaustedError, CallBudget, RequestError
from recall_guard.harness import rewrite_fidelity
from recall_guard.harness.intervention_types import ExperimentProtocol, text_hash
from recall_guard.harness.rewriting import REWRITE_METHOD_VERSION, REWRITE_TEMPLATE
from scripts import prepare_completion_inputs as inputs

DEFAULT_DIRECTORY = Path("data/experiments/2026-10-31-completion-api")
PROBE_PROMPT = "Reply with the word OK."
PHASE_CAPS = {"eligibility": 4, "calibration": 220, "nim_evaluation": 144, "anthropic_evaluation": 144}


def _read(path):
    return json.loads(path.read_bytes())


def _write_new(path, value, *, exact_resume=False):
    """Exclusive UTF-8/LF write; differing existing bytes are never replaced."""
    content = inputs._json_bytes(value)
    if exact_resume and path.exists() and path.read_bytes() == content:
        return
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def model_configs():
    """Public, key-free configurations; no clients or environment lookup."""
    transport = {"timeout_s": 90, "max_retries": 0, "min_call_interval_s": 1.5}
    return {
        "nim": ModelConfig("openai_chat", "https://integrate.api.nvidia.com/v1/chat/completions",
                           "openai/gpt-oss-20b", provider="nvidia_nim", **transport,
                           parameters={"max_tokens": 2048, "temperature": 0.0, "logprobs": True, "top_logprobs": 20}),
        "anthropic": ModelConfig("anthropic_messages", "https://api.anthropic.com/v1/messages",
                                 "claude-opus-5-5", provider="anthropic", **transport, parameters={"max_tokens": 2048}),
    }


def _method(config):
    public = config.to_dict()
    semantic = {name: public[name] for name in ("protocol", "endpoint", "model", "parameters")}
    identity = {"method_version": REWRITE_METHOD_VERSION, "rewrite_template_hash": text_hash(REWRITE_TEMPLATE),
                "verification_template_hash": text_hash(rewrite_fidelity.VERIFICATION_TEMPLATE),
                "checker": rewrite_fidelity.checker_identity(), "rewriter": semantic, "verifier": semantic}
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return identity | {"intervention_id": "rewrite:" + text_hash(canonical)}


def _plan(directory):
    validated = inputs.validate(directory)
    configs = model_configs()
    spec = EnsembleSpec(draws=3, max_workers=1, min_parsed=2, max_transport_failure_ratio=1 / 3,
                        retain_draws=True)
    ensemble = {field.name: getattr(spec, field.name) for field in fields(spec) if field.name != "draws"}
    ensemble["tail"] = "two_sided"
    evaluation = _read(directory / "evaluation.json")["items"]
    audit = {"method": "census of every planned item for each provider, selected before generation",
             "item_ids": [item["item_id"] for item in evaluation], "per_provider": 16,
             "include": ["accepted", "rejected", "unperformed"],
             "retain": ["original", "candidate", "automated_checks", "auditor_identity", "judgment", "disagreements"],
             "agent_inspection": "Root agent inspection is not independent human audit; efficacy remains unestablished without independent evidence."}
    # Exercise existing wire-enum construction without an executable unscored
    # NIM protocol: its two real calibration fingerprints do not exist yet.
    protocol = ExperimentProtocol(configs["anthropic"], configs["anthropic"], configs["anthropic"],
        validated["artifact_sha256"], _read(directory / "cohort-definitions.json")["models"]["claude-opus-5-5"],
        draws=3, bootstrap_seed=0, bootstrap_count=2000, task_metric="accuracy", utility_tolerance=0.05,
        audit_selection=audit, request_budget=144, rewrite_method_version=REWRITE_METHOD_VERSION,
        ensemble_settings=ensemble)
    protocol.build_ensemble_spec()
    hashes = dict(validated["artifact_sha256"])
    hashes["preparation-report.json"] = inputs.digest((directory / "preparation-report.json").read_bytes())
    return {
        "schema_version": 1, "study_id": "completion-api-2026-10-31-exploratory-pilot-v1",
        "status": "frozen_plan_only", "frozen_before_model_calls": True,
        "inputs": {"sha256": hashes, "counts": validated["inclusion_counts"],
                   "evaluation_order": [item["item_id"] for item in evaluation],
                   "cohorts": _read(directory / "cohort-definitions.json"),
                   "model_provenance": _read(directory / "model-provenance.json")},
        "models": {name: config.to_dict() for name, config in configs.items()},
        "auxiliary_roles": {"rewriter": "anthropic", "verifier": "anthropic", "shared_across_target_experiments": True,
                            "rewrite_variability": "Separately sampled rewrites across provider runs; cross-provider differences also include rewrite variation."},
        "anthropic_api_version": ANTHROPIC_API_VERSION,
        "anthropic_thinking": "Adaptive thinking always on; no disabled thinking, temperature or seed sent.",
        "method": _method(configs["anthropic"]),
        "evaluation": {"conditions": ["original_single", "original_ensemble", "rewritten_single", "rewritten_ensemble"],
                       "draws": 3, "single_draw_index": 0, "single_source": "reuse draw 0; no additional model requests",
                       "rewrite_scope": "shared_per_item", "arm_order": "alternating by frozen evaluation item index",
                       "ensemble_settings": ensemble, "task_metric": "accuracy", "utility_tolerance": 0.05,
                       "utility_tolerance_units": "probability units (5 percentage points)",
                       "bootstrap": {"seed": 0, "count": 2000, "level": 0.95, "unit": "paired item cluster, retaining all arms/draws"},
                       "audit": audit, "failures": "Retain all failures, rejected/unperformed rewrites and all-attempted denominators; no fallback or successful replacement selection."},
        "calibration": {"target": "nim", "interventions": ["original", "rewritten"], "is_items": 20, "oos_items": 20,
                        "min_valid": 10, "min_auc": 0.6, "seed": 0, "max_workers": 1,
                        "fit_fingerprints": None, "fingerprint_status": "Derived from actual task6.3 fitted states before scored evaluation; never placeholders.",
                        "rewrite_policy": "Same method and Opus auxiliary configs as evaluation; retain accepted subset, rejection, weakness and failure without outcome tuning.",
                        "frontier_mcs": "unavailable; native Anthropic has no genuine logprobs"},
        "budget": {"study_cap": 512, "phase_caps": dict(PHASE_CAPS),
                   "nominal_requests": {"eligibility": 2, "calibration": 206, "nim_evaluation": 131, "anthropic_evaluation": 131},
                   "nominal_breakdown": {"calibration": {"fit_generations": 120, "auxiliary": 80, "workflow_and_fit_probes": 6},
                                         "per_evaluation": {"target_draws": 96, "auxiliary": 32, "role_probes": 3}},
                   "nominal_total": 470, "accounting": "Every physical probe, target, rewrite, verification and retry counts in its phase; max_retries=0.",
                   "transfers": "No automatic unused-cap transfers; calibration stays 220 after eligibility.",
                   "phase_clients": "Fresh phase budget/clients; identical role configs may reuse one client/pacer within each phase. Retain per-phase ledgers."},
        "eligibility": {"prompt": PROBE_PROMPT, "prompt_sha256": text_hash(PROBE_PROMPT), "order": ["nim", "anthropic"],
                        "features": {"nim": {"mia_scoring": True}, "anthropic": {"mia_scoring": False}},
                        "rerun": "No automatic replay of an existing eligibility directory, including ambiguous intent."},
        "effective_settings": "Requested values are separate from confirmed effective values; omitted defaults remain unknown. Successful acceptance is not confirmation.",
        "validation_levels": {"implemented": ["nvidia_nim", "anthropic"], "offline": "Synthetic fixtures only; not live access evidence.",
                              "live": "Only the separate actual eligibility receipt establishes current inference eligibility; no batch effects measured yet."},
        "limitations": ["Exploratory 16-item pilot, no power or equivalence claim.",
                        "Temporal cohorts and historical gaps are weak membership proxies, not proof of memorization or exclusion.",
                        "Independent human fidelity audit is not yet performed; automated acceptance is not semantic proof.",
                        "Calibration quality and valid paired outcomes are unknown before task6.3/task6.4.",
                        "Credits, sufficient future balance and monetary cost are unknown; probe success proves current eligibility only.",
                        "OpenAI and Mistral live coverage remains unmeasured; no general provider-family claim.",
                        "Hosted serving revision and unechoed effective settings remain unknown."]}


def freeze(directory):
    plan = _plan(directory)
    _write_new(directory / "study-plan.json", plan, exact_resume=True)
    return plan


def validate(directory):
    expected = _plan(directory)
    if (directory / "study-plan.json").read_bytes() != inputs._json_bytes(expected):
        raise ValueError("Saved frozen plan differs from validated inputs, configuration, methods or study rules")
    if sum(expected["budget"]["phase_caps"].values()) != expected["budget"]["study_cap"]:
        raise ValueError("Study phase caps must sum to the hard study cap")
    return expected


def _clean(value, secrets):
    """Redact known values again at the final public-only artifact boundary."""
    if isinstance(value, dict):
        return {_clean(key, secrets): _clean(child, secrets) for key, child in value.items()}
    if isinstance(value, list):
        return [_clean(child, secrets) for child in value]
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, "[REDACTED]")
    return value


def _failure(provider, category):
    return {"phase": "eligibility", "provider": provider, "category": category}


def _event_snapshot(event):
    # Provider diagnostics may echo arbitrary request envelopes. Preserve the
    # public numeric/status/code/parameter fields, never a raw error body.
    public = event.to_dict()
    public["error"] = "Physical request failed; inspect status/code/parameter." if event.error else None
    public["detail"] = None
    return public


def _category(error):
    if isinstance(error, BudgetExhaustedError):
        return "budget_exhausted"
    if isinstance(error, RequestError):
        if error.status_code in {401, 403}:
            return "authentication"
        if error.status_code == 402 or error.code in {"insufficient_quota", "billing_hard_limit_reached", "billing_not_active", "credit_balance_exhausted", "insufficient_credits", "usage_limit_reached"}:
            return "billing"
        if error.parameter or error.status_code == 400:
            return "parameter"
        return "request_failure"
    return "runtime_failure"


def probe(directory, *, nim_credential_env, anthropic_credential_env, environment=None):
    """Run at most two physical probes, once, after the immutable plan exists.

    Resolve only the two chosen references. Persist intent before each probe and
    safe partial snapshots afterward. A crash leaves the directory ambiguous;
    it is never retried automatically, even if the latest ledger was not saved.
    """
    return _run_probe_attempt(directory, validate(directory), directory / "eligibility",
                              nim_credential_env, anthropic_credential_env, environment)


def _recovery_link(directory, base):
    return {"path": "eligibility/receipt.json", "sha256": inputs.digest((directory / "eligibility/receipt.json").read_bytes()),
            "failure": base["failure"], "used": base["budget"]["used"]}


def _run_probe_attempt(directory, plan, location, nim_credential_env, anthropic_credential_env, environment, base=None):
    names = {"nim": nim_credential_env, "anthropic": anthropic_credential_env}
    if any(not isinstance(name, str) or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None for name in names.values()):
        raise ValueError("Supply explicit environment-variable names, never literal values")
    location.mkdir(exist_ok=False)
    source = os.environ if environment is None else environment
    selected = {provider: source.get(name) for provider, name in names.items()}
    secrets = tuple(sorted({v for v in selected.values() if isinstance(v, str) and v}, key=len, reverse=True))
    previous = [] if base is None else base["events"]
    budget = CallBudget(PHASE_CAPS["eligibility"] - len(previous))
    receipt = {"schema_version": 1, "status": "incomplete", "evidence_kind": "actual_generation_probe",
               "study_plan_sha256": inputs.digest((directory / "study-plan.json").read_bytes()),
               "started_at": datetime.now(UTC).isoformat(), "finished_at": None,
               "providers": {} if base is None else dict(base["providers"]), "failure": None, "cost": None, "balance": None,
               "remaining_phase_caps": {name: cap for name, cap in PHASE_CAPS.items() if name != "eligibility"},
               "prompt": PROBE_PROMPT, "prompt_sha256": text_hash(PROBE_PROMPT)}
    if base is not None:
        receipt["recovery_of"] = _recovery_link(directory, base)

    def snapshot(name):
        new_events = [_event_snapshot(event) for event in budget.events]
        if base is not None:
            new_events = [event | {"source_attempt": "recovery-01", "source_call_id": event["call_id"],
                                   "call_id": len(previous) + event["call_id"]} for event in new_events]
        receipt["events"] = previous + new_events
        receipt["budget"] = {"limit": 4, "used": len(previous) + budget.used, "remaining": budget.remaining}
        _write_new(location / name, _clean(receipt, secrets))

    snapshot("00-initial.json")
    for provider, value in selected.items():
        if not isinstance(value, str) or not value.strip():
            receipt["failure"] = _failure(provider, "missing_credential")
            snapshot("01-missing.json")
            break
    else:
        configs = model_configs()
        order = plan["eligibility"]["order"] if base is None else ["anthropic", "nim"]
        order = [provider for provider in order if not receipt["providers"].get(provider, {}).get("ready_for_batch", False)]
        for index, provider in enumerate(order, 1):
            # The directory and immutable numbered intents survive all crashes.
            intent = {"status": "started", "provider": provider, "study_plan_sha256": receipt["study_plan_sha256"],
                      "physical_calls_before": budget.used, "prompt_sha256": text_hash(PROBE_PROMPT)}
            _write_new(location / ("intent.json" if index == 1 else f"intent-{index:02}.json"), intent)
            before = budget.used
            config = replace(configs[provider], credential_ref=names[provider])
            features = FeatureChoices(mia_scoring=provider == "nim")
            factory = OpenAICompatibleClient if provider == "nim" else AnthropicClient
            try:
                client = factory(config, budget, features=features, environment={names[provider]: selected[provider]})
                prefix = "eligibility/probe/" if base is None else "eligibility/recovery-01/probe/"
                report = client.probe(PROBE_PROMPT, stage=prefix + provider)
                expected = set(config.inspect(features).capabilities)
                ready = (report.ready_for_batch and set(report.capabilities) == expected
                         and budget.used == before + 1 and budget.events[-1].status == "succeeded")
                receipt["providers"][provider] = {
                    "config": config.to_dict(), "ready_for_batch": ready,
                    "capabilities": {name: evidence.to_dict() for name, evidence in report.capabilities.items()},
                    "raw_response": None, "provider_returned_model": None, "confirmed_effective_values": None,
                    "unavailable_metadata_reason": "Public probe API returns capability report only; it does not expose raw response or serving revision.",
                    "anthropic_api_version": ANTHROPIC_API_VERSION if provider == "anthropic" else None}
                if not ready:
                    receipt["failure"] = _failure(provider, "capability")
            except Exception as error:
                receipt["failure"] = _failure(provider, _category(error))
            snapshot(f"{index:02}-{provider}.json")
            if receipt["failure"]:
                break
    receipt["status"] = "failed" if receipt["failure"] else "eligible"
    receipt["finished_at"] = datetime.now(UTC).isoformat()
    snapshot("receipt.json")
    return _clean(receipt, secrets)


def _verify_saved_receipt(directory, plan, location, base=None):
    if not (location / "receipt.json").exists():
        raise ValueError("Eligibility receipt is incomplete or ambiguous; automatic replay is forbidden")
    receipt = _read(location / "receipt.json")
    previous = [] if base is None else base["events"]
    remaining = {name: cap for name, cap in PHASE_CAPS.items() if name != "eligibility"}
    checkpoints = sorted(location.glob("[0-9][0-9]-*.json"))
    latest = _read(checkpoints[-1])
    events = receipt["events"]
    try:
        started = datetime.fromisoformat(receipt["started_at"])
        finished = datetime.fromisoformat(receipt["finished_at"])
        terminal = started.tzinfo is not None and finished.tzinfo is not None and finished >= started
    except (ValueError, TypeError):
        terminal = False
    if (receipt["study_plan_sha256"] != inputs.digest((directory / "study-plan.json").read_bytes())
            or receipt["remaining_phase_caps"] != remaining or receipt["cost"] is not None or receipt["balance"] is not None
            or receipt["budget"] != {"limit": 4, "used": len(events), "remaining": 4 - len(events)}
            or len(events) > 4 or receipt["events"] != latest["events"] or receipt["providers"] != latest["providers"]
            or receipt["failure"] != latest["failure"] or receipt["prompt"] != PROBE_PROMPT
            or receipt["prompt_sha256"] != text_hash(PROBE_PROMPT) or not terminal
            or receipt["schema_version"] != 1 or receipt["evidence_kind"] != "actual_generation_probe"
            or (base is not None and (receipt.get("recovery_of") != _recovery_link(directory, base)
                                     or events[:len(previous)] != previous))):
        raise ValueError("Eligibility receipt identity, phase budget or checkpoint mismatch")
    for index, event in enumerate(events, 1):
        prefix = "eligibility/recovery-01/probe/" if base is not None and index > len(previous) else "eligibility/probe/"
        provider = event["stage"].removeprefix(prefix)
        if (event["call_id"] != index or event["attempt"] != 1 or event["status"] not in {"succeeded", "failed"}
                or provider not in plan["models"] or event["model"] != plan["models"][provider]["model"]
                or event["latency_s"] is None or event["latency_s"] < 0 or event["cost"] is not None
                or (base is not None and index > len(previous)
                    and (event.get("source_attempt") != "recovery-01" or event.get("source_call_id") != index - len(previous)))):
            raise ValueError("Eligibility receipt contains invalid or incomplete physical events")
    for provider, evidence in receipt["providers"].items():
        if evidence["config"] != plan["models"].get(provider):
            raise ValueError("Eligibility receipt config differs from frozen plan")
        report = CapabilityReport({name: CapabilityEvidence(**value) for name, value in evidence["capabilities"].items()},
                                  FeatureChoices(mia_scoring=provider == "nim"))
        required = set(model_configs()[provider].inspect(report.features).capabilities)
        if evidence["ready_for_batch"] != (report.ready_for_batch and set(report.capabilities) == required):
            raise ValueError("Eligibility receipt capability evidence disagrees with readiness")
    if receipt["status"] == "eligible":
        successes = {event["model"] for event in events if event["status"] == "succeeded"}
        if (receipt["failure"] is not None or set(receipt["providers"]) != {"nim", "anthropic"}
                or any(not value["ready_for_batch"] for value in receipt["providers"].values())
                or successes != {config["model"] for config in plan["models"].values()}
                or (base is None and (len(events) != 2 or [event["stage"] for event in events]
                                     != ["eligibility/probe/nim", "eligibility/probe/anthropic"]))):
            raise ValueError("Eligibility receipt does not prove both actual generation probes")
    elif receipt["status"] != "failed" or receipt["failure"] is None:
        raise ValueError("Eligibility receipt status is incomplete or invalid")
    return receipt


def verify_receipt(directory):
    """Verify initial/recovery evidence offline; fixture tests are not live."""
    plan = validate(directory)
    base = _verify_saved_receipt(directory, plan, directory / "eligibility")
    recovery = directory / "eligibility/recovery-01"
    if recovery.exists():
        if base["status"] != "failed":
            raise ValueError("Recovery requires an initial completed failed receipt")
        return _verify_saved_receipt(directory, plan, recovery, base)
    return base


def recover(directory, *, nim_credential_env, anthropic_credential_env, environment=None):
    """One operator-invoked recovery from verified failure, within the same cap.

    Existing recovery or ambiguous initial intent never replays. Successful
    initial providers are retained; missing providers get one probe each,
    Anthropic first. Failure stops this sole recovery; there is no retry loop.
    """
    plan = validate(directory)
    base = _verify_saved_receipt(directory, plan, directory / "eligibility")
    if base["status"] != "failed":
        raise ValueError("Recovery requires an initial completed failed receipt")
    if base["budget"]["remaining"] == 0:
        raise ValueError("Eligibility phase budget is exhausted")
    return _run_probe_attempt(directory, plan, directory / "eligibility/recovery-01",
                              nim_credential_env, anthropic_credential_env, environment, base)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "validate", "probe", "recover", "verify-receipt"))
    parser.add_argument("--directory", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument("--nim-credential-env")
    parser.add_argument("--anthropic-credential-env")
    args = parser.parse_args(argv)
    try:
        if args.action in {"probe", "recover"}:
            result = {"probe": probe, "recover": recover}[args.action](
                args.directory, nim_credential_env=args.nim_credential_env, anthropic_credential_env=args.anthropic_credential_env)
        else:
            result = {"freeze": freeze, "validate": validate, "verify-receipt": verify_receipt}[args.action](args.directory)
        print(json.dumps({"status": result["status"], "budget": result.get("budget"), "failure": result.get("failure")},
                         ensure_ascii=True, allow_nan=False))
        return 0 if result["status"] in {"frozen_plan_only", "eligible"} else 1
    except (OSError, ValueError, KeyError):
        # Arbitrary exception bodies, paths and references never reach output.
        print(json.dumps({"status": "rejected", "reason": "Frozen plan/input/receipt mismatch or existing eligibility directory; inspect offline before retry."}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
