"""One explicit NIM 600-second generation check within the frozen study cap.

``freeze``, ``validate`` and ``verify-receipt`` are offline and environment-free.
Only ``probe`` resolves the selected NIM environment reference. This separate,
post-observation amendment neither changes the original plan nor establishes
that latency caused the earlier failures. No existing attempt is replayed.
"""

import argparse
import json
import math
import os
import re
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from recall_guard.core.completion import CapabilityEvidence, CapabilityReport, FeatureChoices
from recall_guard.core.providers import OpenAICompatibleClient
from recall_guard.core.request_control import CallBudget, RequestError
from scripts import prepare_completion_pilot as pilot

AMENDMENT = "eligibility/latency-amendment.json"
ATTEMPT = "eligibility/latency-01"
STAGE = "eligibility/latency-01/probe/nim"
SOURCE_FILES = (
    "study-plan.json", "eligibility/00-initial.json", "eligibility/01-nim.json",
    "eligibility/intent.json", "eligibility/receipt.json",
    "eligibility/recovery-01/00-initial.json", "eligibility/recovery-01/01-anthropic.json",
    "eligibility/recovery-01/02-nim.json", "eligibility/recovery-01/intent.json",
    "eligibility/recovery-01/intent-02.json", "eligibility/recovery-01/receipt.json",
)
RECEIPT_FIELDS = {
    "schema_version", "status", "evidence_kind", "study_plan_sha256", "started_at", "finished_at",
    "providers", "failure", "cost", "balance", "remaining_phase_caps", "prompt", "prompt_sha256", "events", "budget",
}
EVENT_FIELDS = {
    "call_id", "stage", "model", "attempt", "started_at", "timestamp", "status", "latency_s", "usage",
    "cost", "status_code", "error", "parameter", "detail", "code",
}
PROVIDER_FIELDS = {
    "config", "ready_for_batch", "capabilities", "raw_response", "provider_returned_model",
    "confirmed_effective_values", "unavailable_metadata_reason", "anthropic_api_version",
}
UNKNOWN_METADATA = "Public probe API returns capability report only; it does not expose raw response or serving revision."


def _same(actual, expected):
    """Compare strict JSON values, including boolean/integer distinctions."""
    if pilot.inputs._json_bytes(actual) != pilot.inputs._json_bytes(expected):
        raise ValueError("Transport amendment source, receipt or configuration mismatch")


def _keys(value, expected):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError("Unknown or missing receipt fields")


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("Invalid physical-event numeric metadata")


def _time(value):
    try:
        result = datetime.fromisoformat(value)
        if result.tzinfo is not None:
            return result
    except (ValueError, TypeError):
        pass
    raise ValueError("Invalid receipt timestamp")


def _event(event, index, stage, model, *, source=None):
    _keys(event, EVENT_FIELDS | ({"source_attempt", "source_call_id"} if source else set()))
    _same({k: event[k] for k in ("call_id", "attempt", "stage", "model", "cost", "detail")},
          {"call_id": index, "attempt": 1, "stage": stage, "model": model, "cost": None, "detail": None})
    if source:
        _same([event["source_attempt"], event["source_call_id"]], list(source))
    _number(event["started_at"])
    _number(event["latency_s"])
    _time(event["timestamp"])
    if event["status"] not in {"succeeded", "failed"}:
        raise ValueError("Incomplete physical event")
    _same(event["error"], None if event["status"] == "succeeded" else "Physical request failed; inspect status/code/parameter.")
    if event["status"] == "succeeded" and any(event[k] is not None for k in ("status_code", "code", "parameter")):
        raise ValueError("Successful event contains failure diagnostics")
    status = event["status_code"]
    if status is not None and (type(status) is not int or not 100 <= status <= 599):
        raise ValueError("Invalid HTTP status")
    for name in ("code", "parameter"):
        value = event[name]
        if value is not None and (not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.:\[\]-]{1,128}", value)):
            raise ValueError("Invalid safe diagnostic")
    usage = event["usage"]
    if usage is not None and (not isinstance(usage, dict) or any(
        name not in {"input_tokens", "output_tokens", "total_tokens", "prompt_tokens", "completion_tokens",
                     "cache_creation_input_tokens", "cache_read_input_tokens", "reasoning_tokens"}
        or type(count) is not int or count < 0 for name, count in usage.items()
    )):
        raise ValueError("Invalid provider-reported usage")


def _provider(value, config, provider, event):
    _keys(value, PROVIDER_FIELDS)
    _same(value["config"], config.to_dict())
    _same([value["raw_response"], value["provider_returned_model"], value["confirmed_effective_values"],
           value["unavailable_metadata_reason"], value["anthropic_api_version"]],
          [None, None, None, UNKNOWN_METADATA, pilot.ANTHROPIC_API_VERSION if provider == "anthropic" else None])
    features = FeatureChoices(mia_scoring=provider == "nim")
    required = set(config.inspect(features).capabilities)
    _keys(value["capabilities"], required)
    for evidence in value["capabilities"].values():
        _keys(evidence, {"status", "source", "detail", "observed"})
        _same(evidence["source"], "Authenticated generation probe" if provider == "nim"
              else "Anthropic Messages API 2026-10-21 authenticated generation probe")
    report = CapabilityReport({name: CapabilityEvidence(**item) for name, item in value["capabilities"].items()}, features)
    ready = report.ready_for_batch and event["status"] == "succeeded"
    _same(value["ready_for_batch"], ready)
    return ready


def _receipt(receipt, directory, extra=()):
    _keys(receipt, RECEIPT_FIELDS | set(extra))
    _same([receipt["schema_version"], receipt["evidence_kind"], receipt["study_plan_sha256"], receipt["cost"], receipt["balance"],
           receipt["remaining_phase_caps"], receipt["prompt"], receipt["prompt_sha256"]],
          [1, "actual_generation_probe", pilot.inputs.digest((directory / "study-plan.json").read_bytes()), None, None,
           {k: v for k, v in pilot.PHASE_CAPS.items() if k != "eligibility"}, pilot.PROBE_PROMPT, pilot.text_hash(pilot.PROBE_PROMPT)])
    if _time(receipt["finished_at"]) < _time(receipt["started_at"]):
        raise ValueError("Receipt finish precedes start")
    _same(receipt["budget"], {"limit": 4, "used": len(receipt["events"]), "remaining": 4 - len(receipt["events"])})


def _intent(plan_hash, provider, used, **extra):
    return {"status": "started", "provider": provider, "study_plan_sha256": plan_hash,
            "physical_calls_before": used, "prompt_sha256": pilot.text_hash(pilot.PROBE_PROMPT), **extra}


def _checkpoint(receipt, count, providers, failure):
    return receipt | {"status": "incomplete", "finished_at": None, "events": receipt["events"][:count],
                      "providers": providers, "failure": failure,
                      "budget": {"limit": 4, "used": count, "remaining": 4 - count}}


def _sources(directory):
    """Accept only the completed, failed three-request trajectory being amended."""
    plan = pilot.validate(directory)
    try:
        initial = pilot._verify_saved_receipt(directory, plan, directory / "eligibility")
        base = pilot._verify_saved_receipt(directory, plan, directory / "eligibility/recovery-01", initial)
        _receipt(initial, directory)
        _receipt(base, directory, {"recovery_of"})
        failure = pilot._failure("nim", "request_failure")
        _same([initial["status"], initial["failure"], initial["providers"], len(initial["events"])], ["failed", failure, {}, 1])
        _same([base["status"], base["failure"], sorted(base["providers"]), base["budget"]],
              ["failed", failure, ["anthropic"], {"limit": 4, "used": 3, "remaining": 1}])
        _same(base["events"][:1], initial["events"])
        if _time(base["started_at"]) < _time(initial["finished_at"]):
            raise ValueError("Recovery starts before initial completion")
        stages = ["eligibility/probe/nim", "eligibility/recovery-01/probe/anthropic", "eligibility/recovery-01/probe/nim"]
        for index, (event, stage) in enumerate(zip(base["events"], stages, strict=True), 1):
            provider = "anthropic" if index == 2 else "nim"
            _event(event, index, stage, pilot.model_configs()[provider].model,
                   source=None if index == 1 else ("recovery-01", index - 1))
            _same([event["status"], event["code"], event["status_code"], event["parameter"]],
                  ["succeeded", None, None, None] if index == 2 else ["failed", "timeout", None, None])
            owner = initial if index == 1 else base
            if not _time(owner["started_at"]) <= _time(event["timestamp"]) <= _time(owner["finished_at"]):
                raise ValueError("Source event outside its recorded attempt")
        if not _provider(base["providers"]["anthropic"], pilot.model_configs()["anthropic"], "anthropic", base["events"][1]):
            raise ValueError("Anthropic readiness is missing")
        # Validate every retained checkpoint/intent as well as the terminal receipts.
        expected = {
            "eligibility/00-initial.json": _checkpoint(initial, 0, {}, None),
            "eligibility/01-nim.json": _checkpoint(initial, 1, {}, failure),
            "eligibility/intent.json": _intent(initial["study_plan_sha256"], "nim", 0),
            "eligibility/recovery-01/00-initial.json": _checkpoint(base, 1, {}, None),
            "eligibility/recovery-01/01-anthropic.json": _checkpoint(base, 2, base["providers"], None),
            "eligibility/recovery-01/02-nim.json": _checkpoint(base, 3, base["providers"], failure),
            "eligibility/recovery-01/intent.json": _intent(base["study_plan_sha256"], "anthropic", 0),
            "eligibility/recovery-01/intent-02.json": _intent(base["study_plan_sha256"], "nim", 1),
        }
        for name, value in expected.items():
            _same(pilot._read(directory / name), value)
        allowed = {"00-initial.json", "01-nim.json", "intent.json", "receipt.json", "recovery-01", "latency-amendment.json", "latency-01"}
        if set(p.name for p in (directory / "eligibility").iterdir()) - allowed:
            raise ValueError("Unknown eligibility source")
        _same(sorted(p.name for p in (directory / "eligibility/recovery-01").iterdir()),
              sorted(Path(name).name for name in SOURCE_FILES if name.startswith("eligibility/recovery-01/")))
        hashes = {name: pilot.inputs.digest((directory / name).read_bytes()) for name in SOURCE_FILES}
        return base, hashes
    except (KeyError, TypeError, OSError, OverflowError) as error:
        raise ValueError("Invalid or incomplete source receipts") from error


def _amendment(directory):
    base, hashes = _sources(directory)
    return {
        "schema_version": 1, "status": "frozen_transport_amendment_only", "amendment_id": "nim-latency-01",
        "post_observation": True, "preregistered": False,
        "reason": "Operator latency clarification after two recorded 90-second NIM timeouts; latency as root cause remains unproven.",
        "sources": hashes, "change": {"provider": "nim", "field": "timeout_s", "previous": 90, "amended": 600},
        "config": replace(pilot.model_configs()["nim"], timeout_s=600).to_dict(),
        "budget": {"study_cap": 512, "phase_caps": dict(pilot.PHASE_CAPS), "eligibility_used": base["budget"]["used"],
                   "eligibility_remaining": 1, "new_call_limit": 1},
        "attempt_directory": ATTEMPT, "prompt": pilot.PROBE_PROMPT, "prompt_sha256": pilot.text_hash(pilot.PROBE_PROMPT),
        "limitations": ["Probe success establishes current inference eligibility only; cost and balance remain unknown.",
                        "No automatic replay, second live action, phase-cap transfer, calibration or evaluation.",
                        "Later calibration and NIM evaluation must explicitly bind this transport amendment before execution.",
                        "Synthetic tests are not live coverage; MIA readiness requires observed genuine token evidence."],
    }


def freeze(directory):
    """Append a fixed amendment without modifying any original study bytes."""
    if (directory / ATTEMPT).exists():
        raise FileExistsError("Latency attempt exists; no refreezing or replay")
    expected = _amendment(directory)
    pilot._write_new(directory / AMENDMENT, expected, exact_resume=True)
    return expected


def validate(directory):
    """Strict offline verification; no client construction or environment access."""
    expected = _amendment(directory)
    if (directory / AMENDMENT).read_bytes() != pilot.inputs._json_bytes(expected):
        raise ValueError("Frozen transport amendment differs from verified original sources")
    return expected


def probe(directory, *, nim_credential_env, environment=None):
    """One explicit NIM-only action; all existing attempt directories are final."""
    amendment = validate(directory)
    if not isinstance(nim_credential_env, str) or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", nim_credential_env) is None:
        raise ValueError("Supply an explicit environment-variable name")
    location = directory / ATTEMPT
    location.mkdir(exist_ok=False)
    source = os.environ if environment is None else environment
    selected = source.get(nim_credential_env)
    secrets = (selected,) if isinstance(selected, str) and selected else ()
    base = pilot._read(directory / "eligibility/recovery-01/receipt.json")
    budget = CallBudget(1)
    receipt = {k: v for k, v in base.items() if k != "recovery_of"}
    receipt.update(status="incomplete", started_at=datetime.now(UTC).isoformat(), finished_at=None, failure=None,
                   providers=dict(base["providers"]), amendment_sha256=pilot.inputs.digest((directory / AMENDMENT).read_bytes()),
                   base_receipt_sha256=amendment["sources"]["eligibility/recovery-01/receipt.json"])

    def snapshot(name):
        new_events = [pilot._event_snapshot(event) | {"call_id": 4, "source_attempt": "latency-01", "source_call_id": 1}
                      for event in budget.events]
        receipt["events"] = base["events"] + new_events
        receipt["budget"] = {"limit": 4, "used": 3 + budget.used, "remaining": budget.remaining}
        pilot._write_new(location / name, pilot._clean(receipt, secrets))

    snapshot("00-initial.json")
    if not isinstance(selected, str) or not selected.strip():
        receipt["failure"] = pilot._failure("nim", "missing_credential")
        snapshot("01-missing.json")
    else:
        pilot._write_new(location / "intent.json", _intent(receipt["study_plan_sha256"], "nim", 3,
                                                         amendment_sha256=receipt["amendment_sha256"]))
        config = replace(pilot.model_configs()["nim"], timeout_s=600, credential_ref=nim_credential_env)
        features = FeatureChoices(mia_scoring=True)
        try:
            client = OpenAICompatibleClient(config, budget, features=features, environment={nim_credential_env: selected})
            report = client.probe(pilot.PROBE_PROMPT, stage=STAGE)
            ready = (report.ready_for_batch and set(report.capabilities) == set(config.inspect(features).capabilities)
                     and budget.used == 1 and budget.events[0].status == "succeeded")
            receipt["providers"]["nim"] = {
                "config": config.to_dict(), "ready_for_batch": ready,
                "capabilities": {name: evidence.to_dict() for name, evidence in report.capabilities.items()},
                "raw_response": None, "provider_returned_model": None, "confirmed_effective_values": None,
                "unavailable_metadata_reason": UNKNOWN_METADATA, "anthropic_api_version": None}
            if not ready:
                receipt["failure"] = pilot._failure("nim", "capability")
        except Exception as error:
            receipt["failure"] = pilot._failure("nim", pilot._category(error))
        snapshot("01-nim.json")
    receipt["status"] = "failed" if receipt["failure"] else "eligible"
    receipt["finished_at"] = datetime.now(UTC).isoformat()
    snapshot("receipt.json")
    return pilot._clean(receipt, secrets)


def verify_receipt(directory):
    """Verify the final amended receipt and retained history entirely offline."""
    amendment = validate(directory)
    base = pilot._read(directory / "eligibility/recovery-01/receipt.json")
    location = directory / ATTEMPT
    try:
        if not (location / "receipt.json").is_file():
            raise ValueError("Latency receipt is incomplete or ambiguous; no replay")
        receipt = pilot._read(location / "receipt.json")
        _receipt(receipt, directory, {"amendment_sha256", "base_receipt_sha256"})
        if _time(receipt["started_at"]) < _time(base["finished_at"]):
            raise ValueError("Latency attempt starts before recovery completion")
        _same([receipt["amendment_sha256"], receipt["base_receipt_sha256"]],
              [pilot.inputs.digest((directory / AMENDMENT).read_bytes()), amendment["sources"]["eligibility/recovery-01/receipt.json"]])
        _same(receipt["events"][:3], base["events"])
        if len(receipt["events"]) not in {3, 4}:
            raise ValueError("Invalid amended event count")
        _same(receipt["providers"].get("anthropic"), base["providers"]["anthropic"])
        _same(pilot._read(location / "00-initial.json"), _checkpoint(receipt, 3, base["providers"], None))
        failure = receipt["failure"]
        category = None if failure is None else failure["category"]
        if category not in {None, "missing_credential", "capability", "budget_exhausted", "authentication", "billing", "parameter", "request_failure", "runtime_failure"}:
            raise ValueError("Unknown failure category")
        _same(failure, None if category is None else pilot._failure("nim", category))
        _same(receipt["status"], "eligible" if failure is None else "failed")
        if category == "missing_credential":
            _same(receipt["events"], base["events"])
            _same(receipt["providers"], base["providers"])
            expected_files = {"00-initial.json", "01-missing.json", "receipt.json"}
            checkpoint = "01-missing.json"
        else:
            expected_files = {"00-initial.json", "01-nim.json", "intent.json", "receipt.json"}
            checkpoint = "01-nim.json"
            _same(pilot._read(location / "intent.json"), _intent(receipt["study_plan_sha256"], "nim", 3,
                                                               amendment_sha256=receipt["amendment_sha256"]))
            if len(receipt["events"]) != 4:
                # A local constructor/configuration failure consumes no request.
                if category != "runtime_failure":
                    raise ValueError("Missing underlying generation request")
            else:
                event = receipt["events"][3]
                _event(event, 4, STAGE, amendment["config"]["model"], source=("latency-01", 1))
                if _time(event["timestamp"]) < _time(receipt["started_at"]) or _time(event["timestamp"]) > _time(receipt["finished_at"]):
                    raise ValueError("Physical event outside amended attempt")
                ready = False
                if "nim" in receipt["providers"]:
                    ready = _provider(receipt["providers"]["nim"], replace(pilot.model_configs()["nim"], timeout_s=600), "nim", event)
                if (failure is None and not ready) or (category == "capability" and (ready or event["status"] != "succeeded")):
                    raise ValueError("MIA capability evidence or underlying request disagrees with eligibility")
                if category in {"authentication", "billing", "parameter", "request_failure"}:
                    if event["status"] != "failed":
                        raise ValueError("Request failure has successful event")
                    error = RequestError("safe", status_code=event["status_code"], code=event["code"], parameter=event["parameter"])
                    _same(pilot._category(error), category)
            if "nim" in receipt["providers"] and category not in {None, "capability"}:
                raise ValueError("Provider evidence disagrees with failed probe stage")
            if set(receipt["providers"]) - {"anthropic", "nim"}:
                raise ValueError("Unknown provider evidence")
        _same(pilot._read(location / checkpoint), receipt | {"status": "incomplete", "finished_at": None})
        _same(sorted(p.name for p in location.iterdir()), sorted(expected_files))
        return receipt
    except (KeyError, TypeError, OSError, OverflowError) as error:
        raise ValueError("Invalid, incomplete or ambiguous latency receipt") from error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "validate", "probe", "verify-receipt"))
    parser.add_argument("--directory", type=Path, default=pilot.DEFAULT_DIRECTORY)
    parser.add_argument("--nim-credential-env")
    args = parser.parse_args(argv)
    try:
        if args.action == "probe":
            result = probe(args.directory, nim_credential_env=args.nim_credential_env)
        else:
            result = {"freeze": freeze, "validate": validate, "verify-receipt": verify_receipt}[args.action](args.directory)
        print(json.dumps({"status": result["status"], "budget": result.get("budget"), "failure": result.get("failure")}, ensure_ascii=True, allow_nan=False))
        return 0 if result["status"] in {"frozen_transport_amendment_only", "eligible"} else 1
    except (OSError, ValueError, KeyError, TypeError):
        print(json.dumps({"status": "rejected", "reason": "Amendment/source/receipt mismatch or existing attempt; inspect offline. No automatic replay."}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
