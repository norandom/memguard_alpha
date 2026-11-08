"""Bounded Modal pilot. Default planning reads no credentials and calls no SDK.

Execution requires a separately reviewed approval record and already staged weights.
Run from the checkout using ``uv run --no-sync python scripts/probe_modal_model.py``.
"""

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path, PurePosixPath

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deploy.modal import serve
from deploy.modal.manifest import Resources, load_plan
from recall_guard.core.completion import ModelConfig
from recall_guard.core.harmony_evidence import HarmonyTokenMap
from recall_guard.core.local_evidence import LocalEvidenceError, validate_local_evidence
from recall_guard.core.profiles import load_profile
from recall_guard.core.providers import HarmonyOpenAICompatibleClient, OpenAICompatibleClient
from recall_guard.core.request_control import CallBudget, RequestError
from scripts.modal_cost_report import append_billing, cost_report, validate_budget

MANIFEST = Path("configs/modal/gpt-oss-20b-pilot.json")
PROMPTS = (
    "Reply with the word OK.",
    "Name one primary color in one word.",
    "What is two plus two? Answer briefly.",
    "Complete the phrase: The sky is",
    "Name a common household pet in one word.",
    "Write the opposite of hot in one word.",
    "Name a season in one word.",
    "Reply with a short greeting.",
)
SETTINGS = {"temperature": 0, "top_p": 1, "logprobs": True, "top_logprobs": 5, "max_tokens": 512}


def diagnostic_text(value, limit=2000):
    """Redact child-process credential values; never load a credential file."""
    secrets = set()
    for name, secret in os.environ.items():
        if secret and any(marker in name.upper() for marker in ("API_KEY", "TOKEN", "SECRET", "PASSWORD", "BEARER")):
            secrets.add(secret)
            if "PROXY" in name.upper() or "BEARER" in name.upper():
                secrets.update(part for part in secret.removeprefix("Bearer ").split(".") if part)
    value = re.sub(r"(?i)\bBearer\s+[^\s,;\"']+", "Bearer [REDACTED]", value)
    for secret in sorted(secrets, key=len, reverse=True):
        value = value.replace(secret, "[REDACTED]")
    return value[:limit]


def error_diagnostic(exc, stage):
    """Bounded metadata only: no locals, source lines, responses or inference text."""
    if stage == "inference":
        mark_sensitive_error_chain(exc)
    frames = []
    trace = exc.__traceback__
    while trace is not None:
        code = trace.tb_frame.f_code
        frames.append({"filename": diagnostic_text(Path(code.co_filename).name, 200),
                       "function": diagnostic_text(code.co_name, 200), "lineno": trace.tb_lineno})
        trace = trace.tb_next
    result = {"stage": stage, "error": type(exc).__name__, "frames": frames[-12:]}
    if stage != "inference":
        # RequestError may embed arbitrary provider response detail in its string.
        result["message"] = diagnostic_message(exc)
        chain, seen = [], {id(exc)}
        cause = exc.__cause__ or (None if exc.__suppress_context__ else exc.__context__)
        while cause is not None and id(cause) not in seen and len(chain) < 4:
            seen.add(id(cause))
            chain.append({"error": type(cause).__name__, "message": diagnostic_message(cause)})
            cause = cause.__cause__ or (None if cause.__suppress_context__ else cause.__context__)
        if chain:
            result["chain"] = chain
    return result


def mark_sensitive_error_chain(exc):
    """Keep inference/response ancestry opaque even during later cleanup failures."""
    pending, seen = [exc], set()
    while pending:
        error = pending.pop()
        if error is None or id(error) in seen:
            continue
        seen.add(id(error))
        error._modal_diagnostic_sensitive = True
        pending.extend((error.__cause__, error.__context__))


def diagnostic_message(exc):
    if isinstance(exc, RequestError):
        mark_sensitive_error_chain(exc)
        return "provider_or_transport_failure"
    if getattr(exc, "_modal_diagnostic_sensitive", False):
        mark_sensitive_error_chain(exc)
        return "sensitive_error_detail_withheld"
    return diagnostic_text(str(exc))


def record_error(receipt, path, exc, stage):
    """Keep the first failure authoritative while retaining secondary failures."""
    diagnostic = error_diagnostic(exc, stage)
    receipt.setdefault("diagnostics", []).append(diagnostic)
    receipt.setdefault("error", diagnostic["error"])
    receipt.setdefault("error_stage", stage)
    persist(path, receipt)
    exc._modal_diagnostic_receipt = id(receipt)


@contextmanager
def diagnostic_stage(receipt, path, stage):
    prior = len(receipt.get("diagnostics", []))
    current = {"stage": stage}
    try:
        yield current
    except (Exception, asyncio.CancelledError, KeyboardInterrupt) as exc:
        recorded = receipt.get("diagnostics", [])[prior:]
        if not recorded:
            record_error(receipt, path, exc, current["stage"])
        elif getattr(exc, "_modal_diagnostic_receipt", None) != id(receipt):
            # A lifecycle exit can fail while unwinding an already recorded body error.
            record_error(receipt, path, exc, "cleanup")
        raise


def digest(data):
    return hashlib.sha256(data).hexdigest()


def tokenizer_counts(plan, root):
    """Count the actual one-user-message chat wire input with the pinned template.

    No downloads, fallback estimates or fabricated usage. Only the exact HF snapshot
    layout is accepted. The serving provenance must later match these file hashes.
    """
    hashes = tokenizer_hashes(plan, root)
    if hashes is None:
        return None, None
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_directory(plan, root)), local_files_only=True,
                                              trust_remote_code=False)
    counts = [len(tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True
    )) for prompt in PROMPTS]
    return counts, hashes


def tokenizer_directory(plan, root):
    return (root / "hub" / ("models--" + plan.artifacts.tokenizer.replace("/", "--"))
            / "snapshots" / plan.artifacts.tokenizer_revision)


def tokenizer_hashes(plan, root):
    """Local pinned tokenizer file hashes, in the serving provenance's shape; None if absent."""
    directory = tokenizer_directory(plan, root)
    files = [directory / name for name in ("tokenizer.json", "tokenizer_config.json")]
    if any(not file.is_file() for file in files):
        return None
    hashes = {file.name: digest(file.read_bytes()) for file in files}
    template = directory / "chat_template.jinja"
    hashes[template.name] = digest(template.read_bytes()) if template.is_file() else None
    return hashes


def planning(manifest=MANIFEST, *, tokenizer_root=Path("models")):
    plan = load_plan(manifest)
    try:
        counts, hashes = tokenizer_counts(plan, tokenizer_root)
    except (ImportError, OSError, ValueError):
        counts, hashes = None, None
    blockers = list(plan.execution_blockers)
    if counts is None:
        blockers.append("pinned_tokenizer_unavailable")
    elif any(count + 512 > plan.resources.context_tokens for count in counts):
        blockers.append("context_allowance_exceeded")
    blockers.append("remote_weight_cache_unverified")
    cost = json.loads(manifest.read_text(encoding="utf-8"))["cost"]
    compute_rate = (cost["gpu_usd_per_second"] * plan.resources.gpu_count
                    + cost["physical_core_usd_per_second"] * plan.resources.cpu_physical_cores
                    + cost["memory_gib_usd_per_second"] * plan.resources.memory_gib)
    return {
        "mode": "plan", "manifest_sha256": digest(manifest.read_bytes()),
        "prompts": list(PROMPTS), "prompts_sha256": digest(json.dumps(PROMPTS).encode()),
        "input_tokens": counts, "tokenizer_sha256": hashes, "settings": SETTINGS,
        "limits": asdict(plan.limits), "execution_blockers": blockers,
        "inference_posts": 0, "cloud_mutations": 0,
        "cost_review": cost,
        "resource_estimate_gross_usd": compute_rate * plan.limits.gpu_phase_seconds,
        "estimated_total_gross_usd": None,
        "resources": asdict(plan.resources), "artifacts": asdict(plan.artifacts),
    }


def persist(path, receipt):
    """Flush the owned receipt before admission; callers never reopen an old run."""
    data = json.dumps(receipt, indent=2, allow_nan=False)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


APPROVAL_ASSERTIONS = ("paid_execution_authorized", "account_access_verified", "operator_allowance_confirmed",
                       "model_access_verified", "pricing_and_budget_reviewed")


def approval_record(path, plan, hashes):
    """Shared approval gate: five explicit assertions, exact frozen input hashes,
    explicit workspace/endpoint and a reviewed provider budget reference."""
    record = json.loads(path.read_text(encoding="utf-8"))
    if any(record.get(key) is not True for key in APPROVAL_ASSERTIONS):
        raise ValueError("execution_approval_or_prerequisites_missing")
    if any(record.get(key) != value for key, value in hashes.items()):
        raise ValueError("approval_plan_mismatch")
    if record.get("workspace") != plan.target.workspace or not plan.target.workspace or not plan.target.endpoint:
        raise ValueError("explicit_workspace_and_endpoint_required")
    if not isinstance(record.get("provider_budget_reference"), str) or not record["provider_budget_reference"]:
        raise ValueError("reviewed_provider_budget_required")
    return record


def approval_check(path, frozen, plan):
    record = approval_record(path, plan, {"manifest_sha256": frozen["manifest_sha256"],
                                          "prompts_sha256": frozen["prompts_sha256"]})
    if frozen["input_tokens"] is None or "context_allowance_exceeded" in frozen["execution_blockers"]:
        raise ValueError("pinned_tokenizer_context_required")
    budget = validate_budget(record.get("monthly_budget"), plan)
    if budget["requested_allowance_usd"] < frozen["resource_estimate_gross_usd"]:
        raise ValueError("allowance_below_configured_compute_estimate")
    return budget


async def verify_weight_cache(plan, sdk):
    """Authenticated CPU control-plane inventory only; never create/stage a volume.

    Weight staging is a separate paid prerequisite. Require the pinned index and
    every indexed shard with nonzero size, rather than mistaking config for weights.
    """
    volume = sdk.Volume.from_name("recall-guard-gpt-oss-20b-weights", environment_name=plan.target.environment,
                                  create_if_missing=False)
    base = "hub/models--" + plan.artifacts.model.replace("/", "--") + "/snapshots/" + plan.artifacts.model_revision
    entries = await asyncio.wait_for(volume.listdir.aio(base, recursive=True), 30)
    files = {entry.path.lstrip("/"): entry.size for entry in entries}
    index_path = base + "/model.safetensors.index.json"
    if files.get(index_path, 0) <= 0:
        raise ValueError("cpu_staged_weight_cache_missing")
    parts = []
    async for chunk in volume.read_file.aio(index_path):
        parts.append(chunk)
        if sum(map(len, parts)) > 8 * 1024 * 1024:
            raise ValueError("weight_index_too_large")
    index = json.loads(b"".join(parts))
    shards = set(index["weight_map"].values())
    if not shards or any(PurePosixPath(shard).name != shard or files.get(base + "/" + shard, 0) <= 0 for shard in shards):
        raise ValueError("cpu_staged_weight_shards_missing")
    for filename in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        if files.get(base + "/" + filename, 0) <= 0:
            raise ValueError("cpu_staged_artifact_missing")
    return {"status": "verified_inventory", "shards": len(shards), "model_revision": plan.artifacts.model_revision}


def environment_status(plan, observations, hashes):
    ready = [row for row in observations if row.get("phase") in {"ready", "closing"}]
    if not ready:
        return "unknown"
    for row in ready:
        if (row.get("resolved_model_revision") != plan.artifacts.model_revision
                or row.get("resolved_tokenizer_revision") != plan.artifacts.tokenizer_revision
                or row.get("runtime") != plan.artifacts.runtime
                or row.get("tokenizer_sha256") != hashes
                or row.get("score_semantics") != plan.artifacts.score_semantics
                or not row.get("quantization_config") or not row.get("gpu")):
            return "unknown_or_changed"
    fields = ("resolved_model_revision", "resolved_tokenizer_revision", "runtime", "tokenizer_sha256", "gpu", "quantization_config")
    if any(any(row.get(key) != ready[0].get(key) for key in fields) for row in ready):
        return "drift"
    return "observed_stable"


def harmony_client(config, budget):
    """Final-answer evidence through the pinned modal-gpt-oss-20b Harmony profile."""
    harmony = load_profile("modal-gpt-oss-20b").harmony
    token_map = HarmonyTokenMap.from_tokenizer_json(harmony.resolve_path(), tokenizer_revision=harmony.tokenizer_revision)
    if harmony.tokenizer_sha256 is not None and token_map.tokenizer_sha256 != harmony.tokenizer_sha256:
        raise ValueError("pinned_harmony_tokenizer_mismatch")
    config = replace(config, parameters={**config.parameters, "return_token_ids": True})
    return HarmonyOpenAICompatibleClient(config, budget, token_map=token_map)


async def attempts(plan, receipt, path, *, client_factory=OpenAICompatibleClient, cancelled=lambda: False,
                   prompts=PROMPTS):
    budget = CallBudget(len(prompts))
    deadline = receipt["lifecycle"]["deadline_unix_s"]
    for position, prompt in enumerate(prompts):
        remaining = deadline - time.time()
        if cancelled() or remaining <= 0 or not receipt["lifecycle"]["admission_open"]:
            receipt["status"] = "admission_closed"
            break
        config = ModelConfig(protocol="openai_chat", provider="modal", model=plan.artifacts.model,
            endpoint=plan.target.endpoint + "/chat/completions", credential_ref=plan.target.credential_env,
            parameters=SETTINGS, max_retries=0, timeout_s=min(120, remaining))
        client = client_factory(config, budget)
        attempt = {"stage": "eligibility" if position == 0 else "measurement", "position": position,
                   "status": "intent", "input_tokens": receipt["plan"]["input_tokens"][position],
                   "output_allowance_tokens": 512, "identity_reference": receipt["app_id"],
                   "started_at_unix_s": time.time()}
        receipt["attempts"].append(attempt)
        persist(path, receipt)
        started = time.monotonic()
        try:
            # The synchronous adapter has no cancellable socket. A wall-clock timeout
            # closes admission and never replays a possibly still running POST.
            response = await asyncio.wait_for(asyncio.to_thread(client.generate_for_stage, prompt,
                stage=attempt["stage"]), timeout=min(120, remaining))
            validate_local_evidence(response, required_alternatives=5)
            attempt.update(status="eligible", distribution="partial_top_k")
            usage = response.usage or {}
            attempt["usage"] = {key: usage[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens") if key in usage}
        except LocalEvidenceError as exc:
            attempt.update(status="ineligible", reason=str(exc), diagnostic=error_diagnostic(exc, "inference"))
        except RequestError as exc:
            attempt.update(status="unavailable", reason="provider_or_transport_failure", http_status=exc.status_code,
                           unknown_outcome=True, diagnostic=error_diagnostic(exc, "inference"))
        except (TimeoutError, asyncio.CancelledError, KeyboardInterrupt) as exc:
            attempt.update(status="unavailable", reason="unknown_outcome", unknown_outcome=True,
                           diagnostic=error_diagnostic(exc, "inference"))
        except Exception as exc:
            attempt.update(status="unavailable", reason="unknown_outcome", unknown_outcome=True,
                           error=type(exc).__name__, diagnostic=error_diagnostic(exc, "inference"))
        finally:
            attempt["elapsed_seconds"] = time.monotonic() - started
            receipt["inference_posts"] = budget.used
            if budget.events:
                event = budget.events[-1]
                attempt["http_status"] = event.status_code
            persist(path, receipt)
        if attempt["status"] != "eligible" or cancelled() or time.time() >= deadline:
            receipt["status"] = attempt["status"]
            break
    else:
        receipt["status"] = "completed"


def unauthenticated_health_guard(plan, receipt, path):
    """Require proxy denial before any GPU readiness call; never send credentials."""
    event = {"operation": "unauthenticated_health", "category": "control", "status": "intent",
             "inference_posts": 0, "started_at_unix_s": time.time(), "observed_gross_usd": None}
    receipt["control_events"].append(event)
    persist(path, receipt)
    try:
        with requests.Session() as session:
            session.trust_env = False
            session.headers.clear()
            response = session.get(plan.target.endpoint.removesuffix("/v1") + "/health",
                                   allow_redirects=False, timeout=(5, 10))
            event["http_status"] = response.status_code
        if event["http_status"] != 401:
            event["status"] = "unexpected_status"
            raise ValueError("unauthenticated_health_expected_401")
        event["status"] = "denied"
    except requests.RequestException:
        event["status"] = "unavailable"
        raise ValueError("unauthenticated_health_unavailable") from None
    finally:
        event["finished_at_unix_s"] = time.time()
        persist(path, receipt)


async def run_lifecycle(plan, receipt, receipt_path, *, preflight, body, tokenizer_hashes, sdk=None):
    """Shared paid sequence for every Modal runner mode.

    Order: budget preflight, CPU weight-cache inventory, credential preflight,
    expiry-controller deploy, model deploy, armed expiry, endpoint identity,
    unauthenticated 401 guard, readiness, provenance, then ``body(app, endpoint)``
    only on an observed-stable environment, and verified stop of the model app
    and the expiry controller on every exit. ``preflight()`` returns the budget
    review and must raise before any cloud call. Never raises; failures are in
    the receipt.
    """
    app = None
    try:
        with diagnostic_stage(receipt, receipt_path, "budget_preflight"):
            receipt["budget_review"] = preflight()
            serving_plan = load_plan(serve.ROOT / plan.serving_manifest)
            if ((plan.target.app, plan.target.environment, plan.artifacts, plan.resources, plan.limits) !=
                    (serving_plan.target.app, serving_plan.target.environment, serving_plan.artifacts,
                     serving_plan.resources, serving_plan.limits)):
                raise ValueError("runner_and_uploaded_serving_manifest_differ")
        with diagnostic_stage(receipt, receipt_path, "cache_inventory"):
            sdk = serve._sdk(sdk)
            cache_event = {"operation": "weight_cache_inventory", "category": "control", "status": "intent",
                           "started_at_unix_s": time.time(), "observed_gross_usd": None}
            receipt["control_events"].append(cache_event)
            persist(receipt_path, receipt)
            receipt["weight_cache"] = await asyncio.wait_for(verify_weight_cache(plan, sdk), 60)
            cache_event.update(status="verified", finished_at_unix_s=time.time())
        with diagnostic_stage(receipt, receipt_path, "credential_preflight"):
            # Resolve credentials only after explicit execution approval; never export them.
            OpenAICompatibleClient(ModelConfig(protocol="openai_chat", model=plan.artifacts.model,
                endpoint=plan.target.endpoint + "/chat/completions", credential_ref=plan.target.credential_env), CallBudget(0))
        with diagnostic_stage(receipt, receipt_path, "controller_deploy"):
            expiry = serve.create_expiry_app(sdk=sdk, manifest=plan.serving_manifest)
            await asyncio.wait_for(expiry.deploy.aio(environment_name=plan.target.environment), 600)
        with diagnostic_stage(receipt, receipt_path, "model_deploy"):
            app = serve.create_app(sdk=sdk, manifest=plan.serving_manifest)
            receipt["gpu_started_at_unix_s"] = time.time()
            persist(receipt_path, receipt)
            await asyncio.wait_for(app.deploy.aio(environment_name=plan.target.environment), 600)
            receipt["app_id"] = app.app_id
        with diagnostic_stage(receipt, receipt_path, "expiry_arming") as lifecycle_stage:
            async with serve.experiment_lifecycle(plan, app.app_id,
                    gpu_started_at_unix_s=receipt["gpu_started_at_unix_s"], receipt=receipt["lifecycle"], sdk=sdk):
                lifecycle_stage["stage"] = "cleanup"
                with diagnostic_stage(receipt, receipt_path, "endpoint_check"):
                    service = sdk.Cls.from_name(plan.target.app, "Serving", environment_name=plan.target.environment)()
                    endpoint = await asyncio.wait_for(service.web.get_web_url.aio(), 5)
                    if endpoint != plan.target.endpoint.removesuffix("/v1"):
                        raise ValueError("deployed_endpoint_mismatch")
                with diagnostic_stage(receipt, receipt_path, "proxy_denial"):
                    persist(receipt_path, receipt)
                    await asyncio.to_thread(unauthenticated_health_guard, plan, receipt, receipt_path)
                with diagnostic_stage(receipt, receipt_path, "readiness"):
                    event = {"operation": "readiness", "category": "control", "status": "intent", "inference_posts": 0,
                             "started_at_unix_s": time.time(), "observed_gross_usd": None}
                    receipt["control_events"].append(event)
                    persist(receipt_path, receipt)
                    remaining = receipt["lifecycle"]["deadline_unix_s"] - time.time()
                    startup_remaining = receipt["gpu_started_at_unix_s"] + 600 - time.time()
                    if min(startup_remaining, remaining) <= 0:
                        raise TimeoutError("startup_allowance_exhausted")
                    readiness = await asyncio.wait_for(service.readiness.remote.aio(), min(startup_remaining, remaining))
                    if not readiness.get("ready") or readiness.get("app_id") != app.app_id:
                        raise ValueError("readiness_identity_or_process_unverified")
                    event["status"] = "ready"
                    receipt["ready_at_unix_s"] = time.time()
                    event["finished_at_unix_s"] = receipt["ready_at_unix_s"]
                with diagnostic_stage(receipt, receipt_path, "provenance"):
                    receipt["observed_before"] = await asyncio.wait_for(asyncio.to_thread(serve.read_observations, plan, app.app_id, sdk=sdk), 10)
                    receipt["environment_status"] = environment_status(plan, receipt["observed_before"], tokenizer_hashes)
                with diagnostic_stage(receipt, receipt_path, "inference"):
                    if receipt["environment_status"] == "observed_stable":
                        await body(app, endpoint)
                    else:
                        receipt["status"] = "environment_ineligible"
    except (Exception, asyncio.CancelledError, KeyboardInterrupt) as exc:
        receipt["status"] = "stopped"
        receipt.setdefault("error", type(exc).__name__)
    finally:
        # Lifecycle also writes cleanup on arming failure. A partially failed deploy
        # may already own an app; stop its exact ID even before lifecycle entry.
        if app is not None and app.app_id:
            receipt["app_id"] = app.app_id
            if "cleanup" not in receipt["lifecycle"]:
                try:
                    receipt["lifecycle"]["cleanup"] = await serve.stop_experiment(plan, app.app_id, sdk=sdk)
                except (Exception, asyncio.CancelledError, KeyboardInterrupt) as exc:
                    receipt["lifecycle"]["cleanup"] = {"cleanup": "unverified", "error": type(exc).__name__}
                    record_error(receipt, receipt_path, exc, "cleanup")
            try:
                receipt["observed_after"] = await asyncio.wait_for(asyncio.to_thread(serve.read_observations, plan, app.app_id, sdk=sdk), 10)
                receipt["environment_status"] = environment_status(plan,
                    receipt.get("observed_before", []) + receipt["observed_after"], tokenizer_hashes)
                if not any(row.get("phase") == "closing" for row in receipt["observed_after"]):
                    receipt["environment_status"] = "unknown"
            except Exception as exc:
                receipt["environment_status"] = "unknown"
                record_error(receipt, receipt_path, exc, "provenance_after")


async def execute(manifest, receipt_path, approval, *, tokenizer_root=Path("models"), sdk=None,
                  harmony_eligibility_only=False):
    serve.require_operator_python()
    frozen = planning(manifest, tokenizer_root=tokenizer_root)
    plan = load_plan(manifest)
    # An existing receipt is never resumable, even when it contains no finished calls.
    with receipt_path.open("x", encoding="utf-8"):
        pass
    receipt = {"plan": frozen, "attempts": [], "control_events": [], "lifecycle": {},
               "staging_events": [{"operation": "external_weight_staging", "status": "separate_prerequisite",
                                   "estimated_gross_usd": None, "observed_gross_usd": None}],
               "inference_posts": 0, "status": "reserved", "calibration": "unavailable",
               "environment_status": "unknown", "evidence_status": "unobserved",
               "mode": "harmony_eligibility" if harmony_eligibility_only else "pilot"}
    persist(receipt_path, receipt)

    async def body(app, endpoint):
        if harmony_eligibility_only:
            # One eligibility POST only; measurement prompts stay unsent.
            await attempts(plan, receipt, receipt_path, client_factory=harmony_client, prompts=PROMPTS[:1])
        else:
            await attempts(plan, receipt, receipt_path)

    try:
        await run_lifecycle(plan, receipt, receipt_path, preflight=lambda: approval_check(approval, frozen, plan),
                            body=body, tokenizer_hashes=frozen["tokenizer_sha256"], sdk=sdk)
    finally:
        receipt["evidence_status"] = ("eligible" if receipt["attempts"] and all(
            row["status"] == "eligible" for row in receipt["attempts"]) else "ineligible_or_unobserved")
        receipt["finished_at_unix_s"] = time.time()
        receipt["cost_report"] = cost_report(receipt, frozen["cost_review"], plan.resources)
        persist(receipt_path, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--tokenizer-root", type=Path, default=Path("models"))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--harmony-eligibility-only", action="store_true",
                        help="with --execute: send only the first prompt, through the Harmony final-answer profile")
    parser.add_argument("--reconcile-billing", type=Path,
                        help="append a reviewed local billing snapshot to --receipt; no cloud access")
    args = parser.parse_args()
    if args.execute and (args.receipt is None or args.approval is None):
        parser.error("--execute requires --receipt NEW_PATH and --approval REVIEWED_RECORD")
    if args.harmony_eligibility_only and not args.execute:
        parser.error("--harmony-eligibility-only requires --execute")
    if args.reconcile_billing:
        if args.execute or args.receipt is None:
            parser.error("--reconcile-billing requires --receipt and excludes --execute")
        result = json.loads(args.receipt.read_text(encoding="utf-8"))
        append_billing(result, json.loads(args.reconcile_billing.read_text(encoding="utf-8")))
        # Use the original frozen cost basis, never a later edited manifest.
        result["cost_report"] = cost_report(result, result["plan"]["cost_review"], Resources(**result["plan"]["resources"]))
        persist(args.receipt, result)
    elif args.execute:
        result = asyncio.run(execute(args.manifest, args.receipt, args.approval, tokenizer_root=args.tokenizer_root,
                                     harmony_eligibility_only=args.harmony_eligibility_only))
    else:
        result = planning(args.manifest, tokenizer_root=args.tokenizer_root)
    print(json.dumps(result, indent=2))
    return 0 if args.reconcile_billing or result.get("mode") == "plan" or result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
