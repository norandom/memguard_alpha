"""Bounded Modal pilot. Default planning reads no credentials and calls no SDK.

Execution requires a separately reviewed approval record and already staged weights.
Run from the checkout using ``uv run --no-sync python scripts/probe_modal_model.py``.
"""

import argparse
import asyncio
import hashlib
import json
import os
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deploy.modal import serve
from deploy.modal.manifest import Resources, load_plan
from recall_guard.core.completion import ModelConfig
from recall_guard.core.local_evidence import LocalEvidenceError, validate_local_evidence
from recall_guard.core.providers import OpenAICompatibleClient
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


def digest(data):
    return hashlib.sha256(data).hexdigest()


def tokenizer_counts(plan, root):
    """Count the actual one-user-message chat wire input with the pinned template.

    No downloads, fallback estimates or fabricated usage. Only the exact HF snapshot
    layout is accepted. The serving provenance must later match these file hashes.
    """
    directory = (root / "hub" / ("models--" + plan.artifacts.tokenizer.replace("/", "--"))
                 / "snapshots" / plan.artifacts.tokenizer_revision)
    files = [directory / name for name in ("tokenizer.json", "tokenizer_config.json")]
    if any(not file.is_file() for file in files):
        return None, None
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False)
    counts = [len(tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True, add_generation_prompt=True
    )) for prompt in PROMPTS]
    hashes = {file.name: digest(file.read_bytes()) for file in files}
    template = directory / "chat_template.jinja"
    hashes[template.name] = digest(template.read_bytes()) if template.is_file() else None
    return counts, hashes


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


def approval_check(path, frozen, plan):
    record = json.loads(path.read_text(encoding="utf-8"))
    required = ("paid_execution_authorized", "account_access_verified", "operator_allowance_confirmed",
                "model_access_verified", "pricing_and_budget_reviewed")
    if any(record.get(key) is not True for key in required):
        raise ValueError("execution_approval_or_prerequisites_missing")
    if record.get("manifest_sha256") != frozen["manifest_sha256"] or record.get("prompts_sha256") != frozen["prompts_sha256"]:
        raise ValueError("approval_plan_mismatch")
    if record.get("workspace") != plan.target.workspace or not plan.target.workspace or not plan.target.endpoint:
        raise ValueError("explicit_workspace_and_endpoint_required")
    if not isinstance(record.get("provider_budget_reference"), str) or not record["provider_budget_reference"]:
        raise ValueError("reviewed_provider_budget_required")
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


async def attempts(plan, receipt, path, *, client_factory=OpenAICompatibleClient, cancelled=lambda: False):
    budget = CallBudget(8)
    deadline = receipt["lifecycle"]["deadline_unix_s"]
    for position, prompt in enumerate(PROMPTS):
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
            attempt.update(status="ineligible", reason=str(exc))
        except RequestError as exc:
            attempt.update(status="unavailable", reason="provider_or_transport_failure", http_status=exc.status_code,
                           unknown_outcome=True)
        except (TimeoutError, asyncio.CancelledError, KeyboardInterrupt):
            attempt.update(status="unavailable", reason="unknown_outcome", unknown_outcome=True)
        except Exception as exc:
            attempt.update(status="unavailable", reason="unknown_outcome", unknown_outcome=True,
                           error=type(exc).__name__)
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


async def execute(manifest, receipt_path, approval, *, tokenizer_root=Path("models"), sdk=None):
    frozen = planning(manifest, tokenizer_root=tokenizer_root)
    plan = load_plan(manifest)
    # An existing receipt is never resumable, even when it contains no finished calls.
    with receipt_path.open("x", encoding="utf-8"):
        pass
    receipt = {"plan": frozen, "attempts": [], "control_events": [], "lifecycle": {},
               "staging_events": [{"operation": "external_weight_staging", "status": "separate_prerequisite",
                                   "estimated_gross_usd": None, "observed_gross_usd": None}],
               "inference_posts": 0, "status": "reserved", "calibration": "unavailable",
               "environment_status": "unknown", "evidence_status": "unobserved"}
    persist(receipt_path, receipt)
    app = None
    try:
        receipt["budget_review"] = approval_check(approval, frozen, plan)
        serving_plan = load_plan(serve.ROOT / "configs/modal/gpt-oss-20b-pilot.json")
        if ((plan.target.app, plan.target.environment, plan.artifacts, plan.resources, plan.limits) !=
                (serving_plan.target.app, serving_plan.target.environment, serving_plan.artifacts,
                 serving_plan.resources, serving_plan.limits)):
            raise ValueError("runner_and_uploaded_serving_manifest_differ")
        sdk = serve._sdk(sdk)
        cache_event = {"operation": "weight_cache_inventory", "category": "control", "status": "intent",
                       "started_at_unix_s": time.time(), "observed_gross_usd": None}
        receipt["control_events"].append(cache_event)
        persist(receipt_path, receipt)
        receipt["weight_cache"] = await asyncio.wait_for(verify_weight_cache(plan, sdk), 60)
        cache_event.update(status="verified", finished_at_unix_s=time.time())
        # Resolve credentials only after explicit execution approval; never export them.
        OpenAICompatibleClient(ModelConfig(protocol="openai_chat", model=plan.artifacts.model,
            endpoint=plan.target.endpoint + "/chat/completions", credential_ref=plan.target.credential_env), CallBudget(0))
        expiry = serve.create_expiry_app(sdk=sdk)
        await asyncio.wait_for(expiry.deploy.aio(environment_name=plan.target.environment), 600)
        app = serve.create_app(sdk=sdk)
        receipt["gpu_started_at_unix_s"] = time.time()
        persist(receipt_path, receipt)
        await asyncio.wait_for(app.deploy.aio(environment_name=plan.target.environment), 600)
        receipt["app_id"] = app.app_id
        async with serve.experiment_lifecycle(plan, app.app_id,
                gpu_started_at_unix_s=receipt["gpu_started_at_unix_s"], receipt=receipt["lifecycle"], sdk=sdk):
            persist(receipt_path, receipt)
            event = {"operation": "readiness", "category": "control", "status": "intent", "inference_posts": 0,
                     "started_at_unix_s": time.time(), "observed_gross_usd": None}
            receipt["control_events"].append(event)
            persist(receipt_path, receipt)
            remaining = receipt["lifecycle"]["deadline_unix_s"] - time.time()
            service = sdk.Cls.from_name(plan.target.app, "Serving", environment_name=plan.target.environment)()
            startup_remaining = receipt["gpu_started_at_unix_s"] + 600 - time.time()
            if min(startup_remaining, remaining) <= 0:
                raise TimeoutError("startup_allowance_exhausted")
            readiness = await asyncio.wait_for(service.readiness.remote.aio(), min(startup_remaining, remaining))
            if not readiness.get("ready") or readiness.get("app_id") != app.app_id:
                raise ValueError("readiness_identity_or_process_unverified")
            event["status"] = "ready"
            receipt["ready_at_unix_s"] = time.time()
            event["finished_at_unix_s"] = receipt["ready_at_unix_s"]
            endpoint = await asyncio.wait_for(service.web.get_web_url.aio(), 5)
            if endpoint != plan.target.endpoint.removesuffix("/v1"):
                raise ValueError("deployed_endpoint_mismatch")
            receipt["observed_before"] = await asyncio.wait_for(asyncio.to_thread(serve.read_observations, plan, app.app_id, sdk=sdk), 10)
            receipt["environment_status"] = environment_status(plan, receipt["observed_before"], frozen["tokenizer_sha256"])
            if receipt["environment_status"] == "observed_stable":
                await attempts(plan, receipt, receipt_path)
            else:
                receipt["status"] = "environment_ineligible"
    except (Exception, asyncio.CancelledError, KeyboardInterrupt) as exc:
        receipt.update(status="stopped", error=type(exc).__name__)
    finally:
        # Lifecycle also writes cleanup on arming failure. A partially failed deploy
        # may already own an app; stop its exact ID even before lifecycle entry.
        if app is not None and app.app_id:
            receipt["app_id"] = app.app_id
            if "cleanup" not in receipt["lifecycle"]:
                receipt["lifecycle"]["cleanup"] = await serve.stop_experiment(plan, app.app_id, sdk=sdk)
            try:
                receipt["observed_after"] = await asyncio.wait_for(asyncio.to_thread(serve.read_observations, plan, app.app_id, sdk=sdk), 10)
                receipt["environment_status"] = environment_status(plan,
                    receipt.get("observed_before", []) + receipt["observed_after"], frozen["tokenizer_sha256"])
                if not any(row.get("phase") == "closing" for row in receipt["observed_after"]):
                    receipt["environment_status"] = "unknown"
            except Exception:
                receipt["environment_status"] = "unknown"
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
    parser.add_argument("--reconcile-billing", type=Path,
                        help="append a reviewed local billing snapshot to --receipt; no cloud access")
    args = parser.parse_args()
    if args.execute and (args.receipt is None or args.approval is None):
        parser.error("--execute requires --receipt NEW_PATH and --approval REVIEWED_RECORD")
    if args.reconcile_billing:
        if args.execute or args.receipt is None:
            parser.error("--reconcile-billing requires --receipt and excludes --execute")
        result = json.loads(args.receipt.read_text(encoding="utf-8"))
        append_billing(result, json.loads(args.reconcile_billing.read_text(encoding="utf-8")))
        # Use the original frozen cost basis, never a later edited manifest.
        result["cost_report"] = cost_report(result, result["plan"]["cost_review"], Resources(**result["plan"]["resources"]))
        persist(args.receipt, result)
    elif args.execute:
        result = asyncio.run(execute(args.manifest, args.receipt, args.approval, tokenizer_root=args.tokenizer_root))
    else:
        result = planning(args.manifest, tokenizer_root=args.tokenizer_root)
    print(json.dumps(result, indent=2))
    return 0 if args.reconcile_billing or result.get("mode") == "plan" or result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
