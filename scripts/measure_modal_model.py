"""Run a frozen measurement plan against a deployed Modal serving manifest's endpoint.

Default is a dry run: it reads only local files (no credentials, no SDK, no
network) and prints planned counts, the cost estimate and blockers. Execution
requires ``--execute`` with a reviewed approval record, a new journal and a new
receipt, and runs inside the shared pilot lifecycle (deploy, armed expiry,
endpoint identity, 401 guard, readiness, provenance, verified stop of the model
app and the expiry controller). Requests use the local measurement journal,
execution and report code with the serving manifest's provider profile: the
``modal-gpt-oss-20b`` Harmony profile for the GPT-OSS-20B manifests and the
generic-mapping ``modal-gemma3-4b`` profile for the Gemma 3 4B manifest.

Operator container: sh scripts/modal_operator.sh measure-plan PLAN MANIFEST CORPUS [APPROVAL] [--modal-manifest PATH]
"""

import argparse
import asyncio
import json
import math
import os
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deploy.modal import serve
from deploy.modal.manifest import GEMMA_MANIFEST, MEASUREMENT_MANIFEST, PILOT_MANIFEST, load_plan
from recall_guard.core.local_evidence import LocalModelProvenance
from recall_guard.core.profiles import connect, load_profile
from recall_guard.core.request_control import BudgetExhaustedError, CallBudget
from recall_guard.harness.local_evaluation import LocalAttemptBudget
from recall_guard.harness.local_evaluation import load_plan as load_measurement_plan
from recall_guard.harness.local_execution import HostObservation, execute_plan
from recall_guard.harness.local_reporting import build_report, report_paths, write_report
from scripts.modal_cost_report import cost_report, validate_budget
from scripts.probe_modal_model import (
    approval_record,
    digest,
    persist,
    run_lifecycle,
    tokenizer_hashes,
)

PROFILE_ID = "modal-gpt-oss-20b"
#: Provider profile per reviewed serving manifest; each manifest has exactly one.
PROFILES = {PILOT_MANIFEST: PROFILE_ID, MEASUREMENT_MANIFEST: PROFILE_ID, GEMMA_MANIFEST: "modal-gemma3-4b"}
MODAL_MANIFEST = Path(MEASUREMENT_MANIFEST)
# Expiry controller request (serve.create_expiry_app): 0.125 core, 128 MiB,
# running for the GPU phase plus its 150 s stop/verification allowance.
CONTROLLER_CORES, CONTROLLER_GIB, CONTROLLER_EXTRA_SECONDS = 0.125, 0.125, 150


def profile_id(modal_plan):
    """The provider profile reviewed for this serving manifest."""
    return PROFILES[modal_plan.serving_manifest]


def compute_rate(cost, resources):
    """Configured resource request rate in gross USD per second (not an invoice)."""
    return (cost["gpu_usd_per_second"] * resources.gpu_count
            + cost["physical_core_usd_per_second"] * resources.cpu_physical_cores
            + cost["memory_gib_usd_per_second"] * resources.memory_gib)


def cost_estimate(cost, modal_plan, cap_usd=None):
    """Worst case: the full GPU phase plus the cleanup tail and the expiry controller.

    The admission window is the GPU phase, shortened if the USD cap affords less.
    """
    rate = compute_rate(cost, modal_plan.resources)
    limits = modal_plan.limits
    controller_rate = (CONTROLLER_CORES * cost["physical_core_usd_per_second"]
                       + CONTROLLER_GIB * cost["memory_gib_usd_per_second"])
    controller = controller_rate * (limits.gpu_phase_seconds + CONTROLLER_EXTRA_SECONDS)
    gpu = rate * (limits.gpu_phase_seconds + limits.cleanup_seconds)
    window = limits.gpu_phase_seconds
    if cap_usd is not None:
        affordable = math.floor((cap_usd - controller) / rate) - limits.cleanup_seconds
        window = max(0, min(window, affordable))
    return {"rate_usd_per_second": rate, "gpu_phase_seconds": limits.gpu_phase_seconds,
            "cleanup_seconds": limits.cleanup_seconds, "worst_case_gpu_usd": gpu,
            "worst_case_expiry_controller_usd": controller, "worst_case_total_usd": gpu + controller,
            "cap_usd": cap_usd, "admission_window_seconds": window,
            "basis": "configured_rates_times_full_gpu_phase_plus_cleanup_not_invoice"}


def plan_blockers(modal_plan, local_plan, profile, hashes, estimate, cost):
    """Hard mismatches between the frozen measurement plan and the serving manifest."""
    blockers = []
    config, limits, modal_limits = local_plan.config, local_plan.limits, modal_plan.limits
    if modal_plan.target.endpoint is None:
        blockers.append("modal_endpoint_unset")
    elif config.endpoint != modal_plan.target.endpoint + "/chat/completions":
        blockers.append("plan_endpoint_differs_from_manifest")
    if config.model != modal_plan.artifacts.model:
        blockers.append("plan_model_differs_from_manifest")
    # A profile that declares a model identity (the Gemma revision) binds the plan to it.
    declared = profile.model_config.local_identity
    if (replace(config, local_identity=declared, endpoint=profile.model_config.endpoint) != profile.model_config
            or (declared is not None and (config.local_identity is None
                                          or config.local_identity.declared != declared.declared))):
        blockers.append("plan_config_differs_from_profile")
    if config.credential_ref != "env:" + modal_plan.target.credential_env:
        blockers.append("plan_credential_reference_differs_from_manifest")
    if local_plan.authentication != "environment" or local_plan.requested_execution_mode != "GPU":
        blockers.append("remote_gpu_with_environment_credential_required")
    if local_plan.physical_requests > modal_limits.inference_posts:
        blockers.append("planned_requests_exceed_manifest_posts")
    if config.max_retries != modal_limits.retries:
        blockers.append("retries_differ_from_manifest")
    if limits.max_output_tokens != modal_limits.output_tokens_per_post:
        blockers.append("output_tokens_differ_from_manifest")
    if limits.max_requests * limits.max_output_tokens > modal_limits.total_output_tokens:
        blockers.append("total_output_tokens_exceed_manifest")
    if limits.context_tokens != modal_plan.resources.context_tokens:
        blockers.append("context_differs_from_manifest")
    if config.timeout_s > modal_limits.request_seconds:
        blockers.append("request_timeout_exceeds_manifest")
    if limits.max_elapsed_s > modal_limits.gpu_phase_seconds:
        blockers.append("elapsed_bound_exceeds_gpu_phase")
    settings = {key: config.parameters.get(key) for key in ("temperature", "top_p", "logprobs", "top_logprobs")}
    if settings != {key: getattr(modal_limits, key) for key in settings}:
        blockers.append("generation_settings_differ_from_manifest")
    # Only the Harmony mapping needs the pinned Harmony tokenizer; the generic mapping has none.
    if profile.harmony is not None and not profile.harmony.resolve_path().is_file():
        blockers.append("pinned_harmony_tokenizer_unavailable")
    if hashes is None:
        blockers.append("pinned_tokenizer_unavailable")
    if estimate["worst_case_total_usd"] > cost["target_usd"]:
        blockers.append("worst_case_exceeds_reviewed_usd_target")
    return blockers


def measurement_planning(plan_path, manifest_path, corpus_path, *, modal_manifest=MODAL_MANIFEST,
                         tokenizer_root=Path("models"), approval=None):
    """Offline dry run: local files only; no credential, SDK or network access."""
    modal_plan = load_plan(modal_manifest)
    local_plan = load_measurement_plan(plan_path, manifest_path, corpus_path)
    selected = profile_id(modal_plan)
    profile = load_profile(selected)
    cost = json.loads(Path(modal_manifest).read_text(encoding="utf-8"))["cost"]
    hashes = tokenizer_hashes(modal_plan, tokenizer_root)
    estimate = cost_estimate(cost, modal_plan)
    blockers = plan_blockers(modal_plan, local_plan, profile, hashes, estimate, cost)
    splits = {split: sum(item.split == split for item in local_plan.items)
              for split in ("is_memorized", "oos_control", "held_out")}
    limits = local_plan.limits
    frozen = {
        "mode": "measurement_plan", "profile": selected,
        "manifest_sha256": digest(Path(modal_manifest).read_bytes()),
        "plan_sha256": digest(Path(plan_path).read_bytes()),
        "measurement_manifest_sha256": local_plan.manifest_sha256, "corpus_sha256": local_plan.corpus_sha256,
        "serving_manifest": modal_plan.serving_manifest, "tokenizer_sha256": hashes,
        "items_per_split": splits, "ensemble_draws": local_plan.ensemble_draws,
        "logical_requests": local_plan.logical_requests, "physical_requests": local_plan.physical_requests,
        "max_input_tokens": limits.max_input_tokens, "max_output_tokens": limits.max_output_tokens,
        "worst_case_tokens": local_plan.physical_requests * (limits.max_input_tokens + limits.max_output_tokens),
        "plan_limits": {"max_requests": limits.max_requests, "max_total_tokens": limits.max_total_tokens,
                        "max_elapsed_s": limits.max_elapsed_s, "context_tokens": limits.context_tokens},
        "manifest_limits": {"inference_posts": modal_plan.limits.inference_posts,
                            "total_output_tokens": modal_plan.limits.total_output_tokens,
                            "gpu_phase_seconds": modal_plan.limits.gpu_phase_seconds},
        "cost_review": cost, "cost_estimate": estimate, "resources": asdict(modal_plan.resources),
        "plan_blockers": blockers,
        "execution_blockers": [*modal_plan.execution_blockers, *blockers],
        "inference_posts": 0, "cloud_mutations": 0,
    }
    if approval is not None:
        try:
            frozen["approval"] = {"status": "valid", **measurement_approval(approval, frozen, modal_plan)}
        except (ValueError, KeyError, TypeError, OSError) as exc:
            reason = str(exc) if isinstance(exc, ValueError) else type(exc).__name__
            frozen["approval"] = {"status": "refused", "reason": reason}
    return frozen


def approval_hashes(frozen):
    return {key: frozen[key] for key in ("manifest_sha256", "plan_sha256", "measurement_manifest_sha256",
                                          "corpus_sha256")}


def measurement_approval(path, frozen, modal_plan):
    """Same five assertions and budget block as the pilot, bound to these input hashes.

    The record's requested_allowance_usd is the run's USD cap: the worst-case
    estimate must fit inside it, and it bounds the admission window.
    """
    record = approval_record(path, modal_plan, approval_hashes(frozen))
    if frozen["plan_blockers"]:
        raise ValueError("measurement_plan_blocked")
    budget = validate_budget(record.get("monthly_budget"), modal_plan)
    estimate = cost_estimate(frozen["cost_review"], modal_plan, cap_usd=budget["requested_allowance_usd"])
    if budget["requested_allowance_usd"] < estimate["worst_case_total_usd"]:
        raise ValueError("usd_cap_below_worst_case_estimate")
    return {**budget, "cost_estimate": estimate}


class ModalAttemptBudget(LocalAttemptBudget):
    """Local measurement journal with a hard wall-clock admission deadline.

    Admission closes when the lifecycle lease closes or when a full request
    timeout no longer fits before the deadline; nothing is ever replayed.
    The client is built with ``connect`` semantics from the serving manifest's
    profile (Harmony for GPT-OSS-20B by default), its endpoint taken from the
    deployed URL, which must equal the plan endpoint.
    """

    def __init__(self, plan, path, *, deadline_unix_s, lease, endpoint, wall_clock=time.time,
                 profile_id=PROFILE_ID, **options):
        self.deadline_unix_s, self.lease, self.endpoint, self._wall_clock = deadline_unix_s, lease, endpoint, wall_clock
        self.profile_id = profile_id
        super().__init__(plan, path, **options)

    def client(self, *, environment=None):
        environment = dict(os.environ if environment is None else environment)
        profile = load_profile(self.profile_id)
        environment[profile.endpoint_ref.removeprefix("env:")] = self.endpoint
        client = connect(replace(profile, model_config=self.plan.config), self, environment=environment)
        if client.config.endpoint != self.plan.config.endpoint:
            raise ValueError("deployed_endpoint_differs_from_plan")
        credential = environment.get(self.plan.config.credential_ref.removeprefix("env:"))
        self._report_secrets = (credential,) if credential else ()
        return client

    def _admit(self, **kwargs):
        if (not self.lease.get("admission_open")
                or self._wall_clock() + self.plan.config.timeout_s > self.deadline_unix_s):
            with self._journal_lock:
                self._stopped = True
            raise BudgetExhaustedError("modal_gpu_phase_admission_closed")
        return super()._admit(**kwargs)


def _max_model_len(command):
    if isinstance(command, list) and "--max-model-len" in command[:-1]:
        value = command[command.index("--max-model-len") + 1]
        return int(value) if str(value).isdigit() else None
    return None


def modal_observer(modal_plan, app_id, model, sdk=None):
    """Serving provenance (SDK Dict reads only) as the local sidecar observation.

    One ready container without a closing record counts as the loaded model;
    a restart or closed container leaves nothing loaded and fails validation.
    """
    def observe():
        rows = serve.read_observations(modal_plan, app_id, sdk=sdk)
        ready = [row for row in rows if row.get("phase") == "ready"]
        closed = any(row.get("phase") == "closing" for row in rows)
        row = ready[0] if len(ready) == 1 else {}
        gpu = row.get("gpu") or []
        runtime = row.get("runtime") or ""
        provenance = LocalModelProvenance(
            runtime="vllm" if runtime.startswith("vllm==") else None,
            runtime_version=runtime.removeprefix("vllm==") or None,
            artifact_revision=row.get("resolved_model_revision"),
            quantization=(row.get("quantization_config") or {}).get("quant_method"),
            tokenizer=row.get("resolved_tokenizer_revision"),
            backend="GPU" if gpu else None,
            driver_version=gpu[0].get("driver_version") if gpu else None,
            context_length=_max_model_len(row.get("server_command")))
        return HostObservation(provenance, (model,) if len(ready) == 1 and not closed else ())
    return observe


def journal_attempts(journal):
    """Cost-report rows from the journal: status and timing only, never content."""
    return [{"stage": "eligibility" if row.get("purpose") == "probe" else "measurement",
             "status": "eligible" if row.get("status") == "succeeded" else row.get("status"),
             "http_status": row.get("http_status"), "elapsed_seconds": row.get("latency_s") or 0}
            for row in journal.get("attempts", [])]


async def execute_measurement(plan_path, manifest_path, corpus_path, approval, journal, receipt_path, *,
                              modal_manifest=MODAL_MANIFEST, tokenizer_root=Path("models"), sdk=None):
    serve.require_operator_python()
    frozen = measurement_planning(plan_path, manifest_path, corpus_path, modal_manifest=modal_manifest,
                                  tokenizer_root=tokenizer_root)
    modal_plan = load_plan(modal_manifest)
    local_plan = load_measurement_plan(plan_path, manifest_path, corpus_path)
    if journal.exists() or any(path.exists() for path in report_paths(journal)):
        raise FileExistsError("measurement_journal_or_report_exists")
    # Neither receipts nor journals are resumable; both are created exclusively.
    with receipt_path.open("x", encoding="utf-8"):
        pass
    receipt = {"mode": "measurement", "plan": frozen, "attempts": [], "control_events": [], "lifecycle": {},
               "staging_events": [{"operation": "external_weight_staging", "status": "separate_prerequisite",
                                   "estimated_gross_usd": None, "observed_gross_usd": None}],
               "inference_posts": 0, "status": "reserved", "environment_status": "unknown",
               "journal": journal.name, "measurement": {"status": "not_started"}}
    persist(receipt_path, receipt)

    async def body(app, endpoint):
        window = receipt["budget_review"]["cost_estimate"]["admission_window_seconds"]
        deadline = min(receipt["lifecycle"]["deadline_unix_s"], receipt["gpu_started_at_unix_s"] + window)
        budget = ModalAttemptBudget(local_plan, journal, deadline_unix_s=deadline, lease=receipt["lifecycle"],
                                    endpoint=endpoint + "/v1/chat/completions", profile_id=frozen["profile"])
        receipt["measurement"] = {"status": "executing", "admission_deadline_unix_s": deadline}
        persist(receipt_path, receipt)
        observer = modal_observer(modal_plan, app.app_id, local_plan.config.model, sdk)
        try:
            # The deadline closes admission; the final request is bounded by its own timeout.
            result = await asyncio.wait_for(asyncio.to_thread(execute_plan, budget, observer),
                                            max(1.0, deadline - time.time()) + modal_plan.limits.request_seconds)
        except BaseException:
            budget.cancellation.set()
            raise
        finally:
            receipt["inference_posts"] = budget.used
            receipt["attempts"] = journal_attempts(budget.safe_receipt())
            persist(receipt_path, receipt)
        report = build_report(budget, result)
        write_report(journal, report)
        receipt["measurement"].update(
            status=result.status, calibrated_claims_allowed=result.calibrated_claims_allowed,
            observed_execution_mode=result.observed_execution_mode, physical_attempts=budget.used,
            stages=report["stages"], report=[path.name for path in report_paths(journal)])
        receipt["status"] = "completed" if result.status == "completed" else result.status

    def preflight():
        review = measurement_approval(approval, frozen, modal_plan)
        # Credential and pinned tokenizer presence through the profile; no request.
        profile = load_profile(frozen["profile"])
        environment = dict(os.environ)
        environment[profile.endpoint_ref.removeprefix("env:")] = local_plan.config.endpoint
        connect(replace(profile, model_config=local_plan.config), CallBudget(0), environment=environment)
        return review

    try:
        await run_lifecycle(modal_plan, receipt, receipt_path, preflight=preflight, body=body,
                            tokenizer_hashes=frozen["tokenizer_sha256"], sdk=sdk)
    finally:
        receipt["finished_at_unix_s"] = time.time()
        receipt["cost_report"] = cost_report(receipt, frozen["cost_review"], modal_plan.resources)
        persist(receipt_path, receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path, help="measurement manifest (local plan format)")
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--modal-manifest", type=Path, default=MODAL_MANIFEST)
    parser.add_argument("--tokenizer-root", type=Path, default=Path("models"))
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--execute", action="store_true", help="paid run; default is an offline dry run")
    parser.add_argument("--journal", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args(argv)
    if args.execute and (args.approval is None or args.journal is None or args.receipt is None):
        parser.error("--execute requires --approval REVIEWED_RECORD, a new --journal and a new --receipt")
    try:
        if not args.execute:
            result = measurement_planning(args.plan, args.manifest, args.corpus, modal_manifest=args.modal_manifest,
                                          tokenizer_root=args.tokenizer_root, approval=args.approval)
            print(json.dumps(result, indent=2))
            return 0
        result = asyncio.run(execute_measurement(args.plan, args.manifest, args.corpus, args.approval, args.journal,
                                                 args.receipt, modal_manifest=args.modal_manifest,
                                                 tokenizer_root=args.tokenizer_root))
    except (ValueError, TypeError, KeyError, OSError):
        # Parsing failures may contain private paths or corpus material.
        print("Modal measurement refused: invalid plan, manifest, corpus, journal or receipt.", file=sys.stderr)
        return 2
    print(json.dumps({key: result.get(key) for key in (
        "status", "error", "error_stage", "inference_posts", "environment_status", "measurement", "app_id")}
        | {"cleanup": result["lifecycle"].get("cleanup", {}).get("cleanup"),
           "expiry_cleanup": result["lifecycle"].get("expiry_cleanup", {}).get("cleanup")}, indent=2))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
