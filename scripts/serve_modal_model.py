"""Serving route of the Modal arm: stage the pinned weights, serve the app, stop it.

Each mode reads one serving manifest and writes a private receipt; ``stage`` and
``serve`` also take one stage's limits file from the frozen study plan
(deploy/modal/stage_limits.py). Neither mode has an approval gate of its own: the
study plan's approval record is the only approval, checked by the study runner
before it sends any request. Credentials pass by name only and never reach a
receipt or the output.

- ``stage``: CPU-only weight staging of the manifest's pinned revision into the
  manifest's own volume, with HF_TOKEN passed by name. It refuses, before any SDK
  call, when running to its timeout could cost more than the stage's ``max_usd``.
- ``serve``: deploys the expiry controller and the serving app, arms the controller
  at the GPU start plus the stage's ``max_seconds`` (never the manifest's GPU
  phase), records the registered endpoint URL and the deadline, and exits leaving
  the app reachable. The study runner sets MODAL_ENDPOINT_URL from the receipt and
  authenticates with MODAL_PROXY_BEARER. A failed serve stops what it started and
  reports ``stopped`` only when no app remains, otherwise ``unverified`` with a
  recovery command.
- ``stop-serve``: stops the serving app by its ID and then the expiry controller,
  and records ``verified_stopped`` or ``unverified`` in the serve receipt.

Operator container: sh scripts/modal_operator.sh stage|serve MANIFEST STAGE_LIMITS RECEIPT
and sh scripts/modal_operator.sh stop-serve MANIFEST RECEIPT
"""

import argparse
import asyncio
import json
import math
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deploy.modal import serve
from deploy.modal.manifest import load_plan
from deploy.modal.stage_limits import parse_stage_limits, require_serving_limits, require_staging_limits
from scripts.probe_modal_model import (
    diagnostic_stage,
    diagnostic_text,
    digest,
    persist,
    record_error,
    verify_weight_cache,
)

#: A deployed app's provider-registered web URL: no path, query or user information.
REGISTERED_URL = re.compile(r"https://[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)?\.modal\.run")
ENDPOINT_ENV = "MODAL_ENDPOINT_URL"
#: Local wait beyond the staging timeout, for container startup and the result.
STAGING_WAIT_SLACK_SECONDS = serve.STAGING_STARTUP_SECONDS + 60


def stage_inputs(manifest: Path, stage_limits: Path):
    """Local files only: the reviewed plan, its cost rates and the validated stage limits."""
    plan = load_plan(manifest)
    reviewed = serve.ROOT / plan.serving_manifest
    uploaded = load_plan(reviewed)
    cost = json.loads(manifest.read_text(encoding="utf-8"))["cost"]
    if ((plan.target.app, plan.target.environment, plan.artifacts, plan.resources, plan.limits, cost)
            != (uploaded.target.app, uploaded.target.environment, uploaded.artifacts, uploaded.resources,
                uploaded.limits, json.loads(reviewed.read_text(encoding="utf-8"))["cost"])):
        raise ValueError("Manifest differs from the reviewed serving manifest that is uploaded")
    data = stage_limits.read_bytes()
    limits = parse_stage_limits(data, plan)
    receipt = {"manifest": plan.serving_manifest, "manifest_sha256": digest(manifest.read_bytes()),
               "plan_sha256": limits.plan_sha256, "stage_limits": asdict(limits),
               "stage_limits_sha256": digest(data), "volume": plan.volume_name}
    return plan, limits, cost, receipt


def reserve(path: Path):
    """A receipt is never reused, even when it records no cloud call."""
    with path.open("x", encoding="utf-8"):
        pass


def staged_summary(plan, result) -> dict:
    """Copy only the expected fields; any other remote output stays out of the receipt."""
    expected = {"volume": plan.volume_name, "model_revision": plan.artifacts.model_revision,
                "tokenizer_revision": plan.artifacts.tokenizer_revision}
    if not isinstance(result, dict) or any(result.get(key) != value for key, value in expected.items()):
        raise ValueError("Staging result differs from the manifest's volume or revision")
    sizes = {key: result.get(key) for key in ("files", "bytes")}
    if any(type(value) is not int or value <= 0 for value in sizes.values()):
        raise ValueError("Staging result has no file sizes")
    duration = result.get("duration_seconds")
    if type(duration) not in (int, float) or not math.isfinite(duration) or duration < 0:
        raise ValueError("Staging result has no duration")
    return expected | sizes | {"duration_seconds": float(duration)}


async def stage(manifest: Path, stage_limits: Path, receipt_path: Path, *, sdk=None) -> dict:
    """CPU-only weight staging within its stage limit; failures after the receipt are in it."""
    serve.require_operator_python()
    plan, limits, cost, receipt = stage_inputs(manifest, stage_limits)
    worst = require_staging_limits(limits, plan, cost)
    reserve(receipt_path)
    receipt.update(mode="stage", status="reserved", worst_case_usd=worst, timeout_seconds=limits.max_seconds,
                   resources={"gpu": None, "cpu_cores": plan.resources.cpu_physical_cores,
                              "memory_gib": plan.resources.memory_gib},
                   secret_env=["HF_TOKEN"], approval_gate="study_plan_record_checked_by_runner")
    persist(receipt_path, receipt)
    try:
        with diagnostic_stage(receipt, receipt_path, "staging"):
            sdk = serve._sdk(sdk)
            app, function = serve.create_staging_app(max_seconds=limits.max_seconds, sdk=sdk,
                                                     manifest=plan.serving_manifest)
            receipt["started_at_unix_s"] = time.time()
            persist(receipt_path, receipt)
            # The ephemeral app stops when this block exits, at the latest after its timeout.
            async with app.run(environment_name=plan.target.environment):
                result = await asyncio.wait_for(function.remote.aio(),
                                                limits.max_seconds + STAGING_WAIT_SLACK_SECONDS)
            receipt["finished_at_unix_s"] = time.time()
            receipt["staged"] = staged_summary(plan, result)
        with diagnostic_stage(receipt, receipt_path, "cache_inventory"):
            receipt["weight_cache"] = await asyncio.wait_for(verify_weight_cache(plan, sdk), 60)
        receipt["status"] = "staged"
    except (Exception, asyncio.CancelledError, KeyboardInterrupt) as exc:
        receipt["status"] = "stopped"
        receipt.setdefault("error", type(exc).__name__)
    finally:
        receipt.setdefault("finished_at_unix_s", time.time())
        persist(receipt_path, receipt)
    return receipt


def _failed_serve_outcome(receipt, app_id, controller):
    """Status and cleanup from what the failed serve's stop actually observed.

    ``status`` is "stopped" only when no app was deployed or its stop is verified;
    ``cleanup`` also needs the controller's verified stop, or is "nothing_deployed".
    """
    lifecycle = receipt["lifecycle"]
    app_stop = lifecycle.get("cleanup", {})
    controller_stop = lifecycle.get("expiry_cleanup", {})
    app_stopped = app_id is None or app_stop.get("cleanup") == "verified_stopped"
    receipt["status"] = "stopped" if app_stopped else "unverified"
    if app_id is None and not controller:
        receipt["cleanup"] = "nothing_deployed"
    elif app_stopped and (not controller or controller_stop.get("cleanup") == "verified_stopped"):
        receipt["cleanup"] = "verified_stopped"
    else:
        receipt["cleanup"] = "unverified"
        receipt["recovery_command"] = ((app_stop if not app_stopped else controller_stop).get("recovery_command")
                                       or "sh scripts/modal_operator.sh apps")


async def _stop_failed_serve(plan, receipt, receipt_path, app, controller, sdk):
    """Stop what a failed serve started: the app by its ID, then its controller.

    After an unverified app stop the controller is kept only when an input was
    spawned for this app, since that input still stops the app at its deadline.
    Without one the app is recorded as left deployed and unarmed, and the idle
    controller is stopped.
    """
    lifecycle = receipt["lifecycle"]
    lifecycle["admission_open"] = False
    app_id = app.app_id if app is not None and app.app_id else None
    try:
        if app_id is not None:
            receipt["app_id"] = app_id
            lifecycle["cleanup"] = await serve.stop_experiment(plan, app_id, sdk=sdk)
            if lifecycle["cleanup"]["cleanup"] != "verified_stopped":
                if "expiry_call_id" in lifecycle:
                    return
                lifecycle["unarmed_app_left_deployed"] = True
        if controller:
            lifecycle["expiry_cleanup"] = await serve.stop_expiry_controller(plan, sdk=sdk)
    except (Exception, asyncio.CancelledError, KeyboardInterrupt) as exc:
        record_error(receipt, receipt_path, exc, "cleanup")
    finally:
        _failed_serve_outcome(receipt, app_id, controller)


async def serve_app(manifest: Path, stage_limits: Path, receipt_path: Path, *, sdk=None) -> dict:
    """Deploy, arm the expiry at the stage lifetime, record the endpoint and exit serving."""
    serve.require_operator_python()
    plan, limits, _cost, receipt = stage_inputs(manifest, stage_limits)
    require_serving_limits(limits)
    reserve(receipt_path)
    receipt.update(mode="serve", status="reserved", lifetime_seconds=limits.max_seconds, lifecycle={},
                   endpoint_env=ENDPOINT_ENV, credential_env=plan.target.credential_env,
                   approval_gate="study_plan_record_checked_by_runner")
    persist(receipt_path, receipt)
    app, controller = None, False
    try:
        with diagnostic_stage(receipt, receipt_path, "cache_inventory"):
            sdk = serve._sdk(sdk)
            receipt["weight_cache"] = await asyncio.wait_for(verify_weight_cache(plan, sdk), 60)
        with diagnostic_stage(receipt, receipt_path, "controller_deploy"):
            expiry = serve.create_expiry_app(sdk=sdk, manifest=plan.serving_manifest,
                                             lifetime_seconds=limits.max_seconds)
            controller = True
            await asyncio.wait_for(expiry.deploy.aio(environment_name=plan.target.environment), 600)
        with diagnostic_stage(receipt, receipt_path, "model_deploy"):
            app = serve.create_app(sdk=sdk, manifest=plan.serving_manifest)
            receipt["gpu_started_at_unix_s"] = time.time()
            persist(receipt_path, receipt)
            await asyncio.wait_for(app.deploy.aio(environment_name=plan.target.environment), 600)
            receipt["app_id"] = app.app_id
            persist(receipt_path, receipt)
        with diagnostic_stage(receipt, receipt_path, "expiry_arming"):
            # The stage limit bounds this app's lifetime, never the manifest's GPU phase.
            await serve.arm_expiry(plan, app.app_id, gpu_started_at_unix_s=receipt["gpu_started_at_unix_s"],
                                   lease=receipt["lifecycle"], lifetime_seconds=limits.max_seconds, sdk=sdk)
            receipt["deadline_unix_s"] = receipt["lifecycle"]["deadline_unix_s"]
            persist(receipt_path, receipt)
        with diagnostic_stage(receipt, receipt_path, "endpoint_check"):
            service = sdk.Cls.from_name(plan.target.app, "Serving", environment_name=plan.target.environment)()
            endpoint = await asyncio.wait_for(service.web.get_web_url.aio(), 5)
            if not isinstance(endpoint, str) or not REGISTERED_URL.fullmatch(endpoint):
                raise ValueError("registered_endpoint_invalid")
            if plan.target.endpoint is not None and endpoint != plan.target.endpoint.removesuffix("/v1"):
                raise ValueError("deployed_endpoint_mismatch")
            receipt.update(endpoint_url=endpoint + "/v1", chat_completions_url=endpoint + "/v1/chat/completions")
        receipt["status"] = "serving"
    except (Exception, asyncio.CancelledError, KeyboardInterrupt) as exc:
        receipt.setdefault("error", type(exc).__name__)
        await _stop_failed_serve(plan, receipt, receipt_path, app, controller, sdk)
    finally:
        persist(receipt_path, receipt)
    return receipt


async def _stopped_by_controller(plan, app_id, deadline_unix_s, sdk) -> bool:
    """This lifecycle's controller record that it verified the app stopped at the deadline.

    The record must carry this receipt's deadline, so a record from another lifecycle
    of the same app ID never counts.
    """
    if deadline_unix_s is None:
        return False
    try:
        store = serve._expiry_store(plan, sdk)
        record = await asyncio.wait_for(store.get.aio((app_id, "expiry_cleanup")), serve.CONTROL_TIMEOUT_SECONDS)
    except Exception:
        return False
    return (isinstance(record, dict) and record.get("app_id") == app_id
            and record.get("deadline_unix_s") == deadline_unix_s and record.get("cleanup") == "verified_stopped")


async def stop_serve(manifest: Path, receipt_path: Path, *, sdk=None) -> dict:
    """Stop the served app by ID, then its controller; the serve receipt records the outcome.

    The controller is stopped only once the app is verified stopped, by this call or
    by the controller itself at the deadline; otherwise it stays armed.
    """
    serve.require_operator_python()
    plan = load_plan(manifest)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict) or receipt.get("mode") != "serve" or receipt.get("manifest") != plan.serving_manifest:
        raise ValueError("Receipt is not a serve receipt for this manifest")
    app_id = receipt.get("app_id")
    stop = {"requested_at_unix_s": time.time(), "cleanup": "unverified"}
    receipt["stop"] = stop
    try:
        sdk = serve._sdk(sdk)
        app_stopped = app_id is None
        if app_id is not None:
            stop["app"] = await serve.stop_experiment(plan, app_id, sdk=sdk)
            app_stopped = stop["app"]["cleanup"] == "verified_stopped"
            # Only when the name resolves to no deployment: a newer lifecycle's app and
            # controller under the same names must never be disarmed by an old receipt.
            if (not app_stopped and stop["app"].get("error") == "NotFoundError"
                    and await _stopped_by_controller(plan, app_id, receipt.get("deadline_unix_s"), sdk)):
                stop["app_stopped_by"] = "expiry_controller"
                app_stopped = True
        if app_stopped:
            stop["expiry"] = await serve.stop_expiry_controller(plan, sdk=sdk)
            if stop["expiry"]["cleanup"] == "verified_stopped":
                stop["cleanup"] = "verified_stopped"
    except (Exception, asyncio.CancelledError, KeyboardInterrupt) as exc:
        stop["error"] = type(exc).__name__
    finally:
        stop["finished_at_unix_s"] = time.time()
        receipt["cleanup"] = stop["cleanup"]
        if stop["cleanup"] == "verified_stopped":
            receipt["status"] = "stopped"
        persist(receipt_path, receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("stage", "serve", "stop-serve"))
    parser.add_argument("--manifest", required=True, type=Path, help="reviewed serving manifest")
    parser.add_argument("--stage-limits", type=Path, help="stage limits from the frozen study plan (stage, serve)")
    parser.add_argument("--receipt", required=True, type=Path,
                        help="new private receipt (stage, serve), or the serve receipt (stop-serve)")
    args = parser.parse_args(argv)
    if (args.stage_limits is None) != (args.mode == "stop-serve"):
        parser.error("stage and serve require --stage-limits; stop-serve takes none")
    try:
        if args.mode == "stage":
            result = asyncio.run(stage(args.manifest, args.stage_limits, args.receipt))
        elif args.mode == "serve":
            result = asyncio.run(serve_app(args.manifest, args.stage_limits, args.receipt))
        else:
            result = asyncio.run(stop_serve(args.manifest, args.receipt))
    except (ValueError, TypeError, KeyError, OSError) as exc:
        # Refusals name the failed check; file errors may name private paths.
        reason = "cannot read an input or create the receipt" if isinstance(exc, OSError) else str(exc)
        print(f"Modal {args.mode} refused: {diagnostic_text(reason, 300)}", file=sys.stderr)
        return 2
    summary = {key: result.get(key) for key in (
        "mode", "status", "error", "error_stage", "volume", "plan_sha256", "app_id", "endpoint_url",
        "deadline_unix_s", "cleanup", "recovery_command")}
    if "lifecycle" in result:
        summary["unarmed_app_left_deployed"] = result["lifecycle"].get("unarmed_app_left_deployed", False)
    print(json.dumps(summary, indent=2))
    if args.mode == "stop-serve":
        return 0 if result["cleanup"] == "verified_stopped" else 1
    return 0 if result["status"] == {"stage": "staged", "serve": "serving"}[args.mode] else 1


if __name__ == "__main__":
    raise SystemExit(main())
