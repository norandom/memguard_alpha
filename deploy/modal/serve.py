"""Optional, explicitly constructed Modal Web Function; importing is offline.

Only the separately approved operator runner may deploy the returned App. Weights
must already be staged in the retained volume; serving cannot download them.
SDK contract: Modal 1.6.3 (_partial_function.py, app.py, image.py, dict.py).
https://modal.com/docs/guide/webhook-proxy-auth documents the existing client's
Bearer <proxy-token-id>.<proxy-token-secret> credential; never inject it here.
vLLM flags: https://github.com/vllm-project/vllm/blob/v0.17.0/vllm/engine/arg_utils.py
"""

import asyncio
import hashlib
import json
import math
import re
import subprocess
import sys
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from deploy.modal.manifest import UPLOAD_ALLOWLIST, Plan, load_plan

ROOT = Path(__file__).resolve().parents[2]
CACHE = "/cache/huggingface"
CONTROL_TIMEOUT_SECONDS = 5
ARM_TIMEOUT_SECONDS = 30


def _sdk(sdk):
    if sdk is None:
        import modal as sdk

        if version("modal") != "1.6.3":
            raise ValueError("Lifecycle requires Modal 1.6.3")
    return sdk


def _expiry_store(plan, sdk):
    return sdk.Dict.from_name(
        plan.target.app + "-provenance", environment_name=plan.target.environment, create_if_missing=False
    )


def _app_identity(app_id):
    if not isinstance(app_id, str) or not re.fullmatch(r"ap-[A-Za-z0-9]{22}", app_id):
        raise ValueError("Invalid experiment app identity")


async def stop_experiment(plan: Plan, app_id: str, *, sdk=None) -> dict:
    """Two-minute stop/verification budget, no volume deletion or GPU invocation.

    Modal 1.6.3 public App.lookup returns a handle bound to an app ID; stop uses
    that ID and info(refresh=True) fetches lifecycle.state (modal.types.AppState).
    A replacement deployment with a different ID is never stopped by name.
    """
    _app_identity(app_id)
    sdk = _sdk(sdk)
    result = {
        "category": "control", "app_id": app_id, "cleanup": "unverified", "control_calls": 0,
        "retained_cache": "recall-guard-gpt-oss-20b-weights", "observed_usd": None,
        "recovery_command": f"uv run --env-file .env --project deploy/modal --no-sync python -m modal app stop {app_id} --yes",
        "verification_budget_seconds": plan.limits.cleanup_seconds,
        "estimated_usd": None,
    }
    started = time.monotonic()

    async def control(awaitable):
        result["control_calls"] += 1
        return await asyncio.wait_for(awaitable, CONTROL_TIMEOUT_SECONDS)

    try:
        async with asyncio.timeout(plan.limits.cleanup_seconds):
            app = await control(sdk.App.lookup.aio(
                plan.target.app, environment_name=plan.target.environment, create_if_missing=False
            ))
            if app.app_id != app_id:
                raise ValueError("Experiment app identity mismatch")
            await control(app.stop.aio())
            for _ in range(24):
                info = await control(app.info.aio(refresh=True))
                if info.lifecycle.state.value == "stopped":
                    result["cleanup"] = "verified_stopped"
                    break
                await asyncio.sleep(5)
    except Exception as exc:
        result["error"] = type(exc).__name__  # Provider messages may contain credentials.
    result["elapsed_seconds"] = time.monotonic() - started
    return result


async def expire_experiment(plan: Plan, app_id: str, deadline_unix_s: float, arm_id: str, *, sdk=None) -> dict:
    """Runs in the separate deployed CPU app, independently of the local client.

    Deployed Apps persist until stopped; spawn submits an ASYNC invocation whose
    FunctionCall ID is recoverable after client exit (Modal docs /guide/apps and
    /guide/function-invocation-methods; verified against installed SDK 1.6.3).
    Queue delays/control failure can prevent timely stop: this is no spend guarantee.
    """
    _app_identity(app_id)
    remaining = deadline_unix_s - time.time()
    if not math.isfinite(remaining) or remaining > plan.limits.gpu_phase_seconds:
        raise ValueError("Invalid expiry deadline")
    sdk = _sdk(sdk)
    store = _expiry_store(plan, sdk)
    started = time.monotonic()
    if remaining > 0:
        owner = await asyncio.wait_for(sdk.App.lookup.aio(
            plan.target.app, environment_name=plan.target.environment, create_if_missing=False
        ), CONTROL_TIMEOUT_SECONDS)
        if owner.app_id != app_id:
            raise ValueError("Experiment app identity mismatch")
        armed = {"app_id": app_id, "arm_id": arm_id, "deadline_unix_s": deadline_unix_s}
        created = await asyncio.wait_for(
            store.put.aio((app_id, "expiry"), armed, skip_if_exists=True), CONTROL_TIMEOUT_SECONDS
        )
        if not created:
            raise RuntimeError("Expiry already armed; never extend its deadline")
        await asyncio.sleep(max(0, deadline_unix_s - time.time()))
    result = await stop_experiment(plan, app_id, sdk=sdk)
    result.update({
        "category": "expiry", "deadline_unix_s": deadline_unix_s,
        "controller_cpu_cores": 0.125, "controller_memory_mib": 128,
        # Rates from the frozen manifest, checked 2026-11-06; excludes burst usage,
        # image builds/storage/egress. Observed cost always remains unknown here.
        "estimated_usd": (time.monotonic() - started) * (0.125 * 0.0000131 + 0.125 * 0.00000222),
        "estimate_source": "configs/modal/gpt-oss-20b-pilot.json cost; 2026-11-06",
        "cost_caveat": "Resource request estimate; burst use and other charges excluded; no exact dollar ceiling",
    })
    await asyncio.wait_for(store.put.aio((app_id, "expiry_cleanup"), result), CONTROL_TIMEOUT_SECONDS)
    return result


@asynccontextmanager
async def experiment_lifecycle(plan: Plan, app_id: str, *, gpu_started_at_unix_s: float, receipt=None, sdk=None):
    """Future runner must enter before inference and persist the mutable receipt.

    Capture gpu_started_at_unix_s BEFORE GPU deployment/startup, never at readiness.
    Acknowledgment proves the remote CPU input is executing, rather than merely
    queued. Pass receipt={} to retain cleanup even when __aenter__ fails. Exit
    closes admission before bounded cleanup on every Python exit.
    """
    _app_identity(app_id)
    sdk = _sdk(sdk)
    lease = receipt if receipt is not None else {}
    lease.update({
        "app_id": app_id, "admission_open": False,
        "arming": {"category": "control", "control_calls": 0, "estimated_usd": None, "observed_usd": None},
    })
    try:
        deadline = gpu_started_at_unix_s + plan.limits.gpu_phase_seconds
        if not math.isfinite(deadline) or not gpu_started_at_unix_s <= time.time() < deadline:
            raise ValueError("GPU phase has no remaining time")
        arm_id = uuid.uuid4().hex
        function = sdk.Function.from_name(
            plan.target.app + "-expiry", "expire", environment_name=plan.target.environment
        )
        try:
            async with asyncio.timeout(ARM_TIMEOUT_SECONDS):
                lease["arming"]["control_calls"] += 1
                call = await asyncio.wait_for(function.spawn.aio(app_id, deadline, arm_id), CONTROL_TIMEOUT_SECONDS)
                lease.update(expiry_call_id=call.object_id, deadline_unix_s=deadline)
                store = _expiry_store(plan, sdk)
                while time.time() < deadline:
                    lease["arming"]["control_calls"] += 1
                    armed = await asyncio.wait_for(store.get.aio((app_id, "expiry")), CONTROL_TIMEOUT_SECONDS)
                    if armed == {"app_id": app_id, "arm_id": arm_id, "deadline_unix_s": deadline}:
                        lease["admission_open"] = True
                        break
                    await asyncio.sleep(0.25)
                if not lease["admission_open"]:
                    raise RuntimeError("Remote expiry not armed")
        except TimeoutError as exc:
            raise RuntimeError("Remote expiry acknowledgment timed out; admission closed") from exc
        yield lease
    finally:
        lease["admission_open"] = False
        lease["cleanup"] = await stop_experiment(plan, app_id, sdk=sdk)


def create_expiry_app(*, sdk=None, root: Path = ROOT):
    """Construct only: operator must separately deploy this CPU app before GPU startup."""
    sdk = _sdk(sdk)
    plan = load_plan(root / "configs/modal/gpt-oss-20b-pilot.json")
    app = sdk.App(plan.target.app + "-expiry")
    image = sdk.Image.debian_slim(python_version="3.12")
    for relative in UPLOAD_ALLOWLIST:
        path = root / relative
        if not path.is_file() or any(p.is_symlink() or p.is_junction() for p in (path, *path.parents)):
            raise ValueError("Unsafe expiry upload")
        image = image.add_local_file(path, "/root/" + relative)

    @app.function(
        name="expire", image=image, gpu=None, cpu=0.125, memory=128,
        min_containers=0, max_containers=1, buffer_containers=0, scaledown_window=2,
        retries=0, timeout=1350, startup_timeout=30, include_source=False, serialized=True,
    )
    @sdk.concurrent(max_inputs=1)
    async def expire(app_id, deadline_unix_s, arm_id):
        return await expire_experiment(plan, app_id, deadline_unix_s, arm_id)

    return app


def require_remote_expiry(plan: Plan, app_id: str, store):
    """Fail startup closed before vLLM launches unless a remote controller is armed."""
    record = asyncio.run(asyncio.wait_for(store.get.aio((app_id, "expiry")), CONTROL_TIMEOUT_SECONDS))
    if (
        not isinstance(record, dict)
        or record.get("app_id") != app_id
        or not isinstance(record.get("arm_id"), str)
        or not isinstance(record.get("deadline_unix_s"), (float, int))
        or not time.time() < record["deadline_unix_s"] <= time.time() + plan.limits.gpu_phase_seconds
    ):
        raise RuntimeError("Remote expiry is not armed; serving admission closed")


def server_command(plan: Plan) -> list[str]:
    """Pass through native completion evidence; never normalize a top-k subset."""
    a, r = plan.artifacts, plan.resources
    return [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--model",
        a.model,
        "--revision",
        a.model_revision,
        "--tokenizer",
        a.tokenizer,
        "--tokenizer-revision",
        a.tokenizer_revision,
        "--served-model-name",
        a.model,
        "--quantization",
        a.quantization,
        "--attention-backend",
        a.attention_backend,
        "--moe-backend",
        a.moe_backend,
        "--tensor-parallel-size",
        str(r.tensor_parallel_size),
        "--max-model-len",
        str(r.context_tokens),
        "--max-num-seqs",
        str(r.max_num_seqs),
        "--max-logprobs",
        str(plan.limits.top_logprobs),
        "--logprobs-mode",
        "raw_logprobs",
        "--generation-config",
        "vllm",
        "--no-enable-log-requests",
        "--disable-uvicorn-access-log",
    ]


def observe_gpu() -> list[dict] | None:
    """Observed NVIDIA identity, separately from requested resource allocation."""
    fields = ("name", "uuid", "driver_version", "memory.total")
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        rows = [line.split(",") for line in result.stdout.strip().splitlines()]
        if not rows or any(len(row) != len(fields) for row in rows):
            return None
        names = ("name", "uuid", "driver_version", "memory_total_mib")
        return [dict(zip(names, (value.strip() for value in row), strict=True)) for row in rows]
    except (OSError, subprocess.SubprocessError):
        return None


def snapshot(plan: Plan, app_id: str, phase: str, *, cache: Path = Path(CACHE)) -> dict:
    """No prompts/secrets; unavailable observations remain null, never configured facts."""
    a = plan.artifacts
    model = cache / "hub" / ("models--" + a.model.replace("/", "--")) / "snapshots" / a.model_revision
    tokenizer = cache / "hub" / ("models--" + a.tokenizer.replace("/", "--")) / "snapshots" / a.tokenizer_revision
    config = (
        json.loads((model / "config.json").read_text(encoding="utf-8")) if (model / "config.json").is_file() else {}
    )
    try:
        runtime = "vllm==" + version("vllm")
    except PackageNotFoundError:
        runtime = None
    hashes = {}
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        path = tokenizer / name
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    return {
        "app_id": app_id,
        "app_name": plan.target.app,
        "environment": plan.target.environment,
        "phase": phase,
        "observed_at": datetime.now(UTC).isoformat(),
        "configured_artifacts": asdict(a),
        "configured_resources": asdict(plan.resources),
        "resolved_model_revision": model.name if (model / "config.json").is_file() else None,
        "resolved_tokenizer_revision": tokenizer.name if (tokenizer / "tokenizer.json").is_file() else None,
        "quantization_config": config.get("quantization_config"),
        "tokenizer_sha256": hashes,
        "runtime": runtime,
        "gpu": observe_gpu(),
        "server_command": server_command(plan),
        "score_semantics": a.score_semantics,
        "distribution": "partial_top_k",
    }


def read_observations(plan: Plan, app_id: str, *, sdk=None) -> list[dict]:
    """Authenticated SDK Dict reads only: never invokes or wakes the GPU function.

    App ID must come from the deployed App, not its reusable name. Missing phases
    mean unknown, including when a container is killed before its exit hook runs.
    """
    if not app_id.startswith("ap-"):
        raise ValueError("Invalid app identity")
    if sdk is None:
        import modal as sdk
    store = sdk.Dict.from_name(
        plan.target.app + "-provenance",
        environment_name=plan.target.environment,
        create_if_missing=False,
    )
    results = []
    for key, value in store.items():
        if isinstance(key, tuple) and len(key) == 3 and key[0] == app_id:
            if (value.get("app_id"), value.get("app_name"), value.get("environment")) != (
                app_id,
                plan.target.app,
                plan.target.environment,
            ):
                raise ValueError("Provenance deployment identity mismatch")
            results.append(value)
    return results


def create_app(*, sdk=None, root: Path = ROOT):
    """Construct lazy SDK handles only; caller controls deployment and environment."""
    root = root.absolute()
    files = [root / relative for relative in UPLOAD_ALLOWLIST]
    for path in files:
        if (
            not path.is_file()
            or not path.resolve().is_relative_to(root.resolve())
            or any(parent.is_symlink() or parent.is_junction() for parent in (path, *path.parents))
        ):
            raise ValueError("Unsafe serving upload")
    plan = load_plan(root / "configs/modal/gpt-oss-20b-pilot.json")
    if sdk is None:
        import modal as sdk

        if version("modal") != plan.artifacts.modal_sdk:
            raise ValueError("Serving SDK differs from manifest")
    app = sdk.App(plan.target.app)
    image = sdk.Image.from_registry(plan.artifacts.image)
    for path, relative in zip(files, UPLOAD_ALLOWLIST, strict=True):
        image = image.add_local_file(path, "/root/" + relative)
    image = image.env(
        {
            "HF_HOME": str(CACHE),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "VLLM_NO_USAGE_STATS": "1",
            "DO_NOT_TRACK": "1",
        }
    )
    weights = sdk.Volume.from_name("recall-guard-gpt-oss-20b-weights", environment_name=plan.target.environment,
                                   create_if_missing=False)
    observations = sdk.Dict.from_name(
        plan.target.app + "-provenance",
        environment_name=plan.target.environment,
        create_if_missing=True,
    )

    @app.cls(
        image=image,
        gpu=plan.resources.gpu + ":1",
        cpu=plan.resources.cpu_physical_cores,
        memory=plan.memory_mib,
        volumes={str(CACHE): weights},
        min_containers=0,
        max_containers=1,
        buffer_containers=0,
        scaledown_window=60,
        retries=0,
        timeout=plan.limits.request_seconds,
        startup_timeout=plan.limits.startup_seconds,
        include_source=False,
        serialized=True,
    )
    @sdk.concurrent(max_inputs=1)
    class Serving:
        @sdk.enter()
        def start(self):
            self.instance_id = uuid.uuid4().hex
            self.process = None
            require_remote_expiry(plan, app.app_id, observations)
            self.record("starting")
            if version("vllm") != plan.artifacts.runtime.split("==")[1]:
                raise RuntimeError("Serving runtime differs from manifest")
            # Suppress engine diagnostics too: errors can contain request contents.
            self.process = subprocess.Popen(server_command(plan), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            deadline = time.monotonic() + plan.limits.startup_seconds
            while time.monotonic() < deadline and self.process.poll() is None:
                try:
                    with urlopen("http://127.0.0.1:8000/health", timeout=2) as response:
                        if response.status == 200:
                            self.record("ready")
                            return
                except (URLError, TimeoutError):
                    pass
                time.sleep(1)
            raise RuntimeError("Serving startup failed; no inference sent")

        def record(self, phase):
            if app.app_id is None:
                raise RuntimeError("Missing deployment identity")
            observations[(app.app_id, self.instance_id, phase)] = snapshot(plan, app.app_id, phase)

        @sdk.method()
        def readiness(self):
            """Authenticated SDK startup observation; enter has readied vLLM, no tokens."""
            return {"app_id": app.app_id, "ready": self.process is not None and self.process.poll() is None}

        @sdk.web_server(8000, startup_timeout=plan.limits.startup_seconds,
                        label=plan.target.app, requires_proxy_auth=True)
        def web(self):
            pass  # Server was started by the enter hook; responses are native vLLM.

        @sdk.exit()
        def close(self):
            try:
                self.record("closing")
            finally:
                if self.process is not None:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.process.kill()

    return app
