"""Optional, explicitly constructed Modal Web Function; importing is offline.

Only the separately approved operator runner may deploy the returned App. Weights
must already be staged in the manifest's retained volume; serving cannot download
them. ``create_staging_app`` stages them in a separate CPU-only run.
SDK contract: Modal 1.6.3 (_partial_function.py, app.py, image.py, dict.py).
https://modal.com/docs/guide/webhook-proxy-auth documents the existing client's
Bearer <proxy-token-id>.<proxy-token-secret> credential; never inject it here.
vLLM flags: https://github.com/vllm-project/vllm/blob/v0.17.0/vllm/engine/arg_utils.py
"""

import asyncio
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath
from urllib.error import URLError
from urllib.request import urlopen

from deploy.modal.manifest import PILOT_MANIFEST, Plan, load_plan, upload_allowlist

ROOT = Path(__file__).resolve().parents[2]
CACHE = "/cache/huggingface"
CONTROL_TIMEOUT_SECONDS = 5
ARM_TIMEOUT_SECONDS = 30
#: Staging image client, at the versions deploy/modal/uv.lock pins.
STAGING_PACKAGES = ("huggingface_hub==0.34.4", "hf-xet==1.8.0")
STAGING_STARTUP_SECONDS = 60
#: Non-weight snapshot files staged when the revision has them; weights come from the index.
STAGED_FILES = ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                "tokenizer.model", "special_tokens_map.json", "added_tokens.json", "chat_template.json",
                "chat_template.jinja", "preprocessor_config.json", "processor_config.json", "LICENSE", "NOTICE")
WEIGHT_INDEX = "model.safetensors.index.json"
#: Required in addition when config.json declares a vision_config (Gemma 3 4B does).
VISION_PROCESSOR_FILES = ("preprocessor_config.json", "processor_config.json")


def require_operator_python():
    """Serialized functions must match both pinned remote images' Python minor."""
    if sys.version_info[:2] != (3, 12):
        raise ValueError(
            "Modal execution requires Python 3.12 to match the serving and expiry images; "
            "run uv sync --project deploy/modal --locked, then use uv run --project deploy/modal --no-sync"
        )


def _sdk(sdk):
    if sdk is None:
        import modal as sdk

        if sdk.__version__ != "1.6.3":
            raise ValueError("Lifecycle requires Modal 1.6.3")
    return sdk


def _expiry_store(plan, sdk=None, *, create_if_missing=False):
    return _sdk(sdk).Dict.from_name(
        plan.target.app + "-provenance", environment_name=plan.target.environment,
        create_if_missing=create_if_missing,
    )


def _app_identity(app_id):
    if not isinstance(app_id, str) or not re.fullmatch(r"ap-[A-Za-z0-9]{22}", app_id):
        raise ValueError("Invalid experiment app identity")


def running_app_id() -> str:
    """Observed container identity, without capturing the serialized class's App.

    Modal 1.6.3 has no public running-App getter. Its private _init_container
    copies running_app.app_id before setting _App._container_app; this pinned
    seam must be rechecked before an SDK upgrade. Never look up a reusable name.
    """
    import modal

    if modal.__version__ != "1.6.3":
        raise RuntimeError("Container identity requires Modal 1.6.3")
    from modal.app import _App

    app = _App._get_container_app()
    if app is None:
        raise RuntimeError("Missing running container app identity")
    app_id = app.app_id
    _app_identity(app_id)
    return app_id


async def stop_experiment(plan: Plan, app_id: str, *, sdk=None, app_name: str | None = None) -> dict:
    """Two-minute stop/verification budget, no volume deletion or GPU invocation.

    Modal 1.6.3 public App.lookup returns a handle bound to an app ID; stop uses
    that ID and info(refresh=True) fetches lifecycle.state (modal.types.AppState).
    A replacement deployment with a different ID is never stopped by name.
    """
    _app_identity(app_id)
    sdk = _sdk(sdk)
    result = {
        "category": "control", "app_id": app_id, "cleanup": "unverified", "control_calls": 0,
        "retained_cache": plan.volume_name, "observed_usd": None,
        "recovery_command": f"sh scripts/modal_operator.sh stop {app_id}",
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
                app_name or plan.target.app, environment_name=plan.target.environment, create_if_missing=False
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


async def stop_expiry_controller(plan: Plan, *, sdk=None) -> dict:
    """Stop the armed CPU controller once its experiment app is verified stopped.

    Otherwise it idles, billed, until its deadline. Looked up by its own name in
    the experiment environment and stopped by the resolved ID with verification.
    """
    sdk = _sdk(sdk)
    name = plan.target.app + "-expiry"
    try:
        app = await asyncio.wait_for(sdk.App.lookup.aio(
            name, environment_name=plan.target.environment, create_if_missing=False
        ), CONTROL_TIMEOUT_SECONDS)
    except Exception as exc:
        return {"category": "control", "app_name": name, "cleanup": "unverified", "error": type(exc).__name__}
    result = await stop_experiment(plan, app.app_id, sdk=sdk, app_name=name)
    result["app_name"] = name
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
        "estimate_source": plan.serving_manifest + " cost; 2026-11-06",
        "cost_caveat": "Resource request estimate; burst use and other charges excluded; no exact dollar ceiling",
    })
    await asyncio.wait_for(store.put.aio((app_id, "expiry_cleanup"), result), CONTROL_TIMEOUT_SECONDS)
    return result


def app_lifetime_seconds(plan: Plan, lifetime_seconds: int | None = None) -> int:
    """Hard app lifetime: the manifest's GPU phase by default, or a shorter stage limit."""
    if lifetime_seconds is None:
        return plan.limits.gpu_phase_seconds
    if type(lifetime_seconds) is not int or not 0 < lifetime_seconds <= plan.limits.gpu_phase_seconds:
        raise ValueError("App lifetime must be whole seconds within the manifest's GPU phase")
    return lifetime_seconds


async def arm_expiry(plan: Plan, app_id: str, *, gpu_started_at_unix_s: float, lease: dict,
                     lifetime_seconds: int | None = None, sdk=None) -> dict:
    """Arm the remote controller at ``gpu_started_at_unix_s`` plus the app lifetime.

    The lifetime is the manifest's GPU phase unless the caller passes a stage limit.
    Opens ``lease["admission_open"]`` only on the controller's acknowledgment; stops
    nothing itself, so a caller that leaves the app running relies on the controller.
    """
    _app_identity(app_id)
    sdk = _sdk(sdk)
    lease.update({
        "app_id": app_id, "admission_open": False,
        "arming": {"category": "control", "control_calls": 0, "estimated_usd": None, "observed_usd": None},
    })
    deadline = gpu_started_at_unix_s + app_lifetime_seconds(plan, lifetime_seconds)
    if not math.isfinite(deadline) or not gpu_started_at_unix_s <= time.time() < deadline:
        raise ValueError("GPU phase has no remaining time")
    arm_id = uuid.uuid4().hex
    function = sdk.Function.from_name(
        plan.target.app + "-expiry", "expire", environment_name=plan.target.environment
    )
    try:
        async with asyncio.timeout(ARM_TIMEOUT_SECONDS):
            # Explicit creation precedes controller spawn: its first write
            # must find the store, even before a serving container exists.
            store = _expiry_store(plan, sdk, create_if_missing=True)
            lease["arming"]["control_calls"] += 1
            await asyncio.wait_for(store.hydrate.aio(), CONTROL_TIMEOUT_SECONDS)
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
    return lease


@asynccontextmanager
async def experiment_lifecycle(plan: Plan, app_id: str, *, gpu_started_at_unix_s: float, receipt=None, sdk=None,
                               lifetime_seconds: int | None = None):
    """Future runner must enter before inference and persist the mutable receipt.

    Capture gpu_started_at_unix_s BEFORE GPU deployment/startup, never at readiness.
    Acknowledgment proves the remote CPU input is executing, rather than merely
    queued. Pass receipt={} to retain cleanup even when __aenter__ fails. Exit
    closes admission before bounded cleanup on every Python exit. Without
    ``lifetime_seconds`` the deadline is the manifest's full GPU phase.
    """
    _app_identity(app_id)
    sdk = _sdk(sdk)
    lease = receipt if receipt is not None else {}
    try:
        await arm_expiry(plan, app_id, gpu_started_at_unix_s=gpu_started_at_unix_s, lease=lease,
                         lifetime_seconds=lifetime_seconds, sdk=sdk)
        yield lease
    finally:
        lease["admission_open"] = False
        lease["cleanup"] = await stop_experiment(plan, app_id, sdk=sdk)
        if lease["cleanup"]["cleanup"] == "verified_stopped":
            lease["expiry_cleanup"] = await stop_expiry_controller(plan, sdk=sdk)


def create_expiry_app(*, sdk=None, root: Path = ROOT, manifest: str = PILOT_MANIFEST,
                      lifetime_seconds: int | None = None):
    """Construct only: operator must separately deploy this CPU app before GPU startup.

    ``manifest`` selects a reviewed serving manifest. The app lifetime bounds the
    controller's own timeout (lifetime plus 150 s for the stop and verification):
    the manifest's GPU phase by default, or the stage limit passed as ``lifetime_seconds``.
    """
    require_operator_python()
    sdk = _sdk(sdk)
    allowlist = upload_allowlist(manifest)
    plan = load_plan(root / manifest)
    if plan.serving_manifest != manifest:
        raise ValueError("Serving manifest differs from its reviewed upload allowlist")
    lifetime = app_lifetime_seconds(plan, lifetime_seconds)
    app = sdk.App(plan.target.app + "-expiry")
    image = sdk.Image.debian_slim(python_version="3.12")
    for relative in allowlist:
        path = root / relative
        if not path.is_file() or any(p.is_symlink() or p.is_junction() for p in (path, *path.parents)):
            raise ValueError("Unsafe expiry upload")
        image = image.add_local_file(path, "/root/" + relative)

    @app.function(
        name="expire", image=image, gpu=None, cpu=0.125, memory=128,
        min_containers=0, max_containers=1, buffer_containers=0, scaledown_window=2,
        retries=0, timeout=lifetime + 150, startup_timeout=30, include_source=False,
        serialized=True,
    )
    @sdk.concurrent(max_inputs=1)
    async def expire(app_id, deadline_unix_s, arm_id):
        return await expire_experiment(plan, app_id, deadline_unix_s, arm_id)

    return app


def _dereference_snapshot(directory: Path, hub_root: Path) -> None:
    """Replace each pointer symlink of a snapshot with its target's bytes.

    huggingface_hub's ``cache_dir`` layout links ``snapshots/<revision>/<file>``
    into ``blobs/``. The serve-mode inventory reads the volume through the SDK,
    which refuses a symlink, and the serving container needs no link either, so
    the staged snapshot holds the bytes itself. A link that leaves the cache or
    points at anything but a regular file refuses the staging.
    """
    for path in sorted(directory.rglob("*")):
        if not path.is_symlink():
            continue
        target = path.resolve()
        if not target.is_file() or not target.is_relative_to(hub_root):
            raise ValueError(f"Staged snapshot link {path.name} leaves the cache or names no regular file")
        staged = path.with_name(path.name + ".staging")
        shutil.copyfile(target, staged)
        os.replace(staged, path)


def stage_weights(plan: Plan, *, cache: Path = Path(CACHE), hub=None, commit=None) -> dict:
    """Runs in the CPU-only staging container, never on the operator's machine.

    Downloads the pinned revision's indexed weight shards plus its config and
    tokenizer files into the mounted volume's Hugging Face cache, the layout that
    serving reads offline, replaces the cache's pointer symlinks with regular
    files and drops the blobs and lock files they pointed at, then commits the
    volume. On a re-run the pointer files already exist as regular files, which
    the Hub client returns for a commit-hash revision without downloading. The
    Hub client reads HF_TOKEN from the container's secret by name; this code
    never reads it. Returns sizes only.
    """
    started = time.monotonic()
    if hub is None:
        import huggingface_hub as hub
    a = plan.artifacts
    hub_cache = str(cache / "hub")
    index = Path(hub.hf_hub_download(a.model, WEIGHT_INDEX, revision=a.model_revision, cache_dir=hub_cache))
    if index.stat().st_size > 8 * 1024 * 1024:
        raise ValueError("Weight index too large")
    shards = set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values())
    if not shards or any(PurePosixPath(shard).name != shard for shard in shards):
        raise ValueError("Weight index names no plain shard files")
    model = Path(hub.snapshot_download(a.model, revision=a.model_revision, cache_dir=hub_cache,
                                       allow_patterns=sorted(shards | {WEIGHT_INDEX, *STAGED_FILES})))
    tokenizer = model
    if (a.tokenizer, a.tokenizer_revision) != (a.model, a.model_revision):
        tokenizer = Path(hub.snapshot_download(a.tokenizer, revision=a.tokenizer_revision, cache_dir=hub_cache,
                                               allow_patterns=list(STAGED_FILES)))
    if (model.name, tokenizer.name) != (a.model_revision, a.tokenizer_revision):
        raise ValueError("Staged snapshot differs from the pinned revision")
    hub_root = Path(hub_cache).resolve()
    snapshots = {model, tokenizer}
    for directory in snapshots:
        for stale in sorted(directory.rglob("*.staging")):
            stale.unlink()
        _dereference_snapshot(directory, hub_root)
    # The blobs go only once no snapshot of the repository links into them any more: the model and the
    # tokenizer may share a repository at different revisions, and an older snapshot may still link.
    for repo_dir in {directory.parent.parent for directory in snapshots}:
        linked = [path for path in sorted((repo_dir / "snapshots").rglob("*")) if path.is_symlink()]
        if linked:
            raise ValueError(f"Staged repository {repo_dir.name} still links into its blobs: {linked[0].name}")
        blobs = repo_dir / "blobs"
        if blobs.is_dir():
            shutil.rmtree(blobs)
    locks = Path(hub_cache) / ".locks"
    if locks.is_dir():
        shutil.rmtree(locks)
    required = [model / name for name in sorted(shards | {WEIGHT_INDEX, "config.json"})]
    required += [tokenizer / name for name in ("tokenizer.json", "tokenizer_config.json")]
    config = model / "config.json"
    if config.is_file() and "vision_config" in json.loads(config.read_text(encoding="utf-8")):
        # A multimodal checkpoint's processor files: a pattern miss fails here, not at the paid smoke run.
        required += [model / name for name in VISION_PROCESSOR_FILES]
    if any(not path.is_file() or path.stat().st_size <= 0 for path in required):
        raise ValueError("Staged snapshot misses a required file")
    files = sorted({path for directory in {model, tokenizer} for path in directory.rglob("*") if path.is_file()})
    if commit is None:
        commit = _sdk(None).Volume.from_name(plan.volume_name, environment_name=plan.target.environment).commit
    commit()
    return {"volume": plan.volume_name, "model_revision": a.model_revision, "tokenizer_revision": a.tokenizer_revision,
            "files": len(files), "bytes": sum(path.stat().st_size for path in files),
            "duration_seconds": time.monotonic() - started}


def create_staging_app(*, max_seconds: int, sdk=None, root: Path = ROOT, manifest: str = PILOT_MANIFEST):
    """Construct only: a CPU-only weight staging run into the manifest's own volume.

    No GPU; CPU and memory are capped at the manifest's request, the timeout is the
    stage limit ``max_seconds`` and nothing retries, so a run that reaches its timeout
    stops. HF_TOKEN reaches the container as an ephemeral Secret built from the
    operator process's variable by name; no named Modal secret is created. Returns
    the App and its ``stage`` Function, for one ephemeral ``app.run``.
    """
    require_operator_python()
    sdk = _sdk(sdk)
    allowlist = upload_allowlist(manifest)
    plan = load_plan(root / manifest)
    if plan.serving_manifest != manifest:
        raise ValueError("Serving manifest differs from its reviewed upload allowlist")
    if type(max_seconds) is not int or not 0 < max_seconds <= plan.limits.gpu_phase_seconds:
        raise ValueError("Staging timeout must be whole seconds within the manifest's GPU phase")
    app = sdk.App(plan.target.app + "-staging")
    image = sdk.Image.debian_slim(python_version="3.12").pip_install(*STAGING_PACKAGES)
    image = image.env({"HF_HOME": CACHE, "HF_HUB_DISABLE_TELEMETRY": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1",
                       "DO_NOT_TRACK": "1"})
    for relative in allowlist:
        path = root / relative
        if not path.is_file() or any(p.is_symlink() or p.is_junction() for p in (path, *path.parents)):
            raise ValueError("Unsafe staging upload")
        image = image.add_local_file(path, "/root/" + relative)
    weights = sdk.Volume.from_name(plan.volume_name, environment_name=plan.target.environment,
                                   create_if_missing=True)
    cores, mib = plan.resources.cpu_physical_cores, plan.memory_mib

    @app.function(
        name="stage", image=image, gpu=None, cpu=(cores, cores), memory=(mib, mib), volumes={CACHE: weights},
        secrets=[sdk.Secret.from_local_environ(["HF_TOKEN"])],
        min_containers=0, max_containers=1, buffer_containers=0, scaledown_window=2,
        retries=0, timeout=max_seconds, startup_timeout=STAGING_STARTUP_SECONDS, include_source=False,
        serialized=True,
    )
    def stage():
        return stage_weights(plan)

    return app, stage


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


def _weight_format(artifacts) -> list[str]:
    """Quantized checkpoints name their method; a dense bf16 checkpoint names its dtype.

    vLLM 0.17.0 has no "bf16" quantization method and no "none" MoE backend
    (vllm/config/model.py ModelDType, vllm/config/kernel.py MoEBackend), so the
    generic-mapping Gemma manifest passes ``--dtype bfloat16`` and no MoE flag.
    """
    if artifacts.quantization == "bf16":
        options = ["--dtype", "bfloat16"]
    else:
        options = ["--quantization", artifacts.quantization]
    options += ["--attention-backend", artifacts.attention_backend]
    if artifacts.moe_backend != "none":
        options += ["--moe-backend", artifacts.moe_backend]
    return options


#: Bytes of the engine's output kept in a startup failure: enough for the final traceback, never a request.
STARTUP_TAIL_BYTES = 4096


def startup_tail(log) -> str:
    """The last bytes of the engine's output file, read only after a failed startup.

    No request has been sent at that point, so the text can hold engine
    diagnostics only; it reaches the raised error and the control-plane store,
    never a receipt.
    """
    log.flush()
    log.seek(0, os.SEEK_END)
    size = log.tell()
    log.seek(max(0, size - STARTUP_TAIL_BYTES))
    return log.read().decode("utf-8", errors="replace").strip()


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
        *_weight_format(a),
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


def create_app(*, sdk=None, root: Path = ROOT, manifest: str = PILOT_MANIFEST):
    """Construct lazy SDK handles only; caller controls deployment and environment.

    ``manifest`` selects a reviewed serving manifest (pilot by default).
    """
    require_operator_python()
    root = root.absolute()
    allowlist = upload_allowlist(manifest)
    files = [root / relative for relative in allowlist]
    for path in files:
        if (
            not path.is_file()
            or not path.resolve().is_relative_to(root.resolve())
            or any(parent.is_symlink() or parent.is_junction() for parent in (path, *path.parents))
        ):
            raise ValueError("Unsafe serving upload")
    plan = load_plan(root / manifest)
    if plan.serving_manifest != manifest:
        raise ValueError("Serving manifest differs from its reviewed upload allowlist")
    if sdk is None:
        import modal as sdk

        if version("modal") != plan.artifacts.modal_sdk:
            raise ValueError("Serving SDK differs from manifest")
    app = sdk.App(plan.target.app)
    image = sdk.Image.from_registry(
        plan.artifacts.image,
        setup_dockerfile_commands=[
            'RUN /usr/bin/python3 -c "import sys; assert sys.version_info[:2] == (3, 12)" '
            '&& ln -s /usr/bin/python3 /usr/local/bin/python'
        ],
    )
    # The registry image starts the vLLM CLI; Modal must start its own bootstrap.
    image = image.entrypoint([])
    # CPU bootstrap observed 4.12.2; the image's pydantic_core requires Sentinel.
    image = image.pip_install("typing_extensions==4.16.0")
    image = image.env(
        {
            "HF_HOME": str(CACHE),
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "VLLM_NO_USAGE_STATS": "1",
            "DO_NOT_TRACK": "1",
        }
    )
    # Deferred local files must follow all image build operations, including env.
    for path, relative in zip(files, allowlist, strict=True):
        image = image.add_local_file(path, "/root/" + relative)
    weights = sdk.Volume.from_name(plan.volume_name, environment_name=plan.target.environment,
                                   create_if_missing=False)
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
            self.app_id = running_app_id()
            self.observations = _expiry_store(plan)
            require_remote_expiry(plan, self.app_id, self.observations)
            self.record("starting")
            if version("vllm") != plan.artifacts.runtime.split("==")[1]:
                raise RuntimeError("Serving runtime differs from manifest")
            # The engine's output stays in an anonymous file inside the container and is read once, on a
            # failed startup, before any request was sent; after readiness it is never read, since engine
            # errors can contain request contents.
            self.log = tempfile.TemporaryFile(mode="w+b")
            self.process = subprocess.Popen(server_command(plan), stdout=self.log, stderr=subprocess.STDOUT)
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
            detail = startup_tail(self.log)
            exit_code = self.process.poll()
            self.record("failed")
            # Keyed outside the provenance reader's key space (its keys start with the app ID), so the
            # tail never reaches read_observations and the receipts built from it.
            self.observations[("startup_failure", self.app_id, self.instance_id)] = {
                "app_id": self.app_id, "exit_code": exit_code, "detail": detail}
            engine = f"engine exit {exit_code}" if exit_code is not None else "engine still running at the readiness timeout"
            raise RuntimeError(f"Serving startup failed; no inference sent ({engine}): {detail}")

        def record(self, phase):
            self.observations[(self.app_id, self.instance_id, phase)] = snapshot(plan, self.app_id, phase)

        @sdk.method()
        def readiness(self):
            """Authenticated SDK startup observation; enter has readied vLLM, no tokens."""
            return {"app_id": self.app_id, "ready": self.process is not None and self.process.poll() is None}

        @sdk.web_server(8000, startup_timeout=plan.limits.startup_seconds,
                        label=plan.target.app, requires_proxy_auth=True)
        def web(self):
            pass  # Server was started by the enter hook; responses are native vLLM.

        @sdk.exit()
        def close(self):
            try:
                if getattr(self, "app_id", None) is not None:
                    self.record("closing")
            finally:
                if self.process is not None:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                if getattr(self, "log", None) is not None:
                    self.log.close()

    return app
