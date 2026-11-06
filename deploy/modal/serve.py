"""Optional, explicitly constructed Modal Web Function; importing is offline.

Only the separately approved operator runner may deploy the returned App. Weights
must already be staged in the retained volume; serving cannot download them.
SDK contract: Modal 1.6.3 (_partial_function.py, app.py, image.py, dict.py).
https://modal.com/docs/guide/webhook-proxy-auth documents the existing client's
Bearer <proxy-token-id>.<proxy-token-secret> credential; never inject it here.
vLLM flags: https://github.com/vllm-project/vllm/blob/v0.17.0/vllm/engine/arg_utils.py
"""

import hashlib
import json
import subprocess
import sys
import time
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from deploy.modal.manifest import UPLOAD_ALLOWLIST, Plan, load_plan

ROOT = Path(__file__).resolve().parents[2]
CACHE = "/cache/huggingface"


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
    weights = sdk.Volume.from_name("recall-guard-gpt-oss-20b-weights", create_if_missing=False)
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

        @sdk.web_server(8000, startup_timeout=plan.limits.startup_seconds, requires_proxy_auth=True)
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
