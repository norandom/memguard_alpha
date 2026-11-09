"""Offline serving recipe tests; fake SDK calls never allocate compute."""

import json
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from deploy.modal import serve
from deploy.modal.manifest import load_plan

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "configs/modal/gpt-oss-20b-pilot.json"


@pytest.mark.parametrize("factory", [serve.create_app, serve.create_expiry_app])
@pytest.mark.parametrize("provided_sdk", [False, True])
def test_wrong_python_blocks_recipe_before_sdk_import_or_calls(monkeypatch, factory, provided_sdk):
    import builtins

    monkeypatch.setattr(serve, "sys", SimpleNamespace(version_info=(3, 14, 9)))
    original_import = builtins.__import__
    calls = []

    def guarded_import(name, *args, **kwargs):
        if name == "modal":
            calls.append("import")
            raise AssertionError("SDK imported before Python compatibility check")
        return original_import(name, *args, **kwargs)

    def forbidden(*args, **kwargs):
        calls.append("App")
        raise AssertionError("SDK called before Python compatibility check")

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    sdk = SimpleNamespace(App=forbidden) if provided_sdk else None
    with pytest.raises(ValueError, match="Python 3.12"):
        factory(sdk=sdk)
    assert calls == []


class FakeSDK:
    def __init__(self):
        self.calls = []
        self.store = {}
        self.App = lambda name: SimpleNamespace(app_id="ap-test", cls=self.cls)
        self.Image = SimpleNamespace(from_registry=self.image)
        self.Volume = SimpleNamespace(from_name=self.volume)
        self.Dict = SimpleNamespace(from_name=self.dictionary)

    def image(self, *args, **kwargs):
        self.calls.append(("image", args, kwargs))
        return SimpleNamespace(add_local_file=self.add_file, env=self.env, entrypoint=self.entrypoint,
                               pip_install=self.pip_install)

    def entrypoint(self, commands):
        self.calls.append(("entrypoint", (commands,), {}))
        return SimpleNamespace(add_local_file=self.add_file, env=self.env, entrypoint=self.entrypoint,
                               pip_install=self.pip_install)

    def pip_install(self, *packages):
        self.calls.append(("pip_install", packages, {}))
        return SimpleNamespace(add_local_file=self.add_file, env=self.env, entrypoint=self.entrypoint,
                               pip_install=self.pip_install)

    def add_file(self, *args, **kwargs):
        self.calls.append(("upload", args, kwargs))
        return SimpleNamespace(add_local_file=self.add_file, env=self.env, entrypoint=self.entrypoint,
                               pip_install=self.pip_install)

    def env(self, settings):
        self.calls.append(("env", (), settings))
        return SimpleNamespace(add_local_file=self.add_file, env=self.env, entrypoint=self.entrypoint,
                               pip_install=self.pip_install)

    def volume(self, *args, **kwargs):
        self.calls.append(("volume", args, kwargs))
        return "weights"

    def dictionary(self, *args, **kwargs):
        self.calls.append(("dict", args, kwargs))
        return self.store

    def cls(self, **kwargs):
        self.calls.append(("cls", (), kwargs))

        def wrap(cls):
            self.service = cls
            return cls

        return wrap

    def concurrent(self, **kwargs):
        self.calls.append(("concurrent", (), kwargs))
        return lambda cls: cls

    def enter(self):
        return lambda fn: fn

    def method(self):
        return lambda fn: fn

    def exit(self):
        return lambda fn: fn

    def web_server(self, *args, **kwargs):
        self.calls.append(("web_server", args, kwargs))
        return lambda fn: fn


def test_recipe_auth_resources_cache_and_uploads(monkeypatch):
    monkeypatch.setattr(serve, "require_operator_python", lambda: None)
    sdk = FakeSDK()
    serve.create_app(sdk=sdk)
    calls = {name: kwargs for name, _, kwargs in sdk.calls}
    assert calls["web_server"]["requires_proxy_auth"] is True
    assert calls["web_server"]["label"] == load_plan(MANIFEST).target.app
    bootstrap = calls["image"]["setup_dockerfile_commands"]
    assert len(bootstrap) == 1
    assert "sys.version_info[:2] == (3, 12)" in bootstrap[0]
    assert "ln -s /usr/bin/python3 /usr/local/bin/python" in bootstrap[0]
    assert calls["concurrent"] == {"max_inputs": 1}
    config = calls["cls"]
    assert {
        key: config[key]
        for key in (
            "gpu",
            "cpu",
            "memory",
            "min_containers",
            "max_containers",
            "buffer_containers",
            "scaledown_window",
            "include_source",
            "retries",
        )
    } == dict(
        gpu="A100-40GB:1",
        cpu=4,
        memory=32768,
        min_containers=0,
        max_containers=1,
        buffer_containers=0,
        scaledown_window=60,
        include_source=False,
        retries=0,
    )
    assert config["volumes"] == {"/cache/huggingface": "weights"}
    assert calls["env"]["HF_HUB_OFFLINE"] == "1"
    operations = [name for name, _, _ in sdk.calls]
    assert operations.index("env") < operations.index("upload")
    assert [args for name, args, _ in sdk.calls if name == "entrypoint"] == [([],)]
    assert operations.index("entrypoint") < operations.index("upload")
    assert [args for name, args, _ in sdk.calls if name == "pip_install"] == [("typing_extensions==4.16.0",)]
    assert operations.index("entrypoint") < operations.index("pip_install") < operations.index("env")
    uploads = [args for name, args, _ in sdk.calls if name == "upload"]
    assert len(uploads) == 3
    assert all(Path(args[0]).resolve().is_relative_to(ROOT) for args in uploads)
    assert all("models" not in Path(args[0]).parts for args in uploads)


def test_runtime_command_keeps_real_partial_logprobs():
    command = serve.server_command(load_plan(MANIFEST))
    assert command[:3] == [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]
    assert command[command.index("--max-logprobs") + 1] == "5"
    assert command[command.index("--logprobs-mode") + 1] == "raw_logprobs"
    assert command[command.index("--max-num-seqs") + 1] == "1"
    assert "--no-enable-log-requests" in command
    assert "--enable-log-outputs" not in command
    assert "--speculative-config" not in command


GEMMA = ROOT / "configs/modal/gemma-3-4b-measurement.json"
#: The GPT-OSS-20B vLLM command before task 3.4, element for element.
GPT_OSS_COMMAND = [
    "--host", "0.0.0.0", "--port", "8000", "--model", "openai/gpt-oss-20b",
    "--revision", "6cee5e81ee83917806bbde320786a8fb61efebee", "--tokenizer", "openai/gpt-oss-20b",
    "--tokenizer-revision", "6cee5e81ee83917806bbde320786a8fb61efebee", "--served-model-name", "openai/gpt-oss-20b",
    "--quantization", "mxfp4", "--attention-backend", "TRITON_ATTN", "--moe-backend", "marlin",
    "--tensor-parallel-size", "1", "--max-model-len", "4096", "--max-num-seqs", "1", "--max-logprobs", "5",
    "--logprobs-mode", "raw_logprobs", "--generation-config", "vllm", "--no-enable-log-requests",
    "--disable-uvicorn-access-log",
]


def test_gpt_oss_command_is_unchanged():
    for manifest in (MANIFEST, ROOT / "configs/modal/gpt-oss-20b-measurement.json"):
        command = serve.server_command(load_plan(manifest))
        assert command[:3] == [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]
        assert command[3:] == GPT_OSS_COMMAND


def test_gemma_command_serves_dense_bf16_without_quantization_or_moe_flags():
    plan = load_plan(GEMMA)
    command = serve.server_command(plan)
    assert command[command.index("--model") + 1] == command[command.index("--tokenizer") + 1] == "google/gemma-3-4b-it"
    assert command[command.index("--revision") + 1] == plan.artifacts.model_revision
    assert command[command.index("--tokenizer-revision") + 1] == plan.artifacts.tokenizer_revision
    assert command[command.index("--dtype") + 1] == "bfloat16"
    assert command[command.index("--attention-backend") + 1] == "FLASH_ATTN"
    assert "--quantization" not in command and "--moe-backend" not in command
    assert command[command.index("--max-logprobs") + 1] == str(plan.limits.top_logprobs) == "5"
    assert command[command.index("--logprobs-mode") + 1] == "raw_logprobs"
    assert command[command.index("--max-num-seqs") + 1] == "1"
    assert command[command.index("--generation-config") + 1] == "vllm"
    assert "--no-enable-log-requests" in command and "--enable-log-outputs" not in command


def test_gemma_recipe_selects_its_manifest_without_a_harmony_tokenizer(monkeypatch):
    monkeypatch.setattr(serve, "require_operator_python", lambda: None)
    sdk = FakeSDK()
    serve.create_app(sdk=sdk, manifest="configs/modal/gemma-3-4b-measurement.json")
    calls = {name: kwargs for name, _, kwargs in sdk.calls}
    assert calls["cls"]["gpu"] == "A10:1" and calls["cls"]["memory"] == 16384
    assert calls["web_server"]["label"] == load_plan(GEMMA).target.app
    uploads = [args for name, args, _ in sdk.calls if name == "upload"]
    assert [args[1] for args in uploads][-1] == "/root/configs/modal/gemma-3-4b-measurement.json"
    assert all("models" not in Path(args[0]).parts for args in uploads)


def test_control_plane_retrieval_never_calls_gpu_and_checks_identity():
    sdk = FakeSDK()
    plan = load_plan(MANIFEST)
    sdk.store[("ap-test", "container", "ready")] = {
        "app_id": "ap-test",
        "app_name": plan.target.app,
        "environment": plan.target.environment,
        "phase": "ready",
    }
    sdk.store[("ap-other", "container", "ready")] = {"app_id": "ap-other"}
    result = serve.read_observations(plan, "ap-test", sdk=sdk)
    assert len(result) == 1
    assert sdk.calls == [
        (
            "dict",
            (plan.target.app + "-provenance",),
            {
                "environment_name": plan.target.environment,
                "create_if_missing": False,
            },
        )
    ]
    sdk.store[("ap-test", "container", "closing")] = {"app_id": "ap-wrong"}
    with pytest.raises(ValueError, match="identity"):
        serve.read_observations(plan, "ap-test", sdk=sdk)


def test_missing_gpu_observation_stays_unknown(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError()))
    assert serve.observe_gpu() is None


def test_import_is_optional_and_offline():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import deploy.modal.serve; import recall_guard; "
                "assert 'modal' not in sys.modules; assert 'vllm' not in sys.modules"
            ),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_unsafe_upload_rejected_before_sdk_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(serve, "require_operator_python", lambda: None)
    # The checkout root itself may be symlink-free; this tests a parent-directory escape.
    sdk = FakeSDK()
    with pytest.raises(ValueError, match="upload"):
        serve.create_app(sdk=sdk, root=tmp_path)
    assert sdk.calls == []


def test_snapshot_has_observed_runtime_and_no_claimed_hardware(monkeypatch, tmp_path):
    plan = load_plan(MANIFEST)
    monkeypatch.setattr(serve, "observe_gpu", lambda: None)
    monkeypatch.setattr(serve, "version", lambda name: "0.17.0" if name == "vllm" else "1.6.3")
    snapshot = serve.snapshot(plan, "ap-test", "ready", cache=tmp_path)
    assert snapshot["gpu"] is None
    assert snapshot["runtime"] == "vllm==0.17.0"
    assert snapshot["resolved_model_revision"] is None
    assert snapshot["resolved_tokenizer_revision"] is None
    assert snapshot["score_semantics"] == "natural_log_probabilities"
    assert snapshot["distribution"] == "partial_top_k"
    json.dumps(snapshot)


def test_startup_and_closing_observations_use_control_plane(monkeypatch):
    monkeypatch.setattr(serve, "require_operator_python", lambda: None)
    observed_id = "ap-" + "a" * 22
    monkeypatch.setattr(serve, "running_app_id", lambda: observed_id)
    sdk = FakeSDK()
    serve.create_app(sdk=sdk)
    monkeypatch.setattr(serve, "_expiry_store", lambda plan: sdk.store)
    process = SimpleNamespace(
        poll=lambda: None,
        terminate=lambda: calls.append("terminate"),
        wait=lambda **kwargs: calls.append("wait"),
    )
    calls = []
    monkeypatch.setattr(serve, "require_remote_expiry", lambda plan, app_id, store: calls.append(("expiry", app_id)))
    monkeypatch.setattr(serve, "version", lambda name: "0.17.0")
    monkeypatch.setattr(
        serve,
        "snapshot",
        lambda plan, app_id, phase: {
            "app_id": app_id,
            "phase": phase,
        },
    )
    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(
        serve, "urlopen", lambda url, **kwargs: calls.append(url) or nullcontext(SimpleNamespace(status=200))
    )
    service = sdk.service()
    service.start()
    assert service.readiness() == {"app_id": observed_id, "ready": True}
    service.web()
    service.close()
    assert [value["phase"] for value in sdk.store.values()] == ["starting", "ready", "closing"]
    assert calls == [("expiry", observed_id), "http://127.0.0.1:8000/health", "terminate", "wait"]
    assert all(key[0] == observed_id for key in sdk.store)


def test_missing_running_identity_blocks_expiry_and_process_before_observations(monkeypatch):
    monkeypatch.setattr(serve, "require_operator_python", lambda: None)
    sdk = FakeSDK()
    serve.create_app(sdk=sdk)

    def missing_identity():
        raise RuntimeError("Missing running container app identity")

    def forbidden(*args, **kwargs):
        pytest.fail("Startup continued without an observed container app ID")

    monkeypatch.setattr(serve, "running_app_id", missing_identity)
    monkeypatch.setattr(serve, "require_remote_expiry", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    service = sdk.service()
    with pytest.raises(RuntimeError, match="Missing running"):
        service.start()
    service.close()
    assert sdk.store == {}


def test_cache_observations_hash_actual_pinned_files(monkeypatch, tmp_path):
    plan = load_plan(MANIFEST)
    directory = tmp_path / "hub/models--openai--gpt-oss-20b/snapshots" / plan.artifacts.model_revision
    directory.mkdir(parents=True)
    (directory / "config.json").write_text('{"quantization_config":{"quant_method":"mxfp4"}}')
    (directory / "tokenizer.json").write_text("{}")
    monkeypatch.setattr(serve, "observe_gpu", lambda: [{"name": "actual-gpu"}])
    monkeypatch.setattr(serve, "version", lambda name: "0.17.0")
    observation = serve.snapshot(plan, "ap-test", "ready", cache=tmp_path)
    assert observation["resolved_model_revision"] == plan.artifacts.model_revision
    assert observation["resolved_tokenizer_revision"] == plan.artifacts.tokenizer_revision
    assert observation["quantization_config"] == {"quant_method": "mxfp4"}
    assert len(observation["tokenizer_sha256"]["tokenizer.json"]) == 64
    assert observation["tokenizer_sha256"]["chat_template.jinja"] is None
    assert observation["gpu"] == [{"name": "actual-gpu"}]
