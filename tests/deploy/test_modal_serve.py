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
        return SimpleNamespace(add_local_file=self.add_file, env=self.env)

    def add_file(self, *args, **kwargs):
        self.calls.append(("upload", args, kwargs))
        return SimpleNamespace(add_local_file=self.add_file, env=self.env)

    def env(self, settings):
        self.calls.append(("env", (), settings))
        return self

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
    sdk = FakeSDK()
    serve.create_app(sdk=sdk)
    process = SimpleNamespace(
        poll=lambda: None,
        terminate=lambda: calls.append("terminate"),
        wait=lambda **kwargs: calls.append("wait"),
    )
    calls = []
    monkeypatch.setattr(serve, "require_remote_expiry", lambda *args: calls.append("expiry"))
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
    service.web()
    service.close()
    assert [value["phase"] for value in sdk.store.values()] == ["starting", "ready", "closing"]
    assert calls == ["expiry", "http://127.0.0.1:8000/health", "terminate", "wait"]
    assert all(key[0] == "ap-test" for key in sdk.store)


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
