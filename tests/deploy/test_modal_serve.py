"""Offline serving recipe tests; fake SDK calls never allocate compute."""

import hashlib
import json
import os
import subprocess
import sys
from contextlib import nullcontext
from dataclasses import replace
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


@pytest.mark.parametrize("manifest,volume", [
    ("configs/modal/gpt-oss-20b-pilot.json", "recall-guard-gpt-oss-20b-weights"),
    ("configs/modal/gpt-oss-20b-measurement.json", "recall-guard-gpt-oss-20b-weights"),
    ("configs/modal/gemma-3-4b-measurement.json", "recall-guard-gemma-3-4b-weights"),
])
def test_each_manifest_mounts_its_own_reviewed_volume(monkeypatch, manifest, volume):
    monkeypatch.setattr(serve, "require_operator_python", lambda: None)
    sdk = FakeSDK()
    serve.create_app(sdk=sdk, manifest=manifest)
    volumes = [(args, kwargs) for name, args, kwargs in sdk.calls if name == "volume"]
    plan = load_plan(ROOT / manifest)
    assert plan.volume_name == volume == json.loads((ROOT / manifest).read_text(encoding="utf-8"))["cache"]["volume"]
    assert volumes == [((volume,), {"environment_name": plan.target.environment, "create_if_missing": False})]


class StagingSDK:
    """Records the staging recipe; never reads the environment or builds anything."""

    def __init__(self):
        self.calls = []
        self.Image = SimpleNamespace(debian_slim=lambda **kwargs: self.image("debian_slim", (), kwargs))
        self.Volume = SimpleNamespace(from_name=lambda *args, **kwargs: self.record("volume", args, kwargs))
        self.Secret = SimpleNamespace(from_local_environ=lambda env_keys: self.record("secret", (env_keys,), {}))
        self.App = lambda name: self.record("app", (name,), {}, SimpleNamespace(function=self.function))

    def record(self, name, args, kwargs, value=None):
        self.calls.append((name, args, kwargs))
        return value if value is not None else (name, args)

    def image(self, name, args, kwargs):
        self.calls.append((name, args, kwargs))
        return SimpleNamespace(pip_install=lambda *packages: self.image("pip_install", packages, {}),
                               env=lambda settings: self.image("env", (), settings),
                               add_local_file=lambda *a, **k: self.image("upload", a, k))

    def function(self, **options):
        self.calls.append(("function", (), options))

        def wrap(fn):
            self.body = fn
            return SimpleNamespace(remote=SimpleNamespace(aio=None))

        return wrap


def test_staging_recipe_is_cpu_only_bounded_and_names_hf_token_only(monkeypatch):
    import tomllib

    monkeypatch.setattr(serve, "require_operator_python", lambda: None)
    sdk = StagingSDK()
    app, function = serve.create_staging_app(max_seconds=300, sdk=sdk, manifest="configs/modal/gemma-3-4b-measurement.json")
    plan = load_plan(GEMMA)
    calls = {name: (args, kwargs) for name, args, kwargs in sdk.calls}
    assert calls["app"][0] == (plan.target.app + "-staging",)
    options = calls["function"][1]
    assert options["gpu"] is None and "gpu" in options
    assert options["cpu"] == (4, 4) and options["memory"] == (16384, 16384)
    assert options["timeout"] == 300 and options["retries"] == 0
    assert options["max_containers"] == 1 and options["min_containers"] == options["buffer_containers"] == 0
    assert options["include_source"] is False and options["serialized"] is True
    assert options["volumes"] == {"/cache/huggingface": ("volume", ("recall-guard-gemma-3-4b-weights",))}
    assert calls["volume"][1] == {"environment_name": plan.target.environment, "create_if_missing": True}
    # HF_TOKEN by name only: the recipe passes the variable name, never a value.
    assert options["secrets"] == [("secret", (["HF_TOKEN"],))]
    lock = tomllib.loads((ROOT / "deploy/modal/uv.lock").read_text(encoding="utf-8"))
    locked = {package["name"]: package["version"] for package in lock["package"]}
    assert calls["pip_install"][0] == (f"huggingface_hub=={locked['huggingface-hub']}", f"hf-xet=={locked['hf-xet']}")
    assert calls["env"][1]["HF_HOME"] == "/cache/huggingface" and "HF_HUB_OFFLINE" not in calls["env"][1]
    uploads = [args[1] for name, args, _ in sdk.calls if name == "upload"]
    assert uploads == ["/root/deploy/modal/serve.py", "/root/deploy/modal/manifest.py",
                       "/root/configs/modal/gemma-3-4b-measurement.json"]
    assert function.remote is not None and sdk.body.__name__ == "stage"


@pytest.mark.parametrize("max_seconds", [0, -1, 86143, True, 300.0])
def test_staging_recipe_refuses_a_timeout_outside_the_gpu_phase(monkeypatch, max_seconds):
    monkeypatch.setattr(serve, "require_operator_python", lambda: None)
    sdk = StagingSDK()
    with pytest.raises(ValueError, match="Staging timeout"):
        serve.create_staging_app(max_seconds=max_seconds, sdk=sdk, manifest="configs/modal/gemma-3-4b-measurement.json")
    assert sdk.calls == []


class FakeHub:
    """Pinned-revision Hub listing with an extra unindexed weight copy that is never fetched."""

    def __init__(self, plan, *, shards=("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")):
        self.revision = plan.artifacts.model_revision
        self.shards = shards
        self.files = {**dict.fromkeys(shards, b"w" * 32), "config.json": b"{}", "tokenizer.json": b"{}",
                      "tokenizer_config.json": b"{}", "chat_template.json": b"{}", "README.md": b"r",
                      "original/model.safetensors": b"x" * 64}
        self.requests = []

    def snapshot(self, cache_dir, repo):
        return Path(cache_dir) / ("models--" + repo.replace("/", "--")) / "snapshots" / self.revision

    def hf_hub_download(self, repo, filename, *, revision, cache_dir):
        self.requests.append(("file", repo, filename, revision))
        path = self.snapshot(cache_dir, repo) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"weight_map": {f"layer.{i}": shard for i, shard in enumerate(self.shards)}}))
        return str(path)

    def snapshot_download(self, repo, *, revision, cache_dir, allow_patterns):
        self.requests.append(("snapshot", repo, tuple(allow_patterns), revision))
        directory = self.snapshot(cache_dir, repo)
        for name, data in self.files.items():
            if name in allow_patterns:
                (directory / name).parent.mkdir(parents=True, exist_ok=True)
                (directory / name).write_bytes(data)
        return str(directory)


def test_staging_body_fetches_the_indexed_shards_into_the_serving_layout_and_commits(tmp_path):
    plan = load_plan(GEMMA)
    hub, commits = FakeHub(plan), []
    result = serve.stage_weights(plan, cache=tmp_path, hub=hub, commit=lambda: commits.append("commit"))
    snapshot = tmp_path / "hub/models--google--gemma-3-4b-it/snapshots" / plan.artifacts.model_revision
    assert commits == ["commit"]
    assert (snapshot / "model-00002-of-00002.safetensors").is_file()
    assert not (snapshot / "original").exists() and not (snapshot / "README.md").exists()
    assert [request[0] for request in hub.requests] == ["file", "snapshot"]
    assert all(request[3] == plan.artifacts.model_revision for request in hub.requests)
    assert result["volume"] == "recall-guard-gemma-3-4b-weights"
    assert (result["model_revision"], result["tokenizer_revision"]) == (plan.artifacts.model_revision,) * 2
    files = [path for path in snapshot.rglob("*") if path.is_file()]
    assert result["files"] == len(files) == 7
    assert result["bytes"] == sum(path.stat().st_size for path in files)
    assert result["duration_seconds"] >= 0
    # The serving snapshot reads this layout offline.
    observed = serve.snapshot(plan, "ap-test", "ready", cache=tmp_path)
    assert observed["resolved_model_revision"] == plan.artifacts.model_revision


@pytest.mark.parametrize("shards", [("../escape.safetensors",), ()])
def test_staging_body_refuses_an_index_without_plain_shards(tmp_path, shards):
    plan = load_plan(GEMMA)
    with pytest.raises(ValueError, match="shard"):
        serve.stage_weights(plan, cache=tmp_path, hub=FakeHub(plan, shards=shards), commit=lambda: None)


VISION_CONFIG = json.dumps({"architectures": ["Gemma3ForConditionalGeneration"],
                            "vision_config": {"model_type": "siglip_vision_model"}}).encode()


@pytest.mark.parametrize("processor_files", [False, True])
def test_staging_body_requires_processor_files_for_a_vision_config(tmp_path, processor_files):
    plan = load_plan(GEMMA)
    hub, commits = FakeHub(plan), []
    hub.files["config.json"] = VISION_CONFIG
    if processor_files:
        hub.files.update({"preprocessor_config.json": b"{}", "processor_config.json": b"{}"})
        result = serve.stage_weights(plan, cache=tmp_path, hub=hub, commit=lambda: commits.append("commit"))
        assert result["files"] == 9 and commits == ["commit"]
    else:
        # A silent pattern miss fails here, before the volume commit, not at the paid smoke run.
        with pytest.raises(ValueError, match="required file"):
            serve.stage_weights(plan, cache=tmp_path, hub=hub, commit=lambda: commits.append("commit"))
        assert commits == []


def test_staging_body_needs_no_processor_files_without_a_vision_config(tmp_path):
    plan = load_plan(GEMMA)
    hub, commits = FakeHub(plan), []
    hub.files["config.json"] = json.dumps({"architectures": ["Gemma3ForCausalLM"]}).encode()
    result = serve.stage_weights(plan, cache=tmp_path, hub=hub, commit=lambda: commits.append("commit"))
    assert result["files"] == 7 and commits == ["commit"]


def test_staging_body_refuses_a_snapshot_missing_required_files(tmp_path):
    plan = load_plan(GEMMA)
    hub, commits = FakeHub(plan), []
    del hub.files["tokenizer.json"]
    with pytest.raises(ValueError, match="required file"):
        serve.stage_weights(plan, cache=tmp_path, hub=hub, commit=lambda: commits.append("commit"))
    assert commits == []


def require_symlinks(directory: Path) -> None:
    """The Hub cache layout needs symlinks; Windows refuses them without a privilege."""
    target = directory / "symlink-target"
    target.write_bytes(b"")
    try:
        (directory / "symlink-probe").symlink_to(target)
    except OSError as error:
        pytest.skip(f"Symlinks unavailable on this platform: {error}")


class CacheLayoutHub(FakeHub):
    """What huggingface_hub writes with cache_dir: bytes under blobs, relative symlinks as pointer files, lock files."""

    def __init__(self, plan, **kwargs):
        super().__init__(plan, **kwargs)
        self.written = {}

    def store(self, cache_dir, repo, name, data, revision=None):
        repo_dir = Path(cache_dir) / ("models--" + repo.replace("/", "--"))
        blob = repo_dir / "blobs" / hashlib.sha256(data).hexdigest()
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(data)
        lock = Path(cache_dir) / ".locks" / repo_dir.name / (blob.name + ".lock")
        lock.parent.mkdir(parents=True, exist_ok=True)
        lock.write_bytes(b"")
        pointer = repo_dir / "snapshots" / (revision or self.revision) / name
        pointer.parent.mkdir(parents=True, exist_ok=True)
        if not pointer.is_symlink():
            pointer.symlink_to(os.path.relpath(blob, pointer.parent))
        self.written[name] = data
        return pointer

    def hf_hub_download(self, repo, filename, *, revision, cache_dir):
        self.requests.append(("file", repo, filename, revision))
        index = json.dumps({"weight_map": {f"layer.{i}": shard for i, shard in enumerate(self.shards)}}).encode()
        return str(self.store(cache_dir, repo, filename, index, revision))

    def snapshot_download(self, repo, *, revision, cache_dir, allow_patterns):
        self.requests.append(("snapshot", repo, tuple(allow_patterns), revision))
        for name, data in self.files.items():
            if name in allow_patterns:
                self.store(cache_dir, repo, name, data, revision)
        return str(Path(cache_dir) / ("models--" + repo.replace("/", "--")) / "snapshots" / revision)


class RerunHub(FakeHub):
    """A second run: a commit-hash revision resolves to the existing pointer files and downloads nothing."""

    def hf_hub_download(self, repo, filename, *, revision, cache_dir):
        self.requests.append(("file", repo, filename, revision))
        return str(self.snapshot(cache_dir, repo) / filename)

    def snapshot_download(self, repo, *, revision, cache_dir, allow_patterns):
        self.requests.append(("snapshot", repo, tuple(allow_patterns), revision))
        return str(self.snapshot(cache_dir, repo))


def test_staging_body_replaces_cache_symlinks_with_regular_files_before_committing(tmp_path):
    require_symlinks(tmp_path)
    plan = load_plan(GEMMA)
    hub, seen = CacheLayoutHub(plan), []
    repo = tmp_path / "hub/models--google--gemma-3-4b-it"
    snapshot = repo / "snapshots" / plan.artifacts.model_revision

    def commit():
        seen.append({"symlinks": sorted(path.name for path in snapshot.rglob("*") if path.is_symlink()),
                     "blobs": (repo / "blobs").exists(), "locks": (tmp_path / "hub/.locks").exists()})

    result = serve.stage_weights(plan, cache=tmp_path, hub=hub, commit=commit)
    # The commit saw the dereferenced layout: no pointer link, no blobs, no lock files.
    assert seen == [{"symlinks": [], "blobs": False, "locks": False}]
    files = sorted(snapshot.rglob("*"))
    assert files and all(path.is_file() and not path.is_symlink() for path in files)
    assert {path.name: path.read_bytes() for path in files} == hub.written
    assert result["files"] == len(hub.written) == 7
    assert result["bytes"] == sum(len(data) for data in hub.written.values())
    observed = serve.snapshot(plan, "ap-test", "ready", cache=tmp_path)
    assert observed["resolved_model_revision"] == plan.artifacts.model_revision


@pytest.mark.parametrize("target", ["outside", "missing"])
def test_staging_body_refuses_a_snapshot_link_that_leaves_the_cache(tmp_path, target):
    require_symlinks(tmp_path)
    plan = load_plan(GEMMA)
    hub, commits = CacheLayoutHub(plan), []
    snapshot = hub.snapshot(tmp_path / "hub", plan.artifacts.model)
    snapshot.mkdir(parents=True)
    (tmp_path / "outside.json").write_bytes(b"{}")
    (snapshot / "stray.json").symlink_to(tmp_path / (target + ".json"))
    with pytest.raises(ValueError, match="stray.json"):
        serve.stage_weights(plan, cache=tmp_path, hub=hub, commit=lambda: commits.append("commit"))
    assert commits == [] and (snapshot / "stray.json").is_symlink()


def test_staging_body_dereferences_a_tokenizer_at_another_revision_of_the_same_repo(tmp_path):
    require_symlinks(tmp_path)
    plan = load_plan(GEMMA)
    plan = replace(plan, artifacts=replace(plan.artifacts, tokenizer_revision="b" * 40))
    hub, seen = CacheLayoutHub(plan), []
    repo = tmp_path / "hub/models--google--gemma-3-4b-it"
    model = repo / "snapshots" / plan.artifacts.model_revision
    tokenizer = repo / "snapshots" / ("b" * 40)

    def commit():
        seen.append({"links": [path.name for path in repo.rglob("*") if path.is_symlink()],
                     "blobs": (repo / "blobs").exists()})

    result = serve.stage_weights(plan, cache=tmp_path, hub=hub, commit=commit)
    # Both snapshots shared one blobs directory; it went only after the second snapshot was dereferenced.
    assert seen == [{"links": [], "blobs": False}]
    for snapshot in (model, tokenizer):
        files = list(snapshot.rglob("*"))
        assert files and all(path.is_file() and not path.is_symlink() for path in files)
    assert (tokenizer / "tokenizer.json").read_bytes() == hub.written["tokenizer.json"]
    assert (model / "model-00001-of-00002.safetensors").read_bytes() == hub.written["model-00001-of-00002.safetensors"]
    counted = {path for directory in (model, tokenizer) for path in directory.rglob("*") if path.is_file()}
    assert result["files"] == len(counted) == 11
    assert result["bytes"] == sum(path.stat().st_size for path in counted)
    assert result["tokenizer_revision"] == "b" * 40


def test_staging_body_tolerates_a_rerun_over_regular_pointer_files_without_blobs(tmp_path):
    plan = load_plan(GEMMA)
    first = serve.stage_weights(plan, cache=tmp_path, hub=FakeHub(plan), commit=lambda: None)
    repo = tmp_path / "hub/models--google--gemma-3-4b-it"
    snapshot = repo / "snapshots" / plan.artifacts.model_revision
    before = {path.name: path.read_bytes() for path in snapshot.rglob("*")}
    hub, commits = RerunHub(plan), []
    result = serve.stage_weights(plan, cache=tmp_path, hub=hub, commit=lambda: commits.append("commit"))
    assert commits == ["commit"] and [request[0] for request in hub.requests] == ["file", "snapshot"]
    assert {path.name: path.read_bytes() for path in snapshot.rglob("*")} == before
    assert all(path.is_file() and not path.is_symlink() for path in snapshot.rglob("*"))
    assert (result["files"], result["bytes"]) == (first["files"], first["bytes"]) == (7, first["bytes"])
    assert not (repo / "blobs").exists() and not (tmp_path / "hub/.locks").exists()


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
