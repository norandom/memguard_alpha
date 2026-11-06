"""Task 8.1 setup tests: all operations must remain local."""

import copy
import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from deploy.modal.manifest import load_plan

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "configs/modal/gpt-oss-20b-pilot.json"


def test_frozen_plan_and_units():
    plan = load_plan(MANIFEST)
    assert plan.resources.cpu_physical_cores == 4
    assert plan.resources.memory_gib == 32
    assert plan.memory_mib == 32768
    assert plan.limits.inference_posts == 8
    assert plan.limits.total_output_tokens == 4096
    assert "workspace" in plan.execution_blockers
    assert "provider_budget" in plan.execution_blockers


def test_plan_does_not_access_network_or_environment(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Offline planning attempted network or credential access")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(os, "getenv", forbidden)
    assert load_plan(MANIFEST).artifacts.modal_sdk == "1.6.3"


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("resources", "memory_gib", 32768),
        ("resources", "gpu_count", True),
        ("limits", "inference_posts", 9),
        ("limits", "request_seconds", 121),
        ("artifacts", "model_revision", "main"),
        ("artifacts", "image", "vllm/vllm-openai:latest"),
        ("target", "credential_env", "secret.value"),
    ],
)
def test_invalid_plan_rejected(tmp_path, section, key, value):
    raw = copy.deepcopy(json.loads(MANIFEST.read_text(encoding="utf-8")))
    raw[section][key] = value
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        load_plan(path)


def test_default_cli_is_offline_and_never_resolves_credentials(monkeypatch):
    monkeypatch.setenv("MODAL_PROXY_BEARER", "do-not-print-this-secret")
    result = subprocess.run(
        [sys.executable, str(ROOT / "deploy/modal/manifest.py"), str(MANIFEST)],
        capture_output=True,
        text=True,
        check=True,
    )
    output = json.loads(result.stdout)
    assert output["mode"] == "plan"
    assert output["cloud_mutations"] == output["inference_posts"] == 0
    assert "do-not-print-this-secret" not in result.stdout + result.stderr


def test_upload_allowlist():
    plan = load_plan(MANIFEST)
    assert plan.upload_allowed("deploy/modal/serve.py")
    for path in (
        "models/weights",
        "localmodels/weights",
        ".env",
        "deploy/modal/.env",
        "secrets/key",
        "../serve.py",
        "C:/secrets",
        "deploy/modal/../.env",
    ):
        assert not plan.upload_allowed(path)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda raw: raw.pop("provenance"),
        lambda raw: raw["provenance"].pop("license_prerequisites"),
        lambda raw: raw["cost"].update(hard_ceiling=True),
        lambda raw: raw["cost"].update(pricing_checked_at="undated"),
        lambda raw: raw["cost"].update(provider_budget={"remaining_usd": 5}),
        lambda raw: raw["upload_allowlist"].append("models/weights"),
        lambda raw: raw["cache"].update(staging_gpu_count=1),
        lambda raw: raw["target"].update(endpoint="https://user:secret@example.org/v1"),
        lambda raw: raw["target"].update(secret="never-admit-secrets"),
    ],
)
def test_missing_or_unsafe_setup_rejected(tmp_path, mutate):
    raw = json.loads(MANIFEST.read_text(encoding="utf-8"))
    mutate(raw)
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError):
        load_plan(path)


def test_isolated_sdk_lock():
    import tomllib

    project = tomllib.loads((ROOT / "deploy/modal/pyproject.toml").read_text(encoding="utf-8"))
    lock = tomllib.loads((ROOT / "deploy/modal/uv.lock").read_text(encoding="utf-8"))
    assert set(project["project"]["dependencies"]) == {
        "modal==1.6.3", "recall-guard", "transformers==4.55.4", "huggingface-hub==0.34.4",
        "jinja2==3.1.6",
    }
    assert project["tool"]["uv"]["sources"]["recall-guard"] == {"path": "../..", "editable": True}
    packages = {package["name"]: package for package in lock["package"]}
    assert packages["modal"]["version"] == "1.6.3"
    assert packages["transformers"]["version"] == "4.55.4"
    assert packages["huggingface-hub"]["version"] == "0.34.4"
    assert packages["jinja2"]["version"] == "3.1.6"
    assert Path(packages["recall-guard"]["source"]["editable"]) == Path("../..")
    assert not {"torch", "vllm", "accelerate"} & packages.keys()
    core = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert not any(
        name in dependency
        for name in ("modal", "transformers", "huggingface-hub", "torch", "vllm")
        for dependency in core["project"]["dependencies"]
    )
