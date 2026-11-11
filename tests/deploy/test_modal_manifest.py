"""Task 8.1 setup tests: all operations must remain local."""

import copy
import json
import os
import re
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
    assert "paid_execution_not_authorized" in plan.execution_blockers
    assert "provider_budget" in plan.execution_blockers


def test_plan_does_not_access_network_or_environment(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Offline planning attempted network or credential access")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(os, "getenv", forbidden)
    assert load_plan(MANIFEST).artifacts.modal_sdk == "1.6.3"


@pytest.mark.parametrize("endpoint", ["https://example.modal.run/v1", "https://example.us-east.modal.run/v1"])
def test_explicit_global_and_regional_endpoints(tmp_path, endpoint):
    raw = json.loads(MANIFEST.read_text(encoding="utf-8"))
    raw["target"]["endpoint"] = endpoint
    path = tmp_path / "regional.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    assert load_plan(path).target.endpoint == endpoint


@pytest.mark.parametrize("endpoint", [
    "https://example.modal.run.evil.org/v1", "https://user:secret@example.us-east.modal.run/v1",
    "https://example.us-east.modal.run/v1?redirect=evil", "http://example.modal.run/v1",
])
def test_unsafe_endpoint_variants_rejected(tmp_path, endpoint):
    raw = json.loads(MANIFEST.read_text(encoding="utf-8"))
    raw["target"]["endpoint"] = endpoint
    path = tmp_path / "unsafe.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid explicit Modal endpoint"):
        load_plan(path)


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


GEMMA = ROOT / "configs/modal/gemma-3-4b-measurement.json"
GEMMA_REVISION = "093f9f388b31de276ce2de164bdc2081324b9767"
#: Modal's maximum function timeout (https://modal.com/docs/guide/timeouts).
MODAL_MAX_TIMEOUT_SECONDS = 86400


def test_gemma_manifest_loads_without_a_harmony_tokenizer():
    from deploy.modal.manifest import GEMMA_MANIFEST, REVIEWED_LIMITS

    plan = load_plan(GEMMA)
    raw = json.loads(GEMMA.read_text(encoding="utf-8"))
    assert plan.serving_manifest == GEMMA_MANIFEST and raw["upload_allowlist"][2] == GEMMA_MANIFEST
    a = plan.artifacts
    assert (a.model, a.tokenizer, a.model_revision, a.tokenizer_revision) == (
        "google/gemma-3-4b-it", "google/gemma-3-4b-it", GEMMA_REVISION, GEMMA_REVISION)
    assert (a.quantization, a.moe_backend, a.attention_backend, a.runtime) == (
        "bf16", "none", "TRITON_ATTN", "vllm==0.17.0")
    assert (plan.resources.gpu, plan.resources.gpu_count, plan.memory_mib) == ("A10", 1, 16384)
    assert raw["cache"]["volume"] == "recall-guard-gemma-3-4b-weights" and raw["cache"]["staging_gpu_count"] == 0
    assert (plan.limits, raw["cost"]["target_usd"]) == REVIEWED_LIMITS[GEMMA_MANIFEST]
    assert "paid_execution_not_authorized" in plan.execution_blockers
    provenance = raw["provenance"]
    assert provenance["license"] != "Apache-2.0" and "Gemma Terms of Use" in provenance["license"]
    assert "gated=manual" in provenance["access"] and "HF_TOKEN" in provenance["access"]
    # Credentials by name only: no token value, and the HF credential is never a deployment input.
    assert "HF_TOKEN" not in json.dumps(raw["target"]) and not re.search(r"hf_[A-Za-z0-9]{20,}", json.dumps(raw))
    assert "\u2014" not in GEMMA.read_text(encoding="utf-8")  # No em dash.


def test_gemma_ceiling_covers_the_serving_arm_and_fits_modal_timeouts():
    limits = load_plan(GEMMA).limits
    draws, renderings, macro_months, edgar_calibration, negative_control, smoke = 32, 2, 126, 40 + 100, 40, 32
    measurement = macro_months * renderings * draws + edgar_calibration * draws + negative_control * draws + smoke
    assert (limits.eligibility_posts, limits.measurement_posts) == (1, measurement) == (1, 13856)
    assert limits.inference_posts == limits.eligibility_posts + limits.measurement_posts
    assert limits.total_output_tokens == limits.inference_posts * limits.output_tokens_per_post
    assert limits.retries == 0 and limits.temperature == 0 and limits.top_logprobs == 5
    # Six seconds per worst-case post plus five GPU lifecycles' startup allowance.
    assert limits.gpu_phase_seconds == limits.inference_posts * 6 + 5 * limits.startup_seconds
    # The expiry controller's timeout is the GPU phase plus 150 s and must fit Modal's maximum.
    assert limits.gpu_phase_seconds + 150 <= MODAL_MAX_TIMEOUT_SECONDS


def test_gpt_oss_manifests_keep_their_reviewed_candidate():
    from deploy.modal.manifest import MEASUREMENT_MANIFEST, PILOT_MANIFEST, REVIEWED_LIMITS

    for path in (MANIFEST, ROOT / MEASUREMENT_MANIFEST):
        plan = load_plan(path)
        assert plan.artifacts.model == "openai/gpt-oss-20b" and plan.artifacts.moe_backend == "marlin"
        assert plan.resources.gpu == "A100-40GB"
        assert json.loads(path.read_text(encoding="utf-8"))["cache"]["volume"] == "recall-guard-gpt-oss-20b-weights"
    assert REVIEWED_LIMITS[PILOT_MANIFEST][1] == 5 and REVIEWED_LIMITS[MEASUREMENT_MANIFEST][1] == 2


#: The GPT-OSS-20B manifests as committed before task 3.5 (sha256 of the LF-normalized bytes).
GPT_OSS_MANIFEST_SHA256 = {
    "configs/modal/gpt-oss-20b-pilot.json": "cf656468675ecd2dbf39e4586e419a1681db32808d0eb5b923298485bd71fc12",
    "configs/modal/gpt-oss-20b-measurement.json": "316fb1c91db3afd2403253b189282680d88fa600e15659bbe4bf0ad20ad5b545",
}


def test_gpt_oss_manifests_are_byte_identical():
    import hashlib

    for relative, expected in GPT_OSS_MANIFEST_SHA256.items():
        data = (ROOT / relative).read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(data).hexdigest() == expected, relative


@pytest.mark.parametrize("relative,volume", [
    ("configs/modal/gpt-oss-20b-pilot.json", "recall-guard-gpt-oss-20b-weights"),
    ("configs/modal/gpt-oss-20b-measurement.json", "recall-guard-gpt-oss-20b-weights"),
    ("configs/modal/gemma-3-4b-measurement.json", "recall-guard-gemma-3-4b-weights"),
])
def test_each_manifest_resolves_its_own_weight_volume(relative, volume):
    plan = load_plan(ROOT / relative)
    assert plan.volume_name == volume
    assert json.loads((ROOT / relative).read_text(encoding="utf-8"))["cache"]["volume"] == volume


def test_gemma_caveats_bound_the_arm_by_stage_limits_not_the_per_lifecycle_phase():
    caveats = json.loads(GEMMA.read_text(encoding="utf-8"))["cost"]["caveats"]
    assert "applies the full gpu_phase_seconds to every GPU lifecycle it arms without a stage limit" in caveats
    assert "the per-stage limits (task 7.3) bound the arm" in caveats
    assert "arms each lifecycle's expiry at its stage's max_seconds, never at gpu_phase_seconds" in caveats


@pytest.mark.parametrize(
    "source,mutate,message",
    [
        # A manifest serves only its own reviewed candidate: no mixing of models, resources or volumes.
        ("gemma", lambda raw, other: raw.update(artifacts=other["artifacts"]), "Unsupported artifact"),
        ("gemma", lambda raw, other: raw.update(resources=other["resources"]), "Unsupported artifact"),
        ("gemma", lambda raw, other: raw.update(cache=other["cache"]), "Cache staging"),
        ("gemma", lambda raw, other: raw["limits"].update(inference_posts=13858), "Measurement limits"),
        ("gemma", lambda raw, other: raw["cost"].update(target_usd=40), "reviewed USD target"),
        ("gemma", lambda raw, other: raw["artifacts"].update(moe_backend="marlin"), "Unsupported artifact"),
        ("gemma", lambda raw, other: raw["cache"].update(staging_gpu_count=1), "Cache staging"),
        ("pilot", lambda raw, other: raw.update(artifacts=other["artifacts"]), "Unsupported artifact"),
        ("pilot", lambda raw, other: raw.update(cache=other["cache"]), "Cache staging"),
        ("pilot", lambda raw, other: raw.update(resources=other["resources"]), "Unsupported artifact"),
    ],
)
def test_reviewed_candidates_cannot_be_mixed_or_raised(tmp_path, source, mutate, message):
    pilot = json.loads(MANIFEST.read_text(encoding="utf-8"))
    gemma = json.loads(GEMMA.read_text(encoding="utf-8"))
    raw, other = (gemma, pilot) if source == "gemma" else (pilot, gemma)
    mutate(raw, copy.deepcopy(other))
    path = tmp_path / "mixed.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_plan(path)


def test_gemma_cli_plan_is_offline():
    result = subprocess.run([sys.executable, str(ROOT / "deploy/modal/manifest.py"), str(GEMMA)],
                            capture_output=True, text=True, check=True)
    output = json.loads(result.stdout)
    assert output["mode"] == "plan" and output["memory_mib"] == 16384
    assert output["cloud_mutations"] == output["inference_posts"] == 0


def test_isolated_sdk_lock():
    import tomllib

    project = tomllib.loads((ROOT / "deploy/modal/pyproject.toml").read_text(encoding="utf-8"))
    lock = tomllib.loads((ROOT / "deploy/modal/uv.lock").read_text(encoding="utf-8"))
    assert project["project"]["requires-python"] == ">=3.12,<3.13"
    assert lock["requires-python"] == "==3.12.*"
    assert (ROOT / "deploy/modal/.python-version").read_text().strip() == "3.12"
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
