"""NPU (FastFlowLM) Phase A wiring: inputs, cutoff filter, observer, smoke and gateway tags. Offline."""
import json
import math
import shutil
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import pytest
import requests

from recall_guard.core.loader import load_cutoffs
from recall_guard.core.profiles import FLM_INSTALLED_TAGS_SOURCE
from recall_guard.harness.local_evaluation import load_plan
from scripts import build_phase_a_inputs as build
from scripts import measure_local_models as cli
from scripts import smoke_local_models as smoke
from scripts.probe_local_model import OpenAICompatibleClient

ROOT = Path(__file__).resolve().parents[2]
NPU_ENDPOINT = "http://172.17.32.1:11541/v1/chat/completions"
DIGEST = "sha256:06cb8f39d3f84850c3e1938de304605cf842fddbb5fef946fa88140d811ff117"


def inputs(directory, profile):
    return tuple(directory / profile / name for name in ("plan.json", "manifest.json", "corpus.json"))


@pytest.fixture(scope="module")
def npu_output(tmp_path_factory):
    output = tmp_path_factory.mktemp("npu")
    current = Path.cwd()
    try:
        import os
        os.chdir(ROOT)
        assert build.main(["--output", str(output), "--profiles", "npu", "--npu-endpoint", NPU_ENDPOINT]) == 0
    finally:
        os.chdir(current)
    return output


def flm_metadata(model="gemma3:4b", loaded=True, context=2048, npu=True):
    return {"runtime_version": "1.0.12", "digest": DIGEST, "loaded_models": [model] if loaded else [],
            "effective_context_tokens": context, "model_allocation_bytes": None,
            "accelerator_allocation_bytes": None, "observed_backend": "NPU" if npu else None,
            "npu_available": npu, "model_installed": True}


# --- inputs ---------------------------------------------------------------------------------------


def test_npu_plans_use_the_flm_gateway_npu_mode_and_declared_identity(npu_output):
    for profile_id, total in (("flm-llama3.2-3b", 321), ("flm-gemma3-4b", 321), ("flm-gemma4-e4b", 262)):
        plan = load_plan(*inputs(npu_output, profile_id))
        assert plan.logical_requests == plan.physical_requests == plan.limits.max_requests == total
        assert plan.config.endpoint == NPU_ENDPOINT and plan.config.credential_ref == "env:LOCAL_MODEL_API_KEY"
        assert plan.requested_execution_mode == "NPU" and plan.config.provider == "fastflowlm"
        assert plan.config.local_identity.declared.backend == "NPU"
        assert plan.config.parameters["top_logprobs"] == 5 and plan.config.parameters["max_tokens"] == 32
        assert (plan.seed, plan.min_valid, plan.min_auc, plan.ensemble_draws) == (0, 50, 0.6, 0)
    index = json.loads((npu_output / "index.json").read_text(encoding="utf-8"))
    assert index["request_totals"] == {"local": 0, "modal": 0, "npu": 904}


def test_gemma4_controls_are_filtered_after_its_cutoff_and_counts_are_recorded(npu_output):
    manifest = json.loads(inputs(npu_output, "flm-gemma4-e4b")[1].read_text(encoding="utf-8"))
    assert manifest["corpus_filter"] == {
        "cutoff_key": "google/gemma-4-e4b-it", "cutoff": "2025-01-31",
        "calibration_control": {"before": 100, "after": 74}, "held_out_control": {"before": 40, "after": 33}}
    plan = load_plan(*inputs(npu_output, "flm-gemma4-e4b"))
    counts = {split: sum(item.split == split for item in plan.items) for split in ("is_memorized", "oos_control")}
    assert counts == {"is_memorized": 40, "oos_control": 74}
    held = [item.label for item in plan.items if item.split == "held_out"]
    assert sorted(held) == [0] * 33 + [1] * 40
    corpus = json.loads(inputs(npu_output, "flm-gemma4-e4b")[2].read_text(encoding="utf-8"))
    kept = {row["item_id"] for split in corpus.values() for row in split}
    rows = build.read_rows(ROOT / "data/edgar")
    for items in rows.values():
        for row in items:
            if row["label"] == 0:
                assert (build.sha256(row["metadata"]["url"].encode()) in kept) == (
                    row["metadata"]["published_at"] > "2025-01-31")
    # The other NPU profiles keep the whole corpus.
    other = json.loads(inputs(npu_output, "flm-gemma3-4b")[1].read_text(encoding="utf-8"))["corpus_filter"]
    assert other["calibration_control"] == {"before": 100, "after": 100}


def test_cutoff_filter_reports_instead_of_padding():
    rows = {"is_memorized": [{"label": 1, "metadata": {"published_at": "2020-01-01"}}],
            "oos_control": [{"label": 0, "metadata": {"published_at": f"2025-0{m}-15"}} for m in range(1, 7)],
            "held_out": [{"label": 0, "metadata": {"published_at": "2025-06-01"}}]}
    kept, counts = build.cutoff_rows(rows, date(2025, 1, 31), 5)
    assert len(kept["oos_control"]) == 5 and counts["calibration_control"] == {"before": 6, "after": 5}
    with pytest.raises(ValueError, match="control_below_min_valid:5<6"):
        build.cutoff_rows(rows, date(2025, 1, 31), 6)
    late = rows | {"is_memorized": [{"label": 1, "metadata": {"published_at": "2025-01-31"}}]}
    with pytest.raises(ValueError, match="memorized_row_not_before_cutoff"):
        build.cutoff_rows(late, date(2025, 1, 31), 1)


def test_cutoff_keys_cover_every_profile_and_gemma4_follows_the_registry_rules():
    cutoffs = load_cutoffs(ROOT / "data/cutoffs.yaml")
    assert set(build.CUTOFF_KEYS) == set(build.PROFILES)
    assert all(key in cutoffs for key in build.CUTOFF_KEYS.values())
    assert build.CUTOFF_KEYS["flm-gemma3-4b"] == build.CUTOFF_KEYS["ollama-gemma3-4b"] == "google/gemma-3-4b-it"
    assert build.CUTOFF_KEYS["flm-llama3.2-3b"] == "meta/llama-3.2-3b-instruct"
    assert cutoffs["google/gemma-4-e4b-it"] == date(2025, 1, 31)
    text = (ROOT / "data/cutoffs.yaml").read_text(encoding="utf-8")
    assert "a cutoff date of January 2025" in text and "https://huggingface.co/google/gemma-4-E4B-it" in text


@pytest.mark.parametrize("endpoint", [None, "http://172.17.32.1:11435/v1/chat/completions",
                                      "http://172.17.32.1:11541/v1/models"])
def test_npu_profiles_require_the_flm_gateway_endpoint(tmp_path, endpoint):
    with pytest.raises(SystemExit):
        build.main(["--output", str(tmp_path), "--profiles", "npu",
                    *(["--npu-endpoint", endpoint] if endpoint else [])])


# --- measurement observer and host gate -----------------------------------------------------------


def test_fastflowlm_observer_authenticates_and_reports_npu_provenance(npu_output, monkeypatch):
    plan = load_plan(*inputs(npu_output, "flm-gemma3-4b"))
    seen = {}

    def observe(endpoint, model, headers=None, **_):
        seen.update(endpoint=endpoint, model=model, headers=headers)
        return flm_metadata()
    monkeypatch.setattr("recall_guard.core.profiles.observe_fastflowlm_host", observe)
    monkeypatch.setenv("LOCAL_MODEL_API_KEY", "test-token")
    observation = cli.fastflowlm_observer(plan, "flm-gemma3-4b")()
    provenance = observation.provenance
    assert (provenance.runtime, provenance.runtime_version, provenance.backend) == ("fastflowlm", "1.0.12", "NPU")
    assert provenance.context_length == 2048 and observation.loaded_models == ("gemma3:4b",)
    observation.validate(plan, after_probe=True)
    assert seen == {"endpoint": NPU_ENDPOINT, "model": "gemma3:4b", "headers": {"Authorization": "Bearer test-token"}}


def test_fastflowlm_observer_is_refused_for_an_ollama_manifest(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(ROOT)
    assert build.main(["--output", str(tmp_path), "--profiles", "ollama-gemma3-4b",
                       "--local-endpoint", "http://172.17.32.1:11435/v1/chat/completions"]) == 0
    args = [*sum((["--" + n, str(p)] for n, p in zip(("plan", "manifest", "corpus"),
                                                       inputs(tmp_path, "ollama-gemma3-4b"), strict=True)), []),
            "--execute", "--journal", str(tmp_path / "journal.json"), "--observer", "fastflowlm"]
    assert cli.main(args) == 2 and not (tmp_path / "journal.json").exists()


def test_host_gate_runs_the_fastflowlm_preflight_branch(npu_output, tmp_path, monkeypatch):
    from datetime import UTC, datetime
    snapshot = tmp_path / "host.json"
    snapshot.write_text(json.dumps({"available_bytes": 5 * 1024 ** 3, "total_bytes": 32 * 1024 ** 3, "host": "h",
                                    "source": "windows_host_snapshot",
                                    "observed_at": datetime.now(UTC).isoformat()}))
    calls = []

    def get(self, url, **kwargs):
        calls.append(url.rsplit("/", 2)[-2:])
        answer = requests.Response()
        answer.status_code = 200
        body = {"/api/ps": {"models": []}}.get("/" + url.split("/", 3)[3], {})
        answer._content = json.dumps(body).encode()
        return answer
    monkeypatch.setattr(requests.Session, "get", get)
    monkeypatch.setenv("LOCAL_MODEL_API_KEY", "test-token")
    plan = load_plan(*inputs(npu_output, "flm-gemma3-4b"))
    assert cli.host_gate(plan, snapshot, "flm-gemma3-4b") == "insufficient_free_ram"  # 5 GiB < 6 GiB, not loaded
    assert calls == [["api", "ps"]]


# --- smoke (eligibility probe) --------------------------------------------------------------------


@pytest.fixture
def flm_rig(monkeypatch):
    state = {"mass": 1 + 1.8e-6, "posts": []}

    def post(self, payload):
        assert self.budget.used == 1 and self.config.max_retries == 0
        assert payload["model"] == "gemma3:4b" and payload["top_logprobs"] == 5 and payload["max_tokens"] == 32
        state["posts"].append(payload)
        mass = state["mass"]
        token = {"token": "OK", "logprob": math.log(0.99 * mass), "top_logprobs": [
            {"token": "OK", "logprob": math.log(0.99 * mass)}] + [
            {"token": f"x{i}", "logprob": math.log(0.0025 * mass)} for i in range(4)]}
        body = {"choices": [{"message": {"content": "OK"}, "logprobs": {"content": [token]}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11}}
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(body).encode()
        return response
    monkeypatch.setattr(OpenAICompatibleClient, "_post", post)
    return state


def test_flm_profiles_are_selectable_and_run_alone():
    assert set(smoke.flm_profiles()) == {"flm-llama3.2-3b", "flm-gemma3-4b", "flm-gemma4-e4b"}
    assert smoke.main(["--profile", "flm-gemma3-4b", "--profile", "ollama-gemma3-4b",
                       "--endpoint", NPU_ENDPOINT]) == 2
    assert smoke.main(["--profile", "flm-gemma3-4b"]) == 2  # explicit gateway endpoint required


def test_flm_smoke_dry_run_sends_no_inference(flm_rig):
    journal = smoke.run_flm_smoke("flm-gemma3-4b", NPU_ENDPOINT, observe=flm_metadata)
    assert journal["status"] == "dry_run" and flm_rig["posts"] == []
    assert journal["plan"]["logprob_mass_tolerance"] == 5e-6 and journal["plan"]["unload"] == "host_side_process_stop"


def test_flm_smoke_one_request_is_eligible_within_gemma3_tolerance(flm_rig, tmp_path):
    out = tmp_path / "flm.json"
    journal = smoke.run_flm_smoke("flm-gemma3-4b", NPU_ENDPOINT, out, execute=True, observe=flm_metadata)
    row = journal["attempts"][0]
    assert journal["status"] == "completed" and row["status"] == "eligible" and len(flm_rig["posts"]) == 1
    assert row["environment_status"] == "matched" and row["physical_attempts"] == 1
    assert row["alternative_mass_max"] == pytest.approx(1 + 1.8e-6, abs=1e-9)
    assert json.loads(out.read_text(encoding="utf-8"))["attempts"][0]["status"] == "eligible"


def test_flm_smoke_mass_beyond_tolerance_is_ineligible(flm_rig, tmp_path):
    flm_rig["mass"] = 1 + 2e-5
    journal = smoke.run_flm_smoke("flm-gemma3-4b", NPU_ENDPOINT, tmp_path / "flm.json", execute=True,
                                  observe=flm_metadata)
    row = journal["attempts"][0]
    assert row["status"] == "ineligible" and row["evidence_reason"] == "alternative_mass_exceeds_one"


def test_flm_smoke_refuses_when_the_model_is_not_served(flm_rig, tmp_path):
    journal = smoke.run_flm_smoke("flm-gemma3-4b", NPU_ENDPOINT, tmp_path / "flm.json", execute=True,
                                  observe=lambda: flm_metadata(loaded=False))
    assert journal["attempts"][0]["reason"] == "flm_model_not_served_or_identity_unknown"
    assert flm_rig["posts"] == []


# --- gateway tag synthesis (PowerShell; Windows host only) ----------------------------------------


def _pwsh(*args: str) -> subprocess.CompletedProcess[str]:
    """Run PowerShell 7; retry once when the Store app-execution alias refuses the launch.

    On hosts where ``pwsh`` resolves to the WindowsApps alias, CreateProcess intermittently
    fails with ``PermissionError`` (WinError 5). A real permission problem fails twice.
    """
    for attempt in (1, 2):
        try:
            return subprocess.run(["pwsh", "-NoProfile", *args], capture_output=True, text=True, timeout=120)
        except PermissionError:
            if attempt == 2:
                raise
            time.sleep(1.0)
    raise AssertionError("unreachable")


@pytest.mark.skipif(sys.platform != "win32" or shutil.which("pwsh") is None, reason="needs Windows PowerShell 7")
@pytest.mark.skipif(not (ROOT / "models/flm").is_dir(), reason="needs the untracked FLM model store at models/flm (host only)")
def test_gateway_synthesizes_installed_only_tags_with_catalog_digest(tmp_path):
    fake = tmp_path / "flm"
    fake.mkdir()
    listing = {"models": [
        {"name": "gemma3:4b", "model": "gemma3:4b", "installed": True, "size": 4, "footprint": 4.4,
         "default_context_length": 65536, "details": {"quantization_level": "Q4_1"}},
        {"name": "qwen3:8b", "model": "qwen3:8b", "installed": False, "size": 8, "footprint": 5.0,
         "default_context_length": 32768, "details": {}}]}
    (fake / "listing.json").write_text(json.dumps(listing), encoding="utf-8")
    (fake / "flm.cmd").write_text(f'@type "{fake / "listing.json"}"\r\n', encoding="utf-8")
    (fake / "model_info.json").write_text(json.dumps({"gemma3:4b": [
        {"path": "config.json"}, {"path": "model.q4nx", "lfs": {"oid": DIGEST.removeprefix("sha256:")}}]}),
        encoding="utf-8")
    script = ROOT / "scripts/hosts/serve_local_flm.ps1"
    result = _pwsh("-File", str(script), "tags", "gemma3:4b", "-FlmPath", str(fake / "flm.cmd"))
    assert result.returncode == 0, result.stderr
    tags = json.loads(result.stdout)
    assert tags["source"] == FLM_INSTALLED_TAGS_SOURCE and tags["runtime"] == "fastflowlm"
    assert [m["name"] for m in tags["models"]] == ["gemma3:4b"] and tags["models"][0]["digest"] == DIGEST
    assert tags["serving"] == {"model": "gemma3:4b", "context_length": 2048,
                               "source": "flm serve --ctx-len (process start)"}
    refused = _pwsh("-File", str(script), "tags", "qwen3:8b", "-FlmPath", str(fake / "flm.cmd"))
    assert refused.returncode != 0 and "not installed" in (refused.stdout + refused.stderr)
