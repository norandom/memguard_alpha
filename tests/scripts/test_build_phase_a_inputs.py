"""Offline Phase A input builder and the local runner's gateway/host-gate wiring."""
import json
from hashlib import sha256
from pathlib import Path

import pytest

from recall_guard.harness.local_evaluation import load_plan
from scripts import build_phase_a_inputs as build
from scripts import measure_local_models as cli

ROOT = Path(__file__).resolve().parents[2]
ENDPOINT = "http://172.17.32.1:11435/v1/chat/completions"


def inputs(directory, profile):
    return tuple(directory / profile / name for name in ("plan.json", "manifest.json", "corpus.json"))


@pytest.fixture
def local_output(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    assert build.main(["--output", str(tmp_path), "--profiles", "ollama-gemma3-4b",
                       "--local-endpoint", ENDPOINT]) == 0
    return tmp_path


def test_local_plan_is_phase_a_with_bounded_tokens_and_labels(local_output):
    plan = load_plan(*inputs(local_output, "ollama-gemma3-4b"))
    assert plan.logical_requests == plan.physical_requests == plan.limits.max_requests == 321
    assert plan.ensemble_draws == 0 and (plan.seed, plan.min_valid, plan.min_auc) == (0, 50, 0.6)
    assert plan.config.endpoint == ENDPOINT and plan.config.credential_ref == "env:LOCAL_MODEL_API_KEY"
    assert plan.authentication == "environment" and plan.config.model == "gemma3:4b"
    assert {k: plan.config.parameters[k] for k in ("max_tokens", "temperature", "top_p", "top_logprobs")} == {
        "max_tokens": 32, "temperature": 0, "top_p": 1, "top_logprobs": 5}
    assert plan.config.max_retries == 0
    limits = plan.limits
    assert limits.context_tokens == 2048 and limits.max_input_tokens + limits.max_output_tokens <= 2048
    assert limits.max_total_tokens == 321 * (limits.max_input_tokens + 32)
    held = [item for item in plan.items if item.split == "held_out"]
    assert len(held) == 80 and sorted(item.label for item in held) == [0] * 40 + [1] * 40
    corpus = json.loads((local_output / "ollama-gemma3-4b/corpus.json").read_text(encoding="utf-8"))
    first = json.loads((ROOT / "data/edgar/calibration/is_memorized.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert corpus["is_memorized"][0]["item_id"] == sha256(first["metadata"]["url"].encode()).hexdigest()
    # Declared count = UTF-8 bytes + template allowance: an upper bound, never an estimate.
    assert corpus["is_memorized"][0]["input_tokens"] == len(first["prompt"].encode()) + 128


def test_index_hashes_files_and_counts_complete_profiles(local_output):
    index = json.loads((local_output / "index.json").read_text(encoding="utf-8"))
    assert index["request_totals"] == {"local": 321, "modal": 0, "npu": 0}
    assert "modal-gpt-oss-20b" in index["missing_profiles"]
    for name, digest in index["files"].items():
        assert sha256((local_output / name).read_bytes()).hexdigest() == digest


def test_rebuild_is_byte_stable(local_output, tmp_path):
    other = tmp_path / "again"
    assert build.main(["--output", str(other), "--profiles", "ollama-gemma3-4b", "--local-endpoint", ENDPOINT]) == 0
    for path in inputs(local_output, "ollama-gemma3-4b"):
        assert path.read_bytes() == (other / "ollama-gemma3-4b" / path.name).read_bytes()


def test_local_profiles_require_gateway_endpoint(tmp_path):
    with pytest.raises(SystemExit):
        build.main(["--output", str(tmp_path), "--profiles", "local"])


def test_modal_plan_passes_measure_modal_dry_run(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    endpoint = json.loads(build.MODAL_MANIFEST.read_text(encoding="utf-8"))["target"]["endpoint"]
    monkeypatch.setattr(build, "modal_token_counter",
                        lambda root: (lambda prompt: len(prompt) // 3, endpoint, {"tokenizer.json": "x"}))
    assert build.main(["--output", str(tmp_path), "--profiles", "modal"]) == 0
    from scripts import measure_modal_model as measure
    monkeypatch.setattr(measure, "tokenizer_hashes", lambda plan, root: {"tokenizer.json": "x"})
    profile = measure.load_profile(measure.PROFILE_ID)
    monkeypatch.setattr(type(profile.harmony), "resolve_path", lambda self: Path(__file__))
    frozen = measure.measurement_planning(*inputs(tmp_path, "modal-gpt-oss-20b"))
    assert frozen["plan_blockers"] == []
    assert frozen["logical_requests"] == frozen["physical_requests"] == 321
    assert frozen["items_per_split"] == {"is_memorized": 40, "oos_control": 100, "held_out": 80}
    assert frozen["cost_estimate"]["worst_case_total_usd"] <= 2
    index = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    assert index["request_totals"] == {"local": 0, "modal": 321, "npu": 0}


def test_observer_authenticates_metadata_reads(local_output, monkeypatch):
    plan = load_plan(*inputs(local_output, "ollama-gemma3-4b"))
    seen = {}

    def observe(endpoint, model, headers=None, **_):
        seen.update(endpoint=endpoint, model=model, headers=headers)
        return {"model_installed": True, "runtime_version": "0.1", "digest": "d", "observed_backend": "CPU",
                "effective_context_tokens": 2048, "loaded_models": [], "model_allocation_bytes": None,
                "accelerator_allocation_bytes": None}
    monkeypatch.setattr("scripts.probe_local_model.observe_ollama_host", observe)
    monkeypatch.setenv("LOCAL_MODEL_API_KEY", "test-token")
    observation = cli.ollama_observer(plan)()
    assert observation.provenance.context_length == 2048
    assert seen["headers"] == {"Authorization": "Bearer test-token"} and seen["endpoint"] == ENDPOINT


def test_host_gate_refusal_precedes_journal(local_output, tmp_path, monkeypatch, capsys):
    snapshots = []

    def preflight(config, snapshot):
        snapshots.append((config.credential_ref, snapshot.available_bytes))
        return type("Report", (), {"passed": False, "reason": "insufficient_free_ram"})()
    monkeypatch.setattr("scripts.probe_local_model.host_preflight", preflight)
    snapshot = tmp_path / "host.json"
    from datetime import UTC, datetime
    snapshot.write_text(json.dumps({"available_bytes": 1, "total_bytes": 2, "host": "h", "source": "windows_host_snapshot",
                                    "observed_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")}))
    journal = tmp_path / "journal.json"
    args = [*sum((["--" + n, str(p)] for n, p in zip(("plan", "manifest", "corpus"),
                                                       inputs(local_output, "ollama-gemma3-4b"), strict=True)), []),
            "--execute", "--journal", str(journal), "--observer", "ollama", "--host-snapshot", str(snapshot)]
    assert cli.main(args) == 2
    assert "insufficient_free_ram" in capsys.readouterr().err
    assert snapshots == [("env:LOCAL_MODEL_API_KEY", 1)] and not journal.exists()


def test_modal_plan_freezes_low_reasoning_effort_and_1024_output_tokens(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    endpoint = json.loads(build.MODAL_MANIFEST.read_text(encoding="utf-8"))["target"]["endpoint"]
    monkeypatch.setattr(build, "modal_token_counter",
                        lambda root: (lambda prompt: len(prompt) // 3, endpoint, {"tokenizer.json": "x"}))
    assert build.main(["--output", str(tmp_path / "r2"), "--profiles", "modal"]) == 0
    manifest = json.loads((tmp_path / "r2/modal-gpt-oss-20b/manifest.json").read_text(encoding="utf-8"))
    parameters = manifest["config"]["parameters"]
    assert parameters["reasoning_effort"] == "low" and parameters["max_tokens"] == 1024
    limits = json.loads((tmp_path / "r2/modal-gpt-oss-20b/plan.json").read_text(encoding="utf-8"))["limits"]
    assert limits["max_output_tokens"] == 1024
    assert limits["max_input_tokens"] + limits["max_output_tokens"] <= limits["context_tokens"] == 4096


def test_modal_build_refuses_inputs_that_exceed_the_context(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    endpoint = json.loads(build.MODAL_MANIFEST.read_text(encoding="utf-8"))["target"]["endpoint"]
    monkeypatch.setattr(build, "modal_token_counter",
                        lambda root: (lambda prompt: 3100, endpoint, {"tokenizer.json": "x"}))
    with pytest.raises(ValueError, match="context_exceeded"):
        build.main(["--output", str(tmp_path), "--profiles", "modal"])
