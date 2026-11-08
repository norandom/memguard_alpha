"""--host-snapshot and --endpoint on the probe, smoke and adopter example (release-1.0 M1.7b). Offline."""

import json
import runpy
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import requests

from recall_guard.core.profiles import HostMemorySnapshot
from scripts import probe_local_model as probe
from scripts import smoke_local_models as smoke

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "tests/fixtures/local_provider/candidate-manifest.json"
DOCKER_HOST = "http://host.docker.internal:11434/v1/chat/completions"
GIB = 1024 ** 3


@pytest.fixture(autouse=True)
def no_http(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("no HTTP in flag tests")

    monkeypatch.setattr(requests.Session, "request", forbidden)


def write_snapshot(tmp_path, available=24 * GIB, age=timedelta(minutes=1)):
    path = tmp_path / "host-memory.json"
    path.write_text(json.dumps({"available_bytes": available, "total_bytes": 32 * GIB,
                                "observed_at": (datetime.now(UTC) - age).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                "host": "MINI", "source": "windows_host_snapshot"}), encoding="utf-8")
    return path


def test_probe_flags_pass_snapshot_and_endpoint(tmp_path, monkeypatch):
    seen = {}

    def run_probe(manifest, candidate, host, journal, **kwargs):
        seen.update(kwargs)
        return {"status": "dry_run"}

    monkeypatch.setattr(probe, "run_probe", run_probe)
    path = write_snapshot(tmp_path)
    assert probe.main(["--candidate", "ollama-gpt-oss-20b", "--dry-run", "--endpoint", DOCKER_HOST,
                       "--host-snapshot", str(path)]) == 0
    assert seen["endpoint"] == DOCKER_HOST and seen["host_snapshot"] == HostMemorySnapshot.from_file(path)


def test_probe_endpoint_override_and_snapshot_reach_preflight(tmp_path, monkeypatch):
    seen = []

    def host_preflight(config, host_snapshot=None):
        seen.append((config.endpoint, host_snapshot))
        return probe.HostPreflightReport("ollama-gpt-oss-20b", config.model, passed=False,
                                         reason="host_memory_stale", recommendation="stale")

    monkeypatch.setattr(probe, "host_preflight", host_preflight)
    snapshot = HostMemorySnapshot.from_file(write_snapshot(tmp_path))
    with pytest.raises(probe.PreflightError):
        probe.run_probe(MANIFEST, "ollama-gpt-oss-20b", tmp_path / "host.json", tmp_path / "journal.json",
                        endpoint=DOCKER_HOST, host_snapshot=snapshot)
    assert seen == [(DOCKER_HOST, snapshot)] and not (tmp_path / "journal.json").exists()
    with pytest.raises(ValueError, match="explicit_full_chat_endpoint_required"):
        probe.run_probe(MANIFEST, "ollama-gpt-oss-20b", tmp_path / "h", tmp_path / "j", dry_run=True,
                        endpoint="http://host.docker.internal:11434/v1")


def test_probe_unreadable_snapshot_is_refused_plainly(tmp_path, capsys):
    assert probe.main(["--candidate", "ollama-gpt-oss-20b", "--dry-run",
                       "--host-snapshot", str(tmp_path / "missing.json")]) == 2
    assert "Cannot read host memory snapshot missing.json" in capsys.readouterr().err


def test_smoke_stale_snapshot_blocks_in_preflight_with_panel(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("RECALL_GUARD_IN_CONTAINER", "1")
    path = write_snapshot(tmp_path, age=timedelta(hours=1))
    assert smoke.main(["--profile", "ollama-gpt-oss-20b", "--endpoint", DOCKER_HOST,
                       "--host-snapshot", str(path)]) == 2
    err = capsys.readouterr().err
    assert "too old to trust" in err and "Take a fresh snapshot" in err and "Traceback" not in err


def test_smoke_container_without_snapshot_is_unknown_and_reads_no_memory(monkeypatch, capsys):
    monkeypatch.setenv("RECALL_GUARD_IN_CONTAINER", "1")
    monkeypatch.setattr(smoke, "available_memory_bytes", lambda: pytest.fail("read container memory"))
    gate = smoke.memory_gate("http://127.0.0.1:11434/v1/chat/completions", 16 * GIB)
    assert gate["available_bytes"] is None and not gate["admitted"] and gate["source"] == "unknown_container_host"
    assert smoke.main(["--profile", "ollama-gpt-oss-20b"]) == 2
    assert "Free memory on the model host is unknown" in capsys.readouterr().err


def test_smoke_memory_gate_uses_fresh_snapshot_only(tmp_path):
    fresh = HostMemorySnapshot.from_file(write_snapshot(tmp_path, available=20 * GIB))
    gate = smoke.memory_gate(DOCKER_HOST, 16 * GIB, host_snapshot=fresh)
    assert gate["admitted"] and gate["source"] == "windows_host_snapshot" and gate["available_bytes"] == 20 * GIB
    stale = HostMemorySnapshot.from_file(write_snapshot(tmp_path, available=20 * GIB, age=timedelta(hours=2)))
    gate = smoke.memory_gate(DOCKER_HOST, 16 * GIB, host_snapshot=stale)
    assert not gate["admitted"] and gate["source"] == "host_memory_stale"


@pytest.fixture
def example():
    return runpy.run_path(str(ROOT / "examples" / "adopter_flow.py"))


@pytest.mark.parametrize("available,code,passed", [(24 * GIB, 0, True), (13 * GIB, 2, False)])
def test_adopter_dry_run_evaluates_snapshot_offline(example, tmp_path, capsys, available, code, passed):
    path = write_snapshot(tmp_path, available=available)
    assert example["main"](["--endpoint", DOCKER_HOST, "--host-snapshot", str(path)]) == code
    captured = capsys.readouterr()
    record = json.loads(captured.out)
    assert record["endpoint"] == DOCKER_HOST and record["network_requests"] == 0
    assert record["host_memory"]["ram_gate_passed"] is passed
    assert record["host_memory"]["snapshot"]["host"] == "MINI"
    if not passed:
        assert record["host_memory"]["reason"] == "insufficient_free_ram" and "Not enough free memory" in captured.err


def test_adopter_flags_rejected_with_fixture(example):
    with pytest.raises(SystemExit):
        example["main"](["--fixture", "--endpoint", DOCKER_HOST])
