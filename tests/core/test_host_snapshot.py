"""Container-aware host preflight and timestamped host memory snapshots (release-1.0 M1.7a, C20, C21)."""

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
import requests

from recall_guard.core import profiles
from recall_guard.core.profiles import (
    SNAPSHOT_CLOCK_SKEW,
    SNAPSHOT_MAX_AGE,
    HostMemorySnapshot,
    PreflightError,
    ProfileError,
    load_profile,
    preflight_host,
    running_in_container,
)
from recall_guard.harness.request_failures import describe_failure

GIB = 1024 ** 3
NOW = datetime(2026, 11, 8, 12, 0, tzinfo=UTC)
LOOPBACK = "http://127.0.0.1:11434/v1/chat/completions"
DOCKER_HOST = "http://host.docker.internal:11434/v1/chat/completions"
#: Same shape as scripts/host_memory_snapshot.ps1 output.
PS1_OUTPUT = {"available_bytes": 24 * GIB, "total_bytes": 32 * GIB, "observed_at": "2026-11-08T11:58:00Z",
              "host": "MINI", "source": "windows_host_snapshot"}


@pytest.fixture(autouse=True)
def no_http_or_psutil(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("no real HTTP or memory reads in these tests")

    monkeypatch.setattr(requests.Session, "request", forbidden)
    monkeypatch.setattr(profiles, "_psutil_memory", forbidden)


class FakeOllama:
    def __init__(self, running=()):
        self.calls, self.running = [], list(running)

    def __call__(self, url):
        self.calls.append(url)
        if url.endswith("/api/version"):
            return {"version": "0.12.3"}
        if url.endswith("/api/tags"):
            return {"models": [{"name": "gpt-oss:20b", "digest": "a" * 64}]}
        return {"models": self.running}


def gpt_oss(endpoint=LOOPBACK):
    profile = load_profile("ollama-gpt-oss-20b")
    return replace(profile, model_config=replace(profile.model_config, endpoint=endpoint))


def snapshot(available=24 * GIB, age=timedelta(minutes=2), **overrides):
    return HostMemorySnapshot(available, 32 * GIB, (NOW - age).isoformat(), host="MINI", **overrides)


# --- C20: container detection -------------------------------------------------

@pytest.mark.parametrize("value,expected", [("1", True), ("true", True), ("0", False), ("no", False)])
def test_container_override_wins(tmp_path, value, expected):
    (tmp_path / ".dockerenv").write_text("")
    assert running_in_container(environment={"RECALL_GUARD_IN_CONTAINER": value}, root=tmp_path) is expected


@pytest.mark.parametrize("marker", [".dockerenv", "run/.containerenv"])
def test_container_marker_files(tmp_path, marker):
    assert running_in_container(environment={}, root=tmp_path) is False
    (tmp_path / marker).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / marker).write_text("")
    assert running_in_container(environment={}, root=tmp_path) is True


def test_container_cgroup_marker(tmp_path):
    cgroup = tmp_path / "proc" / "1" / "cgroup"
    cgroup.parent.mkdir(parents=True)
    cgroup.write_text("0::/init.scope\n")
    assert running_in_container(environment={}, root=tmp_path) is False
    cgroup.write_text("12:memory:/docker/abc123\n")
    assert running_in_container(environment={}, root=tmp_path) is True


def test_container_loopback_without_snapshot_is_unknown_and_never_reads_memory():
    reads, http = [], FakeOllama()
    report = preflight_host(gpt_oss(), environment={}, memory_reader=lambda: reads.append(1) or (64 * GIB, None),
                            http_get=http, in_container=True)
    assert not report.passed and report.reason == "host_memory_unknown"
    assert reads == [] and http.calls == [] and report.measured_free_ram_bytes is None
    assert "running in a container" in report.recommendation
    assert "scripts/host_memory_snapshot.ps1" in report.recommendation
    assert "--host-snapshot /outputs/host-memory.json" in report.recommendation


def test_container_detection_defaults_from_environment(monkeypatch):
    monkeypatch.setenv("RECALL_GUARD_IN_CONTAINER", "1")
    report = preflight_host(gpt_oss(), environment={}, memory_reader=lambda: pytest.fail("read local memory"))
    assert report.reason == "host_memory_unknown"


def test_container_with_fresh_snapshot_uses_it():
    http = FakeOllama()
    report = preflight_host(gpt_oss(DOCKER_HOST), environment={}, host_snapshot=snapshot(), http_get=http,
                            in_container=True, now=NOW)
    assert report.passed and report.memory_source == "host_snapshot" and report.ram_gate == "enforced"
    assert report.measured_free_ram_bytes == 24 * GIB and report.total_ram_bytes == 32 * GIB


# --- C21: freshness ----------------------------------------------------------

def test_fresh_low_snapshot_blocks_against_the_requirement():
    http = FakeOllama()
    report = preflight_host(gpt_oss(DOCKER_HOST), environment={}, host_snapshot=snapshot(13 * GIB),
                            http_get=http, in_container=True, now=NOW)
    assert not report.passed and report.reason == "insufficient_free_ram"
    assert http.calls == ["http://host.docker.internal:11434/api/ps"]  # One read-only check, nothing loaded.


def test_snapshot_keeps_already_loaded_exception():
    http = FakeOllama(running=[{"name": "gpt-oss:20b", "size": 14 * GIB, "size_vram": 0}])
    report = preflight_host(gpt_oss(DOCKER_HOST), environment={}, host_snapshot=snapshot(2 * GIB),
                            http_get=http, in_container=True, now=NOW)
    assert report.passed and report.ram_gate == "skipped_model_already_loaded"


@pytest.mark.parametrize("age,words", [(SNAPSHOT_MAX_AGE + timedelta(seconds=1), "old"),
                                       (-SNAPSHOT_CLOCK_SKEW - timedelta(seconds=1), "in the future")])
def test_stale_or_future_snapshot_is_refused_without_http(age, words):
    http = FakeOllama()
    report = preflight_host(gpt_oss(DOCKER_HOST), environment={}, host_snapshot=snapshot(64 * GIB, age=age),
                            http_get=http, in_container=True, now=NOW)
    assert not report.passed and report.reason == "host_memory_stale" and http.calls == []
    assert report.measured_free_ram_bytes is None and words in report.recommendation
    assert "MINI" in report.recommendation and "Or use --profile modal-gpt-oss-20b" in report.recommendation
    presentation = describe_failure(PreflightError(report))
    assert presentation.title == "The host memory snapshot is too old to trust"
    assert presentation.next_steps[0].startswith("Take a fresh snapshot on the model host")
    assert "--profile" not in presentation.explanation and "On the model host" not in presentation.explanation


def test_small_clock_skew_and_custom_max_age():
    ahead = snapshot(age=-timedelta(seconds=30))
    assert preflight_host(gpt_oss(DOCKER_HOST), environment={}, host_snapshot=ahead, http_get=FakeOllama(),
                          in_container=True, now=NOW).passed
    old = snapshot(age=timedelta(minutes=30))
    assert preflight_host(gpt_oss(DOCKER_HOST), environment={}, host_snapshot=old, http_get=FakeOllama(),
                          in_container=True, now=NOW, max_snapshot_age=timedelta(hours=1)).passed
    with pytest.raises(ValueError):
        preflight_host(gpt_oss(), environment={}, host_snapshot=old, max_snapshot_age=timedelta(0))


def test_snapshot_without_observed_at_is_unknown():
    report = preflight_host(gpt_oss(DOCKER_HOST), environment={}, host_snapshot={"available_bytes": 64 * GIB},
                            http_get=FakeOllama(), in_container=True, now=NOW)
    assert report.reason == "host_memory_unknown" and "no observed_at" in report.recommendation
    assert describe_failure(PreflightError(report)).title == "Free memory on the model host is unknown"


@pytest.mark.parametrize("bad", ["yesterday", "2026-11-08T12:00:00", 17])
def test_invalid_observed_at_is_rejected(bad):
    with pytest.raises(ProfileError, match="observed_at"):
        HostMemorySnapshot(GIB, observed_at=bad)


def test_snapshot_file_round_trip_matches_ps1_shape(tmp_path):
    path = tmp_path / "host-memory.json"
    # PowerShell 5.1 may write a BOM; the loader tolerates it.
    path.write_text(json.dumps(PS1_OUTPUT, indent=4), encoding="utf-8-sig")
    loaded = HostMemorySnapshot.from_file(path)
    assert loaded.to_dict() == PS1_OUTPUT
    assert loaded.observed_datetime == datetime(2026, 11, 8, 11, 58, tzinfo=UTC)
    assert HostMemorySnapshot.coerce(loaded.to_dict()) == loaded
    assert preflight_host(gpt_oss(DOCKER_HOST), environment={}, host_snapshot=loaded, http_get=FakeOllama(),
                          in_container=True, now=NOW).passed
    (tmp_path / "broken.json").write_text("{", encoding="utf-8")
    with pytest.raises(ProfileError, match="Cannot read host memory snapshot broken.json"):
        HostMemorySnapshot.from_file(tmp_path / "broken.json")
