"""Transport amendment tests use synthetic HTTP only, never live credentials."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import requests

from scripts import prepare_completion_latency as latency
from scripts import prepare_completion_pilot as pilot

DATA = Path(__file__).resolve().parents[2] / "data/experiments/2026-10-31-completion-api"
ENV = {"FIXTURE_NIM": "synthetic-nim-private", "FIXTURE_ANTHROPIC": "synthetic-opus-private"}


def nim(*, evidence=True, content="OK"):
    choice = {"message": {"content": content}, "finish_reason": "stop"}
    if evidence:
        choice["logprobs"] = {"content": [{"token": "OK", "logprob": -0.1,
                                          "top_logprobs": [{"token": "OK", "logprob": -0.1}]}]}
    return 200, {"model": "openai/gpt-oss-20b", "choices": [choice],
                 "usage": {"prompt_tokens": 6, "completion_tokens": 1}}


def anthropic():
    return 200, {"model": "claude-opus-5-5", "content": [{"type": "text", "text": "OK"}],
                 "stop_reason": "end_turn", "usage": {"input_tokens": 6, "output_tokens": 1}}


@pytest.fixture
def wire(monkeypatch):
    pending, calls = [], []

    def send(adapter, prepared, **kwargs):
        calls.append((prepared, kwargs))
        assert pending, "Unexpected synthetic request"
        entry = pending.pop(0)
        if isinstance(entry, BaseException):
            raise entry
        response = requests.Response()
        response.status_code, body = entry
        response._content = json.dumps(body).encode()
        response.request, response.url = prepared, prepared.url
        return response

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)
    return pending, calls


@pytest.fixture
def directory(tmp_path, wire):
    result = tmp_path / "inputs"
    shutil.copytree(DATA, result, ignore=shutil.ignore_patterns("study-plan.json", "eligibility"))
    pilot.freeze(result)
    wire[0].append(requests.Timeout("synthetic timeout"))
    pilot.probe(result, nim_credential_env="FIXTURE_NIM", anthropic_credential_env="FIXTURE_ANTHROPIC", environment=ENV)
    wire[0].extend([anthropic(), requests.Timeout("synthetic timeout")])
    pilot.recover(result, nim_credential_env="FIXTURE_NIM", anthropic_credential_env="FIXTURE_ANTHROPIC", environment=ENV)
    assert len(wire[1]) == 3
    return result


def run_probe(directory, environment=None):
    return latency.probe(directory, nim_credential_env="FIXTURE_NIM",
                         environment={"FIXTURE_NIM": ENV["FIXTURE_NIM"]} if environment is None else environment)


def saved(directory):
    return {path.relative_to(directory).as_posix(): path.read_bytes() for path in directory.rglob("*.json")}


def change(path, mutate):
    value = json.loads(path.read_bytes())
    mutate(value)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_freeze_is_offline_transport_only_bound_to_original_bytes(directory, wire):
    before = saved(directory)
    amendment = latency.freeze(directory)
    assert amendment == latency.validate(directory) == latency.freeze(directory)
    assert amendment["status"] == "frozen_transport_amendment_only"
    assert amendment["post_observation"] is True
    assert amendment["preregistered"] is False
    assert amendment["change"] == {"provider": "nim", "field": "timeout_s", "previous": 90, "amended": 600}
    config = amendment["config"]
    assert config == pilot.model_configs()["nim"].to_dict() | {"timeout_s": 600}
    assert amendment["budget"] == {"study_cap": 512, "phase_caps": pilot.PHASE_CAPS,
                                   "eligibility_used": 3, "eligibility_remaining": 1, "new_call_limit": 1}
    for path, content in before.items():
        assert (directory / path).read_bytes() == content
    assert amendment["sources"] == {path: hashlib.sha256(content).hexdigest() for path, content in before.items()
                                    if path == "study-plan.json" or path.startswith("eligibility/")}
    assert len(wire[1]) == 3
    assert not (directory / "eligibility/latency-01").exists()


def test_one_probe_only_nim_reference_and_preserves_prior_events(directory, wire):
    amendment = latency.freeze(directory)
    before = saved(directory)
    prior = pilot.verify_receipt(directory)

    class OnlyNim:
        def get(self, name):
            assert name == "FIXTURE_NIM"
            return ENV[name]

    wire[0].append(nim())
    result = run_probe(directory, OnlyNim())
    assert result["status"] == "eligible" and result["failure"] is None
    assert result["budget"] == {"limit": 4, "used": 4, "remaining": 0}
    assert result["events"][:3] == prior["events"]
    event = result["events"][3]
    assert (event["call_id"], event["source_attempt"], event["source_call_id"], event["attempt"]) == (4, "latency-01", 1, 1)
    assert event["stage"] == "eligibility/latency-01/probe/nim" and event["status"] == "succeeded"
    assert result["providers"]["anthropic"] == prior["providers"]["anthropic"]
    assert result["providers"]["nim"]["config"] == amendment["config"]
    assert result["remaining_phase_caps"] == {"calibration": 220, "nim_evaluation": 144, "anthropic_evaluation": 144}
    assert result["cost"] is result["balance"] is None
    assert latency.verify_receipt(directory) == result
    assert len(wire[1]) == 4
    prepared, options = wire[1][-1]
    assert options["timeout"] == 600
    assert json.loads(prepared.body) == {"model": "openai/gpt-oss-20b",
                                       "messages": [{"role": "user", "content": pilot.PROBE_PROMPT}],
                                       **amendment["config"]["parameters"]}
    assert all((directory / path).read_bytes() == content for path, content in before.items())
    with pytest.raises(FileExistsError):
        run_probe(directory, OnlyNim())
    assert len(wire[1]) == 4


@pytest.mark.parametrize("reply,category,code,status", [
    (requests.Timeout(ENV["FIXTURE_NIM"]), "request_failure", "timeout", None),
    ((401, {"error": {"message": ENV["FIXTURE_NIM"], "code": "invalid_api_key"}}), "authentication", "invalid_api_key", 401),
    ((402, {"error": {"message": ENV["FIXTURE_NIM"], "code": "insufficient_quota"}}), "billing", "insufficient_quota", 402),
    ((400, {"error": {"message": ENV["FIXTURE_NIM"], "param": "top_logprobs"}}), "parameter", None, 400),
    (nim(evidence=False), "capability", None, None),
    (nim(content=""), "capability", None, None),
])
def test_failures_preserve_known_diagnostics_and_no_retries(directory, wire, reply, category, code, status):
    latency.freeze(directory)
    wire[0].append(reply)
    result = run_probe(directory)
    assert result["status"] == "failed" and result["failure"] == {"phase": "eligibility", "provider": "nim", "category": category}
    assert result["events"][3]["code"] == code and result["events"][3]["status_code"] == status
    assert result["budget"] == {"limit": 4, "used": 4, "remaining": 0}
    assert latency.verify_receipt(directory) == result
    assert len(wire[1]) == 4
    content = b"".join(p.read_bytes() for p in (directory / "eligibility/latency-01").glob("*.json"))
    assert ENV["FIXTURE_NIM"].encode() not in content and b"FIXTURE_NIM" not in content


def test_missing_key_is_terminal_zero_new_calls(directory, wire):
    latency.freeze(directory)
    result = run_probe(directory, {})
    assert result["status"] == "failed" and result["failure"]["category"] == "missing_credential"
    assert result["budget"] == {"limit": 4, "used": 3, "remaining": 1}
    assert latency.verify_receipt(directory) == result
    with pytest.raises(FileExistsError):
        run_probe(directory)
    assert len(wire[1]) == 3


def test_exclusive_intent_before_post_and_crash_never_replayed(directory, wire, monkeypatch):
    latency.freeze(directory)

    def crash(*args, **kwargs):
        intent = json.loads((directory / "eligibility/latency-01/intent.json").read_bytes())
        assert intent["physical_calls_before"] == 3 and intent["provider"] == "nim"
        assert intent["amendment_sha256"] == hashlib.sha256((directory / "eligibility/latency-amendment.json").read_bytes()).hexdigest()
        raise KeyboardInterrupt

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", crash)
    with pytest.raises(KeyboardInterrupt):
        run_probe(directory)
    with pytest.raises(FileExistsError):
        run_probe(directory)
    with pytest.raises(ValueError, match="incomplete|ambiguous"):
        latency.verify_receipt(directory)


@pytest.mark.parametrize("mutation", [
    lambda x: x["config"].update(timeout_s=90),
    lambda x: x["config"].update(model="another-model"),
    lambda x: x["config"]["parameters"].update(max_tokens=64),
    lambda x: x["budget"]["phase_caps"].update(calibration=221),
    lambda x: x.update(post_observation=False),
    lambda x: x.update(credential_ref="SECRET"),
    lambda x: x["sources"].update({"eligibility/recovery-01/receipt.json": "0" * 64}),
])
def test_amendment_tamper_stops_before_environment(directory, wire, mutation):
    latency.freeze(directory)
    change(directory / "eligibility/latency-amendment.json", mutation)
    class NoEnvironment:
        def get(self, name):
            raise AssertionError("Rejected configuration read environment")
    with pytest.raises(ValueError):
        run_probe(directory, NoEnvironment())
    assert len(wire[1]) == 3


@pytest.mark.parametrize("mutation", [
    lambda x: x["events"][2].update(code="connection_error"),
    lambda x: x["events"][2].update(attempt=2),
    lambda x: x["events"][2].update(status="succeeded"),
    lambda x: x["events"][2].update(latency_s=float("nan")),
    lambda x: x["providers"]["anthropic"]["capabilities"]["text_generation"].update(observed=False),
    lambda x: x["providers"]["anthropic"].update(raw_response={"headers": {"Authorization": "SECRET"}}),
    lambda x: x.update(credential_ref="SECRET"),
    lambda x: x.update(status="eligible"),
    lambda x: x["budget"].update(used=4, remaining=0),
])
def test_invalid_source_rejected_before_freeze(directory, mutation):
    for name in ("receipt.json", "02-nim.json"):
        change(directory / "eligibility/recovery-01" / name, mutation)
    with pytest.raises(ValueError):
        latency.freeze(directory)
    assert not (directory / "eligibility/latency-amendment.json").exists()


@pytest.mark.parametrize("mutation", [
    lambda x: x["events"][0].update(latency_s=1),
    lambda x: x["events"][3].update(source_call_id=2),
    lambda x: x["events"][3].update(status="started"),
    lambda x: x["events"][3].update(stage="eligibility/probe/nim"),
    lambda x: x["events"][3].update(attempt=2),
    lambda x: x["events"][3].update(cost=0),
    lambda x: x["events"][3].update(latency_s=float("inf")),
    lambda x: x["events"][3].update(timestamp="unknown"),
    lambda x: x["providers"]["nim"]["capabilities"].pop("top_logprobs"),
    lambda x: x["providers"]["nim"]["capabilities"]["logprobs"].update(observed=False),
    lambda x: x["providers"]["nim"]["config"].update(timeout_s=90),
    lambda x: x.update(credential_ref="SECRET"),
    lambda x: x.update(balance=0),
    lambda x: x.update(status="unknown"),
    lambda x: x["budget"].update(remaining=1),
])
def test_new_receipt_strictly_rejects_tampering(directory, wire, mutation):
    latency.freeze(directory)
    wire[0].append(nim())
    run_probe(directory)
    for name in ("receipt.json", "01-nim.json"):
        change(directory / "eligibility/latency-01" / name, mutation)
    with pytest.raises(ValueError):
        latency.verify_receipt(directory)


def test_single_budget_blocks_underlying_second_call(directory, wire, monkeypatch):
    latency.freeze(directory)
    wire[0].append(nim())
    def excessive(self, prompt, *, stage):
        self.generate_for_stage(prompt, stage=stage)
        self.generate_for_stage(prompt, stage=stage)
    monkeypatch.setattr(pilot.OpenAICompatibleClient, "probe", excessive)
    result = run_probe(directory)
    assert result["failure"]["category"] == "budget_exhausted"
    assert len(wire[1]) == 4 and result["budget"]["used"] == 4
    assert latency.verify_receipt(directory) == result


def test_source_bytes_changed_after_freeze_are_rejected(directory, wire):
    latency.freeze(directory)
    path = directory / "eligibility/recovery-01/receipt.json"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError):
        run_probe(directory)
    assert len(wire[1]) == 3
    assert not (directory / "eligibility/latency-01").exists()


def test_secret_diagnostics_are_redacted_at_final_boundary(directory, wire):
    latency.freeze(directory)
    wire[0].append((400, {"error": {"message": ENV["FIXTURE_NIM"], "code": ENV["FIXTURE_NIM"],
                                    "param": ENV["FIXTURE_NIM"], "detail": {"Authorization": ENV["FIXTURE_NIM"]}}}))
    result = run_probe(directory)
    assert result["events"][3]["code"] == result["events"][3]["parameter"] == "[REDACTED]"
    assert latency.verify_receipt(directory) == result
    for path in (directory / "eligibility/latency-01").glob("*.json"):
        assert ENV["FIXTURE_NIM"].encode() not in path.read_bytes()
        assert b"Authorization" not in path.read_bytes()


def test_ambiguous_directory_is_never_reused_or_refrozen(directory, wire):
    latency.freeze(directory)
    (directory / "eligibility/latency-01").mkdir()
    class NoEnvironment:
        def get(self, name):
            raise AssertionError("Existing attempt accessed environment")
    with pytest.raises(FileExistsError):
        run_probe(directory, NoEnvironment())
    with pytest.raises(FileExistsError):
        latency.freeze(directory)
    with pytest.raises(ValueError, match="incomplete|ambiguous"):
        latency.verify_receipt(directory)
    assert len(wire[1]) == 3


def test_offline_commands_without_clients_environment_or_network(directory):
    code = '''
import os, socket, sys
from pathlib import Path
from scripts import prepare_completion_latency as p
from scripts import prepare_completion_pilot as base
class NoEnvironment(dict):
    def get(self, key, *args):
        if key in {"LANGUAGE", "LC_ALL", "LC_MESSAGES", "LANG", "COLUMNS", "LINES"}: return None
        raise AssertionError("environment accessed")
    def __getitem__(self, key):
        if key in {"COLUMNS", "LINES"}: raise KeyError(key)
        raise AssertionError("environment accessed")
def forbidden(*args, **kwargs): raise AssertionError("client or network accessed")
os.environ = NoEnvironment()
socket.socket = forbidden
base.OpenAICompatibleClient = base.AnthropicClient = forbidden
p.OpenAICompatibleClient = forbidden
assert p.main(["freeze", "--directory", sys.argv[1]]) == 0
assert p.main(["validate", "--directory", sys.argv[1]]) == 0
assert p.main(["verify-receipt", "--directory", sys.argv[1]]) == 1
'''
    result = subprocess.run([sys.executable, "-c", code, str(directory)], capture_output=True,
                            env={"PYTHONIOENCODING": "cp1252", "SYSTEMROOT": os.environ.get("SYSTEMROOT", "C:/Windows")})
    assert result.returncode == 0, result.stderr.decode("cp1252")
    assert result.stdout.isascii()


def test_completed_receipt_verifies_without_environment_clients_or_network(directory, wire):
    latency.freeze(directory)
    wire[0].append(nim())
    run_probe(directory)
    code = '''
import os, socket, sys
from pathlib import Path
from scripts import prepare_completion_latency as p
from scripts import prepare_completion_pilot as base
def forbidden(*args, **kwargs): raise AssertionError("client, environment or network accessed")
class NoEnvironment(dict):
    get = __getitem__ = forbidden
os.environ = NoEnvironment()
socket.socket = forbidden
base.OpenAICompatibleClient = base.AnthropicClient = p.OpenAICompatibleClient = forbidden
assert p.verify_receipt(Path(sys.argv[1]))["status"] == "eligible"
'''
    result = subprocess.run([sys.executable, "-c", code, str(directory)], capture_output=True,
                            env={"SYSTEMROOT": os.environ.get("SYSTEMROOT", "C:/Windows")})
    assert result.returncode == 0, result.stderr.decode("utf-8")
