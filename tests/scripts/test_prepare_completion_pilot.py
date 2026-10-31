"""Pilot setup runs offline except synthetic wire fixtures, never real credentials."""

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import requests

from scripts import prepare_completion_pilot as pilot

DATA = Path(__file__).resolve().parents[2] / "data/experiments/2026-10-31-completion-api"
ENV = {"FIXTURE_NIM": "synthetic-nim-private", "FIXTURE_ANTHROPIC": "synthetic-anthropic-private"}


@pytest.fixture
def directory(tmp_path):
    result = tmp_path / "inputs"
    shutil.copytree(DATA, result, ignore=shutil.ignore_patterns("study-plan.json", "eligibility"))
    return result


@pytest.fixture
def wire(monkeypatch):
    pending, calls = [], []

    def send(adapter, prepared, **kwargs):
        calls.append((prepared, kwargs))
        assert pending, "Unexpected synthetic model request"
        entry = pending.pop(0)
        if isinstance(entry, Exception):
            raise entry
        response = requests.Response()
        response.status_code, body = entry
        response._content = json.dumps(body).encode()
        response.request = prepared
        response.url = prepared.url
        return response

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)
    return pending, calls


def nim(*, evidence=True):
    choice = {"message": {"content": "OK"}, "finish_reason": "stop"}
    if evidence:
        choice["logprobs"] = {"content": [{"token": "OK", "logprob": -0.1,
                                          "top_logprobs": [{"token": "OK", "logprob": -0.1}]}]}
    return 200, {"id": "fixture-nim", "model": "openai/gpt-oss-20b", "choices": [choice],
                 "usage": {"prompt_tokens": 6, "completion_tokens": 1}}


def anthropic():
    return 200, {"id": "fixture-opus", "model": "claude-opus-5-5", "content": [{"type": "text", "text": "OK"}],
                 "stop_reason": "end_turn", "usage": {"input_tokens": 6, "output_tokens": 1}}


def run_probe(directory, environment=ENV):
    return pilot.probe(directory, nim_credential_env="FIXTURE_NIM", anthropic_credential_env="FIXTURE_ANTHROPIC",
                       environment=environment)


def test_frozen_plan_binds_inputs_models_conditions_methods_and_caps(directory):
    plan = pilot.freeze(directory)
    assert plan == pilot.validate(directory)
    assert plan["inputs"]["counts"] == {"evaluation": 16, "calibration_is": 20, "calibration_oos": 20}
    assert plan["inputs"]["sha256"]["evaluation.json"] == hashlib.sha256((directory / "evaluation.json").read_bytes()).hexdigest()
    assert plan["budget"]["phase_caps"] == {"eligibility": 4, "calibration": 220, "nim_evaluation": 144,
                                             "anthropic_evaluation": 144}
    assert sum(plan["budget"]["phase_caps"].values()) == plan["budget"]["study_cap"] == 512
    assert plan["evaluation"]["conditions"] == ["original_single", "original_ensemble", "rewritten_single", "rewritten_ensemble"]
    assert plan["evaluation"]["draws"] == 3 and plan["evaluation"]["single_draw_index"] == 0
    assert plan["evaluation"]["utility_tolerance"] == 0.05
    assert len(plan["evaluation"]["audit"]["item_ids"]) == 16
    assert plan["models"]["nim"]["parameters"] == {"max_tokens": 2048, "temperature": 0.0, "logprobs": True, "top_logprobs": 20}
    assert plan["models"]["anthropic"]["parameters"] == {"max_tokens": 2048}
    assert plan["calibration"]["fit_fingerprints"] is None
    assert plan["method"]["checker"]["deterministic_version"]
    serialized = (directory / "study-plan.json").read_text(encoding="utf-8")
    assert "credential" not in serialized and "headers" not in serialized and str(directory) not in serialized
    assert b"\r" not in (directory / "study-plan.json").read_bytes()


def test_freeze_exact_resume_and_tamper_refusal(directory):
    plan = pilot.freeze(directory)
    original = (directory / "study-plan.json").read_bytes()
    assert pilot.freeze(directory) == plan
    assert (directory / "study-plan.json").read_bytes() == original
    changed = copy.deepcopy(plan)
    changed["models"]["nim"]["parameters"]["max_tokens"] = 64
    (directory / "study-plan.json").write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen plan"):
        pilot.validate(directory)
    with pytest.raises((ValueError, FileExistsError)):
        pilot.freeze(directory)


def test_input_tamper_stops_before_clients_or_calls(directory, wire):
    pilot.freeze(directory)
    (directory / "evaluation.json").write_bytes((directory / "evaluation.json").read_bytes() + b"\n")
    with pytest.raises(ValueError):
        run_probe(directory)
    assert not wire[1]


def test_missing_second_key_has_zero_calls_and_partial_failure_receipt(directory, wire):
    pilot.freeze(directory)
    result = run_probe(directory, {"FIXTURE_NIM": ENV["FIXTURE_NIM"]})
    assert result["status"] == "failed" and result["failure"]["category"] == "missing_credential"
    assert result["failure"]["provider"] == "anthropic"
    assert result["budget"]["used"] == 0 and not wire[1]
    assert pilot.verify_receipt(directory)["status"] == "failed"


def test_actual_fixture_probes_exact_settings_and_shared_eligibility_ledger(directory, wire):
    pilot.freeze(directory)
    wire[0].extend([nim(), anthropic()])
    result = run_probe(directory)
    assert result["status"] == "eligible" and result["evidence_kind"] == "actual_generation_probe"
    assert result["budget"]["used"] == 2 and result["budget"]["remaining"] == 2
    assert result["remaining_phase_caps"]["calibration"] == 220
    assert [e["stage"] for e in result["events"]] == ["eligibility/probe/nim", "eligibility/probe/anthropic"]
    assert all(e["status"] == "succeeded" and e["attempt"] == 1 and e["latency_s"] >= 0 for e in result["events"])
    assert all(e["usage"] and e["cost"] is None for e in result["events"])
    assert all(p["ready_for_batch"] for p in result["providers"].values())
    for index, name in enumerate(("nim", "anthropic")):
        prepared, options = wire[1][index]
        body = json.loads(prepared.body)
        assert body == {"model": result["providers"][name]["config"]["model"],
                        "messages": [{"role": "user", "content": "Reply with the word OK."}],
                        **result["providers"][name]["config"]["parameters"]}
        assert options["timeout"] == 90
        assert result["providers"][name]["confirmed_effective_values"] is None
        assert result["providers"][name]["raw_response"] is None
    assert wire[1][1][0].headers["anthropic-version"] == "2026-10-21"
    assert pilot.verify_receipt(directory) == result
    all_bytes = b"".join(p.read_bytes() for p in (directory / "eligibility").glob("*.json"))
    assert all(value.encode() not in all_bytes for value in ENV.values())
    assert b"FIXTURE_NIM" not in all_bytes and b"Authorization" not in all_bytes


@pytest.mark.parametrize("status,category", [(401, "authentication"), (402, "billing"), (400, "parameter")])
def test_hard_failures_no_retries_and_retained_safe_receipts(directory, wire, status, category):
    pilot.freeze(directory)
    wire[0].append((status, {"error": {"message": ENV["FIXTURE_NIM"], "param": "max_tokens",
                                      "detail": ENV["FIXTURE_NIM"]}}))
    result = run_probe(directory)
    assert result["status"] == "failed" and result["failure"]["category"] == category
    assert result["failure"]["provider"] == "nim" and result["failure"]["phase"] == "eligibility"
    assert len(wire[1]) == result["budget"]["used"] == 1
    assert result["events"][0]["status_code"] == status
    assert ENV["FIXTURE_NIM"] not in json.dumps(result)
    assert pilot.verify_receipt(directory) == result


def test_missing_mia_evidence_fails_instead_of_switching_to_text(directory, wire):
    pilot.freeze(directory)
    wire[0].append(nim(evidence=False))
    result = run_probe(directory)
    assert result["status"] == "failed" and result["failure"]["category"] == "capability"
    assert result["budget"]["used"] == 1 and len(wire[1]) == 1
    assert not result["providers"]["nim"]["ready_for_batch"]


def test_complete_receipt_never_reissues_paid_probes(directory, wire):
    pilot.freeze(directory)
    wire[0].extend([nim(), anthropic()])
    run_probe(directory)
    with pytest.raises(FileExistsError, match="eligibility"):
        run_probe(directory)
    assert len(wire[1]) == 2


def test_durable_intent_precedes_network_and_ambiguous_restart_refused(directory, wire, monkeypatch):
    pilot.freeze(directory)
    original = requests.adapters.HTTPAdapter.send

    def crash(*args, **kwargs):
        intent = json.loads((directory / "eligibility/intent.json").read_bytes())
        assert intent["provider"] == "nim" and intent["status"] == "started"
        raise KeyboardInterrupt

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", crash)
    with pytest.raises(KeyboardInterrupt):
        run_probe(directory)
    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", original)
    with pytest.raises(FileExistsError):
        run_probe(directory)
    with pytest.raises(ValueError, match="incomplete|ambiguous"):
        pilot.verify_receipt(directory)
    assert not wire[1]


def test_offline_fresh_process_no_environment_or_network_and_cp1252(directory):
    code = '''
import os, socket, sys
from pathlib import Path
from scripts import prepare_completion_pilot as p
class NoEnvironment(dict):
    def get(self, key, *a, **k):
        if key in {"LANGUAGE", "LC_ALL", "LC_MESSAGES", "LANG", "COLUMNS", "LINES"}: return None
        raise AssertionError("credential environment accessed")
    def __getitem__(self, k):
        if k in {"COLUMNS", "LINES"}: raise KeyError(k)
        raise AssertionError("credential environment accessed")
os.environ = NoEnvironment()
socket.socket = lambda *a, **k: (_ for _ in ()).throw(AssertionError("network accessed"))
p.main(["freeze", "--directory", sys.argv[1]])
p.main(["validate", "--directory", sys.argv[1]])
'''
    result = subprocess.run([sys.executable, "-c", code, str(directory)], capture_output=True,
                            env={"PYTHONIOENCODING": "cp1252", "SYSTEMROOT": os.environ.get("SYSTEMROOT", "C:/Windows")})
    assert result.returncode == 0, result.stderr.decode("cp1252")
    assert result.stdout.isascii()


def test_receipt_tamper_is_rejected(directory, wire):
    pilot.freeze(directory)
    wire[0].extend([nim(), anthropic()])
    run_probe(directory)
    path = directory / "eligibility/receipt.json"
    value = json.loads(path.read_bytes())
    value["remaining_phase_caps"]["calibration"] = 224
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="receipt|budget|phase"):
        pilot.verify_receipt(directory)


def test_physical_eligibility_cap_blocks_extra_requests_and_retains_partial_ledger(directory, wire, monkeypatch):
    pilot.freeze(directory)
    wire[0].extend([nim()] * 4)

    def excessive_probe(self, prompt, *, stage):
        for _ in range(5):
            self.generate_for_stage(prompt, stage=stage)

    monkeypatch.setattr(pilot.OpenAICompatibleClient, "probe", excessive_probe)
    result = run_probe(directory)
    assert result["failure"] == {"phase": "eligibility", "provider": "nim", "category": "budget_exhausted"}
    assert result["budget"] == {"limit": 4, "used": 4, "remaining": 0}
    assert len(wire[1]) == 4 and all(event["status"] == "succeeded" for event in result["events"])
    assert pilot.verify_receipt(directory) == result


def test_frozen_method_matches_actual_rewrite_policy_without_clients(directory):
    from recall_guard.harness.rewriting import EvidenceRewriter, RewritePolicy

    class NoCall:
        config = pilot.model_configs()["anthropic"]

        def generate(self, prompt, **parameters):
            raise AssertionError("Offline method identity must never generate")

    policy = RewritePolicy(EvidenceRewriter(NoCall()), NoCall())
    assert pilot.freeze(directory)["method"]["intervention_id"] == policy.intervention_id


def run_recovery(directory, environment=ENV):
    return pilot.recover(directory, nim_credential_env="FIXTURE_NIM", anthropic_credential_env="FIXTURE_ANTHROPIC",
                         environment=environment)


def initial_timeout(directory, wire):
    pilot.freeze(directory)
    wire[0].append(requests.Timeout("synthetic timeout"))
    return run_probe(directory)


def test_explicit_recovery_counts_timeout_and_checks_other_provider_before_retry(directory, wire):
    first = initial_timeout(directory, wire)
    history = {p.name: p.read_bytes() for p in (directory / "eligibility").glob("*.json")}
    plan_bytes = (directory / "study-plan.json").read_bytes()
    wire[0].extend([anthropic(), nim()])
    result = run_recovery(directory)
    assert result["status"] == "eligible" and result["budget"] == {"limit": 4, "used": 3, "remaining": 1}
    assert len(wire[1]) == 3 and "anthropic.com" in wire[1][1][0].url
    assert result["events"][0] == first["events"][0] and result["events"][0]["code"] == "timeout"
    assert [e["call_id"] for e in result["events"]] == [1, 2, 3]
    assert [e["stage"] for e in result["events"]][1:] == ["eligibility/recovery-01/probe/anthropic", "eligibility/recovery-01/probe/nim"]
    assert result["recovery_of"]["sha256"] == hashlib.sha256(history["receipt.json"]).hexdigest()
    assert result["recovery_of"]["failure"] == first["failure"]
    assert (directory / "study-plan.json").read_bytes() == plan_bytes
    assert all((directory / "eligibility" / name).read_bytes() == content for name, content in history.items())
    assert pilot.verify_receipt(directory) == result
    all_bytes = b"".join(p.read_bytes() for p in (directory / "eligibility/recovery-01").glob("*.json"))
    assert all(value.encode() not in all_bytes for value in ENV.values())
    with pytest.raises(FileExistsError):
        run_recovery(directory)
    assert len(wire[1]) == 3


def test_explicit_recovery_retains_prior_success_without_repeating_it(directory, wire):
    pilot.freeze(directory)
    wire[0].extend([nim(), requests.Timeout("synthetic timeout")])
    first = run_probe(directory)
    wire[0].append(anthropic())
    result = run_recovery(directory)
    assert result["status"] == "eligible" and result["budget"]["used"] == 3
    assert result["providers"]["nim"] == first["providers"]["nim"]
    assert len(wire[1]) == 3 and "anthropic.com" in wire[1][-1][0].url
    assert pilot.verify_receipt(directory) == result


def test_explicit_recovery_missing_key_spends_zero_new_requests(directory, wire):
    initial_timeout(directory, wire)
    result = run_recovery(directory, {"FIXTURE_NIM": ENV["FIXTURE_NIM"]})
    assert result["status"] == "failed" and result["failure"]["category"] == "missing_credential"
    assert result["budget"]["used"] == len(wire[1]) == 1
    assert pilot.verify_receipt(directory) == result


@pytest.mark.parametrize("state", ["eligible", "ambiguous_initial", "ambiguous_recovery"])
def test_explicit_recovery_rejects_nonterminal_or_existing_history_before_environment(directory, wire, state):
    class ForbiddenEnvironment:
        def get(self, key):
            raise AssertionError("Rejected recovery must not access selected values")

    if state == "eligible":
        pilot.freeze(directory)
        wire[0].extend([nim(), anthropic()])
        run_probe(directory)
    elif state == "ambiguous_initial":
        pilot.freeze(directory)
        (directory / "eligibility").mkdir()
    else:
        initial_timeout(directory, wire)
        (directory / "eligibility/recovery-01").mkdir()
    before = len(wire[1])
    with pytest.raises((ValueError, FileExistsError)):
        run_recovery(directory, ForbiddenEnvironment())
    assert len(wire[1]) == before
    if state == "ambiguous_recovery":
        with pytest.raises(ValueError, match="incomplete|ambiguous"):
            pilot.verify_receipt(directory)


def test_explicit_recovery_does_not_reset_exhausted_phase_cap(directory, wire, monkeypatch):
    pilot.freeze(directory)
    wire[0].extend([nim()] * 4)

    def excessive_probe(self, prompt, *, stage):
        for _ in range(5):
            self.generate_for_stage(prompt, stage=stage)

    monkeypatch.setattr(pilot.OpenAICompatibleClient, "probe", excessive_probe)
    run_probe(directory)
    with pytest.raises(ValueError, match="exhausted"):
        run_recovery(directory)
    assert len(wire[1]) == 4
