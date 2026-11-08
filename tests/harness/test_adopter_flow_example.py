"""Offline adopter flow (release-1.0 M1.5, APR 4.1): profile -> preflight -> fallback -> run -> record."""

import http.client
import json
import runpy
from io import StringIO
from pathlib import Path

import pytest
import requests
from rich.console import Console

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "adopter_flow.py"


@pytest.fixture
def example(monkeypatch):
    def block_http(*args, **kwargs):
        raise AssertionError("Outbound HTTP is forbidden in the adopter fixture")

    monkeypatch.setattr(requests.sessions.Session, "request", block_http)
    monkeypatch.setattr(http.client.HTTPConnection, "connect", block_http)
    monkeypatch.setattr("recall_guard.core.profiles._psutil_memory",
                        lambda: pytest.fail("fixture must not read this machine's memory"))
    return runpy.run_path(str(EXAMPLE))


def run(example, scenario):
    return example["run_fixture"](scenario, console=Console(file=StringIO(), width=120))


def test_main_fixture_prints_panel_once_then_one_complete_record(example, capsys):
    assert example["main"](["--fixture"]) == 0
    output = capsys.readouterr().out
    panel, record = output[:output.index("\n{")], json.loads(output[output.index("\n{"):])
    assert "Not enough free memory" in panel and "13.4 GiB free, 20.0 GiB required for gpt-oss:20b" in panel
    assert panel.count("--profile ollama-gemma3-4b") == 1 and panel.count("--profile modal-gpt-oss-20b") == 1
    assert [(row["profile"], row["passed"], row["reason"]) for row in record["preflight"]] == [
        ("ollama-gpt-oss-20b", False, "insufficient_free_ram"), ("modal-gpt-oss-20b", False, "missing_credential"),
        ("ollama-gemma3-4b", True, None)]
    assert record["selected_profile"] == "ollama-gemma3-4b" and record["model"] == "gemma3:4b"
    assert record["endpoint"] == "http://localhost:11434/v1/chat/completions"
    assert record["status"] == "completed"
    assert record["scenario_note"].startswith("Simulated: the fixture endpoint omits logprobs")
    assert "Real gemma3:4b on Ollama returned aligned logprobs" in record["scenario_note"]


@pytest.mark.parametrize("scenario", ["observed_missing_tokens", "declared_missing_tokens", "no_fallback_authorized",
                                      "token_evidence_without_fit"])
def test_every_scenario_carries_a_simulation_note_also_in_help(example, scenario, capsys):
    note = run(example, scenario)["scenario_note"]
    assert note == example["SCENARIO_NOTES"][scenario] and note.startswith("Simulated:")
    with pytest.raises(SystemExit):
        example["main"](["--help"])
    assert scenario + ":" in " ".join(capsys.readouterr().out.split())


def test_observed_missing_tokens_select_authorized_completion_only_plan(example):
    record = run(example, "observed_missing_tokens")
    assert record["omission"] == {"reason": "missing_token_evidence", "omitted_controls": ["logprobs", "top_logprobs"],
                                  "mia_unavailable_reason": "MIA/MCS unavailable: missing_token_evidence"}
    assert record["settings"]["requested"]["logprobs"] is True
    assert not {"logprobs", "top_logprobs"} & set(record["settings"]["selected"])
    assert record["features"]["requested"]["mia_scoring"] and not record["features"]["selected"]["mia_scoring"]
    assert record["fallback_policy"]["omit_logprob_controls"] is True and record["policy_id"]
    assert record["method"] == {"requested": "rewritten_single", "selected": "rewritten_single"}
    # The first target probe observes no tokens; the rebuilt selected target is probed afresh.
    assert record["preparation"]["probes"] == {"target": 2, "rewriter": 1, "verifier": 1}
    accepted, rejected = record["items"]
    assert accepted["stages"] == {"preflight": "ready", "rewrite": "generated", "fidelity_deterministic": "passed",
                                  "fidelity_semantic": "passed", "target": "completed"}
    assert accepted["observed_method"] == "rewritten_single" and accepted["answer"] == "up"
    for item in record["items"]:
        assert item["mcs"].startswith("MIA/MCS unavailable in the selected completion-only plan")
    # Fidelity rejection stops the item before the target; nothing falls back.
    assert rejected["stages"]["fidelity_deterministic"] == "rejected"
    assert rejected["stages"]["fidelity_semantic"] == "not_performed" and rejected["stages"]["target"] == "skipped"
    assert rejected["failure_kind"] == "fidelity_rejected" and rejected["observed_method"] is None


def test_every_probe_retry_and_generation_is_counted_exactly_once(example):
    requests_ = run(example, "observed_missing_tokens")["requests"]
    assert requests_["physical_requests"] == requests_["unique_call_ids"] == requests_["budget_used"] == 12
    assert requests_["by_phase_and_kind"] == {"preparation/probe": 4, "evaluation/probe": 3,
                                              "evaluation/generation": 4, "evaluation/retry": 1}
    assert sum(requests_["by_stage"].values()) == 12
    assert requests_["by_stage"]["rewrite"] == 3  # Accepted item: 503 + retry; rejected item: 1.
    assert requests_["statuses"] == {"failed": 1, "succeeded": 11}
    assert requests_["usage"]["requests_without_usage"] == 1 and requests_["usage"]["complete"] is False
    assert requests_["cost"]["total"] is None and requests_["cost"]["status"] == "unknown"


def test_declared_missing_tokens_resolve_before_any_target_probe(example):
    record = run(example, "declared_missing_tokens")
    assert record["omission"]["reason"] == "unsupported_token_evidence"
    assert record["preparation"]["probes"]["target"] == 1
    assert record["requests"]["by_phase_and_kind"]["preparation/probe"] == 3
    assert record["status"] == "completed"


def test_without_authorized_fallback_missing_tokens_stop_before_evaluation(example):
    record = run(example, "no_fallback_authorized")
    assert record["fallback_policy"] is None and record["omission"]["reason"] is None
    assert record["preparation"]["stopping_reason"] == "missing_token_evidence"
    assert record["items"] == [] and record["requests"]["physical_requests"] == 1


def test_token_evidence_without_matching_fit_is_refused_not_downgraded(example):
    record = run(example, "token_evidence_without_fit")
    assert record["preparation"]["stopping_reason"] == "matching_calibration_required"
    assert record["omission"]["reason"] is None and record["features"]["selected"]["mia_scoring"] is True
    assert record["items"] == []


def test_dry_run_reads_nothing(example, capsys):
    assert example["main"]([]) == 0
    record = json.loads(capsys.readouterr().out)
    assert record["status"] == "dry_run" and record["network_requests"] == 0
