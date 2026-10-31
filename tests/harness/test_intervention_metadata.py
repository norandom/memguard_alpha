"""Synthetic publication boundaries; no provider calls or credential values."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from recall_guard.harness.intervention_config import experiment_plan
from recall_guard.harness.intervention_journal import ExperimentJournal, ExperimentRun
from recall_guard.harness.intervention_types import ExperimentProtocol
from scripts.run_intervention import main
from tests.harness.test_intervention_config import config_files, mutate
from tests.harness.test_intervention_report import recorded_run
from tests.harness.test_intervention_types import protocol

MARKER = "SYNTHETIC_MARKER_NOT_A_REAL_CREDENTIAL"
ERROR = "credential and authorization fields"
FIELDS = ("Authorization", "aUtHoRiZaTiOn", "X-API-Key", "api_KEY", "credential-ref", "HEADERS")


@pytest.mark.parametrize("location", ["audit_selection", "cohort_definitions", "ensemble_settings"])
@pytest.mark.parametrize("field", FIELDS)
@pytest.mark.parametrize("construction", ["direct", "from_dict"])
def test_protocol_rejects_nested_execution_metadata(location, field, construction):
    value = protocol().to_dict()
    value[location]["review"] = [{"nested": {field: MARKER}}]
    with pytest.raises(ValueError, match=ERROR) as caught:
        if construction == "from_dict":
            ExperimentProtocol.from_dict(value)
        else:
            protocol(**{location: value[location]})
    assert MARKER not in str(caught.value)


def unsafe_snapshot(location, field):
    """Keep corruption as raw JSON so legacy checkpoint fixtures stay realistic."""
    snapshot = recorded_run().to_dict()
    checkpoint = snapshot["checkpoint"]
    containers = {
        "draw": checkpoint["draws"][0]["metadata"],
        "draw_response": checkpoint["draws"][0]["response"]["usage"],
        "rewrite": checkpoint["rewrites"][0]["fidelity_checks"],
        "rewrite_response": checkpoint["rewrites"][0]["rewrite_response"].setdefault("usage", {}),
        "request": checkpoint["requests"][0]["event"]["usage"],
        "preflight": snapshot["preflight"],
        "preflight_response": snapshot["preflight"]["responses"]["target"],
        "outcome": snapshot["arms"][0]["outcome"],
        "outcome_response": snapshot["arms"][0]["outcome"].setdefault("response", {}),
    }
    container = containers[location]
    if location == "rewrite_response" and container is None:
        container = checkpoint["rewrites"][0]["rewrite_response"]["usage"] = {}
    if location in {"draw_response", "rewrite_response", "request"}:
        container[field] = 7  # Typed usage values are integer counters.
    else:
        container["review"] = [{"nested": {field: MARKER}}]
    return snapshot


LOCATIONS = ("draw", "draw_response", "rewrite", "rewrite_response", "request", "preflight",
             "preflight_response", "outcome", "outcome_response")


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("location", LOCATIONS)
@pytest.mark.parametrize("field", FIELDS)
def test_journal_rejects_runtime_metadata_without_touching_disk(tmp_path, existing, location, field):
    journal = ExperimentJournal(tmp_path / "checkpoint.json")
    if existing:
        journal.save(recorded_run())
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    result = ExperimentRun.from_dict(unsafe_snapshot(location, field))
    with pytest.raises(ValueError, match=ERROR) as caught:
        journal.save(result)
    assert MARKER not in str(caught.value)
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("field", FIELDS)
def test_journal_checks_write_ahead_intent_before_opening_a_file(tmp_path, existing, field):
    journal = ExperimentJournal(tmp_path / "checkpoint.json")
    if existing:
        journal.save(recorded_run())
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    with pytest.raises(ValueError, match=ERROR) as caught:
        journal.save(recorded_run(), {"kind": "arm", "nested": [{field: MARKER}]})
    assert MARKER not in str(caught.value)
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


@pytest.mark.parametrize("location", [*LOCATIONS, "protocol", "intent"])
def test_unsafe_historical_journal_load_is_rejected_without_rewriting(tmp_path, location):
    snapshot = recorded_run().to_dict() if location in {"protocol", "intent"} else unsafe_snapshot(location, "headers")
    intent = None
    if location == "protocol":
        snapshot["checkpoint"]["protocol"]["audit_selection"]["authorization"] = MARKER
    if location == "intent":
        intent = {"nested": [{"headers": {"Authorization": MARKER}}]}
    path = tmp_path / "checkpoint.json"
    path.write_text(json.dumps({"journal_version": 1, **snapshot, "intent": intent}), encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(ValueError, match=ERROR) as caught:
        ExperimentJournal(path).load()
    assert MARKER not in str(caught.value)
    assert path.read_bytes() == before and list(tmp_path.iterdir()) == [path]


def test_plan_rejects_unsafe_derived_metadata_before_publication():
    config = SimpleNamespace(protocol=protocol(ensemble_settings={"min_parsed": 1}), items=(),
                             historical={"review": [{"headers": {"Authorization": MARKER}}]})
    with pytest.raises(ValueError, match=ERROR) as caught:
        experiment_plan(config)
    assert MARKER not in str(caught.value)


@pytest.mark.parametrize("location", ["audit_selection", "cohort_definitions", "ensemble_settings"])
@pytest.mark.parametrize("field", FIELDS)
def test_cli_dry_run_rejects_metadata_without_output_files_or_calls(tmp_path, capsys, location, field):
    pp, ip, _, _ = config_files(tmp_path)
    mutate(pp, lambda value: value[location].update({"review": [{"nested": {field: MARKER}}]}))
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir()}

    def forbidden(*args, **kwargs):
        pytest.fail("Metadata rejection must happen before client construction/provider work")

    assert main(["--dry-run", "--protocol", str(pp), "--items", str(ip)], workflow_factory=forbidden) == 2
    captured = capsys.readouterr()
    assert not captured.out
    assert ERROR in captured.err and MARKER not in captured.err
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == before


def test_journal_preserves_only_schema_known_task_payload_exemptions(tmp_path):
    run = recorded_run()
    task_value = {"headers": {"Authorization": MARKER}, "credential_ref": ["quoted example"]}
    prose = json.dumps(task_value, ensure_ascii=False) + " 東京"
    item = replace(run.checkpoint.items[0], label=task_value, instructions=prose, evidence=prose)
    draw = replace(run.checkpoint.draws[0], prompt=prose,
                   metadata=dict(run.checkpoint.draws[0].metadata) | {"answer": task_value})
    rewrite = replace(run.checkpoint.rewrites[0], original_evidence=prose, rewritten_evidence=prose,
                      rewrite_prompt=prose, verification_prompt=prose)
    arm = replace(run.arms[0], outcome=dict(run.arms[0].outcome) | {"answer": task_value})
    checkpoint = replace(run.checkpoint, items=(item, *run.checkpoint.items[1:]),
                         draws=(draw, *run.checkpoint.draws[1:]),
                         rewrites=(rewrite, *run.checkpoint.rewrites[1:]),
                         requests=(replace(run.checkpoint.requests[0], prompt_hash=draw.prompt_hash),))
    run = replace(run, checkpoint=checkpoint, arms=(arm, *run.arms[1:]))
    journal = ExperimentJournal(tmp_path / "checkpoint.json")
    journal.save(run)
    assert journal.load().to_dict() == run.to_dict()


@pytest.mark.parametrize("location", ["draw", "outcome", "preflight"])
def test_answer_key_outside_schema_known_task_slots_is_still_metadata(tmp_path, location):
    snapshot = recorded_run().to_dict()
    container = {"draw": snapshot["checkpoint"]["draws"][0]["metadata"],
                 "outcome": snapshot["arms"][0]["outcome"], "preflight": snapshot["preflight"]}[location]
    container["nested"] = {"answer": {"Authorization": MARKER}}
    with pytest.raises(ValueError, match=ERROR):
        ExperimentJournal(tmp_path / "checkpoint.json").save(SimpleNamespace(to_dict=lambda: snapshot))
    assert not list(tmp_path.iterdir())
