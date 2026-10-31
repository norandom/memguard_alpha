"""Portable reporting fixtures and fresh-process, network-free metric replay."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from recall_guard.core.completion import ModelResponse, TokenAlternative, TokenEvidence
from recall_guard.core.request_control import RequestEvent
from recall_guard.harness.intervention_journal import ExperimentRun
from recall_guard.harness.intervention_metrics import summarize_experiment
from recall_guard.harness.intervention_report import export_experiment, load_experiment_report
from recall_guard.harness.intervention_types import EvidenceArm, RequestRecord
from tests.harness.test_intervention_metrics import fixture


def recorded_run(*, partial=False):
    run = fixture([["up", "down", "down", "up"], ["down", "up", "up", "up"], ["up"] * 4],
                  rejected=("1",), pending=(("2", "rewritten"),) if partial else ())
    item = replace(run.checkpoint.items[0], evidence="München 東京: quoted path C:\\source\\facts.\nΔ = 2%.")
    rewrites = tuple(replace(record, original_evidence=item.evidence,
                             rewritten_evidence="A review: " + item.evidence,
                             rewrite_prompt="Rewrite " + item.evidence,
                             verification_prompt="Verify " + item.evidence,
                             rewrite_response=ModelResponse("A review: " + item.evidence),
                             verification_response=ModelResponse('{"equivalent":true}'))
                     if record.item_id == "0" else record for record in run.checkpoint.rewrites)
    response = ModelResponse("raw 東京 response\n", (TokenEvidence("raw", -0.5, (TokenAlternative("raw", -0.5),)),),
                             {"input_tokens": 11, "output_tokens": 3}, "stop",
                             {"max_tokens": 64, "temperature": 0.3}, {"temperature": 0.3}, "request-α")
    draws = tuple(replace(draw, response=response, metadata=dict(draw.metadata) | {
        "guarded_score": {"parse_ok": True, "p_memorized": 0.25, "raw_confidence": 0.8,
                          "memguard_confidence": 0.6},
    }) if draw.identity == ("0", EvidenceArm.ORIGINAL, 0) else draw for draw in run.checkpoint.draws)
    request = RequestRecord("execution", RequestEvent(1, "target", "target", 1, 2.0,
                            "2026-10-31T12:00:00+00:00", "succeeded", 0.3,
                            {"input_tokens": 11, "output_tokens": 3}),
                            "target", draws[0].prompt_hash, "0", EvidenceArm.ORIGINAL, 0)
    protocol = replace(run.checkpoint.protocol,
                       target_model=replace(run.checkpoint.protocol.target_model, credential_ref="PRIVATE_KEY_ENV"))
    checkpoint = replace(run.checkpoint, protocol=protocol, items=(item, *run.checkpoint.items[1:]),
                         rewrites=rewrites, draws=draws, requests=(request,))
    return replace(run, checkpoint=checkpoint, preflight={"responses": {"target": ModelResponse("OK").to_dict()}},
                   stop_reason="budget exhausted" if partial else None)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")


def rehash(directory, artifact):
    path = directory / "manifest.json"
    manifest = read_json(path)
    manifest["artifacts"][artifact]["sha256"] = hashlib.sha256(
        (directory / manifest["artifacts"][artifact]["path"]).read_bytes()).hexdigest()
    write_json(path, manifest)


@pytest.mark.parametrize("partial", [False, True])
def test_export_retains_exact_records_and_analysis_options(tmp_path, partial):
    run = recorded_run(partial=partial)
    destination = tmp_path / "report"
    manifest_path = export_experiment(run, destination, confidence_level=0.8, confidence_component=None)
    assert manifest_path == destination / "manifest.json"
    report = load_experiment_report(destination)
    assert report.run.to_dict() == run.to_dict()
    assert report.recompute_metrics() == report.metrics == summarize_experiment(
        run, confidence_level=0.8, confidence_component=None)
    assert report.manifest["analysis"] == {"version": 1, "confidence_level": 0.8, "confidence_component": None}
    assert report.manifest["status"] == run.checkpoint.status.value
    assert report.run.stop_reason == run.stop_reason
    assert report.metrics["pooled"]["interpretation"]["mitigation_success"] is False


def test_utf8_lf_byte_hashes_and_credential_free_manifest(tmp_path):
    destination = tmp_path / "report"
    export_experiment(recorded_run(), destination)
    manifest = read_json(destination / "manifest.json")
    assert manifest["integrity"] == "sha256_exact_utf8_file_bytes_including_lf; not_provider_origin_evidence"
    for entry in manifest["artifacts"].values():
        content = (destination / entry["path"]).read_bytes()
        assert b"\r\n" not in content and content.endswith(b"\n")
        assert hashlib.sha256(content).hexdigest() == entry["sha256"]
    records = (destination / "records.json").read_text(encoding="utf-8")
    assert "München 東京" in records and "PRIVATE_KEY_ENV" not in records
    assert "credential_ref" not in records and "credential_ref" not in json.dumps(manifest)
    assert str(tmp_path) not in json.dumps(manifest)
    assert load_experiment_report(destination).run.checkpoint.items[0].evidence.endswith("Δ = 2%.")


@pytest.mark.parametrize("location", ["draw", "preflight", "provenance"])
def test_structured_credentials_in_execution_metadata_are_rejected_before_writing(tmp_path, location):
    run = recorded_run()
    unsafe = {"diagnostics": {"headers": {"Authorization": "synthetic-fixture-value"}}}
    if location == "draw":
        draw = replace(run.checkpoint.draws[0], metadata=dict(run.checkpoint.draws[0].metadata) | unsafe)
        run = replace(run, checkpoint=replace(run.checkpoint, draws=(draw, *run.checkpoint.draws[1:])))
    elif location == "preflight":
        run = replace(run, preflight=dict(run.preflight) | unsafe)
    else:
        arm = replace(run.arms[0], outcome=dict(run.arms[0].outcome) | {"provenance": unsafe})
        run = replace(run, arms=(arm, *run.arms[1:]))
    destination = tmp_path / "report"
    with pytest.raises(ValueError, match="credential and authorization fields"):
        export_experiment(run, destination)
    assert not destination.exists()


def protocol_metadata_run(location, field):
    run = recorded_run()
    protocol = run.checkpoint.protocol.to_dict()
    metadata = protocol[location]
    metadata["review"] = [{"nested": {field: "SYNTHETIC_REVIEW_FIXTURE"}}]
    protocol = type(run.checkpoint.protocol).from_dict(protocol)
    return replace(run, checkpoint=replace(run.checkpoint, protocol=protocol))


@pytest.mark.parametrize("location", [
    "audit_selection", "cohort_definitions", "ensemble_settings",
])
@pytest.mark.parametrize("field", ["headers", "Authorization", "credential_ref", "X-API-Key"])
def test_protocol_credentials_are_rejected_before_export_creates_destination(tmp_path, location, field):
    destination = tmp_path / "report"
    with pytest.raises(ValueError, match="credential and authorization fields"):
        export_experiment(protocol_metadata_run(location, field), destination)
    assert not destination.exists()


@pytest.mark.parametrize("location", [
    "audit_selection", "cohort_definitions", "ensemble_settings",
])
@pytest.mark.parametrize("field", ["headers", "Authorization", "credential_ref", "X-API-Key"])
def test_rehashed_protocol_credentials_are_rejected_on_load(tmp_path, location, field):
    destination = tmp_path / "report"
    export_experiment(recorded_run(), destination)
    records = read_json(destination / "records.json")
    protocol = records["run"]["checkpoint"]["protocol"]
    protocol[location]["review"] = [{"nested": {field: "SYNTHETIC_REVIEW_FIXTURE"}}]
    # Historical artifacts can contain unsafe raw JSON even though the current
    # protocol constructor rejects it. Keep the identity/metrics coherent using
    # the unchanged canonical protocol hash, without constructing unsafe types.
    protocol_id = hashlib.sha256(json.dumps(protocol, sort_keys=True, ensure_ascii=False, allow_nan=False,
                                            separators=(",", ":")).encode("utf-8")).hexdigest()
    records["run"]["checkpoint"]["protocol_id"] = protocol_id
    manifest = read_json(destination / "manifest.json")
    metrics = read_json(destination / "metrics.json")["metrics"]
    metrics["protocol_id"] = protocol_id
    metrics["provenance"]["cohort_definitions"] = protocol["cohort_definitions"]
    manifest["protocol_id"] = protocol_id
    write_json(destination / "records.json", records)
    write_json(destination / "metrics.json", {"schema_version": 1, "metrics": metrics})
    write_json(destination / "manifest.json", manifest)
    rehash(destination, "records")
    rehash(destination, "metrics")
    # A coherent identity, derived settings, metrics and byte hashes must not
    # authenticate execution metadata containing structured credentials.
    with pytest.raises(ValueError, match="credential and authorization fields"):
        load_experiment_report(destination)


def response_metadata_run(location, field):
    snapshot = recorded_run().to_dict()
    checkpoint = snapshot["checkpoint"]
    if location.startswith("arm_response"):
        response = ModelResponse("Factual response content").to_dict()
        snapshot["arms"][0]["outcome"]["response"] = response
        container = response if location == "arm_response" else response[location.removeprefix("arm_response_")]
        if container is None:
            response["usage"] = container = {}
    elif location == "draw_response_usage":
        container = checkpoint["draws"][0]["response"]["usage"]
    elif location in {"rewrite_response_usage", "verification_response_usage"}:
        response = checkpoint["rewrites"][0][location.removesuffix("_usage")]
        response["usage"] = container = {}
    else:
        assert location == "request_event_usage"
        container = checkpoint["requests"][0]["event"]["usage"]
    # Typed usage counters accept arbitrary names but only integer values.
    container[field] = 7 if location.endswith("_usage") else {"nested": ["SYNTHETIC_REVIEW_FIXTURE"]}
    return ExperimentRun.from_dict(snapshot)


_RESPONSE_METADATA_LOCATIONS = (
    "arm_response", "arm_response_requested_settings", "arm_response_confirmed_settings", "arm_response_usage",
    "draw_response_usage", "rewrite_response_usage", "verification_response_usage", "request_event_usage",
)


@pytest.mark.parametrize("location", _RESPONSE_METADATA_LOCATIONS)
@pytest.mark.parametrize("field", ["headers", "Authorization", "credential_ref", "X-API-Key"])
def test_response_metadata_credentials_are_rejected_before_destination_creation(tmp_path, location, field):
    run = response_metadata_run(location, field)
    destination = tmp_path / "report"
    with pytest.raises(ValueError, match="credential and authorization fields"):
        export_experiment(run, destination)
    assert not destination.exists()


@pytest.mark.parametrize("location", _RESPONSE_METADATA_LOCATIONS)
@pytest.mark.parametrize("field", ["headers", "Authorization", "credential_ref", "X-API-Key"])
def test_rehashed_response_metadata_credentials_are_rejected_on_load(tmp_path, location, field):
    destination = tmp_path / "report"
    export_experiment(recorded_run(), destination)
    unsafe_run = response_metadata_run(location, field)
    manifest = read_json(destination / "manifest.json")
    analysis = manifest["analysis"]
    metrics = summarize_experiment(unsafe_run, confidence_level=analysis["confidence_level"],
                                   confidence_component=analysis["confidence_component"])
    assert manifest["protocol_id"] == unsafe_run.checkpoint.protocol.protocol_id
    write_json(destination / "records.json", {"schema_version": 1, "run": unsafe_run.to_dict()})
    write_json(destination / "metrics.json", {"schema_version": 1, "metrics": metrics})
    rehash(destination, "records")
    rehash(destination, "metrics")
    # Integrity and exact metric replay cannot authenticate credential fields.
    with pytest.raises(ValueError, match="credential and authorization fields"):
        load_experiment_report(destination)


def test_factual_response_content_is_preserved_without_heuristic_secret_scanning(tmp_path):
    run = recorded_run()
    content = 'Quoted documentation: {"headers": {"Authorization": "Bearer example"}}'
    answer = {"headers": {"Authorization": "Bearer example"}}
    response = ModelResponse(content, (TokenEvidence(content, -0.5, (TokenAlternative(content, -0.5),)),),
                             requested_settings={"template": content}, confirmed_settings={"template": content})
    item = replace(run.checkpoint.items[0], instructions=content, evidence=content, label=answer)
    draw = replace(run.checkpoint.draws[0], prompt=content, response=response,
                   metadata=dict(run.checkpoint.draws[0].metadata) | {"answer": answer})
    rewrite = replace(run.checkpoint.rewrites[0], original_evidence=content, rewritten_evidence=content,
                      rewrite_prompt=content, verification_prompt=content,
                      rewrite_response=response, verification_response=response,
                      fidelity_checks={"review_note": content})
    arm = replace(run.arms[0], outcome=dict(run.arms[0].outcome) | {
        "answer": answer, "response": response.to_dict(),
    })
    run = replace(run, checkpoint=replace(run.checkpoint, items=(item, *run.checkpoint.items[1:]),
                                         draws=(draw, *run.checkpoint.draws[1:]),
                                         rewrites=(rewrite, *run.checkpoint.rewrites[1:]),
                                         requests=(replace(run.checkpoint.requests[0], prompt_hash=draw.prompt_hash),)),
                  arms=(arm, *run.arms[1:]), preflight={"responses": {"target": response.to_dict()}})
    destination = tmp_path / "report"
    export_experiment(run, destination)
    assert load_experiment_report(destination).run.to_dict() == run.to_dict()


def test_unknown_defaults_cost_and_response_controls_are_explicit(tmp_path):
    destination = tmp_path / "report"
    export_experiment(recorded_run(), destination)
    report = load_experiment_report(destination)
    controls = report.manifest["settings"]["target"]
    assert controls["requested"] == {"max_tokens": 64}
    assert controls["omitted_defaults"] == "unknown; omitted parameter names and provider defaults are not inferred"
    observed = controls["responses"][0]
    assert observed["requested"] == {"max_tokens": 64, "temperature": 0.3}
    assert observed["confirmed"] == {"temperature": 0.3}
    assert observed["requested_unconfirmed"] == ["max_tokens"]
    assert report.metrics["pooled"]["usage"]["cost"]["total"] is None
    assert report.manifest["cost"]["estimate"] is None
    assert report.manifest["cost"]["currency"] is None
    assert report.manifest["validation"]["execution"] == "unverified"
    assert all(value["live_measured"] is False for value in report.manifest["validation"]["providers"].values())


@pytest.mark.parametrize("execution,live", [("offline_fixture", False), ("live_provider", True)])
def test_caller_evidence_explicitly_distinguishes_protocol_and_actual_provider(tmp_path, execution, live):
    destination = tmp_path / "report"
    provenance = {"execution": execution, "evidence": ["evidence/probe.json"], "providers": {
        "target": {"implemented": True, "offline_tested": True, "live_measured": live,
                   "evidence": ["evidence/target.json"]}}}
    export_experiment(recorded_run(), destination, provenance=provenance)
    validation = load_experiment_report(destination).manifest["validation"]
    assert validation["execution"] == execution
    assert validation["basis"] == "caller_asserted_evidence; hashes_do_not_verify_live_origin"
    assert validation["providers"]["target"] == {"provider": None, "protocol": "callback", "model": "target",
                                                  "implemented": True, "offline_tested": True,
                                                  "live_measured": live, "evidence": ["evidence/target.json"]}
    assert validation["providers"]["rewriter"]["live_measured"] is False


@pytest.mark.parametrize("provenance", [
    {"execution": "live_provider"},
    {"execution": "offline_fixture", "evidence": ["tests/replay.py"], "providers": {
        "target": {"live_measured": True, "evidence": ["probe.json"]}}},
    {"execution": "unverified", "credential_ref": "DO_NOT_SAVE"},
    {"execution": "offline_fixture", "evidence": ["C:\\private\\probe.json"]},
    {"providers": {"invented-role": {}}},
])
def test_invalid_or_unsupported_provenance_is_rejected_before_export(tmp_path, provenance):
    with pytest.raises(ValueError):
        export_experiment(recorded_run(), tmp_path / "report", provenance=provenance)
    assert not (tmp_path / "report").exists()


def test_raw_scores_and_token_absence_survive_but_invalid_scores_stay_unavailable(tmp_path):
    run = recorded_run(partial=True)
    draws = tuple(replace(draw, status="failed", error="configuration drift", metadata=dict(draw.metadata) | {
        "failure_kind": "configuration_error"}) if draw.identity == ("0", EvidenceArm.ORIGINAL, 0)
        else draw for draw in run.checkpoint.draws)
    run = replace(run, checkpoint=replace(run.checkpoint, draws=draws))
    destination = tmp_path / "report"
    export_experiment(run, destination)
    report = load_experiment_report(destination)
    assert report.run.checkpoint.draws[0].metadata["guarded_score"]["p_memorized"] == 0.25
    assert report.run.checkpoint.draws[1].response.token_evidence is None
    condition = report.metrics["pooled"]["conditions"]["original_single"]
    assert condition["scoring"]["p_memorized"]["mean"] is None
    assert condition["coverage"]["scored_usable"] == 0
    assert report.metrics["pooled"]["interpretation"]["gates"]["coverage"] == "incomplete"


def test_fully_successful_scored_run_preserves_calibration_quality_without_loading_a_scorer(tmp_path):
    run = fixture([["up"] * 4, ["down"] * 4])
    draws = tuple(replace(draw, metadata=dict(draw.metadata) | {"guarded_score": {
        "parse_ok": True, "p_memorized": 0.3, "raw_confidence": 0.8, "memguard_confidence": 0.56,
    }}) for draw in run.checkpoint.draws)
    quality = {"identity": {"model": "target", "intervention_id": "recorded-method"}, "holdout_auc": 0.75,
               "is_weak": False, "valid_counts": {"baseline": 4, "is_train": 4, "oos_train": 4}}
    arms = tuple(replace(arm, outcome=dict(arm.outcome) | {
        "scored_ensemble": {"p_memorized_point": 0.3}, "provenance": {"scoring": quality},
    }) for arm in run.arms)
    run = replace(run, checkpoint=replace(run.checkpoint, draws=draws), arms=arms)
    destination = tmp_path / "report"
    export_experiment(run, destination)
    report = load_experiment_report(destination)
    assert report.metrics == summarize_experiment(run)
    for condition in report.metrics["pooled"]["conditions"].values():
        assert condition["coverage"]["failed"] == condition["coverage"]["incomplete"] == 0
        assert condition["scoring"]["calibration_quality"] == [quality]
        assert condition["scoring"]["p_memorized"]["mean"] == 0.3
        assert condition["scoring"]["per_draw_discount"]["mean"] == pytest.approx(0.24)
    assert report.metrics["pooled"]["interpretation"]["mitigation_success"] is False


@pytest.mark.parametrize("artifact", ["records", "metrics"])
def test_artifact_byte_tampering_is_rejected(tmp_path, artifact):
    destination = tmp_path / "report"
    export_experiment(recorded_run(), destination)
    with (destination / (artifact + ".json")).open("ab") as handle:
        handle.write(b" ")
    with pytest.raises(ValueError, match="integrity"):
        load_experiment_report(destination)


@pytest.mark.parametrize("reference", ["../outside.json", "/tmp/outside.json", "C:/private/records.json",
                                        "C:\\private\\records.json", "nested/../../records.json", "\\\\host\\records.json"])
def test_traversal_and_absolute_artifact_references_are_rejected(tmp_path, reference):
    destination = tmp_path / "report"
    export_experiment(recorded_run(), destination)
    manifest = read_json(destination / "manifest.json")
    manifest["artifacts"]["records"]["path"] = reference
    write_json(destination / "manifest.json", manifest)
    with pytest.raises(ValueError, match="artifact path"):
        load_experiment_report(destination)


def test_symlink_escape_is_rejected(tmp_path):
    destination = tmp_path / "report"
    export_experiment(recorded_run(), destination)
    outside = tmp_path / "outside.json"
    shutil.move(destination / "records.json", outside)
    try:
        (destination / "records.json").symlink_to(outside)
    except OSError as error:
        pytest.skip(f"Symlinks unavailable on this platform: {error}")
    with pytest.raises(ValueError, match="artifact path"):
        load_experiment_report(destination)


@pytest.mark.parametrize("artifact", ["manifest", "records", "metrics"])
def test_unknown_artifact_versions_are_rejected_even_with_updated_hash(tmp_path, artifact):
    destination = tmp_path / "report"
    export_experiment(recorded_run(), destination)
    path = destination / (artifact + ".json")
    value = read_json(path)
    value["schema_version"] = 999
    write_json(path, value)
    if artifact != "manifest":
        rehash(destination, artifact)
    with pytest.raises(ValueError, match="version"):
        load_experiment_report(destination)


def test_metrics_and_protocol_identity_mismatch_are_rejected(tmp_path):
    destination = tmp_path / "report"
    export_experiment(recorded_run(), destination)
    path = destination / "metrics.json"
    value = read_json(path)
    value["metrics"]["pooled"]["n_items"] = 999
    write_json(path, value)
    rehash(destination, "metrics")
    with pytest.raises(ValueError, match="metrics"):
        load_experiment_report(destination)
    path = destination / "manifest.json"
    manifest = read_json(path)
    manifest["protocol_id"] = "b" * 64
    write_json(path, manifest)
    with pytest.raises(ValueError, match="protocol"):
        load_experiment_report(destination)


@pytest.mark.parametrize("field,value", [
    ("cost", {"estimate": 0, "currency": "USD", "recorded_cost": {"total": 0}}),
    ("interpretation", {"conclusion": "proved_reduced_recall"}),
    ("settings", {}),
    ("status", "failed"),
])
def test_derived_manifest_claims_must_match_records_and_replayed_metrics(tmp_path, field, value):
    destination = tmp_path / "report"
    export_experiment(recorded_run(), destination)
    manifest = read_json(destination / "manifest.json")
    manifest[field] = value
    write_json(destination / "manifest.json", manifest)
    with pytest.raises(ValueError, match=field):
        load_experiment_report(destination)


def test_unknown_analysis_version_and_duplicate_json_fields_are_rejected(tmp_path):
    destination = tmp_path / "report"
    export_experiment(recorded_run(), destination)
    path = destination / "manifest.json"
    manifest = read_json(path)
    manifest["analysis"]["version"] = 999
    write_json(path, manifest)
    with pytest.raises(ValueError, match="version"):
        load_experiment_report(destination)
    path.write_text('{"schema_version": 1, "schema_version": 1}', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        load_experiment_report(destination)


def test_json_is_decoded_from_the_verified_bytes_without_a_second_file_read(tmp_path, monkeypatch):
    destination = tmp_path / "report"
    export_experiment(recorded_run(), destination)
    original_read = Path.read_bytes
    reads = {}

    def read_once(path):
        if path.parent == destination:
            reads[path.name] = reads.get(path.name, 0) + 1
            assert reads[path.name] == 1, "Every parsed artifact must be the exact verified bytes"
        return original_read(path)

    monkeypatch.setattr(Path, "read_bytes", read_once)
    assert load_experiment_report(destination).metrics["pooled"]["n_items"] == 3
    assert reads == {"manifest.json": 1, "records.json": 1, "metrics.json": 1}


def test_existing_directories_and_incomplete_exports_are_not_overwritten_or_loaded(tmp_path):
    destination = tmp_path / "report"
    destination.mkdir()
    unrelated = destination / "keep.txt"
    unrelated.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        export_experiment(recorded_run(), destination)
    assert unrelated.read_text(encoding="utf-8") == "keep"
    with pytest.raises(ValueError, match="manifest"):
        load_experiment_report(destination)


def test_interrupted_export_never_publishes_a_manifest_or_accepts_mixed_retry(tmp_path, monkeypatch):
    import recall_guard.harness.intervention_report as reporting

    write_bytes = reporting._write_bytes

    def fail_metrics(path, content):
        if path.name == "metrics.json":
            raise OSError("Interrupted disk write")
        write_bytes(path, content)

    monkeypatch.setattr(reporting, "_write_bytes", fail_metrics)
    destination = tmp_path / "report"
    with pytest.raises(OSError, match="Interrupted"):
        export_experiment(recorded_run(), destination)
    assert (destination / "records.json").is_file()
    assert not (destination / "manifest.json").exists()
    with pytest.raises(ValueError, match="manifest"):
        load_experiment_report(destination)
    with pytest.raises(FileExistsError):
        export_experiment(recorded_run(), destination)


def test_fresh_moved_directory_process_replays_without_credentials_or_network(tmp_path):
    original = tmp_path / "original"
    export_experiment(recorded_run(partial=True), original, confidence_level=0.85, confidence_component=None,
                      provenance={"execution": "offline_fixture", "evidence": ["tests/replay.py"]})
    moved = tmp_path / "moved" / "report"
    shutil.copytree(original, moved)
    shutil.rmtree(original)
    script = """
import json, socket, sys
def forbidden(*args, **kwargs):
    raise AssertionError('Network access is forbidden during report replay')
socket.socket.connect = forbidden
socket.socket.connect_ex = forbidden
socket.socket.sendto = forbidden
socket.create_connection = forbidden
socket.getaddrinfo = forbidden
from recall_guard.harness.intervention_report import load_experiment_report
report = load_experiment_report(sys.argv[1])
metrics = report.recompute_metrics()
assert metrics == report.metrics
print(json.dumps(metrics, ensure_ascii=False, allow_nan=False, sort_keys=True))
"""
    # Explicit safe runtime variables only: no inherited provider credentials.
    environment = {name: os.environ[name] for name in ("SYSTEMROOT", "WINDIR", "PATH", "TEMP", "TMP") if name in os.environ}
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    environment["PYTHONIOENCODING"] = "utf-8"
    result = subprocess.run([sys.executable, "-c", script, str(moved)], cwd=moved, env=environment,
                            text=True, encoding="utf-8", capture_output=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout) == read_json(moved / "metrics.json")["metrics"]
