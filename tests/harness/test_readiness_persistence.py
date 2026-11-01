"""Offline v2 contracts; no clients, credential lookup or provider work."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from recall_guard.core.completion import FeatureChoices, ModelConfig, ModelResponse
from recall_guard.core.ensemble import EnsembleSpec
from recall_guard.core.readiness_evidence import PartialResponse, ProbeEvidence
from recall_guard.core.readiness_types import ReadinessObservation, ReadinessPolicy
from recall_guard.core.request_control import RequestEvent
from recall_guard.harness import intervention_types as t
from recall_guard.harness.scorer_client import CalibrationIdentity

STAMP = "2026-11-01T10:00:00+00:00"


def config(**kwargs):
    return ModelConfig("openai_chat", "https://example.test/v1/chat/completions", "model", **kwargs)


def observation():
    return ReadinessObservation("op-1", "probe", "model", STAMP, 1.0, "unknown_outcome", call_id=1)


def plan(*, paired=False):
    from recall_guard.harness.workflow_plan import CompletionFallbackPolicy, WorkflowPlan, WorkflowRequest

    target = config(parameters={"logprobs": True, "top_logprobs": 2, "custom": [True, None]},
                    readiness=ReadinessPolicy(), credential_ref="env:KEY")
    roles = {"target": target}
    options = {}
    if paired:
        roles.update(rewriter=config(), verifier=config())
        options = {"rewrite_policy_id": "caller-policy", "ensemble": EnsembleSpec(draws=1, min_parsed=1)}
    requested = WorkflowRequest(roles, FeatureChoices(mia_scoring=True), **options)
    selected = WorkflowRequest(roles | {"target": replace(target, parameters={"custom": [True, None]})},
                               FeatureChoices(rewriting=paired), **options)
    fallback = CompletionFallbackPolicy(FeatureChoices(rewriting=paired), omit_logprob_controls=True)
    return WorkflowPlan(requested, selected, STAMP, fallback_policy=fallback,
                        omitted_controls=("logprobs", "top_logprobs"), reason="missing_token_evidence")


def protocol(**kwargs):
    values = dict(target_model=config(), rewriter_model=config(), verifier_model=config(),
                  data_hashes={"evaluation": "a" * 64}, cohort_definitions={"general": {}},
                  draws=1, bootstrap_seed=1, bootstrap_count=10, task_metric="accuracy",
                  utility_tolerance=0.1, audit_selection={"method": "all"}, request_budget=4,
                  rewrite_method_version="rewrite-v1")
    return t.ExperimentProtocol(**(values | kwargs))


def artifact(*, paired=False):
    p = plan(paired=paired)
    probe = ProbeEvidence(p.requested.roles["target"], p.requested.roles["target"].inspect(p.requested.features),
                          PartialResponse("parțial", requested_settings=p.requested.roles["target"].parameters),
                          (observation(),))
    event = RequestEvent(1, "probe", "model", 1, 0.0, STAMP, status="failed", purpose="probe", operation_id="op-1")
    request = t.RequestRecord("prep-1", event, "target", t.text_hash("probe"), phase="preparation")
    return t.PreparationArtifact("prep-1", p, (request,), {"target": (probe,)}, stopping_reason="unknown_outcome")


def test_optional_readiness_preserves_legacy_export_and_calibration_identity():
    old = config(parameters={"custom": [True, None]})
    new = replace(old, readiness=ReadinessPolicy())
    assert "readiness" not in old.to_dict()
    assert new.to_dict().pop("readiness") == ReadinessPolicy().to_dict()
    assert CalibrationIdentity.from_config(old, "original") == CalibrationIdentity.from_config(new, "original")
    assert ModelConfig.from_dict(new.to_dict()).to_dict() == new.to_dict()
    assert ModelResponse.from_dict(ModelResponse("ok").to_dict()).to_dict() == ModelResponse("ok").to_dict()


def test_trace_roundtrip_and_legacy_absence():
    response = ModelResponse("ok", readiness_trace=(observation(),))
    assert ModelResponse.from_dict(json.loads(json.dumps(response.to_dict()))) == response
    assert "readiness_trace" not in ModelResponse("ok").to_dict()
    event = RequestEvent(1, "probe", "model", 1, 0.0, STAMP)
    assert "purpose" not in event.to_dict()
    extended = replace(event, purpose="status", operation_id="op-1", request_id="request-1")
    assert RequestEvent.from_dict(extended.to_dict()) == extended


def test_plan_roundtrip_omits_credentials_but_keeps_runtime_references():
    p = plan()
    payload = json.loads(json.dumps(p.to_dict()))
    restored = type(p).from_dict(payload)
    assert restored.plan_id == p.plan_id
    assert restored.to_dict() == payload
    assert p.selected.roles["target"].credential_ref == "env:KEY"
    assert restored.selected.roles["target"].credential_ref is None
    assert "KEY" not in json.dumps(payload)
    assert restored.mia_unavailable_reason
    assert restored.selected.method == "original_single"
    assert restored.policy_id == p.fallback_policy.policy_id


@pytest.mark.parametrize("parameters", [{"custom": [1, None]}, {"custom": [True]}, {"custom": [True, None], "extra": None}])
def test_fallback_cannot_change_non_logprob_json_identity(parameters):
    p = plan()
    changed = replace(p.selected, roles={"target": replace(p.selected.roles["target"], parameters=parameters)})
    with pytest.raises(ValueError, match="parameters|configuration"):
        replace(p, selected=changed)


@pytest.mark.parametrize("changes", [{"omitted_controls": ()}, {"fallback_policy": None}, {"reason": "timeout"}])
def test_plan_rejects_unapproved_or_misrepresented_fallback(changes):
    with pytest.raises(ValueError):
        replace(plan(), **changes)


def test_partial_preparation_roundtrip_and_binding_arithmetic():
    a = artifact()
    payload = json.loads(json.dumps(a.to_dict()))
    restored = t.PreparationArtifact.from_dict(payload)
    assert restored.to_dict() == payload
    assert restored.artifact_hash == a.artifact_hash
    partial = restored.evidence["target"][0].response
    assert isinstance(partial, PartialResponse) and not partial.usable_answer
    binding = a.bind(total_request_cap=4)
    assert (binding.preparation_used_requests, binding.remaining_evaluation_requests) == (1, 3)
    assert binding.plan_id == a.plan.plan_id
    binding.validate_artifact(restored)
    with pytest.raises(ValueError):
        replace(binding, remaining_evaluation_requests=4)
    with pytest.raises(ValueError):
        a.bind(total_request_cap=0)


def test_v2_protocol_and_checkpoint_bind_the_same_plan_and_total_cap():
    a = artifact(paired=True)
    p = protocol(schema_version=2, target_model=a.plan.selected.roles["target"],
                 workflow_plan=a.plan, preparation=a.bind(total_request_cap=4), ensemble_settings={"min_parsed": 1})
    restored = t.ExperimentProtocol.from_dict(json.loads(json.dumps(p.to_dict())))
    assert restored.protocol_id == p.protocol_id
    assert restored.to_dict() == p.to_dict()
    checkpoint = t.ExperimentCheckpoint(p, (), schema_version=2)
    assert t.ExperimentCheckpoint.from_dict(checkpoint.to_dict()).to_dict() == checkpoint.to_dict()
    with pytest.raises(ValueError, match="cap|budget"):
        replace(p, request_budget=5)
    with pytest.raises(ValueError, match="schema|version"):
        t.ExperimentCheckpoint(p, ())


@pytest.mark.parametrize("field,value", [("readiness", {}), ("workflow_plan", None), ("fallback_policy", None), ("preparation", None)])
def test_v1_rejects_new_fields_even_explicit_null(field, value):
    payload = protocol().to_dict()
    if field == "readiness":
        payload["target_model"][field] = value
    else:
        payload[field] = value
    with pytest.raises((ValueError, TypeError)):
        t.ExperimentProtocol.from_dict(payload)


def test_v1_direct_contract_rejects_readiness_and_preserves_exact_bytes():
    p = protocol()
    original = json.dumps(p.to_dict(), ensure_ascii=False, sort_keys=True)
    assert json.dumps(t.ExperimentProtocol.from_dict(p.to_dict()).to_dict(), ensure_ascii=False, sort_keys=True) == original
    with pytest.raises(ValueError, match="version|schema"):
        replace(p, target_model=replace(p.target_model, readiness=ReadinessPolicy()))


@pytest.mark.parametrize("path", [(), ("plan",), ("evidence", "target", 0),
                                      ("evidence", "target", 0, "response"),
                                      ("evidence", "target", 0, "observations", 0)])
def test_v2_loaders_reject_unknown_runtime_metadata(path):
    payload = artifact().to_dict()
    value = payload
    for key in path:
        value = value[key]
    value["client"] = "must not be accepted"
    with pytest.raises((ValueError, TypeError)):
        t.PreparationArtifact.from_dict(payload)


def test_preparation_and_evaluation_namespaces_cannot_overlap():
    a = artifact(paired=True)
    p = protocol(schema_version=2, target_model=a.plan.selected.roles["target"],
                 workflow_plan=a.plan, preparation=a.bind(total_request_cap=4), ensemble_settings={"min_parsed": 1})
    with pytest.raises(ValueError, match="phase|execution"):
        t.ExperimentCheckpoint(p, (), requests=a.requests, schema_version=2)


def test_v2_is_not_silently_executed_by_legacy_runner(tmp_path):
    from recall_guard.harness.intervention import ExperimentRunner

    calls = []
    with pytest.raises(ValueError, match="[Vv]ersion.2"):
        ExperimentRunner(protocol(schema_version=2), (), lambda budget: calls.append(budget), tmp_path / "journal.json")
    assert calls == []


@pytest.mark.parametrize("container,path", [
    ("config", ("readiness",)), ("config", ()),
    ("response", ()), ("response", ("readiness_trace", 0)),
    ("plan", ("requested",)), ("plan", ("requested", "features")),
    ("plan", ("fallback_policy",)), ("plan", ("selected", "roles", "target")),
])
def test_nested_codecs_do_not_accept_undeclared_fields(container, path):
    objects = {"config": config(readiness=ReadinessPolicy()),
               "response": ModelResponse("ok", readiness_trace=(observation(),)), "plan": plan()}
    original = objects[container]
    payload = original.to_dict()
    node = payload
    for key in path:
        node = node[key]
    node["callback"] = "runtime callback"
    with pytest.raises((ValueError, TypeError)):
        type(original).from_dict(payload)


def test_pending_receipt_and_verified_policy_roundtrip():
    from recall_guard.core.readiness_types import PendingReceipt, PendingStatusPolicy

    route = PendingStatusPolicy("fixture", "model", config().endpoint, "https://example.test/status/{request_id}",
                                "NVCF-REQID", True)
    policy = ReadinessPolicy(pending_status=route, max_poll_requests=2)
    selected = config(readiness=policy)
    restored = ModelConfig.from_dict(selected.to_dict())
    assert restored.readiness == policy
    receipt = PendingReceipt("op", "model", "12345678-1234-1234-1234-123456789abc", STAMP)
    assert PendingReceipt.from_dict(receipt.to_dict()) == receipt
    with pytest.raises(ValueError):
        PendingReceipt.from_dict(receipt.to_dict() | {"complete": True})
    with pytest.raises(ValueError):
        replace(selected, model="other-model")


def test_complete_missing_token_probe_evidence_stays_complete_but_not_ready():
    target = config()
    probe = ProbeEvidence(target, target.inspect(FeatureChoices(mia_scoring=True)), ModelResponse("answer"), (observation(),))
    restored = ProbeEvidence.from_dict(probe.to_dict())
    assert restored.complete and not restored.ready
    assert restored.response.content == "answer"
    assert restored.report.mia_unavailable_reason


@pytest.mark.parametrize("mutation", [
    lambda value: value["response"].update(complete=1),
    lambda value: value["response"].update(complete=True),
    lambda value: value["response"].update(client={}),
])
def test_partial_probe_cannot_be_promoted_or_contain_runtime_objects(mutation):
    payload = artifact().evidence["target"][0].to_dict()
    mutation(payload)
    with pytest.raises((ValueError, TypeError)):
        ProbeEvidence.from_dict(payload)


def test_new_contracts_reject_runtime_state_without_traversing_it():
    from recall_guard.harness.workflow_plan import WorkflowRequest

    with pytest.raises((TypeError, ValueError)):
        WorkflowRequest({"target": object()}, FeatureChoices())
    with pytest.raises((TypeError, ValueError)):
        replace(plan().requested, rewrite_policy_id=object())
    with pytest.raises((TypeError, ValueError)):
        replace(plan(), fallback_policy=object())
    with pytest.raises((TypeError, ValueError)):
        replace(artifact(), evidence={"target": (object(),)})


@pytest.mark.parametrize("parameters", [{"custom": [True, None]}, {"custom": [1, None]}, {"custom": [True]},
                                       {"custom": [True, None], "extra": None}])
def test_plan_identity_preserves_json_types_and_absent_null(parameters):
    from recall_guard.harness.workflow_plan import WorkflowPlan, WorkflowRequest

    request = WorkflowRequest({"target": config(parameters=parameters)}, FeatureChoices())
    p = WorkflowPlan(request, request, STAMP)
    assert type(p).from_dict(p.to_dict()).plan_id == p.plan_id
    baseline = WorkflowRequest({"target": config(parameters={"custom": [True, None]})}, FeatureChoices())
    assert (WorkflowPlan(baseline, baseline, STAMP).plan_id == p.plan_id) is (parameters == {"custom": [True, None]} and parameters["custom"][0] is True)


def test_v2_failed_draw_keeps_partial_output_and_never_counts_it_complete():
    draw = t.DrawRecord("item", "original", 0, status="failed", error="interrupted", partial_response=PartialResponse("parțial"))
    restored = t.DrawRecord.from_dict(json.loads(json.dumps(draw.to_dict())))
    assert restored.response is None and not restored.partial_response.usable_answer
    with pytest.raises(ValueError):
        replace(draw, status="completed", error=None, response=ModelResponse("ok"), prompt="prompt")


def test_v2_checkpoint_charges_only_remaining_evaluation_allowance_and_separates_ids():
    a = artifact(paired=True)
    p = protocol(schema_version=2, target_model=a.plan.selected.roles["target"], workflow_plan=a.plan,
                 preparation=a.bind(total_request_cap=2), request_budget=2, ensemble_settings={"min_parsed": 1})
    event = RequestEvent(1, "probe", "model", 1, 0.0, STAMP, status="failed", purpose="probe")
    request = t.RequestRecord("eval-1", event, "target", t.text_hash("probe"), phase="evaluation")
    checkpoint = t.ExperimentCheckpoint(p, (), requests=(request,), schema_version=2)
    restored = t.ExperimentCheckpoint.from_dict(checkpoint.to_dict())
    assert restored.requests[0].identity == ("evaluation", "eval-1", 1)
    assert a.requests[0].identity == ("preparation", "prep-1", 1)
    with pytest.raises(ValueError, match="budget"):
        replace(checkpoint, requests=(request, replace(request, event=replace(event, call_id=2))))
    with pytest.raises(ValueError, match="execution"):
        replace(checkpoint, requests=(replace(request, execution_id="prep-1"),))
    with pytest.raises(ValueError, match="Version-2"):
        restored.validate_restart(p, ())


@pytest.mark.parametrize("field", ["purpose", "operation_id", "request_id"])
def test_version_one_rejects_new_request_fields_even_null(field):
    event = RequestEvent(1, "probe", "model", 1, 0.0, STAMP)
    request = t.RequestRecord("eval", event, "target", t.text_hash("probe"))
    payload = t.ExperimentCheckpoint(protocol(), (), requests=(request,)).to_dict()
    payload["requests"][0]["event"][field] = None
    with pytest.raises(ValueError, match="version 2"):
        t.ExperimentCheckpoint.from_dict(payload)


def test_version_one_rejects_readiness_trace_even_on_failed_draws():
    item = t.ExperimentItem("i", "instructions", "evidence", "2026-01-01", 1, "general")
    draw = t.DrawRecord("i", "original", 0, status="failed", error="missing evidence",
                        response=ModelResponse("answer", readiness_trace=(observation(),)))
    with pytest.raises(ValueError, match="version 2"):
        t.ExperimentCheckpoint(protocol(), (item,), draws=(draw,))


def test_binding_rejects_changed_artifact_and_paired_protocol_rejects_library_shape():
    a = artifact()
    binding = a.bind(total_request_cap=4)
    with pytest.raises(ValueError, match="binding"):
        binding.validate_artifact(replace(a, stopping_reason="cancelled"))
    with pytest.raises(ValueError, match="Paired"):
        protocol(schema_version=2, target_model=a.plan.selected.roles["target"], workflow_plan=a.plan, preparation=binding)


def test_legacy_report_cannot_publish_v2_plan_fields_under_v1_envelope(tmp_path):
    from recall_guard.harness.intervention_journal import ExperimentRun
    from recall_guard.harness.intervention_report import export_experiment

    checkpoint = t.ExperimentCheckpoint(protocol(schema_version=2), (), schema_version=2)
    destination = tmp_path / "report"
    with pytest.raises(ValueError, match="[Vv]ersion.2"):
        export_experiment(ExperimentRun(checkpoint), destination)
    assert not destination.exists()


def test_legacy_report_replay_rejects_v2_checkpoint_envelope():
    from recall_guard.harness.intervention_journal import ExperimentRun
    from recall_guard.harness.intervention_report import _check_snapshot_metadata

    checkpoint = t.ExperimentCheckpoint(protocol(schema_version=2), (), schema_version=2)
    with pytest.raises(ValueError, match="Unsupported report version"):
        _check_snapshot_metadata(ExperimentRun(checkpoint).to_dict())


@pytest.mark.parametrize("name", ["anthropic-evaluation", "anthropic-evaluation-live"])
def test_saved_version_one_reports_replay_exact_bytes_and_metrics(name):
    from recall_guard.harness.intervention_report import _json_bytes, load_experiment_report

    directory = Path(__file__).resolve().parents[2] / "data" / "experiments" / "2026-10-31-completion-api" / name
    report = load_experiment_report(directory)
    assert report.recompute_metrics() == report.metrics
    assert _json_bytes({"schema_version": 1, "run": report.run.to_dict()}) == (directory / "records.json").read_bytes()
    assert _json_bytes({"schema_version": 1, "metrics": report.metrics}) == (directory / "metrics.json").read_bytes()


@pytest.mark.parametrize("settings", ["requested_settings", "confirmed_settings"])
def test_v1_checkpoint_rejects_stream_response_metadata(settings):
    directory = Path(__file__).resolve().parents[2] / "data" / "experiments" / "2026-10-31-completion-api" / "anthropic-evaluation-live"
    payload = json.loads((directory / "records.json").read_text(encoding="utf-8"))["run"]["checkpoint"]
    response = next(draw["response"] for draw in payload["draws"] if draw["response"] is not None)
    response[settings]["stream"] = True
    with pytest.raises(ValueError, match="[Ss]tream.*version 2"):
        t.ExperimentCheckpoint.from_dict(payload)

