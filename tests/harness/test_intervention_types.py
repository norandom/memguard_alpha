"""Offline persistence and restart invariants for completion experiments."""

import json
from dataclasses import FrozenInstanceError, replace

import pytest

from recall_guard.core.completion import ModelConfig, ModelResponse, TokenAlternative, TokenEvidence
from recall_guard.core.consensus import Tail
from recall_guard.core.ensemble import EnsembleSpec, LocationMode, MultimodalAction, ReferenceMode, reduce_draws
from recall_guard.core.request_control import RequestEvent
from recall_guard.harness import intervention_types as t


def protocol(**overrides):
    values = {
        "target_model": ModelConfig("callback", None, "target", credential_ref="env:TARGET_KEY",
                                    parameters={"max_tokens": 32}),
        "rewriter_model": ModelConfig("callback", None, "rewriter"),
        "verifier_model": ModelConfig("callback", None, "verifier"),
        "data_hashes": {"evaluation": "a" * 64, "cutoffs": "b" * 64},
        "cohort_definitions": {"general": {"historical_claims": False}},
        "draws": 2,
        "bootstrap_seed": 17,
        "bootstrap_count": 500,
        "task_metric": "direction_accuracy_v1",
        "utility_tolerance": 0.02,
        "audit_selection": {"method": "stratified", "seed": 8, "per_status": 2},
        "request_budget": 12,
        "rewrite_method_version": "narrative-v1",
        "ensemble_settings": {"min_usable": 1, "reduction": "majority_v1"},
    }
    return t.ExperimentProtocol(**(values | overrides))


def item(**overrides):
    return t.ExperimentItem(**({
        "item_id": "item-1", "instructions": "Return a direction in JSON.",
        "evidence": "Acme rose 2% on 2024-01-02. Café shares were flat.",
        "as_of": "2024-01-02", "label": {"direction": 1}, "cohort": "general",
    } | overrides))


def event(call_id=1, **overrides):
    return RequestEvent(**({
        "call_id": call_id, "stage": "target", "model": "target", "attempt": 1,
        "started_at": 10.0, "timestamp": "2026-10-30T18:00:00+00:00",
        "status": "succeeded", "latency_s": 0.25,
    } | overrides))


def request(call_id=1, **overrides):
    return t.RequestRecord(**({
        "execution_id": "execution-1", "event": event(call_id), "model_role": "target",
        "prompt_hash": t.text_hash("original prompt"), "item_id": "item-1", "arm": "original",
        "draw_index": call_id - 1,
    } | overrides))


def draw(draw_index=0, **overrides):
    return t.DrawRecord(**({
        "item_id": "item-1", "arm": "original", "draw_index": draw_index,
        "status": "completed", "prompt": "original prompt", "response": ModelResponse('{"direction":1}'),
    } | overrides))


def checkpoint(**overrides):
    return t.ExperimentCheckpoint(**({
        "protocol": protocol(), "items": (item(),), "status": "failed",
        "draws": (draw(), draw(1, status="failed", response=None, error="quota exhausted")),
        "requests": (request(), request(2, event=event(2, status="failed", error="quota exhausted"))),
    } | overrides))


def test_protocol_hash_is_stable_detached_and_credential_free():
    p = protocol()
    data = json.loads(json.dumps(p.to_dict(), ensure_ascii=False))
    restored = t.ExperimentProtocol.from_dict(data)
    assert restored.protocol_id == p.protocol_id
    assert restored.target_model.credential_ref is None
    assert "credential_ref" not in json.dumps(data)
    assert "TARGET_KEY" not in json.dumps(data)
    assert replace(p, target_model=replace(p.target_model, credential_ref="OTHER_KEY")).protocol_id == p.protocol_id
    assert protocol(data_hashes={"cutoffs": "b" * 64, "evaluation": "a" * 64}).protocol_id == p.protocol_id
    assert replace(p, draws=3).protocol_id != p.protocol_id
    data["audit_selection"]["seed"] = 99
    assert p.audit_selection["seed"] == 8
    with pytest.raises(TypeError):
        p.cohort_definitions["general"]["historical_claims"] = True
    with pytest.raises(FrozenInstanceError):
        p.draws = 3


@pytest.mark.parametrize("name, enum_type", [
    ("tail", Tail), ("location_mode", LocationMode),
    ("multimodal_action", MultimodalAction), ("reference_mode", ReferenceMode),
])
def test_ensemble_contract_normalizes_every_serialized_enum_without_changing_identity(name, enum_type):
    for member in enum_type:
        original = protocol(ensemble_settings={"min_parsed": 1, name: member})
        serialized = json.loads(json.dumps(original.to_dict(), ensure_ascii=False, allow_nan=False))
        restored = t.ExperimentProtocol.from_dict(serialized)
        assert getattr(original.build_ensemble_spec(), name) is member
        assert getattr(restored.build_ensemble_spec(), name) is member
        assert restored.protocol_id == original.protocol_id
        assert restored.to_dict() == serialized
        assert type(restored.ensemble_settings[name]) is str


def test_ensemble_contract_preserves_all_options_and_shared_generation_controls():
    settings = {
        "draws": 8, "max_workers": 2, "min_parsed": 4, "grid": 0.1, "confidence": 0.9,
        "tail": "two_sided", "agreement_target": 0.2, "location_mode": "trimmed", "trim": 0.1,
        "multimodal_action": "raise", "mass_min": 0.4, "trough_steps": 2, "density_ratio": 5.0,
        "min_cluster_draws": 6, "min_cluster_density": 2.0, "max_total_requests": 24,
        "max_transport_failure_ratio": 0.5, "retain_draws": True, "reference_mode": "per_draw",
        "max_tokens": 32, "temperature": 0.5,
    }
    target = replace(protocol().target_model, parameters={"max_tokens": 32, "temperature": 0.5})
    original = protocol(draws=8, target_model=target, ensemble_settings=settings)
    serialized = json.loads(json.dumps(original.to_dict(), ensure_ascii=False, allow_nan=False))
    restored = t.ExperimentProtocol.from_dict(serialized)
    expected = EnsembleSpec(**(settings | {
        "tail": Tail.TWO_SIDED, "location_mode": LocationMode.TRIMMED,
        "multimodal_action": MultimodalAction.RAISE, "reference_mode": ReferenceMode.PER_DRAW,
    }))
    assert original.build_ensemble_spec() == restored.build_ensemble_spec() == expected
    assert restored.to_dict() == serialized and restored.protocol_id == original.protocol_id


@pytest.mark.parametrize("action", ["flag", "raise"])
def test_serialized_multimodal_policy_controls_the_existing_reducer(action):
    original = protocol(draws=8, ensemble_settings={
        "min_parsed": 1, "grid": 0.1, "mass_min": 0.4, "multimodal_action": action,
    })
    restored = t.ExperimentProtocol.from_dict(json.loads(json.dumps(original.to_dict())))
    draws = [ModelResponse(value) for value in ["-0.9"] * 4 + ["0.9"] * 4]
    spec = restored.build_ensemble_spec()
    if action == "raise":
        with pytest.raises(ValueError, match="axis.*separated clusters"):
            reduce_draws(draws, spec, decide=lambda response: "up",
                         components=lambda response: {"axis": float(response.content)})
    else:
        result = reduce_draws(draws, spec, decide=lambda response: "up",
                              components=lambda response: {"axis": float(response.content)})
        assert result.multimodal == ("axis",) and result.location == {}


def test_roundtrip_retains_inputs_rewrite_responses_token_evidence_and_partial_failures():
    p = protocol()
    evidence = (TokenEvidence("UP", -0.1, (TokenAlternative("UP", -0.1),)),)
    response = ModelResponse("UP", token_evidence=evidence, usage={"output_tokens": 1},
                             requested_settings={"max_tokens": 32}, finish_reason="stop", request_id="provider-1")
    rewrite = t.RewriteRecord(
        item_id="item-1", original_evidence=item().evidence, rewritten_evidence="Acme rose 2%. Café was flat.",
        status="failed", stage="verification", error="verifier response malformed",
        rewrite_prompt="preserve facts", verification_prompt="compare facts",
        rewrite_response=ModelResponse("Acme rose 2%. Café was flat."),
        verification_response=ModelResponse("not-json"), fidelity_checks={"numbers": True, "semantic": None},
    )
    requests = (
        request(), request(2, event=event(2, status="failed", error="quota exhausted", usage={"input_tokens": 7})),
        request(3, model_role="rewriter", event=event(3, stage="rewrite", model="rewriter"),
                prompt_hash=t.text_hash("preserve facts"), arm=None, draw_index=None),
        request(4, model_role="verifier", event=event(4, stage="verification", model="verifier"),
                prompt_hash=t.text_hash("compare facts"), arm=None, draw_index=None),
    )
    saved = checkpoint(protocol=p, draws=(draw(response=response), draw(1, status="failed", response=None,
                                                                              error="quota exhausted")),
                       rewrites=(rewrite,), requests=requests)
    portable = json.loads(json.dumps(saved.to_dict(), ensure_ascii=False, allow_nan=False))
    restored = t.ExperimentCheckpoint.from_dict(portable)
    assert restored.to_dict() == saved.to_dict()
    assert restored.items[0].label == {"direction": 1}
    assert restored.items[0].as_of == "2024-01-02"
    assert restored.rewrites[0].verification_response.content == "not-json"
    assert restored.draws[0].response.token_evidence == evidence
    assert restored.draws[1].error == "quota exhausted"
    assert restored.requests[1].event.status == "failed"
    assert restored.requests[1].event.usage["input_tokens"] == 7
    assert restored.requests[0].event.usage is None
    assert restored.requests[0].event.cost is None
    assert restored.draws[0].response.confirmed_settings == {}
    portable["items"][0]["label"]["direction"] = -1
    assert saved.items[0].label["direction"] == 1
    assert restored.validate_restart(p, saved.items).prior_used_requests == 4


@pytest.mark.parametrize("location", ["realized", "alternative"])
def test_censored_response_schema_roundtrip_preserves_raw_evidence_and_unavailability(location):
    evidence = (TokenEvidence("UP", -9999.0 if location == "realized" else -0.1, (
        TokenAlternative("UP", -9999.0 if location == "alternative" else -0.1),
    )),)
    original = draw(response=ModelResponse("UP", evidence))
    data = json.loads(json.dumps(original.to_dict(), ensure_ascii=False, allow_nan=False))
    restored = t.DrawRecord.from_dict(data)
    assert restored.to_dict() == original.to_dict()
    assert restored.response.content == "UP" and restored.response.token_evidence == evidence
    assert "censored" in restored.response.mia_unavailable_reason
    with pytest.raises(ValueError, match="censored"):
        restored.response.to_legacy()


def test_resume_returns_exact_remaining_budget_and_new_request_namespace():
    saved = checkpoint()
    state = saved.validate_restart(saved.protocol, saved.items)
    assert state.prior_used_requests == 2
    assert state.remaining_requests == 10
    assert state.execution_id != "execution-1"
    assert state.execution_id != saved.validate_restart(saved.protocol, saved.items).execution_id
    exhausted = replace(saved, protocol=replace(saved.protocol, request_budget=2))
    assert exhausted.validate_restart(exhausted.protocol, exhausted.items).remaining_requests == 0


@pytest.mark.parametrize("change", [
    {"draws": 3}, {"bootstrap_seed": 18}, {"bootstrap_count": 501},
    {"task_metric": "new_metric"}, {"utility_tolerance": 0.03}, {"request_budget": 13},
    {"data_hashes": {"evaluation": "c" * 64}},
    {"audit_selection": {"method": "all"}}, {"rewrite_method_version": "narrative-v2"},
    {"target_model": ModelConfig("callback", None, "different-model")},
])
def test_resume_refuses_changed_frozen_protocol(change):
    saved = checkpoint()
    with pytest.raises(ValueError, match="protocol"):
        saved.validate_restart(protocol(**change), saved.items)


@pytest.mark.parametrize("change", [
    {"instructions": "Return prose."}, {"evidence": "changed facts"}, {"label": {"direction": -1}},
    {"as_of": "2024-01-03"}, {"cohort": "other"}, {"item_id": "different-id"},
])
def test_resume_refuses_changed_inputs(change):
    saved = checkpoint()
    with pytest.raises(ValueError, match="inputs"):
        saved.validate_restart(saved.protocol, (item(**change),))


def test_resume_preserves_item_order_for_counterbalancing():
    saved = checkpoint(items=(item(), item(item_id="item-2")))
    with pytest.raises(ValueError, match="inputs"):
        saved.validate_restart(saved.protocol, tuple(reversed(saved.items)))


@pytest.mark.parametrize("case", ["in_flight", "pending_draw", "missing_draw", "pending_rewrite"])
def test_resume_rejects_ambiguous_requests_without_running_anything(case):
    saved = checkpoint()
    if case == "in_flight":
        saved = replace(saved, requests=(request(event=event(status="started", latency_s=None)),))
    elif case == "pending_draw":
        saved = replace(saved, draws=(draw(status="pending", response=None),))
    elif case == "missing_draw":
        saved = replace(saved, draws=())
    else:
        saved = replace(saved, requests=(request(model_role="rewriter", arm=None, draw_index=None,
                                                 event=event(stage="rewrite", model="rewriter")),),
                        rewrites=(t.RewriteRecord(item_id="item-1", original_evidence=item().evidence,
                                                  status="pending", stage="rewrite"),))
    with pytest.raises(ValueError, match="[Aa]mbiguous|in.flight"):
        saved.validate_restart(saved.protocol, saved.items)


def test_unscheduled_pending_records_are_safe_and_do_not_consume_budget():
    p = protocol()
    saved = checkpoint(protocol=p, status="pending", requests=(),
                       draws=(draw(status="pending", response=None),),
                       rewrites=(t.RewriteRecord("item-1", item().evidence, status="pending"),))
    assert saved.validate_restart(p, saved.items).prior_used_requests == 0
    assert saved.validate_restart(p, saved.items).remaining_requests == p.request_budget


def test_namespaced_request_ids_allow_resume_without_losing_prior_usage():
    saved = checkpoint(requests=(request(), request(execution_id="execution-2", draw_index=1)))
    assert saved.validate_restart(saved.protocol, saved.items).prior_used_requests == 2


@pytest.mark.parametrize("case", ["duplicate_draw", "duplicate_request", "ledger_gap", "over_budget",
                                   "wrong_model", "prompt_mismatch", "wrong_evidence", "draw_out_of_range"])
def test_checkpoint_rejects_corrupt_pairing_or_physical_ledger(case):
    saved = checkpoint()
    with pytest.raises(ValueError):
        if case == "duplicate_draw":
            replace(saved, draws=(draw(), draw()))
        elif case == "duplicate_request":
            replace(saved, requests=(request(), request()))
        elif case == "ledger_gap":
            replace(saved, requests=(request(2),))
        elif case == "over_budget":
            replace(saved, protocol=replace(saved.protocol, request_budget=1))
        elif case == "wrong_model":
            replace(saved, requests=(request(event=event(model="another-model")),))
        elif case == "prompt_mismatch":
            replace(saved, requests=(request(prompt_hash="d" * 64),))
        elif case == "wrong_evidence":
            replace(saved, rewrites=(t.RewriteRecord("item-1", "different evidence"),))
        else:
            replace(saved, draws=(draw(2),))


def test_completed_marker_requires_every_planned_arm_draw_and_terminal_rewrite():
    p = protocol()
    with pytest.raises(ValueError, match="completed"):
        checkpoint(status="completed")
    complete_draws = tuple(draw(index, arm=arm) for arm in ("original", "rewritten") for index in range(p.draws))
    complete_requests = tuple(request(index + 1, arm=entry.arm, draw_index=entry.draw_index)
                              for index, entry in enumerate(complete_draws))
    saved = checkpoint(protocol=p, status="completed", draws=complete_draws, requests=complete_requests,
                       rewrites=(t.RewriteRecord("item-1", item().evidence, status="completed",
                                                  rewritten_evidence="Preserved evidence."),))
    assert saved.validate_restart(p, saved.items).remaining_requests == p.request_budget - 4


def test_restart_refuses_partial_rewrite_output_without_safe_completion_marker():
    saved = checkpoint(rewrites=(t.RewriteRecord("item-1", item().evidence, rewritten_evidence="partial",
                                                rewrite_response=ModelResponse("partial")),))
    with pytest.raises(ValueError, match="[Aa]mbiguous"):
        saved.validate_restart(saved.protocol, saved.items)


def test_restart_refuses_output_whose_physical_request_accounting_is_missing():
    saved = checkpoint(requests=())
    with pytest.raises(ValueError, match="[Aa]mbiguous|ledger"):
        saved.validate_restart(saved.protocol, saved.items)


def test_import_detects_changed_stored_protocol_and_input_content():
    saved = checkpoint()
    data = saved.to_dict()
    data["protocol"]["bootstrap_seed"] += 1
    with pytest.raises(ValueError, match="protocol hash"):
        t.ExperimentCheckpoint.from_dict(data)
    data = saved.to_dict()
    data["items"][0]["label"]["direction"] = -1
    with pytest.raises(ValueError, match="input hashes"):
        t.ExperimentCheckpoint.from_dict(data)


def test_invalid_physical_event_metadata_cannot_break_json_export():
    protocol()
    with pytest.raises((TypeError, ValueError)):
        request(event=event(error=RuntimeError("unsanitized exception")))


@pytest.mark.parametrize("case", ["undeclared_cohort", "unaccepted_rewrite"])
def test_checkpoint_rejects_records_that_cannot_support_protocol_comparisons(case):
    saved = checkpoint()
    with pytest.raises(ValueError):
        if case == "undeclared_cohort":
            replace(saved, items=(item(cohort="undeclared"),))
        else:
            replace(saved, draws=(draw(arm="rewritten"),), requests=())


@pytest.mark.parametrize("case", ["version", "nan", "missing_response", "failure_without_error", "bad_date"])
def test_persisted_schema_rejects_unsupported_or_unreplayable_values(case):
    p = protocol()
    with pytest.raises((TypeError, ValueError)):
        if case == "version":
            t.ExperimentProtocol.from_dict(p.to_dict() | {"schema_version": 3})
        elif case == "nan":
            replace(p, audit_selection={"size": float("nan")})
        elif case == "missing_response":
            draw(response=None)
        elif case == "failure_without_error":
            draw(status="failed", response=None)
        else:
            item(as_of="not-a-date")
