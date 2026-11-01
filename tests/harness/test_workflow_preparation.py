"""Preparation uses local, explicitly accounted callback fixtures only."""

import json
from dataclasses import replace
from threading import Event, Thread

import pytest

from recall_guard.core.completion import (
    CapabilityEvidence,
    FeatureChoices,
    ModelConfig,
    ModelResponse,
    TokenAlternative,
    TokenEvidence,
)
from recall_guard.core.ensemble import EnsembleSpec
from recall_guard.core.provider_anthropic import AnthropicClient
from recall_guard.core.provider_callback import CallbackClient
from recall_guard.core.readiness_evidence import PartialResponse
from recall_guard.core.readiness_types import CancellationToken
from recall_guard.core.request_control import CallBudget, RequestError
from recall_guard.harness.intervention_types import PreparationArtifact
from recall_guard.harness.rewriting import EvidenceRewriter, RewritePolicy
from recall_guard.harness.scorer import MemoryGuardedScorer
from recall_guard.harness.workflow_plan import CompletionFallbackPolicy, WorkflowRequest
from recall_guard.harness.workflow_preparation import FactoryContractError, prepare_workflow


def config(**kwargs):
    return ModelConfig(protocol="callback", endpoint=None, model="fixture", max_retries=0, **kwargs)


def factory_for(outputs, clients):
    def factory(config, budget, *, features):
        def callback(prompt, request, **parameters):
            def exchange():
                output = outputs.pop(0)
                if isinstance(output, Exception):
                    raise output
                return replace(output, requested_settings=parameters)
            return request(exchange)
        client = CallbackClient(config, budget, callback, features=features,
                                physical_request_hook="request", strict_budget=True)
        clients.append(client)
        return client
    return factory


def test_observed_fallback_rebuilds_and_reprobes_exact_selected_configuration():
    requested = WorkflowRequest({"target": config(parameters={"logprobs": True, "top_logprobs": 2,
                                                             "temperature": 0.4, "max_tokens": 30})},
                                FeatureChoices(mia_scoring=True))
    clients, budget = [], CallBudget(2)
    result = prepare_workflow(requested, CompletionFallbackPolicy(FeatureChoices(), True),
                              factory_for([ModelResponse("OK"), ModelResponse("OK")], clients), budget)
    assert result.ready
    assert result.workflow.scorer is None
    assert result.plan.reason == "missing_token_evidence"
    assert result.plan.omitted_controls == ("logprobs", "top_logprobs")
    assert result.plan.selected.roles["target"].parameters == {"temperature": 0.4, "max_tokens": 30}
    assert len(clients) == 2 and all(client.budget is budget for client in clients)
    assert len(result.artifact.evidence["target"]) == 2
    assert budget.used == 2
    assert all(record.phase == "preparation" for record in result.artifact.requests)
    assert PreparationArtifact.from_dict(result.artifact.to_dict()) == result.artifact
    assert result.binding.remaining_evaluation_requests == 0
    with pytest.raises((AttributeError, TypeError)):
        result.plan.selected.roles["target"] = requested.roles["target"]


def test_declared_unsupported_resolves_before_strict_factory():
    cfg = config(capabilities={"logprobs": CapabilityEvidence("unsupported", "caller declaration")})
    clients = []
    result = prepare_workflow(WorkflowRequest({"target": cfg}, FeatureChoices(mia_scoring=True)),
                              CompletionFallbackPolicy(FeatureChoices()),
                              factory_for([ModelResponse("OK")], clients), CallBudget(1))
    assert result.ready and result.plan.reason == "unsupported_token_evidence"
    assert len(clients) == 1 and not clients[0].features.mia_scoring


@pytest.mark.parametrize("parameters", [{"logprobs": False}, {"logprobs": None}, {"top_logprobs": None},
                                        {"top_logprobs": 0}])
def test_explicit_contradictions_never_fallback_or_call_factory(parameters):
    def forbidden(*args, **kwargs):
        pytest.fail("Contradiction must fail before factory")
    result = prepare_workflow(WorkflowRequest({"target": config(parameters=parameters)}, FeatureChoices(mia_scoring=True)),
                              CompletionFallbackPolicy(FeatureChoices(), True), forbidden, CallBudget(3))
    assert not result.ready and result.plan.reason is None
    assert result.artifact.stopping_reason == "configuration_error"


@pytest.mark.parametrize("output", [ModelResponse(""), RequestError("timeout", code="timeout"),
                                     RequestError("malformed", code="invalid_response"),
                                     RequestError("partial", response_evidence=PartialResponse("unfinished", reason="interrupted")),
                                     RequestError("gateway", status_code=504),
                                     RequestError("rejected", status_code=400, parameter="temperature"),
                                     RequestError("denied", status_code=401)])
def test_unrelated_failures_are_not_token_triggers(output):
    clients = []
    result = prepare_workflow(WorkflowRequest({"target": config()}, FeatureChoices(mia_scoring=True)),
                              CompletionFallbackPolicy(FeatureChoices(), True), factory_for([output], clients), CallBudget(3))
    assert not result.ready and result.plan.reason is None
    assert len(clients) == 1 and len(result.artifact.requests) == 1
    assert len(result.artifact.evidence["target"]) == 1


def test_without_authorization_missing_evidence_is_inspectable_failure():
    result = prepare_workflow(WorkflowRequest({"target": config()}, FeatureChoices(mia_scoring=True)), None,
                              factory_for([ModelResponse("OK")], []), CallBudget(1))
    assert not result.ready and result.plan.reason is None
    assert result.artifact.evidence["target"][0].response.content == "OK"


def test_exhaustion_after_fallback_preserves_selected_plan_and_first_probe():
    result = prepare_workflow(WorkflowRequest({"target": config()}, FeatureChoices(mia_scoring=True)),
                              CompletionFallbackPolicy(FeatureChoices()),
                              factory_for([ModelResponse("OK")], []), CallBudget(1))
    assert not result.ready and result.plan.reason == "missing_token_evidence"
    assert result.artifact.stopping_reason == "request_budget"
    assert len(result.artifact.requests) == 1


def test_cancelled_preparation_has_immutable_empty_ledger():
    cancellation = CancellationToken()
    cancellation.cancel()
    result = prepare_workflow(WorkflowRequest({"target": config()}, FeatureChoices()), None,
                              factory_for([], []), CallBudget(1), cancellation=cancellation)
    assert not result.ready and result.artifact.stopping_reason == "caller_cancelled"
    assert result.artifact.requests == ()
    assert "credential" not in json.dumps(result.artifact.to_dict())


def token_response():
    return ModelResponse("OK", token_evidence=(TokenEvidence("OK", -0.1,
                        top_logprobs=(TokenAlternative("OK", -0.1), TokenAlternative("NO", -2.3))),))


def test_missing_calibration_is_not_a_fallback_trigger():
    result = prepare_workflow(WorkflowRequest({"target": config()}, FeatureChoices(mia_scoring=True)),
                              CompletionFallbackPolicy(FeatureChoices()), factory_for([token_response()], []), CallBudget(2))
    assert not result.ready and result.plan.reason is None
    assert result.artifact.stopping_reason == "matching_calibration_required"
    assert result.artifact.evidence["target"][0].ready
    assert len(result.artifact.requests) == 1


@pytest.mark.parametrize("mismatch", [False, True])
def test_supplied_real_fit_must_match_selected_target_before_readiness(mismatch):
    cfg = config()
    def callback(prompt, request):
        base = -0.2 if prompt.startswith("is-") else -5.0 if prompt.startswith("oos-") else -1.0
        evidence = tuple(TokenEvidence("x", base + i * 0.01, (
            TokenAlternative("x", base), TokenAlternative("y", base - 1), TokenAlternative("z", base + 0.1)))
            for i in range(5))
        return request(lambda: ModelResponse("Direction: 1\nConfidence: 0.8", token_evidence=evidence))
    calibration_client = CallbackClient(cfg, CallBudget(64), callback, features=FeatureChoices(mia_scoring=True),
                                        physical_request_hook="request", strict_budget=True)
    scorer = MemoryGuardedScorer.calibrate_client(client=calibration_client, is_memorized=[f"is-{i}" for i in range(8)],
        oos_control=[f"oos-{i}" for i in range(8)], min_valid=4, max_workers=1, min_auc=0)
    selected_config = replace(cfg, model="changed") if mismatch else cfg
    result = prepare_workflow(WorkflowRequest({"target": selected_config}, FeatureChoices(mia_scoring=True)),
        CompletionFallbackPolicy(FeatureChoices()), factory_for([token_response()], []), CallBudget(1), scorer=scorer)
    assert result.plan.reason is None
    assert result.ready is not mismatch
    assert (result.workflow.scorer is scorer) if result.ready else result.artifact.stopping_reason == "configuration_error"


def test_repeated_preparation_refuses_used_budget_without_replenishing():
    budget = CallBudget(2)
    request = WorkflowRequest({"target": config()}, FeatureChoices(mia_scoring=True))
    first = prepare_workflow(request, None, factory_for([ModelResponse("OK")], []), budget)
    events = budget.events
    with pytest.raises(ValueError, match="unused"):
        prepare_workflow(request, CompletionFallbackPolicy(FeatureChoices()), factory_for([], []), budget)
    assert budget.used == 1 and budget.remaining == 1 and budget.events == events
    assert first.binding.preparation_used_requests == 1


@pytest.mark.parametrize("code,status", [("unsupported_parameter", 400), ("invalid_parameter", 422),
                                        ("invalid_request_error", 400)])
def test_exact_typed_token_parameter_rejection_can_fallback(code, status):
    request = WorkflowRequest({"target": config(parameters={"logprobs": True, "temperature": 0.2})},
                              FeatureChoices(mia_scoring=True))
    result = prepare_workflow(request, CompletionFallbackPolicy(FeatureChoices(), True),
        factory_for([RequestError("exact rejected control", status_code=status, code=code, parameter="logprobs"),
                     ModelResponse("OK")], []), CallBudget(2))
    assert result.ready and result.plan.reason == "logprob_controls_rejected"
    assert result.plan.omitted_controls == ("logprobs",)
    assert result.artifact.requests[0].event.status == "failed"


@pytest.mark.parametrize("status,code,parameter", [(401, "unsupported_parameter", "logprobs"),
    (429, "insufficient_quota", "logprobs"), (400, "unrelated_error", "logprobs"),
    (400, "unsupported_parameter", "top_logprobs"), (400, "unsupported_parameter", "temperature")])
def test_imprecise_or_unrelated_rejections_never_select_fallback(status, code, parameter):
    request = WorkflowRequest({"target": config(parameters={"logprobs": True})}, FeatureChoices(mia_scoring=True))
    result = prepare_workflow(request, CompletionFallbackPolicy(FeatureChoices(), True), factory_for([
        RequestError("logprobs missing (untrusted message)", status_code=status, code=code, parameter=parameter)], []),
        CallBudget(2))
    assert not result.ready and result.plan.reason is None


def test_newly_activated_rewrite_and_verifier_are_fresh_and_probed():
    auxiliary = config(parameters={"max_tokens": 90})
    template_factory = factory_for([], [])
    old_budget = CallBudget(0)
    old_rewriter = template_factory(auxiliary, old_budget, features=FeatureChoices())
    old_verifier = template_factory(auxiliary, old_budget, features=FeatureChoices())
    policy = RewritePolicy(EvidenceRewriter(old_rewriter), old_verifier)
    request = WorkflowRequest({"target": config(), "rewriter": auxiliary, "verifier": auxiliary},
                              FeatureChoices(mia_scoring=True), rewrite_policy_id=policy.intervention_id)
    clients, budget = [], CallBudget(4)
    result = prepare_workflow(request, CompletionFallbackPolicy(FeatureChoices(rewriting=True)),
        factory_for([ModelResponse("OK")] * 4, clients), budget, rewrite_policy=policy)
    assert result.ready and budget.used == 4
    assert set(result.artifact.evidence) == {"target", "rewriter", "verifier"}
    assert result.workflow.rewrite_policy.rewriter.client is clients[2]
    assert result.workflow.rewrite_policy.verifier is clients[3]
    assert old_budget.used == 0
    assert [record.model_role for record in result.artifact.requests] == ["target", "target", "rewriter", "verifier"]


@pytest.mark.parametrize("draw_count", [1, 3])
def test_explicit_ensemble_choice_is_retained_without_invented_sampling(draw_count):
    draws = EnsembleSpec(draws=draw_count, min_parsed=1, max_workers=1)
    request = WorkflowRequest({"target": config()}, FeatureChoices(mia_scoring=True, ensemble_draws=draw_count), ensemble=draws)
    result = prepare_workflow(request, CompletionFallbackPolicy(FeatureChoices(ensemble_draws=draw_count)),
                              factory_for([ModelResponse("OK")] * 2, []), CallBudget(2))
    assert result.ready and result.plan.selected.ensemble is draws
    assert result.plan.selected.roles["target"].parameters == {}


@pytest.mark.parametrize("raises_after_call", [False, True])
def test_factory_provider_work_is_detected_and_counted(raises_after_call):
    def bad_factory(config, budget, *, features):
        client = factory_for([ModelResponse("hidden")], [])(config, budget, features=features)
        client.generate("Unauthorized factory generation")
        if raises_after_call:
            raise RuntimeError("private untrusted factory exception")
        return client
    budget = CallBudget(2)
    request = WorkflowRequest({"target": config()}, FeatureChoices())
    with pytest.raises(FactoryContractError, match="without provider work") as failure:
        prepare_workflow(request, None, bad_factory, budget)
    assert failure.value.plan.requested is request
    assert failure.value.events == budget.events
    assert budget.used == failure.value.used_requests == 1
    assert budget.remaining == failure.value.remaining_requests == 1
    assert failure.value.events[0].stage == "target"
    assert failure.value.__context__ is failure.value.__cause__ is None


@pytest.mark.parametrize("original,changed", [({"nested": [True]}, {"nested": [1]}),
    ({"nested": [1]}, {"nested": [1.0]}), ({"nested": [1, 2]}, {"nested": [2, 1]}), ({}, {"value": None})])
def test_factory_exact_configuration_checks_json_types_array_order_and_presence(original, changed):
    def factory(cfg, budget, *, features):
        return factory_for([], [])(replace(cfg, parameters=changed), budget, features=features)
    budget = CallBudget(1)
    result = prepare_workflow(WorkflowRequest({"target": config(parameters=original)}, FeatureChoices()),
                              None, factory, budget)
    assert not result.ready and result.artifact.stopping_reason == "factory_contract_mismatch"
    assert budget.used == 0


def test_confirmed_parameter_contradiction_does_not_trigger_fallback():
    result = prepare_workflow(WorkflowRequest({"target": config(parameters={"temperature": 0.2})},
                                              FeatureChoices(mia_scoring=True)),
        CompletionFallbackPolicy(FeatureChoices()), factory_for([
            ModelResponse("OK", confirmed_settings={"temperature": 0.9})], []), CallBudget(2))
    assert not result.ready and result.plan.reason is None
    assert result.artifact.stopping_reason == "response_settings_mismatch"


def test_nested_generation_settings_survive_exact_fallback_and_serialization():
    parameters = {"logprobs": True, "response_format": {"type": "json_schema", "schema": {"enabled": [True, 1]}}}
    response = ModelResponse("OK", confirmed_settings={"response_format": parameters["response_format"]})
    result = prepare_workflow(WorkflowRequest({"target": config(parameters=parameters)}, FeatureChoices(mia_scoring=True)),
        CompletionFallbackPolicy(FeatureChoices(), True), factory_for([response, response], []), CallBudget(2))
    assert result.ready and result.plan.reason == "missing_token_evidence"
    assert result.plan.selected.to_dict()["roles"]["target"]["parameters"] == {
        "response_format": parameters["response_format"]}
    assert PreparationArtifact.from_dict(result.to_dict()) == result.artifact


def test_cancel_during_initial_observation_admits_no_selected_probe():
    cancellation = CancellationToken()
    budget = CallBudget(3)
    result = prepare_workflow(WorkflowRequest({"target": config()}, FeatureChoices(mia_scoring=True)),
        CompletionFallbackPolicy(FeatureChoices()), factory_for([ModelResponse("OK")], []), budget,
        cancellation=cancellation, on_observation=lambda observation: cancellation.cancel())
    assert not result.ready and result.artifact.stopping_reason == "caller_cancelled"
    assert budget.used == len(result.artifact.requests) == 1


def test_callback_hook_contract_without_actual_admission_cannot_prepare():
    def factory(cfg, budget, *, features):
        return CallbackClient(cfg, budget, lambda prompt, request: ModelResponse("OK"),
                              features=features, physical_request_hook="request")
    result = prepare_workflow(WorkflowRequest({"target": config()}, FeatureChoices()), None, factory, CallBudget(1))
    assert not result.ready and result.artifact.stopping_reason == "unverified_request_accounting"
    assert result.artifact.requests == ()


def test_legacy_cancellation_after_retryable_failure_prevents_retry_admission():
    cancellation, budget = CancellationToken(), CallBudget(3)
    def factory(cfg, budget, *, features):
        def callback(prompt, request):
            def fail():
                cancellation.cancel()
                raise RequestError("temporary fixture failure", status_code=503, retryable=True)
            return request(fail)
        return CallbackClient(cfg, budget, callback, features=features,
                              physical_request_hook="request", strict_budget=True)
    cfg = replace(config(), max_retries=2, retry_backoff_s=0)
    result = prepare_workflow(WorkflowRequest({"target": cfg}, FeatureChoices()), None, factory, budget,
                              cancellation=cancellation)
    assert not result.ready and result.artifact.stopping_reason == "caller_cancelled"
    assert budget.used == len(result.artifact.requests) == 1
    assert budget.events[0].status_code == 503 and budget.events[0].purpose == "probe"
    assert result.artifact.evidence["target"][0].observations[-1].state == "cancelled"


def test_legacy_cancellation_while_queued_for_pacing_prevents_second_admission():
    queued = Event()
    class ObservedCancellation(CancellationToken):
        def wait(self, timeout_s):
            queued.set()
            return super().wait(timeout_s)
    cancellation, budget = ObservedCancellation(), CallBudget(3)
    def factory(cfg, budget, *, features):
        def callback(prompt, request):
            request(lambda: ModelResponse("first physical operation"))
            return request(lambda: ModelResponse("forbidden second operation"))
        return CallbackClient(cfg, budget, callback, features=features,
                              physical_request_hook="request", strict_budget=True)
    cfg = replace(config(), min_call_interval_s=60)
    outcomes = []
    worker = Thread(target=lambda: outcomes.append(prepare_workflow(
        WorkflowRequest({"target": cfg}, FeatureChoices()), None, factory, budget, cancellation=cancellation)), daemon=True)
    worker.start()
    try:
        assert queued.wait(3), "The second physical operation must be queued for pacing"
    finally:
        cancellation.cancel()
        worker.join(3)
    assert not worker.is_alive() and len(outcomes) == 1
    result = outcomes[0]
    assert budget.used == len(result.artifact.requests) == 1
    assert budget.events[0].status == "succeeded"
    assert not result.ready and result.artifact.stopping_reason == "caller_cancelled"
    assert result.artifact.evidence["target"][0].observations[-1].state == "cancelled"


def test_native_declared_unsupported_tokens_resolve_without_enabling_readiness(monkeypatch):
    from types import SimpleNamespace

    cfg = ModelConfig("anthropic_messages", "https://example.invalid/v1/messages", "native-fixture",
                      credential_ref="LOCAL_DUMMY", parameters={"max_tokens": 16}, max_retries=0)
    calls = []
    def post(self, payload):
        calls.append(payload)
        return SimpleNamespace(status_code=200, headers={}, json=lambda: {
            "content": [{"type": "text", "text": "OK"}], "stop_reason": "end_turn"})
    monkeypatch.setattr(AnthropicClient, "_post", post)
    result = prepare_workflow(WorkflowRequest({"target": cfg}, FeatureChoices(mia_scoring=True)),
        CompletionFallbackPolicy(FeatureChoices()), lambda config, budget, features: AnthropicClient(
            config, budget, features=features, environment={"LOCAL_DUMMY": "dummy-only"}), CallBudget(1))
    assert result.ready and result.plan.reason == "unsupported_token_evidence"
    assert len(calls) == 1 and result.plan.selected.roles["target"].readiness is None
    assert result.artifact.evidence["target"][0].response.content == "OK"
    assert "dummy-only" not in json.dumps(result.to_dict())


def test_native_unrelated_invalid_controls_are_not_a_fallback_trigger():
    cfg = ModelConfig("anthropic_messages", "https://example.invalid/v1/messages", "fixture",
                      parameters={"max_tokens": 16, "temperature": 0.3})
    result = prepare_workflow(WorkflowRequest({"target": cfg}, FeatureChoices(mia_scoring=True)),
                              CompletionFallbackPolicy(FeatureChoices()), factory_for([], []), CallBudget(1))
    assert not result.ready and result.plan.reason is None
    assert result.artifact.stopping_reason == "configuration_error"
