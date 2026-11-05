"""Selected application plans execute only their explicitly prepared method."""

import json
from dataclasses import replace

import pytest

from recall_guard.core.completion import FeatureChoices, ModelResponse
from recall_guard.core.ensemble import EnsembleSpec
from recall_guard.core.readiness_types import CancellationToken
from recall_guard.core.request_control import CallBudget, RequestError
from recall_guard.harness.rewriting import EvidenceRewriter, RewritePolicy
from recall_guard.harness.workflow_plan import CompletionFallbackPolicy, WorkflowRequest
from recall_guard.harness.workflow_preparation import prepare_workflow
from tests.harness.test_workflow import CANDIDATE, VERDICT, item
from tests.harness.test_workflow_preparation import config, factory_for


def prepared(*, rewrite=False, ensemble=None, candidate=CANDIDATE, verdict=VERDICT, cancellation=None, retries=0,
             on_observation=None):
    cfg = replace(config(parameters={"logprobs": True, "top_logprobs": 2}), max_retries=retries)
    roles = {"target": cfg, "rewriter": replace(config(), model="rewrite"), "verifier": replace(config(), model="verify")}
    policy_clients = []
    factory = factory_for([], policy_clients)
    policy = RewritePolicy(EvidenceRewriter(factory(roles["rewriter"], CallBudget(1), features=FeatureChoices())),
                           factory(roles["verifier"], CallBudget(1), features=FeatureChoices()))
    features = FeatureChoices(rewriting=rewrite, ensemble_draws=ensemble.draws if ensemble else 1)
    request = WorkflowRequest(roles, replace(features, mia_scoring=True), policy.intervention_id, ensemble)
    outputs = [ModelResponse("OK"), ModelResponse("OK")]
    if rewrite:
        outputs += [ModelResponse("OK"), ModelResponse("OK")]
    outputs += [ModelResponse("OK")] * (3 if rewrite else 1)  # Fresh execution preflight.
    if rewrite:
        outputs += [ModelResponse(candidate), ModelResponse(verdict)]
    outputs += [ModelResponse("up")] * features.ensemble_draws
    clients, budget = [], CallBudget(30)
    result = prepare_workflow(request, CompletionFallbackPolicy(features, True), factory_for(outputs, clients),
                              budget, rewrite_policy=policy, cancellation=cancellation, on_observation=on_observation)
    assert result.ready
    return result, budget


@pytest.mark.parametrize("rewrite", [False, True])
@pytest.mark.parametrize("ensemble", [None, EnsembleSpec(draws=1, max_workers=1, min_parsed=1),
                                     EnsembleSpec(draws=3, max_workers=1, min_parsed=2)])
def test_prepared_defaults_execute_selected_single_ensemble_rewrite(rewrite, ensemble):
    preparation, budget = prepared(rewrite=rewrite, ensemble=ensemble)
    historical = preparation.to_dict()
    before = budget.used
    result = preparation.workflow.run(item())
    assert result.status == "completed"
    assert result.method == preparation.plan.selected.method
    assert result.n_attempted == (ensemble.draws if ensemble else 1)
    assert result.provenance["workflow_plan"]["plan_id"] == preparation.plan.plan_id
    assert result.provenance["workflow_plan"]["reason"] == "missing_token_evidence"
    assert result.p_memorized is None and result.guarded_score is None
    assert "completion-only" in result.unavailable_reasons["p_memorized"]
    assert result.provenance["readiness_trace"]
    assert budget.used - before == (3 if rewrite else 1) + (2 if rewrite else 0) + result.n_attempted
    assert preparation.to_dict() == historical
    with pytest.raises(TypeError):
        result.provenance["workflow_plan"]["plan_id"] = "changed"
    with pytest.raises(TypeError):
        result.provenance["readiness_trace"][0]["state"] = "ready"
    assert all(event.stage not in {"rewrite", "verify"} for event in budget.events) if not rewrite else result.rewrite


@pytest.mark.parametrize("override", [{"rewrite": True}, {"ensemble": EnsembleSpec(draws=1, min_parsed=1)},
                                      {"ensemble": EnsembleSpec(draws=2, min_parsed=1)}])
def test_selected_plan_refuses_different_execution_choices_before_calls(override):
    preparation, budget = prepared()
    before = budget.used
    with pytest.raises(ValueError, match="selected plan"):
        preparation.workflow.run(item(), **override)
    assert budget.used == before


def test_prepared_cancelled_run_retains_plan_and_admits_nothing():
    cancellation = CancellationToken()
    preparation, budget = prepared(cancellation=cancellation)
    cancellation.cancel()
    before = budget.used
    result = preparation.workflow.run(item())
    assert result.status == "failed" and result.failure_kind == "cancelled"
    assert budget.used == before and result.n_attempted == 0
    assert result.provenance["readiness_trace"][-1]["state"] == "cancelled"
    assert result.provenance["workflow_plan"]["plan_id"] == preparation.plan.plan_id


def test_rejected_prepared_rewrite_never_admits_target_generation():
    verdict = json.loads(VERDICT)
    verdict["equivalent"] = False
    preparation, budget = prepared(rewrite=True, verdict=json.dumps(verdict))
    before = budget.used
    result = preparation.workflow.run(item())
    assert result.status == "failed" and result.rewrite.status == "failed"
    assert result.response is None and result.n_attempted == 0 and result.accepted_evidence is None
    assert [event.stage for event in budget.events[before:]] == ["probe", "probe", "probe", "rewrite", "verify"]
    assert result.provenance["workflow_plan"]["plan_id"] == preparation.plan.plan_id


def test_cancellation_during_legacy_physical_attempt_prevents_retry_and_queued_draws():
    token = CancellationToken()
    preparation, budget = prepared(ensemble=EnsembleSpec(draws=3, max_workers=1, min_parsed=1),
                                   cancellation=token, retries=2)
    workflow = preparation.workflow
    # The first operation completes; subsequent physical admissions must stop.
    original = workflow.target._callback
    count = 0
    def callback(prompt, request, **parameters):
        nonlocal count
        count += 1
        if count == 1:
            return original(prompt, request=request, **parameters)
        def exchange():
            token.cancel()
            raise RequestError("retryable", retryable=True)
        return request(exchange)
    workflow.target._callback = callback
    before = budget.used
    result = workflow.run(item())
    assert result.status == "failed" and result.failure_kind == "cancelled"
    assert budget.used - before == 2  # One fresh probe and one failed target attempt.
    assert len(result.draws) == 1 and result.n_unattempted == 2
    assert result.provenance["readiness_trace"][-1]["reason"] == "caller_cancelled"


@pytest.mark.parametrize("parameters", [{"custom": [1]}, {"custom": [True, None]}, {"custom": None}])
def test_selected_configuration_uses_canonical_json_identity(parameters):
    preparation, budget = prepared()
    preparation.workflow.target.config = replace(preparation.workflow.target.config, parameters=parameters)
    before = budget.used
    with pytest.raises(ValueError, match="selected plan"):
        preparation.workflow.run(item())
    assert budget.used == before


@pytest.mark.parametrize("ensemble", [None, EnsembleSpec(draws=3, max_workers=1, min_parsed=1)])
def test_selected_mia_capability_drift_stops_queued_scoring_without_fallback(ensemble):
    from tests.harness.test_workflow_scoring import Client, fit

    source = Client()
    scorer = fit(source)
    cfg = source.config
    good = source.generate("eval")
    outputs = [good, good, ModelResponse("complete answer without tokens"), good, good]
    budget = CallBudget(12)
    request = WorkflowRequest({"target": cfg}, FeatureChoices(mia_scoring=True,
                              ensemble_draws=ensemble.draws if ensemble else 1), ensemble=ensemble)
    preparation = prepare_workflow(request, CompletionFallbackPolicy(FeatureChoices()),
                                   factory_for(outputs, []), budget, scorer=scorer)
    assert preparation.ready and preparation.plan.reason is None
    before = budget.used
    result = preparation.workflow.run(item())
    assert result.status == "failed" and result.failure_kind == "configuration_error"
    assert result.method == request.method and result.calibration_identity == scorer.calibration_identity
    assert result.p_memorized is None and budget.used - before == 2
    assert len(result.draws) == 1 and result.draws[0].status == "failed"
    assert result.draws[0].response.content == "complete answer without tokens"
    assert result.provenance["readiness_trace"][-1]["reason"] == "missing_token_evidence"


def test_cancelled_cached_preflight_records_stop_without_target_admission():
    token = CancellationToken()
    preparation, budget = prepared(cancellation=token)
    assert preparation.workflow.preflight().ready_for_batch
    token.cancel()
    before = budget.used
    result = preparation.workflow.run(item())
    assert result.status == "failed" and result.failure_kind == "cancelled"
    assert budget.used == before
    assert result.n_attempted == 0


@pytest.mark.parametrize("failure", ["pending", "partial"])
def test_readiness_execution_defers_or_retains_partial_without_usable_draws(failure):
    from recall_guard.core.completion import ModelConfig
    from recall_guard.core.providers import OpenAICompatibleClient
    from recall_guard.core.readiness_types import ReadinessPolicy
    from tests.core.test_provider_readiness import BODY, REQUEST_ID, server, sse

    failed = ((202, {"NVCF-REQID": REQUEST_ID}, {}) if failure == "pending" else
              (200, {"Content-Type": "text/event-stream"},
               b'data: {"choices":[{"index":0,"delta":{"content":"unfinished"}}]}\n\n'))
    complete = (200, {}, BODY)
    parameters = {}
    if failure == "partial":
        parameters = {"stream": True}
        complete = (200, {"Content-Type": "text/event-stream"},
                    sse({"choices": [{"index": 0, "delta": {"content": "OK"}, "finish_reason": "stop"}]}, "[DONE]"))
    with server([complete] * 3 + [failed]) as (endpoint, seen):
        cfg = ModelConfig("openai_chat", endpoint, "fixture-model", provider="fixture",
                          parameters=parameters, readiness=ReadinessPolicy(), max_retries=0)
        ensemble = EnsembleSpec(draws=3, max_workers=1, min_parsed=1)
        features = FeatureChoices(ensemble_draws=3)
        budget = CallBudget(10)
        preparation = prepare_workflow(WorkflowRequest({"target": cfg}, features, ensemble=ensemble), None,
            lambda config, budget, features: OpenAICompatibleClient(config, budget, features=features,
                                                                     allow_unauthenticated=True), budget)
        # Preparation and fresh execution preflight consume separate responses.
        # The target's first draw succeeds; the next draw cannot yield a usable answer.
        result = preparation.workflow.run(item())
        assert result.status == "failed" and result.p_memorized is None and result.response is None
        assert len(seen) == budget.used == 4
        assert result.n_usable <= 1 and result.n_unattempted >= 1
        trace = result.provenance["readiness_trace"]
        assert any(event["state"] == ("deferred" if failure == "pending" else "unknown_outcome") for event in trace)
        if failure == "partial":
            assert result.draws[-1].response is None
            assert result.draws[-1].diagnostics["response_evidence"]["content"] == "unfinished"
            assert not result.draws[-1].diagnostics["response_evidence"]["complete"]


@pytest.mark.parametrize("field", ["workflow_plan", "readiness_trace"])
def test_new_outcome_metadata_cannot_enter_a_version_one_run(field):
    from recall_guard.harness.intervention_journal import ExperimentRun
    from tests.harness.test_intervention_report import recorded_run

    snapshot = recorded_run().to_dict()
    snapshot["arms"][0]["outcome"].setdefault("provenance", {})[field] = {}
    with pytest.raises(ValueError, match="schema version 2"):
        ExperimentRun.from_dict(snapshot)


@pytest.mark.parametrize("ensemble", [None, EnsembleSpec(draws=3, max_workers=1, min_parsed=1)])
def test_selected_mia_uses_existing_supplied_response_scorer_and_reducer(ensemble):
    from tests.harness.test_workflow_scoring import Client, fit

    source = Client()
    scorer = fit(source)
    good = source.generate("eval")
    budget = CallBudget(12)
    features = FeatureChoices(mia_scoring=True, ensemble_draws=ensemble.draws if ensemble else 1)
    preparation = prepare_workflow(WorkflowRequest({"target": source.config}, features, ensemble=ensemble), None,
        factory_for([good] * 8, []), budget, scorer=scorer)
    assert preparation.ready and preparation.workflow.scorer is scorer
    result = preparation.workflow.run(item())
    assert result.status == "completed" and result.p_memorized is not None
    assert result.calibration_identity == scorer.calibration_identity
    assert result.p_memorized == (result.scored_ensemble.p_memorized_point if ensemble else result.guarded_score.p_memorized)
    before = budget.used
    with pytest.raises(ValueError, match="selected plan"):
        preparation.workflow.run(item(), scorer=None)
    assert budget.used == before


def test_json_boolean_number_configuration_drift_is_rejected():
    cfg = config(parameters={"custom": [True]})
    budget = CallBudget(3)
    preparation = prepare_workflow(WorkflowRequest({"target": cfg}, FeatureChoices()), None,
                                   factory_for([ModelResponse("OK")], []), budget)
    preparation.workflow.target.config = replace(cfg, parameters={"custom": [1]})
    with pytest.raises(ValueError, match="selected plan"):
        preparation.workflow.run(item())
    assert budget.used == 1


def test_cancellation_after_rewrite_generation_never_admits_verifier_or_target():
    token = CancellationToken()
    preparation, budget = prepared(rewrite=True, cancellation=token)
    rewriter = preparation.workflow.rewrite_policy.rewriter.client
    original = rewriter._callback
    calls = 0
    def callback(prompt, request, **parameters):
        nonlocal calls
        calls += 1
        result = original(prompt, request=request, **parameters)
        if calls == 2:
            token.cancel()
        return result
    rewriter._callback = callback
    before = budget.used
    result = preparation.workflow.run(item())
    assert result.status == "failed" and result.rewrite.status == "failed"
    assert result.response is None and result.n_attempted == 0
    assert [event.stage for event in budget.events[before:]] == ["probe", "probe", "probe", "rewrite"]
    assert result.provenance["readiness_trace"][-1]["state"] == "cancelled"


def test_observer_failure_is_sanitized_and_stops_queued_generation():
    marker = "SYNTHETIC_CALLBACK_DIAGNOSTIC"
    def observer(observation):
        if observation.stage == "target":
            raise RuntimeError(marker)
    preparation, budget = prepared(ensemble=EnsembleSpec(draws=3, max_workers=1, min_parsed=1),
                                   on_observation=observer)
    before = budget.used
    result = preparation.workflow.run(item())
    assert result.status == "failed" and result.n_attempted == 1 and budget.used - before == 2
    assert result.provenance["readiness_trace"][-1]["reason"] == "callback_failed"
    assert marker not in repr(result)


def test_post_response_scoring_drift_signals_waiting_physical_admissions(monkeypatch):
    from recall_guard.harness.workflow_execution import WorkflowExecution
    from tests.harness.test_workflow_scoring import Client, fit

    source = Client()
    scorer = fit(source)
    good = source.generate("eval")
    preparation = prepare_workflow(WorkflowRequest({"target": source.config}, FeatureChoices(mia_scoring=True)),
                                   None, factory_for([good] * 3, []), CallBudget(5), scorer=scorer)
    executions = []
    original = WorkflowExecution.generate
    def generate(execution, *args, **kwargs):
        executions.append(execution)
        return original(execution, *args, **kwargs)
    def reject(*args, **kwargs):
        raise ValueError("Calibration evidence changed")
    monkeypatch.setattr(WorkflowExecution, "generate", generate)
    monkeypatch.setattr(scorer, "score_response", reject)
    result = preparation.workflow.run(item())
    assert result.status == "failed" and result.failure_kind == "configuration_error"
    assert len(executions) == 1
    # The same token is checked by every guarded pacing/retry admission.
    assert executions[0].cancellation.cancelled


@pytest.mark.parametrize("mode", ["cancelled", "observer", "settings", "scorer"])
def test_new_local_failures_redact_registered_secret_and_preserve_evidence(mode, monkeypatch):
    from recall_guard.core.provider_callback import CallbackClient
    from recall_guard.harness.intervention_journal import arm_snapshot
    from tests.harness.test_workflow_scoring import Client, fit

    marker = "SYNTHETIC_REGISTERED_FAILURE_SECRET"
    source = Client()
    scorer = fit(source) if mode == "scorer" else None
    good = source.generate("eval")
    cfg = source.config if scorer else config(parameters={"temperature": 0.4})
    token, calls = CancellationToken(), []
    failed = ModelResponse(marker, token_evidence=tuple(replace(t, token=marker) for t in good.token_evidence),
                           usage={"output_tokens": 7}, confirmed_settings={"temperature": 0.8} if mode == "settings" else {})
    def factory(config, budget, *, features):
        def callback(prompt, request, **parameters):
            def exchange():
                calls.append(prompt)
                if len(calls) == 3:
                    if mode == "cancelled":
                        token.cancel()
                    return failed
                return good if scorer else "OK"
            return request(exchange)
        return CallbackClient(config, budget, callback, features=features, physical_request_hook="request",
                              strict_budget=True, secrets=[marker])
    def observe(observation):
        if mode == "observer" and observation.stage == "target":
            raise RuntimeError(marker)
    preparation = prepare_workflow(WorkflowRequest({"target": cfg}, FeatureChoices(mia_scoring=scorer is not None)),
                                   None, factory, CallBudget(8), cancellation=token, on_observation=observe,
                                   scorer=scorer)
    if scorer:
        def reject(*args, **kwargs):
            raise ValueError(marker)
        monkeypatch.setattr(scorer, "score_response", reject)
    result = preparation.workflow.run(item())
    assert result.status == "failed" and len(calls) == 3 and result.n_usable == 0
    assert marker not in repr(result)
    assert marker not in json.dumps(arm_snapshot(result))
    response = result.draws[0].response
    assert isinstance(response, ModelResponse) and "REDACTED" in response.content
    assert response.usage == {"output_tokens": 7}
    assert response.token_evidence[0].logprob == failed.token_evidence[0].logprob
    assert marker not in response.token_evidence[0].token
    assert result.failure_kind == ("cancelled" if mode == "cancelled" else
                                   "configuration_error" if mode in {"settings", "scorer"} else "request_error")


def test_two_worker_local_stop_keeps_root_failure_and_never_claims_caller_cancellation(monkeypatch):
    from threading import Event, Lock

    from recall_guard.core.provider_callback import CallbackClient
    from recall_guard.harness.workflow import _RecordingClient

    caller, queued, sibling_stopped = CancellationToken(), Event(), Event()
    counts, count_lock = {"logical": 0, "physical": 0}, Lock()
    cfg = replace(config(parameters={"temperature": 0.4}), min_call_interval_s=0.15)
    def factory(config, budget, *, features):
        def callback(prompt, request, **parameters):
            with count_lock:
                counts["logical"] += 1
                logical = counts["logical"]
            if logical == 4:
                queued.set()
            def exchange():
                with count_lock:
                    counts["physical"] += 1
                    physical = counts["physical"]
                if physical == 3:
                    assert queued.wait(3)
                return ModelResponse("OK", confirmed_settings={"temperature": 0.8} if physical == 3 else {})
            return request(exchange)
        return CallbackClient(config, budget, callback, features=features, physical_request_hook="request", strict_budget=True)
    ensemble = EnsembleSpec(draws=2, max_workers=2, min_parsed=1, max_transport_failure_ratio=1)
    preparation = prepare_workflow(WorkflowRequest({"target": cfg}, FeatureChoices(ensemble_draws=2), ensemble=ensemble),
                                   None, factory, CallBudget(8), cancellation=caller)
    original = _RecordingClient.update
    def update(recorder, index, **changes):
        original(recorder, index, **changes)
        if changes.get("status") == "failed" and index == 1:
            sibling_stopped.set()
        if changes.get("failure_kind") == "configuration_error" and index == 0:
            assert sibling_stopped.wait(3)
    monkeypatch.setattr(_RecordingClient, "update", update)
    result = preparation.workflow.run(item())
    assert result.status == "failed" and result.failure_kind == "configuration_error"
    assert counts["physical"] == preparation.workflow.budget.used == 3
    assert not caller.cancelled and len(result.draws) == 2
    assert result.provenance["failure_diagnostics"]["code"] == "response_settings_mismatch"
    assert result.provenance["readiness_trace"][-1]["reason"] == "response_settings_mismatch"
    assert not any(event["reason"] == "caller_cancelled" for event in result.provenance["readiness_trace"])

