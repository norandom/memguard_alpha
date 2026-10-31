"""Offline integration of the four application inference conditions."""

import json
from dataclasses import replace
from threading import Lock

import pytest

from recall_guard.core.completion import (
    CapabilityEvidence,
    CapabilityReport,
    CapabilityStatus,
    FeatureChoices,
    ModelConfig,
    ModelResponse,
    TokenAlternative,
    TokenEvidence,
)
from recall_guard.core.ensemble import EnsembleSpec, reduce_draws
from recall_guard.core.provider_anthropic import AnthropicClient
from recall_guard.core.provider_callback import CallbackClient
from recall_guard.core.providers import OpenAICompatibleClient
from recall_guard.core.request_control import BudgetExhaustedError, CallBudget, RequestError
from recall_guard.harness.intervention_types import RecordStatus
from recall_guard.harness.rewriting import EvidenceRewriter, RewriteInput, RewritePolicy, render_target_prompt
from recall_guard.harness.workflow import RecallWorkflow

SOURCE = "Acme rose 2% on 2024-01-02; estimates remain uncertain."
CANDIDATE = "Let's review: " + SOURCE
CHECKS = (
    "claims", "entities", "dates", "quantities", "units", "uncertainty", "qualifications", "attribution",
    "instruction_injection",
)
VERDICT = json.dumps({"equivalent": True, "checks": dict.fromkeys(CHECKS, True), "discrepancies": []})


def item():
    return RewriteInput("item-1", "Return up or down.", SOURCE, "2024-01-02")


class Client:
    """Minimal clients deliberately have unknown physical request accounting."""

    def __init__(self, outcomes, *, role="target", parameters=None, capabilities=None):
        self.config = ModelConfig("callback", None, role, parameters=parameters or {}, capabilities=capabilities or {})
        self.outcomes = list(outcomes)
        self.calls = []
        self.probes = []
        self.lock = Lock()

    def generate(self, prompt, **parameters):
        with self.lock:
            self.calls.append((prompt, parameters))
            outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def probe(self):
        self.probes.append(True)
        return CapabilityReport({
            name: CapabilityEvidence(CapabilityStatus.SUPPORTED, "Actual fixture generation", observed=True)
            for name in self.config.inspect().capabilities
        })


def setup(outcomes=None, *, candidate=CANDIDATE, verdict=VERDICT, **workflow_settings):
    target = Client([ModelResponse("up")] * 12 if outcomes is None else outcomes)
    rewriter = Client([ModelResponse(candidate)] * 4, role="rewriter")
    verifier = Client([ModelResponse(verdict)] * 4, role="verifier")
    policy = RewritePolicy(EvidenceRewriter(rewriter), verifier)
    return RecallWorkflow(target, rewrite_policy=policy, **workflow_settings), target, rewriter, verifier


def spec(**changes):
    return EnsembleSpec(**({"draws": 3, "min_parsed": 1, "max_workers": 1, "retain_draws": True} | changes))


@pytest.mark.parametrize("rewrite", [False, True])
@pytest.mark.parametrize("ensemble", [None, spec()])
def test_four_conditions_share_one_configuration_and_keep_explicit_missing_scores(rewrite, ensemble):
    workflow, target, rewriter, verifier = setup()
    result = workflow.run(item(), rewrite=rewrite, ensemble=ensemble, decide=lambda response: response.content)
    assert result.status == RecordStatus.COMPLETED and result.error is None
    assert result.original_evidence == SOURCE
    assert result.accepted_evidence == (CANDIDATE if rewrite else SOURCE)
    assert result.response.content == result.answer == "up"
    assert result.n_requested == result.n_attempted == result.n_usable == (1 if ensemble is None else 3)
    assert (result.ensemble is None) == (ensemble is None)
    assert result.guarded_score is None and result.mia_features is None and result.p_memorized is None
    assert result.unavailable_reasons["mia_features"] and result.unavailable_reasons["guarded_score"]
    assert result.unavailable_reasons["p_memorized"]
    assert result.rewrite_scope == ("shared_per_item" if rewrite else None)
    assert (result.rewrite is None) == (not rewrite)
    assert len(rewriter.calls) == len(verifier.calls) == int(rewrite)
    assert len(rewriter.probes) == len(verifier.probes) == int(rewrite)
    assert len(target.probes) == 1
    assert [draw.draw_index for draw in result.draws] == list(range(result.n_attempted))
    assert all(draw.identity == ("item-1", draw.arm, draw.draw_index) for draw in result.draws)
    assert all(draw.intervention_id == (workflow.rewrite_policy.intervention_id if rewrite else None)
               for draw in result.draws)
    assert all(prompt == render_target_prompt(item(), evidence=result.accepted_evidence) for prompt, _ in target.calls)
    assert result.provenance["request_accounting"]["target"]["status"] == "unverified"


def test_one_workflow_reuses_preflight_for_all_four_conditions_but_one_rewrite_per_run():
    workflow, target, rewriter, verifier = setup()
    for rewrite, ensemble in [(False, None), (False, spec()), (True, None), (True, spec())]:
        assert workflow.run(item(), rewrite=rewrite, ensemble=ensemble).status == RecordStatus.COMPLETED
    assert len(target.probes) == len(rewriter.probes) == len(verifier.probes) == 1
    assert len(rewriter.calls) == len(verifier.calls) == 2


@pytest.mark.parametrize("candidate, verdict", [("Acme rose 3% on 2024-01-02.", VERDICT), (CANDIDATE, "{}")])
def test_failed_rewrite_preserves_partial_record_and_never_falls_back(candidate, verdict):
    workflow, target, rewriter, verifier = setup(candidate=candidate, verdict=verdict)
    result = workflow.run(item(), rewrite=True, ensemble=spec())
    assert result.status == RecordStatus.FAILED and result.response is None and result.ensemble is None
    assert result.rewrite.status == RecordStatus.FAILED
    assert result.rewrite.rewrite_response is not None
    assert result.accepted_evidence is None and result.n_attempted == result.n_usable == 0
    assert target.calls == [] and len(rewriter.calls) == 1


def test_original_and_rewritten_draws_retain_associations_with_failed_parse_and_canonical_order():
    raw = [ModelResponse("up:z"), ModelResponse("invalid"), RequestError("unavailable", status_code=503),
           ModelResponse("down:a"), ModelResponse("up:a")]
    workflow, _, _, _ = setup(raw)

    def decide(response):
        if ":" not in response.content:
            raise ValueError("SECRET_PARSE_INPUT")
        return response.content.split(":")[0]

    result = workflow.run(item(), rewrite=True, ensemble=spec(draws=5, max_transport_failure_ratio=1), decide=decide)
    assert result.status == RecordStatus.COMPLETED
    assert result.n_requested == result.n_attempted == 5 and result.n_usable == 3
    assert [draw.draw_index for draw in result.draws] == [0, 1, 2, 3, 4]
    assert [draw.response for draw in result.draws] == [raw[0], raw[1], None, raw[3], raw[4]]
    assert [draw.status for draw in result.draws] == [RecordStatus.COMPLETED, RecordStatus.FAILED,
                                                  RecordStatus.FAILED, RecordStatus.COMPLETED, RecordStatus.COMPLETED]
    assert result.draws[1].failure_kind == "projection"
    assert "SECRET" not in result.draws[1].error
    assert result.ensemble.draws == (raw[3], raw[4], raw[0])
    replay = reduce_draws([raw[0], raw[3], raw[4]], spec(draws=5, max_transport_failure_ratio=1), decide=decide,
                          n_requested=5, fail_counts={"http": 1, "projection": 1})
    assert result.ensemble.draws_sha256 == replay.draws_sha256
    assert result.ensemble.agreement == replay.agreement and result.ensemble.consensus is raw[4]
    assert len({draw.evidence_hash for draw in result.draws}) == 1


@pytest.mark.parametrize("failure", [BudgetExhaustedError("secret"), RequestError("rejected", status_code=401),
                                    RequestError("bad control", parameter="temperature", code="unsupported_parameter")])
def test_fatal_target_failures_preserve_partial_draws_and_never_return_survivor_consensus(failure):
    raw = ModelResponse("up")
    workflow, target, _, _ = setup([raw, failure, ModelResponse("up")])
    result = workflow.run(item(), ensemble=spec(draws=8, max_transport_failure_ratio=1))
    assert result.status == RecordStatus.FAILED and result.ensemble is None and result.response is None
    assert result.n_requested == 8 and result.n_attempted == 2 and result.n_usable == 1
    assert result.draws[0].response is raw and result.draws[1].status == RecordStatus.FAILED
    assert len(target.calls) == 2


@pytest.mark.parametrize("failure, kind", [
    pytest.param(BudgetExhaustedError("secret"), "budget_exhausted", id="budget"),
    pytest.param(RequestError("rejected", status_code=401), "authentication_error", id="authentication"),
    pytest.param(RequestError("bad control", status_code=400, parameter="temperature", code="unsupported_parameter"),
                 "configuration_error", id="configuration"),
    pytest.param(RequestError("not eligible", status_code=402), "eligibility_error", id="payment-required"),
    pytest.param(RequestError("quota exhausted", status_code=429, code="insufficient_quota"),
                 "eligibility_error", id="hard-quota"),
])
@pytest.mark.parametrize("ensemble", [None, spec(draws=8, max_transport_failure_ratio=1)], ids=["single", "ensemble"])
def test_fatal_answer_projection_failures_retain_draw_kind_raw_response_and_stop(failure, kind, ensemble):
    raw = ModelResponse("bad")
    preceding = [] if ensemble is None else [ModelResponse("up")]
    workflow, target, _, _ = setup(preceding + [raw, ModelResponse("down")])

    def decide(response):
        if response is raw:
            raise failure
        return response.content

    result = workflow.run(item(), ensemble=ensemble, decide=decide)
    assert result.status == RecordStatus.FAILED and result.failure_kind == kind
    assert result.response is None and result.ensemble is None
    assert result.n_attempted == len(target.calls) == len(preceding) + 1
    assert result.n_usable == len(preceding)
    assert [draw.response for draw in result.draws] == preceding + [raw]
    failed_draw = result.draws[-1]
    assert failed_draw.status == RecordStatus.FAILED and failed_draw.failure_kind == kind
    assert failed_draw.error == result.error
    expected_diagnostics = None if isinstance(failure, BudgetExhaustedError) else {
        "status_code": failure.status_code, "parameter": failure.parameter, "code": failure.code,
    }
    assert result.provenance["failure_diagnostics"] == expected_diagnostics


@pytest.mark.parametrize("outcomes, ensemble", [([ModelResponse("bad")], None),
                                              ([ModelResponse("bad")] * 3, spec(min_parsed=2))])
def test_unusable_draws_keep_raw_responses_and_do_not_select_a_later_success(outcomes, ensemble):
    workflow, target, _, _ = setup(outcomes + [ModelResponse("up")])

    def decide(response):
        if response.content == "bad":
            raise ValueError("private source evidence")
        return response.content

    result = workflow.run(item(), ensemble=ensemble, decide=decide)
    assert result.status == RecordStatus.FAILED and result.response is None
    assert result.n_usable == 0 and all(draw.response is not None for draw in result.draws)
    assert len(target.calls) == len(outcomes)
    assert "private source" not in repr(result)


def test_component_failure_preserves_all_outputs_and_cannot_report_success():
    workflow, _, _, _ = setup()

    def components(response):
        raise ValueError("PRIVATE_COMPONENT_INPUT")

    result = workflow.run(item(), ensemble=spec(), components=components)
    assert result.status == RecordStatus.FAILED and result.ensemble is None
    assert len(result.draws) == 3 and all(draw.response for draw in result.draws)
    assert "PRIVATE_COMPONENT_INPUT" not in repr(result)


def test_raw_token_evidence_survives_without_inventing_fitted_scores():
    evidence = (TokenEvidence("up", -0.1, (TokenAlternative("up", -0.1),)),)
    response = ModelResponse("up", token_evidence=evidence)
    workflow, _, _, _ = setup([response])
    result = workflow.run(item())
    assert result.response is response and result.draws[0].response.token_evidence is evidence
    assert result.p_memorized is None and "calibrat" in result.unavailable_reasons["p_memorized"].lower()


def test_duplicate_response_objects_still_have_distinct_draw_identities():
    response = ModelResponse("up")
    workflow, _, _, _ = setup([response] * 3)
    result = workflow.run(item(), ensemble=spec(max_workers=3))
    assert [draw.draw_index for draw in result.draws] == [0, 1, 2]
    assert all(draw.response is response and draw.status == RecordStatus.COMPLETED for draw in result.draws)


def test_preflight_is_explicit_cached_and_refreshable_and_reports_unechoed_settings():
    workflow, target, _, _ = setup()
    first = workflow.preflight()
    assert first.ready_for_batch and first.reports["target"].ready_for_batch
    assert workflow.preflight() == first and len(target.probes) == 1
    workflow.preflight(refresh=True)
    assert len(target.probes) == 2
    result = workflow.run(item())
    assert dict(result.response.confirmed_settings) == {}


def test_required_capability_incompatibility_fails_before_any_probe_or_target():
    unavailable = CapabilityEvidence(CapabilityStatus.UNSUPPORTED, "Fixture declaration")
    target = Client([ModelResponse("up")], capabilities={"text_generation": unavailable})
    with pytest.raises(ValueError, match="Unsupported"):
        RecallWorkflow(target)
    assert target.calls == target.probes == []


@pytest.mark.parametrize("features, ensemble", [(FeatureChoices(mia_scoring=True), None),
                                              (FeatureChoices(ensemble_draws=4), spec(draws=3))])
def test_incompatible_features_are_explicit_before_calls(features, ensemble):
    target = Client([ModelResponse("up")])
    with pytest.raises(ValueError):
        workflow = RecallWorkflow(target, features=features)
        workflow.run(item(), ensemble=ensemble)
    assert target.calls == target.probes == []


def test_rewrite_policy_drift_rejected_before_scheduling_any_new_calls():
    workflow, target, rewriter, verifier = setup()
    rewriter.config = replace(rewriter.config, model="changed")
    with pytest.raises(ValueError, match="changed"):
        workflow.run(item(), rewrite=True)
    assert target.calls == target.probes == verifier.calls == verifier.probes == []


def test_unsupported_rewrite_scope_or_missing_policy_is_rejected():
    target = Client([ModelResponse("up")])
    with pytest.raises(ValueError, match="scope"):
        RecallWorkflow(target, rewrite_scope="per_draw")
    workflow = RecallWorkflow(target)
    with pytest.raises(ValueError, match="policy"):
        workflow.run(item(), rewrite=True)
    assert target.calls == target.probes == []


def test_failed_preflight_never_schedules_target_or_rewriter():
    workflow, target, rewriter, verifier = setup()
    target.probe = lambda: CapabilityReport({"text_generation": CapabilityEvidence("unverified", "No text")})
    result = workflow.run(item(), rewrite=True)
    assert result.status == RecordStatus.FAILED and result.stage == "preflight"
    assert result.preflight.ready_for_batch is False
    assert target.calls == rewriter.calls == verifier.calls == []


def cooperative(role, content, budget, *, max_retries=0):
    config = ModelConfig("callback", None, role, max_retries=max_retries, retry_backoff_s=0)
    calls = []

    def callback(prompt, *, request, **parameters):
        def operation():
            calls.append((prompt, parameters))
            result = content.pop(0) if isinstance(content, list) else content
            if isinstance(result, Exception):
                raise result
            return result
        return request(operation)

    return CallbackClient(config, budget, callback, physical_request_hook="request", strict_budget=True), calls


def test_one_shared_budget_accounts_for_probes_rewrite_verification_target_and_retries():
    budget = CallBudget(20)
    target, calls = cooperative("target", ["OK", TimeoutError("retry"), "up", "up", "up"], budget, max_retries=1)
    rewriter, _ = cooperative("rewriter", ["OK", CANDIDATE], budget)
    verifier, _ = cooperative("verifier", ["OK", VERDICT], budget)
    workflow = RecallWorkflow(target, rewrite_policy=RewritePolicy(EvidenceRewriter(rewriter), verifier),
                              budget=budget, strict_budget=True)
    result = workflow.run(item(), rewrite=True, ensemble=spec())
    assert result.status == RecordStatus.COMPLETED and result.n_attempted == 3
    assert budget.used == 9 and len(calls) == 5
    assert [event.stage for event in budget.events] == ["probe"] * 3 + ["rewrite", "verify"] + ["target"] * 4
    assert [event.attempt for event in budget.events][-4:] == [1, 2, 1, 1]
    assert result.provenance["physical_requests_used"] == 9


def test_budget_exhaustion_in_verifier_retains_generated_rewrite_without_target():
    budget = CallBudget(4)
    target, calls = cooperative("target", "up", budget)
    rewriter, _ = cooperative("rewriter", ["OK", CANDIDATE], budget)
    verifier, _ = cooperative("verifier", ["OK", VERDICT], budget)
    workflow = RecallWorkflow(target, rewrite_policy=RewritePolicy(EvidenceRewriter(rewriter), verifier),
                              budget=budget, strict_budget=True)
    result = workflow.run(item(), rewrite=True)
    assert result.status == RecordStatus.FAILED and result.rewrite.rewrite_response.content == CANDIDATE
    assert result.failure_kind == "budget_exhausted" and result.n_attempted == 0
    assert len(calls) == 1 and budget.used == 4


@pytest.mark.parametrize("kind", ["opaque", "generic", "different_budget"])
def test_strict_accounting_rejects_unknown_or_unshared_clients_before_invocation(kind):
    budget = CallBudget(10)
    if kind == "opaque":
        target = CallbackClient(ModelConfig("callback", None, "target"), budget, lambda prompt: "up")
    elif kind == "generic":
        target = Client([ModelResponse("up")])
    else:
        target, _ = cooperative("target", "up", CallBudget(10))
    with pytest.raises(ValueError, match="accounting|budget"):
        RecallWorkflow(target, budget=budget, strict_budget=True)
    assert budget.used == 0


@pytest.mark.parametrize("client_type, protocol", [(OpenAICompatibleClient, "openai_chat"),
                                                 (AnthropicClient, "anthropic_messages")])
def test_stock_http_adapters_expose_read_only_actual_shared_budget(client_type, protocol):
    budget = CallBudget(10)
    config = ModelConfig(protocol, "https://fixture.example/v1", "target", credential_ref="FIXTURE_KEY",
                         parameters={"max_tokens": 10})
    client = client_type(config, budget, environment={"FIXTURE_KEY": "fixture-value"})
    assert client.budget is budget
    with pytest.raises(AttributeError):
        client.budget = CallBudget(1)


def test_callback_budget_property_is_read_only_and_opaque_accounting_stays_unknown():
    budget = CallBudget(10)
    client = CallbackClient(ModelConfig("callback", None, "target"), budget, lambda prompt: "up")
    assert client.budget is budget and client.request_accounting.status == CapabilityStatus.UNVERIFIED
    with pytest.raises(AttributeError):
        client.budget = CallBudget(1)


@pytest.mark.parametrize("override", [{"max_tokens": 8}, {"temperature": 0.2}])
def test_ensemble_controls_differing_from_configuration_fail_before_probe(override):
    workflow, target, _, _ = setup()
    with pytest.raises(ValueError, match="ModelConfig"):
        workflow.run(item(), ensemble=spec(**override))
    assert target.calls == target.probes == []


def test_matching_explicit_ensemble_controls_use_the_exact_probed_configuration():
    target = Client([ModelResponse("up")] * 3, parameters={"max_tokens": 8, "temperature": 0.2})
    workflow = RecallWorkflow(target)
    result = workflow.run(item(), ensemble=spec(max_tokens=8, temperature=0.2))
    assert result.status == RecordStatus.COMPLETED and len(target.probes) == 1
    assert [parameters for _, parameters in target.calls] == [{"max_tokens": 8, "temperature": 0.2}] * 3


def test_feature_draw_count_requires_explicit_matching_ensemble_instead_of_silent_single():
    target = Client([ModelResponse("up")])
    workflow = RecallWorkflow(target, features=FeatureChoices(ensemble_draws=3))
    with pytest.raises(ValueError, match="EnsembleSpec"):
        workflow.run(item())
    assert target.calls == target.probes == []


def test_opaque_budget_ledger_is_not_reported_as_known_total_physical_requests():
    budget = CallBudget(10)
    client = CallbackClient(ModelConfig("callback", None, "opaque"), budget, lambda prompt: "up")
    result = RecallWorkflow(client).run(item())
    assert result.status == RecordStatus.COMPLETED and budget.used == 0
    assert result.provenance["physical_requests_used"] is None
    assert result.provenance["physical_budget_limit"] is None
    assert result.provenance["observed_budget_requests"] == 0


@pytest.mark.parametrize("metadata", ["config", "requested", "confirmed"])
def test_mid_generation_configuration_change_or_contradictory_settings_stops_and_retains_raw_response(metadata):
    target = Client([ModelResponse("up")], parameters={"temperature": 0.2})
    original_generate = target.generate

    def generate(prompt, **parameters):
        response = original_generate(prompt, **parameters)
        if metadata == "config":
            target.config = replace(target.config, model="changed")
        elif metadata == "requested":
            response = replace(response, requested_settings={"temperature": 0.9})
        else:
            response = replace(response, confirmed_settings={"temperature": 0.9})
        return response

    target.generate = generate
    result = RecallWorkflow(target).run(item(), ensemble=spec(draws=6))
    assert result.status == RecordStatus.FAILED and result.failure_kind == "configuration_error"
    assert result.response is None and result.ensemble is None and result.n_attempted == 1
    assert result.draws[0].response is not None and len(target.calls) == 1


def test_changed_configuration_during_probe_is_not_cached_or_followed_by_inference():
    workflow, target, _, _ = setup()
    original_probe = target.probe

    def probe():
        report = original_probe()
        target.config = replace(target.config, model="changed")
        return report

    target.probe = probe
    result = workflow.run(item())
    assert result.status == RecordStatus.FAILED and result.stage == "preflight"
    assert result.failure_kind == "configuration_error" and target.calls == []


def test_fatal_parameter_diagnostics_are_retained_without_scheduling_later_attempts():
    workflow, target, _, _ = setup([ModelResponse("up"), RequestError("unsupported", parameter="seed", code="bad_seed")])
    result = workflow.run(item(), ensemble=spec(draws=8, max_transport_failure_ratio=1))
    assert result.provenance["failure_diagnostics"] == {"status_code": None, "parameter": "seed", "code": "bad_seed"}
    assert result.failure_kind == "configuration_error" and len(target.calls) == result.n_attempted == 2


def test_transport_failure_ratio_uses_existing_refusal_and_keeps_every_attempt():
    workflow, _, _, _ = setup([ModelResponse("up"), RequestError("unavailable", status_code=503), ModelResponse("up")])
    result = workflow.run(item(), ensemble=spec(max_transport_failure_ratio=0))
    assert result.status == RecordStatus.FAILED and result.n_attempted == 3 and result.n_usable == 2
    assert result.ensemble is None and result.response is None


def test_minimal_client_without_probe_uses_actual_generation_for_preflight():
    target = Client([ModelResponse("OK"), ModelResponse("up")])
    target.probe = None
    result = RecallWorkflow(target).run(item())
    assert result.status == RecordStatus.COMPLETED and len(target.calls) == 2
    assert result.preflight.responses["target"].content == "OK"


@pytest.mark.parametrize("location", ["realized", "alternative"])
def test_censored_generic_client_evidence_is_reported_before_target_inference(location):
    parameters = {"logprobs": True}
    if location == "alternative":
        parameters["top_logprobs"] = 2
    evidence = (TokenEvidence("OK", -9999.0 if location == "realized" else -0.2, (
        TokenAlternative("OK", -9999.0 if location == "alternative" else -0.2),
    )),)
    response = ModelResponse("OK", evidence, requested_settings=parameters)

    class GenericClient:
        config = ModelConfig("callback", None, "target", parameters=parameters)

        def __init__(self):
            self.calls = []

        def generate(self, prompt, **overrides):
            self.calls.append((prompt, overrides))
            return response

    target = GenericClient()
    result = RecallWorkflow(target).run(item())
    assert result.status == RecordStatus.FAILED and result.stage == "preflight"
    assert not result.preflight.ready_for_batch and len(target.calls) == 1
    assert result.n_attempted == 0 and result.draws == ()
    assert result.preflight.responses["target"] is response
    report = result.preflight.reports["target"]
    name = "logprobs" if location == "realized" else "top_logprobs"
    assert report.capabilities[name].status == CapabilityStatus.UNSUPPORTED
    assert "censored" in report.capabilities[name].detail
    assert response.token_evidence == evidence


def test_declared_observed_capabilities_do_not_skip_actual_workflow_preflight():
    evidence = CapabilityEvidence("supported", "Earlier declaration", observed=True)
    target = Client([ModelResponse("up")], capabilities={"text_generation": evidence})
    result = RecallWorkflow(target).run(item())
    assert result.status == RecordStatus.COMPLETED and len(target.probes) == 1


@pytest.mark.parametrize("failure, kind", [(BudgetExhaustedError("secret"), "budget_exhausted"),
                                         (RequestError("rejected", status_code=403), "authentication_error"),
                                         (RequestError("bad", parameter="seed"), "configuration_error")])
def test_fatal_component_failures_retain_classification_and_all_raw_draws(failure, kind):
    workflow, _, _, _ = setup()

    def components(response):
        raise failure

    result = workflow.run(item(), ensemble=spec(), components=components)
    assert result.status == RecordStatus.FAILED and result.failure_kind == kind
    assert result.n_attempted == 3 and all(draw.response is not None for draw in result.draws)
    assert result.response is None and result.ensemble is None


def test_admitted_wave_survivors_remain_raw_aborted_records_after_fatal_stop():
    from threading import Barrier

    target = Client([])
    admitted = Barrier(2)
    lock = Lock()
    calls = []

    def generate(prompt, **parameters):
        with lock:
            index = len(calls)
            calls.append(prompt)
        admitted.wait(timeout=5)
        if index == 0:
            raise RequestError("unsupported", parameter="temperature")
        return ModelResponse("up")

    target.generate = generate
    result = RecallWorkflow(target).run(item(), ensemble=spec(draws=6, max_workers=2,
                                                            max_transport_failure_ratio=1))
    assert result.status == RecordStatus.FAILED and result.n_attempted == 2 and result.n_unattempted == 4
    assert len(calls) == 2 and {draw.draw_index for draw in result.draws} == {0, 1}
    assert all(draw.status == RecordStatus.FAILED for draw in result.draws)
    survivor = next(draw for draw in result.draws if draw.response is not None)
    assert survivor.response.content == "up" and survivor.failure_kind == "aborted"


def test_failed_refreshed_probe_invalidates_prior_success_instead_of_reusing_it():
    workflow, target, _, _ = setup()
    assert workflow.preflight().ready_for_batch

    def failed_probe():
        raise RequestError("credential rejected", status_code=401)

    target.probe = failed_probe
    failed = workflow.preflight(refresh=True)
    assert failed.ready_for_batch is False
    result = workflow.run(item())
    assert result.status == RecordStatus.FAILED and result.stage == "preflight" and target.calls == []


def test_unknown_auxiliary_accounting_leaves_total_unknown_even_with_an_admitted_target():
    budget = CallBudget(10)
    workflow, _, rewriter, verifier = setup()
    target, _ = cooperative("target", "up", budget)
    workflow = RecallWorkflow(target, budget=budget,
                              rewrite_policy=RewritePolicy(EvidenceRewriter(rewriter), verifier))
    result = workflow.run(item(), rewrite=True)
    assert result.status == RecordStatus.COMPLETED and budget.used == 2
    assert result.provenance["physical_requests_used"] is None
    assert result.provenance["physical_budget_limit"] is None
    assert result.provenance["observed_budget_requests"] == 2


def test_preflight_parameter_rejection_preserves_affected_role_and_diagnostics():
    workflow, target, rewriter, _ = setup()

    def rejected_probe():
        raise RequestError("unsupported", status_code=400, parameter="temperature", code="unsupported_parameter")

    rewriter.probe = rejected_probe
    result = workflow.run(item(), rewrite=True)
    assert result.status == RecordStatus.FAILED and result.preflight.affected_role == "rewriter"
    assert result.preflight.diagnostics["parameter"] == "temperature"
    assert target.calls == rewriter.calls == []


def test_target_drift_during_auxiliary_call_preserves_accepted_rewrite_and_requested_provenance():
    workflow, target, rewriter, _ = setup()
    requested = target.config.to_dict()
    original_generate = rewriter.generate

    def generate(prompt, **parameters):
        response = original_generate(prompt, **parameters)
        target.config = replace(target.config, model="changed-during-rewrite")
        return response

    rewriter.generate = generate
    result = workflow.run(item(), rewrite=True)
    assert result.status == RecordStatus.FAILED and result.failure_kind == "configuration_error"
    assert result.rewrite.status == RecordStatus.COMPLETED and result.n_attempted == 0
    assert result.provenance["target"] == requested and target.calls == []


def test_failed_target_drift_result_reports_requested_configuration_instead_of_changed_model():
    workflow, target, _, _ = setup()
    requested = target.config.to_dict()
    original_generate = target.generate

    def generate(prompt, **parameters):
        response = original_generate(prompt, **parameters)
        target.config = replace(target.config, model="changed-after-generation")
        return response

    target.generate = generate
    result = workflow.run(item())
    assert result.status == RecordStatus.FAILED and result.provenance["target"] == requested
    assert result.draws[0].response.content == "up"


@pytest.mark.parametrize("status, code", [(402, None), (429, "insufficient_quota"),
                                        (429, "billing_hard_limit_reached"), (400, "billing_not_active"),
                                        (400, "credit_balance_exhausted"), (429, "insufficient_credits"),
                                        (429, "usage_limit_reached")])
def test_hard_eligibility_failure_after_valid_draw_stops_further_attempts(status, code):
    workflow, target, _, _ = setup([ModelResponse("up"), RequestError("not eligible", status_code=status, code=code)])
    result = workflow.run(item(), ensemble=spec(draws=8, max_transport_failure_ratio=1))
    assert result.status == RecordStatus.FAILED and result.failure_kind == "eligibility_error"
    assert result.n_attempted == len(target.calls) == 2 and result.n_usable == 1
    assert result.response is None and result.ensemble is None
    assert result.provenance["failure_diagnostics"] == {"status_code": status, "code": code, "parameter": None}


def test_transient_rate_limit_stays_an_ordinary_transport_draw_failure():
    workflow, target, _, _ = setup([ModelResponse("up"), RequestError("rate limited", status_code=429),
                                  ModelResponse("down")])
    result = workflow.run(item(), ensemble=spec(max_transport_failure_ratio=1))
    assert result.status == RecordStatus.COMPLETED and result.n_attempted == len(target.calls) == 3
    assert result.n_usable == 2 and result.ensemble.n_parsed == 2


def test_failed_ensemble_keeps_the_requested_reduction_settings_in_provenance():
    workflow, _, _, _ = setup([ModelResponse("up"), RequestError("failed", status_code=503), ModelResponse("up")])
    result = workflow.run(item(), ensemble=spec(min_parsed=3, max_transport_failure_ratio=1))
    assert result.status == RecordStatus.FAILED
    assert result.provenance["ensemble"]["min_parsed"] == result.n_requested == 3
    assert result.provenance["ensemble"]["max_transport_failure_ratio"] == 1
