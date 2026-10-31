"""Matched calibration and supplied-response scoring use the existing estimator."""

import hashlib
import json
from dataclasses import replace

import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from recall_guard.core.completion import (
    CapabilityEvidence,
    CapabilityReport,
    FeatureChoices,
    ModelConfig,
    ModelResponse,
    TokenAlternative,
    TokenEvidence,
)
from recall_guard.core.ensemble import EnsembleSpec
from recall_guard.core.provider_callback import CallbackClient
from recall_guard.core.request_control import CallBudget, RequestError
from recall_guard.harness.intervention_types import RecordStatus
from recall_guard.harness.rewriting import EvidenceRewriter, RewriteInput, RewritePolicy, render_target_prompt
from recall_guard.harness.scorer import MemoryGuardedScorer, ScoredDraw
from recall_guard.harness.workflow import RecallWorkflow

SETTINGS = {"logprobs": True, "top_logprobs": 3, "max_tokens": 512, "temperature": 0.0}
CHECKS = ("claims", "entities", "dates", "quantities", "units", "uncertainty", "qualifications",
          "attribution", "instruction_injection")
VERDICT = json.dumps({"equivalent": True, "checks": dict.fromkeys(CHECKS, True), "discrepancies": []})


def corpus(prefix):
    return [RewriteInput(f"{prefix}-{i}", "Return Direction and Confidence.",
                         f"Acme rose 2% on 2024-01-02; uncertainty remains. {prefix}{i}", "2024-01-02")
            for i in range(8)]


class Client:
    def __init__(self, role="target"):
        self.config = ModelConfig("callback", None, role, parameters=SETTINGS if role == "target" else {})
        self.calls, self.probes = [], []
        self.outcomes = []
        self.mutate = lambda response: response

    def generate(self, prompt, **parameters):
        self.calls.append(prompt)
        if self.outcomes:
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        if prompt == "probe" and self.config.model != "target":
            return ModelResponse("OK")
        if self.config.model == "rewriter":
            source = json.loads(prompt.split("EVIDENCE_DATA_JSON\n")[1])["evidence"]
            return ModelResponse("Let's review: " + source)
        if self.config.model == "verifier":
            return ModelResponse(VERDICT)
        base = -0.2 if "IS" in prompt else -5 if "OOS" in prompt else -1
        jitter = int(hashlib.sha256(prompt.encode()).hexdigest(), 16) % 1000 / 20000
        tokens = tuple(TokenEvidence("x", base + jitter + i * 0.01, (
            TokenAlternative("x", base + jitter), TokenAlternative("y", base - 1),
            TokenAlternative("z", base + 0.1),
        )) for i in range(5))
        return self.mutate(ModelResponse("Direction: 1\nConfidence: 0.8", tokens,
                                         requested_settings=dict(self.config.parameters) | parameters))

    def probe(self):
        self.probes.append(True)
        response = self.generate("probe")
        required = self.config.inspect(FeatureChoices(mia_scoring=self.config.model == "target"))
        return CapabilityReport({name: CapabilityEvidence(
            "unsupported" if name in {"logprobs", "top_logprobs"} and response.mia_unavailable_reason
            else "supported", "Actual fixture generation", observed=True,
        ) for name in required.capabilities}, required.features)


def setup():
    target, rewriter, verifier = Client(), Client("rewriter"), Client("verifier")
    policy = RewritePolicy(EvidenceRewriter(rewriter), verifier)
    return target, policy, rewriter, verifier


def fit(target, *, intervention_id="original"):
    return MemoryGuardedScorer.calibrate_client(
        client=target, is_memorized=[render_target_prompt(i, evidence=i.evidence) for i in corpus("IS")],
        oos_control=[render_target_prompt(i, evidence=i.evidence) for i in corpus("OOS")],
        min_valid=4, max_workers=1, intervention_id=intervention_id,
    )


def spec(**changes):
    return EnsembleSpec(**({"draws": 3, "max_workers": 1, "min_parsed": 2, "retain_draws": True} | changes))


@pytest.mark.parametrize("rewrite", [False, True])
@pytest.mark.parametrize("ensemble", [False, True])
def test_scored_workflow_scores_actual_responses_once_with_matching_fit(rewrite, ensemble, mocker):
    target, policy, rewriter, verifier = setup()
    scorer = fit(target, intervention_id=policy.intervention_id if rewrite else "original")
    workflow = RecallWorkflow(target, rewrite_policy=policy, scorer=scorer)
    spy = mocker.spy(scorer, "score_response")
    calls_before = len(target.calls)
    result = workflow.run(corpus("eval")[0], rewrite=rewrite, ensemble=spec() if ensemble else None)
    assert result.status == RecordStatus.COMPLETED
    assert spy.call_count == (3 if ensemble else 1)
    assert len(target.calls) - calls_before == 1 + (3 if ensemble else 1)  # actual preflight + target draws
    assert len(rewriter.calls) == len(verifier.calls) == (2 if rewrite else 0)
    assert all(draw.guarded_score.parse_ok for draw in result.draws)
    assert all(draw.guarded_score.memguard_confidence == 0.8 * (1 - draw.guarded_score.p_memorized)
               for draw in result.draws)
    assert result.calibration_identity == scorer.calibration_identity
    if ensemble:
        assert result.scored_ensemble.n_parsed == 3
        assert result.p_memorized == result.scored_ensemble.p_memorized_point
        assert result.guarded_score is None and result.mia_features is None
        assert result.scored_ensemble.sampled_at == result.ensemble.sampled_at
    else:
        assert result.guarded_score == result.draws[0].guarded_score
        assert result.mia_features == result.guarded_score.features
        assert result.p_memorized == result.guarded_score.p_memorized


def test_mismatched_scorer_rejected_before_any_probe_or_rewrite_and_explicit_override_works():
    target, policy, rewriter, verifier = setup()
    original, rewritten = fit(target), fit(target, intervention_id=policy.intervention_id)
    workflow = RecallWorkflow(target, rewrite_policy=policy, scorer=original)
    calls = len(target.calls), len(target.probes)
    for operation in [workflow.run, lambda item, **kwargs: workflow.preflight(**kwargs)]:
        with pytest.raises(ValueError, match="identity|intervention"):
            operation(corpus("eval")[0], rewrite=True)
        assert (len(target.calls), len(target.probes)) == calls
        assert not rewriter.calls and not verifier.calls
    assert workflow.run(corpus("eval")[0], rewrite=True, scorer=rewritten).status == RecordStatus.COMPLETED
    unscored = workflow.run(corpus("eval")[0], scorer=None)
    assert unscored.status == RecordStatus.COMPLETED and unscored.p_memorized is None


@pytest.mark.parametrize("censored", [False, True])
def test_missing_selected_evidence_stops_queue_and_retains_raw_responses(censored):
    target, policy, _, _ = setup()
    scorer = fit(target)
    normal = target.generate("eval")
    missing = replace(normal, token_evidence=(TokenEvidence("x", -9999.0, normal.token_evidence[0].top_logprobs),)
                      if censored else None)
    target.outcomes = [normal, normal, missing, normal]  # preflight, draw zero, evidence disappears
    workflow = RecallWorkflow(target, rewrite_policy=policy, scorer=scorer)
    calls_before = len(target.calls)
    result = workflow.run(corpus("eval")[0], ensemble=spec(draws=4))
    assert result.status == RecordStatus.FAILED and result.failure_kind == "configuration_error"
    assert result.p_memorized is None and result.scored_ensemble is None
    assert len(target.calls) - calls_before == 3 and len(result.draws) == 2
    assert result.draws[0].guarded_score.parse_ok and result.draws[1].response == missing
    assert result.draws[1].response.mia_unavailable_reason
    assert result.unavailable_reasons["p_memorized"]


def test_matching_calibration_rewrites_once_and_reuses_each_oos_prompt_with_true_counts():
    target, policy, rewriter, verifier = setup()
    workflow = RecallWorkflow(target, rewrite_policy=policy)
    result = workflow.calibrate(is_memorized=corpus("IS"), oos_control=corpus("OOS"), rewrite=True,
                                min_valid=4, max_workers=1)
    assert result.status == RecordStatus.COMPLETED and result.preparation_complete
    assert result.scorer.calibration_identity == result.identity
    assert result.identity.intervention_id == policy.intervention_id
    assert result.counts["is_memorized"] == {"input": 8, "attempted": 8, "accepted": 8, "training_valid": 8}
    assert result.counts["oos_control"] == {"input": 8, "attempted": 8, "accepted": 8,
                                           "training_valid": 8, "baseline_valid": 8}
    assert result.holdout_auc == result.scorer.holdout_auc and result.is_weak == result.scorer.is_weak
    assert roc_auc_score(result.holdout_labels, result.holdout_probabilities) == result.holdout_auc
    assert result.provenance["holdout_labels"] == result.holdout_labels
    assert result.provenance["holdout_probabilities"] == result.holdout_probabilities
    assert len(rewriter.calls) == len(verifier.calls) == 17  # one probe, one preparation per item
    for record in result.items:
        assert record.rewrite.status == RecordStatus.COMPLETED
        assert record.prompt == render_target_prompt(record.item, evidence=record.rewrite.rewritten_evidence)
        assert target.calls.count(record.prompt) == (2 if record.corpus == "oos_control" else 1)
    assert all("IS-" not in prompt and "OOS-" not in prompt for prompt in rewriter.calls + verifier.calls)
    assert workflow.run(corpus("eval")[0], rewrite=True, scorer=result.scorer).status == RecordStatus.COMPLETED


def test_original_calibration_has_no_auxiliary_calls():
    target, policy, rewriter, verifier = setup()
    result = RecallWorkflow(target, rewrite_policy=policy).calibrate(
        is_memorized=corpus("IS"), oos_control=corpus("OOS"), min_valid=4, max_workers=1)
    assert result.scorer and result.identity.intervention_id == "original"
    assert not rewriter.calls and not verifier.calls
    assert all(record.rewrite is None and record.prompt == render_target_prompt(record.item, evidence=record.item.evidence)
               for record in result.items)


@pytest.mark.parametrize("failure", [RequestError("credits", status_code=402),
                                    RequestError("quota", status_code=429, code="insufficient_quota"),
                                    RequestError("bad", status_code=400, parameter="temperature")])
def test_calibration_auxiliary_fatal_failure_stops_items_and_no_survivor_fit(failure):
    target, policy, rewriter, verifier = setup()
    rewriter.outcomes = [ModelResponse("OK"), ModelResponse("Let's review: " + corpus("IS")[0].evidence), failure]
    result = RecallWorkflow(target, rewrite_policy=policy).calibrate(
        is_memorized=corpus("IS"), oos_control=corpus("OOS"), rewrite=True, min_valid=4, max_workers=1)
    assert result.status == RecordStatus.FAILED and result.scorer is None
    assert result.failure_kind in {"configuration_error", "eligibility_error"}
    assert len(rewriter.calls) == 3 and len(result.items) == 16
    assert result.items[0].rewrite.status == RecordStatus.COMPLETED
    assert result.items[1].rewrite.status == RecordStatus.FAILED
    assert result.counts["is_memorized"]["attempted"] == 2
    assert not any("Evidence (data" in prompt for prompt in target.calls)


def test_nonfatal_rejected_calibration_rows_remain_explicit_with_survivor_counts():
    target, policy, rewriter, _ = setup()
    rewriter.outcomes = [ModelResponse("OK"), ModelResponse("Changed 99% fact")]
    result = RecallWorkflow(target, rewrite_policy=policy).calibrate(
        is_memorized=corpus("IS"), oos_control=corpus("OOS"), rewrite=True, min_valid=4, max_workers=1)
    assert result.status == RecordStatus.COMPLETED and result.scorer is not None
    assert not result.preparation_complete and result.items[0].prompt is None
    assert result.items[0].rewrite.status == RecordStatus.FAILED
    assert result.counts["is_memorized"]["accepted"] == result.counts["is_memorized"]["training_valid"] == 7
    assert all("Changed 99%" not in prompt for prompt in target.calls)


@pytest.mark.parametrize("change", ["text_only", "empty", "wrong_type", "min_valid", "max_workers", "missing_probe",
                                   "negative_seed", "oversized_seed"])
def test_invalid_calibration_choice_makes_no_calls(change):
    target, policy, rewriter, verifier = setup()
    options = {"is_memorized": corpus("IS"), "oos_control": corpus("OOS"), "rewrite": True, "min_valid": 4}
    if change == "text_only":
        target.config = replace(target.config, parameters={"logprobs": False})
    elif change == "empty":
        options["is_memorized"] = []
    elif change == "wrong_type":
        options["oos_control"] = ["not a RewriteInput"]
    elif change == "missing_probe":
        target.probe = None
    elif change in {"negative_seed", "oversized_seed"}:
        options["seed"] = -1 if change == "negative_seed" else 2**32
    else:
        options[change] = 0
    with pytest.raises((ValueError, TypeError)):
        RecallWorkflow(target, rewrite_policy=policy).calibrate(**options)
    assert not target.calls and not rewriter.calls and not verifier.calls


def test_valid_counts_are_actual_feature_rows_rather_than_accepted_preparation_rows(mocker):
    target, policy, _, _ = setup()
    generate = target.generate
    occurrences = {}

    def skip_some(prompt, **parameters):
        response = generate(prompt, **parameters)
        occurrences[prompt] = occurrences.get(prompt, 0) + 1
        if (("OOS3" in prompt and occurrences[prompt] == 1)  # baseline
                or "IS1" in prompt  # training IS
                or "OOS4" in prompt and occurrences[prompt] == 2):  # training OOS
            raise TimeoutError("ordinary row timeout")
        return response

    mocker.patch.object(target, "generate", side_effect=skip_some)
    result = RecallWorkflow(target, rewrite_policy=policy).calibrate(
        is_memorized=corpus("IS"), oos_control=corpus("OOS"), min_valid=4, max_workers=1)
    assert result.status == RecordStatus.COMPLETED and result.preparation_complete
    assert result.counts["is_memorized"]["accepted"] == result.counts["oos_control"]["accepted"] == 8
    assert result.counts["is_memorized"]["training_valid"] == 7
    assert result.counts["oos_control"]["training_valid"] == result.counts["oos_control"]["baseline_valid"] == 7


def test_answer_consensus_scored_representative_and_mean_multiplier_are_distinct_and_replayable():
    target, policy, _, _ = setup()
    scorer = fit(target)
    low, middle, high = [target.generate(text) for text in ("OOS0", "OOS1", "IS0")]
    low = replace(low, content="Direction: -1\nConfidence: 0.7")
    middle = replace(middle, content="Direction: 1\nConfidence: 0.8\nZulu")
    high = replace(high, content="Direction: 1\nConfidence: 0.6\nAlpha")
    target.outcomes = [target.generate("probe"), high, low, middle, high]
    workflow = RecallWorkflow(target, rewrite_policy=policy, scorer=scorer)
    ensemble_spec = spec(draws=4, max_workers=2)
    result = workflow.run(corpus("eval")[0], ensemble=ensemble_spec,
                          decide=lambda response: "up" if "Direction: 1" in response.content else "down")
    assert result.status == RecordStatus.COMPLETED and result.answer == "up"
    assert result.response == high
    assert result.scored_ensemble.consensus.p_memorized != result.p_memorized
    assert result.scored_ensemble.consensus.raw_confidence != result.draws[0].guarded_score.raw_confidence
    before = len(target.calls)
    scored_draws = [ScoredDraw(draw.guarded_score, draw.response.content, draw.draw_index // ensemble_spec.max_workers)
                    for draw in result.draws]
    replay = scorer.reduce_scored_draws(result.draws[0].prompt, list(reversed(scored_draws)), spec=ensemble_spec,
                                        sampled_at=result.ensemble.sampled_at)
    assert replay == result.scored_ensemble
    assert len(target.calls) == before
    assert scorer.reduce_scored_draws(result.draws[0].prompt, scored_draws, spec=ensemble_spec).sampled_at is None


def test_scored_replay_matches_legacy_ensemble_including_failures(mocker):
    target = Client()
    scorer = fit(target)
    prompt = "eval"
    raw = [target.generate(text) for text in ("IS0", "OOS0", "OOS1")]
    target.outcomes = [raw[0], TimeoutError(), raw[1], raw[2]]
    ensemble_spec = spec(draws=4)
    live = scorer.score_ensemble(prompt, spec=ensemble_spec, conservative_quantile=0.9)
    scored = [ScoredDraw(scorer.score_response(prompt, response, client=target), response.content, wave)
              for response, wave in zip(raw, [0, 2, 3], strict=True)]
    no_calls = mocker.patch.object(target, "generate", side_effect=AssertionError("replay must not generate"))
    replay = scorer.reduce_scored_draws(prompt, scored, spec=ensemble_spec, failures={"timeout": 1},
                                        conservative_quantile=0.9, sampled_at=live.sampled_at)
    assert replay == live
    no_calls.assert_not_called()


@pytest.mark.parametrize("kind", ["missing", "censored"])
def test_selected_fallback_probe_checks_token_evidence_with_omitted_logprob_flags(kind):
    target, policy, _, _ = setup()
    target.config = replace(target.config, parameters={})
    scorer = fit(target)
    target.probe = None
    normal = target.generate("probe")
    target.outcomes = [replace(normal, token_evidence=None if kind == "missing" else
                               (TokenEvidence("x", -9999.0, normal.token_evidence[0].top_logprobs),))]
    before = len(target.calls)
    result = RecallWorkflow(target, rewrite_policy=policy, scorer=scorer).run(corpus("eval")[0])
    assert result.status == RecordStatus.FAILED and result.failure_kind == "capability_unavailable"
    assert len(target.calls) == before + 1 and not result.draws
    assert result.preflight.responses["target"].mia_unavailable_reason


def test_scored_parse_failures_have_independent_valid_counts_and_enforce_minimum():
    target, _, _, _ = setup()
    scorer = fit(target)
    valid = target.generate("eval")
    invalid = replace(valid, content="Direction missing from this answer")
    target.outcomes = [valid, valid, invalid, valid]
    workflow = RecallWorkflow(target, scorer=scorer)
    result = workflow.run(corpus("eval")[0], ensemble=spec())
    assert result.status == RecordStatus.COMPLETED and result.n_usable == 3
    assert result.scored_ensemble.n_parsed == 2 and result.scored_ensemble.fail_counts == (("parse_failure", 1),)
    assert not result.draws[1].guarded_score.parse_ok
    target.outcomes = [valid, invalid, valid]
    failed = workflow.run(corpus("eval")[0], ensemble=spec(min_parsed=3))
    assert failed.status == RecordStatus.FAILED and failed.failure_kind == "scoring_unavailable"
    assert failed.scored_ensemble.n_parsed == 2 and failed.p_memorized is None
    assert all(draw.response is not None for draw in failed.draws)


def test_selected_scoring_preserves_nonfatal_target_failure_classification_and_live_timestamp():
    target, _, _, _ = setup()
    scorer = fit(target)
    normal = target.generate("eval")
    target.outcomes = [normal, normal, TimeoutError(), TimeoutError()]
    result = RecallWorkflow(target, scorer=scorer).run(corpus("eval")[0], ensemble=spec())
    assert result.status == RecordStatus.FAILED and result.failure_kind == "runtime_error"
    assert result.scored_ensemble.n_parsed == 1 and result.p_memorized is None
    assert result.scored_ensemble.sampled_at == result.sampled_at and result.sampled_at is not None


@pytest.mark.parametrize("failure", [RequestError("hard credits", status_code=402),
                                    RequestError("auth", status_code=401)])
def test_fit_failure_does_not_return_survivor_calibration_or_hide_category(failure, mocker):
    target, _, _, _ = setup()
    generate = target.generate

    def fail_after_preparation(prompt, **parameters):
        response = generate(prompt, **parameters)
        if "IS2" in prompt:
            raise failure
        return response

    mocker.patch.object(target, "generate", side_effect=fail_after_preparation)
    result = RecallWorkflow(target).calibrate(is_memorized=corpus("IS"), oos_control=corpus("OOS"),
                                              min_valid=4, max_workers=1)
    assert result.status == RecordStatus.FAILED and result.scorer is None
    assert result.failure_kind == ("eligibility_error" if failure.status_code == 402 else "authentication_error")
    assert result.preparation_complete
    assert result.counts["is_memorized"]["training_valid"] is None
    assert result.diagnostics["status_code"] == failure.status_code
    assert not any("IS3" in prompt for prompt in target.calls)


def test_late_billing_failure_stops_before_fitting_viable_survivors(mocker):
    target, _, _, _ = setup()
    generate = target.generate
    failure = RequestError("credits exhausted", status_code=402, code="insufficient_credits",
                           detail="Fund account", usage={"input_tokens": 11}, retryable=False)
    fit_spy = mocker.spy(LogisticRegression, "fit")
    split_spy = mocker.patch("recall_guard.mia.mcs.train_test_split", wraps=train_test_split)

    def fail_late_in_oos_training(prompt, **parameters):
        response = generate(prompt, **parameters)
        # Two probes, eight baseline rows, eight IS and five OOS training rows
        # have succeeded: enough survivors for sklearn to fit without a guard.
        if len(target.calls) == 24:
            assert "OOS5" in prompt
            raise failure
        return response

    mocker.patch.object(target, "generate", side_effect=fail_late_in_oos_training)
    result = RecallWorkflow(target).calibrate(is_memorized=corpus("IS"), oos_control=corpus("OOS"),
                                              min_valid=4, max_workers=1)
    assert result.status == RecordStatus.FAILED and result.scorer is None
    assert result.failure_kind == "eligibility_error" and result.error == "credits exhausted"
    assert result.diagnostics == {"status_code": 402, "code": "insufficient_credits", "parameter": None,
                                  "detail": "Fund account", "retryable": False, "usage": {"input_tokens": 11}}
    assert result.preparation_complete and len(result.items) == 16
    assert result.counts["is_memorized"]["training_valid"] is None
    assert result.counts["oos_control"]["training_valid"] is None
    assert len(target.calls) == 24 and not any("OOS6" in prompt for prompt in target.calls[18:])
    fit_spy.assert_not_called()
    split_spy.assert_not_called()


def test_strict_calibration_and_inference_count_all_actual_hooks_including_both_probes():
    fixture = Client()
    budget = CallBudget(27)  # calibration 2 probes + 8 baseline + 16 training, then cached preflight + one draw

    def callback(prompt, request, **parameters):
        return request(lambda: fixture.generate(prompt, **parameters))

    target = CallbackClient(fixture.config, budget, callback, physical_request_hook="request", strict_budget=True)
    workflow = RecallWorkflow(target, budget=budget, strict_budget=True)
    calibration = workflow.calibrate(is_memorized=corpus("IS"), oos_control=corpus("OOS"),
                                      min_valid=4, max_workers=1)
    assert calibration.status == RecordStatus.COMPLETED and budget.used == 26
    assert calibration.provenance["physical_requests_used"] == 26
    result = workflow.run(corpus("eval")[0], scorer=calibration.scorer)
    assert result.status == RecordStatus.COMPLETED and budget.used == 27
    assert result.provenance["physical_requests_used"] == 27


def test_generic_target_failure_retains_sanitized_usage_detail_and_retryability():
    target, _, _, _ = setup()
    failure = RequestError("quota", status_code=402, code="insufficient_credits", detail="Fund account",
                           usage={"input_tokens": 11}, retryable=False)
    target.outcomes = [target.generate("probe"), failure]
    result = RecallWorkflow(target).run(corpus("eval")[0])
    assert result.failure_kind == "eligibility_error"
    diagnostics = result.provenance["failure_diagnostics"]
    assert diagnostics["detail"] == "Fund account" and diagnostics["retryable"] is False
    assert diagnostics["usage"] == {"input_tokens": 11}
    assert result.provenance["physical_requests_used"] is None


def test_nonfatal_draw_failure_retains_provider_usage_when_the_ensemble_succeeds():
    target, _, _, _ = setup()
    normal = target.generate("eval")
    failure = RequestError("busy", status_code=503, detail="Try later", retryable=True, usage={"output_tokens": 2})
    target.outcomes = [normal, normal, failure, normal]
    result = RecallWorkflow(target).run(corpus("eval")[0], ensemble=spec(max_transport_failure_ratio=1))
    assert result.status == RecordStatus.COMPLETED
    assert result.draws[1].diagnostics == {
        "status_code": 503, "code": None, "parameter": None, "detail": "Try later", "retryable": True,
        "usage": {"output_tokens": 2},
    }
