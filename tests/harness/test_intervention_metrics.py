"""Analytic item-paired fixtures, failure denominators and physical resource accounting."""

import json
from dataclasses import asdict, replace

import pytest

from recall_guard.core.completion import ModelResponse
from recall_guard.core.request_control import RequestEvent
from recall_guard.harness.intervention_journal import ExperimentArm, ExperimentRun
from recall_guard.harness.intervention_metrics import summarize_experiment
from recall_guard.harness.intervention_types import (
    DrawRecord,
    EvidenceArm,
    ExperimentCheckpoint,
    ExperimentItem,
    RecordStatus,
    RequestRecord,
    RewriteRecord,
    text_hash,
)
from recall_guard.harness.scorer_client import CalibrationIdentity
from tests.harness.test_intervention import protocol

CONDITIONS = ("original_single", "original_ensemble", "rewritten_single", "rewritten_ensemble")


def fixture(outcomes, *, cohorts=None, rejected=(), pending=(), failed_zero=(), missing=()):
    """Rows give [original single, ensemble, rewritten single, ensemble] answers.

    Label is always 'up'; draw 1 supplies the ensemble answer independently of
    the predesignated draw 0. No statistics implementation is used by fixtures.
    """
    cohorts = cohorts or ["sample"] * len(outcomes)
    p = protocol(draws=2, cohort_definitions={name: {} for name in cohorts}, bootstrap_seed=17,
                 bootstrap_count=400)
    items = tuple(ExperimentItem(str(i), "Choose", "Evidence", "2024-01-01", "up", cohorts[i])
                  for i in range(len(outcomes)))
    draws, rewrites, arms = [], [], []
    for item, answers in zip(items, outcomes, strict=True):
        rejected_item = item.item_id in rejected
        rewrite_pending = (item.item_id, "rewritten") in pending
        rewrites.append(RewriteRecord(
            item.item_id, item.evidence,
            RecordStatus.PENDING if rewrite_pending else RecordStatus.FAILED if rejected_item else RecordStatus.COMPLETED,
            rewritten_evidence=None if rewrite_pending else "Evidence rewritten",
            fidelity_checks={} if rewrite_pending else {
                "deterministic": {"status": "rejected" if rejected_item else "passed"},
                "semantic": {"status": "not_performed" if rejected_item else "passed"}},
            error="Fidelity failed" if rejected_item else None))
        for arm_index, arm in enumerate(EvidenceArm):
            is_pending = (item.item_id, arm.value) in pending
            is_rejected = arm == EvidenceArm.REWRITTEN and rejected_item
            for draw_index in range(2):
                if (item.item_id, arm.value, draw_index) in missing:
                    continue
                failed = is_rejected or (item.item_id, arm.value) in failed_zero and draw_index == 0
                answer = None if failed or is_pending else answers[arm_index * 2 + draw_index]
                status = RecordStatus.PENDING if is_pending else RecordStatus.FAILED if failed else RecordStatus.COMPLETED
                draws.append(DrawRecord(item.item_id, arm, draw_index, status,
                                        prompt=None if is_rejected or is_pending else "prompt",
                                        response=None if is_rejected or is_pending else ModelResponse("raw"),
                                        error="failed" if failed else None,
                                        metadata={"attempted": not is_rejected and not is_pending,
                                                  "answer": answer,
                                                  "components": None if answer is None else {"confidence": 0.8},
                                                  "failure_kind": "fidelity_rejected" if is_rejected else
                                                      "projection" if failed else None}))
            if not is_pending:
                arms.append(ExperimentArm(item.item_id, arm, {
                    "status": "failed" if is_rejected else "completed",
                    "answer": None if is_rejected else answers[arm_index * 2 + 1],
                    "failure_kind": "fidelity_rejected" if is_rejected else None,
                    "ensemble": None if is_rejected else {"agreement": 0.75, "agreement_ci": [0.2, 1]},
                    "scored_ensemble": None, "provenance": {},
                }))
    status = RecordStatus.PENDING if pending or missing else RecordStatus.COMPLETED
    return ExperimentRun(ExperimentCheckpoint(p, items, status, tuple(draws), tuple(rewrites)), tuple(arms))


def test_identical_paired_results_have_exact_zero_effect_despite_varying_accuracy():
    run = fixture([["up"] * 4, ["down"] * 4, ["up"] * 4, ["down"] * 4])
    metrics = summarize_experiment(run)
    assert metrics["provenance"]["condition_order"] == list(CONDITIONS)
    assert metrics["pooled"]["conditions"]["original_single"]["accuracy"]["all_attempted"]["estimate"] == 0.5
    for contrast in metrics["pooled"]["effects"].values():
        for scope in ("all_attempted", "valid_pairs"):
            assert contrast[scope]["estimate"] == 0
            assert contrast[scope]["interval"] == [0, 0]
            assert contrast[scope]["n_eligible"] == 4
            assert contrast[scope]["interval_status"] == "degenerate"
            assert contrast[scope]["equivalence_proof"] is False


def test_separate_rewriting_and_aggregation_analytic_effects_and_cohorts():
    run = fixture([["down", "up", "up", "up"], ["down", "down", "down", "up"],
                   ["up", "up", "up", "up"], ["up", "down", "down", "down"]],
                  cohorts=["early", "early", "late", "late"])
    metrics = summarize_experiment(run)
    effects = metrics["pooled"]["effects"]
    assert effects["rewriting_single"]["valid_pairs"]["estimate"] == 0
    assert effects["rewriting_ensemble"]["valid_pairs"]["estimate"] == 0.25
    assert effects["aggregation_original"]["valid_pairs"]["estimate"] == 0
    assert effects["aggregation_rewritten"]["valid_pairs"]["estimate"] == 0.25
    assert metrics["cohorts"]["early"]["effects"]["rewriting_single"]["valid_pairs"]["estimate"] == 0.5
    assert metrics["historical"]["available"] is False  # Names do not establish sourced cutoffs.


def test_identity_pairing_and_bootstrap_replay_ignore_record_order():
    run = fixture([["up", "up", "down", "up"], ["down", "down", "up", "down"],
                   ["up", "down", "up", "up"]])
    reordered = replace(run, checkpoint=replace(run.checkpoint, items=tuple(reversed(run.checkpoint.items)),
                                                draws=tuple(reversed(run.checkpoint.draws)),
                                                rewrites=tuple(reversed(run.checkpoint.rewrites))),
                        arms=tuple(reversed(run.arms)))
    metrics = summarize_experiment(run)
    assert metrics["pooled"] == summarize_experiment(reordered)["pooled"]
    assert metrics == summarize_experiment(ExperimentRun.from_dict(run.to_dict()))
    assert metrics["provenance"]["bootstrap"] == {
        "method": "percentile_paired_item_bootstrap", "unit": "item", "level": 0.95,
        "seed": 17, "count": 400, "sampling": "unstratified_within_each_eligible_comparison",
        "joint_intervals": False}


def test_rejected_rewrites_remain_in_all_attempted_and_block_success():
    run = fixture([["up"] * 4, ["up"] * 4, ["down"] * 4], rejected=("0", "1"))
    metrics = summarize_experiment(run)["pooled"]
    rewritten = metrics["conditions"]["rewritten_single"]
    assert rewritten["coverage"]["terminal"] == 3
    assert rewritten["coverage"]["target_attempted"] == 1
    assert rewritten["accuracy"]["all_attempted"]["estimate"] == 0
    effect = metrics["effects"]["rewriting_single"]
    assert effect["all_attempted"]["n_eligible"] == 3
    assert effect["all_attempted"]["estimate"] == pytest.approx(-2 / 3)
    assert effect["valid_pairs"]["n_eligible"] == 1
    assert effect["valid_pairs"]["estimate"] == 0
    assert metrics["fidelity"]["rejected"] == 2
    assert metrics["interpretation"]["gates"]["fidelity"] == "failed"
    assert metrics["interpretation"]["gates"]["utility"] == "failed"
    assert metrics["interpretation"]["mitigation_success"] is False


def test_failed_draw_zero_is_not_replaced_by_an_ensemble_draw():
    result = summarize_experiment(fixture([["up"] * 4, ["up"] * 4], failed_zero=(("0", "original"),)))
    conditions = result["pooled"]["conditions"]
    assert conditions["original_single"]["accuracy"]["all_attempted"]["estimate"] == 0.5
    assert conditions["original_ensemble"]["accuracy"]["all_attempted"]["estimate"] == 1
    effect = result["pooled"]["effects"]["aggregation_original"]
    assert effect["all_attempted"]["estimate"] == 0.5
    assert effect["valid_pairs"]["estimate"] == 0
    assert effect["valid_pairs"]["n_eligible"] == 1


def test_pending_and_missing_coverage_are_not_observed_model_errors():
    run = fixture([["up"] * 4, ["up"] * 4], pending=(("1", "rewritten"),),
                  missing=(("1", "original", 0),))
    metrics = summarize_experiment(run)["pooled"]
    original = metrics["conditions"]["original_single"]
    assert original["coverage"]["planned"] == 2
    assert original["coverage"]["incomplete"] == 1
    assert original["accuracy"]["all_attempted"]["denominator"] == 1
    assert original["accuracy"]["all_attempted"]["estimate"] == 1
    effect = metrics["effects"]["rewriting_single"]["all_attempted"]
    assert effect["n_eligible"] == 1 and effect["n_incomplete"] == 1
    assert effect["interval"] is None and effect["interval_status"] == "inadequate_items"
    assert metrics["interpretation"]["gates"]["coverage"] == "incomplete"


def test_zero_valid_pairs_are_unavailable_not_a_fabricated_zero_effect():
    metrics = summarize_experiment(fixture([["up"] * 4], rejected=("0",)))["pooled"]
    effect = metrics["effects"]["rewriting_single"]["valid_pairs"]
    assert effect["estimate"] is None and effect["interval"] is None
    assert effect["n_eligible"] == 0 and effect["interval_status"] == "unavailable"
    assert metrics["fidelity"]["semantic_unperformed"] == 1


def test_explicit_confidence_is_separate_from_real_scoring_and_declares_defaults():
    run = fixture([["up"] * 4, ["up"] * 4])
    draws = []
    for draw in run.checkpoint.draws:
        metadata = dict(draw.metadata)
        if draw.arm == EvidenceArm.REWRITTEN:
            metadata["components"] = {"confidence": 0.6}
        draws.append(replace(draw, metadata=metadata))
    metrics = summarize_experiment(replace(run, checkpoint=replace(run.checkpoint, draws=tuple(draws))))
    assert metrics["provenance"]["confidence"]["component"] == "confidence"
    assert metrics["provenance"]["confidence"]["ensemble_aggregation"] == "mean_of_explicit_draw_confidences"
    effect = metrics["pooled"]["effects"]["rewriting_single"]["confidence"]
    assert effect["estimate"] == pytest.approx(-0.2) and effect["n_eligible"] == 2
    for condition in metrics["pooled"]["conditions"].values():
        assert condition["scoring"]["p_memorized"]["mean"] is None
        assert condition["coverage"]["scored_usable"] == 0
    json.dumps(metrics, allow_nan=False)


def test_answer_and_scored_usability_are_independent_and_discounts_are_per_draw():
    run = fixture([["up"] * 4, ["up"] * 4])
    draws = []
    for draw in run.checkpoint.draws:
        score = {"parse_ok": True, "raw_confidence": 0.2 if draw.draw_index == 0 else 1.0,
                 "memguard_confidence": 0.1 if draw.draw_index == 0 else 0.1,
                 "p_memorized": 0.5 if draw.draw_index == 0 else 0.9}
        draws.append(replace(draw, metadata=dict(draw.metadata) | {"guarded_score": score, "components": None}))
    arms = tuple(replace(arm, outcome=arm.outcome | {
        "status": "failed", "failure_kind": "scoring_unavailable", "scored_ensemble": {
            "p_memorized_point": None}}) for arm in run.arms)
    metrics = summarize_experiment(replace(run, arms=arms, checkpoint=replace(run.checkpoint, draws=tuple(draws))))
    condition = metrics["pooled"]["conditions"]["original_ensemble"]
    assert condition["coverage"]["answer_usable"] == 2
    assert condition["coverage"]["scored_usable"] == 0
    assert condition["accuracy"]["all_attempted"]["estimate"] == 0  # selected workflow failed
    assert condition["accuracy"]["usable_answers"]["estimate"] == 1  # parsed answer retained
    assert condition["scoring"]["per_draw_discount"]["mean"] == pytest.approx(0.5)
    assert condition["scoring"]["per_draw_discount"]["n"] == 4
    assert condition["confidence"]["sources"] == {"guarded_score.raw_confidence": 4}


def scored_fixture(probabilities):
    run = fixture([["up"] * 4 for _ in probabilities])
    draws = tuple(replace(draw, metadata=dict(draw.metadata) | {"guarded_score": {
        "parse_ok": True, "raw_confidence": 0.8, "p_memorized": probabilities[int(draw.item_id)],
        "memguard_confidence": 0.8 * (1 - probabilities[int(draw.item_id)]),
    }}) for draw in run.checkpoint.draws)
    arms = tuple(replace(arm, outcome=arm.outcome | {"scored_ensemble": {
        "p_memorized_point": probabilities[int(arm.item_id)]}}) for arm in run.arms)
    return replace(run, checkpoint=replace(run.checkpoint, draws=draws), arms=arms)


def invalidate_scored_item(run, item_id):
    # The runner retains raw responses/scores after post-arm calibration drift,
    # but explicitly invalidates the draw and clears aggregate output fields.
    draws = tuple(replace(draw, status=RecordStatus.FAILED, error="Calibration changed",
                          metadata=dict(draw.metadata) | {"failure_kind": "configuration_error"})
                  if draw.item_id == item_id else draw for draw in run.checkpoint.draws)
    arms = tuple(replace(arm, outcome=arm.outcome | {
        "status": "failed", "failure_kind": "configuration_error", "answer": None,
        "ensemble": None, "scored_ensemble": None}) if arm.item_id == item_id else arm for arm in run.arms)
    return replace(run, checkpoint=replace(run.checkpoint, draws=draws), arms=arms)


def test_configuration_drift_keeps_retained_mcs_audit_only():
    run = invalidate_scored_item(scored_fixture([0.9]), "0")
    raw = run.to_dict()
    metrics = summarize_experiment(run)
    for row in metrics["pooled"]["items"]:
        assert row["scored_usable"] is False
        assert row["p_memorized"] is None
    for condition in metrics["pooled"]["conditions"].values():
        assert condition["coverage"]["scored_usable"] == 0
        assert condition["scoring"]["p_memorized"] == {"mean": None, "n": 0}
        assert condition["scoring"]["per_draw_discount"] == {"mean": None, "n": 0}
    assert run.to_dict() == raw
    replay = ExperimentRun.from_dict(json.loads(json.dumps(raw)))
    assert all(draw.metadata["guarded_score"]["p_memorized"] == 0.9 for draw in replay.checkpoint.draws)
    assert all(draw.response.content == "raw" for draw in replay.checkpoint.draws)
    assert summarize_experiment(replay) == metrics


def test_mixed_valid_and_invalid_mcs_aggregation_excludes_invalid_items_after_json_replay():
    run = invalidate_scored_item(scored_fixture([0.2, 0.9]), "1")
    metrics = summarize_experiment(run)
    for name, condition in metrics["pooled"]["conditions"].items():
        assert condition["coverage"]["terminal"] == 2
        assert condition["coverage"]["scored_usable"] == 1
        assert condition["scoring"]["p_memorized"] == {"mean": 0.2, "n": 1}
        assert condition["scoring"]["per_draw_raw_confidence"]["n"] == (1 if name.endswith("single") else 2)
    assert all(row["p_memorized"] is None for row in metrics["pooled"]["items"] if row["item_id"] == "1")
    replay = ExperimentRun.from_dict(json.loads(json.dumps(run.to_dict())))
    assert summarize_experiment(replay) == metrics
    assert all(draw.metadata["guarded_score"]["p_memorized"] == 0.9
               for draw in replay.checkpoint.draws if draw.item_id == "1")


@pytest.mark.parametrize("pending", [False, True], ids=["failed-score-parse", "pending-score"])
def test_unusable_single_score_has_no_published_numeric_value(pending):
    run = scored_fixture([0.9])
    draws = tuple(replace(draw, status=RecordStatus.PENDING if pending else RecordStatus.FAILED,
                          response=None if pending else draw.response, error=None if pending else "Score parse failed",
                          metadata=dict(draw.metadata) | {"failure_kind": None if pending else "scoring_unavailable",
                              "guarded_score": dict(draw.metadata["guarded_score"]) | {"parse_ok": pending}})
                  for draw in run.checkpoint.draws)
    run = replace(run, checkpoint=replace(run.checkpoint, draws=draws,
                                          status=RecordStatus.PENDING if pending else RecordStatus.COMPLETED), arms=())
    metrics = summarize_experiment(run)["pooled"]
    for row in metrics["items"]:
        assert row["scored_usable"] is False
        assert row["p_memorized"] is None
    for name in ("original_single", "rewritten_single"):
        condition = metrics["conditions"][name]
        assert condition["scoring"]["p_memorized"] == {"mean": None, "n": 0}
        assert condition["coverage"]["answer_usable"] == (0 if pending else 1)


def test_valid_draw_score_survives_answer_projection_failure_and_missing_ensemble():
    run = scored_fixture([0.2])
    draws = tuple(replace(draw, status=RecordStatus.FAILED, error="Answer projection failed",
                          metadata=dict(draw.metadata) | {"answer": None, "failure_kind": "projection"})
                  for draw in run.checkpoint.draws)
    run = replace(run, checkpoint=replace(run.checkpoint, draws=draws), arms=())
    conditions = summarize_experiment(run)["pooled"]["conditions"]
    for name, condition in conditions.items():
        assert condition["coverage"]["answer_usable"] == 0
        assert condition["scoring"]["per_draw_raw_confidence"]["n"] == (1 if name.endswith("single") else 2)
        assert condition["scoring"]["p_memorized"] == (
            {"mean": 0.2, "n": 1} if name.endswith("single") else {"mean": None, "n": 0})


def test_failure_gate_blocks_success_even_when_all_attempted_accuracy_improves():
    metrics = summarize_experiment(fixture([["down"] * 4, ["down", "down", "up", "up"],
                                           ["down", "down", "up", "up"]], rejected=("0",)))["pooled"]
    assert metrics["effects"]["rewriting_single"]["all_attempted"]["estimate"] == pytest.approx(2 / 3)
    assert metrics["interpretation"]["gates"]["utility"] == "within_observed_tolerance"
    assert metrics["interpretation"]["gates"]["failures"] == "failed"
    assert metrics["interpretation"]["mitigation_success"] is False


def test_nonconstant_pair_interval_resamples_items_not_individual_conditions():
    run = fixture([["up"] * 4, ["down", "down", "up", "up"]])
    metrics = summarize_experiment(run, confidence_level=0.9)
    effect = metrics["pooled"]["effects"]["rewriting_single"]["valid_pairs"]
    assert effect["estimate"] == 0.5
    # Two paired deltas are 0 and 1. An item bootstrap has support {0,.5,1}
    # with masses {1/4,1/2,1/4}; both tails here must include 0 and 1.
    assert effect["interval"] == [0, 1]
    assert metrics["provenance"]["bootstrap"]["level"] == 0.9


def test_inadequate_bootstrap_count_does_not_fabricate_a_precision_interval():
    run = fixture([["up"] * 4, ["down", "down", "up", "up"]])
    run = replace(run, checkpoint=replace(run.checkpoint, protocol=replace(run.checkpoint.protocol,
                                                                         bootstrap_count=1)))
    effect = summarize_experiment(run)["pooled"]["effects"]["rewriting_single"]["valid_pairs"]
    assert effect["estimate"] == 0.5 and effect["interval"] is None
    assert effect["interval_status"] == "inadequate_resamples"


def test_json_sequence_labels_replay_without_tuple_list_accuracy_drift():
    run = fixture([[("up",)] * 4, [("up",)] * 4])
    run = replace(run, checkpoint=replace(run.checkpoint, items=tuple(replace(item, label=("up",))
                                                                    for item in run.checkpoint.items)))
    before = summarize_experiment(run)
    after = summarize_experiment(ExperimentRun.from_dict(json.loads(json.dumps(run.to_dict()))))
    assert before == after
    assert before["pooled"]["conditions"]["original_single"]["accuracy"]["all_attempted"]["estimate"] == 1


def test_calibration_quality_is_reported_as_scoring_evidence():
    run = fixture([["up"] * 4])
    quality = {"identity": {"model": "target", "intervention_id": None}, "holdout_auc": 0.6,
               "is_weak": True, "valid_counts": {"is_holdout": 4, "oos_holdout": 4}}
    arms = tuple(replace(arm, outcome=arm.outcome | {"provenance": {"scoring": quality}})
                 for arm in run.arms)
    condition = summarize_experiment(replace(run, arms=arms))["pooled"]["conditions"]["original_single"]
    assert condition["scoring"]["calibration_quality"] == [quality]
    assert condition["scoring"]["items_with_calibration_quality"] == 1


#: A calibrator's identity as the legacy five-key payload, and as the ``asdict`` export
#: that runs saved after the source extension: ten keys, the new ones at their defaults.
LEGACY_IDENTITY = {"protocol": "openai_chat", "endpoint": "http://127.0.0.1:11570/v1/chat/completions",
                   "model": "target", "parameters_json": '{"logprobs":true,"top_logprobs":5}',
                   "intervention_id": "recorded-method"}
ASDICT_IDENTITY = LEGACY_IDENTITY | {"local_identity": None, "response_mapping": "generic",
                                     "mapping_revision": None, "estimate_source": "top_k", "corpus_sha256": None}


def _scored(run, identities):
    """Give every arm of item i a persisted scoring snapshot that carries ``identities[i]``."""
    def scoring(identity):
        return {"identity": identity, "holdout_auc": 0.7, "is_weak": False,
                "valid_counts": {"is_holdout": 4, "oos_holdout": 4}}
    return replace(run, arms=tuple(
        replace(arm, outcome=arm.outcome | {"provenance": {"scoring": scoring(identities[int(arm.item_id)])}})
        for arm in run.arms))


def test_one_calibrator_saved_in_the_legacy_and_asdict_shapes_is_listed_once():
    # The two shapes are what the identity class itself produces for one calibrator.
    identity = CalibrationIdentity.from_dict(LEGACY_IDENTITY)
    assert identity.to_dict() == LEGACY_IDENTITY
    assert json.loads(json.dumps(asdict(identity))) == ASDICT_IDENTITY and len(ASDICT_IDENTITY) == 10
    run = _scored(fixture([["up"] * 4, ["up"] * 4]), [LEGACY_IDENTITY, ASDICT_IDENTITY])
    for name in CONDITIONS:
        scoring = summarize_experiment(run)["pooled"]["conditions"][name]["scoring"]
        assert [quality["identity"] for quality in scoring["calibration_quality"]] == [LEGACY_IDENTITY], name
        assert scoring["items_with_calibration_quality"] == 2


@pytest.mark.parametrize("change", [{"model": "other-target"}, {"corpus_sha256": "a" * 64}],
                         ids=["legacy-field", "extension-field"])
def test_different_calibrators_are_still_listed_separately(change):
    run = _scored(fixture([["up"] * 4, ["up"] * 4]), [LEGACY_IDENTITY, ASDICT_IDENTITY | change])
    scoring = summarize_experiment(run)["pooled"]["conditions"]["original_single"]["scoring"]
    assert len(scoring["calibration_quality"]) == 2
    assert scoring["items_with_calibration_quality"] == 2


def test_fidelity_reports_unchanged_text_independently_of_acceptance():
    run = fixture([["up"] * 4, ["down"] * 4])
    rewrites = tuple(replace(record, rewritten_evidence=record.original_evidence) for record in run.checkpoint.rewrites)
    metrics = summarize_experiment(replace(run, checkpoint=replace(run.checkpoint, rewrites=rewrites)))["pooled"]
    assert metrics["fidelity"]["accepted"] == 2
    assert metrics["fidelity"]["unchanged_evidence"]["numerator"] == 2
    assert metrics["fidelity"]["unchanged_evidence"]["estimate"] == 1
    assert metrics["conditions"]["original_ensemble"]["agreement"]["mean"] == 0.75


def test_physical_usage_preserves_failed_calls_partial_counters_roles_and_unknown_cost():
    run = fixture([["up"] * 4])
    events = (
        RequestEvent(1, "target", "target", 1, 0, "2026-01-01T00:00:00+00:00", "failed", 2,
                     {"prompt_tokens": 10, "total_tokens": 10}, 0.1),
        RequestEvent(2, "target", "target", 2, 2, "2026-01-01T00:00:02+00:00", "succeeded", 3,
                     {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}),
        RequestEvent(3, "rewrite", "rewriter", 1, 5, "2026-01-01T00:00:05+00:00", "succeeded", 1),
    )
    requests = tuple(RequestRecord("exec", event, "rewriter" if index == 2 else "target", text_hash("prompt"),
                                   "0" if index < 2 else None,
                                   EvidenceArm.ORIGINAL if index < 2 else None, 0 if index < 2 else None)
                     for index, event in enumerate(events))
    metrics = summarize_experiment(replace(run, checkpoint=replace(run.checkpoint, requests=requests)))
    usage = metrics["pooled"]["usage"]
    assert usage["physical_requests"] == 3
    assert usage["cost"]["total"] is None and usage["cost"]["known_subtotal"] == 0.1
    assert usage["cost"]["recorded_requests"] == 1
    target = next(group for group in usage["groups"] if group["role"] == "target")
    assert target["counters"]["total_tokens"]["known_subtotal"] == 25
    assert target["counters"]["prompt_tokens"]["known_subtotal"] == 20
    assert target["counters"]["completion_tokens"]["total"] is None
    assert target["counters"]["completion_tokens"]["recorded_requests"] == 1
    assert target["latency_s"]["sum"] == 5
    assert "combined_tokens" not in target
    # Condition views share physical events; their costs must not be added.
    single = metrics["pooled"]["conditions"]["original_single"]["usage"]
    ensemble = metrics["pooled"]["conditions"]["original_ensemble"]["usage"]
    assert single["physical_requests"] == ensemble["physical_requests"] == 2


def test_empty_ledger_is_unknown_not_zero_usage_or_cost():
    metrics = summarize_experiment(fixture([["up"] * 4]))["pooled"]["usage"]
    assert metrics["cost"]["total"] is None
    assert metrics["cost"]["known_subtotal"] is None
    assert metrics["groups"] == []


def test_unique_failure_diagnostics_fill_unknown_usage_without_counting_it_twice():
    run = fixture([["up"] * 4], failed_zero=(("0", "original"),))
    first = run.checkpoint.draws[0]
    draws = (replace(first, metadata=dict(first.metadata) | {"diagnostics": {"usage": {"total_tokens": 7}}}),
             *run.checkpoint.draws[1:])
    event = RequestEvent(1, "target", "target", 1, 0, "2026-01-01T00:00:00+00:00", "failed", 1)
    request = RequestRecord("exec", event, "target", text_hash("prompt"), "0", EvidenceArm.ORIGINAL, 0)
    run = replace(run, checkpoint=replace(run.checkpoint, draws=draws, requests=(request,)))
    group = summarize_experiment(run)["pooled"]["usage"]["groups"][0]
    assert group["counters"]["total_tokens"]["total"] == 7
    assert group["usage_sources"] == {"unique_failure_diagnostics": 1}
    ledger = replace(request, event=replace(event, usage={"total_tokens": 3}))
    run = replace(run, checkpoint=replace(run.checkpoint, requests=(ledger,)))
    group = summarize_experiment(run)["pooled"]["usage"]["groups"][0]
    assert group["counters"]["total_tokens"]["total"] == 3
    assert group["usage_sources"] == {"physical_event": 1}


def test_retry_response_usage_is_not_guessed_onto_multiple_physical_calls():
    run = fixture([["up"] * 4])
    first = run.checkpoint.draws[0]
    draws = (replace(first, response=ModelResponse("raw", usage={"total_tokens": 123})),
             *run.checkpoint.draws[1:])
    events = tuple(RequestEvent(i + 1, "target", "target", i + 1, i,
                                f"2026-01-01T00:00:0{i}+00:00", "failed" if i == 0 else "succeeded", 1)
                   for i in range(2))
    requests = tuple(RequestRecord("exec", event, "target", text_hash("prompt"), "0", EvidenceArm.ORIGINAL, 0)
                     for event in events)
    run = replace(run, checkpoint=replace(run.checkpoint, draws=draws, requests=requests))
    group = summarize_experiment(run)["pooled"]["usage"]["groups"][0]
    assert group["physical_requests"] == 2 and group["counters"] == {}
    assert group["usage_sources"] == {"unavailable": 2}


def test_pending_physical_activity_is_started_without_becoming_an_answer_failure():
    run = fixture([["up"] * 4], pending=(("0", "original"), ("0", "rewritten")))
    event = RequestEvent(1, "target", "target", 1, 0, "2026-01-01T00:00:00+00:00")
    request = RequestRecord("exec", event, "target", text_hash("prompt"), "0", EvidenceArm.ORIGINAL, 0)
    run = replace(run, checkpoint=replace(run.checkpoint, requests=(request,)))
    condition = summarize_experiment(run)["pooled"]["conditions"]["original_single"]
    assert condition["coverage"]["started"] == 1 and condition["coverage"]["terminal"] == 0
    assert condition["coverage"]["physical_target_attempted"] == 1
    assert condition["coverage"]["target_attempted"] == 0
    assert condition["accuracy"]["all_attempted"]["estimate"] is None
    assert condition["coverage"]["failed"] == 0


@pytest.mark.parametrize("level", [0, 1, float("nan"), True])
def test_analysis_configuration_validation(level):
    with pytest.raises(ValueError, match="confidence_level"):
        summarize_experiment(fixture([["up"] * 4]), confidence_level=level)


def test_unknown_task_metric_is_not_silently_treated_as_accuracy():
    run = fixture([["up"] * 4])
    run = replace(run, checkpoint=replace(run.checkpoint, protocol=replace(run.checkpoint.protocol,
                                                                         task_metric="invented")))
    with pytest.raises(ValueError, match="task_metric"):
        summarize_experiment(run)
