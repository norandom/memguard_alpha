"""Offline paired execution, exact request association and durable restart safety."""

import json
from dataclasses import replace
from threading import Lock

import pytest

from recall_guard.core.completion import ModelConfig
from recall_guard.core.consensus import Tail
from recall_guard.core.ensemble import LocationMode, MultimodalAction, ReferenceMode
from recall_guard.core.provider_callback import CallbackClient
from recall_guard.core.request_control import CallBudget, RequestError
from recall_guard.harness.intervention import ExperimentRunner
from recall_guard.harness.intervention_types import (
    EvidenceArm,
    ExperimentItem,
    ExperimentProtocol,
    RecordStatus,
    text_hash,
)
from recall_guard.harness.rewriting import REWRITE_METHOD_VERSION, EvidenceRewriter, RewritePolicy
from recall_guard.harness.workflow import RecallWorkflow
from tests.harness.test_workflow_scoring import SETTINGS, Client, fit

SOURCE = "Acme rose 2% on 2024-01-02; uncertainty remains."
PROBE = "Reply with the word OK."
CHECKS = ("claims", "entities", "dates", "quantities", "units", "uncertainty", "qualifications",
          "attribution", "instruction_injection")
VERDICT = json.dumps({"equivalent": True, "checks": dict.fromkeys(CHECKS, True), "discrepancies": []})


def items(count=2):
    return tuple(ExperimentItem(f"item-{index}", "Return up or down.", SOURCE, "2024-01-02",
                                "PRIVATE_LABEL", "PRIVATE_COHORT") for index in range(count))


def protocol(**changes):
    configs = {role: ModelConfig("callback", None, role, parameters={"max_tokens": 64}, max_retries=0)
               for role in ("target", "rewriter", "verifier")}
    return ExperimentProtocol(**({
        **{role + "_model": config for role, config in configs.items()},
        "data_hashes": {"evaluation": "a" * 64}, "cohort_definitions": {"PRIVATE_COHORT": {}},
        "draws": 3, "bootstrap_seed": 1, "bootstrap_count": 100, "task_metric": "accuracy",
        "utility_tolerance": 0.01, "audit_selection": {"method": "all"}, "request_budget": 100,
        "rewrite_method_version": REWRITE_METHOD_VERSION,
        "ensemble_settings": {"max_workers": 1, "min_parsed": 1, "retain_draws": True,
                              "max_transport_failure_ratio": 1},
    } | changes))


class Factory:
    def __init__(self, p, *, candidate=None, target=None, multiple=False):
        self.protocol, self.candidate, self.target = p, candidate, target
        self.calls, self.budgets, self.workflows = [], [], []
        self.multiple = multiple
        self.lock = Lock()

    def __call__(self, budget):
        self.budgets.append(budget)

        def callback(role):
            def invoke(prompt, *, physical_request, **settings):
                with self.lock:
                    self.calls.append((role, prompt, settings))
                if self.multiple:
                    physical_request(lambda: "intermediate")

                def operation():
                    if prompt == PROBE:
                        if role == "target" and self.protocol.target_model.parameters.get("logprobs") and self.target:
                            return self.target(prompt)
                        return "OK"
                    if role == "target":
                        return self.target(prompt) if self.target else "up"
                    if role == "rewriter":
                        return self.candidate if self.candidate is not None else "Let's review: " + SOURCE
                    return VERDICT
                return physical_request(operation)
            return invoke

        clients = {role: CallbackClient(getattr(self.protocol, role + "_model"), budget, callback(role),
                                        physical_request_hook="physical_request", strict_budget=True)
                   for role in ("target", "rewriter", "verifier")}
        policy = RewritePolicy(EvidenceRewriter(clients["rewriter"]), clients["verifier"])
        workflow = RecallWorkflow(clients["target"], rewrite_policy=policy, budget=budget, strict_budget=True)
        self.workflows.append(workflow)
        return workflow


def runner(tmp_path, p=None, *, values=None, **factory_options):
    p = p or protocol()
    factory = Factory(p, **factory_options)
    return ExperimentRunner(p, values or items(), factory, tmp_path / "checkpoint.json"), factory


def test_serialized_two_sided_tail_uses_the_actual_two_sided_agreement_interval(tmp_path):
    p = protocol(ensemble_settings=dict(protocol().ensemble_settings) | {"tail": "two_sided"})
    restored = ExperimentProtocol.from_dict(json.loads(json.dumps(p.to_dict())))
    execution, factory = runner(tmp_path, restored, values=items(1))
    result = execution.run()
    assert all(condition.status == RecordStatus.COMPLETED for condition in result.conditions)
    for arm in result.arms:
        assert arm.outcome["ensemble"]["agreement_ci"] == pytest.approx([0.4385029682449546, 1.0])
    assert len(factory.calls) == 11


@pytest.mark.parametrize("reference_mode", list(ReferenceMode))
def test_typed_and_serialized_settings_produce_equal_component_results_and_resume(tmp_path, reference_mode):
    settings = dict(protocol().ensemble_settings) | {
        "tail": Tail.TWO_SIDED, "location_mode": LocationMode.MEAN,
        "multimodal_action": MultimodalAction.FLAG, "reference_mode": reference_mode,
    }
    typed = protocol(ensemble_settings=settings)
    serialized = json.loads(json.dumps(typed.to_dict(), ensure_ascii=False, allow_nan=False))
    restored = ExperimentProtocol.from_dict(serialized)
    results = []
    for name, p in (("typed", typed), ("serialized", restored)):
        responses = iter(["0.0", "0.0", "0.9"] * 2)
        execution, factory = runner(tmp_path / name, p, values=items(1),
                                    target=lambda prompt, responses=responses: next(responses))
        partial = execution.run(max_arms=1, decide=lambda response: "up",
                                components=lambda response: {"axis": float(response.content)})
        assert partial.checkpoint.status == RecordStatus.PENDING
        assert execution.load().checkpoint.protocol.to_dict() == serialized
        result = execution.run(decide=lambda response: "up",
                               components=lambda response: {"axis": float(response.content)})
        assert all(condition.status == RecordStatus.COMPLETED for condition in result.conditions)
        assert len([call for call in factory.calls if call[0] == "target" and call[1] != PROBE]) == 6
        results.append(result)
        for arm in result.arms:
            assert arm.outcome["ensemble"]["location"] == {"axis": pytest.approx(0.3)}
            assert arm.outcome["ensemble"]["agreement_ci"] == pytest.approx([0.4385029682449546, 1.0])
        assert execution.load().checkpoint.to_dict() == result.checkpoint.to_dict()
    assert typed.protocol_id == restored.protocol_id
    assert results[0].conditions == results[1].conditions
    for typed_arm, restored_arm in zip(results[0].arms, results[1].arms, strict=True):
        for key in ("location", "agreement_ci", "draws_sha256", "n_requested", "n_parsed"):
            assert typed_arm.outcome["ensemble"][key] == restored_arm.outcome["ensemble"][key]


@pytest.mark.parametrize("name, value", [
    ("tail", "invalid"), ("location_mode", "invalid"), ("multimodal_action", "invalid"),
    ("reference_mode", "invalid"), ("tail", None), ("location_mode", 1),
    ("multimodal_action", True), ("reference_mode", []),
    ("draws", 3.0), ("draws", True), ("draws", 2),
    ("max_workers", 1.5), ("max_workers", True), ("min_parsed", 1.0), ("min_parsed", None),
    ("trough_steps", 1.5), ("min_cluster_draws", 8.0), ("max_total_requests", 20.0), ("max_tokens", 64.0),
    ("retain_draws", 1), ("retain_draws", "false"), ("retain_draws", None),
    ("confidence", True), ("confidence", None), ("grid", "0.1"), ("trim", False),
    ("agreement_target", "0.2"), ("mass_min", True), ("density_ratio", "10"),
    ("min_cluster_density", None), ("max_transport_failure_ratio", True), ("temperature", "0.5"),
    ("grid", float("inf")), ("density_ratio", float("nan")), ("min_cluster_density", float("inf")),
    ("max_tokens", 65), ("temperature", 0.5), ("unknown_option", 1),
])
def test_invalid_ensemble_settings_fail_before_workflow_factory_or_provider_calls(tmp_path, name, value):
    factory = Factory(protocol())
    with pytest.raises((TypeError, ValueError)):
        p = protocol(ensemble_settings=dict(protocol().ensemble_settings) | {name: value})
        ExperimentRunner(p, items(1), factory, tmp_path / "checkpoint.json").run()
    assert not factory.budgets and not factory.calls
    assert not (tmp_path / "checkpoint.json").exists()


def test_saved_four_conditions_reuse_predesignated_draw_zero_and_balance_order(tmp_path):
    execution, factory = runner(tmp_path)
    result = execution.run()
    assert result.checkpoint.status == RecordStatus.COMPLETED
    assert len(result.checkpoint.draws) == 12
    assert len(result.conditions) == 8
    assert [(arm.item_id, arm.arm) for arm in result.arms] == [
        ("item-0", EvidenceArm.ORIGINAL), ("item-0", EvidenceArm.REWRITTEN),
        ("item-1", EvidenceArm.REWRITTEN), ("item-1", EvidenceArm.ORIGINAL)]
    for condition in result.conditions:
        assert condition.draw_indices == ((0,) if condition.method.endswith("single") else (0, 1, 2))
        assert condition.status == RecordStatus.COMPLETED
    assert len([call for call in factory.calls if call[0] == "target" and call[1] != PROBE]) == 12
    assert len([call for call in factory.calls if call[0] == "rewriter" and call[1] != PROBE]) == 2
    assert all("PRIVATE_LABEL" not in prompt and "PRIVATE_COHORT" not in prompt for _, prompt, _ in factory.calls)
    saved = execution.load()
    assert saved.checkpoint.to_dict() == result.checkpoint.to_dict()
    assert saved.conditions == result.conditions
    assert saved.checkpoint.validate_restart(protocol(), items()).remaining_requests == 81
    for draw in saved.checkpoint.draws:
        assert draw.metadata["attempted"] is True
        assert draw.metadata["evidence_hash"] == text_hash(
            SOURCE if draw.arm == EvidenceArm.ORIGINAL else "Let's review: " + SOURCE)
        assert draw.metadata["guarded_score"] is None


def test_failed_draw_zero_is_never_replaced_by_successful_draw(tmp_path):
    counter = 0

    def target(prompt):
        nonlocal counter
        counter += 1
        return "invalid" if counter == 1 else "up"

    execution, _ = runner(tmp_path, values=items(1), target=target)

    def decide(response):
        if response.content == "invalid":
            raise ValueError("private parse input")
        return response.content

    result = execution.run(decide=decide)
    single = next(c for c in result.conditions if c.method == "original_single")
    ensemble = next(c for c in result.conditions if c.method == "original_ensemble")
    assert single.status == RecordStatus.FAILED and single.answer is None
    assert ensemble.status == RecordStatus.COMPLETED and ensemble.answer == "up"
    assert result.checkpoint.draws[0].response.content == "invalid"


def test_rejected_rewrite_keeps_missing_draws_and_completes_failed_conditions(tmp_path):
    execution, factory = runner(tmp_path, candidate="Acme rose 3% on 2024-01-02.")
    result = execution.run()
    assert result.checkpoint.status == RecordStatus.COMPLETED
    rewritten = [draw for draw in result.checkpoint.draws if draw.arm == EvidenceArm.REWRITTEN]
    assert len(rewritten) == 6 and all(draw.status == RecordStatus.FAILED for draw in rewritten)
    assert all(draw.metadata["attempted"] is False and draw.response is None for draw in rewritten)
    assert all(draw.metadata["failure_kind"] == "fidelity_rejected" for draw in rewritten)
    assert not [r for r in result.checkpoint.requests if r.arm == EvidenceArm.REWRITTEN]
    assert len([call for call in factory.calls if call[0] == "target" and call[1] != PROBE]) == 6


@pytest.mark.parametrize("error", [RequestError("auth", status_code=401), RequestError("credits", status_code=402),
                                   RequestError("parameter", status_code=400, parameter="max_tokens")])
def test_fatal_failure_preserves_partial_state_and_stops_future_requests(tmp_path, error):
    def target(prompt):
        raise error

    execution, factory = runner(tmp_path, target=target)
    result = execution.run()
    assert result.checkpoint.status == RecordStatus.FAILED and result.stop_reason
    assert len(result.checkpoint.requests) == 4
    assert len([call for call in factory.calls if call[0] == "target" and call[1] != PROBE]) == 1
    assert any(draw.status == RecordStatus.PENDING for draw in result.checkpoint.draws)
    before = len(factory.calls)
    assert execution.run().stop_reason == result.stop_reason
    assert len(factory.calls) == before


def test_budget_exhaustion_preserves_unattempted_identity_without_exceeding_cap(tmp_path):
    execution, factory = runner(tmp_path, protocol(request_budget=5))
    result = execution.run()
    assert result.checkpoint.status == RecordStatus.FAILED
    assert result.stop_reason == "budget_exhausted"
    assert len(result.checkpoint.requests) == factory.budgets[0].used == 5
    assert sum(draw.metadata["attempted"] is True for draw in result.checkpoint.draws) == 3
    assert len([r for r in result.checkpoint.requests if r.item_id]) == 2
    assert [draw.metadata["physical_attempts"] for draw in result.checkpoint.draws[:3]] == [1, 1, 0]


def test_safe_resume_skips_terminal_arms_and_uses_remaining_budget_namespace(tmp_path):
    execution, factory = runner(tmp_path)
    partial = execution.run(max_arms=1)
    assert partial.checkpoint.status == RecordStatus.PENDING and len(partial.arms) == 1
    used = len(partial.checkpoint.requests)
    result = execution.run()
    assert result.checkpoint.status == RecordStatus.COMPLETED
    assert factory.budgets[1].limit == protocol().request_budget - used
    original = [r for r in result.checkpoint.requests if r.item_id == "item-0" and r.arm == EvidenceArm.ORIGINAL]
    assert len(original) == 3
    assert len({r.execution_id for r in result.checkpoint.requests}) == 2
    assert result.checkpoint.validate_restart(protocol(), items()).remaining_requests == 78
    calls_before = len(factory.calls)
    assert execution.run().checkpoint.status == RecordStatus.COMPLETED
    assert len(factory.calls) == calls_before


def test_leftover_write_ahead_intent_refuses_resume_before_factory(tmp_path, monkeypatch):
    execution, factory = runner(tmp_path)
    original_run = RecallWorkflow.run

    def crash(workflow, *args, **kwargs):
        original_run(workflow, *args, **kwargs)
        raise KeyboardInterrupt

    monkeypatch.setattr(RecallWorkflow, "run", crash)
    with pytest.raises(KeyboardInterrupt):
        execution.run()
    saved = json.loads((tmp_path / "checkpoint.json").read_text(encoding="utf-8"))
    assert saved["intent"]["kind"] == "arm"
    assert len(saved["checkpoint"]["requests"]) == 3  # latest checkpoint predates the target calls
    before = len(factory.budgets)
    with pytest.raises(ValueError, match="intent|Ambiguous"):
        execution.run()
    assert len(factory.budgets) == before


def test_incompatible_inputs_and_protocol_rejected_before_factory(tmp_path):
    execution, factory = runner(tmp_path)
    execution.run(max_arms=1)
    for p, values in [(replace(protocol(), draws=4), items()), (protocol(), tuple(reversed(items())))]:
        incompatible = ExperimentRunner(p, values, factory, tmp_path / "checkpoint.json")
        with pytest.raises(ValueError, match="protocol|inputs"):
            incompatible.run()
    assert len(factory.budgets) == 1


def test_multihook_requests_and_concurrent_draws_keep_exact_stage_identity(tmp_path):
    p = protocol(ensemble_settings={"min_parsed": 1, "max_workers": 3, "retain_draws": True})
    execution, _ = runner(tmp_path, p, multiple=True, values=items(1))
    result = execution.run()
    requests = result.checkpoint.requests
    assert len(requests) == 22
    for arm in EvidenceArm:
        for index in range(3):
            physical = [r for r in requests if r.arm == arm and r.draw_index == index]
            assert len(physical) == 2
            assert physical[0].event.stage == physical[1].event.stage
            assert physical[0].event.attempt == physical[1].event.attempt == 1
    for r in requests:
        if r.item_id is None:
            assert r.prompt_hash == text_hash(PROBE)
        elif r.model_role == "target":
            draw = next(draw for draw in result.checkpoint.draws if draw.identity == (r.item_id, r.arm, r.draw_index))
            assert r.prompt_hash == draw.prompt_hash
    assert result.checkpoint.validate_restart(p, items(1)).prior_used_requests == 22


def test_runtime_writer_lock_refuses_concurrent_execution(tmp_path):
    execution, factory = runner(tmp_path)
    (tmp_path / "checkpoint.json.lock").write_text("existing writer", encoding="utf-8")
    with pytest.raises(ValueError, match="writer|lock"):
        execution.run()
    assert not factory.budgets


def test_runner_rejects_configuration_drift_and_opaque_accounting_before_calls(tmp_path):
    p = protocol()
    factory = Factory(p)

    def wrong(budget):
        workflow = factory(budget)
        workflow.target.config = replace(workflow.target.config, model="changed")
        return workflow

    with pytest.raises(ValueError, match="configuration|model"):
        ExperimentRunner(p, items(), wrong, tmp_path / "checkpoint.json").run()
    assert not factory.calls


def test_probe_failure_is_retained_without_fabricated_probe_response(tmp_path):
    p = protocol()
    factory = Factory(p)

    def inaccessible(budget):
        workflow = factory(budget)
        workflow.target._callback = lambda prompt, physical_request, **settings: physical_request(
            lambda: (_ for _ in ()).throw(RequestError("no access", status_code=401)))
        return workflow

    result = ExperimentRunner(p, items(), inaccessible, tmp_path / "checkpoint.json").run()
    assert result.checkpoint.status == RecordStatus.FAILED
    assert len(result.checkpoint.requests) == 1 and not result.arms
    request = result.checkpoint.requests[0]
    assert request.item_id is None and request.model_role == "target" and request.prompt_hash == text_hash(PROBE)
    assert result.preflight["responses"] == {}
    assert all(draw.status == RecordStatus.PENDING for draw in result.checkpoint.draws)


def test_retry_events_stay_bound_to_their_draw_with_identical_prompts(tmp_path):
    p = protocol()
    p = replace(p, target_model=replace(p.target_model, max_retries=1, retry_backoff_s=0))
    counter = 0

    def target(prompt):
        nonlocal counter
        counter += 1
        if counter == 1:
            raise TimeoutError("retry this draw")
        return "up"

    execution, _ = runner(tmp_path, p, target=target, values=items(1))
    result = execution.run()
    first = [r for r in result.checkpoint.requests if r.arm == EvidenceArm.ORIGINAL and r.draw_index == 0]
    assert [r.event.attempt for r in first] == [1, 2]
    assert [r.event.status for r in first] == ["failed", "succeeded"]
    assert len({r.event.stage for r in first}) == 1
    assert len([r for r in result.checkpoint.requests if r.item_id and r.model_role == "target"]) == 7


@pytest.mark.parametrize("kind", ["preflight", "arm"])
def test_intent_is_durable_before_the_first_physical_operation(tmp_path, kind):
    p = protocol()
    factory = Factory(p)

    def checked(budget):
        workflow = factory(budget)
        callback = workflow.target._callback

        def inspect(prompt, **settings):
            value = json.loads((tmp_path / "checkpoint.json").read_text(encoding="utf-8"))
            assert value["intent"]["kind"] == ("preflight" if prompt == PROBE else "arm")
            if value["intent"]["kind"] == kind:
                raise KeyboardInterrupt
            return callback(prompt, **settings)

        workflow.target._callback = inspect
        return workflow

    execution = ExperimentRunner(p, items(), checked, tmp_path / "checkpoint.json")
    with pytest.raises(KeyboardInterrupt):
        execution.run()
    before = len(factory.budgets)
    with pytest.raises(ValueError, match="intent"):
        execution.run()
    assert len(factory.budgets) == before


def test_crash_after_terminal_save_before_intent_clear_still_refuses_restart(tmp_path, monkeypatch):
    execution, factory = runner(tmp_path)
    save = execution.journal.save

    def crash(result, intent=None):
        if result.arms and intent is None:
            raise OSError("disk unavailable before clear")
        save(result, intent)

    monkeypatch.setattr(execution.journal, "save", crash)
    with pytest.raises(OSError):
        execution.run()
    value = json.loads((tmp_path / "checkpoint.json").read_text(encoding="utf-8"))
    assert value["intent"]["kind"] == "arm" and len(value["arms"]) == 1
    before = len(factory.budgets)
    with pytest.raises(ValueError, match="intent"):
        execution.run()
    assert len(factory.budgets) == before


def test_missing_terminal_arm_marker_never_replays_saved_target_output(tmp_path):
    execution, factory = runner(tmp_path)
    execution.run(max_arms=1)
    path = tmp_path / "checkpoint.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["arms"] = []
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="completion markers"):
        execution.run()
    assert len(factory.budgets) == 1


def test_credential_references_can_be_reattached_to_a_reloaded_protocol(tmp_path):
    p = protocol()
    live = replace(p, target_model=replace(p.target_model, credential_ref="TARGET_SECRET_REFERENCE"))
    portable = ExperimentProtocol.from_dict(p.to_dict())
    factory = Factory(live)
    result = ExperimentRunner(portable, items(1), factory, tmp_path / "checkpoint.json").run()
    assert result.checkpoint.status == RecordStatus.COMPLETED
    assert "TARGET_SECRET_REFERENCE" not in (tmp_path / "checkpoint.json").read_text(encoding="utf-8")


def scored_runner(tmp_path):
    p = protocol(target_model=ModelConfig("callback", None, "target", parameters=SETTINGS))
    target = Client()
    factory = Factory(p, target=target.generate)
    policy_id = factory(CallBudget(100)).rewrite_policy.intervention_id
    original, rewritten = fit(target), fit(target, intervention_id=policy_id)
    p = replace(p, data_hashes=dict(p.data_hashes) | {"original_calibration": original.calibration_fingerprint,
                                                    "rewritten_calibration": rewritten.calibration_fingerprint})
    factory.protocol = p
    return ExperimentRunner(p, items(1), factory, tmp_path / "checkpoint.json",
                            original_scorer=original, rewritten_scorer=rewritten), factory, original, rewritten


def test_actual_per_draw_scoring_and_fit_identity_survive_runtime_roundtrip(tmp_path):
    execution, _, original, rewritten = scored_runner(tmp_path)
    result = execution.run()
    assert result.checkpoint.status == RecordStatus.COMPLETED
    for arm in result.arms:
        fit_used = original if arm.arm == EvidenceArm.ORIGINAL else rewritten
        assert arm.outcome["calibration_identity"]["intervention_id"] == fit_used.calibration_identity.intervention_id
        assert arm.outcome["scored_ensemble"]["p_memorized_point"] is not None
    saved = execution.load()
    for draw in saved.checkpoint.draws:
        score = draw.metadata["guarded_score"]
        assert score["parse_ok"] and score["features"] is not None
        assert score["memguard_confidence"] == score["raw_confidence"] * (1 - score["p_memorized"])
        assert draw.response.token_evidence


def test_scoring_requires_both_fits_and_rejects_changed_numeric_state_before_probes(tmp_path):
    execution, factory, original, _ = scored_runner(tmp_path)
    original._mcs.classifier.coef_[0, 0] += 0.1
    with pytest.raises(ValueError, match="fingerprint"):
        execution.run()
    assert not factory.calls
    with pytest.raises(ValueError, match="both"):
        ExperimentRunner(execution.protocol, items(), factory, tmp_path / "other.json", original_scorer=original)
    with pytest.raises(ValueError, match="both"):
        ExperimentRunner(execution.protocol, items(), factory, tmp_path / "other.json")


def test_resume_rejects_replacement_fit_even_with_same_model_identity(tmp_path):
    execution, factory, _, rewritten = scored_runner(tmp_path)
    execution.run(max_arms=1)
    rewritten._baseline.feature_means["loss"] += 0.25
    before = len(factory.calls)
    with pytest.raises(ValueError, match="fingerprint"):
        execution.run()
    assert len(factory.calls) == before


@pytest.mark.parametrize("replacement_top_p", [0.1, 0.9])
def test_safe_resume_freezes_observed_fit_settings_before_any_new_request(tmp_path, replacement_top_p):
    execution, factory, original, rewritten = scored_runner(tmp_path)

    def observed_fit(top_p):
        client = Client()
        client.mutate = lambda response: replace(response, confirmed_settings={"top_p": top_p})
        return fit(client, intervention_id=rewritten.calibration_identity.intervention_id)

    frozen = observed_fit(0.1)
    p = replace(execution.protocol, data_hashes=dict(execution.protocol.data_hashes) | {
        "rewritten_calibration": frozen.calibration_fingerprint})
    factory.protocol = p
    path = tmp_path / "checkpoint.json"
    execution = ExperimentRunner(p, items(1), factory, path, original_scorer=original, rewritten_scorer=frozen)
    partial = execution.run(max_arms=1)
    assert partial.checkpoint.status == RecordStatus.PENDING
    replacement = observed_fit(replacement_top_p)
    assert replacement.calibration_identity == frozen.calibration_identity
    resumed = ExperimentRunner(p, items(1), factory, path, original_scorer=original, rewritten_scorer=replacement)
    before = len(factory.calls)
    if replacement_top_p != 0.1:
        with pytest.raises(ValueError, match="fingerprint"):
            resumed.run()
        assert len(factory.calls) == before
        assert resumed.load().checkpoint.to_dict() == partial.checkpoint.to_dict()
    else:
        assert resumed.run().checkpoint.status == RecordStatus.COMPLETED
        assert len(factory.calls) > before


def test_non_json_projection_becomes_visible_failure_without_losing_the_journal(tmp_path):
    execution, _ = runner(tmp_path, values=items(1))
    result = execution.run(decide=lambda response: frozenset({response.content}))
    assert result.checkpoint.status == RecordStatus.COMPLETED
    assert all(condition.status == RecordStatus.FAILED for condition in result.conditions)
    assert execution.load().checkpoint.status == RecordStatus.COMPLETED


def test_opaque_callback_is_rejected_before_probe_or_corpus_calls(tmp_path):
    p = protocol()
    calls = []

    def factory(budget):
        target = CallbackClient(p.target_model, budget, lambda prompt, **settings: calls.append(prompt) or "up")
        return RecallWorkflow(target, budget=budget, strict_budget=False)

    with pytest.raises(ValueError, match="strict"):
        ExperimentRunner(p, items(), factory, tmp_path / "checkpoint.json").run()
    assert not calls


def test_intervention_mismatched_fit_is_rejected_before_any_probe(tmp_path):
    execution, factory, original, _ = scored_runner(tmp_path)
    p = replace(execution.protocol, data_hashes=dict(execution.protocol.data_hashes) | {
        "rewritten_calibration": original.calibration_fingerprint})
    factory.protocol = p
    mismatch = ExperimentRunner(p, items(1), factory, tmp_path / "checkpoint.json",
                                original_scorer=original, rewritten_scorer=original)
    with pytest.raises(ValueError, match="identity|intervention"):
        mismatch.run()
    assert not factory.calls


def test_fit_mutation_during_an_arm_invalidates_its_conditions_and_stops_future_arms(tmp_path):
    execution, factory, original, _ = scored_runner(tmp_path)
    target = factory.target

    def mutate(prompt):
        response = target(prompt)
        if prompt != PROBE:
            original._mcs.classifier.intercept_[0] += 1
        return response

    factory.target = mutate
    result = execution.run()
    assert result.stop_reason == "configuration_error" and result.checkpoint.status == RecordStatus.FAILED
    assert len(result.arms) == 1
    assert result.conditions[0].status == result.conditions[1].status == RecordStatus.FAILED
    assert all(draw.response is not None for draw in result.checkpoint.draws[:3])


def test_identical_model_ids_across_roles_still_get_exact_probe_role_identity(tmp_path):
    p = protocol()
    shared = replace(p.target_model, model="same-model")
    p = replace(p, target_model=shared, rewriter_model=shared, verifier_model=shared)
    execution, _ = runner(tmp_path, p, values=items(1))
    result = execution.run()
    probes = [r for r in result.checkpoint.requests if r.item_id is None]
    assert [r.model_role for r in probes] == ["target", "rewriter", "verifier"]
    assert all(r.event.model == "same-model" and r.prompt_hash == text_hash(PROBE) for r in probes)


@pytest.mark.parametrize("components", [lambda response: {"confidence": float("nan")},
                                      lambda response: {"confidence": float("inf")},
                                      lambda response: {2: 0.5}])
def test_invalid_component_projection_retains_raw_responses_and_terminal_journal(tmp_path, components):
    execution, _ = runner(tmp_path, values=items(1))
    result = execution.run(components=components)
    assert result.checkpoint.status == RecordStatus.COMPLETED
    assert all(condition.status == RecordStatus.FAILED for condition in result.conditions)
    assert all(draw.response is not None and draw.metadata["components"] is None for draw in result.checkpoint.draws)
    assert all(any(draw.arm == arm and draw.metadata["failure_kind"] == "components"
                   for draw in result.checkpoint.draws) for arm in EvidenceArm)
    assert execution.load().checkpoint.status == RecordStatus.COMPLETED
    value = json.loads((tmp_path / "checkpoint.json").read_text(encoding="utf-8"))
    assert value["intent"] is None


def test_accidental_factory_probe_is_rejected_without_automatic_duplicate_on_resume(tmp_path):
    p = protocol()
    factory = Factory(p)

    def calls_during_construction(budget):
        workflow = factory(budget)
        workflow.target.probe()
        return workflow

    execution = ExperimentRunner(p, items(), calls_during_construction, tmp_path / "checkpoint.json")
    with pytest.raises(ValueError, match="factory"):
        execution.run()
    assert len(factory.calls) == 1
    with pytest.raises(ValueError, match="intent"):
        execution.run()
    assert len(factory.calls) == 1


def test_fatal_budget_stop_in_last_arm_stays_failed_when_every_draw_slot_is_terminal(tmp_path):
    execution, factory = runner(tmp_path, protocol(request_budget=10), values=items(1))
    result = execution.run()
    assert result.stop_reason == "budget_exhausted"
    assert result.checkpoint.status == RecordStatus.FAILED
    assert len(result.arms) == 2
    assert all(draw.status != RecordStatus.PENDING for draw in result.checkpoint.draws)
    assert result.checkpoint.draws[-1].metadata["physical_attempts"] == 0
    assert len(result.checkpoint.requests) == 10
    before = len(factory.calls)
    assert execution.run().checkpoint.status == RecordStatus.FAILED
    assert len(factory.calls) == before
