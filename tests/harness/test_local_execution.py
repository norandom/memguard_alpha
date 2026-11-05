"""Offline integration: real provider parsing, calibration and ensemble scoring."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from recall_guard.core.completion import ModelConfig
from recall_guard.core.local_evidence import LocalModelIdentity, LocalModelProvenance
from recall_guard.harness.local_evaluation import (
    LocalAttemptBudget,
    LocalMeasurementPlan,
    MeasurementItem,
    MeasurementLimits,
)
from recall_guard.harness.local_execution import execute_plan


@pytest.fixture
def execution(tmp_path, monkeypatch):
    config = ModelConfig('openai_chat', 'http://localhost:11434/v1/chat/completions', 'test',
        parameters={'max_tokens': 32, 'temperature': 0, 'logprobs': True, 'top_logprobs': 3},
        timeout_s=2, max_retries=0, local_identity=LocalModelIdentity())
    items = tuple(MeasurementItem(f'{split}-{i}', split, f'{split}-{i}', 20)
        for split in ('is_memorized', 'oos_control') for i in range(8)) + (
        MeasurementItem('held', 'held_out', 'held', 20),)
    plan = LocalMeasurementPlan(config, 'none', 'NPU', 'a'*64, 'b'*64, '{prompt}',
        ('mia', 'mcs'), 3, MeasurementLimits(29, 2048, 100, 32, 3828, 120), items,
        min_valid=4)
    budget = LocalAttemptBudget(plan, tmp_path / 'journal.json')
    state = SimpleNamespace(calls=0, fail_at=None, drift=False, close_fail=False,
                            observations=0, unparsed_at=28, cancel_at=None, loaded=('test',), context=2048)
    def post(client, payload):
        state.calls += 1
        if state.calls == state.fail_at:
            raise TimeoutError('PRIVATE timeout')
        if state.calls == state.cancel_at:
            budget.cancellation.set()
        prompt = payload['messages'][-1]['content']
        base = -0.2 if prompt.startswith('is_memorized') else -5 if prompt.startswith('oos_control') else -1
        base += (sum(map(ord, prompt)) % 10) / 1000
        content = 'unparsed' if state.calls == state.unparsed_at else 'Direction: 1\nConfidence: 0.8'
        entries = [{'token': char, 'logprob': base + i / 1000,
                    'top_logprobs': [{'token': 'a', 'logprob': base},
                                    {'token': 'b', 'logprob': base - 1}]} for i, char in enumerate(content)]
        body = {'choices': [{'message': {'content': content}, 'logprobs': {'content': entries},
                             'finish_reason': 'stop'}]}
        return SimpleNamespace(status_code=200, headers={}, json=lambda: body)
    monkeypatch.setattr('recall_guard.core.providers.OpenAICompatibleClient._post', post)
    def observer():
        from recall_guard.harness.local_execution import HostObservation
        state.observations += 1
        if state.close_fail and state.observations >= 3:
            raise OSError('PRIVATE metadata')
        return HostObservation(LocalModelProvenance(runtime='ollama', runtime_version='1',
            artifact_digest='changed' if state.drift and state.observations >= 3 else 'digest',
            context_length=state.context, backend=None), state.loaded)
    return budget, state, observer


def test_execution_counts_probe_freezes_arms_and_retains_unparsed(execution):
    budget, state, observer = execution
    result = execute_plan(budget, observer)
    assert state.calls == budget.plan.logical_requests == 29
    assert result.status == 'completed'
    assert result.calibrated_claims_allowed
    assert result.observed_execution_mode == 'unknown'
    assert not result.npu_verified
    rows = [row for row in result.rows if row.item_id == 'held']
    assert [(row.arm, row.draw_index) for row in rows] == [('single', 0), ('ensemble', 0), ('ensemble', 1), ('ensemble', 2)]
    assert [row.status for row in rows] == ['scored', 'scored', 'unparsed', 'scored']
    assert len(result.ensembles) == 1
    assert result.ensembles['held'].n_parsed == 2
    assert result.batch_check.atomic is False


@pytest.mark.parametrize('failure', ['drift', 'close_fail'])
def test_closing_failure_invalidates_every_score(execution, failure):
    budget, state, observer = execution
    setattr(state, failure, True)
    result = execute_plan(budget, observer)
    assert not result.calibrated_claims_allowed
    assert result.status == 'condition_mismatched'
    assert not result.ensembles
    assert all(row.score is None for row in result.rows)


@pytest.mark.parametrize('failure', ['fail_at', 'cancel_at'])
def test_uncertain_or_cancelled_call_stops_physical_work(execution, failure):
    budget, state, observer = execution
    setattr(state, failure, 27)
    result = execute_plan(budget, observer)
    assert state.calls == 27
    assert not result.calibrated_claims_allowed
    rows = [r for r in result.rows if r.item_id == 'held']
    assert len(rows) == 4
    assert rows[-1].status == 'not_attempted'


@pytest.mark.parametrize(('loaded', 'context'), [(('test', 'other'), 2048), (('test',), 4096)])
def test_preflight_refuses_other_models_and_wrong_context(execution, loaded, context):
    budget, state, observer = execution
    state.loaded, state.context = loaded, context
    result = execute_plan(budget, observer)
    assert state.calls == 0
    assert not result.calibrated_claims_allowed


def test_unknown_open_context_must_be_verified_after_probe(execution):
    budget, state, observer = execution
    state.context = None
    result = execute_plan(budget, observer)
    assert state.calls == 1
    assert not result.calibrated_claims_allowed


def test_plan_commits_fit_settings(execution):
    budget, _, _ = execution
    before = budget.plan.safe_export()
    after = replace(budget.plan, seed=7, min_auc=0.7, ensemble_min_parsed=2).safe_export()
    assert before != after
    assert after['seed'] == 7


def test_invalid_evidence_after_successful_probe_is_not_skipped(execution, monkeypatch):
    budget, state, observer = execution
    from recall_guard.core.providers import OpenAICompatibleClient
    post = OpenAICompatibleClient._post
    def invalid(client, payload):
        response = post(client, payload)
        if state.calls == 27:
            response.json()['choices'][0]['logprobs']['content'][0]['token'] = 'WRONG'
        return response
    monkeypatch.setattr(OpenAICompatibleClient, '_post', invalid)
    result = execute_plan(budget, observer)
    assert state.calls == 27
    assert not result.calibrated_claims_allowed
    assert result.status == 'ineligible'
    assert all(row.score is None for row in result.rows)


def test_unknown_context_is_verified_after_probe_before_corpus(execution):
    budget, state, observer = execution
    def fresh():
        state.context = None if state.observations == 0 else 2048
        return observer()
    result = execute_plan(budget, fresh)
    assert result.status == 'completed'
    assert state.calls == 29


def test_same_journal_cannot_execute_again(execution):
    budget, state, observer = execution
    execute_plan(budget, observer)
    with pytest.raises(ValueError, match='already_started'):
        execute_plan(budget, observer)
    assert state.calls == 29


def test_native_semantics_survive_probe_sidecar_refresh(execution, monkeypatch):
    budget, state, observer = execution
    from recall_guard.core.local_evidence import ScoreSemantics
    from recall_guard.core.providers import OpenAICompatibleClient
    semantics = ScoreSemantics('full_vocabulary_log_softmax', 'e', 'base_model')
    post = OpenAICompatibleClient._post
    def native(client, payload):
        response = post(client, payload)
        response.json()['recall_guard_local'] = LocalModelIdentity(
            observed=LocalModelProvenance(score_semantics=semantics)).to_dict()
        return response
    monkeypatch.setattr(OpenAICompatibleClient, '_post', native)
    def fresh():
        observation = observer()
        if state.observations >= 3:
            observation = replace(observation, provenance=replace(observation.provenance, score_semantics=semantics))
        return observation
    result = execute_plan(budget, fresh)
    assert result.status == 'completed'
    probe = next(row for row in result.rows if row.arm == 'probe')
    assert probe.response.local_identity.observed.score_semantics == semantics


def test_cli_execute_requires_explicit_journal_and_observer():
    from scripts.measure_local_models import main
    with pytest.raises(SystemExit) as error:
        main(['--plan', 'p', '--manifest', 'm', '--corpus', 'c', '--execute'])
    assert error.value.code == 2


@pytest.mark.parametrize('defect', ['misaligned', 'missing_alternatives', 'empty'])
def test_actual_probe_requires_complete_aligned_evidence(execution, monkeypatch, defect):
    budget, state, observer = execution
    from recall_guard.core.providers import OpenAICompatibleClient
    post = OpenAICompatibleClient._post
    def invalid(client, payload):
        response = post(client, payload)
        choice = response.json()['choices'][0]
        if defect == 'misaligned':
            choice['logprobs']['content'][0]['token'] = 'WRONG'
        elif defect == 'missing_alternatives':
            choice['logprobs']['content'][0]['top_logprobs'] = []
        else:
            choice['message']['content'] = ''
            choice['logprobs']['content'] = []
        return response
    monkeypatch.setattr(OpenAICompatibleClient, '_post', invalid)
    result = execute_plan(budget, observer)
    assert state.calls == 1
    assert not result.calibrated_claims_allowed


def test_cli_execution_uses_fresh_observer_and_persistent_journal(execution, tmp_path, monkeypatch, capsys):
    from scripts import measure_local_models as cli
    budget, state, observer = execution
    monkeypatch.setattr(cli, 'load_plan', lambda *args: budget.plan)
    monkeypatch.setattr(cli, 'ollama_observer', lambda plan: observer)
    journal = tmp_path / 'cli.json'
    assert cli.main(['--plan', 'p', '--manifest', 'm', '--corpus', 'c', '--execute',
                     '--journal', str(journal), '--observer', 'ollama']) == 0
    import json
    output = json.loads(capsys.readouterr().out)
    assert output['attempts'] == 29
    assert output['observed_execution_mode'] == 'unknown'
    assert json.loads(journal.read_text())['status'] == 'completed'
    assert 'PRIVATE' not in journal.read_text()


def test_calibration_unknown_outcome_stops_corpus_calls(execution):
    budget, state, observer = execution
    state.fail_at = 3
    result = execute_plan(budget, observer)
    assert state.calls == 3
    assert not result.calibrated_claims_allowed
    assert all(r.status == 'not_attempted' for r in result.rows if r.item_id == 'held')


def test_failed_single_transport_is_not_an_unparsed_completion(execution, monkeypatch):
    budget, state, observer = execution
    from recall_guard.core.providers import OpenAICompatibleClient
    from recall_guard.core.request_control import RequestError
    post = OpenAICompatibleClient._post
    def rejected(client, payload):
        if state.calls == 25:
            state.calls += 1
            raise RequestError('PRIVATE server response', status_code=503, retryable=True)
        return post(client, payload)
    monkeypatch.setattr(OpenAICompatibleClient, '_post', rejected)
    result = execute_plan(budget, observer)
    single = next(row for row in result.rows if row.arm == 'single')
    assert single.status == 'failed'
    assert single.response is None
    assert single.score.p_memorized is None
    assert state.calls == 29


def test_post_probe_digest_loss_cannot_be_repaired_by_closing_recovery(execution):
    budget, state, observer = execution
    def fresh():
        observation = observer()
        if state.observations == 2:
            observation = replace(observation, provenance=replace(observation.provenance, artifact_digest=None))
        return observation
    result = execute_plan(budget, fresh)
    assert state.calls == 1
    assert not result.calibrated_claims_allowed
    assert result.reason == 'probe_condition_mismatched'
    assert all(row.status == 'not_attempted' for row in result.rows if row.item_id == 'held')
