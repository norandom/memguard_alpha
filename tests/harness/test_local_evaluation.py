import json
from dataclasses import FrozenInstanceError
from hashlib import sha256

import pytest

from recall_guard.core.local_evidence import LocalModelIdentity
from recall_guard.harness import local_evaluation as local


def digest(data):
    return sha256(data).hexdigest()


@pytest.fixture
def inputs(tmp_path):
    corpus = {'is_memorized': [{'item_id': 'is', 'prompt': 'private IS', 'input_tokens': 4}],
              'oos_control': [{'item_id': 'oos', 'prompt': 'private OOS', 'input_tokens': 4}],
              'held_out': [{'item_id': 'test', 'prompt': 'private heldout', 'input_tokens': 4}]}
    config = {'local_identity': LocalModelIdentity().to_dict(), 'protocol': 'openai_chat', 'model': 'local-model',
              'endpoint': 'http://localhost:11434/v1/chat/completions',
              'parameters': {'max_tokens': 32, 'temperature': 0, 'logprobs': True, 'top_logprobs': 5},
              'max_retries': 1, 'timeout_s': 2, 'retry_backoff_s': 0}
    manifest = {'config': config, 'authentication': 'none', 'requested_execution_mode': 'CPU'}
    corpus_path, manifest_path = tmp_path / 'corpus.json', tmp_path / 'manifest.json'
    corpus_path.write_text(json.dumps(corpus), encoding='utf-8')
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
    data = {'schema_version': 1, 'manifest_sha256': digest(manifest_path.read_bytes()),
            'corpus_sha256': digest(corpus_path.read_bytes()), 'prompt_template': '{prompt}',
            'selected_features': ['mia'], 'ensemble_draws': 2,
            'limits': {'max_requests': 14, 'context_tokens': 2048, 'max_input_tokens': 512,
                       'max_output_tokens': 32, 'max_total_tokens': 7616, 'max_elapsed_s': 60}}
    plan_path = tmp_path / 'plan.json'
    plan_path.write_text(json.dumps(data), encoding='utf-8')
    return plan_path, manifest_path, corpus_path


def test_frozen_plan_counts_real_calibration_work(inputs):
    plan = local.load_plan(*inputs)
    assert plan.logical_requests == 7  # probe + IS + OOS baseline + OOS training + single + 2 draws
    assert plan.physical_requests == 14
    with pytest.raises(FrozenInstanceError):
        plan.ensemble_draws = 5
    assert 'private' not in json.dumps(plan.safe_export())



def test_changed_hash_refused(inputs):
    inputs[2].write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='hash_mismatch'):
        local.load_plan(*inputs)


@pytest.mark.parametrize(('field', 'value', 'reason'), [
    ('max_requests', 13, 'request_bound'), ('max_total_tokens', 1, 'total_token_bound'),
    ('context_tokens', 32, 'exceed_context'), ('max_elapsed_s', 1, 'timeout_exceeds'),
    ('max_output_tokens', True, 'positive'), ('max_elapsed_s', float('inf'), 'positive'),
])
def test_invalid_bounds(inputs, field, value, reason):
    data = json.loads(inputs[0].read_text())
    data['limits'][field] = value
    inputs[0].write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(ValueError, match=reason):
        local.load_plan(*inputs)


def response(plan):
    from recall_guard.core.completion import ModelResponse
    return ModelResponse(content='private output')


def test_durable_intent_precedes_probe_and_retry_and_exports_are_safe(inputs, tmp_path):
    from recall_guard.core.request_control import RequestControl, RequestError
    plan = local.load_plan(*inputs)
    path = tmp_path / 'attempts.json'
    budget = local.LocalAttemptBudget(plan, path)
    control = RequestControl(plan.config, budget)
    def operation():
        journal = json.loads(path.read_text())
        assert journal['attempts'][-1]['status'] == 'unknown_outcome'
        assert len(journal['attempts']) == budget.used
        if budget.used == 1:
            raise RequestError('private prompt SECRET', status_code=503, retryable=True,
                               detail='private detail', usage={'SECRET': 3, 'prompt_tokens': 4})
        return response(plan)
    control.run(operation, stage='private stage', purpose='probe')
    assert [event.purpose for event in budget.events] == ['probe', 'retry']
    journal = path.read_text()
    assert 'private' not in journal and 'SECRET' not in journal
    assert [entry['status'] for entry in json.loads(journal)['attempts']] == ['failed', 'succeeded']
    with pytest.raises(FileExistsError):
        local.LocalAttemptBudget(plan, path)


def test_unknown_outcome_stops_retry_and_reopen(inputs, tmp_path):
    from recall_guard.core.request_control import BudgetExhaustedError, RequestControl
    plan = local.load_plan(*inputs)
    path = tmp_path / 'attempts.json'
    budget = local.LocalAttemptBudget(plan, path)
    def timeout():
        raise TimeoutError('private timeout')
    with pytest.raises(BudgetExhaustedError, match='stopped'):
        RequestControl(plan.config, budget).run(timeout, stage='probe')
    assert budget.used == 1
    assert json.loads(path.read_text())['attempts'][0]['status'] == 'unknown_outcome'
    with pytest.raises(FileExistsError):
        local.LocalAttemptBudget(plan, path)


def test_cancellation_retains_completed_and_stops_admission(inputs, tmp_path):
    from recall_guard.core.request_control import BudgetExhaustedError, RequestControl
    plan = local.load_plan(*inputs)
    budget = local.LocalAttemptBudget(plan, tmp_path / 'attempts.json')
    control = RequestControl(plan.config, budget)
    def complete():
        budget.cancellation.set()
        return response(plan)
    control.run(complete, stage='calibration')
    with pytest.raises(BudgetExhaustedError):
        control.run(lambda: pytest.fail('physical request after cancellation'), stage='single')
    assert budget.used == 1
    assert json.loads(budget.path.read_text())['attempts'][0]['status'] == 'succeeded'


def test_elapsed_and_serial_admission(inputs, tmp_path):
    from recall_guard.core.request_control import BudgetExhaustedError
    plan = local.load_plan(*inputs)
    now = [0]
    budget = local.LocalAttemptBudget(plan, tmp_path / 'attempts.json', clock=lambda: now[0])
    event = budget._admit(stage='probe', model=plan.config.model, attempt=1, started_at=0)
    with pytest.raises(BudgetExhaustedError, match='serial'):
        budget._admit(stage='probe', model=plan.config.model, attempt=1, started_at=0)
    budget._finish(event, status='succeeded', latency_s=1)
    now[0] = 59
    with pytest.raises(BudgetExhaustedError, match='elapsed'):
        budget._admit(stage='probe', model=plan.config.model, attempt=1, started_at=59)
    assert budget.used == 1


def test_persistence_failure_never_calls_transport(inputs, tmp_path, monkeypatch):
    from recall_guard.core.request_control import RequestControl
    plan = local.load_plan(*inputs)
    budget = local.LocalAttemptBudget(plan, tmp_path / 'attempts.json')
    def fail(*args):
        raise OSError('disk full')
    monkeypatch.setattr(local, '_persist', fail)
    with pytest.raises(OSError):
        RequestControl(plan.config, budget).run(lambda: pytest.fail('unpersisted POST'), stage='probe')
    assert budget.used == 1


def test_interrupt_leaves_unknown_intent(inputs, tmp_path):
    from recall_guard.core.request_control import RequestControl
    plan = local.load_plan(*inputs)
    budget = local.LocalAttemptBudget(plan, tmp_path / 'attempts.json')
    def interrupt():
        raise KeyboardInterrupt
    with pytest.raises(KeyboardInterrupt):
        RequestControl(plan.config, budget).run(interrupt, stage='probe')
    assert json.loads(budget.path.read_text())['attempts'][0]['status'] == 'unknown_outcome'


def test_physical_limit_includes_all_stages(inputs, tmp_path):
    from recall_guard.core.request_control import BudgetExhaustedError, RequestControl
    plan = local.load_plan(*inputs)
    budget = local.LocalAttemptBudget(plan, tmp_path / 'attempts.json')
    control = RequestControl(plan.config, budget)
    for _ in range(plan.limits.max_requests):
        control.run(lambda: response(plan), stage='calibration')
    with pytest.raises(BudgetExhaustedError):
        control.run(lambda: pytest.fail('budget overrun'), stage='ensemble')
    assert budget.used == 14



def test_cli_dry_run_never_constructs_client(inputs, monkeypatch, capsys):
    from scripts.measure_local_models import main
    monkeypatch.setattr(local, 'OpenAICompatibleClient', lambda *a, **k: pytest.fail('dry-run client'))
    assert main(['--plan', str(inputs[0]), '--manifest', str(inputs[1]), '--corpus', str(inputs[2]),
                 '--dry-run']) == 0
    output = capsys.readouterr().out
    assert json.loads(output)['status'] == 'dry_run'
    assert 'private' not in output


def test_existing_adapter_uses_journal_and_no_ambient_credential(inputs, tmp_path, monkeypatch):
    plan = local.load_plan(*inputs)
    budget = local.LocalAttemptBudget(plan, tmp_path / 'attempts.json')
    monkeypatch.setenv('NVIDIA_API_KEY', 'SECRET')
    client = budget.client()
    assert client._credential is None
    class Reply:
        status_code = 200
        headers = {}
        def json(self):
            return {'choices': [{'message': {'content': 'OK'}, 'finish_reason': 'stop'}]}
    def post(payload):
        assert json.loads(budget.path.read_text())['attempts'][0]['status'] == 'unknown_outcome'
        assert payload['model'] == plan.config.model
        return Reply()
    monkeypatch.setattr(client, '_post', post)
    assert client.generate('private').content == 'OK'
    assert budget.used == 1


def test_reported_token_overrun_stops_later_admission(inputs, tmp_path):
    from recall_guard.core.completion import ModelResponse
    from recall_guard.core.request_control import BudgetExhaustedError, RequestControl
    plan = local.load_plan(*inputs)
    budget = local.LocalAttemptBudget(plan, tmp_path / 'attempts.json')
    control = RequestControl(plan.config, budget)
    control.run(lambda: ModelResponse('OK', usage={'completion_tokens': 33}), stage='probe')
    with pytest.raises(BudgetExhaustedError):
        control.run(lambda: pytest.fail('token overrun'), stage='probe')



def test_explicit_credential_reference_and_local_identity(inputs, tmp_path, monkeypatch):
    manifest = json.loads(inputs[1].read_text())
    manifest.update(authentication='environment', credential_ref='LOCAL_TEST_KEY')
    inputs[1].write_text(json.dumps(manifest), encoding='utf-8')
    data = json.loads(inputs[0].read_text())
    data['manifest_sha256'] = digest(inputs[1].read_bytes())
    inputs[0].write_text(json.dumps(data), encoding='utf-8')
    plan = local.load_plan(*inputs)
    budget = local.LocalAttemptBudget(plan, tmp_path / 'attempts.json')
    monkeypatch.setenv('NVIDIA_API_KEY', 'WRONG')
    client = budget.client(environment={'LOCAL_TEST_KEY': 'SELECTED_SECRET'})
    assert client._credential == 'SELECTED_SECRET'
    assert 'SELECTED_SECRET' not in json.dumps(plan.safe_export())
    with pytest.raises(ValueError, match='missing'):
        budget.client(environment={'NVIDIA_API_KEY': 'WRONG'})


def test_programmatic_plan_cannot_bypass_validation(inputs):
    from dataclasses import replace
    plan = local.load_plan(*inputs)
    with pytest.raises(ValueError, match='local_identity'):
        replace(plan, config=replace(plan.config, local_identity=None))
    with pytest.raises(ValueError, match='item_key'):
        replace(plan, items=plan.items + plan.items)
    with pytest.raises(ValueError, match='request_bound'):
        replace(plan, ensemble_draws=1000)
    with pytest.raises(ValueError, match='authentication'):
        replace(plan, config=replace(plan.config, credential_ref='LOCAL_KEY'))


def test_gateway_timeout_is_unknown_and_never_retried(inputs, tmp_path):
    from recall_guard.core.request_control import BudgetExhaustedError, RequestControl, RequestError
    plan = local.load_plan(*inputs)
    budget = local.LocalAttemptBudget(plan, tmp_path / 'attempts.json')
    def timeout():
        raise RequestError('gateway timeout', status_code=504, retryable=True)
    with pytest.raises(BudgetExhaustedError):
        RequestControl(plan.config, budget).run(timeout, stage='probe')
    assert budget.used == 1
    assert json.loads(budget.path.read_text())['attempts'][0]['status'] == 'unknown_outcome'


def test_received_response_without_final_answer_is_a_known_item_failure(inputs, tmp_path):
    from recall_guard.core.request_control import RequestControl, RequestError
    plan = local.load_plan(*inputs)
    budget = local.LocalAttemptBudget(plan, tmp_path / 'attempts.json')
    def truncated():
        raise RequestError('Harmony response token mapping failed', code='response_mapping_failed',
                           usage={'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15})
    with pytest.raises(RequestError):
        RequestControl(plan.config, budget).run(truncated, stage='probe')
    attempt = json.loads(budget.path.read_text())['attempts'][0]
    assert attempt['status'] != 'unknown_outcome' and attempt['code'] == 'response_mapping_failed'
    assert not budget._stopped


@pytest.mark.parametrize('parameters', [{'n': 20}, {'n': True}, {'n': 1.0}, {'n': None}, {'best_of': 2}, {'best_of': 1}])
def test_programmatic_plan_rejects_unbudgeted_completion_controls(inputs, parameters):
    from dataclasses import replace
    plan = local.load_plan(*inputs)
    config = replace(plan.config, parameters=dict(plan.config.parameters) | parameters)
    with pytest.raises(ValueError, match='single_completion'):
        replace(plan, config=config)


@pytest.mark.parametrize('parameters', [{'n': 20}, {'n': True}, {'n': 1.0}, {'n': None}, {'best_of': 2}, {'best_of': 1}])
def test_file_plan_rejects_unbudgeted_completion_controls(inputs, parameters):
    manifest = json.loads(inputs[1].read_text())
    manifest['config']['parameters'].update(parameters)
    inputs[1].write_text(json.dumps(manifest), encoding='utf-8')
    data = json.loads(inputs[0].read_text())
    data['manifest_sha256'] = digest(inputs[1].read_bytes())
    inputs[0].write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(ValueError, match='single_completion'):
        local.load_plan(*inputs)


def test_explicit_single_completion_is_budgeted(inputs):
    from dataclasses import replace
    plan = local.load_plan(*inputs)
    config = replace(plan.config, parameters=dict(plan.config.parameters) | {'n': 1})
    assert replace(plan, config=config).physical_requests == plan.physical_requests


def test_zero_draws_is_a_single_answer_only_plan(inputs):
    data = json.loads(inputs[0].read_text())
    data['ensemble_draws'] = 0
    inputs[0].write_text(json.dumps(data), encoding='utf-8')
    plan = local.load_plan(*inputs)
    assert plan.logical_requests == 5  # probe + IS + OOS baseline + OOS training + single
    data['ensemble_draws'] = -1
    inputs[0].write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(ValueError, match='nonnegative'):
        local.load_plan(*inputs)


def _relabel(inputs, labels):
    plan_path, manifest_path, corpus_path = inputs
    corpus = json.loads(corpus_path.read_text())
    for split, label in labels.items():
        corpus[split][0]['label'] = label
    corpus_path.write_text(json.dumps(corpus), encoding='utf-8')
    data = json.loads(plan_path.read_text())
    data['corpus_sha256'] = digest(corpus_path.read_bytes())
    plan_path.write_text(json.dumps(data), encoding='utf-8')


def test_unlabeled_plans_load_and_export_unchanged(inputs):
    plan = local.load_plan(*inputs)
    assert all(item.label is None for item in plan.items)
    assert all('label' not in item for item in plan.safe_export()['items'])


def test_labeled_items_load_and_export_labels(inputs):
    _relabel(inputs, {'is_memorized': 1, 'oos_control': 0, 'held_out': 0})
    plan = local.load_plan(*inputs)
    assert {item.split: item.label for item in plan.items} == {'is_memorized': 1, 'oos_control': 0, 'held_out': 0}
    assert [item['label'] for item in plan.safe_export()['items']] == [1, 0, 0]


@pytest.mark.parametrize('labels', [{'is_memorized': 0}, {'oos_control': 1}, {'held_out': 2},
                                    {'held_out': True}, {'held_out': '1'}])
def test_invalid_labels_refused(inputs, labels):
    _relabel(inputs, labels)
    with pytest.raises(ValueError, match='invalid_item_label'):
        local.load_plan(*inputs)
