import json
from pathlib import Path

import pytest
import requests

from recall_guard.core.completion import ModelResponse, TokenAlternative, TokenEvidence
from scripts import probe_local_model as probe

# Synthetic contract data travels with tests into the filtered CI context.
MANIFEST = Path(__file__).resolve().parents[1] / 'fixtures/local_provider/candidate-manifest.json'


def response(content=' OK', logprob=-0.1, alternatives=5):
    return ModelResponse(content, (TokenEvidence(' OK', logprob, tuple(
        TokenAlternative(str(i), -i - 1.0) for i in range(alternatives))),))


def test_exact_unicode_and_whitespace_alignment():
    probe.validate_evidence(response())
    with pytest.raises(ValueError, match='alignment'):
        probe.validate_evidence(response('OK'))
    unicode_response = ModelResponse(' café\n', (TokenEvidence(' café\n', -.1, (
        TokenAlternative('other', -2.0),)),))
    probe.validate_evidence(unicode_response, 1)


@pytest.mark.parametrize('value,reason', [
    (ModelResponse(''), 'empty_output'),
    (ModelResponse('hello'), 'missing_token_evidence'),
    (response(logprob=-9999), 'censored'),
    (response(alternatives=1), 'missing_alternatives'),
])
def test_rejects_unusable_evidence(value, reason):
    with pytest.raises(ValueError, match=reason):
        probe.validate_evidence(value)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    candidate = manifest['candidates'][0]
    host = {'endpoint': candidate['endpoint'], 'configured_context_tokens': 2048,
            'max_loaded_models': 1, 'num_parallel': 1, 'process_id': 123,
            'context_source': 'process-start-environment'}
    host_path = tmp_path / 'host.json'
    host_path.write_text(json.dumps(host), encoding='utf-8')
    observed = {'runtime_version': 'test', 'digest': candidate['manifest_sha256'],
                'loaded_models': [], 'effective_context_tokens': None,
                'model_allocation_bytes': None, 'accelerator_allocation_bytes': None,
                'observed_backend': None}
    monkeypatch.setattr(probe, 'observe_host', lambda config: dict(observed), raising=False)
    return tmp_path, host_path, candidate, observed


def install_response(monkeypatch, tmp_path, value=None, error=None):
    calls = []
    def post(self, payload):
        calls.append(payload)
        journal = json.loads((tmp_path / 'journal.json').read_text(encoding='utf-8'))
        assert journal['attempts'][-1]['status'] == 'intent'
        assert self.budget.used == 1
        assert self.config.credential_ref is None
        assert self._credential is None
        assert self.config.max_retries == 0
        assert self.config.timeout_s == 300
        if error:
            raise error
        result = requests.Response()
        result.status_code = 200
        token = {'token': ' OK', 'logprob': -.1, 'top_logprobs': [
            {'token': str(i), 'logprob': -i - 1.0} for i in range(5)]}
        body = value or {'choices': [{'message': {'content': ' OK'},
                                     'logprobs': {'content': [token]}}]}
        result._content = json.dumps(body).encode()
        return result
    monkeypatch.setattr(probe.OpenAICompatibleClient, '_post', post)
    return calls


def run(setup, **kwargs):
    path, host, candidate, _ = setup
    return probe.run_probe(MANIFEST, candidate['id'], host, path / 'journal.json', **kwargs)


def test_real_adapter_budget_and_safe_receipt(setup, monkeypatch):
    path, _, _, _ = setup
    calls = install_response(monkeypatch, path)
    result = run(setup)
    assert result['status'] == 'eligible'
    assert result['physical_attempts'] == 1
    assert result['events'][0]['status'] == 'succeeded'
    assert calls[0]['max_tokens'] == 32
    assert 'num_ctx' not in calls[0]
    exported = (path / 'journal.json').read_text(encoding='utf-8')
    assert 'Reply with' not in exported
    assert 'top_logprobs' not in result.get('response', {})


def test_missing_logprobs_is_ineligible(setup, monkeypatch):
    install_response(monkeypatch, setup[0], {'choices': [{'message': {'content': 'secret answer'}}]})
    result = run(setup)
    assert result['status'] == 'ineligible'
    assert result['reason'] == 'missing_token_evidence'
    assert 'secret answer' not in json.dumps(result)


def test_timeout_no_retry_no_replay_and_no_diagnostics_leak(setup, monkeypatch):
    calls = install_response(monkeypatch, setup[0], error=probe.RequestError(
        'PRIVATE SECRET', code='timeout', detail='PRIVATE SECRET', retryable=True))
    result = run(setup)
    assert result['status'] == 'unavailable'
    assert result['host_execution_stopped'] is None
    assert result['reason'] == 'timeout'
    assert len(calls) == 1
    assert 'PRIVATE SECRET' not in json.dumps(result)
    with pytest.raises(ValueError, match='unknown_outcome'):
        run(setup)
    assert len(calls) == 1


def test_two_attempt_limit_and_dry_run(setup, monkeypatch):
    calls = install_response(monkeypatch, setup[0])
    dry = run(setup, dry_run=True)
    assert dry['physical_attempts'] == 0
    assert not calls
    run(setup)
    run(setup)
    with pytest.raises(ValueError, match='candidate_budget'):
        run(setup)
    assert len(calls) == 2


def test_context_and_digest_gate_prevents_inference(setup, monkeypatch):
    calls = install_response(monkeypatch, setup[0])
    setup[3]['effective_context_tokens'] = 4096
    result = run(setup)
    assert result['reason'] == 'context_bound_not_established'
    assert result['physical_attempts'] == 0
    assert not calls



def test_censored_alternative_rejected():
    value = ModelResponse('OK', (TokenEvidence('OK', -.1, (TokenAlternative('no', -9999),)),))
    with pytest.raises(ValueError, match='censored'):
        probe.validate_evidence(value, 1)


def test_digest_and_other_loaded_model_gate(setup, monkeypatch):
    calls = install_response(monkeypatch, setup[0])
    setup[3]['digest'] = 'different'
    assert run(setup)['reason'] == 'model_digest_missing_or_mismatched'
    setup[3]['digest'] = setup[2]['manifest_sha256']
    setup[3]['loaded_models'] = ['another-model']
    assert run(setup)['reason'] == 'other_model_loaded'
    assert not calls


def test_missing_host_context_gate(setup, monkeypatch):
    calls = install_response(monkeypatch, setup[0])
    setup[1].write_text('{}', encoding='utf-8')
    assert run(setup)['reason'] == 'host_configuration_not_established'
    assert not calls


def test_interrupted_intent_cannot_be_replayed(setup, monkeypatch):
    calls = install_response(monkeypatch, setup[0])
    run(setup)
    path = setup[0] / 'journal.json'
    journal = json.loads(path.read_text(encoding='utf-8'))
    journal['attempts'][0]['status'] = 'intent'
    path.write_text(json.dumps(journal), encoding='utf-8')
    with pytest.raises(ValueError, match='unknown_outcome'):
        run(setup)
    assert len(calls) == 1


def test_shared_journal_never_transfers_blocked_npu_budget(setup, monkeypatch):
    calls = install_response(monkeypatch, setup[0])
    run(setup)
    run(setup)
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    gpt = manifest['candidates'][1]
    setup[3]['digest'] = gpt['manifest_sha256']
    for _ in range(2):
        result = probe.run_probe(MANIFEST, gpt['id'], setup[1], setup[0] / 'journal.json')
        assert result['physical_attempts'] == 1
    with pytest.raises(ValueError, match='candidate_budget'):
        probe.run_probe(MANIFEST, gpt['id'], setup[1], setup[0] / 'journal.json')
    assert len(calls) == 4
    with pytest.raises(ValueError, match='candidate_not_in_frozen'):
        probe.run_probe(MANIFEST, 'amd-gemma3-4b-npu', setup[1], setup[0] / 'journal.json')


def test_frozen_plan_drift_blocks_admission(setup, monkeypatch):
    calls = install_response(monkeypatch, setup[0])
    run(setup)
    monkeypatch.setattr(probe, 'PROMPT', 'changed private prompt')
    with pytest.raises(ValueError, match='frozen_plan_changed'):
        run(setup)
    assert len(calls) == 1


def test_elapsed_limit_records_overrun_without_claiming_host_stopped(setup, monkeypatch):
    calls = install_response(monkeypatch, setup[0])
    ticks = iter([0, 301])
    monkeypatch.setattr(probe, 'monotonic', lambda: next(ticks))
    result = run(setup)
    assert result['reason'] == 'attempt_elapsed_bound_exceeded'
    assert result['elapsed_s'] == 301
    assert result['host_execution_stopped'] is None
    with pytest.raises(ValueError, match='unknown_outcome'):
        run(setup)
    assert len(calls) == 1


def test_timeout_stays_unresolved_even_if_post_metadata_drifts(setup, monkeypatch):
    calls = install_response(monkeypatch, setup[0], error=probe.RequestError('timeout', code='timeout'))
    original = dict(setup[3])
    drifted = original | {'digest': 'changed'}
    observations = iter([original, drifted])
    monkeypatch.setattr(probe, 'observe_host', lambda config: next(observations))
    result = run(setup)
    assert result['unknown_outcome'] is True
    with pytest.raises(ValueError, match='unknown_outcome'):
        run(setup)
    assert len(calls) == 1
