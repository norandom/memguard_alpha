import json

import pytest
import requests

from scripts import smoke_local_models as smoke
from scripts.probe_local_model import OpenAICompatibleClient, RequestError

ENDPOINT = 'http://127.0.0.1:11434/v1/chat/completions'
MODELS = ('phi4-mini:3.8b', 'gpt-oss:20b')
GIB = 1024 ** 3


@pytest.fixture
def rig(tmp_path, monkeypatch):
    state = {'loaded': [], 'posts': [], 'unloads': [], 'context': 2048, 'missing': set(),
             'error': None, 'body': None, 'unload_ok': True}
    path = tmp_path / 'receipt.json'
    host = {'endpoint': ENDPOINT, 'configured_context_tokens': 2048, 'max_loaded_models': 1,
            'num_parallel': 1, 'process_id': 123, 'context_source': 'process-start-environment'}

    def observe(config):
        return {'digest': None if config.model in state['missing'] else 'sha256:' + 'a' * 64,
                'runtime_version': '0.15.0', 'loaded_models': list(state['loaded']),
                'effective_context_tokens': state['context'] if state['loaded'] else None,
                'model_allocation_bytes': 100, 'accelerator_allocation_bytes': 0,
                'observed_backend': 'CPU'}

    def unload(config):
        state['unloads'].append(config.model)
        if state['unload_ok']:
            state['loaded'] = []

    def post(self, payload):
        receipt = json.loads(path.read_text())
        assert receipt['attempts'][-1]['status'] == 'intent'
        assert self.budget.used == 1
        assert self.config.max_retries == 0 and self.config.timeout_s == 300
        assert not state['loaded']
        assert payload['max_tokens'] == 32 and payload['top_logprobs'] == 5
        assert payload['temperature'] == 0 and payload['top_p'] == 1 and payload['logprobs']
        state['posts'].append(payload['model'])
        state['loaded'] = [payload['model']]
        if state['error']:
            raise state['error']
        token = {'token': 'OK', 'logprob': -.1, 'top_logprobs': [
            {'token': str(i), 'logprob': -i - 1.0} for i in range(5)]}
        body = state['body'] or {'choices': [{'message': {'content': 'OK'},
                                            'logprobs': {'content': [token]}}],
                                 'usage': {'prompt_tokens': 10, 'completion_tokens': 1, 'total_tokens': 11}}
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(body).encode()
        return response

    monkeypatch.setattr(smoke, 'observe_host', observe, raising=False)
    monkeypatch.setattr(smoke, 'unload_model', unload, raising=False)
    monkeypatch.setattr(OpenAICompatibleClient, '_post', post)

    def run(**kwargs):
        return smoke.run_smoke(ENDPOINT, host, path, models=MODELS, **kwargs)
    return state, path, run


def test_default_dry_run_has_no_inference_or_unload(rig):
    state, path, run = rig
    result = run()
    assert result['status'] == 'dry_run'
    assert not state['posts'] and not state['unloads'] and not path.exists()


@pytest.mark.parametrize('available,expected', [(None, 1), (16 * GIB - 1, 1),
                                               (16 * GIB, 2), (17 * GIB, 2)])
def test_memory_gate_and_serial_real_adapter(rig, available, expected):
    state, path, run = rig

    def memory():
        assert not state['loaded']
        return available

    result = run(execute=True, memory_reader=memory)
    assert len(state['posts']) == expected
    assert state['unloads'] == state['posts']
    assert result['attempts'][0]['status'] == 'eligible'
    assert result['attempts'][1]['memory_gate']['available_bytes'] == available
    assert result['attempts'][1]['memory_gate']['threshold_bytes'] == 16 * GIB
    assert result['attempts'][0]['tokens'][0]['logprob'] == -.1
    assert result['attempts'][0]['tokens'][0]['alternative_count'] == 5
    assert 'OK' not in path.read_text()
    with pytest.raises(FileExistsError):
        run(execute=True, memory_reader=memory)
    assert len(state['posts']) == expected


@pytest.mark.parametrize('kind,reason', [('missing', 'missing_token_evidence'),
                                       ('censored', 'censored'), ('misaligned', 'alignment')])
def test_bad_evidence_and_context_separate(rig, kind, reason):
    state, _, run = rig
    token = {'token': 'OK', 'logprob': -9999 if kind == 'censored' else -.1,
             'top_logprobs': [{'token': str(i), 'logprob': -1.0} for i in range(5)]}
    state['body'] = {'choices': [{'message': {'content': 'PRIVATE' if kind == 'misaligned' else 'OK'},
                                 'logprobs': None if kind == 'missing' else {'content': [token]}}]}
    state['context'] = 4096
    result = run(execute=True, memory_reader=lambda: None)['attempts'][0]
    assert result['evidence_status'] == 'ineligible'
    assert reason in result['evidence_reason']
    assert result['environment_status'] == 'mismatched'
    assert 'PRIVATE' not in json.dumps(result)


def test_valid_evidence_context_mismatch(rig):
    state, _, run = rig
    state['context'] = 4096
    result = run(execute=True, memory_reader=lambda: None)['attempts'][0]
    assert result['evidence_status'] == 'valid'
    assert result['environment_status'] == 'mismatched'
    assert result['status'] == 'ineligible'


def test_unload_failure_stops(rig):
    state, _, run = rig
    state['unload_ok'] = False
    result = run(execute=True, memory_reader=lambda: 20 * GIB)
    assert len(state['posts']) == 1
    assert result['stop_reason'] == 'unload_not_verified'


@pytest.mark.parametrize('error', [RequestError('PRIVATE', code='timeout'), KeyboardInterrupt()])
def test_unknown_or_cancelled_stops_preserves_receipt(rig, error):
    state, path, run = rig
    state['error'] = error
    result = run(execute=True, memory_reader=lambda: 20 * GIB)
    assert len(state['posts']) == 1 and not state['unloads']
    assert result['attempts'][0]['unknown_outcome']
    assert json.loads(path.read_text())['status'] == 'stopped'
    assert 'PRIVATE' not in path.read_text()


def test_missing_model_skipped_no_substitution(rig):
    state, _, run = rig
    state['missing'].add(MODELS[0])
    result = run(execute=True, memory_reader=lambda: 20 * GIB)
    assert state['posts'] == [MODELS[1]]
    assert result['attempts'][0]['reason'] == 'model_missing_or_digest_unknown'


def test_unrelated_loaded_refused(rig):
    state, _, run = rig
    state['loaded'] = ['other']
    result = run(execute=True, memory_reader=lambda: 20 * GIB)
    assert not state['posts'] and not state['unloads']
    assert result['stop_reason'] == 'other_model_loaded'


def test_remote_gate_never_uses_client_ram(monkeypatch):
    def forbidden():
        pytest.fail('remote host must not use client memory')
    monkeypatch.setattr(smoke, 'available_memory_bytes', forbidden)
    result = smoke.memory_gate('http://server:11434/v1/chat/completions', 16 * GIB)
    assert result['available_bytes'] is None and not result['admitted']
    verified = smoke.memory_gate('http://server:11434/v1/chat/completions', 16 * GIB, lambda: 17 * GIB)
    assert verified['admitted'] and verified['source'] == 'verified_host_callback'


def test_windows_actual_free_ram_reader(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(smoke.sys, 'platform', 'win32')

    def run(command, **kwargs):
        assert 'FreePhysicalMemory' in command[-1]
        assert kwargs['timeout'] == 10
        return SimpleNamespace(stdout='16777216\n')

    monkeypatch.setattr(smoke.subprocess, 'run', run)
    assert smoke.available_memory_bytes() == 16 * GIB


@pytest.mark.parametrize('threshold', [0, -1, float('nan'), float('inf')])
def test_invalid_threshold_rejected_without_io(rig, threshold):
    state, path, run = rig
    with pytest.raises(ValueError, match='positive'):
        run(execute=True, minimum_available_gib=threshold)
    assert not path.exists() and not state['posts']


def test_initial_selected_model_unloaded_before_memory_sample(rig):
    state, _, run = rig
    state['loaded'] = [MODELS[0]]
    result = run(execute=True, memory_reader=lambda: 20 * GIB)
    assert result['status'] == 'completed'
    assert state['unloads'] == [MODELS[0], MODELS[0], MODELS[1]]


def test_identity_change_since_freeze_prevents_inference(rig, monkeypatch):
    state, _, run = rig
    original = smoke.observe_host
    calls = 0

    def observe(config):
        nonlocal calls
        calls += 1
        result = original(config)
        if calls > len(MODELS):
            result['digest'] = 'sha256:' + 'b' * 64
        return result

    monkeypatch.setattr(smoke, 'observe_host', observe)
    result = run(execute=True, memory_reader=lambda: 20 * GIB)
    assert not state['posts']
    assert all(row['reason'] == 'model_digest_missing_or_mismatched' for row in result['attempts'])


def test_profile_selection_maps_curated_profiles_and_keeps_explicit_endpoint():
    models, endpoint = smoke.profile_selection(['ollama-phi4-mini', 'ollama-gpt-oss-20b'], None)
    assert models == ['phi4-mini:3.8b', 'gpt-oss:20b']
    assert endpoint == 'http://localhost:11434/v1/chat/completions'
    assert smoke.profile_selection([], None) == ([], smoke.DEFAULT_ENDPOINT)
    assert smoke.profile_selection(['ollama-gemma3-4b'], ENDPOINT) == (['gemma3:4b'], ENDPOINT)
    with pytest.raises(ValueError):
        smoke.profile_selection(['nim-llama3.1-70b'], None)


def test_profile_preflight_uses_selected_endpoint_and_main_prints_panel(monkeypatch, capsys):
    from recall_guard.core.profiles import HostPreflightReport

    seen = []

    def check(profile, *, environment):
        seen.append((profile.profile_id, profile.model_config.endpoint, dict(environment)))
        return HostPreflightReport(profile.profile_id, profile.model_config.model, passed=False,
                                   reason='insufficient_free_ram',
                                   recommendation='13.4 GiB free, 20.0 GiB required for gpt-oss:20b.',
                                   fallback_profile_ids=('modal-gpt-oss-20b',),
                                   required_free_ram_bytes=20 * GIB)

    with pytest.raises(smoke.PreflightError):
        smoke.profile_preflight(['ollama-gpt-oss-20b'], ENDPOINT, check=check)
    assert seen == [('ollama-gpt-oss-20b', ENDPOINT, {})]
    monkeypatch.setattr(smoke, 'preflight_host', check)
    monkeypatch.setattr(smoke.profile_preflight, '__kwdefaults__', {'check': check})
    monkeypatch.setattr('sys.argv', ['smoke', '--profile', 'ollama-gpt-oss-20b'])
    assert smoke.main() == 2
    err = capsys.readouterr().err
    assert 'Not enough free memory' in err and '--profile modal-gpt-oss-20b' in err and 'Traceback' not in err
