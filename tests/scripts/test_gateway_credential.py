"""Bearer credential for the authenticating Windows Ollama gateway (release-1.0 D3). Offline."""

import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest
import requests

from recall_guard.core.profiles import (
    HostPreflightReport,
    ProfileError,
    load_profile,
    metadata_headers,
    preflight_host,
)
from scripts import smoke_local_models as smoke
from scripts.probe_local_model import CallBudget, ProbeConfig, connect_probe

ROOT = Path(__file__).resolve().parents[2]
GATEWAY = 'http://172.17.32.1:11435/v1/chat/completions'
REF = 'env:LOCAL_MODEL_API_KEY'
TOKEN = 'test-token-not-a-secret'
GIB = 1024 ** 3


class Answer:
    def __init__(self, body):
        self.status_code, self._body = 200, body

    def json(self):
        return self._body


@pytest.fixture
def http(monkeypatch):
    seen = []
    digest = 'a' * 64

    def get(self, url, **kwargs):
        seen.append(('GET', url, dict(kwargs.get('headers') or {})))
        if url.endswith('/api/version'):
            return Answer({'version': '0.40.11'})
        if url.endswith('/api/tags'):
            return Answer({'models': [{'name': 'gemma3:4b', 'digest': digest}]})
        return Answer({'models': []})

    def post(self, url, **kwargs):
        seen.append(('POST', url, dict(kwargs.get('headers') or {})))
        return Answer({})

    monkeypatch.setattr(requests.Session, 'get', get)
    monkeypatch.setattr(requests.Session, 'post', post)
    return seen


def protected_gemma():
    from dataclasses import replace

    profile = load_profile('ollama-gemma3-4b')
    return replace(profile, model_config=replace(profile.model_config, endpoint=GATEWAY, credential_ref=REF),
                   allow_unauthenticated=False)


def test_metadata_headers_only_for_named_credential():
    assert metadata_headers(load_profile('ollama-gemma3-4b'), {'LOCAL_MODEL_API_KEY': TOKEN}) == {}
    assert metadata_headers(protected_gemma(), {'LOCAL_MODEL_API_KEY': TOKEN}) == {
        'Authorization': f'Bearer {TOKEN}'}
    with pytest.raises(ProfileError, match='LOCAL_MODEL_API_KEY') as error:
        metadata_headers(protected_gemma(), {'LOCAL_MODEL_API_KEY': ' '})
    assert TOKEN not in str(error.value)


def test_preflight_sends_bearer_on_every_metadata_call(http):
    snapshot = {'available_bytes': 24 * GIB, 'total_bytes': 32 * GIB, 'host': 'MINI',
                'observed_at': datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ'), 'source': 'windows_host_snapshot'}
    report = preflight_host(protected_gemma(), environment={'LOCAL_MODEL_API_KEY': TOKEN},
                            host_snapshot=snapshot, in_container=True)
    assert report.passed and report.runtime_version == '0.40.11'
    assert [url.rsplit('/', 2)[-2:] for _, url, _ in http] == [['api', 'version'], ['api', 'tags'], ['api', 'ps']]
    assert all(headers == {'Authorization': f'Bearer {TOKEN}'} for _, _, headers in http)


def test_unauthenticated_preflight_sends_no_authorization(http):
    snapshot = {'available_bytes': 24 * GIB, 'total_bytes': None, 'host': 'MINI',
                'observed_at': datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ'), 'source': 'windows_host_snapshot'}
    assert preflight_host('ollama-gemma3-4b', environment={'LOCAL_MODEL_API_KEY': TOKEN},
                          host_snapshot=snapshot, in_container=True).passed
    assert http and all(headers == {} for _, _, headers in http)


def test_probe_config_carries_credential_to_client_metadata_and_unload(http, monkeypatch):
    monkeypatch.setenv('LOCAL_MODEL_API_KEY', TOKEN)
    monkeypatch.setenv('NVIDIA_API_KEY', 'never-forwarded')
    config = ProbeConfig('gemma3:4b', 'gemma3:4b', GATEWAY, '', '', REF)
    assert config.environment() == {'LOCAL_MODEL_API_KEY': TOKEN}
    assert config.profile().allow_unauthenticated is False
    client = connect_probe(config, CallBudget(1))
    assert client.config.credential_ref == REF
    smoke.unload_model(config)
    smoke.observe_host(config)
    assert {headers['Authorization'] for _, _, headers in http} == {f'Bearer {TOKEN}'}
    plain = ProbeConfig('gemma3:4b', 'gemma3:4b', GATEWAY, '', '')
    assert plain.environment() == {} and plain.auth_headers() == {} and plain.profile().allow_unauthenticated


def test_smoke_profile_preflight_uses_credential_profile(monkeypatch):
    monkeypatch.setenv('LOCAL_MODEL_API_KEY', TOKEN)
    seen = []

    def check(profile, *, environment):
        seen.append((profile.model_config.credential_ref, profile.allow_unauthenticated, dict(environment)))
        return HostPreflightReport(profile.profile_id, profile.model_config.model, passed=True)

    smoke.profile_preflight(['ollama-gemma3-4b'], GATEWAY, check=check, credential_ref=REF)
    assert seen == [(REF, False, {'LOCAL_MODEL_API_KEY': TOKEN})]


def test_smoke_refuses_unset_credential_without_http(monkeypatch, capsys):
    monkeypatch.delenv('LOCAL_MODEL_API_KEY', raising=False)

    def forbidden(*args, **kwargs):
        raise AssertionError('no HTTP when the credential is unset')

    monkeypatch.setattr(requests.Session, 'request', forbidden)
    assert smoke.main(['--profile', 'ollama-gemma3-4b', '--endpoint', GATEWAY, '--credential-ref', REF]) == 2
    err = capsys.readouterr().err
    assert 'LOCAL_MODEL_API_KEY' in err and 'Traceback' not in err


def test_smoke_dry_run_records_authentication_mode(http, monkeypatch):
    monkeypatch.setenv('LOCAL_MODEL_API_KEY', TOKEN)
    journal = smoke.run_smoke(GATEWAY, {}, models=['gemma3:4b'], credential_ref=REF)
    assert journal['status'] == 'dry_run' and journal['plan']['authentication'] == 'environment'
    assert TOKEN not in json.dumps(journal)


@pytest.mark.skipif(shutil.which('pwsh') is None, reason='PowerShell 7 not installed')
def test_gateway_script_parses():
    script = ROOT / 'scripts' / 'serve_local_ollama.ps1'
    command = ("$e=$null; [System.Management.Automation.Language.Parser]::ParseFile("
               f"'{script}', [ref]$null, [ref]$e) | Out-Null; exit $e.Count")
    assert subprocess.run(['pwsh', '-NoProfile', '-Command', command], timeout=60).returncode == 0
