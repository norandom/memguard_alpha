"""The documented local example exercises real scoring without a model host."""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


def run_example(*args):
    return subprocess.run([sys.executable, 'examples/local_provider.py', *args],
                          capture_output=True, text=True, check=True)


def test_default_is_zero_call_dry_run():
    result = json.loads(run_example().stdout)
    assert result['status'] == 'dry_run'
    assert result['network_requests'] == 0
    assert result['plan']['logical_requests'] == 29


def test_fixture_runs_real_provider_and_scorer():
    result = json.loads(run_example('--fixture').stdout)
    assert result['status'] == 'completed'
    assert result['fixture_attempts'] == 29
    assert result['scored_held_out_rows'] == 4
    assert result['network_requests'] == 0


def test_written_inputs_load_in_measurement_cli(tmp_path):
    target = tmp_path / 'inputs'
    run_example('--write-inputs', str(target))
    result = subprocess.run([sys.executable, 'scripts/measure_local_models.py',
        '--plan', str(target / 'plan.json'), '--manifest', str(target / 'manifest.json'),
        '--corpus', str(target / 'corpus.json'), '--dry-run'],
        capture_output=True, text=True, check=True)
    assert json.loads(result.stdout)['plan']['physical_requests'] == 29
    assert not list(Path(target).glob('*journal*'))


def test_fixture_blocks_real_network_and_reports_failure(monkeypatch, capsys):
    import requests

    from examples import local_provider

    def forbidden(*args, **kwargs):
        raise AssertionError('offline example attempted network access')

    monkeypatch.setattr(requests.Session, 'request', forbidden)
    assert local_provider.main(['--fixture']) == 0
    assert json.loads(capsys.readouterr().out)['status'] == 'completed'
    monkeypatch.setattr(local_provider, 'execute_plan',
                        lambda *args: SimpleNamespace(status='ineligible', rows=()))
    assert local_provider.main(['--fixture']) == 2
    assert json.loads(capsys.readouterr().out)['status'] == 'ineligible'
