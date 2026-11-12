"""Task 6.4: the 30-minute soak runner of the full-distribution host, on fakes and short durations.

``scripts/soak_full_distribution.py`` sends one batch request every interval through the gateway,
samples free physical memory and the recorded host process, counts the host's and the gateway's
memory log lines, and writes a receipt with timestamps, sizes, status codes, timings and memory
only. Here the "gateway" is an in-process host on an ephemeral loopback port with a fake model.
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path

import psutil
import pytest

from scripts import soak_full_distribution as soak
from scripts.hosts import serve_full_distribution as sfd
from tests.scripts.test_serve_full_distribution import DIGEST
from tests.scripts.test_serve_full_distribution_batch import CachedFakeLlama
from tests.scripts.test_serve_full_distribution_lifecycle import PWSH, _free_ports, _kill_tree, _running_hosts

NAME = 'SOAK_TEST_BEARER'
TOKEN = 'soak-test-bearer-not-a-secret'
ENVIRONMENT = {NAME: TOKEN}


@pytest.fixture
def host_url():
    """Start in-process hosts (``make_server`` on port 0) in threads; return a factory of base URLs."""
    started = []

    def start(memory_guard=None):
        host = sfd.FullDistributionHost(CachedFakeLlama(), model_digest=DIGEST, memory_guard=memory_guard)
        server = sfd.make_server(host, port=0)
        thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': 0.05}, daemon=True)
        thread.start()
        started.append((server, thread))
        return f'http://127.0.0.1:{server.server_address[1]}'

    yield start
    for server, thread in started:
        server.shutdown()
        server.server_close()
        thread.join(5)


def run(url, **changes):
    values = {'credential_name': NAME, 'environment': ENVIRONMENT, 'duration_s': 0.6, 'interval_s': 0.2,
              'sample_s': 0.05, 'model_digest': DIGEST} | changes
    return soak.soak(url, **values)


def test_soak_sends_one_batch_per_interval_and_samples_memory(host_url):
    receipt = run(host_url())
    requests = receipt['requests']
    assert [entry['index'] for entry in requests] == [0, 1, 2]  # ceil(0.6 / 0.2) requests.
    assert all(entry['status'] == 'ok' and entry['status_code'] == 200 for entry in requests)
    assert all(entry['responses'] == len(soak.SOAK_ITEM['response_texts']) and entry['positions'] > 0
               for entry in requests)
    for entry, due in zip(requests, (0.0, 0.2, 0.4), strict=True):  # On schedule, never early.
        # Lateness is bounded loosely: requests run one after another and a loaded machine delays them.
        assert due <= entry['offset_s'] < due + 1.0
    summary = receipt['summary']
    assert (summary['requests_sent'], summary['requests_expected'], summary['ok']) == (3, 3, 3)
    assert summary['distinct_evidence'] == 1  # The same batch gives the same evidence every time.
    assert summary['memory_samples'] >= 6 and summary['min_free_ram_bytes'] > 0
    assert receipt['memory'][-1]['t_s'] >= 0.55  # Sampling covers the whole duration.
    assert receipt['outcome'] == {'passed': True, 'failures': []}
    assert receipt['host'] is None and receipt['credential'] == {'name': NAME, 'source': 'environment'}
    assert TOKEN not in json.dumps(receipt)


def test_soak_records_memory_low_refusals_as_retryable_host_busy_and_fails(host_url):
    guard = sfd.MemoryGuard(10 ** 15, probe=lambda: 1)  # Always below the floor.
    receipt = run(host_url(guard), duration_s=0.4)
    assert {(entry['status'], entry['status_code'], entry['retryable']) for entry in receipt['requests']} == {
        ('host_busy', 503, True)}
    assert receipt['summary']['by_status'] == {'host_busy': 2}
    assert not receipt['outcome']['passed']
    assert any('2 of 2 requests failed' in failure for failure in receipt['outcome']['failures'])


def test_a_credential_the_client_refuses_is_recorded_per_request_without_its_value(host_url):
    receipt = run(host_url(), environment={NAME: TOKEN + ' '}, duration_s=0.4)
    assert {entry['status'] for entry in receipt['requests']} == {'refused_before_request'}
    assert not receipt['outcome']['passed'] and TOKEN not in json.dumps(receipt)


def _fake_host(tmp_path, *, faulthandler=True):
    args = [sys.executable, '-I', *(['-X', 'faulthandler'] if faulthandler else []), '-c', 'import time; time.sleep(60)']
    process = subprocess.Popen(args)
    state = tmp_path / 'state'
    state.mkdir()
    (state / 'state.json').write_text(json.dumps({'host_pid': process.pid, 'gateway_pid': None,
                                                   'memory_floor_bytes': 1234, 'host_port': 1, 'gateway_port': 2}),
                                      encoding='utf-8')
    for name in ('gateway.log', 'host.out.log'):
        (state / name).write_text('before the soak memory free_ram_bytes=1 "memory_sample"\n', encoding='utf-8')
    return process, state


def _writer(state, stop):
    """Append the gateway's and the host's memory lines while the soak runs."""
    while not stop.wait(0.05):
        with (state / 'gateway.log').open('a', encoding='utf-8') as log:
            log.write('2026-11-10T00:00:00.0000000Z memory free_ram_bytes=5 floor_bytes=1234\n')
        with (state / 'host.out.log').open('a', encoding='utf-8') as log:
            log.write('{"memory_sample": {"free_ram_bytes": 5}}\n')


@pytest.mark.parametrize('faulthandler', [True, False])
def test_soak_checks_the_recorded_host_and_counts_the_memory_log_lines(host_url, tmp_path, faulthandler):
    process, state = _fake_host(tmp_path, faulthandler=faulthandler)
    stop = threading.Event()
    writer = threading.Thread(target=_writer, args=(state, stop), daemon=True)
    writer.start()
    try:
        receipt = run(host_url(), state_folder=state)
    finally:
        stop.set()
        writer.join(5)
        _kill_tree(psutil.Process(process.pid))
    host = receipt['host']
    assert (host['host_pid'], host['memory_floor_bytes'], host['faulthandler']) == (process.pid, 1234, faulthandler)
    assert host['alive_throughout'] is True
    lines = receipt['log_lines']
    # Only lines written during the soak count; a slow runner may fit few 50 ms writes into it.
    assert lines['gateway_memory_samples'] >= 1 and lines['host_memory_samples'] >= 1
    assert lines['gateway_memory_low_refusals'] == 0 and lines['host_refusals'] == 0
    assert receipt['outcome']['passed'] is faulthandler
    if not faulthandler:
        assert receipt['outcome']['failures'] == ['the host was not launched with python -X faulthandler']


def test_soak_fails_when_the_host_dies_during_it(host_url, tmp_path):
    process, state = _fake_host(tmp_path)
    killer = threading.Timer(0.25, lambda: _kill_tree(psutil.Process(process.pid)))
    killer.start()
    try:
        receipt = run(host_url(), state_folder=state)
    finally:
        killer.join(5)
    assert receipt['host']['alive_throughout'] is False
    assert any(sample['host_alive'] is False for sample in receipt['memory'])
    assert 'the host process exited during the soak' in receipt['outcome']['failures']


def test_soak_follows_the_recorded_listener_when_the_launcher_exits(host_url, tmp_path):
    """The host is the process that listens: its uv launcher exited while it served in the live soak of
    2026-11-10. The launcher's exit is recorded; only the listener's death fails the soak."""
    listener = subprocess.Popen([sys.executable, '-I', '-X', 'faulthandler', '-c', 'import time; time.sleep(60)'])
    launcher, state = _fake_host(tmp_path, faulthandler=False)  # uv's own command line has no -X faulthandler.
    data = json.loads((state / 'state.json').read_text(encoding='utf-8'))
    (state / 'state.json').write_text(json.dumps(data | {'listener_pid': listener.pid}), encoding='utf-8')
    stop = threading.Event()
    writer = threading.Thread(target=_writer, args=(state, stop), daemon=True)
    writer.start()
    killer = threading.Timer(0.25, lambda: _kill_tree(psutil.Process(launcher.pid)))
    killer.start()
    try:
        receipt = run(host_url(), state_folder=state)
    finally:
        killer.join(5)
        stop.set()
        writer.join(5)
        _kill_tree(psutil.Process(listener.pid))
    host = receipt['host']
    assert (host['host_pid'], host['listener_pid']) == (launcher.pid, listener.pid)
    assert host['faulthandler'] is True and host['alive_throughout'] is True
    assert host['launcher_alive_throughout'] is False
    assert receipt['outcome'] == {'passed': True, 'failures': []}


def test_receipt_is_written_without_the_credential_and_refused_if_it_would_hold_it(host_url, tmp_path):
    receipt = run(host_url(), duration_s=0.2)
    path = soak.write_receipt(tmp_path / 'soak', receipt, TOKEN)
    assert path == tmp_path / 'soak' / 'receipt.json'
    text = path.read_text(encoding='utf-8') + (tmp_path / 'soak' / 'summary.md').read_text(encoding='utf-8')
    assert TOKEN not in text and 'Soak of the full-distribution host (task 6.4)' in text and '—' not in text
    leaky = receipt | {'note': f'bearer {TOKEN}'}
    with pytest.raises(RuntimeError, match='credential'):
        soak.write_receipt(tmp_path / 'leaky', leaky, TOKEN)
    assert not (tmp_path / 'leaky' / 'receipt.json').exists()


def test_the_credential_is_read_by_name_from_the_environment_then_the_env_file(tmp_path):
    env_file = tmp_path / 'test.env'
    env_file.write_text(f'OTHER=1\n{NAME} = "{TOKEN}"\n', encoding='utf-8')
    assert soak.read_credential(NAME, env_file, {}) == (TOKEN, '.env')
    assert soak.read_credential(NAME, env_file, {NAME: 'from-env'}) == ('from-env', 'environment')
    assert soak.read_credential(NAME, tmp_path / 'missing.env', {}) == (None, None)
    env_file.write_text(f'{NAME}=\n', encoding='utf-8')
    assert soak.read_credential(NAME, env_file, {}) == (None, None)


def test_main_refuses_to_start_without_the_credential(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv(NAME, raising=False)
    code = soak.main(['--credential-name', NAME, '--env-file', str(tmp_path / 'missing.env'),
                      '--output', str(tmp_path / 'soak'), '--duration-s', '1'])
    assert code == 2 and NAME in capsys.readouterr().err
    assert not (tmp_path / 'soak').exists()


def test_main_runs_the_soak_and_writes_the_receipt(host_url, tmp_path, monkeypatch):
    monkeypatch.setenv(NAME, TOKEN)
    monkeypatch.setattr(soak, 'MODEL_DIGEST', DIGEST)
    code = soak.main(['--gateway-url', host_url(), '--credential-name', NAME, '--output', str(tmp_path / 'soak'),
                      '--state-dir', str(tmp_path / 'no-state'), '--duration-s', '0.4', '--interval-s', '0.2',
                      '--sample-s', '0.05'])
    receipt = json.loads((tmp_path / 'soak' / 'receipt.json').read_text(encoding='utf-8'))
    # No host state: the soak cannot show the host's faulthandler launch or liveness, so it fails.
    assert code == 1 and receipt['summary']['ok'] == 2
    assert receipt['outcome']['failures'] == ['no host state in ' + str(tmp_path / 'no-state')]
    assert TOKEN not in json.dumps(receipt)


@pytest.mark.skipif(PWSH is None, reason='needs PowerShell 7 (pwsh) on a Windows host')
def test_teardown_runs_the_lifecycle_stop_and_finds_nothing_left(tmp_path):
    if running := _running_hosts():
        pytest.skip(f'a full-distribution host or gateway runs on this machine (pid {running})')
    host_port, port = _free_ports(2)
    result = soak.teardown(state_dir=tmp_path / 'state', host_port=host_port, port=port)
    assert result['exit_code'] == 0 and result['listeners'] == [] and result['processes'] == []
    assert any('Stopped: no host or gateway process remains' in line for line in result['output'])
    assert soak.teardown_failures(result) == []
    assert soak.teardown_failures(result | {'listeners': [{'port': port}]}) != []
    assert Path(result['script']).name == 'serve_full_distribution.ps1'
