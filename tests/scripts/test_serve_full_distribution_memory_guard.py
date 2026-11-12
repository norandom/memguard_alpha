"""Task 6.4: the host's memory guard, with fakes (no model load, no GPU).

The host samples free physical memory every 5 s and refuses new requests with 503
``{"error": {"code": "memory_low"}}`` while it is below the profile's ``memory_floor_bytes``;
every sample and refusal is logged as one JSON line. The client maps 503 to the retryable
``host_busy``. Memory readings come from a fake probe, never from this machine.
"""
from __future__ import annotations

import json
import threading
import time

import pytest

from recall_guard.core.full_distribution import BATCH_ENDPOINT_PATH, ENDPOINT_PATH
from recall_guard.core.profiles import load_profile
from recall_guard.core.request_control import RequestError
from scripts.hosts import serve_full_distribution as sfd
from tests.scripts.test_serve_full_distribution import (
    DIGEST,
    FakeLlama,
    client_for,
    listening,
    post_raw,
    request,
    serve,  # noqa: F401  (pytest fixture)
)

FLOOR = 1000


class Probe:
    """Free-memory readings in order; the last one repeats. ``fail`` raises once first."""

    def __init__(self, *values, fail=False):
        self.values, self.fail, self.calls = list(values), fail, 0

    def __call__(self):
        self.calls += 1
        if self.fail:
            self.fail = False
            raise OSError('reading failed')
        return self.values.pop(0) if len(self.values) > 1 else self.values[0]


def guarded(fake, *readings):
    records = []
    guard = sfd.MemoryGuard(FLOOR, probe=Probe(*readings), log=records.append)
    return sfd.FullDistributionHost(fake, model_digest=DIGEST, memory_guard=guard), guard, records


def batch_body():
    return {'model_digest': DIGEST, 'prompt_text': 'Rate the month.', 'response_texts': [' Neutral.'], 'top_k_keep': 5}


def _wait(condition, seconds=10.0):
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, 'timed out'
        time.sleep(0.01)


def test_below_the_floor_both_routes_answer_503_memory_low_and_nothing_is_decoded(serve):  # noqa: F811
    fake = FakeLlama()
    host, guard, records = guarded(fake, FLOOR - 1)
    url = serve(host)
    client = client_for(url)
    with pytest.raises(RequestError) as single:
        client.evidence('Rate the month.', ' Neutral.')
    with pytest.raises(RequestError) as batch:
        client.evidence_batch('Rate the month.', [' Neutral.', ' Positive.'])
    for caught in (single, batch):
        assert (caught.value.status_code, caught.value.code, caught.value.retryable) == (503, 'host_busy', True)
    for path, body in ((ENDPOINT_PATH, request()), (BATCH_ENDPOINT_PATH, batch_body())):
        status, payload = post_raw(url, json.dumps(body).encode(), path=path)
        assert status == 503 and set(payload) == {'error'} and payload['error']['code'] == 'memory_low'
        assert str(FLOOR) in payload['error']['message']
    assert fake.decodes == []
    refusals = [record['refused'] for record in records if 'refused' in record]
    assert len(refusals) == 4 and guard.refusals == 4
    assert all(item['code'] == 'memory_low' and item['free_ram_bytes'] == FLOOR - 1 and item['floor_bytes'] == FLOOR
               for item in refusals)
    samples = [record['memory_sample'] for record in records if 'memory_sample' in record]
    assert len(samples) == 4 and all(item['below_floor'] and item['reason'] == 'request' for item in samples)


def test_at_the_floor_requests_are_served_and_a_recovery_is_admitted():
    fake = FakeLlama()
    host, guard, records = guarded(fake, FLOOR - 1, FLOOR, 10 * FLOOR)
    statuses = [host.respond(json.dumps(request()).encode())[0] for _ in range(3)]
    assert statuses == [503, 200, 200]
    assert len(fake.decodes) == 2 and guard.refusals == 1
    assert [record['memory_sample']['below_floor'] for record in records if 'memory_sample' in record] == [
        True, False, False]


def test_a_malformed_body_is_still_refused_while_memory_is_low():
    host, _, _ = guarded(FakeLlama(), 0)
    assert host.respond(b'{not json')[0] == 503  # Refused before any parsing or model work.


def test_the_guard_samples_in_the_background_until_stopped():
    records = []
    guard = sfd.MemoryGuard(FLOOR, probe=Probe(2 * FLOOR), interval_s=0.01, log=records.append)
    guard.start()
    try:
        _wait(lambda: guard.samples >= 3)
    finally:
        guard.stop()
    assert not guard.running
    count = guard.samples
    time.sleep(0.1)
    assert guard.samples == count  # Nothing samples after stop.
    sample = records[0]['memory_sample']
    assert set(sample) == {'at', 'free_ram_bytes', 'floor_bytes', 'below_floor', 'reason'}
    assert (sample['free_ram_bytes'], sample['floor_bytes'], sample['below_floor'], sample['reason']) == (
        2 * FLOOR, FLOOR, False, 'periodic')
    assert guard.last['free_ram_bytes'] == 2 * FLOOR


def test_a_failing_reading_is_logged_and_sampling_continues():
    records = []
    guard = sfd.MemoryGuard(FLOOR, probe=Probe(2 * FLOOR, fail=True), interval_s=0.01, log=records.append)
    guard.start()
    try:
        _wait(lambda: guard.samples >= 2)
    finally:
        guard.stop()
    assert 'OSError' in records[0]['memory_sample_failed']
    assert any('memory_sample' in record for record in records[1:])


@pytest.mark.parametrize('floor', [0, -1, True, 1.5, None, '1000'])
def test_the_guard_refuses_a_floor_that_is_not_a_positive_integer(floor):
    with pytest.raises((TypeError, ValueError)):
        sfd.MemoryGuard(floor, probe=Probe(1))


def test_the_default_floor_is_the_profiles():
    floor = load_profile('llamacpp-gemma3-4b').host_requirements.memory_floor()
    assert sfd._memory_floor_bytes() == floor == 1_610_612_736
    assert sfd.MEMORY_SAMPLE_S == 5.0


def test_main_passes_the_memory_floor_to_serve(monkeypatch):
    seen = []
    monkeypatch.setattr(sfd, 'serve', lambda port, **kwargs: seen.append((port, kwargs)) or 0)
    assert sfd.main(['--port', '1234']) == 0
    assert sfd.main(['--port', '1234', '--memory-floor-bytes', '5']) == 0
    assert seen == [(1234, {'memory_floor_bytes': None}), (1234, {'memory_floor_bytes': 5})]
    for value in ('0', '-3'):
        with pytest.raises(SystemExit):
            sfd.main(['--memory-floor-bytes', value])


def _guard_threads():
    return [thread for thread in threading.enumerate() if thread.name == 'memory-guard' and thread.is_alive()]


def test_serve_samples_memory_reports_the_floor_and_refuses_below_it(monkeypatch, capsys):
    servers = []
    original = sfd.make_server

    def capture(*args, **kwargs):
        servers.append(original(*args, **kwargs))
        return servers[-1]

    fake = FakeLlama()
    monkeypatch.setattr(sfd, 'make_server', capture)
    monkeypatch.setattr(sfd, 'open_host', lambda: sfd.FullDistributionHost(fake, model_digest=DIGEST))
    monkeypatch.setattr(sfd, '_free_ram', lambda: FLOOR - 1)
    result = []
    thread = threading.Thread(target=lambda: result.append(sfd.serve(0, memory_floor_bytes=FLOOR)), daemon=True)
    thread.start()
    try:
        _wait(lambda: servers and listening(servers[0].server_address[1]))
        url = f'http://127.0.0.1:{servers[0].server_address[1]}'
        with pytest.raises(RequestError) as caught:
            client_for(url).evidence_batch('Rate the month.', [' Neutral.'])
    finally:
        if servers:
            servers[0].shutdown()
        thread.join(30)
    assert caught.value.code == 'host_busy'
    assert result == [0] and fake.closed and fake.decodes == []
    assert _guard_threads() == []
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith('{')]
    [ready] = [line for line in lines if 'listening' in line]
    assert ready['memory_floor_bytes'] == FLOOR and ready['memory_sample_seconds'] == sfd.MEMORY_SAMPLE_S
    assert ready['batch_endpoint'] == BATCH_ENDPOINT_PATH and isinstance(ready['faulthandler'], bool)
    assert any('memory_sample' in line for line in lines)
    assert any(line.get('refused', {}).get('code') == 'memory_low' for line in lines)


def test_serve_stops_the_guard_when_the_start_is_refused(monkeypatch, capsys):
    def refuse():
        raise sfd.StartRefused('digest_mismatch: test')

    monkeypatch.setattr(sfd, 'open_host', refuse)
    monkeypatch.setattr(sfd, '_free_ram', lambda: 10 * FLOOR)
    assert sfd.serve(0, memory_floor_bytes=FLOOR) == 2
    assert _guard_threads() == []
    assert 'digest_mismatch' in capsys.readouterr().err
