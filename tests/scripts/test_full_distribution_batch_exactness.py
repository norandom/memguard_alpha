"""Task 6.4: the exactness check's batch mode, end to end on fakes (no llama-server, no model).

``--check-against-server --batch`` records llama-server's full-vocabulary reference for the
two single pairs plus one prompt carrying two responses, serves every item on the batch
route twice and each pair once on the single route, and compares within 1e-4 with identical
ranks. Here the reference comes from the same fake logits llama-server would see, so a
faithful host passes and a host whose KV cache is not rolled back fails.

Two passes (user correction 2026-11-10: never two models loaded): ``--record`` starts only
llama-server, writes its outputs to a recording file and stops it; ``--compare`` loads only
the host and compares against that file. The tests below show that each pass touches only
its own side and that the comparison refuses a missing, failed, mismatched or altered recording.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from recall_guard.core.full_distribution import BATCH_ENDPOINT_PATH
from scripts.hosts import serve_full_distribution as sfd
from tests.scripts.test_serve_full_distribution import free_port, log_softmax
from tests.scripts.test_serve_full_distribution_batch import CachedFakeLlama


class StaleCacheLlama(CachedFakeLlama):
    """A llama.cpp that accepted a continuation on a cache that was never rolled back."""

    def continue_logits(self, tokens, start):
        self.calls.append(('continue', start, tuple(tokens)))
        yield from self._decode(list(tokens), start, None)

    def truncate(self, n_keep):
        self.calls.append(('truncate', n_keep))


@pytest.fixture
def rig(monkeypatch):
    """Replace llama-server, the model load and the machine checks with fakes; return a setter for the host fake."""
    holder = {'fake': CachedFakeLlama()}

    def record_reference(pairs, log_path):
        log_path.write_text('fake llama-server log\n', encoding='utf-8')  # The real pass opens the server log.
        reference = CachedFakeLlama()  # The same logits llama-server would compute.
        tokens, positions, vectors = [], [], []
        for pair in pairs:
            prompt_ids = reference.tokenize(pair['prompt_text'])
            response_ids = reference.tokenize(pair['response_text'], add_special=False)
            tokens.append({'prompt_ids': prompt_ids, 'response_ids': response_ids})
            sequence, start = prompt_ids + response_ids, len(prompt_ids)
            for t in range(start, len(sequence)):
                vectors.append(log_softmax(reference.row(sequence[:t])) + 3e-4)  # The server's float32 offset.
                positions.append({'pair': pair['name'], 'position': t - start, 't': t, 'target': sequence[t],
                                  'n_entries': reference.n_vocab, 'seconds': 0.0})
        return tokens, positions, vectors, {'command': ['fake-llama-server'], 'after_stop': {'listeners': []}}

    monkeypatch.setattr(sfd, '_preflight', lambda port: {'refusals': []})
    monkeypatch.setattr(sfd, 'build_info', lambda: {'llama_cpp': ['fake b12199'], 'dll': 'fake'})
    monkeypatch.setattr(sfd, '_record_reference', record_reference)
    monkeypatch.setattr(sfd, '_processes', lambda prefixes=('llama-server',): [])
    monkeypatch.setattr(sfd, '_await_free_ram', lambda minimum, seconds=0: {'free_ram_bytes': minimum})
    monkeypatch.setattr(sfd, '_min_free_ram_bytes', lambda: 1)
    monkeypatch.setattr(sfd, '_wait_until_free', lambda ports, seconds=0: [])
    monkeypatch.setattr(sfd, '_other_states', lambda ports: 0)
    monkeypatch.setattr(sfd, 'open_host', lambda: sfd.FullDistributionHost(holder['fake'], model_digest=sfd.MODEL_DIGEST))
    return holder


def _receipt(output):
    return json.loads((output / 'receipt.json').read_text(encoding='utf-8'))


def test_batch_items_are_the_two_single_pairs_plus_one_prompt_with_two_responses():
    items = sfd.BATCH_EXACTNESS_ITEMS
    singles = [{'name': item['name'], 'prompt_text': item['prompt_text'], 'response_text': item['response_texts'][0]}
               for item in items[:2]]
    assert singles == [dict(pair) for pair in sfd.EXACTNESS_PAIRS]
    assert [len(item['response_texts']) for item in items] == [1, 1, 2]
    pairs = sfd.batch_reference_pairs(items)
    assert [pair['name'] for pair in pairs] == [f'{item["name"]}/{index}' for item in items
                                                for index in range(len(item['response_texts']))]
    shared = [pair for pair in pairs if pair['item'] == items[2]['name']]
    assert len({pair['prompt_text'] for pair in shared}) == 1 and len({pair['response_text'] for pair in shared}) == 2


def test_the_batch_check_passes_on_a_faithful_host_with_per_item_timing(rig, tmp_path):
    assert sfd.check_against_server(tmp_path, port=free_port(), batch=True) == 0
    receipt = _receipt(tmp_path)
    assert receipt['outcome'] == {'passed': True, 'refused': False, 'failures': []}
    assert (receipt['task'], receipt['mode']) == ('6.4', 'batch')
    assert receipt['settings']['batch_route'] == BATCH_ENDPOINT_PATH
    assert receipt['settings']['kv_rollback'] == 'llama_memory_seq_rm'
    comparison = receipt['comparison']
    assert comparison['within_tolerance'] and comparison['rank_mismatches'] == 0
    assert all(value < 1e-9 for value in comparison['max_abs_diff'].values())
    pairs = sfd.batch_reference_pairs(sfd.BATCH_EXACTNESS_ITEMS)
    fake = rig['fake']
    lengths = [len(fake.tokenize(pair['response_text'], add_special=False)) for pair in pairs]
    assert comparison['positions_checked'] == sum(lengths)
    assert receipt['evidence_lengths'] == {pair['name']: n for pair, n in zip(pairs, lengths, strict=True)}
    assert receipt['batch_matches_single'] == {pair['name']: True for pair in pairs}
    assert receipt['shared_prompt_ids_identical'] is True
    assert receipt['repeatability']['all_identical']
    batch_timing = receipt['per_item_timing']['batch']
    assert [(entry['run'], entry['item']) for entry in batch_timing] == [
        (run, item['name']) for run in (0, 1) for item in sfd.BATCH_EXACTNESS_ITEMS]
    for entry, item in zip(batch_timing, [*sfd.BATCH_EXACTNESS_ITEMS] * 2, strict=True):
        assert len(entry['responses']) == len(item['response_texts'])
        assert entry['request_seconds'] >= entry['total_seconds'] >= entry['prompt_seconds'] >= 0
    assert [entry['pair'] for entry in receipt['per_item_timing']['single']] == [pair['name'] for pair in pairs]
    summary = (tmp_path / 'summary.md').read_text(encoding='utf-8')
    assert 'batch mode (task 6.4)' in summary and 'Batch route equals the single route' in summary
    assert 'Per-item timing' in summary and '—' not in summary


def test_the_prompt_is_decoded_once_per_item_on_the_batch_route(rig, tmp_path):
    sfd.check_against_server(tmp_path, port=free_port(), batch=True)
    fake = rig['fake']
    shared = sfd.BATCH_EXACTNESS_ITEMS[2]
    prompt = tuple(fake.tokenize(shared['prompt_text']))
    decodes = [call for call in fake.calls if call[0] == 'decode' and call[1] == prompt]
    rollbacks = [call for call in fake.calls if call == ('truncate', len(prompt))]
    # Two batch runs and two row passes decode the shared prompt once each; the two single-route
    # requests decode it with their responses (longer sequences, not counted here).
    assert len(decodes) == 4 and len(rollbacks) == 4 * len(shared['response_texts'])


def test_a_host_that_does_not_roll_back_fails_the_batch_check(rig, tmp_path):
    rig['fake'] = StaleCacheLlama()
    assert sfd.check_against_server(tmp_path, port=free_port(), batch=True) == 1
    receipt = _receipt(tmp_path)
    failures = receipt['outcome']['failures']
    assert not receipt['outcome']['passed']
    assert any('beyond' in failure for failure in failures)
    assert any('differs from the single route' in failure for failure in failures)


def test_a_cache_that_refuses_a_stale_continuation_fails_the_batch_check(rig, tmp_path):
    rig['fake'] = CachedFakeLlama(rollback=False)
    assert sfd.check_against_server(tmp_path, port=free_port(), batch=True) == 1
    assert any('HTTP 500' in failure for failure in _receipt(tmp_path)['outcome']['failures'])


def test_the_single_check_is_unchanged(rig, tmp_path):
    assert sfd.check_against_server(tmp_path, port=free_port()) == 0
    receipt = _receipt(tmp_path)
    assert receipt['task'] == '6.1' and 'mode' not in receipt and 'batch_matches_single' not in receipt
    assert [pair['name'] for pair in receipt['pairs']] == [pair['name'] for pair in sfd.EXACTNESS_PAIRS]
    assert receipt['comparison']['within_tolerance'] and receipt['repeatability']['all_identical']
    assert (tmp_path / 'summary.md').read_text(encoding='utf-8').startswith(
        '# Full-distribution host exactness check (task 6.1)')


def test_main_routes_batch_to_its_own_receipt_folder(monkeypatch):
    seen = []
    monkeypatch.setattr(sfd, 'check_against_server', lambda output, **kwargs: seen.append((output, kwargs)) or 0)
    assert sfd.main(['--check-against-server', '--batch']) == 0
    assert sfd.main(['--check-against-server']) == 0
    assert seen[0][0] == sfd.BATCH_RECEIPT_DIR == sfd.RECEIPT_DIR / 'batch' and seen[0][1]['batch'] is True
    assert seen[1][0] == sfd.RECEIPT_DIR and seen[1][1]['batch'] is False
    with pytest.raises(SystemExit):
        sfd.main(['--batch'])  # Batch is a mode of the exactness check only.


# --- two passes: record (llama-server only), then compare (host only) -----------------------------


def _forbidden(what):
    def call(*args, **kwargs):
        raise AssertionError(what)
    return call


def _recording(output):
    return json.loads((output / sfd.RECORDING_NAME).read_text(encoding='utf-8'))


def test_the_record_pass_writes_the_reference_and_never_loads_the_host(rig, tmp_path, monkeypatch):
    monkeypatch.setattr(sfd, 'open_host', _forbidden('the record pass must not load the host'))
    assert sfd.check_against_server(tmp_path, port=free_port(), batch=True, mode='record') == 0
    recording = _recording(tmp_path)
    assert recording['schema'] == sfd.RECORDING_SCHEMA and recording['mode'] == 'batch'
    assert recording['outcome'] == {'passed': True, 'refused': False, 'failures': []}
    pairs = sfd.batch_reference_pairs(sfd.BATCH_EXACTNESS_ITEMS)
    assert [pair['name'] for pair in recording['pairs']] == [pair['name'] for pair in pairs]
    assert len(recording['tokens']) == len(pairs)
    fake = rig['fake']
    positions = sum(len(fake.tokenize(pair['response_text'], add_special=False)) for pair in pairs)
    assert len(recording['positions']) == positions
    vectors = recording['vectors']
    data = (tmp_path / vectors['file']).read_bytes()
    assert vectors['sha256'] == hashlib.sha256(data).hexdigest() and vectors['bytes'] == len(data)
    assert vectors['shape'] == [positions, fake.n_vocab]
    assert recording['reference_lifecycle']['after_stop'] == {'listeners': []}
    assert recording['teardown']['listeners'] == [] and recording['teardown']['llama_server_processes'] == []
    assert not (tmp_path / 'receipt.json').exists()  # The comparison receipt is pass B's.


def test_the_compare_pass_reads_the_recording_and_never_starts_llama_server(rig, tmp_path, monkeypatch):
    assert sfd.check_against_server(tmp_path, port=free_port(), batch=True, mode='record') == 0
    monkeypatch.setattr(sfd, '_record_reference', _forbidden('the compare pass must not start llama-server'))
    assert sfd.check_against_server(tmp_path, port=free_port(), batch=True, mode='compare') == 0
    receipt = _receipt(tmp_path)
    assert receipt['outcome'] == {'passed': True, 'refused': False, 'failures': []}
    assert receipt['mode'] == 'batch' and receipt['passes'] == 'two'
    reference = receipt['reference_recording']
    assert reference['file'] == sfd.RECORDING_NAME and reference['outcome']['passed']
    assert reference['vectors'] == _recording(tmp_path)['vectors']
    assert receipt['reference_lifecycle'] == _recording(tmp_path)['reference_lifecycle']
    comparison = receipt['comparison']
    assert comparison['within_tolerance'] and comparison['rank_mismatches'] == 0
    assert all(value < 1e-9 for value in comparison['max_abs_diff'].values())
    assert receipt['batch_matches_single'] and all(receipt['batch_matches_single'].values())
    assert len(receipt['per_item_timing']['batch']) == 2 * len(sfd.BATCH_EXACTNESS_ITEMS)
    assert 'recorded by a separate pass' in (tmp_path / 'summary.md').read_text(encoding='utf-8')


def test_the_record_pass_creates_its_output_folder(rig, tmp_path):
    output = tmp_path / 'exactness' / 'batch'  # The default folder does not exist before the first record pass.
    assert sfd.check_against_server(output, port=free_port(), batch=True, mode='record') == 0
    assert _recording(output)['outcome']['passed'] and (output / 'reference-llama-server.log').is_file()


def test_the_compare_pass_needs_a_recording(rig, tmp_path, monkeypatch):
    monkeypatch.setattr(sfd, 'open_host', _forbidden('no host load without a recording'))
    assert sfd.check_against_server(tmp_path, port=free_port(), batch=True, mode='compare') == 3
    receipt = _receipt(tmp_path)
    assert receipt['outcome']['refused'] and not receipt['outcome']['passed']
    assert any('recording' in failure for failure in receipt['outcome']['failures'])


def test_the_compare_pass_refuses_a_recording_of_other_pairs(rig, tmp_path, monkeypatch):
    assert sfd.check_against_server(tmp_path, port=free_port(), mode='record') == 0  # The single-route pairs.
    monkeypatch.setattr(sfd, 'open_host', _forbidden('no host load for a mismatched recording'))
    assert sfd.check_against_server(tmp_path, port=free_port(), batch=True, mode='compare') == 3
    assert any('recording' in failure for failure in _receipt(tmp_path)['outcome']['failures'])


def test_the_compare_pass_refuses_an_altered_vector_file(rig, tmp_path, monkeypatch):
    assert sfd.check_against_server(tmp_path, port=free_port(), batch=True, mode='record') == 0
    vectors = tmp_path / _recording(tmp_path)['vectors']['file']
    vectors.write_bytes(vectors.read_bytes() + b'\0')
    monkeypatch.setattr(sfd, 'open_host', _forbidden('no host load for an altered recording'))
    assert sfd.check_against_server(tmp_path, port=free_port(), batch=True, mode='compare') == 3
    assert any('sha256' in failure for failure in _receipt(tmp_path)['outcome']['failures'])


def test_a_reference_server_that_does_not_stop_fails_the_record_pass_and_blocks_the_compare(
        rig, tmp_path, monkeypatch):
    original = sfd._record_reference

    def left_running(pairs, log_path):
        tokens, positions, vectors, lifecycle = original(pairs, log_path)
        return tokens, positions, vectors, lifecycle | {'after_stop': {'listeners': [{'port': 11560}]}}

    monkeypatch.setattr(sfd, '_record_reference', left_running)
    assert sfd.check_against_server(tmp_path, port=free_port(), batch=True, mode='record') == 1
    recording = _recording(tmp_path)
    assert not recording['outcome']['passed']
    assert any('did not stop cleanly' in failure for failure in recording['outcome']['failures'])
    monkeypatch.setattr(sfd, 'open_host', _forbidden('no host load after a failed record pass'))
    assert sfd.check_against_server(tmp_path, port=free_port(), batch=True, mode='compare') == 3


def test_the_compare_pass_refuses_while_a_llama_server_is_resident(rig, tmp_path, monkeypatch):
    assert sfd.check_against_server(tmp_path, port=free_port(), batch=True, mode='record') == 0
    monkeypatch.setattr(sfd, '_preflight', lambda port: {'refusals': ['a llama-server process is running']})
    monkeypatch.setattr(sfd, 'open_host', _forbidden('no host load beside a resident llama-server'))
    assert sfd.check_against_server(tmp_path, port=free_port(), batch=True, mode='compare') == 3
    assert _receipt(tmp_path)['outcome']['failures'] == ['preflight: a llama-server process is running']


def test_a_host_that_does_not_roll_back_fails_the_two_pass_check(rig, tmp_path):
    assert sfd.check_against_server(tmp_path, port=free_port(), batch=True, mode='record') == 0
    rig['fake'] = StaleCacheLlama()
    assert sfd.check_against_server(tmp_path, port=free_port(), batch=True, mode='compare') == 1
    failures = _receipt(tmp_path)['outcome']['failures']
    assert any('beyond' in failure for failure in failures)
    assert any('differs from the single route' in failure for failure in failures)


@pytest.mark.parametrize('narrow', [True, False])
def test_a_recording_keeps_the_reference_vectors_exactly(tmp_path, narrow):
    rng = np.random.default_rng(7)
    vectors = [rng.normal(-12.0, 3.0, 64) for _ in range(3)]
    if narrow:  # llama-server's logprobs are float32 values: stored as float32, without loss.
        vectors = [vector.astype(np.float32).astype(np.float64) for vector in vectors]
    vectors[0][-1] = np.nan  # An id beyond the vocabulary is absent from the server's entries.
    header = sfd.write_recording(tmp_path, {'schema': sfd.RECORDING_SCHEMA}, vectors)
    assert header['vectors']['dtype'] == ('float32' if narrow else 'float64')
    loaded = sfd.load_recorded_vectors(tmp_path, header)
    assert len(loaded) == 3 and all(a.dtype == np.float64 for a in loaded)
    assert all(np.array_equal(a, b, equal_nan=True) for a, b in zip(loaded, vectors, strict=True))


def test_main_routes_the_record_and_compare_passes(monkeypatch):
    seen = []
    monkeypatch.setattr(sfd, 'check_against_server', lambda output, **kwargs: seen.append((output, kwargs)) or 0)
    assert sfd.main(['--check-against-server', '--batch', '--record']) == 0
    assert sfd.main(['--check-against-server', '--batch', '--compare']) == 0
    assert sfd.main(['--check-against-server', '--batch']) == 0
    assert [kwargs['mode'] for _, kwargs in seen] == ['record', 'compare', None]
    assert all(output == sfd.BATCH_RECEIPT_DIR and kwargs['batch'] for output, kwargs in seen)
    for argv in (['--record'], ['--compare'], ['--check-against-server', '--record', '--compare']):
        with pytest.raises(SystemExit):
            sfd.main(argv)


def test_compare_position_still_detects_a_rank_change():
    row = np.random.default_rng(3).normal(0.0, 3.0, 300).astype(np.float32)
    server = log_softmax(row)
    target = int(np.argsort(-server)[1])
    server[int(np.argmax(server))] -= 10.0  # The server ranks the target first; the host ranks it second.
    result = sfd.compare_position(row, server, target, sfd.position_evidence(row, target, 10))
    assert result['rank'] == 2 and result['ref_rank'] == 1
    assert not sfd.summarise([result])['within_tolerance']
