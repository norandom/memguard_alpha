"""Task 6.4: the batch route of the full-distribution host, with fakes (no model load, no GPU).

``POST /v1/full_distribution_batch`` decodes the prompt once, teacher-forces each response
against the cached prompt and rolls the KV cache back to the prompt length after each
response, so every evidence object equals the single route's for that pair. The fake
``Llama`` here models the KV cache; the ctypes wrapper's rollback is checked against a fake
``llama.dll`` that records its calls.
"""
from __future__ import annotations

import ctypes
import json
import threading

import numpy as np
import pytest

from recall_guard.core.full_distribution import BATCH_ENDPOINT_PATH, ENDPOINT_PATH
from recall_guard.core.request_control import RequestError
from scripts.hosts import serve_full_distribution as sfd
from scripts.hosts.llamacpp_native import N_CTX, Llama
from tests.scripts.test_serve_full_distribution import (
    DIGEST,
    FakeLlama,
    brute_force,
    client_for,
    post_raw,
    request,
    serve,  # noqa: F401  (pytest fixture)
)

PROMPT = 'Rate the month.'
RESPONSES = [' Neutral, then positive.', 'x', ' Negative.']  # Longer, one-token and shorter responses.


class CachedFakeLlama(FakeLlama):
    """FakeLlama with a modelled KV cache of sequence 0.

    ``incremental_logits`` clears the cache and decodes from position 0; ``continue_logits``
    decodes on top of the cache and refuses a start other than the cached length;
    ``truncate`` rolls the cache back. Each row depends on the whole cache content, so a
    stale cache would change the evidence. ``calls`` records the order of operations.
    """

    def __init__(self, *args, rollback=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache, self.calls, self.rollback = [], [], rollback

    def incremental_logits(self, tokens, rows=None):
        tokens = list(tokens)
        self.decodes.append(tuple(tokens))
        self.calls.append(('decode', tuple(tokens)))
        self.cache = []
        yield from self._decode(tokens, 0, None if rows is None else set(rows))

    def continue_logits(self, tokens, start):
        tokens = list(tokens)
        self.calls.append(('continue', start, tuple(tokens)))
        if start != len(self.cache):
            raise RuntimeError(f'the cache holds {len(self.cache)} positions, not {start}')
        yield from self._decode(tokens, start, None)

    def _decode(self, tokens, start, wanted):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            for i, token in enumerate(tokens, start):
                if self.delay:
                    threading.Event().wait(self.delay)
                self.cache.append(token)
                if wanted is None or i in wanted:
                    yield i, self.row(self.cache)
        finally:
            with self._lock:
                self.active -= 1

    def truncate(self, n_keep):
        self.calls.append(('truncate', n_keep))
        if self.rollback:
            del self.cache[n_keep:]

    def cached_length(self):
        return len(self.cache)


def batch_request(prompt=PROMPT, responses=None, top_k_keep=100, digest=DIGEST):
    return {'model_digest': digest, 'prompt_text': prompt,
            'response_texts': list(RESPONSES if responses is None else responses), 'top_k_keep': top_k_keep}


# ---------- batch assembly ----------

def test_batch_returns_one_evidence_per_response_in_order_equal_to_the_single_route():
    fake = CachedFakeLlama()
    host = sfd.FullDistributionHost(fake, model_digest=DIGEST)
    payload = host.evidence_batch(batch_request())
    assert tuple(payload) == ('evidence',) and len(payload['evidence']) == len(RESPONSES)
    assert payload['evidence'] == [brute_force(fake, PROMPT, response, 100) for response in RESPONSES]
    assert payload['evidence'] == [host.evidence(request(PROMPT, response)) for response in RESPONSES]


def test_prompt_is_decoded_once_and_the_cache_rolls_back_to_the_prompt_after_each_response():
    fake = CachedFakeLlama()
    sfd.FullDistributionHost(fake, model_digest=DIGEST).evidence_batch(batch_request())
    prompt_ids = fake.tokenize(PROMPT)
    responses = [fake.tokenize(text, add_special=False) for text in RESPONSES]
    n = len(prompt_ids)
    expected = [('decode', tuple(prompt_ids))]
    for ids in responses:
        if len(ids) > 1:  # A one-token response is scored from the prompt's last row alone.
            expected.append(('continue', n, tuple(ids[:-1])))
        expected.append(('truncate', n))
    assert fake.calls == expected
    assert fake.cache == prompt_ids  # Rolled back after the last response too.


def test_a_cache_that_was_not_rolled_back_is_refused_not_scored():
    host = sfd.FullDistributionHost(CachedFakeLlama(rollback=False), model_digest=DIGEST)
    status, payload = host.respond(json.dumps(batch_request()).encode(), batch=True)
    assert status == 500 and payload['error']['code'] == 'internal_error'


def test_kept_count_is_uniform_and_a_smaller_k_is_a_prefix_in_every_response():
    fake = CachedFakeLlama(ties=True)
    host = sfd.FullDistributionHost(fake, model_digest=DIGEST)
    wide = host.evidence_batch(batch_request(top_k_keep=100))['evidence']
    narrow = host.evidence_batch(batch_request(top_k_keep=7))['evidence']
    for wide_item, narrow_item in zip(wide, narrow, strict=True):
        assert {len(position) for position in narrow_item['top_logprobs']} == {7}
        assert narrow_item['top_logprobs'] == [position[:7] for position in wide_item['top_logprobs']]


def test_batch_records_per_item_timings():
    fake = CachedFakeLlama()
    host = sfd.FullDistributionHost(fake, model_digest=DIGEST)
    host.evidence_batch(batch_request())
    timings = host.last_batch_timings
    lengths = [len(fake.tokenize(text, add_special=False)) for text in RESPONSES]
    assert timings['prompt_ids'] == len(fake.tokenize(PROMPT))
    assert [item['response_ids'] for item in timings['responses']] == lengths
    assert [item['decodes'] for item in timings['responses']] == [length - 1 for length in lengths]
    seconds = [timings['prompt_seconds'], timings['total_seconds'], *(item['seconds'] for item in timings['responses'])]
    assert all(isinstance(value, float) and value >= 0 for value in seconds)
    assert timings['total_seconds'] >= timings['prompt_seconds'] + sum(item['seconds'] for item in timings['responses'])


def test_batch_logit_rows_follow_the_batch_computation():
    fake = CachedFakeLlama()
    host = sfd.FullDistributionHost(fake, model_digest=DIGEST)
    prompt_ids = fake.tokenize(PROMPT)
    responses = [fake.tokenize(text, add_special=False) for text in RESPONSES]
    rows = host.batch_logit_rows(prompt_ids, responses)
    for ids, block in zip(responses, rows, strict=True):
        sequence = prompt_ids + ids
        expected = np.stack([fake.row(sequence[:i + 1]) for i in range(len(prompt_ids) - 1, len(sequence) - 1)])
        assert block.shape == (len(ids), fake.n_vocab) and np.array_equal(block, expected)
    assert fake.cache == prompt_ids


# ---------- refusals ----------

def test_413_applies_to_the_prompt_plus_the_longest_response(serve):  # noqa: F811
    fake = CachedFakeLlama()
    url = serve(sfd.FullDistributionHost(fake, model_digest=DIGEST))
    client = client_for(url)
    prompt = 'a' * (N_CTX - 3)  # BOS + 2045 prompt ids.
    assert len(client.evidence_batch(prompt, ['bb', 'b'])) == 2  # 2045 + 1 + 2 = 2048 fits.
    calls = len(fake.calls)
    with pytest.raises(RequestError) as caught:
        client.evidence_batch(prompt, ['b', 'bbb'])
    assert (caught.value.status_code, caught.value.code) == (413, 'oversize')
    assert 'longest response' in caught.value.detail
    assert len(fake.calls) == calls  # Refused before any decode.


@pytest.mark.parametrize('responses', [[' fine', ''], [' fine', 'has FAIL in it'], ['\ud800']])
def test_a_response_that_fails_to_tokenise_is_422_before_any_decode(responses):
    fake = CachedFakeLlama(fail_marker='FAIL')
    host = sfd.FullDistributionHost(fake, model_digest=DIGEST)
    status, payload = host.respond(json.dumps(batch_request(responses=responses)).encode(), batch=True)
    assert status == 422 and payload['error']['code'] == 'tokenisation_failed'
    assert fake.calls == []


def test_a_prompt_that_fails_to_tokenise_is_422():
    fake = CachedFakeLlama(fail_marker='FAIL')
    host = sfd.FullDistributionHost(fake, model_digest=DIGEST)
    status, payload = host.respond(json.dumps(batch_request(prompt='FAIL')).encode(), batch=True)
    assert status == 422 and payload['error']['code'] == 'tokenisation_failed' and fake.calls == []


def test_a_different_model_digest_is_409(serve):  # noqa: F811
    url = serve(sfd.FullDistributionHost(CachedFakeLlama(), model_digest=DIGEST))
    with pytest.raises(RequestError) as caught:
        client_for(url, digest='sha256:' + 'cd' * 32).evidence_batch(PROMPT, RESPONSES)
    assert (caught.value.status_code, caught.value.code) == (409, 'digest_mismatch')


@pytest.mark.parametrize('body', [
    json.dumps({key: value for key, value in batch_request().items() if key != 'response_texts'}).encode(),
    json.dumps(batch_request() | {'response_texts': ' Neutral.'}).encode(),
    json.dumps(batch_request() | {'response_texts': []}).encode(),
    json.dumps(batch_request() | {'response_texts': [' Neutral.', 7]}).encode(),
    json.dumps(batch_request() | {'response_texts': [None]}).encode(),
    json.dumps(batch_request() | {'response_text': ' Neutral.'}).encode(),
    json.dumps(batch_request(top_k_keep=0)).encode(),
    json.dumps(batch_request(top_k_keep=True)).encode(),
    json.dumps(batch_request() | {'model_digest': 7}).encode(),
    json.dumps(batch_request() | {'prompt_text': None}).encode(),
    json.dumps(request()).encode(),  # The single route's body on the batch route.
    b'{"model_digest": "a", "model_digest": "b", "prompt_text": "", "response_texts": ["x"], "top_k_keep": 1}',
    b'[1, 2]',
    b'{not json',
], ids=['missing-responses', 'responses-text', 'responses-empty', 'response-number', 'response-null',
        'unknown-field', 'top-k-zero', 'top-k-bool', 'digest-number', 'prompt-null', 'single-body', 'duplicate-key',
        'not-an-object', 'invalid-json'])
def test_a_malformed_batch_body_is_400_before_any_decode(serve, body):  # noqa: F811
    fake = CachedFakeLlama()
    url = serve(sfd.FullDistributionHost(fake, model_digest=DIGEST))
    status, payload = post_raw(url, body, path=BATCH_ENDPOINT_PATH)
    assert status == 400 and payload['error']['code'] == 'malformed'
    assert fake.calls == []


# ---------- serving ----------

def test_both_routes_are_served_and_other_paths_are_404(serve):  # noqa: F811
    fake = CachedFakeLlama()
    url = serve(sfd.FullDistributionHost(fake, model_digest=DIGEST))
    client = client_for(url)
    batch = client.evidence_batch(PROMPT, RESPONSES)
    assert [item.to_dict() for item in batch] == [brute_force(fake, PROMPT, response, 100) for response in RESPONSES]
    assert client.evidence(PROMPT, RESPONSES[0]).to_dict() == batch[0].to_dict()
    for path in (BATCH_ENDPOINT_PATH + '/', ENDPOINT_PATH + '_batches', '/v1/chat/completions'):
        status, payload = post_raw(url, json.dumps(batch_request()).encode(), path=path)
        assert status == 404 and payload['error']['code'] == 'not_found'


def test_batch_requests_are_served_one_at_a_time(serve):  # noqa: F811
    fake = CachedFakeLlama(delay=0.002)
    client = client_for(serve(sfd.FullDistributionHost(fake, model_digest=DIGEST)))
    results, errors = [], []

    def call():
        try:
            results.append(client.evidence_batch('p' * 20, [' r' * 5, ' s' * 4]))
        except Exception as error:  # Collected and asserted below.
            errors.append(error)

    threads = [threading.Thread(target=call) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    assert errors == [] and len(results) == 3 and fake.max_active == 1
    assert len({json.dumps([item.to_dict() for item in result]) for result in results}) == 1


# ---------- the ctypes wrapper: continue on the cached prompt, roll back with llama_memory_seq_rm ----------

class FakeBatch:
    def __init__(self):
        self.n_tokens, self.token, self.pos, self.n_seq_id, self.seq_id, self.logits = 0, [0], [0], [0], [[0]], [0]


class FakeLib:
    """The llama.dll calls the wrapper makes, on a KV cache of sequence 0 held as a list of positions."""

    def __init__(self, n_vocab=6):
        self.n_vocab, self.calls, self.cache = n_vocab, [], []
        self.refuse_rm = self.ignore_rm = False
        self._logits = (ctypes.c_float * n_vocab)()

    def llama_get_memory(self, ctx):
        return 'memory'

    def llama_memory_clear(self, memory, data):
        self.calls.append(('clear',))
        self.cache = []

    def llama_memory_seq_pos_max(self, memory, seq_id):
        return max(self.cache, default=-1)

    def llama_memory_seq_rm(self, memory, seq_id, p0, p1):
        self.calls.append(('seq_rm', memory, seq_id, p0, p1))
        if self.refuse_rm:
            return False
        if not self.ignore_rm:
            self.cache = [p for p in self.cache if p < p0 or 0 <= p1 <= p]
        return True

    def llama_batch_init(self, n_tokens, embd, n_seq_max):
        return FakeBatch()

    def llama_batch_free(self, batch):
        pass

    def llama_decode(self, ctx, batch):
        token, position = batch.token[0], batch.pos[0]
        self.calls.append(('decode', token, position, batch.seq_id[0][0], batch.logits[0]))
        self.cache.append(position)
        for k in range(self.n_vocab):
            self._logits[k] = float(100 * token + position + k)
        return 0

    def llama_get_logits_ith(self, ctx, index):
        return ctypes.cast(self._logits, ctypes.POINTER(ctypes.c_float))


def fake_llama() -> Llama:
    llama = object.__new__(Llama)  # No DLL, no model: only the methods under test.
    llama.lib, llama.ctx, llama.n_vocab = FakeLib(), 'ctx', 6
    return llama


def test_continue_logits_extends_the_cached_prompt_one_decode_at_a_time_without_clearing():
    llama = fake_llama()
    assert [position for position, _ in llama.incremental_logits([7, 8, 9], rows=[2])] == [2]
    assert llama.cached_length() == 3
    llama.lib.calls.clear()
    rows = list(llama.continue_logits([4, 5], start=3))
    assert [position for position, _ in rows] == [3, 4]
    assert llama.lib.calls == [('decode', 4, 3, 0, 1), ('decode', 5, 4, 0, 1)]  # No clear, sequence 0.
    assert [row.tolist() for _, row in rows] == [[403.0 + k for k in range(6)], [504.0 + k for k in range(6)]]
    assert llama.cached_length() == 5


def test_continue_logits_refuses_a_start_other_than_the_cached_length_and_the_context():
    llama = fake_llama()
    list(llama.incremental_logits([7, 8, 9]))
    with pytest.raises(RuntimeError, match='holds 3 positions'):
        list(llama.continue_logits([4], start=2))
    with pytest.raises(ValueError, match='context'):
        list(llama.continue_logits([4] * (N_CTX - 2), start=3))
    assert not any(call[0] == 'decode' and call[2] >= 3 for call in llama.lib.calls)


def test_truncate_rolls_back_with_llama_memory_seq_rm_and_verifies_the_cached_length():
    llama = fake_llama()
    list(llama.incremental_logits([7, 8, 9]))
    list(llama.continue_logits([4, 5], start=3))
    llama.truncate(3)
    assert llama.lib.calls[-1] == ('seq_rm', 'memory', 0, 3, -1)  # Sequence 0, from the prompt length to the end.
    assert llama.cached_length() == 3
    list(llama.continue_logits([6], start=3))  # The next response continues on the prompt again.
    llama.lib.ignore_rm = True
    with pytest.raises(RuntimeError, match='holds 4 positions'):
        llama.truncate(3)
    llama.lib.refuse_rm = True
    with pytest.raises(RuntimeError, match='llama_memory_seq_rm'):
        llama.truncate(3)
