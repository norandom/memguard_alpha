"""Task 6.1: the teacher-forcing full-distribution host, with a fake ``Llama`` (no model load, no GPU)."""
from __future__ import annotations

import hashlib
import http.client
import json
import socket
import socketserver
import threading
import time
import zlib
from urllib.parse import urlsplit

import numpy as np
import pytest

from recall_guard.core.full_distribution import FullDistributionClient, FullDistributionEvidence
from recall_guard.core.profiles import load_profile
from recall_guard.core.request_control import RequestError
from scripts import serve_full_distribution as sfd
from scripts.llamacpp_native import token_statistics

DIGEST = 'sha256:' + 'ab' * 32
BEARER_ENV = {'TEST_FULL_DISTRIBUTION_BEARER': 'test-bearer'}
FIELDS = ('model_digest', 'n_vocab', 'response_token_ids', 'logprob', 'entropy', 'rank', 'top_logprobs')


class FakeLlama:
    """Byte-level tokenizer and deterministic logits per prefix; records every decode pass."""

    BOS = 1

    def __init__(self, n_vocab=300, *, ties=False, delay=0.0, nan=False, fail_marker=None):
        self.n_vocab = n_vocab
        self.ties, self.delay, self.nan, self.fail_marker = ties, delay, nan, fail_marker
        self.decodes = []
        self.active = self.max_active = 0
        self.closed = False
        self._lock = threading.Lock()

    def tokenize(self, text, add_special=True, parse_special=False):
        assert parse_special is False
        if self.fail_marker is not None and self.fail_marker in text:
            raise RuntimeError('tokenize failed')
        ids = [self.BOS] if add_special else []
        return ids + [2 + byte % (self.n_vocab - 2) for byte in text.encode('utf-8')]

    def row(self, prefix):
        seed = zlib.crc32(np.asarray(prefix, dtype=np.int64).tobytes())
        values = np.random.default_rng(seed).normal(0.0, 3.0, self.n_vocab)
        if self.ties:
            values = np.round(values)  # Many exact ties, also across the top-k boundary.
        if self.nan:
            values[3] = np.nan
        return values.astype(np.float32)

    def incremental_logits(self, tokens, rows=None):
        tokens = list(tokens)
        wanted = None if rows is None else set(rows)
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.decodes.append(tuple(tokens))
        try:
            for i in range(len(tokens)):
                if self.delay:
                    time.sleep(self.delay)
                if wanted is None or i in wanted:
                    yield i, self.row(tokens[:i + 1])
        finally:
            with self._lock:
                self.active -= 1

    def close(self):
        self.closed = True


def log_softmax(row):
    x = np.asarray(row, dtype=np.float64)
    shifted = x - x.max()
    return shifted - np.log(np.exp(shifted).sum())


def brute_force(fake, prompt, response, top_k_keep):
    """Expected wire evidence computed independently of the host code."""
    prompt_ids, response_ids = fake.tokenize(prompt), fake.tokenize(response, add_special=False)
    ids = prompt_ids + response_ids
    rows = np.stack([fake.row(ids[:i + 1]) for i in range(len(prompt_ids) - 1, len(ids) - 1)])
    logprob, entropy, rank = token_statistics(rows, response_ids)
    kept = min(top_k_keep, fake.n_vocab)
    top = []
    for row in rows:
        logp = log_softmax(row)
        order = sorted(range(fake.n_vocab), key=lambda token: (-logp[token], token))[:kept]
        top.append([[token, float(logp[token])] for token in order])
    return {'model_digest': DIGEST, 'n_vocab': fake.n_vocab, 'response_token_ids': response_ids,
            'logprob': logprob.tolist(), 'entropy': entropy.tolist(), 'rank': rank.tolist(), 'top_logprobs': top}


def request(prompt='Rate the month.', response=' Neutral.', top_k_keep=100, digest=DIGEST):
    return {'model_digest': digest, 'prompt_text': prompt, 'response_text': response, 'top_k_keep': top_k_keep}


@pytest.fixture
def serve():
    """Start ``make_server(host, port=0)`` in a thread; return the base URL."""
    started = []

    def start(host):
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


def client_for(url, digest=DIGEST):
    return FullDistributionClient(url, model_digest=digest, credential_ref='env:TEST_FULL_DISTRIBUTION_BEARER',
                                  environment=BEARER_ENV, timeout_s=60)


def post_raw(url, body, *, path=sfd.ENDPOINT_PATH, content_length=True):
    parsed = urlsplit(url)
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=60)
    try:
        if content_length:
            connection.request('POST', path, body=body, headers={'Content-Type': 'application/json'})
        else:
            connection.putrequest('POST', path)
            connection.putheader('Content-Type', 'application/json')
            connection.endheaders()
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()


def free_port():
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        return probe.getsockname()[1]


def listening(port):
    try:
        with socket.create_connection(('127.0.0.1', port), timeout=1):
            return True
    except OSError:
        return False


# ---------- evidence assembly and the wire contract ----------

def test_evidence_is_exactly_what_the_contract_accepts_and_matches_brute_force():
    fake = FakeLlama()
    host = sfd.FullDistributionHost(fake, model_digest=DIGEST)
    payload = host.evidence(request())
    assert tuple(payload) == FIELDS
    assert payload == brute_force(fake, 'Rate the month.', ' Neutral.', 100)
    evidence = FullDistributionEvidence.from_dict(json.loads(json.dumps(payload)))
    assert evidence.to_dict() == payload
    assert evidence.model_digest == DIGEST and evidence.n_vocab == 300


def test_client_round_trip_against_the_running_fake_host(serve):
    fake = FakeLlama()
    url = serve(sfd.FullDistributionHost(fake, model_digest=DIGEST))
    evidence = client_for(url).evidence('Rate the month.', ' Neutral.', top_k_keep=100)
    assert isinstance(evidence, FullDistributionEvidence)
    assert evidence.to_dict() == brute_force(fake, 'Rate the month.', ' Neutral.', 100)


def test_alternatives_are_sorted_descending_canonical_on_ties_and_a_prefix_for_smaller_k():
    fake = FakeLlama(ties=True)
    host = sfd.FullDistributionHost(fake, model_digest=DIGEST)
    wide = host.evidence(request(top_k_keep=100))
    assert wide == brute_force(fake, 'Rate the month.', ' Neutral.', 100)  # (-logprob, id) order, ties included.
    for k in (1, 5, 37):
        narrow = host.evidence(request(top_k_keep=k))
        assert [len(position) for position in narrow['top_logprobs']] == [k] * len(narrow['logprob'])
        assert narrow['top_logprobs'] == [position[:k] for position in wide['top_logprobs']]
    for position in wide['top_logprobs']:
        values = [logprob for _, logprob in position]
        assert values == sorted(values, reverse=True)


@pytest.mark.parametrize(('n_vocab', 'top_k_keep', 'kept'), [(300, 100, 100), (300, 7, 7), (50, 100, 50)])
def test_kept_count_is_uniform_at_every_position(serve, n_vocab, top_k_keep, kept):
    url = serve(sfd.FullDistributionHost(FakeLlama(n_vocab), model_digest=DIGEST))
    evidence = client_for(url).evidence('A prompt of some length.', ' A response of some length.', top_k_keep=top_k_keep)
    assert {len(position) for position in evidence.top_logprobs} == {kept}


def test_actual_token_logprob_equals_its_alternative_entry():
    host = sfd.FullDistributionHost(FakeLlama(), model_digest=DIGEST)
    payload = host.evidence(request(top_k_keep=300))  # Full vocabulary kept, so the actual token is always listed.
    for token, logprob, rank, position in zip(payload['response_token_ids'], payload['logprob'], payload['rank'],
                                              payload['top_logprobs'], strict=True):
        assert dict(map(tuple, position))[token] == logprob
        assert rank == 1 + sum(1 for _, value in position if value > logprob)


# ---------- the prompt/response boundary ----------

def test_prompt_and_response_are_tokenised_separately_and_evidence_covers_only_the_response():
    fake = FakeLlama()
    host = sfd.FullDistributionHost(fake, model_digest=DIGEST)
    prompt, response = 'Rate the month.', ' Neutral, then positive.'
    payload = host.evidence(request(prompt, response))
    prompt_ids, response_ids = fake.tokenize(prompt), fake.tokenize(response, add_special=False)
    assert prompt_ids[0] == FakeLlama.BOS and FakeLlama.BOS not in response_ids
    assert payload['response_token_ids'] == response_ids
    assert len(payload['logprob']) == len(payload['top_logprobs']) == len(response_ids)
    assert fake.decodes[-1] == tuple(prompt_ids + response_ids[:-1])  # Teacher forcing; the last token needs no decode.


def test_empty_prompt_scores_the_response_after_bos_only():
    fake = FakeLlama()
    payload = sfd.FullDistributionHost(fake, model_digest=DIGEST).evidence(request('', 'x'))
    assert payload['response_token_ids'] == fake.tokenize('x', add_special=False)
    assert fake.decodes[-1] == (FakeLlama.BOS,)


# ---------- error codes ----------

def test_a_different_model_digest_is_409(serve):
    url = serve(sfd.FullDistributionHost(FakeLlama(), model_digest=DIGEST))
    with pytest.raises(RequestError) as caught:
        client_for(url, digest='sha256:' + 'cd' * 32).evidence('Rate the month.', ' Neutral.')
    assert (caught.value.status_code, caught.value.code) == (409, 'digest_mismatch')


def test_prompt_plus_response_beyond_the_context_is_413(serve):
    fake = FakeLlama()
    url = serve(sfd.FullDistributionHost(fake, model_digest=DIGEST))
    client = client_for(url)
    at_limit = client.evidence('a' * (sfd.N_CTX - 2), 'b')  # BOS + 2046 prompt ids + 1 response id = 2048.
    assert len(at_limit.logprob) == 1
    with pytest.raises(RequestError) as caught:
        client.evidence('a' * (sfd.N_CTX - 1), 'b')
    assert (caught.value.status_code, caught.value.code) == (413, 'oversize')
    assert 'context' in caught.value.detail
    assert len(fake.decodes) == 1  # The refused request never reached the model.


def test_a_body_beyond_one_mebibyte_is_413_without_reading_it(serve):
    url = serve(sfd.FullDistributionHost(FakeLlama(), model_digest=DIGEST))
    parsed = urlsplit(url)
    head = (f'POST {sfd.ENDPOINT_PATH} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n'
            f'Content-Length: {sfd.MAX_BODY_BYTES + 1}\r\n\r\n').encode()
    with socket.create_connection((parsed.hostname, parsed.port), timeout=30) as sock:
        sock.sendall(head)
        reply = b''
        while chunk := sock.recv(65536):
            reply += chunk
    status_line, _, rest = reply.partition(b'\r\n')
    assert status_line.split()[1] == b'413'
    body = json.loads(rest.partition(b'\r\n\r\n')[2])
    assert body['error']['code'] == 'oversize' and 'body' in body['error']['message']


def test_a_body_of_exactly_one_mebibyte_is_accepted(serve):
    url = serve(sfd.FullDistributionHost(FakeLlama(), model_digest=DIGEST))
    body = json.dumps(request()).encode()
    body += b' ' * (sfd.MAX_BODY_BYTES - len(body))
    status, payload = post_raw(url, body)
    assert status == 200 and tuple(payload) == FIELDS


def test_an_empty_response_tokenisation_is_422(serve):
    fake = FakeLlama()
    url = serve(sfd.FullDistributionHost(fake, model_digest=DIGEST))
    status, payload = post_raw(url, json.dumps(request(response='')).encode())
    assert status == 422 and payload['error']['code'] == 'tokenisation_failed'
    assert fake.decodes == []


@pytest.mark.parametrize('response', ['\ud800', 'contains FAIL here'])
def test_a_text_that_fails_to_tokenise_is_422(serve, response):
    url = serve(sfd.FullDistributionHost(FakeLlama(fail_marker='FAIL'), model_digest=DIGEST))
    with pytest.raises(RequestError) as caught:
        client_for(url).evidence('Rate the month.', response)
    assert (caught.value.status_code, caught.value.code) == (422, 'tokenisation_failed')


def test_a_prompt_that_fails_to_tokenise_is_422(serve):
    url = serve(sfd.FullDistributionHost(FakeLlama(fail_marker='FAIL'), model_digest=DIGEST))
    with pytest.raises(RequestError) as caught:
        client_for(url).evidence('FAIL', ' Neutral.')
    assert (caught.value.status_code, caught.value.code) == (422, 'tokenisation_failed')


@pytest.mark.parametrize('body', [
    b'{not json',
    b'\xff\xfe not utf-8',
    b'[1, 2]',
    json.dumps({key: value for key, value in request().items() if key != 'response_text'}).encode(),
    json.dumps({key: value for key, value in request().items() if key != 'model_digest'}).encode(),
    json.dumps(request() | {'temperature': 0}).encode(),
    json.dumps(request(top_k_keep=0)).encode(),
    json.dumps(request(top_k_keep=True)).encode(),
    json.dumps(request(top_k_keep='5')).encode(),
    json.dumps(request() | {'prompt_text': 7}).encode(),
    b'{"model_digest": "a", "model_digest": "b", "prompt_text": "", "response_text": "x", "top_k_keep": 1}',
], ids=['invalid-json', 'not-utf8', 'not-an-object', 'missing-response', 'missing-digest', 'unknown-field',
        'top-k-zero', 'top-k-bool', 'top-k-text', 'prompt-not-text', 'duplicate-key'])
def test_a_malformed_body_is_400(serve, body):
    fake = FakeLlama()
    url = serve(sfd.FullDistributionHost(fake, model_digest=DIGEST))
    status, payload = post_raw(url, body)
    assert status == 400 and payload['error']['code'] == 'malformed'
    assert fake.decodes == []


def test_a_missing_content_length_is_400(serve):
    url = serve(sfd.FullDistributionHost(FakeLlama(), model_digest=DIGEST))
    status, payload = post_raw(url, b'', content_length=False)
    assert status == 400 and payload['error']['code'] == 'malformed'


def test_an_unknown_path_is_404(serve):
    url = serve(sfd.FullDistributionHost(FakeLlama(), model_digest=DIGEST))
    status, payload = post_raw(url, json.dumps(request()).encode(), path='/v1/chat/completions')
    assert status == 404 and payload['error']['code'] == 'not_found'


def test_non_finite_logits_are_a_500_not_evidence(serve):
    url = serve(sfd.FullDistributionHost(FakeLlama(nan=True), model_digest=DIGEST))
    with pytest.raises(RequestError) as caught:
        client_for(url).evidence('Rate the month.', ' Neutral.')
    assert (caught.value.status_code, caught.value.code) == (500, 'http_error')


# ---------- start: digest, single load, warm-up ----------

class Factory:
    def __init__(self):
        self.calls, self.instance = [], None

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        self.instance = FakeLlama()
        return self.instance


def model_file(tmp_path, content=b'GGUF fake model bytes'):
    path = tmp_path / 'model.gguf'
    path.write_bytes(content)
    return path, 'sha256:' + hashlib.sha256(content).hexdigest()


def test_start_refuses_a_gguf_whose_digest_differs_from_the_pin(tmp_path):
    path, _ = model_file(tmp_path)
    factory = Factory()
    with pytest.raises(sfd.StartRefused, match='digest'):
        sfd.open_host(path, pinned_digest='sha256:' + '0' * 64, llama_factory=factory)
    assert factory.calls == []  # Refused before any model load.


def test_main_refuses_to_start_and_never_listens_on_a_digest_mismatch(tmp_path, monkeypatch, capsys):
    path, _ = model_file(tmp_path)
    monkeypatch.setattr(sfd, 'MODEL_PATH', path)
    monkeypatch.setattr(sfd, 'Llama', lambda **kwargs: pytest.fail('model loaded despite a digest mismatch'))
    port = free_port()
    assert sfd.main(['--port', str(port)]) == 2
    assert not listening(port)
    assert 'digest_mismatch' in capsys.readouterr().err


def test_start_loads_one_model_warms_up_once_and_hashes_the_gguf_once(tmp_path, monkeypatch, serve):
    path, pinned = model_file(tmp_path)
    hashed = []
    original = sfd.gguf_digest
    monkeypatch.setattr(sfd, 'gguf_digest', lambda p: hashed.append(p) or original(p))
    factory = Factory()
    host = sfd.open_host(path, pinned_digest=pinned, llama_factory=factory)
    fake = factory.instance
    warm_up = tuple(fake.tokenize(sfd.WARMUP_TEXT))
    assert factory.calls == [{'model_path': path}]
    assert host.model_digest == pinned
    assert fake.decodes == [warm_up]  # Warmed up at start, before the port listens.
    client = client_for(serve(host), digest=pinned)
    for _ in range(3):
        client.evidence('Rate the month.', ' Neutral.')
    assert fake.decodes.count(warm_up) == 1 and len(fake.decodes) == 4
    assert hashed == [path] and len(factory.calls) == 1


# ---------- serving: one request at a time, loopback only ----------

def test_requests_are_served_one_at_a_time(serve):
    fake = FakeLlama(delay=0.002)
    host = sfd.FullDistributionHost(fake, model_digest=DIGEST)
    client = client_for(serve(host))
    results, errors = [], []

    def call():
        try:
            results.append(client.evidence('p' * 20, ' r' * 10))
        except Exception as error:  # Collected and asserted below.
            errors.append(error)

    threads = [threading.Thread(target=call) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    assert errors == [] and len(results) == 3
    assert fake.max_active == 1


def test_the_model_is_used_by_one_caller_at_a_time_even_without_the_server():
    fake = FakeLlama(delay=0.002)
    host = sfd.FullDistributionHost(fake, model_digest=DIGEST)
    threads = [threading.Thread(target=host.evidence, args=(request('p' * 20, ' r' * 10),)) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    assert len(fake.decodes) == 3 and fake.max_active == 1


def test_the_server_binds_loopback_only_serially_and_never_shares_a_port():
    host = sfd.FullDistributionHost(FakeLlama(), model_digest=DIGEST)
    server = sfd.make_server(host, port=0)
    try:
        assert server.server_address[0] == '127.0.0.1'
        assert not isinstance(server, socketserver.ThreadingMixIn)
    finally:
        server.server_close()
    with socket.socket() as taken:
        taken.bind(('127.0.0.1', 0))
        taken.listen()
        with pytest.raises(OSError):
            sfd.make_server(host, port=taken.getsockname()[1])


def test_defaults_match_the_llamacpp_profile():
    profile = load_profile('llamacpp-gemma3-4b')
    assert (sfd.HOST, sfd.PORT) == ('127.0.0.1', urlsplit(profile.full_distribution.endpoint).port)
    assert sfd.MODEL_DIGEST == profile.full_distribution.model_digest
    assert sfd.N_CTX == profile.host_requirements.context_tokens
    assert sfd.ENDPOINT_PATH == '/v1/full_distribution'


# ---------- the exactness-check comparison (pure maths; the live check is a receipt) ----------

def evidence_for(row, target, kept=10):
    logp = log_softmax(row)
    logprob, entropy, rank = token_statistics(row[None, :], [target])
    order = sorted(range(len(row)), key=lambda token: (-logp[token], token))[:kept]
    return float(logprob[0]), float(entropy[0]), int(rank[0]), [[token, float(logp[token])] for token in order]


def test_comparison_redoes_the_server_float32_normaliser_in_float64():
    row = np.random.default_rng(0).normal(0.0, 3.0, 400).astype(np.float32)
    server = log_softmax(row) + 3e-4  # A float32 normaliser leaves a constant log offset per position.
    result = sfd.compare_position(row, server, 17, evidence_for(row, 17))
    assert result['max_abs_diff_full_vector'] < 1e-9
    assert result['abs_diff_logprob'] < 1e-9 and result['abs_diff_entropy'] < 1e-9
    assert result['rank'] == result['ref_rank'] and result['missing'] == 0
    assert result['top_k_id_overlap'] == 10 and result['evidence_matches_logits']
    assert sfd.summarise([result])['within_tolerance']


def test_comparison_fails_on_a_real_difference_or_a_missing_entry():
    row = np.random.default_rng(1).normal(0.0, 3.0, 400).astype(np.float32)
    server = log_softmax(row)
    server[5] += 1e-3
    assert not sfd.summarise([sfd.compare_position(row, server, 17, evidence_for(row, 17))])['within_tolerance']
    server = log_softmax(row)
    server[9] = np.nan
    result = sfd.compare_position(row, server, 17, evidence_for(row, 17))
    assert result['missing'] == 1 and not sfd.summarise([result])['within_tolerance']


def test_server_vector_places_every_entry_by_token_id():
    vector = sfd.server_vector([{'id': 2, 'logprob': -0.5}, {'id': 0, 'logprob': -1.5}], 4)
    assert vector[0] == -1.5 and vector[2] == -0.5 and np.isnan(vector[1]) and np.isnan(vector[3])
