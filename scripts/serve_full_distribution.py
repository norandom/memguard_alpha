"""Teacher-forcing full-distribution host for prompt plus response (task 6.1, FullDistributionHost).

Native Windows model host (the model-host exception of the execution
environment): run it on the Windows side with the checkout's
``uv run --no-sync python scripts/serve_full_distribution.py``, never in a
container. One process loads ONE llama.cpp model instance through
``scripts/llamacpp_native.py`` (b12199, Vulkan iGPU, the reference server's
settings) on the pinned GGUF, computes the file's SHA-256 once and refuses to
start when it differs from the pin, decodes a fixed warm-up text once and only
then listens on 127.0.0.1:11580, so a listening port means ready. It binds
loopback only and serves one request at a time; the bearer gateway of task 6.2
(port 11581) owns authentication and its 401.

``POST /v1/full_distribution`` with ``{model_digest, prompt_text, response_text,
top_k_keep}`` returns exactly the wire shape that
``FullDistributionEvidence.from_dict`` (recall_guard/core/full_distribution.py)
accepts; the host checks its own answer against that contract before sending.

Tokenisation: the prompt with ``add_special=True`` (BOS as the model requires),
the response with ``add_special=False``, both with ``parse_special=False``, ids
concatenated. No chat template is applied; the text is scored as sent. The
evidence covers the response ids only. This prompt/response boundary can differ
from the generation's own tokenisation of the same text, where a token may
straddle the boundary.

Teacher forcing decodes ONE token per ``llama_decode`` on a growing KV cache,
the route R40 matched to llama-server within 1e-6. For each response position
the full logit row gives, in float64: the actual token's logprob, the entropy
(nats) and its rank (``token_statistics``), and the top ``min(top_k_keep,
n_vocab)`` alternatives ``(token_id, logprob)`` chosen by ``argpartition`` and
sorted by descending logprob, ties broken by the smaller id, so every position
keeps the same count and any smaller k is a prefix.

``POST /v1/full_distribution_batch`` (task 6.4) with ``{model_digest,
prompt_text, response_texts[], top_k_keep}`` returns ``{"evidence": [...]}``,
one evidence object per response in request order, each equal to the single
route's for that pair. The prompt is tokenised and decoded once (one token per
decode, as on the single route); the logits of its last position predict every
response's first token; each response is then teacher-forced one decode at a
time on top of the cached prompt and the KV cache is rolled back to the prompt
length (``llama_memory_seq_rm``) after each response. The time of the prompt
and of each response is kept in ``last_batch_timings``.

Errors are JSON ``{"error": {"code", "message"}}``: 409 ``digest_mismatch``;
413 ``oversize`` (prompt plus response ids beyond the 2048-token context, on
the batch route prompt plus the longest response, or a body beyond 1 MiB); 422
``tokenisation_failed`` (a text fails to tokenise, or a response, or the
prompt, tokenises to no ids); 400 ``malformed``; 404 ``not_found``; 500
``internal_error``; 503 ``memory_low``.

Memory guard (task 6.4): the host samples free physical memory every 5 s and
before each request and logs every sample as one JSON line on stdout. While
free memory is below the floor (``--memory-floor-bytes``, by default the
llamacpp-gemma3-4b profile's ``host_requirements.memory_floor()``, one quarter
of ``min_free_ram_bytes``) it refuses new requests with 503 ``memory_low``
and logs the refusal; the client reports that as the retryable ``host_busy``.
The lifecycle script launches the host with ``python -X faulthandler`` so a
native death leaves a Python stack in its log; nothing restarts it.

``--check-against-server --batch`` (task 6.4) runs the same check on the batch
route: the two pairs as one-response batches plus one prompt carrying two
responses, each batch served twice and each pair once more on the single route
(the batch evidence must equal it), with the per-item timing in the receipt
under runs/finance-sim/hosts/full-distribution/exactness/batch/.

``--check-against-server`` runs the exactness check instead of serving: it
starts llama-server b12199 on 127.0.0.1:11560 (the R40 reference method),
records its full-vocabulary logprobs for every response position of two fixed
synthetic prompt/response pairs, stops it and verifies the port is free, then
loads this host (so one model is resident at a time), serves the pairs twice on
127.0.0.1:11580 and compares within 1e-4 after a float64 renormalisation of
the server's float32 normaliser, with identical ranks and bitwise identical
runs. The private receipt goes to runs/finance-sim/hosts/full-distribution/exactness/.

Two passes (user correction 2026-11-10: never two models loaded, memory-heavy
steps one after another): ``--check-against-server [--batch] --record`` is pass
A, which starts only llama-server, writes its reference (server tokens,
positions and the full-vocabulary vectors) to ``reference-recording.json`` and
``reference-vectors.npz`` in the output folder, stops it and verifies that no
listener and no llama-server process remains. ``--check-against-server [--batch]
--compare`` is pass B, run as a separate process afterwards: it refuses while a
llama-server is resident or when the recording is missing, failed, made for
other pairs or settings, or its vector file no longer matches the recorded
SHA-256; otherwise it loads only the host and compares against the recording.
Without either flag both passes run one after the other in one process, the
comparison again reading the recording from the file.
"""
from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import re
import socketserver
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # Run as a file; the bindings live in scripts/, the contract in recall_guard/.
    sys.path.insert(0, str(ROOT))
from recall_guard.core.full_distribution import (  # noqa: E402
    BATCH_ENDPOINT_PATH,
    ENDPOINT_PATH,
    FullDistributionEvidence,
    logprobs_from_logits,
)
from scripts.llamacpp_native import (  # noqa: E402
    LLAMA_DIR,
    MODEL_DIGEST,
    MODEL_PATH,
    N_CTX,
    SERVER_FLAGS,
    SERVER_PORT,
    Llama,
    build_info,
    post_json,
    token_statistics,
)

HOST, PORT = '127.0.0.1', 11580
MAX_BODY_BYTES = 1 << 20
SOCKET_TIMEOUT_S = 60  # A stalled client cannot hold the one-at-a-time server.
WARMUP_TEXT = 'The quick brown fox jumps over the lazy dog.'
REQUEST_FIELDS = ('model_digest', 'prompt_text', 'response_text', 'top_k_keep')
BATCH_REQUEST_FIELDS = ('model_digest', 'prompt_text', 'response_texts', 'top_k_keep')
MEMORY_SAMPLE_S = 5.0  # The memory guard's sampling interval (task 6.4).
_CONTENT_LENGTH = re.compile(r'[0-9]{1,12}')


class StartRefused(RuntimeError):
    """The host refuses to start (for example a GGUF whose digest differs from the pin)."""


class HostError(Exception):
    """A refused request: HTTP ``status`` with an error ``code`` and a message."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def error_body(code: str, message: str) -> dict:
    return {'error': {'code': code, 'message': message}}


def gguf_digest(path) -> str:
    """``sha256:<hex>`` of the model file, read once at start."""
    with open(path, 'rb') as handle:
        return 'sha256:' + hashlib.file_digest(handle, 'sha256').hexdigest()


def _unique_keys(pairs):
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError('duplicate key')
    return dict(pairs)


def parse_request(body: bytes, validate=None) -> dict:
    """Decode a request body (UTF-8 JSON object, no duplicate keys) and validate its fields.

    ``validate`` defaults to the single route's ``validate_request``.
    """
    try:
        data = json.loads(body.decode('utf-8'), object_pairs_hook=_unique_keys)
    except (UnicodeDecodeError, ValueError):
        raise HostError(400, 'malformed', 'the body must be a UTF-8 JSON object without duplicate keys') from None
    return (validate_request if validate is None else validate)(data)


def _exact_fields(data, fields) -> None:
    if not isinstance(data, Mapping):
        raise HostError(400, 'malformed', 'the body must be a JSON object')
    missing = [name for name in fields if name not in data]
    unknown = len(set(data) - set(fields))
    if missing or unknown:
        raise HostError(400, 'malformed', f'expected exactly {list(fields)}; missing {missing}, '
                                          f'{unknown} unknown field(s)')


def _top_k_keep(value) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise HostError(400, 'malformed', 'top_k_keep must be a positive integer')


def validate_request(data) -> dict:
    """Exactly the four request fields: three strings and a positive integer ``top_k_keep``."""
    _exact_fields(data, REQUEST_FIELDS)
    for name in REQUEST_FIELDS[:3]:
        if not isinstance(data[name], str):
            raise HostError(400, 'malformed', f'{name} must be a string')
    _top_k_keep(data['top_k_keep'])
    return {name: data[name] for name in REQUEST_FIELDS}


def validate_batch_request(data) -> dict:
    """Exactly the four batch fields: two strings, a nonempty list of strings and a positive ``top_k_keep``."""
    _exact_fields(data, BATCH_REQUEST_FIELDS)
    for name in BATCH_REQUEST_FIELDS[:2]:
        if not isinstance(data[name], str):
            raise HostError(400, 'malformed', f'{name} must be a string')
    texts = data['response_texts']
    if not isinstance(texts, list) or not texts or not all(isinstance(text, str) for text in texts):
        raise HostError(400, 'malformed', 'response_texts must be a nonempty list of strings')
    _top_k_keep(data['top_k_keep'])
    return {name: list(data[name]) if name == 'response_texts' else data[name] for name in BATCH_REQUEST_FIELDS}


def top_token_ids(logprobs: np.ndarray, kept: int) -> np.ndarray:
    """The ``kept`` most likely token ids of one row, by descending logprob, then ascending id.

    ``argpartition`` finds the kept-th largest value; every id at or above it is
    a candidate, so ties across the boundary resolve canonically by id.
    """
    n_vocab = logprobs.shape[0]
    if kept < n_vocab:
        boundary = logprobs[np.argpartition(-logprobs, kept - 1)[kept - 1]]
        candidates = np.flatnonzero(logprobs >= boundary)
    else:
        candidates = np.arange(n_vocab)
    order = np.lexsort((candidates, -logprobs[candidates]))
    return candidates[order[:kept]]


def position_evidence(logits_row, target: int, kept: int) -> tuple[float, float, int, list[list]]:
    """(logprob, entropy, rank, alternatives) of one response position from its full logit row."""
    row = np.asarray(logits_row)[None, :]
    try:
        logprobs = logprobs_from_logits(row)[0]  # float64 log-softmax (3.5).
    except ValueError as error:
        raise HostError(500, 'internal_error', f'the logits are not usable: {error}') from None
    logprob, entropy, rank = token_statistics(row, [target])
    alternatives = [[int(token), float(logprobs[token])] for token in top_token_ids(logprobs, kept)]
    return float(logprob[0]), float(entropy[0]), int(rank[0]), alternatives


class MemoryGuard:
    """Free physical memory, sampled every ``interval_s`` and before each request (task 6.4).

    ``probe()`` returns free physical memory in bytes (by default
    ``psutil.virtual_memory().available``). Every sample, periodic or taken
    when a request arrives, goes to ``log`` as ``{"memory_sample": {...}}``, so
    the host log shows the memory trend before a refusal or a native death.
    ``admit()`` refuses a request with 503 ``memory_low`` while the sample is
    below ``floor_bytes`` and logs ``{"refused": {...}}``.
    """

    def __init__(self, floor_bytes: int, *, probe=None, interval_s: float = MEMORY_SAMPLE_S, log=None) -> None:
        if isinstance(floor_bytes, bool) or not isinstance(floor_bytes, int):
            raise TypeError('floor_bytes must be an integer')
        if floor_bytes < 1:
            raise ValueError('floor_bytes must be positive')
        self.floor_bytes, self.interval_s = floor_bytes, float(interval_s)
        self.probe = _free_ram if probe is None else probe
        self.log = log if log is not None else (lambda record: None)
        self.last: dict | None = None
        self.samples = self.refusals = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def sample(self, reason: str = 'periodic') -> dict:
        free = int(self.probe())
        record = {'at': _now(), 'free_ram_bytes': free, 'floor_bytes': self.floor_bytes,
                  'below_floor': free < self.floor_bytes, 'reason': reason}
        with self._lock:
            self.last, self.samples = record, self.samples + 1
        self.log({'memory_sample': record})
        return record

    def admit(self) -> None:
        """Raise 503 ``memory_low`` while free physical memory is below the floor."""
        record = self.sample('request')
        if record['below_floor']:
            with self._lock:
                self.refusals += 1
            self.log({'refused': {'code': 'memory_low', **record}})
            raise HostError(503, 'memory_low', f'free physical memory is {record["free_ram_bytes"]} bytes, below '
                                               f'the floor of {self.floor_bytes} bytes; retry later')

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name='memory-guard', daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(max(5.0, 2 * self.interval_s))

    def _run(self) -> None:
        while True:
            try:
                self.sample()
            except Exception as error:  # A failed reading is logged; sampling goes on.
                self.log({'memory_sample_failed': f'{type(error).__name__}: {str(error)[:200]}', 'at': _now()})
            if self._stop.wait(self.interval_s):
                return


class FullDistributionHost:
    """One loaded model and its teacher-forcing evidence; one caller at a time.

    ``llama`` provides ``tokenize(text, add_special, parse_special)``,
    ``incremental_logits(tokens, rows)``, ``n_vocab`` and ``close()``, and for
    the batch route ``continue_logits(tokens, start)`` and ``truncate(n_keep)``
    (``scripts.llamacpp_native.Llama``, or a fake in tests). ``memory_guard``
    (a ``MemoryGuard``) admits each request when given.
    """

    def __init__(self, llama, *, model_digest: str, n_ctx: int = N_CTX, memory_guard: MemoryGuard | None = None) -> None:
        self.llama, self.model_digest, self.n_ctx = llama, model_digest, n_ctx
        self.n_vocab = int(llama.n_vocab)
        self.timings: dict[str, float] = {}
        self.memory_guard = memory_guard
        self.last_batch_timings: dict | None = None
        self._lock = threading.Lock()

    def warm_up(self) -> None:
        """Decode the fixed warm-up text once and discard the result."""
        with self._lock:
            started = time.perf_counter()
            for _ in self.llama.incremental_logits(self.llama.tokenize(WARMUP_TEXT)):
                pass
            self.timings['warmup_seconds'] = time.perf_counter() - started

    def close(self) -> None:
        with self._lock:
            self.llama.close()

    def _tokenize(self, text: str, *, add_special: bool, what: str) -> list[int]:
        try:
            return [int(token) for token in self.llama.tokenize(text, add_special=add_special, parse_special=False)]
        except (RuntimeError, UnicodeError, ValueError):
            raise HostError(422, 'tokenisation_failed', f'the {what} text fails to tokenise') from None

    def evidence(self, request: Mapping) -> dict:
        """Teacher-force prompt plus response; return the evidence wire dict for the response positions."""
        request = validate_request(request)
        if request['model_digest'] != self.model_digest:
            raise HostError(409, 'digest_mismatch', f'this host serves {self.model_digest}')
        with self._lock:
            prompt_ids = self._tokenize(request['prompt_text'], add_special=True, what='prompt')
            response_ids = self._tokenize(request['response_text'], add_special=False, what='response')
            if not response_ids:
                raise HostError(422, 'tokenisation_failed', 'the response tokenises to no ids')
            if not prompt_ids:
                raise HostError(422, 'tokenisation_failed', 'the prompt tokenises to no ids, so the first '
                                                            'response token has no context')
            total = len(prompt_ids) + len(response_ids)
            if total > self.n_ctx:
                raise HostError(413, 'oversize', f'prompt plus response is {total} ids, beyond the context '
                                                 f'of {self.n_ctx}')
            kept = min(request['top_k_keep'], self.n_vocab)
            ids = prompt_ids + response_ids
            first = len(prompt_ids) - 1  # The logits at the last prompt position predict the first response id.
            columns = {'logprob': [], 'entropy': [], 'rank': [], 'top_logprobs': []}
            rows = self.llama.incremental_logits(ids[:-1], rows=range(first, len(ids) - 1))
            try:
                for expected, ((position, row), target) in enumerate(zip(rows, response_ids, strict=True), first):
                    if position != expected:
                        raise HostError(500, 'internal_error', 'the decode returned positions out of order')
                    for name, value in zip(columns, position_evidence(row, target, kept), strict=True):
                        columns[name].append(value)
            finally:
                rows.close()
        return self._payload(response_ids, columns)

    def _payload(self, response_ids: list[int], columns: dict) -> dict:
        """The evidence wire dict, checked against its contract before it is sent."""
        payload = {'model_digest': self.model_digest, 'n_vocab': self.n_vocab,
                   'response_token_ids': response_ids, **columns}
        try:
            FullDistributionEvidence.from_dict(payload)
        except (TypeError, ValueError) as error:
            raise HostError(500, 'internal_error', f'the evidence fails its contract: {error}') from None
        return payload

    def evidence_batch(self, request: Mapping) -> dict:
        """Teacher-force each response against the prompt decoded once; ``{"evidence": [...]}`` in request order.

        Every refusal (digest, tokenisation, size) happens before any decode. The
        prompt is decoded one token per decode from an empty cache, as on the
        single route; each response then continues on the cached prompt and the
        KV cache is rolled back to the prompt length after it, so each response
        position sees the single route's computation.
        """
        request = validate_batch_request(request)
        if request['model_digest'] != self.model_digest:
            raise HostError(409, 'digest_mismatch', f'this host serves {self.model_digest}')
        with self._lock:
            started = time.perf_counter()
            prompt_ids = self._tokenize(request['prompt_text'], add_special=True, what='prompt')
            responses = []
            for index, text in enumerate(request['response_texts']):
                ids = self._tokenize(text, add_special=False, what=f'response {index}')
                if not ids:
                    raise HostError(422, 'tokenisation_failed', f'response {index} tokenises to no ids')
                responses.append(ids)
            if not prompt_ids:
                raise HostError(422, 'tokenisation_failed', 'the prompt tokenises to no ids, so the first '
                                                            'response token has no context')
            total = len(prompt_ids) + max(len(ids) for ids in responses)
            if total > self.n_ctx:
                raise HostError(413, 'oversize', f'prompt plus the longest response is {total} ids, beyond the '
                                                 f'context of {self.n_ctx}')
            kept = min(request['top_k_keep'], self.n_vocab)
            last = self._decode_prompt(prompt_ids)
            timings = {'prompt_ids': len(prompt_ids), 'prompt_seconds': time.perf_counter() - started, 'responses': []}
            evidence = []
            for response_ids in responses:
                began = time.perf_counter()
                columns = {'logprob': [], 'entropy': [], 'rank': [], 'top_logprobs': []}
                for _, row, target in self._response_rows(len(prompt_ids), last, response_ids):
                    for name, value in zip(columns, position_evidence(row, target, kept), strict=True):
                        columns[name].append(value)
                self.llama.truncate(len(prompt_ids))  # Roll back to the prompt for the next response.
                evidence.append(self._payload(response_ids, columns))
                timings['responses'].append({'response_ids': len(response_ids), 'decodes': len(response_ids) - 1,
                                             'seconds': time.perf_counter() - began})
            timings['total_seconds'] = time.perf_counter() - started
            self.last_batch_timings = timings
        return {'evidence': evidence}

    def batch_logit_rows(self, prompt_ids: list[int], responses: list[list[int]]) -> list[np.ndarray]:
        """Full logit rows of every response position by the batch computation (for the exactness check)."""
        with self._lock:
            last = self._decode_prompt(prompt_ids)
            blocks = []
            for response_ids in responses:
                blocks.append(np.stack([row for _, row, _ in self._response_rows(len(prompt_ids), last,
                                                                                    response_ids)]))
                self.llama.truncate(len(prompt_ids))
            return blocks

    def _decode_prompt(self, prompt_ids: list[int]) -> np.ndarray:
        """Decode the prompt from an empty cache, one token per decode; the logits of its last position."""
        last = len(prompt_ids) - 1
        rows = self.llama.incremental_logits(prompt_ids, rows=range(last, len(prompt_ids)))
        try:
            decoded = list(rows)
        finally:
            rows.close()
        if [position for position, _ in decoded] != [last]:
            raise HostError(500, 'internal_error', 'the prompt decode returned positions out of order')
        return decoded[0][1]

    def _response_rows(self, prompt_length: int, last, response_ids: list[int]):
        """Yield (position, logit row, target) of each response position on the cached prompt."""
        yield prompt_length - 1, last, response_ids[0]
        if len(response_ids) == 1:
            return
        rows = self.llama.continue_logits(response_ids[:-1], start=prompt_length)
        try:
            for expected, ((position, row), target) in enumerate(zip(rows, response_ids[1:], strict=True),
                                                                 prompt_length):
                if position != expected:
                    raise HostError(500, 'internal_error', 'the decode returned positions out of order')
                yield position, row, target
        finally:
            rows.close()

    def respond(self, body: bytes, *, batch: bool = False) -> tuple[int, dict]:
        """HTTP status and JSON payload for one request body (the batch route when ``batch``); never raises.

        The memory guard admits the request first, before any parsing or model work.
        """
        try:
            if self.memory_guard is not None:
                self.memory_guard.admit()
            if batch:
                return 200, self.evidence_batch(parse_request(body, validate=validate_batch_request))
            return 200, self.evidence(parse_request(body))
        except HostError as error:
            return error.status, error_body(error.code, error.message)
        except Exception as error:  # A failed decode answers 500 instead of stopping the serial server.
            return 500, error_body('internal_error', f'{type(error).__name__}: {str(error)[:200]}')


class _Handler(BaseHTTPRequestHandler):
    server_version = 'recall-guard-full-distribution/1'
    timeout = SOCKET_TIMEOUT_S

    def do_POST(self) -> None:
        routes = {ENDPOINT_PATH: False, BATCH_ENDPOINT_PATH: True}
        if self.path not in routes:
            self._send(404, error_body('not_found', f'only POST {ENDPOINT_PATH} and POST {BATCH_ENDPOINT_PATH} '
                                                    'are served'))
            return
        length = self.headers.get('Content-Length')
        if self.headers.get('Transfer-Encoding') or length is None or not _CONTENT_LENGTH.fullmatch(length):
            self._send(400, error_body('malformed', 'a Content-Length header is required'))
            return
        if int(length) > MAX_BODY_BYTES:
            self.close_connection = True
            self._send(413, error_body('oversize', f'the body exceeds {MAX_BODY_BYTES} bytes'))
            return
        body = self.rfile.read(int(length))
        if len(body) != int(length):
            self._send(400, error_body('malformed', 'the body is shorter than its Content-Length'))
            return
        self._send(*self.server.full_distribution_host.respond(body, batch=routes[self.path]))

    def _send(self, status: int, payload: dict) -> None:
        data = json.dumps(payload, allow_nan=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class _LoopbackServer(HTTPServer):
    """Serial (not threading) HTTP server on 127.0.0.1 that never shares its port."""

    allow_reuse_address = False  # SO_REUSEADDR on Windows would let a second listener share the port.

    def __init__(self, host: FullDistributionHost | None, port: int, *, activate: bool = True) -> None:
        self.full_distribution_host = host
        super().__init__((HOST, port), _Handler, bind_and_activate=False)
        try:
            self.server_bind()
            if activate:
                self.server_activate()
        except BaseException:
            self.server_close()
            raise

    def server_bind(self) -> None:
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = HOST, self.server_address[1]  # No reverse DNS lookup.


def make_server(host: FullDistributionHost | None, port: int = PORT, *, activate: bool = True) -> HTTPServer:
    """Bind 127.0.0.1:``port`` (0 picks a free port); listen at once unless ``activate`` is False."""
    return _LoopbackServer(host, port, activate=activate)


def open_host(model_path=None, *, pinned_digest: str = MODEL_DIGEST, llama_factory=None) -> FullDistributionHost:
    """Check the GGUF digest once, load one model instance and warm it up once."""
    model_path = Path(MODEL_PATH if model_path is None else model_path)
    factory = Llama if llama_factory is None else llama_factory
    started = time.perf_counter()
    try:
        digest = gguf_digest(model_path)
    except OSError as error:
        raise StartRefused(f'model_unreadable: {model_path}: {error.strerror or error}') from None
    hash_seconds = time.perf_counter() - started
    if digest != pinned_digest:
        raise StartRefused(f'digest_mismatch: {model_path.name} is {digest}, the pin is {pinned_digest}')
    started = time.perf_counter()
    llama = factory(model_path=model_path)
    load_seconds = time.perf_counter() - started
    host = FullDistributionHost(llama, model_digest=digest)
    try:
        host.warm_up()
    except BaseException:
        host.close()
        raise
    host.timings |= {'hash_seconds': hash_seconds, 'load_seconds': load_seconds}
    return host


def _log_line(record: dict) -> None:
    print(json.dumps(record), flush=True)


def _memory_floor_bytes() -> int:
    """The llamacpp-gemma3-4b profile's memory floor (``host_requirements.memory_floor()``)."""
    from recall_guard.core.profiles import load_profile
    floor = load_profile('llamacpp-gemma3-4b').host_requirements.memory_floor()
    if floor is None:
        raise StartRefused('the llamacpp-gemma3-4b profile records no memory floor')
    return int(floor)


def serve(port: int = PORT, *, memory_floor_bytes: int | None = None) -> int:
    """Reserve the port, sample free memory, start the host, then listen until interrupted."""
    try:
        server = make_server(None, port, activate=False)
    except OSError as error:
        print(f'refused to start: port_unavailable: {HOST}:{port}: {error.strerror or error}', file=sys.stderr)
        return 2
    guard = None
    try:
        guard = MemoryGuard(_memory_floor_bytes() if memory_floor_bytes is None else memory_floor_bytes,
                            log=_log_line)
        guard.start()  # Samples during the hash, the load and the warm-up too.
        host = open_host()
    except StartRefused as error:
        if guard is not None:
            guard.stop()
        server.server_close()
        print(f'refused to start: {error}', file=sys.stderr)
        return 2
    except BaseException:  # A failed load: stop sampling and release the port before the error propagates.
        if guard is not None:
            guard.stop()
        server.server_close()
        raise
    host.memory_guard = guard
    server.full_distribution_host = host
    try:
        server.server_activate()
        print(json.dumps({'listening': f'{HOST}:{server.server_address[1]}', 'endpoint': ENDPOINT_PATH,
                          'batch_endpoint': BATCH_ENDPOINT_PATH, 'model_digest': host.model_digest,
                          'n_vocab': host.n_vocab, 'memory_floor_bytes': guard.floor_bytes,
                          'memory_sample_seconds': guard.interval_s, 'faulthandler': faulthandler.is_enabled(),
                          **host.timings}), flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        guard.stop()
        host.close()
    return 0


# ---------- exactness check against llama-server (--check-against-server) ----------

RECEIPT_DIR = ROOT / 'runs/finance-sim/hosts/full-distribution/exactness'
TOLERANCE = 1e-4
REFERENCE_N_PROBS = 262145  # At least the vocabulary: every entry of the next-token distribution (R40).
REFERENCE_BODY = {'n_predict': 1, 'temperature': 0, 'n_probs': REFERENCE_N_PROBS, 'post_sampling_probs': False,
                  'cache_prompt': True, 'seed': 0}
SERVER_FLOOR = -1e30  # llama-server's float32 floor (about -3.4e38) for underflowed entries.
CHECK_TOP_K = 100
EXACTNESS_PAIRS = (  # Short, synthetic, no licensed text.
    {'name': 'json_loadings', 'prompt_text': 'Return the regime loadings for the month as JSON.\n',
     'response_text': '{"growth": 0.25, "inflation": -0.5, "liquidity": 0.1}'},
    {'name': 'plain_sentence', 'prompt_text': 'Summarise the note in one sentence. Note: output rose and prices '
                                              'were stable.\nSummary:',
     'response_text': ' Output rose while prices held steady.'},
)
BATCH_RECEIPT_DIR = RECEIPT_DIR / 'batch'
BATCH_EXACTNESS_ITEMS = (  # The two pairs as one-response batches, plus one prompt carrying two responses (6.4).
    *({'name': pair['name'], 'prompt_text': pair['prompt_text'], 'response_texts': [pair['response_text']]}
      for pair in EXACTNESS_PAIRS),
    {'name': 'shared_prompt', 'prompt_text': 'Name the macro regime of the month and give one reason.\nAnswer:',
     'response_texts': [' Expansion, because output and employment rose together.', ' Slowdown.']},
)
KV_ROLLBACK = 'llama_memory_seq_rm'  # Exported by llama.dll b12199 (llama_memory_* API); llama_kv_self_seq_rm is not.
OTHER_HOST_PORTS = (11570, 11571, 11581)  # The 6.2/6.3 llama.cpp hosts and gateways must not be resident.


def batch_reference_pairs(items) -> list[dict]:
    """Each (prompt, response) of the batch items as a reference pair named ``<item>/<index>``."""
    return [{'name': f'{item["name"]}/{index}', 'item': item['name'], 'index': index,
             'prompt_text': item['prompt_text'], 'response_text': text}
            for item in items for index, text in enumerate(item['response_texts'])]
RAM_WAIT_S = 600.0  # How long a load waits for the profile's free-RAM minimum before it is refused.


def server_vector(entries, size: int) -> np.ndarray:
    """llama-server ``top_logprobs`` entries placed by token id into a float64 vector (NaN where absent)."""
    vector = np.full(size, np.nan, dtype=np.float64)
    for entry in entries:
        vector[int(entry['id'])] = float(entry['logprob'])
    return vector


def compare_position(host_row, server_logprobs, target: int, evidence) -> dict:
    """One response position: the host's logit row and emitted evidence against llama-server's vector.

    llama-server normalises ``n_probs`` in float32, so its probabilities sum to 1
    only within about 4e-4 (a constant log offset per position). As in R40 its
    vector is re-normalised in float64 over the entries above its float32 floor,
    so the comparison is between the logits themselves. ``evidence`` is the
    host's ``(logprob, entropy, rank, alternatives)`` for this position.
    """
    logprob, entropy, rank, alternatives = evidence
    host_logprobs = logprobs_from_logits(np.asarray(host_row)[None, :])[0]
    server = np.asarray(server_logprobs, dtype=np.float64)[:host_logprobs.shape[0]].copy()
    usable = np.isfinite(server) & (server > SERVER_FLOOR)
    missing = int(np.isnan(server).sum())
    floored = int((~usable).sum()) - missing
    sum_p_minus_1 = float(np.exp(server[usable]).sum() - 1)
    peak = server[usable].max()
    server[usable] -= peak + np.log(np.exp(server[usable] - peak).sum())
    ref_logprob = float(server[target])
    p = np.exp(server[usable])
    ref_entropy = float(-(p * server[usable]).sum())
    ref_rank = int(1 + (server[usable] > ref_logprob).sum())
    ids = np.array([token for token, _ in alternatives], dtype=np.int64)
    values = np.array([value for _, value in alternatives], dtype=np.float64)
    server_top = top_token_ids(np.where(usable, server, -np.inf), len(ids))
    own = position_evidence(host_row, target, len(ids))
    return {'missing': missing, 'floored': floored, 'server_sum_p_minus_1': sum_p_minus_1,
            'max_abs_diff_full_vector': float(np.abs(host_logprobs[usable] - server[usable]).max()),
            'abs_diff_logprob': abs(logprob - ref_logprob), 'abs_diff_entropy': abs(entropy - ref_entropy),
            'max_abs_diff_alternatives': float(np.abs(values - server[ids]).max()) if len(ids) else 0.0,
            'rank': int(rank), 'ref_rank': ref_rank,
            'top_k_id_overlap': len(set(ids.tolist()) & set(server_top.tolist())), 'top_k': len(ids),
            'evidence_matches_logits': own == (logprob, entropy, rank, [list(pair) for pair in alternatives])}


_DIFFERENCES = ('max_abs_diff_full_vector', 'abs_diff_logprob', 'abs_diff_entropy', 'max_abs_diff_alternatives')


def summarise(positions, tolerance: float = TOLERANCE) -> dict:
    """Worst differences over positions; within tolerance only if every difference, rank and entry agrees."""
    worst = {name: float(np.max([position[name] for position in positions])) for name in _DIFFERENCES}
    rank_mismatches = sum(position['rank'] != position['ref_rank'] for position in positions)
    missing = sum(position['missing'] for position in positions)
    consistent = all(position['evidence_matches_logits'] for position in positions)
    within = (bool(positions) and all(value <= tolerance for value in worst.values())  # NaN fails.
              and rank_mismatches == 0 and missing == 0 and consistent)
    return {'positions_checked': len(positions), 'tolerance': tolerance, 'max_abs_diff': worst,
            'rank_mismatches': rank_mismatches, 'missing_entries': missing,
            'floored_entries': sum(position['floored'] for position in positions),
            'evidence_matches_logits': consistent,
            'min_top_k_id_overlap': min(position['top_k_id_overlap'] for position in positions),
            'within_tolerance': within}


def _now() -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def _listeners(ports) -> list[dict]:
    import psutil  # Core dependency; only the live check needs it.
    return [{'port': c.laddr.port, 'address': c.laddr.ip, 'pid': c.pid}
            for c in psutil.net_connections(kind='tcp') if c.status == psutil.CONN_LISTEN and c.laddr.port in ports]


def _other_states(ports) -> int:
    import psutil
    return sum(1 for c in psutil.net_connections(kind='tcp')
               if c.status != psutil.CONN_LISTEN and c.laddr and c.laddr.port in ports)


def _processes(prefixes=('llama-server',)) -> list[dict]:
    import psutil
    return [{'pid': p.info['pid'], 'name': p.info['name']} for p in psutil.process_iter(['pid', 'name'])
            if (p.info['name'] or '').lower().startswith(prefixes)]


def _free_ram() -> int:
    import psutil
    return int(psutil.virtual_memory().available)


def _await_free_ram(minimum: int, seconds: float = RAM_WAIT_S) -> dict:
    """Wait until free RAM reaches ``minimum`` for three samples in a row; refuse the load otherwise.

    Other workloads on the host (builds, editors) move free RAM by gigabytes
    within seconds, so one sample is not evidence that a load fits.
    """
    started, deadline, streak, samples = time.monotonic(), time.monotonic() + seconds, 0, []
    while True:
        samples.append(_free_ram())
        streak = streak + 1 if samples[-1] >= minimum else 0
        if streak >= 3:
            return {'free_ram_bytes': min(samples[-3:]), 'minimum': minimum,
                    'waited_seconds': time.monotonic() - started, 'samples': len(samples)}
        if time.monotonic() > deadline:
            raise RuntimeError(f'free RAM stayed below {minimum} bytes for {seconds:.0f} s (last {samples[-1]}); '
                               'one load does not fit now')
        time.sleep(2)


def _ollama_loaded_models() -> list[str] | None:
    try:
        with urllib.request.urlopen('http://127.0.0.1:11434/api/ps', timeout=5) as reply:
            return [model.get('name') for model in json.loads(reply.read()).get('models', [])]
    except (OSError, ValueError):
        return None  # Not listening (or not Ollama): nothing resident there.


def _min_free_ram_bytes() -> int:
    from recall_guard.core.profiles import load_profile
    return int(load_profile('llamacpp-gemma3-4b').host_requirements.min_free_ram_bytes)


def _wait_until_free(ports, seconds: float = 15.0) -> list[dict]:
    deadline = time.monotonic() + seconds
    while (remaining := _listeners(ports)) and time.monotonic() < deadline:
        time.sleep(0.5)
    return remaining


def _preflight(port: int) -> dict:
    ports = (SERVER_PORT, port, *OTHER_HOST_PORTS)
    found = {'checked_at': _now(), 'listeners': _listeners(ports), 'llama_server_processes': _processes(),
             'other_model_host_processes': _processes(('ollama', 'flm')),
             'ollama_loaded_models': _ollama_loaded_models(), 'free_ram_bytes': _free_ram(),
             'min_free_ram_bytes_one_load': _min_free_ram_bytes()}
    reasons = []
    if found['listeners']:
        reasons.append(f'ports in use: {sorted({entry["port"] for entry in found["listeners"]})}')
    if found['llama_server_processes']:
        reasons.append('a llama-server process is running')
    if found['ollama_loaded_models']:
        reasons.append(f'Ollama has models loaded: {found["ollama_loaded_models"]}')
    found['refusals'] = reasons  # Free RAM is gated before each load (_await_free_ram), not here.
    return found


def _wait_healthy(base: str, server: subprocess.Popen, seconds: float = 240.0) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f'llama-server exited with {server.returncode} before it was ready')
        try:
            with urllib.request.urlopen(base + '/health', timeout=5) as reply:
                if json.loads(reply.read()).get('status') == 'ok':
                    return
        except (OSError, ValueError):
            pass
        time.sleep(1)
    raise RuntimeError('llama-server did not become healthy in time')


def _record_reference(pairs, log_path: Path) -> tuple[list[dict], list[dict], list[np.ndarray], dict]:
    """Full-vocabulary logprobs of llama-server at every response position (R40 incremental method).

    Every prefix 1..n-1 is requested in order with ``cache_prompt`` so the server
    evaluates exactly one new token per request, the computation the host
    performs; the full vector is requested only at response positions.
    """
    command = [str(LLAMA_DIR / 'llama-server.exe'), '-m', str(MODEL_PATH), '--host', HOST, '--port', str(SERVER_PORT),
               *SERVER_FLAGS]
    base = f'http://{HOST}:{SERVER_PORT}'
    lifecycle = {'command': command, 'ram_gate': _await_free_ram(_min_free_ram_bytes()), 'started_at': _now()}
    tokens, positions, vectors = [], [], []
    with log_path.open('w', encoding='utf-8') as log:
        server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        lifecycle['pid'] = server.pid
        try:
            _wait_healthy(base, server)
            lifecycle['ready_at'] = _now()
            for pair in pairs:
                tokens.append({
                    'prompt_ids': post_json(base, '/tokenize', {'content': pair['prompt_text'], 'add_special': True,
                                                                'parse_special': False})['tokens'],
                    'response_ids': post_json(base, '/tokenize', {'content': pair['response_text'],
                                                                  'add_special': False, 'parse_special': False})['tokens']})
            post_json(base, '/completion', REFERENCE_BODY | {'prompt': tokens[0]['prompt_ids'][:8],  # Warm-up.
                                                             'cache_prompt': False, 'n_probs': 0})
            for pair, ids in zip(pairs, tokens, strict=True):
                sequence, start = ids['prompt_ids'] + ids['response_ids'], len(ids['prompt_ids'])
                for t in range(1, len(sequence)):
                    full = t >= start
                    started = time.perf_counter()
                    reply = post_json(base, '/completion', REFERENCE_BODY | {
                        'prompt': sequence[:t], 'n_probs': REFERENCE_N_PROBS if full else 0})
                    seconds = time.perf_counter() - started
                    if reply['timings']['prompt_n'] != 1:
                        raise RuntimeError(f'llama-server evaluated {reply["timings"]["prompt_n"]} tokens at t={t}')
                    if full:
                        entries = reply['completion_probabilities'][0]['top_logprobs']
                        vectors.append(server_vector(entries, REFERENCE_N_PROBS))
                        positions.append({'pair': pair['name'], 'position': t - start, 't': t, 'target': sequence[t],
                                          'n_entries': len(entries), 'seconds': seconds})
                        print(f'reference {pair["name"]} t={t} entries={len(entries)} {seconds:.1f}s',
                              file=sys.stderr, flush=True)
        finally:
            server.terminate()
            try:
                server.wait(timeout=60)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=30)
            lifecycle |= {'stopped_at': _now(), 'exit_code': server.returncode}
    lifecycle['after_stop'] = {'listeners': _wait_until_free((SERVER_PORT,)),
                               'llama_server_processes': _processes(), 'checked_at': _now()}
    return tokens, positions, vectors, lifecycle


def _post_bytes(url: str, payload: dict) -> bytes:
    request = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'),
                                     headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=900) as reply:
            return reply.read()
    except urllib.error.HTTPError as error:
        raise RuntimeError(f'host answered HTTP {error.code}: {error.read()[:500]!r}') from None


def _host_phase(pairs, tokens, port: int) -> tuple[dict, dict, dict, dict]:
    """Load the host (after the reference server stopped), serve the pairs twice, then take logit rows twice."""
    lifecycle = {'ram_gate': _await_free_ram(_min_free_ram_bytes()), 'started_at': _now()}
    host = open_host()
    lifecycle |= {'model_digest': host.model_digest, 'n_vocab': host.n_vocab, **host.timings}
    try:
        tokenisation = {pair['name']: {
            'prompt_ids_match_server': host.llama.tokenize(pair['prompt_text']) == ids['prompt_ids'],
            'response_ids_match_server': host.llama.tokenize(pair['response_text'], add_special=False) == ids['response_ids']}
            for pair, ids in zip(pairs, tokens, strict=True)}
        server = make_server(host, port)
        thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': 0.2}, daemon=True)
        thread.start()
        lifecycle |= {'listening': f'{HOST}:{server.server_address[1]}', 'listening_at': _now()}
        runs = []
        try:
            for _ in range(2):
                run = {}
                for pair in pairs:
                    started = time.perf_counter()
                    body = _post_bytes(f'http://{HOST}:{port}{ENDPOINT_PATH}', {
                        'model_digest': MODEL_DIGEST, 'prompt_text': pair['prompt_text'],
                        'response_text': pair['response_text'], 'top_k_keep': CHECK_TOP_K})
                    run[pair['name']] = {'body': body, 'seconds': time.perf_counter() - started}
                runs.append(run)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(30)
            lifecycle['server_stopped_at'] = _now()
        rows = {}
        for pair, ids in zip(pairs, tokens, strict=True):
            sequence, first = ids['prompt_ids'] + ids['response_ids'], len(ids['prompt_ids']) - 1
            rows[pair['name']] = [np.stack([row for _, row in host.llama.incremental_logits(
                sequence[:-1], rows=range(first, len(sequence) - 1))]) for _ in range(2)]
    finally:
        host.close()
        lifecycle['model_closed_at'] = _now()
    lifecycle['after_stop'] = {'listeners': _wait_until_free((port,)), 'checked_at': _now()}
    return tokenisation, runs, rows, lifecycle


def _host_phase_batch(items, pairs, tokens, port: int) -> tuple[dict, list, dict, dict, dict]:
    """Load the host, serve every item on the batch route twice and every pair once on the single route,
    then take the batch computation's logit rows twice (the server stopped before the load)."""
    lifecycle = {'ram_gate': _await_free_ram(_min_free_ram_bytes()), 'started_at': _now()}
    host = open_host()
    lifecycle |= {'model_digest': host.model_digest, 'n_vocab': host.n_vocab, **host.timings}
    ids = {pair['name']: entry for pair, entry in zip(pairs, tokens, strict=True)}
    try:
        tokenisation = {pair['name']: {
            'prompt_ids_match_server': host.llama.tokenize(pair['prompt_text']) == ids[pair['name']]['prompt_ids'],
            'response_ids_match_server': host.llama.tokenize(pair['response_text'], add_special=False)
            == ids[pair['name']]['response_ids']} for pair in pairs}
        server = make_server(host, port)
        thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': 0.2}, daemon=True)
        thread.start()
        lifecycle |= {'listening': f'{HOST}:{server.server_address[1]}', 'listening_at': _now()}
        runs, singles = [], {}
        try:
            for _ in range(2):
                run = {}
                for item in items:
                    started = time.perf_counter()
                    body = _post_bytes(f'http://{HOST}:{port}{BATCH_ENDPOINT_PATH}', {
                        'model_digest': MODEL_DIGEST, 'prompt_text': item['prompt_text'],
                        'response_texts': item['response_texts'], 'top_k_keep': CHECK_TOP_K})
                    run[item['name']] = {'body': body, 'seconds': time.perf_counter() - started,
                                         'host_timings': host.last_batch_timings}
                runs.append(run)
            for pair in pairs:
                started = time.perf_counter()
                body = _post_bytes(f'http://{HOST}:{port}{ENDPOINT_PATH}', {
                    'model_digest': MODEL_DIGEST, 'prompt_text': pair['prompt_text'],
                    'response_text': pair['response_text'], 'top_k_keep': CHECK_TOP_K})
                singles[pair['name']] = {'body': body, 'seconds': time.perf_counter() - started}
        finally:
            server.shutdown()
            server.server_close()
            thread.join(30)
            lifecycle['server_stopped_at'] = _now()
        rows = {}
        for item in items:
            members = [pair['name'] for pair in pairs if pair['item'] == item['name']]
            twice = [host.batch_logit_rows(ids[members[0]]['prompt_ids'], [ids[name]['response_ids'] for name in members])
                     for _ in range(2)]
            rows |= {name: [twice[0][k], twice[1][k]] for k, name in enumerate(members)}
    finally:
        host.close()
        lifecycle['model_closed_at'] = _now()
    lifecycle['after_stop'] = {'listeners': _wait_until_free((port,)), 'checked_at': _now()}
    return tokenisation, runs, singles, rows, lifecycle


def _per_item_timing(runs, singles) -> dict:
    """Request time per batch (with the host's prompt and per-response times) and per single-route pair."""
    return {'batch': [{'run': index, 'item': name, 'request_seconds': entry['seconds'], **(entry['host_timings'] or {})}
                      for index, run in enumerate(runs) for name, entry in run.items()],
            'single': [{'pair': name, 'request_seconds': entry['seconds']} for name, entry in singles.items()]}


def _batch_summary_lines(receipt: dict) -> list[str]:
    lines = ['## Per-item timing (batch route)', '',
             '| Run | Item | Responses | Prompt ids | Prompt s | Response s | Request s |', '|---|---|---|---|---|---|---|']
    for entry in receipt.get('per_item_timing', {}).get('batch', []):
        each = ', '.join(f'{item["seconds"]:.2f}' for item in entry.get('responses', []))
        lines.append(f'| {entry["run"]} | {entry["item"]} | {len(entry.get("responses", []))} | '
                     f'{entry.get("prompt_ids")} | {entry.get("prompt_seconds", 0):.2f} | {each} | '
                     f'{entry["request_seconds"]:.2f} |')
    return [*lines, '']


def _write_summary(path: Path, receipt: dict) -> None:
    comparison, outcome = receipt.get('comparison', {}), receipt['outcome']
    worst = comparison.get('max_abs_diff', {})
    batch = receipt.get('mode') == 'batch'
    title = ('# Full-distribution host exactness check, batch mode (task 6.4)' if batch
             else '# Full-distribution host exactness check (task 6.1)')
    recorded = receipt.get('reference_recording') or {}
    vectors = recorded.get('vectors') or {}
    lines = [title, '',
             f'- Outcome: **{"passed" if outcome["passed"] else "failed"}**'
             + (f' ({"; ".join(outcome["failures"])})' if outcome['failures'] else ''),
             f'- Started {receipt["started_at"]}, finished {receipt.get("finished_at")}',
             f'- Model: `{receipt["model"]["digest"]}` at `{receipt["model"]["path"]}`',
             f'- llama.cpp: {"; ".join(receipt.get("build", {}).get("llama_cpp", []))}',
             f'- Reference: llama-server on {HOST}:{SERVER_PORT}, flags `{" ".join(SERVER_FLAGS)}`, stopped before the '
             f'host loaded (one model resident at a time)',
             *([f'- Reference recorded by a separate pass from {recorded.get("started_at")} to '
                f'{recorded.get("finished_at")} into `{recorded.get("file")}` (vectors `{vectors.get("file")}`, '
                f'{vectors.get("bytes")} bytes, {vectors.get("dtype")}, sha256 `{vectors.get("sha256")}`); this '
                'pass loaded only the host'] if receipt.get('passes') == 'two' and recorded else []),
             f'- Host: {HOST}:{receipt["settings"]["host_port"]}, top_k_keep {CHECK_TOP_K}, '
             'prompt add_special=True, response add_special=False, parse_special=False, no chat template', '']
    if batch:
        lines[-1:-1] = [f'- Batch route `{BATCH_ENDPOINT_PATH}`: the prompt is decoded once per item, each response '
                        f'continues on the cached prompt and the KV cache is rolled back with `{KV_ROLLBACK}` after '
                        f'each; items: {", ".join(item["name"] for item in receipt["items"])}']
    if comparison:
        lines += ['| Measure | Value |', '|---|---|',
                  f'| Positions compared | {comparison["positions_checked"]} |',
                  *[f'| Max abs difference, {name} | {value:.3g} |' for name, value in worst.items()],
                  f'| Rank mismatches | {comparison["rank_mismatches"]} |',
                  f'| Missing server entries | {comparison["missing_entries"]} |',
                  f'| Min top-{CHECK_TOP_K} id overlap with the server | {comparison["min_top_k_id_overlap"]} |',
                  f'| Within {TOLERANCE:g} after float64 renormalisation | {comparison["within_tolerance"]} |',
                  f'| Two host runs bitwise identical | {receipt["repeatability"]["all_identical"]} |', '']
        if batch:
            lines[-1:-1] = [f'| Batch route equals the single route on every pair | '
                            f'{all(receipt.get("batch_matches_single", {}).values())} |']
            lines += _batch_summary_lines(receipt)
    teardown = receipt.get('teardown', {})
    lines += [f'- Teardown: listeners on {teardown.get("ports")}: {teardown.get("listeners")}; llama-server processes: '
              f'{teardown.get("llama_server_processes")}; other TCP states on those ports: '
              f'{teardown.get("other_tcp_states")}', '',
              'The host scores prompt and response tokenised separately; that boundary can differ from the '
              "generation's own tokenisation of the same text.", '']
    path.write_text('\n'.join(lines), encoding='utf-8')


# ---------- the two passes (user correction 2026-11-10: never two models loaded) ----------

RECORDING_SCHEMA = 'finance-sim-full-distribution-reference/1'
RECORDING_NAME = 'reference-recording.json'
RECORDING_VECTORS = 'reference-vectors.npz'
PASS_MODES = ('record', 'compare')


class RecordingError(ValueError):
    """A reference recording the compare pass cannot use (missing, failed, mismatched or altered)."""


def _check_pairs(batch: bool) -> list[dict]:
    return batch_reference_pairs(BATCH_EXACTNESS_ITEMS) if batch else [dict(pair) for pair in EXACTNESS_PAIRS]


def _plain(value):
    """``value`` as it reads back from JSON (tuples become lists), so a recording compares equal."""
    return json.loads(json.dumps(value))


def _model_identity() -> dict:
    return {'digest': MODEL_DIGEST, 'path': str(MODEL_PATH.relative_to(ROOT))}


def _reference_settings() -> dict:
    return _plain({'reference_port': SERVER_PORT, 'server_flags': SERVER_FLAGS, 'reference_request': REFERENCE_BODY,
                   'n_probs': REFERENCE_N_PROBS})


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_recording(output: Path, recording: Mapping, vectors) -> dict:
    """Write the reference vectors (``RECORDING_VECTORS``) and the recording (``RECORDING_NAME``); return the recording.

    llama-server's logprobs are float32 values, so the stacked vectors are stored as float32 when
    that is exact (NaN for absent ids included) and as float64 otherwise; the vector file's size
    and SHA-256 go into the recording. Without vectors any older vector file is removed, so a
    stale file never pairs with a new recording.
    """
    output.mkdir(parents=True, exist_ok=True)
    recording = dict(recording)
    path = output / RECORDING_VECTORS
    if len(vectors):
        stack = np.stack([np.asarray(vector, dtype=np.float64) for vector in vectors])
        with np.errstate(over='ignore'):  # A value beyond float32 becomes inf and keeps float64.
            narrow = stack.astype(np.float32)
        stored = narrow if np.array_equal(narrow.astype(np.float64), stack, equal_nan=True) else stack
        with path.open('wb') as handle:
            np.savez_compressed(handle, vectors=stored)
        recording['vectors'] = {'file': RECORDING_VECTORS, 'sha256': _file_sha256(path), 'bytes': path.stat().st_size,
                                'shape': list(stack.shape), 'dtype': str(stored.dtype)}
    else:
        path.unlink(missing_ok=True)
        recording['vectors'] = None
    (output / RECORDING_NAME).write_text(json.dumps(recording, indent=1) + '\n', encoding='utf-8')
    return recording


def load_recorded_vectors(output: Path, recording: Mapping) -> list[np.ndarray]:
    """The recording's reference vectors as float64, after checking the file's SHA-256, shape and dtype."""
    meta = recording.get('vectors')
    if not isinstance(meta, Mapping):
        raise RecordingError('the recording holds no reference vectors')
    name = meta.get('file')
    if not isinstance(name, str) or Path(name).name != name:
        raise RecordingError('the recording names no plain vector file')
    path = output / name
    if not path.is_file():
        raise RecordingError(f'the vector file {name} is missing')
    if _file_sha256(path) != meta.get('sha256'):
        raise RecordingError(f'the vector file {name} does not match its recorded sha256')
    with np.load(path, allow_pickle=False) as data:
        stack = data['vectors']
    if list(stack.shape) != meta.get('shape') or str(stack.dtype) != meta.get('dtype'):
        raise RecordingError(f'the vector file {name} differs from its recorded shape or dtype')
    return [row.astype(np.float64) for row in stack]


def read_recording(output: Path, *, pairs, batch: bool) -> tuple[dict, list[np.ndarray]]:
    """The record pass's recording for these pairs, with its vectors; RecordingError when it cannot be used."""
    path = output / RECORDING_NAME
    try:
        recording = json.loads(path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        raise RecordingError(f'no reference recording at {path}; run the record pass first') from None
    except (OSError, ValueError) as error:
        raise RecordingError(f'the reference recording {path} is unreadable ({type(error).__name__})') from None
    if not isinstance(recording, dict) or recording.get('schema') != RECORDING_SCHEMA:
        raise RecordingError(f'{path.name} is not a {RECORDING_SCHEMA} recording')
    outcome = recording.get('outcome') or {}
    if not outcome.get('passed'):
        raise RecordingError(f'the record pass did not pass ({"; ".join(outcome.get("failures") or ["no outcome"])})')
    expected = {'mode': 'batch' if batch else 'single', 'model': _model_identity(), 'settings': _reference_settings(),
                'pairs': _plain([dict(pair) for pair in pairs])}
    for key, value in expected.items():
        if recording.get(key) != value:
            raise RecordingError(f"the recording's {key} differs from this check's")
    tokens, positions = recording.get('tokens'), recording.get('positions')
    if not isinstance(tokens, list) or len(tokens) != len(pairs) or not isinstance(positions, list):
        raise RecordingError('the recording does not hold the server tokens and positions of every pair')
    vectors = load_recorded_vectors(output, recording)
    if len(vectors) != len(positions):
        raise RecordingError(f'the recording holds {len(vectors)} vectors for {len(positions)} positions')
    return recording, vectors


def record_reference_pass(output: Path, *, port: int = PORT, batch: bool = False) -> int:
    """Pass A: start only llama-server on 127.0.0.1:11560, record its full-vocabulary reference for
    the pairs, stop it and verify that no listener and no llama-server process remains; write the
    recording (``RECORDING_NAME`` plus ``RECORDING_VECTORS``) into ``output``. The host is never
    loaded here. 0 when recorded and torn down cleanly, 3 when the preflight refuses."""
    output.mkdir(parents=True, exist_ok=True)  # Before the server log is opened in it.
    pairs = _check_pairs(batch)
    recording = {'schema': RECORDING_SCHEMA, 'task': '6.4' if batch else '6.1', 'pass': 'record',
                 'mode': 'batch' if batch else 'single', 'started_at': _now(), 'model': _model_identity(),
                 'settings': _reference_settings(), 'pairs': [dict(pair) for pair in pairs]}
    failures, vectors = [], []
    recording['preflight'] = preflight = _preflight(port)
    if preflight['refusals']:
        failures = [f'preflight: {reason}' for reason in preflight['refusals']]
        recording |= {'finished_at': _now(), 'outcome': {'passed': False, 'refused': True, 'failures': failures}}
        write_recording(output, recording, [])
        print(json.dumps({'recording': str(output / RECORDING_NAME), **recording['outcome']}))
        return 3
    try:
        recording['build'] = build_info()
        tokens, positions, vectors, recording['reference_lifecycle'] = _record_reference(
            pairs, output / 'reference-llama-server.log')
        recording |= {'tokens': tokens, 'positions': positions}
        if recording['reference_lifecycle']['after_stop']['listeners'] or _processes():
            failures.append('the reference llama-server did not stop cleanly')
    except Exception as error:  # Recorded; the server is already stopped by _record_reference.
        failures.append(f'error: {type(error).__name__}: {error}')
    recording['teardown'] = teardown = {'ports': [SERVER_PORT], 'listeners': _wait_until_free((SERVER_PORT,)),
                                        'llama_server_processes': _processes(), 'checked_at': _now()}
    if teardown['listeners'] or teardown['llama_server_processes']:
        failures.append('a listener or llama-server process remains after the record pass')
    recording |= {'finished_at': _now(), 'outcome': {'passed': not failures, 'refused': False, 'failures': failures}}
    recording = write_recording(output, recording, [] if failures else vectors)
    print(json.dumps({'recording': str(output / RECORDING_NAME), 'vectors': recording['vectors'],
                      **recording['outcome']}))
    return 0 if not failures else 1


def _base_receipt(pairs, *, port: int, batch: bool, passes: str) -> dict:
    receipt = {'schema': 'finance-sim-full-distribution-exactness/1', 'task': '6.4' if batch else '6.1',
               'started_at': _now(), 'passes': passes, 'model': _model_identity(),
               'settings': {'reference_port': SERVER_PORT, 'host_port': port, 'server_flags': SERVER_FLAGS,
                            'reference_request': REFERENCE_BODY, 'top_k_keep': CHECK_TOP_K, 'tolerance': TOLERANCE,
                            'tokenisation': {'prompt': {'add_special': True, 'parse_special': False},
                                             'response': {'add_special': False, 'parse_special': False},
                                             'template': 'none'}},
               'pairs': [dict(pair) for pair in pairs]}
    if batch:
        receipt['mode'] = 'batch'
        receipt['settings'] |= {'batch_route': BATCH_ENDPOINT_PATH, 'kv_rollback': KV_ROLLBACK,
                                'prompt_decode': 'once per item, one token per llama_decode as on the single route'}
        receipt['items'] = [dict(item) for item in BATCH_EXACTNESS_ITEMS]
    return receipt


def check_against_server(output: Path, *, port: int = PORT, batch: bool = False, mode: str | None = None) -> int:
    """Run the exactness check (on the batch route when ``batch``); 0 when it passes.

    ``mode='record'`` runs pass A only (llama-server, written to the recording in ``output``);
    ``mode='compare'`` runs pass B only (the host against that recording; ``receipt.json`` and
    ``summary.md``). Without ``mode`` both passes run one after the other in this process, the
    comparison again reading the recording from the file.
    """
    if mode not in (None, *PASS_MODES):
        raise ValueError(f'mode must be one of {PASS_MODES} or None')
    if mode == 'record':
        return record_reference_pass(output, port=port, batch=batch)
    if mode == 'compare':
        return compare_pass(output, port=port, batch=batch, passes='two')
    code = record_reference_pass(output, port=port, batch=batch)
    if code:
        receipt = _base_receipt(_check_pairs(batch), port=port, batch=batch, passes='one process')
        recorded = json.loads((output / RECORDING_NAME).read_text(encoding='utf-8'))['outcome']
        receipt['outcome'] = {'passed': False, 'refused': recorded['refused'],
                              'failures': [f'record pass: {failure}' for failure in recorded['failures']]}
        return _finish(output, receipt, code)
    return compare_pass(output, port=port, batch=batch, passes='one process')


def compare_pass(output: Path, *, port: int = PORT, batch: bool = False, passes: str = 'two') -> int:
    """Pass B: with no llama-server resident, load only the host, serve the pairs (on the batch route
    when ``batch``) and compare against the record pass's recording in ``output``; write
    ``receipt.json`` and ``summary.md``. 0 when it passes, 3 when the recording or the preflight refuses."""
    output.mkdir(parents=True, exist_ok=True)
    pairs = _check_pairs(batch)
    receipt = _base_receipt(pairs, port=port, batch=batch, passes=passes)
    failures = []
    try:
        recording, vectors = read_recording(output, pairs=pairs, batch=batch)
    except RecordingError as error:
        receipt['outcome'] = {'passed': False, 'refused': True, 'failures': [f'recording: {error}']}
        return _finish(output, receipt, 3)
    receipt['reference_recording'] = {key: recording.get(key) for key in (
        'started_at', 'finished_at', 'outcome', 'vectors', 'preflight', 'teardown', 'build')} | {'file': RECORDING_NAME}
    receipt['reference_lifecycle'] = recording.get('reference_lifecycle')
    tokens, positions = recording['tokens'], recording['positions']
    for pair, ids in zip(receipt['pairs'], tokens, strict=True):
        pair |= {'server_prompt_ids': ids['prompt_ids'], 'server_response_ids': ids['response_ids']}
    receipt['preflight'] = preflight = _preflight(port)
    if preflight['refusals']:
        failures += [f'preflight: {reason}' for reason in preflight['refusals']]
        receipt['outcome'] = {'passed': False, 'refused': True, 'failures': failures}
        return _finish(output, receipt, 3)
    try:
        receipt['build'] = build_info()
        if receipt['build'] != recording.get('build'):
            raise RuntimeError('the llama.cpp build differs from the record pass')
        if batch:
            tokenisation, runs, singles, rows, receipt['host_lifecycle'] = _host_phase_batch(
                BATCH_EXACTNESS_ITEMS, pairs, tokens, port)
            replies = {name: json.loads(entry['body'])['evidence'] for name, entry in runs[0].items()}
            evidence = {pair['name']: replies[pair['item']][pair['index']] for pair in pairs}
        else:
            tokenisation, runs, rows, receipt['host_lifecycle'] = _host_phase(EXACTNESS_PAIRS, tokens, port)
            evidence = {name: json.loads(entry['body']) for name, entry in runs[0].items()}
        receipt['tokenisation_agreement'] = tokenisation
        n_vocab = receipt['host_lifecycle']['n_vocab']
        compared, cursor = [], {pair['name']: 0 for pair in pairs}
        for meta, vector in zip(positions, vectors, strict=True):
            name, j = meta['pair'], meta['position']
            assert cursor[name] == j
            cursor[name] += 1
            record = evidence[name]
            if np.isfinite(vector[n_vocab:]).any():
                raise RuntimeError('llama-server reported token ids beyond the host vocabulary')
            result = compare_position(rows[name][0][j], vector[:n_vocab], meta['target'], (
                record['logprob'][j], record['entropy'][j], record['rank'][j], record['top_logprobs'][j]))
            compared.append(meta | {'n_vocab': n_vocab, 'full_vocabulary': meta['n_entries'] >= n_vocab} | result)
        receipt['reference_positions'] = compared
        receipt['comparison'] = summary = summarise(compared)
        receipt['repeatability'] = repeat = {
            'evidence_bytes_identical': {name: runs[0][name]['body'] == runs[1][name]['body'] for name in runs[0]},
            'logit_rows_bitwise_identical': {name: bool(np.array_equal(a, b)) for name, (a, b) in rows.items()},
            'evidence_sha256': {name: hashlib.sha256(runs[0][name]['body']).hexdigest() for name in runs[0]},
            'request_seconds': [{name: entry['seconds'] for name, entry in run.items()} for run in runs]}
        repeat['all_identical'] = all(repeat['evidence_bytes_identical'].values()) and all(
            repeat['logit_rows_bitwise_identical'].values())
        receipt['evidence_lengths'] = {name: len(record['logprob']) for name, record in evidence.items()}
        if not all(all(entry.values()) for entry in tokenisation.values()):
            failures.append('host tokenisation differs from llama-server /tokenize')
        if any(record['response_token_ids'] != pair['server_response_ids'] or len(record['logprob']) != len(
                pair['server_response_ids']) for record, pair in zip(evidence.values(), receipt['pairs'], strict=True)):
            failures.append('evidence does not cover exactly the response ids')
        if not all(meta['full_vocabulary'] for meta in compared):
            failures.append('llama-server returned fewer entries than the vocabulary')
        if not summary['within_tolerance']:
            failures.append(f'host differs from llama-server beyond {TOLERANCE:g} or in rank')
        if not repeat['all_identical']:
            failures.append('two host runs are not bitwise identical')
        if batch:
            receipt['batch_matches_single'] = matches = {
                pair['name']: json.loads(singles[pair['name']]['body']) == evidence[pair['name']] for pair in pairs}
            receipt['shared_prompt_ids_identical'] = shared = all(
                len({tuple(pair['server_prompt_ids']) for pair in receipt['pairs'] if pair['item'] == item['name']}) == 1
                for item in BATCH_EXACTNESS_ITEMS)
            receipt['per_item_timing'] = _per_item_timing(runs, singles)
            if not all(matches.values()):
                failures.append('the batch route differs from the single route on the same pairs')
            if not shared:
                failures.append('llama-server tokenised a shared prompt differently across its responses')
    except Exception as error:  # Recorded in the receipt; every started process is already stopped.
        failures.append(f'error: {type(error).__name__}: {error}')
    ports = (SERVER_PORT, port)
    receipt['teardown'] = teardown = {'ports': list(ports), 'listeners': _wait_until_free(ports),
                                      'llama_server_processes': _processes(), 'other_tcp_states': _other_states(ports),
                                      'checked_at': _now()}
    if teardown['listeners'] or teardown['llama_server_processes']:
        failures.append('a listener or llama-server process remains after teardown')
    receipt['outcome'] = {'passed': not failures, 'refused': False, 'failures': failures}
    return _finish(output, receipt, 0 if not failures else 1)


def _finish(output: Path, receipt: dict, code: int) -> int:
    receipt['finished_at'] = _now()
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=1) + '\n', encoding='utf-8')
    _write_summary(output / 'summary.md', receipt)
    print(json.dumps({'receipt': str(output / 'receipt.json'), **receipt['outcome']}))
    return code


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--port', type=int, default=PORT, help='loopback port (default %(default)s)')
    parser.add_argument('--memory-floor-bytes', type=int, default=None,
                        help="refuse new requests with 503 memory_low below this free physical memory (default: "
                             "the llamacpp-gemma3-4b profile's memory floor)")
    parser.add_argument('--check-against-server', action='store_true',
                        help='run the exactness check against llama-server instead of serving')
    parser.add_argument('--batch', action='store_true',
                        help='with --check-against-server: check the batch route (task 6.4)')
    passes = parser.add_mutually_exclusive_group()
    passes.add_argument('--record', action='store_true',
                        help='with --check-against-server: pass A only: start llama-server, write its reference to '
                             f'{RECORDING_NAME} and {RECORDING_VECTORS} in the output folder, stop it, verify it is gone')
    passes.add_argument('--compare', action='store_true',
                        help='with --check-against-server: pass B only: load the host and compare against the '
                             'recording of pass A in the output folder')
    parser.add_argument('--output', default=None,
                        help='receipt and recording folder for --check-against-server (default: exactness/, or '
                             'exactness/batch/)')
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error('--port must be between 1 and 65535')
    if args.memory_floor_bytes is not None and args.memory_floor_bytes < 1:
        parser.error('--memory-floor-bytes must be a positive number of bytes')
    if args.batch and not args.check_against_server:
        parser.error('--batch is a mode of --check-against-server')
    if (args.record or args.compare) and not args.check_against_server:
        parser.error('--record and --compare are passes of --check-against-server')
    if args.check_against_server:
        output = Path(args.output) if args.output else (BATCH_RECEIPT_DIR if args.batch else RECEIPT_DIR)
        mode = 'record' if args.record else 'compare' if args.compare else None
        return check_against_server(output, port=args.port, batch=args.batch, mode=mode)
    return serve(args.port, memory_floor_bytes=args.memory_floor_bytes)


if __name__ == '__main__':
    sys.exit(main())
