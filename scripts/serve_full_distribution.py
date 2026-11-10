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

Errors are JSON ``{"error": {"code", "message"}}``: 409 ``digest_mismatch``;
413 ``oversize`` (prompt plus response ids beyond the 2048-token context, or a
body beyond 1 MiB); 422 ``tokenisation_failed`` (a text fails to tokenise, or
the response, or the prompt, tokenises to no ids); 400 ``malformed``; 404
``not_found``; 500 ``internal_error``.

``--check-against-server`` runs the exactness check instead of serving: it
starts llama-server b12199 on 127.0.0.1:11560 (the R40 reference method),
records its full-vocabulary logprobs for every response position of two fixed
synthetic prompt/response pairs, stops it and verifies the port is free, then
loads this host (so one model is resident at a time), serves the pairs twice on
127.0.0.1:11580 and compares within 1e-4 after a float64 renormalisation of
the server's float32 normaliser, with identical ranks and bitwise identical
runs. The private receipt goes to runs/finance-sim/hosts/full-distribution/exactness/.
"""
from __future__ import annotations

import argparse
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


def parse_request(body: bytes) -> dict:
    """Decode a request body (UTF-8 JSON object, no duplicate keys) and validate its fields."""
    try:
        data = json.loads(body.decode('utf-8'), object_pairs_hook=_unique_keys)
    except (UnicodeDecodeError, ValueError):
        raise HostError(400, 'malformed', 'the body must be a UTF-8 JSON object without duplicate keys') from None
    return validate_request(data)


def validate_request(data) -> dict:
    """Exactly the four request fields: three strings and a positive integer ``top_k_keep``."""
    if not isinstance(data, Mapping):
        raise HostError(400, 'malformed', 'the body must be a JSON object')
    missing = [name for name in REQUEST_FIELDS if name not in data]
    unknown = len(set(data) - set(REQUEST_FIELDS))
    if missing or unknown:
        raise HostError(400, 'malformed', f'expected exactly {list(REQUEST_FIELDS)}; missing {missing}, '
                                          f'{unknown} unknown field(s)')
    for name in REQUEST_FIELDS[:3]:
        if not isinstance(data[name], str):
            raise HostError(400, 'malformed', f'{name} must be a string')
    top_k_keep = data['top_k_keep']
    if isinstance(top_k_keep, bool) or not isinstance(top_k_keep, int) or top_k_keep < 1:
        raise HostError(400, 'malformed', 'top_k_keep must be a positive integer')
    return {name: data[name] for name in REQUEST_FIELDS}


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


class FullDistributionHost:
    """One loaded model and its teacher-forcing evidence; one caller at a time.

    ``llama`` provides ``tokenize(text, add_special, parse_special)``,
    ``incremental_logits(tokens, rows)``, ``n_vocab`` and ``close()``
    (``scripts.llamacpp_native.Llama``, or a fake in tests).
    """

    def __init__(self, llama, *, model_digest: str, n_ctx: int = N_CTX) -> None:
        self.llama, self.model_digest, self.n_ctx = llama, model_digest, n_ctx
        self.n_vocab = int(llama.n_vocab)
        self.timings: dict[str, float] = {}
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
        payload = {'model_digest': self.model_digest, 'n_vocab': self.n_vocab,
                   'response_token_ids': response_ids, **columns}
        try:
            FullDistributionEvidence.from_dict(payload)
        except (TypeError, ValueError) as error:
            raise HostError(500, 'internal_error', f'the evidence fails its contract: {error}') from None
        return payload

    def respond(self, body: bytes) -> tuple[int, dict]:
        """HTTP status and JSON payload for one request body; never raises."""
        try:
            return 200, self.evidence(parse_request(body))
        except HostError as error:
            return error.status, error_body(error.code, error.message)
        except Exception as error:  # A failed decode answers 500 instead of stopping the serial server.
            return 500, error_body('internal_error', f'{type(error).__name__}: {str(error)[:200]}')


class _Handler(BaseHTTPRequestHandler):
    server_version = 'recall-guard-full-distribution/1'
    timeout = SOCKET_TIMEOUT_S

    def do_POST(self) -> None:
        if self.path != ENDPOINT_PATH:
            self._send(404, error_body('not_found', f'only POST {ENDPOINT_PATH} is served'))
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
        self._send(*self.server.full_distribution_host.respond(body))

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


def serve(port: int = PORT) -> int:
    """Reserve the port, start the host, then listen until interrupted."""
    try:
        server = make_server(None, port, activate=False)
    except OSError as error:
        print(f'refused to start: port_unavailable: {HOST}:{port}: {error.strerror or error}', file=sys.stderr)
        return 2
    try:
        host = open_host()
    except StartRefused as error:
        server.server_close()
        print(f'refused to start: {error}', file=sys.stderr)
        return 2
    server.full_distribution_host = host
    try:
        server.server_activate()
        print(json.dumps({'listening': f'{HOST}:{server.server_address[1]}', 'endpoint': ENDPOINT_PATH,
                          'model_digest': host.model_digest, 'n_vocab': host.n_vocab, **host.timings}), flush=True)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
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
OTHER_HOST_PORTS = (11570, 11571, 11581)  # The 6.2/6.3 llama.cpp hosts and gateways must not be resident.
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


def _write_summary(path: Path, receipt: dict) -> None:
    comparison, outcome = receipt.get('comparison', {}), receipt['outcome']
    worst = comparison.get('max_abs_diff', {})
    lines = ['# Full-distribution host exactness check (task 6.1)', '',
             f'- Outcome: **{"passed" if outcome["passed"] else "failed"}**'
             + (f' ({"; ".join(outcome["failures"])})' if outcome['failures'] else ''),
             f'- Started {receipt["started_at"]}, finished {receipt.get("finished_at")}',
             f'- Model: `{receipt["model"]["digest"]}` at `{receipt["model"]["path"]}`',
             f'- llama.cpp: {"; ".join(receipt.get("build", {}).get("llama_cpp", []))}',
             f'- Reference: llama-server on {HOST}:{SERVER_PORT}, flags `{" ".join(SERVER_FLAGS)}`, stopped before the '
             f'host loaded (one model resident at a time)',
             f'- Host: {HOST}:{receipt["settings"]["host_port"]}, top_k_keep {CHECK_TOP_K}, '
             'prompt add_special=True, response add_special=False, parse_special=False, no chat template', '']
    if comparison:
        lines += ['| Measure | Value |', '|---|---|',
                  f'| Positions compared | {comparison["positions_checked"]} |',
                  *[f'| Max abs difference, {name} | {value:.3g} |' for name, value in worst.items()],
                  f'| Rank mismatches | {comparison["rank_mismatches"]} |',
                  f'| Missing server entries | {comparison["missing_entries"]} |',
                  f'| Min top-{CHECK_TOP_K} id overlap with the server | {comparison["min_top_k_id_overlap"]} |',
                  f'| Within {TOLERANCE:g} after float64 renormalisation | {comparison["within_tolerance"]} |',
                  f'| Two host runs bitwise identical | {receipt["repeatability"]["all_identical"]} |', '']
    teardown = receipt.get('teardown', {})
    lines += [f'- Teardown: listeners on {teardown.get("ports")}: {teardown.get("listeners")}; llama-server processes: '
              f'{teardown.get("llama_server_processes")}; other TCP states on those ports: '
              f'{teardown.get("other_tcp_states")}', '',
              'The host scores prompt and response tokenised separately; that boundary can differ from the '
              "generation's own tokenisation of the same text.", '']
    path.write_text('\n'.join(lines), encoding='utf-8')


def check_against_server(output: Path, *, port: int = PORT) -> int:
    """Run the exactness check and write ``receipt.json`` and ``summary.md``; 0 when it passes."""
    output.mkdir(parents=True, exist_ok=True)
    receipt = {'schema': 'finance-sim-full-distribution-exactness/1', 'task': '6.1', 'started_at': _now(),
               'model': {'digest': MODEL_DIGEST, 'path': str(MODEL_PATH.relative_to(ROOT))},
               'settings': {'reference_port': SERVER_PORT, 'host_port': port, 'server_flags': SERVER_FLAGS,
                            'reference_request': REFERENCE_BODY, 'top_k_keep': CHECK_TOP_K, 'tolerance': TOLERANCE,
                            'tokenisation': {'prompt': {'add_special': True, 'parse_special': False},
                                             'response': {'add_special': False, 'parse_special': False},
                                             'template': 'none'}},
               'pairs': [dict(pair) for pair in EXACTNESS_PAIRS]}
    failures = []
    receipt['preflight'] = preflight = _preflight(port)
    if preflight['refusals']:
        failures += [f'preflight: {reason}' for reason in preflight['refusals']]
        receipt['outcome'] = {'passed': False, 'refused': True, 'failures': failures}
        return _finish(output, receipt, 3)
    try:
        receipt['build'] = build_info()
        tokens, positions, vectors, receipt['reference_lifecycle'] = _record_reference(
            EXACTNESS_PAIRS, output / 'reference-llama-server.log')
        for pair, ids in zip(receipt['pairs'], tokens, strict=True):
            pair |= {'server_prompt_ids': ids['prompt_ids'], 'server_response_ids': ids['response_ids']}
        if receipt['reference_lifecycle']['after_stop']['listeners'] or _processes():
            raise RuntimeError('the reference llama-server did not stop cleanly')
        tokenisation, runs, rows, receipt['host_lifecycle'] = _host_phase(EXACTNESS_PAIRS, tokens, port)
        receipt['tokenisation_agreement'] = tokenisation
        n_vocab = receipt['host_lifecycle']['n_vocab']
        compared, cursor = [], {pair['name']: 0 for pair in EXACTNESS_PAIRS}
        evidence = {name: json.loads(entry['body']) for name, entry in runs[0].items()}
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
    parser.add_argument('--check-against-server', action='store_true',
                        help='run the exactness check against llama-server instead of serving')
    parser.add_argument('--output', default=str(RECEIPT_DIR), help='receipt folder for --check-against-server')
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error('--port must be between 1 and 65535')
    if args.check_against_server:
        return check_against_server(Path(args.output), port=args.port)
    return serve(args.port)


if __name__ == '__main__':
    sys.exit(main())
