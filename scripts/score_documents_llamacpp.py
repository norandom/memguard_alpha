"""Teacher-forced per-token scores of EDGAR documents under a local GGUF (R37).

Host-side tooling, run NATIVELY on Windows with the project's .venv Python
(stdlib + numpy only; no Recall Guard import). The model runs on the iGPU
through the user-installed llama.cpp build (winget, b12199, Vulkan).

Route (R37): llama-server b12199 cannot return prompt-token probabilities
(``echo`` on /v1/completions returns nothing per prompt token), and one
full-vocabulary ``n_probs`` JSON response (about 27 MB) per prefix takes 2-8 s,
so 79k positions would take days. This script therefore drives the same
``llama.dll`` in-process through ctypes, with the server's settings
(``-ngl 99 -dev Vulkan0 -fa off --swa-full``, n_ctx 2048, n_batch 2048, n_ubatch
512, one sequence) and one warm-up decode after load.

Default route ``incremental``: one token per ``llama_decode`` on a growing KV
cache, i.e. exactly what llama-server computes for ``cache_prompt`` requests
with ``n_predict`` 1 per prefix. ``verify`` shows it matches that server
computation within 1e-6 (logprob/entropy; 7e-6 over the full vector, identical
ranks) once the server's float32 normaliser is redone in float64. The faster
``batched`` route (one decode per document, an output row per position) differs
from every server computation by up to about 0.1 nats, because Vulkan kernels
change with the batch shape, so it is kept only for comparison.
llama-server runs only as the independent reference in ``reference``.

For each position t >= 1 of the tokenized raw text (BOS added as the model
requires, no chat template, special tokens not parsed) the FULL distribution at
position t-1 gives, in float64: the logprob of the actual token t, the entropy
(nats) and the rank of token t (1 + number of tokens with a strictly larger
logit). Only these arrays are stored, never vocabulary vectors.

Subcommands:
  reference  launch llama-server on 127.0.0.1:11560, store /tokenize ids for
             every document and full-vocabulary n_probs at sampled positions of
             two documents (--mode incremental: cache_prompt, one new token per
             request; --mode prefix: each prefix from scratch), then stop it.
  score      score documents (all splits, or --limit / --docs) into JSONL.
  verify     both routes against both references, plus bitwise repeat runs.

Outputs go under runs/container/gemma-doc/ (ignored), read in the dev
container by scripts/evaluate_document_likelihood.py.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LLAMA_DIR = Path(os.environ.get('LOCALAPPDATA', '')) / 'Microsoft/WinGet/Packages/ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe'
MODEL_DIGEST = 'sha256:f06e91e18fc74295d0c781bf91622508c75655e2c18ed869d52447ffe1e667e1'
MODEL_PATH = ROOT / 'models/blobs' / MODEL_DIGEST.replace(':', '-')
OUT = ROOT / 'runs/container/gemma-doc'
SPLITS = (('calibration', 'is_memorized'), ('calibration', 'oos_control'),
          ('held_out', 'is_memorized'), ('held_out', 'oos_control'))
N_CTX, N_BATCH, N_UBATCH, N_THREADS = 2048, 2048, 512, 12
SERVER_PORT = 11560
ROUTES = {'incremental': 'teacher forcing, one token per llama_decode on a growing KV cache '
                          '(= llama-server cache_prompt with n_predict 1 per prefix)',
          'batched': 'teacher forcing, one llama_decode per document with an output row per position'}
SERVER_FLAGS = ['-ngl', '99', '-dev', 'Vulkan0', '-fa', 'off', '-c', str(N_CTX), '-np', '1', '--swa-full', '--no-webui']


def load_documents():
    documents = []
    for split, group in SPLITS:
        path = ROOT / 'data/edgar' / split / f'{group}.jsonl'
        for index, line in enumerate(path.read_text(encoding='utf-8').splitlines()):
            item = json.loads(line)
            text = item['prompt']
            documents.append({'doc_id': f'{split}/{group}/{index}', 'split': split, 'group': group,
                              'index': index, 'label': int(item['label']),
                              'accession': item.get('metadata', {}).get('accession'),
                              'text': text, 'text_sha256': hashlib.sha256(text.encode('utf-8')).hexdigest()})
    return documents


def token_statistics(logits_rows, targets):
    """Per-row logprob of the target, entropy (nats) and rank (1 = argmax), float64.

    ``logits_rows[i]`` is the full next-token logit vector whose target is
    ``targets[i]``. Rank counts tokens with a strictly larger logit, plus one.
    """
    x = np.asarray(logits_rows, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    rows = np.arange(x.shape[0])
    peak = x.max(axis=1, keepdims=True)
    shifted = x - peak
    lse = np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    logp = shifted - lse
    p = np.exp(logp)
    entropy = -(p * logp).sum(axis=1)
    target_logit = x[rows, targets]
    rank = 1 + (x > target_logit[:, None]).sum(axis=1)
    return logp[rows, targets], entropy, rank.astype(np.int64)


# ---------- llama.dll through ctypes (struct layouts of llama.h at 372dc5ca1) ----------

class ModelParams(ctypes.Structure):
    _fields_ = [('devices', ctypes.c_void_p), ('tensor_buft_overrides', ctypes.c_void_p),
                ('n_gpu_layers', ctypes.c_int32), ('split_mode', ctypes.c_int), ('load_mode', ctypes.c_int),
                ('lazy_mode', ctypes.c_int), ('main_gpu', ctypes.c_int32), ('tensor_split', ctypes.c_void_p),
                ('progress_callback', ctypes.c_void_p), ('progress_callback_user_data', ctypes.c_void_p),
                ('kv_overrides', ctypes.c_void_p), ('vocab_only', ctypes.c_bool), ('check_tensors', ctypes.c_bool),
                ('use_extra_bufts', ctypes.c_bool), ('no_host', ctypes.c_bool), ('no_alloc', ctypes.c_bool),
                ('load_mtp', ctypes.c_bool)]


class ContextParams(ctypes.Structure):
    _fields_ = [('n_ctx', ctypes.c_uint32), ('n_batch', ctypes.c_uint32), ('n_ubatch', ctypes.c_uint32),
                ('n_seq_max', ctypes.c_uint32), ('n_rs_seq', ctypes.c_uint32), ('n_outputs_max', ctypes.c_uint32),
                ('n_outputs_max_per_seq', ctypes.c_uint32), ('n_threads', ctypes.c_int32),
                ('n_threads_batch', ctypes.c_int32), ('ctx_type', ctypes.c_int), ('rope_scaling_type', ctypes.c_int),
                ('pooling_type', ctypes.c_int), ('attention_type', ctypes.c_int), ('flash_attn_type', ctypes.c_int),
                ('rope_freq_base', ctypes.c_float), ('rope_freq_scale', ctypes.c_float),
                ('yarn_ext_factor', ctypes.c_float), ('yarn_attn_factor', ctypes.c_float),
                ('yarn_beta_fast', ctypes.c_float), ('yarn_beta_slow', ctypes.c_float),
                ('yarn_orig_ctx', ctypes.c_uint32), ('defrag_thold', ctypes.c_float),
                ('cb_eval', ctypes.c_void_p), ('cb_eval_user_data', ctypes.c_void_p),
                ('type_k', ctypes.c_int), ('type_v', ctypes.c_int), ('moe_cache_size', ctypes.c_size_t),
                ('moe_cache_cpu_share', ctypes.c_float), ('abort_callback', ctypes.c_void_p),
                ('abort_callback_data', ctypes.c_void_p), ('embeddings', ctypes.c_bool),
                ('offload_kqv', ctypes.c_bool), ('no_perf', ctypes.c_bool), ('op_offload', ctypes.c_bool),
                ('swa_full', ctypes.c_bool), ('kv_unified', ctypes.c_bool), ('samplers', ctypes.c_void_p),
                ('n_samplers', ctypes.c_size_t), ('ctx_other', ctypes.c_void_p)]


class Batch(ctypes.Structure):
    _fields_ = [('n_tokens', ctypes.c_int32), ('token', ctypes.POINTER(ctypes.c_int32)),
                ('embd', ctypes.POINTER(ctypes.c_float)), ('pos', ctypes.POINTER(ctypes.c_int32)),
                ('n_seq_id', ctypes.POINTER(ctypes.c_int32)),
                ('seq_id', ctypes.POINTER(ctypes.POINTER(ctypes.c_int32))), ('logits', ctypes.POINTER(ctypes.c_int8))]


class Llama:
    """One loaded model and context on Vulkan0, all layers offloaded, flash attention off."""

    def __init__(self, llama_dir=LLAMA_DIR, model_path=MODEL_PATH):
        os.add_dll_directory(str(llama_dir))
        self.ggml = ctypes.CDLL(str(Path(llama_dir) / 'ggml.dll'))
        lib = self.lib = ctypes.CDLL(str(Path(llama_dir) / 'llama.dll'))
        self.ggml.ggml_backend_load_all_from_path.argtypes = [ctypes.c_char_p]
        self.ggml.ggml_backend_load_all_from_path(str(llama_dir).encode())
        self.ggml.ggml_backend_dev_by_name.restype = ctypes.c_void_p
        self.ggml.ggml_backend_dev_by_name.argtypes = [ctypes.c_char_p]
        lib.llama_backend_init()
        lib.llama_model_default_params.restype = ModelParams
        lib.llama_context_default_params.restype = ContextParams
        lib.llama_model_load_from_file.restype = ctypes.c_void_p
        lib.llama_model_load_from_file.argtypes = [ctypes.c_char_p, ModelParams]
        lib.llama_init_from_model.restype = ctypes.c_void_p
        lib.llama_init_from_model.argtypes = [ctypes.c_void_p, ContextParams]
        lib.llama_model_get_vocab.restype = ctypes.c_void_p
        lib.llama_model_get_vocab.argtypes = [ctypes.c_void_p]
        lib.llama_vocab_n_tokens.argtypes = [ctypes.c_void_p]
        lib.llama_vocab_get_add_bos.restype = ctypes.c_bool
        lib.llama_vocab_get_add_bos.argtypes = [ctypes.c_void_p]
        lib.llama_vocab_get_add_eos.restype = ctypes.c_bool
        lib.llama_vocab_get_add_eos.argtypes = [ctypes.c_void_p]
        lib.llama_tokenize.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32,
                                       ctypes.POINTER(ctypes.c_int32), ctypes.c_int32, ctypes.c_bool, ctypes.c_bool]
        lib.llama_batch_init.restype = Batch
        lib.llama_batch_init.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
        lib.llama_batch_free.argtypes = [Batch]
        lib.llama_decode.argtypes = [ctypes.c_void_p, Batch]
        lib.llama_get_logits_ith.restype = ctypes.POINTER(ctypes.c_float)
        lib.llama_get_logits_ith.argtypes = [ctypes.c_void_p, ctypes.c_int32]
        lib.llama_get_memory.restype = ctypes.c_void_p
        lib.llama_get_memory.argtypes = [ctypes.c_void_p]
        lib.llama_memory_clear.argtypes = [ctypes.c_void_p, ctypes.c_bool]
        lib.llama_free.argtypes = [ctypes.c_void_p]
        lib.llama_model_free.argtypes = [ctypes.c_void_p]

        device = self.ggml.ggml_backend_dev_by_name(b'Vulkan0')
        if not device:
            raise RuntimeError('Vulkan0 device not found')
        self._devices = (ctypes.c_void_p * 2)(device, None)
        mp = lib.llama_model_default_params()
        mp.devices = ctypes.cast(self._devices, ctypes.c_void_p)
        mp.n_gpu_layers = 99
        self.model = lib.llama_model_load_from_file(str(model_path).encode(), mp)
        if not self.model:
            raise RuntimeError('model load failed')
        cp = lib.llama_context_default_params()
        cp.n_ctx, cp.n_batch, cp.n_ubatch, cp.n_seq_max = N_CTX, N_BATCH, N_UBATCH, 1
        cp.n_threads = cp.n_threads_batch = N_THREADS
        cp.flash_attn_type = 0  # LLAMA_FLASH_ATTN_TYPE_DISABLED, as -fa off
        cp.embeddings = False
        cp.swa_full = True  # As --swa-full (also the llama.h default); the server needs it to reuse an SWA cache.
        self.context_params = {name: getattr(cp, name) for name, _ in ContextParams._fields_
                               if not name.endswith(('callback', 'callback_data', 'user_data', 'samplers', 'ctx_other'))
                               and name != 'cb_eval'}
        self.ctx = lib.llama_init_from_model(self.model, cp)
        if not self.ctx:
            raise RuntimeError('context init failed')
        self.vocab = lib.llama_model_get_vocab(self.model)
        self.n_vocab = int(lib.llama_vocab_n_tokens(self.vocab))
        self.add_bos = bool(lib.llama_vocab_get_add_bos(self.vocab))
        self.add_eos = bool(lib.llama_vocab_get_add_eos(self.vocab))

    def tokenize(self, text):
        raw = text.encode('utf-8')
        capacity = len(raw) + 8
        buffer = (ctypes.c_int32 * capacity)()
        n = self.lib.llama_tokenize(self.vocab, raw, len(raw), buffer, capacity, True, False)
        if n < 0:
            raise RuntimeError('tokenize failed')
        return list(buffer[:n])

    def logits(self, tokens, rows=None):
        """Full logits for every position (or only ``rows``), shape (len, n_vocab), float32 copy."""
        n = len(tokens)
        if n > N_CTX or n > N_BATCH:
            raise ValueError('document longer than the context')
        self.lib.llama_memory_clear(self.lib.llama_get_memory(self.ctx), True)
        batch = self.lib.llama_batch_init(n, 0, 1)
        try:
            for i, token in enumerate(tokens):
                batch.token[i] = token
                batch.pos[i] = i
                batch.n_seq_id[i] = 1
                batch.seq_id[i][0] = 0
                batch.logits[i] = 1
            batch.n_tokens = n
            status = self.lib.llama_decode(self.ctx, batch)
            if status != 0:
                raise RuntimeError(f'llama_decode returned {status}')
            wanted = range(n) if rows is None else rows
            out = np.empty((len(wanted), self.n_vocab), dtype=np.float32)
            for k, i in enumerate(wanted):
                pointer = self.lib.llama_get_logits_ith(self.ctx, i)
                out[k] = np.ctypeslib.as_array(pointer, shape=(self.n_vocab,))
            return out
        finally:
            self.lib.llama_batch_free(batch)

    def incremental_logits(self, tokens, rows=None):
        """Yield (i, logits at position i) decoding ONE token per llama_decode on a growing KV cache.

        This is the computation llama-server performs with cache_prompt for a
        prefix that extends the cached one by a single token.
        """
        if len(tokens) > N_CTX:
            raise ValueError('document longer than the context')
        wanted = None if rows is None else set(rows)
        self.lib.llama_memory_clear(self.lib.llama_get_memory(self.ctx), True)
        batch = self.lib.llama_batch_init(1, 0, 1)
        try:
            batch.n_tokens, batch.n_seq_id[0], batch.seq_id[0][0], batch.logits[0] = 1, 1, 0, 1
            for i, token in enumerate(tokens):
                batch.token[0], batch.pos[0] = token, i
                status = self.lib.llama_decode(self.ctx, batch)
                if status != 0:
                    raise RuntimeError(f'llama_decode returned {status}')
                if wanted is None or i in wanted:
                    pointer = self.lib.llama_get_logits_ith(self.ctx, -1)
                    yield i, np.ctypeslib.as_array(pointer, shape=(self.n_vocab,)).copy()
        finally:
            self.lib.llama_batch_free(batch)

    def score(self, tokens, route='incremental', chunk=64):
        """Per-token (logprob, entropy, rank) for targets tokens[1:]."""
        if route == 'incremental':
            parts = [token_statistics(row[None, :], [tokens[i + 1]]) for i, row in self.incremental_logits(tokens[:-1])]
            return tuple(np.concatenate(values) for values in zip(*parts, strict=True))
        logits = self.logits(tokens[:-1])
        parts = [token_statistics(logits[s:s + chunk], tokens[s + 1:s + 1 + chunk]) for s in range(0, len(logits), chunk)]
        return tuple(np.concatenate(values) for values in zip(*parts, strict=True))

    def close(self):
        self.lib.llama_free(self.ctx)
        self.lib.llama_model_free(self.model)
        self.lib.llama_backend_free()


def build_info():
    exe = LLAMA_DIR / 'llama-server.exe'
    version = subprocess.run([str(exe), '--version'], capture_output=True, text=True, timeout=60)
    lines = [line.strip() for line in (version.stdout + version.stderr).splitlines() if 'version' in line or 'built' in line]
    return {'llama_cpp': lines, 'dll': 'llama.dll via ctypes', 'dll_sha256': hashlib.sha256((LLAMA_DIR / 'llama.dll').read_bytes()).hexdigest()}


# ---------- subcommands ----------

def cmd_score(args):
    documents = load_documents()
    if args.docs:
        wanted = set(args.docs.split(','))
        documents = [d for d in documents if d['doc_id'] in wanted]
    if args.limit:
        documents = documents[:args.limit]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    llama = Llama()
    load_seconds = time.perf_counter() - started
    warm = llama.tokenize(documents[0]['text'])
    t0 = time.perf_counter()
    llama.score(warm[:32], args.route)  # Warm-up decode after load; its result is discarded.
    warmup_seconds = time.perf_counter() - t0
    records_path = out / 'scores.jsonl'
    timings = []
    with records_path.open('w', encoding='utf-8', newline='\n') as handle:
        for document in documents:
            tokens = llama.tokenize(document['text'])
            t0 = time.perf_counter()
            logprob, entropy, rank = llama.score(tokens, args.route)
            seconds = time.perf_counter() - t0
            timings.append({'doc_id': document['doc_id'], 'n_tokens': len(tokens), 'seconds': seconds})
            record = {key: document[key] for key in ('doc_id', 'split', 'group', 'index', 'label', 'accession', 'text_sha256')}
            record |= {'n_vocab': llama.n_vocab, 'n_tokens': len(tokens), 'token_ids': tokens,
                       'logprob': logprob.tolist(), 'entropy': entropy.tolist(), 'rank': rank.tolist()}
            handle.write(json.dumps(record) + '\n')
            print(f"{document['doc_id']} n={len(tokens)} {seconds:.2f}s", file=sys.stderr, flush=True)
    n_vocab, add_bos, add_eos, context_params = llama.n_vocab, llama.add_bos, llama.add_eos, llama.context_params
    llama.close()
    total_tokens = sum(t['n_tokens'] for t in timings)
    total_seconds = sum(t['seconds'] for t in timings)
    manifest = {
        'schema': 'gemma-doc-scores/1', 'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'model': {'name': 'gemma3:4b (Ollama Phase A blob)', 'digest': MODEL_DIGEST, 'path': str(MODEL_PATH.relative_to(ROOT))},
        'build': build_info(),
        'flags': {'route': args.route, 'route_detail': ROUTES[args.route],
                  'equivalent_server_flags': SERVER_FLAGS, 'n_gpu_layers': 99, 'device': 'Vulkan0',
                  'flash_attn': 'disabled', 'context_params': context_params, 'warmup_decodes': 1,
                  'warmup_seconds': warmup_seconds, 'load_seconds': load_seconds},
        'template': 'none', 'tokenization': {'add_special': True, 'parse_special': False, 'add_bos': add_bos, 'add_eos': add_eos},
        'statistics': {'logprob': 'log p(token t | tokens < t) from the full softmax at t-1, float64, nats',
                       'entropy': 'entropy of the full distribution at t-1, nats',
                       'rank': '1 + number of vocabulary entries with a strictly larger logit'},
        'n_vocab': n_vocab, 'n_documents': len(timings), 'n_tokens': total_tokens,
        'n_scored_positions': total_tokens - len(timings), 'scoring_seconds': total_seconds,
        'tokens_per_second': total_tokens / total_seconds if total_seconds else None,
        'documents': [{'doc_id': d['doc_id'], 'text_sha256': d['text_sha256'], **{k: t[k] for k in ('n_tokens', 'seconds')}}
                      for d, t in zip(documents, timings, strict=True)],
        'scores_sha256': hashlib.sha256(records_path.read_bytes()).hexdigest(),
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: manifest[k] for k in ('n_documents', 'n_tokens', 'scoring_seconds', 'tokens_per_second')}))


def _post(base, path, body, timeout=900):
    request = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(request, timeout=timeout).read())


def reference_positions(n_tokens):
    """Sampled target positions t (1 <= t < n): edges, the 512 ubatch boundary, and evenly spaced."""
    picks = {1, 2, 3, n_tokens // 2, n_tokens - 1} | set(np.linspace(1, n_tokens - 1, 8).astype(int).tolist())
    picks |= {t for t in (511, 512, 513, 514) if t < n_tokens}
    return sorted(picks)


def cmd_reference(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    documents = load_documents()
    log = (out / f'llama-server-{args.mode}.log').open('w')
    server = subprocess.Popen([str(LLAMA_DIR / 'llama-server.exe'), '-m', str(MODEL_PATH), '--host', '127.0.0.1',
                               '--port', str(SERVER_PORT), *SERVER_FLAGS], stdout=log, stderr=subprocess.STDOUT)
    base = f'http://127.0.0.1:{SERVER_PORT}'
    try:
        for _ in range(240):
            try:
                if json.loads(urllib.request.urlopen(base + '/health', timeout=5).read()).get('status') == 'ok':
                    break
            except Exception:
                time.sleep(1)
        tokens = {d['doc_id']: _post(base, '/tokenize', {'content': d['text'], 'add_special': True, 'parse_special': False})['tokens']
                  for d in documents}
        chosen = [args.doc_a, max(tokens, key=lambda k: len(tokens[k])) if args.doc_b == 'longest' else args.doc_b]
        body = {'n_predict': 1, 'temperature': 0, 'n_probs': args.n_probs, 'post_sampling_probs': False,
                'cache_prompt': False, 'seed': 0}
        _post(base, '/completion', body | {'prompt': tokens[chosen[0]][:8]})  # Warm-up request after load.
        rows, meta = [], []
        incremental = args.mode == 'incremental'
        for doc_id in chosen:
            ids = tokens[doc_id]
            sampled = reference_positions(len(ids))
            # prefix: each sampled prefix evaluated from scratch in one batch (cache_prompt false).
            # incremental: every prefix 1..max in order with cache_prompt true, so the server
            # evaluates exactly one new token per request; full n_probs only at sampled positions.
            steps = range(1, max(sampled) + 1) if incremental else sampled
            for t in steps:
                full = t in sampled
                request = body | {'prompt': ids[:t], 'cache_prompt': incremental, 'n_probs': args.n_probs if full else 0}
                t0 = time.perf_counter()
                response = _post(base, '/completion', request)
                seconds = time.perf_counter() - t0
                evaluated = response['timings']['prompt_n']
                if evaluated != (1 if incremental else t):
                    raise RuntimeError(f'server evaluated {evaluated} tokens at t={t}')
                if not full:
                    continue
                entries = response['completion_probabilities'][0]['top_logprobs']
                vector = np.full(args.n_probs, np.nan, dtype=np.float64)
                for entry in entries:
                    vector[entry['id']] = entry['logprob']
                rows.append(vector.astype(np.float32))
                meta.append({'doc_id': doc_id, 't': t, 'target': ids[t], 'n_entries': len(entries), 'seconds': seconds})
                print(f'{doc_id} t={t} entries={len(entries)} {seconds:.1f}s', file=sys.stderr, flush=True)
        np.save(out / f'reference_{args.mode}_logprobs.npy', np.stack(rows))
        (out / f'reference_{args.mode}.json').write_text(json.dumps({'mode': args.mode, 'server_flags': SERVER_FLAGS, 'port': SERVER_PORT, 'documents': chosen,
                                                         'request': body, 'positions': meta, 'server_tokens': tokens,
                                                         'build': build_info()}, indent=1) + '\n', encoding='utf-8')
    finally:
        server.terminate()
        server.wait(timeout=60)
        log.close()


def _compare(reference, ref_rows, results, renormalise):
    """Max differences of one route (``results[doc_id]`` = stats, repeat stats, {t-1: logits}) to one reference.

    llama-server normalises its n_probs in float32 (its probabilities sum to 1
    within about 4e-4, a constant log offset per position). With
    ``renormalise`` the server's full-vocabulary vector is re-normalised in
    float64 first, so the comparison is between the logits themselves.
    """
    worst = {'logprob': 0.0, 'entropy': 0.0, 'full_vector_logprob': 0.0}
    rank_mismatch, positions = 0, []
    for row, position in zip(ref_rows, reference['positions'], strict=True):
        doc_id, t = position['doc_id'], position['t']
        (logprob, entropy, rank), _, rows = results[doc_id]
        ok = np.isfinite(row) & (row > -1e30)  # Server float32 floor at about -3.4e38.
        row = row.copy()
        sum_p_minus_1 = float(np.exp(row[ok]).sum() - 1)
        if renormalise:
            peak = row[ok].max()
            row[ok] -= peak + np.log(np.exp(row[ok] - peak).sum())
        ref_lp = row[position['target']]
        p = np.exp(row[ok])
        ref_entropy = float(-(p * row[ok]).sum())
        ref_rank = int(1 + (row > ref_lp).sum())
        x = rows[t - 1].astype(np.float64)
        mine = x - x.max() - np.log(np.exp(x - x.max()).sum())
        vector_diff = float(np.abs(mine[ok] - row[ok]).max())
        d_lp, d_h = float(abs(logprob[t - 1] - ref_lp)), float(abs(entropy[t - 1] - ref_entropy))
        worst = {'logprob': max(worst['logprob'], d_lp), 'entropy': max(worst['entropy'], d_h),
                 'full_vector_logprob': max(worst['full_vector_logprob'], vector_diff)}
        rank_mismatch += int(rank[t - 1] != ref_rank)
        positions.append({'doc_id': doc_id, 't': t, 'floored': int((row <= -1e30).sum()), 'server_sum_p_minus_1': sum_p_minus_1,
                          'abs_diff_logprob': d_lp, 'abs_diff_entropy': d_h, 'rank': int(rank[t - 1]), 'ref_rank': ref_rank,
                          'max_abs_diff_full_vector': vector_diff})
    return {'positions_checked': len(positions), 'renormalised_float64': renormalise, 'max_abs_diff': worst,
            'rank_mismatches': rank_mismatch, 'within_1e-4': max(worst.values()) <= 1e-4 and rank_mismatch == 0,
            'positions': positions}


def cmd_verify(args):
    """Both routes against both server references on the same two documents, plus repeat runs."""
    out = Path(args.output)
    references = {}
    for mode in ('incremental', 'prefix'):
        if (out / f'reference_{mode}.json').exists():
            references[mode] = (json.loads((out / f'reference_{mode}.json').read_text(encoding='utf-8')),
                                np.load(out / f'reference_{mode}_logprobs.npy').astype(np.float64))
    reference = references['incremental'][0]
    documents = {d['doc_id']: d for d in load_documents()}
    llama = Llama()
    report = {'n_vocab': llama.n_vocab, 'add_bos': llama.add_bos, 'add_eos': llama.add_eos}
    mismatched = [k for k, ids in reference['server_tokens'].items() if llama.tokenize(documents[k]['text']) != ids]
    report['tokenization'] = {'documents': len(reference['server_tokens']), 'mismatched_vs_server_tokenize': mismatched}
    llama.score(llama.tokenize(documents[reference['documents'][0]]['text'])[:32])  # Warm-up.
    report['routes'] = {}
    for route in ('incremental', 'batched'):
        results, seconds = {}, 0.0
        for doc_id in reference['documents']:
            tokens = llama.tokenize(documents[doc_id]['text'])
            t0 = time.perf_counter()
            first = llama.score(tokens, route)
            seconds += time.perf_counter() - t0
            wanted = sorted({p['t'] - 1 for _, (r, _) in references.items() for p in r['positions'] if p['doc_id'] == doc_id})
            if route == 'incremental':
                rows = dict(llama.incremental_logits(tokens, wanted))
            else:
                rows = dict(zip(wanted, llama.logits(tokens, wanted), strict=True))
            results[doc_id] = (first, llama.score(tokens, route), rows)
        n_tokens = sum(len(llama.tokenize(documents[d]['text'])) for d in reference['documents'])
        entry = {'repeat_run_bitwise_identical': {d: all(np.array_equal(a, b) for a, b in zip(r[0], r[1], strict=True))
                                                  for d, r in results.items()},
                 'tokens_per_second': n_tokens / seconds}
        for mode, (ref, rows) in references.items():
            entry[f'vs_{mode}_reference'] = _compare(ref, rows, results, renormalise=True)
            entry[f'vs_{mode}_reference_raw_float32_normaliser'] = _compare(ref, rows, results, renormalise=False)
        report['routes'][route] = entry
    llama.close()
    (out / 'verification.json').write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    summary = {route: {k: ({m: v[m] for m in ('max_abs_diff', 'rank_mismatches', 'within_1e-4')} if k.startswith('vs_') else v)
                       for k, v in entry.items()} for route, entry in report['routes'].items()}
    print(json.dumps({'tokenization': report['tokenization'], 'routes': summary}, indent=1))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    score = sub.add_parser('score')
    score.add_argument('--output', default=str(OUT / 'scores'))
    score.add_argument('--limit', type=int)
    score.add_argument('--docs')
    score.add_argument('--route', choices=tuple(ROUTES), default='incremental')
    reference = sub.add_parser('reference')
    reference.add_argument('--output', default=str(OUT / 'verification'))
    reference.add_argument('--doc-a', default='calibration/is_memorized/0')
    reference.add_argument('--doc-b', default='longest')
    reference.add_argument('--n-probs', type=int, default=262145)
    reference.add_argument('--mode', choices=('prefix', 'incremental'), default='incremental')
    verify = sub.add_parser('verify')
    verify.add_argument('--output', default=str(OUT / 'verification'))
    args = parser.parse_args(argv)
    {'score': cmd_score, 'reference': cmd_reference, 'verify': cmd_verify}[args.command](args)


if __name__ == '__main__':
    main()
