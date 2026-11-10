"""llama.cpp on this host: the pinned GGUF, ctypes bindings to ``llama.dll`` and float64 token statistics.

Shared by ``scripts/score_documents_llamacpp.py`` (R37/R40 document scores) and
``scripts/serve_full_distribution.py`` (task 6.1 teacher-forcing host). Host-side
tooling, run NATIVELY on Windows with the checkout's Python (stdlib + numpy
only; no Recall Guard import). The model runs on the iGPU through the
user-installed llama.cpp build (winget, b12199, Vulkan) with the settings of the
reference server (``-ngl 99 -dev Vulkan0 -fa off --swa-full``, n_ctx 2048,
n_batch 2048, n_ubatch 512, one sequence).

``Llama.incremental_logits`` decodes ONE token per ``llama_decode`` on a growing
KV cache, which R40 showed matches llama-server's ``cache_prompt`` computation
within 1e-6 once the server's float32 normaliser is redone in float64. The
batched ``Llama.logits`` route changes the Vulkan kernels with the batch shape
and differs by up to about 0.1 nats, so it is kept only for comparison.

For the batch route of task 6.4, ``Llama.continue_logits`` decodes further tokens
(again one per ``llama_decode``) on top of a cached prompt and ``Llama.truncate``
rolls the KV cache back to the prompt length with ``llama_memory_seq_rm``. The
installed b12199 ``llama.dll`` exports the ``llama_memory_*`` API
(``llama_get_memory``, ``llama_memory_seq_rm``, ``llama_memory_seq_pos_max``);
the older ``llama_kv_self_seq_rm`` is not exported, so it is not bound.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import subprocess
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LLAMA_DIR = Path(os.environ.get('LOCALAPPDATA', '')) / 'Microsoft/WinGet/Packages/ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe'
MODEL_DIGEST = 'sha256:f06e91e18fc74295d0c781bf91622508c75655e2c18ed869d52447ffe1e667e1'
MODEL_PATH = ROOT / 'models/blobs' / MODEL_DIGEST.replace(':', '-')
N_CTX, N_BATCH, N_UBATCH, N_THREADS = 2048, 2048, 512, 12
SERVER_PORT = 11560  # The reference llama-server of R40 and of the 6.1 exactness check.
SERVER_FLAGS = ['-ngl', '99', '-dev', 'Vulkan0', '-fa', 'off', '-c', str(N_CTX), '-np', '1', '--swa-full', '--no-webui']


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
        lib.llama_memory_seq_rm.restype = ctypes.c_bool   # (memory, seq_id, p0, p1); p1 < 0 means to the end.
        lib.llama_memory_seq_rm.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
        lib.llama_memory_seq_pos_max.restype = ctypes.c_int32   # -1 when the sequence is empty.
        lib.llama_memory_seq_pos_max.argtypes = [ctypes.c_void_p, ctypes.c_int32]
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

    def tokenize(self, text, add_special=True, parse_special=False):
        """Token ids of ``text``; ``add_special`` adds BOS as the model requires, ``parse_special`` reads
        special-token text as tokens. The defaults are the R37/R40 document setting."""
        raw = text.encode('utf-8')
        capacity = len(raw) + 8
        buffer = (ctypes.c_int32 * capacity)()
        n = self.lib.llama_tokenize(self.vocab, raw, len(raw), buffer, capacity, bool(add_special), bool(parse_special))
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

    def cached_length(self):
        """Positions the KV cache holds for sequence 0 (``llama_memory_seq_pos_max`` plus one; 0 when empty)."""
        return int(self.lib.llama_memory_seq_pos_max(self.lib.llama_get_memory(self.ctx), 0)) + 1

    def continue_logits(self, tokens, start):
        """Yield (i, logits at position i) for i = start, start + 1, ..., ONE token per llama_decode, on top of
        the cached positions 0..start-1, which are kept (no clear).

        After ``incremental_logits`` decoded a prompt of ``start`` tokens, this is the computation one
        ``incremental_logits`` call over the prompt plus ``tokens`` performs for these positions (task 6.4).
        """
        if start < 1 or start + len(tokens) > N_CTX:
            raise ValueError('the continuation runs beyond the context')
        held = self.cached_length()
        if held != start:
            raise RuntimeError(f'the KV cache holds {held} positions, not {start}')
        batch = self.lib.llama_batch_init(1, 0, 1)
        try:
            batch.n_tokens, batch.n_seq_id[0], batch.seq_id[0][0], batch.logits[0] = 1, 1, 0, 1
            for i, token in enumerate(tokens, start):
                batch.token[0], batch.pos[0] = token, i
                status = self.lib.llama_decode(self.ctx, batch)
                if status != 0:
                    raise RuntimeError(f'llama_decode returned {status}')
                pointer = self.lib.llama_get_logits_ith(self.ctx, -1)
                yield i, np.ctypeslib.as_array(pointer, shape=(self.n_vocab,)).copy()
        finally:
            self.lib.llama_batch_free(batch)

    def truncate(self, n_keep):
        """Roll the KV cache of sequence 0 back to positions 0..n_keep-1 (``llama_memory_seq_rm`` from n_keep on)
        and check that exactly those positions remain."""
        if not self.lib.llama_memory_seq_rm(self.lib.llama_get_memory(self.ctx), 0, n_keep, -1):
            raise RuntimeError('llama_memory_seq_rm refused the rollback')
        held = self.cached_length()
        if held != n_keep:
            raise RuntimeError(f'after the rollback the KV cache holds {held} positions, not {n_keep}')

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


def post_json(base, path, body, timeout=900):
    """POST ``body`` as JSON to a loopback llama-server and return the decoded JSON reply."""
    request = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
    return json.loads(urllib.request.urlopen(request, timeout=timeout).read())
