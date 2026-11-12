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

The ctypes bindings (``Llama``), ``token_statistics`` and the pinned paths live
in ``scripts/hosts/llamacpp_native.py``, shared with the task 6.1 teacher-forcing host
(``scripts/hosts/serve_full_distribution.py``).

Outputs go under runs/container/gemma-doc/ (ignored), read in the dev
container by scripts/evaluate_document_likelihood.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # Run as a file; the shared bindings live in scripts/.
    sys.path.insert(0, str(ROOT))
from scripts.hosts.llamacpp_native import (  # noqa: E402,F401  (token_statistics is re-exported for existing callers)
    LLAMA_DIR,
    MODEL_DIGEST,
    MODEL_PATH,
    SERVER_FLAGS,
    SERVER_PORT,
    Llama,
    build_info,
    post_json,
    token_statistics,
)

OUT = ROOT / 'runs/container/gemma-doc'
SPLITS = (('calibration', 'is_memorized'), ('calibration', 'oos_control'),
          ('held_out', 'is_memorized'), ('held_out', 'oos_control'))
ROUTES = {'incremental': 'teacher forcing, one token per llama_decode on a growing KV cache '
                          '(= llama-server cache_prompt with n_predict 1 per prefix)',
          'batched': 'teacher forcing, one llama_decode per document with an output row per position'}


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
        tokens = {d['doc_id']: post_json(base, '/tokenize', {'content': d['text'], 'add_special': True, 'parse_special': False})['tokens']
                  for d in documents}
        chosen = [args.doc_a, max(tokens, key=lambda k: len(tokens[k])) if args.doc_b == 'longest' else args.doc_b]
        body = {'n_predict': 1, 'temperature': 0, 'n_probs': args.n_probs, 'post_sampling_probs': False,
                'cache_prompt': False, 'seed': 0}
        post_json(base, '/completion', body | {'prompt': tokens[chosen[0]][:8]})  # Warm-up request after load.
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
                response = post_json(base, '/completion', request)
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
