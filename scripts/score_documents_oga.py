"""Teacher-forced per-token scores of EDGAR documents under Gemma 4 E4B on the AMD NPU (R37, R39).

Host-side tooling, run NATIVELY on Windows with the Ryzen AI OGA venv Python
(``tools/oga-venv``, onnxruntime-genai-directml-ryzenai 0.18.0; stdlib + numpy
+ onnxruntime_genai, no Recall Guard import). WSL and Docker have no NPU.

Same method as ``scripts/score_documents_llamacpp.py`` (Gemma 3 4B, R40): raw
document text, no chat template, BOS (id 2) then the document tokens from the
model's own tokenizer (``tokenizer.json`` of the OGA folder). For every
position t >= 1 the FULL next-token distribution at t-1 (logits of shape
(1, 1, 262144)) gives, in float64: the logprob of the actual token t, the
entropy (nats) and the rank (1 + number of tokens with a strictly larger
logit). Only these arrays are stored, never vocabulary vectors. The record
format is the llama.cpp scorer's, so ``scripts/evaluate_document_likelihood.py``
reads it unchanged.

Route (R39), the only one: OGA returns logits for the last position only, so
the document is fed one token per ``append_tokens`` call (decode route) and
the logits are read after every step. Prefill gives slightly different logits
than decode, and several generators on one loaded model are not reproducible,
so EVERY document is scored on a freshly loaded ``og.Model`` with one
generator. ``run.json`` pins the configuration; a resumed run with any other
route, model or selection is refused.

Selection: Gemma 4's training cutoff is 2025-01-31 (``data/cutoffs.yaml``,
``google/gemma-4-e4b-it``). All memorized rows predate it; control rows are
kept only when published strictly after it. Document ids keep their corpus
index (``held_out/oos_control/17``).

Resumable: ``score`` appends one record per finished document to
``scores.jsonl`` (and its timing to ``timings.jsonl``), skips documents
already written, drops a truncated last line, and when every selected
document is present rewrites ``scores.jsonl`` in corpus order and writes
``manifest.json`` (model repo@revision, main file sha256 against the Hugging
Face LFS hashes, OGA version, NPU driver, route, text hashes, timings).

Subcommands:
  score    score the selection (or --docs) into --output; --check adds a direct
           full-vector check of argmax, rank, logprob and entropy per position.
  compare  compare two ``score --check`` outputs (bitwise repeat check).

Outputs go under runs/container/gemma4-doc/ (ignored), read in the dev
container by scripts/evaluate_document_likelihood.py --published-after 2025-01-31.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # Run as a file with ``python -I``; the shared math lives in scripts/.
    sys.path.insert(0, str(ROOT))
from scripts.score_documents_llamacpp import token_statistics  # noqa: E402

MODEL_REPO = 'amd/gemma-4-e4b-it-mobius-int4_rai_1.9.0_npu'
MODEL_REVISION = '5fcdd87322d42559835f8ecca3e31ae311a262b4'
MODEL_DIR = ROOT / 'models/oga/gemma-4-e4b-it-mobius-int4_rai_1.9.0_npu'
MAIN_FILES = ('genai_config.json', 'tokenizer.json', 'tokenizer_config.json', 'rai_config.json',
              'decoder/model.onnx', 'decoder/model.onnx.data', 'decoder/model.bin', 'decoder/model.pb.bin',
              'embedding/model.onnx', 'embedding/model.onnx.data')
OUT = ROOT / 'runs/container/gemma4-doc'
CUTOFF = '2025-01-31'  # data/cutoffs.yaml: google/gemma-4-e4b-it
SPLITS = (('calibration', 'is_memorized'), ('calibration', 'oos_control'),
          ('held_out', 'is_memorized'), ('held_out', 'oos_control'))
BOS = 2
N_VOCAB = 262144
ROUTE = 'decode'
ROUTE_DETAIL = ('decode, fresh model per document: a new og.Model and one og.Generator per document; BOS, then '
                'one append_tokens call per document token, full last-position logits read after every step')
SEARCH_OPTIONS = {'do_sample': False, 'top_k': 1}
MAX_LENGTH_MARGIN = 8
RECORD_KEYS = ('doc_id', 'split', 'group', 'index', 'label', 'accession', 'text_sha256')


# ---------- corpus and selection ----------

def load_documents(data_dir=ROOT / 'data/edgar'):
    documents = []
    for split, group in SPLITS:
        path = Path(data_dir) / split / f'{group}.jsonl'
        for index, line in enumerate(path.read_text(encoding='utf-8').splitlines()):
            item = json.loads(line)
            text, metadata = item['prompt'], item.get('metadata', {})
            documents.append({'doc_id': f'{split}/{group}/{index}', 'split': split, 'group': group, 'index': index,
                              'label': int(item['label']), 'accession': metadata.get('accession'),
                              'published_at': metadata.get('published_at'), 'text': text,
                              'text_sha256': hashlib.sha256(text.encode('utf-8')).hexdigest()})
    return documents


def keep_document(label: int, published_at, published_after) -> bool:
    """Memorized rows always; control rows only if published strictly after ``published_after``."""
    if published_after is None or label == 1:
        return True
    return published_at is not None and published_at > published_after


def select_documents(documents, published_after=CUTOFF):
    """The scored selection; memorized rows must predate the cutoff."""
    late = [d['doc_id'] for d in documents if d['label'] == 1 and published_after and
            (d['published_at'] is None or d['published_at'] > published_after)]
    if late:
        raise SystemExit(f'memorized rows not before the cutoff: {late[:5]}')
    return [d for d in documents if keep_document(d['label'], d['published_at'], published_after)]


def selection_counts(documents):
    counts = {}
    for d in documents:
        counts[f"{d['split']}/{d['group']}"] = counts.get(f"{d['split']}/{d['group']}", 0) + 1
    return counts


# ---------- scoring ----------

def direct_check(row, target):
    """Independent full-vector check of one position: argsort rank, argmax, logaddexp normaliser."""
    x = row.astype(np.float64)
    descending = np.sort(x)[::-1]
    rank = int(np.searchsorted(-descending, -x[target], side='left')) + 1
    logp = x - np.logaddexp.reduce(x)
    return {'rank': rank, 'argmax': int(np.argmax(x)), 'logprob': float(logp[target]),
            'entropy': float(-(np.exp(logp) * logp).sum()), 'sum_p_minus_1': float(np.exp(logp).sum() - 1.0)}


def score_tokens(og, model, tokens, *, n_vocab=None, check=False):
    """(logprob, entropy, rank, logits_sha256, check summary) for targets tokens[1:], decode route."""
    n_vocab = N_VOCAB if n_vocab is None else n_vocab
    params = og.GeneratorParams(model)
    params.set_search_options(**SEARCH_OPTIONS, max_length=len(tokens) + MAX_LENGTH_MARGIN)
    generator = og.Generator(model, params)
    digest = hashlib.sha256()
    logprob, entropy, rank = [], [], []
    summary = {'positions': 0, 'shape_ok': 0, 'finite': 0, 'rank_agree': 0, 'argmax_is_target': 0, 'rank_one': 0,
               'rank_one_iff_argmax_logit': 0, 'max_abs_diff_logprob': 0.0, 'max_abs_diff_entropy': 0.0,
               'max_abs_sum_p_minus_1': 0.0, 'argmax': []} if check else None
    try:
        generator.append_tokens([tokens[0]])
        for i in range(1, len(tokens)):
            logits = np.array(generator.get_logits(), dtype=np.float32)
            if logits.shape != (1, 1, n_vocab):
                raise RuntimeError(f'logits shape {logits.shape} at position {i}')
            row = logits.reshape(-1)
            if not np.isfinite(row).all():
                raise RuntimeError(f'non-finite logits at position {i}')
            digest.update(row.tobytes())
            lp, h, r = token_statistics(row[None, :], [tokens[i]])
            logprob.append(float(lp[0]))
            entropy.append(float(h[0]))
            rank.append(int(r[0]))
            if check:
                d = direct_check(row, tokens[i])
                summary['positions'] += 1
                summary['shape_ok'] += 1
                summary['finite'] += 1
                summary['rank_agree'] += int(d['rank'] == rank[-1])
                summary['argmax_is_target'] += int(d['argmax'] == tokens[i])
                summary['rank_one'] += int(rank[-1] == 1)
                summary['rank_one_iff_argmax_logit'] += int((rank[-1] == 1) == bool(row[tokens[i]] == row[d['argmax']]))
                summary['max_abs_diff_logprob'] = max(summary['max_abs_diff_logprob'], abs(d['logprob'] - logprob[-1]))
                summary['max_abs_diff_entropy'] = max(summary['max_abs_diff_entropy'], abs(d['entropy'] - entropy[-1]))
                summary['max_abs_sum_p_minus_1'] = max(summary['max_abs_sum_p_minus_1'], abs(d['sum_p_minus_1']))
                summary['argmax'].append(d['argmax'])
            if i < len(tokens) - 1:
                generator.append_tokens([tokens[i]])
    finally:
        del generator
    return logprob, entropy, rank, digest.hexdigest(), summary


def tokenize(tokenizer, text):
    """BOS then the raw-text tokens; the tokenizer's own BOS is not doubled."""
    ids = [int(t) for t in tokenizer.encode(text)]
    bos_from_tokenizer = bool(ids) and ids[0] == BOS
    tokens = ids if bos_from_tokenizer else [BOS] + ids
    if BOS in tokens[1:]:
        raise RuntimeError('BOS inside the document tokens')
    return tokens, bos_from_tokenizer


def score_document(og, model_dir, document, *, check=False):
    """Load a fresh og.Model, score one document, release the model."""
    t0 = time.perf_counter()
    model = og.Model(str(model_dir))
    load_seconds = time.perf_counter() - t0
    try:
        tokenizer = og.Tokenizer(model)
        tokens, bos_from_tokenizer = tokenize(tokenizer, document['text'])
        device_type = str(getattr(model, 'device_type', 'unknown'))
        t0 = time.perf_counter()
        logprob, entropy, rank, logits_sha256, summary = score_tokens(og, model, tokens, check=check)
        seconds = time.perf_counter() - t0
        del tokenizer
    finally:
        del model
        gc.collect()
    record = {key: document[key] for key in RECORD_KEYS}
    record |= {'n_vocab': N_VOCAB, 'n_tokens': len(tokens), 'token_ids': tokens,
               'logprob': logprob, 'entropy': entropy, 'rank': rank}
    timing = {'doc_id': document['doc_id'], 'n_tokens': len(tokens), 'load_seconds': load_seconds, 'seconds': seconds,
              'logits_sha256': logits_sha256, 'bos_from_tokenizer': bos_from_tokenizer, 'device_type': device_type,
              'pid': os.getpid(), 'finished_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    if summary is not None:
        timing['check'] = summary
    return record, timing


# ---------- resumable output ----------

def read_jsonl(path: Path, *, repair=False):
    """Complete JSON lines; with ``repair`` a truncated last line is cut off the file."""
    if not path.exists():
        return []
    raw = path.read_bytes()
    *complete, tail = raw.split(b'\n')  # ``tail`` is empty unless the last write was cut short.
    try:
        items = [json.loads(line) for line in complete if line.strip()]
    except json.JSONDecodeError:
        raise SystemExit(f'{path}: corrupt line inside the file') from None
    if tail.strip() and repair:
        with path.open('r+b') as handle:
            handle.truncate(len(raw) - len(tail))
    return items


def append_line(path: Path, item: dict):
    with path.open('a', encoding='utf-8', newline='\n') as handle:
        handle.write(json.dumps(item) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


def run_config(documents, published_after, check, og_version):
    return {'schema': 'gemma4-doc-run/1', 'model_repo': MODEL_REPO, 'model_revision': MODEL_REVISION,
            'route': ROUTE, 'route_detail': ROUTE_DETAIL, 'search_options': SEARCH_OPTIONS,
            'max_length': f'n_tokens + {MAX_LENGTH_MARGIN}', 'template': 'none', 'bos_id': BOS,
            'control_published_after': published_after, 'check': check, 'onnxruntime_genai': og_version,
            'doc_ids': [d['doc_id'] for d in documents]}


def completed(out: Path, documents):
    """Records already written for this selection; any foreign or changed record stops the run."""
    by_id = {d['doc_id']: d for d in documents}
    done = {}
    for record in read_jsonl(out / 'scores.jsonl', repair=True):
        document = by_id.get(record['doc_id'])
        if document is None or document['text_sha256'] != record['text_sha256']:
            raise SystemExit(f"{record['doc_id']}: record does not belong to this selection")
        if record['doc_id'] in done:
            raise SystemExit(f"{record['doc_id']}: written twice")
        done[record['doc_id']] = record
    return done


def score_selection(og, documents, out: Path, *, model_dir=MODEL_DIR, published_after=CUTOFF, check=False,
                    og_version='unknown', log=sys.stderr):
    """Score every not-yet-written document; return True when the selection is complete."""
    out.mkdir(parents=True, exist_ok=True)
    config = run_config(documents, published_after, check, og_version)
    config_path = out / 'run.json'
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding='utf-8'))
        if previous != config:
            changed = sorted(k for k in set(previous) | set(config) if previous.get(k) != config.get(k))
            raise SystemExit(f'{out}: run.json differs ({changed}); refusing to mix runs')
    else:
        config_path.write_text(json.dumps(config, indent=1) + '\n', encoding='utf-8')
    done = completed(out, documents)
    read_jsonl(out / 'timings.jsonl', repair=True)
    pending = [d for d in documents if d['doc_id'] not in done]
    print(f'pid: {os.getpid()} done={len(done)} pending={len(pending)}', file=log, flush=True)
    for k, document in enumerate(pending, 1):
        record, timing = score_document(og, model_dir, document, check=check)
        append_line(out / 'timings.jsonl', timing)
        append_line(out / 'scores.jsonl', record)
        print(f"[{len(done) + k}/{len(documents)}] {document['doc_id']} n={record['n_tokens']} "
              f"load={timing['load_seconds']:.1f}s score={timing['seconds']:.1f}s", file=log, flush=True)
    return len(completed(out, documents)) == len(documents)


# ---------- manifest ----------

def sha256_file(path: Path, chunk=1 << 24):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def main_file_hashes(model_dir: Path):
    """sha256 of the main model files against the Hugging Face LFS hash in the download cache."""
    files = {}
    for name in MAIN_FILES:
        path = model_dir / name
        meta = model_dir / '.cache/huggingface/download' / f'{name}.metadata'
        entry = {'present': path.exists()}
        if meta.exists():
            lines = meta.read_text(encoding='utf-8').split()
            entry |= {'hf_revision': lines[0], 'hf_etag': lines[1] if len(lines) > 1 else None}
        if path.exists():
            entry |= {'size': path.stat().st_size, 'sha256': sha256_file(path)}
            if entry.get('hf_etag') and len(entry['hf_etag']) == 64:
                entry['matches_hf_lfs_sha256'] = entry['sha256'] == entry['hf_etag']
        files[name] = entry
    return files


def npu_driver():
    command = ("Get-CimInstance Win32_PnPSignedDriver | Where-Object { $_.DeviceName -match 'NPU' } | "
               "Select-Object DeviceName,DriverVersion,DriverDate,Manufacturer | ConvertTo-Json -Compress")
    try:
        result = subprocess.run(['powershell', '-NoProfile', '-Command', command], capture_output=True, text=True,
                                timeout=120)
        return json.loads(result.stdout) if result.stdout.strip() else None
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        return {'error': repr(error)[:200]}


def package_version(name):
    from importlib import metadata

    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def finalize(out: Path, documents, *, model_dir=MODEL_DIR, published_after=CUTOFF, og_version='unknown',
             build=None, hash_files=True):
    """Rewrite scores.jsonl in selection order and write manifest.json."""
    done = completed(out, documents)
    if len(done) != len(documents):
        raise SystemExit(f'{len(documents) - len(done)} documents not scored yet')
    timings = {}
    for timing in read_jsonl(out / 'timings.jsonl'):
        timings[timing['doc_id']] = timing  # The last timing per document is the one of its written record.
    records_path = out / 'scores.jsonl'
    temporary = out / 'scores.jsonl.tmp'
    temporary.write_text(''.join(json.dumps(done[d['doc_id']]) + '\n' for d in documents), encoding='utf-8',
                         newline='\n')
    os.replace(temporary, records_path)
    rows = [timings.get(d['doc_id'], {}) for d in documents]
    total_tokens = sum(done[d['doc_id']]['n_tokens'] for d in documents)
    scoring_seconds = sum(t.get('seconds', 0.0) for t in rows)
    load_seconds = sum(t.get('load_seconds', 0.0) for t in rows)
    bos_flags = {t.get('bos_from_tokenizer') for t in rows}
    manifest = {
        'schema': 'gemma-doc-scores/1', 'created_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'model': {'name': 'google/gemma-4-e4b-it (AMD Ryzen AI NPU build, int4)',
                  'digest': f'{MODEL_REPO}@{MODEL_REVISION}', 'repo': MODEL_REPO, 'revision': MODEL_REVISION,
                  'path': str(Path(model_dir).relative_to(ROOT)) if Path(model_dir).is_relative_to(ROOT) else str(model_dir),
                  'files': main_file_hashes(Path(model_dir)) if hash_files else 'not hashed'},
        'build': build or {},
        'flags': {'route': ROUTE, 'route_detail': ROUTE_DETAIL, 'search_options': SEARCH_OPTIONS,
                  'max_length': f'n_tokens + {MAX_LENGTH_MARGIN}', 'fresh_model_per_document': True,
                  'generators_per_model': 1, 'warmup_decodes': 0,
                  'device_types': sorted({str(t.get('device_type')) for t in rows})},
        'template': 'none',
        'tokenization': {'tokenizer': 'og.Tokenizer of the OGA model folder (tokenizer.json)', 'bos_id': BOS,
                         'bos_from_tokenizer': sorted(bos_flags, key=str), 'raw_text': True, 'chat_template': None},
        'selection': {'control_published_after': published_after, 'rule': 'memorized rows all; control rows with '
                      'metadata.published_at strictly after the cutoff', 'cutoff_source': 'data/cutoffs.yaml '
                      'google/gemma-4-e4b-it', 'counts': selection_counts(documents)},
        'statistics': {'logprob': 'log p(token t | tokens < t) from the full softmax at t-1, float64, nats',
                       'entropy': 'entropy of the full distribution at t-1, nats',
                       'rank': '1 + number of vocabulary entries with a strictly larger logit'},
        'n_vocab': N_VOCAB, 'n_documents': len(documents), 'n_tokens': total_tokens,
        'n_scored_positions': total_tokens - len(documents), 'scoring_seconds': scoring_seconds,
        'load_seconds': load_seconds, 'tokens_per_second': total_tokens / scoring_seconds if scoring_seconds else None,
        'documents': [{'doc_id': d['doc_id'], 'text_sha256': d['text_sha256'], 'n_tokens': done[d['doc_id']]['n_tokens'],
                       **{k: t.get(k) for k in ('seconds', 'load_seconds', 'logits_sha256', 'pid', 'finished_at')}}
                      for d, t in zip(documents, rows, strict=True)],
        'scores_sha256': hashlib.sha256(records_path.read_bytes()).hexdigest(),
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    return manifest


# ---------- subcommands ----------

def cmd_score(args):
    import onnxruntime_genai as og

    published_after = None if args.published_after == 'none' else args.published_after
    documents = select_documents(load_documents(), published_after)
    if args.docs:
        order = args.docs.split(',')
        by_id = {d['doc_id']: d for d in documents}
        missing = [d for d in order if d not in by_id]
        if missing:
            raise SystemExit(f'not in the selection: {missing}')
        documents = [by_id[d] for d in order]
    out = Path(args.output)
    started = time.perf_counter()
    complete = score_selection(og, documents, out, published_after=published_after, check=args.check,
                               og_version=og.__version__)
    if not complete:
        raise SystemExit('selection incomplete')
    build = {'onnxruntime_genai': og.__version__,
             'package': {name: package_version(name) for name in ('onnxruntime-genai-directml-ryzenai', 'numpy')},
             'python': sys.version, 'npu_driver': npu_driver(), 'executable': Path(sys.executable).name}
    manifest = finalize(out, documents, published_after=published_after, og_version=og.__version__, build=build,
                        hash_files=not args.no_hash)
    print(json.dumps({k: manifest[k] for k in ('n_documents', 'n_tokens', 'scoring_seconds', 'load_seconds',
                                               'tokens_per_second')} | {'session_seconds': time.perf_counter() - started}))


def compare_runs(a: Path, b: Path):
    """Bitwise comparison of two outputs of the same documents (any order), plus their direct checks."""
    records = [{r['doc_id']: r for r in read_jsonl(p / 'scores.jsonl')} for p in (a, b)]
    timings = [{t['doc_id']: t for t in read_jsonl(p / 'timings.jsonl')} for p in (a, b)]
    if set(records[0]) != set(records[1]):
        raise SystemExit('the two runs scored different documents')
    report = {'documents': {}}
    for doc_id in sorted(records[0]):
        x, y = records[0][doc_id], records[1][doc_id]
        tx, ty = timings[0][doc_id], timings[1][doc_id]
        entry = {'n_tokens': x['n_tokens'], 'token_ids_equal': x['token_ids'] == y['token_ids'],
                 'arrays_bitwise_equal': all(x[k] == y[k] for k in ('logprob', 'entropy', 'rank')),
                 'logits_sha256_equal': tx['logits_sha256'] == ty['logits_sha256'], 'logits_sha256': tx['logits_sha256'],
                 'pids': [tx['pid'], ty['pid']], 'device_types': [tx['device_type'], ty['device_type']]}
        for label, t in (('a', tx), ('b', ty)):
            c = t.get('check')
            if c:
                n = c['positions']
                entry[f'check_{label}'] = {
                    'positions': n, 'all_shapes_1x1x262144': c['shape_ok'] == n, 'all_finite': c['finite'] == n,
                    'rank_agrees_with_argsort': c['rank_agree'] == n,
                    'rank_one_iff_target_has_max_logit': c['rank_one_iff_argmax_logit'] == n,
                    'argmax_is_target': c['argmax_is_target'], 'rank_one': c['rank_one'],
                    'max_abs_diff_logprob': c['max_abs_diff_logprob'], 'max_abs_diff_entropy': c['max_abs_diff_entropy'],
                    'max_abs_sum_p_minus_1': c['max_abs_sum_p_minus_1']}
        if tx.get('check') and ty.get('check'):
            entry['argmax_sequences_equal'] = tx['check']['argmax'] == ty['check']['argmax']
        report['documents'][doc_id] = entry
    docs = report['documents'].values()
    report['all_bitwise_identical'] = all(e['arrays_bitwise_equal'] and e['logits_sha256_equal'] and e['token_ids_equal']
                                          for e in docs)
    report['all_checks_pass'] = all(e[f'check_{s}']['all_shapes_1x1x262144'] and e[f'check_{s}']['all_finite'] and
                                    e[f'check_{s}']['rank_agrees_with_argsort'] and
                                    e[f'check_{s}']['rank_one_iff_target_has_max_logit'] and
                                    e[f'check_{s}']['max_abs_diff_logprob'] < 1e-9
                                    for e in docs for s in ('a', 'b') if f'check_{s}' in e)
    return report


def cmd_compare(args):
    report = compare_runs(Path(args.a), Path(args.b))
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=1) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=1))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='command', required=True)
    score = sub.add_parser('score')
    score.add_argument('--output', default=str(OUT / 'scores'))
    score.add_argument('--docs', help='comma-separated doc ids, scored in this order')
    score.add_argument('--published-after', default=CUTOFF, help="control cutoff, or 'none'")
    score.add_argument('--check', action='store_true', help='direct full-vector check at every position')
    score.add_argument('--no-hash', action='store_true', help='skip hashing the model files (verification runs)')
    compare = sub.add_parser('compare')
    compare.add_argument('a')
    compare.add_argument('b')
    compare.add_argument('--output')
    args = parser.parse_args(argv)
    {'score': cmd_score, 'compare': cmd_compare}[args.command](args)


if __name__ == '__main__':
    main()
