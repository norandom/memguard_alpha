"""Offline tests for the Gemma 4 OGA document scorer and the control filter (R37; no model, no NPU)."""
from __future__ import annotations

import hashlib
import json
import types
from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts import evaluate_document_likelihood as evaluate
from scripts import score_documents_oga as oga
from scripts.score_documents_llamacpp import token_statistics

ROOT = Path(__file__).resolve().parents[2]
VOCAB = 50


def _fake_logits(prefix):
    seed = int(hashlib.sha256(json.dumps(prefix).encode()).hexdigest()[:8], 16)
    return np.random.default_rng(seed).normal(0, 3, VOCAB).astype(np.float32)


def fake_og(loads):
    """A stand-in for onnxruntime_genai: logits are a function of the appended prefix only."""

    class Model:
        device_type = 'Fake'

        def __init__(self, path):
            loads.append(path)

    class Tokenizer:
        def __init__(self, model):
            pass

        def encode(self, text):
            return np.array([3 + (ord(c) % (VOCAB - 3)) for c in text], dtype=np.int32)

    class GeneratorParams:
        def __init__(self, model):
            self.options = {}

        def set_search_options(self, **options):
            self.options = options

    class Generator:
        def __init__(self, model, params):
            self.prefix = []

        def append_tokens(self, tokens):
            self.prefix += [int(t) for t in tokens]

        def get_logits(self):
            return _fake_logits(self.prefix).reshape(1, 1, VOCAB)

    return types.SimpleNamespace(Model=Model, Tokenizer=Tokenizer, GeneratorParams=GeneratorParams, Generator=Generator)


@pytest.fixture
def small(monkeypatch):
    monkeypatch.setattr(oga, 'N_VOCAB', VOCAB)
    loads = []
    return fake_og(loads), loads


def _documents(n=5):
    return [{'doc_id': f'calibration/is_memorized/{i}', 'split': 'calibration', 'group': 'is_memorized', 'index': i,
             'label': 1, 'accession': f'a{i}', 'published_at': '2020-01-01', 'text': f'press release {i} ' * (2 + i),
             'text_sha256': hashlib.sha256((f'press release {i} ' * (2 + i)).encode()).hexdigest()} for i in range(n)]


def test_decode_route_matches_full_prefix_statistics(small):
    og, _ = small
    tokens = [oga.BOS, 7, 9, 11, 13, 7]
    logprob, entropy, rank, digest, check = oga.score_tokens(og, og.Model('m'), tokens, n_vocab=VOCAB, check=True)
    expected = token_statistics(np.stack([_fake_logits(tokens[:i]) for i in range(1, len(tokens))]), tokens[1:])
    assert logprob == pytest.approx(expected[0].tolist(), abs=1e-12)
    assert entropy == pytest.approx(expected[1].tolist(), abs=1e-12) and rank == expected[2].tolist()
    assert check['positions'] == check['rank_agree'] == check['rank_one_iff_argmax_logit'] == len(tokens) - 1
    assert check['max_abs_diff_logprob'] < 1e-9 and len(digest) == 64


def test_direct_check_handles_ties():
    row = np.array([1.0, 3.0, 3.0, -2.0], dtype=np.float32)
    assert oga.direct_check(row, 2)['rank'] == 1 and token_statistics(row[None], [2])[2][0] == 1
    assert oga.direct_check(row, 0)['rank'] == 3 == token_statistics(row[None], [0])[2][0]


def test_tokenize_prepends_bos_once():
    class Tok:
        def __init__(self, ids):
            self.ids = ids

        def encode(self, text):
            return self.ids
    assert oga.tokenize(Tok([5, 6]), 'x') == ([oga.BOS, 5, 6], False)
    assert oga.tokenize(Tok([oga.BOS, 5]), 'x') == ([oga.BOS, 5], True)
    with pytest.raises(RuntimeError):
        oga.tokenize(Tok([5, oga.BOS]), 'x')


def test_record_format_matches_llamacpp_scorer(small, tmp_path):
    og, loads = small
    documents = _documents(2)
    assert oga.score_selection(og, documents, tmp_path, model_dir='m', og_version='t', log=open(tmp_path / 'log', 'w'))
    records = oga.read_jsonl(tmp_path / 'scores.jsonl')
    keys = {'doc_id', 'split', 'group', 'index', 'label', 'accession', 'text_sha256', 'n_vocab', 'n_tokens',
            'token_ids', 'logprob', 'entropy', 'rank'}
    assert all(set(r) == keys for r in records)              # the keys cmd_score of score_documents_llamacpp writes
    assert all(len(r['logprob']) == r['n_tokens'] - 1 and r['token_ids'][0] == oga.BOS for r in records)
    assert len(loads) == 2                                     # one fresh model per document


def test_resume_skips_written_documents_and_repairs_a_cut_line(small, tmp_path):
    og, loads = small
    documents = _documents(5)
    log = open(tmp_path / 'log', 'w')
    reference = tmp_path / 'reference'
    oga.score_selection(og, documents, reference, model_dir='m', og_version='t', log=log)
    oga.finalize(reference, documents, model_dir='m', og_version='t', hash_files=False)

    out = tmp_path / 'resumed'
    real = oga.score_document
    calls = []

    def interrupted(*args, **kwargs):
        calls.append(1)
        if len(calls) == 3:
            raise KeyboardInterrupt                            # the run stops while scoring the third document
        return real(*args, **kwargs)

    oga.score_document = interrupted
    try:
        with pytest.raises(KeyboardInterrupt):
            oga.score_selection(og, documents, out, model_dir='m', og_version='t', log=log)
    finally:
        oga.score_document = real
    assert len(oga.read_jsonl(out / 'scores.jsonl')) == 2
    with (out / 'scores.jsonl').open('a', encoding='utf-8') as handle:
        handle.write('{"doc_id": "calibration/is_memo')      # a write cut short
    loads.clear()
    assert oga.score_selection(og, documents, out, model_dir='m', og_version='t', log=log)
    assert len(loads) == 3                                     # only the three missing documents were scored
    manifest = oga.finalize(out, documents, model_dir='m', og_version='t', hash_files=False)
    assert (out / 'scores.jsonl').read_bytes() == (reference / 'scores.jsonl').read_bytes()
    assert manifest['scores_sha256'] == hashlib.sha256((out / 'scores.jsonl').read_bytes()).hexdigest()
    assert manifest['flags']['route'] == 'decode' and manifest['n_documents'] == 5
    loads.clear()
    assert oga.score_selection(og, documents, out, model_dir='m', og_version='t', log=log) and loads == []


def test_resume_refuses_another_route_or_selection(small, tmp_path):
    og, _ = small
    documents = _documents(3)
    log = open(tmp_path / 'log', 'w')
    oga.score_selection(og, documents, tmp_path, model_dir='m', og_version='t', log=log)
    with pytest.raises(SystemExit, match='run.json differs'):
        oga.score_selection(og, documents, tmp_path, model_dir='m', og_version='t', check=True, log=log)
    with pytest.raises(SystemExit, match='run.json differs'):
        oga.score_selection(og, documents, tmp_path, model_dir='m', published_after=None, og_version='t', log=log)
    changed = [d | {'text_sha256': '0' * 64} if i == 0 else d for i, d in enumerate(documents)]
    with pytest.raises(SystemExit, match='does not belong'):
        oga.completed(tmp_path, changed)
    with pytest.raises(SystemExit, match='not scored yet'):
        oga.finalize(tmp_path, _documents(4), model_dir='m', hash_files=False)


def test_compare_runs_detects_differences(small, tmp_path):
    og, _ = small
    documents = _documents(2)
    log = open(tmp_path / 'log', 'w')
    for name, order in (('a', documents), ('b', documents[::-1])):
        oga.score_selection(og, order, tmp_path / name, model_dir='m', check=True, og_version='t', log=log)
    report = oga.compare_runs(tmp_path / 'a', tmp_path / 'b')
    assert report['all_bitwise_identical'] and report['all_checks_pass']
    lines = (tmp_path / 'b' / 'scores.jsonl').read_text().splitlines()
    record = json.loads(lines[0])
    record['logprob'][0] += 1e-12
    (tmp_path / 'b' / 'scores.jsonl').write_text('\n'.join([json.dumps(record), *lines[1:]]) + '\n')
    assert not oga.compare_runs(tmp_path / 'a', tmp_path / 'b')['all_bitwise_identical']


def test_control_filter_on_the_edgar_splits():
    cutoffs = yaml.safe_load((ROOT / 'data/cutoffs.yaml').read_text(encoding='utf-8'))
    assert str(cutoffs['models']['google/gemma-4-e4b-it']) == oga.CUTOFF
    selected = oga.select_documents(oga.load_documents())
    assert oga.selection_counts(selected) == {'calibration/is_memorized': 40, 'calibration/oos_control': 74,
                                              'held_out/is_memorized': 40, 'held_out/oos_control': 33}
    assert all(d['published_at'] > oga.CUTOFF for d in selected if d['label'] == 0)
    for split in ('calibration', 'held_out'):                  # the evaluator applies the same rule
        assert set(evaluate.load_split(ROOT / 'data/edgar', split, oga.CUTOFF)) == {
            d['doc_id'] for d in selected if d['split'] == split}
    assert len(evaluate.load_split(ROOT / 'data/edgar', 'held_out')) == 80   # unchanged without a cutoff
    with pytest.raises(SystemExit, match='memorized'):
        oga.select_documents(oga.load_documents(), '2010-01-01')


def _corpus(tmp_path, rng):
    """Corpus with dated controls; only those after 2025-01-31 are scored."""
    data, records = tmp_path / 'data', []
    for split, (n_mem, n_ctl) in {'calibration': (12, 24), 'held_out': (8, 12)}.items():
        (data / split).mkdir(parents=True)
        for group, count, label in (('is_memorized', n_mem, 1), ('oos_control', n_ctl, 0)):
            lines = []
            for index in range(count):
                text = f'{split} {group} {index} ' + 'press release ' * int(rng.integers(5, 15))
                date = '2020-01-01' if label else ('2024-10-01' if index % 3 == 0 else '2025-06-01')
                lines.append(json.dumps({'prompt': text, 'label': label, 'metadata': {'published_at': date}}))
                if label == 0 and date <= oga.CUTOFF:
                    continue
                k = int(rng.integers(30, 80))
                records.append({'doc_id': f'{split}/{group}/{index}', 'split': split, 'label': label,
                                'text_sha256': hashlib.sha256(text.encode()).hexdigest(), 'n_tokens': k + 1,
                                'logprob': (-np.abs(rng.normal(2 - 0.7 * label, 1, k))).tolist(),
                                'entropy': np.abs(rng.normal(1.5, 0.5, k)).tolist(),
                                'rank': rng.integers(1, 30, k).tolist()})
            (data / split / f'{group}.jsonl').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return data, records


def test_evaluate_with_published_after(tmp_path):
    data, records = _corpus(tmp_path, np.random.default_rng(2))
    scores = tmp_path / 'scores'
    scores.mkdir()
    raw = ''.join(json.dumps(r) + '\n' for r in records).encode()
    (scores / 'scores.jsonl').write_bytes(raw)
    manifest = {'model': {'name': 'synthetic', 'digest': 'x@0'}, 'build': {}, 'flags': {}, 'template': 'none',
                'tokenization': {}, 'n_vocab': 30, 'n_documents': len(records), 'n_tokens': 0, 'n_scored_positions': 0,
                'selection': {'control_published_after': oga.CUTOFF}, 'scores_sha256': hashlib.sha256(raw).hexdigest()}
    (scores / 'manifest.json').write_text(json.dumps(manifest))
    out = tmp_path / 'eval'
    with pytest.raises(SystemExit, match='scored documents do not match'):   # filtered scores, unfiltered corpus
        evaluate.freeze_on_calibration(records, data)
    with pytest.raises(SystemExit, match='does not match the scores manifest'):
        evaluate.main(['--scores', str(scores), '--output', str(out), '--data', str(data)])
    evaluate.main(['--scores', str(scores), '--output', str(out), '--data', str(data), '--published-after', oga.CUTOFF])
    result = json.loads((out / 'evaluation.json').read_text())
    frozen = json.loads((out / 'frozen.json').read_text())
    assert frozen['selection'] == {'control_published_after': oga.CUTOFF}
    assert frozen['n_memorized'] == 12 and frozen['n_control'] == 16
    assert result['held_out']['n_memorized'] == 8 and result['held_out']['n_control'] == 8
    assert result['scores_manifest']['selection']['control_published_after'] == oga.CUTOFF
