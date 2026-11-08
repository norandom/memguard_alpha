"""Offline tests for the R37 document-likelihood scorer math and evaluation (no model, no network)."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from scripts.evaluate_document_likelihood import evaluate_held_out, freeze_on_calibration, main
from scripts.score_documents_llamacpp import reference_positions, token_statistics

SPLITS = {'calibration': (12, 20), 'held_out': (8, 8)}


def test_token_statistics_matches_brute_force():
    rng = np.random.default_rng(3)
    logits = rng.normal(0, 3, (5, 40)).astype(np.float32)
    logits[2, 7] = logits[2, 9] = logits[2].max() + 1.0       # tie at the top
    targets = np.array([0, 5, 9, 39, int(np.argmax(logits[4]))])
    logprob, entropy, rank = token_statistics(logits, targets)
    for i, target in enumerate(targets):
        x = logits[i].astype(np.float64)
        p = np.exp(x) / np.exp(x).sum()
        assert logprob[i] == pytest.approx(np.log(p[target]), abs=1e-12)
        assert entropy[i] == pytest.approx(-(p * np.log(p)).sum(), abs=1e-12)
        assert rank[i] == 1 + int((x > x[target]).sum())
    assert rank[2] == 1 and rank[4] == 1                      # ties share rank 1


def test_token_statistics_survives_large_logits():
    logits = np.array([[1000.0, 0.0, -1000.0]], dtype=np.float32)
    logprob, entropy, rank = token_statistics(logits, [0])
    assert np.isfinite(logprob).all() and logprob[0] == pytest.approx(0.0) and entropy[0] >= 0 and rank[0] == 1


def test_reference_positions_cover_edges_and_ubatch_boundary():
    assert reference_positions(577)[:3] == [1, 2, 3] and 576 in reference_positions(577)
    assert {511, 512, 513} <= set(reference_positions(577))
    assert max(reference_positions(300)) == 299


def _corpus(tmp_path, rng):
    data = tmp_path / 'data'
    records = []
    for split, (n_mem, n_ctl) in SPLITS.items():
        (data / split).mkdir(parents=True)
        for group, count, label in (('is_memorized', n_mem, 1), ('oos_control', n_ctl, 0)):
            lines = []
            for index in range(count):
                text = f'{split} {group} {index} ' + 'press release ' * int(rng.integers(5, 15))
                lines.append(json.dumps({'prompt': text, 'label': label, 'metadata': {}}))
                k = int(rng.integers(30, 80))
                shift = 0.7 * label
                records.append({'doc_id': f'{split}/{group}/{index}', 'split': split, 'label': label,
                                'text_sha256': hashlib.sha256(text.encode()).hexdigest(), 'n_tokens': k + 1,
                                'logprob': (-np.abs(rng.normal(2 - shift, 1, k))).tolist(),
                                'entropy': np.abs(rng.normal(1.5 - shift, 0.5, k)).tolist(),
                                'rank': rng.integers(1, 30, k).tolist()})
            (data / split / f'{group}.jsonl').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return data, records


def test_fit_uses_calibration_only(tmp_path):
    data, records = _corpus(tmp_path, np.random.default_rng(0))
    frozen, calibrator = freeze_on_calibration(records, data)
    # Held-out records changed arbitrarily (or removed): the frozen fit is identical.
    tampered = [r | {'logprob': [-9.0] * len(r['logprob'])} if r['split'] == 'held_out' else r for r in records]
    calibration_only = [r for r in records if r['split'] == 'calibration']
    for variant in (tampered, calibration_only):
        again, _ = freeze_on_calibration(variant, data)
        assert again == frozen
    assert frozen['fitted_on'] == 'calibration' and frozen['n_calibration'] == 32
    held = evaluate_held_out(records, data, frozen, calibrator)
    assert held['n_held_out'] == 16 and set(held['single_feature']) == set(frozen['features'])


def test_end_to_end_writes_frozen_before_held_out(tmp_path):
    data, records = _corpus(tmp_path, np.random.default_rng(1))
    scores = tmp_path / 'scores'
    scores.mkdir()
    raw = ''.join(json.dumps(r) + '\n' for r in records).encode()
    (scores / 'scores.jsonl').write_bytes(raw)
    manifest = {'model': {'name': 'synthetic', 'digest': 'sha256:0'}, 'build': {}, 'flags': {}, 'template': 'none',
                'tokenization': {}, 'n_vocab': 30, 'n_documents': len(records), 'n_tokens': 0,
                'n_scored_positions': 0, 'scores_sha256': hashlib.sha256(raw).hexdigest()}
    (scores / 'manifest.json').write_text(json.dumps(manifest))
    out = tmp_path / 'eval'
    main(['--scores', str(scores), '--output', str(out), '--data', str(data)])
    result = json.loads((out / 'evaluation.json').read_text())
    assert result['frozen_sha256'] == hashlib.sha256((out / 'frozen.json').read_bytes()).hexdigest()
    assert result['held_out']['calibrator_discrimination']['n_resamples'] == 1000
    with pytest.raises(SystemExit):                            # hash check guards identity
        (scores / 'scores.jsonl').write_bytes(raw + b'\n')
        main(['--scores', str(scores), '--output', str(out), '--data', str(data)])
