"""Evaluate document-likelihood features for one model (release-1.0, register R37).

Runs in the dev container. Reads the teacher-forced per-token scores written
natively by ``scripts/score_documents_llamacpp.py`` (default
``/outputs/gemma-doc/scores``) or ``scripts/score_documents_oga.py`` (Gemma 4,
``/outputs/gemma4-doc/scores`` with ``--published-after 2025-01-31``), checks
every record against the EDGAR splits by text hash, computes ``recall_guard.mia.document_likelihood`` features, and:

1. fits the calibrator on the calibration split ONLY, reports 5-fold stratified
   CV AUC there, chooses each single feature's sign there, and writes
   ``frozen.json`` (feature list, seed, signs, coefficients, calibration hashes)
   BEFORE any held-out feature is computed;
2. then scores the held-out split: calibrator AUC (1,000-resample stratified
   bootstrap 95 % interval), Brier with interval, base-rate reference Brier,
   ECE and reliability bins, and each single feature's unfitted AUC.

``--published-after DATE`` keeps only control rows published strictly after
DATE (memorized rows all, document ids keep their corpus index); it must equal
the scores manifest's ``selection.control_published_after``. Without it the
splits are used whole and the outputs are as before.

Writes ``evaluation.json`` and ``evaluation.md`` next to ``frozen.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from recall_guard.harness.local_reporting import held_out_calibration, held_out_discrimination
from recall_guard.mia.document_likelihood import (
    FEATURES,
    SEED,
    cross_validated_auc,
    document_features,
    feature_matrix,
    feature_signs,
    fit_calibrator,
)

ROOT = Path(__file__).resolve().parent.parent
GROUPS = ('is_memorized', 'oos_control')


def keep_document(label: int, published_at, published_after) -> bool:
    """Memorized rows always; control rows only if published strictly after ``published_after``."""
    if published_after is None or label == 1:
        return True
    return published_at is not None and published_at > published_after


def load_split(data_dir: Path, split: str, published_after=None) -> dict[str, dict]:
    documents = {}
    for group in GROUPS:
        for index, line in enumerate((data_dir / split / f'{group}.jsonl').read_text(encoding='utf-8').splitlines()):
            item = json.loads(line)
            if not keep_document(int(item['label']), item.get('metadata', {}).get('published_at'), published_after):
                continue
            documents[f'{split}/{group}/{index}'] = {'text': item['prompt'], 'label': int(item['label'])}
    return documents


def split_features(records: list[dict], documents: dict[str, dict], split: str):
    """Feature rows and labels for one split, after identity checks against the corpus."""
    chosen = [r for r in records if r['split'] == split]
    if {r['doc_id'] for r in chosen} != set(documents):
        raise SystemExit(f'{split}: scored documents do not match the corpus')
    rows, labels, ids = [], [], []
    for record in sorted(chosen, key=lambda r: r['doc_id']):
        document = documents[record['doc_id']]
        if hashlib.sha256(document['text'].encode('utf-8')).hexdigest() != record['text_sha256']:
            raise SystemExit(f"{record['doc_id']}: text hash mismatch")
        if document['label'] != record['label'] or len(record['logprob']) != record['n_tokens'] - 1:
            raise SystemExit(f"{record['doc_id']}: label or length mismatch")
        rows.append(document_features(record['logprob'], record['entropy'], record['rank'], document['text']))
        labels.append(document['label'])
        ids.append(record['doc_id'])
    return rows, np.array(labels), ids


def freeze_on_calibration(records, data_dir: Path, published_after=None) -> tuple[dict, object]:
    """Everything decided from the calibration split alone."""
    rows, labels, ids = split_features(records, load_split(data_dir, 'calibration', published_after), 'calibration')
    matrix = feature_matrix(rows)
    calibrator = fit_calibrator(matrix, labels)
    model = calibrator.pipeline
    frozen = {'features': list(FEATURES), 'seed': SEED,
              'calibrator': 'StandardScaler + LogisticRegression(class_weight=balanced, random_state=seed)',
              'fitted_on': 'calibration', 'n_calibration': int(labels.size),
              'n_memorized': int(labels.sum()), 'n_control': int((labels == 0).sum()),
              'calibration_doc_ids_sha256': hashlib.sha256('\n'.join(ids).encode()).hexdigest(),
              'coefficients_standardised': dict(zip(FEATURES, model[-1].coef_[0].tolist(), strict=True)),
              'intercept': float(model[-1].intercept_[0]),
              'scaler_mean': dict(zip(FEATURES, model[0].mean_.tolist(), strict=True)),
              'scaler_scale': dict(zip(FEATURES, model[0].scale_.tolist(), strict=True)),
              'single_feature_signs': feature_signs(matrix, labels),
              'cv': cross_validated_auc(matrix, labels),
              'calibration_feature_means': {g: dict(zip(FEATURES, matrix[labels == v].mean(axis=0).tolist(), strict=True))
                                            for g, v in (('memorized', 1), ('control', 0))}}
    if published_after is not None:
        frozen['selection'] = {'control_published_after': published_after}
    return frozen, calibrator


def evaluate_held_out(records, data_dir: Path, frozen: dict, calibrator) -> dict:
    published_after = frozen.get('selection', {}).get('control_published_after')
    rows, labels, _ = split_features(records, load_split(data_dir, 'held_out', published_after), 'held_out')
    matrix = feature_matrix(rows, frozen['features'])
    probabilities = calibrator.predict_memorized(matrix)
    single = {}
    for j, name in enumerate(frozen['features']):
        sign = frozen['single_feature_signs'][name]
        single[name] = {'sign': sign, **held_out_discrimination(sign * matrix[:, j], labels, seed=SEED)}
    return {'n_held_out': int(labels.size), 'n_memorized': int(labels.sum()), 'n_control': int((labels == 0).sum()),
            'calibrator_discrimination': held_out_discrimination(probabilities, labels, seed=SEED),
            'calibrator_calibration': held_out_calibration(probabilities, labels, seed=SEED),
            'single_feature': single,
            'held_out_feature_means': {g: dict(zip(frozen['features'], matrix[labels == v].mean(axis=0).tolist(), strict=True))
                                       for g, v in (('memorized', 1), ('control', 0))}}


def _ci(d, key='auc', low='ci_low', high='ci_high'):
    return f"{d[key]:.3f} [{d[low]:.3f}, {d[high]:.3f}]"


def render(frozen, held, scores_manifest) -> str:
    cal, disc = held['calibrator_calibration'], held['calibrator_discrimination']
    lines = [f"# Document likelihood: {scores_manifest['model']['name']}", '',
             f"Model {scores_manifest['model']['digest']}; template none; "
             f"{scores_manifest['n_documents']} documents, {scores_manifest['n_scored_positions']} scored positions.", '',
             '| Feature | sign (calibration) | held-out AUC [95 % CI] |', '|---|---|---|']
    lines += [f"| {name} | {v['sign']:+d} | {_ci(v)} |" for name, v in held['single_feature'].items()]
    cv = frozen['cv']
    lines += ['', f"Calibrator 5-fold CV AUC on calibration: mean {cv['mean_auc']:.3f} (sd {cv['std_auc']:.3f}), "
                  f"pooled {cv['pooled_auc']:.3f}.",
              f"Held-out calibrator AUC {_ci(disc)}; Brier {_ci(cal, 'brier', 'brier_ci_low', 'brier_ci_high')} "
              f"(base-rate reference {cal['brier_reference']:.3f}); ECE {cal['ece']:.3f}.", '',
              '| bin | count | mean predicted | observed memorized |', '|---|---|---|---|']
    lines += [f"| {b['low']:.1f}-{b['high']:.1f} | {b['count']} | {b['mean_predicted']:.3f} | "
              f"{b['observed_memorized_fraction']:.3f} |" for b in cal['reliability_bins']]
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--scores', default='/outputs/gemma-doc/scores')
    parser.add_argument('--output', default='/outputs/gemma-doc/evaluation')
    parser.add_argument('--data', default=str(ROOT / 'data/edgar'))
    parser.add_argument('--published-after', help='keep control rows published strictly after this date')
    args = parser.parse_args(argv)
    scores_dir, out, data_dir = Path(args.scores), Path(args.output), Path(args.data)
    scores_manifest = json.loads((scores_dir / 'manifest.json').read_text(encoding='utf-8'))
    raw = (scores_dir / 'scores.jsonl').read_bytes()
    if hashlib.sha256(raw).hexdigest() != scores_manifest['scores_sha256']:
        raise SystemExit('scores.jsonl does not match its manifest')
    if scores_manifest.get('selection', {}).get('control_published_after') != args.published_after:
        raise SystemExit('--published-after does not match the scores manifest selection')
    records = [json.loads(line) for line in raw.decode('utf-8').splitlines()]
    out.mkdir(parents=True, exist_ok=True)

    frozen, calibrator = freeze_on_calibration(records, data_dir, args.published_after)
    frozen['scores_sha256'] = scores_manifest['scores_sha256']
    frozen['model_digest'] = scores_manifest['model']['digest']
    (out / 'frozen.json').write_text(json.dumps(frozen, indent=2) + '\n', encoding='utf-8')
    frozen_sha256 = hashlib.sha256((out / 'frozen.json').read_bytes()).hexdigest()

    held = evaluate_held_out(records, data_dir, frozen, calibrator)
    result = {'frozen_sha256': frozen_sha256, 'frozen': frozen, 'held_out': held,
              'scores_manifest': {k: scores_manifest[k] for k in ('model', 'build', 'flags', 'template', 'tokenization',
                                                                  'n_vocab', 'n_documents', 'n_tokens', 'scores_sha256')}}
    if 'selection' in scores_manifest:
        result['scores_manifest']['selection'] = scores_manifest['selection']
    (out / 'evaluation.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    (out / 'evaluation.md').write_text(render(frozen, held, scores_manifest), encoding='utf-8')
    print(render(frozen, held, scores_manifest))


if __name__ == '__main__':
    main()
