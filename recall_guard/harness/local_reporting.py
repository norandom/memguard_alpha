"""Safe local reports. Rows are keyed by item/arm/zero-based draw, never position.

All probabilities and AUC are dimensionless existing scorer outputs. Bytes and
seconds are explicit in resource column names. No prompts, output tokens, raw
responses or exception strings are exported. The one provider HTTP failure that
stopped a batch keeps its bounded body (R2.7) after secret and prompt redaction.
Unknown quantities are null.
"""
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from recall_guard.harness.evaluator import _parse_confidence, _parse_direction
from recall_guard.harness.local_execution import HostResources, MeasurementRow

_STAGES = ('probe', 'baseline', 'calibration', 'single', 'ensemble')
_MODES = {'CPU', 'GPU', 'CPU/GPU offload', 'NPU', 'mixed NPU', 'unknown'}
_REASONS = {'one_selected_loaded_model_required', 'effective_context_not_established',
            'probe_condition_mismatched', 'unexpected_calibration_order', 'budget_or_cancellation',
            'invalid_evidence_identity_or_fit', 'host_or_calibration_unavailable',
            'probe_unavailable', 'invalid_evidence_or_identity', 'generation_unavailable',
            'parse_failure', 'timeout', 'no_logprobs', 'error'}


def _key(item_id):
    return hashlib.sha256(item_id.encode('utf-8')).hexdigest()


def _planned_rows(plan):
    yield MeasurementRow('__probe__', 'probe', 0)
    for item in plan.items:
        arms = [('single', 1), ('ensemble', plan.ensemble_draws)] if item.split == 'held_out' else (
            [('baseline', 1), ('calibration', 1)] if item.split == 'oos_control' else [('calibration', 1)])
        for arm, count in arms:
            for draw in range(count):
                yield MeasurementRow(item.item_id, arm, draw)


def _safe_value(value, budget):
    if isinstance(value, str):
        value = budget.redact_report_text(value)
        # A model/template/runtime field must not become a route for corpus export.
        for item in budget.plan.items:
            value = value.replace(item.prompt, '[PRIVATE]')
        return value
    if isinstance(value, dict):
        # Runtime template metadata can contain system/private prompt text.
        return {('template_sha256' if key == 'template' else key):
                (None if child is None else _key(child)) if key == 'template'
                else _safe_value(child, budget) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe_value(child, budget) for child in value]
    return value


def _row_record(row, allowed, budget):
    response = row.response
    completed = response is not None
    parsed = completed and _parse_direction(response.content) is not None and _parse_confidence(response.content) is not None
    usable = bool(allowed and row.status == 'scored' and row.score is not None
                  and row.score.parse_ok and row.score.p_memorized is not None)
    identity = None if response is None or response.local_identity is None else response.local_identity.to_dict()
    # Feature-based score: needs a captured, non-failed response with valid evidence, not a parsed answer.
    feature = (row.feature_p_memorized if allowed and row.status in {'scored', 'unparsed'} else None)
    return {'item_sha256': _key(row.item_id), 'arm': row.arm, 'draw_index': row.draw_index,
            'status': row.status if row.status in {'not_attempted', 'received', 'failed', 'scored', 'unparsed'} else 'unknown',
            'attempted': bool(row.physical_call_ids), 'completed': completed, 'parsed': bool(parsed),
            'usable': usable, 'physical_call_ids': list(row.physical_call_ids),
            'reason': row.reason if row.reason in _REASONS else None,
            'p_memorized': row.score.p_memorized if usable else None,
            'feature_scored': feature is not None, 'feature_p_memorized': feature,
            'local_identity': _safe_value(identity, budget)}


HELD_OUT_RESAMPLES = 1000


def held_out_discrimination(scores, labels, *, seed=0, n_resamples=HELD_OUT_RESAMPLES, confidence=0.95):
    """ROC-AUC of labeled held-out scores with a seeded stratified bootstrap interval.

    Each resample draws memorized and control items separately with replacement
    at their observed counts, so every resample has both classes. The interval is
    the percentile interval. Needs at least one usable item per class, else null.
    """
    import numpy as np
    from sklearn.metrics import roc_auc_score

    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels, dtype=int)
    positives, negatives = scores[labels == 1], scores[labels == 0]
    result = {'auc': None, 'ci_low': None, 'ci_high': None, 'confidence': confidence,
              'n_resamples': n_resamples, 'seed': seed, 'method': 'stratified_percentile_bootstrap',
              'n_memorized': int(positives.size), 'n_control': int(negatives.size)}
    if not positives.size or not negatives.size:
        return result
    truth = np.r_[np.ones(positives.size), np.zeros(negatives.size)]
    rng = np.random.default_rng(seed)
    resampled = [roc_auc_score(truth, np.r_[rng.choice(positives, positives.size),
                                            rng.choice(negatives, negatives.size)])
                 for _ in range(n_resamples)]
    tail = (1 - confidence) / 2
    low, high = np.quantile(resampled, [tail, 1 - tail])
    return result | {'auc': float(roc_auc_score(truth, np.r_[positives, negatives])),
                     'ci_low': float(low), 'ci_high': float(high)}


def _split_counts(plan, rows):
    """Per corpus split: logical rows (OOS counts baseline and calibration rows).

    valid = completed and not failed; held-out rows additionally need a
    feature-based score. Neither calibration nor held-out validity requires a
    parsed answer, since both use logprob features; parse failures are in unparsed.
    """
    split_of = {_key(item.item_id): item.split for item in plan.items}
    counts = {}
    for split in ('is_memorized', 'oos_control', 'held_out'):
        group = [r for r in rows if split_of.get(r['item_sha256']) == split
                 and (r['arm'] == 'single' if split == 'held_out' else r['arm'] in {'baseline', 'calibration'})]
        valid = [r for r in group if (r['feature_scored'] if split == 'held_out'
                                      else r['completed'] and r['status'] != 'failed')]
        counts[split] = {'planned': len(group), 'attempted': sum(r['attempted'] for r in group),
                         'valid': len(valid), 'failed': sum(r['status'] == 'failed' for r in group),
                         'unparsed': sum(r['completed'] and not r['parsed'] for r in group)}
    return counts


def _held_out_evaluation(plan, rows, allowed):
    labels = {_key(item.item_id): item.label for item in plan.items if item.split == 'held_out'}
    labeled = {key: label for key, label in labels.items() if label is not None}
    # Feature-based p_memorized (MIA features + fitted calibrator), independent of answer parsing.
    usable = [r for r in rows if r['arm'] == 'single' and r['feature_scored'] and r['item_sha256'] in labeled]
    evaluation = held_out_discrimination([r['feature_p_memorized'] for r in usable],
                                         [labeled[r['item_sha256']] for r in usable], seed=plan.seed)
    reason = ('no_labeled_held_out_items' if not labeled else 'calibrated_claims_not_allowed' if not allowed
              else 'insufficient_usable_items_per_class' if evaluation['auc'] is None else None)
    return {'arm': 'single', 'score': 'feature_p_memorized', 'n_held_out': len(labels), 'n_labeled': len(labeled), 'n_usable': len(usable),
            'reason': reason, **evaluation}


def build_report(budget, result):
    """Assemble safe artifacts from captured evidence without calling a provider.

    completed = response captured; failed = execution/evidence failure (may also
    have a captured response); parsed uses the existing output parser and does
    not establish usable calibration. Unparsed is completed minus parsed.
    Physical retries stay in attempts, logical denominators stay in stages.
    """
    plan = budget.plan
    actual = {(r.item_id, r.arm, r.draw_index): r for r in result.rows}
    if len(actual) != len(result.rows):
        raise ValueError('duplicate_measurement_key')
    rows = [_row_record(actual.pop((r.item_id, r.arm, r.draw_index), r),
                        result.calibrated_claims_allowed, budget) for r in _planned_rows(plan)]
    if actual:
        raise ValueError('unexpected_measurement_key')
    stages = {}
    for arm in _STAGES:
        group = [r for r in rows if r['arm'] == arm]
        counts = {name: sum(r[name] for r in group) for name in ('attempted', 'completed', 'parsed', 'usable')}
        stages[arm] = {'planned': len(group), **counts,
                       'failed': sum(r['status'] == 'failed' for r in group),
                       'unparsed': counts['completed'] - counts['parsed'],
                       'unattempted': len(group) - counts['attempted']}
    singles = {r['item_sha256']: r for r in rows if r['arm'] == 'single' and r['usable']}
    ensembles = []
    if result.calibrated_claims_allowed:
        for item in plan.items:
            score = result.ensembles.get(item.item_id)
            if item.split == 'held_out' and score is not None:
                ensembles.append({'item_sha256': _key(item.item_id), 'n_requested': score.n_requested,
                    'n_parsed': score.n_parsed, 'p_memorized_point': score.p_memorized_point})
    pairs = [{'item_sha256': e['item_sha256'], 'single_p_memorized': singles[e['item_sha256']]['p_memorized'],
              'ensemble_p_memorized_point': e['p_memorized_point']} for e in ensembles
             if e['item_sha256'] in singles and e['p_memorized_point'] is not None]
    held_out = _held_out_evaluation(plan, rows, result.calibrated_claims_allowed)
    observations, resources = [], []
    for stage in ('opening', 'after_probe', 'closing'):
        observation = result.host_observations.get(stage)
        observations.append({'stage': stage, 'observed': observation is not None,
            'provenance': None if observation is None else _safe_value(asdict(observation.provenance), budget)})
        resources.append({'stage': stage, **asdict(HostResources() if observation is None else observation.resources)})
    receipt = budget.safe_receipt()
    # Explicitly select safe scalar attempt fields; arbitrary server diagnostics never pass through.
    attempts = [{name: a.get(name) for name in ('call_id', 'attempt', 'status', 'latency_s', 'http_status',
                                               'usage', 'cost', 'host_execution_stopped')} for a in receipt['attempts']]
    return {'schema_version': 1, 'status': result.status,
        'reason': result.reason if result.reason in _REASONS else None,
        'failure': None if result.failure is None else _safe_value(dict(result.failure), budget),
        'calibrated_claims_allowed': result.calibrated_claims_allowed,
        'calibration_fit_completed': result.calibration_fit_completed,
        'batch_check': None if result.batch_check is None else asdict(result.batch_check),
        'selected_model': _safe_value(plan.config.model, budget), 'plan': plan.safe_export(),
        'observed_execution_mode': result.observed_execution_mode if result.observed_execution_mode in _MODES else 'unknown',
        'npu_verified': result.npu_verified, 'observations': observations, 'resources': resources,
        'hardware': {'cpu': None, 'gpu': None, 'npu': None}, 'concurrent_requests': 1,
        'generation_settings': {k: value if isinstance(value, (int, float, bool)) else None
            for k in ('max_tokens', 'temperature', 'top_p', 'logprobs', 'top_logprobs')
            for value in (plan.config.parameters.get(k),)},
        'stages': stages, 'splits': _split_counts(plan, rows), 'held_out_evaluation': held_out,
        'rows': rows, 'ensembles': ensembles, 'paired_items': pairs,
        'attempts': attempts, 'physical_attempts': len(attempts),
        'claims': {'endpoint_feasibility': 'probe_passed' if result.endpoint_probe_passed else 'not_established',
                   'discrimination': {'calibrator_holdout_auc': result.holdout_auc if result.calibrated_claims_allowed else None,
                                      'held_out_evaluation_auc': held_out['auc']},
                   'calibration_quality': 'not_measured', 'mitigation': 'not_performed',
                   'exact_full_vocabulary_divergence': 'not_performed', 'intervention_before_token_selection': 'not_performed'},
        'historical_nim': {'status': 'NIM unavailable', 'http_status': 504, 'cause': None},
        'frontier_completion_only': 'separate_path_not_performed',
        'limitations': ['Sidecar observations are not atomic identity guarantees.',
            'AUC measures discrimination, not probability calibration or mitigation.',
            'Null means unknown or unavailable, never zero; allocation is not dedicated GPU memory.',
            'Request latency includes transport and host work; load and generation times require separate observations.',
            'Client timeout does not establish that host generation stopped.',
            'Paired rows include only common usable item keys; all draw denominators remain in stages.'],
        'representation': {'rows': 'One planned item/arm/zero-based draw; item_sha256 is a stable key.',
            'identity': 'Template content is represented only by template_sha256.',
            'attempts': 'One physical call_id including retries; join rows via physical_call_ids.',
            'stages': 'Logical counts: completed=response captured; parsed=existing output parser; failed may overlap completed.',
            'probabilities': 'Dimensionless; ensemble point estimate is not consensus.p_memorized.',
            'resources': 'One snapshot stage per row; _bytes are bytes, _s are seconds.'}}


def report_paths(journal):
    return Path(str(journal) + '.report.json'), Path(str(journal) + '.report.md')


def render_markdown(report):
    """Readable summary from safe structured data, with explicit denominators."""
    lines = ['# Local measurement report', '', f"Status: {report['status']}",
             f"Observed mode: {report['observed_execution_mode']}", '',
             '| Stage | Planned | Attempted | Completed | Parsed | Failed | Unparsed | Unattempted | Usable |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for stage, counts in report['stages'].items():
        lines.append('| ' + stage + ' | ' + ' | '.join(str(counts[k]) for k in
            ('planned', 'attempted', 'completed', 'parsed', 'failed', 'unparsed', 'unattempted', 'usable')) + ' |')
    lines += ['', '| Split | Planned | Attempted | Valid | Failed | Unparsed |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for split, counts in report.get('splits', {}).items():
        lines.append('| ' + split + ' | ' + ' | '.join(str(counts[k]) for k in
            ('planned', 'attempted', 'valid', 'failed', 'unparsed')) + ' |')
    held = report.get('held_out_evaluation') or {}
    lines += ['', f"Common usable pairs: {len(report['paired_items'])}",
        f"Calibration discrimination AUC: {report['claims']['discrimination']['calibrator_holdout_auc']}",
        f"Held-out AUC (single answer, feature-based p_memorized; {held.get('n_memorized')} memorized / {held.get('n_control')} control scored): "
        f"{held.get('auc')}, {held.get('n_resamples')}-resample stratified bootstrap "
        f"{held.get('confidence')} interval [{held.get('ci_low')}, {held.get('ci_high')}]"
        + (f" ({held['reason']})" if held.get('reason') else ''),
        'Calibration quality: not measured. Mitigation: not performed.',
        'NIM unavailable: historical HTTP 504; cause unknown. Opus/frontier work is separate.', '',
        'Model, runtime identity, draw rows, resource observations and physical attempts are in the companion JSON.', '',
        *['- ' + text for text in report['limitations']], '']
    return '\n'.join(lines)


def write_report(journal, report):
    """Exclusive artifacts preserve earlier receipts and never overwrite journals."""
    json_path, markdown_path = report_paths(journal)
    with json_path.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    with markdown_path.open('x', encoding='utf-8') as stream:
        stream.write(render_markdown(report))

