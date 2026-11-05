"""Safe local reports. Rows are keyed by item/arm/zero-based draw, never position.

All probabilities and AUC are dimensionless existing scorer outputs. Bytes and
seconds are explicit in resource column names. No prompts, output tokens, raw
responses or provider error strings are exported. Unknown quantities are null.
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
    return {'item_sha256': _key(row.item_id), 'arm': row.arm, 'draw_index': row.draw_index,
            'status': row.status if row.status in {'not_attempted', 'received', 'failed', 'scored', 'unparsed'} else 'unknown',
            'attempted': bool(row.physical_call_ids), 'completed': completed, 'parsed': bool(parsed),
            'usable': usable, 'physical_call_ids': list(row.physical_call_ids),
            'reason': row.reason if row.reason in _REASONS else None,
            'p_memorized': row.score.p_memorized if usable else None,
            'local_identity': _safe_value(identity, budget)}


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
        'stages': stages, 'rows': rows, 'ensembles': ensembles, 'paired_items': pairs,
        'attempts': attempts, 'physical_attempts': len(attempts),
        'claims': {'endpoint_feasibility': 'probe_passed' if result.endpoint_probe_passed else 'not_established',
                   'discrimination': {'calibrator_holdout_auc': result.holdout_auc if result.calibrated_claims_allowed else None,
                                      'held_out_evaluation_auc': None},
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
    lines += ['', f"Common usable pairs: {len(report['paired_items'])}",
        f"Calibration discrimination AUC: {report['claims']['discrimination']['calibrator_holdout_auc']}",
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

