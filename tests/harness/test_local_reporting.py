"""Report contracts using the real execution path and offline provider fixtures."""
import json
from dataclasses import replace

import pytest

from recall_guard.harness.local_execution import execute_plan
from recall_guard.harness.local_reporting import build_report
from tests.harness.test_local_execution import execution as _execution_fixture

execution = _execution_fixture


def test_report_all_stages_and_common_usable_pairs(execution):
    budget, state, observer = execution
    result = execute_plan(budget, observer)
    report = build_report(budget, result)
    assert set(report['stages']) == {'probe', 'baseline', 'calibration', 'single', 'ensemble'}
    assert report['stages']['ensemble'] == dict(planned=3, attempted=3, completed=3,
        parsed=2, unparsed=1, failed=0, unattempted=0, usable=2)
    assert report['stages']['calibration']['planned'] == 16
    assert report['stages']['baseline']['planned'] == 8
    assert len(report['paired_items']) == 1
    assert report['paired_items'][0]['ensemble_p_memorized_point'] == result.ensembles['held'].p_memorized_point
    assert report['claims']['discrimination']['calibrator_holdout_auc'] == result.holdout_auc
    assert report['claims']['mitigation'] == 'not_performed'
    assert report['resources'][-1]['load_time_s'] is None
    assert report['resources'][-1]['dedicated_gpu_memory_bytes'] is None
    assert report['observed_execution_mode'] == 'unknown'
    assert report['historical_nim'] == {'status': 'NIM unavailable', 'http_status': 504, 'cause': None}
    assert state.calls == 29


@pytest.mark.parametrize('failure', ['drift', 'timeout'])
def test_partial_and_mismatched_denominators(execution, failure):
    budget, state, observer = execution
    state.drift = failure == 'drift'
    state.fail_at = 3 if failure == 'timeout' else None
    report = build_report(budget, execute_plan(budget, observer))
    assert report['paired_items'] == []
    assert report['claims']['discrimination']['calibrator_holdout_auc'] is None
    assert report['claims']['endpoint_feasibility'] == 'probe_passed'
    assert report['stages']['single']['usable'] == 0
    if failure == 'drift':
        assert report['calibration_fit_completed']
        assert report['stages']['single']['parsed'] == 1
        assert report['stages']['ensemble']['unparsed'] == 1
    else:
        assert report['stages']['baseline']['attempted'] == 2
        assert report['stages']['baseline']['failed'] == 1
        assert report['stages']['baseline']['unattempted'] == 6
        assert report['stages']['single']['unattempted'] == 1


def test_safe_export_excludes_private_material_and_errors(execution):
    budget, _, observer = execution
    result = execute_plan(budget, observer)
    result.reason = 'PRIVATE error body'
    result.rows[0].reason = 'PRIVATE response'
    result.rows.reverse()  # Pair by item identity, never position.
    report = build_report(budget, result)
    serialized = json.dumps(report, allow_nan=False)
    assert 'PRIVATE' not in serialized
    assert 'Direction:' not in serialized
    assert 'Confidence:' not in serialized
    assert len(report['paired_items']) == 1
    assert len({(r['item_sha256'], r['arm'], r['draw_index']) for r in report['rows']}) == 29


def test_credentials_redacted_from_model_and_observed_identity(execution):
    from recall_guard.core.local_evidence import LocalModelProvenance
    from recall_guard.harness.local_execution import HostObservation, HostResources
    budget, _, observer = execution
    result = execute_plan(budget, observer)
    budget.plan = replace(budget.plan, authentication='environment', config=replace(
        budget.plan.config, model='gemma-SECRET', credential_ref='env:LOCAL_KEY'))
    budget.client(environment={'LOCAL_KEY': 'SECRET'})  # registers without making a call
    result.host_observations['closing'] = HostObservation(
        LocalModelProvenance(runtime='ollama', runtime_version='SECRET', backend='CPU',
                             template='PRIVATE system prompt'),
        ('gemma-SECRET',), HostResources(model_allocation_bytes=123, accelerator_allocation_bytes=0))
    report = build_report(budget, result)
    assert 'SECRET' not in json.dumps(report)
    assert 'PRIVATE system prompt' not in json.dumps(report)
    assert report['selected_model'] == 'gemma-[REDACTED]'
    assert report['observations'][-1]['provenance']['runtime'] == 'ollama'
    assert report['resources'][-1]['accelerator_allocation_bytes'] == 0
    assert report['resources'][-1]['dedicated_gpu_memory_bytes'] is None


def test_pairing_requires_common_item_not_just_usable_arm(execution):
    from recall_guard.harness.local_evaluation import MeasurementItem
    from recall_guard.harness.local_execution import MeasurementRow
    budget, _, observer = execution
    result = execute_plan(budget, observer)
    budget.plan = replace(budget.plan, items=budget.plan.items + (
        MeasurementItem('other-held', 'held_out', 'private other prompt', 20),),
        limits=replace(budget.plan.limits, max_requests=33, max_total_tokens=4356))
    original = next(r for r in result.rows if r.arm == 'single')
    result.rows.append(replace(original, item_id='other-held'))
    result.rows.extend(MeasurementRow('other-held', 'ensemble', i) for i in range(3))
    result.ensembles['other-held'] = result.ensembles.pop('held')
    original.score = None
    report = build_report(budget, result)
    # The only retained ensemble belongs to other-held, and joins its single.
    assert len(report['paired_items']) == 1
    pair = report['paired_items'][0]
    assert pair['item_sha256'] == next(r['item_sha256'] for r in report['rows']
        if r['arm'] == 'single' and r['usable'])


def test_cli_persists_partial_report_and_refuses_existing(execution, tmp_path, monkeypatch):
    from scripts import measure_local_models as cli
    budget, state, observer = execution
    monkeypatch.setattr(cli, 'load_plan', lambda *args: budget.plan)
    monkeypatch.setattr(cli, 'ollama_observer', lambda plan: observer)
    state.fail_at = 3
    path = tmp_path / 'partial.json'
    args = ['--plan', 'p', '--manifest', 'm', '--corpus', 'c', '--execute',
            '--journal', str(path), '--observer', 'ollama']
    assert cli.main(args) == 2
    report_path = tmp_path / 'partial.json.report.json'
    report = json.loads(report_path.read_text())
    assert report['stages']['baseline']['failed'] == 1
    assert (tmp_path / 'partial.json.report.md').exists()
    previous = report_path.read_bytes()
    assert cli.main(args) == 2
    assert report_path.read_bytes() == previous
    assert state.calls == 3


def test_stopping_provider_failure_body_is_exported_with_prompts_redacted(execution):
    budget, _, observer = execution
    result = execute_plan(budget, observer)
    prompt = budget.plan.items[0].prompt
    result.failure = {'status_code': 500, 'code': None, 'detail': f'load failed while serving {prompt}'}
    report = build_report(budget, result)
    assert report['failure']['status_code'] == 500
    assert report['failure']['detail'] == 'load failed while serving [PRIVATE]'


def test_held_out_discrimination_perfect_and_seeded():
    from recall_guard.harness.local_reporting import held_out_discrimination
    perfect = held_out_discrimination([0.9, 0.8, 0.7, 0.2, 0.1], [1, 1, 1, 0, 0], seed=0)
    assert perfect['auc'] == 1.0 and perfect['ci_low'] == perfect['ci_high'] == 1.0
    assert (perfect['n_memorized'], perfect['n_control'], perfect['n_resamples']) == (3, 2, 1000)
    scores = [0.9, 0.4, 0.6, 0.3, 0.5, 0.2, 0.7, 0.1]
    labels = [1, 1, 1, 1, 0, 0, 0, 0]
    first = held_out_discrimination(scores, labels, seed=0)
    assert first == held_out_discrimination(scores, labels, seed=0)
    assert first['auc'] == pytest.approx(11 / 16)
    assert 0 <= first['ci_low'] < first['auc'] < first['ci_high'] <= 1
    assert held_out_discrimination(scores, labels, seed=1) != first


def test_held_out_discrimination_needs_both_classes():
    from recall_guard.harness.local_reporting import held_out_discrimination
    result = held_out_discrimination([0.5, 0.6], [1, 1])
    assert result['auc'] is None and result['ci_low'] is None and result['n_control'] == 0


def test_held_out_calibration_perfect_predictor():
    from recall_guard.harness.local_reporting import held_out_calibration
    result = held_out_calibration([1.0, 1.0, 1.0, 0.0, 0.0], [1, 1, 1, 0, 0], seed=0)
    assert result['brier'] == 0.0 and result['brier_ci_low'] == result['brier_ci_high'] == 0.0
    assert result['ece'] == 0.0 and result['brier_reference'] == pytest.approx(0.6 * 0.4)
    assert (result['mean_predicted'], result['base_rate']) == (pytest.approx(0.6), pytest.approx(0.6))
    assert [(b['low'], b['high'], b['count']) for b in result['reliability_bins']] == [(0.0, 0.1, 2), (0.9, 1.0, 3)]


def test_held_out_calibration_constant_base_rate_equals_reference():
    from recall_guard.harness.local_reporting import held_out_calibration
    labels = [1, 1, 0, 0, 0, 0, 0, 0]
    result = held_out_calibration([0.25] * 8, labels, seed=0)
    assert result['brier'] == pytest.approx(result['brier_reference']) == pytest.approx(0.25 * 0.75)
    assert result['ece'] == pytest.approx(0.0) and result['mean_predicted'] == pytest.approx(result['base_rate'])
    assert result['reliability_bins'] == [{'low': 0.2, 'high': 0.3, 'count': 8, 'mean_predicted': 0.25,
                                           'observed_memorized_fraction': 0.25}]


def test_held_out_calibration_miscalibrated_bins_and_seed():
    import json

    from recall_guard.harness.local_reporting import held_out_calibration
    # Overconfident: every item predicted 0.95 or 0.85 regardless of label.
    scores = [0.95, 0.95, 0.85, 0.95, 0.85, 0.85]
    labels = [1, 0, 1, 0, 0, 1]
    result = held_out_calibration(scores, labels, seed=3)
    bins = result['reliability_bins']
    assert [(b['low'], b['count']) for b in bins] == [(0.8, 3), (0.9, 3)]
    assert bins[0]['mean_predicted'] == pytest.approx(0.85) and bins[0]['observed_memorized_fraction'] == pytest.approx(2 / 3)
    assert bins[1]['mean_predicted'] == pytest.approx(0.95) and bins[1]['observed_memorized_fraction'] == pytest.approx(1 / 3)
    assert result['ece'] == pytest.approx((3 * (0.85 - 2 / 3) + 3 * (0.95 - 1 / 3)) / 6) and result['ece'] > 0
    assert result['brier'] > result['brier_reference']
    assert result['brier_ci_low'] <= result['brier'] <= result['brier_ci_high']
    assert result == held_out_calibration(scores, labels, seed=3)
    json.dumps(result, allow_nan=False)
    assert all(type(result[k]) is float for k in ('brier', 'brier_ci_low', 'brier_ci_high', 'brier_reference',
                                                  'ece', 'mean_predicted', 'base_rate'))


def test_held_out_calibration_needs_two_per_class():
    from recall_guard.harness.local_reporting import held_out_calibration
    result = held_out_calibration([0.9, 0.8, 0.1], [1, 1, 0])
    assert result['brier'] is None and result['ece'] is None and result['brier_reference'] is None
    assert result['reliability_bins'] == [] and result['base_rate'] is None


def test_report_held_out_auc_from_labels_and_split_counts(execution):
    from recall_guard.harness.local_reporting import _held_out_evaluation, _key
    budget, _, observer = execution
    result = execute_plan(budget, observer)
    report = build_report(budget, result)
    assert report['held_out_evaluation']['reason'] == 'no_labeled_held_out_items'
    assert report['claims']['discrimination']['held_out_evaluation_auc'] is None
    assert report['claims']['calibration_quality'] == 'not_measured'
    assert report['claims']['calibration_quality_reason'] == 'no_labeled_held_out_items'
    assert report['splits']['is_memorized'] == dict(planned=8, attempted=8, valid=8, failed=0, unparsed=0, item_evidence_codes={})
    assert report['splits']['oos_control']['planned'] == 16
    assert report['splits']['held_out'] == dict(planned=1, attempted=1, valid=1, failed=0, unparsed=0, item_evidence_codes={})
    # Synthetic usable singles: labels come from the plan, scores from usable rows only.
    plan = replace(budget.plan, items=tuple(replace(i, label=1 if i.item_id.endswith(('0', '1')) else 0)
                                            if i.split == 'held_out' else i for i in budget.plan.items))
    items = [replace(budget.plan.items[0], item_id=f'h{i}', split='held_out', label=i % 2) for i in range(6)]
    plan = replace(plan, items=plan.items + tuple(items),
                   limits=replace(plan.limits, max_requests=99, max_total_tokens=99 * 132))
    rows = [{'item_sha256': _key(f'h{i}'), 'arm': 'single', 'feature_scored': i != 5,
             'feature_p_memorized': 0.1 + 0.1 * i if i != 5 else None}
            for i in range(6)]
    evaluation = _held_out_evaluation(plan, rows, True)
    assert (evaluation['n_labeled'], evaluation['n_usable'], evaluation['reason']) == (7, 5, None)
    assert (evaluation['n_memorized'], evaluation['n_control']) == (2, 3)
    assert evaluation["auc"] == pytest.approx(0.5)  # 3 of 6 memorized/control pairs ordered
    # AUC needs one per class; calibration needs two, so 2 memorized / 3 control is enough.
    assert evaluation['calibration_reason'] is None and evaluation['brier'] is not None
    assert evaluation['base_rate'] == pytest.approx(0.4)
    assert _held_out_evaluation(plan, [], False)['reason'] == 'calibrated_claims_not_allowed'
    assert 'Held-out AUC' in __import__('recall_guard.harness.local_reporting', fromlist=['x']).render_markdown(report)


def test_unparsed_held_out_answer_keeps_feature_score_but_public_score_fails_parse(execution):
    from recall_guard.harness.scorer import FAIL_PARSE
    budget, state, observer = execution
    state.unparsed_at = 26  # probe 1 + baseline 8 + calibration 16, then the held-out single answer
    result = execute_plan(budget, observer)
    single = next(r for r in result.rows if r.arm == 'single')
    assert result.calibrated_claims_allowed and single.status == 'unparsed'
    assert single.score.parse_ok is False and single.score.fail_reason == FAIL_PARSE
    assert single.score.p_memorized is None
    assert 0 <= single.feature_p_memorized <= 1
    record = next(r for r in build_report(budget, result)['rows'] if r['arm'] == 'single')
    assert (record['parsed'], record['usable'], record['feature_scored']) == (False, False, True)
    assert record['feature_p_memorized'] == single.feature_p_memorized


def test_held_out_auc_uses_feature_scores_of_unparsed_answers(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from recall_guard.core.completion import ModelConfig
    from recall_guard.core.local_evidence import LocalModelIdentity, LocalModelProvenance
    from recall_guard.harness.local_evaluation import (
        LocalAttemptBudget,
        LocalMeasurementPlan,
        MeasurementItem,
        MeasurementLimits,
    )
    from recall_guard.harness.local_execution import HostObservation
    config = ModelConfig('openai_chat', 'http://localhost:11434/v1/chat/completions', 'test',
        parameters={'max_tokens': 32, 'temperature': 0, 'logprobs': True, 'top_logprobs': 3},
        timeout_s=2, max_retries=0, local_identity=LocalModelIdentity())
    items = tuple(MeasurementItem(f'{split}-{i}', split, f'{split}-{i}', 20, int(split == 'is_memorized'))
                  for split in ('is_memorized', 'oos_control') for i in range(8)) + tuple(
        MeasurementItem(f'held-{kind}-{i}', 'held_out', f'held-{kind}-{i}', 20, int(kind == 'mem'))
        for kind in ('mem', 'ctl') for i in range(3))
    requests = 1 + 8 + 16 + 6
    plan = LocalMeasurementPlan(config, 'none', 'CPU', 'a' * 64, 'b' * 64, '{prompt}', ('mia', 'mcs'), 0,
                                MeasurementLimits(requests, 2048, 100, 32, requests * 132, 120), items, min_valid=4)
    budget = LocalAttemptBudget(plan, tmp_path / 'journal.json')

    def post(client, payload):
        prompt = payload['messages'][-1]['content']
        base = -0.2 if 'mem' in prompt or prompt.startswith('is_') else -5
        base += (sum(map(ord, prompt)) % 10) / 1000
        # Held-out answers never match the direction/confidence parser.
        content = 'no answer' if prompt.startswith('held') else 'Direction: 1\nConfidence: 0.8'
        entries = [{'token': char, 'logprob': base + i / 1000,
                    'top_logprobs': [{'token': 'a', 'logprob': base}, {'token': 'b', 'logprob': base - 1}]}
                   for i, char in enumerate(content)]
        body = {'choices': [{'message': {'content': content}, 'logprobs': {'content': entries},
                             'finish_reason': 'stop'}]}
        return SimpleNamespace(status_code=200, headers={}, json=lambda: body)
    monkeypatch.setattr('recall_guard.core.providers.OpenAICompatibleClient._post', post)

    def observer():
        return HostObservation(LocalModelProvenance(runtime='ollama', runtime_version='1', artifact_digest='d',
                                                    context_length=2048, backend=None), ('test',))
    result = execute_plan(budget, observer)
    report = build_report(budget, result)
    assert report['stages']['single']['parsed'] == 0 and report['stages']['single']['usable'] == 0
    held = report['held_out_evaluation']
    assert (held['n_labeled'], held['n_usable'], held['n_memorized'], held['n_control']) == (6, 6, 3, 3)
    assert held['reason'] is None and held['auc'] is not None and held['ci_low'] <= held['auc'] <= held['ci_high']
    assert report['claims']['discrimination']['held_out_evaluation_auc'] == held['auc']
    claim = report['claims']['calibration_quality']
    assert claim['brier'] == held['brier'] and claim['ece'] == held['ece'] and claim['base_rate'] == 0.5
    assert report['claims']['calibration_quality_reason'] is None
    assert sum(b['count'] for b in held['reliability_bins']) == 6
    from recall_guard.harness.local_reporting import render_markdown
    markdown = render_markdown(report)
    assert 'Held-out Brier score' in markdown and 'Held-out ECE' in markdown
    assert report['splits']['held_out']['valid'] == 6 and report['splits']['held_out']['unparsed'] == 6
