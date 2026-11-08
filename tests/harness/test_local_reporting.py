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
