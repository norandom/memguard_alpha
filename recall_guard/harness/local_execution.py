"""Serial local execution; raw responses stay in memory, never in the journal.

Rows have explicit item/arm/draw keys. The scorer owns all feature mathematics;
re-scoring captured responses performs no additional generation. Closing sidecar
checks apply to the entire batch and never provide an atomic identity guarantee.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace

from recall_guard.core.ensemble import EnsembleSpec
from recall_guard.core.local_evidence import (
    LocalBatchCheck,
    LocalModelProvenance,
    merge_local_observations,
)
from recall_guard.core.request_control import BudgetExhaustedError
from recall_guard.harness.local_evaluation import LocalAttemptBudget
from recall_guard.harness.request_failures import retained_error, retained_failure
from recall_guard.harness.scorer import GuardedScore, MemoryGuardedScorer


@dataclass(frozen=True)
class HostResources:
    """Observed quantities only; bytes/seconds, never inferred from allocation."""
    model_allocation_bytes: int | None = None
    accelerator_allocation_bytes: int | None = None
    dedicated_gpu_memory_bytes: int | None = None
    shared_gpu_memory_bytes: int | None = None
    system_ram_usage_bytes: int | None = None
    total_system_ram_bytes: int | None = None
    load_time_s: float | None = None
    generation_time_s: float | None = None

    def __post_init__(self):
        for name, value in asdict(self).items():
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                      or not math.isfinite(value) or value < 0
                                      or (name.endswith('_bytes') and not isinstance(value, int))):
                raise ValueError('invalid_resource_observation')


@dataclass(frozen=True)
class HostObservation:
    provenance: LocalModelProvenance
    loaded_models: tuple[str, ...]
    resources: HostResources = field(default_factory=HostResources)

    def validate(self, plan, *, after_probe=False):
        allowed = ((plan.config.model,),) if after_probe else ((), (plan.config.model,))
        if tuple(self.loaded_models) not in allowed:
            raise ValueError('one_selected_loaded_model_required')
        context = self.provenance.context_length
        if context != plan.limits.context_tokens and (after_probe or context is not None):
            raise ValueError('effective_context_not_established')


@dataclass
class MeasurementRow:
    item_id: str
    arm: str
    draw_index: int
    status: str = 'not_attempted'
    score: GuardedScore | None = None
    response: object | None = field(default=None, repr=False)
    reason: str | None = None
    physical_call_ids: tuple[int, ...] = ()
    #: Bounded sanitized status/code/body of a provider HTTP failure (R2.7), or
    #: ``{'error_class', 'code'}`` of an excluded per-item evidence defect.
    failure: dict | None = None
    #: Held-out single answers only: calibrated probability from MIA features,
    #: independent of answer parsing (parse success stays in ``score``/``status``).
    feature_p_memorized: float | None = None


@dataclass
class LocalExecutionResult:
    rows: list[MeasurementRow]
    status: str = 'unavailable'
    calibrated_claims_allowed: bool = False
    batch_check: LocalBatchCheck | None = None
    observed_execution_mode: str = 'unknown'
    npu_verified: bool = False
    ensembles: dict = field(default_factory=dict)
    holdout_auc: float | None = None
    reason: str | None = None
    host_observations: dict[str, HostObservation] = field(default_factory=dict)
    calibration_fit_completed: bool = False
    endpoint_probe_passed: bool = False
    #: Bounded sanitized status/code/body of the stopping provider HTTP failure (R2.7),
    #: or class and prompt/response-free message of the stopping ValueError.
    failure: dict | None = None


class _CaptureClient:
    """Delegate the original transport, keeping all draws including failures."""

    def __init__(self, client, budget, observer, result, opening_observation):
        self.client, self.budget, self.observer, self.result = client, budget, observer, result
        self.opening_provenance = opening_observation.provenance
        self.config = client.config
        self.active_rows = []
        self.offset = 0
        self.current_row = None
        self.scorer = None
        self.calibration_index = 0
        self.calibration_items = ([item for item in budget.plan.items if item.split == 'oos_control']
            + [item for item in budget.plan.items if item.split == 'is_memorized']
            + [item for item in budget.plan.items if item.split == 'oos_control'])

    def _check_running(self):
        if self.budget.stopped:
            raise BudgetExhaustedError('local_measurement_stopped')

    def probe_with_evidence(self):
        self._check_running()
        row = MeasurementRow('__probe__', 'probe', 0)
        self.current_row = row
        self.result.rows.insert(0, row)
        start = self.budget.used
        try:
            evidence = self.client.probe_with_evidence()
            row.response = evidence.response
            row.status = 'received' if evidence.response is not None else 'failed'
            self._check_running()
            observed = self.observer()
            self.result.host_observations['after_probe'] = observed
            observed.validate(self.budget.plan, after_probe=True)
            # The response already includes the opening sidecar. It cannot
            # restore a field that a fresh host observation has lost.
            if not _preserves_known_observations(
                    self.opening_provenance, observed.provenance):
                raise ValueError('probe_condition_mismatched')
            self.result.observed_execution_mode = observed.provenance.backend or 'unknown'
            # Unknown opening fields can be established by the first actual probe.
            # Known fields must still match, and native conflicts remain fatal.
            previous = None if evidence.response is None else evidence.response.local_identity
            merged, conflicts = merge_local_observations(
                observed.provenance, None if previous is None else previous.observed)
            if conflicts:
                raise ValueError('probe_condition_mismatched')
            check = self.client.end_local_batch(merged)
            self.client.begin_local_batch(merged)
            if set(check.revalidation_fields) - set(check.unknown_fields):
                raise ValueError('probe_condition_mismatched')
            if evidence.response is not None:
                identity = replace(evidence.response.local_identity, observed=merged)
                evidence = replace(evidence, response=replace(evidence.response, local_identity=identity))
                row.response = evidence.response
            return evidence
        except Exception as error:
            row.status, row.reason = 'failed', 'probe_unavailable'
            row.failure = retained_failure(error)
            raise
        finally:
            row.physical_call_ids = tuple(range(start + 1, self.budget.used + 1))

    def reject_item_evidence(self, code):
        """The scorer excluded the current received row for a per-item evidence defect.

        Identity checks already passed; the row is counted as failed and the run
        continues. ``code`` is a stable local-admission reason, never content.
        """
        row = self.current_row
        if row is not None and row.response is not None:
            row.status, row.reason = 'failed', 'invalid_item_evidence'
            row.failure = {'error_class': 'LocalEvidenceError', 'code': code}

    def generate(self, prompt, **parameters):
        # The scorer adapter reaches generation only after its complete probe
        # and identity/capability checks pass, even if fitting later fails.
        self.result.endpoint_probe_passed = True
        self._check_running()
        if self.active_rows:
            row = self.active_rows[self.offset]
            self.offset += 1
        else:
            index = self.calibration_index
            item = self.calibration_items[index]
            self.calibration_index += 1
            if item.prompt != prompt:
                raise ValueError('unexpected_calibration_order')
            arm = 'baseline' if index < sum(i.split == 'oos_control' for i in self.budget.plan.items) else 'calibration'
            row = MeasurementRow(item.item_id, arm, 0)
            self.result.rows.append(row)
        self.current_row = row
        start = self.budget.used
        try:
            row.response = self.client.generate_for_stage(prompt, stage=row.arm, **parameters)
            row.status = 'received'
            self._check_running()
            if self.scorer is not None:
                row.score = self.scorer.score_response(prompt, row.response, client=self)
                if row.arm == 'single':
                    row.feature_p_memorized = self.scorer.feature_p_memorized(prompt, row.response, client=self)
                row.status = 'scored' if row.score.parse_ok else 'unparsed'
                row.reason = row.score.fail_reason
            return row.response
        except Exception as error:
            if row.response is None:
                row.response = getattr(error, 'response_evidence', None)
            row.status = 'failed'
            row.reason = 'invalid_evidence_or_identity' if isinstance(error, ValueError) else 'generation_unavailable'
            row.failure = retained_failure(error)
            raise
        finally:
            row.physical_call_ids = tuple(range(start + 1, self.budget.used + 1))


def _preserves_known_observations(opening, fresh):
    def fields(provenance):
        values = asdict(provenance)
        semantics = values.pop('score_semantics')
        return values | {f'score_semantics.{key}': value for key, value in semantics.items()}
    before, after = fields(opening), fields(fresh)
    return all(value is None or after[name] == value for name, value in before.items())


def _held_out_rows(plan):
    return [MeasurementRow(item.item_id, arm, index) for item in plan.items if item.split == 'held_out'
            for arm, count in (('single', 1), ('ensemble', plan.ensemble_draws)) for index in range(count)]


def execute_plan(budget: LocalAttemptBudget, observer) -> LocalExecutionResult:
    """Run once with fresh caller-supplied metadata; never load or unload models.

    A failed/unknown/cancelled batch exposes no calibrated scores. Raw captured
    responses are private runtime evidence for downstream reporting (task 5.3).
    """
    plan = budget.plan
    budget.begin_execution()
    result = LocalExecutionResult(_held_out_rows(plan))
    capture = None
    opened = False
    try:
        client = budget.client()
        before = observer()
        result.host_observations['opening'] = before
        before.validate(plan)
        capture = _CaptureClient(client, budget, observer, result, before)
        client.begin_local_batch(before.provenance)
        opened = True
        scorer = MemoryGuardedScorer.calibrate_client(client=capture,
            is_memorized=[i.prompt for i in plan.items if i.split == 'is_memorized'],
            oos_control=[i.prompt for i in plan.items if i.split == 'oos_control'],
            min_valid=plan.min_valid, min_auc=plan.min_auc, seed=plan.seed, max_workers=1)
        result.calibration_fit_completed = True
        capture._check_running()
        capture.scorer = scorer
        result.holdout_auc = scorer.holdout_auc
        spec = None if not plan.ensemble_draws else EnsembleSpec(
            draws=plan.ensemble_draws, min_parsed=plan.ensemble_min_parsed,
            max_workers=1, retain_draws=True, max_tokens=plan.config.parameters['max_tokens'],
            temperature=plan.config.parameters.get('temperature'))
        for item in plan.items:
            if item.split != 'held_out':
                continue
            rows = [r for r in result.rows if r.item_id == item.item_id and r.arm in {'single', 'ensemble'}]
            capture.active_rows, capture.offset = rows[:1], 0
            rows[0].score = scorer.score(item.prompt)
            capture._check_running()
            if spec is None:
                continue
            capture.active_rows, capture.offset = rows[1:], 0
            result.ensembles[item.item_id] = scorer.score_ensemble(item.prompt, spec=spec)
            capture._check_running()
        result.status = 'completed'
    except Exception as error:
        result.status = 'stopped' if budget.stopped or isinstance(error, BudgetExhaustedError) else 'ineligible'
        safe_reasons = {'one_selected_loaded_model_required', 'effective_context_not_established',
                        'probe_condition_mismatched', 'unexpected_calibration_order'}
        result.reason = (str(error) if str(error) in safe_reasons else
                         'budget_or_cancellation' if isinstance(error, BudgetExhaustedError) else
                         'invalid_evidence_identity_or_fit' if isinstance(error, ValueError) else
                         'host_or_calibration_unavailable')
        private = [item.prompt for item in plan.items] + [
            row.response.content for row in result.rows if row.response is not None]
        result.failure = retained_failure(error) or retained_error(error, private_texts=private) or next(
            (row.failure for row in result.rows
            if row.failure is not None and 'status_code' in row.failure), None)
        if capture is not None and capture.current_row is not None and capture.current_row.status == 'received':
            capture.current_row.status, capture.current_row.reason = 'failed', result.reason
    finally:
        if opened:
            try:
                after = observer()
                result.host_observations['closing'] = after
                after.validate(plan, after_probe=True)
                closing = after.provenance
            except Exception:
                closing = None
            result.batch_check = client.end_local_batch(closing)
            if not result.batch_check.condition_matched:
                result.status = 'condition_mismatched'
        result.calibrated_claims_allowed = (result.status == 'completed' and result.batch_check is not None
                                            and result.batch_check.condition_matched)
        if not result.calibrated_claims_allowed:
            result.ensembles.clear()
            result.holdout_auc = None
            for row in result.rows:
                row.score = None
                row.feature_p_memorized = None
        budget.finish_execution(result.status)
    return result
