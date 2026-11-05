"""Serial local execution; raw responses stay in memory, never in the journal.

Rows have explicit item/arm/draw keys. The scorer owns all feature mathematics;
re-scoring captured responses performs no additional generation. Closing sidecar
checks apply to the entire batch and never provide an atomic identity guarantee.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

from recall_guard.core.ensemble import EnsembleSpec
from recall_guard.core.local_evidence import (
    LocalBatchCheck,
    LocalModelProvenance,
    merge_local_observations,
)
from recall_guard.core.request_control import BudgetExhaustedError
from recall_guard.harness.local_evaluation import LocalAttemptBudget
from recall_guard.harness.scorer import GuardedScore, MemoryGuardedScorer


@dataclass(frozen=True)
class HostObservation:
    provenance: LocalModelProvenance
    loaded_models: tuple[str, ...]

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
        except Exception:
            row.status, row.reason = 'failed', 'probe_unavailable'
            raise
        finally:
            row.physical_call_ids = tuple(range(start + 1, self.budget.used + 1))

    def generate(self, prompt, **parameters):
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
                row.status = 'scored' if row.score.parse_ok else 'unparsed'
                row.reason = row.score.fail_reason
            return row.response
        except Exception as error:
            if row.response is None:
                row.response = getattr(error, 'response_evidence', None)
            row.status = 'failed'
            row.reason = 'invalid_evidence_or_identity' if isinstance(error, ValueError) else 'generation_unavailable'
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
        before.validate(plan)
        capture = _CaptureClient(client, budget, observer, result, before)
        client.begin_local_batch(before.provenance)
        opened = True
        scorer = MemoryGuardedScorer.calibrate_client(client=capture,
            is_memorized=[i.prompt for i in plan.items if i.split == 'is_memorized'],
            oos_control=[i.prompt for i in plan.items if i.split == 'oos_control'],
            min_valid=plan.min_valid, min_auc=plan.min_auc, seed=plan.seed, max_workers=1)
        capture._check_running()
        capture.scorer = scorer
        result.holdout_auc = scorer.holdout_auc
        spec = EnsembleSpec(draws=plan.ensemble_draws, min_parsed=plan.ensemble_min_parsed,
            max_workers=1, retain_draws=True, max_tokens=plan.config.parameters['max_tokens'],
            temperature=plan.config.parameters.get('temperature'))
        for item in plan.items:
            if item.split != 'held_out':
                continue
            rows = [r for r in result.rows if r.item_id == item.item_id and r.arm in {'single', 'ensemble'}]
            capture.active_rows, capture.offset = rows[:1], 0
            rows[0].score = scorer.score(item.prompt)
            capture._check_running()
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
        if capture is not None and capture.current_row is not None and capture.current_row.status == 'received':
            capture.current_row.status, capture.current_row.reason = 'failed', result.reason
    finally:
        if opened:
            try:
                after = observer()
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
        budget.finish_execution(result.status)
    return result
