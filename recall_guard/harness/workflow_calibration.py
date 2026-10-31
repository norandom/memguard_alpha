"""Prepare label-free calibration inputs through the inference rewrite policy.

Corpus roles route local classifier labels only. Target prompts contain the
application instructions/as-of/evidence; auxiliary prompts contain no IDs,
corpus roles or classifier labels. Accepted OOS prompts are materialized once
and reused for the existing baseline and training calls.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

from recall_guard.core.completion import FeatureChoices, validate_features
from recall_guard.core.request_control import BudgetExhaustedError, RequestError
from recall_guard.harness.intervention_types import RecordStatus, RewriteRecord
from recall_guard.harness.request_failures import FATAL_FAILURES, request_failure, rewrite_failure_kind
from recall_guard.harness.rewriting import RewriteInput, render_target_prompt
from recall_guard.harness.scorer import MemoryGuardedScorer
from recall_guard.harness.scorer_client import CalibrationIdentity
from recall_guard.harness.workflow_types import WorkflowPreflight


@dataclass(frozen=True)
class CalibrationItem:
    """One source row and its retained preparation outcome, in input order.

    Pending means unattempted after a previous fatal failure. A rejected row
    has no target prompt and is excluded explicitly from both fitting stages.
    """

    corpus: str
    item: RewriteInput
    status: RecordStatus = RecordStatus.PENDING
    prompt: str | None = None
    rewrite: RewriteRecord | None = None
    error: str | None = None
    failure_kind: str | None = None


@dataclass(frozen=True)
class WorkflowCalibration:
    """Fit outcome, including every rejected/unattempted preparation row.

    Completed refers to fitting on the explicit accepted subset; inspect
    ``preparation_complete`` and counts before describing the whole corpus.
    Valid counts are from owning MIA producers after feature skips, before the
    train/holdout split. They remain unknown on a failed fit, rather than being
    inferred from accepted input counts. No result installs a default scorer.
    """

    identity: CalibrationIdentity
    status: RecordStatus
    items: tuple[CalibrationItem, ...]
    counts: Mapping[str, Mapping[str, int | None]]
    scorer: MemoryGuardedScorer | None = None
    holdout_auc: float | None = None
    is_weak: bool | None = None
    preflight: WorkflowPreflight | None = None
    error: str | None = None
    failure_kind: str | None = None
    diagnostics: Mapping | None = None
    provenance: Mapping = field(default_factory=dict)
    holdout_labels: tuple[int, ...] | None = None
    holdout_probabilities: tuple[float, ...] | None = None

    @property
    def preparation_complete(self) -> bool:
        return all(item.status == RecordStatus.COMPLETED for item in self.items)


def _validate_inputs(is_memorized, oos_control, min_auc, min_valid, seed, max_workers):
    corpora = {"is_memorized": is_memorized, "oos_control": oos_control}
    for name, rows in corpora.items():
        if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
            raise TypeError(f"{name} must be a sequence of RewriteInput")
        if any(not isinstance(item, RewriteInput) for item in rows):
            raise TypeError(f"{name} must contain only label-free RewriteInput values")
    if type(min_valid) is not int or min_valid < 1 or type(max_workers) is not int or max_workers < 1:
        raise ValueError("min_valid and max_workers must be positive integers")
    if type(seed) is not int:
        raise TypeError("seed must be an integer")
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError("seed must be in [0, 2**32 - 1] for the existing sklearn calibration")
    if isinstance(min_auc, bool) or not isinstance(min_auc, (float, int)) or not 0 <= min_auc <= 1:
        raise ValueError("min_auc must be in [0, 1]")
    _validate_sizes(len(is_memorized), len(oos_control), min_valid)
    items = tuple(CalibrationItem(name, item) for name, rows in corpora.items() for item in rows)
    if len({record.item.item_id for record in items}) != len(items):
        raise ValueError("Calibration item IDs must be unique across the two corpora")
    return items


def _validate_sizes(n_is, n_oos, min_valid):
    if n_is < 2 or n_oos < max(2, min_valid) or math.ceil(0.25 * (n_is + n_oos)) < 2:
        raise ValueError("Calibration corpora cannot meet baseline/class/stratified-holdout minimum counts")


def _counts(items, scorer):
    valid = scorer.calibration_valid_counts if scorer else {}
    result = {}
    for corpus in ("is_memorized", "oos_control"):
        rows = [record for record in items if record.corpus == corpus]
        result[corpus] = {"input": len(rows), "attempted": sum(r.status != RecordStatus.PENDING for r in rows),
                          "accepted": sum(r.status == RecordStatus.COMPLETED for r in rows),
                          "training_valid": valid.get(corpus)}
    result["oos_control"]["baseline_valid"] = valid.get("baseline")
    return result


def _record_failure(record):
    metadata = record.fidelity_checks.get("verification", record.fidelity_checks.get("generation", {}))
    return rewrite_failure_kind(metadata.get("failure_kind")), metadata.get("diagnostics")


def calibrate_workflow(workflow, *, is_memorized, oos_control, rewrite, min_auc, min_valid, seed, max_workers):
    items = list(_validate_inputs(is_memorized, oos_control, min_auc, min_valid, seed, max_workers))
    rewrite, roles, overrides, _ = workflow._selection(rewrite, None, scorer=None,
                                                      validate_draws=False, validate_scoring=False)
    target_config = workflow.target.config
    validate_features(target_config, FeatureChoices(mia_scoring=True))
    if not callable(getattr(workflow.target, "probe", None)):
        raise ValueError("MIA calibration requires a target with an actual generation probe; use a CallbackClient adapter")
    intervention_id = workflow.rewrite_policy.intervention_id if rewrite else "original"
    identity = CalibrationIdentity.from_config(target_config, intervention_id)
    provenance = {"target": target_config.to_dict(), "rewrite_scope": "shared_per_item" if rewrite else None,
                  "min_valid": min_valid, "min_auc": min_auc, "seed": seed, "max_workers": max_workers}
    # Early eligibility/capability gate covers every role before preparation.
    # calibrate_client also performs its own fit-specific actual probe. Both
    # requests remain in the shared physical budget; none is hidden or skipped.
    preflight = workflow._preflight(roles, overrides, mia_scoring=True)

    def outcome(scorer=None, *, error=None, kind=None, diagnostics=None):
        holdout = {"holdout_labels": scorer.holdout_labels if scorer else None,
                   "holdout_probabilities": scorer.holdout_probabilities if scorer else None}
        return WorkflowCalibration(
            identity=identity, status=RecordStatus.COMPLETED if scorer else RecordStatus.FAILED,
            items=tuple(items), counts=_counts(items, scorer), scorer=scorer,
            holdout_auc=scorer.holdout_auc if scorer else None, is_weak=scorer.is_weak if scorer else None,
            preflight=preflight, error=error, failure_kind=kind, diagnostics=diagnostics,
            provenance=dict(provenance) | workflow._request_provenance(roles) | holdout, **holdout,
        )

    if not preflight.ready_for_batch:
        return outcome(error=preflight.error, kind=preflight.failure_kind, diagnostics=preflight.diagnostics)
    for index, prepared in enumerate(items):
        record = None
        try:
            if workflow.target.config != target_config:
                raise ValueError("Target configuration changed during calibration preparation")
            record = workflow.rewrite_policy.rewrite(prepared.item) if rewrite else None
            if record is not None and record.status != RecordStatus.COMPLETED:
                kind, diagnostics = _record_failure(record)
                items[index] = replace(prepared, status=RecordStatus.FAILED, rewrite=record,
                                       error=record.error, failure_kind=kind)
                if kind in FATAL_FAILURES:
                    return outcome(error=record.error, kind=kind, diagnostics=diagnostics)
                continue
            if rewrite and workflow.rewrite_policy.intervention_id != intervention_id:
                raise ValueError("Rewrite intervention changed during calibration preparation")
            evidence = record.rewritten_evidence if record else prepared.item.evidence
            items[index] = replace(prepared, status=RecordStatus.COMPLETED, rewrite=record,
                                   prompt=render_target_prompt(prepared.item, evidence=evidence))
        except Exception as error:
            kind, message, diagnostics = request_failure(error)
            if isinstance(error, ValueError):
                kind, message = "configuration_error", "Calibration configuration changed during preparation."
            items[index] = replace(prepared, status=RecordStatus.FAILED, rewrite=record,
                                   error=message, failure_kind=kind)
            return outcome(error=message, kind=kind, diagnostics=diagnostics)
    prompts = {corpus: [row.prompt for row in items if row.corpus == corpus and row.prompt is not None]
               for corpus in ("is_memorized", "oos_control")}
    try:
        _validate_sizes(len(prompts["is_memorized"]), len(prompts["oos_control"]), min_valid)
        if workflow.target.config != target_config:
            raise ValueError("Target configuration changed during calibration preparation")
        if rewrite:
            workflow.rewrite_policy.validate_configuration()
        scorer = MemoryGuardedScorer.calibrate_client(client=workflow.target, intervention_id=intervention_id,
                                                      min_auc=min_auc, min_valid=min_valid, seed=seed,
                                                      max_workers=max_workers, **prompts)
    except Exception as error:
        # The scorer preserves its historical ConfigurationError wrapper at
        # probe/auth boundaries. Its sanitized transport cause remains the
        # authoritative category for calibration orchestration.
        cause = error.__cause__
        classified = cause if isinstance(cause, (RequestError, BudgetExhaustedError)) else error
        kind, message, diagnostics = request_failure(classified)
        if isinstance(error, ValueError):
            kind, message = "calibration_error", "Calibration failed minimum-count, feature or configuration checks."
        return outcome(error=message, kind=kind, diagnostics=diagnostics)
    return outcome(scorer)


__all__ = ["CalibrationItem", "WorkflowCalibration"]
