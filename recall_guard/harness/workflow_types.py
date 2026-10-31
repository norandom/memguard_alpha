"""Inference outcomes separate raw attempt order from canonical ensemble order."""

from collections.abc import Hashable, Mapping
from dataclasses import dataclass, field

from recall_guard.core.completion import CapabilityReport, ModelResponse
from recall_guard.core.ensemble import EnsembleResult
from recall_guard.harness.intervention_types import EvidenceArm, RecordStatus, RewriteRecord


@dataclass(frozen=True)
class WorkflowDraw:
    """One zero-based attempt; rejected projections keep the original response.

    ``intervention_id`` identifies the pipeline shared across items. Together,
    ``item_id`` and ``evidence_hash`` associate the actual shared rewrite used
    for this item's draws. Neither identity is a canonical order.
    """

    item_id: str
    arm: EvidenceArm
    draw_index: int
    status: RecordStatus = RecordStatus.PENDING
    prompt: str | None = None
    response: ModelResponse | None = None
    error: str | None = None
    intervention_id: str | None = None
    evidence_hash: str | None = None
    answer: Hashable | None = None
    components: Mapping[str, float] | None = None
    failure_kind: str | None = None

    @property
    def identity(self) -> tuple[str, EvidenceArm, int]:
        return self.item_id, self.arm, self.draw_index


@dataclass(frozen=True)
class WorkflowPreflight:
    """Actual required-feature probes for the active roles and exact settings."""

    reports: Mapping[str, CapabilityReport] = field(default_factory=dict)
    responses: Mapping[str, ModelResponse] = field(default_factory=dict)
    affected_role: str | None = None
    error: str | None = None
    failure_kind: str | None = None
    diagnostics: Mapping | None = None

    @property
    def ready_for_batch(self) -> bool:
        return self.error is None and bool(self.reports) and all(r.ready_for_batch for r in self.reports.values())


@dataclass(frozen=True)
class WorkflowResult:
    """Label-free inference, retaining stages even when generation/reduction fails.

    ``draws`` is in logical admission order, including failed draws. Retained
    ``ensemble.draws`` uses the existing reducer's canonical usable order.
    Physical calls include probes and retries and belong to the shared budget.
    Answer statistics do not estimate memorization; scoring is a separate path.
    """

    item_id: str
    original_evidence: str
    accepted_evidence: str | None
    status: RecordStatus
    stage: str
    method: str
    n_requested: int
    draws: tuple[WorkflowDraw, ...] = ()
    rewrite: RewriteRecord | None = None
    rewrite_scope: str | None = None
    response: ModelResponse | None = None
    answer: Hashable | None = None
    ensemble: EnsembleResult[ModelResponse] | None = None
    preflight: WorkflowPreflight | None = None
    error: str | None = None
    failure_kind: str | None = None
    guarded_score: None = None
    mia_features: None = None
    p_memorized: None = None
    unavailable_reasons: Mapping[str, str] = field(default_factory=dict)
    provenance: Mapping = field(default_factory=dict)

    @property
    def n_attempted(self) -> int:
        return len(self.draws)

    @property
    def n_usable(self) -> int:
        return sum(draw.status == RecordStatus.COMPLETED for draw in self.draws)

    @property
    def n_unattempted(self) -> int:
        return self.n_requested - self.n_attempted
