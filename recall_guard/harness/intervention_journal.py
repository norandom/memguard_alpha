"""Durable local experiment journal and explicit, credential-free outcome snapshots.

This is runtime restart state, not the portable reporting format. One writer
holds an exclusive lock. Atomic UTF-8 replacements preserve write-ahead intent
through crashes; an ambiguous intent always requires explicit reconciliation.
"""

import json
import os
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from uuid import uuid4

from recall_guard.harness.intervention_types import EvidenceArm, ExperimentCheckpoint, RecordStatus


def json_value(value):
    """Detach only JSON values; never traverse arbitrary object/client state."""
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def preflight_snapshot(result):
    return {
        "reports": {role: {"capabilities": {name: evidence.to_dict() for name, evidence in report.capabilities.items()},
                           "features": asdict(report.features)} for role, report in result.reports.items()},
        "responses": {role: response.to_dict() for role, response in result.responses.items()},
        "affected_role": result.affected_role, "error": result.error,
        "failure_kind": result.failure_kind, "diagnostics": result.diagnostics,
    }


def arm_snapshot(result):
    """Whitelist outcomes; live WorkflowCalibration/scorer objects never enter I/O."""
    ensemble = None
    if result.ensemble is not None:
        value = result.ensemble
        ensemble = {name: getattr(value, name) for name in (
            "location", "location_snapped", "grid_adherence", "multimodal", "agreement", "agreement_ci",
            "draw_dependence", "max_tokens", "temperature", "sampled_at", "n_requested", "n_parsed",
            "fail_counts", "draws_sha256",
        )}
        ensemble["component_verdicts"] = [
            [name, None if verdict is None else asdict(verdict)] for name, verdict in value.component_verdicts]
        ensemble["consensus"] = value.consensus.to_dict()
    return json_value({
        "status": result.status.value, "method": result.method, "stage": result.stage,
        "error": result.error, "failure_kind": result.failure_kind, "answer": result.answer,
        "n_requested": result.n_requested, "n_attempted": result.n_attempted, "n_usable": result.n_usable,
        "n_unattempted": result.n_unattempted, "sampled_at": result.sampled_at, "ensemble": ensemble,
        "response": None if result.response is None else result.response.to_dict(),
        "scored_ensemble": None if result.scored_ensemble is None else asdict(result.scored_ensemble),
        "calibration_identity": None if result.calibration_identity is None else asdict(result.calibration_identity),
        "p_memorized": result.p_memorized, "unavailable_reasons": dict(result.unavailable_reasons),
        "provenance": dict(result.provenance),
    })


@dataclass(frozen=True)
class ExperimentArm:
    """One terminal workflow result; draws remain in the checkpoint's logical order."""

    item_id: str
    arm: EvidenceArm
    outcome: dict

    def __post_init__(self):
        object.__setattr__(self, "arm", EvidenceArm(self.arm))
        if self.outcome["status"] not in {RecordStatus.COMPLETED, RecordStatus.FAILED}:
            raise ValueError("Saved experiment arms must be terminal")


@dataclass(frozen=True)
class ExperimentCondition:
    item_id: str
    arm: EvidenceArm
    method: str
    draw_indices: tuple[int, ...]
    status: RecordStatus
    answer: object = None


@dataclass(frozen=True)
class ExperimentRun:
    """Rows are item/arm/draw; conditions explicitly share draw zero.

    Arm order is scheduling order. Conditions use fixed original-single,
    original-ensemble, rewritten-single, rewritten-ensemble order per item.
    Failed conditions can belong to a completed execution. A stop_reason
    indicates early termination and is retained on future invocations.
    """

    checkpoint: ExperimentCheckpoint
    arms: tuple[ExperimentArm, ...] = ()
    preflight: dict = field(default_factory=dict)
    stop_reason: str | None = None

    @property
    def conditions(self) -> tuple[ExperimentCondition, ...]:
        draws = {draw.identity: draw for draw in self.checkpoint.draws}
        arms = {(arm.item_id, arm.arm): arm for arm in self.arms}
        conditions = []
        for item in self.checkpoint.items:
            for arm in EvidenceArm:
                single = draws[(item.item_id, arm, 0)]
                outcome = arms.get((item.item_id, arm))
                conditions.append(ExperimentCondition(item.item_id, arm, arm.value + "_single", (0,),
                                                       single.status, single.metadata.get("answer")))
                conditions.append(ExperimentCondition(
                    item.item_id, arm, arm.value + "_ensemble", tuple(range(self.checkpoint.protocol.draws)),
                    RecordStatus.PENDING if outcome is None else RecordStatus(outcome.outcome["status"]),
                    None if outcome is None else outcome.outcome["answer"],
                ))
        return tuple(conditions)

    def to_dict(self):
        return {"checkpoint": self.checkpoint.to_dict(),
                "arms": [{"item_id": arm.item_id, "arm": arm.arm.value, "outcome": arm.outcome} for arm in self.arms],
                "preflight": self.preflight, "stop_reason": self.stop_reason}

    @classmethod
    def from_dict(cls, values):
        return cls(ExperimentCheckpoint.from_dict(values["checkpoint"]),
                   tuple(ExperimentArm(**arm) for arm in values["arms"]),
                   values["preflight"], values["stop_reason"])


class ExperimentJournal:
    def __init__(self, path):
        self.path = Path(path)

    @contextmanager
    def writer(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock = self.path.with_name(self.path.name + ".lock")
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            raise ValueError("Experiment writer lock exists; reconcile an abandoned writer before restart") from None
        try:
            os.close(descriptor)
            yield
        finally:
            lock.unlink()

    def load(self):
        values = json.loads(self.path.read_text(encoding="utf-8"))
        if type(values["journal_version"]) is not int or values["journal_version"] != 1:
            raise ValueError("Unsupported experiment journal version")
        if values["intent"] is not None:
            raise ValueError("Ambiguous provider operation intent prevents automatic restart")
        return ExperimentRun.from_dict(values)

    def save(self, result, intent=None):
        temporary = self.path.with_name(self.path.name + ".tmp-" + uuid4().hex)
        values = {"journal_version": 1, **result.to_dict(), "intent": intent}
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(values, handle, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def finish(self, result, intent):
        # Persist terminal outputs BEFORE clearing intent. A crash between
        # replacements is conservatively ambiguous even with terminal outputs.
        self.save(result, intent)
        self.save(result)
