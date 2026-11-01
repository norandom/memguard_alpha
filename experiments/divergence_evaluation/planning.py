"""Local configuration inspection only; execution admission is a separate phase."""

import json
from pathlib import Path

from .contracts import MCS_UNAVAILABLE, EvaluationPlan, Phase


def plan_locally(config_path: Path) -> dict:
    """Read exactly the supplied local JSON configuration, preserving unresolved fields.

    Identity strings are declarations, not verification that artifacts exist or that
    hardware can load them. This function grants no execution approval.
    """
    plan = EvaluationPlan.from_json(Path(config_path).read_text(encoding="utf-8"))
    required = (
        "main_revision", "tokenizer_revision", "tokenizer_mapping_hash", "harmony_template_hash",
        "image_digest", "software_revision", "precision", "hardware", "runtime_identity", "parser_version",
        "corpus_hashes", "auxiliary_artifact_hashes", "split_manifest_hash", "dataset_document_count",
        "dataset_token_count", "fitting_limits", "screening_criteria", "worker_grace_seconds", "memory_margin_bytes",
        "execution_phase",
    )
    unresolved = [name for name in required if not getattr(plan, name)]
    budgeted = {budget.phase for budget in plan.phase_budgets}
    unresolved.extend(f"phase_budgets.{phase.value}" for phase in Phase if phase not in budgeted)
    unresolved.extend(("artifact_availability", "corpus_rights_and_temporal_audit", "runtime_feasibility", "approval"))
    return {
        "schema_version": 1,
        "plan": json.loads(plan.to_json()),
        "ready_for_execution": False,
        "unresolved_prerequisites": unresolved,
        "monetary_cost_usd": None,
        "cost_unknown_reason": "Planning performs no work and has no billing evidence",
        "mcs_unavailable_reason": MCS_UNAVAILABLE,
    }
