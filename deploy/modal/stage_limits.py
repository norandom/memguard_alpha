"""Per-stage limits of the Modal serving arm; pure validation, no SDK or network.

Task 7.3 writes one small JSON file per stage from the frozen study plan. The weight
staging step and the serve mode take that file as input and have no approval gate of
their own: the study plan's approval record stays the only approval. Every limit must
fit inside the serving manifest's reviewed ceiling. ``plan_sha256`` names the frozen
study plan file; task 9.4 checks it against that plan before any stage.
"""

import json
import math
import re
from dataclasses import dataclass, fields
from pathlib import Path

from deploy.modal.manifest import REVIEWED_LIMITS, Plan

#: Serving-arm stages in execution order.
STAGES = ("weight_staging", "deployment", "smoke_run", "calibration", "full_run", "negative_control")
WEIGHT_STAGING = STAGES[0]


@dataclass(frozen=True)
class StageLimits:
    plan_sha256: str  # 64 lowercase hex, hash of the frozen study plan file
    stage: str  # one of STAGES
    max_requests: int
    max_output_tokens: int
    max_seconds: int  # hard app lifetime (serve) or function timeout (staging)
    max_usd: float


def _unique_fields(pairs):
    names = [name for name, _ in pairs]
    if len(names) != len(set(names)):
        raise ValueError("Stage limits repeat a field")
    return dict(pairs)


def _no_constant(_name):
    raise ValueError("Stage limits must be finite numbers")


def reviewed_ceiling(plan: Plan) -> dict:
    """The serving manifest's reviewed ceiling, keyed by stage limit field."""
    limits, target_usd = REVIEWED_LIMITS[plan.serving_manifest]
    return {"max_requests": limits.inference_posts, "max_output_tokens": limits.total_output_tokens,
            "max_seconds": limits.gpu_phase_seconds, "max_usd": target_usd}


def parse_stage_limits(data: str | bytes, plan: Plan) -> StageLimits:
    """Validate one stage's limits against the manifest's reviewed ceiling, or refuse."""
    raw = json.loads(data, object_pairs_hook=_unique_fields, parse_constant=_no_constant)
    if not isinstance(raw, dict) or set(raw) != {field.name for field in fields(StageLimits)}:
        raise ValueError("Invalid stage limits fields")
    if not isinstance(raw["plan_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", raw["plan_sha256"]):
        raise ValueError("plan_sha256 must be the frozen study plan's sha256 as 64 lowercase hex characters")
    if raw["stage"] not in STAGES:
        raise ValueError("Unknown stage; expected one of " + ", ".join(STAGES))
    for name in ("max_requests", "max_output_tokens", "max_seconds"):
        if type(raw[name]) is not int or raw[name] < 0:
            raise ValueError(f"Stage limit {name} must be a non-negative integer")
    if raw["max_seconds"] == 0:
        raise ValueError("Stage limit max_seconds must be positive")
    if type(raw["max_usd"]) not in (int, float) or not math.isfinite(raw["max_usd"]) or raw["max_usd"] <= 0:
        raise ValueError("Stage limit max_usd must be a positive finite number")
    for name, ceiling in reviewed_ceiling(plan).items():
        if raw[name] > ceiling:
            raise ValueError(f"Stage limit {name} exceeds the manifest's reviewed ceiling")
    return StageLimits(raw["plan_sha256"], raw["stage"], raw["max_requests"], raw["max_output_tokens"],
                       raw["max_seconds"], float(raw["max_usd"]))


def load_stage_limits(path: str | Path, plan: Plan) -> StageLimits:
    return parse_stage_limits(Path(path).read_bytes(), plan)


def staging_worst_case_usd(limits: StageLimits, plan: Plan, cost: dict) -> float:
    """CPU-only staging run to its timeout at the manifest's cores and memory, no GPU."""
    rate = (plan.resources.cpu_physical_cores * cost["physical_core_usd_per_second"]
            + plan.resources.memory_gib * cost["memory_gib_usd_per_second"])
    return rate * limits.max_seconds


def require_staging_limits(limits: StageLimits, plan: Plan, cost: dict) -> float:
    """Refuse staging unless its worst case fits max_usd; returns that worst case in USD."""
    if limits.stage != WEIGHT_STAGING:
        raise ValueError("Weight staging takes the weight_staging stage limits")
    worst = staging_worst_case_usd(limits, plan, cost)
    if worst > limits.max_usd:
        raise ValueError(f"Staging refused: running to its {limits.max_seconds} s timeout costs up to "
                         f"{worst:.4f} USD, above its max_usd of {limits.max_usd}")
    return worst


def require_serving_limits(limits: StageLimits) -> None:
    """The serve mode takes a serving stage's limits, never the weight staging limits."""
    if limits.stage == WEIGHT_STAGING:
        raise ValueError("The serve mode takes a serving stage's limits, not weight_staging")
