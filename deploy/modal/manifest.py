"""Pure, local setup validation. Run from the checkout with a manifest path.

No SDK, environment lookup, network, deployment or inference exists in this module.
Account-specific unknowns are execution blockers, not fabricated account facts.
"""

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import get_type_hints


@dataclass(frozen=True)
class Target:
    workspace: str | None
    environment: str
    app: str
    endpoint: str | None
    credential_env: str


@dataclass(frozen=True)
class Artifacts:
    model: str
    model_revision: str
    tokenizer: str
    tokenizer_revision: str
    quantization: str
    runtime: str
    image: str
    platform: str
    modal_sdk: str
    score_semantics: str
    attention_backend: str
    moe_backend: str
    observed_hardware: str | None


@dataclass(frozen=True)
class Resources:
    gpu: str
    gpu_count: int
    cpu_physical_cores: int
    memory_gib: int
    context_tokens: int
    tensor_parallel_size: int
    max_num_seqs: int
    concurrency: int
    min_containers: int
    max_containers: int
    buffer_containers: int
    idle_seconds: int


@dataclass(frozen=True)
class Limits:
    inference_posts: int
    eligibility_posts: int
    measurement_posts: int
    retries: int
    output_tokens_per_post: int
    total_output_tokens: int
    request_seconds: int
    gpu_phase_seconds: int
    startup_seconds: int
    cleanup_seconds: int
    temperature: int
    top_p: int
    logprobs: bool
    top_logprobs: int


def _typed(cls, raw):
    if not isinstance(raw, dict) or set(raw) != set(get_type_hints(cls)):
        raise ValueError(f"Invalid {cls.__name__} fields")
    for key, expected in get_type_hints(cls).items():
        valid = isinstance(raw[key], expected)
        if expected is int:
            valid = type(raw[key]) is int
        if not valid:
            raise ValueError(f"Invalid {cls.__name__}.{key} type")
    return cls(**raw)


PILOT_MANIFEST = "configs/modal/gpt-oss-20b-pilot.json"
MEASUREMENT_MANIFEST = "configs/modal/gpt-oss-20b-measurement.json"
GEMMA_MANIFEST = "configs/modal/gemma-3-4b-measurement.json"
_SOURCES = ("deploy/modal/serve.py", "deploy/modal/manifest.py")
# Reviewed bounds per serving manifest, as (limits, reviewed cost.target_usd); load_plan
# requires the manifest's target_usd to equal the second element. The pilot entry is pinned
# by existing approval records; the measurement entry covers M4 Phase A (321 requests,
# 1,024 output tokens per request with reasoning_effort low, USD 2.00 cap). The Gemma entry
# is the serving-arm ceiling of the finance study (13,857 requests, 256 output tokens per
# request, 86,142 s GPU phase, USD 34.34 target; arithmetic in that manifest's caveats).
# A new manifest needs its own reviewed entry here.
REVIEWED_LIMITS = {
    PILOT_MANIFEST: (Limits(8, 1, 7, 0, 512, 4096, 120, 1200, 600, 120, 0, 1, True, 5), 5),
    MEASUREMENT_MANIFEST: (Limits(321, 1, 320, 0, 1024, 328704, 120, 2700, 600, 120, 0, 1, True, 5), 2),
    GEMMA_MANIFEST: (Limits(13857, 1, 13856, 0, 256, 3547392, 120, 86142, 600, 120, 0, 1, True, 5), 34.34),
}
_GPT_OSS_ARTIFACTS = ("openai/gpt-oss-20b", "openai/gpt-oss-20b", "mxfp4", "vllm==0.17.0", "1.6.3", "linux/amd64",
                      "natural_log_probabilities", "TRITON_ATTN", "marlin")
_GPT_OSS_RESOURCES = Resources("A100-40GB", 1, 4, 32, 4096, 1, 1, 1, 0, 1, 0, 60)
# Dense bf16 model: no quantization method and no MoE kernel (moe_backend "none").
_GEMMA_ARTIFACTS = ("google/gemma-3-4b-it", "google/gemma-3-4b-it", "bf16", "vllm==0.17.0", "1.6.3", "linux/amd64",
                    "natural_log_probabilities", "FLASH_ATTN", "none")
_GEMMA_RESOURCES = Resources("A10", 1, 4, 16, 4096, 1, 1, 1, 0, 1, 0, 60)


def _cache(volume: str) -> dict:
    return {"volume": volume, "mount_path": "/cache/huggingface", "staging_gpu_count": 0,
            "stage_separately": True, "retain_after_cleanup": True}


# Reviewed serving candidate per manifest: artifacts, resources and weight cache.
_REVIEWED_CANDIDATES = {
    PILOT_MANIFEST: (_GPT_OSS_ARTIFACTS, _GPT_OSS_RESOURCES, _cache("recall-guard-gpt-oss-20b-weights")),
    MEASUREMENT_MANIFEST: (_GPT_OSS_ARTIFACTS, _GPT_OSS_RESOURCES, _cache("recall-guard-gpt-oss-20b-weights")),
    GEMMA_MANIFEST: (_GEMMA_ARTIFACTS, _GEMMA_RESOURCES, _cache("recall-guard-gemma-3-4b-weights")),
}


def upload_allowlist(manifest: str = PILOT_MANIFEST) -> tuple[str, ...]:
    """Exact upload set for one reviewed serving manifest (repository-relative)."""
    if manifest not in REVIEWED_LIMITS:
        raise ValueError("Unreviewed serving manifest")
    return (*_SOURCES, manifest)


@dataclass(frozen=True)
class Plan:
    target: Target
    artifacts: Artifacts
    resources: Resources
    limits: Limits
    execution_blockers: tuple[str, ...]
    serving_manifest: str = PILOT_MANIFEST

    @property
    def memory_mib(self) -> int:
        """Modal's memory argument uses MiB, while the manifest uses GiB."""
        return self.resources.memory_gib * 1024

    @property
    def volume_name(self) -> str:
        """The weight volume reviewed for this serving manifest (one volume per manifest).

        load_plan requires the manifest's ``cache`` to equal this reviewed entry.
        """
        return _REVIEWED_CANDIDATES[self.serving_manifest][2]["volume"]

    def upload_allowed(self, relative_path: str) -> bool:
        """Exact file allowlist; callers must also reject symlinks before upload."""
        path = PurePosixPath(relative_path)
        return relative_path == path.as_posix() and relative_path in upload_allowlist(self.serving_manifest)


UPLOAD_ALLOWLIST = upload_allowlist(PILOT_MANIFEST)


def load_plan(path: str | Path) -> Plan:
    """Validate the fixed candidate; no account facts or secrets are resolved."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        set(raw)
        != {
            "schema_version",
            "target",
            "artifacts",
            "resources",
            "limits",
            "cache",
            "upload_allowlist",
            "cost",
            "provenance",
        }
        or type(raw["schema_version"]) is not int
        or raw["schema_version"] != 1
    ):
        raise ValueError("Invalid manifest schema")
    target = _typed(Target, raw["target"])
    artifacts = _typed(Artifacts, raw["artifacts"])
    resources = _typed(Resources, raw["resources"])
    limits = _typed(Limits, raw["limits"])
    for name in (target.workspace, target.environment, target.app):
        if name is not None and not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,62}", name):
            raise ValueError("Invalid deployment identifier")
    if target.endpoint is not None and not re.fullmatch(
        r"https://[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)?\.modal\.run/v1", target.endpoint
    ):
        raise ValueError("Invalid explicit Modal endpoint")
    if not re.fullmatch(r"[A-Z][A-Z0-9_]*", target.credential_env):
        raise ValueError("Use an environment reference, never a credential value")
    for revision in (artifacts.model_revision, artifacts.tokenizer_revision):
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Artifact revisions must be commits")
    if not re.fullmatch(r"vllm/vllm-openai@sha256:[0-9a-f]{64}", artifacts.image):
        raise ValueError("Serving image must be a digest")
    configuration = (
        artifacts.model,
        artifacts.tokenizer,
        artifacts.quantization,
        artifacts.runtime,
        artifacts.modal_sdk,
        artifacts.platform,
        artifacts.score_semantics,
        artifacts.attention_backend,
        artifacts.moe_backend,
    )
    if configuration not in {candidate[0] for candidate in _REVIEWED_CANDIDATES.values()}:
        raise ValueError("Unsupported artifact configuration; requires revalidation")
    if resources not in {candidate[1] for candidate in _REVIEWED_CANDIDATES.values()}:
        raise ValueError("Resources differ from the reviewed candidate or units")
    allowlist = raw["upload_allowlist"]
    if (not isinstance(allowlist, list) or len(allowlist) != 3 or tuple(allowlist[:2]) != _SOURCES
            or allowlist[2] not in REVIEWED_LIMITS or allowlist[2] not in _REVIEWED_CANDIDATES):
        raise ValueError("Upload allowlist differs from the reviewed files")
    serving_manifest = allowlist[2]
    reviewed_artifacts, reviewed_resources, reviewed_cache = _REVIEWED_CANDIDATES[serving_manifest]
    # Each manifest serves only its own reviewed candidate; no mixing across models.
    if (configuration, resources) != (reviewed_artifacts, reviewed_resources):
        raise ValueError("Unsupported artifact configuration; requires revalidation")
    reviewed_limits, reviewed_target_usd = REVIEWED_LIMITS[serving_manifest]
    if limits != reviewed_limits:
        raise ValueError("Pilot limits differ from the reviewed bounds" if serving_manifest == PILOT_MANIFEST
                         else "Measurement limits differ from the reviewed bounds")
    if raw["cache"] != reviewed_cache:
        raise ValueError("Cache staging must be separate and CPU-only")
    cost = raw["cost"]
    expected_cost_fields = {
        "pricing_checked_at",
        "pricing_source",
        "gpu_usd_per_second",
        "physical_core_usd_per_second",
        "memory_gib_usd_per_second",
        "volume_usd_per_gib_month",
        "target_usd",
        "hard_ceiling",
        "provider_budget",
        "caveats",
    }
    if not isinstance(cost, dict) or set(cost) != expected_cost_fields:
        raise ValueError("Missing cost assumptions")
    if cost["provider_budget"] is not None:
        raise ValueError("Account budget records belong to the separately reviewed execution plan")
    if any(not isinstance(cost[key], str) or not cost[key] for key in ("pricing_source", "caveats")):
        raise ValueError("Pricing source and enforcement caveats required")
    try:
        timestamp = datetime.fromisoformat(cost["pricing_checked_at"].replace("Z", "+00:00"))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("Invalid pricing timestamp") from exc
    if (timestamp.tzinfo is None or cost["hard_ceiling"] is not False
            or type(cost["target_usd"]) not in (int, float) or cost["target_usd"] != reviewed_target_usd):
        raise ValueError("Dated reviewed USD target required; never an exact spend ceiling")
    for field in (
        "gpu_usd_per_second",
        "physical_core_usd_per_second",
        "memory_gib_usd_per_second",
        "volume_usd_per_gib_month",
    ):
        if type(cost[field]) not in (int, float) or not 0 < cost[field] < 1:
            raise ValueError("Invalid USD rate")
    provenance = raw["provenance"]
    if (
        not isinstance(provenance, dict)
        or any(not isinstance(value, str) or not value for value in provenance.values())
        or not {
            "access",
            "license",
            "license_source",
            "license_prerequisites",
            "model_api",
            "image_registry",
            "image_index_digest",
            "sdk_source",
            "recipe_source",
            "resources_source",
            "compatibility",
            "checked_on",
            "image_tag",
        }
        <= set(provenance)
    ):
        raise ValueError("Artifact access, license and pin provenance required")
    blockers = tuple(
        name for name, value in (("workspace", target.workspace), ("endpoint", target.endpoint)) if value is None
    )
    # Budget review is a later runner responsibility; a setup plan never authorizes execution.
    blockers += ("provider_budget", "account_access_unverified", "paid_execution_not_authorized")
    return Plan(target, artifacts, resources, limits, blockers, serving_manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    try:
        plan = load_plan(args.manifest)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Invalid plan: {type(exc).__name__}\n")
    print(
        json.dumps(
            {
                "mode": "plan",
                "cloud_mutations": 0,
                "inference_posts": 0,
                "memory_mib": plan.memory_mib,
                "execution_blockers": plan.execution_blockers,
            }
        )
    )


if __name__ == "__main__":
    main()
