"""Declarative provider profiles and the single client factory (release-1.0 R2.1-2.2).

A :class:`ProviderProfile` wraps an existing :class:`ModelConfig` and adds only
what the config cannot express: hosting mode, response mapping, pinned Harmony
tokenizer metadata, declared host requirements and the explicit unauthenticated
choice. :func:`connect` dispatches to the existing transport clients and never
issues an inference request. :func:`preflight_host` (R2.3) runs first by
default: credential/tokenizer presence, the free-RAM gate and read-only Ollama
or FastFlowLM metadata, returning a friendly per-host proposal instead of loading a model.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from importlib import resources
from pathlib import Path
from urllib.parse import urlsplit

import requests

from recall_guard.core.completion import (
    HARMONY_ONLY_PARAMETERS,
    CompletionClient,
    CompletionProtocol,
    FeatureChoices,
    ModelConfig,
)
from recall_guard.core.harmony_evidence import HarmonyTokenMap
from recall_guard.core.provider_anthropic import AnthropicClient
from recall_guard.core.providers import HarmonyOpenAICompatibleClient, OpenAICompatibleClient
from recall_guard.core.request_control import CallBudget

#: Checkout root; only relative Harmony tokenizer paths fall back to it.
REPO_ROOT = Path(__file__).resolve().parents[2]
#: Curated profiles ship as package data (``recall_guard/core/provider_profiles/<id>.json``)
#: so an installed wheel can load them by ID (C18).
PROFILE_RESOURCES = resources.files("recall_guard.core").joinpath("provider_profiles")
#: Filesystem view of the same directory for checkouts and regular installs.
PROFILE_DIRECTORY = Path(str(PROFILE_RESOURCES))
#: Documented endpoint placeholder (docs/guides/modal-provider.md); never contacted.
PLACEHOLDER_MARKER = "your-host"
_PROFILE_ID = re.compile(r"[a-z0-9][a-z0-9.\-]*")
_REVISION = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_ENV_REF = re.compile(r"(?:env:)?[A-Za-z_][A-Za-z0-9_]*")
#: Estimate source names, the same strings as ``recall_guard.mia.estimate_source.EstimateSource``
#: (core cannot import mia).
TOP_K_SOURCE = "top_k"
FULL_DISTRIBUTION_SOURCE = "full_distribution"


class ProfileError(ValueError):
    """A profile is invalid or cannot be connected; never contains a secret."""


class HostingMode(StrEnum):
    LOCAL = "local"
    REMOTE_GPU = "remote_gpu"
    MANAGED_API = "managed_api"


class ResponseMapping(StrEnum):
    GENERIC = "generic"
    HARMONY_VLLM = "harmony_vllm"


def _optional_bytes(value, name: str) -> None:
    if value is not None and (type(value) is not int or value <= 0):
        raise ProfileError(f"{name} must be a positive integer or null")


@dataclass(frozen=True)
class HostResourceRequirement:
    """Declarative pre-load host requirements; enforced by host preflight (M1.4).

    ``memory_floor_bytes`` is the free physical memory below which a running
    model host refuses new requests (503 ``memory_low``, task 6.4); when unset,
    ``memory_floor()`` derives one quarter of ``min_free_ram_bytes``.
    """

    min_free_ram_bytes: int | None = None
    recommended_ram_bytes: int | None = None
    context_tokens: int = 2048
    fallback_profile_ids: tuple[str, ...] = ("modal-gpt-oss-20b", "ollama-gemma3-4b")
    memory_floor_bytes: int | None = None

    def __post_init__(self) -> None:
        _optional_bytes(self.min_free_ram_bytes, "min_free_ram_bytes")
        _optional_bytes(self.recommended_ram_bytes, "recommended_ram_bytes")
        _optional_bytes(self.memory_floor_bytes, "memory_floor_bytes")
        if (self.memory_floor_bytes is not None and self.min_free_ram_bytes is not None
                and self.memory_floor_bytes > self.min_free_ram_bytes):
            raise ProfileError("memory_floor_bytes must not exceed min_free_ram_bytes")
        if type(self.context_tokens) is not int or self.context_tokens < 1:
            raise ProfileError("context_tokens must be a positive integer")
        object.__setattr__(self, "fallback_profile_ids", tuple(self.fallback_profile_ids))
        if not all(isinstance(item, str) and _PROFILE_ID.fullmatch(item) for item in self.fallback_profile_ids):
            raise ProfileError("fallback_profile_ids must contain profile IDs")

    def memory_floor(self) -> int | None:
        """The recorded floor, else one quarter of ``min_free_ram_bytes``, else None."""
        if self.memory_floor_bytes is not None:
            return self.memory_floor_bytes
        return None if self.min_free_ram_bytes is None else self.min_free_ram_bytes // 4

    def to_dict(self) -> dict:
        # The floor is emitted only when set, so profiles without one export unchanged.
        return {"min_free_ram_bytes": self.min_free_ram_bytes, "recommended_ram_bytes": self.recommended_ram_bytes,
                "context_tokens": self.context_tokens, "fallback_profile_ids": list(self.fallback_profile_ids),
                **({} if self.memory_floor_bytes is None else {"memory_floor_bytes": self.memory_floor_bytes})}


@dataclass(frozen=True)
class HarmonyMappingConfig:
    """Pinned tokenizer metadata required when ``response_mapping`` is Harmony."""

    tokenizer_path: str
    tokenizer_revision: str
    tokenizer_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.tokenizer_path, str) or not self.tokenizer_path.strip():
            raise ProfileError("harmony.tokenizer_path must be a nonempty path")
        if not isinstance(self.tokenizer_revision, str) or not _REVISION.fullmatch(self.tokenizer_revision):
            raise ProfileError("harmony.tokenizer_revision must be a pinned 40-hex revision")
        if self.tokenizer_sha256 is not None and (
                not isinstance(self.tokenizer_sha256, str) or not _SHA256.fullmatch(self.tokenizer_sha256)):
            raise ProfileError("harmony.tokenizer_sha256 must be 64 lowercase hex characters or null")

    def resolve_path(self) -> Path:
        """Absolute paths as given; relative ones from the working directory, then the checkout."""
        path = Path(self.tokenizer_path)
        if path.is_absolute() or path.exists():
            return path
        return REPO_ROOT / path

    def to_dict(self) -> dict:
        return {"tokenizer_path": self.tokenizer_path, "tokenizer_revision": self.tokenizer_revision,
                "tokenizer_sha256": self.tokenizer_sha256}


def _is_loopback_url(value) -> bool:
    """True for a credential-free loopback HTTP(S) URL without query or fragment."""
    try:
        parsed = urlsplit(value) if isinstance(value, str) else None
        if parsed is None or parsed.scheme not in {"http", "https"} or parsed.query or parsed.fragment:
            return False
        _ = parsed.port  # Validate a supplied port without echoing malformed input.
    except ValueError:
        return False
    return (parsed.username is None and parsed.password is None
            and (parsed.hostname or "").lower() in LOOPBACK_HOSTS)


@dataclass(frozen=True)
class FullDistributionHostConfig:
    """Full-distribution host that teacher-forces the profile's own model artifact (Requirement 3.4).

    ``endpoint`` is the host's loopback base URL (the client appends its route).
    ``endpoint_ref`` optionally names an environment variable holding the bearer
    gateway's base URL, and ``credential_ref`` names the gateway bearer, both as
    environment-variable names only. ``model_digest`` is the GGUF the host loads;
    a profile requires it to equal its declared ``artifact_digest``.
    ``top_k_keep`` is the number of alternatives the host keeps per position.
    """

    endpoint: str
    model_digest: str
    credential_ref: str
    endpoint_ref: str | None = None
    top_k_keep: int = 100

    def __post_init__(self) -> None:
        if not _is_loopback_url(self.endpoint):
            raise ProfileError("full_distribution.endpoint must be a credential-free loopback HTTP(S) URL")
        if not isinstance(self.model_digest, str) or not _DIGEST.fullmatch(self.model_digest):
            raise ProfileError("full_distribution.model_digest must be sha256: and 64 lowercase hex characters")
        if not isinstance(self.credential_ref, str) or not _ENV_REF.fullmatch(self.credential_ref):
            raise ProfileError("full_distribution.credential_ref must name an environment variable")
        if self.endpoint_ref is not None and (
                not isinstance(self.endpoint_ref, str) or not _ENV_REF.fullmatch(self.endpoint_ref)):
            raise ProfileError("full_distribution.endpoint_ref must name an environment variable")
        if type(self.top_k_keep) is not int or self.top_k_keep < 1:
            raise ProfileError("full_distribution.top_k_keep must be a positive integer")

    def to_dict(self) -> dict:
        return {"endpoint": self.endpoint, "endpoint_ref": self.endpoint_ref, "credential_ref": self.credential_ref,
                "model_digest": self.model_digest, "top_k_keep": self.top_k_keep}


_PROFILE_FIELDS = frozenset({"schema_version", "profile_id", "display_name", "hosting_mode", "model_config",
                             "credential_ref", "endpoint_ref", "response_mapping", "harmony", "host_requirements",
                             "allow_unauthenticated", "notes", "logprob_mass_tolerance", "full_distribution"})
#: Upper bound for a profile's ``logprob_mass_tolerance`` (absorbs rounding, never real mass).
MAX_LOGPROB_MASS_TOLERANCE = 1e-3


@dataclass(frozen=True)
class ProviderProfile:
    """Declarative inference profile wrapping :class:`ModelConfig` (R2.1).

    ``endpoint_ref`` optionally names an environment variable holding a
    user-specific endpoint (for example a registered Modal host). It is
    resolved only by :func:`connect`, like a credential reference.
    ``logprob_mass_tolerance`` (default ``None``: no mass check) lets evidence
    validation require each position's top-k probability mass to be at most
    ``1 + tolerance``, for runtimes whose float32 rounding exceeds 1 slightly.
    ``full_distribution`` (default ``None``) declares a full-distribution host
    on the same model artifact; it owns the capability because core cannot
    import mia. :attr:`estimate_sources` lists the sources the profile offers.
    """

    profile_id: str
    display_name: str
    hosting_mode: HostingMode
    model_config: ModelConfig
    response_mapping: ResponseMapping = ResponseMapping.GENERIC
    harmony: HarmonyMappingConfig | None = None
    host_requirements: HostResourceRequirement = field(default_factory=HostResourceRequirement)
    allow_unauthenticated: bool = False
    endpoint_ref: str | None = None
    notes: str = ""
    logprob_mass_tolerance: float | None = None
    full_distribution: FullDistributionHostConfig | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or not _PROFILE_ID.fullmatch(self.profile_id):
            raise ProfileError("profile_id must be lowercase letters, digits, '.' or '-'")
        if not isinstance(self.display_name, str) or not self.display_name.strip():
            raise ProfileError("display_name must be nonempty")
        try:
            object.__setattr__(self, "hosting_mode", HostingMode(self.hosting_mode))
            object.__setattr__(self, "response_mapping", ResponseMapping(self.response_mapping))
        except ValueError:
            raise ProfileError("unknown hosting_mode or response_mapping") from None
        if not isinstance(self.model_config, ModelConfig):
            raise ProfileError("model_config must be ModelConfig")
        if not isinstance(self.host_requirements, HostResourceRequirement):
            raise ProfileError("host_requirements must be HostResourceRequirement")
        if not isinstance(self.allow_unauthenticated, bool) or not isinstance(self.notes, str):
            raise ProfileError("allow_unauthenticated must be boolean and notes text")
        if self.endpoint_ref is not None and (
                not isinstance(self.endpoint_ref, str) or not _ENV_REF.fullmatch(self.endpoint_ref)):
            raise ProfileError("endpoint_ref must name an environment variable")
        tolerance = self.logprob_mass_tolerance
        if tolerance is not None and (isinstance(tolerance, bool) or not isinstance(tolerance, (int, float))
                                      or not 0 <= tolerance < MAX_LOGPROB_MASS_TOLERANCE):
            raise ProfileError("logprob_mass_tolerance must be a number in [0, 1e-3) or null")
        self._check_mapping()
        self._check_full_distribution()
        if self.allow_unauthenticated and (
                self.hosting_mode != HostingMode.LOCAL or self.model_config.credential_ref is not None):
            raise ProfileError("allow_unauthenticated requires a local profile without a credential reference")

    def _check_mapping(self) -> None:
        if self.response_mapping == ResponseMapping.HARMONY_VLLM:
            if not isinstance(self.harmony, HarmonyMappingConfig):
                raise ProfileError("harmony_vllm mapping requires pinned harmony tokenizer metadata")
            if self.model_config.protocol != CompletionProtocol.OPENAI_CHAT:
                raise ProfileError("harmony_vllm mapping requires the openai_chat protocol")
            if self.model_config.parameters.get("return_token_ids") is not True:
                raise ProfileError("harmony_vllm mapping requires return_token_ids=true")
        else:
            if self.harmony is not None:
                raise ProfileError("harmony metadata is only valid with the harmony_vllm mapping")
            if only := sorted(self.model_config.parameters.keys() & HARMONY_ONLY_PARAMETERS):
                raise ProfileError(f"{', '.join(only)} is only valid with the harmony_vllm mapping")

    def _check_full_distribution(self) -> None:
        host = self.full_distribution
        if host is None:
            return
        if not isinstance(host, FullDistributionHostConfig):
            raise ProfileError("full_distribution must be FullDistributionHostConfig or null")
        identity = self.model_config.local_identity
        declared = None if identity is None else identity.declared.artifact_digest
        if host.model_digest != declared:
            # Generation and full distribution must score the same GGUF.
            raise ProfileError(f"full_distribution.model_digest {host.model_digest} differs from the declared "
                               f"artifact_digest {declared}; both sources must use the same model artifact")

    @property
    def estimate_sources(self) -> tuple[str, ...]:
        """Estimate sources this profile offers: top-k when it requests logprobs, plus full distribution."""
        sources = (TOP_K_SOURCE,) if self.model_config.parameters.get("logprobs") is True else ()
        if self.full_distribution is not None:
            sources += (FULL_DISTRIBUTION_SOURCE,)
        return sources

    def require_source(self, name: str) -> None:
        """Refuse an estimate source this profile does not offer, before any request (Requirement 3.4)."""
        available = self.estimate_sources
        if name not in available:
            raise ProfileError(f"Profile {self.profile_id!r} does not offer estimate source {name!r}; "
                               f"available: {', '.join(available) or 'none'}")

    @property
    def mapping_revision(self) -> str | None:
        """Declared mapping identity; a connected client reports the observed one."""
        if self.harmony is None:
            return None
        return self.harmony.tokenizer_sha256 or self.harmony.tokenizer_revision

    def to_dict(self) -> dict:
        """Credential-free export; ``credential_ref`` is a variable name only."""
        return {
            "schema_version": 1, "profile_id": self.profile_id, "display_name": self.display_name,
            "hosting_mode": self.hosting_mode.value, "model_config": self.model_config.to_dict(),
            "credential_ref": self.model_config.credential_ref, "endpoint_ref": self.endpoint_ref,
            "response_mapping": self.response_mapping.value,
            "harmony": None if self.harmony is None else self.harmony.to_dict(),
            "host_requirements": self.host_requirements.to_dict(),
            "allow_unauthenticated": self.allow_unauthenticated, "notes": self.notes,
            # Emitted only when set, so profiles without a mass check export unchanged.
            **({} if self.logprob_mass_tolerance is None else
               {"logprob_mass_tolerance": self.logprob_mass_tolerance}),
            **({} if self.full_distribution is None else {"full_distribution": self.full_distribution.to_dict()}),
        }

    @classmethod
    def from_dict(cls, data: Mapping) -> ProviderProfile:
        if not isinstance(data, Mapping) or data.keys() - _PROFILE_FIELDS:
            raise ProfileError("Unknown or malformed provider profile fields")
        if data.get("schema_version", 1) != 1:
            raise ProfileError("Unsupported provider profile schema_version")
        try:
            config = ModelConfig.from_dict(data["model_config"])
            config = replace(config, credential_ref=data.get("credential_ref"))
            harmony = data.get("harmony")
            requirements = data.get("host_requirements") or {}
            full_distribution = data.get("full_distribution")
            return cls(
                profile_id=data["profile_id"], display_name=data["display_name"],
                hosting_mode=data["hosting_mode"], model_config=config,
                response_mapping=data.get("response_mapping", ResponseMapping.GENERIC.value),
                harmony=None if harmony is None else HarmonyMappingConfig(**harmony),
                host_requirements=HostResourceRequirement(**requirements),
                allow_unauthenticated=data.get("allow_unauthenticated", False),
                endpoint_ref=data.get("endpoint_ref"), notes=data.get("notes", ""),
                logprob_mass_tolerance=data.get("logprob_mass_tolerance"),
                full_distribution=(None if full_distribution is None
                                   else FullDistributionHostConfig(**full_distribution)),
            )
        except ProfileError:
            raise
        except (KeyError, TypeError, ValueError) as error:
            raise ProfileError(f"Invalid provider profile: {error}") from None


def load_profile(id_or_path: ProviderProfile | str | Path) -> ProviderProfile:
    """Load a curated profile by ID (shipped package data) or a profile JSON by path."""
    if isinstance(id_or_path, ProviderProfile):
        return id_or_path
    if isinstance(id_or_path, str) and _PROFILE_ID.fullmatch(id_or_path) and not id_or_path.endswith(".json"):
        source = PROFILE_RESOURCES.joinpath(f"{id_or_path}.json")
        if not source.is_file():
            known = ", ".join(sorted(available_profiles())) or "none"
            raise ProfileError(f"Unknown provider profile {id_or_path!r}; available: {known}")
    elif isinstance(id_or_path, (str, Path)):
        source = Path(id_or_path)
    else:
        raise TypeError("profile must be a ProviderProfile, profile ID or path")
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProfileError(f"Cannot read provider profile {source.name}: {error.__class__.__name__}") from None
    return ProviderProfile.from_dict(data)


def available_profiles() -> tuple[str, ...]:
    """IDs of the curated profiles shipped inside the package."""
    return tuple(sorted(item.name.removesuffix(".json") for item in PROFILE_RESOURCES.iterdir()
                        if item.is_file() and item.name.endswith(".json")))


def _resolved_config(profile: ProviderProfile, environment: Mapping[str, str]) -> ModelConfig:
    config = profile.model_config
    if profile.endpoint_ref is not None:
        name = profile.endpoint_ref.removeprefix("env:")
        endpoint = environment.get(name)
        if isinstance(endpoint, str) and endpoint.strip():
            try:
                config = replace(config, endpoint=endpoint.strip())
            except ValueError:
                raise ProfileError(f"{name} must be a credential-free HTTP(S) chat-completions URL") from None
    host = urlsplit(config.endpoint or "").hostname or ""
    if PLACEHOLDER_MARKER in host.lower():
        hint = f"set {profile.endpoint_ref.removeprefix('env:')} or " if profile.endpoint_ref else ""
        raise ProfileError(f"Profile {profile.profile_id!r} has a placeholder endpoint; "
                           f"{hint}supply the registered provider endpoint before connecting")
    return config


def _token_map(profile: ProviderProfile) -> HarmonyTokenMap:
    harmony = profile.harmony
    path = harmony.resolve_path()
    if not path.is_file():
        raise ProfileError(f"Profile {profile.profile_id!r} requires the pinned tokenizer at {harmony.tokenizer_path}")
    try:
        token_map = HarmonyTokenMap.from_tokenizer_json(path, tokenizer_revision=harmony.tokenizer_revision)
    except (KeyError, TypeError, ValueError) as error:
        raise ProfileError(f"Pinned Harmony tokenizer is unusable: {error}") from None
    if harmony.tokenizer_sha256 is not None and token_map.tokenizer_sha256 != harmony.tokenizer_sha256:
        raise ProfileError("Pinned Harmony tokenizer SHA-256 differs from the profile")
    return token_map


# --- Host preflight (R2.3, C1) ------------------------------------------------

GIB = 1024 ** 3
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
_SEMVER = re.compile(r"\d+\.\d+\.\d+")
#: A host memory snapshot older than this is refused as stale (C21).
SNAPSHOT_MAX_AGE = timedelta(minutes=10)
#: A snapshot stamped further in the future than this is refused (clock skew allowance).
SNAPSHOT_CLOCK_SKEW = timedelta(minutes=2)
#: Host-side helper that writes a snapshot; the container sees runs/container as /outputs.
SNAPSHOT_HELPER = "scripts/host_memory_snapshot.ps1"
SNAPSHOT_CONTAINER_PATH = "/outputs/host-memory.json"
#: Process-level override for container detection: ``1`` (in a container) or ``0`` (not).
IN_CONTAINER_ENV = "RECALL_GUARD_IN_CONTAINER"
_SNAPSHOT_STEP = (f"On the model host, run {SNAPSHOT_HELPER} (it writes runs/container/host-memory.json), "
                  f"then pass --host-snapshot {SNAPSHOT_CONTAINER_PATH} (the path the container sees).")
_CONTAINER_FILES = (".dockerenv", "run/.containerenv")
_CGROUP_MARKERS = ("docker", "containerd", "kubepods", "libpod", "lxc")


def running_in_container(*, environment: Mapping[str, str] | None = None, root: str | Path = "/") -> bool:
    """True when this process runs inside a container (C20).

    ``RECALL_GUARD_IN_CONTAINER`` (``1``/``0``) wins; otherwise ``/.dockerenv``,
    ``/run/.containerenv`` or a container runtime in ``/proc/1/cgroup``. Inside a
    container local memory belongs to the container or its VM, never the model host.
    """
    environment = os.environ if environment is None else environment
    override = str(environment.get(IN_CONTAINER_ENV, "")).strip().lower()
    if override in {"1", "true", "yes"}:
        return True
    if override in {"0", "false", "no"}:
        return False
    base = Path(root)
    if any((base / name).exists() for name in _CONTAINER_FILES):
        return True
    try:
        cgroup = (base / "proc" / "1" / "cgroup").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return any(marker in cgroup for marker in _CGROUP_MARKERS)


def _parse_observed_at(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.strip())
    if parsed.tzinfo is None:
        raise ValueError("naive timestamp")
    return parsed.astimezone(UTC)


@dataclass(frozen=True)
class HostMemorySnapshot:
    """Free RAM measured ON the model host by a host-side helper (C20, C21).

    Required inside a container and for non-loopback endpoints: a container or
    client machine must never substitute its own memory for the Ollama host's.
    ``observed_at`` is a timezone-aware ISO-8601 time (UTC recommended); a
    snapshot without it, or older than the preflight's maximum age, is refused.
    """

    available_bytes: int
    total_bytes: int | None = None
    observed_at: str | None = None
    host: str | None = None
    source: str = "host_snapshot"

    def __post_init__(self) -> None:
        if type(self.available_bytes) is not int or self.available_bytes < 0:
            raise ProfileError("host_snapshot.available_bytes must be a nonnegative integer")
        if self.total_bytes is not None and (type(self.total_bytes) is not int or self.total_bytes <= 0):
            raise ProfileError("host_snapshot.total_bytes must be a positive integer or null")
        if self.observed_at is not None:
            try:
                _parse_observed_at(self.observed_at)
            except (AttributeError, TypeError, ValueError):
                raise ProfileError("host_snapshot.observed_at must be a timezone-aware ISO-8601 time") from None
        if self.host is not None and (not isinstance(self.host, str) or not self.host.strip()
                                      or len(self.host) > 255):
            raise ProfileError("host_snapshot.host must be a short nonempty name or null")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ProfileError("host_snapshot.source must be nonempty text")

    @property
    def observed_datetime(self) -> datetime | None:
        return None if self.observed_at is None else _parse_observed_at(self.observed_at)

    def to_dict(self) -> dict:
        return {"available_bytes": self.available_bytes, "total_bytes": self.total_bytes,
                "observed_at": self.observed_at, "host": self.host, "source": self.source}

    @classmethod
    def coerce(cls, value: HostMemorySnapshot | Mapping) -> HostMemorySnapshot:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ProfileError("host_snapshot must be a HostMemorySnapshot or mapping")
        allowed = {"available_bytes", "total_bytes", "observed_at", "host", "source"}
        return cls(**{key: value[key] for key in allowed if key in value})

    @classmethod
    def from_file(cls, path: str | Path) -> HostMemorySnapshot:
        """Load the JSON written by the host helper (a UTF-8 BOM is tolerated)."""
        path = Path(path)
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as error:
            raise ProfileError(f"Cannot read host memory snapshot {path.name}: {error.__class__.__name__}") from None
        return cls.coerce(data)

    def freshness_problem(self, *, now: datetime | None = None,
                          max_age: timedelta = SNAPSHOT_MAX_AGE) -> tuple[str, str] | None:
        """``None`` when fresh, else ``(reason, plain-English explanation)``."""
        observed = self.observed_datetime
        where = self.host or "the model host"
        if observed is None:
            return "host_memory_unknown", ("The host memory snapshot has no observed_at time, so its age is "
                                           "unknown and it cannot pass the memory gate.")
        now = (datetime.now(UTC) if now is None else now).astimezone(UTC)
        age = now - observed
        if age < -SNAPSHOT_CLOCK_SKEW:
            return "host_memory_stale", (
                f"The host memory snapshot from {where} is dated {_minutes(-age)} in the future, beyond the "
                f"{_minutes(SNAPSHOT_CLOCK_SKEW)} clock-skew allowance. Check the clocks on both machines.")
        if age > max_age:
            return "host_memory_stale", (
                f"The host memory snapshot from {where} is {_minutes(age)} old; the limit is {_minutes(max_age)}, "
                "because free memory changes quickly.")
        return None


def _minutes(delta: timedelta) -> str:
    minutes = delta.total_seconds() / 60
    return f"{minutes:.1f} minutes" if minutes < 120 else f"{minutes / 60:.1f} hours"


@dataclass(frozen=True)
class HostPreflightReport:
    """Zero-inference readiness verdict for one profile on one host (never global)."""

    profile_id: str
    model: str
    passed: bool
    reason: str | None = None
    recommendation: str = ""
    measured_free_ram_bytes: int | None = None
    required_free_ram_bytes: int | None = None
    total_ram_bytes: int | None = None
    memory_source: str | None = None
    fallback_profile_ids: tuple[str, ...] = ()
    runtime_version: str | None = None
    artifact_digest: str | None = None
    model_installed: bool | None = None
    loaded_models: tuple[str, ...] | None = None
    observed_backend: str | None = None
    effective_context_tokens: int | None = None
    #: ``"enforced"`` or ``"skipped_model_already_loaded"``; None when no RAM gate applies.
    ram_gate: str | None = None

    def to_dict(self) -> dict:
        return {
            "profile_id": self.profile_id, "model": self.model, "passed": self.passed, "reason": self.reason,
            "recommendation": self.recommendation, "measured_free_ram_bytes": self.measured_free_ram_bytes,
            "required_free_ram_bytes": self.required_free_ram_bytes, "total_ram_bytes": self.total_ram_bytes,
            "memory_source": self.memory_source, "fallback_profile_ids": list(self.fallback_profile_ids),
            "runtime_version": self.runtime_version, "artifact_digest": self.artifact_digest,
            "model_installed": self.model_installed,
            "loaded_models": None if self.loaded_models is None else list(self.loaded_models),
            "observed_backend": self.observed_backend, "effective_context_tokens": self.effective_context_tokens,
            "ram_gate": self.ram_gate,
        }


class PreflightError(ProfileError):
    """Host preflight refused the profile; ``report`` carries the friendly proposal."""

    def __init__(self, report: HostPreflightReport) -> None:
        super().__init__(report.recommendation or report.reason or "host preflight failed")
        self.report = report


def _psutil_memory() -> tuple[int, int]:
    import psutil  # Core dependency; imported lazily so offline callers pay nothing.

    memory = psutil.virtual_memory()
    return int(memory.available), int(memory.total)


def _gib(value: int) -> str:
    return f"{value / GIB:.1f} GiB"


def _fallback_text(fallbacks: tuple[str, ...]) -> str:
    return " or ".join(f"--profile {item}" for item in fallbacks)


def _ram_recommendation(model: str, measured: int, required: int, fallbacks: tuple[str, ...]) -> str:
    options = f"Try {_fallback_text(fallbacks)}, or run" if fallbacks else "Run"
    return (f"{_gib(measured)} free, {_gib(required)} required for {model}. Nothing was loaded and no "
            f"request was sent. {options} on a machine with at least {_gib(required)} of free RAM.")


def observe_ollama_host(endpoint: str, model: str, *, http_get=None, headers: Mapping[str, str] | None = None) -> dict:
    """Read-only Ollama metadata (``/api/version``, ``/api/tags``, ``/api/ps``).

    Never generates, pulls, loads or unloads. ``http_get(url) -> object`` returns
    parsed JSON or raises; the default uses a proxy-free ``requests`` session
    sending ``headers`` (for example the bearer token of an authenticating gateway).
    Raises ``ValueError('host_metadata_unavailable')`` on any unusable answer.
    """
    parsed = urlsplit(endpoint)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    try:
        if http_get is None:
            with requests.Session() as session:
                session.trust_env = False

                def fetch(path: str):
                    result = session.get(origin + path, headers=dict(headers or {}), timeout=10,
                                         allow_redirects=False)
                    if result.status_code != 200:
                        raise ValueError("host_metadata_unavailable")
                    return result.json()

                version, tags, running = fetch("/api/version"), fetch("/api/tags"), fetch("/api/ps")
        else:
            version, tags, running = (http_get(origin + path) for path in ("/api/version", "/api/tags", "/api/ps"))
        installed_models, models = tags["models"], running["models"]
        if not isinstance(installed_models, list) or not isinstance(models, list):
            raise TypeError("models must be lists")
        installed = next((m for m in installed_models if isinstance(m, dict) and m.get("name") == model), None)
        current = next((m for m in models if isinstance(m, dict) and m.get("name") == model), {})
        loaded = [model if isinstance(m, dict) and m.get("name") == model else "unexpected_model" for m in models]
    except (requests.RequestException, ValueError, KeyError, TypeError, AttributeError):
        raise ValueError("host_metadata_unavailable") from None

    def count(value: object) -> int | None:
        return value if type(value) is int and value >= 0 else None

    size, vram = count(current.get("size")), count(current.get("size_vram"))
    backend = None
    if size is not None and vram is not None and size > 0:
        backend = "CPU" if vram == 0 else "GPU" if vram >= size else "CPU/GPU offload"
    runtime_version = version.get("version") if isinstance(version, dict) else None
    digest = (installed or {}).get("digest")
    if isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest):
        digest = "sha256:" + digest
    return {"runtime_version": runtime_version if isinstance(runtime_version, str)
                and _SEMVER.fullmatch(runtime_version) else None,
            "digest": digest if isinstance(digest, str) and _DIGEST.fullmatch(digest) else None,
            "loaded_models": loaded, "effective_context_tokens": count(current.get("context_length")),
            "model_allocation_bytes": size, "accelerator_allocation_bytes": vram,
            "observed_backend": backend, "model_installed": installed is not None}


#: Marker of the gateway-synthesized FastFlowLM ``/api/tags`` (scripts/hosts/serve_local_flm.ps1).
#: FLM's own ``/api/tags`` lists its whole catalog, so only this answer proves installation.
FLM_INSTALLED_TAGS_SOURCE = "flm list --filter installed --json"
FLM_RUNTIME = "fastflowlm"
#: Local runtimes whose ``/api/ps`` names the resident model (RAM-gate skip when loaded).
_PS_PROVIDERS = frozenset({"ollama", FLM_RUNTIME})


def _metadata_fetcher(endpoint: str, http_get, headers: Mapping[str, str] | None):
    """``(fetch(path) -> json, close())``: proxy-free, no redirects, 200 only."""
    parsed = urlsplit(endpoint)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    if http_get is not None:
        return (lambda path: http_get(origin + path)), (lambda: None)
    session = requests.Session()
    session.trust_env = False

    def fetch(path: str):
        result = session.get(origin + path, headers=dict(headers or {}), timeout=10, allow_redirects=False)
        if result.status_code != 200:
            raise ValueError("host_metadata_unavailable")
        return result.json()
    return fetch, session.close


def observe_fastflowlm_host(endpoint: str, model: str, *, http_get=None,
                            headers: Mapping[str, str] | None = None) -> dict:
    """Read-only FastFlowLM metadata through the authenticating FLM gateway.

    Routes: ``/api/version``, ``/api/tags`` (gateway-synthesized: installed models
    only, plus the serving process's launch context), ``/api/ps`` and
    ``/api/npu/status``. Never generates, pulls, loads or unloads. FLM reports no
    ``size_vram``: the backend is ``"NPU"`` because FastFlowLM executes only on
    the NPU and the runtime reports the NPU available, never inferred from memory
    allocation. ``model_installed`` is ``None`` when ``/api/tags`` is FLM's own
    catalog (no gateway marker), because the catalog does not show installation.
    Raises ``ValueError('host_metadata_unavailable')`` on any unusable answer.
    """
    fetch, close = _metadata_fetcher(endpoint, http_get, headers)
    try:
        version, tags, running, npu = (fetch(path) for path in
                                       ("/api/version", "/api/tags", "/api/ps", "/api/npu/status"))
        installed_models, models = tags["models"], running["models"]
        if not isinstance(installed_models, list) or not isinstance(models, list) or not isinstance(npu, dict):
            raise TypeError("models must be lists")
        synthesized = tags.get("source") == FLM_INSTALLED_TAGS_SOURCE
        installed = next((m for m in installed_models if isinstance(m, dict) and m.get("name") == model), None)
        loaded = [model if isinstance(m, dict) and m.get("name") == model else "unexpected_model" for m in models]
        serving = tags.get("serving") if synthesized and isinstance(tags.get("serving"), dict) else {}
    except (requests.RequestException, ValueError, KeyError, TypeError, AttributeError):
        raise ValueError("host_metadata_unavailable") from None
    finally:
        close()
    runtime_version = version.get("version") if isinstance(version, dict) else None
    digest = (installed or {}).get("digest") if synthesized else None
    context = serving.get("context_length") if serving.get("model") == model else None
    npu_available = npu.get("npu_available") is True
    return {"runtime_version": runtime_version if isinstance(runtime_version, str)
                and _SEMVER.fullmatch(runtime_version) else None,
            "digest": digest if isinstance(digest, str) and _DIGEST.fullmatch(digest) else None,
            "loaded_models": loaded,
            "effective_context_tokens": context if type(context) is int and context > 0 else None,
            "model_allocation_bytes": None, "accelerator_allocation_bytes": None,
            "observed_backend": "NPU" if npu_available else None, "npu_available": npu_available,
            "model_installed": (installed is not None) if synthesized else None}


def _model_already_loaded(endpoint: str, model: str, http_get=None, headers: Mapping[str, str] | None = None) -> bool:
    """One read-only ``/api/ps`` call; any failure counts as not loaded."""
    parsed = urlsplit(endpoint)
    url = f"{parsed.scheme}://{parsed.netloc}/api/ps"
    try:
        if http_get is None:
            with requests.Session() as session:
                session.trust_env = False
                result = session.get(url, headers=dict(headers or {}), timeout=10, allow_redirects=False)
                if result.status_code != 200:
                    return False
                running = result.json()
        else:
            running = http_get(url)
        models = running["models"]
        return isinstance(models, list) and any(isinstance(m, dict) and m.get("name") == model for m in models)
    except (requests.RequestException, ValueError, KeyError, TypeError, AttributeError):
        return False


def _credential_failure(profile: ProviderProfile, environment: Mapping[str, str]) -> str | None:
    reference = profile.model_config.credential_ref
    if reference is None:
        return None
    name = reference.removeprefix("env:")
    value = environment.get(name)
    if isinstance(value, str) and value.strip():
        return None
    return (f"Profile {profile.profile_id!r} needs credential {name} in the environment. "
            f"Set {name} (the value is never printed or stored) and try again.")


def metadata_headers(profile: ProviderProfile, environment: Mapping[str, str]) -> dict[str, str]:
    """Bearer header for read-only host metadata when the profile names a credential.

    A protected local endpoint (for example the authenticating Windows Ollama
    gateway) refuses ``/api/*`` without it. Unauthenticated profiles get ``{}``.
    """
    reference = profile.model_config.credential_ref
    if reference is None:
        return {}
    value = environment.get(reference.removeprefix("env:"))
    if not isinstance(value, str) or not value.strip():
        raise ProfileError(f"Profile {profile.profile_id!r} needs credential "
                           f"{reference.removeprefix('env:')} in the environment")
    return {"Authorization": f"Bearer {value.strip()}"}


def _is_loopback(endpoint) -> bool:
    return (urlsplit(endpoint or "").hostname or "").lower() in LOOPBACK_HOSTS


def _memory_gate(endpoint, host_snapshot, memory_reader, *, in_container, now, max_age):
    """Return ``(measured, total, source, refusal)``.

    ``refusal`` is ``None`` or ``(reason, explanation)``; ``explanation`` is None
    when the generic unknown-memory text applies. Local memory is read only for a
    loopback endpoint outside a container (C20); snapshots must be fresh (C21).
    """
    if host_snapshot is not None:
        snapshot = HostMemorySnapshot.coerce(host_snapshot)
        refusal = snapshot.freshness_problem(now=now, max_age=max_age)
        if refusal is not None:
            return None, None, snapshot.source, refusal
        return snapshot.available_bytes, snapshot.total_bytes, snapshot.source, None
    if in_container or not _is_loopback(endpoint):
        return None, None, None, ("host_memory_unknown", None)
    try:
        available, total = (memory_reader or _psutil_memory)()
    except (OSError, RuntimeError, ValueError, TypeError):
        return None, None, None, ("host_memory_unknown", None)
    if type(available) is not int or available < 0:
        return None, None, None, ("host_memory_unknown", None)
    return available, total if type(total) is int and total > 0 else None, "local_host_available_ram", None


def _unknown_memory_text(endpoint, model: str, required: int, in_container: bool) -> str:
    host = urlsplit(endpoint or "").hostname or "the model host"
    if in_container:
        return (f"Recall Guard is running in a container, so this machine's memory is not the model host's. "
                f"Free RAM on {host} is unknown, and {model} needs {_gib(required)} free. Nothing was loaded. "
                + _SNAPSHOT_STEP)
    return (f"Free RAM on {host} could not be measured from here, and {model} needs "
            f"{_gib(required)} free. Nothing was loaded. Pass a host memory snapshot taken on {host}, "
            f"or run on the model host itself. {_SNAPSHOT_STEP}")


def preflight_host(
    profile: ProviderProfile | str | Path,
    *,
    environment: Mapping[str, str] | None = None,
    host_snapshot: HostMemorySnapshot | Mapping | None = None,
    memory_reader=None,
    http_get=None,
    in_container: bool | None = None,
    max_snapshot_age: timedelta = SNAPSHOT_MAX_AGE,
    now: datetime | None = None,
) -> HostPreflightReport:
    """Zero-inference host checks before any client is built or model loaded.

    Order: credential and pinned tokenizer presence (no I/O), then the free-RAM
    gate, then read-only Ollama metadata. A failed gate stops there; the only
    HTTP before an ``insufficient_free_ram`` verdict is one read-only ``/api/ps``
    call (loopback local reading or a host snapshot), which passes the gate when
    the target model is already loaded.
    Free RAM is measured with psutil only for a loopback endpoint outside a
    container (``in_container`` defaults to :func:`running_in_container`); a
    container or any other endpoint needs a ``host_snapshot`` taken on the model
    host, no older than ``max_snapshot_age`` (``host_memory_stale`` otherwise).
    Unknown host memory BLOCKS while a RAM requirement is declared: R2.3 says
    the model is not loaded until the requirement is shown to be met, and an
    under-provisioned load is exactly the opaque Ollama HTTP 500 this prevents.
    """
    profile = load_profile(profile)
    environment = os.environ if environment is None else environment
    if not isinstance(environment, Mapping):
        raise TypeError("environment must be a mapping")
    if not isinstance(max_snapshot_age, timedelta) or max_snapshot_age <= timedelta(0):
        raise ValueError("max_snapshot_age must be a positive timedelta")
    model = profile.model_config.model
    requirements = profile.host_requirements
    fallbacks = requirements.fallback_profile_ids
    base = {"profile_id": profile.profile_id, "model": model, "fallback_profile_ids": fallbacks,
            "required_free_ram_bytes": requirements.min_free_ram_bytes}
    if message := _credential_failure(profile, environment):
        return HostPreflightReport(passed=False, reason="missing_credential", recommendation=message, **base)
    if profile.response_mapping == ResponseMapping.HARMONY_VLLM and not profile.harmony.resolve_path().is_file():
        return HostPreflightReport(
            passed=False, reason="missing_tokenizer", **base,
            recommendation=(f"Profile {profile.profile_id!r} requires the pinned tokenizer at "
                            f"{profile.harmony.tokenizer_path} (revision {profile.harmony.tokenizer_revision}). "
                            "Download that pinned tokenizer.json before connecting; nothing was sent."))
    if profile.hosting_mode != HostingMode.LOCAL:
        return HostPreflightReport(passed=True, **base)
    endpoint = _resolved_config(profile, environment).endpoint
    headers = metadata_headers(profile, environment)
    memory: dict = {}
    if requirements.min_free_ram_bytes is not None:
        container = running_in_container() if in_container is None else bool(in_container)
        measured, total, source, refusal = _memory_gate(
            endpoint, host_snapshot, memory_reader if host_snapshot is None else None,
            in_container=container, now=now, max_age=max_snapshot_age)
        memory = {"measured_free_ram_bytes": measured, "total_ram_bytes": total, "memory_source": source}
        if refusal is not None:
            reason, explanation = refusal
            options = f" Or use {_fallback_text(fallbacks)}." if fallbacks else ""
            if explanation is None:
                explanation = _unknown_memory_text(endpoint, model, requirements.min_free_ram_bytes, container)
            else:
                explanation += (f" {model} needs {_gib(requirements.min_free_ram_bytes)} free; nothing was "
                                f"loaded. {_SNAPSHOT_STEP}")
            return HostPreflightReport(passed=False, reason=reason, **base, **memory,
                                       recommendation=explanation + options)
        memory["ram_gate"] = "enforced"
        if measured < requirements.min_free_ram_bytes:
            # A model that is already resident occupies the RAM being measured;
            # one read-only /api/ps call to that same host avoids a false block.
            if (profile.model_config.provider in _PS_PROVIDERS
                    and (host_snapshot is not None or _is_loopback(endpoint))
                    and _model_already_loaded(endpoint, model, http_get, headers)):
                memory["ram_gate"] = "skipped_model_already_loaded"
            else:
                return HostPreflightReport(
                    passed=False, reason="insufficient_free_ram", **base, **memory,
                    recommendation=_ram_recommendation(model, measured, requirements.min_free_ram_bytes,
                                                       fallbacks))
    if profile.model_config.provider == FLM_RUNTIME:
        return _fastflowlm_verdict(profile, endpoint, http_get, headers, base, memory)
    if profile.model_config.provider != "ollama":
        return HostPreflightReport(passed=True, **base, **memory)
    try:
        observed = observe_ollama_host(endpoint, model, http_get=http_get, headers=headers)
    except ValueError:
        return HostPreflightReport(
            passed=False, reason="host_metadata_unavailable", **base, **memory,
            recommendation=(f"Ollama metadata at {urlsplit(endpoint).netloc} did not answer. Start Ollama "
                            "(ollama serve) or check the endpoint; nothing was loaded."))
    host_fields = {"runtime_version": observed["runtime_version"], "artifact_digest": observed["digest"],
                   "model_installed": observed["model_installed"], "loaded_models": tuple(observed["loaded_models"]),
                   "observed_backend": observed["observed_backend"],
                   "effective_context_tokens": observed["effective_context_tokens"]}
    if not observed["model_installed"]:
        return HostPreflightReport(
            passed=False, reason="model_not_installed", **base, **memory, **host_fields,
            recommendation=(f"{model} is not installed on this Ollama host. Nothing is downloaded automatically; "
                            f"run `ollama pull {model}` yourself, then try again."))
    return HostPreflightReport(passed=True, **base, **memory, **host_fields)


def _fastflowlm_verdict(profile, endpoint, http_get, headers, base, memory) -> HostPreflightReport:
    """Observe branch for ``provider: fastflowlm`` (after the shared RAM gate)."""
    model = profile.model_config.model
    where = urlsplit(endpoint).netloc
    try:
        observed = observe_fastflowlm_host(endpoint, model, http_get=http_get, headers=headers)
    except ValueError:
        return HostPreflightReport(
            passed=False, reason="host_metadata_unavailable", **base, **memory,
            recommendation=(f"FastFlowLM metadata at {where} did not answer. On the Windows host run "
                            f"scripts/hosts/serve_local_flm.ps1 start {model}, or check the endpoint; nothing was loaded."))
    host_fields = {"runtime_version": observed["runtime_version"], "artifact_digest": observed["digest"],
                   "model_installed": observed["model_installed"], "loaded_models": tuple(observed["loaded_models"]),
                   "observed_backend": observed["observed_backend"],
                   "effective_context_tokens": observed["effective_context_tokens"]}

    def refuse(reason: str, text: str) -> HostPreflightReport:
        return HostPreflightReport(passed=False, reason=reason, **base, **memory, **host_fields, recommendation=text)

    if observed["model_installed"] is None:
        return refuse("installed_models_unknown",
                      f"{where} answered with FastFlowLM's catalog, which does not show installed models. Use the "
                      "FLM gateway (scripts/hosts/serve_local_flm.ps1), whose /api/tags lists installed models only.")
    if not observed["model_installed"]:
        return refuse("model_not_installed",
                      f"{model} is not installed for FastFlowLM on this host. Nothing is downloaded automatically; "
                      f"run `flm pull {model}` on the host yourself, then try again.")
    if not observed["npu_available"]:
        return refuse("npu_unavailable", "FastFlowLM reports the NPU unavailable (/api/npu/status). Run "
                      "`flm validate` on the host; nothing was sent.")
    identity = profile.model_config.local_identity
    if identity is not None:
        for name, seen in (("runtime_version", observed["runtime_version"]), ("artifact_digest", observed["digest"])):
            expected = getattr(identity.declared, name)
            if expected is not None and seen is not None and expected != seen:
                return refuse("declared_identity_mismatch",
                              f"The host reports {name} {seen}, but profile {profile.profile_id!r} declares "
                              f"{expected}. Install the declared build or use a matching profile; nothing was sent.")
    return HostPreflightReport(passed=True, **base, **memory, **host_fields)


def connect(
    profile: ProviderProfile | str | Path,
    budget: CallBudget,
    *,
    features: FeatureChoices | None = None,
    environment: Mapping[str, str] | None = None,
    enforce_preflight: bool = True,
    host_snapshot: HostMemorySnapshot | Mapping | None = None,
    max_snapshot_age: timedelta = SNAPSHOT_MAX_AGE,
) -> CompletionClient:
    """Build the configured client for ``profile`` without issuing any inference request.

    Dispatch is by ``(protocol, response_mapping)``. Credentials and the
    optional ``endpoint_ref`` are resolved from ``environment`` (default
    ``os.environ``); no env file is read. With ``enforce_preflight`` (default)
    :func:`preflight_host` runs first and a failure raises :class:`PreflightError`
    carrying the friendly proposal; for local Ollama profiles this reads host
    RAM and read-only ``/api/*`` metadata. Pass ``enforce_preflight=False`` only
    when the caller already ran its own host gate.
    """
    profile = load_profile(profile)
    environment = os.environ if environment is None else environment
    if not isinstance(environment, Mapping):
        raise TypeError("environment must be a mapping")
    if enforce_preflight:
        report = preflight_host(profile, environment=environment, host_snapshot=host_snapshot,
                                max_snapshot_age=max_snapshot_age)
        if not report.passed:
            raise PreflightError(report)
    reference = profile.model_config.credential_ref
    if reference is not None:
        value = environment.get(reference.removeprefix("env:"))
        if not isinstance(value, str) or not value.strip():
            raise ProfileError(f"Profile {profile.profile_id!r} needs credential "
                               f"{reference.removeprefix('env:')} in the environment")
    config = _resolved_config(profile, environment)
    protocol, mapping = config.protocol, profile.response_mapping
    if protocol == CompletionProtocol.OPENAI_CHAT and mapping == ResponseMapping.GENERIC:
        return OpenAICompatibleClient(config, budget, features=features, environment=environment,
                                      allow_unauthenticated=profile.allow_unauthenticated)
    if protocol == CompletionProtocol.OPENAI_CHAT and mapping == ResponseMapping.HARMONY_VLLM:
        return HarmonyOpenAICompatibleClient(config, budget, token_map=_token_map(profile), features=features,
                                             environment=environment,
                                             allow_unauthenticated=profile.allow_unauthenticated)
    if protocol == CompletionProtocol.ANTHROPIC_MESSAGES and mapping == ResponseMapping.GENERIC:
        return AnthropicClient(config, budget, features=features, environment=environment)
    raise ProfileError(f"No client for protocol {protocol.value!r} with mapping {mapping.value!r}")


__all__ = [
    "FULL_DISTRIBUTION_SOURCE", "TOP_K_SOURCE", "FullDistributionHostConfig",
    "HarmonyMappingConfig", "HostMemorySnapshot", "HostPreflightReport", "HostResourceRequirement", "HostingMode",
    "PreflightError", "ProfileError", "ProviderProfile", "ResponseMapping", "available_profiles", "connect",
    "load_profile", "metadata_headers", "observe_fastflowlm_host", "observe_ollama_host", "preflight_host",
    "running_in_container",
]
