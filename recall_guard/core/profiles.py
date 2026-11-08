"""Declarative provider profiles and the single client factory (release-1.0 R2.1-2.2).

A :class:`ProviderProfile` wraps an existing :class:`ModelConfig` and adds only
what the config cannot express: hosting mode, response mapping, pinned Harmony
tokenizer metadata, declared host requirements and the explicit unauthenticated
choice. :func:`connect` dispatches to the existing transport clients and never
issues an inference request. :func:`preflight_host` (R2.3) runs first by
default: credential/tokenizer presence, the free-RAM gate and read-only Ollama
metadata, returning a friendly per-host proposal instead of loading a model.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from urllib.parse import urlsplit

import requests

from recall_guard.core.completion import CompletionClient, CompletionProtocol, FeatureChoices, ModelConfig
from recall_guard.core.harmony_evidence import HarmonyTokenMap
from recall_guard.core.provider_anthropic import AnthropicClient
from recall_guard.core.providers import HarmonyOpenAICompatibleClient, OpenAICompatibleClient
from recall_guard.core.request_control import CallBudget

#: Repository root; curated profiles live in ``configs/providers/<id>.json``.
REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE_DIRECTORY = REPO_ROOT / "configs" / "providers"
#: Documented endpoint placeholder (docs/modal-provider.md); never contacted.
PLACEHOLDER_MARKER = "your-host"
_PROFILE_ID = re.compile(r"[a-z0-9][a-z0-9.\-]*")
_REVISION = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_ENV_REF = re.compile(r"(?:env:)?[A-Za-z_][A-Za-z0-9_]*")


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
    """Declarative pre-load host requirements; enforced by host preflight (M1.4)."""

    min_free_ram_bytes: int | None = None
    recommended_ram_bytes: int | None = None
    context_tokens: int = 2048
    fallback_profile_ids: tuple[str, ...] = ("modal-gpt-oss-20b", "ollama-gemma3-4b")

    def __post_init__(self) -> None:
        _optional_bytes(self.min_free_ram_bytes, "min_free_ram_bytes")
        _optional_bytes(self.recommended_ram_bytes, "recommended_ram_bytes")
        if type(self.context_tokens) is not int or self.context_tokens < 1:
            raise ProfileError("context_tokens must be a positive integer")
        object.__setattr__(self, "fallback_profile_ids", tuple(self.fallback_profile_ids))
        if not all(isinstance(item, str) and _PROFILE_ID.fullmatch(item) for item in self.fallback_profile_ids):
            raise ProfileError("fallback_profile_ids must contain profile IDs")

    def to_dict(self) -> dict:
        return {"min_free_ram_bytes": self.min_free_ram_bytes, "recommended_ram_bytes": self.recommended_ram_bytes,
                "context_tokens": self.context_tokens, "fallback_profile_ids": list(self.fallback_profile_ids)}


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


_PROFILE_FIELDS = frozenset({"schema_version", "profile_id", "display_name", "hosting_mode", "model_config",
                             "credential_ref", "endpoint_ref", "response_mapping", "harmony", "host_requirements",
                             "allow_unauthenticated", "notes"})


@dataclass(frozen=True)
class ProviderProfile:
    """Declarative inference profile wrapping :class:`ModelConfig` (R2.1).

    ``endpoint_ref`` optionally names an environment variable holding a
    user-specific endpoint (for example a registered Modal host). It is
    resolved only by :func:`connect`, like a credential reference.
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
        self._check_mapping()
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
        elif self.harmony is not None:
            raise ProfileError("harmony metadata is only valid with the harmony_vllm mapping")

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
            return cls(
                profile_id=data["profile_id"], display_name=data["display_name"],
                hosting_mode=data["hosting_mode"], model_config=config,
                response_mapping=data.get("response_mapping", ResponseMapping.GENERIC.value),
                harmony=None if harmony is None else HarmonyMappingConfig(**harmony),
                host_requirements=HostResourceRequirement(**requirements),
                allow_unauthenticated=data.get("allow_unauthenticated", False),
                endpoint_ref=data.get("endpoint_ref"), notes=data.get("notes", ""),
            )
        except ProfileError:
            raise
        except (KeyError, TypeError, ValueError) as error:
            raise ProfileError(f"Invalid provider profile: {error}") from None


def load_profile(id_or_path: ProviderProfile | str | Path) -> ProviderProfile:
    """Load a curated profile by ID (``configs/providers/<id>.json``) or by path."""
    if isinstance(id_or_path, ProviderProfile):
        return id_or_path
    if isinstance(id_or_path, str) and _PROFILE_ID.fullmatch(id_or_path) and not id_or_path.endswith(".json"):
        path = PROFILE_DIRECTORY / f"{id_or_path}.json"
        if not path.is_file():
            known = ", ".join(sorted(available_profiles())) or "none"
            raise ProfileError(f"Unknown provider profile {id_or_path!r}; available: {known}")
    elif isinstance(id_or_path, (str, Path)):
        path = Path(id_or_path)
    else:
        raise TypeError("profile must be a ProviderProfile, profile ID or path")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProfileError(f"Cannot read provider profile {path.name}: {error.__class__.__name__}") from None
    return ProviderProfile.from_dict(data)


def available_profiles() -> tuple[str, ...]:
    return tuple(sorted(path.stem for path in PROFILE_DIRECTORY.glob("*.json")))


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
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")


@dataclass(frozen=True)
class HostMemorySnapshot:
    """Free RAM measured ON the model host (for example by a host-side helper).

    Required when the endpoint is not loopback: a container or client machine
    must never substitute its own memory for the external Ollama host's.
    """

    available_bytes: int
    total_bytes: int | None = None
    sampled_at: str | None = None
    source: str = "host_snapshot"

    def __post_init__(self) -> None:
        if type(self.available_bytes) is not int or self.available_bytes < 0:
            raise ProfileError("host_snapshot.available_bytes must be a nonnegative integer")
        if self.total_bytes is not None and (type(self.total_bytes) is not int or self.total_bytes <= 0):
            raise ProfileError("host_snapshot.total_bytes must be a positive integer or null")

    @classmethod
    def coerce(cls, value: HostMemorySnapshot | Mapping) -> HostMemorySnapshot:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ProfileError("host_snapshot must be a HostMemorySnapshot or mapping")
        allowed = {"available_bytes", "total_bytes", "sampled_at", "source"}
        return cls(**{key: value[key] for key in allowed if key in value})


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


def observe_ollama_host(endpoint: str, model: str, *, http_get=None) -> dict:
    """Read-only Ollama metadata (``/api/version``, ``/api/tags``, ``/api/ps``).

    Never generates, pulls, loads or unloads. ``http_get(url) -> object`` returns
    parsed JSON or raises; the default uses a proxy-free ``requests`` session.
    Raises ``ValueError('host_metadata_unavailable')`` on any unusable answer.
    """
    parsed = urlsplit(endpoint)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    try:
        if http_get is None:
            with requests.Session() as session:
                session.trust_env = False

                def fetch(path: str):
                    result = session.get(origin + path, timeout=10, allow_redirects=False)
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


def _model_already_loaded(endpoint: str, model: str, http_get=None) -> bool:
    """One read-only ``/api/ps`` call; any failure counts as not loaded."""
    parsed = urlsplit(endpoint)
    url = f"{parsed.scheme}://{parsed.netloc}/api/ps"
    try:
        if http_get is None:
            with requests.Session() as session:
                session.trust_env = False
                result = session.get(url, timeout=10, allow_redirects=False)
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


def _memory_gate(endpoint, host_snapshot, memory_reader):
    """Return ``(measured, total, source)``; ``measured is None`` when unknown."""
    if host_snapshot is not None:
        snapshot = HostMemorySnapshot.coerce(host_snapshot)
        return snapshot.available_bytes, snapshot.total_bytes, snapshot.source
    if (urlsplit(endpoint or "").hostname or "").lower() not in LOOPBACK_HOSTS:
        return None, None, None
    try:
        available, total = (memory_reader or _psutil_memory)()
    except (OSError, RuntimeError, ValueError, TypeError):
        return None, None, None
    if type(available) is not int or available < 0:
        return None, None, None
    return available, total if type(total) is int and total > 0 else None, "local_host_available_ram"


def preflight_host(
    profile: ProviderProfile | str | Path,
    *,
    environment: Mapping[str, str] | None = None,
    host_snapshot: HostMemorySnapshot | Mapping | None = None,
    memory_reader=None,
    http_get=None,
) -> HostPreflightReport:
    """Zero-inference host checks before any client is built or model loaded.

    Order: credential and pinned tokenizer presence (no I/O), then the free-RAM
    gate, then read-only Ollama metadata. A failed gate stops there; the only
    HTTP before an ``insufficient_free_ram`` verdict is one loopback ``/api/ps``
    read, which passes the gate when the target model is already loaded.
    Free RAM is measured with psutil only for a loopback endpoint; any other
    endpoint needs an explicit ``host_snapshot`` taken on the model host.
    Unknown host memory BLOCKS while a RAM requirement is declared: R2.3 says
    the model is not loaded until the requirement is shown to be met, and an
    under-provisioned load is exactly the opaque Ollama HTTP 500 this prevents.
    """
    profile = load_profile(profile)
    environment = os.environ if environment is None else environment
    if not isinstance(environment, Mapping):
        raise TypeError("environment must be a mapping")
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
    memory: dict = {}
    if requirements.min_free_ram_bytes is not None:
        measured, total, source = _memory_gate(endpoint, host_snapshot, memory_reader)
        memory = {"measured_free_ram_bytes": measured, "total_ram_bytes": total, "memory_source": source}
        if measured is None:
            host = urlsplit(endpoint or "").hostname or "the model host"
            options = f" Or use {_fallback_text(fallbacks)}." if fallbacks else ""
            return HostPreflightReport(
                passed=False, reason="host_memory_unknown", **base, **memory,
                recommendation=(f"Free RAM on {host} could not be measured from here, and {model} needs "
                                f"{_gib(requirements.min_free_ram_bytes)} free. Nothing was loaded. Pass a "
                                f"host memory snapshot taken on {host}, or run on the model host itself.{options}"))
        memory["ram_gate"] = "enforced"
        if measured < requirements.min_free_ram_bytes:
            # A model that is already resident occupies the RAM being measured;
            # one read-only /api/ps call (loopback Ollama only) avoids a false block.
            loopback = (urlsplit(endpoint or "").hostname or "").lower() in LOOPBACK_HOSTS
            if (loopback and profile.model_config.provider == "ollama"
                    and _model_already_loaded(endpoint, model, http_get)):
                memory["ram_gate"] = "skipped_model_already_loaded"
            else:
                return HostPreflightReport(
                    passed=False, reason="insufficient_free_ram", **base, **memory,
                    recommendation=_ram_recommendation(model, measured, requirements.min_free_ram_bytes,
                                                       fallbacks))
    if profile.model_config.provider != "ollama":
        return HostPreflightReport(passed=True, **base, **memory)
    try:
        observed = observe_ollama_host(endpoint, model, http_get=http_get)
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


def connect(
    profile: ProviderProfile | str | Path,
    budget: CallBudget,
    *,
    features: FeatureChoices | None = None,
    environment: Mapping[str, str] | None = None,
    enforce_preflight: bool = True,
    host_snapshot: HostMemorySnapshot | Mapping | None = None,
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
        report = preflight_host(profile, environment=environment, host_snapshot=host_snapshot)
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
    "HarmonyMappingConfig", "HostMemorySnapshot", "HostPreflightReport", "HostResourceRequirement", "HostingMode",
    "PreflightError", "ProfileError", "ProviderProfile", "ResponseMapping", "available_profiles", "connect",
    "load_profile", "observe_ollama_host", "preflight_host",
]
