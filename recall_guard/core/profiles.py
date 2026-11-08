"""Declarative provider profiles and the single client factory (release-1.0 R2.1-2.2).

A :class:`ProviderProfile` wraps an existing :class:`ModelConfig` and adds only
what the config cannot express: hosting mode, response mapping, pinned Harmony
tokenizer metadata, declared host requirements and the explicit unauthenticated
choice. :func:`connect` dispatches to the existing transport clients; it never
issues a request. Host preflight (free RAM, Ollama metadata) is a separate,
later gate (M1.4) and is intentionally not performed here.
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


def connect(
    profile: ProviderProfile | str | Path,
    budget: CallBudget,
    *,
    features: FeatureChoices | None = None,
    environment: Mapping[str, str] | None = None,
) -> CompletionClient:
    """Build the configured client for ``profile`` without issuing any request.

    Dispatch is by ``(protocol, response_mapping)``. Credentials and the
    optional ``endpoint_ref`` are resolved from ``environment`` (default
    ``os.environ``); no env file is read. Host preflight is not run here yet:
    the free-RAM/metadata gate (M1.4) is added in front of client construction.
    """
    profile = load_profile(profile)
    environment = os.environ if environment is None else environment
    if not isinstance(environment, Mapping):
        raise TypeError("environment must be a mapping")
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
    "HarmonyMappingConfig", "HostResourceRequirement", "HostingMode", "ProfileError", "ProviderProfile",
    "ResponseMapping", "available_profiles", "connect", "load_profile",
]
