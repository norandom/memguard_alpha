"""Declarative provider profiles and the connect() factory: offline, no HTTP."""

import json
from dataclasses import replace

import pytest
import requests

from recall_guard.core.completion import FeatureChoices, ModelConfig
from recall_guard.core.harmony_evidence import MARKERS
from recall_guard.core.profiles import (
    PROFILE_DIRECTORY,
    HarmonyMappingConfig,
    HostingMode,
    HostResourceRequirement,
    ProfileError,
    ProviderProfile,
    ResponseMapping,
    available_profiles,
    connect,
    load_profile,
)
from recall_guard.core.provider_anthropic import AnthropicClient
from recall_guard.core.providers import HarmonyOpenAICompatibleClient, OpenAICompatibleClient
from recall_guard.core.request_control import CallBudget

SHIPPED = ("ollama-gemma3-4b", "ollama-phi4-mini", "ollama-llama3.2-3b", "ollama-llama3.1-8b",
           "ollama-gpt-oss-20b", "modal-gpt-oss-20b", "nim-llama3.1-70b")
GIB = 1024 ** 3
REVISION = "a" * 40
IDS = dict(zip(MARKERS, (200005, 200006, 200007, 200008, 200002, 200012), strict=True))


@pytest.fixture(autouse=True)
def no_http(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("connect() must not issue HTTP requests")

    monkeypatch.setattr(requests.Session, "request", forbidden)
    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", forbidden)


@pytest.fixture
def tokenizer(tmp_path):
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps({"added_tokens": [
        {"content": label, "id": token_id, "special": True} for label, token_id in IDS.items()
    ]}), encoding="utf-8")
    return path


def harmony_profile(tokenizer, **overrides):
    modal = load_profile("modal-gpt-oss-20b")
    overrides.setdefault("harmony", HarmonyMappingConfig(str(tokenizer), REVISION))
    return replace(modal, **overrides)


MODAL_ENV = {"MODAL_PROXY_BEARER": "test-bearer", "MODAL_ENDPOINT_URL": "https://example.modal.run/v1/chat/completions"}


def test_every_shipped_profile_round_trips_and_is_loadable_by_id_and_path():
    assert set(SHIPPED) <= set(available_profiles())
    for profile_id in SHIPPED:
        path = PROFILE_DIRECTORY / f"{profile_id}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        profile = load_profile(profile_id)
        assert profile.profile_id == profile_id
        assert load_profile(path) == profile == load_profile(str(path))
        assert ProviderProfile.from_dict(profile.to_dict()) == profile
        assert profile.to_dict() == data


def test_profile_table_values():
    gpt = load_profile("ollama-gpt-oss-20b")
    assert gpt.host_requirements.min_free_ram_bytes == 21_474_836_480
    assert gpt.response_mapping == ResponseMapping.GENERIC and gpt.allow_unauthenticated
    assert load_profile("ollama-gemma3-4b").host_requirements.min_free_ram_bytes == 6 * GIB
    assert load_profile("ollama-llama3.1-8b").host_requirements.min_free_ram_bytes == 8 * GIB
    modal = load_profile("modal-gpt-oss-20b")
    assert modal.hosting_mode == HostingMode.REMOTE_GPU and modal.response_mapping == ResponseMapping.HARMONY_VLLM
    assert modal.model_config.credential_ref == "env:MODAL_PROXY_BEARER"
    assert modal.harmony.tokenizer_revision == "6cee5e81ee83917806bbde320786a8fb61efebee"
    nim = load_profile("nim-llama3.1-70b")
    assert nim.model_config.model == "meta/llama-3.1-70b-instruct"
    assert nim.model_config.credential_ref == "env:NVIDIA_API_KEY"


def test_exports_never_contain_resolved_secrets():
    for profile_id in SHIPPED:
        text = json.dumps(load_profile(profile_id).to_dict())
        assert "test-bearer" not in text and "Bearer " not in text


def test_unknown_profile_id_lists_available_profiles():
    with pytest.raises(ProfileError, match="ollama-gemma3-4b"):
        load_profile("no-such-profile")


def test_unknown_fields_rejected():
    data = load_profile("ollama-gemma3-4b").to_dict() | {"api_key": "x"}
    with pytest.raises(ProfileError):
        ProviderProfile.from_dict(data)


def test_harmony_requires_return_token_ids_and_pinned_revision(tokenizer):
    profile = harmony_profile(tokenizer)
    parameters = dict(profile.model_config.parameters)
    parameters.pop("return_token_ids")
    with pytest.raises(ProfileError, match="return_token_ids"):
        replace(profile, model_config=replace(profile.model_config, parameters=parameters))
    with pytest.raises(ProfileError, match="revision"):
        HarmonyMappingConfig(str(tokenizer), "main")
    with pytest.raises(ProfileError, match="harmony"):
        replace(profile, harmony=None)


def test_harmony_metadata_requires_harmony_mapping(tokenizer):
    with pytest.raises(ProfileError):
        replace(load_profile("ollama-gemma3-4b"), harmony=HarmonyMappingConfig(str(tokenizer), REVISION))


def test_unauthenticated_requires_local_without_credential():
    nim = load_profile("nim-llama3.1-70b")
    with pytest.raises(ProfileError, match="local"):
        replace(nim, allow_unauthenticated=True)
    local = load_profile("ollama-gemma3-4b")
    with pytest.raises(ProfileError, match="local"):
        replace(local, model_config=replace(local.model_config, credential_ref="env:LOCAL_MODEL_API_KEY"))


def test_host_requirement_validation():
    with pytest.raises(ProfileError):
        HostResourceRequirement(min_free_ram_bytes=0)
    with pytest.raises(ProfileError):
        HostResourceRequirement(fallback_profile_ids=("Bad ID",))


def test_connect_generic_local_profile_by_id():
    client = connect("ollama-gemma3-4b", CallBudget(1), environment={})
    assert type(client) is OpenAICompatibleClient
    assert client.config == load_profile("ollama-gemma3-4b").model_config
    assert client.budget.used == 0


def test_connect_nim_profile_is_openai_compatible_and_requires_its_credential():
    with pytest.raises(ProfileError, match="NVIDIA_API_KEY"):
        connect("nim-llama3.1-70b", CallBudget(1), environment={})
    client = connect("nim-llama3.1-70b", CallBudget(1), features=FeatureChoices(mia_scoring=True),
                     environment={"NVIDIA_API_KEY": "nvapi-test"})
    assert type(client) is OpenAICompatibleClient
    assert "nvapi-test" not in repr(client.config)


def test_connect_harmony_profile_builds_token_map(tokenizer):
    client = connect(harmony_profile(tokenizer), CallBudget(1), environment=MODAL_ENV)
    assert type(client) is HarmonyOpenAICompatibleClient
    assert client.token_map.tokenizer_revision == REVISION
    assert client.config.endpoint == MODAL_ENV["MODAL_ENDPOINT_URL"]


def test_connect_modal_placeholder_endpoint_fails_clearly(tokenizer):
    environment = {"MODAL_PROXY_BEARER": "test-bearer"}
    with pytest.raises(ProfileError, match="MODAL_ENDPOINT_URL"):
        connect(harmony_profile(tokenizer), CallBudget(1), environment=environment)


def test_connect_modal_missing_credential_fails_before_tokenizer_or_endpoint(tokenizer):
    with pytest.raises(ProfileError, match="MODAL_PROXY_BEARER") as caught:
        connect(harmony_profile(tokenizer), CallBudget(1), environment={})
    assert "test-bearer" not in str(caught.value)


def test_connect_harmony_rejects_missing_or_mismatched_tokenizer(tokenizer, tmp_path):
    missing = harmony_profile(tokenizer, harmony=HarmonyMappingConfig(str(tmp_path / "absent.json"), REVISION))
    with pytest.raises(ProfileError, match="pinned tokenizer"):
        connect(missing, CallBudget(1), environment=MODAL_ENV)
    wrong = harmony_profile(tokenizer, harmony=HarmonyMappingConfig(str(tokenizer), REVISION, "0" * 64))
    with pytest.raises(ProfileError, match="SHA-256"):
        connect(wrong, CallBudget(1), environment=MODAL_ENV)


def test_connect_rejects_malformed_endpoint_reference(tokenizer):
    environment = MODAL_ENV | {"MODAL_ENDPOINT_URL": "https://user:pw@example.modal.run/v1/chat/completions"}
    with pytest.raises(ProfileError, match="MODAL_ENDPOINT_URL") as caught:
        connect(harmony_profile(tokenizer), CallBudget(1), environment=environment)
    assert "pw@" not in str(caught.value)


def test_connect_anthropic_profile():
    config = ModelConfig("anthropic_messages", "https://api.anthropic.com/v1/messages", "claude-test",
                         credential_ref="env:TEST_ANTHROPIC_KEY", provider="anthropic", parameters={"max_tokens": 16})
    profile = ProviderProfile("anthropic-test", "Anthropic test", HostingMode.MANAGED_API, config,
                              host_requirements=HostResourceRequirement(fallback_profile_ids=()))
    assert ProviderProfile.from_dict(profile.to_dict()) == profile
    client = connect(profile, CallBudget(1), environment={"TEST_ANTHROPIC_KEY": "sk-test"})
    assert type(client) is AnthropicClient
    with pytest.raises(ProfileError, match="TEST_ANTHROPIC_KEY"):
        connect(profile, CallBudget(1), environment={})
