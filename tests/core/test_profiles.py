"""Declarative provider profiles and the connect() factory: offline, no HTTP."""

import json
from dataclasses import replace
from datetime import UTC, datetime

import pytest
import requests

from recall_guard.core.completion import FeatureChoices, ModelConfig
from recall_guard.core.harmony_evidence import MARKERS
from recall_guard.core.profiles import (
    FULL_DISTRIBUTION_SOURCE,
    PROFILE_DIRECTORY,
    TOP_K_SOURCE,
    FullDistributionHostConfig,
    HarmonyMappingConfig,
    HostingMode,
    HostMemorySnapshot,
    HostResourceRequirement,
    PreflightError,
    ProfileError,
    ProviderProfile,
    ResponseMapping,
    available_profiles,
    connect,
    load_profile,
    preflight_host,
)
from recall_guard.core.provider_anthropic import AnthropicClient
from recall_guard.core.providers import HarmonyOpenAICompatibleClient, OpenAICompatibleClient
from recall_guard.core.request_control import CallBudget

#: Profiles without the full-distribution capability (older profiles and the Modal Gemma 3 4B serving arm);
#: they offer the top-k source only.
TOP_K_ONLY = ("ollama-gemma3-4b", "ollama-phi4-mini", "ollama-llama3.2-3b", "ollama-llama3.1-8b",
              "ollama-llava-llama3-ctx2048", "ollama-gpt-oss-20b", "modal-gpt-oss-20b", "nim-llama3.1-70b",
              "flm-llama3.2-3b", "flm-gemma3-4b", "flm-gemma4-e4b", "modal-gemma3-4b")
SHIPPED = (*TOP_K_ONLY, "llamacpp-gemma3-4b")
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
    llava = load_profile("ollama-llava-llama3-ctx2048")
    assert llava.model_config.model == "llava-llama3:ctx2048" and llava.host_requirements.context_tokens == 2048
    assert llava.host_requirements.min_free_ram_bytes == 6 * GIB and llava.response_mapping == ResponseMapping.GENERIC
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
    client = connect("ollama-gemma3-4b", CallBudget(1), environment={}, enforce_preflight=False)
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


# --- Host preflight (M1.4, R2.3): fakes only, no host memory or HTTP ----------

DIGEST = "b" * 64
MEASURED_LOW = int(13.4 * GIB)


class FakeOllama:
    def __init__(self, tags=("gpt-oss:20b",), running=()):
        self.calls = []
        self.tags = [{"name": name, "digest": DIGEST} for name in tags]
        self.running = list(running)

    def __call__(self, url):
        self.calls.append(url)
        path = url.split("11434", 1)[-1] if "11434" in url else url
        return {"/api/version": {"version": "0.12.3"}, "/api/tags": {"models": self.tags},
                "/api/ps": {"models": self.running}}[path]


def reader(available, total=32 * GIB):
    calls = []

    def read():
        calls.append(1)
        return available, total
    read.calls = calls
    return read


def gpt_oss(endpoint=None):
    profile = load_profile("ollama-gpt-oss-20b")
    if endpoint is None:
        return profile
    return replace(profile, model_config=replace(profile.model_config, endpoint=endpoint))


def test_insufficient_free_ram_blocks_with_zero_http_and_friendly_proposal():
    http = FakeOllama()
    report = preflight_host(gpt_oss(), environment={}, memory_reader=reader(MEASURED_LOW), http_get=http)
    assert not report.passed and report.reason == "insufficient_free_ram"
    assert http.calls == ["http://localhost:11434/api/ps"]  # Exactly one read-only call, no load.
    assert report.ram_gate == "enforced"
    assert report.measured_free_ram_bytes == MEASURED_LOW
    assert report.required_free_ram_bytes == 20 * GIB and report.total_ram_bytes == 32 * GIB
    assert report.memory_source == "local_host_available_ram"
    assert report.fallback_profile_ids == ("modal-gpt-oss-20b", "ollama-gemma3-4b")
    assert report.recommendation == (
        "13.4 GiB free, 20.0 GiB required for gpt-oss:20b. Nothing was loaded and no request was sent. "
        "Try --profile modal-gpt-oss-20b or --profile ollama-gemma3-4b, or run on a machine with at least "
        "20.0 GiB of free RAM.")


def test_connect_raises_preflight_error_before_any_request(monkeypatch):
    snapshot = HostMemorySnapshot(MEASURED_LOW, observed_at=datetime.now(UTC).isoformat())
    checked = []
    # The only permitted HTTP is the read-only /api/ps check on the snapshot's host; fake it as "not loaded".
    monkeypatch.setattr("recall_guard.core.profiles._model_already_loaded",
                        lambda endpoint, model, http_get=None, headers=None: checked.append(endpoint) or False)
    with pytest.raises(PreflightError) as caught:
        connect(gpt_oss("http://host.docker.internal:11434/v1/chat/completions"), CallBudget(1),
                environment={}, host_snapshot=snapshot)
    assert checked == ["http://host.docker.internal:11434/v1/chat/completions"]
    assert caught.value.report.reason == "insufficient_free_ram"
    assert "13.4 GiB free, 20.0 GiB required for gpt-oss:20b" in str(caught.value)
    assert isinstance(caught.value, ProfileError)


def test_sufficient_free_ram_passes_and_records_ollama_metadata():
    http = FakeOllama(running=[{"name": "gpt-oss:20b", "size": 14 * GIB, "size_vram": 14 * GIB,
                                "context_length": 2048}])
    report = preflight_host(gpt_oss(), environment={}, memory_reader=reader(24 * GIB), http_get=http)
    assert report.passed and report.reason is None
    assert [url.rsplit("/", 1)[-1] for url in http.calls] == ["version", "tags", "ps"]
    assert all(url.startswith("http://localhost:11434/api/") for url in http.calls)
    assert report.artifact_digest == "sha256:" + DIGEST and report.runtime_version == "0.12.3"
    assert report.model_installed and report.loaded_models == ("gpt-oss:20b",)
    assert report.observed_backend == "GPU" and report.effective_context_tokens == 2048
    assert report.to_dict()["passed"] is True


@pytest.mark.parametrize("size,vram,backend", [
    (10 * GIB, 0, "CPU"), (10 * GIB, 10 * GIB, "GPU"), (10 * GIB, 4 * GIB, "CPU/GPU offload"), (0, 0, None),
])
def test_api_ps_cpu_gpu_parsing(size, vram, backend):
    http = FakeOllama(running=[{"name": "gpt-oss:20b", "size": size, "size_vram": vram}])
    report = preflight_host(gpt_oss(), environment={}, memory_reader=reader(24 * GIB), http_get=http)
    assert report.observed_backend == backend


def test_other_loaded_model_is_recorded_without_its_name():
    http = FakeOllama(running=[{"name": "private-model:7b", "size": GIB, "size_vram": 0}])
    report = preflight_host(gpt_oss(), environment={}, memory_reader=reader(24 * GIB), http_get=http)
    assert report.loaded_models == ("unexpected_model",) and report.observed_backend is None


def test_non_loopback_endpoint_without_snapshot_is_unknown_and_never_reads_local_memory():
    local = reader(64 * GIB)
    http = FakeOllama()
    profile = gpt_oss("http://host.docker.internal:11434/v1/chat/completions")
    report = preflight_host(profile, environment={}, memory_reader=local, http_get=http)
    assert not report.passed and report.reason == "host_memory_unknown"
    assert local.calls == [] and http.calls == []
    assert report.measured_free_ram_bytes is None
    assert "host.docker.internal" in report.recommendation and "20.0 GiB" in report.recommendation
    snapshot = {"available_bytes": 24 * GIB, "observed_at": datetime.now(UTC).isoformat()}
    passed = preflight_host(profile, environment={}, host_snapshot=snapshot, memory_reader=local, http_get=http)
    assert passed.passed and passed.memory_source == "host_snapshot" and local.calls == []


def test_missing_tag_fails_with_pull_suggestion_and_never_pulls():
    http = FakeOllama(tags=("gemma3:4b",))
    report = preflight_host(gpt_oss(), environment={}, memory_reader=reader(24 * GIB), http_get=http)
    assert not report.passed and report.reason == "model_not_installed"
    assert "ollama pull gpt-oss:20b" in report.recommendation
    assert report.model_installed is False and report.artifact_digest is None
    assert not any("pull" in url for url in http.calls)


def test_unreachable_ollama_metadata_fails_clearly():
    def down(url):
        raise requests.ConnectionError("refused")
    report = preflight_host(gpt_oss(), environment={}, memory_reader=reader(24 * GIB), http_get=down)
    assert not report.passed and report.reason == "host_metadata_unavailable"


def test_missing_credential_is_reported_without_the_value():
    report = preflight_host("nim-llama3.1-70b", environment={})
    assert not report.passed and report.reason == "missing_credential"
    assert "NVIDIA_API_KEY" in report.recommendation
    assert preflight_host("nim-llama3.1-70b", environment={"NVIDIA_API_KEY": "nvapi-test"}).passed
    assert "nvapi-test" not in json.dumps(
        preflight_host("nim-llama3.1-70b", environment={"NVIDIA_API_KEY": "nvapi-test"}).to_dict())


def test_missing_pinned_tokenizer_fails_preflight(tokenizer, tmp_path):
    missing = harmony_profile(tokenizer, harmony=HarmonyMappingConfig(str(tmp_path / "absent.json"), REVISION))
    report = preflight_host(missing, environment=MODAL_ENV)
    assert not report.passed and report.reason == "missing_tokenizer"
    assert preflight_host(harmony_profile(tokenizer), environment=MODAL_ENV).passed


def test_profile_without_ram_requirement_skips_the_gate():
    profile = replace(gpt_oss(), host_requirements=HostResourceRequirement())
    local = reader(GIB)
    report = preflight_host(profile, environment={}, memory_reader=local, http_get=FakeOllama())
    assert report.passed and local.calls == [] and report.measured_free_ram_bytes is None


def test_ram_gate_passes_when_target_model_is_already_loaded():
    http = FakeOllama(running=[{"name": "gpt-oss:20b", "size": 14 * GIB, "size_vram": 0}])
    report = preflight_host(gpt_oss(), environment={}, memory_reader=reader(MEASURED_LOW), http_get=http)
    assert report.passed and report.ram_gate == "skipped_model_already_loaded"
    assert report.measured_free_ram_bytes == MEASURED_LOW and report.observed_backend == "CPU"
    assert http.calls[0] == "http://localhost:11434/api/ps"


def test_ram_gate_still_blocks_when_another_model_is_loaded():
    http = FakeOllama(running=[{"name": "gemma3:4b", "size": 4 * GIB, "size_vram": 0}])
    report = preflight_host(gpt_oss(), environment={}, memory_reader=reader(MEASURED_LOW), http_get=http)
    assert not report.passed and report.reason == "insufficient_free_ram"
    assert http.calls == ["http://localhost:11434/api/ps"]


def test_modal_profile_freezes_low_reasoning_effort_and_1024_output_tokens():
    parameters = load_profile("modal-gpt-oss-20b").model_config.parameters
    assert parameters["reasoning_effort"] == "low" and parameters["max_tokens"] == 1024
    assert parameters["return_token_ids"] is True and parameters["top_logprobs"] == 5
    assert (parameters["temperature"], parameters["top_p"]) == (0, 1)
    for profile_id in SHIPPED:
        if profile_id != "modal-gpt-oss-20b":
            assert "reasoning_effort" not in load_profile(profile_id).model_config.parameters


@pytest.mark.parametrize("value", ["minimal", "LOW", "", None, 1, True, ["low"]])
def test_reasoning_effort_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="reasoning_effort"):
        ModelConfig("openai_chat", "https://example.test/v1/chat/completions", "openai/gpt-oss-20b",
                    parameters={"reasoning_effort": value})


@pytest.mark.parametrize("value", ["low", "medium", "high"])
def test_reasoning_effort_accepts_vllm_values(value):
    config = ModelConfig("openai_chat", "https://example.test/v1/chat/completions", "openai/gpt-oss-20b",
                         parameters={"reasoning_effort": value})
    assert config.to_dict()["parameters"] == {"reasoning_effort": value}


def test_reasoning_effort_is_only_valid_with_the_harmony_mapping(tokenizer):
    gemma = load_profile("ollama-gemma3-4b")
    config = replace(gemma.model_config, parameters=dict(gemma.model_config.parameters) | {"reasoning_effort": "low"})
    with pytest.raises(ProfileError, match="reasoning_effort is only valid with the harmony_vllm mapping"):
        replace(gemma, model_config=config)
    with pytest.raises(ProfileError, match="harmony_vllm"):
        ProviderProfile.from_dict(gemma.to_dict() | {"model_config": config.to_dict()})
    harmony = harmony_profile(tokenizer)
    medium = replace(harmony.model_config,
                     parameters=dict(harmony.model_config.parameters) | {"reasoning_effort": "medium"})
    assert replace(harmony, model_config=medium).model_config.parameters["reasoning_effort"] == "medium"


def test_generic_clients_never_send_reasoning_effort(tokenizer):
    gemma = load_profile("ollama-gemma3-4b").model_config
    config = replace(gemma, parameters=dict(gemma.parameters) | {"reasoning_effort": "low"})
    with pytest.raises(ValueError, match="Harmony"):
        OpenAICompatibleClient(config, CallBudget(1), allow_unauthenticated=True)
    client = connect("ollama-gemma3-4b", CallBudget(1), environment={}, enforce_preflight=False)
    with pytest.raises(ValueError, match="Harmony"):
        client.generate("prompt", reasoning_effort="low")
    assert client.budget.used == 0
    anthropic = ModelConfig("anthropic_messages", "https://api.anthropic.com/v1/messages", "claude-test",
                            credential_ref="env:TEST_ANTHROPIC_KEY",
                            parameters={"max_tokens": 16, "reasoning_effort": "low"})
    with pytest.raises(ValueError, match="reasoning_effort"):
        AnthropicClient(anthropic, CallBudget(1), environment={"TEST_ANTHROPIC_KEY": "sk-test"})
    harmony = connect(harmony_profile(tokenizer), CallBudget(1), environment=MODAL_ENV)
    assert harmony.config.parameters["reasoning_effort"] == "low"


# --- Estimate sources and the llama.cpp full-distribution profile (Requirement 3.4) ---

LLAMACPP = "llamacpp-gemma3-4b"
GGUF_DIGEST = "sha256:f06e91e18fc74295d0c781bf91622508c75655e2c18ed869d52447ffe1e667e1"
#: Loopback ports fixed by task 3.3; each gateway listens on its loopback port plus one.
LLAMA_SERVER_PORT, FULL_DISTRIBUTION_PORT = 11570, 11580
#: Ollama, its gateway, FastFlowLM, its gateway and the reference llama-server.
RESERVED_PORTS = frozenset({11434, 11435, 11540, 11541, 11560})
STATE_FOLDERS = ("runs/finance-sim/hosts/llama-server", "runs/finance-sim/hosts/full-distribution")


def test_source_names_match_the_mia_estimate_source_values():
    from recall_guard.mia.estimate_source import EstimateSource

    assert (TOP_K_SOURCE, FULL_DISTRIBUTION_SOURCE) == (EstimateSource.TOP_K.value,
                                                        EstimateSource.FULL_DISTRIBUTION.value)


@pytest.mark.parametrize("profile_id", TOP_K_ONLY)
def test_existing_profiles_load_unchanged_with_the_top_k_source_only(profile_id):
    profile = load_profile(profile_id)
    assert profile.full_distribution is None
    assert profile.estimate_sources == ("top_k",)
    assert "full_distribution" not in profile.to_dict()
    profile.require_source("top_k")


def test_llamacpp_profile_offers_both_sources_on_one_gguf_digest():
    profile = load_profile(LLAMACPP)
    assert profile.estimate_sources == ("top_k", "full_distribution")
    host = profile.full_distribution
    assert isinstance(host, FullDistributionHostConfig)
    declared = profile.model_config.local_identity.declared
    assert host.model_digest == declared.artifact_digest == GGUF_DIGEST
    assert (declared.runtime, declared.runtime_version, declared.backend, declared.quantization) == (
        "llama.cpp", "b12199", "Vulkan (iGPU)", "Q4_K_M")
    assert host.top_k_keep == 100 and host.credential_ref == "env:LOCAL_MODEL_API_KEY"
    assert host.endpoint_ref == "env:FULL_DISTRIBUTION_ENDPOINT_URL"
    assert profile.model_config.credential_ref == "env:LOCAL_MODEL_API_KEY"
    assert profile.endpoint_ref == "env:LLAMACPP_ENDPOINT_URL" and not profile.allow_unauthenticated
    ollama = load_profile("ollama-gemma3-4b").model_config.parameters
    assert profile.model_config.parameters["top_logprobs"] == ollama["top_logprobs"]
    assert ProviderProfile.from_dict(profile.to_dict()) == profile


def test_full_distribution_request_is_refused_on_a_top_k_only_profile_and_accepted_on_llamacpp():
    with pytest.raises(ProfileError, match=r"'full_distribution'; available: top_k$"):
        load_profile("ollama-gemma3-4b").require_source("full_distribution")
    llamacpp = load_profile(LLAMACPP)
    llamacpp.require_source("full_distribution")
    llamacpp.require_source("top_k")
    with pytest.raises(ProfileError, match="available: top_k, full_distribution"):
        llamacpp.require_source("full_logits")


def test_modal_gemma_profile_is_generic_top_k_only_and_refuses_full_distribution():
    profile = load_profile("modal-gemma3-4b")
    assert profile.hosting_mode == HostingMode.REMOTE_GPU and profile.response_mapping == ResponseMapping.GENERIC
    assert profile.harmony is None and profile.full_distribution is None
    assert profile.estimate_sources == ("top_k",)
    with pytest.raises(ProfileError, match=r"'full_distribution'; available: top_k$"):
        profile.require_source("full_distribution")
    assert profile.model_config.credential_ref == "env:MODAL_PROXY_BEARER"
    assert profile.endpoint_ref == "env:MODAL_ENDPOINT_URL"
    assert profile.model_config.parameters["top_logprobs"] == 5 and profile.model_config.parameters["temperature"] == 0
    assert profile.model_config.local_identity.declared.artifact_revision == "093f9f388b31de276ce2de164bdc2081324b9767"
    assert "—" not in profile.notes  # No em dash.
    client = connect(profile, CallBudget(1), environment=MODAL_ENV)
    assert isinstance(client, OpenAICompatibleClient) and not isinstance(client, HarmonyOpenAICompatibleClient)
    assert client.budget.used == 0
    with pytest.raises(ProfileError, match="placeholder endpoint"):
        connect(profile, CallBudget(1), environment={"MODAL_PROXY_BEARER": "test-bearer"})


def test_profile_without_logprobs_offers_no_source():
    config = ModelConfig("anthropic_messages", "https://api.anthropic.com/v1/messages", "claude-test",
                         credential_ref="env:TEST_ANTHROPIC_KEY", parameters={"max_tokens": 16})
    profile = ProviderProfile("anthropic-test", "Anthropic test", HostingMode.MANAGED_API, config)
    assert profile.estimate_sources == ()
    with pytest.raises(ProfileError, match="available: none"):
        profile.require_source("top_k")


def _write_profile(tmp_path, data):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_digest_mismatch_between_sources_is_refused_at_load(tmp_path):
    data = json.loads((PROFILE_DIRECTORY / f"{LLAMACPP}.json").read_text(encoding="utf-8"))
    data["full_distribution"]["model_digest"] = "sha256:" + "0" * 64
    with pytest.raises(ProfileError, match="differs from the declared artifact_digest"):
        load_profile(_write_profile(tmp_path, data))
    data["full_distribution"]["model_digest"] = GGUF_DIGEST
    data["model_config"]["local_identity"]["declared"]["artifact_digest"] = None
    with pytest.raises(ProfileError, match="artifact_digest None"):
        load_profile(_write_profile(tmp_path, data))
    gemma = load_profile("ollama-gemma3-4b")
    with pytest.raises(ProfileError, match="artifact_digest"):
        replace(gemma, full_distribution=FullDistributionHostConfig("http://127.0.0.1:11580", GGUF_DIGEST,
                                                                    "env:LOCAL_MODEL_API_KEY"))


@pytest.mark.parametrize("change", [
    {"endpoint": "http://host.docker.internal:11581"}, {"endpoint": "http://user:pw@127.0.0.1:11580"},
    {"endpoint": "http://127.0.0.1:11580?x=1"}, {"model_digest": "f06e" * 16}, {"credential_ref": "Bearer abc"},
    {"endpoint_ref": "https://example.test"}, {"top_k_keep": 0}, {"top_k_keep": True}, {"top_k_keep": 100.0},
    {"api_key": "x"},
])
def test_invalid_full_distribution_host_config_is_refused(tmp_path, change):
    data = json.loads((PROFILE_DIRECTORY / f"{LLAMACPP}.json").read_text(encoding="utf-8"))
    data["full_distribution"] |= change
    with pytest.raises(ProfileError) as caught:
        load_profile(_write_profile(tmp_path, data))
    assert "pw" not in str(caught.value) and "Bearer" not in str(caught.value)


def gateway_port(loopback_port: int) -> int:
    """Repository convention: a bearer gateway listens on its host's loopback port plus one."""
    return loopback_port + 1


def test_llamacpp_ports_follow_the_gateway_rule_and_do_not_collide():
    from urllib.parse import urlsplit

    profile = load_profile(LLAMACPP)
    generation = urlsplit(profile.model_config.endpoint)
    host = urlsplit(profile.full_distribution.endpoint)
    assert (generation.hostname, generation.port) == ("127.0.0.1", LLAMA_SERVER_PORT)
    assert (host.hostname, host.port) == ("127.0.0.1", FULL_DISTRIBUTION_PORT)
    assert (gateway_port(generation.port), gateway_port(host.port)) == (11571, 11581)
    # The rule holds for the existing local hosts too (Ollama 11434/11435, FastFlowLM 11540/11541).
    assert (gateway_port(11434), gateway_port(11540)) == (11435, 11541)
    ports = {generation.port, gateway_port(generation.port), host.port, gateway_port(host.port)}
    assert len(ports) == 4 and not ports & RESERVED_PORTS


def test_llamacpp_ram_refusal_proposes_modal_and_notes_cover_the_second_load():
    profile = load_profile(LLAMACPP)
    # Modal Gemma 3 4B serves the same model family first; GPT-OSS-20B stays the second fallback.
    assert profile.host_requirements.fallback_profile_ids == ("modal-gemma3-4b", "modal-gpt-oss-20b")
    report = preflight_host(profile, environment={"LOCAL_MODEL_API_KEY": "test-key"}, in_container=False,
                            memory_reader=reader(2 * GIB), http_get=FakeOllama())
    assert not report.passed and report.reason == "insufficient_free_ram"
    assert "--profile modal-gemma3-4b or --profile modal-gpt-oss-20b" in report.recommendation
    assert "test-key" not in report.recommendation
    assert "generation process only" in profile.notes


def test_llamacpp_notes_record_state_folders_and_pins():
    profile = load_profile(LLAMACPP)
    for folder in STATE_FOLDERS:
        assert folder in profile.notes
    assert GGUF_DIGEST in profile.notes and "b12199" in profile.notes and "\u2014" not in profile.notes  # No em dash.
