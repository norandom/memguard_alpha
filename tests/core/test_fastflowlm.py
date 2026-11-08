"""FastFlowLM (NPU) provider branch, profiles and the logprob-mass tolerance: offline, no HTTP."""

import math
from dataclasses import replace
from datetime import UTC, datetime

import pytest
import requests

from recall_guard.core.completion import ModelResponse, TokenAlternative, TokenEvidence
from recall_guard.core.local_evidence import LocalEvidenceError, validate_local_evidence
from recall_guard.core.profiles import (
    FLM_INSTALLED_TAGS_SOURCE,
    ProfileError,
    ProviderProfile,
    load_profile,
    observe_fastflowlm_host,
    preflight_host,
)

GIB = 1024 ** 3
GATEWAY = "http://172.17.32.1:11541/v1/chat/completions"
FLM = {"flm-llama3.2-3b": ("llama3.2:3b", 4, "b654a76e"), "flm-gemma3-4b": ("gemma3:4b", 6, "06cb8f39"),
       "flm-gemma4-e4b": ("gemma4-it:e4b", 10, "356d155b")}


@pytest.fixture(autouse=True)
def no_http(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("no real HTTP in offline tests")

    monkeypatch.setattr(requests.Session, "request", forbidden)


class FakeFlm:
    """The FLM gateway's four metadata routes; ``tags`` is the synthesized answer by default."""

    def __init__(self, *, installed=("gemma3:4b", "llama3.2:3b"), loaded=("gemma3:4b",), synthesized=True,
                 npu=True, version="1.0.12", serving="gemma3:4b", digest="sha256:" + "06cb8f39d3f84850c3e1938de304"
                 "605cf842fddbb5fef946fa88140d811ff117"):
        models = [{"name": name, "model": name, "size": 4_000_000_000, "details": {"quantization_level": "Q4_1"},
                   "digest": digest if name == "gemma3:4b" else None} for name in installed]
        tags = {"models": models}
        if synthesized:
            tags |= {"source": FLM_INSTALLED_TAGS_SOURCE, "runtime": "fastflowlm",
                     "serving": {"model": serving, "context_length": 2048}}
        # FLM's /api/ps reports size but never size_vram.
        self.routes = {"/api/version": {"version": version}, "/api/tags": tags,
                       "/api/ps": {"models": [{"name": name, "size": 4_000_000_000} for name in loaded]},
                       "/api/npu/status": {"npu_available": npu, "active_requests": 0}}
        self.calls = []

    def __call__(self, url):
        path = "/" + url.split("/", 3)[3]
        self.calls.append(path)
        return self.routes[path]


def snapshot(available_gib):
    return {"available_bytes": int(available_gib * GIB), "total_bytes": 32 * GIB, "host": "win",
            "source": "windows_host_snapshot", "observed_at": datetime.now(UTC).isoformat()}


def gateway_profile(profile_id="flm-gemma3-4b"):
    profile = load_profile(profile_id)
    return replace(profile, model_config=replace(profile.model_config, endpoint=GATEWAY))


def check(fake, available_gib=20, profile_id="flm-gemma3-4b"):
    return preflight_host(gateway_profile(profile_id), environment={}, host_snapshot=snapshot(available_gib),
                          http_get=fake, in_container=True)


# --- profiles -------------------------------------------------------------------------------------


@pytest.mark.parametrize("profile_id", sorted(FLM))
def test_flm_profiles_declare_the_npu_build_and_phase_a_settings(profile_id):
    model, gib, digest = FLM[profile_id]
    profile = load_profile(profile_id)
    config = profile.model_config
    assert ProviderProfile.from_dict(profile.to_dict()) == profile
    assert (config.protocol, config.provider, config.model) == ("openai_chat", "fastflowlm", model)
    assert config.parameters == {"temperature": 0, "top_p": 1, "max_tokens": 32, "logprobs": True, "top_logprobs": 5}
    assert profile.host_requirements.context_tokens == 2048
    assert profile.host_requirements.min_free_ram_bytes == gib * GIB
    declared = config.local_identity.declared
    assert (declared.runtime, declared.runtime_version, declared.backend) == ("fastflowlm", "1.0.12", "NPU")
    assert (declared.driver_version, declared.quantization, declared.context_length) == (
        "32.0.20102.3930", "Q4_1", 2048)
    assert declared.artifact_digest.startswith("sha256:" + digest) and declared.artifact_revision.endswith("@main")
    assert "up to 100" in profile.notes


def test_only_gemma3_on_flm_relaxes_the_mass_check_and_others_export_unchanged():
    assert load_profile("flm-gemma3-4b").logprob_mass_tolerance == 5e-6
    assert load_profile("flm-gemma3-4b").to_dict()["logprob_mass_tolerance"] == 5e-6
    for profile_id in ("flm-llama3.2-3b", "flm-gemma4-e4b", "ollama-gemma3-4b"):
        assert load_profile(profile_id).logprob_mass_tolerance is None
        assert "logprob_mass_tolerance" not in load_profile(profile_id).to_dict()


@pytest.mark.parametrize("value", [-1e-6, 1e-3, True, "5e-6", math.nan])
def test_profile_mass_tolerance_is_bounded(value):
    data = load_profile("flm-gemma3-4b").to_dict() | {"logprob_mass_tolerance": value}
    with pytest.raises(ProfileError):
        ProviderProfile.from_dict(data)


# --- preflight branch -----------------------------------------------------------------------------


def test_preflight_passes_on_synthesized_tags_with_npu_backend_and_launch_context():
    fake = FakeFlm()
    report = check(fake)
    assert report.passed and report.reason is None and report.ram_gate == "enforced"
    assert report.runtime_version == "1.0.12" and report.model_installed is True
    assert report.observed_backend == "NPU" and report.effective_context_tokens == 2048
    assert report.artifact_digest == load_profile("flm-gemma3-4b").model_config.local_identity.declared.artifact_digest
    assert report.loaded_models == ("gemma3:4b",)
    assert sorted(set(fake.calls)) == ["/api/npu/status", "/api/ps", "/api/tags", "/api/version"]


def test_flm_catalog_tags_never_count_as_installed():
    # FLM's own /api/tags lists the whole catalog: without the gateway marker installation is unknown.
    report = check(FakeFlm(installed=("gemma3:4b", "qwen3:8b"), synthesized=False))
    assert not report.passed and report.reason == "installed_models_unknown" and report.model_installed is None
    assert "serve_local_flm.ps1" in report.recommendation


def test_model_missing_from_installed_list_is_refused_without_download():
    report = check(FakeFlm(installed=("llama3.2:3b",)))
    assert not report.passed and report.reason == "model_not_installed" and "flm pull gemma3:4b" in report.recommendation


def test_npu_unavailable_is_refused_and_backend_is_not_inferred_from_memory():
    report = check(FakeFlm(npu=False))
    assert not report.passed and report.reason == "npu_unavailable" and report.observed_backend is None


def test_declared_runtime_version_mismatch_is_refused():
    report = check(FakeFlm(version="1.0.13"))
    assert not report.passed and report.reason == "declared_identity_mismatch" and "1.0.13" in report.recommendation


def test_ram_gate_uses_the_host_snapshot_and_skips_only_when_flm_has_the_model_loaded():
    low = check(FakeFlm(loaded=()), available_gib=5)
    assert not low.passed and low.reason == "insufficient_free_ram"
    assert low.required_free_ram_bytes == 6 * GIB and low.measured_free_ram_bytes == 5 * GIB
    loaded = check(FakeFlm(loaded=("gemma3:4b",)), available_gib=5)
    assert loaded.passed and loaded.ram_gate == "skipped_model_already_loaded"
    assert not check(FakeFlm(installed=("gemma4-it:e4b",), loaded=(), serving="gemma4-it:e4b"),
                     available_gib=9, profile_id="flm-gemma4-e4b").passed


def test_observer_reports_context_only_for_the_served_model_and_flags_other_loaded_models():
    observed = observe_fastflowlm_host(GATEWAY, "gemma3:4b", http_get=FakeFlm(serving="llama3.2:3b",
                                                                               loaded=("llama3.2:3b",)))
    assert observed["effective_context_tokens"] is None and observed["loaded_models"] == ["unexpected_model"]
    assert observed["model_allocation_bytes"] is None and observed["accelerator_allocation_bytes"] is None


def test_unusable_metadata_is_host_metadata_unavailable():
    fake = FakeFlm()
    del fake.routes["/api/npu/status"]
    report = check(fake)
    assert not report.passed and report.reason == "host_metadata_unavailable"


# --- evidence validator mass tolerance -------------------------------------------------------------


def response(masses):
    tokens = []
    for mass in masses:
        alternatives = (TokenAlternative("OK", math.log(0.9 * mass)), TokenAlternative("O", math.log(0.1 * mass)))
        tokens.append(TokenEvidence("OK", alternatives[0].logprob, top_logprobs=alternatives))
    return ModelResponse("OK" * len(masses), token_evidence=tuple(tokens))


def test_mass_check_is_off_by_default_and_tolerates_gemma3_float32_rounding():
    over = response([1 + 1.8e-6])  # FLM gemma3:4b observation (R34).
    validate_local_evidence(over, required_alternatives=2)  # Default: unchanged, no mass check.
    validate_local_evidence(over, required_alternatives=2, logprob_mass_tolerance=5e-6)
    with pytest.raises(LocalEvidenceError, match="alternative_mass_exceeds_one"):
        validate_local_evidence(over, required_alternatives=2, logprob_mass_tolerance=1e-6)
    with pytest.raises(LocalEvidenceError, match="alternative_mass_exceeds_one"):
        validate_local_evidence(response([0.5, 1 + 1e-5]), required_alternatives=2, logprob_mass_tolerance=5e-6)


@pytest.mark.parametrize("value", [-1e-9, 1e-3, math.inf, True])
def test_mass_tolerance_argument_is_validated(value):
    with pytest.raises(ValueError, match="logprob_mass_tolerance"):
        validate_local_evidence(response([0.9]), logprob_mass_tolerance=value)


def test_mass_tolerance_does_not_relax_the_per_logprob_checks():
    token = TokenEvidence("OK", -1e-7, top_logprobs=(TokenAlternative("OK", -9999.0),))
    with pytest.raises(LocalEvidenceError, match="censored_logprob"):
        validate_local_evidence(ModelResponse("OK", token_evidence=(token,)), logprob_mass_tolerance=5e-6)
