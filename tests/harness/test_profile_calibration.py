"""Calibration identity v2: calibrators never cross evidence profiles (R2.5)."""

import dataclasses
import json
from io import StringIO
from types import SimpleNamespace

import pytest
import requests
from rich.console import Console

from recall_guard.core.completion import FeatureChoices, ModelConfig
from recall_guard.core.harmony_evidence import MARKERS, HarmonyTokenMap
from recall_guard.core.local_evidence import LocalModelIdentity
from recall_guard.core.profiles import HostPreflightReport, PreflightError, load_profile, preflight_host
from recall_guard.core.providers import HarmonyOpenAICompatibleClient, OpenAICompatibleClient
from recall_guard.core.request_control import CallBudget, RequestError
from recall_guard.harness import scorer as scorer_module
from recall_guard.harness.request_failures import (
    describe_failure,
    render_failure_panel,
    request_failure,
    retained_detail,
)
from recall_guard.harness.scorer import MemoryGuardedScorer
from recall_guard.harness.scorer_client import CalibrationIdentity, LegacyLogprobAdapter
from tests.harness.test_scorer_client import IS, OOS, SETTINGS, Client, calibrate

IDS = dict(zip(MARKERS, (200005, 200006, 200007, 200008, 200002, 200012), strict=True))
HARMONY_SETTINGS = SETTINGS | {"return_token_ids": True}


@pytest.fixture
def token_map(tmp_path):
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps({"added_tokens": [
        {"content": label, "id": token_id, "special": True} for label, token_id in IDS.items()
    ]}), encoding="utf-8")
    return HarmonyTokenMap.from_tokenizer_json(path, tokenizer_revision="b" * 40)


def gpt_config():
    return ModelConfig(protocol="openai_chat", endpoint="https://example.test/v1/chat/completions",
                       model="openai/gpt-oss-20b", parameters=HARMONY_SETTINGS, credential_ref="env:TEST_KEY")


class HarmonyDouble(Client, HarmonyOpenAICompatibleClient):
    """Deterministic double that is a Harmony client for identity purposes; no HTTP."""

    def __init__(self, token_map):
        Client.__init__(self, gpt_config())
        self.token_map = token_map


def real_clients(token_map):
    environment = {"TEST_KEY": "secret"}
    harmony = HarmonyOpenAICompatibleClient(gpt_config(), CallBudget(0), token_map=token_map, environment=environment)
    generic = OpenAICompatibleClient(gpt_config(), CallBudget(0), environment=environment)
    return harmony, generic


def test_harmony_client_identity_records_mapping_and_tokenizer_sha(token_map):
    harmony, generic = real_clients(token_map)
    identity = CalibrationIdentity.from_client(harmony, "original")
    assert identity.response_mapping == "harmony_vllm"
    assert identity.mapping_revision == token_map.tokenizer_sha256
    assert identity.to_dict()["response_mapping"] == "harmony_vllm"
    assert CalibrationIdentity.from_dict(identity.to_dict()) == identity
    plain = CalibrationIdentity.from_client(generic, "original")
    assert plain.response_mapping == "generic" and plain.mapping_revision is None
    assert plain != identity


def test_adapter_rejects_generic_client_for_harmony_calibrator(token_map):
    harmony, generic = real_clients(token_map)
    adapter = LegacyLogprobAdapter(harmony, "original", lambda error: False)
    adapter.validate_client(harmony, intervention_id="original")
    with pytest.raises(ValueError, match="response mapping"):
        adapter.validate_client(generic, intervention_id="original")
    reverse = LegacyLogprobAdapter(generic, "original", lambda error: False)
    with pytest.raises(ValueError, match="response mapping"):
        reverse.validate_client(harmony, intervention_id="original")


def test_harmony_fit_cannot_score_with_generic_client_and_vice_versa(token_map):
    harmony_fit = calibrate(HarmonyDouble(token_map))
    assert harmony_fit.calibration_identity.response_mapping == "harmony_vllm"
    with pytest.raises(ValueError, match="response mapping"):
        harmony_fit.validate_client(Client(gpt_config()))
    generic_fit = calibrate(Client(gpt_config()))
    with pytest.raises(ValueError, match="response mapping"):
        generic_fit.validate_client(HarmonyDouble(token_map))
    assert harmony_fit.calibration_fingerprint != generic_fit.calibration_fingerprint


def test_mapping_revision_change_is_rejected(token_map):
    fit = calibrate(HarmonyDouble(token_map))
    other = HarmonyDouble(dataclasses.replace(token_map, tokenizer_sha256="c" * 64))
    with pytest.raises(ValueError, match="response mapping"):
        fit.validate_client(other)


def test_legacy_v1_identity_dicts_still_load_and_export_unchanged():
    legacy = {"protocol": "openai_chat", "endpoint": "https://example.test/chat", "model": "m",
              "parameters_json": json.dumps(SETTINGS, sort_keys=True, separators=(",", ":")),
              "intervention_id": "original"}
    identity = CalibrationIdentity.from_dict(legacy)
    assert identity.response_mapping == "generic" and identity.mapping_revision is None
    assert identity.to_dict() == legacy
    assert identity == calibrate().calibration_identity


def test_stored_asdict_identity_with_local_identity_loads():
    identity = CalibrationIdentity.from_config(
        dataclasses.replace(gpt_config(), local_identity=LocalModelIdentity()), "original")
    stored = dataclasses.asdict(identity)  # shape persisted in workflow/script provenance
    assert CalibrationIdentity.from_dict(stored) == identity
    assert CalibrationIdentity.from_dict(identity.to_dict()) == identity


def test_identity_validation_rejects_unknown_mapping_or_missing_revision():
    base = dict(protocol="openai_chat", endpoint=None, model="m", parameters_json="{}", intervention_id="x")
    with pytest.raises(ValueError):
        CalibrationIdentity(**base, response_mapping="other")
    with pytest.raises(ValueError, match="mapping_revision"):
        CalibrationIdentity(**base, response_mapping="harmony_vllm")
    with pytest.raises(ValueError):
        CalibrationIdentity.from_dict(base | {"api_key": "x"})


def test_from_profile_connects_with_mia_features_and_calibrates(monkeypatch):
    client = Client()
    seen = {}

    def fake_connect(profile, budget, *, features=None, environment=None):
        seen.update(profile=profile, budget=budget, features=features, environment=environment)
        return client

    monkeypatch.setattr(scorer_module, "connect", fake_connect)
    budget = SimpleNamespace()
    scorer, returned = MemoryGuardedScorer.from_profile(
        "ollama-gemma3-4b", budget, is_memorized=IS, oos_control=OOS, environment={}, min_valid=4, max_workers=1)
    assert returned is client
    assert seen == {"profile": "ollama-gemma3-4b", "budget": budget,
                    "features": FeatureChoices(mia_scoring=True), "environment": {}}
    assert client.calls[0][0] == "probe"
    assert scorer.calibration_identity == CalibrationIdentity.from_client(client, "original")


# --- Friendly failure presentation and error-body retention (M1.4, R2.7) ------

SECRET = "synthetic-modal-bearer-0123456789"


def panel_text(presentation):
    stream = StringIO()
    Console(file=stream, width=200, color_system=None).print(render_failure_panel(presentation))
    return stream.getvalue()


def ram_report():
    return HostPreflightReport(
        "ollama-gpt-oss-20b", "gpt-oss:20b", passed=False, reason="insufficient_free_ram",
        recommendation="13.4 GiB free, 20.0 GiB required for gpt-oss:20b. Nothing was loaded.",
        measured_free_ram_bytes=int(13.4 * 1024 ** 3), required_free_ram_bytes=20 * 1024 ** 3,
        fallback_profile_ids=("modal-gpt-oss-20b", "ollama-gemma3-4b"))


def test_describe_insufficient_free_ram_proposes_modal_smaller_profile_or_larger_host():
    presentation = describe_failure(PreflightError(ram_report()), stage="host preflight")
    assert presentation.kind == "insufficient_free_ram"
    assert presentation.profile_id == "ollama-gpt-oss-20b"
    assert presentation.next_steps == ("Use --profile modal-gpt-oss-20b", "Use --profile ollama-gemma3-4b",
                                       "Run on a machine with at least 20.0 GiB of free RAM")
    text = panel_text(presentation)
    assert "Not enough free memory" in text and "13.4 GiB free, 20.0 GiB required for gpt-oss:20b" in text
    assert "ollama-gpt-oss-20b | host preflight" in text


@pytest.mark.parametrize("reason", ["insufficient_free_ram", "host_memory_unknown"])
def test_panel_lists_each_alternative_once_under_what_you_can_do(reason):
    """M1.5: the real preflight recommendation names the fallbacks; the panel shows them once."""
    if reason == "insufficient_free_ram":
        report = preflight_host("ollama-gpt-oss-20b", environment={},
                                memory_reader=lambda: (int(13.4 * 1024 ** 3), 32 * 1024 ** 3),
                                http_get=lambda url: {"models": []})
    else:
        profile = load_profile("ollama-gpt-oss-20b")
        remote = dataclasses.replace(profile.model_config, endpoint="http://model-host.test:11434/v1/chat/completions")
        report = preflight_host(dataclasses.replace(profile, model_config=remote), environment={})
    assert report.reason == reason and "--profile modal-gpt-oss-20b" in report.recommendation
    presentation = describe_failure(PreflightError(report))
    text = panel_text(presentation)
    for fallback in report.fallback_profile_ids:
        assert text.count(f"--profile {fallback}") == 1
    assert text.index("What you can do:") < text.index("--profile modal-gpt-oss-20b")
    assert "Nothing was loaded" in presentation.explanation and "--profile" not in presentation.explanation


@pytest.mark.parametrize("status,code,kind", [
    (401, None, "authentication_error"), (403, None, "authentication_error"),
    (402, None, "eligibility_error"), (400, "insufficient_quota", "eligibility_error"),
    (429, None, "rate_limited"), (503, None, "overloaded"), (529, None, "overloaded"),
    (504, None, "timeout"), (None, "timeout", "timeout"),
    (200, "missing_token_evidence", "missing_token_evidence"),
    (None, "response_mapping_failed", "response_mapping_failed"),
    (500, None, "http_500"), (502, None, "http_500"),
])
def test_describe_failure_mappings(status, code, kind):
    presentation = describe_failure(RequestError("raw message", status_code=status, code=code))
    assert presentation.kind == kind
    assert presentation.title and presentation.explanation and presentation.next_steps
    assert "raw message" not in panel_text(presentation)


def test_unknown_exception_text_is_never_presented():
    presentation = describe_failure(RuntimeError(f"PRIVATE {SECRET}"))
    assert SECRET not in panel_text(presentation) and "PRIVATE" not in panel_text(presentation)


def test_http_500_body_is_retained_bounded_and_redacted(monkeypatch):
    def send(adapter, prepared, **kwargs):
        response = requests.Response()
        response.status_code = 500
        response._content = json.dumps({"error": {"message": f"model failed to load; echoed key {SECRET}",
                                                  "type": "api_error"}}).encode()
        response.url, response.request = prepared.url, prepared
        return response

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)
    config = ModelConfig("openai_chat", "https://example.test/v1/chat/completions", "test-model",
                         credential_ref="env:TEST_FAILURE_KEY", max_retries=0, retry_backoff_s=0)
    client = OpenAICompatibleClient(config, CallBudget(1), environment={"TEST_FAILURE_KEY": SECRET})
    with pytest.raises(RequestError) as caught:
        client.generate("prompt")
    error = caught.value
    assert error.status_code == 500 and SECRET not in (error.detail or "")
    assert "model failed to load" in error.detail and "[REDACTED]" in error.detail
    presentation = describe_failure(error, profile_id="modal-gpt-oss-20b")
    assert presentation.kind == "http_500" and presentation.status_code == 500
    text = panel_text(presentation)
    assert "model failed to load" in text and "[REDACTED]" in text and SECRET not in text
    _, _, diagnostics = request_failure(error)
    assert "model failed to load" in diagnostics["detail"] and SECRET not in diagnostics["detail"]


def test_retained_detail_bounds_and_masks_credential_shaped_tokens():
    body = f"Authorization: Bearer {SECRET} api_key=abcdef123456 nvapi-ABCDEFGH1234 " + "x" * 5000
    kept = retained_detail(body)
    assert SECRET not in kept and "abcdef123456" not in kept and "nvapi-ABCDEFGH1234" not in kept
    assert kept.endswith("...[truncated]") and len(kept) <= 1000 + len(" ...[truncated]")
    assert retained_detail("  ") is None and retained_detail(None) is None
    assert retained_detail(f"echo {SECRET}", secrets=(SECRET,)) == "echo [REDACTED]"
