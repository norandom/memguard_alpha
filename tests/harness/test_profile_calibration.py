"""Calibration identity v2: calibrators never cross evidence profiles (R2.5)."""

import dataclasses
import json
from types import SimpleNamespace

import pytest

from recall_guard.core.completion import FeatureChoices, ModelConfig
from recall_guard.core.harmony_evidence import MARKERS, HarmonyTokenMap
from recall_guard.core.local_evidence import LocalModelIdentity
from recall_guard.core.providers import HarmonyOpenAICompatibleClient, OpenAICompatibleClient
from recall_guard.core.request_control import CallBudget
from recall_guard.harness import scorer as scorer_module
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
