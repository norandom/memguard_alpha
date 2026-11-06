"""Synthetic same-response channel mapping: no server, tokenizer runtime or RNG."""

import copy
import json

import pytest
import requests

from recall_guard.core.completion import ModelConfig
from recall_guard.core.harmony_evidence import MARKERS, HarmonyTokenMap, final_answer_span, map_final_response
from recall_guard.core.local_evidence import LocalEvidenceError, validate_local_evidence
from recall_guard.core.providers import HarmonyOpenAICompatibleClient, OpenAICompatibleClient
from recall_guard.core.request_control import CallBudget, RequestError

IDS = dict(zip(MARKERS, (200005, 200006, 200007, 200008, 200002, 200012), strict=True))


@pytest.fixture
def token_map(tmp_path):
    path = tmp_path / "tokenizer.json"
    path.write_text(json.dumps({"added_tokens": [
        {"content": label, "id": token_id, "special": True} for label, token_id in IDS.items()
    ]}), encoding="utf-8")
    return HarmonyTokenMap.from_tokenizer_json(path, tokenizer_revision="a" * 40)


def body(parts, *, content="OK", reasoning="Think.", finish_reason="stop"):
    entries = [{"token": token, "bytes": list(token.encode()), "logprob": -(i + 1) / 100,
                "top_logprobs": [{"token": token, "logprob": -(i + 1) / 100},
                                 {"token": "other", "logprob": -10.0}]}
               for i, token in enumerate(parts)]
    return {"model": "openai/gpt-oss-20b", "choices": [{"index": 0,
        "message": {"role": "assistant", "content": content, "reasoning": reasoning},
        "finish_reason": finish_reason, "token_ids": [IDS.get(token, 100 + i) for i, token in enumerate(parts)],
        "logprobs": {"content": entries}}]}


def reasoning_body():
    return body(["<|channel|>", "analysis", "<|message|>", "Think.", "<|end|>",
                 "<|start|>", "assistant", "<|channel|>", "final", "<|message|>", "OK", "<|return|>"])


def parse(raw, token_map, *, generic=False):
    config = ModelConfig(protocol="openai_chat", provider="modal", model="openai/gpt-oss-20b",
                         endpoint="http://localhost:8000/v1/chat/completions",
                         parameters={"return_token_ids": True})
    adapter = (OpenAICompatibleClient(config, CallBudget(0), environment={}, allow_unauthenticated=True)
               if generic else HarmonyOpenAICompatibleClient(config, CallBudget(0), token_map=token_map,
                                                               environment={}, allow_unauthenticated=True))
    response = requests.Response()
    response.status_code = 200
    return adapter._parse_response(response, raw, dict(config.parameters))


def test_same_response_reasoning_is_excluded_preserving_selected_probability(token_map):
    raw = reasoning_body()
    original = copy.deepcopy(raw)
    response = parse(raw, token_map)
    validate_local_evidence(response, required_alternatives=2)
    assert [token.logprob for token in response.token_evidence] == [-0.11]
    assert response.token_evidence[0].top_logprobs[1].logprob == -10.0
    assert raw == original


def test_generic_adapter_remains_unchanged(token_map):
    response = parse(reasoning_body(), token_map, generic=True)
    with pytest.raises(LocalEvidenceError, match="token_text_alignment_failed"):
        validate_local_evidence(response)


def test_explicit_final_only_header(token_map):
    raw = body(["<|channel|>", "final", "<|message|>", "O", "K", "<|return|>"], reasoning=None)
    response = parse(raw, token_map)
    validate_local_evidence(response, required_alternatives=2)
    assert [token.token for token in response.token_evidence] == ["O", "K"]


@pytest.mark.parametrize("mutation", [
    lambda raw: raw["choices"][0].update(finish_reason="length"),
    lambda raw: raw["choices"][0].update(token_ids=[1]),
    lambda raw: raw["choices"][0].pop("token_ids"),
    lambda raw: raw["choices"][0]["message"].update(content="Something else"),
    lambda raw: raw["choices"][0]["message"].update(tool_calls=[{"id": "tool"}]),
    lambda raw: raw["choices"][0]["logprobs"]["content"][10].update(bytes=[255]),
])
def test_ambiguous_or_inconsistent_evidence_fails_closed(token_map, mutation):
    raw = reasoning_body()
    mutation(raw)
    with pytest.raises(RequestError, match="Harmony"):
        parse(raw, token_map)


@pytest.mark.parametrize("parts", [
    ["<|channel|>", "analysis", "<|message|>", "Think.", "<|end|>"],
    ["<|channel|>", "final", "<|message|>", "OK", "<|end|>", "<|start|>", "assistant",
     "<|channel|>", "final", "<|message|>", "OK", "<|return|>"],
    ["<|channel|>", "commentary", "<|message|>", "OK", "<|return|>"],
    ["<|start|>", "assistant to=python", "<|channel|>", "analysis", "<|message|>", "OK", "<|call|>"],
    ["<|channel|>", "final", "<|message|>", "OK", "<|return|>", "extra"],
    ["<|channel|>", "final", "<|message|>", "OK"],
])
def test_unsupported_channel_sequences_are_rejected(token_map, parts):
    with pytest.raises(RequestError, match="Harmony"):
        parse(body(parts), token_map)


def test_split_headers_and_explicit_assistant_prefix(token_map):
    raw = body(["<|start|>", "assist", "ant", "<|channel|>", "fi", "nal", "<|message|>",
                "O", "K", "<|return|>"], reasoning=None)
    validate_local_evidence(parse(raw, token_map), required_alternatives=2)


def test_ordinary_answer_marker_text_is_not_treated_as_control_id(token_map):
    raw = body(["<|channel|>", "final", "<|message|>", "<|return|>", "<|return|>"],
               content="<|return|>", reasoning=None)
    raw["choices"][0]["token_ids"][3] = 15
    validate_local_evidence(parse(raw, token_map), required_alternatives=2)


def test_consistent_unicode_and_absent_byte_metadata(token_map):
    raw = body(["<|channel|>", "final", "<|message|>", "é", "<|return|>"], content="é", reasoning=None)
    validate_local_evidence(parse(raw, token_map))
    raw["choices"][0]["logprobs"]["content"][3].pop("bytes")
    validate_local_evidence(parse(raw, token_map))


@pytest.mark.parametrize("changes", [
    {"marker_ids": (1,)}, {"marker_ids": (1, 2, 3, 4, 5, True)},
    {"marker_ids": (1, 1, 1, 1, 1, 1)}, {"special_token_ids": frozenset()},
    {"special_token_ids": {1, 2}}, {"tokenizer_revision": "main"}, {"tokenizer_sha256": "unknown"},
])
def test_direct_profile_construction_cannot_bypass_validation(token_map, changes):
    from dataclasses import replace

    with pytest.raises(ValueError):
        replace(token_map, **changes)


def test_profile_requires_explicit_model_and_metadata_request(token_map):
    for model, parameters in [("other-model", {"return_token_ids": True}), ("openai/gpt-oss-20b", {})]:
        config = ModelConfig(protocol="openai_chat", provider="modal", model=model,
                             endpoint="http://localhost:8000/v1/chat/completions", parameters=parameters)
        with pytest.raises(ValueError, match="Harmony"):
            HarmonyOpenAICompatibleClient(config, CallBudget(0), token_map=token_map,
                                          environment={}, allow_unauthenticated=True)


def test_mapped_positions_keep_original_ids_entries_and_alternatives(token_map):
    raw = reasoning_body()
    span = final_answer_span(raw["choices"][0], token_map)
    assert (span.start, span.stop, span.generated_positions) == (10, 11, 12)
    mapped = map_final_response(raw, token_map)
    assert mapped["choices"][0]["token_ids"] == [raw["choices"][0]["token_ids"][10]]
    assert mapped["choices"][0]["logprobs"]["content"][0] is raw["choices"][0]["logprobs"]["content"][10]


def test_split_utf8_byte_fragment_is_rejected_without_lossy_repair(token_map):
    raw = reasoning_body()
    raw["choices"][0]["logprobs"]["content"][10]["bytes"] = [195]
    with pytest.raises(RequestError, match="Harmony"):
        parse(raw, token_map)


@pytest.mark.parametrize("value", [None, "main", 4])
def test_tokenizer_metadata_requires_revision_and_valid_special_definitions(tmp_path, value):
    path = tmp_path / "malformed.json"
    path.write_text('{"added_tokens": []}', encoding="utf-8")
    with pytest.raises(ValueError):
        HarmonyTokenMap.from_tokenizer_json(path, tokenizer_revision=value)


def test_mapping_reuses_real_transport_and_shared_physical_budget(monkeypatch, token_map):
    calls = []

    def send(adapter, prepared, **kwargs):
        calls.append(json.loads(prepared.body))
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(reasoning_body()).encode()
        return response

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)
    config = ModelConfig(protocol="openai_chat", provider="modal", model="openai/gpt-oss-20b",
                         endpoint="http://localhost:8000/v1/chat/completions", max_retries=0,
                         parameters={"return_token_ids": True, "logprobs": True, "top_logprobs": 2})
    budget = CallBudget(1)
    adapter = HarmonyOpenAICompatibleClient(config, budget, token_map=token_map,
                                           environment={}, allow_unauthenticated=True)
    response = adapter.generate("Reply with the word OK.")
    validate_local_evidence(response, required_alternatives=2)
    assert budget.used == len(calls) == 1
    assert calls[0]["return_token_ids"] is True
    assert len(response.token_evidence) == 1
