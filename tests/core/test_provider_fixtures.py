"""Validate frozen wire examples; these tests do not implement provider adapters."""

import json
from pathlib import Path

import pytest

FIXTURE_DIRECTORY = Path(__file__).parents[1] / "fixtures" / "providers"
PROVIDERS = ("openai", "anthropic", "mistral", "nim")


def _load(provider):
    return json.loads((FIXTURE_DIRECTORY / f"{provider}.json").read_text(encoding="utf-8"))


@pytest.fixture(params=PROVIDERS)
def provider_fixture(request):
    return _load(request.param)


def test_fixture_provenance_and_explicit_protocol(provider_fixture):
    fixture = provider_fixture
    assert fixture["provider"] in PROVIDERS
    assert fixture["protocol"] == (
        "anthropic_messages" if fixture["provider"] == "anthropic" else "openai_chat"
    )
    assert fixture["provenance"] == "synthetic_from_official_documentation"
    assert fixture["reviewed_at"] == "2026-10-30"
    assert fixture["sources"] and all(url.startswith("https://") for url in fixture["sources"])
    assert fixture["request"]["endpoint"].endswith(
        "/v1/messages" if fixture["provider"] == "anthropic" else "/v1/chat/completions"
    )
    assert fixture["request"]["body"]["model"]
    # Credentials/headers never come from recordings or the local environment.
    serialized = json.dumps(fixture).lower()
    for forbidden in ('"authorization"', '"x-api-key"', '"api_key"', "bearer ", "sk-"):
        assert forbidden not in serialized


def test_text_success_has_no_fabricated_token_evidence(provider_fixture):
    response = provider_fixture["responses"]["text_only"]
    assert response["status_code"] == 200
    body = response["body"]
    if provider_fixture["protocol"] == "anthropic_messages":
        assert body["content"] == [{"type": "text", "text": "Bullish"}]
        assert "logprobs" not in body
    else:
        choice = body["choices"][0]
        assert choice["message"]["content"] == "Bullish"
        assert choice.get("logprobs") is None
    assert "temperature" not in body  # Success does not confirm an unechoed control.
    assert body["usage"]


@pytest.mark.parametrize("provider", ("openai", "nim"))
def test_token_evidence_success_is_distinct_from_missing_logprobs(provider):
    response = _load(provider)["responses"]["token_evidence"]
    assert response["status_code"] == 200
    tokens = response["body"]["choices"][0]["logprobs"]["content"]
    assert tokens and "".join(token["token"] for token in tokens) == "Bullish"
    assert all(token["logprob"] < 0 and token["top_logprobs"] for token in tokens)


@pytest.mark.parametrize("case,status_code", (("unauthorized", 401), ("forbidden", 403)))
def test_access_errors_are_distinct_from_absent_token_evidence(provider_fixture, case, status_code):
    response = provider_fixture["responses"][case]
    assert response["status_code"] == status_code
    body = response["body"]
    assert "choices" not in body and "content" not in body
    assert body
    assert "unsupported_fixture_parameter" not in json.dumps(body)


def test_rejected_explicit_parameter_is_retained_with_its_error(provider_fixture):
    response = provider_fixture["responses"]["unsupported_parameter"]
    assert response["status_code"] in (400, 422)
    parameter = response["rejected_parameter"]
    assert response["request_parameters"] == {parameter: True}
    assert parameter in json.dumps(response["body"])
    assert "choices" not in response["body"]


def test_omitted_and_null_logprobs_are_both_represented():
    choices = [_load(provider)["responses"]["text_only"]["body"]["choices"][0]
               for provider in ("openai", "mistral", "nim")]
    assert any("logprobs" not in choice for choice in choices)
    assert any("logprobs" in choice and choice["logprobs"] is None for choice in choices)


def test_anthropic_truncation_preserves_native_stop_reason():
    response = _load("anthropic")["responses"]["truncated"]
    assert response["status_code"] == 200
    assert response["body"]["stop_reason"] == "max_tokens"
    assert response["body"]["usage"]["output_tokens"] > 0
