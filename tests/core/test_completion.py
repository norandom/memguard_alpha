"""Pure completion contracts: configuration, evidence and legacy compatibility."""

import inspect
import json
from dataclasses import FrozenInstanceError

import pytest

from recall_guard.core import completion as c
from recall_guard.core.nvidia_lm import CompletionResult, TokenLogprob


def config(**changes):
    fields = {
        "protocol": "openai_chat",
        "endpoint": "https://example.invalid/v1/chat/completions",
        "model": "chosen-model",
        "credential_ref": "MODEL_API_KEY",
    }
    return c.ModelConfig(**(fields | changes))


def evidence(status="supported", *, observed=False):
    return c.CapabilityEvidence(status=status, source="synthetic test evidence", observed=observed)


def token(*, top=True):
    return c.TokenEvidence(
        token="answer", logprob=-0.2,
        top_logprobs=(c.TokenAlternative(token="answer", logprob=-0.2),) if top else (),
    )


def test_independently_configured_models_and_explicit_protocol():
    target = config(model="target")
    rewriter = config(protocol="anthropic_messages", model="rewriter")
    assert target.model == "target"
    assert rewriter.model == "rewriter"
    assert rewriter.protocol == "anthropic_messages"
    with pytest.raises(ValueError, match="protocol"):
        config(protocol="guess_from_url")
    assert config(protocol="callback", endpoint=None, credential_ref=None).endpoint is None


def test_configuration_is_an_immutable_snapshot_and_exports_json():
    params = {"temperature": 0.0, "stop": ["END"]}
    capabilities = {"temperature": evidence()}
    model = config(parameters=params, capabilities=capabilities)
    params["stop"].append("changed")
    capabilities.clear()
    assert model.parameters["stop"] == ("END",)
    assert "temperature" in model.capabilities
    with pytest.raises(TypeError):
        model.parameters["temperature"] = 1.0
    with pytest.raises(FrozenInstanceError):
        model.model = "changed"
    exported = model.to_dict()
    exported["parameters"]["stop"].append("detached")
    assert model.parameters["stop"] == ("END",)
    assert json.loads(json.dumps(model.to_dict()))["parameters"]["temperature"] == 0.0
    assert "credential_ref" not in exported
    assert "MODEL_API_KEY" not in repr(model)


def test_omitted_and_explicit_null_parameters_remain_distinct():
    assert "temperature" not in config().to_dict()["parameters"]
    assert config(parameters={"temperature": None}).to_dict()["parameters"] == {"temperature": None}


@pytest.mark.parametrize("endpoint", [
    "https://user:secret@example.invalid/chat", "https://example.invalid/chat?api_key=secret",
    "https://example.invalid/chat#secret", "file:///local", "not a url", None,
])
def test_http_endpoint_rejects_embedded_credentials_and_invalid_urls(endpoint):
    with pytest.raises(ValueError, match="endpoint") as error:
        config(endpoint=endpoint)
    assert "secret" not in str(error.value)


@pytest.mark.parametrize("reference", ["sk-secret-value", "Bearer secret", "", " key "])
def test_credential_reference_is_a_reference_not_a_resolved_secret(reference):
    with pytest.raises(ValueError, match="credential_ref") as error:
        config(credential_ref=reference)
    assert reference not in str(error.value) or reference == ""


@pytest.mark.parametrize("parameters", [
    {"api_key": "secret"}, {"Authorization": "secret"}, {"model": "override"},
    {"messages": []}, {"headers": {"x-api-key": "secret"}}, {"extra": {"api_key": "secret"}},
    {"temperature": float("nan")}, {"top_p": float("inf")}, {"max_tokens": 0},
    {"max_tokens": True}, {"seed": 1.5}, {"logprobs": "yes"}, {"top_logprobs": -1},
    {"temperature": -0.1}, {"top_p": 1.1}, {"extra": object()},
])
def test_invalid_or_transport_overriding_parameters_fail_without_exposing_values(parameters):
    with pytest.raises((TypeError, ValueError)) as error:
        config(parameters=parameters)
    assert "secret" not in str(error.value)


@pytest.mark.parametrize(("field", "value"), [
    ("timeout_s", 0), ("timeout_s", float("nan")), ("timeout_s", True),
    ("max_retries", -1), ("max_retries", 1.5), ("max_retries", False),
    ("retry_backoff_s", float("inf")), ("retry_backoff_s", -1),
    ("min_call_interval_s", float("nan")), ("min_call_interval_s", -1),
])
def test_nonfinite_and_invalid_request_controls_fail_locally(field, value):
    with pytest.raises((TypeError, ValueError), match=field):
        config(**{field: value})


def test_inspection_preserves_three_states_and_unobserved_schema_evidence():
    model = config(parameters={"temperature": 0.2, "seed": 4}, capabilities={
        "text_generation": evidence(observed=True), "temperature": evidence(),
        "seed": evidence("unsupported"),
    })
    report = model.inspect()
    assert report.capabilities["text_generation"].status == "supported"
    assert report.capabilities["temperature"].status == "supported"
    assert report.capabilities["seed"].status == "unsupported"
    assert report.unsupported == ("seed",)
    assert report.requires_probe == ("temperature",)
    assert not report.ready_for_batch
    assert "synthetic" in report.capabilities["seed"].source
    undeclared = config().inspect()
    assert undeclared.capabilities["text_generation"].status == "unverified"
    assert undeclared.requires_probe == ("text_generation",)


def test_text_generation_ensemble_and_rewriting_do_not_require_fake_mia():
    model = config(capabilities={"text_generation": evidence(observed=True), "logprobs": evidence("unsupported")})
    choices = c.FeatureChoices(ensemble_draws=4, rewriting=True)
    report = c.validate_features(model, choices, require_verified=True)
    assert report.ready_for_batch
    assert report.mia_unavailable_reason
    assert "not selected" in report.mia_unavailable_reason


def test_mia_requires_complete_capability_evidence_and_rejects_incompatible_choice():
    model = config(capabilities={"text_generation": evidence(observed=True), "logprobs": evidence(observed=True),
                                 "top_logprobs": evidence("unsupported")})
    with pytest.raises(ValueError, match="top_logprobs"):
        c.validate_features(model, c.FeatureChoices(mia_scoring=True))
    assert model.inspect(c.FeatureChoices(mia_scoring=True)).mia_unavailable_reason


def test_unverified_features_can_be_inspected_but_cannot_start_verified_batch():
    model = config()
    assert not c.validate_features(model).ready_for_batch
    with pytest.raises(ValueError, match="unverified|probe"):
        c.validate_features(model, require_verified=True)


def test_complete_observed_capabilities_allow_explicit_mia_choice():
    model = config(capabilities={name: evidence(observed=True)
                                 for name in ("text_generation", "logprobs", "top_logprobs")})
    report = c.validate_features(model, c.FeatureChoices(mia_scoring=True), require_verified=True)
    assert report.ready_for_batch
    assert report.mia_unavailable_reason is None


@pytest.mark.parametrize("parameters", [{"logprobs": False}, {"top_logprobs": 0}])
def test_mia_feature_selection_does_not_override_explicit_parameters(parameters):
    with pytest.raises(ValueError, match="logprobs"):
        c.validate_features(config(parameters=parameters), c.FeatureChoices(mia_scoring=True))


@pytest.mark.parametrize("parameter", ["logprobs", "top_logprobs"])
def test_explicit_null_logprob_settings_conflict_with_mia_even_when_capabilities_are_observed(parameter):
    model = config(parameters={parameter: None}, capabilities={name: evidence(observed=True)
                   for name in ("text_generation", "logprobs", "top_logprobs")})
    with pytest.raises(ValueError, match=rf"{parameter}.*explicit.*null"):
        c.validate_features(model, c.FeatureChoices(mia_scoring=True), require_verified=True)


@pytest.mark.parametrize("parameter", ["logprobs", "top_logprobs"])
def test_explicit_null_logprob_settings_remain_preserved_without_mia(parameter):
    model = config(parameters={parameter: None}, capabilities={name: evidence(observed=True)
                   for name in ("text_generation", parameter)})
    assert c.validate_features(model, require_verified=True).ready_for_batch
    assert model.parameters == {parameter: None}
    assert model.to_dict()["parameters"] == {parameter: None}


@pytest.mark.parametrize("kwargs", [{"ensemble_draws": 0}, {"ensemble_draws": True}, {"mia_scoring": "yes"},
                                  {"rewriting": 1}])
def test_feature_choices_have_valid_types_and_positive_draw_count(kwargs):
    with pytest.raises((TypeError, ValueError)):
        c.FeatureChoices(**kwargs)


def test_text_response_marks_missing_mia_and_usage_as_unavailable():
    response = c.ModelResponse(content="answer")
    assert response.token_evidence is None
    assert response.usage is None
    assert response.mia_unavailable_reason
    assert dict(response.confirmed_settings) == {}
    with pytest.raises(ValueError, match="evidence"):
        response.to_legacy()


def test_response_metadata_is_independent_immutable_and_no_settings_are_invented():
    requested = {"temperature": 0.2, "stop": ["END"]}
    usage = {"input_tokens": 3}
    response = c.ModelResponse(content="answer", requested_settings=requested, usage=usage,
                               finish_reason="length", request_id="request-123")
    requested["stop"].clear()
    usage["input_tokens"] = 99
    assert response.requested_settings["stop"] == ("END",)
    assert response.usage["input_tokens"] == 3
    assert "output_tokens" not in response.usage
    assert dict(response.confirmed_settings) == {}
    with pytest.raises(TypeError):
        response.requested_settings["temperature"] = 0.7
    exported = response.to_dict()
    assert json.loads(json.dumps(exported))["request_id"] == "request-123"
    assert exported["confirmed_settings"] == {}


def test_explicit_legacy_roundtrip_preserves_original_constructor_and_evidence():
    original = CompletionResult("answer", [TokenLogprob("answer", -0.2, [{"token": "answer", "logprob": -0.2}])], 0.3)
    converted = c.ModelResponse.from_legacy(original, requested_settings={"temperature": 0.2})
    assert converted.token_evidence[0].logprob == -0.2
    assert converted.confirmed_settings == {"temperature": 0.3}
    assert converted.requested_settings == {"temperature": 0.2}
    assert converted.mia_unavailable_reason is None
    original.logprobs[0].top_logprobs.clear()
    legacy = converted.to_legacy()
    assert isinstance(legacy, CompletionResult)
    assert legacy.logprobs[0].top_logprobs == [{"token": "answer", "logprob": -0.2}]
    assert legacy.raw_temperature_observed == 0.3
    legacy.logprobs.clear()
    assert len(converted.token_evidence) == 1


def test_unechoed_temperature_stays_unconfirmed_in_legacy_adapter():
    legacy = CompletionResult("answer", [TokenLogprob("answer", -0.2, [{"token": "answer", "logprob": -0.2}])], None)
    response = c.ModelResponse.from_legacy(legacy, requested_settings={"temperature": 0.2})
    assert dict(response.confirmed_settings) == {}
    assert response.to_legacy().raw_temperature_observed is None


@pytest.mark.parametrize("entries", [[], [TokenLogprob("answer", -0.2, [])]])
def test_incomplete_legacy_results_are_not_promoted_to_complete_evidence(entries):
    with pytest.raises(ValueError, match="evidence"):
        c.ModelResponse.from_legacy(CompletionResult("answer", entries, None))


@pytest.mark.parametrize("entries", [(), None])
def test_empty_or_absent_evidence_does_not_convert_to_legacy(entries):
    response = c.ModelResponse(content="answer", token_evidence=entries)
    assert response.mia_unavailable_reason
    with pytest.raises(ValueError, match="evidence"):
        response.to_legacy()


def test_realized_logprobs_without_alternatives_are_not_mia_compatible():
    response = c.ModelResponse(content="answer", token_evidence=(token(top=False),))
    assert "top_logprobs" in response.mia_unavailable_reason
    with pytest.raises(ValueError, match="top_logprobs"):
        response.to_legacy()


@pytest.mark.parametrize(("realized", "alternative", "location"), [
    (-9999.0, -0.2, "realized"), (-0.2, -9999.0, "top_logprobs"),
])
def test_censored_logprobs_remain_raw_but_are_unavailable_for_mia(realized, alternative, location):
    entries = (token(), c.TokenEvidence("rare", realized, (
        c.TokenAlternative("answer", -0.2), c.TokenAlternative("rare", alternative),
    )))
    response = c.ModelResponse("answer rare", entries)
    assert response.content == "answer rare" and response.token_evidence == entries
    exported = json.loads(json.dumps(response.to_dict(), allow_nan=False))
    assert exported["token_evidence"][1]["logprob"] == realized
    assert exported["token_evidence"][1]["top_logprobs"][1]["logprob"] == alternative
    assert "censored" in response.mia_unavailable_reason
    assert location in response.mia_unavailable_reason
    assert "-9999.0" in response.mia_unavailable_reason
    if location == "realized":
        assert "censored" in response.logprobs_unavailable_reason
    else:
        assert response.logprobs_unavailable_reason is None
    with pytest.raises(ValueError, match="censored"):
        response.to_legacy()


@pytest.mark.parametrize(("realized", "alternative"), [(-9999.0, -0.2), (-0.2, -9999.0)])
def test_censored_legacy_logprobs_cannot_be_promoted_to_usable_evidence(realized, alternative):
    legacy = CompletionResult("answer", [TokenLogprob("answer", realized, [
        {"token": "answer", "logprob": alternative},
    ])], None)
    with pytest.raises(ValueError, match="censored"):
        c.ModelResponse.from_legacy(legacy)
    assert legacy.logprobs[0].logprob == realized
    assert legacy.logprobs[0].top_logprobs[0]["logprob"] == alternative


@pytest.mark.parametrize("logprob", [-9998.9, -9999.1, -10000.0])
def test_ordinary_negative_logprobs_are_not_censored_by_a_numeric_threshold(logprob):
    response = c.ModelResponse("answer", (c.TokenEvidence("answer", logprob, (
        c.TokenAlternative("answer", logprob),
    )),))
    assert response.mia_unavailable_reason is None
    restored = c.ModelResponse.from_legacy(response.to_legacy())
    assert restored.to_dict() == response.to_dict()


@pytest.mark.parametrize("kwargs", [
    {"content": None}, {"content": 1}, {"usage": {"input_tokens": -1}}, {"usage": {"input_tokens": True}},
    {"requested_settings": {"Authorization": "secret"}}, {"confirmed_settings": {"temperature": float("nan")}},
])
def test_invalid_response_contracts_fail_at_boundary(kwargs):
    with pytest.raises((TypeError, ValueError)) as error:
        c.ModelResponse(**({"content": "answer"} | kwargs))
    assert "secret" not in str(error.value)


@pytest.mark.parametrize("logprob", [float("nan"), float("inf"), float("-inf"), True, 0.1])
def test_token_evidence_rejects_invalid_logprob_values(logprob):
    with pytest.raises((TypeError, ValueError), match="logprob"):
        c.TokenAlternative(token="answer", logprob=logprob)


def test_synchronous_client_contract_is_structural():
    class CallerClient:
        config = config(protocol="callback", endpoint=None, credential_ref=None)

        def generate(self, prompt: str, **parameters: c.ParameterValue) -> c.ModelResponse:
            return c.ModelResponse(content=prompt, requested_settings=dict(self.config.parameters) | parameters)

    client = CallerClient()
    assert isinstance(client, c.CompletionClient)
    assert not inspect.iscoroutinefunction(c.CompletionClient.generate)
    assert inspect.signature(c.CompletionClient.generate).parameters["parameters"].kind == inspect.Parameter.VAR_KEYWORD
    assert client.generate("answer").content == "answer"
    assert client.generate("answer", temperature=0.5, max_tokens=128).requested_settings == {
        "temperature": 0.5, "max_tokens": 128,
    }
    assert dict(client.config.parameters) == {}


def test_response_optionally_carries_full_distribution_evidence():
    from recall_guard.core.full_distribution import FullDistributionEvidence

    evidence = FullDistributionEvidence(
        model_digest="sha256:" + "a" * 64, n_vocab=4, response_token_ids=(2,), logprob=(-0.1,), entropy=(0.4,),
        rank=(1,), top_logprobs=(((2, -0.1), (0, -2.5)),))
    plain = c.ModelResponse("answer")
    assert plain.full_distribution is None
    assert "full_distribution" not in plain.to_dict()
    carried = c.ModelResponse("answer", full_distribution=evidence)
    assert carried.full_distribution is evidence
    restored = c.ModelResponse.from_dict(json.loads(json.dumps(carried.to_dict())))
    assert restored.full_distribution == evidence
    assert restored == carried
    with pytest.raises(TypeError, match="full_distribution"):
        c.ModelResponse("answer", full_distribution=evidence.to_dict())
