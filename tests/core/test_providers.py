"""Replay synthetic wire fixtures through the actual requests transport boundary."""

import json
import traceback
from dataclasses import replace
from pathlib import Path

import pytest
import requests

from recall_guard.core.completion import (
    CapabilityEvidence,
    CapabilityStatus,
    CompletionClient,
    FeatureChoices,
    ModelConfig,
    ModelResponse,
)
from recall_guard.core.providers import OpenAICompatibleClient
from recall_guard.core.request_control import BudgetExhaustedError, CallBudget, RequestError

FIXTURES = Path(__file__).parents[1] / "fixtures" / "providers"
FAKE_KEY = "synthetic-provider-secret"


def config(provider="openai", **overrides):
    fixture = json.loads((FIXTURES / f"{provider}.json").read_text(encoding="utf-8"))
    body = fixture["request"]["body"]
    return ModelConfig(**({
        "protocol": "openai_chat", "provider": provider,
        "endpoint": fixture["request"]["endpoint"], "model": body["model"],
        "credential_ref": "env:TEST_PROVIDER_KEY", "max_retries": 0, "retry_backoff_s": 0,
        "parameters": {key: value for key, value in body.items() if key not in {"model", "messages"}},
    } | overrides))


def fixture_response(provider="openai", name="text_only"):
    return json.loads((FIXTURES / f"{provider}.json").read_text(encoding="utf-8"))["responses"][name]


def test_probe_explicit_prompt_and_stage_are_provenance_not_body_controls(wire):
    pending, calls = wire
    pending.append(fixture_response())
    budget = CallBudget(1)
    adapter = client(budget=budget)
    adapter.probe("EXACT_PROBE_INPUT", stage="execution/probe/target")
    assert budget.events[0].stage == "execution/probe/target"
    assert calls[0]["body"]["messages"] == [{"role": "user", "content": "EXACT_PROBE_INPUT"}]
    assert "stage" not in calls[0]["body"]


def test_legacy_response_bearing_probe_uses_one_exchange_and_preserves_default_probe_ledger(wire):
    pending, calls = wire
    pending.extend([fixture_response(), fixture_response()])
    budget = CallBudget(2)
    adapter = client(budget=budget)
    evidence = adapter.probe_with_evidence(stage="preparation/probe/target")
    assert evidence.response.content and evidence.complete and evidence.ready
    assert evidence.observations[-1].timestamp
    assert budget.used == len(calls) == 1
    assert budget.events[0].purpose == "probe"
    assert adapter.config.readiness is None
    adapter.probe()
    assert budget.used == len(calls) == 2
    assert "purpose" not in budget.events[1].to_dict()


def test_legacy_typed_rejection_preserves_failure_without_hidden_second_generate(wire):
    pending, calls = wire
    entry = fixture_response()
    entry = dict(entry, status_code=400, body={"error": {"message": "Unsupported control", "param": "logprobs"}})
    pending.append(entry)
    budget = CallBudget(1)
    adapter = client(settings=config(parameters={"logprobs": True}), budget=budget)
    evidence = adapter.probe_with_evidence()
    assert not evidence.ready and evidence.response is None
    assert evidence.report.capabilities["logprobs"].status == CapabilityStatus.UNSUPPORTED
    assert budget.used == len(calls) == 1


def client(settings=None, budget=None, **options):
    return OpenAICompatibleClient(settings or config(), budget or CallBudget(10),
                                  environment={"TEST_PROVIDER_KEY": FAKE_KEY}, **options)


@pytest.fixture
def wire(monkeypatch):
    pending, calls = [], []

    def send(adapter, prepared, **kwargs):
        calls.append({"url": prepared.url, "body": json.loads(prepared.body),
                      "headers": dict(prepared.headers), "timeout": kwargs["timeout"],
                      "hidden_retries": adapter.max_retries.total})
        assert pending, "Unexpected physical HTTP request"
        entry = pending.pop(0)
        if isinstance(entry, Exception):
            raise entry
        response = requests.Response()
        response.status_code = entry["status_code"]
        response._content = json.dumps(entry["body"]).encode("utf-8")
        response.headers.update(entry.get("headers", {}))
        response.url = prepared.url
        response.request = prepared
        return response

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)
    return pending, calls


@pytest.mark.parametrize("provider", ["openai", "mistral", "nim"])
def test_text_fixtures_use_exact_protocol_payload_and_preserve_absent_evidence(provider, wire):
    pending, calls = wire
    pending.append(fixture_response(provider))
    budget = CallBudget(1)
    settings = config(provider)
    adapter = client(settings, budget)
    assert isinstance(adapter, CompletionClient)
    assert calls == []
    response = adapter.generate("Return Bullish.")
    assert response.content == "Bullish"
    assert response.token_evidence is None
    assert response.mia_unavailable_reason
    assert response.finish_reason == "stop"
    assert response.request_id == "fixture-response"
    assert dict(response.usage) == {"prompt_tokens": 8, "completion_tokens": 1, "total_tokens": 9}
    assert dict(response.requested_settings) == settings.to_dict()["parameters"]
    assert dict(response.confirmed_settings) == {}
    assert calls[0]["url"] == settings.endpoint
    assert calls[0]["body"] == {"model": settings.model, "messages": [{"role": "user", "content": "Return Bullish."}],
                                  **settings.to_dict()["parameters"]}
    assert calls[0]["headers"]["Authorization"] == f"Bearer {FAKE_KEY}"
    assert calls[0]["timeout"] == settings.timeout_s
    assert calls[0]["hidden_retries"] == 0
    assert budget.used == 1 and budget.events[0].status == "succeeded"
    assert dict(budget.events[0].usage) == dict(response.usage)


@pytest.mark.parametrize("provider", ["openai", "nim"])
def test_genuine_token_evidence_fixtures_support_legacy_conversion(provider, wire):
    pending, calls = wire
    fixture = fixture_response(provider, "token_evidence")
    pending.append(fixture)
    response = client(config(provider, parameters=fixture["request_parameters"]),
                      features=FeatureChoices(mia_scoring=True)).generate("Return Bullish.")
    assert response.mia_unavailable_reason is None
    assert response.token_evidence[0].token == "Bullish"
    assert response.token_evidence[0].logprob == -0.25
    assert response.token_evidence[0].top_logprobs[1].token == "Bearish"
    assert response.to_legacy().logprobs[0].top_logprobs[1]["logprob"] == -1.75
    assert calls[0]["body"]["top_logprobs"] == 2
    assert "logprobs" not in response.confirmed_settings


def test_overrides_preserve_native_parameter_names_nulls_and_detached_nested_json(wire):
    pending, calls = wire
    pending.extend([fixture_response(), fixture_response()])
    settings = config(parameters={"max_completion_tokens": 64, "temperature": 0.3, "stop": ["END"],
                                  "response_format": {"type": "json_schema", "json_schema": {"name": "answer"}}})
    adapter = client(settings)
    first = adapter.generate("first", temperature=None, seed=7)
    adapter.generate("second")
    assert calls[0]["body"]["max_completion_tokens"] == 64
    assert "max_tokens" not in calls[0]["body"]
    assert calls[0]["body"]["temperature"] is None
    assert calls[0]["body"]["seed"] == 7
    assert calls[0]["body"]["stop"] == ["END"]
    assert calls[0]["body"]["response_format"]["json_schema"] == {"name": "answer"}
    assert calls[1]["body"]["temperature"] == 0.3
    assert "seed" not in calls[1]["body"]
    assert first.requested_settings["temperature"] is None
    assert settings.parameters["temperature"] == 0.3


@pytest.mark.parametrize("override", [{"messages": []}, {"stream": True}, {"api_key": "other"}, {"max_tokens": 0}])
def test_invalid_overrides_fail_before_admission(override, wire):
    _, calls = wire
    budget = CallBudget(1)
    adapter = client(budget=budget)
    with pytest.raises((ValueError, TypeError)):
        adapter.generate("test", **override)
    assert calls == [] and budget.used == 0


def test_declared_unsupported_control_is_rejected_for_config_and_per_call(wire):
    _, calls = wire
    settings = config(capabilities={"temperature": CapabilityEvidence("unsupported", "test declaration")})
    budget = CallBudget(1)
    adapter = client(settings, budget)
    with pytest.raises(ValueError, match="temperature"):
        adapter.generate("test", temperature=0.5)
    with pytest.raises(ValueError, match="temperature"):
        client(replace(settings, parameters={"temperature": 0.5}), budget)
    assert calls == [] and budget.used == 0


@pytest.mark.parametrize("provider", ["openai", "mistral", "nim"])
@pytest.mark.parametrize("name", ["unauthorized", "forbidden", "unsupported_parameter"])
def test_fixture_http_failures_preserve_status_detail_parameter_without_retry_or_stripping(provider, name, wire):
    pending, calls = wire
    fixture = fixture_response(provider, name)
    pending.append(fixture)
    budget = CallBudget(8)
    parameters = fixture.get("request_parameters", {})
    adapter = client(config(provider, parameters=parameters, max_retries=3), budget)
    with pytest.raises(RequestError) as caught:
        adapter.generate("Return Bullish.")
    error = caught.value
    assert error.status_code == fixture["status_code"]
    assert json.loads(error.detail) == fixture["body"]
    assert error.retryable is False
    assert len(calls) == budget.used == 1
    if name == "unsupported_parameter":
        assert error.parameter == fixture["rejected_parameter"]
        assert calls[0]["body"][error.parameter] is True
    assert budget.events[0].status == "failed"
    assert budget.events[0].status_code == error.status_code
    assert budget.events[0].detail == error.detail


@pytest.mark.parametrize("status,code", [(429, "insufficient_quota"), (402, "billing_not_active"),
                                       (404, "model_not_found"), (400, "unsupported_parameter")])
def test_model_billing_and_parameter_errors_remain_distinct_permanent_failures(status, code, wire):
    pending, calls = wire
    pending.append({"status_code": status, "body": {"error": {"message": "rejected", "code": code,
                                                              "param": "max_tokens"},
                                                  "usage": {"completion_tokens": 2}}})
    budget = CallBudget(5)
    with pytest.raises(RequestError) as caught:
        client(config(parameters={"max_tokens": 17}, max_retries=3), budget).generate("test")
    assert caught.value.code == code
    assert caught.value.parameter == "max_tokens"
    assert caught.value.status_code == status
    assert dict(caught.value.usage) == {"completion_tokens": 2}
    assert dict(budget.events[0].usage) == {"completion_tokens": 2}
    assert len(calls) == budget.used == 1
    assert calls[0]["body"]["max_tokens"] == 17 and "max_completion_tokens" not in calls[0]["body"]


@pytest.mark.parametrize("error_type", [requests.Timeout, requests.ConnectionError])
def test_transient_requests_exceptions_retry_only_through_shared_admission(error_type, wire):
    pending, calls = wire
    pending.extend([error_type(f"failed Authorization: Bearer {FAKE_KEY}"), fixture_response()])
    budget = CallBudget(2)
    response = client(config(max_retries=1), budget).generate_for_stage("test", stage="rewrite")
    assert response.content == "Bullish"
    assert len(calls) == budget.used == 2
    assert [event.attempt for event in budget.events] == [1, 2]
    assert [event.stage for event in budget.events] == ["rewrite", "rewrite"]
    assert [event.status for event in budget.events] == ["failed", "succeeded"]
    assert FAKE_KEY not in json.dumps([event.to_dict() for event in budget.events])


def test_exhausted_transport_failure_is_sanitized_and_detached(wire):
    pending, _ = wire
    pending.append(requests.Timeout(f"Authorization Bearer {FAKE_KEY}"))
    with pytest.raises(RequestError) as caught:
        client().generate("test")
    assert caught.value.retryable is True
    assert caught.value.code == "timeout"
    assert caught.value.__cause__ is None and caught.value.__context__ is None
    assert FAKE_KEY not in "".join(traceback.format_exception(caught.value))


def test_transport_exception_with_provider_response_preserves_permanent_status_and_diagnostics(wire):
    pending, calls = wire
    response = requests.Response()
    response.status_code = 401
    body = {"error": {"message": f"Rejected {FAKE_KEY}", "param": "model", "code": "authentication_error"},
            "usage": {"prompt_tokens": 4}}
    response._content = json.dumps(body).encode("utf-8")
    pending.append(requests.ConnectionError(f"Failed with {FAKE_KEY}", response=response))
    budget = CallBudget(5)
    with pytest.raises(RequestError) as caught:
        client(config(max_retries=3), budget).generate("test")
    error = caught.value
    assert error.status_code == 401 and error.retryable is False
    assert error.code == "authentication_error" and error.parameter == "model"
    assert dict(error.usage) == {"prompt_tokens": 4}
    assert FAKE_KEY not in error.detail
    assert len(calls) == budget.used == 1


def test_http_error_redacts_provider_diagnostics_and_ledger_without_retaining_response(wire):
    pending, _ = wire
    pending.append({"status_code": 422, "body": {"error": {"message": f"rejected {FAKE_KEY}",
                                                          "code": f"code-{FAKE_KEY}", "param": f"field-{FAKE_KEY}"}}})
    budget = CallBudget(1)
    with pytest.raises(RequestError) as caught:
        client(budget=budget).generate("test")
    error = caught.value
    assert FAKE_KEY not in str(error) + error.code + error.parameter + error.detail
    assert FAKE_KEY not in json.dumps([event.to_dict() for event in budget.events])
    assert error.__context__ is None and error.__cause__ is None
    assert not hasattr(error, "response")


def test_physical_send_observes_an_already_admitted_inflight_event(wire, monkeypatch):
    pending, _ = wire
    pending.append(fixture_response())
    budget = CallBudget(1)
    intercepted = requests.adapters.HTTPAdapter.send

    def assert_admitted(adapter, prepared, **kwargs):
        assert budget.used == 1
        assert budget.events[0].status == "started"
        assert budget.events[0].stage == "verification"
        return intercepted(adapter, prepared, **kwargs)

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", assert_admitted)
    client(budget=budget).generate_for_stage("test", stage="verification")
    assert budget.events[0].status == "succeeded"


def test_http_transients_retry_with_unchanged_settings_and_bounded_budget(wire):
    pending, calls = wire
    pending.extend([{"status_code": 503, "body": {"error": {"message": "busy", "code": "overloaded"}}},
                    fixture_response()])
    budget = CallBudget(1)
    with pytest.raises(BudgetExhaustedError):
        client(config(max_retries=2), budget).generate("test")
    assert len(calls) == budget.used == 1
    assert budget.events[0].status == "failed"


def test_successful_transient_retry_preserves_every_explicit_parameter(wire):
    pending, calls = wire
    pending.extend([{"status_code": 429, "body": {"error": {"message": "rate limited", "code": "rate_limit"}}},
                    fixture_response()])
    budget = CallBudget(2)
    adapter = client(config(parameters={"max_tokens": 18, "temperature": None, "custom_setting": {"mode": "explicit"}},
                            max_retries=1), budget)
    adapter.generate("test")
    assert calls[0]["body"] == calls[1]["body"]
    assert calls[1]["body"]["max_tokens"] == 18
    assert calls[1]["body"]["temperature"] is None
    assert calls[1]["body"]["custom_setting"] == {"mode": "explicit"}
    assert budget.used == 2


def test_redirect_is_not_followed_and_hidden_transport_retries_are_disabled(wire):
    pending, calls = wire
    pending.extend([{"status_code": 307, "body": {}, "headers": {"Location": "https://other.invalid/chat"}},
                    fixture_response()])
    budget = CallBudget(5)
    with pytest.raises(RequestError) as caught:
        client(config(max_retries=3), budget).generate("test")
    assert caught.value.status_code == 307
    assert len(calls) == budget.used == 1
    assert calls[0]["hidden_retries"] == 0


def test_missing_usage_stays_unknown_and_nested_usage_is_not_invented(wire):
    pending, _ = wire
    first, second = fixture_response(), fixture_response()
    del first["body"]["usage"]
    second["body"]["usage"] = {"prompt_tokens": 11, "completion_tokens_details": {"reasoning_tokens": 3}}
    pending.extend([first, second])
    adapter = client()
    assert adapter.generate("first").usage is None
    assert dict(adapter.generate("second").usage) == {"prompt_tokens": 11}


def test_only_echoed_generation_controls_are_confirmed(wire):
    pending, _ = wire
    fixture = fixture_response()
    fixture["body"]["temperature"] = 0.4
    fixture["body"]["choices"][0]["temperature"] = 0.6
    fixture["headers"] = {"x-request-id": "physical-request-id"}
    pending.append(fixture)
    response = client(config(parameters={"temperature": 0.5, "seed": 3})).generate("test")
    assert dict(response.requested_settings) == {"temperature": 0.5, "seed": 3}
    assert dict(response.confirmed_settings) == {"temperature": 0.6}
    assert response.request_id == "physical-request-id"


@pytest.mark.parametrize("evidence", [None, {"content": []}, {"content": [{"token": "Bullish", "logprob": -0.25}]}])
def test_missing_or_incomplete_evidence_is_valid_text_but_never_valid_mia(evidence, wire):
    pending, _ = wire
    fixture = fixture_response()
    fixture["body"]["choices"][0]["logprobs"] = evidence
    pending.extend([fixture, fixture])
    parameters = {"logprobs": True, "top_logprobs": 2}
    response = client(config(parameters=parameters)).generate("test")
    assert response.content == "Bullish" and response.mia_unavailable_reason
    budget = CallBudget(1)
    with pytest.raises(RequestError, match="MIA/MCS") as caught:
        client(config(parameters=parameters), budget, features=FeatureChoices(mia_scoring=True)).generate("test")
    assert caught.value.code == "missing_token_evidence"
    assert dict(caught.value.usage) == {"prompt_tokens": 8, "completion_tokens": 1, "total_tokens": 9}
    assert budget.events[0].status == "failed"
    assert caught.value.response_evidence == response
    assert caught.value.__context__ is None and caught.value.__cause__ is None


def test_missing_token_failure_retains_sanitized_complete_wire_evidence(wire):
    pending, calls = wire
    fixture = fixture_response()
    fixture["body"]["choices"][0]["message"]["content"] = f"answer-{FAKE_KEY}"
    fixture["headers"] = {"x-request-id": f"id-{FAKE_KEY}"}
    pending.append(fixture)
    budget = CallBudget(5)
    settings = config(parameters={"logprobs": True, "top_logprobs": 2}, max_retries=3)
    with pytest.raises(RequestError) as caught:
        client(settings, budget, features=FeatureChoices(mia_scoring=True)).generate("prompt")
    error = caught.value
    assert isinstance(error.response_evidence, ModelResponse)
    assert error.response_evidence.content == "answer-[REDACTED]"
    assert error.response_evidence.request_id == "id-[REDACTED]"
    assert error.response_evidence.mia_unavailable_reason is not None
    assert FAKE_KEY not in json.dumps(error.response_evidence.to_dict())
    assert error.__cause__ is None and error.__context__ is None
    assert len(calls) == budget.used == 1
    assert budget.events[0].status == "failed"
    assert budget.events[0].usage == error.response_evidence.usage
    assert not {"purpose", "request_id", "operation_id"}.intersection(budget.events[0].to_dict())


@pytest.mark.parametrize("location", ["realized", "alternative"])
def test_censored_wire_logprobs_preserve_text_but_prevent_mia_readiness(location, wire):
    pending, calls = wire
    fixture = fixture_response("openai", "token_evidence")
    entries = fixture["body"]["choices"][0]["logprobs"]["content"]
    entry = entries[0] if location == "realized" else entries[0]["top_logprobs"][1]
    entry["logprob"] = -9999.0
    pending.extend([fixture, fixture, fixture])
    parameters = fixture["request_parameters"]
    response = client(config(parameters=parameters)).generate("text only")
    assert response.content == "Bullish"
    expected_evidence = [{
        "token": entry["token"], "logprob": entry["logprob"],
        "top_logprobs": [{"token": alt["token"], "logprob": alt["logprob"]} for alt in entry["top_logprobs"]],
    } for entry in entries]
    assert response.to_dict()["token_evidence"] == expected_evidence
    assert "censored" in response.mia_unavailable_reason

    budget = CallBudget(2)
    adapter = client(config(parameters=parameters), budget, features=FeatureChoices(mia_scoring=True))
    report = adapter.probe()
    assert report.capabilities["text_generation"].status == CapabilityStatus.SUPPORTED
    expected = CapabilityStatus.UNSUPPORTED if location == "realized" else CapabilityStatus.SUPPORTED
    assert report.capabilities["logprobs"].status == expected
    assert report.capabilities["top_logprobs"].status == CapabilityStatus.UNSUPPORTED
    assert "censored" in report.capabilities["top_logprobs"].detail
    assert not report.ready_for_batch and report.mia_unavailable_reason
    with pytest.raises(RequestError, match="censored") as caught:
        adapter.generate("selected MIA")
    assert caught.value.code == "missing_token_evidence"
    assert dict(caught.value.usage) == dict(response.usage)
    assert budget.used == 2 and len(calls) == 3
    assert [event.stage for event in budget.events] == ["probe", "target"]
    assert budget.events[-1].status == "failed"


@pytest.mark.parametrize("mutation", ["no_choices", "null_content", "non_text", "bad_token"])
def test_malformed_protocol_response_is_not_silently_reinterpreted(mutation, wire):
    pending, calls = wire
    fixture = fixture_response("openai", "token_evidence")
    choice = fixture["body"]["choices"][0]
    if mutation == "no_choices":
        del fixture["body"]["choices"]
    elif mutation == "null_content":
        choice["message"] = {"content": None, "reasoning_content": "internal reasoning"}
    elif mutation == "non_text":
        choice["message"]["content"] = {"text": "not a supported chat string"}
    else:
        del choice["logprobs"]["content"][0]["logprob"]
    pending.append(fixture)
    with pytest.raises(RequestError) as caught:
        client(config(max_retries=3)).generate("test")
    assert caught.value.code == "invalid_response"
    assert caught.value.usage["total_tokens"] == 9
    assert len(calls) == 1


def test_actual_probe_counts_authenticated_generation_and_separates_acceptance_from_effective_settings(wire):
    pending, calls = wire
    pending.append(fixture_response())
    budget = CallBudget(1)
    report = client(config(parameters={"temperature": 0.2, "seed": 42}), budget).probe()
    assert len(calls) == budget.used == 1
    assert calls[0]["url"].endswith("/chat/completions")
    assert calls[0]["headers"]["Authorization"] == f"Bearer {FAKE_KEY}"
    assert budget.events[0].stage == "probe"
    assert set(report.capabilities) == {"text_generation", "temperature", "seed"}
    assert report.ready_for_batch
    assert all(e.status == CapabilityStatus.SUPPORTED and e.observed for e in report.capabilities.values())
    assert "unconfirmed" in report.capabilities["temperature"].detail.lower()
    assert "unconfirmed" in report.capabilities["seed"].detail.lower()


@pytest.mark.parametrize("content", ["", " \t\r\n"])
def test_empty_probe_text_leaves_generation_unverified_with_actionable_diagnostic(content, wire):
    pending, _ = wire
    fixture = fixture_response()
    fixture["body"]["choices"][0]["message"]["content"] = content
    fixture["body"]["choices"][0]["finish_reason"] = "length"
    pending.append(fixture)
    budget = CallBudget(1)
    report = client(config(parameters={"max_completion_tokens": 16}), budget).probe()
    capability = report.capabilities["text_generation"]
    assert capability.status == "unverified" and capability.observed is False
    assert report.ready_for_batch is False
    assert "text_generation" in report.requires_probe
    assert "length" in capability.detail
    assert "max_completion_tokens" in capability.detail and "16" in capability.detail
    assert "probe" in capability.detail.lower()
    assert budget.used == 1 and budget.events[0].status == "succeeded"


def test_nonempty_probe_text_observes_generation_support_even_when_truncated(wire):
    pending, _ = wire
    fixture = fixture_response()
    fixture["body"]["choices"][0]["finish_reason"] = "length"
    pending.append(fixture)
    report = client().probe()
    assert report.capabilities["text_generation"].status == "supported"
    assert report.capabilities["text_generation"].observed is True
    assert report.ready_for_batch


@pytest.mark.parametrize("content", ["", " \t\r\n"])
def test_ordinary_generation_retains_empty_text_and_truncation_metadata(content, wire):
    pending, _ = wire
    fixture = fixture_response()
    fixture["body"]["choices"][0]["message"]["content"] = content
    fixture["body"]["choices"][0]["finish_reason"] = "length"
    pending.append(fixture)
    response = client().generate("test")
    assert response.content == content
    assert response.finish_reason == "length"
    assert response.usage["total_tokens"] == 9


@pytest.mark.parametrize("parameters", [{"logprobs": False}, {"logprobs": None}, {"top_logprobs": 0},
                                       {"top_logprobs": None}])
def test_accepted_disabled_probability_controls_are_valid_text_only_without_claiming_mia(parameters, wire):
    pending, calls = wire
    pending.append(fixture_response())
    report = client(config(parameters=parameters)).probe()
    assert report.ready_for_batch
    assert report.mia_unavailable_reason == "MIA/MCS scoring was not selected."
    name = next(iter(parameters))
    assert report.capabilities[name].status == "supported"
    assert "accepted" in report.capabilities[name].detail.lower()
    assert "unconfirmed" in report.capabilities[name].detail.lower()
    assert calls[0]["body"][name] is parameters[name]


def test_generate_preserves_a_caller_parameter_named_stage_instead_of_consuming_it(wire):
    pending, calls = wire
    pending.append(fixture_response())
    budget = CallBudget(1)
    client(budget=budget).generate("test", stage="provider-native-field")
    assert calls[0]["body"]["stage"] == "provider-native-field"
    assert budget.events[0].stage == "target"


@pytest.mark.parametrize("response_name,missing", [("token_evidence", False), ("text_only", True)])
def test_probe_reports_observed_token_availability_without_synthetic_mia(response_name, missing, wire):
    pending, _ = wire
    pending.append(fixture_response("openai", response_name))
    report = client(config(parameters={"logprobs": True, "top_logprobs": 2}),
                    features=FeatureChoices(mia_scoring=True)).probe()
    assert report.capabilities["text_generation"].status == "supported"
    assert report.ready_for_batch is not missing
    assert bool(report.mia_unavailable_reason) is missing
    assert (report.capabilities["logprobs"].status == "unsupported") is missing
    assert (report.capabilities["top_logprobs"].status == "unsupported") is missing


def test_incomplete_probe_evidence_never_marks_top_logprobs_supported(wire):
    pending, _ = wire
    fixture = fixture_response("openai", "token_evidence")
    del fixture["body"]["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    pending.append(fixture)
    report = client(config(parameters={"logprobs": True, "top_logprobs": 2}),
                    features=FeatureChoices(mia_scoring=True)).probe()
    assert report.capabilities["logprobs"].status == "supported"
    assert report.capabilities["top_logprobs"].status == "unsupported"
    assert not report.ready_for_batch and report.mia_unavailable_reason


def test_probe_auth_error_remains_an_error_instead_of_unsupported_capability(wire):
    pending, calls = wire
    pending.append(fixture_response("openai", "unauthorized"))
    with pytest.raises(RequestError) as caught:
        client().probe()
    assert caught.value.status_code == 401 and len(calls) == 1


@pytest.mark.parametrize("reference,environment", [("TEST_PROVIDER_KEY", {}), ("TEST_PROVIDER_KEY", {"TEST_PROVIDER_KEY": ""}),
                                                  (None, {})])
def test_missing_credentials_fail_locally_without_anonymous_fallback(reference, environment, wire):
    _, calls = wire
    budget = CallBudget(1)
    with pytest.raises(ValueError, match="credential"):
        OpenAICompatibleClient(config(credential_ref=reference), budget, environment=environment)
    assert calls == [] and budget.used == 0


@pytest.mark.parametrize("credential", [
    f"{FAKE_KEY}\n", f"\t{FAKE_KEY}", "synthetic\r\nprovider-secret", "synthetic\x00provider-secret",
    "synthetic\u20acprovider-secret", "synthetic\u4e2dprovider-secret", "synthetic\u2019provider-secret",
], ids=["trailing-newline", "leading-tab", "embedded-crlf", "nul", "euro", "cjk", "right-quote"])
def test_credentials_outside_visible_ascii_are_refused_before_any_request_without_echo(credential, wire):
    _, calls = wire
    budget = CallBudget(1)
    with pytest.raises(ValueError) as caught:
        OpenAICompatibleClient(config(), budget, environment={"TEST_PROVIDER_KEY": credential})
    error = caught.value
    assert error.args == ("Referenced credential contains whitespace, control or non-ASCII characters",)
    for text in (str(error), repr(error), *map(str, error.args)):
        assert credential not in text and "synthetic" not in text
    assert error.__context__ is None and error.__cause__ is None
    assert calls == [] and budget.used == 0


def test_header_encoding_failure_is_a_context_free_transport_error_without_the_header(wire):
    pending, calls = wire
    # http.client raises this while encoding headers; its args hold the whole header value.
    header = f"Bearer {FAKE_KEY}"
    pending.append(UnicodeEncodeError("latin-1", header, 7, 8, "ordinal not in range(256)"))
    budget = CallBudget(5)
    with pytest.raises(RequestError) as caught:
        client(config(max_retries=3), budget).generate("test")
    error = caught.value
    assert error.code == "transport_error" and error.retryable is False
    assert str(error) == "OpenAI-chat request could not be encoded" and error.detail is None
    assert error.__context__ is None and error.__cause__ is None
    rendered = "".join(traceback.format_exception(error)) + repr(error) + repr(error.args)
    assert FAKE_KEY not in rendered and "Bearer" not in rendered
    assert FAKE_KEY not in json.dumps([event.to_dict() for event in budget.events])
    assert len(calls) == budget.used == 1


def test_environment_resolution_uses_supplied_mapping_or_process_environment_without_serializing_key(monkeypatch, wire):
    pending, calls = wire
    pending.append(fixture_response())
    monkeypatch.setenv("TEST_PROVIDER_KEY", FAKE_KEY)
    settings = config(credential_ref="TEST_PROVIDER_KEY")
    adapter = OpenAICompatibleClient(settings, CallBudget(1))
    adapter.generate("test")
    assert calls[0]["headers"]["Authorization"] == f"Bearer {FAKE_KEY}"
    assert FAKE_KEY not in repr(adapter) + json.dumps(settings.to_dict())


def test_anonymous_local_mode_requires_explicit_opt_in_and_records_probe_source(wire):
    pending, calls = wire
    pending.append(fixture_response("nim"))
    settings = config("nim", credential_ref=None, endpoint="http://localhost:8000/v1/chat/completions")
    adapter = client(settings, allow_unauthenticated=True)
    report = adapter.probe()
    assert "Authorization" not in calls[0]["headers"]
    assert "unauthenticated" in report.capabilities["text_generation"].source.lower()
    with pytest.raises(ValueError, match="credential"):
        OpenAICompatibleClient(config(), CallBudget(1), environment={}, allow_unauthenticated=True)


def test_protocol_is_explicit_and_wrong_protocol_rejected_locally(wire):
    _, calls = wire
    with pytest.raises(ValueError, match="openai_chat"):
        client(config(protocol="anthropic_messages"))
    assert calls == []


def test_probe_and_multiple_clients_share_the_same_budget(wire):
    pending, calls = wire
    pending.extend([fixture_response(), fixture_response()])
    budget = CallBudget(2)
    target, rewriter = client(budget=budget), client(budget=budget)
    target.probe()
    rewriter.generate_for_stage("rewrite", stage="rewrite")
    with pytest.raises(BudgetExhaustedError):
        target.generate("target")
    assert len(calls) == budget.used == 2
    assert [event.stage for event in budget.events] == ["probe", "rewrite"]

# Task 3.2 provenance fixtures use the existing HTTP transport.
def test_local_identity_roundtrips_and_legacy_omission():
    from recall_guard.core.local_evidence import LocalModelIdentity

    identity = LocalModelIdentity()
    for value in (config(), ModelResponse('ok')):
        assert 'local_identity' not in value.to_dict()
        enriched = replace(value, local_identity=identity)
        assert type(value).from_dict(enriched.to_dict()).to_dict() == enriched.to_dict()
        assert enriched.to_dict()['local_identity']['observed']['backend'] is None
        with pytest.raises(TypeError, match='local_identity'):
            replace(value, local_identity={})


def test_local_extension_and_sidecar_binding(wire):
    from recall_guard.core.local_evidence import LocalModelIdentity, LocalModelProvenance

    pending, calls = wire
    snapshot = LocalModelProvenance(artifact_digest='sha256:a', runtime='ollama', backend='cpu')
    identity = LocalModelIdentity(observed=snapshot)
    entry = fixture_response()
    entry['body']['recall_guard_local'] = identity.to_dict()
    pending.extend([entry, fixture_response()])
    adapter = client(settings=config(local_identity=LocalModelIdentity()))
    adapter.begin_local_batch(snapshot)
    assert adapter.generate('x').local_identity == identity
    assert adapter.generate('y').local_identity.observed == snapshot
    check = adapter.end_local_batch(snapshot)
    assert check.condition_matched
    assert check.unknown_fields and not check.atomic
    assert all('local_identity' not in call['body'] and 'recall_guard_local' not in call['body'] for call in calls)


@pytest.mark.parametrize('field,value', [('driver_version', '2'), ('backend', 'gpu'), ('context_length', 4096), ('template', 'new')])
def test_local_batch_reports_revalidation(field, value):
    from recall_guard.core.local_evidence import LocalModelProvenance

    before = LocalModelProvenance(artifact_digest='a', driver_version='1', backend='cpu', context_length=2048, template='old')
    adapter = client()
    adapter.begin_local_batch(before)
    check = adapter.end_local_batch(replace(before, **{field: value}))
    assert not check.condition_matched
    assert field in check.revalidation_fields


@pytest.mark.parametrize('after', [None, 'unknown'])
def test_local_batch_cannot_reestablish_identity(after):
    from recall_guard.core.local_evidence import LocalModelProvenance

    adapter = client()
    adapter.begin_local_batch(LocalModelProvenance(artifact_digest='a'))
    check = adapter.end_local_batch(None if after is None else LocalModelProvenance())
    assert not check.condition_matched


@pytest.mark.parametrize('extension', [None, {}, {'version': 2, 'declared': {}, 'observed': {}}])
def test_invalid_local_extension_is_not_silently_ignored(wire, extension):
    pending, _ = wire
    entry = fixture_response()
    entry['body']['recall_guard_local'] = extension
    pending.append(entry)
    with pytest.raises(RequestError) as error:
        client().generate('x')
    assert error.value.code == 'invalid_response'


def test_conflicting_native_and_sidecar_observation_rejected(wire):
    from recall_guard.core.local_evidence import LocalModelIdentity, LocalModelProvenance

    pending, _ = wire
    entry = fixture_response()
    entry['body']['recall_guard_local'] = LocalModelIdentity(observed=LocalModelProvenance(backend='gpu')).to_dict()
    pending.append(entry)
    adapter = client()
    adapter.begin_local_batch(LocalModelProvenance(backend='cpu'))
    with pytest.raises(RequestError) as error:
        adapter.generate('x')
    assert error.value.code == 'local_identity_mismatch'
    assert not adapter.end_local_batch(LocalModelProvenance(backend='cpu')).condition_matched


@pytest.mark.parametrize('name', ['local_identity', 'recall_guard_local'])
def test_provenance_is_not_a_generation_control(name):
    with pytest.raises(ValueError):
        config(parameters={name: {}})


def test_native_local_identity_without_sidecar_preserves_unknown_observations(wire):
    from recall_guard.core.local_evidence import LocalModelIdentity, LocalModelProvenance

    pending, _ = wire
    identity = LocalModelIdentity(declared=LocalModelProvenance(backend='npu'))
    entry = fixture_response()
    entry['body']['recall_guard_local'] = identity.to_dict()
    pending.append(entry)
    result = client().generate('x')
    assert result.local_identity == identity
    assert result.local_identity.observed.backend is None


def test_local_batch_lifecycle_and_unknowns(wire):
    from recall_guard.core.local_evidence import LocalModelProvenance

    pending, _ = wire
    adapter = client()
    with pytest.raises(ValueError, match='no local batch'):
        adapter.end_local_batch(None)
    with pytest.raises(TypeError):
        adapter.begin_local_batch({})
    adapter.begin_local_batch(LocalModelProvenance())
    with pytest.raises(ValueError, match='already active'):
        adapter.begin_local_batch(LocalModelProvenance())
    assert not adapter.end_local_batch(LocalModelProvenance()).condition_matched
    pending.append(fixture_response())
    assert adapter.generate('after batch').local_identity is None


def test_native_drift_within_sidecar_batch_is_retained(wire):
    from recall_guard.core.local_evidence import LocalModelIdentity, LocalModelProvenance

    pending, _ = wire
    for backend in ('cpu', 'gpu'):
        entry = fixture_response()
        entry['body']['recall_guard_local'] = LocalModelIdentity(
            observed=LocalModelProvenance(backend=backend)).to_dict()
        pending.append(entry)
    adapter = client()
    snapshot = LocalModelProvenance(artifact_digest='a')
    adapter.begin_local_batch(snapshot)
    assert adapter.generate('x').local_identity.observed.backend == 'cpu'
    with pytest.raises(RequestError, match='observations disagree'):
        adapter.generate('y')
    assert 'backend' in adapter.end_local_batch(snapshot).revalidation_fields


def test_local_provenance_failure_evidence_redacts_registered_secret(wire):
    from recall_guard.core.local_evidence import LocalModelIdentity, LocalModelProvenance

    pending, _ = wire
    entry = fixture_response()
    entry['body']['recall_guard_local'] = LocalModelIdentity(
        observed=LocalModelProvenance(runtime=FAKE_KEY)).to_dict()
    pending.append(entry)
    with pytest.raises(RequestError) as caught:
        client(features=FeatureChoices(mia_scoring=True)).generate('x')
    assert FAKE_KEY not in json.dumps(caught.value.response_evidence.to_dict())
    assert caught.value.response_evidence.local_identity.observed.runtime is None


def test_local_sidecar_batch_rejects_streaming_before_transport(wire):
    from recall_guard.core.local_evidence import LocalModelProvenance
    from recall_guard.core.readiness_types import ReadinessPolicy

    adapter = client(settings=replace(config(readiness=ReadinessPolicy()), provider='fixture',
                                      endpoint='http://127.0.0.1/v1/chat/completions'))
    adapter.begin_local_batch(LocalModelProvenance(backend='cpu'))
    with pytest.raises(ValueError, match='non-streaming'):
        adapter.generate('x', stream=True)
    assert adapter.budget.used == 0


def test_plain_text_http_500_body_is_retained_and_redacted(monkeypatch):
    """Ollama can answer a failed load with a non-JSON 500; the body must survive (R2.7)."""
    def send(adapter, prepared, **kwargs):
        response = requests.Response()
        response.status_code = 500
        response._content = f"llama runner process has terminated: key={FAKE_KEY}".encode()
        response.url, response.request = prepared.url, prepared
        return response

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)
    with pytest.raises(RequestError) as caught:
        client(budget=CallBudget(1)).generate("prompt")
    assert caught.value.status_code == 500
    assert caught.value.detail == "llama runner process has terminated: key=[REDACTED]"
