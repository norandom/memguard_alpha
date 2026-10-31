"""Replay native Messages fixtures at the physical requests transport boundary."""

import asyncio
import json
import traceback
from dataclasses import replace
from pathlib import Path

import pytest
import requests

from recall_guard.core.completion import (
    CapabilityEvidence,
    CompletionClient,
    FeatureChoices,
    ModelConfig,
)
from recall_guard.core.provider_anthropic import AnthropicClient, validate_anthropic_config
from recall_guard.core.request_control import BudgetExhaustedError, CallBudget, RequestError

FIXTURE = Path(__file__).parents[1] / "fixtures" / "providers" / "anthropic.json"
FAKE_KEY = "synthetic-anthropic-secret"


def fixture_response(name="text_only"):
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["responses"][name]


def test_probe_explicit_prompt_and_stage_are_provenance_not_body_controls(wire):
    pending, calls = wire
    pending.append(fixture_response())
    budget = CallBudget(1)
    adapter = client(budget=budget)
    adapter.probe("EXACT_NATIVE_PROBE_INPUT", stage="execution/probe/target")
    assert budget.events[0].stage == "execution/probe/target"
    assert calls[0]["body"]["messages"] == [{"role": "user", "content": "EXACT_NATIVE_PROBE_INPUT"}]
    assert "stage" not in calls[0]["body"]


def config(**overrides):
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return ModelConfig(**({
        "protocol": "anthropic_messages", "provider": "anthropic",
        "endpoint": fixture["request"]["endpoint"], "model": fixture["request"]["body"]["model"],
        "credential_ref": "env:TEST_ANTHROPIC_KEY", "parameters": {"max_tokens": 16},
        "max_retries": 0, "retry_backoff_s": 0,
    } | overrides))


def client(settings=None, budget=None, **options):
    return AnthropicClient(settings or config(), budget or CallBudget(10),
                           environment={"TEST_ANTHROPIC_KEY": FAKE_KEY}, **options)


def test_public_configuration_validation_is_pure_and_shared_with_constructor(monkeypatch):
    class NoEnvironment(dict):
        def get(self, *args, **kwargs):
            raise AssertionError("Configuration validation must not resolve credentials")

    monkeypatch.setattr("recall_guard.core.provider_anthropic.os.environ", NoEnvironment())
    assert validate_anthropic_config(config()) is None
    invalid = config(parameters={"max_tokens": 16, "temperature": 0})
    for operation in (lambda: validate_anthropic_config(invalid), lambda: AnthropicClient(invalid, CallBudget(0))):
        with pytest.raises(ValueError, match="temperature"):
            operation()


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


@pytest.mark.parametrize("name,finish", [("text_only", "end_turn"), ("truncated", "max_tokens")])
def test_native_fixtures_preserve_metadata_without_fabricated_token_evidence(name, finish, wire):
    pending, calls = wire
    pending.append(fixture_response(name))
    budget = CallBudget(1)
    settings = config()
    adapter = client(settings, budget)
    assert isinstance(adapter, CompletionClient)
    assert calls == []
    response = adapter.generate("Return Bullish.")
    assert response.content == "Bullish" and response.finish_reason == finish
    assert response.token_evidence is None and response.mia_unavailable_reason
    assert response.request_id == "fixture-message"
    assert dict(response.usage) == {"input_tokens": 8, "output_tokens": 1}
    assert dict(response.requested_settings) == {"max_tokens": 16}
    assert dict(response.confirmed_settings) == {}
    assert calls[0]["body"] == {"model": settings.model, "max_tokens": 16,
                                  "messages": [{"role": "user", "content": "Return Bullish."}]}
    assert calls[0]["url"] == settings.endpoint
    assert calls[0]["headers"]["anthropic-version"] == "2026-10-21"
    assert calls[0]["headers"]["x-api-key"] == FAKE_KEY
    assert "Authorization" not in calls[0]["headers"]
    assert calls[0]["timeout"] == settings.timeout_s and calls[0]["hidden_retries"] == 0
    assert budget.used == 1 and budget.events[0].status == "succeeded"
    assert dict(budget.events[0].usage) == dict(response.usage)
    assert budget.events[0].latency_s >= 0 and budget.events[0].cost is None


def test_system_is_separate_and_explicit_overrides_do_not_mutate_config(wire):
    pending, calls = wire
    pending.extend([fixture_response(), fixture_response()])
    settings = config(parameters={"max_tokens": 32, "system": "Keep the required format.", "stop_sequences": ["END"]})
    adapter = client(settings)
    response = adapter.generate("source evidence", max_tokens=64, system=[{"type": "text", "text": "Override"}])
    adapter.generate("second evidence")
    assert calls[0]["body"]["system"] == [{"type": "text", "text": "Override"}]
    assert calls[0]["body"]["messages"] == [{"role": "user", "content": "source evidence"}]
    assert calls[0]["body"]["max_tokens"] == 64 and calls[0]["body"]["stop_sequences"] == ["END"]
    assert calls[1]["body"]["system"] == "Keep the required format."
    assert calls[1]["body"]["max_tokens"] == 32
    assert response.requested_settings["max_tokens"] == 64 and settings.parameters["max_tokens"] == 32


def test_typed_content_joins_only_actual_text_and_skips_thinking_and_other_blocks(wire):
    pending, _ = wire
    fixture = fixture_response()
    fixture["body"]["content"] = [
        {"type": "thinking", "thinking": "private reasoning", "text": "Do not expose this"},
        {"type": "text", "text": "Bull"},
        {"type": "redacted_thinking", "data": "redacted", "text": "Not answer text"},
        {"type": "tool_use", "name": "search", "input": {"text": "Also not answer text"}},
        {"type": "text", "text": "ish"},
    ]
    pending.append(fixture)
    assert client().generate("test").content == "Bullish"


@pytest.mark.parametrize("content", [[], [{"type": "thinking", "thinking": "reasoning"}],
                                      [{"type": "text", "text": ""}], [{"type": "text", "text": " \t\r\n"}]])
def test_empty_or_thinking_only_probe_leaves_text_unverified_and_preserves_truncation(content, wire):
    pending, _ = wire
    fixture = fixture_response("truncated")
    fixture["body"]["content"] = content
    pending.extend([fixture, fixture])
    budget = CallBudget(2)
    adapter = client(budget=budget)
    ordinary = adapter.generate("test")
    assert ordinary.content == (content[0]["text"] if content and content[0]["type"] == "text" else "")
    assert ordinary.finish_reason == "max_tokens" and ordinary.usage["output_tokens"] == 1
    report = adapter.probe()
    capability = report.capabilities["text_generation"]
    assert capability.status == "unverified" and capability.observed is False
    assert not report.ready_for_batch and "text_generation" in report.requires_probe
    assert "max_tokens" in capability.detail and "16" in capability.detail and "probe" in capability.detail.lower()
    assert budget.used == 2 and all(event.status == "succeeded" for event in budget.events)


def test_access_probe_uses_authenticated_generation_with_observed_truncated_text(wire):
    pending, calls = wire
    pending.append(fixture_response("truncated"))
    budget = CallBudget(1)
    adapter = client(budget=budget, features=FeatureChoices(ensemble_draws=3, rewriting=True))
    assert adapter.features == FeatureChoices(ensemble_draws=3, rewriting=True)
    assert not adapter.config.inspect(adapter.features).ready_for_batch
    report = adapter.probe()
    assert len(calls) == budget.used == 1 and budget.events[0].stage == "probe"
    assert calls[0]["url"].endswith("/messages") and calls[0]["body"]["messages"][0]["role"] == "user"
    assert report.ready_for_batch and report.features == adapter.features
    assert set(report.capabilities) == {"text_generation", "max_tokens"}
    assert all(e.status == "supported" and e.observed for e in report.capabilities.values())
    assert "2026-10-21" in report.capabilities["text_generation"].source
    assert "unconfirmed" in report.capabilities["max_tokens"].detail.lower()
    assert report.mia_unavailable_reason


@pytest.mark.parametrize("name,value", [("temperature", 0.3), ("top_p", 0.9), ("seed", 1), ("logprobs", True),
                                      ("top_logprobs", 2), ("max_completion_tokens", 32), ("random_seed", 1),
                                      ("temperature", None), ("logprobs", False), ("top_logprobs", None)])
def test_audited_unsupported_parameters_are_rejected_explicitly_even_when_declared_supported(name, value, wire):
    _, calls = wire
    budget = CallBudget(1)
    settings = config(capabilities={name: CapabilityEvidence("supported", "caller declaration", observed=True)})
    adapter = client(settings, budget)
    with pytest.raises(ValueError, match=name):
        adapter.generate("test", **{name: value})
    with pytest.raises(ValueError, match=name):
        client(replace(settings, parameters={"max_tokens": 16, name: value}), budget)
    assert calls == [] and budget.used == 0


@pytest.mark.parametrize("parameters", [{}, {"max_tokens": None}])
def test_native_required_max_tokens_is_explicit_and_not_replaced_with_a_hidden_default(parameters, wire):
    _, calls = wire
    budget = CallBudget(1)
    with pytest.raises(ValueError, match="max_tokens"):
        client(config(parameters=parameters), budget)
    assert calls == [] and budget.used == 0


def test_mia_selection_is_incompatible_before_any_probe_even_with_forged_declarations(wire):
    _, calls = wire
    budget = CallBudget(1)
    evidence = CapabilityEvidence("supported", "caller declaration", observed=True)
    with pytest.raises(ValueError, match="MIA/MCS"):
        client(config(capabilities={"logprobs": evidence, "top_logprobs": evidence}), budget,
               features=FeatureChoices(mia_scoring=True))
    assert calls == [] and budget.used == 0


@pytest.mark.parametrize("name", ["unauthorized", "forbidden", "unsupported_parameter"])
def test_native_fixture_failures_keep_diagnostics_and_explicit_parameter_without_retry(name, wire):
    pending, calls = wire
    fixture = fixture_response(name)
    pending.append(fixture)
    budget = CallBudget(4)
    requested = {"max_tokens": 16} | fixture.get("request_parameters", {})
    with pytest.raises(RequestError) as caught:
        client(config(parameters=requested, max_retries=3), budget).generate("test")
    error = caught.value
    assert error.status_code == fixture["status_code"]
    assert error.code == fixture["body"]["error"]["type"]
    assert json.loads(error.detail) == fixture["body"]
    assert not error.retryable and len(calls) == budget.used == 1
    assert budget.events[0].status == "failed" and budget.events[0].code == error.code
    if name == "unsupported_parameter":
        assert fixture["rejected_parameter"] in str(error)
        assert calls[0]["body"][fixture["rejected_parameter"]] is True


@pytest.mark.parametrize("status,code,message,parameter", [
    (401, "authentication_error", "Invalid credentials", None),
    (403, "permission_error", "Model access forbidden", None),
    (400, "invalid_request_error", "Your credit balance is too low", None),
    (404, "not_found_error", "Unknown model", "model"),
    (400, "invalid_request_error", "Unsupported custom_field", "custom_field"),
    (429, "insufficient_quota", "No remaining credits", None),
])
def test_auth_billing_model_and_parameter_errors_are_distinct_permanent_failures(status, code, message, parameter, wire):
    pending, calls = wire
    pending.append({"status_code": status, "body": {"type": "error", "error": {
        "type": code, "message": message, "parameter": parameter}, "usage": {"input_tokens": 3}}})
    budget = CallBudget(4)
    with pytest.raises(RequestError) as caught:
        client(config(parameters={"max_tokens": 16, "custom_field": True}, max_retries=3), budget).generate("test")
    assert caught.value.status_code == status and caught.value.code == code
    assert message in str(caught.value) and caught.value.parameter == parameter
    assert dict(caught.value.usage) == {"input_tokens": 3}
    assert dict(budget.events[0].usage) == {"input_tokens": 3}
    assert len(calls) == budget.used == 1


@pytest.mark.parametrize("failure", [requests.Timeout(f"Failed x-api-key: {FAKE_KEY}"),
                                    requests.ConnectionError("network failure"),
                                    {"status_code": 529, "body": {"error": {"type": "overloaded_error"}}}])
def test_only_admitted_transient_retries_repeat_unchanged_native_payload(failure, wire):
    pending, calls = wire
    pending.extend([failure, fixture_response()])
    budget = CallBudget(2)
    response = client(config(max_retries=1), budget).generate_for_stage("test", stage="rewrite")
    assert response.content == "Bullish" and calls[0]["body"] == calls[1]["body"]
    assert len(calls) == budget.used == 2 and [e.attempt for e in budget.events] == [1, 2]
    assert [e.stage for e in budget.events] == ["rewrite", "rewrite"]
    assert [e.status for e in budget.events] == ["failed", "succeeded"]
    assert FAKE_KEY not in json.dumps([event.to_dict() for event in budget.events])


def test_probe_and_target_clients_share_physical_budget_including_failed_attempts(wire):
    pending, calls = wire
    pending.extend([fixture_response(), requests.Timeout("timed out"), fixture_response()])
    budget = CallBudget(2)
    client(budget=budget).probe()
    adapter = client(config(max_retries=1), budget)
    with pytest.raises(BudgetExhaustedError):
        adapter.generate("test")
    assert len(calls) == budget.used == 2
    assert [e.stage for e in budget.events] == ["probe", "target"]


def test_physical_send_requires_an_already_admitted_event(wire, monkeypatch):
    pending, _ = wire
    pending.append(fixture_response())
    budget = CallBudget(1)
    intercepted = requests.adapters.HTTPAdapter.send

    def assert_admitted(adapter, prepared, **kwargs):
        assert budget.used == 1 and budget.events[0].status == "started"
        assert budget.events[0].stage == "verification"
        return intercepted(adapter, prepared, **kwargs)

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", assert_admitted)
    client(budget=budget).generate_for_stage("test", stage="verification")


def test_redirects_and_ambient_auth_are_disabled(wire, monkeypatch):
    pending, calls = wire
    pending.extend([{"status_code": 307, "body": {}, "headers": {"Location": "https://other.invalid/messages"}},
                    fixture_response()])
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid")
    intercepted = requests.Session.send

    def check_session(session, prepared, **kwargs):
        assert session.trust_env is False and kwargs["allow_redirects"] is False
        return intercepted(session, prepared, **kwargs)

    monkeypatch.setattr(requests.Session, "send", check_session)
    budget = CallBudget(4)
    with pytest.raises(RequestError) as caught:
        client(config(max_retries=3), budget).generate("test")
    assert caught.value.status_code == 307 and len(calls) == budget.used == 1
    assert calls[0]["hidden_retries"] == 0 and "Authorization" not in calls[0]["headers"]


@pytest.mark.parametrize("failure", [requests.Timeout(f"Failed x-api-key: {FAKE_KEY}"),
                                    {"status_code": 400, "body": {"error": {"type": f"type-{FAKE_KEY}",
                                     "message": f"Rejected {FAKE_KEY}", "param": f"field-{FAKE_KEY}"}}}])
def test_error_and_ledger_redact_known_credentials_without_retaining_original_exception(failure, wire):
    pending, _ = wire
    pending.append(failure)
    budget = CallBudget(1)
    with pytest.raises(RequestError) as caught:
        client(budget=budget).generate("test")
    error = caught.value
    assert error.__cause__ is None and error.__context__ is None and not hasattr(error, "response")
    assert FAKE_KEY not in "".join(traceback.format_exception(error)) + str(error.detail) + str(error.parameter) + str(error.code)
    assert FAKE_KEY not in json.dumps([event.to_dict() for event in budget.events])


def test_transport_exception_retains_attached_auth_failure_instead_of_retrying(wire):
    pending, calls = wire
    attached = requests.Response()
    attached.status_code = 401
    attached._content = json.dumps(fixture_response("unauthorized")["body"]).encode("utf-8")
    pending.append(requests.ConnectionError(f"Failed {FAKE_KEY}", response=attached))
    with pytest.raises(RequestError) as caught:
        client(config(max_retries=3)).generate("test")
    assert caught.value.status_code == 401 and caught.value.code == "authentication_error"
    assert not caught.value.retryable and len(calls) == 1


def test_nested_cache_and_thinking_usage_is_not_flattened_or_estimated(wire):
    pending, _ = wire
    fixture = fixture_response()
    fixture["body"]["usage"] = {
        "input_tokens": 11, "output_tokens": 8, "cache_creation_input_tokens": 4, "cache_read_input_tokens": 3,
        "cache_creation": {"ephemeral_5m_input_tokens": 4}, "thinking": {"output_tokens": 6},
        "server_tool_use": {"web_search_requests": 0}, "service_tier": "standard", "invalid": -1, "boolean": True,
    }
    pending.append(fixture)
    response = client().generate("test")
    assert dict(response.usage) == {"input_tokens": 11, "output_tokens": 8,
                                   "cache_creation_input_tokens": 4, "cache_read_input_tokens": 3}
    assert "total_tokens" not in response.usage and "thinking_tokens" not in response.usage


@pytest.mark.parametrize("usage", [None, "missing"])
def test_missing_usage_remains_unknown(usage, wire):
    pending, _ = wire
    fixture = fixture_response()
    if usage == "missing":
        del fixture["body"]["usage"]
    else:
        fixture["body"]["usage"] = usage
    pending.append(fixture)
    budget = CallBudget(1)
    assert client(budget=budget).generate("test").usage is None
    assert budget.events[0].usage is None and budget.events[0].cost is None


@pytest.mark.parametrize("content", [None, "OpenAI-like string", ["untyped"], [{"text": "no type"}],
                                      [{"type": "text", "text": None}], [{"type": "text"}]])
def test_malformed_native_content_is_an_invalid_response_not_a_capability_rejection(content, wire):
    pending, calls = wire
    fixture = fixture_response()
    fixture["body"]["content"] = content
    pending.append(fixture)
    with pytest.raises(RequestError) as caught:
        client(config(max_retries=3)).generate("test")
    assert caught.value.code == "invalid_response" and caught.value.status_code is None
    assert dict(caught.value.usage) == {"input_tokens": 8, "output_tokens": 1}
    assert len(calls) == 1


def test_openai_shaped_success_is_not_reinterpreted_as_native_anthropic(wire):
    pending, _ = wire
    pending.append({"status_code": 200, "body": {"choices": [{"message": {"content": "Bullish"}}]}})
    with pytest.raises(RequestError, match="Anthropic") as caught:
        client().generate("test")
    assert caught.value.code == "invalid_response"


def test_request_identity_prefers_native_request_header_and_no_settings_are_invented(wire):
    pending, _ = wire
    fixture = fixture_response()
    fixture["headers"] = {"request-id": "native-request-id"}
    pending.append(fixture)
    response = client().generate("test")
    assert response.request_id == "native-request-id" and dict(response.confirmed_settings) == {}


@pytest.mark.parametrize("reference,environment", [(None, {}), ("TEST_ANTHROPIC_KEY", {}),
                                                  ("TEST_ANTHROPIC_KEY", {"TEST_ANTHROPIC_KEY": " \t"})])
def test_missing_credentials_fail_before_admission_without_anonymous_fallback(reference, environment, wire):
    _, calls = wire
    budget = CallBudget(1)
    with pytest.raises(ValueError, match="credential"):
        AnthropicClient(config(credential_ref=reference), budget, environment=environment)
    assert calls == [] and budget.used == 0


def test_resolves_only_runtime_reference_and_never_exports_credential(monkeypatch, wire):
    pending, calls = wire
    pending.append(fixture_response())
    monkeypatch.setenv("TEST_ANTHROPIC_KEY", FAKE_KEY)
    settings = config(credential_ref="TEST_ANTHROPIC_KEY")
    adapter = AnthropicClient(settings, CallBudget(1))
    adapter.generate("test")
    assert calls[0]["headers"]["x-api-key"] == FAKE_KEY
    assert FAKE_KEY not in repr(adapter) + json.dumps(settings.to_dict())


def test_wrong_protocol_or_declared_unavailable_capability_fails_locally(wire):
    _, calls = wire
    with pytest.raises(ValueError, match="anthropic_messages"):
        client(config(protocol="openai_chat"))
    with pytest.raises(ValueError, match="text_generation"):
        client(config(capabilities={"text_generation": CapabilityEvidence("unsupported", "test")}))
    assert calls == []


def test_unsupported_error_is_not_used_for_a_runtime_constraint(wire):
    pending, _ = wire
    pending.append(RuntimeError("kernel network access disabled"))
    with pytest.raises(RequestError, match="kernel network access disabled") as caught:
        client().generate("test")
    assert caught.value.parameter is None and caught.value.code is None and caught.value.status_code is None


def test_synchronous_generation_works_inside_running_notebook_event_loop(wire):
    pending, _ = wire
    pending.append(fixture_response())
    adapter = client()

    async def notebook_cell():
        return adapter.generate("test")

    assert asyncio.run(notebook_cell()).content == "Bullish"


def test_provider_parameter_named_stage_is_not_consumed_as_ledger_stage(wire):
    pending, calls = wire
    pending.append(fixture_response())
    budget = CallBudget(1)
    client(budget=budget).generate("test", stage="provider-native-field")
    assert calls[0]["body"]["stage"] == "provider-native-field"
    assert budget.events[0].stage == "target"
