"""Caller callback behavior without SDKs, credentials or network calls."""

import asyncio
import gc
import traceback
from dataclasses import replace

import pytest

from recall_guard.core.completion import (
    CapabilityEvidence,
    CapabilityStatus,
    CompletionClient,
    FeatureChoices,
    ModelConfig,
    ModelResponse,
    TokenAlternative,
    TokenEvidence,
)
from recall_guard.core.provider_callback import CallbackClient
from recall_guard.core.request_control import BudgetExhaustedError, CallBudget, RequestError

SECRET = "synthetic-callback-secret"


def config(**overrides):
    return ModelConfig(**({
        "protocol": "callback", "endpoint": None, "model": "caller-model",
        "max_retries": 0, "retry_backoff_s": 0,
    } | overrides))


def client(callback, *, settings=None, budget=None, **options):
    return CallbackClient(settings or config(), budget if budget is not None else CallBudget(10), callback, **options)


def token_response():
    return ModelResponse("OK", token_evidence=(TokenEvidence("OK", -0.2, (TokenAlternative("OK", -0.2),)),),
                         usage={"output_tokens": 1}, confirmed_settings={"temperature": 0.3},
                         finish_reason="stop", request_id="callback-request")


def test_opaque_string_generation_has_unknown_physical_accounting_and_no_fabricated_evidence():
    budget = CallBudget(0)
    adapter = client(lambda prompt: "OK", budget=budget)
    response = adapter.generate("prompt")
    assert isinstance(adapter, CompletionClient)
    assert response.content == "OK"
    assert response.token_evidence is None and response.mia_unavailable_reason
    assert response.usage is None
    assert dict(response.confirmed_settings) == {}
    assert adapter.request_accounting.status == CapabilityStatus.UNVERIFIED
    assert not adapter.request_accounting.observed
    assert budget.events == ()
    assert "unknown" in adapter.request_accounting.detail.lower()


def test_typed_callback_metadata_is_preserved_with_actual_requested_settings():
    source = token_response()
    adapter = client(lambda prompt, **kwargs: source, settings=config(parameters={"temperature": 0.3}))
    response = adapter.generate("prompt")
    assert response.content == source.content and response.token_evidence == source.token_evidence
    assert response.usage == source.usage and response.finish_reason == "stop"
    assert response.request_id == "callback-request"
    assert dict(response.confirmed_settings) == {"temperature": 0.3}
    assert dict(response.requested_settings) == {"temperature": 0.3}
    assert dict(source.requested_settings) == {}


def test_parameters_are_merged_and_forwarded_without_mutating_config_or_request_snapshot():
    observed = []

    def callback(prompt, **parameters):
        observed.append((prompt, parameters.copy()))
        parameters["stop"].append("mutated")
        parameters["custom"]["items"].append(2)
        return "OK"

    settings = config(parameters={"temperature": 0.8, "stop": ["END"], "custom": {"items": [1]}})
    adapter = client(callback, settings=settings)
    response = adapter.generate("prompt", temperature=0.2, seed=12, logprobs=None)
    assert observed[0][0] == "prompt"
    assert observed[0][1]["temperature"] == 0.2 and observed[0][1]["seed"] == 12
    assert "logprobs" in observed[0][1] and observed[0][1]["logprobs"] is None
    assert settings.to_dict()["parameters"] == {"temperature": 0.8, "stop": ["END"], "custom": {"items": [1]}}
    assert response.to_dict()["requested_settings"] == {
        "temperature": 0.2, "stop": ["END"], "custom": {"items": [1]}, "seed": 12, "logprobs": None,
    }


@pytest.mark.parametrize("bad_result", [None, 42, {"content": "OK"}, ["OK"]])
def test_invalid_callback_result_is_explicit_runtime_response_failure(bad_result):
    adapter = client(lambda prompt: bad_result)
    with pytest.raises(RequestError) as caught:
        adapter.generate("prompt")
    assert caught.value.code == "invalid_response"
    assert adapter.request_accounting.status == CapabilityStatus.UNVERIFIED


def test_strict_budget_requires_explicit_hook_before_opaque_callback_invocation():
    calls = []
    with pytest.raises(ValueError, match="strict.*physical|physical.*strict"):
        client(lambda prompt: calls.append(prompt) or "OK", strict_budget=True)
    assert calls == []


def test_hook_counts_each_physical_operation_without_counting_callback_again():
    physical_calls = []

    def callback(prompt, *, request, temperature):
        first = request(lambda: physical_calls.append(prompt) or "draft")
        assert isinstance(first, ModelResponse)
        return request(lambda: physical_calls.append(first.content) or ModelResponse("answer", usage={"output_tokens": 2}))

    budget = CallBudget(2)
    adapter = client(callback, settings=config(parameters={"temperature": 0.3}), budget=budget,
                     physical_request_hook="request", strict_budget=True)
    result = adapter.generate_for_stage("prompt", stage="rewrite")
    assert result.content == "answer" and physical_calls == ["prompt", "draft"]
    assert budget.used == 2
    assert [event.stage for event in budget.events] == ["rewrite", "rewrite"]
    assert all(event.status == "succeeded" and event.cost is None for event in budget.events)
    assert budget.events[0].usage is None and dict(budget.events[1].usage) == {"output_tokens": 2}
    assert adapter.request_accounting.status == CapabilityStatus.SUPPORTED
    assert "caller" in adapter.request_accounting.source.lower()


def test_hook_retry_admission_counts_attempts_and_budget_stops_before_another_physical_call():
    calls = []

    def physical_operation():
        calls.append("physical")
        raise TimeoutError("retry")

    budget = CallBudget(2)
    adapter = client(lambda prompt, request: request(physical_operation), budget=budget,
                     settings=config(max_retries=4), physical_request_hook="request", strict_budget=True)
    with pytest.raises(BudgetExhaustedError):
        adapter.generate("prompt")
    assert calls == ["physical", "physical"]
    assert [event.attempt for event in budget.events] == [1, 2]
    assert all(event.status == "failed" for event in budget.events)


def test_hook_retries_are_bounded_without_reinvoking_entire_callback():
    logical, physical = [], []

    def operation():
        physical.append(1)
        if len(physical) == 1:
            raise TimeoutError("retry")
        return "OK"

    def callback(prompt, request):
        logical.append(prompt)
        return request(operation)

    budget = CallBudget(3)
    adapter = client(callback, settings=config(max_retries=1), budget=budget, physical_request_hook="request")
    assert adapter.generate("prompt").content == "OK"
    assert logical == ["prompt"] and physical == [1, 1]
    assert [event.attempt for event in budget.events] == [1, 2]


def test_exhausted_budget_cannot_be_suppressed_by_hooked_callback():
    def callback(prompt, request):
        try:
            request(lambda: "must not run")
        except BudgetExhaustedError:
            return "cached fallback"

    budget = CallBudget(0)
    adapter = client(callback, budget=budget, physical_request_hook="request", strict_budget=True)
    with pytest.raises(BudgetExhaustedError):
        adapter.generate("prompt")
    assert budget.events == ()


@pytest.mark.parametrize("hooked", [False, True])
def test_budget_exhaustion_is_detached_from_callback_secret_exception_context(hooked):
    def callback(prompt, **parameters):
        try:
            raise RuntimeError(SECRET)
        except RuntimeError:
            if hooked:
                return parameters["request"](lambda: "must not run")
            raise BudgetExhaustedError(f"callback budget: {SECRET}") from None

    adapter = client(callback, budget=CallBudget(0), secrets=(SECRET,),
                     physical_request_hook="request" if hooked else None)
    with pytest.raises(BudgetExhaustedError) as caught:
        adapter.generate("prompt")
    assert caught.value.__cause__ is None and caught.value.__context__ is None
    assert SECRET not in "".join(traceback.format_exception(caught.value))


def test_strict_callback_cannot_return_unaccounted_result_without_using_hook():
    adapter = client(lambda prompt, request: "unaccounted", physical_request_hook="request", strict_budget=True)
    with pytest.raises(RequestError) as caught:
        adapter.generate("prompt")
    assert caught.value.code == "unverified_request_accounting"


def test_hook_cannot_be_used_after_synchronous_callback_returns():
    retained = []

    def callback(prompt, request):
        retained.append(request)
        return request(lambda: "OK")

    budget = CallBudget(2)
    adapter = client(callback, budget=budget, physical_request_hook="request")
    adapter.generate("prompt")
    with pytest.raises(RuntimeError, match="callback|active"):
        retained[0](lambda: "late")
    assert budget.used == 1


def test_stage_parameter_is_forwarded_for_normal_generation():
    adapter = client(lambda prompt, stage: stage, settings=config(parameters={"stage": "generation-control"}))
    assert adapter.generate("prompt").content == "generation-control"


@pytest.mark.parametrize("hook", ["", " ", 17, True])
def test_invalid_hook_parameter_name_is_rejected(hook):
    with pytest.raises((ValueError, TypeError)):
        client(lambda prompt: "OK", physical_request_hook=hook)


def test_generation_parameter_cannot_overwrite_admission_hook():
    with pytest.raises(ValueError, match="request"):
        client(lambda prompt, request: "OK", physical_request_hook="request", settings=config(parameters={"request": "bad"}))
    adapter = client(lambda prompt, request: request(lambda: "OK"), physical_request_hook="request")
    with pytest.raises(ValueError, match="request"):
        adapter.generate("prompt", request="bad")


@pytest.mark.parametrize("protocol", ["openai_chat", "anthropic_messages"])
def test_callback_protocol_must_be_explicit(protocol):
    with pytest.raises(ValueError, match="callback"):
        client(lambda prompt: "OK", settings=config(protocol=protocol, endpoint="https://example.invalid/api"))


def test_unsupported_signature_parameter_is_named_without_dropping_or_retrying():
    calls = []

    def callback(prompt):
        calls.append(prompt)
        return "OK"

    adapter = client(callback, settings=config(parameters={"temperature": 0.3}, max_retries=3))
    with pytest.raises(RequestError) as caught:
        adapter.generate("prompt")
    assert caught.value.code == "callback_parameter_error" and caught.value.parameter == "temperature"
    assert calls == []


def test_internal_type_error_is_runtime_failure_not_unsupported_control():
    def callback(prompt, **parameters):
        raise TypeError("internal callback calculation failed")

    adapter = client(callback, settings=config(parameters={"temperature": 0.3}))
    with pytest.raises(RequestError) as caught:
        adapter.generate("prompt")
    assert caught.value.code == "callback_runtime_error" and caught.value.parameter is None


def test_missing_callback_argument_does_not_blame_a_supported_variadic_control():
    def callback(prompt, caller_dependency, **parameters):
        return "OK"

    adapter = client(callback, settings=config(parameters={"temperature": 0.3}))
    with pytest.raises(RequestError) as caught:
        adapter.generate("prompt")
    assert caught.value.parameter is None
    assert "caller_dependency" in str(caught.value)


def test_hook_transient_failures_stop_at_configured_retry_limit():
    calls = []

    def operation():
        calls.append(1)
        raise TimeoutError("still unavailable")

    budget = CallBudget(10)
    adapter = client(lambda prompt, request: request(operation), settings=config(max_retries=1),
                     budget=budget, physical_request_hook="request")
    with pytest.raises(RequestError) as caught:
        adapter.generate("prompt")
    assert caught.value.code == "callback_runtime_error"
    assert calls == [1, 1] and budget.used == 2


def test_custom_awaitable_result_is_runtime_incompatible_without_awaiting_it():
    awaited = []

    class CustomAwaitable:
        def __await__(self):
            awaited.append(1)
            yield
            return "OK"

    with pytest.raises(RequestError) as caught:
        client(lambda prompt: CustomAwaitable()).generate("prompt")
    assert caught.value.code == "callback_runtime_error" and awaited == []


@pytest.mark.parametrize("hooked", [False, True])
@pytest.mark.parametrize("status", [401, 403])
def test_auth_failures_are_distinguished_sanitized_and_never_retried(hooked, status):
    calls = []

    def operation():
        calls.append(1)
        raise RequestError(f"rejected {SECRET}", status_code=status, detail=f"Authorization: {SECRET}")

    def callback(prompt, **parameters):
        return parameters["request"](operation) if hooked else operation()

    budget = CallBudget(5)
    adapter = client(callback, settings=config(max_retries=3), budget=budget, secrets=(SECRET,),
                     physical_request_hook="request" if hooked else None)
    with pytest.raises(RequestError) as caught:
        adapter.generate("prompt")
    error = caught.value
    assert error.status_code == status and error.code == "callback_auth_error"
    assert calls == [1]
    assert error.__cause__ is None and error.__context__ is None
    assert SECRET not in "".join(traceback.format_exception(error))
    assert SECRET not in str(error.detail)
    assert SECRET not in repr([event.to_dict() for event in budget.events])
    assert budget.used == (1 if hooked else 0)


def test_reported_parameter_error_is_preserved_and_sanitized():
    def callback(prompt, **parameters):
        raise RequestError("unsupported setting", parameter="seed", status_code=400,
                           code="unsupported_parameter", detail=SECRET)

    with pytest.raises(RequestError) as caught:
        client(callback, secrets=(SECRET,)).generate("prompt", seed=42)
    assert caught.value.parameter == "seed" and caught.value.code == "unsupported_parameter"
    assert caught.value.detail == "[REDACTED]"


@pytest.mark.parametrize("hooked", [False, True])
def test_awaitable_callback_result_is_runtime_failure_and_closes_coroutine(hooked, recwarn):
    async def async_result():
        return "OK"

    def callback(prompt, **parameters):
        return parameters["request"](async_result) if hooked else async_result()

    adapter = client(callback, physical_request_hook="request" if hooked else None)

    async def running_kernel():
        with pytest.raises(RequestError) as caught:
            adapter.generate("prompt")
        assert caught.value.code == "callback_runtime_error"
        assert "synchronous" in str(caught.value)
        assert caught.value.parameter is None

    asyncio.run(running_kernel())
    gc.collect()
    assert not any("was never awaited" in str(w.message) for w in recwarn)


def test_async_function_callback_itself_is_runtime_incompatible(recwarn):
    async def callback(prompt):
        return "OK"

    with pytest.raises(RequestError) as caught:
        client(callback).generate("prompt")
    assert caught.value.code == "callback_runtime_error"
    gc.collect()
    assert not any("was never awaited" in str(w.message) for w in recwarn)


def test_opaque_runtime_failure_is_not_retried_or_fabricated_as_physical_call():
    calls = []

    def callback(prompt):
        calls.append(prompt)
        raise RuntimeError(f"asyncio.run cannot be called from a running event loop: {SECRET}")

    budget = CallBudget(20)
    adapter = client(callback, settings=config(max_retries=4), budget=budget, secrets=(SECRET,))
    with pytest.raises(RequestError) as caught:
        adapter.generate("prompt")
    assert caught.value.code == "callback_runtime_error"
    assert SECRET not in str(caught.value) and caught.value.__context__ is None
    assert calls == ["prompt"] and budget.events == ()


@pytest.mark.parametrize("text", ["", "  \n\t"])
def test_empty_probe_does_not_verify_text_generation(text):
    adapter = client(lambda prompt: text)
    report = adapter.probe()
    evidence = report.capabilities["text_generation"]
    assert evidence.status == CapabilityStatus.UNVERIFIED and not evidence.observed
    assert not report.ready_for_batch
    assert adapter.generate("prompt").content == text


def test_probe_records_actual_text_and_acceptance_without_confirming_unechoed_controls():
    observed = []

    def callback(prompt, **parameters):
        observed.append(parameters)
        return "OK"

    adapter = client(callback, settings=config(parameters={"temperature": 0.3}))
    report = adapter.probe()
    assert report.ready_for_batch
    assert report.capabilities["text_generation"].observed
    assert "unconfirmed" in report.capabilities["temperature"].detail
    assert observed == [{"temperature": 0.3}]
    assert adapter.request_accounting.status == CapabilityStatus.UNVERIFIED


def test_hooked_probe_and_target_share_physical_budget_and_stage_accounting():
    budget = CallBudget(2)
    adapter = client(lambda prompt, request: request(lambda: "OK"), budget=budget,
                     physical_request_hook="request", strict_budget=True)
    assert adapter.probe().ready_for_batch
    adapter.generate("target")
    with pytest.raises(BudgetExhaustedError):
        adapter.generate("extra")
    assert [event.stage for event in budget.events] == ["probe", "target"]


def test_missing_requested_mia_is_reported_by_probe_and_rejected_by_generation():
    adapter = client(lambda prompt, **parameters: "OK", settings=config(parameters={"logprobs": True, "top_logprobs": 2}),
                     features=FeatureChoices(mia_scoring=True))
    report = adapter.probe()
    assert report.capabilities["logprobs"].status == CapabilityStatus.UNSUPPORTED
    assert report.capabilities["top_logprobs"].status == CapabilityStatus.UNSUPPORTED
    assert not report.ready_for_batch and report.mia_unavailable_reason
    with pytest.raises(RequestError) as caught:
        adapter.generate("prompt")
    assert caught.value.code == "missing_token_evidence"
    assert isinstance(caught.value.response_evidence, ModelResponse)
    assert caught.value.response_evidence.content == "OK" and caught.value.response_evidence.mia_unavailable_reason
    assert caught.value.__cause__ is None and caught.value.__context__ is None


def test_missing_tokens_retain_callback_evidence_without_inventing_a_physical_event():
    budget = CallBudget(1)
    response = ModelResponse("answer", usage={"output_tokens": 2}, request_id="callback-id")
    adapter = client(lambda prompt, **parameters: response, budget=budget,
                     settings=config(parameters={"logprobs": True, "top_logprobs": 2}),
                     features=FeatureChoices(mia_scoring=True))
    with pytest.raises(RequestError) as caught:
        adapter.generate("prompt")
    assert caught.value.response_evidence.content == response.content
    assert caught.value.response_evidence.usage == response.usage
    assert caught.value.response_evidence.request_id == response.request_id
    assert caught.value.response_evidence.requested_settings == {"logprobs": True, "top_logprobs": 2}
    assert caught.value.__cause__ is None and caught.value.__context__ is None
    assert budget.used == 0


def test_genuine_optional_token_evidence_enables_probe_without_fabrication():
    adapter = client(lambda prompt, **parameters: token_response(),
                     settings=config(parameters={"temperature": 0.3, "logprobs": True, "top_logprobs": 2}),
                     features=FeatureChoices(mia_scoring=True))
    report = adapter.probe()
    assert report.ready_for_batch and report.mia_unavailable_reason is None
    assert "echoed" in report.capabilities["temperature"].detail
    assert adapter.generate("prompt").mia_unavailable_reason is None


@pytest.mark.parametrize("location", ["realized", "alternative"])
def test_censored_callback_evidence_retains_text_without_claiming_mia(location):
    response = replace(token_response(), token_evidence=(TokenEvidence(
        "OK", -9999.0 if location == "realized" else -0.2,
        (TokenAlternative("OK", -9999.0 if location == "alternative" else -0.2),),
    ),))
    settings = config(parameters={"logprobs": True, "top_logprobs": 2})

    def callback(prompt, **parameters):
        return response

    text = client(callback, settings=settings).generate("prompt")
    assert text.content == "OK" and text.token_evidence == response.token_evidence
    adapter = client(callback, settings=settings, features=FeatureChoices(mia_scoring=True))
    report = adapter.probe()
    expected = CapabilityStatus.UNSUPPORTED if location == "realized" else CapabilityStatus.SUPPORTED
    assert report.capabilities["logprobs"].status == expected
    assert report.capabilities["top_logprobs"].status == CapabilityStatus.UNSUPPORTED
    assert "censored" in report.capabilities["top_logprobs"].detail
    assert not report.ready_for_batch
    with pytest.raises(RequestError, match="censored") as caught:
        adapter.generate("prompt")
    assert caught.value.code == "missing_token_evidence"


def test_declared_unsupported_parameter_is_rejected_locally():
    settings = config(parameters={"seed": 4}, capabilities={
        "seed": CapabilityEvidence(CapabilityStatus.UNSUPPORTED, "caller documentation"),
    })
    with pytest.raises(ValueError, match="seed"):
        client(lambda prompt, **parameters: "OK", settings=settings)


def test_configuration_and_prompt_validation_happen_without_callback_invocation():
    calls = []
    with pytest.raises(TypeError):
        CallbackClient(config(), CallBudget(1), 42)
    adapter = client(lambda prompt: calls.append(prompt) or "OK")
    with pytest.raises(TypeError, match="prompt"):
        adapter.generate(42)
    with pytest.raises(ValueError):
        adapter.generate("prompt", temperature=-1)
    with pytest.raises(TypeError):
        client(lambda prompt: "OK", strict_budget="yes")
    assert calls == []


def test_callback_credential_reference_is_not_resolved_or_forwarded(monkeypatch):
    monkeypatch.setenv("CALLBACK_TEST_CREDENTIAL", SECRET)
    received = []
    settings = replace(config(), credential_ref="env:CALLBACK_TEST_CREDENTIAL")
    adapter = client(lambda prompt, **parameters: received.append(parameters) or "OK", settings=settings)
    adapter.generate("prompt")
    assert received == [{}]
    assert SECRET not in repr(settings.to_dict())
