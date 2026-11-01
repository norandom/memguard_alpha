"""Offline behavioral contracts for optional provider readiness."""

import json
from dataclasses import FrozenInstanceError, replace
from threading import Thread

import pytest

from recall_guard.core import readiness_types as r
from recall_guard.core.completion import (
    CapabilityEvidence,
    CapabilityReport,
    FeatureChoices,
    ModelConfig,
    ModelResponse,
    TokenAlternative,
    TokenEvidence,
)

STAMP = "2026-11-01T12:00:00+00:00"
REQUEST_ID = "12345678-1234-1234-1234-123456789abc"


def observation(**changes):
    return r.ReadinessObservation(**({
        "operation_id": "operation-1", "stage": "probe", "model": "selected-model",
        "timestamp": STAMP, "elapsed_s": 0.0, "state": "waiting",
    } | changes))


def pending_policy(**changes):
    return r.PendingStatusPolicy(**({
        "profile": "verified-fixture", "model": "selected-model",
        "endpoint": "https://example.invalid/v1/chat/completions",
        "status_url_template": "https://example.invalid/v1/status/{request_id}",
        "request_id_header": "NVCF-REQID", "verified": True,
    } | changes))


def test_policy_is_immutable_and_has_independent_limits_without_duplicate_caps():
    policy = r.ReadinessPolicy(connect_timeout_s=2, socket_idle_timeout_s=7, max_wait_s=30)
    assert (policy.connect_timeout_s, policy.socket_idle_timeout_s, policy.max_wait_s) == (2, 7, 30)
    assert not policy.allow_ambiguous_replay
    assert policy.pending_status is None
    assert policy.max_poll_requests == 0
    with pytest.raises(FrozenInstanceError):
        policy.max_wait_s = 90
    exported = json.loads(json.dumps(policy.to_dict(), allow_nan=False))
    assert not {"max_retries", "min_call_interval_s", "request_budget", "cancellation", "on_observation"} & exported.keys()
    assert exported["pending_status"] is None


@pytest.mark.parametrize(("name", "value"), [
    ("connect_timeout_s", 0), ("socket_idle_timeout_s", -1), ("max_wait_s", float("nan")),
    ("connect_timeout_s", float("inf")), ("max_wait_s", True), ("max_poll_requests", -1),
    ("max_poll_requests", 1.5), ("max_poll_requests", False), ("backoff_initial_s", -1),
    ("backoff_max_s", float("inf")), ("jitter_fraction", 1.1), ("jitter_fraction", float("nan")),
    ("allow_ambiguous_replay", 1), ("max_event_bytes", 0), ("max_response_bytes", True),
    ("connect_timeout_s", 10**1000), ("pending_status", {}),
])
def test_invalid_policy_fails_locally(name, value):
    with pytest.raises((TypeError, ValueError), match=name):
        r.ReadinessPolicy(**{name: value})


def test_policy_rejects_contradictory_bounds_and_unconfigured_polling():
    for changes in ({"backoff_initial_s": 5, "backoff_max_s": 2},
                    {"max_event_bytes": 20, "max_response_bytes": 10}, {"max_poll_requests": 1}):
        with pytest.raises(ValueError):
            r.ReadinessPolicy(**changes)
    policy = r.ReadinessPolicy(pending_status=pending_policy(), max_poll_requests=3)
    assert policy.to_dict()["pending_status"]["model"] == "selected-model"


@pytest.mark.parametrize("changes", [
    {"verified": False}, {"verified": 1}, {"model": ""},
    {"status_url_template": "https://evil.invalid/status/{request_id}"},
    {"status_url_template": "http://example.invalid/status/{request_id}"},
    {"status_url_template": "https://example.invalid:444/status/{request_id}"},
    {"status_url_template": "https://example.invalid:0/status/{request_id}"},
    {"status_url_template": "https://user:secret@example.invalid/status/{request_id}"},
    {"status_url_template": "https://example.invalid/status/{request_id}?key=secret"},
    {"status_url_template": "https://example.invalid/status/{request_id}#secret"},
    {"status_url_template": "https://example.invalid/status/no-placeholder"},
    {"status_url_template": "https://example.invalid/{other}/{request_id}"},
    {"request_id_header": "Authorization"}, {"request_id_header": "X\r\nsecret"},
])
def test_pending_policy_rejects_unverified_or_unsafe_identity_contracts(changes):
    with pytest.raises((TypeError, ValueError)) as error:
        pending_policy(**changes)
    assert "secret" not in str(error.value)


def test_pending_url_is_bound_to_exact_configuration_and_uuid():
    policy = pending_policy()
    assert policy.status_url(REQUEST_ID, model="selected-model", endpoint=policy.endpoint).endswith(REQUEST_ID)
    for request_id in ("../../secret", "not-uuid", "", REQUEST_ID + "?secret"):
        with pytest.raises(ValueError):
            policy.status_url(request_id, model="selected-model", endpoint=policy.endpoint)
    with pytest.raises(ValueError, match="configuration"):
        policy.status_url(REQUEST_ID, model="other-model", endpoint=policy.endpoint)
    with pytest.raises(ValueError, match="configuration"):
        policy.status_url(REQUEST_ID, model=policy.model, endpoint="https://example.invalid/other")


def test_pending_receipt_retains_identity_without_claiming_completion():
    receipt = r.PendingReceipt("operation-1", "selected-model", REQUEST_ID, STAMP, provider_status="pending")
    assert receipt.complete is False
    assert receipt.to_dict()["status_code"] == 202
    assert receipt.to_dict()["request_id"] == REQUEST_ID
    with pytest.raises(ValueError):
        replace(receipt, request_id="unvalidated")


def test_observations_separate_local_wait_answer_reasoning_and_network_activity():
    waiting = observation(elapsed_s=12, waiting_s=9, transport_bytes=128, keepalive_events=2)
    progress = replace(waiting, state="progress", answer_characters=4, answer_deltas=1,
                       reasoning_characters=10, reasoning_deltas=2)
    assert waiting.state == "waiting" and waiting.answer_characters == waiting.reasoning_characters == 0
    assert progress.to_dict()["reasoning_characters"] == 10
    assert progress.to_dict()["waiting_s"] == 9
    with pytest.raises(FrozenInstanceError):
        progress.state = "ready"
    with pytest.raises(ValueError, match="progress"):
        observation(state="progress", transport_bytes=50, keepalive_events=1)


@pytest.mark.parametrize("changes", [
    {"elapsed_s": float("nan")}, {"waiting_s": -1}, {"waiting_s": 2, "elapsed_s": 1},
    {"timestamp": "2026-11-01T12:00:00"}, {"timestamp": "2026-11-01T12:00:00+01:00"},
    {"state": "invented"}, {"call_id": 0}, {"status_code": 99}, {"status_code": True},
    {"answer_characters": -1}, {"reasoning_deltas": True}, {"delay_s": 3},
    {"delay_source": "backoff"}, {"delay_s": float("inf"), "delay_source": "backoff"},
    {"state": "pending"}, {"state": "ready", "status_code": 202},
])
def test_invalid_observation_evidence_is_rejected(changes):
    with pytest.raises((TypeError, ValueError)):
        observation(**changes)


def test_unknown_gateway_outcome_does_not_reject_credentials():
    unknown = observation(state="unknown_outcome", status_code=504, reason="gateway_timeout")
    assert unknown.credential_eligibility == "unverified"
    assert unknown.to_dict()["status_code"] == 504
    with pytest.raises(ValueError, match="credential"):
        replace(unknown, credential_eligibility="rejected")
    rejected = observation(state="rejected", status_code=401, reason="authentication_rejected",
                           credential_eligibility="rejected")
    assert rejected.credential_eligibility == "rejected"


def test_observation_serialization_is_a_whitelist_not_runtime_metadata():
    event = observation(delay_s=2, delay_source="retry_after")
    serialized = json.loads(json.dumps(event.to_dict(), allow_nan=False))
    assert serialized["delay_source"] == "retry_after"
    for name in ("headers", "authorization", "credential_ref", "on_observation", "cancellation"):
        with pytest.raises(TypeError):
            observation(**{name: "secret"})
        assert name not in serialized


def test_cancellation_is_thread_safe_cooperative_and_runtime_only():
    token = r.CancellationToken()
    assert not token.cancelled
    assert token.wait(0) is False
    thread = Thread(target=token.cancel)
    thread.start()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert token.cancelled and token.wait(0)
    token.cancel()
    with pytest.raises(TypeError):
        json.dumps(token)
    with pytest.raises(ValueError):
        token.wait(float("inf"))


def test_partial_response_retains_real_evidence_but_never_becomes_usable():
    from recall_guard.core.readiness_evidence import PartialResponse

    entries = [TokenEvidence("answer", -0.2, (TokenAlternative("answer", -0.2),))]
    usage = {"output_tokens": 1}
    partial = PartialResponse(content="answer", token_evidence=entries, usage=usage,
                              reason="premature_eof", reasoning_characters=5)
    entries.clear()
    usage["output_tokens"] = 99
    assert not isinstance(partial, ModelResponse)
    assert partial.complete is False and partial.usable_answer is False
    assert "incomplete" in partial.mia_unavailable_reason.lower()
    assert len(partial.token_evidence) == 1 and partial.usage["output_tokens"] == 1
    exported = partial.to_dict()
    assert exported["complete"] is False and exported["reasoning_characters"] == 5
    assert exported["token_evidence"][0]["logprob"] == -0.2
    with pytest.raises(ValueError):
        PartialResponse("x" * 10, max_retained_bytes=5)
    with pytest.raises(ValueError):
        PartialResponse("answer", usage={"output_tokens": -1})


def test_probe_preserves_complete_text_missing_required_tokens_for_fallback():
    from recall_guard.core.readiness_evidence import ProbeEvidence

    config = ModelConfig("openai_chat", "https://example.invalid/chat", "selected-model",
                         credential_ref="PRIVATE_ENV_REFERENCE", parameters={"temperature": 0.2})
    report = CapabilityReport({"text_generation": CapabilityEvidence("supported", "probe", observed=True),
                               "logprobs": CapabilityEvidence("unsupported", "probe", observed=True)},
                              FeatureChoices(mia_scoring=True))
    response = ModelResponse("answer")
    probe = ProbeEvidence(config, report, response, [observation(state="deferred", reason="missing_token_evidence")])
    assert probe.complete and not probe.ready and probe.response is response
    assert probe.config.credential_ref is None
    assert probe.report.mia_unavailable_reason
    exported = json.loads(json.dumps(probe.to_dict(), allow_nan=False))
    assert "PRIVATE_ENV_REFERENCE" not in json.dumps(exported)
    assert exported["config"]["parameters"] == {"temperature": 0.2}
    assert exported["response"]["complete"] is True


def test_probe_ready_requires_complete_answer_and_observed_required_capabilities():
    from recall_guard.core.readiness_evidence import PartialResponse, ProbeEvidence

    config = ModelConfig("callback", None, "selected-model")
    report = CapabilityReport({"text_generation": CapabilityEvidence("supported", "probe", observed=True)})
    trace = [observation(state="ready", elapsed_s=2)]
    probe = ProbeEvidence(config, report, ModelResponse("answer"), trace)
    trace.clear()
    assert probe.ready and probe.observations[0].timestamp == STAMP
    assert probe.to_dict()["report"]["capabilities"]["text_generation"]["observed"]
    for response in (None, PartialResponse("answer")):
        with pytest.raises(ValueError, match="ready"):
            ProbeEvidence(config, report, response, probe.observations)
    with pytest.raises(ValueError, match="model"):
        ProbeEvidence(replace(config, model="different"), report, ModelResponse("answer"), probe.observations)
    unknown = CapabilityReport({"text_generation": CapabilityEvidence("unverified", "not observed")})
    with pytest.raises(ValueError, match="ready"):
        ProbeEvidence(config, unknown, ModelResponse("answer"), probe.observations)


@pytest.mark.parametrize(("report", "response"), [
    (CapabilityReport({}), ModelResponse("answer")),
    (CapabilityReport({"text_generation": CapabilityEvidence("supported", "probe", observed=True)}),
     ModelResponse("")),
    (CapabilityReport({name: CapabilityEvidence("supported", "probe", observed=True)
                       for name in ("text_generation", "logprobs", "top_logprobs")}, FeatureChoices(mia_scoring=True)),
     ModelResponse("answer")),
])
def test_ready_cannot_invent_required_capabilities_or_real_response_evidence(report, response):
    from recall_guard.core.readiness_evidence import ProbeEvidence

    config = ModelConfig("callback", None, "selected-model")
    with pytest.raises(ValueError, match="ready"):
        ProbeEvidence(config, report, response, [observation(state="ready")])


def test_partial_size_limit_covers_utf8_and_token_alternatives_not_only_answer_text():
    from recall_guard.core.readiness_evidence import PartialResponse

    partial = PartialResponse("answer")
    exact_size = len(json.dumps(partial.to_dict(), ensure_ascii=False, allow_nan=False).encode("utf-8"))
    assert replace(partial, max_retained_bytes=exact_size).content == "answer"
    with pytest.raises(ValueError, match="max_retained_bytes"):
        replace(partial, content="\u00e9" * exact_size, max_retained_bytes=exact_size)
    token = TokenEvidence("answer", -0.2, (TokenAlternative("\u00e9" * exact_size, -0.2),))
    with pytest.raises(ValueError, match="max_retained_bytes"):
        replace(partial, token_evidence=(token,), max_retained_bytes=exact_size)


def test_probe_trace_rejects_other_operations_and_backwards_elapsed_time():
    from recall_guard.core.readiness_evidence import ProbeEvidence

    config = ModelConfig("callback", None, "selected-model")
    report = CapabilityReport({"text_generation": CapabilityEvidence("unverified", "not observed")})
    for trace in ([observation(), observation(operation_id="another")],
                  [observation(elapsed_s=2), observation(elapsed_s=1)]):
        with pytest.raises(ValueError):
            ProbeEvidence(config, report, None, trace)


def observed_report(config, *, mia_scoring=False):
    features = FeatureChoices(mia_scoring=mia_scoring)
    return CapabilityReport({name: CapabilityEvidence("supported", "probe", observed=True)
                             for name in config.inspect(features).capabilities}, features)


@pytest.mark.parametrize(("parameters", "tokens"), [
    ({"logprobs": True}, None),
    ({"logprobs": True}, (TokenEvidence("answer", -9999.0),)),
    ({"top_logprobs": 1}, None),
    ({"logprobs": True, "top_logprobs": 2}, (TokenEvidence("answer", -0.2),)),
    ({"top_logprobs": 1}, (TokenEvidence("answer", -0.2, (TokenAlternative("answer", -9999.0),)),)),
])
def test_ready_checks_explicit_token_requests_even_without_mia(parameters, tokens):
    from recall_guard.core.readiness_evidence import ProbeEvidence

    config = ModelConfig("callback", None, "selected-model", parameters=parameters)
    report = observed_report(config)
    assert not report.features.mia_scoring
    response = ModelResponse("answer", tokens, requested_settings=parameters)
    with pytest.raises(ValueError, match="ready"):
        ProbeEvidence(config, report, response, [observation(state="ready")])
    # Retain the complete response under its actual failed capability report.
    missing = "top_logprobs" if parameters.get("top_logprobs") else "logprobs"
    failed_report = replace(report, capabilities=dict(report.capabilities) | {
        missing: CapabilityEvidence("unsupported", "probe", observed=True),
    })
    deferred = ProbeEvidence(config, failed_report, response,
                             [observation(state="deferred", reason="missing_token_evidence")])
    assert deferred.complete and not deferred.ready and deferred.response is response


@pytest.mark.parametrize(("parameters", "tokens", "mia_scoring"), [
    ({"logprobs": True}, (TokenEvidence("answer", -0.2),), False),
    ({"logprobs": True, "top_logprobs": 0}, (TokenEvidence("answer", -0.2),), False),
    ({"logprobs": True, "top_logprobs": None}, (TokenEvidence("answer", -0.2),), False),
    ({"logprobs": False}, None, False),
    ({"logprobs": None, "top_logprobs": 0}, None, False),
    ({"logprobs": True, "top_logprobs": 1},
     (TokenEvidence("answer", -0.2, (TokenAlternative("answer", -0.2),)),), False),
    ({}, (TokenEvidence("answer", -0.2, (TokenAlternative("answer", -0.2),)),), True),
])
def test_ready_accepts_only_the_token_evidence_selected_by_the_caller(parameters, tokens, mia_scoring):
    from recall_guard.core.readiness_evidence import ProbeEvidence

    config = ModelConfig("callback", None, "selected-model", parameters=parameters)
    response = ModelResponse("answer", tokens, requested_settings=parameters)
    evidence = ProbeEvidence(config, observed_report(config, mia_scoring=mia_scoring), response,
                             [observation(state="ready")])
    assert evidence.ready
    if parameters.get("logprobs") and not parameters.get("top_logprobs") and not mia_scoring:
        assert response.logprobs_unavailable_reason is None
        assert response.mia_unavailable_reason is not None


@pytest.mark.parametrize(("selected", "requested"), [
    ({"temperature": 0.2}, {"temperature": 0.9}),
    ({"temperature": 0.2}, {}),
    ({}, {"temperature": 0.2}),
    ({}, {"temperature": None}),
    ({"temperature": None}, {}),
    ({"stop": ["END"]}, {"stop": ["OTHER"]}),
])
def test_ready_requires_exact_requested_configuration_including_omitted_and_null(selected, requested):
    from recall_guard.core.readiness_evidence import ProbeEvidence

    config = ModelConfig("callback", None, "selected-model", parameters=selected)
    response = ModelResponse("answer", requested_settings=requested, confirmed_settings=selected)
    with pytest.raises(ValueError, match="ready"):
        ProbeEvidence(config, observed_report(config), response, [observation(state="ready")])


@pytest.mark.parametrize("parameters", [{}, {"temperature": None}, {"temperature": 0.2, "stop": ["END"]}])
def test_ready_binds_requested_settings_without_inventing_confirmed_settings(parameters):
    from recall_guard.core.readiness_evidence import ProbeEvidence

    config = ModelConfig("callback", None, "selected-model", parameters=parameters)
    response = ModelResponse("answer", requested_settings=parameters)
    probe = ProbeEvidence(config, observed_report(config), response, [observation(state="ready")])
    exported = probe.to_dict()
    assert probe.ready
    assert exported["response"]["requested_settings"] == exported["config"]["parameters"]
    assert exported["response"]["confirmed_settings"] == {}


@pytest.mark.parametrize(("selected_value", "requested_value"), [
    ({"const": True}, {"const": 1}),
    ({"enum": [False]}, {"enum": [0]}),
    ({"enum": [[{"const": True}]]}, {"enum": [[{"const": 1}]]}),
    ({"const": 1}, {"const": 1.0}),
])
def test_ready_preserves_nested_json_types_despite_python_equality(selected_value, requested_value):
    from recall_guard.core.readiness_evidence import ProbeEvidence

    def parameters(schema):
        return {"response_format": {"type": "json_schema", "json_schema": {"name": "answer", "schema": schema}}}

    config = ModelConfig("callback", None, "selected-model", parameters=parameters(selected_value))
    response = ModelResponse("answer", requested_settings=parameters(requested_value))
    assert config.parameters == response.requested_settings
    with pytest.raises(ValueError, match="ready"):
        ProbeEvidence(config, observed_report(config), response, [observation(state="ready")])


def test_ready_ignores_object_key_order_but_preserves_nested_array_order():
    from recall_guard.core.readiness_evidence import ProbeEvidence

    selected = {"response_format": {"type": "json_schema", "json_schema": {
        "name": "answer", "schema": {"enum": [True, 1, None, "\u00e9"], "type": ["boolean", "number", "null", "string"]},
    }}, "temperature": 0.2}
    reordered = {"temperature": 0.2, "response_format": {"json_schema": {
        "schema": {"type": ["boolean", "number", "null", "string"], "enum": [True, 1, None, "\u00e9"]}, "name": "answer",
    }, "type": "json_schema"}}
    config = ModelConfig("callback", None, "selected-model", parameters=selected)
    response = ModelResponse("answer", requested_settings=reordered)
    report = observed_report(config)
    trace = [observation(state="ready")]
    assert ProbeEvidence(config, report, response, trace).ready
    reordered["response_format"]["json_schema"]["schema"]["enum"] = [1, True, None, "\u00e9"]
    changed_response = ModelResponse("answer", requested_settings=reordered)
    with pytest.raises(ValueError, match="ready"):
        ProbeEvidence(config, report, changed_response, trace)
