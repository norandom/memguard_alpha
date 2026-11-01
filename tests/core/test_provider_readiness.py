"""Real local HTTP exchanges for the optional configured observation path."""

import json
import threading
import time
from contextlib import contextmanager
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from recall_guard.core.completion import FeatureChoices, ModelConfig
from recall_guard.core.providers import OpenAICompatibleClient
from recall_guard.core.readiness_evidence import ProbeEvidence
from recall_guard.core.readiness_types import CancellationToken, PendingStatusPolicy, ReadinessPolicy
from recall_guard.core.request_control import CallBudget, RequestError
from recall_guard.harness.workflow_plan import CompletionFallbackPolicy, WorkflowRequest
from recall_guard.harness.workflow_preparation import prepare_workflow

REQUEST_ID = "12345678-1234-1234-1234-123456789abc"
BODY = {"choices": [{"message": {"content": "OK"}, "finish_reason": "stop"}]}


@contextmanager
def server(replies):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self):
            size = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(size) if size else b""
            requests.append((self.command, self.path, json.loads(body) if body else None))
            status, headers, payload = replies.pop(0)
            if callable(payload):
                payload(self)
                return
            raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(status)
            for key, value in headers.items():
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            try:
                self.wfile.write(raw)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass

        do_POST = respond
        do_GET = respond

    http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{http.server_port}/chat", requests
    finally:
        http.shutdown()
        http.server_close()
        thread.join(3)
        assert not thread.is_alive()


def client(endpoint, *, policy=None, parameters=None, cap=5, features=None, retries=2):
    config = ModelConfig("openai_chat", endpoint, "fixture-model", provider="fixture",
                         parameters=parameters or {}, readiness=policy or ReadinessPolicy(
                             backoff_initial_s=0, backoff_max_s=0), max_retries=retries)
    return OpenAICompatibleClient(config, CallBudget(cap), allow_unauthenticated=True, features=features)


def test_json_probe_retains_configured_settings_and_ready_trace():
    with server([(200, {}, BODY)]) as (endpoint, seen):
        c = client(endpoint, parameters={"temperature": None, "max_tokens": 7})
        evidence = c.probe_with_evidence()
        assert evidence.ready and evidence.complete
        assert evidence.response.requested_settings == c.config.parameters
        assert seen[0][2] == {"model": "fixture-model", "messages": [{"role": "user", "content": "Reply with the word OK."}],
                              "temperature": None, "max_tokens": 7}
        assert c.budget.events[0].purpose == "probe"


def test_missing_token_probe_preserves_complete_response_without_ready():
    with server([(200, {}, BODY)]) as (endpoint, _):
        c = client(endpoint, features=FeatureChoices(mia_scoring=True))
        evidence = c.probe_with_evidence()
        assert evidence.complete and not evidence.ready
        assert evidence.response.content == "OK"
        assert set(evidence.report.unsupported) == {"logprobs", "top_logprobs"}


def test_preparation_reuses_actual_readiness_probe_and_reprobes_selected_settings():
    with server([(200, {}, BODY), (200, {}, BODY)]) as (endpoint, seen):
        cfg = client(endpoint, parameters={"logprobs": True, "top_logprobs": 2, "max_tokens": 20}).config
        budget = CallBudget(2)
        result = prepare_workflow(WorkflowRequest({"target": cfg}, FeatureChoices(mia_scoring=True)),
            CompletionFallbackPolicy(FeatureChoices(), True),
            lambda config, budget, features: OpenAICompatibleClient(config, budget, features=features,
                                                                     allow_unauthenticated=True), budget)
        assert result.ready and budget.used == len(seen) == 2
        assert seen[0][2]["logprobs"] is True and "logprobs" not in seen[1][2]
        assert seen[1][2]["max_tokens"] == 20
        assert [probe.response.content for probe in result.artifact.evidence["target"]] == ["OK", "OK"]
        assert all(record.event.purpose == "probe" for record in result.artifact.requests)


def test_preparation_pending_is_inspectable_without_fallback_or_evaluation():
    with server([(202, {"NVCF-REQID": REQUEST_ID, "NVCF-STATUS": "pending"}, {})]) as (endpoint, seen):
        cfg = client(endpoint).config
        result = prepare_workflow(WorkflowRequest({"target": cfg}, FeatureChoices(mia_scoring=True)),
            CompletionFallbackPolicy(FeatureChoices(), True),
            lambda config, budget, features: OpenAICompatibleClient(config, budget, features=features,
                                                                     allow_unauthenticated=True), CallBudget(3))
        assert not result.ready and result.plan.reason is None
        assert result.artifact.stopping_reason == "pending_route_unavailable"
        assert len(seen) == len(result.artifact.requests) == 1
        assert any(observation.request_id == REQUEST_ID
                   for observation in result.artifact.evidence["target"][0].observations)


def test_nim_pending_defers_without_guessing_status_route():
    with server([(202, {"NVCF-REQID": REQUEST_ID, "NVCF-STATUS": "pending"}, {})]) as (endpoint, seen):
        c = client(endpoint)
        c.config = replace(c.config, provider="nvidia_nim", model="openai/gpt-oss-20b")
        # Construct the actual configured client; no remote requests are used.
        c = OpenAICompatibleClient(c.config, c.budget, allow_unauthenticated=True)
        result = c.generate_with_readiness("hello")
        assert result.pending.request_id == REQUEST_ID
        assert result.pending.provider_status == "pending"
        assert result.observations[-1].reason == "pending_route_unavailable"
        assert len(seen) == 1 and c.budget.used == 1


@pytest.mark.parametrize("cap,polls,expected,exchanges", [(5, 2, "ready", 3), (2, 2, "exhausted", 2), (5, 1, "exhausted", 2)])
def test_accepted_work_uses_one_post_and_budgeted_status_gets(cap, polls, expected, exchanges):
    pending = (202, {"NVCF-REQID": REQUEST_ID, "Retry-After": "0"}, {})
    with server([pending, pending, (200, {}, BODY)]) as (endpoint, seen):
        route = PendingStatusPolicy("openai_chat_fixture", "fixture-model", endpoint,
                                    endpoint + "/status/{request_id}", "NVCF-REQID", True)
        c = client(endpoint, policy=ReadinessPolicy(pending_status=route, max_poll_requests=polls,
                   backoff_initial_s=0, backoff_max_s=0), cap=cap)
        result = c.generate_with_readiness("hello")
        assert result.observations[-1].state == expected
        assert [row[0] for row in seen] == ["POST"] + ["GET"] * (exchanges - 1)
        assert [row.purpose for row in c.budget.events] == ["inference"] + ["status"] * (exchanges - 1)
        assert len({row.operation_id for row in c.budget.events}) == 1


def sse(*events):
    return b"".join(b"data: " + (event.encode() if isinstance(event, str) else json.dumps(event).encode()) + b"\n\n" for event in events)


def test_stream_reports_separate_progress_and_exact_settings():
    raw = b": keepalive\n\n" + sse(
        {"choices": [{"index": 0, "delta": {"reasoning_content": "think"}}]},
        {"choices": [{"index": 0, "delta": {"content": "OK"}, "finish_reason": "stop"}]}, "[DONE]")
    with server([(200, {"Content-Type": "text/event-stream"}, raw)]) as (endpoint, _):
        c = client(endpoint, parameters={"stream": True})
        observed = []
        result = c.generate_with_readiness("hello", on_observation=observed.append)
        assert result.ready and result.response.content == "OK"
        assert result.response.requested_settings == {"stream": True}
        assert result.observations[-1].reasoning_characters == 5
        assert result.observations[-1].answer_characters == 2
        assert result.observations[-1].transport_bytes == len(raw)
        assert any(row.state == "progress" for row in observed)


def test_cancel_after_semantic_progress_retains_partial_and_admits_no_replay():
    raw = sse({"choices": [{"index": 0, "delta": {"content": "part"}}]})
    with server([(200, {"Content-Type": "text/event-stream"}, raw)]) as (endpoint, seen):
        c = client(endpoint, parameters={"stream": True})
        token = CancellationToken()
        result = c.generate_with_readiness("hello", cancellation=token,
                    on_observation=lambda row: token.cancel() if row.state == "progress" else None)
        assert result.observations[-1].state == "cancelled"
        assert result.response.content == "part" and not result.response.complete
        assert len(seen) == 1


def test_excessive_server_hint_defers_without_retry():
    with server([(503, {"Retry-After": "40"}, {"error": {"message": "busy"}})]) as (endpoint, seen):
        c = client(endpoint, policy=ReadinessPolicy(max_wait_s=1))
        result = c.generate_with_readiness("hello")
        assert result.observations[-1].state == "deferred"
        assert any(row.delay_s == 40 for row in result.observations)
        assert len(seen) == 1


def test_unknown_profile_rejected_before_credential_resolution():
    config = ModelConfig("openai_chat", "http://127.0.0.1:1/chat", "m", provider="unknown",
                         credential_ref="MISSING", readiness=ReadinessPolicy())
    with pytest.raises(ValueError, match="observation profile"):
        OpenAICompatibleClient(config, CallBudget(1), environment={})


def test_generate_raises_failure_with_trace_for_unusable_pending():
    with server([(202, {"NVCF-REQID": REQUEST_ID}, {})]) as (endpoint, _):
        c = client(endpoint)
        with pytest.raises(RequestError) as failure:
            c.generate("hello")
        assert failure.value.readiness_trace[-1].state == "deferred"


@pytest.mark.parametrize("status,body", [(429, {"error": {"message": "busy"}}),
                                        (503, {"error": {"message": "busy"}}),
                                        (504, {}), (200, b"not json")])
def test_status_failures_never_repost_accepted_generation(status, body):
    pending = (202, {"NVCF-REQID": REQUEST_ID}, {})
    with server([pending, (status, {"Retry-After": "0"}, body), (200, {}, BODY)]) as (endpoint, seen):
        route = PendingStatusPolicy("openai_chat_fixture", "fixture-model", endpoint,
                                    endpoint + "/status/{request_id}", "NVCF-REQID", True)
        c = client(endpoint, policy=ReadinessPolicy(pending_status=route, max_poll_requests=2,
                   allow_ambiguous_replay=True, backoff_initial_s=0, backoff_max_s=0))
        result = c.generate_with_readiness("hello")
        assert seen[0][0] == "POST" and all(row[0] == "GET" for row in seen[1:])
        assert len(seen) <= 3 and c.budget.used == len(seen)
        assert all(row.operation_id == result.observations[0].operation_id for row in c.budget.events)
        assert all(row.purpose == "status" for row in c.budget.events[1:])


@pytest.mark.parametrize("headers", [{}, {"NVCF-REQID": "not-a-uuid"}, {"Location": "http://127.0.0.1:1/steal"}])
def test_pending_without_valid_identity_defers_without_calls_to_locations(headers):
    with server([(202, headers, {})]) as (endpoint, seen):
        c = client(endpoint)
        result = c.generate_with_readiness("hello")
        assert result.observations[-1].state == "deferred"
        assert result.observations[-1].credential_eligibility == "unverified"
        assert len(seen) == 1


@pytest.mark.parametrize("status,code", [(401, None), (403, None), (400, "invalid_parameter"),
                                       (429, "insufficient_quota")])
def test_permanent_rejections_never_retry(status, code):
    with server([(status, {}, {"error": {"message": "no", "code": code}})]) as (endpoint, seen):
        c = client(endpoint)
        result = c.generate_with_readiness("hello")
        assert result.observations[-1].state == "rejected"
        assert len(seen) == 1


def test_server_hint_waits_before_a_budgeted_retry():
    with server([(429, {"Retry-After": "1"}, {}), (200, {}, BODY)]) as (endpoint, seen):
        c = client(endpoint)
        before = time.monotonic()
        result = c.generate_with_readiness("hello")
        assert result.ready and time.monotonic() - before >= 1
        assert len(seen) == 2 and [row.purpose for row in c.budget.events] == ["inference", "retry"]


def test_cancel_before_admission_uses_zero_physical_requests():
    c = client("http://127.0.0.1:1/chat")
    token = CancellationToken()
    token.cancel()
    result = c.generate_with_readiness("hello", cancellation=token)
    assert result.observations[-1].state == "cancelled" and c.budget.used == 0


def test_override_settings_are_preserved_without_injecting_defaults():
    with server([(200, {}, BODY)]) as (endpoint, seen):
        c = client(endpoint, parameters={"max_tokens": 8})
        result = c.generate("hello", temperature=0.3)
        assert result.requested_settings == {"max_tokens": 8, "temperature": 0.3}
        assert "top_p" not in seen[0][2]
        assert c.config.parameters == {"max_tokens": 8}


def test_invalid_poll_profile_rejected_before_any_calls():
    endpoint = "http://127.0.0.1:1/chat"
    route = PendingStatusPolicy("unverified", "fixture-model", endpoint,
                                endpoint + "/{request_id}", "NVCF-REQID", True)
    with pytest.raises(ValueError, match="pending observation profile"):
        client(endpoint, policy=ReadinessPolicy(pending_status=route, max_poll_requests=1))


def test_redirect_is_not_followed():
    with server([]) as (other, stolen), server([(307, {"Location": other}, {})]) as (endpoint, seen):
        c = client(endpoint)
        result = c.generate_with_readiness("hello")
        assert not result.ready and len(seen) == 1 and stolen == []


def test_owned_session_and_response_close_on_stream_failure(monkeypatch):
    import requests

    closed = []
    original = requests.Session

    class OwnedSession(original):
        def request(self, *args, **kwargs):
            response = super().request(*args, **kwargs)
            close = response.close

            def close_response():
                closed.append("response")
                close()

            response.close = close_response
            return response

        def close(self):
            closed.append("session")
            super().close()

    monkeypatch.setattr(requests, "Session", OwnedSession)
    raw = sse({"choices": [{"index": 0, "delta": {"content": "partial"}}]})
    with server([(200, {"Content-Type": "text/event-stream"}, raw)]) as (endpoint, seen):
        c = client(endpoint, parameters={"stream": True})
        result = c.generate_with_readiness("hello")
        assert result.observations[-1].state == "unknown_outcome"
        assert result.response.content == "partial" and not result.response.complete
        assert len(seen) == 1
        assert closed == ["response", "session"]


def test_callback_failure_has_static_diagnostics_and_no_replay():
    raw = sse({"choices": [{"index": 0, "delta": {"content": "OK"}, "finish_reason": "stop"}]}, "[DONE]")
    with server([(200, {"Content-Type": "text/event-stream"}, raw)]) as (endpoint, seen):
        c = client(endpoint, parameters={"stream": True}, policy=ReadinessPolicy(allow_ambiguous_replay=True))

        def observer(row):
            if row.state == "progress":
                raise RuntimeError("private callback detail")

        result = c.generate_with_readiness("hello", on_observation=observer)
        assert result.observations[-1].reason == "callback_failed"
        assert "private callback detail" not in str(result.error)
        assert len(seen) == 1


def test_real_socket_progress_is_delivered_before_eof_and_idle_timeout_cleans_up():
    first = sse({"choices": [{"index": 0, "delta": {"content": "partial"}}]})
    release = threading.Event()
    ended = threading.Event()

    def stalled(handler):
        try:
            handler.send_response(200)
            handler.send_header("Content-Type", "text/event-stream")
            handler.send_header("Content-Length", str(len(first) + 100))
            handler.end_headers()
            handler.wfile.write(first)
            handler.wfile.flush()
            release.wait(3)
        finally:
            ended.set()

    with server([(200, {}, stalled)]) as (endpoint, seen):
        c = client(endpoint, parameters={"stream": True}, policy=ReadinessPolicy(socket_idle_timeout_s=0.1))
        progress = threading.Event()
        result = c.generate_with_readiness("hello", on_observation=lambda row: progress.set() if row.state == "progress" else None)
        release.set()
        assert progress.is_set() and ended.wait(1)
        assert result.observations[-1].state == "unknown_outcome"
        assert result.response.content == "partial" and not result.response.complete
        assert len(seen) == 1


def test_status_socket_timeout_only_retries_gets_under_explicit_permission():
    release = threading.Event()
    ended = threading.Event()

    def stalled(handler):
        try:
            handler.send_response(200)
            handler.send_header("Content-Length", "10")
            handler.end_headers()
            release.wait(3)
        finally:
            ended.set()

    pending = (202, {"NVCF-REQID": REQUEST_ID}, {})
    with server([pending, (200, {}, stalled), (200, {}, BODY)]) as (endpoint, seen):
        route = PendingStatusPolicy("openai_chat_fixture", "fixture-model", endpoint,
                                    endpoint + "/status/{request_id}", "NVCF-REQID", True)
        c = client(endpoint, policy=ReadinessPolicy(pending_status=route, max_poll_requests=2,
                   allow_ambiguous_replay=True, socket_idle_timeout_s=0.1,
                   backoff_initial_s=0, backoff_max_s=0))
        result = c.generate_with_readiness("hello")
        release.set()
        assert ended.wait(1) and result.ready
        assert [row[0] for row in seen] == ["POST", "GET", "GET"]
        assert c.budget.used == 3 and c.budget.events[1].status == "failed"


def test_missing_token_complete_evidence_is_sanitized_before_exposure():
    secret = "dummy-local-secret"
    body = {"choices": [{"message": {"content": secret}, "finish_reason": "stop"}]}
    with server([(200, {}, body)]) as (endpoint, _):
        config = ModelConfig("openai_chat", endpoint, "fixture-model", provider="fixture",
                             credential_ref="LOCAL_DUMMY", readiness=ReadinessPolicy())
        c = OpenAICompatibleClient(config, CallBudget(1), features=FeatureChoices(mia_scoring=True),
                                   environment={"LOCAL_DUMMY": secret})
        evidence = c.probe_with_evidence()
        assert evidence.complete and not evidence.ready
        assert secret not in json.dumps(evidence.to_dict())


@pytest.mark.parametrize("protocol", ["callback", "anthropic_messages"])
def test_other_transports_reject_readiness_before_runtime_work(protocol):
    config = ModelConfig(protocol, None if protocol == "callback" else "http://127.0.0.1:1/messages", "model",
                         readiness=ReadinessPolicy(), parameters={"max_tokens": 4})
    with pytest.raises(ValueError, match="observation profile"):
        if protocol == "callback":
            from recall_guard.core.provider_callback import CallbackClient

            CallbackClient(config, CallBudget(1), lambda *args: pytest.fail("callback invoked"))
        else:
            from recall_guard.core.provider_anthropic import AnthropicClient

            AnthropicClient(config, CallBudget(1), environment={})


@pytest.mark.parametrize("stream", [True, False, None, 1])
def test_stream_remains_rejected_without_opt_in(stream):
    with pytest.raises(ValueError, match="stream"):
        ModelConfig("openai_chat", "http://127.0.0.1:1/chat", "model", parameters={"stream": stream})


def test_keyboard_interrupt_closes_owned_response_and_session(monkeypatch):
    import requests

    closed = []
    original = requests.Session

    class InterruptingSession(original):
        def request(self, *args, **kwargs):
            response = super().request(*args, **kwargs)
            original_close = response.close

            def interrupt(*args, **kwargs):
                raise KeyboardInterrupt

            def close():
                closed.append("response")
                original_close()

            response.iter_content = interrupt
            response.close = close
            return response

        def close(self):
            closed.append("session")
            super().close()

    monkeypatch.setattr(requests, "Session", InterruptingSession)
    with server([(200, {}, BODY)]) as (endpoint, seen):
        c = client(endpoint)
        with pytest.raises(KeyboardInterrupt):
            c.generate("hello")
        assert c.readiness_snapshot().state == "cancelled"
        assert len(seen) == 1 and closed == ["response", "session"]
        assert c.budget.events[0].status == "started"


def test_custom_probe_stage_still_records_probe_purpose():
    with server([(200, {}, BODY)]) as (endpoint, _):
        c = client(endpoint)
        assert c.probe(stage="preparation/target").ready_for_batch
        assert c.budget.events[0].purpose == "probe"
        assert c.budget.events[0].stage == "preparation/target"


def test_explicit_stream_replay_keeps_operation_progress_cumulative():
    partial = sse({"choices": [{"index": 0, "delta": {"content": "abc"}}]})
    complete = sse({"choices": [{"index": 0, "delta": {"content": "OK"}, "finish_reason": "stop"}]}, "[DONE]")
    with server([(200, {"Content-Type": "text/event-stream"}, partial),
                 (200, {"Content-Type": "text/event-stream"}, complete)]) as (endpoint, seen):
        c = client(endpoint, parameters={"stream": True}, policy=ReadinessPolicy(
                   allow_ambiguous_replay=True, backoff_initial_s=0, backoff_max_s=0))
        result = c.generate_with_readiness("hello")
        assert result.ready and result.response.content == "OK" and len(seen) == 2
        assert result.observations[-1].answer_characters == 5
        assert result.observations[-1].transport_bytes == len(partial) + len(complete)
        assert all(a.answer_characters <= b.answer_characters
                   for a, b in zip(result.observations, result.observations[1:], strict=False))


def test_malformed_stream_keeps_actual_supplied_byte_count_without_semantic_progress():
    raw = b"data: invalid\n\n"
    with server([(200, {"Content-Type": "text/event-stream"}, raw)]) as (endpoint, _):
        c = client(endpoint, parameters={"stream": True})
        result = c.generate_with_readiness("hello")
        assert not result.ready and not result.response.complete
        assert result.observations[-1].transport_bytes == len(raw)
        assert all(row.state != "progress" for row in result.observations)


@pytest.mark.parametrize("parameter", ["logprobs", "top_logprobs", "temperature"])
def test_rejected_probe_control_evidence_is_exact_and_roundtrips(parameter):
    parameters = {"logprobs": True, "top_logprobs": 2, "temperature": 0.3}
    error = {"error": {"code": "invalid_parameter", "param": parameter, "message": "unsupported control"}}
    with server([(400, {}, error)]) as (endpoint, seen):
        c = client(endpoint, parameters=parameters)
        evidence = c.probe_with_evidence()
        assert evidence.response is None and not evidence.ready
        assert evidence.report.unsupported == (parameter,)
        rejected = evidence.report.capabilities[parameter]
        assert rejected.observed and "configuration" in rejected.detail
        assert all(entry.status == "unverified" and not entry.observed
                   for name, entry in evidence.report.capabilities.items() if name != parameter)
        payload = json.loads(json.dumps(evidence.to_dict()))
        restored = ProbeEvidence.from_dict(payload)
        assert restored.to_dict() == payload and restored.report.unsupported == (parameter,)
        assert restored.config.parameters == parameters
        assert len(seen) == 1 and c.budget.used == 1


@pytest.mark.parametrize("status,code,param", [
    (401, "invalid_api_key", "logprobs"), (403, "permission_denied", "top_logprobs"),
    (400, "invalid_api_key", "logprobs"), (429, "insufficient_quota", "logprobs"),
    (400, "insufficient_quota", "logprobs"), (503, None, "logprobs"), (504, None, "logprobs"),
    (408, "timeout", "logprobs"), (400, "timeout", "logprobs"),
    (400, "connection_error", "logprobs"), (400, "unknown_failure", "logprobs"),
    (400, "invalid_parameter", None),
])
def test_other_probe_failures_never_claim_missing_token_capability(status, code, param):
    error = {"error": {"code": code, "param": param, "message": "failed"}}
    with server([(status, {}, error)]) as (endpoint, _):
        c = client(endpoint, parameters={"logprobs": True, "top_logprobs": 2}, retries=0)
        evidence = c.probe_with_evidence()
        assert not evidence.ready and evidence.response is None
        assert evidence.report.unsupported == ()
        assert all(entry.status == "unverified" and not entry.observed for entry in evidence.report.capabilities.values())
        assert ProbeEvidence.from_dict(json.loads(json.dumps(evidence.to_dict()))).report.unsupported == ()


def test_probe_rejection_cannot_claim_an_omitted_control_was_observed():
    error = {"error": {"code": "invalid_parameter", "param": "logprobs"}}
    with server([(400, {}, error)]) as (endpoint, _):
        c = client(endpoint, features=FeatureChoices(mia_scoring=True))
        evidence = c.probe_with_evidence()
        assert evidence.report.unsupported == ()
        assert evidence.report.capabilities["logprobs"].status == "unverified"


def test_rejected_probe_evidence_excludes_known_secret_diagnostics_and_headers():
    secret = "local-dummy-credential-for-rejected-probe"
    error = {"error": {"code": "invalid_parameter", "param": "logprobs",
                       "message": secret, "detail": secret}}
    with server([(400, {"x-request-id": secret, "Authorization": secret}, error)]) as (endpoint, _):
        config = ModelConfig("openai_chat", endpoint, "fixture-model", provider="fixture", credential_ref="LOCAL_DUMMY",
                             parameters={"logprobs": True}, readiness=ReadinessPolicy())
        c = OpenAICompatibleClient(config, CallBudget(1), environment={"LOCAL_DUMMY": secret})
        evidence = c.probe_with_evidence()
        serialized = json.dumps(evidence.to_dict())
        assert evidence.report.unsupported == ("logprobs",)
        assert secret not in serialized and "Authorization" not in serialized and "LOCAL_DUMMY" not in serialized
        assert ProbeEvidence.from_dict(json.loads(serialized)).report.unsupported == ("logprobs",)


@pytest.mark.parametrize("status,code", [(422, "invalid_parameter"), (422, "invalid_request_error"),
                                      (400, "invalid_request_error"), (400, "unsupported_parameter"),
                                      (422, "unsupported_parameter"), (400, None), (422, None)])
def test_rejected_probe_validation_status_and_codes(status, code):
    error = {"error": {"code": code, "param": "top_logprobs"}}
    with server([(status, {}, error)]) as (endpoint, seen):
        c = client(endpoint, parameters={"logprobs": True, "top_logprobs": 2})
        evidence = c.probe_with_evidence()
        assert evidence.report.unsupported == ("top_logprobs",)
        assert evidence.report.capabilities["top_logprobs"].observed
        assert evidence.report.capabilities["logprobs"].status == "unverified"
        assert evidence.observations[-1].reason == "parameter_rejected"
        assert ProbeEvidence.from_dict(json.loads(json.dumps(evidence.to_dict()))).to_dict() == evidence.to_dict()
        assert len(seen) == 1 and c.budget.used == 1
