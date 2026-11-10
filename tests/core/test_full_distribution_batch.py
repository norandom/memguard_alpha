"""Batched full-distribution evidence and the busy-host mapping of the gateway client (task 6.4).

``FullDistributionClient.evidence_batch`` sends one prompt with several responses to
``/v1/full_distribution_batch`` and returns one evidence object per response, in order, with a
per-call timeout the plan sets. HTTP 503 (the host's ``memory_low`` refusal) maps to the
retryable ``host_busy`` on both routes. The fake host is a loopback ThreadingHTTPServer.
"""
import json

import pytest
import requests

from recall_guard.core.full_distribution import (
    BATCH_ENDPOINT_PATH,
    ENDPOINT_PATH,
    FullDistributionEvidence,
)
from recall_guard.core.request_control import RequestError
from tests.core.test_full_distribution import DIGEST, TOKEN, assert_no_token, client, host, wire_body

SECOND = {"response_token_ids": [5, 3], "logprob": [-1.5, -0.25], "entropy": [1.7, 0.9], "rank": [2, 1],
          "top_logprobs": [[[2, -0.5], [5, -1.5]], [[3, -0.25], [1, -2.0]]]}
MEMORY_LOW = {"error": {"code": "memory_low", "message": "free physical memory is below the floor"}}


def batch_body(*items):
    return {"evidence": [wire_body(**item) for item in items]}


def test_batch_posts_one_prompt_with_its_responses_and_returns_evidence_in_order():
    with host([(200, batch_body({}, SECOND))]) as (base_url, seen):
        evidence = client(base_url + "/").evidence_batch("Prompt text", ["First", "Second"], top_k_keep=2)
    [request] = seen
    assert (request["method"], request["path"]) == ("POST", BATCH_ENDPOINT_PATH)
    assert request["headers"]["Authorization"] == f"Bearer {TOKEN}"
    assert request["body"] == {"model_digest": DIGEST, "prompt_text": "Prompt text",
                               "response_texts": ["First", "Second"], "top_k_keep": 2}
    assert isinstance(evidence, tuple) and len(evidence) == 2
    assert all(isinstance(item, FullDistributionEvidence) for item in evidence)
    assert [item.to_dict() for item in evidence] == batch_body({}, SECOND)["evidence"]


def test_batch_accepts_a_tuple_of_responses_and_one_response():
    with host([(200, batch_body({})), (200, batch_body({}, SECOND))]) as (base_url, seen):
        fake = client(base_url)
        [only] = fake.evidence_batch("p", ("r",), top_k_keep=2)
        assert len(fake.evidence_batch("p", ("r", "s"), top_k_keep=2)) == 2
    assert only.to_dict() == wire_body()
    assert [entry["body"]["response_texts"] for entry in seen] == [["r"], ["r", "s"]]


def test_batch_timeout_is_set_per_call_from_the_plan_and_defaults_to_the_client(monkeypatch):
    calls = []

    def send(adapter, prepared, **kwargs):
        calls.append({"url": prepared.url, "body": json.loads(prepared.body), "timeout": kwargs["timeout"],
                      "retries": adapter.max_retries.total, "redirects": kwargs.get("allow_redirects")})
        response = requests.Response()
        response.status_code = 200
        single = prepared.url.endswith(ENDPOINT_PATH)
        response._content = json.dumps(wire_body() if single else batch_body({})).encode()
        response.request = prepared
        return response

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)
    fake = client("http://127.0.0.1:11581")
    fake.evidence_batch("p", ["r"])
    fake.evidence_batch("p", ["r"], timeout_s=600)
    client("http://127.0.0.1:11581", timeout_s=7.5).evidence_batch("p", ["r"])
    fake.evidence("p", "r")  # The per-call timeout does not change the client's own.
    assert [call["timeout"] for call in calls] == [120.0, 600.0, 7.5, 120.0]
    assert [call["url"].rsplit(":11581", 1)[1] for call in calls] == [BATCH_ENDPOINT_PATH] * 3 + [ENDPOINT_PATH]
    assert [call["body"]["top_k_keep"] for call in calls] == [100] * 4
    assert [call["retries"] for call in calls] == [0] * 4


@pytest.mark.parametrize("route", ["single", "batch"])
def test_503_memory_low_is_a_retryable_host_busy_on_both_routes(route):
    with host([(503, MEMORY_LOW)]) as (base_url, seen):
        fake = client(base_url)
        with pytest.raises(RequestError) as caught:
            if route == "single":
                fake.evidence("p", "r")
            else:
                fake.evidence_batch("p", ["r", "s"])
    error = caught.value
    assert (error.status_code, error.code, error.retryable) == (503, "host_busy", True)
    assert "memory_low" in error.detail and "503" in str(error)
    assert len(seen) == 1  # One request, no hidden retry: the runner decides when to retry.
    assert_no_token(error)


@pytest.mark.parametrize(("status", "code"), [
    (401, "auth_failed"), (409, "digest_mismatch"), (413, "oversize"), (422, "tokenisation_failed"),
    (500, "http_error"), (404, "http_error"),
])
def test_batch_host_statuses_map_like_the_single_route(status, code):
    with host([(status, {"error": "refused"})]) as (base_url, _):
        with pytest.raises(RequestError) as caught:
            client(base_url).evidence_batch("p", ["r"])
    assert (caught.value.status_code, caught.value.code, caught.value.retryable) == (status, code, False)
    assert "refused" in caught.value.detail


@pytest.mark.parametrize("payload", [
    pytest.param(batch_body({}), id="fewer-than-the-responses"),
    pytest.param(batch_body({}, SECOND, {}), id="more-than-the-responses"),
    pytest.param({"evidence": [wire_body(), wire_body(model_digest="sha256:" + "0" * 64)]}, id="wrong-digest"),
    pytest.param({"evidence": [wire_body(), wire_body(
        top_logprobs=[[[3, -0.25], [1, -2.0], [0, -3.0]], [[2, -0.75]]])]}, id="over-top-k"),
    pytest.param({"evidence": [wire_body(), wire_body(logprob=[-0.25])]}, id="invalid-item"),
    pytest.param({"evidence": [wire_body(), wire_body()], "timings": {}}, id="unknown-field"),
    pytest.param({"items": [wire_body(), wire_body()]}, id="missing-evidence"),
    pytest.param({"evidence": wire_body()}, id="evidence-not-a-list"),
    pytest.param([wire_body(), wire_body()], id="not-an-object"),
    pytest.param(b"not json", id="not-json"),
])
def test_a_batch_reply_without_one_valid_evidence_per_response_is_invalid_evidence(payload):
    with host([(200, payload)]) as (base_url, _):
        with pytest.raises(RequestError) as caught:
            client(base_url).evidence_batch("p", ["r", "s"], top_k_keep=2)
    assert (caught.value.status_code, caught.value.code, caught.value.retryable) == (200, "invalid_evidence", False)
    assert caught.value.__cause__ is None and caught.value.__context__ is None


@pytest.mark.parametrize(("args", "kwargs"), [
    (("p", []), {}), (("p", ()), {}), (("p", "r"), {}), (("p", [""]), {}), (("p", ["r", None]), {}),
    ((None, ["r"]), {}), (("p", ["r"]), {"top_k_keep": 0}), (("p", ["r"]), {"top_k_keep": True}),
    (("p", ["r"]), {"timeout_s": 0}), (("p", ["r"]), {"timeout_s": float("inf")}),
    (("p", ["r"]), {"timeout_s": True}),
], ids=["empty-list", "empty-tuple", "text-not-list", "empty-response", "response-not-text", "prompt-not-text",
        "top-k-zero", "top-k-bool", "timeout-zero", "timeout-inf", "timeout-bool"])
def test_batch_arguments_are_validated_before_any_request(args, kwargs):
    with host([]) as (base_url, seen):
        with pytest.raises((TypeError, ValueError)):
            client(base_url).evidence_batch(*args, **kwargs)
    assert seen == []


def test_batch_refuses_a_bad_credential_before_any_request():
    with host([]) as (base_url, seen):
        with pytest.raises(ValueError, match="credential"):
            client(base_url, environment={}).evidence_batch("p", ["r"])
        with pytest.raises(ValueError, match="whitespace, control or non-ASCII"):
            client(base_url, environment={"FD_GATEWAY_KEY": TOKEN + "\n"}).evidence_batch("p", ["r"])
    assert seen == []


def test_batch_never_reveals_a_bearer_the_host_echoes():
    def echo(handler):
        return {"error": "bad credential", "received": handler.headers["Authorization"]}

    with host([(401, echo), (200, {"evidence": [wire_body(model_digest=TOKEN)]})]) as (base_url, _):
        fake = client(base_url)
        with pytest.raises(RequestError) as rejected:
            fake.evidence_batch("p", ["r"])
        with pytest.raises(RequestError) as invalid:
            fake.evidence_batch("p", ["r"])
    assert rejected.value.code == "auth_failed" and "[redacted]" in rejected.value.detail
    assert invalid.value.code == "invalid_evidence"
    assert_no_token(rejected.value)
    assert_no_token(invalid.value)


def test_batch_transport_failure_is_retryable_and_redacted(monkeypatch):
    def send(adapter, prepared, **kwargs):
        raise requests.ConnectionError(f"reset while sending Authorization: Bearer {TOKEN}")

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)
    with pytest.raises(RequestError) as caught:
        client("http://127.0.0.1:11581").evidence_batch("p", ["r"])
    assert (caught.value.code, caught.value.retryable, caught.value.status_code) == ("transport_error", True, None)
    assert_no_token(caught.value)
