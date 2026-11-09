"""Full-distribution evidence contract, logit conversion and the gateway client.

The fake host is a ThreadingHTTPServer on 127.0.0.1; no external network is used.
"""

import json
import threading
from contextlib import contextmanager
from dataclasses import FrozenInstanceError
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
import requests

from recall_guard.core.full_distribution import (
    _DETAIL_LIMIT,
    ENDPOINT_PATH,
    FullDistributionClient,
    FullDistributionEvidence,
    _redact,
    logprobs_from_logits,
)
from recall_guard.core.request_control import RequestError

DIGEST = "sha256:f06e91e18fc74295d0c781bf91622508c75655e2c18ed869d52447ffe1e667e1"
TOKEN = "fd-test-bearer-7f3a9c41d2"
ENVIRONMENT = {"FD_GATEWAY_KEY": TOKEN}


def wire_body(**changes):
    body = {
        "model_digest": DIGEST,
        "n_vocab": 8,
        "response_token_ids": [3, 5],
        "logprob": [-0.25, -1.5],
        "entropy": [0.9, 1.7],
        "rank": [1, 2],
        "top_logprobs": [[[3, -0.25], [1, -2.0]], [[2, -0.75], [5, -1.5]]],
    }
    return body | changes


@contextmanager
def host(replies):
    """Serve canned (status, payload[, headers]) replies and record each request of any method."""
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self):
            size = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(size) if size else b""
            seen.append({"method": self.command, "path": self.path, "headers": dict(self.headers),
                         "body": json.loads(body) if body else None})
            status, payload, *extra = replies.pop(0)
            if callable(payload):
                payload = payload(self)
            raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(status)
            for key, value in (extra[0] if extra else {}).items():
                self.send_header(key, value)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        do_POST = respond
        do_GET = respond

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", seen
    finally:
        server.shutdown()
        server.server_close()
        thread.join(3)
        assert not thread.is_alive()


def client(base_url, **changes):
    values = {"model_digest": DIGEST, "credential_ref": "env:FD_GATEWAY_KEY", "environment": ENVIRONMENT}
    return FullDistributionClient(base_url, **(values | changes))


def assert_no_token(error: RequestError):
    for text in (str(error), repr(error), error.detail or "", repr(error.detail)):
        assert TOKEN not in text
    assert error.__cause__ is None and error.__context__ is None


def test_successful_response_posts_the_contract_with_the_bearer_and_returns_frozen_evidence():
    with host([(200, wire_body())]) as (base_url, seen):
        evidence = client(base_url + "/").evidence("Prompt text", "Response text", top_k_keep=2)
    assert len(seen) == 1
    assert seen[0]["path"] == ENDPOINT_PATH
    assert seen[0]["headers"]["Authorization"] == f"Bearer {TOKEN}"
    assert seen[0]["body"] == {"model_digest": DIGEST, "prompt_text": "Prompt text",
                               "response_text": "Response text", "top_k_keep": 2}
    assert evidence == FullDistributionEvidence(
        model_digest=DIGEST, n_vocab=8, response_token_ids=(3, 5), logprob=(-0.25, -1.5), entropy=(0.9, 1.7),
        rank=(1, 2), top_logprobs=(((3, -0.25), (1, -2.0)), ((2, -0.75), (5, -1.5))))
    assert isinstance(evidence.top_logprobs[0][0], tuple)
    with pytest.raises(FrozenInstanceError):
        evidence.n_vocab = 9


def test_default_request_keeps_one_hundred_alternatives_with_no_hidden_retries_or_redirects(monkeypatch):
    calls = []

    def send(adapter, prepared, **kwargs):
        calls.append({"body": json.loads(prepared.body), "timeout": kwargs["timeout"],
                      "retries": adapter.max_retries.total})
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(wire_body()).encode()
        response.request = prepared
        return response

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)
    client("http://127.0.0.1:11581").evidence("p", "r")
    client("http://127.0.0.1:11581", timeout_s=7.5).evidence("p", "r")
    assert [call["body"]["top_k_keep"] for call in calls] == [100, 100]
    assert [call["timeout"] for call in calls] == [120.0, 7.5]
    assert [call["retries"] for call in calls] == [0, 0]


@pytest.mark.parametrize(("status", "code"), [
    (401, "auth_failed"), (409, "digest_mismatch"), (413, "oversize"), (422, "tokenisation_failed"),
    (500, "http_error"), (404, "http_error"),
])
def test_host_statuses_map_to_request_error_codes(status, code):
    with host([(status, {"error": "refused"})]) as (base_url, _):
        with pytest.raises(RequestError) as caught:
            client(base_url).evidence("p", "r")
    assert caught.value.status_code == status
    assert caught.value.code == code
    assert caught.value.retryable is False
    assert "refused" in caught.value.detail


@pytest.mark.parametrize("status", [302, 307])
def test_redirects_are_not_followed_to_another_host(status):
    with host([(200, wire_body())]) as (target_url, target_seen):
        moved = (status, {"error": "moved"}, {"Location": target_url + ENDPOINT_PATH})
        with host([moved]) as (base_url, seen):
            with pytest.raises(RequestError) as caught:
                client(base_url).evidence("p", "r")
    assert caught.value.status_code == status
    assert caught.value.code == "http_error"
    assert len(seen) == 1
    assert target_seen == []


def test_bearer_echoed_across_the_detail_limit_leaves_no_prefix():
    pad = "x" * (_DETAIL_LIMIT - 10)
    with host([(401, (pad + TOKEN + "tail").encode())]) as (base_url, _):
        with pytest.raises(RequestError) as caught:
            client(base_url).evidence("p", "r")
    detail = caught.value.detail
    assert len(detail) <= _DETAIL_LIMIT
    assert not any(TOKEN[:size] in detail for size in range(8, len(TOKEN) + 1))
    assert_no_token(caught.value)


@pytest.mark.parametrize("credential", [
    TOKEN + "\n", "\t" + TOKEN, TOKEN[:9] + "\r\n" + TOKEN[9:], TOKEN + " ", TOKEN[:9] + " " + TOKEN[9:],
    TOKEN + "\x00", TOKEN + "€", "中" + TOKEN, TOKEN + "’",
])
def test_credential_with_whitespace_control_or_non_ascii_characters_is_refused_before_any_request(credential):
    with host([(200, wire_body())]) as (base_url, seen):
        with pytest.raises(ValueError) as caught:
            client(base_url, environment={"FD_GATEWAY_KEY": credential}).evidence("p", "r")
    assert str(caught.value) == "Referenced credential contains whitespace, control or non-ASCII characters"
    assert seen == []
    for text in (str(caught.value), repr(caught.value), repr(caught.value.args)):
        assert TOKEN not in text and TOKEN[:9] not in text


def test_header_encoding_failure_is_mapped_without_the_bearer(monkeypatch):
    def send(adapter, prepared, **kwargs):
        raise UnicodeEncodeError("latin-1", f"Bearer {TOKEN}", 7, 8, "ordinal not in range(256)")

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)
    with pytest.raises(RequestError) as caught:
        client("http://127.0.0.1:11581").evidence("p", "r")
    error = caught.value
    assert error.code == "transport_error"
    assert error.retryable is False
    for text in (str(error), repr(error), repr(error.args), repr(error.detail)):
        assert TOKEN not in text and TOKEN[:9] not in text
    assert error.__cause__ is None and error.__context__ is None


def test_redaction_also_removes_the_escaped_form():
    credential = TOKEN[:9] + "\r\n" + TOKEN[9:]
    redacted = _redact(f"Invalid header value {credential!r} and {credential}", credential)
    assert TOKEN[:9] not in redacted and TOKEN[9:] not in redacted


def test_digest_mismatch_names_the_host_refusal():
    with host([(409, {"error": "digest mismatch", "loaded": "sha256:other"})]) as (base_url, seen):
        with pytest.raises(RequestError, match="409") as caught:
            client(base_url).evidence("p", "r")
    assert seen[0]["body"]["model_digest"] == DIGEST
    assert caught.value.code == "digest_mismatch"


def test_transport_failure_is_retryable_and_redacts_the_bearer(monkeypatch):
    def send(adapter, prepared, **kwargs):
        raise requests.ConnectionError(f"refused while sending Authorization: Bearer {TOKEN}")

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)
    with pytest.raises(RequestError) as caught:
        client("http://127.0.0.1:11581").evidence("p", "r")
    assert caught.value.code == "transport_error"
    assert caught.value.retryable is True
    assert caught.value.status_code is None
    assert "[redacted]" in caught.value.detail
    assert_no_token(caught.value)


def test_unreachable_host_is_a_transport_error():
    with host([]) as (base_url, _):
        pass
    with pytest.raises(RequestError) as caught:
        client(base_url, timeout_s=2.0).evidence("p", "r")
    assert caught.value.code == "transport_error"


def test_bearer_echoed_by_the_host_never_reaches_message_detail_or_repr():
    def echo(handler):
        return {"error": "bad credential", "received": handler.headers["Authorization"]}

    with host([(401, echo), (200, wire_body(model_digest=TOKEN))]) as (base_url, _):
        fake = client(base_url)
        with pytest.raises(RequestError) as rejected:
            fake.evidence("p", "r")
        with pytest.raises(RequestError) as invalid:
            fake.evidence("p", "r")
    assert rejected.value.code == "auth_failed"
    assert "[redacted]" in rejected.value.detail
    assert invalid.value.code == "invalid_evidence"
    assert_no_token(rejected.value)
    assert_no_token(invalid.value)
    assert TOKEN not in repr(fake)
    assert all(value is not TOKEN and value != TOKEN for value in vars(fake).values())


@pytest.mark.parametrize("payload", [
    pytest.param(wire_body(logprob=[-0.25]), id="mismatched-lengths"),
    pytest.param(wire_body(top_logprobs=[[[1, -2.0], [3, -0.25]], [[2, -0.75], [5, -1.5]]]), id="unsorted"),
    pytest.param(wire_body(model_digest="sha256:" + "0" * 64), id="wrong-digest"),
    pytest.param(wire_body(top_logprobs=[[[3, -0.25], [1, -2.0], [0, -3.0]], [[2, -0.75]]]), id="over-top-k"),
    pytest.param(wire_body(logprob=[0.1, -1.5]), id="positive-logprob"),
    pytest.param(wire_body(logprob=[-(10**400), -1.5]), id="logprob-integer-overflows-float"),
    pytest.param(wire_body(entropy=[10**400, 1.7]), id="entropy-integer-overflows-float"),
    pytest.param(wire_body(top_logprobs=[[[3, -(10**400)]], [[2, -0.75]]]), id="alternative-overflows-float"),
    pytest.param(wire_body(entropy=[-0.1, 1.7]), id="negative-entropy"),
    pytest.param(wire_body(rank=[0, 2]), id="rank-below-one"),
    pytest.param(wire_body(n_vocab=0), id="empty-vocabulary"),
    pytest.param(wire_body(response_token_ids=[3, 8]), id="token-outside-vocabulary"),
    pytest.param(wire_body(top_logprobs=[[[3, -0.25], [3, -0.25]], [[2, -0.75]]]), id="duplicate-alternative"),
    pytest.param(wire_body(extra=1), id="unknown-field"),
    pytest.param({key: value for key, value in wire_body().items() if key != "entropy"}, id="missing-field"),
    pytest.param(b"not json", id="not-json"),
    pytest.param([], id="not-an-object"),
])
def test_invalid_evidence_is_refused(payload):
    with host([(200, payload)]) as (base_url, _):
        with pytest.raises(RequestError) as caught:
            client(base_url).evidence("p", "r", top_k_keep=2)
    assert caught.value.code == "invalid_evidence"
    assert caught.value.status_code == 200
    assert caught.value.retryable is False


def test_missing_credential_is_refused_before_any_request():
    with host([]) as (base_url, seen):
        with pytest.raises(ValueError, match="credential"):
            client(base_url, environment={}).evidence("p", "r")
        with pytest.raises(ValueError, match="credential"):
            client(base_url, environment={"FD_GATEWAY_KEY": "  "}).evidence("p", "r")
    assert seen == []


@pytest.mark.parametrize("changes", [
    {"base_url": "ftp://127.0.0.1:11581"},
    {"base_url": "http://user:secret@127.0.0.1:11581"},
    {"base_url": "http://127.0.0.1:11581/?key=value"},
    {"credential_ref": f"Bearer {TOKEN}"},
    {"model_digest": " "},
    {"timeout_s": 0},
    {"timeout_s": float("inf")},
    {"environment": [("FD_GATEWAY_KEY", TOKEN)]},
])
def test_client_configuration_is_validated(changes):
    base_url = changes.pop("base_url", "http://127.0.0.1:11581")
    with pytest.raises((TypeError, ValueError)):
        client(base_url, **changes)


@pytest.mark.parametrize(("args", "kwargs"), [
    (("p", ""), {}), (("p", None), {}), (("p", "r"), {"top_k_keep": 0}), (("p", "r"), {"top_k_keep": True}),
])
def test_request_arguments_are_validated(args, kwargs):
    with pytest.raises((TypeError, ValueError)):
        client("http://127.0.0.1:11581").evidence(*args, **kwargs)


def test_evidence_round_trips_through_its_wire_shape():
    evidence = FullDistributionEvidence.from_dict(wire_body())
    assert FullDistributionEvidence.from_dict(json.loads(json.dumps(evidence.to_dict()))) == evidence
    assert evidence.to_dict() == wire_body()


def test_log_softmax_rows_sum_to_one_in_float64():
    logits = np.random.default_rng(7).normal(scale=4.0, size=(5, 300)).astype(np.float32)
    result = logprobs_from_logits(logits)
    assert result.dtype == np.float64
    assert result.shape == logits.shape
    assert np.all(result <= 0)
    np.testing.assert_allclose(np.exp(result).sum(axis=1), 1.0, rtol=0, atol=1e-12)


def test_log_softmax_is_stable_for_large_logits():
    small = np.array([[0.0, -1.0, -2.0, -30.0], [3.0, 3.0, 0.0, -5.0]])
    large = logprobs_from_logits(small + 1e4)
    assert np.all(np.isfinite(large))
    np.testing.assert_allclose(large, logprobs_from_logits(small), rtol=0, atol=1e-9)
    np.testing.assert_allclose(np.exp(large).sum(axis=1), 1.0, rtol=0, atol=1e-12)
    spread = logprobs_from_logits(np.array([[1e4, 0.0, -1e4]]))
    np.testing.assert_allclose(spread, [[0.0, -1e4, -2e4]], rtol=0, atol=1e-9)


@pytest.mark.parametrize("logits", [
    np.zeros(4), np.zeros((2, 0)), np.zeros((1, 2, 3)), np.array([[0.0, np.nan]]), np.array([[0.0, np.inf]]),
])
def test_log_softmax_rejects_malformed_logits(logits):
    with pytest.raises(ValueError):
        logprobs_from_logits(logits)
