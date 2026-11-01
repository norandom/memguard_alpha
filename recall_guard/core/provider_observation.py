"""Owned single HTTP exchanges for explicitly selected readiness profiles.

No admission, retry, polling loop or credential discovery lives here. Fixture
support is restricted to literal loopback hosts and establishes no hosted API
capability. Response Location headers never select a status route.
"""

import ipaddress
import json
from datetime import UTC, datetime
from urllib.parse import urlsplit

import requests

from recall_guard.core.provider_stream import StreamAssembler
from recall_guard.core.readiness_evidence import PartialResponseReason
from recall_guard.core.readiness_types import PendingReceipt, RetryAfterHint
from recall_guard.core.request_control import RequestError


def validate_observation_profile(config):
    """Reject unsupported selections before credential resolution or calls."""
    if config.readiness is None:
        return
    if config.provider not in {"nvidia_nim", "fixture"}:
        raise ValueError("Unsupported OpenAI-chat observation profile")
    if config.provider == "fixture":
        try:
            local = ipaddress.ip_address(urlsplit(config.endpoint).hostname).is_loopback
        except ValueError:
            local = False
        if not local:
            raise ValueError("Fixture observation profile requires a literal loopback endpoint")
    route = config.readiness.pending_status
    if route is not None and (config.provider != "fixture" or route.profile != "openai_chat_fixture"):
        raise ValueError("Unsupported pending observation profile; hosted route remains unverified")


def _retry_hint(response):
    value = response.headers.get("Retry-After")
    return None if value is None else RetryAfterHint(value, datetime.now(UTC).isoformat())


def exchange(config, credential, payload, parameters, context, *, pending=None, parse_response,
             http_failure, transport_failure):
    """Exactly one POST or a model-bound same-origin status GET, owned to EOF."""
    policy = config.readiness
    url, method = config.endpoint, "POST"
    if pending is not None:
        url = policy.pending_status.status_url(pending.request_id, model=config.model, endpoint=config.endpoint)
        method = "GET"
    headers = {"Content-Type": "application/json"}
    if credential is not None:
        headers["Authorization"] = f"Bearer {credential}"
    assembler = None
    failure = None
    try:
        with requests.Session() as session:
            session.trust_env = False
            session.mount("http://", requests.adapters.HTTPAdapter(max_retries=0))
            session.mount("https://", requests.adapters.HTTPAdapter(max_retries=0))
            context.check()
            with session.request(method, url, headers=headers, json=payload if pending is None else None,
                                 timeout=(context.connect_timeout_s, context.socket_idle_timeout_s),
                                 allow_redirects=False, stream=True) as response:
                hint = _retry_hint(response)
                request_id = response.headers.get("x-request-id")
                if response.status_code == 202:
                    name = "NVCF-REQID" if policy.pending_status is None else policy.pending_status.request_id_header
                    try:
                        return PendingReceipt(context.operation_id, config.model, response.headers.get(name),
                                              datetime.now(UTC).isoformat(), response.headers.get("NVCF-STATUS"), hint)
                    except ValueError:
                        raise RequestError("Accepted work has no verified request identity", status_code=202,
                                           code="pending_identity_unavailable", retry_after=hint) from None
                streaming = (200 <= response.status_code < 300 and parameters.get("stream") is True)
                if streaming:
                    if response.headers.get("Content-Type", "").split(";")[0].strip().lower() != "text/event-stream":
                        raise RequestError("Expected event-stream response", code="invalid_response",
                                           status_code=response.status_code)
                    assembler = StreamAssembler(max_event_bytes=policy.max_event_bytes,
                                                max_response_bytes=policy.max_response_bytes,
                                                requested_settings=parameters, request_id=request_id,
                                                on_progress=context.on_progress)
                chunks, size = [], 0
                # A one-byte SSE read exposes small events immediately, rather
                # than blocking until a large chunk or an entire line arrives.
                # JSON reads are bounded too and never use Response.json/text.
                for chunk in response.iter_content(chunk_size=1 if streaming else 4096):
                    context.check()
                    if assembler is not None:
                        assembler.feed(chunk)
                    else:
                        size += len(chunk)
                        if size > policy.max_response_bytes:
                            raise RequestError("Provider response exceeds configured byte limit", code="response_limit")
                        chunks.append(chunk)
                context.check()
                if assembler is not None:
                    context.on_progress(assembler.progress)
                    return assembler.finish()
                raw = b"".join(chunks)
                try:
                    body = json.loads(raw)
                except (ValueError, UnicodeError):
                    body = None
                if not 200 <= response.status_code < 300:
                    error = (http_failure(response, body) if body is not None else
                             RequestError(f"HTTP {response.status_code}: completion request rejected",
                                          status_code=response.status_code, detail=raw.decode("utf-8", errors="replace")))
                    error.retry_after, error.request_id = hint, request_id
                    raise error
                try:
                    return parse_response(response, body, parameters)
                except RequestError as error:
                    error.status_code = response.status_code
                    error.request_id = error.request_id or request_id
                    raise
    except requests.RequestException as error:
        failure = transport_failure(error)
        if assembler is not None:
            failure.response_evidence = assembler.partial(
                PartialResponseReason.CANCELLED if context.cancellation.cancelled else PartialResponseReason.INTERRUPTED)
    except Exception as error:
        if assembler is None or isinstance(error, RequestError):
            if assembler is not None:
                context.on_progress(assembler.progress)
            raise
        # Cooperative stop/callback exceptions still retain the observed bytes.
        # Readiness checks its guard before interpreting this transport result.
        failure = RequestError("Observed response interrupted", code="interrupted",
                               response_evidence=assembler.partial(
                                   PartialResponseReason.CANCELLED if context.cancellation.cancelled
                                   else PartialResponseReason.INTERRUPTED))
    if assembler is not None:
        context.on_progress(assembler.progress)
    raise failure from None
