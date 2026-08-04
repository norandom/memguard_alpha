"""NVIDIA chat-completions HTTP client with logprobs and configurable temperature.

This client always requests `logprobs=true` and `top_logprobs=20` and returns
frozen `CompletionResult` records. A logical `generate()` call may retry on
retryable HTTP failures; the default per-attempt timeout is 15 seconds.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import Lock
from typing import Any

import requests

NVIDIA_CHAT_COMPLETIONS_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
TOP_LOGPROBS = 20
DEFAULT_TIMEOUT_S = 15.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_BACKOFF_S = 2.0
RETRYABLE_HTTP_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

_log = logging.getLogger(__name__)


class LMHTTPError(RuntimeError):
    """A provider HTTP failure that carries its response status code.

    Callers classifying a rejected credential should read :attr:`status_code`
    rather than matching text in the message: a substring search for ``"401"``
    also fires on a trace id, a port, or a byte count, and under an ensemble
    that false-positive discards every draw already paid for.
    """

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


def _retry_after_seconds(response: Any) -> float | None:
    """Parse a ``Retry-After`` header (delta-seconds form) into seconds."""
    headers = getattr(response, "headers", None) or {}
    try:
        raw = headers.get("Retry-After")
    except AttributeError:
        return None
    if raw is None:
        return None
    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        # HTTP-date form is permitted by the RFC but not emitted by this
        # endpoint; fall back to the configured backoff rather than guessing.
        return None
    return seconds if seconds >= 0 else None


@dataclass(frozen=True)
class TokenLogprob:
    """Per-token logprob record returned by the NVIDIA OpenAI-compatible API."""

    token: str
    logprob: float
    top_logprobs: list[dict[str, Any]]


@dataclass(frozen=True)
class CompletionResult:
    """Single chat-completion response with logprobs.

    Attributes
    ----------
    content:
        Assistant message content.
    logprobs:
        Per-token logprob entries. ``top_logprobs`` length is enforced to be
        non-empty by the client (raises on missing data).
    raw_temperature_observed:
        The temperature the API reported as honoured, when exposed. ``None``
        when the API does not echo the temperature back.
    """

    content: str
    logprobs: list[TokenLogprob]
    raw_temperature_observed: float | None


class NvidiaLM:
    """Thin HTTP client around the NVIDIA OpenAI-compatible chat endpoint.

    Always sends ``logprobs=True`` and ``top_logprobs=20``. The default
    ``temperature`` is 0.0 (per Req 10.3) and can be overridden per call.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_backoff_s: float = DEFAULT_RETRY_BACKOFF_S,
        min_call_interval_s: float = 0.0,
    ) -> None:
        if not api_key:
            raise ValueError("api_key must be a non-empty string")
        if not model:
            raise ValueError("model must be a non-empty string")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if retry_backoff_s < 0:
            raise ValueError("retry_backoff_s must be >= 0")
        if min_call_interval_s < 0:
            raise ValueError("min_call_interval_s must be >= 0")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s
        self.min_call_interval_s = min_call_interval_s
        self._last_call_t: float | None = None
        self._pace_lock = Lock()
        self.api_base = NVIDIA_CHAT_COMPLETIONS_URL

    def _reserve_call_slot(self) -> float:
        """Reserve the next paced send slot; return seconds to wait before POST.

        The lock covers only this bookkeeping -- never the network round trip.
        Holding it across the request serialises every concurrent call through
        one client, which is what ``max_workers`` used to run into.

        ``min_call_interval_s`` is defined as the spacing between the *starts*
        of successive requests. Recording the reserved send time (rather than
        the observed completion time) makes that spacing independent of
        endpoint latency; stamping the current clock instead would let a
        still-sleeping thread's slot be handed out twice.

        ``max(now, ...)`` stops an idle client from banking credit and then
        issuing a burst.
        """
        with self._pace_lock:
            now = time.monotonic()
            if self.min_call_interval_s <= 0 or self._last_call_t is None:
                slot = now
            else:
                slot = max(now, self._last_call_t + self.min_call_interval_s)
            self._last_call_t = slot
        return slot - now

    def generate(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> CompletionResult:
        """Send a single chat completion and return parsed logprobs.

        Caps response length at ``max_tokens`` (default 512) so reasoning
        models (gpt-oss-*, nemotron-nano-*) have enough budget to finish
        their reasoning chain AND emit the final ``Direction:`` /
        ``Confidence:`` lines. Non-reasoning models stop early on EOS so
        the higher cap costs nothing for them.

        Raises
        ------
        TimeoutError
            If the underlying HTTP call times out.
        RuntimeError
            If the response body lacks ``logprobs.content`` or any token entry
            is missing its ``top_logprobs`` list.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "logprobs": True,
            "top_logprobs": TOP_LOGPROBS,
        }
        def _paced_post() -> requests.Response:
            wait = self._reserve_call_slot()
            if wait > 0:
                time.sleep(wait)
            return requests.post(
                self.api_base,
                headers=headers,
                json=payload,
                timeout=self.timeout_s,
            )

        last_timeout_exc: Exception | None = None
        last_runtime_exc: Exception | None = None
        last_status: int | None = None
        for attempt in range(self.max_retries + 1):
            retry_after: float | None = None
            try:
                response = _paced_post()
                response.raise_for_status()
                return self._parse_response(response.json())
            except requests.exceptions.Timeout as exc:
                last_timeout_exc = exc
                last_runtime_exc = None
                retryable = True
            except requests.exceptions.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                last_runtime_exc = exc
                last_timeout_exc = None
                last_status = status
                retryable = status in RETRYABLE_HTTP_STATUS
                if not retryable:
                    raise LMHTTPError(
                        f"Model {self.model} request failed: {exc}",
                        status_code=status,
                    ) from exc
                retry_after = _retry_after_seconds(exc.response)
            except requests.exceptions.RequestException as exc:
                last_runtime_exc = exc
                last_timeout_exc = None
                retryable = True

            if attempt < self.max_retries and retryable:
                backoff = self._retry_delay(attempt, retry_after)
                # Logged at DEBUG so a parallel run (8 workers * 50 prompts) does
                # not spam stderr. Final failures still surface via the
                # TimeoutError/RuntimeError raised below, which the evaluator
                # converts into a fail_reason on the row.
                _log.debug(
                    "NvidiaLM transient failure for %s (attempt %d/%d); retrying in %.1fs",
                    self.model, attempt + 1, self.max_retries + 1, backoff,
                )
                time.sleep(backoff)
                continue
            break

        if last_timeout_exc is not None:
            raise TimeoutError(
                f"Model {self.model} timed out after {self.timeout_s} seconds "
                f"(after {self.max_retries + 1} attempt(s))."
            ) from last_timeout_exc
        raise LMHTTPError(
            f"Model {self.model} request failed after {self.max_retries + 1} attempt(s): "
            f"{last_runtime_exc}",
            status_code=last_status,
        ) from last_runtime_exc

    def _retry_delay(self, attempt: int, retry_after: float | None) -> float:
        """Seconds to wait before the next attempt.

        An endpoint-supplied ``Retry-After`` wins outright. Otherwise the
        exponential backoff is fully jittered: rate-limited responses come back
        fast, so without jitter every concurrent draw would sleep for exactly
        the same interval and retry in unison, reproducing the burst that
        triggered the limit.
        """
        if retry_after is not None:
            return retry_after
        ceiling = self.retry_backoff_s * (2 ** attempt)
        return random.uniform(0.0, ceiling) if ceiling > 0 else 0.0

    def _parse_response(self, data: dict[str, Any]) -> CompletionResult:
        try:
            choice = data["choices"][0]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"Model {self.model} response missing 'choices[0]': {data!r}"
            ) from exc
        if not isinstance(choice, dict):
            raise RuntimeError(
                f"Model {self.model} response has malformed 'choices[0]': {choice!r}"
            )

        message = choice.get("message", {}) or {}
        if not isinstance(message, dict):
            raise RuntimeError(
                f"Model {self.model} response has malformed 'message': {message!r}"
            )
        content = message.get("content")
        if not content:
            # Reasoning models (gpt-oss-*, nemotron-nano-*) put output under
            # 'reasoning_content' until reasoning completes. If the answer
            # field is empty, fall back to reasoning_content so the parser
            # can still find Direction:/Confidence: lines.
            content = message.get("reasoning_content") or ""

        logprobs_section = choice.get("logprobs")
        if not isinstance(logprobs_section, dict) or "content" not in logprobs_section:
            raise RuntimeError(
                f"Model {self.model} response missing 'logprobs.content'; "
                "cannot compute MIA features without per-token logprobs."
            )

        token_entries = logprobs_section["content"]
        if not isinstance(token_entries, list) or not token_entries:
            raise RuntimeError(
                f"Model {self.model} response has empty or malformed 'logprobs.content'; "
                "cannot compute MIA features without per-token logprobs."
            )
        parsed: list[TokenLogprob] = []
        for idx, entry in enumerate(token_entries):
            if not isinstance(entry, dict):
                raise RuntimeError(
                    f"Model {self.model} response token #{idx} is malformed: {entry!r}"
                )
            if "top_logprobs" not in entry:
                raise RuntimeError(
                    f"Model {self.model} response token #{idx} is missing "
                    "'top_logprobs'; required for MIA feature computation."
                )
            top_logprobs = entry["top_logprobs"]
            if not isinstance(top_logprobs, list) or not top_logprobs:
                raise RuntimeError(
                    f"Model {self.model} response token #{idx} has empty or malformed "
                    "'top_logprobs'; required for MIA feature computation."
                )
            if "logprob" not in entry:
                raise RuntimeError(
                    f"Model {self.model} response token #{idx} is missing 'logprob'; "
                    "required for MIA feature computation."
                )
            try:
                logprob = float(entry["logprob"])
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Model {self.model} response token #{idx} has non-numeric 'logprob'."
                ) from exc
            parsed.append(
                TokenLogprob(
                    token=str(entry.get("token", "")),
                    logprob=logprob,
                    top_logprobs=list(top_logprobs),
                )
            )

        raw_temp = choice.get("temperature")
        if raw_temp is None:
            raw_temp = data.get("temperature")
        observed = float(raw_temp) if isinstance(raw_temp, (int, float)) else None

        return CompletionResult(
            content=content,
            logprobs=parsed,
            raw_temperature_observed=observed,
        )


def generate_many(
    lm: NvidiaLM,
    prompts: Sequence[str],
    *,
    max_workers: int = 8,
) -> list[CompletionResult | Exception]:
    """Run ``lm.generate`` over many prompts in parallel.

    Returns a list aligned with ``prompts`` (preserves input order).
    Per-prompt failures are returned as the raised exception object so
    the caller can inspect or skip them; nothing is re-raised. The LM's
    own ``generate`` defaults are used (temperature=0, max_tokens=512).
    """
    if not prompts:
        return []
    if max_workers < 1:
        max_workers = 1

    def _one(prompt: str) -> CompletionResult | Exception:
        try:
            return lm.generate(prompt)
        except Exception as exc:  # noqa: BLE001 - caller decides what to do
            return exc

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(_one, prompts))

