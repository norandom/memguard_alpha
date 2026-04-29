"""NVIDIA chat-completions HTTP client with logprobs and configurable temperature.

Implements the `core.nvidia_lm` component from the honest-model-ranking design:
single request per call, 15 s hard timeout, always-on `logprobs=true,
top_logprobs=20`, and a frozen `CompletionResult` return type.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Sequence

import requests

NVIDIA_CHAT_COMPLETIONS_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
TOP_LOGPROBS = 20
DEFAULT_TIMEOUT_S = 15.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_BACKOFF_S = 2.0
RETRYABLE_HTTP_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

_log = logging.getLogger(__name__)


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
        self.api_base = NVIDIA_CHAT_COMPLETIONS_URL

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
        # Pace requests to stay under the API's rate limit. Wait the
        # remaining gap since the last call (per-instance) before issuing
        # a new request.
        if self.min_call_interval_s > 0 and self._last_call_t is not None:
            elapsed = time.monotonic() - self._last_call_t
            wait = self.min_call_interval_s - elapsed
            if wait > 0:
                time.sleep(wait)

        last_timeout_exc: Exception | None = None
        last_runtime_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    self.api_base,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout_s,
                )
                response.raise_for_status()
                self._last_call_t = time.monotonic()
                return self._parse_response(response.json())
            except requests.exceptions.Timeout as exc:
                last_timeout_exc = exc
                last_runtime_exc = None
                retryable = True
            except requests.exceptions.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else None
                last_runtime_exc = exc
                last_timeout_exc = None
                retryable = status in RETRYABLE_HTTP_STATUS
                if not retryable:
                    raise RuntimeError(
                        f"Model {self.model} request failed: {exc}"
                    ) from exc
            except requests.exceptions.RequestException as exc:
                last_runtime_exc = exc
                last_timeout_exc = None
                retryable = True

            if attempt < self.max_retries and retryable:
                backoff = self.retry_backoff_s * (2 ** attempt)
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
        raise RuntimeError(
            f"Model {self.model} request failed after {self.max_retries + 1} attempt(s): "
            f"{last_runtime_exc}"
        ) from last_runtime_exc

    def _parse_response(self, data: dict[str, Any]) -> CompletionResult:
        try:
            choice = data["choices"][0]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"Model {self.model} response missing 'choices[0]': {data!r}"
            ) from exc

        message = choice.get("message", {}) or {}
        content = message.get("content")
        if not content:
            # Reasoning models (gpt-oss-*, nemotron-nano-*) put output under
            # 'reasoning_content' until reasoning completes. If the answer
            # field is empty, fall back to reasoning_content so the parser
            # can still find Direction:/Confidence: lines.
            content = message.get("reasoning_content") or ""

        logprobs_section = choice.get("logprobs")
        if not logprobs_section or "content" not in logprobs_section:
            raise RuntimeError(
                f"Model {self.model} response missing 'logprobs.content'; "
                "cannot compute MIA features without per-token logprobs."
            )

        token_entries = logprobs_section["content"]
        parsed: list[TokenLogprob] = []
        for idx, entry in enumerate(token_entries):
            if "top_logprobs" not in entry:
                raise RuntimeError(
                    f"Model {self.model} response token #{idx} is missing "
                    "'top_logprobs'; required for MIA feature computation."
                )
            parsed.append(
                TokenLogprob(
                    token=entry.get("token", ""),
                    logprob=float(entry.get("logprob", 0.0)),
                    top_logprobs=list(entry["top_logprobs"]),
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
    lm: "NvidiaLM",
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
        except Exception as exc:  # noqa: BLE001 — caller decides what to do
            return exc

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(_one, prompts))

