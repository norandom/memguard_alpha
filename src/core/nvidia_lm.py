"""NVIDIA chat-completions HTTP client with logprobs and configurable temperature.

Implements the `core.nvidia_lm` component from the honest-model-ranking design:
single request per call, 15 s hard timeout, always-on `logprobs=true,
top_logprobs=20`, and a frozen `CompletionResult` return type.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests

NVIDIA_CHAT_COMPLETIONS_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
TOP_LOGPROBS = 20
DEFAULT_TIMEOUT_S = 15.0


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
    ) -> None:
        if not api_key:
            raise ValueError("api_key must be a non-empty string")
        if not model:
            raise ValueError("model must be a non-empty string")
        self.api_key = api_key
        self.model = model
        self.timeout_s = timeout_s
        self.api_base = NVIDIA_CHAT_COMPLETIONS_URL

    def generate(self, prompt: str, temperature: float = 0.0) -> CompletionResult:
        """Send a single chat completion and return parsed logprobs.

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
            "logprobs": True,
            "top_logprobs": TOP_LOGPROBS,
        }
        try:
            response = requests.post(
                self.api_base,
                headers=headers,
                json=payload,
                timeout=self.timeout_s,
            )
            response.raise_for_status()
        except requests.exceptions.Timeout as exc:
            raise TimeoutError(
                f"Model {self.model} timed out after {self.timeout_s} seconds."
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Model {self.model} request failed: {exc}") from exc

        return self._parse_response(response.json())

    def _parse_response(self, data: dict[str, Any]) -> CompletionResult:
        try:
            choice = data["choices"][0]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"Model {self.model} response missing 'choices[0]': {data!r}"
            ) from exc

        content = choice.get("message", {}).get("content", "")

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
