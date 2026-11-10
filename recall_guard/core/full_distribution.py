"""Full-distribution evidence for one response and the client that requests it.

A full-distribution host teacher-forces prompt plus response on a pinned model
digest. For every response position (prompt positions are excluded) it reports
the actual token's log-probability, the entropy of the full next-token
distribution, the actual token's rank and the strongest alternatives taken from
the full vector. Log-probabilities and entropies use natural-log units.

The host sits behind a bearer gateway. The bearer value is read from the
environment at request time and is never stored, logged or included in errors.
A batch request (task 6.4) sends one prompt with several responses; the host
evaluates the prompt once and returns one evidence object per response.
"""

import math
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlsplit

import numpy as np
import requests

from recall_guard.core.readiness_types import _fields
from recall_guard.core.request_control import RequestError

#: Host route, relative to the gateway base URL.
ENDPOINT_PATH = "/v1/full_distribution"
#: Batch route (task 6.4): one prompt, several responses, one evidence object per response in order.
BATCH_ENDPOINT_PATH = "/v1/full_distribution_batch"
_CREDENTIAL_REF = re.compile(r"(?:env:)?[A-Za-z_][A-Za-z0-9_]*")
_STATUS_CODES = {401: "auth_failed", 409: "digest_mismatch", 413: "oversize", 422: "tokenisation_failed",
                 503: "host_busy"}
#: The host answers 503 ``memory_low`` while free physical memory is below its floor; the caller may retry later.
_RETRYABLE_CODES = frozenset({"host_busy"})
_DETAIL_LIMIT = 2000
_REDACTED = "[redacted]"
_POSITION_FIELDS = ("response_token_ids", "logprob", "entropy", "rank", "top_logprobs")
_EVIDENCE_FIELDS = ("model_digest", "n_vocab", *_POSITION_FIELDS)


def _count(value: int, name: str, *, minimum: int, maximum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum or (maximum is not None and value > maximum):
        bound = f"at least {minimum}" if maximum is None else f"between {minimum} and {maximum}"
        raise ValueError(f"{name} must be {bound}")
    return value


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number")
    try:
        value = float(value)
    except OverflowError:
        raise ValueError(f"{name} must be a finite number") from None
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def _logprob(value: float, name: str) -> float:
    value = _finite(value, name)
    if value > 0:
        raise ValueError(f"{name} must be at most zero")
    return value


def _sequence(value, name: str) -> tuple:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple")
    return tuple(value)


@dataclass(frozen=True)
class FullDistributionEvidence:
    """Per-position evidence for one response, taken from the full next-token vector.

    Every per-position tuple is indexed by response position. ``logprob`` is the
    actual token's log-probability, ``entropy`` the entropy of the full
    distribution and ``rank`` the actual token's one-based rank (1 is the most
    likely token). ``top_logprobs[i]`` holds ``(token_id, logprob)`` pairs sorted
    by descending logprob, at most the requested ``top_k_keep``, so the
    alternatives for any smaller k are a prefix. Token ids index a vocabulary of
    ``n_vocab`` entries. Lists from JSON are frozen into tuples.
    """

    model_digest: str
    n_vocab: int
    response_token_ids: tuple[int, ...]
    logprob: tuple[float, ...]
    entropy: tuple[float, ...]
    rank: tuple[int, ...]
    top_logprobs: tuple[tuple[tuple[int, float], ...], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.model_digest, str) or not self.model_digest.strip():
            raise ValueError("model_digest must be nonempty text")
        n_vocab = _count(self.n_vocab, "n_vocab", minimum=1)
        columns = {name: _sequence(getattr(self, name), name) for name in _POSITION_FIELDS}
        if len({len(column) for column in columns.values()}) != 1:
            raise ValueError("per-position fields must have equal lengths")
        frozen = {
            "response_token_ids": tuple(_count(value, "response token id", minimum=0, maximum=n_vocab - 1)
                                        for value in columns["response_token_ids"]),
            "logprob": tuple(_logprob(value, "logprob") for value in columns["logprob"]),
            "entropy": tuple(self._entropy(value) for value in columns["entropy"]),
            "rank": tuple(_count(value, "rank", minimum=1, maximum=n_vocab) for value in columns["rank"]),
            "top_logprobs": tuple(self._alternatives(position, n_vocab) for position in columns["top_logprobs"]),
        }
        for name, value in frozen.items():
            object.__setattr__(self, name, value)

    @staticmethod
    def _entropy(value: float) -> float:
        value = _finite(value, "entropy")
        if value < 0:
            raise ValueError("entropy must be at least zero")
        return value

    @staticmethod
    def _alternatives(position, n_vocab: int) -> tuple[tuple[int, float], ...]:
        pairs = []
        for pair in _sequence(position, "top_logprobs position"):
            pair = _sequence(pair, "top_logprobs alternative")
            if len(pair) != 2:
                raise ValueError("top_logprobs alternatives must be (token_id, logprob) pairs")
            pairs.append((_count(pair[0], "alternative token id", minimum=0, maximum=n_vocab - 1),
                          _logprob(pair[1], "alternative logprob")))
        if len(pairs) > n_vocab or len({token_id for token_id, _ in pairs}) != len(pairs):
            raise ValueError("top_logprobs alternatives must be distinct vocabulary entries")
        if any(later[1] > earlier[1] for earlier, later in zip(pairs, pairs[1:], strict=False)):
            raise ValueError("top_logprobs alternatives must be sorted by descending logprob")
        return tuple(pairs)

    def to_dict(self) -> dict:
        """Return detached JSON-compatible lists in the host's wire shape."""
        return {
            "model_digest": self.model_digest, "n_vocab": self.n_vocab,
            "response_token_ids": list(self.response_token_ids), "logprob": list(self.logprob),
            "entropy": list(self.entropy), "rank": list(self.rank),
            "top_logprobs": [[list(pair) for pair in position] for position in self.top_logprobs],
        }

    @classmethod
    def from_dict(cls, data) -> "FullDistributionEvidence":
        """Load the exact field set; unknown or missing fields are refused."""
        return cls(**_fields(data, _EVIDENCE_FIELDS, required=_EVIDENCE_FIELDS))


def logprobs_from_logits(logits: np.ndarray) -> np.ndarray:
    """Return the float64 log-softmax of each row of ``logits``.

    ``logits`` has shape (positions, vocabulary); the result has the same shape
    and ``exp`` of each row sums to one. Each row's maximum is subtracted before
    exponentiation, so large logits such as 1e4 neither overflow nor lose the
    differences between entries.
    """
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("logits must have shape (positions, vocabulary) with a nonempty vocabulary")
    if not np.all(np.isfinite(values)):
        raise ValueError("logits must be finite")
    shifted = values - values.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def _redact(text: str, secret: str) -> str:
    """Remove the secret and its escaped repr form (as embedded in requests header errors)."""
    for form in (secret, repr(secret)[1:-1]):
        text = text.replace(form, _REDACTED)
    return text


class FullDistributionClient:
    """Request full-distribution evidence from a host through its bearer gateway.

    One call is exactly one POST to ``{base_url}/v1/full_distribution`` (or to
    ``/v1/full_distribution_batch`` for ``evidence_batch``) with
    transport retries and redirects disabled and ambient proxy or netrc
    authentication ignored, so only the referenced bearer reaches the gateway.
    ``credential_ref`` names an environment variable as ``env:NAME`` or as a
    bare ``NAME``, the same two forms ``ModelConfig.credential_ref`` accepts.
    Its value is looked up in ``environment`` (default ``os.environ``) at
    request time and is never kept on the client. A value with surrounding or
    embedded whitespace, control characters or any non-ASCII character is
    refused before any request (an RFC 6750 bearer token is ASCII only).
    """

    def __init__(self, base_url: str, *, model_digest: str, credential_ref: str,
                 timeout_s: float = 120.0, environment: Mapping[str, str] | None = None) -> None:
        try:
            parsed = urlsplit(base_url) if isinstance(base_url, str) else None
            valid = (parsed is not None and parsed.scheme in {"http", "https"} and parsed.hostname
                     and parsed.username is None and parsed.password is None and not parsed.query and not parsed.fragment)
            if valid:
                _ = parsed.port  # Validate a supplied port without echoing malformed input.
        except ValueError:
            valid = False
        if not valid:
            raise ValueError("base_url must be a credential-free HTTP(S) URL without query or fragment")
        if not isinstance(model_digest, str) or not model_digest.strip():
            raise ValueError("model_digest must be nonempty text")
        if not isinstance(credential_ref, str) or _CREDENTIAL_REF.fullmatch(credential_ref) is None:
            raise ValueError("credential_ref must name an environment variable, not contain a credential")
        if _finite(timeout_s, "timeout_s") <= 0:
            raise ValueError("timeout_s must be finite and greater than 0")
        environment = os.environ if environment is None else environment
        if not isinstance(environment, Mapping):
            raise TypeError("environment must be a mapping")
        self.base_url = base_url.rstrip("/")
        self.endpoint = self.base_url + ENDPOINT_PATH
        self.batch_endpoint = self.base_url + BATCH_ENDPOINT_PATH
        self.model_digest = model_digest
        self.credential_ref = credential_ref
        self.timeout_s = float(timeout_s)
        self._environment = environment

    def __repr__(self) -> str:
        return f"FullDistributionClient(base_url={self.base_url!r}, model_digest={self.model_digest!r})"

    def evidence(self, prompt_text: str, response_text: str, *, top_k_keep: int = 100) -> FullDistributionEvidence:
        """Teacher-force ``prompt_text`` plus ``response_text`` on the pinned digest.

        HTTP 401, 409, 413 and 422 become RequestError codes ``auth_failed``,
        ``digest_mismatch``, ``oversize`` and ``tokenisation_failed``; HTTP 503
        (the host's ``memory_low`` refusal) is ``host_busy`` (retryable); another
        status is ``http_error`` and a network failure ``transport_error``
        (retryable). A 200 body that fails validation, including a digest other
        than the requested one or more than ``top_k_keep`` alternatives at a
        position, is ``invalid_evidence``.
        """
        if not isinstance(prompt_text, str) or not isinstance(response_text, str):
            raise TypeError("prompt_text and response_text must be strings")
        if not response_text:
            raise ValueError("response_text must be nonempty")
        _count(top_k_keep, "top_k_keep", minimum=1)
        credential = self._credential()
        payload = {"model_digest": self.model_digest, "prompt_text": prompt_text,
                   "response_text": response_text, "top_k_keep": top_k_keep}
        response, failure = self._post(payload, credential)
        if failure is not None:
            raise failure
        if response.status_code != 200:
            raise self._rejection(response, credential)
        try:
            evidence = self._checked(FullDistributionEvidence.from_dict(response.json()), top_k_keep)
        except (TypeError, ValueError) as error:
            failure = RequestError(_redact(f"Invalid full-distribution evidence: {error}", credential),
                                   status_code=200, code="invalid_evidence")
        if failure is not None:
            raise failure
        return evidence

    def evidence_batch(self, prompt_text: str, response_texts, *, top_k_keep: int = 100,
                       timeout_s: float | None = None) -> tuple[FullDistributionEvidence, ...]:
        """Teacher-force several responses to one prompt in one request (task 6.4).

        The host evaluates ``prompt_text`` once and each of ``response_texts``
        against the cached prompt, so each evidence object equals what
        ``evidence`` returns for that pair. The result holds one evidence
        object per response, in order. ``timeout_s`` bounds this one request
        and defaults to the client's; the plan sets it, since a batch takes
        about one prompt evaluation plus every response's decodes (the study
        uses about 600 s per batch). Status codes map as for ``evidence``,
        including the retryable ``host_busy`` for HTTP 503. A 200 body other
        than ``{"evidence": [...]}`` holding one valid evidence object per
        response is ``invalid_evidence``.
        """
        if not isinstance(prompt_text, str):
            raise TypeError("prompt_text must be a string")
        if not isinstance(response_texts, (list, tuple)) or not all(isinstance(text, str) for text in response_texts):
            raise TypeError("response_texts must be a list or tuple of strings")
        if not response_texts or not all(response_texts):
            raise ValueError("response_texts must hold at least one response, each nonempty")
        _count(top_k_keep, "top_k_keep", minimum=1)
        if timeout_s is not None and _finite(timeout_s, "timeout_s") <= 0:
            raise ValueError("timeout_s must be finite and greater than 0")
        timeout = self.timeout_s if timeout_s is None else float(timeout_s)
        credential = self._credential()
        payload = {"model_digest": self.model_digest, "prompt_text": prompt_text,
                   "response_texts": list(response_texts), "top_k_keep": top_k_keep}
        response, failure = self._post(payload, credential, endpoint=self.batch_endpoint, timeout_s=timeout)
        if failure is not None:
            raise failure
        if response.status_code != 200:
            raise self._rejection(response, credential)
        try:
            data = response.json()
            if not isinstance(data, Mapping) or set(data) != {"evidence"}:
                raise ValueError('the batch reply must be exactly {"evidence": [...]}')
            items = data["evidence"]
            if not isinstance(items, list) or len(items) != len(response_texts):
                raise ValueError(f"the batch reply must hold one evidence object for each of the "
                                 f"{len(response_texts)} responses")
            batch = tuple(self._checked(FullDistributionEvidence.from_dict(item), top_k_keep) for item in items)
        except (TypeError, ValueError) as error:
            failure = RequestError(_redact(f"Invalid full-distribution evidence: {error}", credential),
                                   status_code=200, code="invalid_evidence")
        if failure is not None:
            raise failure
        return batch

    def _credential(self) -> str:
        """The referenced bearer, refused before any request when missing or not a visible-ASCII token."""
        credential = self._environment.get(self.credential_ref.removeprefix("env:"))
        if not isinstance(credential, str) or not credential.strip():
            raise ValueError("Referenced credential is missing or empty")
        if (credential != credential.strip() or not credential.isascii()
                or any(char.isspace() or not char.isprintable() for char in credential)):
            raise ValueError("Referenced credential contains whitespace, control or non-ASCII characters")
        return credential

    def _checked(self, evidence: FullDistributionEvidence, top_k_keep: int) -> FullDistributionEvidence:
        if evidence.model_digest != self.model_digest:
            raise ValueError("host reported a model digest other than the requested one")
        if any(len(position) > top_k_keep for position in evidence.top_logprobs):
            raise ValueError("host returned more alternatives than top_k_keep")
        return evidence

    @staticmethod
    def _rejection(response: requests.Response, credential: str) -> RequestError:
        code = _STATUS_CODES.get(response.status_code, "http_error")
        return RequestError(f"HTTP {response.status_code}: full-distribution request rejected ({code})",
                            status_code=response.status_code, code=code, retryable=code in _RETRYABLE_CODES,
                            detail=_redact(response.text, credential)[:_DETAIL_LIMIT])

    def _post(self, payload: dict, credential: str, *, endpoint: str | None = None,
              timeout_s: float | None = None) -> tuple[requests.Response | None, RequestError | None]:
        """Send one POST; a failure is returned, not raised, so no exception context is kept.

        A UnicodeError (for example latin-1 header encoding in http.client) is
        not a RequestException, and its args hold the header value, so it is
        mapped to a fixed message with no detail. It is not retryable.
        """
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {credential}"}
        try:
            with requests.Session() as session:
                session.trust_env = False
                session.mount("http://", requests.adapters.HTTPAdapter(max_retries=0))
                session.mount("https://", requests.adapters.HTTPAdapter(max_retries=0))
                return session.post(endpoint or self.endpoint, headers=headers, json=payload,
                                    timeout=self.timeout_s if timeout_s is None else timeout_s,
                                    allow_redirects=False), None
        except requests.RequestException as error:
            return None, RequestError("Full-distribution request failed in transport", code="transport_error",
                                      retryable=True, detail=_redact(str(error), credential))
        except UnicodeError:
            return None, RequestError("Full-distribution request could not be encoded", code="transport_error")
