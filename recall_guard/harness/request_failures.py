"""Shared stop categories for configured inference and calibration queues.

Transport adapters still own retries and redaction. This helper classifies only
their sanitized failures; it neither retries nor interprets arbitrary messages.
:func:`describe_failure` and :func:`render_failure_panel` turn a failed host
preflight or a sanitized request failure into plain-English next steps.
"""

import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from recall_guard.core.profiles import HostPreflightReport
from recall_guard.core.request_control import BudgetExhaustedError, RequestError

FATAL_FAILURES = frozenset({"budget_exhausted", "authentication_error", "configuration_error", "eligibility_error",
                            "cancelled", "deferred", "exhausted"})
_INELIGIBLE_CODES = frozenset({
    "insufficient_quota", "billing_hard_limit_reached", "billing_not_active",
    "credit_balance_exhausted", "insufficient_credits", "usage_limit_reached",
})
_CONFIGURATION_CODES = frozenset({
    "missing_token_evidence", "unverified_request_accounting", "configuration_changed",
    "response_settings_mismatch", "unsupported_parameter", "invalid_parameter",
})


def failure_kind(error: Exception) -> str:
    """Keep transient rate limiting distinct from hard credit/quota failures."""
    if isinstance(error, BudgetExhaustedError):
        return "budget_exhausted"
    if not isinstance(error, RequestError):
        return "runtime_error"
    if error.status_code in (401, 403):
        return "authentication_error"
    if error.status_code == 402 or error.code in _INELIGIBLE_CODES:
        return "eligibility_error"
    if error.code in {"caller_cancelled", "cancelled"}:
        return "cancelled"
    if error.code in {"deferred", "waiting_allowance", "pending_route_unavailable", "retry_after_too_large"}:
        return "deferred"
    if error.code in {"exhausted", "request_budget", "retry_limit"}:
        return "exhausted"
    if error.parameter or error.status_code in (400, 404, 422) or error.code in _CONFIGURATION_CODES:
        return "configuration_error"
    return "request_error"


def rewrite_failure_kind(kind: str | None) -> str:
    """Normalize policy-owned configuration failures at orchestration boundaries."""
    return "configuration_error" if kind in {"configuration_changed", "response_settings_mismatch"} else kind or "fidelity_rejected"


def request_failure(error: Exception) -> tuple[str, str, dict | None]:
    """Trust sanitized RequestErrors; never retain arbitrary exception text."""
    kind = failure_kind(error)
    if isinstance(error, BudgetExhaustedError):
        return kind, "Physical request budget exhausted.", None
    if isinstance(error, RequestError):
        diagnostics = {"status_code": error.status_code, "parameter": error.parameter, "code": error.code,
                       "detail": retained_detail(error.detail), "retryable": error.retryable,
                       "usage": dict(error.usage) if error.usage is not None else None}
        if trace := getattr(error, "readiness_trace", None):
            diagnostics["readiness_trace"] = tuple(item.to_dict() for item in trace)
            diagnostics["response_evidence"] = (None if error.response_evidence is None
                                                 else error.response_evidence.to_dict())
        return kind, str(error), diagnostics
    return kind, "Workflow operation failed; exception text was not retained.", None


# --- Friendly failure presentation (R2.7, design 4.6) ------------------------

#: Bound on retained provider error bodies (characters, after redaction).
DETAIL_LIMIT = 1000
_SECRET_PATTERNS = (
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=\-]{8,}"),
    re.compile(r"(?i)((?:api[_-]?key|authorization|x-api-key|access[_-]?token|secret)[\"']?\s*[:=]\s*[\"']?)"
               r"[^\s\"',}]{6,}"),
    re.compile(r"()\b(?:sk|nvapi|hf)-[A-Za-z0-9_\-]{8,}"),
)


def retained_detail(detail: object, *, secrets: Sequence[str] = (), limit: int = DETAIL_LIMIT) -> str | None:
    """Bounded, defensively redacted provider error body for receipts and panels.

    Registered secrets were already removed by the owning RequestControl; this
    adds caller secrets and credential-shaped tokens before truncating.
    """
    if detail is None:
        return None
    text = str(detail)
    for secret in sorted({item for item in secrets if item}, key=len, reverse=True):
        text = text.replace(secret, "[REDACTED]")
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(lambda match: match.group(1) + "[REDACTED]", text)
    text = text.strip()
    if not text:
        return None
    return text if len(text) <= limit else text[:limit] + " ...[truncated]"


def retained_failure(error: Exception, *, secrets: Sequence[str] = ()) -> dict | None:
    """Status, code and bounded body of a provider HTTP failure; None otherwise.

    Transport failures without a provider response (timeouts, disconnects)
    carry only exception text, which is never retained.
    """
    if not isinstance(error, RequestError) or error.status_code is None:
        return None
    return {"status_code": error.status_code, "code": error.code,
            "detail": retained_detail(error.detail, secrets=secrets)}


@dataclass(frozen=True)
class FailurePresentation:
    """Plain-English explanation and next steps for one failure."""

    kind: str
    title: str
    explanation: str
    next_steps: tuple[str, ...] = ()
    detail: str | None = None
    status_code: int | None = None
    profile_id: str | None = None
    stage: str | None = None

    def to_dict(self) -> dict:
        return {"kind": self.kind, "title": self.title, "explanation": self.explanation,
                "next_steps": list(self.next_steps), "detail": self.detail, "status_code": self.status_code,
                "profile_id": self.profile_id, "stage": self.stage}


_PREFLIGHT_TITLES = {
    "insufficient_free_ram": "Not enough free memory for this model",
    "host_memory_unknown": "Free memory on the model host is unknown",
    "model_not_installed": "Model is not installed on this Ollama host",
    "host_metadata_unavailable": "Ollama did not answer",
    "missing_credential": "A credential is missing",
    "missing_tokenizer": "The pinned tokenizer is missing",
}
_OVERLOADED_CODES = frozenset({"overloaded", "overloaded_error", "service_unavailable"})
_MAPPING_CODES = frozenset({"missing_token_evidence", "response_mapping_failed"})
_TIMEOUT_CODES = frozenset({"timeout", "connection_error", "transport_error", "disconnect"})
#: Start of the alternatives sentence that profiles.preflight_host appends to its recommendation.
_ALTERNATIVES_SENTENCE = re.compile(r" (?=Try --profile |Run on a machine |Or use --profile )")


def _preflight_presentation(report: HostPreflightReport, profile_id, stage) -> FailurePresentation:
    steps = []
    if report.reason in {"insufficient_free_ram", "host_memory_unknown"}:
        steps = [f"Use --profile {item}" for item in report.fallback_profile_ids]
    if report.reason == "insufficient_free_ram" and report.required_free_ram_bytes:
        steps.append(f"Run on a machine with at least {report.required_free_ram_bytes / 1024 ** 3:.1f} GiB of free RAM")
    explanation = report.recommendation
    if steps:
        # The alternatives are listed once, under "What you can do"; keep only the facts here.
        explanation = _ALTERNATIVES_SENTENCE.split(explanation, maxsplit=1)[0].rstrip()
    return FailurePresentation(report.reason or "preflight_failed",
                               _PREFLIGHT_TITLES.get(report.reason, "Host preflight did not pass"),
                               explanation, tuple(steps),
                               profile_id=profile_id or report.profile_id, stage=stage)


def _request_kind(error: RequestError) -> str:
    status, code = error.status_code, error.code
    if status in (401, 403):
        return "authentication_error"
    if status == 402 or code in _INELIGIBLE_CODES:
        return "eligibility_error"
    if status == 429 or code == "rate_limited":
        return "rate_limited"
    if status in (503, 529) or code in _OVERLOADED_CODES:
        return "overloaded"
    if status == 504 or code in _TIMEOUT_CODES:
        return "timeout"
    if code in _MAPPING_CODES:
        return code
    if status is not None and status >= 500:
        return "http_500"
    return failure_kind(error)


_REQUEST_TEXT = {
    "authentication_error": ("The provider rejected the credentials",
                             "The endpoint answered 401/403: the key is missing, wrong, expired or lacks access.",
                             ("Check the credential environment variable named by the profile.",
                              "Confirm the key has access to this model.")),
    "eligibility_error": ("The account has no remaining credit or quota",
                          "The provider refused the request for billing or quota reasons; retrying will not help.",
                          ("Check the provider account's credit and quota.", "Or pick another --profile.")),
    "rate_limited": ("The provider is rate limiting requests",
                     "HTTP 429: too many requests in a short time. This is temporary.",
                     ("Wait a little and run again.", "Lower concurrency if you raised it.")),
    "overloaded": ("The provider is overloaded or unavailable",
                   "HTTP 503/529: the service could not take the request right now.",
                   ("Try again later.", "Or use another --profile.")),
    "timeout": ("The request timed out; its outcome is unknown",
                "No answer arrived in time. The provider may still have processed (and billed) the request.",
                ("Check the provider before running again; nothing is retried automatically.",)),
    "missing_token_evidence": ("The endpoint returned no usable token evidence",
                               "The answer lacked aligned token log probabilities, which scoring requires.",
                               ("Use a profile whose endpoint returns logprobs with top alternatives.",)),
    "response_mapping_failed": ("The response could not be mapped",
                                "The response did not match the profile's response mapping (for example Harmony).",
                                ("Check that the profile's response mapping and pinned tokenizer match the server.",)),
    "http_500": ("The provider failed internally",
                 "The server returned an internal error. For Ollama this often means the model could not load, "
                 "for example because of too little free memory.",
                 ("Read the retained error body below.", "Check free RAM, or use a smaller or remote --profile.")),
}


def describe_failure(error: object, *, profile_id: str | None = None, stage: str | None = None) -> FailurePresentation:
    """Map a preflight report or sanitized failure to friendly text; never raw exception text."""
    report = getattr(error, "report", error)
    if isinstance(report, HostPreflightReport):
        return _preflight_presentation(report, profile_id, stage)
    if isinstance(error, RequestError):
        kind = _request_kind(error)
        title, explanation, steps = _REQUEST_TEXT.get(kind, (
            "The request failed", "The provider request did not succeed.", ("Read the retained error body below.",)))
        return FailurePresentation(kind, title, explanation, steps, detail=retained_detail(error.detail),
                                   status_code=error.status_code, profile_id=profile_id, stage=stage)
    kind = failure_kind(error) if isinstance(error, Exception) else "runtime_error"
    return FailurePresentation(kind, "The operation failed", "The operation stopped; exception text was not retained.",
                               profile_id=profile_id, stage=stage)


def render_failure_panel(presentation: FailurePresentation) -> Panel:
    """Rich panel: explanation, next steps, then the bounded sanitized body."""
    body = Text(presentation.explanation)
    if presentation.next_steps:
        body.append("\n\nWhat you can do:", style="bold")
        for step in presentation.next_steps:
            body.append(f"\n  - {step}")
    if presentation.detail:
        body.append("\n\nProvider said: ", style="bold")
        body.append(presentation.detail, style="dim")
    status = None if presentation.status_code is None else f"HTTP {presentation.status_code}"
    labels = [item for item in (presentation.profile_id, presentation.stage, status) if item]
    return Panel(body, title=presentation.title, subtitle=" | ".join(labels) or None, border_style="red",
                 expand=False)


def print_failure(error: object, *, profile_id: str | None = None, stage: str | None = None, file=None) -> None:
    """Render the friendly panel to ``file`` (default stderr); never a traceback."""
    console = Console(file=file or sys.stderr, highlight=False)
    console.print(render_failure_panel(describe_failure(error, profile_id=profile_id, stage=stage)))
