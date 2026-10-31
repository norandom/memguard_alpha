"""Shared stop categories for configured inference and calibration queues.

Transport adapters still own retries and redaction. This helper classifies only
their sanitized failures; it neither retries nor interprets arbitrary messages.
"""

from recall_guard.core.request_control import BudgetExhaustedError, RequestError

FATAL_FAILURES = frozenset({"budget_exhausted", "authentication_error", "configuration_error", "eligibility_error"})
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
        return kind, str(error), {"status_code": error.status_code, "parameter": error.parameter, "code": error.code,
                                  "detail": error.detail, "retryable": error.retryable,
                                  "usage": dict(error.usage) if error.usage is not None else None}
    return kind, "Workflow operation failed; exception text was not retained.", None
