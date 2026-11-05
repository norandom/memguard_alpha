"""Response-bearing probes for transports without adaptive observation.

The supplied operation is the provider's ordinary configured probe exchange;
it owns admission/retries and redaction. This helper never generates a second
answer, changes a transport policy, or populates a historical probe cache.
"""

from datetime import UTC, datetime
from time import monotonic
from uuid import uuid4

from recall_guard.core.completion import CapabilityEvidence, CapabilityReport, ModelResponse
from recall_guard.core.readiness_evidence import ProbeEvidence
from recall_guard.core.readiness_types import ReadinessObservation
from recall_guard.core.request_control import BudgetExhaustedError, RequestError

TOKEN_CONTROLS = frozenset({"logprobs", "top_logprobs"})


def legacy_generation(config, features, budget, operation, report_response, *, stage, cancellation=None,
                      on_observation=None, sanitize_failure):
    """Run an explicitly controlled generation through the existing probe evidence seam.

    The operation retains ordinary transport admission/retries and redaction.
    Already admitted synchronous I/O completes under its transport bounds.
    """
    evidence = legacy_probe(config, features, budget, operation, report_response, stage=stage,
                            cancellation=cancellation, on_observation=on_observation)
    if evidence.ready:
        from dataclasses import replace

        return replace(evidence.response, readiness_trace=evidence.observations)
    terminal = evidence.observations[-1]
    error = RequestError("Controlled generation did not satisfy the selected capabilities.",
                         code=(terminal.reason or terminal.state).value, status_code=terminal.status_code,
                         response_evidence=evidence.response)
    error.readiness_trace = evidence.observations
    error = sanitize_failure(error)
    raise error


def rejected_token_control(error, config):
    """Only typed, exact supplied-control parameter rejection is sufficient."""
    return (isinstance(error, RequestError) and error.status_code in {400, 422}
            and error.code in {None, "invalid_parameter", "invalid_request_error", "unsupported_parameter"}
            and error.parameter in TOKEN_CONTROLS and error.parameter in config.parameters)


def legacy_probe(config, features, budget, operation, report_response, *, stage, cancellation=None,
                 on_observation=None):
    """Retain one configured logical probe, including every admitted retry.

    Cancellation guards every physical admission and interrupts local waits.
    Legacy synchronous transports retain their retry policy and blocking-I/O limits.
    Failed physical diagnostics remain in the sanitized shared budget ledger.
    """
    started, used, response = monotonic(), budget.used, None
    operation_id = str(uuid4())
    unknown = CapabilityEvidence("unverified", "Configured inference probe did not complete")
    report = CapabilityReport({name: unknown for name in config.inspect(features).capabilities}, features)
    state, reason, status_code = "rejected", "malformed_response", None
    def guard():
        if cancellation.cancelled:
            raise RequestError("Caller cancelled the probe", code="caller_cancelled")

    if cancellation is not None and cancellation.cancelled:
        state, reason = "cancelled", "caller_cancelled"
    else:
        try:
            response = operation(None if cancellation is None else guard,
                                 None if cancellation is None else cancellation.wait)
            report = report_response(response)
            candidate = ProbeEvidence(config, report, response, (ReadinessObservation(
                operation_id, stage, config.model, datetime.now(UTC).isoformat(), monotonic() - started,
                "rejected", reason="missing_token_evidence"),))
            if candidate._has_ready_evidence():
                state, reason = "ready", None
            elif isinstance(response, ModelResponse) and response.content.strip() and response.mia_unavailable_reason:
                state, reason = "rejected", "missing_token_evidence"
        except BudgetExhaustedError:
            state, reason = "exhausted", "request_budget"
        except RequestError as error:
            response, status_code = error.response_evidence, error.status_code
            if error.code == "caller_cancelled":
                state, reason = "cancelled", "caller_cancelled"
            elif error.code == "missing_token_evidence" and isinstance(response, ModelResponse):
                reason = "missing_token_evidence"
                report = report_response(response)
            elif rejected_token_control(error, config):
                capabilities = dict(report.capabilities)
                capabilities[error.parameter] = CapabilityEvidence(
                    "unsupported", "Explicit configured probe parameter rejection",
                    "Provider rejected this exact supplied token control.", observed=True)
                report = CapabilityReport(capabilities, features)
                reason = "parameter_rejected"
            elif status_code == 401:
                reason = "authentication_rejected"
            elif status_code == 403:
                reason = "access_rejected"
            elif status_code == 504 or error.code == "timeout":
                state, reason = "unknown_outcome", "gateway_timeout" if status_code == 504 else "timeout"
            elif status_code in {429, 500, 502, 503, 529}:
                state, reason = "temporarily_unavailable", "provider_unavailable"
        # Do not serialize arbitrary local exception text.
    if cancellation is not None and cancellation.cancelled:
        state, reason = "cancelled", "caller_cancelled"
    events = budget.events[used:]
    observation = ReadinessObservation(
        operation_id, stage, config.model, datetime.now(UTC).isoformat(), monotonic() - started, state,
        call_id=events[-1].call_id if events else None, status_code=status_code, reason=reason)
    if on_observation is not None:
        try:
            on_observation(observation)
        except Exception:
            observation = ReadinessObservation(operation_id, stage, config.model, datetime.now(UTC).isoformat(),
                monotonic() - started, "rejected", reason="callback_failed")
    return ProbeEvidence(config, report, response, (observation,))
