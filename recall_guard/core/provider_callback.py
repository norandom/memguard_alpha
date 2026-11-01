"""Synchronous caller completions with explicit physical-accounting limits.

Opaque callbacks remain useful for standalone generation, but their network
requests, retries and costs are unknown. Only caller-cooperative callbacks that
route every physical request through the supplied hook support strict budgets.
"""

import inspect
from collections.abc import Callable, Sequence
from dataclasses import replace

from recall_guard.core.completion import (
    CapabilityEvidence,
    CapabilityReport,
    CapabilityStatus,
    CompletionProtocol,
    FeatureChoices,
    ModelConfig,
    ModelResponse,
    ParameterValue,
    validate_features,
)
from recall_guard.core.request_control import BudgetExhaustedError, CallBudget, RequestControl, RequestError


def _result(value: object, parameters: dict) -> ModelResponse:
    """Normalize actual callback output without manufacturing provider evidence."""
    if inspect.isawaitable(value):
        if inspect.iscoroutine(value):
            value.close()
        raise RequestError(
            "Callback results must be synchronous; awaitables are incompatible with this runtime contract. "
            "Supply a synchronous callback rather than starting another notebook event loop.",
            code="callback_runtime_error",
        )
    if isinstance(value, str):
        return ModelResponse(value, requested_settings=parameters)
    if isinstance(value, ModelResponse):
        return replace(value, requested_settings=parameters)
    raise RequestError("Callback must return str or ModelResponse", code="invalid_response")


def _callback_error(error: Exception) -> RequestError:
    """Classify runtime errors separately from explicit model-parameter failures."""
    status = getattr(error, "status_code", None)
    if status is None:
        status = getattr(getattr(error, "response", None), "status_code", None)
    parameter = getattr(error, "parameter", None)
    code = getattr(error, "code", None)
    if not code:
        code = ("callback_auth_error" if status in (401, 403) else
                "callback_parameter_error" if parameter else "callback_runtime_error")
    try:
        usage = getattr(error, "usage", None)
        return RequestError(str(error), status_code=status, parameter=parameter, code=code,
                            detail=getattr(error, "detail", None), usage=usage,
                            response_evidence=error.response_evidence if isinstance(error, RequestError) else None,
                            request_id=error.request_id if isinstance(error, RequestError) else None,
                            retry_after=error.retry_after if isinstance(error, RequestError) else None,
                            retryable=isinstance(error, (TimeoutError, ConnectionError))
                            or isinstance(error, RequestError) and error.retryable)
    except ValueError:
        # Arbitrary exception metadata cannot establish token usage.
        return RequestError(str(error), status_code=status, parameter=parameter, code=code,
                            detail=getattr(error, "detail", None))


class CallbackClient:
    """Adapt ``callback(prompt, **generation_parameters)`` without an SDK.

    Set ``physical_request_hook`` to the callback keyword receiving a callable
    ``request(operation) -> ModelResponse``. Each synchronous zero-argument
    operation must perform exactly one physical request, with hidden transport
    retries/redirects disabled. The hook normalizes str/ModelResponse output and
    RequestControl owns admission, pacing and bounded retries of that operation.
    A callback can use the hook multiple times; the outer callback is never
    counted or retried as if it were itself a physical request. The hook is only
    valid while that callback is active. This is a caller contract, not a sandbox
    capable of detecting requests made outside the hook.

    ``strict_budget=True`` rejects opaque callbacks before calling them and
    rejects results returned without any hooked physical operation. Without a
    hook, generation remains available but physical counts/retries/pacing are
    unverified and no physical ledger entry is invented for a logical call.
    Inspect ``request_accounting`` separately from model capability probes.

    The caller owns credential handling. No credential reference is resolved or
    forwarded. Register all known credential strings with ``secrets`` so that
    callback diagnostics and request events can be redacted; the adapter cannot
    discover credentials hidden inside a caller's function.
    """

    def __init__(
        self, config: ModelConfig, budget: CallBudget, callback: Callable[..., str | ModelResponse], *,
        features: FeatureChoices | None = None, physical_request_hook: str | None = None,
        strict_budget: bool = False, secrets: Sequence[str] = (),
    ) -> None:
        if not isinstance(config, ModelConfig):
            raise TypeError("config must be ModelConfig")
        if config.protocol != CompletionProtocol.CALLBACK:
            raise ValueError("CallbackClient requires the explicit callback protocol")
        if config.readiness is not None:
            raise ValueError("Callback observation profile is unsupported")
        if not callable(callback):
            raise TypeError("callback must be callable")
        if not isinstance(strict_budget, bool):
            raise TypeError("strict_budget must be boolean")
        if physical_request_hook is not None and (
            not isinstance(physical_request_hook, str) or not physical_request_hook.isidentifier()
        ):
            raise ValueError("physical_request_hook must name a callback keyword parameter")
        if strict_budget and physical_request_hook is None:
            raise ValueError("A strict physical budget requires an explicit every-physical-request callback hook")
        self.config = config
        self.features = FeatureChoices() if features is None else features
        validate_features(config, self.features)
        self._callback = callback
        self._hook_parameter = physical_request_hook
        self._strict_budget = strict_budget
        self._validate_hook_collision(config)
        self._control = RequestControl(config, budget, secrets=secrets)

    @property
    def budget(self) -> CallBudget:
        """Associated budget; opaque callback accounting still remains unknown."""
        return self._control.budget

    @property
    def request_accounting(self) -> CapabilityEvidence:
        """Separate cooperative accounting support from observed model features."""
        if self._hook_parameter is None:
            return CapabilityEvidence(
                CapabilityStatus.UNVERIFIED, "Opaque caller callback",
                "Physical request count, internal retries, pacing and cost are unknown; "
                "strict-budget live experiments must reject this callback.",
            )
        return CapabilityEvidence(
            CapabilityStatus.SUPPORTED, "Caller-declared every-physical-request hook contract",
            "Each physical operation must use the hook with no hidden retries; "
            "accounting relies on caller cooperation and is not independently verified.",
        )

    def _validate_hook_collision(self, config: ModelConfig) -> None:
        if self._hook_parameter is not None and self._hook_parameter in config.parameters:
            raise ValueError(f"Generation parameter {self._hook_parameter!r} conflicts with the physical request hook")

    def _check_signature(self, prompt: str, parameters: dict) -> None:
        try:
            signature = inspect.signature(self._callback)
        except (TypeError, ValueError):
            # Some extension callables do not expose a Python signature.
            return
        try:
            signature.bind(prompt, **parameters)
        except TypeError as error:
            variadic = any(value.kind == inspect.Parameter.VAR_KEYWORD for value in signature.parameters.values())
            unsupported = None if variadic else next(
                (name for name in parameters if name not in signature.parameters), None,
            )
            raise RequestError(str(error), code="callback_parameter_error", parameter=unsupported) from None

    def generate(self, prompt: str, **parameters: ParameterValue) -> ModelResponse:
        """Generate without reserving names from the callback's generation controls."""
        return self._generate(prompt, parameters, stage="target", enforce_mia=self.features.mia_scoring)

    def generate_for_stage(self, prompt: str, *, stage: str, **parameters: ParameterValue) -> ModelResponse:
        """Associate every hooked attempt with an explicit workflow stage."""
        return self._generate(prompt, parameters, stage=stage, enforce_mia=self.features.mia_scoring)

    def _generate(self, prompt: str, overrides: dict, *, stage: str, enforce_mia: bool) -> ModelResponse:
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        if not isinstance(stage, str) or not stage.strip():
            raise ValueError("stage must be a nonempty string")
        settings = replace(self.config, parameters=dict(self.config.parameters) | overrides)
        validate_features(settings, self.features)
        self._validate_hook_collision(settings)
        # Two detached snapshots keep callback mutation out of provenance.
        requested = settings.to_dict()["parameters"]
        forwarded = settings.to_dict()["parameters"]
        active, attempts, exhausted = True, 0, False

        def request(operation: Callable[[], str | ModelResponse]) -> ModelResponse:
            nonlocal attempts, exhausted
            if not active:
                raise RuntimeError("Physical request hook is no longer active outside its synchronous callback")
            if not callable(operation):
                raise TypeError("physical request operation must be callable")

            def physical_operation() -> ModelResponse:
                nonlocal attempts
                attempts += 1
                try:
                    return _result(operation(), requested)
                except Exception as error:
                    raise _callback_error(error) from None

            try:
                return self._control.run(physical_operation, stage=stage)
            except BudgetExhaustedError:
                exhausted = True
                raise

        if self._hook_parameter is not None:
            forwarded[self._hook_parameter] = request
        failure, budget_exhausted = None, False
        try:
            self._check_signature(prompt, forwarded)
            response = _result(self._callback(prompt, **forwarded), requested)
            if exhausted:
                raise BudgetExhaustedError("Physical request budget exhausted")
            if self._strict_budget and not attempts:
                raise RequestError("Callback returned without a hooked physical operation; accounting is unverified",
                                   code="unverified_request_accounting")
            if enforce_mia and (reason := response.mia_unavailable_reason):
                raise RequestError(reason, code="missing_token_evidence", usage=response.usage, response_evidence=response)
        except BudgetExhaustedError:
            budget_exhausted = True
        except Exception as error:
            # Reuse central redaction without inventing an opaque physical event.
            failure = self._control._failure(_callback_error(error))
        finally:
            active = False
        if budget_exhausted:
            # A callback may encounter admission while handling a secret-bearing
            # error; preserve the budget type without its original error chain.
            raise BudgetExhaustedError("Physical request budget exhausted")
        if failure is not None:
            # Outside the handler: no original exception in cause or context.
            raise failure
        return response

    def probe(self, prompt: str = "Reply with the word OK.", *, stage: str = "probe") -> CapabilityReport:
        """Observe actual text/token evidence; accepted settings are not echoed ones.

        A ready model report does not establish strict request accounting for an
        opaque callback. Its separate ``request_accounting`` stays unverified.
        """
        response = self._generate(prompt, {}, stage=stage, enforce_mia=False)
        capabilities = {}
        for name in self.config.inspect(self.features).capabilities:
            status, observed = CapabilityStatus.SUPPORTED, True
            detail = "Actual callback text observed."
            if name == "text_generation" and not response.content.strip():
                status, observed = CapabilityStatus.UNVERIFIED, False
                detail = "No non-whitespace text observed; inspect callback output and repeat the probe before batching."
            elif name == "logprobs" and (self.features.mia_scoring or self.config.parameters.get(name) is True):
                reason = response.logprobs_unavailable_reason
                status = CapabilityStatus.SUPPORTED if reason is None else CapabilityStatus.UNSUPPORTED
                detail = reason or "Genuine token logprobs supplied by callback."
            elif name == "top_logprobs" and (self.features.mia_scoring or self.config.parameters.get(name)):
                status = CapabilityStatus.SUPPORTED if response.mia_unavailable_reason is None else CapabilityStatus.UNSUPPORTED
                detail = response.mia_unavailable_reason or "Genuine alternatives supplied for every token."
            elif name != "text_generation":
                detail = ("Parameter accepted; effective value echoed by callback." if name in response.confirmed_settings
                          else "Parameter accepted; effective value unconfirmed (not echoed).")
            capabilities[name] = CapabilityEvidence(status, "Actual synchronous callback probe", detail, observed=observed)
        return CapabilityReport(capabilities, self.features)
