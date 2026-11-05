"""Explicit capability resolution before workflow construction or evaluation.

Factories accept ``(config, budget, *, features)`` and perform local construction
only. Preparation exclusively owns the initially unused budget during this
synchronous operation; all selected clients retain that same remaining budget.
Runtime rewrite policies and fitted scorers are explicit and never serialized.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from types import MappingProxyType
from uuid import uuid4

from recall_guard.core.completion import (
    CapabilityStatus,
    CompletionClient,
    CompletionProtocol,
    FeatureChoices,
    ModelResponse,
    validate_features,
)
from recall_guard.core.provider_anthropic import AnthropicClient, validate_anthropic_config
from recall_guard.core.provider_callback import CallbackClient
from recall_guard.core.provider_probe import TOKEN_CONTROLS
from recall_guard.core.providers import OpenAICompatibleClient
from recall_guard.core.readiness_evidence import ProbeEvidence
from recall_guard.core.request_control import CallBudget
from recall_guard.harness.intervention_types import PreparationArtifact, PreparationBinding, RequestRecord, text_hash
from recall_guard.harness.rewriting import EvidenceRewriter, RewritePolicy
from recall_guard.harness.workflow import RecallWorkflow
from recall_guard.harness.workflow_plan import (
    CompletionFallbackPolicy,
    WorkflowPlan,
    WorkflowRequest,
    canonical_json,
)

_PROBE_PROMPT = "Reply with the word OK."
_CLIENT_TYPES = (OpenAICompatibleClient, AnthropicClient, CallbackClient)


@dataclass(frozen=True)
class WorkflowPreparation:
    """Immutable historical evidence plus optional runtime-only prepared objects.

    ``to_dict`` exports only the existing whitelisted PreparationArtifact.
    Failure never exposes a runnable workflow. A ready result establishes
    observed capabilities at preparation time, without guaranteeing future ones.
    """

    artifact: PreparationArtifact
    binding: PreparationBinding
    clients: Mapping[str, CompletionClient] = field(repr=False)
    workflow: RecallWorkflow | None = field(default=None, repr=False)

    @property
    def plan(self):
        return self.artifact.plan

    @property
    def ready(self):
        return self.workflow is not None and self.artifact.stopping_reason is None

    def to_dict(self):
        return self.artifact.to_dict()


class _PreparationStop(Exception):
    pass


class FactoryContractError(ValueError):
    """A factory admitted provider work outside the declared probe operation.

    No truthful prompt hash is available for that work, so no preparation
    artifact is fabricated. The immutable selected plan and sanitized event
    snapshots remain inspectable; the original budget retains every admission.
    """

    def __init__(self, plan, budget):
        super().__init__("client_factory must construct clients without provider work")
        self.plan = plan
        self.events = budget.events
        self.used_requests = budget.used
        self.remaining_requests = budget.remaining


def _token_contradiction(request):
    parameters = request.roles["target"].parameters
    return request.features.mia_scoring and (
        "logprobs" in parameters and parameters["logprobs"] is not True
        or "top_logprobs" in parameters and (parameters["top_logprobs"] is None or parameters["top_logprobs"] == 0))


def _declared_missing(request):
    config = request.roles["target"]
    return (config.protocol == CompletionProtocol.ANTHROPIC_MESSAGES
            or any(name in config.capabilities and config.capabilities[name].status == CapabilityStatus.UNSUPPORTED
                   for name in TOKEN_CONTROLS))


def _probe_trigger(probe):
    """No diagnostic text matching or promotion of partial/empty service data."""
    terminal = probe.observations[-1]
    capabilities = probe.report.capabilities
    if terminal.reason == "parameter_rejected" and any(
        name in capabilities and capabilities[name].observed
        and capabilities[name].status == CapabilityStatus.UNSUPPORTED for name in TOKEN_CONTROLS
    ):
        return "logprob_controls_rejected"
    response = probe.response
    if (isinstance(response, ModelResponse) and response.content.strip()
            and canonical_json(response.to_dict()["requested_settings"]) == canonical_json(probe.config.to_dict()["parameters"])
            and terminal.reason == "missing_token_evidence"
            and response.mia_unavailable_reason is not None
            and all(item.status == CapabilityStatus.SUPPORTED and item.observed
                    for name, item in capabilities.items() if name not in TOKEN_CONTROLS)):
        return "missing_token_evidence"
    return None


def _selected_request(requested, fallback, fallback_ensemble):
    target = requested.roles["target"]
    omitted = tuple(sorted(target.parameters.keys() & TOKEN_CONTROLS)) if fallback.omit_logprob_controls else ()
    target = replace(target, parameters={name: value for name, value in target.parameters.items() if name not in omitted})
    ensemble = fallback_ensemble
    if ensemble is None and requested.ensemble is not None and (
        fallback.features.ensemble_draws != 1 or requested.ensemble.draws == 1
    ):
        ensemble = requested.ensemble
    return WorkflowRequest(dict(requested.roles) | {"target": target}, fallback.features,
                           requested.rewrite_policy_id, ensemble), omitted


def _validate_local(request):
    """Provider-local validation without credential resolution or construction."""
    for role in ("target", "rewriter", "verifier") if request.features.rewriting else ("target",):
        config = request.roles[role]
        features = request.features if role == "target" else FeatureChoices()
        validate_features(config, features)
        if config.protocol == CompletionProtocol.ANTHROPIC_MESSAGES:
            validate_anthropic_config(config, features)


def _validate_rewrite_policy(request, policy):
    if not request.features.rewriting:
        return None
    if not isinstance(policy, RewritePolicy):
        raise _PreparationStop("rewrite_policy_required")
    policy.validate_configuration()
    if policy.intervention_id != request.rewrite_policy_id:
        raise _PreparationStop("rewrite_policy_mismatch")
    for role, client in (("rewriter", policy.rewriter.client), ("verifier", policy.verifier)):
        if canonical_json(client.config.to_dict()) != canonical_json(request.roles[role].to_dict()):
            raise _PreparationStop("rewrite_policy_mismatch")


def _bind_rewrite_policy(request, policy, clients):
    _validate_rewrite_policy(request, policy)
    if request.features.rewriting:
        return RewritePolicy(EvidenceRewriter(clients["rewriter"]), clients["verifier"])
    return None


def prepare_workflow(requested, fallback_policy, client_factory, budget, cancellation=None, on_observation=None,
                     *, rewrite_policy=None, scorer=None, fallback_ensemble=None):
    """Resolve and probe an explicit plan, retaining failures without evaluation.

    ``fallback_ensemble`` is required when the caller selects a different
    multi-draw ensemble; no sampling/rewriting settings are invented. A runtime
    RewritePolicy binds the persisted identity and is rebound to fresh clients.
    A fitted scorer is required only if MIA remains selected after resolution.
    """
    if not isinstance(requested, WorkflowRequest):
        raise TypeError("requested must be WorkflowRequest")
    if fallback_policy is not None and not isinstance(fallback_policy, CompletionFallbackPolicy):
        raise TypeError("fallback_policy must be CompletionFallbackPolicy or None")
    if not isinstance(budget, CallBudget) or budget.used:
        raise ValueError("Preparation requires an exclusively owned, unused shared CallBudget")
    if not callable(client_factory):
        raise TypeError("client_factory must be callable")
    execution_id = "preparation-" + str(uuid4())
    plan = WorkflowPlan(requested, requested, datetime.now(UTC).isoformat(), fallback_policy)
    evidence, clients, role_by_call, constructed_clients = {}, {}, {}, []
    stopping_reason, workflow = None, None

    def check_stop():
        if cancellation is not None and cancellation.cancelled:
            raise _PreparationStop("caller_cancelled")
        if budget.remaining == 0:
            raise _PreparationStop("request_budget")

    def construct(role, request):
        check_stop()
        before = budget.used
        failure = None
        try:
            client = client_factory(request.roles[role], budget,
                                    features=request.features if role == "target" else FeatureChoices())
        except (TypeError, ValueError):
            failure = "configuration_error"
        except Exception:
            failure = "factory_construction_failed"
        role_by_call.update({event.call_id: role for event in budget.events[before:]})
        # Outside the factory handler: contract errors retain no original
        # exception/context, arbitrary diagnostic text or unknowable prompt hash.
        if budget.used != before:
            raise FactoryContractError(plan, budget)
        if failure is not None:
            raise _PreparationStop(failure)
        if (not isinstance(client, _CLIENT_TYPES) or client.budget is not budget
                or canonical_json(client.config.to_dict()) != canonical_json(request.roles[role].to_dict())
                or client.features != (request.features if role == "target" else FeatureChoices())):
            raise _PreparationStop("factory_contract_mismatch")
        if isinstance(client, CallbackClient) and client.request_accounting.status != CapabilityStatus.SUPPORTED:
            raise _PreparationStop("unverified_request_accounting")
        if any(client is previous for previous in constructed_clients):
            raise _PreparationStop("factory_reused_client")
        constructed_clients.append(client)
        return client

    def probe(role, client):
        check_stop()
        before = budget.used
        config = client.config
        try:
            result = client.probe_with_evidence(_PROBE_PROMPT, stage=f"{execution_id}/probe/{role}",
                                               cancellation=cancellation, on_observation=on_observation)
        finally:
            role_by_call.update({event.call_id: role for event in budget.events[before:]})
        if not isinstance(result, ProbeEvidence):
            raise _PreparationStop("probe_contract_mismatch")
        if canonical_json(result.config.to_dict()) != canonical_json(config.to_dict()):
            result = replace(result, config=config, observations=tuple(
                replace(item, state="rejected", reason="malformed_response") if item.state == "ready" else item
                for item in result.observations))
        evidence.setdefault(role, []).append(result)
        if budget.used == before and result.ready:
            raise _PreparationStop("unverified_request_accounting")
        if canonical_json(client.config.to_dict()) != canonical_json(config.to_dict()):
            raise _PreparationStop("configuration_changed")
        if isinstance(result.response, ModelResponse):
            expected = config.to_dict()["parameters"]
            if (canonical_json(result.response.to_dict()["requested_settings"]) != canonical_json(expected)
                    or any(name in expected and canonical_json(value) != canonical_json(expected[name])
                           for name, value in result.response.to_dict()["confirmed_settings"].items())):
                raise _PreparationStop("response_settings_mismatch")
        return result

    def select_fallback(reason):
        nonlocal plan
        if fallback_policy is None:
            raise _PreparationStop(reason)
        selected, omitted = _selected_request(requested, fallback_policy, fallback_ensemble)
        plan = WorkflowPlan(requested, selected, datetime.now(UTC).isoformat(), fallback_policy, omitted, reason)
        _validate_local(selected)
        _validate_rewrite_policy(selected, rewrite_policy)

    try:
        if _token_contradiction(requested):
            raise _PreparationStop("configuration_error")
        check_stop()
        # Unrelated declarations always reject before a token fallback is chosen.
        unsupported = requested.roles["target"].inspect(requested.features).unsupported
        if set(unsupported) - TOKEN_CONTROLS:
            raise _PreparationStop("configuration_error")
        target = requested.roles["target"]
        if target.protocol == CompletionProtocol.ANTHROPIC_MESSAGES:
            # Validate unrelated native controls before interpreting its declared
            # lack of tokens. This detached validation does not change the plan.
            validate_anthropic_config(replace(target, parameters={name: value for name, value in target.parameters.items()
                                                                 if name not in TOKEN_CONTROLS}), FeatureChoices())
        if requested.features.mia_scoring and _declared_missing(requested):
            select_fallback("unsupported_token_evidence")
        _validate_local(plan.selected)
        _validate_rewrite_policy(plan.selected, rewrite_policy)
        clients["target"] = construct("target", plan.selected)
        first = probe("target", clients["target"])
        trigger = _probe_trigger(first) if requested.features.mia_scoring and plan.reason is None else None
        if trigger is not None:
            select_fallback(trigger)
            # Fresh enforcement and fresh observations even if parameters match.
            clients["target"] = construct("target", plan.selected)
            first = probe("target", clients["target"])
        if not first.ready:
            raise _PreparationStop((first.observations[-1].reason or first.observations[-1].state).value)
        if plan.selected.features.rewriting:
            if rewrite_policy is None:
                raise _PreparationStop("rewrite_policy_required")
            for role in ("rewriter", "verifier"):
                clients[role] = construct(role, plan.selected)
        bound_policy = _bind_rewrite_policy(plan.selected, rewrite_policy, clients)
        selected_scorer = scorer if plan.selected.features.mia_scoring else None
        if plan.selected.features.mia_scoring and selected_scorer is None:
            raise _PreparationStop("matching_calibration_required")
        candidate = RecallWorkflow(clients["target"], rewrite_policy=bound_policy, scorer=selected_scorer,
                                   features=plan.selected.features, budget=budget, strict_budget=True)
        candidate.validate(rewrite=plan.selected.features.rewriting, ensemble=plan.selected.ensemble)
        for role in ("rewriter", "verifier") if plan.selected.features.rewriting else ():
            observed = probe(role, clients[role])
            if not observed.ready:
                raise _PreparationStop((observed.observations[-1].reason or observed.observations[-1].state).value)
        if cancellation is not None and cancellation.cancelled:
            raise _PreparationStop("caller_cancelled")
        candidate.bind_plan(plan, cancellation=cancellation, on_observation=on_observation)
        workflow = candidate
    except _PreparationStop as stop:
        stopping_reason = str(stop)
    except FactoryContractError:
        raise
    except (TypeError, ValueError):
        stopping_reason = "configuration_error"
    except Exception:
        stopping_reason = "preparation_failed"
    records = tuple(RequestRecord(execution_id, event, role_by_call[event.call_id], text_hash(_PROBE_PROMPT),
                                  phase="preparation") for event in budget.events)
    artifact = PreparationArtifact(execution_id, plan, records, evidence, stopping_reason)
    return WorkflowPreparation(artifact, artifact.bind(total_request_cap=budget.limit),
                               MappingProxyType(dict(clients)), workflow)
