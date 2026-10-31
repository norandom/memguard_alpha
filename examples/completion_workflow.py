"""Offline completion mechanics for applications and Jupyter.

Run from a development checkout: uv run --no-sync python examples/completion_workflow.py
These fixed text callbacks test wiring, never provider support or mitigation.
"""

import json
from collections.abc import Callable
from dataclasses import replace

from recall_guard.core.completion import (
    CapabilityEvidence,
    FeatureChoices,
    ModelConfig,
    ModelResponse,
)
from recall_guard.core.ensemble import EnsembleSpec
from recall_guard.core.provider_anthropic import validate_anthropic_config
from recall_guard.core.provider_callback import CallbackClient
from recall_guard.core.request_control import CallBudget, RequestError
from recall_guard.harness.rewriting import EvidenceRewriter, RewriteInput, RewritePolicy
from recall_guard.harness.workflow import RecallWorkflow, WorkflowResult

SOURCE_EVIDENCE = "Acme rose 2% on 2024-01-02; estimates remain uncertain."
ACCEPTED_EVIDENCE = "Let's review: " + SOURCE_EVIDENCE
REJECTED_EVIDENCE = ACCEPTED_EVIDENCE.replace("2%", "3%")
FIXTURE_VERDICT = json.dumps({
    "equivalent": True,
    "checks": dict.fromkeys((
        "claims", "entities", "dates", "quantities", "units", "uncertainty",
        "qualifications", "attribution", "instruction_injection",
    ), True),
    "discrepancies": [],
})


def fixture_config() -> ModelConfig:
    """Declare a text-only fixture; probe acceptance is still required."""
    return ModelConfig(
        protocol="callback", endpoint=None, model="offline-target-fixture",
        parameters={"max_tokens": 128}, max_retries=0,
        capabilities={
            "text_generation": CapabilityEvidence("supported", "Fixed offline fixture"),
            "max_tokens": CapabilityEvidence("supported", "Fixture accepts this argument"),
            "logprobs": CapabilityEvidence("unsupported", "Fixture returns only text"),
            "top_logprobs": CapabilityEvidence("unsupported", "Fixture returns only text"),
        },
    )


def fixture_item() -> RewriteInput:
    return RewriteInput("mechanics-1", "Return only up or down.", SOURCE_EVIDENCE, "2024-01-02")


def fixture_client(config: ModelConfig, budget: CallBudget, response_text: str) -> CallbackClient:
    """Model one admitted operation per callback; no HTTP request occurs.

    A real callback must put each physical request through request(operation),
    with hidden retries and redirects disabled. This relies on caller cooperation.
    """
    def complete(
        prompt: str, *, max_tokens: int,
        request: Callable[[Callable[[], str]], ModelResponse],
    ) -> ModelResponse:
        # Fixed text deliberately ignores prompt/max_tokens. It cannot test a model.
        # The admission hook normalizes the operation's str to ModelResponse.
        return request(lambda: response_text)

    return CallbackClient(config, budget, complete, physical_request_hook="request", strict_budget=True)


def make_fixture_workflow(
    config: ModelConfig, features: FeatureChoices, budget: CallBudget, *,
    rewritten_evidence: str = ACCEPTED_EVIDENCE,
) -> RecallWorkflow:
    """Share one budget across target and explicitly selected auxiliary roles."""
    target = fixture_client(config, budget, "up")
    policy = None
    if features.rewriting:
        rewriter = fixture_client(replace(config, model="offline-rewrite-fixture"), budget, rewritten_evidence)
        verifier = fixture_client(replace(config, model="offline-verifier-fixture"), budget, FIXTURE_VERDICT)
        policy = RewritePolicy(EvidenceRewriter(rewriter), verifier)
    return RecallWorkflow(target, features=features, rewrite_policy=policy, budget=budget, strict_budget=True)


def result_summary(result: WorkflowResult) -> dict[str, object]:
    """Select readable fields; do not serialize clients or fitted scorer objects."""
    return {
        "status": result.status.value, "method": result.method, "stage": result.stage,
        "answer": result.answer, "n_requested": result.n_requested,
        "n_attempted": result.n_attempted, "n_usable": result.n_usable,
        "agreement": None if result.ensemble is None else result.ensemble.agreement,
        "accepted_evidence": result.accepted_evidence, "rewrite_scope": result.rewrite_scope,
        "p_memorized": result.p_memorized,
        "mcs_unavailable_reason": result.unavailable_reasons.get("p_memorized"),
        "token_evidence_unavailable_reason": (
            None if result.response is None else result.response.mia_unavailable_reason
        ),
        "failure_kind": result.failure_kind, "error": result.error,
    }


def show_local_native_errors() -> None:
    """Validate native choices without constructing a client or looking up a key."""
    native = ModelConfig(
        protocol="anthropic_messages", endpoint="https://api.anthropic.com/v1/messages",
        model="claude-opus-5-5", credential_ref="ANTHROPIC_API_KEY",
        parameters={"max_tokens": 2048}, max_retries=0,
    )
    for label, config, features in (
        ("native_parameter_error", replace(native, parameters={"max_tokens": 2048, "temperature": 0.2}),
         FeatureChoices()),
        ("native_scoring_error", native, FeatureChoices(mia_scoring=True)),
    ):
        try:
            validate_anthropic_config(config, features)
        except ValueError as error:
            print(json.dumps({"example": label, "error": str(error)}))


def show_callback_runtime_error() -> None:
    """An async callback violates the synchronous contract, independently of model controls."""
    async def async_callback(prompt: str, **parameters) -> str:
        return "up"

    # Opaque callbacks can generate, but cannot claim a strict physical budget.
    client = CallbackClient(fixture_config(), CallBudget(1), async_callback)
    try:
        client.generate("runtime-contract demonstration")
    except RequestError as error:
        print(json.dumps({"example": "callback_runtime_error", "code": error.code,
                          "parameter": error.parameter, "error": str(error)}))


def main() -> None:
    config = fixture_config()
    features = FeatureChoices(mia_scoring=False, ensemble_draws=3, rewriting=True)
    ensemble = EnsembleSpec(draws=features.ensemble_draws, min_parsed=3, max_workers=1, retain_draws=True)
    budget = CallBudget(12)
    workflow = make_fixture_workflow(config, features, budget)
    local = config.inspect(features)  # No calls; declarations may still need a probe.
    print(json.dumps({"example": "configuration", "model": config.model, "protocol": config.protocol.value,
                      "endpoint": config.endpoint, "parameters": dict(config.parameters),
                      "draws": features.ensemble_draws, "rewriting": features.rewriting,
                      "requires_probe": local.requires_probe, "unsupported": local.unsupported}))
    preflight = workflow.preflight(rewrite=features.rewriting, ensemble=ensemble)
    print(json.dumps({"example": "preflight", "ready_for_batch": preflight.ready_for_batch,
                      "affected_role": preflight.affected_role, "error": preflight.error}))
    if not preflight.ready_for_batch:
        return
    accepted = workflow.run(fixture_item(), rewrite=features.rewriting, ensemble=ensemble)
    print(json.dumps({"example": "accepted_fixture", **result_summary(accepted)}))

    if features.rewriting:
        rejected_workflow = make_fixture_workflow(config, features, budget, rewritten_evidence=REJECTED_EVIDENCE)
        rejected = rejected_workflow.run(fixture_item(), rewrite=features.rewriting, ensemble=ensemble)
        print(json.dumps({"example": "rejected_fixture", **result_summary(rejected)}))
    print(json.dumps({"example": "fixture_budget", "admitted_operations": budget.used,
                      "limit": budget.limit, "actual_http_requests": 0}))
    show_local_native_errors()
    show_callback_runtime_error()


if __name__ == "__main__":
    main()
