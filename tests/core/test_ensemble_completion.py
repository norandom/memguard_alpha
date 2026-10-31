"""One ensemble reducer for legacy and optional-evidence completion responses."""

from __future__ import annotations

from dataclasses import dataclass, replace
from threading import Lock

import pytest

from recall_guard.core.completion import ModelConfig, ModelResponse, TokenAlternative, TokenEvidence
from recall_guard.core.ensemble import EnsembleSpec, generate_ensemble, reduce_draws
from recall_guard.core.nvidia_lm import CompletionResult, TokenLogprob
from recall_guard.core.request_control import BudgetExhaustedError, CallBudget, RequestControl, RequestError


class _ScriptedClient:
    """Record logical draws without depending on provider configuration."""

    def __init__(self, outcomes):
        self.outcomes = outcomes
        self.calls: list[tuple[str, dict]] = []
        self._lock = Lock()

    def generate(self, prompt: str, **parameters):
        with self._lock:
            outcome = self.outcomes[len(self.calls)]
            self.calls.append((prompt, parameters))
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class _BudgetedClient:
    """Let real request admission account for scripted physical attempts."""

    def __init__(self, outcomes, budget: CallBudget):
        self.physical = _ScriptedClient(outcomes)
        config = ModelConfig(protocol="callback", endpoint=None, model="test-model", max_retries=1, retry_backoff_s=0)
        self.control = RequestControl(config, budget)

    def generate(self, prompt: str, **parameters):
        return self.control.run(lambda: self.physical.generate(prompt, **parameters), stage="target")


def _spec(**settings) -> EnsembleSpec:
    return EnsembleSpec(**{"draws": 4, "max_workers": 2, "min_parsed": 1, **settings})


def _legacy(content: str) -> CompletionResult:
    return CompletionResult(content, [TokenLogprob("x", -0.1, [{"token": "x", "logprob": -0.1}])], None)


def test_text_only_statistics_match_the_exact_legacy_reduction() -> None:
    contents = ["up:0.2", "down:0.1", "up:0.4", "up:0.3"]
    text_draws = [ModelResponse(content, request_id=f"draw-{i}") for i, content in enumerate(contents)]
    legacy_draws = [_legacy(content) for content in contents]
    spec = _spec(retain_draws=True, grid=0.1)
    settings = {
        "decide": lambda response: response.content.split(":")[0],
        "components": lambda response: {"answer_value": float(response.content.split(":")[1])},
        "waves": [0, 0, 1, 1],
        "n_requested": 6,
        "fail_counts": {"http": 1, "projection": 1},
    }

    legacy = reduce_draws(legacy_draws, spec, **settings)
    text_only = reduce_draws(text_draws, spec, **settings)

    # Every field except the intentionally different original response objects
    # must remain exactly equal, including floats, canonical hash and verdicts.
    assert replace(text_only, consensus=legacy.consensus, draws=legacy.draws) == legacy
    assert type(legacy.consensus) is CompletionResult
    assert legacy.consensus is legacy_draws[0]
    assert type(text_only.consensus) is ModelResponse
    assert text_only.consensus is text_draws[0]
    assert text_only.agreement == 0.75
    assert text_only.location == {"answer_value": 0.2}
    assert {id(draw) for draw in text_only.draws} == {id(draw) for draw in text_draws}
    assert all(draw.token_evidence is None for draw in text_only.draws)
    assert text_only.consensus.mia_unavailable_reason is not None


def test_retained_completion_draws_replay_without_a_provider_or_clock() -> None:
    draws = [ModelResponse(content) for content in ("up:b", "down:a", "up:a", "down:b")]
    spec = _spec(retain_draws=True)
    decide = lambda response: response.content.split(":")[0]  # noqa: E731
    baseline = reduce_draws(draws, spec, decide=decide)

    for replay_order in (reversed(draws), [draws[i] for i in (2, 0, 3, 1)]):
        replay = reduce_draws(list(replay_order), spec, decide=decide)
        assert replay == baseline
        assert replay.consensus is draws[1]  # decision ties choose 'down', then content
        assert replay.sampled_at is None


@dataclass(frozen=True)
class _ApplicationAnswer:
    value: str

    @property
    def content(self) -> str:
        return self.value


def test_reduction_only_requires_read_only_content() -> None:
    answer = _ApplicationAnswer("up")
    result = reduce_draws([answer], _spec(draws=1, retain_draws=True), decide=lambda response: response.content)

    assert result.consensus is answer
    assert result.draws == (answer,)
    assert result.location == {}
    assert result.component_verdicts == ()
    assert result.agreement == 1.0


def test_executor_preserves_response_objects_and_missing_evidence() -> None:
    draws = [
        ModelResponse(content, usage={"output_tokens": i + 1}, request_id=f"draw-{i}")
        for i, content in enumerate(("up:b", "up:a", "down:a", "up:c"))
    ]
    client = _ScriptedClient(draws)
    result = generate_ensemble(
        client, "same prompt", _spec(retain_draws=True), decide=lambda response: response.content.split(":")[0]
    )

    assert result.consensus is draws[1]
    assert {id(draw) for draw in result.draws} == {id(draw) for draw in draws}
    assert result.consensus.usage == {"output_tokens": 2}
    assert result.consensus.token_evidence is None
    assert result.consensus.mia_unavailable_reason is not None
    assert result.sampled_at is not None
    assert result.n_requested == result.n_attempted == result.n_parsed == 4
    assert client.calls == [("same prompt", {})] * 4


def test_optional_genuine_token_evidence_remains_attached_to_its_draw() -> None:
    evidence = (TokenEvidence("up", -0.1, (TokenAlternative("up", -0.1),)),)
    response = ModelResponse("up", token_evidence=evidence)
    result = generate_ensemble(_ScriptedClient([response]), "p", _spec(draws=1), decide=lambda draw: draw.content)

    assert result.consensus is response
    assert result.consensus.token_evidence is evidence
    assert result.consensus.mia_unavailable_reason is None


@pytest.mark.parametrize(
    "overrides", [{}, {"temperature": 0.0}, {"max_tokens": 256}, {"temperature": 0.3, "max_tokens": 512}]
)
def test_only_explicit_spec_overrides_reach_the_completion_client(overrides) -> None:
    client = _ScriptedClient([ModelResponse("up")] * 4)
    generate_ensemble(client, "p", _spec(**overrides), decide=lambda response: response.content)

    assert client.calls == [("p", overrides)] * 4


@pytest.mark.parametrize("status_code", [401, 403])
def test_configured_provider_auth_rejection_aborts_remaining_waves(status_code) -> None:
    failure = RequestError("credential rejected", status_code=status_code)
    client = _ScriptedClient([failure, ModelResponse("up")])

    with pytest.raises(RequestError) as caught:
        generate_ensemble(client, "p", _spec(draws=20), decide=lambda response: response.content)

    assert caught.value is failure
    assert len(client.calls) == 2


def test_completion_failures_keep_the_existing_failure_categories_and_denominators() -> None:
    client = _ScriptedClient(
        [
            ModelResponse("up"),
            RequestError("service unavailable", status_code=503),
            ModelResponse("unparseable"),
            TimeoutError("timed out"),
        ]
    )

    def decide(response):
        if response.content == "unparseable":
            raise ValueError("not an answer")
        return response.content

    result = generate_ensemble(client, "p", _spec(max_transport_failure_ratio=1.0), decide=decide)

    assert result.n_requested == 4
    assert result.n_attempted == 4
    assert result.n_parsed == 1
    assert dict(result.fail_counts) == {"http": 1, "projection": 1, "timeout": 1}
    assert result.agreement == 1.0


@pytest.mark.parametrize(
    "spec_settings, message",
    [
        ({"min_parsed": 3, "max_transport_failure_ratio": 1.0}, "usable draws"),
        ({"max_transport_failure_ratio": 0.25}, "transport failures"),
    ],
)
def test_completion_failures_still_enforce_usable_count_and_failure_ratio(spec_settings, message) -> None:
    client = _ScriptedClient([ModelResponse("up"), RequestError("failed", status_code=500)] * 2)

    with pytest.raises(RuntimeError, match=message):
        generate_ensemble(client, "p", _spec(**spec_settings), decide=lambda response: response.content)


def test_legacy_executor_still_returns_the_exact_legacy_completion_objects() -> None:
    draws = [_legacy(content) for content in ("up:b", "down:a", "up:a", "up:c")]
    result = generate_ensemble(
        _ScriptedClient(draws), "p", _spec(retain_draws=True), decide=lambda response: response.content.split(":")[0]
    )

    assert type(result.consensus) is CompletionResult
    assert result.consensus is draws[2]
    assert {id(draw) for draw in result.draws} == {id(draw) for draw in draws}


def test_attempted_draw_count_distinguishes_partial_replay_from_requested_draws() -> None:
    response = ModelResponse("up")
    result = reduce_draws(
        [response],
        _spec(draws=8),
        decide=lambda draw: draw.content,
        n_requested=8,
        fail_counts={"http": 1, "projection": 1},
    )

    assert result.n_requested == 8
    assert result.n_attempted == 3
    assert result.n_parsed == 1


def test_attempted_draw_count_excludes_physical_retries() -> None:
    response = ModelResponse("up")
    budget = CallBudget(5)
    client = _BudgetedClient([TimeoutError("retry once"), response, response, response, response], budget)

    result = generate_ensemble(client, "p", _spec(max_workers=1), decide=lambda draw: draw.content)

    assert result.n_requested == result.n_attempted == result.n_parsed == 4
    assert result.fail_counts == ()
    assert budget.used == len(client.physical.calls) == 5
    assert [event.attempt for event in budget.events] == [1, 2, 1, 1, 1]


def test_shared_physical_budget_stops_an_ensemble_after_a_retry() -> None:
    response = ModelResponse("up")
    budget = CallBudget(3)
    client = _BudgetedClient([TimeoutError("retry once"), response, response], budget)
    client.control.run(lambda: ModelResponse("ready"), stage="probe")

    with pytest.raises(BudgetExhaustedError):
        generate_ensemble(
            client, "p", _spec(max_workers=1, max_transport_failure_ratio=1.0), decide=lambda draw: draw.content
        )

    assert budget.used == 3
    assert len(client.physical.calls) == 2
    assert [event.stage for event in budget.events] == ["probe", "target", "target"]


def test_budget_exhaustion_aborts_instead_of_reporting_survivor_consensus() -> None:
    response = ModelResponse("up")
    failure = BudgetExhaustedError("shared physical-request budget exhausted")
    # The second wave has both a survivor and an exhausted admission. Even with
    # permissive failure/minimum settings, this hard stop must never look like
    # an ordinary failed draw in a completed, unanimous ensemble.
    client = _ScriptedClient([response, response, response, failure])

    with pytest.raises(BudgetExhaustedError) as caught:
        generate_ensemble(client, "p", _spec(max_transport_failure_ratio=1.0), decide=lambda draw: draw.content)

    assert caught.value is failure
    assert len(client.calls) == 4


def test_budget_exhaustion_prevents_scheduling_additional_waves() -> None:
    failure = BudgetExhaustedError("shared physical-request budget exhausted")
    client = _ScriptedClient([ModelResponse("up"), failure] + [ModelResponse("up")] * 6)

    with pytest.raises(BudgetExhaustedError) as caught:
        generate_ensemble(
            client, "p", _spec(draws=8, max_transport_failure_ratio=1.0), decide=lambda draw: draw.content
        )

    assert caught.value is failure
    assert len(client.calls) == 2


def test_budget_exhaustion_in_projection_also_aborts_remaining_waves() -> None:
    failure = BudgetExhaustedError("shared physical-request budget exhausted")
    client = _ScriptedClient([ModelResponse("up"), ModelResponse("budget")] + [ModelResponse("up")] * 6)

    def decide(response):
        if response.content == "budget":
            raise failure
        return response.content

    with pytest.raises(BudgetExhaustedError) as caught:
        generate_ensemble(client, "p", _spec(draws=8, max_transport_failure_ratio=1.0), decide=decide)

    assert caught.value is failure
    assert len(client.calls) == 2
