"""Tests for the harness.smoke module.

Covers Requirements 1.1 (≤ max_size shortlist), 1.2 (timeout / parse exclusion),
1.3 (missing-logprobs exclusion), and 1.4 (per-candidate outcomes returned for
runner-side persistence).
"""

from __future__ import annotations

import dataclasses

import pytest

from src.core.nvidia_lm import CompletionResult
from src.harness.smoke import Shortlist, SmokeOutcome, smoke_test


SMOKE_PROMPTS: list[str] = [
    "prompt-1",
    "prompt-2",
    "prompt-3",
    "prompt-4",
    "prompt-5",
]


def _completion(content: str) -> CompletionResult:
    """Build a minimal CompletionResult with non-empty logprobs."""
    return CompletionResult(
        content=content,
        logprobs=[],  # smoke gate does not inspect logprob list contents itself
        raw_temperature_observed=0.0,
    )


class _FakeLM:
    """Configurable fake LM that records calls and replays scripted behaviour."""

    def __init__(
        self,
        *,
        content: str = "Direction: 1",
        raise_on_call: int | None = None,
        exc: BaseException | None = None,
    ) -> None:
        self.content = content
        self.raise_on_call = raise_on_call
        self.exc = exc
        self.calls: list[str] = []

    def generate(self, prompt: str, temperature: float = 0.0) -> CompletionResult:
        self.calls.append(prompt)
        if (
            self.raise_on_call is not None
            and len(self.calls) >= self.raise_on_call
            and self.exc is not None
        ):
            raise self.exc
        return _completion(self.content)


def _factory_for(model_to_lm: dict[str, _FakeLM]):
    """Return an lm_factory that yields the per-model fake LM."""

    def factory(api_key: str, model: str, timeout_s: float):
        if model not in model_to_lm:
            raise AssertionError(f"unexpected model requested: {model}")
        return model_to_lm[model]

    return factory


def test_smoke_test_passes_clean_model() -> None:
    fakes = {"model-a": _FakeLM(content="Direction: 1")}
    result = smoke_test(
        candidates=["model-a"],
        api_key="key",
        smoke_prompts=SMOKE_PROMPTS,
        lm_factory=_factory_for(fakes),
    )

    assert result.selected == ["model-a"]
    assert len(result.outcomes) == 1
    outcome = result.outcomes[0]
    assert outcome.model == "model-a"
    assert outcome.passed is True
    assert outcome.fail_reason is None


def test_smoke_test_excludes_on_timeout() -> None:
    fakes = {
        "model-a": _FakeLM(raise_on_call=1, exc=TimeoutError("slow")),
    }
    result = smoke_test(
        candidates=["model-a"],
        api_key="key",
        smoke_prompts=SMOKE_PROMPTS,
        lm_factory=_factory_for(fakes),
    )

    assert result.selected == []
    assert len(result.outcomes) == 1
    outcome = result.outcomes[0]
    assert outcome.passed is False
    assert outcome.fail_reason == "timeout"
    # Should stop trying further prompts after the timeout.
    assert len(fakes["model-a"].calls) == 1


def test_smoke_test_excludes_on_missing_logprobs() -> None:
    fakes = {
        "model-a": _FakeLM(
            raise_on_call=1, exc=RuntimeError("missing top_logprobs")
        ),
    }
    result = smoke_test(
        candidates=["model-a"],
        api_key="key",
        smoke_prompts=SMOKE_PROMPTS,
        lm_factory=_factory_for(fakes),
    )

    assert result.selected == []
    outcome = result.outcomes[0]
    assert outcome.passed is False
    assert outcome.fail_reason == "no_logprobs"


def test_smoke_test_excludes_on_parse_failure() -> None:
    fakes = {"model-a": _FakeLM(content="I cannot answer")}
    result = smoke_test(
        candidates=["model-a"],
        api_key="key",
        smoke_prompts=SMOKE_PROMPTS,
        lm_factory=_factory_for(fakes),
    )

    assert result.selected == []
    outcome = result.outcomes[0]
    assert outcome.passed is False
    assert outcome.fail_reason == "parse_failure"


def test_smoke_test_caps_at_max_size() -> None:
    candidates = [f"model-{i:02d}" for i in range(15)]
    fakes = {model: _FakeLM(content="Direction: 0") for model in candidates}
    result = smoke_test(
        candidates=candidates,
        api_key="key",
        smoke_prompts=SMOKE_PROMPTS,
        max_size=10,
        lm_factory=_factory_for(fakes),
    )

    assert len(result.selected) == 10
    assert result.selected == candidates[:10]
    assert len(result.outcomes) == 15
    assert all(o.passed for o in result.outcomes)


def test_smoke_test_preserves_candidate_order() -> None:
    candidates = ["a", "b", "c"]
    fakes = {model: _FakeLM(content="Direction: -1") for model in candidates}
    result = smoke_test(
        candidates=candidates,
        api_key="key",
        smoke_prompts=SMOKE_PROMPTS,
        lm_factory=_factory_for(fakes),
    )

    assert result.selected == ["a", "b", "c"]
    assert [o.model for o in result.outcomes] == ["a", "b", "c"]


def test_smoke_test_outcomes_have_one_entry_per_candidate() -> None:
    fakes = {
        "ok-1": _FakeLM(content="Direction: 1"),
        "ok-2": _FakeLM(content="Direction: 0"),
        "timeout": _FakeLM(raise_on_call=1, exc=TimeoutError("x")),
        "no_logprobs": _FakeLM(
            raise_on_call=2, exc=RuntimeError("missing top_logprobs")
        ),
        "parse": _FakeLM(content="???"),
    }
    candidates = ["ok-1", "ok-2", "timeout", "no_logprobs", "parse"]
    result = smoke_test(
        candidates=candidates,
        api_key="key",
        smoke_prompts=SMOKE_PROMPTS,
        lm_factory=_factory_for(fakes),
    )

    assert len(result.outcomes) == 5
    by_model = {o.model: o for o in result.outcomes}
    assert by_model["ok-1"].passed is True
    assert by_model["ok-2"].passed is True
    assert by_model["timeout"].fail_reason == "timeout"
    assert by_model["no_logprobs"].fail_reason == "no_logprobs"
    assert by_model["parse"].fail_reason == "parse_failure"
    assert result.selected == ["ok-1", "ok-2"]


def test_smoke_outcome_and_shortlist_are_frozen() -> None:
    outcome = SmokeOutcome(model="m", passed=True, fail_reason=None)
    shortlist = Shortlist(selected=["m"], outcomes=[outcome])
    with pytest.raises(dataclasses.FrozenInstanceError):
        outcome.passed = False  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        shortlist.selected = []  # type: ignore[misc]


def test_smoke_test_runs_all_prompts_per_passing_candidate() -> None:
    candidates = ["m1", "m2", "m3"]
    fakes = {model: _FakeLM(content="Direction: 1") for model in candidates}
    smoke_test(
        candidates=candidates,
        api_key="key",
        smoke_prompts=SMOKE_PROMPTS,
        lm_factory=_factory_for(fakes),
    )

    total_calls = sum(len(f.calls) for f in fakes.values())
    assert total_calls == len(SMOKE_PROMPTS) * len(candidates) == 15
    for fake in fakes.values():
        assert len(fake.calls) == 5
