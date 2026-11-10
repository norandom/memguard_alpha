"""Serialised guarded results and draws pre-filled from a recording (task 7.4; Requirements 5.8, 11.1).

``GuardedRun``, ``GuardedDecision``, ``GuardIdentity`` and ``DistributionSummary``
round-trip through plain JSON, and ``guarded_run(recorded=...)`` issues only the
draws that are not recorded, with their deterministic seeds, then reduces the
union. Configured-client doubles answer by seed, as the local host does at a
fixed seed; no test sends a network request.
"""

from __future__ import annotations

import hashlib
import json
import socket
from dataclasses import replace

import pytest

from recall_guard.core.completion import ModelResponse
from recall_guard.core.distribution import ComponentDistribution, DistributionSummary, RecordedDraw
from recall_guard.harness.guarded_decision import (
    DrawMode,
    GuardConfig,
    GuardedDecision,
    GuardedRun,
    GuardIdentity,
    SourceCalibrator,
    decide_from_draws,
    guarded_run,
)
from recall_guard.mia.estimate_source import EstimateSource
from tests.harness.test_guarded_decision import (
    FULL,
    TOP_K,
    FakeFullDistribution,
    definition,
    model_config,
    reply,
    token_evidence,
)
from tests.harness.test_scorer_client import Client, calibrate

PROMPT = "Macro row for 2019-03. Reply with JSON loadings for equity and gold."
SEED_BASE = 100
DRAWS = 8


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("a guarded run attempted a network connection")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)


def warm_config():
    return model_config(temperature=0.7)


@pytest.fixture(scope="module")
def warm():
    """A top-k calibrator fitted at temperature 0.7, the temperature of the seeded draws."""
    return SourceCalibrator.from_scorer(calibrate(Client(warm_config())), source=EstimateSource.TOP_K,
                                        corpus_sha256=None)


@pytest.fixture(scope="module")
def warm_full():
    return SourceCalibrator.from_scorer(calibrate(Client(warm_config())), source=EstimateSource.FULL_DISTRIBUTION,
                                        corpus_sha256=None)


def seeded(source=TOP_K, **overrides) -> GuardConfig:
    values = {"draws": DRAWS, "min_parsed": 6, "temperature": 0.7, "seed_base": SEED_BASE} | overrides
    return GuardConfig(definition(), source, **values)


def content_for(seed: int) -> str:
    """The reply a seed selects: unimodal loadings on the 0.1 grid."""
    step = seed % 4
    return reply(round(0.1 + 0.1 * (step % 3), 1), round(0.4 + 0.1 * (step % 3), 1))


class SeededClient:
    """Configured-client double whose reply and token evidence depend on the seed only.

    ``prose`` lists seeds answered with unparseable text; ``failures`` maps seeds
    to the exception raised instead of a reply.
    """

    def __init__(self, *, configuration=None, prose=(), failures=None, evidence=True):
        self.config = configuration or warm_config()
        self.prose = set(prose)
        self.failures = dict(failures or {})
        self.evidence = evidence
        self.calls: list[dict] = []

    def generate(self, prompt, **parameters):
        self.calls.append(dict(parameters))
        seed = parameters.get("seed")
        if seed in self.failures:
            raise self.failures[seed]
        content = "The regime looks benign." if seed in self.prose else content_for(seed)
        return ModelResponse(content, token_evidence(-0.2 - 0.05 * (seed % 9)) if self.evidence else None,
                             usage={"completion_tokens": 11},
                             requested_settings=dict(self.config.parameters) | parameters)


class Forbidden:
    """A client that must never be asked: every draw is recorded."""

    def __init__(self):
        self.config = warm_config()

    def generate(self, prompt, **parameters):
        pytest.fail("a recorded draw was requested again")


def json_round(payload):
    return json.loads(json.dumps(payload, allow_nan=False))


def full_run(calibrator, **client_options) -> GuardedRun:
    return guarded_run(PROMPT, seeded(), client=SeededClient(**client_options), calibrator=calibrator)


# ---------- serialisation ----------


def test_an_ensemble_run_round_trips_through_plain_json(warm):
    run = full_run(warm, prose={103})
    payload = json_round(run.to_dict())
    restored = GuardedRun.from_dict(payload)
    assert restored == run
    assert restored.to_dict() == payload
    decision = restored.decision
    assert decision.status == "ok" and decision.mode is DrawMode.ENSEMBLE
    assert isinstance(decision.distribution, DistributionSummary)
    assert decision.distribution.n_failed == 1
    assert restored.responses[3].content == "The regime looks benign."
    assert all(isinstance(response, ModelResponse) for response in restored.responses)
    assert restored.evidence == (None,) * DRAWS


def test_summary_identity_and_component_round_trip_with_integer_quantile_levels(warm):
    decision = full_run(warm).decision
    summary = decision.distribution
    payload = json_round(summary.to_dict())
    assert set(payload["components"][0]["quantiles"]) == {"5", "25", "50", "75", "95"}
    restored = DistributionSummary.from_dict(payload)
    assert restored == summary
    assert set(restored.components[0].quantiles) == {5, 25, 50, 75, 95}
    assert restored.decision_agreement_ci == summary.decision_agreement_ci
    component = summary.components[0]
    assert ComponentDistribution.from_dict(json_round(component.to_dict())) == component
    identity = decision.identity
    restored_identity = GuardIdentity.from_dict(json_round(identity.to_dict()))
    assert restored_identity == identity
    assert restored_identity.source == TOP_K


def test_single_failed_and_split_results_round_trip(warm):
    single = guarded_run(PROMPT, GuardConfig(definition(), TOP_K, draws=1, min_parsed=1, mode=DrawMode.SINGLE,
                                             temperature=0.7, seed_base=SEED_BASE),
                         client=SeededClient(), calibrator=warm)
    failed = guarded_run(PROMPT, seeded(min_parsed=8), client=SeededClient(prose={100, 101}), calibrator=warm)
    assert single.decision.distribution is None and single.decision.mode is DrawMode.SINGLE
    assert failed.decision.status == "failed" and failed.decision.decision is None
    for run in (single, failed):
        assert GuardedRun.from_dict(json_round(run.to_dict())) == run
    split = replace(single.decision, decision={"equity": None, "gold": 0.5}, adjusted={"equity": None, "gold": 0.25})
    restored = GuardedDecision.from_dict(json_round(split.to_dict()))
    assert dict(restored.decision) == {"equity": None, "gold": 0.5}
    assert list(restored.decision) == ["equity", "gold"]


def test_full_distribution_evidence_round_trips_with_the_run(warm_full):
    run = guarded_run(PROMPT, seeded(FULL), client=SeededClient(), calibrator=warm_full,
                      full_distribution=FakeFullDistribution())
    restored = GuardedRun.from_dict(json_round(run.to_dict()))
    assert restored == run
    assert restored.evidence[0].model_digest == run.evidence[0].model_digest


@pytest.mark.parametrize(("change", "message"), [
    (lambda payload: payload | {"extra": 1}, "fields"),
    (lambda payload: {key: value for key, value in payload.items() if key != "status"}, "fields"),
    (lambda payload: payload | {"status": "maybe"}, "status"),
    (lambda payload: payload | {"mode": "triple"}, "mode"),
    (lambda payload: payload | {"p_memorized": "high"}, "p_memorized"),
    (lambda payload: payload | {"draws": [payload["draws"][0] | {"seed": "1"}]}, "seed"),
])
def test_a_decision_payload_with_unknown_missing_or_invalid_fields_is_refused(warm, change, message):
    payload = json_round(full_run(warm).decision.to_dict())
    with pytest.raises((ValueError, TypeError), match=message):
        GuardedDecision.from_dict(change(payload))


@pytest.mark.parametrize(("change", "message"), [
    (lambda payload: payload | {"estimate_sd": None, "extra": 0}, "fields"),
    (lambda payload: payload | {"estimate_quantiles": {"5": 0.1}}, "quantile"),
    (lambda payload: payload | {"decision_agreement_ci": [0.5]}, "decision_agreement_ci"),
    (lambda payload: payload | {"n_parsed": -1}, "n_parsed"),
])
def test_a_summary_payload_that_is_not_a_summary_is_refused(warm, change, message):
    payload = json_round(full_run(warm).decision.distribution.to_dict())
    with pytest.raises((ValueError, TypeError), match=message):
        DistributionSummary.from_dict(change(payload))


def test_an_identity_payload_with_an_unknown_source_is_refused(warm):
    payload = json_round(full_run(warm).decision.identity.to_dict())
    with pytest.raises(ValueError, match="source"):
        GuardIdentity.from_dict(payload | {"source": {"source": "logits", "k": 5}})
    with pytest.raises(ValueError, match="fields"):
        GuardIdentity.from_dict(payload | {"bearer": "x"})


def test_a_category_that_is_not_a_json_scalar_cannot_be_serialised(warm):
    decision = replace(full_run(warm).decision, category=("defensive", 1))
    with pytest.raises(ValueError, match="category"):
        decision.to_dict()


def test_a_run_whose_responses_do_not_match_its_draws_is_refused(warm):
    payload = json_round(full_run(warm).to_dict())
    with pytest.raises(ValueError, match="responses"):
        GuardedRun.from_dict(payload | {"responses": payload["responses"][:-1]})


# ---------- recorded draws ----------


def test_recorded_draws_are_prefilled_and_only_the_missing_draws_are_requested(warm):
    complete = full_run(warm)
    recorded = [draw for draw in complete.decision.draws if draw.draw_index in {0, 1, 2, 5}]
    client = SeededClient()
    resumed = guarded_run(PROMPT, seeded(), client=client, calibrator=warm, recorded=recorded)
    assert [call["seed"] for call in client.calls] == [103, 104, 106, 107]
    assert resumed.decision.to_dict() == complete.decision.to_dict()
    assert resumed.responses[0] is None and resumed.responses[3] == complete.responses[3]


def test_a_fully_recorded_ensemble_issues_no_request(warm):
    complete = full_run(warm)
    resumed = guarded_run(PROMPT, seeded(), client=Forbidden(), calibrator=warm, recorded=complete.decision.draws)
    assert resumed.decision == complete.decision
    assert resumed.responses == (None,) * DRAWS


def _failed(index: int, seed: int | None, temperature: float | None = 0.7) -> RecordedDraw:
    return RecordedDraw(draw_index=index, temperature=temperature, seed=seed,
                        response_sha256=hashlib.sha256(b"").hexdigest(), parsed=None,
                        failure_code="request_error:interrupted", estimate=None, evidence_ref=None)


@pytest.mark.parametrize(("recorded", "message"), [
    ([_failed(DRAWS, SEED_BASE + DRAWS)], "draw_index"),
    ([_failed(0, SEED_BASE), _failed(0, SEED_BASE)], "repeat"),
    ([_failed(1, 999)], "seed"),
    ([_failed(2, SEED_BASE + 2, temperature=0.2)], "temperature"),
    (["not a draw"], "RecordedDraw"),
])
def test_recorded_draws_that_do_not_fit_the_configuration_are_refused_before_any_request(warm, recorded, message):
    client = SeededClient()
    with pytest.raises((ValueError, TypeError), match=message):
        guarded_run(PROMPT, seeded(), client=client, calibrator=warm, recorded=recorded)
    assert client.calls == []


def test_hooks_see_each_issued_draw_before_and_after_its_request(warm):
    events = []
    client = SeededClient(prose={102})
    recorded = [_failed(0, SEED_BASE)]
    run = guarded_run(PROMPT, seeded(), client=client, calibrator=warm, recorded=recorded,
                      on_request=lambda index, seed: events.append(("request", index, seed, len(client.calls))),
                      on_draw=lambda draw, response, evidence: events.append(
                          ("draw", draw.draw_index, response.content if response else None, evidence)))
    assert events[0] == ("request", 1, 101, 0)
    assert events[1] == ("draw", 1, content_for(101), None)
    assert ("draw", 2, "The regime looks benign.", None) in events
    assert [event[1] for event in events if event[0] == "request"] == list(range(1, DRAWS))
    assert run.decision.draws[0] == recorded[0]


def test_a_hook_that_raises_before_a_request_stops_the_loop_before_it_is_sent(warm):
    client = SeededClient()

    def stop_at_three(index, seed):
        if index == 3:
            raise RuntimeError("admission gate closed")

    with pytest.raises(RuntimeError, match="admission gate"):
        guarded_run(PROMPT, seeded(), client=client, calibrator=warm, on_request=stop_at_three)
    assert [call["seed"] for call in client.calls] == [100, 101, 102]


def test_recomputation_of_a_serialised_decision_needs_no_request(warm):
    run = full_run(warm)
    restored = GuardedDecision.from_dict(json_round(run.decision.to_dict()))
    assert decide_from_draws(restored.draws, seeded(), restored.identity) == run.decision
