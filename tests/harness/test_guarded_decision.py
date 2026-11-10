"""Guarded decision: ensemble first, source-labelled estimates, replay without requests (task 5.1).

Covers Requirements 1.1, 1.2, 1.4 to 1.8, 5.1, 5.5 and 5.6. Configured-client
doubles return scripted replies that carry top-k token evidence, and a fake
full-distribution host returns per-position evidence. No test sends a network
request: the replay tests make every client raise and block HTTP outright.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import random
import statistics
from types import MappingProxyType

import pytest
import requests

from recall_guard.core.completion import ModelConfig, ModelResponse, TokenAlternative, TokenEvidence
from recall_guard.core.consensus import robust_location
from recall_guard.core.decision import ComponentSpec, DecisionDefinition, adjust, parse_json_components
from recall_guard.core.distribution import RecordedDraw, summarize_distribution
from recall_guard.core.ensemble import canonical_draw_hash
from recall_guard.core.full_distribution import FullDistributionClient, FullDistributionEvidence
from recall_guard.core.local_evidence import LocalModelIdentity, LocalModelProvenance
from recall_guard.core.profiles import ProfileError, load_profile
from recall_guard.core.request_control import BudgetExhaustedError, RequestError
from recall_guard.harness import evaluator, scorer_client
from recall_guard.harness import scorer as scorer_module
from recall_guard.harness.guarded_decision import (
    DrawMode,
    GuardConfig,
    GuardedRun,
    GuardIdentity,
    SourceCalibrator,
    decide_from_draws,
    full_distribution_client,
    guarded_decide,
    guarded_run,
)
from recall_guard.harness.scorer_client import CalibrationIdentity, CalibrationMismatchError
from recall_guard.mia.estimate_source import EstimateSource, SourceLabel, features_from_full, features_from_top_k
from tests.harness.test_scorer import _calibrate as calibrate_legacy
from tests.harness.test_scorer_client import SETTINGS, Client, calibrate

PROMPT = "Macro row for 2019-03. Reply with JSON loadings for equity and gold."
TOP_K = SourceLabel(EstimateSource.TOP_K, 3)
FULL = SourceLabel(EstimateSource.FULL_DISTRIBUTION, 4)
DIGEST = "sha256:" + "a" * 64
OTHER_DIGEST = "sha256:" + "b" * 64


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def payload_sha256(payload) -> str:
    """The evidence reference: SHA-256 of the canonical JSON of the evidence payload."""
    return sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def spread(vector):
    return vector["gold"] - vector["equity"]


def definition(*, grid: float | None = 0.1) -> DecisionDefinition:
    return DecisionDefinition(
        name="toy_allocation",
        components=(ComponentSpec("equity", -1.0, 1.0, grid=grid), ComponentSpec("gold", -1.0, 1.0, grid=grid)),
        parse=parse_json_components,
        summary=spread,
        decide=lambda vector: spread(vector) > 0,
    )


def reply(equity: float, gold: float) -> str:
    return json.dumps({"equity": equity, "gold": gold})


#: Thirty-two unimodal replies on the 0.1 grid; their consensus is equity 0.2, gold 0.5.
UNIMODAL = [reply(e, g) for e, g in zip([0.1, 0.2, 0.2, 0.3] * 8, [0.4, 0.5, 0.5, 0.6] * 8, strict=True)]


def split_replies() -> list[str]:
    """Sixteen replies whose equity loading splits into two separated clusters; gold is constant."""
    values = [-0.5] * 8 + [0.5] * 8
    random.Random(7).shuffle(values)
    return [reply(value, 0.3) for value in values]


def model_config(**parameters) -> ModelConfig:
    return ModelConfig(protocol="openai_chat", endpoint="https://example.test/chat", model="m",
                       parameters={**SETTINGS, **parameters})


def token_evidence(base: float) -> tuple[TokenEvidence, ...]:
    """Five realized tokens with three alternatives each, as a top-k host returns them."""
    return tuple(TokenEvidence("x", base + 0.01 * i, (
        TokenAlternative("x", base), TokenAlternative("y", base - 1.0), TokenAlternative("z", base + 0.1),
    )) for i in range(5))


class ScriptedClient:
    """Configured-client double: one scripted reply per call, with top-k token evidence.

    ``missing_evidence`` lists call indexes whose reply carries no token evidence;
    with ``raise_missing`` such a reply arrives inside a ``missing_token_evidence``
    RequestError, as a client built for MIA scoring reports it. ``failures`` maps
    call indexes to exceptions raised instead of a reply.
    """

    def __init__(self, contents, *, configuration=None, evidence=True, missing_evidence=(), raise_missing=False,
                 failures=None):
        self.config = configuration or model_config()
        self.contents = list(contents)
        self.missing = set(range(len(self.contents))) if not evidence else set(missing_evidence)
        self.raise_missing = raise_missing
        self.failures = dict(failures or {})
        self.calls = []

    def generate(self, prompt, **parameters):
        index = len(self.calls)
        self.calls.append((prompt, dict(parameters)))
        if index in self.failures:
            raise self.failures[index]
        evidence = None if index in self.missing else token_evidence(-0.2 - 0.15 * (index % 7))
        response = ModelResponse(self.contents[index], evidence,
                                 requested_settings=dict(self.config.parameters) | parameters)
        if evidence is None and self.raise_missing:
            raise RequestError("MIA/MCS token evidence is unavailable or empty.", code="missing_token_evidence",
                               response_evidence=response)
        return response


class FakeFullDistribution:
    """Full-distribution host double: four kept alternatives per position, from the full vector.

    ``refuse`` maps call indexes to the RequestError the host answers with instead.
    """

    def __init__(self, refuse=None):
        self.calls = []
        self.refuse = dict(refuse or {})

    def evidence(self, prompt_text, response_text, *, top_k_keep=100):
        index = len(self.calls)
        self.calls.append((prompt_text, response_text, top_k_keep))
        if index in self.refuse:
            raise self.refuse[index]
        base = -0.3 - 0.2 * (index % 5)
        positions = range(6)
        return FullDistributionEvidence(
            model_digest=DIGEST, n_vocab=50, response_token_ids=tuple(positions),
            logprob=tuple(base - 0.01 * i for i in positions), entropy=(1.0,) * 6, rank=(1,) * 6,
            top_logprobs=tuple(((i, base - 0.01 * i), (40, base - 1.0), (41, base - 2.0), (42, base - 3.0))
                               for i in positions),
        )


@dataclasses.dataclass(frozen=True)
class FixedEstimate(SourceCalibrator):
    """Calibrator double whose estimate is fixed, to test what an estimate may never do."""

    value: float = 1.0

    def estimate(self, features):
        return self.value


@pytest.fixture(scope="module")
def scorer():
    return calibrate(Client(model_config()))


@pytest.fixture(scope="module")
def warm_scorer():
    """A calibration fitted at temperature 0.7, for draws taken at that temperature."""
    return calibrate(Client(model_config(temperature=0.7)))


@pytest.fixture(scope="module")
def calibrator(scorer):
    return SourceCalibrator.from_scorer(scorer, source=EstimateSource.TOP_K, corpus_sha256=None)


@pytest.fixture(scope="module")
def full_calibrator(scorer):
    return SourceCalibrator.from_scorer(scorer, source=EstimateSource.FULL_DISTRIBUTION, corpus_sha256=None)


def small(source=TOP_K, **overrides) -> GuardConfig:
    """Eight draws, all required to parse unless overridden."""
    return GuardConfig(definition(), source, **{"draws": 8, "min_parsed": 8, **overrides})


def single(source=TOP_K, **overrides) -> GuardConfig:
    return GuardConfig(definition(), source, draws=1, min_parsed=1, mode=DrawMode.SINGLE, **overrides)


# -- configuration ----------------------------------------------------------------


def test_guard_config_defaults_to_a_32_draw_ensemble():
    config = GuardConfig(definition(), TOP_K)
    assert (config.draws, config.min_parsed, config.mode) == (32, 24, DrawMode.ENSEMBLE)
    assert config.temperature is None and config.seed_base is None


@pytest.mark.parametrize("overrides, message", [
    ({"draws": 8, "min_parsed": 9}, "min_parsed"),
    ({"draws": 0, "min_parsed": 0}, "draws"),
    ({"draws": 8, "min_parsed": 0}, "min_parsed"),
    ({"mode": DrawMode.SINGLE}, "SINGLE"),
    ({"draws": 2, "min_parsed": 1, "mode": DrawMode.SINGLE}, "SINGLE"),
])
def test_guard_config_rejects_impossible_settings_when_built(overrides, message):
    with pytest.raises(ValueError, match=message):
        GuardConfig(definition(), TOP_K, **overrides)


def test_single_draw_config_is_accepted_only_with_one_draw():
    config = single()
    assert (config.draws, config.min_parsed, config.mode) == (1, 1, DrawMode.SINGLE)


# -- ensemble default (1.1, 5.1, 5.2, 5.3, 5.5) -------------------------------------


def test_ensemble_default_draws_32_times_and_returns_consensus_estimate_distribution_and_adjustment(calibrator):
    client = ScriptedClient(UNIMODAL)
    config = GuardConfig(definition(), TOP_K)
    result = guarded_decide(PROMPT, config, client=client, calibrator=calibrator)

    assert client.calls == [(PROMPT, {})] * 32
    assert result.mode is DrawMode.ENSEMBLE and result.status == "ok" and result.reason is None
    assert [draw.draw_index for draw in result.draws] == list(range(32))
    assert [draw.response_sha256 for draw in result.draws] == [sha256(text) for text in UNIMODAL]
    # Without overrides each draw records the profile's temperature and no seed.
    assert {(draw.temperature, draw.seed) for draw in result.draws} == {(0.0, None)}

    assert result.decision == {"equity": 0.2, "gold": 0.5}
    estimates = [draw.estimate for draw in result.draws]
    assert None not in estimates and len(set(estimates)) > 1
    assert result.p_memorized == statistics.fmean(estimates)
    assert result.adjusted == adjust({"equity": 0.2, "gold": 0.5}, result.p_memorized)
    assert result.category is True
    assert result.not_assessed == ()
    assert result.distribution == summarize_distribution(result.draws, config.definition)
    assert (result.distribution.n_requested, result.distribution.n_parsed, result.distribution.n_failed) == (32, 32, 0)
    assert result.draws_sha256 == canonical_draw_hash([draw.response_sha256 for draw in result.draws])
    with pytest.raises(TypeError):
        result.decision["equity"] = 1.0  # type: ignore[index]


def test_consensus_is_the_reduce_draws_median_of_parsed_values(calibrator):
    values = [0.1, 0.4, 0.2, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2]
    client = ScriptedClient([reply(value, 0.5) for value in values])
    result = guarded_decide(PROMPT, small(draws=10), client=client, calibrator=calibrator)
    assert result.decision == {"equity": robust_location(values, mode="median"), "gold": 0.5}


def test_seeded_ensemble_passes_and_records_temperature_and_seed_per_draw(warm_scorer):
    calibrator = SourceCalibrator.from_scorer(warm_scorer, source=EstimateSource.TOP_K, corpus_sha256=None)
    client = ScriptedClient(UNIMODAL, configuration=model_config(temperature=0.7))
    result = guarded_decide(PROMPT, small(temperature=0.7, seed_base=100), client=client, calibrator=calibrator)

    assert client.calls == [(PROMPT, {"temperature": 0.7, "seed": 100 + i}) for i in range(8)]
    assert [(d.draw_index, d.temperature, d.seed) for d in result.draws] == [(i, 0.7, 100 + i) for i in range(8)]
    assert result.identity.temperature == 0.7
    assert result.status == "ok"


def test_per_draw_estimates_come_from_the_top_k_source_never_the_direction_parser(calibrator, mocker):
    for module in (evaluator, scorer_module):
        mocker.patch.object(module, "_parse_direction", side_effect=AssertionError("direction parser used"))
        mocker.patch.object(module, "_parse_confidence", side_effect=AssertionError("confidence parser used"))
    run = guarded_run(PROMPT, small(), client=ScriptedClient(UNIMODAL), calibrator=calibrator)

    assert isinstance(run, GuardedRun)
    assert run.evidence == (None,) * 8
    for draw, response in zip(run.decision.draws, run.responses, strict=True):
        sourced = features_from_top_k(response.content, response.to_legacy().logprobs)
        assert sourced.label == TOP_K
        assert draw.estimate == calibrator.estimate(sourced.features)
        assert draw.evidence_ref == payload_sha256(response.to_dict()["token_evidence"])


def test_full_distribution_source_requests_evidence_for_each_parsed_draw(full_calibrator):
    contents = UNIMODAL[:4] + ["I would rather not say."] + UNIMODAL[4:8] + ["{}"]
    client, full = ScriptedClient(contents), FakeFullDistribution()
    run = guarded_run(PROMPT, small(FULL, draws=10), client=client, calibrator=full_calibrator,
                      full_distribution=full)

    parsed = [draw.draw_index for draw in run.decision.draws if draw.parsed is not None]
    assert parsed == [0, 1, 2, 3, 5, 6, 7, 8]
    assert full.calls == [(PROMPT, contents[i], 100) for i in parsed]
    for draw, response, evidence in zip(run.decision.draws, run.responses, run.evidence, strict=True):
        if draw.parsed is None:
            assert (draw.failure_code, draw.estimate, draw.evidence_ref, evidence) == ("parse_error", None, None, None)
            continue
        sourced = features_from_full(response.content, evidence)
        assert sourced.label == FULL
        assert draw.estimate == full_calibrator.estimate(sourced.features)
        assert draw.evidence_ref == payload_sha256(evidence.to_dict())
    assert run.decision.status == "ok" and run.decision.identity.source == FULL


TOKENISATION_FAILED = RequestError("HTTP 422: full-distribution request rejected (tokenisation_failed)",
                                   status_code=422, code="tokenisation_failed")


def test_full_distribution_refusal_for_one_reply_leaves_only_that_estimate_missing(full_calibrator):
    full = FakeFullDistribution(refuse={2: TOKENISATION_FAILED})
    run = guarded_run(PROMPT, small(FULL), client=ScriptedClient(UNIMODAL), calibrator=full_calibrator,
                      full_distribution=full)

    assert len(full.calls) == 8
    draw = run.decision.draws[2]
    assert (draw.estimate, draw.evidence_ref, run.evidence[2]) == (None, None, None)
    assert draw.parsed is not None and draw.failure_code is None
    assert run.decision.status == "ok"
    assert run.decision.p_memorized == statistics.fmean(d.estimate for d in run.decision.draws if d.estimate is not None)
    assert "7 of 8 parsed draws" in run.decision.reason


@pytest.mark.parametrize("error", [
    RequestError("HTTP 401", status_code=401, code="auth_failed"),
    RequestError("HTTP 409", status_code=409, code="digest_mismatch"),
])
def test_full_distribution_rejected_bearer_or_other_weights_stop_the_run(full_calibrator, error):
    client, full = ScriptedClient(UNIMODAL), FakeFullDistribution(refuse={1: error})
    with pytest.raises(RequestError) as caught:
        guarded_decide(PROMPT, small(FULL), client=client, calibrator=full_calibrator, full_distribution=full)
    assert caught.value is error
    assert len(client.calls) == 2 and len(full.calls) == 2


def test_full_distribution_client_without_its_bearer_raises_instead_of_degrading(full_calibrator, monkeypatch):
    def no_request(*_args, **_kwargs):
        raise AssertionError("a request was sent")

    monkeypatch.setattr(requests.Session, "request", no_request)
    host = FullDistributionClient("http://127.0.0.1:11580", model_digest=DIGEST, credential_ref="env:FD_TEST_KEY",
                                  environment={})
    client = ScriptedClient(UNIMODAL)
    with pytest.raises(ValueError, match="credential"):
        guarded_decide(PROMPT, small(FULL), client=client, calibrator=full_calibrator, full_distribution=host)
    assert len(client.calls) == 1


def test_guarded_decide_returns_the_decision_of_the_run(calibrator):
    run = guarded_run(PROMPT, small(), client=ScriptedClient(UNIMODAL), calibrator=calibrator)
    assert guarded_decide(PROMPT, small(), client=ScriptedClient(UNIMODAL), calibrator=calibrator) == run.decision


# -- labelled single draw (1.2, 1.8) ------------------------------------------------


def test_single_draw_only_on_request_is_labelled_and_carries_no_distribution(calibrator):
    client = ScriptedClient([reply(0.3, -0.2)])
    result = guarded_decide(PROMPT, single(seed_base=5), client=client, calibrator=calibrator)

    assert client.calls == [(PROMPT, {"seed": 5})]
    assert result.mode is DrawMode.SINGLE and result.distribution is None
    assert result.status == "ok" and result.reason is None
    assert result.decision == {"equity": 0.3, "gold": -0.2}
    assert result.p_memorized is not None and result.p_memorized == result.draws[0].estimate
    assert result.adjusted == adjust({"equity": 0.3, "gold": -0.2}, result.p_memorized)
    assert result.category is False
    # The split check never runs on one draw: no component is presented as resolved.
    assert result.not_assessed == ("equity", "gold")
    assert [(d.draw_index, d.seed) for d in result.draws] == [(0, 5)]
    assert result.identity.draws == 1


def test_single_draw_estimate_never_withholds_or_zeroes_the_decision(calibrator):
    certain = FixedEstimate(calibrator.mcs, calibrator.baseline, calibrator.identity, calibrator.fingerprint, 1.0)
    result = guarded_decide(PROMPT, single(), client=ScriptedClient([reply(0.3, -0.2)]), calibrator=certain)

    assert result.status == "ok" and result.mode is DrawMode.SINGLE
    assert result.decision == {"equity": 0.3, "gold": -0.2}
    assert result.p_memorized == 1.0
    assert result.adjusted == adjust({"equity": 0.3, "gold": -0.2}, 1.0)


# -- pass-through statuses (1.4, 1.5) ------------------------------------------------


def test_no_token_evidence_passes_the_decision_through_as_estimate_unavailable(calibrator):
    result = guarded_decide(PROMPT, GuardConfig(definition(), TOP_K), client=ScriptedClient(UNIMODAL, evidence=False),
                            calibrator=calibrator)

    assert result.status == "estimate_unavailable"
    assert result.decision == {"equity": 0.2, "gold": 0.5}
    assert result.adjusted == result.decision
    assert result.p_memorized is None
    assert "0 of 32 parsed draws" in result.reason
    assert all(draw.estimate is None and draw.evidence_ref is None for draw in result.draws)
    assert result.distribution is not None and result.distribution.estimate_quantiles is None


def test_missing_token_evidence_error_keeps_the_reply_and_marks_the_estimate_unavailable(calibrator):
    client = ScriptedClient(UNIMODAL[:8], evidence=False, raise_missing=True)
    run = guarded_run(PROMPT, small(), client=client, calibrator=calibrator)

    assert run.decision.status == "estimate_unavailable"
    assert run.decision.decision == {"equity": 0.2, "gold": 0.5}
    assert [response.content for response in run.responses] == UNIMODAL[:8]
    assert all(draw.failure_code is None for draw in run.decision.draws)


def test_full_distribution_without_a_client_is_refused_before_any_request(full_calibrator):
    client = ScriptedClient(UNIMODAL)
    with pytest.raises(ValueError, match="full-distribution client"):
        guarded_decide(PROMPT, GuardConfig(definition(), FULL), client=client, calibrator=full_calibrator)
    assert client.calls == []


def test_partial_estimates_average_the_draws_that_carry_one_and_state_the_counts(calibrator):
    client = ScriptedClient(UNIMODAL, missing_evidence={0, 1, 2, 3})
    result = guarded_decide(PROMPT, GuardConfig(definition(), TOP_K), client=client, calibrator=calibrator)

    carried = [draw.estimate for draw in result.draws if draw.estimate is not None]
    assert len(carried) == 28
    assert result.status == "ok"
    assert result.p_memorized == statistics.fmean(carried)
    assert result.adjusted == adjust(result.decision, result.p_memorized)
    assert "28 of 32 parsed draws" in result.reason


def test_weak_calibrator_reports_decision_and_estimate_but_skips_adjustment(calibrator):
    weak = dataclasses.replace(calibrator, mcs=dataclasses.replace(calibrator.mcs, is_weak=True))
    assert weak.is_weak
    result = guarded_decide(PROMPT, small(), client=ScriptedClient(UNIMODAL), calibrator=weak)

    assert result.status == "adjustment_skipped_weak_calibrator"
    assert result.decision == {"equity": 0.2, "gold": 0.5}
    assert result.p_memorized == statistics.fmean(draw.estimate for draw in result.draws)
    assert result.adjusted == result.decision
    assert "weak" in result.reason
    assert result.identity.calibrator_is_weak is True


# -- minimum parsed draws (1.6, 5.6) -------------------------------------------------


def test_fewer_parsed_than_min_parsed_fails_with_counts_and_no_decision(calibrator):
    contents = UNIMODAL[:10] + ["No allocation this month."] * 20 + UNIMODAL[:2]
    failures = {30: RequestError("timed out", code="timeout"), 31: RequestError("reset", code="transport_error")}
    client = ScriptedClient(contents, failures=failures)
    result = guarded_decide(PROMPT, GuardConfig(definition(), TOP_K), client=client, calibrator=calibrator)

    assert len(client.calls) == 32
    assert result.status == "failed"
    assert result.decision is None and result.adjusted is None and result.p_memorized is None
    assert result.distribution is None and result.category is None and result.not_assessed == ()
    assert "10 of 32" in result.reason and "min_parsed=24" in result.reason
    assert "parse_error=20" in result.reason
    assert "request_error:timeout=1" in result.reason and "request_error:transport_error=1" in result.reason
    codes = [draw.failure_code for draw in result.draws]
    assert codes[30:] == ["request_error:timeout", "request_error:transport_error"]
    assert result.draws[30].response_sha256 == sha256("")
    assert result.draws_sha256 == canonical_draw_hash([d.response_sha256 for d in result.draws[:10]])


# -- unresolved and not-assessed components (5.7) ------------------------------------


def test_split_component_stays_unresolved_and_unadjusted(calibrator):
    client = ScriptedClient(split_replies())
    result = guarded_decide(PROMPT, small(draws=16), client=client, calibrator=calibrator)

    assert [c.unresolved for c in result.distribution.components] == [True, False]
    assert result.status == "ok"
    assert result.decision == {"equity": None, "gold": 0.3}
    assert result.adjusted == {"equity": None, "gold": adjust({"gold": 0.3}, result.p_memorized)["gold"]}
    assert result.category is None
    assert result.not_assessed == ()


@pytest.mark.parametrize("case", ["too_few_draws", "no_grid"])
def test_component_whose_split_check_could_not_run_is_flagged_not_assessed(calibrator, case):
    if case == "too_few_draws":
        values = [-0.5, 0.5] * 3 + [-0.5]
        config, contents = small(draws=7, min_parsed=4), [reply(v, 0.3) for v in values]
    else:
        config = GuardConfig(definition(grid=None), TOP_K, draws=16, min_parsed=8)
        contents = split_replies()
        values = [json.loads(text)["equity"] for text in contents]
    result = guarded_decide(PROMPT, config, client=ScriptedClient(contents), calibrator=calibrator)

    assert [c.unresolved for c in result.distribution.components] == [None, None]
    assert result.not_assessed == ("equity", "gold")
    # The value is kept, flagged, and adjusted like any present component.
    assert result.decision == {"equity": robust_location(values, mode="median"), "gold": 0.3}
    assert result.adjusted == adjust(result.decision, result.p_memorized)


# -- identity (1.7, 4.2) --------------------------------------------------------------


@pytest.mark.parametrize("configuration, overrides, field", [
    (model_config(max_tokens=256), {}, "settings"),
    (model_config(), {"temperature": 0.7}, "settings"),
    (dataclasses.replace(model_config(), model="other"), {}, "model"),
])
def test_identity_mismatch_is_refused_before_any_request(calibrator, configuration, overrides, field):
    client = ScriptedClient(UNIMODAL, configuration=configuration)
    with pytest.raises(CalibrationMismatchError) as caught:
        guarded_decide(PROMPT, small(**overrides), client=client, calibrator=calibrator)
    assert caught.value.fields == (field,)
    assert client.calls == []


def test_source_mismatch_is_refused_before_any_request(calibrator):
    client, full = ScriptedClient(UNIMODAL), FakeFullDistribution()
    with pytest.raises(CalibrationMismatchError) as caught:
        guarded_decide(PROMPT, small(FULL), client=client, calibrator=calibrator, full_distribution=full)
    assert caught.value.fields == ("estimate_source",)
    assert client.calls == [] and full.calls == []


def test_seed_settings_do_not_change_identity(calibrator):
    seeded_client = ScriptedClient(UNIMODAL, configuration=model_config(seed=9, random_seed=4))
    result = guarded_decide(PROMPT, small(), client=seeded_client, calibrator=calibrator)
    assert result.status == "ok"
    # Without seed_base the configured seed is the one each draw used.
    assert {draw.seed for draw in result.draws} == {9}

    fitted_with_seed = dataclasses.replace(calibrator, identity=dataclasses.replace(
        calibrator.identity, parameters_json=json.dumps(SETTINGS | {"seed": 3}, sort_keys=True, separators=(",", ":"))))
    result = guarded_decide(PROMPT, small(seed_base=50), client=ScriptedClient(UNIMODAL), calibrator=fitted_with_seed)
    assert result.status == "ok"
    assert [draw.seed for draw in result.draws] == [50 + i for i in range(8)]


def test_corpus_hash_of_the_calibrator_is_bound_without_refusal(scorer):
    corpus = sha256("calibration months")
    bound = SourceCalibrator.from_scorer(scorer, source=EstimateSource.TOP_K, corpus_sha256=corpus)
    result = guarded_decide(PROMPT, small(), client=ScriptedClient(UNIMODAL), calibrator=bound)
    assert result.status == "ok"


def test_local_identity_compares_declared_provenance_not_fit_observations(calibrator):
    declared = LocalModelProvenance(artifact_digest=DIGEST)
    fitted = dataclasses.replace(calibrator, identity=dataclasses.replace(calibrator.identity, local_identity=(
        LocalModelIdentity(declared=declared, observed=LocalModelProvenance(runtime="llama.cpp")))))
    same = dataclasses.replace(model_config(), local_identity=LocalModelIdentity(declared=declared))
    result = guarded_decide(PROMPT, small(), client=ScriptedClient(UNIMODAL, configuration=same), calibrator=fitted)
    assert result.status == "ok"

    other = dataclasses.replace(model_config(), local_identity=LocalModelIdentity(
        declared=LocalModelProvenance(artifact_digest=OTHER_DIGEST)))
    client = ScriptedClient(UNIMODAL, configuration=other)
    with pytest.raises(CalibrationMismatchError) as caught:
        guarded_decide(PROMPT, small(), client=client, calibrator=fitted)
    assert caught.value.fields == ("local_identity",)
    assert client.calls == []


def test_guard_identity_records_model_source_settings_calibrator_and_prompt_hash(scorer, calibrator):
    client = ScriptedClient(UNIMODAL, configuration=model_config(seed=9))
    result = guarded_decide(PROMPT, small(), client=client, calibrator=calibrator)

    assert result.identity == GuardIdentity(
        model="m", protocol="openai_chat", endpoint="https://example.test/chat", response_mapping="generic",
        source=TOP_K, settings_json=scorer_client.identity_settings(SETTINGS), temperature=0.0, draws=8,
        calibrator_fingerprint=scorer.calibration_fingerprint, calibrator_is_weak=False,
        prompt_sha256=sha256(PROMPT),
    )
    assert "seed" not in json.loads(result.identity.settings_json)


# -- request failures ---------------------------------------------------------------


@pytest.mark.parametrize("error", [
    RequestError("HTTP 401", status_code=401, code="auth_failed"),
    BudgetExhaustedError("budget exhausted"),
])
def test_rejected_credential_or_exhausted_budget_stops_the_draw_loop(calibrator, error):
    client = ScriptedClient(UNIMODAL, failures={3: error})
    with pytest.raises(type(error)):
        guarded_decide(PROMPT, GuardConfig(definition(), TOP_K), client=client, calibrator=calibrator)
    assert len(client.calls) == 4


# -- recomputation from recorded draws (5.8) -----------------------------------------


def _scenario(name, calibrator, full_calibrator):
    """Return (config, client, calibrator, full-distribution client) for one replay scenario."""
    if name == "ensemble_top_k":
        return small(draws=16), ScriptedClient(UNIMODAL[:16]), calibrator, None
    if name == "single_top_k":
        return single(seed_base=3), ScriptedClient([reply(0.3, -0.2)]), calibrator, None
    if name == "ensemble_full":
        contents = UNIMODAL[:9] + ["no answer"]
        return small(FULL, draws=10), ScriptedClient(contents), full_calibrator, FakeFullDistribution()
    if name == "single_full":
        return single(FULL), ScriptedClient([reply(0.3, -0.2)]), full_calibrator, FakeFullDistribution()
    if name == "weak":
        weak = dataclasses.replace(calibrator, mcs=dataclasses.replace(calibrator.mcs, is_weak=True))
        return small(), ScriptedClient(UNIMODAL[:8]), weak, None
    if name == "estimate_unavailable":
        return small(), ScriptedClient(UNIMODAL[:8], missing_evidence=range(8)), calibrator, None
    if name == "partial":
        return small(), ScriptedClient(UNIMODAL[:8], missing_evidence={1, 4}), calibrator, None
    if name == "failed":
        return small(), ScriptedClient(UNIMODAL[:4] + ["?"] * 4), calibrator, None
    return small(draws=16), ScriptedClient(split_replies()), calibrator, None


@pytest.mark.parametrize("name", ["ensemble_top_k", "single_top_k", "ensemble_full", "single_full", "weak",
                                  "estimate_unavailable", "partial", "failed", "split"])
def test_recomputation_from_recorded_draws_equals_the_live_result_and_issues_no_request(
        calibrator, full_calibrator, monkeypatch, name):
    config, client, source_calibrator, full = _scenario(name, calibrator, full_calibrator)
    live = guarded_decide(PROMPT, config, client=client, calibrator=source_calibrator, full_distribution=full)
    calls = len(client.calls), None if full is None else len(full.calls)

    def no_request(*_args, **_kwargs):
        raise AssertionError("replay issued a request")

    monkeypatch.setattr(client, "generate", no_request)
    if full is not None:
        monkeypatch.setattr(full, "evidence", no_request)
    monkeypatch.setattr(requests.Session, "request", no_request)

    assert decide_from_draws(live.draws, config, live.identity) == live
    stored = [RecordedDraw.from_dict(json.loads(json.dumps(draw.to_dict()))) for draw in live.draws]
    assert decide_from_draws(stored, config, live.identity) == live
    assert (len(client.calls), None if full is None else len(full.calls)) == calls


def test_decide_from_draws_refuses_draws_that_do_not_match_the_config_or_identity(calibrator):
    live = guarded_decide(PROMPT, small(), client=ScriptedClient(UNIMODAL), calibrator=calibrator)
    with pytest.raises(ValueError, match="draws"):
        decide_from_draws(live.draws[:7], small(), live.identity)
    with pytest.raises(ValueError, match="source"):
        decide_from_draws(live.draws, small(FULL), live.identity)
    with pytest.raises(ValueError, match="draws"):
        decide_from_draws(live.draws, small(), dataclasses.replace(live.identity, draws=16))


# -- calibrator wrapper and scorer read-only state ----------------------------------


def test_from_scorer_binds_source_corpus_and_fingerprint(scorer):
    corpus = sha256("calibration months")
    bound = SourceCalibrator.from_scorer(scorer, source=EstimateSource.FULL_DISTRIBUTION, corpus_sha256=corpus)

    assert bound.mcs is scorer.mcs and bound.baseline is scorer.control_baseline
    assert bound.identity == dataclasses.replace(scorer.calibration_identity, estimate_source="full_distribution",
                                                 corpus_sha256=corpus)
    assert bound.fingerprint == scorer.calibration_fingerprint
    assert bound.is_weak is scorer.is_weak is False
    assert bound.holdout_auc_interval() == scorer.holdout_auc_interval
    response = ScriptedClient([reply(0.1, 0.2)]).generate(PROMPT)
    features = features_from_top_k(response.content, response.to_legacy().logprobs).features
    assert bound.estimate(features) == scorer.mcs.predict_proba(features, scorer.control_baseline)


def test_from_scorer_refuses_a_calibration_without_a_configured_identity():
    with pytest.raises(ValueError, match="configured"):
        SourceCalibrator.from_scorer(calibrate_legacy(), source=EstimateSource.TOP_K, corpus_sha256=None)


def test_scorer_exposes_mcs_and_control_baseline_read_only(scorer):
    assert scorer.mcs is scorer._mcs
    assert scorer.control_baseline is scorer._baseline
    with pytest.raises(AttributeError):
        scorer.mcs = None  # type: ignore[misc]
    with pytest.raises(AttributeError):
        scorer.control_baseline = None  # type: ignore[misc]


def test_identity_settings_drops_seed_keys_and_matches_parameters_json():
    identity_settings = scorer_client.identity_settings
    assert identity_settings({"temperature": 0.0, "seed": 3, "random_seed": 4, "max_tokens": 5}) == (
        '{"max_tokens":5,"temperature":0.0}')
    configuration = model_config()
    assert identity_settings(configuration.parameters) == (
        CalibrationIdentity.from_config(configuration, "original").parameters_json)
    nested = MappingProxyType({"stop": ("a", "b"), "extra": MappingProxyType({"k": 1})})
    assert identity_settings(nested) == '{"extra":{"k":1},"stop":["a","b"]}'


# -- full-distribution client from a profile ----------------------------------------


def test_full_distribution_client_resolves_the_profile_endpoint_ref():
    profile = load_profile("llamacpp-gemma3-4b")
    host = profile.full_distribution
    gateway = "http://172.20.0.1:11581"
    client = full_distribution_client(profile, environment={"FULL_DISTRIBUTION_ENDPOINT_URL": gateway})
    assert isinstance(client, FullDistributionClient)
    assert client.base_url == gateway and client.model_digest == host.model_digest
    assert client.credential_ref == host.credential_ref
    assert full_distribution_client(profile, environment={}).base_url == host.endpoint

    secret_like = "https://user:hunter2@gateway.example/"
    with pytest.raises(ProfileError, match="FULL_DISTRIBUTION_ENDPOINT_URL") as caught:
        full_distribution_client(profile, environment={"FULL_DISTRIBUTION_ENDPOINT_URL": secret_like})
    assert "hunter2" not in str(caught.value)


def test_full_distribution_client_refuses_a_profile_without_the_source():
    with pytest.raises(ProfileError, match="available: top_k"):
        full_distribution_client("ollama-gemma3-4b", environment={})
