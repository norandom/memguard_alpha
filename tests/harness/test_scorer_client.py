"""Configured scoring uses real MIA fits and rejects incompatible evidence."""

import dataclasses
import hashlib
from types import SimpleNamespace

import pytest

from recall_guard.core.completion import (
    CapabilityEvidence,
    CapabilityReport,
    CapabilityStatus,
    CompletionProtocol,
    FeatureChoices,
    ModelConfig,
    ModelResponse,
    TokenAlternative,
    TokenEvidence,
)
from recall_guard.core.ensemble import EnsembleSpec
from recall_guard.core.loader import EvalRow
from recall_guard.core.nvidia_lm import CompletionResult
from recall_guard.core.providers import OpenAICompatibleClient
from recall_guard.core.request_control import BudgetExhaustedError, CallBudget, RequestError
from recall_guard.harness.scorer import ConfigurationError, MemoryGuardedScorer
from recall_guard.mia.control import build_baseline
from recall_guard.mia.features import compute_mia_features
from recall_guard.mia.mcs import train

IS = [f"is-{i}" for i in range(8)]
OOS = [f"oos-{i}" for i in range(8)]
SETTINGS = {"max_tokens": 512, "temperature": 0.0, "logprobs": True, "top_logprobs": 3}


def config(**overrides):
    return ModelConfig(protocol="openai_chat", endpoint="https://example.test/chat", model="m",
                       parameters=SETTINGS, **overrides)


class Client:
    """Deterministic synchronous provider double; its probe performs generation."""

    def __init__(self, configuration=None):
        self.config = configuration or config()
        self.calls = []
        self.failure = None
        self.fail_at = None
        self.response_change = lambda response: response
        self.report_change = lambda report: report

    def generate(self, prompt, **parameters):
        self.calls.append((prompt, parameters))
        if self.failure is not None and (self.fail_at is None or len(self.calls) >= self.fail_at):
            raise self.failure
        base = -0.2 if prompt.startswith("is") else -5.0 if prompt.startswith("oos") else -1.0
        jitter = (int(hashlib.sha256(prompt.encode()).hexdigest(), 16) % 1000) / 20000
        evidence = tuple(TokenEvidence("x", base + jitter + i * 0.01, (
            TokenAlternative("x", base + jitter), TokenAlternative("y", base - 1),
            TokenAlternative("z", base + 0.1),
        )) for i in range(5))
        return self.response_change(ModelResponse(
            "Direction: 1\nConfidence: 0.8", evidence,
            requested_settings=dict(self.config.parameters) | parameters,
        ))

    def probe(self):
        response = self.generate("probe")
        required = self.config.inspect(FeatureChoices(mia_scoring=True)).capabilities
        capabilities = {name: CapabilityEvidence(
            CapabilityStatus.UNSUPPORTED if name in {"logprobs", "top_logprobs"}
            and response.mia_unavailable_reason else CapabilityStatus.SUPPORTED,
            "Actual test-double generation", observed=True,
        ) for name in required}
        return self.report_change(CapabilityReport(capabilities, FeatureChoices(mia_scoring=True)))


def calibrate(client=None, **kwargs):
    return MemoryGuardedScorer.calibrate_client(
        client=client or Client(), is_memorized=IS, oos_control=OOS,
        min_valid=4, max_workers=1, **kwargs,
    )


def test_configured_fit_reuses_baseline_and_mcs_and_supplied_response_math():
    client = Client()
    scorer = calibrate(client)
    assert len(client.calls) == 25  # actual probe + control + labelled IS/OOS
    assert client.calls[0][0] == "probe"
    assert scorer.model == "m" and scorer.holdout_auc > 0.6 and not scorer.is_weak

    class LegacyLM:
        model = "m"

        def generate(self, prompt):
            return Client().generate(prompt).to_legacy()

    lm = LegacyLM()
    def rows(prompts):
        return [EvalRow(prompt, 0, {}) for prompt in prompts]
    baseline = build_baseline(lm, rows(OOS), None, min_valid=4)
    mcs = train(lm, rows(IS), rows(OOS), baseline, None)
    response = client.generate("eval")
    calls_before = len(client.calls)
    score = scorer.score_response("eval", response, client=client)
    features = compute_mia_features(response.content, response.to_legacy().logprobs, None)
    assert score.features == features
    assert score.p_memorized == mcs.predict_proba(features, baseline)
    assert score.memguard_confidence == 0.8 * (1 - score.p_memorized)
    assert len(client.calls) == calls_before  # supplied-response API makes no model call
    assert scorer.score("eval") == score
    assert scorer.score_many(["eval", "eval"], max_workers=2) == [score, score]
    assert isinstance(scorer._lm.generate("eval"), CompletionResult)


@pytest.mark.parametrize("missing", [None, (), (TokenEvidence("x", -1),)])
def test_text_or_incomplete_probe_evidence_rejected_before_corpus(missing):
    client = Client()
    client.response_change = lambda response: dataclasses.replace(response, token_evidence=missing)
    with pytest.raises(ValueError, match="MIA|token|capabilit"):
        calibrate(client)
    assert len(client.calls) == 1


@pytest.mark.parametrize("mutation", ["unobserved", "missing", "unsupported"])
def test_probe_must_report_every_required_capability_observed(mutation):
    client = Client()

    def changed(report):
        capabilities = dict(report.capabilities)
        if mutation == "missing":
            capabilities.pop("top_logprobs")
        else:
            capabilities["top_logprobs"] = CapabilityEvidence(
                "unsupported" if mutation == "unsupported" else "supported", "declaration", observed=False,
            )
        return CapabilityReport(capabilities, report.features)

    client.report_change = changed
    with pytest.raises(ValueError, match="probe|capabilit|MIA"):
        calibrate(client)
    assert len(client.calls) == 1


def test_local_incompatible_declaration_rejected_before_probe():
    client = Client(dataclasses.replace(config(), parameters=dict(SETTINGS, logprobs=False)))
    with pytest.raises(ValueError, match="MIA|logprobs"):
        calibrate(client)
    assert not client.calls


def test_identity_is_immutable_and_includes_protocol_model_endpoint_settings_intervention():
    client = Client()
    scorer = calibrate(client, intervention_id="rewrite-template-v3")
    identity = scorer.calibration_identity
    assert identity.model == "m"
    assert identity.endpoint == client.config.endpoint
    assert identity.protocol == CompletionProtocol.OPENAI_CHAT
    assert identity.intervention_id == "rewrite-template-v3"
    with pytest.raises(dataclasses.FrozenInstanceError):
        identity.model = "other"
    response = client.generate("eval")
    with pytest.raises(ValueError, match="intervention|identity"):
        scorer.score_response("eval", response, client=client)
    assert scorer.score_response("eval", response, client=client, intervention_id="rewrite-template-v3").parse_ok


@pytest.mark.parametrize("changed", [
    {"model": "other"}, {"endpoint": "https://other.test/chat"},
    {"protocol": "callback", "endpoint": None},
    {"parameters": dict(SETTINGS, temperature=0.7)},
    {"parameters": dict(SETTINGS, max_tokens=256)},
    {"parameters": {key: value for key, value in SETTINGS.items() if key != "temperature"}},
])
@pytest.mark.parametrize("entry_point", ["score", "score_many", "score_ensemble", "score_response"])
def test_changed_client_identity_rejected_before_generation(changed, entry_point):
    client = Client()
    scorer = calibrate(client)
    response = client.generate("eval")
    calls = len(client.calls)
    client.config = dataclasses.replace(client.config, **changed)
    with pytest.raises(ValueError, match="identity|calibrat"):
        if entry_point == "score_many":
            scorer.score_many(["eval"])
        elif entry_point == "score_ensemble":
            scorer.score_ensemble("eval", spec=EnsembleSpec(draws=2, min_parsed=1))
        elif entry_point == "score_response":
            scorer.score_response("eval", response, client=client)
        else:
            scorer.score("eval")
    assert len(client.calls) == calls


@pytest.mark.parametrize("override", [{"temperature": 0.4}, {"max_tokens": 256}])
def test_ensemble_override_mismatch_fails_before_any_draw(override):
    client = Client()
    scorer = calibrate(client)
    calls = len(client.calls)
    with pytest.raises(ValueError, match="identity|calibrat"):
        scorer.score_ensemble("eval", spec=EnsembleSpec(draws=2, min_parsed=1, **override))
    assert len(client.calls) == calls


def test_matching_ensemble_settings_keep_single_draw_math():
    client = Client()
    scorer = calibrate(client)
    single = scorer.score("eval")
    ensemble = scorer.score_ensemble("eval", spec=EnsembleSpec(
        draws=3, min_parsed=3, max_workers=1, max_tokens=512, temperature=0.0, retain_draws=True,
    ))
    assert ensemble.draws == (single,) * 3
    assert ensemble.p_memorized_point == pytest.approx(single.p_memorized)


@pytest.mark.parametrize("settings", [
    {}, dict(SETTINGS, temperature=0.4), dict(SETTINGS, seed=7),
])
def test_supplied_requested_settings_must_match_fit(settings):
    client = Client()
    scorer = calibrate(client)
    response = dataclasses.replace(client.generate("eval"), requested_settings=settings)
    with pytest.raises(ValueError, match="settings|identity|calibrat"):
        scorer.score_response("eval", response, client=client)


def test_known_confirmed_mismatch_rejected_but_unechoed_settings_remain_valid():
    client = Client()
    scorer = calibrate(client)
    response = client.generate("eval")
    assert scorer.score_response("eval", response, client=client).parse_ok
    with pytest.raises(ValueError, match="confirmed|settings|identity"):
        scorer.score_response("eval", dataclasses.replace(response, confirmed_settings={"temperature": 0.3}),
                              client=client)


@pytest.mark.parametrize("missing", [None, (), (TokenEvidence("x", -1),)])
def test_supplied_incomplete_evidence_is_explicitly_unavailable(missing):
    client = Client()
    scorer = calibrate(client)
    response = dataclasses.replace(client.generate("eval"), token_evidence=missing)
    with pytest.raises(ValueError, match="MIA|token|top_logprobs"):
        scorer.score_response("eval", response, client=client)


@pytest.mark.parametrize("fail_at", [1, 4, 12, 22])
@pytest.mark.parametrize("failure", [BudgetExhaustedError("limit"), RequestError("denied", status_code=403)])
def test_fatal_calibration_error_cannot_be_fitted_from_survivors(fail_at, failure):
    client = Client()
    client.failure, client.fail_at = failure, fail_at
    expected = BudgetExhaustedError if isinstance(failure, BudgetExhaustedError) else ConfigurationError
    with pytest.raises(expected):
        calibrate(client)
    assert len(client.calls) == fail_at


@pytest.mark.parametrize("entry_point", ["score", "score_many", "score_ensemble"])
def test_scoring_budget_exhaustion_propagates_as_hard_stop(entry_point):
    client = Client()
    scorer = calibrate(client)
    client.failure = BudgetExhaustedError("limit")
    calls = len(client.calls)
    with pytest.raises(BudgetExhaustedError):
        if entry_point == "score_many":
            scorer.score_many(["eval", "eval"], max_workers=1)
        elif entry_point == "score_ensemble":
            scorer.score_ensemble("eval", spec=EnsembleSpec(draws=4, min_parsed=1, max_workers=1))
        else:
            scorer.score("eval")
    assert len(client.calls) == calls + 1


def test_lost_evidence_or_response_identity_during_fit_stops_queued_calls():
    for mutate in [lambda response: dataclasses.replace(response, token_evidence=None),
                   lambda response: dataclasses.replace(response, requested_settings={})]:
        client = Client()
        client.response_change = lambda response, mutate=mutate, client=client: (
            mutate(response) if len(client.calls) >= 4 else response
        )
        with pytest.raises(ValueError, match="MIA|token|settings|identity"):
            calibrate(client)
        assert len(client.calls) == 4


def test_provider_probe_and_scoring_share_physical_budget_without_second_generation(monkeypatch):
    budget = CallBudget(26)
    client = OpenAICompatibleClient(config(), budget, features=FeatureChoices(mia_scoring=True),
                                    allow_unauthenticated=True, environment={})

    def post(payload):
        response = Client().generate(payload["messages"][0]["content"])
        return SimpleNamespace(status_code=200, headers={}, json=lambda: {"choices": [{
            "message": {"content": response.content},
            "logprobs": {"content": response.to_dict()["token_evidence"]},
        }]})

    monkeypatch.setattr(client, "_post", post)
    scorer = calibrate(client)
    assert budget.used == 25
    assert budget.events[0].stage == "probe"
    response = client.generate("eval")
    assert scorer.score_response("eval", response, client=client).parse_ok
    assert budget.used == 26
    with pytest.raises(BudgetExhaustedError):
        scorer.score("eval")
    assert budget.used == 26


@pytest.mark.parametrize("location", ["realized", "alternative"])
def test_censored_provider_probe_rejects_calibration_before_corpus(location, monkeypatch):
    budget = CallBudget(30)
    client = OpenAICompatibleClient(config(), budget, features=FeatureChoices(mia_scoring=True),
                                    allow_unauthenticated=True, environment={})

    def post(payload):
        response = Client().generate(payload["messages"][0]["content"])
        entries = response.to_dict()["token_evidence"]
        entry = entries[0] if location == "realized" else entries[0]["top_logprobs"][1]
        entry["logprob"] = -9999.0
        return SimpleNamespace(status_code=200, headers={}, json=lambda: {"choices": [{
            "message": {"content": response.content}, "logprobs": {"content": entries},
        }]})

    monkeypatch.setattr(client, "_post", post)
    with pytest.raises(ValueError, match="MIA|token|capabilit"):
        calibrate(client)
    assert budget.used == 1
    assert [event.stage for event in budget.events] == ["probe"]


@pytest.mark.parametrize("location", ["realized", "alternative"])
def test_censored_supplied_response_cannot_enter_existing_mia_math(location):
    client = Client()
    scorer = calibrate(client)
    response = client.generate("eval")
    evidence = (TokenEvidence("x", -9999.0 if location == "realized" else -0.2, (
        TokenAlternative("x", -9999.0 if location == "alternative" else -0.2),
    )),)
    response = dataclasses.replace(response, token_evidence=evidence)
    calls_before = len(client.calls)
    with pytest.raises(ValueError, match="censored"):
        scorer.score_response("eval", response, client=client)
    assert response.token_evidence == evidence
    assert len(client.calls) == calls_before


def test_provider_missing_evidence_error_is_explicit_and_cannot_fit_surviving_rows():
    client = Client()
    client.failure = RequestError("MIA evidence unavailable", code="missing_token_evidence")
    client.fail_at = 6
    with pytest.raises(ValueError, match="MIA|token|evidence"):
        calibrate(client)
    assert len(client.calls) == 6


def test_confirmed_default_observed_during_fit_cannot_change_at_scoring():
    client = Client(dataclasses.replace(config(), parameters={key: value for key, value in SETTINGS.items()
                                                            if key != "temperature"}))
    client.response_change = lambda response: dataclasses.replace(response, confirmed_settings={"temperature": 0.2})
    scorer = calibrate(client)
    response = dataclasses.replace(client.generate("eval"), confirmed_settings={"temperature": 0.4})
    with pytest.raises(ValueError, match="confirmed|settings|identity"):
        scorer.score_response("eval", response, client=client)


@pytest.mark.parametrize("entry_point", ["score", "score_many", "score_ensemble"])
@pytest.mark.parametrize("failure,reason", [(TimeoutError("slow"), "timeout"),
                                         (RequestError("unavailable", status_code=500), "error")])
def test_configured_ordinary_generation_failures_retain_legacy_failure_records(entry_point, failure, reason):
    client = Client()
    scorer = calibrate(client)
    client.failure = failure
    if entry_point == "score_many":
        scores = scorer.score_many(["eval", "eval"], max_workers=1)
    elif entry_point == "score_ensemble":
        ensemble = scorer.score_ensemble("eval", spec=EnsembleSpec(draws=2, min_parsed=1, max_workers=1))
        scores = [ensemble.consensus]
        assert ensemble.p_memorized_point is None
        # The pre-existing ensemble calls ordinary transport failures "transport".
        reason = "timeout" if isinstance(failure, TimeoutError) else "transport"
    else:
        scores = [scorer.score("eval")]
    assert all(not score.parse_ok and score.p_memorized is None and score.fail_reason == reason for score in scores)


@pytest.mark.parametrize("entry_point", ["score", "score_many", "score_ensemble"])
def test_configured_auth_failure_preserves_configuration_error_semantics(entry_point):
    client = Client()
    scorer = calibrate(client)
    client.failure = RequestError("denied", status_code=403)
    calls = len(client.calls)
    with pytest.raises(ConfigurationError):
        if entry_point == "score_many":
            scorer.score_many(["eval", "eval"], max_workers=1)
        elif entry_point == "score_ensemble":
            scorer.score_ensemble("eval", spec=EnsembleSpec(draws=2, min_parsed=1, max_workers=1))
        else:
            scorer.score("eval")
    assert len(client.calls) == calls + 1


def test_changed_config_cannot_be_hidden_by_matching_ensemble_override():
    client = Client()
    scorer = calibrate(client)
    client.config = dataclasses.replace(client.config, parameters=dict(SETTINGS, temperature=0.8))
    calls = len(client.calls)
    with pytest.raises(ValueError, match="identity"):
        scorer.score_ensemble("eval", spec=EnsembleSpec(draws=2, min_parsed=1, temperature=0.0))
    assert len(client.calls) == calls


def test_public_identity_check_validates_overrides_without_generation():
    client = Client()
    scorer = calibrate(client)
    calls = len(client.calls)
    scorer.validate_client(client, temperature=0.0)
    with pytest.raises(ValueError, match="identity"):
        scorer.validate_client(client, temperature=0.6)
    assert len(client.calls) == calls


def test_configured_no_valid_control_rows_has_provider_neutral_diagnostic():
    client = Client()
    client.failure, client.fail_at = TimeoutError("slow"), 2
    with pytest.raises(ConfigurationError, match="configured credentials") as caught:
        calibrate(client)
    assert "NVIDIA_API_KEY" not in str(caught.value)
