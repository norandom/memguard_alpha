"""The front, offline: what a person gets from ``recall_guard.ask`` (Requirement 11.11).

Every test enters through ``ask`` with fixed replies and a fixture catalogue
and reads the ``FrontAnswer`` a person would read. This file imports the
package root, pytest and the standard library only. The recall calibration
comes from the fixtures ``fitted_scorer`` and ``fitted_config`` of
``conftest.py``, because fitting one offline needs names the root does not
export.

No test uses a key, a network or a model host: for every test of the file,
opening a socket is refused and recorded, and the test fails if one was tried;
credential-like variables are removed from the process environment.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import json
import os
import re
import socket
import sys
from pathlib import Path

import pytest

from recall_guard import (
    WORKED_EXAMPLE_REPLY,
    ComponentSpec,
    DecisionDefinition,
    EstimateSource,
    FixedEvidence,
    FixedReplies,
    FrontAnswer,
    FullDistributionEvidence,
    Mode,
    SourceCalibrator,
    Status,
    ask,
    fixed_reply,
    load_catalogue,
    parse_json_components,
    worked_example,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "catalogue" / "catalogue_fixture.json"

#: The fixture's entries: logprobs qualified and logits not qualified, with a provider profile; logprobs
#: only and not tested; logits only, qualified, without a provider profile; neither mode offered.
LLAMACPP, OLLAMA, OGA, CLOSED = "llamacpp-gemma3-4b", "ollama-gemma3-4b", "oga-phi4-mini", "closed-route"
#: The calibrator fingerprint the fixture records for the llama.cpp entry's logprobs mode.
RECORDED = "sha256:fixture-series"
#: The variable the llama.cpp profile names for its credential and for its full-distribution host's bearer.
CREDENTIAL = "LOCAL_MODEL_API_KEY"

#: The fixed phrases a person reads in ``mode_choice`` and ``limits``.
NAMED = "named by the caller"
QUALIFIED_MODE = "the entry's qualified mode"
CALIBRATED_MODE = "no qualified mode; the mode a calibrator was supplied for"
LOGPROBS_OFFERED = "no qualified mode and no calibrator; logprobs is offered"
LOGITS_LEFT = "no qualified mode and no calibrator; logprobs is not offered, so logits"
NO_QUALIFIED_MODE = "no qualified mode for the entry"
NAMED_MODE_NOT_QUALIFIED = "the named mode is not qualified"
NO_CALIBRATOR = "no calibrator supplied"
NOT_THE_RECORDED_CALIBRATOR = "the supplied calibrator is not the recorded one"
CALIBRATOR_FOR_ANOTHER_MODEL = "the supplied calibrator was fitted for another model or settings"
DRAWS_MISSING = "some draws failed or could not be read"

#: Five output tokens with three alternatives each, as the calibration's settings ask for.
TOKENS = [("x", [("x", -0.2), ("y", -1.2), ("z", -2.1)])] * 5
EXAMPLE_ANSWER = json.loads(WORKED_EXAMPLE_REPLY)
AXES = ("inflation", "growth", "credit_stress", "policy", "risk_appetite")
SECRET = "s3cr3t-Value-0451"


# -- no key, no network, no model host ------------------------------------------------


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Refuse and record every socket; remove credential-like variables. Yields the record of attempts."""
    attempts: list[str] = []

    def refuse(*args, **kwargs):
        attempts.append(f"socket{args!r}")
        raise OSError("the offline integration tests open no socket")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    for name in list(os.environ):
        if re.search(r"KEY|TOKEN|SECRET|PASSWORD|BEARER|CREDENTIAL", name, re.IGNORECASE):
            monkeypatch.delenv(name)
    yield attempts
    # The guard records a failed request on its draw, so a refused socket need not surface as an error.
    assert attempts == [], "a socket was opened"


# -- what the tests ask with ---------------------------------------------------------


@pytest.fixture(scope="module")
def catalogue():
    return load_catalogue(FIXTURE)


@pytest.fixture(scope="module")
def none_qualified():
    """The fixture with no quality record for llama.cpp logprobs and for the in-process entry's logits."""
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    data["entries"][0]["modes"]["logprobs"]["calibrations"] = []
    data["entries"][2]["modes"]["logits"]["calibrations"] = []
    return load_catalogue(data)


@pytest.fixture(scope="module")
def logits_qualified():
    """The fixture with the llama.cpp entry's logits mode qualified under the recorded calibrator too."""
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    modes = data["entries"][0]["modes"]
    modes["logits"]["calibrations"] = [copy.deepcopy(modes["logprobs"]["calibrations"][1])]
    return load_catalogue(data)


@pytest.fixture(scope="module")
def own(fitted_scorer):
    """A logprobs calibrator under its own fingerprint, which no catalogue entry records."""
    return SourceCalibrator.from_scorer(fitted_scorer, source=EstimateSource.TOP_K, corpus_sha256=None)


@pytest.fixture(scope="module")
def recorded(own):
    """The logprobs calibrator the fixture records for the llama.cpp entry."""
    return dataclasses.replace(own, fingerprint=RECORDED)


@pytest.fixture(scope="module")
def for_logits(fitted_scorer):
    """A calibrator for the logits mode: fitted for the full next-token distribution."""
    return SourceCalibrator.from_scorer(fitted_scorer, source=EstimateSource.FULL_DISTRIBUTION, corpus_sha256=None)


def replies(fitted_config, count: int = 8, text: str = WORKED_EXAMPLE_REPLY) -> FixedReplies:
    """``count`` replies with token evidence, from a client that stands for the calibrated model."""
    return FixedReplies([fixed_reply(text, TOKENS)] * count, config=fitted_config)


def evidence(base: float = -0.3) -> FullDistributionEvidence:
    """What a full-distribution host returns for one reply: three positions, two kept alternatives each."""
    positions = range(3)
    return FullDistributionEvidence(
        model_digest="sha256:" + "a" * 64, n_vocab=50, response_token_ids=tuple(positions),
        logprob=tuple(base - 0.01 * i for i in positions), entropy=(1.0,) * 3, rank=(1,) * 3,
        top_logprobs=tuple(((i, base - 0.01 * i), (40, base - 1.0)) for i in positions),
    )


def host(count: int = 8) -> FixedEvidence:
    return FixedEvidence([evidence(-0.3 - 0.1 * i) for i in range(count)])


def run(entry: str, client, **options) -> FrontAnswer:
    """The worked example through the front: eight draws, all of them needed, unless overridden."""
    prompt, definition = worked_example()
    return ask(prompt, definition, entry=entry, client=client, **{"draws": 8, "min_parsed": 8, **options})


def answered(result: FrontAnswer) -> FrontAnswer:
    assert isinstance(result, FrontAnswer)
    assert result.failure is None
    assert result.answer == EXAMPLE_ANSWER
    return result


def no_answer(result: FrontAnswer, step: str) -> tuple[str, str]:
    """Assert a failure at ``step`` without any partial result; return its detail and remedy."""
    assert isinstance(result, FrontAnswer)
    assert result.answer is None and result.p_memorized is None and result.corrected is None
    assert result.limits == ()
    assert result.failure is not None and result.failure.step == step
    assert result.failure.detail.strip() and result.failure.remedy.strip()
    return result.failure.detail, result.failure.remedy


# -- every status (11.2, 11.5) --------------------------------------------------------


def test_a_qualified_entry_with_its_recorded_calibrator_gives_the_estimate_and_the_corrected_answer(
        catalogue, recorded, fitted_config):
    result = answered(run(LLAMACPP, replies(fitted_config), catalogue=catalogue, calibrator=recorded))

    assert (result.entry_id, result.mode, result.status) == (LLAMACPP, Mode.LOGPROBS, Status.QUALIFIED)
    assert result.reason == catalogue.qualification(LLAMACPP, Mode.LOGPROBS).reason == "all four checks passed"
    assert result.limits == ()
    assert result.p_memorized is not None and 0.0 < result.p_memorized < 1.0
    # The corrected answer is the answer scaled by one minus the recall estimate.
    assert set(result.corrected) == set(result.answer) == set(AXES)
    for axis in AXES:
        assert result.corrected[axis] == pytest.approx(result.answer[axis] * (1.0 - result.p_memorized))
    assert result.corrected != result.answer


def test_a_not_qualified_entry_still_answers_uncorrected_with_its_status_and_the_catalogues_reason(
        catalogue, for_logits, fitted_config):
    expected = catalogue.qualification(LLAMACPP, Mode.LOGITS)
    assert expected.status is Status.NOT_QUALIFIED and expected.reason.startswith("discrimination")

    with_estimate = answered(run(LLAMACPP, replies(fitted_config), catalogue=catalogue, mode=Mode.LOGITS,
                                 calibrator=for_logits, full_distribution=host()))
    without = answered(run(LLAMACPP, replies(fitted_config), catalogue=catalogue, mode=Mode.LOGITS))

    for result in (with_estimate, without):
        assert (result.entry_id, result.mode) == (LLAMACPP, Mode.LOGITS)
        assert (result.status, result.reason) == (Status.NOT_QUALIFIED, expected.reason)
        assert result.corrected is None
    assert with_estimate.p_memorized is not None
    assert with_estimate.limits == (NAMED_MODE_NOT_QUALIFIED,)
    assert without.limits == (NAMED_MODE_NOT_QUALIFIED, NO_CALIBRATOR)


def test_a_not_tested_entry_still_answers_uncorrected_with_its_status(catalogue, own, fitted_config):
    with_estimate = answered(run(OLLAMA, replies(fitted_config), catalogue=catalogue, calibrator=own))
    without = answered(run(OLLAMA, replies(fitted_config), catalogue=catalogue))

    for result in (with_estimate, without):
        assert (result.entry_id, result.mode) == (OLLAMA, Mode.LOGPROBS)
        assert (result.status, result.reason) == (Status.NOT_TESTED, "no quality record")
        assert result.corrected is None
    assert with_estimate.p_memorized is not None
    assert with_estimate.limits == (NO_QUALIFIED_MODE,)
    assert without.limits == (NO_QUALIFIED_MODE, NO_CALIBRATOR)


# -- the calibrator (11.6, 11.8) ------------------------------------------------------


def test_a_calibrator_other_than_the_recorded_one_gives_the_estimate_and_no_correction(
        catalogue, own, fitted_config):
    assert own.fingerprint != RECORDED
    result = answered(run(LLAMACPP, replies(fitted_config), catalogue=catalogue, calibrator=own))

    assert result.mode is Mode.LOGPROBS
    assert result.p_memorized is not None and result.corrected is None
    assert result.status is not Status.QUALIFIED and "fingerprint" in result.reason
    assert result.limits == (NOT_THE_RECORDED_CALIBRATOR,)


def test_a_calibrator_fitted_for_another_model_gives_the_answer_without_an_estimate_and_asks_once(
        catalogue, recorded, fitted_config):
    another_model = dataclasses.replace(fitted_config, model="another-model")
    other_settings = dataclasses.replace(fitted_config, parameters={**fitted_config.parameters, "temperature": 0.9})
    clients = [FixedReplies([fixed_reply(WORKED_EXAMPLE_REPLY, TOKENS)] * 8, config=another_model),
               FixedReplies([fixed_reply(WORKED_EXAMPLE_REPLY, TOKENS)] * 8, config=other_settings),
               FixedReplies([WORKED_EXAMPLE_REPLY] * 8)]  # the default: no model the calibrator knows

    for client in clients:
        result = answered(run(LLAMACPP, client, catalogue=catalogue, calibrator=recorded))

        assert (result.p_memorized, result.corrected) == (None, None)
        assert result.status is Status.QUALIFIED  # the catalogue's status for the entry and mode
        assert result.limits == (CALIBRATOR_FOR_ANOTHER_MODEL,)
        assert client.requests == 8  # the model was asked once: eight draws, eight requests


def test_without_a_calibrator_the_answer_comes_with_the_catalogue_status_and_no_estimate(catalogue):
    client = FixedReplies([WORKED_EXAMPLE_REPLY] * 8)
    result = answered(run(LLAMACPP, client, catalogue=catalogue))

    assert (result.entry_id, result.mode) == (LLAMACPP, Mode.LOGPROBS)
    assert (result.status, result.reason) == (Status.QUALIFIED, "all four checks passed")
    assert (result.p_memorized, result.corrected) == (None, None)
    assert result.limits == (NO_CALIBRATOR,)
    assert client.requests == 8


# -- both modes -----------------------------------------------------------------------


def test_the_logprobs_mode_estimates_from_the_token_evidence_of_the_replies(catalogue, recorded, fitted_config):
    with_evidence = answered(run(LLAMACPP, replies(fitted_config), catalogue=catalogue, mode=Mode.LOGPROBS,
                                 calibrator=recorded))
    text_only = answered(run(LLAMACPP, FixedReplies([WORKED_EXAMPLE_REPLY] * 8, config=fitted_config),
                             catalogue=catalogue, mode=Mode.LOGPROBS, calibrator=recorded))

    assert with_evidence.mode is text_only.mode is Mode.LOGPROBS
    assert with_evidence.p_memorized is not None and with_evidence.corrected is not None
    # Replies without token evidence leave nothing to estimate from, and the answer says so.
    assert (text_only.p_memorized, text_only.corrected) == (None, None)
    assert text_only.limits == ("the recall estimate could not be computed",)


def test_the_logits_mode_estimates_from_the_evidence_of_a_full_distribution_host(
        logits_qualified, for_logits, fitted_config):
    prompt, _ = worked_example()
    client, fixed_host = FixedReplies([WORKED_EXAMPLE_REPLY] * 8, config=fitted_config), host()
    result = answered(run(LLAMACPP, client, catalogue=logits_qualified, mode=Mode.LOGITS,
                          calibrator=dataclasses.replace(for_logits, fingerprint=RECORDED),
                          full_distribution=fixed_host))

    assert (result.mode, result.status, result.limits) == (Mode.LOGITS, Status.QUALIFIED, ())
    assert result.p_memorized is not None
    for axis in AXES:
        assert result.corrected[axis] == pytest.approx(result.answer[axis] * (1.0 - result.p_memorized))
    # The host was asked once per reply, with the prompt and that reply.
    assert (client.requests, fixed_host.requests) == (8, 8)
    assert fixed_host.calls == ((prompt, WORKED_EXAMPLE_REPLY),) * 8


def test_a_logits_host_that_returns_no_evidence_leaves_the_answer_without_an_estimate(
        catalogue, for_logits, fitted_config):
    result = answered(run(LLAMACPP, replies(fitted_config), catalogue=catalogue, mode=Mode.LOGITS,
                          calibrator=for_logits, full_distribution=FixedEvidence([])))

    assert (result.p_memorized, result.corrected) == (None, None)
    assert result.limits == (NAMED_MODE_NOT_QUALIFIED, "the recall estimate could not be computed")


# -- the mode choice (11.4) -----------------------------------------------------------


def test_with_no_mode_named_a_qualified_mode_is_chosen_first(catalogue, for_logits, fitted_config):
    plain = answered(run(LLAMACPP, replies(fitted_config), catalogue=catalogue))
    # A calibrator for the other mode does not outrank the qualified mode.
    with_other = answered(run(LLAMACPP, replies(fitted_config), catalogue=catalogue, calibrator=for_logits))
    logits_only = answered(run(OGA, replies(fitted_config), catalogue=catalogue))

    assert (plain.mode, plain.mode_choice) == (Mode.LOGPROBS, QUALIFIED_MODE)
    assert (with_other.mode, with_other.mode_choice) == (Mode.LOGPROBS, QUALIFIED_MODE)
    assert (logits_only.mode, logits_only.mode_choice, logits_only.status) == (
        Mode.LOGITS, QUALIFIED_MODE, Status.QUALIFIED)


def test_with_no_qualified_mode_the_mode_a_calibrator_was_supplied_for_is_chosen(
        none_qualified, for_logits, fitted_config):
    result = answered(run(LLAMACPP, replies(fitted_config), catalogue=none_qualified, calibrator=for_logits,
                          full_distribution=host()))

    assert (result.mode, result.mode_choice) == (Mode.LOGITS, CALIBRATED_MODE)
    assert result.p_memorized is not None


def test_with_no_qualified_mode_and_no_calibrator_logprobs_is_chosen_if_offered(none_qualified, fitted_config):
    result = answered(run(LLAMACPP, replies(fitted_config), catalogue=none_qualified))

    assert (result.mode, result.mode_choice) == (Mode.LOGPROBS, LOGPROBS_OFFERED)
    assert result.limits == (NO_QUALIFIED_MODE, NO_CALIBRATOR)


def test_with_no_qualified_mode_no_calibrator_and_no_logprobs_logits_is_chosen(none_qualified, fitted_config):
    result = answered(run(OGA, replies(fitted_config), catalogue=none_qualified))

    assert (result.mode, result.mode_choice, result.status) == (Mode.LOGITS, LOGITS_LEFT, Status.NOT_TESTED)


@pytest.mark.parametrize("named", [Mode.LOGITS, "logits"])
def test_a_named_mode_is_used_even_when_another_mode_is_the_qualified_one(catalogue, fitted_config, named):
    result = answered(run(LLAMACPP, replies(fitted_config), catalogue=catalogue, mode=named))

    assert (result.mode, result.mode_choice) == (Mode.LOGITS, NAMED)
    assert result.status is Status.NOT_QUALIFIED


# -- the worked example and any other prompt (11.3, 11.10) -----------------------------


def test_the_worked_example_runs_end_to_end_with_one_fixed_reply(catalogue):
    prompt, definition = worked_example()
    client = FixedReplies([WORKED_EXAMPLE_REPLY], cycle=True)

    result = ask(prompt, definition, entry=LLAMACPP, client=client, catalogue=catalogue)

    assert result.failure is None
    assert list(result.answer) == list(AXES)
    assert result.answer == EXAMPLE_ANSWER
    assert all(-1.0 <= value <= 1.0 for value in result.answer.values())
    # The default ensemble: thirty-two requests, each with the prompt byte for byte.
    assert client.requests == 32 and (result.draws_asked, result.draws_used) == (32, 32)
    assert [sent.encode("utf-8") for sent in client.prompts] == [prompt.encode("utf-8")] * 32
    assert (result.entry_id, result.mode, result.status) == (LLAMACPP, Mode.LOGPROBS, Status.QUALIFIED)
    assert result.limits == (NO_CALIBRATOR,)


def test_any_prompt_with_its_own_decision_definition_runs_the_same_way(catalogue, recorded, fitted_config):
    prompt = "  Tomorrow at the coast: how much weight on an umbrella and on sunscreen?\n{no template} %s ¿sí? "
    definition = DecisionDefinition(
        name="beach_bag",
        components=(ComponentSpec("umbrella", 0.0, 1.0), ComponentSpec("sunscreen", 0.0, 1.0)),
        parse=parse_json_components,
        summary=lambda vector: vector["umbrella"] - vector["sunscreen"],
        decide=lambda vector: vector["umbrella"] > vector["sunscreen"],
    )
    texts = ['{"umbrella": 0.7, "sunscreen": 0.2}', '{"umbrella": 0.5, "sunscreen": 0.4}',
             '{"umbrella": 0.6, "sunscreen": 0.3}']
    client = FixedReplies([fixed_reply(text, TOKENS) for text in texts], config=fitted_config, cycle=True)

    result = ask(prompt, definition, entry=LLAMACPP, client=client, catalogue=catalogue, calibrator=recorded,
                 draws=9, min_parsed=9)

    assert result.failure is None and result.limits == ()
    assert result.answer == pytest.approx({"umbrella": 0.6, "sunscreen": 0.3})
    assert (result.mode, result.status) == (Mode.LOGPROBS, Status.QUALIFIED)
    assert result.p_memorized is not None
    for name in ("umbrella", "sunscreen"):
        assert result.corrected[name] == pytest.approx(result.answer[name] * (1.0 - result.p_memorized))
    assert client.prompts == (prompt,) * 9


# -- each failure, and what the caller can change (11.9) -------------------------------


def test_an_unknown_entry_fails_at_the_entry_step_and_names_the_entries_to_choose_from(catalogue):
    client = FixedReplies([WORKED_EXAMPLE_REPLY], cycle=True)
    result = run("no-such-entry", client, catalogue=catalogue)

    detail, remedy = no_answer(result, "entry")
    assert (result.entry_id, result.mode, result.status) == ("no-such-entry", None, None)
    assert "no-such-entry" in detail
    assert all(entry in remedy for entry in (LLAMACPP, OLLAMA, OGA, CLOSED))
    assert client.requests == 0


@pytest.mark.parametrize("named", [Mode.LOGITS, "logits"])
def test_a_named_mode_the_entry_does_not_offer_fails_at_the_mode_step_and_lists_the_modes(catalogue, named):
    client = FixedReplies([WORKED_EXAMPLE_REPLY], cycle=True)
    result = run(OLLAMA, client, catalogue=catalogue, mode=named)

    detail, remedy = no_answer(result, "mode")
    assert result.mode_choice == NAMED
    assert "logits" in detail
    assert "name a mode the entry offers" in remedy and "logprobs is not_tested" in remedy
    assert "logits is not_tested (not offered)" in remedy
    assert client.requests == 0


def test_an_entry_that_offers_neither_mode_fails_at_the_mode_step(catalogue):
    client = FixedReplies([WORKED_EXAMPLE_REPLY], cycle=True)
    result = run(CLOSED, client, catalogue=catalogue)

    _detail, remedy = no_answer(result, "mode")
    assert (result.entry_id, result.mode, result.status) == (CLOSED, None, None)
    assert "use an entry that offers a mode" in remedy
    assert "logprobs is not_tested (not offered)" in remedy and "logits is not_tested (not offered)" in remedy
    assert client.requests == 0


def test_an_entry_without_a_provider_profile_and_no_client_fails_at_the_client_step(catalogue):
    prompt, definition = worked_example()
    result = ask(prompt, definition, entry=OGA, catalogue=catalogue)

    detail, remedy = no_answer(result, "client")
    assert OGA in detail and "no provider profile" in detail
    assert "client=" in remedy
    assert (result.mode, result.status) == (Mode.LOGITS, Status.QUALIFIED)


def test_a_profile_whose_credential_is_not_set_fails_at_the_client_step_and_names_the_variable(catalogue):
    prompt, definition = worked_example()
    result = ask(prompt, definition, entry=LLAMACPP, catalogue=catalogue, environment={})

    detail, remedy = no_answer(result, "client")
    assert CREDENTIAL in detail
    assert LLAMACPP in remedy and "client=" in remedy
    assert (result.mode, result.status) == (Mode.LOGPROBS, Status.QUALIFIED)


@pytest.mark.parametrize("calibrated", [True, False])
def test_replies_that_run_out_below_min_parsed_fail_at_the_request_step(
        catalogue, recorded, fitted_config, calibrated):
    client = replies(fitted_config, count=3)
    result = run(LLAMACPP, client, catalogue=catalogue, calibrator=recorded if calibrated else None)

    detail, remedy = no_answer(result, "request")
    assert "3 of 8 draws parsed" in detail and "fixed_replies_exhausted=5" in detail
    assert "model host" in remedy and "ask again" in remedy
    assert client.requests == 3
    assert (result.draws_asked, result.draws_used) == (None, None)


@pytest.mark.parametrize("calibrated", [True, False])
def test_replies_that_cannot_be_read_into_the_definition_fail_at_the_reply_step(
        catalogue, recorded, fitted_config, calibrated):
    unreadable = ["I would rather not say.", '{"inflation": 0.4}', '{"inflation": "high", "growth": 0}']
    client = FixedReplies(unreadable, config=fitted_config, cycle=True)
    result = run(LLAMACPP, client, catalogue=catalogue, calibrator=recorded if calibrated else None)

    detail, remedy = no_answer(result, "reply")
    assert "0 of 8 draws parsed" in detail
    assert "definition" in remedy and "min_parsed" in remedy
    assert client.requests == 8
    assert (result.entry_id, result.mode, result.status) == (LLAMACPP, Mode.LOGPROBS, Status.QUALIFIED)


# -- draws that went missing ----------------------------------------------------------


@pytest.mark.parametrize("calibrated", [True, False])
def test_an_answer_that_rests_on_fewer_draws_than_asked_says_so_and_counts_them(
        catalogue, recorded, fitted_config, calibrated):
    # Thirty-two draws asked: twenty replies arrive, two of them unreadable, then the host stops answering.
    records = [fixed_reply(WORKED_EXAMPLE_REPLY, TOKENS)] * 18 + [fixed_reply("no idea", TOKENS)] * 2
    client = FixedReplies(records, config=fitted_config)
    prompt, definition = worked_example()

    result = ask(prompt, definition, entry=LLAMACPP, client=client, catalogue=catalogue, min_parsed=16,
                 calibrator=recorded if calibrated else None)

    assert result.failure is None and result.answer == EXAMPLE_ANSWER
    assert (result.draws_asked, result.draws_used) == (32, 18)
    assert result.limits == ((DRAWS_MISSING,) if calibrated else (NO_CALIBRATOR, DRAWS_MISSING))
    assert (result.p_memorized is not None) is calibrated
    assert client.requests == 20


def test_a_complete_ensemble_counts_every_draw_and_reports_none_missing(catalogue, recorded, fitted_config):
    result = answered(run(LLAMACPP, replies(fitted_config), catalogue=catalogue, calibrator=recorded))

    assert (result.draws_asked, result.draws_used) == (8, 8)
    assert DRAWS_MISSING not in result.limits


# -- no credential value in a result --------------------------------------------------


def shows(result: FrontAnswer, value: str) -> bool:
    texts = [repr(result), str(result), *(repr(getattr(result, field.name)) for field in dataclasses.fields(result))]
    if result.failure is not None:
        texts += [result.failure.detail, result.failure.remedy]
    return any(value in text for text in texts)


def test_no_result_carries_the_value_of_a_credential(catalogue, recorded, for_logits, fitted_config, monkeypatch):
    prompt, definition = worked_example()
    given = {CREDENTIAL: SECRET, "LLAMACPP_ENDPOINT_URL": "http://127.0.0.1:9/v1/chat/completions"}

    # An answer, with the credential in the environment the front is given and in the process environment.
    monkeypatch.setenv(CREDENTIAL, SECRET)
    answer = answered(run(LLAMACPP, replies(fitted_config), catalogue=catalogue, calibrator=recorded,
                          environment=given))
    # A failure that is about the credential: the front reads the logits host's bearer from the environment
    # it is given and refuses a malformed one before any request. It names the variable, never the value.
    malformed = f"{SECRET} with-a-space"
    client = replies(fitted_config)
    refused = ask(prompt, definition, entry=LLAMACPP, mode=Mode.LOGITS, client=client, calibrator=for_logits,
                  catalogue=catalogue, draws=8, min_parsed=8, environment={CREDENTIAL: malformed})

    detail, remedy = no_answer(refused, "client")
    assert CREDENTIAL in detail and CREDENTIAL in remedy and client.requests == 0
    for result in (answer, refused):
        assert not shows(result, SECRET)
        assert not shows(result, "127.0.0.1:9")
    assert shows(answer, LLAMACPP)  # the check does find what a result holds


# -- the rules of this file -----------------------------------------------------------


def test_a_socket_is_refused_and_recorded_so_no_test_of_this_file_reaches_a_network(offline):
    with pytest.raises(OSError, match="offline"):
        socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    with pytest.raises(OSError, match="offline"):
        socket.create_connection(("127.0.0.1", 9))

    assert len(offline) == 2  # every other test of the file fails at teardown if this record is not empty
    offline.clear()
    assert not [name for name in os.environ if re.search(r"KEY|TOKEN|SECRET|PASSWORD|BEARER|CREDENTIAL", name, re.I)]


def test_this_file_imports_the_package_root_pytest_and_the_standard_library_only():
    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    modules, names = [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "no relative import"
            modules.append(node.module)
            names += [alias.name for alias in node.names]

    assert "recall_guard" in modules
    # The package is imported at its root only: no recall_guard.harness, .core or .mia, here or by attribute.
    assert [module for module in modules if module.startswith("recall_guard.")] == []
    allowed = set(sys.stdlib_module_names) | {"pytest", "recall_guard"}
    assert [module for module in modules if module.split(".")[0] not in allowed] == []
    assert [name for name in names if name.startswith("_")] == []
    reached = [node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
               and isinstance(node.value, ast.Name) and node.value.id == "recall_guard"]
    assert reached == []
    private = [node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
               and node.attr.startswith("_") and not node.attr.endswith("__")]
    assert private == []
