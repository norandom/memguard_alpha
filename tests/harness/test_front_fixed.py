"""The fixed-reply client, the fixed evidence host and the worked example (task 7.2).

Covers Requirements 11.10 and 11.11 of the qualified-model-catalogue spec. The
catalogue is the fixture catalogue of tests/harness/test_selection.py and the
calibrator the double of tests/harness/test_scorer_client.py. No test sends a
request; the two runs through ``ask`` make ``socket.socket`` raise.
"""

from __future__ import annotations

import dataclasses
import json
import re
import socket
from datetime import date

import pytest

import recall_guard
from recall_guard.core.catalogue import Mode, Status, load_catalogue
from recall_guard.core.completion import CompletionClient, ModelConfig, ModelResponse, TokenAlternative, TokenEvidence
from recall_guard.core.full_distribution import FullDistributionEvidence
from recall_guard.core.request_control import RequestError
from recall_guard.dataset.macro_regime import ASSETS, MACRO_AXES, MacroRow, regime_definition, render_anonymised
from recall_guard.harness import front as front_module
from recall_guard.harness.front import (
    LIMIT_CALIBRATOR_MISMATCH,
    LIMIT_NO_CALIBRATOR,
    WORKED_EXAMPLE_REPLY,
    FixedEvidence,
    FixedReplies,
    FixedReply,
    ask,
    fixed_reply,
    worked_example,
)
from recall_guard.mia.estimate_source import EstimateSource
from tests.dataset.test_macro_regime import ROW as TEST_ROW
from tests.harness.test_guarded_decision import DIGEST, model_config
from tests.harness.test_scorer_client import Client, calibrate
from tests.harness.test_selection import FIXTURE, LLAMACPP, SERIES_FINGERPRINT, bound, fixture_data

#: Three alternatives per token, as the calibration double's settings ask for (top_logprobs 3).
TOKENS = [("x", [("x", -0.2), ("y", -1.2), ("z", -2.1)])] * 5
MONTHS = ("january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
          "november", "december", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec")


@pytest.fixture(scope="module")
def catalogue():
    return load_catalogue(FIXTURE)


@pytest.fixture(scope="module")
def scorer():
    return calibrate(Client(model_config()))


@pytest.fixture(scope="module")
def series(scorer):
    """The top-k calibrator the fixture records for llama.cpp logprobs."""
    return bound(scorer, EstimateSource.TOP_K, SERIES_FINGERPRINT)


@pytest.fixture
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a socket was opened")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


def evidence(base: float = -0.3) -> FullDistributionEvidence:
    positions = range(3)
    return FullDistributionEvidence(
        model_digest=DIGEST, n_vocab=50, response_token_ids=tuple(positions),
        logprob=tuple(base - 0.01 * i for i in positions), entropy=(1.0,) * 3, rank=(1,) * 3,
        top_logprobs=tuple(((i, base - 0.01 * i), (40, base - 1.0)) for i in positions),
    )


# -- FixedReplies -------------------------------------------------------------------


def test_replies_come_back_in_order_with_their_evidence():
    with_evidence = fixed_reply("second", TOKENS)
    record = FixedReply("third", finish_reason="stop", usage={"completion_tokens": 7})
    client = FixedReplies(["first", with_evidence, record])

    responses = [client.generate(f"prompt {i}") for i in range(3)]

    assert all(isinstance(response, ModelResponse) for response in responses)
    assert [response.content for response in responses] == ["first", "second", "third"]
    assert responses[0].token_evidence is None
    assert responses[1].token_evidence == with_evidence.token_evidence
    assert responses[1].token_evidence[0] == TokenEvidence("x", -0.2, (
        TokenAlternative("x", -0.2), TokenAlternative("y", -1.2), TokenAlternative("z", -2.1)))
    assert responses[1].mia_unavailable_reason is None
    assert (responses[2].finish_reason, dict(responses[2].usage)) == ("stop", {"completion_tokens": 7})
    assert (responses[0].finish_reason, responses[0].usage) == (None, None)


def test_the_request_count_and_the_prompts_are_recorded_and_read_only():
    client = FixedReplies(["a", "b"])
    assert (client.requests, client.prompts) == (0, ())

    client.generate("one")
    client.generate("two\nwith a second line ", temperature=0.5)

    assert (client.requests, client.prompts) == (2, ("one", "two\nwith a second line "))
    with pytest.raises(AttributeError):
        client.requests = 0
    with pytest.raises(AttributeError):
        client.prompts = ()


def test_the_response_states_the_settings_the_request_asked_for():
    client = FixedReplies(["a"], config=model_config())
    response = client.generate("p", temperature=0.5, seed=3)
    assert dict(response.requested_settings) == {**model_config().parameters, "temperature": 0.5, "seed": 3}
    assert dict(response.confirmed_settings) == {}


def test_running_out_raises_the_request_error_and_says_how_many_were_given():
    client = FixedReplies(["a", "b"])
    client.generate("p")
    client.generate("p")

    with pytest.raises(RequestError, match="used up.*2 were given") as raised:
        client.generate("p")

    assert raised.value.code == "fixed_replies_exhausted"
    assert raised.value.status_code is None and not raised.value.retryable
    assert (client.requests, client.prompts) == (2, ("p", "p"))


def test_cycle_repeats_the_given_replies():
    client = FixedReplies(["a", "b"], cycle=True)
    assert [client.generate("p").content for _ in range(5)] == ["a", "b", "a", "b", "a"]
    assert client.requests == 5


def test_the_default_configuration_is_offline_and_names_no_endpoint():
    client = FixedReplies(["a"])
    assert isinstance(client, CompletionClient)
    assert isinstance(client.config, ModelConfig)
    assert (client.config.protocol.value, client.config.endpoint, client.config.model) == (
        "callback", None, "fixed-replies")
    assert dict(client.config.parameters) == {} and client.config.credential_ref is None


def test_a_given_configuration_is_the_clients():
    configuration = model_config(temperature=0.7)
    assert FixedReplies(["a"], config=configuration).config is configuration


@pytest.mark.parametrize("replies, error", [
    ("one text, not a sequence", TypeError),
    ([b"bytes"], TypeError),
    ([None], TypeError),
    (7, TypeError),
    ([], ValueError),
])
def test_wrong_replies_are_refused(replies, error):
    with pytest.raises(error):
        FixedReplies(replies, cycle=True)


def test_no_replies_without_cycling_is_a_client_that_cannot_be_reached():
    with pytest.raises(RequestError, match="0 were given"):
        FixedReplies([]).generate("p")


def test_wrong_arguments_are_refused():
    with pytest.raises(TypeError, match="config"):
        FixedReplies(["a"], config={"model": "m"})
    with pytest.raises(TypeError, match="prompt"):
        FixedReplies(["a"]).generate(7)


def test_a_reply_record_is_frozen_and_validates_its_evidence():
    record = fixed_reply("text", TOKENS)
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.text = "other"
    with pytest.raises(TypeError):
        FixedReply(7)
    with pytest.raises(TypeError):
        FixedReply("text", token_evidence=("x",))
    assert FixedReply("text", token_evidence=list(record.token_evidence)).token_evidence == record.token_evidence


def test_the_helper_takes_alternatives_as_a_mapping_and_orders_them_by_logprob():
    record = fixed_reply("ab", [("a", {"b": -2.0, "a": -0.1}), ("b", [("b", -0.3)])])
    first, second = record.token_evidence
    assert (first.token, first.logprob) == ("a", -0.1)
    assert [(alt.token, alt.logprob) for alt in first.top_logprobs] == [("a", -0.1), ("b", -2.0)]
    assert (second.token, second.logprob, len(second.top_logprobs)) == ("b", -0.3, 1)
    assert (record.finish_reason, record.usage) == (None, None)
    assert fixed_reply("ab", [("a", {"a": -0.1})], finish_reason="stop").finish_reason == "stop"


def test_the_helper_refuses_a_token_that_is_not_among_its_alternatives():
    with pytest.raises(ValueError, match="'q'.*TokenEvidence"):
        fixed_reply("q", [("q", [("a", -0.1)])])


# -- FixedReplies through the front ------------------------------------------------


def test_the_worked_example_runs_through_the_front_without_a_calibrator_and_without_a_socket(catalogue, no_network):
    prompt, definition = worked_example()
    client = FixedReplies([WORKED_EXAMPLE_REPLY], cycle=True)

    result = ask(prompt, definition, entry=LLAMACPP, mode=Mode.LOGPROBS, client=client, catalogue=catalogue)

    assert result.failure is None
    assert result.answer == json.loads(WORKED_EXAMPLE_REPLY)
    assert (result.p_memorized, result.corrected, result.selected) == (None, None, None)
    assert (result.mode, result.status, result.limits) == (Mode.LOGPROBS, Status.QUALIFIED, (LIMIT_NO_CALIBRATOR,))
    assert client.requests == 32
    assert client.prompts == (prompt,) * 32
    assert all(sent.encode("utf-8") == prompt.encode("utf-8") for sent in client.prompts)


def test_the_worked_example_runs_through_the_front_with_a_calibrator_and_token_evidence(
        catalogue, series, no_network):
    prompt, definition = worked_example()
    client = FixedReplies([fixed_reply(WORKED_EXAMPLE_REPLY, TOKENS)] * 8, config=model_config())

    result = ask(prompt, definition, entry=LLAMACPP, client=client, calibrator=series, catalogue=catalogue,
                 draws=8, min_parsed=8)

    assert result.failure is None and result.limits == ()
    assert result.answer == json.loads(WORKED_EXAMPLE_REPLY)
    assert definition.decide(result.answer) == 1
    assert (result.mode, result.status) == (Mode.LOGPROBS, Status.QUALIFIED)
    assert result.p_memorized is not None and 0.0 <= result.p_memorized <= 1.0
    assert result.corrected is not None and set(result.corrected) == set(MACRO_AXES)
    assert result.selected is not None and all(draw.evidence_ref for draw in result.selected.decision.draws)
    assert client.requests == 8
    assert client.prompts == (prompt,) * 8


def test_a_calibrator_fitted_for_another_model_is_a_limit_not_an_exception(catalogue, series):
    prompt, definition = worked_example()
    other = dataclasses.replace(model_config(), model="another-model")
    client = FixedReplies([fixed_reply(WORKED_EXAMPLE_REPLY, TOKENS)], config=other, cycle=True)

    result = ask(prompt, definition, entry=LLAMACPP, client=client, calibrator=series, catalogue=catalogue,
                 draws=8, min_parsed=8)

    assert result.failure is None and result.selected is None
    assert result.answer == json.loads(WORKED_EXAMPLE_REPLY)
    assert (result.p_memorized, result.corrected) == (None, None)
    assert result.limits == (LIMIT_CALIBRATOR_MISMATCH,)
    assert client.requests == 8


def test_the_default_configuration_does_not_match_a_calibrator_fitted_elsewhere(catalogue, series):
    prompt, definition = worked_example()
    client = FixedReplies([WORKED_EXAMPLE_REPLY], cycle=True)
    result = ask(prompt, definition, entry=LLAMACPP, client=client, calibrator=series, catalogue=catalogue,
                 draws=8, min_parsed=8)
    assert result.failure is None and result.limits == (LIMIT_CALIBRATOR_MISMATCH,)


@pytest.mark.parametrize("calibrated", [True, False])
def test_running_out_fails_at_the_request_step_through_the_front(catalogue, series, calibrated):
    prompt, definition = worked_example()
    client = FixedReplies([fixed_reply(WORKED_EXAMPLE_REPLY, TOKENS)] * 3, config=model_config())

    result = ask(prompt, definition, entry=LLAMACPP, client=client, catalogue=catalogue, draws=8, min_parsed=8,
                 calibrator=series if calibrated else None)

    assert result.answer is None and result.failure is not None
    assert result.failure.step == "request"
    assert "fixed_replies_exhausted=5" in result.failure.detail
    assert client.requests == 3


def test_replies_that_do_not_parse_fail_at_the_reply_step(catalogue):
    prompt, definition = worked_example()
    client = FixedReplies(["I would rather not say."], cycle=True)
    result = ask(prompt, definition, entry=LLAMACPP, client=client, catalogue=catalogue, draws=8, min_parsed=8)
    assert result.failure is not None and result.failure.step == "reply"


# -- FixedEvidence ------------------------------------------------------------------


def test_the_fixed_host_returns_its_evidence_in_order_and_reports_no_credential_problem():
    first, second = evidence(-0.3), evidence(-0.7)
    host = FixedEvidence([first, second])

    assert host.credential_problem() is None
    assert host.evidence("p", "r1") is first
    assert host.evidence("p", "r2", top_k_keep=4) is second
    assert host.requests == 2
    assert host.calls == (("p", "r1"), ("p", "r2"))
    with pytest.raises(RequestError, match="used up.*2 were given") as raised:
        host.evidence("p", "r3")
    assert raised.value.code == "fixed_evidence_exhausted"
    with pytest.raises(AttributeError):
        host.requests = 0


def test_the_fixed_host_serves_a_batch_in_order():
    items = [evidence(-0.1 * i - 0.1) for i in range(3)]
    host = FixedEvidence(items)

    assert host.evidence_batch("p", ["a", "b"], timeout_s=5.0) == (items[0], items[1])
    assert host.calls == (("p", "a"), ("p", "b"))
    with pytest.raises(RequestError):
        host.evidence_batch("p", ["c", "d"])
    assert host.requests == 2  # a batch that cannot be served whole serves nothing


def test_the_fixed_host_cycles_and_refuses_wrong_evidence():
    one = evidence()
    host = FixedEvidence([one], cycle=True)
    assert [host.evidence("p", "r") for _ in range(3)] == [one] * 3
    with pytest.raises(TypeError):
        FixedEvidence(["not evidence"])
    with pytest.raises(ValueError):
        FixedEvidence([], cycle=True)
    with pytest.raises(TypeError):
        host.evidence("p", 7)


def test_the_logits_mode_runs_through_the_front_with_the_fixed_host(catalogue, scorer, no_network):
    prompt, definition = worked_example()
    full = bound(scorer, EstimateSource.FULL_DISTRIBUTION)
    client = FixedReplies([WORKED_EXAMPLE_REPLY] * 8, config=model_config())
    host = FixedEvidence([evidence(-0.3 - 0.1 * i) for i in range(8)])

    result = ask(prompt, definition, entry=LLAMACPP, mode=Mode.LOGITS, client=client, calibrator=full,
                 full_distribution=host, catalogue=catalogue, draws=8, min_parsed=8)

    assert result.failure is None and result.selected is not None
    assert result.answer == json.loads(WORKED_EXAMPLE_REPLY)
    assert result.p_memorized is not None
    assert result.mode is Mode.LOGITS and result.status is catalogue.qualification(LLAMACPP, Mode.LOGITS).status
    assert (client.requests, host.requests) == (8, 8)
    assert host.calls == ((prompt, WORKED_EXAMPLE_REPLY),) * 8


# -- the worked example (11.10) -----------------------------------------------------


def test_the_worked_example_is_the_anonymised_macro_prompt_with_the_macro_definition():
    prompt, definition = worked_example()
    assert definition is regime_definition
    assert [component.name for component in definition.components] == list(MACRO_AXES)
    assert all((component.lower, component.upper) == (-1.0, 1.0) for component in definition.components)
    assert prompt == render_anonymised(front_module.WORKED_EXAMPLE_ROW)
    assert worked_example() == (prompt, definition)
    for line in ("  - baa10y_z: +0.85", "  - cpi_yoy_z: +1.20", "  - t10y2y_z: -0.65"):
        assert line in prompt.splitlines()


def test_the_worked_example_prompt_carries_no_date_year_month_ticker_or_raw_level():
    prompt, _ = worked_example()
    row = front_module.WORKED_EXAMPLE_ROW
    lowered = prompt.lower()

    assert re.search(r"\d{4}", prompt) is None
    assert re.search(r"\d{1,4}[-/]\d{1,2}([-/]\d{1,4})?", prompt) is None
    assert row.decision_date.isoformat() not in prompt
    words = set(re.findall(r"[a-z]+", lowered))
    assert not words & set(MONTHS)
    for _pseudo, _category, real in ASSETS:
        assert real.lower() not in lowered
        for part in re.findall(r"[a-z0-9.]+", real.lower()):
            if len(part) > 2 and part not in {"month", "treasury", "bill"}:
                assert part not in lowered, part
    numbers = set(re.findall(r"\d+(?:\.\d+)?", prompt))
    for key, value in row.raw.items():
        assert f"{value:g}" not in numbers, key
        assert f"{value:.2f}" not in numbers, key
    assert "As-of date" not in prompt and "Real asset identities" not in prompt and "Raw" not in prompt


def test_the_worked_example_row_is_synthetic_and_matches_no_test_row():
    row = front_module.WORKED_EXAMPLE_ROW
    assert isinstance(row, MacroRow)
    assert row.decision_date == date.min  # a placeholder; no month is meant
    assert row.revised == {}
    assert row.zscores != TEST_ROW.zscores and row.raw != TEST_ROW.raw
    assert not set(row.raw.values()) & set(TEST_ROW.raw.values())
    assert not set(row.zscores.values()) & set(TEST_ROW.zscores.values())
    assert "synthetic" in worked_example.__doc__.lower()


def test_no_macro_fixture_file_holds_the_synthetic_row():
    files = [path for path in FIXTURE.parents[1].rglob("*") if path.is_file() and "macro" in path.name.lower()]
    if not files:
        pytest.skip("the repository has no macro fixture file under tests/fixtures")
    row = front_module.WORKED_EXAMPLE_ROW
    values = [f"{value:g}" for value in (*row.raw.values(), *row.zscores.values())]
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        assert not all(value in text for value in values), path


def test_the_example_reply_parses_with_the_definition_and_is_labelled_invented():
    _, definition = worked_example()
    parsed = definition.parse_response(WORKED_EXAMPLE_REPLY)
    assert parsed == json.loads(WORKED_EXAMPLE_REPLY)
    assert list(parsed) == list(MACRO_AXES)
    assert all(-1.0 <= value <= 1.0 for value in parsed.values())
    source = front_module.__file__
    with open(source, encoding="utf-8") as handle:
        text = handle.read()
    comment = text[:text.index("WORKED_EXAMPLE_REPLY = ")].rsplit("\n\n", 1)[-1]
    assert "invented" in comment and "not a model output" in comment


def test_the_definition_is_the_studys():
    execute = pytest.importorskip("scripts.finance_sim.execute")
    _, definition = worked_example()
    assert execute.DEFINITIONS[execute.MACRO] is definition
    assert definition.name == "macro_regime"


# -- exports and what the module touches -------------------------------------------


@pytest.mark.parametrize("name", ["FixedReplies", "FixedEvidence", "FixedReply", "fixed_reply", "worked_example",
                                  "WORKED_EXAMPLE_REPLY"])
def test_the_names_are_exported_from_the_module_and_the_root(name):
    assert name in front_module.__all__
    assert name in recall_guard.__all__
    assert getattr(recall_guard, name) is getattr(front_module, name)


def test_the_fixed_doubles_read_no_file_and_no_environment(monkeypatch, mocker, no_network):
    import builtins
    import os

    opened = mocker.spy(builtins, "open")
    monkeypatch.setattr(os, "environ", {})
    client = FixedReplies([fixed_reply(WORKED_EXAMPLE_REPLY, TOKENS)])
    client.generate(worked_example()[0])
    FixedEvidence([evidence()]).evidence("p", "r")
    assert opened.call_count == 0


def test_the_fixture_catalogue_is_the_one_the_front_tests_use():
    assert fixture_data()["entries"][0]["entry_id"] == LLAMACPP
