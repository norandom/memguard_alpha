"""The front against a declared model route, opt-in (Requirement 11.12).

The live test runs the worked example through ``recall_guard.ask`` against the
route of a catalogue entry and records the entry, the mode and the status of
the result. It is skipped unless ``RECALL_GUARD_LIVE_ENTRY`` names a catalogue
entry, by which a person declares that entry's model route available. The test
starts, stops and configures no model host: it sends the requests of one small
ensemble to a route a person has brought up. It asserts nothing about model
quality, and neither the record nor the test's output holds a reply text or a
credential.

Variables:

* ``RECALL_GUARD_LIVE_ENTRY``: the id of the catalogue entry to ask through,
  for example ``llamacpp-gemma3-4b``. Unset or empty: the live test is skipped.
* ``RECALL_GUARD_LIVE_DRAWS``: the size of the ensemble, default 8, kept small
  for a local host; three quarters of the draws must read into the definition.
* ``RECALL_GUARD_LIVE_RECORD``: a path; when set, the outcome is written there
  as one JSON object. It is always attached to the test's report.

The front builds the client from the entry's provider profile and the process
environment, so the variables that profile names (its credential and, where
the route is not at the profile's own address, its endpoint) must be set in the
shell the test runs in.

The body of the live test, :func:`run_worked_example`, is the same function the
offline tests below run with fixed replies in place of the host, under the
directory's ``offline`` fixture. This file imports the package root, pytest and
the standard library only.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping
from pathlib import Path

import pytest

from recall_guard import (
    WORKED_EXAMPLE_REPLY,
    FixedReplies,
    FrontAnswer,
    Mode,
    Status,
    ask,
    front,
    load_catalogue,
    worked_example,
)

ENTRY_VARIABLE = "RECALL_GUARD_LIVE_ENTRY"
DRAWS_VARIABLE = "RECALL_GUARD_LIVE_DRAWS"
RECORD_VARIABLE = "RECALL_GUARD_LIVE_RECORD"
#: A small ensemble, to keep the load on a local host low.
DEFAULT_DRAWS = 8
#: The entry of the shipped catalogue the offline exercise asks through.
SHIPPED_ENTRY = "llamacpp-gemma3-4b"
SKIP_MESSAGE = (
    f"set {ENTRY_VARIABLE} to the id of a catalogue entry (for example {SHIPPED_ENTRY}) to run the worked example "
    "against that entry's model route; the route must already be running, because this test starts, stops and "
    "configures no model host"
)
#: The fields of the recorded outcome, in the order they are recorded.
RECORD_FIELDS = ("entry", "mode", "mode_choice", "status", "limits", "draws_asked", "draws_used", "failure")


# -- the body: the worked example through the front ----------------------------------


def live_entry(environment: Mapping[str, str]) -> str:
    """The entry the person declared available; skips the calling test when none is named."""
    entry = environment.get(ENTRY_VARIABLE, "").strip()
    if not entry:
        pytest.skip(SKIP_MESSAGE)
    return entry


def live_draws(environment: Mapping[str, str]) -> tuple[int, int]:
    """The ensemble size and the number of draws that must read: three quarters of it, rounded up."""
    text = environment.get(DRAWS_VARIABLE, "").strip()
    try:
        draws = int(text) if text else DEFAULT_DRAWS
    except ValueError:
        draws = 0
    if draws < 1:
        raise ValueError(f"{DRAWS_VARIABLE} must be a positive whole number of draws; got {text!r}")
    return draws, math.ceil(0.75 * draws)


def run_worked_example(entry: str, client=None, environment: Mapping[str, str] | None = None, *,
                       draws: int = DEFAULT_DRAWS, min_parsed: int | None = None) -> FrontAnswer:
    """Run the worked example through the front for ``entry`` and return what the front returned.

    ``client`` None lets the front build the client from the entry's provider
    profile and ``environment`` (None: the process environment). Checks only
    what holds for any route: an answer with the five axes within their bounds,
    an entry, a mode, a status and limits; or a failure with a step and a
    remedy. Nothing about the quality of the model.
    """
    prompt, definition = worked_example()
    if min_parsed is None:
        min_parsed = math.ceil(0.75 * draws)
    result = ask(prompt, definition, entry=entry, client=client, environment=environment,
                 draws=draws, min_parsed=min_parsed)

    assert isinstance(result, FrontAnswer)
    assert result.entry_id == entry
    assert result.mode_choice in front.MODE_CHOICES
    if result.failure is not None:
        assert result.failure.step in front.FAILURE_STEPS
        assert result.failure.detail.strip() and result.failure.remedy.strip()
        assert (result.answer, result.p_memorized, result.corrected, result.limits) == (None, None, None, ())
        return result
    bounds = {component.name: (component.lower, component.upper) for component in definition.components}
    assert len(bounds) == 5 and set(result.answer) == set(bounds)
    for name, (lower, upper) in bounds.items():
        assert lower <= result.answer[name] <= upper, name
    assert isinstance(result.mode, Mode) and isinstance(result.status, Status)
    assert all(limit in front.LIMITS for limit in result.limits)
    assert result.draws_asked == draws and min_parsed <= result.draws_used <= draws
    return result


def outcome(result: FrontAnswer, draws: int) -> dict:
    """What is recorded of a result: no answer value, no reply text, no failure detail."""
    failure = result.failure
    return {
        "entry": result.entry_id,
        "mode": None if result.mode is None else result.mode.value,
        "mode_choice": result.mode_choice,
        "status": None if result.status is None else result.status.value,
        "limits": list(result.limits),
        "draws_asked": draws,
        "draws_used": result.draws_used,
        "failure": None if failure is None else {"step": failure.step, "remedy": failure.remedy},
    }


def live_run(environment: Mapping[str, str], *, client=None, record_property=None) -> tuple[FrontAnswer, dict]:
    """The live test's steps: read the variables, run the body, record the outcome. Returns result and record."""
    entry = live_entry(environment)
    draws, min_parsed = live_draws(environment)
    result = run_worked_example(entry, client, environment, draws=draws, min_parsed=min_parsed)
    record = outcome(result, draws)
    text = json.dumps(record)
    path = environment.get(RECORD_VARIABLE, "").strip()
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text + "\n", encoding="utf-8")
    if record_property is not None:
        record_property("front_live", record)
    print(f"front live outcome: {text}")
    return result, record


def require_answer(result: FrontAnswer) -> None:
    """Fail the calling test, with the front's step and remedy, when the front reported a failure."""
    if result.failure is not None:
        pytest.fail(f"the front gave no answer for entry {result.entry_id!r}: it failed at step "
                    f"{result.failure.step!r}. What to change: {result.failure.remedy}", pytrace=False)


# -- the live test -------------------------------------------------------------------


def test_the_worked_example_runs_through_the_front_against_the_declared_route(record_property):
    """Skipped unless RECALL_GUARD_LIVE_ENTRY names an entry. An answer with limits passes; a failure fails."""
    result, _record = live_run(os.environ, record_property=record_property)
    require_answer(result)


# -- the same body, offline ----------------------------------------------------------

SECRET = "s3cr3t-Value-0451"


def fixed_host() -> FixedReplies:
    """Fixed replies in place of the model host: one invented reply, repeated for every draw."""
    return FixedReplies([WORKED_EXAMPLE_REPLY], cycle=True)


@pytest.mark.usefixtures("offline")
def test_the_body_runs_offline_with_fixed_replies_for_the_host_and_the_outcome_is_recorded(tmp_path):
    path = tmp_path / "records" / "front-live.json"
    environment = {ENTRY_VARIABLE: f" {SHIPPED_ENTRY} ", RECORD_VARIABLE: str(path), "LOCAL_MODEL_API_KEY": SECRET}
    client, properties = fixed_host(), []

    result, record = live_run(environment, client=client, record_property=lambda *pair: properties.append(pair))

    require_answer(result)  # an answer with limits passes
    expected = load_catalogue().qualification(SHIPPED_ENTRY, result.mode)
    limits = ([] if expected.status is Status.QUALIFIED else [front.LIMIT_NO_QUALIFIED_MODE])
    assert tuple(record) == RECORD_FIELDS
    assert record == {
        "entry": SHIPPED_ENTRY, "mode": result.mode.value, "mode_choice": result.mode_choice,
        "status": expected.status.value, "limits": [*limits, front.LIMIT_NO_CALIBRATOR],
        "draws_asked": DEFAULT_DRAWS, "draws_used": DEFAULT_DRAWS, "failure": None,
    }
    assert record["mode"] in {mode.value for mode in Mode} and record["mode_choice"] in front.MODE_CHOICES
    assert client.requests == DEFAULT_DRAWS and set(client.prompts) == {worked_example()[0]}
    # Written as one JSON object to the named path, and attached to the report.
    written = path.read_text(encoding="utf-8")
    assert json.loads(written) == record and written.count("\n") == 1
    assert properties == [("front_live", record)]
    # No reply text, no answer value and no credential in the record.
    assert WORKED_EXAMPLE_REPLY not in written and "inflation" not in written and SECRET not in written


@pytest.mark.usefixtures("offline")
def test_the_body_checks_the_answer_and_returns_what_the_front_returned():
    result = run_worked_example(SHIPPED_ENTRY, fixed_host(), {})

    assert result.failure is None and result.answer == json.loads(WORKED_EXAMPLE_REPLY)
    assert (result.draws_asked, result.draws_used) == (DEFAULT_DRAWS, DEFAULT_DRAWS)
    # A reply outside a component's bounds does not read into the definition: the body returns the failure.
    outside = FixedReplies([WORKED_EXAMPLE_REPLY.replace("0.4", "4.0")], cycle=True)
    assert run_worked_example(SHIPPED_ENTRY, outside, {}).failure.step == "reply"


@pytest.mark.usefixtures("offline")
def test_the_ensemble_size_comes_from_its_variable_and_three_quarters_must_read(capsys):
    client = fixed_host()
    result, record = live_run({ENTRY_VARIABLE: SHIPPED_ENTRY, DRAWS_VARIABLE: "5"}, client=client)

    assert (client.requests, record["draws_asked"], record["draws_used"]) == (5, 5, 5)
    assert result.failure is None
    assert live_draws({}) == (8, 6) and live_draws({DRAWS_VARIABLE: " 5 "}) == (5, 4)
    assert live_draws({DRAWS_VARIABLE: "1"}) == (1, 1)
    # Without a record path the outcome is still shown in the test's output.
    assert json.loads(capsys.readouterr().out.split("front live outcome: ")[1]) == record
    for wrong in ("0", "-3", "many", "2.5"):
        with pytest.raises(ValueError, match=DRAWS_VARIABLE):
            live_draws({DRAWS_VARIABLE: wrong})


@pytest.mark.usefixtures("offline")
def test_a_host_that_stops_answering_is_recorded_as_a_failure_with_its_step_and_remedy(tmp_path):
    path = tmp_path / "front-live.json"
    client = FixedReplies([WORKED_EXAMPLE_REPLY] * 3)  # three replies for five draws, four of which must read
    result, record = live_run({ENTRY_VARIABLE: SHIPPED_ENTRY, DRAWS_VARIABLE: "5", RECORD_VARIABLE: str(path)},
                              client=client)

    assert result.failure.step == "request"
    assert record["failure"] == {"step": "request", "remedy": result.failure.remedy}
    assert (record["draws_asked"], record["draws_used"], record["limits"]) == (5, None, [])
    assert json.loads(path.read_text(encoding="utf-8")) == record
    # A person who declared the host available is told: the live test fails with the step and the remedy.
    with pytest.raises(pytest.fail.Exception) as failed:
        require_answer(result)
    assert "'request'" in str(failed.value) and result.failure.remedy in str(failed.value)
    assert not failed.value.pytrace


@pytest.mark.usefixtures("offline")
def test_an_unknown_entry_in_the_variable_fails_at_the_fronts_entry_step_not_with_a_traceback():
    client = fixed_host()
    result, record = live_run({ENTRY_VARIABLE: "no-such-entry"}, client=client)

    assert result.failure.step == "entry" and client.requests == 0
    assert (record["entry"], record["mode"], record["status"]) == ("no-such-entry", None, None)
    assert record["failure"]["step"] == "entry" and SHIPPED_ENTRY in record["failure"]["remedy"]
    with pytest.raises(pytest.fail.Exception) as failed:
        require_answer(result)
    assert "'entry'" in str(failed.value) and SHIPPED_ENTRY in str(failed.value)
    assert not failed.value.pytrace


@pytest.mark.parametrize("environment", [{}, {ENTRY_VARIABLE: ""}, {ENTRY_VARIABLE: "   "}])
def test_without_a_declared_entry_the_skip_names_the_variable_and_what_it_must_contain(environment):
    with pytest.raises(pytest.skip.Exception) as skipped:
        live_entry(environment)

    message = str(skipped.value)
    assert message == SKIP_MESSAGE
    assert ENTRY_VARIABLE in message and "catalogue entry" in message and SHIPPED_ENTRY in message
    assert "must already be running" in message and "starts, stops and configures no model host" in message


@pytest.mark.usefixtures("offline")
def test_the_live_test_itself_is_skipped_when_the_variable_is_unset(monkeypatch):
    monkeypatch.delenv(ENTRY_VARIABLE, raising=False)

    with pytest.raises(pytest.skip.Exception, match=ENTRY_VARIABLE):
        test_the_worked_example_runs_through_the_front_against_the_declared_route(lambda *pair: None)
