"""The front call: one prompt, one decision definition, one catalogue entry (task 7.1).

Covers Requirements 11.1 to 11.9 of the qualified-model-catalogue spec. Every
test loads the fixture catalogue, or a mapping derived from it; none reads the
shipped data file. The client, the full-distribution host and the calibrator
are the doubles of tests/harness/test_guarded_decision.py and
tests/harness/test_selection.py, so no test sends a request.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
from pathlib import Path
from types import SimpleNamespace

import pytest

from recall_guard.core.catalogue import Mode, Status, load_catalogue
from recall_guard.core.decision import adjust
from recall_guard.core.full_distribution import FullDistributionClient
from recall_guard.core.profiles import PreflightError, ProfileError, load_profile
from recall_guard.core.request_control import BudgetExhaustedError, RequestError
from recall_guard.harness import front as front_module
from recall_guard.harness import guarded_decision as guard_module
from recall_guard.harness.front import (
    CHOICE_CALIBRATED,
    CHOICE_CALIBRATED_TIE,
    CHOICE_LOGITS,
    CHOICE_LOGPROBS,
    CHOICE_NAMED,
    CHOICE_QUALIFIED,
    CHOICE_QUALIFIED_TIE,
    LIMIT_CALIBRATOR_MISMATCH,
    LIMIT_CALIBRATOR_NOT_RECORDED,
    LIMIT_DRAWS_MISSING,
    LIMIT_ESTIMATE_UNAVAILABLE,
    LIMIT_NAMED_MODE_NOT_QUALIFIED,
    LIMIT_NO_CALIBRATOR,
    LIMIT_NO_QUALIFIED_MODE,
    LIMIT_WEAK_CALIBRATOR,
    FrontAnswer,
    FrontFailure,
    ask,
    choose_mode,
)
from recall_guard.harness.guarded_decision import GuardConfig, SourceCalibrator, guarded_decide
from recall_guard.harness.selection import SelectionError, guarded_select, select
from recall_guard.mia.estimate_source import EstimateSource
from tests.harness.test_guarded_decision import (
    PROMPT,
    UNIMODAL,
    FakeFullDistribution,
    ScriptedClient,
    definition,
    model_config,
)
from tests.harness.test_scorer_client import Client, calibrate
from tests.harness.test_selection import (
    CLOSED,
    FIXTURE,
    LLAMACPP,
    OGA,
    OGA_FINGERPRINT,
    OLLAMA,
    SERIES_FINGERPRINT,
    bound,
    fixture_data,
)

CONSENSUS = {"equity": 0.2, "gold": 0.5}
#: The variable the llama.cpp profile names for its full-distribution host's bearer.
BEARER = load_profile(LLAMACPP).full_distribution.credential_ref.removeprefix("env:")
UNREADABLE = ["I would rather not say."] * 8


@pytest.fixture(scope="module")
def catalogue():
    return load_catalogue(FIXTURE)


@pytest.fixture(scope="module")
def both_qualified():
    """The llama.cpp entry with both modes qualified."""
    data = fixture_data()
    modes = data["entries"][0]["modes"]
    modes["logits"]["calibrations"] = [copy.deepcopy(modes["logprobs"]["calibrations"][1])]
    return load_catalogue(data)


@pytest.fixture(scope="module")
def none_qualified():
    """The llama.cpp entry offers both modes and qualifies none; the OGA entry offers logits only, not tested."""
    data = fixture_data()
    data["entries"][0]["modes"]["logprobs"]["calibrations"] = []
    data["entries"][2]["modes"]["logits"]["calibrations"] = []
    return load_catalogue(data)


@pytest.fixture(scope="module")
def scorer():
    return calibrate(Client(model_config()))


@pytest.fixture(scope="module")
def series(scorer):
    """The top-k calibrator the fixture records for llama.cpp logprobs."""
    return bound(scorer, EstimateSource.TOP_K, SERIES_FINGERPRINT)


@pytest.fixture(scope="module")
def top_k(scorer):
    return bound(scorer, EstimateSource.TOP_K)


@pytest.fixture(scope="module")
def full(scorer):
    return bound(scorer, EstimateSource.FULL_DISTRIBUTION)


def front(entry, *, contents=None, client=None, **options) -> FrontAnswer:
    """Eight scripted draws through the front, all required to parse unless overridden."""
    if client is None:
        client = ScriptedClient(UNIMODAL if contents is None else contents)
    return ask(PROMPT, definition(), entry=entry, client=client, **{"draws": 8, "min_parsed": 8, **options})


# -- mode choice (11.4) -------------------------------------------------------------


def test_rule_one_chooses_the_qualified_mode(catalogue, full):
    assert choose_mode(catalogue.entry(LLAMACPP), catalogue=catalogue, calibrators={}) == (
        Mode.LOGPROBS, CHOICE_QUALIFIED)
    assert choose_mode(catalogue.entry(OGA), catalogue=catalogue, calibrators={}) == (Mode.LOGITS, CHOICE_QUALIFIED)
    # A calibrator for the other mode does not outrank a qualified mode.
    assert choose_mode(catalogue.entry(LLAMACPP), catalogue=catalogue, calibrators={Mode.LOGITS: full}) == (
        Mode.LOGPROBS, CHOICE_QUALIFIED)


def test_rule_one_tie_prefers_logprobs_and_says_so(both_qualified):
    assert choose_mode(both_qualified.entry(LLAMACPP), catalogue=both_qualified, calibrators={}) == (
        Mode.LOGPROBS, CHOICE_QUALIFIED_TIE)


def test_rule_two_chooses_the_mode_a_calibrator_was_supplied_for(none_qualified, full):
    assert choose_mode(none_qualified.entry(LLAMACPP), catalogue=none_qualified, calibrators={Mode.LOGITS: full}) == (
        Mode.LOGITS, CHOICE_CALIBRATED)


def test_rule_two_tie_prefers_logprobs_and_says_so(none_qualified, top_k, full):
    chosen = choose_mode(none_qualified.entry(LLAMACPP), catalogue=none_qualified,
                         calibrators={Mode.LOGITS: full, Mode.LOGPROBS: top_k})
    assert chosen == (Mode.LOGPROBS, CHOICE_CALIBRATED_TIE)


def test_rule_two_ignores_a_calibrator_for_a_mode_the_entry_does_not_offer(catalogue, full):
    assert choose_mode(catalogue.entry(OLLAMA), catalogue=catalogue, calibrators={Mode.LOGITS: full}) == (
        Mode.LOGPROBS, CHOICE_LOGPROBS)


def test_rule_three_chooses_logprobs_when_it_is_offered(none_qualified):
    assert choose_mode(none_qualified.entry(LLAMACPP), catalogue=none_qualified, calibrators={}) == (
        Mode.LOGPROBS, CHOICE_LOGPROBS)


def test_rule_four_chooses_logits_when_logprobs_is_not_offered(none_qualified):
    assert choose_mode(none_qualified.entry(OGA), catalogue=none_qualified, calibrators={}) == (
        Mode.LOGITS, CHOICE_LOGITS)


def test_choose_mode_refuses_an_entry_that_offers_no_mode(catalogue):
    with pytest.raises(SelectionError, match="offers no mode"):
        choose_mode(catalogue.entry(CLOSED), catalogue=catalogue, calibrators={})


def test_the_mode_choice_phrases_are_fixed_and_distinct():
    phrases = (CHOICE_NAMED, CHOICE_QUALIFIED, CHOICE_QUALIFIED_TIE, CHOICE_CALIBRATED, CHOICE_CALIBRATED_TIE,
               CHOICE_LOGPROBS, CHOICE_LOGITS)
    assert CHOICE_NAMED == "named by the caller"
    assert len(set(phrases)) == len(phrases) and all(isinstance(phrase, str) and phrase for phrase in phrases)
    assert "logprobs preferred" in CHOICE_QUALIFIED_TIE and "logprobs preferred" in CHOICE_CALIBRATED_TIE


def test_ask_states_the_rule_that_chose_the_mode(catalogue, none_qualified, series, full):
    assert front(LLAMACPP, catalogue=catalogue, calibrator=series).mode_choice == CHOICE_QUALIFIED
    chosen = front(LLAMACPP, catalogue=none_qualified, calibrator=full, full_distribution=FakeFullDistribution())
    assert (chosen.mode, chosen.mode_choice) == (Mode.LOGITS, CHOICE_CALIBRATED)
    assert front(LLAMACPP, catalogue=none_qualified).mode_choice == CHOICE_LOGPROBS
    assert front(OGA, catalogue=none_qualified).mode_choice == CHOICE_LOGITS


@pytest.mark.parametrize("named", [Mode.LOGITS, "logits"])
def test_a_named_mode_is_used_and_said_to_be_named(catalogue, full, named):
    result = front(LLAMACPP, catalogue=catalogue, mode=named, calibrator=full,
                   full_distribution=FakeFullDistribution())

    assert (result.mode, result.mode_choice) == (Mode.LOGITS, CHOICE_NAMED)
    assert result.failure is None


# -- with a calibrator: the selection interface's result, untouched (11.2, 11.5) -----


def test_the_calibrator_path_is_guarded_select_field_for_field(catalogue, series, mocker):
    chosen = select(LLAMACPP, Mode.LOGPROBS, catalogue=catalogue)
    config = GuardConfig(definition(), chosen.source, draws=8, min_parsed=8)
    direct = guarded_decide(PROMPT, config, client=ScriptedClient(UNIMODAL), calibrator=series)
    expected = guarded_select(PROMPT, chosen, config.definition, client=ScriptedClient(UNIMODAL), calibrator=series,
                              draws=8, min_parsed=8)
    spy = mocker.spy(front_module, "guarded_select")
    client = ScriptedClient(UNIMODAL)

    result = front(LLAMACPP, catalogue=catalogue, client=client, calibrator=series)

    assert isinstance(result, FrontAnswer)
    assert spy.call_count == 1
    assert spy.call_args.args[0] == PROMPT and spy.call_args.args[1] == chosen
    assert spy.call_args.kwargs == {"client": client, "calibrator": series, "full_distribution": None,
                                    "draws": 8, "min_parsed": 8}
    assert result.selected is spy.spy_return
    assert result.selected == expected
    for name in ("entry_id", "mode", "status", "reason", "answer", "p_memorized", "corrected"):
        assert getattr(result, name) == getattr(expected, name), name
    assert result.answer is result.selected.answer and result.corrected is result.selected.corrected
    assert result.p_memorized is result.selected.p_memorized
    # Bit for bit the guard's estimate.
    assert result.p_memorized == direct.p_memorized and result.p_memorized is not None
    assert result.selected.decision == direct
    assert client.calls == [(PROMPT, {})] * 8


def test_a_qualified_run_carries_the_correction_and_no_limit(catalogue, series):
    result = front(LLAMACPP, catalogue=catalogue, calibrator=series)

    assert (result.entry_id, result.mode, result.status) == (LLAMACPP, Mode.LOGPROBS, Status.QUALIFIED)
    assert result.reason == "all four checks passed"
    assert result.answer == CONSENSUS
    assert result.corrected == adjust(result.answer, result.p_memorized) and result.corrected != result.answer
    assert result.limits == () and result.failure is None


def test_a_qualified_logits_run_carries_the_correction(both_qualified, full):
    recorded = dataclasses.replace(full, fingerprint=SERIES_FINGERPRINT)
    host = FakeFullDistribution()
    result = front(LLAMACPP, catalogue=both_qualified, mode=Mode.LOGITS, calibrator=recorded, full_distribution=host)

    assert (result.mode, result.status, result.limits) == (Mode.LOGITS, Status.QUALIFIED, ())
    assert len(host.calls) == 8
    assert result.corrected == adjust(result.answer, result.p_memorized)


def test_a_not_qualified_mode_still_answers_uncorrected_and_marked(catalogue, full):
    result = front(LLAMACPP, catalogue=catalogue, mode=Mode.LOGITS, calibrator=full,
                   full_distribution=FakeFullDistribution())

    assert result.failure is None
    assert (result.status, result.corrected) == (Status.NOT_QUALIFIED, None)
    assert result.reason.startswith("discrimination: held-out AUC 0.55")
    assert result.answer == CONSENSUS and result.p_memorized == result.selected.decision.p_memorized
    assert result.p_memorized is not None
    # The entry has a qualified mode; the caller named the other one.
    assert result.limits == (LIMIT_NAMED_MODE_NOT_QUALIFIED,)


def test_a_not_tested_mode_still_answers_uncorrected_and_marked(catalogue, top_k):
    result = front(OLLAMA, catalogue=catalogue, calibrator=top_k)

    assert result.failure is None
    assert (result.mode, result.mode_choice) == (Mode.LOGPROBS, CHOICE_CALIBRATED)
    assert (result.status, result.reason, result.corrected) == (Status.NOT_TESTED, "no quality record", None)
    assert result.answer == CONSENSUS and result.p_memorized is not None
    assert result.limits == (LIMIT_NO_QUALIFIED_MODE,)


# -- limits (11.8) ------------------------------------------------------------------


def test_the_limit_phrases_are_fixed():
    assert LIMIT_NO_QUALIFIED_MODE == "no qualified mode for the entry"
    assert LIMIT_NAMED_MODE_NOT_QUALIFIED == "the named mode is not qualified"
    assert LIMIT_NO_CALIBRATOR == "no calibrator supplied"
    assert LIMIT_CALIBRATOR_NOT_RECORDED == "the supplied calibrator is not the recorded one"
    assert LIMIT_ESTIMATE_UNAVAILABLE == "the recall estimate could not be computed"
    assert LIMIT_WEAK_CALIBRATOR == "the adjustment was skipped for a weak calibrator"
    assert LIMIT_CALIBRATOR_MISMATCH == "the supplied calibrator was fitted for another model or settings"
    assert LIMIT_DRAWS_MISSING == "some draws failed or could not be read"
    phrases = [value for name, value in vars(front_module).items() if name.startswith("LIMIT_")]
    assert len(phrases) == len(set(phrases)) == 8


@pytest.mark.parametrize("calibrated", [True, False])
def test_limit_for_draws_that_failed_or_could_not_be_read(catalogue, series, calibrated):
    # Eight draws asked: the first request fails, the last reply does not parse, six are used.
    client = ScriptedClient(UNIMODAL[:7] + ["no"], failures={0: TimeoutError()})
    result = front(LLAMACPP, catalogue=catalogue, client=client, min_parsed=6,
                   calibrator=series if calibrated else None)

    assert result.failure is None and result.answer is not None
    assert len(client.calls) == 8
    assert (result.draws_asked, result.draws_used) == (8, 6)
    assert result.limits == ((LIMIT_DRAWS_MISSING,) if calibrated else (LIMIT_NO_CALIBRATOR, LIMIT_DRAWS_MISSING))
    assert (result.p_memorized is not None) is calibrated


def test_limit_for_missing_draws_after_a_calibrator_mismatch(catalogue, series):
    other = dataclasses.replace(model_config(), model="another-model")
    client = ScriptedClient(UNIMODAL[:7] + ["no"], configuration=other)
    result = front(LLAMACPP, catalogue=catalogue, client=client, calibrator=series, min_parsed=6)

    assert result.limits == (LIMIT_CALIBRATOR_MISMATCH, LIMIT_DRAWS_MISSING)
    assert (result.draws_asked, result.draws_used) == (8, 7)


def test_a_complete_ensemble_reports_its_draws_and_no_missing_draw(catalogue, series):
    for result in (front(LLAMACPP, catalogue=catalogue, calibrator=series), front(LLAMACPP, catalogue=catalogue)):
        assert (result.draws_asked, result.draws_used) == (8, 8)
        assert LIMIT_DRAWS_MISSING not in result.limits


def test_a_failure_reports_no_draw_counts(catalogue):
    for result in (front(CLOSED, catalogue=catalogue), front(LLAMACPP, catalogue=catalogue, contents=UNREADABLE)):
        assert result.failure is not None
        assert (result.draws_asked, result.draws_used) == (None, None)


def test_the_draw_counts_are_the_last_fields_and_have_defaults():
    names = [field.name for field in dataclasses.fields(FrontAnswer)]
    assert names[-2:] == ["draws_asked", "draws_used"]
    assert names[:-2] == ["entry_id", "mode", "mode_choice", "status", "reason", "answer", "p_memorized",
                          "corrected", "limits", "failure", "selected"]
    bare = FrontAnswer(entry_id="e", mode=None, mode_choice="", status=None, reason="", answer=None,
                       p_memorized=None, corrected=None, limits=(), failure=None, selected=None)
    assert (bare.draws_asked, bare.draws_used) == (None, None)


@pytest.mark.parametrize("error", [
    RequestError("HTTP 500", status_code=500, code="host_error"),
    RequestError("the response could not be tokenised", code="tokenisation_failed"),
])
def test_a_logits_host_that_fails_on_evidence_still_answers_without_an_estimate(catalogue, full, error):
    host = FakeFullDistribution(refuse=dict.fromkeys(range(8), error))
    result = front(LLAMACPP, catalogue=catalogue, mode=Mode.LOGITS, calibrator=full, full_distribution=host)

    assert result.failure is None and len(host.calls) == 8
    assert (result.answer, result.p_memorized, result.corrected) == (CONSENSUS, None, None)
    assert result.selected.decision.status == "estimate_unavailable"
    assert result.limits == (LIMIT_NAMED_MODE_NOT_QUALIFIED, LIMIT_ESTIMATE_UNAVAILABLE)
    assert (result.draws_asked, result.draws_used) == (8, 8)


def test_limit_for_a_calibrator_other_than_the_recorded_one(catalogue, top_k):
    assert top_k.fingerprint != SERIES_FINGERPRINT
    result = front(LLAMACPP, catalogue=catalogue, calibrator=top_k)

    assert (result.status, result.corrected) == (Status.NOT_TESTED, None)
    assert "fingerprint" in result.reason
    assert result.limits == (LIMIT_CALIBRATOR_NOT_RECORDED,)


def test_limit_for_an_estimate_that_could_not_be_computed(catalogue, series):
    result = front(LLAMACPP, catalogue=catalogue, calibrator=series, client=ScriptedClient(UNIMODAL, evidence=False))

    assert result.selected.decision.status == "estimate_unavailable"
    assert (result.status, result.answer, result.p_memorized, result.corrected) == (
        Status.QUALIFIED, CONSENSUS, None, None)
    assert result.limits == (LIMIT_ESTIMATE_UNAVAILABLE,)


def test_limit_for_a_weak_calibrator(catalogue, series):
    weak = dataclasses.replace(series, mcs=dataclasses.replace(series.mcs, is_weak=True))
    result = front(LLAMACPP, catalogue=catalogue, calibrator=weak)

    assert result.selected.decision.status == "adjustment_skipped_weak_calibrator"
    assert result.answer == CONSENSUS and result.p_memorized is not None and result.corrected is None
    assert result.limits == (LIMIT_WEAK_CALIBRATOR,)


def test_limits_are_listed_together_in_a_fixed_order(catalogue):
    assert front(OLLAMA, catalogue=catalogue).limits == (LIMIT_NO_QUALIFIED_MODE, LIMIT_NO_CALIBRATOR)
    assert front(LLAMACPP, catalogue=catalogue).limits == (LIMIT_NO_CALIBRATOR,)
    assert front(LLAMACPP, catalogue=catalogue, mode=Mode.LOGITS).limits == (
        LIMIT_NAMED_MODE_NOT_QUALIFIED, LIMIT_NO_CALIBRATOR)


# -- without a calibrator (11.6) ----------------------------------------------------


def test_without_a_calibrator_the_answer_comes_back_with_the_status_and_no_estimate(catalogue, series, mocker):
    guarded = front(LLAMACPP, catalogue=catalogue, calibrator=series)
    spies = [mocker.spy(SourceCalibrator, "estimate"), mocker.spy(guard_module, "_estimate"),
             mocker.spy(guard_module, "features_from_top_k"), mocker.spy(guard_module, "features_from_full"),
             mocker.spy(front_module, "guarded_select")]
    client = ScriptedClient(UNIMODAL)

    result = front(LLAMACPP, catalogue=catalogue, client=client)

    assert [spy.call_count for spy in spies] == [0] * len(spies)
    assert client.calls == [(PROMPT, {})] * 8
    assert result.failure is None and result.selected is None
    assert result.answer == CONSENSUS == guarded.answer
    assert (result.p_memorized, result.corrected) == (None, None)
    # The status is the catalogue's for the entry and mode, with its reason.
    assert (result.mode, result.status, result.reason) == (Mode.LOGPROBS, Status.QUALIFIED, "all four checks passed")
    assert result.limits == (LIMIT_NO_CALIBRATOR,)


def test_without_a_calibrator_the_draws_and_the_reduction_are_the_guards(catalogue, mocker):
    spy = mocker.spy(front_module, "decide_without_estimate")
    client = ScriptedClient(UNIMODAL)
    result = front(LLAMACPP, catalogue=catalogue, client=client)

    assert spy.call_count == 1
    assert spy.call_args.args[0] == PROMPT
    assert spy.call_args.kwargs == {"client": client, "draws": 8, "min_parsed": 8}
    assert result.answer is spy.spy_return.decision


def private_reaches(source: str) -> list[str]:
    """The private names of other modules or objects that ``source`` imports or reaches through an attribute.

    A name is private when a dotted part of it starts with an underscore and is
    not a dunder name. An object's access to its own private attributes
    (``self._x``, ``cls._x``) is not a reach into another module.
    """
    def private(name: str) -> bool:
        return any(part.startswith("_") and not (part.startswith("__") and part.endswith("__"))
                   for part in name.split("."))

    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or "", *(alias.name for alias in node.names)]
        elif isinstance(node, ast.Attribute) and not (
                isinstance(node.value, ast.Name) and node.value.id in {"self", "cls"}):
            names = [node.attr]
        else:
            continue
        found.extend(name for name in names if private(name))
    return found


def test_the_front_imports_no_private_name():
    source = Path(front_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    assert [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    # No private name of another module is imported, or reached through an attribute.
    assert private_reaches(source) == []
    assert not hasattr(front_module, "GuardIdentity")


@pytest.mark.parametrize("source, refused", [
    ("from recall_guard.harness.guarded_decision import _estimate\n", ["_estimate"]),
    ("from recall_guard.harness._hidden import name\n", ["recall_guard.harness._hidden"]),
    ("import recall_guard.harness._hidden\n", ["recall_guard.harness._hidden"]),
    ("from recall_guard.harness import guarded_decision as guard\nguard._estimate()\n", ["_estimate"]),
    ("def read(other):\n    return other._private\n", ["_private"]),
    ("class A:\n    def read(self, other):\n        return self._own + other._theirs\n", ["_theirs"]),
    ("class A:\n    def read(self):\n        return self.client._fixed\n", ["_fixed"]),
    ("def read(other):\n    return other.__mangled\n", ["__mangled"]),
])
def test_the_private_name_check_refuses_a_private_name_of_another_module_or_object(source, refused):
    assert private_reaches(source) == refused


@pytest.mark.parametrize("source", [
    "from __future__ import annotations\nimport threading\nfrom recall_guard.core.catalogue import Mode\n",
    "class A:\n    def __init__(self):\n        self._own = 1\n    def read(self):\n        return self._own\n",
    "class A:\n    @classmethod\n    def make(cls):\n        return cls._registry\n",
    "class A:\n    def freeze(self):\n        object.__setattr__(self, 'x', 1)\n        return self._fixed.served\n",
])
def test_the_private_name_check_allows_public_names_and_an_objects_own_private_attributes(source):
    assert private_reaches(source) == []


def test_the_fixed_doubles_keep_their_state_in_ordinary_private_attributes():
    for double in (front_module.FixedReplies(["a"]), front_module.FixedEvidence([])):
        names = set(vars(double)) | set(vars(vars(double)["_fixed"]))
        assert not [name for name in names if "__" in name]
        assert {"_fixed", "_items", "_served"} <= names


def test_without_a_calibrator_the_logits_mode_needs_no_full_distribution_host(catalogue):
    result = front(OGA, catalogue=catalogue)

    assert (result.mode, result.status, result.answer) == (Mode.LOGITS, Status.QUALIFIED, CONSENSUS)
    assert result.limits == (LIMIT_NO_CALIBRATOR,)


def test_the_default_ensemble_is_the_guards_32_draws_with_24_parsed(catalogue):
    client = ScriptedClient(UNIMODAL[:24] + ["no"] * 8)
    result = ask(PROMPT, definition(), entry=LLAMACPP, client=client, catalogue=catalogue)

    assert len(client.calls) == 32 and result.answer == CONSENSUS


# -- one calibrator or a mapping ----------------------------------------------------


def test_a_single_calibrator_serves_the_mode_of_its_estimate_source(none_qualified, top_k, full):
    logits = front(LLAMACPP, catalogue=none_qualified, calibrator=full, full_distribution=FakeFullDistribution())
    logprobs = front(LLAMACPP, catalogue=none_qualified, calibrator=top_k)

    assert (logits.mode, logits.selected.decision.identity.calibrator_fingerprint) == (Mode.LOGITS, full.fingerprint)
    assert (logprobs.mode, logprobs.mode_choice) == (Mode.LOGPROBS, CHOICE_CALIBRATED)
    assert logits.p_memorized is not None and logprobs.p_memorized is not None


def test_a_single_calibrator_for_another_mode_than_the_one_used_is_not_applied(catalogue, full):
    result = front(LLAMACPP, catalogue=catalogue, calibrator=full)

    assert (result.mode, result.mode_choice) == (Mode.LOGPROBS, CHOICE_QUALIFIED)
    assert (result.p_memorized, result.selected, result.limits) == (None, None, (LIMIT_NO_CALIBRATOR,))


def test_a_mapping_of_calibrators_serves_each_mode(none_qualified, top_k, full):
    calibrators = {Mode.LOGPROBS: top_k, "logits": full}
    tie = front(LLAMACPP, catalogue=none_qualified, calibrator=calibrators)
    named = front(LLAMACPP, catalogue=none_qualified, calibrator=calibrators, mode=Mode.LOGITS,
                  full_distribution=FakeFullDistribution())

    assert (tie.mode, tie.mode_choice) == (Mode.LOGPROBS, CHOICE_CALIBRATED_TIE)
    assert tie.selected.decision.identity.calibrator_fingerprint == top_k.fingerprint
    assert (named.mode, named.mode_choice) == (Mode.LOGITS, CHOICE_NAMED)
    assert named.selected.decision.identity.source.source is EstimateSource.FULL_DISTRIBUTION


# -- without a client (11.7) --------------------------------------------------------


def test_without_a_client_the_model_is_reached_through_the_entrys_profile(catalogue, series, mocker):
    client = ScriptedClient(UNIMODAL)
    connect = mocker.patch.object(front_module, "connect", return_value=client)
    environment = {"RECALL_GUARD_FIXTURE_KEY": "s3cr3t-value", "LLAMACPP_ENDPOINT": "http://127.0.0.1:9/v1"}

    result = ask(PROMPT, definition(), entry=LLAMACPP, catalogue=catalogue, calibrator=series, draws=8, min_parsed=8,
                 environment=environment)

    connect.assert_called_once()
    profile, budget = connect.call_args.args
    assert profile.profile_id == catalogue.entry(LLAMACPP).profile_id
    # Room for every draw with the retries the profile allows; no more.
    assert budget.limit == 8 * (profile.model_config.max_retries + 1)
    assert connect.call_args.kwargs["environment"] is environment
    assert len(client.calls) == 8
    assert result.failure is None and result.answer == CONSENSUS and result.status is Status.QUALIFIED
    assert "s3cr3t-value" not in repr(result) and "127.0.0.1:9" not in repr(result)


def test_without_a_client_the_process_environment_is_the_default(catalogue, mocker):
    connect = mocker.patch.object(front_module, "connect", return_value=ScriptedClient(UNIMODAL))

    result = ask(PROMPT, definition(), entry=LLAMACPP, catalogue=catalogue, draws=8, min_parsed=8)

    assert connect.call_args.kwargs["environment"] is None  # connect() reads os.environ itself
    assert result.answer == CONSENSUS


def test_the_logits_host_is_built_from_the_profile_when_none_is_passed(catalogue, full, mocker):
    host = FakeFullDistribution()
    build = mocker.patch.object(front_module, "full_distribution_client", return_value=host)
    environment = {"X": "y", BEARER: "host-bearer"}

    result = front(LLAMACPP, catalogue=catalogue, mode=Mode.LOGITS, calibrator=full, environment=environment)

    build.assert_called_once()
    assert build.call_args.args[0].profile_id == LLAMACPP and build.call_args.kwargs == {"environment": environment}
    assert len(host.calls) == 8 and result.p_memorized is not None


# -- failures (11.9) ----------------------------------------------------------------


def failed(result: FrontAnswer, step: str) -> FrontFailure:
    """Assert that ``result`` is a failure at ``step`` that carries no partial answer; return the failure."""
    assert isinstance(result.failure, FrontFailure)
    assert result.failure.step == step
    assert result.failure.detail and result.failure.remedy
    assert (result.answer, result.p_memorized, result.corrected, result.selected) == (None, None, None, None)
    assert result.limits == ()
    return result.failure


def test_an_unknown_entry_fails_at_the_entry_step_and_names_the_known_ones(catalogue):
    result = front("no-such-entry", catalogue=catalogue)

    failure = failed(result, "entry")
    assert (result.entry_id, result.mode, result.status) == ("no-such-entry", None, None)
    assert "'no-such-entry'" in failure.detail
    assert all(entry.entry_id in failure.remedy for entry in catalogue.entries)


def test_an_entry_without_any_mode_fails_at_the_mode_step_and_lists_the_modes(catalogue):
    result = front(CLOSED, catalogue=catalogue)

    failure = failed(result, "mode")
    assert (result.entry_id, result.mode, result.status) == (CLOSED, None, None)
    assert "logprobs is not_tested (not offered)" in failure.remedy
    assert "logits is not_tested (not offered)" in failure.remedy


@pytest.mark.parametrize("named", [Mode.LOGITS, "logits", "sampling"])
def test_a_named_mode_the_entry_does_not_offer_fails_at_the_mode_step(catalogue, named):
    client = ScriptedClient(UNIMODAL)
    result = front(OLLAMA, catalogue=catalogue, mode=named, client=client)

    failure = failed(result, "mode")
    assert result.mode_choice == CHOICE_NAMED
    assert "logprobs is not_tested;" in failure.remedy and "logits is not_tested (not offered)" in failure.remedy
    assert client.calls == []


def test_a_calibrated_run_of_an_entry_without_a_profile_runs_with_the_k_the_caller_passes(catalogue, scorer):
    calibrator = bound(scorer, EstimateSource.FULL_DISTRIBUTION, OGA_FINGERPRINT)
    host = FakeFullDistribution()
    result = front(OGA, catalogue=catalogue, calibrator=calibrator, full_distribution=host, k=4)

    assert result.failure is None
    assert (result.mode, result.status, result.answer) == (Mode.LOGITS, Status.QUALIFIED, CONSENSUS)
    assert result.selected.decision.identity.source.k == 4
    assert result.p_memorized is not None and result.corrected == adjust(CONSENSUS, result.p_memorized)
    assert result.limits == () and len(host.calls) == 8


def test_a_calibrated_run_of_an_entry_without_a_profile_fails_at_the_mode_step_for_the_missing_k(catalogue, scorer):
    calibrator = bound(scorer, EstimateSource.FULL_DISTRIBUTION, OGA_FINGERPRINT)
    client = ScriptedClient(UNIMODAL)
    result = front(OGA, catalogue=catalogue, calibrator=calibrator, client=client,
                   full_distribution=FakeFullDistribution())

    failure = failed(result, "mode")
    assert (result.mode, result.status) == (Mode.LOGITS, Status.QUALIFIED)
    assert "pass k" in failure.detail
    assert "pass k to ask" in failure.remedy and "alternatives per token" in failure.remedy
    assert "guarded_select" not in failure.remedy
    assert client.calls == []


def test_a_k_that_contradicts_the_profile_fails_at_the_mode_step(catalogue, series):
    client = ScriptedClient(UNIMODAL)
    result = front(LLAMACPP, catalogue=catalogue, calibrator=series, client=client, k=20)

    failure = failed(result, "mode")
    assert (result.mode, result.status) == (Mode.LOGPROBS, Status.QUALIFIED)
    assert "k=20 differs from the 5 alternatives per token" in failure.detail
    assert "leave k out" in failure.remedy
    assert client.calls == []


def test_a_k_that_agrees_with_the_profile_is_accepted(catalogue, series):
    assert front(LLAMACPP, catalogue=catalogue, calibrator=series, k=5).corrected is not None


def test_k_is_not_read_without_a_calibrator_for_the_mode(catalogue):
    assert front(OGA, catalogue=catalogue).answer == CONSENSUS
    assert front(LLAMACPP, catalogue=catalogue, k=20).limits == (LIMIT_NO_CALIBRATOR,)


def test_no_client_and_no_profile_fails_at_the_client_step(catalogue):
    result = ask(PROMPT, definition(), entry=OGA, catalogue=catalogue)

    failure = failed(result, "client")
    assert (result.mode, result.status) == (Mode.LOGITS, Status.QUALIFIED)
    assert "client=" in failure.remedy


@pytest.mark.parametrize("error", [
    ProfileError("Profile 'llamacpp-gemma3-4b' needs credential LLAMACPP_KEY in the environment"),
    PreflightError(SimpleNamespace(recommendation="Free 3 GiB of memory, or use a smaller model.", reason=None)),
])
def test_a_profile_that_cannot_connect_fails_at_the_client_step(catalogue, mocker, error):
    mocker.patch.object(front_module, "connect", side_effect=error)

    result = ask(PROMPT, definition(), entry=LLAMACPP, catalogue=catalogue)

    failure = failed(result, "client")
    assert failure.detail == str(error)
    assert "client=" in failure.remedy
    assert (result.mode, result.status) == (Mode.LOGPROBS, Status.QUALIFIED)


def test_a_logits_host_that_cannot_be_built_fails_at_the_client_step(catalogue, full, mocker):
    mocker.patch.object(front_module, "full_distribution_client", side_effect=ProfileError("no such source"))
    client = ScriptedClient(UNIMODAL)

    result = front(LLAMACPP, catalogue=catalogue, mode=Mode.LOGITS, calibrator=full, client=client)

    assert "full_distribution=" in failed(result, "client").remedy
    assert client.calls == []


class ReferencedHost(FakeFullDistribution):
    """A host double that names its bearer publicly, as FullDistributionClient does."""

    credential_ref = "env:FRONT_TEST_HOST_BEARER"


@pytest.mark.parametrize("environment", [{}, {BEARER: ""}, {BEARER: "   "}])
def test_an_unset_bearer_of_the_profiles_logits_host_fails_at_the_client_step_before_any_request(
        catalogue, full, environment):
    client = ScriptedClient(UNIMODAL)

    result = front(LLAMACPP, catalogue=catalogue, mode=Mode.LOGITS, calibrator=full, client=client,
                   environment=environment)

    failure = failed(result, "client")
    assert BEARER in failure.detail and "not set" in failure.detail
    assert BEARER in failure.remedy and "full_distribution=" in failure.remedy
    assert client.calls == []
    assert (result.mode, result.status) == (Mode.LOGITS, Status.NOT_QUALIFIED)


def test_a_malformed_bearer_of_the_logits_host_fails_at_the_client_step_and_is_not_echoed(catalogue, full):
    client = ScriptedClient(UNIMODAL)

    result = front(LLAMACPP, catalogue=catalogue, mode=Mode.LOGITS, calibrator=full, client=client,
                   environment={BEARER: "s3cr3t value"})

    assert BEARER in failed(result, "client").detail
    assert "s3cr3t" not in repr(result)
    assert client.calls == []


@pytest.fixture
def host_requests(mocker):
    """Answer the evidence requests of every real FullDistributionClient in process; ``calls`` records them."""
    fake = FakeFullDistribution()
    mocker.patch.object(
        FullDistributionClient, "evidence", autospec=True,
        side_effect=lambda self, prompt_text, response_text, *, top_k_keep=100: fake.evidence(
            prompt_text, response_text, top_k_keep=top_k_keep))
    return fake


def test_the_bearer_of_the_profiles_logits_host_is_read_from_the_process_environment_by_default(
        catalogue, full, host_requests, monkeypatch):
    client = ScriptedClient(UNIMODAL)
    monkeypatch.delenv(BEARER, raising=False)

    missing = front(LLAMACPP, catalogue=catalogue, mode=Mode.LOGITS, calibrator=full, client=client)

    assert BEARER in failed(missing, "client").detail
    assert client.calls == [] and host_requests.calls == []

    monkeypatch.setenv(BEARER, "host-bearer")
    result = front(LLAMACPP, catalogue=catalogue, mode=Mode.LOGITS, calibrator=full, client=client)

    assert result.failure is None and result.p_memorized is not None
    assert "host-bearer" not in repr(result)
    assert len(client.calls) == 8 and len(host_requests.calls) == 8


def test_the_profiles_logits_host_with_its_bearer_in_the_passed_environment_proceeds(
        catalogue, full, host_requests, monkeypatch):
    monkeypatch.delenv(BEARER, raising=False)
    client = ScriptedClient(UNIMODAL)

    result = front(LLAMACPP, catalogue=catalogue, mode=Mode.LOGITS, calibrator=full, client=client,
                   environment={BEARER: "host-bearer"})

    assert result.failure is None and result.p_memorized is not None
    assert len(client.calls) == 8 and len(host_requests.calls) == 8


def passed_host(environment) -> FullDistributionClient:
    return FullDistributionClient("http://127.0.0.1:9", model_digest="sha256:front-test",
                                  credential_ref="env:FRONT_TEST_HOST_BEARER", environment=environment)


def test_a_passed_logits_host_is_checked_against_its_own_environment_not_the_fronts(
        catalogue, full, host_requests, monkeypatch):
    monkeypatch.delenv("FRONT_TEST_HOST_BEARER", raising=False)
    client = ScriptedClient(UNIMODAL)
    host = passed_host({"FRONT_TEST_HOST_BEARER": "tok"})

    # No environment= and the variable is absent from the process environment: the client's own lookup succeeds.
    results = [front(LLAMACPP, catalogue=catalogue, mode=Mode.LOGITS, calibrator=full, client=client,
                     full_distribution=host, **options) for options in ({}, {"environment": {}})]

    for result in results:
        assert result.failure is None and result.answer == CONSENSUS and result.p_memorized is not None
    assert len(client.calls) == 16 and len(host_requests.calls) == 16


@pytest.mark.parametrize("own", [{}, {"FRONT_TEST_HOST_BEARER": ""}, {"FRONT_TEST_HOST_BEARER": "s3cr3t value"}])
def test_a_passed_logits_host_whose_own_environment_lacks_the_bearer_fails_at_the_client_step_before_any_request(
        catalogue, full, host_requests, own):
    client = ScriptedClient(UNIMODAL)

    # The front's environment has the variable; the client does not read it there.
    result = front(LLAMACPP, catalogue=catalogue, mode=Mode.LOGITS, calibrator=full, client=client,
                   full_distribution=passed_host(own), environment={"FRONT_TEST_HOST_BEARER": "tok"})

    failure = failed(result, "client")
    assert "FRONT_TEST_HOST_BEARER" in failure.detail and "env:" not in failure.detail
    assert "FRONT_TEST_HOST_BEARER" in failure.remedy and "full_distribution=" in failure.remedy
    assert "s3cr3t" not in repr(result)
    assert client.calls == [] and host_requests.calls == []


def test_a_passed_logits_host_without_the_credential_check_is_not_pre_checked(catalogue, full, monkeypatch):
    monkeypatch.delenv("FRONT_TEST_HOST_BEARER", raising=False)
    host = ReferencedHost()  # names a credential_ref and has no credential_problem method

    result = front(LLAMACPP, catalogue=catalogue, mode=Mode.LOGITS, calibrator=full, full_distribution=host,
                   environment={})

    assert result.failure is None and result.p_memorized is not None and len(host.calls) == 8


def test_a_malformed_generation_credential_fails_at_the_client_step_and_is_not_echoed(catalogue, mocker):
    real = front_module.connect
    # The real connect and client constructor, without the host preflight (free memory, tokenizer files).
    mocker.patch.object(front_module, "connect",
                        side_effect=lambda *args, **kwargs: real(*args, enforce_preflight=False, **kwargs))
    name = load_profile(LLAMACPP).model_config.credential_ref.removeprefix("env:")

    result = ask(PROMPT, definition(), entry=LLAMACPP, catalogue=catalogue, environment={name: "s3cr3t value"})

    failure = failed(result, "client")
    assert name in failure.detail and name in failure.remedy and "client=" in failure.remedy
    assert "s3cr3t" not in repr(result)


def test_another_value_error_from_connect_is_not_turned_into_a_failure(catalogue, mocker):
    mocker.patch.object(front_module, "connect", side_effect=ValueError("something else"))

    with pytest.raises(ValueError, match="something else"):
        ask(PROMPT, definition(), entry=LLAMACPP, catalogue=catalogue)


def test_the_logprobs_mode_does_not_need_the_logits_hosts_bearer(catalogue, series):
    result = front(LLAMACPP, catalogue=catalogue, calibrator=series, environment={})

    assert result.failure is None
    assert (result.mode, result.answer) == (Mode.LOGPROBS, CONSENSUS) and result.p_memorized is not None


def test_the_logits_mode_without_a_calibrator_does_not_need_the_logits_hosts_bearer(catalogue, full):
    plain = front(LLAMACPP, catalogue=catalogue, mode=Mode.LOGITS, environment={})
    # A calibrator for the other mode only: the logits run is still the one without an estimate.
    other = front(LLAMACPP, catalogue=catalogue, mode=Mode.LOGITS, calibrator={Mode.LOGPROBS: full},
                  full_distribution=ReferencedHost(), environment={})

    for result in (plain, other):
        assert result.failure is None and result.answer == CONSENSUS and result.p_memorized is None
        assert LIMIT_NO_CALIBRATOR in result.limits


@pytest.mark.parametrize("error", [
    RequestError("HTTP 401", status_code=401, code="auth_failed"),
    BudgetExhaustedError("budget exhausted"),
])
@pytest.mark.parametrize("calibrated", [True, False])
def test_a_rejected_credential_or_an_exhausted_budget_fails_at_the_request_step(catalogue, series, error, calibrated):
    client = ScriptedClient(UNIMODAL, failures={2: error})
    result = front(LLAMACPP, catalogue=catalogue, client=client, calibrator=series if calibrated else None)

    failure = failed(result, "request")
    assert len(client.calls) == 3
    assert type(error).__name__ in failure.detail
    assert (result.mode, result.status) == (Mode.LOGPROBS, Status.QUALIFIED)


@pytest.mark.parametrize("calibrated", [True, False])
def test_a_model_that_cannot_be_reached_fails_at_the_request_step(catalogue, series, calibrated):
    client = ScriptedClient(UNIMODAL, failures={index: ConnectionError("refused") for index in range(8)})
    result = front(LLAMACPP, catalogue=catalogue, client=client, calibrator=series if calibrated else None)

    failure = failed(result, "request")
    assert "0 of 8 draws parsed" in failure.detail and "request_error:" in failure.detail
    assert "host" in failure.remedy


@pytest.mark.parametrize("calibrated", [True, False])
def test_replies_that_do_not_read_into_the_definition_fail_at_the_reply_step(catalogue, series, calibrated):
    result = front(LLAMACPP, catalogue=catalogue, contents=UNREADABLE, calibrator=series if calibrated else None)

    failure = failed(result, "reply")
    assert "0 of 8 draws parsed" in failure.detail and "parse_error=8" in failure.detail
    assert "definition" in failure.remedy
    assert (result.entry_id, result.mode, result.status) == (LLAMACPP, Mode.LOGPROBS, Status.QUALIFIED)
    assert result.mode_choice == CHOICE_QUALIFIED


def test_too_few_parsed_draws_fail_at_the_reply_step_when_enough_replies_arrived(catalogue):
    contents = UNIMODAL[:5] + ["no"] * 3
    result = front(LLAMACPP, catalogue=catalogue, contents=contents, min_parsed=6,
                   client=ScriptedClient(contents, failures={7: TimeoutError()}))

    assert "5 of 8 draws parsed" in failed(result, "reply").detail


def test_too_few_parsed_draws_fail_at_the_request_step_when_too_few_replies_arrived(catalogue):
    client = ScriptedClient(UNIMODAL, failures={index: TimeoutError() for index in range(3)})
    result = front(LLAMACPP, catalogue=catalogue, client=client, min_parsed=6)

    assert "5 of 8 draws parsed" in failed(result, "request").detail


def test_the_failure_steps_are_the_five_of_the_design():
    assert front_module.FAILURE_STEPS == ("entry", "mode", "client", "request", "reply")
    with pytest.raises(ValueError, match="step"):
        FrontFailure("network", "detail", "remedy")


# -- the prompt is the caller's (11.3) and the result is immutable -------------------


def test_any_prompt_is_passed_to_the_model_unchanged(catalogue, series):
    prompt = "  ¿Cuál es la asignación?\n\n{not a template} %s  "
    client = ScriptedClient(UNIMODAL)
    result = ask(prompt, definition(), entry=LLAMACPP, client=client, catalogue=catalogue, calibrator=series,
                 draws=8, min_parsed=8)

    assert [sent for sent, _parameters in client.calls] == [prompt] * 8
    assert result.failure is None


def test_results_are_immutable(catalogue, series):
    result = front(LLAMACPP, catalogue=catalogue, calibrator=series)
    failure = front(CLOSED, catalogue=catalogue).failure

    with pytest.raises(dataclasses.FrozenInstanceError):
        result.status = Status.NOT_TESTED  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        failure.step = "entry"  # type: ignore[misc]
    with pytest.raises(TypeError):
        result.answer["equity"] = 1.0  # type: ignore[index]
    assert isinstance(result.limits, tuple)


def test_the_shipped_catalogue_is_loaded_only_when_none_is_passed(catalogue, mocker):
    load = mocker.patch.object(front_module, "load_catalogue", return_value=catalogue)

    front(LLAMACPP, catalogue=catalogue)
    assert load.call_count == 0

    assert front(LLAMACPP).answer == CONSENSUS
    load.assert_called_once_with()


# -- programming errors still raise --------------------------------------------------


@pytest.mark.parametrize("overrides, error", [
    ({"prompt": 3}, TypeError),
    ({"definition": "toy_allocation"}, TypeError),
    ({"entry": 5}, TypeError),
    ({"mode": 1}, TypeError),
    ({"calibrator": "sha256:fixture-series"}, TypeError),
    ({"calibrator": {Mode.LOGPROBS: "sha256:fixture-series"}}, TypeError),
    ({"calibrator": {"sampling": None}}, ValueError),
    ({"catalogue": {"entries": []}}, TypeError),
    ({"environment": "PATH"}, TypeError),
    ({"draws": "8"}, TypeError),
    ({"draws": 8, "min_parsed": 9}, ValueError),
])
def test_wrong_arguments_raise_before_any_request(catalogue, overrides, error):
    client = ScriptedClient(UNIMODAL)
    arguments = {"prompt": PROMPT, "definition": definition(), "entry": LLAMACPP, "client": client,
                 "catalogue": catalogue, **overrides}
    prompt, chosen_definition = arguments.pop("prompt"), arguments.pop("definition")

    with pytest.raises(error):
        ask(prompt, chosen_definition, **arguments)
    assert client.calls == []


# -- a calibrator fitted for another model or settings ------------------------------


def test_a_calibrator_fitted_for_other_settings_still_answers_without_an_estimate(catalogue, series, mocker):
    reference = front(LLAMACPP, catalogue=catalogue)
    spies = [mocker.spy(SourceCalibrator, "estimate"), mocker.spy(guard_module, "_estimate")]
    other = ScriptedClient(UNIMODAL, configuration=model_config(temperature=0.7))

    result = front(LLAMACPP, catalogue=catalogue, client=other, calibrator=series)

    # The mismatch is found before any request, so the model is asked once: eight requests, not sixteen.
    assert other.calls == [(PROMPT, {})] * 8
    assert [spy.call_count for spy in spies] == [0, 0]
    assert result.failure is None and result.selected is None
    assert result.answer == CONSENSUS == reference.answer
    assert (result.p_memorized, result.corrected) == (None, None)
    assert (result.mode, result.mode_choice) == (Mode.LOGPROBS, CHOICE_QUALIFIED)
    assert (result.status, result.reason) == (Status.QUALIFIED, "all four checks passed")
    assert result.limits == (LIMIT_CALIBRATOR_MISMATCH,)


def test_a_calibrator_fitted_for_another_model_still_answers_in_the_logits_mode(catalogue, full):
    other = ScriptedClient(UNIMODAL, configuration=dataclasses.replace(model_config(), model="another-model"))
    host = FakeFullDistribution()

    result = front(LLAMACPP, catalogue=catalogue, mode=Mode.LOGITS, client=other, calibrator=full,
                   full_distribution=host)

    assert len(other.calls) == 8 and host.calls == []
    assert (result.answer, result.p_memorized, result.corrected, result.selected) == (CONSENSUS, None, None, None)
    assert result.status is catalogue.qualification(LLAMACPP, Mode.LOGITS).status
    assert result.limits == (LIMIT_NAMED_MODE_NOT_QUALIFIED, LIMIT_CALIBRATOR_MISMATCH)


def test_the_mismatch_limit_is_distinct_from_a_calibrator_that_is_not_the_recorded_one(catalogue, top_k):
    recorded_elsewhere = front(LLAMACPP, catalogue=catalogue, calibrator=top_k)
    mismatched = front(LLAMACPP, catalogue=catalogue, calibrator=top_k,
                       client=ScriptedClient(UNIMODAL, configuration=model_config(temperature=0.7)))

    assert recorded_elsewhere.limits == (LIMIT_CALIBRATOR_NOT_RECORDED,)
    assert recorded_elsewhere.p_memorized is not None and recorded_elsewhere.selected is not None
    assert mismatched.limits == (LIMIT_CALIBRATOR_MISMATCH,)
    assert mismatched.p_memorized is None and mismatched.selected is None


def test_a_mismatched_calibrator_with_unreadable_replies_fails_at_the_reply_step(catalogue, series):
    other = ScriptedClient(UNREADABLE, configuration=model_config(temperature=0.7))
    result = front(LLAMACPP, catalogue=catalogue, client=other, calibrator=series)

    assert "0 of 8 draws parsed" in failed(result, "reply").detail
    assert len(other.calls) == 8
