"""Selection interface: choose a catalogue entry and mode, run the unchanged guard under its status (task 2.2).

Covers Requirements 2.3 to 2.7 of the qualified-model-catalogue spec. Every
test loads the fixture catalogue, or a mapping derived from it; none reads the
shipped data file. The client, the full-distribution host and the calibrator
are the doubles of tests/harness/test_guarded_decision.py, so no test sends a
request.
"""

from __future__ import annotations

import copy
import dataclasses
import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

import recall_guard
from recall_guard.core.catalogue import CatalogueError, Mode, Status, load_catalogue
from recall_guard.core.decision import adjust
from recall_guard.harness import selection as selection_module
from recall_guard.harness.guarded_decision import GuardConfig, SourceCalibrator, guarded_decide
from recall_guard.harness.selection import SelectedDecision, Selection, SelectionError, guarded_select, select
from recall_guard.mia.estimate_source import EstimateSource, SourceLabel
from tests.harness.test_guarded_decision import (
    PROMPT,
    UNIMODAL,
    FakeFullDistribution,
    ScriptedClient,
    definition,
    model_config,
)
from tests.harness.test_scorer_client import Client, calibrate

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "catalogue" / "catalogue_fixture.json"

#: Fixture entries: a profile entry with logprobs qualified and logits not qualified,
#: a profile entry whose only offered mode is not tested, an entry without a profile
#: whose logits mode is qualified, and an entry that offers no mode.
LLAMACPP, OLLAMA, OGA, CLOSED = "llamacpp-gemma3-4b", "ollama-gemma3-4b", "oga-phi4-mini", "closed-route"
SERIES_FINGERPRINT = "sha256:fixture-series"
OGA_FINGERPRINT = "sha256:fixture-oga"

SELECTION_EXPORTS = (
    ("Mode", "recall_guard.core.catalogue"),
    ("Status", "recall_guard.core.catalogue"),
    ("Catalogue", "recall_guard.core.catalogue"),
    ("CatalogueEntry", "recall_guard.core.catalogue"),
    ("ModeRecord", "recall_guard.core.catalogue"),
    ("Qualification", "recall_guard.core.catalogue"),
    ("CatalogueError", "recall_guard.core.catalogue"),
    ("load_catalogue", "recall_guard.core.catalogue"),
    ("qualify", "recall_guard.core.catalogue"),
    ("Selection", "recall_guard.harness.selection"),
    ("SelectedDecision", "recall_guard.harness.selection"),
    ("SelectionError", "recall_guard.harness.selection"),
    ("select", "recall_guard.harness.selection"),
    ("guarded_select", "recall_guard.harness.selection"),
)


def fixture_data() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def catalogue():
    return load_catalogue(FIXTURE)


@pytest.fixture(scope="module")
def scorer():
    return calibrate(Client(model_config()))


def bound(scorer, source: EstimateSource, fingerprint: str | None = None) -> SourceCalibrator:
    """The fake calibration bound to ``source``; ``fingerprint`` replaces the scorer's own."""
    calibrator = SourceCalibrator.from_scorer(scorer, source=source, corpus_sha256=None)
    return calibrator if fingerprint is None else dataclasses.replace(calibrator, fingerprint=fingerprint)


@pytest.fixture(scope="module")
def series_calibrator(scorer):
    """The top-k calibrator whose fingerprint is the one the fixture records for llama.cpp logprobs."""
    return bound(scorer, EstimateSource.TOP_K, SERIES_FINGERPRINT)


def run(selection, calibrator, *, full_distribution=None, contents=None, **options) -> SelectedDecision:
    """Eight scripted draws, all required to parse unless overridden."""
    client = ScriptedClient(UNIMODAL if contents is None else contents)
    return guarded_select(PROMPT, selection, definition(), client=client, calibrator=calibrator,
                          full_distribution=full_distribution, **{"draws": 8, "min_parsed": 8, **options})


# -- select: entry, mode and source (2.3) -------------------------------------------


def test_select_returns_the_entry_mode_qualification_and_top_k_source_from_the_profile(catalogue):
    chosen = select(LLAMACPP, Mode.LOGPROBS, catalogue=catalogue)

    assert isinstance(chosen, Selection)
    assert chosen.entry is catalogue.entry(LLAMACPP)
    assert chosen.mode is Mode.LOGPROBS
    assert chosen.qualification == catalogue.qualification(LLAMACPP, Mode.LOGPROBS)
    assert chosen.qualification.status is Status.QUALIFIED
    # The llama.cpp profile requests five alternatives per token.
    assert chosen.source == SourceLabel(EstimateSource.TOP_K, 5)


def test_select_maps_the_logits_mode_to_the_full_distribution_with_the_profile_kept_count(catalogue):
    chosen = select(LLAMACPP, Mode.LOGITS, catalogue=catalogue)

    assert chosen.source == SourceLabel(EstimateSource.FULL_DISTRIBUTION, 100)
    assert chosen.qualification.status is Status.NOT_QUALIFIED


def test_select_accepts_the_mode_as_text(catalogue):
    assert select(LLAMACPP, "logits", catalogue=catalogue) == select(LLAMACPP, Mode.LOGITS, catalogue=catalogue)


def test_select_takes_k_from_the_caller_for_an_entry_without_a_profile(catalogue):
    chosen = select(OGA, Mode.LOGITS, catalogue=catalogue, k=50)

    assert chosen.source == SourceLabel(EstimateSource.FULL_DISTRIBUTION, 50)
    assert chosen.qualification.status is Status.QUALIFIED


def test_select_requires_k_for_an_entry_without_a_profile(catalogue):
    with pytest.raises(SelectionError, match="k"):
        select(OGA, Mode.LOGITS, catalogue=catalogue)


def test_select_refuses_a_k_that_differs_from_the_profile(catalogue):
    assert select(LLAMACPP, Mode.LOGPROBS, catalogue=catalogue, k=5).source.k == 5
    with pytest.raises(SelectionError, match="5"):
        select(LLAMACPP, Mode.LOGPROBS, catalogue=catalogue, k=20)


@pytest.mark.parametrize("k", [0, -1, True, 2.0])
def test_select_refuses_a_k_that_is_not_a_positive_integer(catalogue, k):
    with pytest.raises(SelectionError, match="positive integer"):
        select(OGA, Mode.LOGITS, catalogue=catalogue, k=k)


# -- select: refusals (2.4) ---------------------------------------------------------


def test_select_refuses_a_mode_the_entry_does_not_offer_and_names_the_offered_ones(catalogue):
    with pytest.raises(SelectionError) as caught:
        select(OLLAMA, Mode.LOGITS, catalogue=catalogue)

    assert str(caught.value) == (
        "Catalogue entry 'ollama-gemma3-4b' does not offer the logits mode "
        "(the route returns at most the top 20 tokens); it offers: logprobs"
    )
    assert isinstance(caught.value, ValueError)


def test_select_refusal_for_an_entry_without_any_mode_says_it_offers_none(catalogue):
    with pytest.raises(SelectionError) as caught:
        select(CLOSED, Mode.LOGPROBS, catalogue=catalogue)

    assert str(caught.value).endswith("it offers: no mode")


def test_select_refuses_an_unknown_mode_name_and_names_the_offered_ones(catalogue):
    with pytest.raises(SelectionError) as caught:
        select(LLAMACPP, "sampling", catalogue=catalogue)

    assert "'sampling'" in str(caught.value) and str(caught.value).endswith("it offers: logprobs, logits")


def test_select_refuses_an_unknown_entry_and_names_the_known_ones(catalogue):
    with pytest.raises(SelectionError) as caught:
        select("no-such-entry", Mode.LOGPROBS, catalogue=catalogue)

    assert "'no-such-entry'" in str(caught.value)
    assert all(entry.entry_id in str(caught.value) for entry in catalogue.entries)
    assert not isinstance(caught.value, CatalogueError)


# -- select: no mode named (2.7) ----------------------------------------------------


def test_select_without_a_mode_returns_the_single_qualified_mode(catalogue):
    assert select(LLAMACPP, catalogue=catalogue) == select(LLAMACPP, Mode.LOGPROBS, catalogue=catalogue)
    assert select(OGA, catalogue=catalogue, k=50).mode is Mode.LOGITS


def test_select_without_a_mode_asks_for_one_when_no_mode_is_qualified(catalogue):
    with pytest.raises(SelectionError) as caught:
        select(OLLAMA, catalogue=catalogue)

    assert str(caught.value) == (
        "Catalogue entry 'ollama-gemma3-4b' has no single qualified mode; name one. "
        "Modes: logprobs is not_tested; logits is not_tested (not offered)"
    )


def test_select_without_a_mode_asks_for_one_when_both_modes_are_qualified():
    data = fixture_data()
    modes = data["entries"][0]["modes"]
    modes["logits"]["calibrations"] = [copy.deepcopy(modes["logprobs"]["calibrations"][1])]
    both = load_catalogue(data)

    with pytest.raises(SelectionError) as caught:
        select(LLAMACPP, catalogue=both)

    assert str(caught.value).endswith("name one. Modes: logprobs is qualified; logits is qualified")
    assert select(LLAMACPP, Mode.LOGITS, catalogue=both).qualification.status is Status.QUALIFIED


def test_select_loads_the_shipped_catalogue_only_when_none_is_passed(catalogue, mocker):
    load = mocker.patch.object(selection_module, "load_catalogue", return_value=catalogue)

    select(LLAMACPP, Mode.LOGPROBS, catalogue=catalogue)
    assert load.call_count == 0

    assert select(LLAMACPP, Mode.LOGPROBS) == select(LLAMACPP, Mode.LOGPROBS, catalogue=catalogue)
    load.assert_called_once_with()


# -- guarded_select: qualified (2.3, 2.6) -------------------------------------------


def test_qualified_selection_returns_the_correction_with_entry_mode_status_and_reason(catalogue, series_calibrator):
    chosen = select(LLAMACPP, Mode.LOGPROBS, catalogue=catalogue)
    result = run(chosen, series_calibrator)

    assert isinstance(result, SelectedDecision)
    assert (result.entry_id, result.mode, result.status) == (LLAMACPP, Mode.LOGPROBS, Status.QUALIFIED)
    assert result.reason == chosen.qualification.reason == "all four checks passed"
    assert result.decision.status == "ok"
    assert result.answer == {"equity": 0.2, "gold": 0.5}
    assert result.answer is result.decision.decision
    assert result.corrected is result.decision.adjusted
    assert result.corrected == adjust(result.answer, result.p_memorized)
    assert result.corrected != result.answer


def test_the_guard_runs_unchanged_and_the_recall_estimate_is_the_guards(catalogue, series_calibrator, mocker):
    chosen = select(LLAMACPP, Mode.LOGPROBS, catalogue=catalogue)
    config = GuardConfig(definition(), chosen.source, draws=8, min_parsed=8)
    direct = guarded_decide(PROMPT, config, client=ScriptedClient(UNIMODAL), calibrator=series_calibrator)
    spy = mocker.spy(selection_module, "guarded_decide")

    client = ScriptedClient(UNIMODAL)
    result = guarded_select(PROMPT, chosen, config.definition, client=client, calibrator=series_calibrator,
                            draws=8, min_parsed=8)

    assert spy.call_count == 1
    assert spy.call_args.args == (PROMPT, config)
    assert spy.call_args.kwargs == {"client": client, "calibrator": series_calibrator, "full_distribution": None}
    assert result.decision is spy.spy_return
    assert result.decision == direct
    assert result.p_memorized is result.decision.p_memorized
    assert result.p_memorized == direct.p_memorized and result.p_memorized is not None
    assert client.calls == [(PROMPT, {})] * 8


def test_guarded_select_defaults_to_the_guards_32_draw_ensemble(catalogue, series_calibrator):
    chosen = select(LLAMACPP, Mode.LOGPROBS, catalogue=catalogue)
    client = ScriptedClient(UNIMODAL)
    result = guarded_select(PROMPT, chosen, definition(), client=client, calibrator=series_calibrator)

    assert len(client.calls) == 32 and result.decision.identity.draws == 32
    assert result.decision.identity.source == chosen.source


def test_qualified_logits_selection_runs_the_full_distribution_source(catalogue, scorer):
    chosen = select(OGA, Mode.LOGITS, catalogue=catalogue, k=4)
    full = FakeFullDistribution()
    result = run(chosen, bound(scorer, EstimateSource.FULL_DISTRIBUTION, OGA_FINGERPRINT), full_distribution=full)

    assert len(full.calls) == 8
    assert (result.entry_id, result.mode, result.status) == (OGA, Mode.LOGITS, Status.QUALIFIED)
    assert result.decision.identity.source == SourceLabel(EstimateSource.FULL_DISTRIBUTION, 4)
    assert result.corrected == adjust(result.answer, result.p_memorized)


# -- guarded_select: not qualified and not tested (2.5, 2.6) ------------------------


def test_not_qualified_selection_returns_the_answer_without_the_correction(catalogue, scorer):
    chosen = select(LLAMACPP, Mode.LOGITS, catalogue=catalogue)
    result = run(chosen, bound(scorer, EstimateSource.FULL_DISTRIBUTION), full_distribution=FakeFullDistribution())

    assert (result.entry_id, result.mode, result.status) == (LLAMACPP, Mode.LOGITS, Status.NOT_QUALIFIED)
    assert result.reason == chosen.qualification.reason
    assert result.reason.startswith("discrimination: held-out AUC 0.55")
    assert result.corrected is None
    # The guard itself ran and adjusted; the selection withholds the correction.
    assert result.decision.status == "ok" and result.decision.adjusted != result.decision.decision
    assert result.answer == {"equity": 0.2, "gold": 0.5}
    assert result.p_memorized == result.decision.p_memorized


def test_not_tested_selection_returns_the_answer_without_the_correction(catalogue, scorer):
    chosen = select(OLLAMA, Mode.LOGPROBS, catalogue=catalogue)
    result = run(chosen, bound(scorer, EstimateSource.TOP_K))

    assert (result.entry_id, result.mode, result.status) == (OLLAMA, Mode.LOGPROBS, Status.NOT_TESTED)
    assert result.reason == "no quality record"
    assert result.corrected is None
    assert result.answer == {"equity": 0.2, "gold": 0.5}
    assert result.p_memorized == result.decision.p_memorized


# -- guarded_select: calibrator fingerprint -----------------------------------------


def test_fingerprint_mismatch_downgrades_a_qualified_run_to_not_tested(catalogue, scorer, series_calibrator):
    chosen = select(LLAMACPP, Mode.LOGPROBS, catalogue=catalogue)
    other = bound(scorer, EstimateSource.TOP_K)
    assert other.fingerprint != SERIES_FINGERPRINT
    result = run(chosen, other)

    assert (result.entry_id, result.mode, result.status) == (LLAMACPP, Mode.LOGPROBS, Status.NOT_TESTED)
    assert result.corrected is None
    assert "fingerprint" in result.reason
    assert other.fingerprint in result.reason and SERIES_FINGERPRINT in result.reason
    # Nothing else differs from the run with the recorded calibrator.
    matching = run(chosen, series_calibrator)
    assert result.answer == matching.answer and result.p_memorized == matching.p_memorized
    assert result.decision.status == "ok"
    assert chosen.qualification.status is Status.QUALIFIED


def test_a_qualified_record_without_a_fingerprint_downgrades_the_run_to_not_tested(series_calibrator):
    data = fixture_data()
    data["entries"][0]["modes"]["logprobs"]["calibrations"][1]["calibrator_fingerprint"] = None
    unrecorded = load_catalogue(data)
    chosen = select(LLAMACPP, Mode.LOGPROBS, catalogue=unrecorded)
    assert chosen.qualification.status is Status.QUALIFIED
    result = run(chosen, series_calibrator)

    assert result.status is Status.NOT_TESTED
    assert result.corrected is None
    assert "no calibrator fingerprint" in result.reason
    assert result.answer == {"equity": 0.2, "gold": 0.5}


def test_an_unqualified_status_is_kept_whatever_the_fingerprint(catalogue, scorer):
    chosen = select(LLAMACPP, Mode.LOGITS, catalogue=catalogue)
    result = run(chosen, bound(scorer, EstimateSource.FULL_DISTRIBUTION, "sha256:any"),
                 full_distribution=FakeFullDistribution())

    assert result.status is Status.NOT_QUALIFIED and result.reason == chosen.qualification.reason


# -- guarded_select: guard status other than ok -------------------------------------


def test_a_qualified_run_whose_guard_status_is_not_ok_exposes_no_correction(catalogue, series_calibrator):
    chosen = select(LLAMACPP, Mode.LOGPROBS, catalogue=catalogue)
    client = ScriptedClient(UNIMODAL, evidence=False)
    result = guarded_select(PROMPT, chosen, definition(), client=client, calibrator=series_calibrator,
                            draws=8, min_parsed=8)

    assert result.decision.status == "estimate_unavailable"
    assert result.status is Status.QUALIFIED
    assert result.corrected is None
    assert "estimate_unavailable" in result.reason
    assert result.answer == {"equity": 0.2, "gold": 0.5} and result.p_memorized is None


def test_a_failed_guard_run_carries_no_answer_and_still_states_entry_mode_and_status(catalogue, series_calibrator):
    chosen = select(LLAMACPP, Mode.LOGPROBS, catalogue=catalogue)
    result = run(chosen, series_calibrator, contents=["I would rather not say."] * 8)

    assert result.decision.status == "failed"
    assert (result.entry_id, result.mode, result.status) == (LLAMACPP, Mode.LOGPROBS, Status.QUALIFIED)
    assert (result.answer, result.corrected, result.p_memorized) == (None, None, None)
    assert "failed" in result.reason


def test_guarded_select_refuses_anything_but_a_selection(series_calibrator):
    with pytest.raises(TypeError, match="Selection"):
        guarded_select(PROMPT, LLAMACPP, definition(), client=ScriptedClient(UNIMODAL), calibrator=series_calibrator)


def test_results_are_immutable(catalogue, series_calibrator):
    chosen = select(LLAMACPP, Mode.LOGPROBS, catalogue=catalogue)
    result = run(chosen, series_calibrator)

    with pytest.raises(dataclasses.FrozenInstanceError):
        chosen.mode = Mode.LOGITS  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.status = Status.NOT_TESTED  # type: ignore[misc]
    with pytest.raises(TypeError):
        result.corrected["equity"] = 1.0  # type: ignore[index]


# -- package root -------------------------------------------------------------------


@pytest.mark.parametrize("name, module", SELECTION_EXPORTS)
def test_catalogue_and_selection_names_are_exported_from_the_root_as_the_module_objects(name, module):
    assert name in recall_guard.__all__
    assert getattr(recall_guard, name) is getattr(importlib.import_module(module), name)


def test_root_exports_have_no_duplicate_and_all_resolve():
    exported = list(recall_guard.__all__)
    assert len(exported) == len(set(exported))
    assert [name for name in exported if not hasattr(recall_guard, name)] == []


def test_importing_the_root_loads_no_matplotlib_and_reads_no_catalogue_file():
    code = (
        "import builtins, pathlib, sys\n"
        "opened = []\n"
        "real_open, real_read = builtins.open, pathlib.Path.read_text\n"
        "def spy_open(file, *args, **kwargs):\n"
        "    opened.append(str(file))\n"
        "    return real_open(file, *args, **kwargs)\n"
        "def spy_read(self, *args, **kwargs):\n"
        "    opened.append(str(self))\n"
        "    return real_read(self, *args, **kwargs)\n"
        "builtins.open, pathlib.Path.read_text = spy_open, spy_read\n"
        "import recall_guard\n"
        "heavy = sorted(m for m in sys.modules if m.split('.')[0] in {'matplotlib', 'vectorbt'})\n"
        "assert not heavy, heavy\n"
        "assert not [name for name in opened if 'catalogue' in name.lower()], opened\n"
        "assert 'recall_guard.harness.selection' in sys.modules\n"
        "assert recall_guard.select.__module__ == 'recall_guard.harness.selection'\n"
        "print('lean-ok')\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
    assert "lean-ok" in done.stdout
