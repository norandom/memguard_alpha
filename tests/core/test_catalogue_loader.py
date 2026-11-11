"""Catalogue loader: validation, availability from the profile, filters. Offline."""

import copy
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from recall_guard.core import catalogue as catalogue_module
from recall_guard.core.catalogue import (
    Catalogue,
    CatalogueError,
    Mode,
    Status,
    load_catalogue,
    qualify,
)
from recall_guard.core.profiles import available_profiles, load_profile

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "catalogue" / "catalogue_fixture.json"
SOURCE_OF_MODE = {Mode.LOGPROBS: "top_k", Mode.LOGITS: "full_distribution"}


@pytest.fixture(scope="module")
def catalogue() -> Catalogue:
    return load_catalogue(FIXTURE)


@pytest.fixture
def data() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _ids(entries: tuple) -> list[str]:
    return [entry.entry_id for entry in entries]


def _series(data: dict) -> dict:
    """The time-series calibration of the first entry's logprobs mode."""
    return data["entries"][0]["modes"]["logprobs"]["calibrations"][1]


# --- loading ---------------------------------------------------------------


def test_loads_entries_in_file_order(catalogue: Catalogue) -> None:
    assert _ids(catalogue.entries) == ["llamacpp-gemma3-4b", "ollama-gemma3-4b", "oga-phi4-mini", "closed-route"]
    entry = catalogue.entries[0]
    assert (entry.model, entry.route) == ("gemma3:4b", "llama.cpp server, local")
    assert entry.weights_identity == "sha256:fixture-gemma3-4b"
    assert set(entry.modes) == {Mode.LOGPROBS, Mode.LOGITS}
    assert all(record.mode is mode for mode, record in entry.modes.items())


def test_loads_from_a_path_a_string_path_and_a_mapping(catalogue: Catalogue, data: dict) -> None:
    assert load_catalogue(str(FIXTURE)).entries == catalogue.entries
    assert load_catalogue(data).entries == catalogue.entries


def test_default_load_reads_the_packaged_file_and_says_when_it_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(catalogue_module, "_packaged_catalogue", lambda: tmp_path / "catalogue_data.json")
    with pytest.raises(CatalogueError, match=r"catalogue_data\.json"):
        load_catalogue()
    (tmp_path / "catalogue_data.json").write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    assert len(load_catalogue().entries) == 4


def test_packaged_file_is_package_data_of_core() -> None:
    packaged = catalogue_module._packaged_catalogue()
    assert packaged.name == "catalogue_data.json"
    assert Path(str(packaged)).parent == Path(catalogue_module.__file__).parent


def test_unreadable_file_and_invalid_json_are_catalogue_errors(tmp_path: Path) -> None:
    with pytest.raises(CatalogueError, match="Cannot read"):
        load_catalogue(tmp_path / "absent.json")
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    with pytest.raises(CatalogueError, match="Cannot read"):
        load_catalogue(broken)


def test_import_reads_no_catalogue_file() -> None:
    code = (
        "import builtins, pathlib, sys\n"
        "seen = []\n"
        "real_open, real_read = builtins.open, pathlib.Path.read_text\n"
        "def spy_open(file, *args, **kwargs):\n"
        "    seen.append(str(file)); return real_open(file, *args, **kwargs)\n"
        "def spy_read(self, *args, **kwargs):\n"
        "    seen.append(str(self)); return real_read(self, *args, **kwargs)\n"
        "import recall_guard.core.profiles\n"
        "builtins.open, pathlib.Path.read_text = spy_open, spy_read\n"
        "import recall_guard.core.catalogue\n"
        "sys.exit(1 if any(name.endswith('.json') for name in seen) else 0)\n"
    )
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0


def test_evidence_and_calibration_values_are_loaded(catalogue: Catalogue) -> None:
    record = catalogue.entry("llamacpp-gemma3-4b").modes[Mode.LOGPROBS]
    assert record.evidence.register_ids == ("R43", "R40")
    assert record.evidence.method == "reply_features"
    assert record.boundary.domain == "financial_time_series"
    assert record.quality.holdout_auc == 0.71
    assert (record.quality.interval_low, record.quality.interval_high) == (0.64, 0.78)
    assert record.quality.n_prompts == 40
    assert (record.quality.n_exposed, record.quality.n_fenced) == (120, 110)
    assert (record.quality.n_valid_exposed, record.quality.n_valid_fenced) == (480, 440)
    assert record.cutoff.difference == 0.21
    assert record.calibrator_fingerprint == "sha256:fixture-series"
    assert record.features == ("entropy", "top1_probability")
    assert catalogue.entry("ollama-gemma3-4b").modes[Mode.LOGITS].features_unavailable == ("tail_mass",)


def test_time_series_calibration_is_the_record_and_the_rest_is_other_evidence(catalogue: Catalogue) -> None:
    record = catalogue.entry("llamacpp-gemma3-4b").modes[Mode.LOGPROBS]
    assert qualify(record).status is Status.QUALIFIED
    (other,) = record.other_evidence
    assert other.boundary.domain == "document_text"
    assert other.evidence.register_ids == ("R26",)
    assert other.quality.holdout_auc == 0.9
    assert (other.mode, other.available, other.other_evidence) == (record.mode, record.available, ())
    assert qualify(other).status is Status.NOT_QUALIFIED


def test_a_single_document_text_calibration_is_the_record(data: dict) -> None:
    del data["entries"][0]["modes"]["logprobs"]["calibrations"][1]
    loaded = load_catalogue(data)
    record = loaded.entry("llamacpp-gemma3-4b").modes[Mode.LOGPROBS]
    assert record.boundary.domain == "document_text"
    assert record.other_evidence == ()
    assert loaded.qualification("llamacpp-gemma3-4b", Mode.LOGPROBS).status is Status.NOT_QUALIFIED


def test_mode_without_calibration_has_an_empty_record(catalogue: Catalogue) -> None:
    record = catalogue.entry("ollama-gemma3-4b").modes[Mode.LOGPROBS]
    assert (record.boundary, record.quality, record.cutoff, record.evidence) == (None, None, None, None)
    assert record.calibrator_fingerprint is None
    assert record.other_evidence == ()


# --- validation errors -----------------------------------------------------


def _set(path: tuple, value: object) -> Callable[[dict], None]:
    def change(data: dict) -> None:
        target = data
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value

    return change


def _drop(path: tuple) -> Callable[[dict], None]:
    def change(data: dict) -> None:
        target = data
        for key in path[:-1]:
            target = target[key]
        del target[path[-1]]

    return change


def _duplicate_entry(data: dict) -> None:
    data["entries"].append(copy.deepcopy(data["entries"][3]))


def _second_series_calibration(data: dict) -> None:
    calibrations = data["entries"][0]["modes"]["logprobs"]["calibrations"]
    calibrations.append(copy.deepcopy(calibrations[1]))


ENTRY = ("entries", 0)
LOGPROBS = (*ENTRY, "modes", "logprobs")
SERIES = (*LOGPROBS, "calibrations", 1)
QUALITY = (*SERIES, "quality")
CUTOFF = (*SERIES, "cutoff")
EVIDENCE = (*SERIES, "evidence")

INVALID = {
    "wrong schema": (_set(("schema",), "recall-guard-catalogue/2"), "schema"),
    "unknown top-level key": (_set(("status",), "qualified"), "status"),
    "missing top-level key": (_drop(("rule",)), "rule"),
    "entries not a list": (_set(("entries",), {}), "entries"),
    "unknown entry key": (_set((*ENTRY, "status"), "qualified"), "status"),
    "missing entry key": (_drop((*ENTRY, "weights_group")), "weights_group"),
    "empty entry id": (_set((*ENTRY, "entry_id"), ""), "entry_id"),
    "model not text": (_set((*ENTRY, "model"), 4), "model"),
    "duplicate entry id": (_duplicate_entry, "Duplicate entry_id 'closed-route'"),
    "missing mode": (_drop((*ENTRY, "modes", "logits")), "modes"),
    "unknown mode": (_set((*ENTRY, "modes", "embeddings"), {}), "modes"),
    "unknown mode key": (_set((*LOGPROBS, "status"), "qualified"), "status"),
    "missing mode key": (_drop((*LOGPROBS, "features")), "features"),
    "features not a list of text": (_set((*LOGPROBS, "features"), "entropy"), "features"),
    "unknown calibration key": (_set((*SERIES, "status"), "qualified"), "status"),
    "missing calibration key": (_drop((*SERIES, "calibrator_fingerprint")), "calibrator_fingerprint"),
    "unknown quality key": (_set((*QUALITY, "auc"), 0.7), "auc"),
    "missing quality key": (_drop((*QUALITY, "n_prompts")), "n_prompts"),
    "unknown cutoff key": (_set((*CUTOFF, "p"), 0.01), "'p'"),
    "unknown boundary key": (_set((*SERIES, "boundary", "note"), "x"), "note"),
    "unknown evidence key": (_set((*EVIDENCE, "report"), "x"), "report"),
    "evidence id without R": (_set((*EVIDENCE, "register_ids"), ["43"]), "R<digits>"),
    "evidence id with a suffix": (_set((*EVIDENCE, "register_ids"), ["R43b"]), "R<digits>"),
    "evidence id in lower case": (_set((*EVIDENCE, "register_ids"), ["r43"]), "R<digits>"),
    "evidence id not text": (_set((*EVIDENCE, "register_ids"), [43]), "R<digits>"),
    "evidence without ids": (_set((*EVIDENCE, "register_ids"), []), "register_ids"),
    "evidence without method": (_set((*EVIDENCE, "method"), ""), "method"),
    "quality without evidence": (_set((*SERIES, "evidence"), None), "without evidence"),
    "cutoff without evidence": (
        lambda data: (_set((*SERIES, "quality"), None)(data), _set((*SERIES, "evidence"), None)(data)),
        "without evidence",
    ),
    "unknown profile": (_set((*ENTRY, "profile_id"), "no-such-profile"), "no-such-profile"),
    "quality interval low above high": (_set((*QUALITY, "interval_low"), 0.8), "interval"),
    "quality interval without method": (_set((*QUALITY, "interval_method"), None), "interval_method"),
    "quality interval without confidence": (_set((*QUALITY, "confidence"), None), "confidence"),
    "confidence outside (0, 1)": (_set((*QUALITY, "confidence"), 95), "confidence"),
    "cutoff interval low above high": (_set((*CUTOFF, "interval_low"), 0.4), "interval"),
    "cutoff difference disagrees with the means": (_set((*CUTOFF, "difference"), 0.31), "difference"),
    "auc not a number": (_set((*QUALITY, "holdout_auc"), "0.71"), "holdout_auc"),
    "auc a boolean": (_set((*QUALITY, "holdout_auc"), True), "holdout_auc"),
    "count not an integer": (_set((*QUALITY, "n_exposed"), 120.5), "n_exposed"),
    "missing valid-draw key": (_drop((*QUALITY, "n_valid_exposed")), "n_valid_exposed"),
    "valid draws not an integer": (_set((*QUALITY, "n_valid_exposed"), 480.5), "n_valid_exposed"),
    "valid draws negative": (_set((*QUALITY, "n_valid_fenced"), -1), "n_valid_fenced"),
    "valid draws a boolean": (_set((*QUALITY, "n_valid_fenced"), True), "n_valid_fenced"),
    "valid draws below the held-out size": (
        _set((*QUALITY, "n_valid_fenced"), 109),
        "n_valid_fenced 109 is smaller than the held-out n_fenced 110",
    ),
    "p outside [0, 1]": (_set((*CUTOFF, "p_value"), 1.5), "p_value"),
    "rule min_auc differs": (_set(("rule", "min_auc"), 0.7), "min_auc"),
    "rule min_valid differs": (_set(("rule", "min_valid"), 30), "min_valid"),
    "rule alpha differs": (_set(("rule", "alpha"), 0.01), "alpha"),
    "rule with unknown key": (_set(("rule", "min_p"), 0.1), "min_p"),
    "two time-series calibrations": (_second_series_calibration, "more than one"),
    "availability contradicts the profile": (_set((*LOGPROBS, "available"), False), "contradicts"),
    "logits availability contradicts the profile": (
        _set(("entries", 1, "modes", "logits", "available"), True),
        "contradicts",
    ),
    "no availability without a profile": (_set(("entries", 2, "modes", "logits", "available"), None), "available"),
    "availability not a boolean": (_set(("entries", 2, "modes", "logits", "available"), "yes"), "available"),
    "unavailable mode without a reason": (
        _set(("entries", 2, "modes", "logprobs", "unavailable_reason"), None),
        "unavailable_reason",
    ),
    "unavailable logits without a requirement": (
        _set(("entries", 3, "modes", "logits", "requires"), None),
        "requires",
    ),
    "available mode with an unavailable reason": (_set((*LOGPROBS, "unavailable_reason"), "none"), "unavailable_reason"),
}


@pytest.mark.parametrize("case", INVALID)
def test_invalid_data_is_refused(data: dict, case: str) -> None:
    change, message = INVALID[case]
    change(data)
    with pytest.raises(CatalogueError, match=message):
        load_catalogue(data)


@pytest.mark.parametrize("keys", [("n_valid_exposed",), ("n_valid_fenced",), ("n_valid_exposed", "n_valid_fenced")])
def test_valid_draws_may_be_null_and_then_the_sample_check_has_no_input(data: dict, keys: tuple[str, ...]) -> None:
    for key in keys:
        _series(data)["quality"][key] = None
    loaded = load_catalogue(data)
    quality = loaded.entry("llamacpp-gemma3-4b").modes[Mode.LOGPROBS].quality
    assert all(getattr(quality, key) is None for key in keys)
    result = loaded.qualification("llamacpp-gemma3-4b", Mode.LOGPROBS)
    assert result.status is Status.NOT_TESTED
    assert result.reason == "sample: no valid draws per class entering the fit recorded"


def test_valid_draws_equal_to_the_held_out_sizes_are_accepted(data: dict) -> None:
    _series(data)["quality"].update(n_valid_exposed=120, n_valid_fenced=110)
    assert load_catalogue(data).qualification("llamacpp-gemma3-4b", Mode.LOGPROBS).status is Status.QUALIFIED


def test_root_must_be_a_mapping(tmp_path: Path) -> None:
    listed = tmp_path / "list.json"
    listed.write_text("[]", encoding="utf-8")
    with pytest.raises(CatalogueError):
        load_catalogue(listed)


def test_error_names_the_entry(data: dict) -> None:
    _set(("entries", 2, "modes", "logits", "calibrations", 0, "evidence", "register_ids"), ["41"])(data)
    with pytest.raises(CatalogueError, match="oga-phi4-mini"):
        load_catalogue(data)


def test_difference_within_rounding_is_accepted(data: dict) -> None:
    _series(data)["cutoff"]["difference"] = 0.2104
    assert load_catalogue(data).entry("llamacpp-gemma3-4b").modes[Mode.LOGPROBS].cutoff.difference == 0.2104


# --- availability ----------------------------------------------------------


def test_availability_equals_the_estimate_sources_of_the_profile(catalogue: Catalogue) -> None:
    with_profile = [entry for entry in catalogue.entries if entry.profile_id is not None]
    assert len(with_profile) == 2
    for entry in with_profile:
        sources = load_profile(entry.profile_id).estimate_sources
        for mode, source in SOURCE_OF_MODE.items():
            assert entry.modes[mode].available is (source in sources), (entry.entry_id, mode)


def test_every_shipped_profile_gets_its_availability_from_the_profile(data: dict) -> None:
    template = data["entries"][1]
    entries = []
    for profile_id in available_profiles():
        entry = copy.deepcopy(template)
        entry["entry_id"] = entry["profile_id"] = profile_id
        sources = load_profile(profile_id).estimate_sources
        for mode, source in SOURCE_OF_MODE.items():
            record = entry["modes"][mode.value]
            record["available"] = None
            record["unavailable_reason"] = None if source in sources else "reason from the data file"
            record["requires"] = "requirement from the data file"
        entries.append(entry)
    data["entries"] = entries
    loaded = load_catalogue(data)
    assert _ids(loaded.entries) == list(available_profiles())
    for entry in loaded.entries:
        sources = load_profile(entry.profile_id).estimate_sources
        for mode, source in SOURCE_OF_MODE.items():
            record = entry.modes[mode]
            assert record.available is (source in sources), (entry.entry_id, mode)
            assert record.unavailable_reason == (None if record.available else "reason from the data file")
            assert record.requires == "requirement from the data file"


def test_reason_and_requirement_come_from_the_data_file(catalogue: Catalogue) -> None:
    logits = catalogue.entry("ollama-gemma3-4b").modes[Mode.LOGITS]
    assert logits.available is False
    assert logits.unavailable_reason == "the route returns at most the top 20 tokens"
    assert logits.requires.startswith("a route that returns the full next-token distribution")


def test_entry_without_profile_takes_availability_from_the_data_file(catalogue: Catalogue) -> None:
    entry = catalogue.entry("oga-phi4-mini")
    assert entry.profile_id is None
    assert entry.modes[Mode.LOGPROBS].available is False
    assert entry.modes[Mode.LOGPROBS].unavailable_reason == "the runtime has no logprobs field in its replies"
    assert entry.modes[Mode.LOGITS].available is True
    assert entry.modes[Mode.LOGITS].requires == "in-process access to the logits of each generated token"


# --- listing and filters ---------------------------------------------------


def test_entry_by_id_and_unknown_id(catalogue: Catalogue) -> None:
    assert catalogue.entry("closed-route").model == "closed-model"
    with pytest.raises(CatalogueError, match="no-such-entry"):
        catalogue.entry("no-such-entry")


def test_qualification_of_each_entry_and_mode(catalogue: Catalogue) -> None:
    statuses = {
        (entry.entry_id, mode.value): catalogue.qualification(entry.entry_id, mode).status.value
        for entry in catalogue.entries
        for mode in Mode
    }
    assert statuses == {
        ("llamacpp-gemma3-4b", "logprobs"): "qualified",
        ("llamacpp-gemma3-4b", "logits"): "not_qualified",
        ("ollama-gemma3-4b", "logprobs"): "not_tested",
        ("ollama-gemma3-4b", "logits"): "not_tested",
        ("oga-phi4-mini", "logprobs"): "not_tested",
        ("oga-phi4-mini", "logits"): "qualified",
        ("closed-route", "logprobs"): "not_tested",
        ("closed-route", "logits"): "not_tested",
    }
    entry = catalogue.entry("llamacpp-gemma3-4b")
    assert catalogue.qualification(entry.entry_id, Mode.LOGITS) == qualify(entry.modes[Mode.LOGITS])
    assert catalogue.qualification(entry.entry_id, "logits") == qualify(entry.modes[Mode.LOGITS])
    with pytest.raises(CatalogueError):
        catalogue.qualification("no-such-entry", Mode.LOGITS)


def test_filter_without_arguments_lists_everything(catalogue: Catalogue) -> None:
    assert catalogue.filter() == catalogue.entries


def test_filter_by_mode_returns_entries_where_the_mode_is_available(catalogue: Catalogue) -> None:
    assert _ids(catalogue.filter(mode=Mode.LOGPROBS)) == ["llamacpp-gemma3-4b", "ollama-gemma3-4b"]
    assert _ids(catalogue.filter(mode=Mode.LOGITS)) == ["llamacpp-gemma3-4b", "oga-phi4-mini"]
    assert _ids(catalogue.filter(mode="logits")) == ["llamacpp-gemma3-4b", "oga-phi4-mini"]


def test_filter_by_status_matches_any_mode(catalogue: Catalogue) -> None:
    assert _ids(catalogue.filter(status=Status.QUALIFIED)) == ["llamacpp-gemma3-4b", "oga-phi4-mini"]
    assert _ids(catalogue.filter(status=Status.NOT_QUALIFIED)) == ["llamacpp-gemma3-4b"]
    assert _ids(catalogue.filter(status="not_tested")) == ["ollama-gemma3-4b", "oga-phi4-mini", "closed-route"]


def test_filter_by_mode_and_status_uses_the_status_of_that_mode(catalogue: Catalogue) -> None:
    assert _ids(catalogue.filter(mode=Mode.LOGPROBS, status=Status.QUALIFIED)) == ["llamacpp-gemma3-4b"]
    assert _ids(catalogue.filter(mode=Mode.LOGITS, status=Status.QUALIFIED)) == ["oga-phi4-mini"]
    assert _ids(catalogue.filter(mode=Mode.LOGITS, status=Status.NOT_QUALIFIED)) == ["llamacpp-gemma3-4b"]
    assert _ids(catalogue.filter(mode=Mode.LOGITS, status=Status.NOT_TESTED)) == ["ollama-gemma3-4b", "closed-route"]
    assert catalogue.filter(mode=Mode.LOGPROBS, status=Status.NOT_QUALIFIED) == ()


def test_filter_refuses_an_unknown_mode_or_status(catalogue: Catalogue) -> None:
    with pytest.raises(CatalogueError, match="embeddings"):
        catalogue.filter(mode="embeddings")
    with pytest.raises(CatalogueError, match="passed"):
        catalogue.filter(status="passed")


def test_without_logprobs_lists_entries_whose_logits_mode_is_available(catalogue: Catalogue) -> None:
    assert _ids(catalogue.without_logprobs()) == ["llamacpp-gemma3-4b", "oga-phi4-mini"]


def test_related_entries_share_a_weights_group(catalogue: Catalogue) -> None:
    assert _ids(catalogue.related("llamacpp-gemma3-4b")) == ["ollama-gemma3-4b"]
    assert _ids(catalogue.related("ollama-gemma3-4b")) == ["llamacpp-gemma3-4b"]
    # Two entries without a group are not related to each other.
    assert catalogue.related("oga-phi4-mini") == ()
    assert catalogue.related("closed-route") == ()
    with pytest.raises(CatalogueError):
        catalogue.related("no-such-entry")


def test_entry_with_neither_mode_available_is_not_usable(catalogue: Catalogue) -> None:
    assert catalogue.usable("closed-route") is False
    assert catalogue.usable("oga-phi4-mini") is True
    assert catalogue.usable("ollama-gemma3-4b") is True
    assert catalogue.usable("llamacpp-gemma3-4b") is True
    with pytest.raises(CatalogueError):
        catalogue.usable("no-such-entry")
