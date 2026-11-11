"""The shipped catalogue data: loads, is traceable to the research register, and carries no status. Offline."""

import copy
import json
import re
from dataclasses import fields
from pathlib import Path

import pytest

from recall_guard.core.catalogue import (
    Catalogue,
    CatalogueError,
    Mode,
    ModeRecord,
    Status,
    load_catalogue,
    qualify,
)
from recall_guard.core.full_distribution import FullDistributionEvidence
from recall_guard.core.nvidia_lm import TokenLogprob
from recall_guard.core.profiles import available_profiles
from recall_guard.mia.estimate_source import features_from_full, features_from_top_k
from recall_guard.mia.features import MiaFeatures
from scripts.finance_sim.calibrate import FEATURE_ORDER

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
REGISTER = DOCS / "research-register.md"
FIXTURE = ROOT / "tests" / "fixtures" / "catalogue" / "catalogue_fixture.json"

OGA_ENTRY = "ryzenai-oga-gemma4-e4b"
FIRST_CONTENT_IDS = {"R26", "R30", "R38", "R40", "R41", "R43"}
METHOD_WORDING = {
    "row_bootstrap": "row-level stratified bootstrap",
    "prompt_clustered_bootstrap": "prompt-clustered",
}

NQ, NT = Status.NOT_QUALIFIED, Status.NOT_TESTED
# (logprobs, logits) as derived today; a later change of the data or the rule shows here.
EXPECTED_STATUSES = {
    "flm-gemma3-4b": (NQ, NT),
    "flm-gemma4-e4b": (NQ, NT),
    "flm-llama3.2-3b": (NT, NT),
    "llamacpp-gemma3-4b": (NT, NT),
    "modal-gemma3-4b": (NQ, NT),
    "modal-gpt-oss-20b": (NQ, NT),
    "nim-llama3.1-70b": (NT, NT),
    "ollama-gemma3-4b": (NQ, NT),
    "ollama-gpt-oss-20b": (NT, NT),
    "ollama-llama3.1-8b": (NQ, NT),
    "ollama-llama3.2-3b": (NQ, NT),
    "ollama-llava-llama3-ctx2048": (NT, NT),
    "ollama-phi4-mini": (NT, NT),
    OGA_ENTRY: (NT, NQ),
}
EXPECTED_REASONS = {
    # Boundary and sample pass (132 and 145 valid draws entering the fit); the next input is not recorded yet.
    ("llamacpp-gemma3-4b", Mode.LOGPROBS): "discrimination: no permutation p recorded",
    ("llamacpp-gemma3-4b", Mode.LOGITS): "discrimination: no permutation p recorded",
    # Valid draws entering the fit are not recorded for this arm; the recorded failing check decides.
    ("modal-gemma3-4b", Mode.LOGPROBS): "discrimination: held-out AUC 0.583 is below 0.6",
    ("ollama-gemma3-4b", Mode.LOGPROBS): "boundary: recall boundary domain is document_text, not financial_time_series",
    (OGA_ENTRY, Mode.LOGITS): "boundary: recall boundary domain is document_text, not financial_time_series",
    ("ollama-phi4-mini", Mode.LOGPROBS): "no quality record",
}
EXPECTED_CHECKS = {
    ("llamacpp-gemma3-4b", Mode.LOGPROBS): {"boundary": True, "sample": True, "discrimination": None, "cutoff": None},
    ("llamacpp-gemma3-4b", Mode.LOGITS): {"boundary": True, "sample": True, "discrimination": None, "cutoff": None},
    ("modal-gemma3-4b", Mode.LOGPROBS): {"boundary": True, "sample": None, "discrimination": False, "cutoff": None},
}
# Valid draws per class entering the fit, held-out included: recorded only where a source states them.
EXPECTED_VALID_DRAWS = {
    ("llamacpp-gemma3-4b", Mode.LOGPROBS, "financial_time_series"): (132, 145),
    ("llamacpp-gemma3-4b", Mode.LOGPROBS, "document_text"): (160, 400),
    ("llamacpp-gemma3-4b", Mode.LOGITS, "financial_time_series"): (132, 145),
    ("llamacpp-gemma3-4b", Mode.LOGITS, "document_text"): (160, 400),
}
# Requirement 4.1: what the method computes, and what neither path can supply (no reference model is set).
METHOD_FEATURES = tuple(FEATURE_ORDER)
METHOD_FEATURES_UNAVAILABLE = tuple(field.name for field in fields(MiaFeatures) if field.name not in FEATURE_ORDER)


@pytest.fixture(scope="module")
def catalogue() -> Catalogue:
    return load_catalogue()


def _calibrations(catalogue: Catalogue) -> list[tuple[str, ModeRecord]]:
    """Every recorded calibration, the one the rule reads and the others."""
    found = []
    for entry in catalogue.entries:
        for mode, record in entry.modes.items():
            for item in (record, *record.other_evidence):
                if item.evidence is not None:
                    found.append((f"{entry.entry_id} {mode.value} {'+'.join(item.evidence.register_ids)}", item))
    return found


# --- the shipped file ---------------------------------------------------------


def test_one_entry_per_shipped_profile_plus_the_route_without_a_profile(catalogue: Catalogue) -> None:
    ids = [entry.entry_id for entry in catalogue.entries]
    assert sorted(ids) == sorted((*available_profiles(), OGA_ENTRY))
    for entry in catalogue.entries:
        assert entry.profile_id == (None if entry.entry_id == OGA_ENTRY else entry.entry_id)


def test_every_entry_and_mode_has_today_s_status_with_a_reason(catalogue: Catalogue) -> None:
    statuses = {
        entry.entry_id: tuple(
            catalogue.qualification(entry.entry_id, mode).status for mode in (Mode.LOGPROBS, Mode.LOGITS)
        )
        for entry in catalogue.entries
    }
    assert statuses == EXPECTED_STATUSES
    assert catalogue.filter(status=Status.QUALIFIED) == ()
    for entry in catalogue.entries:
        for mode in Mode:
            assert catalogue.qualification(entry.entry_id, mode).reason.strip()
    for (entry_id, mode), reason in EXPECTED_REASONS.items():
        assert catalogue.qualification(entry_id, mode).reason == reason
    for (entry_id, mode), checks in EXPECTED_CHECKS.items():
        assert dict(catalogue.qualification(entry_id, mode).checks) == checks


def test_valid_draws_entering_the_fit_are_recorded_only_where_a_source_states_them(catalogue: Catalogue) -> None:
    recorded = {}
    for entry in catalogue.entries:
        for mode, record in entry.modes.items():
            for item in (record, *record.other_evidence):
                if item.quality is None:
                    continue
                valid = (item.quality.n_valid_exposed, item.quality.n_valid_fenced)
                assert (valid[0] is None) == (valid[1] is None), entry.entry_id
                if valid[0] is not None:
                    key = (entry.entry_id, mode, item.boundary.domain)
                    assert key not in recorded, key
                    assert item.evidence.method == "reply_features" and item.evidence.register_ids == ("R43",), key
                    recorded[key] = valid
    assert recorded == EXPECTED_VALID_DRAWS


def test_available_modes_list_the_method_s_features_and_what_they_cannot_supply(catalogue: Catalogue) -> None:
    assert METHOD_FEATURES == ("loss", "min_k", "min_k_pp", "zlib_ratio")
    assert METHOD_FEATURES_UNAVAILABLE == ("ref_delta",)
    both = 0
    for entry in catalogue.entries:
        logprobs, logits = entry.modes[Mode.LOGPROBS], entry.modes[Mode.LOGITS]
        for record in (logprobs, logits, *logprobs.other_evidence, *logits.other_evidence):
            where = f"{entry.entry_id} {record.mode.value}"
            if record.available:
                assert record.features == METHOD_FEATURES, where
                assert record.features_unavailable == METHOD_FEATURES_UNAVAILABLE, where
            else:
                # Nothing is computed on a mode that is not available.
                assert (record.features, record.features_unavailable) == ((), ()), where
        if logprobs.available and logits.available:
            both += 1
            assert logprobs.features == logits.features, entry.entry_id
            assert logprobs.features_unavailable == logits.features_unavailable, entry.entry_id
    assert both == 1


def test_both_evidence_paths_compute_the_listed_features_and_no_reference_delta() -> None:
    alternatives = (((1, -0.2), (2, -1.9), (3, -3.5)), ((4, -0.6), (5, -1.1), (2, -2.8)))
    evidence = FullDistributionEvidence(
        model_digest="sha256:" + "0" * 64,
        n_vocab=10,
        response_token_ids=(1, 5),
        logprob=(-0.2, -1.1),
        entropy=(0.7, 1.0),
        rank=(1, 2),
        top_logprobs=alternatives,
    )
    top_k = [
        TokenLogprob(
            token=f"<{token_id}>",
            logprob=logprob,
            top_logprobs=[{"token": f"<{alt}>", "logprob": value} for alt, value in position],
        )
        for token_id, logprob, position in zip(evidence.response_token_ids, evidence.logprob, alternatives, strict=True)
    ]
    for sourced in (features_from_top_k("ab", top_k), features_from_full("ab", evidence)):
        for name in METHOD_FEATURES:
            assert isinstance(getattr(sourced.features, name), float), name
        for name in METHOD_FEATURES_UNAVAILABLE:
            assert getattr(sourced.features, name) is None, name


def test_unavailable_modes_state_why_and_logits_state_what_they_require(catalogue: Catalogue) -> None:
    for entry in catalogue.entries:
        logprobs, logits = entry.modes[Mode.LOGPROBS], entry.modes[Mode.LOGITS]
        for record in (logprobs, logits):
            assert record.available == (record.unavailable_reason is None)
            if not record.available:
                assert qualify(record).status is Status.NOT_TESTED
        if not logits.available or not logprobs.available:
            assert logits.requires
    oga = catalogue.entry(OGA_ENTRY)
    assert not oga.modes[Mode.LOGPROBS].available and oga.modes[Mode.LOGITS].available
    assert [entry.entry_id for entry in catalogue.without_logprobs()] == ["llamacpp-gemma3-4b", OGA_ENTRY]
    assert all(catalogue.usable(entry.entry_id) for entry in catalogue.entries)


def test_no_fingerprint_no_permutation_p_and_no_cutoff_check_yet(catalogue: Catalogue) -> None:
    calibrations = _calibrations(catalogue)
    assert calibrations
    for where, record in calibrations:
        assert record.calibrator_fingerprint is None, where
        assert record.cutoff is None, where
        assert record.quality is not None, where
        assert record.quality.permutation_p is None, where


def test_methods_and_boundaries_of_the_first_content(catalogue: Catalogue) -> None:
    for where, record in _calibrations(catalogue):
        ids = set(record.evidence.register_ids)
        assert ids <= FIRST_CONTENT_IDS, where
        assert record.boundary is not None, where
        # Document likelihood scores the document, not the reply: evidence of a different method.
        assert (record.evidence.method == "document_likelihood") == bool(ids & {"R40", "R41"}), where
        if record.evidence.method == "document_likelihood":
            assert record.mode is Mode.LOGITS and record.boundary.domain == "document_text", where
        if record.boundary.domain == "financial_time_series":
            assert ids == {"R43"} and record.evidence.method == "reply_features", where
        assert record.quality.interval_method == (
            "prompt_clustered_bootstrap" if ids == {"R43"} else "row_bootstrap"
        ), where
        assert record.quality.confidence == 0.95, where
        assert (record.quality.n_prompts is None) == (ids != {"R43"}), where
    recorded = {where.rsplit(" ", 1)[1] for where, _ in _calibrations(catalogue)}
    assert recorded == FIRST_CONTENT_IDS


def test_entries_without_a_recorded_calibrator_carry_no_quality(catalogue: Catalogue) -> None:
    without = (
        "flm-llama3.2-3b",
        "nim-llama3.1-70b",
        "ollama-gpt-oss-20b",
        "ollama-llava-llama3-ctx2048",
        "ollama-phi4-mini",
    )
    for entry_id in without:
        for record in catalogue.entry(entry_id).modes.values():
            assert record.quality is None and record.evidence is None and record.other_evidence == ()


def test_shared_weights_are_related_only_where_stated(catalogue: Catalogue) -> None:
    assert [entry.entry_id for entry in catalogue.related("llamacpp-gemma3-4b")] == ["ollama-gemma3-4b"]
    grouped = {entry.entry_id for entry in catalogue.entries if entry.weights_group is not None}
    assert grouped == {"llamacpp-gemma3-4b", "ollama-gemma3-4b"}
    for entry in catalogue.entries:
        if entry.weights_group is not None:
            assert entry.weights_identity is not None


# --- traceability to the research register --------------------------------------


def _flat(text: str) -> str:
    """Collapse line wraps, including those of a quoted note ("> " at the start of a line)."""
    return re.sub(r"\s+", " ", re.sub(r"^> ?", "", text, flags=re.MULTILINE))


def _sources(register_id: str) -> str:
    """The register entry and the dated reports it links to, whitespace collapsed."""
    register = REGISTER.read_text(encoding="utf-8")
    heading = re.search(rf"^### {register_id} · .*$", register, flags=re.MULTILINE)
    assert heading is not None, f"{register_id} is not an entry heading of the research register"
    rest = register[heading.end() :]
    following = re.search(r"^##+ ", rest, flags=re.MULTILINE)
    section = rest[: following.start()] if following else rest
    texts = [section]
    for link in re.findall(r"\]\((results/[^)#\s]+\.md)\)", section):
        texts.append((DOCS / link).read_text(encoding="utf-8"))
    return _flat("\n".join(texts))


@pytest.mark.skipif(not REGISTER.is_file(), reason="the docs tree is not present")
def test_every_evidence_id_and_recorded_value_is_in_the_register_or_its_reports(catalogue: Catalogue) -> None:
    checked = recorded_valid = 0
    for where, record in _calibrations(catalogue):
        text = " ".join(_sources(register_id) for register_id in record.evidence.register_ids)
        quality = record.quality
        auc, low, high = (
            f"{value:.3f}" for value in (quality.holdout_auc, quality.interval_low, quality.interval_high)
        )
        assert f"{auc} [{low}, {high}]" in text or f"| {auc} | {low} to {high} |" in text, (
            f"{where}: AUC {auc} with interval {low} to {high} is not written in its sources"
        )
        exposed, fenced = quality.n_exposed, quality.n_fenced
        pairs = (f"{exposed} and {fenced}", f"({exposed}, {fenced})", f"{exposed}/{fenced}")
        assert any(pair in text for pair in pairs), f"{where}: class counts {exposed}, {fenced} are not in its sources"
        valid = (quality.n_valid_exposed, quality.n_valid_fenced)
        if valid != (None, None):
            stated = re.search(r"Valid draws entering each fit \(holdout included\): ([^.]*)\.", text)
            assert stated is not None, f"{where}: no statement of the valid draws entering the fit in its sources"
            assert re.search(rf"\b{valid[0]} \w+ and {valid[1]} \w+\b", stated.group(1)), (
                f"{where}: valid draws {valid[0]} and {valid[1]} entering the fit are not in its sources"
            )
            recorded_valid += 1
        if quality.n_prompts is not None:
            assert f"from {quality.n_prompts} prompts" in text, (
                f"{where}: {quality.n_prompts} prompts not in its sources"
            )
        percent = f"{quality.confidence * 100:.0f}"
        assert f"{percent} percent" in text or f"{percent} %" in text, f"{where}: confidence level not in its sources"
        assert METHOD_WORDING[quality.interval_method] in text, f"{where}: interval method not stated in its sources"
        checked += 1
    assert checked == 14
    assert recorded_valid == len(EXPECTED_VALID_DRAWS)


# --- the two rules the first content relies on ----------------------------------


@pytest.fixture
def data() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _entry(data: dict, entry_id: str) -> dict:
    return next(entry for entry in data["entries"] if entry["entry_id"] == entry_id)


def test_logits_must_state_what_it_requires_when_logprobs_is_unavailable(data: dict) -> None:
    # Requirement 3.2: holds whether or not the logits mode is available.
    assert _entry(data, "oga-phi4-mini")["modes"]["logits"]["available"] is True
    _entry(data, "oga-phi4-mini")["modes"]["logits"]["requires"] = None
    with pytest.raises(CatalogueError, match=r"'oga-phi4-mini' logits: .*logprobs mode is unavailable.*requires"):
        load_catalogue(data)


def test_logits_requirement_stays_optional_while_logprobs_is_available(data: dict) -> None:
    _entry(data, "llamacpp-gemma3-4b")["modes"]["logits"]["requires"] = None
    assert load_catalogue(data).entry("llamacpp-gemma3-4b").modes[Mode.LOGITS].requires is None


@pytest.mark.parametrize(
    ("entry_id", "mode"),
    [("closed-route", "logprobs"), ("closed-route", "logits"), ("ollama-gemma3-4b", "logits")],
)
def test_an_unavailable_mode_must_not_carry_calibrations(data: dict, entry_id: str, mode: str) -> None:
    calibration = copy.deepcopy(_entry(data, "oga-phi4-mini")["modes"]["logits"]["calibrations"][0])
    _entry(data, entry_id)["modes"][mode]["calibrations"] = [calibration]
    with pytest.raises(CatalogueError, match=rf"'{entry_id}' {mode}: an unavailable mode must not carry calibrations"):
        load_catalogue(data)
