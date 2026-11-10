"""Stored calibrators: plan binding, kind separation and tamper refusal (task 7.6; Requirements 4.1, 4.3).

A calibrator is stored per arm, source and kind under the calibrators folder
(here under ``tmp_path``) and loads back only for the frozen plan, arm, source
and kind it was fitted on, so the macro calibrator is never applied to EDGAR
prompts. No network, credentials or model host is used.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from recall_guard.mia.features import MiaFeatures
from scripts.finance_sim.calibrate import (
    EDGAR,
    MACRO,
    CalibrationDraw,
    CalibrationError,
    CalibrationRecord,
    calibrator_path,
    edgar_corpus,
    load_calibrator,
    macro_corpus,
    write_calibrator,
)
from scripts.finance_sim.plan import PlanChangedError
from tests.scripts.finance_sim.test_calibration import (
    PLAN,
    SEED,
    calibrate,
    edgar_releases,
    frozen_plan,
    macro_row,
    macro_rows,
    no_network,  # noqa: F401 - autouse fixture: no test here may open a connection
)


@pytest.fixture(scope="module")
def stored(tmp_path_factory):
    return calibrate(tmp_path_factory.mktemp("stored"))


def path_of(stored, kind: str, source: str = "top_k"):
    return stored.run.paths[(kind, source)]


def test_a_stored_calibrator_loads_back_identically_and_estimates_the_same(stored):
    for (kind, source), record in stored.run.records.items():
        loaded = load_calibrator(path_of(stored, kind, source), PLAN, arm="sampling", kind=kind, source=source)
        assert loaded.to_dict() == record.to_dict()
        assert loaded.calibrator.fingerprint == record.calibrator.fingerprint
        features = MiaFeatures(loss=0.4, min_k=-0.9, min_k_pp=-0.5, zlib_ratio=0.02, ref_delta=None)
        assert loaded.calibrator.estimate(features) == record.calibrator.estimate(features)
        assert loaded.calibrator.mcs.holdout_probabilities == record.calibrator.mcs.holdout_probabilities
    loaded = load_calibrator(path_of(stored, EDGAR), PLAN, arm="sampling", kind=EDGAR, source="top_k")
    assert loaded.calibrator.holdout_auc_interval() == stored.run.records[(EDGAR, "top_k")].holdout_auc_interval
    assert path_of(stored, MACRO) == calibrator_path("sampling", "top_k", MACRO, stored.root / "calibrators")
    assert path_of(stored, MACRO).relative_to(stored.root).as_posix() == "calibrators/sampling/top_k/macro.json"


def test_full_distribution_calibrators_record_how_their_evidence_was_requested(stored):
    for kind in (MACRO, EDGAR):
        full = json.loads(path_of(stored, kind, "full_distribution").read_text(encoding="utf-8"))
        assert full["evidence_settings"] == {"mode": "batch_per_prompt_and_rendering", "timeout_s": 600.0,
                                             "top_k_keep": 100, "host_busy_retries": 5, "host_busy_wait_s": 60.0}
        assert json.loads(path_of(stored, kind).read_text(encoding="utf-8"))["evidence_settings"] is None


def test_the_macro_calibrator_is_never_loaded_for_edgar_prompts(stored, tmp_path):
    with pytest.raises(CalibrationError, match="macro calibrator.*EDGAR"):
        load_calibrator(path_of(stored, MACRO), PLAN, arm="sampling", kind=EDGAR, source="top_k")
    # Copied to the EDGAR path, the file still names its kind and corpus, so it is refused there too.
    copied = calibrator_path("sampling", "top_k", EDGAR, tmp_path)
    copied.parent.mkdir(parents=True)
    copied.write_bytes(path_of(stored, MACRO).read_bytes())
    with pytest.raises(CalibrationError, match="kind"):
        load_calibrator(copied, PLAN, arm="sampling", kind=EDGAR, source="top_k")
    with pytest.raises(CalibrationError, match="source"):
        load_calibrator(path_of(stored, EDGAR), PLAN, arm="sampling", kind=EDGAR, source="full_distribution")
    with pytest.raises(CalibrationError, match="arm"):
        load_calibrator(path_of(stored, EDGAR), PLAN, arm="serving", kind=EDGAR, source="top_k")


def test_a_calibrator_of_another_plan_is_refused(stored):
    with pytest.raises(PlanChangedError):
        load_calibrator(path_of(stored, MACRO), frozen_plan(seed=SEED + 1), arm="sampling", kind=MACRO,
                        source="top_k")


def test_the_corpus_is_checked_against_the_frozen_plan_and_the_supplied_prompts(stored):
    corpus = macro_corpus(PLAN, macro_rows())
    assert load_calibrator(path_of(stored, MACRO), PLAN, arm="sampling", kind=MACRO, source="top_k",
                           corpus=corpus).corpus_sha256 == corpus.corpus_sha256
    month = PLAN.months.calibration[0]
    changed = replace(macro_row(month), raw={"baa10y": 9.0, "cpi_yoy": 9.0, "t10y2y": 9.0})
    other = macro_corpus(PLAN, [changed] + macro_rows()[1:])
    with pytest.raises(CalibrationError, match="corpus"):
        load_calibrator(path_of(stored, MACRO), PLAN, arm="sampling", kind=MACRO, source="top_k", corpus=other)
    with pytest.raises(CalibrationError, match="corpus"):
        load_calibrator(path_of(stored, MACRO), PLAN, arm="sampling", kind=MACRO, source="top_k",
                        corpus=edgar_corpus(PLAN, edgar_releases()))


def _tampered(stored, tmp_path, change) -> object:
    payload = json.loads(path_of(stored, EDGAR).read_text(encoding="utf-8"))
    change(payload)
    target = calibrator_path("sampling", "top_k", EDGAR, tmp_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload), encoding="utf-8")
    return target


def _swap_first_entries(payload):
    entries = payload["corpus"]["entries"]
    entries[0], entries[1] = entries[1], entries[0]


@pytest.mark.parametrize(("change", "message"), [
    (lambda p: p["fit"]["coefficients"][0].__setitem__(0, 9.0), "fingerprint"),
    (lambda p: p["fit"].__setitem__("is_weak", True), "fingerprint"),
    (lambda p: p["corpus"]["entries"][0].__setitem__("prompt_sha256", "0" * 64), "corpus_sha256"),
    (_swap_first_entries, "corpus"),
    (lambda p: p.__setitem__("schema", "finance-sim-calibrator/0"), "schema"),
    (lambda p: p.__setitem__("extra", 1), "fields"),
])
def test_a_tampered_calibrator_is_refused(stored, tmp_path, change, message):
    target = _tampered(stored, tmp_path, change)
    with pytest.raises(CalibrationError, match=message):
        load_calibrator(target, PLAN, arm="sampling", kind=EDGAR, source="top_k")


def test_a_calibrator_file_with_duplicate_keys_or_nan_is_refused(stored, tmp_path):
    text = path_of(stored, EDGAR).read_text(encoding="utf-8")
    target = calibrator_path("sampling", "top_k", EDGAR, tmp_path)
    target.parent.mkdir(parents=True)
    for broken in (text.replace('"kind": "edgar"', '"kind": "edgar", "kind": "macro"', 1),
                   text.replace('"holdout_auc": 1.0', '"holdout_auc": NaN', 1)):
        assert broken != text
        target.write_text(broken, encoding="utf-8")
        with pytest.raises(CalibrationError):
            load_calibrator(target, PLAN, arm="sampling", kind=EDGAR, source="top_k")


def test_writing_never_replaces_a_different_calibrator(stored, tmp_path):
    record = stored.run.records[(MACRO, "top_k")]
    path = write_calibrator(record, tmp_path / "store")
    assert write_calibrator(record, tmp_path / "store") == path  # the same content is a no-op
    assert CalibrationRecord.from_dict(json.loads(path.read_text(encoding="utf-8"))).to_dict() == record.to_dict()
    before = path.read_bytes()
    refit = calibrate(tmp_path / "refit", signal=False, sources=("top_k",)).run.records[(MACRO, "top_k")]
    with pytest.raises(CalibrationError, match="refusing to replace"):
        write_calibrator(refit, tmp_path / "store")
    assert path.read_bytes() == before


def test_calibration_draws_round_trip_and_refuse_unknown_fields(stored):
    for draw in stored.draws[:3]:
        payload = json.loads(json.dumps(draw.to_dict()))
        assert CalibrationDraw.from_dict(payload) == draw
    payload = stored.draws[0].to_dict() | {"content": "secret reply"}
    with pytest.raises(CalibrationError, match="fields"):
        CalibrationDraw.from_dict(payload)
    with pytest.raises((TypeError, ValueError)):
        CalibrationDraw.from_dict(stored.draws[0].to_dict() | {"label": 2})
