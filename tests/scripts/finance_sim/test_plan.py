"""Study plan splits, frozen fields, hashes and changed-plan refusal (task 7.2; Requirements 4.1, 6.6, 8.2)."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from dataclasses import fields, replace
from datetime import UTC, datetime, timedelta, timezone

import numpy as np
import pytest

from recall_guard.dataset.macro_regime import ASSETS, EXPOSURE
from scripts.finance_sim import plan as plan_module
from scripts.finance_sim.plan import (
    AccuracyDefinition,
    AnalysisSettings,
    EdgarRow,
    MonthClass,
    PlanChangedError,
    StudyPlan,
    ensure_unchanged,
    freeze_plan,
    load_plan,
    split_edgar,
    split_months,
    write_plan,
)

GEMMA = "google/gemma-3-4b-it"
#: Gemma 3 4B's cutoff is 2024-08-31, so the August 2024 decision date is still pre-cutoff.
GEMMA_CUTOFF = "2024-08"
CREATED_AT = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)
SEED = 20261110
EDGAR_FILE_NAMES = ("calibration/is_memorized.jsonl", "calibration/oos_control.jsonl",
                    "held_out/is_memorized.jsonl", "held_out/oos_control.jsonl")
METRICS = ["estimate_difference_pre_cutoff", "estimate_difference_post_cutoff", "accuracy_difference_pre_cutoff",
           "accuracy_difference_post_cutoff", "did_accuracy_gap", "brier_difference", "log_loss_difference",
           "achieved_power", "half_ensemble_variance_summary", "half_ensemble_variance_p_memorized",
           "first_draw_rate", "random_answerer_null"]


def month_range(first: str, last: str) -> list[str]:
    year, month = int(first[:4]), int(first[5:])
    out = []
    while f"{year:04d}-{month:02d}" <= last:
        out.append(f"{year:04d}-{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return out


STUDY_MONTHS = month_range("2016-01", "2026-06")


def classes(cutoffs: dict[str, str], months: list[str] = STUDY_MONTHS) -> list[MonthClass]:
    """One MonthClass per month; months compare as YYYY-MM strings, so ``<=`` is the cutoff test."""
    return [MonthClass(month, {model: month <= cutoff for model, cutoff in cutoffs.items()}) for month in months]


def edgar_rows(memorized: int = 40, control: int = 100, negative: int = 40, unused: int = 40) -> list[EdgarRow]:
    rows = [EdgarRow(f"cal-mem-{i:03d}", 1, "calibration") for i in range(memorized)]
    rows += [EdgarRow(f"cal-ctl-{i:03d}", 0, "calibration") for i in range(control)]
    rows += [EdgarRow(f"held-ctl-{i:03d}", 0, "held_out") for i in range(negative)]
    rows += [EdgarRow(f"held-mem-{i:03d}", 1, "held_out") for i in range(unused)]
    return rows


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


EDGAR_FILES = {name: digest(name) for name in EDGAR_FILE_NAMES}
ITEMS = {"macro-2016-01": digest("macro-2016-01"), "macro-2026-06": digest("macro-2026-06"),
         "edgar-cal-mem-000": digest("edgar-cal-mem-000"), "edgar-held-ctl-039": digest("edgar-held-ctl-039")}


def frozen(**overrides) -> StudyPlan:
    arguments = {"seed": SEED, "created_at": CREATED_AT, "items": ITEMS, "edgar_files": EDGAR_FILES} | overrides
    return freeze_plan(classes({GEMMA: GEMMA_CUTOFF}), edgar_rows(), **arguments)


def gemma_split(seed: int = SEED):
    return split_months(classes({GEMMA: GEMMA_CUTOFF}), seed=seed)


def years(months) -> Counter:
    return Counter(int(month[:4]) for month in months)


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def rehash(payload: dict) -> dict:
    """Recompute the section and plan hashes the way the design defines them, for validation-only tests."""
    out = {key: value for key, value in payload.items() if key not in ("content_sha256", "plan_sha256")}
    out["content_sha256"] = {key: digest(canonical(value)) for key, value in out.items() if isinstance(value, dict)}
    out["plan_sha256"] = digest(canonical(out))
    return out


# ---------- month split (4.1) ----------

def test_gemma_split_has_disjoint_calibration_pilot_and_evaluation_months_covering_every_month():
    split = gemma_split()
    calibration, pilot, evaluation = set(split.calibration), set(split.pilot), set(split.evaluation)
    assert len(STUDY_MONTHS) == 126
    assert (len(calibration), len(pilot), len(evaluation)) == (24, 12, 90)
    assert not calibration & pilot and not calibration & evaluation and not pilot & evaluation
    assert calibration | pilot | evaluation == set(STUDY_MONTHS)
    assert split.months == tuple(STUDY_MONTHS) and split.models == (GEMMA,)


def test_calibration_takes_three_pre_cutoff_months_per_calendar_year_2016_to_2023():
    split = gemma_split()
    assert years(split.calibration) == {year: 3 for year in range(2016, 2024)}
    assert all(month <= GEMMA_CUTOFF for month in split.calibration)


def test_pilot_months_are_all_before_the_cutoff_one_per_year_2016_to_2024_plus_three_extra():
    split = gemma_split()
    post = [month for month in STUDY_MONTHS if month > GEMMA_CUTOFF]
    assert all(month <= GEMMA_CUTOFF for month in split.pilot)
    assert set(split.pilot) <= set(split.pre_cutoff[GEMMA])
    assert max(split.pilot) < min(post)
    counts = years(split.pilot)
    assert set(counts) == set(range(2016, 2025))
    assert sorted(counts.values()) == [1] * 6 + [2] * 3


def test_evaluation_keeps_every_post_cutoff_month_and_the_remaining_68_pre_cutoff_months():
    split = gemma_split()
    post = [month for month in STUDY_MONTHS if month > GEMMA_CUTOFF]
    assert len(post) == 22 and len(split.pre_cutoff[GEMMA]) == 104
    assert set(post) <= set(split.evaluation)
    assert len([month for month in split.evaluation if month <= GEMMA_CUTOFF]) == 68


def test_split_follows_the_frozen_draw_order_of_numpy_default_rng():
    """Calibration first, then the three extra pilot years, then the pilot months in year order."""
    pre = [month for month in STUDY_MONTHS if month <= GEMMA_CUTOFF]
    rng = np.random.default_rng(SEED)
    calibration = []
    for year in range(2016, 2024):
        pool = [month for month in pre if int(month[:4]) == year]
        calibration += sorted(pool[int(i)] for i in rng.choice(len(pool), size=3, replace=False))
    pilot_years = list(range(2016, 2025))
    extra = {pilot_years[int(i)] for i in rng.choice(len(pilot_years), size=3, replace=False)}
    pilot = []
    for year in pilot_years:
        pool = [month for month in pre if int(month[:4]) == year and month not in calibration]
        pilot += sorted(pool[int(i)] for i in rng.choice(len(pool), size=1 + (year in extra), replace=False))
    split = gemma_split()
    assert split.calibration == tuple(sorted(calibration))
    assert split.pilot == tuple(sorted(pilot))
    assert {year for year, count in years(split.pilot).items() if count == 2} == extra


def test_split_is_reproducible_from_the_seed_and_changes_with_it():
    assert gemma_split() == gemma_split()
    assert gemma_split(SEED + 1).calibration != gemma_split().calibration


def test_a_month_is_eligible_for_calibration_or_pilot_only_before_every_models_cutoff():
    split = split_months(classes({GEMMA: GEMMA_CUTOFF, "vendor/earlier-model": "2024-06"}), seed=SEED)
    assert {"2024-07", "2024-08"} <= set(split.evaluation)
    assert all(month <= "2024-06" for month in split.calibration + split.pilot)
    assert split.pre_cutoff["vendor/earlier-model"] == tuple(month_range("2016-01", "2024-06"))
    assert split.pre_cutoff[GEMMA] == tuple(month_range("2016-01", GEMMA_CUTOFF))


def test_split_refuses_a_pilot_year_without_an_eligible_month():
    with pytest.raises(ValueError, match="pilot year 2024"):
        split_months(classes({GEMMA: GEMMA_CUTOFF, "meta/llama-3.1-8b-instruct": "2023-12"}), seed=SEED)


def test_split_refuses_a_calibration_year_with_fewer_than_three_eligible_months():
    months = [month for month in STUDY_MONTHS if not month.startswith("2017-") or month in ("2017-01", "2017-02")]
    with pytest.raises(ValueError, match="calibration year 2017"):
        split_months(classes({GEMMA: GEMMA_CUTOFF}, months), seed=SEED)


def test_split_keeps_working_when_item_assembly_excluded_some_months():
    months = [month for month in STUDY_MONTHS if month not in ("2019-05", "2025-01")]
    split = split_months(classes({GEMMA: GEMMA_CUTOFF}, months), seed=SEED)
    assert set(split.calibration) | set(split.pilot) | set(split.evaluation) == set(months)
    assert len(split.evaluation) == 88


@pytest.mark.parametrize("make", [
    pytest.param(lambda: [], id="empty"),
    pytest.param(lambda: classes({GEMMA: GEMMA_CUTOFF}) + classes({GEMMA: GEMMA_CUTOFF}, ["2018-03"]), id="duplicate"),
    pytest.param(lambda: classes({GEMMA: GEMMA_CUTOFF})[:-1] + classes({"other/model": "2024-08"}, ["2026-06"]),
                 id="different-models"),
    pytest.param(lambda: classes({GEMMA: GEMMA_CUTOFF})[:-1] + [MonthClass("2026-06", {GEMMA: True})],
                 id="pre-cutoff-after-post-cutoff"),
])
def test_split_refuses_inconsistent_month_inputs(make):
    with pytest.raises(ValueError):
        split_months(make(), seed=SEED)


@pytest.mark.parametrize(("month", "flags", "error"), [
    ("2016-13", {GEMMA: True}, ValueError),
    ("2016-1", {GEMMA: True}, ValueError),
    ("2016-01-31", {GEMMA: True}, ValueError),
    ("2016-01", {}, ValueError),
    ("2016-01", {GEMMA: 1}, TypeError),
    ("2016-01", {"": True}, ValueError),
])
def test_month_class_validates_month_and_flags(month, flags, error):
    with pytest.raises(error):
        MonthClass(month, flags)


@pytest.mark.parametrize("seed", [-1, True, 1.5, "1"])
def test_split_refuses_an_invalid_seed(seed):
    with pytest.raises((TypeError, ValueError)):
        split_months(classes({GEMMA: GEMMA_CUTOFF}), seed=seed)


# ---------- EDGAR calibration rows and negative-control items ----------

def test_edgar_split_freezes_40_memorized_and_100_control_calibration_rows_and_40_negative_control_items():
    split = split_edgar(edgar_rows())
    assert split.calibration_memorized == tuple(f"cal-mem-{i:03d}" for i in range(40))
    assert split.calibration_control == tuple(f"cal-ctl-{i:03d}" for i in range(100))
    assert split.negative_control == tuple(f"held-ctl-{i:03d}" for i in range(40))
    assert split.unused == tuple(f"held-mem-{i:03d}" for i in range(40))


def test_edgar_calibration_rows_and_negative_control_items_are_disjoint():
    split = split_edgar(edgar_rows())
    calibration = set(split.calibration_memorized) | set(split.calibration_control)
    assert len(calibration) == 140 and len(set(split.negative_control)) == 40
    assert not calibration & set(split.negative_control)
    assert not (calibration | set(split.negative_control)) & set(split.unused)


def test_edgar_split_refuses_a_row_in_both_calibration_and_negative_control():
    rows = edgar_rows(negative=39) + [EdgarRow("cal-ctl-000", 0, "held_out")]
    with pytest.raises(ValueError, match="cal-ctl-000"):
        split_edgar(rows)


@pytest.mark.parametrize("counts", [
    {"memorized": 39}, {"memorized": 41}, {"control": 99}, {"negative": 41}, {"negative": 0},
])
def test_edgar_split_refuses_counts_other_than_40_100_and_40(counts):
    with pytest.raises(ValueError):
        split_edgar(edgar_rows(**counts))


def test_edgar_split_accepts_any_number_of_unused_held_out_memorized_rows():
    assert split_edgar(edgar_rows(unused=0)).unused == ()


@pytest.mark.parametrize(("row_id", "label", "split", "error"), [
    ("", 1, "calibration", ValueError),
    ("x", 2, "calibration", ValueError),
    ("x", True, "calibration", TypeError),
    ("x", 1, "evaluation", ValueError),
    (7, 1, "calibration", ValueError),
])
def test_edgar_row_validates_id_label_and_split(row_id, label, split, error):
    with pytest.raises(error):
        EdgarRow(row_id, label, split)


# ---------- frozen fields (6.6, 8.2) ----------

def test_plan_freezes_the_pre_registered_settings():
    analysis = frozen().analysis
    assert list(analysis.metrics) == METRICS
    assert analysis.effect_size == 0.10 and analysis.tolerance == 0.10
    assert analysis.n_resamples == 2000 and analysis.block_length == 6
    assert analysis.negative_control_block_length == 2
    assert analysis.alpha == 0.05 and analysis.power_threshold == 0.8
    assert analysis.first_draw_band == (0.25, 0.75)


def test_plan_freezes_the_accuracy_definition_with_exposure_as_plain_dicts():
    plan = frozen()
    payload = plan.to_dict()
    accuracy = payload["accuracy"]
    assert plan.accuracy.rule == accuracy["rule"] == "macro_regime.is_correct"
    assert accuracy["exposure"] == {category: dict(row) for category, row in EXPOSURE.items()}
    assert type(accuracy["exposure"]) is dict and all(type(row) is dict for row in accuracy["exposure"].values())
    assert set(accuracy["sources"]) == {category for _, category, _ in ASSETS}
    sources = " ".join(accuracy["sources"].values())
    assert all(name in sources for name in ("SWDA.L", "XLK", "IAU", "DTB3", "adjClose"))
    assert "M+1" in accuracy["returns"] and "never substituted" in accuracy["missing"]
    json.dumps(payload, allow_nan=False)


def test_plan_records_seed_date_schema_items_edgar_files_and_section_hashes():
    payload = frozen().to_dict()
    assert payload["schema_version"] == "finance-sim-plan/1"
    assert payload["created_at"] == "2026-11-10T12:00:00+00:00"
    assert payload["seed"] == SEED and payload["items"] == ITEMS and payload["edgar_files"] == EDGAR_FILES
    assert payload["excluded_models"] == {}
    sections = {"items", "excluded_models", "months", "edgar", "edgar_files", "accuracy", "analysis"}
    assert set(payload["content_sha256"]) == sections
    for name in sections:
        assert payload["content_sha256"][name] == digest(canonical(payload[name]))


def test_plan_sha256_is_the_sha256_of_the_canonical_json_of_every_frozen_field_but_itself():
    plan = frozen()
    payload = plan.to_dict()
    body = {key: value for key, value in payload.items() if key != "plan_sha256"}
    assert payload["plan_sha256"] == plan.plan_sha256 == digest(canonical(body))


def test_created_at_is_normalised_to_utc():
    plan = frozen(created_at=datetime(2026, 11, 10, 13, 0, tzinfo=timezone(timedelta(hours=1))))
    assert plan.created_at == "2026-11-10T12:00:00+00:00"
    assert plan.plan_sha256 == frozen().plan_sha256


def test_written_plan_loads_back_identical_and_formatting_does_not_change_the_hash(tmp_path):
    plan = frozen()
    path = write_plan(plan, tmp_path / "plan" / "plan.json")
    loaded = load_plan(path)
    assert loaded == plan and loaded.plan_sha256 == plan.plan_sha256
    path.write_text(json.dumps(json.loads(path.read_text(encoding="utf-8"))), encoding="utf-8")
    assert load_plan(path, expected_sha256=plan.plan_sha256).plan_sha256 == plan.plan_sha256


# ---------- changed-plan refusal (8.2) ----------

def _leaf_paths(value, path=()):
    """Every dict key down to the leaves; in lists, the first and the last element."""
    if isinstance(value, dict) and value:
        for key in value:
            yield from _leaf_paths(value[key], (*path, key))
    elif isinstance(value, list) and value:
        for index in sorted({0, len(value) - 1}):
            yield from _leaf_paths(value[index], (*path, index))
    else:
        yield path


def _dict_paths(value, path=()):
    if isinstance(value, dict):
        yield path
        for key, item in value.items():
            yield from _dict_paths(item, (*path, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _dict_paths(item, (*path, index))


def _changed(value):
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 0.01
    if isinstance(value, str):
        return value + "x"
    if isinstance(value, dict):
        return {"extra": "x"}
    if isinstance(value, list):
        return ["extra"]
    raise AssertionError(f"unexpected leaf {value!r}")


def _at(payload, path):
    for key in path:
        payload = payload[key]
    return payload


def _refused(path, payload) -> bool:
    path.write_text(json.dumps(payload), encoding="utf-8")
    try:
        load_plan(path)
    except PlanChangedError:
        return True
    return False


def test_load_refuses_a_change_to_any_field_of_the_file(tmp_path):
    original = frozen().to_dict()
    path = tmp_path / "plan.json"
    leaves = list(_leaf_paths(original))
    assert ("plan_sha256",) in leaves and ("analysis", "effect_size") in leaves and ("excluded_models",) in leaves
    accepted = []
    for leaf in leaves:
        mutated = copy.deepcopy(original)
        parent = _at(mutated, leaf[:-1])
        parent[leaf[-1]] = _changed(parent[leaf[-1]])
        if not _refused(path, mutated):
            accepted.append(leaf)
    assert accepted == []


def test_load_refuses_a_removed_or_an_added_field(tmp_path):
    original = frozen().to_dict()
    path = tmp_path / "plan.json"
    accepted = []
    for dict_path in _dict_paths(original):
        target = _at(original, dict_path)
        for key in target:
            mutated = copy.deepcopy(original)
            del _at(mutated, dict_path)[key]
            if not _refused(path, mutated):
                accepted.append(("removed", *dict_path, key))
        mutated = copy.deepcopy(original)
        _at(mutated, dict_path)["added_field"] = 1
        if not _refused(path, mutated):
            accepted.append(("added", *dict_path))
    assert accepted == []


def test_load_refuses_a_plan_whose_hash_differs_from_the_recordings(tmp_path):
    path = write_plan(frozen(), tmp_path / "plan.json")
    with pytest.raises(PlanChangedError, match="recording"):
        load_plan(path, expected_sha256=frozen(seed=SEED + 1).plan_sha256)


def _exposure_with(category: str, axis: str, value: float) -> dict:
    exposure = {name: dict(row) for name, row in EXPOSURE.items()}
    exposure[category][axis] = value
    return exposure


#: One change per frozen field; schema_version is refused at construction (see below).
CHANGES = {
    "created_at": lambda p: replace(p, created_at="2026-11-11T12:00:00+00:00"),
    "seed": lambda p: replace(p, seed=p.seed + 1),
    "items": lambda p: replace(p, items={**p.items, "macro-2016-02": digest("macro-2016-02")}),
    "excluded_models": lambda p: replace(p, excluded_models={"vendor/no-cutoff": "no sourced cutoff"}),
    "months": lambda p: replace(p, months=gemma_split(p.seed + 1)),
    "edgar": lambda p: replace(p, edgar=replace(
        p.edgar, calibration_control=(p.edgar.negative_control[0], *p.edgar.calibration_control[1:]),
        negative_control=(p.edgar.calibration_control[0], *p.edgar.negative_control[1:]))),
    "edgar_files": lambda p: replace(p, edgar_files={**p.edgar_files, "held_out/oos_control.jsonl": digest("new")}),
    "accuracy.rule": lambda p: replace(p, accuracy=replace(p.accuracy, rule="macro_regime.other_rule")),
    "accuracy.exposure": lambda p: replace(p, accuracy=replace(
        p.accuracy, exposure=_exposure_with("world_equity", "growth", 0.9))),
    "accuracy.sources": lambda p: replace(p, accuracy=replace(
        p.accuracy, sources={**p.accuracy.sources, "gold_commodity": "another source"})),
    "accuracy.returns": lambda p: replace(p, accuracy=replace(p.accuracy, returns="another return rule")),
    "accuracy.outcome": lambda p: replace(p, accuracy=replace(p.accuracy, outcome="another outcome")),
    "accuracy.missing": lambda p: replace(p, accuracy=replace(p.accuracy, missing="another missing rule")),
    "analysis.metrics": lambda p: replace(p, analysis=replace(p.analysis, metrics=p.analysis.metrics[:-1])),
    "analysis.effect_size": lambda p: replace(p, analysis=replace(p.analysis, effect_size=0.2)),
    "analysis.tolerance": lambda p: replace(p, analysis=replace(p.analysis, tolerance=0.2)),
    "analysis.n_resamples": lambda p: replace(p, analysis=replace(p.analysis, n_resamples=1000)),
    "analysis.block_length": lambda p: replace(p, analysis=replace(p.analysis, block_length=3)),
    "analysis.negative_control_block_length": lambda p: replace(
        p, analysis=replace(p.analysis, negative_control_block_length=3)),
    "analysis.alpha": lambda p: replace(p, analysis=replace(p.analysis, alpha=0.01)),
    "analysis.power_threshold": lambda p: replace(p, analysis=replace(p.analysis, power_threshold=0.9)),
    "analysis.first_draw_band": lambda p: replace(p, analysis=replace(p.analysis, first_draw_band=(0.2, 0.8))),
    # The 7.3 sections, each added with the sections it depends on; tests/scripts/finance_sim/test_plan_arms.py
    # changes every field inside them.
    "arms": lambda p: replace(p, arms=plan_module.study_arms()),
    "pilot": lambda p: _with_pilot(p),
    "limits": lambda p: _with_limits(p),
    "scored_probability": lambda p: replace(p, scored_probability=plan_module.ScoredProbability()),
}


def _with_pilot(p: StudyPlan) -> StudyPlan:
    arms = plan_module.study_arms()
    return replace(p, arms=arms, pilot=plan_module.PilotSettings.for_split(p.months, arms))


def _with_limits(p: StudyPlan) -> StudyPlan:
    planned = _with_pilot(p)
    return replace(planned, limits=plan_module.size_limits(planned.months, planned.edgar, planned.arms,
                                                           planned.pilot, plan_module.load_serving_manifest()))


def test_the_change_table_covers_every_frozen_field():
    expected = {f.name for f in fields(StudyPlan)} - {"accuracy", "analysis", "schema_version"}
    expected |= {f"accuracy.{f.name}" for f in fields(AccuracyDefinition)}
    expected |= {f"analysis.{f.name}" for f in fields(AnalysisSettings)}
    assert set(CHANGES) == expected


@pytest.mark.parametrize("name", sorted(CHANGES))
def test_ensure_unchanged_refuses_a_plan_after_any_field_changes(name):
    plan = frozen()
    ensure_unchanged(plan, plan.plan_sha256)
    changed = CHANGES[name](plan)
    assert changed.plan_sha256 != plan.plan_sha256
    with pytest.raises(PlanChangedError):
        ensure_unchanged(changed, plan.plan_sha256)


def test_another_schema_version_is_refused():
    with pytest.raises(ValueError, match="schema"):
        replace(frozen(), schema_version="finance-sim-plan/2")


def test_write_plan_never_replaces_a_different_frozen_plan(tmp_path):
    path = write_plan(frozen(), tmp_path / "plan.json")
    before = path.read_bytes()
    assert write_plan(frozen(), path) == path
    with pytest.raises(PlanChangedError):
        write_plan(frozen(seed=SEED + 1), path)
    assert path.read_bytes() == before


def test_load_refuses_a_plan_whose_accuracy_exposure_differs_from_the_library(tmp_path, monkeypatch):
    path = write_plan(frozen(), tmp_path / "plan.json")
    monkeypatch.setattr(plan_module, "EXPOSURE", _exposure_with("gold_commodity", "policy", 0.2))
    with pytest.raises(PlanChangedError, match="EXPOSURE"):
        load_plan(path)


@pytest.mark.parametrize(("section", "change", "message"), [
    ("months", lambda m: m.update(pilot=sorted([*m["pilot"], m["calibration"][0]])), "disjoint"),
    ("months", lambda m: m.update(evaluation=m["evaluation"][:-1]), "every month"),
    ("months", lambda m: m.update(pilot=sorted([*m["pilot"][1:], m["evaluation"][-1]]),
                                  evaluation=sorted([*m["evaluation"][:-1], m["pilot"][0]])), "cutoff"),
    ("edgar", lambda e: e.update(negative_control=[e["calibration_control"][0], *e["negative_control"][1:]]),
     "disjoint"),
])
def test_load_validates_a_consistently_rehashed_plan(tmp_path, section, change, message):
    payload = frozen().to_dict()
    change(payload[section])
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(rehash(payload)), encoding="utf-8")
    with pytest.raises(ValueError, match=message) as raised:
        load_plan(path)
    assert not isinstance(raised.value, PlanChangedError)


def test_load_refuses_a_stale_section_hash_even_under_a_recomputed_plan_hash(tmp_path):
    payload = frozen().to_dict()
    payload["content_sha256"]["analysis"] = digest("stale")
    payload["plan_sha256"] = digest(canonical({key: value for key, value in payload.items() if key != "plan_sha256"}))
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PlanChangedError, match="analysis"):
        load_plan(path)


def test_from_dict_refuses_a_plan_hash_that_does_not_match_the_content():
    payload = frozen().to_dict()
    assert StudyPlan.from_dict(payload) == frozen()
    payload["plan_sha256"] = digest("another plan")
    with pytest.raises(PlanChangedError, match="plan_sha256"):
        StudyPlan.from_dict(payload)


def test_load_refuses_duplicate_keys_and_non_finite_numbers(tmp_path):
    path = write_plan(frozen(), tmp_path / "plan.json")
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace('"seed":', '"seed": 1, "seed":', 1), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_plan(path)
    path.write_text(text.replace('"alpha": 0.05', '"alpha": NaN', 1), encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        load_plan(path)


# ---------- boundary validation ----------

@pytest.mark.parametrize("overrides", [
    pytest.param({"created_at": datetime(2026, 11, 10, 12, 0)}, id="naive-created-at"),
    pytest.param({"items": {}}, id="no-items"),
    pytest.param({"items": {"macro-2016-01": "abc"}}, id="bad-item-hash"),
    pytest.param({"items": {"macro-2016-01": digest("a").upper()}}, id="upper-case-item-hash"),
    pytest.param({"edgar_files": {k: v for k, v in EDGAR_FILES.items() if k != "held_out/oos_control.jsonl"}},
                 id="missing-edgar-file"),
    pytest.param({"edgar_files": {**EDGAR_FILES, "other.jsonl": digest("o")}}, id="extra-edgar-file"),
    pytest.param({"excluded_models": {GEMMA: "no sourced cutoff"}}, id="excluded-model-in-split"),
    pytest.param({"excluded_models": {"vendor/x": ""}}, id="excluded-without-reason"),
])
def test_freeze_plan_validates_its_inputs(overrides):
    with pytest.raises((TypeError, ValueError)):
        frozen(**overrides)


@pytest.mark.parametrize("overrides", [
    {"effect_size": 0.0}, {"effect_size": float("nan")}, {"tolerance": -0.1}, {"n_resamples": 0},
    {"n_resamples": 2000.0}, {"block_length": 0}, {"negative_control_block_length": True}, {"alpha": 1.0},
    {"alpha": 0.0}, {"power_threshold": 1.5}, {"first_draw_band": (0.75, 0.25)}, {"first_draw_band": (0.25,)},
    {"first_draw_band": (-0.1, 0.5)}, {"metrics": ()}, {"metrics": ("did_accuracy_gap", "did_accuracy_gap")},
    {"metrics": ("Not A Name",)},
])
def test_analysis_settings_validate_their_values(overrides):
    with pytest.raises((TypeError, ValueError)):
        AnalysisSettings(**overrides)


@pytest.mark.parametrize("sha", ["", "abc", "0" * 63, "g" * 64, None])
def test_ensure_unchanged_and_load_refuse_a_malformed_recorded_hash(tmp_path, sha):
    plan = frozen()
    path = write_plan(plan, tmp_path / "plan.json")
    with pytest.raises((TypeError, ValueError)):
        ensure_unchanged(plan, sha)
    if sha is not None:
        with pytest.raises(ValueError):
            load_plan(path, expected_sha256=sha)
