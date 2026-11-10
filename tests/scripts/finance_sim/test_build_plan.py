"""The plan-building CLI on recorded responses (task 7.3; Requirements 8.1, 8.2, 8.4, 8.10).

``build_plan.main`` runs the real item assembly (7.1) against the recorded-response
fakes of test_items.py: FRED is served at the HTTP adapter, FMP and DTB3 through the
injectable transport, and the keys come from a fake process environment. Every
output goes to a temporary private root; nothing is written under ``runs/``, no
model host is contacted and no credential is read from a file.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import shutil
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from deploy.modal.manifest import load_plan as load_modal_plan
from deploy.modal.stage_limits import STAGES, parse_stage_limits
from recall_guard.core.profiles import load_profile
from recall_guard.dataset import fred
from recall_guard.dataset.fred import RollingRateLimiter
from scripts.finance_sim import build_plan, items
from scripts.finance_sim import limits as study_limits
from scripts.finance_sim import plan as plan_module
from scripts.finance_sim.plan import load_plan, require_complete, sha256_of
from tests.scripts.finance_sim.test_items import (
    FMP_KEY,
    FRED_KEY,
    NO_CUTOFF,
    FakeClock,
    FakeFred,
    RecordedTransport,
    assert_no_key,
    fred_routes,
)

GEMMA = "google/gemma-3-4b-it"
ENVIRON = {"FRED_API_KEY": FRED_KEY, "FMP_API_TOKEN": FMP_KEY}
TODAY = date(2026, 11, 10)
CREATED_AT = "2026-11-10T12:00:00+00:00"
SEED = 20261110
ROOT = Path(plan_module.__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "configs" / "modal" / "gemma-3-4b-measurement.json"
ACCESSION = re.compile(r"edgar-(calibration|negative_control|unused)-(\d{10}-\d{2}-\d{6})")


def arguments(root: Path, *extra: str) -> list[str]:
    return ["--private-root", str(root), "--created-at", CREATED_AT, "--seed", str(SEED), *extra]


def run_main(argv: list[str], **kwargs) -> SimpleNamespace:
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = build_plan.main(argv, **kwargs)
    return SimpleNamespace(code=code, stdout=out.getvalue(), stderr=err.getvalue())


def install_fakes(patch: pytest.MonkeyPatch, working: Path) -> SimpleNamespace:
    clock = FakeClock()
    patch.setattr(fred, "RATE_LIMITER", RollingRateLimiter(clock=clock, sleep=clock.sleep))
    fake = FakeFred(fred_routes)
    patch.setattr(requests.adapters.HTTPAdapter, "send",
                  lambda adapter, prepared, **kwargs: fake.send(adapter, prepared, **kwargs))
    working.mkdir()
    patch.chdir(working)
    return SimpleNamespace(fred=fake, transport=RecordedTransport(), working=working)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """One full build of the 126 decision months on the recorded responses."""
    base = tmp_path_factory.mktemp("build")
    with pytest.MonkeyPatch.context() as patch:
        fakes = install_fakes(patch, base / "working")
        root = base / "private"
        result = run_main(arguments(root), environ=ENVIRON, transport=fakes.transport, today=TODAY)
        assert result.code == 0, result.stderr
        receipt = json.loads((root / "plan" / "receipt.json").read_text("utf-8"))
        plan = load_plan(root / "plan" / "plan.json", expected_sha256=receipt["plan_sha256"])
        yield SimpleNamespace(result=result, root=root, base=base, receipt=receipt, plan=plan, **vars(fakes))


@pytest.fixture
def world(monkeypatch, tmp_path):
    fakes = install_fakes(monkeypatch, tmp_path / "working")
    return SimpleNamespace(root=tmp_path / "private", tmp=tmp_path, **vars(fakes))


def no_request(world) -> bool:
    return world.fred.calls == [] and world.transport.calls == []


# ---------- the frozen plan and its files ----------

def test_the_cli_freezes_a_complete_plan_and_reports_its_plan_sha256(built):
    summary = json.loads(built.result.stdout)
    assert summary["status"] == "frozen"
    assert summary["plan_sha256"] == built.receipt["plan_sha256"] == built.plan.plan_sha256
    assert require_complete(built.plan) is built.plan
    assert built.plan.created_at == CREATED_AT and built.plan.seed == SEED
    assert built.receipt["status"] == "frozen" and built.receipt["files"]["plan"] == "plan/plan.json"


def test_every_output_stays_under_the_private_root_and_no_key_is_written(built):
    files = {path.relative_to(built.root).as_posix() for path in built.root.rglob("*") if path.is_file()}
    expected = {"plan/items.json", "plan/plan.json", "plan/receipt.json", "data/realized_returns.json",
                *(f"plan/stage_limits/{stage}.json" for stage in STAGES)}
    assert expected <= files
    assert all(name in expected or name.startswith(("fred/", "data/fmp/")) for name in files)
    assert sorted(path.name for path in built.base.iterdir()) == ["private", "working"]
    assert list(built.working.iterdir()) == []
    assert_no_key(built.base)


def test_the_plan_is_referenced_by_plan_sha256_never_by_its_file_hash(built):
    file_hash = hashlib.sha256((built.root / "plan" / "plan.json").read_bytes()).hexdigest()
    assert file_hash != built.plan.plan_sha256
    for path in built.root.joinpath("plan").rglob("*.json"):
        assert file_hash not in path.read_text("utf-8")


def test_edgar_files_are_filled_from_the_manifest_with_recomputed_hashes(built):
    manifest = json.loads((items.DEFAULT_EDGAR_DIR / "manifest.json").read_text("utf-8"))
    recomputed = {name: hashlib.sha256((items.DEFAULT_EDGAR_DIR / name).read_bytes()).hexdigest()
                  for name in plan_module.EDGAR_FILES}
    assert dict(built.plan.edgar_files) == recomputed
    assert recomputed == {name: manifest["files"][name]["sha256"] for name in plan_module.EDGAR_FILES}
    for name, entry in built.receipt["edgar_files"].items():
        assert entry["sha256"] == entry["manifest_sha256"] == recomputed[name]
        assert entry["rows"] == manifest["files"][name]["rows"]


def test_edgar_row_ids_derive_from_the_accession_and_are_unique_across_the_four_files(built):
    edgar = built.plan.edgar
    groups = {"calibration_memorized": "calibration", "calibration_control": "calibration",
              "negative_control": "negative_control", "unused": "unused"}
    accessions = []
    for group, role in groups.items():
        ids = getattr(edgar, group)
        for row_id in ids:
            match = ACCESSION.fullmatch(row_id)
            assert match and match.group(1) == role
            accessions.append(match.group(2))
    assert (len(edgar.calibration_memorized), len(edgar.calibration_control), len(edgar.negative_control),
            len(edgar.unused)) == (40, 100, 40, 40)
    assert len(set(accessions)) == len(accessions) == 220
    record = json.loads((built.root / "plan" / "items.json").read_text("utf-8"))["edgar"]
    assert [entry["item_id"] for entry in record["negative_control"]] == list(edgar.negative_control)


def test_the_items_cover_every_month_and_every_edgar_id_with_their_row_and_prompt_hashes(built):
    plan = built.plan
    record = json.loads((built.root / "plan" / "items.json").read_text("utf-8"))
    expected = {f"macro-{month}" for month in plan.months.months}
    expected |= set(plan.edgar.calibration_memorized) | set(plan.edgar.calibration_control)
    expected |= set(plan.edgar.negative_control)
    assert set(plan.items) == expected and len(plan.months.months) == 126
    entries = record["macro_items"] + record["edgar"]["calibration"] + record["edgar"]["negative_control"]
    assert len(entries) == len(plan.items)
    for entry in entries:
        assert plan.items[entry["item_id"]] == sha256_of({"prompt_sha256": entry["prompt_sha256"],
                                                          "row_sha256": entry["row_sha256"]})
    assert built.receipt["files"]["items"]["sha256"] == hashlib.sha256(
        (built.root / "plan" / "items.json").read_bytes()).hexdigest()


def test_the_plan_freezes_the_arm_to_registry_id_map_and_takes_the_arms_from_their_profiles(built):
    arms = built.plan.arms
    assert dict(arms.registry_ids) == {"sampling": GEMMA, "serving": GEMMA} == built.receipt["arms"]
    assert arms.arms["sampling"].served_model == "gemma3:4b" and arms.arms["serving"].served_model == GEMMA
    facts = build_plan.arm_facts()
    assert facts == {arm: dict(value) for arm, value in plan_module.ARM_PROFILE_FACTS.items()}
    assert built.plan.excluded_models == {}


def test_the_plan_freezes_the_gguf_digest_the_profile_declares_and_the_receipt_names_the_roles(built):
    declared = load_profile("llamacpp-gemma3-4b").model_config.local_identity.declared.artifact_digest
    assert built.plan.arms.arms["sampling"].artifact_digest == declared
    assert built.plan.arms.arms["serving"].artifact_digest is None
    assert build_plan.arm_facts()["sampling"]["artifact_digest"] == declared
    assert built.receipt["arm_roles"] == {"sampling": "probe", "serving": "primary"}


def test_a_profile_whose_full_distribution_host_keeps_another_count_is_refused(monkeypatch):
    original = build_plan.load_profile

    def other_kept(profile_id):
        profile = original(profile_id)
        if profile.full_distribution is None:
            return profile
        return replace(profile, full_distribution=replace(profile.full_distribution, top_k_keep=50))

    monkeypatch.setattr(build_plan, "load_profile", other_kept)
    with pytest.raises(build_plan.PlanBuildError, match="top_k_keep"):
        build_plan.arm_facts()


def test_the_frozen_split_and_pilot_have_the_design_sizes(built):
    plan = built.plan
    assert (len(plan.months.calibration), len(plan.months.pilot), len(plan.months.evaluation)) == (24, 12, 90)
    # The pilot gate simulates the probe arm's own full run: its probe evaluation months.
    assert (plan.pilot.pre_cutoff_evaluation_months, plan.pilot.post_cutoff_evaluation_months) == (15, 5)
    counts = built.receipt["counts"]
    assert counts["months"] == 126 and counts["edgar_calibration"] == 140 and counts["negative_control"] == 40
    assert (counts["pre_cutoff_evaluation"], counts["post_cutoff_evaluation"]) == (68, 22)
    assert len(plan.months.probe_evaluation) == counts["probe_evaluation"] == 20
    assert (counts["probe_evaluation_pre_cutoff"], counts["probe_evaluation_post_cutoff"]) == (15, 5)
    assert [entry["max_requests"] for entry in built.receipt["sampling_limits"]] == [944, 192, 320, 320]


def test_the_stage_limits_files_carry_the_plan_sha256_in_execution_order(built):
    modal = load_modal_plan(MANIFEST_PATH)
    serving = built.receipt["serving_limits"]
    assert [entry["stage"] for entry in serving["stages"]] == list(STAGES)
    for entry, (stage, data) in zip(serving["stages"], plan_module.stage_limits_files(built.plan), strict=True):
        path = built.root / entry["path"]
        assert entry["path"] == f"plan/stage_limits/{stage}.json" and path.read_bytes() == data
        assert entry["sha256"] == hashlib.sha256(data).hexdigest()
        assert parse_stage_limits(path.read_bytes(), modal).plan_sha256 == built.plan.plan_sha256


def test_the_receipt_serving_limits_agree_with_the_task_7_5_presentation(built):
    paths = [built.root / "plan" / "stage_limits" / f"{stage}.json" for stage in STAGES]
    presentation = study_limits.present_serving_arm(built.plan, MANIFEST_PATH, paths)
    serving = built.receipt["serving_limits"]
    record = plan_module.serving_limits_record(built.plan)
    assert {key: value for key, value in serving.items() if key != "stages"} == {
        key: value for key, value in record.items() if key != "stages"}
    assert serving["stages"] == [entry | {"path": f"plan/stage_limits/{entry['stage']}.json"}
                                 for entry in record["stages"]]
    assert (serving["plan_sha256"], serving["manifest_sha256"], serving["stage_limits_sha256"]) == (
        presentation.plan_sha256, presentation.manifest_sha256, presentation.stage_limits_sha256)
    assert serving["manifest_sha256"] == hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()
    assert serving["max_usd_total"] == presentation.max_usd_total


def test_the_receipt_names_the_credentials_by_variable_only(built):
    assert built.receipt["credentials"] == {"FRED_API_KEY": "process environment, by name",
                                            "FMP_API_TOKEN": "process environment, by name"}
    text = (built.root / "plan" / "receipt.json").read_text("utf-8")
    assert FRED_KEY not in text and FMP_KEY not in text


def test_a_second_build_is_refused_before_any_request_and_changes_nothing(built):
    before = {path: path.read_bytes() for path in built.root.joinpath("plan").rglob("*") if path.is_file()}
    calls = (len(built.fred.calls), len(built.transport.calls))
    again = run_main(["--private-root", str(built.root), "--created-at", "2026-11-11T12:00:00+00:00"],
                     environ=ENVIRON, transport=built.transport, today=TODAY)
    assert again.code == 1 and "already" in again.stderr and built.plan.plan_sha256 in again.stderr
    assert (len(built.fred.calls), len(built.transport.calls)) == calls
    assert {path: path.read_bytes() for path in built.root.joinpath("plan").rglob("*") if path.is_file()} == before


# ---------- refusals before any request ----------

def test_an_arm_without_an_included_model_is_refused_before_any_request(world):
    with pytest.raises(build_plan.PlanBuildError, match="arm 'serving' has no included model"):
        build_plan.build(environ=ENVIRON, private_root=world.root, registry_ids={"sampling": GEMMA,
                                                                                 "serving": NO_CUTOFF},
                         transport=world.transport, today=TODAY)
    assert no_request(world) and not world.root.exists()


def test_a_registry_map_that_does_not_name_both_arms_is_refused(world):
    with pytest.raises(build_plan.PlanBuildError, match="both arms"):
        build_plan.build(environ=ENVIRON, private_root=world.root, registry_ids={"sampling": GEMMA},
                         transport=world.transport, today=TODAY)
    assert no_request(world)


@pytest.mark.parametrize("missing", ["FRED_API_KEY", "FMP_API_TOKEN"])
def test_a_missing_key_is_refused_naming_only_the_variable(world, missing):
    environ = {name: value for name, value in ENVIRON.items() if name != missing}
    result = run_main(arguments(world.root), environ=environ, transport=world.transport, today=TODAY)
    assert result.code == 1 and missing in result.stderr
    assert FRED_KEY not in result.stderr and FMP_KEY not in result.stderr
    assert no_request(world) and not world.root.exists()


def _edgar_copy(tmp_path: Path) -> Path:
    copy = tmp_path / "edgar"
    shutil.copytree(items.DEFAULT_EDGAR_DIR, copy)
    return copy


def _rehash_manifest(edgar: Path, name: str) -> None:
    manifest = json.loads((edgar / "manifest.json").read_text("utf-8"))
    manifest["files"][name]["sha256"] = hashlib.sha256((edgar / name).read_bytes()).hexdigest()
    (edgar / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_an_edgar_file_that_differs_from_its_manifest_is_refused_before_any_request(world):
    edgar = _edgar_copy(world.tmp)
    target = edgar / "held_out" / "is_memorized.jsonl"
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(build_plan.PlanBuildError, match="held_out/is_memorized.jsonl.*manifest sha256"):
        build_plan.build(environ=ENVIRON, private_root=world.root, edgar_dir=edgar, transport=world.transport,
                         today=TODAY)
    assert no_request(world) and not world.root.exists()


def test_an_accession_repeated_across_the_four_files_is_refused_before_any_request(world):
    edgar = _edgar_copy(world.tmp)
    first = (edgar / "calibration" / "is_memorized.jsonl").read_bytes().split(b"\n")[0]
    target = edgar / "held_out" / "is_memorized.jsonl"
    lines = target.read_bytes().split(b"\n")
    target.write_bytes(b"\n".join([first, *lines[1:]]))
    _rehash_manifest(edgar, "held_out/is_memorized.jsonl")
    with pytest.raises(build_plan.PlanBuildError, match="accession .* appears in both"):
        build_plan.build(environ=ENVIRON, private_root=world.root, edgar_dir=edgar, transport=world.transport,
                         today=TODAY)
    assert no_request(world)


def test_a_private_root_inside_the_repository_outside_runs_or_a_drive_root_is_refused(world):
    for root in (ROOT / "docs" / "finance-sim-private", Path(ROOT.anchor)):
        with pytest.raises(build_plan.PlanBuildError, match="private root"):
            build_plan.build(environ=ENVIRON, private_root=root, transport=world.transport, today=TODAY)
    assert not (ROOT / "docs" / "finance-sim-private").exists()
    assert no_request(world)


def test_a_created_at_without_a_timezone_is_refused_by_the_cli(world):
    result = run_main(["--private-root", str(world.root), "--created-at", "2026-11-10T12:00:00"],
                      environ=ENVIRON, transport=world.transport, today=TODAY)
    assert result.code == 1 and "created_at" in result.stderr
    assert no_request(world)


def test_a_source_failure_exits_with_the_source_error_code_and_writes_no_plan(world):
    world.transport.routes = lambda url, params: (503, f'{{"Error Message": "busy for {FMP_KEY}"}}'.encode())
    result = run_main(arguments(world.root), environ=ENVIRON, transport=world.transport, today=TODAY)
    assert result.code == 3 and "FMP SWDA.L" in result.stderr and "retryable True" in result.stderr
    assert FMP_KEY not in result.stderr
    assert not (world.root / "plan" / "plan.json").exists()


# ---------- coverage check (offline) ----------

def _offline_plan(items_map=None):
    from tests.scripts.finance_sim.test_plan import CREATED_AT as PLAN_CREATED_AT
    from tests.scripts.finance_sim.test_plan import EDGAR_FILES, GEMMA_CUTOFF, classes, digest, edgar_rows

    rows = edgar_rows()
    months = classes({GEMMA: GEMMA_CUTOFF})
    if items_map is None:
        ids = [f"macro-{item.month}" for item in months] + [row.row_id for row in rows if row.row_id.startswith(
            ("cal-", "held-ctl-"))]
        items_map = {item_id: digest(item_id) for item_id in ids}
    return plan_module.freeze_plan(months, rows, seed=SEED, created_at=PLAN_CREATED_AT, items=items_map,
                                   edgar_files=EDGAR_FILES)


def test_the_coverage_check_accepts_items_for_every_month_and_edgar_id():
    plan = _offline_plan()
    build_plan.check_coverage(plan)


@pytest.mark.parametrize(("change", "message"), [
    (lambda ids: ids - {"macro-2019-05"}, "no item for .*macro-2019-05"),
    (lambda ids: ids - {"held-ctl-003"}, "no item for .*held-ctl-003"),
    (lambda ids: ids | {"macro-2030-01"}, "not in the plan.*macro-2030-01"),
    (lambda ids: ids | {"held-mem-000"}, "not in the plan.*held-mem-000"),
])
def test_the_coverage_check_refuses_a_missing_or_an_extra_item(change, message):
    from tests.scripts.finance_sim.test_plan import digest

    complete = _offline_plan()
    ids = change(set(complete.items))
    plan = replace(complete, items={item_id: digest(item_id) for item_id in sorted(ids)})
    with pytest.raises(build_plan.PlanBuildError, match=message):
        build_plan.check_coverage(plan)
