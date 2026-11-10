"""Serving-arm presentation, approval record and cost limit (task 7.5; Requirements 11.2, 11.3).

Pure logic on local files under ``tmp_path``: a copy of the reviewed Gemma 3 4B
serving manifest with an explicit endpoint, six stage-limit files carrying the
frozen plan's hash, and approval records written by the tests. No network,
credentials, Modal SDK or model host is used.
"""

from __future__ import annotations

import hashlib
import json
import socket
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from deploy.modal.manifest import load_plan as load_modal_plan
from deploy.modal.stage_limits import load_stage_limits
from scripts.finance_sim import limits as limits_module
from scripts.finance_sim.limits import (
    APPROVAL_SCHEMA,
    EXIT_INCOMPLETE,
    EXIT_REFUSED,
    AdmissionClosedError,
    ApprovalError,
    BudgetLedger,
    RequestAllowance,
    StudyLimits,
    approval_path,
    approval_template,
    present_serving_arm,
    require_serving_approval,
)
from scripts.finance_sim.plan import (
    EDGAR_FILES,
    EdgarRow,
    MonthClass,
    PlanChangedError,
    StudyPlan,
    freeze_plan,
    write_plan,
)
from scripts.probe_modal_model import APPROVAL_ASSERTIONS

ROOT = Path(__file__).resolve().parents[3]
GEMMA_MANIFEST = ROOT / "configs" / "modal" / "gemma-3-4b-measurement.json"
PILOT_MANIFEST = ROOT / "configs" / "modal" / "gpt-oss-20b-pilot.json"
ENDPOINT = "https://wishinet--recall-guard-gemma-3-4b.modal.run/v1"
GEMMA = "google/gemma-3-4b-it"
CREATED_AT = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)
SEED = 20261110
#: Per stage, in execution order: max_requests, max_output_tokens, max_seconds, max_usd.
STAGE_SIZES = {
    "weight_staging": (0, 0, 3600, 0.32),
    "deployment": (1, 256, 900, 0.4),
    "smoke_run": (32, 8192, 900, 0.2),
    "calibration": (6016, 1540096, 37296, 15.0),
    "full_run": (5760, 1474560, 35160, 14.0),
    "negative_control": (1280, 327680, 8280, 3.5),
}
# Configured Modal rates of the Gemma manifest: A10 + 4 cores + 16 GiB, plus the expiry controller.
RATE = 0.000306 + 4 * 0.0000131 + 16 * 0.00000222 + 0.125 * 0.0000131 + 0.125 * 0.00000222


def month_range(first: str, last: str) -> list[str]:
    year, month = int(first[:4]), int(first[5:])
    out = []
    while f"{year:04d}-{month:02d}" <= last:
        out.append(f"{year:04d}-{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return out


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def frozen_plan(seed: int = SEED) -> StudyPlan:
    months = [MonthClass(month, {GEMMA: month <= "2024-08"}) for month in month_range("2016-01", "2026-06")]
    rows = [EdgarRow(f"cal-mem-{i:03d}", 1, "calibration") for i in range(40)]
    rows += [EdgarRow(f"cal-ctl-{i:03d}", 0, "calibration") for i in range(100)]
    rows += [EdgarRow(f"held-ctl-{i:03d}", 0, "held_out") for i in range(40)]
    return freeze_plan(months, rows, seed=seed, created_at=CREATED_AT,
                       items={"macro-2016-01": digest(b"macro")},
                       edgar_files={name: digest(name.encode()) for name in EDGAR_FILES})


def write_stage_files(directory: Path, plan_sha256: str, sizes: dict = STAGE_SIZES) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for stage, (requests, tokens, seconds, usd) in sizes.items():
        path = directory / f"{stage}.json"
        path.write_text(json.dumps({"plan_sha256": plan_sha256, "stage": stage, "max_requests": requests,
                                    "max_output_tokens": tokens, "max_seconds": seconds, "max_usd": usd}),
                        encoding="utf-8")
        paths.append(path)
    return paths


class Clock:
    def __init__(self) -> None:
        self.now = 50.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def serving(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("the approval gate attempted a network connection")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    plan = frozen_plan()
    raw = json.loads(GEMMA_MANIFEST.read_text(encoding="utf-8"))
    raw["target"]["endpoint"] = ENDPOINT
    manifest = tmp_path / "gemma-3-4b-measurement.json"
    manifest.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    return SimpleNamespace(plan=plan, manifest=manifest, stages=write_stage_files(tmp_path / "stages", plan.plan_sha256),
                           approvals=tmp_path / "private" / "approvals", tmp=tmp_path)


def approved_record(serving, **overrides) -> dict:
    """The operator's record: the template, every assertion set to true after review, and the acknowledgement."""
    presentation = present_serving_arm(serving.plan, serving.manifest, serving.stages)
    return approval_template(presentation) | dict.fromkeys(APPROVAL_ASSERTIONS, True) | {
        "provider_budget_reference": "Modal workspace budget reviewed 2026-11-10 (operator note R-7.5)",
        "approved_by": "operator", "approved_at": "2026-11-10T12:30:00+00:00"} | overrides


def write_record(serving, record: dict, plan_sha256: str | None = None) -> Path:
    path = approval_path(plan_sha256 or serving.plan.plan_sha256, serving.approvals)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return path


def require(serving):
    return require_serving_approval(serving.plan, serving.manifest, serving.stages, approvals_dir=serving.approvals)


# ---------- presentation of the paid arm (11.2) ----------

def test_presentation_lists_requests_token_limits_time_and_maximum_cost_per_stage_in_execution_order(serving):
    presentation = present_serving_arm(serving.plan, serving.manifest, serving.stages)
    assert [stage.stage for stage in presentation.stages] == list(STAGE_SIZES)
    assert [(s.max_requests, s.max_output_tokens, s.max_seconds, s.max_usd) for s in presentation.stages] == \
        list(STAGE_SIZES.values())
    assert presentation.total_requests == 13089
    assert presentation.total_output_tokens == 3350784
    assert presentation.total_seconds == 86136
    assert presentation.max_usd_total == 33.42
    assert presentation.output_tokens_per_request == 256
    assert presentation.usd_per_second == pytest.approx(RATE, rel=1e-12)
    assert presentation.plan_sha256 == serving.plan.plan_sha256
    assert presentation.manifest == "configs/modal/gemma-3-4b-measurement.json"
    assert presentation.manifest_sha256 == digest(serving.manifest.read_bytes())
    files = [[path.stem, digest(path.read_bytes())] for path in serving.stages]
    assert [stage.sha256 for stage in presentation.stages] == [sha for _, sha in files]
    canonical = json.dumps(files, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    assert presentation.stage_limits_sha256 == digest(canonical.encode("utf-8"))
    assert (presentation.workspace, presentation.endpoint) == ("wishinet", ENDPOINT)


def test_the_rendered_presentation_names_every_stage_the_totals_and_the_record_to_write(serving):
    presentation = present_serving_arm(serving.plan, serving.manifest, serving.stages)
    record = approval_path(serving.plan.plan_sha256, serving.approvals)
    text = presentation.render(record)
    positions = [text.index(stage) for stage in STAGE_SIZES]
    assert positions == sorted(positions)
    for fragment in ("13089", "3350784", "86136", "33.42", "256", serving.plan.plan_sha256,
                     presentation.manifest_sha256, presentation.stage_limits_sha256, str(record), "paid"):
        assert fragment in text
    assert "—" not in text


@pytest.mark.parametrize("change,error,message", [
    (lambda files: files[:5], ValueError, "one stage limits file per stage"),
    (lambda files: [files[0], files[2], files[1], *files[3:]], ValueError, "execution order"),
    (lambda files: [*files, files[0]], ValueError, "one stage limits file per stage"),
])
def test_presentation_refuses_missing_extra_or_reordered_stage_files(serving, change, error, message):
    with pytest.raises(error, match=message):
        present_serving_arm(serving.plan, serving.manifest, change(serving.stages))


def test_presentation_refuses_stage_files_of_another_plan(serving):
    other = frozen_plan(seed=SEED + 1)
    stages = write_stage_files(serving.tmp / "other", other.plan_sha256)
    with pytest.raises(PlanChangedError):
        present_serving_arm(serving.plan, serving.manifest, stages)


def test_presentation_refuses_stages_that_together_exceed_the_manifest_ceiling(serving):
    sizes = dict(STAGE_SIZES) | {"calibration": (8200, 2099200, 37296, 15.0)}
    stages = write_stage_files(serving.tmp / "big", serving.plan.plan_sha256, sizes)
    with pytest.raises(ValueError, match="max_requests.*reviewed ceiling"):
        present_serving_arm(serving.plan, serving.manifest, stages)
    sizes = dict(STAGE_SIZES) | {"full_run": (5760, 1474560, 35160, 15.5)}
    stages = write_stage_files(serving.tmp / "costly", serving.plan.plan_sha256, sizes)
    with pytest.raises(ValueError, match="max_usd.*reviewed ceiling"):
        present_serving_arm(serving.plan, serving.manifest, stages)


def test_presentation_refuses_a_manifest_other_than_the_reviewed_gemma_serving_manifest(serving):
    with pytest.raises(ValueError, match="Gemma"):
        present_serving_arm(serving.plan, PILOT_MANIFEST, serving.stages)


# ---------- approval record (11.2): refusal without or with a mismatched record ----------

def test_the_serving_arm_is_refused_without_an_approval_record(serving):
    with pytest.raises(ApprovalError) as refused:
        require(serving)
    message = str(refused.value)
    assert "Serving arm refused" in message and "no approval record" in message
    assert str(approval_path(serving.plan.plan_sha256, serving.approvals)) in message
    assert not serving.approvals.exists()


def test_a_reviewed_record_matching_the_frozen_plan_approves_the_serving_arm(serving):
    path = write_record(serving, approved_record(serving))
    approval = require(serving)
    assert approval.path == path
    assert approval.plan_sha256 == serving.plan.plan_sha256
    assert approval.record["max_usd_total"] == 33.42
    assert approval.limits_for("full_run") == StudyLimits(serving.plan.plan_sha256, "serving", "full_run",
                                                          5760, 1474560, 35160, 14.0)
    assert approval.presentation == present_serving_arm(serving.plan, serving.manifest, serving.stages)
    # The approved limits of every stage are exactly its StageLimits file's.
    modal_plan = load_modal_plan(serving.manifest)
    for path in serving.stages:
        stage_limits = load_stage_limits(path, modal_plan)
        assert StudyLimits.from_stage_limits(stage_limits) == approval.limits_for(stage_limits.stage)


def test_the_template_alone_is_refused_until_the_operator_acknowledges_it(serving):
    presentation = present_serving_arm(serving.plan, serving.manifest, serving.stages)
    template = approval_template(presentation)
    assert template["schema"] == APPROVAL_SCHEMA
    assert all(template[key] is False for key in APPROVAL_ASSERTIONS)
    assert template["presented"]["stages"][4]["max_requests"] == 5760
    write_record(serving, template)
    with pytest.raises(ApprovalError, match="paid_execution_authorized"):
        require(serving)


def test_a_record_naming_another_plan_is_refused_with_both_hashes(serving):
    other = frozen_plan(seed=SEED + 1)
    write_record(serving, approved_record(serving, plan_sha256=other.plan_sha256))
    with pytest.raises(ApprovalError) as refused:
        require(serving)
    assert other.plan_sha256 in str(refused.value) and serving.plan.plan_sha256 in str(refused.value)


def test_a_changed_plan_finds_no_approval_even_when_the_old_record_is_kept(serving):
    write_record(serving, approved_record(serving))
    require(serving)
    changed = frozen_plan(seed=SEED + 1)
    stages = write_stage_files(serving.tmp / "changed", changed.plan_sha256)
    with pytest.raises(ApprovalError, match="no approval record"):
        require_serving_approval(changed, serving.manifest, stages, approvals_dir=serving.approvals)
    # Copying the old record to the new plan's path is refused too.
    write_record(serving, approved_record(serving), plan_sha256=changed.plan_sha256)
    with pytest.raises(ApprovalError, match="names plan_sha256"):
        require_serving_approval(changed, serving.manifest, stages, approvals_dir=serving.approvals)


def test_stage_limits_changed_after_approval_are_refused(serving):
    write_record(serving, approved_record(serving))
    sizes = dict(STAGE_SIZES) | {"full_run": (5000, 1280000, 35160, 14.0)}
    write_stage_files(serving.tmp / "stages", serving.plan.plan_sha256, sizes)
    with pytest.raises(ApprovalError, match="stage_limits_sha256"):
        require(serving)


def test_a_changed_manifest_after_approval_is_refused(serving):
    write_record(serving, approved_record(serving))
    serving.manifest.write_text(serving.manifest.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ApprovalError, match="manifest_sha256"):
        require(serving)


MISMATCHES = [
    ({"manifest_sha256": "0" * 64}, "manifest_sha256"),
    ({"stage_limits_sha256": "0" * 64}, "stage_limits_sha256"),
    ({"paid_execution_authorized": False}, "paid_execution_authorized"),
    ({"pricing_and_budget_reviewed": "yes"}, "pricing_and_budget_reviewed"),
    ({"workspace": "someone-else"}, "workspace"),
    ({"endpoint": "https://other--app.modal.run/v1"}, "endpoint"),
    ({"provider_budget_reference": ""}, "provider_budget_reference"),
    ({"max_usd_total": 33.0}, "max_usd_total"),
    ({"max_usd_total": True}, "max_usd_total"),
    ({"max_usd_total": "40"}, "max_usd_total"),
    ({"approved_by": "  "}, "approved_by"),
    ({"approved_at": "2026-11-10T12:30:00"}, "approved_at"),
    ({"approved_at": "yesterday"}, "approved_at"),
    ({"schema": "finance-sim-serving-approval/0"}, "schema"),
    ({"monthly_budget": {}}, "undeclared"),
]


@pytest.mark.parametrize("overrides,message", MISMATCHES, ids=[message for _, message in MISMATCHES])
def test_a_mismatched_record_refuses_the_serving_arm(serving, overrides, message):
    write_record(serving, approved_record(serving, **overrides))
    with pytest.raises(ApprovalError, match=message) as refused:
        require(serving)
    assert str(refused.value).startswith("Serving arm refused")


def test_a_record_whose_presented_limits_differ_is_refused(serving):
    record = approved_record(serving)
    record["presented"]["stages"][4]["max_requests"] = 9000
    write_record(serving, record)
    with pytest.raises(ApprovalError, match="presented"):
        require(serving)


def test_a_record_missing_the_acknowledgement_is_refused(serving):
    record = approved_record(serving)
    del record["approved_by"]
    write_record(serving, record)
    with pytest.raises(ApprovalError, match="missing"):
        require(serving)


@pytest.mark.parametrize("edit", [
    lambda text: text.replace('"max_usd_total": 33.42', '"max_usd_total": NaN'),
    lambda text: text.replace('"workspace": "wishinet"', '"workspace": "wishinet", "workspace": "wishinet"'),
    lambda text: text[:-2],
])
def test_an_unreadable_record_is_refused(serving, edit):
    path = write_record(serving, approved_record(serving))
    path.write_text(edit(path.read_text(encoding="utf-8")), encoding="utf-8")
    with pytest.raises(ApprovalError, match="unreadable"):
        require(serving)


def test_a_manifest_without_an_endpoint_cannot_be_approved(serving):
    record = approved_record(serving)
    raw = json.loads(serving.manifest.read_text(encoding="utf-8"))
    raw["target"]["endpoint"] = None
    serving.manifest.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    presentation = present_serving_arm(serving.plan, serving.manifest, serving.stages)
    write_record(serving, record | {"manifest_sha256": presentation.manifest_sha256, "endpoint": None,
                                    "presented": approval_template(presentation)["presented"]})
    with pytest.raises(ApprovalError, match="endpoint"):
        require(serving)


# ---------- the serving ledger needs the approval and stops on the cost limit (11.2, 11.3) ----------

def serving_allowance(approval) -> RequestAllowance:
    return RequestAllowance(output_tokens=256, timeout_s=120, usd_per_second=approval.presentation.usd_per_second)


def test_a_serving_ledger_is_refused_without_a_validated_approval(serving):
    limits = StudyLimits(serving.plan.plan_sha256, "serving", "full_run", 5760, 1474560, 35160, 14.0)
    allowance = RequestAllowance(output_tokens=256, timeout_s=120, usd_per_second=RATE)
    path = serving.tmp / "records" / "serving" / "full_run" / "budget.json"
    with pytest.raises(ApprovalError, match="Serving arm refused"):
        BudgetLedger.open(path, serving.plan, limits, allowance)
    with pytest.raises(ApprovalError, match="Serving arm refused"):
        BudgetLedger.open(path, serving.plan, limits, allowance, approval={"approved": True})
    assert not path.exists()


def test_a_serving_ledger_refuses_limits_cost_basis_or_plan_other_than_the_approved_ones(serving):
    write_record(serving, approved_record(serving))
    approval = require(serving)
    path = serving.tmp / "budget.json"
    approved = approval.limits_for("full_run")
    with pytest.raises(ApprovalError, match="differ from the approved"):
        BudgetLedger.open(path, serving.plan, replace(approved, max_requests=6000), serving_allowance(approval),
                          approval=approval)
    with pytest.raises(ApprovalError, match="cost basis"):
        BudgetLedger.open(path, serving.plan, approved, replace(serving_allowance(approval), usd_per_second=RATE * 2),
                          approval=approval)
    other = frozen_plan(seed=SEED + 1)
    with pytest.raises(ApprovalError, match="plan"):
        BudgetLedger.open(path, other, replace(approved, plan_sha256=other.plan_sha256), serving_allowance(approval),
                          approval=approval)
    assert not path.exists()
    ledger = BudgetLedger.open(path, serving.plan, approved, serving_allowance(approval), approval=approval,
                               clock=Clock())
    assert ledger.status == "open" and ledger.counters["estimated_usd"] == 0.0


def test_the_cost_limit_closes_serving_admission_and_marks_the_stage_incomplete(serving):
    write_record(serving, approved_record(serving))
    approval = require(serving)
    clock = Clock()
    path = serving.tmp / "records" / "serving" / "smoke_run" / "budget.json"
    # smoke_run: 32 requests, 8,192 output tokens, 900 s, 0.20 USD. At 60 s per request a request fits
    # only while (elapsed + its 120 s timeout) * rate <= 0.20 USD, which is elapsed <= about 385 s.
    ledger = BudgetLedger.open(path, serving.plan, approval.limits_for("smoke_run"), serving_allowance(approval),
                               approval=approval, clock=clock)
    admitted = 0
    with pytest.raises(AdmissionClosedError) as closed:
        while admitted <= 32:
            admission = ledger.admit()
            admitted += 1
            (path.parent / f"smoke-{admitted:02d}.jsonl").write_text("{}\n", encoding="utf-8")
            clock.advance(60)
            ledger.finish(admission, output_tokens=64)
    assert admitted == 7
    assert (closed.value.reason, ledger.status, ledger.stop_reason) == ("max_usd", "incomplete", "max_usd")
    assert ledger.exit_code == EXIT_INCOMPLETE
    counters = ledger.counters
    assert (counters["requests"], counters["output_tokens"], counters["elapsed_seconds"]) == (7, 448, 420.0)
    assert counters["estimated_usd"] == pytest.approx(420 * RATE)
    assert counters["estimated_usd"] <= 0.2
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert (persisted["status"], persisted["stop_reason"]) == ("incomplete", "max_usd")
    assert len(list(path.parent.glob("smoke-*.jsonl"))) == 7


# ---------- operator commands ----------

def cli_arguments(serving, command: str) -> list[str]:
    plan_file = write_plan(serving.plan, serving.tmp / "plan" / "plan.json")
    return [command, "--plan", str(plan_file), "--manifest", str(serving.manifest),
            "--stage-limits", *map(str, serving.stages), "--approvals", str(serving.approvals)]


def test_present_prints_the_limits_and_a_record_template_and_writes_nothing(serving, capsys):
    assert limits_module.main(cli_arguments(serving, "present")) == 0
    out = capsys.readouterr().out
    assert "13089" in out and "33.42" in out and "approval" in out.lower()
    template = json.loads(out[out.index("\n{") + 1:])
    assert template == approval_template(present_serving_arm(serving.plan, serving.manifest, serving.stages))
    assert not serving.approvals.exists()


def test_check_refuses_without_a_record_and_passes_with_a_matching_one(serving, capsys):
    arguments = cli_arguments(serving, "check")
    assert limits_module.main(arguments) == EXIT_REFUSED
    assert "no approval record" in capsys.readouterr().err
    write_record(serving, approved_record(serving))
    assert limits_module.main(arguments) == 0
    assert "approved" in capsys.readouterr().out
