"""Stage limits and the persisted request budget (task 7.5; Requirement 11.3).

Pure logic on fakes: a fake monotonic clock, fake operations and files under
``tmp_path``. No network, credentials, Modal or model host is used. The serving
arm's cost limit and its approval are covered in ``test_approval.py``.
"""

from __future__ import annotations

import hashlib
import json
import socket
from datetime import UTC, datetime
from pathlib import Path

import pytest

from recall_guard.core.completion import ModelConfig, ModelResponse
from recall_guard.core.request_control import BudgetExhaustedError, RequestControl, RequestError
from scripts.finance_sim.limits import (
    ARMS,
    BUDGET_EXHAUSTED,
    EXIT_INCOMPLETE,
    AdmissionClosedError,
    BudgetLedger,
    LedgerCallBudget,
    RequestAllowance,
    StudyLimits,
    approval_required,
    budget_path,
)
from scripts.finance_sim.plan import EDGAR_FILES, EdgarRow, MonthClass, PlanChangedError, StudyPlan, freeze_plan

GEMMA = "google/gemma-3-4b-it"
CREATED_AT = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)
SEED = 20261110


def month_range(first: str, last: str) -> list[str]:
    year, month = int(first[:4]), int(first[5:])
    out = []
    while f"{year:04d}-{month:02d}" <= last:
        out.append(f"{year:04d}-{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return out


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def frozen_plan(seed: int = SEED) -> StudyPlan:
    """A small but valid frozen plan; only its hash matters to the limits."""
    months = [MonthClass(month, {GEMMA: month <= "2024-08"}) for month in month_range("2016-01", "2026-06")]
    rows = [EdgarRow(f"cal-mem-{i:03d}", 1, "calibration") for i in range(40)]
    rows += [EdgarRow(f"cal-ctl-{i:03d}", 0, "calibration") for i in range(100)]
    rows += [EdgarRow(f"held-ctl-{i:03d}", 0, "held_out") for i in range(40)]
    return freeze_plan(months, rows, seed=seed, created_at=CREATED_AT, items={"macro-2016-01": digest("macro")},
                       edgar_files={name: digest(name) for name in EDGAR_FILES})


PLAN = frozen_plan()


class Clock:
    """A fake monotonic clock in seconds."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("the limits attempted a network connection")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)


def sampling_limits(plan: StudyPlan = PLAN, **overrides) -> StudyLimits:
    values = {"plan_sha256": plan.plan_sha256, "arm": "sampling", "stage": "full_run", "max_requests": 100,
              "max_output_tokens": 100_000, "max_seconds": 100_000} | overrides
    return StudyLimits(**values)


ALLOWANCE = RequestAllowance(output_tokens=100, timeout_s=30)


def open_ledger(path: Path, clock: Clock, *, plan: StudyPlan = PLAN, limits: StudyLimits | None = None,
                allowance: RequestAllowance = ALLOWANCE) -> BudgetLedger:
    return BudgetLedger.open(path, plan, limits or sampling_limits(plan), allowance, clock=clock)


def on_disk(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_until_closed(ledger: BudgetLedger, clock: Clock, stage_dir: Path, *, step_s: float,
                     output_tokens: int | None) -> tuple[int, AdmissionClosedError]:
    """Admit, 'send' and record requests until admission closes; each request leaves a result file."""
    admitted = 0
    with pytest.raises(AdmissionClosedError) as closed:
        while admitted <= 200:
            admission = ledger.admit()
            admitted += 1
            (stage_dir / f"item-{admitted:03d}.jsonl").write_text('{"draw_index": 0}\n', encoding="utf-8")
            clock.advance(step_s)
            ledger.finish(admission, output_tokens=output_tokens)
    assert admitted <= 200, "admission never closed"
    return admitted, closed.value


# ---------- each limit closes admission and marks the stage incomplete (11.3) ----------

LIMIT_CASES = {
    # 3 requests fit; the fourth would exceed max_requests.
    "max_requests": ({"max_requests": 3}, 1.0, 50, 3, {"requests": 3, "output_tokens": 150, "elapsed_seconds": 3.0}),
    # Each request reserves 100 tokens: 0 + 100 and 100 + 100 fit 250, 200 + 100 does not.
    "max_output_tokens": ({"max_output_tokens": 250}, 1.0, 100, 2,
                          {"requests": 2, "output_tokens": 200, "elapsed_seconds": 2.0}),
    # A request is admitted only while elapsed + its 30 s timeout fits 100 s: at 0, 10, ..., 70 s.
    "max_seconds": ({"max_seconds": 100}, 10.0, 10, 8, {"requests": 8, "output_tokens": 80, "elapsed_seconds": 80.0}),
}


@pytest.mark.parametrize("limit", sorted(LIMIT_CASES))
def test_each_limit_closes_admission_and_marks_the_stage_incomplete_with_partial_results_kept(tmp_path, limit):
    overrides, step_s, tokens, expected_admitted, expected_counters = LIMIT_CASES[limit]
    clock = Clock()
    path = budget_path("sampling", "full_run", tmp_path / "records")
    ledger = open_ledger(path, clock, limits=sampling_limits(**overrides))
    stage_dir = path.parent

    admitted, closed = run_until_closed(ledger, clock, stage_dir, step_s=step_s, output_tokens=tokens)

    assert admitted == expected_admitted
    assert isinstance(closed, BudgetExhaustedError)
    assert (closed.reason, closed.status) == (limit, "incomplete")
    assert limit in str(closed) and "incomplete" in str(closed)
    assert (ledger.status, ledger.stop_reason, ledger.exit_code) == ("incomplete", limit, EXIT_INCOMPLETE)
    assert ledger.counters == expected_counters | {"estimated_usd": None}
    # Persisted before raising, so a crash right after still reads as incomplete.
    persisted = on_disk(path)
    assert (persisted["status"], persisted["stop_reason"]) == ("incomplete", limit)
    assert persisted["counters"] == expected_counters | {"estimated_usd": None}
    assert persisted["in_flight"] == []
    # Partial results are kept: every request's record is still there, next to budget.json.
    assert sorted(p.name for p in stage_dir.glob("item-*.jsonl")) == [f"item-{i:03d}.jsonl"
                                                                      for i in range(1, admitted + 1)]
    # Admission stays closed, in this session and after a resume.
    with pytest.raises(AdmissionClosedError):
        ledger.admit()
    ledger.close()
    resumed = open_ledger(path, clock, limits=sampling_limits(**overrides))
    assert (resumed.status, resumed.stop_reason) == ("incomplete", limit)
    with pytest.raises(AdmissionClosedError):
        resumed.admit()
    assert resumed.counters["requests"] == expected_admitted


def test_a_reply_above_its_reservation_closes_admission_on_the_token_limit(tmp_path):
    clock = Clock()
    path = tmp_path / "budget.json"
    ledger = open_ledger(path, clock, limits=sampling_limits(max_output_tokens=150))
    admission = ledger.admit()
    ledger.finish(admission, output_tokens=160)
    assert (ledger.status, ledger.stop_reason) == ("incomplete", "max_output_tokens")
    assert on_disk(path)["counters"]["output_tokens"] == 160
    with pytest.raises(AdmissionClosedError, match="max_output_tokens"):
        ledger.admit()


def test_a_zero_request_limit_admits_nothing(tmp_path):
    ledger = open_ledger(tmp_path / "budget.json", Clock(), limits=sampling_limits(max_requests=0))
    with pytest.raises(AdmissionClosedError) as closed:
        ledger.admit()
    assert closed.value.reason == "max_requests" and ledger.counters["requests"] == 0


# ---------- counters, usage and resume ----------

def test_reported_tokens_are_charged_and_unknown_usage_charges_the_full_reservation(tmp_path):
    ledger = open_ledger(tmp_path / "budget.json", Clock())
    first = ledger.admit()
    ledger.finish(first, output_tokens=40)
    assert ledger.counters["output_tokens"] == 40
    second = ledger.admit()
    ledger.finish(second, output_tokens=None)
    assert ledger.counters["output_tokens"] == 140
    with pytest.raises(ValueError, match="not in flight"):
        ledger.finish(second, output_tokens=1)
    third = ledger.admit()
    for bad in (-1, True, 1.5, "7"):
        with pytest.raises((TypeError, ValueError)):
            ledger.finish(third, output_tokens=bad)
    ledger.finish(third, output_tokens=0)
    assert ledger.counters == {"requests": 3, "output_tokens": 140, "elapsed_seconds": 0.0, "estimated_usd": None}


def test_reservations_in_flight_count_against_the_token_limit(tmp_path):
    ledger = open_ledger(tmp_path / "budget.json", Clock(), limits=sampling_limits(max_output_tokens=200))
    ledger.admit()
    ledger.admit()
    with pytest.raises(AdmissionClosedError, match="max_output_tokens"):
        ledger.admit()


def test_resume_continues_the_counters_and_time_between_sessions_is_not_stage_time(tmp_path):
    clock = Clock()
    path = tmp_path / "records" / "sampling" / "full_run" / "budget.json"
    ledger = open_ledger(path, clock)
    for _ in range(2):
        admission = ledger.admit()
        clock.advance(5)
        ledger.finish(admission, output_tokens=10)
    ledger.close()
    with pytest.raises(RuntimeError, match="closed"):
        ledger.admit()
    clock.advance(3600)

    resumed = open_ledger(path, clock)
    assert resumed.counters == {"requests": 2, "output_tokens": 20, "elapsed_seconds": 10.0, "estimated_usd": None}
    assert resumed.snapshot()["sessions"] == 2
    admission = resumed.admit()
    clock.advance(5)
    resumed.finish(admission, output_tokens=10)
    assert resumed.counters == {"requests": 3, "output_tokens": 30, "elapsed_seconds": 15.0, "estimated_usd": None}
    assert resumed.remaining_requests == 97
    assert on_disk(path)["counters"]["requests"] == 3


def test_a_request_in_flight_when_a_session_died_is_charged_its_full_reservation_on_resume(tmp_path):
    clock = Clock()
    path = tmp_path / "budget.json"
    ledger = open_ledger(path, clock)
    ledger.admit()
    clock.advance(7)
    persisted = on_disk(path)
    assert persisted["counters"]["requests"] == 1
    assert persisted["in_flight"] == [{"request": 1, "output_tokens": 100}]
    del ledger  # the process dies: no finish, no close

    resumed = open_ledger(path, clock)
    snapshot = resumed.snapshot()
    # The intent was persisted at admission (elapsed 0 s); the unknown outcome costs its reservation and timeout.
    assert snapshot["counters"] == {"requests": 1, "output_tokens": 100, "elapsed_seconds": 30.0,
                                    "estimated_usd": None}
    assert snapshot["in_flight"] == [] and snapshot["unknown_outcomes"] == 1
    assert resumed.status == "open"


def test_complete_marks_the_stage_complete_and_admits_nothing_more(tmp_path):
    path = tmp_path / "budget.json"
    ledger = open_ledger(path, Clock())
    admission = ledger.admit()
    with pytest.raises(ValueError, match="in flight"):
        ledger.complete()
    ledger.finish(admission, output_tokens=5)
    ledger.complete()
    assert (ledger.status, ledger.stop_reason, ledger.exit_code) == ("complete", None, 0)
    assert on_disk(path)["status"] == "complete"
    with pytest.raises(AdmissionClosedError, match="complete"):
        ledger.admit()


def test_an_incomplete_stage_cannot_be_marked_complete(tmp_path):
    ledger = open_ledger(tmp_path / "budget.json", Clock(), limits=sampling_limits(max_requests=1))
    ledger.finish(ledger.admit(), output_tokens=1)
    with pytest.raises(AdmissionClosedError):
        ledger.admit()
    with pytest.raises(ValueError, match="incomplete"):
        ledger.complete()
    assert ledger.status == "incomplete"


def test_any_budget_exhaustion_inside_a_session_marks_the_stage_incomplete(tmp_path):
    path = tmp_path / "budget.json"
    with pytest.raises(BudgetExhaustedError):
        with open_ledger(path, Clock()) as ledger:
            ledger.finish(ledger.admit(), output_tokens=1)
            raise BudgetExhaustedError("Physical request budget exhausted")
    assert (ledger.status, ledger.stop_reason) == ("incomplete", BUDGET_EXHAUSTED)
    assert on_disk(path)["status"] == "incomplete"


def test_another_error_inside_a_session_leaves_the_stage_open_for_resume(tmp_path):
    path = tmp_path / "budget.json"
    with pytest.raises(RuntimeError, match="host down"):
        with open_ledger(path, Clock()) as ledger:
            ledger.finish(ledger.admit(), output_tokens=1)
            raise RuntimeError("host down")
    assert on_disk(path)["status"] == "open"
    assert open_ledger(path, Clock()).counters["requests"] == 1


# ---------- refusal: never merged, always the frozen plan ----------

def test_a_budget_of_another_plan_limits_or_allowance_is_refused_never_merged(tmp_path):
    path = tmp_path / "budget.json"
    ledger = open_ledger(path, Clock())
    ledger.finish(ledger.admit(), output_tokens=1)
    ledger.close()
    before = path.read_bytes()

    other = frozen_plan(seed=SEED + 1)
    assert other.plan_sha256 != PLAN.plan_sha256
    with pytest.raises(PlanChangedError, match="never merged"):
        open_ledger(path, Clock(), plan=other, limits=sampling_limits(other))
    with pytest.raises(ValueError, match="never merged"):
        open_ledger(path, Clock(), limits=sampling_limits(max_requests=99))
    with pytest.raises(ValueError, match="never merged"):
        open_ledger(path, Clock(), limits=sampling_limits(stage="calibration"))
    with pytest.raises(ValueError, match="never merged"):
        open_ledger(path, Clock(), allowance=RequestAllowance(output_tokens=256, timeout_s=30))
    assert path.read_bytes() == before


def test_limits_of_another_plan_are_refused_before_any_file_is_written(tmp_path):
    path = tmp_path / "budget.json"
    other = frozen_plan(seed=SEED + 1)
    with pytest.raises(PlanChangedError):
        open_ledger(path, Clock(), limits=sampling_limits(other))
    assert not path.exists()


@pytest.mark.parametrize("text", [
    "[]",
    '{"schema": "finance-sim-budget/1"}',
    "not json",
])
def test_an_unreadable_budget_file_is_refused(tmp_path, text):
    path = tmp_path / "budget.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        open_ledger(path, Clock())
    assert path.read_text(encoding="utf-8") == text


def test_a_tampered_counter_is_refused(tmp_path):
    path = tmp_path / "budget.json"
    open_ledger(path, Clock()).close()
    for counters in ({"requests": -1}, {"requests": True}, {"output_tokens": 1.5}, {"elapsed_seconds": -2.0}):
        payload = on_disk(path)
        payload["counters"] |= counters
        broken = tmp_path / "broken.json"
        broken.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises((TypeError, ValueError)):
            open_ledger(broken, Clock())
    duplicate = path.read_text(encoding="utf-8").replace('"status": "open"', '"status": "open", "status": "complete"')
    path.write_text(duplicate, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        open_ledger(path, Clock())


# ---------- the in-memory guard reuses CallBudget ----------

def control_for(ledger: BudgetLedger, clock: Clock) -> tuple[RequestControl, LedgerCallBudget]:
    config = ModelConfig(protocol="callback", endpoint=None, model="fake-gemma", max_retries=0, retry_backoff_s=0)
    budget = LedgerCallBudget(ledger)
    return RequestControl(config, budget, clock=clock), budget


def test_the_call_budget_admits_through_the_ledger_and_raises_budget_exhausted_on_any_limit(tmp_path):
    clock = Clock()
    path = tmp_path / "budget.json"
    ledger = open_ledger(path, clock, limits=sampling_limits(max_requests=3))
    control, budget = control_for(ledger, clock)
    assert (budget.limit, budget.remaining) == (3, 3)

    def reply() -> ModelResponse:
        clock.advance(2)
        return ModelResponse("{}", usage={"prompt_tokens": 30, "completion_tokens": 7, "total_tokens": 37})

    for _ in range(3):
        control.run(reply, stage="full_run")
    with pytest.raises(BudgetExhaustedError) as closed:
        control.run(reply, stage="full_run")
    assert isinstance(closed.value, AdmissionClosedError) and closed.value.reason == "max_requests"
    assert budget.used == 3
    assert ledger.counters == {"requests": 3, "output_tokens": 21, "elapsed_seconds": 6.0, "estimated_usd": None}
    assert on_disk(path)["status"] == "incomplete"


def test_a_guarded_admission_refused_by_the_in_memory_cap_still_marks_the_stage_incomplete(tmp_path):
    # RequestControl's guarded path refuses on ``remaining == 0`` without calling ``_admit``;
    # leaving the ``with ledger:`` block on that BudgetExhaustedError closes the stage all the same.
    clock = Clock()
    path = tmp_path / "budget.json"
    with pytest.raises(BudgetExhaustedError) as refused:
        with open_ledger(path, clock, limits=sampling_limits(max_requests=1)) as ledger:
            control, budget = control_for(ledger, clock)
            control.run(lambda: ModelResponse("{}", usage={"completion_tokens": 3}), stage="full_run")
            control.run(lambda: ModelResponse("{}"), stage="full_run", admission_guard=lambda: None)
    assert not isinstance(refused.value, AdmissionClosedError)
    assert (ledger.status, ledger.stop_reason) == ("incomplete", "max_requests")
    assert on_disk(path)["counters"]["requests"] == 1 and budget.used == 1


def test_the_call_budget_charges_the_reservation_for_a_failed_request_and_resumes_with_the_remaining_count(tmp_path):
    clock = Clock()
    path = tmp_path / "budget.json"
    ledger = open_ledger(path, clock, limits=sampling_limits(max_requests=5))
    control, _ = control_for(ledger, clock)

    def fails() -> ModelResponse:
        raise TimeoutError("no reply")

    with pytest.raises(RequestError):
        control.run(fails, stage="full_run")
    assert ledger.counters["requests"] == 1 and ledger.counters["output_tokens"] == 100
    ledger.close()

    resumed = open_ledger(path, clock, limits=sampling_limits(max_requests=5))
    budget = LedgerCallBudget(resumed)
    assert (budget.limit, budget.remaining) == (4, 4)


# ---------- validation at the boundary ----------

@pytest.mark.parametrize("overrides", [
    {"plan_sha256": "A" * 64}, {"plan_sha256": "0" * 63}, {"plan_sha256": None},
    {"arm": "local"}, {"arm": "SAMPLING"},
    {"stage": ""}, {"stage": "Full Run"}, {"stage": None},
    {"max_requests": -1}, {"max_requests": True}, {"max_requests": 1.0},
    {"max_output_tokens": -1}, {"max_output_tokens": "256"},
    {"max_seconds": 0}, {"max_seconds": 1.5},
    {"max_usd": 1.0},  # the sampling arm runs on local hosts and has no cost limit
])
def test_invalid_study_limits_are_refused(overrides):
    with pytest.raises((TypeError, ValueError)):
        sampling_limits(**overrides)


@pytest.mark.parametrize("overrides", [
    {"max_usd": None}, {"max_usd": 0}, {"max_usd": float("nan")}, {"max_usd": True}, {"stage": "pilot"},
])
def test_serving_limits_need_a_positive_cost_limit_and_a_serving_stage(overrides):
    values = {"plan_sha256": PLAN.plan_sha256, "arm": "serving", "stage": "full_run", "max_requests": 1,
              "max_output_tokens": 1, "max_seconds": 1, "max_usd": 1.0} | overrides
    with pytest.raises((TypeError, ValueError)):
        StudyLimits(**values)


@pytest.mark.parametrize("overrides", [
    {"output_tokens": 0}, {"output_tokens": True}, {"timeout_s": 0}, {"timeout_s": float("inf")},
    {"usd_per_second": 0}, {"usd_per_second": -1.0},
])
def test_invalid_allowances_are_refused(overrides):
    with pytest.raises((TypeError, ValueError)):
        RequestAllowance(**({"output_tokens": 100, "timeout_s": 30} | overrides))


def test_the_sampling_arm_needs_no_approval_and_refuses_a_cost_basis(tmp_path):
    assert ARMS == ("sampling", "serving")
    assert approval_required("sampling") is False and approval_required("serving") is True
    with pytest.raises(ValueError):
        approval_required("pilot")
    with pytest.raises(ValueError, match="no approval"):
        BudgetLedger.open(tmp_path / "a.json", PLAN, sampling_limits(), ALLOWANCE, approval=object())
    with pytest.raises(ValueError, match="usd_per_second"):
        open_ledger(tmp_path / "b.json", Clock(),
                    allowance=RequestAllowance(output_tokens=100, timeout_s=30, usd_per_second=0.01))
    assert not (tmp_path / "a.json").exists() and not (tmp_path / "b.json").exists()


def test_budget_files_live_under_the_stage_records(tmp_path):
    assert budget_path("serving", "calibration", tmp_path) == tmp_path / "serving" / "calibration" / "budget.json"
    for arm, stage in (("local", "full_run"), ("sampling", "../x"), ("serving", "pilot")):
        with pytest.raises(ValueError):
            budget_path(arm, stage, tmp_path)


def test_the_incomplete_exit_code_is_distinct():
    assert EXIT_INCOMPLETE not in (0, 1, 2)
