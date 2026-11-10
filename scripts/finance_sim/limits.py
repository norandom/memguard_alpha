"""Stage limits, the persisted request budget and the serving arm's approval (task 7.5).

Limits (Requirement 11.3). Every arm and stage has frozen limits
(:class:`StudyLimits`): ``max_requests``, ``max_output_tokens``, ``max_seconds``
and, for the paid serving arm only, ``max_usd``. The sampling arm's come from the
frozen plan (7.3); the serving arm's are its six StageLimits files
(``deploy/modal/stage_limits.py``), which carry the frozen plan's hash. A
:class:`BudgetLedger` keeps the stage's counters in ``budget.json`` next to the
stage's request records (``runs/finance-sim/records/<arm>/<stage>/``): requests
issued, output tokens, elapsed seconds and estimated USD. It is checked before
every request, and a resumed run continues its counters.

Admission rule. The design says that reaching any limit closes admission; this
module takes the stricter reading that a request is admitted only when its worst
case still fits every limit, so an admitted request never takes a counter past its
limit:

- requests: the issued count plus one must fit ``max_requests``; a request counts
  when it is admitted, whatever its outcome;
- output tokens: the counted tokens, plus the reservations of the requests in
  flight, plus this request's cap (:attr:`RequestAllowance.output_tokens`) must fit
  ``max_output_tokens``; a finished request is charged its reported completion
  tokens, or its whole reservation when its usage is unknown;
- time: the stage's elapsed seconds plus the request timeout must fit
  ``max_seconds``; elapsed time counts only while a session of the stage is open;
- cost (serving arm): the estimated USD at that same horizon, elapsed seconds times
  the configured Modal rate of the serving app and its expiry controller, must fit
  ``max_usd``.

When a request does not fit, admission closes: the stage is marked ``incomplete``
with the limit as ``stop_reason``, nothing already recorded is removed, and
:class:`AdmissionClosedError` is raised. It is a ``BudgetExhaustedError``, which the
guarded loop treats as fatal; the runner then exits with :data:`EXIT_INCOMPLETE`. A
counter found above its limit after a request (a reply longer than its cap) closes
admission the same way, and so does any ``BudgetExhaustedError`` that leaves a
``with ledger:`` block. A request still in flight when a session ended without
finishing it is charged its whole reservation and its timeout when the stage is
opened again. :class:`LedgerCallBudget` is the in-memory guard: a ``CallBudget``
whose every admission goes through the ledger.

Approval (Requirement 11.2). Before the paid serving arm starts,
:func:`present_serving_arm` presents its request counts, output-token limits, time
limits and maximum cost per stage, in execution order, with the plan, manifest and
stage-limit hashes. The operator reviews them and writes
``runs/finance-sim/approvals/<plan_sha256>.json``. :func:`require_serving_approval`
validates that record with the shared ``approval_record`` gate of
``scripts/probe_modal_model.py`` (the five explicit assertions, the exact hashes,
the manifest's workspace and endpoint, a reviewed provider budget reference) and
also requires the manifest's endpoint in the record, a ``max_usd_total`` of at
least the presented maximum cost, the presented limits unchanged and the
operator's acknowledgement (``approved_by``, ``approved_at``). A missing or
mismatched record refuses the serving arm before any request, and a serving-arm
ledger opens only with a validated approval of its exact stage limits. The
sampling arm needs no approval record, only the frozen plan.

Operator commands (local files only; nothing is sent and nothing is written)::

    uv run --no-sync python -m scripts.finance_sim.limits present \\
        --plan PLAN --manifest MANIFEST --stage-limits FILE FILE FILE FILE FILE FILE
    uv run --no-sync python -m scripts.finance_sim.limits check (same arguments)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from threading import Lock
from time import monotonic
from types import MappingProxyType, TracebackType
from typing import Any

from deploy.modal.manifest import GEMMA_MANIFEST, Resources
from deploy.modal.manifest import Plan as ModalPlan
from deploy.modal.manifest import load_plan as load_modal_plan
from deploy.modal.stage_limits import STAGES, WEIGHT_STAGING, StageLimits, parse_stage_limits, reviewed_ceiling
from recall_guard.core.request_control import BudgetExhaustedError, CallBudget, RequestEvent
from scripts.finance_sim.plan import PlanChangedError, StudyPlan, ensure_unchanged, load_plan, sha256_of
from scripts.measure_modal_model import CONTROLLER_CORES, CONTROLLER_GIB, compute_rate
from scripts.probe_modal_model import APPROVAL_ASSERTIONS, approval_record, digest

ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = ROOT / "runs" / "finance-sim"
#: Per arm and stage request records (7.4); ``budget.json`` sits beside them.
RECORDS_DIR = RUNS_ROOT / "records"
#: Private approval records, one per frozen plan: ``<plan_sha256>.json``.
APPROVALS_DIR = RUNS_ROOT / "approvals"

SAMPLING_ARM = "sampling"
SERVING_ARM = "serving"
ARMS: tuple[str, ...] = (SAMPLING_ARM, SERVING_ARM)
#: Limits in the order admission checks them; ``stop_reason`` names the first that does not fit.
LIMIT_NAMES: tuple[str, ...] = ("max_requests", "max_output_tokens", "max_seconds", "max_usd")
#: Stop reason when a ``BudgetExhaustedError`` from elsewhere ends the stage's session.
BUDGET_EXHAUSTED = "budget_exhausted"
#: Stop reason when the ledger could not persist an admission; that request is never sent.
BUDGET_NOT_PERSISTED = "budget_not_persisted"
STOP_REASONS: tuple[str, ...] = (*LIMIT_NAMES, BUDGET_EXHAUSTED, BUDGET_NOT_PERSISTED)
OPEN, INCOMPLETE, COMPLETE = "open", "incomplete", "complete"
STATUSES: tuple[str, ...] = (OPEN, INCOMPLETE, COMPLETE)

EXIT_COMPLETE = 0
#: The stage is still open: the run ended early (an error or an interruption) and can resume.
EXIT_UNFINISHED = 1
#: The serving arm was refused: no approval record, or one that does not match the frozen plan.
EXIT_REFUSED = 2
#: A frozen limit closed admission; the partial results are kept and the stage is incomplete (11.3).
EXIT_INCOMPLETE = 3

BUDGET_SCHEMA = "finance-sim-budget/1"
APPROVAL_SCHEMA = "finance-sim-serving-approval/1"
COST_BASIS = ("configured Modal rates of the serving app and its expiry controller times app seconds; "
              "an estimate, not an invoice")

_SHA256 = re.compile(r"[0-9a-f]{64}")
_STAGE = re.compile(r"[a-z][a-z0-9_]*")
_COUNTER_KEYS = frozenset({"requests", "output_tokens", "elapsed_seconds", "estimated_usd"})
_BUDGET_KEYS = frozenset({"schema", "plan_sha256", "arm", "stage", "limits", "allowance", "counters", "in_flight",
                          "unknown_outcomes", "sessions", "status", "stop_reason", "created_at", "updated_at"})
_APPROVAL_KEYS = frozenset(APPROVAL_ASSERTIONS) | frozenset({
    "schema", "plan_sha256", "manifest_sha256", "stage_limits_sha256", "workspace", "endpoint",
    "provider_budget_reference", "max_usd_total", "presented", "approved_by", "approved_at",
})

# ---------- validation helpers ----------

def _utc_now() -> datetime:
    return datetime.now(UTC)


def _iso(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat(timespec="seconds")


def _sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a SHA-256 as 64 lowercase hex characters; got {value!r}")
    return value


def _count(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer; got {value!r}")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}; got {value}")
    return value


def _real(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a number; got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite; got {value!r}")
    return number


def _positive(value: object, name: str) -> float:
    number = _real(value, name)
    if number <= 0:
        raise ValueError(f"{name} must be positive; got {value!r}")
    return number


def _arm(arm: object) -> str:
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {list(ARMS)}; got {arm!r}")
    return arm


def _stage(arm: str, stage: object) -> str:
    if not isinstance(stage, str) or not _STAGE.fullmatch(stage):
        raise ValueError(f"stage must be a lowercase identifier; got {stage!r}")
    if arm == SERVING_ARM and stage not in STAGES:
        raise ValueError(f"a serving-arm stage must be one of {list(STAGES)}; got {stage!r}")
    return stage


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate keys {sorted({key for key in keys if keys.count(key) > 1})}")
    return dict(pairs)


def _no_constant(name: str) -> Any:
    raise ValueError(f"non-finite number {name}")


def _strict_json(text: str) -> Any:
    """Parse JSON refusing duplicate keys and NaN or infinity."""
    return json.loads(text, object_pairs_hook=_no_duplicate_keys, parse_constant=_no_constant)


def _json_form(value: Any) -> Any:
    """``value`` as it reads back from JSON: tuples become lists."""
    return json.loads(json.dumps(value, allow_nan=False))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write atomically: a crash leaves the previous file or the new one, never half of one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


# ---------- limits and allowance ----------

@dataclass(frozen=True)
class StudyLimits:
    """Frozen limits of one arm and stage, bound to the frozen plan by its hash.

    The sampling arm runs on local hosts and has no cost limit (``max_usd`` is
    None); the serving arm's limits are its StageLimits files and need ``max_usd``.
    """

    plan_sha256: str
    arm: str
    stage: str
    max_requests: int
    max_output_tokens: int
    max_seconds: int
    max_usd: float | None = None

    def __post_init__(self) -> None:
        _sha256(self.plan_sha256, "plan_sha256")
        _stage(_arm(self.arm), self.stage)
        _count(self.max_requests, "max_requests")
        _count(self.max_output_tokens, "max_output_tokens")
        _count(self.max_seconds, "max_seconds", minimum=1)
        if self.arm == SERVING_ARM:
            object.__setattr__(self, "max_usd", _positive(self.max_usd, "max_usd"))
        elif self.max_usd is not None:
            raise ValueError("The sampling arm runs on local hosts and has no max_usd; leave it None")

    @classmethod
    def from_stage_limits(cls, limits: StageLimits) -> StudyLimits:
        """The serving arm's limits for one stage, from its validated StageLimits file."""
        if not isinstance(limits, StageLimits):
            raise TypeError(f"limits must be StageLimits; got {type(limits).__name__}")
        return cls(limits.plan_sha256, SERVING_ARM, limits.stage, limits.max_requests, limits.max_output_tokens,
                   limits.max_seconds, limits.max_usd)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RequestAllowance:
    """What one request may use, reserved when it is admitted.

    ``output_tokens`` is the request's output cap (its ``max_tokens``),
    ``timeout_s`` its timeout. ``usd_per_second`` turns the stage's elapsed seconds
    into estimated USD: the serving arm's configured Modal rate, None for the
    sampling arm.
    """

    output_tokens: int
    timeout_s: float
    usd_per_second: float | None = None

    def __post_init__(self) -> None:
        _count(self.output_tokens, "output_tokens", minimum=1)
        object.__setattr__(self, "timeout_s", _positive(self.timeout_s, "timeout_s"))
        if self.usd_per_second is not None:
            object.__setattr__(self, "usd_per_second", _positive(self.usd_per_second, "usd_per_second"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Admission:
    """One admitted request: its number within the stage and the output tokens reserved for it."""

    request: int
    output_tokens: int


class AdmissionClosedError(BudgetExhaustedError):
    """Admission of the stage is closed: a frozen limit is reached, or the stage is complete."""

    def __init__(self, message: str, *, reason: str | None, status: str) -> None:
        super().__init__(message)
        self.reason = reason
        self.status = status


class ApprovalError(ValueError):
    """The serving arm is refused: no approval record, or one that does not match the frozen plan."""


def approval_required(arm: str) -> bool:
    """Only the paid serving arm needs an approval record; the sampling arm needs the frozen plan only."""
    return _arm(arm) == SERVING_ARM


def budget_path(arm: str, stage: str, records_dir: str | Path = RECORDS_DIR) -> Path:
    """``<records_dir>/<arm>/<stage>/budget.json``, beside the stage's request records."""
    return Path(records_dir) / _arm(arm) / _stage(arm, stage) / "budget.json"


# ---------- the persisted budget ----------

class BudgetLedger:
    """The persisted counters of one arm and stage, checked before every request.

    Open it with :meth:`open`. Use it as a context manager so that the session's
    elapsed time is saved on exit and a ``BudgetExhaustedError`` from any source
    closes admission. The ledger never removes a record of the stage.
    """

    def __init__(self, path: Path, limits: StudyLimits, allowance: RequestAllowance, data: dict[str, Any], *,
                 clock: Callable[[], float], now: Callable[[], datetime]) -> None:
        self.path = path
        self.limits = limits
        self.allowance = allowance
        self._data = data
        self._clock = clock
        self._now = now
        self._lock = Lock()
        self._active = True
        self._counting = data["status"] == OPEN
        self._base_elapsed = float(data["counters"]["elapsed_seconds"])
        self._session_start = clock()

    @classmethod
    def open(cls, path: str | Path, plan: StudyPlan, limits: StudyLimits, allowance: RequestAllowance, *,
             approval: ServingApproval | None = None, clock: Callable[[], float] = monotonic,
             now: Callable[[], datetime] = _utc_now) -> BudgetLedger:
        """Open or resume the stage's ledger; refuse another plan, other limits or a missing approval.

        A ledger written under another plan hash, arm, stage, limits or allowance is
        refused, never merged. The serving arm opens only with a validated
        :class:`ServingApproval` whose limits for this stage equal ``limits``.
        """
        if not isinstance(limits, StudyLimits):
            raise TypeError(f"limits must be StudyLimits; got {type(limits).__name__}")
        if not isinstance(allowance, RequestAllowance):
            raise TypeError(f"allowance must be a RequestAllowance; got {type(allowance).__name__}")
        ensure_unchanged(plan, limits.plan_sha256)
        _check_arm(plan, limits, allowance, approval)
        target = Path(path)
        data = _read_budget(target, limits, allowance) if target.exists() else _new_budget(limits, allowance, now())
        ledger = cls(target, limits, allowance, data, clock=clock, now=now)
        ledger._recover()
        return ledger

    # ----- state -----

    @property
    def status(self) -> str:
        return self._data["status"]

    @property
    def stop_reason(self) -> str | None:
        return self._data["stop_reason"]

    @property
    def exit_code(self) -> int:
        """0 when complete, :data:`EXIT_INCOMPLETE` when a limit closed admission, 1 while still open."""
        return {COMPLETE: EXIT_COMPLETE, INCOMPLETE: EXIT_INCOMPLETE}.get(self.status, EXIT_UNFINISHED)

    @property
    def counters(self) -> dict[str, Any]:
        with self._lock:
            self._sync()
            return dict(self._data["counters"])

    @property
    def remaining_requests(self) -> int:
        with self._lock:
            if self._data["status"] != OPEN:
                return 0
            return max(0, self.limits.max_requests - self._data["counters"]["requests"])

    def snapshot(self) -> dict[str, Any]:
        """A detached copy of ``budget.json`` as it stands now."""
        with self._lock:
            self._sync()
            return json.loads(json.dumps(self._data))

    # ----- admission -----

    def admit(self) -> Admission:
        """Admit one request, or close admission and raise :class:`AdmissionClosedError`.

        The admission is persisted before it is returned, so a request is never
        sent without being counted.
        """
        with self._lock:
            if not self._active:
                raise RuntimeError(f"this session of the {self.limits.arm}/{self.limits.stage} ledger is closed")
            if self._data["status"] != OPEN:
                raise self._closed_error()
            reason = self._blocking_limit()
            if reason is not None:
                self._close_admission(reason)
                raise self._closed_error()
            counters = self._data["counters"]
            counters["requests"] += 1
            admission = Admission(counters["requests"], self.allowance.output_tokens)
            self._data["in_flight"].append({"request": admission.request, "output_tokens": admission.output_tokens})
            try:
                self._persist()
            except BaseException:
                # Unpersisted: the request must not be sent, and nothing more is admitted in this session.
                self._data["status"], self._data["stop_reason"] = INCOMPLETE, BUDGET_NOT_PERSISTED
                raise
            return admission

    def finish(self, admission: Admission, *, output_tokens: int | None) -> None:
        """Charge a finished request: its reported output tokens, or its whole reservation when unknown."""
        if not isinstance(admission, Admission):
            raise TypeError(f"admission must be an Admission; got {type(admission).__name__}")
        charged = admission.output_tokens if output_tokens is None else _count(output_tokens, "output_tokens")
        with self._lock:
            entries = self._data["in_flight"]
            entry = next((item for item in entries if item["request"] == admission.request), None)
            if entry is None:
                raise ValueError(f"request {admission.request} is not in flight; it was finished or never admitted")
            entries.remove(entry)
            self._data["counters"]["output_tokens"] += charged
            reason = self._exceeded_limit() if self._data["status"] == OPEN else None
            if reason is not None:
                self._close_admission(reason)
            else:
                self._persist()

    def complete(self) -> None:
        """Mark the stage complete once every planned request has finished; an incomplete stage stays so."""
        with self._lock:
            status = self._data["status"]
            if status == COMPLETE:
                return
            if status == INCOMPLETE:
                raise ValueError(f"the {self.limits.arm}/{self.limits.stage} stage is incomplete "
                                 f"({self._data['stop_reason']}); it cannot be marked complete")
            if self._data["in_flight"]:
                raise ValueError("requests are still in flight; finish them before completing the stage")
            self._freeze()
            self._data["status"] = COMPLETE
            self._persist()

    def stop_admission(self, reason: str | None = None) -> str:
        """Close admission now, naming the limit reached (or ``budget_exhausted``); returns the stop reason."""
        with self._lock:
            if self._data["status"] != OPEN:
                return self._data["stop_reason"] or self._data["status"]
            reason = reason or self._blocking_limit() or BUDGET_EXHAUSTED
            if reason not in STOP_REASONS:
                raise ValueError(f"reason must be one of {list(STOP_REASONS)}; got {reason!r}")
            self._close_admission(reason)
            return reason

    def close(self) -> None:
        """End this session: its elapsed time is saved and nothing more is admitted through this object."""
        with self._lock:
            if self._active:
                self._freeze()
                self._persist()
                self._active = False

    def __enter__(self) -> BudgetLedger:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None,
                 traceback: TracebackType | None) -> None:
        if isinstance(exc, BudgetExhaustedError):
            self.stop_admission()
        self.close()

    # ----- internals (called with the lock held) -----

    def _elapsed(self) -> float:
        if self._counting:
            return self._base_elapsed + max(0.0, self._clock() - self._session_start)
        return self._base_elapsed

    def _freeze(self) -> None:
        self._base_elapsed = self._elapsed()
        self._counting = False

    def _usd(self, seconds: float) -> float | None:
        rate = self.allowance.usd_per_second
        return None if rate is None else seconds * rate

    def _sync(self) -> None:
        counters = self._data["counters"]
        counters["elapsed_seconds"] = self._elapsed()
        counters["estimated_usd"] = self._usd(counters["elapsed_seconds"])

    def _persist(self) -> None:
        self._sync()
        self._data["updated_at"] = _iso(self._now())
        _write_json(self.path, self._data)

    def _blocking_limit(self) -> str | None:
        """The first limit the next request's worst case would not fit, or None."""
        counters, limits = self._data["counters"], self.limits
        if counters["requests"] + 1 > limits.max_requests:
            return "max_requests"
        reserved = sum(entry["output_tokens"] for entry in self._data["in_flight"])
        if counters["output_tokens"] + reserved + self.allowance.output_tokens > limits.max_output_tokens:
            return "max_output_tokens"
        horizon = self._elapsed() + self.allowance.timeout_s
        if horizon > limits.max_seconds:
            return "max_seconds"
        usd = self._usd(horizon)
        if limits.max_usd is not None and usd is not None and usd > limits.max_usd:
            return "max_usd"
        return None

    def _exceeded_limit(self) -> str | None:
        """The first counter already above its limit, or None."""
        counters, limits = self._data["counters"], self.limits
        if counters["requests"] > limits.max_requests:
            return "max_requests"
        if counters["output_tokens"] > limits.max_output_tokens:
            return "max_output_tokens"
        elapsed = self._elapsed()
        if elapsed > limits.max_seconds:
            return "max_seconds"
        usd = self._usd(elapsed)
        if limits.max_usd is not None and usd is not None and usd > limits.max_usd:
            return "max_usd"
        return None

    def _close_admission(self, reason: str) -> None:
        self._freeze()
        self._data["status"], self._data["stop_reason"] = INCOMPLETE, reason
        self._persist()

    def _closed_error(self) -> AdmissionClosedError:
        status, reason = self._data["status"], self._data["stop_reason"]
        arm, stage = self.limits.arm, self.limits.stage
        if status == COMPLETE:
            return AdmissionClosedError(f"The {arm}/{stage} stage is complete; no further request is admitted",
                                        reason=None, status=status)
        self._sync()
        counters, limits = self._data["counters"], self.limits
        usd = "" if limits.max_usd is None else f", estimated USD {counters['estimated_usd']:.4f} of {limits.max_usd}"
        return AdmissionClosedError(
            f"Admission closed for the {arm} arm, stage {stage}: {reason} reached (requests "
            f"{counters['requests']} of {limits.max_requests}, output tokens {counters['output_tokens']} of "
            f"{limits.max_output_tokens}, elapsed {counters['elapsed_seconds']:.1f} s of {limits.max_seconds} s"
            f"{usd}); the stage is incomplete and its partial results are kept",
            reason=reason, status=status)

    def _recover(self) -> None:
        """Charge requests left in flight by a session that ended without finishing them, then persist."""
        with self._lock:
            entries = self._data["in_flight"]
            if entries:
                self._data["counters"]["output_tokens"] += sum(entry["output_tokens"] for entry in entries)
                self._base_elapsed += self.allowance.timeout_s
                self._data["unknown_outcomes"] += len(entries)
                self._data["in_flight"] = []
            if self._data["status"] == OPEN:
                self._data["sessions"] += 1
                reason = self._exceeded_limit()
                if reason is not None:
                    self._close_admission(reason)
                    return
            self._persist()


def _check_arm(plan: StudyPlan, limits: StudyLimits, allowance: RequestAllowance,
               approval: ServingApproval | None) -> None:
    if limits.arm == SAMPLING_ARM:
        if approval is not None:
            raise ValueError("The sampling arm needs no approval record; pass none")
        if allowance.usd_per_second is not None:
            raise ValueError("The sampling arm runs on local hosts; its allowance has no usd_per_second")
        return
    if not isinstance(approval, ServingApproval):
        raise ApprovalError(f"Serving arm refused before any request: stage {limits.stage} needs the validated "
                            f"approval record of plan {plan.plan_sha256} (require_serving_approval)")
    if approval.plan_sha256 != plan.plan_sha256:
        raise ApprovalError(f"Serving arm refused before any request: the approval is for plan "
                            f"{approval.plan_sha256}, the frozen plan is {plan.plan_sha256}")
    if approval.limits_for(limits.stage) != limits:
        raise ApprovalError(f"Serving arm refused before any request: the {limits.stage} limits differ from the "
                            f"approved ones")
    if allowance.usd_per_second != approval.presentation.usd_per_second:
        raise ApprovalError("Serving arm refused before any request: the allowance's cost basis "
                            f"({allowance.usd_per_second} USD per second) is not the presented "
                            f"{approval.presentation.usd_per_second}")


def _new_budget(limits: StudyLimits, allowance: RequestAllowance, created: datetime) -> dict[str, Any]:
    stamp = _iso(created)
    return {
        "schema": BUDGET_SCHEMA, "plan_sha256": limits.plan_sha256, "arm": limits.arm, "stage": limits.stage,
        "limits": limits.to_dict(), "allowance": allowance.to_dict(),
        "counters": {"requests": 0, "output_tokens": 0, "elapsed_seconds": 0.0,
                     "estimated_usd": None if allowance.usd_per_second is None else 0.0},
        "in_flight": [], "unknown_outcomes": 0, "sessions": 0, "status": OPEN, "stop_reason": None,
        "created_at": stamp, "updated_at": stamp,
    }


def _read_budget(path: Path, limits: StudyLimits, allowance: RequestAllowance) -> dict[str, Any]:
    """Validate an existing ``budget.json`` against this stage; anything else is refused, never merged."""
    try:
        data = _strict_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"{path} is not a readable budget ledger ({exc}); refused, never merged") from None
    if not isinstance(data, dict) or set(data) != _BUDGET_KEYS or data["schema"] != BUDGET_SCHEMA:
        raise ValueError(f"{path} is not a {BUDGET_SCHEMA} ledger; refused, never merged")
    if data["plan_sha256"] != limits.plan_sha256:
        raise PlanChangedError(f"{path} was written under plan_sha256 {data['plan_sha256']}; this plan is "
                               f"{limits.plan_sha256}; refused, never merged")
    if (data["arm"], data["stage"], data["limits"], data["allowance"]) != (
            limits.arm, limits.stage, limits.to_dict(), allowance.to_dict()):
        raise ValueError(f"{path} was written for other limits or another allowance of this plan; "
                         "refused, never merged")
    counters = data["counters"]
    if not isinstance(counters, dict) or set(counters) != _COUNTER_KEYS:
        raise ValueError(f"{path}: counters must hold exactly {sorted(_COUNTER_KEYS)}")
    _count(counters["requests"], "counters.requests")
    _count(counters["output_tokens"], "counters.output_tokens")
    if _real(counters["elapsed_seconds"], "counters.elapsed_seconds") < 0:
        raise ValueError(f"{path}: counters.elapsed_seconds must not be negative")
    if counters["estimated_usd"] is not None and _real(counters["estimated_usd"], "counters.estimated_usd") < 0:
        raise ValueError(f"{path}: counters.estimated_usd must not be negative")
    entries = data["in_flight"]
    if not isinstance(entries, list) or any(not isinstance(entry, dict) or set(entry) != {"request", "output_tokens"}
                                            for entry in entries):
        raise ValueError(f"{path}: in_flight must list request numbers and their reservations")
    for entry in entries:
        _count(entry["request"], "in_flight.request", minimum=1)
        _count(entry["output_tokens"], "in_flight.output_tokens")
    _count(data["unknown_outcomes"], "unknown_outcomes")
    _count(data["sessions"], "sessions")
    if data["status"] not in STATUSES:
        raise ValueError(f"{path}: status must be one of {list(STATUSES)}; got {data['status']!r}")
    if (data["status"] == INCOMPLETE) != (data["stop_reason"] is not None) or (
            data["stop_reason"] is not None and data["stop_reason"] not in STOP_REASONS):
        raise ValueError(f"{path}: stop_reason must name the limit of an incomplete stage, and only then")
    for name in ("created_at", "updated_at"):
        if not isinstance(data[name], str):
            raise ValueError(f"{path}: {name} must be a timestamp")
    return data


# ---------- the in-memory guard ----------

def _completion_tokens(usage: object) -> int | None:
    if not isinstance(usage, Mapping):
        return None
    tokens = usage.get("completion_tokens")
    return tokens if isinstance(tokens, int) and not isinstance(tokens, bool) and tokens >= 0 else None


class LedgerCallBudget(CallBudget):
    """A ``CallBudget`` whose every admission goes through the stage's :class:`BudgetLedger`.

    Pass it to the client in place of a plain ``CallBudget``. Its own cap is the
    stage's remaining request count; the ledger additionally checks the token, time
    and cost limits and raises :class:`AdmissionClosedError` (a
    ``BudgetExhaustedError``) when one does not fit. A finished request is charged
    its reported ``completion_tokens``, or its whole reservation when the usage is
    missing (a failed request without usage). ``RequestControl``'s guarded admission
    refuses on ``remaining == 0`` without calling ``_admit``; use the ledger as a
    context manager so that such a ``BudgetExhaustedError`` closes the stage too.
    """

    def __init__(self, ledger: BudgetLedger) -> None:
        if not isinstance(ledger, BudgetLedger):
            raise TypeError(f"ledger must be a BudgetLedger; got {type(ledger).__name__}")
        super().__init__(ledger.remaining_requests)
        self.ledger = ledger
        self._admissions: dict[int, Admission] = {}

    def _admit(self, **kwargs: Any) -> RequestEvent:
        admission = self.ledger.admit()
        try:
            event = super()._admit(**kwargs)
        except BaseException:
            self.ledger.finish(admission, output_tokens=0)  # nothing was sent
            raise
        self._admissions[event.call_id] = admission
        return event

    def _finish(self, event: RequestEvent, **changes: Any) -> None:
        super()._finish(event, **changes)
        admission = self._admissions.pop(event.call_id, None)
        if admission is not None:
            self.ledger.finish(admission, output_tokens=_completion_tokens(changes.get("usage")))


# ---------- presentation of the paid serving arm ----------

@dataclass(frozen=True)
class StagePresentation:
    """One serving-arm stage as presented: its limits and the SHA-256 of its limits file's bytes."""

    stage: str
    max_requests: int
    max_output_tokens: int
    max_seconds: int
    max_usd: float
    sha256: str


@dataclass(frozen=True)
class ServingPresentation:
    """What the operator approves: every serving-arm stage's limits in execution order, with the hashes.

    ``stage_limits_sha256`` is the SHA-256 of the canonical JSON list of
    ``[stage, file sha256]`` pairs in execution order; each file's sha256 is the one
    the serve receipt records for it. ``manifest_sha256`` is the serving manifest
    file's SHA-256, as in the serve receipts.
    """

    plan_sha256: str
    manifest: str
    manifest_sha256: str
    stage_limits_sha256: str
    workspace: str | None
    endpoint: str | None
    stages: tuple[StagePresentation, ...]
    output_tokens_per_request: int
    usd_per_second: float

    @property
    def total_requests(self) -> int:
        return sum(stage.max_requests for stage in self.stages)

    @property
    def total_output_tokens(self) -> int:
        return sum(stage.max_output_tokens for stage in self.stages)

    @property
    def total_seconds(self) -> int:
        return sum(stage.max_seconds for stage in self.stages)

    @property
    def max_usd_total(self) -> float:
        """The maximum cost: the sum of the stages' ``max_usd`` as written, summed exactly."""
        return float(sum((Decimal(str(stage.max_usd)) for stage in self.stages), Decimal(0)))

    def limits_for(self, stage: str) -> StudyLimits:
        for item in self.stages:
            if item.stage == stage:
                return StudyLimits(self.plan_sha256, SERVING_ARM, item.stage, item.max_requests,
                                   item.max_output_tokens, item.max_seconds, item.max_usd)
        raise ValueError(f"no serving-arm stage {stage!r}; the stages are {list(STAGES)}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": SERVING_ARM, "plan_sha256": self.plan_sha256, "manifest": self.manifest,
            "manifest_sha256": self.manifest_sha256, "stage_limits_sha256": self.stage_limits_sha256,
            "workspace": self.workspace, "endpoint": self.endpoint,
            "stages": [asdict(stage) for stage in self.stages],
            "total_requests": self.total_requests, "total_output_tokens": self.total_output_tokens,
            "total_seconds": self.total_seconds, "max_usd_total": self.max_usd_total,
            "output_tokens_per_request": self.output_tokens_per_request,
            "usd_per_second": self.usd_per_second, "cost_basis": COST_BASIS,
        }

    def render(self, record_path: Path | None = None) -> str:
        """Plain text for the operator: the limits to approve, the totals and the record to write."""
        lines = [
            "Serving arm (Modal, paid): nothing runs without an approval record matching the frozen plan.",
            f"Frozen plan:          {self.plan_sha256}",
            f"Serving manifest:     {self.manifest} (sha256 {self.manifest_sha256})",
            f"Workspace, endpoint:  {self.workspace}, {self.endpoint}",
            f"Stage limits sha256:  {self.stage_limits_sha256} ({len(self.stages)} files in execution order)",
            "",
            f"{'Stage':<18}{'Requests':>10}{'Output tokens':>16}{'Seconds':>10}{'Max USD':>10}",
        ]
        lines += [f"{stage.stage:<18}{stage.max_requests:>10}{stage.max_output_tokens:>16}{stage.max_seconds:>10}"
                  f"{stage.max_usd:>10}" for stage in self.stages]
        lines += [
            f"{'total':<18}{self.total_requests:>10}{self.total_output_tokens:>16}{self.total_seconds:>10}"
            f"{self.max_usd_total:>10}",
            "",
            f"Output tokens per request: at most {self.output_tokens_per_request} (the manifest's cap).",
            f"Cost basis: {self.usd_per_second} USD per app second, {COST_BASIS}.",
            f"Maximum cost of the arm: {self.max_usd_total} USD.",
            "",
            "To approve, write the approval record"
            + (f" at {record_path}" if record_path is not None else " under runs/finance-sim/approvals/")
            + ": the five assertions set to true after review, these hashes, the workspace and endpoint, "
              "provider_budget_reference, max_usd_total of at least the maximum cost, the presented limits "
              "unchanged, approved_by and approved_at (UTC).",
        ]
        return "\n".join(lines)


def _serving_rate(cost: Mapping[str, Any], resources: Resources) -> float:
    """Configured USD per second of the serving app and its expiry controller (as measure_modal_model prices them)."""
    return (compute_rate(cost, resources) + CONTROLLER_CORES * cost["physical_core_usd_per_second"]
            + CONTROLLER_GIB * cost["memory_gib_usd_per_second"])


def _check_arm_ceiling(stages: Sequence[StagePresentation], modal_plan: ModalPlan) -> None:
    """The stages together must fit the manifest's reviewed ceiling for the whole arm.

    Requests and output tokens count every stage; seconds and USD count the GPU
    stages only, since the manifest's GPU phase and target exclude the CPU-only
    weight staging.
    """
    ceiling = reviewed_ceiling(modal_plan)
    gpu = [stage for stage in stages if stage.stage != WEIGHT_STAGING]
    totals = {
        "max_requests": sum(stage.max_requests for stage in stages),
        "max_output_tokens": sum(stage.max_output_tokens for stage in stages),
        "max_seconds": sum(stage.max_seconds for stage in gpu),
        "max_usd": sum((Decimal(str(stage.max_usd)) for stage in gpu), Decimal(0)),
    }
    for name, total in totals.items():
        limit = Decimal(str(ceiling[name])) if name == "max_usd" else ceiling[name]
        if total > limit:
            raise ValueError(f"The serving arm's stages together need {name} = {total}, above the manifest's "
                             f"reviewed ceiling of {ceiling[name]} for the whole arm")


def present_serving_arm(plan: StudyPlan, manifest_path: str | Path,
                        stage_limit_paths: Sequence[str | Path]) -> ServingPresentation:
    """Present the paid serving arm's limits for approval (Requirement 11.2); local files only.

    ``stage_limit_paths`` holds one StageLimits file per stage in execution order
    (:data:`~deploy.modal.stage_limits.STAGES`). Each must fit the manifest's
    reviewed ceiling and carry the frozen plan's hash, and together they must fit
    the ceiling of the whole arm.
    """
    if not isinstance(plan, StudyPlan):
        raise TypeError(f"plan must be a StudyPlan; got {type(plan).__name__}")
    manifest = Path(manifest_path)
    manifest_bytes = manifest.read_bytes()
    modal_plan = load_modal_plan(manifest)
    if modal_plan.serving_manifest != GEMMA_MANIFEST:
        raise ValueError(f"The serving arm runs the reviewed Gemma 3 4B manifest {GEMMA_MANIFEST}; "
                         f"{manifest} is {modal_plan.serving_manifest}")
    if modal_plan.artifacts.model not in plan.months.models:
        raise ValueError(f"The serving model {modal_plan.artifacts.model} is not a model of the frozen plan "
                         f"({list(plan.months.models)})")
    if isinstance(stage_limit_paths, str | bytes | Path) or not isinstance(stage_limit_paths, Sequence):
        raise TypeError("stage_limit_paths must be a sequence of paths")
    if len(stage_limit_paths) != len(STAGES):
        raise ValueError(f"The serving arm needs one stage limits file per stage, in execution order "
                         f"({', '.join(STAGES)}); got {len(stage_limit_paths)}")
    stages = []
    for position, (expected, path) in enumerate(zip(STAGES, stage_limit_paths, strict=True), start=1):
        data = Path(path).read_bytes()
        limits = parse_stage_limits(data, modal_plan)
        if limits.stage != expected:
            raise ValueError(f"Stage limits files must be in execution order: file {position} must hold "
                             f"{expected}; {path} holds {limits.stage}")
        try:
            ensure_unchanged(plan, limits.plan_sha256)
        except PlanChangedError as exc:
            raise PlanChangedError(f"{path}: {exc}") from None
        stages.append(StagePresentation(limits.stage, limits.max_requests, limits.max_output_tokens,
                                        limits.max_seconds, limits.max_usd, digest(data)))
    _check_arm_ceiling(stages, modal_plan)
    cost = json.loads(manifest_bytes)["cost"]
    return ServingPresentation(
        plan_sha256=plan.plan_sha256, manifest=modal_plan.serving_manifest,
        manifest_sha256=digest(manifest_bytes),
        stage_limits_sha256=sha256_of([[stage.stage, stage.sha256] for stage in stages]),
        workspace=modal_plan.target.workspace, endpoint=modal_plan.target.endpoint, stages=tuple(stages),
        output_tokens_per_request=modal_plan.limits.output_tokens_per_post,
        usd_per_second=_serving_rate(cost, modal_plan.resources),
    )


# ---------- the approval record ----------

def approval_path(plan_sha256: str, approvals_dir: str | Path = APPROVALS_DIR) -> Path:
    """``<approvals_dir>/<plan_sha256>.json``: one private approval record per frozen plan."""
    return Path(approvals_dir) / f"{_sha256(plan_sha256, 'plan_sha256')}.json"


def approval_template(presentation: ServingPresentation) -> dict[str, Any]:
    """A record to review and complete: every assertion false, the acknowledgement empty."""
    if not isinstance(presentation, ServingPresentation):
        raise TypeError(f"presentation must be a ServingPresentation; got {type(presentation).__name__}")
    return {
        "schema": APPROVAL_SCHEMA,
        **dict.fromkeys(APPROVAL_ASSERTIONS, False),
        "plan_sha256": presentation.plan_sha256, "manifest_sha256": presentation.manifest_sha256,
        "stage_limits_sha256": presentation.stage_limits_sha256,
        "workspace": presentation.workspace, "endpoint": presentation.endpoint,
        "provider_budget_reference": "", "max_usd_total": presentation.max_usd_total,
        "presented": _json_form(presentation.to_dict()), "approved_by": "", "approved_at": "",
    }


@dataclass(frozen=True)
class ServingApproval:
    """A validated approval record and the presentation it approves; the serving ledger requires one."""

    path: Path
    presentation: ServingPresentation
    record: Mapping[str, Any]

    @property
    def plan_sha256(self) -> str:
        return self.presentation.plan_sha256

    def limits_for(self, stage: str) -> StudyLimits:
        return self.presentation.limits_for(stage)


def _refused(detail: str) -> ApprovalError:
    return ApprovalError(f"Serving arm refused before any request: {detail}")


def _shared_gate_detail(code: str, record: Mapping[str, Any], hashes: Mapping[str, str],
                        modal_plan: ModalPlan) -> str:
    if code == "execution_approval_or_prerequisites_missing":
        unset = [name for name in APPROVAL_ASSERTIONS if record.get(name) is not True]
        return f"the operator has not set these assertions to true: {', '.join(unset)} ({code})"
    if code == "approval_plan_mismatch":
        differing = [name for name, value in hashes.items() if record.get(name) != value]
        return (f"the record's {', '.join(differing)} differ from the frozen plan, manifest or stage limits; "
                f"present them again and approve the current values ({code})")
    if code == "explicit_workspace_and_endpoint_required":
        return (f"the manifest must name its workspace and endpoint (workspace {modal_plan.target.workspace!r}, "
                f"endpoint {modal_plan.target.endpoint!r}) and the record's workspace "
                f"{record.get('workspace')!r} must be the manifest's ({code})")
    if code == "reviewed_provider_budget_required":
        return f"provider_budget_reference must name the reviewed provider budget ({code})"
    return code


def _utc_timestamp(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        return False
    return moment.utcoffset() is not None and moment.utcoffset().total_seconds() == 0


def require_serving_approval(plan: StudyPlan, manifest_path: str | Path, stage_limit_paths: Sequence[str | Path], *,
                             approvals_dir: str | Path = APPROVALS_DIR) -> ServingApproval:
    """Refuse the serving arm unless its approval record matches the frozen plan (Requirement 11.2).

    The presentation is rebuilt from the files, so a record approves exactly the
    current plan, manifest and stage limits. Raises :class:`ApprovalError` naming
    what is missing or differs; nothing is sent and nothing is written.
    """
    presentation = present_serving_arm(plan, manifest_path, stage_limit_paths)
    modal_plan = load_modal_plan(Path(manifest_path))
    path = approval_path(presentation.plan_sha256, approvals_dir)
    if not path.is_file():
        raise _refused(f"no approval record at {path}. Present the limits (python -m scripts.finance_sim.limits "
                       f"present), review them and write the record for plan {presentation.plan_sha256}")
    try:
        record = _strict_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise _refused(f"the approval record at {path} is unreadable: {exc}") from None
    if not isinstance(record, dict):
        raise _refused(f"the approval record at {path} does not hold a JSON object")
    if record.get("plan_sha256") != presentation.plan_sha256:
        raise _refused(f"the approval record at {path} names plan_sha256 {record.get('plan_sha256')!r}, but the "
                       f"frozen plan is {presentation.plan_sha256}; present this plan's limits and approve them")
    hashes = {"plan_sha256": presentation.plan_sha256, "manifest_sha256": presentation.manifest_sha256,
              "stage_limits_sha256": presentation.stage_limits_sha256}
    try:
        shared = approval_record(path, modal_plan, hashes)
    except ValueError as exc:
        raise _refused(_shared_gate_detail(str(exc), record, hashes, modal_plan)) from None
    if shared != record:
        raise _refused(f"the approval record at {path} changed while it was read")
    missing, undeclared = sorted(_APPROVAL_KEYS - set(record)), sorted(set(record) - _APPROVAL_KEYS)
    if missing or undeclared:
        raise _refused(f"the approval record must hold exactly the declared fields; missing {missing}, "
                       f"undeclared {undeclared}")
    if record["schema"] != APPROVAL_SCHEMA:
        raise _refused(f"the approval record's schema must be {APPROVAL_SCHEMA!r}; got {record['schema']!r}")
    if record["endpoint"] != modal_plan.target.endpoint:
        raise _refused(f"the record's endpoint {record['endpoint']!r} is not the manifest's "
                       f"{modal_plan.target.endpoint!r}")
    total = record["max_usd_total"]
    if (isinstance(total, bool) or not isinstance(total, int | float) or not math.isfinite(total)
            or total < presentation.max_usd_total):
        raise _refused(f"max_usd_total must be a number of at least the presented maximum cost of "
                       f"{presentation.max_usd_total} USD; got {total!r}")
    if record["presented"] != _json_form(presentation.to_dict()):
        raise _refused("the presented limits in the record differ from the frozen stage limits; present them "
                       "again and approve the current values")
    if not isinstance(record["approved_by"], str) or not record["approved_by"].strip():
        raise _refused("approved_by must name the operator who reviewed the presented limits")
    if not _utc_timestamp(record["approved_at"]):
        raise _refused(f"approved_at must be an ISO 8601 UTC timestamp; got {record['approved_at']!r}")
    return ServingApproval(path, presentation, MappingProxyType(record))


# ---------- operator commands ----------

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Present the paid serving arm's limits or check its approval "
                                                 "record; local files only, nothing is sent or written.")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, text in (("present", "print the limits to approve and an approval record template"),
                       ("check", "validate the approval record against the frozen plan")):
        command = commands.add_parser(name, help=text)
        command.add_argument("--plan", type=Path, required=True, help="the frozen plan file")
        command.add_argument("--manifest", type=Path, required=True, help="the serving manifest")
        command.add_argument("--stage-limits", type=Path, nargs=len(STAGES), required=True, metavar="FILE",
                             help="one stage limits file per stage, in order: " + ", ".join(STAGES))
        command.add_argument("--approvals", type=Path, default=APPROVALS_DIR, help="the approval records folder")
    args = parser.parse_args(argv)
    try:
        plan = load_plan(args.plan)
        if args.command == "present":
            presentation = present_serving_arm(plan, args.manifest, args.stage_limits)
            print(presentation.render(approval_path(plan.plan_sha256, args.approvals)))
            print("\nApproval record template (set each assertion to true only after review):")
            print(json.dumps(approval_template(presentation), indent=2))
            return EXIT_COMPLETE
        approval = require_serving_approval(plan, args.manifest, args.stage_limits, approvals_dir=args.approvals)
    except ApprovalError as exc:
        print(exc, file=sys.stderr)
        return EXIT_REFUSED
    except (OSError, ValueError, TypeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_UNFINISHED
    print(f"approved: {approval.path} matches plan {approval.plan_sha256}; maximum cost "
          f"{approval.presentation.max_usd_total} USD")
    return EXIT_COMPLETE


if __name__ == "__main__":
    raise SystemExit(main())
