"""Pilot and power gate (task 7.8; Requirements 8.9 and 11.3).

Pilot. The frozen pilot months of the month split run through the guarded
execution of task 7.7 (:func:`~scripts.finance_sim.execute.execute_items` with
the stage ``pilot``) on the gated arm the plan names (``plan.pilot.arm``), one
recorded ensemble of the pilot's own ``draws`` and ``min_parsed`` per month and
rendering. Every size comes from the frozen plan, never from this module.

Noise. Per pilot month the noise statistic is the identifying-minus-anonymised
correctness (1 correct, 0 not) of the plan's noise variant (``plan.pilot.
noise_variant``, the ensemble top-k variant), a value in {-1, 0, 1}. A month
whose noise variant has no correctness in either rendering (no decision because
fewer draws parsed than the pilot's minimum, an unresolved component, or no
realized returns) is reported as missing with the reason and left out of the
estimate, never substituted (8.10). Because the pilot's minimum can equal its
draw count, one interrupted request loses its month; such months are counted
separately (``lost_to_interruption``). ``noise_sd`` is the sample standard
deviation (ddof 1) of the statistic over the months that have it.

Power. :func:`~recall_guard.core.significance.simulated_power_two_sample`
evaluates the plan's ``effect_size`` on the pre-cutoff minus post-cutoff gap of
that statistic, with the frozen evaluation month counts of the pilot section
(``pre_cutoff_evaluation_months``, ``post_cutoff_evaluation_months``), the
pilot's ``noise_sd`` applied to both periods, the plan's ``alpha``, its
``n_resamples`` as the number of simulations and its ``seed``. The pilot months
are all before the cutoff, so applying their noise to the post-cutoff period
assumes equal variance in both periods.

Gate (8.9, 11.3). It gates the gated arm only. At an expected power of at least
``power_threshold`` the decision is ``continue``; otherwise, or when the power
cannot be computed (fewer than two months with the statistic, or no variation
in it), it is ``stop_for_decision`` with the expected power (or the reason it is
missing) and the options, and the full run of that arm waits for a decision. The
gate is recorded privately under ``runs/finance-sim/gates/<plan_sha256>.json``
and is never replaced by a different one. :func:`require_gate_passed` is the
check before the gated arm's full run: it recomputes the gate from the stored
pilot results, requires the recorded gate to match, and refuses a stopped gate.
The serving arm is not gated; it reports its achieved power
(:func:`achieved_power`) from its realised counts.

Exclusion. Pilot items size the gate and enter no later analysis:
:func:`confirmatory_results` keeps the results of one analysed stage and drops
every pilot item with the reason, and :func:`achieved_power` uses it.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from recall_guard.core.completion import CompletionClient
from recall_guard.core.full_distribution import FullDistributionClient, FullDistributionEvidence
from recall_guard.core.significance import simulated_power_two_sample
from scripts.finance_sim.calibrate import ANONYMISED, IDENTIFYING, CalibrationRecord, EvidenceSettings, macro_item_id
from scripts.finance_sim.execute import (
    INTERRUPTED,
    RESULTS_DIR,
    ExecutionDraw,
    ExecutionItem,
    ExecutionRun,
    ItemResult,
    execute_items,
    load_result,
    macro_items,
    planned_keys,
    result_path,
)
from scripts.finance_sim.items import MacroItem
from scripts.finance_sim.limits import ARMS, RUNS_ROOT, BudgetLedger
from scripts.finance_sim.plan import StudyPlan, ensure_unchanged, require_complete

PILOT_STAGE = "pilot"
#: The stage the pre-registered macro analyses read (7.10, 7.11, 7.12).
CONFIRMATORY_STAGE = "full_run"
CONTINUE = "continue"
STOP_FOR_DECISION = "stop_for_decision"
DECISIONS: tuple[str, ...] = (CONTINUE, STOP_FOR_DECISION)
GATE_SCHEMA = "finance-sim-power-gate/1"
#: Private gate records, one per frozen plan: ``<plan_sha256>.json``.
GATES_DIR = RUNS_ROOT / "gates"
#: The difference is identifying minus anonymised.
DIFFERENCE_RENDERINGS: tuple[str, str] = (IDENTIFYING, ANONYMISED)
NOISE_STATISTIC = (
    "per month, the identifying minus anonymised correctness (1 correct, 0 not) of the plan's noise variant, a value "
    "in {-1, 0, 1}; a month without a correctness in either rendering is missing with its reason, never substituted; "
    "noise_sd is the sample standard deviation (ddof 1) over the months that have the statistic"
)
POWER_METHOD = (
    "recall_guard.core.significance.simulated_power_two_sample: one-sided pooled two-sample t test of the "
    "pre-cutoff minus post-cutoff mean of the noise statistic at the plan's effect_size, the critical value "
    "simulated under effect 0; both periods carry the same noise_sd; the plan's alpha, n_resamples simulations "
    "and the plan's seed"
)
#: What the operator can decide when the gate stops (design: wider window, more draws).
STOP_OPTIONS: tuple[str, ...] = (
    "widen the window: more evaluation months before and after the cutoff, frozen in a new plan with its own pilot",
    "more draws per rendering, frozen in a new plan with its own pilot",
    "stop the gated arm's full run and report it as stopped at this expected power with the cause (Requirement 11.5)",
)
PILOT_EXCLUDED = ("a pilot item: pilot months size the power gate and are excluded from every later analysis "
                  "(Requirement 8.9)")
TOO_FEW_MONTHS = ("fewer than two months carry the noise statistic, so its standard deviation and the expected power "
                  "cannot be computed")
NO_VARIATION = ("the noise statistic does not vary over the months that carry it (standard deviation 0), so the noise "
                "cannot be estimated from them and the power cannot be computed")
TOO_FEW_PERIOD_MONTHS = ("fewer than two months carry the noise statistic before or after the cutoff, so the pooled "
                         "standard deviation and the achieved power cannot be computed")


class PilotError(ValueError):
    """Pilot results, a gate or a gate record are refused."""


class GateStoppedError(RuntimeError):
    """The power gate stopped the gated arm: its full run waits for a decision."""

    def __init__(self, gate: PowerGate) -> None:
        super().__init__(gate.summary())
        self.gate = gate


# ---------- validation helpers ----------

def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PilotError(f"{name} must be a non-empty string; got {value!r}")
    return value


def _finite(value: object, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise PilotError(f"{name} must be a finite number or None; got {value!r}")
    return float(value)


def _exact(data: object, keys: Sequence[str], name: str) -> Mapping[str, Any]:
    if not isinstance(data, Mapping):
        raise PilotError(f"{name} must be a JSON object")
    if set(data) != set(keys):
        raise PilotError(f"{name} fields differ: missing {sorted(set(keys) - set(data))}, "
                         f"undeclared {sorted(map(str, set(data) - set(keys)))}")
    return data


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        raise PilotError(f"duplicate keys {sorted({key for key in keys if keys.count(key) > 1})}")
    return dict(pairs)


def _no_constant(name: str) -> Any:
    raise PilotError(f"non-finite number {name}")


def _read_json(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PilotError(f"{path} cannot be read: {exc}") from None
    try:
        return json.loads(text, object_pairs_hook=_no_duplicate_keys, parse_constant=_no_constant)
    except PilotError as exc:
        raise PilotError(f"{path}: {exc}") from None
    except ValueError as exc:
        raise PilotError(f"{path} is not valid JSON: {exc}") from None


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


def _results(results: Iterable[ItemResult] | Mapping[Any, ItemResult]) -> tuple[ItemResult, ...]:
    if isinstance(results, Mapping):
        results = results.values()
    if isinstance(results, str | bytes) or not isinstance(results, Iterable):
        raise TypeError("results must be a sequence of ItemResult values or a mapping to them")
    values = tuple(results)
    for result in values:
        if not isinstance(result, ItemResult):
            raise TypeError(f"results must be ItemResult values; got {type(result).__name__}")
    return values


# ---------- the noise statistic per month ----------

_MONTH_KEYS = ("item_id", "pre_cutoff", "identifying", "anonymised", "difference", "missing_reason",
               "interrupted_draws")


@dataclass(frozen=True)
class MonthNoise:
    """One month's noise statistic: each rendering's correctness of the noise variant and their difference.

    ``identifying`` and ``anonymised`` are the noise variant's ``correct`` (None
    when it has none); ``missing_reason`` says why the month has no difference,
    and ``interrupted_draws`` counts the draws of both renderings whose request
    was interrupted (``request_error:interrupted``).
    """

    item_id: str
    pre_cutoff: bool
    identifying: bool | None
    anonymised: bool | None
    missing_reason: str | None
    interrupted_draws: int

    def __post_init__(self) -> None:
        _text(self.item_id, "item_id")
        if not isinstance(self.pre_cutoff, bool):
            raise PilotError(f"pre_cutoff must be a bool; got {self.pre_cutoff!r}")
        for name in ("identifying", "anonymised"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, bool):
                raise PilotError(f"{name} must be a bool or None; got {value!r}")
        complete = self.identifying is not None and self.anonymised is not None
        if complete != (self.missing_reason is None):
            raise PilotError("a month has a missing_reason exactly when a rendering has no correctness")
        if self.missing_reason is not None:
            _text(self.missing_reason, "missing_reason")
        if not _is_int(self.interrupted_draws) or self.interrupted_draws < 0:
            raise PilotError(f"interrupted_draws must be a non-negative integer; got {self.interrupted_draws!r}")

    @property
    def difference(self) -> int | None:
        """Identifying minus anonymised correctness, in {-1, 0, 1}; None for a missing month."""
        if self.identifying is None or self.anonymised is None:
            return None
        return int(self.identifying) - int(self.anonymised)

    @property
    def lost_to_interruption(self) -> bool:
        """A missing month with an interrupted draw in either rendering."""
        return self.difference is None and self.interrupted_draws > 0

    def to_dict(self) -> dict[str, Any]:
        return {"item_id": self.item_id, "pre_cutoff": self.pre_cutoff, "identifying": self.identifying,
                "anonymised": self.anonymised, "difference": self.difference, "missing_reason": self.missing_reason,
                "interrupted_draws": self.interrupted_draws}

    @classmethod
    def from_dict(cls, data: object) -> MonthNoise:
        values = dict(_exact(data, _MONTH_KEYS, "month noise"))
        difference = values.pop("difference")
        month = cls(**values)
        if month.difference != difference:
            raise PilotError(f"month {month.item_id}: the recorded difference {difference!r} is not identifying "
                             f"minus anonymised ({month.difference!r})")
        return month


def _month_noise(item_id: str, pre_cutoff: bool, by_key: Mapping[tuple[str, str], ItemResult],
                 variant: str) -> MonthNoise:
    correct: dict[str, bool | None] = {}
    reasons: list[str] = []
    interrupted = 0
    for rendering in DIFFERENCE_RENDERINGS:
        result = by_key.get((item_id, rendering))
        if result is None:
            correct[rendering] = None
            reasons.append(f"{rendering}: no result for this rendering")
            continue
        interrupted += sum(draw.failure_code == INTERRUPTED for draw in result.recorded_draws)
        outcome = result.variant(variant)
        correct[rendering] = outcome.correct
        if outcome.missing is not None:
            reasons.append(f"{rendering}: {outcome.missing}")
        elif outcome.correct is None:
            reasons.append(f"{rendering}: {outcome.missing_reasons.get('correct', 'no correctness')}")
    return MonthNoise(item_id=item_id, pre_cutoff=pre_cutoff, identifying=correct[IDENTIFYING],
                      anonymised=correct[ANONYMISED], missing_reason="; ".join(reasons) if reasons else None,
                      interrupted_draws=interrupted)


def _spread(values: Sequence[int]) -> tuple[float | None, float | None]:
    """Mean and sample variance (ddof 1); the variance is None with fewer than two values."""
    if not values:
        return None, None
    mean = float(statistics.fmean(values))
    return mean, (float(statistics.variance(values)) if len(values) >= 2 else None)


# ---------- the pilot noise ----------

_NOISE_KEYS = ("plan_sha256", "arm", "noise_variant", "statistic", "months", "months_used", "months_missing",
               "months_lost_to_interruption", "mean_difference", "noise_variance", "noise_sd")


@dataclass(frozen=True)
class PilotNoise:
    """The pilot's noise statistic per pilot month and its spread over the months that carry it."""

    plan_sha256: str
    arm: str
    noise_variant: str
    months: tuple[MonthNoise, ...]

    def __post_init__(self) -> None:
        _text(self.plan_sha256, "plan_sha256")
        if self.arm not in ARMS:
            raise PilotError(f"arm must be one of {list(ARMS)}; got {self.arm!r}")
        _text(self.noise_variant, "noise_variant")
        months = tuple(self.months)
        if not months or not all(isinstance(month, MonthNoise) for month in months):
            raise PilotError("months must hold one MonthNoise per pilot month")
        if len({month.item_id for month in months}) != len(months):
            raise PilotError("months must name each pilot month once")
        object.__setattr__(self, "months", months)

    @property
    def differences(self) -> tuple[int, ...]:
        return tuple(month.difference for month in self.months if month.difference is not None)

    @property
    def months_used(self) -> int:
        return len(self.differences)

    @property
    def months_missing(self) -> int:
        return len(self.months) - self.months_used

    @property
    def months_lost_to_interruption(self) -> int:
        return sum(month.lost_to_interruption for month in self.months)

    @property
    def mean_difference(self) -> float | None:
        return _spread(self.differences)[0]

    @property
    def noise_variance(self) -> float | None:
        return _spread(self.differences)[1]

    @property
    def noise_sd(self) -> float | None:
        variance = self.noise_variance
        return None if variance is None else math.sqrt(variance)

    def to_dict(self) -> dict[str, Any]:
        return {"plan_sha256": self.plan_sha256, "arm": self.arm, "noise_variant": self.noise_variant,
                "statistic": NOISE_STATISTIC, "months": [month.to_dict() for month in self.months],
                "months_used": self.months_used, "months_missing": self.months_missing,
                "months_lost_to_interruption": self.months_lost_to_interruption,
                "mean_difference": self.mean_difference, "noise_variance": self.noise_variance,
                "noise_sd": self.noise_sd}

    @classmethod
    def from_dict(cls, data: object) -> PilotNoise:
        values = _exact(data, _NOISE_KEYS, "pilot noise")
        if not isinstance(values["months"], list):
            raise PilotError("months must be a list")
        noise = cls(plan_sha256=values["plan_sha256"], arm=values["arm"], noise_variant=values["noise_variant"],
                    months=tuple(MonthNoise.from_dict(month) for month in values["months"]))
        if noise.to_dict() != dict(values):
            raise PilotError("the recorded pilot noise does not recompute from its months")
        return noise


def estimate_noise(plan: StudyPlan, results: Iterable[ItemResult] | Mapping[Any, ItemResult]) -> PilotNoise:
    """The pilot noise from the complete pilot stage's results (Requirement 8.9).

    Every result must be the plan's, of the pilot arm and stage, a planned item
    and rendering, given once; every planned pilot item and rendering must be
    present, so the gate never reads a partial pilot.
    """
    require_complete(plan)
    arm = plan.pilot.arm
    planned = planned_keys(plan, arm, PILOT_STAGE)
    by_key: dict[tuple[str, str], ItemResult] = {}
    for result in _results(results):
        ensure_unchanged(plan, result.plan_sha256)
        if (result.arm, result.stage) != (arm, PILOT_STAGE):
            raise PilotError(f"the pilot noise reads the {arm} arm's {PILOT_STAGE} results only; got a "
                             f"{result.arm}/{result.stage} result for {result.key}")
        if result.key not in planned:
            raise PilotError(f"{result.key} is not an item and rendering of the frozen pilot")
        if result.key in by_key:
            raise PilotError(f"{result.key} is given more than once")
        by_key[result.key] = result
    absent = [key for key in planned if key not in by_key]
    if absent:
        raise PilotError(f"the pilot is not complete: {len(absent)} of {len(planned)} ensembles have no result "
                         f"(first {absent[0]}); complete the pilot stage before the gate")
    variant = plan.pilot.noise_variant
    pre = set(plan.months.pre_cutoff[plan.arms.registry_ids[arm]])
    months = tuple(_month_noise(macro_item_id(month), month in pre, by_key, variant) for month in plan.months.pilot)
    return PilotNoise(plan_sha256=plan.plan_sha256, arm=arm, noise_variant=variant, months=months)


# ---------- the gate ----------

_GATE_KEYS = ("schema", "plan_sha256", "arm", "stage", "noise", "effect_size", "alpha", "n_sim", "seed",
              "power_threshold", "pre_cutoff_months", "post_cutoff_months", "expected_power", "decision", "reason",
              "options", "method")


@dataclass(frozen=True)
class PowerGate:
    """The gate of the gated arm: the expected power at the frozen sizes and the decision.

    ``decision`` is ``continue`` exactly when ``expected_power`` is at least
    ``power_threshold``; otherwise ``stop_for_decision`` with ``reason`` and the
    ``options``. ``expected_power`` is None when the pilot noise cannot give it,
    with the reason.
    """

    plan_sha256: str
    arm: str
    noise: PilotNoise
    effect_size: float
    alpha: float
    n_sim: int
    seed: int
    power_threshold: float
    pre_cutoff_months: int
    post_cutoff_months: int
    expected_power: float | None
    decision: str
    reason: str | None
    options: tuple[str, ...]
    method: str = POWER_METHOD

    def __post_init__(self) -> None:
        _text(self.plan_sha256, "plan_sha256")
        if not isinstance(self.noise, PilotNoise) or (self.noise.plan_sha256, self.noise.arm) != (self.plan_sha256,
                                                                                                  self.arm):
            raise PilotError("the gate's noise must be the pilot noise of its plan and arm")
        for name in ("effect_size", "alpha", "power_threshold"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        for name in ("n_sim", "seed", "pre_cutoff_months", "post_cutoff_months"):
            if not _is_int(getattr(self, name)):
                raise PilotError(f"{name} must be an integer; got {getattr(self, name)!r}")
        power = _finite(self.expected_power, "expected_power")
        if power is not None and not 0.0 <= power <= 1.0:
            raise PilotError(f"expected_power must be in [0, 1]; got {power}")
        object.__setattr__(self, "expected_power", power)
        passed = power is not None and power >= self.power_threshold
        if self.decision != (CONTINUE if passed else STOP_FOR_DECISION):
            raise PilotError(f"the decision must be {CONTINUE!r} exactly when the expected power is at least the "
                             f"threshold {self.power_threshold}; got {self.decision!r} at {power!r}")
        options = tuple(self.options)
        if passed:
            if self.reason is not None or options:
                raise PilotError("a gate that continues carries no stop reason or options")
        else:
            _text(self.reason, "reason")
            if not options or not all(isinstance(option, str) and option.strip() for option in options):
                raise PilotError("a gate that stops lists the options for the decision")
        object.__setattr__(self, "options", options)
        _text(self.method, "method")

    def summary(self) -> str:
        """The gate in plain prose, for the operator and the stage receipt."""
        noise = self.noise
        power = "not computable" if self.expected_power is None else f"{self.expected_power:.3f}"
        sd = "none" if noise.noise_sd is None else f"{noise.noise_sd:.3f}"
        text = (f"Power gate of the {self.arm} arm under plan {self.plan_sha256}: expected power {power} for the "
                f"pre-registered effect {self.effect_size:g} (threshold {self.power_threshold:g}, alpha "
                f"{self.alpha:g}, {self.pre_cutoff_months} pre-cutoff and {self.post_cutoff_months} post-cutoff "
                f"months), noise_sd {sd} from {noise.months_used} of {len(noise.months)} pilot months "
                f"({noise.months_missing} missing, {noise.months_lost_to_interruption} lost to an interrupted draw). "
                f"Decision: {self.decision}.")
        if self.decision == STOP_FOR_DECISION:
            text += f" Reason: {self.reason}. The full run waits for a decision. Options: " + "; ".join(
                f"({number}) {option}" for number, option in enumerate(self.options, start=1)) + "."
        return text

    def to_dict(self) -> dict[str, Any]:
        return {"schema": GATE_SCHEMA, "plan_sha256": self.plan_sha256, "arm": self.arm, "stage": PILOT_STAGE,
                "noise": self.noise.to_dict(), "effect_size": self.effect_size, "alpha": self.alpha,
                "n_sim": self.n_sim, "seed": self.seed, "power_threshold": self.power_threshold,
                "pre_cutoff_months": self.pre_cutoff_months, "post_cutoff_months": self.post_cutoff_months,
                "expected_power": self.expected_power, "decision": self.decision, "reason": self.reason,
                "options": list(self.options), "method": self.method}

    @classmethod
    def from_dict(cls, data: object) -> PowerGate:
        values = dict(_exact(data, _GATE_KEYS, "power gate"))
        if values.pop("schema") != GATE_SCHEMA:
            raise PilotError(f"unsupported power gate schema; expected {GATE_SCHEMA!r}")
        if values.pop("stage") != PILOT_STAGE:
            raise PilotError(f"a power gate belongs to the {PILOT_STAGE} stage")
        if not isinstance(values["options"], list):
            raise PilotError("options must be a list")
        values.update(noise=PilotNoise.from_dict(values["noise"]), options=tuple(values["options"]))
        return cls(**values)


def power_gate(plan: StudyPlan, noise: PilotNoise) -> PowerGate:
    """The gate of the gated arm from its pilot noise, with every size read from the frozen plan (8.9, 11.3)."""
    require_complete(plan)
    if not isinstance(noise, PilotNoise):
        raise TypeError(f"noise must be a PilotNoise; got {type(noise).__name__}")
    ensure_unchanged(plan, noise.plan_sha256)
    pilot, analysis = plan.pilot, plan.analysis
    if not plan.arms.arms[pilot.arm].gated or noise.arm != pilot.arm:
        raise PilotError(f"the power gate gates the plan's gated pilot arm {pilot.arm!r} only")
    if noise.noise_variant != pilot.noise_variant:
        raise PilotError(f"the pilot noise must be that of the frozen noise variant {pilot.noise_variant!r}")
    if [month.item_id for month in noise.months] != [macro_item_id(month) for month in plan.months.pilot]:
        raise PilotError("the pilot noise must hold the frozen pilot months in plan order")
    sd = noise.noise_sd
    power: float | None = None
    reason: str | None = None
    if sd is None:
        reason = TOO_FEW_MONTHS
    elif sd == 0.0:
        reason = NO_VARIATION
    else:
        power = simulated_power_two_sample(analysis.effect_size, sd, pilot.pre_cutoff_evaluation_months,
                                           pilot.post_cutoff_evaluation_months, alpha=analysis.alpha,
                                           n_sim=analysis.n_resamples, seed=plan.seed)
    passed = power is not None and power >= analysis.power_threshold
    if power is not None and not passed:
        reason = (f"the expected power {power:.3f} is below the threshold {analysis.power_threshold:g} at the frozen "
                  "sizes")
    return PowerGate(plan_sha256=plan.plan_sha256, arm=pilot.arm, noise=noise, effect_size=analysis.effect_size,
                     alpha=analysis.alpha, n_sim=analysis.n_resamples, seed=plan.seed,
                     power_threshold=analysis.power_threshold,
                     pre_cutoff_months=pilot.pre_cutoff_evaluation_months,
                     post_cutoff_months=pilot.post_cutoff_evaluation_months, expected_power=power,
                     decision=CONTINUE if passed else STOP_FOR_DECISION, reason=None if passed else reason,
                     options=() if passed else STOP_OPTIONS)


# ---------- the gate record ----------

def gate_path(plan_sha256: str, gates_dir: str | Path = GATES_DIR) -> Path:
    """``<gates_dir>/<plan_sha256>.json``."""
    if not isinstance(plan_sha256, str) or len(plan_sha256) != 64 or any(c not in "0123456789abcdef"
                                                                         for c in plan_sha256):
        raise PilotError(f"plan_sha256 must be 64 lowercase hex characters; got {plan_sha256!r}")
    return Path(gates_dir) / f"{plan_sha256}.json"


def write_gate(gate: PowerGate, gates_dir: str | Path = GATES_DIR) -> Path:
    """Record the gate privately; the same gate is a no-op, replacing a different one is refused."""
    if not isinstance(gate, PowerGate):
        raise TypeError(f"gate must be a PowerGate; got {type(gate).__name__}")
    target = gate_path(gate.plan_sha256, gates_dir)
    payload = gate.to_dict()
    if target.exists():
        if _read_json(target) == json.loads(json.dumps(payload, allow_nan=False)):
            return target
        raise PilotError(f"refusing to replace the different power gate at {target}")
    _write_json(target, payload)
    return target


def load_gate(path: str | Path) -> PowerGate:
    """Read one gate record, refusing a malformed or inconsistent file."""
    return PowerGate.from_dict(_read_json(Path(path)))


def load_pilot_results(plan: StudyPlan, results_dir: str | Path = RESULTS_DIR) -> dict[tuple[str, str], ItemResult]:
    """The stored result of every planned pilot item and rendering; a missing one is refused."""
    require_complete(plan)
    arm = plan.pilot.arm
    loaded: dict[tuple[str, str], ItemResult] = {}
    for item_id, rendering in planned_keys(plan, arm, PILOT_STAGE):
        path = result_path(arm, PILOT_STAGE, item_id, rendering, results_dir)
        if not path.exists():
            raise PilotError(f"the pilot is not complete: no stored result for {(item_id, rendering)} at {path}")
        loaded[(item_id, rendering)] = load_result(path)
    return loaded


def require_gate_passed(plan: StudyPlan, arm: str, *, results_dir: str | Path = RESULTS_DIR,
                        gates_dir: str | Path = GATES_DIR) -> PowerGate | None:
    """The check before an arm's full run: None for an arm that is not gated, else the gate that continues.

    For the gated arm the gate is recomputed from the stored pilot results and
    must equal the recorded gate; a stopped gate raises
    :class:`GateStoppedError` (the full run waits for a decision), and a missing
    or differing record, or an incomplete pilot, raises :class:`PilotError`.
    """
    require_complete(plan)
    if arm not in ARMS:
        raise PilotError(f"arm must be one of {list(ARMS)}; got {arm!r}")
    if not plan.arms.arms[arm].gated:
        return None
    if arm != plan.pilot.arm:
        raise PilotError(f"the gated arm {arm!r} has no pilot in this plan")
    gate = power_gate(plan, estimate_noise(plan, load_pilot_results(plan, results_dir)))
    path = gate_path(plan.plan_sha256, gates_dir)
    if not path.exists():
        raise PilotError(f"no power gate is recorded for plan {plan.plan_sha256} at {path}; run the pilot first")
    if load_gate(path).to_dict() != gate.to_dict():
        raise PilotError(f"the recorded power gate at {path} does not recompute from the stored pilot results; "
                         "refused")
    if gate.decision != CONTINUE:
        raise GateStoppedError(gate)
    return gate


# ---------- the pilot run ----------

@dataclass(frozen=True)
class PilotRun:
    """The pilot's execution, its noise, the gate and where the gate is recorded."""

    run: ExecutionRun
    noise: PilotNoise
    gate: PowerGate
    gate_path: Path


def pilot_items(plan: StudyPlan, items: Iterable[MacroItem]) -> tuple[ExecutionItem, ...]:
    """Both renderings of every frozen pilot month, each equal to the item the plan froze."""
    return macro_items(plan, items, stage=PILOT_STAGE)


def run_pilot(
    plan: StudyPlan, items: Iterable[MacroItem], *, client: CompletionClient, ledger: BudgetLedger,
    calibrators: Mapping[str, CalibrationRecord], full_distribution: FullDistributionClient | None = None,
    evidence: EvidenceSettings | None = None, recorded: Iterable[ExecutionDraw] = (),
    on_draw: Callable[[ExecutionDraw], None] | None = None,
    on_request: Callable[[ExecutionItem, int, int | None], None] | None = None,
    on_evidence: Callable[[ExecutionItem, Mapping[int, FullDistributionEvidence]], None] | None = None,
    results_dir: str | Path = RESULTS_DIR, gates_dir: str | Path = GATES_DIR,
    sleep: Callable[[float], None] = time.sleep,
) -> PilotRun:
    """Run the frozen pilot through the guarded execution, then estimate the noise and record the gate.

    The pilot months run on the plan's gated arm with the ``pilot`` stage of
    :func:`~scripts.finance_sim.execute.execute_items`, which checks the items,
    the ledger, the client and the calibrators before any request and records,
    resumes and limits exactly as for every stage. A frozen limit or missing
    evidence raises from there, before any gate is computed. The gate is then
    computed from the complete pilot and recorded under ``gates_dir``.
    """
    require_complete(plan)
    run = execute_items(plan, pilot_items(plan, items), arm=plan.pilot.arm, stage=PILOT_STAGE, client=client,
                        ledger=ledger, calibrators=calibrators, full_distribution=full_distribution,
                        evidence=evidence, recorded=recorded, on_draw=on_draw, on_request=on_request,
                        on_evidence=on_evidence, results_dir=results_dir, sleep=sleep)
    noise = estimate_noise(plan, run.results)
    gate = power_gate(plan, noise)
    return PilotRun(run=run, noise=noise, gate=gate, gate_path=write_gate(gate, gates_dir))


# ---------- pilot items excluded from later analyses ----------

@dataclass(frozen=True)
class ConfirmatoryResults:
    """The results an analysis of one arm and stage reads, and the pilot items left out with the reason."""

    arm: str
    stage: str
    results: Mapping[tuple[str, str], ItemResult]
    excluded: Mapping[tuple[str, str], str]


def confirmatory_results(plan: StudyPlan, results: Iterable[ItemResult] | Mapping[Any, ItemResult], *, arm: str,
                         stage: str = CONFIRMATORY_STAGE) -> ConfirmatoryResults:
    """Keep the results of ``arm`` and ``stage`` and exclude every pilot item (Requirement 8.9).

    A result of the pilot stage, or of a pilot month, is excluded with
    :data:`PILOT_EXCLUDED`. Any other result must be the plan's, of ``arm`` and
    ``stage``, a planned item and rendering, given once; anything else is
    refused rather than silently dropped.
    """
    require_complete(plan)
    if arm not in ARMS:
        raise PilotError(f"arm must be one of {list(ARMS)}; got {arm!r}")
    if stage == PILOT_STAGE:
        raise PilotError("the pilot stage enters no later analysis")
    planned = set(planned_keys(plan, arm, stage))
    pilot_ids = {macro_item_id(month) for month in plan.months.pilot}
    kept: dict[tuple[str, str], ItemResult] = {}
    excluded: dict[tuple[str, str], str] = {}
    for result in _results(results):
        ensure_unchanged(plan, result.plan_sha256)
        if result.arm != arm:
            raise PilotError(f"the {arm} analysis got a {result.arm}-arm result for {result.key}")
        if result.stage == PILOT_STAGE or result.item_id in pilot_ids:
            if result.key in excluded:
                raise PilotError(f"{result.key} is given more than once")
            excluded[result.key] = PILOT_EXCLUDED
            continue
        if result.stage != stage:
            raise PilotError(f"the {stage} analysis got a {result.stage} result for {result.key}")
        if result.key not in planned:
            raise PilotError(f"{result.key} is not an item and rendering the plan sizes for {arm}/{stage}")
        if result.key in kept:
            raise PilotError(f"{result.key} is given more than once")
        kept[result.key] = result
    return ConfirmatoryResults(arm=arm, stage=stage, results=MappingProxyType(kept),
                               excluded=MappingProxyType(excluded))


# ---------- achieved power of an arm that is not gated ----------

@dataclass(frozen=True)
class AchievedPower:
    """The achieved power of one arm's full run for the pre-registered effect, from its realised counts (8.9).

    ``noise_sd`` is the pooled within-period standard deviation of the noise
    statistic over the evaluation months that carry it, ``pre_cutoff_months``
    and ``post_cutoff_months`` the realised counts of those months; ``power``
    is None with ``reason`` when they cannot give it. ``excluded`` lists the
    pilot items left out.
    """

    plan_sha256: str
    arm: str
    noise_variant: str
    months: tuple[MonthNoise, ...]
    excluded: Mapping[tuple[str, str], str]
    effect_size: float
    alpha: float
    n_sim: int
    seed: int
    pre_cutoff_months: int
    post_cutoff_months: int
    noise_sd: float | None
    power: float | None
    reason: str | None
    method: str = POWER_METHOD

    def to_dict(self) -> dict[str, Any]:
        return {"plan_sha256": self.plan_sha256, "arm": self.arm, "noise_variant": self.noise_variant,
                "statistic": NOISE_STATISTIC, "months": [month.to_dict() for month in self.months],
                "excluded": [{"item_id": item_id, "rendering": rendering, "reason": reason}
                             for (item_id, rendering), reason in self.excluded.items()],
                "effect_size": self.effect_size, "alpha": self.alpha, "n_sim": self.n_sim, "seed": self.seed,
                "pre_cutoff_months": self.pre_cutoff_months, "post_cutoff_months": self.post_cutoff_months,
                "noise_sd": self.noise_sd, "power": self.power, "reason": self.reason, "method": self.method}


def achieved_power(plan: StudyPlan, results: Iterable[ItemResult] | Mapping[Any, ItemResult], *,
                   arm: str) -> AchievedPower:
    """The achieved power of ``arm``'s full run, pilot items excluded; the serving arm reports this, ungated.

    Per planned evaluation month the noise statistic of the plan's noise
    variant; months without it (including a rendering with no result after an
    incomplete run) are missing with the reason. The power is
    :func:`~recall_guard.core.significance.simulated_power_two_sample` at the
    plan's effect size with the realised pre-cutoff and post-cutoff counts and
    their pooled within-period standard deviation.
    """
    analysed = confirmatory_results(plan, results, arm=arm)
    analysis, variant = plan.analysis, plan.pilot.noise_variant
    pre = set(plan.months.pre_cutoff[plan.arms.registry_ids[arm]])
    months_of = {macro_item_id(month): month for month in plan.months.evaluation}
    ordered = list(dict.fromkeys(item_id for item_id, _ in planned_keys(plan, arm, CONFIRMATORY_STAGE)))
    months = tuple(_month_noise(item_id, months_of[item_id] in pre, analysed.results, variant) for item_id in ordered)
    before = [month.difference for month in months if month.difference is not None and month.pre_cutoff]
    after = [month.difference for month in months if month.difference is not None and not month.pre_cutoff]
    sd: float | None = None
    power: float | None = None
    reason: str | None = None
    if len(before) < 2 or len(after) < 2:
        reason = TOO_FEW_PERIOD_MONTHS
    else:
        pooled = ((len(before) - 1) * _spread(before)[1] + (len(after) - 1) * _spread(after)[1]) / (
            len(before) + len(after) - 2)
        sd = math.sqrt(pooled)
        if sd == 0.0:
            reason = NO_VARIATION
        else:
            power = simulated_power_two_sample(analysis.effect_size, sd, len(before), len(after),
                                               alpha=analysis.alpha, n_sim=analysis.n_resamples, seed=plan.seed)
    return AchievedPower(plan_sha256=plan.plan_sha256, arm=arm, noise_variant=variant, months=months,
                         excluded=analysed.excluded, effect_size=analysis.effect_size, alpha=analysis.alpha,
                         n_sim=analysis.n_resamples, seed=plan.seed, pre_cutoff_months=len(before),
                         post_cutoff_months=len(after), noise_sd=sd, power=power, reason=reason)


__all__ = [
    "CONFIRMATORY_STAGE",
    "CONTINUE",
    "DECISIONS",
    "GATES_DIR",
    "GATE_SCHEMA",
    "NOISE_STATISTIC",
    "NO_VARIATION",
    "PILOT_EXCLUDED",
    "PILOT_STAGE",
    "POWER_METHOD",
    "STOP_FOR_DECISION",
    "STOP_OPTIONS",
    "TOO_FEW_MONTHS",
    "TOO_FEW_PERIOD_MONTHS",
    "AchievedPower",
    "ConfirmatoryResults",
    "GateStoppedError",
    "MonthNoise",
    "PilotError",
    "PilotNoise",
    "PilotRun",
    "PowerGate",
    "achieved_power",
    "confirmatory_results",
    "estimate_noise",
    "gate_path",
    "load_gate",
    "load_pilot_results",
    "pilot_items",
    "power_gate",
    "require_gate_passed",
    "run_pilot",
    "write_gate",
]
