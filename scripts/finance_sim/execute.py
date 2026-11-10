"""Guarded execution of study items (task 7.7; Requirements 5.8 and 8.1).

One recorded ensemble per item and rendering, every variant derived from it.

Generation pass (inside ``with ledger:``). For each item and rendering the
ensemble of the stage is drawn once: ``draws`` requests (the arm's, or the
pilot's on the pilot stage) of the frozen arm settings, the temperature of the
arm and seed ``seed_base + draw_index``, each request admitted by the stage's
:class:`~scripts.finance_sim.limits.BudgetLedger`. The ensemble is one
:func:`~recall_guard.harness.guarded_decision.guarded_run` call with the TOP_K
source and the arm's top-k calibrator. Draws an interrupted run recorded
pre-fill it (``recorded=``), so only the missing draw indices are issued, each
with its deterministic seed, and a recorded request is never issued again; its
``on_draw`` hook hands each finished draw to this module's ``on_draw`` before
the next request. A stage whose admission is closed is refused before the
client is called.

Evidence pass (full-distribution source only, after all generation and outside
the ledger session, with the full-distribution host loaded alone). One
``evidence_batch`` request per item and rendering carries the replies of its
parsed draws in draw order, with the settings of
:class:`~scripts.finance_sim.calibrate.EvidenceSettings`: by default the timeout,
busy retries and wait the frozen plan sets for the arm (``ArmPlan.evidence``) and
the kept alternatives the full-distribution calibrator was fitted on, which any
other settings must keep too. ``host_busy`` is retried
a bounded number of times after a wait; retries that run out, any other host
failure, a client timeout or a malformed batch raise
:class:`~scripts.finance_sim.calibrate.EvidenceUnavailableError`. The stage
then stays open (not incomplete, which is reserved for a frozen limit closing
admission), every recorded draw is kept, and a rerun repeats only the batches
still missing; a timed-out batch is never resent in the same run, because the
host may still be computing it. Each evidence object gives its draw's
full-distribution estimate as the guarded-decision API computes one per draw:
the features of ``features_from_full`` through the full-distribution
calibrator, with the SHA-256 of the evidence payload as its reference.

Variants (8.1). From the one ensemble of an item and rendering: ``top_k`` uses
the draws as recorded, ``full_distribution`` the same draws with each estimate
and evidence reference replaced by that draw's full-distribution estimate, and
``none`` the same draws without estimates, so its decision passes through
unadjusted by construction (status ``estimate_unavailable``, the no-adjustment
variant, not a failure). An ensemble variant reduces every draw; a single
variant reduces draw index 0 only. The variants are therefore paired on
identical replies. A variant the arm cannot compute (the serving arm's
full-distribution variants) is reported as missing with the plan's reason.

Per item, rendering and variant the result keeps the decision, the adjusted
decision, ``p_memorized``, the summary value of both (the defensive spread or
the outlook), the category, ``correct`` (``macro_regime.is_correct`` on the
variant's adjusted decision with the item's realized returns; when the
adjustment shrinks a non-zero spread to one that rounds to zero, on the raw
spread's sign, flagged ``sign_from_raw_spread`` with the reason), the scored
probability of the frozen plan (``ScoredProbability.probability`` per draw
mode, then ``.adjusted`` with the variant's ``p_memorized`` when its status is
``ok``, else unchanged), the first-draw rank of the summary value, the
components not assessed and, for every missing value, its reason. The item
result also keeps the recorded draws, the full-distribution estimates, the
per-draw summary values and categories and the identity per source, so the
ensemble analyses (7.11) can reduce any subset of draws again with
``decide_from_draws`` and no request.

Calibrators (4.1, 4.2, 6.3). One per estimate source of the arm, loaded with
:func:`~scripts.finance_sim.calibrate.load_calibrator` for the plan, arm, kind
and source, so the macro calibrator is never applied to EDGAR prompts. Before
any request the records must belong to the frozen plan, arm, item kind and
source, and every calibrator must match the client (the guard's identity
check); any mismatch is refused and nothing is sent.

Recording seam (the contract for the study recording, task 7.4):

- ``on_request(item, draw_index, seed)`` is called before each generation
  request is sent, with the draw's deterministic seed. The recorder persists
  the request then, before the send.
- ``on_draw(draw)`` receives every new draw after its request returned,
  before the next request, and again when its full-distribution estimate
  settles. The recorder keeps the latest version per draw.
- ``on_evidence(item, evidence)`` receives each evidence batch by draw index
  before the draws it settles.
- ``recorded`` takes the :class:`ExecutionDraw` values the recorder kept (a
  draw recorded twice, before and after its evidence settled, keeps the settled
  version). A request the recorder persisted but whose outcome it never heard
  (the process died while it was in flight) is passed back as the failed draw
  :meth:`ExecutionDraw.interrupted` builds: no reply, ``request_error:interrupted``,
  the SHA-256 of the empty string, the seed ``on_request`` reported, the arm's
  temperature and the top-k calibrator's fingerprint. It stays in its ensemble
  as a failed draw and is never sent again, so a resumed stage spends exactly
  one admission per draw and can still complete within limits sized to one
  ensemble per item and rendering.

``ExecutionDraw.to_dict`` and ``from_dict`` are the serialisation, and the
replies the draws hold are private. Results are written
privately under ``<results_dir>/<arm>/<stage>/<item_id>.<rendering>.json``; a
matching stored result is reused and no request is sent, and a stored result
that differs is refused, never merged or replaced.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import os
import re
import time
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

from recall_guard.core.completion import CompletionClient, ModelConfig, ModelResponse
from recall_guard.core.decision import DecisionDefinition
from recall_guard.core.distribution import RecordedDraw
from recall_guard.core.full_distribution import FullDistributionClient, FullDistributionEvidence
from recall_guard.core.request_control import RequestError
from recall_guard.dataset.edgar_outlook import outlook_definition, prompt_sha256
from recall_guard.dataset.macro_regime import RealizedReturns, is_correct, regime_definition
from recall_guard.harness.guarded_decision import (
    REQUEST_ERROR_PREFIX,
    DrawMode,
    GuardConfig,
    GuardedDecision,
    GuardIdentity,
    _guard_identity,
    decide_from_draws,
    guarded_run,
)
from recall_guard.mia.estimate_source import EstimateSource, SourceLabel, features_from_full
from scripts.finance_sim.calibrate import (
    ANONYMISED,
    CALIBRATORS_DIR,
    EDGAR,
    HOST_BUSY,
    IDENTIFYING,
    MACRO,
    OUTLOOK,
    CalibrationRecord,
    EvidenceSettings,
    EvidenceUnavailableError,
    calibrator_path,
    load_calibrator,
    macro_item_id,
    request_allowance,
)
from scripts.finance_sim.items import MacroItem
from scripts.finance_sim.limits import ARMS, OPEN, RUNS_ROOT, BudgetLedger, LedgerCallBudget
from scripts.finance_sim.plan import (
    FULL_DISTRIBUTION,
    NO_ADJUSTMENT,
    SIGN_DECIMALS,
    SINGLE_DRAW_INDEX,
    TOP_K,
    VARIANTS,
    ArmPlan,
    StudyPlan,
    ensure_unchanged,
    full_run_months,
    require_complete,
    sha256_of,
    variant_parts,
)

#: The stages that execute study items, with the kind of item each executes.
EXECUTION_STAGES: Mapping[str, str] = MappingProxyType({"pilot": MACRO, "full_run": MACRO,
                                                        "negative_control": EDGAR})
#: Renderings per item kind, in execution order.
KIND_RENDERINGS: Mapping[str, tuple[str, ...]] = MappingProxyType({MACRO: (IDENTIFYING, ANONYMISED),
                                                                   EDGAR: (OUTLOOK,)})
#: The evaluation decision definition of each item kind.
DEFINITIONS: Mapping[str, DecisionDefinition] = MappingProxyType({MACRO: regime_definition,
                                                                  EDGAR: outlook_definition})
RESULTS_DIR = RUNS_ROOT / "results"
RESULT_SCHEMA = "finance-sim-execution-result/1"
STATUSES: tuple[str, ...] = ("ok", "estimate_unavailable", "adjustment_skipped_weak_calibrator", "failed")
EDGAR_NO_OUTCOME = ("not applicable: an EDGAR outlook has no realized outcome; accuracy and the scored probability "
                    "are defined for macro items only")
REALIZED_MISSING = "no realized next-month returns for this month"
UNRESOLVED_SUMMARY = ("a component is unresolved (split into separated clusters across draws), so the decision has "
                      "no summary value")
#: Failure code of a request that was sent (``on_request`` was called) but whose outcome was lost.
INTERRUPTED = REQUEST_ERROR_PREFIX + "interrupted"
SIGN_FROM_RAW_REASON = ("the adjustment shrinks a non-zero spread to one that rounds to zero at 9 decimals "
                        "(p_memorized at or near 1); the adjustment keeps the decision's sign (Requirement 1.3), so "
                        "correctness is scored on the sign of the raw spread, and the adjustment acts on the scored "
                        "probability, which moves to 0.5")
_SOURCES: tuple[str, ...] = (TOP_K, FULL_DISTRIBUTION)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_ITEM_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_ESTIMATE_KEYS = ("estimate", "evidence_ref", "calibrator_fingerprint")
_DRAW_KEYS = ("item_id", "rendering", "prompt_sha256", "draw", "response", "calibrator_fingerprint",
              "full_distribution")
_VARIANT_KEYS = ("variant", "mode", "adjustment", "missing", "status", "reason", "decision", "adjusted",
                 "p_memorized", "summary_value", "adjusted_summary_value", "category", "correct",
                 "scored_probability", "adjusted_scored_probability", "first_draw_rank", "not_assessed",
                 "draws_sha256", "missing_reasons", "sign_from_raw_spread", "sign_reason")
_RESULT_KEYS = ("schema", "plan_sha256", "arm", "stage", "item_id", "kind", "rendering", "prompt_sha256",
                "ensemble_draws", "min_parsed", "identities", "recorded_draws", "full_distribution",
                "summary_values", "categories", "outcome", "variants")


class ExecutionError(ValueError):
    """Study items, their draws, calibrators or stored results are refused; nothing was sent for them."""


# ---------- validation helpers ----------

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _payload_sha256(payload: object) -> str:
    """SHA-256 of a JSON payload in canonical form, as the guarded-decision API references evidence."""
    return _sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False))


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_real(value: object) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExecutionError(f"{name} must be a non-empty string; got {value!r}")
    return value


def _hex(value: object, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ExecutionError(f"{name} must be a SHA-256 as 64 lowercase hex characters; got {value!r}")
    return value


def _finite(value: object, name: str) -> float | None:
    """A finite number as float, or None."""
    if value is None:
        return None
    if not _is_real(value) or not math.isfinite(value):
        raise ExecutionError(f"{name} must be a finite number or None; got {value!r}")
    return float(value)


def _scalar(value: object, name: str) -> Hashable | None:
    """A category that reads back from JSON unchanged: None, a bool, an int, a string or a finite float."""
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ExecutionError(f"{name} must be None, a bool, an int, a string or a finite float; got {value!r}")


def _exact(data: object, keys: Sequence[str], name: str) -> Mapping[str, Any]:
    if not isinstance(data, Mapping):
        raise ExecutionError(f"{name} must be a JSON object")
    if set(data) != set(keys):
        raise ExecutionError(f"{name} fields differ: missing {sorted(set(keys) - set(data))}, "
                             f"undeclared {sorted(map(str, set(data) - set(keys)))}")
    return data


def _vector(value: object, name: str) -> Mapping[str, float | None] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or not all(isinstance(key, str) and key for key in value):
        raise ExecutionError(f"{name} must map component names to numbers; got {value!r}")
    return MappingProxyType({key: _finite(number, f"{name}[{key!r}]") for key, number in value.items()})


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        raise ExecutionError(f"duplicate keys {sorted({key for key in keys if keys.count(key) > 1})}")
    return dict(pairs)


def _no_constant(name: str) -> Any:
    raise ExecutionError(f"non-finite number {name}")


def _read_json(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ExecutionError(f"{path} cannot be read: {exc}") from None
    try:
        return json.loads(text, object_pairs_hook=_no_duplicate_keys, parse_constant=_no_constant)
    except ExecutionError as exc:
        raise ExecutionError(f"{path}: {exc}") from None
    except ValueError as exc:
        raise ExecutionError(f"{path} is not valid JSON: {exc}") from None


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


# ---------- items ----------

@dataclass(frozen=True)
class ExecutionItem:
    """One study item in one rendering: its prompt and, for a macro item, its realized next-month returns.

    ``realized`` None means the month's accuracy and scored probability are
    reported as missing with ``realized_missing_reason``; an EDGAR item has no
    realized outcome.
    """

    item_id: str
    kind: str
    rendering: str
    prompt: str
    realized: RealizedReturns | None = None
    realized_missing_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.item_id, str) or not _ITEM_ID.fullmatch(self.item_id):
            raise ExecutionError(f"item_id must be letters, digits, dots, dashes or underscores; got {self.item_id!r}")
        if self.kind not in KIND_RENDERINGS:
            raise ExecutionError(f"kind must be one of {list(KIND_RENDERINGS)}; got {self.kind!r}")
        if self.rendering not in KIND_RENDERINGS[self.kind]:
            raise ExecutionError(f"a {self.kind} item has the renderings {list(KIND_RENDERINGS[self.kind])}; "
                                 f"got {self.rendering!r}")
        _text(self.prompt, "prompt")
        if self.realized is not None:
            if not isinstance(self.realized, RealizedReturns):
                raise TypeError(f"realized must be RealizedReturns or None; got {type(self.realized).__name__}")
            if self.kind != MACRO:
                raise ExecutionError("only a macro item has realized returns")
            if self.realized_missing_reason is not None:
                raise ExecutionError("an item with realized returns has no missing reason")
        elif self.realized_missing_reason is not None:
            _text(self.realized_missing_reason, "realized_missing_reason")

    @property
    def key(self) -> tuple[str, str]:
        return self.item_id, self.rendering

    @property
    def prompt_sha256(self) -> str:
        return prompt_sha256(self.prompt)


def _macro_item_sha256(item: MacroItem) -> str:
    """The frozen plan's hash of a macro item: its row hash and both prompt hashes (build_plan.item_hashes)."""
    entry = item.to_dict({})
    return sha256_of({"prompt_sha256": entry["prompt_sha256"], "row_sha256": entry["row_sha256"]})


def macro_items(plan: StudyPlan, items: Iterable[MacroItem], *, stage: str,
                arm: str | None = None) -> tuple[ExecutionItem, ...]:
    """Both renderings of every macro month of ``stage`` (evaluation or pilot months), in plan order.

    The full run's months follow the arm's role (:func:`~scripts.finance_sim.plan.full_run_months`):
    with ``arm`` they are that arm's (the probe arm's frozen probe evaluation
    months, the primary arm's every evaluation month); without it, every
    evaluation month, a superset of the probe arm's. Pilot months belong to the
    gated arm only. Each month's item must be present and equal to the item the
    plan froze (its row and prompt hashes), so a changed row or prompt is
    refused before any request. Items of other months are ignored.
    """
    require_complete(plan)
    if arm is not None and arm not in ARMS:
        raise ExecutionError(f"arm must be one of {list(ARMS)} or None; got {arm!r}")
    if stage == "pilot" and arm not in (None, plan.pilot.arm):
        raise ExecutionError(f"the pilot runs on the gated {plan.pilot.arm} arm only; the {arm} arm has no pilot")
    if stage == "full_run":
        months = plan.months.evaluation if arm is None else full_run_months(plan, arm)
    else:
        months = {"pilot": plan.months.pilot}.get(stage)
    if months is None:
        raise ExecutionError(f"macro items run in the full_run and pilot stages; got {stage!r}")
    by_id: dict[str, MacroItem] = {}
    for item in items:
        if not isinstance(item, MacroItem):
            raise TypeError(f"items must be MacroItem values; got {type(item).__name__}")
        if item.item_id in by_id:
            raise ExecutionError(f"macro item {item.item_id} is given more than once")
        by_id[item.item_id] = item
    built = []
    for month in months:
        item_id = macro_item_id(month)
        item = by_id.get(item_id)
        if item is None:
            raise ExecutionError(f"the macro item of {stage} month {month} is missing")
        if f"{item.decision_date:%Y-%m}" != month:
            raise ExecutionError(f"macro item {item_id} holds the row of {item.decision_date:%Y-%m}")
        if plan.items.get(item_id) != _macro_item_sha256(item):
            raise ExecutionError(f"macro item {item_id} differs from the item the plan froze (row or prompt hash)")
        prompts = item.prompts()
        realized, reason = item.outcome.realized, item.outcome.reason
        built.extend(ExecutionItem(item_id, MACRO, rendering, prompts[rendering], realized=realized,
                                   realized_missing_reason=None if realized is not None else reason or REALIZED_MISSING)
                     for rendering in KIND_RENDERINGS[MACRO])
    return tuple(built)


def planned_keys(plan: StudyPlan, arm: str, stage: str) -> tuple[tuple[str, str], ...]:
    """``(item_id, rendering)`` of every ensemble the frozen plan sizes for ``arm`` and ``stage``, in plan order."""
    context = _stage(plan, arm, stage)
    if stage == "negative_control":
        return tuple((row_id, OUTLOOK) for row_id in plan.edgar.negative_control)
    months = plan.months.pilot if stage == "pilot" else full_run_months(plan, arm)
    renderings = plan.pilot.renderings if stage == "pilot" else KIND_RENDERINGS[context.kind]
    return tuple((macro_item_id(month), rendering) for month in months for rendering in renderings)


# ---------- the stage ----------

@dataclass(frozen=True)
class _Stage:
    """The frozen settings one arm and stage executes with."""

    plan: StudyPlan
    arm: str
    stage: str
    arm_plan: ArmPlan
    kind: str
    draws: int
    min_parsed: int

    @property
    def definition(self) -> DecisionDefinition:
        return DEFINITIONS[self.kind]

    @property
    def sources(self) -> tuple[str, ...]:
        return self.arm_plan.estimate_sources

    @property
    def full(self) -> bool:
        return FULL_DISTRIBUTION in self.sources

    def config(self, label: SourceLabel, *, single: bool = False) -> GuardConfig:
        """The guard configuration of ``label``: the stage's ensemble, or one draw."""
        if single:
            return GuardConfig(self.definition, label, draws=1, min_parsed=1, mode=DrawMode.SINGLE,
                               temperature=self.arm_plan.temperature, seed_base=self.arm_plan.seed_base)
        return GuardConfig(self.definition, label, draws=self.draws, min_parsed=self.min_parsed,
                           temperature=self.arm_plan.temperature, seed_base=self.arm_plan.seed_base)

    def seed(self, index: int, config: ModelConfig) -> int | None:
        """The seed ``guarded_run`` sends with draw ``index``: ``seed_base + index``, else the configured seed."""
        if self.arm_plan.seed_base is not None:
            return self.arm_plan.seed_base + index
        configured = config.parameters.get("seed")
        return configured if _is_int(configured) else None


def _stage(plan: StudyPlan, arm: str, stage: str) -> _Stage:
    require_complete(plan)
    if arm not in ARMS:
        raise ExecutionError(f"arm must be one of {list(ARMS)}; got {arm!r}")
    if stage not in EXECUTION_STAGES:
        raise ExecutionError(f"stage {stage!r} executes no study items; the guarded execution runs the stages "
                             f"{list(EXECUTION_STAGES)}")
    arm_plan = plan.arms.arms[arm]
    if stage == "pilot":
        if arm != plan.pilot.arm:
            raise ExecutionError(f"the pilot runs on the gated {plan.pilot.arm} arm only; the {arm} arm has no pilot")
        draws, min_parsed = plan.pilot.draws, plan.pilot.min_parsed
    else:
        draws, min_parsed = arm_plan.draws, arm_plan.min_parsed
    return _Stage(plan, arm, stage, arm_plan, EXECUTION_STAGES[stage], draws, min_parsed)


def _check_items(context: _Stage, items: Iterable[ExecutionItem]) -> tuple[ExecutionItem, ...]:
    if isinstance(items, str | bytes) or not isinstance(items, Iterable):
        raise TypeError("items must be a sequence of ExecutionItem values")
    items = tuple(items)
    if not items:
        raise ExecutionError("no study items to execute")
    planned = set(planned_keys(context.plan, context.arm, context.stage))
    seen: set[tuple[str, str]] = set()
    for item in items:
        if not isinstance(item, ExecutionItem):
            raise TypeError(f"items must be ExecutionItem values; got {type(item).__name__}")
        if item.kind != context.kind:
            raise ExecutionError(f"the {context.stage} stage executes {context.kind} items; {item.key} is a "
                                 f"{item.kind} item")
        if item.key not in planned:
            raise ExecutionError(f"{item.key} is not an item and rendering the plan sizes for the "
                                 f"{context.arm}/{context.stage} stage")
        if item.key in seen:
            raise ExecutionError(f"{item.key} is given more than once; one ensemble per item and rendering")
        seen.add(item.key)
    return items


def _check_request_path(context: _Stage, client: CompletionClient, ledger: BudgetLedger) -> None:
    """Every request is admitted by the stage's ledger with the client's real allowance and the frozen settings."""
    if not isinstance(ledger, BudgetLedger):
        raise TypeError(f"ledger must be a BudgetLedger; got {type(ledger).__name__}")
    ensure_unchanged(context.plan, ledger.limits.plan_sha256)
    if (ledger.limits.arm, ledger.limits.stage) != (context.arm, context.stage):
        raise ExecutionError(f"the ledger belongs to {ledger.limits.arm}/{ledger.limits.stage}, not "
                             f"{context.arm}/{context.stage}")
    budget = getattr(client, "budget", None)
    if not isinstance(budget, LedgerCallBudget) or budget.ledger is not ledger:
        raise ExecutionError("every study request must be admitted by the stage's ledger: build the client with "
                             "LedgerCallBudget(ledger) for this ledger; nothing was sent")
    config = getattr(client, "config", None)
    if not isinstance(config, ModelConfig):
        raise TypeError("client.config must be a ModelConfig")
    expected = request_allowance(config, usd_per_second=ledger.allowance.usd_per_second)
    if ledger.allowance != expected:
        raise ExecutionError(f"the ledger's request allowance {ledger.allowance.to_dict()} is not the client's real "
                             f"max_tokens and timeout {expected.to_dict()}; nothing was sent")
    arm, parameters = context.arm_plan, config.parameters
    problems = []
    if config.model != arm.served_model:
        problems.append(f"model {config.model!r}, the arm serves {arm.served_model!r}")
    if parameters.get("max_tokens") != arm.max_tokens:
        problems.append(f"max_tokens {parameters.get('max_tokens')!r}, the arm caps {arm.max_tokens}")
    if parameters.get("logprobs") is not True or parameters.get("top_logprobs") != arm.top_logprobs:
        problems.append(f"logprobs {parameters.get('logprobs')!r} and top_logprobs "
                        f"{parameters.get('top_logprobs')!r}, the arm needs true and {arm.top_logprobs}")
    if problems:
        raise ExecutionError(f"the client's settings differ from the frozen {context.arm} arm: {'; '.join(problems)}; "
                             "nothing was sent")


# ---------- calibrators ----------

def load_calibrators(plan: StudyPlan, *, arm: str, kind: str,
                     calibrators_dir: str | Path = CALIBRATORS_DIR) -> dict[str, CalibrationRecord]:
    """The stored calibrator of each estimate source of ``arm`` for ``kind`` items, refusing any other.

    Each file is loaded with :func:`~scripts.finance_sim.calibrate.load_calibrator`
    for the plan's hash, the arm, the kind and the source, so a calibrator of
    another plan, arm, kind or source is refused before any request.
    """
    require_complete(plan)
    if arm not in ARMS:
        raise ExecutionError(f"arm must be one of {list(ARMS)}; got {arm!r}")
    return {source: load_calibrator(calibrator_path(arm, source, kind, calibrators_dir), plan, arm=arm, kind=kind,
                                    source=source)
            for source in plan.arms.arms[arm].estimate_sources}


def _check_calibrators(context: _Stage, calibrators: Mapping[str, CalibrationRecord]) -> dict[str, CalibrationRecord]:
    if not isinstance(calibrators, Mapping):
        raise TypeError("calibrators must map each estimate source to its CalibrationRecord")
    if set(calibrators) != set(context.sources):
        raise ExecutionError(f"the {context.arm} arm needs one calibrator for each of its estimate sources "
                             f"{list(context.sources)}; got {sorted(map(str, calibrators))}")
    for source, record in calibrators.items():
        if not isinstance(record, CalibrationRecord):
            raise TypeError(f"calibrators[{source!r}] must be a CalibrationRecord; got {type(record).__name__}")
        ensure_unchanged(context.plan, record.plan_sha256)
        if record.kind != context.kind:
            raise ExecutionError(f"the {record.kind} calibrator is refused for {context.kind} items: the macro "
                                 "calibrator is never applied to EDGAR prompts, nor the EDGAR calibrator to macro "
                                 "prompts")
        if record.arm != context.arm:
            raise ExecutionError(f"the {source} calibrator was fitted for the {record.arm} arm, not the "
                                 f"{context.arm} arm")
        if record.source.value != source:
            raise ExecutionError(f"calibrators[{source!r}] was fitted for the {record.source.value} source")
    return {source: calibrators[source] for source in context.sources}


def _evidence_settings(context: _Stage, records: Mapping[str, CalibrationRecord],
                       evidence: EvidenceSettings | None) -> EvidenceSettings | None:
    """The evidence pass settings, by default the frozen plan's; ``top_k_keep`` must stay the calibrator's.

    The timeout, the busy retries and their wait are the arm's frozen
    ``ArmPlan.evidence``; the alternatives kept per position are those the
    full-distribution calibrator was fitted on (the plan's, when it was fitted
    with the plan's settings), because its estimates are defined on them.
    """
    if not context.full:
        return None
    fitted = records[FULL_DISTRIBUTION].evidence_settings
    if evidence is None:
        frozen = context.arm_plan.evidence
        evidence = EvidenceSettings(timeout_s=frozen.timeout_s, top_k_keep=fitted.top_k_keep,
                                    host_busy_retries=frozen.host_busy_retries,
                                    host_busy_wait_s=frozen.host_busy_wait_s)
    if not isinstance(evidence, EvidenceSettings):
        raise TypeError("evidence must be EvidenceSettings or None")
    if evidence.top_k_keep != fitted.top_k_keep:
        raise ExecutionError(f"the evidence pass must keep the {fitted.top_k_keep} alternatives the full-distribution "
                             f"calibrator was fitted on; got top_k_keep {evidence.top_k_keep}")
    return evidence


def _labels(context: _Stage, evidence: EvidenceSettings | None) -> dict[str, SourceLabel]:
    """The source label of each estimate source: top-k with the arm's top_logprobs, full with the kept count."""
    labels = {TOP_K: SourceLabel(EstimateSource.TOP_K, context.arm_plan.top_logprobs)}
    if context.full:
        labels[FULL_DISTRIBUTION] = SourceLabel(EstimateSource.FULL_DISTRIBUTION, evidence.top_k_keep)
    return labels


def _identities(context: _Stage, item: ExecutionItem, client: CompletionClient,
                records: Mapping[str, CalibrationRecord], labels: Mapping[str, SourceLabel]) -> dict[str, GuardIdentity]:
    """Per source, the guard's identity of the item's ensemble; a calibrator that differs from the client is refused.

    This is the guarded-decision API's own check before any request
    (``CalibrationMismatchError`` naming the differing fields).
    """
    return {source: _guard_identity(item.prompt, context.config(labels[source]), client, records[source].calibrator)
            for source in context.sources}


# ---------- draws ----------

@dataclass(frozen=True)
class SourceEstimate:
    """One draw's estimate from one source, its evidence reference and the calibrator it came from."""

    estimate: float | None
    evidence_ref: str | None
    calibrator_fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "estimate", _finite(self.estimate, "estimate"))
        if self.evidence_ref is not None:
            _hex(self.evidence_ref, "evidence_ref")
        _text(self.calibrator_fingerprint, "calibrator_fingerprint")

    def to_dict(self) -> dict[str, Any]:
        return {"estimate": self.estimate, "evidence_ref": self.evidence_ref,
                "calibrator_fingerprint": self.calibrator_fingerprint}

    @classmethod
    def from_dict(cls, data: object) -> SourceEstimate:
        return cls(**_exact(data, _ESTIMATE_KEYS, "source estimate"))


@dataclass(frozen=True)
class ExecutionDraw:
    """One recorded study draw: the guard's record, the reply and the full-distribution estimate once settled.

    ``draw`` is the :class:`~recall_guard.core.distribution.RecordedDraw` the
    guarded-decision API recorded with the TOP_K source (its estimate is the
    top-k estimate of the calibrator ``calibrator_fingerprint``). ``response``
    is the reply with its top-k evidence, None exactly when the request
    returned none; it is private. ``full_distribution`` is None until the
    evidence pass settles a parsed draw of an arm with that source, and stays
    None for every other draw.
    """

    item_id: str
    rendering: str
    prompt_sha256: str
    draw: RecordedDraw
    response: ModelResponse | None
    calibrator_fingerprint: str
    full_distribution: SourceEstimate | None = None

    def __post_init__(self) -> None:
        _text(self.item_id, "item_id")
        if self.rendering not in (IDENTIFYING, ANONYMISED, OUTLOOK):
            raise ExecutionError(f"rendering must be identifying, anonymised or outlook; got {self.rendering!r}")
        _hex(self.prompt_sha256, "prompt_sha256")
        if not isinstance(self.draw, RecordedDraw):
            raise TypeError(f"draw must be a RecordedDraw; got {type(self.draw).__name__}")
        failed = self.draw.failure_code is not None and self.draw.failure_code.startswith(REQUEST_ERROR_PREFIX)
        if self.response is None:
            if not failed:
                raise ExecutionError("only a failed request has no reply")
        else:
            if not isinstance(self.response, ModelResponse):
                raise TypeError(f"response must be a ModelResponse or None; got {type(self.response).__name__}")
            if failed or _sha256(self.response.content) != self.draw.response_sha256:
                raise ExecutionError("a reply must be the one whose SHA-256 the draw records")
        _text(self.calibrator_fingerprint, "calibrator_fingerprint")
        if self.full_distribution is not None:
            if not isinstance(self.full_distribution, SourceEstimate):
                raise TypeError("full_distribution must be a SourceEstimate or None")
            if self.draw.parsed is None:
                raise ExecutionError("only a parsed draw carries a full-distribution estimate")

    @classmethod
    def interrupted(cls, item: ExecutionItem, draw_index: int, *, seed: int | None, temperature: float | None,
                    calibrator_fingerprint: str) -> ExecutionDraw:
        """The failed draw a recorder passes back for a request that was sent but whose outcome was lost.

        ``seed`` is the one ``on_request`` reported (the deterministic seed of
        the draw), ``temperature`` the arm's frozen temperature and
        ``calibrator_fingerprint`` the arm's top-k calibrator. The draw carries
        no reply (``response`` None), the SHA-256 of the empty string and the
        failure code ``request_error:interrupted``; it stays in its ensemble as
        a failed draw and is never requested again.
        """
        if not isinstance(item, ExecutionItem):
            raise TypeError(f"item must be an ExecutionItem; got {type(item).__name__}")
        if not _is_int(draw_index) or draw_index < 0:
            raise ExecutionError(f"draw_index must be a non-negative integer; got {draw_index!r}")
        draw = RecordedDraw(draw_index=draw_index, temperature=_finite(temperature, "temperature"), seed=seed,
                            response_sha256=_sha256(""), parsed=None, failure_code=INTERRUPTED, estimate=None,
                            evidence_ref=None)
        return cls(item.item_id, item.rendering, item.prompt_sha256, draw, None, calibrator_fingerprint)

    @property
    def key(self) -> tuple[str, str, int]:
        """``(item_id, rendering, draw_index)``: one draw of one ensemble."""
        return self.item_id, self.rendering, self.draw.draw_index

    @property
    def awaiting_evidence(self) -> bool:
        """A parsed draw without a full-distribution estimate (pending on an arm with that source)."""
        return self.draw.parsed is not None and self.full_distribution is None

    def settled(self, estimate: SourceEstimate) -> ExecutionDraw:
        """This draw with its full-distribution estimate."""
        if not self.awaiting_evidence:
            raise ExecutionError(f"draw {self.key} has no pending full-distribution estimate")
        return replace(self, full_distribution=estimate)

    def to_dict(self) -> dict[str, Any]:
        return {"item_id": self.item_id, "rendering": self.rendering, "prompt_sha256": self.prompt_sha256,
                "draw": self.draw.to_dict(), "response": None if self.response is None else self.response.to_dict(),
                "calibrator_fingerprint": self.calibrator_fingerprint,
                "full_distribution": None if self.full_distribution is None else self.full_distribution.to_dict()}

    @classmethod
    def from_dict(cls, data: object) -> ExecutionDraw:
        values = _exact(data, _DRAW_KEYS, "execution draw")
        try:
            draw = RecordedDraw.from_dict(values["draw"])
            response = None if values["response"] is None else ModelResponse.from_dict(values["response"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ExecutionError(f"execution draw cannot be read: {exc}") from None
        full = values["full_distribution"]
        return cls(item_id=values["item_id"], rendering=values["rendering"], prompt_sha256=values["prompt_sha256"],
                   draw=draw, response=response, calibrator_fingerprint=values["calibrator_fingerprint"],
                   full_distribution=None if full is None else SourceEstimate.from_dict(full))


def _reusable(context: _Stage, items: Sequence[ExecutionItem], recorded: Iterable[ExecutionDraw],
              identities: Mapping[tuple[str, str], Mapping[str, GuardIdentity]],
              records: Mapping[str, CalibrationRecord], config: ModelConfig) -> dict[tuple[str, str, int], ExecutionDraw]:
    """Recorded draws of these items and settings by key; any other recorded draw is refused, never merged."""
    if isinstance(recorded, str | bytes) or not isinstance(recorded, Iterable):
        raise TypeError("recorded must be a sequence of ExecutionDraw values")
    by_key = {item.key: item for item in items}
    top_k = records[TOP_K].calibrator.fingerprint
    full = records[FULL_DISTRIBUTION].calibrator.fingerprint if context.full else None
    reused: dict[tuple[str, str, int], ExecutionDraw] = {}
    for draw in recorded:
        if not isinstance(draw, ExecutionDraw):
            raise TypeError(f"recorded draws must be ExecutionDraw values; got {type(draw).__name__}")
        item = by_key.get((draw.item_id, draw.rendering))
        index = draw.draw.draw_index
        problems = []
        if item is None:
            problems.append("it is not an item and rendering of this run")
        else:
            if draw.prompt_sha256 != item.prompt_sha256:
                problems.append("its prompt differs from the item's")
            if index >= context.draws:
                problems.append(f"its draw_index is outside the {context.draws} draws of the {context.stage} stage")
            expected = (context.seed(index, config), identities[item.key][TOP_K].temperature)
            if (draw.draw.seed, draw.draw.temperature) != expected:
                problems.append("its seed or temperature differs from the frozen arm settings")
            if draw.calibrator_fingerprint != top_k:
                problems.append("its top_k estimate comes from another calibrator")
            if draw.full_distribution is not None:
                if full is None:
                    problems.append(f"it carries a full_distribution estimate, which the {context.arm} arm does not "
                                    "compute")
                elif draw.full_distribution.calibrator_fingerprint != full:
                    problems.append("its full_distribution estimate comes from another calibrator")
        earlier = reused.get(draw.key)
        if earlier is not None and earlier != draw:
            if earlier.full_distribution is None and replace(draw, full_distribution=None) == earlier:
                pass  # the settled version of a pending draw replaces it
            elif draw.full_distribution is None and replace(earlier, full_distribution=None) == draw:
                continue  # the pending version of a settled draw adds nothing
            else:
                problems.append("it is recorded more than once with different content")
        if problems:
            raise ExecutionError(f"recorded draw {draw.key} is refused, never merged: {'; '.join(problems)}")
        reused[draw.key] = draw
    return reused


def _generate(context: _Stage, items: Sequence[ExecutionItem], client: CompletionClient, ledger: BudgetLedger,
              records: Mapping[str, CalibrationRecord], labels: Mapping[str, SourceLabel],
              identities: Mapping[tuple[str, str], Mapping[str, GuardIdentity]],
              reused: Mapping[tuple[str, str, int], ExecutionDraw],
              on_draw: Callable[[ExecutionDraw], None] | None,
              on_request: Callable[[ExecutionItem, int, int | None], None] | None) -> list[ExecutionDraw]:
    """Complete one ensemble per item and rendering, issuing only the draws not recorded."""
    calibrator = records[TOP_K].calibrator
    config = context.config(labels[TOP_K])
    draws: list[ExecutionDraw] = []
    for item in items:
        prefilled = {index: reused[(item.item_id, item.rendering, index)] for index in range(context.draws)
                     if (item.item_id, item.rendering, index) in reused}
        if len(prefilled) < context.draws and ledger.status != OPEN:
            ledger.admit()  # A closed stage raises AdmissionClosedError here; the client is not called.
        made: dict[int, ExecutionDraw] = {}

        def record(draw: RecordedDraw, response: ModelResponse | None, _full: FullDistributionEvidence | None,
                   item: ExecutionItem = item, made: dict[int, ExecutionDraw] = made) -> None:
            entry = ExecutionDraw(item.item_id, item.rendering, item.prompt_sha256, draw, response,
                                  calibrator.fingerprint)
            made[draw.draw_index] = entry
            if on_draw is not None:
                on_draw(entry)

        def announce(index: int, seed: int | None, item: ExecutionItem = item) -> None:
            on_request(item, index, seed)

        run = guarded_run(item.prompt, config, client=client, calibrator=calibrator,
                          recorded=[entry.draw for _, entry in sorted(prefilled.items())],
                          on_request=None if on_request is None else announce, on_draw=record)
        if run.decision.identity != identities[item.key][TOP_K]:
            raise ExecutionError(f"the guarded run of {item.key} has another identity than the one checked before "
                                 "any request")
        ensemble = [prefilled[index] if index in prefilled else made[index] for index in range(context.draws)]
        if tuple(entry.draw for entry in ensemble) != run.decision.draws:
            raise ExecutionError(f"the guarded run of {item.key} recorded other draws than it reported")
        draws.extend(ensemble)
    return draws


def _check_callable(value: object, name: str) -> None:
    if value is not None and not callable(value):
        raise TypeError(f"{name} must be callable or None")


def generate_draws(
    items: Iterable[ExecutionItem], *, plan: StudyPlan, arm: str, stage: str, client: CompletionClient,
    ledger: BudgetLedger, calibrators: Mapping[str, CalibrationRecord], recorded: Iterable[ExecutionDraw] = (),
    on_draw: Callable[[ExecutionDraw], None] | None = None,
    on_request: Callable[[ExecutionItem, int, int | None], None] | None = None,
) -> tuple[ExecutionDraw, ...]:
    """The generation pass alone: one recorded ensemble per item and rendering, in item and draw order.

    Call it inside ``with ledger:`` with the generation host loaded, so that a
    frozen limit closes the stage as incomplete with the recorded draws kept.
    Before any request the stage, the items, the ledger, the client and the
    calibrators are checked, and ``recorded`` draws are validated; a recorded
    draw is never requested again. ``on_request`` and ``on_draw`` follow the
    recording contract of :func:`execute_items`. No full-distribution evidence
    is requested here: an operator stops the generation host, loads the
    full-distribution host and runs :func:`execute_items` with the draws
    recorded here.
    """
    context = _stage(plan, arm, stage)
    items = _check_items(context, items)
    _check_request_path(context, client, ledger)
    records = _check_calibrators(context, calibrators)
    _check_callable(on_draw, "on_draw")
    _check_callable(on_request, "on_request")
    labels = _labels(context, _evidence_settings(context, records, None))
    identities = {item.key: _identities(context, item, client, records, labels) for item in items}
    reused = _reusable(context, items, recorded, identities, records, client.config)
    return tuple(_generate(context, items, client, ledger, records, labels, identities, reused, on_draw,
                           on_request))


# ---------- evidence ----------

def _evidence_batch(full_distribution: FullDistributionClient, item: ExecutionItem, texts: list[str],
                    evidence: EvidenceSettings, sleep: Callable[[float], None]) -> tuple[FullDistributionEvidence, ...]:
    """One batch for one item and rendering; ``host_busy`` is retried after a wait, anything else raised."""
    where = f"{item.item_id} {item.rendering}"
    for attempt in range(evidence.host_busy_retries + 1):
        try:
            batch = full_distribution.evidence_batch(item.prompt, texts, top_k_keep=evidence.top_k_keep,
                                                     timeout_s=evidence.timeout_s)
        except RequestError as error:
            if error.code != HOST_BUSY:
                raise EvidenceUnavailableError(
                    f"the full-distribution host gave no evidence for {where} ({error.code or 'request_error'}); the "
                    "stage stays open with its recorded draws, and a rerun sends only the missing batches",
                    code=error.code, prompt_sha256=item.prompt_sha256) from error
            if attempt == evidence.host_busy_retries:
                raise EvidenceUnavailableError(
                    f"the full-distribution host stayed busy (host_busy) for {where} after "
                    f"{evidence.host_busy_retries} retries {evidence.host_busy_wait_s:g} s apart; the stage stays open "
                    "with its recorded draws", code=HOST_BUSY, prompt_sha256=item.prompt_sha256) from error
            sleep(evidence.host_busy_wait_s)
            continue
        except TimeoutError as error:
            raise EvidenceUnavailableError(
                f"the evidence batch for {where} timed out; it is not resent in this run because the host may still be "
                "computing it, and the stage stays open with its recorded draws", code="timeout",
                prompt_sha256=item.prompt_sha256) from error
        if not isinstance(batch, tuple | list) or len(batch) != len(texts) or not all(
                isinstance(entry, FullDistributionEvidence) for entry in batch):
            raise EvidenceUnavailableError(f"the evidence batch for {where} does not hold one evidence object per "
                                           "reply", code="invalid_evidence", prompt_sha256=item.prompt_sha256)
        return tuple(batch)
    raise AssertionError("unreachable")  # pragma: no cover - the loop returns or raises


def full_distribution_estimate(response_text: str, evidence: FullDistributionEvidence,
                               record: CalibrationRecord) -> SourceEstimate:
    """One draw's full-distribution estimate, as the guarded-decision API computes it per draw.

    The features of ``features_from_full`` through the calibrator; evidence the
    features cannot use, or a non-finite estimate, leaves the estimate None. The
    reference is the SHA-256 of the evidence payload.
    """
    if not isinstance(record, CalibrationRecord) or record.source is not EstimateSource.FULL_DISTRIBUTION:
        raise ExecutionError("a full-distribution estimate needs the full_distribution calibrator")
    reference = _payload_sha256(evidence.to_dict())
    fingerprint = record.calibrator.fingerprint
    try:
        value = record.calibrator.estimate(features_from_full(response_text, evidence).features)
    except (ValueError, RuntimeError):
        return SourceEstimate(None, reference, fingerprint)
    return SourceEstimate(value if _is_real(value) and math.isfinite(value) else None, reference, fingerprint)


def _settle(item: ExecutionItem, draws: list[ExecutionDraw], *, full_distribution: FullDistributionClient,
            evidence: EvidenceSettings, record: CalibrationRecord,
            on_draw: Callable[[ExecutionDraw], None] | None,
            on_evidence: Callable[[ExecutionItem, Mapping[int, FullDistributionEvidence]], None] | None,
            sleep: Callable[[float], None]) -> list[ExecutionDraw]:
    """Settle the item's pending draws with one evidence batch over their replies in draw order."""
    pending = [at for at, draw in enumerate(draws) if draw.awaiting_evidence]
    if not pending:
        return draws
    texts = [draws[at].response.content for at in pending]
    batch = _evidence_batch(full_distribution, item, texts, evidence, sleep)
    if on_evidence is not None:
        on_evidence(item, MappingProxyType({draws[at].draw.draw_index: entry
                                            for at, entry in zip(pending, batch, strict=True)}))
    for at, entry in zip(pending, batch, strict=True):
        draws[at] = draws[at].settled(full_distribution_estimate(draws[at].response.content, entry, record))
        if on_draw is not None:
            on_draw(draws[at])
    return draws


# ---------- variants ----------

@dataclass(frozen=True)
class VariantResult:
    """One variant of one item and rendering, or the reason the arm cannot compute it (``missing``).

    ``decision`` and ``adjusted`` are the guarded decision's vectors,
    ``summary_value`` and ``adjusted_summary_value`` their summary values (None
    when a component is unresolved), ``category`` the decision's category.
    ``correct`` scores the variant's adjusted decision, or the raw decision's
    sign when the adjustment shrinks a non-zero spread to one that rounds to zero
    (``sign_from_raw_spread`` with ``sign_reason``); ``scored_probability``
    is the plan's clipped raw probability and ``adjusted_scored_probability``
    the plan's adjustment of it. ``missing_reasons`` names each value that
    could not be computed and why.
    """

    variant: str
    mode: str
    adjustment: str
    missing: str | None = None
    status: str | None = None
    reason: str | None = None
    decision: Mapping[str, float | None] | None = None
    adjusted: Mapping[str, float | None] | None = None
    p_memorized: float | None = None
    summary_value: float | None = None
    adjusted_summary_value: float | None = None
    category: Hashable | None = None
    correct: bool | None = None
    scored_probability: float | None = None
    adjusted_scored_probability: float | None = None
    first_draw_rank: float | None = None
    not_assessed: tuple[str, ...] = ()
    draws_sha256: str | None = None
    missing_reasons: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    sign_from_raw_spread: bool = False
    sign_reason: str | None = None

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS or (self.mode, self.adjustment) != variant_parts(self.variant):
            raise ExecutionError(f"variant {self.variant!r} must be one of {list(VARIANTS)} with its draw mode and "
                                 "adjustment")
        if self.missing is not None:
            _text(self.missing, "missing")
            if (self.status is not None or self.decision is not None or self.missing_reasons
                    or self.sign_from_raw_spread or self.sign_reason is not None):
                raise ExecutionError("a missing variant carries only its reason")
            return
        if self.status not in STATUSES:
            raise ExecutionError(f"status must be one of {list(STATUSES)}; got {self.status!r}")
        if self.reason is not None:
            _text(self.reason, "reason")
        object.__setattr__(self, "decision", _vector(self.decision, "decision"))
        object.__setattr__(self, "adjusted", _vector(self.adjusted, "adjusted"))
        for name in ("p_memorized", "summary_value", "adjusted_summary_value", "scored_probability",
                     "adjusted_scored_probability", "first_draw_rank"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        object.__setattr__(self, "category", _scalar(self.category, "category"))
        if self.correct is not None and not isinstance(self.correct, bool):
            raise ExecutionError(f"correct must be a bool or None; got {self.correct!r}")
        if isinstance(self.not_assessed, str) or not all(isinstance(name, str) for name in self.not_assessed):
            raise ExecutionError("not_assessed must list component names")
        object.__setattr__(self, "not_assessed", tuple(self.not_assessed))
        if self.draws_sha256 is not None:
            _text(self.draws_sha256, "draws_sha256")
        reasons = self.missing_reasons
        if not isinstance(reasons, Mapping) or not all(isinstance(key, str) for key in reasons):
            raise ExecutionError("missing_reasons must map value names to reasons")
        object.__setattr__(self, "missing_reasons", MappingProxyType({key: _text(value, f"missing_reasons[{key!r}]")
                                                                      for key, value in sorted(reasons.items())}))
        if not isinstance(self.sign_from_raw_spread, bool):
            raise ExecutionError(f"sign_from_raw_spread must be a bool; got {self.sign_from_raw_spread!r}")
        if self.sign_from_raw_spread != (self.sign_reason is not None):
            raise ExecutionError("sign_reason is given exactly when correctness is scored on the raw spread's sign")
        if self.sign_reason is not None:
            _text(self.sign_reason, "sign_reason")

    def to_dict(self) -> dict[str, Any]:
        return {"variant": self.variant, "mode": self.mode, "adjustment": self.adjustment, "missing": self.missing,
                "status": self.status, "reason": self.reason,
                "decision": None if self.decision is None else dict(self.decision),
                "adjusted": None if self.adjusted is None else dict(self.adjusted), "p_memorized": self.p_memorized,
                "summary_value": self.summary_value, "adjusted_summary_value": self.adjusted_summary_value,
                "category": self.category, "correct": self.correct, "scored_probability": self.scored_probability,
                "adjusted_scored_probability": self.adjusted_scored_probability,
                "first_draw_rank": self.first_draw_rank, "not_assessed": list(self.not_assessed),
                "draws_sha256": self.draws_sha256, "missing_reasons": dict(self.missing_reasons),
                "sign_from_raw_spread": self.sign_from_raw_spread, "sign_reason": self.sign_reason}

    @classmethod
    def from_dict(cls, data: object) -> VariantResult:
        values = dict(_exact(data, _VARIANT_KEYS, "variant result"))
        if not isinstance(values["not_assessed"], list):
            raise ExecutionError("not_assessed must be a list")
        values["not_assessed"] = tuple(values["not_assessed"])
        return cls(**values)


def _identity_from_dict(data: object) -> GuardIdentity:
    """``GuardIdentity.from_dict`` (7.4), with a refusal reported as an :class:`ExecutionError`."""
    if not isinstance(data, Mapping):
        raise ExecutionError("a guard identity must be a JSON object")
    try:
        return GuardIdentity.from_dict(data)
    except (TypeError, ValueError) as exc:
        raise ExecutionError(f"guard identity cannot be read: {exc}") from None


@dataclass(frozen=True)
class ItemResult:
    """Every variant of one item and rendering, with the draws and identities they were derived from.

    ``recorded_draws`` are the draws as the guarded-decision API recorded them
    with the TOP_K source; ``full_distribution`` holds each draw's
    full-distribution estimate by draw index (None for an unparsed draw) on an
    arm with that source, and is None otherwise. ``summary_values`` and
    ``categories`` are per draw (None for an unparsed draw), ``outcome`` the
    scored probability's outcome (macro items with realized returns only).
    """

    plan_sha256: str
    arm: str
    stage: str
    item_id: str
    kind: str
    rendering: str
    prompt_sha256: str
    ensemble_draws: int
    min_parsed: int
    identities: Mapping[str, GuardIdentity]
    recorded_draws: tuple[RecordedDraw, ...]
    full_distribution: tuple[SourceEstimate | None, ...] | None
    summary_values: tuple[float | None, ...]
    categories: tuple[Hashable | None, ...]
    outcome: int | None
    variants: tuple[VariantResult, ...]

    def __post_init__(self) -> None:
        _hex(self.plan_sha256, "plan_sha256")
        if self.arm not in ARMS or self.stage not in EXECUTION_STAGES:
            raise ExecutionError(f"unknown arm or stage {self.arm!r}/{self.stage!r}")
        if self.kind != EXECUTION_STAGES[self.stage]:
            raise ExecutionError(f"the {self.stage} stage executes {EXECUTION_STAGES[self.stage]} items")
        if not isinstance(self.item_id, str) or not _ITEM_ID.fullmatch(self.item_id):
            raise ExecutionError(f"item_id must be letters, digits, dots, dashes or underscores; got {self.item_id!r}")
        if self.rendering not in KIND_RENDERINGS[self.kind]:
            raise ExecutionError(f"a {self.kind} item has the renderings {list(KIND_RENDERINGS[self.kind])}")
        _hex(self.prompt_sha256, "prompt_sha256")
        if not (_is_int(self.ensemble_draws) and _is_int(self.min_parsed)
                and 1 <= self.min_parsed <= self.ensemble_draws):
            raise ExecutionError("ensemble_draws and min_parsed must satisfy 1 <= min_parsed <= ensemble_draws")
        if not isinstance(self.identities, Mapping) or TOP_K not in self.identities or not set(
                self.identities) <= set(_SOURCES) or not all(isinstance(identity, GuardIdentity)
                                                             for identity in self.identities.values()):
            raise ExecutionError("identities must map top_k, and full_distribution when computed, to GuardIdentity")
        object.__setattr__(self, "identities", MappingProxyType({source: self.identities[source]
                                                                 for source in _SOURCES if source in self.identities}))
        draws = tuple(self.recorded_draws)
        if [draw.draw_index for draw in draws] != list(range(self.ensemble_draws)) or not all(
                isinstance(draw, RecordedDraw) for draw in draws):
            raise ExecutionError("recorded_draws must hold one RecordedDraw per draw index, in order")
        object.__setattr__(self, "recorded_draws", draws)
        if (self.full_distribution is None) == (FULL_DISTRIBUTION in self.identities):
            raise ExecutionError("full_distribution estimates are kept exactly when that source is computed")
        if self.full_distribution is not None:
            full = tuple(self.full_distribution)
            if len(full) != len(draws) or not all(entry is None or isinstance(entry, SourceEstimate) for entry in full):
                raise ExecutionError("full_distribution must hold one SourceEstimate or None per draw")
            object.__setattr__(self, "full_distribution", full)
        for name in ("summary_values", "categories"):
            values = tuple(getattr(self, name))
            if len(values) != len(draws):
                raise ExecutionError(f"{name} must hold one value per draw")
            check = _finite if name == "summary_values" else _scalar
            object.__setattr__(self, name, tuple(check(value, name) for value in values))
        if self.outcome is not None and self.outcome not in (0, 1):
            raise ExecutionError(f"outcome must be 0, 1 or None; got {self.outcome!r}")
        variants = tuple(self.variants)
        if [variant.variant for variant in variants] != list(VARIANTS) or not all(
                isinstance(variant, VariantResult) for variant in variants):
            raise ExecutionError(f"variants must hold the six variants {list(VARIANTS)} in order")
        object.__setattr__(self, "variants", variants)

    @property
    def key(self) -> tuple[str, str]:
        return self.item_id, self.rendering

    def variant(self, name: str) -> VariantResult:
        """The result of one variant by name."""
        for variant in self.variants:
            if variant.variant == name:
                return variant
        raise KeyError(name)

    def draws_for(self, adjustment: str) -> tuple[RecordedDraw, ...]:
        """The recorded draws as one adjustment reduces them: its source's estimates, or none for no adjustment."""
        return _variant_draws(self.recorded_draws, self.full_distribution, adjustment)

    def to_dict(self) -> dict[str, Any]:
        return {"schema": RESULT_SCHEMA, "plan_sha256": self.plan_sha256, "arm": self.arm, "stage": self.stage,
                "item_id": self.item_id, "kind": self.kind, "rendering": self.rendering,
                "prompt_sha256": self.prompt_sha256, "ensemble_draws": self.ensemble_draws,
                "min_parsed": self.min_parsed,
                "identities": {source: identity.to_dict() for source, identity in self.identities.items()},
                "recorded_draws": [draw.to_dict() for draw in self.recorded_draws],
                "full_distribution": None if self.full_distribution is None else [
                    None if entry is None else entry.to_dict() for entry in self.full_distribution],
                "summary_values": list(self.summary_values), "categories": list(self.categories),
                "outcome": self.outcome, "variants": [variant.to_dict() for variant in self.variants]}

    @classmethod
    def from_dict(cls, data: object) -> ItemResult:
        values = dict(_exact(data, _RESULT_KEYS, "item result"))
        if values.pop("schema") != RESULT_SCHEMA:
            raise ExecutionError(f"unsupported result schema; expected {RESULT_SCHEMA!r}")
        identities = values["identities"]
        if not isinstance(identities, Mapping):
            raise ExecutionError("identities must be a JSON object")
        for name in ("recorded_draws", "summary_values", "categories", "variants"):
            if not isinstance(values[name], list):
                raise ExecutionError(f"{name} must be a list")
        try:
            draws = tuple(RecordedDraw.from_dict(draw) for draw in values["recorded_draws"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ExecutionError(f"recorded draws cannot be read: {exc}") from None
        full = values["full_distribution"]
        if full is not None and not isinstance(full, list):
            raise ExecutionError("full_distribution must be a list or None")
        values.update(
            identities={source: _identity_from_dict(identity) for source, identity in identities.items()},
            recorded_draws=draws,
            full_distribution=None if full is None else tuple(
                None if entry is None else SourceEstimate.from_dict(entry) for entry in full),
            summary_values=tuple(values["summary_values"]), categories=tuple(values["categories"]),
            variants=tuple(VariantResult.from_dict(variant) for variant in values["variants"]))
        return cls(**values)


def _variant_draws(draws: Sequence[RecordedDraw], full: Sequence[SourceEstimate | None] | None,
                   adjustment: str) -> tuple[RecordedDraw, ...]:
    if adjustment == TOP_K:
        return tuple(draws)
    if adjustment == NO_ADJUSTMENT:
        return tuple(replace(draw, estimate=None, evidence_ref=None) for draw in draws)
    if adjustment != FULL_DISTRIBUTION or full is None:
        raise ExecutionError(f"no draws for adjustment {adjustment!r}")
    return tuple(replace(draw, estimate=None if entry is None else entry.estimate,
                         evidence_ref=None if entry is None else entry.evidence_ref)
                 for draw, entry in zip(draws, full, strict=True))


def _summary(definition: DecisionDefinition, vector: Mapping[str, float | None] | None) -> float | None:
    if vector is None or any(value is None for value in vector.values()):
        return None
    return float(definition.summary(dict(vector)))


def _variant(context: _Stage, item: ExecutionItem, variant: str, draws: Sequence[RecordedDraw],
             full: Sequence[SourceEstimate | None] | None, identities: Mapping[str, GuardIdentity]) -> VariantResult:
    mode, adjustment = variant_parts(variant)
    missing = context.arm_plan.missing_variants.get(variant)
    if missing is not None:
        return VariantResult(variant, mode, adjustment, missing=missing)
    identity = identities[FULL_DISTRIBUTION if adjustment == FULL_DISTRIBUTION else TOP_K]
    config = context.config(identity.source)
    selected = _variant_draws(draws, full, adjustment)
    if mode == "single":
        first = selected[SINGLE_DRAW_INDEX]
        decision: GuardedDecision = decide_from_draws(
            (first,), context.config(identity.source, single=True), replace(identity, draws=1))
        scored_draws: Sequence[RecordedDraw] = (first,)
    else:
        decision = decide_from_draws(selected, config, identity)
        scored_draws = selected
    definition = context.definition
    summary = _summary(definition, decision.decision)
    adjusted_summary = _summary(definition, decision.adjusted)
    reasons: dict[str, str] = {}
    correct = probability = adjusted_probability = None
    from_raw = False
    if item.kind != MACRO:
        reasons = {"correct": EDGAR_NO_OUTCOME, "scored_probability": EDGAR_NO_OUTCOME}
    elif decision.status == "failed":
        reasons = dict.fromkeys(("correct", "scored_probability"), f"no decision: {decision.reason}")
    elif item.realized is None:
        reasons = dict.fromkeys(("correct", "scored_probability"), item.realized_missing_reason or REALIZED_MISSING)
    else:
        scored = context.plan.scored_probability
        if summary is None or adjusted_summary is None:
            reasons["correct"] = UNRESOLVED_SUMMARY
        elif round(adjusted_summary, SIGN_DECIMALS) == 0 and round(summary, SIGN_DECIMALS) != 0:
            # Shrunk to a zero spread: the adjustment keeps the sign (1.3), so the raw spread's sign is scored.
            correct, from_raw = is_correct(summary, item.realized), True
        else:
            correct = is_correct(adjusted_summary, item.realized)
        spreads = [definition.summary(dict(draw.parsed)) for draw in scored_draws if draw.parsed is not None]
        probability = scored.probability(spreads, mode=mode)
        adjusted_probability = scored.adjusted(probability, decision.p_memorized if decision.status == "ok" else None)
    distribution = decision.distribution
    rank = None
    if distribution is not None and distribution.summary_value is not None:
        rank = distribution.summary_value.first_draw_rank
    return VariantResult(
        variant, mode, adjustment, status=decision.status, reason=decision.reason, decision=decision.decision,
        adjusted=decision.adjusted, p_memorized=decision.p_memorized, summary_value=summary,
        adjusted_summary_value=adjusted_summary, category=decision.category, correct=correct,
        scored_probability=probability, adjusted_scored_probability=adjusted_probability, first_draw_rank=rank,
        not_assessed=decision.not_assessed, draws_sha256=decision.draws_sha256, missing_reasons=reasons,
        sign_from_raw_spread=from_raw, sign_reason=SIGN_FROM_RAW_REASON if from_raw else None)


def _derive(context: _Stage, item: ExecutionItem, draws: Sequence[RecordedDraw],
            full: Sequence[SourceEstimate | None] | None, identities: Mapping[str, GuardIdentity]) -> ItemResult:
    draws = tuple(sorted(draws, key=lambda draw: draw.draw_index))
    if not all(isinstance(draw, RecordedDraw) for draw in draws):
        raise TypeError("draws must be RecordedDraw values")
    if [draw.draw_index for draw in draws] != list(range(context.draws)):
        raise ExecutionError(f"{item.key} needs exactly one draw per index 0 to {context.draws - 1} of its ensemble")
    if context.full:
        if full is None or len(full) != len(draws):
            raise ExecutionError(f"{item.key} needs one full-distribution estimate (or None) per draw")
        pending = [draw.draw_index for draw, entry in zip(draws, full, strict=True)
                   if draw.parsed is not None and entry is None]
        if pending:
            raise ExecutionError(f"{item.key} draws {pending} await their full-distribution evidence")
        full = tuple(full)
    elif full is not None:
        raise ExecutionError(f"the {context.arm} arm computes no full-distribution estimate")
    if not isinstance(identities, Mapping) or set(identities) != set(context.sources):
        raise ExecutionError(f"identities must name exactly the arm's sources {list(context.sources)}")
    for source, identity in identities.items():
        if not isinstance(identity, GuardIdentity):
            raise TypeError("identities must be GuardIdentity values")
        if (identity.source.source.value, identity.draws, identity.prompt_sha256) != (
                source, context.draws, item.prompt_sha256):
            raise ExecutionError(f"the {source} identity is not that of the {item.key} ensemble")
    definition = context.definition
    outcome = None
    if item.kind == MACRO and item.realized is not None:
        outcome = context.plan.scored_probability.outcome_of(item.realized)
    return ItemResult(
        plan_sha256=context.plan.plan_sha256, arm=context.arm, stage=context.stage, item_id=item.item_id,
        kind=item.kind, rendering=item.rendering, prompt_sha256=item.prompt_sha256, ensemble_draws=context.draws,
        min_parsed=context.min_parsed, identities=identities, recorded_draws=draws, full_distribution=full,
        summary_values=tuple(None if draw.parsed is None else float(definition.summary(dict(draw.parsed)))
                             for draw in draws),
        categories=tuple(None if draw.parsed is None else definition.decide(dict(draw.parsed)) for draw in draws),
        outcome=outcome,
        variants=tuple(_variant(context, item, variant, draws, full, identities) for variant in VARIANTS))


def derive_result(item: ExecutionItem, draws: Sequence[RecordedDraw],
                  full_distribution: Sequence[SourceEstimate | None] | None, *, plan: StudyPlan, arm: str, stage: str,
                  identities: Mapping[str, GuardIdentity]) -> ItemResult:
    """Every variant of one item and rendering from its one ensemble, without a request (Requirement 5.8).

    ``draws`` are the ensemble's draws as the guarded-decision API recorded
    them with the TOP_K source, ``full_distribution`` the settled
    full-distribution estimate of each (None for an unparsed draw) on an arm
    with that source, and ``identities`` the guard identity of the ensemble
    per estimate source.
    """
    context = _stage(plan, arm, stage)
    if not isinstance(item, ExecutionItem):
        raise TypeError(f"item must be an ExecutionItem; got {type(item).__name__}")
    if item.kind != context.kind:
        raise ExecutionError(f"the {stage} stage executes {context.kind} items; {item.key} is a {item.kind} item")
    return _derive(context, item, draws, full_distribution, identities)


# ---------- stored results ----------

def result_path(arm: str, stage: str, item_id: str, rendering: str, results_dir: str | Path = RESULTS_DIR) -> Path:
    """``<results_dir>/<arm>/<stage>/<item_id>.<rendering>.json``."""
    if arm not in ARMS or stage not in EXECUTION_STAGES:
        raise ExecutionError(f"unknown arm or stage {arm!r}/{stage!r}")
    if not isinstance(item_id, str) or not _ITEM_ID.fullmatch(item_id):
        raise ExecutionError(f"item_id must be letters, digits, dots, dashes or underscores; got {item_id!r}")
    if rendering not in (IDENTIFYING, ANONYMISED, OUTLOOK):
        raise ExecutionError(f"unknown rendering {rendering!r}")
    return Path(results_dir) / arm / stage / f"{item_id}.{rendering}.json"


def write_result(result: ItemResult, results_dir: str | Path = RESULTS_DIR) -> Path:
    """Store a result at its path; the same content is a no-op, replacing a different result is refused."""
    if not isinstance(result, ItemResult):
        raise TypeError(f"result must be an ItemResult; got {type(result).__name__}")
    target = result_path(result.arm, result.stage, result.item_id, result.rendering, results_dir)
    payload = result.to_dict()
    if target.exists():
        if _read_json(target) == json.loads(json.dumps(payload, allow_nan=False)):
            return target
        raise ExecutionError(f"refusing to replace the different result at {target}")
    _write_json(target, payload)
    return target


def load_result(path: str | Path) -> ItemResult:
    """Read one stored result, refusing a malformed file."""
    return ItemResult.from_dict(_read_json(Path(path)))


def _reuse(context: _Stage, item: ExecutionItem, stored: ItemResult,
           identities: Mapping[str, GuardIdentity]) -> ItemResult:
    """A stored result of this item, plan, settings and calibrators that recomputes from its draws; else refused."""
    ensure_unchanged(context.plan, stored.plan_sha256)
    expected = (context.arm, context.stage, item.item_id, item.kind, item.rendering, item.prompt_sha256,
                context.draws, context.min_parsed)
    actual = (stored.arm, stored.stage, stored.item_id, stored.kind, stored.rendering, stored.prompt_sha256,
              stored.ensemble_draws, stored.min_parsed)
    if actual != expected:
        raise ExecutionError(f"the stored result of {item.key} belongs to another stage, item, prompt or ensemble "
                             "size; refused, never merged")
    if dict(stored.identities) != dict(identities):
        raise ExecutionError(f"the stored result of {item.key} was computed under another client setting or "
                             "calibrator; refused, never merged")
    recomputed = _derive(context, item, stored.recorded_draws, stored.full_distribution, identities)
    if recomputed.to_dict() != stored.to_dict():
        raise ExecutionError(f"the stored result of {item.key} does not recompute from its recorded draws; refused")
    return stored


# ---------- one stage ----------

@dataclass(frozen=True)
class ExecutionRun:
    """The results of one execution by ``(item_id, rendering)``, their files and the draws settled in it.

    ``reused`` lists the items whose stored result was reused (no request was
    sent for them); ``draws`` holds the settled draws of the others.
    ``calibrators`` reports, per source, the calibrator's fingerprint, weak flag
    and its held-out AUC interval with prompts resampled as clusters, the
    interval to report. ``status`` is the stage ledger's status afterwards.
    """

    results: Mapping[tuple[str, str], ItemResult]
    paths: Mapping[tuple[str, str], Path]
    draws: tuple[ExecutionDraw, ...]
    reused: tuple[tuple[str, str], ...]
    calibrators: Mapping[str, Mapping[str, Any]]
    status: str


def execute_items(
    plan: StudyPlan, items: Iterable[ExecutionItem], *, arm: str, stage: str, client: CompletionClient,
    ledger: BudgetLedger, calibrators: Mapping[str, CalibrationRecord],
    full_distribution: FullDistributionClient | None = None, evidence: EvidenceSettings | None = None,
    recorded: Iterable[ExecutionDraw] = (), on_draw: Callable[[ExecutionDraw], None] | None = None,
    on_request: Callable[[ExecutionItem, int, int | None], None] | None = None,
    on_evidence: Callable[[ExecutionItem, Mapping[int, FullDistributionEvidence]], None] | None = None,
    results_dir: str | Path = RESULTS_DIR, sleep: Callable[[float], None] = time.sleep,
) -> ExecutionRun:
    """Execute study items of one arm and stage: one recorded ensemble each, every variant derived from it.

    Before any request: the plan must be complete; the items must be planned
    items and renderings of the stage, each once and of the stage's kind; the
    ledger must belong to the plan, arm and stage and admit every request of
    the client; the client must carry the frozen arm settings; the calibrators
    must be the plan's, arm's, kind's and sources' and match the client; a
    full-distribution client is needed when the arm computes that source;
    stored results are reused or refused; recorded draws are validated.

    1. Generation pass, in one ``with ledger:`` session: the draws each item
       and rendering still lacks, each announced to ``on_request`` before it is
       sent and handed to ``on_draw`` after it returns (the recording contract
       of the module docstring). A frozen limit raises
       :class:`~scripts.finance_sim.limits.AdmissionClosedError` with the stage
       marked incomplete and every recorded draw kept.
    2. Evidence pass (full-distribution source only), after generation and
       outside the ledger session, one batch per item and rendering, each item
       result written as soon as its draws settle.
       :class:`~scripts.finance_sim.calibrate.EvidenceUnavailableError` leaves
       the stage open with every recorded draw and written result kept; a
       rerun with ``recorded`` sends no generation request and only the
       missing batches.
    3. Once every planned item and rendering of the stage has its result, the
       stage is marked complete; a run of some of them leaves it open.
    """
    context = _stage(plan, arm, stage)
    items = _check_items(context, items)
    _check_request_path(context, client, ledger)
    records = _check_calibrators(context, calibrators)
    settings = _evidence_settings(context, records, evidence)
    if context.full and not callable(getattr(full_distribution, "evidence_batch", None)):
        raise ExecutionError(f"the {arm} arm computes full_distribution estimates: pass a full-distribution client "
                             "with evidence_batch (task 6.4); nothing was sent")
    for value, name in ((on_draw, "on_draw"), (on_request, "on_request"), (on_evidence, "on_evidence"),
                        (sleep, "sleep")):
        _check_callable(value, name)
    labels = _labels(context, settings)
    identities = {item.key: _identities(context, item, client, records, labels) for item in items}
    stored: dict[tuple[str, str], ItemResult] = {}
    for item in items:
        path = result_path(arm, stage, item.item_id, item.rendering, results_dir)
        if path.exists():
            stored[item.key] = _reuse(context, item, load_result(path), identities[item.key])
    reused = _reusable(context, items, recorded, identities, records, client.config)
    remaining = [item for item in items if item.key not in stored]
    with ledger:
        generated = _generate(context, remaining, client, ledger, records, labels, identities, reused, on_draw,
                              on_request)
    by_item: dict[tuple[str, str], list[ExecutionDraw]] = {}
    for draw in generated:
        by_item.setdefault((draw.item_id, draw.rendering), []).append(draw)
    results: dict[tuple[str, str], ItemResult] = {}
    paths: dict[tuple[str, str], Path] = {}
    settled: list[ExecutionDraw] = []
    for item in items:
        if item.key in stored:
            results[item.key] = stored[item.key]
            paths[item.key] = result_path(arm, stage, item.item_id, item.rendering, results_dir)
            continue
        draws = by_item[item.key]
        if context.full:
            draws = _settle(item, draws, full_distribution=full_distribution, evidence=settings,
                            record=records[FULL_DISTRIBUTION], on_draw=on_draw, on_evidence=on_evidence, sleep=sleep)
        settled.extend(draws)
        result = _derive(context, item, [draw.draw for draw in draws],
                         [draw.full_distribution for draw in draws] if context.full else None, identities[item.key])
        paths[item.key] = write_result(result, results_dir)
        results[item.key] = result
    if set(results) == set(planned_keys(plan, arm, stage)) and ledger.status == OPEN:
        ledger.complete()
    quality = {source: MappingProxyType({
        "fingerprint": record.calibrator.fingerprint, "is_weak": record.calibrator.is_weak,
        "holdout_auc_interval_by_prompt": record.holdout_auc_interval_by_prompt}) for source, record in records.items()}
    return ExecutionRun(results=MappingProxyType(results), paths=MappingProxyType(paths), draws=tuple(settled),
                        reused=tuple(key for key in (item.key for item in items) if key in stored),
                        calibrators=MappingProxyType(quality), status=ledger.status)


__all__ = [
    "DEFINITIONS",
    "EDGAR_NO_OUTCOME",
    "EXECUTION_STAGES",
    "INTERRUPTED",
    "KIND_RENDERINGS",
    "REALIZED_MISSING",
    "RESULTS_DIR",
    "RESULT_SCHEMA",
    "SIGN_FROM_RAW_REASON",
    "UNRESOLVED_SUMMARY",
    "ExecutionDraw",
    "ExecutionError",
    "ExecutionItem",
    "ExecutionRun",
    "ItemResult",
    "SourceEstimate",
    "VariantResult",
    "derive_result",
    "execute_items",
    "full_distribution_estimate",
    "generate_draws",
    "load_calibrators",
    "load_result",
    "macro_items",
    "planned_keys",
    "result_path",
    "write_result",
]
