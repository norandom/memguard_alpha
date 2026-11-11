"""Calibration on the held-out macro months and on the EDGAR calibration split (task 7.6).

One calibrator per arm, estimate source and corpus kind (Requirements 4.1, 4.3, 6.3):

- Macro: the frozen calibration months of the plan (``MonthSplit.calibration``,
  on or before every model's cutoff and disjoint from the pilot and evaluation
  months), each rendered with the evaluation renderers of
  :mod:`recall_guard.dataset.macro_regime`: the identifying prompt is the
  memorized class (label 1) and the anonymised prompt the control class (label 0).
- EDGAR: the frozen calibration rows (``EdgarSplit.calibration_memorized``, label
  1, and ``EdgarSplit.calibration_control``, label 0) in the outlook prompt format
  of :mod:`recall_guard.dataset.edgar_outlook`. A negative-control item is never a
  calibration row.

The two kinds are kept apart: each has its own corpus, ledger draws, fit and
file (``<calibrators>/<arm>/<source>/<macro|edgar>.json``), and
:func:`load_calibrator` loads a file only for the kind, arm and source it was
fitted for, so the macro calibrator is never applied to EDGAR prompts.

Corpus. Prompts are listed memorized class first, then control, each in the
frozen plan order. ``corpus_sha256`` is the SHA-256 of the canonical JSON of the
ordered ``[prompt_sha256, label]`` list. A record keeps the plan hash and, per
prompt, the item id, rendering, label and prompt hash, so a load recomputes the
layout from the frozen plan and refuses any difference.

Draws (:func:`collect_draws`, the generation pass). One recorded ensemble per
prompt: ``draws`` requests in the evaluation settings (the configured
temperature, seed ``seed_base + draw_index``), never retried by this loop. Every
request is admitted by the stage's :class:`~scripts.finance_sim.limits.BudgetLedger`:
the client's budget must be that ledger's
:class:`~scripts.finance_sim.limits.LedgerCallBudget`, and the ledger's allowance
must be the client's real ``max_tokens`` and timeout. A reply is parsed with the
evaluation decision definition; each parsed draw gives its top-k feature row from
its own logprobs, through the function the guarded-decision API uses per draw,
and keeps its reply text (private, like every study record) for the evidence
pass. ``recorded`` draws are reused, never requested again.

Settings. The frozen plan (7.3) carries, per arm, the draws of each corpus
(``ArmPlan.calibration_draws``: the probe arm's macro months 8, its EDGAR rows 4),
the fit gates (``ArmPlan.fit``) and the evidence settings (``ArmPlan.evidence``);
:func:`plan_draw_settings`, :func:`plan_fit_settings` and
:func:`plan_evidence_settings` read them, and :func:`run_calibration` uses them
when no settings are passed. The module constants are the defaults only for a
plan without its 7.3 sections and for the corpus-level functions, which take no
plan.

Recording seam (the contract for the study recording, task 7.4, as in
:mod:`scripts.finance_sim.execute`):

- ``on_request(prompt, draw_index, seed)`` is called before each generation
  request is sent, with the corpus prompt and the draw's deterministic seed. The
  recorder persists the request then, before the send.
- ``on_draw(draw)`` receives every new draw after its request returned, before
  the next request, and again when its full-distribution evidence settles. The
  recorder keeps the latest version per draw.
- ``recorded`` takes the :class:`CalibrationDraw` values the recorder kept. A
  request the recorder persisted but whose outcome it never heard (the process
  died while it was in flight) is passed back as the failed draw
  :meth:`CalibrationDraw.interrupted` builds: no reply,
  ``request_error:interrupted``, the SHA-256 of the empty string, the seed
  ``on_request`` reported, the draw temperature and every estimate source of the
  run settled to None. It stays in its ensemble as a failed draw and is never
  sent again, so a resumed stage spends exactly one admission per draw and can
  still complete within limits sized to one ensemble per prompt and rendering.

Evidence (:func:`collect_evidence`, the evidence pass). Full-distribution
evidence is requested after generation, with the full-distribution host loaded
alone (two model loads never coexist), outside the ledger session, as the frozen
plan's evidence rule states: evidence calls never count against the generation
limits. One ``evidence_batch`` request per prompt and rendering carries the
parsed replies in draw order, with the evidence timeout of
:class:`EvidenceSettings` (the frozen plan's 600 s), and each returned
evidence object gives its draw's full-distribution row exactly as the per-draw
path does. A failed or unparsed draw gets no evidence. ``host_busy`` (the host's
memory-low refusal) is retried a bounded number of times after a wait; any other
host failure, or retries that run out, raises :class:`EvidenceUnavailableError`:
the stage stays open (not complete), every recorded draw is kept, nothing is
fitted, and a rerun repeats only the missing evidence. One ensemble therefore
serves both sources of an arm.

Fit. Per source, from the feature rows: the control baseline as
``mia.control.build_baseline`` computes it (per-feature mean and population
standard deviation, floored), then the classifier of ``mia.mcs.train``
(``LogisticRegression(class_weight="balanced", solver="liblinear")``, a stratified
25 percent holdout, weak when the held-out AUC is below ``min_auc``). Unlike
``mcs.train``, which splits rows, the holdout is split by prompt, never by draw,
so sibling draws of one prompt never sit on both sides; with one draw per prompt
the fit equals ``mcs.train``. The split seed is the plan seed modulo 2**32. The
record keeps two held-out AUC intervals: the calibrator's own
(:meth:`SourceCalibrator.holdout_auc_interval`, which resamples draws as if they
were independent) and a prompt-clustered one (held-out prompts resampled with all
their draws), which is the one to report.

Identity. The calibrator's identity is the client's configured identity with the
temperature override applied and seed keys left out, as the guarded-decision API
compares it, plus the estimate source and the corpus hash, set when it is fitted.
The fingerprint follows the scorer's calibration payload scheme over that identity.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import os
import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from recall_guard.core.completion import CompletionClient, ModelConfig, ModelResponse, ParameterValue
from recall_guard.core.decision import DecisionDefinition, DecisionParseError
from recall_guard.core.full_distribution import FullDistributionClient, FullDistributionEvidence
from recall_guard.core.request_control import BudgetExhaustedError, RequestError
from recall_guard.core.significance import clustered_auc_interval, rank_auc
from recall_guard.dataset.edgar_outlook import outlook_definition, prompt_sha256, render_outlook_prompt
from recall_guard.dataset.macro_regime import MacroRow, regime_definition, render_anonymised, render_identifying
from recall_guard.harness.guarded_decision import PARSE_ERROR, REQUEST_ERROR_PREFIX, SourceCalibrator
from recall_guard.harness.request_failures import failure_kind
from recall_guard.harness.scorer_client import CalibrationIdentity, identity_settings
from recall_guard.mia.control import _STD_FLOOR, ControlBaseline, standardise
from recall_guard.mia.estimate_source import EstimateSource, features_from_full, features_from_top_k
from recall_guard.mia.features import MiaFeatures
from recall_guard.mia.mcs import MCSCalibrator
from scripts.finance_sim.limits import (
    ARMS,
    OPEN,
    RUNS_ROOT,
    SERVING_ARM,
    BudgetLedger,
    LedgerCallBudget,
    RequestAllowance,
)
from scripts.finance_sim.plan import (
    EVIDENCE_HOST_BUSY_RETRIES,
    EVIDENCE_HOST_BUSY_WAIT_S,
    EVIDENCE_TIMEOUT_S,
    EVIDENCE_TOP_K_KEEP,
    FIT_MIN_AUC,
    FIT_MIN_VALID,
    ArmPlan,
    PlanChangedError,
    StudyPlan,
    ensure_unchanged,
    sha256_of,
)

MACRO, EDGAR = "macro", "edgar"
KINDS: tuple[str, ...] = (MACRO, EDGAR)
IDENTIFYING, ANONYMISED, OUTLOOK = "identifying", "anonymised", "outlook"
RENDERINGS: Mapping[str, tuple[str, ...]] = MappingProxyType({MACRO: (IDENTIFYING, ANONYMISED), EDGAR: (OUTLOOK,)})
MEMORIZED, CONTROL = 1, 0
#: The ledger stage of both calibrations (the serving arm's StageLimits stage of the same name).
CALIBRATION_STAGE = "calibration"
CALIBRATORS_DIR = RUNS_ROOT / "calibrators"
CALIBRATOR_SCHEMA = "finance-sim-calibrator/1"
#: The scorer's calibration payload scheme (``MemoryGuardedScorer.calibration_fingerprint``).
FIT_SCHEME = "recall_guard.calibration.v1"
#: The classifier's input columns, as ``mcs.train`` orders them without a reference model.
FEATURE_ORDER: tuple[str, ...] = ("loss", "min_k", "min_k_pp", "zlib_ratio")
#: Held-out share of the prompts, as ``mcs.train`` holds out rows.
HOLDOUT_FRACTION = 0.25
#: Prompts are passed as rendered, without a rewriting intervention.
INTERVENTION_ID = "original"
#: Settings of the prompt-clustered interval, those of ``local_reporting.held_out_discrimination``.
INTERVAL_RESAMPLES = 1000
INTERVAL_CONFIDENCE = 0.95
INTERVAL_SEED = 0
CLUSTERED_METHOD = "prompt_clustered_stratified_percentile_bootstrap"
#: Evidence pass: one ``evidence_batch`` request per prompt and rendering over its parsed replies in draw order.
EVIDENCE_MODE = "batch_per_prompt_and_rendering"
#: The defaults below are the frozen plan's values (scripts/finance_sim/plan.py, ``EvidencePlan`` and
#: ``FitPlan``), used only where no plan is at hand. EVIDENCE_TIMEOUT_S bounds one evidence batch (about one
#: prompt evaluation plus every reply's decodes) and EVIDENCE_TOP_K_KEEP is the alternatives kept per position,
#: the profile's ``full_distribution.top_k_keep``; both are imported from the plan module above.
#: ``host_busy`` (HTTP 503, memory low on the host) is retried this many times, after this wait each time.
HOST_BUSY_RETRIES = EVIDENCE_HOST_BUSY_RETRIES
HOST_BUSY_WAIT_S = EVIDENCE_HOST_BUSY_WAIT_S
HOST_BUSY = "host_busy"
#: Failure code of a request that was sent (``on_request`` was called) but whose outcome was lost.
INTERRUPTED = REQUEST_ERROR_PREFIX + "interrupted"
#: The split seed must be a valid ``random_state`` for scikit-learn.
_SEED_MODULUS = 2**32
#: Statuses that mean the credential was rejected; no later draw can succeed.
_AUTH_STATUS = frozenset({401, 403})
_SHA256 = re.compile(r"[0-9a-f]{64}")
_RECORD_KEYS = frozenset({"schema", "plan_sha256", "arm", "kind", "source", "corpus", "draw_settings", "fit_settings",
                          "split_seed", "holdout_prompts", "counts", "holdout_auc_interval",
                          "holdout_auc_interval_by_prompt", "evidence_settings", "fit", "fingerprint"})
_FIT_KEYS = frozenset({"scheme", "identity", "observed_fit_settings", "baseline", "model", "feature_order",
                       "coefficients", "intercept", "classes", "holdout_auc", "is_weak", "valid_counts",
                       "holdout_labels", "holdout_probabilities"})
_DRAW_KEYS = frozenset({"kind", "item_id", "rendering", "label", "prompt_sha256", "draw_index", "seed", "temperature",
                        "response_sha256", "response_text", "failure_code", "features", "evidence_ref"})
_EVIDENCE_KEYS = frozenset({"mode", "timeout_s", "top_k_keep", "host_busy_retries", "host_busy_wait_s"})
_FULL = EstimateSource.FULL_DISTRIBUTION.value


class CalibrationError(ValueError):
    """A calibration corpus, its draws, a fit or a stored calibrator is refused."""


class EvidenceUnavailableError(RuntimeError):
    """The full-distribution host gave no evidence for a prompt's replies: still busy after the retries, or failed.

    ``code`` is the host failure's code (``host_busy`` after the retries ran out).
    The calibration stage stays open with every recorded draw kept and nothing
    fitted; a rerun repeats only the missing evidence.
    """

    def __init__(self, message: str, *, code: str | None, prompt_sha256: str) -> None:
        super().__init__(message)
        self.code = code
        self.prompt_sha256 = prompt_sha256


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
        raise CalibrationError(f"{name} must be a non-empty string; got {value!r}")
    return value


def _hex(value: object, name: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise CalibrationError(f"{name} must be a SHA-256 as 64 lowercase hex characters; got {value!r}")
    return value


def _label(value: object, name: str = "label") -> int:
    if not _is_int(value) or value not in (MEMORIZED, CONTROL):
        raise CalibrationError(f"{name} must be the integer 1 (memorized) or 0 (control); got {value!r}")
    return value


def _kind(value: object) -> str:
    if value not in KINDS:
        raise CalibrationError(f"kind must be one of {list(KINDS)}; got {value!r}")
    return value


def _arm(value: object) -> str:
    if value not in ARMS:
        raise CalibrationError(f"arm must be one of {list(ARMS)}; got {value!r}")
    return value


def _source(value: object) -> EstimateSource:
    try:
        return EstimateSource(value)
    except ValueError:
        raise CalibrationError(f"estimate source must be one of {[s.value for s in EstimateSource]}; "
                               f"got {value!r}") from None


def _temperature(value: object, name: str = "temperature") -> float | None:
    if value is None:
        return None
    if not _is_real(value) or not math.isfinite(value) or value < 0:
        raise CalibrationError(f"{name} must be a finite number of at least 0, or None; got {value!r}")
    return float(value)


def _plain(value: Any) -> Any:
    """JSON-ready form: mappings as dicts, tuples as lists."""
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_plain(item) for item in value]
    return value


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _ in pairs]
    if len(set(keys)) != len(keys):
        raise CalibrationError(f"duplicate keys {sorted({key for key in keys if keys.count(key) > 1})}")
    return dict(pairs)


def _no_constant(name: str) -> Any:
    raise CalibrationError(f"non-finite number {name}")


def _read_json(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CalibrationError(f"{path} cannot be read: {exc}") from None
    try:
        return json.loads(text, object_pairs_hook=_no_duplicate_keys, parse_constant=_no_constant)
    except CalibrationError as exc:
        raise CalibrationError(f"{path}: {exc}") from None
    except ValueError as exc:
        raise CalibrationError(f"{path} is not valid JSON: {exc}") from None


def _exact_keys(data: object, keys: frozenset[str], name: str) -> Mapping[str, Any]:
    if not isinstance(data, Mapping):
        raise CalibrationError(f"{name} must be a JSON object")
    if set(data) != keys:
        raise CalibrationError(f"{name} fields differ: missing {sorted(keys - set(data))}, "
                               f"undeclared {sorted(map(str, set(data) - keys))}")
    return data


# ---------- the corpus ----------

def macro_item_id(month: str) -> str:
    """The study item id of a decision month (``macro-YYYY-MM``), as the item assembly (7.1) names it."""
    return f"macro-{month}"


@dataclass(frozen=True)
class CorpusPrompt:
    """One calibration prompt: its study item, rendering, class label and text."""

    item_id: str
    rendering: str
    label: int
    prompt: str
    prompt_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _text(self.item_id, "item_id")
        if self.rendering not in (IDENTIFYING, ANONYMISED, OUTLOOK):
            raise CalibrationError(f"rendering must be identifying, anonymised or outlook; got {self.rendering!r}")
        _label(self.label)
        _text(self.prompt, "prompt")
        object.__setattr__(self, "prompt_sha256", prompt_sha256(self.prompt))


@dataclass(frozen=True)
class CorpusEntry:
    """What a record keeps of one corpus prompt: no text, only its identity, label and hash."""

    item_id: str
    rendering: str
    label: int
    prompt_sha256: str

    def __post_init__(self) -> None:
        _text(self.item_id, "item_id")
        if self.rendering not in (IDENTIFYING, ANONYMISED, OUTLOOK):
            raise CalibrationError(f"rendering must be identifying, anonymised or outlook; got {self.rendering!r}")
        _label(self.label)
        _hex(self.prompt_sha256, "prompt_sha256")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def corpus_hash(entries: Iterable[CorpusEntry]) -> str:
    """SHA-256 of the canonical JSON of the ordered ``[prompt_sha256, label]`` list."""
    return sha256_of([[entry.prompt_sha256, entry.label] for entry in entries])


@dataclass(frozen=True)
class CalibrationCorpus:
    """The prompts of one calibration corpus, memorized class first, then control, in frozen plan order."""

    kind: str
    plan_sha256: str
    prompts: tuple[CorpusPrompt, ...]

    def __post_init__(self) -> None:
        _kind(self.kind)
        _hex(self.plan_sha256, "plan_sha256")
        prompts = tuple(self.prompts)
        if not all(isinstance(prompt, CorpusPrompt) for prompt in prompts):
            raise TypeError("prompts must be CorpusPrompt values")
        if any(prompt.rendering not in RENDERINGS[self.kind] for prompt in prompts):
            raise CalibrationError(f"a {self.kind} corpus holds only {list(RENDERINGS[self.kind])} renderings")
        labels = [prompt.label for prompt in prompts]
        if MEMORIZED not in labels or CONTROL not in labels:
            raise CalibrationError("a calibration corpus needs prompts of both classes")
        if labels != sorted(labels, reverse=True):
            raise CalibrationError("a calibration corpus lists the memorized class first, then the control class")
        if len({(p.item_id, p.rendering) for p in prompts}) != len(prompts):
            raise CalibrationError("a calibration corpus must not repeat an item rendering")
        if len({p.prompt_sha256 for p in prompts}) != len(prompts):
            raise CalibrationError("a calibration corpus must not repeat a prompt")
        object.__setattr__(self, "prompts", prompts)

    @property
    def definition(self) -> DecisionDefinition:
        """The evaluation decision definition of this kind: the macro regime or the EDGAR outlook."""
        return regime_definition if self.kind == MACRO else outlook_definition

    @property
    def entries(self) -> tuple[CorpusEntry, ...]:
        return tuple(CorpusEntry(p.item_id, p.rendering, p.label, p.prompt_sha256) for p in self.prompts)

    @property
    def corpus_sha256(self) -> str:
        return corpus_hash(self.entries)


def _check_plan(plan: StudyPlan) -> None:
    """The frozen calibration sets must stay apart from the pilot, evaluation and negative-control sets.

    ``StudyPlan`` already refuses overlapping sets when it is built; this check
    keeps the calibration from relying on that alone.
    """
    if not isinstance(plan, StudyPlan):
        raise TypeError(f"plan must be a StudyPlan; got {type(plan).__name__}")
    calibration = set(plan.months.calibration)
    for name in ("evaluation", "pilot"):
        shared = sorted(calibration & set(getattr(plan.months, name)))
        if shared:
            raise CalibrationError(f"the plan's calibration months overlap its {name} months: {shared}")
    rows = set(plan.edgar.calibration_memorized) | set(plan.edgar.calibration_control)
    for name in ("negative_control", "unused"):
        shared = sorted(rows & set(getattr(plan.edgar, name)))
        if shared:
            raise CalibrationError(f"the plan's EDGAR calibration rows overlap its {name} rows: {shared}")


def plan_corpus_layout(plan: StudyPlan, kind: str) -> tuple[tuple[str, str, int], ...]:
    """``(item_id, rendering, label)`` of every corpus prompt, in corpus order, as the frozen plan defines them."""
    _check_plan(plan)
    if _kind(kind) == MACRO:
        months = plan.months.calibration
        return (tuple((macro_item_id(month), IDENTIFYING, MEMORIZED) for month in months)
                + tuple((macro_item_id(month), ANONYMISED, CONTROL) for month in months))
    return (tuple((row_id, OUTLOOK, MEMORIZED) for row_id in plan.edgar.calibration_memorized)
            + tuple((row_id, OUTLOOK, CONTROL) for row_id in plan.edgar.calibration_control))


def macro_corpus(plan: StudyPlan, rows: Iterable[MacroRow]) -> CalibrationCorpus:
    """The macro calibration corpus: both renderings of every frozen calibration month (4.1, 6.3).

    ``rows`` must hold exactly one :class:`MacroRow` for each of the plan's
    calibration months. A row of an evaluation or pilot month is refused, as is
    a missing, repeated or unplanned month, before any request.
    """
    _check_plan(plan)
    by_month: dict[str, MacroRow] = {}
    for row in rows:
        if not isinstance(row, MacroRow):
            raise TypeError(f"rows must hold MacroRow values; got {type(row).__name__}")
        month = f"{row.decision_date:%Y-%m}"
        if month in by_month:
            raise CalibrationError(f"the macro row of {month} is given more than once")
        by_month[month] = row
    supplied = set(by_month)
    for name in ("evaluation", "pilot"):
        shared = sorted(supplied & set(getattr(plan.months, name)))
        if shared:
            raise CalibrationError(f"calibration months must not overlap the {name} months; {name} months given: "
                                   f"{shared}")
    calibration = plan.months.calibration
    unplanned = sorted(supplied - set(calibration))
    if unplanned:
        raise CalibrationError(f"months {unplanned} are not frozen calibration months of the plan")
    missing = sorted(set(calibration) - supplied)
    if missing:
        raise CalibrationError(f"the macro rows of frozen calibration months are missing: {missing}")
    prompts = [CorpusPrompt(macro_item_id(m), IDENTIFYING, MEMORIZED, render_identifying(by_month[m]))
               for m in calibration]
    prompts += [CorpusPrompt(macro_item_id(m), ANONYMISED, CONTROL, render_anonymised(by_month[m])) for m in calibration]
    return CalibrationCorpus(MACRO, plan.plan_sha256, tuple(prompts))


@dataclass(frozen=True)
class EdgarRelease:
    """One EDGAR calibration release: the plan's row id, its label and the record's ``prompt`` text."""

    row_id: str
    label: int
    release_text: str

    def __post_init__(self) -> None:
        _text(self.row_id, "row_id")
        _label(self.label)
        _text(self.release_text, "release_text")


def edgar_corpus(plan: StudyPlan, releases: Iterable[EdgarRelease]) -> CalibrationCorpus:
    """The EDGAR calibration corpus: the frozen calibration rows in the outlook prompt format (4.1).

    ``releases`` must hold exactly the plan's calibration rows with their frozen
    labels. A negative-control item or an unused held-out row is refused, as is
    a missing, repeated, relabelled or unplanned row, before any request.
    """
    _check_plan(plan)
    by_id: dict[str, EdgarRelease] = {}
    for release in releases:
        if not isinstance(release, EdgarRelease):
            raise TypeError(f"releases must hold EdgarRelease values; got {type(release).__name__}")
        if release.row_id in by_id:
            raise CalibrationError(f"EDGAR row {release.row_id!r} is given more than once")
        by_id[release.row_id] = release
    supplied = set(by_id)
    for name, what in (("negative_control", "negative-control items"), ("unused", "unused held-out rows")):
        shared = sorted(supplied & set(getattr(plan.edgar, name)))
        if shared:
            raise CalibrationError(f"EDGAR calibration rows must not overlap the {what}; given: {shared}")
    memorized, control = plan.edgar.calibration_memorized, plan.edgar.calibration_control
    unplanned = sorted(supplied - set(memorized) - set(control))
    if unplanned:
        raise CalibrationError(f"EDGAR rows {unplanned} are not frozen calibration rows of the plan")
    missing = sorted((set(memorized) | set(control)) - supplied)
    if missing:
        raise CalibrationError(f"frozen EDGAR calibration rows are missing: {missing}")
    for row_ids, label in ((memorized, MEMORIZED), (control, CONTROL)):
        relabelled = sorted(row_id for row_id in row_ids if by_id[row_id].label != label)
        if relabelled:
            raise CalibrationError(f"EDGAR rows {relabelled} carry another label than the frozen label {label}")
    prompts = [CorpusPrompt(row_id, OUTLOOK, label, render_outlook_prompt(by_id[row_id].release_text))
               for row_ids, label in ((memorized, MEMORIZED), (control, CONTROL)) for row_id in row_ids]
    return CalibrationCorpus(EDGAR, plan.plan_sha256, tuple(prompts))


# ---------- settings ----------

@dataclass(frozen=True)
class DrawSettings:
    """The evaluation draw settings: ``draws`` requests per prompt, the temperature and the seed base.

    ``temperature`` None keeps the client's configured setting; ``seed_base``
    sets the seed of draw ``i`` to ``seed_base + i``, None sends no seed override.
    """

    draws: int = 32
    temperature: float | None = None
    seed_base: int | None = None

    def __post_init__(self) -> None:
        if not _is_int(self.draws) or self.draws < 1:
            raise CalibrationError(f"draws must be a positive integer; got {self.draws!r}")
        object.__setattr__(self, "temperature", _temperature(self.temperature))
        if self.seed_base is not None and not _is_int(self.seed_base):
            raise CalibrationError(f"seed_base must be an integer or None; got {self.seed_base!r}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FitSettings:
    """The fit's quality gates: the weak-calibrator AUC threshold and the baseline's minimum control rows."""

    min_auc: float = FIT_MIN_AUC
    min_valid: int = FIT_MIN_VALID

    def __post_init__(self) -> None:
        if not _is_real(self.min_auc) or not 0.0 <= self.min_auc <= 1.0:
            raise CalibrationError(f"min_auc must be a number in [0, 1]; got {self.min_auc!r}")
        object.__setattr__(self, "min_auc", float(self.min_auc))
        if not _is_int(self.min_valid) or self.min_valid < 1:
            raise CalibrationError(f"min_valid must be a positive integer; got {self.min_valid!r}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvidenceSettings:
    """The evidence pass: one batch per prompt and rendering, its timeout, the kept alternatives and the busy retries."""

    timeout_s: float = EVIDENCE_TIMEOUT_S
    top_k_keep: int = EVIDENCE_TOP_K_KEEP
    host_busy_retries: int = HOST_BUSY_RETRIES
    host_busy_wait_s: float = HOST_BUSY_WAIT_S

    def __post_init__(self) -> None:
        for name in ("timeout_s", "host_busy_wait_s"):
            value = getattr(self, name)
            if not _is_real(value) or not math.isfinite(value) or value < 0 or (name == "timeout_s" and value == 0):
                raise CalibrationError(f"{name} must be a finite, non-negative number (timeout_s above 0); "
                                       f"got {value!r}")
            object.__setattr__(self, name, float(value))
        if not _is_int(self.top_k_keep) or self.top_k_keep < 1:
            raise CalibrationError(f"top_k_keep must be a positive integer; got {self.top_k_keep!r}")
        if not _is_int(self.host_busy_retries) or self.host_busy_retries < 0:
            raise CalibrationError(f"host_busy_retries must be a non-negative integer; got {self.host_busy_retries!r}")

    def to_dict(self) -> dict[str, Any]:
        return {"mode": EVIDENCE_MODE, **asdict(self)}

    @classmethod
    def from_dict(cls, data: object) -> EvidenceSettings:
        mapping = _exact_keys(data, _EVIDENCE_KEYS, "evidence settings")
        if mapping["mode"] != EVIDENCE_MODE:
            raise CalibrationError(f"unsupported evidence mode {mapping['mode']!r}; expected {EVIDENCE_MODE!r}")
        return cls(**{key: value for key, value in mapping.items() if key != "mode"})


def _frozen_arm(plan: StudyPlan, arm: str) -> ArmPlan:
    """The frozen arm plan of ``arm``; a plan without its 7.3 sections is refused."""
    if not isinstance(plan, StudyPlan):
        raise TypeError(f"plan must be a StudyPlan; got {type(plan).__name__}")
    if plan.arms is None:
        raise CalibrationError(f"plan {plan.plan_sha256} lacks the sections ['arms'] that freeze the calibration "
                               "settings; freeze it with scripts/finance_sim/build_plan.py")
    return plan.arms.arms[_arm(arm)]


def plan_draw_settings(plan: StudyPlan, arm: str, kind: str) -> DrawSettings:
    """The frozen draw settings of one arm's calibration corpus: its draws per prompt, temperature and seed base."""
    frozen = _frozen_arm(plan, arm)
    return DrawSettings(draws=frozen.calibration_draws[_kind(kind)], temperature=frozen.temperature,
                        seed_base=frozen.seed_base)


def plan_fit_settings(plan: StudyPlan, arm: str) -> FitSettings:
    """The frozen fit gates of one arm (``ArmPlan.fit``)."""
    fit = _frozen_arm(plan, arm).fit
    return FitSettings(min_auc=fit.min_auc, min_valid=fit.min_valid)


def plan_evidence_settings(plan: StudyPlan, arm: str) -> EvidenceSettings | None:
    """The frozen evidence settings of one arm (``ArmPlan.evidence``); None for an arm without that source."""
    evidence = _frozen_arm(plan, arm).evidence
    return None if evidence is None else EvidenceSettings(**evidence.to_dict())


def split_seed(plan: StudyPlan) -> int:
    """The holdout split seed: the frozen plan seed modulo 2**32 (a valid scikit-learn ``random_state``)."""
    if not isinstance(plan, StudyPlan):
        raise TypeError(f"plan must be a StudyPlan; got {type(plan).__name__}")
    return plan.seed % _SEED_MODULUS


def estimate_sources(arm: str, sources: Iterable[EstimateSource | str],
                     full_distribution: FullDistributionClient | None) -> tuple[EstimateSource, ...]:
    """Validate the estimate sources to calibrate for ``arm``, before any request.

    The serving arm calibrates top-k only: its full-distribution variants are
    reported as missing, because the full-distribution host serves other weights.
    A full-distribution source needs a full-distribution client.
    """
    parsed = _arm_sources(arm, sources)
    if EstimateSource.FULL_DISTRIBUTION in parsed:
        if full_distribution is None:
            raise CalibrationError("a full_distribution calibrator needs a full-distribution client "
                                   "(full_distribution=...); nothing was sent")
        if not callable(getattr(full_distribution, "evidence_batch", None)):
            raise TypeError("the full-distribution client must offer evidence_batch (task 6.4)")
    return parsed


def _arm_sources(arm: str, sources: Iterable[EstimateSource | str]) -> tuple[EstimateSource, ...]:
    """The sources to calibrate for ``arm``: at least one, none repeated, top_k only on the serving arm."""
    _arm(arm)
    if isinstance(sources, str | EstimateSource):
        raise TypeError("sources must be a sequence of estimate sources")
    parsed = tuple(_source(source) for source in sources)
    if not parsed:
        raise CalibrationError("name at least one estimate source to calibrate")
    if len(set(parsed)) != len(parsed):
        raise CalibrationError("estimate sources must not repeat")
    if EstimateSource.FULL_DISTRIBUTION in parsed and arm == SERVING_ARM:
        raise CalibrationError("the serving arm calibrates top_k only; its full-distribution variants are "
                               "missing because the full-distribution host serves other weights")
    return parsed


def request_allowance(config: ModelConfig, *, usd_per_second: float | None = None) -> RequestAllowance:
    """The ledger allowance of one request of ``config``: its real ``max_tokens`` cap and its timeout."""
    if not isinstance(config, ModelConfig):
        raise TypeError(f"config must be a ModelConfig; got {type(config).__name__}")
    max_tokens = config.parameters.get("max_tokens")
    if not _is_int(max_tokens) or max_tokens < 1:
        raise CalibrationError("the generation settings must cap max_tokens with a positive integer, so the "
                               f"ledger can reserve each request's output; got {max_tokens!r}")
    return RequestAllowance(output_tokens=max_tokens, timeout_s=config.timeout_s, usd_per_second=usd_per_second)


def calibration_identity(client: CompletionClient, settings: DrawSettings, source: EstimateSource | str,
                         corpus: CalibrationCorpus) -> CalibrationIdentity:
    """The identity a calibrator is fitted under, set at fit time (4.1).

    The client's configured identity, with the draw temperature applied and seed
    keys left out (as :func:`~recall_guard.harness.guarded_decision.guarded_run`
    compares it), bound to the estimate source and the corpus hash.
    """
    if not isinstance(settings, DrawSettings):
        raise TypeError("settings must be DrawSettings")
    if not isinstance(corpus, CalibrationCorpus):
        raise TypeError("corpus must be a CalibrationCorpus")
    current = CalibrationIdentity.from_client(client, INTERVENTION_ID)
    parameters = json.loads(current.parameters_json)
    if settings.temperature is not None:
        parameters["temperature"] = settings.temperature
    return replace(current, parameters_json=identity_settings(parameters), estimate_source=_source(source).value,
                   corpus_sha256=corpus.corpus_sha256)


# ---------- draws ----------

@dataclass(frozen=True)
class CalibrationDraw:
    """One recorded calibration draw and its feature row per estimate source.

    ``features`` maps each settled source to the four classifier features of the
    draw, or None when the reply did not parse, the request failed, or the
    source's evidence was missing or unusable. ``evidence_ref`` is the SHA-256 of
    the evidence payload the features came from (None without evidence). A parsed
    draw lacks the ``full_distribution`` key until the evidence pass settles it;
    every other source is settled when the draw is made. ``response_text`` is the
    parsed reply, kept (privately) for the evidence pass; failed and unparsed
    draws keep none.
    """

    kind: str
    item_id: str
    rendering: str
    label: int
    prompt_sha256: str
    draw_index: int
    seed: int | None
    temperature: float | None
    response_sha256: str
    response_text: str | None
    failure_code: str | None
    features: Mapping[str, Mapping[str, float] | None]
    evidence_ref: Mapping[str, str | None]

    def __post_init__(self) -> None:
        _kind(self.kind)
        _text(self.item_id, "item_id")
        if self.rendering not in RENDERINGS[self.kind]:
            raise CalibrationError(f"a {self.kind} draw has a {list(RENDERINGS[self.kind])} rendering; "
                                   f"got {self.rendering!r}")
        _label(self.label)
        _hex(self.prompt_sha256, "prompt_sha256")
        if not _is_int(self.draw_index) or self.draw_index < 0:
            raise CalibrationError(f"draw_index must be a non-negative integer; got {self.draw_index!r}")
        if self.seed is not None and not _is_int(self.seed):
            raise CalibrationError(f"seed must be an integer or None; got {self.seed!r}")
        object.__setattr__(self, "temperature", _temperature(self.temperature))
        _hex(self.response_sha256, "response_sha256")
        if self.failure_code is not None:
            _text(self.failure_code, "failure_code")
            if self.response_text is not None:
                raise CalibrationError("a failed or unparsed draw keeps no reply text")
        elif not isinstance(self.response_text, str) or _sha256(self.response_text) != self.response_sha256:
            raise CalibrationError("a parsed draw keeps its reply text, whose SHA-256 is response_sha256")
        if not isinstance(self.features, Mapping) or not isinstance(self.evidence_ref, Mapping):
            raise CalibrationError("features and evidence_ref must be mappings by estimate source")
        if set(self.features) != set(self.evidence_ref):
            raise CalibrationError("features and evidence_ref must name the same estimate sources")
        if not self.features and self.failure_code is not None:
            raise CalibrationError("a failed or unparsed draw settles every estimate source (None)")
        features: dict[str, Mapping[str, float] | None] = {}
        for name, values in sorted(self.features.items()):
            _source(name)
            if values is not None:
                if self.failure_code is not None:
                    raise CalibrationError("a failed draw carries no features")
                if not isinstance(values, Mapping) or set(values) != set(FEATURE_ORDER):
                    raise CalibrationError(f"features of {name} must hold exactly {list(FEATURE_ORDER)}")
                if not all(_is_real(values[key]) and math.isfinite(values[key]) for key in FEATURE_ORDER):
                    raise CalibrationError(f"features of {name} must be finite numbers")
                values = MappingProxyType({key: float(values[key]) for key in FEATURE_ORDER})
            features[name] = values
        for name, reference in self.evidence_ref.items():
            if reference is not None:
                _hex(reference, f"evidence_ref[{name!r}]")
        object.__setattr__(self, "features", MappingProxyType(features))
        object.__setattr__(self, "evidence_ref", MappingProxyType(dict(sorted(self.evidence_ref.items()))))

    @classmethod
    def interrupted(cls, prompt: CorpusPrompt, draw_index: int, *, seed: int | None, temperature: float | None,
                    sources: Iterable[EstimateSource | str]) -> CalibrationDraw:
        """The failed draw a recorder passes back for a request that was sent but whose outcome was lost.

        ``seed`` is the one ``on_request`` reported (the draw's deterministic
        seed), ``temperature`` the draw temperature and ``sources`` the estimate
        sources of the run, each settled to None. The draw carries no reply, the
        SHA-256 of the empty string and the failure code
        ``request_error:interrupted``; it stays in its ensemble as a failed draw
        and is never requested again. Its corpus (macro or EDGAR) follows from
        the prompt's rendering.
        """
        if not isinstance(prompt, CorpusPrompt):
            raise TypeError(f"prompt must be a CorpusPrompt; got {type(prompt).__name__}")
        if not _is_int(draw_index) or draw_index < 0:
            raise CalibrationError(f"draw_index must be a non-negative integer; got {draw_index!r}")
        if isinstance(sources, str | EstimateSource):
            raise TypeError("sources must be a sequence of estimate sources")
        names = {_source(source).value for source in sources}
        if not names:
            raise CalibrationError("an interrupted draw settles at least one estimate source")
        kind = next(kind for kind, renderings in RENDERINGS.items() if prompt.rendering in renderings)
        empty = {name: None for name in sorted(names)}
        return cls(kind=kind, item_id=prompt.item_id, rendering=prompt.rendering, label=prompt.label,
                   prompt_sha256=prompt.prompt_sha256, draw_index=draw_index, seed=seed, temperature=temperature,
                   response_sha256=_sha256(""), response_text=None, failure_code=INTERRUPTED, features=empty,
                   evidence_ref=dict(empty))

    @property
    def key(self) -> tuple[str, str, str, int]:
        """``(kind, item_id, rendering, draw_index)``: one draw of one prompt."""
        return self.kind, self.item_id, self.rendering, self.draw_index

    @property
    def evidence_pending(self) -> bool:
        """True for a parsed draw whose full-distribution evidence the evidence pass has not settled."""
        return self.failure_code is None and _FULL not in self.features

    def settled(self, row: Mapping[str, float] | None, reference: str | None) -> CalibrationDraw:
        """This draw with its full-distribution row and evidence reference settled."""
        if not self.evidence_pending:
            raise CalibrationError(f"draw {self.key} has no pending full-distribution evidence")
        return replace(self, features={**self.features, _FULL: row}, evidence_ref={**self.evidence_ref, _FULL: reference})

    def _pending_version(self) -> CalibrationDraw:
        """This draw as the generation pass recorded it, before its full-distribution evidence."""
        return replace(self, features={k: v for k, v in self.features.items() if k != _FULL},
                       evidence_ref={k: v for k, v in self.evidence_ref.items() if k != _FULL})

    def feature_row(self, source: EstimateSource) -> MiaFeatures | None:
        """The draw's features of ``source`` as ``MiaFeatures`` (no reference model), or None."""
        values = self.features.get(source.value)
        return None if values is None else MiaFeatures(**values, ref_delta=None)

    def to_dict(self) -> dict[str, Any]:
        return _plain({name: getattr(self, name) for name in sorted(_DRAW_KEYS)})

    @classmethod
    def from_dict(cls, data: object) -> CalibrationDraw:
        return cls(**_exact_keys(data, _DRAW_KEYS, "calibration draw"))


def _is_fatal(error: BaseException) -> bool:
    """A rejected credential or an exhausted request budget: no later request can succeed."""
    return isinstance(error, BudgetExhaustedError) or getattr(error, "status_code", None) in _AUTH_STATUS


def _request(client: CompletionClient, prompt: str,
             parameters: Mapping[str, ParameterValue]) -> tuple[ModelResponse | None, str | None]:
    """One generation request, never retried here, as the guarded-decision API sends it.

    A rejected credential or an exhausted budget (a closed ledger) is raised; any
    other failure is returned as its code. A ``missing_token_evidence`` error that
    carries the complete reply keeps that reply, so only its features are missing.
    """
    try:
        return client.generate(prompt, **parameters), None
    except Exception as error:  # noqa: BLE001 - every other failure is recorded on its draw
        if _is_fatal(error):
            raise
        if (isinstance(error, RequestError) and error.code == "missing_token_evidence"
                and isinstance(error.response_evidence, ModelResponse)):
            return error.response_evidence, None
        if isinstance(error, RequestError) and error.code:
            return None, error.code
        return None, "timeout" if isinstance(error, TimeoutError) else failure_kind(error)


def _row(features: MiaFeatures) -> dict[str, float] | None:
    values = {name: float(getattr(features, name)) for name in FEATURE_ORDER}
    return values if all(math.isfinite(value) for value in values.values()) else None


def _top_k_row(response: ModelResponse) -> tuple[dict[str, float] | None, str | None]:
    """The top-k feature row of one parsed reply and the reference of its token evidence.

    The guarded-decision API's per-draw path: features from the reply's own
    logprobs; missing evidence leaves no reference, unusable evidence no row.
    """
    if not response.token_evidence:
        return None, None
    reference = _payload_sha256(response.to_dict()["token_evidence"])
    try:
        return _row(features_from_top_k(response.content, response.to_legacy().logprobs).features), reference
    except (ValueError, RuntimeError):
        return None, reference


def full_distribution_row(response_text: str,
                          evidence: FullDistributionEvidence) -> tuple[dict[str, float] | None, str]:
    """The full-distribution feature row of one reply and the reference of its evidence.

    The guarded-decision API's per-draw path on that evidence: the reference is
    the SHA-256 of the evidence payload, and evidence that the features cannot
    use (for example uneven alternatives) leaves the row None.
    """
    reference = _payload_sha256(evidence.to_dict())
    try:
        return _row(features_from_full(response_text, evidence).features), reference
    except (ValueError, RuntimeError):
        return None, reference


def _draw_seed(settings: DrawSettings, config: ModelConfig, index: int) -> int | None:
    if settings.seed_base is not None:
        return settings.seed_base + index
    configured = config.parameters.get("seed")
    return configured if _is_int(configured) else None


def _draw_temperature(settings: DrawSettings, config: ModelConfig) -> float | None:
    if settings.temperature is not None:
        return settings.temperature
    configured = config.parameters.get("temperature")
    return float(configured) if _is_real(configured) else None


def _check_request_path(corpus: CalibrationCorpus, client: CompletionClient, ledger: BudgetLedger,
                        sources: Sequence[EstimateSource]) -> None:
    """Every request must be admitted by the calibration stage's ledger with the client's real allowance."""
    if not isinstance(corpus, CalibrationCorpus):
        raise TypeError("corpus must be a CalibrationCorpus")
    if not isinstance(ledger, BudgetLedger):
        raise TypeError(f"ledger must be a BudgetLedger; got {type(ledger).__name__}")
    if corpus.plan_sha256 != ledger.limits.plan_sha256:
        raise PlanChangedError(f"plan changed: the corpus belongs to plan_sha256 {corpus.plan_sha256}, the ledger "
                               f"to {ledger.limits.plan_sha256}")
    if ledger.limits.stage != CALIBRATION_STAGE:
        raise CalibrationError(f"calibration draws belong to the {CALIBRATION_STAGE!r} stage; the ledger is for "
                               f"stage {ledger.limits.stage!r}")
    budget = getattr(client, "budget", None)
    if not isinstance(budget, LedgerCallBudget) or budget.ledger is not ledger:
        raise CalibrationError("every calibration request must be admitted by the stage's ledger: build the client "
                               "with LedgerCallBudget(ledger) for this ledger; nothing was sent")
    config = getattr(client, "config", None)
    if not isinstance(config, ModelConfig):
        raise TypeError("client.config must be a ModelConfig")
    expected = request_allowance(config, usd_per_second=ledger.allowance.usd_per_second)
    if ledger.allowance != expected:
        raise CalibrationError(f"the ledger's request allowance {ledger.allowance.to_dict()} is not the client's real "
                               f"max_tokens and timeout {expected.to_dict()}; nothing was sent")
    if EstimateSource.TOP_K in sources:
        top = config.parameters.get("top_logprobs")
        if config.parameters.get("logprobs") is not True or not _is_int(top) or top < 1:
            raise CalibrationError("a top_k calibrator needs logprobs true and a positive top_logprobs in the "
                                   "generation settings; nothing was sent")


def _reusable(recorded: Iterable[CalibrationDraw], corpus: CalibrationCorpus, settings: DrawSettings,
              config: ModelConfig, sources: Sequence[EstimateSource]) -> dict[tuple, CalibrationDraw]:
    """Recorded draws of this corpus and these settings by key; any other recorded draw is refused.

    A draw recorded twice, as the generation pass left it and again with its
    full-distribution evidence settled, keeps the settled version; any other
    repetition is refused.
    """
    prompts = {(p.item_id, p.rendering): p for p in corpus.prompts}
    names = {source.value for source in sources}
    reused: dict[tuple, CalibrationDraw] = {}
    for draw in recorded:
        if not isinstance(draw, CalibrationDraw):
            raise TypeError(f"recorded draws must be CalibrationDraw values; got {type(draw).__name__}")
        prompt = prompts.get((draw.item_id, draw.rendering))
        problems = []
        if draw.kind != corpus.kind or prompt is None:
            problems.append(f"it is not a prompt of this {corpus.kind} corpus")
        elif (draw.label, draw.prompt_sha256) != (prompt.label, prompt.prompt_sha256):
            problems.append("its label or prompt differs from the corpus")
        if draw.draw_index >= settings.draws:
            problems.append(f"its draw_index is outside the {settings.draws} draws")
        if (draw.seed, draw.temperature) != (_draw_seed(settings, config, draw.draw_index),
                                             _draw_temperature(settings, config)):
            problems.append("its seed or temperature differs from the draw settings")
        settled = set(draw.features)
        expected = names - {_FULL} if draw.evidence_pending else names
        if settled != expected:
            problems.append(f"it carries the sources {sorted(settled)}, not {sorted(names)}")
        earlier = reused.get(draw.key)
        if earlier is not None and earlier != draw:
            if earlier.evidence_pending and not draw.evidence_pending and draw._pending_version() == earlier:
                pass  # the settled version of a pending draw replaces it
            elif draw.evidence_pending and not earlier.evidence_pending and earlier._pending_version() == draw:
                continue  # the pending version of a settled draw adds nothing
            else:
                problems.append("it is recorded more than once with different content")
        if problems:
            raise CalibrationError(f"recorded draw {draw.key} is refused, never merged: {'; '.join(problems)}")
        reused[draw.key] = draw
    return reused


def collect_draws(
    corpus: CalibrationCorpus, settings: DrawSettings, *, client: CompletionClient, ledger: BudgetLedger,
    sources: Iterable[EstimateSource | str], recorded: Iterable[CalibrationDraw] = (),
    on_draw: Callable[[CalibrationDraw], None] | None = None,
    on_request: Callable[[CorpusPrompt, int, int | None], None] | None = None,
) -> tuple[CalibrationDraw, ...]:
    """The generation pass: one recorded ensemble of ``settings.draws`` draws per corpus prompt, in corpus order.

    Call it inside ``with ledger:`` so that a limit, or any other exhausted
    budget, closes the stage as incomplete with the recorded draws kept. Before
    any request the sources, the ledger (plan, stage, allowance) and the client's
    budget are checked, and ``recorded`` draws are validated against the corpus
    and the settings. Each request carries the configured temperature and the
    draw's seed as overrides and is never retried here; a rejected credential or
    a closed ledger is raised, any other failure is recorded on its draw.
    ``on_request`` is called before each request is sent and ``on_draw``
    receives each new draw before the next request (the recording seam of the
    module docstring); a recorded draw, an interrupted one included, is never
    requested again. No full-distribution evidence is requested here: a parsed
    draw leaves that source pending for :func:`collect_evidence`.
    """
    if not isinstance(settings, DrawSettings):
        raise TypeError("settings must be DrawSettings")
    if not isinstance(ledger, BudgetLedger):
        raise TypeError(f"ledger must be a BudgetLedger; got {type(ledger).__name__}")
    sources = _arm_sources(ledger.limits.arm, sources)
    _check_request_path(corpus, client, ledger, sources)
    for hook, name in ((on_draw, "on_draw"), (on_request, "on_request")):
        if hook is not None and not callable(hook):
            raise TypeError(f"{name} must be callable or None")
    reused = _reusable(recorded, corpus, settings, client.config, sources)
    temperature = _draw_temperature(settings, client.config)
    overrides: dict[str, ParameterValue] = {} if settings.temperature is None else {"temperature": settings.temperature}
    draws: list[CalibrationDraw] = []
    for prompt in corpus.prompts:
        for index in range(settings.draws):
            key = (corpus.kind, prompt.item_id, prompt.rendering, index)
            if key in reused:
                draws.append(reused[key])
                continue
            parameters = dict(overrides)
            seed = _draw_seed(settings, client.config, index)
            if settings.seed_base is not None:
                parameters["seed"] = seed
            if on_request is not None:
                on_request(prompt, index, seed)
            response, failure = _request(client, prompt.prompt, parameters)
            common = {"kind": corpus.kind, "item_id": prompt.item_id, "rendering": prompt.rendering,
                      "label": prompt.label, "prompt_sha256": prompt.prompt_sha256, "draw_index": index,
                      "seed": seed, "temperature": temperature}
            empty = {source.value: None for source in sources}
            if response is None:
                draw = CalibrationDraw(**common, response_sha256=_sha256(""), response_text=None,
                                       failure_code=REQUEST_ERROR_PREFIX + failure, features=empty, evidence_ref=empty)
            else:
                try:
                    corpus.definition.parse_response(response.content)
                except DecisionParseError:
                    draw = CalibrationDraw(**common, response_sha256=_sha256(response.content), response_text=None,
                                           failure_code=PARSE_ERROR, features=empty, evidence_ref=empty)
                else:
                    rows = {source.value: _top_k_row(response) for source in sources
                            if source is EstimateSource.TOP_K}
                    draw = CalibrationDraw(**common, response_sha256=_sha256(response.content),
                                           response_text=response.content, failure_code=None,
                                           features={name: row for name, (row, _) in rows.items()},
                                           evidence_ref={name: ref for name, (_, ref) in rows.items()})
            if on_draw is not None:
                on_draw(draw)
            draws.append(draw)
    return tuple(draws)


def _evidence_batch(full_distribution: FullDistributionClient, prompt: CorpusPrompt, texts: list[str],
                    evidence: EvidenceSettings, sleep: Callable[[float], None]) -> tuple[FullDistributionEvidence, ...]:
    """One batch for one prompt and rendering; ``host_busy`` is retried after a wait, anything else raised."""
    for attempt in range(evidence.host_busy_retries + 1):
        try:
            batch = full_distribution.evidence_batch(prompt.prompt, texts, top_k_keep=evidence.top_k_keep,
                                                     timeout_s=evidence.timeout_s)
        except RequestError as error:
            where = f"{prompt.item_id} {prompt.rendering}"
            if error.code != HOST_BUSY:
                raise EvidenceUnavailableError(
                    f"the full-distribution host gave no evidence for {where} ({error.code or 'request_error'}); the "
                    "stage stays open with its recorded draws and nothing is fitted", code=error.code,
                    prompt_sha256=prompt.prompt_sha256) from error
            if attempt == evidence.host_busy_retries:
                raise EvidenceUnavailableError(
                    f"the full-distribution host stayed busy (host_busy) for {where} after "
                    f"{evidence.host_busy_retries} retries {evidence.host_busy_wait_s:g} s apart; the stage stays open "
                    "with its recorded draws and nothing is fitted", code=HOST_BUSY,
                    prompt_sha256=prompt.prompt_sha256) from error
            sleep(evidence.host_busy_wait_s)
            continue
        if not isinstance(batch, tuple | list) or len(batch) != len(texts) or not all(
                isinstance(item, FullDistributionEvidence) for item in batch):
            raise EvidenceUnavailableError(f"the evidence batch for {prompt.item_id} {prompt.rendering} does not hold "
                                           f"one evidence object per reply", code="invalid_evidence",
                                           prompt_sha256=prompt.prompt_sha256)
        return tuple(batch)
    raise AssertionError("unreachable")  # pragma: no cover - the loop returns or raises


def collect_evidence(
    corpus: CalibrationCorpus, draws: Iterable[CalibrationDraw], *, full_distribution: FullDistributionClient,
    evidence: EvidenceSettings = EvidenceSettings(), on_draw: Callable[[CalibrationDraw], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[CalibrationDraw, ...]:
    """The evidence pass: settle every pending full-distribution row, one batch per prompt and rendering.

    Run it after the generation pass, with the full-distribution host loaded
    alone and outside the ledger session (the plan's evidence rule). Per corpus
    prompt, one ``evidence_batch`` request carries the replies of its parsed
    draws that are still pending, in draw order, with ``evidence.timeout_s`` and
    ``evidence.top_k_keep``; each returned evidence object settles the draw at
    its position exactly as the per-draw path would
    (:func:`full_distribution_row`). Failed and unparsed draws get no evidence. A
    ``host_busy`` refusal is retried ``host_busy_retries`` times,
    ``host_busy_wait_s`` apart; any other failure, or the last refusal, raises
    :class:`EvidenceUnavailableError` with the draws settled so far already
    passed to ``on_draw``. Returns the draws in their given order.
    """
    if not isinstance(corpus, CalibrationCorpus):
        raise TypeError("corpus must be a CalibrationCorpus")
    if not isinstance(evidence, EvidenceSettings):
        raise TypeError("evidence must be EvidenceSettings")
    if not callable(getattr(full_distribution, "evidence_batch", None)):
        raise TypeError("the full-distribution client must offer evidence_batch (task 6.4)")
    if on_draw is not None and not callable(on_draw):
        raise TypeError("on_draw must be callable or None")
    draws = list(draws)
    prompts = {(p.item_id, p.rendering): p for p in corpus.prompts}
    pending: dict[tuple[str, str], list[int]] = {}
    for position, draw in enumerate(draws):
        if not isinstance(draw, CalibrationDraw):
            raise TypeError(f"draws must be CalibrationDraw values; got {type(draw).__name__}")
        prompt = prompts.get((draw.item_id, draw.rendering))
        if draw.kind != corpus.kind or prompt is None or draw.prompt_sha256 != prompt.prompt_sha256:
            raise CalibrationError(f"draw {draw.key} is not a draw of this {corpus.kind} corpus")
        if draw.evidence_pending:
            pending.setdefault((draw.item_id, draw.rendering), []).append(position)
    for prompt in corpus.prompts:
        positions = sorted(pending.get((prompt.item_id, prompt.rendering), ()), key=lambda at: draws[at].draw_index)
        if not positions:
            continue
        batch = _evidence_batch(full_distribution, prompt, [draws[at].response_text for at in positions], evidence,
                                sleep)
        for at, item in zip(positions, batch, strict=True):
            draws[at] = draws[at].settled(*full_distribution_row(draws[at].response_text, item))
            if on_draw is not None:
                on_draw(draws[at])
    return tuple(draws)


# ---------- the fit ----------

def _baseline(model: str, control: Sequence[MiaFeatures], min_valid: int) -> ControlBaseline:
    """The control baseline as ``mia.control.build_baseline`` aggregates its rows (no reference model)."""
    means: dict[str, float | None] = {}
    stds: dict[str, float | None] = {}
    for key in FEATURE_ORDER:
        if control:
            values = np.asarray([getattr(row, key) for row in control], dtype=np.float64)
            means[key] = float(np.mean(values))
            std = float(np.std(values))
            stds[key] = _STD_FLOOR if std < _STD_FLOOR else std
        else:
            means[key] = stds[key] = None
    means["ref_delta"] = stds["ref_delta"] = None
    return ControlBaseline(model=model, n_valid=len(control), feature_means=means, feature_stds=stds,
                           is_calibrated=len(control) >= min_valid, min_valid=min_valid)


def _vector(features: MiaFeatures, baseline: ControlBaseline) -> list[float]:
    standardised = standardise(features, baseline)
    return [float(standardised[key]) for key in FEATURE_ORDER]


def _fit_state(identity: CalibrationIdentity, baseline: ControlBaseline, mcs: MCSCalibrator) -> dict[str, Any]:
    """The scorer's calibration payload (``calibration_fingerprint``) for this fit and its bound identity."""
    classifier = mcs.classifier
    return {
        "scheme": FIT_SCHEME, "identity": identity.to_dict(), "observed_fit_settings": None,
        "baseline": asdict(baseline), "model": mcs.model, "feature_order": list(mcs.feature_order),
        "coefficients": classifier.coef_.tolist(), "intercept": classifier.intercept_.tolist(),
        "classes": classifier.classes_.tolist(), "holdout_auc": mcs.holdout_auc, "is_weak": mcs.is_weak,
        "valid_counts": {"baseline": baseline.n_valid, "is_memorized": mcs.n_valid_is, "oos_control": mcs.n_valid_oos},
        "holdout_labels": None if mcs.holdout_labels is None else list(mcs.holdout_labels),
        "holdout_probabilities": None if mcs.holdout_probabilities is None else list(mcs.holdout_probabilities),
    }


def _fingerprint(state: Mapping[str, Any]) -> str:
    """SHA-256 of the payload, encoded as ``MemoryGuardedScorer.calibration_fingerprint`` encodes it."""
    payload = json.dumps(state, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _rank_auc(memorized: np.ndarray, control: np.ndarray) -> float:
    """ROC-AUC as the Mann-Whitney statistic with average ranks for ties: the core ``rank_auc``."""
    return rank_auc(memorized, control)


def _clustered_interval(groups: Sequence[tuple[int, np.ndarray]]) -> dict[str, Any]:
    """Held-out AUC with the core prompt-clustered interval (``clustered_auc_interval``) in the record's fields.

    The held-out prompts are passed as clusters in the order of ``groups``, each with all its draws. With a
    class absent the record keeps its counts and has no AUC and no bounds; the core function is not called.
    """
    kept = [(label, scores) for label, scores in groups if label in (MEMORIZED, CONTROL)]
    positives = [scores for label, scores in kept if label == MEMORIZED]
    negatives = [scores for label, scores in kept if label == CONTROL]
    result: dict[str, Any] = {
        "auc": None, "ci_low": None, "ci_high": None, "confidence": INTERVAL_CONFIDENCE,
        "n_resamples": INTERVAL_RESAMPLES, "seed": INTERVAL_SEED, "method": CLUSTERED_METHOD,
        "n_memorized_prompts": len(positives), "n_control_prompts": len(negatives),
        "n_memorized": int(sum(len(scores) for scores in positives)),
        "n_control": int(sum(len(scores) for scores in negatives)),
    }
    if not positives or not negatives:
        return result
    interval = clustered_auc_interval(
        np.concatenate([scores for _, scores in kept]),
        np.concatenate([np.full(len(scores), label, dtype=np.int64) for label, scores in kept]),
        np.concatenate([np.full(len(scores), index, dtype=np.int64) for index, (_, scores) in enumerate(kept)]),
        n_resamples=INTERVAL_RESAMPLES, confidence=INTERVAL_CONFIDENCE, seed=INTERVAL_SEED)
    auc = _rank_auc(np.concatenate(positives), np.concatenate(negatives))
    return result | {"auc": auc, "ci_low": interval.lo, "ci_high": interval.hi}


@dataclass(frozen=True)
class CalibrationRecord:
    """One fitted calibrator with its plan, corpus, settings, split, counts and held-out intervals."""

    arm: str
    kind: str
    source: EstimateSource
    plan_sha256: str
    entries: tuple[CorpusEntry, ...]
    corpus_sha256: str
    calibrator: SourceCalibrator
    draw_settings: DrawSettings
    fit_settings: FitSettings
    split_seed: int
    holdout_prompts: tuple[str, ...]
    counts: Mapping[str, int]
    holdout_auc_interval: Mapping[str, Any] | None
    holdout_auc_interval_by_prompt: Mapping[str, Any]
    #: How the full-distribution evidence was requested; None for a top_k calibrator, which needs none.
    evidence_settings: EvidenceSettings | None = None

    def __post_init__(self) -> None:
        _arm(self.arm)
        _kind(self.kind)
        object.__setattr__(self, "source", _source(self.source))
        full = self.source is EstimateSource.FULL_DISTRIBUTION
        if full != isinstance(self.evidence_settings, EvidenceSettings) or (
                not full and self.evidence_settings is not None):
            raise CalibrationError("a full_distribution record carries its EvidenceSettings, a top_k record none")
        _hex(self.plan_sha256, "plan_sha256")
        entries = tuple(self.entries)
        if not all(isinstance(entry, CorpusEntry) for entry in entries):
            raise TypeError("entries must be CorpusEntry values")
        object.__setattr__(self, "entries", entries)
        if corpus_hash(entries) != self.corpus_sha256:
            raise CalibrationError("the record's corpus_sha256 is not the hash of its corpus entries")
        if not isinstance(self.calibrator, SourceCalibrator):
            raise TypeError("calibrator must be a SourceCalibrator")
        identity = self.calibrator.identity
        if identity.estimate_source != self.source.value or identity.corpus_sha256 != self.corpus_sha256:
            raise CalibrationError("the calibrator's identity is not bound to the record's estimate source and "
                                   "corpus_sha256")
        if not isinstance(self.draw_settings, DrawSettings) or not isinstance(self.fit_settings, FitSettings):
            raise TypeError("draw_settings and fit_settings must be DrawSettings and FitSettings")
        if not _is_int(self.split_seed) or not 0 <= self.split_seed < _SEED_MODULUS:
            raise CalibrationError(f"split_seed must be an integer in [0, 2**32); got {self.split_seed!r}")
        held = tuple(_hex(value, "holdout prompt") for value in self.holdout_prompts)
        known = {entry.prompt_sha256 for entry in entries}
        if len(set(held)) != len(held) or not set(held) <= known:
            raise CalibrationError("holdout_prompts must be distinct prompts of the corpus")
        object.__setattr__(self, "holdout_prompts", held)
        object.__setattr__(self, "counts", MappingProxyType(dict(self.counts)))

    def to_dict(self) -> dict[str, Any]:
        state = _fit_state(self.calibrator.identity, self.calibrator.baseline, self.calibrator.mcs)
        return _plain({
            "schema": CALIBRATOR_SCHEMA, "plan_sha256": self.plan_sha256, "arm": self.arm, "kind": self.kind,
            "source": self.source.value,
            "corpus": {"corpus_sha256": self.corpus_sha256, "entries": [entry.to_dict() for entry in self.entries]},
            "draw_settings": self.draw_settings.to_dict(), "fit_settings": self.fit_settings.to_dict(),
            "split_seed": self.split_seed, "holdout_prompts": list(self.holdout_prompts), "counts": dict(self.counts),
            "holdout_auc_interval": self.holdout_auc_interval,
            "holdout_auc_interval_by_prompt": self.holdout_auc_interval_by_prompt,
            "evidence_settings": None if self.evidence_settings is None else self.evidence_settings.to_dict(),
            "fit": state, "fingerprint": self.calibrator.fingerprint,
        })

    @classmethod
    def from_dict(cls, data: object) -> CalibrationRecord:
        """Rebuild a stored record; its fingerprint and corpus hash must match its content."""
        payload = _exact_keys(data, _RECORD_KEYS, "calibrator record")
        if payload["schema"] != CALIBRATOR_SCHEMA:
            raise CalibrationError(f"unsupported calibrator schema {payload['schema']!r}; expected {CALIBRATOR_SCHEMA!r}")
        state = _exact_keys(payload["fit"], _FIT_KEYS, "calibrator fit")
        if state["scheme"] != FIT_SCHEME:
            raise CalibrationError(f"unsupported fit scheme {state['scheme']!r}; expected {FIT_SCHEME!r}")
        fingerprint = _hex(payload["fingerprint"], "fingerprint")
        if _fingerprint(state) != fingerprint:
            raise CalibrationError("the fit differs from its recorded fingerprint; refused")
        corpus = _exact_keys(payload["corpus"], frozenset({"corpus_sha256", "entries"}), "calibrator corpus")
        entries = tuple(CorpusEntry(**_exact_keys(entry, frozenset({"item_id", "rendering", "label", "prompt_sha256"}),
                                                  "corpus entry")) for entry in corpus["entries"])
        if corpus_hash(entries) != corpus["corpus_sha256"]:
            raise CalibrationError("the corpus entries differ from the recorded corpus_sha256; refused")
        try:
            calibrator = _calibrator_from_state(state, fingerprint)
        except (KeyError, TypeError, ValueError) as exc:
            raise CalibrationError(f"the fit cannot be rebuilt: {exc}") from None
        return cls(
            arm=payload["arm"], kind=payload["kind"], source=payload["source"], plan_sha256=payload["plan_sha256"],
            entries=entries, corpus_sha256=corpus["corpus_sha256"], calibrator=calibrator,
            draw_settings=DrawSettings(**_exact_keys(payload["draw_settings"],
                                                     frozenset({"draws", "temperature", "seed_base"}), "draw settings")),
            fit_settings=FitSettings(**_exact_keys(payload["fit_settings"], frozenset({"min_auc", "min_valid"}),
                                                   "fit settings")),
            split_seed=payload["split_seed"], holdout_prompts=tuple(payload["holdout_prompts"]),
            counts=payload["counts"], holdout_auc_interval=payload["holdout_auc_interval"],
            holdout_auc_interval_by_prompt=payload["holdout_auc_interval_by_prompt"],
            evidence_settings=(None if payload["evidence_settings"] is None
                               else EvidenceSettings.from_dict(payload["evidence_settings"])),
        )


def _calibrator_from_state(state: Mapping[str, Any], fingerprint: str) -> SourceCalibrator:
    """The SourceCalibrator of a stored fit: the classifier rebuilt from its coefficients."""
    identity = CalibrationIdentity.from_dict(state["identity"])
    baseline = ControlBaseline(**state["baseline"])
    coefficients = np.asarray(state["coefficients"], dtype=np.float64)
    if coefficients.shape != (1, len(FEATURE_ORDER)) or list(state["feature_order"]) != list(FEATURE_ORDER):
        raise ValueError("the classifier must have one row of coefficients over the four features")
    classifier = LogisticRegression(class_weight="balanced", solver="liblinear")
    classifier.coef_ = coefficients
    classifier.intercept_ = np.asarray(state["intercept"], dtype=np.float64)
    classifier.classes_ = np.asarray(state["classes"], dtype=np.int64)
    classifier.n_features_in_ = coefficients.shape[1]
    counts = state["valid_counts"]
    labels, probabilities = state["holdout_labels"], state["holdout_probabilities"]
    mcs = MCSCalibrator(
        model=state["model"], classifier=classifier, feature_order=list(state["feature_order"]),
        holdout_auc=state["holdout_auc"], is_weak=state["is_weak"], n_valid_is=counts["is_memorized"],
        n_valid_oos=counts["oos_control"], holdout_labels=None if labels is None else tuple(labels),
        holdout_probabilities=None if probabilities is None else tuple(probabilities),
    )
    return SourceCalibrator(mcs, baseline, identity, fingerprint)


def _complete_draws(corpus: CalibrationCorpus, draws: Iterable[CalibrationDraw], settings: DrawSettings,
                    source: EstimateSource) -> list[list[CalibrationDraw]]:
    """Every draw of every corpus prompt, by prompt in corpus order and by draw index; anything else is refused."""
    index = {(p.item_id, p.rendering): position for position, p in enumerate(corpus.prompts)}
    by_prompt: list[dict[int, CalibrationDraw]] = [{} for _ in corpus.prompts]
    for draw in draws:
        if not isinstance(draw, CalibrationDraw):
            raise TypeError(f"draws must be CalibrationDraw values; got {type(draw).__name__}")
        position = index.get((draw.item_id, draw.rendering))
        if draw.kind != corpus.kind or position is None:
            raise CalibrationError(f"draw {draw.key} is not a draw of this {corpus.kind} corpus")
        prompt = corpus.prompts[position]
        if (draw.label, draw.prompt_sha256) != (prompt.label, prompt.prompt_sha256):
            raise CalibrationError(f"draw {draw.key} has another label or prompt than the corpus")
        if draw.draw_index >= settings.draws:
            raise CalibrationError(f"draw {draw.key} is outside the {settings.draws} draws per prompt")
        if settings.seed_base is not None and draw.seed != settings.seed_base + draw.draw_index:
            raise CalibrationError(f"draw {draw.key} has another seed than the draw settings")
        if settings.temperature is not None and draw.temperature != settings.temperature:
            raise CalibrationError(f"draw {draw.key} has another temperature than the draw settings")
        if source is EstimateSource.FULL_DISTRIBUTION and draw.evidence_pending:
            raise CalibrationError(f"draw {draw.key} still waits for its full-distribution evidence; run "
                                   "collect_evidence before fitting")
        if source.value not in draw.features:
            raise CalibrationError(f"draw {draw.key} carries no {source.value} features; that source was not drawn")
        if draw.draw_index in by_prompt[position]:
            raise CalibrationError(f"draw {draw.key} is given more than once")
        by_prompt[position][draw.draw_index] = draw
    present = sum(len(found) for found in by_prompt)
    planned = len(corpus.prompts) * settings.draws
    if present != planned:
        raise CalibrationError(f"the calibration draws are incomplete: {present} of {planned} recorded; resume the "
                               "stage before fitting")
    return [[found[i] for i in range(settings.draws)] for found in by_prompt]


def fit_calibrator(
    corpus: CalibrationCorpus, draws: Iterable[CalibrationDraw], source: EstimateSource | str, *,
    identity: CalibrationIdentity, arm: str, settings: DrawSettings, fit: FitSettings = FitSettings(),
    split_seed: int, evidence: EvidenceSettings | None = None,
) -> CalibrationRecord:
    """Fit the calibrator of one corpus and source from its recorded draws; no request is issued (4.1, 4.3).

    Every prompt must have all its draws, and for the full-distribution source
    every parsed draw its settled evidence. Each parsed draw with features of
    ``source`` is one row. The baseline comes from the control rows; the
    classifier is fitted on the training prompts' rows and scored on the
    held-out prompts' rows. ``evidence`` (the settings of the evidence pass) is
    recorded for the full-distribution source and must be given for it.
    Raises :class:`CalibrationError` for an incomplete or foreign draw set,
    pending evidence, an identity not bound to this corpus and source, too few
    control rows (``min_valid``) or too few usable prompts for the split.
    """
    if not isinstance(corpus, CalibrationCorpus):
        raise TypeError("corpus must be a CalibrationCorpus")
    source = _source(source)
    _arm(arm)
    if not isinstance(identity, CalibrationIdentity):
        raise TypeError("identity must be a CalibrationIdentity")
    if not isinstance(settings, DrawSettings) or not isinstance(fit, FitSettings):
        raise TypeError("settings and fit must be DrawSettings and FitSettings")
    if identity.estimate_source != source.value:
        raise CalibrationError(f"the identity's estimate_source {identity.estimate_source!r} is not the fitted source "
                               f"{source.value!r}")
    if identity.corpus_sha256 != corpus.corpus_sha256:
        raise CalibrationError("the identity's corpus_sha256 is not this corpus's hash")
    if not _is_int(split_seed) or not 0 <= split_seed < _SEED_MODULUS:
        raise CalibrationError(f"split_seed must be an integer in [0, 2**32); got {split_seed!r}")
    full = source is EstimateSource.FULL_DISTRIBUTION
    if full and not isinstance(evidence, EvidenceSettings):
        raise CalibrationError("a full_distribution fit records the EvidenceSettings of its evidence pass")
    by_prompt = _complete_draws(corpus, draws, settings, source)

    rows = [[row for row in (draw.feature_row(source) for draw in found if draw.failure_code is None)
             if row is not None] for found in by_prompt]
    control = [row for prompt, found in zip(corpus.prompts, rows, strict=True) if prompt.label == CONTROL
               for row in found]
    baseline = _baseline(identity.model, control, fit.min_valid)
    if not baseline.is_calibrated:
        raise CalibrationError(f"the {corpus.kind} {source.value} control baseline has {baseline.n_valid} usable "
                               f"control draws, fewer than min_valid={fit.min_valid}")
    usable = [position for position, found in enumerate(rows) if found]
    labels = np.asarray([corpus.prompts[position].label for position in usable], dtype=np.int64)
    per_class = {label: int(np.sum(labels == label)) for label in (MEMORIZED, CONTROL)}
    n_holdout = math.ceil(HOLDOUT_FRACTION * len(usable))
    if min(per_class.values()) < 2 or n_holdout < 2 or len(usable) - n_holdout < 2:
        raise CalibrationError(f"{len(usable)} prompts with usable draws ({per_class[MEMORIZED]} memorized, "
                               f"{per_class[CONTROL]} control) cannot support the stratified "
                               f"{HOLDOUT_FRACTION:.0%} holdout split by prompt")
    train_at, holdout_at = train_test_split(np.arange(len(usable)), test_size=HOLDOUT_FRACTION,
                                            random_state=split_seed, stratify=labels)
    if len(set(labels[train_at].tolist())) < 2 or len(set(labels[holdout_at].tolist())) < 2:
        raise CalibrationError("the stratified holdout split by prompt left one class out of the training or the "
                               f"held-out prompts ({per_class[MEMORIZED]} memorized, {per_class[CONTROL]} control "
                               "prompts with usable draws)")

    def stacked(at: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
        x, y, spans = [], [], []
        for position in (usable[int(i)] for i in at):
            start = len(x)
            x.extend(_vector(row, baseline) for row in rows[position])
            y.extend([corpus.prompts[position].label] * len(rows[position]))
            spans.append((position, start))
        return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.int64), spans

    x_train, y_train, _ = stacked(train_at)
    x_holdout, y_holdout, spans = stacked(holdout_at)
    classifier = LogisticRegression(class_weight="balanced", solver="liblinear", random_state=split_seed)
    classifier.fit(x_train, y_train)
    scores = classifier.predict_proba(x_holdout)[:, 1]
    holdout_auc = float(roc_auc_score(y_holdout, scores))
    n_valid_is = sum(len(found) for prompt, found in zip(corpus.prompts, rows, strict=True) if prompt.label == MEMORIZED)
    mcs = MCSCalibrator(
        model=identity.model, classifier=classifier, feature_order=list(FEATURE_ORDER), holdout_auc=holdout_auc,
        is_weak=holdout_auc < fit.min_auc, n_valid_is=n_valid_is, n_valid_oos=len(control),
        holdout_labels=tuple(int(label) for label in y_holdout),
        holdout_probabilities=tuple(float(value) for value in scores),
    )
    calibrator = SourceCalibrator(mcs, baseline, identity, _fingerprint(_fit_state(identity, baseline, mcs)))
    groups = [(corpus.prompts[position].label, scores[start:start + len(rows[position])])
              for position, start in spans]
    all_draws = [draw for found in by_prompt for draw in found]
    counts = {
        "prompts": len(corpus.prompts), "draws": len(all_draws),
        "failed_draws": sum(draw.failure_code is not None for draw in all_draws),
        "draws_without_features": sum(draw.failure_code is None and draw.features[source.value] is None
                                      for draw in all_draws),
        "valid_draws_memorized": n_valid_is, "valid_draws_control": len(control),
        "usable_prompts_memorized": per_class[MEMORIZED], "usable_prompts_control": per_class[CONTROL],
        "train_prompts": len(train_at), "holdout_prompts": len(holdout_at),
        "train_draws": len(y_train), "holdout_draws": len(y_holdout),
    }
    return CalibrationRecord(
        arm=arm, kind=corpus.kind, source=source, plan_sha256=corpus.plan_sha256, entries=corpus.entries,
        corpus_sha256=corpus.corpus_sha256, calibrator=calibrator, draw_settings=settings, fit_settings=fit,
        split_seed=split_seed, holdout_prompts=tuple(corpus.prompts[position].prompt_sha256 for position, _ in spans),
        counts=counts, holdout_auc_interval=calibrator.holdout_auc_interval(),
        holdout_auc_interval_by_prompt=_clustered_interval(groups), evidence_settings=evidence if full else None,
    )


# ---------- storage ----------

def calibrator_path(arm: str, source: EstimateSource | str, kind: str, root: str | Path = CALIBRATORS_DIR) -> Path:
    """``<root>/<arm>/<source>/<macro|edgar>.json``."""
    return Path(root) / _arm(arm) / _source(source).value / f"{_kind(kind)}.json"


def write_calibrator(record: CalibrationRecord, root: str | Path = CALIBRATORS_DIR) -> Path:
    """Store a record at its path; the same content is a no-op, replacing a different calibrator is refused."""
    if not isinstance(record, CalibrationRecord):
        raise TypeError(f"record must be a CalibrationRecord; got {type(record).__name__}")
    target = calibrator_path(record.arm, record.source, record.kind, root)
    payload = record.to_dict()
    if target.exists():
        if _read_json(target) == json.loads(json.dumps(payload, allow_nan=False)):
            return target
        raise CalibrationError(f"refusing to replace the different calibrator at {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)
    return target


def load_calibrator(path: str | Path, plan: StudyPlan, *, arm: str, kind: str, source: EstimateSource | str,
                    corpus: CalibrationCorpus | None = None) -> CalibrationRecord:
    """Load a stored calibrator for one arm, kind and source of the frozen plan, refusing any other.

    The record must carry the plan's hash (``PlanChangedError`` otherwise), name
    the requested arm, kind and source (so the macro calibrator is never applied
    to EDGAR prompts), list the corpus layout the frozen plan defines for that
    kind, and match its fingerprint. With ``corpus`` (rebuilt from the plan and
    the study items) its prompt hashes must match too.
    """
    record = CalibrationRecord.from_dict(_read_json(Path(path)))
    ensure_unchanged(plan, record.plan_sha256)
    source = _source(source)
    if record.kind != _kind(kind):
        raise CalibrationError(f"the {record.kind} calibrator is refused for {kind} prompts: the macro calibrator "
                               "is never applied to EDGAR prompts, nor the EDGAR calibrator to macro prompts "
                               f"(kind {record.kind!r}, requested {kind!r})")
    for name, stored, requested in (("arm", record.arm, _arm(arm)), ("source", record.source.value, source.value)):
        if stored != requested:
            raise CalibrationError(f"the calibrator was fitted for {name} {stored!r}, not {requested!r}")
    layout = tuple((entry.item_id, entry.rendering, entry.label) for entry in record.entries)
    if layout != plan_corpus_layout(plan, kind):
        raise CalibrationError(f"the calibrator's corpus is not the {kind} calibration set of the frozen plan")
    if corpus is not None:
        if not isinstance(corpus, CalibrationCorpus):
            raise TypeError("corpus must be a CalibrationCorpus or None")
        if (corpus.kind, corpus.plan_sha256, corpus.corpus_sha256) != (record.kind, record.plan_sha256,
                                                                      record.corpus_sha256):
            raise CalibrationError("the calibrator's corpus differs from the supplied corpus (kind, plan or "
                                   "corpus_sha256)")
    return record


# ---------- one calibration stage ----------

@dataclass(frozen=True)
class CalibrationRun:
    """The records of one calibration stage by ``(kind, source)``, their files and the draws by kind."""

    records: Mapping[tuple[str, str], CalibrationRecord]
    paths: Mapping[tuple[str, str], Path]
    draws: Mapping[str, tuple[CalibrationDraw, ...]]


def _draw_settings_by_kind(plan: StudyPlan, arm: str,
                           settings: DrawSettings | Mapping[str, DrawSettings] | None) -> dict[str, DrawSettings]:
    """The draw settings of each corpus: the frozen plan's when none are given, else those given (one or per kind)."""
    if settings is None:
        if plan.arms is None:
            raise CalibrationError("a plan without its 7.3 sections freezes no calibration draw settings; pass "
                                   "settings=DrawSettings(...)")
        return {kind: plan_draw_settings(plan, arm, kind) for kind in KINDS}
    if isinstance(settings, DrawSettings):
        return {kind: settings for kind in KINDS}
    if not isinstance(settings, Mapping) or set(settings) != set(KINDS):
        raise TypeError(f"settings must be DrawSettings, a mapping of {list(KINDS)} to DrawSettings, or None")
    for kind in KINDS:
        if not isinstance(settings[kind], DrawSettings):
            raise TypeError(f"settings[{kind!r}] must be DrawSettings")
    return {kind: settings[kind] for kind in KINDS}


def run_calibration(
    plan: StudyPlan, *, arm: str, macro_rows: Iterable[MacroRow], edgar_releases: Iterable[EdgarRelease],
    client: CompletionClient, ledger: BudgetLedger, sources: Iterable[EstimateSource | str],
    settings: DrawSettings | Mapping[str, DrawSettings] | None = None,
    full_distribution: FullDistributionClient | None = None, fit: FitSettings | None = None,
    evidence: EvidenceSettings | None = None, recorded: Iterable[CalibrationDraw] = (),
    on_draw: Callable[[CalibrationDraw], None] | None = None,
    on_request: Callable[[CorpusPrompt, int, int | None], None] | None = None,
    calibrators_dir: str | Path = CALIBRATORS_DIR, sleep: Callable[[float], None] = time.sleep,
) -> CalibrationRun:
    """Macro and EDGAR calibration of one arm: draw both corpora, fit one calibrator per kind and source, store them.

    Settings: without ``settings``, ``fit`` and ``evidence`` the frozen plan's
    apply (:func:`plan_draw_settings` per corpus, :func:`plan_fit_settings`,
    :func:`plan_evidence_settings`), which is how the study runs. A plan
    without its 7.3 sections freezes none: it needs ``settings``, and the fit
    and evidence take the module defaults. Settings that are passed (one
    :class:`DrawSettings` for both corpora, or one per corpus) are used as given
    and recorded in every calibrator file, so a deviation from the frozen plan
    stays visible.

    Before any request: the ledger must belong to ``plan`` and ``arm`` and to the
    calibration stage, both corpora must be the plan's frozen calibration sets,
    and the client and sources are checked as in :func:`collect_draws`.
    ``on_request`` and ``on_draw`` follow the recording seam of the module
    docstring.

    1. Generation pass: both corpora are drawn in one ``with ledger:`` session; a
       frozen limit raises :class:`~scripts.finance_sim.limits.AdmissionClosedError`
       with the stage marked incomplete, the recorded draws kept and nothing fitted.
    2. Evidence pass (full-distribution source only): after generation and
       outside the ledger session, :func:`collect_evidence` per corpus, with the
       full-distribution host loaded alone. :class:`EvidenceUnavailableError`
       leaves the stage open (not complete), the recorded draws kept and nothing
       fitted; a rerun with ``recorded`` sends no generation request and repeats
       only the missing evidence.
    3. The stage is marked complete, then each calibrator is fitted and written
       under ``calibrators_dir``.

    Hosts are never co-resident: the operator may run the generation pass alone
    (``with ledger: collect_draws(...)``), swap the generation host for the
    full-distribution host, and rerun this function with the recorded draws.
    ``on_draw`` receives each draw when it is made and again when its evidence
    settles; the recording keeps the latest version per draw.
    """
    if not isinstance(ledger, BudgetLedger):
        raise TypeError(f"ledger must be a BudgetLedger; got {type(ledger).__name__}")
    ensure_unchanged(plan, ledger.limits.plan_sha256)
    if _arm(arm) != ledger.limits.arm:
        raise CalibrationError(f"the ledger belongs to the {ledger.limits.arm} arm, not the {arm} arm")
    by_kind = _draw_settings_by_kind(plan, arm, settings)
    frozen = plan.arms is not None
    fit = (plan_fit_settings(plan, arm) if frozen else FitSettings()) if fit is None else fit
    if evidence is None:
        evidence = (plan_evidence_settings(plan, arm) if frozen else None) or EvidenceSettings()
    if not isinstance(fit, FitSettings):
        raise TypeError("fit must be FitSettings or None")
    corpora = {MACRO: macro_corpus(plan, macro_rows), EDGAR: edgar_corpus(plan, edgar_releases)}
    chosen = estimate_sources(arm, sources, full_distribution)
    for corpus in corpora.values():
        _check_request_path(corpus, client, ledger, chosen)
    previous: dict[str, list[CalibrationDraw]] = {MACRO: [], EDGAR: []}
    for draw in recorded:
        if not isinstance(draw, CalibrationDraw):
            raise TypeError(f"recorded draws must be CalibrationDraw values; got {type(draw).__name__}")
        previous[draw.kind].append(draw)
    for kind, corpus in corpora.items():
        _reusable(previous[kind], corpus, by_kind[kind], client.config, chosen)
    if not isinstance(evidence, EvidenceSettings):
        raise TypeError("evidence must be EvidenceSettings or None")
    with ledger:
        draws = {kind: collect_draws(corpus, by_kind[kind], client=client, ledger=ledger, sources=chosen,
                                     recorded=previous[kind], on_draw=on_draw, on_request=on_request)
                 for kind, corpus in corpora.items()}
    if EstimateSource.FULL_DISTRIBUTION in chosen:
        draws = {kind: collect_evidence(corpus, draws[kind], full_distribution=full_distribution, evidence=evidence,
                                        on_draw=on_draw, sleep=sleep)
                 for kind, corpus in corpora.items()}
    if ledger.status == OPEN:
        ledger.complete()
    seed = split_seed(plan)
    records = {}
    for kind, corpus in corpora.items():
        for source in chosen:
            identity = calibration_identity(client, by_kind[kind], source, corpus)
            records[(kind, source.value)] = fit_calibrator(corpus, draws[kind], source, identity=identity, arm=arm,
                                                           settings=by_kind[kind], fit=fit, split_seed=seed,
                                                           evidence=evidence)
    paths = {key: write_calibrator(record, calibrators_dir) for key, record in records.items()}
    return CalibrationRun(records=MappingProxyType(records), paths=MappingProxyType(paths),
                          draws=MappingProxyType(draws))


__all__ = [
    "ANONYMISED",
    "CALIBRATION_STAGE",
    "CALIBRATORS_DIR",
    "EDGAR",
    "FEATURE_ORDER",
    "IDENTIFYING",
    "INTERRUPTED",
    "KINDS",
    "MACRO",
    "OUTLOOK",
    "CalibrationCorpus",
    "CalibrationDraw",
    "CalibrationError",
    "CalibrationRecord",
    "CalibrationRun",
    "CorpusEntry",
    "CorpusPrompt",
    "DrawSettings",
    "EdgarRelease",
    "EvidenceSettings",
    "EvidenceUnavailableError",
    "FitSettings",
    "calibration_identity",
    "calibrator_path",
    "collect_draws",
    "collect_evidence",
    "full_distribution_row",
    "corpus_hash",
    "edgar_corpus",
    "estimate_sources",
    "fit_calibrator",
    "load_calibrator",
    "macro_corpus",
    "macro_item_id",
    "plan_corpus_layout",
    "plan_draw_settings",
    "plan_evidence_settings",
    "plan_fit_settings",
    "request_allowance",
    "run_calibration",
    "split_seed",
    "write_calibrator",
]
