"""Per-model evaluator: from raw eval set to ``ModelEvalResult`` with bootstrap CIs.

Implements the ``harness.evaluator`` component from the honest-model-ranking
design (see design.md → Components and Interfaces → harness.evaluator).
Satisfies Requirements 3.3, 4.3, 5.4, 6.1, 6.2, 6.3, 7.1, 7.2, 7.3, and 10.3.

Pipeline per eval row:
1. Call ``model_lm.generate(prompt)``. Map ``TimeoutError`` to
   ``fail_reason="timeout"`` and ``RuntimeError`` to either
   ``"no_logprobs"`` (message mentions ``logprobs`` / ``top_logprobs``) or the
   generic ``"error"`` bucket. Any other exception also falls into ``"error"``.
2. On success, optionally call ``ref_lm.generate(prompt)`` to obtain reference
   logprobs. A reference-side failure does not invalidate the row — it merely
   sets ``ref_logprobs=None``.
3. Parse ``Direction:`` strictly (only ``-1, 0, 1``) and ``Confidence:`` strictly
   (must be a float in ``[0, 1]``). Either parse failure → ``parse_ok=False``,
   ``fail_reason="parse_failure"``.
4. Compute MIA features, standardise against the per-model baseline, run
   ``mcs.predict_proba`` to get ``p_memorized``, and apply the continuous
   penalty ``penalized_confidence = raw_confidence * (1 - p_memorized)``
   (Req 5.4 — no thresholds).

Bootstrap CIs (Req 6.1, 6.3) are computed via ``core.bootstrap.bootstrap_ci``
over parse-OK rows for accuracy and over ``(p_memorized, label)`` pairs from
the supplied ``holdout_records`` for MCS-AUC. When no holdout records are
available, the CI collapses to the calibrator's ``holdout_auc`` point estimate
(documented fallback in design.md → Implementation Notes).

The temperature-not-honoured warning (Req 10.3) is surfaced when any LM call
returns a non-None, non-zero ``raw_temperature_observed``. All other warnings
(weak-calibration, parse-unreliable, not-better-than-baseline, uncalibrated)
are emitted by the ranker, not here.
"""

from __future__ import annotations

import hashlib
import json as _json
import logging
import re
from collections import Counter
from dataclasses import dataclass

from sklearn.metrics import roc_auc_score

from recall_guard.core.bootstrap import bootstrap_ci
from recall_guard.core.loader import EvalRow, EvalSet
from recall_guard.core.nvidia_lm import NvidiaLM, generate_many
from recall_guard.mia.control import ControlBaseline, standardise
from recall_guard.mia.features import MiaFeatures, compute_mia_features
from recall_guard.mia.mcs import MCSCalibrator

logger = logging.getLogger(__name__)


# --- Fail-reason vocabulary --------------------------------------------------

FAIL_TIMEOUT = "timeout"
FAIL_NO_LOGPROBS = "no_logprobs"
FAIL_PARSE = "parse_failure"
FAIL_ERROR = "error"

# Substring markers used to distinguish "missing top_logprobs" RuntimeErrors
# from generic ones (consistent with harness.smoke).
_NO_LOGPROBS_MARKERS: tuple[str, ...] = ("logprobs", "top_logprobs")

WARNING_TEMPERATURE_NOT_HONOURED = "temperature-not-honoured"

# --- Response parsers (layered coercion) -------------------------------------
#
# We coerce on the FIRST call's raw text only — never re-issue the LM. A
# retry would change the logprob trace and contaminate the very signal MCS
# is trying to learn. That rules out DSPy-style reprompt loops; instead, we
# layer cheap post-hoc strategies in order of strictness.
#
# Layered strategies, applied in order (Direction):
#   1. Markdown-tolerant regex on a "Direction: N" line.
#   2. JSON object containing a "direction" key.
#   3. Last-occurrence regex with a permissive 30-char gap (handles
#      "**Final answer — Direction = 1**" variants).
#   4. Word coercion: "higher" / "rose" / "up" → 1, etc.
# Confidence has the same first three layers plus a percentage fallback
# ("Confidence: 65%" → 0.65). We never invent values: if no layer fires we
# return None and the row is marked parse_failure.
_DIRECTION_RE = re.compile(
    r"\bDirection\b[\s\*_:]*([+-]?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_DIRECTION_RE_LOOSE = re.compile(
    r"\bDirection\b[^\d\n]{0,30}([+-]?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_CONFIDENCE_RE = re.compile(
    r"\bConfidence\b[\s\*_:]*([0-9]*\.?[0-9]+)",
    re.IGNORECASE,
)
_CONFIDENCE_RE_LOOSE = re.compile(
    r"\bConfidence\b[^\d\n]{0,30}([0-9]*\.?[0-9]+)",
    re.IGNORECASE,
)
_PERCENT_RE = re.compile(r"\bConfidence\b[^\d\n]{0,30}([0-9]+(?:\.[0-9]+)?)\s*%", re.IGNORECASE)
_JSON_OBJECT_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)
_VALID_DIRECTIONS: frozenset[int] = frozenset({-1, 0, 1})

_DIRECTION_WORDS_UP = ("higher", "rose ", "rises", "rising", "gained", "gains ", "up ", "bullish", "increase", "advanced", "positive close")
_DIRECTION_WORDS_DOWN = ("lower", "fell ", "falls", "falling", "declined", "decline", "down ", "bearish", "decrease", "negative close")
_DIRECTION_WORDS_FLAT = ("unchanged", "flat ", "no change", "no movement", "even close")

#: How many characters of failed-parse responses to retain in the record
#: for diagnostics. Long enough to see the model's intent, short enough to
#: keep records.jsonl readable.
_RAW_EXCERPT_CHARS = 400


# --- Public dataclasses -------------------------------------------------------


@dataclass(frozen=True)
class Record:
    """Per-(model, prompt) record produced by ``evaluate_model``.

    Attributes
    ----------
    model:
        NVIDIA model ID this record was scored for.
    prompt_hash:
        First 16 hex chars of ``sha256(prompt)`` — keeps records.jsonl readable
        without leaking full prompts. Uniquely keys the record together with
        ``model`` (design § Data Models).
    parse_ok:
        ``True`` when ``Direction:`` and ``Confidence:`` both parsed strictly
        and MIA features were computable.
    predicted_direction:
        Parsed integer in ``{-1, 0, 1}``; ``None`` when ``parse_ok`` is False.
    raw_confidence:
        Parsed confidence in ``[0, 1]``; ``None`` on parse failure.
    penalized_confidence:
        ``raw_confidence * (1 - p_memorized)`` (Req 5.4); ``None`` on failure.
    target_direction:
        Ground-truth direction copied from the eval row (always populated).
    features_raw:
        :class:`MiaFeatures` instance; ``None`` when the LM call failed before
        feature computation could run.
    features_standardised:
        Per-feature standardised values (z-score against the model's baseline);
        ``None`` whenever ``features_raw`` is ``None``.
    p_memorized:
        ``p(memorized | features) ∈ [0, 1]``; ``None`` when ``features_raw``
        is ``None``.
    fail_reason:
        One of ``"timeout"`` / ``"no_logprobs"`` / ``"parse_failure"`` /
        ``"error"`` on failure; ``None`` on success.
    raw_response_excerpt:
        First ~400 chars of the raw model response when the row failed to
        parse. Always ``None`` for parse-OK rows (no need to bloat the
        artifact). Use this to inspect *why* a parse failed without re-running
        the model.
    """

    model: str
    prompt_hash: str
    parse_ok: bool
    predicted_direction: int | None
    raw_confidence: float | None
    penalized_confidence: float | None
    target_direction: int
    features_raw: MiaFeatures | None
    features_standardised: dict[str, float | None] | None
    p_memorized: float | None
    fail_reason: str | None
    raw_response_excerpt: str | None = None


@dataclass(frozen=True)
class CIBound:
    """A bootstrap point estimate plus 95% percentile bounds."""

    point: float
    lo: float
    hi: float


@dataclass(frozen=True)
class ModelEvalResult:
    """Aggregate evaluation result for one model on the eval set.

    Attributes
    ----------
    model:
        NVIDIA model ID.
    raw_accuracy:
        Bootstrap CI on ``predicted_direction == target_direction`` over
        parse-OK rows (Req 6.1, 7.3).
    memguard_accuracy:
        Same accuracy denominator as ``raw_accuracy`` — in this spec the
        MemGuard penalty discounts confidence but does not change the
        predicted direction, so the two CIs coincide. The field exists so a
        future confidence-thresholded variant can diverge without breaking
        downstream consumers.
    mcs_auc:
        Bootstrap CI over the ``(p_memorized, label)`` pairs in
        ``holdout_records``. Falls back to ``mcs.holdout_auc`` (point only)
        when no holdout records are supplied (design § Implementation Notes).
    parse_success_rate:
        Fraction of rows with ``parse_ok=True`` (Req 7.2). ``1.0`` for empty
        eval sets (vacuously true).
    parse_failures:
        Count of rows with ``parse_ok=False`` (Req 7.1).
    warnings:
        Subset of ``{"temperature-not-honoured"}``. Other warnings
        (weak-calibration, parse-unreliable, not-better-than-baseline,
        uncalibrated) are added by the ranker.
    records:
        Per-row records in eval-set order.
    """

    model: str
    raw_accuracy: CIBound
    memguard_accuracy: CIBound
    mcs_auc: CIBound
    parse_success_rate: float
    parse_failures: int
    warnings: list[str]
    records: list[Record]


# --- Internal helpers ---------------------------------------------------------


def _hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def _classify_runtime_error(exc: RuntimeError) -> str:
    """Map a RuntimeError raised by ``NvidiaLM`` to a fail_reason label."""
    message = str(exc).lower()
    if any(marker in message for marker in _NO_LOGPROBS_MARKERS):
        return FAIL_NO_LOGPROBS
    return FAIL_ERROR


def _coerce_direction_int(raw: object) -> int | None:
    """Best-effort coercion of any value to a Direction in {-1, 0, 1}."""
    try:
        # Accept "1", "1.0", 1, 1.0, "+1", "-1.5" → -1.
        value = int(float(str(raw)))
    except (TypeError, ValueError):
        return None
    if value not in _VALID_DIRECTIONS:
        return None
    return value


def _coerce_confidence_float(raw: object) -> float | None:
    """Best-effort coercion of any value to a Confidence in [0, 1]."""
    try:
        value = float(str(raw))
    except (TypeError, ValueError):
        return None
    if not (0.0 <= value <= 1.0):
        return None
    return value


def _try_extract_json(content: str) -> dict | None:
    """Find the first balanced ``{...}`` block in content, parse as JSON dict."""
    for match in _JSON_OBJECT_RE.finditer(content):
        try:
            obj = _json.loads(match.group(0))
        except _json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _direction_from_words(content: str) -> int | None:
    """Last-resort: scan the trailing 600 chars of the response for a verbal answer."""
    tail = content[-600:].lower()
    last_up = max((tail.rfind(w) for w in _DIRECTION_WORDS_UP), default=-1)
    last_down = max((tail.rfind(w) for w in _DIRECTION_WORDS_DOWN), default=-1)
    last_flat = max((tail.rfind(w) for w in _DIRECTION_WORDS_FLAT), default=-1)
    pos = max(last_up, last_down, last_flat)
    if pos < 0:
        return None
    if pos == last_up:
        return 1
    if pos == last_down:
        return -1
    return 0


def _parse_direction(content: str) -> int | None:
    """Layered direction parser. See module-level comment for the strategy order."""
    # Layer 1: strict markdown-tolerant regex (use the LAST match — models
    # often restate "Direction" earlier in their reasoning).
    matches = _DIRECTION_RE.findall(content)
    if matches:
        v = _coerce_direction_int(matches[-1])
        if v is not None:
            return v
    # Layer 2: JSON object with a "direction" key.
    obj = _try_extract_json(content)
    if obj is not None:
        for key in ("direction", "Direction", "DIRECTION"):
            if key in obj:
                v = _coerce_direction_int(obj[key])
                if v is not None:
                    return v
    # Layer 3: looser gap regex (also use last match).
    matches = _DIRECTION_RE_LOOSE.findall(content)
    if matches:
        v = _coerce_direction_int(matches[-1])
        if v is not None:
            return v
    # Layer 4: word coercion on the response tail.
    return _direction_from_words(content)


def _parse_confidence(content: str) -> float | None:
    """Layered confidence parser. Returns a float in [0, 1] or None."""
    # Layer 1: strict regex.
    matches = _CONFIDENCE_RE.findall(content)
    if matches:
        v = _coerce_confidence_float(matches[-1])
        if v is not None:
            return v
    # Layer 2: JSON.
    obj = _try_extract_json(content)
    if obj is not None:
        for key in ("confidence", "Confidence", "CONFIDENCE"):
            if key in obj:
                v = _coerce_confidence_float(obj[key])
                if v is not None:
                    return v
    # Layer 3: looser gap regex.
    matches = _CONFIDENCE_RE_LOOSE.findall(content)
    if matches:
        v = _coerce_confidence_float(matches[-1])
        if v is not None:
            return v
    # Layer 4: percentage form ("Confidence: 65%" → 0.65).
    pct_matches = _PERCENT_RE.findall(content)
    if pct_matches:
        try:
            pct = float(pct_matches[-1]) / 100.0
        except ValueError:
            return None
        if 0.0 <= pct <= 1.0:
            return pct
    return None


def _temperature_honoured(observed: float | None) -> bool:
    """An ``observed`` value of ``None`` (API silent) or exactly 0.0 is honoured."""
    if observed is None:
        return True
    return float(observed) == 0.0


def _failure_record(
    model: str,
    prompt_hash: str,
    target: int,
    fail_reason: str,
    raw_excerpt: str | None = None,
) -> Record:
    return Record(
        model=model,
        prompt_hash=prompt_hash,
        parse_ok=False,
        predicted_direction=None,
        raw_confidence=None,
        penalized_confidence=None,
        target_direction=target,
        features_raw=None,
        features_standardised=None,
        p_memorized=None,
        fail_reason=fail_reason,
        raw_response_excerpt=raw_excerpt,
    )


def _accuracy_statistic(records: list[Record]) -> float:
    """Mean of ``(predicted_direction == target_direction)`` over a record list."""
    if not records:
        # bootstrap_ci would never call us with an empty resample, but be
        # defensive: a zero-length statistic is undefined.
        raise ValueError("accuracy statistic called on empty record list")
    correct = sum(
        1
        for r in records
        if r.predicted_direction is not None
        and r.predicted_direction == r.target_direction
    )
    return correct / len(records)


def _auc_statistic(samples: list[tuple[float, int]]) -> float:
    """ROC-AUC over (p_memorized, label) pairs.

    Raises ``ValueError`` when the resample is single-class — bootstrap_ci
    drops such resamples and warns once per call (see design § Implementation
    Notes for the AUC bootstrap fallback rationale).
    """
    if not samples:
        raise ValueError("auc statistic called on empty samples")
    labels = [s[1] for s in samples]
    if len(set(labels)) < 2:
        raise ValueError("single-class resample")
    scores = [s[0] for s in samples]
    return float(roc_auc_score(labels, scores))


def _accuracy_ci(parse_ok_records: list[Record], n: int, seed: int) -> CIBound:
    if not parse_ok_records:
        return CIBound(point=0.0, lo=0.0, hi=0.0)
    point, lo, hi = bootstrap_ci(
        samples=parse_ok_records,
        statistic=_accuracy_statistic,
        n_resamples=n,
        seed=seed,
    )
    return CIBound(point=point, lo=lo, hi=hi)


def _mcs_auc_ci(
    holdout_records: list[Record] | None,
    mcs: MCSCalibrator,
    n: int,
    seed: int,
) -> CIBound:
    """Bootstrap MCS-AUC over holdout (p_memorized, label) pairs.

    Falls back to the calibrator's ``holdout_auc`` point estimate (per
    design.md Implementation Notes) when ``holdout_records`` is None / empty
    or otherwise cannot supply both classes.
    """
    fallback = float(getattr(mcs, "holdout_auc", 0.0))
    if not holdout_records:
        return CIBound(point=fallback, lo=fallback, hi=fallback)

    samples: list[tuple[float, int]] = []
    for r in holdout_records:
        if r.p_memorized is None:
            continue
        # The holdout label is encoded in target_direction per the design's
        # IS/OOS calibration corpus convention (label=1 IS-memorized, 0 OOS).
        samples.append((float(r.p_memorized), int(r.target_direction)))

    if not samples or len({s[1] for s in samples}) < 2:
        return CIBound(point=fallback, lo=fallback, hi=fallback)

    point, lo, hi = bootstrap_ci(
        samples=samples,
        statistic=_auc_statistic,
        n_resamples=n,
        seed=seed,
    )
    return CIBound(point=point, lo=lo, hi=hi)


# --- Public API ---------------------------------------------------------------


def _score_row(
    *,
    model_id: str,
    row: EvalRow,
    primary,
    ref_res,
    ref_lm: NvidiaLM | None,
    baseline: ControlBaseline,
    mcs: MCSCalibrator,
) -> tuple[Record, bool]:
    """Run the per-row pipeline. Returns (record, temperature_violated)."""
    prompt_hash = _hash_prompt(row.prompt)

    if isinstance(primary, TimeoutError):
        return _failure_record(model_id, prompt_hash, row.target_direction, FAIL_TIMEOUT), False
    if isinstance(primary, RuntimeError):
        return (
            _failure_record(
                model_id, prompt_hash, row.target_direction,
                _classify_runtime_error(primary),
            ),
            False,
        )
    if isinstance(primary, Exception):  # pragma: no cover - defensive
        return _failure_record(model_id, prompt_hash, row.target_direction, FAIL_ERROR), False

    temp_violated = not _temperature_honoured(primary.raw_temperature_observed)
    excerpt = (primary.content or "")[:_RAW_EXCERPT_CHARS]

    direction = _parse_direction(primary.content)
    confidence = _parse_confidence(primary.content)
    if direction is None or confidence is None:
        return (
            _failure_record(model_id, prompt_hash, row.target_direction,
                            FAIL_PARSE, raw_excerpt=excerpt),
            temp_violated,
        )

    if ref_lm is None or isinstance(ref_res, Exception) or ref_res is None:
        ref_logprobs = None
    else:
        ref_logprobs = ref_res.logprobs

    try:
        features = compute_mia_features(primary.content, primary.logprobs, ref_logprobs)
    except (ValueError, RuntimeError):
        return (
            _failure_record(model_id, prompt_hash, row.target_direction,
                            FAIL_ERROR, raw_excerpt=excerpt),
            temp_violated,
        )

    standardised = standardise(features, baseline)
    try:
        p_memorized = float(mcs.predict_proba(features, baseline))
    except ValueError:
        return (
            _failure_record(model_id, prompt_hash, row.target_direction,
                            FAIL_ERROR, raw_excerpt=excerpt),
            temp_violated,
        )

    record = Record(
        model=model_id,
        prompt_hash=prompt_hash,
        parse_ok=True,
        predicted_direction=direction,
        raw_confidence=float(confidence),
        penalized_confidence=float(confidence) * (1.0 - p_memorized),
        target_direction=row.target_direction,
        features_raw=features,
        features_standardised=standardised,
        p_memorized=p_memorized,
        fail_reason=None,
    )
    return record, temp_violated


def evaluate_model(
    model_lm: NvidiaLM,
    eval_set: EvalSet,
    baseline: ControlBaseline,
    mcs: MCSCalibrator,
    ref_lm: NvidiaLM | None,
    holdout_records: list[Record] | None = None,
    bootstrap_n: int = 1000,
    seed: int = 0,
    max_workers: int = 1,
) -> ModelEvalResult:
    """Score one model against ``eval_set`` and assemble a ``ModelEvalResult``.

    With ``max_workers > 1`` the per-row primary + reference LM calls
    fan out via ``concurrent.futures.ThreadPoolExecutor`` (results are
    paired with rows by index, so order is preserved). Post-processing
    — parsing, MIA feature compute, MCS scoring — runs serially.

    See module docstring for the row-level pipeline. The function performs no
    I/O beyond the model HTTP calls; all artifact writing is owned by
    ``harness.report`` and ``harness.runner``.
    """
    model_id = getattr(model_lm, "model", "<unknown>")

    prompts = [row.prompt for row in eval_set.rows]
    primary_results = generate_many(model_lm, prompts, max_workers=max_workers)
    ref_results: list = (
        generate_many(ref_lm, prompts, max_workers=max_workers)
        if ref_lm is not None else [None] * len(prompts)
    )

    records: list[Record] = []
    temperature_violated = False
    for row, primary, ref_res in zip(eval_set.rows, primary_results, ref_results, strict=True):
        record, row_temp_violated = _score_row(
            model_id=model_id, row=row, primary=primary, ref_res=ref_res,
            ref_lm=ref_lm, baseline=baseline, mcs=mcs,
        )
        records.append(record)
        if row_temp_violated:
            temperature_violated = True

    n_rows = len(records)
    parse_failures = sum(1 for r in records if not r.parse_ok)
    parse_ok_records = [r for r in records if r.parse_ok]
    parse_success_rate = (
        1.0 if n_rows == 0 else (n_rows - parse_failures) / n_rows
    )

    raw_accuracy = _accuracy_ci(parse_ok_records, n=bootstrap_n, seed=seed)
    # MemGuard accuracy uses the same parse-OK denominator. The penalty
    # affects confidence only — predicted_direction is unchanged in this
    # spec — so the bootstrap statistic is identical. The dataclass field is
    # kept distinct so a future confidence-threshold variant may diverge
    # without breaking the report schema.
    memguard_accuracy = _accuracy_ci(parse_ok_records, n=bootstrap_n, seed=seed)
    mcs_auc = _mcs_auc_ci(holdout_records, mcs, n=bootstrap_n, seed=seed)

    warnings: list[str] = []
    if temperature_violated:
        warnings.append(WARNING_TEMPERATURE_NOT_HONOURED)

    return ModelEvalResult(
        model=model_id,
        raw_accuracy=raw_accuracy,
        memguard_accuracy=memguard_accuracy,
        mcs_auc=mcs_auc,
        parse_success_rate=parse_success_rate,
        parse_failures=parse_failures,
        warnings=warnings,
        records=records,
    )


def compute_majority_baseline(
    eval_set: EvalSet,
    bootstrap_n: int = 1000,
    seed: int = 0,
) -> CIBound:
    """Bootstrap CI on the majority-class baseline accuracy (Req 6.2).

    Returns ``CIBound(0.0, 0.0, 0.0)`` for an empty eval set.
    """
    rows = eval_set.rows
    if not rows:
        return CIBound(0.0, 0.0, 0.0)

    counts = Counter(r.target_direction for r in rows)
    majority_class, _ = counts.most_common(1)[0]
    indicators: list[int] = [1 if r.target_direction == majority_class else 0 for r in rows]

    def _mean(samples: list[int]) -> float:
        return sum(samples) / len(samples)

    point, lo, hi = bootstrap_ci(
        samples=indicators,
        statistic=_mean,
        n_resamples=bootstrap_n,
        seed=seed,
    )
    return CIBound(point=point, lo=lo, hi=hi)


__all__ = [
    "CIBound",
    "ModelEvalResult",
    "Record",
    "compute_majority_baseline",
    "evaluate_model",
]
