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
import logging
import re
from collections import Counter
from dataclasses import dataclass

from sklearn.metrics import roc_auc_score

from src.core.bootstrap import bootstrap_ci
from src.core.loader import EvalRow, EvalSet
from src.core.nvidia_lm import NvidiaLM, TokenLogprob
from src.mia.control import ControlBaseline, standardise
from src.mia.features import MiaFeatures, compute_mia_features
from src.mia.mcs import MCSCalibrator

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

# --- Strict response parsers --------------------------------------------------

# Per-line parser; ``re.MULTILINE`` so ``^`` / ``$`` anchor each line and the
# parser refuses freeform text wrapped around a Direction token.
_DIRECTION_RE = re.compile(r"^\s*Direction:\s*(-?\d+)\s*$", re.MULTILINE)
_CONFIDENCE_RE = re.compile(r"^\s*Confidence:\s*([0-9]*\.?[0-9]+)\s*$", re.MULTILINE)
_VALID_DIRECTIONS: frozenset[int] = frozenset({-1, 0, 1})


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


def _parse_direction(content: str) -> int | None:
    match = _DIRECTION_RE.search(content)
    if match is None:
        return None
    try:
        value = int(match.group(1))
    except ValueError:
        return None
    if value not in _VALID_DIRECTIONS:
        return None
    return value


def _parse_confidence(content: str) -> float | None:
    match = _CONFIDENCE_RE.search(content)
    if match is None:
        return None
    try:
        value = float(match.group(1))
    except ValueError:
        return None
    if not (0.0 <= value <= 1.0):
        return None
    return value


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
    )


def _safe_ref_logprobs(
    ref_lm: NvidiaLM | None, prompt: str
) -> list[TokenLogprob] | None:
    """Best-effort reference-model call. Returns ``None`` on any failure."""
    if ref_lm is None:
        return None
    try:
        ref_result = ref_lm.generate(prompt)
    except (TimeoutError, RuntimeError) as exc:  # ref failure does NOT abort row
        logger.warning(
            "evaluator: reference model %s failed on prompt: %s",
            getattr(ref_lm, "model", "<unknown>"),
            exc,
        )
        return None
    return ref_result.logprobs


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


def evaluate_model(
    model_lm: NvidiaLM,
    eval_set: EvalSet,
    baseline: ControlBaseline,
    mcs: MCSCalibrator,
    ref_lm: NvidiaLM | None,
    holdout_records: list[Record] | None = None,
    bootstrap_n: int = 1000,
    seed: int = 0,
) -> ModelEvalResult:
    """Score one model against ``eval_set`` and assemble a ``ModelEvalResult``.

    See module docstring for the row-level pipeline. The function performs no
    I/O beyond the model HTTP calls; all artifact writing is owned by
    ``harness.report`` and ``harness.runner``.
    """
    model_id = getattr(model_lm, "model", "<unknown>")
    records: list[Record] = []
    temperature_violated = False

    for row in eval_set.rows:
        prompt_hash = _hash_prompt(row.prompt)
        # 1. Primary LM call ---------------------------------------------------
        try:
            primary = model_lm.generate(row.prompt)
        except TimeoutError:
            records.append(
                _failure_record(model_id, prompt_hash, row.target_direction, FAIL_TIMEOUT)
            )
            continue
        except RuntimeError as exc:
            records.append(
                _failure_record(
                    model_id,
                    prompt_hash,
                    row.target_direction,
                    _classify_runtime_error(exc),
                )
            )
            continue
        except Exception:  # pragma: no cover - defensive, mirrors smoke gate
            records.append(
                _failure_record(model_id, prompt_hash, row.target_direction, FAIL_ERROR)
            )
            continue

        if not _temperature_honoured(primary.raw_temperature_observed):
            temperature_violated = True

        # 2. Strict parse ------------------------------------------------------
        direction = _parse_direction(primary.content)
        confidence = _parse_confidence(primary.content)
        if direction is None or confidence is None:
            records.append(
                _failure_record(model_id, prompt_hash, row.target_direction, FAIL_PARSE)
            )
            continue

        # 3. Reference-model call (best-effort) --------------------------------
        ref_logprobs = _safe_ref_logprobs(ref_lm, row.prompt)

        # 4. MIA features + MCS penalty ---------------------------------------
        try:
            features = compute_mia_features(primary.content, primary.logprobs, ref_logprobs)
        except (ValueError, RuntimeError):
            # MIA computation failed (e.g., empty top_logprobs at some position).
            # Per Error Strategy → "Per-row crash inside MIA computation": treat
            # as a parse failure with the generic error label so the row is
            # excluded from accuracy and the runner can keep going.
            records.append(
                _failure_record(model_id, prompt_hash, row.target_direction, FAIL_ERROR)
            )
            continue

        standardised = standardise(features, baseline)
        try:
            p_memorized = float(mcs.predict_proba(features, baseline))
        except ValueError:
            # MCS could not score this row (e.g., baseline missing a feature
            # the calibrator expects). Fall through to a generic error rather
            # than misclassifying as a parse failure.
            records.append(
                _failure_record(model_id, prompt_hash, row.target_direction, FAIL_ERROR)
            )
            continue
        penalized_confidence = float(confidence) * (1.0 - p_memorized)

        records.append(
            Record(
                model=model_id,
                prompt_hash=prompt_hash,
                parse_ok=True,
                predicted_direction=direction,
                raw_confidence=float(confidence),
                penalized_confidence=penalized_confidence,
                target_direction=row.target_direction,
                features_raw=features,
                features_standardised=standardised,
                p_memorized=p_memorized,
                fail_reason=None,
            )
        )

    # Aggregate -----------------------------------------------------------------
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
