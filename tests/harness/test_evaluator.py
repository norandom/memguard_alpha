"""Tests for harness.evaluator.

Covers Requirements 3.3, 4.3, 5.4, 6.1, 6.2, 6.3, 7.1, 7.2, 7.3, 10.3 of the
honest-model-ranking spec — the per-model evaluator that ties NvidiaLM, the
loader, MIA features, control baseline, the MCS calibrator, and bootstrap CIs
into a single ``ModelEvalResult``.

The evaluator is the central integration of tasks 2.1, 2.2, 2.3, 3.1, 3.2, 3.3,
so these tests use mocked LMs / MCS / baseline objects rather than real HTTP
or sklearn fits.
"""

from __future__ import annotations

import dataclasses
import hashlib
from typing import Any
from unittest.mock import MagicMock

import pytest

from recall_guard.core.loader import EvalRow, EvalSet
from recall_guard.core.nvidia_lm import CompletionResult, NvidiaLM, TokenLogprob
from recall_guard.harness.evaluator import (
    CIBound,
    ModelEvalResult,
    Record,
    compute_majority_baseline,
    evaluate_model,
)
from recall_guard.mia.control import ControlBaseline
from recall_guard.mia.mcs import MCSCalibrator

# --- Test fixtures ------------------------------------------------------------


def _make_logprobs(n: int = 5) -> list[TokenLogprob]:
    """Build n synthetic TokenLogprob entries with non-empty top_logprobs."""
    return [
        TokenLogprob(
            token=f"t{i}",
            logprob=-0.5 - 0.1 * i,
            top_logprobs=[{"token": f"t{i}", "logprob": -0.5 - 0.1 * i}, {"token": "x", "logprob": -2.0}],
        )
        for i in range(n)
    ]


def _completion(content: str, temperature: float | None = 0.0) -> CompletionResult:
    return CompletionResult(
        content=content,
        logprobs=_make_logprobs(5),
        raw_temperature_observed=temperature,
    )


def _eval_set(rows: list[EvalRow]) -> EvalSet:
    return EvalSet(rows=rows, cutoff_date=None, path_hash="0" * 64)


def _row(prompt: str, target: int) -> EvalRow:
    return EvalRow(prompt=prompt, target_direction=target, metadata={})


def _baseline() -> ControlBaseline:
    """Synthetic calibrated baseline with finite means/stds and no ref_delta."""
    return ControlBaseline(
        model="m",
        n_valid=50,
        feature_means={
            "loss": 0.5,
            "min_k": 0.7,
            "min_k_pp": 0.0,
            "zlib_ratio": 0.1,
            "ref_delta": None,
        },
        feature_stds={
            "loss": 0.1,
            "min_k": 0.1,
            "min_k_pp": 1.0,
            "zlib_ratio": 0.05,
            "ref_delta": None,
        },
        is_calibrated=True,
        min_valid=50,
    )


def _mcs(p_memorized: float = 0.3, holdout_auc: float = 0.85) -> MCSCalibrator:
    """Mock MCSCalibrator whose predict_proba returns a fixed value."""
    mcs = MagicMock(spec=MCSCalibrator)
    mcs.model = "m"
    mcs.holdout_auc = holdout_auc
    mcs.is_weak = holdout_auc < 0.6
    mcs.feature_order = ["loss", "min_k", "min_k_pp", "zlib_ratio"]
    mcs.predict_proba.return_value = p_memorized
    return mcs


def _lm_returning(scripted_contents: list[str], temperature: float | None = 0.0) -> Any:
    """Build a mock NvidiaLM that returns scripted contents in order, recycling."""
    lm = MagicMock(spec=NvidiaLM)
    lm.model = "m"

    def _generate(prompt: str, temperature_arg: float = 0.0) -> CompletionResult:
        idx = lm._call_idx if hasattr(lm, "_call_idx") else 0
        content = scripted_contents[idx % len(scripted_contents)]
        lm._call_idx = idx + 1
        return _completion(content, temperature=temperature)

    lm._call_idx = 0
    lm.generate.side_effect = _generate
    return lm


def _lm_with_per_call(handlers: list[Any]) -> Any:
    """Mock NvidiaLM where each call invokes the next handler.

    Each handler is either a CompletionResult (returned) or an exception
    instance (raised).
    """
    lm = MagicMock(spec=NvidiaLM)
    lm.model = "m"
    state = {"i": 0}

    def _generate(prompt: str, temperature: float = 0.0) -> CompletionResult:
        i = state["i"]
        state["i"] = i + 1
        h = handlers[i] if i < len(handlers) else handlers[-1]
        if isinstance(h, BaseException):
            raise h
        return h

    lm.generate.side_effect = _generate
    return lm


# --- Dataclass invariants -----------------------------------------------------


def test_record_and_modelevalresult_are_frozen_dataclasses() -> None:
    bound = CIBound(point=0.5, lo=0.4, hi=0.6)
    rec = Record(
        model="m",
        prompt_hash="abc",
        parse_ok=False,
        predicted_direction=None,
        raw_confidence=None,
        penalized_confidence=None,
        target_direction=1,
        features_raw=None,
        features_standardised=None,
        p_memorized=None,
        fail_reason="parse_failure",
    )
    res = ModelEvalResult(
        model="m",
        raw_accuracy=bound,
        memguard_accuracy=bound,
        mcs_auc=bound,
        parse_success_rate=0.0,
        parse_failures=1,
        warnings=[],
        records=[rec],
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        bound.point = 1.0  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.parse_ok = True  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.parse_failures = 0  # type: ignore[misc]


# --- Parse semantics ----------------------------------------------------------


def test_parser_accepts_markdown_bold() -> None:
    """**Direction:** 1 / **Confidence:** 0.7 — phi/llama instinct."""
    from recall_guard.harness.evaluator import _parse_confidence, _parse_direction

    content = "Final answer:\n**Direction:** 1\n**Confidence:** 0.7"
    assert _parse_direction(content) == 1
    assert _parse_confidence(content) == pytest.approx(0.7)


def test_parser_accepts_signed_int_and_float() -> None:
    from recall_guard.harness.evaluator import _parse_confidence, _parse_direction

    assert _parse_direction("Direction: +1\nConfidence: 0.5") == 1
    assert _parse_direction("Direction: -1.0\nConfidence: 0.5") == -1
    assert _parse_confidence("Direction: 1\nConfidence: .65") == pytest.approx(0.65)


@pytest.mark.parametrize(
    "content",
    [
        "Direction: 1.9\nConfidence: 0.5",
        "Direction: -1.5\nConfidence: 0.5",
        "Direction: 0.9\nConfidence: 0.5",
    ],
)
def test_parser_rejects_fractional_directions(content: str) -> None:
    from recall_guard.harness.evaluator import _parse_direction

    assert _parse_direction(content) is None


def test_parser_accepts_annotated_lines() -> None:
    from recall_guard.harness.evaluator import _parse_confidence, _parse_direction

    content = "Direction: 1 (positive close)\nConfidence: 0.65 (moderate)"
    assert _parse_direction(content) == 1
    assert _parse_confidence(content) == pytest.approx(0.65)


def test_parser_falls_back_to_json_block() -> None:
    from recall_guard.harness.evaluator import _parse_confidence, _parse_direction

    content = 'After analysis: {"direction": -1, "confidence": 0.42, "rationale": "..."}'
    assert _parse_direction(content) == -1
    assert _parse_confidence(content) == pytest.approx(0.42)


def test_parser_falls_back_to_word_coercion_on_direction() -> None:
    """When the model never emits 'Direction:' but states the answer in prose."""
    from recall_guard.harness.evaluator import _parse_direction

    assert _parse_direction("After review I believe SPY closed higher today.") == 1
    assert _parse_direction("My conclusion: the ETF closed lower vs the prior session.") == -1
    assert _parse_direction("It was unchanged on the day.") == 0


def test_parser_falls_back_to_percent_for_confidence() -> None:
    from recall_guard.harness.evaluator import _parse_confidence

    assert _parse_confidence("Direction: 1\nConfidence: 65%") == pytest.approx(0.65)
    assert _parse_confidence("Direction: 1\nConfidence: 1%") == pytest.approx(0.01)
    assert _parse_confidence("Direction: 1\nConfidence: 0.5%") == pytest.approx(0.005)


def test_parser_takes_last_match_when_model_restates() -> None:
    """Reasoning models often restate 'Direction' inside their chain."""
    from recall_guard.harness.evaluator import _parse_direction

    content = (
        "Initially I thought Direction: -1 but on review the Direction: 1 is correct."
    )
    assert _parse_direction(content) == 1


def test_parser_rejects_genuine_prose() -> None:
    """'the direction is X' (without colon/asterisk) must not match."""
    from recall_guard.harness.evaluator import _parse_direction

    # No structured marker, no directional keyword tail → should remain None.
    assert _parse_direction("The number 7 appears in this sentence.") is None


def test_parse_failure_captures_raw_excerpt() -> None:
    """Failed parses store the model's response so the user can debug."""
    rows = [_row("p0", 1)]
    lm = _lm_returning(["I am thinking about it but not following format."])
    result = evaluate_model(
        model_lm=lm,
        eval_set=_eval_set(rows),
        baseline=_baseline(),
        mcs=_mcs(),
        ref_lm=None,
        bootstrap_n=10,
        seed=0,
    )
    assert result.records[0].parse_ok is False
    assert result.records[0].fail_reason == "parse_failure"
    assert result.records[0].raw_response_excerpt is not None
    assert "thinking" in result.records[0].raw_response_excerpt


def test_evaluate_model_parses_direction_and_confidence() -> None:
    rows = [_row(f"p{i}", 1) for i in range(3)]
    eval_set = _eval_set(rows)
    lm = _lm_returning([
        "Direction: 1\nConfidence: 0.9",
        "Direction: -1\nConfidence: 0.5",
        "Direction: 0\nConfidence: 0.75",
    ])
    result = evaluate_model(
        model_lm=lm,
        eval_set=eval_set,
        baseline=_baseline(),
        mcs=_mcs(p_memorized=0.2),
        ref_lm=None,
        bootstrap_n=50,
    )

    assert result.parse_failures == 0
    assert result.parse_success_rate == 1.0
    assert len(result.records) == 3
    assert all(r.parse_ok for r in result.records)
    directions = [r.predicted_direction for r in result.records]
    confidences = [r.raw_confidence for r in result.records]
    assert directions == [1, -1, 0]
    assert confidences == [0.9, 0.5, 0.75]


def test_evaluate_model_marks_parse_failure_on_garbage_content() -> None:
    rows = [_row("p1", 1)]
    lm = _lm_returning(["I cannot answer this question."])
    result = evaluate_model(
        model_lm=lm,
        eval_set=_eval_set(rows),
        baseline=_baseline(),
        mcs=_mcs(),
        ref_lm=None,
        bootstrap_n=50,
    )

    rec = result.records[0]
    assert rec.parse_ok is False
    assert rec.fail_reason == "parse_failure"
    assert rec.predicted_direction is None
    assert rec.raw_confidence is None
    assert rec.penalized_confidence is None


def test_evaluate_model_marks_parse_failure_on_out_of_range_direction() -> None:
    rows = [_row("p1", 1)]
    lm = _lm_returning(["Direction: 7\nConfidence: 0.9"])
    result = evaluate_model(
        model_lm=lm,
        eval_set=_eval_set(rows),
        baseline=_baseline(),
        mcs=_mcs(),
        ref_lm=None,
        bootstrap_n=50,
    )
    rec = result.records[0]
    assert rec.parse_ok is False
    assert rec.fail_reason == "parse_failure"


def test_evaluate_model_marks_parse_failure_on_missing_confidence() -> None:
    rows = [_row("p1", 1)]
    lm = _lm_returning(["Direction: 1"])
    result = evaluate_model(
        model_lm=lm,
        eval_set=_eval_set(rows),
        baseline=_baseline(),
        mcs=_mcs(),
        ref_lm=None,
        bootstrap_n=50,
    )
    rec = result.records[0]
    assert rec.parse_ok is False
    assert rec.fail_reason == "parse_failure"


def test_evaluate_model_marks_parse_failure_on_out_of_range_confidence() -> None:
    rows = [_row("p1", 1)]
    lm = _lm_returning(["Direction: 1\nConfidence: 1.5"])
    result = evaluate_model(
        model_lm=lm,
        eval_set=_eval_set(rows),
        baseline=_baseline(),
        mcs=_mcs(),
        ref_lm=None,
        bootstrap_n=50,
    )
    rec = result.records[0]
    assert rec.parse_ok is False
    assert rec.fail_reason == "parse_failure"


# --- LM exception handling ----------------------------------------------------


def test_evaluate_model_marks_no_logprobs_on_runtime_error() -> None:
    rows = [_row("p1", 1)]
    lm = _lm_with_per_call([RuntimeError("missing top_logprobs")])
    result = evaluate_model(
        model_lm=lm,
        eval_set=_eval_set(rows),
        baseline=_baseline(),
        mcs=_mcs(),
        ref_lm=None,
        bootstrap_n=50,
    )
    rec = result.records[0]
    assert rec.parse_ok is False
    assert rec.fail_reason == "no_logprobs"
    assert rec.features_raw is None


def test_evaluate_model_marks_timeout_on_timeout_error() -> None:
    rows = [_row("p1", 1)]
    lm = _lm_with_per_call([TimeoutError("slow")])
    result = evaluate_model(
        model_lm=lm,
        eval_set=_eval_set(rows),
        baseline=_baseline(),
        mcs=_mcs(),
        ref_lm=None,
        bootstrap_n=50,
    )
    rec = result.records[0]
    assert rec.parse_ok is False
    assert rec.fail_reason == "timeout"


def test_evaluate_model_marks_error_on_generic_runtime_error() -> None:
    rows = [_row("p1", 1)]
    lm = _lm_with_per_call([RuntimeError("network reset")])
    result = evaluate_model(
        model_lm=lm,
        eval_set=_eval_set(rows),
        baseline=_baseline(),
        mcs=_mcs(),
        ref_lm=None,
        bootstrap_n=50,
    )
    rec = result.records[0]
    assert rec.parse_ok is False
    assert rec.fail_reason == "error"


# --- Aggregate parse statistics (the explicit task observable) ---------------


def test_evaluate_model_parse_success_rate_and_failures() -> None:
    """The explicit observable from task 4.2: 8 of 10 parse, 2 garbage."""
    rows = [_row(f"p{i}", 1) for i in range(10)]
    contents = ["Direction: 1\nConfidence: 0.9"] * 8 + ["garbage"] * 2
    lm = _lm_returning(contents)
    result = evaluate_model(
        model_lm=lm,
        eval_set=_eval_set(rows),
        baseline=_baseline(),
        mcs=_mcs(p_memorized=0.25),
        ref_lm=None,
        bootstrap_n=50,
    )

    assert result.parse_failures == 2
    assert result.parse_success_rate == 0.8
    parse_ok_count = sum(1 for r in result.records if r.parse_ok)
    assert parse_ok_count == 8


def test_evaluate_model_accuracy_excludes_parse_failures() -> None:
    """Raw accuracy denominator is parse-OK rows only (Req 7.3)."""
    rows = [_row(f"p{i}", 1) for i in range(10)]
    # 6 correct (1) + 2 wrong (-1) + 2 garbage
    contents = (
        ["Direction: 1\nConfidence: 0.9"] * 6
        + ["Direction: -1\nConfidence: 0.9"] * 2
        + ["garbage"] * 2
    )
    lm = _lm_returning(contents)
    result = evaluate_model(
        model_lm=lm,
        eval_set=_eval_set(rows),
        baseline=_baseline(),
        mcs=_mcs(p_memorized=0.0),
        ref_lm=None,
        bootstrap_n=200,
        seed=42,
    )

    # 6 correct out of 8 parse-OK rows -> point estimate 0.75, NOT 0.6 (6/10)
    assert result.raw_accuracy.point == pytest.approx(6 / 8)


# --- Penalty and CI invariants ------------------------------------------------


def test_evaluate_model_penalized_confidence_uses_mcs_predict_proba() -> None:
    """penalized_confidence = raw_confidence * (1 - p_memorized) (Req 5.4)."""
    rows = [_row("p1", 1)]
    lm = _lm_returning(["Direction: 1\nConfidence: 0.8"])
    result = evaluate_model(
        model_lm=lm,
        eval_set=_eval_set(rows),
        baseline=_baseline(),
        mcs=_mcs(p_memorized=0.3),
        ref_lm=None,
        bootstrap_n=50,
    )

    rec = result.records[0]
    assert rec.parse_ok is True
    assert rec.raw_confidence == 0.8
    assert rec.p_memorized == 0.3
    assert rec.penalized_confidence == pytest.approx(0.8 * (1 - 0.3))


def test_evaluate_model_bootstrap_ci_lo_le_point_le_hi() -> None:
    rows = [_row(f"p{i}", 1) for i in range(10)]
    contents = ["Direction: 1\nConfidence: 0.9"] * 7 + ["Direction: -1\nConfidence: 0.5"] * 3
    lm = _lm_returning(contents)
    result = evaluate_model(
        model_lm=lm,
        eval_set=_eval_set(rows),
        baseline=_baseline(),
        mcs=_mcs(p_memorized=0.2),
        ref_lm=None,
        bootstrap_n=500,
        seed=0,
    )

    assert result.raw_accuracy.lo <= result.raw_accuracy.point <= result.raw_accuracy.hi
    assert result.memguard_accuracy.lo <= result.memguard_accuracy.point <= result.memguard_accuracy.hi
    assert result.mcs_auc.lo <= result.mcs_auc.point <= result.mcs_auc.hi


# --- Temperature warning ------------------------------------------------------


def test_evaluate_model_temperature_violation_warning() -> None:
    rows = [_row(f"p{i}", 1) for i in range(3)]
    contents = ["Direction: 1\nConfidence: 0.9"] * 3
    # Mix: first call observed temperature=0.5 (violation), others 0.0
    lm = MagicMock(spec=NvidiaLM)
    lm.model = "m"
    state = {"i": 0}

    def _generate(prompt: str, temperature: float = 0.0) -> CompletionResult:
        i = state["i"]
        state["i"] = i + 1
        observed = 0.5 if i == 0 else 0.0
        return _completion(contents[i], temperature=observed)

    lm.generate.side_effect = _generate

    result = evaluate_model(
        model_lm=lm,
        eval_set=_eval_set(rows),
        baseline=_baseline(),
        mcs=_mcs(p_memorized=0.2),
        ref_lm=None,
        bootstrap_n=50,
    )

    assert "temperature-not-honoured" in result.warnings


def test_evaluate_model_no_temperature_violation() -> None:
    rows = [_row(f"p{i}", 1) for i in range(3)]
    lm = _lm_returning(["Direction: 1\nConfidence: 0.9"] * 3, temperature=0.0)
    result = evaluate_model(
        model_lm=lm,
        eval_set=_eval_set(rows),
        baseline=_baseline(),
        mcs=_mcs(),
        ref_lm=None,
        bootstrap_n=50,
    )

    assert "temperature-not-honoured" not in result.warnings


def test_evaluate_model_no_temperature_violation_when_none() -> None:
    """``raw_temperature_observed=None`` (API didn't expose it) is not a violation."""
    rows = [_row("p1", 1)]
    lm = _lm_returning(["Direction: 1\nConfidence: 0.9"], temperature=None)
    result = evaluate_model(
        model_lm=lm,
        eval_set=_eval_set(rows),
        baseline=_baseline(),
        mcs=_mcs(),
        ref_lm=None,
        bootstrap_n=50,
    )
    assert "temperature-not-honoured" not in result.warnings


# --- Edge cases ---------------------------------------------------------------


def test_evaluate_model_handles_empty_eval_set() -> None:
    eval_set = _eval_set([])
    lm = MagicMock(spec=NvidiaLM)
    lm.model = "m"
    result = evaluate_model(
        model_lm=lm,
        eval_set=eval_set,
        baseline=_baseline(),
        mcs=_mcs(),
        ref_lm=None,
        bootstrap_n=50,
    )

    assert result.parse_failures == 0
    assert result.parse_success_rate == 1.0  # vacuously true
    assert result.records == []
    assert result.raw_accuracy == CIBound(0.0, 0.0, 0.0)
    assert result.memguard_accuracy == CIBound(0.0, 0.0, 0.0)
    # generate must never have been called
    lm.generate.assert_not_called()


def test_evaluate_model_records_in_eval_set_order() -> None:
    prompts = ["alpha", "bravo", "charlie", "delta", "echo"]
    rows = [_row(p, 1) for p in prompts]
    lm = _lm_returning(["Direction: 1\nConfidence: 0.9"] * 5)
    result = evaluate_model(
        model_lm=lm,
        eval_set=_eval_set(rows),
        baseline=_baseline(),
        mcs=_mcs(),
        ref_lm=None,
        bootstrap_n=50,
    )

    expected = [
        hashlib.sha256(p.encode("utf-8")).hexdigest()[:16] for p in prompts
    ]
    assert [r.prompt_hash for r in result.records] == expected


def test_evaluate_model_reference_failure_degrades_cleanly() -> None:
    rows = [_row("p1", 1)]
    lm = _lm_returning(["Direction: 1\nConfidence: 0.9"])
    ref_lm = _lm_with_per_call([TimeoutError("slow")])
    baseline = _baseline()
    baseline.feature_means["ref_delta"] = 0.0
    baseline.feature_stds["ref_delta"] = 1.0
    mcs = _mcs(p_memorized=0.2, holdout_auc=0.85)
    mcs.feature_order = ["loss", "min_k", "min_k_pp", "zlib_ratio"]
    result = evaluate_model(
        model_lm=lm,
        eval_set=_eval_set(rows),
        baseline=baseline,
        mcs=mcs,
        ref_lm=ref_lm,
        bootstrap_n=50,
        seed=0,
    )
    rec = result.records[0]
    assert rec.parse_ok is True
    assert rec.fail_reason is None
    assert rec.p_memorized == pytest.approx(0.2)


def test_evaluate_model_holdout_records_drive_mcs_auc_ci() -> None:
    """When holdout records are passed, MCS-AUC bootstrap CI is computed."""
    rows = [_row("p1", 1)]
    lm = _lm_returning(["Direction: 1\nConfidence: 0.9"])

    # Build synthetic holdout records: 5 IS (label=1, high p_memorized)
    # and 5 OOS (label=0, low p_memorized) so AUC ≈ 1.0.
    holdout: list[Record] = []
    for i in range(5):
        holdout.append(
            Record(
                model="m",
                prompt_hash=f"h{i}",
                parse_ok=True,
                predicted_direction=1,
                raw_confidence=0.9,
                penalized_confidence=0.5,
                target_direction=1,  # label=1 (IS)
                features_raw=None,
                features_standardised=None,
                p_memorized=0.9,
                fail_reason=None,
            )
        )
    for i in range(5):
        holdout.append(
            Record(
                model="m",
                prompt_hash=f"o{i}",
                parse_ok=True,
                predicted_direction=0,
                raw_confidence=0.5,
                penalized_confidence=0.4,
                target_direction=0,  # label=0 (OOS)
                features_raw=None,
                features_standardised=None,
                p_memorized=0.1,
                fail_reason=None,
            )
        )

    result = evaluate_model(
        model_lm=lm,
        eval_set=_eval_set(rows),
        baseline=_baseline(),
        mcs=_mcs(p_memorized=0.2, holdout_auc=0.85),
        ref_lm=None,
        holdout_records=holdout,
        bootstrap_n=200,
        seed=0,
    )

    # AUC for separable distributions is 1.0; CI bounds must satisfy invariant.
    assert 0.5 <= result.mcs_auc.point <= 1.0
    assert result.mcs_auc.lo <= result.mcs_auc.point <= result.mcs_auc.hi


def test_evaluate_model_falls_back_to_holdout_auc_when_no_holdout_records() -> None:
    rows = [_row("p1", 1)]
    lm = _lm_returning(["Direction: 1\nConfidence: 0.9"])
    mcs = _mcs(p_memorized=0.2, holdout_auc=0.85)
    result = evaluate_model(
        model_lm=lm,
        eval_set=_eval_set(rows),
        baseline=_baseline(),
        mcs=mcs,
        ref_lm=None,
        holdout_records=None,
        bootstrap_n=50,
    )

    assert result.mcs_auc.point == pytest.approx(0.85)
    assert result.mcs_auc.lo == pytest.approx(0.85)
    assert result.mcs_auc.hi == pytest.approx(0.85)


# --- Majority baseline -------------------------------------------------------


def test_compute_majority_baseline_uses_bootstrap() -> None:
    rows = [_row(f"p{i}", 1) for i in range(8)] + [_row(f"p{i}", -1) for i in range(2)]
    eval_set = _eval_set(rows)
    bound = compute_majority_baseline(eval_set, bootstrap_n=500, seed=0)
    # Majority-class share is 8/10 = 0.8.
    assert bound.point == pytest.approx(0.8)
    assert bound.lo <= bound.point <= bound.hi


def test_compute_majority_baseline_handles_empty_set() -> None:
    eval_set = _eval_set([])
    bound = compute_majority_baseline(eval_set, bootstrap_n=50, seed=0)
    assert bound == CIBound(0.0, 0.0, 0.0)
