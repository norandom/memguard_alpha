"""Tests for harness.ranker.

Covers Requirements 5.3, 6.4, 7.4, 8.1, 8.2, 8.3, 8.4 of the
honest-model-ranking spec — the composite scorer + top-3 writer that converts
``ModelEvalResult`` instances into a ranked, gate-aware shortlist with an
explanatory ``top3.md``.

The ranker is the synthesis point for gate warnings: weak-calibration,
parse-unreliable, not-better-than-baseline, and uncalibrated each block a
model from the surviving top 3, while ``temperature-not-honoured`` is purely
informational.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from recall_guard.harness.evaluator import CIBound, ModelEvalResult
from recall_guard.harness.ranker import (
    COMPOSITE_FORMULA,
    GATES,
    CompositeScore,
    composite_score,
    write_top3,
)

# --- Fixtures -----------------------------------------------------------------


def _make_result(
    name: str,
    raw_acc: tuple[float, float, float] = (0.7, 0.6, 0.8),
    mg_acc: tuple[float, float, float] = (0.7, 0.6, 0.8),
    auc: tuple[float, float, float] = (0.85, 0.75, 0.95),
    parse: float = 1.0,
    warnings: tuple[str, ...] = (),
) -> ModelEvalResult:
    return ModelEvalResult(
        model=name,
        raw_accuracy=CIBound(*raw_acc),
        memguard_accuracy=CIBound(*mg_acc),
        mcs_auc=CIBound(*auc),
        parse_success_rate=parse,
        parse_failures=0,
        warnings=list(warnings),
        records=[],
    )


def _baseline(point: float = 0.5, lo: float = 0.4, hi: float = 0.55) -> CIBound:
    return CIBound(point=point, lo=lo, hi=hi)


# --- Dataclass invariants -----------------------------------------------------


def test_composite_score_is_frozen_dataclass() -> None:
    score = CompositeScore(
        model="m",
        score=0.5,
        components={"a": 1.0},
        survives_gates=True,
        warnings=[],
    )
    assert dataclasses.is_dataclass(score)
    # frozen → assignment must raise FrozenInstanceError
    try:
        score.model = "x"  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        pass
    else:  # pragma: no cover - defensive
        raise AssertionError("CompositeScore is not frozen")


def test_module_exports_constants() -> None:
    assert COMPOSITE_FORMULA == "memguard_acc_lo * mcs_auc_point * parse_success_rate"
    assert GATES == {"parse_min": 0.8, "mcs_auc_min": 0.6}


# --- Gate behaviour -----------------------------------------------------------


def test_composite_score_passes_all_gates() -> None:
    result = _make_result("good", mg_acc=(0.7, 0.6, 0.8), auc=(0.85, 0.8, 0.9), parse=1.0)
    [score] = composite_score([result], _baseline(0.5, 0.4, 0.55))
    assert score.survives_gates is True
    assert score.warnings == []
    # 0.6 * 0.85 * 1.0 = 0.51
    assert abs(score.score - 0.51) < 1e-9
    assert score.components == {
        "memguard_acc_lo": 0.6,
        "mcs_auc_point": 0.85,
        "parse_success_rate": 1.0,
    }


def test_composite_score_fails_weak_calibration() -> None:
    # mcs_auc.point = 0.55 < mcs_auc_min=0.6
    result = _make_result("weak", mg_acc=(0.7, 0.6, 0.8), auc=(0.55, 0.5, 0.6), parse=1.0)
    [score] = composite_score([result], _baseline(0.5, 0.4, 0.55))
    assert "weak-calibration" in score.warnings
    assert score.survives_gates is False
    assert score.score == 0.0


def test_composite_score_fails_parse_unreliable() -> None:
    result = _make_result("parse", mg_acc=(0.7, 0.6, 0.8), auc=(0.85, 0.8, 0.9), parse=0.7)
    [score] = composite_score([result], _baseline(0.5, 0.4, 0.55))
    assert "parse-unreliable" in score.warnings
    assert score.survives_gates is False
    assert score.score == 0.0


def test_composite_score_fails_not_better_than_baseline() -> None:
    # mg_acc.lo=0.5 ≤ majority.hi=0.55
    result = _make_result("nb", mg_acc=(0.6, 0.5, 0.7), auc=(0.85, 0.8, 0.9), parse=1.0)
    [score] = composite_score([result], _baseline(0.5, 0.4, 0.55))
    assert "not-better-than-baseline" in score.warnings
    assert score.survives_gates is False
    assert score.score == 0.0


def test_composite_score_fails_uncalibrated() -> None:
    result = _make_result(
        "uncal",
        mg_acc=(0.7, 0.6, 0.8),
        auc=(0.85, 0.8, 0.9),
        parse=1.0,
        warnings=("uncalibrated",),
    )
    [score] = composite_score([result], _baseline(0.5, 0.4, 0.55))
    assert "uncalibrated" in score.warnings
    assert score.survives_gates is False
    assert score.score == 0.0


def test_composite_score_temperature_warning_does_not_block_gates() -> None:
    result = _make_result(
        "temp",
        mg_acc=(0.7, 0.6, 0.8),
        auc=(0.85, 0.8, 0.9),
        parse=1.0,
        warnings=("temperature-not-honoured",),
    )
    [score] = composite_score([result], _baseline(0.5, 0.4, 0.55))
    assert "temperature-not-honoured" in score.warnings
    assert score.survives_gates is True
    assert abs(score.score - 0.51) < 1e-9


def test_composite_score_orders_results_in_input_order() -> None:
    a = _make_result("a", mg_acc=(0.7, 0.6, 0.8))
    b = _make_result("b", mg_acc=(0.7, 0.6, 0.8))
    out = composite_score([a, b], _baseline(0.5, 0.4, 0.55))
    assert [s.model for s in out] == ["a", "b"]


# --- Markdown writer ---------------------------------------------------------


def test_write_top3_writes_three_models_when_three_survive(tmp_path: Path) -> None:
    # 5 inputs: 3 pass all gates (different scores so they have a clear order),
    # 2 fail.
    survivors = [
        _make_result(f"s{i}", mg_acc=(0.7, 0.6 + 0.05 * i, 0.8), auc=(0.85, 0.8, 0.9), parse=1.0)
        for i in range(3)
    ]
    nonsurvivors = [
        _make_result("weak", mg_acc=(0.7, 0.6, 0.8), auc=(0.55, 0.5, 0.6), parse=1.0),
        _make_result("parse-bad", mg_acc=(0.7, 0.6, 0.8), auc=(0.85, 0.8, 0.9), parse=0.5),
    ]
    scores = composite_score(survivors + nonsurvivors, _baseline(0.5, 0.4, 0.55))
    target = tmp_path / "top3.md"
    write_top3(scores, target)

    body = target.read_text(encoding="utf-8")
    assert body.startswith("# Top 3 Models")
    # All three survivors must appear
    for s in survivors:
        assert s.model in body
    # Highest-scoring survivor (s2 with mg_acc.lo=0.7) appears before s0 (lo=0.6)
    assert body.index("s2") < body.index("s0")
    # No "Why fewer" section
    assert "Why fewer than three models" not in body
    # Footer with formula and gates
    assert COMPOSITE_FORMULA in body
    assert "parse_min=0.8" in body
    assert "mcs_auc_min=0.6" in body


def test_write_top3_writes_short_list_with_explanation_when_one_survives(
    tmp_path: Path,
) -> None:
    # The exact scenario from the task brief: 3 results — one passes, one weak
    # MCS, one parse-unreliable.
    good = _make_result("good", mg_acc=(0.7, 0.6, 0.8), auc=(0.85, 0.8, 0.9), parse=1.0)
    weak = _make_result("weak", mg_acc=(0.7, 0.6, 0.8), auc=(0.55, 0.5, 0.6), parse=1.0)
    parse_bad = _make_result("parse-bad", mg_acc=(0.7, 0.6, 0.8), auc=(0.85, 0.8, 0.9), parse=0.5)

    scores = composite_score([good, weak, parse_bad], _baseline(0.5, 0.4, 0.55))
    target = tmp_path / "top3.md"
    write_top3(scores, target)

    body = target.read_text(encoding="utf-8")
    assert "# Top 3 Models" in body
    assert "good" in body
    assert "## Why fewer than three models" in body
    # Both non-survivors must be named with the gate they failed
    assert "weak" in body
    assert "parse-bad" in body
    assert "weak-calibration" in body
    assert "parse-unreliable" in body
    # Formula footer is always present
    assert COMPOSITE_FORMULA in body
    assert "parse_min=0.8" in body
    assert "mcs_auc_min=0.6" in body


def test_write_top3_writes_zero_models_when_none_survive(tmp_path: Path) -> None:
    weak = _make_result("weak", mg_acc=(0.7, 0.6, 0.8), auc=(0.55, 0.5, 0.6), parse=1.0)
    parse_bad = _make_result("parse-bad", mg_acc=(0.7, 0.6, 0.8), auc=(0.85, 0.8, 0.9), parse=0.5)

    scores = composite_score([weak, parse_bad], _baseline(0.5, 0.4, 0.55))
    target = tmp_path / "top3.md"
    write_top3(scores, target)

    body = target.read_text(encoding="utf-8")
    assert "# Top 3 Models" in body
    assert "## Why fewer than three models" in body
    assert "weak" in body
    assert "parse-bad" in body
    assert COMPOSITE_FORMULA in body


def test_write_top3_creates_parent_directory(tmp_path: Path) -> None:
    good = _make_result("good", mg_acc=(0.7, 0.6, 0.8), auc=(0.85, 0.8, 0.9), parse=1.0)
    [score] = composite_score([good], _baseline(0.5, 0.4, 0.55))
    target = tmp_path / "nested" / "deeper" / "top3.md"
    assert not target.parent.exists()
    write_top3([score], target)
    assert target.exists()


def test_write_top3_includes_formula_and_gates_footer(tmp_path: Path) -> None:
    good = _make_result("good", mg_acc=(0.7, 0.6, 0.8), auc=(0.85, 0.8, 0.9), parse=1.0)
    scores = composite_score([good], _baseline(0.5, 0.4, 0.55))
    target = tmp_path / "top3.md"
    write_top3(scores, target)

    body = target.read_text(encoding="utf-8")
    assert "## Composite score formula" in body
    assert COMPOSITE_FORMULA in body
    assert "parse_min=0.8" in body
    assert "mcs_auc_min=0.6" in body


def test_write_top3_descending_order_with_stable_ties(tmp_path: Path) -> None:
    # Three results with identical scores — output preserves input order.
    a = _make_result("alpha", mg_acc=(0.7, 0.6, 0.8), auc=(0.85, 0.8, 0.9), parse=1.0)
    b = _make_result("bravo", mg_acc=(0.7, 0.6, 0.8), auc=(0.85, 0.8, 0.9), parse=1.0)
    c = _make_result("charlie", mg_acc=(0.7, 0.6, 0.8), auc=(0.85, 0.8, 0.9), parse=1.0)
    scores = composite_score([a, b, c], _baseline(0.5, 0.4, 0.55))
    target = tmp_path / "top3.md"
    write_top3(scores, target)

    body = target.read_text(encoding="utf-8")
    # All same score → original input order preserved (use unique tokens
    # that cannot collide with the formula footer wording).
    assert body.index("alpha") < body.index("bravo") < body.index("charlie")
    # Composite scores are also returned in input order from composite_score().
    assert [s.model for s in scores] == ["alpha", "bravo", "charlie"]
