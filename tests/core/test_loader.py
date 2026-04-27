"""Tests for src.core.loader: JSONL eval-set loader and cutoff guard.

Covers requirements 2.1, 2.2, 2.3, 2.4, 2.5 from the honest-model-ranking spec.
"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import pytest

from src.core.loader import (
    CutoffViolation,
    EvalRow,
    EvalSet,
    assert_cutoff_safe,
    load_cutoffs,
    load_eval_set,
)


def _write_jsonl(path: Path, lines: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(json.dumps(line) + "\n")


def _row(direction: int, prompt: str | None = None) -> dict:
    return {
        "prompt": prompt if prompt is not None else f"Prompt for direction {direction}",
        "target_direction": direction,
    }


def test_load_eval_set_parses_rows_and_header(tmp_path: Path) -> None:
    """Header + 5 rows -> 5 EvalRows with correct cutoff_date (Req 2.1, 2.5)."""
    eval_path = tmp_path / "eval.jsonl"
    rows: list[dict] = [{"_cutoff_date": "2025-01-01"}]
    rows.extend(_row(d) for d in [-1, 0, 1, -1, 1])
    _write_jsonl(eval_path, rows)

    result = load_eval_set(eval_path)

    assert isinstance(result, EvalSet)
    assert result.cutoff_date == date(2025, 1, 1)
    assert len(result.rows) == 5
    assert all(isinstance(r, EvalRow) for r in result.rows)
    assert [r.target_direction for r in result.rows] == [-1, 0, 1, -1, 1]
    # path_hash is a sha256 hex digest (64 chars)
    assert isinstance(result.path_hash, str)
    assert len(result.path_hash) == 64


def test_load_eval_set_rejects_invalid_target_direction(tmp_path: Path) -> None:
    """target_direction outside {-1,0,1} must raise (Req 2.1)."""
    eval_path = tmp_path / "bad_direction.jsonl"
    _write_jsonl(eval_path, [{"prompt": "p", "target_direction": 2}])

    with pytest.raises(ValueError):
        load_eval_set(eval_path)


def test_load_eval_set_rejects_missing_prompt(tmp_path: Path) -> None:
    """Row missing prompt must raise (Req 2.1)."""
    eval_path = tmp_path / "missing_prompt.jsonl"
    _write_jsonl(eval_path, [{"target_direction": 1}])

    with pytest.raises(ValueError):
        load_eval_set(eval_path)


def test_load_eval_set_warns_on_small_n(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """N<100 -> WARNING about low statistical power (Req 2.2)."""
    eval_path = tmp_path / "small.jsonl"
    rows = [_row(d) for d in ([-1] * 10 + [0] * 10 + [1] * 10)]  # 30 balanced rows
    _write_jsonl(eval_path, rows)

    with caplog.at_level(logging.WARNING, logger="src.core.loader"):
        result = load_eval_set(eval_path)

    assert len(result.rows) == 30
    warning_messages = " ".join(r.getMessage().lower() for r in caplog.records if r.levelno == logging.WARNING)
    assert "statistical power" in warning_messages or "low" in warning_messages
    assert "100" in warning_messages or "30" in warning_messages


def test_load_eval_set_warns_on_imbalance(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Majority class > 60% -> WARNING about imbalance (Req 2.3)."""
    eval_path = tmp_path / "imbalanced.jsonl"
    # 80% majority class (24/30 = 0.80 bullish=1)
    rows = [_row(1) for _ in range(24)] + [_row(-1) for _ in range(6)]
    _write_jsonl(eval_path, rows)

    with caplog.at_level(logging.WARNING, logger="src.core.loader"):
        result = load_eval_set(eval_path)

    assert len(result.rows) == 30
    messages = " ".join(r.getMessage().lower() for r in caplog.records if r.levelno == logging.WARNING)
    assert "imbalance" in messages or "majority" in messages


def test_load_eval_set_returns_full_set_no_split(tmp_path: Path) -> None:
    """30 rows -> all 30 returned, no 80/20 split (Req 2.4)."""
    eval_path = tmp_path / "thirty.jsonl"
    rows = [_row(d) for d in ([-1] * 10 + [0] * 10 + [1] * 10)]
    _write_jsonl(eval_path, rows)

    result = load_eval_set(eval_path)

    assert len(result.rows) == 30
    # Must not be the 80/20 split sizes
    assert len(result.rows) != 24
    assert len(result.rows) != 6


def test_load_cutoffs_parses_yaml(tmp_path: Path) -> None:
    """Minimal yaml -> dict[str, date] (Req 2.5)."""
    cutoffs_path = tmp_path / "cutoffs.yaml"
    cutoffs_path.write_text(
        "models:\n"
        "  meta/llama-3.2-1b-instruct: 2023-09-30\n"
        "  nvidia/nemotron-3-super-120b-a12b: 2024-06-30\n",
        encoding="utf-8",
    )

    cutoffs = load_cutoffs(cutoffs_path)

    assert cutoffs == {
        "meta/llama-3.2-1b-instruct": date(2023, 9, 30),
        "nvidia/nemotron-3-super-120b-a12b": date(2024, 6, 30),
    }


def test_assert_cutoff_safe_rejects_late_cutoff() -> None:
    """Model cutoff later than eval cutoff -> CutoffViolation (Req 2.5)."""
    eval_set = EvalSet(rows=[], cutoff_date=date(2025, 1, 1), path_hash="0" * 64)
    cutoffs = {"nvidia/nemotron-X": date(2025, 6, 1)}

    with pytest.raises(CutoffViolation):
        assert_cutoff_safe(eval_set, ["nvidia/nemotron-X"], cutoffs)


def test_assert_cutoff_safe_rejects_missing_model() -> None:
    """Shortlisted model missing from cutoffs -> CutoffViolation (Req 2.5)."""
    eval_set = EvalSet(rows=[], cutoff_date=date(2025, 1, 1), path_hash="0" * 64)
    cutoffs = {"meta/llama-3.2-1b-instruct": date(2023, 9, 30)}

    with pytest.raises(CutoffViolation):
        assert_cutoff_safe(eval_set, ["nvidia/nemotron-mystery"], cutoffs)


def test_assert_cutoff_safe_passes_when_all_cutoffs_are_earlier() -> None:
    """All cutoffs <= eval cutoff and present -> no exception (Req 2.5)."""
    eval_set = EvalSet(rows=[], cutoff_date=date(2025, 1, 1), path_hash="0" * 64)
    cutoffs = {
        "meta/llama-3.2-1b-instruct": date(2023, 9, 30),
        "nvidia/nemotron-3-super-120b-a12b": date(2024, 6, 30),
    }

    # Should not raise
    assert_cutoff_safe(
        eval_set,
        ["meta/llama-3.2-1b-instruct", "nvidia/nemotron-3-super-120b-a12b"],
        cutoffs,
    )
