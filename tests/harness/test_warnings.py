"""Harness-level integration test for small-N and class-imbalance warnings.

Covers requirements 2.2, 2.3, 2.4 from the honest-model-ranking spec at the
harness end-to-end boundary: when ``load_eval_set`` is invoked through the
harness path on a realistic 30-row JSONL with 80% majority class, both the
"low statistical power" and the "class imbalance" warnings must be emitted at
WARNING level while loading still returns the entire 30-row eval set (no
train/dev split).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from src.core.loader import load_eval_set


def _write_jsonl(path: Path, lines: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(json.dumps(line) + "\n")


def _row(direction: int, idx: int) -> dict:
    return {
        "prompt": f"Prompt #{idx} for direction {direction}",
        "target_direction": direction,
    }


def test_load_eval_set_emits_low_n_and_imbalance_warnings(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """30-row JSONL with 80% majority class -> both warnings + full 30 rows.

    Integration-level evidence for Req 2.2 (low-N), Req 2.3 (imbalance), and
    Req 2.4 (no train/dev split) flowing through the harness loader path.
    """
    eval_path = tmp_path / "small_imbalanced.jsonl"
    rows: list[dict] = []
    rows.extend(_row(1, i) for i in range(24))  # 24 majority (label=1)
    rows.extend(_row(-1, i) for i in range(24, 30))  # 6 minority (label=-1)
    _write_jsonl(eval_path, rows)

    with caplog.at_level(logging.WARNING, logger="src.core.loader"):
        eval_set = load_eval_set(eval_path)

    # Req 2.4: full set returned, no split.
    assert len(eval_set.rows) == 30

    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    # Both warnings must fire (Req 2.2 + Req 2.3).
    assert len(warning_records) >= 2, (
        f"expected at least 2 WARNING records, got {len(warning_records)}: "
        f"{[r.getMessage() for r in warning_records]}"
    )

    messages_lower = [r.getMessage().lower() for r in warning_records]
    joined = " ".join(messages_lower)

    # Req 2.2: low statistical power warning present.
    assert any("low statistical power" in m for m in messages_lower), (
        f"expected a 'low statistical power' warning, got: {messages_lower}"
    )

    # Req 2.3: class imbalance warning present.
    assert "class imbalance" in joined, (
        f"expected a 'class imbalance' warning, got: {messages_lower}"
    )

    # Both must be at WARNING level (not ERROR or CRITICAL).
    assert all(r.levelno == logging.WARNING for r in warning_records)


def test_load_eval_set_no_warnings_when_well_formed(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """100-row JSONL with 50/50 class split -> zero loader WARNINGs.

    Proves the small-N and imbalance warnings are conditional, not
    unconditional, on the harness loader path.
    """
    eval_path = tmp_path / "well_formed.jsonl"
    rows: list[dict] = []
    rows.extend(_row(1, i) for i in range(50))
    rows.extend(_row(-1, i) for i in range(50, 100))
    _write_jsonl(eval_path, rows)

    with caplog.at_level(logging.WARNING, logger="src.core.loader"):
        eval_set = load_eval_set(eval_path)

    assert len(eval_set.rows) == 100

    loader_warnings = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and r.name == "src.core.loader"
    ]
    assert loader_warnings == [], (
        "expected no loader WARNINGs for a well-formed 100-row balanced file, "
        f"got: {[r.getMessage() for r in loader_warnings]}"
    )
