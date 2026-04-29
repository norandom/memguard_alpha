"""Tests for ``harness.report`` (Req 9.1, 9.2, 9.3, 9.4).

Covers:

* ``render_terminal``: includes the majority-baseline row; sorts surviving
  models by score descending while keeping the majority row last.
* ``write_records``: streams one JSON object per ``Record`` to ``records.jsonl``;
  the schema matches the design's per-record artifact contract; parse failures
  are serialised with ``null`` placeholders; ``MiaFeatures`` is serialised as
  a dict (not a stringified dataclass); the parent directory is created.
* ``write_summary_csv``: emits the 15 columns from the design; one row per
  ``ModelEvalResult`` plus a ``__majority_baseline__`` row that fills only the
  raw-accuracy CI cells.
* ``print_artifact_paths``: prints every artifact name and absolute path.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.harness.evaluator import CIBound, ModelEvalResult, Record
from src.harness.ranker import CompositeScore
from src.harness.report import (
    print_artifact_paths,
    render_terminal,
    write_records,
    write_summary_csv,
)
from src.mia.features import MiaFeatures

# --- Synthetic fixtures -------------------------------------------------------


def _make_record(
    model: str,
    target_dir: int,
    parse_ok: bool = True,
    predicted_dir: int = 1,
    p_mem: float = 0.3,
    suffix: str = "0",
) -> Record:
    if parse_ok:
        feats = MiaFeatures(
            loss=0.5,
            min_k=-0.4,
            min_k_pp=0.1,
            zlib_ratio=0.3,
            ref_delta=None,
        )
        std = {
            "loss": 0.0,
            "min_k": 0.0,
            "min_k_pp": 0.0,
            "zlib_ratio": 0.0,
            "ref_delta": None,
        }
        return Record(
            model=model,
            prompt_hash=("abc" + suffix).ljust(16, "0"),
            parse_ok=True,
            predicted_direction=predicted_dir,
            raw_confidence=0.8,
            penalized_confidence=0.8 * (1.0 - p_mem),
            target_direction=target_dir,
            features_raw=feats,
            features_standardised=std,
            p_memorized=p_mem,
            fail_reason=None,
        )
    return Record(
        model=model,
        prompt_hash=("def" + suffix).ljust(16, "0"),
        parse_ok=False,
        predicted_direction=None,
        raw_confidence=None,
        penalized_confidence=None,
        target_direction=target_dir,
        features_raw=None,
        features_standardised=None,
        p_memorized=None,
        fail_reason="parse_failure",
    )


def _make_result(
    name: str,
    n_records: int = 3,
    raw_point: float = 0.7,
    raw_lo: float = 0.6,
    raw_hi: float = 0.8,
    parse_success_rate: float = 1.0,
    parse_failures: int = 0,
    warnings: list[str] | None = None,
) -> ModelEvalResult:
    records = [
        _make_record(name, target_dir=1, suffix=str(i)) for i in range(n_records)
    ]
    return ModelEvalResult(
        model=name,
        raw_accuracy=CIBound(raw_point, raw_lo, raw_hi),
        memguard_accuracy=CIBound(raw_point, raw_lo, raw_hi),
        mcs_auc=CIBound(0.85, 0.75, 0.95),
        parse_success_rate=parse_success_rate,
        parse_failures=parse_failures,
        warnings=list(warnings or []),
        records=records,
    )


def _make_score(
    model: str,
    score: float = 0.5,
    survives: bool = True,
    warnings: list[str] | None = None,
) -> CompositeScore:
    return CompositeScore(
        model=model,
        score=score,
        components={
            "memguard_acc_lo": 0.6,
            "mcs_auc_point": 0.85,
            "parse_success_rate": 1.0,
        },
        survives_gates=survives,
        warnings=list(warnings or []),
    )


# --- render_terminal ----------------------------------------------------------


def test_render_terminal_includes_majority_baseline_row(capsys: pytest.CaptureFixture[str]) -> None:
    results = [_make_result("modelA"), _make_result("modelB")]
    scores = [_make_score("modelA", 0.5), _make_score("modelB", 0.4)]
    majority = CIBound(0.55, 0.45, 0.65)

    render_terminal(results, majority, scores)

    captured = capsys.readouterr().out
    assert "__majority_baseline__" in captured


def test_render_terminal_sorts_models_by_score_descending_and_keeps_majority_last(
    capsys: pytest.CaptureFixture[str],
) -> None:
    results = [
        _make_result("low"),
        _make_result("high"),
        _make_result("mid"),
    ]
    scores = [
        _make_score("low", 0.3),
        _make_score("high", 0.7),
        _make_score("mid", 0.5),
    ]
    majority = CIBound(0.55, 0.45, 0.65)

    render_terminal(results, majority, scores)

    out = capsys.readouterr().out

    # All four labels appear. Compare positions to verify ordering.
    positions = {
        label: out.index(label)
        for label in ("low", "mid", "high", "__majority_baseline__")
    }
    assert positions["high"] < positions["mid"] < positions["low"]
    # Majority baseline always last:
    assert positions["__majority_baseline__"] > max(
        positions["high"], positions["mid"], positions["low"]
    )


def test_render_terminal_renders_warning_strings(
    capsys: pytest.CaptureFixture[str],
) -> None:
    results = [_make_result("flaky", warnings=["temperature-not-honoured"])]
    scores = [_make_score("flaky", 0.4, warnings=["temperature-not-honoured"])]
    majority = CIBound(0.55, 0.45, 0.65)

    render_terminal(results, majority, scores)

    out = capsys.readouterr().out
    assert "temperature-not-honoured" in out


# --- write_records -----------------------------------------------------------


def test_write_records_writes_one_json_per_line(tmp_path: Path) -> None:
    results = [_make_result("modelA", n_records=3), _make_result("modelB", n_records=3)]
    out = tmp_path / "records.jsonl"

    write_records(results, out)

    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 6
    for line in lines:
        json.loads(line)  # raises if malformed


def test_write_records_includes_all_required_fields(tmp_path: Path) -> None:
    results = [_make_result("modelA", n_records=1)]
    out = tmp_path / "records.jsonl"

    write_records(results, out)

    obj = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    expected_keys = {
        "model",
        "prompt_hash",
        "parse_ok",
        "predicted_direction",
        "raw_confidence",
        "penalized_confidence",
        "target_direction",
        "features_raw",
        "features_standardised",
        "p_memorized",
        "fail_reason",
        "raw_response_excerpt",
    }
    assert set(obj.keys()) == expected_keys


def test_write_records_serialises_features_as_dict(tmp_path: Path) -> None:
    results = [_make_result("modelA", n_records=1)]
    out = tmp_path / "records.jsonl"

    write_records(results, out)

    obj = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert isinstance(obj["features_raw"], dict)
    assert set(obj["features_raw"].keys()) == {
        "loss",
        "min_k",
        "min_k_pp",
        "zlib_ratio",
        "ref_delta",
    }
    # ref_delta=None must round-trip as JSON null
    assert obj["features_raw"]["ref_delta"] is None
    # features_standardised is also a dict
    assert isinstance(obj["features_standardised"], dict)


def test_write_records_serialises_parse_failures_with_nulls(tmp_path: Path) -> None:
    failed_record = _make_record("modelA", target_dir=1, parse_ok=False, suffix="x")
    result = ModelEvalResult(
        model="modelA",
        raw_accuracy=CIBound(0.0, 0.0, 0.0),
        memguard_accuracy=CIBound(0.0, 0.0, 0.0),
        mcs_auc=CIBound(0.5, 0.5, 0.5),
        parse_success_rate=0.0,
        parse_failures=1,
        warnings=[],
        records=[failed_record],
    )
    out = tmp_path / "records.jsonl"

    write_records([result], out)

    obj = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert obj["parse_ok"] is False
    assert obj["predicted_direction"] is None
    assert obj["raw_confidence"] is None
    assert obj["penalized_confidence"] is None
    assert obj["features_raw"] is None
    assert obj["features_standardised"] is None
    assert obj["p_memorized"] is None
    assert obj["fail_reason"] == "parse_failure"


def test_write_records_creates_parent_directory(tmp_path: Path) -> None:
    results = [_make_result("modelA", n_records=2)]
    out = tmp_path / "nested" / "run1" / "records.jsonl"

    write_records(results, out)

    assert out.parent.is_dir()
    assert out.exists()
    assert len(out.read_text(encoding="utf-8").splitlines()) == 2


# --- write_summary_csv -------------------------------------------------------


_EXPECTED_CSV_COLUMNS = [
    "model",
    "raw_acc_point",
    "raw_acc_lo",
    "raw_acc_hi",
    "memguard_acc_point",
    "memguard_acc_lo",
    "memguard_acc_hi",
    "mcs_auc_point",
    "mcs_auc_lo",
    "mcs_auc_hi",
    "parse_success_rate",
    "parse_failures",
    "score",
    "survives_gates",
    "warnings",
]


def test_write_summary_csv_has_expected_columns(tmp_path: Path) -> None:
    results = [_make_result("modelA"), _make_result("modelB")]
    scores = [_make_score("modelA", 0.5), _make_score("modelB", 0.4)]
    majority = CIBound(0.55, 0.45, 0.65)
    out = tmp_path / "summary.csv"

    write_summary_csv(results, scores, majority, out)

    with out.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
    assert header == _EXPECTED_CSV_COLUMNS


def test_write_summary_csv_has_one_row_per_model_plus_majority_row(tmp_path: Path) -> None:
    results = [_make_result("modelA"), _make_result("modelB")]
    scores = [_make_score("modelA", 0.5), _make_score("modelB", 0.4)]
    majority = CIBound(0.55, 0.45, 0.65)
    out = tmp_path / "summary.csv"

    write_summary_csv(results, scores, majority, out)

    with out.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    assert len(rows) == 3
    assert [r["model"] for r in rows] == ["modelA", "modelB", "__majority_baseline__"]


def test_write_summary_csv_majority_row_only_fills_raw_acc(tmp_path: Path) -> None:
    results = [_make_result("modelA")]
    scores = [_make_score("modelA", 0.5)]
    majority = CIBound(0.55, 0.45, 0.65)
    out = tmp_path / "summary.csv"

    write_summary_csv(results, scores, majority, out)

    with out.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    majority_row = rows[-1]
    assert majority_row["model"] == "__majority_baseline__"
    assert majority_row["raw_acc_point"] != ""
    assert majority_row["raw_acc_lo"] != ""
    assert majority_row["raw_acc_hi"] != ""
    assert float(majority_row["raw_acc_point"]) == pytest.approx(0.55)
    assert float(majority_row["raw_acc_lo"]) == pytest.approx(0.45)
    assert float(majority_row["raw_acc_hi"]) == pytest.approx(0.65)
    # Other CI fields and score / parse stats are blank for the majority row.
    for blank_col in (
        "memguard_acc_point",
        "memguard_acc_lo",
        "memguard_acc_hi",
        "mcs_auc_point",
        "mcs_auc_lo",
        "mcs_auc_hi",
        "parse_success_rate",
        "parse_failures",
        "score",
        "survives_gates",
        "warnings",
    ):
        assert majority_row[blank_col] == "", (
            f"expected blank for {blank_col}, got {majority_row[blank_col]!r}"
        )


def test_write_summary_csv_writes_score_and_warnings_per_model(tmp_path: Path) -> None:
    results = [
        _make_result("modelA", warnings=["temperature-not-honoured"]),
        _make_result("modelB"),
    ]
    scores = [
        _make_score(
            "modelA", 0.0, survives=False, warnings=["weak-calibration"]
        ),
        _make_score("modelB", 0.42, survives=True, warnings=[]),
    ]
    majority = CIBound(0.55, 0.45, 0.65)
    out = tmp_path / "summary.csv"

    write_summary_csv(results, scores, majority, out)

    with out.open("r", encoding="utf-8", newline="") as fh:
        rows = {row["model"]: row for row in csv.DictReader(fh)}

    assert float(rows["modelA"]["score"]) == pytest.approx(0.0)
    assert rows["modelA"]["survives_gates"].lower() == "false"
    assert "weak-calibration" in rows["modelA"]["warnings"]
    assert float(rows["modelB"]["score"]) == pytest.approx(0.42)
    assert rows["modelB"]["survives_gates"].lower() == "true"


def test_write_summary_csv_creates_parent_directory(tmp_path: Path) -> None:
    results = [_make_result("modelA")]
    scores = [_make_score("modelA", 0.5)]
    majority = CIBound(0.55, 0.45, 0.65)
    out = tmp_path / "nested" / "run2" / "summary.csv"

    write_summary_csv(results, scores, majority, out)

    assert out.exists()


# --- print_artifact_paths ----------------------------------------------------


def test_print_artifact_paths_prints_each_path(capsys: pytest.CaptureFixture[str]) -> None:
    paths = {
        "manifest.json": Path("/tmp/run1/manifest.json"),
        "shortlist.json": Path("/tmp/run1/shortlist.json"),
        "records.jsonl": Path("/tmp/run1/records.jsonl"),
    }

    print_artifact_paths(paths)

    out = capsys.readouterr().out
    for name, path in paths.items():
        assert name in out
        assert str(path) in out
