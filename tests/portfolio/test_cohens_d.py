"""Tests for src.portfolio.cohens_d: per-(model, MIA-feature) Cohen's d.

Covers requirements 1.1, 1.2, 1.3, 1.4, 1.5, 9.1 of cmmd-backtest, task 2.2.

Builds an in-memory fixture run directory under ``tmp_path``: a small
``eval.jsonl`` (prompts + ``metadata.date`` + ``metadata.ticker``), a
matching ``records.jsonl`` keyed by ``prompt_hash``, a ``summary.csv``
with the harness's standard column schema, and a ``cutoffs.yaml``.

No real harness run is required. The tests deliberately reproduce the
hashing convention from ``src.harness.evaluator._hash_prompt`` (sha256
hex, first 16 chars) so the records' ``prompt_hash`` join lines up with
the eval set's ``compute_prompt_hash``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import pandas as pd
import yaml

# ----------------------------------------------------------------------------
# Test helpers
# ----------------------------------------------------------------------------


# The five MiaFeatures field names as written into ``records.jsonl`` by the
# harness's ``_record_to_jsonable``. The artifact's ``feature`` column uses
# these literal names.
_FEATURE_NAMES = ("loss", "min_k", "min_k_pp", "zlib_ratio", "ref_delta")


def _hash_prompt(prompt: str) -> str:
    """Match ``src.harness.evaluator._hash_prompt`` byte-for-byte."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def _write_eval_jsonl(path: Path, rows: list[dict]) -> None:
    """Write a minimal eval JSONL with one prompt + metadata block per row."""
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")


def _write_records_jsonl(path: Path, records: list[dict]) -> None:
    """Write a records.jsonl in the same shape ``harness.report`` produces."""
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False))
            fh.write("\n")


def _write_summary_csv(path: Path, rows: list[dict]) -> None:
    """Write a summary.csv with the actual harness column schema.

    The columns mirror ``src.harness.report.SUMMARY_CSV_COLUMNS``; only
    ``model`` and ``mcs_auc_point`` are load-bearing for Cohen's d, but
    we keep the full schema for fidelity with what the harness writes.
    """
    cols = [
        "model",
        "raw_acc_point", "raw_acc_lo", "raw_acc_hi",
        "memguard_acc_point", "memguard_acc_lo", "memguard_acc_hi",
        "mcs_auc_point", "mcs_auc_lo", "mcs_auc_hi",
        "parse_success_rate", "parse_failures",
        "score", "survives_gates", "warnings",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            full = {c: row.get(c, "") for c in cols}
            writer.writerow(full)


def _make_record(
    *,
    model: str,
    prompt: str,
    parse_ok: bool,
    target_direction: int,
    feature_values: dict[str, float | None] | None,
    p_memorized: float | None = 0.5,
    predicted_direction: int | None = 1,
    raw_confidence: float | None = 0.7,
) -> dict:
    """Build a JSON-serialisable record matching the harness's record schema."""
    if feature_values is None:
        features_raw = None
        features_standardised = None
    else:
        features_raw = {name: feature_values.get(name) for name in _FEATURE_NAMES}
        features_standardised = {name: 0.0 for name in _FEATURE_NAMES}
    return {
        "model": model,
        "prompt_hash": _hash_prompt(prompt),
        "parse_ok": parse_ok,
        "predicted_direction": predicted_direction if parse_ok else None,
        "raw_confidence": raw_confidence if parse_ok else None,
        "penalized_confidence": (
            (raw_confidence or 0.0) * (1.0 - (p_memorized or 0.0))
            if parse_ok else None
        ),
        "target_direction": target_direction,
        "features_raw": features_raw,
        "features_standardised": features_standardised,
        "p_memorized": p_memorized if parse_ok else None,
        "fail_reason": None if parse_ok else "parse_failure",
        "raw_response_excerpt": None,
    }


_IS_DATES = ["2024-01-15", "2024-02-15", "2024-03-15", "2024-04-15"]
_OOS_DATES = ["2024-08-15", "2024-09-15", "2024-10-15", "2024-11-15"]


def _make_primary_records(
    model: str,
    is_loss_values: list[float],
    oos_loss_values: list[float],
) -> tuple[list[dict], list[dict]]:
    eval_rows: list[dict] = []
    records: list[dict] = []
    for i, d in enumerate(_IS_DATES):
        prompt = f"prompt-IS-{d}-XLK"
        eval_rows.append({"prompt": prompt, "target_direction": 1,
                          "metadata": {"date": d, "ticker": "XLK"}})
        loss_v = is_loss_values[i] if i < len(is_loss_values) else is_loss_values[-1]
        records.append(_make_record(
            model=model, prompt=prompt, parse_ok=True, target_direction=1,
            feature_values={"loss": loss_v, "min_k": -1.0 + 0.1 * i,
                            "min_k_pp": 0.5 + 0.1 * i, "zlib_ratio": 0.3 + 0.05 * i,
                            "ref_delta": 0.1 * i},
        ))
    for i, d in enumerate(_OOS_DATES):
        prompt = f"prompt-OOS-{d}-XLK"
        eval_rows.append({"prompt": prompt, "target_direction": 0,
                          "metadata": {"date": d, "ticker": "XLK"}})
        loss_v = oos_loss_values[i] if i < len(oos_loss_values) else oos_loss_values[-1]
        records.append(_make_record(
            model=model, prompt=prompt, parse_ok=True, target_direction=0,
            feature_values={"loss": loss_v, "min_k": -1.5 + 0.1 * i,
                            "min_k_pp": 0.4 + 0.1 * i, "zlib_ratio": 0.25 + 0.05 * i,
                            "ref_delta": 0.05 + 0.1 * i},
        ))
    return eval_rows, records


def _make_other_model_records(entry: dict) -> tuple[list[dict], list[dict]]:
    other_model_id = entry["model"]
    eval_rows: list[dict] = []
    records: list[dict] = []
    for i, d in enumerate(_IS_DATES):
        prompt = f"prompt-IS-{d}-XLK-{other_model_id}"
        eval_rows.append({"prompt": prompt, "target_direction": 1,
                          "metadata": {"date": d, "ticker": "XLK"}})
        records.append(_make_record(
            model=other_model_id, prompt=prompt, parse_ok=True, target_direction=1,
            feature_values={"loss": 0.7 + 0.01 * i, "min_k": -1.0,
                            "min_k_pp": 0.5, "zlib_ratio": 0.3, "ref_delta": 0.0},
        ))
    for i, d in enumerate(_OOS_DATES):
        prompt = f"prompt-OOS-{d}-XLK-{other_model_id}"
        eval_rows.append({"prompt": prompt, "target_direction": 0,
                          "metadata": {"date": d, "ticker": "XLK"}})
        records.append(_make_record(
            model=other_model_id, prompt=prompt, parse_ok=True, target_direction=0,
            feature_values={"loss": 0.2 + 0.01 * i, "min_k": -1.5,
                            "min_k_pp": 0.4, "zlib_ratio": 0.25, "ref_delta": 0.05},
        ))
    return eval_rows, records


def _summary_row(model: str, mcs_auc_point: float, *,
                 acc: str = "0.55", acc_lo: str = "0.50", acc_hi: str = "0.60",
                 auc_lo: str = "0.78", auc_hi: str = "0.86") -> dict:
    return {
        "model": model,
        "raw_acc_point": acc, "raw_acc_lo": acc_lo, "raw_acc_hi": acc_hi,
        "memguard_acc_point": acc, "memguard_acc_lo": acc_lo, "memguard_acc_hi": acc_hi,
        "mcs_auc_point": f"{mcs_auc_point:.6f}",
        "mcs_auc_lo": auc_lo, "mcs_auc_hi": auc_hi,
        "parse_success_rate": "1.0", "parse_failures": "0",
        "score": acc, "survives_gates": "true", "warnings": "",
    }


def _build_fixture_run(
    tmp_path: Path,
    *,
    model: str = "openai/gpt-oss-20b",
    cutoff_iso: str = "2024-06-30",
    is_loss_values: list[float] | None = None,
    oos_loss_values: list[float] | None = None,
    other_models: list[dict] | None = None,
    extra_summary_rows: list[dict] | None = None,
    register_model_in_cutoffs: bool = True,
    mcs_auc_point: float = 0.823,
) -> tuple[Path, Path, Path]:
    """Materialise (run_dir, eval_path, cutoffs_path).

    ``is_loss_values`` and ``oos_loss_values`` set the ``loss`` feature
    on each side of the cutoff. The other four features are filled with
    a deterministic ramp; analytical d-values live on ``loss``.
    """
    if is_loss_values is None:
        is_loss_values = [0.5, 0.5, 0.5, 0.5]
    if oos_loss_values is None:
        oos_loss_values = [0.0, 0.0, 0.0, 0.0]

    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)

    eval_rows, records = _make_primary_records(model, is_loss_values, oos_loss_values)
    if other_models:
        for entry in other_models:
            extra_eval, extra_recs = _make_other_model_records(entry)
            eval_rows.extend(extra_eval)
            records.extend(extra_recs)

    eval_path = tmp_path / "eval.jsonl"
    _write_eval_jsonl(eval_path, eval_rows)
    _write_records_jsonl(run_dir / "records.jsonl", records)

    summary_rows: list[dict] = [_summary_row(model, mcs_auc_point)]
    if other_models:
        for entry in other_models:
            summary_rows.append(_summary_row(
                entry["model"], entry.get("mcs_auc_point", 0.74),
                acc="0.51", acc_lo="0.46", acc_hi="0.56",
                auc_lo="0.69", auc_hi="0.79",
            ))
    if extra_summary_rows:
        summary_rows.extend(extra_summary_rows)
    _write_summary_csv(run_dir / "summary.csv", summary_rows)

    cutoffs: dict = {"models": {}}
    if register_model_in_cutoffs:
        cutoffs["models"][model] = cutoff_iso
    if other_models:
        for entry in other_models:
            if entry.get("register_in_cutoffs", True):
                cutoffs["models"][entry["model"]] = entry.get("cutoff_iso", cutoff_iso)
    cutoffs_path = tmp_path / "cutoffs.yaml"
    cutoffs_path.write_text(yaml.safe_dump(cutoffs))

    return run_dir, eval_path, cutoffs_path


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------


def test_known_d_two_class_distribution(tmp_path: Path) -> None:
    """Synthetic two-class data with analytically computable d=0.5.

    IS values: variance = 1 (equal-spaced around 0.5), n=4
    OOS values: variance = 1 (equal-spaced around 0.0), n=4
    pooled_std = sqrt(((4-1)*1 + (4-1)*1) / (4+4-2)) = sqrt(6/6) = 1.0
    cohens_d = (0.5 - 0.0) / 1.0 = 0.5
    """
    # Pick values whose sample mean is 0.5 and sample variance (ddof=1) = 1.0.
    # Mean centred at 0.5: [0.5 - a, 0.5 - b, 0.5 + b, 0.5 + a] with
    # 2*(a^2 + b^2) / 3 = 1 ⇒ pick a^2 + b^2 = 1.5; choose a=1.0, b=sqrt(0.5).
    import math as _m
    a, b = 1.0, _m.sqrt(0.5)
    is_vals = [0.5 - a, 0.5 - b, 0.5 + b, 0.5 + a]
    oos_vals = [0.0 - a, 0.0 - b, 0.0 + b, 0.0 + a]

    run_dir, eval_path, cutoffs_path = _build_fixture_run(
        tmp_path,
        is_loss_values=is_vals,
        oos_loss_values=oos_vals,
    )

    from src.portfolio.cohens_d import compute_cohens_d

    df = compute_cohens_d(run_dir, eval_path, cutoffs_path)

    assert isinstance(df, pd.DataFrame)
    loss_row = df[(df["model"] == "openai/gpt-oss-20b") & (df["feature"] == "loss")]
    assert len(loss_row) == 1
    assert loss_row.iloc[0]["n_is"] == 4
    assert loss_row.iloc[0]["n_oos"] == 4
    assert abs(loss_row.iloc[0]["mean_is"] - 0.5) < 1e-9
    assert abs(loss_row.iloc[0]["mean_oos"] - 0.0) < 1e-9
    assert abs(loss_row.iloc[0]["pooled_std"] - 1.0) < 1e-9
    assert abs(loss_row.iloc[0]["cohens_d"] - 0.5) < 1e-6
    assert loss_row.iloc[0]["note"] == ""
    assert abs(loss_row.iloc[0]["mcs_auc_holdout"] - 0.823) < 1e-6


def test_identical_classes_yield_zero_d(tmp_path: Path) -> None:
    """When IS and OOS distributions match, d ≈ 0."""
    import math as _m
    a, b = 1.0, _m.sqrt(0.5)
    same_vals = [0.5 - a, 0.5 - b, 0.5 + b, 0.5 + a]

    run_dir, eval_path, cutoffs_path = _build_fixture_run(
        tmp_path,
        is_loss_values=same_vals,
        oos_loss_values=same_vals,
    )

    from src.portfolio.cohens_d import compute_cohens_d
    df = compute_cohens_d(run_dir, eval_path, cutoffs_path)

    loss_row = df[(df["model"] == "openai/gpt-oss-20b") & (df["feature"] == "loss")]
    assert abs(loss_row.iloc[0]["cohens_d"]) < 1e-9
    assert loss_row.iloc[0]["note"] == ""


def test_zero_std_yields_nan_with_note(tmp_path: Path) -> None:
    """When pooled_std == 0 (all values identical on both sides), d = NaN."""
    run_dir, eval_path, cutoffs_path = _build_fixture_run(
        tmp_path,
        is_loss_values=[0.5, 0.5, 0.5, 0.5],
        oos_loss_values=[0.5, 0.5, 0.5, 0.5],
    )

    from src.portfolio.cohens_d import compute_cohens_d
    df = compute_cohens_d(run_dir, eval_path, cutoffs_path)

    loss_row = df[(df["model"] == "openai/gpt-oss-20b") & (df["feature"] == "loss")]
    assert math.isnan(loss_row.iloc[0]["cohens_d"])
    assert loss_row.iloc[0]["note"] == "insufficient samples"


def test_missing_model_in_cutoffs_skipped(tmp_path: Path, caplog) -> None:
    """A model present in records.jsonl but absent from cutoffs.yaml is skipped.

    The function must NOT raise, and the registered model must still appear
    in the output.
    """
    other = {
        "model": "openai/some-other-model",
        "register_in_cutoffs": False,
    }
    run_dir, eval_path, cutoffs_path = _build_fixture_run(
        tmp_path,
        other_models=[other],
    )

    from src.portfolio.cohens_d import compute_cohens_d

    with caplog.at_level("WARNING"):
        df = compute_cohens_d(run_dir, eval_path, cutoffs_path)

    models_in_output = set(df["model"].unique())
    assert "openai/gpt-oss-20b" in models_in_output
    assert "openai/some-other-model" not in models_in_output


def test_artifact_files_written(tmp_path: Path) -> None:
    """Both cohens_d.csv and cohens_d.md exist with the documented schema."""
    run_dir, eval_path, cutoffs_path = _build_fixture_run(tmp_path)

    from src.portfolio.cohens_d import compute_cohens_d
    df = compute_cohens_d(run_dir, eval_path, cutoffs_path)

    csv_path = run_dir / "cohens_d.csv"
    md_path = run_dir / "cohens_d.md"
    assert csv_path.exists()
    assert md_path.exists()

    expected_columns = [
        "model", "feature", "n_is", "n_oos",
        "mean_is", "mean_oos", "pooled_std",
        "cohens_d", "note", "mcs_auc_holdout",
    ]
    assert list(df.columns) == expected_columns

    # One row per (model, feature) for all five features for our single model.
    for feature in _FEATURE_NAMES:
        sub = df[(df["model"] == "openai/gpt-oss-20b") & (df["feature"] == feature)]
        assert len(sub) == 1, f"expected exactly 1 row for {feature}"

    md_text = md_path.read_text(encoding="utf-8")
    assert "openai/gpt-oss-20b" in md_text
    for feature in _FEATURE_NAMES:
        assert feature in md_text
