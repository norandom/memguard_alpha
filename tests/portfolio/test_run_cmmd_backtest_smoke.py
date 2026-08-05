"""Smoke test for ``scripts.run_cmmd_backtest`` (Task 3.2).

Exercises the orchestrator end-to-end on a 10-row miniature eval set
without touching the real NVIDIA / FMP endpoints. The test patches
``recall_guard.harness.runner.run`` to write a deterministic ``records.jsonl`` +
``summary.csv`` + base ``manifest.json`` (this is the simplified path
authorised by the task brief — the full harness pipeline is exercised
elsewhere). The FMP price fetcher is patched with a ``MagicMock`` that
returns a fixed pandas DataFrame.

Asserts every artifact promised by Task 3.2's observable lands in the
run directory and that the manifest's ``backtest`` block matches the
schema documented in ``design.md`` § Manifest extension.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

# ---------------------------------------------------------------------- helpers


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def _load_orchestrator() -> Any:
    """Lazy-import the orchestrator so the test file imports cleanly."""
    return importlib.import_module("run_cmmd_backtest")


def _compute_prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def _build_eval_set(path: Path, *, date_suffix: str = "") -> list[dict]:
    """Write a 10-row miniature eval set straddling 2024-07-01.

    ``date_suffix`` lets a test emit ISO datetimes ("T00:00:00") instead of
    plain dates to prove the normalization contract holds end-to-end.
    """
    rows: list[dict] = []
    # Five pre-cutoff days, five post-cutoff days; alternate tickers.
    pre_dates = ["2023-01-03", "2023-04-12", "2023-09-05", "2024-02-08", "2024-05-22"]
    post_dates = ["2024-08-12", "2024-10-01", "2025-01-15", "2025-03-20", "2025-06-04"]
    tickers = ["SWDA.L", "XLK", "IAU"]
    for i, day in enumerate(pre_dates + post_dates):
        ticker = tickers[i % len(tickers)]
        prompt = f"Question {i}: forecast {ticker} on {day}."
        rows.append({
            "prompt": prompt,
            "target_direction": (1 if i % 2 == 0 else -1),
            "metadata": {"ticker": ticker, "date": day + date_suffix},
        })
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return rows


def _build_records_jsonl(eval_rows: list[dict], records_path: Path) -> None:
    """Write a deterministic ``records.jsonl`` for one model.

    Every eval row produces a parse-OK record with a scripted Direction /
    Confidence and a varied ``p_memorized`` so the CMMD filter has
    something to drop. The minimum subset of MIA features needed for
    ``compute_cohens_d`` to populate the artifact is included.
    """
    records: list[dict] = []
    for i, row in enumerate(eval_rows):
        # Spread p_memorized over (0.05..0.95) so the 80th-percentile cut
        # has a meaningful threshold and at least one survivor.
        p_mem = 0.05 + 0.09 * i
        # Direction cycles through {-1, 0, 1}; confidences vary so the
        # weight matrix has some non-uniform rows.
        direction = (i % 3) - 1
        confidence = 0.40 + 0.05 * (i % 4)
        records.append({
            "model": "openai/gpt-oss-20b",
            "prompt_hash": _compute_prompt_hash(row["prompt"]),
            "parse_ok": True,
            "predicted_direction": direction,
            "raw_confidence": confidence,
            "penalized_confidence": confidence * (1.0 - p_mem),
            "target_direction": row["target_direction"],
            "features_raw": {
                "loss": 1.0 + 0.01 * i,
                "min_k": -2.0 - 0.05 * i,
                "min_k_pp": -0.5 + 0.02 * i,
                "zlib_ratio": 0.8 + 0.01 * i,
                "ref_delta": None,
            },
            "features_standardised": {
                "loss": 0.0, "min_k": 0.0, "min_k_pp": 0.0,
                "zlib_ratio": 0.0, "ref_delta": None,
            },
            "p_memorized": p_mem,
            "fail_reason": None,
            "raw_response_excerpt": None,
        })
    with records_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def _write_summary_csv(
    summary_path: Path,
    *,
    mcs_auc_point: float = 0.72,
    survives_gates: bool = True,
    warnings: str = "",
) -> None:
    """Write a deterministic ``summary.csv`` row for the signal model."""
    header = (
        "model,raw_acc_point,raw_acc_lo,raw_acc_hi,memguard_acc_point,"
        "memguard_acc_lo,memguard_acc_hi,mcs_auc_point,mcs_auc_lo,"
        "mcs_auc_hi,parse_success_rate,parse_failures,score,"
        "survives_gates,warnings"
    )
    row = (
        "openai/gpt-oss-20b,0.55,0.45,0.65,0.55,0.45,0.65,"
        f"{mcs_auc_point},0.65,0.79,1.0,0,0.42,"
        f"{'true' if survives_gates else 'false'},{warnings}"
    )
    summary_path.write_text(header + "\n" + row + "\n", encoding="utf-8")


def _write_base_manifest(run_dir: Path) -> None:
    """Write a minimal pre-existing manifest the orchestrator will extend."""
    payload = {
        "harness_version": "0.4.1",
        "seed": 0,
        "eval_set_hash": "fakehash" * 8,
        "control_corpus_hash": "fakehash" * 8,
        "is_memorized_hash": "fakehash" * 8,
        "cutoffs_hash": "fakehash" * 8,
        "shortlist": ["openai/gpt-oss-20b"],
        "composite_score": {"formula": "test"},
        "mcs_hyperparams": {},
        "bootstrap_n": 50,
        "artifacts": {
            "records": str(run_dir / "records.jsonl"),
            "summary": str(run_dir / "summary.csv"),
            "manifest": str(run_dir / "manifest.json"),
        },
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _make_fake_harness_run(
    eval_rows: list[dict],
    *,
    mcs_auc_point: float = 0.72,
    survives_gates: bool = True,
    warnings: str = "",
):
    """Return a stub callable matching ``harness.runner.run``'s signature.

    The stub writes ``records.jsonl``, ``summary.csv`` and the base
    ``manifest.json`` to ``args.out_dir`` and returns 0. This bypasses
    the real harness pipeline (LM HTTP, MCS calibrator) which is covered
    by the harness's own tests.
    """
    def fake_run(args, *, lm_factory=None):
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        _build_records_jsonl(eval_rows, out_dir / "records.jsonl")
        _write_summary_csv(
            out_dir / "summary.csv",
            mcs_auc_point=mcs_auc_point,
            survives_gates=survives_gates,
            warnings=warnings,
        )
        _write_base_manifest(out_dir)
        # top3.md is harmless if absent for the orchestrator, but the
        # base writer does emit it; we skip it here because the
        # orchestrator never reads it.
        return 0
    return fake_run


def _build_price_frame(eval_rows: list[dict]) -> pd.DataFrame:
    """Build a daily price frame covering [min_date, max_date] of the eval set.

    Linear price paths (≠ flat) so vectorbt's Sharpe / drawdown stats are
    finite and the bootstrap CIs are non-degenerate.
    """
    dates = sorted({date.fromisoformat(r["metadata"]["date"][:10]) for r in eval_rows})
    start, end = dates[0], dates[-1]
    idx = pd.date_range(start, end, freq="D")
    n = len(idx)
    return pd.DataFrame({
        "SWDA.L": [100.0 + 0.10 * i for i in range(n)],
        "XLK":    [100.0 + 0.05 * i for i in range(n)],
        "IAU":    [100.0 - 0.03 * i for i in range(n)],
        "BIL":    [100.0 + 0.001 * i for i in range(n)],
    }, index=idx)


# ----------------------------------------------------------------------- tests


def test_orchestrator_smoke_writes_every_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end smoke: every documented artifact lands in ``run_dir``.

    Patches:

    - ``recall_guard.harness.runner.run`` is replaced by a stub that writes a
      deterministic ``records.jsonl``, ``summary.csv`` and base
      ``manifest.json``.
    - ``fmp_fetcher`` is a ``MagicMock`` returning a fixed pandas frame.

    Asserts: ``records.jsonl``, ``cohens_d.csv``, ``is_oos_gap.csv``,
    ``backtest_summary.csv``, ``equity_curves.csv``, ``daily_returns.csv``,
    ``manifest.json`` all exist; the manifest's ``backtest`` block carries
    the documented schema keys.
    """
    orch = _load_orchestrator()

    # ---- Build the miniature eval set + supporting fixtures.
    eval_path = tmp_path / "eval" / "etf_portfolio.jsonl"
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    eval_rows = _build_eval_set(eval_path)

    is_path = tmp_path / "is.jsonl"
    is_path.write_text(
        json.dumps({"prompt": "p", "label": 1, "metadata": {}}) + "\n",
        encoding="utf-8",
    )
    oos_path = tmp_path / "oos.jsonl"
    oos_path.write_text(
        json.dumps({"prompt": "p", "label": 0, "metadata": {}}) + "\n",
        encoding="utf-8",
    )

    # Cutoffs YAML with the gpt-oss-20b cutoff so half of eval rows
    # land OOS. dump_all is overkill; one model is enough.
    cutoffs_path = tmp_path / "cutoffs.yaml"
    cutoffs_path.write_text(
        "models:\n  openai/gpt-oss-20b: 2024-06-30\n",
        encoding="utf-8",
    )

    run_dir = tmp_path / "run"
    fake_run = _make_fake_harness_run(eval_rows)

    # Patch the orchestrator's bound ``harness_run`` symbol.
    monkeypatch.setattr(orch, "harness_run", fake_run)

    fmp_fetcher = MagicMock(return_value=_build_price_frame(eval_rows))

    rc = orch.main(
        eval_path=eval_path,
        cutoffs_path=cutoffs_path,
        is_memorized_path=is_path,
        oos_control_path=oos_path,
        run_dir=run_dir,
        seed=0,
        bootstrap_n=64,
        fmp_fetcher=fmp_fetcher,
        lm_factory=None,
    )
    assert rc == 0, "orchestrator returned non-zero"

    # FMP fetcher was invoked exactly once with the universe.
    assert fmp_fetcher.call_count == 1
    args, _kwargs = fmp_fetcher.call_args
    assert list(args[0]) == ["SWDA.L", "XLK", "IAU", "BIL"]

    # ---- Required artifacts ------------------------------------------
    expected = [
        "records.jsonl",
        "cohens_d.csv",
        "is_oos_gap.csv",
        "backtest_summary.csv",
        "equity_curves.csv",
        "daily_returns.csv",
        "manifest.json",
    ]
    for name in expected:
        assert (run_dir / name).exists(), f"missing artifact: {name}"

    # ---- Manifest backtest block schema ------------------------------
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "backtest" in manifest, "manifest missing backtest block"
    bt = manifest["backtest"]
    expected_keys = {
        "signal_model", "universe", "cash_ticker", "cmmd_quantile",
        "cmmd_threshold_value", "fees_one_way", "init_cash", "seed",
        "bootstrap_n", "n_is_rows", "n_oos_rows", "artifacts",
    }
    assert expected_keys <= set(bt.keys()), (
        f"backtest block missing keys: {expected_keys - set(bt.keys())}"
    )
    assert bt["signal_model"] == "openai/gpt-oss-20b"
    assert bt["universe"] == ["SWDA.L", "XLK", "IAU", "BIL"]
    assert bt["cash_ticker"] == "BIL"
    assert bt["cmmd_quantile"] == pytest.approx(0.80)
    assert bt["fees_one_way"] == pytest.approx(0.00075)
    assert bt["init_cash"] == pytest.approx(1.0)
    assert bt["n_is_rows"] + bt["n_oos_rows"] == len(eval_rows)
    # Per-row IS/OOS provenance (review-hardening Req 3.5).
    labels = bt["is_oos_by_prompt_hash"]
    assert len(labels) == len(eval_rows)
    assert set(labels.values()) == {"IS", "OOS"}
    assert sum(1 for v in labels.values() if v == "IS") == bt["n_is_rows"]
    expected_hashes = {_compute_prompt_hash(r["prompt"]) for r in eval_rows}
    assert set(labels.keys()) == expected_hashes
    arts = bt["artifacts"]
    for required in (
        "backtest_summary_csv", "equity_curves_csv", "equity_curves_png",
        "daily_returns_csv",
    ):
        assert required in arts, f"backtest.artifacts missing {required}"


def test_orchestrator_accepts_iso_datetime_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ISO datetime metadata.date values ("...T00:00:00") must flow through
    the precheck, gap analyzer, and backtest without crashing (Req 3.3)."""
    orch = _load_orchestrator()

    eval_path = tmp_path / "eval" / "etf_portfolio.jsonl"
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    eval_rows = _build_eval_set(eval_path, date_suffix="T00:00:00")

    is_path = tmp_path / "is.jsonl"
    is_path.write_text(
        json.dumps({"prompt": "p", "label": 1, "metadata": {}}) + "\n",
        encoding="utf-8",
    )
    oos_path = tmp_path / "oos.jsonl"
    oos_path.write_text(
        json.dumps({"prompt": "p", "label": 0, "metadata": {}}) + "\n",
        encoding="utf-8",
    )
    cutoffs_path = tmp_path / "cutoffs.yaml"
    cutoffs_path.write_text(
        "models:\n  openai/gpt-oss-20b: 2024-06-30\n",
        encoding="utf-8",
    )

    run_dir = tmp_path / "run"
    monkeypatch.setattr(orch, "harness_run", _make_fake_harness_run(eval_rows))
    fmp_fetcher = MagicMock(return_value=_build_price_frame(eval_rows))

    rc = orch.main(
        eval_path=eval_path,
        cutoffs_path=cutoffs_path,
        is_memorized_path=is_path,
        oos_control_path=oos_path,
        run_dir=run_dir,
        seed=0,
        bootstrap_n=64,
        fmp_fetcher=fmp_fetcher,
        lm_factory=None,
    )
    assert rc == 0

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    bt = manifest["backtest"]
    # The datetime forms classify identically to their plain-date twins.
    assert bt["n_is_rows"] == 5
    assert bt["n_oos_rows"] == 5
    assert (run_dir / "is_oos_gap.csv").exists()
    assert (run_dir / "backtest_summary.csv").exists()


def test_orchestrator_warns_on_one_sided_eval_span(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture,
) -> None:
    """An OOS-only eval span must emit the 'nothing to remove' warning and
    still complete the run (Req 4.6)."""
    orch = _load_orchestrator()

    eval_path = tmp_path / "eval" / "etf_portfolio.jsonl"
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    eval_rows = _build_eval_set(eval_path)

    is_path = tmp_path / "is.jsonl"
    is_path.write_text(
        json.dumps({"prompt": "p", "label": 1, "metadata": {}}) + "\n",
        encoding="utf-8",
    )
    oos_path = tmp_path / "oos.jsonl"
    oos_path.write_text(
        json.dumps({"prompt": "p", "label": 0, "metadata": {}}) + "\n",
        encoding="utf-8",
    )
    # A cutoff far in the past makes every eval row OOS.
    cutoffs_path = tmp_path / "cutoffs.yaml"
    cutoffs_path.write_text(
        "models:\n  openai/gpt-oss-20b: 2020-01-01\n",
        encoding="utf-8",
    )

    run_dir = tmp_path / "run"
    monkeypatch.setattr(orch, "harness_run", _make_fake_harness_run(eval_rows))
    fmp_fetcher = MagicMock(return_value=_build_price_frame(eval_rows))

    rc = orch.main(
        eval_path=eval_path,
        cutoffs_path=cutoffs_path,
        is_memorized_path=is_path,
        oos_control_path=oos_path,
        run_dir=run_dir,
        seed=0,
        bootstrap_n=64,
        fmp_fetcher=fmp_fetcher,
        lm_factory=None,
    )
    assert rc == 0

    captured = capfd.readouterr()
    assert "only OOS rows" in captured.err
    assert "no cross-regime rows to remove" in captured.err

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["backtest"]["n_is_rows"] == 0
    assert manifest["backtest"]["n_oos_rows"] == len(eval_rows)


def test_orchestrator_aborts_on_weak_calibration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    orch = _load_orchestrator()

    eval_path = tmp_path / "eval" / "etf_portfolio.jsonl"
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    eval_rows = _build_eval_set(eval_path)

    is_path = tmp_path / "is.jsonl"
    is_path.write_text(
        json.dumps({"prompt": "p", "label": 1, "metadata": {}}) + "\n",
        encoding="utf-8",
    )
    oos_path = tmp_path / "oos.jsonl"
    oos_path.write_text(
        json.dumps({"prompt": "p", "label": 0, "metadata": {}}) + "\n",
        encoding="utf-8",
    )
    cutoffs_path = tmp_path / "cutoffs.yaml"
    cutoffs_path.write_text(
        "models:\n  openai/gpt-oss-20b: 2024-06-30\n",
        encoding="utf-8",
    )

    run_dir = tmp_path / "run"
    fake_run = _make_fake_harness_run(
        eval_rows,
        mcs_auc_point=0.55,
        survives_gates=False,
        warnings="weak-calibration",
    )
    monkeypatch.setattr(orch, "harness_run", fake_run)

    fmp_fetcher = MagicMock(return_value=_build_price_frame(eval_rows))
    rc = orch.main(
        eval_path=eval_path,
        cutoffs_path=cutoffs_path,
        is_memorized_path=is_path,
        oos_control_path=oos_path,
        run_dir=run_dir,
        seed=0,
        bootstrap_n=64,
        fmp_fetcher=fmp_fetcher,
        lm_factory=None,
    )

    assert rc == orch.EXIT_MCS_UNCALIBRATED
    assert fmp_fetcher.call_count == 0
    assert not (run_dir / "backtest_summary.csv").exists()
