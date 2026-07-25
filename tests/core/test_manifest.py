"""Tests for recall_guard.core.manifest: run-manifest write/read + file hashing.

Covers requirements 6.5, 8.4, 10.1, 10.2 from the honest-model-ranking spec:
- Manifest persists seed, hashes of inputs, shortlist, composite score formula
  + weights, MCS hyperparameters, bootstrap_n, and artifact paths so a run is
  reproducible.
- ``read(write(m)) == m`` round-trip semantics.
- ``compute_file_hash`` returns a sha256 hex digest, chunk-read so it tolerates
  files larger than memory-friendly buffer sizes.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from recall_guard.core.manifest import (
    Manifest,
    compute_file_hash,
    read_manifest,
    write_manifest,
)


def _sample_manifest() -> Manifest:
    """Construct a populated Manifest for round-trip and human-readable tests."""
    return Manifest(
        harness_version="0.1.2",
        seed=42,
        eval_set_hash="a" * 64,
        control_corpus_hash="b" * 64,
        is_memorized_hash="c" * 64,
        cutoffs_hash="d" * 64,
        shortlist=["meta/llama-3.2-1b-instruct", "nvidia/nemotron-3-super-120b-a12b"],
        composite_score={
            "formula": "memguard_acc_lo * mcs_auc_point * parse_success_rate",
            "weights": {"memguard_acc_lo": 1.0, "mcs_auc_point": 1.0, "parse_success_rate": 1.0},
        },
        mcs_hyperparams={"min_auc": 0.6, "solver": "liblinear", "class_weight": "balanced"},
        bootstrap_n=1000,
        artifacts={
            "shortlist": "runs/2026-04-27/shortlist.json",
            "records": "runs/2026-04-27/records.jsonl",
            "summary": "runs/2026-04-27/summary.csv",
            "top3": "runs/2026-04-27/top3.md",
        },
    )


def test_manifest_round_trip(tmp_path: Path) -> None:
    """read_manifest(write_manifest(out_dir, m)) == m (Req 10.1, 10.2)."""
    manifest = _sample_manifest()

    written_path = write_manifest(tmp_path, manifest)
    loaded = read_manifest(written_path)

    assert loaded == manifest
    # The written file must be exactly manifest.json under out_dir.
    assert written_path == tmp_path / "manifest.json"
    assert written_path.is_file()


def test_manifest_json_is_human_readable(tmp_path: Path) -> None:
    """Written manifest.json must be valid JSON whose top-level keys match
    the dataclass field names (Req 10.1: human-readable manifest)."""
    manifest = _sample_manifest()
    path = write_manifest(tmp_path, manifest)

    raw = path.read_text(encoding="utf-8")
    decoded = json.loads(raw)

    expected_keys = {
        "harness_version",
        "seed",
        "eval_set_hash",
        "control_corpus_hash",
        "is_memorized_hash",
        "cutoffs_hash",
        "shortlist",
        "composite_score",
        "mcs_hyperparams",
        "bootstrap_n",
        "artifacts",
    }
    assert set(decoded.keys()) == expected_keys
    # Sanity: a few values match what we wrote.
    assert decoded["seed"] == 42
    assert decoded["bootstrap_n"] == 1000
    assert decoded["composite_score"]["formula"].startswith("memguard_acc_lo")


def test_write_manifest_creates_out_dir(tmp_path: Path) -> None:
    """write_manifest must create a missing out_dir (Req 10.1)."""
    nested = tmp_path / "nested" / "run1"
    assert not nested.exists()

    written = write_manifest(nested, _sample_manifest())

    assert nested.is_dir()
    assert written == nested / "manifest.json"
    assert written.is_file()


def test_read_manifest_rejects_missing_keys(tmp_path: Path) -> None:
    """read_manifest must raise ValueError naming the missing key (Req 10.1)."""
    manifest = _sample_manifest()
    path = write_manifest(tmp_path, manifest)

    decoded = json.loads(path.read_text(encoding="utf-8"))
    decoded.pop("seed")
    path.write_text(json.dumps(decoded, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        read_manifest(path)
    assert "seed" in str(excinfo.value)


def test_read_manifest_rejects_extra_keys(tmp_path: Path) -> None:
    """Extra unknown keys must be rejected so manifests cannot drift silently."""
    manifest = _sample_manifest()
    path = write_manifest(tmp_path, manifest)

    decoded = json.loads(path.read_text(encoding="utf-8"))
    decoded["unexpected_field"] = "drift"
    path.write_text(json.dumps(decoded, indent=2, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        read_manifest(path)
    assert "unexpected_field" in str(excinfo.value)


def test_compute_file_hash_is_sha256(tmp_path: Path) -> None:
    """compute_file_hash returns sha256 hex digest of file bytes (Req 10.1)."""
    payload = b"the quick brown fox jumps over the lazy dog"
    fp = tmp_path / "small.bin"
    fp.write_bytes(payload)

    expected = hashlib.sha256(payload).hexdigest()
    assert compute_file_hash(fp) == expected


def test_compute_file_hash_chunks_large_file(tmp_path: Path) -> None:
    """Hashing a >16KB file matches the stdlib reference (proves chunked read)."""
    # 64KB of varied bytes; comfortably exceeds an 8KB chunk size.
    payload = bytes((i * 37) % 256 for i in range(64 * 1024))
    fp = tmp_path / "large.bin"
    fp.write_bytes(payload)

    expected = hashlib.sha256(payload).hexdigest()
    assert compute_file_hash(fp) == expected


# --- cmmd-backtest manifest extension (Req 7.5, 8.2) -------------------------

# The pre-existing on-disk schema. The cmmd-backtest manifest extension
# (Req 7.5, 8.2) MUST be additive: when no backtest block is supplied, the
# serialised JSON must contain exactly these keys, byte-for-byte unchanged
# from the previous schema, and round-trip equality must still hold.
_PREEXISTING_MANIFEST_KEYS: set[str] = {
    "harness_version",
    "seed",
    "eval_set_hash",
    "control_corpus_hash",
    "is_memorized_hash",
    "cutoffs_hash",
    "shortlist",
    "composite_score",
    "mcs_hyperparams",
    "bootstrap_n",
    "artifacts",
}


def _backtest_block() -> dict:
    """Build a representative backtest block matching design.md § Manifest extension."""
    return {
        "signal_model": "openai/gpt-oss-20b",
        "universe": ["SWDA.L", "XLK", "IAU", "BIL"],
        "cash_ticker": "BIL",
        "cmmd_quantile": 0.80,
        "cmmd_threshold_value": 0.873,
        "fees_one_way": 0.00075,
        "init_cash": 1.0,
        "seed": 0,
        "bootstrap_n": 1000,
        "n_is_rows": 156,
        "n_oos_rows": 178,
        "artifacts": {
            "backtest_summary_csv": "backtest_summary.csv",
            "equity_curves_csv": "equity_curves.csv",
            "equity_curves_png": "equity_curves.png",
            "daily_returns_csv": "daily_returns.csv",
            "cohens_d_csv": "cohens_d.csv",
            "is_oos_gap_csv": "is_oos_gap.csv",
        },
    }


def test_manifest_round_trip_without_backtest_matches_existing_schema(
    tmp_path: Path,
) -> None:
    """Without a backtest block, the on-disk JSON must keep the pre-existing schema.

    The cmmd-backtest extension is additive (Req 7.5, 8.2). For all runs that
    do not opt into the backtest block, ``manifest.json`` must serialise to
    exactly the 11-key shape that pre-cmmd-backtest harness runs produced —
    no extra ``"backtest": null`` entry, no schema drift — and round-trip
    equality must still hold.
    """
    manifest = _sample_manifest()
    # Sanity: the default backtest field is absent.
    assert getattr(manifest, "backtest", None) is None

    path = write_manifest(tmp_path, manifest)
    decoded = json.loads(path.read_text(encoding="utf-8"))

    assert set(decoded.keys()) == _PREEXISTING_MANIFEST_KEYS
    assert "backtest" not in decoded

    # And the round-trip still gives back an equal Manifest.
    assert read_manifest(path) == manifest


def test_manifest_round_trip_with_backtest_block(tmp_path: Path) -> None:
    """When a backtest block is passed, every required key round-trips through JSON.

    Covers Requirements 7.5 (manifest extension records all listed fields)
    and 8.2 (every run is reproducible from the manifest alone).
    """
    backtest = _backtest_block()
    manifest = Manifest(
        harness_version="0.1.2",
        seed=0,
        eval_set_hash="a" * 64,
        control_corpus_hash="b" * 64,
        is_memorized_hash="c" * 64,
        cutoffs_hash="d" * 64,
        shortlist=["openai/gpt-oss-20b"],
        composite_score={"formula": "f", "weights": None},
        mcs_hyperparams={"min_auc": 0.6},
        bootstrap_n=1000,
        artifacts={"records": "runs/x/records.jsonl"},
        backtest=backtest,
    )

    path = write_manifest(tmp_path, manifest)
    decoded = json.loads(path.read_text(encoding="utf-8"))

    # On-disk schema: pre-existing keys + backtest.
    assert set(decoded.keys()) == _PREEXISTING_MANIFEST_KEYS | {"backtest"}

    # Every required key from design.md § Manifest extension must round-trip.
    required_block_keys = {
        "signal_model",
        "universe",
        "cash_ticker",
        "cmmd_quantile",
        "cmmd_threshold_value",
        "fees_one_way",
        "init_cash",
        "seed",
        "bootstrap_n",
        "n_is_rows",
        "n_oos_rows",
        "artifacts",
    }
    assert set(decoded["backtest"].keys()) == required_block_keys

    # Whole-block equality: dict round-trips byte-for-byte.
    assert decoded["backtest"] == backtest

    # And the dataclass round-trip preserves the block.
    loaded = read_manifest(path)
    assert loaded.backtest == backtest
    assert loaded == manifest
