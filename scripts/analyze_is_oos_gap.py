"""Compute per-model IS / OOS accuracy gap from a finished harness run.

The honest-model-ranking harness writes per-(model, prompt) records to
``records.jsonl``, including the prompt_hash, parse status, predicted
direction, and target direction. This script joins those records against
the eval set (to recover each prompt's date) and the cutoffs registry
(to label each row IS or OOS per model), then emits the **memorization
gap** per model:

    gap = IS_accuracy - OOS_accuracy

Following the paper's design: a large positive gap means the model
performs better on prompts it could have memorized than on prompts it
couldn't, which is the memorization signature. A small / zero gap with
healthy OOS accuracy is what we want.

Usage::

    uv run python scripts/analyze_is_oos_gap.py <run_dir> [eval_set] [cutoffs]

Defaults:
    eval_set = data/eval/etf_direction_multiyear.jsonl
    cutoffs  = data/cutoffs.yaml

Writes ``<run_dir>/is_oos_gap.csv`` and ``<run_dir>/is_oos_gap.md``.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

# Make the repository root importable regardless of where the script runs from.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import yaml

from recall_guard.core.bootstrap import bootstrap_ci


def compute_prompt_hash(prompt: str) -> str:
    """Match the evaluator's hash convention (sha256 hex, first 16 chars)."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def load_eval_metadata(eval_path: Path) -> dict[str, dict]:
    """Map prompt_hash -> row metadata for date lookup."""
    metadata: dict[str, dict] = {}
    with eval_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "_cutoff_date" in row and "prompt" not in row:
                continue
            ph = compute_prompt_hash(row["prompt"])
            metadata[ph] = row.get("metadata") or {}
    return metadata


def _accuracy_ci(records: list[dict], seed: int = 0) -> tuple[float, float, float] | None:
    parse_ok = [r for r in records if r.get("parse_ok")]
    if not parse_ok:
        return None
    indicators = [
        1.0 if r.get("predicted_direction") == r.get("target_direction") else 0.0
        for r in parse_ok
    ]
    return bootstrap_ci(indicators, statistic=lambda xs: sum(xs) / len(xs), n_resamples=1000, seed=seed)


def analyze(run_dir: Path, eval_path: Path, cutoffs_path: Path) -> dict:
    metadata = load_eval_metadata(eval_path)

    with cutoffs_path.open() as f:
        cutoffs_raw = yaml.safe_load(f)["models"]
    cutoffs: dict[str, date] = {}
    for k, v in cutoffs_raw.items():
        if isinstance(v, date):
            cutoffs[k] = v
        else:
            cutoffs[k] = date.fromisoformat(str(v))

    records_by_model: dict[str, list[dict]] = {}
    with (run_dir / "records.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            records_by_model.setdefault(r["model"], []).append(r)

    rows: list[dict] = []
    unmatched_total = 0
    for model, recs in records_by_model.items():
        cutoff = cutoffs.get(model)
        if cutoff is None:
            print(f"WARN: {model} not in cutoffs registry, skipping")
            continue

        is_recs: list[dict] = []
        oos_recs: list[dict] = []
        unmatched = 0
        for r in recs:
            md = metadata.get(r.get("prompt_hash", ""))
            if not md or "date" not in md:
                unmatched += 1
                continue
            row_date = date.fromisoformat(md["date"])
            (is_recs if row_date <= cutoff else oos_recs).append(r)

        unmatched_total += unmatched

        is_ci = _accuracy_ci(is_recs)
        oos_ci = _accuracy_ci(oos_recs)
        gap = (
            (is_ci[0] - oos_ci[0])
            if is_ci is not None and oos_ci is not None
            else None
        )

        rows.append({
            "model": model,
            "cutoff": cutoff.isoformat(),
            "n_is": len(is_recs),
            "n_oos": len(oos_recs),
            "n_unmatched": unmatched,
            "is_acc_point": is_ci[0] if is_ci else None,
            "is_acc_lo": is_ci[1] if is_ci else None,
            "is_acc_hi": is_ci[2] if is_ci else None,
            "oos_acc_point": oos_ci[0] if oos_ci else None,
            "oos_acc_lo": oos_ci[1] if oos_ci else None,
            "oos_acc_hi": oos_ci[2] if oos_ci else None,
            "gap_point": gap,
        })

    return {"rows": rows, "unmatched_total": unmatched_total}


def write_csv(out_path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def write_md(out_path: Path, rows: list[dict], run_dir: Path) -> None:
    def fmt(v: float | None, places: int = 3) -> str:
        return "—" if v is None else f"{v:.{places}f}"

    lines = [
        f"# IS/OOS Memorization Gap — {run_dir.name}",
        "",
        "Per-model accuracy split by whether each eval row's date is before",
        "(IS) or after (OOS) that model's training cutoff. The **gap** column",
        "is `IS_accuracy − OOS_accuracy`. A large positive gap is the paper's",
        "memorization signature: the model performs better on data it could",
        "have memorized than on data it couldn't.",
        "",
        "| Model | Cutoff | n_IS | n_OOS | IS Acc (95% CI) | OOS Acc (95% CI) | Gap |",
        "| --- | --- | ---: | ---: | --- | --- | ---: |",
    ]
    for r in sorted(rows, key=lambda x: x.get("gap_point") or 0, reverse=True):
        is_ci = (
            f"{fmt(r['is_acc_point'])} [{fmt(r['is_acc_lo'])}–{fmt(r['is_acc_hi'])}]"
            if r["is_acc_point"] is not None else "—"
        )
        oos_ci = (
            f"{fmt(r['oos_acc_point'])} [{fmt(r['oos_acc_lo'])}–{fmt(r['oos_acc_hi'])}]"
            if r["oos_acc_point"] is not None else "—"
        )
        lines.append(
            f"| `{r['model']}` | {r['cutoff']} | {r['n_is']} | {r['n_oos']} | "
            f"{is_ci} | {oos_ci} | {fmt(r['gap_point'])} |"
        )
    lines.append("")
    lines.append(
        "_Method: each (model, eval row) pair is labelled IS if the row's "
        "date is on or before the model's training cutoff, OOS otherwise. "
        "Accuracy uses parse-OK rows only. Bootstrap 95% CIs use 1000 resamples._"
    )
    out_path.write_text("\n".join(lines) + "\n")


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: analyze_is_oos_gap.py <run_dir> [eval_set] [cutoffs]", file=sys.stderr)
        return 2
    run_dir = Path(argv[1])
    eval_path = Path(argv[2]) if len(argv) > 2 else Path("data/eval/etf_direction_multiyear.jsonl")
    cutoffs_path = Path(argv[3]) if len(argv) > 3 else Path("data/cutoffs.yaml")

    if not (run_dir / "records.jsonl").exists():
        print(f"missing {run_dir / 'records.jsonl'}", file=sys.stderr)
        return 2

    result = analyze(run_dir, eval_path, cutoffs_path)
    rows = result["rows"]
    if result["unmatched_total"]:
        print(f"WARN: {result['unmatched_total']} records had prompt_hashes not found in {eval_path}")

    csv_path = run_dir / "is_oos_gap.csv"
    md_path = run_dir / "is_oos_gap.md"
    write_csv(csv_path, rows)
    write_md(md_path, rows, run_dir)
    print(f"wrote {csv_path}")
    print(f"wrote {md_path}")

    # Console summary
    for r in sorted(rows, key=lambda x: x.get("gap_point") or 0, reverse=True):
        gap = r["gap_point"]
        gap_s = f"{gap:+.3f}" if gap is not None else "—"
        is_p = f"{r['is_acc_point']:.3f}" if r["is_acc_point"] is not None else "—"
        oos_p = f"{r['oos_acc_point']:.3f}" if r["oos_acc_point"] is not None else "—"
        print(f"  {r['model']:50s}  IS={is_p} (n={r['n_is']:3d})  OOS={oos_p} (n={r['n_oos']:3d})  gap={gap_s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
