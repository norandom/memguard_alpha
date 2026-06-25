"""Per-(model, MIA-feature) Cohen's d artifact for cmmd-backtest.

Implements requirements 1.1, 1.2, 1.3, 1.4, 1.5, and 9.1 of cmmd-backtest:
read a finished harness run's ``records.jsonl``, split each model's
parse-OK rows into IS / OOS by joining ``metadata.date`` against the
model's training cutoff, and compute Cohen's d on the raw (non-
standardised) value of every MIA feature. Writes ``cohens_d.csv`` and
``cohens_d.md`` into the run directory.

Design deviation: the design's ``compute_cohens_d`` signature lists only
``(run_dir, cutoffs_path)``, but ``records.jsonl`` carries ``prompt_hash``
rather than ``metadata.date`` (see ``recall_guard.harness.evaluator.Record``). To
recover the date we have to join records back to the eval set on
``prompt_hash``, so the public signature accepts ``eval_path``
explicitly. The orchestrator (task 3.2) supplies it.

Sentrux boundaries:

- Reads ``records.jsonl`` shape produced by ``recall_guard.harness.report``;
  matches the harness's own ``_hash_prompt`` convention (sha256 hex,
  first 16 chars). Reproduced locally to keep the portfolio layer free
  of upward imports.
- ``portfolio ↔ dataset`` and ``portfolio ↔ mia`` are explicitly
  forbidden by ``.sentrux/rules.toml``; this module imports neither.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


#: Five MIA feature names as serialised into ``records.jsonl`` by the
#: harness's ``_record_to_jsonable``. The artifact's ``feature`` column
#: uses these literal names — the design-time table referred to
#: ``min_k_pct`` but the harness-side dataclass field is ``min_k``.
_FEATURE_NAMES: tuple[str, ...] = (
    "loss",
    "min_k",
    "min_k_pp",
    "zlib_ratio",
    "ref_delta",
)

#: Stable column order for ``cohens_d.csv``. Matches the design's
#: § Cohen's d artifact schema.
_COLUMNS: list[str] = [
    "model",
    "feature",
    "n_is",
    "n_oos",
    "mean_is",
    "mean_oos",
    "pooled_std",
    "cohens_d",
    "note",
    "mcs_auc_holdout",
]

_INSUFFICIENT_NOTE: str = "insufficient samples"


# ----------------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------------


def _compute_prompt_hash(prompt: str) -> str:
    """Reproduce ``recall_guard.harness.evaluator._hash_prompt`` (sha256 hex, 16 chars).

    Reproduced locally rather than imported because importing across the
    harness↔portfolio direction would ripple a dependency on
    ``recall_guard.harness`` into the portfolio layer for no functional gain.
    """
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def _load_eval_metadata(eval_path: Path) -> dict[str, dict]:
    """Map ``prompt_hash -> metadata`` for date / ticker lookup.

    Skips ``_cutoff_date``-only sentinel rows (some eval-set files prepend
    one for OOS guards) — same handling as ``scripts/analyze_is_oos_gap``.
    """
    metadata: dict[str, dict] = {}
    with eval_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "_cutoff_date" in row and "prompt" not in row:
                continue
            prompt = row.get("prompt")
            if not isinstance(prompt, str):
                continue
            ph = _compute_prompt_hash(prompt)
            metadata[ph] = row.get("metadata") or {}
    return metadata


def _load_cutoffs(cutoffs_path: Path) -> dict[str, date]:
    """Parse ``data/cutoffs.yaml`` into ``model_id -> date``."""
    raw = yaml.safe_load(cutoffs_path.read_text(encoding="utf-8"))
    models_block = (raw or {}).get("models") or {}
    cutoffs: dict[str, date] = {}
    for k, v in models_block.items():
        if isinstance(v, date):
            cutoffs[k] = v
        else:
            cutoffs[k] = date.fromisoformat(str(v))
    return cutoffs


def _load_summary_auc(summary_path: Path) -> dict[str, float]:
    """Map ``model -> mcs_auc_point`` from ``summary.csv``.

    The harness writes per-model holdout AUC into the ``mcs_auc_point``
    column (see ``recall_guard.harness.report.SUMMARY_CSV_COLUMNS``). The artifact
    column is named ``mcs_auc_holdout`` per the design schema.
    """
    auc_by_model: dict[str, float] = {}
    if not summary_path.exists():
        return auc_by_model
    with summary_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            model = row.get("model", "").strip()
            if not model or model.startswith("__"):
                # Skip the ``__majority_baseline__`` sentinel and any
                # blank-model rows.
                continue
            raw = row.get("mcs_auc_point", "")
            try:
                auc_by_model[model] = float(raw)
            except (TypeError, ValueError):
                # Leave unset; downstream code surfaces NaN for the cell.
                continue
    return auc_by_model


def _stream_records(records_path: Path):
    """Yield JSON records one at a time from ``records.jsonl``."""
    with records_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _collect_feature_splits(
    records_path: Path,
    metadata_by_hash: dict[str, dict],
    cutoffs: dict[str, date],
) -> dict[tuple[str, str], dict[str, list[float]]]:
    """Bucket per-(model, feature) raw values into IS / OOS lists.

    Returns a nested dict::

        {(model, feature): {"is": [...], "oos": [...]}}

    Records whose model has no cutoff entry in ``cutoffs`` are skipped
    here with a single warning per missing model. parse-OK rows whose
    ``prompt_hash`` is absent from ``metadata_by_hash`` (or whose
    metadata lacks ``date``) are dropped silently — they cannot be
    labelled IS / OOS without a date.
    """
    buckets: dict[tuple[str, str], dict[str, list[float]]] = {}
    warned_missing: set[str] = set()
    seen_models: set[str] = set()

    for rec in _stream_records(records_path):
        model = rec.get("model")
        if not model:
            continue
        seen_models.add(model)
        if model not in cutoffs:
            if model not in warned_missing:
                logger.warning(
                    "cohens_d: model %r missing from cutoffs registry; skipping",
                    model,
                )
                warned_missing.add(model)
            continue
        if not rec.get("parse_ok"):
            continue
        features_raw = rec.get("features_raw")
        if not isinstance(features_raw, dict):
            continue

        prompt_hash = rec.get("prompt_hash") or ""
        md = metadata_by_hash.get(prompt_hash)
        if not md or "date" not in md:
            continue
        try:
            row_date = date.fromisoformat(str(md["date"])[:10])
        except ValueError:
            continue
        side = "is" if row_date <= cutoffs[model] else "oos"

        for feature in _FEATURE_NAMES:
            value = features_raw.get(feature)
            if value is None:
                continue
            try:
                value_f = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value_f):
                continue
            key = (model, feature)
            slot = buckets.setdefault(key, {"is": [], "oos": []})
            slot[side].append(value_f)

    # Make sure every (model, feature) pair appears even when one side
    # has zero samples — the artifact must include each of the five
    # features per model with a clear "insufficient samples" note rather
    # than dropping the row.
    for model in seen_models:
        if model not in cutoffs:
            continue
        for feature in _FEATURE_NAMES:
            buckets.setdefault((model, feature), {"is": [], "oos": []})

    return buckets


def _cohens_d_row(
    model: str,
    feature: str,
    is_values: list[float],
    oos_values: list[float],
    mcs_auc: float | None,
) -> dict:
    """Compute one DataFrame row for ``(model, feature)``.

    Implements the design's pooled-std formula and the Req 1.3 fallback:
    when either subset has < 2 valid samples or ``pooled_std == 0``,
    emit ``cohens_d = NaN`` with ``note = "insufficient samples"``.
    """
    n_is = len(is_values)
    n_oos = len(oos_values)

    if n_is < 2 or n_oos < 2:
        mean_is = float(np.mean(is_values)) if n_is > 0 else float("nan")
        mean_oos = float(np.mean(oos_values)) if n_oos > 0 else float("nan")
        return {
            "model": model,
            "feature": feature,
            "n_is": n_is,
            "n_oos": n_oos,
            "mean_is": mean_is,
            "mean_oos": mean_oos,
            "pooled_std": float("nan"),
            "cohens_d": float("nan"),
            "note": _INSUFFICIENT_NOTE,
            "mcs_auc_holdout": (
                float(mcs_auc) if mcs_auc is not None else float("nan")
            ),
        }

    is_arr = np.asarray(is_values, dtype=np.float64)
    oos_arr = np.asarray(oos_values, dtype=np.float64)
    mean_is = float(is_arr.mean())
    mean_oos = float(oos_arr.mean())
    var_is = float(np.var(is_arr, ddof=1))
    var_oos = float(np.var(oos_arr, ddof=1))
    pooled_std_sq = (
        ((n_is - 1) * var_is + (n_oos - 1) * var_oos) / (n_is + n_oos - 2)
    )
    if pooled_std_sq < 0:
        # Numerical noise can push a true-zero variance slightly negative.
        pooled_std_sq = 0.0
    pooled_std = math.sqrt(pooled_std_sq)

    if pooled_std == 0.0:
        return {
            "model": model,
            "feature": feature,
            "n_is": n_is,
            "n_oos": n_oos,
            "mean_is": mean_is,
            "mean_oos": mean_oos,
            "pooled_std": 0.0,
            "cohens_d": float("nan"),
            "note": _INSUFFICIENT_NOTE,
            "mcs_auc_holdout": (
                float(mcs_auc) if mcs_auc is not None else float("nan")
            ),
        }

    cohens_d = (mean_is - mean_oos) / pooled_std
    return {
        "model": model,
        "feature": feature,
        "n_is": n_is,
        "n_oos": n_oos,
        "mean_is": mean_is,
        "mean_oos": mean_oos,
        "pooled_std": pooled_std,
        "cohens_d": cohens_d,
        "note": "",
        "mcs_auc_holdout": (
            float(mcs_auc) if mcs_auc is not None else float("nan")
        ),
    }


# ----------------------------------------------------------------------------
# Markdown writer
# ----------------------------------------------------------------------------


def _format_float(value: float, places: int = 4) -> str:
    """Render a float for Markdown output; em-dash for NaN."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value:.{places}f}"


def _render_markdown(df: pd.DataFrame, run_dir: Path) -> str:
    """Render the per-model Markdown twin of ``cohens_d.csv``."""
    lines: list[str] = [
        f"# Cohen's d per (model, MIA feature) — {run_dir.name}",
        "",
        "Per-(model, feature) Cohen's d on the raw MIA-feature distributions",
        "split IS vs OOS by each model's training cutoff. The pooled-standard-",
        "deviation denominator is",
        "`sqrt(((n_is-1)*var_is + (n_oos-1)*var_oos) / (n_is+n_oos-2))`.",
        "",
        "Each model's combined holdout MCS-AUC (`mcs_auc_holdout`) is shown",
        "alongside so a reader can compare the composite to the best single",
        "feature.",
        "",
    ]
    if df.empty:
        lines.append("_No rows produced — see logger output for skipped models._")
        return "\n".join(lines) + "\n"

    for model in sorted(df["model"].unique()):
        sub = df[df["model"] == model].sort_values("feature")
        # mcs_auc_holdout is the same for every row of a given model; pull
        # the first non-NaN value for the heading.
        mcs_auc = float("nan")
        if not sub.empty:
            for v in sub["mcs_auc_holdout"]:
                if not (isinstance(v, float) and math.isnan(v)):
                    mcs_auc = float(v)
                    break

        lines.append(f"## `{model}` (mcs_auc_holdout = {_format_float(mcs_auc)})")
        lines.append("")
        lines.append(
            "| feature | n_is | n_oos | mean_is | mean_oos | pooled_std | cohens_d | note |"
        )
        lines.append(
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"
        )
        for _, row in sub.iterrows():
            lines.append(
                f"| `{row['feature']}` | {int(row['n_is'])} | {int(row['n_oos'])} "
                f"| {_format_float(row['mean_is'])} "
                f"| {_format_float(row['mean_oos'])} "
                f"| {_format_float(row['pooled_std'])} "
                f"| {_format_float(row['cohens_d'])} "
                f"| {row['note'] or ''} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def write_cohens_d_artifacts(df: pd.DataFrame, run_dir: Path) -> dict[str, Path]:
    """Write ``cohens_d.csv`` and ``cohens_d.md`` into ``run_dir``.

    Returns a ``{name: Path}`` map so callers (the orchestrator) can
    record the artifact paths in the manifest. Re-running on the same
    DataFrame produces byte-identical files.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "cohens_d.csv"
    md_path = run_dir / "cohens_d.md"

    # CSV: deterministic column order, fixed line terminator. We drive
    # the writer manually so NaN renders as the empty string rather than
    # the locale-dependent "nan" string pandas would emit.
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for _, row in df.iterrows():
            payload: dict[str, str] = {}
            for col in _COLUMNS:
                value = row[col]
                if isinstance(value, float):
                    if math.isnan(value):
                        payload[col] = ""
                    else:
                        payload[col] = f"{value:.6f}"
                else:
                    payload[col] = "" if value is None else str(value)
            writer.writerow(payload)

    md_path.write_text(_render_markdown(df, run_dir), encoding="utf-8")
    return {"cohens_d_csv": csv_path, "cohens_d_md": md_path}


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------


def compute_cohens_d(
    run_dir: Path,
    eval_path: Path,
    cutoffs_path: Path = Path("data/cutoffs.yaml"),
) -> pd.DataFrame:
    """Compute per-(model, feature) Cohen's d and write artifacts.

    Args:
        run_dir: Finished harness run directory containing
            ``records.jsonl`` and ``summary.csv``. Artifacts are
            written here.
        eval_path: Path to the eval-set JSONL whose ``prompt`` strings
            originally fed the harness. Required because the harness's
            ``Record`` schema carries only ``prompt_hash``; the date used
            to label IS / OOS lives on the eval row's
            ``metadata.date``.
        cutoffs_path: Path to ``data/cutoffs.yaml``. Models present in
            ``records.jsonl`` but absent here are logged as a warning
            and excluded from the artifact rather than crashing the run
            (Req 9.1 spirit).

    Returns:
        A ``pandas.DataFrame`` with one row per ``(model, feature)``
        pair and the schema documented in ``design.md``
        § Cohen's d artifact schema. The CSV / MD twins are written to
        ``run_dir`` as a side effect.

    Raises:
        FileNotFoundError: if ``records.jsonl`` is missing from the run
            directory or ``eval_path`` / ``cutoffs_path`` do not exist.
    """
    run_dir = Path(run_dir)
    eval_path = Path(eval_path)
    cutoffs_path = Path(cutoffs_path)

    records_path = run_dir / "records.jsonl"
    summary_path = run_dir / "summary.csv"

    if not records_path.exists():
        raise FileNotFoundError(f"missing records.jsonl in {run_dir}")
    if not eval_path.exists():
        raise FileNotFoundError(f"missing eval set: {eval_path}")
    if not cutoffs_path.exists():
        raise FileNotFoundError(f"missing cutoffs registry: {cutoffs_path}")

    metadata_by_hash = _load_eval_metadata(eval_path)
    cutoffs = _load_cutoffs(cutoffs_path)
    auc_by_model = _load_summary_auc(summary_path)

    buckets = _collect_feature_splits(records_path, metadata_by_hash, cutoffs)

    rows: list[dict] = []
    # Stable sort: by model then by the canonical feature order.
    feature_order = {name: i for i, name in enumerate(_FEATURE_NAMES)}
    sorted_keys = sorted(
        buckets.keys(), key=lambda k: (k[0], feature_order.get(k[1], 999))
    )
    for key in sorted_keys:
        model, feature = key
        slot = buckets[key]
        rows.append(_cohens_d_row(
            model=model,
            feature=feature,
            is_values=slot["is"],
            oos_values=slot["oos"],
            mcs_auc=auc_by_model.get(model),
        ))

    df = pd.DataFrame(rows, columns=_COLUMNS)
    write_cohens_d_artifacts(df, run_dir)
    return df


__all__ = [
    "compute_cohens_d",
    "write_cohens_d_artifacts",
]
