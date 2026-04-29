"""Structured report writers for the harness (Req 9.1, 9.2, 9.3, 9.4).

Implements the ``harness.report`` component from the honest-model-ranking
design (see design.md → Components and Interfaces → harness.report).

Public surface
--------------
* :func:`render_terminal` — ``rich``-backed table rendering of one row per
  shortlisted model plus a ``__majority_baseline__`` row (Req 9.1, 9.2).
* :func:`write_records` — streaming JSONL writer; one JSON object per
  ``Record`` (Req 9.3). Memory stays bounded by writing line-by-line rather
  than building an in-memory list of all records first.
* :func:`write_summary_csv` — flat CSV with a 15-column schema plus a final
  ``__majority_baseline__`` row that fills only the raw-accuracy CI cells
  (Req 9.3 schema half).
* :func:`print_artifact_paths` — final-line summary of every artifact path so
  the operator sees the run output up front (Req 9.4).

Design choices
--------------
* The CSV signature is ``(results, scores, majority, path)``: the design's
  Service Interface lists three arguments but the Req 9.3 observable in
  ``tasks.md`` Task 4.4 demands a majority row, so the majority CIBound is
  threaded through explicitly.
* Sorting in ``render_terminal`` mirrors the ranker's stable-by-input-order
  semantics: within equal scores the input order is preserved.
* JSON serialisation uses ``dataclasses.asdict`` for ``MiaFeatures`` so the
  per-record artifact stays a flat object rather than a stringified
  dataclass repr (audited by ``tests/harness/test_report.py``).
"""

from __future__ import annotations

import csv
import json
import shutil
from collections.abc import Iterable
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from src.harness.evaluator import CIBound, ModelEvalResult, Record
from src.harness.ranker import CompositeScore

# --- Constants ----------------------------------------------------------------

#: Sentinel label for the majority-class baseline row in both the terminal
#: render and ``summary.csv``. Distinct from any model ID by virtue of the
#: leading double underscore.
MAJORITY_LABEL: str = "__majority_baseline__"

#: Stable column order for ``summary.csv`` (Req 9.3 schema half). Kept as a
#: module-level constant so tests and downstream consumers can import it.
SUMMARY_CSV_COLUMNS: list[str] = [
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


# --- Internal helpers ---------------------------------------------------------


def _format_ci(ci: CIBound) -> str:
    """Render ``CIBound`` as ``"point [lo–hi]"`` rounded to 4 decimals."""
    return f"{ci.point:.4f} [{ci.lo:.4f}–{ci.hi:.4f}]"


def _format_percent(value: float) -> str:
    """Render a 0..1 fraction as ``"NN.N%"``."""
    return f"{value * 100.0:.1f}%"


def _format_score(score: CompositeScore | None) -> str:
    """Render a composite score for the terminal table.

    Returns an em-dash for non-survivors so the reader can distinguish
    "gate-failed" from "lowest-ranked"; surviving models show four decimals.
    """
    if score is None or not score.survives_gates:
        return "—"
    return f"{score.score:.4f}"


def _format_warnings(warnings: list[str] | None) -> str:
    """Comma-separated warning string; empty when no warnings."""
    if not warnings:
        return ""
    return ", ".join(warnings)


def _score_for_model(
    model: str, scores: list[CompositeScore]
) -> CompositeScore | None:
    """Lookup the ``CompositeScore`` for ``model`` (None if not found)."""
    for s in scores:
        if s.model == model:
            return s
    return None


def _record_to_jsonable(record: Record) -> dict[str, Any]:
    """Convert one ``Record`` to a JSON-ready dict.

    ``MiaFeatures`` and ``features_standardised`` become plain dicts so the
    per-record artifact (Req 9.3) is a flat object rather than a stringified
    dataclass repr — the test fixture asserts this explicitly.
    """
    if record.features_raw is None:
        features_raw: dict[str, Any] | None = None
    elif is_dataclass(record.features_raw):
        features_raw = asdict(record.features_raw)
    else:  # pragma: no cover - defensive path; MiaFeatures is a dataclass
        features_raw = dict(record.features_raw)  # type: ignore[arg-type]

    if record.features_standardised is None:
        features_standardised: dict[str, Any] | None = None
    else:
        # Already a dict per the Record dataclass; copy to detach from caller.
        features_standardised = dict(record.features_standardised)

    return {
        "model": record.model,
        "prompt_hash": record.prompt_hash,
        "parse_ok": record.parse_ok,
        "predicted_direction": record.predicted_direction,
        "raw_confidence": record.raw_confidence,
        "penalized_confidence": record.penalized_confidence,
        "target_direction": record.target_direction,
        "features_raw": features_raw,
        "features_standardised": features_standardised,
        "p_memorized": record.p_memorized,
        "fail_reason": record.fail_reason,
        "raw_response_excerpt": record.raw_response_excerpt,
    }


def _ensure_parent(path: Path) -> None:
    """Create ``path.parent`` if missing so callers can pass nested paths."""
    path.parent.mkdir(parents=True, exist_ok=True)


# --- Public API: render_terminal ---------------------------------------------


def render_terminal(
    results: list[ModelEvalResult],
    majority: CIBound,
    scores: list[CompositeScore],
    console: Console | None = None,
) -> None:
    """Print one table row per model plus a majority-baseline row (Req 9.1, 9.2).

    Rows for surviving + non-surviving models are sorted by ``score``
    descending (stable within ties on input order); the
    ``__majority_baseline__`` row always renders last so a reader can compare
    every model against it visually.

    The majority row populates only the Raw Acc CI column — the other cells
    are em-dashes since MemGuard accuracy, MCS-AUC, and the composite score
    are not defined for the baseline.

    Parameters
    ----------
    results, scores:
        Aligned by model ID via lookup (not by index) so a missing score does
        not silently misalign rows.
    majority:
        Bootstrap CI on the majority-class baseline accuracy from
        ``compute_majority_baseline``.
    console:
        Optional ``rich.console.Console`` injection point for tests; defaults
        to ``Console()`` (writes to stdout).
    """
    table = Table(show_header=True, header_style="bold")
    table.add_column("Model")
    table.add_column("Raw Acc (CI)")
    table.add_column("MemGuard Acc (CI)")
    table.add_column("MCS-AUC (CI)")
    table.add_column("Parse %")
    table.add_column("Score")
    table.add_column("Warnings")

    # Stable sort: descending by score (None / missing scores treated as 0.0
    # so they fall to the bottom but stay above the majority row).
    indexed = list(enumerate(results))

    def _sort_key(pair: tuple[int, ModelEvalResult]) -> tuple[float, int]:
        idx, result = pair
        score = _score_for_model(result.model, scores)
        score_value = score.score if score is not None else 0.0
        # Negate score for descending order while keeping idx ascending for
        # stability on ties.
        return (-score_value, idx)

    indexed.sort(key=_sort_key)
    sorted_results = [r for _, r in indexed]

    for result in sorted_results:
        score = _score_for_model(result.model, scores)
        # Warnings shown in the terminal merge evaluator + ranker warnings;
        # the ranker passes through informational ones so we deduplicate while
        # preserving order.
        warning_set: list[str] = []
        for w in (result.warnings or []) + (score.warnings if score else []):
            if w not in warning_set:
                warning_set.append(w)

        table.add_row(
            result.model,
            _format_ci(result.raw_accuracy),
            _format_ci(result.memguard_accuracy),
            _format_ci(result.mcs_auc),
            _format_percent(result.parse_success_rate),
            _format_score(score),
            _format_warnings(warning_set),
        )

    # Majority-baseline row: only the Raw Acc CI is meaningful.
    em_dash = "—"
    table.add_row(
        MAJORITY_LABEL,
        _format_ci(majority),
        em_dash,
        em_dash,
        em_dash,
        em_dash,
        "",
    )

    # Auto-detect terminal width when stdout is a TTY; fall back to a wide
    # 200-col console for redirected/captured stdout (pytest, pipes) so the
    # model column and warning strings do not get truncated. Callers can
    # inject their own ``Console`` to override this.
    target = console or Console(width=shutil.get_terminal_size((200, 20)).columns)
    target.print(table)


# --- Public API: write_records ------------------------------------------------


def write_records(results: Iterable[ModelEvalResult], path: Path) -> None:
    """Stream every ``Record`` from every result to ``records.jsonl`` (Req 9.3).

    Memory stays bounded for long runs because the writer opens the file once
    and emits one ``json.dumps`` line per record before moving to the next —
    no all-records list is built in memory.

    The schema is documented in :func:`_record_to_jsonable` and audited by
    ``tests/harness/test_report.py::test_write_records_includes_all_required_fields``.
    """
    target = Path(path)
    _ensure_parent(target)
    with target.open("w", encoding="utf-8") as fh:
        for result in results:
            for record in result.records:
                payload = _record_to_jsonable(record)
                fh.write(json.dumps(payload, ensure_ascii=False))
                fh.write("\n")


# --- Public API: write_summary_csv -------------------------------------------


def _result_row(
    result: ModelEvalResult, score: CompositeScore | None
) -> dict[str, Any]:
    """Build the dict of CSV cells for one ``ModelEvalResult``."""
    if score is None:
        # No matching CompositeScore — the model evaluated but the ranker
        # failed to produce an entry for it. Fall back to a zero score with
        # the evaluator-side warnings so the CSV stays consistent.
        score_value = 0.0
        survives = False
        warnings = list(result.warnings or [])
    else:
        score_value = score.score
        survives = score.survives_gates
        warnings = list(score.warnings or [])

    return {
        "model": result.model,
        "raw_acc_point": f"{result.raw_accuracy.point:.6f}",
        "raw_acc_lo": f"{result.raw_accuracy.lo:.6f}",
        "raw_acc_hi": f"{result.raw_accuracy.hi:.6f}",
        "memguard_acc_point": f"{result.memguard_accuracy.point:.6f}",
        "memguard_acc_lo": f"{result.memguard_accuracy.lo:.6f}",
        "memguard_acc_hi": f"{result.memguard_accuracy.hi:.6f}",
        "mcs_auc_point": f"{result.mcs_auc.point:.6f}",
        "mcs_auc_lo": f"{result.mcs_auc.lo:.6f}",
        "mcs_auc_hi": f"{result.mcs_auc.hi:.6f}",
        "parse_success_rate": f"{result.parse_success_rate:.6f}",
        "parse_failures": str(result.parse_failures),
        "score": f"{score_value:.6f}",
        "survives_gates": "true" if survives else "false",
        "warnings": ";".join(warnings),
    }


def _majority_row(majority: CIBound) -> dict[str, Any]:
    """Build the ``__majority_baseline__`` CSV row.

    Only the raw-accuracy CI cells are populated; every other column is
    blank because MemGuard accuracy, MCS-AUC, and the composite score have
    no meaning for the baseline.
    """
    row = {col: "" for col in SUMMARY_CSV_COLUMNS}
    row["model"] = MAJORITY_LABEL
    row["raw_acc_point"] = f"{majority.point:.6f}"
    row["raw_acc_lo"] = f"{majority.lo:.6f}"
    row["raw_acc_hi"] = f"{majority.hi:.6f}"
    return row


def write_summary_csv(
    results: list[ModelEvalResult],
    scores: list[CompositeScore],
    majority: CIBound,
    path: Path,
) -> None:
    """Write one CSV row per model plus a majority-baseline row (Req 9.3).

    The 15-column schema is fixed in :data:`SUMMARY_CSV_COLUMNS`; every CSV
    consumer (e.g. the qualification notebook) can rely on it.

    The majority row only fills the raw-accuracy CI cells; the rest are blank
    because there is no MemGuard accuracy / MCS-AUC / score notion for the
    baseline (this matches the design's "majority row alongside model rows"
    interpretation).

    Notes
    -----
    The function signature includes ``majority`` even though the design's
    Service Interface lists only three arguments — Task 4.4's observable
    requires the majority row in the CSV, which forces the parameter through.
    """
    target = Path(path)
    _ensure_parent(target)

    with target.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=SUMMARY_CSV_COLUMNS, lineterminator="\n"
        )
        writer.writeheader()

        for result in results:
            score = _score_for_model(result.model, scores)
            writer.writerow(_result_row(result, score))

        writer.writerow(_majority_row(majority))


# --- Public API: print_artifact_paths ----------------------------------------


def print_artifact_paths(
    paths: dict[str, Path], console: Console | None = None
) -> None:
    """Print the final ``Artifacts:`` summary block (Req 9.4).

    Each key/value pair is rendered as ``<name>  <absolute path>`` so the
    operator can copy paths directly out of the terminal.
    """
    target = console or Console()
    target.print("Artifacts:")
    for name, path in paths.items():
        target.print(f"  {name}\t{path}")


__all__ = [
    "MAJORITY_LABEL",
    "SUMMARY_CSV_COLUMNS",
    "print_artifact_paths",
    "render_terminal",
    "write_records",
    "write_summary_csv",
]
