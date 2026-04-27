"""Generic JSONL evaluation-set loader and cutoff-date guard.

Implements the input-source-agnostic JSONL contract for the honest-model-ranking
harness (Req 2.1-2.5):

- ``EvalRow`` / ``EvalSet`` frozen dataclasses describe the in-memory shape.
- ``load_eval_set(path)`` parses an optional ``_cutoff_date`` header line plus
  one ``{prompt, target_direction[, metadata]}`` row per line. It validates the
  row schema strictly (raises ``ValueError`` on bad rows) and emits
  ``logging.WARNING`` records for low-N (<100) and class-imbalance (>60%)
  conditions, rather than raising.
- ``load_cutoffs(path)`` parses ``data/cutoffs.yaml`` of shape
  ``{models: {model_id: YYYY-MM-DD}}`` into a ``dict[str, date]``.
- ``assert_cutoff_safe(eval_set, models, cutoffs)`` enforces the cutoff guard:
  every shortlisted model must appear in ``cutoffs``, and no model's training
  cutoff may post-date the eval set's declared ``cutoff_date``. Violations
  raise ``CutoffViolation``.

The loader never performs a train/dev split (Req 2.4): the entire file is the
evaluation set.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


# Policy thresholds derived from Open Defaults in requirements.md.
_MIN_ROWS_FOR_POWER = 100
_MAX_MAJORITY_SHARE = 0.60
_VALID_DIRECTIONS: frozenset[int] = frozenset({-1, 0, 1})


class CutoffViolation(Exception):
    """Raised when shortlisted models post-date the eval set's cutoff."""


@dataclass(frozen=True)
class EvalRow:
    """One evaluation row: prompt + ground-truth direction + opaque metadata."""

    prompt: str
    target_direction: int  # in {-1, 0, 1}
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class EvalSet:
    """A loaded JSONL eval set plus its cutoff header and content hash."""

    rows: list[EvalRow]
    cutoff_date: date | None
    path_hash: str  # sha256 hex digest of the file bytes


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _coerce_metadata(raw: Any, line_num: int) -> dict[str, str]:
    """Coerce optional metadata into a string-valued dict.

    The design treats metadata as opaque pass-through; we therefore stringify
    values for safe downstream use without inspecting the keys.
    """
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(
            f"Row {line_num}: 'metadata' must be an object if present, got {type(raw).__name__}."
        )
    return {str(k): str(v) for k, v in raw.items()}


def _parse_row(obj: dict[str, Any], line_num: int) -> EvalRow:
    if "prompt" not in obj:
        raise ValueError(f"Row {line_num}: missing required field 'prompt'.")
    prompt = obj["prompt"]
    if not isinstance(prompt, str) or not prompt:
        raise ValueError(
            f"Row {line_num}: 'prompt' must be a non-empty string, got {type(prompt).__name__}."
        )

    if "target_direction" not in obj:
        raise ValueError(f"Row {line_num}: missing required field 'target_direction'.")
    target = obj["target_direction"]
    # bool is a subclass of int; reject it explicitly to keep the contract tight.
    if isinstance(target, bool) or not isinstance(target, int):
        raise ValueError(
            f"Row {line_num}: 'target_direction' must be an int in {{-1,0,1}}, got "
            f"{type(target).__name__}."
        )
    if target not in _VALID_DIRECTIONS:
        raise ValueError(
            f"Row {line_num}: 'target_direction' must be in {{-1,0,1}}, got {target}."
        )

    metadata = _coerce_metadata(obj.get("metadata"), line_num)
    return EvalRow(prompt=prompt, target_direction=target, metadata=metadata)


def load_eval_set(path: Path | str) -> EvalSet:
    """Parse a JSONL eval file into an ``EvalSet``.

    The first line may optionally be a header object containing
    ``{"_cutoff_date": "YYYY-MM-DD"}``. All other lines must be row objects
    matching the input contract (Req 2.1). Logs WARNING records for low-N
    and class-imbalance conditions (Req 2.2, 2.3); never raises for those.
    Returns the entire set as a single list — no train/dev split (Req 2.4).
    """
    path = Path(path)
    cutoff: date | None = None
    rows: list[EvalRow] = []

    with path.open("r", encoding="utf-8") as fh:
        for idx, raw_line in enumerate(fh, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Row {idx}: invalid JSON ({exc.msg}).") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Row {idx}: expected JSON object, got {type(obj).__name__}.")

            # Header: only accepted as the very first non-empty line.
            if idx == 1 and "_cutoff_date" in obj and "prompt" not in obj:
                raw_cutoff = obj["_cutoff_date"]
                if not isinstance(raw_cutoff, str):
                    raise ValueError(
                        f"Header '_cutoff_date' must be an ISO-8601 string, got "
                        f"{type(raw_cutoff).__name__}."
                    )
                try:
                    cutoff = date.fromisoformat(raw_cutoff)
                except ValueError as exc:
                    raise ValueError(
                        f"Header '_cutoff_date' must be ISO-8601 (YYYY-MM-DD): {exc}."
                    ) from exc
                continue

            rows.append(_parse_row(obj, idx))

    _emit_quality_warnings(path, rows)
    return EvalSet(rows=rows, cutoff_date=cutoff, path_hash=_hash_file(path))


def _emit_quality_warnings(path: Path, rows: list[EvalRow]) -> None:
    n = len(rows)
    if n < _MIN_ROWS_FOR_POWER:
        logger.warning(
            "Eval set %s has only %d rows (< %d): low statistical power; "
            "bootstrap CIs will be wide.",
            path,
            n,
            _MIN_ROWS_FOR_POWER,
        )

    if n == 0:
        return

    counts: dict[int, int] = {-1: 0, 0: 0, 1: 0}
    for row in rows:
        counts[row.target_direction] += 1
    majority_class, majority_count = max(counts.items(), key=lambda kv: kv[1])
    majority_share = majority_count / n
    if majority_share > _MAX_MAJORITY_SHARE:
        logger.warning(
            "Eval set %s class imbalance: majority class %d holds %.1f%% of rows "
            "(> %.0f%%); accuracy will be inflated by always-predict-majority.",
            path,
            majority_class,
            majority_share * 100.0,
            _MAX_MAJORITY_SHARE * 100.0,
        )


def load_cutoffs(path: Path | str) -> dict[str, date]:
    """Parse the cutoffs YAML registry into ``{model_id: cutoff_date}``."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)

    if not isinstance(doc, dict) or "models" not in doc:
        raise ValueError(
            f"Cutoffs file {path} must be a YAML mapping with a top-level 'models' key."
        )
    models = doc["models"]
    if not isinstance(models, dict):
        raise ValueError(
            f"Cutoffs file {path}: 'models' must be a mapping of model_id -> date."
        )

    out: dict[str, date] = {}
    for model_id, raw in models.items():
        if isinstance(raw, date):
            out[str(model_id)] = raw
        elif isinstance(raw, str):
            out[str(model_id)] = date.fromisoformat(raw)
        else:
            raise ValueError(
                f"Cutoffs file {path}: model {model_id!r} cutoff must be a date "
                f"or ISO-8601 string, got {type(raw).__name__}."
            )
    return out


def assert_cutoff_safe(
    eval_set: EvalSet,
    models: list[str],
    cutoffs: dict[str, date],
) -> None:
    """Fail-fast guard: every shortlisted model has a cutoff <= eval cutoff.

    Raises:
        CutoffViolation: if any model is missing from ``cutoffs`` or, when
            ``eval_set.cutoff_date`` is set, post-dates it.
    """
    missing = [m for m in models if m not in cutoffs]
    if missing:
        raise CutoffViolation(
            "Shortlisted models missing from cutoffs registry: " + ", ".join(missing)
        )

    if eval_set.cutoff_date is None:
        return

    too_late: list[tuple[str, date]] = [
        (m, cutoffs[m]) for m in models if cutoffs[m] > eval_set.cutoff_date
    ]
    if too_late:
        details = ", ".join(f"{m}={c.isoformat()}" for m, c in too_late)
        raise CutoffViolation(
            f"Models with training cutoffs after eval cutoff "
            f"{eval_set.cutoff_date.isoformat()}: {details}"
        )
