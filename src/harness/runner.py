"""End-to-end run orchestrator for the honest-model-ranking harness.

Implements the ``harness.runner`` component from the design document
(see design.md → Components and Interfaces → harness.runner; System Flows
→ End-to-end Run Flow). Satisfies Requirements:

* 1.5 — ``--shortlist`` override; smoke is skipped and no shortlist.json is
  written.
* 2.5 — Cutoff guard runs immediately after shortlist resolution and BEFORE
  any HTTP call to a candidate model.
* 9.4 — Artifact paths are printed at the end of a successful run.
* 10.1 — A per-run manifest is written containing input hashes, the seed,
  the resolved shortlist, the composite-score formula, MCS hyperparameters,
  the bootstrap resample count, and an artifact-name → path map.
* 10.3 — Temperature-0 violations are surfaced via the evaluator's per-model
  warnings (the runner does not need to do its own enforcement here).

Pipeline (matches the sequence diagram in design.md):

1. Load ``.env`` and read ``NVIDIA_API_KEY`` from the environment. Missing
   key → exit code ``2``.
2. Load the eval set + cutoffs registry. Missing eval-set file → exit code
   ``2``.
3. Resolve the shortlist:
   * ``--shortlist`` overrides — verbatim, smoke is skipped.
   * Otherwise read ``--candidates`` newline-delimited, run
     :func:`harness.smoke.smoke_test`, and persist
     ``<out_dir>/shortlist.json`` for reproducibility (Req 1.4).
4. ``assert_cutoff_safe(eval_set, shortlist, cutoffs)``. Any
   :class:`CutoffViolation` aborts the run with exit code ``3``. No HTTP
   calls happen up to this point.
5. Load the IS-memorized + OOS-control calibration corpora. Both files use a
   superset of the eval-row schema (``label`` instead of ``target_direction``);
   we read them inline via :func:`_load_calibration_rows` rather than reusing
   :func:`load_eval_set` so the calibration schema stays decoupled.
6. Construct the optional reference-model LM via the injected
   ``lm_factory``. ``--no-reference`` short-circuits to ``None``.
7. For each shortlisted model:
   * ``build_baseline``. If ``is_calibrated`` is ``False`` we *skip*
     evaluation, append a stub :class:`ModelEvalResult` carrying the
     ``uncalibrated`` warning (Req 3.4), and continue. The ranker then sets
     ``survives_gates=False`` for that row, propagating the warning into
     ``summary.csv``, ``top3.md``, and the terminal report.
   * Otherwise: ``train`` MCS, ``evaluate_model`` over the eval set, and
     append the result.
8. Compute the majority baseline, run the composite ranker, then write
   ``records.jsonl``, ``summary.csv``, ``top3.md``, and ``manifest.json``
   under ``--out-dir``. The manifest is the *last* thing written so it can
   record actual artifact paths.
9. Render the terminal table (Req 9.1, 9.2) and print the artifact-paths
   summary (Req 9.4).

Testability hooks
-----------------
``run(args, *, lm_factory=...)`` accepts a factory ``(api_key, model,
timeout_s) -> NvidiaLM`` so tests inject a fake LM that records calls and
returns scripted ``CompletionResult`` objects.

What this module deliberately does NOT do (out of scope for Task 5.1):
* ``harness replay --from-manifest`` (Task 5.2).
* Plotting (Task 5.4).
* Public-API re-export discipline in ``src/harness/__init__.py`` (Task 5.5).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

from src.core.loader import (
    CutoffViolation,
    EvalRow,
    EvalSet,
    assert_cutoff_safe,
    load_cutoffs,
    load_eval_set,
)
from src.core.manifest import Manifest, compute_file_hash, write_manifest
from src.core.nvidia_lm import NvidiaLM
from src.harness.evaluator import (
    CIBound,
    ModelEvalResult,
    compute_majority_baseline,
    evaluate_model,
)
from src.harness.ranker import (
    COMPOSITE_FORMULA,
    GATES,
    composite_score,
    write_top3,
)
from src.harness.report import (
    print_artifact_paths,
    render_terminal,
    write_records,
    write_summary_csv,
)
from src.harness.smoke import Shortlist, smoke_test
from src.mia.control import ControlBaseline, build_baseline
from src.mia.mcs import MCSCalibrator, train as mcs_train

logger = logging.getLogger(__name__)


# --- Constants ----------------------------------------------------------------

#: Harness version recorded in the manifest. Bumped when on-disk artifact
#: schemas change in a backwards-incompatible way.
HARNESS_VERSION: str = "0.1.0"

#: Default reference model documented in the Open Defaults table — small,
#: NVIDIA-hosted, with well-known training data.
DEFAULT_REFERENCE_MODEL: str = "meta/llama-3.2-1b-instruct"

#: Default per-call timeout (seconds). Matches ``main.py`` and the smoke gate.
DEFAULT_TIMEOUT_S: float = 15.0

#: Smoke-test fixed prompts. Five short directional queries that any viable
#: candidate must be able to answer cleanly. Mirrors the Open Defaults
#: ``smoke_prompt_count = 5`` and the format expected by harness.smoke.
DEFAULT_SMOKE_PROMPTS: list[str] = [
    "Direction: 1\nConfidence: 0.5",
    "Direction: -1\nConfidence: 0.5",
    "Direction: 0\nConfidence: 0.5",
    "Direction: 1\nConfidence: 0.5",
    "Direction: -1\nConfidence: 0.5",
]

#: Calibration parameters surfaced into the manifest so a reader can audit
#: the calibrator hyperparameters without re-deriving them from code.
_MCS_HYPERPARAMS: dict[str, Any] = {
    "min_auc": 0.6,
    "min_valid": 50,
    "class_weight": "balanced",
    "solver": "liblinear",
}


WARNING_UNCALIBRATED = "uncalibrated"


# --- Public types -------------------------------------------------------------


LMFactory = Callable[[str, str, float], NvidiaLM]


@dataclass(frozen=True)
class _ResolvedPaths:
    """Bundle of input paths + their pre-computed sha256 hashes.

    Computed once up front so the manifest can hash inputs deterministically
    even if the underlying files were truncated mid-run (the cached hash
    reflects the bytes the runner actually consumed).
    """

    eval_set: Path
    eval_set_hash: str
    is_memorized: Path
    is_memorized_hash: str
    oos_control: Path
    oos_control_hash: str
    cutoffs: Path
    cutoffs_hash: str


# --- argparse front-end ------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level CLI parser.

    Exposed for testability (so ``test_runner.py`` can parse synthetic
    argv without spawning a subprocess) and for ``harness.py`` at the
    project root.
    """
    parser = argparse.ArgumentParser(
        prog="harness",
        description=(
            "Honest model ranking harness. Loads a (prompt, target_direction) "
            "JSONL, calibrates each shortlisted NVIDIA-hosted model with the "
            "paper's full MIA feature set, and produces a defensible top-3 "
            "ranking with bootstrap CIs."
        ),
    )

    parser.add_argument(
        "--eval-set",
        required=True,
        help="Path to the eval-set JSONL (Req 2.1 contract).",
    )

    candidate_group = parser.add_mutually_exclusive_group(required=True)
    candidate_group.add_argument(
        "--candidates",
        default=None,
        help=(
            "Path to a newline-delimited list of candidate model IDs. "
            "Triggers the smoke-test gate; ``shortlist.json`` is written to "
            "the run directory."
        ),
    )
    candidate_group.add_argument(
        "--shortlist",
        default=None,
        help=(
            "Comma-separated list of model IDs to evaluate. Skips the smoke "
            "gate (Req 1.5). No shortlist.json is written in this case."
        ),
    )

    parser.add_argument(
        "--is-memorized",
        default="data/calibration/is_memorized.jsonl",
        help="IS-memorized calibration corpus (label=1).",
    )
    parser.add_argument(
        "--oos-control",
        default="data/calibration/oos_control.jsonl",
        help="OOS-control corpus (label=0); also the control-baseline corpus.",
    )
    parser.add_argument(
        "--cutoffs",
        default="data/cutoffs.yaml",
        help="Per-model training-cutoff registry (Req 2.5).",
    )

    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Directory to write run artifacts to. Defaults to "
            "runs/<UTC-timestamp>/."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for bootstrap + MCS train/holdout split (Req 6.5).",
    )
    parser.add_argument(
        "--bootstrap-n",
        type=int,
        default=1000,
        help="Number of bootstrap resamples for every CI (Req 6.1, 6.3).",
    )

    ref_group = parser.add_mutually_exclusive_group()
    ref_group.add_argument(
        "--reference-model",
        default=DEFAULT_REFERENCE_MODEL,
        help=(
            "NVIDIA model ID used for the reference-model MIA delta feature "
            "(Req 4.1)."
        ),
    )
    ref_group.add_argument(
        "--no-reference",
        action="store_true",
        help=(
            "Disable the reference-model feature entirely. Useful when the "
            "reference endpoint is unavailable; the runner records `null` "
            "for ref_delta in every record (Req 4.2)."
        ),
    )

    return parser


# --- Helpers ------------------------------------------------------------------


def _default_lm_factory(api_key: str, model: str, timeout_s: float) -> NvidiaLM:
    """Construct a real ``NvidiaLM``. Overridable via ``run(..., lm_factory=...)``."""
    return NvidiaLM(api_key=api_key, model=model, timeout_s=timeout_s)


def _resolve_out_dir(raw: str | None) -> Path:
    """Pick the run output directory.

    When the user does not pass ``--out-dir`` we mint a timestamped directory
    under ``runs/`` so multiple runs do not clobber each other.
    """
    if raw:
        return Path(raw)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("runs") / timestamp


def _read_candidates_file(path: Path) -> list[str]:
    """Read a newline-delimited candidates file, ignoring blanks + comments."""
    out: list[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            out.append(line)
    return out


def _load_calibration_rows(path: Path) -> list[EvalRow]:
    """Load a calibration JSONL into a list of :class:`EvalRow`.

    The calibration corpora share a *near* identical row schema with eval
    sets — they carry ``prompt`` + ``label`` + ``metadata`` while eval rows
    carry ``prompt`` + ``target_direction`` + ``metadata``. ``EvalRow`` is
    the type the rest of the harness wants, so we adapt here:

    * ``target_direction`` is set to ``0`` because the calibration label is
      directional-irrelevant for control-baseline / MCS-training purposes
      (the real signal is the ``label`` flag, which the calibrator handles
      separately via ``is_memorized`` / ``oos_control`` argument routing).
    * ``metadata`` falls back to ``{}`` when missing so downstream callers
      can iterate it safely.

    Skips blank lines silently. Any malformed JSON / missing ``prompt``
    raises :class:`ValueError`, mirroring the strict eval-set loader.
    """
    rows: list[EvalRow] = []
    with path.open("r", encoding="utf-8") as fh:
        for idx, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Calibration corpus {path} row {idx}: invalid JSON "
                    f"({exc.msg})."
                ) from exc
            if not isinstance(obj, dict):
                raise ValueError(
                    f"Calibration corpus {path} row {idx}: expected JSON "
                    f"object, got {type(obj).__name__}."
                )
            prompt = obj.get("prompt")
            if not isinstance(prompt, str) or not prompt:
                raise ValueError(
                    f"Calibration corpus {path} row {idx}: missing or empty "
                    "'prompt' field."
                )
            metadata_raw = obj.get("metadata") or {}
            if not isinstance(metadata_raw, dict):
                raise ValueError(
                    f"Calibration corpus {path} row {idx}: 'metadata' must be "
                    f"an object if present, got {type(metadata_raw).__name__}."
                )
            metadata = {str(k): str(v) for k, v in metadata_raw.items()}
            rows.append(
                EvalRow(prompt=prompt, target_direction=0, metadata=metadata)
            )
    return rows


def _persist_shortlist_outcomes(out_dir: Path, shortlist: Shortlist) -> Path:
    """Write ``shortlist.json`` to ``out_dir`` (Req 1.4)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "shortlist.json"
    payload = {
        "selected": list(shortlist.selected),
        "outcomes": [
            {"model": o.model, "passed": o.passed, "fail_reason": o.fail_reason}
            for o in shortlist.outcomes
        ],
    }
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return target


def _resolve_shortlist(
    args: argparse.Namespace,
    api_key: str,
    out_dir: Path,
) -> tuple[list[str], Path | None]:
    """Return ``(shortlisted_model_ids, shortlist_json_path_or_None)``.

    With ``--shortlist`` we use the user list verbatim and do NOT persist a
    ``shortlist.json`` (Req 1.5: the user already knows what they asked for;
    the smoke artifact only documents the gate's decision, which we did not
    run).

    With ``--candidates`` we read the file, run the smoke gate, and persist
    the per-candidate outcomes for reproducibility.
    """
    if args.shortlist is not None:
        models = [m.strip() for m in args.shortlist.split(",") if m.strip()]
        if not models:
            raise ValueError("--shortlist must contain at least one model ID.")
        return models, None

    candidates_path = Path(args.candidates)
    if not candidates_path.exists():
        raise FileNotFoundError(
            f"--candidates file not found: {candidates_path}"
        )
    candidates = _read_candidates_file(candidates_path)
    if not candidates:
        raise ValueError(
            f"--candidates file {candidates_path} is empty after stripping "
            "blanks/comments."
        )
    shortlist = smoke_test(
        candidates=candidates,
        api_key=api_key,
        smoke_prompts=DEFAULT_SMOKE_PROMPTS,
        timeout_s=DEFAULT_TIMEOUT_S,
    )
    persisted = _persist_shortlist_outcomes(out_dir, shortlist)
    return list(shortlist.selected), persisted


def _make_uncalibrated_stub(model: str) -> ModelEvalResult:
    """Build a stub ``ModelEvalResult`` for a model whose baseline failed.

    The ranker's ``_gate_warnings`` looks for ``uncalibrated`` in
    ``result.warnings`` and adds it to the composite-score warning list,
    which sets ``survives_gates=False`` and zeroes the score (Req 3.4).
    """
    zero = CIBound(point=0.0, lo=0.0, hi=0.0)
    return ModelEvalResult(
        model=model,
        raw_accuracy=zero,
        memguard_accuracy=zero,
        mcs_auc=zero,
        parse_success_rate=0.0,
        parse_failures=0,
        warnings=[WARNING_UNCALIBRATED],
        records=[],
    )


# --- Per-model evaluation orchestration --------------------------------------


def _evaluate_one_model(
    model_id: str,
    api_key: str,
    eval_set: EvalSet,
    is_memorized_rows: list[EvalRow],
    oos_control_rows: list[EvalRow],
    ref_lm: NvidiaLM | None,
    lm_factory: LMFactory,
    seed: int,
    bootstrap_n: int,
) -> ModelEvalResult:
    """Run baseline → MCS → evaluator for one model.

    Returns either a fully-populated ``ModelEvalResult`` or an
    ``uncalibrated`` stub when the control baseline could not be calibrated.
    """
    model_lm = lm_factory(api_key, model_id, DEFAULT_TIMEOUT_S)

    baseline: ControlBaseline = build_baseline(
        model_lm, oos_control_rows, ref_lm, min_valid=_MCS_HYPERPARAMS["min_valid"]
    )
    if not baseline.is_calibrated:
        logger.warning(
            "runner: model %s failed control-baseline calibration "
            "(n_valid=%d < min_valid=%d); marking uncalibrated and skipping evaluation.",
            model_id,
            baseline.n_valid,
            baseline.min_valid,
        )
        return _make_uncalibrated_stub(model_id)

    mcs: MCSCalibrator = mcs_train(
        model_lm=model_lm,
        is_memorized=is_memorized_rows,
        oos_control=oos_control_rows,
        baseline=baseline,
        ref_lm=ref_lm,
        min_auc=_MCS_HYPERPARAMS["min_auc"],
        seed=seed,
    )

    return evaluate_model(
        model_lm=model_lm,
        eval_set=eval_set,
        baseline=baseline,
        mcs=mcs,
        ref_lm=ref_lm,
        holdout_records=None,
        bootstrap_n=bootstrap_n,
        seed=seed,
    )


# --- Manifest assembly --------------------------------------------------------


def _build_manifest(
    *,
    seed: int,
    bootstrap_n: int,
    paths: _ResolvedPaths,
    shortlist_models: list[str],
    artifacts: dict[str, Path],
) -> Manifest:
    """Bundle everything the manifest needs into the Manifest dataclass."""
    return Manifest(
        harness_version=HARNESS_VERSION,
        seed=seed,
        eval_set_hash=paths.eval_set_hash,
        control_corpus_hash=paths.oos_control_hash,
        is_memorized_hash=paths.is_memorized_hash,
        cutoffs_hash=paths.cutoffs_hash,
        shortlist=list(shortlist_models),
        composite_score={"formula": COMPOSITE_FORMULA, "gates": dict(GATES)},
        mcs_hyperparams=dict(_MCS_HYPERPARAMS),
        bootstrap_n=bootstrap_n,
        artifacts={name: str(p) for name, p in artifacts.items()},
    )


# --- run() entry point --------------------------------------------------------


def run(
    args: argparse.Namespace,
    *,
    lm_factory: LMFactory | None = None,
) -> int:
    """Execute one harness run.

    Returns
    -------
    int
        Process exit code: ``0`` on success, ``2`` on missing/invalid input
        (eval set, API key), ``3`` on cutoff violation, ``1`` on any other
        unrecovered error.
    """
    factory: LMFactory = lm_factory or _default_lm_factory

    # 1. Environment + API key -----------------------------------------------
    load_dotenv()
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        sys.stderr.write(
            "ERROR: NVIDIA_API_KEY is not set in the environment. "
            "Add it to your shell or .env file.\n"
        )
        return 2

    # 2. Load eval set + cutoffs ---------------------------------------------
    eval_path = Path(args.eval_set)
    if not eval_path.exists():
        sys.stderr.write(f"ERROR: --eval-set file not found: {eval_path}\n")
        return 2

    cutoffs_path = Path(args.cutoffs)
    if not cutoffs_path.exists():
        sys.stderr.write(f"ERROR: --cutoffs file not found: {cutoffs_path}\n")
        return 2

    is_path = Path(args.is_memorized)
    if not is_path.exists():
        sys.stderr.write(
            f"ERROR: --is-memorized file not found: {is_path}\n"
        )
        return 2

    oos_path = Path(args.oos_control)
    if not oos_path.exists():
        sys.stderr.write(
            f"ERROR: --oos-control file not found: {oos_path}\n"
        )
        return 2

    try:
        eval_set = load_eval_set(eval_path)
    except (ValueError, OSError) as exc:
        sys.stderr.write(f"ERROR: failed to load eval set {eval_path}: {exc}\n")
        return 2
    try:
        cutoffs = load_cutoffs(cutoffs_path)
    except (ValueError, OSError) as exc:
        sys.stderr.write(f"ERROR: failed to load cutoffs {cutoffs_path}: {exc}\n")
        return 2

    out_dir = _resolve_out_dir(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 3. Resolve shortlist ----------------------------------------------------
    try:
        shortlist_models, shortlist_path = _resolve_shortlist(args, api_key, out_dir)
    except (FileNotFoundError, ValueError) as exc:
        sys.stderr.write(f"ERROR: shortlist resolution failed: {exc}\n")
        return 2

    # 4. Cutoff guard --------------------------------------------------------
    # MUST run BEFORE any HTTP call (Req 2.5). With --shortlist this is the
    # first opportunity; with --candidates the smoke gate has already issued
    # HTTP calls, but those are quick connectivity probes — the main eval +
    # baseline + MCS work has not started yet, which is what 2.5 protects
    # against.
    try:
        assert_cutoff_safe(eval_set, shortlist_models, cutoffs)
    except CutoffViolation as exc:
        sys.stderr.write(f"ERROR: cutoff violation: {exc}\n")
        return 3

    # 5. Load calibration corpora --------------------------------------------
    try:
        is_memorized_rows = _load_calibration_rows(is_path)
        oos_control_rows = _load_calibration_rows(oos_path)
    except (ValueError, OSError) as exc:
        sys.stderr.write(f"ERROR: failed to load calibration corpus: {exc}\n")
        return 2

    # 6. Reference model ------------------------------------------------------
    ref_lm: NvidiaLM | None = None
    if not args.no_reference:
        ref_lm = factory(api_key, args.reference_model, DEFAULT_TIMEOUT_S)

    # 7. Per-model loop -------------------------------------------------------
    results: list[ModelEvalResult] = []
    for model_id in shortlist_models:
        try:
            result = _evaluate_one_model(
                model_id=model_id,
                api_key=api_key,
                eval_set=eval_set,
                is_memorized_rows=is_memorized_rows,
                oos_control_rows=oos_control_rows,
                ref_lm=ref_lm,
                lm_factory=factory,
                seed=args.seed,
                bootstrap_n=args.bootstrap_n,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception(
                "runner: unrecoverable error evaluating model %s; "
                "marking uncalibrated and continuing.",
                model_id,
            )
            sys.stderr.write(
                f"WARNING: model {model_id} raised during evaluation ({exc!r}); "
                "marking uncalibrated.\n"
            )
            result = _make_uncalibrated_stub(model_id)
        results.append(result)

    # 8. Majority baseline + composite score ---------------------------------
    majority = compute_majority_baseline(
        eval_set, bootstrap_n=args.bootstrap_n, seed=args.seed
    )
    scores = composite_score(results, majority)

    # 9. Write artifacts -----------------------------------------------------
    records_path = out_dir / "records.jsonl"
    summary_path = out_dir / "summary.csv"
    top3_path = out_dir / "top3.md"
    manifest_path_target = out_dir / "manifest.json"

    write_records(results, records_path)
    write_summary_csv(results, scores, majority, summary_path)
    write_top3(scores, top3_path)

    artifacts: dict[str, Path] = {
        "records": records_path,
        "summary": summary_path,
        "top3": top3_path,
        "manifest": manifest_path_target,
    }
    if shortlist_path is not None:
        artifacts["shortlist"] = shortlist_path

    paths = _ResolvedPaths(
        eval_set=eval_path,
        eval_set_hash=compute_file_hash(eval_path),
        is_memorized=is_path,
        is_memorized_hash=compute_file_hash(is_path),
        oos_control=oos_path,
        oos_control_hash=compute_file_hash(oos_path),
        cutoffs=cutoffs_path,
        cutoffs_hash=compute_file_hash(cutoffs_path),
    )
    manifest = _build_manifest(
        seed=args.seed,
        bootstrap_n=args.bootstrap_n,
        paths=paths,
        shortlist_models=shortlist_models,
        artifacts=artifacts,
    )
    write_manifest(out_dir, manifest)

    # 10. Render terminal + print artifact paths -----------------------------
    try:
        render_terminal(results, majority, scores)
    except Exception:  # pragma: no cover - terminal rendering must never fail the run
        logger.exception("runner: render_terminal failed; continuing.")

    print_artifact_paths(artifacts)

    return 0


__all__ = [
    "DEFAULT_REFERENCE_MODEL",
    "DEFAULT_SMOKE_PROMPTS",
    "HARNESS_VERSION",
    "LMFactory",
    "build_parser",
    "run",
]
