"""End-to-end run orchestrator for the honest-model-ranking harness.

This module drives the current build-only harness flow:

- resolve the shortlist (`--shortlist` directly or `--candidates` via the smoke test)
- enforce the cutoff guard before any model calls
- build the control baseline and per-model MCS classifier
- evaluate the shortlisted models on the eval set
- rank the results and write the run artifacts

The successful run writes `records.jsonl`, `summary.csv`, `top3.md`, and
`manifest.json`, then prints the artifact paths. When the run starts from
`--candidates`, it also writes `shortlist.json`.

Key behavior:

* `--shortlist` skips the smoke test and does not write `shortlist.json`.
* The cutoff guard runs immediately after shortlist resolution and before any
  HTTP call to a candidate model.
* The manifest records input hashes, the seed, the resolved shortlist, the
  composite-score formula, MCS hyperparameters, the bootstrap count, and the
  artifact path map.
* Temperature-0 problems are surfaced through evaluator warnings rather than a
  runner-specific enforcement layer.

Pipeline summary:

1. Load `.env` and read `NVIDIA_API_KEY`. Missing key -> exit code `2`.
2. Load the eval set and cutoff registry. Missing eval-set file -> exit code `2`.
3. Resolve the shortlist.
4. Run `assert_cutoff_safe(eval_set, shortlist, cutoffs)`. Any
   `CutoffViolation` aborts with exit code `3`.
5. Load the IS and OOS calibration corpora.
6. Construct the optional reference-model LM via the injected `lm_factory`.
7. For each shortlisted model, build the control baseline. If it is not
   calibrated, append a stub `ModelEvalResult` with the `uncalibrated` warning.
   Otherwise train the MCS classifier and evaluate the model on the eval set.
8. Compute the majority baseline, rank the models, and write the run artifacts.
9. Render the terminal table and print the artifact-path summary.

`run(args, *, lm_factory=...)` accepts a factory `(api_key, model, timeout_s) -> NvidiaLM`
so tests can inject a fake LM that records calls and returns scripted `CompletionResult`
objects.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from recall_guard.core.loader import (
    CutoffViolation,
    EvalRow,
    EvalSet,
    assert_cutoff_safe,
    load_cutoffs,
    load_eval_set,
)
from recall_guard.core.manifest import (
    Manifest,
    compute_file_hash,
    write_manifest,
)
from recall_guard.core.nvidia_lm import NvidiaLM
from recall_guard.harness.evaluator import (
    CIBound,
    ModelEvalResult,
    compute_majority_baseline,
    evaluate_model,
)
from recall_guard.harness.ranker import (
    COMPOSITE_FORMULA,
    GATES,
    composite_score,
    write_top3,
)
from recall_guard.harness.report import (
    print_artifact_paths,
    render_terminal,
    write_records,
    write_summary_csv,
)
from recall_guard.harness.smoke import Shortlist, smoke_test
from recall_guard.mia.control import ControlBaseline, build_baseline
from recall_guard.mia.mcs import MCSCalibrator
from recall_guard.mia.mcs import train as mcs_train

logger = logging.getLogger(__name__)


# --- Constants ----------------------------------------------------------------

#: Harness version recorded in the manifest. Bumped when on-disk artifact
#: schemas change in a backwards-incompatible way.
HARNESS_VERSION: str = "0.2.0"

#: Default reference model documented in the Open Defaults table: small,
#: NVIDIA-hosted, with well-known training data.
DEFAULT_REFERENCE_MODEL: str = "meta/llama-3.2-1b-instruct"

#: Default per-call timeout (seconds). 45s accommodates reasoning models
#: (gpt-oss-*, nemotron-nano-*) that emit ~200-token reasoning chains
#: before producing their final ``Direction:``/``Confidence:`` lines.
DEFAULT_TIMEOUT_S: float = 45.0

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


#: Names of the recognised subcommands. Exposed at module scope so the
#: argv-fallback logic in ``parse_argv`` can detect a missing subcommand
#: without re-introspecting the argparse object.
SUBCOMMANDS: tuple[str, ...] = ("build",)


def _add_build_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the ``build`` subcommand's flags onto ``parser``.

    Factored out so the same flags can be attached to a subparser AND
    (conceptually) reused if a future caller wants to construct a build
    parser standalone.
    """
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
    parser.add_argument(
        "--min-call-interval",
        type=float,
        default=0.0,
        help=(
            "Minimum seconds between consecutive NVIDIA API calls per model "
            "instance. Use 1.5-2.0 to pace requests under the rate limit "
            "without burning retry overhead. Default 0.0 (no pacing)."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help=(
            "How many NVIDIA API calls to run in parallel per model. Default 1 "
            "(sequential). Raise it to genuinely parallelise; combine with "
            "--min-call-interval to stay inside a provider rate limit. "
            "Higher = faster but may trigger rate limits."
        ),
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


def build_parser() -> argparse.ArgumentParser:
    """Top-level CLI parser. Single ``build`` flow."""
    parser = argparse.ArgumentParser(
        prog="harness",
        description=(
            "Honest model ranking harness. Loads a (prompt, target_direction) "
            "JSONL, calibrates each shortlisted NVIDIA-hosted model with the "
            "paper's full MIA feature set, and produces a defensible top-3 "
            "ranking with bootstrap CIs."
        ),
    )
    _add_build_arguments(parser)
    return parser


def parse_argv(argv: list[str]) -> argparse.Namespace:
    """Parse CLI arguments. ``build`` is the only mode now.

    Accepts an optional leading ``build`` token for back-compat with
    older scripts that wrote ``harness build --eval-set X``; it gets
    stripped before parsing.
    """
    parser = build_parser()
    if argv and argv[0] == "build":
        argv = argv[1:]
    args = parser.parse_args(argv)
    if args.bootstrap_n < 1:
        parser.error("--bootstrap-n must be >= 1")
    return args


# --- Helpers ------------------------------------------------------------------


def _default_lm_factory(api_key: str, model: str, timeout_s: float) -> NvidiaLM:
    """Construct a real ``NvidiaLM``. Overridable via ``run(..., lm_factory=...)``."""
    return NvidiaLM(api_key=api_key, model=model, timeout_s=timeout_s)


def _make_paced_factory(min_call_interval_s: float):
    """Factory variant that propagates a per-instance call-pacing interval."""
    def _factory(api_key: str, model: str, timeout_s: float) -> NvidiaLM:
        return NvidiaLM(
            api_key=api_key,
            model=model,
            timeout_s=timeout_s,
            min_call_interval_s=min_call_interval_s,
        )
    return _factory


def _resolve_out_dir(raw: str | None) -> Path:
    """Pick the run output directory.

    When the user does not pass ``--out-dir`` we mint a timestamped directory
    under ``runs/`` so multiple runs do not clobber each other.
    """
    if raw:
        return Path(raw)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
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
    sets: they carry ``prompt`` + ``label`` + ``metadata`` while eval rows
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
    *,
    lm_factory: LMFactory | None = None,
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
        lm_factory=lm_factory,
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
    max_workers: int = 1,
) -> ModelEvalResult:
    """Run baseline → MCS → evaluator for one model.

    Returns either a fully-populated ``ModelEvalResult`` or an
    ``uncalibrated`` stub when the control baseline could not be calibrated.
    """
    import time as _time

    model_lm = lm_factory(api_key, model_id, DEFAULT_TIMEOUT_S)

    t0 = _time.monotonic()
    logger.info("runner: %s — building control baseline (%d prompts)…",
                model_id, len(oos_control_rows))
    baseline: ControlBaseline = build_baseline(
        model_lm,
        oos_control_rows,
        ref_lm,
        min_valid=_MCS_HYPERPARAMS["min_valid"],
        max_workers=max_workers,
    )
    if not baseline.is_calibrated:
        logger.warning(
            "runner: %s failed control-baseline calibration "
            "(n_valid=%d < min_valid=%d) after %.1fs; marking uncalibrated and skipping.",
            model_id, baseline.n_valid, baseline.min_valid,
            _time.monotonic() - t0,
        )
        return _make_uncalibrated_stub(model_id)

    t1 = _time.monotonic()
    logger.info("runner: %s — baseline OK (n_valid=%d, %.1fs); training MCS (%d prompts)…",
                model_id, baseline.n_valid, t1 - t0,
                len(is_memorized_rows) + len(oos_control_rows))
    mcs: MCSCalibrator = mcs_train(
        model_lm=model_lm,
        is_memorized=is_memorized_rows,
        oos_control=oos_control_rows,
        baseline=baseline,
        ref_lm=ref_lm,
        min_auc=_MCS_HYPERPARAMS["min_auc"],
        seed=seed,
        max_workers=max_workers,
    )

    t2 = _time.monotonic()
    logger.info("runner: %s — MCS trained (%.1fs); evaluating eval set (%d prompts)…",
                model_id, t2 - t1, len(eval_set.rows))
    result = evaluate_model(
        model_lm=model_lm,
        eval_set=eval_set,
        baseline=baseline,
        mcs=mcs,
        ref_lm=ref_lm,
        holdout_records=None,
        bootstrap_n=bootstrap_n,
        seed=seed,
        max_workers=max_workers,
    )
    logger.info("runner: %s — done in %.1fs (eval %.1fs); parse_rate=%.1f%%, raw_acc=%.3f",
                model_id, _time.monotonic() - t0, _time.monotonic() - t2,
                result.parse_success_rate * 100, result.raw_accuracy.point)
    return result


# --- Manifest assembly --------------------------------------------------------


#: Convention: input file paths are stored in ``Manifest.artifacts`` under
def _build_manifest(
    *,
    seed: int,
    bootstrap_n: int,
    paths: _ResolvedPaths,
    shortlist_models: list[str],
    artifacts: dict[str, Path],
    reference_model: str | None,
    backtest: dict | None = None,
) -> Manifest:
    """Bundle every input the manifest needs into the Manifest dataclass.

    ``backtest`` is the optional cmmd-backtest extension (Req 7.5, 8.2). Pass
    ``None`` (the default) for ordinary harness runs; the manifest then
    serialises to its pre-existing 11-key schema. Pass a dict matching
    design.md § Manifest extension to record the backtest configuration
    alongside the harness manifest.
    """
    mcs_hyperparams = dict(_MCS_HYPERPARAMS)
    mcs_hyperparams["reference_model"] = reference_model

    return Manifest(
        harness_version=HARNESS_VERSION,
        seed=seed,
        eval_set_hash=paths.eval_set_hash,
        control_corpus_hash=paths.oos_control_hash,
        is_memorized_hash=paths.is_memorized_hash,
        cutoffs_hash=paths.cutoffs_hash,
        shortlist=list(shortlist_models),
        composite_score={"formula": COMPOSITE_FORMULA, "gates": dict(GATES)},
        mcs_hyperparams=mcs_hyperparams,
        bootstrap_n=bootstrap_n,
        artifacts={name: str(p) for name, p in artifacts.items()},
        backtest=backtest,
    )


# --- run() entry point --------------------------------------------------------


@dataclass(frozen=True)
class _LoadedInputs:
    eval_path: Path
    eval_set: EvalSet
    cutoffs_path: Path
    cutoffs: dict
    is_path: Path
    is_memorized_rows: list[EvalRow]
    oos_path: Path
    oos_control_rows: list[EvalRow]
    # Hashes captured at load time so the manifest describes the bytes the
    # run actually consumed, even if the files change on disk mid-run.
    eval_hash: str
    cutoffs_hash: str
    is_hash: str
    oos_hash: str


def _load_all_inputs(args: argparse.Namespace) -> _LoadedInputs | int:
    """Validate + load every input file. Returns a bundle, or an exit code on error."""
    eval_path = Path(args.eval_set)
    cutoffs_path = Path(args.cutoffs)
    is_path = Path(args.is_memorized)
    oos_path = Path(args.oos_control)
    for label, p in (("--eval-set", eval_path), ("--cutoffs", cutoffs_path),
                     ("--is-memorized", is_path), ("--oos-control", oos_path)):
        if not p.exists():
            sys.stderr.write(f"ERROR: {label} file not found: {p}\n")
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
    try:
        is_memorized_rows = _load_calibration_rows(is_path)
        oos_control_rows = _load_calibration_rows(oos_path)
    except (ValueError, OSError) as exc:
        sys.stderr.write(f"ERROR: failed to load calibration corpus: {exc}\n")
        return 2

    return _LoadedInputs(
        eval_path=eval_path, eval_set=eval_set,
        cutoffs_path=cutoffs_path, cutoffs=cutoffs,
        is_path=is_path, is_memorized_rows=is_memorized_rows,
        oos_path=oos_path, oos_control_rows=oos_control_rows,
        eval_hash=compute_file_hash(eval_path),
        cutoffs_hash=compute_file_hash(cutoffs_path),
        is_hash=compute_file_hash(is_path),
        oos_hash=compute_file_hash(oos_path),
    )


def _evaluate_all_models(
    *,
    shortlist_models: list[str],
    api_key: str,
    inputs: _LoadedInputs,
    ref_lm: NvidiaLM | None,
    factory: LMFactory,
    args: argparse.Namespace,
) -> list[ModelEvalResult]:
    """Run the per-model evaluation loop with defensive error handling."""
    max_workers = int(getattr(args, "max_workers", 1) or 1)
    results: list[ModelEvalResult] = []
    for model_id in shortlist_models:
        try:
            result = _evaluate_one_model(
                model_id=model_id, api_key=api_key,
                eval_set=inputs.eval_set,
                is_memorized_rows=inputs.is_memorized_rows,
                oos_control_rows=inputs.oos_control_rows,
                ref_lm=ref_lm, lm_factory=factory,
                seed=args.seed, bootstrap_n=args.bootstrap_n,
                max_workers=max_workers,
            )
        except Exception:
            logger.exception(
                "runner: unrecoverable error evaluating model %s; aborting run.",
                model_id,
            )
            raise
        results.append(result)
    return results


def _write_run_artifacts(
    *,
    out_dir: Path,
    results: list[ModelEvalResult],
    scores,
    majority,
    inputs: _LoadedInputs,
    shortlist_models: list[str],
    shortlist_path: Path | None,
    args: argparse.Namespace,
) -> dict[str, Path]:
    """Write records, summary, top3, manifest. Returns the artifact-name → path map."""
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

    # Hashes come from _LoadedInputs (captured at load time), NOT from a
    # fresh read of the files: the manifest must describe the bytes the run
    # consumed even if the files changed on disk during evaluation.
    paths = _ResolvedPaths(
        eval_set=inputs.eval_path,
        eval_set_hash=inputs.eval_hash,
        is_memorized=inputs.is_path,
        is_memorized_hash=inputs.is_hash,
        oos_control=inputs.oos_path,
        oos_control_hash=inputs.oos_hash,
        cutoffs=inputs.cutoffs_path,
        cutoffs_hash=inputs.cutoffs_hash,
    )
    manifest = _build_manifest(
        seed=args.seed, bootstrap_n=args.bootstrap_n, paths=paths,
        shortlist_models=shortlist_models, artifacts=artifacts,
        reference_model=None if args.no_reference else args.reference_model,
    )
    write_manifest(out_dir, manifest)
    return artifacts


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
    if lm_factory is not None:
        factory: LMFactory = lm_factory
    else:
        pace = float(getattr(args, "min_call_interval", 0.0) or 0.0)
        factory = _make_paced_factory(pace) if pace > 0 else _default_lm_factory

    load_dotenv()
    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        sys.stderr.write(
            "ERROR: NVIDIA_API_KEY is not set in the environment. "
            "Add it to your shell or .env file.\n"
        )
        return 2

    loaded = _load_all_inputs(args)
    if isinstance(loaded, int):
        return loaded

    out_dir = _resolve_out_dir(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        shortlist_models, shortlist_path = _resolve_shortlist(
            args, api_key, out_dir, lm_factory=factory
        )
    except (FileNotFoundError, ValueError) as exc:
        sys.stderr.write(f"ERROR: shortlist resolution failed: {exc}\n")
        return 2
    if not shortlist_models:
        sys.stderr.write(
            "ERROR: smoke gate selected no models; aborting run before evaluation.\n"
        )
        return 2

    # Cutoff guard MUST run BEFORE the main eval/baseline/MCS work (Req 2.5).
    try:
        assert_cutoff_safe(loaded.eval_set, shortlist_models, loaded.cutoffs)
    except CutoffViolation as exc:
        sys.stderr.write(f"ERROR: cutoff violation: {exc}\n")
        return 3

    ref_lm: NvidiaLM | None = None
    if not args.no_reference:
        ref_lm = factory(api_key, args.reference_model, DEFAULT_TIMEOUT_S)

    try:
        results = _evaluate_all_models(
            shortlist_models=shortlist_models, api_key=api_key, inputs=loaded,
            ref_lm=ref_lm, factory=factory, args=args,
        )

        majority = compute_majority_baseline(
            loaded.eval_set, bootstrap_n=args.bootstrap_n, seed=args.seed
        )
        scores = composite_score(results, majority)

        artifacts = _write_run_artifacts(
            out_dir=out_dir, results=results, scores=scores, majority=majority,
            inputs=loaded, shortlist_models=shortlist_models,
            shortlist_path=shortlist_path, args=args,
        )
    except Exception as exc:  # pragma: no cover - top-level run guard
        logger.exception("runner: unrecovered error; aborting run.")
        sys.stderr.write(f"ERROR: harness run failed: {exc!r}\n")
        return 1

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
    "parse_argv",
    "run",
]
