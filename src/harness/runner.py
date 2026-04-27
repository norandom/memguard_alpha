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
  the bootstrap resample count, and an artifact-name → path map. Input
  file paths are also stored under ``artifacts`` keys prefixed with
  ``input_`` (``input_eval_set``, ``input_is_memorized``,
  ``input_oos_control``, ``input_cutoffs``) so that ``replay`` can locate
  the original input bytes by name without an extra schema field.
* 10.2 — ``replay --from-manifest PATH --out-dir PATH`` re-runs the
  pipeline with the recorded seed, verifying every input's sha256 against
  the manifest before any work begins. A hash mismatch aborts non-zero
  with a clear, file-naming error (no stale-input runs).
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

What this module deliberately does NOT do (out of scope for Task 5.2):
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
from src.core.manifest import (
    Manifest,
    compute_file_hash,
    read_manifest,
    write_manifest,
)
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

#: Default per-call timeout (seconds). Matches the legacy default and the smoke gate.
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


#: Names of the recognised subcommands. Exposed at module scope so the
#: argv-fallback logic in ``parse_argv`` can detect a missing subcommand
#: without re-introspecting the argparse object.
SUBCOMMANDS: tuple[str, ...] = ("build", "replay")


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
    """Construct the top-level CLI parser with ``build`` and ``replay``
    subcommands.

    The harness CLI surface is documented in design.md (Components and
    Interfaces → harness.runner → CLI Surface):

    * ``harness build [...]`` — the historical end-to-end pipeline. ``build``
      is the default subcommand: passing the build flags directly without
      typing ``build`` (e.g. ``harness --eval-set X --shortlist Y``) is also
      accepted for backwards compatibility (see :func:`parse_argv`).
    * ``harness replay --from-manifest PATH --out-dir PATH`` — re-runs a
      previously recorded run from its persisted manifest (Req 10.2).

    Exposed for testability (so ``test_runner.py`` can parse synthetic argv
    without spawning a subprocess) and for ``harness.py`` at the project
    root.
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

    subparsers = parser.add_subparsers(
        dest="subcommand",
        metavar="{build,replay}",
        help="Subcommand. Defaults to 'build' when omitted.",
    )

    build_p = subparsers.add_parser(
        "build",
        help="Run the full evaluation pipeline (default).",
        description=(
            "Run the full evaluation pipeline: load → smoke shortlist → "
            "control baselines → MCS train → evaluate → rank → top-3."
        ),
    )
    _add_build_arguments(build_p)

    replay_p = subparsers.add_parser(
        "replay",
        help=(
            "Reproduce a prior run from its manifest (Req 10.2). "
            "Verifies input hashes before re-running."
        ),
        description=(
            "Read a manifest produced by a prior 'harness build' run, verify "
            "that every recorded input file still hashes to the value stored "
            "in the manifest, then re-run the pipeline with the recorded seed "
            "and bootstrap_n into a fresh --out-dir. Aborts non-zero on any "
            "input hash mismatch (Req 10.2)."
        ),
    )
    replay_p.add_argument(
        "--from-manifest",
        dest="from_manifest",
        required=True,
        help="Path to a manifest.json produced by a prior 'harness build' run.",
    )
    replay_p.add_argument(
        "--out-dir",
        dest="out_dir",
        required=True,
        help=(
            "Directory to write the replay's artifacts to. Must differ from "
            "the original run's out_dir to avoid clobbering."
        ),
    )

    return parser


def parse_argv(argv: list[str]) -> argparse.Namespace:
    """Parse CLI arguments, defaulting to the ``build`` subcommand when omitted.

    For backwards compatibility, invocations like
    ``harness --eval-set X --shortlist Y`` (no explicit subcommand) are
    rewritten to ``harness build --eval-set X --shortlist Y`` before
    parsing. Any explicit subcommand wins. ``--help`` and ``-h`` at the top
    level are routed through the top-level parser unchanged.
    """
    parser = build_parser()

    # If the first non-flag token is already a known subcommand, parse as-is.
    # Otherwise, prepend `build` so the legacy invocation pattern still works.
    head_is_subcommand = bool(argv) and argv[0] in SUBCOMMANDS
    head_is_help = bool(argv) and argv[0] in ("-h", "--help")
    if argv and not head_is_subcommand and not head_is_help:
        argv = ["build", *argv]

    return parser.parse_args(argv)


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


#: Convention: input file paths are stored in ``Manifest.artifacts`` under
#: keys prefixed with ``input_``. ``replay`` looks them up by these names so
#: it can hash-verify and re-load the same inputs that produced the original
#: ranking. Output artifact keys (``records``, ``summary``, ``top3``,
#: ``manifest``, ``shortlist``) keep their bare names for backwards
#: compatibility with the Task 5.1 manifest schema. (Req 10.2.)
INPUT_ARTIFACT_PREFIX = "input_"
INPUT_EVAL_SET_KEY = INPUT_ARTIFACT_PREFIX + "eval_set"
INPUT_IS_MEMORIZED_KEY = INPUT_ARTIFACT_PREFIX + "is_memorized"
INPUT_OOS_CONTROL_KEY = INPUT_ARTIFACT_PREFIX + "oos_control"
INPUT_CUTOFFS_KEY = INPUT_ARTIFACT_PREFIX + "cutoffs"


def _build_manifest(
    *,
    seed: int,
    bootstrap_n: int,
    paths: _ResolvedPaths,
    shortlist_models: list[str],
    artifacts: dict[str, Path],
    reference_model: str | None,
) -> Manifest:
    """Bundle everything the manifest needs into the Manifest dataclass.

    The ``artifacts`` mapping is augmented with the absolute paths to the
    four input files (eval set, IS-memorized corpus, OOS-control corpus,
    cutoffs registry) keyed under the ``input_*`` namespace so that
    ``replay`` can rehydrate the exact same inputs without an extra
    schema field on Manifest itself.

    The reference model id (or sentinel ``"__none__"`` for ``--no-reference``)
    is recorded in ``mcs_hyperparams`` so ``replay`` can reconstruct the
    reference-model wiring identically.
    """
    artifacts_with_inputs: dict[str, str] = {
        name: str(p) for name, p in artifacts.items()
    }
    artifacts_with_inputs[INPUT_EVAL_SET_KEY] = str(paths.eval_set)
    artifacts_with_inputs[INPUT_IS_MEMORIZED_KEY] = str(paths.is_memorized)
    artifacts_with_inputs[INPUT_OOS_CONTROL_KEY] = str(paths.oos_control)
    artifacts_with_inputs[INPUT_CUTOFFS_KEY] = str(paths.cutoffs)

    mcs_hyperparams = dict(_MCS_HYPERPARAMS)
    # ``None`` survives JSON round-trip as ``null``; tests assert on the
    # rehydrated dict, so we keep the sentinel explicit.
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
        artifacts=artifacts_with_inputs,
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
        reference_model=None if args.no_reference else args.reference_model,
    )
    write_manifest(out_dir, manifest)

    # 10. Render terminal + print artifact paths -----------------------------
    try:
        render_terminal(results, majority, scores)
    except Exception:  # pragma: no cover - terminal rendering must never fail the run
        logger.exception("runner: render_terminal failed; continuing.")

    print_artifact_paths(artifacts)

    return 0


# --- replay() entry point ----------------------------------------------------


#: Process exit code emitted when ``replay`` detects an input-hash mismatch
#: against the manifest. Distinct from ``2`` (missing/invalid input) and ``3``
#: (cutoff violation) so callers can disambiguate the two pre-flight aborts.
EXIT_HASH_MISMATCH = 4


def _read_top3_lines(path: Path) -> list[str]:
    """Return ``top3.md`` as a list of lines, or ``[]`` if the file is missing."""
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8").splitlines()


def _extract_ranked_models(top3_lines: list[str]) -> list[str]:
    """Pull the ranked model order out of a ``top3.md`` file.

    The ranker (see :func:`harness.ranker.write_top3`) renders surviving
    models as numbered list entries of the form
    ``"N. **<model>** — score = ..."``. We extract the model id by
    string-matching on that pattern; the order in the returned list is the
    ranking order.
    """
    out: list[str] = []
    for raw in top3_lines:
        line = raw.strip()
        # Numbered list entries: "1. **<model>** — score = ..."
        if not line or not line[0].isdigit():
            continue
        # Strip leading "<digits>. " so we can pull the bolded model id.
        try:
            after_num = line.split(". ", 1)[1]
        except IndexError:
            continue
        if "**" not in after_num:
            continue
        # Between the first ** and the next ** is the model id.
        try:
            model = after_num.split("**", 2)[1]
        except IndexError:
            continue
        if model and model not in out:
            out.append(model)
    return out


def _reconstruct_args_from_manifest(
    manifest: Manifest, *, out_dir: Path
) -> argparse.Namespace:
    """Build a build-subcommand argparse Namespace from a saved manifest.

    Looks the four input paths up under the ``input_*`` keys in
    ``manifest.artifacts`` (see :data:`INPUT_ARTIFACT_PREFIX`). The
    reference-model wiring is pulled from ``mcs_hyperparams['reference_model']``
    where ``None`` means ``--no-reference`` was set on the original run.

    Raises :class:`KeyError` if the manifest predates Task 5.2 and is missing
    any required ``input_*`` key — callers convert that into a user-facing
    error.
    """
    artifacts = manifest.artifacts
    missing = [
        k
        for k in (
            INPUT_EVAL_SET_KEY,
            INPUT_IS_MEMORIZED_KEY,
            INPUT_OOS_CONTROL_KEY,
            INPUT_CUTOFFS_KEY,
        )
        if k not in artifacts
    ]
    if missing:
        raise KeyError(
            "manifest is missing required input path key(s) "
            f"{sorted(missing)}; this manifest predates the replay feature "
            "and cannot be replayed."
        )

    ref_model_setting = manifest.mcs_hyperparams.get("reference_model")
    no_reference = ref_model_setting is None
    reference_model = ref_model_setting or DEFAULT_REFERENCE_MODEL

    shortlist_csv = ",".join(manifest.shortlist)

    return argparse.Namespace(
        subcommand="build",
        eval_set=artifacts[INPUT_EVAL_SET_KEY],
        is_memorized=artifacts[INPUT_IS_MEMORIZED_KEY],
        oos_control=artifacts[INPUT_OOS_CONTROL_KEY],
        cutoffs=artifacts[INPUT_CUTOFFS_KEY],
        candidates=None,
        shortlist=shortlist_csv,
        out_dir=str(out_dir),
        seed=manifest.seed,
        bootstrap_n=manifest.bootstrap_n,
        reference_model=reference_model,
        no_reference=no_reference,
    )


def replay(
    manifest_path: Path | str,
    out_dir: Path | str,
    *,
    lm_factory: LMFactory | None = None,
) -> int:
    """Re-run the pipeline against a saved manifest (Req 10.2).

    Steps:

    1. Read the manifest at ``manifest_path``.
    2. Locate every input file by the ``input_*`` keys in
       ``manifest.artifacts`` and recompute its sha256. Any mismatch (file
       changed since the manifest was written) returns
       :data:`EXIT_HASH_MISMATCH` and prints an error naming the offending
       file. No HTTP work happens up to this point.
    3. Reconstruct an argparse Namespace from the manifest fields and
       delegate to :func:`run`. The runner re-derives the records, summary,
       top-3, and a fresh manifest under ``out_dir``.
    4. Compare the new ``top3.md`` ordering against the original (when
       reachable via the ``top3`` path recorded in the manifest). When the
       orderings differ, log a WARNING — the spec only requires stability
       within bootstrap CIs, not byte-for-byte equality.

    Parameters
    ----------
    manifest_path:
        Path to a ``manifest.json`` previously written by ``harness build``.
    out_dir:
        Destination directory for the replay's artifacts. Must differ from
        the original ``out_dir`` to avoid clobbering.
    lm_factory:
        Optional override forwarded to :func:`run` so tests can inject a
        deterministic fake LM.

    Returns
    -------
    int
        ``0`` on success, ``2`` for missing/invalid manifest,
        :data:`EXIT_HASH_MISMATCH` (=4) on input hash mismatch, or whatever
        :func:`run` returns when it executes the inner build.
    """
    manifest_p = Path(manifest_path)
    out_path = Path(out_dir)

    # 1. Read the manifest. Failure surfaces a precise error path.
    if not manifest_p.exists():
        sys.stderr.write(
            f"ERROR: --from-manifest file not found: {manifest_p}\n"
        )
        return 2
    try:
        manifest = read_manifest(manifest_p)
    except (ValueError, OSError) as exc:
        sys.stderr.write(
            f"ERROR: failed to read manifest {manifest_p}: {exc}\n"
        )
        return 2

    # 2. Reconstruct the build args namespace. Missing input paths means the
    # manifest predates this feature.
    try:
        replay_args = _reconstruct_args_from_manifest(manifest, out_dir=out_path)
    except KeyError as exc:
        sys.stderr.write(f"ERROR: {exc}\n")
        return 2

    # 3. Hash-verify every input file before any work begins.
    expected_hashes: list[tuple[str, Path, str]] = [
        ("eval_set", Path(replay_args.eval_set), manifest.eval_set_hash),
        (
            "is_memorized",
            Path(replay_args.is_memorized),
            manifest.is_memorized_hash,
        ),
        (
            "oos_control",
            Path(replay_args.oos_control),
            manifest.control_corpus_hash,
        ),
        ("cutoffs", Path(replay_args.cutoffs), manifest.cutoffs_hash),
    ]
    for name, path, expected in expected_hashes:
        if not path.exists():
            sys.stderr.write(
                f"ERROR: replay input '{name}' not found at {path}; cannot "
                "verify hash against manifest.\n"
            )
            return 2
        actual = compute_file_hash(path)
        if actual != expected:
            sys.stderr.write(
                f"ERROR: input hash mismatch for '{name}' at {path}: "
                f"manifest expected {expected}, got {actual}. "
                "The input file changed since the manifest was written; "
                "aborting replay rather than running with stale inputs.\n"
            )
            return EXIT_HASH_MISMATCH

    # 4. Capture the original top-3 ordering (best-effort) BEFORE running, so
    # the inner run() does not overwrite it via a coincident out_dir choice.
    original_top3_path: Path | None = None
    if "top3" in manifest.artifacts:
        candidate = Path(manifest.artifacts["top3"])
        if candidate.exists():
            original_top3_path = candidate
    original_models = (
        _extract_ranked_models(_read_top3_lines(original_top3_path))
        if original_top3_path is not None
        else []
    )

    # 5. Delegate to run() with the reconstructed args. The runner writes a
    # fresh manifest into out_dir; we do not deduplicate against the
    # original, that is the user's responsibility via --out-dir choice.
    rc = run(replay_args, lm_factory=lm_factory)
    if rc != 0:
        return rc

    # 6. Compare orderings. Differences are a warning, not a hard failure.
    new_top3_path = out_path / "top3.md"
    new_models = _extract_ranked_models(_read_top3_lines(new_top3_path))
    if original_models and new_models and original_models != new_models:
        logger.warning(
            "replay: top3 ranking differs from original. "
            "original=%s, replay=%s. The spec requires stability within "
            "bootstrap CIs, not bit-for-bit identity.",
            original_models,
            new_models,
        )
        sys.stderr.write(
            "WARNING: replay ranking differs from original top3.md ordering "
            f"(original={original_models}, replay={new_models}). "
            "This is not a failure — the spec only requires stability "
            "within bootstrap CIs.\n"
        )

    return 0


__all__ = [
    "DEFAULT_REFERENCE_MODEL",
    "DEFAULT_SMOKE_PROMPTS",
    "EXIT_HASH_MISMATCH",
    "HARNESS_VERSION",
    "INPUT_CUTOFFS_KEY",
    "INPUT_EVAL_SET_KEY",
    "INPUT_IS_MEMORIZED_KEY",
    "INPUT_OOS_CONTROL_KEY",
    "LMFactory",
    "build_parser",
    "parse_argv",
    "replay",
    "run",
]
