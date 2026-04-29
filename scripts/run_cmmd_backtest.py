"""End-to-end orchestrator for the cmmd-backtest pipeline (Task 3.2).

Wires every component together:

1. Ensure the eval set exists (build via Task 1.3 if absent).
2. Invoke the harness for ``openai/gpt-oss-20b`` against the eval set,
   producing ``records.jsonl``, ``summary.csv`` and the base
   ``manifest.json``.
3. Run :func:`src.portfolio.cohens_d.compute_cohens_d` on the run dir.
4. Invoke ``scripts/analyze_is_oos_gap.py`` on the run dir as a
   subprocess.
5. Fetch the universe prices via
   :func:`src.portfolio.prices.fetch_universe_prices` for the date span
   of the eval set.
6. Run :func:`src.portfolio.backtest.run_backtest` followed by
   :func:`src.portfolio.backtest.write_backtest_artifacts`.
7. Extend the manifest in place with the ``backtest`` block (Task 3.1).
8. Print every artifact path on success
   (:func:`src.harness.report.print_artifact_paths`).

Failure modes (all exit non-zero, leave the harness artifacts intact):

- harness ``run`` returned non-zero: pass through its exit code.
- MCS calibration failed for ``openai/gpt-oss-20b`` (Req 9.3): exit 4.
- ``scripts/analyze_is_oos_gap.py`` preconditions not met: exit 5.
- :class:`src.portfolio.backtest.BacktestArtifactError` (Req 7.6): exit 6.
- :class:`src.portfolio.prices.PriceFetchError`: exit 7.

In every failure path the ``backtest`` manifest block is NOT written.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

# Make 'src' importable regardless of CWD (mirrors analyze_is_oos_gap.py).
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.core.manifest import read_manifest, write_manifest  # noqa: E402
from src.harness.report import print_artifact_paths  # noqa: E402
from src.harness.runner import LMFactory, parse_argv  # noqa: E402
from src.harness.runner import run as harness_run  # noqa: E402
from src.portfolio.backtest import (  # noqa: E402
    BacktestArtifactError,
    run_backtest,
    write_backtest_artifacts,
)
from src.portfolio.cohens_d import compute_cohens_d  # noqa: E402
from src.portfolio.prices import (  # noqa: E402
    PriceFetchError,
    fetch_universe_prices,
)


def _load_eval_builder() -> Any:
    """Lazy-import ``scripts.build_etf_portfolio_eval`` (Task 1.3).

    The ``scripts/`` directory is not a Python package on the project
    layout (no ``__init__.py``), so we splice it into ``sys.path`` and
    import the module by name. Lazy-loading keeps the import side-effects
    out of the orchestrator's import path so callers that bring their
    own eval set never trigger the builder's network code.
    """
    import importlib

    scripts_dir = _ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    return importlib.import_module("build_etf_portfolio_eval")


logger = logging.getLogger(__name__)


# ----------------------------------------------------------------- constants

#: Universe and cash leg pinned by the spec (design § Goals).
SIGNAL_MODEL: str = "openai/gpt-oss-20b"
UNIVERSE: tuple[str, ...] = ("SWDA.L", "XLK", "IAU", "BIL")
CASH_TICKER: str = "BIL"

#: Harness exit-code bump pattern: 4..7 reserved for cmmd-backtest stages
#: so the caller can distinguish them from the harness's own 0/2/3.
EXIT_MCS_UNCALIBRATED = 4
EXIT_GAP_PRECONDITION = 5
EXIT_BACKTEST_ARTIFACT = 6
EXIT_PRICE_FETCH = 7


# ------------------------------------------------------------------ helpers


def _compute_prompt_hash(prompt: str) -> str:
    """Reproduce ``harness.evaluator._hash_prompt`` (sha256 hex, 16 chars)."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def _read_eval_rows(eval_path: Path) -> list[dict]:
    """Load every non-sentinel row from ``eval_path`` into a list of dicts."""
    rows: list[dict] = []
    with eval_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "_cutoff_date" in row and "prompt" not in row:
                continue
            rows.append(row)
    return rows


def _build_prompt_metadata(eval_rows: list[dict]) -> dict[str, dict[str, str]]:
    """Build ``prompt_hash -> {ticker, date}`` from eval-set rows."""
    metadata: dict[str, dict[str, str]] = {}
    for row in eval_rows:
        prompt = row.get("prompt")
        if not isinstance(prompt, str):
            continue
        md = row.get("metadata") or {}
        ticker = md.get("ticker")
        row_date = md.get("date")
        if not isinstance(ticker, str) or not isinstance(row_date, str):
            continue
        metadata[_compute_prompt_hash(prompt)] = {
            "ticker": ticker,
            "date": row_date[:10],
        }
    return metadata


def _eval_date_span(eval_rows: list[dict]) -> tuple[date, date]:
    """Return ``(min_date, max_date)`` over eval rows' ``metadata.date``."""
    dates: list[date] = []
    for row in eval_rows:
        md = row.get("metadata") or {}
        d = md.get("date")
        if not isinstance(d, str):
            continue
        try:
            dates.append(date.fromisoformat(d[:10]))
        except ValueError:
            continue
    if not dates:
        raise ValueError("eval set has no parseable metadata.date entries.")
    return min(dates), max(dates)


def _load_records_jsonl(records_path: Path) -> list[Any]:
    """Stream ``records.jsonl`` into record-shaped namespaces.

    Returns plain :class:`types.SimpleNamespace` objects exposing the
    attributes :func:`run_backtest` consumes (parse_ok, predicted_direction,
    raw_confidence, p_memorized, prompt_hash). Going through namespaces
    keeps the orchestrator decoupled from the harness.evaluator dataclass
    schema in case extra fields appear later.
    """
    from types import SimpleNamespace

    records: list[Any] = []
    with records_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            records.append(SimpleNamespace(
                model=row.get("model"),
                prompt_hash=row.get("prompt_hash") or "",
                parse_ok=bool(row.get("parse_ok")),
                predicted_direction=row.get("predicted_direction"),
                raw_confidence=row.get("raw_confidence"),
                p_memorized=row.get("p_memorized"),
                target_direction=row.get("target_direction"),
            ))
    return records


def _mcs_calibration_failed(summary_path: Path, *, model: str) -> bool:
    """Return True when MCS calibration failed for ``model`` in summary.csv.

    Detection signal: the model's ``warnings`` cell contains the substring
    ``uncalibrated`` (added by the harness when ``build_baseline`` reports
    ``is_calibrated=False`` — Req 9.3). A missing/empty warnings cell or
    a different warning class (e.g., ``temperature-not-honoured``) is NOT
    a calibration failure.
    """
    import csv

    if not summary_path.exists():
        return False
    with summary_path.open("r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("model", "").strip() != model:
                continue
            warnings = row.get("warnings", "") or ""
            return "uncalibrated" in warnings
    return False


def _gap_preconditions_met(
    records_path: Path,
    cutoffs_path: Path,
    eval_rows: list[dict],
) -> tuple[bool, str]:
    """Validate the IS/OOS gap script's preconditions.

    Returns ``(ok, reason)``. The reason is a clear stderr-friendly
    sentence — empty when ``ok`` is True.
    """
    if not records_path.exists():
        return False, f"missing records file: {records_path}"
    if not cutoffs_path.exists():
        return False, f"missing cutoffs registry: {cutoffs_path}"
    has_iso_date = False
    for row in eval_rows:
        md = row.get("metadata") or {}
        raw = md.get("date")
        if not isinstance(raw, str):
            continue
        try:
            date.fromisoformat(raw[:10])
        except ValueError:
            continue
        has_iso_date = True
        break
    if not has_iso_date:
        return False, (
            "no eval row has a metadata.date parseable as ISO-8601; "
            "scripts/analyze_is_oos_gap.py would fail."
        )
    return True, ""


def _count_is_oos(
    eval_rows: list[dict],
    cutoffs_path: Path,
    *,
    model: str,
    records_path: Path | None = None,
) -> tuple[int, int]:
    """Count parse-OK records that fall IS / OOS for ``model``.

    Joins ``records.jsonl`` to the eval set on ``prompt_hash``,
    filters to ``parse_ok=True``, and labels each surviving record
    IS if its eval-set date is on/before ``cutoffs.yaml[model]``.

    Req 4.1's validation rule wants ``n_is + n_oos`` to equal the
    parse-OK count; that is achieved by filtering on parse_ok=True
    here. Rows whose eval metadata lacks an ISO date are skipped.
    """
    import yaml

    raw = yaml.safe_load(cutoffs_path.read_text(encoding="utf-8"))
    models_block = (raw or {}).get("models") or {}
    cutoff_raw = models_block.get(model)
    if cutoff_raw is None:
        return 0, 0
    cutoff = cutoff_raw if isinstance(cutoff_raw, date) else date.fromisoformat(
        str(cutoff_raw)
    )

    metadata_by_hash: dict[str, dict] = {}
    for row in eval_rows:
        prompt = row.get("prompt")
        if not isinstance(prompt, str):
            continue
        ph = _compute_prompt_hash(prompt)
        metadata_by_hash[ph] = row.get("metadata") or {}

    if records_path is None or not records_path.exists():
        return 0, 0

    n_is = 0
    n_oos = 0
    with records_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("model") != model or not rec.get("parse_ok"):
                continue
            md = metadata_by_hash.get(rec.get("prompt_hash") or "")
            if not md:
                continue
            raw_date = md.get("date")
            if not isinstance(raw_date, str):
                continue
            try:
                d = date.fromisoformat(raw_date[:10])
            except ValueError:
                continue
            if d <= cutoff:
                n_is += 1
            else:
                n_oos += 1
    return n_is, n_oos


def _run_is_oos_gap_subprocess(
    run_dir: Path,
    eval_path: Path,
    cutoffs_path: Path,
) -> int:
    """Invoke ``scripts/analyze_is_oos_gap.py`` as a subprocess.

    Returns the subprocess return code so the orchestrator can surface
    failures cleanly.
    """
    cmd = [
        sys.executable,
        str(_ROOT / "scripts" / "analyze_is_oos_gap.py"),
        str(run_dir),
        str(eval_path),
        str(cutoffs_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.stdout:
        sys.stdout.write(proc.stdout)
    if proc.returncode != 0 and proc.stderr:
        sys.stderr.write(proc.stderr)
    return proc.returncode


def _resolve_run_dir(raw: Path | None) -> Path:
    """Mint a timestamped run dir under ``runs/cmmd_<UTC>`` if not given."""
    if raw is not None:
        return Path(raw)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return Path("runs") / f"cmmd_{ts}"


def _ensure_eval_set(eval_path: Path) -> int:
    """Build the eval set if missing; return non-zero on builder failure."""
    if eval_path.exists():
        return 0
    logger.info("eval set %s missing; building via Task 1.3.", eval_path)
    eval_builder = _load_eval_builder()
    rc = eval_builder.main(out_path=eval_path)
    if rc != 0:
        sys.stderr.write(
            f"ERROR: eval-set builder exited with code {rc}; aborting.\n"
        )
    return rc


def _invoke_harness(
    *,
    eval_path: Path,
    is_memorized_path: Path,
    oos_control_path: Path,
    cutoffs_path: Path,
    run_dir: Path,
    seed: int,
    bootstrap_n: int,
    lm_factory: LMFactory | None,
    extra_argv: list[str],
) -> int:
    """Build the harness argv for ``openai/gpt-oss-20b`` and invoke ``run``."""
    argv: list[str] = [
        "--eval-set", str(eval_path),
        "--shortlist", SIGNAL_MODEL,
        "--is-memorized", str(is_memorized_path),
        "--oos-control", str(oos_control_path),
        "--cutoffs", str(cutoffs_path),
        "--out-dir", str(run_dir),
        "--seed", str(seed),
        "--bootstrap-n", str(bootstrap_n),
        "--no-reference",
    ]
    argv.extend(extra_argv)
    args = parse_argv(argv)
    if lm_factory is not None:
        return harness_run(args, lm_factory=lm_factory)
    return harness_run(args)


def _build_backtest_block(
    *,
    cmmd_threshold: float | None,
    seed: int,
    bootstrap_n: int,
    n_is: int,
    n_oos: int,
    artifact_paths: dict[str, Path],
    cohens_d_paths: dict[str, Path] | None,
    is_oos_gap_path: Path,
) -> dict:
    """Assemble the manifest's ``backtest`` block (design § Manifest extension)."""
    artifacts: dict[str, str] = {
        "backtest_summary_csv": "backtest_summary.csv",
        "backtest_summary_md": "backtest_summary.md",
        "equity_curves_csv": "equity_curves.csv",
        "equity_curves_png": "equity_curves.png",
        "daily_returns_csv": "daily_returns.csv",
    }
    # Make sure the path map only carries the bare filename (the manifest
    # is run-dir-relative).
    for key, path in artifact_paths.items():
        artifacts[key] = path.name
    if cohens_d_paths is not None:
        artifacts["cohens_d_csv"] = "cohens_d.csv"
        artifacts["cohens_d_md"] = "cohens_d.md"
    if is_oos_gap_path.exists():
        artifacts["is_oos_gap_csv"] = is_oos_gap_path.name
    return {
        "signal_model": SIGNAL_MODEL,
        "universe": list(UNIVERSE),
        "cash_ticker": CASH_TICKER,
        "cmmd_quantile": 0.80,
        "cmmd_threshold_value": (
            None if cmmd_threshold is None else float(cmmd_threshold)
        ),
        "fees_one_way": 0.00075,
        "init_cash": 1.0,
        "seed": int(seed),
        "bootstrap_n": int(bootstrap_n),
        "n_is_rows": int(n_is),
        "n_oos_rows": int(n_oos),
        "artifacts": artifacts,
    }


def _extend_manifest(run_dir: Path, backtest_block: dict) -> Path:
    """Read ``manifest.json``, set ``manifest.backtest``, write it back."""
    manifest_path = run_dir / "manifest.json"
    manifest = read_manifest(manifest_path)
    new_manifest = dataclasses.replace(manifest, backtest=backtest_block)
    return write_manifest(run_dir, new_manifest)


# --------------------------------------------------------------------- main


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_cmmd_backtest",
        description=(
            "End-to-end orchestrator: ensure eval set, run harness on "
            f"{SIGNAL_MODEL}, compute Cohen's d, IS/OOS gap, fetch prices, "
            "run backtest, extend manifest with the cmmd-backtest block."
        ),
    )
    parser.add_argument(
        "--eval-set", type=Path,
        default=Path("data/eval/etf_portfolio.jsonl"),
    )
    parser.add_argument(
        "--cutoffs", type=Path, default=Path("data/cutoffs.yaml"),
    )
    parser.add_argument(
        "--is-memorized", type=Path,
        default=Path("data/calibration/is_memorized.jsonl"),
    )
    parser.add_argument(
        "--oos-control", type=Path,
        default=Path("data/calibration/oos_control.jsonl"),
    )
    parser.add_argument(
        "--run-dir", type=Path, default=None,
        help="Directory to write artifacts to. Defaults to runs/cmmd_<UTC>.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap-n", type=int, default=1000)
    return parser


def main(
    *,
    eval_path: Path = Path("data/eval/etf_portfolio.jsonl"),
    cutoffs_path: Path = Path("data/cutoffs.yaml"),
    is_memorized_path: Path = Path("data/calibration/is_memorized.jsonl"),
    oos_control_path: Path = Path("data/calibration/oos_control.jsonl"),
    run_dir: Path | None = None,
    seed: int = 0,
    bootstrap_n: int = 1000,
    fmp_fetcher: Callable[..., Any] = fetch_universe_prices,
    lm_factory: LMFactory | None = None,
    harness_extra_argv: list[str] | None = None,
) -> int:
    """Run the full cmmd-backtest orchestration. Returns a process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    rc = _ensure_eval_set(eval_path)
    if rc != 0:
        return rc

    target_dir = _resolve_run_dir(run_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    rc = _invoke_harness(
        eval_path=eval_path,
        is_memorized_path=is_memorized_path,
        oos_control_path=oos_control_path,
        cutoffs_path=cutoffs_path,
        run_dir=target_dir,
        seed=seed,
        bootstrap_n=bootstrap_n,
        lm_factory=lm_factory,
        extra_argv=list(harness_extra_argv or []),
    )
    if rc != 0:
        return rc

    summary_path = target_dir / "summary.csv"
    if _mcs_calibration_failed(summary_path, model=SIGNAL_MODEL):
        sys.stderr.write(
            f"ERROR: MCS calibration failed for {SIGNAL_MODEL}; "
            "aborting backtest. Check summary.csv warnings column.\n"
        )
        return EXIT_MCS_UNCALIBRATED

    return _run_post_harness_stages(
        target_dir=target_dir,
        eval_path=eval_path,
        cutoffs_path=cutoffs_path,
        seed=seed,
        bootstrap_n=bootstrap_n,
        fmp_fetcher=fmp_fetcher,
    )


def _run_post_harness_stages(
    *,
    target_dir: Path,
    eval_path: Path,
    cutoffs_path: Path,
    seed: int,
    bootstrap_n: int,
    fmp_fetcher: Callable[..., Any],
) -> int:
    """Cohen's d, gap, prices, backtest, manifest extension. ≤120 lines."""
    eval_rows = _read_eval_rows(eval_path)

    logger.info("compute_cohens_d start")
    cohens_d_paths: dict[str, Path] | None = None
    try:
        compute_cohens_d(target_dir, eval_path, cutoffs_path)
        cohens_d_paths = {
            "cohens_d_csv": target_dir / "cohens_d.csv",
            "cohens_d_md": target_dir / "cohens_d.md",
        }
    except FileNotFoundError as exc:
        sys.stderr.write(f"ERROR: compute_cohens_d failed: {exc}\n")
        return 2

    records_path = target_dir / "records.jsonl"
    ok, reason = _gap_preconditions_met(records_path, cutoffs_path, eval_rows)
    if not ok:
        sys.stderr.write(
            f"ERROR: IS/OOS gap preconditions not met: {reason}\n"
        )
        return EXIT_GAP_PRECONDITION

    logger.info("analyze_is_oos_gap start")
    gap_rc = _run_is_oos_gap_subprocess(target_dir, eval_path, cutoffs_path)
    if gap_rc != 0:
        sys.stderr.write(
            f"ERROR: scripts/analyze_is_oos_gap.py exited with {gap_rc}.\n"
        )
        return gap_rc
    is_oos_gap_path = target_dir / "is_oos_gap.csv"

    start, end = _eval_date_span(eval_rows)
    logger.info("fetch_universe_prices start (%s..%s)", start, end)
    try:
        prices = fmp_fetcher(list(UNIVERSE), start, end)
    except PriceFetchError as exc:
        sys.stderr.write(f"ERROR: price fetch failed: {exc}\n")
        return EXIT_PRICE_FETCH

    logger.info("run_backtest start")
    prompt_metadata = _build_prompt_metadata(eval_rows)
    records = _load_records_jsonl(records_path)
    result = run_backtest(
        records,
        prices,
        prompt_metadata,
        cmmd_quantile=0.80,
        fees_one_way=0.00075,
        init_cash=1.0,
        seed=seed,
        bootstrap_n=bootstrap_n,
    )

    try:
        artifact_paths = write_backtest_artifacts(result, target_dir)
    except BacktestArtifactError as exc:
        sys.stderr.write(f"ERROR: backtest artifact write failed: {exc}\n")
        return EXIT_BACKTEST_ARTIFACT

    n_is, n_oos = _count_is_oos(
        eval_rows, cutoffs_path, model=SIGNAL_MODEL,
        records_path=target_dir / "records.jsonl",
    )
    backtest_block = _build_backtest_block(
        cmmd_threshold=result.cmmd.cmmd_threshold,
        seed=seed,
        bootstrap_n=bootstrap_n,
        n_is=n_is,
        n_oos=n_oos,
        artifact_paths=artifact_paths,
        cohens_d_paths=cohens_d_paths,
        is_oos_gap_path=is_oos_gap_path,
    )
    _extend_manifest(target_dir, backtest_block)
    print_artifact_paths(_summarise_artifact_paths(
        target_dir, artifact_paths, cohens_d_paths, is_oos_gap_path,
    ))
    return 0


def _summarise_artifact_paths(
    run_dir: Path,
    backtest_paths: dict[str, Path],
    cohens_d_paths: dict[str, Path] | None,
    is_oos_gap_path: Path,
) -> dict[str, Path]:
    """Bundle every artifact path for the final stdout summary."""
    paths: dict[str, Path] = {
        "records": run_dir / "records.jsonl",
        "summary": run_dir / "summary.csv",
        "manifest": run_dir / "manifest.json",
    }
    if cohens_d_paths is not None:
        paths.update(cohens_d_paths)
    if is_oos_gap_path.exists():
        paths["is_oos_gap_csv"] = is_oos_gap_path
        gap_md = is_oos_gap_path.with_suffix(".md")
        if gap_md.exists():
            paths["is_oos_gap_md"] = gap_md
    paths.update(backtest_paths)
    return paths


def _cli(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    return main(
        eval_path=args.eval_set,
        cutoffs_path=args.cutoffs,
        is_memorized_path=args.is_memorized,
        oos_control_path=args.oos_control,
        run_dir=args.run_dir,
        seed=args.seed,
        bootstrap_n=args.bootstrap_n,
    )


if __name__ == "__main__":
    raise SystemExit(_cli(sys.argv[1:]))
