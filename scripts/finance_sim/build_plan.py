"""Build and freeze the study plan of the finance simulation (task 7.3; Requirements 8.1, 8.2, 8.4, 8.10).

Runs the item assembly (7.1) with the keys ``FRED_API_KEY`` and ``FMP_API_TOKEN``
taken from the process environment by name only, freezes the complete plan (7.2
and 7.3 sections) and writes, under the private root ``runs/finance-sim/``:

- ``plan/items.json``: every study item with its sources (private: it holds the
  realized returns and BAA10Y levels);
- ``plan/plan.json``: the frozen plan, identified by ``StudyPlan.plan_sha256``;
- ``plan/stage_limits/<stage>.json``: the serving arm's six StageLimits files in
  execution order, each carrying ``plan_sha256``, validated against the serving
  manifest's reviewed ceiling;
- ``plan/receipt.json``: the plan's ``plan_sha256``, the items and returns file
  hashes, the recomputed EDGAR file hashes, the arm-to-registry-id map and the
  serving limits with the plan, manifest and stage-limit hashes the approval
  checks. Every reference to the plan is ``plan_sha256``, never a file hash.

Refusals, before any request or write: a frozen plan already at the target (a
plan is frozen once), a private root inside the repository but outside ``runs/``
or a drive root, a missing key, an arm map that does not name both arms, an arm
whose registry id has no included model (no sourced cutoff), an EDGAR file that
differs from its corpus manifest, an SEC accession that repeats across the four
EDGAR files, and a serving manifest other than the reviewed Gemma 3 4B one.
After the assembly the plan must hold an item for every month and EDGAR id, and
nothing else.

Run from the repository: ``uv run --no-sync python -m scripts.finance_sim.build_plan``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from deploy.modal.manifest import GEMMA_MANIFEST
from deploy.modal.manifest import load_plan as load_modal_plan
from deploy.modal.stage_limits import (
    STAGES,
    WEIGHT_STAGING,
    parse_stage_limits,
    require_serving_limits,
    require_staging_limits,
)
from recall_guard.core.profiles import load_profile
from scripts.finance_sim import items as item_assembly
from scripts.finance_sim import plan as plan_module
from scripts.finance_sim.plan import EdgarRow, MonthClass, PlanChangedError, StudyPlan

ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = ROOT / "runs"
DEFAULT_MANIFEST = ROOT / plan_module.SERVING_MANIFEST
#: The plan seed (7.2 split); the user may choose another before the freeze at 9.2.
DEFAULT_SEED = 20261110
RECEIPT_SCHEMA = "finance-sim-plan-receipt/1"
EXIT_FROZEN, EXIT_REFUSED, EXIT_SOURCE_ERROR = 0, 1, 3
#: An SEC accession number, the stable metadata field EDGAR row ids derive from.
_ACCESSION = re.compile(r"\d{10}-\d{2}-\d{6}")
_EDGAR_SPLIT_TAG = {"calibration": "calibration", "negative_control": "held_out", "unused": "held_out"}


class PlanBuildError(ValueError):
    """The plan cannot be built from these inputs; nothing was requested or written for it."""


@dataclass(frozen=True)
class BuildResult:
    plan: StudyPlan
    root: Path
    plan_path: Path
    items_path: Path
    receipt_path: Path
    stage_limit_paths: tuple[Path, ...]
    receipt: Mapping[str, Any]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _private_root(value: Path | str) -> Path:
    """The private root as an absolute path; never a drive root, and inside the repository only under runs/."""
    root = Path(value).resolve()
    if root == Path(root.anchor) or root.parent == root:
        raise PlanBuildError(f"the private root must not be a drive root; got {root}")
    if root.is_relative_to(ROOT) and not root.is_relative_to(RUNS_ROOT):
        raise PlanBuildError(f"a private root inside the repository must be under {RUNS_ROOT}; got {root}")
    return root


def _existing_plan(path: Path) -> str:
    try:
        recorded = json.loads(path.read_text(encoding="utf-8")).get("plan_sha256")
    except (OSError, ValueError, AttributeError):
        recorded = None
    return recorded if isinstance(recorded, str) else "with an unreadable plan_sha256"


def _registry_ids(value: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(plan_module.ARMS):
        raise PlanBuildError(f"the arm-to-registry-id map must name both arms {list(plan_module.ARMS)}; "
                             f"got {sorted(map(str, value)) if isinstance(value, Mapping) else value!r}")
    if any(not isinstance(model, str) or not model.strip() for model in value.values()):
        raise PlanBuildError("every arm must map to a registry id")
    return {arm: value[arm] for arm in plan_module.ARMS}


def _key(name: str, environ: Mapping[str, str] | None) -> str:
    try:
        return item_assembly.api_key_from_environment(name, environ)
    except ValueError as error:
        raise PlanBuildError(str(error)) from None


def edgar_rows(edgar_dir: Path | str = item_assembly.DEFAULT_EDGAR_DIR) -> tuple[list[EdgarRow], dict[str, str],
                                                                                   dict[str, dict[str, Any]]]:
    """The plan's EDGAR rows and file hashes, read from the four corpus files of the manifest.

    Each file's SHA-256 is recomputed from its bytes and must equal the corpus
    manifest's; row counts and labels must agree with it. Row ids derive from the
    SEC accession in each row's metadata, ``edgar-<role>-<accession>`` (the item
    ids of 7.1), and an accession must not repeat across the four files.
    """
    edgar_dir = Path(edgar_dir)
    try:
        manifest = json.loads((edgar_dir / "manifest.json").read_text(encoding="utf-8"))
        entries = manifest["files"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise PlanBuildError(f"the EDGAR corpus manifest in {edgar_dir} is unreadable: {error!r}") from None
    rows: list[EdgarRow] = []
    files: dict[str, str] = {}
    receipt: dict[str, dict[str, Any]] = {}
    seen: dict[str, str] = {}
    for name in plan_module.EDGAR_FILES:
        role = item_assembly.EDGAR_FILES[name]
        entry = entries.get(name) if isinstance(entries, Mapping) else None
        if not isinstance(entry, Mapping) or entry.get("label") not in (0, 1) or isinstance(entry.get("label"), bool):
            raise PlanBuildError(f"the EDGAR corpus manifest has no labelled entry for {name}")
        data = (edgar_dir / name).read_bytes()
        digest = _sha256_bytes(data)
        if digest != entry.get("sha256"):
            raise PlanBuildError(f"EDGAR file {name} does not match its manifest sha256, so the frozen row set "
                                 f"changed")
        lines = [(number, line) for number, line in enumerate(data.decode("utf-8").split("\n"), start=1)
                 if line.strip()]
        if len(lines) != entry.get("rows"):
            raise PlanBuildError(f"EDGAR file {name} holds {len(lines)} rows, but the manifest gives "
                                 f"{entry.get('rows')!r}")
        for number, line in lines:
            where = f"EDGAR file {name} line {number}"
            try:
                row = json.loads(line)
                accession = row["metadata"]["accession"]
                label = row["label"]
            except (ValueError, KeyError, TypeError):
                raise PlanBuildError(f"{where} has no label or metadata accession") from None
            if not isinstance(accession, str) or _ACCESSION.fullmatch(accession) is None:
                raise PlanBuildError(f"{where} has no SEC accession number")
            if isinstance(label, bool) or label != entry["label"]:
                raise PlanBuildError(f"{where} has label {label!r}, but the manifest gives the file label "
                                     f"{entry['label']}")
            if accession in seen:
                raise PlanBuildError(f"SEC accession {accession} appears in both {seen[accession]} and {name}; "
                                     f"EDGAR row ids must be unique across the four files")
            seen[accession] = name
            rows.append(EdgarRow(f"edgar-{role}-{accession}", label, _EDGAR_SPLIT_TAG[role]))
        files[name] = digest
        receipt[name] = {"role": role, "sha256": digest, "manifest_sha256": entry["sha256"], "rows": len(lines)}
    return rows, files, receipt


def arm_facts(profiles: Mapping[str, str] = plan_module.ARM_PROFILES) -> dict[str, dict[str, Any]]:
    """What the plan freezes from each arm's provider profile."""
    facts = {}
    for arm, profile_id in profiles.items():
        profile = load_profile(profile_id)
        parameters = profile.model_config.parameters
        facts[arm] = {"profile_id": profile.profile_id, "served_model": profile.model_config.model,
                      "top_logprobs": parameters.get("top_logprobs"), "max_tokens": parameters.get("max_tokens"),
                      "estimate_sources": profile.estimate_sources}
    return facts


def serving_manifest(path: Path | str = DEFAULT_MANIFEST):
    """The reviewed Gemma 3 4B serving manifest: its plan record, the deploy plan and its cost block."""
    path = Path(path)
    data = path.read_bytes()
    modal_plan = load_modal_plan(path)
    if modal_plan.serving_manifest != GEMMA_MANIFEST:
        raise PlanBuildError(f"the serving arm runs the reviewed Gemma 3 4B manifest {GEMMA_MANIFEST}; "
                             f"{path} is {modal_plan.serving_manifest}")
    record = plan_module.ServingManifest.from_manifest(modal_plan.serving_manifest, data)
    return record, modal_plan, json.loads(data)["cost"]


def month_classes(assembly: item_assembly.ItemAssembly) -> list[MonthClass]:
    """One class per assembled macro item: pre-cutoff or not for each included model."""
    included = assembly.cutoffs.included
    return [MonthClass(f"{item.decision_date:%Y-%m}",
                       {model_id: cutoff.pre_cutoff[item.decision_date] for model_id, cutoff in included.items()})
            for item in assembly.macro]


def item_hashes(record: Mapping[str, Any]) -> dict[str, str]:
    """Per study item, the SHA-256 of its row and prompt hashes, from the ``items.json`` content."""
    entries = [*record["macro_items"], *record["edgar"]["calibration"], *record["edgar"]["negative_control"]]
    hashes = {}
    for entry in entries:
        if entry["item_id"] in hashes:
            raise PlanBuildError(f"item {entry['item_id']} appears twice in the assembly")
        hashes[entry["item_id"]] = plan_module.sha256_of({"prompt_sha256": entry["prompt_sha256"],
                                                          "row_sha256": entry["row_sha256"]})
    return hashes


def check_coverage(plan: StudyPlan) -> None:
    """The plan's items must cover every month and every EDGAR calibration and negative-control id, and no more."""
    expected = {f"macro-{month}" for month in plan.months.months}
    expected |= set(plan.edgar.calibration_memorized) | set(plan.edgar.calibration_control)
    expected |= set(plan.edgar.negative_control)
    missing, extra = sorted(expected - set(plan.items)), sorted(set(plan.items) - expected)
    if missing:
        raise PlanBuildError(f"the plan has no item for {missing}")
    if extra:
        raise PlanBuildError(f"items not in the plan's months or EDGAR sets: {extra}")


def checked_stage_limits(plan: StudyPlan, modal_plan, cost: Mapping[str, Any]) -> tuple[tuple[str, bytes], ...]:
    """The StageLimits files, each validated as the staging step and the serve mode will read it."""
    files = plan_module.stage_limits_files(plan)
    if tuple(stage for stage, _ in files) != STAGES:
        raise PlanBuildError(f"the serving stages must be {list(STAGES)} in execution order")
    for stage, data in files:
        limits = parse_stage_limits(data, modal_plan)
        if limits.plan_sha256 != plan.plan_sha256:
            raise PlanBuildError(f"the {stage} limits do not carry the plan's plan_sha256")
        if stage == WEIGHT_STAGING:
            require_staging_limits(limits, modal_plan, cost)
        else:
            require_serving_limits(limits)
    return files


def _write_once(path: Path, data: bytes) -> None:
    """Write ``data`` atomically; an identical file is kept, a different one is refused."""
    if path.exists():
        if path.read_bytes() == data:
            return
        raise PlanBuildError(f"refusing to replace {path}, which another plan wrote")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, partial = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
        os.replace(partial, path)
    except BaseException:
        Path(partial).unlink(missing_ok=True)
        raise


def build(*, environ: Mapping[str, str] | None = None, private_root: Path | str = item_assembly.PRIVATE_ROOT,
          seed: int = DEFAULT_SEED, created_at: datetime | None = None, manifest: Path | str = DEFAULT_MANIFEST,
          registry_ids: Mapping[str, str] = plan_module.ARM_REGISTRY_IDS,
          cutoffs_path: Path | str = item_assembly.DEFAULT_CUTOFFS,
          edgar_dir: Path | str = item_assembly.DEFAULT_EDGAR_DIR, today: date | None = None,
          transport: item_assembly.Transport = item_assembly.requests_transport) -> BuildResult:
    """Assemble the items, freeze the complete plan and write it with its stage limits and receipt."""
    root = _private_root(private_root)
    paths = item_assembly.PrivatePaths(root)
    plan_path = root / "plan" / "plan.json"
    if plan_path.exists():
        raise PlanBuildError(f"a frozen plan already exists at {plan_path} (plan_sha256 {_existing_plan(plan_path)}); "
                             f"a plan is frozen once and is never rebuilt in place")
    created = datetime.now(UTC).replace(microsecond=0) if created_at is None else created_at
    if not isinstance(created, datetime) or created.utcoffset() is None:
        raise PlanBuildError("created_at must be a timezone-aware datetime")
    fred_key = _key(item_assembly.FRED_KEY_NAME, environ)
    fmp_key = _key(item_assembly.FMP_KEY_NAME, environ)
    arms_map = _registry_ids(registry_ids)
    model_ids = tuple(dict.fromkeys(arms_map.values()))
    cutoffs = item_assembly.classify_cutoffs(item_assembly.decision_months(), model_ids, registry=cutoffs_path)
    for arm, model_id in arms_map.items():
        if model_id in cutoffs.excluded:
            raise PlanBuildError(f"arm {arm!r} has no included model: {cutoffs.excluded[model_id]}")
    rows, edgar_files, edgar_receipt = edgar_rows(edgar_dir)
    arms = plan_module.study_arms(arm_facts(), registry_ids=arms_map)
    manifest_record, modal_plan, cost = serving_manifest(manifest)

    assembly = item_assembly.assemble_items(
        fred_api_key=fred_key, fmp_api_key=fmp_key, model_ids=model_ids, private_root=root,
        cutoffs_path=cutoffs_path, edgar_dir=edgar_dir, today=today, transport=transport)
    assembled_files = {name: entry["sha256"] for name, entry in assembly.edgar.files.items()}
    if assembled_files != edgar_files:
        raise PlanBuildError("the EDGAR files changed while the plan was built")
    record = assembly.to_dict()
    frozen = plan_module.freeze_plan(
        month_classes(assembly), rows, seed=seed, created_at=created, items=item_hashes(record),
        edgar_files=edgar_files, excluded_models=dict(assembly.cutoffs.excluded), arms=arms,
        serving_manifest=manifest_record)
    check_coverage(frozen)
    stage_files = checked_stage_limits(frozen, modal_plan, cost)

    items_sha256 = item_assembly.write_items(assembly)
    plan_module.write_plan(frozen, plan_path)
    stage_paths = []
    serving = plan_module.serving_limits_record(frozen)
    for entry, (stage, data) in zip(serving["stages"], stage_files, strict=True):
        path = root / "plan" / "stage_limits" / f"{stage}.json"
        _write_once(path, data)
        stage_paths.append(path)
        entry["path"] = path.relative_to(root).as_posix()
    split = frozen.months
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "frozen",
        "plan_sha256": frozen.plan_sha256,
        "created_at": frozen.created_at,
        "seed": frozen.seed,
        "files": {
            "plan": plan_path.relative_to(root).as_posix(),
            "items": {"path": paths.items.relative_to(root).as_posix(), "sha256": items_sha256},
            "realized_returns": {"path": paths.realized_returns.relative_to(root).as_posix(),
                                 "sha256": _sha256_bytes(paths.realized_returns.read_bytes())},
        },
        "arms": dict(frozen.arms.registry_ids),
        "excluded_models": dict(frozen.excluded_models),
        "edgar_files": edgar_receipt,
        "serving_limits": serving,
        "sampling_limits": [budget.to_dict() for budget in frozen.limits.stages[plan_module.SAMPLING_ARM]],
        "evidence_requests": frozen.limits.evidence_requests,
        "counts": {
            "months": len(split.months), "calibration": len(split.calibration), "pilot": len(split.pilot),
            "evaluation": len(split.evaluation),
            "pre_cutoff_evaluation": frozen.pilot.pre_cutoff_evaluation_months,
            "post_cutoff_evaluation": frozen.pilot.post_cutoff_evaluation_months,
            "excluded_months": len(assembly.excluded_months),
            "realized_missing": sum(item.outcome.realized is None for item in assembly.macro),
            "edgar_calibration": len(frozen.edgar.calibration_memorized) + len(frozen.edgar.calibration_control),
            "negative_control": len(frozen.edgar.negative_control), "edgar_unused": len(frozen.edgar.unused),
            "items": len(frozen.items),
        },
        "credentials": {item_assembly.FRED_KEY_NAME: "process environment, by name",
                        item_assembly.FMP_KEY_NAME: "process environment, by name"},
        "licensing": item_assembly.LICENSING_NOTE,
    }
    receipt_path = root / "plan" / "receipt.json"
    text = json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    _write_once(receipt_path, text.encode("utf-8"))
    return BuildResult(plan=frozen, root=root, plan_path=plan_path, items_path=paths.items,
                       receipt_path=receipt_path, stage_limit_paths=tuple(stage_paths), receipt=receipt)


def _timestamp(text: str) -> datetime:
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an ISO 8601 timestamp: {text!r}") from None
    if moment.utcoffset() is None:
        raise argparse.ArgumentTypeError(f"created_at needs a timezone: {text!r}")
    return moment


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:  # refusals go to stderr with the refusal exit code
        self.print_usage(sys.stderr)
        print(f"refused: {message}", file=sys.stderr)
        raise SystemExit(EXIT_REFUSED)


def main(argv: Sequence[str] | None = None, *, environ: Mapping[str, str] | None = None,
         transport: item_assembly.Transport | None = None, today: date | None = None) -> int:
    """Build and freeze the plan; 0 frozen, 1 refused, 3 a data source failed (see ``retryable``)."""
    parser = _Parser(description="Assemble the study items, freeze the study plan and write it with its "
                                 "stage limits and receipt under the private root.")
    parser.add_argument("--private-root", type=Path, default=item_assembly.PRIVATE_ROOT,
                        help="private output root (default runs/finance-sim)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="the plan seed of the month split")
    parser.add_argument("--created-at", type=_timestamp, default=None,
                        help="freeze timestamp, ISO 8601 with a timezone (default now, UTC)")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="the serving manifest")
    try:
        args = parser.parse_args(argv)
    except SystemExit as exit_:
        return int(exit_.code or 0)
    try:
        result = build(environ=environ, private_root=args.private_root, seed=args.seed, created_at=args.created_at,
                       manifest=args.manifest, today=today,
                       transport=item_assembly.requests_transport if transport is None else transport)
    except item_assembly.ItemSourceError as error:
        print(f"source error: {error} (source {error.source}, code {error.code}, retryable {error.retryable})",
              file=sys.stderr)
        return EXIT_SOURCE_ERROR
    except (PlanBuildError, PlanChangedError, ValueError, TypeError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return EXIT_REFUSED
    print(json.dumps({"status": "frozen", "plan_sha256": result.plan.plan_sha256, "plan": str(result.plan_path),
                      "receipt": str(result.receipt_path),
                      "stage_limits": [str(path) for path in result.stage_limit_paths]}, indent=2))
    return EXIT_FROZEN


if __name__ == "__main__":
    raise SystemExit(main())
