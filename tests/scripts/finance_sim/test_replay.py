"""Offline worked-example replay and licensing guard (task 8.4; Requirements 10.1 and 10.7).

Fixtures only. One frozen plan is built on a tiny EDGAR corpus written under
``tmp_path`` (the four files and their manifest, so the real corpus loader
reads the negative-control releases) and macro rows with distinctive raw
levels, FMP-style closes and DTB3 rates; the probe (sampling) arm is calibrated
and runs its full run and its negative control through the real 7.6 and 7.7
path on the generation and full-distribution doubles of the 7.12 tests, with
the 7.4 draw log and evidence files; the serving arm's full run is derived from
synthetic temperature-0 draws without a request. The stored outputs of tasks
7.9 to 7.12 are then computed in this process, and the replay command runs in
fresh subprocesses with sockets blocked against those records. Nothing is
written under ``runs/`` or ``docs/``. The Agg backend is selected before
anything imports pyplot.
"""

from __future__ import annotations

import calendar
import csv
import hashlib
import io
import itertools
import json
import re
import shutil
import socket
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402

from recall_guard.core.distribution import RecordedDraw  # noqa: E402
from recall_guard.dataset.macro_regime import (  # noqa: E402
    MacroRow,
    RealizedReturns,
    render_anonymised,
    render_identifying,
)
from recall_guard.harness.guarded_decision import GuardIdentity  # noqa: E402
from recall_guard.mia.estimate_source import EstimateSource, SourceLabel  # noqa: E402
from scripts.finance_sim.analysis import DID, pre_registered_analysis, write_analysis  # noqa: E402
from scripts.finance_sim.calibrate import (  # noqa: E402
    CalibrationDraw,
    DrawSettings,
    EdgarRelease,
    EvidenceSettings,
    FitSettings,
    run_calibration,
)
from scripts.finance_sim.convergence import convergence_table, write_table  # noqa: E402
from scripts.finance_sim.cutoff_check import (  # noqa: E402
    RULE_PROMPT,
    cutoff_check_from_records,
    cutoff_check_path,
    write_cutoff_check,
)
from scripts.finance_sim.ensemble_metrics import (  # noqa: E402
    FIRST_DRAW_RATE,
    HALF_SUMMARY,
    RANDOM_ANSWERER_NULL,
    ensemble_metrics,
    write_metrics,
)
from scripts.finance_sim.execute import (  # noqa: E402
    ExecutionDraw,
    derive_result,
    execute_items,
    load_calibrators,
    load_result,
    macro_items,
    planned_keys,
    result_path,
    write_result,
)
from scripts.finance_sim.figures import DERIVED_ONLY, FIGURE_NUMBERS, PUBLISHED_FIGURES, figure_paths  # noqa: E402
from scripts.finance_sim.figures_catalogue import (  # noqa: E402
    CATALOGUE_FIGURES,
    MODE_COMPARISON,
    REPEAT_BASELINE,
    catalogue_figure_paths,
)
from scripts.finance_sim.figures_catalogue import FIGURE_NUMBERS as CATALOGUE_NUMBERS  # noqa: E402
from scripts.finance_sim.figures_draws import EXAMPLE_DEFAULT  # noqa: E402
from scripts.finance_sim.items import (  # noqa: E402
    CASH_SERIES,
    ITEMS_SCHEMA,
    PRICE_SYMBOLS,
    EdgarItems,
    MacroItem,
    RealizedOutcome,
    load_edgar_items,
    row_to_dict,
)
from scripts.finance_sim.limits import ARMS, BudgetLedger, LedgerCallBudget, StudyLimits, budget_path  # noqa: E402
from scripts.finance_sim.mode_comparison import (  # noqa: E402
    comparison_path,
    mode_comparison_from_records,
    write_comparison,
)
from scripts.finance_sim.negative_control import (  # noqa: E402
    negative_control_change,
    run_negative_control,
    write_report,
)
from scripts.finance_sim.plan import (  # noqa: E402
    ARM_REGISTRY_IDS,
    EDGAR_FILES,
    FULL_DISTRIBUTION,  # noqa: E402
    EdgarRow,
    MonthClass,
    PilotSettings,
    StudyArms,
    StudyPlan,
    freeze_plan,
    load_serving_manifest,
    sha256_of,
    split_months,
    write_plan,
)
from scripts.finance_sim.published_text import check_published_text  # noqa: E402
from scripts.finance_sim.recording import (  # noqa: E402
    EVIDENCE_SCHEMA,
    DrawLog,
    draw_log_path,
    evidence_path,
    pinned_digest,
)
from scripts.finance_sim.repeat_baseline import (  # noqa: E402
    baseline_path,
    repeat_baseline_from_records,
    write_baseline,
)
from scripts.finance_sim.replay import (  # noqa: E402
    CLOSES_LABEL,
    DTB3_LABEL,
    EXAMPLE_RULE,
    MANIFEST_NAME,
    MATPLOTLIB_VERSION,
    PRIVATE_DIR,
    PUBLISHED_DIR,
    REPLY_LABEL,
    SVG_CREATOR,
    WARNING_LABELS,
    Probe,
    ReplayError,
    ReplayInputs,
    ReplayOutput,
    check_guard_inputs,
    check_svg_version,
    compare_published,
    dtb3_probes,
    find_leaks,
    fmp_probes,
    items_probes,
    main,
    month_end_closes,
    number_tokens,
    partition_leaks,
    replay,
    replay_probes,
    reply_probes,
    require_matplotlib,
    series_probe,
    svg_version,
    text_probe,
    value_probe,
    warning_records,
)
from tests.scripts.finance_sim.test_convergence import (  # noqa: E402
    ALLOWANCE,
    DIGEST,
    KEEP,
    SEED_BASE,
    TEMPERATURE,
    Clock,
    FakeFull,
    canonical_sha256,
    digest,
    small_arm,
)
from tests.scripts.finance_sim.test_mode_comparison import StampedHost  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
GEMMA = "google/gemma-3-4b-it"
CREATED_AT = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)
SEED = 20261110
#: Negative-control releases per filing quarter, as uneven as the frozen corpus (40 releases in 9 quarters).
QUARTER_SIZES = {"2024Q3": 1, "2024Q4": 5, "2025Q1": 7, "2025Q2": 4, "2025Q3": 2, "2025Q4": 6, "2026Q1": 8,
                 "2026Q2": 2, "2026Q3": 5}
SYMBOLS = ("world_equity", "tech_sector", "gold_commodity")
DEFENSIVE = {"inflation": 0.5, "growth": -0.5, "credit_stress": 0.5, "policy": 0.0, "risk_appetite": -0.5}
RISK = {axis: -value for axis, value in DEFENSIVE.items()}
#: The first generation call of the full run is answered with prose, so one item's first draw is unparsed.
PROSE_CALL = 0


def month_range(first: str, last: str) -> list[str]:
    year, month = int(first[:4]), int(first[5:])
    out = []
    while f"{year:04d}-{month:02d}" <= last:
        out.append(f"{year:04d}-{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return out


MONTHS = month_range("2016-01", "2026-06")


# ---------- the fixture's licensed series: distinctive raw levels, closes and cash rates ----------

def macro_row(month: str) -> MacroRow:
    """A macro row whose raw levels are long, distinctive floats (the guard probes them one by one)."""
    year, number = int(month[:4]), int(month[5:])
    k = MONTHS.index(month)
    return MacroRow(decision_date=date(year, number, calendar.monthrange(year, number)[1]),
                    raw={"baa10y": 1.93 + 0.0137 * k + 1.1e-9 * ((k * 5) % 11),
                         "cpi_yoy": 1.5 + 0.0213 * k + 1.3e-9 * ((k * 7) % 13),
                         "t10y2y": 0.5 - 0.0071 * k + 1.7e-9 * ((k * 3) % 7)},
                    zscores={"baa10y_z": 0.1 * (k % 7) - 0.3, "cpi_yoy_z": 0.05 * (k % 11) - 0.2,
                             "t10y2y_z": -0.04 * (k % 5) + 0.1},
                    revised={"BAA10Y": False, "CPIAUCSL": False, "T10Y2Y": False})


def dtb3(month: str) -> float:
    """The month-end DTB3 rate in percent, two decimals like the real series (so only runs are probed)."""
    k = MONTHS.index(month)
    return round(4.21 + 0.03 * ((k * 7) % 23), 2)


def closes(month: str) -> dict[str, float]:
    """FMP-style month-end adjusted closes per risky category: six decimals (distinctive values) for two of them,
    two decimals for the gold ETF as FMP quotes it (covered by runs of consecutive closes only)."""
    k = MONTHS.index(month)
    return {"world_equity": round(100.0 * (1.0113 ** k) * (1 + 1e-7 * ((k * 13) % 17)), 6),
            "tech_sector": round(250.0 * (1.0113 ** k) * (1 + 1e-7 * ((k * 13 + 5) % 17)), 6),
            "gold_commodity": round(60.0 * (1.0071 ** k), 2)}


PRICE_WINDOW = ("2016-01-01", "2026-06-30")


def write_price_caches(root: Path) -> tuple[Path, Path]:
    """The private FMP bodies (one per symbol, the assembly's file names) and the cached DTB3 body."""
    fmp_cache, fred_cache = root / "data" / "fmp", root / "fred"
    fmp_cache.mkdir(parents=True)
    fred_cache.mkdir(parents=True)
    for category, symbol in PRICE_SYMBOLS.items():
        rows = [{"symbol": symbol, "date": macro_row(month).decision_date.isoformat(),
                 "adjClose": closes(month)[category]} for month in MONTHS]
        (fmp_cache / f"{symbol}_{PRICE_WINDOW[0]}_{PRICE_WINDOW[1]}.json").write_text(json.dumps(rows),
                                                                                   encoding="utf-8")
    observations = [{"date": macro_row(month).decision_date.isoformat(), "value": f"{dtb3(month):.2f}"}
                    for month in MONTHS]
    (fred_cache / f"{CASH_SERIES}_current_{'0123456789abcdef'}.json").write_text(
        json.dumps({"observations": observations, "count": len(observations)}), encoding="utf-8")
    return fmp_cache, fred_cache


def realized(month: str) -> RealizedReturns | None:
    """Next-month returns from the closes and the DTB3 cash leg; None for the last month (no next month)."""
    k = MONTHS.index(month)
    if k + 1 >= len(MONTHS):
        return None
    now, following = closes(month), closes(MONTHS[k + 1])
    returns = {symbol: following[symbol] / now[symbol] - 1.0 for symbol in SYMBOLS}
    return RealizedReturns(**returns, short_treasury_cash=dtb3(month) / 100.0 / 12.0)


def macro_item(month: str) -> MacroItem:
    row = macro_row(month)
    outcome = realized(month)
    if outcome is None:
        return MacroItem(f"macro-{month}", row, {}, RealizedOutcome(row.decision_date, None, "no next month", {}))
    return MacroItem(f"macro-{month}", row, {}, RealizedOutcome(row.decision_date, outcome, None, {}))


MACRO_ITEMS = [macro_item(month) for month in MONTHS]


def item_hash(item: MacroItem) -> str:
    prompts = {"anonymised": digest(render_anonymised(item.row)), "identifying": digest(render_identifying(item.row))}
    return canonical_sha256({"prompt_sha256": prompts, "row_sha256": canonical_sha256(row_to_dict(item.row))})


# ---------- the EDGAR corpus on disk, read by the real loader ----------

def corpus(root: Path) -> Path:
    """The four corpus files and their manifest: 40 + 100 calibration rows, 40 unused, 40 negative-control rows."""
    counter = itertools.count(1)

    def row(label: int, text: str, published: date) -> dict:
        n = next(counter)
        return {"prompt": text, "label": label,
                "metadata": {"published_at": published.isoformat(),
                             "accession": f"{n:010d}-{published.year % 100:02d}-{n:06d}", "cik": str(1000 + n),
                             "company": f"Company {n}"}}

    def text(n: int, recalled: bool) -> str:
        marker = " (recalled)" if recalled else ""
        return f"Company {n} reports quarterly results{marker}. Revenue rose and the outlook is unchanged."

    files: dict[str, tuple[int, list[dict]]] = {
        "calibration/is_memorized.jsonl": (1, [row(1, text(i, True), date(2023, 1, 1) + timedelta(days=i))
                                              for i in range(40)]),
        "calibration/oos_control.jsonl": (0, [row(0, text(100 + i, False), date(2023, 3, 1) + timedelta(days=i))
                                             for i in range(100)]),
        "held_out/is_memorized.jsonl": (1, [row(1, text(300 + i, True), date(2023, 9, 1) + timedelta(days=i))
                                           for i in range(40)]),
    }
    held_out = []
    for quarter, size in QUARTER_SIZES.items():
        year, number = int(quarter[:4]), int(quarter[5])
        for k in range(size):
            published = date(year, 3 * (number - 1) + 1 + k % 3, 1 + k)
            held_out.append(row(0, text(500 + len(held_out), False), published))
    files["held_out/oos_control.jsonl"] = (0, held_out)
    manifest: dict = {"files": {}}
    for name, (label, rows) in files.items():
        data = "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in rows).encode("utf-8")
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        manifest["files"][name] = {"label": label, "sha256": hashlib.sha256(data).hexdigest(), "rows": len(rows)}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return root


def frozen_plan(edgar: EdgarItems) -> StudyPlan:
    months = [MonthClass(month, {GEMMA: month <= "2024-08"}) for month in MONTHS]
    rows = [EdgarRow(item.item_id, item.label, "calibration") for item in edgar.calibration]
    rows += [EdgarRow(item.item_id, item.label, "held_out") for item in edgar.negative_control]
    items = {item.item_id: item_hash(item) for item in MACRO_ITEMS}
    items |= {item.item_id: sha256_of({"prompt_sha256": item.prompt_sha256, "row_sha256": item.row_sha256})
              for item in (*edgar.calibration, *edgar.negative_control)}
    arms = StudyArms(registry_ids=ARM_REGISTRY_IDS, arms={arm: small_arm(arm) for arm in ("sampling", "serving")})
    pilot = replace(PilotSettings.for_split(split_months(months, seed=SEED), arms), draws=2, min_parsed=2)
    return freeze_plan(months, rows, seed=SEED, created_at=CREATED_AT, items=items,
                       edgar_files={name: edgar.files[name]["sha256"] for name in EDGAR_FILES}, arms=arms,
                       serving_manifest=load_serving_manifest(), pilot=pilot)


# ---------- synthetic serving-arm results (temperature 0), derived without a request ----------

def serving_identity(plan: StudyPlan, item) -> GuardIdentity:
    return GuardIdentity(model=GEMMA, protocol="openai_compatible", endpoint="https://example.invalid/v1",
                         response_mapping="generic", source=SourceLabel(EstimateSource.TOP_K, 5),
                         settings_json='{"logprobs":true,"max_tokens":256,"temperature":0.0,"top_logprobs":5}',
                         temperature=0.0, draws=plan.arms.arms["serving"].draws, calibrator_fingerprint="c" * 64,
                         calibrator_is_weak=False, prompt_sha256=item.prompt_sha256)


def serving_results(plan: StudyPlan) -> list:
    """Every serving-arm full-run result: correct by a fixed pattern, identical draws, known estimates."""
    order = {item_id: number for number, item_id in enumerate(dict.fromkeys(
        item_id for item_id, _ in planned_keys(plan, "serving", "full_run")))}
    results = []
    for item in macro_items(plan, MACRO_ITEMS, stage="full_run", arm="serving"):
        number, identifying = order[item.item_id], item.rendering == "identifying"
        correct = number % 3 != 0 if identifying else number % 2 == 0
        estimate = 0.2 + 0.1 * (number % 4) if identifying else 0.05 * (number % 3)
        if item.realized is None:
            text = json.dumps(DEFENSIVE)
        else:
            defensive_wins = plan.scored_probability.outcome_of(item.realized) == 1
            text = json.dumps(DEFENSIVE if correct == defensive_wins else RISK)
        draws = [RecordedDraw(draw_index=index, temperature=0.0, seed=None, response_sha256=digest(text),
                              parsed=json.loads(text), failure_code=None, estimate=estimate, evidence_ref="e" * 64)
                 for index in range(plan.arms.arms["serving"].draws)]
        results.append(derive_result(item, draws, None, plan=plan, arm="serving", stage="full_run",
                                     identities={"top_k": serving_identity(plan, item)}))
    return results


# ---------- the probe arm through the real calibration and execution, recorded like the runner does ----------

def write_evidence_file(plan: StudyPlan, records_dir: Path, stage: str, item, found: Mapping) -> None:
    """One evidence batch in the 7.4 recording's evidence-file format."""
    indices = sorted(found)
    path = evidence_path("sampling", stage, item.item_id, item.rendering, records_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema": EVIDENCE_SCHEMA, "plan_sha256": plan.plan_sha256, "arm": "sampling", "stage": stage,
        "item_id": item.item_id, "rendering": item.rendering, "prompt_sha256": item.prompt_sha256,
        "batch_sha256": canonical_sha256([item.prompt_sha256, indices]), "model_digest": DIGEST, "top_k_keep": KEEP,
        "draw_indices": indices, "evidence": [found[index].to_dict() for index in indices],
        "recorded_at": "2026-11-10T12:00:00Z"}, sort_keys=True), encoding="utf-8")


def stage_records(plan: StudyPlan, records_dir: Path, stage: str) -> SimpleNamespace:
    """The stage's ledger within the frozen limits, its draw log and an evidence writer, plus what they saw."""
    budget = plan.limits.stage("sampling", stage)
    ledger = BudgetLedger.open(
        budget_path("sampling", stage, records_dir), plan,
        StudyLimits(plan.plan_sha256, "sampling", stage, budget.max_requests, budget.max_output_tokens,
                    budget.max_seconds), ALLOWANCE, clock=Clock())
    log = DrawLog(draw_log_path("sampling", stage, records_dir), plan=plan, loads=ExecutionDraw.from_dict)
    evidence: list = []

    def on_evidence(item, found) -> None:
        evidence.extend(found.values())
        write_evidence_file(plan, records_dir, stage, item, found)

    return SimpleNamespace(ledger=ledger, log=log, evidence=evidence, on_evidence=on_evidence)


def draw_log_state(records_dir: Path) -> dict[Path, tuple[bytes, int]]:
    """Bytes and modification time of every draw log under ``records_dir``, by its path relative to it."""
    return {path.relative_to(records_dir): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in sorted(records_dir.rglob("draws.jsonl"))}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("the replay attempted a network connection")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)


@pytest.fixture(scope="module")
def study(tmp_path_factory) -> SimpleNamespace:
    """The recorded study on fakes and the stored outputs of tasks 7.9 to 7.12, computed here, under tmp_path."""
    root = tmp_path_factory.mktemp("replay")
    edgar_dir = corpus(root / "edgar")
    edgar = load_edgar_items(edgar_dir)
    plan = frozen_plan(edgar)
    assert pinned_digest(plan, "sampling") == DIGEST
    records_dir, results_dir, calibrators_dir = root / "records", root / "results", root / "calibrators"
    ledger = BudgetLedger.open(root / "calibration" / "budget.json", plan,
                               StudyLimits(plan.plan_sha256, "sampling", "calibration", 1000, 1_000_000, 1_000_000),
                               ALLOWANCE, clock=Clock())
    # The generation host declares the weights it serves (the arm's pinned digest), as the recorded study's host
    # does, so both calibrators carry one weights identity and the two evidence modes can be compared; the
    # calibration draws are recorded like the runner records them.
    calibration_log = DrawLog(draw_log_path("sampling", "calibration", records_dir), plan=plan,
                              loads=CalibrationDraw.from_dict)
    run_calibration(plan, arm="sampling", macro_rows=[macro_row(month) for month in plan.months.calibration],
                    edgar_releases=[EdgarRelease(item.item_id, item.label, item.release_text)
                                    for item in edgar.calibration],
                    client=StampedHost(LedgerCallBudget(ledger)), ledger=ledger,
                    settings=DrawSettings(draws=2, temperature=TEMPERATURE, seed_base=SEED_BASE),
                    sources=("top_k", "full_distribution"), full_distribution=FakeFull(),
                    fit=FitSettings(min_auc=0.6, min_valid=10), evidence=EvidenceSettings(top_k_keep=KEEP),
                    on_draw=calibration_log, calibrators_dir=calibrators_dir, sleep=lambda seconds: None)
    macro = load_calibrators(plan, arm="sampling", kind="macro", calibrators_dir=calibrators_dir)
    edgar_calibrators = load_calibrators(plan, arm="sampling", kind="edgar", calibrators_dir=calibrators_dir)
    full_records = stage_records(plan, records_dir, "full_run")
    full = execute_items(plan, macro_items(plan, MACRO_ITEMS, stage="full_run", arm="sampling"), arm="sampling",
                         stage="full_run", client=StampedHost(LedgerCallBudget(full_records.ledger), prose=(PROSE_CALL,)),
                         ledger=full_records.ledger, calibrators=macro, full_distribution=FakeFull(),
                         on_draw=full_records.log, on_evidence=full_records.on_evidence, results_dir=results_dir,
                         sleep=lambda seconds: None)
    assert full.status == "complete"
    negative_records = stage_records(plan, records_dir, "negative_control")
    negative = run_negative_control(plan, edgar.negative_control, arm="sampling",
                                    client=StampedHost(LedgerCallBudget(negative_records.ledger)),
                                    ledger=negative_records.ledger, calibrators=edgar_calibrators,
                                    full_distribution=FakeFull(), on_draw=negative_records.log,
                                    on_evidence=negative_records.on_evidence, results_dir=results_dir,
                                    sleep=lambda seconds: None)
    assert negative.report.complete
    serving = serving_results(plan)
    for result in serving:
        write_result(result, results_dir)
    expected = root / "expected"
    write_analysis(pre_registered_analysis(plan, full.results, arm="sampling"), expected)
    write_analysis(pre_registered_analysis(plan, serving, arm="serving"), expected)
    write_metrics(ensemble_metrics(plan, full.results.values(), arm="sampling"), expected)
    write_metrics(ensemble_metrics(plan, negative.execution.results.values(), arm="sampling",
                                   stage="negative_control"), expected)
    write_metrics(ensemble_metrics(plan, serving, arm="serving"), expected)
    write_metrics(ensemble_metrics(plan, [], arm="serving", stage="negative_control"), expected)
    write_report(negative.report, expected)
    write_report(negative_control_change(plan, [], edgar.negative_control, arm="serving"), expected)
    full_items = macro_items(plan, MACRO_ITEMS, stage="full_run", arm="sampling")
    write_table(convergence_table(plan, tuple(full.results[item.key] for item in full_items),
                                  tuple(full_records.log.latest()), full_records.evidence, arm="sampling",
                                  stage="full_run", calibrators=macro, expected_digest=DIGEST), expected)
    write_table(convergence_table(plan, tuple(negative.execution.results.values()),
                                  tuple(negative_records.log.latest()), negative_records.evidence, arm="sampling",
                                  stage="negative_control", calibrators=edgar_calibrators, expected_digest=DIGEST),
                expected)
    # The three analyses behind the catalogue, computed here the way the replay must: the cutoff check of each arm
    # with stored full-run results, the mode comparison of the arm that has both calibrators with the cutoff result
    # for logits under the identifying prompt passed in, and the repeat baseline over both arms.
    checks = {arm: cutoff_check_from_records(plan, arm=arm, results_dir=results_dir) for arm in ARMS}
    for check in checks.values():
        write_cutoff_check(check, expected)
    logits_cutoff = checks["sampling"].cell(FULL_DISTRIBUTION, RULE_PROMPT).summary.passes
    write_comparison(mode_comparison_from_records(
        plan, arm="sampling", calibrators_dir=calibrators_dir, results_dir=results_dir, records_dir=records_dir,
        analysis_dir=expected, cutoff_check=logits_cutoff), expected)
    write_baseline(repeat_baseline_from_records(plan, records_dir=records_dir), expected)
    draw_logs = draw_log_state(records_dir)
    assert len(draw_logs) == 3, "the probe arm's calibration, full run and negative control are recorded"
    items_path = root / "items.json"
    items_path.write_text(json.dumps({
        "schema": ITEMS_SCHEMA,
        "macro_items": [{"item_id": item.item_id, "kind": "macro", "row": row_to_dict(item.row),
                         "realized": item.outcome.to_dict()["realized"]} for item in MACRO_ITEMS]},
        indent=2, sort_keys=True), encoding="utf-8")
    plan_path = write_plan(plan, root / "plan.json")
    fmp_cache, fred_cache = write_price_caches(root)
    return SimpleNamespace(root=root, plan=plan, plan_path=plan_path, edgar=edgar, edgar_dir=edgar_dir,
                           records_dir=records_dir, results_dir=results_dir, calibrators_dir=calibrators_dir,
                           expected=expected, items_path=items_path, fmp_cache=fmp_cache, fred_cache=fred_cache,
                           full=full, negative=negative, draw_logs=draw_logs, logits_cutoff=logits_cutoff)


def inputs_of(study: SimpleNamespace, **overrides) -> ReplayInputs:
    values = {"plan": study.plan, "results_dir": study.results_dir, "records_dir": study.records_dir,
              "calibrators_dir": study.calibrators_dir, "edgar_dir": study.edgar_dir,
              "items_path": study.items_path, "fmp_cache": study.fmp_cache,
              "fred_cache": study.fred_cache} | overrides
    return ReplayInputs(**values)


def cli_args(study: SimpleNamespace, out_root: Path) -> list[str]:
    return ["--plan", str(study.plan_path), "--plan-sha256", study.plan.plan_sha256, "--out-root", str(out_root),
            "--results-dir", str(study.results_dir), "--records-dir", str(study.records_dir),
            "--calibrators-dir", str(study.calibrators_dir), "--edgar-dir", str(study.edgar_dir),
            "--items", str(study.items_path), "--fmp-cache", str(study.fmp_cache),
            "--fred-cache", str(study.fred_cache)]


#: The replay command in a fresh process with every socket connection forbidden: a request would fail it.
RUN_REPLAY = (
    "import socket, sys\n"
    "def forbidden(*args, **kwargs):\n"
    "    raise AssertionError('the replay attempted a network connection')\n"
    "socket.create_connection = forbidden\n"
    "socket.socket.connect = forbidden\n"
    "from scripts.finance_sim.replay import main\n"
    "raise SystemExit(main(sys.argv[1:]))\n")


def run_replay(study: SimpleNamespace, out_root: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", RUN_REPLAY, *cli_args(study, out_root), *extra],
                          capture_output=True, text=True, cwd=ROOT)


def file_hashes(root: Path) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*")) if path.is_file()}


@pytest.fixture(scope="module")
def replayed(study) -> SimpleNamespace:
    """The replay run once in a fresh process, with the network off."""
    out_root = study.root / "out-a"
    result = run_replay(study, out_root)
    assert result.returncode == 0, result.stderr
    manifest = json.loads((out_root / MANIFEST_NAME).read_text(encoding="utf-8"))
    return SimpleNamespace(root=out_root, private=out_root / PRIVATE_DIR, published=out_root / PUBLISHED_DIR,
                           manifest=manifest, stdout=result.stdout, hashes=file_hashes(out_root))


def csv_rows(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))))


def stored_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------- one command, offline, into a fresh root ----------

def test_the_replay_runs_offline_in_a_fresh_process_and_writes_the_private_and_published_outputs(study, replayed):
    """Every analysis, the negative-control report, the convergence tables and the eight figures, no request."""
    private = sorted(path.name for path in replayed.private.iterdir())
    expected_private = sorted([
        *(f"pre-registered-{arm}-full_run.{suffix}" for arm in ("serving", "sampling") for suffix in ("json", "md")),
        *(f"ensemble-{arm}-{stage}.{suffix}" for arm in ("serving", "sampling")
          for stage in ("full_run", "negative_control") for suffix in ("json", "md")),
        *(f"negative-control-{arm}.{suffix}" for arm in ("serving", "sampling") for suffix in ("json", "md")),
        *(f"convergence-sampling-{stage}.{suffix}" for stage in ("full_run", "negative_control")
          for suffix in ("json", "md")),
        # The three analyses behind the catalogue, named by their own modules: the cutoff check of each arm with
        # stored full-run results, the mode comparison of the arm with both calibrators, one repeat baseline.
        *(path.with_suffix(suffix).name for arm in ARMS for path in [cutoff_check_path(arm)]
          for suffix in (".json", ".md")),
        *(comparison_path("sampling").with_suffix(suffix).name for suffix in (".json", ".md")),
        *(baseline_path().with_suffix(suffix).name for suffix in (".json", ".md")),
    ])
    assert private == expected_private
    published = {path.name for path in replayed.published.iterdir()}
    for name in PUBLISHED_FIGURES:
        assert {path.name for path in figure_paths(name, replayed.published).values()} <= published
        assert {path.name for path in figure_paths(name, replayed.published).values()} == {
            f"{name}.svg", f"{name}.csv", f"{name}.md"}
    for name in CATALOGUE_FIGURES:
        assert {path.name for path in catalogue_figure_paths(name, replayed.published).values()} == {
            f"{name}.svg", f"{name}.csv", f"{name}.md"} <= published
    # One SVG per figure in the one academic style: no light or dark variant, and a numbered caption.
    figures = (*PUBLISHED_FIGURES, *CATALOGUE_FIGURES)
    assert {name for name in published if name.endswith(".svg")} == {f"{name}.svg" for name in figures}
    assert not any("-light" in name or "-dark" in name for name in published)
    assert PUBLISHED_FIGURES == ("gap", "reliability", "draws", "convergence", "draw-spread")
    assert dict(FIGURE_NUMBERS) == {"gap": 1, "reliability": 2, "draws": 3, "convergence": 4, "draw-spread": 8}
    assert sorted({**FIGURE_NUMBERS, **CATALOGUE_NUMBERS}.values()) == list(range(1, len(figures) + 1))
    for name in figures:
        text = (replayed.published / f"{name}.md").read_text(encoding="utf-8")
        number = {**FIGURE_NUMBERS, **CATALOGUE_NUMBERS}[name]
        assert text.startswith(f"Figure {number}. ") and text.count(DERIVED_ONLY) == 1, name
    # The published set is the eight figures, each with its SVG, table and caption file, and nothing else: no
    # analysis markdown is copied beside them (owner's decision, 2026-11-11). Figure 7's table keeps its own name.
    assert published == {f"{name}{suffix}" for name in figures for suffix in (".svg", ".csv", ".md")}
    assert len(figures) == 8 and f"{REPEAT_BASELINE}.md" in published
    private_markdown = {name for name in expected_private if name.endswith(".md")}
    assert private_markdown and not (private_markdown - {f"{REPEAT_BASELINE}.md"}) & published
    assert (replayed.published / f"{REPEAT_BASELINE}.md").read_text(encoding="utf-8").startswith("Figure 7. ")
    assert "published_wording" not in replayed.manifest
    catalogue = replayed.manifest["catalogue"]
    assert catalogue["cutoff_check"] == list(ARMS) and catalogue["mode_comparison"] == ["sampling"]
    assert catalogue["figures"] == list(CATALOGUE_FIGURES) and catalogue["figure_arm"] == "sampling"
    # An arm without the logits mode gets no mode comparison; the manifest says so once, and it is not an error.
    assert [(entry["analysis"], entry["arm"]) for entry in catalogue["not_written"]] == [
        ("mode-comparison", "serving")]
    assert "logits" in catalogue["not_written"][0]["reason"]
    assert not any(name.endswith(".json") for name in published), "the JSON documents with per-item rows stay private"
    assert replayed.manifest["plan_sha256"] == study.plan.plan_sha256
    assert replayed.manifest["probe_arm"] == "sampling" and replayed.manifest["arms"] == ["serving", "sampling"]
    assert set(replayed.manifest["published"]) == {f"{PUBLISHED_DIR.as_posix()}/{name}" for name in published}
    assert set(replayed.manifest["private"]) == {f"{PRIVATE_DIR.as_posix()}/{name}" for name in private}
    for relative, sha256 in {**replayed.manifest["private"], **replayed.manifest["published"]}.items():
        assert replayed.hashes[relative] == sha256
    assert str(replayed.root / MANIFEST_NAME) in replayed.stdout


def test_the_replay_reproduces_the_stored_analyses_and_the_negative_control_byte_for_byte(study, replayed):
    """The 7.10 to 7.12 outputs and the 7.9 report computed in this process, against the replay's, byte for byte."""
    expected = sorted(path.name for path in study.expected.iterdir())
    assert expected == sorted(path.name for path in replayed.private.iterdir())
    for name in expected:
        assert (replayed.private / name).read_bytes() == (study.expected / name).read_bytes(), name


def test_the_named_statistics_are_present_in_the_replayed_outputs(study, replayed):
    """Holm-adjusted p-values, achieved power, the random-answerer null, the first-draw rate, the half-ensemble
    result, the negative-control outcomes and the convergence table are computed, not all missing."""
    serving = stored_json(replayed.private / "pre-registered-serving-full_run.json")
    did = next(entry for entry in serving["metrics"] if entry["metric"] == DID)
    holm = {cell["variant"]: cell["p_holm"] for cell in did["variants"] if cell["p_holm"] is not None}
    assert set(holm) >= {"single_none", "single_top_k", "ensemble_none", "ensemble_top_k"}
    assert all(0.0 <= value <= 1.0 for value in holm.values())
    power = next(entry for entry in serving["metrics"] if entry["metric"] == "achieved_power")
    assert power["power"] is not None and isinstance(power["power"].get("power"), float)
    sampled = stored_json(replayed.private / "ensemble-sampling-full_run.json")["metrics"]
    assert isinstance(sampled[RANDOM_ANSWERER_NULL]["test"]["p_value"], float)
    assert isinstance(sampled[FIRST_DRAW_RATE]["overall"]["rate"], float)
    halves = sampled[HALF_SUMMARY]["results"]
    computed = {name: cell for name, cell in halves.items() if cell["test"] is not None}
    assert computed, halves
    assert all(isinstance(cell["test"]["statistic"], float) for cell in computed.values())
    assert all(cell["holm_p_value"] is not None for cell in computed.values()), "Holm across the computed cells"
    report = stored_json(replayed.private / "negative-control-sampling.json")
    assert report["complete"] is True
    assert {entry["outcome"] for entry in report["variants"]} <= {"no material change", "inconclusive",
                                                                  "discounting honest answers"}
    assert all(entry["outcome"] is not None for entry in report["variants"])
    table = stored_json(replayed.private / "convergence-sampling-full_run.json")
    assert any(row["mean_abs_diff"] is not None for row in table["features"])


def test_two_fresh_processes_produce_identical_bytes_for_every_figure_table_and_document(study, replayed):
    out_root = study.root / "out-b"
    result = run_replay(study, out_root)
    assert result.returncode == 0, result.stderr
    assert file_hashes(out_root) == replayed.hashes


#: The reference render: the figure modules called directly, in a fresh process, from the stored outputs.
RENDER_REFERENCE = (
    "import sys\n"
    "from scripts.finance_sim.figures import render_figures\n"
    "from scripts.finance_sim.figures_convergence import render_convergence\n"
    "from scripts.finance_sim.figures_draws import render_draws\n"
    "from scripts.finance_sim.plan import load_plan\n"
    "plan_path, sha256, analysis_dir, calibrators_dir, results_dir, out_dir = sys.argv[1:7]\n"
    "plan = load_plan(plan_path, expected_sha256=sha256)\n"
    "render_figures(plan, analysis_dir=analysis_dir, calibrators_dir=calibrators_dir, out_dir=out_dir)\n"
    "render_draws(plan, analysis_dir=analysis_dir, results_dir=results_dir, out_dir=out_dir)\n"
    "render_convergence(plan, analysis_dir=analysis_dir, out_dir=out_dir)\n"
    "print('rendered')\n")


def test_every_figure_matches_a_reference_rendered_in_a_fresh_process_from_the_stored_outputs(study, replayed,
                                                                                                tmp_path):
    out_dir = tmp_path / "reference"
    result = subprocess.run([sys.executable, "-c", RENDER_REFERENCE, str(study.plan_path), study.plan.plan_sha256,
                             str(study.expected), str(study.calibrators_dir), str(study.results_dir), str(out_dir)],
                            capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stderr
    for name in PUBLISHED_FIGURES:
        for path in figure_paths(name, out_dir).values():
            assert (replayed.published / path.name).read_bytes() == path.read_bytes(), path.name


def test_a_committed_copy_is_compared_byte_for_byte(study, replayed, tmp_path):
    output = ReplayOutput(root=replayed.root, manifest=replayed.manifest, private=(),
                          published=tuple(sorted(replayed.published.iterdir())))
    assert compare_published(output, replayed.published) == []
    copy = tmp_path / "committed"
    shutil.copytree(replayed.published, copy)
    (copy / "gap.csv").write_text((copy / "gap.csv").read_text(encoding="utf-8") + "extra\n", encoding="utf-8")
    (copy / "draws.svg").unlink()
    assert compare_published(output, copy) == ["draws.svg", "gap.csv"]
    (copy / "old-figure.svg").write_text("<svg/>", encoding="utf-8")
    assert compare_published(output, copy) == ["draws.svg", "gap.csv", "old-figure.svg (stale)"]
    out_root = tmp_path / "compared"
    assert main([*cli_args(study, out_root), "--compare", str(copy), "--require-guard"]) == 1
    manifest = json.loads((out_root / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["guard"]["required"] is True
    hashes = file_hashes(out_root)
    assert {name: value for name, value in hashes.items() if name != MANIFEST_NAME} == {
        name: value for name, value in replayed.hashes.items() if name != MANIFEST_NAME}, \
        "the outputs are written and identical; only the compare differs and the manifest records the requirement"


def test_a_used_output_root_a_drive_root_and_another_plan_are_refused(study, replayed, tmp_path):
    with pytest.raises(ReplayError, match="new or empty"):
        replay(inputs_of(study), replayed.root)
    with pytest.raises(ReplayError, match="drive or filesystem root"):
        replay(inputs_of(study), Path(tmp_path.anchor))
    assert main([*cli_args(study, replayed.root)]) == 2
    args = cli_args(study, tmp_path / "other-plan")
    args[args.index("--plan-sha256") + 1] = "0" * 64
    assert main(args) == 2
    assert not (tmp_path / "other-plan").exists()


# ---------- the matplotlib pin ----------

def test_matplotlib_is_pinned_to_the_version_the_svg_metadata_carries_and_another_is_refused(replayed):
    assert require_matplotlib() == MATPLOTLIB_VERSION == matplotlib.__version__
    assert require_matplotlib(installed=MATPLOTLIB_VERSION) == MATPLOTLIB_VERSION
    with pytest.raises(ReplayError, match=r"matplotlib 3\.9\.4 is installed.*3\.10\.9"):
        require_matplotlib(installed="3.9.4")
    assert replayed.manifest["matplotlib"] == MATPLOTLIB_VERSION
    assert replayed.manifest["svg_creator"] == SVG_CREATOR
    svgs = sorted(replayed.published.glob("*.svg"))
    assert len(svgs) == len(PUBLISHED_FIGURES) + len(CATALOGUE_FIGURES)
    for path in svgs:
        text = path.read_text(encoding="utf-8")
        assert svg_version(text) == SVG_CREATOR == check_svg_version(text, path)
    doctored = svgs[0].read_text(encoding="utf-8").replace(SVG_CREATOR, "Matplotlib v3.9.4, https://matplotlib.org/")
    with pytest.raises(ReplayError, match=r"rendered by 'Matplotlib v3\.9\.4"):
        check_svg_version(doctored, svgs[0])
    with pytest.raises(ReplayError, match="no matplotlib creator metadata"):
        svg_version("<svg></svg>")


def test_the_lock_pins_the_same_matplotlib_version():
    """``uv run --no-sync`` renders with the locked version, which the pin names; a lock bump moves the pin."""
    lock = (ROOT / "uv.lock").read_text(encoding="utf-8")
    start = lock.index('name = "matplotlib"\n')
    version = lock[start:].split("\n")[1]
    assert version == f'version = "{MATPLOTLIB_VERSION}"'


# ---------- the frozen F3 example rule ----------

def test_the_example_rule_is_frozen_and_the_replay_draws_that_item(study, replayed):
    assert EXAMPLE_RULE == EXAMPLE_DEFAULT
    assert "first item in plan order with a decided first draw and a consensus summary value" in EXAMPLE_RULE
    assert replayed.manifest["example_rule"] == EXAMPLE_RULE
    chosen = None
    for item_id, rendering in planned_keys(study.plan, "sampling", "full_run"):
        result = load_result(result_path("sampling", "full_run", item_id, rendering, study.results_dir))
        consensus = result.variant("ensemble_none")
        if result.categories[0] is not None and consensus.category is not None and consensus.summary_value is not None:
            chosen = (item_id, rendering)
            break
    assert chosen is not None
    rows = [row for row in csv_rows(replayed.published / "draws.csv") if row["row"] == "draw"]
    assert rows and {(row["item_id"], row["rendering"]) for row in rows} == {chosen}


# ---------- published tables: derived statistics with the citations, and the licensing guard ----------

def published_text_files(published: Path) -> list[Path]:
    return sorted(path for path in published.iterdir() if path.suffix in (".md", ".csv", ".svg"))


def sequence_probes(label: str, values: list) -> list[Probe]:
    probes = [probe for value in values if (probe := value_probe(label, value)) is not None]
    probes += [probe for start in range(len(values) - 2) if (probe := series_probe(label, values[start:start + 3]))]
    return probes


def fixture_probes(study: SimpleNamespace) -> list[Probe]:
    """The fixture's raw BAA10Y, CPI and T10Y2Y levels, FMP closes, DTB3 rates, realized returns and replies."""
    probes: list[Probe] = []
    for series in ("baa10y", "cpi_yoy", "t10y2y"):
        probes += sequence_probes(f"{series} raw levels", [macro_row(month).raw[series] for month in MONTHS])
    for symbol in SYMBOLS:
        probes += sequence_probes("FMP adjusted closes", [closes(month)[symbol] for month in MONTHS])
    probes += sequence_probes("DTB3 rates", [dtb3(month) for month in MONTHS])
    for name in ("world_equity", "tech_sector", "gold_commodity", "short_treasury_cash"):
        values = [getattr(realized(month), name) for month in MONTHS if realized(month) is not None]
        probes += sequence_probes("realized returns", values)
    probes += reply_probes(study.plan, study.records_dir)
    return probes


def test_published_tables_carry_the_citations_and_only_derived_statistics(study, replayed):
    files = published_text_files(replayed.published)
    figures = (*PUBLISHED_FIGURES, *CATALOGUE_FIGURES)
    assert len(files) == 3 * len(figures)
    for path in files:
        assert path.stem in figures, path.name
        if path.suffix == ".md":
            text = path.read_text(encoding="utf-8")
            assert "Moody's" in text and "BAA10Y" in text and "Financial Modeling Prep (FMP)" in text, path.name
    probes = fixture_probes(study)
    kinds = {probe.kind for probe in probes}
    assert kinds == {"value", "series", "text"}
    assert sum(1 for probe in probes if probe.label == REPLY_LABEL) > 100
    assert find_leaks(files, probes) == []
    guard = replayed.manifest["guard"]
    assert guard["items_probed"] is True and guard["leaks"] == 0 and guard["files_scanned"] == len(files)
    assert guard["warnings"] == [] and guard["warning_labels"] == [DTB3_LABEL] == sorted(WARNING_LABELS)
    assert guard["closes_probed"] is True and guard["dtb3_probed"] is True and guard["required"] is False
    assert (guard["fmp_bodies"], guard["priced_symbols"], guard["dtb3_bodies"]) == (len(PRICE_SYMBOLS),
                                                                                  len(PRICE_SYMBOLS), 1)
    assert guard["probes"]["BAA10Y raw levels (FRED, Moody's)"] > len(MONTHS)
    # Six-decimal closes of two symbols as values (all but the first world close, 100.0) plus runs of three symbols.
    assert guard["probes"][CLOSES_LABEL] == 2 * len(MONTHS) - 1 + 3 * (len(MONTHS) - 2)
    assert guard["probes"][DTB3_LABEL] == len(MONTHS) - 2, "two-decimal rates are probed as runs only"
    assert guard["probes"][REPLY_LABEL] > 100
    own = replay_probes(inputs_of(study))
    assert (own.items_probed, own.closes_probed, own.dtb3_probed) == (True, True, True)
    assert own.counts() == guard["probes"]
    assert find_leaks(files, own.probes) == []
    assert not any(probe.texts[0] in json.dumps(replayed.manifest) for probe in own.probes if probe.kind == "value")
    header = csv_rows(replayed.published / "draws.csv")[0]
    assert not {"correct", "outcome", "realized", "response", "response_text"} & set(header)


#: The columns of the published figure tables that hold a reason, and the keys a published table does not carry.
REASON_COLUMNS = ("missing", "reason", "halves")
INTERNAL_COLUMNS = ("arm", "role")
STORED_WORDING = ("Requirement", "serving arm", "sampling arm", "min_parsed", "in both renderings", ".json",
                  "full-distribution host")


def test_published_figure_tables_give_reasons_routes_and_variants_in_the_published_wording(replayed):
    """Routed from the reviews of the figures: no stored reason, no arm or role column and no internal variant key
    in a published figure table, in the CSV or in its markdown; the private documents keep the stored text."""
    figures = (*PUBLISHED_FIGURES, *CATALOGUE_FIGURES)
    reasons = set()
    for name in figures:
        rows = csv_rows(replayed.published / f"{name}.csv")
        assert rows, name
        assert not set(INTERNAL_COLUMNS) & set(rows[0]), name
        for row in rows:
            for column in REASON_COLUMNS:
                if row.get(column):
                    assert check_published_text(row[column]) == row[column], (name, column)
                    reasons.add(row[column])
            for column in ("variant", "variants"):
                assert not re.search(r"\b(single|ensemble)_\w+", row.get(column, "")), (name, row[column])
        for suffix in (".csv", ".md"):
            text = (replayed.published / f"{name}{suffix}").read_text(encoding="utf-8")
            for stored in STORED_WORDING:
                assert stored not in text, (name, suffix, stored)
    assert "route" in csv_rows(replayed.published / "gap.csv")[0]
    assert {row["variant"] for row in csv_rows(replayed.published / "gap.csv")} == {
        "single draw, unadjusted", "single draw, adjusted with logprobs", "single draw, adjusted with logits",
        "ensemble, unadjusted", "ensemble, adjusted with logprobs", "ensemble, adjusted with logits"}
    # The fixture exercises the translation: a route without logits, calibrators not fitted, an unread first reply,
    # a k above the kept alternatives and a route without recorded draws.
    assert {"logits are not available on this serving route: they would come from other weights than the served "
            "model, and values from different weights are not comparable",
            "no logprobs calibrator has been fitted on macro prompts for this serving route",
            "the first reply could not be read", "k 100 is above the 20 alternatives kept",
            "no draw of the full run is recorded for this serving route"} <= reasons
    # The private analyses keep the stored reason, word for word.
    stored = stored_json(replayed.private / "pre-registered-serving-full_run.json")
    assert "(Requirement 9.4)" in json.dumps(stored["missing_variants"])


def test_the_mode_comparison_of_the_replay_received_the_cutoff_result_for_logits(study, replayed):
    """The three-answer decision needs the cutoff result (Requirement 4.6): the replay computes the cutoff check
    first and passes the logits result under the identifying prompt into the comparison."""
    check = stored_json(replayed.private / cutoff_check_path("sampling").name)
    cell = next(cell for cell in check["cells"]
                if (cell["mode"], cell["prompt_kind"]) == (FULL_DISTRIBUTION, RULE_PROMPT))
    assert cell["summary"]["passes"] is not None, "the fixture has a cutoff check for logits"
    comparison = stored_json(replayed.private / comparison_path("sampling").name)
    assert comparison["cutoff_check"] is cell["summary"]["passes"] is study.logits_cutoff
    assert "was not supplied" not in comparison["reason"]
    assert replayed.manifest["catalogue"]["mode_comparison_cutoff_input"] == {"sampling": cell["summary"]["passes"]}
    markdown = (replayed.private / comparison_path("sampling").with_suffix(".md").name).read_text(encoding="utf-8")
    assert "Cutoff check for logits: not supplied" not in markdown
    caption = (replayed.published / f"{MODE_COMPARISON}.md").read_text(encoding="utf-8").splitlines()[0]
    assert "no cutoff check for logits was supplied" not in caption


def test_an_arm_without_stored_results_gets_no_cutoff_check_and_the_manifest_says_why(study, tmp_path):
    """The shape of the recorded study: the serving route's full run is an open stage without stored item results.
    Its cutoff check is not written and not published, the manifest gives the reason, and the replay still
    publishes the rest."""
    results = tmp_path / "results"
    shutil.copytree(study.results_dir, results)
    serving = [path for path in results.rglob("*.json") if "serving" in path.relative_to(results).parts]
    assert serving
    for path in serving:
        path.unlink()
    output = replay(inputs_of(study, results_dir=results), tmp_path / "out")
    private = {path.name for path in output.private}
    assert cutoff_check_path("serving").name not in private and cutoff_check_path("sampling").name in private
    assert baseline_path().name in private
    catalogue = output.manifest["catalogue"]
    assert catalogue["cutoff_check"] == ["sampling"]
    missing = {(entry["analysis"], entry["arm"]): entry["reason"] for entry in catalogue["not_written"]}
    assert set(missing) == {("cutoff-check", "serving"), ("mode-comparison", "serving")}
    assert "no stored result" in missing[("cutoff-check", "serving")]
    assert not (output.root / PUBLISHED_DIR / cutoff_check_path("serving").with_suffix(".md").name).exists()


def test_a_leak_in_a_new_published_file_still_refuses_the_replay(study, replayed, tmp_path):
    """The new files go through the same guard: a raw BAA10Y level equal to a number of the mode comparison's
    published table refuses the replay, removes the published tree, writes no manifest and exits with code 2."""
    name = f"{MODE_COMPARISON}.csv"
    planted = next(float(cell) for row in csv_rows(replayed.published / name) for cell in row.values()
                   if len(cell) >= 8 and cell.replace("-", "").replace(".", "").isdigit())
    record = json.loads(study.items_path.read_text(encoding="utf-8"))
    for index in range(3):
        record["macro_items"][index]["row"]["raw"]["baa10y"] = planted
    items_path = tmp_path / "items-planted.json"
    items_path.write_text(json.dumps(record), encoding="utf-8")
    out_root = tmp_path / "planted"
    with pytest.raises(ReplayError, match="licensing probe") as refusal:
        replay(inputs_of(study, items_path=items_path), out_root)
    message = str(refusal.value)
    assert f"{name}: BAA10Y raw levels (FRED, Moody's) (" in message
    assert repr(planted) not in message
    assert not (out_root / "docs").exists() and (out_root / PRIVATE_DIR).is_dir()
    assert not (out_root / MANIFEST_NAME).exists()
    args = cli_args(study, tmp_path / "planted-cli")
    args[args.index("--items") + 1] = str(items_path)
    assert main(args) == 2
    assert not (tmp_path / "planted-cli" / "docs").exists() and not (tmp_path / "planted-cli" / MANIFEST_NAME).exists()
    # The guard scans every published file, the new ones among them.
    scanned = replayed.manifest["guard"]["files_scanned"]
    assert scanned == len(published_text_files(replayed.published)) == len(replayed.manifest["published"])


# ---------- private records are opened for reading only ----------

def test_the_replay_leaves_every_draw_log_as_it_is(study, replayed):
    assert draw_log_state(study.records_dir) == study.draw_logs


def test_the_replay_reads_draw_logs_only_through_the_read_only_reader():
    import inspect

    from scripts.finance_sim import replay as module

    source = inspect.getsource(module)
    assert not re.search(r"\bDrawLog\b", source) and ".latest(" not in source and "_repaired_lines" not in source
    assert "read_draw_log" in source


def test_a_draw_log_without_a_final_newline_is_read_and_left_unchanged(study, replayed, tmp_path):
    """The recording's own reader would append the newline; an analysis must not modify a private record."""
    records = tmp_path / "records"
    shutil.copytree(study.records_dir, records)
    log = draw_log_path("sampling", "full_run", records)
    data = log.read_bytes()
    assert data.endswith(b"\n")
    log.write_bytes(data[:-1])
    before = draw_log_state(records)
    output = replay(inputs_of(study, records_dir=records), tmp_path / "out")
    assert draw_log_state(records) == before
    hashes = file_hashes(output.root)
    assert {name: value for name, value in hashes.items() if name != MANIFEST_NAME} == {
        name: value for name, value in replayed.hashes.items() if name != MANIFEST_NAME}, \
        "a complete last line without its newline is used: the outputs are those of the intact log"


def test_a_draw_log_with_a_cut_short_last_line_refuses_the_replay_and_is_left_unchanged(study, tmp_path, capsys):
    records = tmp_path / "records"
    shutil.copytree(study.records_dir, records)
    log = draw_log_path("sampling", "full_run", records)
    with log.open("ab") as stream:
        stream.write(b'{"key": ["macro-2016-05", "identifying", 0], "ver')
    before = draw_log_state(records)
    out_root = tmp_path / "out"
    with pytest.raises(ReplayError, match="incomplete last line") as refusal:
        replay(inputs_of(study, records_dir=records), out_root)
    assert str(log) in str(refusal.value)
    assert draw_log_state(records) == before
    assert not out_root.exists(), "refused before anything is written"
    args = cli_args(study, tmp_path / "out-cli")
    args[args.index("--records-dir") + 1] = str(records)
    assert main(args) == 2
    assert "incomplete last line" in capsys.readouterr().err
    assert draw_log_state(records) == before and not (tmp_path / "out-cli").exists()


def test_the_licensing_guard_fails_on_a_planted_leak(study, replayed, tmp_path):
    """A raw BAA10Y level in a CSV, a DTB3 run in a markdown, an FMP close in an SVG, a reply in a markdown: all
    four are found; the BAA10Y level, the close and the reply refuse, the DTB3 run (public data) only warns."""
    copy = tmp_path / "published"
    shutil.copytree(replayed.published, copy)
    level = macro_row(MONTHS[7]).raw["baa10y"]
    (copy / "gap.csv").write_text((copy / "gap.csv").read_text(encoding="utf-8") + f"leak,{level!r}\n",
                                  encoding="utf-8")
    rates = " | ".join(f"{dtb3(month):.2f}" for month in MONTHS[20:23])
    (copy / "reliability.md").write_text((copy / "reliability.md").read_text(encoding="utf-8")
                                         + f"\n| cash | {rates} |\n", encoding="utf-8")
    close = closes(MONTHS[30])["tech_sector"]
    svg = copy / "convergence.svg"
    svg.write_text(svg.read_text(encoding="utf-8").replace("</svg>", f"<text>{close!r}</text></svg>"),
                   encoding="utf-8")
    reply = next(probe.texts[0] for probe in reply_probes(study.plan, study.records_dir))
    (copy / "draws.md").write_text((copy / "draws.md").read_text(encoding="utf-8") + f"\n{reply}\n",
                                   encoding="utf-8")
    leaks = find_leaks(published_text_files(copy), fixture_probes(study))
    found = {(leak.path.name, leak.label) for leak in leaks}
    assert found == {("gap.csv", "baa10y raw levels"), ("reliability.md", "DTB3 rates"),
                     ("convergence.svg", "FMP adjusted closes"), ("draws.md", REPLY_LABEL)}
    assert any(leak.found == repr(level) for leak in leaks)
    # The replay's own probes (items.json, the FMP and DTB3 cache bodies, the draw logs) find the same plants.
    own = find_leaks(published_text_files(copy), replay_probes(inputs_of(study)).probes)
    assert {(leak.path.name, leak.label) for leak in own} == {
        ("gap.csv", "BAA10Y raw levels (FRED, Moody's)"), ("reliability.md", DTB3_LABEL),
        ("convergence.svg", CLOSES_LABEL), ("draws.md", REPLY_LABEL)}
    assert all(leak.found not in str(leak) for leak in own), "a leak prints no matched value"
    # DTB3 is public data: its match is advisory (a manifest warning by file, label and count), the others refuse.
    refusing, warnings = partition_leaks(own)
    assert {(leak.path.name, leak.label) for leak in refusing} == {
        ("gap.csv", "BAA10Y raw levels (FRED, Moody's)"), ("convergence.svg", CLOSES_LABEL),
        ("draws.md", REPLY_LABEL)}
    assert warnings and {(leak.path.name, leak.label) for leak in warnings} == {("reliability.md", DTB3_LABEL)}
    # The fixture's rates repeat every 23 months, so the planted run matches every identical run probe.
    assert warning_records(warnings) == [{"file": "reliability.md", "label": DTB3_LABEL, "count": len(warnings)}]
    assert all(leak.found not in json.dumps(warning_records(warnings)) for leak in warnings)
    # The replay's own guard: an items.json whose raw level equals a derived number of a published table refuses
    # the replay, removes the published tree, keeps the private documents and names no value.
    planted = next(float(cell) for row in csv_rows(replayed.published / "gap.csv") for cell in row.values()
                   if len(cell) >= 8 and cell.replace("-", "").replace(".", "").isdigit())
    record = json.loads(study.items_path.read_text(encoding="utf-8"))
    for index in range(3):
        record["macro_items"][index]["row"]["raw"]["baa10y"] = planted
    items_path = tmp_path / "items-planted.json"
    items_path.write_text(json.dumps(record), encoding="utf-8")
    out_root = tmp_path / "planted"
    with pytest.raises(ReplayError, match="licensing probe") as refusal:
        replay(inputs_of(study, items_path=items_path), out_root)
    message = str(refusal.value)
    assert "gap.csv: BAA10Y raw levels (FRED, Moody's) (" in message
    assert repr(planted) not in message and f"{planted:.6g}" not in message
    assert not (out_root / "docs").exists() and (out_root / PRIVATE_DIR).is_dir()
    assert not (out_root / MANIFEST_NAME).exists()
    args = cli_args(study, tmp_path / "planted-cli")
    args[args.index("--items") + 1] = str(items_path)
    assert main(args) == 2


def close_like_run(published: Path) -> tuple[Path, tuple[str, ...], list[str]]:
    """The first published figure whose SVG carries, as three consecutive number tokens, a run that reads like
    closes (three distinct values, each at least 10 with at most two decimals): its path, its tokens and the run.
    Which figure carries such a run depends on its coordinates, so the tests take the first that does."""
    for name in PUBLISHED_FIGURES:
        svg = published / f"{name}.svg"
        tokens = number_tokens(svg.read_text(encoding="utf-8"))
        for index in range(len(tokens) - 2):
            window = tokens[index:index + 3]
            if len(set(window)) == 3 and all("." in text and "e" not in text and float(text) >= 10
                                             and len(text.split(".")[1]) <= 2 for text in window):
                return svg, tokens, list(window)
    raise AssertionError("no published figure carries a run of three close-like number tokens")


def test_an_fmp_close_a_published_svg_carries_makes_the_replay_refuse(study, replayed, tmp_path, capsys):
    """A single distinctive close and a run of short closes, planted through the private FMP cache, that
    the published SVG carries as numbers: the replay's own probes find them, the replay refuses, removes the
    published tree, exits 2 and prints no close."""
    svg, tokens, run = close_like_run(replayed.published)
    single = next(text for text in tokens if len(text) >= 8 and float(text) > 1)
    planted = tmp_path / "fmp"
    shutil.copytree(study.fmp_cache, planted)
    world, gold = (next(planted.glob(f"{PRICE_SYMBOLS[category]}_*.json")) for category in ("world_equity",
                                                                                            "gold_commodity"))
    rows = json.loads(world.read_text(encoding="utf-8"))
    rows[5]["adjClose"] = float(single)
    world.write_text(json.dumps(rows), encoding="utf-8")
    rows = json.loads(gold.read_text(encoding="utf-8"))
    for offset, text in enumerate(run):
        rows[10 + offset]["adjClose"] = float(text)
    gold.write_text(json.dumps(rows), encoding="utf-8")
    inputs = inputs_of(study, fmp_cache=planted)
    own = replay_probes(inputs)
    assert own.closes_probed
    leaks = [leak for leak in find_leaks([svg], own.probes) if leak.label == CLOSES_LABEL]
    assert {single, ", ".join(run)} <= {leak.found for leak in leaks}
    out_root = tmp_path / "planted"
    with pytest.raises(ReplayError, match="licensing probe") as refusal:
        replay(inputs, out_root)
    message = str(refusal.value)
    assert f"{svg.name}: {CLOSES_LABEL} (" in message
    assert single not in message and run[1] not in message
    assert not (out_root / "docs").exists() and (out_root / PRIVATE_DIR).is_dir()
    assert not (out_root / MANIFEST_NAME).exists()
    args = cli_args(study, tmp_path / "planted-cli")
    args[args.index("--fmp-cache") + 1] = str(planted)
    assert main(args) == 2
    captured = capsys.readouterr()
    assert CLOSES_LABEL in captured.err and single not in captured.err and run[1] not in captured.err
    assert not (tmp_path / "planted-cli" / "docs").exists()


def daily_body(symbol: str, month_end_values: dict[str, float]) -> list[dict]:
    """A daily-format FMP body: three two-decimal closes per month (days 5 and 15 and the month end), the
    month-end close of the months in ``month_end_values`` set to the given value."""
    rows = []
    for k, month in enumerate(MONTHS):
        month_end = macro_row(month).decision_date
        for day, bump in ((5, 0.37), (15, 0.91), (month_end.day, 0.0)):
            value = month_end_values.get(month) if day == month_end.day else None
            rows.append({"symbol": symbol, "date": month_end.replace(day=day).isoformat(),
                         "adjClose": round(60.0 * (1.0071 ** k) + bump, 2) if value is None else value})
    return rows


def test_month_end_closes_of_a_daily_fmp_body_are_probed_and_a_daily_run_alone_would_miss_them(
        study, replayed, tmp_path, capsys):
    """A real FMP body is daily; the study takes the last close of each month, so three consecutive month-end
    closes stand about twenty-one closes apart and no daily run matches them: the month-end series must be probed."""
    svg, _, run = close_like_run(replayed.published)
    planted = tmp_path / "fmp"
    shutil.copytree(study.fmp_cache, planted)
    gold = next(planted.glob(f"{PRICE_SYMBOLS['gold_commodity']}_*.json"))
    months = MONTHS[40:43]
    rows = daily_body(PRICE_SYMBOLS["gold_commodity"], dict(zip(months, (float(text) for text in run), strict=True)))
    gold.write_text(json.dumps(rows), encoding="utf-8")
    parsed = [(date.fromisoformat(row["date"]), row["adjClose"]) for row in rows]
    assert month_end_closes(parsed)[40:43] == [float(text) for text in run]
    assert len(month_end_closes(parsed)) == len(MONTHS) and len(parsed) == 3 * len(MONTHS)
    daily, bodies = fmp_probes(planted, month_ends=False)
    assert bodies == len(PRICE_SYMBOLS)
    assert not [leak for leak in find_leaks([svg], daily) if leak.label == CLOSES_LABEL], \
        "the daily runs alone do not see three month-end closes"
    inputs = inputs_of(study, fmp_cache=planted)
    own = replay_probes(inputs)
    assert own.fmp_bodies == len(PRICE_SYMBOLS)
    leaks = [leak for leak in find_leaks([svg], own.probes) if leak.label == CLOSES_LABEL]
    assert ", ".join(run) in {leak.found for leak in leaks}
    out_root = tmp_path / "planted"
    with pytest.raises(ReplayError, match="licensing probe") as refusal:
        replay(inputs, out_root)
    message = str(refusal.value)
    assert f"{svg.name}: {CLOSES_LABEL} (" in message and run[1] not in message
    assert not (out_root / "docs").exists() and (out_root / PRIVATE_DIR).is_dir()
    args = cli_args(study, tmp_path / "planted-cli")
    args[args.index("--fmp-cache") + 1] = str(planted)
    assert main(args) == 2
    captured = capsys.readouterr()
    assert CLOSES_LABEL in captured.err and run[1] not in captured.err
    assert not (tmp_path / "planted-cli" / "docs").exists()


def short_run(tokens: tuple[str, ...], *, minimum: float = 0.0, decimals: int = 2) -> list[str]:
    """Three distinct consecutive plain number tokens (no exponent, under eight characters, at most ``decimals``
    decimals, each at least ``minimum``): a rate-like run a cached FRED body can carry, probed as a run only."""
    for index in range(len(tokens) - 2):
        window = tokens[index:index + 3]
        if len(set(window)) == 3 and all("." in text and "e" not in text and len(text) < 8 and float(text) >= minimum
                                         and len(text.split(".")[1]) <= decimals for text in window):
            return list(window)
    raise AssertionError("the figure carries no short run of three distinct number tokens")


def planted_dtb3_cache(study: SimpleNamespace, replayed: SimpleNamespace, tmp_path: Path) -> tuple[Path, list[str]]:
    """A copy of the fixture's FRED cache whose DTB3 body carries, as three consecutive rates, a run of short number
    tokens that the published gap figure carries (SVG coordinates): the real 9.3 finding, planted."""
    svg = replayed.published / "gap.svg"
    run = short_run(number_tokens(svg.read_text(encoding="utf-8")))
    planted = tmp_path / "fred"
    shutil.copytree(study.fred_cache, planted)
    body_path = next(planted.glob(f"{CASH_SERIES}_current_*.json"))
    body = json.loads(body_path.read_text(encoding="utf-8"))
    for offset, text in enumerate(run):
        body["observations"][10 + offset]["value"] = text
    body_path.write_text(json.dumps(body), encoding="utf-8")
    return planted, run


def test_a_dtb3_run_a_published_file_carries_is_recorded_as_a_warning_and_the_replay_publishes(
        study, replayed, tmp_path, capsys):
    """DTB3 (the FRED H.15 three-month bill rate) is public data, not a licensed series: a run of its rates that a
    published SVG carries as coordinates is recorded in replay.json under guard.warnings (file, label, count), the
    published tree is written and the exit code is 0, also under --require-guard; the warning names no rate."""
    planted, run = planted_dtb3_cache(study, replayed, tmp_path)
    inputs = inputs_of(study, fred_cache=planted)
    own = replay_probes(inputs)
    assert own.dtb3_probed and own.dtb3_bodies == 1
    svg = replayed.published / "gap.svg"
    leaks = find_leaks([svg], own.probes)
    assert {leak.label for leak in leaks} == {DTB3_LABEL} and ", ".join(run) in {leak.found for leak in leaks}
    refusing, warnings = partition_leaks(leaks)
    assert refusing == [] and warnings == leaks
    assert warning_records(warnings) == [{"file": "gap.svg", "label": DTB3_LABEL, "count": len(leaks)}]
    out_root = tmp_path / "warned"
    output = replay(inputs, out_root, require_guard=True)
    guard = output.manifest["guard"]
    assert guard["required"] is True and guard["dtb3_probed"] is True and guard["leaks"] == 0
    assert guard["warning_labels"] == [DTB3_LABEL] and guard["probes"][DTB3_LABEL] == own.counts()[DTB3_LABEL]
    assert {record["label"] for record in guard["warnings"]} == {DTB3_LABEL}
    assert {record["file"] for record in guard["warnings"]} >= {"gap.svg"}
    assert all(set(record) == {"file", "label", "count"} and record["count"] >= 1 for record in guard["warnings"])
    assert all(text not in json.dumps(guard["warnings"]) for text in run)
    assert (out_root / MANIFEST_NAME).exists()
    assert json.loads((out_root / MANIFEST_NAME).read_text(encoding="utf-8"))["guard"]["warnings"] == guard["warnings"]
    written = out_root / PUBLISHED_DIR
    assert sorted(path.name for path in written.iterdir()) == sorted(path.name for path in replayed.published.iterdir())
    assert {leak.label for leak in find_leaks([written / "gap.svg"], own.probes)} == {DTB3_LABEL}, \
        "the figure that carries the run is published as rendered"
    args = cli_args(study, tmp_path / "warned-cli")
    args[args.index("--fred-cache") + 1] = str(planted)
    assert main([*args, "--require-guard"]) == 0
    captured = capsys.readouterr()
    assert "advisory" in captured.err and f"gap.svg: {DTB3_LABEL} (" in captured.err
    assert all(text not in captured.err for text in run)
    assert (tmp_path / "warned-cli" / PUBLISHED_DIR).is_dir() and (tmp_path / "warned-cli" / MANIFEST_NAME).exists()


def test_a_planted_baa10y_level_or_fmp_close_still_refuses_beside_a_dtb3_warning(study, replayed, tmp_path, capsys):
    """The advisory label loosens nothing else. With the DTB3 run planted in the FRED cache, a BAA10Y raw level
    planted in items.json or an FMP close planted in the FMP cache refuses the replay as before: the published tree
    removed, no manifest, exit 2, the refusal counting and naming the refusing matches only and never a value."""
    planted_fred, _ = planted_dtb3_cache(study, replayed, tmp_path)
    files = published_text_files(replayed.published)
    level = next(float(cell) for row in csv_rows(replayed.published / "gap.csv") for cell in row.values()
                 if len(cell) >= 8 and cell.replace("-", "").replace(".", "").isdigit())
    record = json.loads(study.items_path.read_text(encoding="utf-8"))
    for index in range(3):
        record["macro_items"][index]["row"]["raw"]["baa10y"] = level
    items_path = tmp_path / "items-planted.json"
    items_path.write_text(json.dumps(record), encoding="utf-8")
    inputs = inputs_of(study, items_path=items_path, fred_cache=planted_fred)
    refusing, warnings = partition_leaks(find_leaks(files, replay_probes(inputs).probes))
    assert {leak.label for leak in refusing} == {"BAA10Y raw levels (FRED, Moody's)"}
    assert {leak.label for leak in warnings} == {DTB3_LABEL}
    out_root = tmp_path / "baa10y"
    with pytest.raises(ReplayError, match="licensing probe") as refusal:
        replay(inputs, out_root, require_guard=True)
    message = str(refusal.value)
    assert message.startswith(f"{len(refusing)} licensing probe(s)")
    assert "gap.csv: BAA10Y raw levels (FRED, Moody's) (" in message
    assert DTB3_LABEL not in message and repr(level) not in message and f"{level:.6g}" not in message
    assert not (out_root / "docs").exists() and (out_root / PRIVATE_DIR).is_dir()
    assert not (out_root / MANIFEST_NAME).exists()
    svg = replayed.published / "gap.svg"
    single = next(text for text in number_tokens(svg.read_text(encoding="utf-8")) if len(text) >= 8 and float(text) > 1)
    planted_fmp = tmp_path / "fmp"
    shutil.copytree(study.fmp_cache, planted_fmp)
    world = next(planted_fmp.glob(f"{PRICE_SYMBOLS['world_equity']}_*.json"))
    rows = json.loads(world.read_text(encoding="utf-8"))
    rows[5]["adjClose"] = float(single)
    world.write_text(json.dumps(rows), encoding="utf-8")
    inputs = inputs_of(study, fmp_cache=planted_fmp, fred_cache=planted_fred)
    out_root = tmp_path / "close"
    with pytest.raises(ReplayError, match="licensing probe") as refusal:
        replay(inputs, out_root, require_guard=True)
    message = str(refusal.value)
    assert f"gap.svg: {CLOSES_LABEL} (" in message and DTB3_LABEL not in message and single not in message
    assert not (out_root / "docs").exists() and not (out_root / MANIFEST_NAME).exists()
    args = cli_args(study, tmp_path / "close-cli")
    args[args.index("--fmp-cache") + 1] = str(planted_fmp)
    args[args.index("--fred-cache") + 1] = str(planted_fred)
    assert main([*args, "--require-guard"]) == 2
    captured = capsys.readouterr()
    assert CLOSES_LABEL in captured.err and "advisory" not in captured.err and single not in captured.err
    assert not (tmp_path / "close-cli" / "docs").exists()


def test_require_guard_refuses_before_writing_unless_every_private_source_was_probed(study, tmp_path, capsys):
    out_root = tmp_path / "unwritten"
    with pytest.raises(ReplayError, match=r"missing: items\.json \(--items\)"):
        replay(inputs_of(study, items_path=None), out_root, require_guard=True)
    assert not out_root.exists()
    empty = tmp_path / "no-closes"
    empty.mkdir()
    with pytest.raises(ReplayError, match=r"3 cached FMP bodies, one per priced symbol \(--fmp-cache holds 0\)"):
        replay(inputs_of(study, fmp_cache=empty), out_root, require_guard=True)
    partial = tmp_path / "two-symbols"
    shutil.copytree(study.fmp_cache, partial)
    next(partial.glob(f"{PRICE_SYMBOLS['tech_sector']}_*.json")).unlink()
    with pytest.raises(ReplayError, match=r"\(--fmp-cache holds 2\)"):
        replay(inputs_of(study, fmp_cache=partial), out_root, require_guard=True)
    with pytest.raises(ReplayError, match=rf"the cached {CASH_SERIES} body \(--fred-cache\)"):
        replay(inputs_of(study, fred_cache=empty), out_root, require_guard=True)
    assert not out_root.exists()
    check_guard_inputs(replay_probes(inputs_of(study)))  # the fixture has every source
    args = cli_args(study, out_root)
    args[args.index("--fred-cache") + 1] = str(empty)
    assert main([*args, "--require-guard"]) == 2
    assert "--fred-cache" in capsys.readouterr().err and not out_root.exists()
    output = replay(inputs_of(study, fred_cache=empty), tmp_path / "written-anyway")
    guard = output.manifest["guard"]
    assert (guard["required"], guard["dtb3_probed"], guard["dtb3_bodies"]) == (False, False, 0), \
        "without the flag a missing source is recorded, not refused"
    assert (output.root / PUBLISHED_DIR).is_dir() and DTB3_LABEL not in guard["probes"]


def test_probes_are_distinctive_values_runs_of_consecutive_values_or_texts():
    assert value_probe("x", 1.93) is None and value_probe("x", "1.93") is None and value_probe("x", True) is None
    assert value_probe("x", 1.9300000011).texts == ("1.9300000011",)
    assert series_probe("x", [1.93, 1.95, 1.97]).texts == ("1.93", "1.95", "1.97")
    assert series_probe("x", [1.93]) is None and series_probe("x", [1.93, None, 1.97]) is None
    assert text_probe("x", "   short ") is None and text_probe("x", '{"outlook": 0.2}') is None
    loadings = '{"credit_stress": 0.1, "growth": -0.2, "inflation": 0.3}'
    assert text_probe("x", f" {loadings}\n").texts == (loadings,)
    with pytest.raises(ValueError):
        Probe("x", "other", ("1",))
    with pytest.raises(ReplayError, match="schema"):
        items_probes({"schema": "other"})
    probes = items_probes({"schema": ITEMS_SCHEMA, "macro_items": [
        {"row": {"raw": {"baa10y": 1.93, "cpi_yoy": 3.1234567}}, "realized": {"world_equity": 0.0123456789}},
        {"row": {"raw": {"baa10y": 1.95, "cpi_yoy": 3.2}}, "realized": None},
        {"row": {"raw": {"baa10y": 1.97, "cpi_yoy": 3.3}}, "realized": {"world_equity": -0.02}}]})
    assert [(probe.label, probe.kind, probe.texts) for probe in probes] == [
        ("BAA10Y raw levels (FRED, Moody's)", "series", ("1.93", "1.95", "1.97")),
        ("CPI year-over-year raw levels (FRED)", "value", ("3.1234567",)),
        ("CPI year-over-year raw levels (FRED)", "series", ("3.1234567", "3.2", "3.3")),
        ("realized returns (from FMP adjusted closes and DTB3)", "value", ("0.0123456789",)),
    ]


def test_fmp_and_dtb3_probes_read_the_cached_bodies_with_the_assemblys_parsers(tmp_path):
    assert fmp_probes(tmp_path / "missing") == ([], 0) and dtb3_probes(tmp_path / "missing") == ([], 0)
    fmp = tmp_path / "fmp"
    fmp.mkdir()
    rows = [{"symbol": "XLK", "date": f"2024-0{month}-05", "adjClose": value}
            for month, value in zip((1, 2, 3, 4), (210.5, 211.123456, 209.75, 212.0), strict=True)]
    (fmp / "XLK_2024-01-01_2024-04-30.json").write_text(json.dumps(rows[::-1]), encoding="utf-8")
    (fmp / "notes.txt").write_text("not a body", encoding="utf-8")
    probes, bodies = fmp_probes(fmp)
    assert bodies == 1
    assert [(probe.kind, probe.texts) for probe in probes] == [
        ("value", ("211.123456",)), ("series", ("210.5", "211.123456", "209.75")),
        ("series", ("211.123456", "209.75", "212.0"))], "one close per month: the month-end series adds nothing"
    assert {probe.label for probe in probes} == {CLOSES_LABEL}
    assert fmp_probes(fmp, month_ends=False) == (probes, 1)
    daily = [{"date": day, "adjClose": value} for day, value in (
        ("2024-01-10", 10.5), ("2024-01-31", 11.25), ("2024-02-09", 12.75), ("2024-02-29", 13.5),
        ("2024-03-15", 14.25), ("2024-03-28", 15.75))]
    (fmp / "SWDA.L_2024-01-01_2024-04-30.json").write_text(json.dumps(daily), encoding="utf-8")
    assert month_end_closes([(date.fromisoformat(row["date"]), row["adjClose"]) for row in daily]) == [
        11.25, 13.5, 15.75]
    both, bodies = fmp_probes(fmp)
    assert bodies == 2
    added = [probe.texts for probe in both if probe.texts not in {p.texts for p in probes}]
    assert added == [("10.5", "11.25", "12.75"), ("11.25", "12.75", "13.5"), ("12.75", "13.5", "14.25"),
                     ("13.5", "14.25", "15.75"), ("11.25", "13.5", "15.75")]
    assert ("11.25", "13.5", "15.75") not in {probe.texts for probe in fmp_probes(fmp, month_ends=False)[0]}
    (fmp / "IAU_2024-01-01_2024-04-30.json").write_text(json.dumps([{"date": "2024-05-01", "adjClose": 1.0}]),
                                                        encoding="utf-8")
    with pytest.raises(ReplayError, match="IAU_2024-01-01_2024-04-30.json cannot be read"):
        fmp_probes(fmp)
    fred = tmp_path / "fred"
    fred.mkdir()
    body = {"observations": [{"date": "2024-01-02", "value": "5.21"}, {"date": "2024-01-03", "value": "."},
                             {"date": "2024-01-04", "value": "5.23"}, {"date": "2024-01-05", "value": "5.2"}]}
    (fred / f"{CASH_SERIES}_current_0123456789abcdef.json").write_text(json.dumps(body), encoding="utf-8")
    (fred / "CPIAUCSL_vintage_0123456789abcdef.json").write_text("{}", encoding="utf-8")
    probes, bodies = dtb3_probes(fred)
    assert bodies == 1 and [(probe.label, probe.kind, probe.texts) for probe in probes] == [
        (DTB3_LABEL, "series", ("5.21", "5.23", "5.2"))]
    (fred / f"{CASH_SERIES}_current_fedcba9876543210.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ReplayError, match="cannot be read"):
        dtb3_probes(fred)


def test_find_leaks_matches_numbers_as_values_not_as_digits_inside_other_tokens(tmp_path):
    path = tmp_path / "table.md"
    path.write_text("| 11.93 | 1.935 | macro-1.93 | 1.930 |\n| 1.95 x 1.97 |\n", encoding="utf-8")
    standalone = Probe("x", "value", ("1.93",))
    assert [leak.found for leak in find_leaks([path], [standalone])] == ["1.93"]
    assert find_leaks([path], [Probe("x", "value", ("11.935",))]) == []
    run = Probe("y", "series", ("1.93", "1.95", "1.97"))
    assert [leak.label for leak in find_leaks([path], [run])] == ["y"]
    assert find_leaks([path], [Probe("y", "series", ("1.95", "1.93", "1.97"))]) == []
    path.write_text("| 1.93 |\n" + "| 0.5 |\n" * 100 + "| 1.95 | 1.97 |\n", encoding="utf-8")
    assert find_leaks([path], [run]) == [], "a run must stand within the span of number tokens"


def test_the_module_imports_without_matplotlib():
    code = ("import sys\n"
            "import scripts.finance_sim.replay\n"
            "heavy = sorted(m for m in sys.modules if m.split('.')[0] == 'matplotlib')\n"
            "assert not heavy, heavy\n"
            "print('lean-ok')\n")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stderr
    assert "lean-ok" in result.stdout
