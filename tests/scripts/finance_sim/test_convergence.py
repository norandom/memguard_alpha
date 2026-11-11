"""Top-k versus full-distribution convergence (task 7.12; Requirements 9.1 to 9.4).

Recorded fixtures only. A small frozen plan is calibrated and a few evaluation
months are executed once, on fakes, through the real 7.6 calibration and 7.7
execution (a generation-host double answering through ``RequestControl`` with
the stage's ``LedgerCallBudget``, and a full-distribution double that answers
batches from a seeded logit vector). The draws go to the 7.4 ``DrawLog``, the
evidence batches to evidence files in the recording's format and the results
to the 7.7 result files, all under ``tmp_path``. The convergence analysis then
reads only those records: no client is built, sockets are blocked, and no
credential, host or file under ``runs/`` is used.
"""

from __future__ import annotations

import calendar
import hashlib
import inspect
import json
import math
import random
import shutil
import socket
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.finance_sim.convergence as convergence_module
from recall_guard.core.completion import ModelConfig, ModelResponse, TokenAlternative, TokenEvidence
from recall_guard.core.full_distribution import FullDistributionEvidence, logprobs_from_logits
from recall_guard.core.nvidia_lm import TokenLogprob
from recall_guard.core.request_control import CallBudget, RequestControl
from recall_guard.dataset.macro_regime import (
    MACRO_AXES,
    MacroRow,
    RealizedReturns,
    regime_definition,
    render_anonymised,
    render_identifying,
)
from recall_guard.harness.guarded_decision import DrawMode, GuardConfig, decide_from_draws
from recall_guard.mia.features import compute_mia_features
from scripts.finance_sim.calibrate import DrawSettings, EdgarRelease, EvidenceSettings, FitSettings, run_calibration
from scripts.finance_sim.convergence import (
    CONVERGENCE_KS,
    DRAWS_LACKING,
    HOST_TOP_K,
    NO_DECISION,
    SERVED_TOP_K,
    ConvergenceError,
    convergence_from_records,
    convergence_table,
    main,
    table_path,
    write_table,
)
from scripts.finance_sim.execute import (
    ExecutionDraw,
    SourceEstimate,
    execute_items,
    full_distribution_estimate,
    load_calibrators,
    macro_items,
    planned_keys,
)
from scripts.finance_sim.items import MacroItem, RealizedOutcome, row_to_dict
from scripts.finance_sim.limits import BudgetLedger, LedgerCallBudget, RequestAllowance, StudyLimits, budget_path
from scripts.finance_sim.plan import (
    ARM_PROFILE_FACTS,
    ARM_REGISTRY_IDS,
    EDGAR_FILES,
    ArmPlan,
    EdgarRow,
    MonthClass,
    PilotSettings,
    PlanChangedError,
    StudyArms,
    StudyPlan,
    freeze_plan,
    load_serving_manifest,
    split_months,
    write_plan,
)
from scripts.finance_sim.recording import EVIDENCE_SCHEMA, DrawLog, draw_log_path, evidence_path, pinned_digest
from tests.scripts.finance_sim.test_recording_draws import CUT, aged, unchanged, writing_calls

GEMMA = "google/gemma-3-4b-it"
CREATED_AT = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)
SEED = 20261110
#: Small ensembles keep the fakes fast; the analysis reads every size from the plan and the results.
DRAWS, MIN_PARSED = 4, 3
SEED_BASE, TEMPERATURE = 1000, 0.7
PARAMETERS = {"temperature": 0.0, "top_p": 1, "max_tokens": 512, "logprobs": True, "top_logprobs": 5}
ALLOWANCE = RequestAllowance(output_tokens=512, timeout_s=30)
#: The fake full-distribution host: a small vocabulary, six response positions, 20 kept alternatives.
N_VOCAB, POSITIONS, KEEP = 64, 6, 20
#: Evaluation months executed in the fixture; the other planned months stay unexecuted.
EXECUTED_MONTHS = 3
#: The generation call answered with prose: draw 0 of the first item, so that item's single-draw variants fail
#: (an unparsed draw has no evidence by design) while its ensemble still meets the minimum.
PROSE_CALL = 0
FEATURES = ("loss", "min_k", "min_k_pp", "zlib_ratio", "estimate")


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_sha256(value) -> str:
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))


def month_range(first: str, last: str) -> list[str]:
    year, month = int(first[:4]), int(first[5:])
    out = []
    while f"{year:04d}-{month:02d}" <= last:
        out.append(f"{year:04d}-{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return out


MONTHS = month_range("2016-01", "2026-06")


def macro_row(month: str) -> MacroRow:
    year, number = int(month[:4]), int(month[5:])
    k = MONTHS.index(month)
    return MacroRow(decision_date=date(year, number, calendar.monthrange(year, number)[1]),
                    raw={"baa10y": 2.0 + 0.01 * k, "cpi_yoy": 1.5 + 0.02 * k, "t10y2y": 0.5 - 0.01 * k},
                    zscores={"baa10y_z": 0.1 * (k % 7) - 0.3, "cpi_yoy_z": 0.05 * (k % 11) - 0.2,
                             "t10y2y_z": -0.04 * (k % 5) + 0.1},
                    revised={"BAA10Y": False, "CPIAUCSL": False, "T10Y2Y": False})


def realized(month: str) -> RealizedReturns:
    rng = random.Random(f"realized:{month}")
    return RealizedReturns(world_equity=rng.uniform(-0.05, 0.05), tech_sector=rng.uniform(-0.07, 0.07),
                           gold_commodity=rng.uniform(-0.04, 0.04), short_treasury_cash=0.003)


def macro_item(month: str) -> MacroItem:
    row = macro_row(month)
    return MacroItem(f"macro-{month}", row, {}, RealizedOutcome(row.decision_date, realized(month), None, {}))


MACRO_ITEMS = [macro_item(month) for month in MONTHS]


def item_hash(item: MacroItem) -> str:
    prompts = {"anonymised": digest(render_anonymised(item.row)), "identifying": digest(render_identifying(item.row))}
    return canonical_sha256({"prompt_sha256": prompts, "row_sha256": canonical_sha256(row_to_dict(item.row))})


EDGAR_ROWS = ([EdgarRow(f"cal-mem-{i:03d}", 1, "calibration") for i in range(40)]
              + [EdgarRow(f"cal-ctl-{i:03d}", 0, "calibration") for i in range(100)]
              + [EdgarRow(f"held-ctl-{i:03d}", 0, "held_out") for i in range(40)]
              + [EdgarRow(f"held-mem-{i:03d}", 1, "held_out") for i in range(40)])


def small_arm(arm: str) -> ArmPlan:
    """The arm's frozen settings on its profile facts (those ``for_arm`` takes), with small ensembles."""
    accepted = inspect.signature(ArmPlan.for_arm).parameters
    facts = {key: value for key, value in ARM_PROFILE_FACTS[arm].items() if key in accepted}
    return replace(ArmPlan.for_arm(arm, **facts), draws=DRAWS, min_parsed=MIN_PARSED)


def frozen_plan(seed: int = SEED) -> StudyPlan:
    months = [MonthClass(month, {GEMMA: month <= "2024-08"}) for month in MONTHS]
    items = {item.item_id: item_hash(item) for item in MACRO_ITEMS} | {row.row_id: digest(row.row_id)
                                                                      for row in EDGAR_ROWS}
    arms = StudyArms(registry_ids=ARM_REGISTRY_IDS, arms={arm: small_arm(arm) for arm in ("sampling", "serving")})
    pilot = replace(PilotSettings.for_split(split_months(months, seed=seed), arms), draws=2, min_parsed=2)
    return freeze_plan(months, EDGAR_ROWS, seed=seed, created_at=CREATED_AT, items=items,
                       edgar_files={name: digest(name) for name in EDGAR_FILES}, arms=arms,
                       serving_manifest=load_serving_manifest(), pilot=pilot)


PLAN = frozen_plan()
#: The sampling arm's pinned GGUF digest, from its frozen provider profile (a local file read).
DIGEST = pinned_digest(PLAN, "sampling")
ARM = PLAN.arms.arms["sampling"]


def memorized(prompt: str) -> bool:
    """The fakes' ground truth: identifying macro prompts and memorized EDGAR releases."""
    return "Real asset identities:" in prompt or "(recalled)" in prompt


def edgar_releases() -> list[EdgarRelease]:
    def text(row_id: str, label: int) -> str:
        marker = " (recalled)" if label == 1 else ""
        return f"Company {row_id} reports quarterly results{marker}. Revenue rose and the outlook is unchanged."

    releases = [EdgarRelease(row_id, 1, text(row_id, 1)) for row_id in PLAN.edgar.calibration_memorized]
    return releases + [EdgarRelease(row_id, 0, text(row_id, 0)) for row_id in PLAN.edgar.calibration_control]


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def model_config() -> ModelConfig:
    return ModelConfig(protocol="callback", endpoint=None, model="gemma3:4b", parameters=dict(PARAMETERS),
                       timeout_s=30, max_retries=0, retry_backoff_s=0)


def base_logprob(prompt: str, seed: object) -> float:
    noise = random.Random(f"{digest(prompt)}:{seed}").uniform(-0.3, 0.3)
    return (-0.15 if memorized(prompt) else -1.2) + noise


def served_evidence(base: float) -> tuple[TokenEvidence, ...]:
    """Six realized tokens with the arm's five alternatives each, as the generation host returns them."""
    values = [min(0.0, base - 0.02 * i) for i in range(POSITIONS)]
    return tuple(TokenEvidence("t", value, tuple(TokenAlternative(token, value - step)
                                                 for token, step in zip("tuvwx", (0.0, 1.0, 2.0, 3.0, 4.0),
                                                                        strict=True)))
                 for value in values)


def reply(prompt: str, seed: object) -> str:
    rng = random.Random(f"reply:{digest(prompt)}:{seed}")
    if "Press release:" in prompt:
        return json.dumps({"outlook": rng.choice([-0.3, 0.1, 0.2, 0.4])})
    return json.dumps({axis: rng.choice([-0.4, -0.2, 0.0, 0.1, 0.3, 0.5]) for axis in MACRO_AXES})


class FakeHost:
    """Generation-host double; every reply runs through RequestControl with the client's budget."""

    def __init__(self, budget: CallBudget, *, prose=()) -> None:
        self.config = model_config()
        self.budget = budget
        self._control = RequestControl(self.config, budget, clock=Clock())
        self.prose = set(prose)
        self.calls = 0

    def generate(self, prompt: str, **parameters) -> ModelResponse:
        index = self.calls
        self.calls += 1

        def operation() -> ModelResponse:
            seed = parameters.get("seed")
            content = "The regime looks benign to me." if index in self.prose else reply(prompt, seed)
            return ModelResponse(content, served_evidence(base_logprob(prompt, seed)), usage={"completion_tokens": 7},
                                 requested_settings=dict(self.config.parameters) | parameters)

        return self._control.run(operation, stage="study")


def full_evidence(prompt: str, text: str, keep: int) -> FullDistributionEvidence:
    """Evidence of a seeded logit vector: the realized token is boosted more for a memorized prompt."""
    rng = np.random.default_rng(int(digest(f"full:{digest(prompt)}:{digest(text)}")[:16], 16))
    logits = rng.normal(0.0, 1.5, size=(POSITIONS, N_VOCAB))
    tokens = rng.integers(0, N_VOCAB, size=POSITIONS)
    logits[np.arange(POSITIONS), tokens] += 4.0 if memorized(prompt) else 1.0
    logprobs = logprobs_from_logits(logits)
    order = np.argsort(-logprobs, axis=1, kind="stable")
    kept = min(keep, N_VOCAB)
    return FullDistributionEvidence(
        model_digest=DIGEST, n_vocab=N_VOCAB, response_token_ids=tuple(int(token) for token in tokens),
        logprob=tuple(float(logprobs[i, token]) for i, token in enumerate(tokens)),
        entropy=tuple(float(-(np.exp(row) * row).sum()) for row in logprobs),
        rank=tuple(int(np.flatnonzero(order[i] == token)[0]) + 1 for i, token in enumerate(tokens)),
        top_logprobs=tuple(tuple((int(j), float(logprobs[i, j])) for j in order[i, :kept]) for i in range(POSITIONS)))


class FakeFull:
    """Full-distribution host double that answers batches only."""

    def evidence(self, prompt_text: str, response_text: str, *, top_k_keep: int = 100) -> FullDistributionEvidence:
        pytest.fail("evidence is requested per batch only")

    def evidence_batch(self, prompt_text: str, response_texts, *, top_k_keep: int = 100,
                       timeout_s: float | None = None) -> tuple[FullDistributionEvidence, ...]:
        return tuple(full_evidence(prompt_text, text, top_k_keep) for text in response_texts)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("the convergence analysis attempted a network connection")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)


def write_evidence_file(records_dir: Path, item, found: Mapping[int, FullDistributionEvidence]) -> None:
    """One evidence batch in the 7.4 recording's evidence-file format."""
    indices = sorted(found)
    path = evidence_path("sampling", "full_run", item.item_id, item.rendering, records_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema": EVIDENCE_SCHEMA, "plan_sha256": PLAN.plan_sha256, "arm": "sampling", "stage": "full_run",
        "item_id": item.item_id, "rendering": item.rendering, "prompt_sha256": item.prompt_sha256,
        "batch_sha256": canonical_sha256([item.prompt_sha256, indices]), "model_digest": DIGEST, "top_k_keep": KEEP,
        "draw_indices": indices, "evidence": [found[index].to_dict() for index in indices],
        "recorded_at": "2026-11-10T12:00:00Z"}, sort_keys=True), encoding="utf-8")


@pytest.fixture(scope="module")
def recorded(tmp_path_factory) -> SimpleNamespace:
    """Calibrate, then execute a few evaluation months once; everything recorded under tmp_path."""
    root = tmp_path_factory.mktemp("convergence")
    records_dir, results_dir, calibrators_dir = root / "records", root / "results", root / "calibrators"
    ledger = BudgetLedger.open(root / "calibration" / "budget.json", PLAN,
                               StudyLimits(PLAN.plan_sha256, "sampling", "calibration", 1000, 1_000_000, 1_000_000),
                               ALLOWANCE, clock=Clock())
    run_calibration(PLAN, arm="sampling", macro_rows=[macro_row(month) for month in PLAN.months.calibration],
                    edgar_releases=edgar_releases(), client=FakeHost(LedgerCallBudget(ledger)), ledger=ledger,
                    settings=DrawSettings(draws=2, temperature=TEMPERATURE, seed_base=SEED_BASE),
                    sources=("top_k", "full_distribution"), full_distribution=FakeFull(),
                    fit=FitSettings(min_auc=0.6, min_valid=10), evidence=EvidenceSettings(top_k_keep=KEEP),
                    calibrators_dir=calibrators_dir, sleep=lambda seconds: None)
    calibrators = load_calibrators(PLAN, arm="sampling", kind="macro", calibrators_dir=calibrators_dir)
    planned_ids = list(dict.fromkeys(item_id for item_id, _ in planned_keys(PLAN, "sampling", "full_run")))
    chosen = set(planned_ids[:EXECUTED_MONTHS])
    items = tuple(item for item in macro_items(PLAN, MACRO_ITEMS, stage="full_run") if item.item_id in chosen)
    assert len(items) == 2 * EXECUTED_MONTHS
    budget = PLAN.limits.stage("sampling", "full_run")
    stage_ledger = BudgetLedger.open(
        budget_path("sampling", "full_run", records_dir), PLAN,
        StudyLimits(PLAN.plan_sha256, "sampling", "full_run", budget.max_requests, budget.max_output_tokens,
                    budget.max_seconds), ALLOWANCE, clock=Clock())
    log = DrawLog(draw_log_path("sampling", "full_run", records_dir), plan=PLAN, loads=ExecutionDraw.from_dict)
    evidence: list[FullDistributionEvidence] = []

    def on_evidence(item, found) -> None:
        evidence.extend(found.values())
        write_evidence_file(records_dir, item, found)

    run = execute_items(PLAN, items, arm="sampling", stage="full_run",
                        client=FakeHost(LedgerCallBudget(stage_ledger), prose=(PROSE_CALL,)), ledger=stage_ledger,
                        calibrators=calibrators, full_distribution=FakeFull(), on_draw=log, on_evidence=on_evidence,
                        results_dir=results_dir, sleep=lambda seconds: None)
    return SimpleNamespace(root=root, records_dir=records_dir, results_dir=results_dir,
                           calibrators_dir=calibrators_dir, calibrators=calibrators, items=items,
                           results=tuple(run.results[item.key] for item in items), draws=tuple(log.latest()),
                           evidence=tuple(evidence))


def build(recorded, **overrides):
    values = {"arm": "sampling", "stage": "full_run", "calibrators": recorded.calibrators, "expected_digest": DIGEST}
    values |= overrides
    results = values.pop("results", recorded.results)
    draws = values.pop("draws", recorded.draws)
    evidence = values.pop("evidence", recorded.evidence)
    plan = values.pop("plan", PLAN)
    return convergence_table(plan, results, draws, evidence, **values)


@pytest.fixture(scope="module")
def table(recorded):
    return build(recorded)


# ---------- brute force on the same recorded responses ----------

def evidence_ref(evidence: FullDistributionEvidence) -> str:
    return canonical_sha256(evidence.to_dict())


def compared_draws(recorded) -> list[tuple]:
    """(result, draw index, reply, evidence) of every parsed draw, found independently of the module."""
    replies = {draw.key: draw.response for draw in recorded.draws}
    by_ref = {evidence_ref(entry): entry for entry in recorded.evidence}
    out = []
    for result in recorded.results:
        for draw, estimate in zip(result.recorded_draws, result.full_distribution, strict=True):
            if draw.parsed is not None:
                out.append((result, draw.draw_index, replies[(result.item_id, result.rendering, draw.draw_index)],
                            by_ref[estimate.evidence_ref]))
    return out


def brute_features(content: str, evidence: FullDistributionEvidence, k: int):
    entries = [TokenLogprob(token=str(token), logprob=logprob,
                            top_logprobs=[{"token": str(alt), "logprob": value} for alt, value in alternatives[:k]])
               for token, logprob, alternatives in zip(evidence.response_token_ids, evidence.logprob,
                                                       evidence.top_logprobs, strict=True)]
    return compute_mia_features(content, entries, None)


def feature_values(features, calibrator) -> dict[str, float | None]:
    try:
        estimate = calibrator.estimate(features)
    except ValueError:
        estimate = None
    return {"loss": features.loss, "min_k": features.min_k, "min_k_pp": features.min_k_pp,
            "zlib_ratio": features.zlib_ratio, "estimate": estimate}


def feature_row(table, evidence: str, k: int, feature: str):
    (row,) = [row for row in table.features if (row.evidence, row.k, row.feature) == (evidence, k, feature)]
    return row


def mass_row(table, evidence: str, k: int):
    (row,) = [row for row in table.mass if (row.evidence, row.k) == (evidence, k)]
    return row


def decision_row(table, evidence: str, k: int, mode: str):
    (row,) = [row for row in table.decisions if (row.evidence, row.k, row.mode) == (evidence, k, mode)]
    return row


def item_changes(table, evidence: str, k: int, mode: str) -> dict:
    return {(row.item_id, row.rendering): row for row in table.item_changes
            if (row.evidence, row.k, row.mode) == (evidence, k, mode)}


# ---------- features and estimates per k (9.1, 9.2) ----------

def test_every_k_up_to_the_kept_count_is_compared_and_the_rest_reported_missing(table):
    assert table.ks == CONVERGENCE_KS == (1, 5, 10, 20, 50, 100)
    assert table.kept == KEEP and table.n_vocab == N_VOCAB and table.served_k == ARM.top_logprobs
    for k in CONVERGENCE_KS:
        for feature in FEATURES:
            row = feature_row(table, HOST_TOP_K, k, feature)
            if k <= KEEP:
                assert row.missing_reason is None and row.n > 0
            else:
                assert row.n == 0 and row.mean_abs_diff is None and str(KEEP) in row.missing_reason
        if k > KEEP:
            assert str(KEEP) in mass_row(table, HOST_TOP_K, k).missing_reason
            assert all(str(KEEP) in decision_row(table, HOST_TOP_K, k, mode).missing_reason
                       for mode in ("ensemble", "single"))
    assert {row.feature for row in table.features if row.evidence == SERVED_TOP_K} == set(FEATURES)


def test_agreement_is_exact_at_the_kept_count_and_for_actual_token_features_at_every_k(table):
    for feature in FEATURES:
        row = feature_row(table, HOST_TOP_K, KEEP, feature)
        assert row.mean_abs_diff == 0.0 and row.max_abs_diff == 0.0 and row.share_within == 1.0
    for k in (k for k in CONVERGENCE_KS if k <= KEEP):
        for feature in ("loss", "min_k", "zlib_ratio"):  # the exact actual-token logprob, whatever k
            assert feature_row(table, HOST_TOP_K, k, feature).mean_abs_diff == 0.0
    assert feature_row(table, HOST_TOP_K, 1, "min_k_pp").mean_abs_diff > 0.0


@pytest.mark.parametrize("k", [1, 5, 10])
def test_feature_agreement_matches_a_brute_force_recomputation_on_the_same_responses(recorded, table, k):
    full_calibrator = recorded.calibrators["full_distribution"].calibrator
    differences: dict[str, list[float]] = {feature: [] for feature in FEATURES}
    for _, _, response, evidence in compared_draws(recorded):
        full = feature_values(brute_features(response.content, evidence, KEEP), full_calibrator)
        truncated = feature_values(brute_features(response.content, evidence, k), full_calibrator)
        for feature in FEATURES:
            if full[feature] is not None and truncated[feature] is not None:
                differences[feature].append(abs(truncated[feature] - full[feature]))
    for feature in FEATURES:
        row = feature_row(table, HOST_TOP_K, k, feature)
        values = differences[feature]
        assert row.n == len(values) > 0
        assert math.isclose(row.mean_abs_diff, sum(values) / len(values), rel_tol=1e-12, abs_tol=1e-15)
        assert row.max_abs_diff == max(values)
        assert row.share_within == sum(value <= 1e-6 for value in values) / len(values)


def test_the_served_row_uses_the_generation_hosts_own_top_k_through_the_top_k_calibrator(recorded, table):
    top_k_calibrator = recorded.calibrators["top_k"].calibrator
    full_calibrator = recorded.calibrators["full_distribution"].calibrator
    differences = []
    for result, index, response, evidence in compared_draws(recorded):
        served = top_k_calibrator.estimate(compute_mia_features(response.content, response.to_legacy().logprobs, None))
        assert served == result.recorded_draws[index].estimate  # the study's top_k estimate of that draw
        full = full_calibrator.estimate(brute_features(response.content, evidence, KEEP))
        differences.append(abs(served - full))
    row = feature_row(table, SERVED_TOP_K, ARM.top_logprobs, "estimate")
    assert row.n == len(differences)
    assert math.isclose(row.mean_abs_diff, sum(differences) / len(differences), rel_tol=1e-12)


def test_mass_coverage_grows_with_k_and_matches_the_kept_alternatives(recorded, table):
    draws = compared_draws(recorded)
    means = []
    for k in (k for k in CONVERGENCE_KS if k <= KEEP):
        row = mass_row(table, HOST_TOP_K, k)
        expected = [float(np.mean([sum(math.exp(value) for _, value in alternatives[:k])
                                   for alternatives in evidence.top_logprobs])) for *_, evidence in draws]
        assert row.n_draws == len(draws)
        assert math.isclose(row.mean_mass, sum(expected) / len(expected), rel_tol=1e-12)
        assert 0.0 < row.min_mass <= row.mean_mass <= 1.0 + 1e-12
        means.append(row.mean_mass)
    assert means == sorted(means)
    assert table.full_mean_mass == mass_row(table, HOST_TOP_K, KEEP).mean_mass
    served = [float(np.mean([sum(math.exp(alt.logprob) for alt in entry.top_logprobs)
                             for entry in response.token_evidence])) for _, _, response, _ in draws]
    assert math.isclose(mass_row(table, SERVED_TOP_K, ARM.top_logprobs).mean_mass, sum(served) / len(served),
                        rel_tol=1e-12)


# ---------- the downstream adjusted decision (9.3) ----------

def test_the_decision_does_not_change_at_the_kept_count(table, recorded):
    for mode in ("ensemble", "single"):
        row = decision_row(table, HOST_TOP_K, KEEP, mode)
        failed = sum(result.variant(f"{mode}_full_distribution").status == "failed" for result in recorded.results)
        assert row.n_items == len(recorded.results) - failed
        assert row.mean_abs_p_change == 0.0 and row.max_abs_p_change == 0.0
        assert row.mean_abs_adjusted_summary_change == 0.0
        assert row.share_category_changed == 0.0 and row.share_status_changed == 0.0


def test_an_item_without_a_decision_is_reported_missing_with_both_statuses(table, recorded):
    first = recorded.results[0]
    assert first.recorded_draws[0].parsed is None  # the prose reply
    assert first.variant("single_full_distribution").status == "failed"
    assert first.variant("ensemble_full_distribution").status != "failed"
    for evidence, k in [(HOST_TOP_K, k) for k in CONVERGENCE_KS if k <= KEEP] + [(SERVED_TOP_K, ARM.top_logprobs)]:
        change = item_changes(table, evidence, k, "single")[first.key]
        assert change.missing_reason == NO_DECISION
        assert (change.status_full, change.status, change.p_full, change.p) == ("failed", "failed", None, None)
        assert change.abs_p_change is None and change.status_changed is None
        assert decision_row(table, evidence, k, "single").missing == {NO_DECISION: 1}
        assert item_changes(table, evidence, k, "ensemble")[first.key].missing_reason is None


@pytest.mark.parametrize("mode", ["ensemble", "single"])
def test_the_full_and_served_decisions_are_the_stored_variants_rederived_from_the_recorded_draws(table, recorded,
                                                                                                mode):
    changes = item_changes(table, SERVED_TOP_K, ARM.top_logprobs, mode)
    assert set(changes) == {result.key for result in recorded.results}
    for result in recorded.results:
        change = changes[result.key]
        full, top_k = result.variant(f"{mode}_full_distribution"), result.variant(f"{mode}_top_k")
        assert (change.status_full, change.p_full) == (full.status, full.p_memorized)
        assert (change.status, change.p) == (top_k.status, top_k.p_memorized)
        assert change.adjusted_summary_full == full.adjusted_summary_value
        assert change.adjusted_summary == top_k.adjusted_summary_value
        if full.status == "failed":
            assert change.missing_reason == NO_DECISION and change.abs_p_change is None
        else:
            assert change.missing_reason is None
            assert change.abs_p_change == abs(top_k.p_memorized - full.p_memorized)


@pytest.mark.parametrize("k", [1, 5])
def test_the_decision_at_k_is_decide_from_draws_on_the_draws_with_their_top_k_estimates(table, recorded, k):
    full_calibrator = recorded.calibrators["full_distribution"].calibrator
    estimates = {(result.key, index): feature_values(brute_features(response.content, evidence, k),
                                                     full_calibrator)["estimate"]
                 for result, index, response, evidence in compared_draws(recorded)}
    changes = item_changes(table, HOST_TOP_K, k, "ensemble")
    p_changes = []
    for result in recorded.results:
        identity = result.identities["full_distribution"]
        draws = tuple(replace(draw, estimate=estimates[(result.key, draw.draw_index)]) if draw.parsed is not None
                      else draw for draw in result.draws_for("full_distribution"))
        decision = decide_from_draws(draws, GuardConfig(regime_definition, identity.source, draws=ARM.draws,
                                                        min_parsed=ARM.min_parsed), identity)
        change = changes[result.key]
        assert (change.status, change.p) == (decision.status, decision.p_memorized)
        assert change.adjusted_summary == regime_definition.summary(dict(decision.adjusted))
        assert change.p_full == result.variant("ensemble_full_distribution").p_memorized
        p_changes.append(abs(decision.p_memorized - change.p_full))
    single = item_changes(table, HOST_TOP_K, k, "single")
    for result in recorded.results:
        identity = replace(result.identities["full_distribution"], draws=1)
        first = result.draws_for("full_distribution")[0]
        if first.parsed is not None:
            first = replace(first, estimate=estimates[(result.key, 0)])
        decision = decide_from_draws((first,), GuardConfig(regime_definition, identity.source, draws=1, min_parsed=1,
                                                           mode=DrawMode.SINGLE), identity)
        assert single[result.key].status == decision.status and single[result.key].p == decision.p_memorized
    row = decision_row(table, HOST_TOP_K, k, "ensemble")
    assert math.isclose(row.mean_abs_p_change, sum(p_changes) / len(p_changes), rel_tol=1e-12, abs_tol=1e-15)
    assert row.max_abs_p_change == max(p_changes)


# ---------- one model and one set of weights (9.4) ----------

def test_the_table_states_that_values_compare_only_within_one_model_and_set_of_weights(table):
    assert table.model == "gemma3:4b" and table.model_digest == DIGEST
    assert "only within one model and one set of weights" in table.comparability
    assert DIGEST in table.comparability and "not comparable" in table.comparability
    assert "top-20" in table.scope and "identical by construction" in table.scope
    rendered = table.to_markdown()
    for text in (table.comparability, table.scope, table.estimator):
        assert text in rendered
    payload = table.to_dict()
    assert payload["comparability"] == table.comparability and payload["model_digest"] == DIGEST


def test_evidence_from_other_weights_is_refused(recorded):
    with pytest.raises(ConvergenceError, match="weights"):
        build(recorded, expected_digest="sha256:" + "0" * 64)
    # One draw re-recorded with evidence from other weights: two digests in one table are refused.
    result = recorded.results[0]
    index = next(draw.draw_index for draw in result.recorded_draws if draw.parsed is not None)
    (draw,) = [draw for draw in recorded.draws if draw.key == (result.item_id, result.rendering, index)]
    original = {evidence_ref(entry): entry for entry in recorded.evidence}[result.full_distribution[index].evidence_ref]
    other = replace(original, model_digest="sha256:" + "1" * 64)
    estimate = full_distribution_estimate(draw.response.content, other, recorded.calibrators["full_distribution"])
    full = list(result.full_distribution)
    full[index] = estimate
    results = (replace(result, full_distribution=tuple(full)), *recorded.results[1:])
    draws = tuple(replace(entry, full_distribution=estimate) if entry.key == draw.key else entry
                  for entry in recorded.draws)
    with pytest.raises(ConvergenceError, match="one model and one set of weights"):
        build(recorded, results=results, draws=draws, evidence=(*recorded.evidence, other), expected_digest=None)


# ---------- sizes from the plan and the records; missing, never substituted (8.10) ----------

def test_sizes_come_from_the_plan_and_the_recorded_results(table, recorded):
    parsed = [draw for result in recorded.results for draw in result.recorded_draws if draw.parsed is not None]
    assert len(parsed) == len(recorded.results) * ARM.draws - 1  # the one prose reply is not parsed
    assert table.n_draws == len(parsed) and feature_row(table, HOST_TOP_K, 5, "loss").n == len(parsed)
    assert table.items == tuple(result.key for result in recorded.results)
    planned = planned_keys(PLAN, "sampling", "full_run")
    executed = {result.key for result in recorded.results}
    missing = {(entry["item_id"], entry["rendering"]) for entry in table.missing_items}
    assert missing == set(planned) - executed and len(missing) == len(planned) - len(executed)
    assert all("no stored result" in entry["reason"] for entry in table.missing_items)
    assert table.missing_draws == ()  # an unparsed draw has no evidence by design: excluded, not missing


def test_a_draw_without_its_evidence_is_reported_missing_and_its_item_is_not_compared(recorded, table):
    result = recorded.results[2]
    index = next(draw.draw_index for draw in result.recorded_draws if draw.parsed is not None)
    dropped = result.full_distribution[index].evidence_ref
    reduced = build(recorded, evidence=tuple(entry for entry in recorded.evidence if evidence_ref(entry) != dropped))
    assert [(entry["item_id"], entry["rendering"], entry["draw_index"]) for entry in reduced.missing_draws] == [
        (result.item_id, result.rendering, index)]
    assert dropped in reduced.missing_draws[0]["reason"]
    assert reduced.n_draws == table.n_draws - 1
    assert feature_row(reduced, HOST_TOP_K, 5, "min_k_pp").n == feature_row(table, HOST_TOP_K, 5, "min_k_pp").n - 1
    for mode in ("ensemble", "single"):
        changes = item_changes(reduced, HOST_TOP_K, 5, mode)
        assert changes[result.key].missing_reason == DRAWS_LACKING and changes[result.key].p is None
        row, before = decision_row(reduced, HOST_TOP_K, 5, mode), decision_row(table, HOST_TOP_K, 5, mode)
        assert row.n_items == before.n_items - 1
        assert row.missing == {**before.missing, DRAWS_LACKING: 1}
        # The served comparison needs no full-distribution evidence, so it still compares this item.
        assert item_changes(reduced, SERVED_TOP_K, ARM.top_logprobs, mode)[result.key] == \
            item_changes(table, SERVED_TOP_K, ARM.top_logprobs, mode)[result.key]


def test_a_recorded_estimate_that_does_not_reproduce_from_its_evidence_is_refused(recorded):
    result = recorded.results[0]
    index = next(draw.draw_index for draw in result.recorded_draws if draw.parsed is not None)
    entry = result.full_distribution[index]
    full = list(result.full_distribution)
    full[index] = SourceEstimate(0.5 if entry.estimate != 0.5 else 0.25, entry.evidence_ref,
                                 entry.calibrator_fingerprint)
    results = (replace(result, full_distribution=tuple(full)), *recorded.results[1:])
    draws = tuple(replace(draw, full_distribution=full[index]) if draw.key == (result.item_id, result.rendering, index)
                  else draw for draw in recorded.draws)
    with pytest.raises(ConvergenceError, match="does not reproduce"):
        build(recorded, results=results, draws=draws)


def test_a_draw_recording_that_differs_from_the_stored_result_is_refused(recorded):
    at = next(position for position, draw in enumerate(recorded.draws) if draw.draw.parsed is not None)
    first = recorded.draws[at]
    changed = replace(first, draw=replace(first.draw, estimate=0.123456))
    with pytest.raises(ConvergenceError, match="draw recording"):
        build(recorded, draws=(*recorded.draws[:at], changed, *recorded.draws[at + 1:]))


def test_the_analysis_is_refused_where_it_is_not_defined(recorded):
    with pytest.raises(ConvergenceError, match="full_distribution"):
        build(recorded, arm="serving")
    with pytest.raises(ConvergenceError, match="pilot"):
        build(recorded, stage="pilot")
    with pytest.raises(PlanChangedError):
        build(recorded, plan=frozen_plan(seed=SEED + 1))
    edgar = load_calibrators(PLAN, arm="sampling", kind="edgar", calibrators_dir=recorded.calibrators_dir)
    with pytest.raises(ConvergenceError, match="calibrator"):
        build(recorded, calibrators=edgar)
    for ks in ((), (5, 5), (10, 5), (0, 5), (1.5,)):
        with pytest.raises(ConvergenceError, match="ks"):
            build(recorded, ks=ks)


# ---------- from the recorded files, without requests (10.7) ----------

def test_the_table_is_produced_from_the_recorded_files_alone(recorded, table):
    from_files = convergence_from_records(PLAN, arm="sampling", stage="full_run", results_dir=recorded.results_dir,
                                          records_dir=recorded.records_dir, calibrators_dir=recorded.calibrators_dir)
    assert from_files.to_dict() == table.to_dict()


def records_copy(recorded, tmp_path, ending: bytes | None) -> Path:
    """A copy of the recorded files whose draw log ends without its newline, or in ``ending`` (a cut line)."""
    records_dir = tmp_path / "records"
    shutil.copytree(recorded.records_dir, records_dir)
    path = draw_log_path("sampling", "full_run", records_dir)
    data = path.read_bytes()
    path.write_bytes(data.rstrip(b"\n") if ending is None else data + ending)
    assert not path.read_bytes().endswith(b"\n")
    return path


def from_copy(recorded, tmp_path):
    return convergence_from_records(PLAN, arm="sampling", stage="full_run", results_dir=recorded.results_dir,
                                    records_dir=tmp_path / "records", calibrators_dir=recorded.calibrators_dir)


def test_reading_the_records_never_modifies_the_draw_log(recorded, table, tmp_path):
    """A complete last line without its newline is used as it is; the log keeps its bytes and its time."""
    path = records_copy(recorded, tmp_path, None)
    files = sorted(found for found in tmp_path.rglob("*") if found.is_file())
    before = aged(path)

    from_files = from_copy(recorded, tmp_path)

    assert unchanged(path, before)
    assert sorted(found for found in tmp_path.rglob("*") if found.is_file()) == files
    assert from_files.to_dict() == table.to_dict() and from_files.to_markdown() == table.to_markdown()


def test_a_draw_log_that_ends_in_an_incomplete_line_is_refused_and_left_as_it_is(recorded, tmp_path):
    """The outputs carry no note of an ignored line, so the analysis refuses instead of computing without it."""
    path = records_copy(recorded, tmp_path, CUT)
    before = aged(path)

    with pytest.raises(ConvergenceError, match="incomplete last line") as refused:
        from_copy(recorded, tmp_path)

    assert str(path) in str(refused.value)
    assert unchanged(path, before)


def test_the_analysis_holds_no_write_other_than_its_own_outputs():
    source = inspect.getsource(convergence_module)
    assert "DrawLog" not in source and ".latest(" not in source and "_repaired_lines" not in source
    assert not hasattr(convergence_module, "DrawLog")
    assert writing_calls(convergence_module) == {
        ("_write_text", "mkdir"), ("_write_text", "open"), ("_write_text", "os.replace"),
        ("write_table", "_write_text")}


def test_the_written_table_is_reproduced_byte_for_byte_and_never_replaced_by_a_different_one(recorded, table,
                                                                                            tmp_path):
    json_path, markdown_path = write_table(table, tmp_path)
    assert json_path == table_path("sampling", "full_run", tmp_path)
    assert json.loads(json_path.read_text(encoding="utf-8")) == table.to_dict()
    assert markdown_path.read_text(encoding="utf-8") == table.to_markdown()
    first = (json_path.read_bytes(), markdown_path.read_bytes())
    again = convergence_from_records(PLAN, arm="sampling", stage="full_run", results_dir=recorded.results_dir,
                                     records_dir=recorded.records_dir, calibrators_dir=recorded.calibrators_dir)
    assert write_table(again, tmp_path) == (json_path, markdown_path)
    assert (json_path.read_bytes(), markdown_path.read_bytes()) == first
    with pytest.raises(ConvergenceError, match="refus"):
        write_table(build(recorded, ks=(1, 5)), tmp_path)
    assert (json_path.read_bytes(), markdown_path.read_bytes()) == first


def test_the_table_carries_derived_statistics_only_never_a_reply(recorded, table):
    text = json.dumps(table.to_dict()) + table.to_markdown()
    for draw in recorded.draws:
        if draw.response is not None:
            assert draw.response.content not in text


def test_the_command_writes_the_table_of_the_frozen_plan_and_refuses_another_plan(recorded, table, tmp_path):
    plan_file = write_plan(PLAN, tmp_path / "plan.json")
    out = tmp_path / "analysis"
    common = ["--plan", str(plan_file), "--arm", "sampling", "--stage", "full_run",
              "--results-dir", str(recorded.results_dir), "--records-dir", str(recorded.records_dir),
              "--calibrators-dir", str(recorded.calibrators_dir), "--out-dir", str(out)]
    assert main([*common, "--plan-sha256", "0" * 64]) == 2
    assert not out.exists()
    assert main([*common, "--plan-sha256", PLAN.plan_sha256]) == 0
    written = json.loads(table_path("sampling", "full_run", out).read_text(encoding="utf-8"))
    assert written == table.to_dict()
