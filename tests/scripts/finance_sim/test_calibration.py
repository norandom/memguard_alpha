"""Calibration on the held-out macro months and on the EDGAR calibration split (task 7.6).

Requirements 4.1, 4.3 and 6.3. Pure logic on fakes: a generation-host double
that answers through ``RequestControl`` with the stage's ``LedgerCallBudget``
(so every request is admitted by the 7.5 ledger), a full-distribution host
double, and files under ``tmp_path`` only. No network, credentials or model
host is used, and nothing is written under ``runs/``.
"""

from __future__ import annotations

import calendar
import hashlib
import json
import math
import random
import socket
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from recall_guard.core.completion import ModelConfig, ModelResponse, TokenAlternative, TokenEvidence
from recall_guard.core.full_distribution import FullDistributionEvidence
from recall_guard.core.loader import EvalRow
from recall_guard.core.nvidia_lm import CompletionResult
from recall_guard.core.request_control import CallBudget, RequestControl, RequestError
from recall_guard.dataset.edgar_outlook import outlook_definition, render_outlook_prompt
from recall_guard.dataset.macro_regime import MacroRow, regime_definition, render_anonymised, render_identifying
from recall_guard.harness.guarded_decision import GuardConfig, guarded_run
from recall_guard.harness.scorer_client import CalibrationMismatchError
from recall_guard.mia import mcs as mcs_module
from recall_guard.mia.control import build_baseline, standardise
from recall_guard.mia.estimate_source import EstimateSource, SourceLabel, features_from_full
from scripts.finance_sim.calibrate import (
    ANONYMISED,
    CALIBRATION_STAGE,
    EDGAR,
    FEATURE_ORDER,
    HOST_BUSY_RETRIES,
    HOST_BUSY_WAIT_S,
    IDENTIFYING,
    INTERRUPTED,
    MACRO,
    OUTLOOK,
    CalibrationDraw,
    CalibrationError,
    DrawSettings,
    EdgarRelease,
    EvidenceSettings,
    EvidenceUnavailableError,
    FitSettings,
    calibration_identity,
    calibrator_path,
    collect_draws,
    collect_evidence,
    edgar_corpus,
    estimate_sources,
    fit_calibrator,
    full_distribution_row,
    macro_corpus,
    plan_corpus_layout,
    plan_draw_settings,
    plan_evidence_settings,
    plan_fit_settings,
    request_allowance,
    run_calibration,
    split_seed,
)
from scripts.finance_sim.limits import (
    AdmissionClosedError,
    BudgetLedger,
    LedgerCallBudget,
    RequestAllowance,
    StudyLimits,
)
from scripts.finance_sim.plan import (
    EDGAR_FILES,
    EdgarRow,
    FitPlan,
    MonthClass,
    PlanChangedError,
    StudyPlan,
    freeze_plan,
    load_serving_manifest,
    study_arms,
)

GEMMA = "google/gemma-3-4b-it"
CREATED_AT = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)
SEED = 20261110
DIGEST = "sha256:" + "f" * 64
#: Two draws per prompt keep the fake runs small: (24 x 2 + 140) prompts x 2 draws = 376 requests.
SETTINGS = DrawSettings(draws=2, temperature=0.7, seed_base=100)
FIT = FitSettings(min_auc=0.6, min_valid=10)
PARAMETERS = {"temperature": 0.0, "top_p": 1, "max_tokens": 64, "logprobs": True, "top_logprobs": 3}
ALLOWANCE = RequestAllowance(output_tokens=64, timeout_s=30)
COMPLETION_TOKENS = 7
MACRO_REPLY = json.dumps({"inflation": 0.3, "growth": -0.2, "credit_stress": 0.1, "policy": 0.0,
                          "risk_appetite": -0.4})
OUTLOOK_REPLY = json.dumps({"outlook": 0.2})


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_sha256(value) -> str:
    return digest(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")))


def month_range(first: str, last: str) -> list[str]:
    year, month = int(first[:4]), int(first[5:])
    out = []
    while f"{year:04d}-{month:02d}" <= last:
        out.append(f"{year:04d}-{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return out


MONTHS = month_range("2016-01", "2026-06")


def frozen_plan(seed: int = SEED, **sections) -> StudyPlan:
    months = [MonthClass(month, {GEMMA: month <= "2024-08"}) for month in MONTHS]
    rows = [EdgarRow(f"cal-mem-{i:03d}", 1, "calibration") for i in range(40)]
    rows += [EdgarRow(f"cal-ctl-{i:03d}", 0, "calibration") for i in range(100)]
    rows += [EdgarRow(f"held-ctl-{i:03d}", 0, "held_out") for i in range(40)]
    rows += [EdgarRow(f"held-mem-{i:03d}", 1, "held_out") for i in range(40)]
    items = {f"macro-{month}": digest(f"macro-{month}") for month in MONTHS}
    items |= {f"edgar-{row.row_id}": digest(row.row_id) for row in rows}
    return freeze_plan(months, rows, seed=seed, created_at=CREATED_AT, items=items,
                       edgar_files={name: digest(name) for name in EDGAR_FILES}, **sections)


#: The small frozen calibration of the complete plan below: 2 draws per macro rendering, 1 per EDGAR row.
COMPLETE_DRAWS = {MACRO: 2, EDGAR: 1}
COMPLETE_PLANNED = 2 * 24 * COMPLETE_DRAWS[MACRO] + 140 * COMPLETE_DRAWS[EDGAR]


def complete_plan(seed: int = SEED) -> StudyPlan:
    """The plan with its 7.3 sections, the sampling arm's calibration made small for the fakes."""
    arms = study_arms()
    sampling = replace(arms.arms["sampling"], calibration_draws=COMPLETE_DRAWS, fit=FitPlan(min_auc=0.6, min_valid=10))
    return frozen_plan(seed, arms=replace(arms, arms={**arms.arms, "sampling": sampling}),
                       serving_manifest=load_serving_manifest())


PLAN = frozen_plan()


def macro_row(month: str) -> MacroRow:
    year, number = int(month[:4]), int(month[5:])
    k = MONTHS.index(month)
    return MacroRow(decision_date=date(year, number, calendar.monthrange(year, number)[1]),
                    raw={"baa10y": 2.0 + 0.01 * k, "cpi_yoy": 1.5 + 0.02 * k, "t10y2y": 0.5 - 0.01 * k},
                    zscores={"baa10y_z": 0.1 * (k % 7) - 0.3, "cpi_yoy_z": 0.05 * (k % 11) - 0.2,
                             "t10y2y_z": -0.04 * (k % 5) + 0.1},
                    revised={"BAA10Y": False, "CPIAUCSL": False, "T10Y2Y": False})


def macro_rows(plan: StudyPlan = PLAN, months=None) -> list[MacroRow]:
    return [macro_row(month) for month in (plan.months.calibration if months is None else months)]


def release_text(row_id: str, label: int) -> str:
    marker = " (recalled)" if label == 1 else ""
    return f"Company {row_id} reports quarterly results{marker}. Revenue rose and the outlook is unchanged."


def edgar_releases(plan: StudyPlan = PLAN) -> list[EdgarRelease]:
    releases = [EdgarRelease(row_id, 1, release_text(row_id, 1)) for row_id in plan.edgar.calibration_memorized]
    releases += [EdgarRelease(row_id, 0, release_text(row_id, 0)) for row_id in plan.edgar.calibration_control]
    return releases


def memorized(prompt: str) -> bool:
    """The fakes' ground truth: identifying macro prompts and memorized EDGAR releases."""
    return "Real asset identities:" in prompt or "(recalled)" in prompt


class Clock:
    """A fake monotonic clock in seconds."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def model_config(**overrides) -> ModelConfig:
    return ModelConfig(protocol="callback", endpoint=None, model="gemma3:4b", parameters={**PARAMETERS, **overrides},
                       timeout_s=30, max_retries=0, retry_backoff_s=0)


def study_limits(plan: StudyPlan = PLAN, **overrides) -> StudyLimits:
    values = {"plan_sha256": plan.plan_sha256, "arm": "sampling", "stage": CALIBRATION_STAGE, "max_requests": 1000,
              "max_output_tokens": 1_000_000, "max_seconds": 1_000_000} | overrides
    return StudyLimits(**values)


def open_ledger(path: Path, *, plan: StudyPlan = PLAN, limits: StudyLimits | None = None,
                allowance: RequestAllowance = ALLOWANCE) -> BudgetLedger:
    return BudgetLedger.open(path, plan, limits or study_limits(plan), allowance, clock=Clock())


def token_evidence(base: float) -> tuple[TokenEvidence, ...]:
    """Six realized tokens with three alternatives each, as a top-k host returns them."""
    values = [min(0.0, base - 0.02 * i) for i in range(6)]
    return tuple(TokenEvidence("t", value, (TokenAlternative("t", value), TokenAlternative("u", value - 1.0),
                                            TokenAlternative("v", value - 2.0))) for value in values)


def base_logprob(prompt: str, seed: object, *, signal: bool) -> float:
    """Memorized prompts are likelier; noise depends on the prompt and the draw's seed."""
    if not signal:
        return -0.6
    noise = random.Random(f"{digest(prompt)}:{seed}").uniform(-0.3, 0.3)
    return (-0.15 if memorized(prompt) else -1.2) + noise


class FakeHost:
    """Generation-host double; every reply runs through RequestControl with the client's budget.

    ``prose`` lists call indexes answered with unparseable text and ``failures``
    maps call indexes to the error the host raises instead of a reply. Replies
    are kept by prompt and seed for the replay comparisons.
    """

    def __init__(self, budget: CallBudget, *, configuration: ModelConfig | None = None, signal: bool = True,
                 prose=(), failures=None, events: list | None = None, crash_at: int | None = None) -> None:
        self.config = configuration or model_config()
        self.budget = budget
        self._control = RequestControl(self.config, budget, clock=Clock())
        self.signal = signal
        self.prose = set(prose)
        self.failures = dict(failures or {})
        self.events = events
        self.crash_at = crash_at
        self.calls: list[tuple[str, dict]] = []
        self.replies: dict[tuple[str, object], ModelResponse] = {}

    def generate(self, prompt: str, **parameters) -> ModelResponse:
        index = len(self.calls)
        self.calls.append((prompt, dict(parameters)))
        if self.events is not None:
            self.events.append("generate")

        def operation() -> ModelResponse:
            if index == self.crash_at:
                raise ProcessDied  # after on_request and after the ledger admitted the request
            if index in self.failures:
                raise self.failures[index]
            content = OUTLOOK_REPLY if "Press release:" in prompt else MACRO_REPLY
            if index in self.prose:
                content = "The regime looks benign to me."
            seed = parameters.get("seed")
            response = ModelResponse(content, token_evidence(base_logprob(prompt, seed, signal=self.signal)),
                                     usage={"completion_tokens": COMPLETION_TOKENS},
                                     requested_settings=dict(self.config.parameters) | parameters)
            self.replies[(prompt, seed)] = response
            return response

        return self._control.run(operation, stage=CALIBRATION_STAGE)


def fake_evidence(prompt: str, response_text: str, position: int, *, signal: bool = True) -> FullDistributionEvidence:
    """The fake host's evidence for the reply at ``position`` of a batch: four kept alternatives per position."""
    base = base_logprob(prompt, f"full:{position}:{digest(response_text)}", signal=signal)
    positions = range(6)
    values = [min(0.0, base - 0.01 * i) for i in positions]
    return FullDistributionEvidence(
        model_digest=DIGEST, n_vocab=50, response_token_ids=tuple(positions), logprob=tuple(values),
        entropy=(1.0,) * 6, rank=(1,) * 6,
        top_logprobs=tuple(((i, value), (40, value - 1.0), (41, value - 2.0), (42, value - 3.0))
                           for i, value in zip(positions, values, strict=True)),
    )


def host_busy() -> RequestError:
    return RequestError("HTTP 503: full-distribution request rejected (host_busy)", status_code=503, code="host_busy",
                        retryable=True)


class FakeFullDistribution:
    """Full-distribution host double that answers batches only, as the calibration must ask.

    ``busy`` is how many requests are refused with ``host_busy`` before the host
    answers; ``failure`` is raised on every request instead (a dead host).
    """

    def __init__(self, *, signal: bool = True, busy: int = 0, failure: Exception | None = None,
                 events: list | None = None) -> None:
        self.signal = signal
        self.busy = busy
        self.failure = failure
        self.events = events
        self.batch_calls: list[tuple[str, tuple[str, ...], int, float | None]] = []

    def evidence(self, prompt_text: str, response_text: str, *, top_k_keep: int = 100) -> FullDistributionEvidence:
        pytest.fail("calibration requested evidence per draw; it must send one batch per prompt and rendering")

    def evidence_batch(self, prompt_text: str, response_texts, *, top_k_keep: int = 100,
                       timeout_s: float | None = None) -> tuple[FullDistributionEvidence, ...]:
        self.batch_calls.append((prompt_text, tuple(response_texts), top_k_keep, timeout_s))
        if self.events is not None:
            self.events.append("evidence")
        if self.failure is not None:
            raise self.failure
        if self.busy:
            self.busy -= 1
            raise host_busy()
        return tuple(fake_evidence(prompt_text, text, position, signal=self.signal)
                     for position, text in enumerate(response_texts))


class Interrupted(Exception):
    """A simulated interruption of the runner after a draw was recorded."""


class ProcessDied(BaseException):
    """A simulated death of the runner process while a request is in flight (nothing catches it)."""


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("the calibration attempted a network connection")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)


def calibrate(root: Path, *, plan: StudyPlan = PLAN, signal: bool = True, settings: DrawSettings = SETTINGS,
              sources=("top_k", "full_distribution"), limits: StudyLimits | None = None, recorded=(),
              on_draw=None, full: FakeFullDistribution | None = None, draws: list | None = None,
              sleeps: list | None = None, events: list | None = None) -> SimpleNamespace:
    """One calibration run of the sampling arm on fakes, with its ledger, host doubles and recorded draws.

    ``draws`` (the recording, one entry per ``on_draw`` call) and ``sleeps`` (the
    waits between busy retries; nothing really sleeps) may be passed in so a test
    can inspect them after the run raised.
    """
    ledger = open_ledger(root / "budget.json", plan=plan, limits=limits)
    host = FakeHost(LedgerCallBudget(ledger), signal=signal, events=events)
    full = FakeFullDistribution(signal=signal, events=events) if full is None else full
    draws = [] if draws is None else draws
    sleeps = [] if sleeps is None else sleeps

    def record(draw) -> None:
        draws.append(draw)
        if on_draw is not None:
            on_draw(draw)

    outcome = SimpleNamespace(ledger=ledger, host=host, full=full, draws=draws, sleeps=sleeps, root=root, run=None)
    outcome.run = run_calibration(plan, arm="sampling", macro_rows=macro_rows(plan), edgar_releases=edgar_releases(plan),
                                  client=host, ledger=ledger, settings=settings, sources=sources,
                                  full_distribution=full, fit=FIT, recorded=recorded, on_draw=record,
                                  calibrators_dir=root / "calibrators", sleep=sleeps.append)
    return outcome


def latest(draws) -> dict:
    """The recording as a store keeps it: the latest version of each draw."""
    return {draw.key: draw for draw in draws}


@pytest.fixture(scope="module")
def calibrated(tmp_path_factory) -> SimpleNamespace:
    return calibrate(tmp_path_factory.mktemp("calibration"), events=[])


def expected_corpus_hash(plan: StudyPlan, kind: str) -> str:
    """The corpus hash recomputed independently from the frozen plan and the renderers."""
    if kind == MACRO:
        rows = {f"{row.decision_date:%Y-%m}": row for row in macro_rows(plan)}
        pairs = [[digest(render_identifying(rows[month])), 1] for month in plan.months.calibration]
        pairs += [[digest(render_anonymised(rows[month])), 0] for month in plan.months.calibration]
    else:
        pairs = [[digest(render_outlook_prompt(release_text(row_id, 1))), 1] for row_id in plan.edgar.calibration_memorized]
        pairs += [[digest(render_outlook_prompt(release_text(row_id, 0))), 0] for row_id in plan.edgar.calibration_control]
    return canonical_sha256(pairs)


# ---------- the macro calibration set (4.1, 6.3) ----------

def test_macro_corpus_renders_every_frozen_calibration_month_identifying_as_memorized_and_anonymised_as_control():
    corpus = macro_corpus(PLAN, macro_rows())
    months = PLAN.months.calibration
    assert len(months) == 24 and corpus.kind == MACRO and corpus.plan_sha256 == PLAN.plan_sha256
    rows = {f"{row.decision_date:%Y-%m}": row for row in macro_rows()}
    expected = [(f"macro-{month}", IDENTIFYING, 1, render_identifying(rows[month])) for month in months]
    expected += [(f"macro-{month}", ANONYMISED, 0, render_anonymised(rows[month])) for month in months]
    assert [(p.item_id, p.rendering, p.label, p.prompt) for p in corpus.prompts] == expected
    assert corpus.definition is regime_definition
    assert plan_corpus_layout(PLAN, MACRO) == tuple(item[:3] for item in expected)


@pytest.mark.parametrize("split", ["evaluation", "pilot"])
def test_macro_corpus_refuses_calibration_months_that_overlap_evaluation_or_pilot(split):
    intruder = getattr(PLAN.months, split)[0]
    rows = macro_rows() + [macro_row(intruder)]
    with pytest.raises(CalibrationError, match=f"overlap the {split} months.*{intruder}"):
        macro_corpus(PLAN, rows)
    swapped = [macro_row(intruder)] + macro_rows()[1:]
    with pytest.raises(CalibrationError, match=split):
        macro_corpus(PLAN, swapped)


def test_macro_corpus_refuses_a_missing_or_repeated_calibration_month_and_a_non_row():
    with pytest.raises(CalibrationError, match="missing.*" + PLAN.months.calibration[0]):
        macro_corpus(PLAN, macro_rows()[1:])
    with pytest.raises(CalibrationError, match="more than once"):
        macro_corpus(PLAN, macro_rows() + macro_rows()[:1])
    with pytest.raises(TypeError):
        macro_corpus(PLAN, macro_rows()[:-1] + ["2016-03"])


# ---------- the EDGAR calibration split (4.1) ----------

def test_edgar_corpus_holds_the_frozen_calibration_rows_in_the_outlook_prompt_format():
    corpus = edgar_corpus(PLAN, edgar_releases())
    memorized_ids, control_ids = PLAN.edgar.calibration_memorized, PLAN.edgar.calibration_control
    assert (len(memorized_ids), len(control_ids)) == (40, 100)
    expected = [(row_id, OUTLOOK, 1, render_outlook_prompt(release_text(row_id, 1))) for row_id in memorized_ids]
    expected += [(row_id, OUTLOOK, 0, render_outlook_prompt(release_text(row_id, 0))) for row_id in control_ids]
    assert [(p.item_id, p.rendering, p.label, p.prompt) for p in corpus.prompts] == expected
    assert corpus.kind == EDGAR and corpus.definition is outlook_definition
    assert plan_corpus_layout(PLAN, EDGAR) == tuple(item[:3] for item in expected)


@pytest.mark.parametrize(("group", "message"), [("negative_control", "negative-control items"),
                                                ("unused", "unused held-out rows")])
def test_edgar_corpus_refuses_rows_that_overlap_the_negative_control_items(group, message):
    intruder = getattr(PLAN.edgar, group)[0]
    releases = edgar_releases() + [EdgarRelease(intruder, 0, release_text(intruder, 0))]
    with pytest.raises(CalibrationError, match=f"{message}.*{intruder}"):
        edgar_corpus(PLAN, releases)


def test_edgar_corpus_refuses_missing_repeated_relabelled_or_unplanned_rows():
    releases = edgar_releases()
    with pytest.raises(CalibrationError, match="missing"):
        edgar_corpus(PLAN, releases[1:])
    with pytest.raises(CalibrationError, match="more than once"):
        edgar_corpus(PLAN, releases + releases[:1])
    relabelled = [replace(releases[0], label=0)] + releases[1:]
    with pytest.raises(CalibrationError, match="label"):
        edgar_corpus(PLAN, relabelled)
    with pytest.raises(CalibrationError, match="not frozen calibration rows"):
        edgar_corpus(PLAN, releases + [EdgarRelease("cal-mem-999", 1, "Another release.")])


def test_a_plan_whose_frozen_sets_overlap_is_refused_as_defence_in_depth():
    # StudyPlan refuses overlapping sets when it is built; this bypasses that check to
    # show the calibration does not rely on it alone.
    for section, field, value in (
            ("months", "evaluation", PLAN.months.evaluation + PLAN.months.calibration[:1]),
            ("edgar", "negative_control", PLAN.edgar.negative_control + PLAN.edgar.calibration_control[:1])):
        plan = frozen_plan()
        object.__setattr__(getattr(plan, section), field, value)
        with pytest.raises(CalibrationError, match="overlap"):
            plan_corpus_layout(plan, MACRO if section == "months" else EDGAR)


def test_corpus_hash_is_the_sha256_of_the_ordered_prompt_hash_and_label_list_and_follows_the_plan():
    for kind, corpus in ((MACRO, macro_corpus(PLAN, macro_rows())), (EDGAR, edgar_corpus(PLAN, edgar_releases()))):
        assert corpus.corpus_sha256 == expected_corpus_hash(PLAN, kind)
        assert [entry.prompt_sha256 for entry in corpus.entries] == [digest(p.prompt) for p in corpus.prompts]
    other = frozen_plan(seed=SEED + 1)
    assert other.months.calibration != PLAN.months.calibration
    assert macro_corpus(other, macro_rows(other)).corpus_sha256 == expected_corpus_hash(other, MACRO)
    assert expected_corpus_hash(other, MACRO) != expected_corpus_hash(PLAN, MACRO)


# ---------- calibration records bound to the frozen plan (4.1, 4.3) ----------

def test_calibration_fits_one_calibrator_per_kind_and_source_whose_corpus_hashes_match_the_frozen_plan(calibrated):
    run = calibrated.run
    assert set(run.records) == {(kind, source) for kind in (MACRO, EDGAR) for source in ("top_k", "full_distribution")}
    for (kind, source), record in run.records.items():
        identity = record.calibrator.identity
        assert (record.arm, record.kind, record.source.value) == ("sampling", kind, source)
        assert record.plan_sha256 == PLAN.plan_sha256
        assert record.corpus_sha256 == identity.corpus_sha256 == expected_corpus_hash(PLAN, kind)
        assert identity.estimate_source == source
        assert tuple((e.item_id, e.rendering, e.label) for e in record.entries) == plan_corpus_layout(PLAN, kind)
        assert json.loads(identity.parameters_json)["temperature"] == 0.7
        assert "seed" not in json.loads(identity.parameters_json)
        assert identity.model == "gemma3:4b"
        assert record.calibrator.is_weak is False and record.calibrator.mcs.holdout_auc == 1.0
        assert run.paths[(kind, source)] == calibrator_path("sampling", source, kind, calibrated.root / "calibrators")
        assert run.paths[(kind, source)].is_file()
    assert run.records[(MACRO, "top_k")].corpus_sha256 != run.records[(EDGAR, "top_k")].corpus_sha256


def test_one_recorded_ensemble_per_rendering_serves_both_sources_and_every_request_goes_through_the_ledger(calibrated):
    planned = (2 * 24 + 140) * SETTINGS.draws
    assert len(calibrated.host.calls) == planned
    assert calibrated.ledger.counters["requests"] == planned
    assert calibrated.ledger.counters["output_tokens"] == planned * COMPLETION_TOKENS
    assert calibrated.ledger.status == "complete"
    # Every draw is recorded when it is made and again when its evidence settles.
    assert len(calibrated.draws) == 2 * planned
    stored = latest(calibrated.draws)
    assert len(stored) == planned
    for kind in (MACRO, EDGAR):
        assert [stored[draw.key] for draw in calibrated.run.draws[kind]] == list(calibrated.run.draws[kind])
        for draw in calibrated.run.draws[kind]:
            assert set(draw.features) == {"top_k", "full_distribution"} and not draw.evidence_pending
            assert all(set(values) == set(FEATURE_ORDER) for values in draw.features.values())


def test_full_distribution_evidence_comes_after_generation_in_one_batch_per_prompt_and_rendering(calibrated):
    events = calibrated.host.events
    assert events.index("evidence") == len(calibrated.host.calls)  # no evidence before the last generation
    assert set(events[events.index("evidence"):]) == {"evidence"}
    corpora = (macro_corpus(PLAN, macro_rows()), edgar_corpus(PLAN, edgar_releases()))
    prompts = [prompt for corpus in corpora for prompt in corpus.prompts]
    assert [call[0] for call in calibrated.full.batch_calls] == [prompt.prompt for prompt in prompts]
    by_prompt = {}
    for kind in (MACRO, EDGAR):
        for draw in calibrated.run.draws[kind]:
            by_prompt.setdefault(draw.prompt_sha256, []).append(draw)
    for prompt_text, texts, top_k_keep, timeout_s in calibrated.full.batch_calls:
        assert (top_k_keep, timeout_s) == (100, 600.0)
        assert list(texts) == [d.response_text for d in sorted(by_prompt[digest(prompt_text)],
                                                              key=lambda d: d.draw_index)]


def test_calibrators_report_held_out_discrimination_with_draw_level_and_prompt_clustered_intervals(calibrated):
    first = calibrated.run.records[(MACRO, "top_k")]
    assert first.holdout_auc_interval == first.calibrator.holdout_auc_interval()
    for record in calibrated.run.records.values():
        interval = record.holdout_auc_interval
        assert interval["method"] == "stratified_percentile_bootstrap" and interval["auc"] == 1.0
        clustered = record.holdout_auc_interval_by_prompt
        assert clustered["method"] == "prompt_clustered_stratified_percentile_bootstrap"
        assert clustered["auc"] == 1.0 and clustered["ci_low"] <= clustered["ci_high"]
        assert clustered["n_memorized_prompts"] + clustered["n_control_prompts"] == len(record.holdout_prompts)
        assert clustered["n_memorized"] + clustered["n_control"] == len(record.calibrator.mcs.holdout_labels)


def test_the_clustered_interval_uses_the_rank_auc_of_roc_auc_score_and_widens_with_clustering():
    from sklearn.metrics import roc_auc_score

    from scripts.finance_sim.calibrate import _clustered_interval, _rank_auc

    rng = np.random.default_rng(3)
    memorized = np.round(rng.normal(0.6, 0.2, size=40), 1)  # rounded, so there are ties
    control = np.round(rng.normal(0.4, 0.2, size=60), 1)
    truth = np.r_[np.ones(40), np.zeros(60)]
    assert _rank_auc(memorized, control) == pytest.approx(roc_auc_score(truth, np.r_[memorized, control]), abs=1e-12)
    # Ten prompts per class whose draws share a prompt effect: resampling prompts, not draws, widens the interval.
    groups = [(label, np.full(8, centre) + rng.normal(0, 0.01, size=8))
              for label, centres in ((1, rng.normal(0.55, 0.1, 10)), (0, rng.normal(0.45, 0.1, 10)))
              for centre in centres]
    clustered = _clustered_interval(groups)
    scores = np.concatenate([scores for _, scores in groups])
    labels = np.concatenate([np.full(scores.size, label) for label, scores in groups])
    from recall_guard.harness.local_reporting import held_out_discrimination

    independent = held_out_discrimination(scores, labels)
    assert clustered["auc"] == pytest.approx(independent["auc"], abs=1e-12)
    assert clustered["ci_high"] - clustered["ci_low"] > independent["ci_high"] - independent["ci_low"]
    assert (clustered["n_memorized_prompts"], clustered["n_control_prompts"]) == (10, 10)


def test_the_clustered_interval_is_the_shared_core_one_without_a_resampling_loop_of_its_own(monkeypatch):
    import inspect

    from recall_guard.core import significance
    from scripts.finance_sim import calibrate

    rng = np.random.default_rng(5)
    # Classes interleaved and unequal draw counts, so the group order and the row alignment both matter.
    groups = [(position % 2, np.round(rng.normal(0.4 + 0.2 * (position % 2), 0.15, size=3 + position % 4), 2))
              for position in range(14)]
    scores = np.concatenate([values for _, values in groups])
    labels = np.concatenate([np.full(values.size, label) for label, values in groups])
    clusters = np.concatenate([np.full(values.size, index) for index, (_, values) in enumerate(groups)])
    core = significance.clustered_auc_interval(scores, labels, clusters)

    record = calibrate._clustered_interval(groups)

    assert list(record) == ["auc", "ci_low", "ci_high", "confidence", "n_resamples", "seed", "method",
                            "n_memorized_prompts", "n_control_prompts", "n_memorized", "n_control"]
    assert (record["ci_low"], record["ci_high"]) == (core.lo, core.hi)
    assert record["auc"] == significance.rank_auc(scores[labels == 1], scores[labels == 0])
    assert (record["confidence"], record["n_resamples"], record["seed"]) == (0.95, 1000, 0)
    assert record["method"] == core.method == "prompt_clustered_stratified_percentile_bootstrap"
    assert (record["n_memorized_prompts"], record["n_control_prompts"]) == (7, 7)
    assert record["n_memorized"] + record["n_control"] == scores.size
    assert calibrate._rank_auc(scores[labels == 1], scores[labels == 0]) == record["auc"]

    # The script calls the core functions and keeps no bootstrap or rank computation of its own.
    for function in (calibrate._clustered_interval, calibrate._rank_auc):
        source = inspect.getsource(function)
        assert "default_rng" not in source and "quantile" not in source and "np.unique" not in source
    calls = []

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return core

    monkeypatch.setattr(calibrate, "clustered_auc_interval", spy)
    assert calibrate._clustered_interval(groups)["ci_low"] == core.lo
    assert calls == [{"n_resamples": 1000, "confidence": 0.95, "seed": 0}]

    # A class without a held-out prompt gives no bounds; the core function is not asked (it would raise).
    calls.clear()
    one_class = calibrate._clustered_interval([(1, np.array([0.5, 0.6])), (1, np.array([0.7]))])
    assert (one_class["auc"], one_class["ci_low"], one_class["ci_high"]) == (None, None, None)
    assert (one_class["n_memorized_prompts"], one_class["n_control_prompts"], one_class["n_memorized"]) == (2, 0, 3)
    assert calls == []


def test_the_clustered_interval_keeps_the_numbers_of_the_calibration_before_the_shared_function():
    """The simulation's interval and the core one both return the pre-refactor numbers, pinned as literals.

    The literals were captured by running the script's own ``_clustered_interval`` and ``_rank_auc`` as they
    stood at commit 507efd3 (the last commit of ``scripts/finance_sim/calibrate.py`` before it delegated to
    ``recall_guard.core.significance``) on exactly this input. They are compared with ``==``, not with a second
    call, so a change in the core bootstrap (order of the generator's draws, cluster order, quantile rule) or
    in the rank AUC fails here instead of silently moving the published intervals (Requirements 6.4, 10.1).
    """
    from recall_guard.core import significance
    from scripts.finance_sim import calibrate

    auc, ci_low, ci_high = 0.7628105590062112, 0.6493186379928315, 0.8640008234519104

    rng = np.random.default_rng(20261111)
    # Sixteen prompts, classes interleaved irregularly, two to seven draws each, scores rounded so they tie.
    pattern = (1, 0, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1)
    groups = [(label, np.round(rng.normal(0.42 + 0.14 * label, 0.15, size=2 + (5 * position) % 6), 1))
              for position, label in enumerate(pattern)]
    scores = np.concatenate([values for _, values in groups])
    labels = np.concatenate([np.full(values.size, label) for label, values in groups])
    clusters = np.concatenate([np.full(values.size, index) for index, (_, values) in enumerate(groups)])
    memorized, control = scores[labels == 1], scores[labels == 0]
    assert [values.size for _, values in groups] == [2, 7, 6, 5, 4, 3, 2, 7, 6, 5, 4, 3, 2, 7, 6, 5]
    assert (memorized.size, control.size) == (28, 46)
    assert np.intersect1d(memorized, control).size == 5  # scores tied across the classes

    record = calibrate._clustered_interval(groups)
    assert (record["auc"], record["ci_low"], record["ci_high"]) == (auc, ci_low, ci_high)
    assert calibrate._rank_auc(memorized, control) == auc
    assert (record["confidence"], record["n_resamples"], record["seed"]) == (0.95, 1000, 0)
    assert (record["n_memorized_prompts"], record["n_control_prompts"]) == (8, 8)
    assert (record["n_memorized"], record["n_control"]) == (28, 46)

    core = significance.clustered_auc_interval(scores, labels, clusters)
    assert (core.lo, core.hi) == (ci_low, ci_high)
    assert significance.rank_auc(memorized, control) == auc


def test_the_holdout_is_split_by_prompt_never_by_draw(calibrated):
    record = calibrated.run.records[(EDGAR, "top_k")]
    held = set(record.holdout_prompts)
    assert len(held) == math.ceil(0.25 * 140)
    held_draws = [draw for draw in calibrated.run.draws[EDGAR] if draw.prompt_sha256 in held]
    assert len(held_draws) == len(held) * SETTINGS.draws == len(record.calibrator.mcs.holdout_labels)
    calibrator = record.calibrator
    rows = [[standardise(draw.feature_row(EstimateSource.TOP_K), calibrator.baseline)[name] for name in FEATURE_ORDER]
            for draw in held_draws]
    scores = calibrator.mcs.classifier.predict_proba(np.asarray(rows))[:, 1]
    assert sorted(calibrator.mcs.holdout_probabilities) == pytest.approx(sorted(scores.tolist()))
    assert sorted(calibrator.mcs.holdout_labels) == sorted(draw.label for draw in held_draws)
    assert record.split_seed == split_seed(PLAN) == PLAN.seed % 2**32


def test_a_calibrator_without_signal_is_marked_weak(tmp_path):
    outcome = calibrate(tmp_path, signal=False, sources=("top_k",))
    record = outcome.run.records[(MACRO, "top_k")]
    assert record.calibrator.mcs.holdout_auc == 0.5 and record.calibrator.is_weak is True
    assert record.holdout_auc_interval["auc"] == 0.5


# ---------- the fit reuses the scorer's estimator ----------

class ReplayLM:
    """Replays the fake host's replies as the legacy LM interface that mcs.train and build_baseline call."""

    model = "gemma3:4b"

    def __init__(self, replies) -> None:
        self.replies = replies

    def generate(self, prompt: str) -> CompletionResult:
        return self.replies[prompt].to_legacy()


def test_with_one_draw_per_prompt_the_fit_equals_build_baseline_and_mcs_train(tmp_path):
    settings = DrawSettings(draws=1, temperature=0.7, seed_base=100)
    ledger = open_ledger(tmp_path / "budget.json")
    host = FakeHost(LedgerCallBudget(ledger))
    corpus = macro_corpus(PLAN, macro_rows())
    with ledger:
        draws = collect_draws(corpus, settings, client=host, ledger=ledger, sources=("top_k",))
    identity = calibration_identity(host, settings, "top_k", corpus)
    record = fit_calibrator(corpus, draws, "top_k", identity=identity, arm="sampling", settings=settings, fit=FIT,
                            split_seed=split_seed(PLAN))

    replay = ReplayLM({prompt: response for (prompt, _), response in host.replies.items()})
    memorized_rows = [EvalRow(p.prompt, 0) for p in corpus.prompts if p.label == 1]
    control_rows = [EvalRow(p.prompt, 0) for p in corpus.prompts if p.label == 0]
    baseline = build_baseline(replay, control_rows, None, min_valid=FIT.min_valid, max_workers=1)
    reference = mcs_module.train(replay, memorized_rows, control_rows, baseline, None, min_auc=FIT.min_auc,
                                 seed=split_seed(PLAN), max_workers=1)
    fitted = record.calibrator
    assert fitted.baseline == baseline
    assert fitted.mcs.feature_order == reference.feature_order == list(FEATURE_ORDER)
    assert np.array_equal(fitted.mcs.classifier.coef_, reference.classifier.coef_)
    assert np.array_equal(fitted.mcs.classifier.intercept_, reference.classifier.intercept_)
    assert (fitted.mcs.holdout_auc, fitted.mcs.is_weak) == (reference.holdout_auc, reference.is_weak)
    assert fitted.mcs.holdout_labels == reference.holdout_labels
    assert fitted.mcs.holdout_probabilities == reference.holdout_probabilities
    assert (fitted.mcs.n_valid_is, fitted.mcs.n_valid_oos) == (reference.n_valid_is, reference.n_valid_oos)


# ---------- draws in the evaluation settings ----------

def generation_pass(tmp_path, **host_options):
    """The macro generation pass on fakes, with full-distribution evidence left pending."""
    ledger = open_ledger(tmp_path / "budget.json")
    host = FakeHost(LedgerCallBudget(ledger), **host_options)
    corpus = macro_corpus(PLAN, macro_rows())
    with ledger:
        draws = collect_draws(corpus, SETTINGS, client=host, ledger=ledger, sources=("top_k", "full_distribution"))
    return corpus, draws, host, ledger


def test_draws_use_the_evaluation_settings_definition_and_leave_full_evidence_pending(tmp_path):
    failure = RequestError("upstream", status_code=500, code="server_error")
    corpus, draws, host, _ = generation_pass(tmp_path, prose={1}, failures={2: failure})
    expected_prompts = [p.prompt for p in corpus.prompts for _ in range(SETTINGS.draws)]
    assert [prompt for prompt, _ in host.calls] == expected_prompts
    assert [parameters for _, parameters in host.calls] == [{"temperature": 0.7, "seed": 100 + i % 2}
                                                            for i in range(len(expected_prompts))]
    assert [(d.draw_index, d.seed, d.temperature) for d in draws[:2]] == [(0, 100, 0.7), (1, 101, 0.7)]
    assert draws[1].failure_code == "parse_error" and draws[1].features == {"top_k": None, "full_distribution": None}
    assert draws[1].response_text is None and not draws[1].evidence_pending
    assert draws[2].failure_code.startswith("request_error:") and draws[2].response_sha256 == digest("")
    assert draws[0].failure_code is None and draws[0].response_sha256 == digest(MACRO_REPLY)
    assert draws[0].response_text == MACRO_REPLY and draws[0].evidence_pending
    assert set(draws[0].features) == {"top_k"} and draws[0].evidence_ref["top_k"] is not None
    assert sum(draw.evidence_pending for draw in draws) == len(draws) - 2


def test_one_evidence_batch_per_prompt_and_rendering_gives_the_rows_of_the_per_draw_path(tmp_path):
    failure = RequestError("upstream", status_code=500, code="server_error")
    corpus, draws, _, _ = generation_pass(tmp_path, prose={1}, failures={2: failure})
    full = FakeFullDistribution()
    settled_calls = []
    settled = collect_evidence(corpus, draws, full_distribution=full, on_draw=settled_calls.append)
    assert len(full.batch_calls) == len(corpus.prompts)  # one request per prompt and rendering
    assert full.batch_calls[0] == (corpus.prompts[0].prompt, (MACRO_REPLY,), 100, 600.0)  # draw 1 did not parse
    assert full.batch_calls[1][1] == (MACRO_REPLY,)  # draw 0 of prompt 1 failed
    assert all(len(call[1]) == SETTINGS.draws for call in full.batch_calls[2:])
    assert [d.key for d in settled] == [d.key for d in draws] and len(settled_calls) == len(draws) - 2
    by_prompt = {}
    for draw in settled:
        if draw.failure_code is None:
            by_prompt.setdefault(draw.prompt_sha256, []).append(draw)
    prompts = {p.prompt_sha256: p.prompt for p in corpus.prompts}
    for sha, group in by_prompt.items():
        for position, draw in enumerate(sorted(group, key=lambda d: d.draw_index)):
            evidence = fake_evidence(prompts[sha], draw.response_text, position)
            expected = features_from_full(draw.response_text, evidence).features
            assert draw.features["full_distribution"] == {name: getattr(expected, name) for name in FEATURE_ORDER}
            assert draw.evidence_ref["full_distribution"] == digest(
                json.dumps(evidence.to_dict(), sort_keys=True, separators=(",", ":")))
            assert full_distribution_row(draw.response_text, evidence) == (
                dict(draw.features["full_distribution"]), draw.evidence_ref["full_distribution"])
            assert draw.features["top_k"] == next(d for d in draws if d.key == draw.key).features["top_k"]
    for failed in (settled[1], settled[2]):
        assert failed.features["full_distribution"] is None and failed.evidence_ref["full_distribution"] is None


def test_host_busy_is_retried_after_the_recorded_wait(tmp_path):
    corpus, draws, _, _ = generation_pass(tmp_path)
    full, sleeps = FakeFullDistribution(busy=2), []
    settled = collect_evidence(corpus, draws, full_distribution=full, sleep=sleeps.append)
    assert sleeps == [HOST_BUSY_WAIT_S, HOST_BUSY_WAIT_S] == [60.0, 60.0]
    assert len(full.batch_calls) == len(corpus.prompts) + 2
    assert full.batch_calls[0] == full.batch_calls[1] == full.batch_calls[2]
    assert not any(draw.evidence_pending for draw in settled)


@pytest.mark.parametrize(("host", "code", "waits"), [
    (lambda: FakeFullDistribution(busy=10**6), "host_busy", HOST_BUSY_RETRIES),
    (lambda: FakeFullDistribution(failure=RequestError("transport", code="transport_error", retryable=True)),
     "transport_error", 0),
    (lambda: FakeFullDistribution(failure=RequestError("HTTP 422", status_code=422, code="tokenisation_failed")),
     "tokenisation_failed", 0),
], ids=["retries-run-out", "dead-host", "refused-batch"])
def test_a_busy_or_dead_evidence_host_leaves_the_stage_open_with_its_draws_and_nothing_fitted(tmp_path, host, code,
                                                                                             waits):
    full, recorded, sleeps = host(), [], []
    with pytest.raises(EvidenceUnavailableError) as unavailable:
        calibrate(tmp_path, full=full, draws=recorded, sleeps=sleeps)
    assert unavailable.value.code == code
    assert sleeps == [60.0] * waits and len(full.batch_calls) == waits + 1  # stopped at the first prompt
    planned = (2 * 24 + 140) * SETTINGS.draws
    ledger = BudgetLedger.open(tmp_path / "budget.json", PLAN, study_limits(), ALLOWANCE)
    assert ledger.status == "open" and ledger.counters["requests"] == planned  # resumable, not complete
    assert len(latest(recorded)) == planned and all(draw.evidence_pending for draw in recorded)
    assert not (tmp_path / "calibrators").exists()
    ledger.close()
    if code != "host_busy":
        return
    # Rerun with the recording and a healthy host: no generation request, only the evidence batches.
    resumed = calibrate(tmp_path, recorded=list(latest(recorded).values()))
    assert resumed.host.calls == [] and len(resumed.full.batch_calls) == 2 * 24 + 140
    assert resumed.ledger.status == "complete" and len(resumed.run.records) == 4


def test_recorded_pending_and_settled_versions_of_a_draw_merge_to_the_settled_one(tmp_path, calibrated):
    stored = list(latest(calibrated.draws).values())
    pending = [draw._pending_version() if draw.failure_code is None else draw for draw in stored]
    resumed = calibrate(tmp_path, recorded=pending + stored)
    assert resumed.host.calls == [] and resumed.full.batch_calls == []
    assert {key: record.to_dict() for key, record in resumed.run.records.items()} == {
        key: record.to_dict() for key, record in calibrated.run.records.items()}


# ---------- limits, recording and resume (11.3) ----------

def test_a_frozen_limit_stops_calibration_incomplete_with_the_recorded_draws_kept(tmp_path):
    with pytest.raises(AdmissionClosedError) as closed:
        calibrate(tmp_path, limits=study_limits(max_requests=50))
    assert closed.value.reason == "max_requests"
    ledger = BudgetLedger.open(tmp_path / "budget.json", PLAN, study_limits(max_requests=50), ALLOWANCE)
    assert (ledger.status, ledger.stop_reason) == ("incomplete", "max_requests")
    assert ledger.counters["requests"] == 50
    assert not (tmp_path / "calibrators").exists()


def test_an_interrupted_calibration_resumes_from_its_recorded_draws_without_repeating_a_request(tmp_path):
    recorded = []

    def crash_after_60(draw):
        recorded.append(draw)
        if len(recorded) == 60:
            raise Interrupted

    with pytest.raises(Interrupted):
        calibrate(tmp_path, sources=("top_k",), on_draw=crash_after_60)
    resumed = calibrate(tmp_path, sources=("top_k",), recorded=list(recorded))
    planned = (2 * 24 + 140) * SETTINGS.draws
    assert len(resumed.host.calls) == planned - 60
    assert resumed.ledger.counters["requests"] == planned and resumed.ledger.status == "complete"
    sent = {(digest(prompt), parameters["seed"]) for prompt, parameters in resumed.host.calls}
    assert not sent & {(draw.prompt_sha256, draw.seed) for draw in recorded}
    assert len(resumed.run.records) == 2


def test_an_interrupted_draw_carries_no_reply_the_empty_hash_and_the_draws_seed_and_temperature():
    prompt = macro_corpus(PLAN, macro_rows()).prompts[0]
    draw = CalibrationDraw.interrupted(prompt, 1, seed=101, temperature=0.7, sources=("top_k", "full_distribution"))
    assert (draw.kind, draw.item_id, draw.rendering, draw.label, draw.prompt_sha256, draw.draw_index) == (
        MACRO, prompt.item_id, prompt.rendering, prompt.label, prompt.prompt_sha256, 1)
    assert (draw.failure_code, draw.response_sha256, draw.response_text, draw.seed, draw.temperature) == (
        INTERRUPTED, digest(""), None, 101, 0.7)
    assert INTERRUPTED == "request_error:interrupted"
    assert dict(draw.features) == dict(draw.evidence_ref) == {"top_k": None, "full_distribution": None}
    assert not draw.evidence_pending
    assert CalibrationDraw.from_dict(json.loads(json.dumps(draw.to_dict(), allow_nan=False))) == draw
    edgar = edgar_corpus(PLAN, edgar_releases()).prompts[0]
    assert CalibrationDraw.interrupted(edgar, 0, seed=100, temperature=0.7, sources=("top_k",)).kind == EDGAR
    with pytest.raises(CalibrationError):
        CalibrationDraw.interrupted(prompt, -1, seed=100, temperature=0.7, sources=("top_k",))


def test_an_interrupted_draw_is_refused_unless_it_carries_the_draws_seed_temperature_and_sources(tmp_path):
    corpus = macro_corpus(PLAN, macro_rows())
    prompt = corpus.prompts[0]
    for number, draw in enumerate((
            CalibrationDraw.interrupted(prompt, 0, seed=SETTINGS.seed_base + 1, temperature=0.7, sources=("top_k",)),
            CalibrationDraw.interrupted(prompt, 0, seed=SETTINGS.seed_base, temperature=0.5, sources=("top_k",)),
            CalibrationDraw.interrupted(prompt, 0, seed=SETTINGS.seed_base, temperature=0.7,
                                        sources=("top_k", "full_distribution")))):
        ledger = open_ledger(tmp_path / f"budget-{number}.json")
        host = FakeHost(LedgerCallBudget(ledger))
        with ledger, pytest.raises(CalibrationError, match="recorded draw"):
            collect_draws(corpus, SETTINGS, client=host, ledger=ledger, sources=("top_k",), recorded=[draw])
        assert host.calls == []


def test_a_request_in_flight_when_the_process_dies_is_passed_back_interrupted_and_the_stage_still_completes(
        tmp_path):
    plan = complete_plan()
    # Zero slack: the stage's limits admit exactly its planned requests and their output reservations.
    limits = study_limits(plan, max_requests=COMPLETE_PLANNED, max_output_tokens=COMPLETE_PLANNED * 64)
    requests: list = []   # the recorder's request lines, written before each send
    outcomes: dict = {}   # its outcome lines, the latest version of each draw

    def on_request(prompt, index, seed) -> None:
        requests.append((prompt, index, seed))

    def on_draw(draw) -> None:
        outcomes[draw.key] = draw

    def run(ledger, host, recorded=()):
        return run_calibration(plan, arm="sampling", macro_rows=macro_rows(plan), edgar_releases=edgar_releases(plan),
                               client=host, ledger=ledger, sources=("top_k", "full_distribution"),
                               full_distribution=FakeFullDistribution(), recorded=recorded, on_request=on_request,
                               on_draw=on_draw, calibrators_dir=tmp_path / "calibrators", sleep=lambda seconds: None)

    crash = 61  # a macro request: the stage drew 61 of its draws when the process died with the 62nd in flight
    ledger = open_ledger(tmp_path / "budget.json", plan=plan, limits=limits)
    first = FakeHost(LedgerCallBudget(ledger), crash_at=crash)
    with pytest.raises(ProcessDied):
        run(ledger, first)
    assert (len(first.calls), len(requests), len(outcomes)) == (crash + 1, crash + 1, crash)
    lost = [(prompt, index, seed) for prompt, index, seed in requests
            if (MACRO, prompt.item_id, prompt.rendering, index) not in outcomes]
    assert len(lost) == 1
    interrupted = [CalibrationDraw.interrupted(prompt, index, seed=seed, temperature=0.7,
                                               sources=("top_k", "full_distribution")) for prompt, index, seed in lost]
    resumed = open_ledger(tmp_path / "budget.json", plan=plan, limits=limits)
    second = FakeHost(LedgerCallBudget(resumed))
    outcome = run(resumed, second, recorded=[*outcomes.values(), *interrupted])
    sent = [(prompt, parameters["seed"]) for prompt, parameters in first.calls + second.calls]
    assert len(sent) == len(set(sent)) == COMPLETE_PLANNED
    assert len(requests) == COMPLETE_PLANNED  # one request line per send, the lost one included
    assert resumed.counters["requests"] == COMPLETE_PLANNED and resumed.status == "complete"
    prompt, index, seed = lost[0]
    draw = next(draw for draw in outcome.draws[MACRO] if draw.key == (MACRO, prompt.item_id, prompt.rendering, index))
    assert (draw.failure_code, draw.seed, draw.temperature, draw.response_sha256, draw.response_text) == (
        INTERRUPTED, seed, 0.7, digest(""), None)
    assert len(outcome.records) == 4
    counts = outcome.records[(MACRO, "top_k")].counts
    assert (counts["draws"], counts["failed_draws"]) == (2 * 24 * COMPLETE_DRAWS[MACRO], 1)


# ---------- the frozen plan's draw, fit and evidence settings ----------

def test_a_complete_plan_supplies_the_draw_fit_and_evidence_settings_of_each_arm():
    plan = complete_plan()
    assert plan_draw_settings(plan, "sampling", MACRO) == DrawSettings(draws=2, temperature=0.7, seed_base=1000)
    assert plan_draw_settings(plan, "sampling", EDGAR) == DrawSettings(draws=1, temperature=0.7, seed_base=1000)
    assert plan_draw_settings(plan, "serving", EDGAR) == DrawSettings(draws=32, temperature=0.0, seed_base=None)
    assert plan_fit_settings(plan, "sampling") == FitSettings(min_auc=0.6, min_valid=10)
    assert plan_fit_settings(plan, "serving") == FitSettings(min_auc=0.6, min_valid=50)
    assert plan_evidence_settings(plan, "sampling") == EvidenceSettings(timeout_s=600.0, top_k_keep=100,
                                                                        host_busy_retries=5, host_busy_wait_s=60.0)
    assert plan_evidence_settings(plan, "serving") is None
    for read in (lambda: plan_fit_settings(PLAN, "sampling"), lambda: plan_evidence_settings(PLAN, "sampling"),
                 lambda: plan_draw_settings(PLAN, "sampling", MACRO)):
        with pytest.raises(ValueError, match="lacks"):
            read()


def test_calibration_on_a_complete_plan_draws_fits_and_requests_evidence_with_its_frozen_settings(tmp_path):
    plan = complete_plan()
    ledger = open_ledger(tmp_path / "budget.json", plan=plan)
    host, full = FakeHost(LedgerCallBudget(ledger)), FakeFullDistribution()
    run = run_calibration(plan, arm="sampling", macro_rows=macro_rows(plan), edgar_releases=edgar_releases(plan),
                          client=host, ledger=ledger, sources=("top_k", "full_distribution"), full_distribution=full,
                          calibrators_dir=tmp_path / "calibrators", sleep=lambda seconds: None)
    assert len(host.calls) == ledger.counters["requests"] == COMPLETE_PLANNED
    edgar_seeds = [parameters["seed"] for prompt, parameters in host.calls if "Press release:" in prompt]
    assert len(edgar_seeds) == 140 and set(edgar_seeds) == {1000}
    for (kind, source), record in run.records.items():
        assert record.draw_settings == plan_draw_settings(plan, "sampling", kind)
        assert record.fit_settings == FitSettings(min_auc=0.6, min_valid=10)
        expected = plan_evidence_settings(plan, "sampling") if source == "full_distribution" else None
        assert record.evidence_settings == expected
    assert {(top_k_keep, timeout_s) for _, _, top_k_keep, timeout_s in full.batch_calls} == {(100, 600.0)}


def test_settings_passed_explicitly_are_used_as_given_and_recorded_in_every_calibrator(tmp_path):
    plan = complete_plan()
    ledger = open_ledger(tmp_path / "budget.json", plan=plan)
    host, full = FakeHost(LedgerCallBudget(ledger)), FakeFullDistribution()
    per_kind = {MACRO: DrawSettings(draws=1, temperature=0.7, seed_base=1000),
                EDGAR: DrawSettings(draws=2, temperature=0.7, seed_base=1000)}
    fit, evidence = FitSettings(min_auc=0.55, min_valid=12), EvidenceSettings(timeout_s=900.0)
    run = run_calibration(plan, arm="sampling", macro_rows=macro_rows(plan), edgar_releases=edgar_releases(plan),
                          client=host, ledger=ledger, sources=("top_k", "full_distribution"), full_distribution=full,
                          settings=per_kind, fit=fit, evidence=evidence, calibrators_dir=tmp_path / "calibrators",
                          sleep=lambda seconds: None)
    assert len(host.calls) == 2 * 24 * 1 + 140 * 2
    for (kind, source), record in run.records.items():
        assert record.draw_settings == per_kind[kind] != plan_draw_settings(plan, "sampling", kind)
        assert record.fit_settings == fit != plan_fit_settings(plan, "sampling")
        assert record.evidence_settings == (evidence if source == "full_distribution" else None)
    assert {timeout_s for *_, timeout_s in full.batch_calls} == {900.0}


def test_a_plan_without_its_arms_needs_explicit_draw_settings(tmp_path):
    ledger = open_ledger(tmp_path / "budget.json")
    host = FakeHost(LedgerCallBudget(ledger))
    with pytest.raises(CalibrationError, match="draw settings"):
        run_calibration(PLAN, arm="sampling", macro_rows=macro_rows(), edgar_releases=edgar_releases(), client=host,
                        ledger=ledger, sources=("top_k",), fit=FIT, calibrators_dir=tmp_path / "calibrators")
    assert host.calls == []


def test_recorded_draws_from_another_corpus_or_setting_are_refused(tmp_path, calibrated):
    corpus = macro_corpus(PLAN, macro_rows())
    edgar_draw = calibrated.run.draws[EDGAR][0]
    macro_draw = calibrated.run.draws[MACRO][0]
    changed = replace(macro_draw, features={**macro_draw.features, "top_k": {name: 0.5 for name in FEATURE_ORDER}})
    foreign = ([edgar_draw], [replace(macro_draw, seed=7)], [replace(macro_draw, label=0)],
               [replace(macro_draw, draw_index=SETTINGS.draws)],
               [replace(macro_draw, features={"full_distribution": macro_draw.features["full_distribution"]},
                        evidence_ref={"full_distribution": macro_draw.evidence_ref["full_distribution"]})],
               [macro_draw, changed])
    for number, recorded in enumerate(foreign):
        ledger = open_ledger(tmp_path / f"budget-{number}.json")
        host = FakeHost(LedgerCallBudget(ledger))
        with ledger, pytest.raises(CalibrationError, match="recorded draw"):
            collect_draws(corpus, SETTINGS, client=host, ledger=ledger, sources=("top_k", "full_distribution"),
                          recorded=recorded)
        assert host.calls == []


# ---------- refusals before any request ----------

def _other_ledger(tmp_path):
    return open_ledger(tmp_path / "other.json")


@pytest.mark.parametrize(("case", "error", "message"), [
    ("plain_budget", CalibrationError, "ledger"),
    ("other_ledger", CalibrationError, "ledger"),
    ("allowance", CalibrationError, "allowance"),
    ("stage", CalibrationError, "stage"),
    ("no_logprobs", CalibrationError, "logprobs"),
    ("no_max_tokens", CalibrationError, "max_tokens"),
    ("full_without_client", CalibrationError, "full-distribution client"),
    ("other_plan", PlanChangedError, "plan"),
])
def test_calibration_off_the_ledger_or_off_its_settings_is_refused_before_any_request(tmp_path, case, error, message):
    limits, allowance, configuration, sources = None, ALLOWANCE, model_config(), ("top_k",)
    plan = PLAN
    if case == "allowance":
        allowance = RequestAllowance(output_tokens=128, timeout_s=30)
    if case == "stage":
        limits = study_limits(stage="full_run")
    if case == "no_logprobs":
        configuration = model_config(logprobs=False)
    if case == "no_max_tokens":
        configuration = replace(model_config(), parameters={k: v for k, v in PARAMETERS.items() if k != "max_tokens"})
    if case == "full_without_client":
        sources = ("full_distribution",)
    if case == "other_plan":
        plan = frozen_plan(seed=SEED + 2)
    ledger = open_ledger(tmp_path / "budget.json", plan=plan, limits=limits or study_limits(plan), allowance=allowance)
    budget = {"plain_budget": CallBudget(1000), "other_ledger": None}.get(case, LedgerCallBudget(ledger))
    host = FakeHost(budget or LedgerCallBudget(_other_ledger(tmp_path)), configuration=configuration)
    with pytest.raises(error, match=message):
        run_calibration(PLAN, arm="sampling", macro_rows=macro_rows(), edgar_releases=edgar_releases(), client=host,
                        ledger=ledger, settings=SETTINGS, sources=sources, fit=FIT,
                        calibrators_dir=tmp_path / "calibrators")
    assert host.calls == []
    assert not (tmp_path / "calibrators").exists()


def test_estimate_sources_are_validated_and_the_serving_arm_calibrates_top_k_only():
    full = FakeFullDistribution()
    assert estimate_sources("sampling", ["top_k", EstimateSource.FULL_DISTRIBUTION], full) == (
        EstimateSource.TOP_K, EstimateSource.FULL_DISTRIBUTION)
    assert estimate_sources("serving", ["top_k"], None) == (EstimateSource.TOP_K,)
    with pytest.raises(CalibrationError, match="serving arm"):
        estimate_sources("serving", ["top_k", "full_distribution"], full)
    for sources in ([], ["top_k", "top_k"], ["logits"]):
        with pytest.raises(CalibrationError):
            estimate_sources("sampling", sources, full)
    with pytest.raises(CalibrationError):
        estimate_sources("local", ["top_k"], None)
    with pytest.raises(TypeError, match="evidence_batch"):
        estimate_sources("sampling", ["full_distribution"], object())


def test_evidence_settings_default_to_the_batch_timeout_and_bounded_busy_retries_and_are_validated():
    assert EvidenceSettings().to_dict() == {"mode": "batch_per_prompt_and_rendering", "timeout_s": 600.0,
                                            "top_k_keep": 100, "host_busy_retries": 5, "host_busy_wait_s": 60.0}
    assert EvidenceSettings.from_dict(EvidenceSettings(timeout_s=900).to_dict()) == EvidenceSettings(timeout_s=900)
    for overrides in ({"timeout_s": 0}, {"timeout_s": float("inf")}, {"top_k_keep": 0}, {"host_busy_retries": -1},
                      {"host_busy_wait_s": -1.0}, {"host_busy_retries": True}):
        with pytest.raises(CalibrationError):
            EvidenceSettings(**overrides)
    with pytest.raises(CalibrationError, match="mode"):
        EvidenceSettings.from_dict(EvidenceSettings().to_dict() | {"mode": "per_draw"})


def test_request_allowance_comes_from_the_real_max_tokens_and_timeout():
    assert request_allowance(model_config()) == RequestAllowance(output_tokens=64, timeout_s=30)
    assert request_allowance(model_config(), usd_per_second=0.001) == RequestAllowance(64, 30, 0.001)
    with pytest.raises(CalibrationError, match="max_tokens"):
        request_allowance(model_config(max_tokens=None))


def test_fitting_an_incomplete_ensemble_or_a_mismatched_identity_is_refused(calibrated):
    corpus = macro_corpus(PLAN, macro_rows())
    draws = calibrated.run.draws[MACRO]
    identity = calibration_identity(calibrated.host, SETTINGS, "top_k", corpus)
    arguments = {"identity": identity, "arm": "sampling", "settings": SETTINGS, "fit": FIT,
                 "split_seed": split_seed(PLAN)}
    with pytest.raises(CalibrationError, match="incomplete"):
        fit_calibrator(corpus, draws[:-1], "top_k", **arguments)
    with pytest.raises(CalibrationError, match="more than once"):
        fit_calibrator(corpus, draws + draws[:1], "top_k", **arguments)
    with pytest.raises(CalibrationError, match="estimate_source"):
        fit_calibrator(corpus, draws, "full_distribution", **arguments)
    full_arguments = arguments | {"identity": calibration_identity(calibrated.host, SETTINGS, "full_distribution",
                                                                   corpus)}
    with pytest.raises(CalibrationError, match="EvidenceSettings"):
        fit_calibrator(corpus, draws, "full_distribution", **full_arguments)
    pending = [draw._pending_version() if draw.failure_code is None else draw for draw in draws]
    with pytest.raises(CalibrationError, match="waits for its full-distribution evidence"):
        fit_calibrator(corpus, pending, "full_distribution", **full_arguments, evidence=EvidenceSettings())
    assert fit_calibrator(corpus, pending, "top_k", **arguments).to_dict() == calibrated.run.records[
        (MACRO, "top_k")].to_dict()  # pending full evidence never changes the top_k fit
    edgar_identity = calibration_identity(calibrated.host, SETTINGS, "top_k", edgar_corpus(PLAN, edgar_releases()))
    with pytest.raises(CalibrationError, match="corpus"):
        fit_calibrator(corpus, draws, "top_k", **(arguments | {"identity": edgar_identity}))
    with pytest.raises(CalibrationError, match="min_valid=10000"):
        fit_calibrator(corpus, draws, "top_k", **(arguments | {"fit": FitSettings(min_valid=10_000)}))
    # Every memorized prompt but one fails to parse: one usable memorized prompt cannot be split.
    unusable = {"top_k": None, "full_distribution": None}
    first = corpus.prompts[0].item_id
    sparse = [replace(draw, failure_code="parse_error", response_text=None, features=unusable, evidence_ref=unusable)
              if draw.label == 1 and draw.item_id != first else draw for draw in draws]
    with pytest.raises(CalibrationError, match="1 memorized.*cannot support"):
        fit_calibrator(corpus, sparse, "top_k", **arguments)


# ---------- the calibrator in the guarded-decision API (4.2) ----------

def test_the_fitted_calibrator_matches_the_evaluation_client_and_refuses_another_source_or_setting(calibrated):
    record = calibrated.run.records[(MACRO, "top_k")]
    prompt = render_identifying(macro_row(PLAN.months.evaluation[0]))
    config = GuardConfig(regime_definition, SourceLabel(EstimateSource.TOP_K, 3), draws=2, min_parsed=1,
                         temperature=0.7, seed_base=100)
    run = guarded_run(prompt, config, client=FakeHost(CallBudget(10)), calibrator=record.calibrator)
    assert run.decision.status == "ok" and run.decision.p_memorized is not None
    full_record = calibrated.run.records[(MACRO, "full_distribution")]
    with pytest.raises(CalibrationMismatchError) as refused:
        guarded_run(prompt, config, client=FakeHost(CallBudget(10)), calibrator=full_record.calibrator)
    assert refused.value.fields == ("estimate_source",)
    with pytest.raises(CalibrationMismatchError) as refused:
        guarded_run(prompt, replace(config, temperature=0.5), client=FakeHost(CallBudget(10)),
                    calibrator=record.calibrator)
    assert refused.value.fields == ("settings",)
