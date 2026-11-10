"""Negative-control decision task (task 7.9; Requirements 7.1, 7.2 and 7.3).

Two parts, both on fakes and under ``tmp_path`` only (no network, credentials or
model host, nothing under ``runs/``):

- The report from fixed item results: the per-release change of the outlook
  against no adjustment, its moving-block bootstrap interval over filing
  quarters (reproduced here independently from the documented scheme), and the
  three-outcome rule against the frozen tolerance, with every size and
  threshold read from the frozen plan.
- The run: one outlook ensemble per negative-control release through the 7.7
  execution entry point with the EDGAR calibrators, fitted by the 7.6
  calibration on the fakes; the macro calibrator is refused before any request.
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

import numpy as np
import pytest

from recall_guard.core.completion import ModelConfig, ModelResponse, TokenAlternative, TokenEvidence
from recall_guard.core.distribution import RecordedDraw
from recall_guard.core.full_distribution import FullDistributionEvidence
from recall_guard.core.request_control import RequestControl
from recall_guard.dataset.edgar_outlook import render_outlook_prompt
from recall_guard.dataset.macro_regime import MACRO_AXES, MacroRow
from recall_guard.harness.guarded_decision import GuardIdentity
from recall_guard.mia.estimate_source import EstimateSource, SourceLabel
from scripts.finance_sim import negative_control as module
from scripts.finance_sim.calibrate import DrawSettings, EdgarRelease, FitSettings, run_calibration
from scripts.finance_sim.execute import (
    UNRESOLVED_SUMMARY,
    ExecutionItem,
    SourceEstimate,
    derive_result,
    load_calibrators,
    load_result,
)
from scripts.finance_sim.items import EdgarItem
from scripts.finance_sim.limits import BudgetLedger, LedgerCallBudget, RequestAllowance, StudyLimits
from scripts.finance_sim.negative_control import (
    DISCOUNTING_HONEST_ANSWERS,
    INCONCLUSIVE,
    NO_MATERIAL_CHANGE,
    NegativeControlError,
    negative_control_change,
    negative_control_items,
    run_negative_control,
)
from scripts.finance_sim.plan import (
    ARM_PROFILE_FACTS,
    ARM_REGISTRY_IDS,
    EDGAR_FILES,
    SERVING_FULL_DISTRIBUTION_REASON,
    AnalysisSettings,
    ArmPlan,
    EdgarRow,
    MonthClass,
    PilotSettings,
    PlanChangedError,
    StudyArms,
    freeze_plan,
    load_serving_manifest,
    sha256_of,
    split_months,
)

GEMMA = "google/gemma-3-4b-it"
CREATED_AT = datetime(2026, 11, 10, 12, 0, tzinfo=UTC)
SEED = 20261110
DIGEST = "sha256:" + "f" * 64
#: Small ensembles keep the fakes fast; the module reads the real sizes from the plan.
DRAWS, MIN_PARSED, PILOT_DRAWS = 3, 2, 2
SEED_BASE, TEMPERATURE = 1000, 0.7
PARAMETERS = {"temperature": 0.0, "top_p": 1, "max_tokens": 512, "logprobs": True, "top_logprobs": 5}
ALLOWANCE = RequestAllowance(output_tokens=512, timeout_s=30)
#: Releases per filing quarter, as uneven as the frozen corpus (40 releases in 9 quarters).
QUARTER_SIZES = {"2024Q3": 1, "2024Q4": 5, "2025Q1": 7, "2025Q2": 4, "2025Q3": 2, "2025Q4": 6, "2026Q1": 8,
                 "2026Q2": 2, "2026Q3": 5}
QUARTERS = list(QUARTER_SIZES)
ADJUSTED = ("single_top_k", "single_full_distribution", "ensemble_top_k", "ensemble_full_distribution")


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def month_range(first: str, last: str) -> list[str]:
    year, month = int(first[:4]), int(first[5:])
    out = []
    while f"{year:04d}-{month:02d}" <= last:
        out.append(f"{year:04d}-{month:02d}")
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return out


MONTHS = month_range("2016-01", "2026-06")


def negative_control_releases() -> list[EdgarItem]:
    """Forty held-out control releases in publication order, spread over the quarters of QUARTER_SIZES."""
    releases = []
    for quarter, size in QUARTER_SIZES.items():
        year, number = int(quarter[:4]), int(quarter[5])
        for k in range(size):
            index = len(releases)
            accession = f"{index + 1:010d}-{year % 100:02d}-{index + 1:06d}"
            published = date(year, 3 * (number - 1) + 1 + k % 3, 1 + k)
            releases.append(EdgarItem(
                item_id=f"edgar-negative_control-{accession}", split="negative_control",
                source_file="held_out/oos_control.jsonl", line=index + 1, label=0, published_at=published,
                quarter=quarter, accession=accession, cik=str(1000 + index), company=f"Company {index}",
                release_text=f"Company {index} reports quarterly results. Revenue rose and the outlook is unchanged.",
                row_sha256=digest(f"row:{index}")))
    return releases


RELEASES = negative_control_releases()


def release_hash(release: EdgarItem) -> str:
    """The plan's item hash of a release, as build_plan.item_hashes takes it."""
    return sha256_of({"prompt_sha256": digest(render_outlook_prompt(release.release_text)),
                      "row_sha256": release.row_sha256})


EDGAR_ROWS = ([EdgarRow(f"edgar-calibration-mem-{i:03d}", 1, "calibration") for i in range(40)]
              + [EdgarRow(f"edgar-calibration-ctl-{i:03d}", 0, "calibration") for i in range(100)]
              + [EdgarRow(release.item_id, 0, "held_out") for release in RELEASES]
              + [EdgarRow(f"edgar-unused-{i:03d}", 1, "held_out") for i in range(40)])


def small_arms() -> StudyArms:
    arms = {arm: replace(ArmPlan.for_arm(arm, **dict(ARM_PROFILE_FACTS[arm])), draws=DRAWS, min_parsed=MIN_PARSED)
            for arm in ("sampling", "serving")}
    return StudyArms(registry_ids=ARM_REGISTRY_IDS, arms=arms)


def frozen_plan(analysis: AnalysisSettings | None = None, seed: int = SEED):
    months = [MonthClass(month, {GEMMA: month <= "2024-08"}) for month in MONTHS]
    items = {f"macro-{month}": digest(month) for month in MONTHS}
    items |= {row.row_id: digest(row.row_id) for row in EDGAR_ROWS}
    items |= {release.item_id: release_hash(release) for release in RELEASES}
    arms = small_arms()
    pilot = replace(PilotSettings.for_split(split_months(months, seed=seed), arms), draws=PILOT_DRAWS,
                    min_parsed=PILOT_DRAWS)
    return freeze_plan(months, EDGAR_ROWS, seed=seed, created_at=CREATED_AT, items=items,
                       edgar_files={name: digest(name) for name in EDGAR_FILES}, analysis=analysis, arms=arms,
                       serving_manifest=load_serving_manifest(), pilot=pilot)


PLAN = frozen_plan()


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("the negative control attempted a network connection")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)


# ---------- fixed item results ----------

def guard_identity(item: ExecutionItem, source: EstimateSource, k: int, *, arm: str = "sampling",
                   weak: bool = False) -> GuardIdentity:
    return GuardIdentity(model=GEMMA, protocol="callback", endpoint=None, response_mapping="generic",
                         source=SourceLabel(source, k), settings_json='{"logprobs":true,"top_logprobs":5}',
                         temperature=TEMPERATURE if arm == "sampling" else 0.0, draws=DRAWS,
                         calibrator_fingerprint=("a" if source is EstimateSource.TOP_K else "b") * 64,
                         calibrator_is_weak=weak, prompt_sha256=item.prompt_sha256)


def fixed_result(release: EdgarItem, outlooks, top_k, full=None, *, arm: str = "sampling", plan=PLAN,
                 weak: bool = False, release_text: str | None = None):
    """One release's item result from fixed draws: outlook and top-k estimate per draw, full estimates per draw.

    An outlook of None is an unparsed draw. The result is derived by the 7.7
    execution's own ``derive_result``, so its variants are what the stage stores.
    """
    item = ExecutionItem(release.item_id, "edgar", "outlook", render_outlook_prompt(release_text or release.release_text))
    sampling = arm == "sampling"
    draws = []
    for index, (outlook, estimate) in enumerate(zip(outlooks, top_k, strict=True)):
        parsed = None if outlook is None else {"outlook": outlook}
        draws.append(RecordedDraw(
            draw_index=index, temperature=TEMPERATURE if sampling else 0.0,
            seed=SEED_BASE + index if sampling else None, response_sha256=digest(f"{release.item_id}:{index}"),
            parsed=parsed, failure_code=None if parsed is not None else "parse_error",
            estimate=None if parsed is None else estimate,
            evidence_ref=None if parsed is None or estimate is None else "e" * 64))
    if sampling:
        estimates = tuple(None if outlook is None else SourceEstimate(value, "f" * 64, "b" * 64)
                          for outlook, value in zip(outlooks, full, strict=True))
        identities = {"top_k": guard_identity(item, EstimateSource.TOP_K, 5, weak=weak),
                      "full_distribution": guard_identity(item, EstimateSource.FULL_DISTRIBUTION, 100, weak=weak)}
    else:
        estimates = None
        identities = {"top_k": guard_identity(item, EstimateSource.TOP_K, 5, arm="serving", weak=weak)}
    return derive_result(item, draws, estimates, plan=plan, arm=arm, stage="negative_control", identities=identities)


def scenario(plan=PLAN, *, arm: str = "sampling") -> list:
    """Fixed results giving the three outcomes on the sampling arm.

    Every draw answers 0.8. The top-k estimates are (e0, r, r) with mean 0.5, so
    the ensemble top-k change is -0.4 on every release; the first draw's e0 is
    0.25 in even quarters and 0 in odd ones, so the single top-k change is -0.2
    or 0 by quarter; the full estimates are 0.05, so both full changes are -0.04.
    """
    results = []
    for release in RELEASES:
        first = 0.25 if QUARTERS.index(release.quarter) % 2 == 0 else 0.0
        rest = (1.5 - first) / 2
        results.append(fixed_result(release, [0.8] * DRAWS, [first, rest, rest], [0.05] * DRAWS, plan=plan, arm=arm))
    return results


def varied() -> list:
    """Outlooks from -0.4 to 0.4 by release, top-k estimate 0.5 and full estimate 0.2 on every draw."""
    return [fixed_result(release, [round(0.1 * (index % 9 - 4), 1)] * DRAWS, [0.5] * DRAWS, [0.2] * DRAWS)
            for index, release in enumerate(RELEASES)]


def changes_by_quarter(results, variant: str) -> list[np.ndarray]:
    """Per filing quarter in calendar order, the changes adjusted minus raw outlook of one variant."""
    quarter_of = {release.item_id: release.quarter for release in RELEASES}
    groups: dict[str, list[float]] = {quarter: [] for quarter in QUARTERS}
    for result in results:
        chosen = result.variant(variant)
        groups[quarter_of[result.item_id]].append(chosen.adjusted_summary_value - chosen.summary_value)
    return [np.asarray(groups[quarter], dtype=float) for quarter in QUARTERS]


def moving_block_interval(groups, statistic, *, block_length, n_resamples, seed, confidence):
    """The documented scheme, written out: ceil(n / L) starts uniform on 0..n-L, L consecutive quarters each."""
    rng = np.random.default_rng(seed)
    n = len(groups)
    kept = []
    for _ in range(n_resamples):
        starts = rng.integers(0, n - block_length + 1, size=math.ceil(n / block_length))
        index = (starts[:, None] + np.arange(block_length)).ravel()[:n]
        values = np.concatenate([groups[i] for i in index])
        if values.size:
            kept.append(statistic(values))
    alpha = 1.0 - confidence
    return float(np.percentile(kept, 100 * alpha / 2)), float(np.percentile(kept, 100 * (1 - alpha / 2)))


# ---------- negative-control items ----------

def test_one_outlook_item_per_frozen_negative_control_release_in_plan_order():
    shuffled = list(RELEASES)
    random.Random(7).shuffle(shuffled)
    items = negative_control_items(PLAN, shuffled)
    assert [item.item_id for item in items] == list(PLAN.edgar.negative_control) == [r.item_id for r in RELEASES]
    assert len(items) == len(PLAN.edgar.negative_control) == 40
    for item, release in zip(items, RELEASES, strict=True):
        assert (item.kind, item.rendering, item.realized) == ("edgar", "outlook", None)
        assert item.prompt == render_outlook_prompt(release.release_text) == release.prompt


def test_releases_that_differ_from_the_frozen_plan_are_refused():
    changed = replace(RELEASES[3], release_text=RELEASES[3].release_text + " Guidance was cut.")
    with pytest.raises(NegativeControlError, match="differs from the release the plan froze"):
        negative_control_items(PLAN, [*RELEASES[:3], changed, *RELEASES[4:]])
    with pytest.raises(NegativeControlError, match="missing"):
        negative_control_items(PLAN, RELEASES[:-1])
    with pytest.raises(NegativeControlError, match="more than once"):
        negative_control_items(PLAN, [*RELEASES, RELEASES[0]])
    calibration = replace(RELEASES[0], item_id="edgar-calibration-ctl-000", split="calibration")
    with pytest.raises(NegativeControlError, match="not a negative-control release"):
        negative_control_items(PLAN, [*RELEASES, calibration])
    moved = replace(RELEASES[0], quarter="2024Q4")
    with pytest.raises(NegativeControlError, match="filing quarter"):
        negative_control_items(PLAN, [moved, *RELEASES[1:]])
    with pytest.raises(TypeError):
        negative_control_items(PLAN, [EdgarRelease(RELEASES[0].item_id, 0, RELEASES[0].release_text)])


# ---------- the change against no adjustment (7.2) ----------

def test_each_adjusted_variant_reports_adjusted_minus_raw_outlook_against_its_no_adjustment_reference():
    results = scenario()
    report = negative_control_change(PLAN, results, RELEASES, arm="sampling")
    assert [variant.variant for variant in report.variants] == list(ADJUSTED)
    expected = {"single_top_k": lambda q: -0.2 if q % 2 == 0 else 0.0, "ensemble_top_k": lambda q: -0.4,
                "single_full_distribution": lambda q: -0.04, "ensemble_full_distribution": lambda q: -0.04}
    for variant in report.variants:
        mode = variant.mode
        assert variant.reference == f"{mode}_none" and variant.missing is None
        assert [item.item_id for item in variant.items] == list(PLAN.edgar.negative_control)
        for item, result in zip(variant.items, results, strict=True):
            reference = result.variant(f"{mode}_none")
            assert item.raw == reference.summary_value == reference.adjusted_summary_value == pytest.approx(0.8)
            assert item.adjusted == result.variant(variant.variant).adjusted_summary_value
            assert item.change == pytest.approx(item.adjusted - item.raw)
            assert item.change == pytest.approx(expected[variant.variant](QUARTERS.index(item.quarter)))
            assert (item.sign_flip, item.missing, item.status) == (False, None, "ok")
        changes = np.asarray([item.change for item in variant.items])
        assert variant.mean_absolute_change == pytest.approx(float(np.mean(np.abs(changes))))
        assert variant.signed_mean_change == pytest.approx(float(np.mean(changes)))
        assert (variant.n_items, variant.sign_flips, variant.sign_flip_share) == (40, 0, 0.0)
        assert dict(variant.statuses) == {"ok": 40}


def test_the_interval_is_a_moving_block_bootstrap_over_filing_quarters_with_the_frozen_settings():
    results = varied()
    report = negative_control_change(PLAN, results, RELEASES, arm="sampling")
    analysis = PLAN.analysis
    settings = {"block_length": analysis.negative_control_block_length, "n_resamples": analysis.n_resamples,
                "seed": PLAN.seed, "confidence": 1.0 - analysis.alpha}
    assert (report.block_length, report.n_resamples, report.seed, report.confidence) == tuple(settings.values())
    assert settings["block_length"] == 2 and settings["n_resamples"] == 2000
    assert report.quarters == tuple(QUARTERS)
    assert dict(report.releases_per_quarter) == QUARTER_SIZES
    for name in ADJUSTED:
        variant = report.variant(name)
        groups = changes_by_quarter(results, name)
        lo, hi = moving_block_interval(groups, lambda values: float(np.mean(np.abs(values))), **settings)
        assert (variant.interval.lo, variant.interval.hi) == (pytest.approx(lo, abs=1e-12),
                                                              pytest.approx(hi, abs=1e-12))
        assert variant.interval.confidence == settings["confidence"]
        assert variant.interval.method == "moving-block percentile bootstrap"
        lo, hi = moving_block_interval(groups, lambda values: float(np.mean(values)), **settings)
        assert (variant.signed_interval.lo, variant.signed_interval.hi) == (pytest.approx(lo, abs=1e-12),
                                                                            pytest.approx(hi, abs=1e-12))
        assert variant.dropped_resamples == 0
        assert variant.interval.lo < variant.interval.hi  # outlooks vary by release, so the interval is not a point
    assert negative_control_change(PLAN, results, RELEASES, arm="sampling") == report  # same seed, same report


def test_the_three_outcomes_against_the_frozen_tolerance_flag_only_an_interval_wholly_above_it():
    report = negative_control_change(PLAN, scenario(), RELEASES, arm="sampling")
    tolerance = PLAN.analysis.tolerance
    assert report.tolerance == tolerance == 0.10
    flagged = report.variant("ensemble_top_k")
    assert flagged.interval.lo > tolerance
    assert (flagged.outcome, flagged.flagged) == (DISCOUNTING_HONEST_ANSWERS, True)
    assert flagged.mean_absolute_change == pytest.approx(0.4)
    straddling = report.variant("single_top_k")
    assert straddling.interval.lo < tolerance < straddling.interval.hi
    assert (straddling.outcome, straddling.flagged) == (INCONCLUSIVE, False)
    for name in ("single_full_distribution", "ensemble_full_distribution"):
        small = report.variant(name)
        assert small.interval.hi < tolerance
        assert (small.outcome, small.flagged) == (NO_MATERIAL_CHANGE, False)
        assert small.mean_absolute_change == pytest.approx(0.04)
    assert report.flagged == ("ensemble_top_k",)


def test_the_tolerance_block_length_and_resample_count_are_read_from_the_frozen_plan():
    loose = frozen_plan(AnalysisSettings(tolerance=0.5, n_resamples=300, negative_control_block_length=len(QUARTERS)))
    assert loose.plan_sha256 != PLAN.plan_sha256
    report = negative_control_change(loose, scenario(loose), RELEASES, arm="sampling")
    assert (report.tolerance, report.n_resamples, report.block_length) == (0.5, 300, len(QUARTERS))
    assert report.flagged == ()
    for variant in report.variants:
        # One block of every quarter: each resample is the whole series, so the interval is the statistic itself.
        assert variant.interval.lo == pytest.approx(variant.mean_absolute_change)
        assert variant.interval.hi == pytest.approx(variant.mean_absolute_change)
        assert variant.outcome == NO_MATERIAL_CHANGE
    strict = frozen_plan(AnalysisSettings(tolerance=0.01))
    assert negative_control_change(strict, scenario(strict), RELEASES, arm="sampling").flagged == ADJUSTED


def test_a_sign_flip_is_a_positive_outlook_discounted_to_zero_and_is_reported_beside_the_change():
    results = [fixed_result(release, [0.8 if index % 4 == 0 else -0.3] * DRAWS, [1.0] * DRAWS, [0.0] * DRAWS)
               for index, release in enumerate(RELEASES)]
    report = negative_control_change(PLAN, results, RELEASES, arm="sampling")
    top = report.variant("ensemble_top_k")
    assert top.sign_flips == 10 and top.sign_flip_share == pytest.approx(0.25)
    assert [item.sign_flip for item in top.items] == [index % 4 == 0 for index in range(40)]
    full = report.variant("ensemble_full_distribution")
    assert (full.sign_flips, full.mean_absolute_change) == (0, pytest.approx(0.0))
    assert full.outcome == NO_MATERIAL_CHANGE


def test_a_weak_calibrator_makes_no_change_and_its_status_is_reported():
    results = [fixed_result(release, [0.8] * DRAWS, [0.9] * DRAWS, [0.9] * DRAWS, weak=True) for release in RELEASES]
    report = negative_control_change(PLAN, results, RELEASES, arm="sampling")
    for variant in report.variants:
        assert dict(variant.statuses) == {"adjustment_skipped_weak_calibrator": 40}
        assert variant.mean_absolute_change == 0.0 and variant.outcome == NO_MATERIAL_CHANGE


# ---------- the serving arm and missing values (7.1, 8.10) ----------

def test_the_serving_arm_reports_its_full_distribution_variants_as_missing_with_the_plans_reason():
    results = scenario(arm="serving")
    report = negative_control_change(PLAN, results, RELEASES, arm="serving")
    assert report.arm == "serving"
    assert [variant.variant for variant in report.variants] == list(ADJUSTED)
    reasons = PLAN.arms.arms["serving"].missing_variants
    for name in ("single_full_distribution", "ensemble_full_distribution"):
        variant = report.variant(name)
        assert variant.missing == reasons[name] == SERVING_FULL_DISTRIBUTION_REASON
        assert (variant.outcome, variant.interval, variant.mean_absolute_change, variant.items) == (None, None, None, ())
        assert variant.flagged is False
    assert report.variant("ensemble_top_k").outcome == DISCOUNTING_HONEST_ANSWERS
    assert report.variant("single_top_k").outcome == INCONCLUSIVE
    assert report.flagged == ("ensemble_top_k",)
    with pytest.raises(NegativeControlError, match="serving"):
        negative_control_change(PLAN, results, RELEASES, arm="sampling")


def test_unparsed_unresolved_and_absent_releases_are_missing_with_their_reason_never_substituted():
    results = varied()
    # Draw 0 unparsed: the single variants have no decision, the ensemble still has two parsed draws.
    results[5] = fixed_result(RELEASES[5], [None, 0.3, 0.3], [0.5] * DRAWS, [0.2] * DRAWS)
    unresolved = results[6].variant("ensemble_top_k")
    blank = {"outlook": None}
    variants = tuple(replace(unresolved, decision=blank, adjusted=blank, summary_value=None,
                             adjusted_summary_value=None, category=None)
                     if variant.variant == "ensemble_top_k" else variant for variant in results[6].variants)
    results[6] = replace(results[6], variants=variants)
    absent = results.pop(7)
    report = negative_control_change(PLAN, results, RELEASES, arm="sampling")
    assert report.complete is False
    assert set(report.missing_results) == {absent.item_id}
    assert "no result" in report.missing_results[absent.item_id]
    single = report.variant("single_top_k")
    by_id = {item.item_id: item for item in single.items}
    assert by_id[RELEASES[5].item_id].change is None and by_id[RELEASES[5].item_id].status == "failed"
    assert by_id[RELEASES[5].item_id].missing.startswith("no decision: 0 of 1 draws parsed")
    assert absent.item_id not in by_id and single.n_items == 38
    ensemble = report.variant("ensemble_top_k")
    by_id = {item.item_id: item for item in ensemble.items}
    assert by_id[RELEASES[6].item_id].missing == UNRESOLVED_SUMMARY and by_id[RELEASES[6].item_id].change is None
    assert by_id[RELEASES[5].item_id].change == pytest.approx(-0.15)
    assert ensemble.n_items == 38
    computable = [item.change for item in ensemble.items if item.change is not None]
    assert ensemble.mean_absolute_change == pytest.approx(float(np.mean(np.abs(computable))))
    assert dict(single.statuses) == {"ok": 38, "failed": 1}


def test_a_variant_without_any_computable_change_or_interval_is_missing_with_its_reason():
    # Draw 0 never parses: no single variant has a decision on any release; the ensembles still do.
    results = [fixed_result(release, [None, 0.3, 0.3], [0.5] * DRAWS, [0.2] * DRAWS) for release in RELEASES]
    report = negative_control_change(PLAN, results, RELEASES, arm="sampling")
    for name in ("single_top_k", "single_full_distribution"):
        variant = report.variant(name)
        assert variant.missing == module.NO_CHANGE
        assert (variant.outcome, variant.interval, variant.flagged, variant.n_items) == (None, None, False, 0)
        assert dict(variant.statuses) == {"failed": 40} and len(variant.items) == 40
    assert report.variant("ensemble_top_k").outcome is not None
    # A frozen block length longer than the series of filing quarters gives no interval, never a substitute.
    long_blocks = frozen_plan(AnalysisSettings(negative_control_block_length=len(QUARTERS) + 1))
    report = negative_control_change(long_blocks, scenario(long_blocks), RELEASES, arm="sampling")
    for variant in report.variants:
        assert variant.missing.startswith("the interval cannot be computed") and "block_length" in variant.missing
        assert (variant.outcome, variant.interval, variant.flagged) == (None, None, False)


def test_results_of_another_plan_stage_release_or_prompt_are_refused():
    results = varied()
    other = frozen_plan(AnalysisSettings(tolerance=0.2))
    with pytest.raises(PlanChangedError):
        negative_control_change(other, results, RELEASES, arm="sampling")
    with pytest.raises(NegativeControlError, match="more than once"):
        negative_control_change(PLAN, [*results, results[0]], RELEASES, arm="sampling")
    rewritten = fixed_result(RELEASES[2], [0.1] * DRAWS, [0.5] * DRAWS, [0.2] * DRAWS, release_text="Another text.")
    with pytest.raises(NegativeControlError, match="prompt"):
        negative_control_change(PLAN, [*results[:2], rewritten, *results[3:]], RELEASES, arm="sampling")
    stranger = replace(results[0], item_id="edgar-calibration-ctl-000")
    with pytest.raises(NegativeControlError, match="not a negative-control release"):
        negative_control_change(PLAN, [*results[1:], stranger], RELEASES, arm="sampling")
    with pytest.raises(NegativeControlError, match="negative_control stage"):
        negative_control_change(PLAN, [replace(results[0], stage="full_run", kind="macro",
                                               rendering="identifying"), *results[1:]], RELEASES, arm="sampling")
    with pytest.raises(NegativeControlError, match="arm"):
        negative_control_change(PLAN, results, RELEASES, arm="local")
    unpaired = tuple(replace(variant, summary_value=variant.summary_value + 0.1)
                     if variant.variant == "ensemble_top_k" else variant for variant in results[4].variants)
    with pytest.raises(NegativeControlError, match="not paired"):
        negative_control_change(PLAN, [*results[:4], replace(results[4], variants=unpaired), *results[5:]],
                                RELEASES, arm="sampling")


def test_the_report_states_its_settings_and_the_anchor_and_serialises_to_json():
    report = negative_control_change(PLAN, scenario(), RELEASES, arm="sampling")
    payload = json.loads(json.dumps(report.to_dict(), allow_nan=False))
    assert payload["plan_sha256"] == PLAN.plan_sha256 and payload["arm"] == "sampling"
    assert payload["stage"] == "negative_control" and payload["complete"] is True
    assert (payload["tolerance"], payload["block_length"], payload["n_resamples"]) == (0.1, 2, 2000)
    assert payload["flagged"] == ["ensemble_top_k"]
    assert '{"outlook": 0.2}' in payload["anchor_note"] and "cancels" in payload["anchor_note"]
    assert "filing quarters" in payload["method"]
    entry = payload["variants"][2]
    assert entry["variant"] == "ensemble_top_k" and entry["outcome"] == DISCOUNTING_HONEST_ANSWERS
    assert set(entry["interval"]) == {"lo", "hi", "confidence", "method"}
    assert len(entry["items"]) == 40 and entry["items"][0]["quarter"] == "2024Q3"


# ---------- the run through the 7.7 entry point (7.1) ----------

def macro_row(month: str) -> MacroRow:
    year, number = int(month[:4]), int(month[5:])
    k = MONTHS.index(month)
    return MacroRow(decision_date=date(year, number, calendar.monthrange(year, number)[1]),
                    raw={"baa10y": 2.0 + 0.01 * k, "cpi_yoy": 1.5 + 0.02 * k, "t10y2y": 0.5 - 0.01 * k},
                    zscores={"baa10y_z": 0.1 * (k % 7) - 0.3, "cpi_yoy_z": 0.05 * (k % 11) - 0.2,
                             "t10y2y_z": -0.04 * (k % 5) + 0.1},
                    revised={"BAA10Y": False, "CPIAUCSL": False, "T10Y2Y": False})


def calibration_releases() -> list[EdgarRelease]:
    releases = [EdgarRelease(row_id, 1, f"Company {row_id} reports results (recalled). Revenue rose.")
                for row_id in PLAN.edgar.calibration_memorized]
    return releases + [EdgarRelease(row_id, 0, f"Company {row_id} reports results. Revenue rose.")
                       for row_id in PLAN.edgar.calibration_control]


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def model_config() -> ModelConfig:
    return ModelConfig(protocol="callback", endpoint=None, model="gemma3:4b", parameters=dict(PARAMETERS),
                       timeout_s=30, max_retries=0, retry_backoff_s=0)


def memorized(prompt: str) -> bool:
    return "Real asset identities:" in prompt or "(recalled)" in prompt


def base_logprob(prompt: str, seed: object) -> float:
    noise = random.Random(f"{digest(prompt)}:{seed}").uniform(-0.3, 0.3)
    return (-0.15 if memorized(prompt) else -1.2) + noise


def token_evidence(base: float) -> tuple[TokenEvidence, ...]:
    values = [min(0.0, base - 0.02 * i) for i in range(6)]
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
    """Generation-host double; every reply runs through RequestControl with the client's ledger budget."""

    def __init__(self, budget) -> None:
        self.config = model_config()
        self.budget = budget
        self._control = RequestControl(self.config, budget, clock=Clock())
        self.calls: list[tuple[str, dict]] = []

    def generate(self, prompt: str, **parameters) -> ModelResponse:
        self.calls.append((prompt, dict(parameters)))

        def operation() -> ModelResponse:
            seed = parameters.get("seed")
            return ModelResponse(reply(prompt, seed), token_evidence(base_logprob(prompt, seed)),
                                 usage={"completion_tokens": 7},
                                 requested_settings=dict(self.config.parameters) | parameters)

        return self._control.run(operation, stage="study")


class FakeFull:
    """Full-distribution host double that answers batches only."""

    def __init__(self) -> None:
        self.batch_calls: list[tuple[str, tuple[str, ...]]] = []

    def evidence(self, prompt_text: str, response_text: str, *, top_k_keep: int = 100):
        pytest.fail("evidence must be requested in one batch per item and rendering")

    def evidence_batch(self, prompt_text: str, response_texts, *, top_k_keep: int = 100,
                       timeout_s: float | None = None) -> tuple[FullDistributionEvidence, ...]:
        self.batch_calls.append((prompt_text, tuple(response_texts)))
        out = []
        for position, text in enumerate(response_texts):
            values = [min(0.0, base_logprob(prompt_text, f"full:{position}:{digest(text)}") - 0.01 * i)
                      for i in range(6)]
            out.append(FullDistributionEvidence(
                model_digest=DIGEST, n_vocab=50, response_token_ids=tuple(range(6)), logprob=tuple(values),
                entropy=(1.0,) * 6, rank=(1,) * 6,
                top_logprobs=tuple(((i, v), (40, v - 1.0), (41, v - 2.0), (42, v - 3.0)) for i, v in enumerate(values))))
        return tuple(out)


@pytest.fixture(scope="module")
def calibrators_dir(tmp_path_factory) -> Path:
    """Macro and EDGAR calibrators of the sampling arm, fitted by the 7.6 calibration on the fakes."""
    root = tmp_path_factory.mktemp("negative-control")
    ledger = BudgetLedger.open(root / "calibration" / "budget.json", PLAN,
                               StudyLimits(PLAN.plan_sha256, "sampling", "calibration", 1000, 1_000_000, 1_000_000),
                               ALLOWANCE, clock=Clock())
    run_calibration(PLAN, arm="sampling", macro_rows=[macro_row(month) for month in PLAN.months.calibration],
                    edgar_releases=calibration_releases(), client=FakeHost(LedgerCallBudget(ledger)), ledger=ledger,
                    settings=DrawSettings(draws=2, temperature=TEMPERATURE, seed_base=SEED_BASE),
                    sources=("top_k", "full_distribution"), full_distribution=FakeFull(),
                    fit=FitSettings(min_auc=0.6, min_valid=10), calibrators_dir=root / "calibrators",
                    sleep=lambda seconds: None)
    return root / "calibrators"


def stage_ledger(root: Path) -> BudgetLedger:
    budget = PLAN.limits.stage("sampling", "negative_control")
    limits = StudyLimits(PLAN.plan_sha256, "sampling", "negative_control", budget.max_requests,
                         budget.max_output_tokens, budget.max_seconds)
    return BudgetLedger.open(root / "negative_control" / "budget.json", PLAN, limits, ALLOWANCE, clock=Clock())


def test_one_outlook_ensemble_per_release_runs_through_the_execution_entry_point_with_the_edgar_calibrators(
        tmp_path, calibrators_dir):
    ledger = stage_ledger(tmp_path)
    host, full = FakeHost(LedgerCallBudget(ledger)), FakeFull()
    edgar = load_calibrators(PLAN, arm="sampling", kind="edgar", calibrators_dir=calibrators_dir)
    requests, draws = [], []
    outcome = run_negative_control(PLAN, RELEASES, arm="sampling", client=host, ledger=ledger, calibrators=edgar,
                                   full_distribution=full, on_request=lambda item, index, seed: requests.append(
                                       (item.item_id, index, seed)), on_draw=draws.append,
                                   results_dir=tmp_path / "results", sleep=lambda seconds: None)
    prompts = [render_outlook_prompt(release.release_text) for release in RELEASES]
    assert len(host.calls) == len(requests) == 40 * DRAWS == PLAN.limits.stage("sampling", "negative_control").max_requests
    assert [prompt for prompt, _ in host.calls] == [prompt for prompt in prompts for _ in range(DRAWS)]
    assert [parameters.get("seed") for _, parameters in host.calls] == [SEED_BASE + i for i in range(DRAWS)] * 40
    assert [prompt for prompt, _ in full.batch_calls] == prompts
    assert outcome.execution.status == ledger.status == "complete"
    assert set(outcome.execution.results) == {(release.item_id, "outlook") for release in RELEASES}
    for result in outcome.execution.results.values():
        assert (result.stage, result.kind, result.arm) == ("negative_control", "edgar", "sampling")
        for source in ("top_k", "full_distribution"):
            assert result.identities[source].calibrator_fingerprint == edgar[source].calibrator.fingerprint
    report = outcome.report
    assert report.complete and report.arm == "sampling" and [v.variant for v in report.variants] == list(ADJUSTED)
    assert all(variant.n_items == 40 and variant.outcome is not None for variant in report.variants)
    stored = [load_result(path) for path in outcome.execution.paths.values()]
    assert negative_control_change(PLAN, stored, RELEASES, arm="sampling") == report
    # A rerun reuses every stored result: no request, the same report.
    resumed = stage_ledger(tmp_path)
    rerun_host, rerun_full = FakeHost(LedgerCallBudget(resumed)), FakeFull()
    again = run_negative_control(PLAN, RELEASES, arm="sampling", client=rerun_host, ledger=resumed,
                                 calibrators=edgar, full_distribution=rerun_full, results_dir=tmp_path / "results",
                                 sleep=lambda seconds: None)
    assert (rerun_host.calls, rerun_full.batch_calls) == ([], [])
    assert set(again.execution.reused) == set(outcome.execution.results)
    assert again.report == report


def test_the_calibrators_default_to_the_stored_edgar_calibrators_of_the_arm(tmp_path, calibrators_dir):
    ledger = stage_ledger(tmp_path)
    host = FakeHost(LedgerCallBudget(ledger))
    outcome = run_negative_control(PLAN, RELEASES, arm="sampling", client=host, ledger=ledger,
                                   calibrators_dir=calibrators_dir, full_distribution=FakeFull(),
                                   results_dir=tmp_path / "results", sleep=lambda seconds: None)
    edgar = load_calibrators(PLAN, arm="sampling", kind="edgar", calibrators_dir=calibrators_dir)
    result = next(iter(outcome.execution.results.values()))
    assert result.identities["top_k"].calibrator_fingerprint == edgar["top_k"].calibrator.fingerprint


def test_the_macro_calibrator_is_never_applied_and_is_refused_before_any_request(tmp_path, calibrators_dir):
    ledger = stage_ledger(tmp_path)
    host, full = FakeHost(LedgerCallBudget(ledger)), FakeFull()
    macro = load_calibrators(PLAN, arm="sampling", kind="macro", calibrators_dir=calibrators_dir)
    edgar = load_calibrators(PLAN, arm="sampling", kind="edgar", calibrators_dir=calibrators_dir)
    for calibrators in (macro, {"top_k": edgar["top_k"], "full_distribution": macro["full_distribution"]}):
        with pytest.raises(NegativeControlError, match="macro calibrator is never applied"):
            run_negative_control(PLAN, RELEASES, arm="sampling", client=host, ledger=ledger, calibrators=calibrators,
                                 full_distribution=full, results_dir=tmp_path / "results")
    assert host.calls == [] and full.batch_calls == [] and ledger.status == "open"
    assert not (tmp_path / "results").exists()


def test_the_run_refuses_releases_outside_the_frozen_plan_before_any_request(tmp_path, calibrators_dir):
    ledger = stage_ledger(tmp_path)
    host = FakeHost(LedgerCallBudget(ledger))
    with pytest.raises(NegativeControlError, match="missing"):
        run_negative_control(PLAN, RELEASES[:-1], arm="sampling", client=host, ledger=ledger,
                             calibrators_dir=calibrators_dir, full_distribution=FakeFull(),
                             results_dir=tmp_path / "results")
    assert host.calls == []
