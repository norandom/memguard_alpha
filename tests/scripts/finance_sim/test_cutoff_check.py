"""Tests for the cutoff check (qualified-model-catalogue task 3.2; Requirements 5.3, 5.4, 6.1 to 6.3, 6.5, 6.9, 6.11).

The recall estimate inside the training range against after the cutoff, per
mode (estimate source) and prompt kind, from stored item results built here
without a request: a passing case, a failing case and a case with too few
months, which is reported as missing and never as a value.
"""

from __future__ import annotations

import json
import statistics
from types import SimpleNamespace

import numpy as np
import pytest

from recall_guard.core.distribution import RecordedDraw
from recall_guard.core.significance import Statistic, detectable_effect, permutation_mean_difference
from recall_guard.harness.guarded_decision import GuardIdentity
from recall_guard.mia.estimate_source import EstimateSource, SourceLabel
from scripts.finance_sim.cutoff_check import (
    MINIMUM_MONTHS,
    MODES,
    PASS_ALPHA,
    STATISTIC_NAMES,
    CutoffCheckError,
    CutoffSummary,
    cutoff_check,
    cutoff_check_from_records,
    cutoff_check_path,
    main,
    write_cutoff_check,
)
from scripts.finance_sim.execute import SourceEstimate, derive_result, macro_items, write_result
from scripts.finance_sim.plan import full_run_months, write_plan
from tests.scripts.finance_sim.test_execute import DRAWS, GEMMA, MACRO_ITEMS, PLAN, digest

VECTOR = {"inflation": 0.2, "growth": -0.1, "credit_stress": 0.1, "policy": 0.0, "risk_appetite": -0.2}
PRE = set(PLAN.months.pre_cutoff[GEMMA])
N_RESAMPLES, SEED, BLOCK_LENGTH = PLAN.analysis.n_resamples, PLAN.seed, PLAN.analysis.block_length
CONFIDENCE = 1 - PLAN.analysis.alpha
SUMMARY_FIELDS = ("inside_mean", "after_mean", "difference", "interval_low", "interval_high", "p_value", "n_inside",
                  "n_after")


def months_of(arm: str) -> list[str]:
    return sorted(full_run_months(PLAN, arm))


def identity(arm: str, item, source: str) -> GuardIdentity:
    label = SourceLabel(EstimateSource.TOP_K, 5) if source == "top_k" else SourceLabel(
        EstimateSource.FULL_DISTRIBUTION, 50)
    temperature = PLAN.arms.arms[arm].temperature
    return GuardIdentity(model=GEMMA, protocol="openai_compatible", endpoint="https://example.invalid/v1",
                         response_mapping="generic", source=label,
                         settings_json=json.dumps({"temperature": temperature}), temperature=temperature, draws=DRAWS,
                         calibrator_fingerprint=("c" if source == "top_k" else "d") * 64, calibrator_is_weak=False,
                         prompt_sha256=item.prompt_sha256)


def stored(arm: str, item, top, full=None):
    """A stored result whose draws carry the estimates ``top`` (None: the draw did not parse) and ``full``."""
    arm_plan = PLAN.arms.arms[arm]
    text = json.dumps(VECTOR, sort_keys=True)
    draws = []
    for index, estimate in enumerate(top):
        seed = None if arm_plan.temperature == 0 else arm_plan.seed_base + index
        if estimate is None:
            draws.append(RecordedDraw(index, arm_plan.temperature, seed, digest(""), None, "parse_error", None, None))
        else:
            draws.append(RecordedDraw(index, arm_plan.temperature, seed, digest(text), dict(VECTOR), None, estimate,
                                      "e" * 64))
    sources = ("top_k",) if full is None else ("top_k", "full_distribution")
    estimates = None if full is None else [None if value is None else SourceEstimate(value, "f" * 64, "d" * 64)
                                           for value in full]
    return derive_result(item, draws, estimates, plan=PLAN, arm=arm, stage="full_run",
                         identities={source: identity(arm, item, source) for source in sources})


def spread(base: float, k: int) -> list[float]:
    """Three per-draw estimates around ``base`` that differ by month ``k`` and by draw."""
    return [round(base + 0.01 * ((k * 7) % 5) - 0.02 + 0.01 * j, 4) for j in range(DRAWS)]


def sampling_case(after_months: int | None = None) -> SimpleNamespace:
    """The probe arm: top-k identifying falls after the cutoff, full-distribution identifying rises (a failing case).

    One inside month has an unparsed first draw in the identifying rendering
    (its estimate is the mean of the other two) and one inside month has no
    parsed draw at all in the anonymised rendering (no estimate: missing).
    """
    months = months_of("sampling")
    inside = [month for month in months if month in PRE]
    after = [month for month in months if month not in PRE][:after_months]
    items = {(item.item_id, item.rendering): item for item in macro_items(PLAN, MACRO_ITEMS, stage="full_run",
                                                                        arm="sampling")}
    results, values = [], {}
    for k, month in enumerate([*inside, *after]):
        is_inside = month in PRE
        cells = {
            ("top_k", "identifying"): spread(0.7 if is_inside else 0.2, k),
            ("full_distribution", "identifying"): spread(0.3 if is_inside else 0.6, k),
            ("top_k", "anonymised"): spread(0.3, k),
            ("full_distribution", "anonymised"): spread(0.25, k + 1),
        }
        if month == inside[2]:
            cells["top_k", "identifying"][0] = None
            cells["full_distribution", "identifying"][0] = None
        if month == inside[4]:
            cells["top_k", "anonymised"] = [None] * DRAWS
            cells["full_distribution", "anonymised"] = [None] * DRAWS
        for rendering in ("identifying", "anonymised"):
            results.append(stored("sampling", items[f"macro-{month}", rendering], cells["top_k", rendering],
                                  cells["full_distribution", rendering]))
        for key, estimates in cells.items():
            carried = [value for value in estimates if value is not None]
            if carried:
                values.setdefault(key, {})[month] = statistics.fmean(carried)
    return SimpleNamespace(results=results, values=values, inside=inside, after=after)


@pytest.fixture(scope="module")
def sampling():
    case = sampling_case()
    return SimpleNamespace(case=case, check=cutoff_check(PLAN, case.results, arm="sampling"))


@pytest.fixture(scope="module")
def serving():
    """The primary arm: every evaluation month stored, the top-k source only."""
    months = months_of("serving")
    items = macro_items(PLAN, MACRO_ITEMS, stage="full_run", arm="serving")
    index = {month: k for k, month in enumerate(months)}
    results = []
    for item in items:
        month = item.item_id.removeprefix("macro-")
        base = (0.6 if month in PRE else 0.4) if item.rendering == "identifying" else 0.35
        results.append(stored("serving", item, spread(base, index[month])))
    return SimpleNamespace(months=months, check=cutoff_check(PLAN, results, arm="serving"))


def groups(case, key) -> tuple[np.ndarray, np.ndarray]:
    by_month = case.values[key]
    return (np.asarray([by_month[month] for month in case.inside if month in by_month]),
            np.asarray([by_month[month] for month in case.after if month in by_month]))


def resampled_means(values: np.ndarray, seed: int) -> np.ndarray:
    """Means of the plain month-level bootstrap (block length 1), recomputed independently."""
    rng = np.random.default_rng(seed)
    return np.asarray([values[rng.integers(0, values.size, size=values.size)].mean() for _ in range(N_RESAMPLES)])


def test_the_probe_arm_has_fifteen_months_inside_and_five_after(sampling):
    assert (len(sampling.case.inside), len(sampling.case.after)) == (15, 5)
    assert [(cell.mode, cell.prompt_kind) for cell in sampling.check.cells] == [
        (mode, kind) for mode in MODES for kind in ("identifying", "anonymised")]
    assert MODES == ("top_k", "full_distribution")
    assert sampling.check.to_dict()["planned_months"] == {"inside": 15, "after": 5}


def test_a_passing_case_reports_means_difference_interval_and_the_one_sided_test(sampling):
    cell = sampling.check.cell("top_k", "identifying")
    inside, after = groups(sampling.case, ("top_k", "identifying"))
    assert tuple(cell.statistics) == STATISTIC_NAMES == ("mean_inside", "mean_after", "difference")
    assert all(isinstance(statistic, Statistic) for statistic in cell.statistics.values())
    mean_inside, mean_after, difference = (cell.statistics[name] for name in STATISTIC_NAMES)
    assert mean_inside.value == pytest.approx(inside.mean()) and mean_inside.n == 15
    assert mean_after.value == pytest.approx(after.mean()) and mean_after.n == 5
    assert difference.value == pytest.approx(inside.mean() - after.mean()) and difference.n == 20
    assert difference.effect_size == difference.value
    for statistic in cell.statistics.values():
        assert statistic.unit == "month" and statistic.missing is None and not statistic.by_construction
        assert statistic.interval.confidence == CONFIDENCE
        assert statistic.interval.lo <= statistic.value <= statistic.interval.hi
    # The calculation is exposed: the mean recomputes from its inputs.
    assert mean_inside.inputs["sum_of_month_estimates"] / mean_inside.inputs["n_months"] == pytest.approx(
        mean_inside.value)
    assert difference.inputs["mean_inside"] - difference.inputs["mean_after"] == pytest.approx(difference.value)
    assert (difference.inputs["n_inside"], difference.inputs["n_after"]) == (15, 5)
    # The one-sided block permutation test with blocks = months, months in time order.
    expected = permutation_mean_difference(np.concatenate([inside, after]), [1] * 15 + [0] * 5, np.arange(20),
                                           alternative="greater", n_permutations=N_RESAMPLES, seed=SEED)
    assert difference.test.p_value == expected.p_value and difference.test.statistic == pytest.approx(difference.value)
    assert difference.test.parameters["alternative"] == "greater"
    assert "null" in difference.test.method
    assert difference.test.p_value < PASS_ALPHA == 0.05
    assert cell.summary.passes is True


def test_the_interval_is_a_month_block_bootstrap_within_each_group_and_falls_back_to_block_length_one(sampling):
    cell = sampling.check.cell("top_k", "identifying")
    inside, after = groups(sampling.case, ("top_k", "identifying"))
    difference = cell.statistics["difference"]
    # Five months after the cutoff are fewer than the plan's block length of 6: block length 1, said in the record.
    assert BLOCK_LENGTH == 6 and cell.block_length == 1
    assert "block length 1" in cell.block_length_note and "6" in cell.block_length_note
    assert "block length 1" in difference.formula
    assert difference.inputs["block_length"] == 1 and difference.inputs["n_resamples"] == N_RESAMPLES
    resampled = resampled_means(inside, SEED) - resampled_means(after, SEED + 1)
    assert difference.interval.lo == pytest.approx(np.percentile(resampled, 2.5))
    assert difference.interval.hi == pytest.approx(np.percentile(resampled, 97.5))
    assert "moving-block percentile bootstrap" in difference.interval.method
    standard_error = float(np.std(resampled, ddof=1))
    assert difference.inputs["standard_error"] == pytest.approx(standard_error)
    assert difference.detectable_effect == pytest.approx(detectable_effect(standard_error)) and standard_error > 0
    assert cell.statistics["mean_inside"].detectable_effect is None and cell.statistics["mean_inside"].test is None


def test_the_month_estimate_is_the_mean_of_the_draws_that_carry_one_and_a_month_without_any_is_missing(sampling):
    case = sampling.case
    partial = case.inside[2]
    assert case.values["top_k", "identifying"][partial] == pytest.approx(statistics.fmean(spread(0.7, 2)[1:]))
    cell = sampling.check.cell("top_k", "anonymised")
    assert cell.statistics["mean_inside"].n == 14 and cell.summary.n_inside == 14
    months = next(entry for entry in sampling.check.to_dict()["cells"]
                  if (entry["mode"], entry["prompt_kind"]) == ("top_k", "anonymised"))["months"]
    assert months == {"inside": 14, "after": 5, "planned_inside": 15, "planned_after": 5, "without_stored_result": 0,
                      "without_estimate": 1}
    assert "draws that carry one" in sampling.check.to_dict()["definitions"]["month_estimate"]


def test_a_failing_case_does_not_pass(sampling):
    reversed_cell = sampling.check.cell("full_distribution", "identifying")
    inside, after = groups(sampling.case, ("full_distribution", "identifying"))
    assert reversed_cell.summary.difference == pytest.approx(inside.mean() - after.mean())
    assert reversed_cell.summary.difference < 0 and reversed_cell.summary.p_value > 0.9
    assert reversed_cell.summary.passes is False
    flat = sampling.check.cell("top_k", "anonymised")
    assert flat.summary.p_value >= PASS_ALPHA and flat.summary.passes is False


def test_the_summary_has_exactly_the_fields_of_the_catalogues_cutoff_check(sampling):
    cell = sampling.check.cell("top_k", "identifying")
    summary = cell.summary
    assert isinstance(summary, CutoffSummary)
    assert tuple(summary.to_dict()) == (*SUMMARY_FIELDS, "passes")
    difference = cell.statistics["difference"]
    assert summary.to_dict() == {
        "inside_mean": cell.statistics["mean_inside"].value, "after_mean": cell.statistics["mean_after"].value,
        "difference": difference.value, "interval_low": difference.interval.lo,
        "interval_high": difference.interval.hi, "p_value": difference.test.p_value, "n_inside": 15, "n_after": 5,
        "passes": True}
    document = sampling.check.to_dict()
    entry = document["cells"][0]
    assert entry["summary"] == summary.to_dict() and entry["catalogue_mode"] == "logprobs"
    assert entry["rule_prompt"] is True and document["cells"][1]["rule_prompt"] is False
    assert document["cells"][2]["catalogue_mode"] == "logits"


def test_too_few_months_are_reported_as_missing_and_never_as_a_value():
    case = sampling_case(after_months=1)
    check = cutoff_check(PLAN, case.results, arm="sampling")
    for cell in check.cells:
        assert cell.summary.n_after == 1 < MINIMUM_MONTHS == 2
        for name, statistic in cell.statistics.items():
            assert statistic.value is None and statistic.interval is None and statistic.test is None, name
            assert statistic.detectable_effect is None and statistic.effect_size is None
            assert "after the cutoff" in statistic.missing and "at least 2" in statistic.missing
        summary = cell.summary.to_dict()
        assert [summary[name] for name in SUMMARY_FIELDS[:6]] == [None] * 6
        assert summary["passes"] is None and summary["n_inside"] in (14, 15)
    entry = check.to_dict()["cells"][0]
    assert entry["statistics"]["difference"]["value"] is None
    assert entry["months"]["without_stored_result"] == 4
    rows = [line for line in check.to_markdown().splitlines() if line.startswith("| logprobs (top_k) | identifying")]
    assert rows and all("missing" in row for row in rows)


def test_the_primary_arm_uses_the_plans_block_length_and_its_full_distribution_cells_are_missing(serving):
    inside = [month for month in serving.months if month in PRE]
    cell = serving.check.cell("top_k", "identifying")
    assert (cell.summary.n_inside, cell.summary.n_after) == (len(inside), len(serving.months) - len(inside))
    assert cell.block_length == BLOCK_LENGTH and cell.block_length_note is None
    assert cell.statistics["difference"].inputs["block_length"] == BLOCK_LENGTH
    assert cell.summary.difference == pytest.approx(0.2, abs=0.01) and cell.summary.passes is True
    assert cell.summary.interval_low < cell.summary.difference < cell.summary.interval_high
    assert serving.check.cell("top_k", "anonymised").summary.passes is False
    for kind in ("identifying", "anonymised"):
        missing = serving.check.cell("full_distribution", kind)
        assert (missing.summary.n_inside, missing.summary.n_after, missing.summary.passes) == (0, 0, None)
        for statistic in missing.statistics.values():
            assert statistic.value is None and "records no full_distribution estimate" in statistic.missing


def test_every_statistic_states_its_estimand_calculation_sample_size_and_unit(sampling):
    for cell in sampling.check.cells:
        for statistic in cell.statistics.values():
            assert "month" in statistic.estimand and cell.prompt_kind in statistic.estimand
            assert statistic.formula and statistic.inputs and statistic.unit == "month"
            assert statistic.independent_units is True
    document = sampling.check.to_dict()
    assert "serial" in document["definitions"]["independence"]
    assert document["settings"] == {
        "alpha": PASS_ALPHA, "block_length": BLOCK_LENGTH, "confidence": CONFIDENCE, "detectable_alpha": 0.05,
        "detectable_power": 0.8, "minimum_months_per_group": 2, "n_permutations": N_RESAMPLES,
        "n_resamples": N_RESAMPLES, "seed": SEED}
    assert document["plan_sha256"] == PLAN.plan_sha256 and document["arm"] == "sampling"
    assert document["model"] == GEMMA


def test_the_markdown_table_shows_value_interval_n_and_the_calculation_and_no_item_or_reply(sampling):
    text = sampling.check.to_markdown()
    assert "| Mode | Prompt | Statistic | Value | Interval | n | p-value | Detectable effect | Calculation |" in text
    assert "| Mode | Prompt | Inside | After | Difference | Interval | p-value | n inside | n after | Passes |" in text
    row = next(line for line in text.splitlines()
               if line.startswith("| logprobs (top_k) | identifying | difference"))
    cell = sampling.check.cell("top_k", "identifying")
    assert f"{cell.summary.difference:.4g}" in row and "mean_inside - mean_after" in row
    assert f"mean_inside={cell.summary.inside_mean:.6g}" in row and "n_inside=15" in row and "n_after=5" in row
    assert PLAN.plan_sha256 in text and "one-sided" in text
    document = sampling.check.document
    for private in ("macro-20", '"inflation"', "response", "—"):
        assert private not in text and private not in document


def test_the_outputs_are_canonical_written_once_and_a_different_one_is_refused(sampling, tmp_path):
    out = tmp_path / "analysis"
    json_path, markdown_path = write_cutoff_check(sampling.check, out)
    assert (json_path, markdown_path) == (cutoff_check_path("sampling", out),
                                          cutoff_check_path("sampling", out).with_suffix(".md"))
    assert (json_path.name, markdown_path.name) == ("cutoff-check-sampling.json", "cutoff-check-sampling.md")
    raw = json_path.read_bytes()
    assert b"\r" not in raw and b"\r" not in markdown_path.read_bytes()
    assert raw.decode("utf-8") == json.dumps(json.loads(raw), indent=2, sort_keys=True, allow_nan=False) + "\n"
    assert markdown_path.read_text(encoding="utf-8") == sampling.check.to_markdown()
    again = cutoff_check(PLAN, sampling.case.results, arm="sampling")
    assert again.document == sampling.check.document  # fixed seeds: the replay can compare bytes
    assert write_cutoff_check(again, out) == (json_path, markdown_path) and json_path.read_bytes() == raw
    different = cutoff_check(PLAN, sampling_case(after_months=3).results, arm="sampling")
    with pytest.raises(CutoffCheckError, match="refusing to replace"):
        write_cutoff_check(different, out)
    assert json_path.read_bytes() == raw
    with pytest.raises(CutoffCheckError, match="arm"):
        cutoff_check_path("other", out)


def test_from_records_reads_the_stored_results_and_the_command_writes_them(sampling, tmp_path):
    results_dir = tmp_path / "results"
    for result in sampling.case.results:
        write_result(result, results_dir)
    replayed = cutoff_check_from_records(PLAN, arm="sampling", results_dir=results_dir)
    assert replayed.document == sampling.check.document
    plan_file = write_plan(PLAN, tmp_path / "plan.json")
    out = tmp_path / "analysis"
    common = ["--plan", str(plan_file), "--arm", "sampling", "--results-dir", str(results_dir), "--out-dir", str(out)]
    assert main([*common, "--plan-sha256", "0" * 64]) == 2
    assert not out.exists()
    assert main([*common, "--plan-sha256", PLAN.plan_sha256]) == 0
    assert cutoff_check_path("sampling", out).read_text(encoding="utf-8") == sampling.check.document


def test_results_of_another_arm_are_refused(sampling):
    with pytest.raises(ValueError, match="sampling-arm result"):
        cutoff_check(PLAN, sampling.case.results[:2], arm="serving")
    with pytest.raises(CutoffCheckError, match="arm"):
        cutoff_check(PLAN, [], arm="other")
