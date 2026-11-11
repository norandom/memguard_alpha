"""Cutoff check: the recall estimate inside the training range against after the cutoff.

Qualified-model-catalogue task 3.2 (Requirements 5.3, 5.4, 6.1, 6.2, 6.3, 6.5,
6.9 and 6.11). Recall is the model's knowledge of a financial time series
beyond the analysis date, for dates inside its training range. If the recall
estimate (``p_memorized``) tracks recall, it is higher for analysis months
inside the training range than for months after the model's training cutoff.
This analysis measures that, per mode and prompt kind, and feeds the fourth
check of the catalogue's qualification rule: estimate inside above after,
one-sided p below 0.05.

Inputs, private and read-only; no request is sent. The frozen plan and the
stored item results of one arm's full run
(:func:`~scripts.finance_sim.execute.load_result`). The months are the arm's
full-run months (:func:`~scripts.finance_sim.plan.full_run_months`); a month is
inside the training range when ``plan.months.pre_cutoff`` lists it for the
arm's model and after the cutoff otherwise. Pilot items are excluded through
:func:`~scripts.finance_sim.pilot.confirmatory_results`.

Unit of analysis: the month. Per month, mode (``top_k``, the catalogue's
logprobs mode, or ``full_distribution``, its logits mode) and prompt kind (the
``identifying`` or the ``anonymised`` rendering), the month estimate is the mean
of the per-draw estimates of that source over the draws that carry one. It
does not depend on the ensemble's minimum of parsed draws. A month without a
stored result or without any estimate is missing for that cell.

Statistics per mode and prompt kind, each a
:class:`~recall_guard.core.significance.Statistic`:

- ``mean_inside`` and ``mean_after``: the mean month estimate of each group,
  with a moving-block percentile bootstrap interval over that group's months.
- ``difference``: inside minus after. Interval: the months are resampled within
  each group with :func:`~recall_guard.core.significance.block_bootstrap`
  (inside with the plan's seed, after with seed + 1) and the interval is the
  percentile interval of the resampled difference. The block length is the
  plan's analysis block length; when a group has fewer months than that, block
  length 1 is used and the record says so. Test: the one-sided block
  permutation test
  :func:`~recall_guard.core.significance.permutation_mean_difference` with
  ``alternative="greater"`` and blocks = months (null: the two groups have the
  same mean). Effect size: the difference itself, in units of the estimate.
  Detectable effect: at power 0.8 and two-sided alpha 0.05 from the bootstrap
  standard error of the difference.

Fewer than two months in either group gives records without a value that name
what is missing (6.9); a value is never substituted. An arm that records no
full-distribution estimate has those cells missing with that reason.

Beside the records, each cell carries a summary with exactly the fields of the
catalogue's ``CutoffCheck`` and ``passes`` (difference above 0 and p below
0.05). The identifying prompt is the recall-exposed one and the one the rule
reads; the anonymised one is reported beside it.

Outputs: the canonical JSON document (sorted keys, LF) and its Markdown table,
``cutoff-check-<arm>.json`` and ``.md`` under ``runs/finance-sim/analysis/``,
written once and never replaced by a different one. Derived statistics only:
month estimates appear only as aggregates, and no item, reply or licensed
value is listed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from recall_guard.core.significance import (
    DETECTABLE_ALPHA,
    DETECTABLE_POWER,
    Interval,
    Statistic,
    TestResult,
    block_bootstrap,
    detectable_effect,
    permutation_mean_difference,
)
from scripts.finance_sim.calibrate import ANONYMISED, IDENTIFYING, macro_item_id
from scripts.finance_sim.execute import RESULTS_DIR, ItemResult, load_result, planned_keys, result_path
from scripts.finance_sim.limits import ARMS, RUNS_ROOT
from scripts.finance_sim.pilot import CONFIRMATORY_STAGE, confirmatory_results
from scripts.finance_sim.plan import (
    FULL_DISTRIBUTION,
    TOP_K,
    StudyPlan,
    full_run_months,
    load_plan,
    require_complete,
)

#: Analysis outputs (design: ``runs/finance-sim/analysis/``).
ANALYSIS_DIR = RUNS_ROOT / "analysis"
CUTOFF_SCHEMA = "finance-sim-cutoff-check/1"
#: The modes, as estimate sources, and the catalogue's name of each.
MODES: tuple[str, ...] = (TOP_K, FULL_DISTRIBUTION)
CATALOGUE_MODES: Mapping[str, str] = MappingProxyType({TOP_K: "logprobs", FULL_DISTRIBUTION: "logits"})
#: The prompt kinds; the first is the recall-exposed prompt the qualification rule reads.
PROMPT_KINDS: tuple[str, ...] = (IDENTIFYING, ANONYMISED)
RULE_PROMPT = IDENTIFYING
STATISTIC_NAMES: tuple[str, ...] = ("mean_inside", "mean_after", "difference")
#: A group needs at least this many months with an estimate.
MINIMUM_MONTHS = 2
#: The qualification rule's level for the one-sided test.
PASS_ALPHA = 0.05
UNIT = "month"

MONTH_ESTIMATE = (
    "per month, mode and prompt kind: the mean of the per-draw estimates (p_memorized) of that mode's source over the "
    "draws that carry one; it does not depend on the ensemble's minimum of parsed draws; a month without a stored "
    "result or without any estimate is missing for that cell"
)
GROUPS = (
    "inside: the arm's full-run months on or before the training cutoff of the arm's model (the plan's pre_cutoff "
    "months); after: its full-run months after that cutoff"
)
INDEPENDENCE = (
    "one row per month; the permutation test treats months as exchangeable under the null, and the interval allows "
    "serial dependence between months within a block of block_length months (none at block length 1)"
)
PASS_RULE = "passes when the difference inside minus after is above 0 and the one-sided p-value is below 0.05"
DIFFERENCE_INTERVAL_METHOD = (
    "moving-block percentile bootstrap of the difference, months resampled within each group (inside with the seed, "
    "after with seed + 1)"
)
EXIT_WRITTEN, EXIT_REFUSED = 0, 2
_SHA256_LENGTH = 64


class CutoffCheckError(ValueError):
    """Inputs or outputs of the cutoff check are refused."""


@dataclass(frozen=True)
class CutoffSummary:
    """The fields of the catalogue's ``CutoffCheck`` and whether the check passes; None where it is missing."""

    inside_mean: float | None
    after_mean: float | None
    difference: float | None
    interval_low: float | None
    interval_high: float | None
    p_value: float | None
    n_inside: int
    n_after: int
    passes: bool | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CutoffCell:
    """The cutoff check of one mode and prompt kind."""

    mode: str
    prompt_kind: str
    statistics: Mapping[str, Statistic]
    summary: CutoffSummary
    months: Mapping[str, int]
    block_length: int | None
    block_length_note: str | None


@dataclass(frozen=True)
class CutoffCheckResult:
    """The cutoff check of one arm: its cells, the canonical JSON document and its Markdown table."""

    plan_sha256: str
    arm: str
    cells: tuple[CutoffCell, ...]
    document: str

    def cell(self, mode: str, prompt_kind: str) -> CutoffCell:
        """The cell of one mode (``top_k`` or ``full_distribution``) and prompt kind."""
        for cell in self.cells:
            if (cell.mode, cell.prompt_kind) == (mode, prompt_kind):
                return cell
        raise KeyError((mode, prompt_kind))

    def to_dict(self) -> dict[str, Any]:
        return json.loads(self.document)

    def to_markdown(self) -> str:
        return _markdown(self.to_dict())


# ---------- month estimates ----------

def _estimates(result: ItemResult, mode: str) -> list[float] | None:
    """The per-draw estimates of ``mode`` that a stored result carries; None when it does not record the source."""
    if mode == TOP_K:
        return [draw.estimate for draw in result.recorded_draws if draw.estimate is not None]
    if result.full_distribution is None:
        return None
    return [entry.estimate for entry in result.full_distribution if entry is not None and entry.estimate is not None]


@dataclass(frozen=True)
class _Groups:
    """Month estimates of one cell in time order, by group, and the counts of what is missing."""

    inside: np.ndarray
    after: np.ndarray
    months: Mapping[str, int]
    source_recorded: bool


def _groups(months: Sequence[tuple[str, bool]], by_key: Mapping[tuple[str, str], ItemResult], mode: str,
            prompt_kind: str) -> _Groups:
    values: dict[bool, list[float]] = {True: [], False: []}
    no_result = no_estimate = 0
    recorded = False
    for month, inside in months:
        result = by_key.get((macro_item_id(month), prompt_kind))
        if result is None:
            no_result += 1
            continue
        estimates = _estimates(result, mode)
        recorded = recorded or estimates is not None
        if not estimates:
            no_estimate += 1
            continue
        values[inside].append(float(np.mean(estimates)))
    planned_inside = sum(1 for _, inside in months if inside)
    counts = {"inside": len(values[True]), "after": len(values[False]), "planned_inside": planned_inside,
              "planned_after": len(months) - planned_inside, "without_stored_result": no_result,
              "without_estimate": no_estimate}
    return _Groups(inside=np.asarray(values[True], dtype=float), after=np.asarray(values[False], dtype=float),
                   months=MappingProxyType(counts), source_recorded=recorded or mode == TOP_K or not by_key)


# ---------- the statistics of one cell ----------

def _estimand(name: str, mode: str, prompt_kind: str) -> str:
    what = {
        "mean_inside": "mean month estimate over the months inside the training range",
        "mean_after": "mean month estimate over the months after the training cutoff",
        "difference": ("difference of the mean month estimate, months inside the training range minus months after "
                       "the training cutoff"),
    }[name]
    return (f"{what}; month estimate: the recall estimate (p_memorized) of the {mode} source "
            f"({CATALOGUE_MODES[mode]} mode) under the {prompt_kind} prompt, averaged over the month's draws")


def _missing_cell(mode: str, prompt_kind: str, groups: _Groups, reason: str) -> dict[str, Statistic]:
    n_inside, n_after = groups.months["inside"], groups.months["after"]
    inputs = {"n_inside": n_inside, "n_after": n_after, "minimum_months_per_group": MINIMUM_MONTHS}
    formulas = {"mean_inside": "sum_of_month_estimates / n_months", "mean_after": "sum_of_month_estimates / n_months",
                "difference": "mean_inside - mean_after"}
    sizes = {"mean_inside": n_inside, "mean_after": n_after, "difference": n_inside + n_after}
    return {name: Statistic(name=name, estimand=_estimand(name, mode, prompt_kind), formula=formulas[name],
                            inputs=inputs, unit=UNIT, n=sizes[name], independent_units=True, value=None, interval=None,
                            test=None, effect_size=None, detectable_effect=None, by_construction=False, missing=reason)
            for name in STATISTIC_NAMES}


def _mean(values: np.ndarray) -> float:
    return float(np.mean(values))


def _computed_cell(mode: str, prompt_kind: str, groups: _Groups, *, block_length: int, n_resamples: int, seed: int,
                   confidence: float) -> dict[str, Statistic]:
    inside, after = groups.inside, groups.after
    n_inside, n_after = int(inside.size), int(after.size)
    settings = {"block_length": block_length, "n_resamples": n_resamples, "seed": seed, "confidence": confidence}
    after_means: list[float] = []

    def after_mean(rows: np.ndarray) -> float:
        after_means.append(_mean(rows))
        return after_means[-1]

    after_interval = block_bootstrap(after_mean, after, block_length=block_length, n_resamples=n_resamples,
                                     seed=seed + 1, confidence=confidence)
    inside_interval = block_bootstrap(_mean, inside, block_length=block_length, n_resamples=n_resamples, seed=seed,
                                      confidence=confidence)
    # The same inside resamples as above (same seed), each paired with one independent after resample.
    paired = iter(after_means)
    differences: list[float] = []

    def resampled_difference(rows: np.ndarray) -> float:
        differences.append(_mean(rows) - next(paired))
        return differences[-1]

    interval = replace(block_bootstrap(resampled_difference, inside, block_length=block_length,
                                       n_resamples=n_resamples, seed=seed, confidence=confidence),
                       method=DIFFERENCE_INTERVAL_METHOD)
    standard_error = float(np.std(differences, ddof=1)) if len(differences) > 1 else 0.0
    test = permutation_mean_difference(np.concatenate([inside, after]), [1] * n_inside + [0] * n_after,
                                       np.arange(n_inside + n_after), alternative="greater",
                                       n_permutations=n_resamples, seed=seed)
    mean_inside, mean_after = _mean(inside), _mean(after)
    difference = mean_inside - mean_after
    interval_text = (f"interval: percentile interval at confidence {confidence:g} of the mean over n_resamples "
                     f"moving-block resamples of the group's months, block length {block_length}")

    def group(name: str, values: np.ndarray, value: float, group_interval: Interval, group_seed: int) -> Statistic:
        return Statistic(
            name=name, estimand=_estimand(name, mode, prompt_kind),
            formula=f"sum_of_month_estimates / n_months; {interval_text}",
            inputs={"sum_of_month_estimates": float(values.sum()), "n_months": int(values.size),
                    **settings, "seed": group_seed},
            unit=UNIT, n=int(values.size), independent_units=True, value=value, interval=group_interval, test=None,
            effect_size=None, detectable_effect=None, by_construction=False, missing=None)

    return {
        "mean_inside": group("mean_inside", inside, mean_inside, inside_interval, seed),
        "mean_after": group("mean_after", after, mean_after, after_interval, seed + 1),
        "difference": Statistic(
            name="difference", estimand=_estimand("difference", mode, prompt_kind),
            formula=(f"mean_inside - mean_after; interval: percentile interval at confidence {confidence:g} of the "
                     "difference over n_resamples moving-block resamples of the months within each group, block "
                     f"length {block_length}; p: one-sided permutation test over months, (1 + number of permuted "
                     "differences at least the observed one) / (1 + n_permutations); effect size: the difference; "
                     "detectable effect: (z_0.975 + z_0.80) x standard_error, the standard deviation of the "
                     "resampled difference"),
            inputs={"mean_inside": mean_inside, "mean_after": mean_after, "n_inside": n_inside, "n_after": n_after,
                    **settings, "n_permutations": n_resamples, "standard_error": standard_error},
            unit=UNIT, n=n_inside + n_after, independent_units=True, value=difference, interval=interval, test=test,
            effect_size=difference, detectable_effect=detectable_effect(standard_error), by_construction=False,
            missing=None),
    }


def _cell(plan: StudyPlan, arm: str, mode: str, prompt_kind: str, groups: _Groups) -> CutoffCell:
    n_inside, n_after = groups.months["inside"], groups.months["after"]
    if min(n_inside, n_after) < MINIMUM_MONTHS:
        if not groups.source_recorded:
            reason = f"the {arm} arm records no {mode} estimate, so no month carries one"
        else:
            reason = (f"months with a {mode} estimate under the {prompt_kind} prompt: {n_inside} inside the training "
                      f"range and {n_after} after the cutoff; at least {MINIMUM_MONTHS} are needed in each group")
        return CutoffCell(
            mode=mode, prompt_kind=prompt_kind,
            statistics=MappingProxyType(_missing_cell(mode, prompt_kind, groups, reason)),
            summary=CutoffSummary(None, None, None, None, None, None, n_inside, n_after, None), months=groups.months,
            block_length=None, block_length_note=None)
    planned = plan.analysis.block_length
    block_length, note = planned, None
    if min(n_inside, n_after) < planned:
        block_length = 1
        note = (f"a group has fewer months ({min(n_inside, n_after)}) than the plan's block length {planned}, so "
                "block length 1 is used: months are resampled singly")
    statistics = _computed_cell(mode, prompt_kind, groups, block_length=block_length,
                                n_resamples=plan.analysis.n_resamples, seed=plan.seed,
                                confidence=1.0 - plan.analysis.alpha)
    difference = statistics["difference"]
    summary = CutoffSummary(
        inside_mean=statistics["mean_inside"].value, after_mean=statistics["mean_after"].value,
        difference=difference.value, interval_low=difference.interval.lo, interval_high=difference.interval.hi,
        p_value=difference.test.p_value, n_inside=n_inside, n_after=n_after,
        passes=bool(difference.value > 0.0 and difference.test.p_value < PASS_ALPHA))
    return CutoffCell(mode=mode, prompt_kind=prompt_kind, statistics=MappingProxyType(statistics), summary=summary,
                      months=groups.months, block_length=block_length, block_length_note=note)


# ---------- the document ----------

def _json_value(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _interval_dict(interval: Interval | None) -> dict[str, Any] | None:
    if interval is None:
        return None
    return {"lo": interval.lo, "hi": interval.hi, "confidence": interval.confidence, "method": interval.method}


def _test_dict(test: TestResult | None) -> dict[str, Any] | None:
    if test is None:
        return None
    return {"statistic": test.statistic, "interval": _interval_dict(test.interval), "p_value": test.p_value,
            "method": test.method, "parameters": {key: _json_value(value) for key, value in test.parameters.items()}}


def _statistic_dict(statistic: Statistic) -> dict[str, Any]:
    return {"name": statistic.name, "estimand": statistic.estimand, "formula": statistic.formula,
            "inputs": {key: _json_value(value) for key, value in statistic.inputs.items()}, "unit": statistic.unit,
            "n": statistic.n, "independent_units": statistic.independent_units, "value": statistic.value,
            "interval": _interval_dict(statistic.interval), "test": _test_dict(statistic.test),
            "effect_size": statistic.effect_size, "detectable_effect": statistic.detectable_effect,
            "by_construction": statistic.by_construction, "missing": statistic.missing}


def _cell_dict(cell: CutoffCell) -> dict[str, Any]:
    return {"mode": cell.mode, "catalogue_mode": CATALOGUE_MODES[cell.mode], "prompt_kind": cell.prompt_kind,
            "rule_prompt": cell.prompt_kind == RULE_PROMPT, "months": dict(cell.months),
            "block_length": cell.block_length, "block_length_note": cell.block_length_note,
            "statistics": {name: _statistic_dict(statistic) for name, statistic in cell.statistics.items()},
            "summary": cell.summary.to_dict()}


def _check_scope(plan: StudyPlan, arm: str) -> None:
    if not isinstance(plan, StudyPlan):
        raise TypeError(f"plan must be a StudyPlan; got {type(plan).__name__}")
    require_complete(plan)
    if arm not in ARMS:
        raise CutoffCheckError(f"arm must be one of {list(ARMS)}; got {arm!r}")


def cutoff_check(plan: StudyPlan, results: Iterable[ItemResult] | Mapping[Any, ItemResult], *,
                 arm: str) -> CutoffCheckResult:
    """The cutoff check of ``arm`` from the stored item results of its full run; no request is sent.

    ``results`` may include pilot items, which are excluded; anything else must
    be a planned result of the arm's full run, given once. Every size and seed
    comes from the frozen plan (see the module docstring for the statistics).
    """
    _check_scope(plan, arm)
    by_key = confirmatory_results(plan, results, arm=arm, stage=CONFIRMATORY_STAGE).results
    model = plan.arms.registry_ids[arm]
    pre = set(plan.months.pre_cutoff[model])
    months = [(month, month in pre) for month in sorted(full_run_months(plan, arm))]
    cells = tuple(_cell(plan, arm, mode, prompt_kind, _groups(months, by_key, mode, prompt_kind))
                  for mode in MODES for prompt_kind in PROMPT_KINDS)
    n_inside = sum(1 for _, inside in months if inside)
    document = {
        "schema": CUTOFF_SCHEMA, "plan_sha256": plan.plan_sha256, "arm": arm, "stage": CONFIRMATORY_STAGE,
        "role": plan.arms.arms[arm].role, "model": model,
        "planned_months": {"inside": n_inside, "after": len(months) - n_inside},
        "settings": {"block_length": plan.analysis.block_length, "n_resamples": plan.analysis.n_resamples,
                     "n_permutations": plan.analysis.n_resamples, "seed": plan.seed, "alpha": PASS_ALPHA,
                     "confidence": 1.0 - plan.analysis.alpha, "minimum_months_per_group": MINIMUM_MONTHS,
                     "detectable_power": DETECTABLE_POWER, "detectable_alpha": DETECTABLE_ALPHA},
        "definitions": {"month_estimate": MONTH_ESTIMATE, "groups": GROUPS, "independence": INDEPENDENCE,
                        "passes": PASS_RULE,
                        "rule_prompt": (f"the {RULE_PROMPT} prompt is the recall-exposed one and the one the "
                                        "qualification rule reads; the other is reported beside it")},
        "cells": [_cell_dict(cell) for cell in cells],
    }
    text = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    return CutoffCheckResult(plan_sha256=plan.plan_sha256, arm=arm, cells=cells, document=text)


def cutoff_check_from_records(plan: StudyPlan, *, arm: str,
                              results_dir: str | Path = RESULTS_DIR) -> CutoffCheckResult:
    """The cutoff check from the stored result of every planned item and rendering of the arm's full run."""
    _check_scope(plan, arm)
    results = []
    for item_id, rendering in planned_keys(plan, arm, CONFIRMATORY_STAGE):
        path = result_path(arm, CONFIRMATORY_STAGE, item_id, rendering, results_dir)
        if path.exists():
            results.append(load_result(path))
    return cutoff_check(plan, results, arm=arm)


# ---------- the markdown table ----------

def _number(value: float | None) -> str:
    return "" if value is None else f"{value:.4g}"


def _interval_text(interval: Mapping[str, Any] | None) -> str:
    if interval is None:
        return ""
    return f"{interval['lo']:.4g} to {interval['hi']:.4g} ({interval['confidence'] * 100:g} percent)"


def _input_text(value: object) -> str:
    return f"{value:.6g}" if isinstance(value, float) else str(value)


def _cell_text(text: str) -> str:
    return text.replace("|", "/").replace("\n", " ")


def _row(cells: Iterable[str]) -> str:
    return "| " + " | ".join(_cell_text(cell) for cell in cells) + " |"


def _markdown(data: Mapping[str, Any]) -> str:
    settings, planned = data["settings"], data["planned_months"]
    lines = [
        f"# Cutoff check: {data['arm']} arm",
        "",
        f"Recall estimate inside the training range against after the training cutoff. The {data['arm']} arm "
        f"({data['role']} role), model {data['model']}, full run. Frozen plan {data['plan_sha256']}. Planned months: "
        f"{planned['inside']} inside the training range, {planned['after']} after the cutoff.",
        "",
        "## Statistics",
        "",
        "| Mode | Prompt | Statistic | Value | Interval | n | p-value | Detectable effect | Calculation |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    summaries = []
    for cell in data["cells"]:
        label = f"{cell['catalogue_mode']} ({cell['mode']})"
        for name in STATISTIC_NAMES:
            statistic = cell["statistics"][name]
            inputs = ", ".join(f"{key}={_input_text(value)}" for key, value in statistic["inputs"].items())
            if statistic["value"] is None:
                lines.append(_row([label, cell["prompt_kind"], name, "missing", "", str(statistic["n"]), "", "",
                                   f"{statistic['missing']}; inputs: {inputs}"]))
                continue
            test = statistic["test"]
            lines.append(_row([label, cell["prompt_kind"], name, _number(statistic["value"]),
                               _interval_text(statistic["interval"]), str(statistic["n"]),
                               "" if test is None else _number(test["p_value"]),
                               _number(statistic["detectable_effect"]),
                               f"{statistic['formula']}; inputs: {inputs}"]))
        summary = cell["summary"]
        if summary["passes"] is None:
            summaries.append([label, cell["prompt_kind"], "missing", "missing", "missing", "", "",
                              str(summary["n_inside"]), str(summary["n_after"]), "missing"])
        else:
            summaries.append([label, cell["prompt_kind"], _number(summary["inside_mean"]),
                              _number(summary["after_mean"]), _number(summary["difference"]),
                              f"{summary['interval_low']:.4g} to {summary['interval_high']:.4g}",
                              _number(summary["p_value"]), str(summary["n_inside"]), str(summary["n_after"]),
                              "yes" if summary["passes"] else "no"])
    lines += [
        "",
        "## Summary for the catalogue",
        "",
        "| Mode | Prompt | Inside | After | Difference | Interval | p-value | n inside | n after | Passes |",
        "|---|---|---|---|---|---|---|---|---|---|",
        *(_row(row) for row in summaries),
        "",
        "## Method",
        "",
        f"- Unit of analysis: the month. Month estimate: {MONTH_ESTIMATE}.",
        f"- Groups: {GROUPS}.",
        f"- Independence: {INDEPENDENCE}.",
        f"- Interval: moving-block percentile bootstrap at confidence {settings['confidence']:g} with "
        f"{settings['n_resamples']} resamples and seed {settings['seed']}; the plan's block length is "
        f"{settings['block_length']} months, and a cell in which a group has fewer months uses block length 1.",
        "- Test: one-sided block permutation test of the mean difference, blocks = months; null hypothesis: the two "
        "groups have the same mean; alternative: the mean inside is higher; test statistic: the difference; "
        f"{settings['n_permutations']} permutations, seed {settings['seed']}. The four tests are reported "
        "unadjusted; the qualification rule reads the identifying prompt of one mode at a time.",
        f"- Detectable effect: at power {settings['detectable_power']:g} and two-sided alpha "
        f"{settings['detectable_alpha']:g}, (z_0.975 + z_0.80) x the bootstrap standard error of the difference.",
        f"- Rule: {PASS_RULE}; {data['definitions']['rule_prompt']}.",
        f"- Missing: a cell with fewer than {settings['minimum_months_per_group']} months in either group shows no "
        "value and names what is missing.",
    ]
    notes = [f"- Block length, {cell['catalogue_mode']} ({cell['mode']}), {cell['prompt_kind']} prompt: "
             f"{cell['block_length_note']}." for cell in data["cells"] if cell["block_length_note"] is not None]
    lines += [*notes, "- Derived statistics only: no item, reply or licensed series is listed.", ""]
    return "\n".join(lines)


# ---------- the written outputs ----------

def cutoff_check_path(arm: str, out_dir: str | Path = ANALYSIS_DIR) -> Path:
    """``<out_dir>/cutoff-check-<arm>.json``; its Markdown table sits beside it with ``.md``."""
    if arm not in ARMS:
        raise CutoffCheckError(f"unknown arm {arm!r} for the cutoff check")
    return Path(out_dir) / f"cutoff-check-{arm}.json"


def _write_text(path: Path, text: str) -> None:
    """Write atomically: a crash leaves the previous file or the new one, never half of one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _read_text(path: Path) -> str:
    with path.open(encoding="utf-8", newline="") as stream:
        return stream.read()


def write_cutoff_check(check: CutoffCheckResult, out_dir: str | Path = ANALYSIS_DIR) -> tuple[Path, Path]:
    """Write the JSON document and the Markdown table once; the same outputs are a no-op, different ones refused."""
    if not isinstance(check, CutoffCheckResult):
        raise TypeError(f"check must be a CutoffCheckResult; got {type(check).__name__}")
    json_path = cutoff_check_path(check.arm, out_dir)
    markdown_path = json_path.with_suffix(".md")
    texts = {json_path: check.document, markdown_path: check.to_markdown()}
    for path, text in texts.items():
        if path.exists() and _read_text(path) != text:
            raise CutoffCheckError(f"refusing to replace the different cutoff check at {path}")
    for path, text in texts.items():
        if not path.exists():
            _write_text(path, text)
    return json_path, markdown_path


def main(argv: Sequence[str] | None = None) -> int:
    """Write the cutoff check of one arm from the private results of the frozen plan.

    Exit codes: 0 written (or already written identically), 2 refused (another
    plan, results of another arm, or different outputs at the path).
    """
    parser = argparse.ArgumentParser(description="Recall estimate inside and after the training cutoff, from "
                                                 "recorded results; sends no request.")
    parser.add_argument("--plan", type=Path, required=True, help="the frozen plan.json")
    parser.add_argument("--plan-sha256", required=True, help="the plan hash the results carry")
    parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=ANALYSIS_DIR)
    args = parser.parse_args(argv)
    if len(args.plan_sha256) != _SHA256_LENGTH:
        print("cutoff check: refused: --plan-sha256 must be 64 hex characters", file=sys.stderr)
        return EXIT_REFUSED
    try:
        plan = load_plan(args.plan, expected_sha256=args.plan_sha256)
        paths = write_cutoff_check(cutoff_check_from_records(plan, arm=args.arm, results_dir=args.results_dir),
                                   args.out_dir)
    except (ValueError, TypeError, OSError) as exc:
        print(f"cutoff check: refused: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    for path in paths:
        print(path)
    return EXIT_WRITTEN


__all__ = [
    "ANALYSIS_DIR",
    "CATALOGUE_MODES",
    "CUTOFF_SCHEMA",
    "MINIMUM_MONTHS",
    "MODES",
    "PASS_ALPHA",
    "PROMPT_KINDS",
    "RULE_PROMPT",
    "STATISTIC_NAMES",
    "CutoffCell",
    "CutoffCheckError",
    "CutoffCheckResult",
    "CutoffSummary",
    "cutoff_check",
    "cutoff_check_from_records",
    "cutoff_check_path",
    "main",
    "write_cutoff_check",
]


if __name__ == "__main__":
    raise SystemExit(main())
