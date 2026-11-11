"""Logprobs against logits on the same weights and prompts (qualified-model-catalogue 4.3 to 4.7, 6).

The research question: on the same weights and the same prompts, do logits
(estimate source ``full_distribution``) improve the recall correction that
logprobs (``top_k``) give, or can they stand in. This module answers it from
private records alone and sends no request:

* the two calibrator records of one arm and one corpus kind, each with its
  held-out labels and probabilities, its held-out prompts and its identity;
* the calibration draws, which say which held-out rows belong to which prompt
  when a prompt lost draws (without them the rows of a prompt can be told
  apart only when every held-out prompt kept all its draws);
* the stored item results of the arm's run on that kind (full run for macro);
* the written convergence table of that run, which is read, not recomputed.

Every reported value is a :class:`~recall_guard.core.significance.Statistic`
with its estimand, formula, inputs with counts, unit of analysis, n and whether
the units are independent (6.1 to 6.3). A statistic whose input is absent has
no value and names the input (6.9).

Statistics
    ``holdout_auc_<mode>``: the held-out AUC of each mode over the held-out
    draws, with the prompt-clustered interval and the label permutation test at
    prompt level. ``paired_auc_difference``: logits minus logprobs on the same
    held-out prompts and draws, both AUCs recomputed on shared prompt
    resamples, with its test and detectable effect; it is computed only when
    both calibrators hold out the same prompts and draws.
    ``estimate_agreement_*``: agreement of the two recall estimates prompt by
    prompt (mean difference, limits of agreement, mean absolute difference,
    rank correlation). A prompt is one item (month) in one rendering; its
    estimate in a mode is the mean of that mode's per-draw estimates over the
    draws that carry an estimate in both modes (:data:`PROMPT_ESTIMATE`, the
    per-prompt quantity of the cutoff check). n is the number of prompts; the
    intervals resample months. ``correction_factor_difference`` (4.4): the
    difference between the modes in the correction factor the consumer applies
    to the tilt, ``1 - p_memorized`` of the stored ensemble decision
    (:data:`CORRECTION_FACTOR`), logits minus logprobs, over the items with an
    ensemble estimate in both modes, resampled by month;
    ``correction_factor_mean_<mode>``: the mean factor of each mode over the
    same items. The agreement and the correction factor use different
    estimates and so have different n (:data:`ESTIMATES_COMPARED`).
    ``convergence_<evidence>_k<k>``: the mean
    absolute difference between the estimate from k kept alternatives and the
    full-distribution estimate, as the convergence analysis recorded it.

Unit of analysis
    The calibrator statistics resample prompts: the draws of a prompt share its
    label and text. The run statistics have one row per prompt and resample
    months: the two renderings of a month share the month's data. The month
    bootstrap treats months as exchangeable; serial dependence between months
    is not modelled.

Answer (4.6)
    ``improve`` when the interval of the paired AUC difference excludes 0 in
    favour of logits on the financial time-series boundary (corpus kind macro)
    and the cutoff check passed for logits; the cutoff result is an input, and
    without it ``improve`` is not concluded. ``match`` when that interval lies
    within :data:`AUC_MARGIN` either side of 0 and both limits of agreement of
    the prompt estimates lie within :data:`AGREEMENT_MARGIN` either side of 0
    by the outer bounds of their intervals: the interval low of the lower
    limit and the interval high of the upper limit (:data:`MATCH_RULE`).
    Otherwise ``undecided``, with the detectable effect of the paired
    difference at its sample size (6.11).

Private records
    Every record is opened for reading only. The calibration draw log is read
    through :func:`~scripts.finance_sim.recording.read_draw_log`, never through
    the recording's own reader, which repairs a log in place.

Refusal (4.7)
    When the two calibrators carry different weights digests, or one carries
    none, the comparison is refused with the reason and nothing is computed.

The outputs, ``mode-comparison-<arm>.json`` and ``.md``, hold derived
statistics only: no reply text, no licensed value, no item and no prompt.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from recall_guard.core.significance import (
    AGREEMENT_STATISTICS,
    Statistic,
    agreement,
    clustered_auc_interval,
    paired_auc_difference,
    permutation_auc_test,
    rank_auc,
)
from scripts.finance_sim.calibrate import (
    CALIBRATION_STAGE,
    CALIBRATORS_DIR,
    MACRO,
    CalibrationDraw,
    CalibrationRecord,
    calibrator_path,
    load_calibrator,
)
from scripts.finance_sim.convergence import HOST_TOP_K, SERVED_TOP_K, TABLE_SCHEMA, table_path
from scripts.finance_sim.execute import (
    EXECUTION_STAGES,
    RESULTS_DIR,
    ItemResult,
    load_result,
    planned_keys,
    result_path,
)
from scripts.finance_sim.limits import ARMS, RECORDS_DIR, RUNS_ROOT, SAMPLING_ARM
from scripts.finance_sim.plan import FULL_DISTRIBUTION, TOP_K, StudyPlan, ensure_unchanged, require_complete
from scripts.finance_sim.recording import draw_log_path, read_draw_log

COMPARISON_SCHEMA = "finance-sim-mode-comparison/1"
ANALYSIS_DIR = RUNS_ROOT / "analysis"

#: The two evidence modes and the estimate source each one is.
LOGPROBS = "logprobs"
LOGITS = "logits"
MODE_SOURCES: Mapping[str, str] = {LOGPROBS: TOP_K, LOGITS: FULL_DISTRIBUTION}
#: The stored variant that holds the ensemble decision of each mode: its ``p_memorized`` is what the consumer applies.
ENSEMBLE_VARIANTS: Mapping[str, str] = {mode: f"ensemble_{source}" for mode, source in MODE_SOURCES.items()}

#: The three answers (4.6).
IMPROVE = "improve"
MATCH = "match"
UNDECIDED = "undecided"

#: ``match`` needs the interval of the paired AUC difference within this margin either side of 0 ...
AUC_MARGIN = 0.05
#: ... and both limits of agreement of the prompt estimates within this margin either side of 0, checked on the
#: outer bounds of their intervals.
AGREEMENT_MARGIN = 0.10

#: The quantity the correction-factor statistics compare (4.4: the quantity the consumer uses).
CORRECTION_FACTOR = (
    "the correction factor of an item (a month in one rendering) in a mode is 1 minus the ensemble recall estimate: "
    "1 - p_memorized of that mode's stored ensemble decision, the factor the consumer applies to the tilt; "
    "p_memorized is the mean of the mode's per-draw estimates over the parsed draws of the item that carry one, and "
    "the decision exists only when the ensemble's minimum of parsed draws is met; an item without an ensemble "
    "decision or without an estimate in either mode is excluded and counted; an item whose decision carries the "
    "estimate but skipped the adjustment for a weak calibrator is included and counted; the difference is logits minus "
    "logprobs, so a positive value means logits soften the tilt less; its interval resamples months whole"
)
#: Why the agreement and the correction factor have different n (stated once in the outputs).
ESTIMATES_COMPARED = (
    "the agreement uses the per-prompt mean estimate (the mean of the per-draw estimates over the draws that carry "
    "both), which covers more prompts and is independent of the ensemble's minimum of parsed draws; the correction "
    "factor uses the ensemble estimate (p_memorized of the stored ensemble decision), the value the consumer "
    "applies; the two therefore have different n"
)
#: The quantity the agreement compares (4.3: prompt by prompt).
PROMPT_ESTIMATE = (
    "a prompt is one item (a month) in one rendering; its recall estimate in a mode is the mean of that mode's "
    "per-draw recall estimates over the draws of the prompt that carry an estimate in both modes, so the two means "
    "are over the same draws; this is the per-prompt quantity of the cutoff check, whose item estimate is the mean "
    "of the per-draw estimates; a prompt with no draw that carries both estimates is excluded and counted; n is "
    "the number of prompts, and the intervals resample months whole, so the two renderings of a month stay together"
)
#: How ``match`` reads the agreement (4.6).
MATCH_RULE = (
    f"match needs the interval of the paired AUC difference within {AUC_MARGIN:g} either side of 0 and both limits "
    f"of agreement of the prompt estimates within {AGREEMENT_MARGIN:g} either side of 0, checked on the outer "
    f"interval bounds of the two limits: the interval low of the lower limit is at least -{AGREEMENT_MARGIN:g} and "
    f"the interval high of the upper limit is at most {AGREEMENT_MARGIN:g}; the point values of the limits alone "
    "do not make a match"
)

#: Resampling settings; the AUC interval uses those of the calibration record, so it reproduces it.
AUC_RESAMPLES = 1000
PERMUTATIONS = 2000
PAIRED_RESAMPLES = 2000
AGREEMENT_RESAMPLES = 2000
CONFIDENCE = 0.95
SEED = 0

_AUC_FORMULA = ("(R_memorized - n_memorized_rows (n_memorized_rows + 1) / 2) / (n_memorized_rows n_control_rows), "
                "R_memorized the sum of the average ranks of the memorized rows among all held-out rows")
_CONVERGENCE_NAME = "convergence_by_alternatives_kept"
_CONVERGENCE_FORMULA = "sum(abs(estimate_k,i - estimate_full,i)) / n_draws, read from the convergence analysis"


class ModeComparisonError(ValueError):
    """Inputs that do not belong together, or an output that would replace a different one."""


# ---------- the statistic records ----------

def _missing(name: str, estimand: str, formula: str, unit: str, what: str,
             inputs: Mapping[str, float | int] | None = None, n: int = 0) -> Statistic:
    """A statistic without a value that names its missing input (6.9)."""
    return Statistic(name=name, estimand=estimand, formula=formula, inputs=inputs or {}, unit=unit, n=n,
                     independent_units=False, value=None, interval=None, test=None, effect_size=None,
                     detectable_effect=None, by_construction=False, missing=what)


def weights_identity(record: CalibrationRecord) -> str | None:
    """The weights digest a calibrator's identity carries: the observed one, else the declared one, else None."""
    local = record.calibrator.identity.local_identity
    if local is None:
        return None
    return local.observed.artifact_digest or local.declared.artifact_digest


# ---------- the held-out rows of a calibrator ----------

@dataclass(frozen=True)
class _Holdout:
    """The held-out rows of one calibrator: score, label and prompt index per row, and which draw each row is."""

    scores: np.ndarray
    labels: np.ndarray
    clusters: np.ndarray
    keys: tuple[tuple[str, int], ...]

    @property
    def n_prompts(self) -> int:
        return int(self.clusters.max()) + 1

    def both_classes(self) -> bool:
        return 0 < int(self.labels.sum()) < self.labels.size


def _holdout(mode: str, record: CalibrationRecord | None,
             draws: tuple[CalibrationDraw, ...] | None) -> _Holdout | str:
    """The held-out rows of ``record`` by prompt, or the input that is missing.

    The record keeps the held-out rows prompt by prompt in the order of
    ``holdout_prompts``, each prompt with its usable draws in draw order. With
    the calibration draws the usable draws of each prompt are read from them;
    without, the rows are assigned only when every held-out prompt kept all
    its draws. Calibration draws that do not give the held-out rows are a
    missing input like any other (6.9): the calibrator statistics are then
    reported missing, and the statistics of the run are still computed.
    """
    source = MODE_SOURCES[mode]
    if record is None:
        return f"the calibrator record of the {mode} mode ({source})"
    mcs = record.calibrator.mcs
    if mcs.holdout_labels is None or mcs.holdout_probabilities is None:
        return f"the held-out labels and probabilities of the {mode} calibrator"
    labels = np.asarray(mcs.holdout_labels, dtype=np.int64)
    scores = np.asarray(mcs.holdout_probabilities, dtype=np.float64)
    prompts = record.holdout_prompts
    per_prompt = record.draw_settings.draws
    if draws is not None:
        usable: dict[str, list[int]] = {}
        for draw in draws:
            if (draw.kind == record.kind and draw.failure_code is None
                    and draw.features.get(source) is not None):
                usable.setdefault(draw.prompt_sha256, []).append(draw.draw_index)
        indices = [sorted(usable.get(prompt, ())) for prompt in prompts]
        if sum(len(found) for found in indices) != labels.size or not all(indices):
            return (f"calibration draws that give the held-out rows of the {mode} calibrator: the calibration "
                    f"draws do not give the held-out rows ({sum(len(found) for found in indices)} usable draws on "
                    f"its {len(prompts)} held-out prompts, {labels.size} held-out rows in the record)")
    elif labels.size == len(prompts) * per_prompt:
        indices = [list(range(per_prompt))] * len(prompts)
    else:
        return (f"the calibration draws: the {labels.size} held-out rows of the {mode} calibrator are not "
                f"{per_prompt} draws on each of its {len(prompts)} held-out prompts, so the rows of a prompt "
                "cannot be told apart from the record alone")
    label_of = {entry.prompt_sha256: entry.label for entry in record.entries}
    expected = [label_of[prompt] for prompt, found in zip(prompts, indices, strict=True) for _ in found]
    if expected != labels.tolist():
        return (f"held-out rows of the {mode} calibrator in the order of its held-out prompts: the calibration "
                "draws do not give the held-out rows (the held-out labels are not those of the held-out prompts "
                "in order)")
    clusters = np.asarray([position for position, found in enumerate(indices) for _ in found], dtype=np.int64)
    keys = tuple((prompt, index) for prompt, found in zip(prompts, indices, strict=True) for index in found)
    return _Holdout(scores=scores, labels=labels, clusters=clusters, keys=keys)


def _class_counts(rows: _Holdout) -> dict[str, int]:
    prompt_labels = [int(rows.labels[rows.clusters == position][0]) for position in range(rows.n_prompts)]
    return {"n_rows": int(rows.labels.size), "n_prompts": rows.n_prompts,
            "n_memorized_prompts": sum(prompt_labels), "n_control_prompts": rows.n_prompts - sum(prompt_labels),
            "n_memorized_rows": int(rows.labels.sum()), "n_control_rows": int(rows.labels.size - rows.labels.sum())}


def _auc_statistic(mode: str, rows: _Holdout | str) -> Statistic:
    name = f"holdout_auc_{mode}"
    estimand = (f"probability that a held-out draw of a memorized prompt gets a higher {mode} "
                f"({MODE_SOURCES[mode]}) recall estimate than a held-out draw of a control prompt (ROC-AUC), over "
                "the held-out draws of the calibrator")
    if isinstance(rows, str):
        return _missing(name, estimand, _AUC_FORMULA, "prompt", rows)
    counts = _class_counts(rows)
    if not rows.both_classes():
        return _missing(name, estimand, _AUC_FORMULA, "prompt", "held-out prompts of both classes", counts,
                        rows.n_prompts)
    positive, negative = rows.scores[rows.labels == 1], rows.scores[rows.labels == 0]
    value = rank_auc(positive, negative)
    interval = clustered_auc_interval(rows.scores, rows.labels, rows.clusters, n_resamples=AUC_RESAMPLES,
                                      confidence=CONFIDENCE, seed=SEED)
    test = permutation_auc_test(rows.scores, rows.labels, rows.clusters, n_permutations=PERMUTATIONS, seed=SEED)
    rank_sum = value * positive.size * negative.size + positive.size * (positive.size + 1) / 2
    return Statistic(
        name=name, estimand=estimand, formula=_AUC_FORMULA,
        inputs=counts | {"rank_sum_memorized": float(rank_sum), "n_resamples": AUC_RESAMPLES,
                         "n_permutations": PERMUTATIONS},
        unit="prompt", n=rows.n_prompts, independent_units=rows.labels.size == rows.n_prompts, value=value,
        interval=interval, test=test, effect_size=value - 0.5, detectable_effect=None, by_construction=False,
        missing=None)


def _pairing(logprobs: _Holdout | str, logits: _Holdout | str) -> str | None:
    """Why the two calibrators cannot be paired, or None when they hold out the same prompts and draws."""
    for rows in (logits, logprobs):
        if isinstance(rows, str):
            return rows
    if logprobs.keys != logits.keys:
        return ("held-out rows on the same prompts and draws in both modes: the logprobs calibrator holds out "
                f"{logprobs.n_prompts} prompts with {logprobs.labels.size} rows, the logits calibrator "
                f"{logits.n_prompts} prompts with {logits.labels.size} rows, and they are not the same draws")
    return None


def _paired_statistic(logprobs: _Holdout | str, logits: _Holdout | str, unpaired: str | None) -> Statistic:
    name = "paired_auc_difference"
    estimand = ("held-out AUC of the logits (full_distribution) estimate minus held-out AUC of the logprobs "
                "(top_k) estimate, over the same held-out prompts and draws on the same weights")
    formula = ("AUC(logits) - AUC(logprobs), each AUC over all held-out rows; interval and test from prompt "
               "resamples stratified by label, both AUCs recomputed on every resample")
    if unpaired is not None:
        return _missing(name, estimand, formula, "prompt", unpaired)
    counts = _class_counts(logits)
    if not logits.both_classes():
        return _missing(name, estimand, formula, "prompt", "held-out prompts of both classes", counts,
                        logits.n_prompts)
    value, interval, test = paired_auc_difference(logits.scores, logprobs.scores, logits.labels, logits.clusters,
                                                  n_resamples=PAIRED_RESAMPLES, confidence=CONFIDENCE, seed=SEED)
    return Statistic(
        name=name, estimand=estimand, formula=formula,
        inputs=counts | {"auc_logits": float(test.parameters["auc_a"]),
                         "auc_logprobs": float(test.parameters["auc_b"]), "n_resamples": PAIRED_RESAMPLES,
                         "standard_error": float(test.parameters["standard_error"])},
        unit="prompt", n=logits.n_prompts, independent_units=logits.labels.size == logits.n_prompts, value=value,
        interval=interval, test=test, effect_size=value,
        detectable_effect=float(test.parameters["detectable_effect"]), by_construction=False, missing=None)


# ---------- the stored item results ----------

def _checked_results(plan: StudyPlan, arm: str, kind: str, results: Iterable[ItemResult] | None,
                     records: Mapping[str, CalibrationRecord | None]) -> tuple[ItemResult, ...] | None:
    if results is None:
        return None
    if isinstance(results, str | bytes) or not isinstance(results, Iterable):
        raise TypeError("results must be a sequence of ItemResult values or None")
    checked = tuple(results)
    for result in checked:
        if not isinstance(result, ItemResult):
            raise TypeError(f"results must be ItemResult values; got {type(result).__name__}")
        ensure_unchanged(plan, result.plan_sha256)
        if result.arm != arm or result.kind != kind or result.stage == "pilot":
            raise ModeComparisonError(f"a result of the {result.arm} arm's {result.stage} stage ({result.kind} "
                                      f"items) is not a {kind} result of the {arm} arm outside the pilot")
        for mode, record in records.items():
            identity = result.identities.get(MODE_SOURCES[mode])
            if (record is not None and identity is not None
                    and identity.calibrator_fingerprint != record.calibrator.fingerprint):
                raise ModeComparisonError(f"a stored result was scored with another {mode} calibrator than the "
                                          "compared one; the modes compare only on the same calibrators")
    return checked or None


@dataclass(frozen=True)
class _PromptEstimates:
    """The recall estimate of each prompt of the run in both modes (:data:`PROMPT_ESTIMATE`), with its month."""

    logits: tuple[float, ...]
    logprobs: tuple[float, ...]
    months: tuple[str, ...]
    counts: Mapping[str, int]
    missing: str | None


def _prompt_estimates(results: tuple[ItemResult, ...] | None) -> _PromptEstimates:
    """One row per stored prompt with a draw that carries both estimates: the two means over those draws."""
    logits: list[float] = []
    logprobs: list[float] = []
    months: list[str] = []
    n_draws = 0
    for result in results or ():
        if result.full_distribution is None:
            continue
        pairs = [(entry.estimate, draw.estimate)
                 for draw, entry in zip(result.recorded_draws, result.full_distribution, strict=True)
                 if draw.estimate is not None and entry is not None and entry.estimate is not None]
        if not pairs:
            continue
        logits.append(float(np.mean([pair[0] for pair in pairs])))
        logprobs.append(float(np.mean([pair[1] for pair in pairs])))
        months.append(result.item_id)
        n_draws += len(pairs)
    stored = len(results or ())
    counts = {"n_prompts": len(logits), "n_months": len(set(months)), "n_draws": n_draws,
              "n_stored_prompts": stored, "prompts_without_a_draw_with_both_estimates": stored - len(logits)}
    if results is None:
        missing = "the stored item results of the run"
    elif not logits:
        missing = "stored item results with a draw that carries both estimates"
    else:
        missing = None
    return _PromptEstimates(tuple(logits), tuple(logprobs), tuple(months), counts, missing)


def _per_prompt(statistic: Statistic, rows: _PromptEstimates, **changes: Any) -> Statistic:
    """``statistic`` of :func:`agreement` reported per prompt: n is the prompts, the months are in the inputs."""
    return replace(statistic, unit="prompt", n=len(rows.logits),
                   independent_units=bool(rows.logits) and len(rows.logits) == len(set(rows.months)),
                   inputs=dict(statistic.inputs) | dict(rows.counts),
                   missing=rows.missing if rows.missing is not None else statistic.missing, **changes)


def _agreement_statistics(rows: _PromptEstimates, stage: str) -> list[Statistic]:
    """Agreement of the two recall estimates prompt by prompt (4.3), resampled by month."""
    described = (f"a row is a prompt (item and rendering) of the {stage} stage; a = the mean of its logits "
                 "(full_distribution) per-draw recall estimates, b = the mean of its logprobs (top_k) per-draw "
                 "recall estimates, both over the draws of the prompt that carry an estimate in both modes; "
                 "interval from resampling months whole")
    found = agreement(rows.logits or None, rows.logprobs or None, rows.months or None,
                      n_resamples=AGREEMENT_RESAMPLES, confidence=CONFIDENCE, seed=SEED, unit="month")
    return [_per_prompt(found[name], rows, name=f"estimate_agreement_{name}",
                        estimand=f"{found[name].estimand}; {described}")
            for name in AGREEMENT_STATISTICS]


@dataclass(frozen=True)
class _EnsembleFactors:
    """The correction factor of each item of the run in both modes (:data:`CORRECTION_FACTOR`), with its month."""

    logits: tuple[float, ...]
    logprobs: tuple[float, ...]
    months: tuple[str, ...]
    counts: Mapping[str, int]
    missing: str | None


def _ensemble_factors(results: tuple[ItemResult, ...] | None) -> _EnsembleFactors:
    """One row per stored item whose ensemble decision carries a recall estimate in both modes: ``1 - p_memorized``.

    An item is excluded and counted when the ensemble variant of a mode was
    not computed (it carries only its reason) or its decision has no
    ``p_memorized`` (a failed decision, or no parsed draw with an estimate).
    An item whose decision carries the estimate but skipped the adjustment (a
    weak calibrator) is included with that estimate and counted.
    """
    factors: dict[str, list[float]] = {LOGITS: [], LOGPROBS: []}
    months: list[str] = []
    no_decision = no_estimate = not_applied = 0
    for result in results or ():
        variants = {mode: result.variant(name) for mode, name in ENSEMBLE_VARIANTS.items()}
        if any(variant.missing is not None for variant in variants.values()):
            no_decision += 1
        elif any(variant.p_memorized is None for variant in variants.values()):
            no_estimate += 1
        else:
            for mode, variant in variants.items():
                factors[mode].append(1.0 - float(variant.p_memorized))
            months.append(result.item_id)
            not_applied += any(variant.status != "ok" for variant in variants.values())
    counts = {"n_items": len(months), "n_months": len(set(months)), "n_stored_items": len(results or ()),
              "items_excluded": no_decision + no_estimate,
              "items_without_an_ensemble_decision_in_a_mode": no_decision,
              "items_without_an_ensemble_estimate_in_a_mode": no_estimate,
              "items_included_whose_adjustment_was_skipped_in_a_mode": not_applied}
    if results is None:
        missing = "the stored item results of the run"
    elif not months:
        missing = ("stored item results with an ensemble decision that carries a recall estimate in both modes "
                   f"(of {len(results)} stored items, {no_decision} have no ensemble decision in a mode and "
                   f"{no_estimate} an ensemble decision without an estimate)")
    else:
        missing = None
    return _EnsembleFactors(tuple(factors[LOGITS]), tuple(factors[LOGPROBS]), tuple(months), counts, missing)


def _correction_statistics(rows: _EnsembleFactors, stage: str) -> list[Statistic]:
    """The correction factor the consumer applies (4.4): the difference between the modes, and each mode's mean.

    The factor of an item in a mode is ``1 - p_memorized`` of that mode's
    stored ensemble decision. The difference is resampled by month; the two
    means are plain statistics over the same items, for the level.
    """
    items = (f"over the items (month and rendering) of the {stage} stage whose stored ensemble decision carries a "
             "recall estimate (p_memorized) in both modes")
    difference_name = "correction_factor_difference"
    difference_estimand = (
        "difference between the modes in the correction factor the consumer applies to the tilt, 1 minus the "
        f"ensemble recall estimate: mean {items} of the factor with the logits (full_distribution) estimate minus "
        "the factor with the logprobs (top_k) estimate; logits minus logprobs: a positive value means logits "
        "soften the tilt less; an item without an ensemble decision or without an estimate in a mode is excluded "
        "and counted; interval from resampling months whole, so the renderings of a month stay together")
    difference_formula = "sum((1 - p_memorized_logits,i) - (1 - p_memorized_logprobs,i)) / n_items"
    mean_formula = "sum(1 - p_memorized_i) / n_items"
    mean_estimands = {
        mode: (f"mean correction factor of the {mode} ({MODE_SOURCES[mode]}) mode, 1 minus the ensemble recall "
               f"estimate, {items}: the same items as correction_factor_difference; the level beside the "
               "difference, reported without an interval")
        for mode in (LOGITS, LOGPROBS)}
    if rows.missing is not None:
        return [_missing(difference_name, difference_estimand, difference_formula, "prompt", rows.missing,
                         rows.counts),
                *(_missing(f"correction_factor_mean_{mode}", mean_estimands[mode], mean_formula, "prompt",
                           rows.missing, rows.counts) for mode in (LOGITS, LOGPROBS))]
    n_items = len(rows.months)
    independent = n_items == len(set(rows.months))
    found = agreement(rows.logits, rows.logprobs, rows.months, n_resamples=AGREEMENT_RESAMPLES,
                      confidence=CONFIDENCE, seed=SEED, unit="month")["mean_difference"]
    statistics = [replace(found, name=difference_name, estimand=difference_estimand, formula=difference_formula,
                          unit="prompt", n=n_items, independent_units=independent,
                          inputs=dict(found.inputs) | dict(rows.counts))]
    for mode, factors in ((LOGITS, rows.logits), (LOGPROBS, rows.logprobs)):
        statistics.append(Statistic(
            name=f"correction_factor_mean_{mode}", estimand=mean_estimands[mode], formula=mean_formula,
            inputs=dict(rows.counts) | {"sum_of_factors": float(np.sum(factors))}, unit="prompt", n=n_items,
            independent_units=independent, value=float(np.mean(factors)), interval=None, test=None,
            effect_size=None, detectable_effect=None, by_construction=False, missing=None))
    return statistics


# ---------- convergence, read from the convergence analysis ----------

def _convergence_statistics(plan: StudyPlan, arm: str, kind: str, weights: str | None,
                            convergence: Mapping[str, Any] | None) -> list[Statistic]:
    estimand = ("mean absolute difference between the recall estimate from k kept alternatives per token and the "
                "full-distribution estimate, over the compared draws of the run")

    def absent(what: str) -> list[Statistic]:
        return [_missing(_CONVERGENCE_NAME, estimand, _CONVERGENCE_FORMULA, "draw", what)]

    if convergence is None:
        return absent("the convergence analysis of the run (its written table)")
    if not isinstance(convergence, Mapping) or convergence.get("schema") != TABLE_SCHEMA:
        raise ModeComparisonError(f"convergence must be a {TABLE_SCHEMA} table or None")
    ensure_unchanged(plan, convergence["plan_sha256"])
    if convergence["arm"] != arm:
        raise ModeComparisonError(f"the convergence table is of the {convergence['arm']} arm, not the {arm} arm")
    if convergence["kind"] != kind:
        return absent(f"a convergence analysis on {kind} items (the table given is on {convergence['kind']} items)")
    if weights is not None and convergence["model_digest"] != weights:
        return absent("a convergence analysis on the same weights (the table given is on other weights)")
    rows = [row for row in convergence["features"] if row["feature"] == "estimate"]
    if not rows:
        return absent("the estimate rows of the convergence analysis")
    how = {HOST_TOP_K: "the first k alternatives of the full-distribution evidence through the logits calibrator",
           SERVED_TOP_K: "the generation host's own k logprobs through the logprobs calibrator"}
    statistics = []
    for row in rows:
        name = f"convergence_{row['evidence']}_k{row['k']}"
        described = f"{estimand}; k = {row['k']}, estimate from {how.get(row['evidence'], row['evidence'])}"
        inputs = {"k": int(row["k"]), "n_draws": int(row["n"]), "kept": int(convergence["kept"])}
        if row["mean_abs_diff"] is None:
            what = row["missing_reason"] or "compared draws at this number of alternatives"
            statistics.append(_missing(name, described, _CONVERGENCE_FORMULA, "draw", what, inputs, int(row["n"])))
            continue
        statistics.append(Statistic(
            name=name, estimand=described, formula=_CONVERGENCE_FORMULA, inputs=inputs, unit="draw",
            n=int(row["n"]), independent_units=False, value=float(row["mean_abs_diff"]), interval=None, test=None,
            effect_size=None, detectable_effect=None,
            by_construction=row["evidence"] == HOST_TOP_K and row["k"] == convergence["kept"], missing=None))
    return statistics


# ---------- the answer ----------

def _decide(kind: str, calibrator: tuple[Statistic, ...], lower: Statistic, upper: Statistic,
            cutoff_check: bool | None) -> tuple[str, str, float | None]:
    """One of the three answers, its reason and the detectable effect of the paired difference (4.6).

    ``calibrator`` holds the calibrator statistics, the paired AUC difference
    last. ``match`` is decided on intervals throughout: the interval of the
    paired difference, and for the agreement the outer bounds of the intervals
    of the two limits (the low of the lower limit, the high of the upper one),
    because a claim that logits can stand in must hold with the uncertainty of
    the limits, not only at their point values.
    """
    paired = calibrator[-1]
    if paired.value is None or paired.interval is None:
        absent = [statistic for statistic in calibrator if statistic.value is None]
        names = ", ".join(statistic.name for statistic in absent)
        what = "; ".join(dict.fromkeys(str(statistic.missing) for statistic in absent))
        return UNDECIDED, f"{names} cannot be computed; missing: {what}", None
    lo, hi = paired.interval.lo, paired.interval.hi
    interval = f"{lo:.4g} to {hi:.4g}"
    notes = []
    if lo > 0.0:
        favours = f"the interval {interval} of the paired AUC difference excludes 0 in favour of logits"
        if kind != MACRO:
            notes.append(f"{favours}, but improve is concluded on the financial time-series boundary (macro) only")
        elif cutoff_check is True:
            return IMPROVE, f"{favours} on the macro boundary and the cutoff check passed for logits", \
                paired.detectable_effect
        elif cutoff_check is None:
            notes.append(f"{favours}, but the cutoff check for logits was not supplied, so improve cannot be "
                         "concluded")
        else:
            notes.append(f"{favours}, but the cutoff check for logits did not pass")
    within = -AUC_MARGIN <= lo and hi <= AUC_MARGIN
    if not within:
        notes.append(f"the interval {interval} of the paired AUC difference is not within {AUC_MARGIN:g} either "
                     "side of 0" + ("" if lo > 0.0 else " and does not exclude 0 in favour of logits"))
    elif lower.value is None or upper.value is None:
        notes.append(f"the limits of agreement cannot be computed; missing: {lower.missing or upper.missing}")
    elif lower.interval is None or upper.interval is None:
        notes.append("the intervals of the limits of agreement cannot be computed (no month resample gave a "
                     "defined limit), so the limits cannot be checked on their interval bounds")
    else:
        limits = (f"the limits of agreement of the prompt estimates, {lower.value:.4g} to {upper.value:.4g}, "
                  f"checked on the outer bounds of their intervals (interval low of the lower limit "
                  f"{lower.interval.lo:.4g}, interval high of the upper limit {upper.interval.hi:.4g})")
        if -AGREEMENT_MARGIN <= lower.interval.lo and upper.interval.hi <= AGREEMENT_MARGIN:
            return MATCH, (f"the interval {interval} of the paired AUC difference is within {AUC_MARGIN:g} either "
                           f"side of 0 and {limits} are within {AGREEMENT_MARGIN:g} either side of 0"), \
                paired.detectable_effect
        notes.append(f"{limits} are not within {AGREEMENT_MARGIN:g} either side of 0")
    return UNDECIDED, "; ".join(notes), paired.detectable_effect


# ---------- the comparison ----------

def _statistic_dict(statistic: Statistic) -> dict[str, Any]:
    test = statistic.test
    return {
        "name": statistic.name, "estimand": statistic.estimand, "formula": statistic.formula,
        "inputs": dict(statistic.inputs), "unit": statistic.unit, "n": statistic.n,
        "independent_units": statistic.independent_units, "value": statistic.value,
        "interval": None if statistic.interval is None else asdict(statistic.interval),
        "test": None if test is None else {
            "statistic": test.statistic, "p_value": test.p_value, "method": test.method,
            "parameters": dict(test.parameters),
            "interval": None if test.interval is None else asdict(test.interval)},
        "effect_size": statistic.effect_size, "detectable_effect": statistic.detectable_effect,
        "by_construction": statistic.by_construction, "missing": statistic.missing,
    }


@dataclass(frozen=True)
class ModeComparison:
    """The comparison of one arm and corpus kind: the answer with its statistics, or the refusal with its reason.

    ``refusal`` is set, and ``answer`` and ``statistics`` are empty, when the
    two modes do not carry the same weights (4.7). ``pairing`` is the reason
    the calibrators are not paired, or None when they are. ``cutoff_check`` is
    the cutoff result for logits the caller supplied, or None.
    ``incomplete_draw_log_lines`` counts the incomplete last lines of the
    calibration draw log that the read-only reader ignored.
    """

    plan_sha256: str
    arm: str
    kind: str
    stage: str
    model: str | None
    weights: str | None
    refusal: str | None
    answer: str | None
    reason: str | None
    detectable_effect: float | None
    cutoff_check: bool | None
    pairing: str | None
    statistics: tuple[Statistic, ...]
    incomplete_draw_log_lines: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": COMPARISON_SCHEMA, "plan_sha256": self.plan_sha256, "arm": self.arm, "kind": self.kind,
            "stage": self.stage, "model": self.model, "weights": self.weights, "modes": dict(MODE_SOURCES),
            "refusal": self.refusal, "answer": self.answer, "reason": self.reason,
            "detectable_effect": self.detectable_effect, "cutoff_check": self.cutoff_check, "pairing": self.pairing,
            "margins": {"auc_difference": AUC_MARGIN, "agreement_limits": AGREEMENT_MARGIN},
            "definitions": {"prompt_estimate": PROMPT_ESTIMATE, "correction_factor": CORRECTION_FACTOR,
                            "estimates_compared": ESTIMATES_COMPARED, "match_rule": MATCH_RULE},
            "calibration_draw_log": {"incomplete_last_lines_ignored": self.incomplete_draw_log_lines},
            "settings": {"auc_resamples": AUC_RESAMPLES, "permutations": PERMUTATIONS,
                         "paired_resamples": PAIRED_RESAMPLES, "agreement_resamples": AGREEMENT_RESAMPLES,
                         "confidence": CONFIDENCE, "seed": SEED},
            "statistics": [_statistic_dict(statistic) for statistic in self.statistics],
        }

    @property
    def document(self) -> str:
        """The JSON file: sorted keys and LF newlines, so a replay can compare bytes."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n"

    def to_markdown(self) -> str:
        return _markdown(self)


def mode_comparison(
    plan: StudyPlan, *, arm: str = SAMPLING_ARM, kind: str = MACRO, logprobs: CalibrationRecord | None,
    logits: CalibrationRecord | None, results: Iterable[ItemResult] | None,
    calibration_draws: Iterable[CalibrationDraw] | None = None, convergence: Mapping[str, Any] | None = None,
    cutoff_check: bool | None = None, incomplete_draw_log_lines: int = 0,
) -> ModeComparison:
    """Compare the two modes of one arm on one corpus kind from their records; no request is sent.

    Args:
        plan: The frozen plan every input must belong to.
        arm: The arm both calibrators were fitted for.
        kind: The corpus kind, ``macro`` (the financial time-series boundary) or ``edgar``.
        logprobs: The ``top_k`` calibrator record, or None when there is none.
        logits: The ``full_distribution`` calibrator record, or None when there is none.
        results: The stored item results of the arm's run on ``kind``, or None.
        calibration_draws: The recorded calibration draws, which give the prompt of each held-out row, or None.
        convergence: The written convergence table of that run (its JSON object), or None.
        cutoff_check: Whether the cutoff check passed for logits, or None when it was not supplied.
        incomplete_draw_log_lines: How many incomplete last lines the read-only reader of the calibration draw
            log ignored (0 or 1); it is stated in the outputs.

    Raises:
        ModeComparisonError: for inputs of another arm, kind or calibrator. Different weights are not an error:
            the comparison is returned refused. Calibration draws that do not give the held-out rows are not an
            error either: the calibrator statistics are reported missing and the answer is ``undecided``.
    """
    require_complete(plan)
    stages = [stage for stage, stage_kind in EXECUTION_STAGES.items() if stage_kind == kind and stage != "pilot"]
    if arm not in ARMS or len(stages) != 1:
        raise ModeComparisonError(f"unknown arm or corpus kind {arm!r}/{kind!r} for the mode comparison")
    (stage,) = stages
    if cutoff_check is not None and not isinstance(cutoff_check, bool):
        raise TypeError("cutoff_check must be True, False or None")
    if (isinstance(incomplete_draw_log_lines, bool) or not isinstance(incomplete_draw_log_lines, int)
            or incomplete_draw_log_lines < 0):
        raise TypeError("incomplete_draw_log_lines must be a non-negative integer")
    records = {LOGPROBS: logprobs, LOGITS: logits}
    for mode, record in records.items():
        if record is None:
            continue
        if not isinstance(record, CalibrationRecord):
            raise TypeError(f"{mode} must be a CalibrationRecord or None; got {type(record).__name__}")
        ensure_unchanged(plan, record.plan_sha256)
        if (record.arm, record.kind, record.source.value) != (arm, kind, MODE_SOURCES[mode]):
            raise ModeComparisonError(f"the {mode} record is the {record.arm} arm's {record.kind} calibrator of the "
                                      f"{record.source.value} source; the {kind} comparison of the {arm} arm needs "
                                      f"its {MODE_SOURCES[mode]} calibrator")
    present = [record for record in records.values() if record is not None]
    model = present[0].calibrator.identity.model if present else None
    digests = {mode: weights_identity(record) for mode, record in records.items() if record is not None}

    def refused(reason: str) -> ModeComparison:
        return ModeComparison(plan_sha256=plan.plan_sha256, arm=arm, kind=kind, stage=stage, model=model,
                              weights=None, refusal=reason, answer=None, reason=None, detectable_effect=None,
                              cutoff_check=cutoff_check, pairing=None, statistics=(),
                              incomplete_draw_log_lines=incomplete_draw_log_lines)

    if len(digests) == 2:
        for mode, digest in digests.items():
            if digest is None:
                return refused(f"the {mode} calibrator carries no weights digest, so the two modes are not shown "
                               "to be measured on the same weights; their values are not compared")
        if digests[LOGPROBS] != digests[LOGITS]:
            return refused(f"the two modes were measured on different weights (logprobs {digests[LOGPROBS]}, "
                           f"logits {digests[LOGITS]}); values compare only within one set of weights")
    weights = next((digest for digest in digests.values() if digest is not None), None)

    draws = None if calibration_draws is None else tuple(calibration_draws)
    checked = _checked_results(plan, arm, kind, results, records)
    rows = {mode: _holdout(mode, record, draws) for mode, record in records.items()}
    unpaired = _pairing(rows[LOGPROBS], rows[LOGITS])
    calibrator = (_auc_statistic(LOGPROBS, rows[LOGPROBS]), _auc_statistic(LOGITS, rows[LOGITS]),
                  _paired_statistic(rows[LOGPROBS], rows[LOGITS], unpaired))
    estimates = _prompt_estimates(checked)
    agreements = _agreement_statistics(estimates, stage)
    by_name = {statistic.name: statistic for statistic in agreements}
    answer, reason, detectable = _decide(kind, calibrator, by_name["estimate_agreement_lower_limit_of_agreement"],
                                         by_name["estimate_agreement_upper_limit_of_agreement"], cutoff_check)
    statistics = (*calibrator, *agreements, *_correction_statistics(_ensemble_factors(checked), stage),
                  *_convergence_statistics(plan, arm, kind, weights, convergence))
    return ModeComparison(plan_sha256=plan.plan_sha256, arm=arm, kind=kind, stage=stage, model=model,
                          weights=weights, refusal=None, answer=answer, reason=reason,
                          detectable_effect=detectable, cutoff_check=cutoff_check, pairing=unpaired,
                          statistics=statistics, incomplete_draw_log_lines=incomplete_draw_log_lines)


def mode_comparison_from_records(
    plan: StudyPlan, *, arm: str = SAMPLING_ARM, kind: str = MACRO, calibrators_dir: str | Path = CALIBRATORS_DIR,
    results_dir: str | Path = RESULTS_DIR, records_dir: str | Path = RECORDS_DIR,
    analysis_dir: str | Path = ANALYSIS_DIR, cutoff_check: bool | None = None,
) -> ModeComparison:
    """The comparison from the private records alone; an input that was not recorded is reported missing.

    Reads the arm's two calibrator files of ``kind``, the calibration stage's
    draw recording, the stored result of every planned item of the run on
    ``kind`` and the written convergence table of that run. Every private
    record is opened for reading only: the draw recording goes through
    :func:`~scripts.finance_sim.recording.read_draw_log`, which never repairs
    it, and an incomplete last line it ignored is stated in the outputs.
    """
    require_complete(plan)
    stages = [stage for stage, stage_kind in EXECUTION_STAGES.items() if stage_kind == kind and stage != "pilot"]
    if arm not in ARMS or len(stages) != 1:
        raise ModeComparisonError(f"unknown arm or corpus kind {arm!r}/{kind!r} for the mode comparison")
    (stage,) = stages
    records: dict[str, CalibrationRecord | None] = {}
    for mode, source in MODE_SOURCES.items():
        path = calibrator_path(arm, source, kind, calibrators_dir)
        offered = source in plan.arms.arms[arm].estimate_sources
        records[mode] = (load_calibrator(path, plan, arm=arm, kind=kind, source=source)
                         if offered and path.exists() else None)
    reading = read_draw_log(draw_log_path(arm, CALIBRATION_STAGE, records_dir), plan=plan,
                            loads=CalibrationDraw.from_dict)
    results = []
    for item_id, rendering in planned_keys(plan, arm, stage):
        path = result_path(arm, stage, item_id, rendering, results_dir)
        if path.exists():
            results.append(load_result(path))
    table = table_path(arm, stage, analysis_dir)
    convergence = json.loads(table.read_text(encoding="utf-8")) if table.exists() else None
    return mode_comparison(plan, arm=arm, kind=kind, logprobs=records[LOGPROBS], logits=records[LOGITS],
                           results=results or None, calibration_draws=reading.draws or None,
                           convergence=convergence, cutoff_check=cutoff_check,
                           incomplete_draw_log_lines=reading.incomplete_last_lines)


# ---------- the markdown table ----------

def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _number(value: float | int | None) -> str:
    if value is None:
        return "none"
    return str(value) if isinstance(value, int) else f"{value:.4g}"


def _row(statistic: Statistic) -> list[str]:
    if statistic.value is None:
        value = f"missing: {statistic.missing}"
    else:
        value = _number(statistic.value) + (" (equal by construction)" if statistic.by_construction else "")
    interval = statistic.interval
    interval_text = "none" if interval is None else (
        f"{interval.lo:.4g} to {interval.hi:.4g} ({interval.confidence * 100:g} percent, {interval.method})")
    test = statistic.test
    if test is None:
        test_text = "none: equal by construction" if statistic.by_construction else "none"
    else:
        test_text = f"p = {_number(test.p_value)}; statistic {_number(test.statistic)}; {test.method}"
    inputs = ", ".join(f"{name}={_number(value) if isinstance(value, int) else f'{value:.6g}'}"
                       for name, value in sorted(statistic.inputs.items()))
    return [statistic.name, value, interval_text, f"{statistic.n} ({statistic.unit})",
            "yes" if statistic.independent_units else "no", test_text, _number(statistic.effect_size),
            _number(statistic.detectable_effect), f"{statistic.formula}; inputs: {inputs or 'none'}",
            statistic.estimand]


def _markdown(comparison: ModeComparison) -> str:
    lines = [
        f"# Logprobs against logits: {comparison.arm} arm, {comparison.kind} corpus",
        "",
        f"Logprobs is the {TOP_K} estimate source, logits the {FULL_DISTRIBUTION} one. Model {comparison.model}, "
        f"weights {comparison.weights}, frozen plan {comparison.plan_sha256}. Item results: the "
        f"{comparison.stage} stage.",
        "",
    ]
    if comparison.refusal is not None:
        return "\n".join([*lines, f"Refused: {comparison.refusal}.", "", "No value is compared (Requirement 4.7).",
                          ""])
    cutoff = {None: "not supplied", True: "passed", False: "did not pass"}[comparison.cutoff_check]
    detectable = ("none" if comparison.detectable_effect is None
                  else f"{comparison.detectable_effect:.4g} AUC (power 0.8, two-sided alpha 0.05)")
    lines += [
        f"Answer: {comparison.answer}. Reason: {comparison.reason}. Detectable effect of the paired AUC "
        f"difference at this sample size: {detectable}. Cutoff check for logits: {cutoff}.",
        "",
        "Pairing: " + ("both calibrators hold out the same prompts and draws." if comparison.pairing is None
                       else f"not paired; missing: {comparison.pairing}."),
        "",
        "| Statistic | Value | Interval | n (unit) | Independent units | Test | Effect size | Detectable effect "
        "| Calculation | Estimand |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    lines.extend("| " + " | ".join(_cell(cell) for cell in _row(statistic)) + " |"
                 for statistic in comparison.statistics)
    lines += [
        "",
        "Rule: improve when the interval of the paired AUC difference excludes 0 in favour of logits on the macro "
        f"corpus and the cutoff check passed for logits; {MATCH_RULE}; otherwise undecided, with the detectable "
        "effect.",
        f"Prompt estimate: {PROMPT_ESTIMATE}.",
        f"Held-out AUC: prompts are resampled whole, stratified by label ({AUC_RESAMPLES} resamples); the "
        f"permutation test shuffles the labels over the prompts ({PERMUTATIONS} permutations, one-sided). Paired "
        f"difference: both AUCs recomputed on the same prompt resamples ({PAIRED_RESAMPLES}), two-sided bootstrap "
        "p-value. The draws of a prompt are not independent, so the prompt is the unit.",
        f"Correction factor: {CORRECTION_FACTOR}.",
        f"Estimates compared: {ESTIMATES_COMPARED}.",
        f"Agreement and correction factor: one row per prompt; months are resampled whole ({AGREEMENT_RESAMPLES} "
        "resamples), because the two renderings of a month share its data; months are treated as exchangeable. "
        "Limits of agreement are mean difference -/+ 1.96 standard deviations of the prompt differences, each "
        "limit with its own interval from the same month resamples. The two mean correction factors are plain "
        "means over the items of the difference and carry no interval.",
        "Convergence rows are read from the convergence analysis, which records no interval; the row at the kept "
        "count is equal by construction and carries no test.",
        f"Seed {SEED}, confidence {CONFIDENCE:g}. A statistic whose input was not recorded shows no value and names "
        "the input. Derived statistics only: no item, prompt, reply or licensed series is listed.",
    ]
    if comparison.incomplete_draw_log_lines:
        lines.append(f"The calibration draw log ended in {comparison.incomplete_draw_log_lines} incomplete last line "
                     "(a write cut short), which was ignored; the log is opened for reading only and was left as "
                     "it is.")
    lines.append("")
    return "\n".join(lines)


# ---------- the written outputs ----------

def comparison_path(arm: str, out_dir: str | Path = ANALYSIS_DIR) -> Path:
    """``<out_dir>/mode-comparison-<arm>.json``; its markdown table sits beside it with ``.md``."""
    if arm not in ARMS:
        raise ModeComparisonError(f"unknown arm {arm!r} for the mode comparison")
    return Path(out_dir) / f"mode-comparison-{arm}.json"


def _write_text(path: Path, text: str) -> None:
    """Write atomically: a crash leaves the previous file or the new one, never half of one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_comparison(comparison: ModeComparison, out_dir: str | Path = ANALYSIS_DIR) -> tuple[Path, Path]:
    """Write the JSON document and the markdown table once; the same outputs are a no-op, different ones refused."""
    if not isinstance(comparison, ModeComparison):
        raise TypeError(f"comparison must be a ModeComparison; got {type(comparison).__name__}")
    json_path = comparison_path(comparison.arm, out_dir)
    markdown_path = json_path.with_suffix(".md")
    texts = {json_path: comparison.document, markdown_path: comparison.to_markdown()}
    for path, text in texts.items():
        if path.exists() and path.read_text(encoding="utf-8") != text:
            raise ModeComparisonError(f"refusing to replace the different mode comparison at {path}")
    for path, text in texts.items():
        if not path.exists():
            _write_text(path, text)
    return json_path, markdown_path


__all__ = [
    "AGREEMENT_MARGIN",
    "ANALYSIS_DIR",
    "AUC_MARGIN",
    "COMPARISON_SCHEMA",
    "IMPROVE",
    "LOGITS",
    "LOGPROBS",
    "MATCH",
    "MATCH_RULE",
    "MODE_SOURCES",
    "PROMPT_ESTIMATE",
    "UNDECIDED",
    "ModeComparison",
    "ModeComparisonError",
    "comparison_path",
    "mode_comparison",
    "mode_comparison_from_records",
    "weights_identity",
    "write_comparison",
]
