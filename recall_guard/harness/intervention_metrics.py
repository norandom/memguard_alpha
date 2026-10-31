"""Offline four-condition metrics with paired item-level uncertainty.

Inference rows are items, aligned by item_id. The fixed condition order is
original single, original ensemble, rewritten single, rewritten ensemble.
Single is always draw 0 of its arm's ensemble, including a failed draw 0.
Neither the four conditions nor repeated target draws are independent samples.
This module performs no generation, score fitting, answer reduction or I/O.
"""

import json
from collections import Counter
from collections.abc import Mapping
from math import fsum, isfinite

import numpy as np

from recall_guard.harness.intervention_journal import ExperimentRun
from recall_guard.harness.intervention_types import EvidenceArm, RecordStatus, text_hash
from recall_guard.harness.intervention_usage import summarize_usage

CONDITION_ORDER = ("original_single", "original_ensemble", "rewritten_single", "rewritten_ensemble")
_CONTRASTS = {
    "rewriting_single": ("original_single", "rewritten_single"),
    "rewriting_ensemble": ("original_ensemble", "rewritten_ensemble"),
    "aggregation_original": ("original_single", "original_ensemble"),
    "aggregation_rewritten": ("rewritten_single", "rewritten_ensemble"),
}


def _number(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float)) and isfinite(value))


def _probability(value):
    return _number(value) and 0 <= value <= 1


def _mean(values):
    known = [float(value) for value in values if _number(value)]
    return {"mean": fsum(known) / len(known) if known else None, "n": len(known)}


def _rate(numerator, denominator):
    return {"estimate": numerator / denominator if denominator else None,
            "numerator": numerator, "denominator": denominator}


def _json_value(value):
    if isinstance(value, Mapping):
        return {key: _json_value(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(child) for child in value]
    return value


def _explicit_confidence(draw, component):
    components = draw.metadata.get("components") or {}
    score = draw.metadata.get("guarded_score") or {}
    if component is not None and _probability(value := components.get(component)):
        return float(value), "components." + component
    if _probability(value := score.get("raw_confidence")):
        return float(value), "guarded_score.raw_confidence"
    return None, None


def _draw_answer_usable(draw):
    # Projection and scoring are independent. A scoring failure can preserve a
    # valid projected answer. A configuration drift explicitly invalidates it.
    return (draw.response is not None and draw.metadata.get("answer") is not None
            and draw.status != RecordStatus.PENDING and draw.metadata.get("failure_kind") != "configuration_error")


def _condition_row(item, arm, single, draws, outcome, component):
    available = [draw for draw in draws if draw is not None]
    complete_draws = len(available) == len(draws) and all(draw.status != RecordStatus.PENDING for draw in available)
    terminal = complete_draws and (single or outcome is not None)
    first = draws[0]
    answer = (None if first is None else first.metadata.get("answer")) if single else (
        None if outcome is None else outcome.get("answer"))
    usable = (first is not None and _draw_answer_usable(first)) if single else (
        terminal and answer is not None and outcome.get("failure_kind") != "configuration_error")
    attempted = [draw for draw in available if draw.metadata.get("attempted", draw.response is not None)]
    parsed = [draw for draw in available if _draw_answer_usable(draw)]
    confidences, sources = [], Counter()
    for draw in parsed:
        value, source = _explicit_confidence(draw, component)
        if value is not None:
            confidences.append(value)
            sources[source] += 1
    scored = [draw.metadata["guarded_score"] for draw in available
              if draw.status != RecordStatus.PENDING and draw.metadata.get("failure_kind") != "configuration_error"
              and draw.metadata.get("guarded_score") and draw.metadata["guarded_score"].get("parse_ok")
              and _probability(draw.metadata["guarded_score"].get("p_memorized"))]
    score = (None if first is None else first.metadata.get("guarded_score")) if single else (
        None if outcome is None else outcome.get("scored_ensemble"))
    p_memorized = None if not score else score.get("p_memorized" if single else "p_memorized_point")
    if single and score and not score.get("parse_ok"):
        p_memorized = None
    ensemble = None if single or outcome is None else outcome.get("ensemble")
    failure_kind = (None if first is None else first.metadata.get("failure_kind")) if single else (
        None if outcome is None else outcome.get("failure_kind"))
    # Retained raw scores remain auditable even when the condition is pending
    # or invalidated; only validated usable scores enter published MCS values.
    scored_usable = terminal and _probability(p_memorized) and failure_kind != "configuration_error"
    failed = terminal and (first.status == RecordStatus.FAILED if single else outcome["status"] == "failed")
    return {
        "item_id": item.item_id, "condition": arm.value + ("_single" if single else "_ensemble"),
        "terminal": terminal, "started": terminal or bool(attempted), "target_attempted": bool(attempted),
        "answer_usable": bool(usable), "scored_usable": scored_usable,
        "failed": failed, "failure_kind": failure_kind, "answer": _json_value(answer),
        "correct": bool(usable and not failed and _json_value(answer) == _json_value(item.label)),
        "answer_correct": bool(usable and _json_value(answer) == _json_value(item.label)),
        "confidence": _mean(confidences)["mean"] if usable else None,
        "confidence_draws": len(confidences), "confidence_sources": dict(sources),
        "agreement": None if ensemble is None else ensemble.get("agreement"),
        "agreement_ci": None if ensemble is None else ensemble.get("agreement_ci"),
        "p_memorized": float(p_memorized) if scored_usable else None,
        "draws": {"planned": len(draws), "terminal": sum(draw.status != RecordStatus.PENDING for draw in available),
                  "attempted": len(attempted), "responses": sum(draw.response is not None for draw in available),
                  "answer_usable": len(parsed), "scored_usable": len(scored),
                  "failures": dict(sorted(Counter(draw.metadata.get("failure_kind") or "unspecified"
                                                  for draw in available if draw.status == RecordStatus.FAILED).items()))},
        "scores": scored,
    }


def _item_rows(run, component):
    checkpoint = run.checkpoint
    draws = {draw.identity: draw for draw in checkpoint.draws}
    outcomes = {(arm.item_id, arm.arm): arm.outcome for arm in run.arms}
    item_ids = {item.item_id for item in checkpoint.items}
    if len(outcomes) != len(run.arms) or any(item_id not in item_ids for item_id, _ in outcomes):
        raise ValueError("Duplicate or unknown item/arm outcome identity")
    rows = {}
    for item in sorted(checkpoint.items, key=lambda value: value.item_id):
        rows[item.item_id] = {}
        for arm in EvidenceArm:
            arm_draws = [draws.get((item.item_id, arm, index)) for index in range(checkpoint.protocol.draws)]
            for single in (True, False):
                row = _condition_row(item, arm, single, arm_draws[:1] if single else arm_draws,
                                     outcomes.get((item.item_id, arm)), component)
                if any(request.item_id == item.item_id and (
                    request.model_role == "target" and request.arm == arm
                    and (not single or request.draw_index == 0)
                    or arm == EvidenceArm.REWRITTEN and request.model_role != "target")
                       for request in checkpoint.requests):
                    row["started"] = True
                rows[item.item_id][row["condition"]] = row
    return rows


def _paired_effect(rows, baseline, comparison, scope, protocol, level):
    """One-dimensional deltas have one entry per eligible item, in item_id order.

    Both condition values move together in every resampled item. This estimates
    a separate marginal interval per contrast, not simultaneous/joint intervals.
    Accuracy deltas use probability units (1 = 100 percentage points).
    """
    pairs = [(values[baseline], values[comparison]) for values in rows.values()]
    terminal = [(left, right) for left, right in pairs if left["terminal"] and right["terminal"]]
    eligible = terminal
    field = "correct"
    if scope == "valid_pairs":
        field = "answer_correct"
        eligible = [(left, right) for left, right in terminal if left["answer_usable"] and right["answer_usable"]]
    elif scope == "confidence":
        field = "confidence"
        eligible = [(left, right) for left, right in terminal
                    if left[field] is not None and right[field] is not None]
    baseline_values = [float(left[field]) for left, _ in eligible]
    comparison_values = [float(right[field]) for _, right in eligible]
    item_deltas = np.asarray([right - left for left, right in zip(baseline_values, comparison_values, strict=True)])
    count = len(item_deltas)
    estimate = float(item_deltas.mean()) if count else None
    interval, status = None, "unavailable" if not count else "inadequate_items"
    if count >= 2 and protocol.bootstrap_count < 2:
        status = "inadequate_resamples"
    elif count >= 2:
        rng = np.random.default_rng(protocol.bootstrap_seed)
        # Each bootstrap replicate contains count ITEM rows. Draws and the
        # four conditions are never expanded into the resampling population.
        bootstrap_means = np.asarray([item_deltas[rng.integers(0, count, count)].mean()
                                      for _ in range(protocol.bootstrap_count)])
        tail = (1 - level) / 2
        interval = np.quantile(bootstrap_means, [tail, 1 - tail]).tolist()
        status = "degenerate" if interval[0] == interval[1] else "estimated"
    return {"estimate": estimate, "interval": interval, "interval_status": status, "n_eligible": count,
            "baseline_mean": _mean(baseline_values)["mean"], "comparison_mean": _mean(comparison_values)["mean"],
            "n_planned": len(pairs), "n_incomplete": len(pairs) - len(terminal),
            "n_failed_pairs": sum(left["failed"] or right["failed"] or not left["answer_usable"]
                                  or not right["answer_usable"] for left, right in terminal),
            "unit": "probability_difference", "equivalence_proof": False,
            "adequacy": "no_power_or_equivalence_assessment" if count >= 2 else "inadequate_items"}


def _fidelity(checkpoint, item_ids):
    records = {record.item_id: record for record in checkpoint.rewrites if record.item_id in item_ids}
    accepted = rejected = unperformed = pending = unchanged = candidates = semantic_unperformed = 0
    for item_id in item_ids:
        record = records.get(item_id)
        if record is None or record.status == RecordStatus.PENDING:
            pending += 1
        elif record.status == RecordStatus.COMPLETED:
            accepted += 1
        elif any(isinstance(value, Mapping) and value.get("status") == "rejected"
                 for value in record.fidelity_checks.values()):
            rejected += 1
        else:
            unperformed += 1
        if record is not None:
            semantic_unperformed += record.fidelity_checks.get("semantic", {}).get("status") not in {"passed", "rejected"}
            if record.rewritten_evidence is not None:
                candidates += 1
                unchanged += text_hash(record.original_evidence) == text_hash(record.rewritten_evidence)
    terminal = accepted + rejected + unperformed
    return {"planned": len(item_ids), "terminal": terminal, "accepted": accepted, "rejected": rejected,
            "unperformed_or_indeterminate": unperformed, "pending_or_missing": pending,
            "semantic_unperformed": semantic_unperformed,
            "acceptance_rate": _rate(accepted, terminal), "accepted_of_planned": _rate(accepted, len(item_ids)),
            "unchanged_evidence": _rate(unchanged, candidates),
            "interpretation": "Automated acceptance is not a guarantee of factual equivalence or absence of future knowledge."}


def _condition_summary(run, name, rows, item_ids):
    values = [conditions[name] for conditions in rows.values()]
    terminal = [row for row in values if row["terminal"]]
    usable = [row for row in terminal if row["answer_usable"]]
    sources, failures = Counter(), Counter()
    for row in values:
        sources.update(row["confidence_sources"])
        failures.update(row["draws"]["failures"])
    arm = EvidenceArm.ORIGINAL if name.startswith("original") else EvidenceArm.REWRITTEN
    single = name.endswith("single")
    requests = [request for request in run.checkpoint.requests if request.item_id in item_ids
                and ((request.model_role == "target" and request.arm == arm
                      and (not single or request.draw_index == 0))
                     or arm == EvidenceArm.REWRITTEN and request.model_role != "target")]
    scores = [score for row in values for score in row["scores"]]
    qualities = {}
    quality_items = 0
    for saved_arm in run.arms:
        if saved_arm.item_id in item_ids and saved_arm.arm == arm:
            quality = saved_arm.outcome.get("provenance", {}).get("scoring")
            if quality is not None:
                quality_items += 1
                # Only this explicit, persisted scoring snapshot is exported;
                # no live scorer or classifier state is traversed.
                snapshot = {key: quality.get(key) for key in ("identity", "holdout_auc", "is_weak", "valid_counts")}
                qualities[json.dumps(snapshot, sort_keys=True, allow_nan=False)] = snapshot
    discounts = [score["raw_confidence"] - score["memguard_confidence"] for score in scores
                 if _probability(score.get("raw_confidence")) and _probability(score.get("memguard_confidence"))]
    draw_counts = {key: sum(row["draws"][key] for row in values)
                   for key in ("planned", "terminal", "attempted", "responses", "answer_usable", "scored_usable")}
    return {
        "coverage": {"planned": len(values), "terminal": len(terminal),
                     "started": sum(row["started"] for row in values),
                     "target_attempted": sum(row["target_attempted"] for row in values),
                     "physical_target_attempted": len({request.item_id for request in requests if request.model_role == "target"}),
                     "answer_usable": len(usable), "scored_usable": sum(row["scored_usable"] for row in terminal),
                     "failed": sum(row["failed"] for row in terminal), "incomplete": len(values) - len(terminal),
                     "unstarted": sum(not row["started"] for row in values)},
        "accuracy": {"all_attempted": _rate(sum(row["correct"] for row in terminal), len(terminal)),
                     "usable_answers": _rate(sum(row["answer_correct"] for row in usable), len(usable))},
        "draws": draw_counts | {"failures": dict(sorted(failures.items()))},
        "parse_rate": _rate(draw_counts["answer_usable"], draw_counts["attempted"]),
        "failure_rate": _rate(sum(row["failed"] for row in terminal), len(terminal)),
        "confidence": _mean([row["confidence"] for row in usable]) | {
            "sources": dict(sorted(sources.items())), "explicit_draws": sum(row["confidence_draws"] for row in values),
            "answer_usable_draws": draw_counts["answer_usable"]},
        "agreement": _mean([row["agreement"] for row in terminal]),
        "scoring": {"p_memorized": _mean([row["p_memorized"] for row in terminal]),
                    "per_draw_raw_confidence": _mean([score.get("raw_confidence") for score in scores]),
                    "per_draw_memguard_confidence": _mean([score.get("memguard_confidence") for score in scores]),
                    "per_draw_discount": _mean(discounts),
                    "calibration_quality": [qualities[key] for key in sorted(qualities)],
                    "items_with_calibration_quality": quality_items,
                    "interpretation": "Existing calibrated MIA/MCS evidence; confidence alone is not MCS."},
        "usage": summarize_usage(run.checkpoint, requests),
    }


def _interpretation(conditions, effects, fidelity, tolerance):
    complete = all(value["coverage"]["incomplete"] == 0 for value in conditions.values())
    failures = any(value["coverage"]["failed"] for value in conditions.values())
    deltas = [effects[name]["all_attempted"]["estimate"] for name in ("rewriting_single", "rewriting_ensemble")]
    utility = ("failed" if any(delta is not None and delta < -tolerance for delta in deltas)
               else "unavailable" if any(delta is None for delta in deltas) else "within_observed_tolerance")
    return {"mitigation_success": False, "conclusion": "not_established", "utility_tolerance": tolerance,
            "gates": {"coverage": "complete" if complete else "incomplete",
                      "failures": "failed" if failures else "no_recorded_failures",
                      "fidelity": "accepted_automatically" if fidelity["accepted"] == fidelity["planned"]
                          and fidelity["planned"] else "failed" if fidelity["rejected"] or fidelity["unperformed_or_indeterminate"] else "incomplete",
                      "utility": utility, "independent_fidelity_audit": "not_assessed", "power": "not_established"},
            "limits": ["Lower task accuracy alone is not evidence of reduced recall.",
                       "Confidence changes and historical gaps are recall-related proxies only.",
                       "Failed/rejected interventions cannot establish mitigation success.",
                       "Intervals on tiny or degenerate samples are not equivalence or power evidence.",
                       "An exploratory pilot is not a powered mitigation-success result."]}


def _scope_summary(run, all_rows, item_ids, level):
    rows = {item_id: all_rows[item_id] for item_id in sorted(item_ids)}
    conditions = {name: _condition_summary(run, name, rows, item_ids) for name in CONDITION_ORDER}
    effects = {name: {"baseline": left, "comparison": right,
                      **{scope: _paired_effect(rows, left, right, scope, run.checkpoint.protocol, level)
                         for scope in ("all_attempted", "valid_pairs", "confidence")}}
               for name, (left, right) in _CONTRASTS.items()}
    fidelity = _fidelity(run.checkpoint, item_ids)
    requests = [request for request in run.checkpoint.requests if request.item_id in item_ids]
    public_rows = [{name: value for name, value in row.items() if name != "scores"}
                   for values in rows.values() for row in values.values()]
    return {"n_items": len(rows), "conditions": conditions, "effects": effects, "fidelity": fidelity,
            "items": public_rows, "usage": summarize_usage(run.checkpoint, requests),
            "interpretation": _interpretation(conditions, effects, fidelity, run.checkpoint.protocol.utility_tolerance)}


def summarize_experiment(run: ExperimentRun, *, confidence_level: float = 0.95,
                         confidence_component: str | None = "confidence") -> dict:
    """Return JSON-safe metrics from retained records without new model calls.

    Only the predeclared ``accuracy`` metric is supported. Analysis defaults
    are explicit in returned provenance; frozen protocol values are untouched.
    All-attempted means terminal planned condition outcomes, including rejected
    rewrites without target calls. Pending/missing conditions remain incomplete,
    not model errors. Paired all-attempted effects require both terminal outcomes;
    valid-pair effects additionally require both usable answers. Thus different
    scopes always disclose different eligible denominators.

    Confidence uses components[confidence_component], then an explicitly stored
    GuardedScore.raw_confidence, in [0, 1]. Ensemble confidence is the per-item
    mean of these observed draw confidences, with source/missingness coverage;
    it is neither the ensemble reducer's confidence nor a new MCS estimate.
    """
    if not isinstance(run, ExperimentRun):
        raise TypeError("run must be ExperimentRun")
    if not _number(confidence_level) or not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")
    if confidence_component is not None and (not isinstance(confidence_component, str) or not confidence_component.strip()):
        raise ValueError("confidence_component must be a nonempty name or None")
    protocol = run.checkpoint.protocol
    if protocol.task_metric != "accuracy":
        raise ValueError("Unsupported task_metric; this analysis supports explicit accuracy only")
    rows = _item_rows(run, confidence_component)
    cohorts = {name: {item.item_id for item in run.checkpoint.items if item.cohort == name}
               for name in sorted(protocol.cohort_definitions)}
    pooled = _scope_summary(run, rows, set(rows), confidence_level)
    # Probes/calibration without item identities belong only to the pooled
    # resource total. Shared condition resources are not summed into this total.
    pooled["usage"] = summarize_usage(run.checkpoint, run.checkpoint.requests)
    return {
        "protocol_id": protocol.protocol_id, "task_metric": protocol.task_metric,
        "provenance": {
            "condition_order": list(CONDITION_ORDER), "item_alignment": "lexicographic_item_id",
            "single_draw_index": 0, "single_reuses_ensemble_draw": True,
            "bootstrap": {"method": "percentile_paired_item_bootstrap", "unit": "item", "level": confidence_level,
                          "seed": protocol.bootstrap_seed, "count": protocol.bootstrap_count,
                          "sampling": "unstratified_within_each_eligible_comparison", "joint_intervals": False},
            "confidence": {"component": confidence_component, "fallback": "guarded_score.raw_confidence",
                           "range": [0, 1], "ensemble_aggregation": "mean_of_explicit_draw_confidences"},
            "scopes": {
                "all_attempted": "terminal planned condition outcomes; rejected/unattempted intervention failures included; pending/missing excluded",
                "valid_pairs": "same item_id with two terminal answer-usable conditions",
                "target_attempted": "logical target invocation from saved draw metadata; physical calls counted separately",
                "incomplete": "pending/missing draw or missing terminal ensemble outcome; never an observed error"},
            "cohort_definitions": protocol.to_dict()["cohort_definitions"],
            "stop_reason": run.stop_reason, "checkpoint_status": run.checkpoint.status.value,
        },
        "pooled": pooled, "cohorts": {name: _scope_summary(run, rows, ids, confidence_level)
                                        for name, ids in cohorts.items()},
        "historical": {"available": False,
                       "reason": "This analyzer does not validate historical pre/post cohort matching and sourced cutoffs; historical recall inference is unavailable.",
                       "interpretation": "Any later historical gap or difference-of-gaps is a recall-related proxy, not proof of memorization."},
    }
