"""Pure scoring for S&P 500 recall and paired continuation; no model access."""

import json
import math
from collections import Counter
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation, localcontext
from statistics import mean


def parse_answer(content: str) -> dict:
    """Strict single-field JSON; null is abstention, malformed data is invalid."""
    def unique_object(pairs):
        if len(pairs) != 1 or pairs[0][0] != "value":
            raise ValueError("Expected only the value field")
        return dict(pairs)

    try:
        parsed = json.loads(content, parse_float=Decimal, parse_int=Decimal,
                            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
                            object_pairs_hook=unique_object)
        if not isinstance(parsed, dict) or set(parsed) != {"value"}:
            raise ValueError("Expected JSON object")
        value = parsed["value"]
        if value is None:
            return {"status": "abstained", "value": None}
        if not isinstance(value, Decimal) or not value.is_finite() or value <= 0:
            raise ValueError("Expected positive finite number")
        # Reject values that cannot be represented safely in downstream metrics.
        if not math.isfinite(float(value)) or float(value) == 0:
            raise ValueError("Number outside metric representation")
        return {"status": "parsed", "value": str(value)}
    except (ValueError, TypeError, InvalidOperation, OverflowError):
        return {"status": "invalid", "value": None}


def score_answer(item: dict, arm: str, content: str) -> dict:
    parsed = parse_answer(content)
    result = {"item_id": item["item_id"], "arm": arm, **parsed}
    if arm not in {"recall", "identified", "anonymous"}:
        raise ValueError("Unknown measurement arm")
    if parsed["status"] != "parsed":
        return result
    predicted = Decimal(parsed["value"])
    actual_close = Decimal(item["target_close"])
    if arm == "recall":
        relative_error = abs(float(predicted / actual_close - 1))
        result.update(
            absolute_relative_error=relative_error,
            within_0_1_percent=relative_error <= 0.001,
            within_1_percent=relative_error <= 0.01,
        )
        # Use explicit half-up source precision rather than Python's tie convention.
        with localcontext() as context:
            context.prec = max(28, predicted.adjusted() + 5, actual_close.adjusted() + 5)
            result["exact_at_two_decimals"] = (
                predicted.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                == actual_close.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
        return result
    last_close = Decimal(item["history"][-1]["close"])
    observed_return = math.log(float(actual_close / last_close))
    displayed_last_rebased = Decimal(item["rebased_history"][-1])
    predicted_return = math.log(float(predicted)) - math.log(float(displayed_last_rebased))
    scale = item["volatility_scale"]
    signed_error = predicted_return - observed_return
    result.update(
        observed_log_return=observed_return, predicted_log_return=predicted_return,
        signed_error=signed_error, squared_error=signed_error**2,
        normalized_absolute_error=abs(signed_error) / scale,
        unchanged_close_normalized_error=abs(observed_return) / scale,
    )
    return result


def paired_advantage(identified: dict, anonymous: dict) -> float | None:
    if (identified["item_id"] != anonymous["item_id"]
            or identified["arm"] != "identified" or anonymous["arm"] != "anonymous"):
        raise ValueError("Expected identified/anonymous scores for the same item")
    if identified["status"] != "parsed" or anonymous["status"] != "parsed":
        return None
    return anonymous["normalized_absolute_error"] - identified["normalized_absolute_error"]


def summarize_scores(items: list[dict], scores: list[dict]) -> dict:
    """Retain all planned denominators; only complete pairs enter the contrast."""
    item_ids = {item["item_id"] for item in items}
    if len(item_ids) != len(items):
        raise ValueError("Duplicate planned item")
    by_operation = {}
    for score in scores:
        key = score["item_id"], score["arm"]
        if key in by_operation or key[0] not in item_ids or key[1] not in {"recall", "identified", "anonymous"}:
            raise ValueError("Duplicate or unknown scored operation")
        if score["status"] not in {"parsed", "abstained", "invalid", "failed", "unperformed"}:
            raise ValueError("Unknown score status")
        by_operation[key] = score
    result = {}
    for model in sorted({item["model"] for item in items}):
        cohorts = {}
        for cohort in ("pre", "post"):
            selected = [item for item in items if item["model"] == model and item["cohort"] == cohort]
            arm_results = {}
            for arm in ("recall", "identified", "anonymous"):
                rows = [by_operation.get((item["item_id"], arm), {"status": "unperformed"}) for item in selected]
                counts = Counter(row["status"] for row in rows)
                parsed = [row for row in rows if row["status"] == "parsed"]
                metrics = {}
                if arm == "recall":
                    metrics["exact_matches"] = sum(row["exact_at_two_decimals"] for row in parsed)
                    metrics["mean_absolute_relative_error"] = (
                        mean(row["absolute_relative_error"] for row in parsed) if parsed else None)
                else:
                    metrics["mean_normalized_absolute_error"] = (
                        mean(row["normalized_absolute_error"] for row in parsed) if parsed else None)
                arm_results[arm] = {"planned": len(selected), "status_counts": dict(counts), **metrics}
            advantages = []
            for item in selected:
                identified = by_operation.get((item["item_id"], "identified"))
                anonymous = by_operation.get((item["item_id"], "anonymous"))
                if identified is not None and anonymous is not None:
                    if (advantage := paired_advantage(identified, anonymous)) is not None:
                        advantages.append(advantage)
            cohorts[cohort] = {
                "arms": arm_results, "common_usable_pairs": len(advantages),
                "mean_identity_advantage": mean(advantages) if advantages else None,
            }
        pre = cohorts["pre"]["mean_identity_advantage"]
        post = cohorts["post"]["mean_identity_advantage"]
        result[model] = {"cohorts": cohorts, "cutoff_contrast": pre - post if pre is not None and post is not None else None,
                         "confidence_interval": None, "interpretation": "mechanics pilot; no inferential claim"}
    return result
