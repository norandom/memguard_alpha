"""Local S&P 500 snapshot validation and leakage-free pilot inputs."""

import csv
import hashlib
import io
import json
import math
import random
import statistics
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path

from scripts.prepare_completion_inputs import sessions

HISTORY_LENGTH = 20
VOLATILITY_FLOOR = 0.0001  # One basis point in daily log-return units.
ARMS = ("recall", "identified", "anonymous")
SEED = 20261105


@dataclass(frozen=True)
class Observation:
    day: date
    close: Decimal | None


@dataclass(frozen=True)
class PriceWindow:
    target: Observation
    history: tuple[Observation, ...]

    @property
    def volatility(self) -> float:
        returns = [math.log(float(current.close / previous.close))
                   for previous, current in zip(self.history, self.history[1:], strict=False)]
        return statistics.stdev(returns)

    @property
    def rebased(self) -> tuple[str, ...]:
        anchor_close = self.history[0].close
        return tuple(format(100 * observation.close / anchor_close, ".8f") for observation in self.history)


def parse_snapshot(raw: bytes, completed_through: date) -> tuple[Observation, ...]:
    """One row per source date; missing values remain None, never forward-filled."""
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    if reader.fieldnames != ["observation_date", "SP500"]:
        raise ValueError("Expected the FRED SP500 observation_date,SP500 schema")
    observations = []
    previous_day = None
    for row in reader:
        if None in row or any(value is None for value in row.values()):
            raise ValueError("Malformed source row")
        day = date.fromisoformat(row["observation_date"])
        if previous_day is not None and day <= previous_day:
            raise ValueError("Source dates must be strictly increasing and unique")
        if day > completed_through:
            raise ValueError("Source contains an observation beyond the completed-date boundary")
        previous_day = day
        value = row["SP500"].strip()
        try:
            close = None if value in {"", "."} else Decimal(value)
        except InvalidOperation:
            raise ValueError("Invalid source close") from None
        if close is not None and (not close.is_finite() or close <= 0):
            raise ValueError("Source close must be finite and positive")
        observations.append(Observation(day, close))
    if not observations or not any(observation.close is not None for observation in observations):
        raise ValueError("Empty usable source")
    return tuple(observations)


def load_snapshot(directory: Path, completed_through: date):
    raw = (directory / "SP500.csv").read_bytes()
    metadata = json.loads((directory / "source.json").read_text(encoding="utf-8"))
    if metadata.get("series") != "SP500" or metadata.get("sha256") != hashlib.sha256(raw).hexdigest():
        raise ValueError("Source identity or content hash mismatch")
    return parse_snapshot(raw, completed_through), metadata


def windows(observations: tuple[Observation, ...], start: date, end: date):
    """Reject missing exchange sessions inside a context rather than bridge a gap."""
    valid = [observation for observation in observations if observation.close is not None]
    candidates = []
    rejected = []
    for index, target in enumerate(valid):
        if not start <= target.day <= end or index < HISTORY_LENGTH:
            continue
        history = tuple(valid[index - HISTORY_LENGTH:index])
        expected = sessions(history[0].day.isoformat(), target.day.isoformat())
        actual = [observation.day.isoformat() for observation in (*history, target)]
        if actual != expected:
            rejected.append(target.day.isoformat())
            continue
        candidates.append(PriceWindow(target, history))
    return candidates, rejected


def cohort_ranges(cutoff: date, last_available: date):
    """Exclude cutoff month plus 30 days; use equal <=90-day candidate spans."""
    pre_end = cutoff.replace(day=1) - timedelta(days=31)
    post_start = cutoff + timedelta(days=31)
    span_days = min(90, (last_available - post_start).days + 1)
    if span_days < 45:
        raise ValueError("Fewer than 45 calendar days after cutoff buffer")
    return {"pre": (pre_end - timedelta(days=span_days - 1), pre_end),
            "post": (post_start, post_start + timedelta(days=span_days - 1))}


def select_matched_dates(pre_candidates, post_candidates):
    """One prior-volatility quartile pair each; never inspect target close."""
    if min(len(pre_candidates), len(post_candidates)) < 16:
        raise ValueError("Need at least 16 candidate dates in each cohort")
    rng = random.Random(SEED)
    def sort_key(window):
        return window.volatility, window.target.day
    pre_sorted = sorted(pre_candidates, key=sort_key)
    post_sorted = sorted(post_candidates, key=sort_key)
    selected = []
    for quartile in range(4):
        pre_bin = pre_sorted[len(pre_sorted) * quartile // 4:len(pre_sorted) * (quartile + 1) // 4]
        post_bin = post_sorted[len(post_sorted) * quartile // 4:len(post_sorted) * (quartile + 1) // 4]
        before = rng.choice(pre_bin)
        after = min(post_bin, key=lambda window: (
            abs(math.log(max(window.volatility, VOLATILITY_FLOOR) / max(before.volatility, VOLATILITY_FLOOR))),
            window.target.day))
        selected.append((quartile, before, after))
    return selected


def render_prompts(window: PriceWindow) -> dict[str, str]:
    """Only recall/identified include identity; target prices never enter prompts."""
    output_rule = (
        ' Return only a JSON object {"value": NUMBER}, or {"value": null} if you cannot answer. '
        "Do not use browsing, tools or retrieval. Do not include commentary."
    )
    continuation = (
        "Predict the next trading observation's closing value from the preceding 20 closing observations below. "
        "Values are rebased so the first observation equals 100. "
        "Return your prediction in these same rebased units."
    )
    identified_rows = "\n".join(f"{observation.day.isoformat()},{value}"
                               for observation, value in zip(window.history, window.rebased, strict=True))
    anonymous_rows = "\n".join(f"{index},{value}" for index, value in enumerate(window.rebased, 1))
    return {
        "recall": (f"What was the S&P 500 price index closing value on {window.target.day.isoformat()}? "
                   "Return index points, not SPY ETF prices or a total-return index." + output_rule),
        "identified": (continuation + output_rule +
                       f"\nSeries: S&P 500 price index. Target trading date: {window.target.day.isoformat()}."
                       "\nobservation,value\n" + identified_rows),
        "anonymous": continuation + output_rule + "\nobservation,value\n" + anonymous_rows,
    }


def window_record(model: str, cohort: str, quartile: int, window: PriceWindow) -> dict:
    return {
        "item_id": f"{model.replace('/', '_')}:{cohort}:{window.target.day.isoformat()}",
        "model": model, "cohort": cohort, "volatility_quartile": quartile,
        "target_date": window.target.day.isoformat(),
        "target_close": str(window.target.close),
        "history": [{"date": observation.day.isoformat(), "close": str(observation.close)}
                    for observation in window.history],
        "rebased_history": window.rebased,
        "trailing_volatility": window.volatility,
        "volatility_scale": max(window.volatility, VOLATILITY_FLOOR),
        "prompts": render_prompts(window),
        "split": "development_mechanics_only",
    }
