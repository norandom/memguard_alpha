"""Build the three-asset portfolio eval set for the cmmd-backtest spec.

Universe: ``SWDA.L``, ``XLK``, ``IAU``. ``BIL`` is the cash leg in the
backtest and never receives a model signal, so we do not emit prompts
for it (Req 3.1).

For each ticker we fetch the EOD price series from FMP and randomly
sample at least 100 distinct trading days in the window
``[2020-01-01, today]``. The sample is stratified so both
pre-2024-07-01 and post-2024-07-01 days appear, which makes the eval
set straddle the gpt-oss-20b training cutoff (Req 3.3) and lets CMMD
filtering have anything to remove.

The prompt template mirrors ``scripts/build_etf_multiyear_eval.py`` —
the same "commitment" rubric (no refusal, no reasoning, two-line
answer).

Output: ``data/eval/etf_portfolio.jsonl`` with at least 300 rows
(3 tickers × 100 days minimum).
"""
from __future__ import annotations

import json
import os
import random
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Callable

import requests
from dotenv import load_dotenv

ETFS: dict[str, str] = {
    "SWDA.L": "iShares Core MSCI World UCITS ETF",
    "XLK": "Technology Select Sector SPDR Fund",
    "IAU": "iShares Gold Trust",
}

START_DATE = date(2020, 1, 1)
CUTOFF_DATE = date(2024, 7, 1)  # gpt-oss-20b training cutoff (2024-06-30 + 1)
MIN_PER_TICKER = 100  # Req 3.2 / 3.5 floor
TARGET_PER_TICKER = 110  # margin so a few empty days don't drop us under 100
DEFAULT_SEED = 0
OUT_PATH = Path("data/eval/etf_portfolio.jsonl")

FetchFn = Callable[[str, str], list[dict]]


def fetch_eod(ticker: str, api_key: str) -> list[dict]:
    """FMP EOD light endpoint, identical pattern to the multiyear builder."""
    url = "https://financialmodelingprep.com/stable/historical-price-eod/light"
    r = requests.get(url, params={"symbol": ticker, "apikey": api_key}, timeout=30)
    r.raise_for_status()
    payload = r.json()
    return payload if isinstance(payload, list) else []


def _build_prompt(ticker: str, name: str, today_str: str) -> str:
    """Same commitment template as ``build_etf_multiyear_eval.py``."""
    return (
        f"You are participating in a forecasting benchmark. You must "
        f"commit to a directional guess based on whatever you recall "
        f"or can pattern-match — refusing to answer counts as a wrong "
        f"answer.\n\n"
        f"Question: Did {ticker} ({name}) close higher, lower, or "
        f"unchanged on {today_str} compared to the previous "
        f"trading day?\n\n"
        f"Rules:\n"
        f"- DO NOT refuse, hedge, or say you don't have data.\n"
        f"- DO NOT explain your reasoning.\n"
        f"- If you have no information, guess and set Confidence: 0.5.\n"
        f"- Output ONLY the two lines below — nothing before, nothing after.\n\n"
        f"Direction: 1\n"
        f"Confidence: 0.65\n\n"
        f"(Replace the example values. Direction must be 1 = higher, "
        f"-1 = lower, or 0 = unchanged. Confidence must be a decimal "
        f"between 0.0 and 1.0.)"
    )


def _valid_window_pairs(eod: list[dict]) -> list[tuple[dict, dict]]:
    """Return (prev, today) pairs whose ``today`` falls in [START_DATE, today]."""
    in_window = [
        e for e in eod
        if "date" in e and "price" in e
        and date.fromisoformat(e["date"]) >= START_DATE
    ]
    in_window.sort(key=lambda e: e["date"])
    if len(in_window) < 2:
        return []
    return [(in_window[i - 1], in_window[i]) for i in range(1, len(in_window))]


def _stratified_sample(
    pairs: list[tuple[dict, dict]],
    rng: random.Random,
    target: int,
) -> list[tuple[dict, dict]]:
    """Sample ``target`` pairs ensuring both pre- and post-CUTOFF_DATE dates.

    We split the candidate pool by ``today`` date relative to the cutoff,
    then draw proportionally from each half so the resulting sample has
    representation from both — which is what Req 3.3 needs.
    """
    pre = [p for p in pairs if date.fromisoformat(p[1]["date"]) < CUTOFF_DATE]
    post = [p for p in pairs if date.fromisoformat(p[1]["date"]) >= CUTOFF_DATE]

    if not pre or not post:
        # Caller-side validation handles the empty-half case via the
        # cross-ticker check; here we just return what we can.
        if len(pairs) <= target:
            return list(pairs)
        return rng.sample(pairs, target)

    # Proportional allocation, but guarantee at least 1 from each half if
    # both halves have data and target >= 2.
    target = min(target, len(pairs))
    pre_target = max(1, round(target * len(pre) / (len(pre) + len(post))))
    post_target = target - pre_target
    if post_target < 1:
        post_target = 1
        pre_target = target - 1

    pre_target = min(pre_target, len(pre))
    post_target = min(post_target, len(post))

    sampled_pre = rng.sample(pre, pre_target) if pre_target else []
    sampled_post = rng.sample(post, post_target) if post_target else []
    combined = sampled_pre + sampled_post
    combined.sort(key=lambda p: p[1]["date"])
    return combined


def sample_eval_rows(
    eod: list[dict],
    ticker: str,
    name: str,
    rng: random.Random,
    target: int = TARGET_PER_TICKER,
) -> list[dict]:
    """Build eval rows for a single ticker from its EOD series."""
    pairs = _valid_window_pairs(eod)
    sampled = _stratified_sample(pairs, rng, target)
    rows: list[dict] = []
    for prev, today in sampled:
        change = float(today["price"]) - float(prev["price"])
        if change > 0:
            direction = 1
        elif change < 0:
            direction = -1
        else:
            direction = 0
        rows.append({
            "prompt": _build_prompt(ticker, name, today["date"]),
            "target_direction": direction,
            "metadata": {"ticker": ticker, "date": today["date"]},
        })
    return rows


def main(
    *,
    fetch_fn: FetchFn = fetch_eod,
    out_path: Path = OUT_PATH,
    seed: int = DEFAULT_SEED,
) -> int:
    """Build the eval set. Returns a process exit code (0 on success).

    ``fetch_fn`` is dependency-injected so tests can mock FMP without
    touching ``requests``.
    """
    load_dotenv()
    api_key = os.environ.get("FMP_API_KEY")
    if not api_key:
        print("FMP_API_KEY missing from .env", file=sys.stderr)
        return 2

    rng = random.Random(seed)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    for ticker, name in ETFS.items():
        try:
            eod = fetch_fn(ticker, api_key)
        except requests.RequestException as exc:
            print(
                f"FMP fetch failed for {ticker}: {exc}",
                file=sys.stderr,
            )
            return 3

        valid_pairs = _valid_window_pairs(eod)
        if len(valid_pairs) < MIN_PER_TICKER:
            print(
                f"{ticker}: FMP returned only {len(valid_pairs)} valid "
                f"trading days in [{START_DATE}, today], need at least "
                f"{MIN_PER_TICKER} (Req 3.5).",
                file=sys.stderr,
            )
            return 4

        ticker_rows = sample_eval_rows(eod, ticker, name, rng, TARGET_PER_TICKER)

        # Defensive: stratification can in theory hand back fewer rows
        # than the floor if either half of the window is starved on this
        # ticker. Surface that as a clear failure rather than writing a
        # thin file.
        dates = [r["metadata"]["date"] for r in ticker_rows]
        n_pre = sum(1 for d in dates if date.fromisoformat(d) < CUTOFF_DATE)
        n_post = sum(1 for d in dates if date.fromisoformat(d) >= CUTOFF_DATE)
        if len(ticker_rows) < MIN_PER_TICKER:
            print(
                f"{ticker}: only {len(ticker_rows)} rows produced after "
                f"sampling, need {MIN_PER_TICKER} (Req 3.2).",
                file=sys.stderr,
            )
            return 5
        if n_pre == 0 or n_post == 0:
            half = "pre-2024-07-01" if n_pre == 0 else "post-2024-07-01"
            print(
                f"{ticker}: sampled rows lack any {half} dates; the eval "
                f"set must straddle the gpt-oss-20b cutoff (Req 3.3).",
                file=sys.stderr,
            )
            return 6

        print(
            f"  {ticker}: {len(ticker_rows)} rows "
            f"(pre-cutoff={n_pre}, post-cutoff={n_post})"
        )
        all_rows.extend(ticker_rows)

    rng.shuffle(all_rows)
    with out_path.open("w") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")

    years = Counter(row["metadata"]["date"][:4] for row in all_rows)
    directions = Counter(row["target_direction"] for row in all_rows)
    tickers = Counter(row["metadata"]["ticker"] for row in all_rows)
    print(f"\nWrote {len(all_rows)} eval rows to {out_path}")
    print(f"Seed: {seed}")
    print(f"Per-ticker counts: {dict(sorted(tickers.items()))}")
    print(f"Year distribution: {dict(sorted(years.items()))}")
    print(f"Direction distribution: {dict(sorted(directions.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
