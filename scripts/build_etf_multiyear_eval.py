"""Build a multi-year ETF-direction eval set spanning IS+OOS dates.

For each of SPY, QQQ, GLD, URTH (iShares MSCI World ETF), fetch the full
historical EOD price series from FMP, then randomly sample
TARGET_PER_TICKER trading days from 2020-01-01 → today. For each sampled
day, compute the next-day directional change as the target.

The resulting eval set spans dates BOTH pre-cutoff and post-cutoff for
every model in data/cutoffs.yaml, so each row is "in-sample" (IS) for
some models and "out-of-sample" (OOS) for others. The post-processing
script slices records.jsonl by row.metadata.date vs each model's cutoff
to compute the per-model memorization gap.

The output JSONL has NO ``_cutoff_date`` header line, so the harness's
``assert_cutoff_safe`` guard stays silent (it's still meaningful per row,
just enforced at analysis time rather than load time).
"""
from __future__ import annotations

import json
import os
import random
import sys
from datetime import date
from pathlib import Path

import requests
from dotenv import load_dotenv

ETFS: dict[str, str] = {
    "SPY": "SPDR S&P 500 ETF",
    "QQQ": "Invesco QQQ Trust (Nasdaq 100)",
    "GLD": "SPDR Gold Shares",
    "URTH": "iShares MSCI World ETF",
}

START_DATE = date(2020, 1, 1)
TARGET_PER_TICKER = 50  # 4 tickers x 50 = 200 eval rows
SEED = 0
OUT_PATH = Path("data/eval/etf_direction_multiyear.jsonl")


def fetch_eod(ticker: str, api_key: str) -> list[dict]:
    url = "https://financialmodelingprep.com/stable/historical-price-eod/light"
    r = requests.get(url, params={"symbol": ticker, "apikey": api_key}, timeout=30)
    r.raise_for_status()
    payload = r.json()
    return payload if isinstance(payload, list) else []


def sample_eval_rows(
    eod: list[dict],
    ticker: str,
    name: str,
    rng: random.Random,
    target: int,
) -> list[dict]:
    """Pick ``target`` random trading days post START_DATE and build eval rows."""
    rows: list[dict] = []
    in_window = [
        e for e in eod
        if "date" in e and "price" in e
        and date.fromisoformat(e["date"]) >= START_DATE
    ]
    in_window.sort(key=lambda e: e["date"])
    if len(in_window) < 2:
        return rows
    # Each row needs (yesterday, today). Index from 1 → len-1 of consecutive
    # pairs gives len-1 candidate rows.
    indices = list(range(1, len(in_window)))
    if len(indices) <= target:
        sampled = indices
    else:
        sampled = sorted(rng.sample(indices, target))

    for i in sampled:
        prev = in_window[i - 1]
        today = in_window[i]
        change = float(today["price"]) - float(prev["price"])
        if change > 0:
            direction = 1
        elif change < 0:
            direction = -1
        else:
            direction = 0
        prompt = (
            f"You are participating in a forecasting benchmark. You must "
            f"commit to a directional guess based on whatever you recall "
            f"or can pattern-match — refusing to answer counts as a wrong "
            f"answer.\n\n"
            f"Question: Did {ticker} ({name}) close higher, lower, or "
            f"unchanged on {today['date']} compared to the previous "
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
        rows.append({
            "prompt": prompt,
            "target_direction": direction,
            "metadata": {"ticker": ticker, "date": today["date"]},
        })
    return rows


def main() -> int:
    load_dotenv()
    api_key = os.environ.get("FMP_API_KEY")
    if not api_key:
        print("FMP_API_KEY missing from .env", file=sys.stderr)
        return 2

    rng = random.Random(SEED)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    for ticker, name in ETFS.items():
        try:
            eod = fetch_eod(ticker, api_key)
        except requests.RequestException as exc:
            print(f"  {ticker}: FMP fetch failed: {exc}", file=sys.stderr)
            continue
        ticker_rows = sample_eval_rows(eod, ticker, name, rng, TARGET_PER_TICKER)
        print(f"  {ticker}: {len(ticker_rows)} rows")
        all_rows.extend(ticker_rows)

    rng.shuffle(all_rows)
    with OUT_PATH.open("w") as f:
        for row in all_rows:
            f.write(json.dumps(row) + "\n")

    # Distribution stats
    from collections import Counter
    years = Counter(row["metadata"]["date"][:4] for row in all_rows)
    directions = Counter(row["target_direction"] for row in all_rows)
    print(f"\nWrote {len(all_rows)} eval rows to {OUT_PATH}")
    print(f"Year distribution: {dict(sorted(years.items()))}")
    print(f"Direction distribution: {dict(sorted(directions.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
