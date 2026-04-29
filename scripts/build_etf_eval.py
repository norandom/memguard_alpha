"""Build a tiny ETF-direction eval set from FMP historical EOD prices.

For each of SPY, QQQ, GLD, URTH (iShares MSCI World ETF), fetch the last
~30 post-cutoff trading days' close prices and turn each consecutive pair
into one eval row:

    {
      "prompt": "Date: YYYY-MM-DD. Predict whether <TICKER> (<NAME>) closed
                 higher or lower than the previous trading day.\\nFormat:\\n
                 Direction: [1/-1/0]\\nConfidence: [0.0-1.0]",
      "target_direction": -1 | 0 | 1,
      "metadata": {"ticker": "...", "date": "..."}
    }

Cutoff: 2024-10-01 (one day after the latest model cutoff in
data/cutoffs.yaml). The first line of the file is the
``{"_cutoff_date": "2024-10-01"}`` header consumed by ``load_eval_set``.
"""
from __future__ import annotations

import json
import os
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

CUTOFF = date(2024, 10, 1)
TARGET_ROWS_PER_TICKER = 30
OUT_PATH = Path("data/eval/etf_direction.jsonl")


def fetch_eod(ticker: str, api_key: str) -> list[dict]:
    url = "https://financialmodelingprep.com/stable/historical-price-eod/light"
    params = {"symbol": ticker, "apikey": api_key}
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    payload = r.json()
    return payload if isinstance(payload, list) else []


def build_eval_rows(eod: list[dict], ticker: str, name: str) -> list[dict]:
    rows: list[dict] = []
    post = [
        e for e in eod
        if "date" in e and "price" in e
        and date.fromisoformat(e["date"]) >= CUTOFF
    ]
    post.sort(key=lambda e: e["date"])
    if len(post) < 2:
        return rows
    for i in range(1, min(len(post), TARGET_ROWS_PER_TICKER + 1)):
        prev_close = float(post[i - 1]["price"])
        today_close = float(post[i]["price"])
        change = today_close - prev_close
        if change > 0:
            direction = 1
        elif change < 0:
            direction = -1
        else:
            direction = 0
        prompt = (
            f"Date: {post[i]['date']}. Predict whether {ticker} ({name}) "
            f"closed higher or lower than the previous trading day.\n"
            f"Format:\n"
            f"Direction: [1/-1/0]\n"
            f"Confidence: [0.0-1.0]"
        )
        rows.append({
            "prompt": prompt,
            "target_direction": direction,
            "metadata": {"ticker": ticker, "date": post[i]["date"]},
        })
    return rows


def main() -> int:
    load_dotenv()
    api_key = os.environ.get("FMP_API_KEY")
    if not api_key:
        print("FMP_API_KEY missing from .env", file=sys.stderr)
        return 2

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    for ticker, name in ETFS.items():
        try:
            eod = fetch_eod(ticker, api_key)
        except requests.RequestException as exc:
            print(f"  {ticker}: FMP fetch failed: {exc}", file=sys.stderr)
            continue
        ticker_rows = build_eval_rows(eod, ticker, name)
        print(f"  {ticker}: {len(ticker_rows)} rows")
        all_rows.extend(ticker_rows)

    with OUT_PATH.open("w") as f:
        f.write(json.dumps({"_cutoff_date": CUTOFF.isoformat()}) + "\n")
        for row in all_rows:
            f.write(json.dumps(row) + "\n")

    direction_counts: dict[int, int] = {}
    for r in all_rows:
        direction_counts[r["target_direction"]] = direction_counts.get(r["target_direction"], 0) + 1
    print(f"\nWrote {len(all_rows)} eval rows to {OUT_PATH}")
    print(f"Direction distribution: {dict(sorted(direction_counts.items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
