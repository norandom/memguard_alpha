"""Collect/build/validate dated Task 6.1 inputs without model APIs or credentials.

Use ``uv run python -m scripts.prepare_completion_inputs`` from a checkout.
Collection is explicit; build/validate use saved bytes offline. Labels and
window provenance remain offline; use label_free_input for model-facing input.
"""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import UTC, date, datetime, time, timedelta
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

from recall_guard.harness.intervention_types import ExperimentItem
from recall_guard.harness.rewriting import RewriteInput, render_target_prompt

TICKERS = ("IAU", "SPY")
SALT = "completion-api-2026-10-31-input-selection-v1"
COMPLETED_THROUGH = "2026-10-30"
INSTRUCTIONS = (
    "Predict the direction of the next trading session's quoted close relative to the last quoted close in the evidence. "
    "Use 1 for higher, -1 for lower, and 0 for unchanged. Prices are rounded to the nearest USD 0.01, half up. "
    'Return only a JSON object in this format: {"direction": <-1|0|1>, "confidence": <0..1>}. '
    "Confidence is your probability that your chosen direction is correct."
)
# Full-day NYSE/Arca closures; early-close days remain valid daily sessions.
# Official sources and retrieval dates are in preparation-plan.json.
HOLIDAYS = {
    2021: "01-01 01-18 02-15 04-02 05-31 07-05 09-06 11-25 12-24",
    2022: "01-17 02-21 04-15 05-30 06-20 07-04 09-05 11-24 12-26",
    2023: "01-02 01-16 02-20 04-07 05-29 06-19 07-04 09-04 11-23 12-25",
    2024: "01-01 01-15 02-19 03-29 05-27 06-19 07-04 09-02 11-28 12-25",
    2025: "01-01 01-09 01-20 02-17 04-18 05-26 06-19 07-04 09-01 11-27 12-25",
    2026: "01-01 01-19 02-16 04-03 05-25 06-19 07-03 09-07 11-26 12-25",
}


def digest(content):
    return hashlib.sha256(content).hexdigest()


def write_json(path, value):
    """Persist deterministic UTF-8/LF bytes without newline translation."""
    path.write_bytes((json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode())


def sessions(start, end):
    result = []
    day = date.fromisoformat(start)
    while day.isoformat() <= end:
        if day.year not in HOLIDAYS:
            raise ValueError("Session calendar covers only 2021 through 2026")
        if day.weekday() < 5 and day.strftime("%m-%d") not in HOLIDAYS[day.year].split():
            result.append(day.isoformat())
        day += timedelta(days=1)
    return result


def _positive(value):
    return isinstance(value, (int, Decimal)) and not isinstance(value, bool) and Decimal(value).is_finite() and value > 0


def parse_chart(raw, ticker, *, completed_through):
    """Validate aligned daily bars; preserve holes so windows cannot bridge them.

    Bars have one row per source exchange session, exact JSON Decimal closes,
    and UTC timestamp identity. Missing/invalid sessions are reported separately.
    """
    payload = json.loads(raw.decode("utf-8"), parse_float=Decimal,
                         parse_constant=lambda value: (_ for _ in ()).throw(ValueError("non-finite source number")))
    if payload["chart"]["error"] is not None or len(payload["chart"]["result"] or []) != 1:
        raise ValueError("Expected one successful chart result")
    result = payload["chart"]["result"][0]
    meta = result["meta"]
    for field, expected in (("symbol", ticker), ("currency", "USD"),
                            ("exchangeTimezoneName", "America/New_York"), ("instrumentType", "ETF")):
        if meta.get(field) != expected:
            raise ValueError(f"Unexpected source {field} (timezone must be America/New_York)")
    stamps = result["timestamp"]
    closes = result["indicators"]["quote"][0]["close"]
    if not stamps or len(stamps) != len(closes):
        raise ValueError("Timestamp/close length mismatch or empty source")
    if len(set(stamps)) != len(stamps):
        raise ValueError("Timestamps must be unique")
    if stamps != sorted(stamps):
        raise ValueError("Timestamps must be ordered")
    bars, invalid, incomplete, dates = [], [], [], []
    for stamp, close in zip(stamps, closes, strict=True):
        if type(stamp) is not int:
            raise ValueError("Expected integral session timestamp")
        local = datetime.fromtimestamp(stamp, ZoneInfo("America/New_York"))
        day = local.date().isoformat()
        if not sessions(day, day) or not time(9, 30) <= local.time() <= time(16):
            raise ValueError("Timestamp does not align with an exchange session")
        dates.append(day)
        if day > completed_through:
            incomplete.append(day)
            continue
        if not _positive(close):
            invalid.append(day)
            close = None
        bars.append({"date": day, "timestamp": stamp, "close": None if close is None else Decimal(close)})
    if len(set(dates)) != len(dates):
        raise ValueError("Exchange dates must be unique")
    if not bars:
        raise ValueError("No completed daily sessions")
    missing = sorted(set(sessions(bars[0]["date"], completed_through)) - set(dates))
    actions = []
    for kind, events in result.get("events", {}).items():
        if kind not in {"dividends", "splits", "capitalGains"}:
            raise ValueError("Unrecognized corporate action type")
        for event in events.values():
            day = datetime.fromtimestamp(event["date"], ZoneInfo("America/New_York")).date().isoformat()
            actions.append({"type": kind, "date": day})
    adjusted = result["indicators"].get("adjclose", [])
    if adjusted and len(adjusted[0]["adjclose"]) != len(stamps):
        raise ValueError("Adjusted-close length mismatch")
    return {"ticker": ticker, "timezone": meta["exchangeTimezoneName"], "bars": bars,
            "missing_sessions": missing, "invalid_sessions": invalid, "incomplete_sessions": incomplete,
            "actions": sorted(actions, key=lambda action: (action["date"], action["type"])),
            "adjusted_close_available": bool(adjusted), "raw_sha256": digest(raw),
            "quote_vs_adjclose_different_sessions": sum(a != q for a, q in zip(adjusted[0]["adjclose"], closes, strict=True))
            if adjusted else None}


def _price(close):
    return close.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def _item(series, cohort, dates, bars):
    as_of, outcome = dates[-2:]
    prices = [_price(bars[day]["close"]) for day in dates]
    direction = int(prices[-1] > prices[-2]) - int(prices[-1] < prices[-2])
    ticker = series["ticker"]
    evidence = (f"ticker: {ticker}; currency: USD; units: USD per share.\n"
                "source: Yahoo Finance daily quoted closes, retrieved retrospectively; prices may be revised "
                "or split-adjusted by the source and are not total returns.\n"
                "five trailing trading-session closes, rounded to USD 0.01:\n" +
                "\n".join(f"{day}: {price:.2f}" for day, price in zip(dates[:-1], prices[:-1], strict=True)))
    # Opaque ID contains neither cohort, label nor future date.
    identity = digest(f"{ticker}|{as_of}".encode())[:20]
    item = ExperimentItem("price-" + identity, INSTRUCTIONS, evidence, as_of, direction, cohort)
    return {"item": item, "ticker": ticker, "evidence_dates": dates[:-1], "outcome_date": outcome,
            "window_dates": dates, "outcome_close": str(prices[-1]), "as_of_close": str(prices[-2])}


def candidates(series, cohort, start, end, *, cutoff, period):
    """Enumerate expected as-of sessions; one exclusive exclusion reason each."""
    bars = {bar["date"]: bar for bar in series["bars"]}
    calendar = sessions(min(start, min(bars)), max(end, max(bars)))
    actions = {action["date"] for action in series["actions"]}
    counts = Counter({name: 0 for name in ("expected_as_of", "boundary", "missing_session", "invalid_close",
                                          "corporate_action", "cutoff", "flat", "eligible")})
    result = []
    for index, as_of in enumerate(calendar):
        if not start <= as_of <= end:
            continue
        counts["expected_as_of"] += 1
        window = calendar[max(0, index - 4):index + 2]
        reason = None
        if index < 4 or len(window) != 6 or window[0] < start or window[-1] > end:
            reason = "boundary"
        elif cutoff and ((period == "pre" and window[-1] > cutoff) or (period == "post" and window[0] <= cutoff)):
            reason = "cutoff"
        elif any(day not in bars for day in window):
            reason = "missing_session"
        elif any(bars[day]["close"] is None for day in window):
            reason = "invalid_close"
        elif actions.intersection(window):
            reason = "corporate_action"
        if reason:
            counts[reason] += 1
            continue
        row = _item(series, cohort, window, bars)
        if row["item"].label == 0:
            counts["flat"] += 1
        else:
            counts["eligible"] += 1
            result.append(row)
    return result, dict(counts)


def select_balanced(rows, *, per_stratum, salt, cohorts=None):
    """SHA256 rank once; greedily enforce quota and disjoint six-session windows.

    Overlap applies across both directions within ticker/cohort. Outcome labels
    define the research strata; no model responses are consulted.
    """
    declared = set(cohorts) if cohorts is not None else {r["item"].cohort for r in rows}
    if not declared:
        raise ValueError("Insufficient candidates: no cohorts")
    quotas = {(cohort, ticker, direction): per_stratum for cohort in declared
              for ticker in TICKERS for direction in (-1, 1)}
    chosen, used, counts = [], defaultdict(set), Counter(overlap=0, quota_full=0, selected=0)

    def rank(row):
        return digest((salt + "|" + row["item"].item_id).encode()), row["item"].item_id

    for row in sorted(rows, key=rank):
        item, ticker = row["item"], row["ticker"]
        key = item.cohort, ticker, item.label
        if quotas[key] == 0:
            counts["quota_full"] += 1
        elif used[item.cohort, ticker].intersection(row["window_dates"]):
            counts["overlap"] += 1
        else:
            chosen.append(row)
            used[item.cohort, ticker].update(row["window_dates"])
            quotas[key] -= 1
            counts["selected"] += 1
    if any(quotas.values()):
        raise ValueError(f"Insufficient nonoverlapping candidates for strata: {quotas}")
    return chosen, dict(counts)


def evaluation_order(rows):
    """Four-item blocks balance original-first parity within cohort and stratum."""
    result = []
    for ticker in TICKERS:
        for direction in (-1, 1):
            groups = {cohort: [r for r in rows if r["ticker"] == ticker and r["item"].label == direction
                               and r["item"].cohort == cohort] for cohort in ("pre", "post")}
            if any(len(group) != 2 for group in groups.values()):
                raise ValueError("Evaluation order requires two items per cohort/ticker/direction")
            result.extend([groups["pre"][0], groups["post"][0], groups["post"][1], groups["pre"][1]])
    return result


def label_free_input(item):
    """Explicitly drop cohort/label before passing input to rewriter/target."""
    return RewriteInput(item.item_id, item.instructions, item.evidence, item.as_of)


def fetch_chart(ticker, directory):
    """Public unauthenticated GET with ambient proxies/netrc/auth disabled."""
    if ticker not in TICKERS:
        raise ValueError("Only predeclared tickers may be collected")
    path = directory / (ticker + ".chart.json")
    if path.exists():
        raise FileExistsError(path.name)
    session = requests.Session()
    session.trust_env = False
    session.headers.update({"User-Agent": "recall-guard-public-input-preparation/1.0"})
    url = "https://query1.finance.yahoo.com/v8/finance/chart/" + ticker
    params = {"range": "5y", "interval": "1d", "events": "div,splits,capitalGains", "includePrePost": "false"}
    try:
        response = session.get(url, params=params, timeout=60, allow_redirects=False)
        response.raise_for_status()
        if response.status_code != 200:
            raise ValueError("Expected HTTP 200 without redirects")
        raw = response.content
        path.write_bytes(raw)
        return {"path": "raw/" + path.name, "request_url": url, "request_params": params,
                "response_url": response.url, "retrieved_at_utc": datetime.now(UTC).isoformat(),
                "sha256": digest(raw), "bytes": len(raw), "http_status": response.status_code,
                "ambient_auth_proxies_netrc": "disabled (Session.trust_env=False)"}
    finally:
        session.close()


def _json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode()


def _read(path):
    return json.loads(path.read_bytes())


def _strata(rows):
    counts = Counter(f"{r['item'].cohort}/{r['ticker']}/{r['item'].label}" for r in rows)
    return dict(sorted(counts.items()))


def derive_artifacts(directory):
    """Recompute all derived artifacts from pinned raw bytes, offline.

    The preparation report is not an efficacy report. Each row in window
    provenance is one selected item, aligned by opaque item_id to items files.
    """
    plan = _read(directory / "preparation-plan.json")
    registry = _read(directory / "cutoff-registry.json")
    manifest = _read(directory / "source-manifest.json")
    if plan["selection_salt"] != SALT or plan["completed_through"] != COMPLETED_THROUGH or tuple(plan["tickers"]) != TICKERS:
        raise ValueError("Preparation plan differs from the versioned builder")
    if set(manifest["sources"]) != set(TICKERS):
        raise ValueError("Source manifest must cover exactly the declared tickers")
    sources, summaries = {}, {}
    for ticker in TICKERS:
        metadata = manifest["sources"][ticker]
        if metadata["path"] != "raw/" + ticker + ".chart.json":
            raise ValueError("Raw source path must be portable and ticker-specific")
        raw = (directory / metadata["path"]).read_bytes()
        if metadata["sha256"] != digest(raw) or metadata["bytes"] != len(raw):
            raise ValueError("Raw source byte hash/length mismatch: " + ticker)
        series = parse_chart(raw, ticker, completed_through=COMPLETED_THROUGH)
        sources[ticker] = series
        summaries[ticker] = {key: value for key, value in series.items() if key != "bars"}
        summaries[ticker].update({"first_session": series["bars"][0]["date"],
                                  "last_session": series["bars"][-1]["date"], "completed_bars": len(series["bars"]),
                                  "request": metadata, "action_counts": dict(Counter(a["type"] for a in series["actions"]))})
    pool, candidate_audit = defaultdict(list), {}
    for window in plan["windows"]:
        for ticker, series in sources.items():
            rows, counts = candidates(series, window["cohort"], window["start"], window["end"],
                                      cutoff=window["cutoff"], period=window["period"])
            pool[window["dataset"]].extend(rows)
            candidate_audit[window["dataset"] + "/" + window["cohort"] + "/" + ticker] = counts
    selected, selection_audit = {}, {}
    for dataset, rows in pool.items():
        quotas = {w["per_stratum"] for w in plan["windows"] if w["dataset"] == dataset}
        if len(quotas) != 1:
            raise ValueError("All cohorts in a dataset require the same stratum quota")
        cohorts = {w["cohort"] for w in plan["windows"] if w["dataset"] == dataset}
        selected[dataset], selection_audit[dataset] = select_balanced(rows, per_stratum=quotas.pop(), salt=SALT, cohorts=cohorts)
    selected["evaluation"] = evaluation_order(selected["evaluation"])
    selected_rows = selected["evaluation"] + selected["calibration"]
    if len({r["item"].item_id for r in selected_rows}) != len(selected_rows):
        raise ValueError("Selected item IDs must be unique")
    if len({r["item"].evidence for r in selected_rows}) != len(selected_rows):
        raise ValueError("Selected evidence must be unique")
    eval_dates = {d for r in selected["evaluation"] for d in r["window_dates"]}
    calib_dates = {d for r in selected["calibration"] for d in r["window_dates"]}
    if eval_dates.intersection(calib_dates):
        raise ValueError("Calibration and evaluation dates/windows must be disjoint")
    files = {
        "evaluation.json": {"schema_version": 1, "items": [r["item"].to_dict() for r in selected["evaluation"]]},
        **{f"calibration-{name}.json": {"schema_version": 1,
            "items": [r["item"].to_dict() for r in selected["calibration"] if r["item"].cohort == cohort]}
           for name, cohort in (("is", "is_proxy"), ("oos", "oos_proxy"))},
    }
    windows = {}
    for row in selected_rows:
        item = row["item"]
        windows[item.item_id] = {key: value for key, value in row.items() if key != "item"}
        windows[item.item_id].update({"cohort": item.cohort, "label": item.label, "as_of": item.as_of,
                                     "input_hash": item.input_hash, "evidence_hash": digest(item.evidence.encode()),
                                     "target_prompt_hash": digest(render_target_prompt(label_free_input(item), evidence=item.evidence).encode()),
                                     "source_sha256": sources[row["ticker"]]["raw_sha256"]})
    files["selected-windows.json"] = {"schema_version": 1, "offline_only": True, "items": windows}
    definitions = {}
    for model, basis in registry["models"].items():
        definitions[model] = {cohort: {"description": f"common {cohort}-cutoff timing proxy",
            "historical": {"period": cohort, "matching_group": "common-price-evidence-v1",
                           "matching": plan["matching"], "cutoff": {"model": model, **basis},
                           "outcome_dates": {r["item"].item_id: r["outcome_date"] for r in selected["evaluation"]
                                             if r["item"].cohort == cohort}}} for cohort in ("pre", "post")}
    files["cohort-definitions.json"] = {"schema_version": 1, "models": definitions}
    fixed = ("preparation-plan.json", "cutoff-registry.json", "model-provenance.json", "source-manifest.json")
    hashes = {name: digest((directory / name).read_bytes()) for name in fixed}
    hashes.update({name: digest(_json_bytes(value)) for name, value in files.items()})
    hashes.update({metadata["path"]: metadata["sha256"] for metadata in manifest["sources"].values()})
    files["preparation-report.json"] = {
        "schema_version": 1, "status": "inputs_prepared_only", "live_provider_calls": 0,
        "protocol_settings_frozen": False, "selection_salt": SALT, "sources": summaries,
        "artifact_sha256": hashes, "candidate_validation": candidate_audit,
        "exclusion_accounting": "Each expected as_of has exactly one first-match outcome: boundary, cutoff, missing_session, invalid_close, corporate_action, flat, or eligible. Selection then partitions eligible rows into overlap, quota_full or selected.",
        "selection_validation": selection_audit,
        "eligible_strata": {name: _strata(rows) for name, rows in pool.items()},
        "selected_strata": {name: _strata(rows) for name, rows in selected.items()},
        "inclusion_counts": {"evaluation": len(selected["evaluation"]), "calibration_is": len(files["calibration-is.json"]["items"]),
                             "calibration_oos": len(files["calibration-oos.json"]["items"])},
        "separation": {"unique_item_ids": True, "unique_evidence_hashes": True, "calibration_evaluation_dates_disjoint": True,
                       "within_ticker_cohort_windows_disjoint": True, "overlap_scope": "same ticker; six dates including outcome"},
        "arm_order": {"pattern_per_ticker_direction": ["pre", "post", "post", "pre"],
                      "cohort_counts_per_even_odd_index": {"pre": [4, 4], "post": [4, 4]}},
        "source_format_differences": {"source": "none by design", "task_and_format": "none by design",
                                      "remaining": "calendar years, market regimes, price levels and distributions differ; not causal matching"},
        "price_basis": plan["price_basis"], "timezone": "America/New_York",
        "constant_direction_accuracy_baseline": 0.5, "selected_windows": "selected-windows.json",
        "cutoff_provenance": "model-provenance.json", "model_specific_cohorts": "cohort-definitions.json",
        "unsupported_inference": _read(directory / "model-provenance.json")["unsupported_inference"],
        "limitations": plan["limitations"],
        "reproduce_offline": "uv run python -m scripts.prepare_completion_inputs validate --directory data/experiments/2026-10-31-completion-api",
    }
    return files


def build(directory):
    files = derive_artifacts(directory)
    for name, value in files.items():
        path, content = directory / name, _json_bytes(value)
        if path.exists() and path.read_bytes() != content:
            raise FileExistsError("Refusing to replace different frozen artifact: " + name)
        path.write_bytes(content)
    return files["preparation-report.json"]


def validate(directory):
    """Fresh raw-byte recomputation catches tampering, leakage and stale hashes."""
    files = derive_artifacts(directory)
    for name, value in files.items():
        if (directory / name).read_bytes() != _json_bytes(value):
            raise ValueError("Derived artifact differs from pinned-source recomputation: " + name)
    return {"status": "validated_inputs_only", "inclusion_counts": files["preparation-report.json"]["inclusion_counts"],
            "artifact_sha256": files["preparation-report.json"]["artifact_sha256"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("collect", "build", "validate"))
    parser.add_argument("--directory", type=Path, default=Path("data/experiments/2026-10-31-completion-api"))
    args = parser.parse_args(argv)
    if args.action == "collect":
        path = args.directory / "source-manifest.json"
        if path.exists():
            raise FileExistsError("Refusing to overwrite source manifest")
        raw = args.directory / "raw"
        raw.mkdir(parents=True, exist_ok=True)
        metadata = {ticker: fetch_chart(ticker, raw) for ticker in TICKERS}
        write_json(path, {"schema_version": 1, "sources": metadata})
        print(json.dumps({"collected": {ticker: value["sha256"] for ticker, value in metadata.items()}}))
    elif args.action == "build":
        print(json.dumps({"inclusion_counts": build(args.directory)["inclusion_counts"]}))
    else:
        print(json.dumps(validate(args.directory), indent=2))


if __name__ == "__main__":
    main()
