"""Prerequisite receipt for the memorization-adjusted finance simulation (task 1.2).

Runs inside the dev container through ``dagger call run-with-secrets``. It checks
the credentialed data and model-access prerequisites, folds in the Windows host
facts that ``check_prerequisites.ps1`` recorded (llama.cpp, GGUF digest, Node,
palette validator, environment template), and writes a private receipt
(``receipt.json`` and ``summary.md``). If any required prerequisite fails, the
receipt names it and the outcome is ``stop_for_decision``.

Secrets arrive as environment variables. Only their names and whether they are
set are recorded; response text is redacted against every secret value before it
is kept, and the receipt is refused if any value would still appear in it.
Licensed data (FMP prices, BAA10Y) is reduced to first and last dates, counts and
gaps; no value series is stored.

Usage (container)::

    uv run --no-sync python -m scripts.finance_sim.check_prerequisites \
        --host-facts /inputs/host_facts.json --out /outputs
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import requests

from recall_guard.core.loader import load_cutoffs

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CUTOFFS = ROOT / "data" / "cutoffs.yaml"
RECEIPT_SCHEMA = "finance-sim-prerequisites/1"
HOST_FACTS_SCHEMA = "finance-sim-host-facts/1"

WINDOW_START = date(2016, 1, 1)
WINDOW_END = date(2026, 7, 31)
#: First and last trading days may sit a few days inside the window (holidays, weekends).
EDGE_TOLERANCE_DAYS = 7
#: A longer gap between consecutive observations means missing data, not a holiday.
MAX_GAP_DAYS = 10

#: The worked example's risky assets (``recall_guard/portfolio/prices.py``); cash is FRED DTB3.
ETFS = {"world_equity": "SWDA.L", "tech": "XLK", "gold": "IAU"}
FMP_BASE = "https://financialmodelingprep.com/stable/historical-price-eod"
FRED_BASE = "https://api.stlouisfed.org/fred/series"
VINTAGE_SERIES = ("CPIAUCSL", "T10Y2Y", "BAA10Y", "DTB3")
CPI_PROBE_AS_OF = date(2016, 2, 1)
HF_REPO = "google/gemma-3-4b-it"
HF_PROBE_FILE = "config.json"
CUTOFF_MODEL_ID = "google/gemma-3-4b-it"
TIMEOUT_S = 30

#: Credentials the run may receive, by name; values are never recorded.
SECRET_NAMES = (
    "FMP_API_TOKEN", "FRED_API_KEY", "HF_TOKEN", "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET",
    "MODAL_PROXY_BEARER", "MODAL_ENDPOINT_URL", "LOCAL_MODEL_API_KEY",
)


# ---------- pure helpers ----------

def make_check(check_id: str, group: str, passed: bool, *, required: bool = True, **detail: Any) -> dict:
    """One receipt entry; ``required`` entries decide the outcome."""
    return {"id": check_id, "group": group, "status": "pass" if passed else "fail",
            "required": required, "detail": detail}


def _parse_date(raw: Any) -> date | None:
    if not isinstance(raw, str):
        return None
    try:
        return date.fromisoformat(raw[:10])
    except ValueError:
        return None


def evaluate_span(dates: list[date], start: date, end: date, *,
                  edge_tolerance_days: int = EDGE_TOLERANCE_DAYS, max_gap_days: int = MAX_GAP_DAYS) -> dict:
    """Coverage of ``[start, end]`` by observation dates: edges within tolerance, no long gap."""
    inside = sorted({d for d in dates if start <= d <= end})
    if not inside:
        return {"first": None, "last": None, "count": 0, "max_gap_days": None,
                "covers_start": False, "covers_end": False, "gap_ok": False, "covers": False}
    gaps = [(b - a).days for a, b in zip(inside, inside[1:], strict=False)]
    max_gap = max(gaps, default=0)
    covers_start = inside[0] <= start + timedelta(days=edge_tolerance_days)
    covers_end = inside[-1] >= end - timedelta(days=edge_tolerance_days)
    gap_ok = max_gap <= max_gap_days
    return {"first": inside[0].isoformat(), "last": inside[-1].isoformat(), "count": len(inside),
            "max_gap_days": max_gap, "covers_start": covers_start, "covers_end": covers_end,
            "gap_ok": gap_ok, "covers": covers_start and covers_end and gap_ok}


def fmp_dates(payload: Any, field: str) -> list[date]:
    """Sorted dates of FMP end-of-day rows that carry a numeric ``field``."""
    if not isinstance(payload, list):
        raise ValueError("FMP payload is not a list")
    out = []
    for row in payload:
        if not isinstance(row, dict) or not isinstance(row.get(field), int | float):
            continue
        parsed = _parse_date(row.get("date"))
        if parsed is not None:
            out.append(parsed)
    return sorted(out)


def fred_observation_dates(payload: Any) -> list[date]:
    """Sorted dates of FRED observations with a value (FRED marks missing values with '.')."""
    out = []
    for obs in (payload or {}).get("observations", []):
        if obs.get("value") in (None, "", "."):
            continue
        parsed = _parse_date(obs.get("date"))
        if parsed is not None:
            out.append(parsed)
    return sorted(out)


def classify_vintages(payload: Any, window_start: date) -> dict:
    """ALFRED vintage status: point in time for the whole window, partial, or none (revised data)."""
    count = int((payload or {}).get("count", 0) or 0)
    vintages = (payload or {}).get("vintage_dates") or []
    first = _parse_date(vintages[0]) if vintages else None
    if count == 0 or first is None:
        status = "none"
    elif first <= window_start:
        status = "point_in_time"
    else:
        status = "partial"
    return {"vintage_count": count, "first_vintage": first.isoformat() if first else None,
            "status": status, "revised_data": status != "point_in_time"}


def evaluate_point_in_time(payload: Any, as_of: date) -> dict:
    """A real-time query is point in time when every value was published on or before ``as_of``."""
    observations = [o for o in (payload or {}).get("observations", []) if o.get("value") not in (None, "", ".")]
    published = [_parse_date(o.get("realtime_start")) for o in observations]
    dates = [_parse_date(o.get("date")) for o in observations]
    known_then = all(p is not None and p <= as_of for p in published)
    before = all(d is not None and d < as_of for d in dates)
    ok = bool(observations) and known_then and before
    last = max((d for d in dates if d is not None), default=None)
    return {"as_of": as_of.isoformat(), "count": len(observations),
            "last_observation": last.isoformat() if last else None, "ok": ok}


def check_registry(path: Path | str, model_id: str) -> dict:
    """The model has a cutoff in the registry and its line cites a source URL."""
    path = Path(path)
    cutoffs = load_cutoffs(path)
    pattern = re.compile(rf"^\s*{re.escape(model_id)}\s*:\s*(\S+)\s*#\s*(https?://\S+)", re.MULTILINE)
    match = pattern.search(path.read_text(encoding="utf-8"))
    cutoff = cutoffs.get(model_id)
    return make_check("registry_gemma3_4b", "registry", cutoff is not None and match is not None,
                      model_id=model_id, registry=path.name,
                      cutoff=cutoff.isoformat() if cutoff else None,
                      source=match.group(2) if match else None)


def redact(text: str, secrets: dict[str, str]) -> str:
    for name, value in secrets.items():
        if value:
            text = text.replace(value, f"[{name}]")
    return text


def find_leaks(text: str, secrets: dict[str, str]) -> list[str]:
    return [name for name, value in secrets.items() if value and value in text]


def aggregate(checks: list[dict]) -> dict:
    """Stop rule: any failed required check stops the task for a decision."""
    failed = [c["id"] for c in checks if c["required"] and c["status"] != "pass"]
    return {"status": "fail" if failed else "pass", "failed": failed,
            "decision": "stop_for_decision" if failed else "proceed"}


# ---------- live checks (HTTP through an injected session) ----------

class CountingHttp:
    """Counts requests per service; never logs URLs, parameters or headers."""

    def __init__(self, http: Any):
        self.http = http
        self.counts = {"fmp": 0, "fred": 0, "huggingface": 0}

    def _count(self, url: str) -> None:
        for service, host in (("fmp", "financialmodelingprep.com"), ("fred", "stlouisfed.org"),
                              ("huggingface", "huggingface.co")):
            if host in url:
                self.counts[service] += 1

    def get(self, url: str, **kwargs: Any) -> Any:
        self._count(url)
        return self.http.get(url, **kwargs)

    def head(self, url: str, **kwargs: Any) -> Any:
        self._count(url)
        return self.http.head(url, **kwargs)


def _call(http: Any, method: str, url: str, **kwargs: Any) -> tuple[Any, str | None]:
    """Return ``(response, None)`` or ``(None, exception class)``; exception text can hold the URL key."""
    try:
        return getattr(http, method)(url, timeout=TIMEOUT_S, **kwargs), None
    except requests.RequestException as exc:
        return None, type(exc).__name__


def _json(response: Any) -> Any:
    try:
        return response.json()
    except ValueError:
        return None


def _error_text(payload: Any, secrets: dict[str, str]) -> str | None:
    if isinstance(payload, dict):
        for key in ("Error Message", "error_message", "error", "message"):
            if isinstance(payload.get(key), str):
                return redact(payload[key], secrets)[:200]
    return None


def _fmp_span(http: Any, endpoint: str, field: str, symbol: str, token: str, secrets: dict) -> dict:
    params = {"symbol": symbol, "from": WINDOW_START.isoformat(), "to": WINDOW_END.isoformat(), "apikey": token}
    response, error = _call(http, "get", f"{FMP_BASE}/{endpoint}", params=params)
    if response is None:
        return {"endpoint": endpoint, "error": error, "covers": False}
    payload = _json(response)
    result: dict[str, Any] = {"endpoint": endpoint, "http_status": response.status_code}
    if response.status_code != 200 or not isinstance(payload, list):
        result.update(covers=False, error=_error_text(payload, secrets) or "unexpected payload")
        return result
    result.update(evaluate_span(fmp_dates(payload, field), WINDOW_START, WINDOW_END))
    return result


def check_fmp(http: Any, token: str, secrets: dict) -> list[dict]:
    checks = []
    for role, symbol in ETFS.items():
        check_id = f"fmp_adjusted_{role}"
        if not token:
            checks.append(make_check(check_id, "data", False, symbol=symbol,
                                     reason="FMP_API_TOKEN is not set; no request made"))
            continue
        adjusted = _fmp_span(http, "dividend-adjusted", "adjClose", symbol, token, secrets)
        detail = {"symbol": symbol, "field": "adjClose", **adjusted}
        if not adjusted["covers"]:
            # Informational only: whether unadjusted closes would cover the window.
            fallback = _fmp_span(http, "light", "price", symbol, token, secrets)
            detail["unadjusted_fallback"] = fallback
        checks.append(make_check(check_id, "data", bool(adjusted["covers"]), **detail))
    return checks


def check_fred(http: Any, key: str, secrets: dict) -> list[dict]:
    ids = ["fred_key_works", "cpi_point_in_time", *(f"vintages_{s}" for s in VINTAGE_SERIES[1:]), "dtb3_coverage"]
    if not key:
        reason = "FRED_API_KEY is not set; no request made"
        return [make_check(i, "data", False, reason=reason) for i in ids]

    def fred(path: str, **params: Any) -> tuple[Any, dict]:
        query = {"api_key": key, "file_type": "json", **params}
        response, error = _call(http, "get", f"{FRED_BASE}/{path}", params=query)
        if response is None:
            return None, {"error": error}
        payload = _json(response)
        if response.status_code != 200:
            return None, {"http_status": response.status_code, "error": _error_text(payload, secrets)}
        return payload, {"http_status": 200}

    vintages: dict[str, dict] = {}
    first, meta = fred("vintagedates", series_id="CPIAUCSL", limit=1, sort_order="asc")
    checks = [make_check("fred_key_works", "data", first is not None, series_id="CPIAUCSL", **meta)]
    if first is None:
        reason = "FRED key check failed; no further FRED request made"
        return checks + [make_check(i, "data", False, reason=reason) for i in ids[1:]]
    vintages["CPIAUCSL"] = classify_vintages(first, WINDOW_START)
    for series in VINTAGE_SERIES[1:]:
        payload, meta = fred("vintagedates", series_id=series, limit=1, sort_order="asc")
        vintages[series] = classify_vintages(payload, WINDOW_START) if payload is not None else meta

    probe_payload, meta = fred("observations", series_id="CPIAUCSL",
                               realtime_start=CPI_PROBE_AS_OF.isoformat(), realtime_end=CPI_PROBE_AS_OF.isoformat(),
                               observation_start=(CPI_PROBE_AS_OF - timedelta(days=120)).isoformat())
    probe = evaluate_point_in_time(probe_payload, CPI_PROBE_AS_OF) if probe_payload is not None else meta
    cpi_ok = vintages["CPIAUCSL"]["status"] == "point_in_time" and bool(probe.get("ok"))
    checks.append(make_check("cpi_point_in_time", "data", cpi_ok, series_id="CPIAUCSL",
                             vintages=vintages["CPIAUCSL"], probe=probe))
    for series in VINTAGE_SERIES[1:]:
        recorded = "status" in vintages[series]
        checks.append(make_check(f"vintages_{series}", "data", recorded, series_id=series, **vintages[series]))

    dtb3, meta = fred("observations", series_id="DTB3", observation_start=WINDOW_START.isoformat(),
                      observation_end=WINDOW_END.isoformat())
    span = evaluate_span(fred_observation_dates(dtb3), WINDOW_START, WINDOW_END) if dtb3 is not None else meta
    checks.append(make_check("dtb3_coverage", "data", bool(span.get("covers")), series_id="DTB3",
                             role="cash leg", **span))
    return checks


def check_hf(http: Any, token: str) -> dict:
    if not token:
        return make_check("hf_gated_revision", "tooling", False, repo=HF_REPO,
                          reason="HF_TOKEN is not set; no request made")
    headers = {"Authorization": f"Bearer {token}"}
    response, error = _call(http, "get", f"https://huggingface.co/api/models/{HF_REPO}", headers=headers)
    if response is None or response.status_code != 200:
        return make_check("hf_gated_revision", "tooling", False, repo=HF_REPO,
                          info_status=getattr(response, "status_code", None), error=error)
    info = _json(response) or {}
    revision = info.get("sha") if isinstance(info.get("sha"), str) else None
    if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
        return make_check("hf_gated_revision", "tooling", False, repo=HF_REPO, info_status=200,
                          reason="no commit revision in the model info")
    probe, error = _call(http, "head", f"https://huggingface.co/{HF_REPO}/resolve/{revision}/{HF_PROBE_FILE}",
                         headers=headers, allow_redirects=False)
    status = getattr(probe, "status_code", None)
    error_code = (probe.headers.get("x-error-code") if probe is not None else None) or error
    return make_check("hf_gated_revision", "tooling", status in (200, 302, 307), repo=HF_REPO,
                      gated=info.get("gated"), revision=revision, probe_file=HF_PROBE_FILE,
                      file_status=status, error_code=error_code)


def load_host_facts(path: Path) -> list[dict]:
    """Host checks recorded by the PowerShell wrapper; a missing or malformed file is a failed check."""
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        return [make_check("host_facts", "host", False, reason=f"host facts unreadable: {type(exc).__name__}")]
    if not isinstance(doc, dict) or doc.get("schema") != HOST_FACTS_SCHEMA or not isinstance(doc.get("checks"), list):
        return [make_check("host_facts", "host", False, reason="host facts schema mismatch")]
    out = []
    for entry in doc["checks"]:
        valid = isinstance(entry, dict) and isinstance(entry.get("id"), str) and entry.get("status") in ("pass", "fail")
        if not valid:
            return [make_check("host_facts", "host", False, reason="malformed host check entry")]
        out.append({"id": entry["id"], "group": entry.get("group", "host"), "status": entry["status"],
                    "required": bool(entry.get("required", True)), "detail": entry.get("detail") or {}})
    return out


def run_checks(env: dict[str, str], http: Any, host_facts: Path, cutoffs: Path) -> dict:
    secrets = {name: env.get(name, "") or "" for name in SECRET_NAMES}
    counting = CountingHttp(http)
    checks = load_host_facts(host_facts)
    checks += check_fmp(counting, secrets["FMP_API_TOKEN"], secrets)
    checks += check_fred(counting, secrets["FRED_API_KEY"], secrets)
    checks.append(check_registry(cutoffs, CUTOFF_MODEL_ID))
    checks.append(check_hf(counting, secrets["HF_TOKEN"]))
    checks.append(make_check("modal_credentials", "tooling",
                             bool(secrets["MODAL_TOKEN_ID"] and secrets["MODAL_TOKEN_SECRET"]),
                             names=["MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"]))
    checks.append(make_check("local_model_api_key", "credentials", bool(secrets["LOCAL_MODEL_API_KEY"]),
                             names=["LOCAL_MODEL_API_KEY"]))
    return {
        "schema": RECEIPT_SCHEMA,
        "task": "1.2",
        "feature": "memorization-adjusted-finance-sim",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "window": {"start": WINDOW_START.isoformat(), "end": WINDOW_END.isoformat(),
                   "edge_tolerance_days": EDGE_TOLERANCE_DAYS, "max_gap_days": MAX_GAP_DAYS},
        "assets": {**ETFS, "cash": "FRED DTB3"},
        "credentials": {name: bool(value) for name, value in secrets.items()},
        "requests": dict(counting.counts),
        "licensing": "FMP prices and BAA10Y (Moody's) are licensed: only dates, counts and gaps are kept here.",
        "checks": checks,
        "outcome": aggregate(checks),
    }


def render_summary(receipt: dict) -> str:
    outcome = receipt["outcome"]
    lines = [
        "# Finance-sim prerequisites (task 1.2)",
        "",
        f"Generated {receipt['generated_at']}. Outcome: **{outcome['status']}**, decision `{outcome['decision']}`.",
        "",
        "| Check | Group | Status | Required |",
        "|-------|-------|--------|----------|",
    ]
    lines += [f"| {c['id']} | {c['group']} | {c['status']} | {'yes' if c['required'] else 'no'} |"
              for c in receipt["checks"]]
    lines += ["", "Failed prerequisites: " + (", ".join(outcome["failed"]) or "none") + ".",
              "", "Credentials by name: " + ", ".join(f"{k}={'set' if v else 'not set'}"
                                                     for k, v in receipt["credentials"].items()) + ".",
              "", f"Requests: {receipt['requests']}.", "", receipt["licensing"], ""]
    return "\n".join(lines)


def make_http() -> Any:
    return requests.Session()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host-facts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cutoffs", type=Path, default=DEFAULT_CUTOFFS)
    args = parser.parse_args(argv)
    env = {name: os.environ.get(name, "") for name in SECRET_NAMES}
    receipt = run_checks(env, make_http(), args.host_facts, args.cutoffs)
    text = json.dumps(receipt, indent=2, sort_keys=False) + "\n"
    summary = render_summary(receipt)
    leaks = find_leaks(text + summary, env)
    if leaks:
        print(f"refusing to write a receipt that contains secret values: {leaks}", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "receipt.json").write_text(text, encoding="utf-8")
    (args.out / "summary.md").write_text(summary, encoding="utf-8")
    for check in receipt["checks"]:
        print(f"{check['status']:4}  {check['id']}")
    print(f"outcome: {receipt['outcome']['status']} ({receipt['outcome']['decision']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
