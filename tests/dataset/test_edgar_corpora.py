"""Offline tests for the SEC EDGAR 8-K EX-99.1 corpus builder (release-1.0 M3.3, D1).

No test makes a network call: HTTP goes through ``FakeEdgar``, a synthetic
in-memory EDGAR that serves form.idx, index.json, -index.htm and exhibits.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from recall_guard.dataset import edgar_corpora as ec

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "edgar"

# Small synthetic setup: memorized window 2022-01-01..2022-06-29,
# gap 2022-06-30..2023-03-31, control 2023-04-01..2023-12-31.
CUTOFFS = {"early/model": date(2022, 6, 30), "late/model": date(2023, 3, 31)}
START = date(2022, 1, 1)
TODAY = date(2023, 12, 31)
TARGETS = ec.Targets(calibration_memorized=4, calibration_control=5, held_out_memorized=3, held_out_control=3)


class FakeResponse:
    def __init__(self, status: int, body: bytes = b"", headers: dict[str, str] | None = None) -> None:
        self.status_code = status
        self.content = body
        self.headers = headers or {}


class FakeSession:
    """Returns queued responses; records every URL requested."""

    def __init__(self, responses: list[FakeResponse] | None = None) -> None:
        self.responses = list(responses or [])
        self.calls: list[str] = []

    def get(self, url: str, timeout: float) -> FakeResponse:
        self.calls.append(url)
        return self.responses.pop(0) if self.responses else FakeResponse(200, b"ok")


def _paragraph(n: int) -> str:
    return (
        f"The company today announced results for the period and said that revenue for item {n} increased "
        "compared with the same period of the prior year, driven by demand for its products and services. "
        "The chief executive officer said the teams continued to execute on the strategy and that the company "
        "will invest in growth with a strong balance sheet. The board of directors also approved a dividend "
        "that is payable to shareholders of record at the close of business, and the company reaffirmed its "
        "outlook for the full year in the statement issued on the same day."
    )


def exhibit_html(title: str, n: int) -> str:
    return (
        "<html><body><p>Exhibit 99.1</p><p>FOR IMMEDIATE RELEASE</p>"
        f"<p><b>{title}</b></p><p>{_paragraph(n)}</p>"
        "<table><tr><td>Revenue</td><td>1.0</td></tr></table></body></html>"
    )


def index_htm(folder_path: str, doc: str) -> str:
    return (
        "<table class='tableFile'><tr><th>Seq</th><th>Description</th><th>Document</th><th>Type</th></tr>"
        f"<tr><td>1</td><td>8-K</td><td><a href='{folder_path}/d8k.htm'>d8k.htm</a></td><td>8-K</td></tr>"
        f"<tr><td>2</td><td>PR</td><td><a href='{folder_path}/{doc}'>{doc}</a></td><td>EX-99.1</td></tr></table>"
    )


class FakeEdgar:
    """Synthetic EDGAR. ``filings`` maps quarter label -> list of filing dicts."""

    def __init__(self, filings: dict[str, list[dict[str, Any]]]) -> None:
        self.filings = filings
        self.by_folder: dict[str, dict[str, Any]] = {}
        for rows in filings.values():
            for f in rows:
                self.by_folder[f"/Archives/edgar/data/{int(f['cik'])}/{f['accession'].replace('-', '')}"] = f
        self.calls: list[str] = []

    def get(self, url: str, timeout: float) -> FakeResponse:
        self.calls.append(url)
        path = url.removeprefix(ec.SEC_BASE)
        if path.endswith("/form.idx"):
            _, _, _, _, year, qtr, _ = path.split("/")
            return FakeResponse(200, self._form_idx(f"{year}Q{qtr[-1]}").encode("latin-1"))
        folder, name = path.rsplit("/", 1)
        filing = self.by_folder.get(folder)
        if filing is None:
            return FakeResponse(404)
        if name == "index.json":
            items = [{"name": "d8k.htm", "type": "text.gif"}, {"name": filing["doc"], "type": "text.gif"}]
            return FakeResponse(200, json.dumps({"directory": {"item": items}}).encode())
        if name.endswith("-index.htm"):
            if filing.get("no_exhibit"):
                return FakeResponse(200, b"<table></table>")
            return FakeResponse(200, index_htm(folder, filing["doc"]).encode())
        if name == filing["doc"]:
            return FakeResponse(200, exhibit_html(filing["title"], filing["n"]).encode())
        return FakeResponse(404)

    def _form_idx(self, quarter: str) -> str:
        lines = ["Description: Master Index", "", "Form Type   Company Name" + " " * 50 + "CIK         Date Filed  File Name", "-" * 100]
        for f in self.filings.get(quarter, []):
            file_name = f"edgar/data/{int(f['cik'])}/{f['accession']}.txt"
            lines.append(f"{f['form']:<12}{f['company']:<62}{f['cik']:<12}{f['date'].isoformat():<12}{file_name}")
        return "\n".join(lines) + "\n"


def make_filings(extra: dict[str, list[dict[str, Any]]] | None = None) -> dict[str, list[dict[str, Any]]]:
    """Six 8-Ks per quarter on every month start, plus noise forms."""
    filings: dict[str, list[dict[str, Any]]] = {}
    n = 0
    for year in (2022, 2023):
        for qtr in range(1, 5):
            rows = []
            for i in range(6):
                n += 1
                month = (qtr - 1) * 3 + 1 + i % 3
                rows.append(
                    {
                        "form": "8-K" if i != 5 else "10-Q",
                        "company": f"COMPANY {n} INC",
                        "cik": str(100000 + n),
                        "date": date(year, month, 1 + i),
                        "accession": f"0000{n:06d}-{year % 100:02d}-{n:06d}",
                        "doc": f"ex991_{n}.htm",
                        "title": f"Company {n} Announces Quarterly Results Number {n}",
                        "n": n,
                    }
                )
            filings[f"{year}Q{qtr}"] = rows
    for quarter, rows in (extra or {}).items():
        filings.setdefault(quarter, []).extend(rows)
    return filings


def _filing(n: int, day: date, **over: Any) -> dict[str, Any]:
    base = {
        "form": "8-K",
        "company": f"EDGE {n} CORP",
        "cik": str(900000 + n),
        "date": day,
        "accession": f"0009{n:06d}-{day.year % 100:02d}-{n:06d}",
        "doc": f"edge_{n}.htm",
        "title": f"Edge Corp {n} Announces A Boundary Event Today",
        "n": 10000 + n,
    }
    base.update(over)
    return base


def make_client(fake: Any, **kwargs: Any) -> ec.EdgarClient:
    limiter = ec.RateLimiter(clock=lambda: 0.0, sleep=lambda _s: None)
    return ec.EdgarClient(session=fake, limiter=limiter, sleep=lambda _s: None, **kwargs)


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# ---------- parsing ------------------------------------------------------------


def test_parse_form_index_keeps_exact_8k_rows() -> None:
    entries = ec.parse_form_index((FIXTURES / "form_sample.idx").read_text(encoding="latin-1"))
    assert [(e.company, e.cik, e.date_filed) for e in entries] == [
        ("ACME CORP", "1000045", date(2015, 1, 2)),
        ("WIDGETS & SONS  HOLDINGS INC", "20", date(2015, 3, 15)),
    ]
    acme = entries[0]
    assert acme.accession == "0001193125-15-000123"
    assert acme.folder_url == "https://www.sec.gov/Archives/edgar/data/1000045/000119312515000123"


def test_parse_form_index_requires_header() -> None:
    with pytest.raises(ValueError, match="header"):
        ec.parse_form_index("no header here\n")


def test_index_htm_selects_ex991_not_ex992() -> None:
    url = ec.exhibit_from_index_htm((FIXTURES / "index_sample.htm").read_text(encoding="utf-8"))
    assert url == "https://www.sec.gov/Archives/edgar/data/1000045/000119312515000123/dex991.htm"


def test_index_json_selection_uses_types_when_present() -> None:
    folder = "https://www.sec.gov/Archives/edgar/data/1/0001"
    typed = {"directory": {"item": [
        {"name": "d8k.htm", "type": "8-K"},
        {"name": "ex992.htm", "type": "EX-99.2"},
        {"name": "ex991.htm", "type": "EX-99.1"},
    ]}}
    assert ec.index_json_has_types(typed)
    assert ec.exhibit_from_index_json(typed, folder) == f"{folder}/ex991.htm"
    icons = {"directory": {"item": [{"name": "ex991.htm", "type": "text.gif"}]}}
    assert not ec.index_json_has_types(icons)
    assert ec.exhibit_from_index_json(icons, folder) is None


def test_locate_exhibit_falls_back_to_index_htm_only_without_types() -> None:
    folder = "/Archives/edgar/data/7/000000000722000001"
    typed = json.dumps({"directory": {"item": [{"name": "pr.htm", "type": "EX-99.1"}]}}).encode()
    entry = ec.IndexEntry("8-K", "X", "7", date(2022, 1, 3), "edgar/data/7/0000000007-22-000001.txt")
    session = FakeSession([FakeResponse(200, typed)])
    assert ec.locate_exhibit(make_client(session), entry) == f"{ec.SEC_BASE}{folder}/pr.htm"
    assert len(session.calls) == 1
    untyped = json.dumps({"directory": {"item": [{"name": "pr.htm", "type": "text.gif"}]}}).encode()
    session = FakeSession([FakeResponse(200, untyped), FakeResponse(200, index_htm(folder, "pr.htm").encode())])
    assert ec.locate_exhibit(make_client(session), entry) == f"{ec.SEC_BASE}{folder}/pr.htm"
    assert session.calls[1].endswith("/0000000007-22-000001-index.htm")


def test_html_to_text_and_prompt() -> None:
    html = (FIXTURES / "exhibit_sample.htm").read_text(encoding="utf-8")
    text = ec.html_to_text(html)
    assert "do not include" not in " ".join(text.lines)
    assert all("  " not in line for line in text.lines)
    assert 0 < text.table_fraction < 0.2
    built = ec.build_exhibit_prompt(html)
    assert isinstance(built, tuple)
    title, prompt = built
    assert title == "Acme Corp Reports Record Fourth Quarter and Full Year 2014 Results"
    assert prompt.startswith(title + "\n\nSPRINGFIELD, Jan. 2, 2015 -- Acme Corp")
    assert len(prompt) <= 1500


@pytest.mark.parametrize(
    ("html", "reason"),
    [
        ("<p>Exhibit 99.1</p><p>Acme Corp Announces A Short Thing</p><p>Too short.</p>", "too short"),
        ("<table>" + "<tr><td>Revenue line item</td><td>1,234.5</td></tr>" * 200 + "</table>", "mostly tables"),
        (
            "<p>Acme Corp annonce ses résultats annuels</p><p>"
            + "La société a annoncé aujourd'hui une hausse du chiffre d'affaires pour l'exercice. " * 10
            + "</p>",
            "not English",
        ),
    ],
)
def test_exhibit_skip_reasons(html: str, reason: str) -> None:
    built = ec.build_exhibit_prompt(html)
    assert isinstance(built, str) and reason in built


# ---------- windows and plan ----------------------------------------------------------


def test_windows_exclude_cutoff_days_and_gap() -> None:
    w = ec.compute_windows(CUTOFFS, START, TODAY)
    assert w.label_for(date(2022, 6, 29)) == 1
    assert w.label_for(date(2022, 6, 30)) is None  # exactly the earliest cutoff
    assert w.label_for(date(2022, 12, 1)) is None  # gap
    assert w.label_for(date(2023, 3, 31)) is None  # exactly the latest cutoff
    assert w.label_for(date(2023, 4, 1)) == 0
    assert w.label_for(date(2021, 12, 31)) is None  # before start


def test_plan_for_default_models_uses_registry_extremes() -> None:
    from recall_guard.core.loader import load_cutoffs

    root = Path(__file__).resolve().parents[2]
    cutoffs = ec.select_cutoffs(load_cutoffs(root / "data" / "cutoffs.yaml"), ec.DEFAULT_MODELS)
    p = ec.plan(cutoffs, start=ec.DEFAULT_START, today=date(2026, 11, 8), targets=ec.Targets(), seed=0, max_requests=3000)
    assert p["earliest_cutoff"] == "2023-03-31" and p["latest_cutoff"] == "2024-08-31"
    assert p["memorized"]["window"] == ["2015-01-01", "2023-03-30"]
    assert p["control"]["window"] == ["2024-09-01", "2026-11-08"]
    assert len(p["memorized"]["quarters"]) == 33 and p["memorized"]["per_quarter_cap"] == 4
    assert len(p["control"]["quarters"]) == 10 and p["control"]["per_quarter_cap"] == 15


def test_select_cutoffs_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="missing"):
        ec.select_cutoffs(CUTOFFS, ["early/model", "nope/model"])


# ---------- full build -----------------------------------------------------------


BOUNDARY = {
    "2022Q2": [_filing(1, date(2022, 6, 30))],  # exactly earliest cutoff
    "2023Q1": [_filing(2, date(2023, 3, 31))],  # exactly latest cutoff (gap quarter)
    "2022Q3": [_filing(3, date(2022, 9, 1))],  # gap
}


def _build(tmp_path: Path, filings: dict[str, list[dict[str, Any]]], **kwargs: Any) -> tuple[ec.BuildResult, FakeEdgar]:
    fake = FakeEdgar(filings)
    client = make_client(fake, cache_dir=kwargs.pop("cache_dir", None), max_requests=kwargs.pop("max_requests", 3000))
    result = ec.build_edgar_corpora(tmp_path / "out", CUTOFFS, client, seed=kwargs.pop("seed", 7), start=START,
                                    today=TODAY, targets=TARGETS)
    return result, fake


def test_build_writes_four_disjoint_files_with_correct_dates(tmp_path: Path) -> None:
    result, _ = _build(tmp_path, make_filings(BOUNDARY))
    assert result.complete
    rows = {key: read_rows(result.paths[key]) for key in ec.OUTPUT_FILES}
    assert {k: len(v) for k, v in rows.items()} == {
        "calibration_memorized": 4, "calibration_control": 5, "held_out_memorized": 3, "held_out_control": 3,
    }
    all_rows = [r for v in rows.values() for r in v]
    urls = [r["metadata"]["url"] for r in all_rows]
    titles = [r["prompt"].split("\n\n")[0] for r in all_rows]
    company_dates = [(r["metadata"]["cik"], r["metadata"]["published_at"]) for r in all_rows]
    assert len(set(urls)) == len(set(titles)) == len(set(company_dates)) == len(all_rows)
    for key, (_, label) in ec.OUTPUT_FILES.items():
        for r in rows[key]:
            assert r["label"] == label
            assert r["metadata"]["source"] == "sec-edgar-8k-ex99.1"
            published = date.fromisoformat(r["metadata"]["published_at"])
            if label == 1:
                assert START <= published < CUTOFFS["early/model"]
            else:
                assert CUTOFFS["late/model"] < published <= TODAY
            assert set(r["metadata"]) == {"published_at", "source", "url", "cik", "company", "accession"}
    assert not any(r["metadata"]["company"].startswith("EDGE") for r in all_rows)


def test_build_is_stratified_and_deterministic(tmp_path: Path) -> None:
    a, _ = _build(tmp_path / "a", make_filings())
    b, _ = _build(tmp_path / "b", make_filings())
    c, _ = _build(tmp_path / "c", make_filings(), seed=8)
    for key in ec.OUTPUT_FILES:
        assert a.paths[key].read_bytes() == b.paths[key].read_bytes()
    assert any(a.paths[k].read_bytes() != c.paths[k].read_bytes() for k in ec.OUTPUT_FILES)
    # memorized pool 7 over 2 quarters: cap 5; control pool 8 over 3 quarters: cap 4
    for key in ("calibration_memorized", "held_out_memorized"):
        assert set(a.manifest["files"][ec.OUTPUT_FILES[key][0]]["per_quarter"]) <= {"2022Q1", "2022Q2"}
    pooled: dict[str, int] = {}
    for key in ("calibration_control", "held_out_control"):
        for q, n in a.manifest["files"][ec.OUTPUT_FILES[key][0]]["per_quarter"].items():
            pooled[q] = pooled.get(q, 0) + n
    assert set(pooled) == {"2023Q2", "2023Q3", "2023Q4"} and max(pooled.values()) <= 4


def test_dedupe_by_company_date_and_title(tmp_path: Path) -> None:
    dup_day = date(2023, 4, 3)
    extra = {
        "2023Q2": [
            _filing(10, dup_day, cik="555", company="SAME CO"),
            _filing(11, dup_day, cik="555", company="SAME CO", title="Different Title For The Same Company Day"),
            _filing(12, date(2023, 5, 3), title="Shared Headline Announces Results Today"),
            _filing(13, date(2023, 5, 4), title="Shared Headline Announces Results Today"),
            _filing(14, date(2023, 5, 5), no_exhibit=True),
        ]
    }
    fake = FakeEdgar(make_filings(extra))
    client = make_client(fake)
    seen = ec._Seen()
    records, cap_hit = ec.sample_pool(client, label=0, window=(date(2023, 4, 1), TODAY), pool_size=100, seed=1, seen=seen)
    assert not cap_hit
    ciks_days = [(r.cik, r.published_at) for r in records]
    assert len(ciks_days) == len(set(ciks_days))
    assert sum(r.cik == "555" for r in records) == 1
    assert sum("Shared Headline" in r.prompt for r in records) == 1
    assert not any(r.accession.startswith("0009000014") for r in records)


def test_manifest_hashes_and_metadata(tmp_path: Path) -> None:
    result, _ = _build(tmp_path, make_filings())
    manifest = json.loads(result.paths["manifest"].read_text(encoding="utf-8"))
    for rel, entry in manifest["files"].items():
        data = (tmp_path / "out" / rel).read_bytes()
        assert entry["sha256"] == hashlib.sha256(data).hexdigest()
        assert entry["rows"] == len(data.splitlines())
        assert sum(entry["per_quarter"].values()) == entry["rows"]
    assert manifest["seed"] == 7
    assert manifest["source"] == "sec-edgar-8k-ex99.1"
    assert manifest["cutoffs"] == {"early/model": "2022-06-30", "late/model": "2023-03-31"}
    assert manifest["retrieval_date"] == "2023-12-31"
    assert manifest["requests_used"] > 0 and manifest["complete"] is True


def test_cache_makes_rerun_request_free(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    first, _ = _build(tmp_path / "a", make_filings(), cache_dir=cache)
    second, fake = _build(tmp_path / "b", make_filings(), cache_dir=cache)
    assert fake.calls == [] and second.manifest["requests_used"] == 0
    for key in ec.OUTPUT_FILES:
        assert first.paths[key].read_bytes() == second.paths[key].read_bytes()


def test_request_cap_stops_build_and_reports_shortfall(tmp_path: Path) -> None:
    result, fake = _build(tmp_path, make_filings(), max_requests=10)
    assert len(fake.calls) == 10
    assert result.manifest["request_cap_hit"] is True and result.complete is False
    assert result.manifest["requests_used"] == 10


# ---------- HTTP client ---------------------------------------------------------


def test_rate_limiter_spaces_requests_at_five_per_second() -> None:
    now = [0.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    limiter = ec.RateLimiter(5.0, clock=lambda: now[0], sleep=sleep)
    session = FakeSession()
    client = ec.EdgarClient(session=session, limiter=limiter, sleep=sleep)
    for i in range(6):
        client.get(f"{ec.SEC_BASE}/x/{i}")
    assert len(session.calls) == 6
    assert now[0] == pytest.approx(1.0)  # 5 gaps of 0.2 s
    assert all(s == pytest.approx(0.2) for s in sleeps)


def test_request_cap_counts_retries_and_raises() -> None:
    session = FakeSession([FakeResponse(429, headers={"Retry-After": "1"}), FakeResponse(200, b"a")])
    delays: list[float] = []
    client = make_client(session, max_requests=3)
    client.sleep = delays.append
    assert client.get(f"{ec.SEC_BASE}/a") == b"a"
    assert client.requests_used == 2 and delays == [1.0]
    client.get(f"{ec.SEC_BASE}/b")
    with pytest.raises(ec.RequestCapReached):
        client.get(f"{ec.SEC_BASE}/c")
    assert len(session.calls) == 3


def test_retry_is_bounded_on_503() -> None:
    session = FakeSession([FakeResponse(503)] * 10)
    delays: list[float] = []
    client = make_client(session, max_retries=2, backoff_seconds=1.0)
    client.sleep = delays.append
    with pytest.raises(ec.EdgarHTTPError) as excinfo:
        client.get(f"{ec.SEC_BASE}/a")
    assert excinfo.value.status == 503
    assert len(session.calls) == 3 and delays == [1.0, 2.0]


def test_404_is_cached_as_negative(tmp_path: Path) -> None:
    session = FakeSession([FakeResponse(404)])
    client = make_client(session, cache_dir=tmp_path)
    for _ in range(2):
        with pytest.raises(ec.EdgarHTTPError):
            client.get(f"{ec.SEC_BASE}/Archives/missing.htm")
    assert len(session.calls) == 1


def test_cache_rejects_foreign_and_dot_urls(tmp_path: Path) -> None:
    client = make_client(FakeSession(), cache_dir=tmp_path)
    with pytest.raises(ValueError):
        client.get("https://example.com/x")
    with pytest.raises(ValueError):
        client.get(f"{ec.SEC_BASE}/Archives/../../etc/passwd")


def test_user_agent_required_and_never_echoed() -> None:
    with pytest.raises(ec.EdgarConfigError, match="SEC_USER_AGENT is not set"):
        ec.require_user_agent({})
    with pytest.raises(ec.EdgarConfigError) as excinfo:
        ec.require_user_agent({"SEC_USER_AGENT": "secretvalue"})
    assert "secretvalue" not in str(excinfo.value)
    assert ec.require_user_agent({"SEC_USER_AGENT": "Jane Doe jane@example.com"}) == "Jane Doe jane@example.com"
