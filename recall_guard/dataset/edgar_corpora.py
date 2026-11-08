"""SEC EDGAR 8-K press-release corpora for calibration and held-out evaluation.

Release-1.0 task M3.3 (D1 data source change). The FMP calibration corpus mixed
two FMP sources across labels and covered only three publication dates. This
builder draws BOTH classes from one public-domain source -- the EX-99.1 exhibit
(usually the press release) of Form 8-K filings on SEC EDGAR -- so source is
constant and only the publication date relative to the model cutoffs differs.

Discovery and parsing
---------------------
1. Quarterly full index ``/Archives/edgar/full-index/{YYYY}/QTR{n}/form.idx``
   (fixed-width). Form type and company are cut at the header's column
   positions; CIK, date filed and file name are the last three
   whitespace-free tokens. Only rows with form type exactly ``8-K`` are used.
2. Filing directory ``/Archives/edgar/data/{cik}/{accession-no-dashes}/index.json``.
   If its items carry an SEC document type (``EX-99.1``) that item is chosen.
   The directory listing normally carries only icon types (``text.gif``), so the
   usual path is the fallback: the filing index page ``{accession}-index.htm``,
   whose document table has a ``Type`` column. Filename heuristics are NOT used:
   a wrong pick (EX-99.2, EX-10.1) would silently change the source.
3. The exhibit (``.htm``/``.html``/``.txt``) is converted to text with the
   stdlib ``html.parser``; whitespace is normalised. Title = first meaningful
   line (boilerplate such as "Exhibit 99.1" or "FOR IMMEDIATE RELEASE" is
   skipped); prompt = ``title + "\\n\\n" + body`` capped at 1500 characters,
   matching the FMP builder. Exhibits that are too short, mostly tables or not
   English-looking are skipped with one WARNING each.

Row schema (same as the FMP builder, with extra metadata)::

    {"prompt": str, "label": 1|0,
     "metadata": {"published_at": "YYYY-MM-DD",   # the 8-K filing date
                  "source": "sec-edgar-8k-ex99.1", "url": <exhibit URL>,
                  "cik": str, "company": str, "accession": str}}

``published_at`` is the filing date from the full index. The press release is
usually issued the same day or one business day earlier; the filing date is the
only date EDGAR certifies, so it is used for both classes.

Sampling
--------
Memorized (label 1): filed on/after ``start`` and strictly BEFORE the earliest
cutoff of the selected models. Control (label 0): filed strictly AFTER the latest
cutoff and on/before today. A filing dated exactly on a cutoff, or between the
cutoffs, goes nowhere. Each class is one pool sized calibration + held-out,
filled round-robin across calendar quarters (quarter order reshuffled every
round with the seed) with at most ``ceil(pool / quarters) + 1`` rows per
quarter. Candidates within a quarter are a seeded shuffle of the index rows.
Dedupe is pool-wide across both classes by exhibit URL, title hash and
CIK + filing date, so the four output files are disjoint. Each pool is then
split by a seeded shuffle into calibration and held-out.

HTTP and fair access
--------------------
One ``requests.Session`` with ``User-Agent`` from ``SEC_USER_AGENT`` (required,
never logged). At most 5 requests per second (sleep-based limiter), bounded
retries with backoff on 429/503, a hard cap on total network attempts, and an
optional on-disk cache (200 bodies and 404 markers) so reruns do not refetch.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import re
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from recall_guard.dataset.fmp_corpora import _build_prompt, _title_hash

logger = logging.getLogger(__name__)

SOURCE = "sec-edgar-8k-ex99.1"
SEC_BASE = "https://www.sec.gov"
USER_AGENT_ENV = "SEC_USER_AGENT"

#: The six models measured in release-1.0 (measurement-plan.md).
DEFAULT_MODELS: tuple[str, ...] = (
    "microsoft/phi-4-mini-instruct",
    "meta/llama-3.2-3b-instruct",
    "google/gemma-3-4b-it",
    "meta/llama-3.1-8b-instruct",
    "xtuner/llava-llama-3-8b-v1_1",
    "openai/gpt-oss-20b",
)
DEFAULT_START = date(2015, 1, 1)
DEFAULT_MAX_REQUESTS = 3000
MAX_REQUESTS_PER_SECOND = 5.0

#: Body text (after the title) shorter than this is skipped.
MIN_BODY_CHARS = 400
#: Share of document text inside table cells above which it is "mostly tables".
MAX_TABLE_FRACTION = 0.8
#: Minimum share of common English function words among excerpt tokens.
MIN_STOPWORD_FRACTION = 0.12
#: Candidates tried per quarter pick before the quarter is marked exhausted.
MAX_ATTEMPTS_PER_PICK = 12

_EX991_TYPE = re.compile(r"^EX-99\.0?1$", re.IGNORECASE)
_TEXT_SUFFIXES = (".htm", ".html", ".txt")
_STOPWORDS = frozenset(
    "the and of to in a for is on with that by as its will are from at an be this has have our which".split()
)
_BOILERPLATE_LINE = re.compile(
    r"^(exhibit|ex-?99|press release|news release|for immediate release|"
    r"media (contact|release)|investor (contact|relations)|contacts?\b|page \d+|-?\s*\d+\s*-?$)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Targets:
    """Row targets per output file (measurement-plan.md, D1)."""

    calibration_memorized: int = 40
    calibration_control: int = 100
    held_out_memorized: int = 40
    held_out_control: int = 40

    @property
    def memorized_pool(self) -> int:
        return self.calibration_memorized + self.held_out_memorized

    @property
    def control_pool(self) -> int:
        return self.calibration_control + self.held_out_control


OUTPUT_FILES: dict[str, tuple[str, int]] = {
    # key -> (relative path, label)
    "calibration_memorized": ("calibration/is_memorized.jsonl", 1),
    "calibration_control": ("calibration/oos_control.jsonl", 0),
    "held_out_memorized": ("held_out/is_memorized.jsonl", 1),
    "held_out_control": ("held_out/oos_control.jsonl", 0),
}


# ---------- errors -----------------------------------------------------------


class EdgarConfigError(RuntimeError):
    """Configuration problem detected before any request (e.g. no User-Agent)."""


class RequestCapReached(RuntimeError):
    """The hard cap on network requests was reached."""


class EdgarHTTPError(RuntimeError):
    """Non-retryable or retry-exhausted HTTP failure."""

    def __init__(self, url: str, status: int) -> None:
        super().__init__(f"HTTP {status} for {url}")
        self.url = url
        self.status = status


# ---------- quarters and windows -----------------------------------------------


@dataclass(frozen=True, order=True)
class Quarter:
    year: int
    qtr: int

    @classmethod
    def of(cls, day: date) -> Quarter:
        return cls(day.year, (day.month - 1) // 3 + 1)

    def next(self) -> Quarter:
        return Quarter(self.year + 1, 1) if self.qtr == 4 else Quarter(self.year, self.qtr + 1)

    @property
    def label(self) -> str:
        return f"{self.year}Q{self.qtr}"

    @property
    def index_url(self) -> str:
        return f"{SEC_BASE}/Archives/edgar/full-index/{self.year}/QTR{self.qtr}/form.idx"


def quarters_between(first: date, last: date) -> list[Quarter]:
    """Calendar quarters touching the inclusive date range ``[first, last]``."""
    if last < first:
        return []
    out, q, end = [], Quarter.of(first), Quarter.of(last)
    while q <= end:
        out.append(q)
        q = q.next()
    return out


@dataclass(frozen=True)
class Windows:
    """Inclusive filing-date windows per class; the gap between cutoffs is dropped."""

    earliest_cutoff: date
    latest_cutoff: date
    memorized: tuple[date, date]
    control: tuple[date, date]

    def label_for(self, filed: date) -> int | None:
        if self.memorized[0] <= filed <= self.memorized[1]:
            return 1
        if self.control[0] <= filed <= self.control[1]:
            return 0
        return None


def compute_windows(cutoffs: Mapping[str, date], start: date, today: date) -> Windows:
    if not cutoffs:
        raise ValueError("cutoffs must name at least one model")
    earliest, latest = min(cutoffs.values()), max(cutoffs.values())
    if start >= earliest:
        raise ValueError(f"start {start} must precede the earliest cutoff {earliest}")
    if latest >= today:
        raise ValueError(f"no control window: latest cutoff {latest} >= today {today}")
    return Windows(
        earliest_cutoff=earliest,
        latest_cutoff=latest,
        memorized=(start, earliest - timedelta(days=1)),
        control=(latest + timedelta(days=1), today),
    )


def select_cutoffs(all_cutoffs: Mapping[str, date], models: Sequence[str]) -> dict[str, date]:
    missing = [m for m in models if m not in all_cutoffs]
    if missing:
        raise ValueError(f"models missing from cutoffs registry: {', '.join(missing)}")
    return {m: all_cutoffs[m] for m in models}


def per_quarter_cap(pool: int, quarters: int) -> int:
    return math.ceil(pool / quarters) + 1 if quarters else 0


def plan(
    cutoffs: Mapping[str, date],
    *,
    start: date,
    today: date,
    targets: Targets,
    seed: int,
    max_requests: int,
) -> dict[str, Any]:
    """Pure description of a build; the dry run prints it. No I/O."""
    w = compute_windows(cutoffs, start, today)
    mem_q = quarters_between(*w.memorized)
    ctl_q = quarters_between(*w.control)
    return {
        "source": SOURCE,
        "seed": seed,
        "today": today.isoformat(),
        "cutoffs": {m: d.isoformat() for m, d in cutoffs.items()},
        "earliest_cutoff": w.earliest_cutoff.isoformat(),
        "latest_cutoff": w.latest_cutoff.isoformat(),
        "memorized": {
            "window": [d.isoformat() for d in w.memorized],
            "quarters": [q.label for q in mem_q],
            "pool": targets.memorized_pool,
            "per_quarter_cap": per_quarter_cap(targets.memorized_pool, len(mem_q)),
        },
        "control": {
            "window": [d.isoformat() for d in w.control],
            "quarters": [q.label for q in ctl_q],
            "pool": targets.control_pool,
            "per_quarter_cap": per_quarter_cap(targets.control_pool, len(ctl_q)),
        },
        "targets": {key: getattr(targets, key) for key in OUTPUT_FILES},
        "max_requests": max_requests,
        "index_requests": len(mem_q) + len(ctl_q),
        "requests_per_candidate": "2-3 (index.json, -index.htm fallback, exhibit)",
    }


# ---------- form.idx parsing ---------------------------------------------------


@dataclass(frozen=True)
class IndexEntry:
    form_type: str
    company: str
    cik: str
    date_filed: date
    file_name: str

    @property
    def accession(self) -> str:
        return self.file_name.rsplit("/", 1)[-1].removesuffix(".txt")

    @property
    def folder_url(self) -> str:
        return f"{SEC_BASE}/Archives/edgar/data/{int(self.cik)}/{self.accession.replace('-', '')}"


def _parse_index_date(raw: str) -> date | None:
    raw = raw.strip()
    try:
        if len(raw) == 8 and raw.isdigit():
            return date(int(raw[:4]), int(raw[4:6]), int(raw[6:]))
        return date.fromisoformat(raw)
    except ValueError:
        return None


def parse_form_index(text: str, form_types: Iterable[str] = ("8-K",)) -> list[IndexEntry]:
    """Parse an EDGAR ``form.idx`` and keep rows whose form type matches exactly."""
    wanted = set(form_types)
    lines = text.splitlines()
    company_col = None
    body_start = None
    for i, line in enumerate(lines):
        if line.startswith("Form Type") and "Company Name" in line:
            company_col = line.index("Company Name")
        elif company_col is not None and line.startswith("---"):
            body_start = i + 1
            break
    if company_col is None or body_start is None:
        raise ValueError("form.idx header not found")
    entries: list[IndexEntry] = []
    for line in lines[body_start:]:
        if not line.strip():
            continue
        form_type = line[:company_col].strip()
        if form_type not in wanted:
            continue
        parts = line[company_col:].rsplit(None, 3)
        if len(parts) != 4:
            logger.warning("skip index row: unparseable line %r", line[:120])
            continue
        company, cik, raw_date, file_name = parts
        filed = _parse_index_date(raw_date)
        if filed is None or not cik.isdigit():
            logger.warning("skip index row: bad CIK/date in %r", line[:120])
            continue
        entries.append(IndexEntry(form_type, company.strip(), cik, filed, file_name.strip()))
    return entries


# ---------- filing index: locate EX-99.1 ----------------------------------------


def exhibit_from_index_json(payload: Mapping[str, Any], folder_url: str) -> str | None:
    """Return the EX-99.1 URL if the directory listing carries document types."""
    items = (payload.get("directory") or {}).get("item") or []
    for item in items:
        doc_type = str(item.get("type") or "")
        name = str(item.get("name") or "")
        if _EX991_TYPE.match(doc_type) and name.lower().endswith(_TEXT_SUFFIXES):
            return f"{folder_url}/{name}"
    return None


def index_json_has_types(payload: Mapping[str, Any]) -> bool:
    items = (payload.get("directory") or {}).get("item") or []
    return any(str(item.get("type") or "").upper().startswith("EX-") for item in items)


class _TableRowParser(HTMLParser):
    """Collect ``(cell texts, hrefs)`` per table row."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[tuple[list[str], list[str]]] = []
        self._cells: list[str] | None = None
        self._hrefs: list[str] = []
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self._cells, self._hrefs = [], []
        elif tag in ("td", "th") and self._cells is not None:
            self._cell = []
        elif tag == "a" and self._cells is not None:
            href = dict(attrs).get("href")
            if href:
                self._hrefs.append(href)

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th") and self._cells is not None and self._cell is not None:
            self._cells.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._cells is not None:
            self.rows.append((self._cells, self._hrefs))
            self._cells = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)


def exhibit_from_index_htm(html: str) -> str | None:
    """Return the EX-99.1 URL from a filing's ``-index.htm`` document table."""
    parser = _TableRowParser()
    parser.feed(html)
    for cells, hrefs in parser.rows:
        if not hrefs or not any(_EX991_TYPE.match(c) for c in cells):
            continue
        href = hrefs[0].removeprefix("/ix?doc=")
        if not href.lower().endswith(_TEXT_SUFFIXES):
            continue
        return href if href.startswith("http") else f"{SEC_BASE}{href}"
    return None


# ---------- HTML to text -------------------------------------------------------


_BLOCK_TAGS = frozenset(
    "p div br tr li h1 h2 h3 h4 h5 h6 table center title blockquote pre ul ol dl dt dd hr".split()
)


class _TextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.table_chars = 0
        self.total_chars = 0
        self._skip = 0
        self._cell_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in ("script", "style", "head"):
            self._skip += 1
        elif tag in ("td", "th"):
            self._cell_depth += 1
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")
        elif tag in ("td", "th"):
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style", "head") and self._skip:
            self._skip -= 1
        elif tag in ("td", "th") and self._cell_depth:
            self._cell_depth -= 1
        if tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        visible = len("".join(data.split()))
        self.total_chars += visible
        if self._cell_depth:
            self.table_chars += visible
        self.parts.append(data)


@dataclass(frozen=True)
class ExhibitText:
    lines: list[str]
    table_fraction: float


def html_to_text(html: str) -> ExhibitText:
    """Strip tags; return non-empty whitespace-normalised lines and table share."""
    parser = _TextParser()
    parser.feed(html)
    parser.close()
    raw = "".join(parser.parts).replace("\xa0", " ")
    lines = [" ".join(line.split()) for line in raw.splitlines()]
    lines = [line for line in lines if line]
    fraction = parser.table_chars / parser.total_chars if parser.total_chars else 0.0
    return ExhibitText(lines=lines, table_fraction=fraction)


def _is_title_line(line: str) -> bool:
    if len(line) < 20 or _BOILERPLATE_LINE.match(line):
        return False
    letters = sum(ch.isalpha() for ch in line)
    return letters >= 0.6 * len(line.replace(" ", "")) and len(line.split()) >= 3


def english_looking(text: str) -> bool:
    tokens = re.findall(r"[A-Za-z]+", text.lower())
    if len(tokens) < 30:
        return False
    non_space = [ch for ch in text if not ch.isspace()]
    ascii_share = sum(ch.isascii() for ch in non_space) / len(non_space)
    stop_share = sum(t in _STOPWORDS for t in tokens) / len(tokens)
    return ascii_share >= 0.95 and stop_share >= MIN_STOPWORD_FRACTION


def build_exhibit_prompt(html: str) -> tuple[str, str] | str:
    """Return ``(title, prompt)`` or a skip reason string."""
    text = html_to_text(html)
    if text.table_fraction > MAX_TABLE_FRACTION:
        return f"mostly tables ({text.table_fraction:.0%} of text in table cells)"
    title_idx = next((i for i, line in enumerate(text.lines) if _is_title_line(line)), None)
    if title_idx is None:
        return "no title line"
    title = text.lines[title_idx]
    body = " ".join(text.lines[title_idx + 1 :])
    if len(body) < MIN_BODY_CHARS:
        return f"too short ({len(body)} body chars < {MIN_BODY_CHARS})"
    prompt = _build_prompt(title, body)
    if not english_looking(prompt):
        return "not English-looking"
    return title, prompt


# ---------- HTTP client --------------------------------------------------------


class RateLimiter:
    """Sleep so consecutive request starts are at least ``1/rate`` seconds apart."""

    def __init__(
        self,
        rate_per_second: float = MAX_REQUESTS_PER_SECOND,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.interval = 1.0 / rate_per_second
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None

    def wait(self) -> None:
        if self._last is not None:
            delay = self._last + self.interval - self._clock()
            if delay > 0:
                self._sleep(delay)
        self._last = self._clock()


def require_user_agent(env: Mapping[str, str] | None = None) -> str:
    """Read ``SEC_USER_AGENT``; fail clearly (without echoing it) when unusable."""
    value = (env if env is not None else os.environ).get(USER_AGENT_ENV, "").strip()
    if not value:
        raise EdgarConfigError(
            f"{USER_AGENT_ENV} is not set. SEC fair access requires a User-Agent like "
            "'Your Name contact@example.com'; export it and pass it to the container with -e."
        )
    if "@" not in value or " " not in value:
        raise EdgarConfigError(
            f"{USER_AGENT_ENV} must look like 'Name contact@domain' (a name and an email address)."
        )
    return value


@dataclass
class EdgarClient:
    """Rate-limited, capped, caching GET client for www.sec.gov."""

    session: Any
    max_requests: int = DEFAULT_MAX_REQUESTS
    cache_dir: Path | None = None
    limiter: RateLimiter = field(default_factory=RateLimiter)
    max_retries: int = 3
    backoff_seconds: float = 2.0
    sleep: Callable[[float], None] = time.sleep
    timeout: float = 60.0
    requests_used: int = 0
    cache_hits: int = 0

    def _cache_path(self, url: str) -> Path | None:
        if self.cache_dir is None:
            return None
        if not url.startswith(SEC_BASE + "/"):
            raise ValueError(f"refusing non-SEC URL: {url}")
        rel = url[len(SEC_BASE) + 1 :]
        parts = [p for p in rel.split("/") if p]
        if any(p in (".", "..") for p in parts):
            raise ValueError(f"refusing URL with dot segments: {url}")
        return self.cache_dir.joinpath(*parts)

    def get(self, url: str) -> bytes:
        cache = self._cache_path(url)
        if cache is not None:
            if cache.is_file():
                self.cache_hits += 1
                return cache.read_bytes()
            missing = cache.with_name(cache.name + ".404")
            if missing.is_file():
                self.cache_hits += 1
                raise EdgarHTTPError(url, 404)
        body = self._fetch(url)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache.with_name(cache.name + ".part")
            tmp.write_bytes(body)
            tmp.replace(cache)
        return body

    def _fetch(self, url: str) -> bytes:
        for attempt in range(self.max_retries + 1):
            if self.requests_used >= self.max_requests:
                raise RequestCapReached(f"request cap of {self.max_requests} reached")
            self.limiter.wait()
            self.requests_used += 1
            response = self.session.get(url, timeout=self.timeout)
            status = response.status_code
            if status == 200:
                return response.content
            if status in (429, 503) and attempt < self.max_retries:
                delay = self._retry_delay(response, attempt)
                logger.warning("HTTP %s for %s; retry %d in %.1fs", status, url, attempt + 1, delay)
                self.sleep(delay)
                continue
            if status == 404:
                cache = self._cache_path(url)
                if cache is not None:
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    cache.with_name(cache.name + ".404").write_bytes(b"")
            raise EdgarHTTPError(url, status)
        raise AssertionError("unreachable")  # pragma: no cover

    def _retry_delay(self, response: Any, attempt: int) -> float:
        retry_after = str((getattr(response, "headers", None) or {}).get("Retry-After", "")).strip()
        if retry_after.isdigit():
            return min(float(retry_after), 60.0)
        return min(self.backoff_seconds * (2**attempt), 60.0)


def make_session(user_agent: str) -> Any:
    import requests

    session = requests.Session()
    session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
    return session


def default_cache_dir() -> Path:
    """``/outputs/edgar-cache`` in the dev container (runs/container mount), else local."""
    outputs = Path("/outputs")
    return outputs / "edgar-cache" if outputs.is_dir() else Path("runs/container/edgar-cache")


# ---------- building records ---------------------------------------------------


@dataclass(frozen=True)
class EdgarRecord:
    prompt: str
    label: int
    published_at: date
    url: str
    cik: str
    company: str
    accession: str
    title_hash: str
    quarter: str

    def to_row(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "label": self.label,
            "metadata": {
                "published_at": self.published_at.isoformat(),
                "source": SOURCE,
                "url": self.url,
                "cik": self.cik,
                "company": self.company,
                "accession": self.accession,
            },
        }


def _decode(body: bytes) -> str:
    return body.decode("utf-8", errors="replace")


def locate_exhibit(client: EdgarClient, entry: IndexEntry) -> str | None:
    """index.json first; ``-index.htm`` only when the listing lacks document types."""
    payload = json.loads(_decode(client.get(f"{entry.folder_url}/index.json")))
    if index_json_has_types(payload):
        return exhibit_from_index_json(payload, entry.folder_url)
    html = _decode(client.get(f"{entry.folder_url}/{entry.accession}-index.htm"))
    return exhibit_from_index_htm(html)


@dataclass
class _Seen:
    urls: set[str] = field(default_factory=set)
    title_hashes: set[str] = field(default_factory=set)
    company_dates: set[tuple[str, date]] = field(default_factory=set)


def resolve_candidate(
    client: EdgarClient, entry: IndexEntry, label: int, quarter: Quarter, seen: _Seen
) -> EdgarRecord | None:
    """Turn one index row into a record, or warn and return None."""
    key = (entry.cik, entry.date_filed)
    if key in seen.company_dates:
        logger.warning("skip %s: duplicate company+date (CIK %s, %s)", entry.accession, entry.cik, entry.date_filed)
        return None
    try:
        url = locate_exhibit(client, entry)
        if url is None:
            logger.warning("skip %s: no EX-99.1 text exhibit", entry.accession)
            return None
        if url in seen.urls:
            logger.warning("skip %s: duplicate URL %s", entry.accession, url)
            return None
        built = build_exhibit_prompt(_decode(client.get(url)))
    except EdgarHTTPError as exc:
        logger.warning("skip %s: %s", entry.accession, exc)
        return None
    except (ValueError, json.JSONDecodeError) as exc:
        logger.warning("skip %s: unparseable filing index (%s)", entry.accession, exc)
        return None
    if isinstance(built, str):
        logger.warning("skip %s: %s (%s)", entry.accession, built, url)
        return None
    title, prompt = built
    title_hash = _title_hash(title)
    if title_hash in seen.title_hashes:
        logger.warning("skip %s: duplicate title %r", entry.accession, title[:80])
        return None
    seen.urls.add(url)
    seen.title_hashes.add(title_hash)
    seen.company_dates.add(key)
    return EdgarRecord(
        prompt=prompt,
        label=label,
        published_at=entry.date_filed,
        url=url,
        cik=entry.cik,
        company=entry.company,
        accession=entry.accession,
        title_hash=title_hash,
        quarter=quarter.label,
    )


def _quarter_candidates(
    client: EdgarClient, quarter: Quarter, window: tuple[date, date], seed: int, label: int
) -> list[IndexEntry]:
    entries = parse_form_index(client.get(quarter.index_url).decode("latin-1"))
    eligible = [e for e in entries if window[0] <= e.date_filed <= window[1]]
    eligible.sort(key=lambda e: (e.date_filed, e.cik, e.accession))
    random.Random(f"{seed}:{label}:{quarter.label}").shuffle(eligible)
    return eligible


def sample_pool(
    client: EdgarClient,
    *,
    label: int,
    window: tuple[date, date],
    pool_size: int,
    seed: int,
    seen: _Seen,
) -> tuple[list[EdgarRecord], bool]:
    """Fill one class pool round-robin across quarters. Returns (records, cap_hit)."""
    quarters = quarters_between(*window)
    cap = per_quarter_cap(pool_size, len(quarters))
    candidates: dict[Quarter, list[IndexEntry] | None] = dict.fromkeys(quarters)
    cursor: Counter[Quarter] = Counter()
    taken: Counter[Quarter] = Counter()
    exhausted: set[Quarter] = set()
    records: list[EdgarRecord] = []
    rng = random.Random(f"{seed}:{label}:rounds")
    try:
        while len(records) < pool_size:
            progress = False
            order = list(quarters)
            rng.shuffle(order)
            for q in order:
                if len(records) >= pool_size:
                    break
                if q in exhausted or taken[q] >= cap:
                    continue
                if candidates[q] is None:
                    try:
                        candidates[q] = _quarter_candidates(client, q, window, seed, label)
                    except (EdgarHTTPError, ValueError) as exc:
                        logger.warning("skip quarter %s: %s", q.label, exc)
                        candidates[q] = []
                record = _pick_from_quarter(client, q, candidates[q] or [], cursor, label, seen)
                if record is None:
                    exhausted.add(q)
                    continue
                records.append(record)
                taken[q] += 1
                progress = True
            if not progress:
                break
    except RequestCapReached as exc:
        logger.error("%s; class label=%d stops at %d/%d rows", exc, label, len(records), pool_size)
        return records, True
    return records, False


def _pick_from_quarter(
    client: EdgarClient,
    quarter: Quarter,
    candidates: list[IndexEntry],
    cursor: Counter[Quarter],
    label: int,
    seen: _Seen,
) -> EdgarRecord | None:
    for _ in range(MAX_ATTEMPTS_PER_PICK):
        if cursor[quarter] >= len(candidates):
            return None
        entry = candidates[cursor[quarter]]
        cursor[quarter] += 1
        record = resolve_candidate(client, entry, label, quarter, seen)
        if record is not None:
            return record
    logger.warning("quarter %s: %d candidates in a row skipped; marking exhausted", quarter.label, MAX_ATTEMPTS_PER_PICK)
    return None


def split_pool(
    records: Sequence[EdgarRecord], n_calibration: int, n_held_out: int, seed: int, label: int
) -> tuple[list[EdgarRecord], list[EdgarRecord]]:
    """Seeded split; calibration is filled first when the pool is short."""
    shuffled = sorted(records, key=lambda r: (r.published_at, r.url))
    random.Random(f"{seed}:{label}:split").shuffle(shuffled)
    calibration = shuffled[:n_calibration]
    held_out = shuffled[n_calibration : n_calibration + n_held_out]
    order = lambda r: (r.published_at, r.url)  # noqa: E731
    return sorted(calibration, key=order), sorted(held_out, key=order)


# ---------- output -------------------------------------------------------------


def _write_jsonl(path: Path, records: Sequence[EdgarRecord]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(r.to_row(), ensure_ascii=False) + "\n" for r in records).encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class BuildResult:
    manifest: dict[str, Any]
    paths: dict[str, Path]
    complete: bool


def build_edgar_corpora(
    out_dir: Path | str,
    cutoffs: Mapping[str, date],
    client: EdgarClient,
    *,
    seed: int = 0,
    start: date = DEFAULT_START,
    today: date | None = None,
    targets: Targets | None = None,
) -> BuildResult:
    """Build the four disjoint EDGAR corpora plus ``manifest.json``."""
    targets = targets or Targets()
    today = today or date.today()
    windows = compute_windows(cutoffs, start, today)
    out_dir = Path(out_dir)
    seen = _Seen()

    memorized, cap_mem = sample_pool(
        client, label=1, window=windows.memorized, pool_size=targets.memorized_pool, seed=seed, seen=seen
    )
    control: list[EdgarRecord] = []
    cap_ctl = cap_mem
    if not cap_mem:
        control, cap_ctl = sample_pool(
            client, label=0, window=windows.control, pool_size=targets.control_pool, seed=seed, seen=seen
        )

    cal_m, held_m = split_pool(memorized, targets.calibration_memorized, targets.held_out_memorized, seed, 1)
    cal_c, held_c = split_pool(control, targets.calibration_control, targets.held_out_control, seed, 0)
    files = {
        "calibration_memorized": cal_m,
        "calibration_control": cal_c,
        "held_out_memorized": held_m,
        "held_out_control": held_c,
    }
    paths: dict[str, Path] = {}
    file_entries: dict[str, Any] = {}
    complete = True
    for key, records in files.items():
        rel, label = OUTPUT_FILES[key]
        path = out_dir / rel
        digest = _write_jsonl(path, records)
        target = getattr(targets, key)
        if len(records) < target:
            complete = False
            logger.warning("%s: %d rows, short of target %d", rel, len(records), target)
        paths[key] = path
        file_entries[rel] = {
            "label": label,
            "rows": len(records),
            "target": target,
            "sha256": digest,
            "per_quarter": dict(sorted(Counter(r.quarter for r in records).items())),
            "published_range": [
                min(r.published_at for r in records).isoformat() if records else None,
                max(r.published_at for r in records).isoformat() if records else None,
            ],
        }

    manifest = plan(cutoffs, start=start, today=today, targets=targets, seed=seed, max_requests=client.max_requests)
    manifest.update(
        {
            "retrieval_date": today.isoformat(),
            "start": start.isoformat(),
            "published_at": "8-K filing date from the EDGAR quarterly full index",
            "dedupe": ["exhibit url", "sha256(title)", "cik + filing date"],
            "files": file_entries,
            "requests_used": client.requests_used,
            "cache_hits": client.cache_hits,
            "request_cap_hit": cap_mem or cap_ctl,
            "complete": complete,
        }
    )
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths["manifest"] = manifest_path
    return BuildResult(manifest=manifest, paths=paths, complete=complete)
