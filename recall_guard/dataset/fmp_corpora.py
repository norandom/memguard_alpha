"""FMP-backed calibration corpus builder for honest-model-ranking.

Implements Requirement 11 of the spec: build the IS/OOS calibration corpora
(`data/calibration/is_memorized.jsonl` label=1 and
`data/calibration/oos_control.jsonl` label=0) from real, dated articles
fetched via the Financial Modeling Prep (FMP) news endpoints, plus an
``update_oos`` mode that incrementally appends new post-cutoff articles to
the OOS corpus only.

The module exposes three public symbols (re-exported from
``recall_guard.dataset.__init__``):

- ``ArticleRecord`` -- frozen dataclass describing one calibration row.
- ``build_calibration`` -- one-shot builder driven by the cutoff registry.
- ``update_oos`` -- incremental refresh of the OOS corpus, never IS.

It also exposes a small ``fetch_articles`` helper for direct FMP pagination
which the build/update routines compose internally and which is the only
location that performs HTTP I/O (``requests.get``). Tests mock that call.

CLI surface (``python -m recall_guard.dataset.fmp_corpora --help``):

    python -m recall_guard.dataset.fmp_corpora build [--cutoffs PATH] [--out DIR]
                                            [--target N] [--include-stock-news]
    python -m recall_guard.dataset.fmp_corpora update [--out DIR] [--since YYYY-MM-DD]

Both subcommands read ``FMP_API_KEY`` from the environment (with
``python-dotenv`` loading ``.env`` when present).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

logger = logging.getLogger(__name__)


# Default endpoints per Open Defaults (Req 11):
# - "fmp-articles": broad financial commentary
# - "news/general-latest": broader market news
# - "news/stock-latest" is opt-in via --include-stock-news, NOT a default.
DEFAULT_ENDPOINTS: tuple[str, ...] = ("fmp-articles", "news/general-latest")

# Long pre-cutoff window. Anything earlier than this is unlikely to be
# present in financial news archives anyway and is bounded purely so we
# can pass a concrete `from` date to FMP.
_EPOCH = date(2010, 1, 1)

# Per-article body truncation per design ("title + body excerpt, capped at
# ≈1500 chars").
_PROMPT_MAX_CHARS = 1500

# Pagination safety cap (prevents accidental infinite loops if FMP returns
# duplicate-but-non-empty pages).
_MAX_PAGES_PER_ENDPOINT = 50

# Default page size; FMP supports up to 100 per page on most news endpoints.
_PAGE_LIMIT = 100

# Body field probe order. FMP varies by endpoint (per task spec).
_BODY_FIELDS: tuple[str, ...] = ("text", "content", "body")

# Date field probe order. ``news/general-latest`` and ``news/stock-latest``
# expose ``publishedDate``; ``fmp-articles`` exposes ``date``.
_DATE_FIELDS: tuple[str, ...] = ("publishedDate", "date")

# URL field probe order. ``news/*`` endpoints expose ``url``;
# ``fmp-articles`` exposes ``link``.
_URL_FIELDS: tuple[str, ...] = ("url", "link")


# ---------- public dataclass -------------------------------------------------


@dataclass(frozen=True)
class ArticleRecord:
    """One calibration article in canonical in-memory form.

    ``prompt`` is the concatenated title + body excerpt, capped at
    ``_PROMPT_MAX_CHARS`` characters with surrounding whitespace trimmed.
    ``label`` is 1 for IS-memorized rows (pre-earliest-cutoff) and 0 for
    OOS rows (post-latest-cutoff).
    """

    prompt: str
    label: int
    published_at: date
    source: str
    url: str


# ---------- HTTP layer -------------------------------------------------------


def fetch_articles(
    endpoint: str,
    api_key: str,
    from_date: date,
    to_date: date,
    page: int,
    limit: int,
) -> list[dict]:
    """Fetch one page of articles from an FMP news endpoint.

    Builds the canonical FMP ``stable/`` URL with ``from``, ``to``,
    ``page``, ``limit``, and ``apikey`` query parameters. Raises
    ``RuntimeError`` (with status code + endpoint) on any non-200 response.

    Returns the parsed JSON list (assumed to be a list of article dicts).
    """
    url = (
        "https://financialmodelingprep.com/stable/"
        f"{endpoint}?from={from_date.isoformat()}&to={to_date.isoformat()}"
        f"&page={page}&limit={limit}&apikey={api_key}"
    )
    response = requests.get(url, timeout=15)
    if response.status_code != 200:
        raise RuntimeError(
            f"FMP endpoint {endpoint!r} returned HTTP {response.status_code}"
        )
    payload = response.json()
    if not isinstance(payload, list):
        # Defensive: design assumes a top-level list.
        return []
    return payload


# ---------- internal helpers -------------------------------------------------


def _resolve_api_key(api_key: str | None) -> str:
    if api_key:
        return api_key
    env_key = os.environ.get("FMP_API_KEY")
    if not env_key:
        raise RuntimeError(
            "FMP_API_KEY is not set; pass api_key= or export FMP_API_KEY."
        )
    return env_key


def _parse_published(raw: object) -> date | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    candidate = raw.strip()
    # FMP date conventions: "YYYY-MM-DD HH:MM:SS" is the common form, but
    # a small subset of endpoints return bare "YYYY-MM-DD".
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(candidate, fmt).date()
        except ValueError:
            continue
    return None


def _extract_body(article: dict) -> str:
    for field in _BODY_FIELDS:
        value = article.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _build_prompt(title: str, body: str) -> str:
    title = (title or "").strip()
    body = (body or "").strip()
    if title and body:
        prompt = f"{title}\n\n{body}"
    else:
        prompt = title or body
    if len(prompt) > _PROMPT_MAX_CHARS:
        prompt = prompt[:_PROMPT_MAX_CHARS]
    return prompt.strip()


def _title_hash(title: str) -> str:
    return hashlib.sha256(title.strip().lower().encode("utf-8")).hexdigest()


def _record_to_jsonl(record: ArticleRecord) -> dict:
    return {
        "prompt": record.prompt,
        "label": record.label,
        "metadata": {
            "published_at": record.published_at.isoformat(),
            "source": record.source,
            "url": record.url,
        },
    }


def _normalise_article(
    article: dict,
    *,
    source: str,
) -> ArticleRecord | None:
    """Convert a raw FMP article dict to a partial ArticleRecord.

    Returns None on missing/unparseable date or empty body, after emitting
    a single WARNING per skip (Req 11.4). The ``label`` field is filled in
    by the caller once date-bucketing has happened; we set 0 here as a
    safe placeholder that the caller MUST overwrite.
    """
    url = ""
    for field in _URL_FIELDS:
        v = article.get(field)
        if isinstance(v, str) and v.strip():
            url = v.strip()
            break
    title = str(article.get("title") or "").strip()

    raw_date: object = None
    for field in _DATE_FIELDS:
        v = article.get(field)
        if v not in (None, ""):
            raw_date = v
            break
    published = _parse_published(raw_date)
    if published is None:
        logger.warning(
            "skip article: unparseable publishedDate %r (url=%s)",
            raw_date,
            url or "<missing>",
        )
        return None

    body = _extract_body(article)
    if not body:
        logger.warning(
            "skip article: empty body (url=%s, published=%s)",
            url or "<missing>",
            published.isoformat(),
        )
        return None

    return ArticleRecord(
        prompt=_build_prompt(title, body),
        label=0,  # placeholder; caller overrides via dataclasses.replace
        published_at=published,
        source=source,
        url=url,
    )


def _write_jsonl(path: Path, records: list[ArticleRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(_record_to_jsonl(record), ensure_ascii=False) + "\n")


def _append_jsonl(path: Path, records: list[ArticleRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(_record_to_jsonl(record), ensure_ascii=False) + "\n")


def _read_existing_oos(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


# ---------- build_calibration ------------------------------------------------


def build_calibration(
    out_dir: Path | str,
    cutoffs: dict[str, date],
    target_per_corpus: int = 100,
    api_key: str | None = None,
    endpoints: Sequence[str] = DEFAULT_ENDPOINTS,
    today: date | None = None,
    is_strata: int = 5,
) -> tuple[Path, Path]:
    """Build both calibration corpora from FMP news endpoints.

    Filters strictly by publication date (Req 11.2):
      - IS rows: published BEFORE ``min(cutoffs.values())``; sampled across
        ``is_strata`` equal-width chronological sub-windows of
        ``(_EPOCH, earliest_cutoff)`` so the corpus does not cluster on the
        cutoff edge (task 1.5). Per-bucket target is
        ``target_per_corpus // is_strata``; the LAST bucket absorbs any
        remainder so the totals always sum to ``target_per_corpus``.
      - OOS rows: published AFTER ``max(cutoffs.values())`` and on/before
        today. OOS clustering at "now" is acceptable: recent articles are
        uniformly unseen by every in-registry model, so the OOS sampler
        keeps a single full window (no stratification).
      - articles in the gap between earliest and latest cutoff are dropped

    Deduplicates by URL exact-match and by sha256(title) (Req 11.3) across
    ALL sub-windows and ALL endpoints -- a single set per side persists for
    the whole run, so an article from sub-window 2 cannot reappear under a
    different bucket in sub-window 3.

    Skips articles missing a body or a parseable ``publishedDate`` and emits
    one WARNING per skip (Req 11.4).

    Writes ``out_dir/is_memorized.jsonl`` and ``out_dir/oos_control.jsonl``
    as JSONL, one row per line, with the schema::

        {"prompt": str, "label": int,
         "metadata": {"published_at": "YYYY-MM-DD",
                      "source": str, "url": str}}

    Returns ``(is_path, oos_path)``.
    """
    if not cutoffs:
        raise ValueError("cutoffs must be a non-empty mapping of model_id -> date.")
    if is_strata < 1:
        raise ValueError(f"is_strata must be >= 1, got {is_strata}.")

    api_key = _resolve_api_key(api_key)
    today = today or date.today()

    earliest_cutoff = min(cutoffs.values())
    latest_cutoff = max(cutoffs.values())
    if latest_cutoff >= today:
        raise ValueError(
            f"No OOS window available: latest cutoff {latest_cutoff.isoformat()} "
            f">= today {today.isoformat()}."
        )

    out_dir = Path(out_dir)
    is_path = out_dir / "is_memorized.jsonl"
    oos_path = out_dir / "oos_control.jsonl"

    is_records: list[ArticleRecord] = []
    oos_records: list[ArticleRecord] = []
    seen_urls: set[str] = set()
    seen_title_hashes: set[str] = set()

    _collect_is_records(
        endpoints=endpoints, api_key=api_key, today=today,
        earliest_cutoff=earliest_cutoff, latest_cutoff=latest_cutoff,
        is_records=is_records, oos_records=oos_records,
        seen_urls=seen_urls, seen_title_hashes=seen_title_hashes,
        target_per_corpus=target_per_corpus, is_strata=is_strata,
    )
    _collect_oos_records(
        endpoints=endpoints, api_key=api_key, today=today,
        earliest_cutoff=earliest_cutoff, latest_cutoff=latest_cutoff,
        is_records=is_records, oos_records=oos_records,
        seen_urls=seen_urls, seen_title_hashes=seen_title_hashes,
        target_per_corpus=target_per_corpus,
    )
    _warn_shortfall(is_records, oos_records, target_per_corpus, endpoints, is_strata)

    _write_jsonl(is_path, is_records)
    _write_jsonl(oos_path, oos_records)
    return is_path, oos_path


def _collect_is_records(
    *,
    endpoints: Sequence[str],
    api_key: str,
    today: date,
    earliest_cutoff: date,
    latest_cutoff: date,
    is_records: list[ArticleRecord],
    oos_records: list[ArticleRecord],
    seen_urls: set[str],
    seen_title_hashes: set[str],
    target_per_corpus: int,
    is_strata: int,
) -> None:
    """IS sampling stratified across K equal-width sub-windows."""
    is_buckets = _split_is_window(_EPOCH, earliest_cutoff, is_strata)
    base_per_bucket = target_per_corpus // is_strata
    remainder = target_per_corpus - base_per_bucket * is_strata
    last_idx = is_strata - 1

    for bucket_idx, (bucket_from, bucket_to) in enumerate(is_buckets):
        bucket_target = base_per_bucket + (remainder if bucket_idx == last_idx else 0)
        if bucket_target <= 0:
            continue
        bucket_start_count = len(is_records)
        for endpoint in endpoints:
            if len(is_records) - bucket_start_count >= bucket_target:
                break
            _paginate_window(
                endpoint=endpoint, api_key=api_key,
                window_from=bucket_from, window_to=bucket_to, today=today,
                earliest_cutoff=earliest_cutoff, latest_cutoff=latest_cutoff,
                is_records=is_records, oos_records=oos_records,
                seen_urls=seen_urls, seen_title_hashes=seen_title_hashes,
                is_target=bucket_start_count + bucket_target,
                oos_target=0,
            )
        added = len(is_records) - bucket_start_count
        if added < bucket_target:
            logger.warning(
                "IS sub-window bucket %d (%s -> %s) came up short: %d / %d rows.",
                bucket_idx, bucket_from.isoformat(), bucket_to.isoformat(),
                added, bucket_target,
            )


def _collect_oos_records(
    *,
    endpoints: Sequence[str],
    api_key: str,
    today: date,
    earliest_cutoff: date,
    latest_cutoff: date,
    is_records: list[ArticleRecord],
    oos_records: list[ArticleRecord],
    seen_urls: set[str],
    seen_title_hashes: set[str],
    target_per_corpus: int,
) -> None:
    """OOS sampling: single full window. Clustering at 'now' is acceptable."""
    oos_window_from = latest_cutoff + timedelta(days=1)
    if oos_window_from > today:
        return
    for endpoint in endpoints:
        if len(oos_records) >= target_per_corpus:
            break
        _paginate_window(
            endpoint=endpoint, api_key=api_key,
            window_from=oos_window_from, window_to=today, today=today,
            earliest_cutoff=earliest_cutoff, latest_cutoff=latest_cutoff,
            is_records=is_records, oos_records=oos_records,
            seen_urls=seen_urls, seen_title_hashes=seen_title_hashes,
            is_target=0,
            oos_target=target_per_corpus,
        )


def _warn_shortfall(
    is_records: list[ArticleRecord],
    oos_records: list[ArticleRecord],
    target_per_corpus: int,
    endpoints: Sequence[str],
    is_strata: int,
) -> None:
    if len(is_records) < target_per_corpus:
        logger.warning(
            "IS corpus came up short of target: %d of %d rows from endpoints %s "
            "across %d sub-windows.",
            len(is_records), target_per_corpus, list(endpoints), is_strata,
        )
    if len(oos_records) < target_per_corpus:
        logger.warning(
            "OOS corpus came up short: %d / %d rows from endpoints %s.",
            len(oos_records), target_per_corpus, list(endpoints),
        )


def _split_is_window(
    epoch: date, earliest_cutoff: date, k: int
) -> list[tuple[date, date]]:
    """Split ``[epoch, earliest_cutoff]`` into ``k`` equal-width sub-windows.

    The LAST bucket's upper bound is ``earliest_cutoff`` exactly so we do
    not lose cutoff-edge articles; intermediate boundaries are computed
    from ``(earliest_cutoff - epoch) / k`` and the next bucket starts the
    day after the previous bucket's upper bound to keep the windows
    non-overlapping.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}.")
    total_days = (earliest_cutoff - epoch).days
    if total_days <= 0:
        return [(epoch, earliest_cutoff)] if k >= 1 else []
    width = total_days // k
    buckets: list[tuple[date, date]] = []
    cursor = epoch
    for i in range(k):
        if i == k - 1:
            upper = earliest_cutoff
        else:
            upper = cursor + timedelta(days=width)
            if upper >= earliest_cutoff:
                upper = earliest_cutoff
        buckets.append((cursor, upper))
        cursor = upper + timedelta(days=1)
        if cursor > earliest_cutoff:
            # Out of range: subsequent buckets are degenerate; stop early
            # but keep `k` slots by collapsing to the cutoff itself.
            for _ in range(i + 1, k - 1):
                buckets.append((earliest_cutoff, earliest_cutoff))
            if i < k - 1:
                buckets.append((earliest_cutoff, earliest_cutoff))
            return buckets[:k]
    return buckets


def _paginate_window(
    *,
    endpoint: str,
    api_key: str,
    window_from: date,
    window_to: date,
    today: date,
    earliest_cutoff: date,
    latest_cutoff: date,
    is_records: list[ArticleRecord],
    oos_records: list[ArticleRecord],
    seen_urls: set[str],
    seen_title_hashes: set[str],
    is_target: int,
    oos_target: int,
) -> None:
    """Paginate one ``(endpoint, window)`` pass and ingest into the buckets."""
    if window_from > window_to:
        return
    page = 0
    while (
        len(is_records) < is_target
        or len(oos_records) < oos_target
    ) and page < _MAX_PAGES_PER_ENDPOINT:
        articles = fetch_articles(
            endpoint=endpoint,
            api_key=api_key,
            from_date=window_from,
            to_date=window_to,
            page=page,
            limit=_PAGE_LIMIT,
        )
        if not articles:
            break
        added_this_page = _ingest_page(
            articles=articles,
            source=endpoint,
            earliest_cutoff=earliest_cutoff,
            latest_cutoff=latest_cutoff,
            today=today,
            is_records=is_records,
            oos_records=oos_records,
            seen_urls=seen_urls,
            seen_title_hashes=seen_title_hashes,
            is_target=is_target,
            oos_target=oos_target,
        )
        page += 1
        if added_this_page == 0 and len(articles) < _PAGE_LIMIT:
            break


def _ingest_page(
    *,
    articles: list[dict],
    source: str,
    earliest_cutoff: date,
    latest_cutoff: date,
    today: date,
    is_records: list[ArticleRecord],
    oos_records: list[ArticleRecord],
    seen_urls: set[str],
    seen_title_hashes: set[str],
    is_target: int,
    oos_target: int,
) -> int:
    """Bucket one page of raw articles into IS/OOS lists with dedup.

    Returns the number of records added across both buckets (used by the
    caller to detect 'no progress' pages).
    """
    added = 0
    for article in articles:
        partial = _normalise_article(article, source=source)
        if partial is None:
            continue

        url = partial.url
        title_hash = _title_hash(_extract_title(article))

        if url and url in seen_urls:
            continue
        if title_hash in seen_title_hashes:
            continue

        published = partial.published_at
        # Include the cutoff date itself in IS: the registry records the
        # LATER day of the documented cutoff month (e.g. 2023-12-31 for
        # "December 2023"), and articles published on that date are still
        # potentially memorisable per the registry's sourcing comment.
        if published <= earliest_cutoff and len(is_records) < is_target:
            record = _with_label(partial, 1)
            is_records.append(record)
            if url:
                seen_urls.add(url)
            seen_title_hashes.add(title_hash)
            added += 1
        elif (
            published > latest_cutoff
            and published <= today
            and len(oos_records) < oos_target
        ):
            record = _with_label(partial, 0)
            oos_records.append(record)
            if url:
                seen_urls.add(url)
            seen_title_hashes.add(title_hash)
            added += 1
        # Anything between earliest and latest cutoff (inclusive) is dropped.
    return added


def _with_label(record: ArticleRecord, label: int) -> ArticleRecord:
    return ArticleRecord(
        prompt=record.prompt,
        label=label,
        published_at=record.published_at,
        source=record.source,
        url=record.url,
    )


def _extract_title(article: dict) -> str:
    return str(article.get("title") or "").strip()


# ---------- update_oos -------------------------------------------------------


def _parse_meta_date(raw: object) -> date | None:
    """Parse a ``metadata.published_at`` value (may be ISO string or other)."""
    if not isinstance(raw, str):
        return None
    parsed = _parse_published(raw)
    if parsed is not None:
        return parsed
    try:
        return date.fromisoformat(raw)
    except ValueError:
        return None


def _index_existing_oos(rows: list[dict]) -> tuple[set[str], set[str], date | None]:
    """Build ``(urls, title_hashes, max_published)`` from the OOS file rows."""
    urls: set[str] = set()
    title_hashes: set[str] = set()
    max_published: date | None = None
    for row in rows:
        meta = row.get("metadata") or {}
        url = str(meta.get("url") or "")
        if url:
            urls.add(url)
        # Older rows do not retain the title separately; approximate
        # title-hash dedup via the prompt's first line.
        prompt = str(row.get("prompt") or "")
        first_line = prompt.split("\n", 1)[0].strip()
        if first_line:
            title_hashes.add(_title_hash(first_line))
        parsed = _parse_meta_date(meta.get("published_at"))
        if parsed is not None and (max_published is None or parsed > max_published):
            max_published = parsed
    return urls, title_hashes, max_published


def _fetch_new_oos_records(
    *,
    endpoints: Sequence[str],
    api_key: str,
    from_date: date,
    today: date,
    since_date: date,
    existing_urls: set[str],
    existing_title_hashes: set[str],
) -> list[ArticleRecord]:
    """Paginate the FMP endpoints and return new (deduped, label=0) records."""
    new_records: list[ArticleRecord] = []
    for endpoint in endpoints:
        for page in range(_MAX_PAGES_PER_ENDPOINT):
            articles = fetch_articles(
                endpoint=endpoint,
                api_key=api_key,
                from_date=from_date,
                to_date=today,
                page=page,
                limit=_PAGE_LIMIT,
            )
            if not articles:
                break
            page_added = 0
            for article in articles:
                record = _try_make_oos_record(
                    article=article,
                    source=endpoint,
                    since_date=since_date,
                    today=today,
                    existing_urls=existing_urls,
                    existing_title_hashes=existing_title_hashes,
                )
                if record is None:
                    continue
                new_records.append(record)
                page_added += 1
            if page_added == 0 and len(articles) < _PAGE_LIMIT:
                break
    return new_records


def _try_make_oos_record(
    *,
    article: dict,
    source: str,
    since_date: date,
    today: date,
    existing_urls: set[str],
    existing_title_hashes: set[str],
) -> ArticleRecord | None:
    """Normalise one article into an OOS record, or ``None`` to skip.

    Updates the dedup sets in place when the record is accepted.
    """
    partial = _normalise_article(article, source=source)
    if partial is None:
        return None
    url = partial.url
    if url and url in existing_urls:
        return None
    title_hash = _title_hash(_extract_title(article))
    if title_hash in existing_title_hashes:
        return None
    if partial.published_at <= since_date or partial.published_at > today:
        return None
    if url:
        existing_urls.add(url)
    existing_title_hashes.add(title_hash)
    return _with_label(partial, 0)


def update_oos(
    out_dir: Path | str,
    api_key: str | None = None,
    endpoints: Sequence[str] = DEFAULT_ENDPOINTS,
    today: date | None = None,
    since: date | None = None,
) -> Path:
    """Append new post-cutoff articles to the OOS corpus only (Req 11.5).

    Reads the existing ``out_dir/oos_control.jsonl``, derives the latest
    ``published_at`` (or uses ``since`` when supplied), fetches articles
    after that date from each endpoint, dedups against the existing rows
    (URL + title hash), and appends the new rows in place. Never modifies
    ``is_memorized.jsonl``.

    Raises ``FileNotFoundError`` if the OOS file does not exist.
    """
    out_dir = Path(out_dir)
    oos_path = out_dir / "oos_control.jsonl"
    if not oos_path.exists():
        raise FileNotFoundError(
            f"OOS corpus not found at {oos_path}; run build_calibration first."
        )

    api_key = _resolve_api_key(api_key)
    today = today or date.today()

    existing_rows = _read_existing_oos(oos_path)
    if not existing_rows and since is None:
        raise ValueError(
            f"OOS corpus at {oos_path} is empty; cannot derive since-date. "
            "Pass since=YYYY-MM-DD or rebuild via build_calibration."
        )

    existing_urls, existing_title_hashes, max_published = _index_existing_oos(existing_rows)
    since_date = since if since is not None else max_published
    assert since_date is not None  # established by the empty-file guard above

    from_date = since_date + timedelta(days=1)
    if from_date > today:
        return oos_path  # nothing to do

    new_records = _fetch_new_oos_records(
        endpoints=endpoints,
        api_key=api_key,
        from_date=from_date,
        today=today,
        since_date=since_date,
        existing_urls=existing_urls,
        existing_title_hashes=existing_title_hashes,
    )
    if new_records:
        _append_jsonl(oos_path, new_records)
    return oos_path


# ---------- CLI --------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m recall_guard.dataset.fmp_corpora",
        description=(
            "FMP-backed calibration corpus builder for the honest-model-ranking "
            "harness. Produces is_memorized.jsonl + oos_control.jsonl from "
            "real, dated FMP news articles, with strict pre/post-cutoff date "
            "filtering and URL/title-hash deduplication."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser(
        "build",
        help="Build both calibration corpora from FMP news endpoints (one-shot).",
    )
    p_build.add_argument(
        "--cutoffs",
        type=Path,
        default=Path("data/cutoffs.yaml"),
        help="Path to cutoffs.yaml registry (default: data/cutoffs.yaml).",
    )
    p_build.add_argument(
        "--out",
        type=Path,
        default=Path("data/calibration"),
        help="Output directory for both JSONL corpora (default: data/calibration/).",
    )
    p_build.add_argument(
        "--target",
        type=int,
        default=100,
        help="Target row count per corpus (default: 100).",
    )
    p_build.add_argument(
        "--include-stock-news",
        action="store_true",
        help="Add the news/stock-latest endpoint to the default list.",
    )
    p_build.add_argument(
        "--is-strata",
        type=int,
        default=5,
        help=(
            "Number of equal-width chronological sub-windows to split the IS "
            "window into for stratified sampling (default: 5). Per-bucket "
            "target is target_per_corpus // is_strata; the last bucket "
            "absorbs any remainder. OOS sampling is single-window."
        ),
    )

    p_update = sub.add_parser(
        "update",
        help="Append new post-cutoff articles to oos_control.jsonl only.",
    )
    p_update.add_argument(
        "--out",
        type=Path,
        default=Path("data/calibration"),
        help="Directory containing oos_control.jsonl (default: data/calibration/).",
    )
    p_update.add_argument(
        "--since",
        type=str,
        default=None,
        help="Override the auto-derived since-date (ISO YYYY-MM-DD).",
    )

    return parser


def _load_dotenv_quiet() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        # python-dotenv is declared in pyproject; absence is non-fatal here.
        return


def _cli_build(args: argparse.Namespace) -> int:
    from recall_guard.core.loader import load_cutoffs

    endpoints: tuple[str, ...] = DEFAULT_ENDPOINTS
    if args.include_stock_news:
        endpoints = DEFAULT_ENDPOINTS + ("news/stock-latest",)

    cutoffs = load_cutoffs(args.cutoffs)
    is_path, oos_path = build_calibration(
        out_dir=args.out,
        cutoffs=cutoffs,
        target_per_corpus=args.target,
        endpoints=endpoints,
        is_strata=args.is_strata,
    )
    is_n = sum(1 for _ in is_path.open("r", encoding="utf-8"))
    oos_n = sum(1 for _ in oos_path.open("r", encoding="utf-8"))
    print(f"Wrote {is_n} IS rows to {is_path}, {oos_n} OOS rows to {oos_path}.")
    return 0


def _cli_update(args: argparse.Namespace) -> int:
    since: date | None = None
    if args.since:
        since = date.fromisoformat(args.since)
    oos_path = args.out / "oos_control.jsonl"
    rows_before = sum(1 for _ in oos_path.open("r", encoding="utf-8")) if oos_path.exists() else 0
    update_oos(out_dir=args.out, since=since)
    rows_after = sum(1 for _ in oos_path.open("r", encoding="utf-8"))
    print(f"Appended {rows_after - rows_before} new OOS rows to {oos_path}.")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _load_dotenv_quiet()
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    if args.command == "build":
        return _cli_build(args)
    if args.command == "update":
        return _cli_update(args)
    parser.error(f"Unknown command: {args.command!r}")
    return 2  # unreachable; parser.error exits


if __name__ == "__main__":
    raise SystemExit(main())
