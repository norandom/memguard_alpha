"""Tests for recall_guard.dataset.fmp_corpora: FMP-backed calibration corpus builder.

Covers requirement 11.1-11.5 from the honest-model-ranking spec.

All HTTP traffic is mocked via ``pytest-mock`` patching ``requests.get``;
no test in this module should make a real outbound call.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from recall_guard.dataset.fmp_corpora import (
    DEFAULT_ENDPOINTS,
    ArticleRecord,
    build_calibration,
    fetch_articles,
    update_oos,
)

# ---------- helpers ----------------------------------------------------------


def _article(
    *,
    title: str = "A title",
    body: str = "A long enough body to be useful for memorisation testing.",
    body_field: str = "content",
    published: str = "2024-01-15 09:00:00",
    url: str | None = None,
) -> dict[str, Any]:
    """Synthetic FMP article matching the documented endpoint shape."""
    if url is None:
        url = f"https://example.com/{hashlib.sha1(title.encode()).hexdigest()[:10]}"
    return {
        "title": title,
        body_field: body,
        "publishedDate": published,
        "url": url,
    }


def _mock_response(mocker, payload_pages: list[list[dict[str, Any]]]):
    """Patch ``requests.get`` to return date-window-filtered payloads.

    The mock parses the ``from=`` and ``to=`` query parameters from the
    requested URL and returns only those articles in ``payload_pages[0]``
    whose ``publishedDate`` falls inside that window. This mirrors FMP's
    own filtering and is required for the sub-window stratified IS sampler
    in ``build_calibration``: each sub-window call gets only the articles
    relevant to that bucket.

    The first call returns articles from window 1, the second uses window
    2, and so on -- but each call's payload is independently filtered, not
    consumed. After ``payload_pages`` is exhausted the mock returns ``[]``
    (the builder's end-of-pagination convention).

    For backwards compatibility with the existing tests: passing a single
    ``[articles]`` page means every sub-window call sees the SAME source
    list, with date-window filtering applied per call. This lets one test
    payload populate multiple buckets correctly.
    """
    pages = list(payload_pages)
    # When exactly one payload page is supplied, treat it as a "shared"
    # payload visible to every fetch_articles call (subject to date-window
    # filtering). When multiple are supplied, treat them as a per-call
    # FIFO queue (legacy behaviour).
    shared = pages[0] if len(pages) == 1 else None

    def _side_effect(*args, **kwargs):
        url = args[0] if args else kwargs.get("url", "")
        # Parse from=YYYY-MM-DD and to=YYYY-MM-DD out of the URL.
        from_d = None
        to_d = None
        for chunk in str(url).split("&"):
            if chunk.startswith("from="):
                try:
                    from_d = date.fromisoformat(chunk[len("from="):])
                except ValueError:
                    from_d = None
            elif "from=" in chunk:
                key = chunk.split("from=", 1)[1]
                try:
                    from_d = date.fromisoformat(key)
                except ValueError:
                    from_d = None
            if chunk.startswith("to="):
                try:
                    to_d = date.fromisoformat(chunk[len("to="):])
                except ValueError:
                    to_d = None

        if shared is not None:
            source = shared
        elif pages:
            source = pages.pop(0)
        else:
            source = []

        def _in_window(article: dict[str, Any]) -> bool:
            if from_d is None or to_d is None:
                return True
            raw = str(article.get("publishedDate", "")).strip()
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                try:
                    from datetime import datetime as _dt
                    parsed = _dt.strptime(raw, fmt).date()
                    return from_d <= parsed <= to_d
                except ValueError:
                    continue
            # Unparseable dates pass through so the builder can emit its
            # own "skip article: unparseable publishedDate" WARNING; FMP
            # would not pre-filter these either.
            return True

        payload = [a for a in source if _in_window(a)]
        resp = mocker.Mock()
        resp.status_code = 200
        resp.json = lambda payload=payload: payload
        return resp

    return mocker.patch("requests.get", side_effect=_side_effect)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------- ArticleRecord ----------------------------------------------------


def test_article_record_is_frozen_dataclass() -> None:
    """ArticleRecord matches the design contract: frozen, exact field types."""
    rec = ArticleRecord(
        prompt="title + body",
        label=1,
        published_at=date(2024, 1, 15),
        source="fmp-articles",
        url="https://example.com/x",
    )

    assert dataclasses.is_dataclass(rec)
    params = rec.__dataclass_params__
    assert params.frozen is True

    fields = {f.name: f.type for f in dataclasses.fields(ArticleRecord)}
    assert set(fields.keys()) == {"prompt", "label", "published_at", "source", "url"}

    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.label = 0  # type: ignore[misc]


# ---------- fetch_articles ---------------------------------------------------


def test_fetch_articles_raises_on_non_200(mocker) -> None:
    resp = mocker.Mock()
    resp.status_code = 500
    resp.json = lambda: {"error": "boom"}
    mocker.patch("requests.get", return_value=resp)

    with pytest.raises(RuntimeError) as exc:
        fetch_articles(
            endpoint="fmp-articles",
            api_key="k",
            from_date=date(2020, 1, 1),
            to_date=date(2020, 12, 31),
            page=0,
            limit=100,
        )
    msg = str(exc.value)
    assert "fmp-articles" in msg
    assert "500" in msg


def test_fetch_articles_builds_correct_url(mocker) -> None:
    resp = mocker.Mock()
    resp.status_code = 200
    resp.json = lambda: []
    mock_get = mocker.patch("requests.get", return_value=resp)

    fetch_articles(
        endpoint="news/general-latest",
        api_key="SECRET",
        from_date=date(2024, 1, 1),
        to_date=date(2024, 6, 30),
        page=2,
        limit=50,
    )

    args, kwargs = mock_get.call_args
    url = args[0] if args else kwargs["url"]
    assert "https://financialmodelingprep.com/stable/news/general-latest" in url
    assert "from=2024-01-01" in url
    assert "to=2024-06-30" in url
    assert "page=2" in url
    assert "limit=50" in url
    assert "apikey=SECRET" in url
    assert kwargs.get("timeout") == 15


# ---------- build_calibration ------------------------------------------------


def test_build_calibration_filters_by_date_window(tmp_path: Path, mocker) -> None:
    """Articles before earliest cutoff -> IS; after latest cutoff -> OOS;
    in the gap -> dropped."""
    cutoffs = {"a": date(2024, 1, 1), "b": date(2024, 12, 1)}
    today = date(2026, 4, 1)

    articles = [
        _article(title="t-pre-2020", published="2020-01-15 00:00:00"),
        _article(title="t-pre-2023", published="2023-06-15 00:00:00"),
        _article(title="t-gap-2024-mar", published="2024-03-15 00:00:00"),
        _article(title="t-gap-2024-aug", published="2024-08-15 00:00:00"),
        _article(title="t-post-2025", published="2025-02-15 00:00:00"),
        _article(title="t-post-2026", published="2026-01-15 00:00:00"),
    ]
    _mock_response(mocker, [articles])

    is_path, oos_path = build_calibration(
        out_dir=tmp_path,
        cutoffs=cutoffs,
        target_per_corpus=10,
        api_key="key",
        endpoints=("fmp-articles",),
        today=today,
    )

    is_rows = _read_jsonl(is_path)
    oos_rows = _read_jsonl(oos_path)

    assert len(is_rows) == 2  # both pre-2024-01-01 articles
    assert all(r["label"] == 1 for r in is_rows)

    assert len(oos_rows) == 2  # both post-2024-12-01 articles
    assert all(r["label"] == 0 for r in oos_rows)


def test_build_calibration_dedups_by_url(tmp_path: Path, mocker) -> None:
    cutoffs = {"a": date(2024, 1, 1), "b": date(2024, 12, 1)}
    today = date(2026, 4, 1)
    same_url = "https://example.com/dup"
    articles = [
        _article(title="t1", published="2020-01-15 00:00:00", url=same_url),
        _article(title="t2-different-title", published="2020-02-15 00:00:00", url=same_url),
    ]
    _mock_response(mocker, [articles])

    is_path, _ = build_calibration(
        out_dir=tmp_path,
        cutoffs=cutoffs,
        target_per_corpus=10,
        api_key="key",
        endpoints=("fmp-articles",),
        today=today,
    )

    is_rows = _read_jsonl(is_path)
    assert len(is_rows) == 1


def test_build_calibration_dedups_by_title_hash(tmp_path: Path, mocker) -> None:
    cutoffs = {"a": date(2024, 1, 1), "b": date(2024, 12, 1)}
    today = date(2026, 4, 1)
    articles = [
        _article(
            title="Identical Title",
            published="2020-01-15 00:00:00",
            url="https://example.com/u1",
        ),
        _article(
            title="Identical Title",
            published="2020-02-15 00:00:00",
            url="https://example.com/u2",
        ),
    ]
    _mock_response(mocker, [articles])

    is_path, _ = build_calibration(
        out_dir=tmp_path,
        cutoffs=cutoffs,
        target_per_corpus=10,
        api_key="key",
        endpoints=("fmp-articles",),
        today=today,
    )

    assert len(_read_jsonl(is_path)) == 1


def test_build_calibration_skips_missing_body_with_warning(
    tmp_path: Path, mocker, caplog
) -> None:
    cutoffs = {"a": date(2024, 1, 1), "b": date(2024, 12, 1)}
    today = date(2026, 4, 1)
    articles = [
        _article(title="ok", published="2020-01-15 00:00:00"),
        # Empty body across both possible body fields:
        {
            "title": "no body",
            "content": "",
            "text": "",
            "publishedDate": "2020-02-15 00:00:00",
            "url": "https://example.com/empty",
        },
    ]
    _mock_response(mocker, [articles])

    with caplog.at_level(logging.WARNING, logger="recall_guard.dataset.fmp_corpora"):
        is_path, _ = build_calibration(
            out_dir=tmp_path,
            cutoffs=cutoffs,
            target_per_corpus=10,
            api_key="key",
            endpoints=("fmp-articles",),
            today=today,
        )

    is_rows = _read_jsonl(is_path)
    assert len(is_rows) == 1
    skip_warnings = [r for r in caplog.records if "skip" in r.getMessage().lower()]
    assert any("body" in r.getMessage().lower() for r in skip_warnings)


def test_build_calibration_skips_unparseable_date_with_warning(
    tmp_path: Path, mocker, caplog
) -> None:
    cutoffs = {"a": date(2024, 1, 1), "b": date(2024, 12, 1)}
    today = date(2026, 4, 1)
    articles = [
        _article(title="ok", published="2020-01-15 00:00:00"),
        _article(title="bad date", published="not a date"),
    ]
    _mock_response(mocker, [articles])

    with caplog.at_level(logging.WARNING, logger="recall_guard.dataset.fmp_corpora"):
        is_path, _ = build_calibration(
            out_dir=tmp_path,
            cutoffs=cutoffs,
            target_per_corpus=10,
            api_key="key",
            endpoints=("fmp-articles",),
            today=today,
        )

    is_rows = _read_jsonl(is_path)
    assert len(is_rows) == 1
    date_warnings = [
        r
        for r in caplog.records
        if "skip" in r.getMessage().lower() and "date" in r.getMessage().lower()
    ]
    assert date_warnings


def test_build_calibration_writes_label_1_and_label_0_correctly(
    tmp_path: Path, mocker
) -> None:
    cutoffs = {"a": date(2024, 1, 1), "b": date(2024, 12, 1)}
    today = date(2026, 4, 1)
    articles = [
        _article(title="is1", published="2020-01-15 00:00:00"),
        _article(title="is2", published="2022-06-15 00:00:00"),
        _article(title="oos1", published="2025-03-15 00:00:00"),
        _article(title="oos2", published="2026-02-15 00:00:00"),
    ]
    _mock_response(mocker, [articles])

    is_path, oos_path = build_calibration(
        out_dir=tmp_path,
        cutoffs=cutoffs,
        target_per_corpus=10,
        api_key="key",
        endpoints=("fmp-articles",),
        today=today,
    )

    is_rows = _read_jsonl(is_path)
    oos_rows = _read_jsonl(oos_path)
    assert is_rows and all(r["label"] == 1 for r in is_rows)
    assert oos_rows and all(r["label"] == 0 for r in oos_rows)
    # Schema check: every row has prompt + label + metadata{published_at,source,url}.
    for row in is_rows + oos_rows:
        assert isinstance(row["prompt"], str) and row["prompt"]
        assert isinstance(row["label"], int)
        meta = row["metadata"]
        assert set(meta.keys()) >= {"published_at", "source", "url"}
        # ISO-8601 published_at:
        date.fromisoformat(meta["published_at"])


def test_build_calibration_raises_when_no_oos_window(tmp_path: Path) -> None:
    cutoffs = {"a": date(2099, 1, 1), "b": date(2099, 12, 1)}
    today = date(2026, 4, 1)
    with pytest.raises(ValueError):
        build_calibration(
            out_dir=tmp_path,
            cutoffs=cutoffs,
            target_per_corpus=10,
            api_key="key",
            endpoints=("fmp-articles",),
            today=today,
        )


def test_build_calibration_requires_api_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    cutoffs = {"a": date(2024, 1, 1), "b": date(2024, 12, 1)}
    with pytest.raises(RuntimeError):
        build_calibration(
            out_dir=tmp_path,
            cutoffs=cutoffs,
            target_per_corpus=10,
            api_key=None,
            endpoints=("fmp-articles",),
            today=date(2026, 4, 1),
        )


# ---------- update_oos -------------------------------------------------------


def _seed_oos(out_dir: Path, rows: list[dict[str, Any]]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / "oos_control.jsonl"
    with p.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return p


def test_update_oos_appends_only_new_post_max_date(tmp_path: Path, mocker) -> None:
    existing = [
        {
            "prompt": "old prompt 1",
            "label": 0,
            "metadata": {
                "published_at": "2026-02-10",
                "source": "fmp-articles",
                "url": "https://example.com/old1",
            },
        },
        {
            "prompt": "old prompt 2",
            "label": 0,
            "metadata": {
                "published_at": "2026-02-15",
                "source": "fmp-articles",
                "url": "https://example.com/old2",
            },
        },
    ]
    _seed_oos(tmp_path, existing)

    articles = [
        _article(title="article-2026-02-10", published="2026-02-10 00:00:00",
                 url="https://example.com/old1"),  # older than max + dup -> skipped
        _article(title="article-2026-02-20", published="2026-02-20 00:00:00",
                 url="https://example.com/new1"),
        _article(title="article-2026-03-01", published="2026-03-01 00:00:00",
                 url="https://example.com/new2"),
    ]
    _mock_response(mocker, [articles])

    out_path = update_oos(
        out_dir=tmp_path,
        api_key="key",
        endpoints=("fmp-articles",),
        today=date(2026, 4, 1),
    )

    rows = _read_jsonl(out_path)
    assert len(rows) == 4  # 2 existing + 2 new
    new_urls = {r["metadata"]["url"] for r in rows[2:]}
    assert new_urls == {"https://example.com/new1", "https://example.com/new2"}


def test_update_oos_dedups_against_existing_rows(tmp_path: Path, mocker) -> None:
    existing = [
        {
            "prompt": "x",
            "label": 0,
            "metadata": {
                "published_at": "2026-02-15",
                "source": "fmp-articles",
                "url": "https://example.com/X",
            },
        }
    ]
    _seed_oos(tmp_path, existing)

    articles = [
        _article(title="dup-X", published="2026-03-01 00:00:00",
                 url="https://example.com/X"),
        _article(title="brand-new-Y", published="2026-03-02 00:00:00",
                 url="https://example.com/Y"),
    ]
    _mock_response(mocker, [articles])

    out_path = update_oos(
        out_dir=tmp_path,
        api_key="key",
        endpoints=("fmp-articles",),
        today=date(2026, 4, 1),
    )

    rows = _read_jsonl(out_path)
    urls = [r["metadata"]["url"] for r in rows]
    assert urls.count("https://example.com/X") == 1
    assert "https://example.com/Y" in urls
    assert len(rows) == 2


def test_update_oos_does_not_modify_is_memorized(tmp_path: Path, mocker) -> None:
    sentinel = {
        "prompt": "DO NOT TOUCH",
        "label": 1,
        "metadata": {
            "published_at": "2020-01-15",
            "source": "fmp-articles",
            "url": "https://example.com/sentinel",
        },
    }
    is_path = tmp_path / "is_memorized.jsonl"
    with is_path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(sentinel) + "\n")
    is_bytes_before = is_path.read_bytes()

    _seed_oos(
        tmp_path,
        [
            {
                "prompt": "x",
                "label": 0,
                "metadata": {
                    "published_at": "2026-02-15",
                    "source": "fmp-articles",
                    "url": "https://example.com/x",
                },
            }
        ],
    )

    articles = [
        _article(title="new", published="2026-03-15 00:00:00",
                 url="https://example.com/new"),
    ]
    _mock_response(mocker, [articles])

    update_oos(
        out_dir=tmp_path,
        api_key="key",
        endpoints=("fmp-articles",),
        today=date(2026, 4, 1),
    )

    assert is_path.read_bytes() == is_bytes_before


def test_update_oos_raises_when_oos_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        update_oos(
            out_dir=tmp_path,
            api_key="key",
            endpoints=("fmp-articles",),
            today=date(2026, 4, 1),
        )


# ---------- defaults sanity check --------------------------------------------


def test_default_endpoints_exclude_stock_news() -> None:
    """Per design: stock-latest is opt-in via parameter, not a default."""
    assert "news/stock-latest" not in DEFAULT_ENDPOINTS
    assert "fmp-articles" in DEFAULT_ENDPOINTS
    assert "news/general-latest" in DEFAULT_ENDPOINTS


# ---------- IS-stratification (task 1.5) -------------------------------------


def test_build_calibration_stratifies_is_across_years(
    tmp_path: Path, mocker
) -> None:
    """Sub-window stratification must spread IS rows across distinct years.

    Synthetic FMP coverage: 50 articles per year for years 2010, 2013, 2016,
    2019, 2022 (one year per IS sub-window when ``is_strata=5``). With
    ``target_per_corpus=100`` the stratified sampler should write ~20 rows
    per bucket and therefore land at least 4 distinct years in the output.
    """
    cutoffs = {"a": date(2024, 1, 1), "b": date(2024, 12, 1)}
    today = date(2026, 4, 1)

    seed_years = [2010, 2013, 2016, 2019, 2022]
    articles: list[dict[str, Any]] = []
    for year in seed_years:
        for n in range(50):
            articles.append(
                _article(
                    title=f"y{year}-n{n}",
                    published=f"{year}-06-15 00:00:00",
                    url=f"https://example.com/{year}/{n}",
                )
            )
    # Add OOS coverage so the OOS bucket can complete without warnings.
    for n in range(120):
        articles.append(
            _article(
                title=f"oos-{n}",
                published="2026-02-15 00:00:00",
                url=f"https://example.com/oos/{n}",
            )
        )
    _mock_response(mocker, [articles])

    is_path, _ = build_calibration(
        out_dir=tmp_path,
        cutoffs=cutoffs,
        target_per_corpus=100,
        api_key="key",
        endpoints=("fmp-articles",),
        today=today,
        is_strata=5,
    )

    is_rows = _read_jsonl(is_path)
    years = {
        date.fromisoformat(r["metadata"]["published_at"]).year for r in is_rows
    }
    assert len(years) >= 4, (
        f"expected IS rows from >= 4 distinct years, got {sorted(years)}"
    )


def test_build_calibration_remainder_goes_to_last_bucket(
    tmp_path: Path, mocker
) -> None:
    """``target_per_corpus=23, is_strata=5`` -> 4 per bucket + 7 in last."""
    cutoffs = {"a": date(2024, 1, 1), "b": date(2024, 12, 1)}
    today = date(2026, 4, 1)

    # Provide thick coverage in every bucket so per-bucket cap is the limit.
    seed_years = [2011, 2014, 2017, 2020, 2023]
    articles: list[dict[str, Any]] = []
    for year in seed_years:
        for n in range(50):
            articles.append(
                _article(
                    title=f"y{year}-n{n}",
                    published=f"{year}-06-15 00:00:00",
                    url=f"https://example.com/{year}/{n}",
                )
            )
    # Sufficient OOS to satisfy the OOS bucket; not the focus of this test.
    for n in range(50):
        articles.append(
            _article(
                title=f"oos-{n}",
                published="2026-03-01 00:00:00",
                url=f"https://example.com/oos/{n}",
            )
        )
    _mock_response(mocker, [articles])

    is_path, _ = build_calibration(
        out_dir=tmp_path,
        cutoffs=cutoffs,
        target_per_corpus=23,
        api_key="key",
        endpoints=("fmp-articles",),
        today=today,
        is_strata=5,
    )

    is_rows = _read_jsonl(is_path)
    counts: dict[int, int] = {}
    for r in is_rows:
        y = date.fromisoformat(r["metadata"]["published_at"]).year
        counts[y] = counts.get(y, 0) + 1

    # Total must equal target.
    assert sum(counts.values()) == 23
    # First four buckets get target_per_corpus // is_strata == 4 rows each.
    early_years = sorted(counts.keys())[:4]
    for y in early_years:
        assert counts[y] == 4, (
            f"bucket year {y} expected 4 rows, got {counts[y]} (counts={counts})"
        )
    # Last bucket absorbs the remainder.
    last_year = sorted(counts.keys())[-1]
    assert counts[last_year] == 7, (
        f"last bucket expected 7 rows (4 base + 3 remainder), got {counts[last_year]}"
    )


def test_build_calibration_continues_after_empty_bucket(
    tmp_path: Path, mocker, caplog
) -> None:
    """An empty sub-window logs a WARNING and the next bucket still fills."""
    cutoffs = {"a": date(2024, 1, 1), "b": date(2024, 12, 1)}
    today = date(2026, 4, 1)

    # Coverage in every year except those that fall in the FIRST sub-window
    # (~2010-01-01 to 2012-09-25 with K=5). Skipping the first bucket forces
    # the per-bucket "came up short" WARNING.
    populated_years = [2014, 2017, 2020, 2023]
    articles: list[dict[str, Any]] = []
    for year in populated_years:
        for n in range(20):
            articles.append(
                _article(
                    title=f"y{year}-n{n}",
                    published=f"{year}-06-15 00:00:00",
                    url=f"https://example.com/{year}/{n}",
                )
            )
    for n in range(50):
        articles.append(
            _article(
                title=f"oos-{n}",
                published="2026-03-01 00:00:00",
                url=f"https://example.com/oos/{n}",
            )
        )
    _mock_response(mocker, [articles])

    with caplog.at_level(logging.WARNING, logger="recall_guard.dataset.fmp_corpora"):
        is_path, _ = build_calibration(
            out_dir=tmp_path,
            cutoffs=cutoffs,
            target_per_corpus=20,
            api_key="key",
            endpoints=("fmp-articles",),
            today=today,
            is_strata=5,
        )

    # The empty first bucket emitted a per-bucket WARNING (mentions the
    # bucket index or sub-window phrase).
    bucket_warnings = [
        r
        for r in caplog.records
        if "bucket" in r.getMessage().lower()
        and ("0" in r.getMessage() or "short" in r.getMessage().lower())
    ]
    assert bucket_warnings, (
        "expected a per-bucket WARNING for the empty first sub-window; "
        f"saw: {[r.getMessage() for r in caplog.records]}"
    )

    is_rows = _read_jsonl(is_path)
    # The non-empty buckets still contributed rows (>= 4 per non-empty
    # bucket * 4 buckets = 16, but we cap at target_per_corpus=20 with last
    # bucket absorbing the remainder, so >= the 4 base rows from each
    # populated bucket = 16).
    assert len(is_rows) >= 16, f"expected >= 16 IS rows, got {len(is_rows)}"


def test_build_calibration_dedup_persists_across_buckets(
    tmp_path: Path, mocker
) -> None:
    """A URL appearing in two sub-windows is written exactly once."""
    cutoffs = {"a": date(2024, 1, 1), "b": date(2024, 12, 1)}
    today = date(2026, 4, 1)

    duplicate_url = "https://example.com/dup-across-buckets"
    duplicate_title = "Duplicate-Across-Buckets-Title"

    articles: list[dict[str, Any]] = [
        # Same URL+title published in 2011 (bucket 0) and 2017 (bucket 2).
        _article(
            title=duplicate_title,
            published="2011-06-15 00:00:00",
            url=duplicate_url,
        ),
        _article(
            title=duplicate_title,
            published="2017-06-15 00:00:00",
            url=duplicate_url,
        ),
        # Filler articles to keep the run progressing.
        _article(
            title="filler-2014",
            published="2014-06-15 00:00:00",
            url="https://example.com/filler-2014",
        ),
        _article(
            title="filler-2020",
            published="2020-06-15 00:00:00",
            url="https://example.com/filler-2020",
        ),
        _article(
            title="filler-2023",
            published="2023-06-15 00:00:00",
            url="https://example.com/filler-2023",
        ),
    ]
    for n in range(20):
        articles.append(
            _article(
                title=f"oos-{n}",
                published="2026-03-01 00:00:00",
                url=f"https://example.com/oos/{n}",
            )
        )
    _mock_response(mocker, [articles])

    is_path, _ = build_calibration(
        out_dir=tmp_path,
        cutoffs=cutoffs,
        target_per_corpus=10,
        api_key="key",
        endpoints=("fmp-articles",),
        today=today,
        is_strata=5,
    )

    is_rows = _read_jsonl(is_path)
    dup_count = sum(
        1 for r in is_rows if r["metadata"]["url"] == duplicate_url
    )
    assert dup_count == 1, (
        f"expected duplicate URL written exactly once across all buckets, "
        f"got {dup_count}"
    )


def test_build_calibration_default_is_strata_is_five(tmp_path: Path, mocker) -> None:
    """``is_strata`` defaults to 5 per the task brief and design Open Defaults."""
    cutoffs = {"a": date(2024, 1, 1), "b": date(2024, 12, 1)}
    today = date(2026, 4, 1)

    # Coverage across 5 distinct years, one per bucket, so default-K behaviour
    # is observable in the output.
    seed_years = [2011, 2014, 2017, 2020, 2023]
    articles: list[dict[str, Any]] = []
    for year in seed_years:
        for n in range(20):
            articles.append(
                _article(
                    title=f"y{year}-n{n}",
                    published=f"{year}-06-15 00:00:00",
                    url=f"https://example.com/{year}/{n}",
                )
            )
    for n in range(20):
        articles.append(
            _article(
                title=f"oos-{n}",
                published="2026-03-01 00:00:00",
                url=f"https://example.com/oos/{n}",
            )
        )
    _mock_response(mocker, [articles])

    is_path, _ = build_calibration(
        out_dir=tmp_path,
        cutoffs=cutoffs,
        target_per_corpus=20,
        api_key="key",
        endpoints=("fmp-articles",),
        today=today,
        # Note: NO is_strata passed -- relying on the new default.
    )

    is_rows = _read_jsonl(is_path)
    years = {date.fromisoformat(r["metadata"]["published_at"]).year for r in is_rows}
    # Default K=5 -> at least 4 distinct years covered.
    assert len(years) >= 4, (
        f"default is_strata should stratify; got years {sorted(years)}"
    )


def test_cli_build_exposes_is_strata_flag() -> None:
    """`python -m recall_guard.dataset.fmp_corpora build --help` lists --is-strata."""
    from recall_guard.dataset.fmp_corpora import _build_arg_parser

    parser = _build_arg_parser()
    parser.format_help()  # smoke-render the help text
    # Subparser help is rendered separately; assert by introspecting the build
    # subparser's actions directly.
    build_subparser = parser._subparsers._group_actions[0].choices["build"]
    flag_names = {
        action.option_strings[0]
        for action in build_subparser._actions
        if action.option_strings
    }
    assert "--is-strata" in flag_names, (
        f"expected --is-strata in build CLI flags; saw {flag_names}"
    )
