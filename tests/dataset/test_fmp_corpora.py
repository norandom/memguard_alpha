"""Tests for src.dataset.fmp_corpora: FMP-backed calibration corpus builder.

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

from src.dataset.fmp_corpora import (
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
    """Patch ``requests.get`` to return the given payloads, one per call.

    After exhausting ``payload_pages`` returns an empty list, which is the
    convention the builder uses to detect end-of-pagination.
    """
    pages = list(payload_pages)

    def _side_effect(*args, **kwargs):
        if pages:
            payload = pages.pop(0)
        else:
            payload = []
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

    is_titles = {r["metadata"]["url"].split("/")[-1] for r in is_rows}
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

    with caplog.at_level(logging.WARNING, logger="src.dataset.fmp_corpora"):
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

    with caplog.at_level(logging.WARNING, logger="src.dataset.fmp_corpora"):
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
