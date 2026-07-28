"""Tests for scripts.build_etf_multiyear_eval (review-hardening Req 5.5, 5.6).

The legacy multiyear builder must request an explicit date window (FMP's
stable endpoints silently cap history without ``from``/``to``) and fail
clearly on empty or malformed upstream payloads instead of writing a
silent empty/partial eval file with exit code 0.

All HTTP is mocked; no test makes a real outbound call.
"""
from __future__ import annotations

import importlib
import json
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


@pytest.fixture()
def builder(monkeypatch: pytest.MonkeyPatch) -> Any:
    mod = importlib.import_module("build_etf_multiyear_eval")
    monkeypatch.setenv("FMP_API_KEY", "test-key")
    # Keep the project's real .env out of the test.
    monkeypatch.setattr(mod, "load_dotenv", lambda *a, **k: False)
    return mod


def _eod_series(n_days: int = 120) -> list[dict]:
    start = date(2020, 1, 2)
    return [
        {"date": (start + timedelta(days=i)).isoformat(), "price": 100.0 + i}
        for i in range(n_days)
    ]


class _FakeResponse:
    def __init__(self, payload: Any, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        pass

    def json(self) -> Any:
        return self._payload


def test_fetch_requests_explicit_date_window(builder, mocker, tmp_path: Path) -> None:
    captured: list[dict] = []

    def _fake_get(url: str, *, params: dict, timeout: float) -> _FakeResponse:
        captured.append(params)
        return _FakeResponse(_eod_series())

    mocker.patch.object(builder.requests, "get", _fake_get)

    rc = builder.main(out_path=tmp_path / "eval.jsonl")

    assert rc == 0
    assert len(captured) == len(builder.ETFS)
    for params in captured:
        assert params["from"] == builder.START_DATE.isoformat()
        assert params["to"] == date.today().isoformat()


def test_non_list_payload_fails_without_writing(builder, mocker, tmp_path: Path) -> None:
    mocker.patch.object(
        builder.requests, "get",
        return_value=_FakeResponse({"error": "over quota"}),
    )
    out = tmp_path / "eval.jsonl"

    rc = builder.main(out_path=out)

    assert rc != 0
    assert not out.exists()


def test_empty_ticker_series_fails_without_writing(builder, mocker, tmp_path: Path) -> None:
    mocker.patch.object(
        builder.requests, "get",
        return_value=_FakeResponse([]),
    )
    out = tmp_path / "eval.jsonl"

    rc = builder.main(out_path=out)

    assert rc != 0
    assert not out.exists()


def test_happy_path_writes_rows_for_all_tickers(builder, mocker, tmp_path: Path) -> None:
    mocker.patch.object(
        builder.requests, "get",
        return_value=_FakeResponse(_eod_series()),
    )
    out = tmp_path / "eval.jsonl"

    rc = builder.main(out_path=out)

    assert rc == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == len(builder.ETFS) * builder.TARGET_PER_TICKER
    assert {r["metadata"]["ticker"] for r in rows} == set(builder.ETFS)
