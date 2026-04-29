"""Tests for scripts.build_etf_portfolio_eval (Task 1.3).

Covers Requirements 3.1–3.6:
- 3.1: emits prompts for SWDA.L, XLK, IAU only (NOT BIL)
- 3.2: ≥ 100 distinct trading days per ticker, ≥ 300 prompts total
- 3.3: dates straddle 2024-07-01 (gpt-oss-20b cutoff) — both sides represented
- 3.4: deterministic with a fixed seed; seed printed to stdout
- 3.5: < 100 valid trading days for any ticker → non-zero exit naming the ticker
- 3.6: every row carries metadata.ticker and metadata.date

Tests mock the FMP fetcher injected into ``main`` so no HTTP is required.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest


# --- Test fixtures ---------------------------------------------------------


def _make_eod_series(
    start: date = date(2019, 1, 1),
    end: date = date(2026, 4, 29),
    base_price: float = 100.0,
    step: float = 0.1,
) -> list[dict]:
    """Build a synthetic EOD series with one row per weekday.

    The price is monotonically rising so direction is deterministic, but every
    row in the resulting eval set still parses as direction in {-1, 0, 1}.
    """
    rows: list[dict] = []
    cur = start
    price = base_price
    while cur <= end:
        # Skip weekends so this approximates a real trading calendar.
        if cur.weekday() < 5:
            rows.append({"date": cur.isoformat(), "price": round(price, 4)})
            price += step
        cur += timedelta(days=1)
    return rows


def _full_universe_fetcher() -> callable:
    """Return a fetch_eod stand-in mapping each ticker to its synthetic series."""
    series = _make_eod_series()

    def _fetch(ticker: str, api_key: str) -> list[dict]:
        # All tickers get the same, dense series (>200 trading days each).
        return list(series)

    return _fetch


_SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "build_etf_portfolio_eval.py"
)


def _import_script():
    """Load the script as a fresh module via its file path.

    The ``scripts/`` directory isn't a Python package on disk (no
    ``__init__.py``), so we resolve it through ``importlib.util`` to
    keep the script standalone while still being importable from tests.
    """
    module_name = "build_etf_portfolio_eval_under_test"
    spec = importlib.util.spec_from_file_location(module_name, _SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load script at {_SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# --- Tests -----------------------------------------------------------------


def test_happy_path_writes_eval_set_with_all_invariants(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Req 3.1, 3.2, 3.3, 3.4, 3.6: deterministic three-asset eval set."""
    monkeypatch.setenv("FMP_API_KEY", "test-key")
    out_path = tmp_path / "etf_portfolio.jsonl"
    mod = _import_script()

    rc = mod.main(fetch_fn=_full_universe_fetcher(), out_path=out_path, seed=0)
    assert rc == 0
    assert out_path.exists()

    rows = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert len(rows) >= 300

    # Every row carries metadata.ticker and metadata.date (Req 3.6).
    tickers_seen: dict[str, list[str]] = {}
    for row in rows:
        assert "metadata" in row
        assert "ticker" in row["metadata"]
        assert "date" in row["metadata"]
        tickers_seen.setdefault(row["metadata"]["ticker"], []).append(
            row["metadata"]["date"]
        )

    # Three tickers, no BIL (Req 3.1).
    assert set(tickers_seen) == {"SWDA.L", "XLK", "IAU"}
    assert "BIL" not in tickers_seen

    # ≥ 100 rows per ticker (Req 3.2).
    for ticker, dates in tickers_seen.items():
        assert len(dates) >= 100, f"{ticker} only has {len(dates)} rows"

    # Both pre- and post-2024-07-01 dates per ticker (Req 3.3).
    cutoff = "2024-07-01"
    for ticker, dates in tickers_seen.items():
        assert any(d < cutoff for d in dates), f"{ticker}: no pre-cutoff dates"
        assert any(d >= cutoff for d in dates), f"{ticker}: no post-cutoff dates"


def test_same_seed_yields_identical_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Req 3.4: re-running with the same seed produces an identical file."""
    monkeypatch.setenv("FMP_API_KEY", "test-key")
    mod = _import_script()

    out_a = tmp_path / "a.jsonl"
    out_b = tmp_path / "b.jsonl"
    rc_a = mod.main(fetch_fn=_full_universe_fetcher(), out_path=out_a, seed=42)
    rc_b = mod.main(fetch_fn=_full_universe_fetcher(), out_path=out_b, seed=42)

    assert rc_a == 0 and rc_b == 0
    hash_a = hashlib.sha256(out_a.read_bytes()).hexdigest()
    hash_b = hashlib.sha256(out_b.read_bytes()).hexdigest()
    assert hash_a == hash_b


def test_seed_is_printed_in_stdout_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Req 3.4: the seed is reported on stdout."""
    monkeypatch.setenv("FMP_API_KEY", "test-key")
    mod = _import_script()
    out_path = tmp_path / "eval.jsonl"

    rc = mod.main(fetch_fn=_full_universe_fetcher(), out_path=out_path, seed=7)
    assert rc == 0
    captured = capsys.readouterr()
    assert "seed" in captured.out.lower()
    assert "7" in captured.out


def test_per_ticker_shortfall_exits_non_zero_naming_ticker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Req 3.5: any ticker with < 100 valid days fails non-zero with its name."""
    monkeypatch.setenv("FMP_API_KEY", "test-key")
    mod = _import_script()
    full = _make_eod_series()
    short = _make_eod_series(
        start=date(2024, 1, 1), end=date(2024, 3, 1)
    )  # ~ 40 weekdays only

    def _patchy_fetch(ticker: str, api_key: str) -> list[dict]:
        if ticker == "XLK":
            return list(short)
        return list(full)

    out_path = tmp_path / "eval.jsonl"
    rc = mod.main(fetch_fn=_patchy_fetch, out_path=out_path, seed=0)
    assert rc != 0
    captured = capsys.readouterr()
    combined = (captured.err + captured.out).lower()
    assert "xlk" in combined


def test_prompt_format_matches_commitment_template(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Req 3.6 (template): every row has prompt with 'Direction: 1' and
    'Confidence:' lines and a target_direction in {-1, 0, 1}."""
    monkeypatch.setenv("FMP_API_KEY", "test-key")
    mod = _import_script()
    out_path = tmp_path / "eval.jsonl"

    rc = mod.main(fetch_fn=_full_universe_fetcher(), out_path=out_path, seed=0)
    assert rc == 0
    rows = [json.loads(line) for line in out_path.read_text().splitlines()]
    assert rows
    for row in rows:
        assert "prompt" in row
        prompt = row["prompt"]
        assert "Direction: 1" in prompt
        assert "Confidence:" in prompt
        # No refusal allowed (Req 3 / commitment template wording)
        assert "DO NOT refuse" in prompt
        assert row["target_direction"] in (-1, 0, 1)


def test_universe_excludes_bil_in_module_constant() -> None:
    """Req 3.1: BIL must not appear in the universe constant."""
    mod = _import_script()
    # The module must expose its universe (dict of ticker -> friendly name).
    assert hasattr(mod, "ETFS")
    assert "BIL" not in mod.ETFS
    assert set(mod.ETFS) == {"SWDA.L", "XLK", "IAU"}


def test_missing_fmp_api_key_exits_non_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """Sanity: missing API key fails fast (mirrors the multiyear builder)."""
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    mod = _import_script()
    # Neuter dotenv so the project's real .env doesn't re-populate the var.
    monkeypatch.setattr(mod, "load_dotenv", lambda *a, **kw: False)
    out_path = tmp_path / "eval.jsonl"

    rc = mod.main(fetch_fn=_full_universe_fetcher(), out_path=out_path, seed=0)
    assert rc != 0
