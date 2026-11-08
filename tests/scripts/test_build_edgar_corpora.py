"""CLI tests for scripts/build_edgar_corpora.py: dry run and missing User-Agent make no request."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

CUTOFFS = str(PROJECT_ROOT / "data" / "cutoffs.yaml")


@pytest.fixture()
def cli(monkeypatch: pytest.MonkeyPatch) -> Any:
    import requests

    def no_network(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("network call attempted")

    monkeypatch.setattr(requests.Session, "request", no_network)
    return importlib.import_module("build_edgar_corpora")


def _no_session(_ua: str) -> Any:
    raise AssertionError("session must not be created")


def test_dry_run_prints_plan_without_requests(cli: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str],
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEC_USER_AGENT", "Jane Doe jane@example.com")
    code = cli.main(
        ["--out-dir", str(tmp_path / "out"), "--dry-run", "--cutoffs", CUTOFFS, "--today", "2026-11-08",
         "--cache-dir", str(tmp_path / "cache")],
        session_factory=_no_session,
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "jane@example.com" not in out
    plan = json.loads(out)
    assert plan["earliest_cutoff"] == "2023-03-31" and plan["latest_cutoff"] == "2024-08-31"
    assert plan["targets"] == {"calibration_memorized": 40, "calibration_control": 100,
                               "held_out_memorized": 40, "held_out_control": 40}
    assert plan["max_requests"] == 3000 and plan["SEC_USER_AGENT_set"] is True
    assert not (tmp_path / "out").exists() and not (tmp_path / "cache").exists()


def test_missing_user_agent_fails_before_any_request(cli: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                     capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    code = cli.main(["--out-dir", str(tmp_path / "out"), "--cutoffs", CUTOFFS, "--today", "2026-11-08"],
                    session_factory=_no_session)
    assert code == 1
    assert "SEC_USER_AGENT is not set" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_unknown_model_is_a_config_error(cli: Any, tmp_path: Path) -> None:
    code = cli.main(["--out-dir", str(tmp_path), "--dry-run", "--cutoffs", CUTOFFS, "--models", "nope/model"],
                    session_factory=_no_session)
    assert code == 1
