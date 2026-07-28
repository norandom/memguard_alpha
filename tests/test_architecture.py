"""Architecture-gate tests (review-hardening Req 6.1).

The gate itself lives in ``scripts/check_architecture.py`` (stdlib-only so
it runs inside the CI lint container). These tests prove two things:

1. the real package passes the gate, and
2. the gate actually fires on upward imports, sibling imports, and the
   banned ``dspy`` dependency — so a green run means something.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

checker = importlib.import_module("check_architecture")


def test_real_package_has_no_violations() -> None:
    violations = checker.find_violations(PROJECT_ROOT / "recall_guard")
    assert violations == []


def _make_pkg(root: Path, module_rel: str, body: str) -> Path:
    pkg = root / "recall_guard"
    target = pkg / module_rel
    target.parent.mkdir(parents=True, exist_ok=True)
    for d in [target.parent, pkg]:
        (d / "__init__.py").touch()
    target.write_text(body, encoding="utf-8")
    return pkg


def test_gate_flags_upward_import(tmp_path: Path) -> None:
    pkg = _make_pkg(
        tmp_path, "core/loader.py", "from recall_guard.harness import runner\n"
    )
    violations = checker.find_violations(pkg)
    assert len(violations) == 1
    assert "core" in violations[0] and "harness" in violations[0]


def test_gate_flags_sibling_import(tmp_path: Path) -> None:
    pkg = _make_pkg(
        tmp_path, "portfolio/prices.py", "import recall_guard.dataset.fmp_corpora\n"
    )
    violations = checker.find_violations(pkg)
    assert len(violations) == 1
    assert "portfolio" in violations[0] and "dataset" in violations[0]


def test_gate_flags_banned_dspy(tmp_path: Path) -> None:
    pkg = _make_pkg(tmp_path, "harness/runner.py", "import dspy\n")
    violations = checker.find_violations(pkg)
    assert violations == ["harness/runner.py: banned import 'dspy'"]


def test_gate_allows_downward_and_root_imports(tmp_path: Path) -> None:
    pkg = _make_pkg(
        tmp_path, "harness/runner.py", "from recall_guard.core import loader\n"
    )
    (pkg / "__init__.py").write_text(
        "from recall_guard.harness.scorer import MemoryGuardedScorer\n",
        encoding="utf-8",
    )
    assert checker.find_violations(pkg) == []
