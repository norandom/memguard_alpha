"""The ensemble surface must not loosen the constraints the package enforces.

`tests/test_packaging.py` and `tests/test_architecture.py` already pin the
dependency set and the layer graph in general. These assertions are specific to
what the ensemble work added, and exist because each is a constraint that a
plausible future change would quietly break.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = PROJECT_ROOT / "recall_guard"

#: Ceilings mirrored from `.sentrux/rules.toml`. The structural tool is not run
#: in CI, and the tree already exceeds these in `portfolio/backtest.py`, so the
#: contract is "no *new* violations" -- hence checking the new modules only.
MAX_FN_LINES = 120
MAX_CC = 25

NEW_MODULES = (
    PACKAGE_ROOT / "core" / "consensus.py",
    PACKAGE_ROOT / "core" / "ensemble.py",
)


def _functions(path: Path):
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            length = node.end_lineno - node.lineno + 1
            complexity = 1 + sum(
                isinstance(
                    child,
                    ast.If | ast.For | ast.While | ast.ExceptHandler | ast.BoolOp | ast.IfExp,
                )
                for child in ast.walk(node)
            )
            yield node.name, length, complexity


@pytest.mark.parametrize("path", NEW_MODULES, ids=lambda p: p.name)
def test_new_modules_introduce_no_ceiling_violation(path: Path) -> None:
    over = [
        (name, length, cc)
        for name, length, cc in _functions(path)
        if length > MAX_FN_LINES or cc > MAX_CC
    ]
    assert not over, f"{path.name}: {over}"


@pytest.mark.parametrize("path", NEW_MODULES, ids=lambda p: p.name)
def test_new_modules_import_no_third_party_runtime_dependency(path: Path) -> None:
    """Stdlib and same-layer only.

    The reduction is deliberately hand-rolled: scipy resolves transitively via
    scikit-learn but is undeclared, so importing it would create a dependency
    the packaging contract does not record.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    assert "scipy" not in roots
    assert "numpy" not in roots, "the reduction is stdlib-only by design"
    assert roots <= {"recall_guard"} | set(sys.stdlib_module_names)


def test_new_modules_use_absolute_imports_only() -> None:
    """Relative imports are invisible to the architecture gate.

    The checker only inspects `ImportFrom` nodes with `level == 0`, so a
    relative import could cross a layer boundary with CI green.
    """
    for path in NEW_MODULES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        relative = [
            node.module for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.level != 0
        ]
        assert not relative, f"{path.name} uses relative imports: {relative}"


def test_ensemble_modules_live_in_a_registered_layer() -> None:
    """A module under an unregistered directory is skipped by the gate entirely.

    `check_architecture` resolves the layer from the first path segment; an
    unknown segment yields no order, and every layer-direction check for that
    file is silently skipped. Keeping the new code in `core` avoids the trap --
    this asserts nobody later moves it somewhere unpinned.
    """
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    try:
        import check_architecture
    finally:
        sys.path.pop(0)

    for path in NEW_MODULES:
        layer = path.relative_to(PACKAGE_ROOT).parts[0]
        assert layer in check_architecture.LAYERS, f"{path} sits outside the gate"
    assert check_architecture.find_violations(PACKAGE_ROOT) == []


def test_ensemble_surface_imports_without_the_heavy_stacks() -> None:
    """The ensemble exports must not drag matplotlib or vectorbt into the root."""
    code = (
        "import sys, recall_guard\n"
        "from recall_guard import EnsembleSpec, EnsembledScore, generate_ensemble\n"
        "heavy = sorted(m for m in sys.modules "
        "if m.split('.')[0] in {'matplotlib', 'vectorbt'})\n"
        "assert not heavy, heavy\n"
        "assert not any(n.startswith('plot_') for n in recall_guard.__all__)\n"
        "print('lean-ok')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "lean-ok" in result.stdout


def test_existing_public_types_are_still_constructible() -> None:
    """Additive only: nothing about the pre-existing surface may change."""
    import inspect

    from recall_guard import GuardedScore, MemoryGuardedScorer

    fields = [f.name for f in GuardedScore.__dataclass_fields__.values()]
    assert fields == [
        "prompt_hash",
        "parse_ok",
        "signal",
        "raw_confidence",
        "p_memorized",
        "memguard_confidence",
        "features",
        "fail_reason",
    ]
    score = GuardedScore("h", False, None, None, None, None, None, "error")
    assert hash(score) is not None
    assert list(inspect.signature(MemoryGuardedScorer.score).parameters) == ["self", "prompt"]
