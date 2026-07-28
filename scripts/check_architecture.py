"""Structural architecture gate for CI (review-hardening Req 6.1).

Enforces the import-graph rules from ``.sentrux/rules.toml`` with nothing
but the stdlib, so the check can run inside the Dagger lint container
where the sentrux MCP plugin is unavailable:

- Layer order (``core=2 <- {dataset, mia, portfolio}=1 <- harness=0``):
  a module may import only layers with a strictly greater order (further
  down the stack) or its own layer. Equal-order siblings are therefore
  forbidden in both directions, matching the explicit ``[[boundaries]]``
  pairs in the rules file.
- Banned dependency: ``dspy`` must not be imported anywhere under
  ``recall_guard`` (legacy-pipeline guard).

Complexity/length/cycle ceilings from the rules file remain sentrux's
job; this gate covers the dependency-direction rules that CI must not
let regress.

Exit status: 0 when clean, 1 with one violation per line on stderr.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

#: Mirror of .sentrux/rules.toml [[layers]] — name -> order.
LAYERS: dict[str, int] = {
    "harness": 0,
    "dataset": 1,
    "mia": 1,
    "portfolio": 1,
    "core": 2,
}

#: Mirror of the legacy [[boundaries]] guard: banned import roots.
BANNED_ROOTS: frozenset[str] = frozenset({"dspy"})


def _imported_names(tree: ast.AST) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.append(node.module)
    return names


def find_violations(package_root: Path) -> list[str]:
    """Scan every module under ``package_root`` and return violations."""
    violations: list[str] = []
    for py in sorted(package_root.rglob("*.py")):
        rel = py.relative_to(package_root)
        src_layer = rel.parts[0] if len(rel.parts) > 1 else None
        src_order = LAYERS.get(src_layer) if src_layer else None

        tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        for name in _imported_names(tree):
            root = name.split(".")[0]
            if root in BANNED_ROOTS:
                violations.append(f"{rel}: banned import {name!r}")
                continue
            if root != package_root.name or src_order is None:
                # Third-party/stdlib imports and package-root modules
                # (the curated public API may re-export from any layer).
                continue
            parts = name.split(".")
            dst_layer = parts[1] if len(parts) > 1 else None
            dst_order = LAYERS.get(dst_layer) if dst_layer else None
            if dst_order is None or dst_layer == src_layer:
                continue
            if dst_order <= src_order:
                violations.append(
                    f"{rel}: layer {src_layer!r} (order {src_order}) must not "
                    f"import {name!r} (layer {dst_layer!r}, order {dst_order})"
                )
    return violations


def main() -> int:
    package_root = Path(__file__).resolve().parent.parent / "recall_guard"
    violations = find_violations(package_root)
    for line in violations:
        print(line, file=sys.stderr)
    if violations:
        print(f"architecture check FAILED: {len(violations)} violation(s)", file=sys.stderr)
        return 1
    print("architecture check OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
