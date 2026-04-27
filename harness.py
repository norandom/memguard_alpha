"""Top-level CLI entry point for the honest-model-ranking harness.

Replaces the legacy top-level entry point for evaluation runs. Delegates parsing and
orchestration to :mod:`src.harness.runner`; this script exists only so the
user can invoke ``python harness.py ...`` from the project root.

The harness exposes two subcommands (see design.md → harness.runner →
CLI Surface):

* ``harness build [...]`` — run the full evaluation pipeline. ``build`` is
  the default subcommand: invocations like
  ``python harness.py --eval-set X --shortlist Y`` (no explicit subcommand)
  are rewritten to ``python harness.py build --eval-set X --shortlist Y``
  by :func:`src.harness.runner.parse_argv` for backwards compatibility.
* ``harness replay --from-manifest PATH --out-dir PATH`` — reproduce a
  prior run from its persisted manifest (Req 10.2). Verifies that every
  input file's sha256 matches the manifest before re-running; aborts
  non-zero on mismatch.
"""

from __future__ import annotations

import sys
from pathlib import Path

from src.harness.runner import parse_argv, replay, run


def main(argv: list[str] | None = None) -> int:
    """Parse argv, dispatch to the chosen subcommand, return its exit code.

    ``argv`` defaults to :data:`sys.argv[1:]` when ``None`` so callers can
    inject synthetic argv for testing.
    """
    args = parse_argv(list(sys.argv[1:] if argv is None else argv))

    subcommand = getattr(args, "subcommand", None) or "build"
    if subcommand == "replay":
        return replay(
            manifest_path=Path(args.from_manifest),
            out_dir=Path(args.out_dir),
        )
    # Default / explicit "build".
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
