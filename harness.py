"""Top-level CLI entry point for the honest-model-ranking harness.

Single mode: build. Run a full evaluation pipeline (load → smoke shortlist →
control baselines → MCS train → evaluate → rank → top-3) and write artifacts
to the chosen --out-dir.

Parsing and orchestration live in :mod:`recall_guard.harness.runner`; this script just
forwards argv.
"""

from __future__ import annotations

import logging
import sys

from recall_guard.harness.runner import parse_argv, run


def main(argv: list[str] | None = None) -> int:
    """Parse argv and run the build pipeline. Returns the exit code."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_argv(list(sys.argv[1:] if argv is None else argv))
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
