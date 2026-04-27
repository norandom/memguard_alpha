"""Top-level CLI entry point for the honest-model-ranking harness.

Replaces the legacy ``main.py`` for evaluation runs. Delegates parsing and
orchestration to :mod:`src.harness.runner`; this script exists only so the
user can invoke ``python harness.py ...`` from the project root.

Replay mode (``harness replay --from-manifest ...``) is intentionally NOT
implemented here — it is owned by Task 5.2.
"""

from __future__ import annotations

import sys

from src.harness.runner import build_parser, run


def main() -> int:
    """Parse argv, dispatch to ``runner.run``, return its exit code."""
    parser = build_parser()
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
