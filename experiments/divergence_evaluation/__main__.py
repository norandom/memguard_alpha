"""Credential-free planning: python -m experiments.divergence_evaluation plan."""

import argparse
import json
from pathlib import Path

from .planning import plan_locally


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan_parser = commands.add_parser("plan", help="inspect a local plan without runtime access")
    plan_parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = plan_locally(args.config)
    except (ValueError, OSError):
        parser.exit(2, "Invalid local plan configuration; no runtime access performed.\n")
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
