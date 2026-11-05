"""Validate a frozen local measurement plan without making inference calls.

PowerShell: uv run --no-sync python scripts/measure_local_models.py --help
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recall_guard.harness.local_evaluation import load_plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--corpus', required=True, type=Path)
    parser.add_argument('--dry-run', required=True, action='store_true')
    args = parser.parse_args(argv)
    try:
        plan = load_plan(args.plan, args.manifest, args.corpus)
    except (ValueError, TypeError, KeyError, OSError):
        # Parsing failures may contain private paths or corpus material.
        print('Local measurement refused: invalid plan, manifest or corpus.', file=sys.stderr)
        return 2
    print(json.dumps({'status': 'dry_run', 'plan': plan.safe_export()}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
