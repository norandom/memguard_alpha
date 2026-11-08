"""Build the SEC EDGAR 8-K EX-99.1 calibration and held-out corpora (release-1.0 M3.3, D1).

Writes, under ``--out-dir``::

    calibration/is_memorized.jsonl   (40, label 1)
    calibration/oos_control.jsonl    (100, label 0)
    held_out/is_memorized.jsonl      (40, label 1)
    held_out/oos_control.jsonl       (40, label 0)
    manifest.json                    (counts per quarter, seed, cutoffs, SHA-256, requests)

Requires ``SEC_USER_AGENT`` ("Name contact@domain") except for ``--dry-run``,
which prints the plan and makes no request. In the dev container::

    docker compose run --rm -e SEC_USER_AGENT dev uv run --no-sync \\
        python scripts/build_edgar_corpora.py --out-dir /outputs/edgar

See ``recall_guard/dataset/edgar_corpora.py`` for discovery, sampling and
fair-access rules. Exit codes: 0 complete, 1 configuration error, 2 short of a
target (request cap or too few candidates; rerun with the cache to resume).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recall_guard.core.loader import load_cutoffs  # noqa: E402
from recall_guard.dataset import edgar_corpora as ec  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start", type=date.fromisoformat, default=ec.DEFAULT_START, help="YYYY-MM-DD")
    parser.add_argument("--today", type=date.fromisoformat, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--cache-dir", type=Path, default=None, help="default: /outputs/edgar-cache in the container")
    parser.add_argument("--no-cache", action="store_true", help="disable the on-disk cache")
    parser.add_argument("--max-requests", type=int, default=ec.DEFAULT_MAX_REQUESTS)
    parser.add_argument("--models", nargs="+", default=list(ec.DEFAULT_MODELS))
    parser.add_argument("--cutoffs", type=Path, default=Path("data/cutoffs.yaml"))
    parser.add_argument("--dry-run", action="store_true", help="print the plan; no network")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, *, session_factory=ec.make_session) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    today = args.today or date.today()
    try:
        cutoffs = ec.select_cutoffs(load_cutoffs(args.cutoffs), args.models)
        the_plan = ec.plan(
            cutoffs, start=args.start, today=today, targets=ec.Targets(), seed=args.seed,
            max_requests=args.max_requests,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    cache_dir = None if args.no_cache else (args.cache_dir or ec.default_cache_dir())
    if args.dry_run:
        the_plan["out_dir"] = str(args.out_dir)
        the_plan["cache_dir"] = str(cache_dir) if cache_dir else None
        the_plan[f"{ec.USER_AGENT_ENV}_set"] = bool(os.environ.get(ec.USER_AGENT_ENV, "").strip())
        print(json.dumps(the_plan, indent=2))
        return 0
    try:
        user_agent = ec.require_user_agent()
    except ec.EdgarConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    client = ec.EdgarClient(session=session_factory(user_agent), max_requests=args.max_requests, cache_dir=cache_dir)
    result = ec.build_edgar_corpora(
        args.out_dir, cutoffs, client, seed=args.seed, start=args.start, today=today,
    )
    for rel, entry in result.manifest["files"].items():
        print(f"{rel}: {entry['rows']}/{entry['target']} rows")
    print(f"requests used: {client.requests_used} of {args.max_requests} (cache hits {client.cache_hits})")
    print(f"manifest: {result.paths['manifest']}")
    return 0 if result.complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
