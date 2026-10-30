#!/usr/bin/env bash
# Run the honest-model-ranking harness.
#
# Reads NVIDIA_API_KEY from .env or the shell. Override any of the variables
# below at the call site, e.g.:
#   SHORTLIST="meta/llama-3.1-8b-instruct" ./start.sh
#   EVAL_SET=data/eval/etf_direction_multiyear.jsonl ./start.sh
#
# The default eval set is the cmmd-backtest 3-asset universe
# (SWDA.L, XLK, IAU; BIL is the cash leg in the backtest and is not
# prompted) so the harness numbers and the cmmd-backtest numbers come
# from the same prompt stream. Build the file via
# `uv run python scripts/build_etf_portfolio_eval.py` if it is missing.

set -euo pipefail

cd "$(dirname "$0")"

EVAL_SET="${EVAL_SET:-data/eval/etf_portfolio.jsonl}"
IS_MEMORIZED="${IS_MEMORIZED:-data/calibration/is_memorized.jsonl}"
OOS_CONTROL="${OOS_CONTROL:-data/calibration/oos_control.jsonl}"
CUTOFFS="${CUTOFFS:-data/cutoffs.yaml}"
# Current hosted default, verified against the NIM catalog on 2026-10-30.
# Historical cutoff entries remain available for interpreting old runs.
SHORTLIST="${SHORTLIST:-openai/gpt-oss-20b}"
MAX_WORKERS="${MAX_WORKERS:-1}"
MIN_CALL_INTERVAL="${MIN_CALL_INTERVAL:-1.5}"
OUT_DIR="${OUT_DIR:-runs/$(date -u +%Y%m%dT%H%M%SZ)}"

exec uv run python harness.py \
  --eval-set        "$EVAL_SET" \
  --is-memorized    "$IS_MEMORIZED" \
  --oos-control     "$OOS_CONTROL" \
  --cutoffs         "$CUTOFFS" \
  --shortlist       "$SHORTLIST" \
  --max-workers     "$MAX_WORKERS" \
  --min-call-interval "$MIN_CALL_INTERVAL" \
  --out-dir         "$OUT_DIR" \
  "$@"
