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
# Default shortlist: four models with high parse rates on NVIDIA's free tier.
# Three vendors (Meta, OpenAI, Microsoft), three cutoff dates (2023-12,
# 2024-06), sizes 3B–20B. Dropped because they parsed below 10% in the
# previous run (free-tier endpoint dropped most calls):
# nvidia/llama-3.3-nemotron-super-49b-v1.5, nvidia/nvidia-nemotron-nano-9b-v2.
# Also skipped (rate-limited, time out): llama-3.1-70b, llama-3.1-405b,
# llama-3.3-70b, gpt-oss-120b. Add any of them via SHORTLIST if you have
# paid capacity.
SHORTLIST="${SHORTLIST:-meta/llama-3.1-8b-instruct,meta/llama-3.2-3b-instruct,openai/gpt-oss-20b,microsoft/phi-4-mini-instruct}"
MAX_WORKERS="${MAX_WORKERS:-8}"
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
