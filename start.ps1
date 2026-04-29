#!/usr/bin/env pwsh
# Recall-guard harness runner (PowerShell mirror of start.sh).
#
# Reads NVIDIA_API_KEY / FMP_API_KEY from .env or the current shell.
# Override any variable at the call site:
#   $env:SHORTLIST = "meta/llama-3.1-8b-instruct"; .\start.ps1
#   $env:EVAL_SET  = "data/eval/etf_direction_multiyear.jsonl"; .\start.ps1
#   .\start.ps1 --no-reference          # extra harness flags pass through
#
# The default eval set is the cmmd-backtest 3-asset universe
# (SWDA.L / XLK / IAU; BIL holds residual cash in the backtest and is
# never prompted). Build the file via
# `uv run python scripts/build_etf_portfolio_eval.py` if it is missing.

$ErrorActionPreference = "Stop"

Push-Location $PSScriptRoot
try {
    $eval_set       = if ($env:EVAL_SET)          { $env:EVAL_SET }          else { "data/eval/etf_portfolio.jsonl" }
    $is_memorized   = if ($env:IS_MEMORIZED)      { $env:IS_MEMORIZED }      else { "data/calibration/is_memorized.jsonl" }
    $oos_control    = if ($env:OOS_CONTROL)       { $env:OOS_CONTROL }       else { "data/calibration/oos_control.jsonl" }
    $cutoffs        = if ($env:CUTOFFS)           { $env:CUTOFFS }           else { "data/cutoffs.yaml" }
    $shortlist      = if ($env:SHORTLIST)         { $env:SHORTLIST }         else { "meta/llama-3.1-8b-instruct,meta/llama-3.2-3b-instruct,openai/gpt-oss-20b,microsoft/phi-4-mini-instruct" }
    $max_workers    = if ($env:MAX_WORKERS)       { $env:MAX_WORKERS }       else { "8" }
    $min_call_int   = if ($env:MIN_CALL_INTERVAL) { $env:MIN_CALL_INTERVAL } else { "1.5" }
    $out_dir        = if ($env:OUT_DIR)           { $env:OUT_DIR }           else { "runs/$((Get-Date).ToUniversalTime().ToString('yyyyMMddTHHmmssZ'))" }

    & uv run python harness.py `
        --eval-set          $eval_set `
        --is-memorized      $is_memorized `
        --oos-control       $oos_control `
        --cutoffs           $cutoffs `
        --shortlist         $shortlist `
        --max-workers       $max_workers `
        --min-call-interval $min_call_int `
        --out-dir           $out_dir `
        @args
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
