"""recall_guard: measured inference-without-recall.

The public entry point. A consumer typically needs only::

    from recall_guard import MemoryGuardedScorer

    scorer = MemoryGuardedScorer.calibrate(
        api_key=...,                 # NVIDIA NIM key
        model="meta/llama-3.1-8b-instruct",
        is_memorized=[...],          # prompts dated before the model's cutoff
        oos_control=[...],           # prompts dated after it
    )
    guarded = scorer.score("<your prompt>")
    guarded.signal, guarded.p_memorized, guarded.memguard_confidence

This module re-exports the façade plus a curated set of ``core`` and ``mia``
primitives. It deliberately re-exports no plotting symbol and nothing from the
``portfolio`` (backtest) layer, so ``import recall_guard`` never pulls in
matplotlib or vectorbt (Req 4.1, 4.3). The plotting helpers remain available, on
demand, via ``recall_guard.harness`` (lazy) and the backtest engine via
``recall_guard.portfolio.backtest`` (requires the ``backtest`` extra).
"""

from __future__ import annotations

from recall_guard.core import (
    CompletionResult,
    CutoffViolation,
    EvalRow,
    EvalSet,
    Manifest,
    NvidiaLM,
    TokenLogprob,
    assert_cutoff_safe,
    bootstrap_ci,
    compute_file_hash,
    load_cutoffs,
    load_eval_set,
    read_manifest,
    write_manifest,
)
from recall_guard.harness.scorer import (
    ConfigurationError,
    GuardedScore,
    MemoryGuardedScorer,
)
from recall_guard.mia import (
    LOGPROB_FLOOR,
    ControlBaseline,
    MCSCalibrator,
    MiaFeatures,
    build_baseline,
    compute_mia_features,
    standardise,
    train_mcs,
)

__all__ = [
    # façade (the headline surface)
    "MemoryGuardedScorer",
    "GuardedScore",
    "ConfigurationError",
    # core primitives
    "NvidiaLM",
    "CompletionResult",
    "TokenLogprob",
    "EvalRow",
    "EvalSet",
    "load_eval_set",
    "load_cutoffs",
    "assert_cutoff_safe",
    "CutoffViolation",
    "bootstrap_ci",
    "Manifest",
    "write_manifest",
    "read_manifest",
    "compute_file_hash",
    # mia primitives
    "MiaFeatures",
    "compute_mia_features",
    "LOGPROB_FLOOR",
    "ControlBaseline",
    "build_baseline",
    "standardise",
    "MCSCalibrator",
    "train_mcs",
]
