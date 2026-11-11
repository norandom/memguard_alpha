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

This module re-exports the façade, the ensemble-first guarded-decision API
(``guarded_decide`` with its decision definitions, recorded-draw results and
evidence sources) and a curated set of ``core`` and ``mia`` primitives. It
deliberately re-exports no plotting symbol and nothing from the
``portfolio`` (backtest) layer, so ``import recall_guard`` never pulls in
matplotlib or vectorbt (Req 4.1, 4.3). The plotting helpers remain available, on
demand, via ``recall_guard.harness`` (lazy) and the backtest engine via
``recall_guard.portfolio.backtest`` (requires the ``backtest`` extra).
"""

from __future__ import annotations

from recall_guard.core import (
    CompletionResult,
    CostEstimate,
    CutoffViolation,
    EnsembleResult,
    EnsembleSpec,
    EvalRow,
    EvalSet,
    LocationMode,
    Manifest,
    MultimodalAction,
    MultimodalVerdict,
    NvidiaLM,
    ReferenceMode,
    Tail,
    TokenLogprob,
    assert_cutoff_safe,
    bootstrap_ci,
    canonical_draw_hash,
    compute_file_hash,
    detect_multimodal,
    estimate_cost,
    generate_ensemble,
    grid_adherence,
    lag_dependence,
    load_cutoffs,
    load_eval_set,
    read_manifest,
    reduce_draws,
    robust_location,
    scale_floor,
    smallest_certifiable_n,
    smallest_detectable_split_n,
    snap_to_grid,
    wilson_interval,
    write_manifest,
)
from recall_guard.harness.scorer import (
    ConfigurationError,
    EnsembledScore,
    GuardedScore,
    MemoryGuardedScorer,
    ScoredDraw,
)
from recall_guard.harness.workflow import RecallWorkflow, WorkflowDraw, WorkflowPreflight, WorkflowResult
from recall_guard.harness.workflow_calibration import CalibrationItem, WorkflowCalibration
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

# isort: split
# guarded decision (ensemble-first): one block after the existing imports
from recall_guard.core.decision import (
    ComponentSpec,
    DecisionDefinition,
    DecisionDefinitionError,
    DecisionParseError,
    adjust,
    parse_json_components,
)
from recall_guard.core.distribution import ComponentDistribution, DistributionSummary, RecordedDraw
from recall_guard.core.full_distribution import FullDistributionClient, FullDistributionEvidence
from recall_guard.harness.guarded_decision import (
    DrawMode,
    GuardConfig,
    GuardedDecision,
    GuardedRun,
    GuardIdentity,
    GuardStatus,
    SourceCalibrator,
    decide_from_draws,
    guarded_decide,
    guarded_run,
)
from recall_guard.mia.estimate_source import EstimateSource, SourceLabel

# isort: split
# qualification catalogue and selection: importing them reads no catalogue file
from recall_guard.core.catalogue import (
    Catalogue,
    CatalogueEntry,
    CatalogueError,
    Mode,
    ModeRecord,
    Qualification,
    Status,
    load_catalogue,
    qualify,
)
from recall_guard.harness.selection import SelectedDecision, Selection, SelectionError, guarded_select, select

__all__ = [
    # façade (the headline surface)
    "MemoryGuardedScorer",
    "GuardedScore",
    "ConfigurationError",
    "RecallWorkflow",
    "WorkflowResult",
    "WorkflowDraw",
    "WorkflowPreflight",
    "WorkflowCalibration",
    "CalibrationItem",
    "ScoredDraw",
    # guarded decision (ensemble-first)
    #   API
    "guarded_decide",
    "guarded_run",
    "decide_from_draws",
    "GuardConfig",
    "DrawMode",
    "GuardStatus",
    "GuardIdentity",
    "GuardedDecision",
    "GuardedRun",
    "SourceCalibrator",
    #   definitions
    "DecisionDefinition",
    "ComponentSpec",
    "DecisionDefinitionError",
    "DecisionParseError",
    "parse_json_components",
    "adjust",
    #   results
    "RecordedDraw",
    "DistributionSummary",
    "ComponentDistribution",
    #   sources
    "EstimateSource",
    "SourceLabel",
    "FullDistributionClient",
    "FullDistributionEvidence",
    # qualification catalogue and selection
    #   catalogue
    "Mode",
    "Status",
    "Catalogue",
    "CatalogueEntry",
    "ModeRecord",
    "Qualification",
    "CatalogueError",
    "load_catalogue",
    "qualify",
    #   selection
    "Selection",
    "SelectedDecision",
    "SelectionError",
    "select",
    "guarded_select",
    # ensemble surface (opt-in; see EnsembleSpec)
    "EnsembledScore",
    "EnsembleSpec",
    "EnsembleResult",
    "CostEstimate",
    "LocationMode",
    "MultimodalAction",
    "MultimodalVerdict",
    "ReferenceMode",
    "Tail",
    "canonical_draw_hash",
    "detect_multimodal",
    "estimate_cost",
    "generate_ensemble",
    "grid_adherence",
    "lag_dependence",
    "reduce_draws",
    "robust_location",
    "scale_floor",
    "smallest_certifiable_n",
    "smallest_detectable_split_n",
    "snap_to_grid",
    "wilson_interval",
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
