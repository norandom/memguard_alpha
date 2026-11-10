"""The package root re-exports the guarded-decision API (task 5.2, Requirement 11.6).

An adopter calls the API with either evidence source from ``recall_guard``
alone: the guarded-decision functions and types, the decision definitions, the
recorded-draw results and the sources. The existing root exports keep their
names and order; the lean-import test in tests/harness/test_scorer.py guards
that ``import recall_guard`` still loads no matplotlib.
"""

from __future__ import annotations

import importlib

import pytest

import recall_guard

#: The guarded-decision block of ``recall_guard.__all__``, in order, with the defining module of each name.
GUARDED_DECISION_EXPORTS = (
    # API
    ("guarded_decide", "recall_guard.harness.guarded_decision"),
    ("guarded_run", "recall_guard.harness.guarded_decision"),
    ("decide_from_draws", "recall_guard.harness.guarded_decision"),
    ("GuardConfig", "recall_guard.harness.guarded_decision"),
    ("DrawMode", "recall_guard.harness.guarded_decision"),
    ("GuardStatus", "recall_guard.harness.guarded_decision"),
    ("GuardIdentity", "recall_guard.harness.guarded_decision"),
    ("GuardedDecision", "recall_guard.harness.guarded_decision"),
    ("GuardedRun", "recall_guard.harness.guarded_decision"),
    ("SourceCalibrator", "recall_guard.harness.guarded_decision"),
    # definitions
    ("DecisionDefinition", "recall_guard.core.decision"),
    ("ComponentSpec", "recall_guard.core.decision"),
    ("DecisionDefinitionError", "recall_guard.core.decision"),
    ("DecisionParseError", "recall_guard.core.decision"),
    ("parse_json_components", "recall_guard.core.decision"),
    ("adjust", "recall_guard.core.decision"),
    # results
    ("RecordedDraw", "recall_guard.core.distribution"),
    ("DistributionSummary", "recall_guard.core.distribution"),
    ("ComponentDistribution", "recall_guard.core.distribution"),
    # sources
    ("EstimateSource", "recall_guard.mia.estimate_source"),
    ("SourceLabel", "recall_guard.mia.estimate_source"),
    ("FullDistributionClient", "recall_guard.core.full_distribution"),
    ("FullDistributionEvidence", "recall_guard.core.full_distribution"),
)
GUARDED_DECISION_NAMES = tuple(name for name, _module in GUARDED_DECISION_EXPORTS)

#: ``recall_guard.__all__`` before task 5.2; these names and their order must not change.
EXISTING_EXPORTS = (
    "MemoryGuardedScorer", "GuardedScore", "ConfigurationError", "RecallWorkflow", "WorkflowResult",
    "WorkflowDraw", "WorkflowPreflight", "WorkflowCalibration", "CalibrationItem", "ScoredDraw",
    "EnsembledScore", "EnsembleSpec", "EnsembleResult", "CostEstimate", "LocationMode", "MultimodalAction",
    "MultimodalVerdict", "ReferenceMode", "Tail", "canonical_draw_hash", "detect_multimodal", "estimate_cost",
    "generate_ensemble", "grid_adherence", "lag_dependence", "reduce_draws", "robust_location", "scale_floor",
    "smallest_certifiable_n", "smallest_detectable_split_n", "snap_to_grid", "wilson_interval",
    "NvidiaLM", "CompletionResult", "TokenLogprob", "EvalRow", "EvalSet", "load_eval_set", "load_cutoffs",
    "assert_cutoff_safe", "CutoffViolation", "bootstrap_ci", "Manifest", "write_manifest", "read_manifest",
    "compute_file_hash",
    "MiaFeatures", "compute_mia_features", "LOGPROB_FLOOR", "ControlBaseline", "build_baseline", "standardise",
    "MCSCalibrator", "train_mcs",
)
#: The last name of the existing façade block; the guarded-decision block follows it.
LAST_FACADE_EXPORT = "ScoredDraw"


@pytest.mark.parametrize("name, module", GUARDED_DECISION_EXPORTS)
def test_guarded_decision_name_is_exported_from_the_root_as_the_module_object(name, module):
    assert name in recall_guard.__all__
    assert getattr(recall_guard, name) is getattr(importlib.import_module(module), name)


def test_guarded_decision_block_follows_the_facade_block_in_declared_order():
    exported = list(recall_guard.__all__)
    start = exported.index(LAST_FACADE_EXPORT) + 1
    assert tuple(exported[start:start + len(GUARDED_DECISION_NAMES)]) == GUARDED_DECISION_NAMES


def test_existing_root_exports_and_their_order_are_unchanged():
    exported = list(recall_guard.__all__)
    assert len(exported) == len(set(exported))
    assert tuple(name for name in exported if name not in GUARDED_DECISION_NAMES) == EXISTING_EXPORTS
    assert set(exported) == set(EXISTING_EXPORTS) | set(GUARDED_DECISION_NAMES)


def test_every_root_export_resolves():
    assert [name for name in recall_guard.__all__ if not hasattr(recall_guard, name)] == []
