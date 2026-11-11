"""The package root re-exports the guarded-decision API (task 5.2, Requirement 11.6).

An adopter calls the API with either evidence source from ``recall_guard``
alone: the guarded-decision functions and types, the decision definitions, the
recorded-draw results and the sources. The qualification catalogue and the
selection interface (qualified-model-catalogue, task 2.2) follow as one further
block, pinned the same way, and the front (task 7.1) as the block after it.
The existing root exports keep their
names and order; the lean-import test in tests/harness/test_scorer.py guards
that ``import recall_guard`` still loads no matplotlib.
"""

from __future__ import annotations

import importlib
import subprocess
import sys

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

#: The catalogue and selection block of ``recall_guard.__all__``, in order, with the defining module of each name.
CATALOGUE_SELECTION_EXPORTS = (
    # catalogue
    ("Mode", "recall_guard.core.catalogue"),
    ("Status", "recall_guard.core.catalogue"),
    ("Catalogue", "recall_guard.core.catalogue"),
    ("CatalogueEntry", "recall_guard.core.catalogue"),
    ("ModeRecord", "recall_guard.core.catalogue"),
    ("Qualification", "recall_guard.core.catalogue"),
    ("CatalogueError", "recall_guard.core.catalogue"),
    ("load_catalogue", "recall_guard.core.catalogue"),
    ("qualify", "recall_guard.core.catalogue"),
    # selection
    ("Selection", "recall_guard.harness.selection"),
    ("SelectedDecision", "recall_guard.harness.selection"),
    ("SelectionError", "recall_guard.harness.selection"),
    ("select", "recall_guard.harness.selection"),
    ("guarded_select", "recall_guard.harness.selection"),
)
CATALOGUE_SELECTION_NAMES = tuple(name for name, _module in CATALOGUE_SELECTION_EXPORTS)

#: The front block of ``recall_guard.__all__`` (qualified-model-catalogue, task 7.1), in order.
FRONT_EXPORTS = (
    ("ask", "recall_guard.harness.front"),
    ("FrontAnswer", "recall_guard.harness.front"),
    ("FrontFailure", "recall_guard.harness.front"),
    ("choose_mode", "recall_guard.harness.front"),
    # offline use and the worked example (task 7.2)
    ("FixedReplies", "recall_guard.harness.front"),
    ("FixedReply", "recall_guard.harness.front"),
    ("fixed_reply", "recall_guard.harness.front"),
    ("FixedEvidence", "recall_guard.harness.front"),
    ("worked_example", "recall_guard.harness.front"),
    ("WORKED_EXAMPLE_REPLY", "recall_guard.harness.front"),
)
FRONT_NAMES = tuple(name for name, _module in FRONT_EXPORTS)

#: The blocks added after the existing exports were pinned; every other name of ``__all__`` is an existing one.
ADDED_NAMES = GUARDED_DECISION_NAMES + CATALOGUE_SELECTION_NAMES + FRONT_NAMES

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


@pytest.mark.parametrize("name, module", CATALOGUE_SELECTION_EXPORTS)
def test_catalogue_and_selection_name_is_exported_from_the_root_as_the_module_object(name, module):
    assert name in recall_guard.__all__
    assert getattr(recall_guard, name) is getattr(importlib.import_module(module), name)


def test_catalogue_and_selection_block_follows_the_guarded_decision_block_in_declared_order():
    exported = list(recall_guard.__all__)
    start = exported.index(GUARDED_DECISION_NAMES[-1]) + 1
    assert tuple(exported[start:start + len(CATALOGUE_SELECTION_NAMES)]) == CATALOGUE_SELECTION_NAMES


@pytest.mark.parametrize("name, module", FRONT_EXPORTS)
def test_front_name_is_exported_from_the_root_as_the_module_object(name, module):
    assert name in recall_guard.__all__
    assert getattr(recall_guard, name) is getattr(importlib.import_module(module), name)


def test_front_block_follows_the_catalogue_and_selection_block_in_declared_order():
    exported = list(recall_guard.__all__)
    start = exported.index(CATALOGUE_SELECTION_NAMES[-1]) + 1
    assert tuple(exported[start:start + len(FRONT_NAMES)]) == FRONT_NAMES


def test_importing_the_root_with_the_front_loads_no_plotting_library_and_reads_no_data_file():
    code = (
        "import builtins, pathlib, sys\n"
        "opened = []\n"
        "real_open, real_read = builtins.open, pathlib.Path.read_text\n"
        "def spy_open(file, *args, **kwargs):\n"
        "    opened.append(str(file))\n"
        "    return real_open(file, *args, **kwargs)\n"
        "def spy_read(self, *args, **kwargs):\n"
        "    opened.append(str(self))\n"
        "    return real_read(self, *args, **kwargs)\n"
        "builtins.open, pathlib.Path.read_text = spy_open, spy_read\n"
        "import recall_guard\n"
        "heavy = sorted(m for m in sys.modules if m.split('.')[0] in {'matplotlib', 'vectorbt'})\n"
        "assert not heavy, heavy\n"
        # The files the front reads when called: the catalogue and a provider profile.
        "data = [name for name in opened if 'catalogue' in name.lower() or 'provider_profiles' in name.lower()]\n"
        "assert not data, data\n"
        "assert recall_guard.ask.__module__ == 'recall_guard.harness.front'\n"
        "print('lean-ok')\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
    assert "lean-ok" in done.stdout


def test_added_blocks_share_no_name_with_each_other_or_the_existing_exports():
    assert len(ADDED_NAMES) == len(set(ADDED_NAMES))
    assert set(ADDED_NAMES).isdisjoint(EXISTING_EXPORTS)


def test_existing_root_exports_and_their_order_are_unchanged():
    exported = list(recall_guard.__all__)
    assert len(exported) == len(set(exported))
    assert tuple(name for name in exported if name not in ADDED_NAMES) == EXISTING_EXPORTS
    assert set(exported) == set(EXISTING_EXPORTS) | set(ADDED_NAMES)


def test_every_root_export_resolves():
    assert [name for name in recall_guard.__all__ if not hasattr(recall_guard, name)] == []
