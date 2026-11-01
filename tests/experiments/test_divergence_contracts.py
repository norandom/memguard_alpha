"""Offline contract and planning boundaries: no runtime or credential access."""

import json
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace

import pytest

from experiments.divergence_evaluation.contracts import (
    Arm,
    AuxiliaryManifest,
    Cohort,
    CorpusDocument,
    EvaluationPlan,
    FittingLimits,
    GenerationSettings,
    ItemResult,
    OperationRecord,
    OperationStatus,
    Phase,
    PhaseBudget,
    Split,
)
from experiments.divergence_evaluation.planning import plan_locally


def test_offline_contract_round_trips():
    contracts = [
        EvaluationPlan(),
        EvaluationPlan(
            fitting_limits=FittingLimits(2, 20, 1024),
            phase_budgets=(PhaseBudget(Phase.EVALUATION, 32, 32768, 3600, 1000, 1000, 1),),
            execution_phase=Phase.EVALUATION,
        ),
        CorpusDocument("doc", "source", "a" * 64, "local-rights", "2019-01-01T00:00:00Z",
                       "2019-01-02T00:00:00Z", Cohort.HISTORICAL, Split.FITTING, ("fact",)),
        AuxiliaryManifest("older", "b" * 64, ("a" * 64,), (("source", "older", 1, 8),), FittingLimits(2, 20, 1024)),
        OperationRecord("op", Phase.SCREENING, "item", Arm.BASELINE, "2026-11-01T00:00:00Z",
                        OperationStatus.INTENT),
        ItemResult("item", Cohort.HISTORICAL, Arm.ADJUSTED, None, "absent", None, 0, "length"),
    ]
    for contract in contracts:
        assert type(contract).from_json(contract.to_json()) == contract
        assert json.loads(contract.to_json())["schema_version"] == 1
    with pytest.raises(FrozenInstanceError):
        contracts[0].main_model_id = "substitute"


@pytest.mark.parametrize("changes", [
    {"main_model_id": "another/model"}, {"main_revision": "main"},
    {"schema_version": 2}, {"execution_phase": "invented"},
    {"alpha_grid": (float("nan"),)}, {"alpha_grid": (0.0,)},
    {"image_digest": "latest"}, {"tokenizer_mapping_hash": "bad"},
])
def test_invalid_plan_fails(changes):
    with pytest.raises(ValueError):
        EvaluationPlan(**changes)


@pytest.mark.parametrize("context,output", [(0, 1), (8, 9), (8, 0), (True, 1), (8, 1.5)])
def test_generation_limits_are_integer_tokens(context, output):
    with pytest.raises(ValueError):
        GenerationSettings(context, output)


@pytest.mark.parametrize("changes", [
    {"max_operations": -1}, {"max_generated_tokens": True},
    {"wall_time_seconds": float("inf")}, {"gpu_hours": -1},
    {"monetary_cap_usd": 1}, {"phase": "unknown"},
])
def test_budget_limits_and_paid_rate_are_validated(changes):
    settings = dict(phase=Phase.EVALUATION, max_operations=32, max_generated_tokens=32768,
                    wall_time_seconds=3600, peak_host_memory_bytes=1000,
                    peak_device_memory_bytes=1000, gpu_hours=1)
    settings.update(changes)
    with pytest.raises(ValueError):
        PhaseBudget(**settings)


def test_phase_budgets_cannot_duplicate_or_hide_output_shortfall():
    budget = PhaseBudget(Phase.EVALUATION, 32, 100, 60, 1000, 1000, 1)
    with pytest.raises(ValueError, match="duplicate"):
        EvaluationPlan(phase_budgets=(budget, budget))
    with pytest.raises(ValueError, match="generated"):
        EvaluationPlan(phase_budgets=(budget,))


def test_planning_retains_unknowns_and_requires_every_phase_budget(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(EvaluationPlan().to_json(), encoding="utf-8")
    report = plan_locally(path)
    assert report["ready_for_execution"] is False
    assert "main_revision" in report["unresolved_prerequisites"]
    assert "phase_budgets.evaluation" in report["unresolved_prerequisites"]
    assert report["monetary_cost_usd"] is None
    assert report["mcs_unavailable_reason"]
    assert "approval" in report["unresolved_prerequisites"]


def test_unknown_fields_and_nested_bad_values_are_not_silently_ignored():
    payload = json.loads(EvaluationPlan().to_json())
    payload["api_key"] = "must-not-be-echoed"
    with pytest.raises(ValueError, match="unknown fields") as error:
        EvaluationPlan.from_json(json.dumps(payload))
    assert "must-not-be-echoed" not in str(error.value)
    payload.pop("api_key")
    payload["generation"]["max_output_tokens"] = -1
    with pytest.raises(ValueError):
        EvaluationPlan.from_json(json.dumps(payload))


def test_operation_and_item_unknown_evidence_is_not_zero():
    operation = OperationRecord("op", Phase.EVALUATION, "item", Arm.BASELINE,
                                "2026-11-01T00:00:00Z", OperationStatus.INCOMPLETE)
    assert operation.monetary_cost_usd is None
    assert operation.generated_tokens is None
    item = ItemResult("item", Cohort.HISTORICAL, Arm.BASELINE, None, "absent", None, 0, "incomplete")
    assert item.mcs is None
    assert item.mcs_unavailable_reason
    with pytest.raises(ValueError):
        replace(operation, elapsed_seconds=-1)


def test_cli_never_imports_inference_or_reads_credentials(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text(EvaluationPlan().to_json(), encoding="utf-8")
    script = '''
import builtins, os, runpy, socket, sys
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'transformers', 'requests', 'dotenv', 'huggingface_hub'}:
        raise AssertionError('runtime import attempted')
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
class NoEnvironment(dict):
    def __getitem__(self, key):
        if key in {'COLUMNS', 'LINES'}:
            raise KeyError(key)
        raise AssertionError('credential environment read')
    def get(self, key, default=None):
        if key in {'LANGUAGE', 'LC_ALL', 'LC_MESSAGES', 'LANG'}:
            return default
        raise AssertionError('credential environment read')
os.environ = NoEnvironment()
def no_network(*args, **kwargs): raise AssertionError('network access')
socket.socket = no_network
sys.argv = ['divergence_evaluation', 'plan', '--config', sys.argv[1]]
runpy.run_module('experiments.divergence_evaluation', run_name='__main__')
'''
    process = subprocess.run([sys.executable, "-c", script, str(path)], capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    assert json.loads(process.stdout)["ready_for_execution"] is False


def test_cli_invalid_target_fails_without_echoing_config(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text('{"main_model_id": "unapproved-model", "api_key": "never-echo-this"}', encoding="utf-8")
    process = subprocess.run(
        [sys.executable, "-m", "experiments.divergence_evaluation", "plan", "--config", str(path)],
        capture_output=True, text=True,
    )
    assert process.returncode == 2
    assert "no runtime access" in process.stderr
    assert "never-echo-this" not in process.stderr + process.stdout
