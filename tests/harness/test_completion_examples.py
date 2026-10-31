"""Execute the adopter examples offline, including a real current-Python kernel."""

import http.client
import json
import re
import runpy
import subprocess
import sys
from pathlib import Path

import nbformat
import pytest
import requests
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient

from recall_guard.core.completion import FeatureChoices
from recall_guard.core.ensemble import EnsembleSpec
from recall_guard.core.request_control import CallBudget

ROOT = Path(__file__).resolve().parents[2]
APPLICATION = ROOT / "examples" / "completion_workflow.py"
NOTEBOOK = ROOT / "notebooks" / "completion_workflow.ipynb"
HTTP_GUARD = """
import http.client
import requests

def block_model_http(*args, **kwargs):
    raise AssertionError("Outbound HTTP is forbidden in the completion fixture")

requests.sessions.Session.request = block_model_http
http.client.HTTPConnection.connect = block_model_http
"""


def select_documented_features(source: str, draws: int, rewrite: bool) -> str:
    """Apply the documented edits to a temporary application or notebook cell."""
    source, changed = re.subn(
        r"features = FeatureChoices\(mia_scoring=False, ensemble_draws=\d+, rewriting=(?:True|False)\)",
        f"features = FeatureChoices(mia_scoring=False, ensemble_draws={draws}, rewriting={rewrite})",
        source, count=1,
    )
    assert changed == 1, "The example's explicit feature selection was not found"
    if draws == 1:
        source, changed = re.subn(r"ensemble = EnsembleSpec\([^)]*\)", "ensemble = None", source, count=1)
        assert changed == 1, "The documented single-answer ensemble choice was not applied"
    return source


@pytest.fixture
def example(monkeypatch):
    def block_model_http(*args, **kwargs):
        raise AssertionError("Outbound HTTP is forbidden in the completion fixture")

    monkeypatch.setattr(requests.sessions.Session, "request", block_model_http)
    monkeypatch.setattr(http.client.HTTPConnection, "connect", block_model_http)
    return runpy.run_path(str(APPLICATION))


@pytest.mark.parametrize("draws", [1, 3])
@pytest.mark.parametrize("rewrite", [False, True])
def test_example_uses_selected_features_and_keeps_missing_mcs(example, draws, rewrite):
    config = example["fixture_config"]()
    features = FeatureChoices(ensemble_draws=draws, rewriting=rewrite)
    budget = CallBudget(8)
    workflow = example["make_fixture_workflow"](config, features, budget)
    ensemble = EnsembleSpec(draws=draws, min_parsed=draws, max_workers=1) if draws > 1 else None
    preflight = workflow.preflight(rewrite=rewrite, ensemble=ensemble)
    assert preflight.ready_for_batch
    result = workflow.run(example["fixture_item"](), rewrite=rewrite, ensemble=ensemble)
    assert result.status == "completed"
    assert result.n_requested == result.n_attempted == result.n_usable == draws
    assert result.p_memorized is None and result.mia_features is None
    assert result.response.token_evidence is None and result.response.mia_unavailable_reason
    assert result.unavailable_reasons["p_memorized"]
    assert result.accepted_evidence == (example["ACCEPTED_EVIDENCE"] if rewrite else example["SOURCE_EVIDENCE"])
    assert budget.used == draws + 1 + (4 if rewrite else 0)
    assert {event.stage for event in budget.events} == ({"probe", "rewrite", "verify", "target"} if rewrite
                                                      else {"probe", "target"})


def test_rejected_fixture_does_not_fall_back_or_admit_target_draws(example):
    budget = CallBudget(4)
    workflow = example["make_fixture_workflow"](
        example["fixture_config"](), FeatureChoices(ensemble_draws=3, rewriting=True), budget,
        rewritten_evidence=example["REJECTED_EVIDENCE"],
    )
    result = workflow.run(example["fixture_item"](), rewrite=True,
                          ensemble=EnsembleSpec(draws=3, min_parsed=3, max_workers=1))
    assert result.status == "failed" and result.stage == "deterministic"
    assert result.response is None and result.accepted_evidence is None
    assert result.n_attempted == result.n_usable == 0
    assert result.rewrite.error
    assert budget.used == 4
    assert [event.stage for event in budget.events] == ["probe", "probe", "probe", "rewrite"]


@pytest.mark.parametrize("draws", [1, 3])
@pytest.mark.parametrize("rewrite", [False, True])
def test_application_runs_offline_and_separates_configuration_from_runtime_errors(tmp_path, draws, rewrite):
    application = tmp_path / APPLICATION.name
    source = select_documented_features(APPLICATION.read_text(encoding="utf-8"), draws, rewrite)
    application.write_text(source, encoding="utf-8")
    command = HTTP_GUARD + f"\nimport runpy\nrunpy.run_path({str(application)!r}, run_name='__main__')\n"
    completed = subprocess.run([sys.executable, "-c", command], cwd=ROOT, text=True,
                               encoding="utf-8", capture_output=True, timeout=60, check=True)
    summaries = [json.loads(line) for line in completed.stdout.splitlines()]
    rows = {summary["example"]: summary for summary in summaries}
    assert rows["configuration"]["draws"] == draws
    assert rows["configuration"]["rewriting"] == rewrite
    assert rows["accepted_fixture"]["n_requested"] == rows["accepted_fixture"]["n_usable"] == draws
    assert rows["accepted_fixture"]["p_memorized"] is None
    assert rows["accepted_fixture"]["token_evidence_unavailable_reason"]
    assert rows["fixture_budget"]["admitted_operations"] == draws + (9 if rewrite else 1)
    if rewrite:
        assert rows["rejected_fixture"]["n_attempted"] == 0
        assert rows["rejected_fixture"]["accepted_evidence"] is None
        assert rows["rejected_fixture"]["answer"] is None
    else:
        assert rows["accepted_fixture"]["accepted_evidence"] == "Acme rose 2% on 2024-01-02; estimates remain uncertain."
        assert "rejected_fixture" not in rows
    assert "temperature" in rows["native_parameter_error"]["error"]
    assert "MIA/MCS" in rows["native_scoring_error"]["error"]
    assert rows["callback_runtime_error"]["code"] == "callback_runtime_error"
    assert rows["callback_runtime_error"]["parameter"] is None


@pytest.mark.parametrize("draws", [1, 3])
@pytest.mark.parametrize("rewrite", [False, True])
def test_notebook_executes_from_its_directory_with_the_current_python(tmp_path, draws, rewrite):
    kernel_name = "completion-example"
    kernel_directory = tmp_path / "kernels" / kernel_name
    kernel_directory.mkdir(parents=True)
    (kernel_directory / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Completion example test", "language": "python",
    }), encoding="utf-8")
    specs = KernelSpecManager(kernel_dirs=[str(kernel_directory.parent)])
    manager = KernelManager(kernel_name=kernel_name, kernel_spec_manager=specs)
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    selection_cells = [cell for cell in notebook.cells
                       if cell.cell_type == "code" and "features = FeatureChoices(" in cell.source]
    assert len(selection_cells) == 1
    selection_cells[0].source = select_documented_features(selection_cells[0].source, draws, rewrite)
    notebook.cells.insert(0, nbformat.v4.new_code_cell(HTTP_GUARD + "\nimport sys\nfrom pathlib import Path\n"
        + f"assert Path(sys.executable).resolve() == Path({sys.executable!r}).resolve()\n"
        + f"assert Path.cwd().resolve() == Path({str(NOTEBOOK.parent)!r}).resolve()\n"))
    notebook.cells.append(nbformat.v4.new_code_cell("""
assert accepted.status == "completed" and accepted.n_requested == accepted.n_usable == features.ensemble_draws
assert accepted.p_memorized is None and accepted.unavailable_reasons["p_memorized"]
assert accepted.response.token_evidence is None
if features.rewriting:
    assert accepted.rewrite_scope == "shared_per_item"
    assert rejected.status == "failed" and rejected.n_attempted == 0
    assert rejected.accepted_evidence is None and rejected.response is None
else:
    assert accepted.rewrite_scope is None and accepted.accepted_evidence == SOURCE_EVIDENCE
    assert rejected is None
    assert all(event.model == config.model for event in budget.events)
    assert {event.stage for event in budget.events} == {"probe", "target"}
assert budget.used == features.ensemble_draws + (9 if features.rewriting else 1)
"""))
    client = NotebookClient(notebook, km=manager, timeout=60, kernel_name=kernel_name,
                            resources={"metadata": {"path": str(NOTEBOOK.parent)}})
    try:
        client.execute(cleanup_kc=True)
    finally:
        # Supplying km makes this test its owner, including failed startup paths.
        try:
            if client.kc is not None:
                client.kc.stop_channels()
        finally:
            try:
                if manager.is_alive():
                    manager.shutdown_kernel(now=True)
            finally:
                if client.km is not None:
                    manager.cleanup_resources()
        assert not manager.is_alive(), "The test-owned notebook kernel is still running"
