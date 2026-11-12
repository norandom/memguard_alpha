"""The usage notebook of the two evidence modes runs offline, and its committed outputs are current.

``docs/notebooks/recall_adjusted_answers.ipynb`` asks the worked example in the
logprobs mode and in the logits mode on the invented offline setup
(``recall_guard.offline_example``). The documentation shows the notebook with
its stored outputs, so a fresh run must print what is stored.
"""

import json
import sys
from pathlib import Path

import nbformat
import pytest
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient

from recall_guard import Mode, OfflineExample, Status, ask, front, load_catalogue, offline_example, worked_example
from tests.integration.test_pit_front_demo import HTTP_GUARD, code_cells, imported_names, submodule_imports

ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "docs" / "notebooks" / "recall_adjusted_answers.ipynb"


def is_online(cell) -> bool:
    """A cell of the online part: it sends requests to a running model only when a person turns its group on."""
    return "online" in cell.metadata.get("tags", [])


def printed(cell) -> str:
    return "".join(output.text for output in cell.outputs
                   if output.output_type == "stream" and output.name == "stdout")


def stdout(notebook) -> list[str]:
    """What each offline code cell printed, in order."""
    return [printed(cell) for cell in code_cells(notebook) if not is_online(cell)]


@pytest.fixture(scope="module")
def answers():
    prompt, definition = worked_example()
    example = offline_example()
    return example, {mode: ask(prompt, definition, **example.arguments(mode)) for mode in Mode}


@pytest.mark.usefixtures("offline")
def test_the_offline_example_gives_an_estimate_and_a_corrected_answer_in_both_modes():
    prompt, definition = worked_example()
    example = offline_example()
    assert isinstance(example, OfflineExample)
    for mode in Mode:
        result = ask(prompt, definition, **example.arguments(mode))
        assert result.failure is None and result.limits == ()
        assert (result.entry_id, result.mode, result.status) == (example.entry, mode, Status.QUALIFIED)
        assert 0.0 < result.p_memorized < 1.0
        for name, value in result.answer.items():
            assert result.corrected[name] == pytest.approx(value * (1.0 - result.p_memorized), abs=1e-12)


def test_the_two_modes_give_the_same_answer_and_different_estimates(answers):
    _example, results = answers
    assert results[Mode.LOGPROBS].answer == results[Mode.LOGITS].answer
    assert results[Mode.LOGPROBS].p_memorized != results[Mode.LOGITS].p_memorized


def test_the_offline_example_is_invented_and_names_no_shipped_entry(answers):
    example, _results = answers
    assert example.entry not in {entry.entry_id for entry in load_catalogue().entries}
    assert [entry.entry_id for entry in example.catalogue.entries] == [example.entry]
    assert set(example.calibrators) == set(Mode)
    # Each call gets its own fixed client and host, so one run does not use up another's replies.
    assert example.arguments(Mode.LOGPROBS)["client"] is not example.arguments(Mode.LOGPROBS)["client"]


def test_the_notebook_imports_only_the_package_root_and_says_that_its_setup_is_invented():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    source = "\n".join(cell.source for cell in code_cells(notebook) if not is_online(cell))
    assert submodule_imports(source) == []
    assert {"ask", "offline_example", "worked_example"} <= set(imported_names(source))
    prose = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "markdown")
    assert "invented" in prose and "say nothing about any model" in prose
    assert "—" not in prose  # no em dash
    # The caller is told that the estimate applies to periods on or before the training cutoff only.
    assert "training cutoff" in prose and "Do not correct" in prose


@pytest.mark.usefixtures("offline")
def test_ask_draws_at_the_given_temperature_and_a_calibrator_needs_the_temperature_it_was_fitted_at():
    """The example's calibrators were fitted at the client's temperature, 0. Asked at 0.7 they are not used."""
    prompt, definition = worked_example()
    example = offline_example()
    fitted = ask(prompt, definition, **example.arguments(Mode.LOGPROBS), temperature=0.0)
    assert fitted.failure is None and fitted.p_memorized is not None and fitted.limits == ()
    unset = ask(prompt, definition, **example.arguments(Mode.LOGPROBS))
    assert unset.p_memorized == fitted.p_memorized
    other = ask(prompt, definition, **example.arguments(Mode.LOGPROBS), temperature=0.7)
    assert other.failure is None and other.answer == fitted.answer
    assert other.p_memorized is None and other.corrected is None
    assert front.LIMIT_CALIBRATOR_MISMATCH in other.limits
    with pytest.raises((TypeError, ValueError)):
        ask(prompt, definition, **example.arguments(Mode.LOGPROBS), temperature=-1.0)


def test_the_online_cells_are_tagged_and_do_nothing_unless_their_group_is_turned_on():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    online = [cell for cell in code_cells(notebook) if is_online(cell)]
    groups = {tag for cell in online for tag in cell.metadata["tags"] if tag != "online"}
    assert groups == {"online-setup", "online-logprobs", "online-logits", "online-npu"}
    for cell in online:
        if "online-setup" not in cell.metadata["tags"]:
            assert cell.source.startswith('if ONLINE == "'), "an online cell runs only for its group"
    assert 'os.environ.get("RECALL_GUARD_ONLINE", "")' in "\n".join(cell.source for cell in online)


def test_the_notebook_executes_offline_and_prints_what_is_stored(tmp_path, answers, monkeypatch):
    kernel_name = "recall-modes-notebook"
    kernel_directory = tmp_path / "kernels" / kernel_name
    kernel_directory.mkdir(parents=True)
    (kernel_directory / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Recall modes notebook test", "language": "python",
    }), encoding="utf-8")
    manager = KernelManager(kernel_name=kernel_name,
                            kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_directory.parent)]))
    working_directory = tmp_path / "work"
    working_directory.mkdir()
    monkeypatch.delenv("RECALL_GUARD_ONLINE", raising=False)  # the kernel inherits this process's environment
    stored = stdout(nbformat.read(NOTEBOOK, as_version=4))
    assert sum(1 for text in stored if text) >= 4, "the committed notebook carries its outputs"
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    notebook.cells.insert(0, nbformat.v4.new_code_cell(HTTP_GUARD))
    client = NotebookClient(notebook, km=manager, timeout=120, kernel_name=kernel_name,
                            resources={"metadata": {"path": str(working_directory)}})
    try:
        client.execute(cleanup_kc=True)
    finally:
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

    errors = [output for cell in code_cells(notebook) for output in cell.outputs if output.output_type == "error"]
    assert errors == []
    assert stdout(notebook)[1:] == stored, "run the notebook again and commit its outputs"
    # With no group turned on, the online cells send nothing and print nothing but the setup line.
    for cell in code_cells(notebook):
        if is_online(cell) and "online-setup" not in cell.metadata["tags"]:
            assert printed(cell) == "", cell.source.splitlines()[0]
    _example, results = answers
    text = "".join(stored)
    for mode, result in results.items():
        assert f"corrected = answer x (1 - {result.p_memorized:.3f})" in text
        assert mode.value in text
    for entry in load_catalogue().entries:
        assert entry.entry_id in text
