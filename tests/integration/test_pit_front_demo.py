"""The front's demos run offline: the script with ``--fixture`` and the notebook as committed.

Both enter through ``recall_guard.ask`` with fixed replies. The script runs in a
subprocess without credential variables and with connections blocked, so the
offline claim is tested, not assumed. The notebook runs in a kernel of the
current interpreter, from a temporary working directory, with HTTP blocked (the
kernel itself needs sockets to talk to the test).
"""

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import nbformat
import pytest
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient

from recall_guard import WORKED_EXAMPLE_REPLY, FixedReplies, Mode, ask, load_catalogue, worked_example

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "examples" / "pit_front.py"
NOTEBOOK = ROOT / "notebooks" / "pit_front.ipynb"
DEFAULT_ENTRY = "llamacpp-gemma3-4b"
AXES = ("inflation", "growth", "credit_stress", "policy", "risk_appetite")
FIXED_NOTICE = "invented fixed reply"
SOCKET_MARKER = "SOCKET-ATTEMPT"
JSON_KEYS = {"entry", "mode", "mode_choice", "status", "reason", "answer", "p_memorized", "corrected", "limits",
             "failure", "fixed_reply"}

#: Runs a script with every way out of the process refused: connecting a socket, sending on an unconnected
#: one and resolving a name. An attempt leaves a marker on stderr even when the caller handles the refusal.
#: Creating a socket object is left alone: urllib3 creates and binds one to the loopback address when it is
#: imported, to learn whether IPv6 exists, and connects it to nothing.
SOCKET_GUARD = """
import runpy
import socket
import sys


def blocked(*args, **kwargs):
    sys.stderr.write({marker!r} + "\\n")
    raise ConnectionRefusedError("connections are blocked in this test")


socket.socket.connect = blocked
socket.socket.connect_ex = blocked
socket.socket.sendto = blocked
socket.create_connection = blocked
socket.getaddrinfo = blocked
sys.argv = [{script!r}, *{arguments!r}]
runpy.run_path({script!r}, run_name="__main__")
"""

HTTP_GUARD = """
import http.client
import requests


def block_model_http(*args, **kwargs):
    raise AssertionError("Outbound HTTP is forbidden in the offline demo")


requests.sessions.Session.request = block_model_http
http.client.HTTPConnection.connect = block_model_http
"""


def scrubbed_environment():
    """What an interpreter needs to start, and nothing else: no credential or endpoint variable."""
    keep = ("SYSTEMROOT", "SYSTEMDRIVE", "PATH", "PATHEXT", "TEMP", "TMP", "TMPDIR", "HOME", "USERPROFILE")
    environment = {name: os.environ[name] for name in keep if name in os.environ}
    environment.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    return environment


def run_script(tmp_path, *arguments, script=SCRIPT):
    """Run the script as ``python script arguments`` would, with connections blocked and credentials absent."""
    code = SOCKET_GUARD.format(marker=SOCKET_MARKER, script=str(script), arguments=list(arguments))
    return subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env=scrubbed_environment(), text=True,
                          encoding="utf-8", capture_output=True, timeout=120, check=False)


def fixture_answer(entry=DEFAULT_ENTRY, mode=None):
    """What the front returns for the worked example with the fixed reply, asked in this process."""
    prompt, definition = worked_example()
    return ask(prompt, definition, entry=entry, mode=mode, client=FixedReplies([WORKED_EXAMPLE_REPLY], cycle=True))


def entry_without_logprobs():
    return next(entry.entry_id for entry in load_catalogue().entries if not entry.modes[Mode.LOGPROBS].available)


def submodule_imports(source):
    """Every import in ``source`` that reaches below the package root of recall_guard."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found += [alias.name for alias in node.names if alias.name.startswith("recall_guard.")]
        elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("recall_guard."):
            found.append(node.module)
    return found


def imported_names(source):
    """The names ``source`` imports from the package root."""
    return [alias.name for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.ImportFrom) and node.module == "recall_guard" for alias in node.names]


def code_cells(notebook):
    return [cell for cell in notebook.cells if cell.cell_type == "code"]


def test_the_socket_guard_refuses_and_marks_an_attempt(tmp_path):
    probe = tmp_path / "probe.py"
    probe.write_text("import socket\nsocket.socket().connect(('127.0.0.1', 9))\n", encoding="utf-8")
    completed = run_script(tmp_path, script=probe)
    assert completed.returncode != 0
    assert SOCKET_MARKER in completed.stderr
    probe.write_text("import requests\nrequests.get('http://127.0.0.1:9/', timeout=1)\n", encoding="utf-8")
    completed = run_script(tmp_path, script=probe)
    assert completed.returncode != 0
    assert SOCKET_MARKER in completed.stderr


def test_script_runs_offline_and_shows_entry_mode_status_and_limits(tmp_path):
    completed = run_script(tmp_path, "--fixture")
    assert completed.returncode == 0, completed.stderr
    assert SOCKET_MARKER not in completed.stderr
    output = completed.stdout
    expected = fixture_answer()
    assert expected.failure is None and expected.limits

    for entry in load_catalogue().entries:  # The listing names every entry of the catalogue.
        assert entry.entry_id in output
    assert f"entry: {expected.entry_id}" in output
    assert f"mode: {expected.mode.value}" in output
    assert expected.mode_choice in output
    assert f"status: {expected.status.value.replace('_', ' ')}" in output
    assert expected.reason in output
    assert "limits:" in output
    for limit in expected.limits:
        assert f"- {limit}" in output
    for axis in AXES:
        assert f"{axis}: {expected.answer[axis]:+.2f}" in output
    assert "recall estimate: none" in output
    assert "corrected answer: none, because" in output
    assert output.count(FIXED_NOTICE) == 1
    assert "not a model's quality" in output

    prompt, _ = worked_example()
    assert prompt in output  # The prompt is shown as it is sent.
    assert "\N{EM DASH}" not in output.replace(prompt, "")


def test_script_prints_one_json_object_for_programs(tmp_path):
    completed = run_script(tmp_path, "--fixture", "--json")
    assert completed.returncode == 0, completed.stderr
    assert SOCKET_MARKER not in completed.stderr
    result = json.loads(completed.stdout)
    expected = fixture_answer()
    assert set(result) == JSON_KEYS
    assert result["entry"] == DEFAULT_ENTRY
    assert result["mode"] == expected.mode.value
    assert result["mode_choice"] == expected.mode_choice
    assert result["status"] == expected.status.value
    assert result["reason"] == expected.reason
    assert result["answer"] == dict(expected.answer) and set(result["answer"]) == set(AXES)
    assert result["p_memorized"] is None and result["corrected"] is None
    assert result["limits"] == list(expected.limits)
    assert result["failure"] is None
    assert result["fixed_reply"] is True


def test_script_answers_in_the_logits_mode_for_an_entry_without_logprobs(tmp_path):
    entry = entry_without_logprobs()
    completed = run_script(tmp_path, "--fixture", "--entry", entry)
    assert completed.returncode == 0, completed.stderr
    expected = fixture_answer(entry)
    assert expected.mode is Mode.LOGITS
    assert f"entry: {entry}" in completed.stdout
    assert "mode: logits" in completed.stdout
    assert expected.mode_choice in completed.stdout


@pytest.mark.parametrize("case", ["mode_not_offered", "unknown_entry"])
def test_script_prints_the_failed_step_with_its_remedy_and_exits_non_zero(tmp_path, case):
    if case == "mode_not_offered":
        entry, arguments, step = entry_without_logprobs(), ["--mode", "logprobs"], "mode"
    else:
        entry, arguments, step = "no-such-entry", [], "entry"
    completed = run_script(tmp_path, "--fixture", "--entry", entry, *arguments)
    expected = fixture_answer(entry, Mode.LOGPROBS if arguments else None)
    assert expected.failure is not None and expected.failure.step == step

    assert completed.returncode not in (0, 2), completed.stderr  # 2 is argparse's usage error.
    assert "Traceback" not in completed.stderr
    assert SOCKET_MARKER not in completed.stderr
    output = completed.stdout
    assert f"failed at step: {step}" in output
    assert expected.failure.detail in output
    assert f"what to change: {expected.failure.remedy}" in output
    for axis in AXES:  # No partial result is shown as an answer.
        assert f"{axis}:" not in output

    as_json = run_script(tmp_path, "--fixture", "--json", "--entry", entry, *arguments)
    assert as_json.returncode == completed.returncode
    result = json.loads(as_json.stdout)
    assert result["failure"] == {"step": step, "detail": expected.failure.detail, "remedy": expected.failure.remedy}
    assert result["answer"] is None and result["p_memorized"] is None and result["corrected"] is None


def test_script_without_fixture_reports_a_route_that_is_not_up(tmp_path):
    """Without ``--fixture`` the front builds the client; with no route there is a failure, not a traceback."""
    completed = run_script(tmp_path, "--json")
    assert completed.returncode not in (0, 2), completed.stderr
    assert "Traceback" not in completed.stderr
    result = json.loads(completed.stdout)
    assert result["fixed_reply"] is False
    assert result["answer"] is None
    assert result["failure"]["step"] in {"client", "request"}
    assert result["failure"]["remedy"]


def test_script_and_notebook_import_only_the_package_root():
    script = SCRIPT.read_text(encoding="utf-8")
    assert submodule_imports(script) == []
    assert "ask" in imported_names(script)
    assert not [name for name in imported_names(script) if name.startswith("_")]

    notebook = nbformat.read(NOTEBOOK, as_version=4)
    names = []
    for cell in code_cells(notebook):
        assert submodule_imports(cell.source) == []
        names += imported_names(cell.source)
    assert "ask" in names
    assert not [name for name in names if name.startswith("_")]


def test_notebook_is_committed_offline_by_default_with_cleared_outputs_and_plain_prose():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    cells = code_cells(notebook)
    assert all(cell.outputs == [] and cell.execution_count is None for cell in cells)
    assert "FIXTURE = True" in cells[0].source
    prose = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "markdown")
    assert prose.count(FIXED_NOTICE) == 1
    assert "\N{EM DASH}" not in "\n".join(cell.source for cell in notebook.cells)


def test_notebook_executes_offline_and_shows_entry_mode_status_and_limits(tmp_path):
    kernel_name = "pit-front-demo"
    kernel_directory = tmp_path / "kernels" / kernel_name
    kernel_directory.mkdir(parents=True)
    (kernel_directory / "kernel.json").write_text(json.dumps({
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Front demo test", "language": "python",
    }), encoding="utf-8")
    specs = KernelSpecManager(kernel_dirs=[str(kernel_directory.parent)])
    manager = KernelManager(kernel_name=kernel_name, kernel_spec_manager=specs)
    working_directory = tmp_path / "work"
    working_directory.mkdir()
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    notebook.cells.insert(0, nbformat.v4.new_code_cell(
        HTTP_GUARD + "\nimport sys\nfrom pathlib import Path\n"
        + f"assert Path(sys.executable).resolve() == Path({sys.executable!r}).resolve()\n"
        + f"assert Path.cwd().resolve() == Path({str(working_directory)!r}).resolve()\n"))
    notebook.cells.append(nbformat.v4.new_code_cell(
        "assert FIXTURE is True\n"
        "assert client.requests == 32 and set(client.prompts) == {prompt}\n"
        "assert result.failure is None and result.p_memorized is None and result.corrected is None\n"))
    client = NotebookClient(notebook, km=manager, timeout=120, kernel_name=kernel_name,
                            resources={"metadata": {"path": str(working_directory)}})
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

    errors = [output for cell in code_cells(notebook) for output in cell.outputs if output.output_type == "error"]
    assert errors == []
    printed = "".join(output.text for cell in code_cells(notebook) for output in cell.outputs
                      if output.output_type == "stream" and output.name == "stdout")
    expected = fixture_answer()
    for entry in load_catalogue().entries:
        assert entry.entry_id in printed
    assert f"entry: {expected.entry_id}" in printed
    assert f"mode: {expected.mode.value}" in printed
    assert expected.mode_choice in printed
    assert f"status: {expected.status.value.replace('_', ' ')}" in printed
    assert expected.reason in printed
    assert "limits:" in printed
    for limit in expected.limits:
        assert f"- {limit}" in printed
    for axis in AXES:
        assert f"{axis}: {expected.answer[axis]:+.2f}" in printed
    assert "recall estimate: none" in printed
    assert "corrected answer: none, because" in printed
    assert list(working_directory.iterdir()) == []  # The demo writes nothing.
