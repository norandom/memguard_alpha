"""Launch inheritance of ``scripts/serve_local_flm.ps1`` (upstream fix found in task 6.2).

``start`` launches the FLM process and then the gateway with ``Start-Process``. Unless the inherit
flag is cleared on the script's stdout and stderr handles first, both children hold a capturing
caller's pipes, and the caller waits until they exit. These tests start no FastFlowLM and load no
model. One runs the script's ``Disable-OutputInheritance`` before launching a sleeping Python child,
as the sibling test of ``serve_full_distribution.ps1`` does. The other runs ``start`` from a
temporary copy of the script, so its state, logs and token file stay under a temporary root, with a
fake ``flm`` that lists one installed tag and holds an ephemeral loopback port. They need
PowerShell 7 on a Windows host and skip elsewhere.
"""
import contextlib
import functools
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/serve_local_flm.ps1"
TOKEN_NAME = "LOCAL_MODEL_API_KEY"
TOKEN = "test-token-not-a-secret"


def _pwsh_executable() -> str | None:
    """Prefer an installed PowerShell 7 over the Store app-execution alias (tasks.md Implementation Notes)."""
    if sys.platform != "win32":
        return None
    installed = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "PowerShell/7/pwsh.exe"
    return str(installed) if installed.is_file() else shutil.which("pwsh")


PWSH = _pwsh_executable()
pytestmark = pytest.mark.skipif(PWSH is None, reason="needs PowerShell 7 (pwsh) on a Windows host")


def _launch(start):
    """Run ``start()``; retry once when Windows refuses to launch the ``pwsh`` app-execution alias.

    On hosts where ``pwsh`` resolves to the WindowsApps alias, CreateProcess intermittently fails
    with ``PermissionError`` (WinError 5). A real permission problem fails twice.
    """
    for attempt in (1, 2):
        try:
            return start()
        except PermissionError:
            if attempt == 2:
                raise
            time.sleep(1.0)
    raise AssertionError("unreachable")


@functools.cache
def _pwsh_binary() -> str:
    """The pwsh.exe behind ``PWSH``, resolved once, so later launches skip the WindowsApps alias."""
    resolved = _launch(lambda: subprocess.run([PWSH, "-NoProfile", "-NonInteractive", "-Command",
                                               "(Get-Process -Id $PID).Path"], capture_output=True, text=True,
                                              timeout=120)).stdout.strip()
    try:
        return resolved if resolved and Path(resolved).is_file() else PWSH
    except OSError:
        return PWSH


def _pwsh(*args: str, timeout: float = 120) -> subprocess.CompletedProcess[str]:
    """Run pwsh with captured output: the call returns only once every holder of its pipes exits."""
    return _launch(lambda: subprocess.run([_pwsh_binary(), "-NoProfile", "-NonInteractive", *args],
                                          capture_output=True, text=True, encoding="utf-8", errors="replace",
                                          timeout=timeout))


def _free_ports(count: int) -> list[int]:
    sockets = [socket.socket() for _ in range(count)]
    try:
        for item in sockets:
            item.bind(("127.0.0.1", 0))
        return [item.getsockname()[1] for item in sockets]
    finally:
        for item in sockets:
            item.close()


def _processes(pids) -> list[psutil.Process]:
    """Handles for the recorded PIDs that still run, with their process trees."""
    found = []
    for pid in pids:
        with contextlib.suppress(psutil.NoSuchProcess):
            root = psutil.Process(pid)
            found.extend([root, *root.children(recursive=True)])
    return found


def _kill(processes: list[psutil.Process]) -> None:
    for process in processes:
        with contextlib.suppress(psutil.NoSuchProcess):
            process.kill()
    psutil.wait_procs(processes, timeout=10)


FUNCTION_PROBE = r"""
param([string]$Script)
$ast = [System.Management.Automation.Language.Parser]::ParseFile($Script, [ref]$null, [ref]$null)
$definition = $ast.Find({ param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
                          $node.Name -eq 'Disable-OutputInheritance' }, $true)
if (-not $definition) { throw 'function Disable-OutputInheritance not found' }
. ([scriptblock]::Create($definition.Extent.Text))
"""


def test_disable_output_inheritance_keeps_a_launched_child_off_the_callers_pipes(tmp_path):
    body = "\n".join((
        "Disable-OutputInheritance",
        f"$child = Start-Process -FilePath '{sys.executable}' -WindowStyle Hidden -PassThru "
        "-ArgumentList '-I', '-c', '\"import time; time.sleep(60)\"' "
        f"-RedirectStandardOutput '{tmp_path / 'child.out'}' -RedirectStandardError '{tmp_path / 'child.err'}'",
        "$child.Id | ConvertTo-Json -Compress"))
    probe = tmp_path / "function_probe.ps1"
    probe.write_text(FUNCTION_PROBE + body + "\n", encoding="utf-8")
    started = time.monotonic()
    result = _pwsh("-File", str(probe), str(SCRIPT))
    elapsed = time.monotonic() - started
    assert result.returncode == 0, result.stdout + result.stderr
    child = _processes([json.loads(result.stdout.strip().splitlines()[-1])])
    try:
        assert elapsed < 30, f"the caller waited {elapsed:.1f} s, until the launched child exited"
        assert child and child[0].is_running()
    finally:
        _kill(child)


#: Stands in for flm.exe: answers ``list`` and ``version``; ``serve`` holds its port for 60 s.
FAKE_FLM = r'''
import json
import socket
import sys
import time

command = sys.argv[1:2]
if command == ["list"]:
    print(json.dumps({"models": [{"name": "gemma3:4b", "model": "gemma3:4b", "installed": True, "size": 4,
                                  "footprint": 4.4, "default_context_length": 2048, "details": {}}]}))
elif command == ["version"]:
    print(json.dumps({"version": "0.0.0-fake"}))
elif command == ["serve"]:
    listener = socket.create_server(("127.0.0.1", int(sys.argv[sys.argv.index("-p") + 1])))
    time.sleep(60)
else:
    sys.exit(f"unexpected fake flm command: {sys.argv[1:]}")
'''


def test_start_returns_while_the_flm_process_and_the_gateway_keep_running(tmp_path):
    """``start`` from a captured caller returns in seconds; the FLM process and gateway stay up."""
    if " " in str(tmp_path):
        pytest.skip("the script passes its own path unquoted to the gateway's Start-Process")
    root = tmp_path / "root"
    (root / "scripts").mkdir(parents=True)
    (root / "models/flm").mkdir(parents=True)
    script = root / "scripts" / SCRIPT.name  # $Root is the copy's parent folder: state and .env stay here.
    shutil.copyfile(SCRIPT, script)
    env_file = root / ".env"
    env_file.write_text(f"{TOKEN_NAME}={TOKEN}\n", encoding="utf-8")
    fake = tmp_path / "fake"
    fake.mkdir()
    (fake / "fake_flm.py").write_text(FAKE_FLM, encoding="utf-8")
    (fake / "flm.cmd").write_text(f'@"{sys.executable}" -I "{fake / "fake_flm.py"}" %*\n', encoding="utf-8")
    flm_port, port = _free_ports(2)
    started = time.monotonic()
    result = _pwsh("-File", str(script), "start", "gemma3:4b", "-FlmPath", str(fake / "flm.cmd"),
                   "-BindAddress", "127.0.0.1", "-FlmPort", str(flm_port), "-Port", str(port),
                   "-StartTimeoutSeconds", "30", timeout=180)
    elapsed = time.monotonic() - started
    state_file = root / "runs/flm-gateway/state.json"
    state = json.loads(state_file.read_text(encoding="utf-8-sig")) if state_file.exists() else {}
    children = _processes([state[name] for name in ("gateway_pid", "flm_pid") if state.get(name)])
    try:
        assert result.returncode == 0, result.stdout + result.stderr
        assert elapsed < 30, f"the caller waited {elapsed:.1f} s, until the FLM process and the gateway exited"
        assert state["flm_pid"] in [child.pid for child in children]
        assert state["gateway_pid"] in [child.pid for child in children]
        assert all(child.is_running() for child in children)
        assert f"Gateway 127.0.0.1:{port}" in result.stdout and f"{TOKEN_NAME} existing" in result.stdout
        assert TOKEN not in result.stdout + result.stderr
        assert env_file.read_text(encoding="utf-8") == f"{TOKEN_NAME}={TOKEN}\n"
    finally:
        _kill(children)
