"""Lifecycle of the full-distribution host behind its bearer gateway (task 6.2, Requirement 11.1).

``scripts/hosts/serve_full_distribution.ps1`` starts the teacher-forcing host of task 6.1 on loopback,
puts the bearer gateway in front of it, stops both on idle or on request and verifies that no
process and no listener remain. These tests load no model: they parse the script's ``param``
defaults, drive the real gateway against a fake upstream host (an ephemeral loopback HTTP server
in its own process), run the RAM gate's refusal and check the stop verification. They use
temporary state folders, ephemeral loopback ports and a temporary token file, never the root
``.env`` or the real host ports. They need PowerShell 7 on a Windows host and skip elsewhere.
"""
import contextlib
import functools
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import psutil
import pytest
import requests

from recall_guard.core.full_distribution import ENDPOINT_PATH, FullDistributionClient
from recall_guard.core.profiles import _ram_recommendation, load_profile
from recall_guard.core.request_control import RequestError
from tests.core.test_profiles import FULL_DISTRIBUTION_PORT, LLAMACPP, STATE_FOLDERS, gateway_port

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/hosts/serve_full_distribution.ps1"
PROFILE_FILE = ROOT / "recall_guard/core/provider_profiles/llamacpp-gemma3-4b.json"
STATE_FOLDER = STATE_FOLDERS[1]  # runs/finance-sim/hosts/full-distribution
TOKEN_NAME = "LOCAL_MODEL_API_KEY"
TOKEN = "test-token-not-a-secret"
DIGEST = "sha256:" + "a" * 64
EVIDENCE = {"model_digest": DIGEST, "n_vocab": 8, "response_token_ids": [3, 5], "logprob": [-0.25, -1.5],
            "entropy": [0.9, 1.7], "rank": [1, 2], "top_logprobs": [[[3, -0.25], [1, -2.0]], [[2, -0.5], [5, -1.5]]]}


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
    return _launch(lambda: subprocess.run([_pwsh_binary(), "-NoProfile", "-NonInteractive", *args],
                                          capture_output=True, text=True, encoding="utf-8", errors="replace",
                                          timeout=timeout))


def _script(*args: str, timeout: float = 120) -> subprocess.CompletedProcess[str]:
    return _pwsh("-File", str(SCRIPT), *args, timeout=timeout)


def _free_ports(count: int) -> list[int]:
    sockets = [socket.socket() for _ in range(count)]
    try:
        for item in sockets:
            item.bind(("127.0.0.1", 0))
        return [item.getsockname()[1] for item in sockets]
    finally:
        for item in sockets:
            item.close()


def _listening(port: int) -> bool:
    return any(c.status == psutil.CONN_LISTEN and c.laddr and c.laddr.port == port
               for c in psutil.net_connections(kind="tcp"))


def _wait_for(condition, seconds: float, what: str) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.2)
    raise AssertionError(f"timed out after {seconds} s waiting for {what}")


def _tree(root: psutil.Process) -> list[psutil.Process]:
    try:
        return [root, *root.children(recursive=True)]
    except psutil.NoSuchProcess:
        return [root]


def _kill_tree(root: psutil.Process) -> None:
    """Kill through handles taken at launch; psutil refuses a PID that was reused since."""
    processes = _tree(root)
    for process in processes:
        with contextlib.suppress(psutil.NoSuchProcess):
            process.kill()
    psutil.wait_procs(processes, timeout=10)


def _alive(processes: list[psutil.Process]) -> list[int]:
    return [process.pid for process in processes if process.is_running() and process.status() != psutil.STATUS_ZOMBIE]


def _session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False  # No ambient proxy between the test and the loopback gateway.
    return session


@pytest.fixture
def rig(tmp_path):
    """Temporary state folder and token file plus two free ephemeral loopback ports."""
    env_file = tmp_path / "token.env"
    env_file.write_text(f"{TOKEN_NAME}={TOKEN}\n", encoding="utf-8")
    host_port, port = _free_ports(2)
    rig = SimpleNamespace(state=tmp_path / "state", env_file=env_file, host_port=host_port, port=port)
    rig.args = ("-StateDir", str(rig.state), "-EnvPath", str(env_file), "-HostPort", str(host_port),
                "-Port", str(port))
    return rig


def _write_state(rig, **fields) -> None:
    rig.state.mkdir(parents=True, exist_ok=True)
    state = {"host_pid": None, "host_name": "python", "gateway_pid": None, "bind_address": "127.0.0.1",
             "host_port": rig.host_port, "gateway_port": rig.port, "idle_minutes": 30,
             "started_at": "2026-11-10T00:00:00.0000000Z", "stopped_at": None, "stop_reason": None} | fields
    (rig.state / "state.json").write_text(json.dumps(state), encoding="utf-8")


def _read_state(rig) -> dict:
    return json.loads((rig.state / "state.json").read_text(encoding="utf-8-sig"))


# --- param defaults (ports and state folder fixed by task 3.3) -----------------------------------

PARAM_PROBE = r"""
param([string]$Script)
$errors = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($Script, [ref]$null, [ref]$errors)
$params = foreach ($item in $ast.ParamBlock.Parameters) {
    $set = $item.Attributes | Where-Object { $_.TypeName.Name -eq 'ValidateSet' } | Select-Object -First 1
    [ordered]@{
        name = $item.Name.VariablePath.UserPath
        default = if ($item.DefaultValue) { $item.DefaultValue.SafeGetValue() } else { $null }
        validate_set = if ($set) { @($set.PositionalArguments | ForEach-Object { $_.SafeGetValue() }) } else { $null }
    }
}
[ordered]@{ errors = @($errors).Count; params = @($params) } | ConvertTo-Json -Depth 4 -Compress
"""


def test_param_defaults_follow_the_profile_ports_and_the_state_folder(tmp_path):
    probe = tmp_path / "param_probe.ps1"
    probe.write_text(PARAM_PROBE, encoding="utf-8")
    result = _pwsh("-File", str(probe), str(SCRIPT))
    assert result.returncode == 0, result.stderr
    parsed = json.loads(result.stdout)
    assert parsed["errors"] == 0
    params = {item["name"]: item for item in parsed["params"]}
    defaults = {name: item["default"] for name, item in params.items()}

    profile = load_profile(LLAMACPP)
    host = urlsplit(profile.full_distribution.endpoint)
    assert (host.hostname, host.port) == ("127.0.0.1", FULL_DISTRIBUTION_PORT)
    assert defaults["HostPort"] == host.port == 11580
    assert defaults["Port"] == gateway_port(host.port) == 11581
    assert defaults["BindAddress"] == host.hostname  # Loopback by default, never all interfaces.
    assert defaults["StateDir"] == STATE_FOLDER == "runs/finance-sim/hosts/full-distribution"
    assert STATE_FOLDER in profile.notes
    assert defaults["EnvPath"] == ".env"
    assert (ROOT / defaults["ProfilePath"]).resolve() == PROFILE_FILE.resolve()
    assert load_profile(ROOT / defaults["ProfilePath"]).profile_id == LLAMACPP
    assert defaults["IdleMinutes"] == 30
    assert params["Action"]["validate_set"] == ["start", "stop", "status", "gateway"]


# --- gateway request handling against a fake upstream host --------------------------------------

FAKE_HOST = r'''
"""Stand-in for the full-distribution host: records each request and answers in the host's shape."""
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

with open(sys.argv[1], encoding="utf-8") as handle:
    CONFIG = json.load(handle)


class Handler(BaseHTTPRequestHandler):
    def _answer(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        with open(CONFIG["log"], "a", encoding="utf-8") as log:
            log.write(json.dumps({"method": self.command, "path": self.path,
                                  "authorization": self.headers.get("Authorization"),
                                  "body": body.decode("utf-8")}) + "\n")
        try:
            digest = json.loads(body).get("model_digest")
        except ValueError:
            digest = None
        if digest == CONFIG["digest"]:
            status, payload = 200, CONFIG["evidence"]
        else:
            status, payload = 409, {"error": {"code": "digest_mismatch", "message": "another digest"}}
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    do_POST = do_GET = _answer

    def log_message(self, *args):
        pass


server = HTTPServer(("127.0.0.1", 0), Handler)
print(server.server_address[1], flush=True)
server.serve_forever()
'''


@pytest.fixture
def fake_host(tmp_path):
    """A loopback HTTP server in its own process, standing in for the teacher-forcing host."""
    script, config, log = tmp_path / "fake_host.py", tmp_path / "fake_host.json", tmp_path / "upstream.jsonl"
    script.write_text(FAKE_HOST, encoding="utf-8")
    config.write_text(json.dumps({"log": str(log), "digest": DIGEST, "evidence": EVIDENCE}), encoding="utf-8")
    process = subprocess.Popen([sys.executable, "-I", str(script), str(config)], stdout=subprocess.PIPE, text=True)
    handle = psutil.Process(process.pid)
    try:
        port = int(process.stdout.readline())

        def seen() -> list[dict]:
            return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []
        yield SimpleNamespace(process=process, handle=handle, port=port, seen=seen)
    finally:
        _kill_tree(handle)
        process.stdout.close()


@contextlib.contextmanager
def _gateway(rig, tmp_path):
    """Run the script's ``gateway`` action in its own pwsh process until the block ends."""
    out, err = tmp_path / "gateway.stdout.txt", tmp_path / "gateway.stderr.txt"
    with open(out, "wb") as stdout, open(err, "wb") as stderr:
        process = _launch(lambda: subprocess.Popen(
            [_pwsh_binary(), "-NoProfile", "-NonInteractive", "-File", str(SCRIPT), "gateway", *rig.args],
            stdout=stdout, stderr=stderr))
    handle = psutil.Process(process.pid)
    try:
        _wait_for(lambda: _listening(rig.port) or process.poll() is not None, 60, "the gateway to listen")
        assert process.poll() is None, err.read_text(encoding="utf-8", errors="replace")
        yield SimpleNamespace(process=process, out=out, err=err)
    finally:
        _kill_tree(handle)
        process.wait(timeout=10)


def _client(rig, credential: str, digest: str = DIGEST) -> FullDistributionClient:
    return FullDistributionClient(f"http://127.0.0.1:{rig.port}", model_digest=digest,
                                  credential_ref=f"env:{TOKEN_NAME}", timeout_s=30,
                                  environment={TOKEN_NAME: credential})


def test_gateway_requires_the_bearer_and_forwards_only_the_evidence_route(rig, fake_host, tmp_path):
    _write_state(rig, host_pid=fake_host.process.pid, host_port=fake_host.port)
    url = f"http://127.0.0.1:{rig.port}"
    payload = {"model_digest": DIGEST, "prompt_text": "Synthetic prompt.", "response_text": " OK", "top_k_keep": 2}
    with _gateway(rig, tmp_path) as gateway, _session() as session:
        # Without the bearer: 401 on the served route and on any other, and nothing reaches the host.
        refused = session.post(url + ENDPOINT_PATH, json=payload, timeout=30)
        assert (refused.status_code, refused.json()) == (401, {"error": "unauthorized"})
        assert session.get(url + "/health", timeout=30).status_code == 401
        assert session.post(url + ENDPOINT_PATH, json=payload, headers={"Authorization": f"Basic {TOKEN}"},
                            timeout=30).status_code == 401
        with pytest.raises(RequestError) as caught:
            _client(rig, "not-the-token").evidence("Synthetic prompt.", " OK", top_k_keep=2)
        assert (caught.value.status_code, caught.value.code) == (401, "auth_failed")
        assert fake_host.seen() == []

        # With the bearer: evidence in the host's wire shape; the gateway strips the Authorization header.
        evidence = _client(rig, TOKEN).evidence("Synthetic prompt.", " OK", top_k_keep=2)
        assert evidence.to_dict() == EVIDENCE
        [forwarded] = fake_host.seen()
        assert (forwarded["method"], forwarded["path"], forwarded["authorization"]) == ("POST", ENDPOINT_PATH, None)
        assert json.loads(forwarded["body"]) == payload

        # Any other method or path is refused with the bearer and never forwarded.
        bearer = {"Authorization": f"Bearer {TOKEN}"}
        for method, path in (("GET", ENDPOINT_PATH), ("PUT", ENDPOINT_PATH), ("POST", ENDPOINT_PATH + "/"),
                             ("POST", ENDPOINT_PATH + "?top_k_keep=5"), ("POST", "/v1/chat/completions"),
                             ("GET", "/v1/models"), ("GET", "/health"), ("POST", "/")):
            answer = session.request(method, url + path, json=payload, headers=bearer, timeout=30)
            assert answer.status_code == 404, (method, path, answer.status_code)
        assert len(fake_host.seen()) == 1

        # The host's own refusals pass through unchanged.
        with pytest.raises(RequestError) as caught:
            _client(rig, TOKEN, digest="sha256:" + "b" * 64).evidence("Synthetic prompt.", " OK", top_k_keep=2)
        assert (caught.value.status_code, caught.value.code) == (409, "digest_mismatch")
        assert len(fake_host.seen()) == 2

    log = (rig.state / "gateway.log").read_text(encoding="utf-8")
    assert f"POST {ENDPOINT_PATH} 200" in log and f"POST {ENDPOINT_PATH} 401" in log and "GET /health 404" in log
    for text in (log, gateway.out.read_text(encoding="utf-8", errors="replace"),
                 gateway.err.read_text(encoding="utf-8", errors="replace"), json.dumps(_read_state(rig))):
        assert TOKEN not in text


def test_gateway_refuses_to_run_without_the_token(rig, fake_host, tmp_path):
    _write_state(rig, host_pid=fake_host.process.pid, host_port=fake_host.port)
    rig.env_file.write_text(f"{TOKEN_NAME}=\n", encoding="utf-8")
    result = _script("gateway", *rig.args, timeout=60)
    assert result.returncode != 0 and f"{TOKEN_NAME} missing" in result.stdout + result.stderr
    assert not _listening(rig.port)


def test_gateway_idle_stop_stops_the_host_tree_and_its_listener(rig, fake_host, tmp_path):
    host = _tree(fake_host.handle)
    _write_state(rig, host_pid=fake_host.process.pid, host_port=fake_host.port, idle_minutes=0.05)  # 3 s
    with _gateway(rig, tmp_path) as gateway:
        assert gateway.process.wait(timeout=60) == 0
    assert _alive(host) == [] and not _listening(rig.port)
    state = _read_state(rig)
    assert state["stop_reason"] == "idle_timeout" and state["stopped_at"]
    assert "idle timeout" in (rig.state / "gateway.log").read_text(encoding="utf-8")


def test_gateway_stops_itself_when_the_host_exits(rig, fake_host, tmp_path):
    _write_state(rig, host_pid=fake_host.process.pid, host_port=fake_host.port)
    with _gateway(rig, tmp_path) as gateway:
        _kill_tree(fake_host.handle)
        assert gateway.process.wait(timeout=60) == 0
    assert not _listening(rig.port)
    assert _read_state(rig)["stop_reason"] == "host_exited"


def _sleeper(*extra: str) -> tuple[subprocess.Popen, psutil.Process]:
    process = subprocess.Popen([sys.executable, "-I", "-c", "import time; time.sleep(120)", *extra])
    return process, psutil.Process(process.pid)


def test_gateway_keeps_serving_while_the_recorded_listener_lives_after_its_launcher_exits(rig, fake_host, tmp_path):
    """The uv launcher can exit (code 0) while its python host keeps listening, as in the 6.4 soak on
    2026-11-10: the gateway watches the listener recorded at readiness, logs the launcher's exit and stops
    only when the listener itself exits."""
    launcher, handle = _sleeper()
    log = rig.state / "gateway.log"
    try:
        _write_state(rig, host_pid=launcher.pid, host_name=Path(handle.name()).stem, host_port=fake_host.port,
                     listener_pid=fake_host.process.pid, listener_name=Path(fake_host.handle.name()).stem)
        with _gateway(rig, tmp_path) as gateway:
            _kill_tree(handle)
            _wait_for(lambda: (log.exists() and "launcher" in log.read_text(encoding="utf-8"))
                      or gateway.process.poll() is not None, 30, "the launcher exit line")
            assert gateway.process.poll() is None, "the gateway stopped although the listener still serves"
            assert _client(rig, TOKEN).evidence("Synthetic prompt.", " OK", top_k_keep=2).to_dict() == EVIDENCE
            _kill_tree(fake_host.handle)  # Now the host itself exits.
            assert gateway.process.wait(timeout=60) == 0
        assert not _listening(rig.port)
        assert _read_state(rig)["stop_reason"] == "host_exited"
        text = log.read_text(encoding="utf-8")
        assert f"host launcher pid {launcher.pid} exited" in text
        assert f"listener pid {fake_host.process.pid} still serves" in text and "host exited" in text
    finally:
        _kill_tree(handle)
        launcher.wait(timeout=10)


FUNCTION_TEXT_PROBE = r"""
param([string]$Script, [string]$Name)
$ast = [System.Management.Automation.Language.Parser]::ParseFile($Script, [ref]$null, [ref]$null)
$found = $ast.Find({ param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
                    $n.Name -eq $Name }, $true)
$found.Extent.Text | ConvertTo-Json -Compress
"""


def test_start_records_the_listener_at_readiness(tmp_path):
    """So that a stop and the gateway reach the python host even after its uv launcher exited (design:
    lifecycle scripts, the listener's process id is recorded at readiness)."""
    probe = tmp_path / "function_text_probe.ps1"
    probe.write_text(FUNCTION_TEXT_PROBE, encoding="utf-8")
    result = _pwsh("-File", str(probe), str(SCRIPT), "Start-Serving")
    assert result.returncode == 0, result.stderr
    text = json.loads(result.stdout)
    assert "listener_pid = $null; listener_name = $null" in text
    assert "$state.listener_pid = $listener.Pid" in text and "$state.listener_name = " in text


# --- stop verification ---------------------------------------------------------------------------


#: The script's own patterns: the host's command line (uv and python) and the gateway's (pwsh).
HOST_COMMAND = re.compile(r"(?<![\w.-])serve_full_distribution\.py\b")
GATEWAY_COMMAND = re.compile(r"serve_full_distribution\.ps1\"?\s+gateway\b")


def _running_hosts() -> list[int]:
    """Processes of a real host or gateway on this machine; the stop tests need none of them."""
    found = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        line = subprocess.list2cmdline(process.info["cmdline"] or [])
        if HOST_COMMAND.search(line) or GATEWAY_COMMAND.search(line):
            found.append(process.info["pid"])
    return found


@pytest.fixture
def quiet_host():
    if running := _running_hosts():
        pytest.skip(f"a full-distribution host or gateway runs on this machine (pid {running})")


def test_stop_and_status_report_nothing_remains_when_nothing_runs(rig, quiet_host):
    result = _script("stop", *rig.args)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Stopped: no host or gateway process remains" in result.stdout
    assert str(rig.host_port) in result.stdout and str(rig.port) in result.stdout
    status = _script("status", *rig.args)
    assert status.returncode == 0 and "No host state" in status.stdout and "No listener" in status.stdout


def test_stop_kills_the_recorded_host_tree_and_records_the_stop(rig, quiet_host):
    spawn = "import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']); "
    root = subprocess.Popen([sys.executable, "-I", "-c", spawn + "time.sleep(120)"])
    handle = psutil.Process(root.pid)
    try:
        _wait_for(lambda: len(_tree(handle)) >= 2, 30, "the fake host tree")
        time.sleep(1.0)  # Let the inner sleeper start too.
        tree = _tree(handle)
        _write_state(rig, host_pid=root.pid, host_name=Path(handle.name()).stem)
        result = _script("stop", *rig.args)
        assert result.returncode == 0, result.stdout + result.stderr
        assert _alive(tree) == []
        state = _read_state(rig)
        assert state["stop_reason"] == "stop_command" and state["stopped_at"]
    finally:
        _kill_tree(handle)
        root.wait(timeout=10)


def test_stop_reaches_the_recorded_listener_when_its_launcher_is_gone(rig, quiet_host):
    """The live 6.4 soak left the python host listening after uv exited; stop reaches it through the
    listener recorded at readiness."""
    listener, handle = _sleeper("scripts/hosts/serve_full_distribution.py")  # A host command line, as the real one.
    try:
        _write_state(rig, host_pid=None, host_name="uv", listener_pid=listener.pid,
                     listener_name=Path(handle.name()).stem)
        result = _script("stop", *rig.args)
        assert result.returncode == 0, result.stdout + result.stderr
        assert _alive(_tree(handle)) == []
        assert "Stopped: no host or gateway process remains" in result.stdout
    finally:
        _kill_tree(handle)
        listener.wait(timeout=10)


def test_stop_does_not_kill_an_unrelated_process_that_reused_the_recorded_listener_pid(rig, quiet_host):
    other, handle = _sleeper()
    try:
        _write_state(rig, host_pid=None, host_name="uv", listener_pid=other.pid, listener_name="uv")  # Name differs.
        result = _script("stop", *rig.args)
        assert result.returncode == 0, result.stdout + result.stderr
        assert other.poll() is None
    finally:
        _kill_tree(handle)
        other.wait(timeout=10)


def test_stop_does_not_kill_an_unrelated_process_that_reused_the_recorded_pid(rig, quiet_host):
    other = subprocess.Popen([sys.executable, "-I", "-c", "import time; time.sleep(120)"])
    handle = psutil.Process(other.pid)
    try:
        _write_state(rig, host_pid=other.pid, host_name="uv")  # The recorded name does not match.
        result = _script("stop", *rig.args)
        assert result.returncode == 0, result.stdout + result.stderr
        assert other.poll() is None
    finally:
        _kill_tree(handle)
        other.wait(timeout=10)


def test_stop_reports_a_listener_that_remains_and_exits_non_zero(rig, quiet_host):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", rig.host_port))
        listener.listen()
        result = _script("stop", *rig.args)
    assert result.returncode == 1
    assert f"STILL LISTENING 127.0.0.1:{rig.host_port}" in result.stdout


def test_stop_reports_a_stray_host_process_and_exits_non_zero(rig, quiet_host):
    stray = subprocess.Popen([sys.executable, "-I", "-c", "import time; time.sleep(120)",
                              "scripts/hosts/serve_full_distribution.py"])
    handle = psutil.Process(stray.pid)
    try:
        result = _script("stop", *rig.args)
        assert result.returncode == 1
        assert f"STILL RUNNING host pid {stray.pid}" in result.stdout
        assert stray.poll() is None  # Reported, never killed: it is not the recorded tree.
    finally:
        _kill_tree(handle)
        stray.wait(timeout=10)


# --- start refusals (nothing is loaded) ----------------------------------------------------------


@pytest.mark.parametrize("address", ["0.0.0.0", "::", "*", "localhost"])
def test_start_refuses_all_interfaces_and_names(rig, address):
    result = _script("start", "-BindAddress", address, *rig.args)
    assert result.returncode != 0 and "Refusing bind address" in result.stdout + result.stderr
    assert not (rig.state / "state.json").exists()
    assert rig.env_file.read_text(encoding="utf-8") == f"{TOKEN_NAME}={TOKEN}\n"


def _ram_profile(tmp_path, sibling_port: int, per_load: int) -> tuple[dict, Path]:
    data = json.loads(PROFILE_FILE.read_text(encoding="utf-8"))
    data["host_requirements"]["min_free_ram_bytes"] = per_load
    data["model_config"]["endpoint"] = f"http://127.0.0.1:{sibling_port}/v1/chat/completions"
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return data, path


def test_ram_gate_refuses_one_load_with_the_preflight_wording_and_loads_nothing(rig, tmp_path, quiet_host):
    """One model host runs at a time, so the gate is the profile's min_free_ram_bytes, never doubled
    (user correction 2026-11-10)."""
    per_load = 2 ** 50  # 1 PiB: no host has it free.
    [sibling] = _free_ports(1)  # No llama-server listens there.
    data, profile_file = _ram_profile(tmp_path, sibling, per_load)

    result = _script("start", *rig.args, "-ProfilePath", str(profile_file), "-RamWaitSeconds", "3")
    output = result.stdout + result.stderr
    assert result.returncode == 2, output
    assert "Waiting for free RAM" in output  # Progress line while waiting.
    fallbacks = tuple(data["host_requirements"]["fallback_profile_ids"])
    expected = _ram_recommendation(data["model_config"]["model"], 0, per_load, fallbacks)
    assert expected.split(" free, ", 1)[1] in output  # Same sentence as preflight_host, for one load.
    assert "--profile modal-gemma3-4b" in output and "on a machine with at least" in output
    assert "one load" in output and "two loads" not in output
    assert not (rig.state / "state.json").exists() and not _listening(rig.host_port)
    assert rig.env_file.read_text(encoding="utf-8") == f"{TOKEN_NAME}={TOKEN}\n"


@contextlib.contextmanager
def _named_listener(tmp_path, name: str):
    """A copy of the base interpreter named ``name`` listening on an ephemeral loopback port.

    Its process name is ``name`` without the extension, as a resident llama-server's is. The copy
    takes the interpreter's DLLs and finds the standard library through PYTHONHOME.
    """
    base = Path(getattr(sys, "_base_executable", sys.executable)).parent
    needed = [base / "python.exe", *sorted(base.glob("python3*.dll")), *sorted(base.glob("vcruntime140*.dll"))]
    if not all(path.is_file() for path in needed) or not (base / "Lib").is_dir():
        pytest.skip(f"no copyable base interpreter under {base}")
    folder = tmp_path / "fake-sibling"
    folder.mkdir()
    for path in needed[1:]:
        shutil.copy2(path, folder / path.name)
    executable = folder / name
    shutil.copy2(needed[0], executable)
    code = ("import socket, time; s = socket.socket(); s.bind(('127.0.0.1', 0)); s.listen(); "
            "print(s.getsockname()[1], flush=True); time.sleep(120)")
    environment = {key: value for key, value in os.environ.items() if not key.upper().startswith("PYTHON")}
    environment["PYTHONHOME"] = str(base)
    process = subprocess.Popen([str(executable), "-c", code], stdout=subprocess.PIPE, text=True, env=environment)
    handle = psutil.Process(process.pid)
    try:
        port = int(process.stdout.readline())
        _wait_for(lambda: _listening(port), 30, f"the {name} listener")
        yield port
    finally:
        _kill_tree(handle)
        process.stdout.close()


def test_a_resident_llama_server_refuses_the_start_before_the_ram_gate(rig, tmp_path, quiet_host):
    """Only one model host at a time: a llama-server listening on the generation port refuses the start."""
    per_load = 2 ** 50  # Without the sibling refusal the RAM gate refuses, so nothing is ever loaded here.
    with _named_listener(tmp_path, "llama-server.exe") as sibling:
        _, profile_file = _ram_profile(tmp_path, sibling, per_load)
        result = _script("start", *rig.args, "-ProfilePath", str(profile_file), "-RamWaitSeconds", "3")
    output = result.stdout + result.stderr
    assert result.returncode == 2, output
    assert "only one model host runs at a time" in output and f"port {sibling}" in output
    assert "serve_local_llamacpp.ps1 stop" in output
    assert "Nothing was loaded and no request was sent." in output
    assert "Waiting for free RAM" not in output  # Refused before the RAM gate.
    assert not (rig.state / "state.json").exists() and not _listening(rig.host_port)
    assert rig.env_file.read_text(encoding="utf-8") == f"{TOKEN_NAME}={TOKEN}\n"


# --- functions extracted from the script (RAM gate with mocked readings, launch inheritance) ------

FUNCTION_PROBE = r"""
param([string]$Script)
$ast = [System.Management.Automation.Language.Parser]::ParseFile($Script, [ref]$null, [ref]$null)
foreach ($name in 'Get-RamRequirement', 'Wait-FreeRam', 'Disable-OutputInheritance') {
    $definition = $ast.Find({ param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
                              $node.Name -eq $name }, $true)
    if (-not $definition) { throw "function $name not found" }
    . ([scriptblock]::Create($definition.Extent.Text))
}
"""


def _run_functions(tmp_path, body: str) -> object:
    probe = tmp_path / "function_probe.ps1"
    probe.write_text(FUNCTION_PROBE + body + "\n", encoding="utf-8")
    result = _pwsh("-File", str(probe), str(SCRIPT))
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_ram_requirement_is_one_load_never_doubled(tmp_path):
    result = _run_functions(tmp_path, "Get-RamRequirement 6442450944 | ConvertTo-Json -Compress")
    assert result == {"per_load_bytes": 6442450944, "loads": 1, "required_bytes": 6442450944}


@pytest.mark.parametrize(("samples", "seconds", "passed", "count"), [
    ([13, 13, 13], 600, True, 3),
    ([13, 1, 13, 13, 13], 600, True, 5),  # A dip resets the run of three.
    ([1], 0, False, 1),                    # Past the deadline below the gate: refused at once.
    ([13, 13, 1], 0, False, 3),
])
def test_free_ram_wait_needs_three_samples_in_a_row(tmp_path, samples, seconds, passed, count):
    queue = ", ".join(f"{gib}GB" for gib in samples)
    body = ("$script:readings = [Collections.Generic.Queue[long]]::new([long[]]@(" + queue + "))\n"
            "function Get-FreeRamBytes { $script:readings.Dequeue() }\n"
            "function Start-Sleep { }\n"
            f"Wait-FreeRam 12GB {seconds} | ConvertTo-Json -Compress")
    result = _run_functions(tmp_path, body)
    assert (result["passed"], result["samples"]) == (passed, count)
    assert result["required_bytes"] == 12 * 1024 ** 3


def test_launched_children_do_not_hold_the_callers_output_pipes(tmp_path):
    """A caller that captures the script's output must not wait for the long-lived host and gateway."""
    body = "\n".join((
        "Disable-OutputInheritance",
        f"$child = Start-Process -FilePath '{sys.executable}' -WindowStyle Hidden -PassThru "
        "-ArgumentList '-I', '-c', '\"import time; time.sleep(60)\"' "
        f"-RedirectStandardOutput '{tmp_path / 'child.out'}' -RedirectStandardError '{tmp_path / 'child.err'}'",
        "$child.Id | ConvertTo-Json -Compress"))
    started = time.monotonic()
    pid = _run_functions(tmp_path, body)  # capture_output=True: blocks while any child holds the pipes.
    elapsed = time.monotonic() - started
    assert elapsed < 50, f"the caller waited {elapsed:.1f} s, until the launched child exited"
    child = psutil.Process(pid)
    try:
        assert child.is_running()
    finally:
        _kill_tree(child)
