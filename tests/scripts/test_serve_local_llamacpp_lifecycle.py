"""Lifecycle of llama-server behind its bearer gateway (task 6.3, Requirements 3.1 and 11.1).

``scripts/hosts/serve_local_llamacpp.ps1`` starts llama-server b12199 on the pinned GGUF on loopback,
puts the bearer gateway in front of it, stops both on idle, on host exit or on request and
verifies that no process and no listener remain. These tests load no model: they parse the
script's ``param`` defaults and its server command line, drive the real gateway against a fake
llama-server (an ephemeral loopback HTTP server in its own process), run the RAM gate's branches
and check the stop verification. They use temporary state folders, ephemeral loopback ports and a
temporary token file, never the root ``.env`` or the real host ports. They need PowerShell 7 on a
Windows host and skip elsewhere; the check of the installed build also needs llama.cpp b12199.
"""
import contextlib
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path, PureWindowsPath
from types import SimpleNamespace
from urllib.parse import urlsplit

import psutil
import pytest

from recall_guard.core.completion import FeatureChoices
from recall_guard.core.profiles import _ram_recommendation, connect, load_profile
from recall_guard.core.request_control import CallBudget, RequestError
from scripts.hosts.llamacpp_native import LLAMA_DIR, MODEL_DIGEST, MODEL_PATH, SERVER_FLAGS
from tests.core.test_profiles import GGUF_DIGEST, LLAMA_SERVER_PORT, LLAMACPP, STATE_FOLDERS, gateway_port
from tests.scripts.test_serve_full_distribution_lifecycle import (
    PARAM_PROBE,
    PWSH,
    _alive,
    _free_ports,
    _kill_tree,
    _launch,
    _listening,
    _pwsh,
    _pwsh_binary,
    _session,
    _tree,
    _wait_for,
)

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/hosts/serve_local_llamacpp.ps1"
SIBLING_SCRIPT = ROOT / "scripts/hosts/serve_full_distribution.ps1"
PROFILE_FILE = ROOT / "recall_guard/core/provider_profiles/llamacpp-gemma3-4b.json"
LLAMA_SERVER = LLAMA_DIR / "llama-server.exe"
STATE_FOLDER = STATE_FOLDERS[0]  # runs/finance-sim/hosts/llama-server
BLOB = MODEL_DIGEST.replace(":", "-")
TOKEN_NAME = "LOCAL_MODEL_API_KEY"
TOKEN = "test-token-not-a-secret"
CHAT_PATH = "/v1/chat/completions"
#: The routes the gateway forwards: generation, and what the runner reads before it (models, health, digest).
ROUTES = (("POST", CHAT_PATH), ("GET", "/v1/models"), ("GET", "/health"), ("GET", "/props"))
UNSERVED = (("POST", "/props"), ("PUT", "/props"), ("GET", CHAT_PATH), ("POST", CHAT_PATH + "/"),
            ("POST", CHAT_PATH + "?stream=true"), ("GET", "/health?include_slots"), ("GET", "/HEALTH"),
            ("POST", "/v1/completions"), ("POST", "/completion"), ("POST", "/tokenize"), ("GET", "/slots"),
            ("GET", "/metrics"), ("POST", "/v1/full_distribution"), ("DELETE", "/v1/models"), ("GET", "/"))

pytestmark = pytest.mark.skipif(PWSH is None, reason="needs PowerShell 7 (pwsh) on a Windows host")


def _token(text: str, logprob: float) -> dict:
    """One generated token in llama-server's OpenAI shape, with five alternatives (the profile's top_logprobs)."""
    names = [text, *(f"{text}{rank}" for rank in range(1, 5))]
    alternatives = [{"id": 100 + rank, "token": name, "bytes": list(name.encode()), "logprob": logprob - 1.5 * rank}
                    for rank, name in enumerate(names)]
    return {"id": 100, "token": text, "bytes": list(text.encode()), "logprob": logprob, "top_logprobs": alternatives}


CHAT = {"id": "chatcmpl-test", "object": "chat.completion", "created": 1, "model": "gemma3:4b",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "OK."},
                     "logprobs": {"content": [_token("OK", -0.05), _token(".", -0.3)]}}],
        "usage": {"completion_tokens": 2, "prompt_tokens": 12, "total_tokens": 14}}
PROPS = {"model_path": str(PureWindowsPath(r"D:\checkout\models\blobs") / BLOB), "total_slots": 1,
         "build_info": "b12199-372dc5ca1"}
MODELS = {"object": "list", "data": [{"id": BLOB, "object": "model", "owned_by": "llamacpp"}]}
ANSWERS = {f"POST {CHAT_PATH}": [200, CHAT], "GET /v1/models": [200, MODELS], "GET /health": [200, {"status": "ok"}],
           "GET /props": [200, PROPS]}
LOADING = [503, {"error": {"code": 503, "message": "Loading model", "type": "unavailable_error"}}]


def _script(*args: str, timeout: float = 120) -> subprocess.CompletedProcess[str]:
    return _pwsh("-File", str(SCRIPT), *args, timeout=timeout)


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
    state = {"host_pid": None, "host_name": "python", "listener_pid": None, "listener_name": None,
             "gateway_pid": None, "bind_address": "127.0.0.1", "host_port": rig.host_port, "gateway_port": rig.port,
             "idle_minutes": 30, "started_at": "2026-11-10T00:00:00.0000000Z", "stopped_at": None,
             "stop_reason": None} | fields
    (rig.state / "state.json").write_text(json.dumps(state), encoding="utf-8")


def _read_state(rig) -> dict:
    return json.loads((rig.state / "state.json").read_text(encoding="utf-8-sig"))


def _gateway_log(rig) -> str:
    return (rig.state / "gateway.log").read_text(encoding="utf-8")


# --- param defaults and the server command line (task 3.3 ports, scripts/hosts/llamacpp_native.py) -----


def test_param_defaults_follow_the_profile_ports_the_state_folder_and_the_pinned_gguf(tmp_path):
    probe = tmp_path / "param_probe.ps1"
    probe.write_text(PARAM_PROBE, encoding="utf-8")
    result = _pwsh("-File", str(probe), str(SCRIPT))
    assert result.returncode == 0, result.stderr
    parsed = json.loads(result.stdout)
    assert parsed["errors"] == 0
    params = {item["name"]: item for item in parsed["params"]}
    defaults = {name: item["default"] for name, item in params.items()}

    profile = load_profile(LLAMACPP)
    host = urlsplit(profile.model_config.endpoint)
    assert (host.hostname, host.port, host.path) == ("127.0.0.1", LLAMA_SERVER_PORT, CHAT_PATH)
    assert defaults["HostPort"] == host.port == 11570
    assert defaults["Port"] == gateway_port(host.port) == 11571
    assert defaults["BindAddress"] == host.hostname  # Loopback by default, never all interfaces.
    assert defaults["StateDir"] == STATE_FOLDER == "runs/finance-sim/hosts/llama-server"
    assert STATE_FOLDER in profile.notes
    assert defaults["EnvPath"] == ".env"
    assert (ROOT / defaults["ProfilePath"]).resolve() == PROFILE_FILE.resolve()
    assert load_profile(ROOT / defaults["ProfilePath"]).profile_id == LLAMACPP
    assert defaults["IdleMinutes"] == 30
    assert params["Action"]["validate_set"] == ["start", "stop", "status", "gateway"]
    # The pinned GGUF of scripts/hosts/llamacpp_native.py, which is the profile's declared digest.
    assert (ROOT / defaults["ModelPath"]).resolve() == MODEL_PATH.resolve()
    assert PureWindowsPath(defaults["ModelPath"]).name == BLOB
    assert MODEL_DIGEST == GGUF_DIGEST == profile.model_config.local_identity.declared.artifact_digest
    assert defaults["LlamaDir"] == ""  # Empty: llamacpp_native.LLAMA_DIR (checked with the command line).


FUNCTION_PROBE = r"""
param([string]$Script, [string]$Names)
$ErrorActionPreference = 'Stop'
$ast = [System.Management.Automation.Language.Parser]::ParseFile($Script, [ref]$null, [ref]$null)
foreach ($name in $Names -split ',') {
    $definition = $ast.Find({ param($node) $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
                              $node.Name -eq $name }, $true)
    if (-not $definition) { throw "function $name not found" }
    . ([scriptblock]::Create($definition.Extent.Text))
}
"""


def _run_functions(tmp_path, names, body: str) -> object:
    """Define the named functions from the script, run ``body`` and decode its last output line as JSON."""
    probe = tmp_path / "function_probe.ps1"
    probe.write_text(FUNCTION_PROBE + body + "\n", encoding="utf-8")
    result = _pwsh("-File", str(probe), str(SCRIPT), ",".join(names))
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_server_command_is_the_reference_flags_with_the_model_host_and_port(tmp_path):
    body = (f"@((Get-ServerCommand '' '{MODEL_PATH}' 11570), (Get-ServerCommand '{tmp_path}' 'model.gguf' 12345)) | "
            "ConvertTo-Json -Depth 3 -Compress")
    default, custom = _run_functions(tmp_path, ["Get-ServerCommand"], body)
    assert Path(default["exe"]) == LLAMA_SERVER
    # llama-server b12199 has no switch for logprobs: each request asks for logprobs and top_logprobs.
    assert default["arguments"] == ["-m", str(MODEL_PATH), "--host", "127.0.0.1", "--port", "11570", *SERVER_FLAGS]
    assert Path(custom["exe"]) == tmp_path / "llama-server.exe"
    assert custom["arguments"] == ["-m", "model.gguf", "--host", "127.0.0.1", "--port", "12345", *SERVER_FLAGS]


def test_server_build_is_read_from_the_version_output(tmp_path):
    texts = ["version: 0.9.0-dev (build 12199, commit 372dc5ca1)\nbuilt with Clang 20.1.8 for Windows x86_64",
             "version: 12199 (372dc5ca1)\nbuilt with MSVC", "no version here"]
    body = ("@(" + ", ".join(f"[string](Get-ServerBuild '{text}')" for text in texts) + ") | ConvertTo-Json -Compress")
    assert _run_functions(tmp_path, ["Get-ServerBuild"], body) == ["12199", "12199", ""]


@pytest.mark.skipif(not LLAMA_SERVER.is_file(), reason="needs the winget llama.cpp build of scripts/hosts/llamacpp_native.py")
def test_installed_llama_server_is_the_pinned_build_and_takes_logprobs_per_request(tmp_path):
    def output(*args: str) -> str:
        done = _launch(lambda: subprocess.run([str(LLAMA_SERVER), *args], capture_output=True, text=True,
                                              encoding="utf-8", errors="replace", timeout=120))
        return done.stdout + done.stderr

    version = tmp_path / "version.txt"
    version.write_text(output("--version"), encoding="utf-8")
    build = _run_functions(tmp_path, ["Get-ServerBuild"], f"Get-ServerBuild ([IO.File]::ReadAllText('{version}'))")
    declared = load_profile(LLAMACPP).model_config.local_identity.declared.runtime_version
    assert f"b{build}" == declared == "b12199"
    usage = output("--help")
    assert "--ctx-size" in usage and "--swa-full" in usage  # The real option list, not an error text.
    # No server switch enables or sizes logprobs in this build, so the command line adds none.
    assert not re.search(r"--(?:logprobs|top-logprobs|n-probs|post-sampling-probs)\b", usage)


def test_model_file_must_be_the_pinned_blob_by_name_and_sha256(tmp_path):
    content = b"not a model"
    digest = "sha256:" + hashlib.sha256(content).hexdigest()
    pinned = tmp_path / digest.replace(":", "-")
    pinned.write_bytes(content)
    renamed = tmp_path / "model.gguf"
    renamed.write_bytes(content)
    altered = tmp_path / "other" / digest.replace(":", "-")
    altered.parent.mkdir()
    altered.write_bytes(b"changed bytes")
    checks = [f"'{pinned}' $digest -Hash", f"'{renamed}' $digest", f"'{altered}' $digest",
              f"'{altered}' $digest -Hash", f"'{tmp_path / 'missing'}' $digest"]
    body = (f"$digest = '{digest}'\n@(" + ", ".join(f"[string](Test-ModelFile {check})" for check in checks) +
            ") | ConvertTo-Json -Compress")
    passed, wrong_name, name_only, wrong_bytes, missing = _run_functions(tmp_path, ["Test-ModelFile"], body)
    assert (passed, name_only) == ("", "")  # The name check alone does not read the file.
    assert "is not the pinned blob" in wrong_name
    assert "not the pinned" in wrong_bytes and digest in wrong_bytes
    assert "not found" in missing


# --- readiness: GET /health answers ok once the model is loaded ----------------------------------

FAKE_SERVER = r'''
"""Stand-in for llama-server: records each request and answers its routes in llama-server's shape."""
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

with open(sys.argv[1], encoding="utf-8") as handle:
    CONFIG = json.load(handle)
NOT_FOUND = [404, {"error": {"code": 404, "message": "File Not Found", "type": "not_found_error"}}]
NO_MESSAGES = [400, {"error": {"code": 400, "message": "'messages' is required", "type": "invalid_request_error"}}]


class Handler(BaseHTTPRequestHandler):
    def _answer(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        with open(CONFIG["log"], "a", encoding="utf-8") as log:
            log.write(json.dumps({"method": self.command, "path": self.path,
                                  "authorization": self.headers.get("Authorization"),
                                  "body": body.decode("utf-8")}) + "\n")
        status, payload = CONFIG["routes"].get(f"{self.command} {self.path}", NOT_FOUND)
        if self.command == "POST" and status == 200:
            try:
                request = json.loads(body)
            except ValueError:
                request = None
            if not isinstance(request, dict) or "messages" not in request:
                status, payload = NO_MESSAGES
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    do_GET = do_POST = do_PUT = do_DELETE = _answer

    def log_message(self, *args):
        pass


server = HTTPServer(("127.0.0.1", 0), Handler)
print(server.server_address[1], flush=True)
server.serve_forever()
'''


@contextlib.contextmanager
def _fake_server(tmp_path, routes: dict):
    """A loopback HTTP server in its own process, standing in for llama-server."""
    script, config, log = tmp_path / "fake_llama_server.py", tmp_path / "fake_llama_server.json", tmp_path / "up.jsonl"
    script.write_text(FAKE_SERVER, encoding="utf-8")
    config.write_text(json.dumps({"log": str(log), "routes": routes}), encoding="utf-8")
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


@pytest.fixture
def fake_server(tmp_path):
    with _fake_server(tmp_path, ANSWERS) as server:
        yield server


@pytest.mark.parametrize(("health", "ready"), [
    ([200, {"status": "ok"}], True),
    (LOADING, False),                     # Listening while the model loads: not ready.
    ([200, {"status": "loading"}], False),
    (None, False),                        # Nothing listens.
])
def test_readiness_is_health_ok_not_a_listening_port(tmp_path, health, ready):
    if health is None:
        [port] = _free_ports(1)
        assert _run_functions(tmp_path, ["Test-HostHealth"], f"Test-HostHealth {port} | ConvertTo-Json") is False
        return
    with _fake_server(tmp_path, {"GET /health": health}) as server:
        result = _run_functions(tmp_path, ["Test-HostHealth"], f"Test-HostHealth {server.port} | ConvertTo-Json")
        assert result is ready
        assert [(item["method"], item["path"]) for item in server.seen()] == [("GET", "/health")]


# --- gateway request handling against a fake llama-server ---------------------------------------


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


def _connect(rig, credential: str):
    """The study's client: connect() on the llamacpp profile, pointed at the gateway by LLAMACPP_ENDPOINT_URL."""
    environment = {TOKEN_NAME: credential, "LLAMACPP_ENDPOINT_URL": f"http://127.0.0.1:{rig.port}{CHAT_PATH}"}
    return connect(LLAMACPP, CallBudget(4), features=FeatureChoices(mia_scoring=True), environment=environment,
                   enforce_preflight=False)  # Host RAM is the script's gate; these tests load no model.


def test_gateway_requires_the_bearer_and_forwards_only_the_llama_server_routes(rig, fake_server, tmp_path):
    _write_state(rig, host_pid=fake_server.process.pid, host_port=fake_server.port)
    url = f"http://127.0.0.1:{rig.port}"
    request = {"model": "gemma3:4b", "messages": [{"role": "user", "content": "Synthetic prompt."}]}
    with _gateway(rig, tmp_path) as gateway, _session() as session:
        # Without the bearer: 401 on every route, served or not, and nothing reaches llama-server.
        for method, path in (*ROUTES, ("POST", "/completion"), ("GET", "/slots")):
            answer = session.request(method, url + path, json=request if method == "POST" else None, timeout=30)
            assert (answer.status_code, answer.json()) == (401, {"error": "unauthorized"}), (method, path)
        for header in (f"Basic {TOKEN}", "Bearer not-the-token", TOKEN):
            assert session.get(url + "/props", headers={"Authorization": header}, timeout=30).status_code == 401
        with pytest.raises(RequestError) as caught:
            _connect(rig, "not-the-token").generate("Synthetic prompt.")
        assert caught.value.status_code == 401
        assert fake_server.seen() == []

        # With the bearer: generation returns the per-token top-5 alternatives through the profile's mapping.
        response = _connect(rig, TOKEN).generate("Synthetic prompt.")
        assert response.content == "OK." and response.mia_unavailable_reason is None
        assert [len(entry.top_logprobs) for entry in response.token_evidence] == [5, 5]
        [forwarded] = fake_server.seen()
        assert (forwarded["method"], forwarded["path"], forwarded["authorization"]) == ("POST", CHAT_PATH, None)
        sent = json.loads(forwarded["body"])
        assert (sent["logprobs"], sent["top_logprobs"]) == (True, 5)
        assert sent["messages"] == [{"role": "user", "content": "Synthetic prompt."}]

        # The read-only routes pass through unchanged, the Authorization header stripped.
        bearer = {"Authorization": f"Bearer {TOKEN}"}
        for method, path in ROUTES[1:]:
            answer = session.request(method, url + path, headers=bearer, timeout=30)
            assert (answer.status_code, answer.json()) == (200, ANSWERS[f"{method} {path}"][1]), (method, path)
        model_path = session.get(url + "/props", headers=bearer, timeout=30).json()["model_path"]
        assert PureWindowsPath(model_path).name == BLOB  # The runner's digest check (7.4) reads this.
        assert [(item["method"], item["path"], item["authorization"]) for item in fake_server.seen()[1:]] == [
            *((method, path, None) for method, path in ROUTES[1:]), ("GET", "/props", None)]

        # Any other method or path is refused with the bearer and never forwarded.
        before = len(fake_server.seen())
        for method, path in UNSERVED:
            answer = session.request(method, url + path, json=request, headers=bearer, timeout=30)
            assert answer.status_code == 404, (method, path, answer.status_code)
        assert len(fake_server.seen()) == before

        # llama-server's own refusals pass through unchanged.
        refused = session.post(url + CHAT_PATH, json={"model": "gemma3:4b"}, headers=bearer, timeout=30)
        assert refused.status_code == 400 and refused.json()["error"]["type"] == "invalid_request_error"

    log = _gateway_log(rig)
    for line in (f"POST {CHAT_PATH} 200", "GET /props 200", "GET /v1/models 200", "GET /health 200",
                 "GET /health 401", "POST /props 404", f"POST {CHAT_PATH} 400"):
        assert line in log, line
    for text in (log, gateway.out.read_text(encoding="utf-8", errors="replace"),
                 gateway.err.read_text(encoding="utf-8", errors="replace"), json.dumps(_read_state(rig))):
        assert TOKEN not in text


def test_gateway_refuses_to_run_without_the_token(rig, fake_server):
    _write_state(rig, host_pid=fake_server.process.pid, host_port=fake_server.port)
    rig.env_file.write_text(f"{TOKEN_NAME}=\n", encoding="utf-8")
    result = _script("gateway", *rig.args, timeout=60)
    assert result.returncode != 0 and f"{TOKEN_NAME} missing" in result.stdout + result.stderr
    assert not _listening(rig.port)


def test_gateway_idle_stop_stops_the_host_tree_and_logs_the_leftover_check(rig, fake_server, tmp_path):
    host = _tree(fake_server.handle)
    _write_state(rig, host_pid=fake_server.process.pid, host_port=fake_server.port, idle_minutes=0.05)  # 3 s
    with _gateway(rig, tmp_path) as gateway:
        assert gateway.process.wait(timeout=90) == 0
    assert _alive(host) == [] and not _listening(rig.port) and not _listening(fake_server.port)
    state = _read_state(rig)
    assert state["stop_reason"] == "idle_timeout" and state["stopped_at"]
    log = _gateway_log(rig)
    assert "idle timeout" in log
    [check] = [line for line in log.splitlines() if "leftover check" in line]
    assert "no host or gateway process remains" in check
    assert str(fake_server.port) in check and str(rig.port) in check


def test_gateway_stops_itself_and_logs_the_leftover_check_when_the_host_exits(rig, fake_server, tmp_path):
    _write_state(rig, host_pid=fake_server.process.pid, host_port=fake_server.port)
    with _gateway(rig, tmp_path) as gateway:
        _kill_tree(fake_server.handle)
        assert gateway.process.wait(timeout=90) == 0
    assert not _listening(rig.port)
    assert _read_state(rig)["stop_reason"] == "host_exited"
    log = _gateway_log(rig)
    assert "host exited" in log
    assert any("leftover check: no host or gateway process remains" in line for line in log.splitlines())


def test_gateway_leftover_check_names_a_listener_that_remains(rig, tmp_path):
    sleeper = subprocess.Popen([sys.executable, "-I", "-c", "import time; time.sleep(120)"])
    handle = psutil.Process(sleeper.pid)
    try:
        with socket.socket() as stray:
            stray.bind(("127.0.0.1", rig.host_port))
            stray.listen()
            _write_state(rig, host_pid=sleeper.pid, host_name=Path(handle.name()).stem, idle_minutes=0.05)
            with _gateway(rig, tmp_path) as gateway:
                assert gateway.process.wait(timeout=90) == 0
        assert _alive(_tree(handle)) == []
        assert f"leftover check: STILL LISTENING 127.0.0.1:{rig.host_port} pid {os.getpid()}" in _gateway_log(rig)
    finally:
        _kill_tree(handle)
        sleeper.wait(timeout=10)


# --- stop verification ---------------------------------------------------------------------------

#: The script's own pattern for a gateway (pwsh) command line.
GATEWAY_COMMAND = re.compile(r"serve_local_llamacpp\.ps1\"?\s+gateway\b")


@pytest.fixture
def quiet_host():
    """The stop and start tests need no real gateway of this script on the machine."""
    running = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        with contextlib.suppress(psutil.Error):
            if GATEWAY_COMMAND.search(subprocess.list2cmdline(process.info["cmdline"] or [])):
                running.append(process.info["pid"])
    if running:
        pytest.skip(f"a llama-server gateway runs on this machine (pid {running})")


def _sleeper(*extra: str) -> tuple[subprocess.Popen, psutil.Process]:
    process = subprocess.Popen([sys.executable, "-I", "-c", "import time; time.sleep(120)", *extra])
    return process, psutil.Process(process.pid)


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
    listener, handle = _sleeper()
    try:
        _write_state(rig, host_pid=None, listener_pid=listener.pid, listener_name=Path(handle.name()).stem)
        result = _script("stop", *rig.args)
        assert result.returncode == 0, result.stdout + result.stderr
        assert _alive(_tree(handle)) == []
    finally:
        _kill_tree(handle)
        listener.wait(timeout=10)


def test_stop_does_not_kill_an_unrelated_process_that_reused_the_recorded_pid(rig, quiet_host):
    other, handle = _sleeper()
    try:
        _write_state(rig, host_pid=other.pid, host_name="llama-server", listener_pid=other.pid,
                     listener_name="llama-server")  # The recorded name does not match.
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


def test_stop_reports_a_stray_llama_server_on_its_port_and_exits_non_zero(rig, quiet_host):
    [elsewhere] = _free_ports(1)
    stray, handle = _sleeper("llama-server.exe", "-m", "model.gguf", "--port", str(rig.host_port))
    other, other_handle = _sleeper("llama-server.exe", "--port", str(elsewhere))  # Another server's port.
    try:
        result = _script("stop", *rig.args)
        assert result.returncode == 1
        assert re.search(rf"STILL RUNNING host pid {stray.pid}\b", result.stdout), result.stdout
        assert not re.search(rf"pid {other.pid}\b", result.stdout), result.stdout
        assert stray.poll() is None and other.poll() is None  # Reported, never killed: not the recorded tree.
    finally:
        for process, item in ((stray, handle), (other, other_handle)):
            _kill_tree(item)
            process.wait(timeout=10)


# --- start refusals (nothing is loaded) ----------------------------------------------------------


@pytest.fixture
def dummy_install(tmp_path):
    """A llama.cpp folder and model blob that exist but are never run or loaded: every refusal comes first."""
    llama = tmp_path / "llama"
    llama.mkdir()
    (llama / "llama-server.exe").write_bytes(b"")
    model = tmp_path / "blobs" / BLOB
    model.parent.mkdir()
    model.write_bytes(b"")
    return SimpleNamespace(llama=str(llama), args=("-LlamaDir", str(llama), "-ModelPath", str(model)))


@pytest.mark.parametrize("address", ["0.0.0.0", "::", "::ffff:0.0.0.0", "0", "*", "localhost"])
def test_start_refuses_all_interfaces_and_names(rig, dummy_install, address):
    result = _script("start", "-BindAddress", address, *rig.args, *dummy_install.args)
    assert result.returncode != 0 and "Refusing bind address" in result.stdout + result.stderr
    assert not (rig.state / "state.json").exists()
    assert rig.env_file.read_text(encoding="utf-8") == f"{TOKEN_NAME}={TOKEN}\n"


def test_start_refuses_a_model_that_is_not_the_pinned_blob(rig, dummy_install, tmp_path, quiet_host):
    other = tmp_path / "model.gguf"
    other.write_bytes(b"")
    result = _script("start", *rig.args, "-LlamaDir", dummy_install.llama, "-ModelPath", str(other))
    assert result.returncode != 0 and "is not the pinned blob" in result.stdout + result.stderr
    assert not (rig.state / "state.json").exists()


def _ram_profile(tmp_path, sibling_port: int, per_load: int) -> tuple[dict, Path]:
    data = json.loads(PROFILE_FILE.read_text(encoding="utf-8"))
    data["host_requirements"]["min_free_ram_bytes"] = per_load
    data["full_distribution"]["endpoint"] = f"http://127.0.0.1:{sibling_port}"
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return data, path


@contextlib.contextmanager
def _python_listener():
    """A python process listening on an ephemeral loopback port, as the full-distribution host does."""
    code = ("import socket, time; s = socket.socket(); s.bind(('127.0.0.1', 0)); s.listen(); "
            "print(s.getsockname()[1], flush=True); time.sleep(120)")
    process = subprocess.Popen([sys.executable, "-I", "-c", code], stdout=subprocess.PIPE, text=True)
    handle = psutil.Process(process.pid)
    try:
        port = int(process.stdout.readline())
        _wait_for(lambda: _listening(port), 30, "the python listener")
        yield port
    finally:
        _kill_tree(handle)
        process.stdout.close()


def _refused_start(rig, dummy_install, profile_file) -> str:
    result = _script("start", *rig.args, *dummy_install.args, "-ProfilePath", str(profile_file),
                     "-RamWaitSeconds", "3")
    output = result.stdout + result.stderr
    assert result.returncode == 2, output
    assert "Nothing was loaded and no request was sent." in output
    assert not (rig.state / "state.json").exists() and not _listening(rig.host_port)
    assert rig.env_file.read_text(encoding="utf-8") == f"{TOKEN_NAME}={TOKEN}\n"
    return output


def test_ram_gate_requires_one_load_with_the_preflight_wording(rig, dummy_install, tmp_path, quiet_host):
    """One model host runs at a time, so the gate is the profile's min_free_ram_bytes, never doubled
    (user correction 2026-11-10)."""
    per_load = 2 ** 50  # 1 PiB: no host has it free.
    [sibling] = _free_ports(1)  # Nothing listens there.
    data, profile_file = _ram_profile(tmp_path, sibling, per_load)
    output = _refused_start(rig, dummy_install, profile_file)
    assert "Waiting for free RAM" in output  # Progress line while waiting.
    assert "--profile modal-gemma3-4b" in output and "on a machine with at least" in output
    fallbacks = tuple(data["host_requirements"]["fallback_profile_ids"])
    expected = _ram_recommendation(data["model_config"]["model"], 0, per_load, fallbacks)
    assert expected.split(" free, ", 1)[1] in output  # Same sentence as preflight_host, for one load.
    assert "one load" in output and "two loads" not in output


def test_a_resident_full_distribution_host_refuses_the_start_before_the_ram_gate(
        rig, dummy_install, tmp_path, quiet_host):
    """Only one model host at a time: a python process on the full-distribution port refuses the start."""
    per_load = 2 ** 50  # Without the sibling refusal the RAM gate refuses, so nothing is ever loaded here.
    with _python_listener() as sibling:
        _, profile_file = _ram_profile(tmp_path, sibling, per_load)
        output = _refused_start(rig, dummy_install, profile_file)
    assert "only one model host runs at a time" in output and f"port {sibling}" in output
    assert "serve_full_distribution.ps1 stop" in output
    assert "Waiting for free RAM" not in output  # Refused before the RAM gate.


def test_sibling_is_resident_only_when_a_python_process_listens_on_its_port(tmp_path):
    [free] = _free_ports(1)
    with _python_listener() as python_port:
        body = "\n".join((
            "$own = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Loopback, 0)",  # Owned by pwsh, not python.
            "$own.Start()",
            f"@((Test-SiblingResident {python_port}), (Test-SiblingResident {free}), "
            "(Test-SiblingResident $own.LocalEndpoint.Port), (Get-RamRequirement 6442450944)) | ConvertTo-Json -Compress"))
        python, nothing, other, need = _run_functions(
            tmp_path, ["Get-Listeners", "Test-SiblingResident", "Get-RamRequirement"], body)
    assert (python, nothing, other) == (True, False, False)
    assert need == {"per_load_bytes": 6442450944, "loads": 1, "required_bytes": 6442450944}  # Never doubled.


# --- helpers shared with the 6.2 lifecycle script --------------------------------------------------

#: Copied verbatim from scripts/hosts/serve_full_distribution.ps1, whose tests cover them; kept identical here.
SHARED_HELPERS = ("Read-State", "Write-State", "Read-Token", "Initialize-Token", "Get-ProcessTree", "Test-Tracked",
                  "Stop-Tree", "Get-Listeners", "Get-FreeRamBytes", "Format-Gib", "Wait-FreeRam", "Send-Response",
                  "Send-Error", "Read-Request", "Test-Token", "Disable-OutputInheritance")
DEFINITION_PROBE = r"""
param([string]$Script)
$ast = [System.Management.Automation.Language.Parser]::ParseFile($Script, [ref]$null, [ref]$null)
$found = [ordered]@{}
foreach ($node in $ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] }, $true)) {
    $found[$node.Name] = $node.Extent.Text
}
$found | ConvertTo-Json -Compress
"""


def test_shared_helpers_are_verbatim_copies_of_the_full_distribution_script(tmp_path):
    probe = tmp_path / "definition_probe.ps1"
    probe.write_text(DEFINITION_PROBE, encoding="utf-8")

    def definitions(script: Path) -> dict[str, str]:
        result = _pwsh("-File", str(probe), str(script))
        assert result.returncode == 0, result.stderr
        return {name: text.replace("\r\n", "\n") for name, text in json.loads(result.stdout).items()}

    ours, sibling = definitions(SCRIPT), definitions(SIBLING_SCRIPT)
    for name in SHARED_HELPERS:
        assert name in sibling, name
        assert ours.get(name) == sibling[name], name
