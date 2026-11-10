"""Task 6.4: the lifecycle script's batch route, gateway memory sampling and faulthandler launch.

``scripts/serve_full_distribution.ps1`` forwards ``POST /v1/full_distribution_batch`` besides the
single route, samples free physical memory into gateway.log (also while it waits for a long
host request), logs the host's 503 ``memory_low`` refusals and launches the host with
``python -X faulthandler`` and the profile's memory floor. The gateway runs against a fake
upstream host; no model is loaded. Needs PowerShell 7 on a Windows host; skipped elsewhere.
"""
import contextlib
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from types import SimpleNamespace

import psutil
import pytest

from recall_guard.core.full_distribution import BATCH_ENDPOINT_PATH, ENDPOINT_PATH, FullDistributionClient
from recall_guard.core.request_control import RequestError
from tests.scripts.test_serve_full_distribution_lifecycle import (
    DIGEST,
    EVIDENCE,
    FUNCTION_PROBE,
    PROFILE_FILE,
    PWSH,
    SCRIPT,
    TOKEN,
    TOKEN_NAME,
    _free_ports,
    _gateway,
    _kill_tree,
    _pwsh,
    _read_state,
    _session,
    _wait_for,
    _write_state,
)

pytestmark = pytest.mark.skipif(PWSH is None, reason="needs PowerShell 7 (pwsh) on a Windows host")
FLOOR = 1_610_612_736

FAKE_HOST = r'''
"""Stand-in for the full-distribution host: records each request and answers by route."""
import json
import sys
import time
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
        time.sleep(CONFIG["delay"])
        if CONFIG["refusal"] is not None:
            status, payload = 503, CONFIG["refusal"]
        elif self.path == CONFIG["batch_path"]:
            status, payload = 200, {"evidence": [CONFIG["evidence"]] * len(json.loads(body)["response_texts"])}
        else:
            status, payload = 200, CONFIG["evidence"]
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
MEMORY_LOW = {"error": {"code": "memory_low", "message": "free physical memory is below the floor"}}


@pytest.fixture
def rig(tmp_path):
    """Temporary state folder and token file plus two free loopback ports; the gateway samples every second."""
    env_file = tmp_path / "token.env"
    env_file.write_text(f"{TOKEN_NAME}={TOKEN}\n", encoding="utf-8")
    host_port, port = _free_ports(2)
    rig = SimpleNamespace(state=tmp_path / "state", env_file=env_file, host_port=host_port, port=port)
    rig.args = ("-StateDir", str(rig.state), "-EnvPath", str(env_file), "-HostPort", str(host_port),
                "-Port", str(port), "-MemorySampleSeconds", "1")
    return rig


@contextlib.contextmanager
def upstream(tmp_path, *, delay=0.0, refusal=None):
    script, config, log = tmp_path / "fake_host.py", tmp_path / "fake_host.json", tmp_path / "upstream.jsonl"
    script.write_text(FAKE_HOST, encoding="utf-8")
    config.write_text(json.dumps({"log": str(log), "evidence": EVIDENCE, "batch_path": BATCH_ENDPOINT_PATH,
                                  "delay": delay, "refusal": refusal}), encoding="utf-8")
    process = subprocess.Popen([sys.executable, "-I", str(script), str(config)], stdout=subprocess.PIPE, text=True)
    handle = psutil.Process(process.pid)
    try:
        port = int(process.stdout.readline())

        def seen() -> list[dict]:
            return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()] if log.exists() else []
        yield SimpleNamespace(process=process, port=port, seen=seen)
    finally:
        _kill_tree(handle)
        process.stdout.close()


def _client(rig, credential=TOKEN, timeout_s=30.0):
    return FullDistributionClient(f"http://127.0.0.1:{rig.port}", model_digest=DIGEST,
                                  credential_ref=f"env:{TOKEN_NAME}", timeout_s=timeout_s,
                                  environment={TOKEN_NAME: credential})


def _log_lines(rig) -> list[str]:
    return (rig.state / "gateway.log").read_text(encoding="utf-8").splitlines()


def _stamp(line: str) -> datetime:
    """The gateway log's round-trip ('o') timestamp, cut to microseconds."""
    text = line.split(" ", 1)[0].rstrip("Z")
    head, _, fraction = text.partition(".")
    return datetime.fromisoformat(f"{head}.{(fraction + '000000')[:6]}").replace(tzinfo=UTC)


def _memory_lines(rig) -> list[str]:
    return [line for line in _log_lines(rig) if " memory free_ram_bytes=" in line]


def test_gateway_forwards_the_batch_route_with_the_bearer_and_samples_memory(rig, tmp_path):
    payload = {"model_digest": DIGEST, "prompt_text": "Synthetic prompt.", "response_texts": [" A", " B"],
               "top_k_keep": 2}
    url = f"http://127.0.0.1:{rig.port}"
    with upstream(tmp_path) as host:
        _write_state(rig, host_pid=host.process.pid, host_port=host.port, memory_floor_bytes=FLOOR)
        with _gateway(rig, tmp_path), _session() as session:
            refused = session.post(url + BATCH_ENDPOINT_PATH, json=payload, timeout=30)
            assert (refused.status_code, refused.json()) == (401, {"error": "unauthorized"})
            assert host.seen() == []
            batch = _client(rig).evidence_batch("Synthetic prompt.", [" A", " B"], top_k_keep=2)
            assert [item.to_dict() for item in batch] == [EVIDENCE, EVIDENCE]
            [forwarded] = host.seen()
            assert (forwarded["method"], forwarded["path"], forwarded["authorization"]) == (
                "POST", BATCH_ENDPOINT_PATH, None)
            assert json.loads(forwarded["body"]) == payload
            assert _client(rig).evidence("Synthetic prompt.", " A", top_k_keep=2).to_dict() == EVIDENCE
            bearer = {"Authorization": f"Bearer {TOKEN}"}
            for method, path in (("GET", BATCH_ENDPOINT_PATH), ("POST", BATCH_ENDPOINT_PATH + "/"),
                                 ("POST", BATCH_ENDPOINT_PATH + "?x=1"), ("POST", "/v1/full_distribution_batches")):
                answer = session.request(method, url + path, json=payload, headers=bearer, timeout=30)
                assert answer.status_code == 404, (method, path)
            assert [entry["path"] for entry in host.seen()] == [BATCH_ENDPOINT_PATH, ENDPOINT_PATH]
            time.sleep(2.5)  # At least two more one-second samples while idle.
    lines = _log_lines(rig)
    assert f"POST {BATCH_ENDPOINT_PATH} 200" in "\n".join(lines) and f"POST {BATCH_ENDPOINT_PATH} 401" in "\n".join(lines)
    assert any(f"memory_floor={FLOOR}" in line and "listening" in line for line in lines)
    samples = _memory_lines(rig)
    assert len(samples) >= 3 and all(f"floor_bytes={FLOOR}" in line for line in samples)
    assert all(int(line.split("free_ram_bytes=", 1)[1].split()[0]) > 0 for line in samples)
    assert TOKEN not in "\n".join(lines) and TOKEN not in json.dumps(_read_state(rig))


def test_gateway_keeps_sampling_memory_while_it_waits_for_a_slow_host(rig, tmp_path):
    with upstream(tmp_path, delay=4.0) as host:
        _write_state(rig, host_pid=host.process.pid, host_port=host.port, memory_floor_bytes=FLOOR)
        with _gateway(rig, tmp_path):
            started = datetime.now(UTC)
            assert len(_client(rig).evidence_batch("p", [" A"], top_k_keep=2)) == 1
            finished = datetime.now(UTC)
    during = [line for line in _memory_lines(rig) if started.timestamp() + 0.5 < _stamp(line).timestamp()
              < finished.timestamp() - 0.5]
    assert len(during) >= 2, _log_lines(rig)


def test_gateway_logs_the_hosts_memory_low_refusal_and_passes_the_503_through(rig, tmp_path):
    with upstream(tmp_path, refusal=MEMORY_LOW) as host:
        _write_state(rig, host_pid=host.process.pid, host_port=host.port, memory_floor_bytes=FLOOR)
        with _gateway(rig, tmp_path):
            for call in (lambda: _client(rig).evidence_batch("p", [" A"]), lambda: _client(rig).evidence("p", " A")):
                with pytest.raises(RequestError) as caught:
                    call()
                assert (caught.value.status_code, caught.value.code, caught.value.retryable) == (503, "host_busy", True)
                assert "memory_low" in caught.value.detail
    refusals = [line for line in _log_lines(rig) if "refused memory_low" in line]
    assert len(refusals) == 2 and BATCH_ENDPOINT_PATH in refusals[0] and ENDPOINT_PATH in refusals[1]


def test_gateway_takes_the_floor_from_the_profile_when_the_state_has_none(rig, tmp_path):
    log = rig.state / "gateway.log"
    with upstream(tmp_path) as host:
        _write_state(rig, host_pid=host.process.pid, host_port=host.port)
        with _gateway(rig, tmp_path):
            # The gateway listens first, then reads the profile and writes its first line.
            _wait_for(lambda: log.exists() and "memory_floor=" in log.read_text(encoding="utf-8"), 30,
                      "the gateway's listening line")
    assert any(f"memory_floor={FLOOR}" in line for line in _log_lines(rig))


# --- functions extracted from the script ---------------------------------------------------------

PROBE_FUNCTIONS = ("Get-HostArguments", "Test-MemoryLowRefusal", "Read-HostProfile")


def _run(tmp_path, body: str) -> object:
    probe = tmp_path / "guard_probe.ps1"
    header = FUNCTION_PROBE.replace("'Get-RamRequirement', 'Wait-FreeRam', 'Disable-OutputInheritance'",
                                    ", ".join(f"'{name}'" for name in PROBE_FUNCTIONS))
    assert header != FUNCTION_PROBE
    probe.write_text(header + "$HostScript = 'scripts/serve_full_distribution.py'\n" + body + "\n", encoding="utf-8")
    result = _pwsh("-File", str(probe), str(SCRIPT))
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_the_host_is_launched_with_faulthandler_and_the_memory_floor(tmp_path):
    arguments = _run(tmp_path, f"ConvertTo-Json -Compress -InputObject @(Get-HostArguments 11580 {FLOOR})")
    assert arguments == ["run", "--no-sync", "python", "-X", "faulthandler", "scripts/serve_full_distribution.py",
                         "--port", "11580", "--memory-floor-bytes", str(FLOOR)]


def test_read_host_profile_returns_the_recorded_floor_or_one_quarter_of_the_minimum(tmp_path):
    data = json.loads(PROFILE_FILE.read_text(encoding="utf-8"))
    del data["host_requirements"]["memory_floor_bytes"]
    data["host_requirements"]["min_free_ram_bytes"] = 8 * 1024 ** 3
    derived = tmp_path / "derived.json"
    derived.write_text(json.dumps(data), encoding="utf-8")
    recorded, quarter = _run(tmp_path, "\n".join((
        f"$ProfileFile = '{PROFILE_FILE}'", "$a = (Read-HostProfile).memory_floor_bytes",
        f"$ProfileFile = '{derived}'", "$b = (Read-HostProfile).memory_floor_bytes",
        "ConvertTo-Json -Compress -InputObject @($a, $b)")))
    assert (recorded, quarter) == (FLOOR, 2 * 1024 ** 3)


@pytest.mark.parametrize(("status", "body", "expected"), [
    (503, json.dumps(MEMORY_LOW), True),
    (503, json.dumps({"error": {"code": "busy"}}), False),
    (503, json.dumps({"error": "memory_low"}), False),
    (503, "not json", False),
    (503, "", False),
    (500, json.dumps(MEMORY_LOW), False),
    (200, json.dumps({"evidence": []}), False),
])
def test_only_the_hosts_memory_low_503_counts_as_a_refusal(tmp_path, status, body, expected):
    encoded = ", ".join(str(byte) for byte in body.encode("utf-8"))
    result = _run(tmp_path, f"ConvertTo-Json -Compress -InputObject @(Test-MemoryLowRefusal {status} "
                            f"([byte[]]@({encoded})))")
    assert result == [expected]


def test_start_launches_the_host_through_the_faulthandler_arguments(tmp_path):
    probe = tmp_path / "start_probe.ps1"
    probe.write_text(r"""
param([string]$Script)
$ast = [System.Management.Automation.Language.Parser]::ParseFile($Script, [ref]$null, [ref]$null)
$start = $ast.Find({ param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and
                    $n.Name -eq 'Start-Serving' }, $true)
$start.Extent.Text | ConvertTo-Json -Compress
""", encoding="utf-8")
    result = _pwsh("-File", str(probe), str(SCRIPT))
    assert result.returncode == 0, result.stderr
    text = json.loads(result.stdout)
    assert "Get-HostArguments $HostPort $hostProfile.memory_floor_bytes" in text
    assert "-ArgumentList $hostArguments" in text
    assert "memory_floor_bytes = $hostProfile.memory_floor_bytes" in text
    assert "'-MemorySampleSeconds'" in text
    assert "'run', '--no-sync', 'python', $HostScript" not in text  # The launch without faulthandler is gone.
