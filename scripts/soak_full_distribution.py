"""Soak of the full-distribution host through its bearer gateway (task 6.4, Requirement 11.1).

Native Windows host-side check, run from the checkout after
``pwsh -File scripts/hosts/serve_full_distribution.ps1 start``::

    uv run --no-sync python scripts/soak_full_distribution.py [--duration-s 1800] [--interval-s 30]
        [--sample-s 5] [--stop-after]

For ``--duration-s`` it sends one batch request (``SOAK_ITEM``: one synthetic prompt with several
responses) every ``--interval-s`` through the gateway with ``FullDistributionClient.evidence_batch``,
and every ``--sample-s`` it records free physical memory and whether the recorded host process
lives. The bearer is read by name (``LOCAL_MODEL_API_KEY``) from the environment, else from the
root ``.env``; its value is never printed or logged, and the receipt is checked not to contain it
before it is written. From the host state folder it records the host's launch (``python -X
faulthandler``), its memory floor and the memory lines the host and the gateway logged during the
soak. ``--stop-after`` then runs the lifecycle script's ``stop`` and records that nothing listens
on the host or gateway port and that no host or gateway process remains. The receipt
(``receipt.json``, ``summary.md``) goes to runs/finance-sim/hosts/full-distribution/soak/ and
holds timestamps, sizes, status codes, timings and memory readings only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # Run as a file; the host helpers live in scripts/, the client in recall_guard/.
    sys.path.insert(0, str(ROOT))
from recall_guard.core.full_distribution import FullDistributionClient  # noqa: E402
from recall_guard.core.request_control import RequestError  # noqa: E402
from scripts.hosts.serve_full_distribution import HOST, MODEL_DIGEST, PORT, _free_ram, _listeners, _now  # noqa: E402

STATE_DIR = ROOT / 'runs/finance-sim/hosts/full-distribution'
SCRIPT = ROOT / 'scripts/hosts/serve_full_distribution.ps1'
CREDENTIAL_NAME = 'LOCAL_MODEL_API_KEY'
GATEWAY_URL = f'http://{HOST}:{PORT + 1}'  # The gateway listens on the host's loopback port plus one.
SOAK_ITEM = {  # Synthetic, no licensed text: one prompt carrying four responses.
    'name': 'soak',
    'prompt_text': 'Name the macro regime of the month in one word, then give one reason.\nAnswer:',
    'response_texts': [' Expansion. Output and employment rose together.',
                       ' Slowdown. New orders fell for a third month.',
                       ' Recovery. Credit spreads narrowed.',
                       ' Stagflation. Prices rose while output fell.'],
}
#: The lifecycle script's own patterns: the host's command line (uv and python) and the gateway's (pwsh).
HOST_COMMAND = re.compile(r'(?<![\w.-])serve_full_distribution\.py\b')
GATEWAY_COMMAND = re.compile(r'serve_full_distribution\.ps1"?\s+gateway\b')


def read_credential(name: str, env_file: Path, environ: Mapping[str, str]) -> tuple[str | None, str | None]:
    """The bearer named ``name`` and where it came from: the environment, else ``env_file`` (as the gateway reads it)."""
    value = environ.get(name)
    if value:
        return value, 'environment'
    try:
        lines = Path(env_file).read_text(encoding='utf-8-sig').splitlines()
    except OSError:
        return None, None
    pattern = re.compile(rf'^\s*{re.escape(name)}\s*=\s*(.*?)\s*$')
    for line in lines:
        if (match := pattern.match(line)) and (value := match.group(1).strip('"\'')):
            return value, '.env'
    return None, None


def _has_faulthandler(command: list[str]) -> bool:
    return '-Xfaulthandler' in command or any(
        first == '-X' and second == 'faulthandler' for first, second in zip(command, command[1:], strict=False))


def _process(pid):
    """``(handle, [handle and its descendants])`` for a live pid, else ``(None, [])``; the handle is immune to PID reuse."""
    import psutil
    try:
        handle = psutil.Process(int(pid))
        return handle, [handle, *handle.children(recursive=True)]
    except (TypeError, ValueError, psutil.Error):
        return None, []


def _host_snapshot(state_folder: Path):
    """The recorded host and its launch command lines, plus handles taken now (immune to PID reuse).

    The host is the process that listens (``listener_pid``, recorded at readiness by the lifecycle
    script); its uv launcher (``host_pid``) can exit while it still serves, so liveness follows the
    listener when one is recorded and the launcher's handle is kept apart. Returns ``(snapshot,
    host handle, launcher handle or None)``.
    """
    import psutil
    try:
        state = json.loads((state_folder / 'state.json').read_text(encoding='utf-8-sig'))
    except (OSError, ValueError):
        return None, None, None
    pid, listener_pid = state.get('host_pid'), state.get('listener_pid')
    snapshot = {'host_pid': pid, 'listener_pid': listener_pid, 'gateway_pid': state.get('gateway_pid'),
                'memory_floor_bytes': state.get('memory_floor_bytes'), 'host_arguments': state.get('host_arguments'),
                'faulthandler': False, 'alive_at_start': False}
    launcher, tree = _process(pid)
    listener, listener_tree = _process(listener_pid) if listener_pid and listener_pid != pid else (None, [])
    handle = listener or launcher
    snapshot['alive_at_start'] = handle is not None
    try:
        snapshot['faulthandler'] = any(_has_faulthandler(process.cmdline()) for process in [*tree, *listener_tree])
    except psutil.Error:
        pass
    return snapshot, handle, (launcher if listener is not None else None)


def _alive(handle) -> bool | None:
    import psutil
    if handle is None:
        return None
    try:
        return handle.is_running() and handle.status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


def _sizes(state_folder: Path | None) -> dict[str, int]:
    if state_folder is None:
        return {}
    return {name: (state_folder / name).stat().st_size if (state_folder / name).exists() else 0
            for name in ('gateway.log', 'host.out.log')}


def _appended(state_folder: Path, name: str, offset: int) -> list[str]:
    try:
        with (state_folder / name).open('rb') as handle:
            handle.seek(offset)
            return handle.read().decode('utf-8', errors='replace').splitlines()
    except OSError:
        return []


class _Sampler(threading.Thread):
    """Free physical memory and host liveness at 0, ``seconds``, 2 ``seconds``, ... and once more when stopped.

    The schedule is fixed, so timer slack does not accumulate over a long soak.
    """

    def __init__(self, seconds: float, started: float, handle, launcher=None) -> None:
        super().__init__(name='soak-memory', daemon=True)
        self.seconds, self.started, self.handle, self.launcher = seconds, started, handle, launcher
        self.samples: list[dict] = []
        self.halt = threading.Event()

    def _sample(self) -> None:
        sample = {'t_s': round(time.perf_counter() - self.started, 3), 'at': _now(),
                  'free_ram_bytes': _free_ram(), 'host_alive': _alive(self.handle)}
        if self.launcher is not None:  # The uv launcher, apart from the listener that serves.
            sample['launcher_alive'] = _alive(self.launcher)
        self.samples.append(sample)

    def run(self) -> None:
        due = 0
        while True:
            self._sample()
            due += 1
            if self.halt.wait(max(0.0, self.started + due * self.seconds - time.perf_counter())):
                self._sample()  # The closing reading at the end of the soak.
                return


def _request(client: FullDistributionClient, index: int, offset: float, top_k_keep: int, timeout_s: float) -> dict:
    record = {'index': index, 'at': _now(), 'offset_s': round(offset, 3)}
    started = time.perf_counter()
    try:
        batch = client.evidence_batch(SOAK_ITEM['prompt_text'], SOAK_ITEM['response_texts'], top_k_keep=top_k_keep,
                                      timeout_s=timeout_s)
        body = json.dumps([item.to_dict() for item in batch], allow_nan=False).encode('utf-8')
        record |= {'status': 'ok', 'status_code': 200, 'retryable': False, 'responses': len(batch),
                   'positions': sum(len(item.logprob) for item in batch), 'evidence_bytes': len(body),
                   'evidence_sha256': hashlib.sha256(body).hexdigest()}
    except RequestError as error:  # Only the code and status are recorded, never the host's message.
        record |= {'status': error.code or 'request_error', 'status_code': error.status_code,
                   'retryable': bool(error.retryable)}
    except ValueError:  # The client refused the credential before any request (a fixed, value-free message).
        record |= {'status': 'refused_before_request', 'status_code': None, 'retryable': False}
    record['seconds'] = round(time.perf_counter() - started, 3)
    return record


def soak(base_url: str, *, credential_name: str, environment: Mapping[str, str], duration_s: float,
         interval_s: float, sample_s: float, state_folder: Path | None = None, top_k_keep: int = 100,
         timeout_s: float = 600.0, model_digest: str = MODEL_DIGEST, credential_source: str = 'environment') -> dict:
    """Run the soak and return its receipt (with ``outcome``); nothing is started or stopped here.

    Requests are due at 0, ``interval_s``, 2 ``interval_s``, ... before ``duration_s``; a request
    that is late (the previous one took longer than the interval) starts at once. Memory is
    sampled until ``duration_s`` has passed.
    """
    client = FullDistributionClient(base_url, model_digest=model_digest, credential_ref=credential_name,
                                    timeout_s=timeout_s, environment=environment)
    host, handle, launcher = _host_snapshot(state_folder) if state_folder is not None else (None, None, None)
    offsets = _sizes(state_folder)
    receipt = {'schema': 'finance-sim-full-distribution-soak/1', 'task': '6.4', 'started_at': _now(),
               'gateway': base_url, 'model_digest': model_digest,
               'credential': {'name': credential_name, 'source': credential_source},
               'settings': {'duration_s': duration_s, 'interval_s': interval_s, 'sample_s': sample_s,
                            'top_k_keep': top_k_keep, 'timeout_s': timeout_s,
                            'item': {'name': SOAK_ITEM['name'], 'prompt_chars': len(SOAK_ITEM['prompt_text']),
                                     'response_chars': [len(text) for text in SOAK_ITEM['response_texts']]}},
               'state_folder': None if state_folder is None else str(state_folder), 'host': host}
    started = time.perf_counter()  # High resolution; time.monotonic ticks at about 16 ms on Windows.
    sampler = _Sampler(sample_s, started, handle, launcher)
    sampler.start()
    requests = []
    try:
        for index in range(math.ceil(duration_s / interval_s)):
            while (wait := started + index * interval_s - time.perf_counter()) > 0:
                time.sleep(wait)
            requests.append(_request(client, index, time.perf_counter() - started, top_k_keep, timeout_s))
        while (rest := started + duration_s - time.perf_counter()) > 0:
            time.sleep(rest)  # Memory samples cover the whole duration.
    finally:
        sampler.halt.set()
        sampler.join(max(5.0, 2 * sample_s))
    receipt |= {'finished_at': _now(), 'requests': requests, 'memory': sampler.samples}
    if host is not None:
        host['alive_throughout'] = all(sample['host_alive'] for sample in sampler.samples)
        # Recorded, not a failure: the listener that serves is the host (lifecycle script, task 6.4).
        host['launcher_alive_throughout'] = (all(sample['launcher_alive'] for sample in sampler.samples)
                                             if launcher is not None else None)
        gateway = _appended(state_folder, 'gateway.log', offsets['gateway.log'])
        hosted = _appended(state_folder, 'host.out.log', offsets['host.out.log'])
        receipt['log_lines'] = {
            'gateway_memory_samples': sum(' memory free_ram_bytes=' in line for line in gateway),
            'gateway_memory_low_refusals': sum('refused memory_low' in line for line in gateway),
            'host_memory_samples': sum('"memory_sample"' in line for line in hosted),
            'host_refusals': sum('"refused"' in line for line in hosted)}
    receipt['summary'] = _summary(receipt, duration_s, interval_s)
    receipt['outcome'] = outcome(receipt)
    return receipt


def _summary(receipt: dict, duration_s: float, interval_s: float) -> dict:
    requests, memory = receipt['requests'], receipt['memory']
    seconds = [entry['seconds'] for entry in requests]
    free = [sample['free_ram_bytes'] for sample in memory]
    floor = (receipt.get('host') or {}).get('memory_floor_bytes')
    return {'requests_expected': math.ceil(duration_s / interval_s), 'requests_sent': len(requests),
            'ok': sum(entry['status'] == 'ok' for entry in requests),
            'by_status': dict(Counter(entry['status'] for entry in requests)),
            'distinct_evidence': len({entry['evidence_sha256'] for entry in requests if entry['status'] == 'ok'}),
            'seconds_min': min(seconds, default=None), 'seconds_median': statistics.median(seconds) if seconds else None,
            'seconds_max': max(seconds, default=None), 'late_requests': sum(
                entry['offset_s'] > entry['index'] * interval_s + 1.0 for entry in requests),
            'memory_samples': len(memory), 'min_free_ram_bytes': min(free, default=None),
            'max_free_ram_bytes': max(free, default=None),
            'samples_below_floor': None if not floor else sum(value < floor for value in free)}


def outcome(receipt: dict) -> dict:
    """``{"passed", "failures"}`` from the requests, the memory samples, the host checks and any teardown."""
    summary, failures = receipt['summary'], []
    if summary['requests_sent'] < summary['requests_expected']:
        failures.append(f'sent {summary["requests_sent"]} of {summary["requests_expected"]} requests')
    failed = summary['requests_sent'] - summary['ok']
    if failed:
        codes = ', '.join(sorted(status for status in summary['by_status'] if status != 'ok'))
        failures.append(f'{failed} of {summary["requests_sent"]} requests failed: {codes}')
    if summary['distinct_evidence'] > 1:
        failures.append('the evidence for the same batch changed between requests')
    if not summary['memory_samples']:
        failures.append('no memory samples')
    if receipt.get('state_folder') is not None:
        host = receipt.get('host')
        if host is None:
            failures.append(f'no host state in {receipt["state_folder"]}')
        else:
            if not host['alive_throughout']:
                failures.append('the host process exited during the soak')
            if not host['faulthandler']:
                failures.append('the host was not launched with python -X faulthandler')
            if not receipt['log_lines']['gateway_memory_samples']:
                failures.append('the gateway logged no memory samples during the soak')
            if not receipt['log_lines']['host_memory_samples']:
                failures.append('the host logged no memory samples during the soak')
    if 'teardown' in receipt:
        failures += teardown_failures(receipt['teardown'])
    return {'passed': not failures, 'failures': failures}


def _pwsh() -> str:
    installed = Path(os.environ.get('ProgramFiles', r'C:\Program Files')) / 'PowerShell/7/pwsh.exe'
    found = str(installed) if installed.is_file() else shutil.which('pwsh')
    if not found:
        raise RuntimeError('PowerShell 7 (pwsh) is needed to run the lifecycle stop')
    return found


def _serving_processes() -> list[dict]:
    import psutil
    found = []
    for process in psutil.process_iter(['pid', 'name', 'cmdline']):
        line = subprocess.list2cmdline(process.info['cmdline'] or [])
        kind = 'gateway' if GATEWAY_COMMAND.search(line) else 'host' if HOST_COMMAND.search(line) else None
        if kind and process.info['pid'] != os.getpid():
            found.append({'kind': kind, 'pid': process.info['pid'], 'name': process.info['name']})
    return found


def teardown(*, state_dir: Path, host_port: int = PORT, port: int = PORT + 1, script: Path = SCRIPT) -> dict:
    """Run the lifecycle script's ``stop`` and check that no listener and no host or gateway process remains."""
    command = [_pwsh(), '-NoProfile', '-NonInteractive', '-File', str(script), 'stop', '-StateDir', str(state_dir),
               '-HostPort', str(host_port), '-Port', str(port)]
    for attempt in (1, 2):  # The Store pwsh alias can refuse one launch (tasks.md Implementation Notes).
        try:
            result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace',
                                    timeout=180)
            break
        except PermissionError:
            if attempt == 2:
                raise
            time.sleep(1.0)
    return {'script': str(script), 'action': 'stop', 'exit_code': result.returncode,
            'output': (result.stdout + result.stderr).strip().splitlines()[-6:],
            'listeners': _listeners((host_port, port)), 'processes': _serving_processes(), 'checked_at': _now()}


def teardown_failures(result: Mapping) -> list[str]:
    failures = []
    if result['exit_code'] != 0:
        failures.append(f'the lifecycle stop exited with {result["exit_code"]}')
    if result['listeners'] or result['processes']:
        failures.append(f'after stop, listeners {result["listeners"]} and processes {result["processes"]} remain')
    return failures


def _summary_markdown(receipt: dict) -> str:
    summary, outcome_ = receipt['summary'], receipt['outcome']
    settings, host = receipt['settings'], receipt.get('host') or {}

    def gib(value):
        return 'n/a' if value is None else f'{value / 1024 ** 3:.2f} GiB'

    lines = ['# Soak of the full-distribution host (task 6.4)', '',
             f'- Outcome: **{"passed" if outcome_["passed"] else "failed"}**'
             + (f' ({"; ".join(outcome_["failures"])})' if outcome_['failures'] else ''),
             f'- Started {receipt["started_at"]}, finished {receipt.get("finished_at")}',
             f'- Gateway {receipt["gateway"]}, bearer `{receipt["credential"]["name"]}` by name '
             f'(from {receipt["credential"]["source"]}), model `{receipt["model_digest"]}`',
             f'- One batch request every {settings["interval_s"]} s for {settings["duration_s"]} s '
             f'({len(SOAK_ITEM["response_texts"])} responses to one prompt, top_k_keep {settings["top_k_keep"]}, '
             f'timeout {settings["timeout_s"]} s); free memory sampled every {settings["sample_s"]} s', '',
             '| Measure | Value |', '|---|---|',
             f'| Requests sent / expected | {summary["requests_sent"]} / {summary["requests_expected"]} |',
             f'| Requests by status | {summary["by_status"]} |',
             f'| Distinct evidence across requests | {summary["distinct_evidence"]} |',
             f'| Request seconds (min / median / max) | {summary["seconds_min"]} / {summary["seconds_median"]} / '
             f'{summary["seconds_max"]} |',
             f'| Memory samples | {summary["memory_samples"]} |',
             f'| Free memory (min / max) | {gib(summary["min_free_ram_bytes"])} / {gib(summary["max_free_ram_bytes"])} |',
             f'| Host memory floor | {gib(host.get("memory_floor_bytes"))} |',
             f'| Samples below the floor | {summary["samples_below_floor"]} |']
    if host:
        served = (f'Host listener pid {host["listener_pid"]}' if host.get('listener_pid') not in (None, host['host_pid'])
                  else f'Host pid {host["host_pid"]}')
        lines += [f'| {served} alive throughout | {host.get("alive_throughout")} |',
                  *([f'| Host launcher (uv) pid {host["host_pid"]} alive throughout | '
                     f'{host["launcher_alive_throughout"]} |'] if host.get('launcher_alive_throughout') is not None else []),
                  f'| Host launched with python -X faulthandler | {host["faulthandler"]} |',
                  *[f'| Log lines during the soak: {name} | {value} |'
                    for name, value in receipt.get('log_lines', {}).items()]]
    if 'teardown' in receipt:
        teardown_ = receipt['teardown']
        lines += [f'| Lifecycle stop exit code | {teardown_["exit_code"]} |',
                  f'| Listeners after stop | {teardown_["listeners"]} |',
                  f'| Host or gateway processes after stop | {teardown_["processes"]} |']
    lines += ['', 'The receipt holds timestamps, sizes, status codes, timings and memory readings only.', '']
    return '\n'.join(lines)


def write_receipt(output: Path, receipt: dict, credential: str) -> Path:
    """Write ``receipt.json`` and ``summary.md``; refuse when either would contain the credential."""
    text, summary = json.dumps(receipt, indent=1) + '\n', _summary_markdown(receipt)
    if credential and (credential in text or credential in summary):
        raise RuntimeError('refusing to write a soak receipt that contains the credential')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'receipt.json').write_text(text, encoding='utf-8')
    (output / 'summary.md').write_text(summary, encoding='utf-8')
    return output / 'receipt.json'


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--gateway-url', default=GATEWAY_URL, help='gateway base URL (default %(default)s)')
    parser.add_argument('--credential-name', default=CREDENTIAL_NAME, help='bearer variable name (default %(default)s)')
    parser.add_argument('--env-file', default=str(ROOT / '.env'), help='file to read the bearer from by name')
    parser.add_argument('--state-dir', default=str(STATE_DIR), help='host state folder (default %(default)s)')
    parser.add_argument('--output', default=None, help='receipt folder (default: <state-dir>/soak)')
    parser.add_argument('--duration-s', type=float, default=1800.0)
    parser.add_argument('--interval-s', type=float, default=30.0)
    parser.add_argument('--sample-s', type=float, default=5.0)
    parser.add_argument('--timeout-s', type=float, default=600.0, help='per batch request (the study plan uses about 600)')
    parser.add_argument('--top-k-keep', type=int, default=100)
    parser.add_argument('--stop-after', action='store_true', help='run the lifecycle stop and record the teardown')
    args = parser.parse_args(argv)
    if min(args.duration_s, args.interval_s, args.sample_s, args.timeout_s) <= 0 or args.top_k_keep < 1:
        parser.error('durations, intervals, the timeout and top_k_keep must be positive')
    credential, source = read_credential(args.credential_name, Path(args.env_file), os.environ)
    if not credential:
        print(f'refused: {args.credential_name} is not set in the environment or in {args.env_file}', file=sys.stderr)
        return 2
    state_dir = Path(args.state_dir)
    receipt = soak(args.gateway_url, credential_name=args.credential_name,
                   environment={args.credential_name: credential}, duration_s=args.duration_s,
                   interval_s=args.interval_s, sample_s=args.sample_s, state_folder=state_dir,
                   top_k_keep=args.top_k_keep, timeout_s=args.timeout_s, model_digest=MODEL_DIGEST,
                   credential_source=source)
    if args.stop_after:
        receipt['teardown'] = teardown(state_dir=state_dir)
        receipt['outcome'] = outcome(receipt)
    path = write_receipt(Path(args.output) if args.output else state_dir / 'soak', receipt, credential)
    print(json.dumps({'receipt': str(path), **receipt['outcome']}))
    return 0 if receipt['outcome']['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
