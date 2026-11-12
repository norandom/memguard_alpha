"""The study runner of the finance simulation (tasks 9.1 to 9.4; Requirements 8.2, 10.7, 11.1, 11.3, 11.6).

One command line drives every stage of the frozen plan through the committed
study modules, with the recording, resume, limits, approval and host rules of
tasks 7.4 to 7.9 wired exactly as their forward notes ask:

- ``build-plan`` assembles the items and freezes the plan through
  :mod:`scripts.finance_sim.build_plan` (keys by name from the process
  environment, nothing else read).
- ``check-hosts`` runs the host checks of one arm without admitting a request:
  llama-server through its gateway (``GET /v1/models`` and the GGUF named by
  ``GET /props`` against the plan's pin), the full-distribution host (the
  client's digest, a probe batch) and the Modal endpoint (``GET /v1/models``
  listing the arm's registry id).
- ``calibrate``, ``pilot``, ``full-run`` and ``negative-control`` run one stage
  of one arm in ``--pass generation``, ``--pass evidence`` or ``--pass both``
  (``evidence`` is the same as ``--pass evidence`` on the stage it names). The
  sampling arm's two local hosts are never loaded together, so its live
  operator runs the generation pass with llama-server up, stops it, starts
  the full-distribution host and runs the evidence pass, which repeats only
  the missing evidence batches; ``both`` serves the serving arm (no evidence)
  and the offline validation on fakes.
- ``gate`` reads the recorded power gate (:func:`~scripts.finance_sim.pilot.require_gate_passed`)
  and honours the frozen ``pilot.gate_policy``: ``inform`` reports a stop and
  lets the probe arm proceed, ``block`` maps the stop to exit code 4.
- ``smoke-run`` sends the serving arm's one smoke request (stage limits
  ``smoke_run``) and checks that the reply carries the declared top-k logprobs.
- ``teardown`` records the captured output of the host scripts' ``stop``
  commands in the stage receipt (:func:`~scripts.finance_sim.recording.record_teardown`).
- ``analyses`` writes both pre-registered analyses, the negative-control
  report and the convergence tables from the stored results, and ``replay``
  runs the offline worked-example replay of task 8.4 (``--require-guard``).

Wiring (the forward notes of 7.3b, 7.4, 7.7, 7.8, 7.9 and 7.12). Every
generation stage runs inside a :class:`~scripts.finance_sim.recording.StageSession`
(the stage's ledger, the receipt with its host checks, status and exit code,
and for the serving arm the validated approval and the serve lease), the
evidence pass inside an :class:`~scripts.finance_sim.recording.EvidenceSession`
that never opens the ledger. The entry points of the study modules get the
runner's :class:`StageRecorder` through ``on_request`` (a 7.4 request line,
flushed to disk before the send), ``on_draw`` (the 7.4 outcome line and the
versioned :class:`~scripts.finance_sim.recording.DrawLog`), ``on_evidence``
(one evidence file per item and rendering in the EvidenceSession format at
``recording.evidence_path``, merged when a file exists) and ``recorded`` (the
latest version of every draw; a request line without an outcome and without a
logged draw is closed as the failed draw ``request_error:interrupted`` and never
resent; a request the ledger or the gateway refused before generation gets an
outcome with ``sent: false``, so its draw index stays open). The serving arm's
admission is gated on the serve receipt's deadline before each request. Hosts
are checked before admission and loaded one at a time by the operator. After
a request failed (a transport error, a gateway 502 or a timeout, recorded as
the failed draw ``request_error:<code>``), the recorder runs 7.4's
post-failure probe of the generation host and records it in the receipt: a
dead host, or one serving other weights, ends the session with the stage open
(exit 1) before another planned draw is burned on it; a busy host lets the
pass go on. A timed-out evidence batch is resent only after the
full-distribution host answers the probe batch again (the host check of the
evidence pass). After a frozen limit closes a stage, the results of the fully
recorded items are derived and the stage is reported incomplete (exit 3).
``macro_items`` is
always called with ``arm=``, no calibration setting is overridden (the plan's
draw, fit and evidence settings apply), the evidence policy of the idle check
is built from ``ArmPlan.evidence``, and a sampling-arm client whose timeout is
not the plan's ``probe_request_timeout_seconds`` is refused before any request.

Exit codes: 0 complete (or the asked pass done), 1 the stage stays open (a
host failure, missing evidence or an interruption: resume with the same
command), 2 refused before any request, 3 incomplete (a frozen limit closed
admission; partial results kept), 4 the power gate stopped the gated arm under
``gate_policy = "block"``. The stage receipt (``receipt.json``) and the
session line the runner appends to ``runner.jsonl`` carry the status and exit
code of the process: a pass that raised is recorded as incomplete (exit 3, a
frozen limit, whether the ledger's ``AdmissionClosedError`` or the plain
``BudgetExhaustedError`` of a spent request cap), open (exit 1, a host failure
or a missing-evidence failure, stop reason ``evidence_missing``, resumable),
refused (exit 2, the refusal classes ``main`` maps) or failed (exit 1, any
other exception, which also ends the process with its traceback); the
exception class is named in ``last_session.error`` and the runner line's
``error`` (refused and failed also in ``stop_reason``), never its message, and
a pass that raised is never recorded as complete whatever the ledger says. The
receipt's ``last_session.evidence_batches`` counts the evidence batches the
pass posted (fix 9.1b).

Live route (tasks 9.2 and 9.3; user decision 2026-11-10: the overnight live
stages run natively, the hosts native too, one model host at a time; the
Compose route stays the alternative). Native route, from PowerShell at the
repository root, with ``LOCAL_MODEL_API_KEY``, ``FRED_API_KEY`` and
``FMP_API_TOKEN`` loaded by name from the root ``.env`` into the session (never
printed); the gateways keep their loopback defaults (no bind address
argument) and the runner reaches them through the two endpoint variables::

    pwsh -NoProfile -File scripts/hosts/serve_local_llamacpp.ps1 start -IdleMinutes 180
    $env:LLAMACPP_ENDPOINT_URL = "http://127.0.0.1:11571/v1/chat/completions"
    $env:FULL_DISTRIBUTION_ENDPOINT_URL = "http://127.0.0.1:11581"
    uv run --no-sync python -m scripts.finance_sim.run_study check-hosts --arm sampling --host generation `
      --plan runs/finance-sim/plan/plan.json --plan-sha256 <plan_sha256>
    uv run --no-sync python -m scripts.finance_sim.run_study calibrate --arm sampling --pass generation `
      --plan runs/finance-sim/plan/plan.json --plan-sha256 <plan_sha256>
    pwsh -NoProfile -File scripts/hosts/serve_local_llamacpp.ps1 stop *> runs/finance-sim/records/sampling/calibration/llama-server-stop.txt
    pwsh -NoProfile -File scripts/hosts/serve_full_distribution.ps1 start -IdleMinutes 180
    uv run --no-sync python -m scripts.finance_sim.run_study calibrate --arm sampling --pass evidence `
      --plan runs/finance-sim/plan/plan.json --plan-sha256 <plan_sha256>
    pwsh -NoProfile -File scripts/hosts/serve_full_distribution.ps1 stop *> runs/finance-sim/records/sampling/calibration/full-distribution-stop.txt
    uv run --no-sync python -m scripts.finance_sim.run_study teardown --arm sampling --stage calibration `
      --stop-output runs/finance-sim/records/sampling/calibration/llama-server-stop.txt runs/finance-sim/records/sampling/calibration/full-distribution-stop.txt `
      --plan runs/finance-sim/plan/plan.json --plan-sha256 <plan_sha256>

The same two passes run ``pilot``, ``full-run --arm sampling`` and
``negative-control --arm sampling``; ``gate`` between the pilot and the full
run. ``--pass both`` is refused for the sampling arm on live hosts (its two
local hosts are never loaded together). After a failed request the runner
probes the generation host again: a dead gateway ends the session with the
stage open (exit 1); restart the host with its script and resume with the
same command, only the unrecorded draws are issued.

Compose alternative (the runner in the ``dev`` container, the hosts native):
the gateways bind the Podman adapter address and the container reaches them
as ``host.containers.internal``::

    docker compose run --rm dev getent hosts host.containers.internal
    pwsh -NoProfile -File scripts/hosts/serve_local_llamacpp.ps1 start -BindAddress <adapter IP> -IdleMinutes 180
    docker compose run --rm -e LOCAL_MODEL_API_KEY `
      -e LLAMACPP_ENDPOINT_URL=http://host.containers.internal:11571/v1/chat/completions `
      -e FULL_DISTRIBUTION_ENDPOINT_URL=http://host.containers.internal:11581 `
      -v "${PWD}/runs/finance-sim:/app/runs/finance-sim" dev `
      uv run --no-sync python -m scripts.finance_sim.run_study check-hosts --arm sampling --host generation `
      --plan runs/finance-sim/plan/plan.json --plan-sha256 <plan_sha256>
    pwsh -NoProfile -File scripts/hosts/serve_full_distribution.ps1 start -BindAddress <adapter IP> -IdleMinutes 180

The serving arm (task 9.4) runs ``smoke-run``, ``calibrate``, ``full-run`` and
``negative-control`` with ``--arm serving --pass both``, each after
``sh scripts/modal_operator.sh serve <manifest> runs/finance-sim/plan/stage_limits/<stage>.json <receipt>``
with ``--manifest``, ``--approvals-dir`` and ``--serve-receipt <receipt>`` and
``MODAL_PROXY_BEARER`` and ``MODAL_ENDPOINT_URL`` loaded by name (``-e`` on the
Compose route), and ``stop-serve`` plus ``teardown --stop-output <receipt>``
after each stage.

What the offline validation (the fakes of task 9.1) could not exercise, so
the first live stage of each route is their check: the real Vulkan throughput
of llama-server against the plan's 6 s request rate, the real gateway's 502
once the host is gone and its idle stop after ``-IdleMinutes``, the real
``/props`` path of the pinned GGUF, and the container route (the adapter bind
and ``host.containers.internal``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Protocol

from deploy.modal.stage_limits import STAGES as SERVING_STAGES
from recall_guard.core.completion import CompletionClient, ModelConfig
from recall_guard.core.full_distribution import FullDistributionEvidence
from recall_guard.core.profiles import ProfileError, connect, load_profile
from recall_guard.core.request_control import BudgetExhaustedError
from recall_guard.dataset.macro_regime import MacroRow, RealizedReturns
from recall_guard.harness.guarded_decision import REQUEST_ERROR_PREFIX, full_distribution_client
from recall_guard.harness.scorer_client import identity_settings
from scripts.finance_sim import build_plan as build_plan_module
from scripts.finance_sim import replay as replay_module
from scripts.finance_sim.analysis import analysis_from_records, write_analysis
from scripts.finance_sim.calibrate import (
    CALIBRATION_STAGE,
    EDGAR,
    MACRO,
    CalibrationCorpus,
    CalibrationDraw,
    CalibrationError,
    CorpusPrompt,
    EdgarRelease,
    EvidenceUnavailableError,
    collect_draws,
    edgar_corpus,
    macro_corpus,
    plan_draw_settings,
    run_calibration,
)
from scripts.finance_sim.convergence import CONVERGENCE_STAGES, convergence_from_records, write_table
from scripts.finance_sim.ensemble_metrics import ANALYSED_STAGES, ensemble_metrics_from_records, write_metrics
from scripts.finance_sim.execute import (
    DEFINITIONS,
    EXECUTION_STAGES,
    INTERRUPTED,
    ExecutionDraw,
    ExecutionError,
    ExecutionItem,
    ExecutionRun,
    execute_items,
    generate_draws,
    load_calibrators,
    macro_items,
    planned_keys,
)
from scripts.finance_sim.items import (
    DEFAULT_EDGAR_DIR,
    ITEMS_SCHEMA,
    PRIVATE_ROOT,
    EdgarItems,
    MacroItem,
    PrivatePaths,
    RealizedOutcome,
    load_edgar_items,
)
from scripts.finance_sim.limits import (
    ARMS,
    EXIT_COMPLETE,
    EXIT_INCOMPLETE,
    EXIT_REFUSED,
    EXIT_UNFINISHED,
    SAMPLING_ARM,
    SERVING_ARM,
    AdmissionClosedError,
    ApprovalError,
    BudgetLedger,
    LedgerCallBudget,
    RequestAllowance,
    ServingApproval,
    StudyLimits,
    budget_path,
    require_serving_approval,
)
from scripts.finance_sim.negative_control import (
    NegativeControlError,
    negative_control_from_records,
    negative_control_items,
    run_negative_control,
    write_report,
)
from scripts.finance_sim.pilot import (
    CONFIRMATORY_STAGE,
    PILOT_STAGE,
    STOP_FOR_DECISION,
    GateStoppedError,
    PilotError,
    PowerGate,
    require_gate_passed,
    run_pilot,
)
from scripts.finance_sim.plan import PlanChangedError, StudyPlan, load_plan, sha256_of
from scripts.finance_sim.recording import (
    EVIDENCE_BATCHES,
    EVIDENCE_SCHEMA,
    RECORD_SCHEMA,
    REFUSAL_TYPES,
    DrawLog,
    EvidencePolicy,
    EvidenceSession,
    HostCheck,
    HostFailure,
    ServeLease,
    StageSession,
    check_full_distribution_host,
    check_llama_server,
    check_serving_host,
    draw_log_path,
    evidence_path,
    pinned_digest,
    record_teardown,
    request_allowance,
    requests_path,
    stage_directory,
)

ROOT = Path(__file__).resolve().parents[2]
#: The runner's own exit code: the power gate stopped the gated arm under ``gate_policy = "block"``.
EXIT_GATE_STOPPED = 4
GENERATION, EVIDENCE, BOTH = "generation", "evidence", "both"
PASSES: tuple[str, ...] = (GENERATION, EVIDENCE, BOTH)
#: The stage each stage command runs.
STAGE_OF: Mapping[str, str] = {"calibrate": CALIBRATION_STAGE, "pilot": PILOT_STAGE, "full-run": CONFIRMATORY_STAGE,
                               "negative-control": "negative_control"}
SMOKE_STAGE = "smoke_run"
RUNNER_LOG = "runner.jsonl"
RUNNER_SCHEMA = "finance-sim-runner-session/1"
GATE_INFORM, GATE_BLOCK = "inform", "block"
#: Hosts of the ``check-hosts`` command.
HOST_GENERATION, HOST_FULL_DISTRIBUTION, HOST_ALL = "generation", "full-distribution", "all"
#: The smoke run's one prompt is the anonymised rendering of the first evaluation month: no identifying data.
SMOKE_RENDERING = "anonymised"
_RELEASE_TEXT_KEY = "response_text"
#: The wall clock of the serving arm's deadline gate (``time.time``); a seam the offline validation replaces.
wall_clock: Callable[[], float] = time.time


class RunnerError(ValueError):
    """The runner refuses before any request, naming the problem."""


def _utc_now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def _iso(moment: datetime) -> str:
    return moment.isoformat(timespec="seconds")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _wall() -> float:
    return wall_clock()


def _append_line(path: Path, payload: Mapping[str, Any], *, fsync: bool) -> None:
    """Append one JSON line; with ``fsync`` it reaches the disk before the call returns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, sort_keys=True, allow_nan=False, ensure_ascii=False) + "\n").encode("utf-8")
    with path.open("ab") as stream:
        stream.write(data)
        stream.flush()
        if fsync:
            os.fsync(stream.fileno())


def _read_lines(path: Path) -> list[dict[str, Any]]:
    """The JSON lines of a record; a last line cut short by a crash (no newline) is dropped, never repaired."""
    if not path.exists():
        return []
    data = path.read_bytes()
    segments = data.split(b"\n")
    tail = segments.pop() if segments else b""
    lines = []
    for number, raw in enumerate(segments, start=1):
        if not raw.strip():
            continue
        try:
            line = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise RunnerError(f"{path} line {number} is not valid JSON ({exc}); refused") from None
        if not isinstance(line, dict):
            raise RunnerError(f"{path} line {number} is not a JSON object; refused")
        lines.append(line)
    if tail.strip():
        try:
            line = json.loads(tail.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            return lines  # cut short by a crash
        if isinstance(line, dict):
            lines.append(line)
    return lines


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RunnerError(f"{path} cannot be read ({exc})") from None


# ---------- paths ----------

@dataclass(frozen=True)
class StudyPaths:
    """Where the study lives: the private root (``runs/finance-sim``) and the folders under it."""

    root: Path
    plan: Path
    items: Path
    stage_limits: Path
    records: Path
    results: Path
    calibrators: Path
    gates: Path
    approvals: Path
    analysis: Path
    edgar: Path

    @classmethod
    def under(cls, root: str | Path, **overrides: Path | None) -> StudyPaths:
        root = Path(root)
        defaults = {"plan": root / "plan" / "plan.json", "items": PrivatePaths(root).items,
                    "stage_limits": root / "plan" / "stage_limits", "records": root / "records",
                    "results": root / "results", "calibrators": root / "calibrators", "gates": root / "gates",
                    "approvals": root / "approvals", "analysis": root / "analysis", "edgar": DEFAULT_EDGAR_DIR}
        values = {name: (overrides.get(name) or default) for name, default in defaults.items()}
        return cls(root=root, **values)

    @property
    def private(self) -> PrivatePaths:
        return PrivatePaths(self.root)


# ---------- the study items ----------

@dataclass(frozen=True)
class StudyItems:
    """The macro items of the private ``items.json`` and the EDGAR corpus items the plan froze."""

    macro: tuple[MacroItem, ...]
    edgar: EdgarItems

    def macro_rows(self, months: Iterable[str]) -> list[MacroRow]:
        wanted = set(months)
        return [item.row for item in self.macro if f"{item.decision_date:%Y-%m}" in wanted]

    def calibration_releases(self) -> list[EdgarRelease]:
        return [EdgarRelease(item.item_id, item.label, item.release_text) for item in self.edgar.calibration]


def _macro_item(entry: Mapping[str, Any]) -> MacroItem:
    try:
        row = entry["row"]
        macro = MacroRow(decision_date=date.fromisoformat(row["decision_date"]), raw=row["raw"],
                         zscores=row["zscores"], revised=row["revised"])
        realized = entry["realized"]
        outcome = RealizedOutcome(macro.decision_date, None if realized is None else RealizedReturns(**realized),
                                  entry["realized_missing_reason"], {})
        return MacroItem(item_id=entry["item_id"], row=macro, observed_through=dict(entry.get("observed_through", {})),
                         outcome=outcome)
    except (KeyError, TypeError, ValueError) as exc:
        raise RunnerError(f"items.json holds a macro item that cannot be read ({exc})") from None


def load_items(path: str | Path, edgar_dir: str | Path = DEFAULT_EDGAR_DIR) -> StudyItems:
    """The study items: the private ``items.json`` of the build and the frozen EDGAR corpus."""
    record = _read_json(Path(path))
    if not isinstance(record, Mapping) or record.get("schema") != ITEMS_SCHEMA:
        raise RunnerError(f"{path} is not a {ITEMS_SCHEMA} items file")
    entries = record.get("macro_items")
    if not isinstance(entries, list):
        raise RunnerError(f"{path} lists no macro items")
    return StudyItems(macro=tuple(_macro_item(entry) for entry in entries), edgar=load_edgar_items(edgar_dir))


# ---------- the hosts ----------

class Hosts(Protocol):
    """How the runner reaches the model hosts: the clients and the checks, never a start or a stop."""

    def generation_config(self, plan: StudyPlan, arm: str) -> ModelConfig: ...

    def generation(self, plan: StudyPlan, arm: str, ledger: BudgetLedger) -> CompletionClient: ...

    def full_distribution(self, plan: StudyPlan) -> Any: ...

    def check_generation(self, plan: StudyPlan, arm: str, config: ModelConfig) -> HostCheck: ...

    def check_full_distribution(self, plan: StudyPlan, client: Any) -> HostCheck: ...


@dataclass(frozen=True)
class LiveHosts:
    """The live hosts of the plan's provider profiles, reached through their gateways.

    The generation client is :func:`~recall_guard.core.profiles.connect` with the
    stage ledger's :class:`~scripts.finance_sim.limits.LedgerCallBudget` and the
    gateway URL the profile's ``endpoint_ref`` names in the environment
    (``LLAMACPP_ENDPOINT_URL``, ``MODAL_ENDPOINT_URL``); the host preflight is
    not enforced here because the hosts are loaded by their Windows-side scripts
    (their RAM gate) and checked by the runner before admission, and the runner
    runs in a container whose memory says nothing about the host. The
    full-distribution client comes from
    :func:`~recall_guard.harness.guarded_decision.full_distribution_client`
    (``FULL_DISTRIBUTION_ENDPOINT_URL``). Credentials stay references.
    """

    environment: Mapping[str, str]

    def generation_config(self, plan: StudyPlan, arm: str) -> ModelConfig:
        profile = load_profile(plan.arms.arms[arm].profile_id)
        config = profile.model_config
        if profile.endpoint_ref is not None:
            endpoint = self.environment.get(profile.endpoint_ref.removeprefix("env:"))
            if isinstance(endpoint, str) and endpoint.strip():
                config = replace(config, endpoint=endpoint.strip())
        return config

    def generation(self, plan: StudyPlan, arm: str, ledger: BudgetLedger) -> CompletionClient:
        return connect(plan.arms.arms[arm].profile_id, LedgerCallBudget(ledger), environment=self.environment,
                       enforce_preflight=False)

    def full_distribution(self, plan: StudyPlan) -> Any:
        return full_distribution_client(plan.arms.arms[SAMPLING_ARM].profile_id, environment=self.environment)

    def check_generation(self, plan: StudyPlan, arm: str, config: ModelConfig) -> HostCheck:
        if arm == SAMPLING_ARM:
            return check_llama_server(config, expected_digest=pinned_digest(plan), environment=self.environment)
        return check_serving_host(config, expected_model=plan.arms.registry_ids[SERVING_ARM],
                                  environment=self.environment)

    def check_full_distribution(self, plan: StudyPlan, client: Any) -> HostCheck:
        return check_full_distribution_host(client, expected_digest=pinned_digest(plan))


# ---------- the recorder (7.4 lines, the draw log and the evidence files) ----------

@dataclass(frozen=True)
class RequestFields:
    """What a 7.4 request line records about the ensemble a draw belongs to."""

    settings_json: str
    config: Mapping[str, Any]
    source: Mapping[str, Any]
    calibrator_fingerprint: str | None
    temperature: float | None


@dataclass(frozen=True)
class OutcomeFields:
    """What a 7.4 outcome line records about a finished draw (the reply itself stays in the draw log)."""

    response_sha256: str
    failure_code: str | None
    evidence_ref: str | None
    usage: Mapping[str, Any] | None
    draw: Mapping[str, Any]


Slot = tuple[str, str, int]


class StageRecorder:
    """The recording of one stage driven through the 7.6 to 7.9 entry points.

    ``on_request`` writes the 7.4 request line of a draw and flushes it to disk
    before the send; ``on_draw`` appends the draw to the stage's
    :class:`~scripts.finance_sim.recording.DrawLog` (every version; the latest
    wins) and writes the outcome line of a draw's first version;
    ``on_evidence`` writes the evidence batch of an item and rendering in the
    EvidenceSession file format (merged with an existing file). ``recorded``
    closes every request line without an outcome and without a logged draw as
    the failed draw ``request_error:interrupted`` (built by the stage's
    factory) and returns the latest version of every draw for ``recorded=``.
    ``not_sent`` closes the request a refusal interrupted before generation
    with ``sent: false``, so its draw index stays open. ``after_failure`` is
    7.4's post-failure host probe: it runs once the outcome line of a draw
    that failed in its request (``request_error:<code>``, not the resume's
    ``interrupted``) is written, and ends the session when the host is gone.
    """

    def __init__(self, plan: StudyPlan, arm: str, stage: str, *, records_dir: Path, loads: Callable[[Any], Any],
                 slot: Callable[[Any], Slot], request_fields: Callable[[Any], RequestFields],
                 outcome_fields: Callable[[Any], OutcomeFields], host_digest: str, session: int,
                 gate: Callable[[], None] | None = None, after_failure: Callable[[], None] | None = None,
                 now: Callable[[], datetime] = _utc_now, clock: Callable[[], float] = time.monotonic) -> None:
        self.plan, self.arm, self.stage = plan, arm, stage
        self.directory = stage_directory(arm, stage, records_dir)
        self.records_dir = records_dir
        self.log = DrawLog(draw_log_path(arm, stage, records_dir), plan=plan, loads=loads, now=now)
        self.slot, self._request_fields, self._outcome_fields = slot, request_fields, outcome_fields
        self.host_digest, self.session = host_digest, session
        self._gate, self._after_failure, self._now, self._clock = gate, after_failure, now, clock
        self._pending: dict[Slot, float] = {}
        self._seen: set[Slot] = set()
        self.counts = {"requests_announced": 0, "draws_recorded": 0, EVIDENCE_BATCHES: 0, "not_sent": 0,
                       "interrupted_closed": 0}

    def _header(self, kind: str, item_id: str, rendering: str) -> dict[str, Any]:
        return {"schema": RECORD_SCHEMA, "kind": kind, "plan_sha256": self.plan.plan_sha256, "arm": self.arm,
                "stage": self.stage, "item_id": item_id, "rendering": rendering}

    def on_request(self, item: Any, index: int, seed: int | None) -> None:
        """Before each generation request: the serving deadline gate, then the request line, flushed."""
        if self._gate is not None:
            self._gate()
        fields = self._request_fields(item)
        line = self._header("request", item.item_id, item.rendering) | {
            "draw_index": index, "seed": seed, "temperature": fields.temperature, "requested_at": _iso(self._now()),
            "prompt_sha256": item.prompt_sha256, "settings_json": fields.settings_json, "config": dict(fields.config),
            "source": dict(fields.source), "calibrator_fingerprint": fields.calibrator_fingerprint,
            "host_digest": self.host_digest, "session": self.session}
        _append_line(requests_path(self.arm, self.stage, item.item_id, self.records_dir), line, fsync=True)
        self._pending[(item.item_id, item.rendering, index)] = self._clock()
        self.counts["requests_announced"] += 1

    def on_draw(self, draw: Any) -> None:
        """After each draw, and again when its evidence settles: the log keeps every version."""
        self.log(draw)
        slot = self.slot(draw)
        if slot in self._seen:
            return
        self._seen.add(slot)
        self.counts["draws_recorded"] += 1
        started = self._pending.pop(slot, None)
        fields = self._outcome_fields(draw)
        line = self._header("outcome", slot[0], slot[1]) | {
            "draw_index": slot[2], "sent": True, "recorded_at": _iso(self._now()),
            "response_sha256": fields.response_sha256, "failure_code": fields.failure_code,
            "usage": None if fields.usage is None else dict(fields.usage),
            "latency_s": None if started is None else max(0.0, self._clock() - started),
            "evidence_ref": fields.evidence_ref, "draw": dict(fields.draw), "response": None, "full_evidence": None,
            "closed_by": None}
        _append_line(requests_path(self.arm, self.stage, slot[0], self.records_dir), line, fsync=False)
        code = fields.failure_code
        if (self._after_failure is not None and code is not None and code.startswith(REQUEST_ERROR_PREFIX)
                and code != INTERRUPTED):
            self._after_failure()

    def not_sent(self, reason: str) -> None:
        """Close the request a refusal interrupted before generation: ``sent: false``, the index stays open."""
        for (item_id, rendering, index), _ in list(self._pending.items()):
            line = self._header("outcome", item_id, rendering) | {
                "draw_index": index, "sent": False, "recorded_at": _iso(self._now()), "response_sha256": None,
                "failure_code": f"not_sent:{reason}", "usage": None, "latency_s": None, "evidence_ref": None,
                "draw": None, "response": None, "full_evidence": None, "closed_by": None}
            _append_line(requests_path(self.arm, self.stage, item_id, self.records_dir), line, fsync=True)
            self.counts["not_sent"] += 1
        self._pending.clear()

    def on_evidence(self, item: Any, found: Mapping[int, FullDistributionEvidence], *, top_k_keep: int,
                    texts: Mapping[int, str]) -> None:
        """One evidence batch of ``item`` in the 7.4 evidence-file format, merged with an existing file."""
        path = evidence_path(self.arm, self.stage, item.item_id, item.rendering, self.records_dir)
        evidence: dict[int, dict[str, Any]] = {}
        if path.exists():
            existing = _read_json(path)
            if not isinstance(existing, Mapping) or existing.get("schema") != EVIDENCE_SCHEMA:
                raise RunnerError(f"{path} is not a {EVIDENCE_SCHEMA} evidence file; refused")
            if existing.get("plan_sha256") != self.plan.plan_sha256:
                raise PlanChangedError(f"{path} was recorded under plan_sha256 {existing.get('plan_sha256')}; this "
                                       f"plan is {self.plan.plan_sha256}; refused, never merged")
            if existing.get("prompt_sha256") != item.prompt_sha256:
                raise RunnerError(f"{path} holds the evidence of another prompt; refused, never merged")
            evidence.update(zip(existing["draw_indices"], existing["evidence"], strict=True))
        digest = None
        for index, entry in found.items():
            evidence[index] = entry.to_dict()
            digest = entry.model_digest
        indices = sorted(evidence)
        batch = sha256_of({"rendering": item.rendering, "prompt_sha256": item.prompt_sha256,
                           "responses": [[index, _sha256_text(texts[index])] for index in indices if index in texts],
                           "top_k_keep": top_k_keep, "model_digest": digest})
        _write_json(path, {"schema": EVIDENCE_SCHEMA, "plan_sha256": self.plan.plan_sha256, "arm": self.arm,
                           "stage": self.stage, "item_id": item.item_id, "rendering": item.rendering,
                           "prompt_sha256": item.prompt_sha256, "batch_sha256": batch, "model_digest": digest,
                           "top_k_keep": top_k_keep, "draw_indices": indices,
                           "evidence": [evidence[index] for index in indices], "recorded_at": _iso(self._now())})
        self.counts[EVIDENCE_BATCHES] += 1

    def _slots(self) -> dict[Slot, dict[str, Any]]:
        """Per draw of this stage: its open request line and its sent outcome, from every item record."""
        slots: dict[Slot, dict[str, Any]] = {}
        if not self.directory.exists():
            return slots
        for path in sorted(self.directory.glob("*.jsonl")):
            if path.name == RUNNER_LOG or path == self.log.path:
                continue
            for number, line in enumerate(_read_lines(path), start=1):
                where = f"{path} line {number}"
                if line.get("schema") != RECORD_SCHEMA:
                    raise RunnerError(f"{where} is not a {RECORD_SCHEMA} record line; refused")
                if line.get("plan_sha256") != self.plan.plan_sha256:
                    raise PlanChangedError(f"{where} was recorded under plan_sha256 {line.get('plan_sha256')}; this "
                                           f"plan is {self.plan.plan_sha256}; refused, never merged")
                if (line.get("arm"), line.get("stage")) != (self.arm, self.stage):
                    raise RunnerError(f"{where} belongs to {line.get('arm')}/{line.get('stage')}; refused")
                if line.get("kind") not in ("request", "outcome"):
                    continue
                index = line.get("draw_index")
                if isinstance(index, bool) or not isinstance(index, int) or index < 0:
                    raise RunnerError(f"{where} has no valid draw_index; refused")
                if not isinstance(line.get("item_id"), str) or not isinstance(line.get("rendering"), str):
                    raise RunnerError(f"{where} names no item and rendering; refused")
                slot = (line["item_id"], line["rendering"], index)
                entry = slots.setdefault(slot, {"request": None, "outcome": None})
                if line["kind"] == "request":
                    if entry["request"] is not None and entry["outcome"] is None:
                        raise RunnerError(f"{where} requests draw {index} again although it was already sent; a "
                                          "recorded request is never issued again; refused")
                    entry["request"], entry["outcome"] = line, None
                elif entry["request"] is None:
                    raise RunnerError(f"{where} records an outcome of draw {index} without its request; refused")
                elif line.get("sent") is True:
                    entry["outcome"] = line
                elif line.get("sent") is False:
                    entry["request"] = None  # refused before generation: the index stays open
                else:
                    raise RunnerError(f"{where} does not say whether draw {index} was sent; refused")
        return slots

    def recorded(self, interrupted: Callable[[str, str, int, int | None], Any]) -> list[Any]:
        """The latest version of every draw, with every lost request closed as an interrupted draw."""
        latest = self.log.latest()
        logged = {self.slot(draw) for draw in latest}
        self._seen.update(logged)
        for slot, entry in sorted(self._slots().items()):
            if entry["request"] is None or entry["outcome"] is not None or slot in logged:
                continue
            draw = interrupted(slot[0], slot[1], slot[2], entry["request"].get("seed"))
            self.log(draw)
            fields = self._outcome_fields(draw)
            line = self._header("outcome", slot[0], slot[1]) | {
                "draw_index": slot[2], "sent": True, "recorded_at": _iso(self._now()),
                "response_sha256": fields.response_sha256, "failure_code": fields.failure_code, "usage": None,
                "latency_s": None, "evidence_ref": None, "draw": dict(fields.draw), "response": None,
                "full_evidence": None, "closed_by": "resume"}
            _append_line(requests_path(self.arm, self.stage, slot[0], self.records_dir), line, fsync=True)
            self._seen.add(slot)
            latest.append(draw)
            self.counts["interrupted_closed"] += 1
        return latest


def fully_recorded(draws: Iterable[Any], slot: Callable[[Any], Slot],
                   planned: Mapping[tuple[str, str], int]) -> set[tuple[str, str]]:
    """The item renderings whose every planned draw index has a recorded version."""
    present: dict[tuple[str, str], set[int]] = {}
    for draw in draws:
        item_id, rendering, index = slot(draw)
        present.setdefault((item_id, rendering), set()).add(index)
    return {key for key, count in planned.items() if present.get(key, set()) >= set(range(count))}


# ---------- the stage ----------

@dataclass(frozen=True)
class ServingInputs:
    """What the serving arm needs besides the plan: the validated approval and the serve receipt of the stage."""

    approval: ServingApproval
    serve_receipt: Path | None


@dataclass(frozen=True)
class StageSetup:
    """One stage of one arm with everything the runner wires for it."""

    plan: StudyPlan
    arm: str
    stage: str
    paths: StudyPaths
    hosts: Hosts
    items: StudyItems
    serving: ServingInputs | None

    @property
    def arm_plan(self) -> Any:
        return self.plan.arms.arms[self.arm]

    @property
    def kind(self) -> str:
        return EXECUTION_STAGES[self.stage] if self.stage in EXECUTION_STAGES else CALIBRATION_STAGE

    def limits(self) -> StudyLimits:
        if self.serving is not None:
            return self.serving.approval.limits_for(self.stage)
        budget = self.plan.limits.stage(self.arm, self.stage)
        return StudyLimits(self.plan.plan_sha256, self.arm, self.stage, budget.max_requests,
                           budget.max_output_tokens, budget.max_seconds)

    def allowance(self, config: ModelConfig) -> RequestAllowance:
        rate = None if self.serving is None else self.serving.approval.presentation.usd_per_second
        return request_allowance(self.plan, self.arm, config, usd_per_second=rate)

    def check_client_config(self, config: ModelConfig) -> None:
        """The frozen timeout rule (7.3b): the probe arm's client timeout is the plan's request timeout."""
        if self.arm == SAMPLING_ARM:
            expected = self.plan.limits.probe_request_timeout_seconds
            if config.timeout_s != expected:
                raise RunnerError(f"the sampling arm's client timeout is {config.timeout_s} s; the frozen plan's "
                                  f"request timeout is {expected} s; nothing was sent")
        if config.max_retries != 0:
            raise RunnerError(f"the client retries (max_retries={config.max_retries}); nothing was sent")

    def evidence_policy(self) -> EvidencePolicy | None:
        """The idle-check policy of the evidence pass, built from the frozen ``ArmPlan.evidence``."""
        frozen = self.arm_plan.evidence
        if frozen is None:
            return None
        return EvidencePolicy(timeout_s=frozen.timeout_s, top_k_keep=frozen.top_k_keep,
                              max_busy_retries=frozen.host_busy_retries, busy_wait_s=frozen.host_busy_wait_s)

    # ----- the items and the corpora -----

    def corpora(self) -> dict[str, CalibrationCorpus]:
        return {MACRO: macro_corpus(self.plan, self.items.macro_rows(self.plan.months.calibration)),
                EDGAR: edgar_corpus(self.plan, self.items.calibration_releases())}

    def execution_items(self) -> tuple[ExecutionItem, ...]:
        if self.stage == "negative_control":
            return negative_control_items(self.plan, self.items.edgar.negative_control)
        return macro_items(self.plan, self.items.macro, stage=self.stage, arm=self.arm)

    def calibrators(self) -> dict[str, Any]:
        return load_calibrators(self.plan, arm=self.arm, kind=self.kind, calibrators_dir=self.paths.calibrators)

    def planned_draws(self) -> dict[tuple[str, str], int]:
        """Draws per item and rendering the frozen plan sizes for this stage."""
        if self.stage == CALIBRATION_STAGE:
            planned = {}
            for kind, corpus in self.corpora().items():
                draws = plan_draw_settings(self.plan, self.arm, kind).draws
                planned.update({(prompt.item_id, prompt.rendering): draws for prompt in corpus.prompts})
            return planned
        draws = self.plan.pilot.draws if self.stage == PILOT_STAGE else self.arm_plan.draws
        return dict.fromkeys(planned_keys(self.plan, self.arm, self.stage), draws)

    # ----- the recorder of this stage -----

    def recorder(self, records_dir: Path, *, config: ModelConfig, host_digest: str, session: int,
                 gate: Callable[[], None] | None, calibrators: Mapping[str, Any] | None,
                 after_failure: Callable[[], None] | None = None) -> StageRecorder:
        if self.stage == CALIBRATION_STAGE:
            return self._calibration_recorder(records_dir, config, host_digest, session, gate, after_failure)
        return self._execution_recorder(records_dir, config, host_digest, session, gate, calibrators, after_failure)

    def _calibration_recorder(self, records_dir: Path, config: ModelConfig, host_digest: str, session: int,
                              gate: Callable[[], None] | None,
                              after_failure: Callable[[], None] | None) -> StageRecorder:
        settings = {kind: plan_draw_settings(self.plan, self.arm, kind) for kind in (MACRO, EDGAR)}
        sources = list(self.arm_plan.estimate_sources)

        def request_fields(prompt: CorpusPrompt) -> RequestFields:
            kind = EDGAR if prompt.rendering == "outlook" else MACRO
            chosen = settings[kind]
            temperature = chosen.temperature if chosen.temperature is not None else config.parameters.get("temperature")
            parameters = dict(config.parameters)
            if chosen.temperature is not None:
                parameters["temperature"] = chosen.temperature
            return RequestFields(settings_json=identity_settings(parameters), config=chosen.to_dict(),
                                 source={"sources": sources}, calibrator_fingerprint=None,
                                 temperature=None if temperature is None else float(temperature))

        def outcome_fields(draw: CalibrationDraw) -> OutcomeFields:
            payload = {key: value for key, value in draw.to_dict().items() if key != _RELEASE_TEXT_KEY}
            return OutcomeFields(response_sha256=draw.response_sha256, failure_code=draw.failure_code,
                                 evidence_ref=draw.evidence_ref.get("top_k"), usage=None, draw=payload)

        return StageRecorder(self.plan, self.arm, self.stage, records_dir=records_dir, loads=CalibrationDraw.from_dict,
                             slot=lambda draw: (draw.item_id, draw.rendering, draw.draw_index),
                             request_fields=request_fields, outcome_fields=outcome_fields, host_digest=host_digest,
                             session=session, gate=gate, after_failure=after_failure)

    def _execution_recorder(self, records_dir: Path, config: ModelConfig, host_digest: str, session: int,
                            gate: Callable[[], None] | None, calibrators: Mapping[str, Any] | None,
                            after_failure: Callable[[], None] | None) -> StageRecorder:
        if calibrators is None:
            raise RunnerError("an execution stage records under its top-k calibrator; none was loaded")
        arm = self.arm_plan
        draws, min_parsed = ((self.plan.pilot.draws, self.plan.pilot.min_parsed) if self.stage == PILOT_STAGE
                             else (arm.draws, arm.min_parsed))
        fields = RequestFields(
            settings_json=identity_settings({**dict(config.parameters), "temperature": arm.temperature}),
            config={"definition": DEFINITIONS[self.kind].name, "draws": draws, "min_parsed": min_parsed,
                    "mode": "ensemble", "temperature": arm.temperature, "seed_base": arm.seed_base},
            source={"source": "top_k", "k": arm.top_logprobs},
            calibrator_fingerprint=calibrators["top_k"].calibrator.fingerprint, temperature=arm.temperature)

        def outcome_fields(draw: ExecutionDraw) -> OutcomeFields:
            usage = None if draw.response is None or draw.response.usage is None else dict(draw.response.usage)
            return OutcomeFields(response_sha256=draw.draw.response_sha256, failure_code=draw.draw.failure_code,
                                 evidence_ref=draw.draw.evidence_ref, usage=usage, draw=draw.draw.to_dict())

        return StageRecorder(self.plan, self.arm, self.stage, records_dir=records_dir, loads=ExecutionDraw.from_dict,
                             slot=lambda draw: (draw.item_id, draw.rendering, draw.draw.draw_index),
                             request_fields=lambda item: fields, outcome_fields=outcome_fields,
                             host_digest=host_digest, session=session, gate=gate, after_failure=after_failure)

    def interrupted_factory(self, calibrators: Mapping[str, Any] | None,
                            items: Sequence[Any]) -> Callable[[str, str, int, int | None], Any]:
        """How a lost request of this stage becomes the failed draw ``request_error:interrupted``."""
        by_key = {(item.item_id, item.rendering): item for item in items}
        if self.stage == CALIBRATION_STAGE:
            settings = {kind: plan_draw_settings(self.plan, self.arm, kind) for kind in (MACRO, EDGAR)}
            sources = self.arm_plan.estimate_sources

            def calibration(item_id: str, rendering: str, index: int, seed: int | None) -> CalibrationDraw:
                prompt = by_key.get((item_id, rendering))
                if prompt is None:
                    raise RunnerError(f"a request of {item_id}/{rendering} was recorded, but the stage has no such "
                                      "prompt; refused")
                chosen = settings[EDGAR if rendering == "outlook" else MACRO]
                return CalibrationDraw.interrupted(prompt, index, seed=seed, temperature=chosen.temperature,
                                                   sources=sources)
            return calibration
        fingerprint = calibrators["top_k"].calibrator.fingerprint
        temperature = self.arm_plan.temperature

        def execution(item_id: str, rendering: str, index: int, seed: int | None) -> ExecutionDraw:
            item = by_key.get((item_id, rendering))
            if item is None:
                raise RunnerError(f"a request of {item_id}/{rendering} was recorded, but the stage has no such item; "
                                  "refused")
            return ExecutionDraw.interrupted(item, index, seed=seed, temperature=temperature,
                                             calibrator_fingerprint=fingerprint)
        return execution


# ---------- the stage run ----------

@dataclass
class StageOutcome:
    """What one command did: the pass, the counts, the stage status, the exit code and, when the pass raised, the
    class name of the exception (``error``, never its message)."""

    mode: str
    exit_code: int
    status: str | None = None
    stop_reason: str | None = None
    counts: Mapping[str, int] | None = None
    results: int | None = None
    partial: bool = False
    gate: PowerGate | None = None
    note: str | None = None
    error: str | None = None


def _ledger_status(path: Path) -> tuple[str, str | None]:
    if not path.exists():
        return "open", None
    data = _read_json(path)
    return data.get("status", "open"), data.get("stop_reason")


def _lease_gate(lease: ServeLease | None, ledger: BudgetLedger,
                allowance: RequestAllowance) -> Callable[[], None] | None:
    """The serving arm's admission gate: a request is admitted only while it can finish before the serve deadline."""
    if lease is None:
        return None

    def gate() -> None:
        if lease.admits(_wall(), allowance.timeout_s):
            return
        reason = ledger.stop_admission("max_seconds")
        raise AdmissionClosedError(
            f"Admission closed for the serving arm, stage {lease.stage}: a request now could run past the serve "
            f"deadline {lease.deadline_unix_s} (one serve per stage per approval); the stage is incomplete and its "
            "partial results are kept", reason=reason, status=ledger.status)
    return gate


def _write_runner_line(setup: StageSetup, outcome: StageOutcome, records_dir: Path, started: datetime) -> None:
    line = {"schema": RUNNER_SCHEMA, "plan_sha256": setup.plan.plan_sha256, "arm": setup.arm, "stage": setup.stage,
            "mode": outcome.mode, "started_at": _iso(started), "finished_at": _iso(_utc_now()),
            "counts": dict(outcome.counts or {}), "results": outcome.results, "partial": outcome.partial,
            "status": outcome.status, "stop_reason": outcome.stop_reason, "exit_code": outcome.exit_code,
            "gate": None if outcome.gate is None else {"decision": outcome.gate.decision,
                                                       "expected_power": outcome.gate.expected_power},
            "note": outcome.note, "error": outcome.error}
    _append_line(stage_directory(setup.arm, setup.stage, records_dir) / RUNNER_LOG, line, fsync=True)


def _session_error(session: StageSession | EvidenceSession) -> str | None:
    """The class name of the exception that ended the session, from its receipt; None when it ended normally."""
    last = session.receipt.get("last_session") or {}
    return last.get("error")


def _failure_probe(session: StageSession, setup: StageSetup, config: ModelConfig) -> Callable[[], None]:
    """7.4's post-failure host probe, handed to the recorder as ``after_failure``.

    After a draw failed in its request (a transport error, a gateway 502, a
    timeout), the generation host is checked again and the check recorded in
    the stage receipt; a dead host, or one serving other weights, raises
    :class:`HostFailure` (the session ends with the stage open, exit 1), a busy
    host lets the pass go on. The session's own probe is used, so the mapping
    of the check to the stop reason is the one of the recording module.
    """
    hosts, plan, arm = setup.hosts, setup.plan, setup.arm

    def after_failure() -> None:
        session._probe_after_failure(lambda: hosts.check_generation(plan, arm, config))

    return after_failure


def _generation_pass(setup: StageSetup, session: StageSession, client: CompletionClient, check: HostCheck,
                     calibrators: Mapping[str, Any] | None) -> StageOutcome:
    """Issue the draws the stage still lacks on the generation host; no evidence, nothing fitted or derived."""
    plan, arm, stage = setup.plan, setup.arm, setup.stage
    lease = None
    if setup.serving is not None:
        lease = ServeLease.from_receipt(setup.serving.serve_receipt, plan=plan, approval=setup.serving.approval,
                                        stage=stage)
        if client.config.endpoint != lease.endpoint_url + "/chat/completions":
            raise RunnerError(f"the client's endpoint {client.config.endpoint} is not the serve receipt's "
                              f"{lease.endpoint_url}; nothing was sent")
    recorder = setup.recorder(setup.paths.records, config=client.config, host_digest=check.observed or "",
                              session=session.receipt["sessions"], calibrators=calibrators,
                              gate=_lease_gate(lease, session.ledger, session.ledger.allowance),
                              after_failure=_failure_probe(session, setup, client.config))
    status, _ = _ledger_status(budget_path(arm, stage, setup.paths.records))
    if status == "complete":
        return StageOutcome(GENERATION, EXIT_COMPLETE, counts=recorder.counts,
                            note="the stage is complete; nothing to generate")
    if status != "open":
        raise RunnerError(f"the {arm}/{stage} stage is {status}; a generation pass admits nothing more. Run the "
                          "evidence pass (sampling arm) or the stage command (serving arm) to derive the results "
                          "of the fully recorded items")
    try:
        if stage == CALIBRATION_STAGE:
            corpora = setup.corpora()
            prompts = [prompt for corpus in corpora.values() for prompt in corpus.prompts]
            recorded = recorder.recorded(setup.interrupted_factory(None, prompts))
            with session.ledger:
                for kind, corpus in corpora.items():
                    collect_draws(corpus, plan_draw_settings(plan, arm, kind), client=client, ledger=session.ledger,
                                  sources=setup.arm_plan.estimate_sources,
                                  recorded=[draw for draw in recorded if draw.kind == kind],
                                  on_draw=recorder.on_draw, on_request=recorder.on_request)
        else:
            items = setup.execution_items()
            recorded = recorder.recorded(setup.interrupted_factory(calibrators, items))
            generate_draws(items, plan=plan, arm=arm, stage=stage, client=client, ledger=session.ledger,
                           calibrators=calibrators, recorded=recorded, on_draw=recorder.on_draw,
                           on_request=recorder.on_request)
    except BudgetExhaustedError:
        recorder.not_sent("admission_closed")
        raise
    except Exception as error:
        if getattr(error, "status_code", None) in (401, 403):
            recorder.not_sent("auth_failed")
        raise
    planned = setup.planned_draws()
    complete = fully_recorded(recorder.log.latest(), recorder.slot, planned)
    done = len(complete) == len(planned)
    note = (f"generation pass recorded every planned draw of {len(planned)} ensembles; the stage stays open until "
            "the evidence pass" if done else f"{len(complete)} of {len(planned)} ensembles fully recorded")
    return StageOutcome(GENERATION, EXIT_COMPLETE if done else EXIT_UNFINISHED, counts=recorder.counts, note=note)


def _settle_pass(setup: StageSetup, ledger: BudgetLedger, client: CompletionClient, full: Any, check: HostCheck,
                 calibrators: Mapping[str, Any] | None, session_number: int, lease: ServeLease | None,
                 mode: str, after_failure: Callable[[], None] | None = None,
                 on_batch: Callable[[], None] | None = None) -> StageOutcome:
    """The entry point of the stage with the recorded draws: the evidence pass, the results, the fit or the gate.

    ``after_failure`` is the generation session's post-failure probe (``both``
    generates here); the evidence pass has no generation host and passes none.
    ``on_batch`` is called once per evidence batch the pass posts (the session
    counts them in the receipt): per evidence file of an execution stage, and
    per calibration prompt whose pending draws the evidence pass settles.
    """
    plan, arm, stage, paths = setup.plan, setup.arm, setup.stage, setup.paths
    recorder = setup.recorder(paths.records, config=client.config, host_digest=check.observed or "",
                              session=session_number, calibrators=calibrators,
                              gate=_lease_gate(lease, ledger, ledger.allowance), after_failure=after_failure)
    policy = setup.evidence_policy()
    top_k_keep = None if policy is None else policy.top_k_keep
    texts: dict[tuple[str, str], dict[int, str]] = {}

    def posted_batch() -> None:
        if on_batch is not None:
            on_batch()

    def remember(draw: Any) -> None:
        if isinstance(draw, ExecutionDraw) and draw.response is not None:
            texts.setdefault((draw.item_id, draw.rendering), {})[draw.draw.draw_index] = draw.response.content
        recorder.on_draw(draw)

    def on_evidence(item: ExecutionItem, found: Mapping[int, FullDistributionEvidence]) -> None:
        recorder.on_evidence(item, found, top_k_keep=top_k_keep or 0, texts=texts.get(item.key, {}))
        posted_batch()

    status, stop_reason = _ledger_status(budget_path(arm, stage, paths.records))
    partial = status == "incomplete"
    planned = setup.planned_draws()
    if stage == CALIBRATION_STAGE:
        corpora = setup.corpora()
        prompts = [prompt for corpus in corpora.values() for prompt in corpus.prompts]
        recorded = recorder.recorded(setup.interrupted_factory(None, prompts))
        complete = fully_recorded(recorded, recorder.slot, planned)
        if partial:
            return StageOutcome(mode, EXIT_INCOMPLETE, status=status, stop_reason=stop_reason, partial=True,
                                counts=recorder.counts, note="a frozen limit closed the calibration; nothing is fitted")
        if mode == EVIDENCE and len(complete) != len(planned):
            raise RunnerError(f"the generation pass is not complete ({len(complete)} of {len(planned)} ensembles "
                              "recorded); rerun --pass generation with the generation host before the evidence pass")
        pending = {(draw.item_id, draw.rendering) for draw in recorded if draw.evidence_pending}

        def settled(draw: CalibrationDraw) -> None:
            """A prompt's pending draws settle in one batch: its first settled draw counts the batch posted."""
            recorder.on_draw(draw)
            key = (draw.item_id, draw.rendering)
            if draw.evidence_pending:
                pending.add(key)
            elif draw.failure_code is None and key in pending:
                pending.discard(key)
                recorder.counts[EVIDENCE_BATCHES] += 1
                posted_batch()

        try:
            run = run_calibration(plan, arm=arm, macro_rows=setup.items.macro_rows(plan.months.calibration),
                                  edgar_releases=setup.items.calibration_releases(), client=client, ledger=ledger,
                                  sources=setup.arm_plan.estimate_sources, full_distribution=full,
                                  recorded=recorded, on_draw=settled, on_request=recorder.on_request,
                                  calibrators_dir=paths.calibrators)
        except BudgetExhaustedError:
            recorder.not_sent("admission_closed")
            raise
        return StageOutcome(mode, EXIT_COMPLETE, counts=recorder.counts, results=len(run.paths),
                            note="calibrators written: " + ", ".join(str(path) for path in run.paths.values()))
    items = setup.execution_items()
    recorded = recorder.recorded(setup.interrupted_factory(calibrators, items))
    for draw in recorded:
        if isinstance(draw, ExecutionDraw) and draw.response is not None:
            texts.setdefault((draw.item_id, draw.rendering), {})[draw.draw.draw_index] = draw.response.content
    complete = fully_recorded(recorded, recorder.slot, planned)
    if partial:
        items = tuple(item for item in items if item.key in complete)
        if not items:
            return StageOutcome(mode, EXIT_INCOMPLETE, status=status, stop_reason=stop_reason, partial=True,
                                counts=recorder.counts, results=0,
                                note="a frozen limit closed the stage before any ensemble was fully recorded")
        chosen = {item.key for item in items}
        recorded = [draw for draw in recorded if (draw.item_id, draw.rendering) in chosen]
    elif mode == EVIDENCE and len(complete) != len(planned):
        raise RunnerError(f"the generation pass is not complete ({len(complete)} of {len(planned)} ensembles "
                          "recorded); rerun --pass generation with the generation host before the evidence pass")
    common = {"client": client, "ledger": ledger, "calibrators": calibrators, "full_distribution": full,
              "recorded": recorded, "on_draw": remember, "on_request": recorder.on_request,
              "on_evidence": on_evidence, "results_dir": paths.results}
    gate: PowerGate | None = None
    note = None
    try:
        if partial or stage == CONFIRMATORY_STAGE:
            run = execute_items(plan, items, arm=arm, stage=stage, **common)
        elif stage == PILOT_STAGE:
            pilot = run_pilot(plan, setup.items.macro, gates_dir=paths.gates, **common)
            run, gate = pilot.run, pilot.gate
            note = gate.summary()
        else:
            negative = run_negative_control(plan, setup.items.edgar.negative_control, arm=arm, **common)
            run = negative.execution
            note = ("negative-control report complete" if negative.report.complete
                    else "negative-control report partial: a frozen release has no result")
    except BudgetExhaustedError:
        recorder.not_sent("admission_closed")
        raise
    except Exception as error:
        if getattr(error, "status_code", None) in (401, 403):
            recorder.not_sent("auth_failed")
        raise
    exit_code = EXIT_INCOMPLETE if partial else _exit_of(run)
    if partial:
        note = f"partial: results derived for {len(run.results)} of {len(planned)} fully recorded ensembles"
    return StageOutcome(mode, exit_code, counts=recorder.counts, results=len(run.results), partial=partial,
                        gate=gate, note=note)


def _exit_of(run: ExecutionRun) -> int:
    return {"complete": EXIT_COMPLETE, "incomplete": EXIT_INCOMPLETE}.get(run.status, EXIT_UNFINISHED)


def run_stage(setup: StageSetup, mode: str) -> StageOutcome:
    """Run one pass of the stage inside its 7.4 session.

    The receipt and the runner line record the host checks, the status and the
    exit code of the pass; a pass that raised is recorded as refused or failed
    with the exception class (the session's rule), never as complete.
    """
    if mode not in PASSES:
        raise RunnerError(f"pass must be one of {list(PASSES)}; got {mode!r}")
    plan, arm, stage, paths, hosts = setup.plan, setup.arm, setup.stage, setup.paths, setup.hosts
    if mode == EVIDENCE and arm != SAMPLING_ARM:
        raise RunnerError("the serving arm has no evidence pass: its full-distribution variants are missing by design; "
                          "run its stages with --pass both")
    if stage == PILOT_STAGE and arm != plan.pilot.arm:
        raise RunnerError(f"the pilot runs on the gated {plan.pilot.arm} arm only")
    config = hosts.generation_config(plan, arm)
    setup.check_client_config(config)
    limits = setup.limits()
    allowance = setup.allowance(config)
    calibrators = None if stage == CALIBRATION_STAGE else setup.calibrators()
    started = _utc_now()
    if mode == EVIDENCE:
        session = EvidenceSession.open(plan, stage=stage, arm=arm, records_dir=paths.records,
                                       evidence_failures=EVIDENCE_FAILURES)
        ledger = BudgetLedger.open(budget_path(arm, stage, paths.records), plan, limits, allowance)
        outcome: StageOutcome | None = None
        try:
            with session:
                full = hosts.full_distribution(plan)
                check = session.require_host(hosts.check_full_distribution(plan, full))
                client = hosts.generation(plan, arm, ledger)
                outcome = _settle_pass(setup, ledger, client, full, check, calibrators, session.receipt["sessions"],
                                       None, mode, on_batch=lambda: session.count(EVIDENCE_BATCHES))
        finally:
            ledger.close()
            if outcome is None:
                outcome = StageOutcome(mode, session.exit_code, status=session.status)
            outcome.status, outcome.stop_reason = session.status, session.receipt["stop_reason"]
            outcome.error = _session_error(session)
            _write_runner_line(setup, outcome, paths.records, started)
        return outcome
    serving = setup.serving
    session = StageSession.open(plan, arm=arm, stage=stage, limits=limits, allowance=allowance,
                                approval=None if serving is None else serving.approval,
                                serve_receipt=None if serving is None else serving.serve_receipt,
                                records_dir=paths.records, evidence_failures=EVIDENCE_FAILURES)
    outcome = None
    try:
        with session:
            client = hosts.generation(plan, arm, session.ledger)
            check = session.require_host(hosts.check_generation(plan, arm, client.config))
            if mode == GENERATION:
                outcome = _generation_pass(setup, session, client, check, calibrators)
            else:
                full = None
                if setup.arm_plan.evidence is not None:
                    full = hosts.full_distribution(plan)
                    session.require_host(hosts.check_full_distribution(plan, full))
                lease = None
                if serving is not None:
                    lease = ServeLease.from_receipt(serving.serve_receipt, plan=plan, approval=serving.approval,
                                                    stage=stage)
                    if client.config.endpoint != lease.endpoint_url + "/chat/completions":
                        raise RunnerError(f"the client's endpoint {client.config.endpoint} is not the serve receipt's "
                                          f"{lease.endpoint_url}; nothing was sent")
                outcome = _settle_pass(setup, session.ledger, client, full, check, calibrators,
                                       session.receipt["sessions"], lease, mode,
                                       after_failure=_failure_probe(session, setup, client.config),
                                       on_batch=lambda: session.count(EVIDENCE_BATCHES))
    finally:
        if outcome is None:
            outcome = StageOutcome(mode, session.exit_code)
        outcome.status, outcome.stop_reason = session.status, session.receipt["stop_reason"]
        outcome.error = _session_error(session)
        if mode != GENERATION:
            outcome.exit_code = session.exit_code if outcome.exit_code == EXIT_COMPLETE else outcome.exit_code
        _write_runner_line(setup, outcome, paths.records, started)
    return outcome


# ---------- the gate ----------

@dataclass(frozen=True)
class GateOutcome:
    """The recorded gate's decision and what the frozen policy makes of it."""

    gate: PowerGate | None
    proceed: bool
    exit_code: int
    message: str


def check_gate(plan: StudyPlan, arm: str, *, results_dir: Path, gates_dir: Path,
               require: Callable[..., PowerGate | None] = require_gate_passed) -> GateOutcome:
    """The gate before an arm's full run, honouring ``pilot.gate_policy`` (user decision 2026-11-10: inform)."""
    try:
        gate = require(plan, arm, results_dir=results_dir, gates_dir=gates_dir)
    except GateStoppedError as stopped:
        policy = plan.pilot.gate_policy
        if policy == GATE_INFORM:
            return GateOutcome(stopped.gate, True, EXIT_COMPLETE,
                               f"{stopped.gate.summary()} Policy {GATE_INFORM}: the {arm} arm proceeds; the expected "
                               "power is reported")
        return GateOutcome(stopped.gate, False, EXIT_GATE_STOPPED,
                           f"{stopped.gate.summary()} Policy {policy}: the full run waits for a decision")
    if gate is None:
        return GateOutcome(None, True, EXIT_COMPLETE, f"the {arm} arm is not gated; it reports its achieved power")
    return GateOutcome(gate, True, EXIT_COMPLETE, gate.summary())


# ---------- commands ----------

def _add_study_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--private-root", type=Path, default=PRIVATE_ROOT,
                        help="the private study root (default runs/finance-sim)")
    parser.add_argument("--plan", type=Path, help="the frozen plan.json (default <private-root>/plan/plan.json)")
    parser.add_argument("--plan-sha256", required=True, help="the plan hash every record carries")
    parser.add_argument("--items", type=Path, help="the private items.json (default <private-root>/plan/items.json)")
    parser.add_argument("--edgar-dir", type=Path, help="the frozen EDGAR corpus (default data/edgar)")
    for name in ("records", "results", "calibrators", "gates", "approvals", "analysis", "stage-limits"):
        parser.add_argument(f"--{name}-dir", type=Path, help=f"default <private-root>/{name.replace('-', '_')}")


def _add_stage_arguments(parser: argparse.ArgumentParser, *, arm: bool = True) -> None:
    if arm:
        parser.add_argument("--arm", required=True, choices=ARMS)
    parser.add_argument("--pass", dest="mode", choices=PASSES, default=BOTH,
                        help="generation (the generation host up), evidence (the full-distribution host up, "
                             "sampling arm) or both (the serving arm, or fakes)")
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs" / "modal" / "gemma-3-4b-measurement.json",
                        help="the serving manifest (serving arm)")
    parser.add_argument("--serve-receipt", type=Path, help="the serve receipt of this stage (serving arm)")


def _paths(args: argparse.Namespace) -> StudyPaths:
    return StudyPaths.under(args.private_root, plan=args.plan, items=args.items, edgar=args.edgar_dir,
                            records=args.records_dir, results=args.results_dir, calibrators=args.calibrators_dir,
                            gates=args.gates_dir, approvals=args.approvals_dir, analysis=args.analysis_dir,
                            stage_limits=args.stage_limits_dir)


def _load(args: argparse.Namespace) -> tuple[StudyPaths, StudyPlan]:
    paths = _paths(args)
    return paths, load_plan(paths.plan, expected_sha256=args.plan_sha256)


def _serving_inputs(args: argparse.Namespace, plan: StudyPlan, paths: StudyPaths) -> ServingInputs | None:
    if args.arm != SERVING_ARM:
        if getattr(args, "serve_receipt", None) is not None:
            raise RunnerError("only the serving arm runs under a serve receipt")
        return None
    stage_limits = [paths.stage_limits / f"{stage}.json" for stage in SERVING_STAGES]
    approval = require_serving_approval(plan, args.manifest, stage_limits, approvals_dir=paths.approvals)
    if args.serve_receipt is None:
        raise RunnerError("the serving arm needs the serve receipt of this stage (--serve-receipt)")
    print(f"approval: {approval.path} matches plan {approval.plan_sha256}; maximum cost "
          f"{approval.presentation.max_usd_total} USD")
    return ServingInputs(approval=approval, serve_receipt=args.serve_receipt)


def _setup(args: argparse.Namespace, hosts: Hosts, stage: str) -> StageSetup:
    paths, plan = _load(args)
    items = load_items(paths.items, paths.edgar)
    return StageSetup(plan=plan, arm=args.arm, stage=stage, paths=paths, hosts=hosts, items=items,
                      serving=_serving_inputs(args, plan, paths))


def _print_outcome(setup: StageSetup, outcome: StageOutcome) -> None:
    print(json.dumps({"arm": setup.arm, "stage": setup.stage, "pass": outcome.mode, "status": outcome.status,
                      "stop_reason": outcome.stop_reason, "exit_code": outcome.exit_code,
                      "counts": dict(outcome.counts or {}), "results": outcome.results, "partial": outcome.partial,
                      "note": outcome.note}, sort_keys=True))


def command_stage(args: argparse.Namespace, hosts: Hosts) -> int:
    stage = STAGE_OF[args.command] if args.command != "evidence" else args.stage
    if args.command == "evidence":
        args.mode = EVIDENCE
    if args.command == "pilot":
        args.arm = SAMPLING_ARM
    setup = _setup(args, hosts, stage)
    if args.mode == BOTH and setup.arm == SAMPLING_ARM and isinstance(hosts, LiveHosts):
        raise RunnerError("--pass both is refused for the sampling arm on live hosts: its two local hosts are never "
                          "loaded together. Run --pass generation with llama-server up, then --pass evidence with "
                          "the full-distribution host; nothing was sent")
    if stage == CONFIRMATORY_STAGE and setup.plan.arms.arms[setup.arm].gated and args.mode != EVIDENCE:
        decision = check_gate(setup.plan, setup.arm, results_dir=setup.paths.results, gates_dir=setup.paths.gates)
        print(decision.message)
        if not decision.proceed:
            return decision.exit_code
    outcome = run_stage(setup, args.mode)
    _print_outcome(setup, outcome)
    if outcome.gate is not None and outcome.gate.decision == STOP_FOR_DECISION and (
            setup.plan.pilot.gate_policy != GATE_INFORM):
        return EXIT_GATE_STOPPED
    return outcome.exit_code


def command_gate(args: argparse.Namespace) -> int:
    paths, plan = _load(args)
    decision = check_gate(plan, args.arm, results_dir=paths.results, gates_dir=paths.gates)
    print(decision.message)
    return decision.exit_code


def command_check_hosts(args: argparse.Namespace, hosts: Hosts) -> int:
    paths, plan = _load(args)
    checks: list[HostCheck] = []
    which = args.host
    if which in (HOST_GENERATION, HOST_ALL):
        checks.append(hosts.check_generation(plan, args.arm, hosts.generation_config(plan, args.arm)))
    if which in (HOST_FULL_DISTRIBUTION, HOST_ALL) and args.arm == SAMPLING_ARM:
        checks.append(hosts.check_full_distribution(plan, hosts.full_distribution(plan)))
    for check in checks:
        print(json.dumps(check.to_dict(), sort_keys=True))
    return EXIT_COMPLETE if checks and all(check.ok for check in checks) else EXIT_REFUSED


def command_smoke_run(args: argparse.Namespace, hosts: Hosts) -> int:
    """The serving arm's smoke request: one reply must carry the declared top-k logprobs (task 9.4)."""
    args.arm = SERVING_ARM
    setup = _setup(args, hosts, SMOKE_STAGE)
    config = hosts.generation_config(setup.plan, SERVING_ARM)
    setup.check_client_config(config)
    session = StageSession.open(setup.plan, arm=SERVING_ARM, stage=SMOKE_STAGE, limits=setup.limits(),
                                allowance=setup.allowance(config), approval=setup.serving.approval,
                                serve_receipt=setup.serving.serve_receipt, records_dir=setup.paths.records)
    wanted = setup.arm_plan.top_logprobs
    with session:
        client = hosts.generation(setup.plan, SERVING_ARM, session.ledger)
        session.require_host(hosts.check_generation(setup.plan, SERVING_ARM, client.config))
        item = next(item for item in macro_items(setup.plan, setup.items.macro, stage=CONFIRMATORY_STAGE,
                                                 arm=SERVING_ARM) if item.rendering == SMOKE_RENDERING)
        with session.ledger:
            response = client.generate(item.prompt)
        counts = sorted({len(position.top_logprobs) for position in response.token_evidence})
        passed = bool(response.token_evidence) and counts == [wanted]
        _write_json(session.directory / "smoke.json", {
            "plan_sha256": setup.plan.plan_sha256, "stage": SMOKE_STAGE, "item_id": item.item_id,
            "rendering": item.rendering, "prompt_sha256": item.prompt_sha256,
            "response_sha256": _sha256_text(response.content), "usage": dict(response.usage or {}),
            "top_logprobs_declared": wanted, "top_logprobs_observed": counts, "positions": len(response.token_evidence),
            "passed": passed, "recorded_at": _iso(_utc_now())})
        if passed:
            session.complete()
    print(json.dumps({"stage": SMOKE_STAGE, "passed": passed, "top_logprobs_observed": counts,
                      "declared": wanted, "status": session.status}, sort_keys=True))
    return EXIT_COMPLETE if passed else EXIT_REFUSED


def command_teardown(args: argparse.Namespace) -> int:
    paths, plan = _load(args)
    entries = record_teardown(plan, args.arm, args.stage, args.stop_output, records_dir=paths.records)
    for entry in entries:
        print(json.dumps({"path": entry["path"], "kind": entry["kind"], "verified_stopped": entry["verified_stopped"],
                          "detail": entry["detail"]}, sort_keys=True))
    return EXIT_COMPLETE if all(entry["verified_stopped"] for entry in entries) else EXIT_UNFINISHED


def _stage_complete(paths: StudyPaths, arm: str, stage: str) -> bool:
    return _ledger_status(budget_path(arm, stage, paths.records))[0] == "complete"


def command_analyses(args: argparse.Namespace) -> int:
    paths, plan = _load(args)
    items = load_items(paths.items, paths.edgar)
    arms = tuple(args.arm) if args.arm else tuple(ARMS)
    if not args.allow_partial:
        missing = [f"{arm}/{stage}" for arm in arms for stage in ANALYSED_STAGES
                   if not _stage_complete(paths, arm, stage)]
        if missing:
            raise RunnerError(f"stages not complete: {missing}; the analyses are written once from complete stages "
                              "(pass --allow-partial to write them from partial results)")
    written: list[Path] = []
    for arm in arms:
        written += write_analysis(analysis_from_records(plan, arm=arm, results_dir=paths.results), paths.analysis)
        for stage in ANALYSED_STAGES:
            written += write_metrics(ensemble_metrics_from_records(plan, arm=arm, stage=stage,
                                                                   results_dir=paths.results), paths.analysis)
        report = negative_control_from_records(plan, items.edgar.negative_control, arm=arm, results_dir=paths.results)
        if not report.complete:
            print(f"negative-control report of the {arm} arm is partial: a frozen release has no result")
        written += write_report(report, paths.analysis)
        if arm == SAMPLING_ARM:
            for stage in CONVERGENCE_STAGES:
                written += write_table(convergence_from_records(plan, arm=arm, stage=stage, results_dir=paths.results,
                                                                records_dir=paths.records,
                                                                calibrators_dir=paths.calibrators), paths.analysis)
    for path in written:
        print(path)
    return EXIT_COMPLETE


def command_replay(args: argparse.Namespace) -> int:
    paths = _paths(args)
    private = paths.private
    argv = ["--plan", str(paths.plan), "--plan-sha256", args.plan_sha256, "--out-root", str(args.out_root),
            "--results-dir", str(paths.results), "--records-dir", str(paths.records),
            "--calibrators-dir", str(paths.calibrators), "--edgar-dir", str(paths.edgar), "--items", str(paths.items),
            "--fmp-cache", str(private.fmp_cache), "--fred-cache", str(private.fred_cache)]
    if args.compare is not None:
        argv += ["--compare", str(args.compare)]
    if args.require_guard:
        argv.append("--require-guard")
    return replay_module.main(argv)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_study", description=__doc__.split("\n\n", 1)[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     epilog=__doc__.split("Live route", 1)[1] if "Live route" in __doc__ else None)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-plan", help="assemble the items and freeze the plan (build_plan.py)")
    build.add_argument("--private-root", type=Path, default=PRIVATE_ROOT,
                       help="private output root (default runs/finance-sim)")
    build.add_argument("--seed", type=int, default=build_plan_module.DEFAULT_SEED, help="the plan seed")
    build.add_argument("--created-at", default=None, help="freeze timestamp, ISO 8601 with a timezone")
    build.add_argument("--manifest", type=Path, default=build_plan_module.DEFAULT_MANIFEST,
                       help="the serving manifest")
    hosts = commands.add_parser("check-hosts", help="the host checks of one arm, nothing admitted")
    _add_study_arguments(hosts)
    hosts.add_argument("--arm", required=True, choices=ARMS)
    hosts.add_argument("--host", choices=(HOST_GENERATION, HOST_FULL_DISTRIBUTION, HOST_ALL), default=HOST_ALL)
    for name, text in (("calibrate", "macro and EDGAR calibration of one arm"),
                       ("pilot", "the pilot of the gated arm and its power gate"),
                       ("full-run", "the full run of one arm (gate checked for the gated arm)"),
                       ("negative-control", "the negative control of one arm")):
        stage = commands.add_parser(name, help=text)
        _add_study_arguments(stage)
        _add_stage_arguments(stage, arm=name != "pilot")
    evidence = commands.add_parser("evidence", help="the evidence pass of a sampling-arm stage")
    _add_study_arguments(evidence)
    evidence.add_argument("--stage", required=True, choices=(CALIBRATION_STAGE, *EXECUTION_STAGES))
    evidence.add_argument("--arm", choices=(SAMPLING_ARM,), default=SAMPLING_ARM)
    evidence.add_argument("--manifest", type=Path, default=None)
    evidence.add_argument("--serve-receipt", type=Path, default=None)
    gate = commands.add_parser("gate", help="the recorded power gate under the frozen gate policy")
    _add_study_arguments(gate)
    gate.add_argument("--arm", choices=ARMS, default=SAMPLING_ARM)
    smoke = commands.add_parser("smoke-run", help="the serving arm's one smoke request")
    _add_study_arguments(smoke)
    smoke.add_argument("--manifest", type=Path, default=ROOT / "configs" / "modal" / "gemma-3-4b-measurement.json")
    smoke.add_argument("--serve-receipt", type=Path, required=True)
    teardown = commands.add_parser("teardown", help="record the host scripts' stop outputs in the stage receipt")
    _add_study_arguments(teardown)
    teardown.add_argument("--arm", required=True, choices=ARMS)
    teardown.add_argument("--stage", required=True)
    teardown.add_argument("--stop-output", type=Path, nargs="+", required=True, metavar="FILE")
    analyses = commands.add_parser("analyses", help="both pre-registered analyses, the negative-control report and "
                                                    "the convergence tables from the stored results")
    _add_study_arguments(analyses)
    analyses.add_argument("--arm", action="append", choices=ARMS)
    analyses.add_argument("--allow-partial", action="store_true")
    replay = commands.add_parser("replay", help="the offline worked-example replay (task 8.4)")
    _add_study_arguments(replay)
    replay.add_argument("--out-root", type=Path, required=True)
    replay.add_argument("--compare", type=Path)
    replay.add_argument("--require-guard", action="store_true")
    return parser


#: Refused before any request (exit 2): the study modules' refusals (each a ValueError) and the recording module's
#: refusal types, the same rule the stage receipt applies when a pass raises (``REFUSAL_TYPES``).
_REFUSALS = (RunnerError, PlanChangedError, ExecutionError, CalibrationError, NegativeControlError, PilotError,
             ApprovalError, ProfileError, *REFUSAL_TYPES)
#: A missing-evidence failure of the study modules (the full-distribution host gave no evidence): the stage stays
#: open (exit 1) and a rerun repeats only the missing batches; the sessions record it as open with evidence_missing.
EVIDENCE_FAILURES: tuple[type[BaseException], ...] = (EvidenceUnavailableError,)


def main(argv: Sequence[str] | None = None, *, hosts: Hosts | None = None,
         environ: Mapping[str, str] | None = None, transport: Any = None, today: date | None = None) -> int:
    """The runner's command line; ``hosts``, ``environ``, ``transport`` and ``today`` are the seams of the tests."""
    args = build_parser().parse_args(argv)
    environment = os.environ if environ is None else environ
    if args.command == "build-plan":
        forwarded = ["--private-root", str(args.private_root), "--seed", str(args.seed),
                     "--manifest", str(args.manifest)]
        if args.created_at is not None:
            forwarded += ["--created-at", args.created_at]
        return build_plan_module.main(forwarded, environ=environ, transport=transport, today=today)
    live = LiveHosts(environment) if hosts is None else hosts
    try:
        if args.command in STAGE_OF or args.command == "evidence":
            return command_stage(args, live)
        if args.command == "gate":
            return command_gate(args)
        if args.command == "check-hosts":
            return command_check_hosts(args, live)
        if args.command == "smoke-run":
            return command_smoke_run(args, live)
        if args.command == "teardown":
            return command_teardown(args)
        if args.command == "analyses":
            return command_analyses(args)
        return command_replay(args)
    except GateStoppedError as stopped:
        print(f"run_study: gate stopped: {stopped}", file=sys.stderr)
        return EXIT_GATE_STOPPED
    except AdmissionClosedError as closed:
        print(f"run_study: incomplete: {closed}", file=sys.stderr)
        return EXIT_INCOMPLETE
    except BudgetExhaustedError as exhausted:
        # RequestControl's plain refusal once the capped budget is spent: a frozen limit, like the ledger's own.
        print(f"run_study: incomplete: {exhausted}", file=sys.stderr)
        return EXIT_INCOMPLETE
    except (HostFailure, EvidenceUnavailableError) as failure:
        print(f"run_study: the stage stays open: {failure}", file=sys.stderr)
        return EXIT_UNFINISHED
    except _REFUSALS as refusal:
        print(f"run_study: refused: {type(refusal).__name__}: {refusal}", file=sys.stderr)
        return EXIT_REFUSED


__all__ = [
    "BOTH",
    "EVIDENCE",
    "EXIT_GATE_STOPPED",
    "GATE_BLOCK",
    "GATE_INFORM",
    "GENERATION",
    "PASSES",
    "RUNNER_LOG",
    "RUNNER_SCHEMA",
    "SMOKE_STAGE",
    "STAGE_OF",
    "GateOutcome",
    "Hosts",
    "LiveHosts",
    "OutcomeFields",
    "RequestFields",
    "RunnerError",
    "ServingInputs",
    "StageOutcome",
    "StageRecorder",
    "StageSetup",
    "StudyItems",
    "StudyPaths",
    "build_parser",
    "check_gate",
    "fully_recorded",
    "load_items",
    "main",
    "run_stage",
]


if __name__ == "__main__":
    raise SystemExit(main())
