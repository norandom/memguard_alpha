"""End-to-end offline validation of the study runner (task 9.1; Requirements 8.2, 10.7, 11.3, 11.6).

One runner, every stage in order, nothing live: the plan is built and frozen on
the recorded FRED responses of the 7.1 tests (served at the HTTP adapter) and
recorded FMP and DTB3 bodies (the injectable transport, synthetic month-end
closes and rates for every month of the study), the sampling arm is calibrated,
piloted, gated, run and negative-controlled on the 7.7 generation double behind
the 7.4 gateway double and a full-distribution double that serves the plan's
pinned digest, the serving arm runs under a written approval record and serve
receipts on the same generation double with the Modal profile's settings, the
analyses are written, and the replay of task 8.4 runs under ``--require-guard``
in fresh processes with every socket connection forbidden, reproducing the
analyses byte for byte. Every file goes under ``tmp_path``; nothing is written
under ``runs/`` or ``docs/``, no model host runs and no credential is read.
"""

from __future__ import annotations

import calendar
import contextlib
import hashlib
import io
import json
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import replace
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from deploy.modal.stage_limits import STAGES
from recall_guard.core.full_distribution import FullDistributionEvidence, logprobs_from_logits
from recall_guard.core.request_control import RequestError
from recall_guard.dataset import fred
from scripts.finance_sim import items as items_module
from scripts.finance_sim import run_study
from scripts.finance_sim.calibrate import CALIBRATION_STAGE, CalibrationDraw, CalibrationError, calibrator_path
from scripts.finance_sim.convergence import read_evidence_file
from scripts.finance_sim.execute import ExecutionDraw, load_result, macro_items, planned_keys, result_path
from scripts.finance_sim.figures import PUBLISHED_FIGURES
from scripts.finance_sim.limits import (
    EXIT_COMPLETE,
    EXIT_INCOMPLETE,
    EXIT_REFUSED,
    EXIT_UNFINISHED,
    ApprovalError,
    BudgetLedger,
    LedgerCallBudget,
    StudyLimits,
    approval_path,
    approval_template,
    budget_path,
    present_serving_arm,
    require_serving_approval,
)
from scripts.finance_sim.pilot import STOP_FOR_DECISION, GateStoppedError, gate_path, load_gate
from scripts.finance_sim.plan import load_plan
from scripts.finance_sim.recording import (
    PROBE_PROMPT,
    DrawLog,
    HostCheckError,
    check_full_distribution_host,
    check_llama_server,
    check_serving_host,
    draw_log_path,
    evidence_path,
    read_records,
    receipt_path,
    request_allowance,
)
from scripts.finance_sim.replay import MANIFEST_NAME, PRIVATE_DIR, PUBLISHED_DIR
from scripts.finance_sim.run_study import EXIT_GATE_STOPPED, RUNNER_LOG, load_items
from scripts.probe_modal_model import APPROVAL_ASSERTIONS
from tests.scripts.finance_sim.test_build_plan import CREATED_AT, ENVIRON, SEED, TODAY, install_fakes
from tests.scripts.finance_sim.test_execute import FakeHost, ProcessDied, digest, memorized
from tests.scripts.finance_sim.test_items import RecordedTransport, observations_body
from tests.scripts.finance_sim.test_recording import ENVIRONMENT as SAMPLING_ENVIRONMENT
from tests.scripts.finance_sim.test_recording import FakeGateway, sampling_config
from tests.scripts.finance_sim.test_recording_hosts import (
    FD_BASE,
    GEMMA,
    GEMMA_MANIFEST,
    MODAL_ENDPOINT,
    MODAL_TOKEN,
    Connection,
    ModelsRoute,
    serving_config,
)

ROOT = Path(__file__).resolve().parents[3]
SERVING_ENVIRONMENT = {"MODAL_PROXY_BEARER": MODAL_TOKEN}
FIRST_CLOSE, LAST_CLOSE = (2015, 12), (2026, 7)
#: Synthetic month-end adjusted closes per symbol: six decimals for two of them, two for the gold ETF as FMP
#: quotes it (the 8.4 fixture's design: distinctive values, or runs of consecutive values, for the guard).
CLOSE_BASE = {"SWDA.L": 100.0, "XLK": 250.0, "IAU": 60.0}
#: The fake full-distribution host's vocabulary and response length (the 7.12 double's shape).
N_VOCAB, POSITIONS = 50, 6


def run(argv: list[str], **kwargs) -> SimpleNamespace:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = run_study.main(argv, **kwargs)
    return SimpleNamespace(code=code, stdout=out.getvalue(), stderr=err.getvalue())


# ---------- the recorded market bodies: every month of the study ----------

def month_ends() -> list[date]:
    year, month = FIRST_CLOSE
    days = []
    while (year, month) <= LAST_CLOSE:
        days.append(date(year, month, calendar.monthrange(year, month)[1]))
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
    return days


def close_of(symbol: str, k: int) -> float:
    if symbol == "IAU":
        return round(CLOSE_BASE[symbol] * (1.0071 ** k), 2)
    shift = 5 if symbol == "XLK" else 0
    return round(CLOSE_BASE[symbol] * (1.0113 ** k) * (1 + 1e-7 * ((k * 13 + shift) % 17)), 6)


def fmp_body(symbol: str) -> list[dict]:
    rows = [{"symbol": symbol, "date": day.isoformat(), "adjOpen": close_of(symbol, k), "adjHigh": close_of(symbol, k),
             "adjLow": close_of(symbol, k), "adjClose": close_of(symbol, k), "volume": 1000}
            for k, day in enumerate(month_ends())]
    return list(reversed(rows))


def dtb3_rows() -> list[tuple[str, str]]:
    return [(day.isoformat(), f"{4.21 + 0.03 * ((k * 7) % 23):.2f}") for k, day in enumerate(month_ends())]


def market_routes(url: str, params: dict) -> tuple[int, bytes]:
    if url == items_module.FMP_URL:
        return 200, json.dumps(fmp_body(params["symbol"])).encode()
    if url == f"{fred.BASE_URL}/observations" and params.get("series_id") == "DTB3":
        return 200, json.dumps(observations_body(dtb3_rows(), "2026-11-09")).encode()
    raise AssertionError(f"unexpected transport request to {url}")


# ---------- the hosts on fakes ----------

def pinned_evidence(pin: str, prompt: str, text: str, keep: int) -> FullDistributionEvidence:
    """The 7.12 double's evidence of a seeded logit vector, keeping ``min(keep, N_VOCAB)`` alternatives per position
    as the host guarantees, under the plan's pinned GGUF digest; the realized token is boosted for a memorized prompt."""
    rng = np.random.default_rng(int(digest(f"full:{digest(prompt)}:{digest(text)}")[:16], 16))
    logits = rng.normal(0.0, 1.5, size=(POSITIONS, N_VOCAB))
    tokens = rng.integers(0, N_VOCAB, size=POSITIONS)
    logits[np.arange(POSITIONS), tokens] += 4.0 if memorized(prompt) else 1.0
    logprobs = logprobs_from_logits(logits)
    order = np.argsort(-logprobs, axis=1, kind="stable")
    kept = min(keep, N_VOCAB)
    return FullDistributionEvidence(
        model_digest=pin, n_vocab=N_VOCAB, response_token_ids=tuple(int(token) for token in tokens),
        logprob=tuple(float(logprobs[i, token]) for i, token in enumerate(tokens)),
        entropy=tuple(float(-(np.exp(row) * row).sum()) for row in logprobs),
        rank=tuple(int(np.flatnonzero(order[i] == token)[0]) + 1 for i, token in enumerate(tokens)),
        top_logprobs=tuple(tuple((int(j), float(logprobs[i, j])) for j in order[i, :kept]) for i in range(POSITIONS)))


class FakeFullHost:
    """The full-distribution host behind its gateway double: batches only, the pinned digest, a probe batch.

    ``fail_after`` makes every study batch after that many fail in transport (a
    host that dies mid-pass); the probe batch of the host check still answers.
    """

    def __init__(self, pin: str, *, fail_after: int | None = None) -> None:
        self.base_url = FD_BASE
        self.model_digest = pin
        self.fail_after = fail_after
        self.batch_calls: list[tuple[str, tuple[str, ...], int, float | None]] = []

    def evidence(self, prompt_text: str, response_text: str, *, top_k_keep: int = 100) -> FullDistributionEvidence:
        pytest.fail("the runner requested evidence per draw; one batch per item and rendering")

    def evidence_batch(self, prompt_text: str, response_texts, *, top_k_keep: int = 100,
                       timeout_s: float | None = None) -> tuple[FullDistributionEvidence, ...]:
        self.batch_calls.append((prompt_text, tuple(response_texts), top_k_keep, timeout_s))
        if self.fail_after is not None and prompt_text != PROBE_PROMPT and len(self.batches) > self.fail_after:
            raise RequestError("connection refused", code="transport_error", retryable=True)
        return tuple(pinned_evidence(self.model_digest, prompt_text, text, top_k_keep) for text in response_texts)

    @property
    def batches(self) -> list[tuple[str, tuple[str, ...], int, float | None]]:
        return [call for call in self.batch_calls if call[0] != PROBE_PROMPT]


class Wall:
    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class AdvancingHost(FakeHost):
    """The generation double whose every request advances a wall clock (the serve deadline scenario)."""

    def __init__(self, budget, *, wall: Wall, step_s: float, **kwargs) -> None:
        super().__init__(budget, **kwargs)
        self.wall, self.step_s = wall, step_s

    def generate(self, prompt: str, **parameters):
        self.wall.now += self.step_s
        return super().generate(prompt, **parameters)


class DyingHost(FakeHost):
    """The generation double whose host dies at call ``dies_at``: that request and every later one fail in
    transport (the client records them, never retries), and the gateway stops answering from then on."""

    def __init__(self, budget, *, gateway: FakeGateway, dies_at: int, **kwargs) -> None:
        super().__init__(budget, **kwargs)
        self.gateway, self.dies_at = gateway, dies_at

    def generate(self, prompt: str, **parameters):
        index = len(self.calls)
        if index >= self.dies_at:
            self.gateway.unreachable = True
            self.failures[index] = RequestError("connection refused", code="transport_error", retryable=True)
        return super().generate(prompt, **parameters)


class GuardedHost(FakeHost):
    """The generation double admitting through RequestControl's guarded path, as the live providers do: a spent
    request cap raises the plain BudgetExhaustedError at ``remaining == 0`` before the ledger is asked."""

    def __init__(self, budget, **kwargs) -> None:
        super().__init__(budget, **kwargs)
        run = self._control.run
        self._control.run = lambda operation, **options: run(operation, admission_guard=lambda: None, **options)


class FakeHosts:
    """The runner's host seam on fakes: the 7.7 generation double, the 7.4 gateway doubles, the pinned digest.

    ``prose`` answers every generation request with unparseable text (the live
    calibration attempt 1, whose every draw failed to parse); ``evidence_fail_after``
    is the full-distribution host dying after that many study batches; ``guarded``
    admits through RequestControl's guarded path like the live providers.
    """

    def __init__(self, pin: str, *, crash_at: int | None = None, dies_at: int | None = None,
                 timeout_s: float | None = None, wall: Wall | None = None, step_s: float = 0.0,
                 prose: bool = False, evidence_fail_after: int | None = None, guarded: bool = False) -> None:
        self.pin, self.crash_at, self.dies_at = pin, crash_at, dies_at
        self.timeout_s, self.wall, self.step_s, self.prose = timeout_s, wall, step_s, prose
        self.evidence_fail_after, self.guarded = evidence_fail_after, guarded
        self.gateway = FakeGateway()
        self.models = ModelsRoute()
        self.generation_clients: list[FakeHost] = []
        self.full_clients: list[FakeFullHost] = []

    def generation_config(self, plan, arm):
        if arm == "sampling":
            return sampling_config() if self.timeout_s is None else sampling_config(timeout_s=self.timeout_s)
        return serving_config()

    def generation(self, plan, arm, ledger):
        config = self.generation_config(plan, arm)
        if self.wall is not None:
            host = AdvancingHost(LedgerCallBudget(ledger), wall=self.wall, step_s=self.step_s, configuration=config)
        elif self.dies_at is not None:
            host = DyingHost(LedgerCallBudget(ledger), gateway=self.gateway, dies_at=self.dies_at,
                             configuration=config)
        elif self.guarded:
            host = GuardedHost(LedgerCallBudget(ledger), configuration=config)
        else:
            host = FakeHost(LedgerCallBudget(ledger), configuration=config, crash_at=self.crash_at,
                            prose=range(1000) if self.prose else ())
        self.generation_clients.append(host)
        return host

    def full_distribution(self, plan):
        client = FakeFullHost(self.pin, fail_after=self.evidence_fail_after)
        self.full_clients.append(client)
        return client

    def check_generation(self, plan, arm, config):
        if arm == "sampling":
            return check_llama_server(config, expected_digest=self.pin, environment=SAMPLING_ENVIRONMENT,
                                      get=self.gateway)
        return check_serving_host(config, expected_model=GEMMA, environment=SERVING_ENVIRONMENT, get=self.models)

    def check_full_distribution(self, plan, client):
        return check_full_distribution_host(client, expected_digest=self.pin,
                                            connect=lambda address, timeout: Connection())


# ---------- the serving arm's approval and serve receipts ----------

def approve(base: Path, plan, root: Path) -> tuple[Path, object]:
    raw = json.loads(GEMMA_MANIFEST.read_text(encoding="utf-8"))
    raw["target"]["endpoint"] = MODAL_ENDPOINT
    manifest = base / "gemma-3-4b-measurement.json"
    manifest.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    stages = [root / "plan" / "stage_limits" / f"{stage}.json" for stage in STAGES]
    presentation = present_serving_arm(plan, manifest, stages)
    record = approval_template(presentation) | dict.fromkeys(APPROVAL_ASSERTIONS, True) | {
        "provider_budget_reference": "Modal workspace budget reviewed 2026-11-10 (offline validation)",
        "approved_by": "operator", "approved_at": "2026-11-10T12:30:00+00:00"}
    path = approval_path(plan.plan_sha256, root / "approvals")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")
    return manifest, require_serving_approval(plan, manifest, stages, approvals_dir=root / "approvals")


def write_serve_receipt(path: Path, plan, approval, stage: str, *, started: float,
                        deadline: float | None = None) -> Path:
    limits = approval.limits_for(stage)
    stage_sha = next(item.sha256 for item in approval.presentation.stages if item.stage == stage)
    receipt = {
        "mode": "serve", "status": "serving", "manifest": approval.presentation.manifest,
        "manifest_sha256": approval.presentation.manifest_sha256, "plan_sha256": plan.plan_sha256,
        "stage_limits": {"plan_sha256": plan.plan_sha256, "stage": stage, "max_requests": limits.max_requests,
                         "max_output_tokens": limits.max_output_tokens, "max_seconds": limits.max_seconds,
                         "max_usd": limits.max_usd},
        "stage_limits_sha256": stage_sha, "volume": "recall-guard-gemma-3-4b-weights",
        "lifetime_seconds": limits.max_seconds, "lifecycle": {}, "endpoint_env": "MODAL_ENDPOINT_URL",
        "credential_env": "MODAL_PROXY_BEARER", "approval_gate": "study_plan_record_checked_by_runner",
        "gpu_started_at_unix_s": started, "app_id": f"ap-{stage}", "endpoint_url": MODAL_ENDPOINT,
        "deadline_unix_s": started + limits.max_seconds if deadline is None else deadline,
        "chat_completions_url": MODAL_ENDPOINT + "/chat/completions",
    }
    path.write_text(json.dumps(receipt), encoding="utf-8")
    return path


def forbid_sockets(patch: pytest.MonkeyPatch) -> None:
    def forbidden(*args, **kwargs):
        pytest.fail("the runner attempted a network connection")

    patch.setattr(socket, "create_connection", forbidden)
    patch.setattr(socket.socket, "connect", forbidden)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    forbid_sockets(monkeypatch)


# ---------- the whole study through the runner ----------

@pytest.fixture(scope="module")
def study(tmp_path_factory) -> SimpleNamespace:
    """Every stage in order through ``run_study.main`` on fakes, under one temporary private root."""
    base = tmp_path_factory.mktemp("run-study")
    root = base / "private"
    with pytest.MonkeyPatch.context() as patch:
        forbid_sockets(patch)
        install_fakes(patch, base / "working")
        built = run(["build-plan", "--private-root", str(root), "--created-at", CREATED_AT, "--seed", str(SEED)],
                    environ=ENVIRON, transport=RecordedTransport(market_routes), today=TODAY)
        assert built.code == 0, built.stderr
        receipt = json.loads((root / "plan" / "receipt.json").read_text(encoding="utf-8"))
        plan = load_plan(root / "plan" / "plan.json", expected_sha256=receipt["plan_sha256"])
        pin = plan.arms.arms["sampling"].artifact_digest
        hosts = FakeHosts(pin)
        common = ["--private-root", str(root), "--plan-sha256", plan.plan_sha256]
        steps: dict[str, SimpleNamespace] = {}

        def step(name: str, argv: list[str]) -> SimpleNamespace:
            before, before_full = len(hosts.generation_clients), len(hosts.full_clients)
            result = run([*argv, *common], hosts=hosts)
            result.hosts = hosts.generation_clients[before:]
            result.full = hosts.full_clients[before_full:]
            steps[name] = result
            return result

        step("check-hosts", ["check-hosts", "--arm", "sampling"])
        step("calibrate-generation", ["calibrate", "--arm", "sampling", "--pass", "generation"])
        step("calibrate-evidence", ["calibrate", "--arm", "sampling", "--pass", "evidence"])
        step("pilot-generation", ["pilot", "--pass", "generation"])
        step("pilot-evidence", ["pilot", "--pass", "evidence"])
        step("gate", ["gate", "--arm", "sampling"])
        step("full-run-generation", ["full-run", "--arm", "sampling", "--pass", "generation"])
        step("full-run-evidence", ["evidence", "--stage", "full_run"])
        step("negative-generation", ["negative-control", "--arm", "sampling", "--pass", "generation"])
        step("negative-evidence", ["negative-control", "--arm", "sampling", "--pass", "evidence"])
        manifest, approval = approve(base, plan, root)
        started = time.time() - 60
        receipts = {stage: write_serve_receipt(base / f"serve-{stage}.json", plan, approval, stage, started=started)
                    for stage in ("smoke_run", "calibration", "full_run", "negative_control")}

        def serving(stage: str) -> list[str]:
            return ["--manifest", str(manifest), "--serve-receipt", str(receipts[stage])]

        step("smoke-run", ["smoke-run", *serving("smoke_run")])
        step("serving-calibration", ["calibrate", "--arm", "serving", "--pass", "both", *serving("calibration")])
        step("serving-full-run", ["full-run", "--arm", "serving", "--pass", "both", *serving("full_run")])
        step("serving-negative", ["negative-control", "--arm", "serving", "--pass", "both",
                                  *serving("negative_control")])
        stop = root / "records" / "sampling" / "full_run" / "llama-server-stop.txt"
        stop.write_text("Stopped: no host or gateway process remains; nothing listens on 11570 or 11571.\n",
                        encoding="utf-8")
        step("teardown", ["teardown", "--arm", "sampling", "--stage", "full_run", "--stop-output", str(stop)])
        step("analyses", ["analyses"])
        yield SimpleNamespace(base=base, root=root, plan=plan, pin=pin, receipt=receipt, steps=steps, hosts=hosts,
                              manifest=manifest, approval=approval, receipts=receipts, common=common, stop=stop)


def stage_receipt(study, arm: str, stage: str) -> dict:
    return json.loads(receipt_path(arm, stage, study.root / "records").read_text(encoding="utf-8"))


def budget(study, arm: str, stage: str) -> dict:
    return json.loads(budget_path(arm, stage, study.root / "records").read_text(encoding="utf-8"))


def planned_requests(plan, arm: str, stage: str) -> int:
    return plan.limits.stage(arm, stage).max_requests


def outcome_of(result: SimpleNamespace) -> dict:
    return json.loads(result.stdout.strip().splitlines()[-1])


def runner_lines(records: Path, stage: str) -> list[dict]:
    text = (records / "sampling" / stage / RUNNER_LOG).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines()]


def read_receipt(records: Path, stage: str) -> dict:
    return json.loads(receipt_path("sampling", stage, records).read_text(encoding="utf-8"))


# ---------- the plan and the host checks ----------

def test_the_plan_is_frozen_through_the_runner_on_recorded_responses(study):
    assert study.receipt["status"] == "frozen" and study.plan.plan_sha256 == study.receipt["plan_sha256"]
    assert study.receipt["counts"]["realized_missing"] == 0
    assert (study.root / "plan" / "items.json").is_file()
    assert {path.name for path in (study.root / "data" / "fmp").iterdir()} == {
        f"{symbol}_2015-12-01_2026-07-31.json" for symbol in CLOSE_BASE}
    assert any(path.name.startswith("DTB3_current_") for path in (study.root / "fred").iterdir())


def test_check_hosts_reports_the_pinned_hosts_without_admitting_a_request(study):
    result = study.steps["check-hosts"]
    assert result.code == EXIT_COMPLETE, result.stderr
    checks = [json.loads(line) for line in result.stdout.splitlines()]
    assert [(check["host"], check["status"], check["observed"]) for check in checks] == [
        ("llama_server", "ok", study.pin), ("full_distribution", "ok", study.pin)]
    assert result.hosts == [] and MODAL_TOKEN not in result.stdout
    assert result.full[0].batch_calls[0][0] == PROBE_PROMPT and result.full[0].batches == []


# ---------- the sampling arm, the hosts loaded one at a time ----------

def test_calibration_runs_in_two_passes_with_the_hosts_loaded_one_at_a_time(study):
    plan, generation, evidence = study.plan, study.steps["calibrate-generation"], study.steps["calibrate-evidence"]
    assert generation.code == EXIT_COMPLETE, generation.stderr
    assert len(generation.hosts[0].calls) == planned_requests(plan, "sampling", CALIBRATION_STAGE) == 944
    assert generation.full == [] and "every planned draw" in outcome_of(generation)["note"]
    open_receipt = outcome_of(generation)
    assert (open_receipt["status"], open_receipt["pass"]) == ("open", "generation")
    assert evidence.code == EXIT_COMPLETE, evidence.stderr
    assert evidence.hosts[0].calls == [] and len(evidence.full[0].batches) == 2 * 24 + 140
    assert all(call[3] == plan.arms.arms["sampling"].evidence.timeout_s for call in evidence.full[0].batches)
    receipt = stage_receipt(study, "sampling", CALIBRATION_STAGE)
    assert (receipt["status"], receipt["exit_code"]) == ("complete", EXIT_COMPLETE)
    assert [check["host"] for check in receipt["host_checks"]] == ["llama_server", "full_distribution"]
    assert [session["kind"] for session in (receipt["last_session"],)] == ["evidence"] and receipt["sessions"] == 2
    assert budget(study, "sampling", CALIBRATION_STAGE)["counters"]["requests"] == 944
    for source in ("top_k", "full_distribution"):
        for kind in ("macro", "edgar"):
            assert calibrator_path("sampling", source, kind, study.root / "calibrators").is_file()
    first = plan.months.calibration[0]
    lines = read_records("sampling", CALIBRATION_STAGE, f"macro-{first}", study.root / "records")
    assert [line["kind"] for line in lines] == ["request", "outcome"] * 16
    requests = [line for line in lines if line["kind"] == "request"]
    assert {line["rendering"] for line in requests} == {"identifying", "anonymised"}
    assert all(line["host_digest"] == study.pin and line["config"]["draws"] == 8 for line in requests)
    assert all(line["source"] == {"sources": ["top_k", "full_distribution"]} for line in requests)
    assert all(line["seed"] == plan.arms.arms["sampling"].seed_base + line["draw_index"] for line in requests)
    outcomes = [line for line in lines if line["kind"] == "outcome"]
    assert all(line["sent"] is True and line["response"] is None and "response_text" not in line["draw"]
               for line in outcomes)
    log = DrawLog(draw_log_path("sampling", CALIBRATION_STAGE, study.root / "records"), plan=plan,
                  loads=CalibrationDraw.from_dict)
    latest = log.latest()
    assert len(latest) == 944 and all(not draw.evidence_pending for draw in latest if draw.failure_code is None)
    versions = sum(1 for _ in log.path.read_text(encoding="utf-8").splitlines())
    assert versions == 944 + sum(1 for draw in latest if draw.failure_code is None)
    assert MODAL_TOKEN not in log.path.read_text(encoding="utf-8")


def test_the_pilot_runs_through_the_guarded_execution_and_the_gate_follows_the_frozen_policy(study):
    plan, generation, evidence = study.plan, study.steps["pilot-generation"], study.steps["pilot-evidence"]
    assert generation.code == EXIT_COMPLETE and len(generation.hosts[0].calls) == 192 == planned_requests(
        plan, "sampling", "pilot")
    assert evidence.code == EXIT_COMPLETE, evidence.stderr
    assert len(evidence.full[0].batches) == 24 and evidence.hosts[0].calls == []
    gate = load_gate(gate_path(plan.plan_sha256, study.root / "gates"))
    assert gate.decision == STOP_FOR_DECISION and plan.pilot.gate_policy == "inform"
    assert "Decision: stop_for_decision" in evidence.stdout
    decision = study.steps["gate"]
    assert decision.code == EXIT_COMPLETE and "proceeds" in decision.stdout
    stored = {key: load_result(result_path("sampling", "pilot", *key, study.root / "results"))
              for key in planned_keys(plan, "sampling", "pilot")}
    assert len(stored) == 24 and all(result.ensemble_draws == 8 for result in stored.values())
    assert stage_receipt(study, "sampling", "pilot")["status"] == "complete"


def test_the_full_run_and_the_negative_control_complete_within_the_frozen_limits(study):
    plan = study.plan
    for stage, generation, evidence in (("full_run", "full-run-generation", "full-run-evidence"),
                                        ("negative_control", "negative-generation", "negative-evidence")):
        first, second = study.steps[generation], study.steps[evidence]
        assert first.code == EXIT_COMPLETE, first.stderr
        assert second.code == EXIT_COMPLETE, second.stderr
        keys = planned_keys(plan, "sampling", stage)
        assert len(first.hosts[0].calls) == planned_requests(plan, "sampling", stage) == 8 * len(keys)
        assert len(second.full[0].batches) == len(keys) and second.hosts[0].calls == []
        counters = budget(study, "sampling", stage)["counters"]
        assert counters["requests"] == 8 * len(keys)
        receipt = stage_receipt(study, "sampling", stage)
        assert (receipt["status"], receipt["exit_code"]) == ("complete", EXIT_COMPLETE)
        for key in keys:
            result = load_result(result_path("sampling", stage, *key, study.root / "results"))
            parsed = [draw for draw in result.recorded_draws if draw.parsed is not None]
            evidence_file = evidence_path("sampling", stage, *key, study.root / "records")
            found = read_evidence_file(evidence_file, plan, result)
            assert len(found) == len(parsed) and all(entry.model_digest == study.pin for entry in found)
            lines = read_records("sampling", stage, key[0], study.root / "records")
            assert sum(1 for line in lines if line["kind"] == "request" and line["rendering"] == key[1]) == 8
    assert "complete" in outcome_of(study.steps["negative-evidence"])["note"]
    assert study.steps["full-run-generation"].stdout.splitlines()[0].startswith("Power gate")


# ---------- the serving arm under its approval and serve lease ----------

def test_the_serving_arm_runs_under_its_approval_and_serve_lease(study):
    plan = study.plan
    smoke = study.steps["smoke-run"]
    assert smoke.code == EXIT_COMPLETE, smoke.stderr
    assert json.loads(smoke.stdout.splitlines()[-1])["passed"] is True and len(smoke.hosts[0].calls) == 1
    smoke_record = json.loads((study.root / "records" / "serving" / "smoke_run" / "smoke.json").read_text("utf-8"))
    assert smoke_record["top_logprobs_observed"] == [5] and "response_text" not in smoke_record
    assert stage_receipt(study, "serving", "smoke_run")["status"] == "complete"
    for stage, name in (("calibration", "serving-calibration"), ("full_run", "serving-full-run"),
                        ("negative_control", "serving-negative")):
        result = study.steps[name]
        assert result.code == EXIT_COMPLETE, result.stderr
        assert "approval:" in result.stdout and MODAL_TOKEN not in result.stdout
        host = result.hosts[0]
        assert len(host.calls) == planned_requests(plan, "serving", stage)
        assert all("seed" not in parameters for _, parameters in host.calls) and result.full == []
        receipt = stage_receipt(study, "serving", stage)
        assert (receipt["status"], receipt["exit_code"]) == ("complete", EXIT_COMPLETE)
        assert receipt["host_checks"][0]["host"] == "serving" and receipt["host_checks"][0]["observed"] == GEMMA
        assert (study.root / "records" / "serving" / stage / "serve.json").is_file()
    for key in planned_keys(plan, "serving", "full_run"):
        assert not evidence_path("serving", "full_run", *key, study.root / "records").exists()
        result = load_result(result_path("serving", "full_run", *key, study.root / "results"))
        assert result.variant("ensemble_full_distribution").missing
    for source in ("top_k",):
        for kind in ("macro", "edgar"):
            assert calibrator_path("serving", source, kind, study.root / "calibrators").is_file()


def test_teardown_evidence_is_recorded_in_the_stage_receipt(study):
    result = study.steps["teardown"]
    assert result.code == EXIT_COMPLETE, result.stderr
    receipt = stage_receipt(study, "sampling", "full_run")
    assert receipt["teardown_verified"] is True and receipt["teardown"][0]["path"] == str(study.stop)


def test_every_runner_session_is_summarised_beside_the_stage_records(study):
    lines = [json.loads(line) for line in (study.root / "records" / "sampling" / "full_run" / RUNNER_LOG)
             .read_text(encoding="utf-8").splitlines()]
    assert [line["mode"] for line in lines] == ["generation", "evidence"]
    assert lines[0]["counts"]["requests_announced"] == 320 and lines[1]["counts"]["evidence_batches"] == 40
    assert lines[1]["results"] == 40 and lines[1]["status"] == "complete" and lines[1]["exit_code"] == 0


def test_the_receipt_counts_the_evidence_batches_of_its_last_session(study):
    """Fix 9.1b: the receipt carries the batches its evidence session posted (before a later session replaces it)."""
    for stage, step in ((CALIBRATION_STAGE, "calibrate-evidence"), ("pilot", "pilot-evidence"),
                        ("full_run", "full-run-evidence"), ("negative_control", "negative-evidence")):
        posted = len(study.steps[step].full[0].batches)
        assert posted > 0, stage
        last = stage_receipt(study, "sampling", stage)["last_session"]
        assert (last["kind"], last["evidence_batches"], last["error"]) == ("evidence", posted, None), stage
        assert runner_lines(study.root / "records", stage)[-1]["counts"]["evidence_batches"] == posted, stage


# ---------- the analyses and the replay, byte for byte ----------

def test_the_analyses_are_written_once_from_complete_stages(study):
    result = study.steps["analyses"]
    assert result.code == EXIT_COMPLETE, result.stderr
    written = {Path(line).name for line in result.stdout.splitlines()}
    expected = set()
    for arm in ("sampling", "serving"):
        expected |= {f"pre-registered-{arm}-full_run.json", f"pre-registered-{arm}-full_run.md",
                     f"ensemble-{arm}-full_run.json", f"ensemble-{arm}-full_run.md",
                     f"ensemble-{arm}-negative_control.json", f"ensemble-{arm}-negative_control.md",
                     f"negative-control-{arm}.json", f"negative-control-{arm}.md"}
    expected |= {"convergence-sampling-full_run.json", "convergence-sampling-full_run.md",
                 "convergence-sampling-negative_control.json", "convergence-sampling-negative_control.md"}
    assert written == expected and all(Path(line).is_relative_to(study.root) for line in result.stdout.splitlines())
    again = run(["analyses", *study.common], hosts=study.hosts)
    assert again.code == EXIT_COMPLETE and {Path(line).name for line in again.stdout.splitlines()} == expected


RUN_REPLAY = (
    "import socket, sys\n"
    "def forbidden(*args, **kwargs):\n"
    "    raise AssertionError('the replay attempted a network connection')\n"
    "socket.create_connection = forbidden\n"
    "socket.socket.connect = forbidden\n"
    "from scripts.finance_sim.run_study import main\n"
    "raise SystemExit(main(sys.argv[1:]))\n")


def run_replay(study, out_root: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", RUN_REPLAY, "replay", *study.common, "--out-root", str(out_root),
                           "--require-guard", *extra], capture_output=True, text=True, cwd=ROOT)


def file_hashes(root: Path) -> dict[str, str]:
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*")) if path.is_file()}


@pytest.fixture(scope="module")
def replayed(study) -> SimpleNamespace:
    out_root = study.base / "replay-a"
    result = run_replay(study, out_root)
    assert result.returncode == 0, result.stderr
    manifest = json.loads((out_root / MANIFEST_NAME).read_text(encoding="utf-8"))
    return SimpleNamespace(root=out_root, manifest=manifest, hashes=file_hashes(out_root), stdout=result.stdout)


def test_the_replay_under_require_guard_reproduces_the_analyses_byte_for_byte(study, replayed):
    guard = replayed.manifest["guard"]
    assert guard["required"] is True and guard["items_probed"] and guard["closes_probed"] and guard["dtb3_probed"]
    assert guard["fmp_bodies"] == 3 and guard["leaks"] == 0
    private = replayed.root / PRIVATE_DIR
    analyses = sorted(path for path in (study.root / "analysis").iterdir() if path.is_file())
    assert len(analyses) == 20
    for path in analyses:
        assert (private / path.name).read_bytes() == path.read_bytes(), path.name
    published = replayed.root / PUBLISHED_DIR
    # One SVG per published figure, named by the figure modules' own set: no themed variant beside it.
    assert sorted(path.name for path in published.iterdir() if path.suffix == ".svg") == sorted(
        f"{name}.svg" for name in PUBLISHED_FIGURES)
    assert all(Path(line).is_relative_to(study.base) for line in replayed.stdout.splitlines())


def test_a_second_fresh_process_reproduces_every_replayed_file_and_the_comparison_passes(study, replayed):
    out_root = study.base / "replay-b"
    result = run_replay(study, out_root, "--compare", str(replayed.root / PUBLISHED_DIR))
    assert result.returncode == 0, result.stderr
    assert file_hashes(out_root) == replayed.hashes


# ---------- resume, frozen limits and refusals ----------

def scenario_arguments(study, tmp_path: Path) -> list[str]:
    return [*study.common, "--records-dir", str(tmp_path / "records"), "--results-dir", str(tmp_path / "results"),
            "--gates-dir", str(tmp_path / "gates")]


def test_an_interrupted_generation_pass_resumes_without_repeating_a_request(study, tmp_path):
    arguments = scenario_arguments(study, tmp_path)
    crashed = FakeHosts(study.pin, crash_at=5)
    with pytest.raises(ProcessDied):
        run(["pilot", "--pass", "generation", *arguments], hosts=crashed)
    first = crashed.generation_clients[0]
    assert len(first.calls) == 6
    receipt = json.loads(receipt_path("sampling", "pilot", tmp_path / "records").read_text(encoding="utf-8"))
    assert (receipt["status"], receipt["exit_code"]) == ("open", EXIT_UNFINISHED)
    resumed = FakeHosts(study.pin)
    result = run(["pilot", "--pass", "generation", *arguments], hosts=resumed)
    assert result.code == EXIT_COMPLETE, result.stderr
    second = resumed.generation_clients[0]
    assert len(first.calls) + len(second.calls) == 192
    sent = [(prompt, parameters.get("seed")) for prompt, parameters in first.calls + second.calls]
    assert len(set(sent)) == 192
    assert outcome_of(result)["counts"]["interrupted_closed"] == 1
    lost_prompt = first.calls[5][0]
    item = next(item for item in macro_items(study.plan, load_items(study.root / "plan" / "items.json").macro,
                                             stage="pilot", arm="sampling") if item.prompt == lost_prompt)
    lines = read_records("sampling", "pilot", item.item_id, tmp_path / "records")
    closing = [line for line in lines if line.get("closed_by") == "resume"]
    assert len(closing) == 1 and closing[0]["failure_code"] == "request_error:interrupted"
    assert closing[0]["rendering"] == item.rendering
    settled = run(["pilot", "--pass", "evidence", *arguments], hosts=FakeHosts(study.pin))
    assert settled.code == EXIT_COMPLETE, settled.stderr
    stored = load_result(result_path("sampling", "pilot", item.item_id, item.rendering, tmp_path / "results"))
    assert [draw.failure_code for draw in stored.recorded_draws].count("request_error:interrupted") == 1
    assert json.loads(budget_path("sampling", "pilot", tmp_path / "records").read_text("utf-8"))["counters"][
        "requests"] == 192


def test_a_host_that_dies_mid_pass_ends_the_session_open_after_one_failed_draw_and_a_resume_completes_it(
        study, tmp_path):
    arguments = scenario_arguments(study, tmp_path)
    dying = FakeHosts(study.pin, dies_at=7)
    stopped = run(["pilot", "--pass", "generation", *arguments], hosts=dying)
    assert stopped.code == EXIT_UNFINISHED and "after a failed request" in stopped.stderr, stopped.stderr
    first = dying.generation_clients[0]
    assert len(first.calls) == 8
    receipt = json.loads(receipt_path("sampling", "pilot", tmp_path / "records").read_text(encoding="utf-8"))
    assert (receipt["status"], receipt["stop_reason"], receipt["exit_code"]) == ("open", "host_unavailable",
                                                                                 EXIT_UNFINISHED)
    assert [check["status"] for check in receipt["host_checks"]] == ["ok", "unreachable"]
    ledger = json.loads(budget_path("sampling", "pilot", tmp_path / "records").read_text("utf-8"))
    assert (ledger["status"], ledger["counters"]["requests"]) == ("open", 8)
    log = DrawLog(draw_log_path("sampling", "pilot", tmp_path / "records"), plan=study.plan,
                  loads=ExecutionDraw.from_dict)
    failed = [entry.draw.failure_code for entry in log.latest() if entry.draw.failure_code is not None]
    assert len(log.latest()) == 8 and len(failed) == 1
    assert failed[0].startswith("request_error:") and failed[0] != "request_error:interrupted"
    resumed = FakeHosts(study.pin)
    result = run(["pilot", "--pass", "generation", *arguments], hosts=resumed)
    assert result.code == EXIT_COMPLETE, result.stderr
    second = resumed.generation_clients[0]
    assert len(first.calls) + len(second.calls) == 192
    sent = [(prompt, parameters.get("seed")) for prompt, parameters in first.calls + second.calls]
    assert len(set(sent)) == 192
    assert outcome_of(result)["counts"]["interrupted_closed"] == 0
    assert json.loads(budget_path("sampling", "pilot", tmp_path / "records").read_text("utf-8"))["counters"][
        "requests"] == 192


def test_a_serve_deadline_closes_admission_and_partial_results_are_derived_from_fully_recorded_items(
        study, tmp_path, monkeypatch):
    plan, approval = study.plan, study.approval
    limits = approval.limits_for("full_run")
    wall = Wall(1_800_000_000.0)
    receipt = write_serve_receipt(tmp_path / "serve-full.json", plan, approval, "full_run", started=wall.now)
    monkeypatch.setattr(run_study, "wall_clock", wall)
    step = (limits.max_seconds - serving_config().timeout_s) / 40 + 1e-6
    hosts = FakeHosts(study.pin, wall=wall, step_s=step)
    arguments = [*scenario_arguments(study, tmp_path), "--manifest", str(study.manifest),
                 "--serve-receipt", str(receipt)]
    stopped = run(["full-run", "--arm", "serving", "--pass", "both", *arguments], hosts=hosts)
    assert stopped.code == EXIT_INCOMPLETE, stopped.stderr
    assert len(hosts.generation_clients[0].calls) == 40
    ledger = json.loads(budget_path("serving", "full_run", tmp_path / "records").read_text("utf-8"))
    assert (ledger["status"], ledger["stop_reason"], ledger["counters"]["requests"]) == ("incomplete", "max_seconds",
                                                                                        40)
    stage = json.loads(receipt_path("serving", "full_run", tmp_path / "records").read_text("utf-8"))
    assert (stage["status"], stage["stop_reason"], stage["exit_code"]) == ("incomplete", "max_seconds",
                                                                           EXIT_INCOMPLETE)
    assert not (tmp_path / "results").exists()
    derived = run(["full-run", "--arm", "serving", "--pass", "both", *arguments], hosts=FakeHosts(study.pin))
    assert derived.code == EXIT_INCOMPLETE, derived.stderr
    outcome = outcome_of(derived)
    assert outcome["partial"] is True and outcome["results"] == 1 and outcome["counts"]["requests_announced"] == 0
    first_key = planned_keys(plan, "serving", "full_run")[0]
    assert result_path("serving", "full_run", *first_key, tmp_path / "results").is_file()
    assert len(list((tmp_path / "results" / "serving" / "full_run").iterdir())) == 1


def test_the_gate_policy_block_maps_a_stop_to_its_exit_code_and_inform_proceeds(study):
    plan = study.plan
    gate = load_gate(gate_path(plan.plan_sha256, study.root / "gates"))

    def stopped(*args, **kwargs):
        raise GateStoppedError(gate)

    inform = run_study.check_gate(plan, "sampling", results_dir=study.root / "results", gates_dir=study.root / "gates",
                                  require=stopped)
    assert inform.proceed and inform.exit_code == EXIT_COMPLETE and "proceeds" in inform.message
    blocking = replace(plan, pilot=replace(plan.pilot, gate_policy="block"))
    block = run_study.check_gate(blocking, "sampling", results_dir=study.root / "results",
                                 gates_dir=study.root / "gates", require=stopped)
    assert not block.proceed and block.exit_code == EXIT_GATE_STOPPED and "waits for a decision" in block.message
    serving = run_study.check_gate(plan, "serving", results_dir=study.root / "results",
                                   gates_dir=study.root / "gates")
    assert serving.proceed and serving.gate is None and "achieved power" in serving.message


def test_a_wrong_pass_or_client_is_refused_before_any_request(study, tmp_path):
    arguments = scenario_arguments(study, tmp_path)
    hosts = FakeHosts(study.pin)
    serving = run(["calibrate", "--arm", "serving", "--pass", "evidence", "--manifest", str(study.manifest),
                   "--serve-receipt", str(study.receipts["calibration"]), *arguments], hosts=hosts)
    assert serving.code == EXIT_REFUSED and "no evidence pass" in serving.stderr
    slow = FakeHosts(study.pin, timeout_s=30)
    timeout = run(["negative-control", "--arm", "sampling", "--pass", "generation", *arguments], hosts=slow)
    assert timeout.code == EXIT_REFUSED and "request timeout is 300 s" in timeout.stderr
    assert slow.generation_clients == [] and hosts.generation_clients == []
    assert not (tmp_path / "records").exists()
    shutil.copytree(study.root / "records" / "sampling" / "full_run", tmp_path / "records" / "sampling" / "full_run")
    complete = run(["full-run", "--arm", "sampling", "--pass", "generation", *study.common,
                    "--records-dir", str(tmp_path / "records")], hosts=hosts)
    assert complete.code == EXIT_COMPLETE and "nothing to generate" in outcome_of(complete)["note"]
    assert hosts.generation_clients[0].calls == []


def test_pass_both_is_refused_for_the_sampling_arm_on_live_hosts(study, tmp_path):
    refused = run(["pilot", *scenario_arguments(study, tmp_path)], environ={})
    assert refused.code == EXIT_REFUSED and "never loaded together" in refused.stderr, refused.stderr
    assert not (tmp_path / "records").exists()


# ---------- a pass that raises records refused or failed, never complete (fix 9.1b) ----------

def test_the_live_refusal_an_evidence_pass_whose_every_draw_failed_to_parse_is_recorded_as_refused(study, tmp_path):
    """The archived calibration attempt 1: every draw unparsed, the ledger complete before the fit, the fit refused."""
    records, calibrators = tmp_path / "records", tmp_path / "calibrators"
    arguments = [*study.common, "--records-dir", str(records), "--calibrators-dir", str(calibrators)]
    prose = FakeHosts(study.pin, prose=True)
    generated = run(["calibrate", "--arm", "sampling", "--pass", "generation", *arguments], hosts=prose)
    assert generated.code == EXIT_COMPLETE, generated.stderr
    assert len(prose.generation_clients[0].calls) == 944 and "every planned draw" in outcome_of(generated)["note"]
    refused = run(["calibrate", "--arm", "sampling", "--pass", "evidence", *arguments], hosts=FakeHosts(study.pin))
    assert refused.code == EXIT_REFUSED and "CalibrationError" in refused.stderr, refused.stderr
    assert json.loads(budget_path("sampling", CALIBRATION_STAGE, records).read_text("utf-8"))["status"] == "complete"
    receipt = read_receipt(records, CALIBRATION_STAGE)
    assert (receipt["status"], receipt["stop_reason"], receipt["exit_code"]) == ("refused", "CalibrationError",
                                                                                 EXIT_REFUSED)
    last = receipt["last_session"]
    assert (last["kind"], last["session"], last["error"], last["evidence_batches"]) == ("evidence", 2,
                                                                                        "CalibrationError", 0)
    assert [check["host"] for check in receipt["host_checks"]] == ["llama_server", "full_distribution"]
    assert receipt["sessions"] == 2 and receipt["teardown_verified"] is None and receipt["teardown"] == []
    lines = runner_lines(records, CALIBRATION_STAGE)
    assert [(line["mode"], line["status"], line["exit_code"], line["error"]) for line in lines] == [
        ("generation", "open", EXIT_COMPLETE, None), ("evidence", "refused", EXIT_REFUSED, "CalibrationError")]
    assert "benign" not in receipt_path("sampling", CALIBRATION_STAGE, records).read_text(encoding="utf-8")
    assert not calibrators.exists()


#: The study entry point each stage command and pass hands the recorder to; the tests below make it raise.
PASS_ENTRY_POINTS = {
    ("calibrate", "generation"): "collect_draws", ("calibrate", "evidence"): "run_calibration",
    ("pilot", "generation"): "generate_draws", ("pilot", "evidence"): "run_pilot",
    ("full-run", "generation"): "generate_draws", ("full-run", "evidence"): "execute_items",
    ("negative-control", "generation"): "generate_draws", ("negative-control", "evidence"): "run_negative_control",
}


def raising_pass(study, tmp_path: Path, command: str, mode: str) -> tuple[list[str], Path, str]:
    """The stage command on a records folder of its own; an evidence pass starts from the study's complete records."""
    stage = run_study.STAGE_OF[command]
    records = tmp_path / "records"
    if mode == "evidence":
        shutil.copytree(study.root / "records" / "sampling" / stage, records / "sampling" / stage)
    arm = [] if command == "pilot" else ["--arm", "sampling"]
    return [command, *arm, "--pass", mode, *study.common, "--records-dir", str(records)], records, stage


@pytest.mark.parametrize(("command", "mode"), sorted(PASS_ENTRY_POINTS))
def test_a_refused_pass_records_refused_with_exit_2_never_complete(study, tmp_path, monkeypatch, command, mode):
    argv, records, stage = raising_pass(study, tmp_path, command, mode)
    before = read_receipt(records, stage) if receipt_path("sampling", stage, records).exists() else None

    def refuse(*args, **kwargs):
        raise CalibrationError("no usable prompts: every draw failed to parse")

    monkeypatch.setattr(run_study, PASS_ENTRY_POINTS[(command, mode)], refuse)
    result = run(argv, hosts=FakeHosts(study.pin))
    assert result.code == EXIT_REFUSED and "CalibrationError" in result.stderr, result.stderr
    receipt = read_receipt(records, stage)
    assert (receipt["status"], receipt["stop_reason"], receipt["exit_code"]) == ("refused", "CalibrationError",
                                                                                 EXIT_REFUSED)
    assert (receipt["last_session"]["kind"], receipt["last_session"]["error"]) == (mode, "CalibrationError")
    assert receipt["host_checks"] and receipt["sessions"] == (0 if before is None else before["sessions"]) + 1
    assert receipt["teardown"] == ([] if before is None else before["teardown"])
    line = runner_lines(records, stage)[-1]
    assert (line["mode"], line["status"], line["exit_code"], line["error"]) == (mode, "refused", EXIT_REFUSED,
                                                                                "CalibrationError")
    assert "failed to parse" not in receipt_path("sampling", stage, records).read_text(encoding="utf-8")


@pytest.mark.parametrize(("command", "mode"), sorted(PASS_ENTRY_POINTS))
def test_a_pass_another_exception_ends_records_failed_with_exit_1(study, tmp_path, monkeypatch, command, mode):
    argv, records, stage = raising_pass(study, tmp_path, command, mode)

    def fail(*args, **kwargs):
        raise RuntimeError("the host answered with an unknown shape")

    monkeypatch.setattr(run_study, PASS_ENTRY_POINTS[(command, mode)], fail)
    with pytest.raises(RuntimeError, match="unknown shape"):
        run(argv, hosts=FakeHosts(study.pin))
    receipt = read_receipt(records, stage)
    assert (receipt["status"], receipt["stop_reason"], receipt["exit_code"]) == ("failed", "RuntimeError",
                                                                                 EXIT_UNFINISHED)
    assert (receipt["last_session"]["kind"], receipt["last_session"]["error"]) == (mode, "RuntimeError")
    line = runner_lines(records, stage)[-1]
    assert (line["mode"], line["status"], line["exit_code"], line["error"]) == (mode, "failed", EXIT_UNFINISHED,
                                                                                "RuntimeError")
    assert "unknown shape" not in receipt_path("sampling", stage, records).read_text(encoding="utf-8")


@pytest.mark.parametrize("refusal", run_study._REFUSALS, ids=lambda cls: cls.__name__)
def test_every_refusal_class_of_the_runner_is_recorded_as_refused_with_the_exit_code_it_returns(
        study, tmp_path, monkeypatch, refusal):
    def refuse(*args, **kwargs):
        raise refusal("refused before the first draw")

    monkeypatch.setattr(run_study, "generate_draws", refuse)
    records = tmp_path / "records"
    result = run(["pilot", "--pass", "generation", *study.common, "--records-dir", str(records)],
                 hosts=FakeHosts(study.pin))
    receipt = read_receipt(records, "pilot")
    assert result.code == EXIT_REFUSED == receipt["exit_code"], result.stderr
    reason = {HostCheckError: "host_check", ApprovalError: "approval"}.get(refusal, refusal.__name__)
    assert (receipt["status"], receipt["stop_reason"], receipt["last_session"]["error"]) == ("refused", reason,
                                                                                            refusal.__name__)
    assert runner_lines(records, "pilot")[-1]["exit_code"] == EXIT_REFUSED


@pytest.mark.parametrize(("guarded", "error"), [(True, "BudgetExhaustedError"), (False, "AdmissionClosedError")],
                         ids=["guarded-plain-error", "unguarded-ledger-refusal"])
def test_a_frozen_request_limit_reached_mid_pass_is_recorded_incomplete_with_exit_3(study, tmp_path, guarded, error):
    """The live cap: LedgerCallBudget caps itself at the ledger's remaining requests and the providers admit through
    RequestControl's guarded path, which raises the plain BudgetExhaustedError at zero before the ledger is asked (the
    unguarded path gets the ledger's AdmissionClosedError); either way the receipt and the runner line say incomplete
    with the ledger's reason and the CLI exits 3."""
    plan, records = study.plan, tmp_path / "records"
    consumed, frozen = 10, plan.limits.stage("sampling", "pilot")
    limits = StudyLimits(plan.plan_sha256, "sampling", "pilot", frozen.max_requests, frozen.max_output_tokens,
                         frozen.max_seconds)
    ledger = BudgetLedger.open(budget_path("sampling", "pilot", records), plan, limits,
                               request_allowance(plan, "sampling", sampling_config()))
    with ledger:
        for _ in range(consumed):
            ledger.finish(ledger.admit(), output_tokens=1)
    hosts = FakeHosts(study.pin, guarded=guarded)
    result = run(["pilot", "--pass", "generation", *study.common, "--records-dir", str(records)], hosts=hosts)
    assert result.code == EXIT_INCOMPLETE and "incomplete" in result.stderr, result.stderr
    assert len(hosts.generation_clients[0].calls) == frozen.max_requests - consumed + 1  # the refused call is the last
    stored = json.loads(budget_path("sampling", "pilot", records).read_text(encoding="utf-8"))
    assert (stored["status"], stored["stop_reason"], stored["counters"]["requests"]) == ("incomplete", "max_requests",
                                                                                        frozen.max_requests)
    receipt = read_receipt(records, "pilot")
    assert (receipt["status"], receipt["stop_reason"], receipt["exit_code"]) == ("incomplete", "max_requests",
                                                                                 EXIT_INCOMPLETE)
    assert receipt["last_session"]["error"] == error
    line = runner_lines(records, "pilot")[-1]
    assert (line["mode"], line["status"], line["stop_reason"], line["exit_code"], line["error"]) == (
        "generation", "incomplete", "max_requests", EXIT_INCOMPLETE, error)


@pytest.mark.parametrize(("command", "stage", "total"), [("calibrate", CALIBRATION_STAGE, 2 * 24 + 140),
                                                         ("full-run", "full_run", 40)])
def test_a_missing_evidence_failure_leaves_the_stage_open_and_a_rerun_resends_only_the_missing_batches(
        study, tmp_path, command, stage, total):
    records, answered = tmp_path / "records", 5
    common = [*study.common, "--records-dir", str(records)]
    if command == "calibrate":
        common += ["--calibrators-dir", str(tmp_path / "calibrators")]
        evidence = common
    else:
        evidence = [*common, "--results-dir", str(tmp_path / "results")]
    generated = run([command, "--arm", "sampling", "--pass", "generation", *common], hosts=FakeHosts(study.pin))
    assert generated.code == EXIT_COMPLETE, generated.stderr
    dying = FakeHosts(study.pin, evidence_fail_after=answered)
    stopped = run([command, "--arm", "sampling", "--pass", "evidence", *evidence], hosts=dying)
    assert stopped.code == EXIT_UNFINISHED and "stays open" in stopped.stderr, stopped.stderr
    assert len(dying.full_clients[0].batches) == answered + 1
    receipt = read_receipt(records, stage)
    assert (receipt["status"], receipt["stop_reason"], receipt["exit_code"]) == ("open", "evidence_missing",
                                                                                 EXIT_UNFINISHED)
    assert (receipt["last_session"]["error"], receipt["last_session"]["evidence_batches"]) == (
        "EvidenceUnavailableError", answered)
    line = runner_lines(records, stage)[-1]
    assert (line["mode"], line["status"], line["stop_reason"], line["exit_code"], line["error"]) == (
        "evidence", "open", "evidence_missing", EXIT_UNFINISHED, "EvidenceUnavailableError")
    working = FakeHosts(study.pin)
    resumed = run([command, "--arm", "sampling", "--pass", "evidence", *evidence], hosts=working)
    assert resumed.code == EXIT_COMPLETE, resumed.stderr
    assert len(working.full_clients[0].batches) == total - answered
    receipt = read_receipt(records, stage)
    assert (receipt["status"], receipt["exit_code"], receipt["last_session"]["error"]) == ("complete", EXIT_COMPLETE,
                                                                                          None)
    assert receipt["last_session"]["evidence_batches"] == total - answered


def test_the_runner_help_names_the_live_commands_of_the_operator():
    text = run_study.build_parser().format_help()
    for line in ("serve_local_llamacpp.ps1 start -IdleMinutes 180", "serve_full_distribution.ps1 start -IdleMinutes 180",
                 '$env:LLAMACPP_ENDPOINT_URL = "http://127.0.0.1:11571/v1/chat/completions"',
                 '$env:FULL_DISTRIBUTION_ENDPOINT_URL = "http://127.0.0.1:11581"',
                 "serve_local_llamacpp.ps1 start -BindAddress", "serve_full_distribution.ps1 start -BindAddress",
                 "docker compose run --rm", "host.containers.internal", "modal_operator.sh serve",
                 "--pass generation", "--pass evidence", "teardown --arm sampling", "Vulkan throughput",
                 "idle stop", "/props", "container route"):
        assert line in text, line
    assert text.index("$env:LLAMACPP_ENDPOINT_URL") < text.index("docker compose run --rm")
