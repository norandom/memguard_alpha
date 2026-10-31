"""Command integration with no credentials or external API requests."""

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from recall_guard.core.provider_anthropic import AnthropicClient
from recall_guard.core.providers import OpenAICompatibleClient
from recall_guard.harness.intervention_report import load_experiment_report
from scripts.run_intervention import main
from tests.harness.test_intervention import PROBE, SOURCE, VERDICT, Factory
from tests.harness.test_intervention_config import config_files, mutate, write_json

ROOT = Path(__file__).resolve().parents[2]


def cp1252_offline_command(tmp_path, args):
    """Start a credential-free process with strict Windows console encodings."""
    script = """
import os, runpy, socket, sys
import dotenv
import recall_guard.core.providers as providers
import recall_guard.core.provider_anthropic as anthropic
import recall_guard.core.provider_callback as callback
def forbidden(*args, **kwargs):
    raise AssertionError('No network, clients, environment files or credential access allowed')
socket.socket.connect = forbidden
socket.socket.connect_ex = forbidden
socket.create_connection = forbidden
socket.getaddrinfo = forbidden
providers.OpenAICompatibleClient.__init__ = forbidden
anthropic.AnthropicClient.__init__ = forbidden
callback.CallbackClient.__init__ = forbidden
dotenv.load_dotenv = forbidden
class NoEnvironment(dict):
    def get(self, name, default=None):
        if name in {'LANGUAGE', 'LC_ALL', 'LC_MESSAGES', 'LANG', 'COLUMNS', 'LINES'}:
            return default
        return forbidden()
    def __getitem__(self, name):
        if name in {'COLUMNS', 'LINES'}:
            raise KeyError(name)
        return forbidden()
os.environ = NoEnvironment()
sys.stdout.reconfigure(encoding='cp1252', errors='strict')
sys.stderr.reconfigure(encoding='cp1252', errors='strict')
command = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(command, run_name='__main__')
"""
    # Windows needs its system directory to initialize stdlib _overlapped.
    # Forward only these loader variables; the command cannot read credentials.
    environment = {name: os.environ[name] for name in ("SYSTEMROOT", "WINDIR") if name in os.environ}
    environment["PYTHONPATH"] = str(ROOT)
    return subprocess.run([sys.executable, "-c", script, str(ROOT / "scripts/run_intervention.py"), *args],
                          cwd=tmp_path, env=environment, capture_output=True,
                          text=True, encoding="utf-8", errors="backslashreplace", timeout=45)


def test_cp1252_dry_run_preserves_unicode_metadata_without_credentials(tmp_path):
    pp, ip, _, _ = config_files(tmp_path, audit_selection={"method": "東京"})
    before = {path: path.read_bytes() for path in (pp, ip)}
    result = cp1252_offline_command(tmp_path, ["--dry-run", "--protocol", str(pp), "--items", str(ip)])
    assert result.returncode == 0, result.stderr
    assert result.stdout.isascii()
    plan = json.loads(result.stdout)
    assert plan["audit_selection"]["method"] == "東京"
    assert plan["physical_calls"]["nominal_total"] == 19
    assert {path: path.read_bytes() for path in (pp, ip)} == before
    assert set(tmp_path.iterdir()) == {pp, ip}


def test_cp1252_replay_preserves_unicode_component_names_without_credentials(tmp_path):
    pp, ip, p, _ = config_files(tmp_path, models="callback")
    factory = Factory(p, target=lambda prompt: '{"answer":"up","確信度":0.7}')
    assert main(run_args(pp, ip, tmp_path, answer_format="json", confidence_component="確信度"),
                workflow_factory=factory) == 0
    report_path = tmp_path / "report"
    before = {path: path.read_bytes() for path in report_path.iterdir()}
    result = cp1252_offline_command(tmp_path, ["--replay", str(report_path)])
    assert result.returncode == 0, result.stderr
    assert result.stdout.isascii()
    metrics = json.loads(result.stdout)
    assert metrics == load_experiment_report(report_path).metrics
    assert metrics["provenance"]["confidence"]["component"] == "確信度"
    assert {path: path.read_bytes() for path in report_path.iterdir()} == before
    assert "確信度".encode() in before[report_path / "manifest.json"]
    assert all(b"\r" not in content for content in before.values())


def test_cp1252_unicode_path_error_is_actionable_without_traceback(tmp_path):
    missing = tmp_path / "東京" / "protocol.json"
    result = cp1252_offline_command(tmp_path, ["--dry-run", "--protocol", str(missing),
                                             "--items", str(tmp_path / "items.json")])
    assert result.returncode == 2, result.stderr
    assert not result.stdout
    assert result.stderr.isascii()
    assert "Experiment configuration/execution error:" in result.stderr
    assert "protocol.json" in result.stderr
    assert "\\u6771\\u4eac" in result.stderr
    assert "Traceback" not in result.stderr and "UnicodeEncodeError" not in result.stderr
    assert not list(tmp_path.iterdir())


def test_fresh_dry_run_never_reads_environment_or_constructs_clients(tmp_path):
    pp, ip, _, _ = config_files(tmp_path)
    script = """
import os, runpy, socket, sys
def forbidden(*args, **kwargs):
    raise AssertionError('No network or credential environment access allowed')
socket.socket.connect = forbidden
socket.socket.connect_ex = forbidden
socket.create_connection = forbidden
socket.getaddrinfo = forbidden
import recall_guard.core.providers as providers
import recall_guard.core.provider_anthropic as anthropic
providers.OpenAICompatibleClient.__init__ = forbidden
anthropic.AnthropicClient.__init__ = forbidden
class NoEnvironment(dict):
    def get(self, name, default=None):
        # argparse/gettext consult locale/terminal variables, never credentials.
        if name in {'LANGUAGE', 'LC_ALL', 'LC_MESSAGES', 'LANG', 'COLUMNS', 'LINES'}:
            return default
        return forbidden()
    def __getitem__(self, name):
        if name in {'COLUMNS', 'LINES'}:
            raise KeyError(name)
        return forbidden()
os.environ = NoEnvironment()
command_path = sys.argv[3]
sys.argv = ['run_intervention.py', '--protocol', sys.argv[1], '--items', sys.argv[2], '--dry-run']
runpy.run_path(command_path, run_name='__main__')
"""
    env = {name: os.environ[name] for name in ("SYSTEMROOT", "WINDIR", "PATH", "TEMP", "TMP") if name in os.environ}
    env.update(PYTHONPATH=str(ROOT), PYTHONIOENCODING="utf-8")
    result = subprocess.run([sys.executable, "-c", script, str(pp), str(ip), str(ROOT / "scripts/run_intervention.py")],
                            cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8", timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    plan = json.loads(result.stdout)
    assert plan["physical_calls"]["nominal_total"] == 19
    assert plan["ready_for_batch"] is False
    assert not list(tmp_path.glob("*checkpoint*"))


def test_help_is_standalone():
    result = subprocess.run([sys.executable, str(ROOT / "scripts/run_intervention.py"), "--help"],
                            capture_output=True, text=True, encoding="utf-8", timeout=30)
    assert result.returncode == 0, result.stderr
    assert "--dry-run" in result.stdout and "--target-credential-env" in result.stdout


def test_dry_run_rejects_literal_credential_values_without_reading_environment(tmp_path, capsys):
    pp, ip, _, _ = config_files(tmp_path)
    assert main(["--dry-run", "--protocol", str(pp), "--items", str(ip),
                 "--target-credential-env", "Bearer synthetic-value"]) == 2
    assert "target: credential_ref must name an environment variable" in capsys.readouterr().err


class ProviderFixture:
    """Exercise native probes, physical admission and response normalization."""

    def __init__(self, monkeypatch, *, fail_role=None, candidate=None, target=None, retry_once=False):
        self.calls = []
        self.fail_role, self.candidate, self.target = fail_role, candidate, target
        self.retry_once = retry_once
        monkeypatch.setattr(OpenAICompatibleClient, "_post", self.post)
        monkeypatch.setattr(AnthropicClient, "_post", self.post)

    def post(self, payload):
        import requests

        role, prompt = payload["model"], payload["messages"][0]["content"]
        self.calls.append((role, prompt))
        response = requests.Response()
        response.status_code = 401 if self.fail_role == role else 503 if self.retry_once and len(self.calls) == 1 else 200
        if response.status_code != 200:
            body = {"error": {"message": "Fixture authentication rejected", "code": "invalid_api_key"}}
        else:
            content = "OK" if prompt == PROBE else (
                self.target or "PRIVATE_LABEL" if role == "target" else
                self.candidate or "Let's review: " + SOURCE if role == "rewriter" else VERDICT)
            body = {"choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                    "content": [{"type": "text", "text": content}], "stop_reason": "end_turn",
                    "usage": {"input_tokens": 2, "output_tokens": 1}}
        response._content = json.dumps(body).encode()
        return response


def run_args(pp, ip, tmp_path, *, output="report", **options):
    args = ["--run", "--protocol", str(pp), "--items", str(ip), "--checkpoint", str(tmp_path / "journal.json"),
            "--output", str(tmp_path / output)]
    for name, value in options.items():
        args += ["--" + name.replace("_", "-"), str(value)]
    return args


def credential_args(monkeypatch):
    args = []
    for role in ("target", "rewriter", "verifier"):
        name = f"FIXTURE_{role.upper()}_KEY"
        monkeypatch.setenv(name, "SYNTHETIC_TEST_VALUE")
        args += [f"--{role}-credential-env", name]
    return args


def test_full_native_four_condition_run_probes_first_and_replays_offline(tmp_path, monkeypatch, capsys):
    pp, ip, p, _ = config_files(tmp_path)
    p = replace(p, rewriter_model=replace(p.rewriter_model, protocol="anthropic_messages"))
    write_json(pp, p.to_dict())
    fixture = ProviderFixture(monkeypatch)
    args = run_args(pp, ip, tmp_path, confidence_level=.8) + credential_args(monkeypatch)
    assert main(args) == 0
    assert fixture.calls[:3] == [(role, PROBE) for role in ("target", "rewriter", "verifier")]
    assert len(fixture.calls) == 19
    report = load_experiment_report(tmp_path / "report")
    assert len(report.run.conditions) == 8
    assert len(report.run.checkpoint.draws) == 12
    assert report.manifest["analysis"]["confidence_level"] == .8
    assert report.manifest["validation"]["execution"] == "unverified"
    assert report.recompute_metrics() == report.metrics
    records = (tmp_path / "report/records.json").read_text(encoding="utf-8")
    assert "SYNTHETIC_TEST_VALUE" not in records and "FIXTURE_TARGET_KEY" not in records
    capsys.readouterr()
    assert main(["--replay", str(tmp_path / "report")]) == 0
    assert json.loads(capsys.readouterr().out) == report.metrics
    assert len(fixture.calls) == 19


def test_preflight_failure_exports_and_schedules_no_batch(tmp_path, monkeypatch):
    pp, ip, _, _ = config_files(tmp_path)
    fixture = ProviderFixture(monkeypatch, fail_role="rewriter")
    assert main(run_args(pp, ip, tmp_path) + credential_args(monkeypatch)) == 1
    assert fixture.calls == [("target", PROBE), ("rewriter", PROBE)]
    report = load_experiment_report(tmp_path / "report")
    assert report.run.stop_reason == "authentication_error"
    assert len(report.run.checkpoint.requests) == 2 and not report.run.arms


def test_budget_failure_exports_partial_and_resume_does_not_reset_cap(tmp_path, monkeypatch):
    pp, ip, _, _ = config_files(tmp_path, request_budget=10)
    fixture = ProviderFixture(monkeypatch)
    credentials = credential_args(monkeypatch)
    assert main(run_args(pp, ip, tmp_path) + credentials) == 1
    report = load_experiment_report(tmp_path / "report")
    assert report.run.stop_reason == "budget_exhausted"
    assert len(fixture.calls) == len(report.run.checkpoint.requests) == 10
    assert main(run_args(pp, ip, tmp_path, output="resume") + credentials) == 1
    assert len(fixture.calls) == 10
    assert len(load_experiment_report(tmp_path / "resume").run.checkpoint.requests) == 10


def test_native_probe_retry_consumes_physical_shared_budget(tmp_path, monkeypatch):
    pp, ip, p, _ = config_files(tmp_path)
    p = replace(p, target_model=replace(p.target_model, max_retries=1, retry_backoff_s=0))
    write_json(pp, p.to_dict())
    fixture = ProviderFixture(monkeypatch, retry_once=True)
    assert main(run_args(pp, ip, tmp_path) + credential_args(monkeypatch)) == 0
    requests = load_experiment_report(tmp_path / "report").run.checkpoint.requests
    assert len(fixture.calls) == len(requests) == 20
    assert [request.event.attempt for request in requests[:2]] == [1, 2]
    assert [request.event.status for request in requests[:2]] == ["failed", "succeeded"]


def test_safe_resume_reprobes_and_only_calls_pending_arms(tmp_path, monkeypatch, capsys):
    pp, ip, _, _ = config_files(tmp_path)
    fixture = ProviderFixture(monkeypatch)
    credentials = credential_args(monkeypatch)
    assert main(run_args(pp, ip, tmp_path, max_arms=1) + credentials) == 1
    assert len(fixture.calls) == 6
    capsys.readouterr()
    assert main(["--dry-run", "--protocol", str(pp), "--items", str(ip),
                 "--checkpoint", str(tmp_path / "journal.json")]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["physical_calls"]["hard_remaining_cap"] == 94
    assert plan["physical_calls"]["nominal_total"] == 16
    assert len(fixture.calls) == 6
    assert main(run_args(pp, ip, tmp_path, output="resume") + credentials) == 0
    assert fixture.calls[6:9] == [(role, PROBE) for role in ("target", "rewriter", "verifier")]
    assert len(fixture.calls) == 22
    assert len(load_experiment_report(tmp_path / "resume").run.checkpoint.requests) == 22
    # Refuse overwrite before any new probe or physical call.
    assert main(run_args(pp, ip, tmp_path, output="resume") + credentials) == 2
    assert len(fixture.calls) == 22


def test_strict_callback_injection_uses_same_cli_path_without_dynamic_imports(tmp_path):
    pp, ip, p, _ = config_files(tmp_path, models="callback")
    factory = Factory(p)
    assert main(run_args(pp, ip, tmp_path), workflow_factory=factory) == 0
    assert len(factory.calls) == 19
    assert len(load_experiment_report(tmp_path / "report").run.conditions) == 8


def test_json_ensemble_mean_reduces_explicit_confidence_components(tmp_path):
    pp, ip, p, _ = config_files(tmp_path, count=1, models="callback")
    mutate(pp, lambda d: d["ensemble_settings"].update(location_mode="mean"))
    responses = iter([0.0, 0.0, 0.9] * 2)
    factory = Factory(p, target=lambda prompt: json.dumps({"answer": "up", "confidence": next(responses)}))
    assert main(run_args(pp, ip, tmp_path, answer_format="json"), workflow_factory=factory) == 0
    report = load_experiment_report(tmp_path / "report")
    for arm in report.run.arms:
        assert arm.outcome["status"] == "completed"
        assert arm.outcome["ensemble"]["location"] == {"confidence": pytest.approx(0.3)}
        assert arm.outcome["ensemble"]["n_parsed"] == 3
    assert len(factory.calls) == 11


def test_json_two_sided_tail_reports_actual_agreement_interval(tmp_path):
    pp, ip, p, _ = config_files(tmp_path, count=1, models="callback")
    mutate(pp, lambda d: d["ensemble_settings"].update(tail="two_sided"))
    factory = Factory(p, target=lambda prompt: '{"answer":"up","confidence":0.7}')
    assert main(run_args(pp, ip, tmp_path, answer_format="json"), workflow_factory=factory) == 0
    report = load_experiment_report(tmp_path / "report")
    for arm in report.run.arms:
        assert arm.outcome["status"] == "completed"
        assert arm.outcome["ensemble"]["agreement_ci"] == pytest.approx([0.4385029682449546, 1.0])
    assert len(factory.calls) == 11


@pytest.mark.parametrize("action", ["flag", "raise"])
def test_json_multimodal_policy_flags_or_fails_the_actual_reduction(tmp_path, action):
    pp, ip, p, _ = config_files(tmp_path, count=1, models="callback", draws=8)
    mutate(pp, lambda d: d["ensemble_settings"].update(grid=0.1, mass_min=0.4, multimodal_action=action))
    responses = iter([0.1] * 4 + [0.9] * 4 + [0.1] * 4 + [0.9] * 4)
    factory = Factory(p, target=lambda prompt: json.dumps({"answer": "up", "confidence": next(responses)}))
    assert main(run_args(pp, ip, tmp_path, answer_format="json"), workflow_factory=factory) == 0
    report = load_experiment_report(tmp_path / "report")
    for arm in report.run.arms:
        if action == "raise":
            assert arm.outcome["status"] == "failed" and arm.outcome["ensemble"] is None
            assert arm.outcome["failure_kind"] == "runtime_error"
        else:
            assert arm.outcome["status"] == "completed"
            assert arm.outcome["ensemble"]["multimodal"] == ["confidence"]
            assert arm.outcome["ensemble"]["location"] == {}
    assert all(draw.status == "completed" for draw in report.run.checkpoint.draws)
    assert len(factory.calls) == 21


@pytest.mark.parametrize("change,match", [
    (lambda d: d.update(schema_version=7), "schema_version"),
    (lambda d: d["target_model"]["parameters"].update(max_tokens=False), "max_tokens"),
    (lambda d: d["data_hashes"].update(evaluation="f" * 64), "hash"),
    (lambda d: d["data_hashes"].update(original_calibration="f" * 64, rewritten_calibration="e" * 64), "fitted scorers"),
    (lambda d: d["ensemble_settings"].update(location_mode="invented"), "location_mode"),
    (lambda d: d["ensemble_settings"].update(confidence=None), "confidence"),
    (lambda d: d["ensemble_settings"].update(unknown_option=1), "Unknown ensemble"),
])
def test_invalid_configuration_and_missing_fits_make_zero_calls(tmp_path, monkeypatch, capsys, change, match):
    pp, ip, _, _ = config_files(tmp_path)
    mutate(pp, change)
    fixture = ProviderFixture(monkeypatch)
    assert main(run_args(pp, ip, tmp_path) + credential_args(monkeypatch)) == 2
    assert match in capsys.readouterr().err
    assert fixture.calls == [] and not (tmp_path / "journal.json").exists()


def test_missing_credentials_and_native_callback_message_precede_batch(tmp_path, monkeypatch, capsys):
    pp, ip, _, _ = config_files(tmp_path)
    fixture = ProviderFixture(monkeypatch)
    monkeypatch.delenv("MISSING_FIXTURE_KEY", raising=False)
    assert main(run_args(pp, ip, tmp_path, target_credential_env="MISSING_FIXTURE_KEY")) == 2
    assert "target: referenced credential is missing" in capsys.readouterr().err
    assert fixture.calls == []
    pp, ip, _, _ = config_files(tmp_path, models="callback")
    assert main(run_args(pp, ip, tmp_path)) == 2
    assert "physical-admission hook" in capsys.readouterr().err
    assert fixture.calls == []


@pytest.mark.parametrize("confidence,disable", [(0.7, False), (None, False), ("not-a-number", True)])
def test_explicit_json_projection_and_missing_confidence_stays_unavailable(tmp_path, monkeypatch, confidence, disable):
    pp, ip, _, _ = config_files(tmp_path)
    body = {"direction": "PRIVATE_LABEL"}
    if confidence is not None:
        body["confidence"] = confidence
    fixture = ProviderFixture(monkeypatch, target=json.dumps(body))
    args = run_args(pp, ip, tmp_path, answer_format="json", answer_field="direction") + credential_args(monkeypatch)
    if disable:
        args.append("--no-confidence-component")
    assert main(args) == 0
    report = load_experiment_report(tmp_path / "report")
    first = report.run.checkpoint.draws[0]
    assert first.metadata["answer"] == "PRIVATE_LABEL" and first.metadata["guarded_score"] is None
    assert dict(first.metadata["components"] or {}) == ({"confidence": confidence} if confidence is not None and not disable else {})
    assert report.manifest["analysis"]["confidence_component"] == (None if disable else "confidence")
    assert len(fixture.calls) == 19


def test_deterministic_fidelity_rejection_reduces_verifier_and_target_calls(tmp_path, monkeypatch):
    pp, ip, _, _ = config_files(tmp_path)
    fixture = ProviderFixture(monkeypatch, candidate="Acme rose 9% on 2024-01-02; uncertainty remains.")
    assert main(run_args(pp, ip, tmp_path) + credential_args(monkeypatch)) == 0
    report = load_experiment_report(tmp_path / "report")
    assert len(fixture.calls) == 11
    assert all(record.status == "failed" for record in report.run.checkpoint.rewrites)
    assert all(call[1] == PROBE for call in fixture.calls if call[0] == "verifier")


def test_blank_text_is_failed_projection_and_explicit_text_confidence_is_rejected(tmp_path, monkeypatch):
    pp, ip, _, _ = config_files(tmp_path)
    fixture = ProviderFixture(monkeypatch, target="   ")
    credentials = credential_args(monkeypatch)
    assert main(run_args(pp, ip, tmp_path, confidence_component="confidence") + credentials) == 2
    assert not fixture.calls
    assert main(run_args(pp, ip, tmp_path) + credentials) == 0
    report = load_experiment_report(tmp_path / "report")
    assert all(draw.status == "failed" for draw in report.run.checkpoint.draws)


def test_ambiguous_journal_resume_and_overlapping_output_fail_without_requests(tmp_path, monkeypatch):
    pp, ip, _, _ = config_files(tmp_path)
    fixture = ProviderFixture(monkeypatch)
    credentials = credential_args(monkeypatch)
    assert main(run_args(pp, ip, tmp_path, max_arms=1) + credentials) == 1
    mutate(tmp_path / "journal.json", lambda d: d.update(intent={"kind": "arm"}))
    assert main(run_args(pp, ip, tmp_path, output="resume") + credentials) == 2
    assert len(fixture.calls) == 6


@pytest.mark.parametrize("change", ["projection", "missing", "corrupt", "partial", "version", "duplicate"])
def test_projection_sidecar_prevents_mixed_semantics_and_unowned_resume(tmp_path, monkeypatch, capsys, change):
    pp, ip, _, _ = config_files(tmp_path)
    fixture = ProviderFixture(monkeypatch)
    credentials = credential_args(monkeypatch)
    assert main(run_args(pp, ip, tmp_path, max_arms=1) + credentials) == 1
    sidecar = tmp_path / "journal.json.cli.json"
    assert sidecar.exists()
    if change == "missing":
        sidecar.unlink()
    elif change == "corrupt":
        sidecar.write_text("invalid", encoding="utf-8")
    elif change == "partial":
        sidecar.write_text('{"schema_version":', encoding="utf-8")
    elif change == "version":
        mutate(sidecar, lambda d: d.update(schema_version=2))
    elif change == "duplicate":
        sidecar.write_text('{"schema_version":2,' + sidecar.read_text(encoding="utf-8")[1:], encoding="utf-8")
    args = run_args(pp, ip, tmp_path, output="resume") + credentials
    if change == "projection":
        args += ["--answer-format", "json", "--answer-field", "direction"]
    assert main(args) == 2
    assert len(fixture.calls) == 6
    assert "sidecar" in capsys.readouterr().err


def test_dry_run_reads_but_never_creates_projection_sidecar(tmp_path, monkeypatch):
    pp, ip, _, _ = config_files(tmp_path)
    fixture = ProviderFixture(monkeypatch)
    args = ["--dry-run", "--protocol", str(pp), "--items", str(ip), "--checkpoint", str(tmp_path / "journal.json")]
    assert main(args) == 0
    assert not (tmp_path / "journal.json.cli.json").exists()
    credentials = credential_args(monkeypatch)
    assert main(run_args(pp, ip, tmp_path, max_arms=1) + credentials) == 1
    sidecar = tmp_path / "journal.json.cli.json"
    before = sidecar.read_bytes()
    assert main(args) == 0
    assert sidecar.read_bytes() == before and len(fixture.calls) == 6
    assert main(args + ["--answer-format", "json"]) == 2
    assert sidecar.read_bytes() == before and len(fixture.calls) == 6
    assert main(["--run", "--protocol", str(pp), "--items", str(ip), "--output", str(tmp_path / "nested"),
                 "--checkpoint", str(tmp_path / "nested/journal.json")] + credentials) == 2
    assert len(fixture.calls) == 6


def test_sidecar_creation_race_never_overwrites_conflicting_state(tmp_path, monkeypatch):
    pp, ip, _, _ = config_files(tmp_path)
    fixture = ProviderFixture(monkeypatch)
    credentials = credential_args(monkeypatch)
    original_open = Path.open
    sidecar = tmp_path / "journal.json.cli.json"

    def race(path, mode="r", *args, **kwargs):
        if path == sidecar and mode == "x":
            path.write_text("partial competing state", encoding="utf-8")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", race)
    assert main(run_args(pp, ip, tmp_path) + credentials) == 2
    assert sidecar.read_text(encoding="utf-8") == "partial competing state"
    assert fixture.calls == [] and not (tmp_path / "journal.json").exists()


def test_replay_in_fresh_process_with_network_and_credentials_forbidden(tmp_path):
    pp, ip, p, _ = config_files(tmp_path, models="callback")
    assert main(run_args(pp, ip, tmp_path), workflow_factory=Factory(p)) == 0
    script = """
import runpy, socket, sys
def forbidden(*args, **kwargs):
    raise AssertionError('No network allowed')
socket.socket.connect = forbidden
socket.create_connection = forbidden
socket.getaddrinfo = forbidden
command, report = sys.argv[1:]
sys.argv = [command, '--replay', report]
runpy.run_path(command, run_name='__main__')
"""
    env = {name: os.environ[name] for name in ("SYSTEMROOT", "WINDIR", "PATH", "TEMP", "TMP") if name in os.environ}
    env.update(PYTHONPATH=str(ROOT), PYTHONIOENCODING="utf-8")
    result = subprocess.run([sys.executable, "-c", script, str(ROOT / "scripts/run_intervention.py"), str(tmp_path / "report")],
                            cwd=tmp_path, env=env, capture_output=True, text=True, encoding="utf-8", timeout=45)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == load_experiment_report(tmp_path / "report").metrics
