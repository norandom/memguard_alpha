"""Offline runner integration: cloud calls and inference transports are faked."""

import asyncio
import json
import sys
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from deploy.modal.manifest import load_plan
from scripts import probe_modal_model as probe

MANIFEST = Path(__file__).resolve().parents[2] / "configs/modal/gpt-oss-20b-pilot.json"


def test_default_plan_is_offline_and_freezes_prompts(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("planning must not create an HTTP session")

    monkeypatch.setattr(requests, "Session", forbidden)
    monkeypatch.setattr(probe.serve, "_sdk", forbidden)
    original_read = Path.read_text

    def no_env_file(path, *args, **kwargs):
        assert path.name != ".env"
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", no_env_file)
    with monkeypatch.context() as blocked:
        blocked.setattr(probe.os.environ, "items", forbidden)
        result = probe.planning(MANIFEST, tokenizer_root=tmp_path)
    assert result["mode"] == "plan"
    assert result["inference_posts"] == result["cloud_mutations"] == 0
    assert len(result["prompts"]) == 8
    assert len(result["prompts_sha256"]) == 64
    assert result["input_tokens"] is None
    assert "pinned_tokenizer_unavailable" in result["execution_blockers"]


def ready(plan, phase="ready", **overrides):
    return dict(resolved_model_revision=plan.artifacts.model_revision,
                resolved_tokenizer_revision=plan.artifacts.tokenizer_revision,
                runtime=plan.artifacts.runtime, tokenizer_sha256={"test": "hash"},
                score_semantics=plan.artifacts.score_semantics, gpu=[{"name": "A100", "uuid": "one"}],
                quantization_config={"quant_method": "mxfp4"}, phase=phase) | overrides


@pytest.fixture
def configured(tmp_path, monkeypatch):
    raw = json.loads(MANIFEST.read_text())
    raw["target"].update(workspace="test", endpoint="https://test.modal.run/v1")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(raw))
    monkeypatch.setattr(probe, "tokenizer_counts", lambda *args: ([20] * 8, {"test": "hash"}))
    monkeypatch.setenv("MODAL_PROXY_BEARER", "test.secret-never-export")
    frozen = probe.planning(manifest)
    approval = tmp_path / "approval.json"
    approval.write_text(json.dumps(dict.fromkeys(("paid_execution_authorized", "account_access_verified",
        "operator_allowance_confirmed", "model_access_verified", "pricing_and_budget_reviewed"), True) |
        {"manifest_sha256": frozen["manifest_sha256"], "prompts_sha256": frozen["prompts_sha256"],
         "workspace": "test", "provider_budget_reference": "operator-reviewed-record",
         "monthly_budget": {"scope": "environment_compute", "workspace": "test", "environment": "recall-guard-pilot",
                            "period_month": datetime.now(UTC).strftime("%Y-%m"), "checked_at": datetime.now(UTC).isoformat(),
                            "source": "operator_review", "monthly_limit_usd": 20, "prior_usage_usd": 0,
                            "requested_allowance_usd": 5, "accounting_available": True, "operator_confirmed": True}}))
    return manifest, approval, tmp_path / "receipt.json"


def install_transport(monkeypatch, path, *, malformed=False, failure=None):
    calls = []

    def post(self, payload):
        receipt = json.loads(path.read_text())
        assert receipt["attempts"][-1]["status"] == "intent"
        assert self.config.max_retries == 0
        assert self.config.timeout_s <= 120
        assert payload["temperature"] == 0 and payload["top_p"] == 1
        assert payload["max_tokens"] == 512 and payload["top_logprobs"] == 5
        calls.append(payload)
        if failure:
            raise failure
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps({"choices": [{"message": {"content": "OK"},
            "logprobs": {"content": [{"token": "reasoning" if malformed else "OK", "logprob": -.1,
            "top_logprobs": [{"token": str(i), "logprob": -i-1.0} for i in range(5)]}]}}]}).encode()
        return response

    monkeypatch.setattr(probe.OpenAICompatibleClient, "_post", post)
    return calls


def attempt_receipt(manifest):
    return {"plan": probe.planning(manifest), "attempts": [], "app_id": "ap-" + "a" * 22,
            "lifecycle": {"deadline_unix_s": time.time() + 1000, "admission_open": True}}


def test_eight_serial_posts_have_durable_intents_and_no_raw_output(configured, monkeypatch):
    manifest, _, path = configured
    calls = install_transport(monkeypatch, path)
    receipt = attempt_receipt(manifest)
    asyncio.run(probe.attempts(load_plan(manifest), receipt, path))
    assert len(calls) == receipt["inference_posts"] == 8
    assert [row["stage"] for row in receipt["attempts"]] == ["eligibility"] + ["measurement"] * 7
    assert receipt["status"] == "completed"
    assert "test.secret" not in path.read_text() and '"content"' not in path.read_text()


@pytest.mark.parametrize("failure,malformed", [(None, True), (requests.Timeout("private secret"), False)])
def test_ineligible_or_uncertain_eligibility_stops_without_replay(configured, monkeypatch, failure, malformed):
    manifest, _, path = configured
    calls = install_transport(monkeypatch, path, failure=failure, malformed=malformed)
    receipt = attempt_receipt(manifest)
    asyncio.run(probe.attempts(load_plan(manifest), receipt, path))
    assert len(calls) == 1
    assert receipt["attempts"][0]["status"] != "eligible"
    assert "private secret" not in path.read_text()


@pytest.mark.parametrize("expired,cancelled", [(True, False), (False, True)])
def test_deadline_and_cancellation_prevent_admission(configured, monkeypatch, expired, cancelled):
    manifest, _, path = configured
    calls = install_transport(monkeypatch, path)
    receipt = attempt_receipt(manifest)
    if expired:
        receipt["lifecycle"]["deadline_unix_s"] = time.time() - 1
    asyncio.run(probe.attempts(load_plan(manifest), receipt, path, cancelled=lambda: cancelled))
    assert not calls
    assert receipt["status"] == "admission_closed"


def test_identity_unknown_and_drift_never_establish_match():
    plan = load_plan(MANIFEST)
    assert probe.environment_status(plan, [], {"test": "hash"}) == "unknown"
    assert probe.environment_status(plan, [ready(plan)], {"test": "hash"}) == "observed_stable"
    assert probe.environment_status(plan, [ready(plan, gpu=None)], {"test": "hash"}) == "unknown_or_changed"
    assert probe.environment_status(plan, [ready(plan), ready(plan, "closing", gpu=[{"uuid": "other"}])],
                                    {"test": "hash"}) == "drift"


def fake_cloud(monkeypatch, plan, *, arming_failure=False, health_guard=True):
    # This fake control plane is independent of the test runner's Python minor.
    monkeypatch.setattr(probe.serve, "require_operator_python", lambda: None)
    monkeypatch.setenv("TEST_PROVIDER_SECRET", "provider private secret")
    events = []
    app = SimpleNamespace(app_id="ap-" + "a" * 22)

    async def deploy_cpu(**kwargs):
        events.append("cpu")

    async def deploy_gpu(**kwargs):
        events.append("gpu")

    app.deploy = SimpleNamespace(aio=deploy_gpu)
    cpu = SimpleNamespace(deploy=SimpleNamespace(aio=deploy_cpu))

    @asynccontextmanager
    async def lifecycle(plan, app_id, *, gpu_started_at_unix_s, receipt, sdk):
        events.append("arm")
        assert gpu_started_at_unix_s <= time.time()
        try:
            if arming_failure:
                raise RuntimeError("provider private secret")
            receipt.update(admission_open=True, deadline_unix_s=gpu_started_at_unix_s + 1200)
            yield receipt
        finally:
            receipt.update(admission_open=False, cleanup={"cleanup": "verified_stopped"})
            events.append("stop")

    async def readiness():
        events.append("readiness")
        return {"ready": True, "app_id": app.app_id}

    async def url():
        return "https://test.modal.run"

    service = SimpleNamespace(readiness=SimpleNamespace(remote=SimpleNamespace(aio=readiness)),
                              web=SimpleNamespace(get_web_url=SimpleNamespace(aio=url)))
    sdk = SimpleNamespace(Cls=SimpleNamespace(from_name=lambda *a, **k: lambda: service))

    async def cache(*args):
        events.append("cache")
        return {"status": "verified_inventory"}

    monkeypatch.setattr(probe, "verify_weight_cache", cache)
    if health_guard:
        monkeypatch.setattr(probe, "unauthenticated_health_guard", lambda *a: None)
    monkeypatch.setattr(probe.serve, "create_app", lambda **kwargs: app)
    monkeypatch.setattr(probe.serve, "create_expiry_app", lambda **kwargs: cpu)
    monkeypatch.setattr(probe.serve, "experiment_lifecycle", lifecycle)
    monkeypatch.setattr(probe.serve, "read_observations", lambda *a, **k: [ready(plan, "ready"), ready(plan, "closing")])
    return sdk, events


def test_registered_endpoint_mismatch_stops_before_http_or_readiness(monkeypatch, configured):
    manifest, approval, path = configured
    plan = load_plan(manifest)
    sdk, events = fake_cloud(monkeypatch, plan)

    async def different_url():
        return "https://different.modal.run"

    sdk.Cls.from_name()().web.get_web_url.aio = different_url

    def no_http(*args):
        raise AssertionError("HTTP must follow endpoint identity verification")

    monkeypatch.setattr(probe, "unauthenticated_health_guard", no_http)
    receipt = asyncio.run(probe.execute(manifest, path, approval, sdk=sdk))
    assert receipt["error_stage"] == "endpoint_check"
    assert "readiness" not in events
    assert receipt["inference_posts"] == 0
    assert receipt["lifecycle"]["cleanup"]["cleanup"] == "verified_stopped"


@pytest.mark.parametrize("outcome", [401, 200, 403, 302, requests.Timeout("private secret"),
                                     requests.ConnectionError("private secret")])
def test_unauthenticated_denial_precedes_readiness_and_preserves_cleanup(configured, monkeypatch, outcome):
    manifest, approval, path = configured
    sdk, events = fake_cloud(monkeypatch, load_plan(manifest), health_guard=False)
    calls = install_transport(monkeypatch, path)

    class Session:
        trust_env = True
        headers = {"Authorization": "inherited private secret"}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, url, **kwargs):
            assert events == ["cache", "cpu", "gpu", "arm"]
            assert not self.trust_env and not self.headers
            assert url == "https://test.modal.run/health"
            assert kwargs == {"allow_redirects": False, "timeout": (5, 10)}
            durable = json.loads(path.read_text())
            event = durable["control_events"][-1]
            assert event["operation"] == "unauthenticated_health"
            assert event["status"] == "intent" and event["inference_posts"] == 0
            assert durable["attempts"] == [] and durable["inference_posts"] == 0
            events.append("unauthenticated_health")
            if isinstance(outcome, Exception):
                raise outcome
            return SimpleNamespace(status_code=outcome)

    monkeypatch.setattr(requests, "Session", Session)
    receipt = asyncio.run(probe.execute(manifest, path, approval, sdk=sdk))
    assert "unauthenticated_health" in events
    assert ("readiness" in events) == (outcome == 401)
    assert len(calls) == (8 if outcome == 401 else 0)
    event = next(row for row in receipt["control_events"] if row["operation"] == "unauthenticated_health")
    assert event["status"] == ("denied" if outcome == 401 else "unavailable" if isinstance(outcome, Exception)
                               else "unexpected_status")
    assert event.get("http_status") == (None if isinstance(outcome, Exception) else outcome)
    assert receipt["lifecycle"]["cleanup"]["cleanup"] == "verified_stopped"
    assert events[-1] == "stop"
    assert "private secret" not in path.read_text()
    assert "test.secret" not in path.read_text()


@pytest.mark.parametrize("arming_failure", [False, True])
def test_execute_persists_cleanup_even_when_arming_fails(configured, monkeypatch, arming_failure):
    manifest, approval, path = configured
    sdk, events = fake_cloud(monkeypatch, load_plan(manifest), arming_failure=arming_failure)
    calls = install_transport(monkeypatch, path)
    receipt = asyncio.run(probe.execute(manifest, path, approval, sdk=sdk))
    assert receipt["lifecycle"]["cleanup"]["cleanup"] == "verified_stopped"
    assert events[:3] == ["cache", "cpu", "gpu"]
    assert events[-1] == "stop"
    assert len(calls) == (0 if arming_failure else 8)
    assert receipt["calibration"] == "unavailable"
    assert "private secret" not in path.read_text()
    with pytest.raises(FileExistsError):
        asyncio.run(probe.execute(manifest, path, approval, sdk=sdk))


@pytest.mark.parametrize("counts", [None, [3585] * 8])
def test_missing_tokenizer_or_context_overflow_blocks_before_cloud(configured, monkeypatch, counts):
    monkeypatch.setattr(probe.serve, "require_operator_python", lambda: None)
    manifest, approval, path = configured
    monkeypatch.setattr(probe, "tokenizer_counts", lambda *a: (counts, None))
    receipt = asyncio.run(probe.execute(manifest, path, approval, sdk=object()))
    assert receipt["status"] == "stopped" and not receipt["attempts"]
    assert not receipt["lifecycle"]


@pytest.mark.parametrize("minor", [11, 13, 14])
@pytest.mark.parametrize("sdk", [None, object()])
def test_wrong_python_blocks_execute_before_sdk_or_cache(configured, monkeypatch, minor, sdk):
    manifest, approval, path = configured
    monkeypatch.setattr(probe.serve, "sys", SimpleNamespace(version_info=(3, minor, 0)))
    calls = []

    def forbidden(*args, **kwargs):
        calls.append("sdk_or_cache")
        raise AssertionError("wrong Python must fail locally")

    monkeypatch.setattr(probe.serve, "_sdk", forbidden)
    monkeypatch.setattr(probe, "verify_weight_cache", forbidden)
    with pytest.raises(ValueError, match="Python 3.12"):
        asyncio.run(probe.execute(manifest, path, approval, sdk=sdk))
    assert calls == []
    assert not path.exists()


def test_cpu_cache_inventory_rejects_config_without_weights():
    async def entries(*args, **kwargs):
        return [SimpleNamespace(path="config.json", size=10)]

    sdk = SimpleNamespace(Volume=SimpleNamespace(from_name=lambda *a, **k:
        SimpleNamespace(listdir=SimpleNamespace(aio=entries))))
    with pytest.raises(ValueError, match="cpu_staged_weight_cache_missing"):
        asyncio.run(probe.verify_weight_cache(load_plan(MANIFEST), sdk))


def test_insufficient_monthly_headroom_stops_before_cloud(configured, monkeypatch):
    manifest, approval, path = configured
    record = json.loads(approval.read_text())
    record["monthly_budget"]["prior_usage_usd"] = 19
    approval.write_text(json.dumps(record))
    sdk, events = fake_cloud(monkeypatch, load_plan(manifest))
    receipt = asyncio.run(probe.execute(manifest, path, approval, sdk=sdk))
    assert not events and not receipt["attempts"]
    assert receipt["cost_report"]["observed_gross_usd"] is None
    assert receipt["cost_report"]["cold_startup_seconds"] is None


def test_cleanup_failure_remains_visible_after_success(configured, monkeypatch):
    manifest, approval, path = configured
    sdk, _ = fake_cloud(monkeypatch, load_plan(manifest))

    @asynccontextmanager
    async def failed_cleanup(plan, app_id, *, gpu_started_at_unix_s, receipt, sdk):
        receipt.update(admission_open=True, deadline_unix_s=time.time() + 1200)
        yield receipt
        receipt.update(admission_open=False, cleanup={"cleanup": "unverified"})

    monkeypatch.setattr(probe.serve, "experiment_lifecycle", failed_cleanup)
    install_transport(monkeypatch, path)
    receipt = asyncio.run(probe.execute(manifest, path, approval, sdk=sdk))
    assert receipt["status"] == "completed"
    assert receipt["cost_report"]["usable_measurement_count"] == 7
    assert receipt["cost_report"]["cleanup_state"] == "unverified"
    assert receipt["cost_report"]["cold_startup_seconds"] >= 0


def test_billing_cli_preserves_attempt_history_and_frozen_prices(configured, monkeypatch, capsys):
    manifest, _, path = configured
    receipt = {"plan": probe.planning(manifest), "attempts": [{"stage": "measurement", "status": "eligible"}],
               "status": "stopped", "lifecycle": {"cleanup": {"cleanup": "unverified"}}}
    probe.persist(path, receipt)
    billing = path.with_name("billing.json")
    billing.write_text(json.dumps({"observed_gross_usd": 2, "credits_usd": 1, "source": "billing_review",
                                  "checked_at": datetime.now(UTC).isoformat(), "coverage": "whole_pilot",
                                  "generated_text": "private-output", "credential": "private-token"}))
    monkeypatch.setattr(sys, "argv", ["probe", "--receipt", str(path), "--reconcile-billing", str(billing)])
    assert probe.main() == 0
    updated = json.loads(path.read_text())
    assert updated["attempts"] == receipt["attempts"] and updated["plan"] == receipt["plan"]
    assert updated["cost_report"]["observed_usd_per_usable_measurement"] == 2
    assert updated["cost_report"]["cleanup_state"] == "unverified"
    assert "private-" not in capsys.readouterr().out
    assert "private-" not in path.read_text()


@pytest.mark.parametrize("controller", [True, False])
def test_deployment_failure_has_durable_sanitized_stage(configured, monkeypatch, controller):
    manifest, approval, path = configured
    sdk, _ = fake_cloud(monkeypatch, load_plan(manifest))
    monkeypatch.setenv("TEST_API_KEY", "synthetic-key-993")
    monkeypatch.setenv("TEST_TOKEN", "synthetic-token-994")
    monkeypatch.setenv("TEST_PASSWORD", "synthetic-password-995")

    async def rejected(**kwargs):
        raise ValueError("Invalid deployment: synthetic-key-993 synthetic-token-994 synthetic-password-995 "
                         "test.secret-never-export secret-never-export Bearer extra-proxy.extra-secret")

    app = SimpleNamespace(app_id=None, deploy=SimpleNamespace(aio=rejected))
    monkeypatch.setattr(probe.serve, "create_expiry_app" if controller else "create_app", lambda **k: app)
    receipt = asyncio.run(probe.execute(manifest, path, approval, sdk=sdk))
    assert receipt["error"] == "ValueError"
    assert receipt["error_stage"] == ("controller_deploy" if controller else "model_deploy")
    diagnostic = receipt["diagnostics"][0]
    assert "Invalid deployment" in diagnostic["message"]
    assert diagnostic["frames"][-1]["function"] == "rejected"
    assert set(diagnostic["frames"][-1]) == {"filename", "function", "lineno"}
    durable = path.read_text()
    for secret in ("synthetic-key-993", "synthetic-token-994", "synthetic-password-995",
                   "test.secret-never-export", "secret-never-export", "extra-proxy", "extra-secret"):
        assert secret not in durable
    assert json.loads(durable)["diagnostics"] == receipt["diagnostics"]


def test_bounded_diagnostics_preserve_chain_and_exclude_inference_detail(monkeypatch):
    monkeypatch.setenv("TEST_SECRET", "nested-secret")
    try:
        try:
            raise ValueError("nested-secret")
        except ValueError as cause:
            raise RuntimeError("useful failure " + "x" * 10000) from cause
    except RuntimeError as exc:
        diagnostic = probe.error_diagnostic(exc, "controller_deploy")
        assert len(diagnostic["message"]) <= 2000
        assert diagnostic["chain"][0]["error"] == "ValueError"
        assert "nested-secret" not in json.dumps(diagnostic)
        hidden = probe.error_diagnostic(exc, "inference")
        assert "message" not in hidden and "chain" not in hidden


def test_primary_deploy_error_survives_cleanup_and_observation_failure(configured, monkeypatch):
    manifest, approval, path = configured
    sdk, _ = fake_cloud(monkeypatch, load_plan(manifest))

    async def rejected(**kwargs):
        raise ValueError("primary deployment rejection")

    async def cleanup(*args, **kwargs):
        raise RuntimeError("secondary cleanup rejection")

    def observations(*args, **kwargs):
        raise OSError("secondary observations rejection")

    app = SimpleNamespace(app_id="ap-" + "a" * 22, deploy=SimpleNamespace(aio=rejected))
    monkeypatch.setattr(probe.serve, "create_app", lambda **k: app)
    monkeypatch.setattr(probe.serve, "stop_experiment", cleanup)
    monkeypatch.setattr(probe.serve, "read_observations", observations)
    receipt = asyncio.run(probe.execute(manifest, path, approval, sdk=sdk))
    assert receipt["error"] == "ValueError" and receipt["error_stage"] == "model_deploy"
    assert receipt["diagnostics"][0]["message"] == "primary deployment rejection"
    assert [d["stage"] for d in receipt["diagnostics"]] == ["model_deploy", "cleanup", "provenance_after"]
    assert receipt["lifecycle"]["cleanup"]["cleanup"] == "unverified"
    assert json.loads(path.read_text())["error"] == "ValueError"


@pytest.mark.parametrize("stage", ["budget_preflight", "cache_inventory", "expiry_arming", "proxy_denial",
                                  "readiness", "endpoint_check", "provenance"])
def test_control_failure_stage_is_identifiable(configured, monkeypatch, stage):
    manifest, approval, path = configured
    sdk, _ = fake_cloud(monkeypatch, load_plan(manifest))

    def rejected(*args, **kwargs):
        raise ValueError("actionable control rejection")

    async def rejected_async(*args, **kwargs):
        rejected()

    if stage == "budget_preflight":
        monkeypatch.setattr(probe, "approval_check", rejected)
    elif stage == "cache_inventory":
        monkeypatch.setattr(probe, "verify_weight_cache", rejected_async)
    elif stage == "proxy_denial":
        monkeypatch.setattr(probe, "unauthenticated_health_guard", rejected)
    elif stage == "provenance":
        monkeypatch.setattr(probe.serve, "read_observations", rejected)
    elif stage == "expiry_arming":
        @asynccontextmanager
        async def lifecycle(*args, **kwargs):
            rejected()
            yield
        monkeypatch.setattr(probe.serve, "experiment_lifecycle", lifecycle)
    else:
        service = sdk.Cls.from_name()()
        monkeypatch.setattr(service.readiness.remote if stage == "readiness" else service.web.get_web_url,
                            "aio", rejected_async)
    receipt = asyncio.run(probe.execute(manifest, path, approval, sdk=sdk))
    assert receipt["status"] == "stopped" and receipt["error_stage"] == stage
    assert receipt["diagnostics"][0]["message"] == "actionable control rejection"
    assert len(receipt["diagnostics"]) == (2 if stage == "provenance" else 1)
    assert not receipt["attempts"]


def test_lifecycle_cleanup_exception_cannot_mask_readiness_reason(configured, monkeypatch):
    manifest, approval, path = configured
    sdk, _ = fake_cloud(monkeypatch, load_plan(manifest))

    @asynccontextmanager
    async def lifecycle(*args, receipt, **kwargs):
        receipt.update(admission_open=True, deadline_unix_s=time.time() + 1000)
        try:
            yield
        finally:
            receipt["cleanup"] = {"cleanup": "unverified"}
            raise RuntimeError("secondary lifecycle failure")

    async def readiness():
        raise ValueError("primary readiness rejection")

    monkeypatch.setattr(probe.serve, "experiment_lifecycle", lifecycle)
    monkeypatch.setattr(sdk.Cls.from_name()().readiness.remote, "aio", readiness)
    receipt = asyncio.run(probe.execute(manifest, path, approval, sdk=sdk))
    assert receipt["error"] == "ValueError" and receipt["error_stage"] == "readiness"
    assert [d["stage"] for d in receipt["diagnostics"]] == ["readiness", "cleanup"]
    assert receipt["diagnostics"][0]["message"] == "primary readiness rejection"


def test_diagnostics_bound_chain_and_never_serialize_request_error_payload():
    cause = None
    for _ in range(8):
        error = RuntimeError("x" * 10000)
        error.__cause__ = cause
        cause = error
    diagnostic = probe.error_diagnostic(cause, "model_deploy")
    assert len(diagnostic["chain"]) == 4
    assert len(json.dumps(diagnostic)) < 12000
    request_error = probe.RequestError("private-response-body", detail="private-provider-detail", status_code=503)
    for stage in ("inference", "model_deploy"):
        diagnostic = probe.error_diagnostic(request_error, stage)
        assert "private-" not in json.dumps(diagnostic)


def test_inference_diagnostic_persists_without_generated_exception_text(configured):
    manifest, _, path = configured

    class Client:
        def __init__(self, *args):
            pass

        def generate_for_stage(self, *args, **kwargs):
            raise RuntimeError("private-generated-response-body")

    receipt = attempt_receipt(manifest)
    asyncio.run(probe.attempts(load_plan(manifest), receipt, path, client_factory=Client))
    diagnostic = receipt["attempts"][0]["diagnostic"]
    assert diagnostic["stage"] == "inference" and diagnostic["error"] == "RuntimeError"
    assert diagnostic["frames"] and "message" not in diagnostic
    assert "private-generated" not in path.read_text()


def test_cleanup_chain_never_discloses_inference_failure_in_receipt_or_cli(configured, monkeypatch, capsys):
    manifest, approval, path = configured
    sdk, _ = fake_cloud(monkeypatch, load_plan(manifest))

    async def inference(*args, **kwargs):
        try:
            raise ValueError("private-nested-generated-answer")
        except ValueError as cause:
            raise ValueError("private-generated-answer") from cause

    @asynccontextmanager
    async def lifecycle(*args, receipt, **kwargs):
        receipt.update(admission_open=True, deadline_unix_s=time.time() + 1000)
        try:
            yield
        finally:
            receipt["cleanup"] = {"cleanup": "unverified"}
            raise RuntimeError("useful cleanup rejection")

    monkeypatch.setattr(probe, "attempts", inference)
    monkeypatch.setattr(probe.serve, "experiment_lifecycle", lifecycle)
    monkeypatch.setattr(probe.serve, "_sdk", lambda sdk: sdk or fake_sdk)
    fake_sdk = sdk
    monkeypatch.setattr(sys, "argv", ["probe", "--execute", "--manifest", str(manifest),
                                    "--approval", str(approval), "--receipt", str(path)])
    assert probe.main() == 1
    printed = capsys.readouterr().out
    durable = path.read_text()
    assert "private-generated-answer" not in durable
    assert "private-nested-generated-answer" not in durable
    assert "private-generated-answer" not in printed
    assert "private-nested-generated-answer" not in printed
    receipt = json.loads(durable)
    assert receipt["error_stage"] == "inference"
    assert receipt["diagnostics"][-1]["message"] == "useful cleanup rejection"


def test_request_error_chain_is_transitively_opaque():
    try:
        try:
            raise ValueError("private-deep-response-body")
        except ValueError as cause:
            raise ValueError("private-response-body") from cause
    except ValueError as cause:
        error = probe.RequestError("private-request-message", detail="private-request-detail")
        error.__cause__ = cause
    diagnostic = probe.error_diagnostic(error, "model_deploy")
    assert "private-" not in json.dumps(diagnostic)
    assert diagnostic["error"] == "RequestError"
    outer = RuntimeError("useful ordinary deployment failure")
    outer.__cause__ = error
    diagnostic = probe.error_diagnostic(outer, "model_deploy")
    assert "private-" not in json.dumps(diagnostic)
    assert diagnostic["message"] == "useful ordinary deployment failure"
