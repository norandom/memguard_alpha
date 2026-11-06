"""Offline runner integration: cloud calls and inference transports are faked."""

import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from deploy.modal.manifest import load_plan
from scripts import probe_modal_model as probe

MANIFEST = Path(__file__).resolve().parents[2] / "configs/modal/gpt-oss-20b-pilot.json"


def test_default_plan_is_offline_and_freezes_prompts(tmp_path):
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
         "workspace": "test", "provider_budget_reference": "operator-reviewed-record"}))
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


def fake_cloud(monkeypatch, plan, *, arming_failure=False):
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
    monkeypatch.setattr(probe.serve, "create_app", lambda **kwargs: app)
    monkeypatch.setattr(probe.serve, "create_expiry_app", lambda **kwargs: cpu)
    monkeypatch.setattr(probe.serve, "experiment_lifecycle", lifecycle)
    monkeypatch.setattr(probe.serve, "read_observations", lambda *a, **k: [ready(plan, "ready"), ready(plan, "closing")])
    return sdk, events


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
    manifest, approval, path = configured
    monkeypatch.setattr(probe, "tokenizer_counts", lambda *a: (counts, None))
    receipt = asyncio.run(probe.execute(manifest, path, approval, sdk=object()))
    assert receipt["status"] == "stopped" and not receipt["attempts"]
    assert not receipt["lifecycle"]


def test_cpu_cache_inventory_rejects_config_without_weights():
    async def entries(*args, **kwargs):
        return [SimpleNamespace(path="config.json", size=10)]

    sdk = SimpleNamespace(Volume=SimpleNamespace(from_name=lambda *a, **k:
        SimpleNamespace(listdir=SimpleNamespace(aio=entries))))
    with pytest.raises(ValueError, match="cpu_staged_weight_cache_missing"):
        asyncio.run(probe.verify_weight_cache(load_plan(MANIFEST), sdk))
