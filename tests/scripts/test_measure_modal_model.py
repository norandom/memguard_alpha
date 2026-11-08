"""Offline Modal measurement runner: the control plane and transports are faked."""

import asyncio
import json
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from deploy.modal import serve
from deploy.modal.manifest import MEASUREMENT_MANIFEST, PILOT_MANIFEST, load_plan
from recall_guard.core.local_evidence import LocalModelIdentity, LocalModelProvenance
from recall_guard.core.providers import OpenAICompatibleClient
from recall_guard.harness.local_execution import HostObservation, execute_plan
from scripts import measure_modal_model as measure
from scripts import probe_modal_model as probe

ROOT = Path(__file__).resolve().parents[2]
APP_ID = "ap-" + "a" * 22
EXPIRY_ID = "ap-" + "c" * 22
SECRET = "test.secret-never-export"


def write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def sha(path):
    return probe.digest(path.read_bytes())


def plan_inputs(tmp_path, endpoint, *, sizes=(8, 8, 2), draws=0, max_requests=None, max_elapsed=2700):
    profile = measure.load_profile(measure.PROFILE_ID)
    config = profile.model_config.to_dict() | {"endpoint": endpoint + "/chat/completions",
                                               "local_identity": LocalModelIdentity().to_dict()}
    manifest = write_json(tmp_path / "measurement-manifest.json", {
        "config": config, "authentication": "environment", "credential_ref": "env:MODAL_PROXY_BEARER",
        "requested_execution_mode": "GPU"})
    corpus = write_json(tmp_path / "corpus.json", {
        split: [{"item_id": f"{split}-{i}", "prompt": f"{split}-{i} private", "input_tokens": 40}
                for i in range(count)]
        for split, count in zip(("is_memorized", "oos_control", "held_out"), sizes, strict=True)})
    logical = 1 + sizes[0] + 2 * sizes[1] + sizes[2] * (1 + draws)
    max_requests = logical if max_requests is None else max_requests
    plan = write_json(tmp_path / "plan.json", {
        "schema_version": 1, "manifest_sha256": sha(manifest), "corpus_sha256": sha(corpus),
        "prompt_template": "{prompt}", "selected_features": ["mia", "mcs"], "ensemble_draws": draws,
        "min_valid": 4, "seed": 0, "min_auc": 0.6,
        "limits": {"max_requests": max_requests, "context_tokens": 4096, "max_input_tokens": 600,
                   "max_output_tokens": 512, "max_total_tokens": max_requests * 1112, "max_elapsed_s": max_elapsed}})
    return plan, manifest, corpus


@pytest.fixture
def setup(tmp_path, monkeypatch):
    raw = json.loads((ROOT / MEASUREMENT_MANIFEST).read_text(encoding="utf-8"))
    raw["target"].update(workspace="test", endpoint="https://test.modal.run/v1")
    modal_manifest = write_json(tmp_path / "modal.json", raw)
    tokenizer = (tmp_path / "models/hub/models--openai--gpt-oss-20b/snapshots"
                 / raw["artifacts"]["tokenizer_revision"])
    tokenizer.mkdir(parents=True)
    (tokenizer / "tokenizer.json").write_text("{}")
    (tokenizer / "tokenizer_config.json").write_text("{}")
    profile = measure.load_profile(measure.PROFILE_ID)
    profile = replace(profile, harmony=replace(profile.harmony, tokenizer_path=str(tokenizer / "tokenizer.json")))
    monkeypatch.setattr(measure, "load_profile", lambda _: profile)
    monkeypatch.setenv("MODAL_PROXY_BEARER", SECRET)
    inputs = plan_inputs(tmp_path, "https://test.modal.run/v1")
    options = {"modal_manifest": modal_manifest, "tokenizer_root": tmp_path / "models"}
    return SimpleNamespace(tmp=tmp_path, inputs=inputs, options=options, modal_manifest=modal_manifest,
                           plan=load_plan(modal_manifest), approval=tmp_path / "approval.json")


def write_approval(setup, **budget):
    frozen = measure.measurement_planning(*setup.inputs, **setup.options)
    record = dict.fromkeys(probe.APPROVAL_ASSERTIONS, True) | measure.approval_hashes(frozen) | {
        "workspace": "test", "provider_budget_reference": "operator-reviewed-record",
        "monthly_budget": {"scope": "environment_compute", "workspace": "test", "environment": "recall-guard-pilot",
                           "period_month": datetime.now(UTC).strftime("%Y-%m"),
                           "checked_at": datetime.now(UTC).isoformat(), "source": "operator_review",
                           "monthly_limit_usd": 30, "prior_usage_usd": 0, "requested_allowance_usd": 2,
                           "accounting_available": True, "operator_confirmed": True} | budget}
    return write_json(setup.approval, record)


def install_transport(monkeypatch, calls, *, fail_at=None):
    def post(client, payload):
        calls.append(payload)
        assert client.config.max_retries == 0 and payload["max_tokens"] == 512 and payload["top_logprobs"] == 5
        if len(calls) == fail_at:
            raise requests.Timeout("private secret")
        prompt = payload["messages"][-1]["content"]
        base = -0.2 if prompt.startswith("is_memorized") else -5 if prompt.startswith("oos_control") else -1
        base += (sum(map(ord, prompt)) % 10) / 1000
        content = "Direction: 1\nConfidence: 0.8"
        entries = [{"token": char, "logprob": base + i / 1000,
                    "top_logprobs": [{"token": str(k), "logprob": base - k} for k in range(5)]}
                   for i, char in enumerate(content)]
        body = {"choices": [{"message": {"content": content}, "logprobs": {"content": entries},
                             "finish_reason": "stop"}]}
        return SimpleNamespace(status_code=200, headers={}, json=lambda: body)

    monkeypatch.setattr(OpenAICompatibleClient, "_post", post)


def generic_connect(profile, budget, *, environment):
    """Same endpoint resolution as connect, without the pinned Harmony tokenizer."""
    endpoint = environment[profile.endpoint_ref.removeprefix("env:")]
    return OpenAICompatibleClient(replace(profile.model_config, endpoint=endpoint), budget, environment=environment)


def aio(fn):
    return SimpleNamespace(aio=fn)


class Cloud:
    """Fake control plane driving the real experiment_lifecycle and stop paths."""

    def __init__(self, plan, hashes):
        self.plan, self.events, self.store, self.state = plan, [], {}, {APP_ID: "deployed", EXPIRY_ID: "deployed"}
        row = {"app_id": APP_ID, "app_name": plan.target.app, "environment": plan.target.environment,
               "resolved_model_revision": plan.artifacts.model_revision,
               "resolved_tokenizer_revision": plan.artifacts.tokenizer_revision, "runtime": plan.artifacts.runtime,
               "tokenizer_sha256": hashes, "score_semantics": plan.artifacts.score_semantics,
               "gpu": [{"name": "A100", "uuid": "one", "driver_version": "550"}],
               "quantization_config": {"quant_method": "mxfp4"}, "server_command": serve.server_command(plan)}
        self.row = row
        self.store[(APP_ID, "one", "ready")] = row | {"phase": "ready"}
        self.App = SimpleNamespace(lookup=aio(self.lookup))
        dictionary = SimpleNamespace(items=lambda: list(self.store.items()), get=aio(self._get),
                                     put=aio(self._put), hydrate=aio(self._hydrate))
        self.Dict = SimpleNamespace(from_name=lambda *a, **k: dictionary)
        self.Function = SimpleNamespace(from_name=lambda *a, **k: SimpleNamespace(spawn=aio(self.spawn)))
        service = SimpleNamespace(readiness=SimpleNamespace(remote=aio(self.readiness)),
                                  web=SimpleNamespace(get_web_url=aio(self.url)))
        self.Cls = SimpleNamespace(from_name=lambda *a, **k: lambda: service)

    async def _get(self, key):
        return self.store.get(key)

    async def _put(self, key, value, skip_if_exists=False):
        self.store[key] = value
        return True

    async def _hydrate(self):
        self.events.append("hydrate")

    async def spawn(self, app_id, deadline, arm_id):
        self.events.append("arm")
        self.store[(app_id, "expiry")] = {"app_id": app_id, "arm_id": arm_id, "deadline_unix_s": deadline}
        return SimpleNamespace(object_id="fc-test")

    async def lookup(self, name, **kwargs):
        app_id = EXPIRY_ID if name.endswith("-expiry") else APP_ID

        async def stop():
            self.events.append("stop_expiry" if app_id == EXPIRY_ID else "stop")
            self.state[app_id] = "stopped"
            if app_id == APP_ID:
                self.store[(APP_ID, "one", "closing")] = self.row | {"phase": "closing"}

        async def info(refresh):
            return SimpleNamespace(lifecycle=SimpleNamespace(state=SimpleNamespace(value=self.state[app_id])))

        return SimpleNamespace(app_id=app_id, stop=aio(stop), info=aio(info))

    async def readiness(self):
        self.events.append("readiness")
        return {"ready": True, "app_id": APP_ID}

    async def url(self):
        return "https://test.modal.run"


@pytest.fixture
def cloud(setup, monkeypatch):
    monkeypatch.setattr(serve, "require_operator_python", lambda: None)
    hashes = probe.tokenizer_hashes(setup.plan, setup.options["tokenizer_root"])
    sdk = Cloud(setup.plan, hashes)

    async def cache(*args):
        sdk.events.append("cache")
        return {"status": "verified_inventory"}

    def factory(name):
        def create(*, sdk, manifest):
            assert manifest == MEASUREMENT_MANIFEST
            app = SimpleNamespace(app_id=APP_ID if name == "gpu" else None)

            async def deploy(**kwargs):
                sdk.events.append(name)

            app.deploy = aio(deploy)
            return app
        return create

    monkeypatch.setattr(probe, "verify_weight_cache", cache)
    monkeypatch.setattr(probe, "unauthenticated_health_guard", lambda *a: sdk.events.append("401"))
    monkeypatch.setattr(serve, "create_expiry_app", factory("cpu"))
    monkeypatch.setattr(serve, "create_app", factory("gpu"))
    monkeypatch.setattr(measure, "connect", generic_connect)
    return sdk


def run(setup, cloud, journal_name="journal.json", receipt_name="receipt.json"):
    return asyncio.run(measure.execute_measurement(*setup.inputs, setup.approval, setup.tmp / journal_name,
                                                   setup.tmp / receipt_name, sdk=cloud, **setup.options))


def test_dry_run_phase_a_counts_cost_and_no_cloud(tmp_path, setup, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("dry run must not touch the SDK, network or credentials")

    monkeypatch.setattr(serve, "_sdk", forbidden)
    monkeypatch.setattr(requests, "Session", forbidden)
    monkeypatch.delenv("MODAL_PROXY_BEARER")
    phase_a = tmp_path / "phase-a"
    phase_a.mkdir()
    inputs = plan_inputs(phase_a, "https://test.modal.run/v1", sizes=(40, 100, 80))
    frozen = measure.measurement_planning(*inputs, **setup.options)
    assert frozen["logical_requests"] == frozen["physical_requests"] == 321
    assert frozen["items_per_split"] == {"is_memorized": 40, "oos_control": 100, "held_out": 80}
    assert frozen["worst_case_tokens"] == 321 * 1112
    assert frozen["plan_blockers"] == []
    assert frozen["inference_posts"] == frozen["cloud_mutations"] == 0
    estimate = frozen["cost_estimate"]
    assert estimate["gpu_phase_seconds"] == 2700
    assert 1.9 < estimate["worst_case_total_usd"] <= 2.0
    assert "private" not in json.dumps(frozen)


def test_dry_run_reports_plan_blockers(tmp_path, setup):
    other = tmp_path / "other"
    other.mkdir()
    inputs = plan_inputs(other, "https://elsewhere.modal.run/v1", draws=4, max_requests=400)
    blockers = measure.measurement_planning(*inputs, **setup.options)["plan_blockers"]
    assert "plan_endpoint_differs_from_manifest" in blockers
    assert "planned_requests_exceed_manifest_posts" not in blockers  # 1 + 8 + 16 + 2 x 5 = 35
    huge = tmp_path / "huge"
    huge.mkdir()
    inputs = plan_inputs(huge, "https://test.modal.run/v1", sizes=(40, 100, 80), draws=4)
    assert "planned_requests_exceed_manifest_posts" in measure.measurement_planning(
        *inputs, **setup.options)["plan_blockers"]


def test_cli_dry_run_is_default(setup, capsys):
    plan, manifest, corpus = setup.inputs
    code = measure.main(["--plan", str(plan), "--manifest", str(manifest), "--corpus", str(corpus),
                         "--modal-manifest", str(setup.modal_manifest),
                         "--tokenizer-root", str(setup.options["tokenizer_root"])])
    assert code == 0
    assert json.loads(capsys.readouterr().out)["mode"] == "measurement_plan"


def test_completed_run_uses_journal_report_and_verified_cleanup(setup, cloud, monkeypatch):
    write_approval(setup)
    calls = []
    install_transport(monkeypatch, calls)
    receipt = run(setup, cloud)
    assert receipt["status"] == "completed", receipt.get("diagnostics")
    assert len(calls) == receipt["inference_posts"] == 27
    assert all(call["model"] == "openai/gpt-oss-20b" for call in calls)
    assert cloud.events[:5] == ["cache", "cpu", "gpu", "hydrate", "arm"]
    assert cloud.events.index("401") < cloud.events.index("readiness")
    assert cloud.events[-2:] == ["stop", "stop_expiry"]
    lifecycle = receipt["lifecycle"]
    assert lifecycle["cleanup"]["cleanup"] == lifecycle["expiry_cleanup"]["cleanup"] == "verified_stopped"
    assert lifecycle["expiry_cleanup"]["app_id"] == EXPIRY_ID
    assert receipt["environment_status"] == "observed_stable"
    assert receipt["measurement"]["calibrated_claims_allowed"]
    assert receipt["measurement"]["observed_execution_mode"] == "GPU"
    journal = json.loads((setup.tmp / "journal.json").read_text())
    assert len(journal["attempts"]) == 27 and journal["status"] == "completed"
    assert (setup.tmp / "journal.json.report.json").exists()
    assert receipt["cost_report"]["physical_inference_attempt_count"] == 27
    durable = (setup.tmp / "receipt.json").read_text() + (setup.tmp / "journal.json.report.json").read_text()
    assert SECRET not in durable and "private" not in durable


def test_approval_mismatch_refuses_before_cloud(setup, cloud, monkeypatch):
    write_approval(setup)
    record = json.loads(setup.approval.read_text())
    record["corpus_sha256"] = "0" * 64
    write_json(setup.approval, record)
    calls = []
    install_transport(monkeypatch, calls)
    receipt = run(setup, cloud)
    assert receipt["status"] == "stopped" and receipt["error_stage"] == "budget_preflight"
    assert receipt["diagnostics"][0]["message"] == "approval_plan_mismatch"
    assert not cloud.events and not calls and not receipt["lifecycle"]
    assert not (setup.tmp / "journal.json").exists()


def test_usd_cap_below_worst_case_refuses_before_cloud(setup, cloud, monkeypatch):
    write_approval(setup, requested_allowance_usd=1.5)
    calls = []
    install_transport(monkeypatch, calls)
    receipt = run(setup, cloud)
    assert receipt["error_stage"] == "budget_preflight"
    assert receipt["diagnostics"][0]["message"] == "usd_cap_below_worst_case_estimate"
    assert not cloud.events and not calls


def test_existing_journal_never_replays(setup, cloud, monkeypatch):
    write_approval(setup)
    calls = []
    install_transport(monkeypatch, calls)
    assert run(setup, cloud)["status"] == "completed"
    events, posts = list(cloud.events), len(calls)
    with pytest.raises(FileExistsError):
        run(setup, cloud, receipt_name="second-receipt.json")
    assert cloud.events == events and len(calls) == posts
    assert not (setup.tmp / "second-receipt.json").exists()


def test_uncertain_outcome_stops_without_replay_and_still_cleans_up(setup, cloud, monkeypatch):
    write_approval(setup)
    calls = []
    install_transport(monkeypatch, calls, fail_at=5)
    receipt = run(setup, cloud)
    assert len(calls) == receipt["inference_posts"] == 5
    assert receipt["measurement"]["status"] == "stopped"
    assert receipt["lifecycle"]["expiry_cleanup"]["cleanup"] == "verified_stopped"
    assert "private secret" not in (setup.tmp / "receipt.json").read_text()


def test_deadline_closes_admission_without_replay(setup, monkeypatch):
    plan = measure.load_measurement_plan(*setup.inputs)
    clock = [1000.0]
    calls = []
    install_transport(monkeypatch, calls)
    monkeypatch.setattr(measure, "connect", generic_connect)

    def post_and_tick(client, payload):
        clock[0] += 100
        return original(client, payload)

    original = OpenAICompatibleClient._post
    monkeypatch.setattr(OpenAICompatibleClient, "_post", post_and_tick)
    lease = {"admission_open": True}
    budget = measure.ModalAttemptBudget(plan, setup.tmp / "deadline.json", deadline_unix_s=1000 + 120 + 450,
                                        lease=lease, endpoint="https://test.modal.run/v1/chat/completions",
                                        wall_clock=lambda: clock[0])

    def observer():
        return HostObservation(LocalModelProvenance(runtime="vllm", context_length=4096, backend="GPU"),
                               ("openai/gpt-oss-20b",))

    result = execute_plan(budget, observer)
    # Admissions at t=1000..1400 fit a 120 s timeout before 1570; t=1500 does not.
    assert len(calls) == budget.used == 5
    assert result.status == "stopped" and not result.calibrated_claims_allowed
    journal = json.loads((setup.tmp / "deadline.json").read_text())
    assert len(journal["attempts"]) == 5 and journal["status"] == "stopped"
    with pytest.raises(ValueError, match="already_started"):
        budget.begin_execution()
    lease["admission_open"] = False
    with pytest.raises(FileExistsError):
        measure.ModalAttemptBudget(plan, setup.tmp / "deadline.json", deadline_unix_s=1e12, lease=lease,
                                   endpoint="https://test.modal.run/v1/chat/completions")


def test_closed_lease_admits_nothing(setup, monkeypatch):
    plan = measure.load_measurement_plan(*setup.inputs)
    calls = []
    install_transport(monkeypatch, calls)
    monkeypatch.setattr(measure, "connect", generic_connect)
    budget = measure.ModalAttemptBudget(plan, setup.tmp / "closed.json", deadline_unix_s=time.time() + 3000,
                                        lease={"admission_open": False},
                                        endpoint="https://test.modal.run/v1/chat/completions")
    result = execute_plan(budget, lambda: HostObservation(LocalModelProvenance(context_length=4096), ()))
    assert not calls and budget.used == 0 and not result.calibrated_claims_allowed


def test_deployed_endpoint_must_equal_plan_endpoint(setup, monkeypatch):
    plan = measure.load_measurement_plan(*setup.inputs)
    monkeypatch.setattr(measure, "connect", generic_connect)
    budget = measure.ModalAttemptBudget(plan, setup.tmp / "endpoint.json", deadline_unix_s=time.time() + 3000,
                                        lease={"admission_open": True},
                                        endpoint="https://other.modal.run/v1/chat/completions")
    with pytest.raises(ValueError, match="deployed_endpoint_differs_from_plan"):
        budget.client()


def test_real_connect_builds_harmony_client_from_deployed_endpoint(tmp_path, monkeypatch):
    profile = measure.load_profile(measure.PROFILE_ID)
    if not profile.harmony.resolve_path().is_file():
        pytest.skip("pinned GPT-OSS tokenizer is not present in this checkout")
    monkeypatch.setenv("MODAL_PROXY_BEARER", SECRET)
    endpoint = "https://test.modal.run/v1"
    plan = measure.load_measurement_plan(*plan_inputs(tmp_path, endpoint))
    budget = measure.ModalAttemptBudget(plan, tmp_path / "harmony.json", deadline_unix_s=time.time() + 3000,
                                        lease={"admission_open": True}, endpoint=endpoint + "/chat/completions")
    client = budget.client()
    assert isinstance(client, probe.HarmonyOpenAICompatibleClient)
    assert client.config.endpoint == endpoint + "/chat/completions"
    assert client.budget is budget


def test_measurement_manifest_is_reviewed_and_pilot_is_unchanged():
    measurement, pilot = load_plan(ROOT / MEASUREMENT_MANIFEST), load_plan(ROOT / PILOT_MANIFEST)
    assert measurement.serving_manifest == MEASUREMENT_MANIFEST and pilot.serving_manifest == PILOT_MANIFEST
    assert (measurement.limits.gpu_phase_seconds, measurement.limits.inference_posts) == (2700, 321)
    assert pilot.limits.gpu_phase_seconds == 1200 and pilot.limits.inference_posts == 8
    assert (measurement.target, measurement.artifacts, measurement.resources) == (
        pilot.target, pilot.artifacts, pilot.resources)


def test_measurement_manifest_limits_cannot_be_raised(tmp_path):
    raw = json.loads((ROOT / MEASUREMENT_MANIFEST).read_text(encoding="utf-8"))
    raw["limits"]["gpu_phase_seconds"] = 3600
    with pytest.raises(ValueError, match="Measurement limits"):
        load_plan(write_json(tmp_path / "raised.json", raw))
    raw = json.loads((ROOT / PILOT_MANIFEST).read_text(encoding="utf-8"))
    raw["limits"]["gpu_phase_seconds"] = 2700
    with pytest.raises(ValueError, match="Pilot limits"):
        load_plan(write_json(tmp_path / "pilot.json", raw))


def test_serving_and_expiry_recipes_follow_selected_manifest(monkeypatch):
    monkeypatch.setattr(serve, "require_operator_python", lambda: None)
    options, uploads = {}, []

    def function(**kwargs):
        options.update(kwargs)
        return lambda fn: fn

    image = SimpleNamespace(add_local_file=lambda path, remote: uploads.append(remote) or image)
    sdk = SimpleNamespace(App=lambda name: SimpleNamespace(name=name, function=function),
                          Image=SimpleNamespace(debian_slim=lambda **k: image), concurrent=lambda **k: lambda fn: fn)
    serve.create_expiry_app(sdk=sdk, manifest=MEASUREMENT_MANIFEST)
    assert options["timeout"] == 2700 + 150
    assert uploads[-1] == "/root/" + MEASUREMENT_MANIFEST
    uploads.clear()
    serve.create_expiry_app(sdk=sdk)
    assert options["timeout"] == 1350 and uploads[-1] == "/root/" + PILOT_MANIFEST
    with pytest.raises(ValueError, match="Unreviewed"):
        serve.create_expiry_app(sdk=sdk, manifest="configs/modal/other.json")
