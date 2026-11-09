"""Offline serving route on Modal: the control plane is faked and credentials are fake values."""

import asyncio
import hashlib
import json
import re
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest

from deploy.modal import serve
from deploy.modal.manifest import GEMMA_MANIFEST, load_plan
from scripts import serve_modal_model as route

ROOT = Path(__file__).resolve().parents[2]
GEMMA = ROOT / GEMMA_MANIFEST
APP_ID, EXPIRY_ID = "ap-" + "a" * 22, "ap-" + "c" * 22
URL = "https://wishinet--recall-guard-gemma-3-4b.modal.run"
#: Fake values in each credential's real format; none is a real credential.
SECRETS = {"HF_TOKEN": "hf_" + "Q7" * 17, "MODAL_TOKEN_ID": "ak-" + "R8" * 11,
           "MODAL_TOKEN_SECRET": "as-" + "S9" * 11, "MODAL_PROXY_BEARER": "wk-" + "T1" * 10 + ".ws-" + "U2" * 10}
TOKEN_LIKE = re.compile(r"hf_[A-Za-z0-9]{20,}|\b(?:ak|as|wk|ws)-[A-Za-z0-9]{16,}")


def aio(fn):
    return SimpleNamespace(aio=fn)


def assert_no_token(text):
    assert not any(value in text for value in SECRETS.values())
    assert not any(part in text for part in SECRETS["MODAL_PROXY_BEARER"].split("."))
    assert not TOKEN_LIKE.search(text)


class NotFoundError(Exception):
    """Stands in for the SDK's error when a stopped deployment no longer resolves by name."""


class FakeModal:
    """Deploys, arms, serves and stops fake apps; no network, compute or credential use."""

    def __init__(self, plan, *, shards=("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")):
        self.plan, self.shards = plan, shards
        self.events, self.store, self.volumes, self.spawned = [], {}, [], []
        self.state = {}
        self.url = URL
        self.names = {plan.target.app: APP_ID, plan.target.app + "-expiry": EXPIRY_ID}
        self.App = SimpleNamespace(lookup=aio(self.lookup))
        self.Dict = SimpleNamespace(from_name=lambda *a, **k: SimpleNamespace(
            get=aio(self.get), put=aio(self.put), hydrate=aio(self.hydrate)))
        self.Function = SimpleNamespace(from_name=lambda *a, **k: SimpleNamespace(spawn=aio(self.spawn)))
        self.Volume = SimpleNamespace(from_name=self.volume)
        service = SimpleNamespace(web=SimpleNamespace(get_web_url=aio(self.web_url)))
        self.Cls = SimpleNamespace(from_name=lambda *a, **k: lambda: service)
        self.expiry_app = SimpleNamespace(deploy=aio(partial(self.deploy, EXPIRY_ID)))
        self.serving_app = SimpleNamespace(app_id=APP_ID, deploy=aio(partial(self.deploy, APP_ID)))

    async def deploy(self, app_id, *, environment_name):
        assert environment_name == self.plan.target.environment
        self.events.append(("deploy", app_id))
        self.state[app_id] = "deployed"

    def volume(self, name, **kwargs):
        self.volumes.append((name, kwargs))
        base = f"hub/models--google--gemma-3-4b-it/snapshots/{self.plan.artifacts.model_revision}"

        async def listdir(path, recursive):
            names = (*self.shards, "model.safetensors.index.json", "config.json", "tokenizer.json",
                     "tokenizer_config.json")
            return [SimpleNamespace(path=base + "/" + item, size=100) for item in names]

        async def read_file(path):
            yield json.dumps({"weight_map": {str(i): shard for i, shard in enumerate(self.shards)}}).encode()

        return SimpleNamespace(listdir=aio(listdir), read_file=aio(read_file))

    async def lookup(self, name, **kwargs):
        assert kwargs == {"environment_name": self.plan.target.environment, "create_if_missing": False}
        app_id = self.names[name]
        self.events.append(("lookup", app_id))
        if self.state.get(app_id) != "deployed":
            raise NotFoundError(name)
        return SimpleNamespace(app_id=app_id, stop=aio(partial(self.stop, app_id)),
                               info=aio(partial(self.info, app_id)))

    async def stop(self, app_id):
        self.events.append(("stop", app_id))
        self.state[app_id] = "stopped"

    async def info(self, app_id, *, refresh):
        return SimpleNamespace(lifecycle=SimpleNamespace(state=SimpleNamespace(value=self.state[app_id])))

    async def hydrate(self):
        self.events.append(("hydrate",))

    async def get(self, key):
        return self.store.get(key)

    async def put(self, key, value, skip_if_exists=False):
        if skip_if_exists and key in self.store:
            return False
        self.store[key] = value
        return True

    async def spawn(self, app_id, deadline_unix_s, arm_id):
        # The remote controller writes this acknowledgment once its input runs.
        self.spawned.append((app_id, deadline_unix_s, arm_id))
        self.store[(app_id, "expiry")] = {"app_id": app_id, "arm_id": arm_id, "deadline_unix_s": deadline_unix_s}
        return SimpleNamespace(object_id="fc-" + "f" * 22)

    async def web_url(self):
        return self.url


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(serve, "require_operator_python", lambda: None)
    for name, value in SECRETS.items():
        monkeypatch.setenv(name, value)
    study_plan = tmp_path / "frozen-study-plan.json"
    study_plan.write_text(json.dumps({"fixture": "frozen study plan", "arms": ["sampling", "serving"]}))
    plan_sha256 = hashlib.sha256(study_plan.read_bytes()).hexdigest()

    def limits(stage, **values):
        path = tmp_path / f"{stage}-limits.json"
        path.write_text(json.dumps({"plan_sha256": plan_sha256, "stage": stage, "max_requests": 0,
                                    "max_output_tokens": 0, "max_seconds": 300, "max_usd": 1.0} | values))
        return path

    plan = load_plan(GEMMA)
    cloud = FakeModal(plan)
    cloud.expiry_lifetimes = []
    monkeypatch.setattr(serve, "create_expiry_app", lambda *, sdk, manifest, lifetime_seconds=None: (
        cloud.events.append(("create_expiry", manifest)) or cloud.expiry_lifetimes.append(lifetime_seconds)
        or cloud.expiry_app))
    monkeypatch.setattr(serve, "create_app", lambda *, sdk, manifest: (
        cloud.events.append(("create_app", manifest)) or cloud.serving_app))
    return SimpleNamespace(tmp=tmp_path, plan=plan, plan_sha256=plan_sha256, limits=limits, cloud=cloud)


def forbidden_sdk(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Refusal must precede any SDK call")

    monkeypatch.setattr(serve, "_sdk", forbidden)
    monkeypatch.setattr(serve, "create_staging_app", forbidden)
    monkeypatch.setattr(serve, "create_app", forbidden)
    monkeypatch.setattr(serve, "create_expiry_app", forbidden)


# Serve mode


def test_serve_arms_the_stage_lifetime_and_the_controller_stops_the_app_at_that_deadline(fixture, monkeypatch):
    cloud, receipt_path = fixture.cloud, fixture.tmp / "serve-receipt.json"
    start = 1_800_000_000.0
    clock = {"now": start}
    monkeypatch.setattr(serve.time, "time", lambda: clock["now"])
    receipt = asyncio.run(route.serve_app(GEMMA, fixture.limits("deployment", max_seconds=300), receipt_path,
                                          sdk=cloud))
    assert receipt["status"] == "serving" and "error" not in receipt
    assert receipt["gpu_started_at_unix_s"] == start
    # The stage limit bounds the lifetime, never the manifest's 86,142 s GPU phase.
    assert receipt["deadline_unix_s"] == receipt["lifecycle"]["deadline_unix_s"] == start + 300
    assert cloud.spawned[0][:2] == (APP_ID, start + 300) and len(cloud.spawned) == 1
    assert fixture.plan.limits.gpu_phase_seconds == 86142 and receipt["lifetime_seconds"] == 300
    assert receipt["endpoint_url"] == URL + "/v1" and receipt["chat_completions_url"] == URL + "/v1/chat/completions"
    assert (receipt["endpoint_env"], receipt["credential_env"]) == ("MODAL_ENDPOINT_URL", "MODAL_PROXY_BEARER")
    assert receipt["plan_sha256"] == fixture.plan_sha256 and receipt["app_id"] == APP_ID
    assert receipt["stage_limits"]["stage"] == "deployment"
    assert {name for name, _ in cloud.volumes} == {"recall-guard-gemma-3-4b-weights"} == {receipt["volume"]}
    assert [event for event in cloud.events if event[0].startswith(("create", "deploy"))] == [
        ("create_expiry", GEMMA_MANIFEST), ("deploy", EXPIRY_ID), ("create_app", GEMMA_MANIFEST), ("deploy", APP_ID)]
    # The controller's own timeout follows the stage lifetime too.
    assert cloud.expiry_lifetimes == [300]
    # The serve mode exits leaving the app reachable; only the controller stops it.
    assert cloud.state == {APP_ID: "deployed", EXPIRY_ID: "deployed"}
    assert json.loads(receipt_path.read_text()) == receipt

    app_id, deadline, arm_id = cloud.spawned[0]
    del cloud.store[(app_id, "expiry")]  # The controller below writes its own acknowledgment.
    sleeps = []

    async def sleep(seconds):
        sleeps.append(seconds)
        clock["now"] += seconds

    monkeypatch.setattr(serve.asyncio, "sleep", sleep)
    expiry = asyncio.run(serve.expire_experiment(fixture.plan, app_id, deadline, arm_id, sdk=cloud))
    assert sleeps == [300] and clock["now"] == start + 300
    assert expiry["cleanup"] == "verified_stopped" and cloud.state[APP_ID] == "stopped"

    # After the auto-stop the app no longer resolves by name; the controller's record proves the stop.
    stopped = asyncio.run(route.stop_serve(GEMMA, receipt_path, sdk=cloud))
    assert stopped["cleanup"] == stopped["stop"]["cleanup"] == "verified_stopped"
    assert stopped["stop"]["app_stopped_by"] == "expiry_controller"
    assert stopped["stop"]["expiry"]["app_id"] == EXPIRY_ID
    assert set(cloud.state.values()) == {"stopped"}  # No running app remains.
    assert json.loads(receipt_path.read_text())["cleanup"] == "verified_stopped"


def test_stop_serve_stops_the_app_then_its_controller_and_reports_no_running_app(fixture):
    cloud, receipt_path = fixture.cloud, fixture.tmp / "serve-receipt.json"
    assert asyncio.run(route.serve_app(GEMMA, fixture.limits("full_run"), receipt_path, sdk=cloud))["status"] == "serving"
    receipt = asyncio.run(route.stop_serve(GEMMA, receipt_path, sdk=cloud))
    assert receipt["cleanup"] == "verified_stopped" and receipt["status"] == "stopped"
    assert receipt["stop"]["app"]["cleanup"] == receipt["stop"]["expiry"]["cleanup"] == "verified_stopped"
    assert receipt["stop"]["app"]["retained_cache"] == "recall-guard-gemma-3-4b-weights"
    assert "app_stopped_by" not in receipt["stop"]
    stops = [event for event in cloud.events if event[0] == "stop"]
    assert stops == [("stop", APP_ID), ("stop", EXPIRY_ID)]
    assert set(cloud.state.values()) == {"stopped"}


def test_unverified_app_stop_keeps_the_controller_armed(fixture, monkeypatch):
    cloud, receipt_path = fixture.cloud, fixture.tmp / "serve-receipt.json"
    asyncio.run(route.serve_app(GEMMA, fixture.limits("calibration"), receipt_path, sdk=cloud))

    async def never_stops(app_id):
        cloud.events.append(("stop", app_id))

    async def instant(seconds):
        pass

    cloud.stop = never_stops
    monkeypatch.setattr(serve.asyncio, "sleep", instant)
    receipt = asyncio.run(route.stop_serve(GEMMA, receipt_path, sdk=cloud))
    assert receipt["cleanup"] == receipt["stop"]["cleanup"] == "unverified" and receipt["status"] == "serving"
    assert "expiry" not in receipt["stop"] and cloud.state[EXPIRY_ID] == "deployed"


@pytest.mark.parametrize("stage", ["arming", "endpoint"])
def test_failed_serve_stops_the_app_and_its_controller(fixture, monkeypatch, stage):
    cloud, receipt_path = fixture.cloud, fixture.tmp / "serve-receipt.json"
    if stage == "arming":
        async def no_ack(*args):
            return SimpleNamespace(object_id="fc-no-ack")

        cloud.Function.from_name = lambda *a, **k: SimpleNamespace(spawn=aio(no_ack))
        monkeypatch.setattr(serve, "ARM_TIMEOUT_SECONDS", 0.01)
    else:
        cloud.url = "https://elsewhere.example.org"
    receipt = asyncio.run(route.serve_app(GEMMA, fixture.limits("smoke_run"), receipt_path, sdk=cloud))
    assert receipt["status"] == "stopped" and receipt["cleanup"] == "verified_stopped"
    assert receipt["error_stage"] == {"arming": "expiry_arming", "endpoint": "endpoint_check"}[stage]
    assert receipt["lifecycle"]["cleanup"]["cleanup"] == "verified_stopped"
    assert receipt["lifecycle"]["expiry_cleanup"]["cleanup"] == "verified_stopped"
    assert receipt["lifecycle"]["admission_open"] is False and "endpoint_url" not in receipt
    assert "unarmed_app_left_deployed" not in receipt["lifecycle"]
    assert set(cloud.state.values()) == {"stopped"}


def never_verifies(cloud, monkeypatch):
    """The serving app's stop is never observed; the controller still stops normally."""
    stop = cloud.stop

    async def serving_never_stops(app_id):
        if app_id != APP_ID:
            return await stop(app_id)
        cloud.events.append(("stop", app_id))

    async def instant(seconds):
        pass

    cloud.stop = serving_never_stops
    monkeypatch.setattr(serve.asyncio, "sleep", instant)


def test_failed_arming_with_an_unverified_stop_reports_the_unarmed_app_left_deployed(fixture, monkeypatch, capsys):
    cloud, receipt_path = fixture.cloud, fixture.tmp / "serve-receipt.json"

    async def store_unavailable():
        raise RuntimeError("store unavailable")

    cloud.hydrate = store_unavailable  # Arming fails before any controller input is spawned.
    never_verifies(cloud, monkeypatch)
    monkeypatch.setattr(serve, "_sdk", lambda sdk=None: cloud)
    code = route.main(["serve", "--manifest", str(GEMMA), "--stage-limits", str(fixture.limits("full_run")),
                       "--receipt", str(receipt_path)])
    summary = json.loads(capsys.readouterr().out)
    receipt = json.loads(receipt_path.read_text())
    lifecycle = receipt["lifecycle"]
    assert code == 1 and receipt["error_stage"] == "expiry_arming"
    assert receipt["status"] == receipt["cleanup"] == "unverified"
    assert "expiry_call_id" not in lifecycle and not cloud.spawned
    assert lifecycle["unarmed_app_left_deployed"] is True and lifecycle["cleanup"]["cleanup"] == "unverified"
    recovery = f"sh scripts/modal_operator.sh stop {APP_ID}"
    assert lifecycle["cleanup"]["recovery_command"] == receipt["recovery_command"] == recovery
    # The idle controller holds no armed input, so it is stopped; the unarmed app stays deployed.
    assert lifecycle["expiry_cleanup"]["cleanup"] == "verified_stopped"
    assert cloud.state == {APP_ID: "deployed", EXPIRY_ID: "stopped"}
    assert (summary["status"], summary["cleanup"], summary["recovery_command"],
            summary["unarmed_app_left_deployed"]) == ("unverified", "unverified", recovery, True)


@pytest.mark.parametrize("stop_verified", [True, False])
def test_interrupt_during_deploy_after_creation_reports_the_actual_cleanup(fixture, monkeypatch, stop_verified):
    cloud, receipt_path = fixture.cloud, fixture.tmp / "serve-receipt.json"

    async def interrupted_deploy(*, environment_name):
        cloud.state[APP_ID] = "deployed"  # The provider created the app before the interrupt.
        raise KeyboardInterrupt

    cloud.serving_app.deploy = aio(interrupted_deploy)
    if not stop_verified:
        never_verifies(cloud, monkeypatch)
    receipt = asyncio.run(route.serve_app(GEMMA, fixture.limits("full_run"), receipt_path, sdk=cloud))
    assert receipt["error"] == "KeyboardInterrupt" and receipt["error_stage"] == "model_deploy"
    assert receipt["app_id"] == APP_ID and "expiry_call_id" not in receipt["lifecycle"]
    assert receipt["lifecycle"]["expiry_cleanup"]["cleanup"] == "verified_stopped"
    if stop_verified:
        assert receipt["status"] == "stopped" and receipt["cleanup"] == "verified_stopped"
        assert "unarmed_app_left_deployed" not in receipt["lifecycle"]
        assert set(cloud.state.values()) == {"stopped"}
    else:
        assert receipt["status"] == receipt["cleanup"] == "unverified"
        assert receipt["lifecycle"]["unarmed_app_left_deployed"] is True
        assert cloud.state == {APP_ID: "deployed", EXPIRY_ID: "stopped"}
    assert json.loads(receipt_path.read_text()) == receipt


def test_failed_serve_before_any_deployment_is_stopped_with_nothing_left(fixture):
    cloud, receipt_path = fixture.cloud, fixture.tmp / "serve-receipt.json"
    cloud.shards = ()  # The weight cache inventory fails before anything is deployed.
    receipt = asyncio.run(route.serve_app(GEMMA, fixture.limits("full_run"), receipt_path, sdk=cloud))
    assert receipt["error_stage"] == "cache_inventory" and "app_id" not in receipt
    assert receipt["status"] == "stopped" and receipt["cleanup"] == "nothing_deployed"
    assert cloud.state == {} and "expiry_cleanup" not in receipt["lifecycle"]


def serve_then_expire(fixture, monkeypatch):
    """Serve, then let the controller stop the app at its deadline; returns the receipt path."""
    cloud, receipt_path = fixture.cloud, fixture.tmp / "serve-receipt.json"
    clock = {"now": 1_800_000_000.0}
    monkeypatch.setattr(serve.time, "time", lambda: clock["now"])
    asyncio.run(route.serve_app(GEMMA, fixture.limits("full_run"), receipt_path, sdk=cloud))
    app_id, deadline, arm_id = cloud.spawned[0]
    del cloud.store[(app_id, "expiry")]

    async def sleep(seconds):
        clock["now"] += seconds

    monkeypatch.setattr(serve.asyncio, "sleep", sleep)
    asyncio.run(serve.expire_experiment(fixture.plan, app_id, deadline, arm_id, sdk=cloud))
    assert cloud.state[APP_ID] == "stopped" and cloud.state[EXPIRY_ID] == "deployed"
    return receipt_path


def test_stop_serve_accepts_only_the_controller_record_of_this_lifecycle(fixture, monkeypatch):
    cloud = fixture.cloud
    receipt_path = serve_then_expire(fixture, monkeypatch)
    # A verified record from another lifecycle of the same app ID never proves this one stopped.
    cloud.store[(APP_ID, "expiry_cleanup")]["deadline_unix_s"] += 1
    receipt = asyncio.run(route.stop_serve(GEMMA, receipt_path, sdk=cloud))
    assert receipt["cleanup"] == receipt["stop"]["cleanup"] == "unverified"
    assert "app_stopped_by" not in receipt["stop"] and "expiry" not in receipt["stop"]
    assert cloud.state[EXPIRY_ID] == "deployed"


def test_an_old_receipt_never_disarms_a_newer_lifecycle(fixture, monkeypatch):
    cloud = fixture.cloud
    receipt_path = serve_then_expire(fixture, monkeypatch)
    # A newer serve redeployed the app under the same name; its controller is the same app name.
    newer = "ap-" + "b" * 22
    cloud.names[fixture.plan.target.app] = newer
    cloud.state[newer] = "deployed"
    receipt = asyncio.run(route.stop_serve(GEMMA, receipt_path, sdk=cloud))
    assert receipt["cleanup"] == "unverified" and receipt["stop"]["app"]["error"] == "ValueError"
    assert "app_stopped_by" not in receipt["stop"] and "expiry" not in receipt["stop"]
    assert cloud.state[newer] == cloud.state[EXPIRY_ID] == "deployed"


def test_serve_refuses_staging_limits_and_limits_above_the_ceiling_before_any_sdk_call(fixture, monkeypatch):
    forbidden_sdk(monkeypatch)
    receipt_path = fixture.tmp / "serve-receipt.json"
    with pytest.raises(ValueError, match="not weight_staging"):
        asyncio.run(route.serve_app(GEMMA, fixture.limits("weight_staging"), receipt_path))
    with pytest.raises(ValueError, match="max_seconds exceeds"):
        asyncio.run(route.serve_app(GEMMA, fixture.limits("full_run", max_seconds=86143), receipt_path))
    other = fixture.tmp / "other-plan.json"
    other.write_text(json.dumps(json.loads(fixture.limits("full_run").read_text()) | {"plan_sha256": "X" * 64}))
    with pytest.raises(ValueError, match="plan_sha256"):
        asyncio.run(route.serve_app(GEMMA, other, receipt_path))
    assert not receipt_path.exists()


def test_serve_never_reuses_a_receipt(fixture):
    receipt_path = fixture.tmp / "serve-receipt.json"
    receipt_path.write_text("{}")
    with pytest.raises(FileExistsError):
        asyncio.run(route.serve_app(GEMMA, fixture.limits("full_run"), receipt_path, sdk=fixture.cloud))
    assert fixture.cloud.events == []


@pytest.mark.parametrize("echo", ["deploy_error", "web_url"])
def test_serve_receipt_and_output_hold_no_token_when_the_sdk_echoes_one(fixture, monkeypatch, capsys, echo):
    cloud, receipt_path = fixture.cloud, fixture.tmp / "serve-receipt.json"
    if echo == "deploy_error":
        async def leaky_deploy(**kwargs):
            raise RuntimeError("deploy failed for " + SECRETS["MODAL_TOKEN_SECRET"] + " " + SECRETS["MODAL_PROXY_BEARER"])

        cloud.serving_app.deploy = aio(leaky_deploy)
    else:
        cloud.url = URL + "/?token=" + SECRETS["MODAL_TOKEN_ID"] + "&hf=" + SECRETS["HF_TOKEN"]
    monkeypatch.setattr(serve, "_sdk", lambda sdk=None: cloud)
    code = route.main(["serve", "--manifest", str(GEMMA), "--stage-limits", str(fixture.limits("full_run")),
                       "--receipt", str(receipt_path)])
    output = capsys.readouterr()
    receipt = json.loads(receipt_path.read_text())
    # The failed deploy left an app ID whose stop cannot be observed, so it reports unverified.
    assert code == 1 and receipt["status"] == {"deploy_error": "unverified", "web_url": "stopped"}[echo]
    assert receipt["error_stage"] == {"deploy_error": "model_deploy", "web_url": "endpoint_check"}[echo]
    assert_no_token(receipt_path.read_text() + output.out + output.err)


# Staging mode


class StagingCloud(FakeModal):
    """Adds the ephemeral staging run; its remote result is configurable."""

    def __init__(self, plan, result):
        super().__init__(plan)
        self.result = result
        self.recipe = None

    def staging_app(self, *, max_seconds, sdk, manifest):
        self.recipe = {"max_seconds": max_seconds, "manifest": manifest}

        @asynccontextmanager
        async def run(*, environment_name):
            self.events.append(("run", environment_name))
            yield
            self.events.append(("run_exit",))

        async def remote():
            self.events.append(("remote",))
            if isinstance(self.result, BaseException):
                raise self.result
            return self.result

        return SimpleNamespace(run=run), SimpleNamespace(remote=aio(remote))


def staged_result(plan, **extra):
    return {"volume": plan.volume_name, "model_revision": plan.artifacts.model_revision,
            "tokenizer_revision": plan.artifacts.tokenizer_revision, "files": 9, "bytes": 8_600_311_000,
            "duration_seconds": 412.5} | extra


def test_staging_runs_within_its_limit_and_records_volume_revision_bytes_and_duration(fixture, monkeypatch):
    cloud = StagingCloud(fixture.plan, staged_result(fixture.plan))
    monkeypatch.setattr(serve, "create_staging_app", cloud.staging_app)
    receipt_path = fixture.tmp / "stage-receipt.json"
    limits = fixture.limits("weight_staging", max_seconds=3600, max_usd=0.32)
    receipt = asyncio.run(route.stage(GEMMA, limits, receipt_path, sdk=cloud))
    assert receipt["status"] == "staged" and "error" not in receipt
    assert cloud.recipe == {"max_seconds": 3600, "manifest": GEMMA_MANIFEST}
    assert [event[0] for event in cloud.events] == ["run", "remote", "run_exit"]
    assert receipt["staged"] == staged_result(fixture.plan)
    assert receipt["volume"] == receipt["staged"]["volume"] == "recall-guard-gemma-3-4b-weights"
    assert receipt["weight_cache"]["status"] == "verified_inventory"
    assert {name for name, _ in cloud.volumes} == {"recall-guard-gemma-3-4b-weights"}
    assert receipt["worst_case_usd"] == pytest.approx(0.316512) and receipt["timeout_seconds"] == 3600
    assert receipt["resources"] == {"gpu": None, "cpu_cores": 4, "memory_gib": 16}
    assert receipt["secret_env"] == ["HF_TOKEN"] and receipt["plan_sha256"] == fixture.plan_sha256
    assert receipt["stage_limits_sha256"] == hashlib.sha256(limits.read_bytes()).hexdigest()
    assert receipt["manifest_sha256"] == hashlib.sha256(GEMMA.read_bytes()).hexdigest()
    assert receipt["finished_at_unix_s"] >= receipt["started_at_unix_s"]
    assert_no_token(receipt_path.read_text())


@pytest.mark.parametrize("limits", [
    {"max_seconds": 3600, "max_usd": 0.31},  # Below 4 cores and 16 GiB at the manifest rates for 3,600 s.
    {"max_seconds": 86143, "max_usd": 34},  # Above the manifest's GPU phase ceiling.
    {"stage": "deployment"},  # A serving stage's limits.
])
def test_staging_refuses_above_its_input_limit_before_any_sdk_call(fixture, monkeypatch, capsys, limits):
    forbidden_sdk(monkeypatch)
    stage = limits.pop("stage", "weight_staging")
    path = fixture.limits(stage, **limits)
    receipt_path = fixture.tmp / "stage-receipt.json"
    code = route.main(["stage", "--manifest", str(GEMMA), "--stage-limits", str(path), "--receipt", str(receipt_path)])
    output = capsys.readouterr()
    assert code == 2 and "Modal stage refused" in output.err
    assert not receipt_path.exists()
    assert_no_token(output.out + output.err)


@pytest.mark.parametrize("echo", ["result", "error"])
def test_staging_receipt_holds_no_token_when_the_sdk_echoes_one(fixture, monkeypatch, capsys, echo):
    result = (staged_result(fixture.plan, token=SECRETS["HF_TOKEN"], header="Bearer " + SECRETS["MODAL_PROXY_BEARER"])
              if echo == "result" else RuntimeError("401 for " + SECRETS["HF_TOKEN"] + " " + SECRETS["MODAL_TOKEN_ID"]))
    cloud = StagingCloud(fixture.plan, result)
    monkeypatch.setattr(serve, "create_staging_app", cloud.staging_app)
    monkeypatch.setattr(serve, "_sdk", lambda sdk=None: cloud)
    receipt_path = fixture.tmp / "stage-receipt.json"
    code = route.main(["stage", "--manifest", str(GEMMA), "--stage-limits",
                       str(fixture.limits("weight_staging", max_seconds=3600, max_usd=0.32)),
                       "--receipt", str(receipt_path)])
    output = capsys.readouterr()
    receipt = json.loads(receipt_path.read_text())
    assert receipt["status"] == ("staged" if echo == "result" else "stopped")
    assert code == (0 if echo == "result" else 1)
    assert_no_token(receipt_path.read_text() + output.out + output.err)


@pytest.mark.parametrize("change", [{"volume": "recall-guard-gpt-oss-20b-weights"}, {"model_revision": "0" * 40},
                                    {"bytes": 0}, {"duration_seconds": float("nan")}])
def test_staging_refuses_a_result_for_another_volume_revision_or_without_sizes(fixture, monkeypatch, change):
    cloud = StagingCloud(fixture.plan, staged_result(fixture.plan) | change)
    monkeypatch.setattr(serve, "create_staging_app", cloud.staging_app)
    receipt = asyncio.run(route.stage(GEMMA, fixture.limits("weight_staging", max_seconds=3600, max_usd=0.32),
                                      fixture.tmp / "stage-receipt.json", sdk=cloud))
    assert receipt["status"] == "stopped" and receipt["error_stage"] == "staging" and "staged" not in receipt


def test_cli_requires_stage_limits_except_for_stop_serve(capsys):
    with pytest.raises(SystemExit):
        route.main(["serve", "--manifest", str(GEMMA), "--receipt", "r.json"])
    with pytest.raises(SystemExit):
        route.main(["stop-serve", "--manifest", str(GEMMA), "--stage-limits", "l.json", "--receipt", "r.json"])
    assert "stop-serve takes none" in capsys.readouterr().err


def test_stop_serve_refuses_a_receipt_of_another_mode_or_manifest(fixture):
    receipt_path = fixture.tmp / "stage-receipt.json"
    receipt_path.write_text(json.dumps({"mode": "stage", "manifest": GEMMA_MANIFEST}))
    with pytest.raises(ValueError, match="serve receipt"):
        asyncio.run(route.stop_serve(GEMMA, receipt_path, sdk=fixture.cloud))
    receipt_path.write_text(json.dumps({"mode": "serve", "manifest": "configs/modal/gpt-oss-20b-pilot.json"}))
    with pytest.raises(ValueError, match="serve receipt"):
        asyncio.run(route.stop_serve(GEMMA, receipt_path, sdk=fixture.cloud))
    assert fixture.cloud.events == []
