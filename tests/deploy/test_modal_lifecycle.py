"""Offline lifecycle checks: no credentials, cloud calls or inference."""

import asyncio
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from deploy.modal import serve
from deploy.modal.manifest import load_plan

PLAN = load_plan(serve.ROOT / "configs/modal/gpt-oss-20b-pilot.json")


def aio(fn):
    return SimpleNamespace(aio=fn)


class Control:
    def __init__(self):
        self.events = []
        self.store = {}
        self.state = "deployed"
        self.app_id = "ap-aaaaaaaaaaaaaaaaaaaaaa"
        self.App = SimpleNamespace(lookup=aio(self.lookup))
        self.Dict = SimpleNamespace(from_name=lambda *a, **k: SimpleNamespace(get=aio(self.get), put=aio(self.put)))
        self.Function = SimpleNamespace(from_name=lambda *a, **k: SimpleNamespace(spawn=aio(self.spawn)))

    async def lookup(self, name, **kwargs):
        assert name == PLAN.target.app
        assert kwargs == {"environment_name": PLAN.target.environment, "create_if_missing": False}
        self.events.append("lookup")
        return SimpleNamespace(app_id=self.app_id, stop=aio(self.stop), info=aio(self.info))

    async def stop(self):
        self.events.append("stop")
        self.state = "stopped"

    async def info(self, refresh):
        assert refresh
        self.events.append("info")
        return SimpleNamespace(lifecycle=SimpleNamespace(state=SimpleNamespace(value=self.state)))

    async def get(self, key):
        return self.store.get(key)

    async def put(self, key, value, skip_if_exists=False):
        if skip_if_exists and key in self.store:
            return False
        self.store[key] = value
        return True

    async def spawn(self, app_id, deadline_unix_s, arm_id):
        self.events.append("spawn")
        self.store[(app_id, "expiry")] = {"app_id": app_id, "arm_id": arm_id, "deadline_unix_s": deadline_unix_s}
        return SimpleNamespace(object_id="fc-independent")


def test_stop_verifies_only_owned_app_and_keeps_storage():
    sdk = Control()
    result = asyncio.run(serve.stop_experiment(PLAN, "ap-aaaaaaaaaaaaaaaaaaaaaa", sdk=sdk))
    assert result["cleanup"] == "verified_stopped"
    assert sdk.events == ["lookup", "stop", "info"]
    assert result["retained_cache"] == "recall-guard-gpt-oss-20b-weights"
    assert result["observed_usd"] is None
    assert result["category"] == "control"


def test_wrong_app_is_never_stopped():
    sdk = Control()
    sdk.app_id = "ap-bbbbbbbbbbbbbbbbbbbbbb"
    result = asyncio.run(serve.stop_experiment(PLAN, "ap-aaaaaaaaaaaaaaaaaaaaaa", sdk=sdk))
    assert result["cleanup"] == "unverified"
    assert "stop" not in sdk.events
    assert result["app_id"] == "ap-aaaaaaaaaaaaaaaaaaaaaa"
    assert result["recovery_command"] == "uv run --env-file .env --project deploy/modal --no-sync python -m modal app stop ap-aaaaaaaaaaaaaaaaaaaaaa --yes"


def test_control_plane_failure_and_timeout_are_bounded(monkeypatch):
    sdk = Control()

    async def hang(*a, **k):
        await asyncio.Event().wait()

    sdk.App.lookup = aio(hang)
    monkeypatch.setattr(serve, "CONTROL_TIMEOUT_SECONDS", 0.001)
    result = asyncio.run(serve.stop_experiment(PLAN, "ap-aaaaaaaaaaaaaaaaaaaaaa", sdk=sdk))
    assert result["cleanup"] == "unverified"
    assert result["error"] == "TimeoutError"
    assert result["control_calls"] == 1


@pytest.mark.parametrize("failure", [None, RuntimeError, asyncio.CancelledError])
def test_lifecycle_arms_before_admission_and_cleans_every_exit(failure):
    sdk = Control()

    async def run():
        context = pytest.raises(failure) if failure else nullcontext()
        with context:
            async with serve.experiment_lifecycle(PLAN, "ap-aaaaaaaaaaaaaaaaaaaaaa", gpu_started_at_unix_s=serve.time.time(), sdk=sdk) as lease:
                assert lease["expiry_call_id"] == "fc-independent"
                assert sdk.events == ["spawn"]
                if failure:
                    raise failure()
        assert "stop" in sdk.events

    asyncio.run(run())


def test_missing_arm_ack_fails_closed_and_stops(monkeypatch):
    sdk = Control()

    async def no_ack(*a):
        return SimpleNamespace(object_id="fc-no-ack")

    sdk.Function.from_name = lambda *a, **k: SimpleNamespace(spawn=aio(no_ack))
    monkeypatch.setattr(serve, "ARM_TIMEOUT_SECONDS", 0.001)

    async def run():
        receipt = {}
        with pytest.raises(RuntimeError, match="expiry"):
            async with serve.experiment_lifecycle(PLAN, "ap-aaaaaaaaaaaaaaaaaaaaaa", gpu_started_at_unix_s=serve.time.time(), receipt=receipt, sdk=sdk):
                pytest.fail("inference admitted without remote expiry acknowledgment")
        assert "stop" in sdk.events
        assert receipt["cleanup"]["app_id"] == "ap-aaaaaaaaaaaaaaaaaaaaaa"
        assert receipt["admission_open"] is False

    asyncio.run(run())


def test_remote_expiry_survives_absent_client_and_never_extends_deadline(monkeypatch):
    sdk = Control()
    now = serve.time.time()
    monkeypatch.setattr(serve.time, "time", lambda: now)
    sleeps = []
    clock = [0]
    monkeypatch.setattr(serve.time, "monotonic", lambda: clock[0])

    async def sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(serve.asyncio, "sleep", sleep)
    receipt = asyncio.run(serve.expire_experiment(PLAN, "ap-aaaaaaaaaaaaaaaaaaaaaa", now + 1200, "arm-token", sdk=sdk))
    assert sleeps == [1200]
    assert receipt["cleanup"] == "verified_stopped"
    assert receipt["category"] == "expiry"
    assert receipt["estimated_usd"] > 0
    assert receipt["observed_usd"] is None
    assert "stop" in sdk.events
    with pytest.raises(RuntimeError, match="already"):
        asyncio.run(serve.expire_experiment(PLAN, "ap-aaaaaaaaaaaaaaaaaaaaaa", now + 1200, "other-token", sdk=sdk))


@pytest.mark.parametrize("delta", [1201, float("nan"), float("inf")])
def test_expiry_rejects_invalid_deadline_before_control_calls(delta):
    sdk = Control()
    with pytest.raises(ValueError):
        asyncio.run(serve.expire_experiment(PLAN, "ap-aaaaaaaaaaaaaaaaaaaaaa", serve.time.time() + delta, "token", sdk=sdk))
    assert sdk.events == []


def test_late_controller_stops_immediately_without_opening_admission():
    sdk = Control()
    receipt = asyncio.run(serve.expire_experiment(PLAN, "ap-aaaaaaaaaaaaaaaaaaaaaa", serve.time.time() - 1, "token", sdk=sdk))
    assert receipt["cleanup"] == "verified_stopped"
    assert ("ap-aaaaaaaaaaaaaaaaaaaaaa", "expiry") not in sdk.store


def test_remote_controller_rejects_wrong_app_before_ack():
    sdk = Control()
    sdk.app_id = "ap-bbbbbbbbbbbbbbbbbbbbbb"
    with pytest.raises(ValueError, match="identity"):
        asyncio.run(serve.expire_experiment(PLAN, "ap-aaaaaaaaaaaaaaaaaaaaaa", serve.time.time() + 1200, "token", sdk=sdk))
    assert sdk.store == {}
    assert sdk.events == ["lookup"]


@pytest.mark.parametrize("record", [None, {}, {"app_id": "ap-other"}])
def test_gpu_startup_requires_remote_ack(record):
    sdk = Control()
    sdk.store[("ap-aaaaaaaaaaaaaaaaaaaaaa", "expiry")] = record
    with pytest.raises(RuntimeError, match="expiry"):
        serve.require_remote_expiry(PLAN, "ap-aaaaaaaaaaaaaaaaaaaaaa", sdk.Dict.from_name())


def test_gpu_startup_accepts_live_ack_only():
    sdk = Control()
    record = {"app_id": "ap-aaaaaaaaaaaaaaaaaaaaaa", "arm_id": "token", "deadline_unix_s": serve.time.time() + 1100}
    sdk.store[("ap-aaaaaaaaaaaaaaaaaaaaaa", "expiry")] = record
    serve.require_remote_expiry(PLAN, "ap-aaaaaaaaaaaaaaaaaaaaaa", sdk.Dict.from_name())
    record["deadline_unix_s"] = serve.time.time() - 1
    with pytest.raises(RuntimeError, match="expiry"):
        serve.require_remote_expiry(PLAN, "ap-aaaaaaaaaaaaaaaaaaaaaa", sdk.Dict.from_name())


def test_stop_verification_waits_at_most_two_minutes(monkeypatch):
    sdk = Control()
    clock = [0]

    async def stop():
        sdk.events.append("stop")
        sdk.state = "stopping"

    async def sleep(seconds):
        clock[0] += seconds

    sdk.stop = stop
    monkeypatch.setattr(serve.asyncio, "sleep", sleep)
    monkeypatch.setattr(serve.time, "monotonic", lambda: clock[0])
    receipt = asyncio.run(serve.stop_experiment(PLAN, "ap-aaaaaaaaaaaaaaaaaaaaaa", sdk=sdk))
    assert receipt["cleanup"] == "unverified"
    assert receipt["elapsed_seconds"] == 120
    assert receipt["control_calls"] == 26


def test_cpu_recipe_has_no_gpu_or_cache_and_bounded_runtime():
    options = {}

    def function(**kwargs):
        options.update(kwargs)
        return lambda fn: fn

    image = SimpleNamespace(add_local_file=lambda *a, **k: image)
    sdk = SimpleNamespace(
        App=lambda name: SimpleNamespace(name=name, function=function),
        Image=SimpleNamespace(debian_slim=lambda **k: image),
        concurrent=lambda **k: lambda fn: fn,
    )
    app = serve.create_expiry_app(sdk=sdk)
    assert app.name == PLAN.target.app + "-expiry"
    assert options["gpu"] is None
    assert "volumes" not in options
    assert options["cpu"] == 0.125
    assert options["memory"] == 128
    assert options["timeout"] == 1350
    assert options["retries"] == options["min_containers"] == options["buffer_containers"] == 0
    assert options["max_containers"] == 1
