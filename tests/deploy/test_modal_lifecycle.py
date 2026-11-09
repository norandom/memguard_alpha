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
        self.expiry_app_id = "ap-cccccccccccccccccccccc"
        self.expiry_state = "deployed"
        self.dict_options = []
        self.App = SimpleNamespace(lookup=aio(self.lookup))
        self.Dict = SimpleNamespace(from_name=self.dictionary)
        self.Function = SimpleNamespace(from_name=lambda *a, **k: SimpleNamespace(spawn=aio(self.spawn)))

    def dictionary(self, *args, **kwargs):
        self.dict_options.append(kwargs)
        return SimpleNamespace(get=aio(self.get), put=aio(self.put), hydrate=aio(self.hydrate))

    async def hydrate(self):
        self.events.append("hydrate")

    async def lookup(self, name, **kwargs):
        assert kwargs == {"environment_name": PLAN.target.environment, "create_if_missing": False}
        if name == PLAN.target.app + "-expiry":
            self.events.append("lookup_expiry")
            return SimpleNamespace(app_id=self.expiry_app_id, stop=aio(self.stop_expiry), info=aio(self.info_expiry))
        assert name == PLAN.target.app
        self.events.append("lookup")
        return SimpleNamespace(app_id=self.app_id, stop=aio(self.stop), info=aio(self.info))

    async def stop_expiry(self):
        self.events.append("stop_expiry")
        self.expiry_state = "stopped"

    async def info_expiry(self, refresh):
        return SimpleNamespace(lifecycle=SimpleNamespace(state=SimpleNamespace(value=self.expiry_state)))

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
    assert result["recovery_command"] == "sh scripts/modal_operator.sh stop ap-aaaaaaaaaaaaaaaaaaaaaa"


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
                assert sdk.events == ["hydrate", "spawn"]
                assert sdk.dict_options == [
                    {"environment_name": PLAN.target.environment, "create_if_missing": True},
                    {"environment_name": PLAN.target.environment, "create_if_missing": False},
                ]
                assert lease["arming"]["control_calls"] == 3
                if failure:
                    raise failure()
        assert "stop" in sdk.events

    asyncio.run(run())


@pytest.mark.parametrize("timeout", [False, True])
def test_store_creation_failure_blocks_spawn_and_admission_and_cleans(monkeypatch, timeout):
    sdk = Control()

    async def fail_creation():
        if timeout:
            await asyncio.Event().wait()
        raise RuntimeError("store unavailable")

    sdk.hydrate = fail_creation
    monkeypatch.setattr(serve, "CONTROL_TIMEOUT_SECONDS", 0.001)

    async def run():
        lease = {}
        with pytest.raises(RuntimeError):
            async with serve.experiment_lifecycle(
                PLAN, sdk.app_id, gpu_started_at_unix_s=serve.time.time(), receipt=lease, sdk=sdk
            ):
                pytest.fail("Admission opened before provenance store creation")
        assert "spawn" not in sdk.events
        assert "stop" in sdk.events
        assert lease["admission_open"] is False
        assert lease["arming"]["control_calls"] == 1

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


def test_cpu_recipe_has_no_gpu_or_cache_and_bounded_runtime(monkeypatch):
    monkeypatch.setattr(serve, "require_operator_python", lambda: None)
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


def expiry_recipe(monkeypatch, apps=None, **kwargs):
    monkeypatch.setattr(serve, "require_operator_python", lambda: None)
    options, apps = {}, [] if apps is None else apps

    def function(**values):
        options.update(values)
        return lambda fn: fn

    image = SimpleNamespace(add_local_file=lambda *a, **k: image)
    sdk = SimpleNamespace(
        App=lambda name: apps.append(name) or SimpleNamespace(name=name, function=function),
        Image=SimpleNamespace(debian_slim=lambda **k: image),
        concurrent=lambda **k: lambda fn: fn,
    )
    serve.create_expiry_app(sdk=sdk, **kwargs)
    return options, apps


@pytest.mark.parametrize("manifest,lifetime,timeout", [
    ("configs/modal/gpt-oss-20b-pilot.json", None, 1200 + 150),
    ("configs/modal/gpt-oss-20b-pilot.json", 300, 300 + 150),
    ("configs/modal/gemma-3-4b-measurement.json", None, 86142 + 150),
    ("configs/modal/gemma-3-4b-measurement.json", 300, 300 + 150),
])
def test_controller_timeout_follows_the_stage_lifetime_when_one_is_passed(monkeypatch, manifest, lifetime, timeout):
    options = {} if lifetime is None else {"lifetime_seconds": lifetime}
    recipe, _ = expiry_recipe(monkeypatch, manifest=manifest, **options)
    assert recipe["timeout"] == timeout and recipe["gpu"] is None


@pytest.mark.parametrize("lifetime", [0, 1201, True])
def test_controller_refuses_an_invalid_lifetime_before_creating_its_app(monkeypatch, lifetime):
    apps = []
    with pytest.raises(ValueError, match="lifetime"):
        expiry_recipe(monkeypatch, apps=apps, lifetime_seconds=lifetime)
    assert apps == []


@pytest.mark.parametrize("lifetime,expected", [(None, 1200), (300, 300), (1200, 1200)])
def test_deadline_is_the_gpu_phase_unless_a_stage_lifetime_is_passed(monkeypatch, lifetime, expected):
    sdk = Control()
    now = serve.time.time()
    monkeypatch.setattr(serve.time, "time", lambda: now)
    options = {} if lifetime is None else {"lifetime_seconds": lifetime}
    spawned = []
    original = sdk.spawn

    async def spawn(app_id, deadline_unix_s, arm_id):
        spawned.append(deadline_unix_s)
        return await original(app_id, deadline_unix_s, arm_id)

    sdk.Function.from_name = lambda *a, **k: SimpleNamespace(spawn=aio(spawn))

    async def run():
        async with serve.experiment_lifecycle(PLAN, sdk.app_id, gpu_started_at_unix_s=now, sdk=sdk,
                                              **options) as lease:
            return dict(lease)

    lease = asyncio.run(run())
    assert lease["deadline_unix_s"] == spawned[0] == now + expected
    assert lease["admission_open"] is True


@pytest.mark.parametrize("lifetime", [0, -1, 1201, True, 300.0])
def test_invalid_stage_lifetime_never_arms_and_still_cleans(lifetime):
    sdk = Control()

    async def run():
        lease = {}
        with pytest.raises(ValueError, match="lifetime"):
            async with serve.experiment_lifecycle(PLAN, sdk.app_id, gpu_started_at_unix_s=serve.time.time(),
                                                  receipt=lease, sdk=sdk, lifetime_seconds=lifetime):
                pytest.fail("Admission opened with an invalid app lifetime")
        return lease

    lease = asyncio.run(run())
    assert "spawn" not in sdk.events and "stop" in sdk.events
    assert lease["admission_open"] is False and lease["cleanup"]["cleanup"] == "verified_stopped"


def test_armed_stage_lifetime_is_where_the_remote_controller_stops_the_app(monkeypatch):
    """Arm with a 300 s stage lifetime, then run the controller from the spawned arguments."""
    sdk = Control()
    start = serve.time.time()
    clock = {"now": start}
    monkeypatch.setattr(serve.time, "time", lambda: clock["now"])
    lease = asyncio.run(serve.arm_expiry(PLAN, sdk.app_id, gpu_started_at_unix_s=start, lease={},
                                         lifetime_seconds=300, sdk=sdk))
    assert lease["deadline_unix_s"] == start + 300 and lease["admission_open"] is True
    assert "stop" not in sdk.events  # Arming alone leaves the app running.
    armed = sdk.store.pop((sdk.app_id, "expiry"))  # The fake spawn wrote the controller's own acknowledgment.

    async def sleep(seconds):
        clock["now"] += seconds

    monkeypatch.setattr(serve.asyncio, "sleep", sleep)
    receipt = asyncio.run(serve.expire_experiment(PLAN, sdk.app_id, armed["deadline_unix_s"], armed["arm_id"], sdk=sdk))
    assert clock["now"] == start + 300
    assert receipt["cleanup"] == "verified_stopped" and sdk.state == "stopped"


def test_verified_cleanup_also_stops_the_armed_expiry_controller(monkeypatch):
    monkeypatch.setattr(serve.asyncio, "sleep", lambda *_: asyncio.sleep(0))
    sdk = Control()

    async def run():
        lease = {}
        async with serve.experiment_lifecycle(PLAN, sdk.app_id, gpu_started_at_unix_s=serve.time.time(),
                                              receipt=lease, sdk=sdk):
            pass
        return lease

    lease = asyncio.run(run())
    assert lease["cleanup"]["cleanup"] == "verified_stopped"
    assert lease["expiry_cleanup"]["cleanup"] == "verified_stopped"
    assert lease["expiry_cleanup"]["app_id"] == "ap-cccccccccccccccccccccc"
    assert sdk.state == sdk.expiry_state == "stopped"
    assert sdk.events.index("stop") < sdk.events.index("stop_expiry")


def test_unverified_experiment_stop_leaves_the_expiry_controller_armed(monkeypatch):
    monkeypatch.setattr(serve.asyncio, "sleep", lambda *_: asyncio.sleep(0))
    sdk = Control()

    async def never_stops():
        sdk.events.append("stop")

    sdk.stop = never_stops

    async def run():
        lease = {}
        async with serve.experiment_lifecycle(PLAN, sdk.app_id, gpu_started_at_unix_s=serve.time.time(),
                                              receipt=lease, sdk=sdk):
            pass
        return lease

    lease = asyncio.run(run())
    assert lease["cleanup"]["cleanup"] != "verified_stopped"
    assert "expiry_cleanup" not in lease and sdk.expiry_state == "deployed"
