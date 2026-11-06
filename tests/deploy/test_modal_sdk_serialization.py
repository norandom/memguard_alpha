"""Optional pinned-SDK regressions; no network, credentials, deployment or GPU.

Run with the separate Modal 1.6.3 Python 3.12 environment and pytest available.
The core test environment skips this module when Modal is absent.
"""

# Optional imports must follow the module-level skip in the core environment.
# ruff: noqa: E402

import inspect
from types import SimpleNamespace

import pytest

modal = pytest.importorskip("modal")

from modal._image import _Image
from modal._serialization import serialize
from modal._utils.async_utils import synchronizer
from modal.app import _App

from deploy.modal import serve


@pytest.fixture(autouse=True)
def offline_sdk(monkeypatch):
    assert modal.__version__ == "1.6.3"

    def forbidden(*args, **kwargs):
        pytest.fail("Optional SDK regression attempted a network operation")

    import socket

    for name in ("connect", "connect_ex", "sendto", "bind"):
        monkeypatch.setattr(socket.socket, name, forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)


def test_serving_serializes_before_class_or_function_hydration():
    app = serve.create_app()
    internal_app = synchronizer._translate_in(app)
    cls = internal_app._local_state.classes["Serving"]
    raw = cls._get_user_cls()
    assert not cls._is_hydrated
    assert all(not fn._is_hydrated for fn in internal_app._local_state.functions.values())
    assert set(inspect.getclosurevars(raw.record).nonlocals) == {"plan"}
    assert set(inspect.getclosurevars(synchronizer._translate_in(raw.start).raw_f).nonlocals) == {"plan"}
    assert not inspect.getclosurevars(synchronizer._translate_in(raw.readiness).raw_f).nonlocals

    # Use the untouched raw class. No SDK handles may require hydration merely
    # to serialize class code; runtime Dict lookup occurs only in its enter hook.
    payload = serialize(raw)
    assert isinstance(payload, bytes) and payload
    assert not cls._is_hydrated
    assert all(not fn._is_hydrated for fn in internal_app._local_state.functions.values())


def test_serving_image_resets_inherited_entrypoint_with_real_sdk(monkeypatch):
    commands = []
    original = _Image.dockerfile_commands

    def record_commands(self, *args, **kwargs):
        commands.extend(args)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(_Image, "dockerfile_commands", record_commands)
    serve.create_app()
    assert commands.count("ENTRYPOINT []") == 1


def test_serving_image_pins_typing_extensions_with_real_sdk(monkeypatch):
    packages = []
    original = modal.Image.pip_install

    def record_packages(self, *args, **kwargs):
        packages.extend(args)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(modal.Image, "pip_install", record_packages)
    serve.create_app()
    assert packages == ["typing_extensions==4.16.0"]


def test_runtime_identity_uses_observed_container_context(monkeypatch):
    observed_id = "ap-" + "a" * 22
    monkeypatch.setattr(_App, "_container_app", SimpleNamespace(app_id=observed_id))
    assert serve.running_app_id() == observed_id


@pytest.mark.parametrize("app_id", [None, "", "reusable-app-name", "ap-short"])
def test_invalid_container_identity_fails_closed(monkeypatch, app_id):
    monkeypatch.setattr(_App, "_container_app", SimpleNamespace(app_id=app_id))
    with pytest.raises(ValueError, match="app identity"):
        serve.running_app_id()


def test_no_container_context_fails_closed(monkeypatch):
    monkeypatch.setattr(_App, "_container_app", None)
    with pytest.raises(RuntimeError, match="Missing running"):
        serve.running_app_id()


def test_changed_sdk_version_fails_before_context_lookup(monkeypatch):
    monkeypatch.setattr(modal, "__version__", "1.7.0")
    with pytest.raises(RuntimeError, match="Modal 1.6.3"):
        serve.running_app_id()
