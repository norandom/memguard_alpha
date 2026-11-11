"""Fixtures of the integration tests: the offline guard and one recall calibration, fitted offline.

The integration tests enter through ``recall_guard.ask`` and import the
package root only. Two things live here for them:

* ``offline``: refuses and records every way out of the process to a network
  and removes credential-like variables. A test file asks for it (for every
  test with ``pytestmark = pytest.mark.usefixtures("offline")``, or per test);
  it is not applied to the whole directory, because the demo tests start a
  notebook kernel, which talks to the test over local sockets.
* ``fitted_config`` and ``fitted_scorer``: the model configuration a recall
  calibration was fitted for, and the fitted scorer. Nothing here opens a
  connection.
"""

from __future__ import annotations

import hashlib
import os
import re
import socket

import pytest

from recall_guard import MemoryGuardedScorer, ModelConfig

# Why these internal imports remain: fitting a calibrator offline needs a client that reports its
# capabilities without a request, and the capability and response types are not root names. With a real
# host none of them is needed: the root's ``MemoryGuardedScorer`` (``calibrate``) and
# ``SourceCalibrator.from_scorer`` suffice.
from recall_guard.core.completion import (
    CapabilityEvidence,
    CapabilityReport,
    CapabilityStatus,
    FeatureChoices,
    ModelResponse,
    TokenAlternative,
    TokenEvidence,
)

#: Three alternatives per token: what the calibration requests and what the tests' fixed replies carry.
SETTINGS = {"max_tokens": 512, "temperature": 0.0, "logprobs": True, "top_logprobs": 3}

#: Names of environment variables that may hold a credential; the offline guard removes them.
CREDENTIAL_LIKE = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|BEARER|CREDENTIAL", re.IGNORECASE)
#: What every refusal of the offline guard says.
OFFLINE_REFUSAL = "the offline integration tests open no socket"


@pytest.fixture
def offline(monkeypatch):
    """Refuse and record every way to a network; remove credential-like variables. Yields the record.

    Refused: creating a socket (``socket.socket``), ``socket.create_connection``, name resolution
    (``socket.getaddrinfo``), and ``connect``, ``connect_ex`` and ``sendto`` of the socket class itself, so
    that a library that bound ``socket.socket`` before this fixture ran is still caught when it connects.
    Each refusal raises OSError and is recorded; the test fails at teardown if the record is not empty,
    because a caller may swallow the error (the guard records a failed request on its draw).
    """
    attempts: list[str] = []

    def refuser(what: str):
        def refuse(*args, **kwargs):
            attempts.append(f"{what}{args!r}")
            raise OSError(OFFLINE_REFUSAL)
        return refuse

    # The class first, while the name still leads to it; then the module's names.
    for method in ("connect", "connect_ex", "sendto"):
        monkeypatch.setattr(socket.socket, method, refuser(f"socket.{method}"))
    monkeypatch.setattr(socket, "socket", refuser("socket"))
    monkeypatch.setattr(socket, "create_connection", refuser("create_connection"))
    monkeypatch.setattr(socket, "getaddrinfo", refuser("getaddrinfo"))
    for name in list(os.environ):
        if CREDENTIAL_LIKE.search(name):
            monkeypatch.delenv(name)
    yield attempts
    assert attempts == [], f"a network connection was attempted: {attempts}"


class CalibrationHost:
    """An in-process stand-in for a model host during the fit: token evidence that separates the two corpora."""

    def __init__(self, config: ModelConfig) -> None:
        self.config = config

    def generate(self, prompt, **parameters):
        base = -0.2 if prompt.startswith("is") else -5.0 if prompt.startswith("oos") else -1.0
        jitter = (int(hashlib.sha256(prompt.encode()).hexdigest(), 16) % 1000) / 20000
        evidence = tuple(TokenEvidence("x", base + jitter + i * 0.01, (
            TokenAlternative("x", base + jitter), TokenAlternative("y", base - 1), TokenAlternative("z", base + 0.1),
        )) for i in range(5))
        return ModelResponse("Direction: 1\nConfidence: 0.8", evidence,
                             requested_settings=dict(self.config.parameters) | parameters)

    def probe(self):
        self.generate("probe")
        features = FeatureChoices(mia_scoring=True)
        required = self.config.inspect(features).capabilities
        return CapabilityReport(
            {name: CapabilityEvidence(CapabilityStatus.SUPPORTED, "in-process generation", observed=True)
             for name in required}, features)


@pytest.fixture(scope="session")
def fitted_config() -> ModelConfig:
    """The model configuration the calibration was fitted for; a fixed-reply client stands for it."""
    return ModelConfig(protocol="openai_chat", endpoint="https://example.test/chat", model="m", parameters=SETTINGS)


@pytest.fixture(scope="session")
def fitted_scorer(fitted_config) -> MemoryGuardedScorer:
    """A recall calibration fitted on sixteen invented prompts, without a request."""
    return MemoryGuardedScorer.calibrate_client(
        client=CalibrationHost(fitted_config),
        is_memorized=[f"is-{i}" for i in range(8)], oos_control=[f"oos-{i}" for i in range(8)],
        min_valid=4, max_workers=1,
    )
