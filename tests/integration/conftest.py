"""Fixtures of the integration tests: one recall calibration, fitted offline.

The integration tests enter through ``recall_guard.ask`` and import the
package root only. Fitting a calibration without a model host needs names the
root does not export (the configuration and capability types of
``recall_guard.core.completion``), so the fit lives here, behind two fixtures:
``fitted_config``, the model configuration the calibration was fitted for, and
``fitted_scorer``, the fitted scorer. Nothing here opens a connection.
"""

from __future__ import annotations

import hashlib

import pytest

from recall_guard import MemoryGuardedScorer
from recall_guard.core.completion import (
    CapabilityEvidence,
    CapabilityReport,
    CapabilityStatus,
    FeatureChoices,
    ModelConfig,
    ModelResponse,
    TokenAlternative,
    TokenEvidence,
)

#: Three alternatives per token: what the calibration requests and what the tests' fixed replies carry.
SETTINGS = {"max_tokens": 512, "temperature": 0.0, "logprobs": True, "top_logprobs": 3}


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
