"""The guarded-decision API works from an installed wheel alone (task 5.2, Requirement 11.6).

The test builds the wheel with ``uv build``, installs it into a fresh
environment under ``tmp_path`` and runs a script there with ``-I`` and
``cwd=tmp_path``, so neither the checkout nor its research data, model hosts or
study code is importable. The script imports the guarded-decision exports from
the package root, calibrates the existing MCS estimator on a fake configured
client (no request leaves the process), runs ``guarded_decide`` as an ensemble
and replays it with ``decide_from_draws``, and checks that no plotting or
backtest module was imported and that the installed package tree holds no
scripts, data, runs or docs folder. The test skips when ``uv`` is not on PATH;
the Linux CI image provides it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
UV = shutil.which("uv")
MARKER = "installed-wheel-ok"
#: Generous per-step limit; the whole test normally takes well under a minute with a warm uv cache.
STEP_TIMEOUT = 300

SCRIPT = r'''
"""Run the guarded-decision API from the installed wheel on fake clients only."""

import hashlib
import importlib.util
import json
import statistics
import sys
from pathlib import Path

import requests

import recall_guard
from recall_guard import (
    ComponentDistribution,
    ComponentSpec,
    DecisionDefinition,
    DecisionDefinitionError,
    DecisionParseError,
    DistributionSummary,
    DrawMode,
    EstimateSource,
    FullDistributionClient,
    FullDistributionEvidence,
    GuardConfig,
    GuardedDecision,
    GuardedRun,
    GuardIdentity,
    GuardStatus,
    MemoryGuardedScorer,
    RecordedDraw,
    SourceCalibrator,
    SourceLabel,
    adjust,
    decide_from_draws,
    guarded_decide,
    guarded_run,
    parse_json_components,
)
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

EXPORTS = {
    "guarded_decide": guarded_decide, "guarded_run": guarded_run, "decide_from_draws": decide_from_draws,
    "GuardConfig": GuardConfig, "DrawMode": DrawMode, "GuardStatus": GuardStatus, "GuardIdentity": GuardIdentity,
    "GuardedDecision": GuardedDecision, "GuardedRun": GuardedRun, "SourceCalibrator": SourceCalibrator,
    "DecisionDefinition": DecisionDefinition, "ComponentSpec": ComponentSpec,
    "DecisionDefinitionError": DecisionDefinitionError, "DecisionParseError": DecisionParseError,
    "parse_json_components": parse_json_components, "adjust": adjust,
    "RecordedDraw": RecordedDraw, "DistributionSummary": DistributionSummary,
    "ComponentDistribution": ComponentDistribution,
    "EstimateSource": EstimateSource, "SourceLabel": SourceLabel,
    "FullDistributionClient": FullDistributionClient, "FullDistributionEvidence": FullDistributionEvidence,
}
missing = sorted(set(EXPORTS) - set(recall_guard.__all__))
assert not missing, missing

# The package comes from this environment, not from a checkout.
package = Path(recall_guard.__file__).resolve().parent
assert package.is_relative_to(Path(sys.prefix).resolve()), (package, sys.prefix)


def no_network(*_args, **_kwargs):
    raise AssertionError("the installed-wheel check attempted a network request")


requests.Session.request = no_network

SETTINGS = {"max_tokens": 64, "temperature": 0.0, "logprobs": True, "top_logprobs": 3}
CONFIG = ModelConfig(protocol="openai_chat", endpoint="https://example.test/chat", model="fake-model",
                     parameters=SETTINGS)
PROMPT = "Macro row for 2019-03. Reply with JSON loadings for equity and gold."
IS = [f"is-{i}" for i in range(8)]
OOS = [f"oos-{i}" for i in range(8)]
REPLIES = [json.dumps({"equity": e, "gold": g}) for e, g in zip([0.1, 0.2, 0.2, 0.3], [0.4, 0.5, 0.5, 0.6])]


def evidence(base):
    """Five realized tokens with three alternatives each, as a top-k host returns them."""
    return tuple(TokenEvidence("x", base + 0.01 * i, (
        TokenAlternative("x", base), TokenAlternative("y", base - 1.0), TokenAlternative("z", base + 0.1),
    )) for i in range(5))


class FakeClient:
    """Configured completion client double: scripted replies with token evidence, no host."""

    def __init__(self):
        self.config = CONFIG
        self.calls = []
        self.decisions = 0

    def generate(self, prompt, **parameters):
        self.calls.append((prompt, dict(parameters)))
        requested = dict(self.config.parameters) | parameters
        if prompt == PROMPT:
            index = self.decisions
            self.decisions += 1
            return ModelResponse(REPLIES[index % len(REPLIES)], evidence(-0.2 - 0.15 * (index % 7)),
                                 requested_settings=requested)
        # Calibration prompts: the "is" corpus is more likely under the model than the controls.
        base = -0.2 if prompt.startswith("is") else -5.0 if prompt.startswith("oos") else -1.0
        jitter = (int(hashlib.sha256(prompt.encode()).hexdigest(), 16) % 1000) / 20000
        return ModelResponse("Direction: 1\nConfidence: 0.8", evidence(base + jitter), requested_settings=requested)

    def probe(self):
        self.generate("probe")
        required = self.config.inspect(FeatureChoices(mia_scoring=True)).capabilities
        return CapabilityReport({name: CapabilityEvidence(CapabilityStatus.SUPPORTED, "fake generation", observed=True)
                                 for name in required}, FeatureChoices(mia_scoring=True))


def spread(vector):
    return vector["gold"] - vector["equity"]


client = FakeClient()
scorer = MemoryGuardedScorer.calibrate_client(client=client, is_memorized=IS, oos_control=OOS, min_valid=4,
                                              max_workers=1)
calibrator = SourceCalibrator.from_scorer(scorer, source=EstimateSource.TOP_K, corpus_sha256=None)
assert not calibrator.is_weak

definition = DecisionDefinition(
    name="toy_allocation",
    components=(ComponentSpec("equity", -1.0, 1.0, grid=0.1), ComponentSpec("gold", -1.0, 1.0, grid=0.1)),
    parse=parse_json_components,
    summary=spread,
    decide=lambda vector: spread(vector) > 0,
)
try:
    ComponentSpec("equity", 1.0, -1.0)
except DecisionDefinitionError:
    pass
else:
    raise AssertionError("an inverted component was accepted")
try:
    definition.parse_response("not a JSON reply")
except DecisionParseError:
    pass
else:
    raise AssertionError("an unparsable reply was accepted")

config = GuardConfig(definition, SourceLabel(EstimateSource.TOP_K, 3))
assert config.mode is DrawMode.ENSEMBLE and config.draws == 32
before = len(client.calls)
result = guarded_decide(PROMPT, config, client=client, calibrator=calibrator)
assert client.calls[before:] == [(PROMPT, {})] * 32

assert isinstance(result, GuardedDecision) and isinstance(result.identity, GuardIdentity)
assert result.status == "ok", (result.status, result.reason)
assert result.mode is DrawMode.ENSEMBLE
assert dict(result.decision) == {"equity": 0.2, "gold": 0.5}
assert result.p_memorized == statistics.fmean(draw.estimate for draw in result.draws)
assert result.adjusted == adjust(result.decision, result.p_memorized)
assert isinstance(result.distribution, DistributionSummary)
assert all(isinstance(component, ComponentDistribution) for component in result.distribution.components)
assert len(result.draws) == 32 and all(isinstance(draw, RecordedDraw) for draw in result.draws)


# Replay from the recorded draws issues no request and equals the live result.
def no_request(*_args, **_kwargs):
    raise AssertionError("decide_from_draws issued a request")


client.generate = no_request
assert decide_from_draws(result.draws, config, result.identity) == result
stored = [RecordedDraw.from_dict(json.loads(json.dumps(draw.to_dict()))) for draw in result.draws]
assert decide_from_draws(stored, config, result.identity) == result

heavy = sorted(name for name in sys.modules if name.split(".")[0] in {"matplotlib", "vectorbt"})
assert not heavy, heavy

# No research data, model hosts or study code ships with the package.
forbidden = sorted(str(path.relative_to(package)) for path in package.rglob("*")
                   if path.is_dir() and path.name in {"scripts", "data", "runs", "docs"})
assert not forbidden, forbidden
assert importlib.util.find_spec("scripts") is None and importlib.util.find_spec("tests") is None

print("installed-wheel-ok")
'''


def _run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                            timeout=STEP_TIMEOUT)
    assert result.returncode == 0, f"{command} failed:\n{result.stdout}\n{result.stderr}"
    return result


@pytest.mark.skipif(UV is None, reason="uv is not on PATH; the installed-wheel test builds and installs with uv")
def test_installed_wheel_runs_the_guarded_decision_api_without_the_checkout(tmp_path):
    assert UV is not None
    dist = tmp_path / "dist"
    _run([UV, "build", "--wheel", "--out-dir", str(dist)], cwd=ROOT)
    wheels = sorted(dist.glob("recall_guard-*.whl"))
    assert len(wheels) == 1, wheels
    with zipfile.ZipFile(wheels[0]) as archive:
        top_level = {name.split("/")[0] for name in archive.namelist()}
    assert {name for name in top_level if not name.endswith(".dist-info")} == {"recall_guard"}, top_level

    venv = tmp_path / "venv"
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    _run([UV, "venv", str(venv), "--python", version], cwd=tmp_path)
    python = venv / "Scripts" / "python.exe" if os.name == "nt" else venv / "bin" / "python"
    assert python.is_file(), python
    _run([UV, "pip", "install", "--python", str(python), str(wheels[0])], cwd=tmp_path)

    script = tmp_path / "use_installed_api.py"
    script.write_text(SCRIPT, encoding="utf-8")
    result = subprocess.run([str(python), "-I", str(script)], cwd=tmp_path, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=STEP_TIMEOUT)
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    assert MARKER in result.stdout
