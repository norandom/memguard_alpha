"""Tests for the inference-without-recall façade (Task 2.1-2.3).

Covers Requirements 3.1-3.7, 4.1, 4.3:
- calibrate builds a baseline + MCS and surfaces holdout_auc / is_weak (3.4)
- empty / rejected credential -> ConfigurationError (3.7)
- score returns a GuardedScore with the memguard discount (3.1)
- p_memorized parity with the batch harness evaluator (3.3)
- reference model optional (3.5); arbitrary prompt (3.6)
- top-level export + lean import, no matplotlib/vectorbt, no plot_* (3.2, 4.1, 4.3)

A deterministic in-process fake LM stands in for NvidiaLM (no HTTP). IS prompts get
a low loss, OOS prompts a high loss, so the MCS calibrator has a clean separation.
"""

from __future__ import annotations

import dataclasses
import hashlib
import subprocess
import sys

import pytest

from recall_guard.core.loader import EvalRow, EvalSet
from recall_guard.core.nvidia_lm import CompletionResult, TokenLogprob
from recall_guard.harness.evaluator import evaluate_model
from recall_guard.harness.scorer import (
    ConfigurationError,
    GuardedScore,
    MemoryGuardedScorer,
)

# --- deterministic fake LM ----------------------------------------------------


def _completion(base: float, content: str = "Direction: 1\nConfidence: 0.8") -> CompletionResult:
    tops = [
        {"token": "x", "logprob": base},
        {"token": "y", "logprob": base - 1.0},
        {"token": "z", "logprob": base + 0.5},
    ]
    lps = [
        TokenLogprob(token="x", logprob=base + 0.01 * i, top_logprobs=tops)
        for i in range(5)
    ]
    return CompletionResult(content=content, logprobs=lps, raw_temperature_observed=0.0)


class _FakeLM:
    """Deterministic LM: IS prompts -> low loss, OOS prompts -> high loss."""

    def __init__(self, model: str = "fake-model") -> None:
        self.model = model

    def generate(self, prompt: str, temperature: float = 0.0) -> CompletionResult:
        if prompt.startswith("is"):
            base = -0.2
        elif prompt.startswith("oos"):
            base = -5.0
        else:
            base = -1.0
        jitter = (int(hashlib.sha256(prompt.encode()).hexdigest(), 16) % 1000) / 1000.0 * 0.05
        return _completion(base + jitter)


class _RaisingLM:
    def __init__(self, exc: BaseException, model: str = "fake-model") -> None:
        self.model = model
        self._exc = exc

    def generate(self, prompt: str, temperature: float = 0.0) -> CompletionResult:
        raise self._exc


def _factory(_kind=_FakeLM):
    def factory(api_key: str, model: str, timeout_s: float):
        return _kind(model=model)

    return factory


_IS = [f"is-{i}" for i in range(8)]
_OOS = [f"oos-{i}" for i in range(8)]


def _calibrate(**overrides) -> MemoryGuardedScorer:
    kwargs = dict(
        api_key="key",
        model="m",
        is_memorized=_IS,
        oos_control=_OOS,
        min_auc=0.6,
        min_valid=4,
        seed=0,
        max_workers=1,
        lm_factory=_factory(),
    )
    kwargs.update(overrides)
    return MemoryGuardedScorer.calibrate(**kwargs)


# --- dataclass / error surface ------------------------------------------------


def test_guarded_score_is_frozen_and_error_is_runtimeerror() -> None:
    gs = GuardedScore(
        prompt_hash="abc", parse_ok=False, signal=None, raw_confidence=None,
        p_memorized=None, memguard_confidence=None, features=None, fail_reason="error",
    )
    assert dataclasses.is_dataclass(gs) and gs.__dataclass_params__.frozen
    with pytest.raises(dataclasses.FrozenInstanceError):
        gs.parse_ok = True  # type: ignore[misc]
    assert issubclass(ConfigurationError, RuntimeError)


# --- calibrate (Req 3.4, 3.7) -------------------------------------------------


def test_calibrate_returns_scorer_with_quality_signals() -> None:
    scorer = _calibrate()
    assert scorer.model == "m"
    assert 0.0 <= scorer.holdout_auc <= 1.0
    assert isinstance(scorer.is_weak, bool)
    # Separable synthetic data -> strong calibrator, not weak.
    assert scorer.holdout_auc > 0.6
    assert scorer.is_weak is False


def test_calibrate_empty_api_key_raises_configuration_error() -> None:
    with pytest.raises(ConfigurationError, match="api_key"):
        _calibrate(api_key="")


def test_calibrate_no_usable_responses_raises_configuration_error() -> None:
    bad = _factory(lambda model: _RaisingLM(RuntimeError("HTTP 401 unauthorized"), model))
    with pytest.raises(ConfigurationError):
        _calibrate(lm_factory=bad)


# --- score (Req 3.1, 3.6) -----------------------------------------------------


def test_score_returns_guarded_score_with_memguard_discount() -> None:
    scorer = _calibrate()
    gs = scorer.score("an arbitrary eval prompt")
    assert gs.parse_ok is True
    assert gs.signal == 1
    assert gs.raw_confidence == pytest.approx(0.8)
    assert 0.0 <= gs.p_memorized <= 1.0
    assert gs.memguard_confidence == pytest.approx(0.8 * (1.0 - gs.p_memorized))
    assert gs.features is not None


# --- parity with the batch harness (Req 3.3) ---------------------------------


def test_p_memorized_parity_with_evaluator() -> None:
    scorer = _calibrate()
    prompt = "an arbitrary eval prompt"

    # Same lm + baseline + mcs through the batch evaluator path.
    eval_set = EvalSet(
        rows=[EvalRow(prompt=prompt, target_direction=1, metadata={})],
        cutoff_date=None,
        path_hash="0" * 64,
    )
    result = evaluate_model(
        model_lm=scorer._lm,           # noqa: SLF001 - parity check reuses internals
        eval_set=eval_set,
        baseline=scorer._baseline,     # noqa: SLF001
        mcs=scorer._mcs,               # noqa: SLF001
        ref_lm=None,
        bootstrap_n=10,
        seed=0,
    )
    harness_p = result.records[0].p_memorized
    facade_p = scorer.score(prompt).p_memorized
    assert facade_p == pytest.approx(harness_p, abs=1e-12)


# --- reference model optionality (Req 3.5) -----------------------------------


def test_calibrate_with_reference_model_works() -> None:
    scorer = _calibrate(reference_model="ref")
    gs = scorer.score("an arbitrary eval prompt")
    assert gs.parse_ok is True
    assert 0.0 <= gs.p_memorized <= 1.0


# --- score-time error handling (Req 3.7) -------------------------------------


def test_score_auth_failure_raises_configuration_error() -> None:
    scorer = _calibrate()
    scorer._lm = _RaisingLM(RuntimeError("HTTP 401 Unauthorized"))  # noqa: SLF001
    with pytest.raises(ConfigurationError):
        scorer.score("prompt")


def test_score_timeout_returns_failure_record_not_raise() -> None:
    scorer = _calibrate()
    scorer._lm = _RaisingLM(TimeoutError("slow"))  # noqa: SLF001
    gs = scorer.score("prompt")
    assert gs.parse_ok is False
    assert gs.fail_reason == "timeout"


def test_score_reference_failure_degrades_cleanly() -> None:
    scorer = _calibrate(reference_model="ref")
    scorer._ref_lm = _RaisingLM(TimeoutError("slow"), model="ref")  # noqa: SLF001
    gs = scorer.score("prompt")
    assert gs.parse_ok is True
    assert gs.fail_reason is None
    assert 0.0 <= gs.p_memorized <= 1.0


def test_score_many_preserves_order() -> None:
    scorer = _calibrate()
    prompts = ["p0", "p1", "p2"]
    out = scorer.score_many(prompts, max_workers=1)
    assert len(out) == 3
    assert all(isinstance(g, GuardedScore) and g.parse_ok for g in out)


# --- top-level export + lean import (Req 3.2, 4.1, 4.3) ----------------------


def test_facade_is_exported_from_package_root() -> None:
    import recall_guard

    assert recall_guard.MemoryGuardedScorer is MemoryGuardedScorer
    assert "MemoryGuardedScorer" in recall_guard.__all__


def test_import_recall_guard_stays_lean() -> None:
    """A clean subprocess import of recall_guard pulls no matplotlib/vectorbt."""
    code = (
        "import sys, recall_guard\n"
        "heavy = sorted(m for m in sys.modules if m.split('.')[0] in {'matplotlib', 'vectorbt'})\n"
        "assert not heavy, heavy\n"
        "assert not any(n.startswith('plot_') for n in recall_guard.__all__), recall_guard.__all__\n"
        "assert 'MemoryGuardedScorer' in recall_guard.__all__\n"
        "print('lean-ok')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "lean-ok" in r.stdout
