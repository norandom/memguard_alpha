"""Integration test for the runner's cutoff-guard rejection — Task 6.2.

Requirement 2.5: when a shortlisted model's training cutoff post-dates the
eval-set ``_cutoff_date``, ``runner.run`` MUST abort before any HTTP call to
the candidate model. The runner converts the underlying
:class:`src.core.loader.CutoffViolation` into a non-zero exit code (= 3 today,
exposed as the convention documented in ``runner.py``).

This file is a parallel integration check to
``tests/harness/test_runner.py::test_run_aborts_on_cutoff_violation_before_any_http_call``;
it focuses specifically on the ``mock_lm.generate.call_count == 0`` invariant
that requirement 2.5 protects, plus a "safe" arm that proves the same plumbing
DOES proceed to call the LM when the cutoffs line up.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.core.nvidia_lm import CompletionResult, TokenLogprob
from src.harness import runner as runner_mod

# --- Fixture paths -----------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"
TINY_EVAL = FIXTURES_DIR / "tiny_eval.jsonl"  # _cutoff_date = 2025-06-30
TINY_IS = FIXTURES_DIR / "tiny_is_memorized.jsonl"
TINY_OOS = FIXTURES_DIR / "tiny_oos_control.jsonl"
TINY_CUTOFFS = FIXTURES_DIR / "tiny_cutoffs.yaml"
# In tiny_cutoffs.yaml:
#   late-cutoff-model: 2027-01-01  -> after 2025-06-30 (violates eval cutoff)
#   mockA, mockB:      2023-12-31  -> before 2025-06-30 (safe)


# --- Fake LM machinery -------------------------------------------------------


def _make_top_logprobs() -> list[dict[str, Any]]:
    return [{"token": f"tok{i}", "logprob": -1.0 - 0.1 * i} for i in range(20)]


def _make_logprobs(content: str) -> list[TokenLogprob]:
    tokens = content.split() or ["x"]
    return [
        TokenLogprob(
            token=tok,
            logprob=-0.5 - 0.05 * (i % 5),
            top_logprobs=_make_top_logprobs(),
        )
        for i, tok in enumerate(tokens)
    ]


def _make_completion(direction: int = 1, confidence: float = 0.7) -> CompletionResult:
    content = f"Direction: {direction}\nConfidence: {confidence}"
    return CompletionResult(
        content=content,
        logprobs=_make_logprobs(content),
        raw_temperature_observed=0.0,
    )


class _CountingFakeLM:
    """LM stub that tracks ``generate`` invocations.

    ``generate.call_count`` mirrors ``unittest.mock.Mock`` semantics so the
    test can assert it directly per the task observable.
    """

    def __init__(self, model: str, *, direction_cycle: list[int] | None = None) -> None:
        self.model = model
        self.api_key = "fake"
        self.timeout_s = 15.0
        self.api_base = "fake://"
        self._cycle = direction_cycle or [1, -1, 0, 1, -1]
        # ``generate`` is a bound method below, but we attach a sibling
        # callable wrapper that exposes ``call_count``. We keep ``calls`` for
        # backwards compatibility with the existing fake style.
        self.calls: list[str] = []

    def generate(self, prompt: str, temperature: float = 0.0) -> CompletionResult:
        self.calls.append(prompt)
        idx = (len(self.calls) - 1) % len(self._cycle)
        return _make_completion(direction=self._cycle[idx], confidence=0.7)

    @property
    def call_count(self) -> int:  # pragma: no cover - convenience accessor
        return len(self.calls)


class _ForbiddenLM:
    """LM stub whose ``generate`` raises ``AssertionError`` if invoked.

    Used to prove the runner aborts before any HTTP call: if the cutoff guard
    fails to fire, the runner will reach ``generate`` and the test will fail
    with a clear diagnostic.
    """

    def __init__(self, model: str) -> None:
        self.model = model
        self.api_key = "fake"
        self.timeout_s = 15.0
        self.api_base = "fake://"
        self.calls: list[str] = []

    def generate(self, prompt: str, temperature: float = 0.0) -> CompletionResult:
        self.calls.append(prompt)
        raise AssertionError(
            f"generate() must not be called for {self.model!r} — cutoff guard "
            f"should have aborted the run before any HTTP traffic."
        )

    @property
    def call_count(self) -> int:
        return len(self.calls)


def _make_factory(fakes: dict[str, Any]):
    """Return an ``lm_factory`` that yields the configured fake LMs."""

    def factory(api_key: str, model: str, timeout_s: float):
        if model not in fakes:
            fakes[model] = _CountingFakeLM(model=model)
        return fakes[model]

    return factory


def _build_args(
    monkeypatch: pytest.MonkeyPatch,
    out_dir: Path,
    *,
    shortlist: str,
    seed: int = 0,
    bootstrap_n: int = 50,
):
    monkeypatch.setenv("NVIDIA_API_KEY", "test-api-key")
    parser = runner_mod.build_parser()
    return parser.parse_args(
        [
            "--eval-set",
            str(TINY_EVAL),
            "--is-memorized",
            str(TINY_IS),
            "--oos-control",
            str(TINY_OOS),
            "--cutoffs",
            str(TINY_CUTOFFS),
            "--out-dir",
            str(out_dir),
            "--seed",
            str(seed),
            "--bootstrap-n",
            str(bootstrap_n),
            "--shortlist",
            shortlist,
            "--no-reference",
        ]
    )


# --- Tests -------------------------------------------------------------------


def test_cutoff_violation_aborts_before_any_http_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Req 2.5: a shortlisted model whose cutoff post-dates the eval cutoff
    causes the runner to abort with a non-zero exit code BEFORE
    ``lm.generate`` is ever called.

    Eval cutoff (tiny_eval.jsonl): 2025-06-30
    Model cutoff (late-cutoff-model in tiny_cutoffs.yaml): 2027-01-01
    => CutoffViolation is raised; runner returns 3.
    """
    out_dir = tmp_path / "violation"
    args = _build_args(monkeypatch, out_dir, shortlist="late-cutoff-model")

    forbidden = _ForbiddenLM("late-cutoff-model")
    fakes: dict[str, Any] = {"late-cutoff-model": forbidden}

    rc = runner_mod.run(args, lm_factory=_make_factory(fakes))

    # Non-zero exit (3 by convention; see runner.py docstring & cutoff branch).
    assert rc != 0, "runner.run must exit non-zero on a cutoff violation"
    assert rc == 3, (
        f"runner.run should exit 3 on CutoffViolation per the runner contract, "
        f"got {rc}"
    )
    # Observable: NO HTTP call to the candidate model.
    assert forbidden.call_count == 0, (
        f"generate() was called {forbidden.call_count} time(s); the cutoff "
        f"guard must abort BEFORE any HTTP traffic."
    )
    # No artifacts should have been written either, because the run aborted
    # before the artifact-writing phase. This guards against silent partial
    # writes that would mask a guard regression.
    assert not (out_dir / "records.jsonl").exists()
    assert not (out_dir / "summary.csv").exists()
    assert not (out_dir / "top3.md").exists()
    assert not (out_dir / "manifest.json").exists()


def test_cutoff_violation_with_mocker_call_count(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mocker
) -> None:
    """Same guarantee as the previous test but using ``pytest-mock`` to track
    ``generate.call_count`` directly, matching the task's "OR track call
    count via ``mocker.Mock``" guidance.
    """
    out_dir = tmp_path / "violation-mocker"
    args = _build_args(monkeypatch, out_dir, shortlist="late-cutoff-model")

    fake = _CountingFakeLM("late-cutoff-model")
    # Wrap ``generate`` in a Mock so ``call_count`` is exposed verbatim.
    mock_generate = mocker.patch.object(fake, "generate", wraps=fake.generate)

    rc = runner_mod.run(args, lm_factory=_make_factory({"late-cutoff-model": fake}))

    assert rc != 0
    assert rc == 3
    assert mock_generate.call_count == 0, (
        f"mock_lm.generate.call_count must be 0; was {mock_generate.call_count}"
    )


def test_cutoff_safe_run_proceeds_normally(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Inverse arm: with only ``mockA`` + ``mockB`` (both 2023-12-31, before
    the eval cutoff 2025-06-30), the cutoff guard passes and the runner
    proceeds to call the LM. This proves the guard fires only when it should.
    """
    out_dir = tmp_path / "safe"
    args = _build_args(monkeypatch, out_dir, shortlist="mockA,mockB")

    fakes: dict[str, Any] = {
        "mockA": _CountingFakeLM("mockA"),
        "mockB": _CountingFakeLM("mockB"),
    }
    rc = runner_mod.run(args, lm_factory=_make_factory(fakes))

    assert rc == 0, f"safe run should exit 0; got {rc}"
    # At least one LM call across the shortlist.
    total_calls = fakes["mockA"].call_count + fakes["mockB"].call_count
    assert total_calls > 0, "safe run should have made at least one LM call"
    # Sanity: artifacts were written this time.
    assert (out_dir / "records.jsonl").exists()
    assert (out_dir / "manifest.json").exists()
