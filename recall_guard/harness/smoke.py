"""Smoke-test gate that produces a candidate shortlist for the harness.

Implements the `harness.smoke` component from the honest-model-ranking design
(Requirements 1.1, 1.2, 1.3, 1.4). For each candidate model the gate runs N
fixed smoke prompts via `core.nvidia_lm.NvidiaLM` and excludes any model that
times out, returns no logprobs, fails to emit a parseable `Direction:` value,
or otherwise errors. The returned `Shortlist` carries one `SmokeOutcome` per
candidate so the runner can persist the full pass/fail-reason record (1.4);
this module deliberately does no I/O.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from recall_guard.core.nvidia_lm import DEFAULT_TIMEOUT_S, NvidiaLM
from recall_guard.harness.evaluator import _parse_direction

# Documented fail_reason values (Req 1.2 / 1.3 / design.md harness.smoke).
FAIL_TIMEOUT = "timeout"
FAIL_NO_LOGPROBS = "no_logprobs"
FAIL_PARSE = "parse_failure"
FAIL_ERROR = "error"

# Substring used to distinguish "missing top_logprobs" RuntimeErrors from
# unrelated runtime errors raised by NvidiaLM (e.g. network failures).
_NO_LOGPROBS_MARKERS = ("logprobs", "top_logprobs")


LMFactory = Callable[[str, str, float], NvidiaLM]


@dataclass(frozen=True)
class SmokeOutcome:
    """Per-candidate smoke-test outcome.

    `fail_reason` is `None` on pass and one of `"timeout"`, `"no_logprobs"`,
    `"parse_failure"`, or `"error"` on fail.
    """

    model: str
    passed: bool
    fail_reason: str | None


@dataclass(frozen=True)
class Shortlist:
    """Result of the smoke-test gate.

    `selected` contains the passing models in candidate order, capped at
    `max_size`. `outcomes` contains one entry per candidate (regardless of
    pass/fail) so the runner can persist a reproducible artifact (Req 1.4).
    """

    selected: list[str]
    outcomes: list[SmokeOutcome]


def _default_lm_factory(api_key: str, model: str, timeout_s: float) -> NvidiaLM:
    return NvidiaLM(api_key=api_key, model=model, timeout_s=timeout_s)


def _classify_runtime_error(exc: RuntimeError) -> str:
    """Map a RuntimeError raised by NvidiaLM to a smoke fail_reason.

    NvidiaLM raises RuntimeError for both missing-logprobs responses and other
    HTTP failures. The smoke gate distinguishes the missing-logprobs case
    because Req 1.3 requires it to be reported as `no_logprobs`; everything
    else falls through to the generic `error` bucket.
    """
    message = str(exc).lower()
    if any(marker in message for marker in _NO_LOGPROBS_MARKERS):
        return FAIL_NO_LOGPROBS
    return FAIL_ERROR


def smoke_test(
    candidates: list[str],
    api_key: str,
    smoke_prompts: list[str],
    max_size: int = 10,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    lm_factory: LMFactory | None = None,
) -> Shortlist:
    """Run the smoke-test gate over `candidates` and return a Shortlist.

    Parameters
    ----------
    candidates:
        Ordered candidate model IDs to evaluate.
    api_key:
        NVIDIA API key forwarded to the LM factory.
    smoke_prompts:
        The fixed smoke prompts. Every candidate runs every prompt unless an
        exclusion fires earlier (Req 1.2).
    max_size:
        Hard cap on `Shortlist.selected` (Req 1.1; default 10).
    timeout_s:
        Per-call timeout forwarded to the LM client (Req 1.2).
    lm_factory:
        Optional `(api_key, model, timeout_s) -> NvidiaLM` factory used for
        test injection. Defaults to the real `NvidiaLM` constructor.

    Returns
    -------
    Shortlist
        `selected` capped at `max_size`, plus one `SmokeOutcome` per
        candidate. The function performs no I/O; the runner persists
        `outcomes` to `shortlist.json` (Req 1.4).
    """
    factory: LMFactory = lm_factory or _default_lm_factory
    outcomes: list[SmokeOutcome] = []
    selected: list[str] = []

    for model in candidates:
        outcome = _smoke_one(model, api_key, smoke_prompts, timeout_s, factory)
        outcomes.append(outcome)
        if outcome.passed and len(selected) < max_size:
            selected.append(model)

    return Shortlist(selected=selected, outcomes=outcomes)


def _smoke_one(
    model: str,
    api_key: str,
    smoke_prompts: list[str],
    timeout_s: float,
    factory: LMFactory,
) -> SmokeOutcome:
    """Run all smoke prompts against one model and classify the result."""
    lm = factory(api_key, model, timeout_s)
    parse_ok = True
    for prompt in smoke_prompts:
        try:
            result = lm.generate(prompt)
        except TimeoutError:
            return SmokeOutcome(model=model, passed=False, fail_reason=FAIL_TIMEOUT)
        except RuntimeError as exc:
            return SmokeOutcome(
                model=model, passed=False, fail_reason=_classify_runtime_error(exc)
            )
        except Exception as exc:  # pragma: no cover - defensive
            return SmokeOutcome(
                model=model,
                passed=False,
                fail_reason=f"{FAIL_ERROR}:{type(exc).__name__}",
            )
        if _parse_direction(result.content) is None:
            parse_ok = False

    if not parse_ok:
        return SmokeOutcome(model=model, passed=False, fail_reason=FAIL_PARSE)
    return SmokeOutcome(model=model, passed=True, fail_reason=None)
