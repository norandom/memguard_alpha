"""Opt-in N-draw execution and reduction over one prompt.

Asking the same prompt many times and reducing the replies is worthwhile
because the serving stack is nondeterministic even at ``temperature=0``: the
model rarely repeats a parameter vector exactly, yet usually reaches the same
decision. A single draw is therefore a poor estimate of the parameters and a
good estimate of the decision, and an ensemble is how a caller gets both, with
the disagreement made explicit rather than averaged away.

Nothing here is reachable unless a caller passes an :class:`EnsembleSpec`.
There is no implicit default instance, no environment variable, and no
process-wide toggle -- a hidden switch would change what a run persisted
without changing what the caller wrote, which is precisely what an audit trail
cannot tolerate.

Two callbacks, not one. ``decide`` maps a reply to a hashable decision and
drives agreement; ``components`` optionally maps a reply to named scalars and
drives location and multimodality. One callback cannot serve both: "agreement"
needs a categorical outcome while "location" needs an ordered scalar, and
conflating them is how a reported consensus ends up naming a different reading
than the reported location.

Execution runs in waves rather than one flat fan-out, so the request budget can
be enforced, a rejected credential can abort before the remaining draws are paid
for, and replies can be reduced incrementally instead of accumulating.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Hashable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import StrEnum

from recall_guard.core.consensus import (
    MultimodalVerdict,
    Tail,
    detect_multimodal,
    grid_adherence,
    lag_dependence,
    robust_location,
    smallest_certifiable_n,
    snap_to_grid,
    wilson_interval,
)
from recall_guard.core.nvidia_lm import CompletionResult, NvidiaLM

#: Version tag mixed into the draw-set hash preimage. Bump only when the
#: canonicalisation changes, since that changes every replayed hash.
_HASH_SCHEME = "recall_guard.draws.v1"

#: Separator between draws in the hash preimage. A byte that cannot appear in
#: decoded JSON text, so two draws cannot be concatenated into one.
_HASH_SEPARATOR = "\x00"

#: Statuses that mean the credential was rejected rather than the request failed.
_AUTH_STATUS = frozenset({401, 403})


class LocationMode(StrEnum):
    """Which location estimator to apply to an unflagged component."""

    MEAN = "mean"
    MEDIAN = "median"
    TRIMMED = "trimmed"


class MultimodalAction(StrEnum):
    """What to do with a component that holds separated clusters.

    Silently averaging across one is the single behaviour that must never be
    available: it launders a real disagreement into false precision, returning a
    value the model effectively never emitted.
    """

    FLAG = "flag"
    RAISE = "raise"


class ReferenceMode(StrEnum):
    """Whether the optional reference draw varies per ensemble draw."""

    FIXED = "fixed"
    PER_DRAW = "per_draw"


@dataclass(frozen=True)
class CostEstimate:
    """What an ensemble would cost, computed without issuing anything."""

    worst_case_requests: int
    estimated_seconds: float | None


@dataclass(frozen=True)
class EnsembleResult:
    """One ensemble's reduced answer plus the evidence behind it.

    ``component_verdicts`` carries the separated-cluster check for **every**
    component, not only the flagged ones. A verdict of ``separated=False`` with
    masses near the threshold is a very different situation from one with no
    mass on either side, and only the caller can judge which matters -- so the
    result reports what the test saw rather than only its boolean conclusion. A
    ``None`` verdict means the check did not run at all.

    ``max_tokens`` and ``temperature`` record the settings the draws were taken
    under. An ensemble is an audit artifact, and "under what generation settings"
    belongs next to the draw-set digest: a consensus sampled at a different token
    budget than production is not measuring the production decision.
    """

    consensus: CompletionResult
    location: Mapping[str, float]
    location_snapped: Mapping[str, float] | None
    grid_adherence: Mapping[str, float] | None
    multimodal: tuple[str, ...]
    component_verdicts: tuple[tuple[str, MultimodalVerdict | None], ...]
    agreement: float
    agreement_ci: tuple[float, float] | None
    draw_dependence: float | None
    max_tokens: int | None
    temperature: float | None
    n_requested: int
    n_parsed: int
    fail_counts: tuple[tuple[str, int], ...]
    draws_sha256: str
    draws: tuple[CompletionResult, ...] = field(default=())


@dataclass(frozen=True)
class EnsembleSpec:
    """Opt-in ensemble configuration.

    Every default here is **provisional**: all of them were calibrated against a
    single measurement date at a crisis onset, chosen because it was the hard
    case. Whether they generalise to calmer regimes is unmeasured, which is why
    each threshold is a field rather than a literal.

    ``max_tokens`` and ``temperature`` default to ``None``, meaning the client's
    own defaults. **Set them to whatever production uses.** An ensemble drawn at
    a different token budget is not measuring the production decision -- and on a
    reasoning model the budget is not a detail, because the chain of thought
    consumes it and truncates the reply before the payload a caller parses.
    Measured on one such model, dropping from a 2048-token production budget to
    the 512-token client default took the parse rate from 95% to 48%.

    ``draws`` is sized for **agreement precision**, not for component-split
    detection; those are different numbers and the second is larger. See
    :func:`~recall_guard.core.consensus.smallest_detectable_split_n`.
    """

    draws: int = 64
    max_workers: int = 8
    min_parsed: int = 24
    grid: float | None = None
    confidence: float = 0.95
    tail: Tail = Tail.ONE_SIDED
    agreement_target: float | None = None
    location_mode: LocationMode = LocationMode.MEDIAN
    trim: float = 0.25
    multimodal_action: MultimodalAction = MultimodalAction.FLAG
    mass_min: float = 0.25
    trough_steps: int = 3
    density_ratio: float = 10.0
    min_cluster_draws: int = 8
    min_cluster_density: float = 1.5
    max_total_requests: int | None = None
    max_transport_failure_ratio: float = 0.25
    retain_draws: bool = False
    reference_mode: ReferenceMode = ReferenceMode.FIXED
    max_tokens: int | None = None
    temperature: float | None = None

    def __post_init__(self) -> None:
        if self.draws < 1:
            raise ValueError(f"draws must be >= 1; got {self.draws}")
        if not 1 <= self.min_parsed <= self.draws:
            raise ValueError(
                f"min_parsed must satisfy 1 <= min_parsed <= draws; "
                f"got min_parsed={self.min_parsed}, draws={self.draws}"
            )
        if self.max_workers < 1:
            raise ValueError(f"max_workers must be >= 1; got {self.max_workers}")
        if self.grid is not None and not (self.grid > 0 and math.isfinite(self.grid)):
            raise ValueError(f"grid must be a positive finite number; got {self.grid!r}")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError(f"confidence must be in (0, 1); got {self.confidence}")
        if not 0.0 < self.mass_min <= 0.5:
            raise ValueError(f"mass_min must be in (0, 0.5]; got {self.mass_min}")
        if self.trough_steps < 1:
            raise ValueError(f"trough_steps must be >= 1; got {self.trough_steps}")
        if self.density_ratio < 1.0:
            raise ValueError(f"density_ratio must be >= 1; got {self.density_ratio}")
        if self.min_cluster_draws < 2:
            raise ValueError(
                f"min_cluster_draws must be >= 2; got {self.min_cluster_draws}"
            )
        if self.min_cluster_density < 1.0:
            raise ValueError(
                f"min_cluster_density must be >= 1; got {self.min_cluster_density}"
            )
        if not 0.0 <= self.trim < 0.5:
            raise ValueError(f"trim must be in [0, 0.5); got {self.trim}")
        if not 0.0 <= self.max_transport_failure_ratio <= 1.0:
            raise ValueError(
                f"max_transport_failure_ratio must be in [0, 1]; "
                f"got {self.max_transport_failure_ratio}"
            )
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1 when set; got {self.max_tokens}")
        if self.temperature is not None and not 0.0 <= self.temperature <= 2.0:
            raise ValueError(
                f"temperature must be in [0, 2] when set; got {self.temperature}"
            )
        if self.max_total_requests is not None and self.max_total_requests < 1:
            raise ValueError(
                f"max_total_requests must be >= 1 when set; got {self.max_total_requests}"
            )

        floor = self.smallest_certifiable_n
        if floor is not None and floor > self.draws:
            raise ValueError(
                f"agreement_target={self.agreement_target} cannot be certified at "
                f"draws={self.draws} under {self.tail.value} confidence "
                f"{self.confidence}: at least {floor} unanimous draws are required. "
                "Raise draws, lower the target, or drop it."
            )

    @property
    def smallest_certifiable_n(self) -> int | None:
        """Draws needed to certify ``agreement_target``, or ``None`` if unset.

        Unanimity is the best case, so this is a hard floor -- below it no
        observed agreement can clear the target, whatever the model returns.
        """
        if self.agreement_target is None:
            return None
        return smallest_certifiable_n(
            self.agreement_target, confidence=self.confidence, tail=self.tail
        )


def estimate_cost(
    spec: EnsembleSpec,
    *,
    max_retries: int,
    has_reference: bool,
    seconds_per_request: float | None = None,
) -> CostEstimate:
    """Worst-case request count and duration, without issuing any request.

    The nominal draw count is the *floor*, not the worst case: each logical draw
    can become ``max_retries + 1`` requests, and a configured reference model
    doubles the whole thing.
    """
    if max_retries < 0:
        raise ValueError(f"max_retries must be >= 0; got {max_retries}")
    per_draw = (max_retries + 1) * (2 if has_reference else 1)
    worst_case = spec.draws * per_draw
    seconds = None
    if seconds_per_request is not None:
        waves = math.ceil(spec.draws / spec.max_workers)
        seconds = waves * per_draw * seconds_per_request
    return CostEstimate(worst_case_requests=worst_case, estimated_seconds=seconds)


def canonical_draw_hash(contents: Sequence[str]) -> str:
    """SHA-256 over the draw set's reply text, independent of arrival order.

    Covers reply **text only**. Logprob structures are excluded because their
    key ordering comes from the provider's JSON and is not stable across servers
    or library versions, and timing and thread identity are excluded because
    they are not properties of the answer.

    Sorting before hashing is what makes the digest a property of the draw
    *set*; the tie-break rules elsewhere in this module recover a deterministic
    order from content alone, so nothing depends on how the draws arrived.
    """
    digest = hashlib.sha256()
    digest.update(_HASH_SCHEME.encode("utf-8"))
    for content in sorted(contents):
        digest.update(_HASH_SEPARATOR.encode("utf-8"))
        digest.update(content.encode("utf-8"))
    return digest.hexdigest()


def _is_auth_failure(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None)
    return status in _AUTH_STATUS if status is not None else False


def _classify(exc: BaseException) -> str:
    if isinstance(exc, TimeoutError):
        return "timeout"
    if getattr(exc, "status_code", None) is not None:
        return "http"
    return "transport"


def _canonical_order(
    draws: Sequence[CompletionResult],
    decisions: Sequence[Hashable],
) -> list[int]:
    """Indices of ``draws`` in content order.

    Under an ensemble every prompt is identical, so which reply lands at which
    arrival index is thread-scheduling noise. Sorting by content first makes
    every downstream step a function of the draw set rather than of the run.
    """
    return sorted(
        range(len(draws)),
        key=lambda i: (draws[i].content, hashlib.sha256(draws[i].content.encode()).hexdigest()),
    )


def _select_consensus(
    draws: Sequence[CompletionResult],
    decisions: Sequence[Hashable],
    modal: Hashable,
) -> CompletionResult:
    """The modal-class draw whose content sorts first.

    Deliberately an actually-observed draw, never a synthesised one: a composed
    vector can be a point the model never considered. The tie-break is content
    only, so a replay of the stored set picks the same draw.
    """
    candidates = [d for d, decision in zip(draws, decisions, strict=True) if decision == modal]
    return min(candidates, key=lambda d: d.content)


def _reduce_components(
    parsed: Sequence[CompletionResult],
    components: Callable[[CompletionResult], Mapping[str, float]],
    spec: EnsembleSpec,
) -> tuple[
    dict[str, float],
    dict[str, float] | None,
    dict[str, float] | None,
    tuple[str, ...],
    tuple[tuple[str, MultimodalVerdict | None], ...],
]:
    """Per-component location, with the multimodality gate applied first.

    Returns the verdict for every component, flagged or not: a near-miss split
    and a clearly unimodal component are different situations, and discarding
    the distinction leaves a caller unable to tell a confident answer from a
    lucky one.
    """
    per_axis: dict[str, list[float]] = {}
    for draw in parsed:
        for axis, value in components(draw).items():
            per_axis.setdefault(axis, []).append(float(value))

    location: dict[str, float] = {}
    snapped: dict[str, float] | None = {} if spec.grid is not None else None
    adherence: dict[str, float] | None = {} if spec.grid is not None else None
    flagged: list[str] = []
    verdicts: list[tuple[str, MultimodalVerdict | None]] = []

    for axis in sorted(per_axis):
        values = per_axis[axis]
        if adherence is not None and spec.grid is not None:
            adherence[axis] = grid_adherence(values, spec.grid)
        verdict = detect_multimodal(
            values,
            grid=spec.grid,
            mass_min=spec.mass_min,
            trough_steps=spec.trough_steps,
            density_ratio=spec.density_ratio,
            min_draws=spec.min_cluster_draws,
            min_cluster_density=spec.min_cluster_density,
        )
        verdicts.append((axis, verdict))
        if verdict is not None and verdict.separated:
            # No location is reported for a flagged component. There is no single
            # value to estimate across a genuine split, and no trim fraction
            # escapes the gap -- reporting one would be the false-precision the
            # whole gate exists to prevent.
            flagged.append(axis)
            if spec.multimodal_action is MultimodalAction.RAISE:
                raise ValueError(
                    f"component {axis!r} holds two separated clusters "
                    f"({verdict.lower_mass:.0%} below / {verdict.upper_mass:.0%} above a "
                    f"{verdict.trough_mass:.0%} gap); refusing to reduce it to one location"
                )
            continue
        located = robust_location(values, mode=spec.location_mode.value, trim=spec.trim)
        location[axis] = located
        if snapped is not None and spec.grid is not None:
            snapped[axis] = snap_to_grid(located, spec.grid)

    return location, snapped, adherence, tuple(flagged), tuple(verdicts)


def reduce_draws(
    draws: Sequence[CompletionResult],
    spec: EnsembleSpec,
    *,
    decide: Callable[[CompletionResult], Hashable],
    components: Callable[[CompletionResult], Mapping[str, float]] | None = None,
    waves: Sequence[int] | None = None,
    n_requested: int | None = None,
    fail_counts: Mapping[str, int] | None = None,
) -> EnsembleResult:
    """Reduce a draw set to one answer. Pure: no I/O, no randomness, no clock.

    Separated from execution so a stored draw set can be replayed into a
    bit-identical result without contacting a model, which is what makes an
    ensemble auditable after the fact.
    """
    if not draws:
        raise ValueError("cannot reduce an empty draw set")

    order = _canonical_order(draws, [])
    ordered = [draws[i] for i in order]
    decisions = [decide(d) for d in ordered]

    tally: dict[Hashable, int] = {}
    for decision in decisions:
        tally[decision] = tally.get(decision, 0) + 1
    # Ties break toward the lexicographically smaller repr, never dict order.
    modal = min(tally, key=lambda d: (-tally[d], repr(d)))
    agreeing = tally[modal]
    agreement = agreeing / len(ordered)

    location: dict[str, float] = {}
    snapped = adherence = None
    flagged: tuple[str, ...] = ()
    verdicts: tuple[tuple[str, MultimodalVerdict | None], ...] = ()
    if components is not None:
        location, snapped, adherence, flagged, verdicts = _reduce_components(
            ordered, components, spec
        )

    dependence = None
    if waves is not None:
        ordered_waves = [waves[i] for i in order]
        dependence = lag_dependence(decisions, ordered_waves)

    return EnsembleResult(
        consensus=_select_consensus(ordered, decisions, modal),
        location=location,
        location_snapped=snapped,
        grid_adherence=adherence,
        multimodal=flagged,
        component_verdicts=verdicts,
        agreement=agreement,
        agreement_ci=wilson_interval(
            agreeing, len(ordered), confidence=spec.confidence, tail=spec.tail
        ),
        draw_dependence=dependence,
        max_tokens=spec.max_tokens,
        temperature=spec.temperature,
        n_requested=n_requested if n_requested is not None else len(draws),
        n_parsed=len(ordered),
        fail_counts=tuple(sorted((fail_counts or {}).items())),
        draws_sha256=canonical_draw_hash([d.content for d in ordered]),
        draws=tuple(ordered) if spec.retain_draws else (),
    )


def generate_ensemble(
    lm: NvidiaLM,
    prompt: str,
    spec: EnsembleSpec,
    *,
    decide: Callable[[CompletionResult], Hashable],
    components: Callable[[CompletionResult], Mapping[str, float]] | None = None,
) -> EnsembleResult:
    """Draw ``spec.draws`` replies to ``prompt`` and reduce them.

    Raises
    ------
    ValueError
        If a component holds separated clusters and the spec asks to raise.
    RuntimeError
        If the request budget is exhausted, too few draws are usable, or
        transport failures exceed the configured share. Each of these is a
        refusal to report a confident answer computed from survivors.
    """
    parsed: list[CompletionResult] = []
    wave_tags: list[int] = []
    failures: dict[str, int] = {}
    issued = 0
    wave_index = 0

    while len(parsed) + sum(failures.values()) < spec.draws:
        remaining = spec.draws - (len(parsed) + sum(failures.values()))
        size = min(spec.max_workers, remaining)
        if spec.max_total_requests is not None and issued + size > spec.max_total_requests:
            raise RuntimeError(
                f"ensemble request budget exhausted: {spec.max_total_requests} requests "
                f"allowed, {issued} already issued, next wave needs {size}"
            )

        with ThreadPoolExecutor(max_workers=size) as pool:
            outcomes = list(
                pool.map(lambda _: _safe_draw(lm, prompt, spec), range(size))
            )
        issued += size

        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                if _is_auth_failure(outcome):
                    # Abort rather than paying for the remaining draws; a rejected
                    # credential will not start working mid-ensemble.
                    raise outcome
                failures[_classify(outcome)] = failures.get(_classify(outcome), 0) + 1
                continue
            try:
                decide(outcome)
            except Exception:  # noqa: BLE001 - a caller callback may raise anything
                failures["projection"] = failures.get("projection", 0) + 1
                continue
            parsed.append(outcome)
            wave_tags.append(wave_index)
        wave_index += 1

    transport_failures = sum(
        count for reason, count in failures.items() if reason != "projection"
    )
    if transport_failures / spec.draws > spec.max_transport_failure_ratio:
        raise RuntimeError(
            f"transport failures {transport_failures}/{spec.draws} exceed the configured "
            f"limit of {spec.max_transport_failure_ratio:.0%}; refusing to report a "
            "consensus computed from the survivors"
        )
    if len(parsed) < spec.min_parsed:
        raise RuntimeError(
            f"only {len(parsed)} usable draws of {spec.draws} requested, below "
            f"min_parsed={spec.min_parsed}; refusing to report a consensus"
        )

    return reduce_draws(
        parsed,
        spec,
        decide=decide,
        components=components,
        waves=wave_tags,
        n_requested=spec.draws,
        fail_counts=failures,
    )


def _safe_draw(
    lm: NvidiaLM, prompt: str, spec: EnsembleSpec
) -> CompletionResult | BaseException:
    """One draw under the spec's generation settings.

    Only overrides that the caller actually set are forwarded, so an unset spec
    reproduces the client's own defaults exactly -- which keeps a one-draw
    ensemble byte-identical to the single-draw path.
    """
    kwargs: dict[str, object] = {}
    if spec.max_tokens is not None:
        kwargs["max_tokens"] = spec.max_tokens
    if spec.temperature is not None:
        kwargs["temperature"] = spec.temperature
    try:
        return lm.generate(prompt, **kwargs)
    except Exception as exc:  # noqa: BLE001 - triaged by the caller
        return exc


__all__ = [
    "CostEstimate",
    "EnsembleResult",
    "EnsembleSpec",
    "LocationMode",
    "MultimodalAction",
    "ReferenceMode",
    "canonical_draw_hash",
    "estimate_cost",
    "generate_ensemble",
    "reduce_draws",
]
