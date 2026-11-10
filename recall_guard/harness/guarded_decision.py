"""Ensemble-first guarded decision: draw, estimate per draw, reduce, adjust.

An adopter asks a model for a decision (named numbers, see
:class:`~recall_guard.core.decision.DecisionDefinition`) and wants it adjusted
for memorization without wiring the guard by hand. :func:`guarded_decide` draws
the same prompt :attr:`GuardConfig.draws` times on the same model and settings,
records every draw as a :class:`~recall_guard.core.distribution.RecordedDraw`,
computes a per-draw memorization estimate from the configured evidence source,
and returns one :class:`GuardedDecision`: the consensus decision, the mean
estimate, the distribution summary, the adjusted decision and a status.

This is the package's one ensemble-first surface. A single draw is returned
only when the configuration asks for :attr:`DrawMode.SINGLE`; such a result is
labelled, carries no distribution, and its estimate never withholds or zeroes
the decision. :class:`~recall_guard.harness.scorer.MemoryGuardedScorer` keeps
its single-draw ``score`` and ``score_many`` unchanged.

Estimates come from the response's own token evidence, top-k logprobs or
full-distribution evidence (:mod:`recall_guard.mia.estimate_source`), never
from the evaluator's direction parser. A calibrator is bound to the model,
source and settings it was fitted on; that identity is checked before any
request, with settings compared without seed keys, because a seed selects a
sample and does not change the calibrated distribution.

Every result is a pure function of its recorded draws, the configuration and
the identity, so :func:`decide_from_draws` recomputes it from stored draws
without a request.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import os
import statistics
from collections import Counter
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal

from recall_guard.core.completion import CompletionClient, ModelResponse, ParameterValue
from recall_guard.core.consensus import robust_location
from recall_guard.core.decision import DecisionDefinition, DecisionParseError, adjust
from recall_guard.core.distribution import DistributionSummary, RecordedDraw, summarize_distribution
from recall_guard.core.ensemble import canonical_draw_hash
from recall_guard.core.full_distribution import FullDistributionClient, FullDistributionEvidence
from recall_guard.core.local_evidence import LocalModelIdentity
from recall_guard.core.profiles import FULL_DISTRIBUTION_SOURCE, ProfileError, ProviderProfile, load_profile
from recall_guard.core.request_control import BudgetExhaustedError, RequestError
from recall_guard.harness.request_failures import failure_kind
from recall_guard.harness.scorer_client import CalibrationIdentity, identity_settings
from recall_guard.mia.control import ControlBaseline
from recall_guard.mia.estimate_source import EstimateSource, SourceLabel, features_from_full, features_from_top_k
from recall_guard.mia.features import MiaFeatures
from recall_guard.mia.mcs import MCSCalibrator

if TYPE_CHECKING:
    from pathlib import Path

    from recall_guard.harness.scorer import MemoryGuardedScorer

#: Failure code of a reply that does not parse into the decision definition.
PARSE_ERROR = "parse_error"
#: Prefix of the failure code of a request that returned no reply.
REQUEST_ERROR_PREFIX = "request_error:"
#: Statuses that mean the credential was rejected; no later draw can succeed.
_AUTH_STATUS = frozenset({401, 403})

GuardStatus = Literal["ok", "estimate_unavailable", "adjustment_skipped_weak_calibrator", "failed"]


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_real(value: object) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _payload_sha256(payload: object) -> str:
    """SHA-256 of a JSON payload in canonical form (sorted keys, compact separators)."""
    return _sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False))


class DrawMode(Enum):
    """Ensemble by default; a single draw only on explicit request."""

    ENSEMBLE = "ensemble"
    SINGLE = "single"


@dataclass(frozen=True)
class GuardConfig:
    """How one guarded decision is drawn and reduced; validated when built, before any request.

    ``draws`` requests of the same prompt form the ensemble and at least
    ``min_parsed`` of them must parse. ``mode`` is :attr:`DrawMode.ENSEMBLE`
    unless a single draw is requested explicitly, which needs ``draws == 1``
    and therefore ``min_parsed == 1``. ``temperature`` None keeps the client's
    configured setting. ``seed_base`` sets the seed of draw ``i`` to
    ``seed_base + i``; None sends no seed override.
    """

    definition: DecisionDefinition
    source: SourceLabel
    draws: int = 32
    min_parsed: int = 24
    mode: DrawMode = DrawMode.ENSEMBLE
    temperature: float | None = None
    seed_base: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.definition, DecisionDefinition):
            raise TypeError("definition must be a DecisionDefinition")
        if not isinstance(self.source, SourceLabel):
            raise TypeError("source must be a SourceLabel")
        try:
            object.__setattr__(self, "mode", DrawMode(self.mode))
        except ValueError:
            raise ValueError(f"mode must be ensemble or single; got {self.mode!r}") from None
        for name in ("draws", "min_parsed"):
            if not _is_int(getattr(self, name)):
                raise TypeError(f"{name} must be an integer; got {getattr(self, name)!r}")
        if self.draws < 1:
            raise ValueError(f"draws must be at least 1; got {self.draws}")
        if self.mode is DrawMode.SINGLE and self.draws != 1:
            raise ValueError(f"SINGLE mode draws exactly once; got draws={self.draws}")
        if not 1 <= self.min_parsed <= self.draws:
            raise ValueError(
                f"min_parsed must satisfy 1 <= min_parsed <= draws; got min_parsed={self.min_parsed}, draws={self.draws}"
            )
        if self.temperature is not None:
            if not _is_real(self.temperature) or not math.isfinite(self.temperature) or self.temperature < 0:
                raise ValueError(f"temperature must be a finite number of at least 0, or None; got {self.temperature!r}")
            object.__setattr__(self, "temperature", float(self.temperature))
        if self.seed_base is not None and not _is_int(self.seed_base):
            raise TypeError(f"seed_base must be an integer or None; got {self.seed_base!r}")


@dataclass(frozen=True)
class GuardIdentity:
    """What one guarded decision was computed with (Requirement 1.7).

    ``settings_json`` is the canonical JSON of the client's generation settings
    with the configured temperature applied and seed keys left out
    (:func:`~recall_guard.harness.scorer_client.identity_settings`).
    ``temperature`` is the temperature those settings give every draw, None when
    the client configures none. ``calibrator_fingerprint`` is the scorer's
    ``calibration_fingerprint``; ``calibrator_is_weak`` is the calibrator's weak
    flag, which the fingerprint also covers, recorded so that
    :func:`decide_from_draws` reproduces the status without the calibrator.
    ``prompt_sha256`` is the SHA-256 of the prompt text.
    """

    model: str
    protocol: str
    endpoint: str | None
    response_mapping: str
    source: SourceLabel
    settings_json: str
    temperature: float | None
    draws: int
    calibrator_fingerprint: str
    calibrator_is_weak: bool
    prompt_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.source, SourceLabel):
            raise TypeError("source must be a SourceLabel")
        if not _is_int(self.draws) or self.draws < 1:
            raise ValueError(f"draws must be a positive integer; got {self.draws!r}")
        if not isinstance(self.calibrator_is_weak, bool):
            raise TypeError("calibrator_is_weak must be a boolean")


@dataclass(frozen=True)
class GuardedDecision:
    """One guarded decision with the evidence behind it.

    ``decision`` is the consensus of the parsed draws, or the one draw in
    :attr:`DrawMode.SINGLE`, as a read-only mapping in declared component
    order. A component split into separated clusters is None there and stays
    None in ``adjusted`` (Requirement 5.7). ``not_assessed`` names the
    components whose split check could not run (no grid, too few parsed draws,
    or a single draw): their values are kept but are never presented as
    resolved. ``category`` is the definition's ``decide`` applied to
    ``decision`` when every component is present, else None.

    ``p_memorized`` is the mean of the per-draw estimates over the parsed draws
    that carry one, the set the distribution summary uses, or the single draw's
    estimate. ``distribution`` is None for a single draw and on failure.

    Statuses:

    - ``ok``: ``adjusted`` scales every present component by
      ``1 - clip(p_memorized)``; ``reason`` states the counts when only some
      parsed draws carry an estimate.
    - ``estimate_unavailable``: no parsed draw carries an estimate; the decision
      passes through unchanged (``adjusted == decision``) and ``p_memorized`` is None.
    - ``adjustment_skipped_weak_calibrator``: the calibrator is weak; the
      decision and ``p_memorized`` are reported and ``adjusted == decision``.
    - ``failed``: fewer parsed draws than ``min_parsed``; no decision,
      adjustment, estimate or distribution, and ``reason`` gives the counts.

    ``draws`` are the recorded draws; :func:`decide_from_draws` recomputes this
    result from them. ``draws_sha256`` is
    :func:`~recall_guard.core.ensemble.canonical_draw_hash` over the parsed
    draws' response hashes, None when no draw parsed.
    """

    status: GuardStatus
    mode: DrawMode
    decision: Mapping[str, float | None] | None
    adjusted: Mapping[str, float | None] | None
    p_memorized: float | None
    distribution: DistributionSummary | None
    reason: str | None
    identity: GuardIdentity
    draws_sha256: str | None
    category: Hashable | None
    not_assessed: tuple[str, ...]
    draws: tuple[RecordedDraw, ...]


@dataclass(frozen=True)
class GuardedRun:
    """A guarded decision with the replies and evidence it was computed from, by draw index.

    ``responses`` holds None for a request that returned no reply. ``evidence``
    holds full-distribution evidence for the ``FULL_DISTRIBUTION`` source and
    None elsewhere.
    """

    decision: GuardedDecision
    responses: tuple[ModelResponse | None, ...]
    evidence: tuple[FullDistributionEvidence | None, ...]


@dataclass(frozen=True)
class SourceCalibrator:
    """The existing MCS calibrator bound to its evidence source and calibration identity.

    ``identity`` is the extended calibration identity (model, settings,
    estimate source, corpus hash) and ``fingerprint`` the scorer's
    ``calibration_fingerprint``.
    """

    mcs: MCSCalibrator
    baseline: ControlBaseline
    identity: CalibrationIdentity
    fingerprint: str

    def __post_init__(self) -> None:
        if not isinstance(self.mcs, MCSCalibrator):
            raise TypeError("mcs must be an MCSCalibrator")
        if not isinstance(self.baseline, ControlBaseline):
            raise TypeError("baseline must be a ControlBaseline")
        if not isinstance(self.identity, CalibrationIdentity):
            raise TypeError("identity must be a CalibrationIdentity")
        if not isinstance(self.fingerprint, str) or not self.fingerprint:
            raise ValueError("fingerprint must be a nonempty string")

    def estimate(self, features: MiaFeatures) -> float:
        """Calibrated ``p(memorized | features)``: the fitted classifier on the standardised features.

        Raises ``ValueError`` when a core feature cannot be standardised.
        """
        return float(self.mcs.predict_proba(features, self.baseline))

    @property
    def is_weak(self) -> bool:
        """True when the held-out AUC is below the calibration threshold."""
        return bool(self.mcs.is_weak)

    def holdout_auc_interval(self) -> dict | None:
        """Held-out AUC with its seeded stratified bootstrap interval; None without a retained holdout."""
        labels, probabilities = self.mcs.holdout_labels, self.mcs.holdout_probabilities
        if labels is None or probabilities is None:
            return None
        # Import at use: local_reporting imports local_execution, which imports the scorer.
        from recall_guard.harness.local_reporting import held_out_discrimination

        return held_out_discrimination(probabilities, labels)

    @classmethod
    def from_scorer(
        cls, scorer: MemoryGuardedScorer, *, source: EstimateSource, corpus_sha256: str | None
    ) -> SourceCalibrator:
        """Bind a configured-client calibration to ``source`` and the calibration corpus hash.

        Raises ``ValueError`` for a legacy calibration, which has no calibration identity.
        """
        identity = scorer.calibration_identity
        if identity is None:
            raise ValueError(
                "SourceCalibrator.from_scorer needs a configured-client calibration (calibrate_client or "
                "from_profile); a legacy calibration has no calibration identity"
            )
        bound = replace(identity, estimate_source=EstimateSource(source).value, corpus_sha256=corpus_sha256)
        return cls(scorer.mcs, scorer.control_baseline, bound, scorer.calibration_fingerprint)


def _comparable(identity: CalibrationIdentity) -> CalibrationIdentity:
    """The identity as compared: settings without seed keys, local provenance as declared only.

    Calibration fills the observed half of a local identity from its first
    response, so only the declared half (artifact digest, runtime and the other
    configured provenance) can be compared with a client's configuration.
    """
    local = identity.local_identity
    return replace(
        identity,
        parameters_json=identity_settings(json.loads(identity.parameters_json)),
        local_identity=None if local is None else LocalModelIdentity(declared=local.declared),
    )


def _guard_identity(
    prompt: str, config: GuardConfig, client: CompletionClient, calibrator: SourceCalibrator
) -> GuardIdentity:
    """Refuse a client whose identity differs from the calibrator's, then describe the run.

    Raises :class:`~recall_guard.harness.scorer_client.CalibrationMismatchError`
    naming every differing field (Requirement 4.2); nothing has been sent.
    """
    current = CalibrationIdentity.from_client(client, calibrator.identity.intervention_id)
    parameters = json.loads(current.parameters_json)
    if config.temperature is not None:
        parameters["temperature"] = config.temperature
    settings_json = identity_settings(parameters)
    candidate = replace(current, parameters_json=settings_json, estimate_source=config.source.source.value,
                        corpus_sha256=calibrator.identity.corpus_sha256)
    _comparable(calibrator.identity).require_match(_comparable(candidate))
    temperature = parameters.get("temperature")
    return GuardIdentity(
        model=current.model,
        protocol=current.protocol.value,
        endpoint=current.endpoint,
        response_mapping=current.response_mapping,
        source=config.source,
        settings_json=settings_json,
        temperature=float(temperature) if _is_real(temperature) else None,
        draws=config.draws,
        calibrator_fingerprint=calibrator.fingerprint,
        calibrator_is_weak=calibrator.is_weak,
        prompt_sha256=_sha256(prompt),
    )


def _is_fatal(error: BaseException) -> bool:
    """A rejected credential or an exhausted request budget: no later request can succeed."""
    return isinstance(error, BudgetExhaustedError) or getattr(error, "status_code", None) in _AUTH_STATUS


def _request(
    client: CompletionClient, prompt: str, parameters: Mapping[str, ParameterValue]
) -> tuple[ModelResponse | None, str | None]:
    """One generation request, never retried: the reply, or None and the failure code.

    A rejected credential or an exhausted budget is raised, as the existing
    ensemble does. A ``missing_token_evidence`` error that carries the complete
    reply (a client built for MIA scoring) keeps that reply: its decision can
    parse, and only its estimate is missing (Requirement 1.4).
    """
    try:
        return client.generate(prompt, **parameters), None
    except Exception as error:  # noqa: BLE001 - every other failure is recorded on its draw
        if _is_fatal(error):
            raise
        if (isinstance(error, RequestError) and error.code == "missing_token_evidence"
                and isinstance(error.response_evidence, ModelResponse)):
            return error.response_evidence, None
        if isinstance(error, RequestError) and error.code:
            return None, error.code
        return None, "timeout" if isinstance(error, TimeoutError) else failure_kind(error)


def _estimate(
    prompt: str, response: ModelResponse, config: GuardConfig, calibrator: SourceCalibrator,
    full_distribution: FullDistributionClient | None,
) -> tuple[float | None, str | None, FullDistributionEvidence | None]:
    """Per-draw estimate from the configured source, its evidence reference and full evidence.

    The estimate is None when the evidence is missing or unusable; the
    reference is the SHA-256 of the evidence payload whenever evidence arrived.
    A full-distribution request that the host refuses for this response
    (a ``RequestError`` such as a tokenisation failure or a transport error)
    leaves the estimate None. A rejected credential or a digest mismatch (the
    host serves other weights) cannot recover within the run and is raised, as
    is any other error, such as a missing bearer.
    """
    full = config.source.source is EstimateSource.FULL_DISTRIBUTION
    evidence = None
    if full:
        try:
            evidence = full_distribution.evidence(prompt, response.content)
        except RequestError as error:
            if _is_fatal(error) or error.code == "digest_mismatch":
                raise
            return None, None, None
        reference = _payload_sha256(evidence.to_dict())
    elif response.token_evidence:
        reference = _payload_sha256(response.to_dict()["token_evidence"])
    else:
        return None, None, None
    try:
        if full:
            sourced = features_from_full(response.content, evidence)
        else:
            sourced = features_from_top_k(response.content, response.to_legacy().logprobs)
        value = calibrator.estimate(sourced.features)
    except (ValueError, RuntimeError):
        return None, reference, evidence
    return (value if _is_real(value) and math.isfinite(value) else None), reference, evidence


def _record(
    index: int, prompt: str, config: GuardConfig, identity: GuardIdentity, seed: int | None,
    response: ModelResponse | None, failure: str | None, calibrator: SourceCalibrator,
    full_distribution: FullDistributionClient | None,
) -> tuple[RecordedDraw, FullDistributionEvidence | None]:
    """Turn one reply, or one failed request, into its recorded draw."""
    common = {"draw_index": index, "temperature": identity.temperature, "seed": seed}
    if response is None:
        # Nothing was received, so the response hash is that of empty content.
        return RecordedDraw(**common, response_sha256=_sha256(""), parsed=None,
                            failure_code=REQUEST_ERROR_PREFIX + str(failure), estimate=None, evidence_ref=None), None
    response_sha256 = _sha256(response.content)
    try:
        parsed = config.definition.parse_response(response.content)
    except DecisionParseError:
        return RecordedDraw(**common, response_sha256=response_sha256, parsed=None, failure_code=PARSE_ERROR,
                            estimate=None, evidence_ref=None), None
    estimate, reference, evidence = _estimate(prompt, response, config, calibrator, full_distribution)
    return RecordedDraw(**common, response_sha256=response_sha256, parsed=parsed, failure_code=None,
                        estimate=estimate, evidence_ref=reference), evidence


def guarded_run(
    prompt: str, config: GuardConfig, *, client: CompletionClient, calibrator: SourceCalibrator,
    full_distribution: FullDistributionClient | None = None,
) -> GuardedRun:
    """Draw ``prompt`` ``config.draws`` times, estimate per draw and return the guarded decision.

    Before any request: a ``FULL_DISTRIBUTION`` source without a
    full-distribution client is refused (``ValueError``) and the client's
    identity is checked against the calibrator's
    (:class:`~recall_guard.harness.scorer_client.CalibrationMismatchError`).
    The loop then issues at most ``config.draws`` requests, each with the
    configured temperature and per-draw seed as overrides, never a retry. A
    rejected credential or an exhausted budget stops the loop and is raised;
    any other failure is recorded on its draw. Estimates are computed for
    parsed draws only, so a ``FULL_DISTRIBUTION`` source requests evidence
    once per parsed draw and never for a reply that did not parse.
    """
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    if not isinstance(config, GuardConfig):
        raise TypeError("config must be a GuardConfig")
    if config.source.source is EstimateSource.FULL_DISTRIBUTION and full_distribution is None:
        raise ValueError("A full_distribution estimate source needs a full-distribution client "
                         "(full_distribution=...); nothing was sent")
    identity = _guard_identity(prompt, config, client, calibrator)
    overrides: dict[str, ParameterValue] = {}
    if config.temperature is not None:
        overrides["temperature"] = config.temperature
    configured_seed = client.config.parameters.get("seed")
    draws: list[RecordedDraw] = []
    responses: list[ModelResponse | None] = []
    evidence: list[FullDistributionEvidence | None] = []
    for index in range(config.draws):
        parameters = dict(overrides)
        if config.seed_base is not None:
            seed = parameters["seed"] = config.seed_base + index
        else:
            seed = configured_seed if _is_int(configured_seed) else None
        response, failure = _request(client, prompt, parameters)
        draw, full = _record(index, prompt, config, identity, seed, response, failure, calibrator, full_distribution)
        draws.append(draw)
        responses.append(response)
        evidence.append(full)
    return GuardedRun(decide_from_draws(draws, config, identity), tuple(responses), tuple(evidence))


def guarded_decide(
    prompt: str, config: GuardConfig, *, client: CompletionClient, calibrator: SourceCalibrator,
    full_distribution: FullDistributionClient | None = None,
) -> GuardedDecision:
    """The guarded decision of :func:`guarded_run`, without the replies and evidence."""
    return guarded_run(prompt, config, client=client, calibrator=calibrator,
                       full_distribution=full_distribution).decision


def _failed_reason(draws: Sequence[RecordedDraw], n_parsed: int, min_parsed: int) -> str:
    codes = Counter(draw.failure_code for draw in draws if draw.failure_code is not None)
    detail = ", ".join(f"{code}={count}" for code, count in sorted(codes.items()))
    return f"{n_parsed} of {len(draws)} draws parsed, below min_parsed={min_parsed}; failed draws: {detail}."


def decide_from_draws(draws: Sequence[RecordedDraw], config: GuardConfig, identity: GuardIdentity) -> GuardedDecision:
    """Recompute a guarded decision from recorded draws; no request is issued (Requirement 5.8).

    Works for either source and draw mode: the per-draw estimates are on the
    records and the calibrator's weak flag is on the identity. Raises
    ``ValueError`` when the identity's source or draw count differs from the
    configuration, when the number of draws differs from ``config.draws``, for
    a repeated ``draw_index``, or for a parsed vector whose names differ from
    the definition's components.
    """
    if not isinstance(config, GuardConfig):
        raise TypeError("config must be a GuardConfig")
    if not isinstance(identity, GuardIdentity):
        raise TypeError("identity must be a GuardIdentity")
    if identity.source != config.source:
        raise ValueError(f"identity source {identity.source} differs from the configured source {config.source}")
    if identity.draws != config.draws:
        raise ValueError(f"identity records {identity.draws} draws, the configuration draws={config.draws}")
    draws = tuple(draws)
    if not all(isinstance(draw, RecordedDraw) for draw in draws):
        raise TypeError("draws must be RecordedDraw records")
    ordered = tuple(sorted(draws, key=lambda draw: draw.draw_index))
    if len(ordered) != config.draws:
        raise ValueError(f"{len(ordered)} recorded draws given for a configuration with draws={config.draws}")
    if len({draw.draw_index for draw in ordered}) != len(ordered):
        raise ValueError("recorded draws repeat a draw_index")
    names = tuple(spec.name for spec in config.definition.components)
    parsed = [draw for draw in ordered if draw.parsed is not None]
    for draw in parsed:
        if set(draw.parsed) != set(names):
            raise ValueError(f"draw {draw.draw_index}: parsed components {sorted(draw.parsed)} differ from "
                             f"definition {config.definition.name!r} components {sorted(names)}")
    common = {
        "mode": config.mode, "identity": identity, "draws": ordered,
        "draws_sha256": canonical_draw_hash([draw.response_sha256 for draw in parsed]) if parsed else None,
    }
    if len(parsed) < config.min_parsed:
        return GuardedDecision(status="failed", decision=None, adjusted=None, p_memorized=None, distribution=None,
                               reason=_failed_reason(ordered, len(parsed), config.min_parsed), category=None,
                               not_assessed=(), **common)

    if config.mode is DrawMode.SINGLE:
        distribution = None
        decision: dict[str, float | None] = {name: parsed[0].parsed[name] for name in names}
        # The split check is a property of a distribution; one draw never has it.
        not_assessed = names
    else:
        distribution = summarize_distribution(ordered, config.definition)
        verdicts = {component.name: component.unresolved for component in distribution.components}
        decision = {
            name: None if verdicts[name] is True
            else robust_location([draw.parsed[name] for draw in parsed], mode="median")
            for name in names
        }
        not_assessed = tuple(name for name in names if verdicts[name] is None)
    present = {name: value for name, value in decision.items() if value is not None}
    category = None
    if config.definition.decide is not None and len(present) == len(names):
        category = config.definition.decide(present)

    carried = [draw.estimate for draw in parsed if draw.estimate is not None]
    p_memorized = statistics.fmean(carried) if carried else None
    counts = None
    if carried and len(carried) < len(parsed):
        counts = (f"The estimate uses {len(carried)} of {len(parsed)} parsed draws; "
                  f"the other {len(parsed) - len(carried)} carry no estimate.")
    if p_memorized is None:
        status: GuardStatus = "estimate_unavailable"
        adjusted = dict(decision)
        reason = (f"No parsed draw carries a memorization estimate (estimates on 0 of {len(parsed)} parsed draws), "
                  "so the decision is passed through unadjusted.")
    elif identity.calibrator_is_weak:
        status = "adjustment_skipped_weak_calibrator"
        adjusted = dict(decision)
        reason = ("The calibrator is weak (held-out discrimination below its threshold), so the adjustment was "
                  "skipped and the decision is passed through unadjusted.") + (f" {counts}" if counts else "")
    else:
        status = "ok"
        scaled = adjust(present, p_memorized)
        adjusted = {name: None if decision[name] is None else scaled[name] for name in names}
        reason = counts
    return GuardedDecision(
        status=status, decision=MappingProxyType(decision), adjusted=MappingProxyType(adjusted),
        p_memorized=p_memorized, distribution=distribution, reason=reason, category=category,
        not_assessed=not_assessed, **common,
    )


def full_distribution_client(
    profile: ProviderProfile | str | Path, *, environment: Mapping[str, str] | None = None
) -> FullDistributionClient:
    """Build a profile's full-distribution client, resolving ``full_distribution.endpoint_ref``.

    The gateway base URL comes from the environment variable that
    ``endpoint_ref`` names when it is set and nonempty, else from the declared
    loopback ``endpoint``, as :func:`~recall_guard.core.profiles.connect`
    resolves a generation endpoint. The bearer stays a reference
    (``credential_ref``) that the client reads at request time. No request is
    sent. A profile without the full-distribution source is refused, naming the
    sources it offers.
    """
    profile = load_profile(profile)
    profile.require_source(FULL_DISTRIBUTION_SOURCE)
    environment = os.environ if environment is None else environment
    if not isinstance(environment, Mapping):
        raise TypeError("environment must be a mapping")
    host = profile.full_distribution
    base_url, name = host.endpoint, None
    if host.endpoint_ref is not None:
        value = environment.get(host.endpoint_ref.removeprefix("env:"))
        if isinstance(value, str) and value.strip():
            base_url, name = value.strip(), host.endpoint_ref.removeprefix("env:")
    try:
        return FullDistributionClient(base_url, model_digest=host.model_digest, credential_ref=host.credential_ref,
                                      environment=environment)
    except ValueError:
        if name is None:
            raise
        # The value is not echoed: an endpoint variable may hold a credential by mistake.
        raise ProfileError(f"{name} must be a credential-free HTTP(S) base URL") from None


__all__ = [
    "PARSE_ERROR",
    "REQUEST_ERROR_PREFIX",
    "DrawMode",
    "GuardConfig",
    "GuardIdentity",
    "GuardStatus",
    "GuardedDecision",
    "GuardedRun",
    "SourceCalibrator",
    "decide_from_draws",
    "full_distribution_client",
    "guarded_decide",
    "guarded_run",
]
