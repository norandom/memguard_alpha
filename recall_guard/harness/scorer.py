"""Public inference-without-recall façade: :class:`MemoryGuardedScorer`.

This is the stable surface that downstream projects (e.g. Global_Macro_AI_Factors'
``macro_framework`` Track A) consume. It wraps the existing primitives (the NVIDIA
LM client, the per-model control baseline, and the MCS contamination calibrator)
behind two phases:

* :meth:`MemoryGuardedScorer.calibrate` runs the model over the control + IS/OOS
  corpora and trains the per-model calibrator (HTTP-heavy, done once).
* :meth:`MemoryGuardedScorer.score` / :meth:`score_many` turn one prompt into a
  :class:`GuardedScore`: the parsed directional signal, the raw MIA features, the
  calibrated ``p_memorized``, and the MemGuard-discounted confidence.

The score path reuses the evaluator's parser, the MIA feature computation, the
control-baseline standardisation, and the calibrator's ``predict_proba`` verbatim,
so ``p_memorized`` is bit-for-bit identical to what the batch harness produces for
the same inputs (Req 3.3). The façade adds no statistics of its own.

One example of consuming this façade: a macro overlay multiplies each
AI-generated Black-Litterman view magnitude by ``(1 - p_memorized)`` before
it can move money, which is the same discount ``memguard_confidence`` applies
to ``raw_confidence``. A weak or missing score passes the raw exposure
through, and parse failures fall back to the consumer's risk-parity core.
The score is a discount, not a certificate: no model is presumed clean, and
the consumer owns the fallback policy.

Layer note: this module lives in the ``harness`` layer (top of the stack), so it may
depend on ``core``, ``mia``, and ``harness.evaluator``. It imports nothing from
``harness.plots`` or ``portfolio``, so re-exporting it from the package root keeps
``import recall_guard`` free of matplotlib/vectorbt (Req 4.1, 4.3).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from recall_guard.core.consensus import lag_dependence, wilson_interval
from recall_guard.core.ensemble import EnsembleSpec, canonical_draw_hash
from recall_guard.core.loader import EvalRow
from recall_guard.core.nvidia_lm import NvidiaLM, generate_many
from recall_guard.harness.evaluator import (
    FAIL_ERROR,
    FAIL_PARSE,
    FAIL_TIMEOUT,
    _classify_runtime_error,
    _hash_prompt,
    _parse_confidence,
    _parse_direction,
)
from recall_guard.mia.control import ControlBaseline, build_baseline
from recall_guard.mia.features import MiaFeatures, compute_mia_features
from recall_guard.mia.mcs import MCSCalibrator
from recall_guard.mia.mcs import train as _mcs_train

#: Factory contract for constructing an LM, matching ``harness.runner.LMFactory``.
LMFactory = Callable[[str, str, float], NvidiaLM]

#: Response statuses that mean the credential was rejected. Preferred over text
#: matching whenever the failure carries a status code (ensemble-consensus 7.6).
_AUTH_STATUS: frozenset[int] = frozenset({401, 403})

#: Fallback substrings for failures that carry no status code -- an offline
#: double, a transport-layer error, or a caller-constructed ``RuntimeError``.
#: Only consulted when :attr:`LMHTTPError.status_code` is unavailable, because a
#: bare substring search for ``"401"`` also matches a trace id, a port, or a
#: byte count (Req 3.7).
_AUTH_MARKERS: tuple[str, ...] = (
    "401",
    "403",
    "unauthorized",
    "forbidden",
    "authentication",
    "invalid api key",
    "invalid_api_key",
)


class ConfigurationError(RuntimeError):
    """Raised when the NIM credential is absent, empty, or rejected (Req 3.7)."""


@dataclass(frozen=True)
class GuardedScore:
    """One guarded inference result.

    Attributes
    ----------
    prompt_hash:
        First 16 hex chars of ``sha256(prompt)`` (matches the harness convention).
    parse_ok:
        ``True`` when the response parsed and the MIA/MCS pipeline ran.
    signal:
        Parsed direction in ``{-1, 0, 1}``; ``None`` on failure.
    raw_confidence:
        Parsed confidence in ``[0, 1]``; ``None`` on failure.
    p_memorized:
        Calibrated ``p(memorized | features) ∈ [0, 1]``; ``None`` on failure.
    memguard_confidence:
        ``raw_confidence * (1 - p_memorized)``; ``None`` on failure.
    features:
        The raw :class:`MiaFeatures`; ``None`` on failure.
    fail_reason:
        One of ``"timeout"`` / ``"no_logprobs"`` / ``"parse_failure"`` / ``"error"``
        on failure; ``None`` on success.
    """

    prompt_hash: str
    parse_ok: bool
    signal: int | None
    raw_confidence: float | None
    p_memorized: float | None
    memguard_confidence: float | None
    features: MiaFeatures | None
    fail_reason: str | None


def _is_auth_error(exc: BaseException) -> bool:
    """Return True if ``exc`` is a NIM auth/authorisation failure.

    A status code, when the failure carries one, is authoritative in both
    directions: a 500 whose body happens to contain ``401`` is not an auth
    failure. Text matching remains the fallback for failures with no status.
    """
    status = getattr(exc, "status_code", None)
    if status is not None:
        return status in _AUTH_STATUS
    message = str(exc).lower()
    return any(marker in message for marker in _AUTH_MARKERS)


def _default_factory(min_call_interval_s: float) -> LMFactory:
    def _factory(api_key: str, model: str, timeout_s: float) -> NvidiaLM:
        return NvidiaLM(
            api_key=api_key,
            model=model,
            timeout_s=timeout_s,
            min_call_interval_s=min_call_interval_s,
        )

    return _factory


def _rows(prompts: Sequence[str]) -> list[EvalRow]:
    """Adapt calibration prompt strings to ``EvalRow`` (label routing is by corpus)."""
    return [EvalRow(prompt=p, target_direction=0, metadata={}) for p in prompts]


@dataclass(frozen=True)
class EnsembledScore:
    """One prompt scored over many draws.

    **`p_memorized_point` is the exposure multiplier.** ``consensus.p_memorized``
    is the score of one actually-observed draw and is evidence only -- the two
    can differ, because the representative draw is selected by rank while the
    point estimate is a reduction over all draws. Using the consensus draw's
    score to scale exposure would silently substitute a single draw for the
    ensemble, which is the thing this feature exists to stop.

    ``p_memorized_point is None`` exactly when the ensemble failed, and it is
    never ``0.0`` in that case: zero would mean "pass 100% of exposure through",
    the opposite of what an unusable measurement should imply.
    """

    consensus: GuardedScore
    p_memorized_point: float | None
    p_memorized_ci: tuple[float, float] | None
    p_memorized_conservative: float | None
    agreement: float | None
    agreement_ci: tuple[float, float] | None
    draw_dependence: float | None
    n_requested: int
    n_parsed: int
    fail_counts: tuple[tuple[str, int], ...]
    draws_sha256: str
    draws: tuple[GuardedScore, ...] = ()


class MemoryGuardedScorer:
    """Calibrated, per-model inference-without-recall scorer.

    Construct via :meth:`calibrate` (which performs the model calls and training),
    then call :meth:`score` / :meth:`score_many`.
    """

    def __init__(
        self,
        *,
        lm: NvidiaLM,
        baseline: ControlBaseline,
        mcs: MCSCalibrator,
        ref_lm: NvidiaLM | None,
    ) -> None:
        self._lm = lm
        self._baseline = baseline
        self._mcs = mcs
        self._ref_lm = ref_lm

    # -- public read-only state ------------------------------------------------

    @property
    def model(self) -> str:
        return self._mcs.model

    @property
    def holdout_auc(self) -> float:
        """Held-out IS/OOS separation of the trained calibrator (Req 3.4)."""
        return self._mcs.holdout_auc

    @property
    def is_weak(self) -> bool:
        """``True`` when ``holdout_auc`` is below the calibration gate (Req 3.4)."""
        return self._mcs.is_weak

    # -- construction ----------------------------------------------------------

    @classmethod
    def calibrate(
        cls,
        *,
        api_key: str,
        model: str,
        is_memorized: Sequence[str],
        oos_control: Sequence[str],
        reference_model: str | None = None,
        min_auc: float = 0.6,
        min_valid: int = 50,
        seed: int = 0,
        max_workers: int = 8,
        timeout_s: float = 45.0,
        min_call_interval_s: float = 0.0,
        lm_factory: LMFactory | None = None,
    ) -> MemoryGuardedScorer:
        """Build the control baseline and train the MCS calibrator for ``model``.

        Raises
        ------
        ConfigurationError
            If ``api_key`` is empty, or if the model returns no usable responses
            during calibration (the typical signature of a rejected credential,
            an unavailable model, or an unreachable endpoint) (Req 3.7).
        ValueError
            If a class has too few usable rows to calibrate / train (Req 3.4).
        """
        if not api_key:
            raise ConfigurationError(
                "NVIDIA api_key is required to calibrate a MemoryGuardedScorer; "
                "got an empty value. Set NVIDIA_API_KEY or pass api_key=."
            )

        factory = lm_factory or _default_factory(min_call_interval_s)
        lm = factory(api_key, model, timeout_s)
        ref_lm = factory(api_key, reference_model, timeout_s) if reference_model else None

        oos_rows = _rows(oos_control)
        is_rows = _rows(is_memorized)

        baseline = build_baseline(
            lm, oos_rows, ref_lm, min_valid=min_valid, max_workers=max_workers
        )
        if baseline.n_valid == 0:
            raise ConfigurationError(
                f"model {model!r} returned no usable responses during calibration; "
                "check NVIDIA_API_KEY, the model id, and endpoint availability."
            )
        if not baseline.is_calibrated:
            raise ValueError(
                f"control baseline could not calibrate for {model!r}: "
                f"{baseline.n_valid} usable rows < min_valid={min_valid}."
            )

        mcs = _mcs_train(
            model_lm=lm,
            is_memorized=is_rows,
            oos_control=oos_rows,
            baseline=baseline,
            ref_lm=ref_lm,
            min_auc=min_auc,
            seed=seed,
            max_workers=max_workers,
        )
        return cls(lm=lm, baseline=baseline, mcs=mcs, ref_lm=ref_lm)

    # -- scoring ---------------------------------------------------------------

    def score(self, prompt: str) -> GuardedScore:
        """Score one prompt into a :class:`GuardedScore`.

        Raises
        ------
        ConfigurationError
            If the NIM endpoint rejects the credential while scoring (Req 3.7).
        """
        primary = self._safe_generate(self._lm, prompt)
        ref_res = self._safe_generate(self._ref_lm, prompt) if self._ref_lm else None
        return self._build_guarded_score(prompt, primary, ref_res)

    def score_many(self, prompts: Sequence[str], *, max_workers: int = 8) -> list[GuardedScore]:
        """Score many prompts (parallel LM calls); preserves input order."""
        primaries = generate_many(self._lm, list(prompts), max_workers=max_workers)
        refs: list = (
            generate_many(self._ref_lm, list(prompts), max_workers=max_workers)
            if self._ref_lm is not None
            else [None] * len(prompts)
        )
        return [
            self._build_guarded_score(p, primary, ref_res)
            for p, primary, ref_res in zip(prompts, primaries, refs, strict=True)
        ]

    def score_ensemble(
        self,
        prompt: str,
        *,
        spec: EnsembleSpec,
        conservative_quantile: float | None = None,
    ) -> EnsembledScore:
        """Score one prompt over ``spec.draws`` draws and reduce the results.

        A single scoring is close to uninformative as an exposure multiplier:
        measured on one identical prompt, its 95% band spans two thirds of the
        unit interval. Ensembling narrows that, and reports what is left.

        Each draw is scored through the *unchanged* single-draw path and the
        resulting scores are then reduced. Averaging the intermediate features
        and scoring once would be a different quantity -- the calibrator is a
        sigmoid, so the score of the mean is not the mean of the scores.

        The point estimate is the **mean**, because attenuation is linear in the
        score and the mean is therefore unbiased for expected attenuation. No
        symmetric trimming is applied: the upper tail of this distribution is
        the contamination evidence the score exists to report, so trimming it
        away would discard the signal and shift the estimate toward the
        risk-increasing side.

        Parameters
        ----------
        spec:
            Explicit configuration. There is no default instance.
        conservative_quantile:
            Optional upper quantile of the score, for a caller who would rather
            withhold more exposure than risk withholding too little.

        Raises
        ------
        ConfigurationError
            If the endpoint rejects the credential while drawing.
        """
        if conservative_quantile is not None and not 0.0 <= conservative_quantile <= 1.0:
            raise ValueError(
                f"conservative_quantile must be in [0, 1]; got {conservative_quantile!r}"
            )

        prompt_hash = _hash_prompt(prompt)
        # The reference draw is held fixed across the ensemble: varying it would
        # double the request count for one of four features. The cost is that
        # every draw's score is then correlated through that shared reference,
        # so the reported spread understates the true spread.
        ref_res = self._safe_generate(self._ref_lm, prompt) if self._ref_lm else None

        scored: list[GuardedScore] = []
        failures: dict[str, int] = {}
        contents: list[str] = []
        waves: list[int] = []

        for wave, batch in enumerate(_wave_sizes(spec)):
            with ThreadPoolExecutor(max_workers=batch) as pool:
                draws = list(pool.map(lambda _: self._safe_generate(self._lm, prompt), range(batch)))
            for draw in draws:
                if isinstance(draw, RuntimeError) and _is_auth_error(draw):
                    raise ConfigurationError(
                        f"NIM rejected the credential while scoring model {self.model!r}: {draw}"
                    )
                if isinstance(draw, BaseException) or draw is None:
                    reason = "timeout" if isinstance(draw, TimeoutError) else "transport"
                    failures[reason] = failures.get(reason, 0) + 1
                    continue
                guarded = self._build_guarded_score(prompt, draw, ref_res)
                if not guarded.parse_ok:
                    failures[guarded.fail_reason or FAIL_ERROR] = (
                        failures.get(guarded.fail_reason or FAIL_ERROR, 0) + 1
                    )
                    continue
                scored.append(guarded)
                contents.append(draw.content)
                waves.append(wave)

        return self._reduce_scores(
            prompt_hash, scored, contents, waves, failures, spec, conservative_quantile
        )

    def _reduce_scores(
        self,
        prompt_hash: str,
        scored: list[GuardedScore],
        contents: list[str],
        waves: list[int],
        failures: dict[str, int],
        spec: EnsembleSpec,
        conservative_quantile: float | None,
    ) -> EnsembledScore:
        fail_counts = tuple(sorted(failures.items()))
        if len(scored) < spec.min_parsed or not scored:
            # Report the failure rather than a consensus over the survivors. A
            # confident-looking answer computed from a handful of draws is the
            # false-success artifact this package refuses to mint.
            return EnsembledScore(
                consensus=_fail(prompt_hash, _modal_reason(failures)),
                p_memorized_point=None,
                p_memorized_ci=None,
                p_memorized_conservative=None,
                agreement=None,
                agreement_ci=None,
                draw_dependence=None,
                n_requested=spec.draws,
                n_parsed=len(scored),
                fail_counts=fail_counts,
                draws_sha256=canonical_draw_hash(contents),
                draws=(),
            )

        order = sorted(range(len(scored)), key=lambda i: (scored[i].p_memorized, contents[i]))
        ordered = [scored[i] for i in order]
        values = [g.p_memorized for g in ordered]

        signals = [g.signal for g in ordered]
        tally: dict[int | None, int] = {}
        for signal in signals:
            tally[signal] = tally.get(signal, 0) + 1
        modal = min(tally, key=lambda s: (-tally[s], repr(s)))

        return EnsembledScore(
            # The representative draw is ranked by the score itself, so it sits
            # at the middle of the very distribution being reduced. It is still
            # evidence, not the multiplier -- see the class docstring.
            consensus=ordered[(len(ordered) - 1) // 2],
            p_memorized_point=math.fsum(values) / len(values),
            p_memorized_ci=_empirical_interval(values, spec.confidence),
            p_memorized_conservative=(
                _quantile(values, conservative_quantile)
                if conservative_quantile is not None
                else None
            ),
            agreement=tally[modal] / len(ordered),
            agreement_ci=wilson_interval(
                tally[modal], len(ordered), confidence=spec.confidence, tail=spec.tail
            ),
            draw_dependence=lag_dependence(signals, [waves[i] for i in order]),
            n_requested=spec.draws,
            n_parsed=len(ordered),
            fail_counts=fail_counts,
            draws_sha256=canonical_draw_hash(contents),
            draws=tuple(ordered) if spec.retain_draws else (),
        )

    # -- internals -------------------------------------------------------------

    @staticmethod
    def _safe_generate(lm: NvidiaLM | None, prompt: str):
        if lm is None:
            return None
        try:
            return lm.generate(prompt)
        except (TimeoutError, RuntimeError) as exc:
            return exc

    def _build_guarded_score(self, prompt: str, primary, ref_res) -> GuardedScore:
        prompt_hash = _hash_prompt(prompt)

        if isinstance(primary, TimeoutError):
            return _fail(prompt_hash, FAIL_TIMEOUT)
        if isinstance(primary, RuntimeError):
            if _is_auth_error(primary):
                raise ConfigurationError(
                    f"NIM rejected the credential while scoring model {self.model!r}: {primary}"
                )
            return _fail(prompt_hash, _classify_runtime_error(primary))
        if isinstance(primary, BaseException) or primary is None:
            return _fail(prompt_hash, FAIL_ERROR)

        content = primary.content
        direction = _parse_direction(content)
        confidence = _parse_confidence(content)
        if direction is None or confidence is None:
            return _fail(prompt_hash, FAIL_PARSE)

        ref_logprobs = None
        if self._ref_lm is not None and ref_res is not None and not isinstance(ref_res, BaseException):
            ref_logprobs = ref_res.logprobs

        try:
            features = compute_mia_features(content, primary.logprobs, ref_logprobs)
        except (ValueError, RuntimeError):
            return _fail(prompt_hash, FAIL_ERROR)

        try:
            p_memorized = float(self._mcs.predict_proba(features, self._baseline))
        except ValueError:
            return _fail(prompt_hash, FAIL_ERROR)

        return GuardedScore(
            prompt_hash=prompt_hash,
            parse_ok=True,
            signal=direction,
            raw_confidence=float(confidence),
            p_memorized=p_memorized,
            memguard_confidence=float(confidence) * (1.0 - p_memorized),
            features=features,
            fail_reason=None,
        )


def _wave_sizes(spec: EnsembleSpec) -> list[int]:
    """Draw counts per wave, so a budget can be enforced between them."""
    full, remainder = divmod(spec.draws, spec.max_workers)
    sizes = [spec.max_workers] * full
    if remainder:
        sizes.append(remainder)
    return sizes


def _modal_reason(failures: dict[str, int]) -> str:
    """The most common failure, ties broken by name so replay is stable."""
    if not failures:
        return FAIL_ERROR
    return min(failures, key=lambda reason: (-failures[reason], reason))


def _empirical_interval(values: Sequence[float], confidence: float) -> tuple[float, float]:
    """Descriptive spread of the observed scores, by order statistic.

    Deterministic on purpose: a bootstrap band would need a stored seed, and the
    reduction is specified to contain no randomness so a persisted draw set
    replays exactly. This describes the draws actually taken -- it is not an
    inferential interval for the underlying quantity, and widening it would not
    make it one.
    """
    ordered = sorted(values)
    tail = (1.0 - confidence) / 2.0
    last = len(ordered) - 1
    lo = ordered[min(last, math.floor(tail * last))]
    hi = ordered[max(0, math.ceil((1.0 - tail) * last))]
    return (lo, hi)


def _quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(q * (len(ordered) - 1)))]


def _fail(prompt_hash: str, fail_reason: str) -> GuardedScore:
    return GuardedScore(
        prompt_hash=prompt_hash,
        parse_ok=False,
        signal=None,
        raw_confidence=None,
        p_memorized=None,
        memguard_confidence=None,
        features=None,
        fail_reason=fail_reason,
    )


__all__ = [
    "ConfigurationError",
    "EnsembledScore",
    "GuardedScore",
    "LMFactory",
    "MemoryGuardedScorer",
]
