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
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime

from recall_guard.core.completion import CompletionClient, ModelResponse, ParameterValue
from recall_guard.core.consensus import lag_dependence, wilson_interval
from recall_guard.core.ensemble import EnsembleSpec, canonical_draw_hash
from recall_guard.core.loader import EvalRow
from recall_guard.core.nvidia_lm import NvidiaLM, check_model_access, generate_many
from recall_guard.core.request_control import BudgetExhaustedError
from recall_guard.harness.evaluator import (
    FAIL_ERROR,
    FAIL_PARSE,
    FAIL_TIMEOUT,
    _classify_runtime_error,
    _hash_prompt,
    _parse_confidence,
    _parse_direction,
)
from recall_guard.harness.scorer_client import CalibrationIdentity, LegacyLogprobAdapter
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

    ``sampled_at`` records when the draws were taken. The sampled distribution
    moves between sessions as well as within one, so a consensus has a shelf
    life and a cached one is not the same as a fresh one.

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
    max_tokens: int | None
    temperature: float | None
    sampled_at: str | None
    n_requested: int
    n_parsed: int
    fail_counts: tuple[tuple[str, int], ...]
    draws_sha256: str
    draws: tuple[GuardedScore, ...] = ()


@dataclass(frozen=True)
class ScoredDraw:
    """Retained score, exact response content and original executor wave.

    A score with ``parse_ok=False`` contributes its existing failure reason.
    Transport failures without a score are supplied separately to the reducer.
    Waves describe acquisition order, independently of canonical score order.
    """

    score: GuardedScore
    content: str
    wave: int


class MemoryGuardedScorer:
    """Calibrated, per-model inference-without-recall scorer.

    Construct via :meth:`calibrate` (which performs the model calls and training),
    then call :meth:`score` / :meth:`score_many`.
    """

    def __init__(
        self,
        *,
        lm: NvidiaLM | LegacyLogprobAdapter,
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

    @property
    def holdout_labels(self) -> tuple[int, ...] | None:
        """Actual held-out class labels paired with ``holdout_probabilities``."""
        return self._mcs.holdout_labels

    @property
    def holdout_probabilities(self) -> tuple[float, ...] | None:
        """Actual predictions used for ``holdout_auc``; never fresh predictions."""
        return self._mcs.holdout_probabilities

    @property
    def calibration_identity(self) -> CalibrationIdentity | None:
        """Configured fit provenance; legacy calibrations retain their old contract."""
        return self._lm.identity if isinstance(self._lm, LegacyLogprobAdapter) else None

    @property
    def calibration_valid_counts(self) -> dict[str, int | None]:
        """Actual valid rows, after MIA computation, before the train/holdout split."""
        return {"baseline": self._baseline.n_valid, "is_memorized": self._mcs.n_valid_is,
                "oos_control": self._mcs.n_valid_oos}

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

        for access_model in (lm, ref_lm):
            if access_model is not None:
                try:
                    check_model_access(access_model)
                except (TimeoutError, RuntimeError, ValueError) as error:
                    raise ConfigurationError(
                        f"model {access_model.model!r} access check failed; "
                        "verify inference authorization, model ID and logprob support"
                    ) from error

        return cls._fit(
            lm=lm, ref_lm=ref_lm, is_memorized=is_memorized, oos_control=oos_control,
            min_auc=min_auc, min_valid=min_valid, seed=seed, max_workers=max_workers,
        )

    @classmethod
    def calibrate_client(
        cls,
        *,
        client: CompletionClient,
        is_memorized: Sequence[str],
        oos_control: Sequence[str],
        intervention_id: str = "original",
        min_auc: float = 0.6,
        min_valid: int = 50,
        seed: int = 0,
        max_workers: int = 8,
    ) -> MemoryGuardedScorer:
        """Fit the existing MIA/MCS estimator with a configured completion client.

        A real generation probe must verify all feature evidence before corpus
        calls. Text-only clients cannot calibrate this estimator. The immutable
        fit identity includes exact requested settings and intervention ID.
        ``intervention_id`` labels already prepared calibration prompts; this
        method does not rewrite them. A rewritten fit needs both corpora to
        have passed the same pipeline used at inference.

        Configured scoring currently uses the primary model only. Legacy
        :meth:`calibrate` continues to support an optional reference model.
        Credentials, pacing, retries and physical budgets belong to ``client``.
        """
        lm = LegacyLogprobAdapter(client, intervention_id, _is_auth_error)
        try:
            lm.probe()
        except BudgetExhaustedError:
            raise
        except (TimeoutError, RuntimeError) as error:
            raise ConfigurationError(
                f"model {lm.model!r} access check failed; verify inference authorization and MIA support"
            ) from error
        try:
            scorer = cls._fit(
                lm=lm, ref_lm=None, is_memorized=is_memorized, oos_control=oos_control,
                min_auc=min_auc, min_valid=min_valid, seed=seed, max_workers=max_workers,
            )
            lm.finish_calibration()
            return scorer
        except BudgetExhaustedError:
            raise
        except RuntimeError as error:
            if _is_auth_error(error):
                raise ConfigurationError(f"Credential rejected while calibrating model {lm.model!r}") from error
            raise

    @classmethod
    def _fit(
        cls, *, lm, ref_lm, is_memorized: Sequence[str], oos_control: Sequence[str],
        min_auc: float, min_valid: int, seed: int, max_workers: int,
    ) -> MemoryGuardedScorer:
        """Reuse baseline/train verbatim for both public construction paths."""
        model = lm.model
        oos_rows, is_rows = _rows(oos_control), _rows(is_memorized)

        try:
            baseline = build_baseline(
                lm, oos_rows, ref_lm, min_valid=min_valid, max_workers=max_workers
            )
        finally:
            if isinstance(lm, LegacyLogprobAdapter):
                lm.raise_if_fatal()
        if baseline.n_valid == 0:
            credential_hint = (
                "check configured credentials, model ID, endpoint and token evidence."
                if isinstance(lm, LegacyLogprobAdapter)
                else "check NVIDIA_API_KEY, the model id, and endpoint availability."
            )
            raise ConfigurationError(
                f"model {model!r} returned no usable responses during calibration; "
                + credential_hint
            )
        if not baseline.is_calibrated:
            raise ValueError(
                f"control baseline could not calibrate for {model!r}: "
                f"{baseline.n_valid} usable rows < min_valid={min_valid}."
            )

        train_hooks = {"before_fit": lm.raise_if_fatal} if isinstance(lm, LegacyLogprobAdapter) else {}
        try:
            mcs = _mcs_train(
                model_lm=lm,
                is_memorized=is_rows,
                oos_control=oos_rows,
                baseline=baseline,
                ref_lm=ref_lm,
                min_auc=min_auc,
                seed=seed,
                max_workers=max_workers,
                **train_hooks,
            )
        finally:
            if isinstance(lm, LegacyLogprobAdapter):
                lm.raise_if_fatal()
        return cls(lm=lm, baseline=baseline, mcs=mcs, ref_lm=ref_lm)

    # -- scoring ---------------------------------------------------------------

    def validate_client(
        self, client: CompletionClient, *, intervention_id: str = "original", **parameters: ParameterValue,
    ) -> None:
        """Check workflow/client identity before issuing a target request."""
        if not isinstance(self._lm, LegacyLogprobAdapter):
            raise ValueError("Configured response scoring requires a configured-client calibration")
        self._lm.validate_client(client, intervention_id=intervention_id, **parameters)

    def score_response(
        self, prompt: str, response: ModelResponse, *, client: CompletionClient,
        intervention_id: str = "original",
    ) -> GuardedScore:
        """Score already generated genuine evidence without another model call.

        The caller supplies its originating client and intervention identity;
        ModelResponse itself does not contain endpoint/model provenance. Exact
        requested settings and known confirmed contradictions are also checked.
        Missing evidence or identity mismatches raise ValueError explicitly.
        """
        self.validate_client(client, intervention_id=intervention_id)
        primary = self._lm.validate_response(response)
        return self._build_guarded_score(prompt, primary, None)

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

    def reduce_scored_draws(
        self, prompt: str, draws: Sequence[ScoredDraw], *, spec: EnsembleSpec,
        failures: Mapping[str, int] | None = None, conservative_quantile: float | None = None,
        sampled_at: str | None = None,
    ) -> EnsembledScore:
        """Replay independently scored draws through the existing scored reducer.

        Makes no model or feature calls. ``sampled_at=None`` means replay; a
        live caller supplies its original acquisition timestamp. A workflow
        using the unchanged executor assigns wave ``draw_index // max_workers``.
        This keeps wave membership attached to each score during canonical
        sorting. Missing/unattempted draws are never invented as failures.
        """
        if not isinstance(spec, EnsembleSpec):
            raise TypeError("spec must be EnsembleSpec")
        if conservative_quantile is not None and not 0 <= conservative_quantile <= 1:
            raise ValueError("conservative_quantile must be in [0, 1]")
        counts = dict(failures or {})
        if any(not isinstance(reason, str) or not reason or type(count) is not int or count < 0
               for reason, count in counts.items()):
            raise ValueError("Failure counts must have named, nonnegative integer counts")
        prompt_hash = _hash_prompt(prompt)
        scored, contents, waves = [], [], []
        for draw in draws:
            if not isinstance(draw, ScoredDraw) or not isinstance(draw.score, GuardedScore):
                raise TypeError("draws must contain ScoredDraw values with GuardedScore evidence")
            if not isinstance(draw.content, str) or type(draw.wave) is not int or draw.wave < 0:
                raise ValueError("Each scored draw needs exact text and a nonnegative executor wave")
            if draw.score.prompt_hash != prompt_hash:
                raise ValueError("Scored draw belongs to a different prompt")
            if not draw.score.parse_ok:
                reason = draw.score.fail_reason or FAIL_ERROR
                counts[reason] = counts.get(reason, 0) + 1
            else:
                scored.append(draw.score)
                contents.append(draw.content)
                waves.append(draw.wave)
        if len(scored) + sum(counts.values()) > spec.draws:
            raise ValueError("Retained scores and failures exceed the requested draws")
        return self._reduce_scores(prompt_hash, scored, contents, waves, counts, spec,
                                   conservative_quantile, sampled_at)

    def score_many(self, prompts: Sequence[str], *, max_workers: int = 8) -> list[GuardedScore]:
        """Score many prompts (parallel LM calls); preserves input order."""
        if isinstance(self._lm, LegacyLogprobAdapter):
            self._lm.validate_client(self._lm.client, intervention_id=self._lm.identity.intervention_id)
        try:
            primaries = generate_many(self._lm, list(prompts), max_workers=max_workers)
        finally:
            if isinstance(self._lm, LegacyLogprobAdapter):
                try:
                    self._lm.raise_if_fatal()
                except BudgetExhaustedError:
                    raise
                except RuntimeError as error:
                    if _is_auth_error(error):
                        raise ConfigurationError(f"Credential rejected while scoring model {self.model!r}") from error
                    raise
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

        if isinstance(self._lm, LegacyLogprobAdapter):
            overrides = {name: value for name, value in (
                ("max_tokens", spec.max_tokens), ("temperature", spec.temperature),
            ) if value is not None}
            self._lm.validate_client(self._lm.client, intervention_id=self._lm.identity.intervention_id, **overrides)

        prompt_hash = _hash_prompt(prompt)
        started_at = datetime.now(UTC).isoformat()
        # The reference draw is held fixed across the ensemble: varying it would
        # double the request count for one of four features. The cost is that
        # every draw's score is then correlated through that shared reference,
        # so the reported spread understates the true spread.
        ref_res = (
            self._safe_generate(self._ref_lm, prompt, spec=spec) if self._ref_lm else None
        )

        scored: list[GuardedScore] = []
        failures: dict[str, int] = {}
        contents: list[str] = []
        waves: list[int] = []

        for wave, batch in enumerate(_wave_sizes(spec)):
            with ThreadPoolExecutor(max_workers=batch) as pool:
                draws = list(
                    pool.map(
                        lambda _: self._safe_generate(self._lm, prompt, spec=spec),
                        range(batch),
                    )
                )
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
            prompt_hash, scored, contents, waves, failures, spec,
            conservative_quantile, started_at,
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
        sampled_at: str | None,
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
                max_tokens=spec.max_tokens,
                temperature=spec.temperature,
                sampled_at=sampled_at,
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
            max_tokens=spec.max_tokens,
            temperature=spec.temperature,
            sampled_at=sampled_at,
            n_requested=spec.draws,
            n_parsed=len(ordered),
            fail_counts=fail_counts,
            draws_sha256=canonical_draw_hash(contents),
            draws=tuple(ordered) if spec.retain_draws else (),
        )

    # -- internals -------------------------------------------------------------

    @staticmethod
    def _safe_generate(lm: NvidiaLM | LegacyLogprobAdapter | None, prompt: str, *, spec: EnsembleSpec | None = None):
        """One draw, optionally under an ensemble spec's generation settings.

        With no spec the client's own defaults apply, which is what keeps the
        single-draw path -- and a one-draw ensemble that overrides nothing --
        byte-identical to what it always was.
        """
        if lm is None:
            return None
        kwargs: dict[str, object] = {}
        if spec is not None and spec.max_tokens is not None:
            kwargs["max_tokens"] = spec.max_tokens
        if spec is not None and spec.temperature is not None:
            kwargs["temperature"] = spec.temperature
        try:
            return lm.generate(prompt, **kwargs)
        except BudgetExhaustedError:
            raise
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
    "ScoredDraw",
]
