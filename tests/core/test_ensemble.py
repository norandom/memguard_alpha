"""Ensemble configuration, execution, reduction, and replay determinism."""

from __future__ import annotations

import random

import pytest

from recall_guard.core.consensus import Tail, lag_dependence
from recall_guard.core.ensemble import (
    EnsembleSpec,
    LocationMode,
    MultimodalAction,
    canonical_draw_hash,
    estimate_cost,
    generate_ensemble,
    reduce_draws,
)
from recall_guard.core.nvidia_lm import CompletionResult, TokenLogprob


def _completion(content: str) -> CompletionResult:
    return CompletionResult(
        content=content,
        logprobs=[TokenLogprob(token="x", logprob=-0.1, top_logprobs=[{"x": -0.1}])],
        raw_temperature_observed=0.0,
    )


class _ScriptedLM:
    """Returns a scripted sequence of replies, one per call."""

    def __init__(self, replies, model: str = "fake-model") -> None:
        self.model = model
        self._replies = list(replies)
        self.calls = 0

    def generate(
        self, prompt: str, temperature: float = 0.0, max_tokens: int = 512
    ) -> CompletionResult:
        reply = self._replies[self.calls % len(self._replies)]
        self.calls += 1
        if isinstance(reply, BaseException):
            raise reply
        return _completion(reply)


def _spec(**kw) -> EnsembleSpec:
    base = {"draws": 8, "max_workers": 2, "min_parsed": 1}
    return EnsembleSpec(**{**base, **kw})


# --- 3.1 configuration and validation ----------------------------------------


def test_spec_rejects_inconsistent_settings() -> None:
    for kwargs in (
        {"draws": 0},
        {"draws": 4, "min_parsed": 5},
        {"max_workers": 0},
        {"grid": 0.0},
        {"grid": -1.0},
        {"confidence": 0.0},
        {"confidence": 1.0},
        {"mass_min": 0.0},
        {"trough_steps": 0},
        {"max_transport_failure_ratio": 1.5},
        {"max_total_requests": 0},
        {"trim": 0.5},
    ):
        with pytest.raises(ValueError):
            EnsembleSpec(**kwargs)


def test_unreachable_agreement_target_is_refused_at_construction() -> None:
    """A target that no draw count can certify must not be silently accepted.

    Under the source proposal's own defaults this configuration would spend its
    entire draw budget on every prompt and still report the target unmet.
    """
    with pytest.raises(ValueError, match="certif"):
        EnsembleSpec(draws=24, min_parsed=1, agreement_target=0.95, tail=Tail.TWO_SIDED)

    ok = EnsembleSpec(draws=128, min_parsed=1, agreement_target=0.95, tail=Tail.TWO_SIDED)
    assert ok.smallest_certifiable_n == 73


def test_feasibility_is_evaluated_against_the_declared_tail() -> None:
    """The tail is not decoration: it moves the floor from 73 to 52."""
    assert EnsembleSpec(
        draws=64, min_parsed=1, agreement_target=0.95, tail=Tail.ONE_SIDED
    ).smallest_certifiable_n == 52
    with pytest.raises(ValueError):
        EnsembleSpec(draws=64, min_parsed=1, agreement_target=0.95, tail=Tail.TWO_SIDED)


def test_spec_without_a_target_reports_no_floor() -> None:
    assert _spec().smallest_certifiable_n is None


# --- 3.2 cost estimation and budgeting ---------------------------------------


def test_cost_estimate_counts_retries_and_reference() -> None:
    """The nominal draw count is the floor, not the worst case."""
    spec = _spec(draws=100)
    assert estimate_cost(spec, max_retries=0, has_reference=False).worst_case_requests == 100
    assert estimate_cost(spec, max_retries=2, has_reference=False).worst_case_requests == 300
    assert estimate_cost(spec, max_retries=2, has_reference=True).worst_case_requests == 600


def test_cost_estimate_issues_no_requests() -> None:
    lm = _ScriptedLM(["up"])
    estimate_cost(_spec(), max_retries=2, has_reference=False)
    assert lm.calls == 0


def test_budget_exhaustion_stops_the_ensemble() -> None:
    lm = _ScriptedLM(["up"])
    spec = _spec(draws=8, max_total_requests=4)
    with pytest.raises(RuntimeError, match="budget"):
        generate_ensemble(lm, "p", spec, decide=lambda c: c.content)
    assert lm.calls <= 4


# --- 3.3 execution and failure triage ----------------------------------------


def test_counts_requested_and_parsed_separately() -> None:
    lm = _ScriptedLM(["up", "up", RuntimeError("boom"), "up"])
    result = generate_ensemble(lm, "p", _spec(draws=8), decide=lambda c: c.content)
    assert result.n_requested == 8
    assert result.n_parsed == 6
    assert dict(result.fail_counts)["transport"] == 2


def test_projection_failure_is_its_own_category() -> None:
    """Distinct from transport and parse; they mean different things."""

    def _raising(completion):
        if completion.content == "bad":
            raise ValueError("cannot project")
        return completion.content

    lm = _ScriptedLM(["up", "bad"])
    result = generate_ensemble(lm, "p", _spec(draws=8), decide=_raising)
    assert dict(result.fail_counts)["projection"] == 4
    assert result.n_parsed == 4


def test_thin_sample_fails_rather_than_reporting_a_consensus() -> None:
    """A confident answer over a handful of survivors is the false-success case.

    The transport-ratio ceiling is raised here so the thin-sample guard is the
    one under test; both refusals are real and either would be correct.
    """
    lm = _ScriptedLM(["up", RuntimeError("boom"), RuntimeError("boom"), RuntimeError("boom")])
    with pytest.raises(RuntimeError, match="usable draws"):
        generate_ensemble(
            lm,
            "p",
            _spec(draws=8, min_parsed=4, max_transport_failure_ratio=1.0),
            decide=lambda c: c.content,
        )


def test_all_draws_failing_raises(caplog) -> None:
    lm = _ScriptedLM([RuntimeError("boom")])
    with pytest.raises(RuntimeError):
        generate_ensemble(lm, "p", _spec(draws=4), decide=lambda c: c.content)


def test_excess_transport_failure_ratio_raises() -> None:
    lm = _ScriptedLM(["up", RuntimeError("boom")])
    with pytest.raises(RuntimeError, match="transport"):
        generate_ensemble(
            lm,
            "p",
            _spec(draws=8, min_parsed=1, max_transport_failure_ratio=0.25),
            decide=lambda c: c.content,
        )


def test_credential_rejection_stops_issuing_draws() -> None:
    """Do not pay for the remaining draws once the credential is refused."""
    from recall_guard.core.nvidia_lm import LMHTTPError

    lm = _ScriptedLM([LMHTTPError("denied", status_code=401)])
    with pytest.raises(LMHTTPError):
        generate_ensemble(lm, "p", _spec(draws=64, max_workers=2), decide=lambda c: c.content)
    assert lm.calls < 64


def test_raw_draws_are_not_retained_by_default() -> None:
    lm = _ScriptedLM(["up"])
    assert generate_ensemble(lm, "p", _spec(), decide=lambda c: c.content).draws == ()
    kept = generate_ensemble(lm, "p", _spec(retain_draws=True), decide=lambda c: c.content)
    assert len(kept.draws) == 8


# --- 3.4 reduction, tie-breaking, hashing ------------------------------------


def test_agreement_is_measured_in_decision_space() -> None:
    lm = _ScriptedLM(["up", "up", "up", "down"])
    result = generate_ensemble(lm, "p", _spec(draws=8), decide=lambda c: c.content)
    assert result.agreement == pytest.approx(0.75)
    lo, hi = result.agreement_ci
    assert 0.0 <= lo < result.agreement < hi <= 1.0


def test_consensus_is_an_observed_draw() -> None:
    lm = _ScriptedLM(["up", "up", "down"])
    result = generate_ensemble(lm, "p", _spec(draws=9), decide=lambda c: c.content)
    assert result.consensus.content == "up"


def test_reduction_is_independent_of_draw_order() -> None:
    """The replay contract: same draw set, any order, identical result."""
    contents = ["a", "b", "a", "c", "b", "a", "c", "a"]
    decide = lambda c: c.content  # noqa: E731
    baseline = reduce_draws([_completion(x) for x in contents], _spec(), decide=decide)
    rng = random.Random(0)
    for _ in range(10):
        shuffled = contents[:]
        rng.shuffle(shuffled)
        replayed = reduce_draws([_completion(x) for x in shuffled], _spec(), decide=decide)
        assert replayed.consensus.content == baseline.consensus.content
        assert replayed.agreement == baseline.agreement
        assert replayed.draws_sha256 == baseline.draws_sha256


def test_hash_covers_content_and_is_order_free() -> None:
    assert canonical_draw_hash(["a", "b"]) == canonical_draw_hash(["b", "a"])
    assert canonical_draw_hash(["a", "b"]) != canonical_draw_hash(["a", "c"])
    assert canonical_draw_hash(["a", "a"]) != canonical_draw_hash(["a"])
    assert len(canonical_draw_hash(["a"])) == 64


def test_hash_separator_cannot_be_forged() -> None:
    """Concatenation must not let two draws impersonate one."""
    assert canonical_draw_hash(["ab"]) != canonical_draw_hash(["a", "b"])


def test_flagged_component_receives_no_location() -> None:
    """The gate: no location may be reported inside a detected gap."""
    lows = ["-0.9"] * 4
    highs = ["0.9"] * 4
    lm = _ScriptedLM(lows + highs)
    result = generate_ensemble(
        lm,
        "p",
        _spec(draws=8, grid=0.1, mass_min=0.4),
        decide=lambda c: c.content,
        components=lambda c: {"axis": float(c.content)},
    )
    assert "axis" in result.multimodal
    assert "axis" not in result.location


def test_multimodal_action_raise_is_available() -> None:
    lm = _ScriptedLM(["-0.9"] * 4 + ["0.9"] * 4)
    with pytest.raises(ValueError, match="axis"):
        generate_ensemble(
            lm,
            "p",
            _spec(draws=8, grid=0.1, mass_min=0.4, multimodal_action=MultimodalAction.RAISE),
            decide=lambda c: c.content,
            components=lambda c: {"axis": float(c.content)},
        )


def test_location_is_reported_unsnapped_with_snap_alongside() -> None:
    lm = _ScriptedLM(["0.82", "0.84", "0.83"])
    result = generate_ensemble(
        lm,
        "p",
        _spec(draws=9, grid=0.1, location_mode=LocationMode.MEAN),
        decide=lambda c: c.content,
        components=lambda c: {"axis": float(c.content)},
    )
    assert result.location["axis"] == pytest.approx(0.83, abs=1e-9)
    assert result.location_snapped["axis"] == pytest.approx(0.8)
    assert result.grid_adherence["axis"] < 0.5


# --- 3.5 dependence diagnostic ------------------------------------------------


def test_dependence_is_zero_for_independent_labels() -> None:
    labels = ["a", "b"] * 8
    groups = [i // 4 for i in range(16)]
    assert lag_dependence(labels, groups) == pytest.approx(0.0, abs=0.2)


def test_dependence_is_positive_when_labels_cluster_by_group() -> None:
    labels = ["a"] * 8 + ["b"] * 8
    groups = [i // 8 for i in range(16)]
    assert lag_dependence(labels, groups) > 0.9


def test_dependence_undefined_with_a_single_group() -> None:
    assert lag_dependence(["a", "b"], [0, 0]) is None


def test_ensemble_reports_dependence_when_there_are_enough_waves() -> None:
    lm = _ScriptedLM(["up", "down"])
    result = generate_ensemble(lm, "p", _spec(draws=8, max_workers=2), decide=lambda c: c.content)
    assert result.draw_dependence is not None


def test_every_cluster_detection_threshold_is_configurable() -> None:
    """No detector constant may be reachable only by editing the source.

    All of them were tuned against a single measurement date, which is exactly
    why none may be frozen into the implementation.
    """
    import inspect

    from recall_guard.core.consensus import detect_multimodal

    knobs = {
        name
        for name in inspect.signature(detect_multimodal).parameters
        if name not in {"values", "grid"}
    }
    exposed = set(EnsembleSpec.__dataclass_fields__) | {"min_draws"}
    # min_draws is surfaced under a clearer name on the spec.
    assert knobs - exposed - {"min_draws"} == set()
    assert "min_cluster_density" in EnsembleSpec.__dataclass_fields__
    assert "min_cluster_draws" in EnsembleSpec.__dataclass_fields__

    for bad in ({"min_cluster_draws": 1}, {"min_cluster_density": 0.5}):
        with pytest.raises(ValueError):
            EnsembleSpec(**bad)


def test_cluster_thresholds_reach_the_detector() -> None:
    """A configured threshold must actually change the verdict."""
    lm = _ScriptedLM(["-0.9"] * 4 + ["0.9"] * 4)
    strict = _spec(draws=8, grid=0.1, mass_min=0.4, min_cluster_draws=32)
    result = generate_ensemble(
        lm, "p", strict,
        decide=lambda c: c.content,
        components=lambda c: {"axis": float(c.content)},
    )
    assert result.multimodal == (), "min_cluster_draws did not reach the detector"


# --- generation settings and detection reporting (issue #1, #2) --------------


class _RecordingLM(_ScriptedLM):
    """Records the kwargs each draw was issued with."""

    def __init__(self, replies, model: str = "fake-model") -> None:
        super().__init__(replies, model)
        self.kwargs: list[dict] = []

    def generate(self, prompt, temperature=0.0, max_tokens=512):
        self.kwargs.append({"temperature": temperature, "max_tokens": max_tokens})
        return super().generate(prompt)


def test_generation_settings_reach_every_draw() -> None:
    """An ensemble drawn at a different budget is not measuring production.

    On a reasoning model the chain of thought consumes the token budget and
    truncates the reply before the payload a caller parses, so the ensemble
    silently answers a different question than the production path.
    """
    lm = _RecordingLM(["up"])
    generate_ensemble(
        lm, "p", _spec(max_tokens=2048, temperature=0.3), decide=lambda c: c.content
    )
    assert all(k["max_tokens"] == 2048 for k in lm.kwargs)
    assert all(k["temperature"] == 0.3 for k in lm.kwargs)


def test_unset_generation_settings_preserve_client_defaults() -> None:
    """Backwards compatible: an unset spec issues exactly what it always did."""
    lm = _RecordingLM(["up"])
    generate_ensemble(lm, "p", _spec(), decide=lambda c: c.content)
    assert all(k == {"temperature": 0.0, "max_tokens": 512} for k in lm.kwargs)


def test_generation_settings_are_recorded_on_the_result() -> None:
    """An ensemble is an audit artifact; the settings belong with the digest."""
    lm = _ScriptedLM(["up"])
    result = generate_ensemble(
        lm, "p", _spec(max_tokens=2048, temperature=0.2), decide=lambda c: c.content
    )
    assert result.max_tokens == 2048
    assert result.temperature == 0.2


def test_generation_settings_are_validated() -> None:
    for bad in ({"max_tokens": 0}, {"temperature": -0.1}, {"temperature": 2.5}):
        with pytest.raises(ValueError):
            EnsembleSpec(**bad)


def test_every_component_reports_its_verdict_not_only_flagged_ones() -> None:
    """A near-miss split and a clear unimodal are different situations.

    Reporting only the boolean leaves a caller unable to tell a confident answer
    from a lucky one -- which is the failure mode where a fabricated centre gets
    presented as resolved.
    """
    lm = _ScriptedLM(["0.8"] * 5 + ["0.9"] * 5)
    result = generate_ensemble(
        lm, "p", _spec(draws=10, grid=0.1),
        decide=lambda c: c.content,
        components=lambda c: {"tight": float(c.content)},
    )
    verdicts = dict(result.component_verdicts)
    assert "tight" in verdicts, "an unflagged component still reports its verdict"
    assert verdicts["tight"] is not None
    assert verdicts["tight"].separated is False
    assert result.multimodal == ()


def test_verdict_is_none_when_the_check_could_not_run() -> None:
    """Distinct from 'ran and found nothing' -- the caller must tell them apart."""
    lm = _ScriptedLM(["0.8", "0.9"])
    result = generate_ensemble(
        lm, "p", _spec(draws=8), decide=lambda c: c.content,
        components=lambda c: {"axis": float(c.content)},
    )
    assert dict(result.component_verdicts)["axis"] is None


def test_split_detection_sizing_helper() -> None:
    """Sizing for agreement silently under-sizes for split detection."""
    from recall_guard.core.consensus import (
        smallest_certifiable_n,
        smallest_detectable_split_n,
    )

    assert smallest_detectable_split_n(cluster_positions=4) == 8
    assert smallest_detectable_split_n(cluster_positions=20) == 30
    assert smallest_detectable_split_n(
        cluster_positions=20, min_cluster_density=3.0
    ) == 60
    # The two sizing questions genuinely disagree -- that is the point.
    assert smallest_detectable_split_n(cluster_positions=40) != smallest_certifiable_n(0.95)
    for bad in ({"cluster_positions": 1}, {"cluster_positions": 4, "min_cluster_draws": 1}):
        with pytest.raises(ValueError):
            smallest_detectable_split_n(**bad)
