"""Recorded draws and the distribution summary, checked against hand-computed values."""

from __future__ import annotations

import json
import math
import random

import pytest

from recall_guard.core.consensus import Tail, detect_multimodal, wilson_interval
from recall_guard.core.decision import ComponentSpec, DecisionDefinition, parse_json_components
from recall_guard.core.distribution import (
    AGREEMENT_CONFIDENCE,
    AGREEMENT_TAIL,
    INDEPENDENCE_NOTE,
    QUANTILE_LEVELS,
    ComponentDistribution,
    DistributionSummary,
    RecordedDraw,
    summarize_distribution,
)


def _spread(vector):
    return vector["gold"] - vector["equity"]


def _definition(**overrides) -> DecisionDefinition:
    fields = {
        "name": "toy",
        "components": (
            ComponentSpec("equity", -1.0, 1.0, grid=0.1),
            ComponentSpec("gold", -1.0, 1.0, grid=0.1),
        ),
        "parse": parse_json_components,
        "summary": _spread,
        "decide": lambda vector: _spread(vector) > 0,
    }
    fields.update(overrides)
    return DecisionDefinition(**fields)


def _draw(index, equity, gold, *, estimate=None, sha=None) -> RecordedDraw:
    return RecordedDraw(
        draw_index=index,
        temperature=0.7,
        seed=100 + index,
        response_sha256=sha or f"{index:064x}",
        parsed={"equity": equity, "gold": gold},
        failure_code=None,
        estimate=estimate,
        evidence_ref=f"evidence/{index}.json",
    )


def _failed(index, code="parse") -> RecordedDraw:
    return RecordedDraw(
        draw_index=index,
        temperature=0.7,
        seed=100 + index,
        response_sha256=f"{index:064x}",
        parsed=None,
        failure_code=code,
        estimate=None,
        evidence_ref=None,
    )


# Fixed set: draws 0 and 3 give the same answer, so four distinct answers.
_EQUITY = [0.2, -0.1, 0.4, 0.2, 0.0]
_GOLD = [0.5, 0.3, -0.2, 0.5, 0.1]
_ESTIMATES = [0.1, 0.3, 0.2, 0.5, 0.4]


def _fixed_draws() -> list[RecordedDraw]:
    return [
        _draw(i, e, g, estimate=p) for i, (e, g, p) in enumerate(zip(_EQUITY, _GOLD, _ESTIMATES, strict=True))
    ]


def _component(summary: DistributionSummary, name: str) -> ComponentDistribution:
    return next(c for c in summary.components if c.name == name)


def _quantiles(mapping):
    return [mapping[level] for level in QUANTILE_LEVELS]


# -- per component -------------------------------------------------------------


def test_component_quantiles_spread_and_sign_match_hand_computation():
    summary = summarize_distribution(_fixed_draws(), _definition())

    assert [c.name for c in summary.components] == ["equity", "gold"]
    equity = _component(summary, "equity")
    # Sorted [-0.1, 0.0, 0.2, 0.2, 0.4]; linear position h = 4p.
    assert _quantiles(equity.quantiles) == pytest.approx([-0.08, 0.0, 0.2, 0.2, 0.36])
    assert equity.iqr == pytest.approx(0.2)
    assert equity.sd == pytest.approx(math.sqrt(0.152 / 4))
    # Median 0.2 is positive; 0.2, 0.4, 0.2 share its sign, 0.0 and -0.1 do not.
    assert equity.sign_agreement == pytest.approx(0.6)

    gold = _component(summary, "gold")
    # Sorted [-0.2, 0.1, 0.3, 0.5, 0.5].
    assert _quantiles(gold.quantiles) == pytest.approx([-0.14, 0.1, 0.3, 0.5, 0.5])
    assert gold.iqr == pytest.approx(0.4)
    assert gold.sign_agreement == pytest.approx(0.8)


def test_quantile_levels_are_the_declared_five():
    summary = summarize_distribution(_fixed_draws(), _definition())
    assert tuple(QUANTILE_LEVELS) == (5, 25, 50, 75, 95)
    for component in summary.components:
        assert tuple(component.quantiles) == (5, 25, 50, 75, 95)


def test_zero_counts_as_its_own_sign():
    draws = [_draw(i, e, 0.0) for i, e in enumerate([0.0, 0.0, 0.1, -0.1, 0.0])]
    summary = summarize_distribution(draws, _definition())
    # Median 0.0: only the three zeros share its sign.
    assert _component(summary, "equity").sign_agreement == pytest.approx(0.6)
    assert _component(summary, "gold").sign_agreement == 1.0


def test_first_draw_rank_is_the_mid_rank_of_draw_zero():
    summary = summarize_distribution(_fixed_draws(), _definition())
    # equity draw 0 = 0.2: two values below, two tied (itself included) -> (2 + 1) / 5.
    assert _component(summary, "equity").first_draw_rank == pytest.approx(0.6)
    # gold draw 0 = 0.5: three below, two tied -> (3 + 1) / 5.
    assert _component(summary, "gold").first_draw_rank == pytest.approx(0.8)
    # summary draw 0 = 0.3 in [-0.6, 0.1, 0.3, 0.3, 0.4] -> (2 + 1) / 5.
    assert summary.summary_value is not None
    assert summary.summary_value.first_draw_rank == pytest.approx(0.6)


def test_first_draw_rank_is_none_when_draw_zero_failed():
    draws = [_failed(0)] + [_draw(i, e, g) for i, (e, g) in enumerate(zip(_EQUITY, _GOLD, strict=True), start=1)]
    summary = summarize_distribution(draws, _definition())
    assert all(c.first_draw_rank is None for c in summary.components)
    assert summary.summary_value is not None
    assert summary.summary_value.first_draw_rank is None


# -- summary value, agreement, estimates ---------------------------------------


def test_summary_value_distribution():
    summary = summarize_distribution(_fixed_draws(), _definition())
    value = summary.summary_value
    assert value is not None
    # gold - equity = [0.3, 0.4, -0.6, 0.3, 0.1]; sorted [-0.6, 0.1, 0.3, 0.3, 0.4].
    assert _quantiles(value.quantiles) == pytest.approx([-0.46, 0.1, 0.3, 0.3, 0.38])
    assert value.iqr == pytest.approx(0.2)
    assert value.sign_agreement == pytest.approx(0.8)
    # Five draws are below the split check's minimum, and no component is split.
    assert value.unresolved is None


def test_no_summary_mapping_gives_no_summary_value():
    summary = summarize_distribution(_fixed_draws(), _definition(summary=None))
    assert summary.summary_value is None


def test_decision_agreement_and_wilson_interval_come_from_the_ensemble_reduction():
    summary = summarize_distribution(_fixed_draws(), _definition())
    # spread > 0 holds for four of five draws.
    assert summary.decision_agreement == pytest.approx(0.8)
    expected = wilson_interval(4, 5, confidence=AGREEMENT_CONFIDENCE, tail=AGREEMENT_TAIL)
    assert summary.decision_agreement_ci == pytest.approx(expected)


def test_agreement_interval_is_pinned_to_one_sided_95_percent():
    assert AGREEMENT_CONFIDENCE == 0.95
    assert AGREEMENT_TAIL is Tail.ONE_SIDED
    ci = summarize_distribution(_fixed_draws(), _definition()).decision_agreement_ci
    assert ci == pytest.approx(wilson_interval(4, 5, confidence=0.95, tail=Tail.ONE_SIDED))
    # One-sided 95 percent bounds are the two ends of a 90 percent two-sided interval.
    assert ci == pytest.approx(wilson_interval(4, 5, confidence=0.90, tail=Tail.TWO_SIDED))


def test_no_decide_gives_no_agreement():
    summary = summarize_distribution(_fixed_draws(), _definition(decide=None))
    assert summary.decision_agreement is None
    assert summary.decision_agreement_ci is None


def test_estimate_quantiles_and_spread():
    summary = summarize_distribution(_fixed_draws(), _definition())
    assert summary.estimate_quantiles is not None
    assert _quantiles(summary.estimate_quantiles) == pytest.approx([0.12, 0.2, 0.3, 0.4, 0.48])
    assert summary.estimate_sd == pytest.approx(math.sqrt(0.10 / 4))


def test_missing_estimates_give_none():
    draws = [_draw(i, e, g) for i, (e, g) in enumerate(zip(_EQUITY, _GOLD, strict=True))]
    summary = summarize_distribution(draws, _definition())
    assert summary.estimate_quantiles is None
    assert summary.estimate_sd is None


# -- counts and distinct answers -----------------------------------------------


def test_counts_and_distinct_answers():
    draws = _fixed_draws() + [_failed(5), _failed(6, "timeout")]
    summary = summarize_distribution(draws, _definition())
    assert (summary.n_requested, summary.n_parsed, summary.n_failed) == (7, 5, 2)
    assert summary.distinct_answers == 4
    # Failed draws do not change any statistic of the parsed ones.
    clean = summarize_distribution(_fixed_draws(), _definition())
    assert summary.components == clean.components
    assert summary.summary_value == clean.summary_value


def test_identical_draws_give_one_distinct_answer():
    draws = [_draw(i, 0.3, -0.2, estimate=0.25, sha="a" * 64) for i in range(10)]
    summary = summarize_distribution(draws, _definition())
    assert summary.distinct_answers == 1
    equity = _component(summary, "equity")
    assert equity.sd == 0.0
    assert equity.iqr == 0.0
    assert equity.sign_agreement == 1.0
    assert equity.first_draw_rank == 0.5
    assert summary.decision_agreement == 1.0
    assert summary.estimate_sd == 0.0
    assert summary.lag1_dependence is None


def test_single_parsed_draw_has_zero_spread():
    summary = summarize_distribution([_draw(0, 0.4, 0.1, estimate=0.2)], _definition())
    equity = _component(summary, "equity")
    assert equity.sd == 0.0
    assert all(v == pytest.approx(0.4) for v in equity.quantiles.values())
    assert summary.estimate_sd == 0.0
    assert summary.lag1_dependence is None


# -- unresolved ------------------------------------------------------------------


def _split_draws() -> list[RecordedDraw]:
    values = [-0.5] * 8 + [0.5] * 8
    random.Random(7).shuffle(values)
    return [_draw(i, v, 0.0) for i, v in enumerate(values)]


def test_separated_clusters_mark_the_component_unresolved():
    draws = _split_draws()
    summary = summarize_distribution(draws, _definition())
    verdict = detect_multimodal([d.parsed["equity"] for d in draws], grid=0.1)
    assert verdict is not None and verdict.separated
    assert _component(summary, "equity").unresolved is True
    # gold is constant over 16 draws: the check ran and found no split.
    assert _component(summary, "gold").unresolved is False


def test_unresolved_uses_each_component_grid():
    definition = _definition(
        components=(ComponentSpec("equity", -1.0, 1.0, grid=None), ComponentSpec("gold", -1.0, 1.0, grid=0.1))
    )
    summary = summarize_distribution(_split_draws(), definition)
    # Without a lattice the separated-cluster check does not run, which is not "resolved".
    assert _component(summary, "equity").unresolved is None
    assert _component(summary, "gold").unresolved is False


def test_unresolved_is_none_without_a_grid():
    definition = _definition(
        components=(ComponentSpec("equity", -1.0, 1.0), ComponentSpec("gold", -1.0, 1.0))
    )
    summary = summarize_distribution(_split_draws(), definition)
    assert [c.unresolved for c in summary.components] == [None, None]
    assert summary.summary_value is not None
    assert summary.summary_value.unresolved is None


def test_unresolved_is_none_below_eight_parsed_draws():
    values = [-0.5, 0.5] * 3 + [-0.5]
    draws = [_draw(i, v, 0.0) for i, v in enumerate(values)] + [_failed(7), _failed(8)]
    summary = summarize_distribution(draws, _definition())
    assert summary.n_parsed == 7
    assert [c.unresolved for c in summary.components] == [None, None]


def test_unimodal_set_of_eight_or_more_is_resolved():
    values = [0.1, 0.2, 0.2, 0.3, 0.2, 0.1, 0.2, 0.3, 0.2, 0.2]
    summary = summarize_distribution([_draw(i, v, 0.4) for i, v in enumerate(values)], _definition())
    assert [c.unresolved for c in summary.components] == [False, False]
    # No component is split, and the summary value has no grid of its own.
    assert summary.summary_value is not None
    assert summary.summary_value.unresolved is None


def test_summary_value_is_unresolved_when_a_component_is_split():
    summary = summarize_distribution(_split_draws(), _definition())
    assert summary.summary_value is not None
    assert summary.summary_value.unresolved is True


# -- dependence and order --------------------------------------------------------


def test_lag1_dependence_of_the_summary_value():
    summary = summarize_distribution(_fixed_draws(), _definition())
    # x = [0.3, 0.4, -0.6, 0.3, 0.1]; pairs (x_t, x_t+1), Pearson.
    assert summary.lag1_dependence == pytest.approx(-0.29 / math.sqrt(0.66 * 0.61))


def test_lag1_dependence_falls_back_to_the_first_component():
    summary = summarize_distribution(_fixed_draws(), _definition(summary=None))
    # equity = [0.2, -0.1, 0.4, 0.2, 0.0]; a = [0.2, -0.1, 0.4, 0.2], b = [-0.1, 0.4, 0.2, 0.0].
    a = [0.2, -0.1, 0.4, 0.2]
    b = [-0.1, 0.4, 0.2, 0.0]
    ma, mb = sum(a) / 4, sum(b) / 4
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b, strict=True))
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    assert summary.lag1_dependence == pytest.approx(cov / math.sqrt(va * vb))


def test_lag1_dependence_needs_three_parsed_draws():
    draws = [_draw(0, 0.1, 0.2), _draw(1, 0.3, -0.4), _failed(2)]
    assert summarize_distribution(draws, _definition()).lag1_dependence is None


def test_independence_note_is_stated():
    summary = summarize_distribution(_fixed_draws(), _definition())
    assert summary.independence_note == INDEPENDENCE_NOTE
    assert "assume independent draws" in INDEPENDENCE_NOTE
    assert "not corrected" in INDEPENDENCE_NOTE


def test_summary_is_independent_of_input_order():
    draws = _fixed_draws() + [_failed(5)]
    shuffled = list(draws)
    random.Random(3).shuffle(shuffled)
    assert summarize_distribution(shuffled, _definition()) == summarize_distribution(draws, _definition())


# -- refusals --------------------------------------------------------------------


def test_empty_or_unparsed_draw_sets_are_refused():
    with pytest.raises(ValueError, match="empty"):
        summarize_distribution([], _definition())
    with pytest.raises(ValueError, match="no draw parsed"):
        summarize_distribution([_failed(0), _failed(1)], _definition())


def test_duplicate_draw_indices_are_refused():
    draws = [_draw(0, 0.1, 0.2), _draw(0, 0.3, 0.4)]
    with pytest.raises(ValueError, match="duplicate draw_index 0"):
        summarize_distribution(draws, _definition())


def test_parsed_vectors_must_match_the_definition():
    bad = RecordedDraw(
        draw_index=0,
        temperature=None,
        seed=None,
        response_sha256="0" * 64,
        parsed={"equity": 0.1},
        failure_code=None,
        estimate=None,
        evidence_ref=None,
    )
    with pytest.raises(ValueError, match="components"):
        summarize_distribution([bad], _definition())


# -- the record ------------------------------------------------------------------


@pytest.mark.parametrize(
    "draw",
    [
        _draw(3, -0.1, 0.7, estimate=0.42),
        _failed(4, "timeout"),
        RecordedDraw(
            draw_index=0,
            temperature=None,
            seed=None,
            response_sha256="f" * 64,
            parsed={"equity": 0.1, "gold": -0.30000000000000004},
            failure_code=None,
            estimate=None,
            evidence_ref=None,
        ),
    ],
)
def test_records_round_trip_unchanged(draw):
    restored = RecordedDraw.from_dict(json.loads(json.dumps(draw.to_dict())))
    assert restored == draw
    assert restored.to_dict() == draw.to_dict()


def test_record_parsed_vector_is_a_read_only_copy():
    source = {"equity": 0.1, "gold": 0.2}
    draw = RecordedDraw(
        draw_index=0,
        temperature=0.7,
        seed=1,
        response_sha256="0" * 64,
        parsed=source,
        failure_code=None,
        estimate=None,
        evidence_ref=None,
    )
    with pytest.raises(TypeError):
        draw.parsed["equity"] = 0.9  # type: ignore[index]
    source["equity"] = 0.9
    assert draw.parsed == {"equity": 0.1, "gold": 0.2}
    # Equality and the dict form are unaffected by the read-only wrapper.
    twin = RecordedDraw(
        draw_index=0,
        temperature=0.7,
        seed=1,
        response_sha256="0" * 64,
        parsed={"gold": 0.2, "equity": 0.1},
        failure_code=None,
        estimate=None,
        evidence_ref=None,
    )
    assert draw == twin
    data = draw.to_dict()
    assert type(data["parsed"]) is dict
    assert json.loads(json.dumps(data)) == data
    assert RecordedDraw.from_dict(data) == draw


def test_record_dict_has_exactly_the_record_fields():
    data = _draw(1, 0.1, 0.2, estimate=0.3).to_dict()
    assert set(data) == {
        "draw_index",
        "temperature",
        "seed",
        "response_sha256",
        "parsed",
        "failure_code",
        "estimate",
        "evidence_ref",
    }
    with pytest.raises(ValueError, match="unexpected"):
        RecordedDraw.from_dict({**data, "variant": "anonymised"})
    del data["seed"]
    with pytest.raises(ValueError, match="missing"):
        RecordedDraw.from_dict(data)


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"parsed": None}, "failure_code must be set exactly when parsed is None"),
        ({"failure_code": "parse"}, "failure_code must be set exactly when parsed is None"),
        ({"draw_index": -1}, "draw_index"),
        ({"draw_index": True}, "draw_index"),
        ({"estimate": float("nan")}, "estimate"),
        ({"parsed": {"equity": float("inf"), "gold": 0.0}}, "parsed"),
        ({"response_sha256": ""}, "response_sha256"),
        ({"seed": True}, "seed"),
        ({"seed": 1.5}, "seed"),
        ({"parsed": None, "failure_code": ""}, "failure_code must be a non-empty string"),
    ],
)
def test_invalid_records_are_refused(overrides, fragment):
    fields = {
        "draw_index": 0,
        "temperature": 0.0,
        "seed": 1,
        "response_sha256": "0" * 64,
        "parsed": {"equity": 0.1, "gold": 0.2},
        "failure_code": None,
        "estimate": 0.1,
        "evidence_ref": None,
    }
    fields.update(overrides)
    with pytest.raises(ValueError, match=fragment):
        RecordedDraw(**fields)
