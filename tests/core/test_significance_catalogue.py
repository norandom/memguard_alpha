"""Statistics of the qualified-model-catalogue spec, checked against known answers.

Covers requirements 4.3, 5.3, 6.1 to 6.6 and 6.9 to 6.11: the ``Statistic``
record, the prompt-clustered AUC interval (number for number the one of
``scripts/finance_sim/calibrate.py``), the cluster-level label permutation test
for AUC, the paired AUC difference on shared resamples, the agreement
statistics, the block-level permutation test of a mean difference and the
detectable effect from the bootstrap standard error.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from recall_guard.core.significance import (
    CLUSTERED_AUC_METHOD,
    Interval,
    Statistic,
    TestResult,
    agreement,
    clustered_auc_interval,
    detectable_effect,
    paired_auc_difference,
    permutation_auc_test,
    permutation_mean_difference,
    rank_auc,
)

Z_SUM = 1.959963984540054 + 0.8416212335729143


def _clustered_scores(seed: int = 3, n_clusters: int = 8, draws: int = 12):
    """Rows of ``2 * n_clusters`` prompts, ``draws`` each, with a prompt-level shift."""
    rng = np.random.default_rng(seed)
    scores, labels, clusters = [], [], []
    for index in range(2 * n_clusters):
        label = index % 2
        centre = 0.6 * label + rng.normal(0.0, 0.5)
        scores.extend(np.round(centre + rng.normal(0.0, 0.1, size=draws), 2))
        labels.extend([label] * draws)
        clusters.extend([f"prompt-{index}"] * draws)
    return np.asarray(scores), np.asarray(labels), np.asarray(clusters)


def _statistic(**changes: object) -> Statistic:
    fields: dict[str, object] = {
        "name": "mean_difference",
        "estimand": "mean of a minus b over prompts",
        "formula": "sum(a_i - b_i) / n",
        "inputs": {"n_rows": 4, "n_clusters": 4},
        "unit": "prompt",
        "n": 4,
        "independent_units": True,
        "value": 0.125,
        "interval": Interval(lo=0.0, hi=0.25, confidence=0.95, method="cluster percentile bootstrap"),
        "test": None,
        "effect_size": 0.125,
        "detectable_effect": 0.3,
        "by_construction": False,
        "missing": None,
    }
    return Statistic(**(fields | changes))


# ---------------------------------------------------------------------------
# Statistic record (6.1, 6.2, 6.3, 6.9, 6.10)
# ---------------------------------------------------------------------------


def test_statistic_holds_its_calculation() -> None:
    statistic = _statistic()
    assert statistic.estimand == "mean of a minus b over prompts"
    assert statistic.formula == "sum(a_i - b_i) / n"
    assert dict(statistic.inputs) == {"n_rows": 4, "n_clusters": 4}
    assert (statistic.unit, statistic.n, statistic.independent_units) == ("prompt", 4, True)
    assert statistic.value == 0.125
    assert statistic.missing is None
    with pytest.raises(TypeError):
        statistic.inputs["n_rows"] = 5  # type: ignore[index]


def test_statistic_without_a_value_names_the_missing_input() -> None:
    absent = _statistic(value=None, interval=None, effect_size=None, detectable_effect=None, missing="a")
    assert absent.value is None
    assert absent.missing == "a"
    with pytest.raises(ValueError, match="missing"):
        _statistic(value=None, interval=None, effect_size=None, detectable_effect=None)
    with pytest.raises(ValueError, match="missing"):
        _statistic(missing="a")
    with pytest.raises(ValueError, match="no value"):
        _statistic(value=None, missing="a")  # an interval without a value


def test_statistic_refuses_invalid_fields() -> None:
    with pytest.raises(ValueError, match="unit"):
        _statistic(unit="token")
    with pytest.raises(ValueError, match="estimand"):
        _statistic(estimand="")
    with pytest.raises(ValueError, match="formula"):
        _statistic(formula="")
    with pytest.raises(ValueError, match="'n'"):
        _statistic(n=-1)
    with pytest.raises(ValueError, match="value"):
        _statistic(value=math.nan)
    result = TestResult(statistic=0.0, interval=None, p_value=1.0, method="none", parameters={})
    with pytest.raises(ValueError, match="by construction"):
        _statistic(by_construction=True, test=result)


# ---------------------------------------------------------------------------
# Detectable effect (6.11)
# ---------------------------------------------------------------------------


def test_detectable_effect_is_the_sum_of_two_normal_quantiles_times_the_standard_error() -> None:
    assert detectable_effect(0.1) == pytest.approx(Z_SUM * 0.1, rel=1e-12)
    assert detectable_effect(0.0) == 0.0
    assert detectable_effect(0.1, power=0.9, alpha=0.01) > detectable_effect(0.1)
    with pytest.raises(ValueError):
        detectable_effect(-0.1)
    with pytest.raises(ValueError):
        detectable_effect(0.1, power=1.0)


# ---------------------------------------------------------------------------
# Rank AUC and the prompt-clustered interval (6.4)
# ---------------------------------------------------------------------------


def test_rank_auc_equals_roc_auc_score_with_ties() -> None:
    scores, labels, _ = _clustered_scores()
    assert len(np.unique(scores)) < len(scores)  # the constructed scores have ties
    assert rank_auc(scores[labels == 1], scores[labels == 0]) == pytest.approx(roc_auc_score(labels, scores), abs=1e-12)
    assert rank_auc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 1.0
    assert rank_auc(np.array([1.0, 1.0]), np.array([1.0])) == 0.5


def test_clustered_interval_reproduces_the_calibration_script_number_for_number() -> None:
    from scripts.finance_sim.calibrate import (
        CLUSTERED_METHOD,
        INTERVAL_CONFIDENCE,
        INTERVAL_RESAMPLES,
        INTERVAL_SEED,
        _clustered_interval,
        _rank_auc,
    )

    scores, labels, clusters = _clustered_scores()
    names = list(dict.fromkeys(clusters.tolist()))
    groups = [(int(labels[clusters == name][0]), scores[clusters == name]) for name in names]
    expected = _clustered_interval(groups)

    interval = clustered_auc_interval(scores, labels, clusters)

    assert (INTERVAL_RESAMPLES, INTERVAL_CONFIDENCE, INTERVAL_SEED) == (1000, 0.95, 0)
    assert interval.lo == expected["ci_low"]
    assert interval.hi == expected["ci_high"]
    assert interval.confidence == expected["confidence"]
    assert interval.method == CLUSTERED_METHOD == CLUSTERED_AUC_METHOD
    assert rank_auc(scores[labels == 1], scores[labels == 0]) == expected["auc"]
    assert rank_auc(scores[labels == 1], scores[labels == 0]) == _rank_auc(scores[labels == 1], scores[labels == 0])
    explicit = clustered_auc_interval(scores, labels, clusters, n_resamples=1000, confidence=0.95, seed=0)
    assert explicit == interval


def test_clustered_interval_resamples_prompts_not_draws() -> None:
    scores, labels, clusters = _clustered_scores()
    clustered = clustered_auc_interval(scores, labels, clusters)
    by_draw = clustered_auc_interval(scores, labels, np.arange(scores.size))
    assert clustered.hi - clustered.lo > 2 * (by_draw.hi - by_draw.lo)


def test_clustered_interval_is_stable_under_its_seed() -> None:
    scores, labels, clusters = _clustered_scores()
    first = clustered_auc_interval(scores, labels, clusters, n_resamples=200, seed=7)
    assert clustered_auc_interval(scores, labels, clusters, n_resamples=200, seed=7) == first
    assert clustered_auc_interval(scores, labels, clusters, n_resamples=200, seed=8) != first


def test_clustered_interval_refuses_unusable_input() -> None:
    scores, labels, clusters = _clustered_scores()
    with pytest.raises(ValueError, match="both classes"):
        clustered_auc_interval(scores, np.ones_like(labels), clusters)
    mixed = labels.copy()
    mixed[0] = 1 - mixed[0]
    with pytest.raises(ValueError, match="one label"):
        clustered_auc_interval(scores, mixed, clusters)
    with pytest.raises(ValueError, match="0 or 1"):
        clustered_auc_interval(scores, labels + 1, clusters)
    with pytest.raises(ValueError, match="aligned"):
        clustered_auc_interval(scores[:-1], labels, clusters)
    with pytest.raises(ValueError, match="n_resamples"):
        clustered_auc_interval(scores, labels, clusters, n_resamples=0)
    with pytest.raises(ValueError, match="seed"):
        clustered_auc_interval(scores, labels, clusters, seed=1.5)


# ---------------------------------------------------------------------------
# Label permutation test for AUC, at cluster level (6.5)
# ---------------------------------------------------------------------------


def _separated(n_clusters_per_class: int = 3, draws: int = 30):
    """Perfectly separated classes: every memorized draw scores above every control draw."""
    scores, labels, clusters = [], [], []
    for index in range(2 * n_clusters_per_class):
        label = int(index < n_clusters_per_class)
        scores.extend(label * 10.0 + index + np.arange(draws) / (10.0 * draws))
        labels.extend([label] * draws)
        clusters.extend([index] * draws)
    return np.asarray(scores), np.asarray(labels), np.asarray(clusters)


def test_permutation_auc_test_permutes_whole_clusters() -> None:
    scores, labels, clusters = _separated()
    result = permutation_auc_test(scores, labels, clusters)

    assert result.statistic == 1.0
    # 20 ways to label 6 prompts 3 and 3, one of them the observed: exact p = 1/20.
    # Permuting the 180 draws instead would give p near 1/2001.
    assert result.p_value == pytest.approx(1 / 20, abs=0.02)
    assert result.interval is None
    assert "cluster" in result.method and "no information" in result.method
    assert result.parameters["n_clusters"] == 6
    assert result.parameters["n"] == 180
    assert result.parameters["n_permutations"] == 2000
    assert result.parameters["alternative"] == "greater"

    by_draw = permutation_auc_test(scores, labels, np.arange(scores.size))
    assert by_draw.p_value == pytest.approx(1 / 2001)


def test_permutation_auc_test_known_answers_and_seed_stability() -> None:
    scores, labels, clusters = _clustered_scores()
    result = permutation_auc_test(scores, labels, clusters, n_permutations=500, seed=4)
    assert result.statistic == pytest.approx(roc_auc_score(labels, scores), abs=1e-12)
    assert result == permutation_auc_test(scores, labels, clusters, n_permutations=500, seed=4)
    assert result.p_value != permutation_auc_test(scores, labels, clusters, n_permutations=500, seed=5).p_value

    flat = permutation_auc_test(np.ones(scores.size), labels, clusters, n_permutations=100)
    assert (flat.statistic, flat.p_value) == (0.5, 1.0)

    with pytest.raises(ValueError, match="one label"):
        permutation_auc_test(scores, np.arange(scores.size) % 2, clusters)
    with pytest.raises(ValueError, match="n_permutations"):
        permutation_auc_test(scores, labels, clusters, n_permutations=0)


# ---------------------------------------------------------------------------
# Paired AUC difference on shared resamples (4.3, 6.4, 6.5, 6.6, 6.11)
# ---------------------------------------------------------------------------


def test_paired_auc_difference_known_answers() -> None:
    scores, labels, clusters = _separated()
    difference, interval, test = paired_auc_difference(scores, np.zeros(scores.size), labels, clusters)
    assert difference == 0.5
    assert (interval.lo, interval.hi, interval.confidence) == (0.5, 0.5, 0.95)
    assert test.statistic == 0.5
    assert test.interval == interval
    assert test.p_value == pytest.approx(2 / 2001)
    assert "equal AUC" in test.method
    assert test.parameters["n_clusters"] == 6
    assert test.parameters["standard_error"] == 0.0
    assert test.parameters["detectable_effect"] == 0.0

    same, same_interval, same_test = paired_auc_difference(scores, scores, labels, clusters, n_resamples=100)
    assert same == 0.0
    assert (same_interval.lo, same_interval.hi) == (0.0, 0.0)
    assert same_test.p_value == 1.0


def test_paired_auc_difference_recomputes_both_aucs_on_the_same_resample() -> None:
    scores, labels, clusters = _clustered_scores()
    # AUC(-s) = 1 - AUC(s) on every resample, so the difference is 2 AUC(s) - 1 resample by resample.
    difference, interval, test = paired_auc_difference(scores, -scores, labels, clusters, n_resamples=1000, seed=0)
    single = clustered_auc_interval(scores, labels, clusters, n_resamples=1000, seed=0)

    assert difference == pytest.approx(2 * roc_auc_score(labels, scores) - 1, abs=1e-12)
    assert interval.lo == pytest.approx(2 * single.lo - 1, abs=1e-12)
    assert interval.hi == pytest.approx(2 * single.hi - 1, abs=1e-12)
    standard_error = test.parameters["standard_error"]
    assert standard_error > 0
    assert test.parameters["detectable_effect"] == pytest.approx(Z_SUM * standard_error, rel=1e-12)
    assert (test.parameters["power"], test.parameters["alpha"]) == (0.8, 0.05)


def test_paired_auc_difference_is_stable_under_its_seed() -> None:
    scores, labels, clusters = _clustered_scores()
    other = np.random.default_rng(9).normal(size=scores.size)
    first = paired_auc_difference(scores, other, labels, clusters, n_resamples=200, seed=2)
    assert paired_auc_difference(scores, other, labels, clusters, n_resamples=200, seed=2) == first
    assert paired_auc_difference(scores, other, labels, clusters, n_resamples=200, seed=3)[1] != first[1]
    with pytest.raises(ValueError, match="aligned"):
        paired_auc_difference(scores, other[:-1], labels, clusters)


# ---------------------------------------------------------------------------
# Agreement of two estimates (4.3, 6.1 to 6.6, 6.9, 6.11)
# ---------------------------------------------------------------------------

AGREEMENT_KEYS = {
    "mean_difference",
    "lower_limit_of_agreement",
    "upper_limit_of_agreement",
    "mean_absolute_difference",
    "rank_correlation",
}


def test_agreement_known_answers() -> None:
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([1.5, 1.5, 3.5, 3.0])
    result = agreement(a, b, np.arange(4))

    assert set(result) == AGREEMENT_KEYS
    spread = math.sqrt((0.625**2 * 2 + 0.375**2 + 0.875**2) / 3)  # sd of (-0.5, 0.5, -0.5, 1.0) around 0.125
    assert result["mean_difference"].value == pytest.approx(0.125)
    assert result["lower_limit_of_agreement"].value == pytest.approx(0.125 - 1.96 * spread)
    assert result["upper_limit_of_agreement"].value == pytest.approx(0.125 + 1.96 * spread)
    assert result["mean_absolute_difference"].value == pytest.approx(0.625)
    assert result["rank_correlation"].value == pytest.approx(3.5 / math.sqrt(22.5))


def test_agreement_statistics_expose_their_calculation() -> None:
    rng = np.random.default_rng(1)
    b = rng.normal(size=40)
    a = b + rng.normal(0.05, 0.2, size=40)
    result = agreement(a, b, np.arange(40))

    for name, statistic in result.items():
        assert statistic.name == name
        assert statistic.estimand and statistic.formula
        assert dict(statistic.inputs)["n_rows"] == 40
        assert dict(statistic.inputs)["n_clusters"] == 40
        assert (statistic.unit, statistic.n, statistic.independent_units) == ("prompt", 40, True)
        assert statistic.missing is None and not statistic.by_construction
        assert statistic.interval is not None
        assert statistic.interval.lo <= statistic.value <= statistic.interval.hi
        assert statistic.interval.confidence == 0.95 and "cluster" in statistic.interval.method

    mean_difference = result["mean_difference"]
    assert mean_difference.inputs["sum_of_differences"] == pytest.approx(float(np.sum(a - b)))
    assert mean_difference.test is not None
    assert 0.0 < mean_difference.test.p_value <= 1.0
    assert "mean difference is zero" in mean_difference.test.method
    standard_error = mean_difference.test.parameters["standard_error"]
    assert standard_error == pytest.approx(0.2 / math.sqrt(40), rel=0.5)
    assert mean_difference.detectable_effect == pytest.approx(Z_SUM * standard_error, rel=1e-12)
    assert mean_difference.effect_size == pytest.approx(mean_difference.value / float(np.std(a - b, ddof=1)))
    assert result["rank_correlation"].effect_size == result["rank_correlation"].value
    assert agreement(a, b, np.arange(40), unit="month")["mean_difference"].unit == "month"


def test_agreement_resamples_clusters_and_is_stable_under_its_seed() -> None:
    rng = np.random.default_rng(2)
    b = rng.normal(size=12)
    a = b + rng.normal(0.0, 0.3, size=12)
    single = agreement(a, b, np.arange(12), n_resamples=300, seed=5)
    # Twenty copies of every row inside its cluster add no information: the same
    # clusters are drawn, so the interval is the same. A row bootstrap would shrink it.
    repeated = agreement(np.repeat(a, 20), np.repeat(b, 20), np.repeat(np.arange(12), 20), n_resamples=300, seed=5)

    for name in ("mean_difference", "mean_absolute_difference"):
        assert repeated[name].value == pytest.approx(single[name].value)
        assert repeated[name].interval.lo == pytest.approx(single[name].interval.lo)
        assert repeated[name].interval.hi == pytest.approx(single[name].interval.hi)
    assert repeated["mean_difference"].n == 12
    assert repeated["mean_difference"].inputs["n_rows"] == 240
    assert repeated["mean_difference"].independent_units is False

    assert agreement(a, b, np.arange(12), n_resamples=300, seed=5) == single
    other = agreement(a, b, np.arange(12), n_resamples=300, seed=6)
    assert other["mean_difference"].interval != single["mean_difference"].interval
    assert other["mean_difference"].value == single["mean_difference"].value


def test_agreement_with_a_missing_input_gives_no_value() -> None:
    b = np.array([0.1, 0.2, 0.3])
    for result, missing in (
        (agreement(None, b, np.arange(3)), "a"),
        (agreement(b, None, np.arange(3)), "b"),
        (agreement(b, b, None), "clusters"),
        (agreement(np.array([]), np.array([]), np.array([])), "a"),
    ):
        assert set(result) == AGREEMENT_KEYS
        for statistic in result.values():
            assert statistic.value is None
            assert statistic.missing == missing
            assert statistic.interval is None and statistic.test is None
            assert statistic.effect_size is None and statistic.detectable_effect is None

    one_cluster = agreement(b, b + 0.1, np.zeros(3))
    assert all(statistic.value is None for statistic in one_cluster.values())
    assert "two clusters" in one_cluster["mean_difference"].missing

    constant = agreement(np.array([1.0, 1.0, 1.0]), b, np.arange(3), n_resamples=50)
    assert constant["rank_correlation"].value is None
    assert "variation" in constant["rank_correlation"].missing
    assert constant["mean_difference"].value == pytest.approx(0.8)

    with pytest.raises(ValueError, match="aligned"):
        agreement(b, b[:-1], np.arange(3))


# ---------------------------------------------------------------------------
# One-sided permutation test of a mean difference, at block level (5.3, 6.5)
# ---------------------------------------------------------------------------


def _months(rows: int = 25):
    """Six months, three inside the cutoff (group 1) with higher values, ``rows`` rows each."""
    values, groups, blocks = [], [], []
    for month in range(6):
        inside = int(month < 3)
        values.extend(inside * 1.0 + month / 100.0 + np.arange(rows) / (1000.0 * rows))
        groups.extend([inside] * rows)
        blocks.extend([f"2024-{month + 1:02d}"] * rows)
    return np.asarray(values), np.asarray(groups), np.asarray(blocks)


def test_permutation_mean_difference_permutes_whole_blocks() -> None:
    values, groups, blocks = _months()
    result = permutation_mean_difference(values, groups, blocks)

    assert result.statistic == pytest.approx(values[groups == 1].mean() - values[groups == 0].mean())
    # 20 ways to put 3 of 6 months inside, one of them the observed: exact p = 1/20.
    assert result.p_value == pytest.approx(1 / 20, abs=0.02)
    assert result.interval is None
    assert "block" in result.method and "same mean" in result.method
    assert result.parameters["alternative"] == "greater"
    assert result.parameters["n_blocks"] == 6
    assert result.parameters["n"] == 150
    assert result.parameters["n_permutations"] == 2000

    by_row = permutation_mean_difference(values, groups, np.arange(values.size))
    assert by_row.p_value == pytest.approx(1 / 2001)


def test_permutation_mean_difference_alternatives_and_seed_stability() -> None:
    values, groups, blocks = _months()
    less = permutation_mean_difference(values, groups, blocks, alternative="less")
    assert less.p_value == 1.0
    two_sided = permutation_mean_difference(values, groups, blocks, alternative="two-sided")
    assert two_sided.p_value == pytest.approx(2 / 20, abs=0.03)  # the observed labelling and its mirror

    first = permutation_mean_difference(values, groups, blocks, n_permutations=300, seed=11)
    assert permutation_mean_difference(values, groups, blocks, n_permutations=300, seed=11) == first
    assert permutation_mean_difference(values, groups, blocks, n_permutations=300, seed=12).p_value != first.p_value

    equal = permutation_mean_difference(np.ones(values.size), groups, blocks, n_permutations=50)
    assert (equal.statistic, equal.p_value) == (0.0, 1.0)


def test_permutation_mean_difference_refuses_unusable_input() -> None:
    values, groups, blocks = _months()
    with pytest.raises(ValueError, match="alternative"):
        permutation_mean_difference(values, groups, blocks, alternative="larger")
    with pytest.raises(ValueError, match="one group"):
        permutation_mean_difference(values, np.arange(values.size) % 2, blocks)
    with pytest.raises(ValueError, match="both groups"):
        permutation_mean_difference(values, np.zeros_like(groups), blocks)
    with pytest.raises(ValueError, match="aligned"):
        permutation_mean_difference(values[:-1], groups, blocks)
    with pytest.raises(ValueError, match="finite"):
        permutation_mean_difference(np.where(np.arange(values.size) == 0, np.nan, values), groups, blocks)
