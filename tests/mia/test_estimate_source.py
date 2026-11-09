"""Tests for `recall_guard.mia.estimate_source`.

Validates Requirements 3.1, 3.3, 9.1 and 9.2 and design.md "### mia" >
"#### EstimateSources": one feature set from top-k and from full-distribution
evidence, source labels with k, truncation of full evidence to any smaller k
and the probability mass the first k alternatives cover.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from recall_guard.core.full_distribution import FullDistributionEvidence
from recall_guard.core.nvidia_lm import TokenLogprob
from recall_guard.mia.estimate_source import (
    EstimateSource,
    SourcedFeatures,
    SourceLabel,
    features_from_full,
    features_from_top_k,
    top_k_mass,
    truncate_to_top_k,
)
from recall_guard.mia.features import compute_mia_features

CONTENT = "Rates rose by 25 basis points in March."

# Three response positions with three kept alternatives each. Position 2's
# actual token (id 5) is outside the kept alternatives (rank 4).
ACTUAL_IDS = (2, 4, 5)
ACTUAL_LOGPROBS = (-0.5, -3.0, -4.0)
ALTERNATIVES = (
    ((2, -0.5), (7, -1.5), (1, -2.0)),
    ((0, -0.3), (9, -1.8), (4, -3.0)),
    ((3, -0.2), (8, -2.5), (6, -2.9)),
)


def _evidence(alternatives=ALTERNATIVES, ids=ACTUAL_IDS, logprobs=ACTUAL_LOGPROBS) -> FullDistributionEvidence:
    n = len(ids)
    return FullDistributionEvidence(
        model_digest="sha256:" + "0" * 64,
        n_vocab=10,
        response_token_ids=ids,
        logprob=logprobs,
        entropy=(1.1, 1.4, 1.2)[:n],
        rank=(1, 3, 4)[:n],
        top_logprobs=alternatives,
    )


def _random_evidence(seed: int, *, positions: int = 12, n_vocab: int = 64, kept: int = 16) -> FullDistributionEvidence:
    """Evidence derived from random full vectors by a float64 log-softmax."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(scale=3.0, size=(positions, n_vocab))
    shifted = logits - logits.max(axis=1, keepdims=True)
    logp = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    ids, actual, entropy, rank, alternatives = [], [], [], [], []
    for row in logp:
        order = np.argsort(-row, kind="stable")
        token_id = int(rng.integers(n_vocab))
        ids.append(token_id)
        actual.append(min(float(row[token_id]), 0.0))
        entropy.append(max(float(-(np.exp(row) * row).sum()), 0.0))
        rank.append(int(np.nonzero(order == token_id)[0][0]) + 1)
        alternatives.append(tuple((int(t), min(float(row[t]), 0.0)) for t in order[:kept]))
    return FullDistributionEvidence(
        model_digest="sha256:" + "1" * 64,
        n_vocab=n_vocab,
        response_token_ids=tuple(ids),
        logprob=tuple(actual),
        entropy=tuple(entropy),
        rank=tuple(rank),
        top_logprobs=tuple(alternatives),
    )


def _hand_top_k(actual: tuple[float, ...], tops: list[list[float]]) -> list[TokenLogprob]:
    """Top-k evidence in the NVIDIA shape: dicts with 'token' and 'logprob'."""
    return [
        TokenLogprob(token="x", logprob=lp, top_logprobs=[{"token": "y", "logprob": v} for v in top])
        for lp, top in zip(actual, tops, strict=True)
    ]


# --- Truncation ---------------------------------------------------------


@pytest.mark.parametrize("k", [1, 2, 3])
def test_truncated_features_equal_hand_built_top_k(k: int) -> None:
    """Full evidence truncated to k gives the features of top-k evidence with the same alternatives."""
    hand = {
        1: [[-0.5], [-0.3], [-0.2]],
        2: [[-0.5, -1.5], [-0.3, -1.8], [-0.2, -2.5]],
        3: [[-0.5, -1.5, -2.0], [-0.3, -1.8, -3.0], [-0.2, -2.5, -2.9]],
    }[k]
    expected = compute_mia_features(CONTENT, _hand_top_k(ACTUAL_LOGPROBS, hand), None)

    sourced = features_from_top_k(CONTENT, truncate_to_top_k(_evidence(), k))

    assert sourced.features == expected
    assert sourced.label == SourceLabel(EstimateSource.TOP_K, k)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("k", [1, 5, 10, 16])
def test_truncated_features_equal_hand_built_top_k_on_derived_evidence(seed: int, k: int) -> None:
    """The equality holds for evidence derived from full vectors, at every k."""
    evidence = _random_evidence(seed)
    tops = [[lp for _, lp in position[:k]] for position in evidence.top_logprobs]
    expected = compute_mia_features(CONTENT, _hand_top_k(evidence.logprob, tops), None)

    assert features_from_top_k(CONTENT, truncate_to_top_k(evidence, k)).features == expected


def test_truncation_shape_matches_top_k_entries() -> None:
    """Entries carry the actual logprob and the first k alternatives as '<id>' dicts."""
    entries = truncate_to_top_k(_evidence(), 2)

    assert isinstance(entries, list)
    assert [entry.token for entry in entries] == ["<2>", "<4>", "<5>"]
    assert [entry.logprob for entry in entries] == list(ACTUAL_LOGPROBS)
    assert entries[1].top_logprobs == [{"token": "<0>", "logprob": -0.3}, {"token": "<9>", "logprob": -1.8}]
    assert all(len(entry.top_logprobs) == 2 for entry in entries)


# --- Full distribution ----------------------------------------------------


@pytest.mark.parametrize("seed", [0, 3])
def test_features_from_full_equals_top_k_at_the_kept_count(seed: int) -> None:
    """`features_from_full` is `features_from_top_k` on all kept alternatives."""
    for evidence, kept in ((_evidence(), 3), (_random_evidence(seed), 16)):
        full = features_from_full(CONTENT, evidence)
        top_k = features_from_top_k(CONTENT, truncate_to_top_k(evidence, kept))

        assert isinstance(full, SourcedFeatures)
        assert full.features == top_k.features
        assert full.label == SourceLabel(EstimateSource.FULL_DISTRIBUTION, kept)


def test_features_use_content_for_zlib_ratio() -> None:
    """`content` is the zlib denominator, so only zlib_ratio changes with it."""
    a = features_from_full(CONTENT, _evidence()).features
    b = features_from_full(CONTENT * 4, _evidence()).features

    assert a.zlib_ratio != b.zlib_ratio
    assert (a.loss, a.min_k, a.min_k_pp, a.ref_delta) == (b.loss, b.min_k, b.min_k_pp, b.ref_delta)
    assert a.ref_delta is None


# --- Labels ----------------------------------------------------------------


def test_labels_carry_source_and_k() -> None:
    """Every estimate is labelled with its source and k."""
    evidence = _random_evidence(4)

    assert EstimateSource.TOP_K.value == "top_k"
    assert EstimateSource.FULL_DISTRIBUTION.value == "full_distribution"
    for k in (1, 7, 16):
        assert features_from_top_k(CONTENT, truncate_to_top_k(evidence, k)).label == SourceLabel(
            EstimateSource.TOP_K, k
        )
    assert features_from_full(CONTENT, evidence).label == SourceLabel(EstimateSource.FULL_DISTRIBUTION, 16)


def test_top_k_label_on_host_logprobs_and_unchanged_features() -> None:
    """Host top-k logprobs keep the single-draw features and are labelled with their k."""
    host = _hand_top_k((-0.1, -0.7, -2.2), [[-0.1, -2.5], [-0.4, -0.7], [-0.9, -1.3]])

    sourced = features_from_top_k(CONTENT, host)

    assert sourced.features == compute_mia_features(CONTENT, host, None)
    assert sourced.label == SourceLabel(EstimateSource.TOP_K, 2)


def test_top_k_label_has_no_k_when_positions_differ() -> None:
    """When positions carry different numbers of alternatives, no single k is claimed."""
    host = _hand_top_k((-0.1, -0.7), [[-0.1, -2.5], [-0.4, -0.7, -1.9]])

    assert features_from_top_k(CONTENT, host).label == SourceLabel(EstimateSource.TOP_K, None)


# --- Mass coverage ----------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 5])
def test_top_k_mass_matches_brute_force_and_is_non_decreasing(seed: int) -> None:
    """Per-position mass equals a brute-force sum of exp and does not fall as k grows."""
    for evidence, kept in ((_evidence(), 3), (_random_evidence(seed), 16)):
        previous = np.zeros(len(evidence.top_logprobs))
        for k in range(1, kept + 1):
            mass = top_k_mass(evidence, k)
            brute = [sum(math.exp(lp) for _, lp in position[:k]) for position in evidence.top_logprobs]

            assert isinstance(mass, np.ndarray)
            assert mass.dtype == np.float64
            assert mass.shape == (len(evidence.top_logprobs),)
            np.testing.assert_allclose(mass, brute, rtol=1e-12, atol=0.0)
            assert np.all(mass >= previous)
            assert np.all(mass <= 1.0 + 1e-12)
            previous = mass


def test_top_k_mass_on_literal_evidence() -> None:
    """A hand-checked value: position 0 at k = 2 is exp(-0.5) + exp(-1.5)."""
    mass = top_k_mass(_evidence(), 2)

    assert mass[0] == pytest.approx(math.exp(-0.5) + math.exp(-1.5), abs=1e-15)


# --- Refusals ----------------------------------------------------------------


@pytest.mark.parametrize("k", [0, -1, 4, 100])
def test_out_of_range_k_is_refused(k: int) -> None:
    """k below 1 or above the kept count is refused by truncation and mass."""
    with pytest.raises(ValueError, match="k must be between 1 and the kept count 3"):
        truncate_to_top_k(_evidence(), k)
    with pytest.raises(ValueError, match="k must be between 1 and the kept count 3"):
        top_k_mass(_evidence(), k)


@pytest.mark.parametrize("k", [True, 2.0, "2", None])
def test_non_integer_k_is_refused(k: object) -> None:
    with pytest.raises(TypeError, match="k must be an integer"):
        truncate_to_top_k(_evidence(), k)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="k must be an integer"):
        top_k_mass(_evidence(), k)  # type: ignore[arg-type]


def test_evidence_without_a_single_kept_count_is_refused() -> None:
    """Ragged, empty-alternative and empty evidence have no kept count."""
    ragged = _evidence(alternatives=(ALTERNATIVES[0], ALTERNATIVES[1][:2], ALTERNATIVES[2]))
    no_alternatives = _evidence(alternatives=((), (), ()))
    empty = FullDistributionEvidence(
        model_digest="sha256:" + "0" * 64, n_vocab=10, response_token_ids=(), logprob=(), entropy=(), rank=(),
        top_logprobs=(),
    )
    for evidence, message in (
        (ragged, "same number of alternatives"),
        (no_alternatives, "keeps no alternatives"),
        (empty, "no response positions"),
    ):
        with pytest.raises(ValueError, match=message):
            features_from_full(CONTENT, evidence)
        with pytest.raises(ValueError, match=message):
            truncate_to_top_k(evidence, 1)
        with pytest.raises(ValueError, match=message):
            top_k_mass(evidence, 1)
