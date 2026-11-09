"""One MIA feature set from either estimate source.

Implements the `EstimateSources` component of the memorization-adjusted
finance-sim design (Requirements 3.1, 3.3, 9.1 and 9.2). Both sources feed the
existing ``compute_mia_features``, so one calibrator type serves both and a
calibrator is fitted per source:

- ``TOP_K``: the response's own top-k logprobs, as a host returns them.
- ``FULL_DISTRIBUTION``: full-distribution evidence, whose top alternatives
  were taken from the full next-token vector.

Every result carries a ``SourceLabel`` with the source and k. Full evidence can
be truncated to any smaller k, which yields the same ``TokenLogprob`` shape a
top-k host returns, and ``top_k_mass`` reports the probability mass the first
k alternatives cover at each position.

Full evidence must keep the same, nonzero number of alternatives at every
position (a host keeps ``min(top_k_keep, n_vocab)``), so its kept count is one
number. Pure functions with no I/O and no global state.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np

from recall_guard.core.full_distribution import FullDistributionEvidence
from recall_guard.core.nvidia_lm import TokenLogprob
from recall_guard.mia.features import MiaFeatures, compute_mia_features


class EstimateSource(Enum):
    """Where the token evidence behind an estimate came from."""

    TOP_K = "top_k"
    FULL_DISTRIBUTION = "full_distribution"


@dataclass(frozen=True)
class SourceLabel:
    """Source and k of one estimate.

    Attributes
    ----------
    source:
        The evidence source.
    k:
        Alternatives per position. For ``FULL_DISTRIBUTION`` it is the kept
        count. For ``TOP_K`` it is ``None`` when positions carry different
        numbers of alternatives, so no single k describes the evidence.
    """

    source: EstimateSource
    k: int | None


@dataclass(frozen=True)
class SourcedFeatures:
    """MIA features together with the label of the evidence they came from."""

    features: MiaFeatures
    label: SourceLabel


def _token(token_id: int) -> str:
    """Render a token id as the placeholder string used in truncated evidence."""
    return f"<{token_id}>"


def _kept_count(evidence: FullDistributionEvidence) -> int:
    """Return the number of alternatives kept at every position.

    Raises
    ------
    ValueError
        If the evidence has no positions, a position has no alternatives, or
        positions keep different numbers of alternatives.
    """
    counts = {len(position) for position in evidence.top_logprobs}
    if not counts:
        raise ValueError("full-distribution evidence has no response positions")
    if len(counts) != 1:
        raise ValueError(
            "full-distribution evidence must keep the same number of alternatives at every position"
        )
    kept = counts.pop()
    if kept == 0:
        raise ValueError("full-distribution evidence keeps no alternatives")
    return kept


def _checked_k(evidence: FullDistributionEvidence, k: int) -> int:
    """Validate ``1 <= k <= kept count`` and return ``k``."""
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be an integer")
    kept = _kept_count(evidence)
    if k < 1 or k > kept:
        raise ValueError(f"k must be between 1 and the kept count {kept}, got {k}")
    return k


def features_from_top_k(content: str, logprobs: Sequence[TokenLogprob]) -> SourcedFeatures:
    """Compute MIA features from a response's own top-k logprobs.

    Parameters
    ----------
    content:
        The response text, used for ``zlib_ratio``.
    logprobs:
        Per-token entries with non-empty ``top_logprobs`` lists, as for
        ``compute_mia_features``.

    Returns
    -------
    SourcedFeatures
        Features labelled ``TOP_K`` with k the alternatives per position
        (``None`` when positions differ).
    """
    entries = list(logprobs)
    features = compute_mia_features(content, entries, None)
    counts = {len(entry.top_logprobs or []) for entry in entries}
    k = counts.pop() if len(counts) == 1 else None
    return SourcedFeatures(features=features, label=SourceLabel(EstimateSource.TOP_K, k))


def truncate_to_top_k(evidence: FullDistributionEvidence, k: int) -> list[TokenLogprob]:
    """Build top-k ``TokenLogprob`` entries from full-distribution evidence.

    Each entry carries the actual token's logprob and the first ``k`` kept
    alternatives, as dicts with ``token`` and ``logprob`` keys. Token strings
    are the ids rendered as ``"<id>"``; the features use only the numbers.

    Raises
    ------
    TypeError
        If ``k`` is not an integer.
    ValueError
        If ``k`` is below 1 or above the kept count, or the evidence has no
        single nonzero kept count.
    """
    k = _checked_k(evidence, k)
    return [
        TokenLogprob(
            token=_token(token_id),
            logprob=logprob,
            top_logprobs=[
                {"token": _token(alt_id), "logprob": alt_logprob}
                for alt_id, alt_logprob in alternatives[:k]
            ],
        )
        for token_id, logprob, alternatives in zip(
            evidence.response_token_ids, evidence.logprob, evidence.top_logprobs, strict=True
        )
    ]


def features_from_full(content: str, evidence: FullDistributionEvidence) -> SourcedFeatures:
    """Compute MIA features from full-distribution evidence.

    The same computation as ``features_from_top_k`` on all kept alternatives,
    labelled ``FULL_DISTRIBUTION`` with k the kept count.
    """
    kept = _kept_count(evidence)
    features = compute_mia_features(content, truncate_to_top_k(evidence, kept), None)
    return SourcedFeatures(
        features=features, label=SourceLabel(EstimateSource.FULL_DISTRIBUTION, kept)
    )


def top_k_mass(evidence: FullDistributionEvidence, k: int) -> np.ndarray:
    """Return, per position, the probability mass of the first ``k`` alternatives.

    The mass is the sum of ``exp`` of the first ``k`` kept alternatives'
    logprobs, as a float64 array with one entry per response position.

    Raises
    ------
    TypeError
        If ``k`` is not an integer.
    ValueError
        If ``k`` is below 1 or above the kept count, or the evidence has no
        single nonzero kept count.
    """
    k = _checked_k(evidence, k)
    return np.asarray(
        [
            float(np.sum(np.exp(np.asarray([lp for _, lp in alternatives[:k]], dtype=np.float64))))
            for alternatives in evidence.top_logprobs
        ],
        dtype=np.float64,
    )


__all__ = [
    "EstimateSource",
    "SourceLabel",
    "SourcedFeatures",
    "features_from_full",
    "features_from_top_k",
    "top_k_mass",
    "truncate_to_top_k",
]
