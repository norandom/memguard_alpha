"""MIA feature computation for one (model, prompt, response) record.

Implements the `mia.features` component from the honest-model-ranking
design. Computes the five MIA features defined by Requirements 4.1, 4.2,
and 4.3:

- ``loss``:       mean negative logprob of the realised tokens.
- ``min_k``:      mean of the bottom-K clipped logprobs.
- ``min_k_pp``:   mean of the bottom-K per-position z-scores against each
                  token's ``top_logprobs`` distribution (Min-K%++).
- ``zlib_ratio``: ``-sum(clipped_logprobs) / len(zlib.compress(response, 9))``.
- ``ref_delta``:  ``loss_self - loss_ref`` (``None`` when no reference run).

Pure function with no I/O and no global state. Numerical stability is enforced
by clipping individual logprobs to a finite floor (``LOGPROB_FLOOR``)
before any averaging, and by flooring per-position standard deviation at
``1e-6`` for the Min-K%++ z-score.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Any

import numpy as np

from recall_guard.core.nvidia_lm import TokenLogprob

LOGPROB_FLOOR: float = -30.0
"""Lower bound for individual logprob values, applied before averaging.

Prevents a single ``-inf`` (or extremely negative) per-token logprob from
poisoning ``loss`` / ``min_k`` / ``zlib_ratio`` / ``ref_delta``.
"""

_STD_FLOOR: float = 1e-6


@dataclass(frozen=True)
class MiaFeatures:
    """Five MIA features for one (model, prompt, response) record.

    Attributes
    ----------
    loss:
        Mean negative logprob of the realised tokens (clipped at floor).
        Low loss means the model found the text easy to predict, which is
        what stored text looks like.
    min_k:
        Mean of the bottom ``int(len * k)`` clipped logprobs (Min-K%).
        Negative; lower means more "memorized". Looks only at the hardest
        tokens, because that is where memorization shows first: if the
        model breezes through even those, it has probably seen the text.
    min_k_pp:
        Mean of the bottom-K per-position z-scores (Min-K%++). Same idea as
        ``min_k``, but each token is graded against its own candidate
        distribution instead of an absolute scale.
    zlib_ratio:
        ``-sum(clipped_logprobs) / len(zlib.compress(response, 9))``.
        ``0.0`` when ``response`` is empty. Dividing by the compressed size
        cancels plain repetitiveness; a repetitive text is cheap to predict
        AND cheap to compress, so what remains is the confidence the model
        has beyond what the text's redundancy explains.
    ref_delta:
        ``loss_self - loss_ref``; ``None`` when ``ref_logprobs is None``.
        The reference model anchors what "normal" confidence looks like
        for the same text, so shared easiness cancels and model-specific
        recall remains.
    """

    loss: float
    min_k: float
    min_k_pp: float
    zlib_ratio: float
    ref_delta: float | None


def _clip_logprob(value: float) -> float:
    """Clip a single logprob to ``LOGPROB_FLOOR`` from below."""
    return value if value > LOGPROB_FLOOR else LOGPROB_FLOOR


def _clipped_array(logprobs: list[TokenLogprob]) -> np.ndarray:
    """Return a 1-D float64 array of clipped per-token logprobs."""
    return np.asarray(
        [_clip_logprob(float(t.logprob)) for t in logprobs], dtype=np.float64
    )


def _bottom_k_count(n: int, k: float) -> int:
    """Number of entries in the bottom-K slice. Floored at 1."""
    return max(1, int(n * k))


def _loss(clipped: np.ndarray) -> float:
    """Mean negative logprob (loss) for a clipped logprob array."""
    return float(-np.mean(clipped))


def _top_logprob_floats(entry: TokenLogprob, position: int) -> np.ndarray:
    """Extract the float ``logprob`` values from a token's ``top_logprobs``.

    Raises
    ------
    ValueError
        If the entry has an empty or missing ``top_logprobs`` list, which
        violates the design's per-position invariant.
    """
    top: list[Any] = list(entry.top_logprobs or [])
    if not top:
        raise ValueError(
            f"top_logprobs is empty or missing for token at position {position}"
        )
    floats: list[float] = []
    for cand in top:
        # Each candidate is a dict from the NVIDIA OpenAI-compatible API
        # with at least a 'logprob' key. Be defensive about object-style
        # entries that expose `.logprob` as well.
        if isinstance(cand, dict):
            if "logprob" not in cand:
                raise ValueError(
                    f"top_logprobs candidate missing 'logprob' at position {position}"
                )
            floats.append(float(cand["logprob"]))
        else:
            floats.append(float(cand.logprob))
    return np.asarray(floats, dtype=np.float64)


def _min_k_pp(logprobs: list[TokenLogprob], clipped: np.ndarray, k: float) -> float:
    """Compute Min-K%++: mean of bottom-K per-position z-scores."""
    z_scores = np.empty(len(logprobs), dtype=np.float64)
    for i, entry in enumerate(logprobs):
        top = _top_logprob_floats(entry, position=i)
        mean_i = float(np.mean(top))
        std_i = float(np.std(top))  # ddof=0
        if std_i < _STD_FLOOR:
            std_i = _STD_FLOOR
        z_scores[i] = (clipped[i] - mean_i) / std_i

    bottom_n = _bottom_k_count(len(z_scores), k)
    bottom = np.sort(z_scores)[:bottom_n]
    return float(np.mean(bottom))


def _zlib_ratio(response: str, clipped: np.ndarray) -> float:
    """zlib-normalised log-likelihood. ``0.0`` on empty response."""
    if response == "":
        return 0.0
    compressed_len = len(zlib.compress(response.encode("utf-8"), 9))
    return float(-np.sum(clipped)) / max(compressed_len, 1)


def compute_mia_features(
    response: str,
    logprobs: list[TokenLogprob],
    ref_logprobs: list[TokenLogprob] | None,
    k: float = 0.2,
) -> MiaFeatures:
    """Compute the five MIA features for one record.

    Parameters
    ----------
    response:
        The model's emitted text. Used only for the zlib-ratio denominator.
    logprobs:
        Per-token logprob entries from ``core.nvidia_lm.NvidiaLM.generate``.
        Must be non-empty, and each entry must carry a non-empty
        ``top_logprobs`` list (precondition from design).
    ref_logprobs:
        Per-token logprobs from a reference model on the same prompt; or
        ``None`` to disable the reference-delta feature.
    k:
        Fraction of tokens used for the bottom-K slice in Min-K% and
        Min-K%++. Defaults to 0.2 (the paper's setting).

    Returns
    -------
    MiaFeatures
        Frozen dataclass with all five features.

    Raises
    ------
    ValueError
        If ``logprobs`` is empty, or any entry has an empty/missing
        ``top_logprobs`` list.
    """
    if not logprobs:
        raise ValueError("logprobs is empty")

    clipped = _clipped_array(logprobs)
    loss_self = _loss(clipped)

    # Min-K%: bottom-K clipped logprobs
    bottom_n = _bottom_k_count(len(clipped), k)
    min_k = float(np.mean(np.sort(clipped)[:bottom_n]))

    # Min-K%++: per-position z-scores
    min_k_pp = _min_k_pp(logprobs, clipped, k)

    zlib_ratio = _zlib_ratio(response, clipped)

    if ref_logprobs is None:
        ref_delta: float | None = None
    else:
        if not ref_logprobs:
            raise ValueError("ref_logprobs is empty")
        ref_clipped = _clipped_array(ref_logprobs)
        ref_delta = loss_self - _loss(ref_clipped)

    return MiaFeatures(
        loss=loss_self,
        min_k=min_k,
        min_k_pp=min_k_pp,
        zlib_ratio=zlib_ratio,
        ref_delta=ref_delta,
    )


__all__ = ["LOGPROB_FLOOR", "MiaFeatures", "compute_mia_features"]
