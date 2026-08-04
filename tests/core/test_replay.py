"""Replay: a stored draw set must reduce to the same answer, forever.

This is the contract that makes an ensemble auditable. Someone holding the
persisted replies must be able to re-derive the reported consensus without
contacting a model, and get bit-identical output -- otherwise the reported
answer is a property of one execution rather than of the evidence.

Nothing here touches the network. If any test in this file needs an LM, the
reduction has stopped being a pure function of the draw set.
"""

from __future__ import annotations

import random
import struct

import pytest

from recall_guard.core.ensemble import (
    EnsembleSpec,
    LocationMode,
    canonical_draw_hash,
    reduce_draws,
)
from recall_guard.core.nvidia_lm import CompletionResult, TokenLogprob

#: A frozen draw set standing in for persisted evidence. Values sit on a 0.1
#: lattice and include an exact zero and a duplicated reply, both of which are
#: tie sources the reduction has to resolve on content alone.
STORED_REPLIES: tuple[str, ...] = (
    "0.8",
    "0.7",
    "0.8",
    "-0.0",
    "0.6",
    "0.8",
    "0.7",
    "0.9",
    "0.0",
)


def _draw(content: str) -> CompletionResult:
    return CompletionResult(
        content=content,
        logprobs=[TokenLogprob(token="t", logprob=-0.5, top_logprobs=[{"t": -0.5}])],
        raw_temperature_observed=0.0,
    )


def _reduce(replies, *, spec=None):
    spec = spec or EnsembleSpec(draws=9, max_workers=3, min_parsed=1, grid=0.1)
    return reduce_draws(
        [_draw(r) for r in replies],
        spec,
        decide=lambda c: c.content,
        components=lambda c: {"axis": float(c.content)},
    )


def _fingerprint(result) -> tuple:
    """Everything a persisted artifact would carry, compared bitwise."""
    return (
        result.consensus.content,
        result.draws_sha256,
        struct.pack("<d", result.agreement),
        tuple(struct.pack("<d", v) for v in result.agreement_ci),
        tuple((k, struct.pack("<d", v)) for k, v in sorted(result.location.items())),
        tuple(
            (k, struct.pack("<d", v))
            for k, v in sorted((result.location_snapped or {}).items())
        ),
        result.multimodal,
        result.n_parsed,
    )


def test_repeated_reduction_is_bit_identical() -> None:
    baseline = _fingerprint(_reduce(STORED_REPLIES))
    for _ in range(25):
        assert _fingerprint(_reduce(STORED_REPLIES)) == baseline


def test_reduction_survives_reordering_of_the_stored_set() -> None:
    """Arrival order is thread-scheduling noise, not evidence.

    Under an ensemble every prompt is identical, so which reply landed at which
    index is not reproducible and must not influence anything.
    """
    baseline = _fingerprint(_reduce(STORED_REPLIES))
    rng = random.Random(20260804)
    for _ in range(50):
        shuffled = list(STORED_REPLIES)
        rng.shuffle(shuffled)
        assert _fingerprint(_reduce(shuffled)) == baseline


def test_signed_zero_in_the_stored_set_does_not_move_the_result() -> None:
    """`-0.0` and `0.0` compare equal but serialise differently."""
    swapped = ["0.0" if r == "-0.0" else r for r in STORED_REPLIES]
    swapped = ["-0.0" if r == "0.0" and i > 5 else r for i, r in enumerate(swapped)]
    a = _reduce(STORED_REPLIES)
    b = _reduce(swapped)
    assert struct.pack("<d", a.location["axis"]) == struct.pack("<d", b.location["axis"])


def test_reduction_needs_no_seed() -> None:
    """No randomness on the consensus path, so nothing needs persisting.

    Seeding the global RNG differently between runs must change nothing; if it
    did, replay would depend on state no artifact records.
    """
    random.seed(1)
    first = _fingerprint(_reduce(STORED_REPLIES))
    random.seed(999_999)
    assert _fingerprint(_reduce(STORED_REPLIES)) == first


def test_reduction_is_stable_across_location_modes() -> None:
    for mode in LocationMode:
        spec = EnsembleSpec(
            draws=9, max_workers=3, min_parsed=1, grid=0.1, location_mode=mode
        )
        baseline = _fingerprint(_reduce(STORED_REPLIES, spec=spec))
        rng = random.Random(7)
        for _ in range(15):
            shuffled = list(STORED_REPLIES)
            rng.shuffle(shuffled)
            assert _fingerprint(_reduce(shuffled, spec=spec)) == baseline, mode


def test_hash_identifies_the_draw_multiset() -> None:
    """A dropped or duplicated draw must change the digest."""
    base = canonical_draw_hash(list(STORED_REPLIES))
    assert base == canonical_draw_hash(sorted(STORED_REPLIES))
    assert base != canonical_draw_hash(list(STORED_REPLIES)[:-1])
    assert base != canonical_draw_hash([*STORED_REPLIES, "0.8"])
    assert base != canonical_draw_hash(["0.9" if r == "0.8" else r for r in STORED_REPLIES])


def test_stored_hash_is_pinned() -> None:
    """Changing canonicalisation changes every replayed artifact.

    Pinned so that such a change has to be deliberate: if this value moves, the
    hash scheme moved with it and previously persisted digests no longer match.
    """
    assert canonical_draw_hash(list(STORED_REPLIES)) == (
        "6dbc62f37404dd854609148d35438144de91c220dbc5e4f21393554b3054a32d"
    )


def test_replay_contacts_no_model() -> None:
    """A guard against the reduction quietly acquiring an I/O dependency."""
    import requests

    def _explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("reduction attempted a network call")

    original = requests.post
    requests.post = _explode
    try:
        assert _reduce(STORED_REPLIES).n_parsed == len(STORED_REPLIES)
    finally:
        requests.post = original


def test_empty_draw_set_is_refused() -> None:
    with pytest.raises(ValueError):
        _reduce([])
