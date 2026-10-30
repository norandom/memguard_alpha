"""Minimal greedy integration of compatible caller-provided logits."""

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

NextLogits = Callable[[tuple[int, ...]], ArrayLike]


@dataclass(frozen=True)
class ModelSource:
    """Caller-declared token mapping and full-prefix logits provider."""

    next_logits: NextLogits
    vocabulary_id: str
    vocabulary_size: int
    eos_token_ids: frozenset[int]
    context_limit: int


@dataclass(frozen=True)
class DecodeResult:
    """Generated IDs, including terminal EOS, and the stopping reason."""

    token_ids: tuple[int, ...]
    finish_reason: Literal["eos", "length"]


class DecodeError(RuntimeError):
    """A source or mathematical failure at a zero-based generation step."""

    def __init__(
        self, message: str, *, role: Literal["main", "older", "newer", "combination"], step: int,
    ) -> None:
        super().__init__(message)
        self.role = role
        self.step = step


def _validate_positive_integer(value: int, *, name: str) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_token_ids(token_ids: Sequence[int], *, name: str, vocabulary_size: int) -> tuple[int, ...]:
    try:
        validated_token_ids = tuple(token_ids)
    except TypeError as error:
        raise TypeError(f"{name} must be a sequence of integer token IDs") from error
    for token_id in validated_token_ids:
        if isinstance(token_id, (bool, np.bool_)) or not isinstance(token_id, Integral):
            raise TypeError(f"{name} must contain non-boolean integer token IDs")
        if not 0 <= token_id < vocabulary_size:
            raise ValueError(f"{name} token IDs must be within vocabulary_size")
    return tuple(int(token_id) for token_id in validated_token_ids)


def _validate_source(source: ModelSource, *, name: str) -> None:
    if not isinstance(source, ModelSource):
        raise TypeError(f"{name} must be a ModelSource")
    if not callable(source.next_logits):
        raise TypeError(f"{name}.next_logits must be callable")
    if not isinstance(source.vocabulary_id, str) or not source.vocabulary_id.strip():
        raise ValueError(f"{name}.vocabulary_id must be a nonempty string")
    _validate_positive_integer(source.vocabulary_size, name=f"{name}.vocabulary_size")
    _validate_positive_integer(source.context_limit, name=f"{name}.context_limit")
    if not isinstance(source.eos_token_ids, frozenset):
        raise TypeError(f"{name}.eos_token_ids must be a frozenset of token IDs")
    _validate_token_ids(tuple(source.eos_token_ids), name=f"{name}.eos_token_ids", vocabulary_size=source.vocabulary_size)


def _validate_decode_request(
    prompt_token_ids: Sequence[int], *,
    main_source: ModelSource, older_source: ModelSource, newer_source: ModelSource,
    alpha: float, max_new_tokens: int,
) -> tuple[int, ...]:
    """Validate declarations without calling models; return an immutable prompt."""
    sources = (("main_source", main_source), ("older_source", older_source), ("newer_source", newer_source))
    for source_name, source in sources:
        _validate_source(source, name=source_name)
    for source_name, source in sources[1:]:
        for field_name in ("vocabulary_id", "vocabulary_size", "eos_token_ids"):
            if getattr(source, field_name) != getattr(main_source, field_name):
                raise ValueError(f"{source_name}.{field_name} must match main_source.{field_name}")
    _validate_positive_integer(max_new_tokens, name="max_new_tokens")
    if isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, Real):
        raise TypeError("alpha must be a finite nonnegative real number")
    try:
        valid_alpha = math.isfinite(alpha) and alpha >= 0
    except (OverflowError, ValueError) as error:
        raise ValueError("alpha must be a finite nonnegative real number") from error
    if not valid_alpha:
        raise ValueError("alpha must be a finite nonnegative real number")
    validated_prompt_token_ids = _validate_token_ids(
        prompt_token_ids, name="prompt_token_ids", vocabulary_size=main_source.vocabulary_size,
    )
    if not validated_prompt_token_ids:
        raise ValueError("prompt_token_ids must be nonempty")
    minimum_context_limit = min(int(source.context_limit) for _, source in sources)
    if len(validated_prompt_token_ids) + int(max_new_tokens) > minimum_context_limit:
        raise ValueError("prompt_token_ids and max_new_tokens exceed the minimum context_limit")
    return validated_prompt_token_ids
