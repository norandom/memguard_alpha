"""Experimental mathematical steering for vocabulary-indexed model logits."""

from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _validate_logits(logits: ArrayLike, *, name: str) -> NDArray[np.float64]:
    """Validate one finite real vector whose only axis indexes the vocabulary."""
    try:
        logits_array = np.asarray(logits)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a one-dimensional real numeric vector") from error
    if logits_array.ndim != 1 or logits_array.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional vector")
    if logits_array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real integer or floating-point values")
    with np.errstate(over="ignore", invalid="ignore"):
        validated_logits = logits_array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(validated_logits)):
        raise ValueError(f"{name} must contain finite float64 values")
    return validated_logits


def combine_logits(
    main_logits: ArrayLike,
    older_logits: ArrayLike,
    newer_logits: ArrayLike,
    *,
    alpha: float,
) -> NDArray[np.float64]:
    """Return ``main_logits + alpha * (older_logits - newer_logits)``.

    Each input has shape ``(vocabulary_size,)`` with corresponding token IDs
    at the same positions. Integer/float inputs are converted to float64;
    compatibility of the token meanings is the caller's responsibility.
    Inputs are never mutated. At zero alpha the validated main vector is
    copied without evaluating the auxiliary difference.

    Raises
    ------
    TypeError
        If logits or alpha have unsupported types.
    ValueError
        If vector shapes, finite values, alpha or the result are invalid.
    """
    main_logits = _validate_logits(main_logits, name="main_logits")
    older_logits = _validate_logits(older_logits, name="older_logits")
    newer_logits = _validate_logits(newer_logits, name="newer_logits")
    for name, auxiliary_logits in (("older_logits", older_logits), ("newer_logits", newer_logits)):
        if auxiliary_logits.shape != main_logits.shape:
            raise ValueError(f"{name} must have the same vocabulary length as main_logits")
    if isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, Real):
        raise TypeError("alpha must be a finite nonnegative real number")
    try:
        alpha = float(alpha)
    except (OverflowError, ValueError) as error:
        raise ValueError("alpha must be a finite nonnegative real number") from error
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be a finite nonnegative real number")
    if alpha == 0:
        return main_logits.copy()
    with np.errstate(over="ignore", invalid="ignore"):
        adjusted_logits = main_logits + alpha * (older_logits - newer_logits)
    if not np.all(np.isfinite(adjusted_logits)):
        raise ValueError("adjusted_logits must be finite; the combination overflowed")
    return adjusted_logits
