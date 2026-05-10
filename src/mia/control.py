"""Per-model control-corpus baseline for MIA feature standardisation.

Implements the `mia.control` component from the honest-model-ranking design.
Satisfies Requirements 3.1, 3.2, 3.3, 3.4:

- ``ControlBaseline`` — frozen dataclass holding per-feature mean/std plus an
  ``is_calibrated`` flag derived from ``n_valid >= min_valid``.
- ``build_baseline(model_lm, control_rows, ref_lm, min_valid=50)`` — calls the
  model on every control row, computes MIA features, drops rows where logprobs
  are missing or the model timed out, and aggregates per-feature mean/std with
  ``numpy.mean`` and ``numpy.std(ddof=0)``. Std is floored at ``_STD_FLOOR``.
- ``standardise(features, baseline)`` — pure function returning a per-feature
  dict of ``(value - mean) / max(std, _STD_FLOOR)``. Passes through ``None`` for
  ``ref_delta`` when either the baseline or the eval-time features lack it.

Only ``build_baseline`` issues HTTP calls (via the injected ``NvidiaLM``).
``standardise`` is pure — no I/O. Skipped rows are reported at WARNING level
(one per skip with the row index); happy paths log nothing.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from src.core.loader import EvalRow
from src.core.nvidia_lm import NvidiaLM, TokenLogprob, generate_many
from src.mia.features import MiaFeatures, compute_mia_features

logger = logging.getLogger(__name__)


_STD_FLOOR: float = 1e-6
"""Lower bound for per-feature standard deviation, used both by ``build_baseline``
when storing ``feature_stds`` and by ``standardise`` when computing the divisor.
"""


_FEATURE_KEYS: tuple[str, ...] = ("loss", "min_k", "min_k_pp", "zlib_ratio", "ref_delta")


@dataclass(frozen=True)
class ControlBaseline:
    """Per-model baseline distribution of every MIA feature on the OOS control corpus.

    Attributes
    ----------
    model:
        The NVIDIA model ID this baseline was built for.
    n_valid:
        Number of control rows where ``model_lm.generate`` returned usable
        logprobs (i.e., did not raise ``TimeoutError`` or ``RuntimeError``).
    feature_means:
        Per-feature mean across the valid rows. Keys are the five MIA feature
        names. ``feature_means["ref_delta"]`` is ``None`` when no reference
        model is configured (or every reference call failed).
    feature_stds:
        Per-feature standard deviation across the valid rows, floored at
        ``_STD_FLOOR``. ``feature_stds["ref_delta"]`` is ``None`` whenever
        ``feature_means["ref_delta"]`` is ``None``.
    is_calibrated:
        ``True`` iff ``n_valid >= min_valid``. Used by the runner to decide
        whether to evaluate the model or surface an ``uncalibrated`` warning.
    min_valid:
        The threshold used (default 50, per the Open Defaults in
        requirements.md).
    """

    model: str
    n_valid: int
    feature_means: dict[str, float | None]
    feature_stds: dict[str, float | None]
    is_calibrated: bool
    min_valid: int


def _aggregate_mean_std(values: Iterable[float]) -> tuple[float, float]:
    """Mean and floored std (ddof=0) for a finite sequence of floats."""
    arr = np.asarray(list(values), dtype=np.float64)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if std < _STD_FLOOR:
        std = _STD_FLOOR
    return mean, std


def build_baseline(
    model_lm: NvidiaLM,
    control_rows: list[EvalRow],
    ref_lm: NvidiaLM | None,
    min_valid: int = 50,
    max_workers: int = 1,
) -> ControlBaseline:
    """Build a per-model control-corpus baseline.

    For each row in ``control_rows``:

    - Call ``model_lm.generate(row.prompt)``. On ``TimeoutError`` or
      ``RuntimeError`` (e.g., missing logprobs) the row is dropped and a
      WARNING is logged with the row index.
    - When ``ref_lm`` is provided, also call ``ref_lm.generate(row.prompt)``.
      A reference-side failure does **not** invalidate the row — it merely
      sets ``ref_logprobs = None`` for that row, so the four other features
      still contribute to the baseline.
    - Compute :class:`MiaFeatures` via :func:`compute_mia_features`.

    Per-feature mean and std are aggregated with ``numpy.mean`` and
    ``numpy.std(ddof=0)``. Std is floored at ``_STD_FLOOR`` to avoid div-by-zero
    in :func:`standardise`. When every valid row has ``ref_delta = None``
    (because ``ref_lm is None`` or every reference call failed), the
    ``ref_delta`` mean and std are stored as ``None``.

    ``is_calibrated`` is set to ``n_valid >= min_valid``.
    """
    # Fan out the model + ref calls in parallel (max_workers=1 keeps the
    # original sequential ordering for tests that mock requests.post).
    prompts = [row.prompt for row in control_rows]
    primary_results = generate_many(model_lm, prompts, max_workers=max_workers)
    ref_results: list = (
        generate_many(ref_lm, prompts, max_workers=max_workers)
        if ref_lm is not None else [None] * len(prompts)
    )

    per_row_features: list[MiaFeatures] = []
    for idx, (primary, ref_res) in enumerate(zip(primary_results, ref_results, strict=True)):
        if isinstance(primary, Exception) or primary is None:
            logger.warning(
                "control baseline: skipping row %d for model %s (logprobs missing or timeout)",
                idx,
                model_lm.model,
            )
            continue
        content, logprobs = primary.content, primary.logprobs

        ref_logprobs: list[TokenLogprob] | None = None
        if ref_lm is not None:
            if isinstance(ref_res, Exception) or ref_res is None:
                logger.warning(
                    "control baseline: ref-model %s failed on row %d; "
                    "ref_delta dropped for this row",
                    ref_lm.model,
                    idx,
                )
            else:
                ref_logprobs = ref_res.logprobs

        try:
            features = compute_mia_features(content, logprobs, ref_logprobs)
        except ValueError:
            logger.warning(
                "control baseline: skipping row %d for model %s "
                "(MIA feature computation failed)",
                idx,
                model_lm.model,
            )
            continue
        per_row_features.append(features)

    n_valid = len(per_row_features)

    feature_means: dict[str, float | None] = {}
    feature_stds: dict[str, float | None] = {}

    if n_valid > 0:
        for key in ("loss", "min_k", "min_k_pp", "zlib_ratio"):
            mean, std = _aggregate_mean_std(getattr(f, key) for f in per_row_features)
            feature_means[key] = mean
            feature_stds[key] = std

        ref_values = [
            f.ref_delta for f in per_row_features if f.ref_delta is not None
        ]
        if ref_values:
            mean, std = _aggregate_mean_std(ref_values)
            feature_means["ref_delta"] = mean
            feature_stds["ref_delta"] = std
        else:
            feature_means["ref_delta"] = None
            feature_stds["ref_delta"] = None
    else:
        for key in _FEATURE_KEYS:
            feature_means[key] = None
            feature_stds[key] = None

    return ControlBaseline(
        model=model_lm.model,
        n_valid=n_valid,
        feature_means=feature_means,
        feature_stds=feature_stds,
        is_calibrated=(n_valid >= min_valid),
        min_valid=min_valid,
    )


def standardise(
    features: MiaFeatures, baseline: ControlBaseline
) -> dict[str, float | None]:
    """Standardise eval-time MIA features against the model's control baseline.

    For each of the four always-present features ``loss``, ``min_k``,
    ``min_k_pp``, ``zlib_ratio`` returns ``(value - mean) / max(std, _STD_FLOOR)``.

    For ``ref_delta`` returns ``None`` whenever either the baseline or the
    eval-time features have no reference value to standardise — i.e., the
    field stays "off" rather than being silently coerced to ``0.0``.
    """
    out: dict[str, float | None] = {}
    for key in ("loss", "min_k", "min_k_pp", "zlib_ratio"):
        mean = baseline.feature_means[key]
        std = baseline.feature_stds[key]
        # The four core features are always populated when n_valid > 0; if a
        # caller hands us an uncalibrated baseline, fall through to None.
        if mean is None or std is None:
            out[key] = None
            continue
        divisor = std if std >= _STD_FLOOR else _STD_FLOOR
        out[key] = (float(getattr(features, key)) - float(mean)) / divisor

    ref_mean = baseline.feature_means.get("ref_delta")
    ref_std = baseline.feature_stds.get("ref_delta")
    if features.ref_delta is None or ref_mean is None or ref_std is None:
        out["ref_delta"] = None
    else:
        divisor = ref_std if ref_std >= _STD_FLOOR else _STD_FLOOR
        out["ref_delta"] = (float(features.ref_delta) - float(ref_mean)) / divisor

    return out


__all__ = ["ControlBaseline", "build_baseline", "standardise", "_STD_FLOOR"]
