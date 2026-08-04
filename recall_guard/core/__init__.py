"""Public API for the ``core`` layer of the honest-model-ranking harness.

Re-exports the consumer-facing surface (Req 12.1) so that callers (the
qualification notebook, future external scripts, and the harness layers
themselves) can import every primitive from the package root without ever
touching internal module paths::

    from recall_guard.core import (
        NvidiaLM, CompletionResult, TokenLogprob,
        EvalRow, EvalSet, load_eval_set, load_cutoffs,
        assert_cutoff_safe, CutoffViolation,
        bootstrap_ci,
        Manifest, write_manifest, read_manifest, compute_file_hash,
    )

The ``__all__`` list pins the documented names so ``from recall_guard.core import *``
behaves predictably and so a typo in a re-exported name fails fast at import
time rather than at the first downstream lookup.
"""

from recall_guard.core.bootstrap import bootstrap_ci
from recall_guard.core.consensus import (
    MultimodalVerdict,
    Tail,
    detect_multimodal,
    grid_adherence,
    lag_dependence,
    robust_location,
    scale_floor,
    smallest_certifiable_n,
    smallest_detectable_split_n,
    snap_to_grid,
    wilson_interval,
)
from recall_guard.core.ensemble import (
    CostEstimate,
    EnsembleResult,
    EnsembleSpec,
    LocationMode,
    MultimodalAction,
    ReferenceMode,
    canonical_draw_hash,
    estimate_cost,
    generate_ensemble,
    reduce_draws,
)
from recall_guard.core.loader import (
    CutoffViolation,
    EvalRow,
    EvalSet,
    assert_cutoff_safe,
    load_cutoffs,
    load_eval_set,
)
from recall_guard.core.manifest import (
    Manifest,
    compute_file_hash,
    read_manifest,
    write_manifest,
)
from recall_guard.core.nvidia_lm import CompletionResult, NvidiaLM, TokenLogprob

__all__ = [
    "Tail",
    "MultimodalVerdict",
    "detect_multimodal",
    "grid_adherence",
    "lag_dependence",
    "robust_location",
    "scale_floor",
    "smallest_certifiable_n",
    "smallest_detectable_split_n",
    "snap_to_grid",
    "wilson_interval",
    "CostEstimate",
    "EnsembleResult",
    "EnsembleSpec",
    "LocationMode",
    "MultimodalAction",
    "ReferenceMode",
    "canonical_draw_hash",
    "estimate_cost",
    "generate_ensemble",
    "reduce_draws",
    "NvidiaLM",
    "CompletionResult",
    "TokenLogprob",
    "EvalRow",
    "EvalSet",
    "load_eval_set",
    "load_cutoffs",
    "assert_cutoff_safe",
    "CutoffViolation",
    "bootstrap_ci",
    "Manifest",
    "write_manifest",
    "read_manifest",
    "compute_file_hash",
]
