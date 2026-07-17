"""Public API for the ``mia`` layer of the honest-model-ranking harness.

Re-exports the consumer-facing surface (Req 12.1) so that callers (the
qualification notebook, future external scripts, and the harness layers
themselves) can import the MIA feature primitives, the per-model control
baseline, and the per-model MCS calibrator from the package root::

    from recall_guard.mia import (
        MiaFeatures, compute_mia_features, LOGPROB_FLOOR,
        ControlBaseline, build_baseline, standardise,
        MCSCalibrator, train_mcs,
    )

``train_mcs`` is the documented public name for the calibrator's training
function (Req 12.1, Task 5.5 brief). The original function is defined as
:func:`recall_guard.mia.mcs.train`; ``train_mcs`` is re-exported as an alias here so
notebook code reads as "train an MCS calibrator" rather than the more
ambiguous bare ``train``. Both names point at the same callable.
"""

from recall_guard.mia.control import ControlBaseline, build_baseline, standardise
from recall_guard.mia.features import LOGPROB_FLOOR, MiaFeatures, compute_mia_features
from recall_guard.mia.mcs import MCSCalibrator
from recall_guard.mia.mcs import train as train_mcs

__all__ = [
    "MiaFeatures",
    "compute_mia_features",
    "LOGPROB_FLOOR",
    "ControlBaseline",
    "build_baseline",
    "standardise",
    "MCSCalibrator",
    "train_mcs",
]
