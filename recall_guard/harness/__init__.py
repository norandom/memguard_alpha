"""Public API for the ``harness`` layer of the honest-model-ranking harness.

Re-exports the consumer-facing surface (Req 12.1) so that the qualification
notebook (Req 12.2) and any external scripts can import every orchestration
helper, evaluator type, ranker primitive, report writer, plotting helper, and
runner entry point from the package root::

    from recall_guard.harness import (
        # smoke
        SmokeOutcome, Shortlist, smoke_test,
        # evaluator
        Record, CIBound, ModelEvalResult, evaluate_model,
        compute_majority_baseline,
        # ranker
        CompositeScore, COMPOSITE_FORMULA, GATES,
        composite_score, write_top3,
        # report
        render_terminal, write_records, write_summary_csv,
        print_artifact_paths,
        # plots
        configure_paper_style,
        plot_mia_feature_distributions, plot_mcs_calibration,
        plot_accuracy_with_ci, plot_mcs_auc_with_ci,
        plot_composite_ranking,
        # runner
        run, build_parser,
    )

The notebook in ``notebooks/qualification.ipynb`` consumes this surface
verbatim; any drift here breaks ``tests/harness/test_notebook.py``'s
"public API imports succeed" smoke test (Req 12.1).
"""

from recall_guard.harness.evaluator import (
    CIBound,
    ModelEvalResult,
    Record,
    compute_majority_baseline,
    evaluate_model,
)
from recall_guard.harness.ranker import (
    COMPOSITE_FORMULA,
    GATES,
    CompositeScore,
    composite_score,
    write_top3,
)
from recall_guard.harness.report import (
    print_artifact_paths,
    render_terminal,
    write_records,
    write_summary_csv,
)
from recall_guard.harness.runner import build_parser, run
from recall_guard.harness.smoke import Shortlist, SmokeOutcome, smoke_test

__all__ = [
    # smoke
    "SmokeOutcome",
    "Shortlist",
    "smoke_test",
    # evaluator
    "Record",
    "CIBound",
    "ModelEvalResult",
    "evaluate_model",
    "compute_majority_baseline",
    # ranker
    "CompositeScore",
    "COMPOSITE_FORMULA",
    "GATES",
    "composite_score",
    "write_top3",
    # report
    "render_terminal",
    "write_records",
    "write_summary_csv",
    "print_artifact_paths",
    # plots
    "configure_paper_style",
    "plot_mia_feature_distributions",
    "plot_mcs_calibration",
    "plot_accuracy_with_ci",
    "plot_mcs_auc_with_ci",
    "plot_composite_ranking",
    # runner
    "run",
    "build_parser",
]

# Plotting names resolved lazily so matplotlib stays off the eager import path.
_LAZY_PLOT_EXPORTS = frozenset(
    {
        "configure_paper_style",
        "plot_accuracy_with_ci",
        "plot_composite_ranking",
        "plot_mcs_auc_with_ci",
        "plot_mcs_calibration",
        "plot_mia_feature_distributions",
    }
)


def __getattr__(name: str):
    """Resolve plotting helpers on demand (keeps matplotlib off the import path)."""
    if name in _LAZY_PLOT_EXPORTS:
        from recall_guard.harness import plots

        return getattr(plots, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
