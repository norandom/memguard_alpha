"""Public API for the ``harness`` layer of the honest-model-ranking harness.

Re-exports the consumer-facing surface (Req 12.1) so that the qualification
notebook (Req 12.2) and any external scripts can import every orchestration
helper, evaluator type, ranker primitive, report writer, plotting helper, and
runner entry point from the package root::

    from src.harness import (
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

from src.harness.evaluator import (
    CIBound,
    ModelEvalResult,
    Record,
    compute_majority_baseline,
    evaluate_model,
)
from src.harness.plots import (
    configure_paper_style,
    plot_accuracy_with_ci,
    plot_composite_ranking,
    plot_mcs_auc_with_ci,
    plot_mcs_calibration,
    plot_mia_feature_distributions,
)
from src.harness.ranker import (
    COMPOSITE_FORMULA,
    GATES,
    CompositeScore,
    composite_score,
    write_top3,
)
from src.harness.report import (
    print_artifact_paths,
    render_terminal,
    write_records,
    write_summary_csv,
)
from src.harness.runner import build_parser, run
from src.harness.smoke import Shortlist, SmokeOutcome, smoke_test

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
