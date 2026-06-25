"""Tests for the qualification notebook (Task 5.5).

Validates Req 12.1, 12.2, 12.3, 12.6:

* Public API re-exports succeed from package roots — no internal-path imports
  required (Req 12.1).
* The notebook parses as valid Jupyter v4 JSON and is non-empty (Req 12.2).
* The notebook contains Markdown cells rendering every formula listed in the
  Req 12.6 table (Loss, Min-K%, Min-K%++, zlib, ref-delta, control
  standardisation, MCS logistic regression, MemGuard penalty, bootstrap CI,
  ROC-AUC, majority-class baseline, composite ranking score).
* The notebook executes end-to-end via ``nbclient.NotebookClient`` when the
  ``HARNESS_NOTEBOOK_MOCK`` toggle is set, with no LM HTTP traffic (Req 12.2).

The mock toggle is documented in the notebook's "test mode" cell: when the
env var is ``"1"`` the notebook replaces ``NvidiaLM`` with an in-process fake
that produces deterministic ``Direction:`` / ``Confidence:`` responses, so the
test suite can exercise the full notebook without a real ``NVIDIA_API_KEY``.
"""

from __future__ import annotations

import json
from pathlib import Path

import nbformat
import pytest

NOTEBOOK = Path(__file__).resolve().parents[2] / "notebooks" / "qualification.ipynb"


# -- Req 12.1: public API re-exports succeed from package roots ----------------


def test_public_api_imports_succeed():
    """Smokes Req 12.1: every name in the public API surface is importable
    from a package root, with no ``from recall_guard.harness.runner import _internal``
    paths anywhere in the public contract."""
    from recall_guard.core import (  # noqa: F401
        EvalRow,
        EvalSet,
        Manifest,
        NvidiaLM,
        assert_cutoff_safe,
        bootstrap_ci,
        load_cutoffs,
        load_eval_set,
        read_manifest,
        write_manifest,
    )
    from recall_guard.harness import (  # noqa: F401
        composite_score,
        compute_majority_baseline,
        configure_paper_style,
        evaluate_model,
        plot_accuracy_with_ci,
        plot_composite_ranking,
        plot_mcs_auc_with_ci,
        plot_mcs_calibration,
        plot_mia_feature_distributions,
        run,
        smoke_test,
        write_top3,
    )
    from recall_guard.mia import (  # noqa: F401
        ControlBaseline,
        MCSCalibrator,
        MiaFeatures,
        build_baseline,
        compute_mia_features,
        train_mcs,
    )
    # If every import resolved, the public API contract holds.
    assert callable(evaluate_model)
    assert callable(train_mcs)
    assert callable(configure_paper_style)


# -- Req 12.2: notebook parses as valid Jupyter v4 JSON -----------------------


def test_notebook_file_exists():
    assert NOTEBOOK.exists(), (
        f"Expected qualification notebook at {NOTEBOOK}; Task 5.5 requires "
        "this file to ship."
    )


def test_notebook_parses_as_valid_jupyter_json():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    assert nb.cells, "qualification notebook has zero cells"
    # Every cell must declare a recognised cell_type so nbclient can dispatch.
    for cell in nb.cells:
        assert cell.cell_type in {"markdown", "code"}, (
            f"Unexpected cell_type {cell.cell_type!r} in qualification notebook"
        )


def test_notebook_has_minimum_cell_count():
    """The Task 5.5 brief calls for >= 25 cells (intro + cutoffs table +
    smoke + 12 equations × ~3 cells each + final figure save + closing).
    This guards against accidental truncation."""
    nb = nbformat.read(NOTEBOOK, as_version=4)
    assert len(nb.cells) >= 25, (
        f"Notebook has {len(nb.cells)} cells; Task 5.5 expects >= 25."
    )


# -- Req 12.6: every required formula is rendered in a Markdown cell ----------


REQUIRED_FORMULA_MARKERS: list[str] = [
    # Loss
    r"\mathcal{L}",
    # Min-K% / Min-K%++
    r"Min-K\%",
    r"Min-K\%++",
    # zlib ratio
    r"zlib",
    # ref-delta
    r"ref}",
    # Control standardisation
    r"ctrl",
    # MCS logistic-regression probability
    r"memorized",
    # MemGuard penalty
    r"penalised",
    # Bootstrap percentile CI
    r"\text{CI}",
    # ROC-AUC
    r"AUC",
    # Majority-class baseline accuracy
    r"\text{acc}_{\text{maj}}",
    # Composite ranking score
    r"AccLowerCI",
]


def test_notebook_includes_required_equation_blocks():
    nb = nbformat.read(NOTEBOOK, as_version=4)
    md_text = "\n".join(c.source for c in nb.cells if c.cell_type == "markdown")
    missing = [s for s in REQUIRED_FORMULA_MARKERS if s not in md_text]
    assert not missing, (
        "qualification notebook is missing the following formula markers in "
        f"its Markdown cells: {missing}. Req 12.6 requires every formula to "
        "render via $$...$$ before its compute cell."
    )


# -- Req 12.2: notebook executes end-to-end in mock mode -----------------------


def test_notebook_executes_in_mock_mode(monkeypatch):
    """Run the notebook with HARNESS_NOTEBOOK_MOCK=1; assert no cell errors.

    The notebook's "test mode" cell detects the env var and swaps NvidiaLM
    for an in-process fake. This keeps the test offline (no NVIDIA_API_KEY,
    no FMP_API_KEY) yet still exercises every public-API entry point used by
    the notebook.
    """
    nbclient = pytest.importorskip("nbclient")
    monkeypatch.setenv("HARNESS_NOTEBOOK_MOCK", "1")
    monkeypatch.setenv("MPLBACKEND", "Agg")

    nb = nbformat.read(NOTEBOOK, as_version=4)
    client = nbclient.NotebookClient(
        nb,
        timeout=60,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK.parent)}},
    )
    # NotebookClient.execute raises CellExecutionError on any failing cell.
    client.execute()


# -- Sanity: notebook is well-formed JSON (cheap, even without nbformat) ------


def test_notebook_is_well_formed_json_on_disk():
    """A redundant byte-level check so a corrupt save is caught even if
    nbformat has been monkey-patched in a test session."""
    with NOTEBOOK.open("r", encoding="utf-8") as fh:
        decoded = json.load(fh)
    assert isinstance(decoded, dict)
    assert "cells" in decoded, "Notebook JSON is missing the 'cells' key."
    assert decoded.get("nbformat", 0) >= 4
