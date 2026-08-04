"""Docs-site content checks (Task 3.1, 3.2).

Covers the static deliverables for Requirements 8.1, 8.5 (config is autodoc + strict)
and 9.1, 9.2, 9.3 (uv git-dependency recipe, runnable façade example, input/
responsibility split). The strict mkdocs *build* itself is exercised by the pipeline
docs operation and the end-to-end validation, not in this unit suite.
"""

from __future__ import annotations

from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def _index() -> str:
    return (_ROOT / "docs" / "index.md").read_text(encoding="utf-8")


def test_docs_files_exist() -> None:
    assert (_ROOT / "mkdocs.yml").exists()
    assert (_ROOT / "docs" / "index.md").exists()
    assert (_ROOT / "docs" / "gen_ref_pages.py").exists()


def test_mkdocs_config_is_strict_and_autodoc() -> None:
    cfg = (_ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    assert "strict: true" in cfg
    assert "mkdocstrings" in cfg
    assert "gen-files" in cfg


def test_index_has_uv_git_dependency_recipe() -> None:
    text = _index()
    assert "git+https://github.com/norandom/memguard_alpha.git" in text
    assert "uv add" in text


def test_index_has_runnable_facade_example() -> None:
    text = _index()
    assert "MemoryGuardedScorer.calibrate(" in text
    assert ".score(" in text
    assert "p_memorized" in text
    assert "memguard_confidence" in text


def test_index_documents_input_responsibility_split() -> None:
    lowered = _index().lower()
    assert "you provide" in lowered
    assert "recall_guard owns" in lowered
    assert "not** own" in lowered or "not own" in lowered


# --- ensemble surface (task 5.3) ---------------------------------------------


def _ensemble() -> str:
    return (Path(__file__).resolve().parents[2] / "docs" / "ensemble.md").read_text(
        encoding="utf-8"
    )


def test_ensemble_page_is_in_the_nav() -> None:
    mkdocs = (Path(__file__).resolve().parents[2] / "mkdocs.yml").read_text(encoding="utf-8")
    assert "ensemble.md" in mkdocs


def test_ensemble_page_names_the_exposure_multiplier() -> None:
    """Exactly one reported value scales exposure, and the page must say which."""
    text = _ensemble()
    assert "p_memorized_point" in text
    assert "exposure multiplier" in text.lower()
    assert "not the multiplier" in text.lower()


def test_ensemble_page_states_the_estimator_difference_as_withheld_exposure() -> None:
    """A score delta reads like rounding; a haircut delta reads like money."""
    lowered = _ensemble().lower()
    assert "haircut" in lowered
    assert "21.1%" in lowered and "13.6%" in lowered


def test_ensemble_page_carries_the_honesty_caveats() -> None:
    lowered = _ensemble().lower()
    assert "independent draws" in lowered
    assert "narrower than its label" in lowered
    assert "conditional on the draws that parsed" in lowered
    assert "separated clusters only" in lowered
    assert "descriptive" in lowered
    assert "provisional" in lowered


def test_ensemble_page_documents_cost_and_quota() -> None:
    lowered = _ensemble().lower()
    assert "rate-limited" in lowered
    assert "quota" in lowered
    assert "retain_draws" in lowered
    assert "estimate_cost" in lowered


def test_ensemble_page_documents_shared_reference_dispersion() -> None:
    """This caveat compounds with draw dependence; both must appear together."""
    lowered = _ensemble().lower()
    assert "shared reference" in lowered or "reference model is drawn once" in lowered
    assert "understates" in lowered
    assert "compound" in lowered


def test_ensemble_page_states_the_opt_in_contract() -> None:
    lowered = _ensemble().lower()
    assert "opt-in" in lowered
    assert "ensemblespec" in lowered
