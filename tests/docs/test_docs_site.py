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
