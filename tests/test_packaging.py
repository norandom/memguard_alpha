"""Packaging-contract tests (review-hardening Req 6.2, 6.3).

Pins the one truthful install contract: consumers get exactly the
``backtest`` and ``docs`` extras; ``dev`` and ``pipeline`` stay uv
dependency groups (checkout tooling via ``uv sync``), and the default
runtime dependency set stays lean.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = _ROOT / "pyproject.toml"
CITATION = _ROOT / "CITATION.cff"
README = _ROOT / "README.md"


def _load() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def test_consumer_extras_are_exactly_backtest_and_docs() -> None:
    data = _load()
    assert set(data["project"]["optional-dependencies"]) == {"backtest", "docs"}


def test_dev_and_pipeline_are_dependency_groups_not_extras() -> None:
    data = _load()
    assert {"dev", "pipeline"} <= set(data["dependency-groups"])
    assert "dev" not in data["project"].get("optional-dependencies", {})
    assert "pipeline" not in data["project"].get("optional-dependencies", {})


def test_default_runtime_dependency_set_stays_lean() -> None:
    data = _load()
    roots = {
        dep.split("[")[0].split(">")[0].split("=")[0].split("<")[0].strip()
        for dep in data["project"]["dependencies"]
    }
    assert roots == {
        "numpy", "scikit-learn", "rich", "pyyaml", "requests", "python-dotenv",
    }
    # The heavyweight stacks must never creep into the default install.
    assert not ({"matplotlib", "vectorbt", "mkdocs", "pytest", "dagger-io"} & roots)


def test_citation_doi_matches_the_released_version() -> None:
    """This drifted once: CITATION.cff carried v0.2.0's DOI under version 0.4.1.

    A stale per-version DOI silently sends citers to the wrong artifact, so the
    released version, the citation metadata, and the README's Zenodo list have to
    agree on one line.
    """
    cff = yaml.safe_load(CITATION.read_text(encoding="utf-8"))
    version = _load()["project"]["version"]
    assert cff["version"] == version
    assert f"- v{version}: <https://doi.org/{cff['doi']}>" in README.read_text(encoding="utf-8")
