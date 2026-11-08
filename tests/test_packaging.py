"""Packaging-contract tests (review-hardening Req 6.2, 6.3).

Pins the one truthful install contract: consumers get exactly the
``backtest`` and ``docs`` extras; ``dev`` and ``pipeline`` stay uv
dependency groups (checkout tooling via ``uv sync``), and the default
runtime dependency set stays lean.
"""
from __future__ import annotations

import json
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
        "numpy", "scikit-learn", "rich", "pyyaml", "requests", "python-dotenv", "psutil",
    }
    # The heavyweight stacks must never creep into the default install.
    assert not ({"matplotlib", "vectorbt", "mkdocs", "pytest", "dagger-io"} & roots)


def test_experiment_dependencies_are_pinned_checkout_tooling() -> None:
    """Inference tooling must not become a consumer extra or runtime dependency."""
    data = _load()
    inference_packages = {"torch", "transformers", "accelerate", "kernels", "triton"}
    group = data["dependency-groups"]["experiment"]
    assert {dep.split("==")[0] for dep in group} == inference_packages
    assert all("==" in dep for dep in group)
    assert "experiment" not in data["project"]["optional-dependencies"]
    runtime = {dep.split("[")[0].split(">=")[0] for dep in data["project"]["dependencies"]}
    assert not inference_packages & runtime


def test_citation_doi_matches_the_released_version() -> None:
    """This drifted once: CITATION.cff carried v0.2.0's DOI under version 0.4.1.

    A stale per-version DOI silently sends citers to the wrong artifact, so the
    released version, the citation metadata, and the README's Zenodo list have to
    agree on one line.
    """
    cff = yaml.safe_load(CITATION.read_text(encoding="utf-8"))
    version = _load()["project"]["version"]
    assert cff["version"] == version
    readme = README.read_text(encoding="utf-8")
    if cff.get("doi"):
        assert f"- v{version}: <https://doi.org/{cff['doi']}>" in readme
    else:
        release_url = f"https://github.com/norandom/memguard_alpha/releases/tag/v{version}"
        assert cff["url"] == release_url
        assert f"- v{version}: [GitHub Release]({release_url}); version-specific DOI pending." in readme


SHIPPED_PROFILES = {"modal-gpt-oss-20b", "nim-llama3.1-70b", "ollama-gemma3-4b", "ollama-gpt-oss-20b",
                    "ollama-llama3.1-8b", "ollama-llama3.2-3b", "ollama-llava-llama3-ctx2048", "ollama-phi4-mini",
                    "flm-llama3.2-3b", "flm-gemma3-4b", "flm-gemma4-e4b"}


def test_provider_profiles_ship_as_package_data_inside_the_wheel_tree() -> None:
    """C18: curated profiles live under the wheel's only package, not a checkout-only folder."""
    wheel = _load()["tool"]["hatch"]["build"]["targets"]["wheel"]
    assert wheel["packages"] == ["recall_guard"]
    assert not {"exclude", "only-include"} & set(wheel)
    directory = _ROOT / "recall_guard" / "core" / "provider_profiles"
    assert {path.stem for path in directory.glob("*.json")} == SHIPPED_PROFILES
    assert not (_ROOT / "configs" / "providers").exists()


def test_provider_profiles_load_through_importlib_resources() -> None:
    from importlib import resources

    from recall_guard.core.profiles import ProviderProfile, available_profiles, load_profile

    package = resources.files("recall_guard.core").joinpath("provider_profiles")
    loaded = {}
    for item in package.iterdir():
        if item.name.endswith(".json"):
            profile = ProviderProfile.from_dict(json.loads(item.read_text(encoding="utf-8")))
            loaded[profile.profile_id] = profile
    assert set(loaded) == SHIPPED_PROFILES == set(available_profiles())
    assert all(load_profile(profile_id) == profile for profile_id, profile in loaded.items())
