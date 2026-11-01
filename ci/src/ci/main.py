"""Reproducible CI / release pipeline for recall_guard, as a Dagger module.

Every step runs in a container, so it behaves identically on a laptop and on a CI
runner (Req 6.1, 6.4). GitHub Actions is a thin control plane that *calls* these
functions rather than re-implementing the steps in YAML.

Functions:
- ``base``        — the root Dockerfile's locked development target.
- ``test``        — run the suite for one Python version.
- ``test_matrix`` — run the suite across the supported Python matrix (Req 6.2).
- ``lint``        — run ruff plus the structural-architecture gate (Req 6.3). The gate
                    (``scripts/check_architecture.py``, stdlib-only) enforces the
                    project import-direction rules in-container;
                    complexity and cycles are reviewed locally without a Sentrux requirement.
- ``build``       — build the wheel + sdist; returns ``dist/`` (Req 6.1, 6.5).
- ``docs``        — strict-build the documentation site; returns ``site/``.

Distribution (the "publish" pipeline op) is a GitHub Release built from ``build``
output, created by the release workflow; there is no PyPI publish.
"""

from typing import Annotated

import dagger
from dagger import dag, function, object_type

#: Supported interpreter matrix (consumer pin + dev).
PYTHONS = ("3.12", "3.14")

# Filter secrets and bulky local state before host upload. The complete build
# exclusions are still read from .dockerignore in base().
SourceDirectory = Annotated[
    dagger.Directory,
    dagger.Ignore([
        "**/.env", "**/.env.*", "**/*.pem", "**/*.key", "**/.aws", "**/.ssh",
        "**/.venv", "**/venv", "**/.git", ".agent", ".agents", ".claude",
        ".codex", ".kiro", "HANDOFF.md", "ci/sdk", "runs", "site", "dist",
        "build", "wheels", "**/__pycache__", "data/lookaheadbench",
    ]),
]


@object_type
class Ci:
    @function
    async def base(self, source: SourceDirectory, python: str = "3.12") -> dagger.Container:
        """Build the root Dockerfile's locked development environment."""
        if python not in PYTHONS:
            raise ValueError(f"Unsupported Python version {python!r}; choose one of {PYTHONS}")
        rules = (await source.file(".dockerignore").contents()).splitlines()
        exclusions = [rule.strip() for rule in rules if rule.strip() and not rule.lstrip().startswith("#")]
        if any(rule.startswith("!") for rule in exclusions):
            raise ValueError("Dagger context filtering requires exclusion-only .dockerignore rules")
        context = dag.directory().with_directory("/", source, exclude=exclusions)
        return context.docker_build(
            target="dev",
            build_args=[dagger.BuildArg(name="PYTHON_VERSION", value=python)],
        )

    @function
    async def test(self, source: SourceDirectory, python: str = "3.12") -> str:
        """Run the test suite for one Python version (Req 6.2)."""
        container = await self.base(source, python)
        return await container.with_exec(["uv", "run", "--no-sync", "pytest", "-q"]).stdout()

    @function
    async def test_matrix(self, source: SourceDirectory) -> str:
        """Run the suite across the supported Python matrix (Req 6.2)."""
        lines: list[str] = []
        for python in PYTHONS:
            out = await self.test(source, python)
            tail = out.strip().splitlines()[-1] if out.strip() else "(no output)"
            lines.append(f"=== python {python} ===\n{tail}")
        return "\n".join(lines)

    @function
    async def lint(self, source: SourceDirectory) -> str:
        """Run ruff plus the structural-architecture gate (Req 6.3)."""
        container = await self.base(source)
        return await (
            container
            .with_exec(["uv", "run", "--no-sync", "ruff", "check", "."])
            .with_exec(["uv", "run", "--no-sync", "python", "scripts/check_architecture.py"])
            .stdout()
        )

    @function
    async def build(self, source: SourceDirectory) -> dagger.Directory:
        """Build the wheel + sdist; returns the ``dist/`` directory (Req 6.1, 6.5)."""
        container = await self.base(source)
        return (
            container
            .with_exec(["uv", "build", "--out-dir", "/outputs/dist"])
            .directory("/outputs/dist")
        )

    @function
    async def docs(self, source: SourceDirectory) -> dagger.Directory:
        """Strict-build the documentation site; returns the ``site/`` directory."""
        container = await self.base(source)
        return (
            container
            .with_env_variable("DISABLE_MKDOCS_2_WARNING", "true")
            .with_exec(["uv", "run", "--no-sync", "mkdocs", "build", "--strict", "--site-dir", "/outputs/site"])
            .directory("/outputs/site")
        )
