"""Reproducible CI / release pipeline for recall_guard, as a Dagger module.

Every step runs in a container, so it behaves identically on a laptop and on a CI
runner (Req 6.1, 6.4). GitHub Actions is a thin control plane that *calls* these
functions rather than re-implementing the steps in YAML.

Functions:
- ``base``        — the project container (uv image + synced environment).
- ``test``        — run the suite for one Python version.
- ``test_matrix`` — run the suite across the supported Python matrix (Req 6.2).
- ``lint``        — run ruff (Req 6.3). The sentrux structural check runs via its own
                    plugin tooling, which is not pip-installable into a container.
- ``build``       — build the wheel + sdist; returns ``dist/`` (Req 6.1, 6.5).
- ``docs``        — strict-build the documentation site; returns ``site/``.
- ``publish``     — token-based upload to a package index (Req 6.1). CI publishing
                    prefers OIDC trusted publishing via the release workflow, so this
                    is the explicit-token / local fallback path.
"""

import dagger
from dagger import dag, function, object_type

#: Supported interpreter matrix (consumer pin + dev).
PYTHONS = ("3.12", "3.14")

#: uv-provided images carry uv + the requested CPython, pinned per call.
_UV_IMAGE = "ghcr.io/astral-sh/uv:python{python}-bookworm-slim"


@object_type
class Ci:
    @function
    def base(self, source: dagger.Directory, python: str = "3.12") -> dagger.Container:
        """Project container for ``python``: uv image + a synced, cached environment."""
        return (
            dag.container()
            .from_(_UV_IMAGE.format(python=python))
            .with_mounted_cache("/root/.cache/uv", dag.cache_volume(f"uv-cache-{python}"))
            .with_env_variable("UV_PYTHON", python)
            .with_directory("/src", source)
            .with_workdir("/src")
            .with_exec(["uv", "sync", "--frozen"])
        )

    @function
    async def test(self, source: dagger.Directory, python: str = "3.12") -> str:
        """Run the test suite for one Python version (Req 6.2)."""
        return await (
            self.base(source, python).with_exec(["uv", "run", "pytest", "-q"]).stdout()
        )

    @function
    async def test_matrix(self, source: dagger.Directory) -> str:
        """Run the suite across the supported Python matrix (Req 6.2)."""
        lines: list[str] = []
        for python in PYTHONS:
            out = await self.test(source, python)
            tail = out.strip().splitlines()[-1] if out.strip() else "(no output)"
            lines.append(f"=== python {python} ===\n{tail}")
        return "\n".join(lines)

    @function
    async def lint(self, source: dagger.Directory) -> str:
        """Run ruff over the project (Req 6.3)."""
        return await (
            self.base(source).with_exec(["uv", "run", "ruff", "check", "."]).stdout()
        )

    @function
    def build(self, source: dagger.Directory) -> dagger.Directory:
        """Build the wheel + sdist; returns the ``dist/`` directory (Req 6.1, 6.5)."""
        return (
            dag.container()
            .from_(_UV_IMAGE.format(python="3.12"))
            .with_directory("/src", source)
            .with_workdir("/src")
            .with_exec(["uv", "build", "--out-dir", "/dist"])
            .directory("/dist")
        )

    @function
    def docs(self, source: dagger.Directory) -> dagger.Directory:
        """Strict-build the documentation site; returns the ``site/`` directory."""
        return (
            self.base(source)
            .with_env_variable("DISABLE_MKDOCS_2_WARNING", "true")
            .with_exec(["uv", "run", "mkdocs", "build", "--strict", "--site-dir", "/site"])
            .directory("/site")
        )

    @function
    async def publish(
        self,
        dist: dagger.Directory,
        token: dagger.Secret,
        index_url: str = "https://upload.pypi.org/legacy/",
    ) -> str:
        """Upload built artifacts to a package index with an explicit token (Req 6.1).

        CI release publishing prefers OIDC trusted publishing (handled by the release
        workflow's publish action), so this token path is the local / non-OIDC fallback.
        """
        return await (
            dag.container()
            .from_(_UV_IMAGE.format(python="3.12"))
            .with_directory("/dist", dist)
            .with_secret_variable("UV_PUBLISH_TOKEN", token)
            .with_exec(
                ["sh", "-c", f"uv publish --publish-url {index_url} /dist/*"]
            )
            .stdout()
        )
