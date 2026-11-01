# Requirements: shared Docker execution

Draft proposal — 2026-11-01. Requested by the user as a simpler way to run this
portable Python/API project. This specification does not authorize live calls,
install Docker, or change the library's domain behavior.

## Requirement 1: One dependency and build recipe

1. Docker development/test execution shall use pyproject.toml and uv.lock as its dependency authority, with frozen installation and no separate requirements file or second lockfile.
2. Docker CLI, Compose and existing Dagger CI shall use one root Dockerfile for the project environment; Dagger shall remain the existing orchestration layer.
3. The default development image shall use Python 3.12; existing CI shall retain its supported 3.12/3.14 selection without duplicate install recipes. Resolved base-image versions/digests shall be recorded and shared rather than separately guessed by each caller.
4. The installed library shall remain usable in ordinary Python and existing Jupyter kernels without Docker, Docker SDK or a daemon as runtime dependencies.

## Requirement 2: Small execution surfaces

1. A development target shall include the locked test/lint/docs/notebook tools. A separate runtime target shall contain the installed package and its locked runtime dependencies, excluding development/plot/notebook tools unless explicitly selected by an adopter.
2. Project commands in the development image shall use uv run --no-sync from the source root; changes to locked dependencies shall require an explicit rebuild rather than incidental runtime synchronization.
3. Source shall be baked into the default image. Host virtual environments shall never be reused inside it; outputs shall persist in an explicit writable mount and remain usable by the host user.
4. An optional notebook service shall bind its published host port to 127.0.0.1, keep token authentication enabled, and use the same development environment. Notebook startup shall not probe providers or inject credentials by default.

## Requirement 3: Explicit inputs and secrets

1. The build context shall exclude credentials, host environments, VCS state, agent scratch state, runs and generated artifacts while retaining tracked test fixtures required by the suite.
2. Credentials shall be supplied only at runtime through existing environment-variable contracts. Builds, default tests and notebook startup shall not require credentials or embed them in layers, build arguments or tracked configuration.
3. Live execution shall remain an explicit user choice with explicit model, parameters and request/wait allowance. Image build, startup and smoke checks shall make zero provider inference calls.
4. The image shall run as a non-root user with writable temporary/output locations. It shall require no privileged mode, Docker socket, database, GPU or background provider monitor.

## Requirement 4: Useful acceptance and documentation

1. One container integration run shall exercise the offline full regression, lint/architecture checks, package build, strict docs build and installed-wheel import without source-tree import leakage. Focused checks shall remain available independently.
2. Validation shall demonstrate explicit completion-only fallback through the existing workflow, truthful MCS unavailability/reporting, output persistence and an optional notebook smoke. Readiness-specific behavior depends on the corresponding implementation tasks, not on a new container business-logic path.
3. README and MkDocs shall provide copyable build/test/notebook/output and explicit live-run instructions using COBESY structure for adopter and engineering audiences, naming Docker prerequisites and the distinction between container reproducibility and provider availability.
4. Existing Dagger function contracts, release artifacts and compatibility matrix shall be preserved. A shared image change shall require one appropriate integration check; unchanged full suites shall not be rerun per helper/reviewer.
