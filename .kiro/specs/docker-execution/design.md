# Design: one Docker environment, existing Python application

Proposal — 2026-11-01. This is tooling for a library, not a hosted application.

## Boundary commitments

Own the root Dockerfile, build-context exclusions, a minimal Compose developer
interface, Dagger environment integration and runnable documentation. Build one
locked Python environment; use the existing API clients and calculations.

### Out of boundary

Provider routing/fallback mathematics, readiness schedulers, live pilots,
credential management services, image publishing, Docker installation, native
host debugging, deployment services and changes to the release Python matrix.

### Allowed dependencies

Docker/Compose for local tooling; the existing Dagger module for CI; existing
uv-managed project dependencies and build backend. No library runtime dependency
is added. Do not introduce a second lockfile or dependency-install script.

### Revalidation triggers

Base image, uv pin, lockfile, build targets or Dagger build integration changes
require rebuilding and appropriate container acceptance. A domain-code change
uses its focused tests and the coherent integration boundary. Provider/model/
generation changes require separate explicit live protocol approval where needed.

## Existing foundation and adoption choice

Ci.base currently builds a uv/Python bookworm-slim image and performs uv sync
--frozen. Dagger already runs tests, lint, package and docs commands. Adopt Docker
as the shared build recipe and keep Dagger as orchestration. Do not maintain a
second dependency recipe in Ci.base or add a nested Docker daemon.

The implementation must verify the pinned Dagger SDK's Dockerfile build API,
target/build-argument support and exclusion handling in one small build spike.
This is the first task, before changing working CI. If the current SDK cannot
consume the required recipe, record the exact incompatibility before proposing
any tool upgrade; this design does not authorize a Dagger upgrade or duplicate
fallback build system.

## Image contracts

| Target | Contents and purpose |
| --- | --- |
| dev | uv, selected supported Python, frozen development dependencies, source/tests/docs; default working directory /app |
| runtime | Same pinned Python family and uv, installed wheel plus locked runtime dependencies; no source/test/dev/notebook tree |

Keep one versioned uv/Python image-tag template and default Python selection in
Dockerfile. Verify that concrete versioned tags support both Python3.12 and3.14
in the first task; record resolved digests in research evidence. Dagger passes
only the supported Python selector; Compose uses the Dockerfile default. This
keeps the version pin in one source without a registry/mapping/helper. Do not
silently change existing Dagger CLI/SDK pins. If immutable per-platform digest
pins become required, propose that concrete change separately.

Layer dependency installation before source copying where valid. Keep the Linux
environment at /opt/venv using UV_PROJECT_ENVIRONMENT, outside any optional source
mount. Freeze dependency resolution at build time. The development target runs
uv run --no-sync commands. The runtime target may run its installed Python module
or import directly from the installed environment; it must not resolve a project
or synchronize development dependencies at startup.

Build the wheel using the existing hatchling/uv build path. Install its runtime
dependency set from the existing lock, with no dev group or unrelated extras.
Verify installed metadata/import from outside the checkout and absence of
matplotlib, vectorbt, notebook and test tools in that runtime environment.
Build tools can exist in an intermediate stage; do not carry them into runtime.

The concrete runtime build path is: install only locked runtime dependencies in
an intermediate environment using `uv sync --frozen --no-dev --no-install-project`,
build the package wheel, then install that exact wheel with `uv pip install
--no-deps` into that environment. This last step must not resolve dependencies.
Copy the installed environment at the same absolute path into the runtime stage
derived from the same Python/OS base. Point PATH at its binaries. Keep build
artifacts and checkout metadata out of that final stage. This produces an actual
wheel install rather than an editable checkout advertised as a runtime image.

Use an unprivileged user, writable home/tmp and /outputs. Test the output bind
mount under both Docker Desktop and the existing WSL Docker route when available;
never assume a Linux UID fixes every host's file permissions. Host paths appear
only in tooling/mount configuration, never in library code.

## Context, credentials and outputs

The .dockerignore must exclude .git, .venv, nested .env files, agent state, runs,
site, caches, generated dist/build, ci/sdk and the downloaded lookahead corpus.
Retain tracked fixtures/test data needed by tests. Verify exclusions using a
synthetic canary, without reading or printing real credentials. Ensure Dagger's
source Directory is filtered equivalently before Dockerfile construction: do not
assume Docker CLI ignore behavior automatically applies to every Dagger API.

Default source is copied into the image. Compose mounts only
./runs/container:/outputs; source edits take effect after a cached rebuild.
An optional notebook-directory mount may be provided for editing, never a host
.venv or whole checkout carrying .env files. Generated notebook outputs need an
explicit writable location. Runtime environment variables are not persisted in
reports or image metadata.

Local live commands may use docker run --env-file .env against an explicit
existing --run protocol. This is Docker run's runtime injection, not Compose's
interpolation-only --env-file behavior. Default services define no credential
env_file. No _FILE adapter or secret manager is needed for this local workflow.
Do not print resolved secret-containing Compose configurations.

## Compose and command contracts

The dev service builds target dev, has no published ports or credentials, and
defaults to a harmless offline command. Commands are passed without an entrypoint
that duplicates uv. Proposed usage, to be verified when implemented:

```text
docker compose build dev
docker compose run --rm dev uv run --no-sync pytest -q
docker compose run --rm dev uv run --no-sync ruff check .
docker compose run --rm dev uv run --no-sync python scripts/check_architecture.py
docker compose --profile notebook up notebook
```

The notebook service is opt-in, reuses dev and mounts outputs. Inside the
container its server listens on 0.0.0.0; host publication is
127.0.0.1:8888:8888. Retain Jupyter token authentication. Do not run provider probes
in image entrypoints, import hooks, notebook startup or healthchecks. No daemon
service, socket mount, privileged mode or GPU configuration is required.

## Dagger integration and compatibility

Ci.base consumes the root Dockerfile dev target with the selected base pin.
Existing test/lint/build/docs functions retain their inputs and output contracts;
commands use the already-built environment with --no-sync. Packaging returns
wheel/sdist and docs returns site as today. Reuse Dockerfile build stages for
packaging if necessary instead of reinstalling dependencies in a separate recipe.
Preserve existing GitHub Actions shims and release compatibility jobs.

This is a tooling migration: leave working Ci.base in place until the shared
recipe build spike succeeds, then switch it as one integration change. Docker
acceptance does not assert a diagnosis or fix for unexplained native process exits.

The documented integration seam is `Directory.docker_build` with a dev target
and a Python build argument. The current official Python SDK reference exposes
both controls; the repository's generated SDK is absent, so compatibility with
engineVersion v0.21.4 still belongs to task1.1. Do not infer pinned-version
compatibility from the latest documentation. Source filtering must be verified
before passing the context; no build secrets or SSH socket are required.

Existing build/docs outputs need an explicit writable directory under non-root
execution. Use /outputs/dist and /outputs/site internally, returning those
directories from Dagger as before. Preserve the function-level wheel/sdist/site
contracts without assuming an unprivileged process can create /dist or /site.

## Validation strategy

1. Inspect a synthetic context-exclusion canary and verify non-root write/import
   behavior, frozen installation and distinct dev/runtime contents.
2. Run focused checks while implementing. At integration, use one dev image for
   the full offline suite, Ruff/architecture, wheel/sdist build and strict MkDocs.
   Provision build/test backend resources at image-build time where required by
   packaging tests. Do not claim an air-gapped suite until demonstrated; all
   acceptance commands must avoid real provider APIs and require no keys.
3. Smoke the installed wheel from outside /app and the runtime target; smoke the
   optional notebook launch, token authentication, localhost publication and
   writable output path. Shut down test services afterward.
4. Run the existing offline completion example. When adaptive readiness task4.1
   is implemented, verify its selected fallback/report path in the same image.
   Docker core acceptance does not depend on unfinished readiness execution.
5. Compare Dagger function outputs to existing contracts and preserve release
   matrix configuration. Do not run the full matrix per helper or reviewer.

The concrete acceptance matrix and stop conditions are in
[acceptance.md](acceptance.md). The cross-spec order is in
[delivery-plan.md](delivery-plan.md). Both describe future checks, not executed
results. A missing Docker daemon is a tooling prerequisite, not a library defect.

## File structure plan

| Path | Change and responsibility |
| --- | --- |
| Dockerfile | New shared dev/runtime build stages and locked installation |
| .dockerignore | New explicit build-context exclusions |
| compose.yaml | New dev and optional notebook commands/mounts/localhost publication |
| ci/src/ci/main.py | Consume shared Dockerfile; preserve orchestration contracts |
| README.md | Adopter quick start and explicit live-run boundary, using COBESY |
| docs/development.md; mkdocs.yml | Engineering commands, outputs, troubleshooting and navigation, using COBESY |
| Existing container/packaging tests | Extend only where needed to protect installed-wheel, context or output contracts; no new test framework |

Do not create an image registry, configuration framework or helper package for
version selection. Verify the versioned template before changing working CI.

## Requirements traceability

| Requirements | Design/acceptance |
| --- | --- |
| 1.1, 1.2, 1.3, 1.4 | Shared Dockerfile/lock, verified SDK integration, pins, optional tooling |
| 2.1, 2.2, 2.3, 2.4 | Image contracts, frozen runtime commands, mounts, notebook smoke |
| 3.1, 3.2, 3.3, 3.4 | Context canary, runtime-only keys, explicit calls, non-root image |
| 4.1, 4.2, 4.3, 4.4 | Integration acceptance, selected flow follow-up, COBESY docs, preserved Dagger/release outputs |
