# Research and decisions — 2026-11-01

## Scope and existing evidence

Integration-focused discovery using kiro-spec-design and kiro-spec-tasks.
The repository's local design template is absent; use the skill's basic
boundary/component/flow/file-plan/validation/traceability structure.

Existing ci/src/ci/main.py already builds a uv/Python bookworm-slim environment
with uv sync --frozen. Dagger owns test/lint/build/docs; it is not a runtime
dependency. The root has no Dockerfile, Compose file or .dockerignore yet.
The project lock and dev dependency group already cover the needed tools.

## Primary sources

- [uv Docker integration](https://docs.astral.sh/uv/guides/integration/docker/):
  adopt uv's images, frozen lock installation and dependency caching. Keep the
  container environment separate from host environments; do not add a second lock.
- [Docker build contexts](https://docs.docker.com/build/concepts/context/):
  exclude credentials and generated state explicitly. Verify equivalent filtering
  when the Dagger source Directory is supplied, rather than assuming CLI behavior.
- [Docker port publishing](https://docs.docker.com/engine/network/port-publishing/):
  publish the optional notebook port on host loopback, with Jupyter auth retained.
- [Compose secrets](https://docs.docker.com/compose/how-tos/use-secrets/):
  runtime secret files exist, but no new secret-file API is justified here.
  Reuse current environment-variable credentials for explicitly invoked local runs.

## Synthesis

Adopt the existing lock, Docker and Dagger; build only the root Dockerfile and
thin Compose interface. No new scheduler, service, daemon or deployment system.
Use dev and lean runtime targets because CI tools are not consumer dependencies.
Keep Python selection and the versioned uv base template in the Dockerfile;
callers select supported Python without duplicating dependency installation.

## Explicit implementation prerequisites and limits

- Verify pinned Dagger SDK target/build-argument/context behavior first. No SDK
  upgrade is assumed. Keep working CI until the shared recipe is proven.
- Verify actual compatible image tags, record versions/digests; this proposal
  deliberately invents no pin or successful build receipt.
- Validate non-root output permissions on available host mount routes. Host
  behavior belongs to tooling configuration, not the library's calculations.
- Existing packaging tests may need build resources; provision them in the image.
  No air-gap claim is made until measured. No real provider APIs are needed.
- Docker core acceptance can precede unfinished adaptive readiness. Its selected
  fallback smoke is an explicit cross-spec follow-up, avoiding a dependency cycle.

## Specification follow-up — 2026-11-01

Reviewed pyproject.toml, ci/pyproject.toml, ci/dagger.json and packaging tests.
The module pins engineVersion v0.21.4; its generated ci/sdk is absent. Keep this
separate from the library's Python minimum and from runtime image dependencies.

The [official Python SDK reference](https://dagger-io.readthedocs.io/en/latest/client.html#dagger.Directory.docker_build)
documents Directory.docker_build with target/build_args. This supports the
chosen integration seam, but latest-reference support does not verify the
pinned SDK. The tagged source fetch was unavailable; task1.1 retains the concrete
compatibility build check. No SDK upgrade is inferred.

The [uv Docker guide](https://docs.astral.sh/uv/guides/integration/docker/#non-editable-installs)
documents copying a non-editable environment between stages. This project's
design tightens that to installation of the built wheel without dependency
resolution, after frozen runtime-dependency installation. Identical Python/base
and environment paths preserve the installed environment across stages.

Added acceptance.md with observable checks, short receipts and bounded stop
conditions. Added delivery-plan.md to make Docker-first order, no-cycle fallback
smoke and availability-versus-efficacy measurement explicit. These are planning
artifacts; no image was built and no provider was contacted.
