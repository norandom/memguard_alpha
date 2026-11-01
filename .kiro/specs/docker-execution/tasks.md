# Tasks: shared Docker execution

Proposal — 2026-11-01. No builds, source cleanup or live inference are part of
writing this specification. Tasks are sequential; existing CI remains working
until its shared-recipe replacement is verified.

- [ ] 1. Build and verify the shared locked environment
- [x] 1.1 Verify the existing Dagger SDK can consume the shared recipe
  - Exercise the pinned SDK's Dockerfile build, target/build arguments and source exclusion handling with a small local build spike; do not upgrade Dagger or create a second installation recipe.
  - Select concrete compatible uv/Python image pins for default 3.12 and supported 3.14 and record them once through the Dockerfile versioned image template.
  - Done when Docker and the current SDK build the same minimal target, select Python correctly and exclude a synthetic secret canary. If the SDK contract fails, record the exact incompatibility before changing CI or proposing an upgrade.
  - _Boundary: Shared build recipe feasibility_
  - _Requirements: 1.1, 1.2, 1.3, 3.1_
- [ ] 1.2 Create the non-root development image and bounded build context
  - Install frozen dev dependencies from uv.lock, separate the container environment from host mounts, copy required source/fixtures and exclude host venv, credentials, generated artifacts and agent state.
  - Provision build/test backend resources needed by existing packaging tests at build time; require no real provider keys. Keep source baked in and writable home/tmp/output locations explicit.
  - Done when focused tests, Ruff and architecture commands run through uv run --no-sync, context canaries are absent, and non-root output files persist to the host without reusing its environment.
  - _Boundary: Dockerfile development target and context_
  - _Requirements: 1.1, 1.3, 2.1, 2.2, 2.3, 3.1, 3.2, 3.3, 3.4_
- [ ] 1.3 Build and smoke the lean installed-package target
  - Reuse locked runtime dependencies and the existing wheel build; exclude dev/notebook/plot tooling and source tree. No resolving or syncing dependencies on startup.
  - Follow the design's locked runtime-dependencies → built wheel → no-dependency wheel installation path; use the same environment path and Python base when copying into runtime.
  - Done when the runtime target imports the installed wheel from outside the checkout as non-root, exposes correct metadata and contains no development extras. Direct non-container wheel use remains unchanged.
  - _Boundary: Dockerfile runtime target_
  - _Requirements: 1.1, 1.4, 2.1, 3.2, 3.3, 3.4, 4.1_

- [ ] 2. Integrate local commands and existing CI
- [ ] 2.1 Add explicit developer and optional notebook commands
  - Define the dev service with no ports/credentials and an explicit output mount; do not bind the host checkout/venv by default.
  - Add an opt-in authenticated notebook service using the dev image with host publication limited to 127.0.0.1. Show runtime-only credential injection only for an explicit existing live protocol.
  - Done when build/test commands and notebook launch work, token authentication is required, output files persist, and startup makes zero provider calls. Shut down smoke services afterward.
  - _Boundary: Compose commands and mounts_
  - _Requirements: 1.2, 2.2, 2.3, 2.4, 3.2, 3.3, 3.4_
- [ ] 2.2 Switch Dagger to the shared recipe without changing outputs
  - Replace duplicate environment construction with the verified Dockerfile dev target and supported Python selector. Preserve test/lint/build/docs function inputs, wheel/sdist/site outputs and existing release matrix/shims.
  - Use the already-installed environment without incidental synchronization; do not mount a Docker socket or add a nested daemon.
  - Write artifacts under the non-root writable output directory and return wheel/sdist/site directories through the unchanged function contracts.
  - Done when one Dagger integration receipt demonstrates the shared build and unchanged function output contracts; no second dependency-install recipe remains.
  - _Boundary: Dagger build integration_
  - _Depends: 1.1, 1.2, 1.3_
  - _Requirements: 1.1, 1.2, 1.3, 2.2, 3.4, 4.1, 4.4_

- [ ] 3. Validate and document the shared execution path
- [ ] 3.1 Accept the image and publish runnable adopter/engineering instructions
  - Run one coherent offline full regression, lint/architecture, packaging, strict docs, installed-wheel/runtime import and notebook/output smoke. Reuse unchanged receipts rather than repeating whole suites per reviewer.
  - Use acceptance.md for command/output checks and delivery-plan.md for cross-spec order; record only a short dated receipt, not a new validation framework.
  - Use COBESY for README and MkDocs instructions: prerequisites, build/test/focused commands, outputs, optional notebook and explicit live-run limits. Explain direct Python use and provider availability independently of container reproducibility.
  - Run the existing offline completion example. Done when copyable commands match verified behavior and all acceptance receipts identify image/source/config, with zero provider calls or credentials required.
  - _Boundary: Container integration acceptance and adopter documentation_
  - _Depends: 2.1, 2.2_
  - _Requirements: 1.4, 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.4, 4.1, 4.2, 4.3, 4.4_
- [ ] 3.2 Smoke the selected fallback/report flow when its implementation is ready
  - Reuse the readiness example and deterministic fixtures; do not add a container-specific client or calculation path.
  - Done when the image runs configured capability selection → authorized fallback → existing execution → truthful report/MCS-unavailable output without provider calls.
  - This cross-spec follow-up depends on adaptive-provider-readiness task4.1 and its necessary execution fixes. It does not block tasks1–3.1 or the usable Docker environment.
  - _Boundary: Existing workflow/container smoke integration_
  - _Depends: 3.1_
  - _Requirements: 4.2_
