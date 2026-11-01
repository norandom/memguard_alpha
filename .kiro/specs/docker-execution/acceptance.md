# Acceptance plan: Docker execution

Specification follow-up — 2026-11-01. All checks below are planned. None is a
successful build/test receipt. Reuse the existing test tools and Dagger functions;
do not build a validation controller or another result-storage framework.

## Command contract

Run these from the repository root after the Docker implementation exists.
The same commands apply from PowerShell or the existing Debian WSL shell, using
that shell's repository path and available Docker daemon. A Docker client alone
is insufficient: establish daemon availability with `docker info` and Compose
availability with `docker compose version` once, before starting image work.
Do not install or reconfigure either automatically.

```text
docker compose build dev
docker compose run --rm dev uv run --no-sync pytest -q
docker compose run --rm dev uv run --no-sync ruff check .
docker compose run --rm dev uv run --no-sync python scripts/check_architecture.py
docker compose run --rm dev uv run --no-sync python examples/completion_workflow.py
docker compose run --rm dev uv build --out-dir /outputs/dist
docker compose run --rm dev uv run --no-sync mkdocs build --strict --site-dir /outputs/site
docker compose --profile notebook up notebook
docker compose --profile notebook down
```

The existing strict-docs environment flag, if still required, is supplied by the
image/service as in current Dagger. Dev must have an empty entrypoint or a
compatible command contract, so commands do not become `uv uv ...`.
Packaging may acquire build-backend resources during the explicit build phase;
this is independent of provider inference. No general air-gap promise is made.

The runtime image has a stable local tag `recall-guard:runtime` for these checks;
Compose's dev build need not build it implicitly:

```text
docker build --target runtime -t recall-guard:runtime .
docker run --rm --network none --workdir /tmp recall-guard:runtime python -c "import recall_guard; print(recall_guard.__file__)"
```

Here `python` is resolved inside the Linux image through its installed environment
on PATH. It is not a hard-coded host interpreter invocation. The import must
resolve under /opt/venv, with no checkout on PYTHONPATH or in the working directory.

## Observable acceptance

| Check | Observable result | Owner/task |
| --- | --- | --- |
| Shared build | Docker and pinned Dagger build dev using the same Dockerfile; 3.12 default and 3.14 selection are demonstrated | 1.1, 2.2 |
| Frozen dependencies | uv.lock is unchanged; no runtime sync; dev contains locked tools; runtime contains only runtime packages and installed Recall Guard | 1.2, 1.3 |
| Context exclusions | Synthetic root/nested credential canaries, host venv and agent scratch files are absent from supplied build contexts and final images; tracked fixtures remain | 1.1, 1.2 |
| Permissions and output | Non-root user writes a harmless marker under /outputs; it appears under runs/container and the host can read/remove it; host source and venv are untouched | 1.2, 2.1 |
| Runtime wheel | Import from /tmp with networking disabled resolves installed package metadata and code under /opt/venv, without checkout, pytest, Jupyter, matplotlib, vectorbt or Dagger SDK | 1.3 |
| Offline workflow | Existing deterministic example completes without credentials or provider inference and preserves current output semantics | 3.1 |
| Regression and artifacts | One identified dev image passes suite, lint/architecture, wheel/sdist and strict docs; Dagger returns expected artifact directories | 2.2, 3.1 |
| Notebook | Explicit profile starts the existing environment's kernel; HTTP without token is denied or redirected to login; host port is loopback-only; output location is writable; service is stopped afterward | 2.1, 3.1 |
| Selected fallback | Once readiness4.1 exists, deterministic selected flow reports authorized omission, actual stages/counts and MCS unavailable without token evidence | 3.2 |

For exclusions use temporary synthetic canaries, never existing credentials.
Preserve Dockerfile/.dockerignore in the source supplied to the Dockerfile builder.
Keep temporary probes local to the acceptance run and clean only their exact
owned paths. No recursive runs cleanup and no maintained probe framework.

## Receipt and bounded failure handling

Use a short dated Markdown receipt with source identity (commit plus relevant
uncommitted changes), base image/uv/Python versions and resolved digests,
Docker/Compose/Dagger versions, commands, exit codes and output artifact paths.
Use existing logs where useful. Never include environment dumps or credentials.
Do not introduce another receipt schema, journal or request ledger.

If the daemon is unavailable, record the exact tooling prerequisite and continue
specification/offline library work; do not repeatedly diagnose host Python.
If pinned SDK build support fails, stop CI migration and preserve working Dagger;
propose a specific compatibility fix. If output permission checks fail, correct
mount/ownership configuration and rerun that check. Full regression repeats only
for changed application/image dependencies or a concrete unresolved concern.

The selected fallback smoke can remain pending while the usable Docker image is
accepted. No acceptance command probes NIM/Anthropic/OpenAI/Mistral, spends the
pilot allowance or changes calibration.
