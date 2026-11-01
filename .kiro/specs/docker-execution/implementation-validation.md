# Docker implementation validation — 2026-11-01

Docker core tasks 1.1–3.1 have executable evidence below. The selected fallback
smoke in task 3.2 depends on unfinished readiness 4.1 and remains deferred; this
report does not accept that feature, claim mitigation efficacy or diagnose host
Python behavior. Current user direction is Docker-only validation.

## Results

| Check | Actual result |
| --- | --- |
| Shared recipe feasibility | Docker and actual Dagger CLI/engine 0.21.4 passed minimal target/build-argument/context checks for Python 3.12.12 and 3.14.0, uv 0.9.7 |
| Development image | Frozen 165 dependencies plus package; non-root UID 1000; host environment/VCS/credential inputs absent; output bind persisted |
| Lean runtime | Frozen 16 runtime dependencies plus installed wheel; import under /opt/venv from /tmp with networking disabled; no checkout, pytest, Jupyter, matplotlib, vectorbt or Dagger SDK |
| Compose offline example | Accepted ensemble, fidelity rejection with zero target draws, unavailable MCS; actual_http_requests0 |
| Notebook profile | Unauthenticated API403; host127.0.0.1:8888; no provider keys; writable persistent output; service/network removed after smoke |
| Pinned Dagger base/build/docs | All exit0; dev target runs as app; wheel/sdist and strict site output contracts preserved, including new development guide |
| Final full container regression | **2385 passed in 279.69s**, exit 0, Python 3.12.12 dev image |
| Focused corrected regression | Entire prepared-execution file: 33 passed in 7.98s, exit 0 |
| Final Docker static checks | Ruff over project plus explicit CI file: All checks passed; architecture check OK; exit0 |

The first full Dagger run completed with 2381 passing and 4 failing tests. All four failures were an
incorrect assertion referencing nonexistent ModelResponse.complete in the
previously unfinished readiness test. The one-line correction requires the
existing ModelResponse type, retaining redaction/numeric evidence/usage/count/
failure assertions. No runtime implementation was changed. Focused verification
passed before the final full Docker regression above. This concrete failure is
why a second full run was justified; matrices were not repeated per helper.

## Review outcome

Independent task-local reviews approved tasks 1.1, 1.2, 1.3, 2.1, 2.2 and 3.1.
Task 1.1 required one remediation to execute the actual pinned Dagger version;
the subsequent reviews found no blocking issues. Final task 3.1 review reused
the source-matched passing container receipts rather than repeating the suite.
Core acceptance is GO. Feature-level selected-fallback smoke remains pending
on readiness 4.1, explicitly outside the current Docker-only work.

## Commands and provenance

Validated from the repository root with the existing configured Docker/WSL
route. Commands use the baked Linux environment, not host interpreter paths:

```text
docker compose build dev
docker compose run --rm dev
docker compose run --rm dev uv run --no-sync pytest -q
docker compose run --rm dev sh -c 'uv run --no-sync ruff check . ci/src/ci/main.py && uv run --no-sync python scripts/check_architecture.py'
docker build --target runtime -t recall-guard:runtime .
docker run --rm --network none --workdir /tmp recall-guard:runtime
```

Notebook checks used explicit profile startup, a local unauthenticated API GET,
port inspection and profile shutdown. The exact pinned CLI was downloaded to a
temporary location; its installed counterpart was not upgraded. Dagger's normal
engine lifecycle replaced the previous0.21.8 engine during compatibility work;
the original installed0.21.8 CLI subsequently restored its default engine via a
successful no-module/no-image-pull GraphQL container-ID query. Other application
containers were not changed.

Base image manifest-list digests:

- Python3.12: `sha256:4f52717de41541452f5318571b05da783d2ddf346a94b4d5c6512a4b51e986bd`.
- Python3.14: `sha256:1a11a7a4b22aea0f74a8b479441ba2c54093094767c219632d83a9e7ccb18d57`.

Final dev image: `sha256:41755c9767cce2e49cf9b83ee3f8ea33c6bccfc31ccc1fc07e33cebcdc56a5e8`.
Runtime acceptance image: `sha256:28d6ea0f7361e80a46a2e5a7510a8d395dcd39a9c00e1b9d77dae4dffa7cb61f`.
The runtime image preceded documentation-only changes; its Dockerfile/runtime
package implementation is unchanged. No full Python3.14 suite or native-host
matrix is claimed by this report; existing release matrix configuration remains.

Docker core commits: 1e5c920, 0d82257, 5340400, 96362c6 and 1870eef.
Validation includes the preserved uncommitted readiness implementation and tests
in the current worktree. Passing regression is not independent acceptance of
readiness3.2; those changes remain uncommitted and outside the Docker delivery.
Local detailed receipts are runs/docker-1-1.md through docker-2-2.md.

No live inference, credential eligibility probe, calibration update or pilot
allowance was consumed. Intermittent WSL connection errors before command startup
were retained as tool failures and not interpreted as test results or library
defects. Docker/Compose/installed Dagger versions were29.9.2/5.6.1/0.21.8;
actual pinned compatibility/CI checks used0.21.4.
