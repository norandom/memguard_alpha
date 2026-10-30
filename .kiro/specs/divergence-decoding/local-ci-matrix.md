# Local CI Matrix — Debian WSL and Native Windows

Date: 2026-10-30  
Result: **PASS**  
Tested source: commit `a6a55e5` (clean working tree at start).

## Test matrix

| Platform | Python | Tests | Duration |
| --- | --- | --- | --- |
| Linux container through Dagger in Debian WSL2 | 3.12.12 | 600 passed | 155.81 seconds |
| Linux container through Dagger in Debian WSL2 | 3.14.2 | 600 passed | 94.75 seconds |
| Native Windows | 3.12.10 | 600 passed | 108.59 seconds |
| Native Windows, isolated environment | 3.14.9 | 600 passed | 272.21 seconds |

All four executions exited 0. Windows runs emitted one non-blocking Jupyter/ZeroMQ event-loop warning each; notebook execution passed.

## Additional checks

- Dagger lint and import-boundary validation: passed, exit 0.
- Dagger strict MkDocs build and export: passed, exit 0.
- Dagger wheel and source-distribution build/export: passed, exit 0.
- Native Windows Ruff and import-boundary checks: passed on both Python versions.
- No application-code or dependency-lock changes were required.

## Reproduction

Debian WSL has Dagger 0.21.8 and Docker 29.9.2. The module declares engine version 0.21.4; the installed 0.21.8 engine ran it successfully. Local engine validation is not a claim that GitHub's pinned CLI has run.

A git archive of the tested commit was extracted to /tmp/recall-guard-ci-a6a55e5. This excludes local virtual environments and credentials and keeps Dagger execution independent of Windows filesystem behavior.

From that directory:

```text
dagger --progress=plain -m ci call test-matrix --source=.
dagger --progress=plain -m ci call lint --source=.
dagger --progress=plain -m ci call build --source=. export --path=/tmp/recall-guard-ci-artifacts/dist
dagger --progress=plain -m ci call docs --source=. export --path=/tmp/recall-guard-ci-artifacts/site
```

Raw Dagger logs are in Debian under /tmp/recall-guard-dagger-tests.log, /tmp/recall-guard-dagger-lint.log, /tmp/recall-guard-dagger-build.log and /tmp/recall-guard-dagger-docs.log.

Native Windows used the workflow's locked install, full pytest suite, Ruff and architecture checker. Python 3.14 used UV_PROJECT_ENVIRONMENT pointing to LocalAppData/Temp/recall-guard-ci-py314 and UV_PYTHON=3.14; the existing Python 3.12 environment was retained. The test command added --tb=short for concise failure output.

## Scope

These are local CI-equivalent executions, not GitHub-hosted workflow runs. No commits were pushed, no docs were deployed and no release was published. Native Windows coverage cannot be inferred from Linux containers; it was run separately as shown above.

This report supersedes the earlier statement that Windows Python 3.14 was configured but not locally verified. No known Windows blocker remains in the tested package/test environments.

