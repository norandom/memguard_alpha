# Delivery order: Docker and the configured-model workflow

Specification follow-up — 2026-11-01. This makes existing task dependencies
explicit; it adds no new implementation phase or approval gate.

| Order | Deliverable | Existing tasks | Exit condition |
| --- | --- | --- | --- |
| 1 | Verify shared Dockerfile consumption with current Dagger; build dev/runtime images and local commands | docker-execution 1.1–2.1 | Locked environments, exclusions, output mount and optional notebook work |
| 2 | Move Dagger environment construction to that recipe and accept the useful container environment | docker-execution 2.2, 3.1 | Existing CI/artifact contracts and offline example work; README/MkDocs commands are verified |
| 3 | Finish the concrete execution defects and coordinated pruning | adaptive-provider-readiness 3.2, 3.3 | Focused failure/cancellation/accounting tests pass; no stale polling callers |
| 4 | Expose preparation → selected existing workflow → compact report | adaptive-provider-readiness 4.1, 4.2 | Actual application/existing-kernel example shows requested/selected/observed behavior |
| 5 | Accept selected workflow in the shared environment | adaptive-provider-readiness 5.1; docker-execution 3.2 | One source-matched offline regression and selected-flow/notebook smoke receipt |
| 6 | Run the previously bounded operational pilot | adaptive-provider-readiness 5.2 | Dated complete or explicitly inconclusive availability report within frozen allowance |

This is the recommended order for the user's Docker-first direction. Steps3–4
may use the existing Dagger environment before Docker migration if that is more
practical; they do not depend on a new container business-logic path. Docker core
acceptance does not wait for readiness4.1, avoiding a cross-spec cycle. At step5,
reuse an unchanged source/image-matched integration receipt where applicable;
do not rerun the same whole suite separately for two spec checkboxes.

## What the report must distinguish

Use the existing plan, result and physical-request evidence. The compact report
has one run-level requested/selected configuration and actual role/model/action
outcomes; request-level rows represent one admitted physical exchange, including
probes/retries. Name columns according to these meanings and include units for
elapsed/wait times. Do not silently align arrays by position across unrelated
draws; use existing identities and document the row grain when building tables.

Show capability observation and fallback reason separately from successful
execution. An authentication/quota/service failure does not establish missing
logprobs. Missing token evidence permits only caller-authorized omission and
completion-only selection; MCS remains unavailable. Agreement is not MCS.
Record skipped/fidelity-rejected stages and all consumed calls truthfully.

## Measurement scope

The approved NIM pilot remains four balanced fixed/adaptive JSON probes, at most
8 additional physical inference requests, zero polls, 1200 seconds controlled
waiting, unchanged model/settings and no paid auxiliary calls. It compares
availability/wait behavior. It does not establish mitigation effect or update
calibration. Preserve failed/censored observations and existing allowances.

A future mitigation-effect comparison must freeze its own prompts/cutoffs,
provider/model/settings, baseline/intervention arms, metrics, physical-call
budget and dated result locations before calls. Cite the
[Jane Street article](https://blog.janestreet.com/mitigating-memorization-in-llms/)
while distinguishing token-selection decoding from the completion-only
alternative. No new measurement allowance is inferred from this delivery plan.
