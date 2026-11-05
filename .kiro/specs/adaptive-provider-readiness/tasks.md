# Tasks: deliver the configured-model flow

Revision 2 proposal — 2026-11-01. This is a scope revision, not an instruction to
resume implementation now. Completed IDs retain their original evidence; their
full original descriptions and approvals are in archive/pre-reassessment-2026-11-01.
No historical test receipt approves the uncommitted task3.2 repair.

## Preserved completed work

- [x] 1. Establish shared contracts and physical accounting
- [x] 1.1 Define readiness policies, observation states and response evidence
- [x] 1.2 Integrate optional configuration and versioned execution-plan contracts
- [x] 1.3 Expose one physical exchange per admitted attempt
- [x] 2. Observe and handle variable provider behavior
- [x] 2.1 Add bounded waiting, cooldown and cancellation decisions
- [x] 2.2 Assemble streamed responses without losing evidence
- [x] 2.3 Integrate configured provider observation and explicit pending handling

## Remaining delivery

- [ ] 3. Complete and simplify selected execution
- [x] 3.1 Prepare requested and selected workflow plans
- [ ] 3.2 Finish truthful execution failure and stop handling
  - Review/consolidate unfinished execution-local adapters before adding more abstraction; retain one request ledger and existing rewrite/ensemble calculations.
  - Fix the reproduced failed-response sanitization paths and publish the originating fatal reason atomically before waking queued work; keep caller cancellation distinct.
  - Done when focused reproductions pass for ordinary/native/callback clients and concurrent queued work, with exact physical counts and unchanged successful-answer semantics.
  - _Boundary: RequestControl and existing workflow execution integration_
  - _Requirements: 1.3, 4.1, 4.2, 4.3, 4.4, 6.2, 6.7, 7.1_
- [ ] 3.3 Prune unused polling and agent-only execution scaffolding
  - Remove the fixture-only active-poll policy/loop/config path with its stale callers/tests; preserve pending request identity, deferral and no duplicate POST. Retain optional streaming integrity and bounded waiting.
  - Inventory exact agent-owned controllers/staging helpers and their callers. Preserve useful receipts and study outputs before deleting only confirmed disposable helpers; do not delete runs recursively.
  - Done when removed symbols have no surviving imports/callers and focused pending/defer/default/stream tests pass. Existing study tools and artifacts still work.
  - _Boundary: Explicit infrastructure-pruning integration described in infrastructure-assessment.md_
  - _Requirements: 1.3, 3.3, 3.4, 7.1, 7.4_

- [ ] 4. Expose the actual product flow
- [ ] 4.1 Integrate preparation, existing execution and compact reporting
  - Extend the actual offline application/notebook example rather than creating a second workflow engine. Show configured endpoint/model/settings and explicit feature/fallback policy.
  - Exercise declared and observed missing tokens, authorized omissions, newly active roles, matching-fit enforcement and fidelity rejection. Preserve unrelated auth/service/config failures.
  - Emit one compact run record and actual stage outcomes with requested/selected/observed method, omission reason, models, usage and unknown cost. Reuse existing codecs; keep historical version-1 replay unchanged.
  - Done when deterministic end-to-end examples show MCS unavailable in completion-only mode, skipped/rejected stages accurately, and every probe/retry/generation counted once.
  - _Boundary: Preparation → RecallWorkflow → example/report integration_
  - _Depends: 3.2, 3.3_
  - _Requirements: 1.1, 1.4, 2.1, 2.2, 2.3, 2.4, 4.2, 4.3, 4.4, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 7.1, 7.5_
- [ ] 4.2 Make the adopter path understandable and runnable
  - Use the requested COBESY skill for README and engineering MkDocs; explain model/capability choices and link dated results. Cite Jane Street and distinguish exact token-selection decoding from the completion-only alternative.
  - Include a runnable Python/existing-kernel example, array axes and table row grain where applicable, and explicit no-key offline instructions. Link the Docker guide rather than duplicating it.
  - Done when examples execute offline and documentation shows the same actual-action report as task4.1, without claims of unmeasured efficacy or available MCS.
  - _Boundary: Adopter example and documentation_
  - _Requirements: 5.1, 6.6, 7.1, 7.5_

- [ ] 5. Validate once, then measure within the existing allowance
- [ ] 5.1 Accept the coherent offline flow in the container environment
  - Run focused behavioral/static checks during changes, then one full offline regression and the actual application/existing-kernel smoke at this boundary. Preserve an exact source/command/result receipt.
  - Cover configured settings/defaults, temporary versus permanent failures, retry delays/backoff/defer/cancel/partial streams, ambiguous replay permission, fidelity failure and exact admission counts. No real provider credentials/calls.
  - Use existing Dagger immediately if the Docker spec is not yet implemented. Shared Docker acceptance is owned by docker-execution; readiness correctness is not blocked on a tooling migration. Preserve release compatibility CI without manually repeating it per helper.
  - Done when the selected flow and regression pass in one identified environment; report unsupported/unverified host routes honestly rather than diagnosing an OS defect from a process exit.
  - _Boundary: Coherent workflow validation_
  - _Depends: 4.1, 4.2_
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 4.1, 4.2, 4.3, 4.4, 5.1, 5.2, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 7.2, 7.3_
- [ ] 5.2 Execute and publish the separately approved bounded operational pilot
  - Freeze dated protocol/order/config/metrics before inference: four balanced fixed/adaptive JSON probes on openai/gpt-oss-20b, unchanged settings, at most one allowed retry each.
  - Enforce at most 8 additional physical inference requests, zero status polls and 1200 seconds controlled waiting, with no paid auxiliary models or automatic calibration. Keep October31 budgets/artifacts immutable.
  - Reuse existing clients/accounting. Only if existing tooling cannot express this fixed protocol safely, add one small driver, not a generic runner/resume framework. Refuse output overwrite or ambiguous restart; retain in-flight intent and consumed allowance.
  - Done when a dated receipt/report includes every success/failure/censored outcome, request/wait totals and unknown usage/cost. Distinguish availability results from mitigation effectiveness; partial/interrupted runs remain inconclusive and never replenish allowance.
  - _Boundary: Fixed pilot integration and dated results_
  - _Depends: 5.1_
  - _Requirements: 1.2, 2.2, 2.3, 4.2, 4.3, 5.3, 5.4, 6.6, 7.1, 7.5_

## Revision map and deferred work

| Original unfinished work | Revised disposition |
| --- | --- |
| 3.2 execution | Still 3.2; narrowed to concrete defects and existing workflow integration |
| 4.1 readiness persistence/resume; 4.2 broad report-engine extension | New readiness expansion deferred; existing historical engine retained. Current 4.1/4.2 deliver the user-facing flow and docs instead. |
| 5.1/5.2 generic readiness-study runner/export expansion | Deferred. Only a bounded fixed-protocol driver, if necessary, is allowed in current 5.2. |
| 6.1 repeated platform matrix | Replaced by coherent integration acceptance in current 5.1; existing release matrix policy is separate. |
| 6.2 approved pilot | Current 5.2; original live bounds unchanged, never reset by renumbering. |

Docker targets, Compose, context exclusions and Dagger recipe sharing are owned
by [docker-execution/tasks.md](../docker-execution/tasks.md). Its deferred
readiness smoke explicitly depends on current task4.1; its core image does not.
Further mitigation-effect trials require a separate frozen protocol and budget.
