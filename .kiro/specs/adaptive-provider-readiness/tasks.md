# Implementation plan: adaptive provider readiness

Status: generated; independent task-graph review PASS on 2026-11-01. Requirements and design were
approved on 2026-11-01. The user approved implementation and the bounded pilot in task 6.2 on
2026-11-01; that authorization is recorded in spec.json. Execute one subtask at a time with independent review.

All tasks must use idiomatic Python and names matching domain meaning,
representation and units. Describe array axes, event/request/item row grain and
alignment where relevant. Preserve native Windows, existing Jupyter kernels,
lightweight imports and old version-1 artifact replay. Sentrux is not required.
Every behavioral task includes meaningful RED/GREEN evidence, failure paths
and the existing Kiro review/verification gates. Do not repeat whole-platform
validation without a source change or specific concern.

- [x] 1. Establish shared contracts and physical accounting
- [x] 1.1 Define readiness policies, observation states and response evidence
  - Define validated immutable policy, observation, pending-receipt, complete/partial probe evidence and runtime cancellation contracts, including documented units and limits.
  - Separate observed provider progress from elapsed local waiting, answer content from reasoning progress, and complete responses from retained partial data.
  - Done when invalid/non-finite policies fail locally, unavailable capabilities remain explicit and serialization excludes runtime controls/credentials.
  - _Boundary: Core configuration and transport contracts_
  - _Requirements: 1.1, 1.2, 1.4, 2.1, 2.2, 2.3, 2.4, 4.1, 4.3, 4.4_

- [x] 1.2 Integrate optional configuration and versioned execution-plan contracts
  - Add optional readiness configuration and trace fields, requested/resolved plan and fallback-policy types, and version-2 persistence contracts with explicit typed loaders and metadata whitelists.
  - Keep absent-option legacy behavior and exact version-1 serialization, identities and saved metrics. Reject readiness/plan fields in version-1 protocols instead of silently upgrading them.
  - Define phase/execution namespaces and preparation-artifact bindings before runtime producers consume them.
  - Done when old artifacts round-trip unchanged and new valid plans/partial evidence round-trip without serializing clients, callbacks, credentials or scorer internals.
  - _Boundary: Explicit integration of configuration and persisted contracts_
  - _Depends: 1.1_
  - _Requirements: 1.1, 1.3, 4.2, 4.3, 4.4, 6.2, 6.3, 6.6_

- [x] 1.3 Expose one physical exchange per admitted attempt
  - Reuse existing admission, pacing and sanitized failure handling through a single-attempt seam; represent pending exchange results without claiming logical answer completion.
  - Preserve typed request IDs, request purposes and retry-delay evidence, and retain complete responses rejected only for missing token evidence.
  - Keep legacy retry behavior when readiness is disabled and count every new physical exchange exactly once.
  - Done when concurrent admission cannot exceed the shared cap, permanent failures cannot retry, and no polling/retry is hidden inside an attempt.
  - _Boundary: Core physical request control_
  - _Depends: 1.1, 1.2_
  - _Requirements: 1.3, 2.1, 2.2, 3.5, 3.6, 4.2, 4.3, 4.4_

- [x] 2. Observe and handle variable provider behavior
- [x] 2.1 Add bounded waiting, cooldown and cancellation decisions
  - Interpret valid server delays, use configured bounded fallback backoff, and share a model/client cooldown across concurrent callers without holding a lock over network operations.
  - Keep ambiguous outcomes visible and require explicit replay permission; stop future admission on cancellation, deferral or limits.
  - Expose immutable transition notifications and an elapsed-time snapshot, with cooperative synchronous-I/O limits documented honestly.
  - Done when fake-clock/concurrency scenarios prove no early retry, no shortened server delay, no duplicate accepted generation and no further admission after a stopping condition.
  - _Boundary: Core readiness state and waiting policy_
  - _Depends: 1.3_
  - _Requirements: 1.2, 2.1, 2.2, 2.4, 3.1, 3.2, 3.5, 3.6, 4.1, 4.2, 4.3, 5.2_

- [x] 2.2 Assemble streamed responses without losing evidence
  - Decode incremental UTF-8 and SSE boundaries, preserve choice-zero answer/token alignment, and track reasoning separately from answer content.
  - Require terminal framing/finish evidence, enforce event/response size limits and retain bounded partial output on interruption or malformed content.
  - Treat absent usage or token probabilities as unavailable; do not infer stream-scoring support from a model name.
  - Done when split-byte, truncated, malformed, oversized and missing-token scenarios preserve correct answer usability and diagnostics.
  - _Boundary: Core streamed response assembly_
  - _Depends: 1.1_
  - _Requirements: 2.1, 3.4, 4.3, 4.4, 5.2_

- [x] 2.3 Integrate configured provider observation and explicit pending handling
  - Connect the readiness loop and stream assembler to the existing configured OpenAI-chat client and evidence-bearing probe contract.
  - Preserve verified pending request headers. Enable status retrieval only with an explicitly supported trusted profile; count each GET and defer the initial gpt-oss profile while its route remains unverified.
  - Reject unsupported observation profiles before calls, preserve redirects/credential controls, and keep native Anthropic/callback default behavior unchanged.
  - Done when local HTTP fixtures prove buffered/streaming generation, server-hint waiting, one-POST budgeted polling under a fixture profile, unsupported-route deferral and cleanup on both supported operating systems.
  - _Boundary: Explicit integration of provider transport and readiness_
  - _Depends: 2.1, 2.2_
  - _Requirements: 1.1, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4, 3.1, 3.3, 3.4, 3.5, 3.6, 4.1, 4.2, 4.4, 5.1, 5.2_

- [ ] 3. Resolve explicit completion-only fallback
- [ ] 3.1 Prepare requested and selected workflow plans
  - Resolve declared/observed missing token evidence before rejecting client construction or selecting a scorer. Apply fallback only when the caller supplied it.
  - Omit only explicitly authorized logprob controls; retain model/endpoint and all other settings. Reject unrelated configuration/service errors, explicit contradictions and missing calibration as fallback triggers.
  - Construct fresh clients on the same preparation budget and validate changed target settings plus newly active rewrite/verifier roles.
  - Done when readiness or failure produces an immutable inspectable plan with actual probe evidence, exact parameter omissions and no hidden generation work in factories.
  - _Boundary: Harness preparation and capability resolution_
  - _Depends: 1.2, 2.3_
  - _Requirements: 1.1, 2.3, 4.2, 6.1, 6.2, 6.3, 6.4, 6.5_

- [ ] 3.2 Execute selected plans through the existing workflow
  - Reuse existing rewriting, fidelity checks and ensemble reduction for the selected plan; text-only fallback has no scorer and retains explicit MCS unavailability.
  - Attach plan identity and readiness trace to actual results and failures. Stop queued required work on cancellation/deferral or capability drift, without changing an evaluation's method mid-run.
  - Preserve existing defaults, disabled auxiliary-call counts and faithful failure behavior for rejected rewrites.
  - Done when complete application paths exercise single/ensemble and rewrite off/on, fallback decisions are observable, and a rejected rewrite admits no target generation.
  - _Boundary: Explicit integration of preparation with workflow execution_
  - _Depends: 3.1_
  - _Requirements: 1.3, 4.1, 4.4, 5.1, 6.1, 6.2, 6.5, 6.6, 6.7_

- [ ] 4. Preserve study accounting and report actual execution
- [ ] 4.1 Bind preparation to frozen evaluation and resume
  - Bind the selected plan and preparation ledger/hash to the version-2 protocol, admitting evaluation only from the remaining combined allowance.
  - Preserve the paired runner's four conditions/all three roles/draw-zero semantics; reject incompatible plan shapes before preparation or network work.
  - Explicitly count fresh evaluation/resume probes in dry-run bounds; historical preparation evidence must not populate a fresh client's probe cache.
  - Preserve write-ahead intent, execution namespaces and refusal of ambiguous automatic restart.
  - Done when preparation, evaluation and resume reconcile without replenishing budgets, missing calls, duplicate generation or changed paired methods.
  - _Boundary: Explicit integration of preparation with experiment persistence and CLI_
  - _Depends: 1.2, 3.2_
  - _Requirements: 3.5, 4.1, 4.2, 4.3, 5.3, 6.3, 6.5, 6.6_

- [ ] 4.2 Report requested, selected and observed actions
  - Add a stage/model/role view derived from saved physical requests, rewrites and draws, including fallback reasons, omitted controls, skipped/rejected stages, available usage and unknown costs.
  - Keep selected parameters distinct from provider-confirmed values and answer agreement distinct from MCS. Group outcomes by selected plan before paired analysis.
  - Extend portable export and offline replay through explicit whitelists, preserving exact old report identities and metrics.
  - Done when report values reconcile with saved ledgers, moved bundles replay offline and neither unexecuted stages nor failed rewrites are presented as successful mitigation.
  - _Boundary: Experiment reporting and offline replay_
  - _Depends: 4.1_
  - _Requirements: 2.1, 4.2, 4.3, 4.4, 5.4, 6.2, 6.3, 6.6, 6.7_

- [ ] 5. Expose reproducible operation and adopter examples
- [ ] 5.1 Add explicit readiness study preparation and replay
  - Provide dry-run, separately invoked live execution and offline replay with frozen model/configuration, observation policies, sample ordering, request/wait caps, fallback choices and metric definitions.
  - Preserve all deferred, unknown, failed and censored observations; separate transport-policy comparison from JSON/SSE configuration comparison and from mitigation effectiveness.
  - Refuse ambiguous restarts and exhausted budgets; perform no credential resolution or provider work during dry-run/replay.
  - Done when deterministic operational scenarios reproduce the same saved metrics and all physical attempts can be reconciled with the plan.
  - _Boundary: Operational readiness study command_
  - _Depends: 4.2_
  - _Requirements: 3.5, 4.1, 4.2, 4.3, 5.2, 5.3, 5.4, 6.6_

- [ ] 5.2 Demonstrate application and notebook choices with actual reports
  - Add offline examples for observed waiting, explicit fallback, unavailable MCS and requested/selected/observed reporting, including an unsupported pending route and fidelity rejection.
  - Use COBESY for adopter/engineering instructions, preserve the Jane Street distinction and label unreleased checkout APIs accurately.
  - Execute the actual documented choices in a current-interpreter notebook with owned-kernel cleanup and no event-loop policy changes.
  - Done when all examples and strict documentation build run and their reports identify only stages that actually executed.
  - _Boundary: Examples and engineering adoption_
  - _Depends: 3.2, 4.2, 5.1_
  - _Requirements: 1.4, 2.4, 5.1, 5.2, 6.1, 6.2, 6.6, 6.7_

- [ ] 6. Validate portability and measure bounded live behavior
- [ ] 6.1 Verify regression, platform and distribution contracts
  - Run focused and full offline tests, lint/architecture gates, Windows/Linux Python 3.12/3.14, actual notebook paths and an external lean installed-wheel smoke.
  - Verify version-1 saved artifact bytes/replay, version-2 trace portability and Unicode paths, cancellation/stream cleanup, exact physical accounting and public naming/units.
  - Stage Dagger input outside any ancestor Git checkout and inspect tracked-only archive identity before upload.
  - Done when fresh required platform evidence passes and live coverage remains separately described.
  - _Boundary: Final integration and platform verification_
  - _Depends: 5.2_
  - _Requirements: 1.3, 4.1, 4.2, 4.3, 4.4, 5.1, 5.2, 6.6_

- [ ] 6.2 Run and report a separately authorized operational pilot
  - Prepare a new dated protocol for the existing NIM model; preserve all October 31 study artifacts and exhausted phase budgets.
  - Initial ceiling explicitly approved on 2026-11-01: at most eight additional physical inference requests, zero status polls while the route is unverified, and 20 minutes of controlled total waiting. Compare four scheduled JSON-mode probes in balanced fixed/adaptive order, with at most one permitted retry each; generation parameters stay identical. Do not switch models or add paid auxiliary-provider calls.
  - Record cancellation, early permanent rejection, excessive provider delay and budget stops as partial/inconclusive results. Additional SSE, calibration or rewrite-effect trials require their own declared configuration and allowance.
  - Done when an authorized bounded attempt and its request/latency/availability report reconcile, including any uncompleted comparisons. A missing authorization or unavailable runtime leaves this task incomplete; offline tests never substitute for a live result.
  - _Boundary: Live operational measurement and dated evidence_
  - _Depends: 5.1, 6.1_
  - _Requirements: 2.2, 3.1, 3.5, 4.1, 4.2, 4.3, 5.3, 5.4, 6.6_

## Implementation notes

- The user explicitly approved the task 6.2 pilot limits on 2026-11-01. This separate allowance does not replenish the October 31 study budgets; offline/platform prerequisites still apply.
- The initial NIM profile has verified pending-ID headers and SSE framing, but an unverified exact polling route and streamed token-evidence support. Keep those limitations observable.
- Returning full text without logprobs can support an explicitly selected completion-only plan. A service timeout, missing fitted calibration or rejected rewrite is not that trigger.
- Fresh evaluation/resume probes are intentional and consume the remaining allowance. Never import preparation observations into an identity-scoped client cache.
- Old Windows interruption causes remain unknown. The proven Dagger staging issue came from an archive under an ignored parent checkout, not from its CLI/module version difference.

- Task 1.1 keeps stdlib policy/state contracts in readiness_types.py and response-dependent evidence in readiness_evidence.py to avoid reciprocal configuration imports. READY must validate explicitly requested token evidence independently of MIA and compare canonical JSON settings, since Python mapping equality conflates nested boolean/number values.

- Task 1.2 persists acyclic plan contracts in harness/workflow_plan.py. Task 4.1 must replace interim version-2 runner/resume guards; task 4.2 must replace version-2-in-version-1 report-envelope guards. A Linux 3.12 canonical full run passed (2061 tests); native Windows full-run termination 0xE0000001 remains unexplained and does not waive task 6.1.

- Task 1.3 exposes RequestControl.run_attempt and typed RequestError.response_evidence (never raw .response). Task 2.1/2.3 must preserve server timing hints on pending responses as well as errors and classify detached failures without status/code conservatively; retryable alone does not prove safe replay. Legacy run intentionally retains old event serialization.

- Task 2.1 adds ReadinessControl.operation(...).run(exchange) with admission-time ExchangeContext bounds and ReadinessResult. Pending transport remains deferred for task 2.3; RetryAfterHint now lives in readiness_types.py and remains import-compatible through request_control.py. Shared cooldown is on RequestControl, including terminal failures. ReadinessOperation._complete sanitizes locally failed complete evidence; preserve that boundary when integrating providers. RFC850 dates use receipt-relative years and valid oversized delays defer without fabricated numeric timing.

- Task 2.2 adds standalone StreamAssembler/StreamProgress. Transport bytes count supplied chunks, independently of processed/event limits; local metadata must be UTF-8 encodable. Constructor stream-setting acceptance remains task 2.3. Focused Windows uv76passed and canonical Linux2235passed; the single full Windows uv exit0xE0000001 is preserved without a causal claim, and task6.1 remains required. All project commands now use uv run --no-sync per user instruction.

- Task 2.3 integrates nvidia_nim observation and explicit loopback fixture polling. Probe capability reports preserve gated exact requested-control rejections; auth/quota/service failures do not imply missing token capability. Task 3.1 must add a compatible response-bearing legacy probe adaptation for readiness=None (including native Anthropic/callback); do not force optional NIM adaptive readiness to prepare frontier fallback. Keep old probe defaults unchanged. Final full Linux2303passed; full platform/kernel/wheel gates remain6.1.
