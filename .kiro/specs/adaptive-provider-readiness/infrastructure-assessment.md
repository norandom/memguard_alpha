# Infrastructure assessment — 2026-11-01

Recall Guard needs configured API calls, existing calculations and an understandable
report. It does not need an OS-specific application architecture. This inventory
is a proposed cleanup plan; no source or historical artifacts were deleted.

## Evidence and decisions

| Area | Decision | Evidence and next action |
| --- | --- | --- |
| `core/request_control.py` | Keep | Existing providers use CallBudget/RequestControl for physical attempts and admission. Keep one ledger and shared failure sanitization. |
| `harness/workflow_plan.py`, `workflow_preparation.py` | Keep; integrate | Explicit feature selection and omissions are product behavior. At audit time prepare_workflow has test callers but no adopter script caller; run_intervention constructs RecallWorkflow directly. Connect preparation to an actual application/example before expanding infrastructure. |
| `core/provider_probe.py`, `readiness_evidence.py` | Keep; consolidate duplication | Response-bearing probes and typed capability rejection distinguish missing token evidence from service/auth failure. Reuse generation and sanitization seams rather than another adapter stack. |
| Uncommitted `harness/workflow_execution.py` and workflow changes | Finish narrowly; consolidate | Existing rewrite/ensemble execution needs execution-local stop and accounting. Two reproduced defects remain under partial, unvalidated remediation: failed-response secret retention and fatal-cause publication racing cancellation. Keep necessary synchronization; do not introduce another scheduler. |
| `core/readiness.py`, `readiness_types.py` | Keep optional; stop expanding | Activated through explicit readiness configuration. Completion-only fallback must not require adaptive waiting. |
| `core/provider_stream.py`, `provider_observation.py` | Keep optional integrity | Existing providers import these modules; streaming requires explicit selection. Deleting files alone would break dependencies. Further stream/profile infrastructure is outside this delivery. |
| Active pending-status polling | Remove from current production scope; plan coordinated pruning | Only a literal-loopback fixture provider/profile currently permits a status route. No hosted NIM route is verified; the approved pilot permits zero polls. Remove the fixture-only production policy/loop/config/test path together. Preserve HTTP 202 request identity, deferral, and protection against duplicate POSTs. |
| Existing ExperimentRunner/Journal/Report and intervention contracts | Keep existing; defer new readiness expansion | CLI callers and historical version-1 artifacts depend on them. A new version-2 paired-runner/resume/report engine is not needed to demonstrate fallback. Preserve compatibility guards and old replay behavior. |
| `scripts/prepare_completion_inputs.py`, `prepare_completion_pilot.py`, `prepare_completion_latency.py` | Keep | Study-specific tools, not runtime imports or disposable agent scripts. They preserve frozen measurement inputs and budgets. |
| Ignored validation controllers and staging helpers under `runs/` | Disposable scaffolding; inventory before removal | No maintained library/script/CI/test callers identified. Preserve useful receipts; remove only exact confirmed agent-owned helper files in a future cleanup task. Never recursively remove runs: it also contains study results. |
| Dagger, CI, release/docs workflows | Keep; share Docker recipe | Existing container pipeline performs tests, lint, packaging and docs. It is tooling, not a library dependency. Replace duplicate environment construction with the shared Dockerfile after its acceptance; retain Dagger orchestration and existing release compatibility checks. |

## Implementation order

1. Stop creating controllers and repeated whole-matrix validation loops.
2. Finish the two concrete execution defects with focused regressions; consolidate
   duplicate failure handling without weakening cancellation or accounting.
3. Deliver preparation → existing workflow → compact actual-action report.
4. Remove fixture-only polling as a coordinated source/config/test change. Inventory
   disposable helpers, retain evidence, then remove exact owned files only.
5. Use the shared Docker environment for one coherent integration acceptance.
   Preserve existing release CI policy; do not replay its matrix per helper.
6. Run the separately bounded pilot only after acceptance, retaining failed and
   censored trials. It measures availability, not mitigation effectiveness.

## Limits of this assessment

Unexplained native process exits do not establish a Windows, uv or package defect.
Docker improves reproducibility; it does not establish their cause or guarantee
provider responsiveness. Runtime Python packages remain real dependencies;
there is no demonstrated requirement for OS-specific business logic.

Original spec and approval history: [archive](archive/pre-reassessment-2026-11-01/).
Current implementation evidence remains in implementation-evidence.json and the
referenced runs receipts; the uncommitted execution repair is not accepted yet.
