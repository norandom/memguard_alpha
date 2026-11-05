# Design: one portable configured-model flow

Revision 2 draft — 2026-11-01. Supersedes the archived pre-reassessment design.
The narrow direction is user-approved; this document is the concrete revised
implementation proposal. No application source is changed by this revision.

## 1. Delivery path

Caller ModelConfig + explicit feature/fallback choices → observe required
capabilities → resolve permitted settings → existing RecallWorkflow → compact
requested/selected/observed report. Reuse existing rewrite/fidelity/ensemble
implementations. Completion-only execution has scorer=None and MCS unavailable.

The first demonstration uses JSON completions. Native Anthropic/callback and
ordinary OpenAI clients can supply response-bearing probes with readiness=None.
Optional bounded waiting/SSE support stays behind explicit choices; neither is
required to use frontier-model fallback. No model/account/parameter guessing.

## 2. Boundary commitments

- Core owns one physical request path, shared budget, transport policy, typed response evidence and sanitization.
- Preparation resolves only the caller's authorized feature/parameter choice and validates selected roles/fit.
- RecallWorkflow remains the execution coordinator. Keep at most one small execution-local stop/evidence context where required by concurrent ensemble admission; do not build another scheduler or reuse probe orchestration as a generic generation framework.
- A thin application/example adapter presents the selected flow and compact report. Existing paired-study engine remains intact but is not a prerequisite.
- Docker/Dagger own reproducible tooling. They add no runtime container/OS dependency to the Python package.

### Out of boundary

New active polling/profile registries; guessed NIM status endpoints; new generic
study/resume/replay engine; extra scoring/ensemble mathematics; automatic model
switches or calibration; OS-native application work; debugging a host process
exit as a prerequisite for every API feature change.

### Allowed dependencies

Existing Python runtime packages, Requests, standard-library synchronization and
serialization; existing workflow and contracts. No new runtime dependency is
required. Docker/Compose/Dagger remain optional tooling outside library imports.

### Revalidation triggers

Actual endpoint/model/parameter/feature or fit changes require appropriate
configured checks and new frozen measurement input. Transport policy changes
are recorded. Container base/lock changes require rebuilding the image and the
relevant integration check. A failed test is investigated at its concrete cause;
an unexplained process exit is not evidence of an OS-specific library defect.

## 3. Keep the domain contracts, reduce their reach

Reuse WorkflowRequest, CompletionFallbackPolicy and WorkflowPlan as the selected
configuration snapshot; no new parallel execution-plan format. Keep existing
CallBudget and RequestControl as the sole physical ledger/admission mechanism.
Preserve existing PreparationArtifact/Binding compatibility where already used,
but do not expand them into a new experiment journal/resume system now.

Inputs remain explicit: role configurations, FeatureChoices, optional
EnsembleSpec, runtime RewritePolicy, matching fitted scorer if MIA is selected,
and fallback omission authority. Runtime objects never enter exported data.
Factory-produced settings compare canonical JSON, including nested bool/number,
array order and absent/null distinctions. A factory performs no provider work;
violations fail with truthful consumed-count/event evidence, not a guessed hash.

## 4. Minimal preparation and execution

1. Reject local contradictions and unrelated unsupported settings.
2. Resolve declared missing token evidence before constructing a strict MIA client. Otherwise perform one actual configured probe using existing admission; retain its response/error evidence.
3. Select fallback only from typed missing/explicitly rejected token controls and caller policy. Keep all unrelated failures as failures.
4. Omit only authorized token controls; construct fresh selected clients and validate changed/new active roles on the same owned budget.
5. Require matching fit if MIA remains selected. Construct the existing workflow with the selected choices; never fit or start evaluation implicitly.
6. Execute that method through existing single/ensemble and rewrite/fidelity paths. Do not switch methods mid-run. A fidelity rejection stops target evaluation while preserving real preflight counts.
7. Return the existing result plus a small immutable provenance/report view. Retain partial failures without claiming available MCS or completed mitigation.

The uncommitted task3.2 implementation must be reviewed for consolidation before
completion. Fix its reproduced failure sanitizer/root-cause race only through
shared helpers: one core-owned failed-response sanitizer and one atomic stop
publication before wakeup. Preserve caller cancellation separately from internal
fatal stop. Already admitted synchronous I/O remains cooperatively bounded.
Remove redundant proxy/adapter/state layers if the same contract can be expressed
in existing workflow/request-control seams; do not delete needed concurrency
protection merely to reduce line count.

## 5. Compact reporting, not another experiment platform

Use one run record with plan identity, requested/selected features and parameter
omissions, plus actual stage rows. Stage row grain is one role/model/action
outcome; physical request rows remain one admitted exchange, including retries.
Join by existing operation/call IDs. Show skipped/rejected stages and reasons,
reported usage, unknown cost, and MCS availability separately from agreement.

The first report is available from the application/notebook example and a small
JSON export using explicit existing codecs. Historical version-1 artifacts stay
byte-compatible. Retain clear guards rejecting unsupported new metadata from
old schemas. Do not implement new version-2 paired-runner/resume/export support
as a prerequisite; that expansion remains deferred until a demonstrated need.

The separately authorized operational pilot may use a small bounded driver and
dated protocol/receipt/report using the same clients and ledger. It is not a
framework: refuse ambiguous restart or output overwrite, preserve in-flight
intent/failure/unknown usage, and do not replenish allowance. An interrupted run
is partial/inconclusive. Its four probes measure availability, not rewrite effect.

## 6. Infrastructure decisions

The authoritative inventory is [infrastructure-assessment.md](infrastructure-assessment.md).

- Remove agent-only validation-controller/staging tooling from the maintained execution path; inventory before later deleting exact disposable files, preserving receipts and study outputs.
- Prune fixture-only active-polling production machinery as a coordinated change. Keep HTTP202 identity + deferral/no duplicate POST; do not ship a fake hosted capability.
- Keep optional existing SSE integrity and bounded waiting; stop expanding transport profiles or requiring them for fallback.
- Consolidate duplicate probe/generation failure handling and excess execution adapters. Keep only mechanisms needed by the observable user flow.
- Keep the existing ExperimentRunner/Journal/Report, study preparation scripts, package CI and Dagger: concrete callers/historical artifacts use them. Defer their new readiness expansion.

## 7. Docker and validation

[../docker-execution/design.md](../docker-execution/design.md) specifies one
Dockerfile/uv.lock environment shared by Compose and existing Dagger. Dev/test
and lean consumer targets are distinct; notebook launch is optional. No database,
web service, GPU runtime, Docker socket mount or background provider monitor.
Docker is a reproducibility choice and does not prove native-host Python behavior.

For a local change run the focused tests and relevant Ruff/architecture checks
through uv. At the completed flow boundary run one full offline regression in
the canonical container and the actual example/notebook smoke. Existing release
CI may retain its compatibility matrix; do not manually replay that matrix after
every subtask. A fresh reviewer can use focused tests and the unchanged full
integration receipt; rerun the full suite only for changed source or a concrete
unresolved integration concern. Native host diagnostics are bounded and separate.

Before the pilot freeze date/config/order/caps/metrics, complete offline/container
acceptance and then run only within the unchanged allowance. Results stay dated
and preserve every failed/censored trial. No provider calls for dry-run/replay.

## 8. File responsibilities

| Path | Current delivery responsibility |
| --- | --- |
| core/request_control.py | Sole admission/budget/retry ledger and sanitizer |
| core/completion.py; providers.py; provider_anthropic.py; provider_callback.py | Explicit configured API contracts/adapters |
| core/provider_probe.py | Response-bearing probes; consolidate duplicate generation/failure handling |
| core/readiness_types.py; readiness.py; readiness_evidence.py | Optional bounded waiting/evidence; prune active polling policy/loop coherently |
| core/provider_stream.py; provider_observation.py | Optional bounded SSE/HTTP integrity; no further profile platform |
| harness/workflow_plan.py; workflow_preparation.py | Explicit selected configuration and preparation |
| harness/workflow.py; rewriting.py; workflow_types.py; request_failures.py | Existing execution/calculation paths and truthful output |
| harness/workflow_execution.py | Review/consolidate unfinished execution-local context; no independent engine |
| examples/completion_workflow.py; docs/completion-api.md; README.md | Actual adopter flow, compact report and accurate scope |
| scripts/measure_provider_readiness.py | Small separately invoked bounded pilot driver, only if existing tools cannot express its fixed protocol safely |
| Existing intervention*.py / scripts/run_intervention.py | Preserve historical studies; new readiness runner/resume integration deferred |
| Dockerfile; .dockerignore; compose.yaml; ci/src/ci/main.py | Shared tooling recipe, specified separately |

Paths above are under recall_guard/ unless another root is shown. Do not create
all-new wrappers merely to fill this table; remove a path if consolidation makes
it unnecessary and update references/tests in the same change.

## 9. Traceability

| Requirements | Implementation/verification |
| --- | --- |
| 1.1, 1.2, 1.3, 1.4 | Configured clients, optional policy and focused default/settings tests |
| 2.1, 2.2, 2.3, 2.4 | Existing typed evidence and compact actual-state reporting |
| 3.1, 3.2, 3.3, 3.4, 3.5, 3.6 | Existing bounded transport, pending deferral and explicit unsupported/failure cases; active polling deferred |
| 4.1, 4.2, 4.3, 4.4 | Shared request control, atomic stop reason, sanitization and partial-response regressions |
| 5.1, 5.2, 5.3, 5.4 | Docker/example smoke, focused tests and separately frozen bounded pilot |
| 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7 | Preparation → existing execution → compact report integration |
| 7.1, 7.2, 7.3, 7.4, 7.5 | Infrastructure pruning, shared container recipe and proportional acceptance |
