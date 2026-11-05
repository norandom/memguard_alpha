# Current scope correction — 2026-11-01

Endpoint-based divergence evaluation supersedes the hosting-first continuation
below. Read .kiro/specs/divergence-decoding-evaluation/scope-amendment.md and
requirements.md revision 2 first. Requirements generated; revised phase approvals
pending. Do not resume the local GPT-OSS runtime callback or GPU smoke tasks under
old approvals. Preserve completed work; no new calls or downloads authorized.
# Latest checkpoint: divergence evaluation implementation

Completed and independently reviewed:1.1 contracts/planning(a6b6977),2.1 corpus
admission/controls(91c403c),1.2 optionalDockerenvironment(97f052d),2.2 bounded
trigramfitter(13e8dc4). Current 73experiments tests/Ruff pass. Earlier blocked
CUDA build has FINISHED; do not keep polling98029. OptionalimageID:
sha256:acdf5deaffe08298a0ada7d99918eaae181140029deff35433ee0ee7e3b28c47.
Network-none package/GPTOSS-class imports and leanwheel isolation approved.
Image source predates final fitter memoryrepair; mountcurrentrepo forchecks and
rebuild currentcontext before any actualfuture model experiment.

Task3.1 runtimecallback active agent divergence_eval_runtime. No model weights,
tokenizer artifacts, realstudyfit, paidcompute or model inference approved/run.
Actual modelruntime task6.2 needs concretebudget approval; localhardware observed
AMD890M integrated with29.6GiB RAM, noNVIDIAlisted. See hardware-observation.md.
Preserve unrelatedreadinessWIP. Current tasks/spec are approved implementation.
Trigrammemorycap conservativelyestimates OWNED fitting allocations; notprocess
RSS/GPU or arbitrarytokenizercallback allocations. Reviewerfound variablemanifest
fact-ID bypass, fixed with preadmissionmetadata walk +7regressions. Final73pass.

Historical checkpoints below may contain superseded pending-build information.
## Environment build checkpoint

Task1.2 source changes pending review: Dockerfile, pyproject.toml, uv.lock,
tests/test_packaging.py, experiments/divergence_evaluation/environment.py,
and spec environment-validation.md.31 focusedtests/Ruff/lockcheckpassed;
lean runtime installed-wheel isolationpassed. OptionalCUDA build stillrunning
in original agent divergence_eval_environment session98029. Sessions are agent-
owned: other agents cannot poll98029. Reactivate original agent to poll, then
cached rebuild same target (initialcontext predates environment.py), followed by
network-none GPT-OSS-class/dependency imports and imageinventory. No modelweights.
Independentdebug EXTERNAL_DEPENDENCY/no establishedcodefault, recommends preserving
build/cache and completing same checks. Task1.2 remains unchecked/uncommitted.
Do not launch duplicate build or change dependencies speculatively.
# Active divergence evaluation implementation — 2026-11-01

User approved model selection, requirements, design and tasks for
.kiro/specs/divergence-decoding-evaluation. Main target is always gpt-oss-20b.
Two tokenizer-aligned trigram auxiliaries are proposed experimental artifacts.
Task 1.1 is complete, independently APPROVED: 26 focused Docker tests and Ruff.
Commit a6b6977 contains approved spec and offline contracts/planning. Task 1.2
optional inference environment is in progress with agent divergence_eval_environment.
Follow Kiro one-task implement/review/commit cycle. No model download, paid
provisioning, smoke or study inference has been approved. Preserve existing
unrelated readiness WIP. Use Docker and uv run --no-sync; no native diagnostics.

Latest NIM facts supersede historical handoff below: three requests across two
attempts (504, accepted eligibility with token evidence, calibration probe504).
No fresh fit/evaluation. 3/146 consumed,143 unused; no automatic further request.
Public follow-up evidence is docs/results/2026-11-01-nim-refresh/followup-01/.
Future runner pacing is2s, singleworker, no retries; old plans immutable1.5s.
Dated single/ensemble summaries and divergence evaluation proposal are under docs.

## Historical handoff
# Current NIM refresh status — 2026-11-01

The user requested NIM open-weight model tests, including ensembles, in Docker.
The prepared helper is scripts/measure_nim_refresh.py. Eighteen focused Docker
tests and independent review passed. HEAD is 936aa4d; helper commit d0beae3.

The user explicitly approved 146 physical requests: eligibility 2, calibration 80,
evaluation 64, nominal 112. This is an original-input baseline with single answers
sharing draw 0 of three-draw ensembles. Rewriting and divergence decoding are
inactive. The separately approved eight-call readiness pilot remains unused.

First live attempt: one physical eligibility request returned HTTP 504 after
302.236 seconds. Credential eligibility is unverified; calibration and evaluation
were not run. Do not label the key invalid or invent ensemble/calibration values.
Public report and immutable JSON evidence:
 docs/results/2026-11-01-nim-refresh.md and its adjacent asset directory.
README/MkDocs were updated; strict Docker docs and focused Ruff passed.
Local output: runs/container/2026-11-01-nim-refresh.

An explicit user choice is pending: retry eligibility once, or stop with saved
results. Do not send another POST before that reply. The first provider outcome
is unknown and automatic ambiguous replay is disabled in the frozen plan.
The lower-cap option --eligibility-request-cap 1 binds 1/80/64 = 145 remaining calls;
with the prior consumed call, the combined ceiling stays 146. Frozen follow-up:
 data/experiments/2026-11-01-nim-refresh/retry-plan.json.

If the user approves, use a NEW output directory:
 /outputs/2026-11-01-nim-refresh-retry-01.
Use --plan-file pointing to retry-plan.json and --eligibility-request-cap 1.
Mount .env read-only at /run/secrets/nim.env; --credential-file selects only
NVIDIA_API_KEY through dotenv parsing. Never print or export credentials.
The current dev image includes the helper and frozen plan:
 f2dee8a8224f75dda2114f766dcce43723e5160e12dba5561aa82792e843e352.
The first attempt used image 94138c218ba138c61e259dedcbe2f4f618c96751a4fdee425ef5a21e48139b0c.

No measurement container remains running; no second request has been sent.
Keep work Docker-only. Preserve the prior uncommitted readiness implementation
and all old results. The history below is superseded where it conflicts.

---
# Current handoff — Docker core validated, 2026-11-01

Latest user instruction: focus on Docker; native Windows work is out of scope.
Docker implementation and validation are complete through task3.1. Do not resume
native-host diagnostics or the old task-by-task platform-matrix loop.

HEAD1025cb4 on feat/completion-api-mitigation, D:/Source/memguard_alpha.
Docker commits:1e5c920,0d82257,5340400,96362c6,1870eef,1025cb4.

Read .kiro/specs/docker-execution/implementation-validation.md and tasks.md.
Six executable core tasks are checked; task3.2 is explicitly blocked on
adaptive-provider-readiness4.1. Docker is usable without that selected-fallback
follow-up. README has Docker quick start; docs/development.md is the engineering
guide, structured with requested COBESY and Jane Street citation/limitations.

Verified: final Compose full suite2385passed279.69s; Docker Ruff(project plus
explicit CI file) and architecture passed; pinned actual Dagger0.21.4 shared
Dockerfile base/build/strictdocs passed; non-root runtime installed-wheel import
from/tmp with networknone passed, no dev packages; notebook unauthenticatedAPI403,
localhost-only publication, no provider keys, persistent output and cleanup.
Default Dagger0.21.8 engine restored with installed CLI, no tool upgrades.
Owned notebook/dev test services and network stopped; exact temporary smoke
markers removed. Local images recall-guard:dev/runtime remain for use.

Preexisting readiness implementation/spec changes remain uncommitted and
preserved. Do not claim task3.2 accepted from the full Docker suite. During
acceptance one bad assertion in its already-untracked test file was corrected:
ModelResponse.complete does not exist; the test now asserts the existing
ModelResponse type, keeping redaction/evidence/count checks. The whole affected
file passed33tests, then full container2385passed. No runtime library changes
were made as part of Docker delivery. Reassessment.md records that correction.

No live inference, eligibility checks or calibration updates in this work.
Existing pilot allowance8inference/0polls/1200s controlled waiting is unchanged,
not replenished or activated by Docker acceptance. Current work stays Docker-only.

Commands from repository root: docker compose build dev; docker compose run
--rm dev (offline fixture); docker compose run --rm dev uv run --no-sync pytest
-q; docker compose --profile notebook up notebook. Create runs/container first.
Use --detach instead of-d through PowerShell wrappers. Explicitly propagate
$LASTEXITCODE in exec scripts. Project Python commands use uv run --no-sync;
no hard-coded host interpreter. Do not repeat broad checks without source changes
or an actual unresolved failure; source-matched passing receipts are sufficient.

## Historical handoff below — superseded where it conflicts with the above

<!-- current-resume-checkpoint -->
# Current restart checkpoint — 2026-11-01

The user resumed approved implementation with “continue”. Keep working autonomously through the approved adaptive-provider-readiness tasks and bounded pilot; do not request those approvals again.

- Workspace: `D:/Source/memguard_alpha`; branch `feat/completion-api-mitigation`; current HEAD `f9dfccb`.
- Readiness tasks 1.1, 1.2, 1.3, 2.1, 2.2, 2.3 and 3.1 are committed and checked. New commits: `6aa1287` (1.2, independent Linux 2061 passed) and `b0a23b4` (1.3, independent Linux 2109 passed). Native Windows full-run interruption remains unexplained; task 6.1 platform matrix is still required.
- Task 2.1 was committed as `5792e0b` after one sanitizer remediation. Fresh independent review APPROVED; Linux 2159 passed, native core 823 passed; exact evidence in `runs/readiness-2-1-fixed-review.md`. Its previous rejection and proof remain preserved.
- Task 2.2 committed as `963eff9` after independent APPROVED and parent source/log verification. Evidence `runs/readiness-2-2-fixed-review-verdict.md`: canonical Linux2235passed/raw0; Windows uv focused76passed; full Windows uv abort0xE0000001 preserved without cause claim. Observed-byte and local Unicode metadata findings fixed. User uv invocation guidance committed in steering.
- Task2.3 committed as `e7a97d5` after one review remediation and one bounded debug/verification retry. Exact approval/evidence `runs/readiness-2-3-fixed-review.md`: Linux2303passed/raw0, all337stage/9overlay hashes verified; native focused483passed earlier. Typed reports now preserve exact requested-control rejection; auth/quota/service failures do not imply missing tokens. Native uv abort cause still unknown; final6.1 matrix remains.
- Task3.1 committed as `f9dfccb` after independent APPROVED and parent9file/hash verification. Evidence `runs/readiness-3-1-review-verdict.md`: canonical Linux2352passed/raw0; frozen task-local45pass, broad443pass before exception-detachment delta. Core response-bearing legacy probes and opt-in guarded cancellation/purpose preserve defaults. Explicit preparation uses existing immutable contracts, runtime policy/scorer/fallback_ensemble, fresh budget/clients/probes. FactoryContractError retains truthful immutable plan/ledger/counts without fake prompt hashes.
- Task3.2 is in review remediation round1 with `/root/prepared_execution_3_2`. Review `runs/readiness-3-2-review-verdict.md` REJECTED two concrete blockers despite fullLinux2380passed/raw0 and clean static/boundary/source audits: new legacy/post-response failure paths retain raw known-secret evidence; owned cancellation signals before fatal latch, allowing queued sibling to replace root configuration failure with caller_cancelled. Repros `runs/readiness-3-2-review-probes.py` and `runs/readiness-3-2-review-stop-reason.py` with receipts. Fix core-owned schema-aware failed-evidence sanitation and atomic root-reason publication before wakeup, with typed distinct caller/internal stop evidence. No4.x work.
- Pre-remediation source11overlays/342files is immutable evidence at `runs/readiness-3-2-frozen-source.json` and `/tmp/readiness-3-2-frozen-r_q4p7kq`. New final source/hash stage and fresh independent review required. Forthcoming report `runs/readiness-3-2-remediation-implementer.md`, require exact Status Report. Task3.2 remains unchecked/uncommitted. Earlier broadnative956passed/1symlinkskip and finalrelevant178passed; no new provider calls. No task3.2 debug rounds used yet.
- Seven of14 executable readiness tasks committed; remaining3.2,4.1,4.2,5.1,5.2,6.1,6.2. No provider calls in continuation; approved NIM pilot remains8inference/0polls/1200s controlled waiting after offline/platform gates. One validation owner; no parent native exec during application tests. All project commands uv run --no-sync; never direct interpreter paths. Historical0xE0000001 remains unproven; final6.1 matrix mandatory.
- Latest user correction: all project commands must use `uv run --no-sync ...` from the repo root; no direct `.venv/Scripts/python.exe` or C: interpreter calls. This is recorded in steering. Parent verified uv resolves the project environment. A focused uv Windows check after source repairs is authorized; retain raw results and do not assume invocation changes explain/fix earlier exits. If abnormal exit recurs, preserve it and use the established Linux route without repeated native attempts. Full platform matrix remains required.
- Both debug rounds were used for completed task 2.2; reports remain under `runs/readiness-2-2-debug*.md`. Its parser and review findings are resolved. Task 2.3 has its own task-local review/debug counters. No provider requests have been made.
- Kiro skill paths remain under `C:/Users/mariu/Source/QOpenBB/.agents/skills/`. Native full interruption remains unresolved; use task-local canonical Linux review plus focused Windows checks, retaining task 6.1's complete platform matrix.
- Parent verification helpers: `runs/finish_readiness_task.py` (selective commit after approval only), `runs/stage_readiness_review.py` (isolated HEAD + explicit overlays), `runs/capture_readiness_review.py` (direct durable logs/raw exits). No source checkout under `runs/`; stage Linux under `/tmp` outside ancestor Git.
- No provider requests have been made during this continuation. The approved pilot remains at most 8 inference requests, 0 polls, 1200 seconds controlled waiting, same NIM model, after offline/platform gates. Old October 31 budgets/artifacts remain immutable.

The older handoff below is historical context. Current task checkboxes, commits and the reports above supersede its stopped/active-task notes. Keep this scratch handoff untracked.
<!-- end-current-resume-checkpoint -->

# Handoff

Generated 2026-11-01 for a quick restart. The user has now resumed work with "continue".
Task 1.2 independent review is active; the stopped-state notes below describe the restart checkpoint.
Workspace: `D:\Source\memguard_alpha`; PowerShell; Europe/Berlin.
**Next action: independent review of adaptive-provider-readiness task 1.2.**
Do not restart implementation or discard the working tree. All workers stopped;
no owned tests, servers or background processes remain running.

## Goal

Support explicitly configured models/endpoints with genuine logprob scoring when
available and a caller-selected completion-only fallback using existing rewriting
and ensembles. Add bounded observation of variable provider readiness and clear
requested/selected/observed reporting, then measure honestly. Keep native
Windows/Jupyter compatibility and idiomatic, understandable code with names that
match domain meaning, representation and units. Jane Street inspires the feature;
completion rewriting is an adaptation, not the exact logit-level decoder:
https://blog.janestreet.com/mitigating-memorization-in-llms/

## Current state

- Branch `feat/completion-api-mitigation`; HEAD **5d9d5d7**.
- Requirements, design and all tasks of `.kiro/specs/adaptive-provider-readiness/`
  are approved. **Do not ask again.** New spec: **1/14 complete**.
- Task **1.1** is independently approved, verified and committed.
- Task **1.2 is READY_FOR_REVIEW**, uncommitted and unchecked. Its exact implementer
  report and commands: `runs/readiness-1-2-implementer-handoff.md`.
- Task 1.2 final scoped checks: **549 passed, 1 skipped, 46.10s**. Earlier broader
  affected checks: **863 passed, 1 skipped**. Ruff, architecture and whitespace
  pass. **No independent review or full suite has run on task 1.2.**
- Last reviewed full suite was task 1.1: **2008 passed, 1 skipped, 1 warning**,
  158.08s, exit 0. `runs/readiness-1-1-final-review-*` contains evidence.
- Original completion-api-mitigation: **22/25 complete**. Platform task 7.3 is
  accepted. Live tasks 6.2, 6.3 and full 6.4 remain incomplete.
- No provider requests were made this session. No push/release requested.

Task 1.2 implements configuration/persisted contracts only. Readiness runtime,
fallback execution and v2 report production are later tasks, not completed APIs.

## Immediate next steps

1. Read `AGENTS.md`, `.kiro/steering/`, active spec `spec.json`, `tasks.md`, relevant
   requirements/design, and `runs/readiness-1-2-implementer-handoff.md`. Inspect
   `git status --short`, `git diff` and the new untracked source/test files.
   `runs/readiness-restart-snapshot-20261101.json` records HEAD and exact hashes
   of all ten task files. Preserve this working tree.
2. Dispatch a **fresh independent Kiro reviewer for 1.2**. Requirements: **1.1,
   1.3, 4.2, 4.3, 4.4, 6.2, 6.3, 6.6**. Design: **2, 4, 5.1, 5.4, 6, 7.1, 7.2,
   8**. Boundary: configuration and persisted-contract integration. No 1.2 review
   or remediation round has occurred yet. Supply actual diff + implementer report.
3. Reviewer runs canonical full pytest once, Ruff/architecture and appropriate
   contract checks using sanitized environment and durable UTF-8 logs. Reusable
   controllers are under `runs/readiness-1-1-final-review*` and
   `runs/task-7-3-validation-controller.py`. No credentials/provider calls.
4. Require exact `## Review Verdict` and `- VERDICT: APPROVED|REJECTED`. Repair
   concrete findings if rejected. After APPROVED, parent verifies fresh evidence
   and selectively commits. Parent helper:
   `.venv/Scripts/python.exe runs/finish_readiness_task.py 1.2 'evidence' <explicit files>`.
   It marks the task and records evidence; do not include this scratch handoff.
5. Re-read tasks.md, then proceed sequentially with **1.3** and the remaining
   approved tasks. One implementer/review/verification/commit cycle per X.Y task.
   Do not skip to live measurement.

## Files in play

| File | Task 1.2 responsibility |
| --- | --- |
| `recall_guard/core/completion.py` | Optional ModelConfig policy and ModelResponse trace, codecs, exact old absent-option exports |
| `recall_guard/core/readiness_types.py` | Immutable policy/observation types from 1.1 plus explicit loaders |
| `recall_guard/core/readiness_evidence.py` | Typed complete/partial probe codecs, exact settings/token evidence |
| `recall_guard/core/request_control.py` | Optional RequestEvent purpose/operation/request IDs; no admission-loop change |
| `recall_guard/harness/workflow_plan.py` (new) | Acyclic WorkflowRequest, CompletionFallbackPolicy and WorkflowPlan |
| `recall_guard/harness/intervention_types.py` | V2 protocols/checkpoints, PreparationArtifact/Binding, phase accounting and partial-response records |
| `recall_guard/harness/intervention.py` | Interim v2 execution guard |
| `recall_guard/harness/intervention_report.py` | Interim v2-in-v1 report guards |
| `tests/harness/test_readiness_persistence.py` (new) | New contracts, malformed fields, budgets, saved artifact/replay and guards |
| `tests/harness/test_intervention_types.py` | Unsupported schema sentinel 2 -> 3 |

Specs, approvals and review reports are in `.kiro/specs/adaptive-provider-readiness/`.
Original implementation evidence is in `.kiro/specs/completion-api-mitigation/`.
Read the implementer handoff for precise task-local commands and evidence.

## Contract decisions to preserve

- Schema default remains 1; existing serialization, identities and replayed metrics
  stay exact. V1 rejects new fields, including explicit null at load boundaries.
- V2 `request_budget` is the original **combined cap**. Preparation-used plus
  remaining evaluation allowance must equal it. Phase execution IDs cannot overlap.
  Preparation history never seeds a fresh client's identity-scoped probe cache.
- Paired protocols require all three roles, selected rewriting and matching draws/
  ensemble. Standalone library/preparation plans permit target-only/rewrite-off.
- Runtime role configs retain caller credential references; serialized plans omit
  them. No live clients, callbacks, cancellation tokens or scorers in artifacts.
- Task **4.1 replaces runner/resume guards** with preparation-aware execution;
  task **4.2 replaces report-envelope guards** with actual v2 report support.
- RequestError retained evidence and physical attempts belong to **1.3/2.3**;
  preparation/workflow producers to **3.1/3.2**. Do not implement them in 1.2.
- Configuration identity uses canonical JSON: Python dict equality conflates
  nested true/1 and false/0. Preserve types, array order, omitted/null; ignore
  object-key insertion order. Transport-only readiness does not change calibration
  identity; generation controls such as stream do.

## Approvals and live allowance

The user explicitly chose **Approve implementation and bounded pilot**.
`spec.json.live_pilot_authorization` records a separate allowance:

- At most **8 additional physical inference requests**, **0 status polls**, and
  **1200 seconds of controlled total waiting**.
- Existing NIM **openai/gpt-oss-20b**; four scheduled JSON probes in balanced
  fixed/adaptive order, at most one permitted retry each, identical generation
  settings. No model switching or paid auxiliary-provider calls.
- Run only after offline/platform gates and a new dated frozen protocol. Retain
  partial/inconclusive outcomes. Extra SSE/calibration/rewrite-effect trials need
  their own declared allowance; do not replenish October 31 budgets.
- Root alone runs live commands. Never display `.env`, keys, environment dumps or
  client/scorer objects. Workers use offline fixtures. No new authorization needed
  for the above bounded pilot once its prerequisites pass.

## What was tried and why it failed

1. **Dagger archive under ignored runs/.** Dagger inferred the ancestor Git root,
   ignored module files and scaffolded a starter Query module. Symptom:
   `unknown command "test" for "dagger call"`. Unchanged source works outside
   every ancestor Git checkout. No module/engine upgrade or product fix needed.
2. **Archived tests retained under runs/.** Canonical pytest collected both trees:
   **62 import-file-mismatch errors**, exit 2. Owned tree moved safely to
   `C:/Users/mariu/AppData/Local/Temp/recall-guard-task-7-3-staging-remediation-5e9f92fc9a6540ecb437107f00ad9c7e/task-7-3-source-b1e923ca`.
   All 441 files (315 original plus pyc) preserved; root collection now passes.
   **Never recreate a source/test checkout under runs/.**
3. **Task 1.1 false READY.** Initial checks missed explicit logprob requirements
   when MCS was off and allowed mismatched response requested settings. First
   remediation used dict equality, still accepting nested boolean/number collisions.
   Final committed code checks genuine requested evidence independently of MCS
   and canonical JSON settings. All six independent reproductions pass. Two
   remediation rounds used for 1.1 only; it is now approved, not pending.
4. **Task 1.2 v2-in-v1 report.** Self-review reproduced the legacy exporter silently
   wrapping a v2 checkpoint in schema 1. Scoped export/replay guards now reject
   this until 4.2. Initial RED: 19 failed/4 passed; report RED: 1 failed. Logs:
   `runs/readiness-1-2-{red,report-red,report-green,focused,final-focused,ruff,architecture}.log`.

## Current best theory

Keep transport/policy in core and fallback selection in the harness before strict
client construction and protocol freezing. Missing/unsupported token evidence can
trigger caller-selected fallback; timeout/504, unrelated parameters, missing
calibration and fidelity rejection cannot. Preserve MCS unavailability and reuse
existing rewriting/ensemble mathematics. Never switch methods mid-evaluation or
label an original answer successful mitigation after rejecting its rewrite.

The initial NIM contract verifies SSE framing and NVCF-REQID/NVCF-STATUS headers,
but its exact pending polling route and streamed logprob support remain unverified.
Keep polling disabled; another model's route is not proof. Cancellation is
cooperative and does not promise an exact OS-I/O interrupt or remote cancellation.

## Completed evidence and original live gaps

- `b1e923c`: original 7.2 integration/readability accepted.
- `273a288`: approved readiness design/tasks recorded.
- `0f232ad`: original 7.3 accepted. Windows 3.12/3.14 each 1919 passed with one
  symlink-privilege skip; Linux 3.12/3.14 each 1920 passed. Real notebooks, external
  lean installed wheels, Unicode replay and strict docs passed. Reports: original
  spec `platform-validation.md` and `integration-validation.md`.
- `5d9d5d7`: readiness 1.1 accepted, 89 focused tests and full 2008 passed.
  Full log SHA-256 `87107448ef69d369f6429230abef891c032805fde3aa837ffccfc2cbf2635bbb`.
  This full run **predates unreviewed 1.2 changes**.

Original study remains incomplete: NIM returned HTTP504 after 302.453s under a
600s timeout; frozen eligibility allowance **4/4 used**, so no more requests under
that allowance. Timeouts did not establish credential/inference eligibility.
Opus used 67 calls for 16 items; all 16 rewrites rejected, zero valid rewrite pairs,
no mitigation-effectiveness conclusion. No fresh NIM calibration/scored comparison.
Keep `data/experiments/2026-10-31-completion-api/` and its 12 docs copies immutable.
Public-facing report: `docs/results/2026-10-31-completion-api-pilot.md`.

## Environment and workflow constraints

- Use `.venv/Scripts/python.exe` or `uv run --no-sync python`; bare Python is the
  Windows Store alias. Tool commands: workdir D:/Source/memguard_alpha, login:false,
  no TTY. C:/Users/mariu/Source/memguard_alpha is a junction here.
- Canonical gates: `.venv/Scripts/python.exe -m pytest -q -ra`,
  `.venv/Scripts/python.exe -m ruff check .`,
  `.venv/Scripts/python.exe scripts/check_architecture.py`.
- Kiro skills: `C:/Users/mariu/Source/QOpenBB/.agents/skills/kiro-*/SKILL.md`.
  Read kiro-impl + templates, kiro-review, kiro-verify-completion. These skills
  explicitly authorize task-local subagents. Exact structured worker STATUS and
  reviewer VERDICT required; two ordinary remediation rounds then bounded debug.
- Parent alone marks tasks and selectively commits explicit paths. Never git add
  -A/., destructive reset, or commit this handoff. Preserve work through restart.
- Ignore Sentrux. Native Windows, explicit UTF-8, no global event-loop changes.
  Use COBESY for later README/adopter and MkDocs/engineering prose; cite Jane Street.
- Debian WSL/Dagger available; tracked-only archives outside Git roots, checked
  for credentials before upload. No pruning/upgrades/manual engine deletion.
- Historical Windows abrupt exits/MemoryError remain unexplained; passing runs
  do not establish that those causes were fixed.
- Previous root handoff is preserved at `runs/HANDOFF-before-20261101-restart.md`.
  This HANDOFF.md remains an uncommitted scratch file.

## Memory pointers

Canonical work-state lists: `adaptive-provider-readiness`,
`completion-api-mitigation`, and closed `completion-api-platform-7-3`.
Latest entries and this handoff supersede historical paused-session notes.
Selected records retrieved using search -> timeline -> observations: **18933**
(explicit fallback/reporting requirements), **18940** (resolver before client
construction). Historical decisions, not current completion evidence. Useful
searches: `adaptive-provider-readiness`, `completion-api platform staging`.

