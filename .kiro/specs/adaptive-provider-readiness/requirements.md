# Requirements: explicit API fallback and bounded provider handling

Revision 2 — 2026-11-01. The user agreed to narrow delivery after reassessment
and requested Docker-based operation. This revision supersedes the archived
pre-reassessment specification. Existing implementation evidence remains history;
it is not proof that the complete user flow has shipped.

## Product outcome

Configure an endpoint/model and explicit parameters, select permitted features,
execute the existing Recall Guard workflow, and see what actually ran. Use token
probabilities when available. Use caller-authorized completion-only rewriting
and/or ensembles when they are unavailable. Keep MCS unavailable in that mode.

Recall Guard is a portable Python library for HTTP API calls and calculations.
Docker supplies a reproducible execution environment; it is not a new application
service or a prerequisite for importing the package in an existing Python kernel.

## Scope decisions

The immediate delivery is JSON API configuration → capability check → explicit
fallback → existing workflow → compact report → bounded measurement. Existing
optional streaming/waiting can remain, but further transport infrastructure is
not a prerequisite for this flow. Active status polling is deferred: the current
hosted NIM route is unverified and the approved pilot permits zero polls.

Preserve existing historical experiment/replay behavior. Defer new version-2
paired-runner/resume integration and a generic readiness-study framework. Do not
add another execution scheduler, request ledger or scoring/ensemble algorithm.

## Requirement 1: Explicit model-specific control

1. The caller's endpoint, model, generation settings and required capabilities shall remain explicit.
2. Request, retry and controlled-wait limits shall be caller-configurable independently of generation parameters.
3. Absent optional readiness/fallback choices shall preserve existing configured-client behavior.
4. Unsupported controls/observation capabilities shall be reported before they are relied on; model names or URLs shall not imply unobserved support.

## Requirement 2: Evidence-based states

1. Actual observations shall distinguish pending acceptance, observed progress, usable completion, temporary failure, permanent rejection and unknown outcome.
2. Timeouts/gateway failures shall retain their actual evidence and leave credential eligibility unverified unless rejection is explicit.
3. A successful probe shall establish capability only for the tested configuration and observation time.
4. Elapsed local waiting and byte/keepalive traffic shall remain distinct from provider answer/reasoning progress.

## Requirement 3: Bounded provider handling

1. A valid server retry delay shall be honored or cause deferral when the remaining allowance is insufficient; it shall never be shortened to fit.
2. A retryable failure without a usable server delay shall use bounded configured backoff and report that decision.
3. An accepted pending request shall retain its identity and defer without duplicate generation. Active status polling is deferred from this delivery; no guessed status route is permitted.
4. If the caller explicitly selects supported streaming, completion framing and bounded partial evidence shall be preserved. Streaming is optional and is not required for completion-only fallback.
5. Adaptive resubmission of ambiguous inference shall require explicit replay permission. Existing disabled-mode retry behavior remains explicit and unchanged.
6. Explicit authentication/access, unrelated parameter and hard-quota rejections shall stop automatic retries and retain their actionable reason.

## Requirement 4: Accounting and truthful failure evidence

1. Budget/retry/wait/cancellation limits shall stop further applicable admission. An internal fatal stop shall retain its originating reason rather than be mislabeled caller cancellation.
2. Existing CallBudget/RequestControl shall count every physical probe, generation and retry once, with purpose/stage. No second accounting system shall be introduced.
3. Reports shall retain actual model/role/stage outcomes, identities, timings and reported usage through explicit whitelists. Failed/stopped response evidence shall use the owning sanitizer; credentials and runtime objects shall not be exported.
4. Partial/incomplete responses shall not become usable answers or MCS evidence. Successful answers shall preserve their existing semantics.

## Requirement 5: Proportionate validation and measurement

1. The library flow shall work in ordinary Python and an existing Jupyter kernel, and in the canonical Docker environment, using the same model/configuration semantics.
2. Focused deterministic tests shall cover changed behavior: declared/observed missing tokens, service/auth failure, waiting/defer/cancel, fidelity rejection and exact physical counts. Existing transport tests shall be retained where that optional capability remains.
3. Live work shall require a dated frozen configuration, method, ordering, physical-request/wait caps and metric definitions. The existing approved allowance is unchanged.
4. Results shall separate operational availability from mitigation effectiveness, retain failed/censored observations, and identify missing evidence. An inconclusive pilot shall remain inconclusive.

## Requirement 6: Explicit fallback and actual-action reporting

1. Only a caller-supplied fallback shall select the existing completion-only rewriting/ensemble workflow when required token evidence is missing or explicitly rejected.
2. Completion-only execution shall have no MIA scorer and shall mark MCS unavailable. Agreement/self-confidence shall not be presented as equivalent MCS.
3. Requested and selected configurations/features shall be retained; only explicitly authorized logprobs/top_logprobs controls may be omitted. Endpoint/model/all other settings remain fixed.
4. Unrelated parameter rejection, timeout/504, auth/quota failure, empty/malformed/partial replies, missing fit and fidelity rejection shall not trigger missing-logprob fallback.
5. Changed target settings and newly active rewrite/verifier roles shall use fresh budgeted validation; selected MIA requires matching fitted calibration. Factories perform zero provider work; no automatic calibration or historical probe-cache seeding.
6. A compact report shall show requested/selected/observed method, model/role/stage, fallback reason/omissions, attempted/skipped/rejected actions, physical counts, usage and unknown cost. A report shall be derived from actual evidence, not a planned stage list.
7. A fidelity-rejected rewrite shall remain a failed intervention, with no target evaluation draw relabeled as successful mitigation. Any real preflight probes shall still be counted.

## Requirement 7: Small portable delivery

1. Implementation shall reuse existing clients, request control, workflow, fidelity checks and ensemble/scoring calculations; names shall express domain meaning and units, including array axes/table row grain.
2. Docker development/test execution shall use the existing uv lockfile and share its build recipe with existing Dagger CI. No OS-specific business-logic path or new hosted service is required.
3. Changed code shall receive focused regression/static checks; one full regression shall run at the coherent integration boundary. Do not repeat full OS matrices or independently rerun unchanged full suites for every helper/reviewer. Existing release compatibility policy is separate.
4. Infrastructure without a current product caller shall be inventoried and removed, consolidated or explicitly deferred through the infrastructure assessment. Preserve existing used study tooling and useful evidence; no blanket deletion of runs or old artifacts.
5. One documented application/notebook path shall expose the complete feature before additional study/resume/profile frameworks are added. README/MkDocs shall use the requested COBESY approach and cite the Jane Street distinction honestly.

## Live allowance and scope boundaries

The approved additional NIM pilot remains at most 8 physical inference requests,
0 status polls and 1200 seconds controlled waiting: four scheduled JSON probes,
balanced fixed/adaptive ordering, at most one permitted retry each, same
openai/gpt-oss-20b model/settings, no paid auxiliary calls. Its prerequisite is
the revised integrated offline/container validation task, not repeated native
OS debugging. No live calls are authorized by merely building/running Docker.
Additional calibration, SSE or mitigation-effect trials require separately
declared configuration/allowance. October 31 budgets/artifacts remain immutable.

Jane Street's token-selection method requires distribution/selection access;
the completion-only API approach is a separately measured alternative, not a
claim of mathematically equivalent decoding: [Jane Street](https://blog.janestreet.com/mitigating-memorization-in-llms/).
