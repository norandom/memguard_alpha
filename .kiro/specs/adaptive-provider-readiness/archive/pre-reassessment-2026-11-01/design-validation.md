# Independent design validation

Date: 2026-11-01. Scope: design quality only; no implementation, provider calls,
credentials, staging, or approval changes.

## Review summary

The revised design preserves the existing core/harness boundary and addresses
all 29 numbered acceptance criteria (4 + 4 + 6 + 4 + 4 + 7). The initial review
identified one integration-contract gap; the first bounded repair resolves it.
No critical findings remain after re-review of sections 5.1 and 7.1.

## Initial finding and resolution

**Issue 1: Selected-plan handoff does not define the experiment's supported shape
or its additional probes.**

**Initial concern:** Design sections 6–7 allowed caller-selected rewriting and ensemble
options, then reconstruct evaluation clients using the remaining budget. However,
`ExperimentRunner._validate_workflow()` requires a rewrite policy and all three
roles; `_preflight()` always selects rewriting; its run schedules both evidence
arms. `RecallWorkflow._preflight()` keys evidence by `id(client)`, so freshly
constructed evaluation clients necessarily issue additional probes. The design
did not explicitly choose the supported experiment plan shapes or whether those
fresh probes remained intentional.

**Impact:** A valid prepared library plan can fail at experiment construction,
or evaluation can perform extra probes absent from its displayed call plan.
Broadening the paired experiment implicitly would also change completion and
metric assumptions in versioned artifacts.

**Suggestion:** Keep the paired runner's existing shape explicitly and reject
incompatible experiment plans before preparation/network work; retain arbitrary
feature choices in the library workflow. State that evaluation and resume
intentionally re-probe fresh clients, charge those calls to the remaining
evaluation budget, and include them in nominal and upper-bound planning. Do not
reuse historical preparation evidence as a fresh client cache implicitly.

**Traceability:** Requirements 4.2, 6.1, 6.5, 6.6. **Evidence:** design sections
6–7; `intervention.py` `_validate_workflow`, `_preflight`; `workflow.py`
`_preflight`; `intervention_config.py` `experiment_plan`.

**Resolution verified:** Section 7.1 now preserves the paired runner's four
conditions, all three roles, equal draw counts, and draw-zero reuse. Incompatible
plans are rejected before preparation/network work while library options remain
available. Fresh factories do no provider work; evaluation/resume preflights
intentionally re-probe, with their calls included in remaining-budget admission
and nominal/upper-bound planning. Preparation evidence cannot seed the fresh
client cache, and separate execution namespaces retain physical accounting.
The initial NO-GO finding is closed.

Section 5.1 also clarifies that blocked synchronous reads do not emit periodic
callbacks. A read-only snapshot computes elapsed time for caller-owned observers;
this preserves the stated cooperative cancellation bounds without promising an
exact deadline or remote inference cancellation.

## Design strengths

- Missing token evidence is preserved and resolved before rejecting client
  construction or selecting a scorer. Fallback requires explicit authorization,
  narrow logprob-control omissions, fresh role validation, and no mid-run switch.
- Physical admission, unknown outcomes, cooperative cancellation, supported
  pending identities, and version-1 serialization are addressed explicitly.
  The unverified gpt-oss polling route and streamed token capability remain
  visibly unavailable; no live support or remote cancellation is claimed.

## Final assessment

**GO for design quality after the first bounded repair.** The revised handoff is
implementable within the existing runner and request-accounting boundaries;
there are zero remaining critical findings. This assessment is not user approval
or evidence of working implementation/live provider support. Obtain user design
approval before task generation or implementation. Design, tasks, and
implementation remain unapproved.

Core steering files `product.md`, `tech.md`, and `structure.md` are absent;
the available `recall-guard-package.md` was read in full. Source inspection was
read-only. No implementation tests or provider-capability measurements were run.
