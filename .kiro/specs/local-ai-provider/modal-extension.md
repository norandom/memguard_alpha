# Modal hosting extension — 2026-11-06

## Decision and scope

Add optional Modal hosting to the local-AI provider work so experiments can use
remote GPU memory without maintaining a model fleet. The user has an account
and requested this specification extension. Requirements are recorded as
[requirement 7](requirements.md#requirement-7-optional-budgeted-remote-hosting).
This document records implementation direction, not a completed design review
or authorization to deploy paid resources.

Keep GPT-OSS-20B in the candidate set. Retain the existing small Gemma, Phi and
Llama candidates, selecting one concrete model revision at a time. Verify the
serving engine, quantization and GPU compatibility before choosing hardware;
weight size alone is not a sufficient capacity check. No new model estate,
NPU driver work or frontier completion-only integration is part of this extension.

## Proposed implementation sequence

1. **Deployment prerequisite and cost plan.** Identify the existing Modal
   workspace/environment and secret configuration without exporting credentials.
   Pin one model revision and serving image. Verify the current GPU/quantization
   support, context memory requirement, SDK version and pricing. Prepare a
   reviewable deployment and shutdown plan with provider budget settings.
2. **Optional serving recipe.** Prefer vLLM behind an authenticated
   OpenAI-compatible endpoint, reusing Recall Guard's existing
   `OpenAICompatibleClient`. Keep Modal SDK and serving dependencies confined to
   deployment tooling. Pin dependencies, cache weights remotely, use zero warm
   minimum containers, a maximum of one container and bounded scale-down.
   Verify the chosen Modal deployment primitive supports those controls.
3. **Capability and lifecycle integration.** Reuse the evidence validator,
   request budget, bounded readiness and attempt receipts. Admit only actual
   aligned logprobs with required alternatives. Obtain remote identity/hardware
   from deployment observations; never use local available RAM as a remote
   admission signal. Handle cold starts, authorization failure, missing models,
   capacity unavailability, timeout and cancellation without automatic model
   substitution or replay of uncertain attempts.
4. **Bounded pilot.** Begin with endpoint eligibility and a small fixed prompt
   set. Proposed spending target: USD 5 total, with provider budget configured
   before execution. Specify the exact request/token/time limits in the reviewed
   pilot plan; this is not permission for an unbounded calibration batch.
   Record cold-start and warm timings, evidence acceptance and cost. Release
   compute at completion and verify it; report persistent storage separately.
5. **Measurement integration and documentation.** Only after eligibility passes,
   prepare separately bounded calibration and paired single-answer/ensemble
   measurements using existing harnesses. Publish dated receipts, runtime
   provenance, cost per usable measurement, and differences from historical NIM
   or local quantizations. Add adopter setup and engineering instructions for
   start, stop, budget verification and recovery.

These are planning work packages. Generate the executable task graph after the
extension design review; do not treat them as completed or approved tasks.

## Acceptance and validation direction

- Offline checks cover authentication redaction, explicit configuration,
  missing/misaligned/censored evidence, frozen identity drift, request bounds,
  uncertain outcomes, cancellation and cleanup failure reporting.
- Deployment checks establish one-container capacity, scale-to-zero behavior,
  authenticated access and preserved caches. CI must not start paid GPUs.
- A live receipt must establish actual logprobs and usable alternatives; a
  successful HTTP response or advertised API compatibility is insufficient.
- Preserve partial top-k status. Logprob availability alone does not prove
  full-vocabulary divergence decoding or control before token selection.
- Keep code idiomatic and small, with names expressing quantities and units;
  document token-position alignment and tabular row grain where relevant.

## Budget assumptions and sources

Research snapshot from November 6, 2026, not fixed contractual prices:
Modal lists L4 at USD 0.000222/second (~0.80/hour), A10 at 0.000306/second
(~1.10/hour), and L40S at 0.000542/second (~1.95/hour), GPU only. CPU, RAM,
startup/retained compute, storage and other applicable charges must be included
in the deployment estimate. Starter advertises USD 30 monthly credits; verify
the user's actual remaining credit rather than assuming the pilot is free.

A USD 5 target is not an exact billing guarantee: verify budget scope,
enforcement delay and in-flight behavior. Combine provider budgets with bounded
requests and resource lifetime. Do not assume a lower GPU hourly rate produces
a lower cost per usable result.

- [Modal endpoints](https://modal.com/docs/guide/endpoints): shared endpoints
  bill by tokens; dedicated endpoints bill for resources and support scaling.
- [Modal pricing](https://modal.com/pricing) and
  [budgets](https://modal.com/docs/guide/budgets): recheck before deployment.
- [vLLM chat API](https://docs.vllm.ai/en/latest/api/vllm/entrypoints/openai/chat_completion/protocol/):
  candidate serving interface; actual endpoint evidence remains mandatory.
- [Jane Street: mitigating memorization](https://blog.janestreet.com/mitigating-memorization-in-llms/):
  provider capability is a prerequisite for some experiments, not evidence that
  mitigation has been implemented or measured.

## Review record

Requirements review: inline PASS. Checked explicit selection, mandatory evidence,
remote provenance, lifecycle failures, bounded spend, portability and separation
from the frontier path. No implementation or live Modal measurement performed.
The user approved the [extension design](modal-design.md). Implementation tasks
8–10 are generated and independently reviewed in [tasks.md](tasks.md), pending
task approval independently of the existing local-provider approvals.
