# Roadmap: three tracks over one provider seam

Updated: 2026-11-08. Narrative view only. **Status is tracked in one place: the
[1.0 release plan](specs/release-1.0/tasks.md).** The handover keeps its evidence.

Recall Guard needs one question answered per track: **does this signal cull
recall (memorization) on held-out data, at a stated cost?** Tracks 2 and 3 share
one measurement; they differ only in where the model runs. Users choose the
backend; scoring code never knows which one served the tokens.

```
                 ┌────────────── Track 1: completion-only (no logprobs) ─────────────┐
 frontier API ──►│ base responses + probability model → recall cull                  │
                 └───────────────────────────────────────────────────────────────────┘
                 ┌────────────── Tracks 2+3: token-evidence measurement ─────────────┐
 Ollama (local) ─┤                                                                   │
 Modal (vLLM) ───┼─► provider profile ─► CompletionClient ─► calibrate ─► divergence │
 NIM (hosted) ───┘    (endpoint, auth,      (one transport)    single /    decoding  │
                       response mapping)                       ensemble    eval      │
                 └───────────────────────────────────────────────────────────────────┘
```

## Track 1: divergence/recall culling over ordinary completion APIs

No logprobs or generation control. Base-model responses plus a probability model
estimate recall exposure. The track is independent of Tracks 2 and 3 and must not
wait on them.

| Spec | Phase | Role |
| --- | --- | --- |
| [completion-api-mitigation](specs/completion-api-mitigation/) | implementing | Main vehicle (OpenAI/Anthropic, Opus alternative) |
| [divergence-decoding](specs/divergence-decoding/) | implemented | Math layer `main + alpha·(older − newer)`; needs caller-owned logits |
| [ensemble-consensus](specs/ensemble-consensus/) | implemented | Multi-draw consensus where single draws are nondeterministic |

**Next:** a measurement plan for the completion-only path (datasets, dev/held-out
split, request and cost limits), the same format as Track 2.

## Track 2: logprob measurement, local (Ollama)

| Model | State on the dev host (32 GB, iGPU dropped by Ollama → CPU only) |
| --- | --- |
| `phi4-mini:3.8b`, `llama3.1:8b`, `llama3.2:3b`, `gemma3:4b` | Eligible: aligned logprobs, top-5 alternatives |
| `llava-llama3:latest` | Context 4096 vs requested 2048; resolve before admission |
| `gpt-oss:20b` | **Deferred on this host, not ruled out.** See below |

GPT-OSS-20B on Ollama, findings from 2026-11-08:

- VRAM is not the constraint. Ollama logs `dropping integrated GPU; to enable,
  set OLLAMA_IGPU_ENABLE=1`, so the Radeon 890M's reported "2 GiB" is never used;
  inference is CPU-only (`size_vram = 0`).
- The model *does* load from `models/`. A load-only request (no prompt, no
  generation) succeeded in 18.6 s with 13.9 GB host allocation, and free RAM fell
  from 13.4 GB to 1.7 GB. The 2026-11-06 HTTP 500 is consistent with
  insufficient free RAM on a busy host. Its body was not retained, so this remains
  the leading hypothesis, not a verified cause.
- Separate, unresolved issue: the 2026-11-05 Ollama response failed token-text
  alignment. This matches the Harmony pattern fixed for vLLM, but Ollama returns
  no token IDs, so the existing ID-based mapper does not apply. Ollama GPT-OSS
  needs its own mapping decision before it can be eligible.
- Planned: retest on the 64 GB Mac (Metal). Record free RAM before load, keep the
  500 response body, then run one eligibility probe.

Specs: [local-ai-provider](specs/local-ai-provider/) (endpoint support, mandatory
token evidence), [divergence-decoding-evaluation](specs/divergence-decoding-evaluation/)
(the study itself, requirements-generated).

## Track 3: Modal as the remote supplement to Track 2

Same measurement, same client and scoring path; Modal hosts the models the local
host cannot.

- GPT-OSS-20B on vLLM 0.17.0: hosting, readiness and auth work. **Logprobs do
  arrive:** response 09 carried 30 probability positions with top-5 alternatives.
  The *repaired final-answer mapping* has been verified only by offline replay.
- **Open gate:** one live eligibility request through `HarmonyOpenAICompatibleClient`
  after the fix. This is the cheapest decisive step and needs explicit
  authorization (pilot allowance USD 5; gross so far ≈ USD 0.30).

NIM: historical failures are recorded as "NIM unavailable". It stays a supported
profile, but no track depends on it.

## Cross-cutting design: user-selectable inference backend

Already true: Ollama, Modal/vLLM and NIM all speak OpenAI chat completions, and
one transport (`OpenAICompatibleClient`) with one budget/readiness control
serves all of them. What is missing is a **user-facing seam**. Clients are built
ad hoc in scripts, and the harness runner/scorer still construct the legacy
`NvidiaLM` directly.

Proposed (new spec, working name `inference-backends`):

1. **Provider profile**: a declarative JSON under `configs/providers/` naming
   protocol, endpoint, credential env reference (from the root `.env`), model ID,
   response mapping (`generic` | `harmony-vllm` | later `harmony-ollama`),
   required parameters (e.g. `return_token_ids`), readiness/cold-start policy and
   an evidence profile (top-k, context). It wraps `ModelConfig`; it is not a second
   config system.
2. **One factory**: `connect(profile) -> CompletionClient` picks the client class
   from the mapping field. Scripts, harness and scorer all go through it; `NvidiaLM`
   becomes a NIM profile behind the same seam.
3. **Capability check before calibration**: each profile runs the existing
   eligibility probe (aligned logprobs, alternatives, context) and records the
   result per host. That is how "deferred on this host" stays a recorded fact
   instead of a code branch.
4. **Friendly host preflight, before any load.** Each local profile declares
   a free-RAM requirement (GPT-OSS-20B: 20 GB, from the 13.9 GB observed CPU
   allocation plus headroom). The tool measures free memory without loading the
   model. When the requirement is not met, it does not attempt the load. It tells
   the adopter what it measured and what is needed, then proposes the Modal profile
   or a larger machine. Running it again later, or on another host (for example a
   64 GB Mac), re-evaluates the requirement; the model is not excluded globally.
   Keep the provider's error body whenever a load still fails.
5. **Calibration identity includes the profile.** A calibrator fitted on
   Modal-Harmony evidence is not reused for Ollama evidence of the "same" model.
6. Hosting SDKs (Modal, accelerators) stay outside the core library, as now.

## Order of work

Authoritative order and status: [1.0 release plan](specs/release-1.0/tasks.md).

1. `inference-backends` spec: requirements → design → tasks (human-gated).
2. Track 3 gate: one authorized live Modal eligibility request.
3. Track 2/3 measurement plan → fresh calibration → single and ensemble results
   reported separately → divergence-decoding evaluation (paired held-out).
4. Mac retest of Ollama GPT-OSS-20B; decide on the Ollama Harmony mapping.
5. Track 1 measurement plan, in parallel with 2–4.

Specs outside these tracks (`cmmd-backtest`, `honest-model-ranking`,
`recall-guard`, `recall-guard-package`, `review-hardening`, `docker-execution`,
`adaptive-provider-readiness`) are platform or downstream work and are not
re-scoped here.
