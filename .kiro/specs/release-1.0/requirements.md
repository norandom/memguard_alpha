# Requirements: release 1.0

Review 2026-11-08: [gap analysis](research.md) and [design draft](design.md). The current
requirements and design await user approval. The review refines method boundaries,
first-class `uv run` + Docker support, `psutil` + `/api/ps` host preflight, and
Rich failure presentation.

Created 2026-11-08 at the user's request: consolidate all open specs into one
linear plan with a single status pointer. Every open concern is either a task
here or a numbered entry in the concern register ([tasks.md](tasks.md)).

## Goal

Answer one question per track, on held-out data at a stated cost: **does this
signal cull recall (memorization)?**

- Track 1: completion-only APIs (no logprobs): base responses plus a probability
  model.
- Track 2: token logprobs from local Ollama models.
- Track 3: the same measurement on Modal, for models the local host cannot run.

Tracks 2 and 3 are one measurement on different hosts. Divergence decoding
([Jane Street](https://blog.janestreet.com/mitigating-memorization-in-llms/))
is evaluated after fresh calibration exists.

## Requirement 1: one tracker

1. The release-1.0 `tasks.md` is the only place progress is recorded. Child specs keep
   their history and detail; their `spec.json` names `release-1.0` as tracker.
2. Tasks run in the listed order. The "Current" pointer names the single active
   task. A task that cannot proceed is marked blocked with its concern ID, and the
   pointer stays on it until the user reorders, parks or unblocks it.
3. Every live call, download or paid resource is a separately gated task with its
   request/cost limit stated before execution.

## Requirement 2: user-selectable inference backend

1. A user selects Ollama, Modal or NIM through a declarative provider profile
   that wraps the existing `ModelConfig`; scoring code does not branch on backend.
2. One factory builds the client from the profile, including the response
   mapping (`generic` or `harmony-vllm`). Scripts, harness and scorer use it; the
   legacy `NvidiaLM` path is reachable as a NIM profile.
3. A local profile declares its free-RAM requirement (GPT-OSS-20B: 20 GB). Before
   any load, the tool measures free memory. When the requirement is unmet it does
   not load. It states the measured and required amounts and proposes the Modal
   profile or a larger machine. The result is per host, never a global exclusion.
4. Each profile's eligibility probe (aligned logprobs, alternatives, effective
   context) is recorded per host before calibration.
5. Calibration identity includes the profile; calibrators are not reused across
   evidence profiles.
6. Hosting SDKs and accelerator runtimes stay outside the core library.
7. When a provider load or request fails, its error body is retained, minus
   credentials.

## Requirement 3: measurement before claims

1. Single-answer and ensemble results are reported separately, each with its
   own request budget.
2. Partial top-k logprobs are not presented as a full next-token distribution.
3. Pilot or availability results are not presented as mitigation efficacy.

## Requirement 4: adopter-friendly documentation

The README states which backend fits which machine, current verified state, and
sizing guidance. Planned features are labeled as planned until shipped.
