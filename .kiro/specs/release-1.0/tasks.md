# Release 1.0: tasks and status

**Current: M1.5** · Updated 2026-11-08 · Narrative: [roadmap](../../roadmap.md)

Review 2026-11-08: [research review](research.md#7-streamlined-task-alignment-for-tasksmd) and [design draft](design.md).
Task list, requirements and the M1.1 design approved by the user on 2026-11-08. Review proposals do not complete tasks, move this
pointer, or renew any live-call allowance.

Legend: `[x]` done · `[ ]` open · **Gate**: `offline` needs no approval beyond
this plan; `authorize` needs explicit user approval of the stated limit before
execution. Source links point to the child-spec detail that a task absorbs.

## M0. Reconcile bookkeeping (offline)

- [x] M0.1 Close specs that are done or superseded
  - _Outcome 2026-11-08: all subtasks of honest-model-ranking (5.2 removed by its design note) and cmmd-backtest were already ticked, with code and tests present; parent boxes ticked. Phases set: cmmd-backtest, recall-guard-package, review-hardening → implemented; adaptive-provider-readiness → implementing; recall-guard → superseded (banner added to its tasks.md). No source code changed._
  - `recall-guard`: superseded by the shipped package; mark superseded (C11).
  - `honest-model-ranking`, `cmmd-backtest`: parent checkboxes open while the
    code ships (`recall_guard/harness`, `recall_guard/portfolio`). Verify each
    parent against code and tests, then tick or list the real remainder.
  - `recall-guard-package`, `review-hardening`, `ensemble-consensus`,
    `divergence-decoding`: all tasks ticked; set phase `implemented`.
  - Done when every child `spec.json` phase matches its `tasks.md` and the spec
    register below shows no "verify" entries.
- [x] M0.2 Commit the consolidation (README, roadmap, release-1.0 spec, `spec.json`
  tracker fields). Preserve `.env.sample` and `.openidea/` unstaged (C12).
  - _Outcome 2026-11-08: committed as f67378d on `main`; not pushed (C13). Steering edits from the design session left unstaged for user review._

## M1. Backend seam and adopter flow (offline)

Absorbs [adaptive-provider-readiness](../adaptive-provider-readiness/tasks.md)
3.2, 3.3, 4.1, 4.2, 5.1 and [docker-execution](../docker-execution/tasks.md) 3.2.
Requirement 2 above is the new scope.

- [x] M1.1 Design: provider profiles, `connect()` factory, RAM preflight,
  calibration identity ([design.md](design.md), Requirement 2). Human-gated design approval.
  - _Outcome 2026-11-08: approved as written, including `psutil` in core and Mistral as a target._
- [x] M1.2 Finish truthful execution failure/stop handling (APR 3.2) and prune
  unused polling scaffolding (APR 3.3).
  - _M1.2a (APR 3.2) accepted 2026-11-08: both 2026-11-01 defects were already fixed by ff70061 on the ordinary, native and callback paths; two focused regressions were added in `tests/harness/test_prepared_execution.py` (5 cases). Focused run: 425 passed; ruff clean. Gap logged as C17. M1.2b (APR 3.3 polling prune) accepted 2026-11-08: fixture-only `PendingStatusPolicy`, poll loop and status GET removed; HTTP 202 now defers after exactly one POST; legacy `pending_status: null`/`max_poll_requests: 0` keys still load for historical NIM-refresh plans. Full offline suite 2867 passed, 2 skipped; ruff and architecture checks clean. The APR 3.3 helper-inventory half is not done; it stays out of scope until a milestone needs it._
- [x] M1.3 Implement profiles + factory; route scripts, harness and scorer
  through it; NIM via profile (R2.1–2.2, 2.5–2.6).
  - _Outcome 2026-11-08: `recall_guard/core/profiles.py` (`ProviderProfile`, `load_profile`, `connect`), seven profiles in `recall_guard/core/provider_profiles/`, `CalibrationIdentity` v2 (rejects reuse across response mappings; legacy dicts unchanged), `MemoryGuardedScorer.from_profile`, `--profile` on the local probe and smoke scripts. NIM works as an OpenAI-compatible profile; `harness/runner.py`, `harness/smoke.py` and `MemoryGuardedScorer.calibrate` still use legacy `NvidiaLM` (different result type, public API kept). The Modal endpoint comes from `MODAL_ENDPOINT_URL`; the placeholder is refused. Full offline suite 2894 passed, 2 skipped; ruff and architecture checks clean. Follow-ups: C18, C19._
- [x] M1.4 Implement the free-RAM preflight with the friendly Modal / larger-host
  proposal and error-body retention (R2.3, 2.7; C1).
  - _Outcome 2026-11-08: `preflight_host()` in `recall_guard/core/profiles.py`, run by `connect()` by default. Free RAM is read with psutil only for loopback endpoints; a remote local endpoint needs a snapshot taken on the model host, otherwise it is blocked as `host_memory_unknown`. A failed gate sends no request; the one exception is a read-only `/api/ps` call, which passes the gate if the model is already loaded. It suggests fallback profiles or a larger machine. Rich failure panels are wired into the probe and smoke scripts. HTTP error bodies (bounded, redacted) are now kept in probe receipts and local execution rows (C1 root cause of the missing body). `psutil` added to core; lock changed by two lines only. Full offline suite 2935 passed, 2 skipped. Cosmetic: the panel lists alternatives twice (M1.5)._
- [ ] M1.5 Integrate the adopter flow and compact report (APR 4.1); update README
  "Choose where your model runs" from planned to shipped (APR 4.2, R4).
  - _M1.5a (code) done 2026-11-08: profiles ship as package data in `recall_guard/core/provider_profiles/` and load from an installed wheel (C18 resolved); the failure panel lists each alternative once; `examples/adopter_flow.py --fixture` runs offline: blocked GPT-OSS → proposed fallback → existing preparation/workflow → one compact record with exact request counts and MCS unavailable in completion-only mode. Every record says token-evidence outcomes are simulated. Full offline suite 2950 passed, 2 skipped. C17 not reachable in this sequential flow. M1.5b (docs, COBESY) open._
- [ ] M1.6 Offline acceptance: one full regression + Docker smoke of the
  selection → fallback → report flow (APR 5.1, Docker 3.2).

## M2. Track 3 gate: Modal live eligibility (authorize)

- [ ] M2.1 One live eligibility request for GPT-OSS-20B through the Harmony
  profile after the response-mapping fix. Limit: 1 request, 0 retries, within
  the USD 5 pilot allowance. Recheck app inventory first; stop all apps after (C3,
  C10). Source: [local-ai-provider](../local-ai-provider/spec.json) modal extension.

## M3. Measurement plan for Tracks 2 and 3 (offline, then authorize)

- [ ] M3.1 Resolve LLaVA effective context, or exclude it from the matched set (C4).
- [ ] M3.2 Write the plan: datasets, development/held-out split, model and
  profile identities (Ollama: Phi-4 Mini, Llama 3.1 8B, Llama 3.2 3B, Gemma 3 4B;
  Modal: GPT-OSS-20B), generation settings, total request/token/cost limits.
  User approves the limits.

## M4. Fresh calibration and evaluation (authorize)

- [ ] M4.1 Fit calibrators per profile; evaluate held-out single-answer results.
- [ ] M4.2 Run the ensemble with its separately approved budget; report separately.
- [ ] M4.3 Publish a dated report (docs/results) and link it from the README.

## M5. Track 1: completion-only pilot (authorize)

Absorbs [completion-api-mitigation](../completion-api-mitigation/tasks.md) 6.2–6.4,
blocked since November on NIM (C6).

- [ ] M5.1 Decision: replace the NIM logprob-capable model with an M4-calibrated
  profile (Modal GPT-OSS-20B or an Ollama model). The spec forbids automatic model
  change, so this needs the user's choice.
- [ ] M5.2 Freeze protocol and preflight (CAM 6.2); matching calibration (CAM 6.3).
- [ ] M5.3 Run the bounded paired pilot with fidelity audit (CAM 6.4) and publish.

## M6. Divergence decoding evaluation

Absorbs [divergence-decoding-evaluation](../divergence-decoding-evaluation/requirements.md)
revision 2 (requirements awaiting review); revision-1 tasks are historical.

- [ ] M6.1 Review revision-2 requirements. Decide the statistical source:
  full-distribution logits (self-hosted callback, e.g. a Modal worker) versus a
  validated approximation from partial top-k (C5, C8).
- [ ] M6.2 Design and tasks (human-gated), reusing completed DDE 1.1–2.2 work.
- [ ] M6.3 Development tuning, frozen configuration, paired held-out measurement
  (authorize), dated report.

## Parked (not on the linear path; reopen by moving into a milestone)

- P1 Ollama GPT-OSS-20B on a host with ≥ 20 GB free RAM (the user's 64 GB Mac).
  Needs a final-channel mapping for Ollama, which returns no token IDs (C1, C2).
- P2 NPU branch: [local-ai-provider](../local-ai-provider/tasks.md) 2.2, 4.1, 4.2 (C7).
- P3 NIM operational pilot: APR 5.2; reopen only after a fresh NIM probe passes (C6).
- P4 iGPU offload trial with `OLLAMA_IGPU_ENABLE=1` (C15).
- P5 Finish the Ollama store move: original `%USERPROFILE%\.ollama\models`
  copy remains (C14).
- P6 Mistral target: the user will supply a local proxy endpoint later (2026-11-08).
  Add it as a provider profile once the endpoint exists; it does not block M1.

## Concern register

| ID | Concern | State | Tracked by |
| --- | --- | --- | --- |
| C1 | Ollama GPT-OSS-20B HTTP 500 on 2026-11-06; 13.9 GB CPU allocation vs ~13.4 GB free; body not retained | Mitigated: RAM gate and body retention shipped (M1.4); cause still a hypothesis | P1 |
| C2 | Ollama GPT-OSS token-text alignment failed (2026-11-05); Harmony ID mapper needs token IDs | Open | P1 |
| C3 | Modal repaired mapping verified only by offline replay | Open | M2.1 |
| C4 | LLaVA effective context 4096 vs requested 2048 | Open | M3.1 |
| C5 | Score normalization semantics unknown; alternatives are partial top-k | Open | M3.2, M6.1 |
| C6 | NIM unavailable (timeouts, HTTP 504) | Recorded | M5.1, P3 |
| C7 | NPU artifact access and runtime/driver compatibility unresolved | Parked | P2 |
| C8 | Divergence decoding needs a validated statistical source, not partial top-k | Open | M6.1 |
| C9 | Calibrators must not cross evidence profiles | Requirement | M1.3 |
| C10 | Cost: USD 5 pilot allowance (≈ 0.30 gross used), USD 30/month cap operator-confirmed, Modal weight storage ≈ USD 1.16/month | Watch | M2–M6 gates |
| C11 | Child spec phases/checkboxes out of sync with shipped code | Resolved 2026-11-08 | M0.1 |
| C12 | Pre-existing user changes `.env.sample`, `.openidea/` | Preserve | M0.2 |
| C13 | Remote push not completed for earlier work | Open | M0.2 |
| C14 | Ollama store move incomplete | Parked | P5 |
| C15 | Integrated GPU unused by Ollama by default | Parked | P4 |
| C16 | Modal attempt 08 raw response not retained | Recorded, no action | — |
| C17 | A failure latched before any readiness trace (e.g. a settings check that fails before the first request) leaves `root_observation` empty, so woken queued observations keep `caller_cancelled`. Not reproducible in current flows | Low; watch | M1.6 |
| C18 | Profiles resolved only from the checkout | Resolved 2026-11-08 (package data) | M1.5 |
| C19 | `MODAL_ENDPOINT_URL` template line | Resolved 2026-11-08 (d8a97b0); value set after M2 deploy | M2.1 |

## Spec register

| Spec | Status under this plan |
| --- | --- |
| adaptive-provider-readiness | Absorbed into M1 (3.2–5.1); 5.2 parked as P3 |
| cmmd-backtest | Implemented (reconciled M0.1) |
| completion-api-mitigation | Absorbed into M5 (6.2–6.4) |
| divergence-decoding | Implemented; math layer reused by M6 |
| divergence-decoding-evaluation | Absorbed into M6 (revision 2) |
| docker-execution | Absorbed into M1.6 (3.2) |
| ensemble-consensus | Implemented; reused by M4.2 |
| honest-model-ranking | Implemented (reconciled M0.1) |
| local-ai-provider | Modal part → M2; NPU → P2; Ollama GPT-OSS → P1 |
| recall-guard | Superseded (M0.1) |
| recall-guard-package | Implemented |
| review-hardening | Implemented |
