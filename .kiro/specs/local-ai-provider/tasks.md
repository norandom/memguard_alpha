# Implementation tasks: local AI provider

Design and tasks approved 2026-11-05. Work on `main` in the original
checkout, using PowerShell on Windows; WSL is only for Dagger CI. No release tag
is part of this plan. Use the existing OpenAI-compatible client for Ollama/NIM.

Tasks run sequentially because they share contracts and one test machine. Each
executable task targets 1–3 hours of engineering work, excluding downloads.
Do not turn a failed feasibility task into unlimited runtime debugging.

Conditional ordering: a failed setup or NPU probe completes an assessment, not
NPU support. Record dependent NPU tasks as skipped with reasons; later Ollama
integration may continue. Never mark skipped NPU work as verified support.

Task approval permits the stated bounded feasibility probes, not a calibration
batch: at most two inference attempts per candidate/configuration, no retries,
2,048-token context, 32 output tokens and 300 seconds per attempt, one active
request and one loaded test model. Freeze the candidate/configuration list and
total cap before execution; server-validation calls count within the same cap.
If the cap cannot cover a required check, retain it as unperformed. Downloads and
larger measurement batches need concrete artifact and budget plans before execution.
No driver, BIOS or WSL changes are included.

- [x] 1. Establish local runtime prerequisites
- [x] 1.1 Inventory existing hosting and prepare the candidate manifest
  - Identify installed Ollama, models and hardware without starting inference.
  - Select exact small Gemma and GPT-OSS-20B candidates separately; record digests, licensing, context configuration, tokenizer/template and available metadata.
  - Prepare a release-matched Gemma NPU artifact/runtime manifest, including download footprint and installation commands. Missing driver support produces a blocked receipt, not a driver installation.
  - Done when a reviewable manifest defines the candidates, prerequisite status and total feasibility cap, with dedicated/shared/system memory distinguished.
  - _Boundary: Local endpoint admission, Gemma NPU feasibility gate_
  - _Requirements: 1.1, 1.2, 1.4, 3.5, 3.7, 6.1, 6.3_

- [x] 1.2 Prepare the isolated optional NPU environment
  - _Outcome: Failed-prerequisite assessment reviewed; see npu-setup-receipt.json. No NPU runtime installed or support verified._
  - Use the selected manifest to set up only the compatible user-space environment and artifact; keep accelerator dependencies outside the library environment.
  - Pin and record runtime, artifact and driver versions. Verify imports and artifact availability without inference.
  - Done when setup is reproducible or a sanitized failed-prerequisite receipt explicitly disables the NPU branch.
  - _Boundary: Gemma NPU feasibility gate_
  - _Requirements: 1.2, 3.1, 3.7, 6.1_

- [ ] 2. Establish feasibility before integration build-out
- [x] 2.1 Probe Ollama using the existing compatible adapter
  - _Outcome: Probe implementation independently reviewed; 181 focused/regression tests passed. Gemma passed one live probe: aligned logprobs and five alternatives, 8.547 seconds, CPU execution, context 2048. GPT-OSS responded in 24.953 seconds on CPU but failed exact token-text alignment and remains ineligible. Two physical attempts total, no retries. See ollama-feasibility-receipt.json; no GPU/NPU or mitigation claim._
  - Build a standalone bounded diagnostic using existing request controls, with frozen prompts, independent candidate receipts and explicit unauthenticated access; never attach a NIM credential.
  - Validate nonempty output, aligned logprobs and required alternatives locally in the probe; do not depend on the later shared validator. Read back the effective context configuration where available.
  - Record observed model/runtime/mode, timing, memory and unknowns. Persist admission before the attempt, redact secrets/private prompts and retain timeout outcomes without replay or assuming remote execution stopped.
  - Done when each admitted candidate has an eligible, ineligible or unavailable receipt; generation alone is not success.
  - _Boundary: Local endpoint admission_
  - _Depends: 1.1_
  - _Requirements: 1.1, 1.3, 1.4, 2.1, 2.2, 2.3, 3.3, 3.4, 3.5, 3.6, 5.1, 5.2, 5.5_

- [ ] 2.2 Probe Gemma token evidence and actual NPU execution
  - _Blocked: NPU setup prerequisites failed in task 1.2; live probe unperformed. This conditional branch does not block Ollama integration._
  - Build a self-contained diagnostic with its own bounded accounting, sanitized receipts and probe-local validation; use fake-runtime tests before live execution.
  - Verify text-only generation, matching prefix/token positions and observed NPU placement. Capture score semantics and perform stable natural-log normalization only over full-vocabulary logits.
  - Reject pruned-only scores, missing alignment and unverified placement; do not substitute another model or infer NPU use from configuration.
  - Done when the receipt and fake-runtime tests establish either actual eligibility or the precise failed gate. Failure disables serving build-out while allowing Ollama work to continue.
  - _Boundary: Gemma NPU feasibility gate_
  - _Depends: 1.2_
  - _Requirements: 2.1, 2.2, 2.4, 2.5, 3.1, 3.2, 3.4, 3.6, 3.7_

- [x] 3. Integrate evidence and calibration provenance
- [x] 3.1 Implement shared local identity and token-evidence validation
  - Consolidate proven probe checks into pure typed contracts; document token positions, vocabulary axes, units and channel mapping with domain-accurate variable names.
  - Cover Unicode, whitespace, censored/invalid scores, missing alternatives and partial-distribution handling with fixtures. Do not manufacture alignment by retokenizing.
  - Done when the validator accepts complete aligned evidence and rejects invalid evidence for every response, with no accelerator imports.
  - _Boundary: Local evidence and provenance_
  - _Depends: 2.1, 2.2_
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 6.4_

- [x] 3.2 Integrate optional provenance into existing provider contracts
  - Add optional configuration/response provenance and parse the versioned local response extension through the existing adapter; keep metadata out of generation settings.
  - Bind Ollama sidecar observations before/after batches without claiming atomic identity guarantees. Record driver, backend, context and template changes as revalidation triggers.
  - Done when fixtures show correct provenance round trips, explicit unknowns and unchanged legacy serialization when optional fields are absent.
  - _Boundary: Local evidence and provenance, existing provider integration_
  - _Requirements: 1.1, 1.2, 1.3, 3.3, 6.1_

- [x] 3.3 Bind local provenance to the existing calibrator
  - Extend client and response identity checks together so response reconstruction preserves provenance.
  - Preserve absent-field legacy identities. Unknown provenance permits labeled within-run exploration but never establishes cross-run reuse.
  - Done when regressions verify drift rejection, missing-evidence score unavailability and unchanged existing-client behavior.
  - _Boundary: Calibration and measurement_
  - _Requirements: 2.6, 4.1, 4.2_

- [ ] 4. Connect eligible NPU evidence to an endpoint, conditionally
- [ ] 4.1 Check an existing NPU server against the required contract
  - _Blocked: Conditional skip because task 2.2 cannot establish NPU eligibility._
  - Execute only after task 2.2 establishes actual eligibility; otherwise record a conditional skip.
  - Within the frozen feasibility budget, verify a vendor endpoint's token evidence, selected controls and provenance; prefer reusing it unchanged.
  - Done when a routing receipt selects the verified existing server, identifies the specific missing server capability, or records that budget prevented verification.
  - _Boundary: Local endpoint admission_
  - _Depends: 2.2, 3.2_
  - _Requirements: 1.3, 2.1, 2.3, 3.1, 6.1_

- [ ] 4.2 Supply a minimal experimental bridge only when necessary
  - _Blocked: Conditional skip because direct NPU eligibility is unverified; no bridge build-out._
  - Execute only after direct NPU eligibility and a recorded failure of the existing server contract. A successful existing server makes this task unnecessary.
  - Implement one configured text-only model, non-streaming compatible responses, genuine token evidence and observed provenance in an isolated host; use loopback and a bounded single-request queue.
  - Test unsupported settings, model mismatch, nontext/streaming rejection, queue saturation and probability semantics with a fake runtime. Stop and rescope if a general server or more than the bounded task is required.
  - Done when contract fixtures pass and any permitted live verification has a receipt; do not claim unperformed live checks passed.
  - _Boundary: Gemma NPU feasibility gate, optional host integration_
  - _Depends: 4.1_
  - _Requirements: 1.1, 1.2, 2.1, 2.2, 2.4, 3.1, 3.6, 5.2, 6.1_

- [x] 5. Integrate bounded local measurements
- [x] 5.1 Implement frozen plans and bounded attempt persistence
  - Validate model/configuration manifests, corpus hashes, token/context/time/request bounds and explicit limits before execution.
  - Reuse existing request controls for journaled admission, cancellation and partial outcomes; include a CLI dry-run that makes no inference calls.
  - Done when tests prove limits include probes/retries, cancellation stops new admissions and unknown outcomes are retained without silent replay or secret/private-prompt export.
  - _Boundary: Calibration and measurement, attempt integration_
  - _Depends: 3.3_
  - _Requirements: 1.3, 5.1, 5.2, 5.5_

- [x] 5.2 Connect plans to existing calibration and ensemble execution
  - Integrate the existing scorer and thin measurement CLI; count the scorer's own eligibility probe and serialize initial-machine draws.
  - Validate every response and identity against the selected features; stop calibrated claims on drift, with no completion-only or hosted fallback.
  - Done when fake endpoint integration runs produce separately tracked single and ensemble arms with frozen splits/settings and correct failure records.
  - _Boundary: Calibration and measurement, scorer integration_
  - _Requirements: 2.1, 2.2, 2.3, 2.6, 3.6, 4.1, 4.2, 4.3_

- [x] 5.3 Produce honest local evaluation reports
  - Report planned/completed/parsed and failed/unparsed denominators, pair only common usable items and retain draw-level identity.
  - Separate AUC, calibration and mitigation claims; report dedicated/shared/system memory, load/generation timing and unknown values correctly.
  - Label the historical NIM attempts "NIM unavailable" while retaining 504 evidence; keep Opus outside this path. Name arrays/columns by role and units and join by item/arm/draw keys.
  - Done when fixture reports expose partial results, actual execution stages and all limitations without credentials or private prompts by default.
  - _Boundary: Calibration and measurement reporting_
  - _Requirements: 3.3, 3.4, 3.5, 4.3, 4.4, 4.5, 5.3, 5.4, 5.5, 6.4_

- [x] 6. Deliver the integration and verify it
- [x] 6.1 Add the runnable local-provider example and technical guidance
  - Demonstrate explicit endpoint/model/logprob settings through existing APIs, with PowerShell commands and no mandatory accelerator imports.
  - Update README and MkDocs with candidate-versus-verified capability tables, dated result links, external-host reachability from Docker CI and Jane Street attribution; keep frontier completion-only work distinct.
  - Done when the example passes against fixtures and documentation clearly describes optional hosting, setup prerequisites, evidence limitations and the existing adapter reuse.
  - _Boundary: Local example and documentation integration_
  - _Depends: 5.3_
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [x] 6.2 Verify integrated behavior and publish the evidence summary
  - _Outcome: Independently reviewed. Dagger: 2722 tests passed; strict docs, package build and installed-wheel fixture smoke passed. See implementation-validation.md. Conditional NPU tasks remain blocked; no additional live calls or calibration batch._
  - Run focused tests, Ruff and architecture checks through PowerShell; run one coherent offline regression and strict docs build through Dagger CI.
  - Verify lean imports, fake endpoint reachability, legacy behavior, evidence rejection, identity drift and cancellation. Include conditional skip reasons.
  - Done when a source-matched summary distinguishes passing automated checks from live feasibility and unperformed measurement. No larger live batch or release tag is implied.
  - _Boundary: Integration validation_
  - _Requirements: 1.3, 2.2, 4.1, 4.2, 5.2, 6.1, 6.2_

Technical documentation is included because it is explicitly required by this
project's accepted requirements, superseding the skill's generic code-only guideline.

## User-authorized smoke-test extension, 2026-11-06

- [x] 7.1 Verify the broader local model set with memory-aware serial admission
  - _Outcome: Independently reviewed live receipt and publication. Exactly five non-GPT requests, zero retries; all five models returned aligned logprobs with five alternatives on CPU. First four environments matched; LLaVA remained ineligible due to 4096 versus 2048 context. Five unloads verified empty. Fresh focused tests: 22 passed; integrated provider scope: 153 passed, one optional skip; strict docs passed. See docs/results/2026-11-06-local-model-smoke.md. No calibration, ensemble, NPU or mitigation measurement._
  - _Current execution selection, 2026-11-06: User requested Ollama without GPT and Modal including GPT-OSS-20B. The local run explicitly selects only the five non-GPT models below (five inference requests maximum); the reusable CLI retains conditional GPT admission for future explicitly selected runs._
  - User explicitly requested testing the other documented models and excluding GPT-OSS when insufficient memory is available.
  - Include `phi4-mini:3.8b`, `llama3.1:8b`, `llama3.2:3b`, `gemma3:4b`, `llava-llama3:latest` and conditional `gpt-oss:20b`. Download missing artifacts into the existing ignored `models/` store; record registry identities and sizes before pulling.
  - Add a reusable smoke CLI using the existing OpenAI-compatible adapter and shared evidence validation, with dry-run default and explicit execution. Check actual available host RAM immediately before admission; unknown memory skips GPT-OSS. Default GPT-OSS threshold is 16 GiB available RAM, an explicit conservative policy rather than a proven minimum.
  - Freeze installed identities/settings before inference, persist intent, allow at most one inference request per model (six total), zero retries, 32 output tokens and 300 seconds per request. Refuse existing receipts and stop on uncertain transport outcomes. No calibration or ensemble measurement is authorized here.
  - Use one loaded model and one active request; unload each tested model and verify unload before the next. Preserve probability-validation results separately from context/identity failures. Report unsupported/missing evidence honestly; do not manufacture alignment or replace models.
  - Add focused offline regressions and independent review, run the bounded live smoke, and record dated outcomes. No driver/NPU/BIOS changes, additional provider or generic hosting framework.
  - _Boundary: Local endpoint admission and operational smoke tooling_
  - _Requirements: 1.1, 1.3, 2.1, 2.2, 2.3, 3.3, 3.4, 3.6, 5.1, 5.2, 5.4, 5.5, 6.4_

## Modal extension: approved implementation and bounded pilot

Design: [modal-design.md](modal-design.md). These tasks are additive; existing
local work and conditional NPU status are unchanged. No parallel markers: the
tasks share the deployment manifest, lifecycle and receipt contracts. Coding
subtasks are intended as focused 1–3 hour units. Technical documentation and
live acceptance are included because the accepted project scope requires them.
Task approval authorizes implementation and offline validation; task 10.2 has
a separate paid-execution gate. No deployment occurs during planning or CI.

- [x] 8. Prepare optional hosting and its lifecycle
- [x] 8.1 Freeze the deployment environment and pilot configuration
  - _Outcome: Independent review APPROVED; 21 focused tests passed, Ruff/architecture/lock checks passed, documented plan reports zero mutations and inference. Account facts and live compatibility remain unverified._
  - Isolate and lock deployment tooling without adding Modal or accelerator dependencies to the core library. Resolve exact SDK, serving-image digest and model/tokenizer commits; verify artifact access and license prerequisites.
  - Freeze GPT-OSS-20B, one A100 40 GB, four physical CPU cores, 32 GiB host RAM, 4096-token context and one scheduled sequence. Record pricing timestamp and credential references without secret values.
  - Validate required fields, resource units and bounds. Exclude local models, credentials and environment files from uploads; separate remote cache staging from GPU allocation.
  - Done when a reproducible configuration passes offline validation and default planning performs zero cloud mutations or inference. No paid image build or staging is included in this task.
  - _Boundary: Manifest and deployment setup_
  - _Requirements: 7.1, 7.3, 7.4, 7.9, 7.10_

- [x] 8.2 Prepare authenticated serving and observed provenance
  - _Outcome: Independent review APPROVED; 30 combined deployment tests passed, Ruff/architecture passed, and real Modal 1.6.3 App construction succeeded with sockets disabled. Live GPU fit and inference evidence remain unverified._
  - Configure the optional serving recipe with proxy authentication, zero warm minimum, one maximum container, zero buffer, 60-second idle scale-down and one concurrent input.
  - Expose the existing compatible completion contract with genuine natural-log probabilities and at least five alternatives; preserve partial evidence semantics. Do not introduce a new provider transport.
  - Capture resolved artifact/runtime/settings and observed GPU information, with authenticated before/after retrieval bound to the deployment identity. Missing observations remain unknown.
  - Done when fixture-based checks verify authentication, resource controls, provenance and cache configuration without starting paid compute; core imports remain Modal-free.
  - _Boundary: Deployment recipe_
  - _Depends: 8.1_
  - _Requirements: 7.1, 7.2, 7.3, 7.5, 7.9, 7.10_

- [x] 8.3 Implement bounded cleanup and remote expiry
  - _Outcome: Independent review APPROVED; 49 deployment tests passed, focused Ruff/architecture passed and real SDK factories constructed offline. Runner must deploy controller first, persist receipt and enforce admission deadline. Live shutdown remains unverified._
  - Stop only the experiment-owned app on completion, cancellation or failure, allowing two minutes for stop verification and retaining the cache.
  - Arm an independent CPU-only expiry controller for the 20-minute GPU deadline before inference admission. Fail closed if expiry cannot be armed; local-client disappearance must not disable expiry.
  - Keep expiry/control observations separately bounded and costed; never call inference or restart GPU compute from expiry handling.
  - Done when offline tests cover client loss, cancellation, wrong-app protection and control-plane failure. Unverified cleanup records the app identifier and recovery command rather than claiming shutdown.
  - _Boundary: Deployment lifecycle_
  - _Depends: 8.2_
  - _Requirements: 7.4, 7.5, 7.6, 7.8_

- [x] 9. Integrate the bounded pilot with existing evidence handling
- [x] 9.1 Enforce the frozen pilot and evidence admission
  - _Outcome: Independent review APPROVED; 61 combined runner/deployment tests passed. Default plan independently verified zero inference/mutations, with account, tokenizer and cache prerequisites explicit. No live execution or calibration._
  - Reuse the existing completion adapter, call budget and validator. Reserve a new receipt exclusively and persist each attempt intent before sending; freeze prompts and their hash.
  - Permit one eligibility request and at most seven subsequent requests, zero retries, serial calls, 512 output tokens each, temperature 0, top_p 1 and five alternatives. Validate prompt plus output allowance against the 4096-token context.
  - Enforce 120 seconds per request and 20 minutes total GPU phase including up to ten minutes startup. Readiness observations must not generate tokens. Stop on cancellation, exhausted bounds or uncertain outcomes without replay.
  - Preserve reasoning/final-answer alignment as returned. Keep evidence eligibility separate from environment status; unknown or changed provenance prevents calibrated claims while retaining diagnostics.
  - Done when offline integration tests establish zero-mutation dry runs, the eight-call ceiling, intent durability, rejection of malformed/misaligned evidence and provenance drift, and cleanup on every exit path.
  - _Boundary: Pilot runner integration_
  - _Depends: 8.1, 8.2, 8.3_
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.10, 7.11_

- [x] 9.2 Report lifecycle outcomes and defensible costs
  - _Outcome: Independent review APPROVED; 76 focused accounting/runner/deployment/architecture tests passed. Unknown full costs remain null; billing snapshots preserve attempts and pricing. No live billing or paid execution._
  - Separate physical inference attempts from control/staging events, cold startup from generation, and usable evidence from HTTP success. Use idiomatic names carrying units; document token positions and table row grain.
  - Record estimated gross USD separately from observed USD and credits. Unknown cost and zero usable measurements produce unavailable cost per usable measurement, never zero cost by inference.
  - Validate the monthly budget record against prior usage and distinguish environment compute allowance from workspace-wide coverage; require an operator-confirmed allowance when accounting access is unavailable. Do not overwrite shared account policy.
  - Append later billing reconciliation without rewriting attempts. Publish neither credentials nor raw generated text; keep cleanup failure visible independently of inference success.
  - Done when accounting/redaction fixtures verify these distinctions and receipts retain sources, timestamps and enforcement caveats for the USD 5 target.
  - _Boundary: Pilot receipt and cost reporting_
  - _Depends: 9.1_
  - _Requirements: 6.4, 7.4, 7.6, 7.7, 7.8, 7.10_

- [x] 10. Validate and document the integrated optional provider
- [x] 10.1 Complete offline acceptance and operator instructions
  - _Outcome: Independent review APPROVED; 76 focused tests, strict docs, isolated SDK/package/chat-template smoke, locks and source hashes verified. Full regression: 2813 passed, one skipped, one obsolete dependency assertion subsequently corrected and verified. See modal-implementation-validation.md. Live pilot remains unperformed._
  - Run focused behavior tests, Ruff, architecture and lean-import checks; exercise the documented dry run and strict documentation build. Fake cloud calls; CI must never allocate paid GPUs.
  - Document account authentication, model access, budget setup, start/stop, retained cache charges, process-loss recovery and unverified cleanup. Link the optional host from local-provider guidance and MkDocs.
  - Cite dated primary sources and Jane Street; distinguish candidate support, observed evidence, calibration discrimination and mitigation efficacy. Keep frontier completion-only work separate.
  - Done when a source-matched validation report covers bounds, auth/redaction, evidence rejection, drift, cancellation, expiry and cost uncertainty, and the documented dry run succeeds without cloud mutation.
  - _Boundary: Integration validation and documentation_
  - _Depends: 9.2_
  - _Requirements: 6.3, 6.4, 6.5, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 7.10, 7.11_

- [x] 10.2 Validate the separately authorized live pilot
  - _Outcome: Independent review APPROVED the bounded negative feasibility outcome. Attempt 08 reached GPT-OSS-20B readiness after reviewed entrypoint and typing_extensions repairs, verified by two CPU-only checks. One eligibility response failed token_text_alignment_failed; zero measurement prompts followed. Configured A100-40GB, observed A100-SXM4-80GB; no 40 GB fit or attributable price claim. Runner verified model shutdown; final inventory confirmed all fifteen owned app entries stopped. Fresh checks: 153 passed, one optional skip; actual SDK 10 passed; Ruff, architecture and strict docs passed. No calibration/ensemble or mitigation measurement. See live-feasibility-validation-2026-11-06.md, docs/results/2026-11-06-modal-pilot.md and modal-alignment-diagnostic-proposal.md. Completion records feasibility, not eligible end-to-end scoring support._
  - Before any paid staging/build/deployment, present the concrete frozen pins, prompts, request/token/time caps, complete cost estimate and budget configuration for user approval. Verify credentials, account allowance and model access first.
  - After that approval, verify unauthorized requests are denied without starting a GPU, then run the eight-request, 20-minute GPU-phase pilot with the USD 5 target and independent expiry. No hardware/model substitution or additional calls on failure.
  - Verify actual GPU/runtime identity, usable five-alternative logprobs, cold/warm timings and app shutdown. Publish dated receipts, observed or explicitly unknown costs and comparison limitations.
  - Done when the authorized pilot has a dated, reviewed feasibility outcome and cleanup receipt, including honest ineligible outcomes. Missing approval or prerequisites leaves this task unchecked with a recorded reason; a blocked receipt is not completed live acceptance.
  - Calibration and single/ensemble measurements remain separately budgeted follow-up work; no automatic batch starts after eligibility.
  - _Boundary: Live integration acceptance_
  - _Depends: 10.1_
  - _Requirements: 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.11_

- [x] 10.3 Repair response-channel token mapping and retest
  - _Outcome: Independent review APPROVED. Explicit Harmony adapter derives final-answer source span from pinned control IDs; generic parsing remains unchanged. One authorized live eligibility request confirmed 30 full-generation positions and one final position. Repaired production parsing passed strict five-alternative validation on offline replay of the same response, preserving original logprobs and alternatives exactly; zero replay HTTP requests. Full regression: 2888 passed, two documented skips; focused 159 passed; Ruff, architecture and strict docs passed. All twelve currently observed owned app entries stopped. See live-feasibility-validation-2026-11-06.md and docs/results/2026-11-06-gpt-oss-response-mapping.md. No additional post-fix generation, calibration or mitigation claim._
  - _Approval: User explicitly requested amending the spec and retesting after agreeing this is an integration bug. Requirements/design amendment and this task are approved for that focused scope, including one additional eligibility POST; no calibration or ensemble batch._
  - Classify the prior mismatch as an integration defect under diagnosis, preserving its original receipt and distinguishing model capability from adapter handling.
  - Inspect the pinned provider contract; reproduce full-generation versus final-answer mapping with deterministic fixtures before editing the adapter. Cover reasoning/control channels, byte fragments, ambiguity, truncation and malformed evidence without matching separate generations.
  - Use explicit response channel/byte ownership to map original token probabilities and alternatives. Keep strict validation; never search for an answer substring, fabricate probabilities or retokenize the answer to force agreement.
  - Run at most one additional fixed-prompt Modal eligibility request, zero retries, unchanged settings and original time/resource bounds. Save only its bounded response body privately under ignored runs/ for replay, excluding headers/credentials/environment; publish structural outcomes only.
  - Verify fixed handling against that same live response through offline replay; label replay explicitly and do not claim another post-fix generation. Stop after the one request regardless of outcome; verify all owned compute stopped and record provisional costs.
  - Independently review focused regressions, the live/replay result and dated spec/docs amendments before completion.
  - _Boundary: Existing completion adapter, local evidence validator, bounded diagnostic and reporting_
  - _Depends: 10.2_
  - _Requirements: 6.4, 7.2, 7.3, 7.6, 7.7, 7.10, 7.12, 7.13, 7.14_
