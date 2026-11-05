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

- [ ] 5. Integrate bounded local measurements
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

- [ ] 5.3 Produce honest local evaluation reports
  - Report planned/completed/parsed and failed/unparsed denominators, pair only common usable items and retain draw-level identity.
  - Separate AUC, calibration and mitigation claims; report dedicated/shared/system memory, load/generation timing and unknown values correctly.
  - Label the historical NIM attempts "NIM unavailable" while retaining 504 evidence; keep Opus outside this path. Name arrays/columns by role and units and join by item/arm/draw keys.
  - Done when fixture reports expose partial results, actual execution stages and all limitations without credentials or private prompts by default.
  - _Boundary: Calibration and measurement reporting_
  - _Requirements: 3.3, 3.4, 3.5, 4.3, 4.4, 4.5, 5.3, 5.4, 5.5, 6.4_

- [ ] 6. Deliver the integration and verify it
- [ ] 6.1 Add the runnable local-provider example and technical guidance
  - Demonstrate explicit endpoint/model/logprob settings through existing APIs, with PowerShell commands and no mandatory accelerator imports.
  - Update README and MkDocs with candidate-versus-verified capability tables, dated result links, external-host reachability from Docker CI and Jane Street attribution; keep frontier completion-only work distinct.
  - Done when the example passes against fixtures and documentation clearly describes optional hosting, setup prerequisites, evidence limitations and the existing adapter reuse.
  - _Boundary: Local example and documentation integration_
  - _Depends: 5.3_
  - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5_

- [ ] 6.2 Verify integrated behavior and publish the evidence summary
  - Run focused tests, Ruff and architecture checks through PowerShell; run one coherent offline regression and strict docs build through Dagger CI.
  - Verify lean imports, fake endpoint reachability, legacy behavior, evidence rejection, identity drift and cancellation. Include conditional skip reasons.
  - Done when a source-matched summary distinguishes passing automated checks from live feasibility and unperformed measurement. No larger live batch or release tag is implied.
  - _Boundary: Integration validation_
  - _Requirements: 1.3, 2.2, 4.1, 4.2, 5.2, 6.1, 6.2_

Technical documentation is included because it is explicitly required by this
project's accepted requirements, superseding the skill's generic code-only guideline.
