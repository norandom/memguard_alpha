# Divergence Decoding — Implementation Gap Analysis

Date: 2026-10-30  
Source baseline: `7a4c74625856534de57c1fe677078893150e7bef`  
Status: analysis complete; requirements remain unapproved; no implementation decision or effectiveness claim.

## Executive assessment

The guard's scoring and ensemble primitives are reusable, but there is no token-generation control surface. This is a new inference capability plus integration and evaluation work, not a small extension to the current reference-model feature. A hybrid approach is the strongest candidate for design: isolate optional generation machinery and extend existing scoring/provenance contracts.

Checkpoint availability and temporal validity remain unverified. No compatible three-model configuration or hardware/runtime combination has been established. Deep external dependency and checkpoint research is deferred to design, as prescribed by the gap-analysis workflow.

## Current implementation and constraints

- `recall_guard/core/nvidia_lm.py:64` defines token log probabilities; `:74` defines completed responses; `:154` sends a complete chat-generation request with top-20 candidate probabilities. It has no per-token intervention, model alignment, or auxiliary-model execution. Completion records omit finish reason, usage, revisions, and probability provenance.
- `recall_guard/harness/scorer.py:228` provides an injected LM factory, but its contract and mandatory credential check are NVIDIA-specific. A caller cannot express a complete divergence configuration through this seam today.
- `recall_guard/harness/scorer.py:504`, `recall_guard/mia/features.py:154`, and `recall_guard/mia/mcs.py:93` provide reusable feature extraction, standardization, probability prediction, and confidence discounting. Preserve parity with the batch evaluator.
- `recall_guard/core/ensemble.py` and scorer ensemble methods already provide draw execution, failure accounting, reductions, timestamps, and generation settings. Raw-response ensembles and guarded-score ensembles have distinct result surfaces.
- `recall_guard/core/manifest.py:41` records input hashes and run parameters. Its optional backtest extension demonstrates compatibility-preserving metadata additions.
- `recall_guard/core/loader.py:247` enforces model cutoffs against the evaluation header. `data/cutoffs.yaml` contains documented sourcing conventions but loads into a model-to-date dictionary, losing structured source evidence.
- `pyproject.toml` has a lean runtime and optional backtest/docs extras, but no local tensor inference stack. Steering requires downward-only layering, no DSPy, and a lightweight root import.

## Requirement-to-asset map

Criterion references use requirement.acceptance-criterion numbering.

| Requirements | Existing assets | Gap classification and implication |
|---|---|---|
| 1.1–1.4 | Optional ensemble pattern; packaging extras; lazy imports; scorer failure contracts | **Missing:** divergence opt-in configuration and experimental status. **Constraint:** absent configuration must preserve legacy behavior and installation footprint. |
| 2.1–2.2 | Cutoff registry and date loader | **Missing:** revisions, structured source evidence, ordered auxiliary cutoffs. **Unknown:** a suitable temporal model pair. Operator claims must not silently become verified cutoff evidence. |
| 2.3–2.6 | Factory injection and input validation | **Missing:** capability negotiation, token/prompt compatibility validation, alpha validation, explicit unsupported-mode failure. **Constraint:** a completion-only adapter cannot perform the algorithm. |
| 3.1–3.6 | Completed response/logprob types | **Missing:** synchronized prefixes, logits combination, normalization/sampling contract, zero-alpha equivalence, auxiliary failure handling and finish status. **Unknown:** runtime capabilities, numerical tolerance and context/token compatibility. |
| 4.1–4.3 | Control baseline and MCS training | **Missing:** calibration identity and mismatch enforcement across revisions, settings, feature contract and corpus. Existing calibration defaults can differ from ensemble generation overrides. |
| 4.4–4.7 | MIA features, MCS, weak-calibration flag, discount rule | **Constraint:** feature reuse does not settle probability semantics. **Missing:** distribution provenance and separation of score movement from mitigation evidence. |
| 5.1–5.3 | Single, multiple and ensemble scorer paths | **Missing:** common effective divergence configuration, calibration binding and mixed-identity rejection, including replay. |
| 5.4–5.5 | Failure accounting, raw ensemble component verdicts, deterministic reduction, exposure-vs-representative distinction | **Constraint:** preserve both ensemble surfaces without pretending they have identical schemas. **Missing:** identity-bearing steered evidence. |
| 6.1–6.2 | Model strings, observed temperature, ensemble timestamps | **Missing:** three-model/revision provenance, distribution transforms, effective mode and calibration identity. |
| 6.3–6.5 | Input hashes, manifest, text-content hash, request/time estimates | **Missing:** configuration/evidence identity, usage collection, explicit unavailable measurements and serving-revision tracking. Text hashes are not probability or configuration hashes. |
| 7.1–7.4 | Evaluation rows, cutoff loader, calibration split, runner | **Missing:** paired three-condition experiments, temporal groups across three cutoffs, matching/stratification, isolated tuning/calibration/final evaluation identities. **Constraint:** normal cutoff guard is not a historical-mitigation experiment protocol. |
| 7.5–7.8 | Accuracy/AUC statistics in evaluator; bootstrap utility; report writer | **Missing:** paired comparison uncertainty, utility-preservation verdict, confidence/validity/resource comparisons, confounding and exposure-proxy qualifications. Existing metrics alone cannot establish mitigation. |
| 8.1–8.4 | Documentation infrastructure and smoke tests | **Missing:** verified runnable triple, resource evidence, setup and limitations documentation. **Unknown:** whether accessible checkpoints satisfy the intended temporal experiment. |

## Important integration findings

### Calibration settings can drift

Calibration currently has no temperature or output-budget arguments and uses generation defaults. Ensemble scoring forwards per-call settings from EnsembleSpec. A divergence backend alone would not prevent scoring under a distribution different from calibration. Design must bind effective settings and reject mismatches without changing legacy default behavior.

### Probability semantics need an explicit contract

Min-K++ uses an unweighted mean and standard deviation of the supplied candidate log probabilities (`mia/features.py:131`). Supplying a full vocabulary rather than the existing top-candidate representation changes feature semantics even if the function is unchanged. Temperature, truncation and candidate selection must be documented and included in the calibration identity.

At greedy decoding, a point-mass sampling distribution is not interchangeable with model likelihoods used for MIA. Design must resolve which pre/post-transformation distribution is scored and avoid using a degenerate sampler distribution as an accidental substitute.

The optional reference is independently generated (`scorer.py:367`); ref_delta compares mean losses. It is not teacher-forced scoring of the primary continuation. Neither reference behavior nor either auxiliary should be described as equivalent to an aligned same-token likelihood reference.

### Historical evaluation needs a distinct protocol

The ordinary harness calls assert_cutoff_safe before evaluation. A historical pre-cutoff cohort with an appropriate historical header would be rejected for a later-trained main model. Omitting the header bypasses that date check but does not create a defensible experiment. Design should provide an explicitly identified research evaluation path with its own temporal validation while retaining the normal guard unchanged.

The registry uses the later pre/post-training cutoff and conservative month boundaries. Preserve those conventions when documenting exposure; a publication date or architecture-family name is not sufficient temporal training evidence.

### Replay evidence is richer than reply text

The existing canonical draw hash intentionally covers reply text only (`core/ensemble.py:280`). Preserve that contract and add a separate configuration/evidence identity. Identical text can have different probability evidence and calibration provenance.

Requirement 5.4 should be interpreted against each existing surface: component verdicts belong to raw-response ensemble machinery, whereas guarded-score ensembles expose a narrower result. Design must state this mapping rather than silently promise fields on both.

## Implementation approaches

### Option A — Extend existing client and scorer directly

Add optional decoding controls to existing generation/scoring types, with a backend strategy that delegates token execution to a controllable runtime.

- Reuses current entry points and minimizes public surface changes.
- Still requires new generation machinery; extra fields on the HTTP request cannot implement divergence decoding.
- Increases responsibility and dependency pressure on the NVIDIA client and risks credential, exception and default-behavior regressions.
- Viable only if orchestration is delegated; treating top-20 completion probabilities as steering inputs is not viable.

### Option B — Separate optional service/package and explicit adapter

Run three-model generation in an optional external runtime, returning probability evidence and provenance through a documented adapter.

- Isolates hardware dependencies and deployment requirements; keeps the default package lean.
- Adds service/version compatibility and deployment burden.
- Adapter must validate capabilities, distribution semantics and identities. A generic chat-compatible response alone is insufficient.
- Scoring still needs calibration/provenance extensions; a wholly separate scoring implementation would violate the reuse/parity objective.

### Option C — Hybrid extension with isolated optional backend

Introduce a provider-neutral generation/evidence contract, isolate the optional decoding runtime, and extend existing calibration, scoring, ensemble and manifest paths.

- Best candidate for design because it separates new token generation from established scoring semantics.
- Supports either an in-process optional runtime or a service adapter; placement remains a design decision.
- Requires deliberate compatibility work across calibration settings, results, errors and serialization.
- Suggested sequencing: establish checkpoint/runtime feasibility; define evidence and identity contracts; integrate generation and calibration; integrate ensembles/provenance; validate controlled evaluation.
- Rollback is omission of explicit opt-in configuration, not silent fallback within a requested steered run.

## Effort and risk

**Effort: XL (2+ weeks), provisional.** Three-model execution, configuration-bound calibration, provenance and a new experimental evaluation protocol span multiple layers. This is a complexity classification, not a delivery estimate; checkpoint and runtime research could dominate elapsed time.

**Risk: High.** The token-generation runtime and model pair are unverified, probability transformations can invalidate calibration, and temporal distribution shifts can mimic mitigation. Passing mocked algorithm tests would not resolve these risks.

## Research needed before design commitments

1. Identify accessible compatible checkpoints with immutable revisions and credible distinct cutoffs covering all training stages; document licenses and provenance. Do not infer compatibility from vocabulary size alone.
2. Verify runtime support for shared-prefix logits, model-specific prompt handling, caches, stop behavior, full-vocabulary adjustment and evidence extraction; determine actual resource requirements.
3. Specify alpha range, transformation ordering, candidate-set semantics, numerical stability and zero-alpha tolerance. Choose and justify main versus adjusted probabilities for features.
4. Specify the complete calibration identity, including any reference model and relevant feature/baseline hyperparameters, and its handling of per-call overrides.
5. Define a matched temporal dataset, leakage/deduplication controls, tuning/calibration/holdout separation, paired uncertainty procedure and predeclared utility/mitigation thresholds.
6. Define optional dependency support across supported Python/platform combinations and check architectural placement against the existing gate.
7. Resolve the cutoff-research evaluation boundary and ensemble-surface interpretation before translating them into tasks.

## Verification plan for later implementation

Reuse existing scorer/evaluator parity, NVIDIA error handling, manifest round-trip, packaging/import, architecture and ensemble replay tests. Add focused checks for the formula on controlled logits; shared-prefix behavior; zero-alpha equivalence; incompatible tokens and non-finite inputs; auxiliary failures; stop/truncation status; calibration mismatch rejection; mixed-identity replay; distribution provenance; and legacy defaults.

A real supported configuration additionally requires an end-to-end generation/calibration/scoring run and measured resources. Empirical mitigation requires the separate held-out evaluation protocol. Neither unit tests nor a successful runtime smoke test proves effectiveness.

## Analysis validation

Static source inspection and an independent read-only scoring/ensemble review informed this report. No runtime dependencies were installed, model calls made, or tests executed. Findings describe the inspected source rather than a demonstrated deployment. Requirements and approval metadata are unchanged. External runtime/version/checkpoint claims remain explicitly unverified design research.


---

## Design Discovery and Decisions — 2026-10-30

### Summary

Complex integration; full discovery. Requirements approved by the user's explicit design command with -y. Earlier gap-analysis approval statements are historical. Used the QOpenBB copies of kiro-spec-design and kiro-verify-completion because this checkout lacks local skills/templates; applied the design template, full-discovery, synthesis and review-gate rules.

### Research Log

- [Transformers Llama API](https://huggingface.co/docs/transformers/main/en/model_doc/llama): full-vocabulary forward logits and per-model cache interfaces make a manual synchronized decode loop plausible. Three-model composition is our design, not a claimed built-in feature.
- [Cache strategies](https://huggingface.co/docs/transformers/main/en/kv_cache): caches add memory and lifecycle decisions. Selected full-prefix execution first for correctness; cache optimization requires parity and identity changes.
- [Transformers installation](https://huggingface.co/docs/transformers/en/installation) and [PyTorch installation](https://pytorch.org/get-started/locally/): documented minimums include Python 3.12 in principle. Exact package pins and Python 3.14/device wheel support were not established. Require resolver and real installation checks before optional-runtime support claims.
- [Meta Llama 3 base card](https://huggingface.co/meta-llama/Meta-Llama-3-8B) reports March 2023 pretraining cutoff; [Llama 3.1 family card](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) reports December 2023. This suggests a base-checkpoint research candidate, not a verified temporal-only intervention. Tokenizer compatibility, pinned artifacts, full training provenance and licenses/access were not validated. Instruct post-training cutoff cannot be inferred from the family pretraining date.
- [Llama 3.2 card](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) also reports December 2023: 3.1/3.2 is not a distinct-cutoff pair on that evidence.
- Existing scorer inspection confirms mandatory NVIDIA credential behavior, a useful injected factory, and fixed calibration defaults. Keep the legacy constructor and add an explicit separate constructor with frozen generation settings.
- Existing architecture checker recognizes only core, mia, dataset, portfolio and harness. Place new runtime code within core and orchestration within harness; do not introduce an unchecked top-level runtime layer.

### Architecture Pattern Evaluation

Selected hybrid approach C. Direct NVIDIA client expansion cannot supply pre-sampling logits. An external service is possible but adds deployment/API scope not needed for the first implementation. A local optional adapter plus existing scoring is the smallest complete approach.

### Design Synthesis

Generalize the generation/evidence contract only enough to support existing and local paths. Adopt model forwarding and tensor operations from Transformers/PyTorch, and existing calibration/reduction code from recall_guard. Build only synchronization, configuration identity, evidence binding and the experimental comparison protocol.

Use one fixed adjusted-pre-temperature top-20 feature contract for v1 instead of offering multiple selectable statistical interpretations. Use existing four-feature calibration; do not reinterpret auxiliaries as reference models. Audited wrappers avoid adding fields to legacy public results. Keep the current text hash and add a separate evidence hash.

### Design Decisions

1. Local full-prefix runtime with strict identical tokenizer/template validation. Conservative support subset; no unverified compatibility mapping.
2. Fixed nonnegative finite alpha, greedy or full-vocabulary categorical sampling, adjusted pre-temperature probability evidence; combined-logit overflow is an error.
3. Calibration identities bind settings and feature semantics. Explicit ensemble overrides must match; baseline/training helpers use bound generation.
4. Separate historical experiment module preserves the ordinary cutoff guard. Paired temporal excess-accuracy comparisons are proxies with confound/utility gates, not causal proof.
5. Required preregistered numeric evaluation thresholds replace unjustified universal defaults. Empty or confounded groups cannot receive a positive verdict.
6. Real model support and empirical mitigation are separate gates. The first implementation prerequisite is a locked environment/checkpoint feasibility spike; no supported triple is promised now.

### Risks and Mitigations

Strict tokenizer equality may reject the candidate pair: report the unmet dependency, do not loosen alignment silently. Full-prefix inference is expensive: measure latency/memory and optimize only with parity evidence. Temporal training changes confound the proxy: report matching, baseline predictiveness and unresolved confounds. Reduced score may merely reflect altered logits: preserve independent calibration and never use score change alone for the verdict.

### Review Outcome

Main-context boundary/coverage/executability review plus an independent runtime-contract review found no unresolved requirements contradiction. Repairs clarified reference-option rejection, conservative tokenizer equality, combined-logit overflow and the non-causal evaluation interpretation. All 45 criteria are mapped in design.md. No models were downloaded, runtime installed, or inference/tests executed during document generation.



---

## Scope Revision — 2026-10-30

The user accepted narrowing the first milestone after design validation. Current requirements.md and design.md now specify only a numpy logits-combination function and a minimal greedy callback loop. Prior full-scope documents and their NO-GO review are preserved in archive/full-scope/. Earlier research, XL estimates and model-runtime decisions above apply to the archived scope, not the current MVP.

No new dependency or hosted/local model runtime is selected. Caller callbacks own model execution; vocabulary declarations are checked but actual tokenizer/cutoff authenticity is not certified. The Kiro requirements/design guidance was reused for the revision. All 18 new criteria map to concrete design contracts and tests. User agreement authorizes the scope reduction; approvals for the revised detailed documents remain pending. No tasks or implementation were generated.


Approval update: The user explicitly approved the revised requirements and design on 2026-10-30. Tasks remain ungenerated and implementation is not yet authorized through the task approval gate.

Implementation evidence (2026-10-30): narrowed MVP code/example implemented and independently reviewed in commits 9b44ce4, a4b2121, 68040c7, 7653491. Final acceptance remains pending upstream Windows test portability and Sentrux availability. Details: implementation-validation.md.

Windows/adoption update (2026-10-30): user removed Sentrux as a gate and authorized portability fixes. Native default Windows full suite now passes all 600 tests. Independent review approved. README and MkDocs use COBESY audience-specific structure and cite Jane Street. Current result: windows-adoption-validation.md.
