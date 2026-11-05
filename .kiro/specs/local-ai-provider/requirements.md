# Requirements: local AI provider with mandatory token evidence

## Project description

Recall Guard users need repeatable measurements against small, locally hosted
models when a hosted provider is unavailable. Extend the existing provider and
calibration workflow to local endpoints, including NPU execution. Prioritize
Gemma for the NPU candidate; retain GPT-OSS-20B in evaluation plans as previously
requested. Admit a configuration only after observing genuine log probabilities.

## Scope and boundaries

- Include local endpoint configuration, capability probes, CPU/GPU and NPU
  execution reporting, calibration, single-answer and optional ensemble evaluation.
- Treat the observed November 5 failure as **NIM unavailable** for those attempts.
  Preserve the HTTP 504 receipts; do not infer a missing key, permanent outage,
  or proven overload. Further NIM diagnosis is outside this spec.
- Opus and other frontier APIs without logprobs belong to the separate
  completion-only alternative path. This spec neither changes nor implements it.
- Local hosting is an optional provider choice, not a library installation
  prerequisite. Do not build a model estate, change BIOS/WSL memory settings,
  or require a GPU fleet.
- Token evidence does not itself establish generation control or successful
  divergence decoding. Those remain separate evaluation capabilities.

## Requirement 1: Explicit local model selection

1. When configuring a local provider, Recall Guard shall accept an explicit endpoint, model identity, generation settings and authentication choice, including an explicitly unauthenticated local endpoint.
2. When preparing a run, Recall Guard shall record model revision, quantization, tokenizer identity and runtime version when available, and mark unavailable fields unknown.
3. If a selected endpoint, model or parameter is unavailable, Recall Guard shall report the specific failure without silently substituting a hosted provider or another model.
4. When selecting local models for initial validation, the evaluation plan shall prioritize a small Gemma model for NPU execution and include GPT-OSS-20B as a separate candidate whose feasibility is measured independently.

## Requirement 2: Mandatory log probabilities

1. Before admitting a configuration to measurement, Recall Guard shall require an actual generation probe with nonempty output and valid, token-aligned generated-token log probabilities.
2. If token evidence is missing, malformed, censored or misaligned, Recall Guard shall mark the configuration ineligible for this path and retain the reason without falling back to completion-only scoring.
3. Where a selected scoring feature requires alternative-token probabilities, Recall Guard shall additionally require the evidence that feature needs before enabling it.
4. When accepting derived log probabilities, Recall Guard shall require full-vocabulary logits for the same prefix and record normalization, log base and whether scores precede or follow sampling transformations.
5. When receiving only a top-k subset, Recall Guard shall preserve its partial status and shall not renormalize it into a claimed full-vocabulary distribution.
6. If calibration or feature evidence is insufficient, Recall Guard shall report the affected score as unavailable rather than substitute ensemble agreement or invented confidence.

## Requirement 3: NPU and resource feasibility

1. When validating NPU execution, the evaluation shall require observed runtime evidence of NPU use and usable token probabilities from that same configuration before reporting NPU support verified.
2. If a runtime silently falls back to CPU or GPU, the evaluation shall report the observed execution mode and shall not count the run as a verified NPU result.
3. When reporting local execution, the evaluation shall distinguish CPU, GPU, CPU/GPU offload, NPU and mixed NPU execution, with unknown allocation explicitly labeled.
4. When measuring resource feasibility, the evaluation shall record available hardware, model footprint, context length, concurrency, load time, generation latency and memory observations where exposed.
5. When interpreting memory requirements, the evaluation shall distinguish dedicated GPU memory, shared system memory and total RAM without treating the reported 2 GB VRAM as a universal model-size limit.
6. When executing the initial hardware pilot, the evaluation shall use one loaded test model and one active inference request at a time.
7. If resource exhaustion or unsupported hardware prevents execution, the evaluation shall retain an explicit failed feasibility result without changing machine configuration automatically.

## Requirement 4: Existing scoring and calibration workflows

1. When using an admitted local endpoint, Recall Guard shall expose it through the existing provider, scoring and calibration workflow rather than require a separate user-facing scoring system.
2. When model revision, quantization, tokenizer, runtime or score semantics change, Recall Guard shall identify existing calibration as condition-mismatched until revalidated.
3. When comparing single answers with ensembles, the evaluation shall use frozen items and recorded generation settings and report draw counts, parsed counts and results separately.
4. When reporting a local result, the evaluation shall distinguish endpoint feasibility, contamination-score discrimination, calibration quality and measured mitigation effects.
5. If only generated-token or partial alternative evidence is available, the evaluation shall not claim exact full-vocabulary divergence decoding or intervention before token selection.

## Requirement 5: Bounded execution and reporting

1. Before a measurement batch, the evaluation shall define request, token, context and elapsed-time bounds covering probes, retries, calibration and ensemble draws.
2. When a user cancels or a configured bound is reached, Recall Guard shall stop admitting new work and retain completed, failed and unknown-outcome attempts without silent replay.
3. When reporting the November 5 NIM attempts, the evaluation shall use the operational status "NIM unavailable" while retaining the observed 504 status and unresolved cause in diagnostic detail.
4. When reporting local results, the evaluation shall identify what model, runtime, hardware and scoring stages actually ran, including missing capabilities and unperformed work.
5. When publishing artifacts, Recall Guard shall exclude credentials and distinguish unknown resource or cost information from zero.

## Requirement 6: Portable integration and understandable documentation

1. When installing Recall Guard without local hosting extras, users shall retain the existing lean API-and-calculation library without NPU or GPU runtime dependencies.
2. When running Recall Guard in its Docker tooling environment, users shall be able to configure a reachable external local-model endpoint without requiring the accelerator runtime inside that container.
3. When documenting setup, the project shall distinguish the optional model-host runtime from Recall Guard and describe verified support separately from candidate support.
4. The implementation shall use idiomatic domain names and document array axes, token positions, probability units, DataFrame row grain and alignment wherever these representations cross interfaces.
5. When documenting divergence-related use, the project shall cite [Jane Street's approach](https://blog.janestreet.com/mitigating-memorization-in-llms/) and distinguish this provider extension from the separate frontier completion-only path.

## Review status

Requirements review: self-reviewed against EARS, coverage and scope boundaries.
Design and tasks await requirements approval under the project's phased workflow.
No new installation, inference batch or hardware change is authorized by this draft.
