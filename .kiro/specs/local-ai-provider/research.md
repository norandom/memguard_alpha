# Candidate evidence and design questions

## Design discovery — 2026-11-05

Used `kiro-spec-design` integration-focused discovery, synthesis and review rules.
Inspected `completion.py`, `providers.py`, `provider_callback.py`,
`provider_probe.py`, `harness/scorer.py` and `harness/scorer_client.py`.
Reuse the existing HTTP client and scorer instead of adding a parallel provider
hierarchy. Extend calibration provenance because the existing identity does not
include model digest, tokenizer, quantization or runtime.

Fresh official checks:

- Ollama documents logprobs on its OpenAI-compatible chat endpoint:
  <https://docs.ollama.com/api/openai-compatibility>. This supports reuse of the
  current adapter but does not verify the installed runtime/model pair.
- AMD's current OGA page links Gemma to
  `amd/gemma-3-4b-npu-basic-text-logit-pruning_rai_1.9.0_npu`.
  The artifact's name makes full-vocabulary access a material verification gate.
  The model page itself could not be fetched during this review; no claim about
  its actual output tensors has been established.
- AMD's page says 1.9 models require matching runtime releases and contains both
  0.18.0 upgrade and 0.14.0 compatibility text. Do not select a package version
  from that contradiction; record the installed release and matched artifact in
  the feasibility receipt before build-out.

Build-versus-adopt decision: adopt a verified serving endpoint where possible.
Only if direct OGA supplies valid evidence and an existing server does not expose
it, build a small optional experimental bridge. A failed direct probe stops that
branch; it does not trigger a different model, CPU substitution or driver project.

Project environment decision: Windows commands use PowerShell. WSL is exclusively
for Dagger CI. No NPU runtime, model download or inference was performed for design.

Self-review covered every requirement, legacy calibration compatibility, special
token alignment, sidecar identity limitations and the distinction between requested
and observed NPU use. The gate passes for a feasibility-first design; it does not
certify NPU support or authorize an inference batch.

2026-11-05. These are design inputs, not local validation results.

- Ollama documents partial CPU/GPU loading and reports the split with `ollama ps`:
  <https://docs.ollama.com/faq>. One model at a time is our pilot policy, not an
  inherent Ollama limitation.
- AMD documents NPU execution and Gemma candidates through its Ryzen AI runtime:
  <https://ryzenai.docs.amd.com/en/main/hybrid_oga.html>.
  Initial candidate: Gemma-3-4b-it-mm-onnx-ryzenai-npu, using text inputs.
- ONNX Runtime GenAI documents `get_logits()`:
  <https://onnxruntime.ai/docs/genai/api/python.html>. This is a possible route to
  genuine normalized log probabilities, not proof that the chosen AMD build or
  serving endpoint exposes them correctly.
- Existing integration seams: `recall_guard/core/completion.py`, `providers.py`,
  `provider_probe.py`, and the existing readiness and calibration workflows.
- Design must choose and probe the smallest suitable adapter for Ollama and for
  the NPU runtime. OpenAI-compatible text generation alone is insufficient.
- Hardware context: Ryzen AI 9 HX 370, Radeon 890M, 32 GB installed system RAM,
  user-reported 2 GB VRAM. Actual runtime allocations and throughput remain to
  be measured. Do not promise GPU residency or NPU performance from model size.
- Preserve the existing GPT-OSS download independently; Gemma is the preferred
  additional small-model candidate, not an implicit replacement.

First feasibility gate: load the selected Gemma artifact, confirm NPU execution,
obtain aligned log probabilities, then measure a bounded short generation. If
that fails, record the precise unsupported capability before building an adapter.
