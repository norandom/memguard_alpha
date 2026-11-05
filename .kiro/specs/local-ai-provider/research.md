# Candidate evidence and design questions

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
