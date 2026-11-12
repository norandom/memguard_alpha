# Local provider feasibility, 2026-11-05

> Register: R08 · Status: feasibility · Track: Local

Gemma 3 4B passed one local evidence probe through Recall Guard's existing
OpenAI-compatible adapter. GPT-OSS-20B also generated a response, but its token
text did not reconstruct that response exactly, so it was ineligible for scoring.
Both ran on CPU, and the NPU prerequisites failed. These two calls establish
endpoint feasibility only; no local calibration, held-out discrimination,
probability calibration or mitigation effect was measured.

## Question

Can local models served through Recall Guard's existing OpenAI-compatible adapter
supply eligible token evidence on this machine, and are the NPU prerequisites in
place?

## Setup

Both calls used `http://localhost:11434/v1/chat/completions`, no authentication,
`temperature=0`, `top_p=1`, `max_tokens=32`, `logprobs=true` and `top_logprobs=5`,
with zero automatic retries and a 300-second per-attempt ceiling. Both model
downloads completed before these calls.

The approved plan allowed up to six attempts across candidates, including two
reserved for NPU; only two were used and no NPU budget transferred. The receipt
also preserves two earlier zero-attempt preflight refusals for missing or
mismatched model digests. They are not extra physical generation calls.

The inventory records an AMD Ryzen AI 9 HX 370, approximately 32 GB RAM and a
Radeon 890M reporting 2 GiB AdapterRAM. That reported GPU field is not measured
dedicated usage or a universal model-size limit.

## Data

One evidence probe per candidate model, two physical generation calls in total.

## Results

| Field | Gemma | GPT-OSS |
| --- | --- | --- |
| Model | `gemma3:4b` | `gpt-oss:20b` |
| Quantization | Q4_K_M | MXFP4 |
| Runtime | Ollama 0.40.11 | Ollama 0.40.11 |
| Physical generation attempts | 1 | 1 |
| Admission | Eligible, aligned evidence | Ineligible, `token_text_alignment_failed` |
| Request elapsed | 8.547 s | 24.953 s |
| Reported prompt / completion tokens | 16 / 3 | 73 / 30 |
| Observed backend | CPU | CPU |
| Effective context | 2,048 tokens | 2,048 tokens |
| Host model allocation | 2,886,687,784 bytes | 13,902,492,466 bytes |
| Host accelerator allocation (`size_vram`) | 0 bytes | 0 bytes |

Gemma supplied realized-token logprobs and five alternatives. GPT-OSS's failure
was alignment, not a missing model or an inference that logprobs were unsupported.

NPU prerequisites failed. The selected AMD Gemma artifact page returned HTTP 401
during inventory. Its revision, footprint and artifact-specific license could not
be pinned. Runtime documentation had unresolved OGA version differences, and
compatibility of the installed driver with the selected release was unverified.
No verified runtime was installed, no NPU inference ran and no serving bridge was
built. The `logit-pruning` artifact name alone cannot establish full-distribution
scores.

This is a completed prerequisite assessment with a failed gate, not evidence
that the hardware cannot support a future compatible configuration. Reassessment
needs an accessible pinned artifact and a resolved runtime/driver combination.

## Limits

Host allocations above are not dedicated VRAM. Shared GPU usage, dedicated GPU
usage and system RAM consumed by these runs remain unknown.

There were no repeated cold/warm runs. Load time, generation-only time and token
rates were not established; request elapsed includes transport and host work.
No GPU or NPU execution was verified. Tokenizer identity and probability
normalization semantics remain unknown, and alternatives are partial top-k
evidence. A fresh probe can overturn eligibility for changed outputs or settings.

Local endpoint evidence does not establish the efficacy of
[Jane Street's generation-time divergence intervention](https://blog.janestreet.com/mitigating-memorization-in-llms/).

## Evidence

The committed source receipts preserve the original observations. The candidate
manifest and inventory describe the earlier inventory stage, including then
incomplete downloads; the later feasibility receipt governs the call results.

- [Ollama feasibility receipt](https://github.com/norandom/recall_guard/blob/main/.kiro/specs/local-ai-provider/ollama-feasibility-receipt.json)
- [Pinned candidate manifest](https://github.com/norandom/recall_guard/blob/main/.kiro/specs/local-ai-provider/candidate-manifest.json)
- [Machine inventory](https://github.com/norandom/recall_guard/blob/main/.kiro/specs/local-ai-provider/inventory-receipt.md)
- [NPU prerequisite receipt](https://github.com/norandom/recall_guard/blob/main/.kiro/specs/local-ai-provider/npu-setup-receipt.json)

Use the [local integration guide](../guides/local-provider.md) to run the offline example
and prepare a separate frozen corpus and budget. No larger live batch follows
from these results. Historical [NIM diagnostics](2026-11-05-nim-request-diagnostic.md)
remain "NIM unavailable" (HTTP 504, cause unknown). Opus and other frontier
completion-only experiments belong to a separate path.
