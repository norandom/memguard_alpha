# Model cutoff registry

The harness reads `data/cutoffs.yaml` to know each model's training cutoff. That date controls which articles count as in-sample (the model has seen them) versus out-of-sample (it hasn't) for that model. Get a cutoff wrong and the labels are wrong, and every downstream statistical claim falls over.

This document is the human-readable companion. Every entry cites a source.

## Sourcing rules

Don't relax these:

1. Each cutoff comes from the official model card or vendor docs.
2. When a model documents both pre-training and post-training cutoffs, use the later one. Post-training data is still memorizable.
3. When the vendor only states a month ("December 2023"), use the last day of that month. It's the most conservative reading.
4. When neither the model card nor a verifiable upstream base-model card gives a date, the model is omitted. The harness then aborts with `CutoffViolation` if the model appears in `--shortlist`. Better fail-fast than silently wrong.

## Active registry (11 models)

| Model ID | Cutoff | Source |
|---|---|---|
| `meta/llama-3.1-8b-instruct` | 2023-12-31 | "Knowledge cutoff: December 2023" — [HF](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |
| `meta/llama-3.1-70b-instruct` | 2023-12-31 | "Knowledge cutoff: December 2023" — [HF](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct) |
| `meta/llama-3.1-405b-instruct` | 2023-12-31 | "Knowledge cutoff: December 2023" — [HF](https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct) |
| `meta/llama-3.2-1b-instruct` ★ | 2023-12-31 | "Knowledge cutoff: December 2023" — [HF](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) |
| `meta/llama-3.2-3b-instruct` | 2023-12-31 | "Knowledge cutoff: December 2023" — [HF](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) |
| `meta/llama-3.3-70b-instruct` | 2023-12-31 | "The pretraining data has a cutoff of December 2023" — [HF](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) |
| `nvidia/llama-3.3-nemotron-super-49b-v1.5` | 2023-12-31 | Inherits Llama-3.3-70B base — [HF](https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1_5) |
| `nvidia/nvidia-nemotron-nano-9b-v2` | 2024-09-30 | "Cutoff date of September 2024" — [HF](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2) |
| `openai/gpt-oss-20b` | 2024-06-30 | gpt-oss model card §2.4 "Knowledge cutoff: June 2024" — [arXiv](https://arxiv.org/html/2508.10925v1) |
| `openai/gpt-oss-120b` | 2024-06-30 | Same source as the 20b sibling — [arXiv](https://arxiv.org/html/2508.10925v1) |
| `microsoft/phi-4-mini-instruct` | 2024-06-30 | "Cutoff date of June 2024 for publicly available data" — [HF](https://huggingface.co/microsoft/Phi-4-mini-instruct) |

★ = the designated reference model for the MIA delta feature.

The harness derives two date windows from this registry:

- IS window: `2010-01-01 → earliest_cutoff` (currently `→ 2023-12-31`). Articles every active model has had a chance to memorize.
- OOS window: `latest_cutoff → today` (currently `2024-09-30 →`). Articles no active model has seen.

## Deferred (cutoff too recent for a usable OOS window)

| Model ID | Cutoff | Reason |
|---|---|---|
| `nvidia/nemotron-3-super-120b-a12b` | 2026-02-24 | Post-training cutoff per the [HF model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16). At today's date the OOS window would be ~2 months wide, too narrow for `target_per_corpus=100`. Re-include after about 2026-08-24. |

## Omitted (no verifiable cutoff)

| Model ID | Reason |
|---|---|
| `mistralai/mixtral-8x22b-instruct-v0.1` | Mistral's [HF card](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1) doesn't state a cutoff. Third-party dates conflict. |
| `qwen/qwen3-next-80b-a3b-instruct` | Qwen's [HF card](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) describes "Pretraining (15T tokens) & Post-training" but doesn't give a cutoff. |

When a deferred or omitted model is requested via `--shortlist`, the runner aborts with `CutoffViolation: missing from cutoffs registry`.
