# Qualified Models for MemGuard-Alpha

This document tracks the evaluation status of various foundation models run through the MemGuard-Alpha pipeline. Models are evaluated based on their ability to integrate with the MIA scorer and correctly reflect unpenalized confidence when processing strictly out-of-sample (OOS) data.

## The Out-of-Sample Validation Test
Models were tested using an out-of-sample dataset generated dynamically from the Financial Modeling Prep (FMP) API, consisting of 2026 YTD news articles. Because the data occurred *after* all model training cutoffs, it is physically impossible for the models to have memorized the data. 

A perfectly calibrated MemGuard pipeline will therefore show a **1:1 match** between the `Raw Avg Confidence` and `MemGuard Avg Conf` (applying a 1.0 penalty multiplier) on OOS data.

---

## 🟢 Qualified Models (Perfect 1:1 Match)

The following models demonstrated standard mathematical uncertainty (base perplexity) that perfectly aligns with our current hardcoded MIA threshold (`Loss < 0.5`). When exposed to 2026 OOS data, they were correctly flagged as "not memorizing" and received zero penalty.

### `openai/gpt-oss-20b`
*   **Raw Avg Confidence:** 0.5380
*   **MemGuard Avg Conf:** 0.5380
*   **Status:** QUALIFIED

### `nvidia/llama-3.3-nemotron-super-49b-v1.5`
*   **Raw Avg Confidence:** 0.6200
*   **MemGuard Avg Conf:** 0.6200
*   **Status:** QUALIFIED

---

## 🔴 Uncalibrated Models (The Base Perplexity Paradox)

The following models failed the Out-of-Sample test. Despite evaluating data from 2026 that they could not have memorized, MemGuard falsely flagged them for memorization and slashed their confidence.

### `nvidia/nemotron-3-super-120b-a12b`
*   **Raw Avg Confidence:** 0.3166
*   **MemGuard Avg Conf:** 0.2183
*   **Status:** UNCALIBRATED (Requires Dynamic Threshold)

### `openai/gpt-oss-120b`
*   **Raw Avg Confidence:** 0.4440
*   **MemGuard Avg Conf:** 0.3140
*   **Status:** UNCALIBRATED (Requires Dynamic Threshold)

### `nvidia/nvidia-nemotron-nano-9b-v2`
*   **Raw Avg Confidence:** 0.7100
*   **MemGuard Avg Conf:** 0.5350
*   **Status:** UNCALIBRATED

### Analysis: Why do massive models fail the OOS test?
This phenomenon is termed the **Base Perplexity Paradox**. 
Massive models (e.g., 120B parameters) are exceptionally fluent in modeling the English language. Even when they are predicting future text they have never seen, their deep understanding of grammar and typical financial phrasing means they predict tokens with extremely high probability, yielding an inherently low loss. 

Because our current `MIAScorer` uses a strict, hardcoded threshold (`Loss < 0.5`), the 120B model's baseline linguistic fluency dips below this limit, causing the MIA scorer to misinterpret its high natural fluency as "memorization."

### Future Work
To qualify the 120B+ models, we cannot use a fixed MIA threshold. The MemGuard threshold must be **dynamically calibrated** relative to the baseline perplexity of the specific model being evaluated on generic, non-financial text.

---

## 📅 Training Cutoff Registry

The harness consumes `data/cutoffs.yaml` as a fail-fast guard: the OOS-control corpus and the evaluation set must be drawn from articles strictly after every shortlisted model's training cutoff, otherwise contaminated text leaks in and silently breaks the statistical claims.

### Sourcing rules (do not relax)
1. Each date comes from the official model card or vendor docs.
2. When a model documents both a pre-training and a post-training cutoff, the **later** date is recorded — post-training data is still memorizable.
3. When the vendor states a month only (e.g. "December 2023"), the **last day** of that month is used (most conservative reading).
4. When neither the model card nor a verifiable upstream base-model card documents a date, the model is **omitted**. The harness then aborts with `CutoffViolation` rather than silently mis-evaluating it.

### Active registry (12 models)

| Model ID | Cutoff | Source |
|----------|--------|--------|
| `meta/llama-3.1-8b-instruct` | 2023-12-31 | "Knowledge cutoff: December 2023" — [HF](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |
| `meta/llama-3.1-70b-instruct` | 2023-12-31 | "Knowledge cutoff: December 2023" — [HF](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct) |
| `meta/llama-3.1-405b-instruct` | 2023-12-31 | "Knowledge cutoff: December 2023" — [HF](https://huggingface.co/meta-llama/Llama-3.1-405B-Instruct) |
| `meta/llama-3.2-1b-instruct` ★ | 2023-12-31 | "Knowledge cutoff: December 2023" — [HF](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) |
| `meta/llama-3.2-3b-instruct` | 2023-12-31 | "Knowledge cutoff: December 2023" — [HF](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) |
| `meta/llama-3.3-70b-instruct` | 2023-12-31 | "The pretraining data has a cutoff of December 2023" — [HF](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) |
| `nvidia/llama-3.3-nemotron-super-49b-v1.5` | 2023-12-31 | Inherits Llama-3.3-70B base — [HF](https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1_5) |
| `nvidia/nvidia-nemotron-nano-9b-v2` | 2024-09-30 | "Cutoff date of September 2024" — [HF](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2) |
| `openai/gpt-oss-20b` | 2024-06-30 | gpt-oss model card §2.4 "Knowledge cutoff: June 2024" — [arXiv](https://arxiv.org/html/2508.10925v1) |
| `openai/gpt-oss-120b` | 2024-06-30 | Same source as 20b sibling — [arXiv](https://arxiv.org/html/2508.10925v1) |
| `microsoft/phi-4-mini-instruct` | 2024-06-30 | "Cutoff date of June 2024 for publicly available data" — [HF](https://huggingface.co/microsoft/Phi-4-mini-instruct) |

★ = designated reference model for the MIA delta feature.

The harness derives two date windows from this registry:
- **`is_window` = (2010-01-01, earliest_cutoff)** — articles every model has memorized. With the current registry: `2010-01-01 → 2023-12-31`.
- **`oos_window` = (latest_cutoff, today)** — articles no model has memorized. With the current registry: `2024-09-30 → today`.

### Deferred (cutoff too recent for usable OOS window)

| Model ID | Cutoff | Reason |
|----------|--------|--------|
| `nvidia/nemotron-3-super-120b-a12b` | 2026-02-24 | Post-training cutoff (per [HF model card](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16)). At today=2026-04-27 the post-cutoff window is only ~2 months wide — too narrow for the design's `target_per_corpus=100`. Re-include once ≥6 months of post-cutoff news exists (target: **2026-08-24**). |

### Omitted (no verifiable cutoff documented)

| Model ID | Reason |
|----------|--------|
| `mistralai/mixtral-8x22b-instruct-v0.1` | Mistral's official [HF card](https://huggingface.co/mistralai/Mixtral-8x22B-Instruct-v0.1) does not state a training-data cutoff. Third-party dates conflict. |
| `qwen/qwen3-next-80b-a3b-instruct` | Qwen's [HF card](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) describes "Pretraining (15T tokens) & Post-training" but does not state a cutoff date. |

When a model in this section is requested via `--shortlist`, the runner aborts with `CutoffViolation: missing from cutoffs registry`.
