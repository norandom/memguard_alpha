# Five local models returned aligned logprobs

Date: 2026-11-06. Phi-4 Mini, both Llamas and Gemma passed token-evidence and
environment checks. LLaVA also returned valid token evidence, but its observed
4096-token context differed from the planned 2048, so that configuration remains
ineligible. GPT was explicitly excluded from this local run; GPT-OSS-20B is the
separate [Modal candidate](2026-11-06-modal-pilot.md).

## What you can use this result for

These observations establish a starting point for endpoint-based calibration
experiments through the existing OpenAI-compatible adapter. They do not establish
memorization detection quality or divergence decoding effectiveness. All five
models ran on CPU with Ollama 0.40.11; no GPU, hybrid or NPU execution was observed.

| Model | Token evidence | Environment | Request elapsed, seconds | Ollama model allocation, GiB |
| --- | --- | --- | ---: | ---: |
| `phi4-mini:3.8b` | Valid, five alternatives | Matched, context 2048 | 10.656 | 2.642 |
| `llama3.1:8b` | Valid, five alternatives | Matched, context 2048 | 13.610 | 4.949 |
| `llama3.2:3b` | Valid, five alternatives | Matched, context 2048 | 7.015 | 2.171 |
| `gemma3:4b` | Valid, five alternatives | Matched, context 2048 | 9.640 | 2.688 |
| `llava-llama3:latest` | Valid, five alternatives | Mismatched: 4096 versus 2048 | 13.312 | 5.222 |

Each row is one physical inference request, including possible cold loading and
transport overhead. Separate load and generation times are unavailable. Allocation
is Ollama's reported model allocation divided by 2^30, not measured total system
RAM or a minimum hardware requirement. Accelerator allocation was zero in every
post-request observation. The [numeric receipt](2026-11-06-local-model-smoke.json)
contains exact byte counts, digests, settings and the private source receipt hash.

## How the bounded test ran

The five-request plan used temperature 0, top_p 1, a 32-token output allowance,
`logprobs: true`, five alternatives, zero retries and a 300-second request limit.
The same short prompt was sent once per model. Installed digests were frozen
before inference and checked afterward. Every model was unloaded and the empty
loaded-model list verified before proceeding; the final list was also empty.
Exactly five inference requests completed, with no unknown outcomes.

The two missing Llama artifacts were installed under ignored `models/` after
verifying full blob sizes and SHA-256 hashes. Standard pulls failed; an explicitly
bounded ranged transfer preserved and verified completed chunks. Llama 3.1 used
one bounded resume. No server restart, system network setting, driver change or
additional inference was used to complete this run.

## What remains to measure

The evidence is partial top-k; score normalization semantics remain unverified.
One or two generated text tokens per probe are enough for this smoke check, not
for calibration, reliability estimates or model ranking. Resolve LLaVA's effective
context through explicit configuration before using it in a matched experiment.
Keep the exhausted five-call receipt unchanged and review a new measurement plan
before sending further requests.

Calibration, single-answer and ensemble measurements need separate datasets and
budgets. Divergence decoding additionally needs a validated statistical source
and generation-control capability; known tokenizers and observed logprobs alone
do not establish those capabilities. See
[Jane Street's method](https://blog.janestreet.com/mitigating-memorization-in-llms/)
and the [evaluation scope](../divergence-decoding-evaluation.md).

The [November 5 feasibility report](2026-11-05-local-provider-feasibility.md) and
[storage test](2026-11-06-local-model-storage.md) remain historical observations.
This fresh run broadens the non-GPT model evidence without changing those receipts.
