# Local models with token evidence

Gemma 3 4B returned aligned token log probabilities through the same
`OpenAICompatibleClient` used for NIM. You can use that adapter with an optional
external model host, then calibrate and score through the existing scorer. Recall
Guard adds no model SDK, GPU/NPU runtime dependency or second scoring API.

The [November 5 feasibility report](results/2026-11-05-local-provider-feasibility.md)
establishes one CPU configuration, not general Ollama compatibility or a
memorization benefit. If a fresh response fails alignment or lacks required
alternatives, that configuration is ineligible even when it generates readable text.

## Start without a model host

For optional remote GPU hosting, use the [Modal operator guide](modal-provider.md).
It starts with a verified offline plan and separates paid setup from the scoring
library. The GPT-OSS-20B Modal configuration remains an unverified host candidate;
it does not inherit the local CPU result above.

From a development checkout with its locked environment installed, use PowerShell:

```powershell
uv run --no-sync python examples/local_provider.py --help
uv run --no-sync python examples/local_provider.py
uv run --no-sync python examples/local_provider.py --fixture
```

The default validates a synthetic plan and prints `dry_run`, 29 planned logical
requests and zero network requests. `--fixture` substitutes fixed HTTP responses
at the adapter boundary. The real provider parser, evidence validator,
`MemoryGuardedScorer.calibrate_client` and ensemble scorer run: expect `completed`,
29 fixture attempts and four scored held-out rows. Neither command contacts
Ollama. Synthetic separable scores demonstrate mechanics, not detection quality.
The fixture returns two alternatives per position; it does not reproduce the
five-alternative live Gemma observation.

The [example source](https://github.com/norandom/memguard_alpha/blob/main/examples/local_provider.py)
shows the complete path. These APIs belong to the development checkout, not the
published v0.5.0 release.

## Choose and configure the external host

Downloaded models live in the checkout's `models/` directory, excluded from Git,
Docker build contexts and Dagger uploads. The existing Ollama store was copied
and SHA-256 verified there on November 6, retaining its blobs and manifests.
The original copy remains pending cleanup; see the
[storage and sequential test receipt](results/2026-11-06-local-model-storage.md).
To start the optional
host from this location, quit any existing Ollama server, then use PowerShell:

```powershell
.\scripts\start_local_ollama.ps1
```

The launcher sets `OLLAMA_MODELS` for its process, resolves the directory from
the checkout, and uses a 2,048-token context with one loaded model and one active
request. Supply `-OllamaPath` if Ollama is installed elsewhere. It changes no
user-level environment setting.
See [Ollama's model storage configuration](https://docs.ollama.com/faq#where-are-models-stored).
Load and test one model at a time, then use `ollama stop MODEL` before the next.

| Configuration | Observed support on 2026-11-05 | Limit |
| --- | --- | --- |
| Ollama 0.40.11, `gemma3:4b`, Q4_K_M | Eligible: aligned realized-token logprobs and five alternatives; CPU | One short probe; score normalization semantics unknown |
| Ollama 0.40.11, `gpt-oss:20b`, MXFP4 | Responded on CPU; ineligible | Exact token-text alignment failed |
| AMD Gemma 3 4B NPU candidate | Prerequisites failed; no inference | Artifact access and release compatibility unresolved |
| GPU or mixed offload | Candidate only | No verified execution on this machine |

Install and operate the host separately from Recall Guard. Select the exact model
artifact, confirm its license and disk/memory requirements, and record its digest,
runtime version and effective context. A GPU is optional: both observed Ollama
calls used CPU. An installed accelerator or a requested device is not evidence
that inference ran on it. The NPU path has no installed verified runtime or built
serving bridge; see the dated report before considering that candidate.

The existing adapter accepts the full chat-completions URL, not just `/v1`:

```python
from recall_guard.core.completion import ModelConfig
from recall_guard.core.local_evidence import LocalModelIdentity
from recall_guard.core.providers import OpenAICompatibleClient
from recall_guard.core.request_control import CallBudget

config = ModelConfig(
    protocol="openai_chat",
    endpoint="http://localhost:11434/v1/chat/completions",
    model="gemma3:4b",
    parameters={"temperature": 0, "top_p": 1, "max_tokens": 32,
                "logprobs": True, "top_logprobs": 5},
    timeout_s=300,
    max_retries=0,
    local_identity=LocalModelIdentity(),  # explicit unknown provenance
)
client = OpenAICompatibleClient(config, CallBudget(1), allow_unauthenticated=True)
# Construction makes no request. Use the bounded workflow below for measurement.
```

Authentication is an explicit choice. Local unauthenticated use needs no NIM key;
never send that key to a local endpoint. A protected endpoint uses
`credential_ref="env:LOCAL_MODEL_API_KEY"` and manifest authentication
`"environment"`; keep the secret in the environment. There is no automatic
hosted fallback. Ollama documents a subset of the OpenAI wire API, so advertised
compatibility does not replace an evidence probe. See its
[compatibility reference](https://docs.ollama.com/api/openai-compatibility).

Use PowerShell for Windows hosting, probes and measurement. WSL is reserved for
Dagger CI. A Docker client must use an external host address reachable from its
container network; container `localhost` refers to the container itself. Set the
full endpoint in the manifest to the address provisioned for that network, and
verify reachability separately. Keep the accelerator runtime on its external
host. Offline CI needs no host, model download or accelerator installation.

## Prepare a frozen measurement

The example can write all three required JSON files into a new directory. This
gives you a runnable file-format reference without starting inference:

```powershell
uv run --no-sync python examples/local_provider.py --write-inputs runs/local-demo
uv run --no-sync python scripts/measure_local_models.py --plan runs/local-demo/plan.json --manifest runs/local-demo/manifest.json --corpus runs/local-demo/corpus.json --dry-run
```

`manifest.json` contains `config` (`ModelConfig.to_dict()`), `authentication` and
`requested_execution_mode`; protected hosts also need `credential_ref` at the
manifest root. It is a measurement configuration, distinct from the historical
candidate inventory. `corpus.json` has exactly three arrays: `is_memorized`,
`oos_control` and `held_out`. Each row has a unique `item_id`, a nonempty `prompt`
and positive `input_tokens`. A minimal row looks like this:

```json
{"item_id": "development-before-001", "prompt": "Your dated task prompt", "input_tokens": 87}
```

Replace the synthetic rows with an independently justified corpus and preserve a
held-out split. Prompt output must match the existing direction/confidence parser;
the fixture uses `Direction: 1` followed by `Confidence: 0.8`. Choose enough
development observations for fitting: the demo's eight rows per class and
`min_valid=4` are only a mechanics example.

`plan.json` freezes schema version 1, SHA-256 hashes of the exact manifest and
corpus file bytes, the prompt template with one `{prompt}` marker, selected
features (`mia`, `mcs`), draw count, seed, fitting thresholds and limits. After
editing either input file, recompute its hash deliberately:

```powershell
(Get-FileHash runs/local-demo/manifest.json -Algorithm SHA256).Hash.ToLowerInvariant()
(Get-FileHash runs/local-demo/corpus.json -Algorithm SHA256).Hash.ToLowerInvariant()
```

Copy those values into `manifest_sha256` and `corpus_sha256` in the plan. Do not
change the inputs after review. The demo's timeout, token counts and prompts are
synthetic; replace them with measured declarations and realistic bounds before
any live use. `input_tokens` counts the fully rendered prompt with the selected
tokenizer, including template overhead. It is caller-declared, not verification
of host context or actual usage.

The logical request count is
`1 + IS + 2 * OOS + held_out * (1 + ensemble_draws)`: the scorer probes once,
uses OOS for both baseline and fitting, then measures separate single and ensemble
arms. The physical cap includes all retries via
`logical_requests * (1 + max_retries)`. Reserve total tokens for every permitted
physical attempt using `max_requests * (max_input_tokens + max_output_tokens)`.
Input plus output must fit `context_tokens`; request timeout must fit the elapsed
bound. These are admission ceilings, not measured token usage or cost estimates.

Only after reviewing a real corpus, host configuration and complete budget, the
execution form is:

```powershell
uv run --no-sync python scripts/measure_local_models.py --plan runs/local-demo/plan.json --manifest runs/local-demo/manifest.json --corpus runs/local-demo/corpus.json --execute --journal runs/local-demo/new-run.json --observer ollama
```

This command performs inference; it is not part of the offline walkthrough. The
observer reads fresh host metadata before the probe, immediately after it and at
closing. It requires the selected loaded model and verified effective context
before sending corpus prompts. Every response must retain valid evidence and
identity. Sidecar observations cannot guarantee atomic identity. Unknown
provenance permits exploratory fitting within a run but does not establish that
a calibrator can be reused across runs. Changes to weights, quantization,
tokenizer/template, runtime, backend, context, sampling or score normalization
require renewed evidence and calibration compatibility checks.

Execution is serial. Cancellation stops later admissions; a timeout cannot prove
that generation already running on the host stopped. The journal records intent
before each physical request and refuses an existing path, even for a completed
run. Do not silently replay uncertain attempts. Reconcile their outcome before
preparing a new plan and journal.

## Read the evidence and its limits

Execution writes the journal plus `.report.json` and `.report.md` beside it. Rows
represent one item, arm and zero-based draw; joins use `item_sha256`, `arm` and
`draw_index`, never DataFrame row position. `physical_call_ids` joins logical rows
to attempts, including retries. Stage counts retain planned, attempted, completed,
parsed, usable, failed, unparsed and unattempted denominators. Completed means a
response was captured; failed can overlap completed. Paired comparisons include
only common usable item keys.

Token evidence is ordered by generated position, with natural-log probability
units. Token bytes must reconstruct response bytes exactly, or ordered token text
must reconstruct text when bytes are absent. Whitespace stripping and
retokenization cannot repair mismatches. Alternatives describe a partial top-k
set; the library never renormalizes them as a full vocabulary. Score semantics
remain unknown unless observed. A host exposing full logits would need axes
`[batch, position, vocabulary]` (or `[vocabulary]` for one step), aligned to the
prefix that selected each generated token; pruned logits alone are insufficient.

Report probabilities and AUC are dimensionless; `_bytes` and `_s` identify memory
and time units. Null means unknown, not zero. Model allocation is distinct from
dedicated GPU memory, shared GPU memory and system RAM usage. Reports omit raw
prompts, generated tokens and provider diagnostics by default. AUC describes
discrimination, not probability calibration or mitigation. The ensemble exposure
estimate is `p_memorized_point`, not consensus evidence `consensus.p_memorized`.

This provider extension supplies evidence to the existing scorer. It does not
intervene before token selection. Jane Street's
[divergence-decoding approach](https://blog.janestreet.com/mitigating-memorization-in-llms/)
steers logits during generation using two auxiliary models; returned partial
logprobs alone do not implement it. See the separate
[decoder contract](divergence-decoding.md) and
[frontier completion-only workflow](completion-api.md). Historical NIM attempts
remain labeled "NIM unavailable" with HTTP 504 and unknown cause; they do not
establish local-model limitations.
