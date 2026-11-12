# Local models with token evidence

Gemma 3 4B returned aligned token log probabilities through the same
`OpenAICompatibleClient` used for NIM. You can use that adapter with an optional
external model host, then calibrate and score through the existing scorer. Recall
Guard adds no model SDK, GPU/NPU runtime dependency or second scoring API.

The [November 5 feasibility report](../results/2026-11-05-local-provider-feasibility.md)
establishes one CPU configuration, not general Ollama compatibility or a
memorization benefit. If a fresh response fails alignment or lacks required
alternatives, that configuration is ineligible even when it generates readable text.

The [November 6 five-model smoke](../results/2026-11-06-local-model-smoke.md) observed
aligned logprobs and five alternatives from Phi-4 Mini, Llama 3.1 8B, Llama 3.2 3B,
Gemma 3 4B and LLaVA on CPU. The first four matched the planned environment;
LLaVA's context differed. GPT was excluded locally. These are feasibility results,
with no calibration or mitigation measurement.

## Choose a provider profile

Each backend is one profile shipped with the package. `connect(profile_id, budget)`
returns the matching client; `preflight_host(profile_id)` runs the same checks
on their own and returns a report.

| Profile | Host | Free RAM required |
| --- | --- | ---: |
| `ollama-phi4-mini`, `ollama-llama3.2-3b`, `ollama-gemma3-4b` | Ollama, local | 6 GiB |
| `ollama-llava-llama3-ctx2048` (derived tag with `num_ctx 2048`; eligibility not yet probed) | Ollama, local | 6 GiB |
| `ollama-llama3.1-8b` | Ollama, local | 8 GiB |
| `ollama-gpt-oss-20b` | Ollama, local | 20 GiB |
| `modal-gpt-oss-20b` | Modal vLLM, Harmony final-answer mapping | remote |
| `nim-llama3.1-70b` | NVIDIA NIM | remote |

Before a local model is used, the preflight reads free RAM (only when the
endpoint is on this machine and the check does not run in a container), then
Ollama's read-only `/api/version`, `/api/tags` and `/api/ps`. It stops before any
inference when:

- free RAM is below the profile's requirement and the model is not already loaded;
- the endpoint is on another host, or the check runs in a container, and no fresh
  memory reading from the model host was given;
- the model tag is not installed (it suggests `ollama pull`; nothing is pulled);
- a credential or the pinned Harmony tokenizer is missing.

Each refusal names what was measured, what is needed and which profiles to try
instead. `scripts/probe_local_model.py --profile ...` and
`scripts/smoke_local_models.py --profile ...` show it as a panel and exit with
status 2. To see the refusal and the fallback without any host, keys or network,
run this in the development container (see the [Docker guide](development.md)):

```text
docker compose run --rm dev uv run --no-sync python examples/adopter_flow.py --fixture
```

`ollama-gpt-oss-20b` uses the generic mapping and was ineligible on Ollama 0.40.11
(token-text alignment failed); a mapping for Ollama's Harmony output is still open.
Use `modal-gpt-oss-20b` for GPT-OSS-20B evidence until then.

When a provider returns an error, the receipt keeps a bounded, redacted copy of
the response body, so an HTTP 500 can be diagnosed afterwards.

## Start without a model host

For optional remote GPU hosting, use the [Modal operator guide](modal-provider.md).
It starts with a verified offline plan and separates paid setup from the scoring
library. The GPT-OSS-20B Modal pilot reached readiness; its response exposed an
adapter mapping bug. The [follow-up diagnostic](../results/2026-11-06-gpt-oss-response-mapping.md)
records the explicit Harmony profile and same-response replay. It does not inherit
the local CPU result or establish calibrated scoring quality.

From a development checkout, build the image once with `docker compose build dev`,
then run:

```text
docker compose run --rm dev uv run --no-sync python examples/local_provider.py --help
docker compose run --rm dev uv run --no-sync python examples/local_provider.py
docker compose run --rm dev uv run --no-sync python examples/local_provider.py --fixture
```

On Windows these run only in the container. On Linux and macOS, the same
`uv run --no-sync ...` commands also work directly in the checkout.

The default validates a synthetic plan and prints `dry_run`, 29 planned logical
requests and zero network requests. `--fixture` substitutes fixed HTTP responses
at the adapter boundary. The real provider parser, evidence validator,
`MemoryGuardedScorer.calibrate_client` and ensemble scorer run: expect `completed`,
29 fixture attempts and four scored held-out rows. Neither command contacts
Ollama. Synthetic separable scores demonstrate mechanics, not detection quality.
The fixture returns two alternatives per position; it does not reproduce the
five-alternative live Gemma observation.

The [example source](https://github.com/norandom/recall_guard/blob/main/examples/local_provider.py)
shows the complete path. These APIs belong to the development checkout, not the
published v0.5.0 release.

## Choose and configure the external host

Downloaded models live in the checkout's `models/` directory, excluded from Git,
Docker build contexts and Dagger uploads. The existing Ollama store was copied
and SHA-256 verified there on November 6, retaining its blobs and manifests.
The original copy is still pending cleanup; see the
[storage and sequential test receipt](../results/2026-11-06-local-model-storage.md).
To start the optional
host from this location, quit any existing Ollama server, then use PowerShell.
This launches the Ollama model host natively on Windows; Recall Guard itself
still runs in a container:

```powershell
.\scripts\hosts\start_local_ollama.ps1
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

On 2026-11-08, with `OLLAMA_IGPU_ENABLE=1`, a load-only request (no prompt) placed
`gemma3:4b` fully on the integrated Radeon 890M through ROCm gfx1150: 35 of 35
layers offloaded, `/api/ps` `size_vram` equal to `size` (2.9 GB). No inference ran
on the GPU yet, so the table above is unchanged. The iGPU shares system RAM, so
the free-RAM gate still applies.

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
`"environment"`; store the secret in the root `.env` and pass only that variable
to the container by name, for example
`docker compose run --rm -e LOCAL_MODEL_API_KEY dev ...`. Compose reads its value
from the root `.env`. There is no automatic
hosted fallback. Ollama documents a subset of the OpenAI wire API, so advertised
compatibility does not replace an evidence probe. See its
[compatibility reference](https://docs.ollama.com/api/openai-compatibility).

On Windows, run probes and measurements in the container; PowerShell only
launches Docker or a native Ollama host and takes Windows memory snapshots.
Container `localhost` refers to the container itself unless it shares the host
network. Set the full endpoint in the manifest to an address reachable from the
container network, and verify reachability separately. Keep the accelerator
runtime on its external host. Offline CI needs no host, model download or
accelerator installation.

### Local Ollama from a container

Inside a container, the RAM gate never reads the container's own memory. It needs
a snapshot taken on the model host less than 10 minutes before the run.

**Recommended on Windows: native Ollama behind the authenticated gateway.**
Ollama runs on Windows to use the GPU; Recall Guard runs in the dev container
under Docker Engine in WSL (NAT networking). `scripts/hosts/serve_local_ollama.ps1`
(PowerShell 7, no admin) starts Ollama like `start_local_ollama.ps1` plus
`OLLAMA_IGPU_ENABLE=1` (use `-NoIgpu` to turn the integrated GPU off), bound to
`127.0.0.1:11434` only. It also starts a small gateway on the Windows address of
the `vEthernet (WSL...)` adapter, port 11435, never on `0.0.0.0` or a LAN adapter.
The gateway requires `Authorization: Bearer` with `LOCAL_MODEL_API_KEY` from the
root `.env` (generated there on first start if missing; the value is never
printed). It returns 401 otherwise, strips the header, and forwards only
`/v1/chat/completions`, `/api/version`, `/api/tags`, `/api/ps`, `/api/show` and an
unload-only `/api/generate` (`{"model": ..., "keep_alive": 0}`). After 30 minutes
without a gateway request (`-IdleMinutes`), it stops itself and Ollama. State and
logs are in `runs/ollama-gateway/`.

In PowerShell on Windows, in the checkout:

```powershell
pwsh -File scripts/hosts/serve_local_ollama.ps1 start
pwsh -File scripts/host_memory_snapshot.ps1
```

In a WSL shell in the checkout (the Windows address is the WSL default gateway):

```sh
GW=$(ip route show default | awk '{print $3}')
docker compose run --rm -e LOCAL_MODEL_API_KEY dev uv run --no-sync python scripts/smoke_local_models.py --profile ollama-gemma3-4b --host-snapshot /outputs/host-memory.json --endpoint "http://$GW:11435/v1/chat/completions" --credential-ref env:LOCAL_MODEL_API_KEY
```

Then, in PowerShell:

```powershell
pwsh -File scripts/hosts/serve_local_ollama.ps1 stop
```

`stop` also stops the model runner and fails unless no process remains and
nothing listens on 11434 or 11435; `status` shows both. `--credential-ref` sends
the token on metadata, inference and unload requests. A measurement manifest uses
the same endpoint with `"credential_ref": "env:LOCAL_MODEL_API_KEY"` and
authentication `"environment"`. On the development host the container reached the
gateway without a firewall change. If Windows Firewall blocks it on your machine,
run once in an elevated PowerShell (check the alias with `Get-NetAdapter`):

```powershell
New-NetFirewallRule -DisplayName "Recall Guard Ollama gateway (WSL)" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 11435 -LocalAddress 172.17.32.1 -InterfaceAlias "vEthernet (WSL (Hyper-V firewall))"
```

**Alternative: Ollama in a container in WSL (CPU).** Run Docker Engine in
WSL and start the optional `ollama` Compose service (`ollama/ollama:0.40.11`,
CPU). It serves the checkout's `models/` store read-only with the same settings
as `scripts/hosts/start_local_ollama.ps1`, never pulls, and publishes no port. Only
containers on the internal `model-host` network reach it, as
`http://ollama:11434`; Windows, the LAN and the internet cannot. From a WSL shell
in the checkout:

```sh
docker compose --profile ollama up -d ollama
sh scripts/host_memory_snapshot.sh
docker compose run --rm dev uv run --no-sync python scripts/smoke_local_models.py --profile ollama-gemma3-4b --host-snapshot /outputs/host-memory.json --endpoint http://ollama:11434/v1/chat/completions
docker compose --profile ollama down
```

The smoke without `--execute` is metadata-only and sends no inference. The
snapshot reads `/proc/meminfo` of the WSL VM, which is the memory the Ollama
container uses; it records host `NAME WSL VM` and source `wsl_vm_snapshot` in
`runs/container/host-memory.json` (the container's `/outputs/host-memory.json`).
WSL limits that VM (by default to half of Windows RAM). If the preflight reports
too little free memory, raise `memory=` under `[wsl2]` in `%UserProfile%\.wslconfig`
and run `wsl --shutdown`. `docker compose run --rm dev` keeps working without the
profile.

**Other Windows Ollama setups.** These also keep the Windows GPU runtime, without
the gateway. Take the snapshot on Windows instead:

```powershell
pwsh -File scripts/host_memory_snapshot.ps1
```

These options are documented, not tested end to end on the development host.

| Setup | Endpoint | Exposure |
| --- | --- | --- |
| Docker Engine in WSL with [mirrored networking](https://learn.microsoft.com/windows/wsl/networking#mirrored-mode-networking) and `--network host` | `http://localhost:11434/v1/chat/completions` | Ollama stays loopback-only |
| Docker Desktop | `http://host.docker.internal:11434/v1/chat/completions` | Ollama stays loopback-only |
| Docker Engine in WSL, NAT mode (WSL default), without the gateway | `http://GATEWAY_IP:11434/v1/chat/completions`; Windows is the WSL default gateway | Ollama must listen beyond loopback, unauthenticated |

With Docker Desktop:

```text
docker compose run --rm dev uv run --no-sync python scripts/smoke_local_models.py --profile ollama-gemma3-4b --host-snapshot /outputs/host-memory.json --endpoint http://host.docker.internal:11434/v1/chat/completions
```

`docker compose run` has no host-network option, so for mirrored networking use
`docker run` on the image Compose built, from a WSL shell in the checkout:

```sh
docker run --rm --network host -v "$PWD/runs/container:/outputs" recall-guard:dev uv run --no-sync python scripts/smoke_local_models.py --profile ollama-gemma3-4b --host-snapshot /outputs/host-memory.json --endpoint http://localhost:11434/v1/chat/completions
```

In NAT mode prefer the authenticated gateway above. Without it, Ollama reads its
listen address from `OLLAMA_HOST`. Bind it only to the Windows address of the
WSL interface (the gateway address), never to `0.0.0.0`, and allow port 11434
only on that interface with a Windows firewall rule. Ollama has no
authentication, so anything that can reach the port can use your models. Check
the interface name with `Get-NetAdapter` before adding the rule.

## Prepare a frozen measurement

The example can write all three required JSON files into a new directory. This
gives you a runnable file-format reference without starting inference:

```text
docker compose run --rm dev uv run --no-sync python examples/local_provider.py --write-inputs /outputs/local-demo
docker compose run --rm dev uv run --no-sync python scripts/measure_local_models.py --plan /outputs/local-demo/plan.json --manifest /outputs/local-demo/manifest.json --corpus /outputs/local-demo/corpus.json --dry-run
```

The files appear under `runs/container/local-demo` on the host.

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
(Get-FileHash runs/container/local-demo/manifest.json -Algorithm SHA256).Hash.ToLowerInvariant()
(Get-FileHash runs/container/local-demo/corpus.json -Algorithm SHA256).Hash.ToLowerInvariant()
```

In a Linux or macOS shell, `sha256sum` (or `shasum -a 256`) on the same files
gives the same values.

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

```text
docker compose run --rm dev uv run --no-sync python scripts/measure_local_models.py --plan /outputs/local-demo/plan.json --manifest /outputs/local-demo/manifest.json --corpus /outputs/local-demo/corpus.json --execute --journal /outputs/local-demo/new-run.json --observer ollama
```

The manifest endpoint must be reachable from the container; see
[Local Ollama from a container](#local-ollama-from-a-container) for the network
setup.

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

Behind the authenticated gateway, add `--host-snapshot /outputs/...json` (a fresh
`host_memory_snapshot.ps1` file). The shared host preflight (free RAM, model
installed, gateway reachable) then runs before the journal is created, and a
refusal names only its reason code. The observer authenticates its metadata reads
with the plan's `credential_ref`.

A corpus row may carry `"label": 1` (memorized) or `0` (out-of-sample). Labels on
`is_memorized`/`oos_control` must match the split; labeled `held_out` rows let the
report compute held-out ROC-AUC of the single answer's feature-based
`feature_p_memorized` (MIA features and the fitted calibrator, as in calibration,
whether or not the answer parses; parse success stays in `parsed`) with a seeded
(plan `seed`) 1,000-resample stratified percentile bootstrap 95% interval
(`held_out_evaluation`), next to the calibrator's internal holdout AUC. `splits`
gives planned, attempted, valid and failed counts per corpus split. Unlabeled
plans load and export as before.

### Release-1.0 Phase A

`scripts/build_phase_a_inputs.py` writes `runs/phase-a/<profile>/` for the five
local profiles and `modal-gpt-oss-20b` from `data/edgar/` (321 logical requests
each, no ensemble draws) plus `runs/phase-a/index.json` with file hashes. The
plan hash covers the manifest endpoint, so the local endpoint is frozen at build
time as the gateway URL (`--local-endpoint http://<WSL gateway>:11435/v1/chat/completions`).
Ollama input tokens are declared as a hard upper bound (UTF-8 bytes plus 128
template tokens); Modal counts are exact with the pinned tokenizer, so build the
Modal profile in the operator environment. `scripts/run_phase_a_local.ps1`
(`-DryRun` first) starts the gateway, runs each local profile in the dev container
with a fresh snapshot, unloads the model, and stops the gateway in a `finally`
block. It refuses to start when outputs under `runs/container/phase-a/` already
exist or when the gateway address no longer matches the frozen endpoint.

Phase A reports measure how well calibrated the held-out single-answer estimate
`feature_p_memorized` is, next to its held-out AUC. `held_out_evaluation` adds the
Brier score (`brier`) with the same seeded 1,000-resample stratified bootstrap 95%
interval (`brier_ci_low`, `brier_ci_high`), the no-skill reference that predicts
the held-out base rate (`brier_reference`), expected calibration error (`ece`) over
10 equal-width bins, the non-empty `reliability_bins` (count, mean predicted,
observed memorized fraction), and calibration-in-the-large (`mean_predicted`
against `base_rate`). These need at least 2 usable items per class; otherwise they
are null and `calibration_reason` says why. `claims.calibration_quality` repeats the
headline numbers, or stays `not_measured` with `claims.calibration_quality_reason`.

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
[decoder contract](../research/divergence-decoding.md) and
[frontier completion-only workflow](completion-api.md). Historical NIM attempts
remain labeled "NIM unavailable" with HTTP 504 and unknown cause; they do not
establish local-model limitations.
