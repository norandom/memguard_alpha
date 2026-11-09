# Local model storage and sequential probes, 2026-11-06

> Register: R11 · Status: feasibility · Track: Local

The complete Ollama store was copied to `models/`: 28 files totaling
22,689,410,671 bytes, and every copied file matched its source SHA-256 hash. The
move is incomplete, because the original copy remains. Of three sequential
probes, `gemma3:4b` was eligible, `gpt-oss:20b` returned HTTP 500 and
`llava-llama3:latest` was ineligible on a context mismatch.

## Question

Can the host run from a project-local Ollama store, and which models pass one
sequential probe each from it?

## Setup

Git, Docker and Dagger exclude the `models/` directory. The running host was
started with `OLLAMA_MODELS` pointing to it, a requested 2,048-token context, one
loaded model, one concurrent request and cloud features disabled.

The original `%USERPROFILE%\.ollama\models` copy remains. Automatic approval
review rejected removal of that directory and creation of a compatibility
junction, reporting only "blocked by policy". No junction or user-level
environment change was made. Use `scripts/start_local_ollama.ps1` for future
project-local launches; ordinary Ollama launches can still use the original
location until cleanup/configuration is completed.

Each request used the existing OpenAI-compatible adapter, a 32-token output
limit, requested logprobs and five alternatives, no retries and a 300-second
ceiling. Each probe started with no model loaded. Models were unloaded between
tests, and no models remained loaded afterward. Three inference attempts ran.

## Data

One probe per model. No model was downloaded and no calibration or ensemble
batch ran.

## Results

| Model | Result | Request elapsed | Observed execution |
| --- | --- | ---: | --- |
| `gemma3:4b` | Eligible: aligned logprobs and five alternatives | 9.532 s | CPU, context 2,048 |
| `gpt-oss:20b` | Unavailable: HTTP 500; cause unresolved | 38.156 s | No loaded model observed afterward |
| `llava-llama3:latest` | Responded, but ineligible: effective context 4,096 differed from requested 2,048 | 13.219 s | CPU |

## Limits

- The GPT-OSS result differs from the November 5 alignment failure; this receipt
  does not assign today's HTTP 500 to storage, memory or any other unverified cause.
- LLaVA's final receipt records a context mismatch; it does not independently
  establish whether its token evidence passed the earlier validation stage.
- These short probes establish no calibration quality or mitigation effect.

## Evidence

Full observations are in the [JSON receipt](2026-11-06-local-model-storage.json).
