# Local model storage and sequential probes, 2026-11-06

The complete Ollama store was copied to `models/`: 28 files totaling
22,689,410,671 bytes. Every copied file matched its source SHA-256 hash.
Git, Docker and Dagger exclude this directory. The running host was started
with `OLLAMA_MODELS` pointing to it, a requested 2,048-token context, one loaded
model, one concurrent request and cloud features disabled.

The move is incomplete: the original `%USERPROFILE%\.ollama\models` copy remains.
Automatic approval review rejected removal of that directory and creation of a
compatibility junction, reporting only "blocked by policy". No junction or
user-level environment change was made. Use `scripts/start_local_ollama.ps1`
for future project-local launches; ordinary Ollama launches can still use the
original location until cleanup/configuration is completed.

## One probe per model

Each request used the existing OpenAI-compatible adapter, a 32-token output
limit, requested logprobs and five alternatives, no retries and a 300-second
ceiling. Each probe started with no model loaded. Models were unloaded between
tests, and no models remained loaded afterward. Three inference attempts ran.

| Model | Result | Request elapsed | Observed execution |
| --- | --- | ---: | --- |
| `gemma3:4b` | Eligible: aligned logprobs and five alternatives | 9.532 s | CPU, context 2,048 |
| `gpt-oss:20b` | Unavailable: HTTP 500; cause unresolved | 38.156 s | No loaded model observed afterward |
| `llava-llama3:latest` | Responded, but ineligible: effective context 4,096 differed from requested 2,048 | 13.219 s | CPU |

The GPT-OSS result differs from the November 5 alignment failure; this receipt
does not assign today's HTTP 500 to storage, memory or any other unverified cause.
LLaVA's final receipt records a context mismatch; it does not independently
establish whether its token evidence passed the earlier validation stage.

These short probes establish no calibration quality or mitigation effect. No
model was downloaded and no calibration or ensemble batch ran. Full observations
are in the [JSON receipt](2026-11-06-local-model-storage.json).
