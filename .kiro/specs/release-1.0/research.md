# Release 1.0 review: providers, methods, ecosystem leverage and adopter UX

Reviewed 2026-11-08 against working-tree source at HEAD `738fbb6`, including the
uncommitted release consolidation. This is the gap analysis and proposed revision
feeding [design.md](design.md), not an implementation approval or a second
progress tracker. Authoritative task status remains in [tasks.md](tasks.md). No
provider requests, model downloads, deployments or paid resources were started.

## Assessment

**Ready for design review (`M1.1`) with four targeted refinements to the 1.0 plan:**

1. **Separate hosting from method, and unblock research early.** The requirements'
   Goal section mixes completion-only methods with Ollama and Modal hosts while
   leaving NIM without a track. Roadmap Track 1 calls completion-only work
   divergence/recall culling, while `M5` actually absorbs a rewriting pilot and
   `M6` omits the Anthropic sample-correction route from
   `divergence-decoding-evaluation` revision 2. Keep `M1` focused on the provider
   profile seam, host preflight and friendly failure presentation so `M2.1` (the
   single live Modal eligibility request) and `M5` (Track 1 completion pilot) are
   not delayed by downstream reporting refactors.
2. **Use the Python ecosystem instead of reinventing plumbing.** Leverage
   already-installed `rich` for adopter preflight/error panels and progress,
   `ruff` (`C901`, `PLR0912`, `PLR0915`) instead of a custom AST/tokenizer line
   counter, `psutil` plus Ollama's HTTP `/api/ps` and `/api/tags` endpoints for
   cross-platform RAM/model preflight, and optional `polars` for validated batch
   reporting after the provider seam lands.
3. **Keep both native `uv run` and Docker Compose first-class.** Local Ollama
   adopters on Windows or macOS (Metal) benefit from direct `uv run` against
   `localhost:11434` and local `psutil` RAM inspection without container-to-host
   networking hurdles (`host.docker.internal` vs WSL2 gateway). Keep Docker
   Compose and Dagger as the reproducible Linux/CI/notebook environment rather
   than breaking native local CLI workflows.
4. **Make the adopter path progressive (`plan/doctor -> probe -> calibrate/score`).**
   Ship curated declarative profiles under `configs/providers/`, automate
   SHA-256 plan hashing instead of requiring manual PowerShell `Get-FileHash`
   copy-paste, and expose `MemoryGuardedScorer.from_profile(...)` alongside
   human-readable failure panels that preserve sanitized error bodies.

### Status and traceability

- `spec.json`: `design-generated` (updated for `M1.1` review), role `tracker`,
  current `M0.1`; requirements, design and tasks await user approval;
  implementation readiness `false`. **20 executable checkboxes, 0 complete.**
- `M0.1` must verify task evidence rather than setting phases from parent boxes
  alone. In particular, `divergence-decoding` is implemented at the mathematical
  layer (`core/divergence*.py`), whereas endpoint integration and empirical
  mitigation efficacy remain experimental (`M6`).
- Parallel tracks: Track 1 (`M5`) and offline preparation do not depend on live
  Modal (`M2`) or NIM availability. When a live gate awaits authorization or an
  external endpoint, the single `Current` pointer may be explicitly moved to an
  independent offline or Track 1 task without losing blocker state.

---

## 1. Product model: three hosting modes, independent methods

The primary token-evidence hosting choice consists of three modes behind one
`CompletionClient` protocol:

| Hosting mode | User-facing purpose | Model choice | Current evidence |
| --- | --- | --- | --- |
| **Ollama — local host** | Local data handling; use available local CPU/GPU/Metal RAM | Curated small models first (`gemma3:4b`, `phi4-mini:3.8b`, `llama3.2:3b`, `llama3.1:8b`); larger models (`gpt-oss:20b`) gated by 20 GiB free-RAM preflight | Four small models passed the Nov 6 CPU smoke; LLaVA context mismatch (`4096` vs `2048`); GPT-OSS Ollama token-text alignment unresolved |
| **Modal — remote GPU** | Run on a pinned remote GPU when local hardware is insufficient or for predictable hardware | Small or large models; pinned vLLM + artifact revisions (`openai/gpt-oss-20b`) | GPT-OSS-20B hosting and logprob capture verified; repaired `harmony-vllm` mapping passed offline replay; awaits 1 live probe (`M2.1`) |
| **NVIDIA NIM — managed API** | Use an existing managed endpoint and `NVIDIA_API_KEY` | Configured endpoint/model; legacy `NvidiaLM` preserved as a profile delegate | Historical integration tested; Nov 5 attempts returned HTTP 504 / timeout ("NIM unavailable") |

Modal is a hosting choice, not a "large-model-only" constraint. Do not claim
Ollama, Modal and NIM executions of the same model name are scientifically
interchangeable: quantization (`Q4_K_M` vs `MXFP4`), tokenizer/template,
effective context, runtime and response mapping change the token evidence.

Methods are selected orthogonal to hosting:

| Method | Required capability | Release 1.0 posture |
| --- | --- | --- |
| **Contamination scoring (single or ensemble)** | Aligned realized-token logprobs, required `top_logprobs` alternatives, condition-matched `CalibrationIdentity` | **Stable 1.0 product path**, subject to host eligibility probe and calibration |
| **Completion-only experiments (Track 1)** | Text generation (`OpenAICompatibleClient` or `AnthropicClient`); optional auxiliary logprob scorer | **Experimental.** Separate evidence rewriting (`M5`) from completion-sample correction (`M6`) |
| **Token-level divergence decoding** | Complete next-token logits `[vocabulary_size]`, shared prefix/tokenizer, step-wise callback control | **Math layer stable; live endpoint evaluation experimental (`M6`)** |

**Anthropic (Opus) and Mistral** remain target/auxiliary provider profiles for the
completion-only method (`M5`/`M6`), not fourth and fifth cards in the primary
logprob-hosting table. Switching from Opus to Mistral changes wire protocol,
model identity and experimental conditions, requiring an explicit profile choice
and fresh calibration—never an automatic fallback.

---

## 2. Python ecosystem leverage vs custom code

To keep maintenance effort low and avoid custom infrastructure bloat, Release 1.0
adopts standard Python ecosystem building blocks where they preserve Recall
Guard's strict audit and numerical invariants:

| Concern | Decision | Rationale |
| --- | --- | --- |
| **Host RAM & Ollama preflight (R2.3)** | Use **`psutil`** (`psutil.virtual_memory().available`) for local host RAM + Ollama's native HTTP `GET /api/ps`, `GET /api/tags`, `GET /api/version` (consolidating `scripts/probe_local_model.py:observe_host`) | Cross-platform (Windows, macOS Metal, Linux) in ~15 lines of Python. Avoids brittle OS-specific PowerShell memory parsers. For containerized runs against a remote/host Ollama, accepts an explicit host-snapshot JSON or reports `local_ram: unknown_across_container_boundary` |
| **CLI UX, plans & error panels (R2.7, R4)** | Use **`rich`** (already a core dependency in `pyproject.toml`) | Provides structured preflight tables, diagnostic panels, progress counters and retry countdowns with zero new dependencies |
| **Code complexity & size gate** | Use **`ruff`** built-in lint rules (`C901` McCabe complexity, `PLR0912` branches, `PLR0915` statements) + existing `scripts/check_architecture.py` | Already in the dev toolchain. Avoids writing and maintaining a bespoke Python `tokenize` LLOC counter script while enforcing complexity caps on touched/new code |
| **Profile & boundary validation** | Keep frozen dataclasses in `core/completion.py`; optionally use **`pydantic>=2.7`** (`TypeAdapter`) if schema boilerplate grows | `ModelConfig` and `ModelResponse` validation (`providers.py:99-148`, `completion.py:33-110`) is already tested and dependency-free. Do not rewrite working core dataclasses just to change libraries |
| **Transport & budget admission** | Retain **`requests`** + existing `RequestControl` / `CallBudget` / `ReadinessControl` | Third-party LLM wrappers (`litellm`, `tenacity`, `instructor`) hide raw error bodies, strip custom vLLM fields (`return_token_ids`, `token_ids`), or bypass atomic physical-request budget accounting |
| **Batch reporting tables** | Optional **`polars`** in `[project.optional-dependencies] analysis` (decoupled from `M1` critical path) | Polars `join(..., validate="1:1" | "1:m")` and strict casting prevent silent row drops and retry-join inflation in batch study reports without slowing down `import recall_guard` |

---

## 3. Unified architecture for Ollama, Modal and NIM

Use a single declarative `ProviderProfile` + `connect()` seam over the existing
ports (`ModelConfig`, `CompletionClient`, `OpenAICompatibleClient`,
`HarmonyOpenAICompatibleClient`, `AnthropicClient`).

```text
configs/providers/*.json
        │
        ▼
ProviderProfile (id, hosting_mode, model_config, response_mapping, host_requirements)
        │
        ├──► preflight(profile) ──► HostPreflightReport (RAM, /api/ps, env credentials, tokenizer files)
        │
        └──► connect(profile, budget, features) ──► CompletionClient
                                                       ├─ OpenAICompatibleClient (Ollama, NIM, generic vLLM, Mistral)
                                                       ├─ HarmonyOpenAICompatibleClient (Modal GPT-OSS-20B vLLM)
                                                       └─ AnthropicClient (Opus completion-only)
```

### How each endpoint runs

1. **Ollama (`hosting_mode="local"`)**
   - **Endpoint:** `http://localhost:11434/v1/chat/completions` (or `http://host.docker.internal:11434/v1/chat/completions` from Docker Desktop, configurable via `RECALL_GUARD_OLLAMA_ORIGIN` or profile override).
   - **Preflight before load (R2.3):** Queries `GET /api/version`, `GET /api/tags` (verifies model digest is present locally without triggering an implicit pull), and `GET /api/ps` (checks already-loaded models and `size_vram` vs `size` to report `CPU`, `GPU`, or `CPU/GPU offload`). When targeting loopback on the same host, checks `psutil.virtual_memory().available` against `profile.host_requirements.min_free_ram_bytes` (20 GiB for `gpt-oss:20b`; 6–8 GiB for 3B–8B models). If insufficient, refuses to load and prints the measured vs required RAM plus the `--profile modal-gpt-oss-20b` or smaller local model alternatives.
   - **Response mapping:** `"generic"` for `gemma3:4b`, `phi4-mini:3.8b`, `llama3.1:8b`, `llama3.2:3b`. (`gpt-oss:20b` on Ollama remains parked in `P1` until an Ollama final-channel mapping without token IDs is designed).

2. **Modal (`hosting_mode="remote_gpu"`)**
   - **Endpoint:** Explicit workspace URL (`https://<workspace>--recall-guard-gpt-oss-20b.modal.run/v1/chat/completions`) authenticated via `credential_ref="env:MODAL_PROXY_BEARER"`.
   - **Control plane vs data plane:** `deploy/modal/` keeps the isolated Python 3.12 operator lock, pinned vLLM 0.17.0 image recipe, CPU expiry watchdog, and volume staging (`probe_modal_model.py`). `recall_guard` core never imports `modal`.
   - **Response mapping:** `"harmony_vllm"`, which loads `HarmonyTokenMap` from the pinned `tokenizer.json` snapshot and instantiates `HarmonyOpenAICompatibleClient` with `return_token_ids=True`.

3. **NVIDIA NIM (`hosting_mode="managed_api"`)**
   - **Endpoint:** `https://integrate.api.nvidia.com/v1/chat/completions` with `credential_ref="env:NVIDIA_API_KEY"` and `"generic"` response mapping.
   - **Compatibility:** Public `NvidiaLM` remains importable for legacy consumers and delegates through the profile/adapter seam.

### Calibration identity across endpoints (R2.5)

Extend `CalibrationIdentity` in `recall_guard/harness/scorer_client.py` with:
- `response_mapping`: `"generic" | "harmony_vllm"`
- `mapping_revision`: `tokenizer_sha256` when `harmony_vllm` is used, else `None`
- `schema_version`: `2` (while preserving version-1 reader compatibility for historical receipts)

This guarantees a calibrator fitted on Modal (`harmony_vllm`, MXFP4) is never
reused on Ollama (`generic`, GGUF), without invalidating fits when a user changes
display names or local port numbers.

---

## 4. Friendly failures, error-body retention and adopter UX

Reuse `RequestError`, `RequestControl`, `ReadinessControl` and
`harness/request_failures.py`. Add a presentation helper (`format_failure_panel`)
that turns every normalized failure into a clear user message while retaining the
sanitized error body (`RequestError.detail`, with registered credentials redacted
and length bounded):

| Observed condition | User-facing summary & next action | Accounting & execution rule |
| --- | --- | --- |
| **Local free RAM below profile minimum** | *"Local host has 13.4 GiB free RAM; `ollama-gpt-oss-20b` requires 20.0 GiB. No model was loaded. Close applications, select `ollama-gemma3-4b`, or use `modal-gpt-oss-20b`."* | Stop preflight; 0 HTTP inference requests; record per-host preflight receipt |
| **Local Ollama unreachable / model not pulled** | *"Cannot reach Ollama at `localhost:11434` (or model tag missing). Run `.\scripts\start_local_ollama.ps1` and verify `ollama list`."* | Stop preflight; never trigger an implicit multi-GB `ollama pull` |
| **HTTP 500 on model load/generation (C1)** | *"Ollama returned HTTP 500 during `<stage>`. Sanitized response body retained in diagnostics."* | Retain `response.text` / JSON error body in `RequestError.detail`; stop without blind retries |
| **HTTP 429 rate limit (transient)** | *"`<provider>` rate-limited request. Waiting `<N>s` within budget (Ctrl+C to stop)."* | Honor `Retry-After` within `ReadinessPolicy` / `CallBudget`; otherwise defer |
| **HTTP 503 / 529 overloaded** | *"`<provider>` is temporarily overloaded. Retry later or choose another profile."* | Bounded retry if configured; classify 529 (Anthropic) and 503 uniformly |
| **HTTP 504 / timeout after POST** | *"`<provider>` timed out after submission. The server may still be processing this request."* | Mark outcome ambiguous (`outcome_ambiguous=True`), usage `None`; no automatic duplicate POST |
| **HTTP 401 / 403 or missing env var** | *"Credential `<credential_ref>` was rejected or is missing from `.env`."* | Fatal stop (`authentication_error`); never echo secret value |
| **Hard quota / billing exhausted (402)** | *"`<provider>` credit or quota allowance is exhausted."* | Fatal stop (`eligibility_error`); no retry loop |
| **Token alignment / missing logprobs** | *"Response text succeeded, but token logprobs failed alignment (`<reason>`). Configuration is ineligible for MIA/MCS scoring."* | Retain response evidence; never emit a numeric `p_memorized` from invalid tokens |

### Keeping the project adopter-friendly

1. **Curated profiles in `configs/providers/`**: Users pick a profile ID
   (`ollama-gemma3-4b`, `ollama-phi4-mini`, `ollama-llama3.1-8b`,
   `ollama-llama3.2-3b`, `ollama-gpt-oss-20b`, `modal-gpt-oss-20b`, `nim-default`)
   instead of hand-writing `ModelConfig` dictionaries.
2. **Progressive CLI (`doctor -> probe -> run`)**:
   - `dry-run / doctor`: validates profile, `.env` credential presence, host free
     RAM, and input hashes with **0 inference calls**.
   - `probe`: sends **1 bounded request** to record host capability/alignment
     evidence before any calibration spend.
   - `calibrate / score`: runs under the frozen `CallBudget`.
3. **Automatic plan hashing**: Provide a helper/CLI flag (`--freeze-plan`) that
   computes `manifest_sha256` and `corpus_sha256` automatically so users do not
   have to run manual PowerShell `Get-FileHash` commands.
4. **3-line Python onboarding**: `MemoryGuardedScorer.from_profile("ollama-gemma3-4b", budget=CallBudget(100))`
   wires `ProviderProfile -> connect() -> LegacyLogprobAdapter -> MemoryGuardedScorer`.

---

## 5. Execution environments: Native `uv run` + Docker Compose

- **Local CLI & Ollama workflows:** Supported via `uv run` in PowerShell, bash,
  or zsh on Windows, macOS, and Linux. This keeps local `localhost:11434` access
  and `psutil` host RAM checks immediate and zero-config for adopters.
- **Reproducible container & CI workflows:** `docker compose run --rm dev` and
  Dagger CI remain the canonical reproducible Linux test and notebook environment.
  When running inside Docker against an external Ollama host, the profile
  endpoint uses `host.docker.internal:11434` (or an explicit host URL) and
  requires an explicit host-resource snapshot if checking a mandatory RAM gate.
- **Credential hygiene:** All secrets live in the root `.env` (never baked into
  images or printed) and are injected only at runtime via `uv run --env-file .env`
  or explicit container environment pass-through.

---

## 6. Maintainability, complexity gates and optional tabular analysis

### Enforcing complexity with Ruff (zero custom gate code)

Instead of maintaining a custom tokenizer script, enable Ruff's built-in
structural lint rules in `pyproject.toml`:
- `C901` (`max-complexity = 10` for new/touched functions)
- `PLR0912` (max branches) and `PLR0915` (max statements)
- Existing oversized modules (`harness/intervention_types.py`,
  `harness/workflow.py`, `dataset/fmp_corpora.py`, `harness/scorer.py`) are
  listed in `[tool.ruff.lint.per-file-ignores]` as a checked baseline so new
  slices cannot add complexity violations while avoiding unrelated formatting churn.

### Tabular analysis contracts (Polars extra, decoupled from `M1` critical path)

- Keep `numpy` and `scikit-learn` in `core`/`mia`, and `pandas`/`vectorbt` in
  `portfolio`.
- When batch reporting is refactored (or in optional `analysis` utilities), use
  explicit schemas (`planned_draws`, `request_attempts`, `draw_results`,
  `ensemble_results`) joining observed results onto `planned_draws` with Polars
  `validate="1:1"` / `"1:m"` so failed or unattempted draws are never dropped
  from denominators.

---

## 7. Streamlined task alignment for `tasks.md`

To keep the path to research results fast and low-risk:

| Milestone | Scope & refinement | Gate |
| --- | --- | --- |
| **M0.1–M0.2** | Reconcile child `spec.json` phases with verified checkboxes; commit consolidation while preserving `.env.sample` and `.openidea/` unstaged | `offline` |
| **M1.1** | Approve [design.md](design.md) covering `ProviderProfile`, `connect()`, `psutil` + `/api/ps` preflight, `CalibrationIdentity` v2, and friendly failure panels | `offline` (human-gated) |
| **M1.2** | Finish atomic fatal-stop/failure sanitization (`APR 3.2`) and prune fixture-only active-polling loop (`APR 3.3`) | `offline` |
| **M1.3** | Implement `recall_guard/core/profiles.py` (`ProviderProfile`, `connect()`, curated `configs/providers/*.json`), update `CalibrationIdentity`, add `MemoryGuardedScorer.from_profile()` | `offline` |
| **M1.4** | Implement `psutil` + Ollama `/api/ps` free-RAM preflight (`20 GiB` gate for `gpt-oss:20b` with Modal/smaller-model guidance) and sanitized error-body retention + Rich failure panels | `offline` |
| **M1.5–M1.6** | Integrate adopter flow + compact report, update README backend guide, and run full offline regression + Docker smoke + Ruff complexity check | `offline` |
| **M2.1** | Run **1 live Modal eligibility request** (`0` retries, within USD 5 pilot allowance) for `modal-gpt-oss-20b` through `HarmonyOpenAICompatibleClient`; stop apps and record receipt | `authorize` |
| **M3.1–M3.2** | Resolve/exclude LLaVA context mismatch (`4096` vs `2048`); freeze Track 2/3 measurement plan and budgets | `offline` -> `authorize` |
| **M4.1–M4.3** | Fit per-profile calibrators, evaluate held-out single-answer and ensemble arms under separate budgets, publish dated report | `authorize` |
| **M5.1–M5.3** | Track 1 completion-only rewriting pilot (can be planned in parallel with `M2–M4` once `M1.3` lands) | `authorize` |
| **M6.1–M6.3** | Divergence decoding evaluation (revision 2): separate full-logit callback route (e.g. Modal worker) from completion-sample correction route | `authorize` |
