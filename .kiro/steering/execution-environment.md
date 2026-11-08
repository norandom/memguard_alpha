# Execution environment

User instruction, 2026-11-08. This supersedes the November 5 native PowerShell
Python execution policy and conflicting Windows support text elsewhere.

- On Windows, run Recall Guard, Python development tools, probes, measurements
  and notebooks in Linux Docker containers. Direct native Windows application
  execution is no longer a supported delivery target.
- Use PowerShell for repository/file operations and Docker launch commands. WSL
  may provide Docker/Dagger infrastructure or the transport to its Docker CLI;
  do not use native Windows Python or ad hoc WSL Python as execution fallbacks.
- Inside the development container, invoke project Python through
  `uv run --no-sync ...` using the existing locked environment. Keep the shared
  Dockerfile, Compose and Dagger recipe; do not create per-provider app images.
- Keep Ollama as an optional external model host. It may run natively on Windows
  to use supported GPU drivers, or on another explicitly selected host. This is
  a model-serving exception, not native Windows Recall Guard support.
- Do not require GPU passthrough for the Recall Guard application container.
  NVIDIA WSL2 passthrough can work; AMD/iGPU support must be observed for the
  chosen runtime. Report CPU/GPU/offload honestly, with no universal GPU claim.
- Configure container-to-Ollama connectivity explicitly. Distinguish Docker
  Desktop host networking from Docker Engine inside WSL; do not assume the same
  hostname reaches Windows in both. Avoid broad unauthenticated exposure.
- Resource preflight must describe the model host, not application-container
  memory. Unknown or stale host evidence cannot pass a required load gate.
- Keep platform-specific hosting/accelerator SDKs outside the core library.
  Preserve portable Python and Linux compatibility; do not add OS branches to
  business logic solely to remove native Windows support.

The release review records the remaining launcher, CI and documentation migration
in [release-1.0/research.md](../specs/release-1.0/research.md). This policy change
does not claim those implementation tasks are complete.

## Local model storage

User instruction, 2026-11-06: store downloaded models under the checkout's
`models/` directory. Ollama uses `OLLAMA_MODELS` pointing to that directory;
preserve its native blobs/manifests layout. Keep model files excluded from Git,
Docker build contexts and Dagger uploads. Test local models sequentially and
unload each model before starting the next.

## Optional remote model hosting

User instruction, 2026-11-06: add Modal hosting to the local-provider spec;
the user already has an account and budget matters. Keep it an optional external
endpoint using the existing adapter, with mandatory observed logprobs. Remote
GPU hosting follows the current Docker application policy and does not add
accelerator/Modal dependencies to the core library. Small and large remote model
profiles may both be offered; switching profiles remains an explicit user choice.
See the
[Modal extension](../specs/local-ai-provider/modal-extension.md) for scope,
cost controls and the proposed bounded pilot.
