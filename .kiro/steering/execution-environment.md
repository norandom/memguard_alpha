# Execution environment

User instruction, 2026-11-05. This policy supersedes earlier guidance suggesting
WSL for ordinary project execution or ad hoc Docker checks.

- On Windows, use PowerShell for repository commands, local model hosting,
  probes, measurements and development tools.
- Invoke project Python through `uv run --no-sync ...` from the repository root.
  Do not invoke a hard-coded base interpreter or `.venv` interpreter directly.
- WSL is only the CI environment for Dagger. Do not use it as a general shell,
  Python workaround, model host or route for ad hoc Docker commands.
- Keep the existing Dockerfile and Dagger pipeline for reproducible CI. This
  does not require accelerator hosting inside CI containers.
- Keep Recall Guard portable: platform-specific model runtimes are optional
  external hosts, not core package dependencies.

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
GPU hosting does not change the PowerShell/WSL policy or add accelerator/Modal
dependencies to the core library. See the
[Modal extension](../specs/local-ai-provider/modal-extension.md) for scope,
cost controls and the proposed bounded pilot.
