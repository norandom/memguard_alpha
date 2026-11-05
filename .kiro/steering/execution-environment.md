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
