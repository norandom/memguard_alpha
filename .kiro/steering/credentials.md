# Credentials

User instruction: use one shared `.env` at the repository root for all project
credentials. `.env.sample` is the tracked template and contains empty values.

- Do not create per-provider or per-tool `.env` files, persist credentials in
  OS/user environment variables, or use provider secret stores for this workflow.
- Do not require `modal token new` or a separate Modal credential/profile file.
  Store both Modal account token fields and the proxy bearer in the root `.env`.
- Pass the root file explicitly to credentialed commands with
  `uv run --env-file .env ...` from the repository root. This supplies values
  only to the child process for SDKs that read process variables; it does not
  make the OS environment a second credential source of truth.
- Keep offline planning independent of credentials: planning commands omit
  `--env-file` and must not read `.env` or resolve secret values.
- Preserve an existing `.env`; never overwrite it with the sample. Add missing
  fields without reading or printing existing secret values unnecessarily.
- Never commit credentials, log their values, include them in receipts, or
  upload `.env` to model hosts. Keep `.env` ignored by Git and build contexts.
- Model identities, endpoints, nonsecret workspace settings, budgets and run
  approvals remain explicit configuration; the shared file stores credentials.

This supersedes earlier setup instructions suggesting shell exports, persistent
environment secrets, secret managers or separate local credential files.
