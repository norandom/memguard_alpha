# Optional Modal deployment tooling

Use the [operator guide](../../docs/guides/modal-provider.md) for the verified offline
plan, isolated install, account/budget prerequisites, manual CPU-only weight
staging strategy, approval record, bounded start/stop and recovery instructions.
The [cost contract](../../docs/research/modal-cost-report.md) defines receipt fields and
later local billing reconciliation.

`serve.py` supplies lazy authenticated vLLM and CPU expiry recipes. Importing it
does not load the SDK or deploy anything. `manifest.py` validates the frozen
candidate in `configs/modal/gpt-oss-20b-pilot.json`. Run from the checkout root:

```powershell
uv run --no-sync python scripts/probe_modal_model.py
```

Planning needs no Modal SDK or credentials. Optional operator tools are isolated:

All credentials belong in the single repository-root `.env`; see `.env.sample`.
Add `--env-file .env` to `uv run` for credentialed operations. Offline planning
omits it. Do not store tokens in OS/user environment secrets or separate Modal
profiles/files.

```powershell
uv sync --project deploy/modal --locked
uv run --project deploy/modal --no-sync python scripts/probe_modal_model.py
```

The operator project pins Python 3.12, matching the serving and CPU expiry images
for serialized functions. `uv sync` selects it through the project-local
`.python-version`; it leaves the root Python environment unchanged. Execution
and recipe construction reject other Python minors before SDK/cloud access.
Offline planning remains independent of this operator-only restriction.

The lock pins Modal 1.6.3, Transformers 4.55.4, Hugging Face Hub 0.34.4 and
Jinja2 3.1.6 for chat templates, and
installs the local Recall Guard package. It adds no model framework to either
environment and no hosting dependency to the core library. The serving image
contains vLLM/CUDA; it is never installed locally by these commands.

Each serving manifest names its own retained weight volume (`Plan.volume_name`;
the GPT-OSS-20B volume is `recall-guard-gpt-oss-20b-weights`). The serving
runner requires the pinned snapshot in that volume, checks its index and all
indexed shards before GPU deployment, and never creates the volume
(`create_if_missing=False`). Only the CPU-only staging recipe,
`create_staging_app`, creates it (`create_if_missing=True`). For the finance
study's serving arm, `scripts/serve_modal_model.py` (through
`modal_operator.sh stage | serve | stop-serve`) stages the pinned revision in
that CPU-only run, bounded by a stage limits file (`stage_limits.py`), serves
the app with its expiry at the stage's `max_seconds`, and stops it with
verification; `HF_TOKEN` passes by name only. Serving-source uploads use the manifest's exact
three-file allowlist. Never upload the checkout, Ollama store, `.env` or credentials.

The A100/MXFP4 host remains a candidate. Offline construction is verified; live
account access, weight loading, token evidence, cost and shutdown are unverified.
No paid staging, build or pilot is authorized by these instructions.
