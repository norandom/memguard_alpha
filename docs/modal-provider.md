# Optional Modal host: review a plan before spending

Use Modal when you want to evaluate GPT-OSS-20B without hosting a GPU locally.
Recall Guard uses `OpenAICompatibleClient` and the existing evidence and
calibration contracts. Start with the offline plan below. A working endpoint,
valid token evidence and a useful contamination score are separate results.

The deployment recipe and bounded runner are verified offline. The
[2026-11-06 live pilot](results/2026-11-06-modal-pilot.md) verified weight staging
and image build, then stopped on a deployment rejection before readiness or
inference. All experiment apps are confirmed stopped; weights remain retained.
A100 execution, endpoint authentication and token evidence remain unverified.
The paid pilot was approved; further retries are halted for diagnostic review
under task 10.2. The original local CPU result remains in the
[local integration guide](local-provider.md).

## Inspect the plan without an SDK or account

From the repository root in PowerShell, with the root development environment:

```powershell
uv run --no-sync python deploy/modal/manifest.py configs/modal/gpt-oss-20b-pilot.json
uv run --no-sync python scripts/probe_modal_model.py
```

Both commands are verified offline. They do not import Modal, resolve credentials,
download artifacts, deploy or send inference. The runner prints `mode: plan`,
`inference_posts: 0` and `cloud_mutations: 0`. Missing workspace, endpoint,
provider budget, account approval, pinned tokenizer and remote cache are explicit
blockers. A successful plan exit does not mean execution is ready. No endpoint
or model is substituted on failure.

The manifest `configs/modal/gpt-oss-20b-pilot.json` pins model and tokenizer commit
`6cee5e81ee83917806bbde320786a8fb61efebee`, MXFP4, vLLM 0.17.0 and the full
Linux AMD64 image digest. Inspect it for the complete digest, sources, license
record and dated pricing assumptions.

| Planned bound | Value |
| --- | --- |
| Host | One A100 40 GB; four physical CPU cores; 32 GiB RAM (32768 MiB API request) |
| Parallel work | One model container, one request, tensor parallel size one, one scheduled sequence |
| Idle scaling | Minimum zero, buffer zero, maximum one; 60-second scaledown window |
| Context | 4096 tokens including input and output allowance |
| Inference | One eligibility probe, then at most seven fixed measurements; zero retries |
| Generation | Temperature 0, top_p 1, logprobs true, top_logprobs 5 |
| Output | At most 512 tokens per request, 4096 across eight requests |
| Time | 120 seconds per request; 600 seconds startup within a 1200-second GPU phase; 120 seconds cleanup |

## Install the optional operator tools

This explicit install downloads Python tooling, not models or remote images:

```powershell
uv sync --project deploy/modal --locked
uv run --project deploy/modal --no-sync python -I -c "import modal, recall_guard; from transformers import AutoTokenizer"
uv run --project deploy/modal --no-sync python scripts/probe_modal_model.py
```

The isolated lock includes Modal 1.6.3, the editable local Recall Guard package,
Transformers 4.55.4, Hugging Face Hub 0.34.4 and Jinja2 3.1.6 for chat templates.
The operator project requires Python 3.12 (`deploy/modal/.python-version`);
`uv sync` selects that minor independently of the root development environment.
Modal serializes the serving class and expiry function, so the operator's Python
minor must match both remote images. Execution and recipe construction reject
other minors locally, before SDK import, cache inventory or deployment. Offline
planning remains usable on the root project's supported Python versions.

The pinned vLLM image uses Python 3.12, matching the expiry image's explicit
`debian_slim(python_version="3.12")`. On 2026-11-06, read-only Docker Registry
manifest/config requests confirmed config digest
`sha256:700d8ac4f37a3f2975b79e67406d7f2689bdd8c87e3e7a62721bf03921940935`:
its build history records `PYTHON_VERSION=3.12` and selects that interpreter as
`python3`. No image layers were pulled for this check.

The local tokenizer needs no PyTorch, CUDA or vLLM install. Python 3.12 tooling,
chat-template rendering and lazy serving/expiry construction were checked with
network socket operations blocked. Transformers reports that only
tokenizer/configuration utilities are available without a model framework.

Recall Guard's default installation has no Modal or accelerator dependency.
`deploy/modal/serve.py` constructs recipes only when called. Use the bounded
runner for execution; there is no module-level app for `modal deploy`.

## Configure account access and budgets after review

These are operator setup instructions, not verified account procedures.
No account state or credentials were inspected for this delivery.

Use the single repository-root `.env` for credentials. The tracked `.env.sample`
lists the fields; if `.env` already exists, add missing fields without replacing
the file. Obtain an account API token in Modal's dashboard and put its ID and
secret in `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`. Do not use shell exports,
OS/user environment secrets, separate provider files or `modal token new`.
Create or select the dedicated `recall-guard-pilot` environment in the intended
workspace. Verify permission to deploy apps, read provenance and stop experiment
compute. Keep values out of shell output and Git.

Create a separate Modal proxy token in workspace settings. Supply its ID and
secret joined by a period in `MODAL_PROXY_BEARER` in that same root `.env`.
Credentialed commands use `uv run --env-file .env ...` so the SDK receives values
only for that child process. Offline planning omits this option and reads no
credentials. This bearer protects the Web Function; SDK control-plane
credentials are separate. The recipe sets `requires_proxy_auth=True`.
[Modal proxy authentication](https://modal.com/docs/guide/webhook-proxy-auth)
documents this scheme (checked 2026-11-06). Live denial without waking a GPU
remains a required task 10.2 check.

Set `target.workspace` and the explicit `https://YOUR-HOST.modal.run/v1` endpoint
in the reviewed manifest before freezing its hash. Confirm the expected URL
with the operator; no URL is discovered or silently rewritten by the runner.
Keep `cost.provider_budget` null: the reviewed budget belongs in the separate
approval record. The serving recipe uploads the canonical manifest, so its app,
environment, artifacts, resources and limits must match the runner's plan.

Review the pinned model card, Apache-2.0 license and applicable notices before
download or redistribution. Public metadata access recorded in the manifest is
not proof of this account's allowance or successful weight access.

Review monthly gross usage and remaining allowance, including prior usage.
[Modal budgets](https://modal.com/docs/guide/budgets) distinguish workspace usage
from environment compute coverage; environment budgets require an eligible plan
(checked 2026-11-06). Have the responsible owner configure suitable policy in
the dashboard. The runner never overwrites shared policy. Unavailable billing
requires an explicitly operator-confirmed allowance; unknown usage is null.

The configured 1200-second compute subtotal is
`1200 * (0.000583 + 4 * 0.0000131 + 32 * 0.00000222) = USD 0.847728`.
It excludes staging, builds, storage, CPU expiry control, cleanup tails, egress
and excess CPU/RAM use. USD 5 is a target, not an exact ceiling. The total estimate
is unknown until these additional costs are reviewed. Recheck
[prices](https://modal.com/pricing) before approval. See the
[receipt and cost contract](modal-cost-report.md) for accounting and reconciliation.

## Stage pinned weights separately, without a GPU

No staging command has been executed or automated by the runner. This manual
strategy is for a separately approved setup window and needs disk/network and
storage cost review. Local Ollama GGUF blobs are incompatible with this vLLM
MXFP4/safetensors recipe. Do not upload the Ollama store or the checkout.

The following PowerShell snippet uses the pinned Hub client on the local CPU.
It downloads selected snapshot files into the ignored `models/` directory, using
the exact directory layout expected by the runner and remote server. No model
framework or GPU is invoked. The download itself is unverified here.

```powershell
@'
import json
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

plan = json.loads(Path("configs/modal/gpt-oss-20b-pilot.json").read_text(encoding="utf-8"))
artifacts = plan["artifacts"]
assert artifacts["model"] == artifacts["tokenizer"]
assert artifacts["model_revision"] == artifacts["tokenizer_revision"]
revision = artifacts["model_revision"]
snapshot = Path("models/hub/models--openai--gpt-oss-20b/snapshots") / revision
snapshot.mkdir(parents=True, exist_ok=True)
index_path = hf_hub_download(artifacts["model"], "model.safetensors.index.json",
                             revision=revision, local_dir=snapshot, token=False)
shards = set(json.loads(Path(index_path).read_text(encoding="utf-8"))["weight_map"].values())
assert shards and all(Path(name).name == name for name in shards)
files = sorted(shards | {"model.safetensors.index.json", "config.json", "tokenizer.json",
    "tokenizer_config.json", "chat_template.jinja", "generation_config.json", "LICENSE", "NOTICE"})
snapshot_download(artifacts["model"], revision=revision, local_dir=snapshot,
                  allow_patterns=files, token=False, max_workers=1)
required = shards | {"model.safetensors.index.json", "config.json", "tokenizer.json", "tokenizer_config.json"}
assert all((snapshot / name).is_file() and (snapshot / name).stat().st_size > 0 for name in required)
'@ | uv run --project deploy/modal --no-sync python -
```

`token=False` avoids reading a cached Hugging Face credential for this public
artifact. Optional template/notice files are copied when present. The pinned
[Hub API](https://huggingface.co/docs/huggingface_hub/v0.34.4/en/package_reference/file_download)
supports revision and local directory selection (checked 2026-11-06).

After reviewing these exact files, upload only this snapshot's selected files to
the dedicated volume, retaining the same `hub/models--.../snapshots/COMMIT/` path.
For Modal 1.6.3, `Volume.from_name(name, environment_name=..., create_if_missing=True)`
and `with volume.batch_upload() as batch: batch.put_file(local_file, remote_path)`
provide a local upload route without creating a GPU. Build `remote_path` from
`hub/models--openai--gpt-oss-20b/snapshots/` plus the pinned revision and filename.
Use volume `recall-guard-gpt-oss-20b-weights` in the reviewed environment. Validate
resolved local paths remain inside the snapshot, reject symlinks/junctions and
upload only `files` from the snippet that exist; exclude Hub `.cache` metadata.
Do not recursively upload `models/`. This is a manual strategy, not a verified
end-to-end upload procedure. The installed SDK signatures were checked against
the [Volume API](https://modal.com/docs/sdk/py/latest/Volume) on 2026-11-06.

Rerun the offline plan after staging the local tokenizer. It counts the actual
one-user-message chat template locally with downloads disabled, hashes tokenizer
files and checks each input plus 512 output tokens against context. Before GPU
deployment the runner inventories the existing remote volume and rejects a
missing index, missing/nonpositive indexed shard or missing required config.
Inventory proves presence, not shard content integrity or a successful vLLM load.

## Freeze approval and start only the reviewed pilot

Save approval outside tracked files, for example `runs/modal-approval.json`.
This is the runner's accepted record shape, intentionally unauthorized. Replace
hashes with the final plan output and all review assertions with actual evidence;
copying this example cannot grant approval.

```json
{
  "paid_execution_authorized": false,
  "account_access_verified": false,
  "operator_allowance_confirmed": false,
  "model_access_verified": false,
  "pricing_and_budget_reviewed": false,
  "manifest_sha256": "FINAL_PLAN_MANIFEST_SHA256",
  "prompts_sha256": "FINAL_PLAN_PROMPTS_SHA256",
  "workspace": "YOUR_WORKSPACE",
  "provider_budget_reference": "operator_budget_review",
  "monthly_budget": {
    "scope": "environment_compute",
    "workspace": "YOUR_WORKSPACE",
    "environment": "recall-guard-pilot",
    "period_month": "2026-11",
    "checked_at": "2026-11-06T08:00:00Z",
    "source": "operator_budget_review",
    "monthly_limit_usd": 20,
    "prior_usage_usd": null,
    "requested_allowance_usd": 5,
    "accounting_available": false,
    "operator_confirmed": false
  }
}
```

All five top-level assertions must be exactly true for execution. Workspace and
both hashes must match. `monthly_budget` needs the current UTC month and a
nonfuture timezone-aware check time; use actual reviewed amounts. With accounting
available, prior usage must be known; otherwise operator confirmation must be
true. Known insufficient headroom always fails. `scope` also accepts `workspace`.
`source` is a 1-100 character letters/digits/underscore/hyphen reference, not raw
billing data. The allowance must cover the compute subtotal; human review must
also cover excluded costs. This record is a review snapshot, not reserved funds.

Only after separate paid approval and prerequisite checks, the execution command
is (not run for this delivery):

```powershell
uv run --env-file .env --project deploy/modal --no-sync python scripts/probe_modal_model.py --execute --approval runs/modal-approval.json --receipt runs/modal-pilot-NEW.json
```

The receipt path must be new and its parent must exist. Never resume or replay
an existing receipt. Startup deploys the CPU expiry controller before the serving
app, arms the deadline, waits for acknowledgment, then invokes SDK readiness
without generating tokens. Missing acknowledgment fails closed. The serving
hook refuses to start vLLM without the armed expiry. The uploaded image is offline
for weights; startup cannot silently download another revision.

## Stop, recover and interpret the receipt

Cancellation, deadline, eligibility failure, HTTP authentication failure or an
uncertain transport result stops admission without replay. The runner's `finally`
path stops only the recorded experiment app ID and verifies its state for up to
two minutes. Local process loss leaves the separately armed CPU controller to
stop that same app at its deadline, without calling inference. Network or control
plane failure can still leave cleanup unverified.

For unverified cleanup, use the exact app ID in the receipt and recovery command:

```powershell
uv run --env-file .env --project deploy/modal --no-sync python -m modal app stop ap-YOUR_EXPERIMENT_ID --yes
```

Run with the reviewed workspace credentials and verify the stopped state in that
environment. Never stop by reusable name or stop unrelated apps. The independent
expiry app and provenance Dict remain; stop/remove them separately only after
confirming no active lease and retaining needed receipts. The weight volume is
retained deliberately and may keep accruing storage charges. Scale-to-zero does
not delete storage or replace verified shutdown. The recorded storage assumption
is USD 0.09/GiB-month; actual billing and retention duration remain unknown.

Receipt `attempts` have one row per physical inference intent/outcome; `position`
is a zero-based frozen prompt index. Lifecycle/control rows are separate.
For new failures, read the legacy `error` type together with `error_stage` and
`diagnostics`. Stages distinguish budget preflight, cache inventory, CPU controller
deployment, model deployment, expiry arming, proxy denial, readiness, endpoint
checks and provenance. Each diagnostic contains a bounded sanitized message and
up to twelve filename/function/line-number frames; chained errors are bounded to
four type/message records. No source lines or locals are captured. Inference
attempt diagnostics contain type and frame metadata only, with the existing
provider failure reason and HTTP status; response bodies and generated text are
excluded. The first failure remains authoritative when cleanup or final
provenance collection also fails; read later diagnostics and cleanup state too.
Inference ancestry and provider response-error branches remain opaque through
later cleanup exception chains, including their nested causes.
Redaction uses nonempty credential values already present in the executing child
process, including proxy bearer components, and never opens `.env`. Default
planning remains independent of credentials. Old receipts are unchanged and
cannot recover previously omitted exception messages.

Token positions in the adapter are generated-token positions, alternatives align
at that same position, and probabilities use natural logarithms. Top-five evidence
is partial, never a renormalized full-vocabulary distribution. GPT-OSS reasoning
and final-answer text must align as returned; stripping or retokenizing to force
acceptance is forbidden. Raw output and token arrays are not published.

Read `evidence_status`, `environment_status`, `calibration`, cleanup and
`cost_report` separately. Usable evidence requires nonempty aligned logprobs and
five alternatives. Runtime, quantization, tokenizer/template, model, context or
score-semantics changes require fresh capability checks and calibration
revalidation. Configured A100 resources are not observed hardware. Unknown/drifted
provenance cannot establish a calibration match. The pilot performs no calibration.

The [cost contract](modal-cost-report.md) distinguishes cold startup, inference
wall time, failed attempts, usable measurements, estimates and observed charges.
Eligibility is excluded from the usable measurement denominator. Unknown cost or
zero usable measurements gives null cost per usable measurement, never a claimed
zero. Later billing is appended without rewriting attempts or frozen rates.

Observed token evidence establishes endpoint feasibility. A held-out AUC would
measure discrimination; calibration and mitigation efficacy require their own
experiments. [Jane Street's approach](https://blog.janestreet.com/mitigating-memorization-in-llms/)
motivates the separate divergence work (source checked 2026-11-06). This provider
extension does not establish generation control or successful divergence decoding.
Frontier APIs without logprobs remain on the separate
[completion-only workflow](completion-api.md), with no silent fallback here.
