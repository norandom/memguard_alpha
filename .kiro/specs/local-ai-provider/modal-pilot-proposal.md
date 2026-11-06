# Modal live pilot: concrete approval packet

Prepared 2026-11-06. Task 10.2 remains pending paid approval and budget confirmation.

Subsequent authorization: the user approved this paid plan and confirmed the
discussed USD 30 monthly workspace usage limit. That limit is operator-confirmed;
the budget-read API denied access, so it is not reported as provider-verified.
The local approval record now permits the USD 5 pilot. Dedicated environment
and volume creation and CPU-only uploads have begun; GPU acceptance remains pending.

## Verified preparation

- Root `.env` account token authenticated for workspace `wishinet`.
- Proxy ID is registered in that workspace; proxy secret remains endpoint-unverified.
- Read-only environment inventory contains `main` only. Proposed dedicated
  environment: `recall-guard-pilot`; creation has not occurred.
- November billing summary: gross USD 0.02792516, billed USD 0 after free-storage
  adjustment. This does not establish remaining credit or configured budget.
- Public pinned model metadata and tokenizer are accessible. Tokenizer/config
  files and weight index downloaded locally into ignored `models/`; no weight
  shards downloaded or uploaded.
- Local prompt counts: 73, 75, 76, 74, 76, 76, 74, 73 tokens. Each fits context
  4096 with the 512-token output allowance.

## Proposed paid work

Create the dedicated environment and weight volume; stage only the index's
three safetensors shards and required configuration/tokenizer/license files.
Total selected size: 13,789,255,711 bytes (12.842 GiB). Exclude the duplicate
`original/model.safetensors` checkpoint. No Ollama blobs or credentials upload.

Use the frozen model/image revisions in `configs/modal/gpt-oss-20b-pilot.json`,
one A100 40 GB, four CPU cores and 32 GiB host RAM. Serve with the explicit label
`recall-guard-gpt-oss-20b-pilot`; planned URL:
`https://wishinet--recall-guard-gpt-oss-20b-pilot.modal.run/v1`.
The runner must verify the actual URL before inference; this URL is not deployed.

Permit one eligibility probe followed by at most seven fixed measurements,
zero retries, 512 output tokens/request, temperature 0, top_p 1, five logprob
alternatives. Total GPU phase: 20 minutes including at most ten minutes startup;
120 seconds/request, plus two minutes cleanup verification. Arm independent
CPU expiry before admission; verify owned-app shutdown. No calibration or ensemble.

## Costs and outstanding confirmation

Prices rechecked on [Modal pricing](https://modal.com/pricing), 2026-11-06.
GPU/CPU/RAM subtotal for 20 minutes: USD 0.847728. Two-minute cleanup at the same
allocation adds approximately USD 0.084773 if fully consumed. Image preparation,
CPU controller, staging, storage and excess resource usage are additional;
final total remains unmeasured. The complete planning allowance below includes
these components rather than treating the compute subtotal as the total.

| Component | Estimate/allowance USD | Assumption |
| --- | ---: | --- |
| GPU/CPU/RAM phase | 0.847728 | 1200 seconds at reviewed requests |
| Cleanup tail | 0.084773 | 120 seconds at the same allocation |
| Two CPU image builds | 0.148128 | 600 seconds each at 4 cores/32 GiB; planning assumption, not observed builder allocation |
| CPU expiry and control | 0.002894 | 1320 seconds at 0.125 core/256 MiB; recipe requests 128 MiB, allow extra memory |
| Retained weights | 1.155802 | Full month at list storage price before allowances |
| Modal network egress | 0.040000 | Reserve 1 GiB at list rate, ignoring free allowance |
| Contingency | 2.720675 | Builds, metadata storage, staging overhead or usage above requests |
| Total planning allowance | 5.000000 | Estimates and reserves, not an invoice or exact ceiling |

Weight staging uses local CPU download and SDK upload, not a remote staging
compute function; its Modal storage/network charges are included above.
Build resource use and duration are assumptions: cancellation of a local
deployment wait may not stop a provider-side build. Usage above assumptions or
control-plane failures can exceed this allowance. Provider budget confirmation
is required in addition to the runner's request/time controls. No automatic
hardware escalation or retries.

Retaining selected weights lists at approximately USD 1.1558/month before
allowances; model cache is retained by this plan. Deleting it later is separate
from stopping compute. Existing workspace budgets/remaining credit are unknown.
Before paid work, the operator must confirm at least USD 5 remaining allowance
and the scope/current month/prior usage of the budget; do not overwrite shared
workspace policy. Current gross usage is not proof of a configured cap.

Local review artifacts: `runs/modal-preflight/account.json`, `artifacts.json`,
`staging-plan.json`, `plan.json`, `approval.json` (ignored, no credentials).
Approval assertions remain false until the user authorizes this concrete paid
plan and confirms budget information. No paid resources created or GPU started.
