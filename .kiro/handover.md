# Handover: hosting and token evidence established

Updated: 2026-11-06. Workspace: `D:\Source\memguard_alpha`.
Implementation baseline: `main`, commit `d0cd222`.

## Start here

We established Ollama and Modal hosting/API integration and verified token
evidence handling. We have not yet measured fresh calibration quality, ensemble
benefits or Jane Street divergence-decoding effectiveness on this infrastructure.
The next stage is a bounded measurement experiment, not more hosting work.

Read the [local-provider spec](specs/local-ai-provider/spec.json),
[tasks](specs/local-ai-provider/tasks.md) and
[validation record](specs/local-ai-provider/live-feasibility-validation-2026-11-06.md).
Tasks 7.1 and 8-10.3 are complete for their recorded scope. Conditional NPU tasks
remain unresolved; there is no overall NPU-inclusive feature GO.

## Verified results

| Surface | Result | Limit |
| --- | --- | --- |
| Ollama: `phi4-mini:3.8b` | Aligned logprobs, five alternatives; environment matched | One short CPU probe |
| Ollama: `llama3.1:8b` | Aligned logprobs, five alternatives; environment matched | One short CPU probe |
| Ollama: `llama3.2:3b` | Aligned logprobs, five alternatives; environment matched | One short CPU probe |
| Ollama: `gemma3:4b` | Aligned logprobs, five alternatives; environment matched | One short CPU probe |
| Ollama: `llava-llama3:latest` | Valid token evidence | Observed context 4096 versus planned 2048; environment mismatch |
| Modal: `openai/gpt-oss-20b` | Hosting, readiness and authenticated inference worked; corrected response mapping passed strict validation | Fix verified by offline replay of one captured live response, not another generation after the fix |

The latest Ollama smoke made exactly five inference requests, zero retries and
zero local GPT requests. All five ran on CPU with Ollama 0.40.11. Each model was
unloaded and verified before the next; the final loaded-model list was empty.
The Ollama server itself may remain running. Score normalization semantics remain
unknown and the observed alternatives are partial top-k evidence.

Both missing Llama artifacts were installed with full blob size/SHA-256 checks.
Models are retained under ignored `models/` in native host/cache layouts.

Reports:

- [Five-model Ollama smoke](../docs/results/2026-11-06-local-model-smoke.md)
- [Original Modal pilot and startup repairs](../docs/results/2026-11-06-modal-pilot.md)
- [GPT-OSS response-mapping repair and retest](../docs/results/2026-11-06-gpt-oss-response-mapping.md)

Each report has a companion JSON containing sanitized evidence and source hashes.

## The GPT-OSS bug and its repair

Pinned vLLM 0.17.0 returns logprobs for the full Harmony generation while
`message.content` contains the final answer. Our generic adapter attached that
full sequence to final-only text. Strict validation detected the mismatch; the
original model-capability interpretation was too broad. This was a
response-handling integration bug, unrelated to nondeterminism across requests.

Retest 09 made one additional eligibility request with metadata-only
`return_token_ids: true`. The response contained 30 probability positions;
the final answer occupied the explicit zero-based span `[28, 29)`. Its one token
had five alternatives. The old adapter reproduced the failure; the repaired
adapter passed strict validation on that same response with HTTP disabled.
Original selected logprobs and alternatives were preserved exactly.

Implementation:

- [Harmony token metadata and ID-based parser](../recall_guard/core/harmony_evidence.py):
  `HarmonyTokenMap.from_tokenizer_json(...)` loads pinned control IDs without a
  tokenizer runtime dependency.
- [Provider adapter](../recall_guard/core/providers.py):
  explicitly select `HarmonyOpenAICompatibleClient(..., token_map=...)` with
  model `openai/gpt-oss-20b` and `return_token_ids=True`.
- [Regression fixtures](../tests/core/test_harmony_evidence.py).
- [Operator guide and configuration example](../docs/modal-provider.md).

Generic client behavior, response serialization, core dependencies and scoring
mathematics remain unchanged. The mapper rejects unsupported/ambiguous channels,
tools, truncation and unsupported byte layouts. It does not find an answer
substring, guess a suffix, retokenize output or fabricate probabilities.

The tokenizer/model revision is
`6cee5e81ee83917806bbde320786a8fb61efebee`. Use the existing pinned tokenizer at
`models/hub/models--openai--gpt-oss-20b/snapshots/<revision>/tokenizer.json`.
Fresh calibration is required for this explicit final-answer evidence profile.

Historical attempt 08's raw response was not retained; its exact layout remains
unknown. Response 09 supplied direct diagnostic evidence. Preserve both original
receipts and distinguish live generation, offline replay and calibration.

## Hosting, requests and cost state

At the last verification, the runner had stopped the Modal model app and a
separate provider inventory confirmed all twelve currently visible owned model,
expiry and diagnostic app entries stopped. Older stopped entries had aged out of
inventory; earlier fifteen-entry observations remain historical facts. Recheck
state before any future operation rather than reusing old app IDs or process IDs.

Modal attempts 01-07 sent zero inference requests. Attempt 08 sent one eligibility
request; retest 09 sent one separately authorized eligibility request. Neither
sent measurement prompts, and replay sent zero HTTP requests. No further
generation, calibration or ensemble batch is currently authorized by this handover.

Both runs requested A100-40GB. Attempt 08 observed A100-SXM4-80GB; response 09
observed A100-SXM4-40GB. Keep requested and observed hardware separate. Do not infer
minimum VRAM, comparable timing or attributable pricing from those labels alone.

The latest workspace snapshot showed provisional gross USD 0.29792825 and billed
USD 0. These are workspace-wide observations, not a reconciled experiment invoice
or a final free-cost claim. The user approved a USD 5 pilot allowance and confirmed
a USD 30 monthly workspace limit; the latter is operator-confirmed, not API-verified.
Retained weight storage remains approximately USD 1.16/month at list rates before
allowances. No volume or local model deletion was requested.

## Verification already completed

- Full offline regression: **2888 passed, 2 skipped**, 369.28 seconds.
- Focused mapping/provider regressions: **159 passed**.
- Independent code, publication and network-forbidden replay review: **APPROVED**.
- Ruff, architecture and strict MkDocs checks passed; latest docs build 14.89 seconds.
- Skips: optional Modal SDK absent from the root environment; an existing symlink
  test lacked host privilege. No host changes were made. One notebook event-loop
  fallback warning did not fail the suite.

Do not repeat the full suite for a documentation-only restart. Re-run appropriate
checks when source changes or a concrete unresolved concern justifies them.

## Private evidence and safe restart

Ignored receipts are under `runs/local-provider/2026-11-06-smoke/` and
`runs/modal-preflight/`. Key Modal files are:

- `pilot-2026-11-06-08.json` and `pilot-2026-11-06-09.json`
- `token-mapping-response-09.json`: private generated response; never print or publish
- `token-mapping-production-replay-09.json` and `token-mapping-structural-09.json`
- `final-observations-09.json` and `billing-after-09.json`

Do not replay inference from old receipts or run the one-off diagnostic helper
again. It deliberately refuses existing output paths. Raw download logs may also
contain signed registry URLs; keep them private.

Read-only/offline inspection from the repository root in PowerShell:

```powershell
git status --short
uv run --no-sync python scripts/probe_modal_model.py
uv run --no-sync python -m pytest tests/core/test_harmony_evidence.py -q
```

The default Modal probe only plans; it allocates no GPU and sends no inference.
The root environment supports library development. Optional Modal operator tooling
has a separate lock under `deploy/modal/` and requires Python 3.12 to match its
serialized remote runtime; do not replace it with Python 3.14 for deployment.

## Next work to propose

1. Prepare a small measurement plan with explicit datasets, development/held-out
   split, model identities, response profiles, generation settings and total
   request/token/cost limits. GPT-OSS-20B remains included.
2. Include a fresh live eligibility check through the corrected Harmony profile
   before calibrating. Resolve LLaVA's effective context before admitting it to a
   matched experiment. Treat historical NIM failures as "NIM unavailable".
3. Fit and evaluate fresh calibrators; report single-answer and ensemble results
   separately, with their additional requests explicitly budgeted.
4. Then evaluate divergence decoding with a validated statistical source,
   development calibration, frozen configuration and paired held-out measurement.
   Partial top-k logprobs do not establish a full next-token distribution or
   generation-control capability. Cite
   [Jane Street's approach](https://blog.janestreet.com/mitigating-memorization-in-llms/).

The frontier completion-only/Opus alternative remains separate. NPU prerequisites
remain unresolved and should not delay the supported endpoint-based measurements.

## Project rules to retain

- Work directly on `main`; no feature branches or worktrees. Releases use explicit
  tags. Remote push was not completed in the preceding work; do not imply it was.
- Use PowerShell on Windows and project Python through `uv run --no-sync python ...`.
  WSL is only the Dagger CI environment. Dagger is also the release CI in Actions.
- Keep one root `.env` for all credentials and `.env.sample` as the empty template.
  Pass `--env-file .env` only to credentialed commands. Never print credentials,
  create provider-specific env files or persist secrets in OS/provider stores.
- Keep hosting SDKs and accelerator dependencies outside the lean core library.
- Prefer idiomatic code and names matching domain meaning, array axes, token
  positions, units and DataFrame row grain. Sentrux is not required.
- Pre-existing user changes: `.env.sample` is modified and `.openidea/` is untracked.
  Preserve both; stage only explicit task files.
