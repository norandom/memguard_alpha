# Modal task 10.1: offline implementation validation

Date: 2026-11-06. Scope: task 10.1 only, requirements 6.3-6.5 and 7.1-7.11.
Design references: modal-design.md Overview, Allowed Dependencies, Revalidation
Triggers, Architecture, File Structure Plan, Components and Interfaces, System
Flow and Pilot Bounds, Cost and Error Handling, Testing Strategy, Review and
Rollout. These design headings have no numbered subsections.

## Decision and limits

Offline integration is ready for independent review. This report does not claim
live Modal acceptance or feature-wide GO. Task 10.2 remains unperformed: no paid
staging, builds, deployment, GPU allocation, inference or weight downloads.
Workspace, endpoint, SDK/proxy authentication, provider budget and prior usage
remain operator-configured and unverified. No secrets were inspected.

The changes provide an operator guide, replace obsolete deployment instructions,
link local/provider/cost navigation and repair isolated dependency wiring. Root
runtime dependencies and execution code are unchanged. No tasks were checked off
and no commit was created. Existing untracked local smoke scripts were preserved.

## Task brief and writing structure

Acceptance: documented planning must succeed without SDK/network/cloud mutation;
optional tools must import the real SDK, installed Recall Guard and tokenizer;
coverage must retain auth/redaction, bounds, rejected evidence, drift,
cancellation, remote expiry, owned cleanup and unknown cost semantics. The guide
must distinguish candidate hardware, endpoint evidence, calibration and mitigation,
include account/budget/start/stop/recovery guidance and cite dated primary sources
and Jane Street. The strict documentation build must include the new guide and
cost report in navigation.

COBESY composition blueprint used before prose: the governing idea is to inspect
a bounded plan before spending. Audience: adopter first, operator/engineer next.
Target behavior: run the offline plan and review its blockers before setup.
Inferred hidden assumption: provider support may be mistaken for a verified host
or a mitigation result. Framing: expose unknowns and let operators choose; the
claim fails if actual token alignment, provenance or cleanup fails. The adopter's
first action leads into isolated tools, account/budget prerequisites, separate
staging, approval, then recovery and interpretation. The intended anchor is the
engineer reviewing the first pilot receipt; this is a documentation inference,
not a claim about team culture. Humanizer and composition audits removed
promotional claims, unsupported performance promises and narrative about edits.

## Integration defects reproduced and resolved

Before changes:

```text
uv run --project deploy/modal --no-sync python -I -c "import modal; import recall_guard; from transformers import AutoTokenizer"
exit 1: ModuleNotFoundError: No module named 'recall_guard'
```

The original isolated project installed only Modal. Added an editable dependency
on the root package, Transformers 4.55.4 and Hugging Face Hub 0.34.4, without
installing root development/experiment groups or adding core dependencies.
A real local synthetic-tokenizer smoke then found another missing dependency:

```text
exit 1: ImportError: apply_chat_template requires jinja2 to be installed.
```

Pinned Jinja2 3.1.6 in the isolated project and regenerated its lock. No feature
flag was introduced: only dependency integration and documentation changed.
RED evidence is actual integration failures, not a newly written test before the
dependency edit. The existing lock-contract test was expanded after the defect
was reproduced; no claim of a new test-first product behavior is made.

## Fresh commands and results

All commands ran from the checkout root using PowerShell with login:false.

| Check | Actual result |
| --- | --- |
| `uv sync --project deploy/modal --locked` | Exit 0; 56 locked packages resolved, final Jinja2/MarkupSafe installed; local package installed editable |
| `uv lock --project deploy/modal --check` | Exit 0; 56 packages resolved in 1 ms |
| `uv run --no-sync python -m pytest tests/deploy tests/scripts/test_probe_modal_model.py tests/scripts/test_modal_cost_report.py tests/test_architecture.py -q` | Exit 0; 76 passed in 5.14 s |
| `uv run --no-sync python -m ruff check deploy/modal tests/deploy scripts/probe_modal_model.py scripts/modal_cost_report.py tests/scripts/test_probe_modal_model.py tests/scripts/test_modal_cost_report.py` | Exit 0; All checks passed |
| `uv run --no-sync python scripts/check_architecture.py` | Exit 0; architecture check OK |
| `uv run --no-sync python -m mkdocs build --strict` | Exit 0; Documentation built in 11.00 seconds |
| `git diff --check` | Exit 0; Git line-ending notices only |

The parent ran one full offline regression with
`uv run --no-sync python -m pytest -q`: 1 failed, 2813 passed, 1 skipped,
1 warning in 256.57 s. The sole failure was the obsolete assertion that optional
tooling dependencies equal only `["modal==1.6.3"]`. The focused run above passes
its updated test, which checks the exact isolated pins, local package source,
absence of model frameworks in the lock and absence of hosting dependencies in
core. One interim focused rerun exposed an assertion about uv's trailing slash
in the editable path; that assertion now compares normalized Paths. No second
full suite was run after these dependency/test corrections; this is not a claim
of a final all-green full regression. No product runtime code changed.

Strict docs emits existing third-party MkDocs transition notices and an INFO
for the existing unlisted anthropic-reweighting-prototype result. There is no
unlisted Modal page or broken Modal link; both provider and cost guides are in
navigation. The transition notices did not fail the strict build.

### Real execution boundary smoke

The two documented root dry runs were executed via `runpy.run_path` under
`uv run --no-sync python -`, with their exact CLI arguments and an audit hook
rejecting socket connect, connect_ex, getaddrinfo, sendto and bind operations.
Both returned plan mode and zero inference_posts/cloud_mutations. Modal was
absent from sys.modules. A root `import recall_guard` also left Modal, torch,
vLLM, matplotlib and vectorbt absent. The unresolved runner blockers were:
workspace, endpoint, provider_budget, account_access_unverified,
paid_execution_not_authorized, pinned_tokenizer_unavailable and
remote_weight_cache_unverified.

`uv run --project deploy/modal --no-sync python scripts/probe_modal_model.py`
also returned plan mode, zero inference and zero mutations with the real optional
environment. The guide's isolated import command succeeded.

The final isolated smoke used `uv run --project deploy/modal --no-sync python -I -`
with the same network audit hook. It imported real Modal, Recall Guard,
AutoTokenizer, PreTrainedTokenizerFast and tokenizers, asserted every optional
pin, constructed create_app() and create_expiry_app() without deployment, then
saved a tiny synthetic WordLevel tokenizer with a Jinja chat template in a
TemporaryDirectory using the snapshot layout expected by tokenizer_counts().
The real AutoTokenizer loaded it with local_files_only=True and the production
helper returned eight counts `[6, 8, 9, 7, 9, 9, 7, 6]` and tokenizer hashes.
No torch/vLLM/matplotlib/vectorbt modules were loaded. Exit 0:

```text
real tokenizer local-only synthetic snapshot: PASS [6, 8, 9, 7, 9, 9, 7, 6]
isolated pinned imports, lazy SDK recipes, chat templating: PASS; network operations blocked
```

This proves dependency compatibility for local tokenization, not GPT-OSS tokenizer
or weight loading. No genuine artifact was downloaded. The audit hook permits
local socket.gethostname: an initial overbroad hook rejected that harmless local
platform lookup during SDK import. The corrected hook blocks network operations;
no credentials or SDK deployment calls were required. Real Modal emits Python
3.14 asyncio deprecation warnings; Transformers prints its expected notice that
no model framework is installed.

## Requirement evidence and honest live gaps

| Requirements | Offline evidence | Unperformed live prerequisite |
| --- | --- | --- |
| 6.3, 7.9 | Optional isolated project, installed-package smoke, lean-import checks; candidate/verified documentation | Remote image startup/fit |
| 6.4 | Guide specifies generated-token/alternative alignment, natural-log units, prompt-index row grain and cost/time units | Actual GPT-OSS response representation |
| 6.5 | Jane Street citation and separate completion-only path in README/guide | Mitigation efficacy remains a separate experiment |
| 7.1, 7.10 | Explicit manifest, SDK fixture auth requirement, redaction/allowlist tests, unauthorized example approval record | Actual account/proxy access and denial before GPU startup |
| 7.2 | Runner/evidence regressions reject missing/misaligned probabilities and reasoning mismatch; no replay | Genuine five-alternative logprobs |
| 7.3, 7.11 | Identity drift/unknown snapshots remain unmatched, configured hardware distinguished; calibration unavailable | Observed GPU/runtime and calibration revalidation |
| 7.4, 7.8 | Frozen caps and budget validation tests; subtotal versus unknown total, prior usage/coverage documented | Complete staging/build/storage/control/egress estimate and approved allowance |
| 7.5 | Fake recipe asserts one container/request; actual SDK recipe construction works offline | Actual remote scaling behavior |
| 7.6 | Ownership, deadline, cancellation, absent-client expiry and cleanup failure/timeout regressions | Actual stopped-state receipt and retained storage inventory |
| 7.7 | Accounting tests separate phases, failures, usable count, null costs, credits and append-only later billing | Observed charges and timings |

The guide's manual staging strategy uses pinned Hub APIs, the ignored models/
snapshot layout and Modal Volume.batch_upload/put_file signatures confirmed in
installed Modal 1.6.3. This is a source-checked setup strategy, not an executed
staging procedure. Every indexed shard must be present with nonzero size before
the runner deploys; inventory does not prove content integrity. Ollama GGUF is
not an input to this vLLM recipe. Storage, provenance Dict and expiry app retention
remain explicit. No paid setup or pilot has been authorized by this report.

## Source identity at validation

SHA-256 over working-tree bytes; report itself excluded.

| Path | SHA-256 |
| --- | --- |
| `README.md` | `9cec16a53ce2f6ebda3b73b4dd3e040122e9a173b55ad73d0b4ab2d7fd7ff61b` |
| `docs/modal-provider.md` | `005b121a1839155123bd7a4d0adaff49fcc476e2f85d2b7abf0e45f87a3bb30b` |
| `docs/local-provider.md` | `00f4c66bb46fb86ec193839deeac014cded7cd7c419e4bc4a269c6bb5501a6f7` |
| `docs/modal-cost-report.md` | `9b542381a60810b5bad86b72c3b44b6b2c474a7819bfa649188851f983404a3d` |
| `mkdocs.yml` | `865749da8cda4cdfc5452b82497310103de4dc5d8b9dd9c469874d38f3c73fea` |
| `deploy/modal/README.md` | `6e25ef766fa6496f4e163629da61c424ca8c9cc9a8259a59ed6b0ecafea1e3c3` |
| `deploy/modal/pyproject.toml` | `a0db546f56cdfad0840f0c9892be1fcbe5af3ecc8f0994dbbbe4c0c251d290dc` |
| `deploy/modal/uv.lock` | `f13959f0fa2a7745349fe996438c47ca75ee9365a9eeadf1b932dba7a169bd2e` |
| `deploy/modal/serve.py` | `1e30e69b413411fc46d0179617ae7d3b4493b320a790198bd35c697183f568a7` |
| `deploy/modal/manifest.py` | `7cfff042f2e07988a50d67767c8cf1211518a12c4ff6dd0e303f062d64190ded` |
| `scripts/probe_modal_model.py` | `2a51679907d3ac55df8af9d4f20bafc39090116897548d253a55d63ad8bf2bd3` |
| `scripts/modal_cost_report.py` | `66fb23ce4325939280aa2dc77a91994d34d58ea57d870e7c9ff1a7852fbb7307` |
| `tests/deploy/test_modal_manifest.py` | `bd58230612129ee9fada4e676fb28329be8fda34d0bf8bc73cbab0ea7ef4c2b8` |
| `configs/modal/gpt-oss-20b-pilot.json` | `a5f5cf700ca9a0cc52ecc9b24685bd51c1127d1a032b30e5ac2ed01cc78bc6d0` |
| `pyproject.toml` | `9c76437fd89db2632085b5cfcb65bded194769b3727c4e51d3dd1badb05174a6` |
