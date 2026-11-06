# Local and Modal live feasibility validation

Date: 2026-11-06. Scope: tasks 7.1 and 10.2, with their approved bounded calls.
This is a feasibility decision, not a feature-wide GO or efficacy claim.

## Evidence and decision

| Surface | Physical inference requests | Outcome |
| --- | ---: | --- |
| Ollama, five explicitly selected non-GPT models | 5 | All returned aligned token evidence with five alternatives; four environments matched, LLaVA context differed |
| Modal GPT-OSS-20B, attempts 01-07 | 0 | Startup integration failures retained as historical receipts |
| Two Modal CPU bootstrap/import diagnostics | 0 | First reproduced missing Sentinel; second confirmed the pinned dependency repair; no GPU |
| Modal GPT-OSS-20B, attempt 08 | 1 | Model ready, eligibility response rejected for token text alignment; zero measurement prompts |

All inference used zero retries. Each Ollama model was unloaded and verified
before the next; the final loaded-model list was empty. Final Modal inventory
confirmed all fifteen owned model, expiry and bootstrap app entries stopped.
Retained model storage is unchanged.

Modal requested A100-40GB but observed NVIDIA A100-SXM4-80GB. Stable observed
identity means consistent before/after snapshots, not a demonstrated match to
40 GB hardware. The run establishes no 40 GB fit or attributable GPU price.
The original attempt-07 cleanup receipt remains unverified; the separate final
inventory confirms later shutdown without rewriting that historical observation.

## Fresh checks

On the final runtime source after the image dependency repair:

- `uv run --no-sync python -m pytest tests/deploy tests/scripts/test_probe_modal_model.py tests/scripts/test_modal_cost_report.py tests/scripts/test_probe_local_model.py tests/scripts/test_smoke_local_models.py -q`:
  **153 passed, 1 skipped**, 14.87 seconds. The skip is the optional SDK module
  in the root environment without Modal.
- Actual Modal 1.6.3 SDK tests separately ran under the optional Python 3.12
  operator environment: **10 passed**, networking blocked. Independent repair
  review confirmed actual SDK entrypoint/pip operations and untouched raw-class
  serialization.
- Scoped Ruff and `scripts/check_architecture.py`: passed.
- `uv run --no-sync python -m mkdocs build --strict --site-dir runs/modal-preflight/docs-final`:
  passed in 21.59 seconds.

Task 7.1's tool, live outcome and public numeric receipt passed independent
review. Entrypoint, dependency, endpoint identity and CPU diagnostic repairs
also passed independent review before their operator steps. The final Modal
outcome/publication review returned **APPROVED**, independently verifying all
eight attempt hashes, public fields, hardware, provisional billing and final
cleanup. Tasks 7.1 and 10.2 recorded bounded feasibility work; the original
attempt-08 response remains excluded from scoring. The subsequent task 10.3
amendment below corrects the interpretation of its integration defect.

## Practical limits

No local GPT, NIM, Opus, NPU, calibration, ensemble or divergence efficacy run was
performed in that original scope. Attempt 08's response was excluded, not evidence
that logprobs are absent or that a particular reasoning-channel defect caused
the mismatch. Generated response text was intentionally not retained.

Public receipts and source hashes are in `docs/results/2026-11-06-local-model-smoke.json`
and `docs/results/2026-11-06-modal-pilot.json`. Private ignored originals remain
under `runs/local-provider/2026-11-06-smoke/` and `runs/modal-preflight/`.
The latest workspace gross USD 0.15792723/billed USD 0 snapshot is provisional,
workspace-wide and not an attributable pilot invoice. Full cost and cost per
usable measurement remain unavailable.

The user subsequently authorized `modal-alignment-diagnostic-proposal.md` and
task 10.3. Conditional NPU prerequisites remain unresolved; none of this work
grants feature-wide GO or authorizes a larger measurement batch.

## Task 10.3: response-handling repair and retest

The pinned vLLM contract and response 09 confirmed an integration bug: probabilities
covered the full Harmony generation while the adapter attached them to final-only
text. Sampling across requests is irrelevant to this same-response invariant.
The opt-in adapter now maps an explicit ID-derived final-message span, retaining
original logprobs and alternatives; generic handling remains unchanged.

- Live diagnostic: one eligibility POST, zero retries and zero measurement
  prompts; response body retained privately for replay without headers or secrets.
- Production adapter replay with HTTP disabled: original handling reproduces the
  failure; repaired handling passes strict five-alternative validation. Original
  final span `[28, 29)` of 30 positions and its values were preserved exactly.
- Focused regressions: **159 passed**. Independent code/publication review:
  **APPROVED**, with no unresolved findings.
- Full offline regression: **2888 passed, 2 skipped**, 369.28 seconds. One notebook
  event-loop fallback warning did not fail the run.
  A targeted skip audit confirmed the optional Modal SDK is absent from the root
  environment and the existing symlink test lacks host privilege; 126 related
  tests passed with those same two skips. No host changes were attempted.
- Ruff and architecture: passed. Strict docs: passed in 19.31 seconds.
- Runner verified model shutdown; final inventory confirmed all twelve currently
  observed owned app entries stopped. Older stopped entries aged out of inventory.

Response 09 observed A100-SXM4-40GB, unlike attempt 08's 80 GB assignment. The new
provisional workspace snapshot is gross USD 0.29792825/billed USD 0, not an
attributable experiment invoice. Public counts, hashes and replay outcomes are in
`docs/results/2026-11-06-gpt-oss-response-mapping.json` and its companion report.
Replay is not a further generation after the fix. Calibration and mitigation
efficacy remain unmeasured.
