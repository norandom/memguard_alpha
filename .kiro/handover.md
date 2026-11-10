# Handover: memorization-adjusted finance simulation, implementation in progress

Updated: 2026-11-10 (restart). Workspace: `D:\Source\memguard_alpha`, branch `main`.
GitHub: `norandom/recall_guard` (renamed from `memguard_alpha` on 2026-11-10; the
local folder name and the Dagger path `/mnt/d/Source/memguard_alpha` are unchanged;
the old URL redirects; Pages publishes at <https://norandom.github.io/recall_guard/>).
Last commit: `cd7a0f4` (repository rename). Working tree: task 5.1 reviewed and
APPROVED but not committed; see "Do this first".

## Start here

Active spec: [memorization-adjusted-finance-sim](specs/memorization-adjusted-finance-sim/)
(`spec.json` phase `tasks-generated`, all approvals true). Progress lives in its
[tasks.md](specs/memorization-adjusted-finance-sim/tasks.md): 15 of 41 sub-tasks
are `[x]` (groups 1 to 4 complete, 3.5 included); 5.1 is done but unmarked. The
`## Implementation Notes` section at the bottom of tasks.md is the running memory
for later tasks; read it before dispatching anything.

Resume with `/kiro-impl memorization-adjusted-finance-sim` after "Do this first".
The earlier [1.0 release plan](specs/release-1.0/tasks.md) remains the tracker for
Phase B and the parked items (P1 to P7); nothing there changed.

## Do this first

1. **Done on 2026-11-10: containers run natively through Podman Desktop.**
   `docker` and `docker compose` are Podman 5.8.3's Docker-compatible CLI and
   Compose provider (Compose is the user's preferred route:
   `docker compose run --rm dev uv run --no-sync ...`); `dagger` v0.21.11 is a
   native CLI whose engine runs as a Podman container; WSL is no longer used. The
   steering and README say so. Re-verify `run-with-secrets` on the Podman engine
   before the next credentialed run. The Windows CI test fixes were applied on
   2026-11-10 (see "Open follow-ups").
2. **Finish task 5.1 (ensemble-first guarded decision).** State in the tree:
   - new: `recall_guard/harness/guarded_decision.py` (644 lines),
     `tests/harness/test_guarded_decision.py` (712 lines, 54 tests);
   - modified: `recall_guard/harness/scorer.py` (+10: read-only properties `mcs`,
     `control_baseline`), `recall_guard/harness/scorer_client.py` (+25:
     `SEED_SETTINGS`, `_plain_settings`, `identity_settings`); nothing else there;
   - no feature-flag residue; RED evidence in
     `runs/finance-sim/validation/task-5.1-red.log`.
   - Independent review (kiro-review, 2026-11-10): **APPROVED**, no remediation.
     Verified: no direction-parser estimates; SINGLE never gated or zeroed; `None`
     components never reach `adjust`; not-assessed components exposed as
     `GuardedDecision.not_assessed`; `p_memorized` over the same estimate set as
     `summarize_distribution`; consensus = per-component median with `None` where
     `unresolved is True`; identity check before any request with seed keys
     excluded; at most `draws` requests, no retries; `decide_from_draws` issues no
     request and equals the live result (nine scenarios).
   - Gate: full native suite `runs/finance-sim/validation/task-5.1-parent-full.log`:
     3,971 passed, 20 skipped, 1 failed. The failure is
     `tests/scripts/test_npu_phase_a.py::test_gateway_synthesizes_installed_only_tags_with_catalog_digest`,
     a host flake (see "Open follow-ups"), not 5.1's code; focused 5.1 runs pass
     (117 passed, `task-5.1-parent-focused.log`, `task-5.1-review-tests.log`).
   - To commit: `git add recall_guard/harness/guarded_decision.py tests/harness/test_guarded_decision.py recall_guard/harness/scorer.py recall_guard/harness/scorer_client.py .kiro/specs/memorization-adjusted-finance-sim/design.md .kiro/specs/memorization-adjusted-finance-sim/tasks.md`
     (explicit paths only; never `git add -A`; `.openidea/` stays untracked), mark
     5.1 `[x]`, message `feat(memorization-adjusted-finance-sim): ensemble-first
     guarded decision` with the gate stated honestly, push.
   - Before committing, append the reviewer's forward notes to tasks.md
     Implementation Notes:
     - `draws_sha256` is `canonical_draw_hash` over the parsed draws'
       `response_sha256` values (the record holds no text), so it is not comparable
       with an `EnsembleResult.draws_sha256` over content; 7.4 and later use the
       same convention.
     - `SourceCalibrator.from_scorer` relabels the fitted identity's source and
       corpus hash from the caller's arguments without comparison; 7.6 must bind
       the source at fit time (one calibrator per model and source), or
       `from_scorer` should refuse a relabel of a non-default source.
     - `_estimate` records the configured `SourceLabel.k`, not an observed one; k
       is bound through `top_logprobs` in `parameters_json` (3.2 note).
     - The guard checks identity once before the loop and does not compare
       `response.confirmed_settings` or the observed local identity; a note for
       7.7 on local hosts.
     - `GuardedDecision`, `GuardIdentity` and `GuardedRun` have no `to_dict`;
       `dataclasses.asdict` fails on mappingproxy fields; `RecordedDraw.to_dict`
       is the only serialiser, so 7.4 needs one for the result.
     - `_request` records any non-fatal exception as `request_error:<code>`
       (as `ensemble._safe_draw` does); 401/403 and `BudgetExhaustedError` stop
       the loop. Failed draws carry `response_sha256 = sha256("")`.
3. **Then dispatch 5.2** (public exports and the installed-wheel test). Its design
   is already amended in `design.md` ("Modified Files" > `recall_guard/__init__.py`
   and `tests/test_installed_wheel.py`): the exact root export list and the wheel
   test mechanics (`uv build --wheel`, `uv venv` under `tmp_path`, `uv pip install`,
   run with `-I` and `cwd=tmp_path`, assert no matplotlib and no research folders;
   skip without `uv`). The existing lean-import test is
   `tests/harness/test_scorer.py::test_import_recall_guard_stays_lean`.

## Validation route (user decisions 2026-11-09)

- Tests, ruff and the architecture check run natively on Windows from PowerShell
  in the locked environment: `uv run --no-sync pytest -q <paths> -p no:cacheprovider`,
  `uv run --no-sync ruff check .`, `uv run --no-sync python scripts/check_architecture.py`.
  Full suite about 6 to 7 minutes; run it once per task, after the flag is removed.
  Logs under `runs/finance-sim/validation/`.
- Dagger is the Linux CI gate only, not the test runner; the user is moving it to
  Podman (see "Do this first"). Never the native Windows `dagger.exe`.
- Mixing platforms is fine; statistical soundness is the invariant. Handle
  Linux-only or Windows-only parts specifically (skip with a reason), do not
  containerise everything.
- Implementer subagents run on Opus at xhigh effort; reviewers on the session
  default. Every task: fresh implementer, independent kiro-review reviewer,
  kiro-verify-completion with fresh evidence, then the parent commits.
- Windows-native skips (baseline): `tests/scripts/test_modal_operator.py` (18,
  needs a POSIX `sh` on PATH), `tests/deploy/test_modal_sdk_serialization.py`
  (modal SDK not in the root env), one symlink test in
  `tests/harness/test_intervention_report.py`.

## Decisions carried over (user)

- Serving arm: Gemma 3 4B on Modal (`google/gemma-3-4b-it`, revision
  `093f9f388b31de276ce2de164bdc2081324b9767`, gated, HF_TOKEN by name), generic
  mapping, no pilot or power gate (reports achieved power), variants no-adjustment
  and top-k only. Reviewed ceiling USD 34.34 for the whole arm
  (`configs/modal/gemma-3-4b-measurement.json`); nothing has been spent.
- Negative control: an EDGAR calibrator per model and source on the EDGAR
  calibration split (40 memorized, 100 control); the macro calibrator is never
  applied to EDGAR prompts; the serving arm also runs the negative control.
- Scored probability for Brier and log loss: share of draws with a positive
  defensive spread (0 or 1 for a single draw), clipped to [0.01, 0.99], shrunk
  toward 0.5 by `1 - p_memorized`.
- Modal serving route in scope (3.5 done): per-manifest weight volume, CPU-only
  staging, `stage | serve | stop-serve` in `scripts/modal_operator.sh`, stage limits
  from a JSON file that 7.3 writes from the frozen plan; serve arms the auto-stop at
  the stage limit, never the manifest ceiling.
- Macro example: adopter-identical prompts and `EXPOSURE` matrix from
  Global_Macro_AI_Factors, BAA10Y for the discontinued HY OAS, DTB3 as the cash
  leg, sign tests on `round(spread, 9)`.

## Committed since 2026-11-09

`7b16ca3` Dagger upload excludes `tools/` · `8df3c62` `--paths` for focused Dagger
tests · `820b365` 1.1 · `828b275` 1.2 · `9c795ca` 2.1 · `c2626ae` 2.2 · `993d573`
2.3 · `ecd81b2` 2.4 · `fcf1108` 2.4 follow-up · `4c0cf9a` 2.5 · `7346ee7` 3.1 ·
`5616081` 3.2 · `2eea39b` 3.3 · `6d80c7a` 3.4 · `a2538fd` native dev loop ·
`aba69c6` 3.5 · `0c247e7` 4.1 · `af1e209` research note R42 (logits as model
fingerprints) · `5d515c9` 4.2 · `3148843` 4.3 · `6a7a72c` handover · `cd7a0f4`
repository rename.

## Design preparation already done

design.md carries concrete contracts for every task through the host group, so
implementers do not invent APIs: DecisionDefinition, DistributionSummary
(three-valued `unresolved`), Significance (Interval, TestResult,
dropped_resamples, half-ensemble unbounded rule), FullDistributionClient,
EstimateSources (scope of "full distribution" = kept top-100 alternatives),
CalibrationIdentity extension, profiles (ports 11570/11571 and 11580/11581,
gateway = loopback + 1, Modal fallback), serving route (StageLimits), FredLoader,
MacroRegimeExample, EDGAR outlook, GuardedDecisionApi, root exports and the
installed-wheel test (5.2), FullDistributionHost (6.1: shared
`scripts/llamacpp_native.py` bindings, exactness receipt against llama-server with
one model resident at a time) and the 6.2/6.3 lifecycle scripts (siblings of
`scripts/serve_local_flm.ps1`, two-load RAM gate, Modal alternative on refusal).
7.x contracts build on 5.1's committed shape.

## Prerequisites and pins (from the 1.2 receipt, private)

`runs/finance-sim/prerequisites/receipt.json`: 18 of 18 pass. FRED_API_KEY is set
in the root `.env`. FMP covers SWDA.L, XLK, IAU 2016-01 to 2026-07; DTB3 the same
span; CPIAUCSL point in time; T10Y2Y and BAA10Y vintages from 2014-01-27. Gemma 3
4B GGUF `sha256:f06e91e18fc74295d0c781bf91622508c75655e2c18ed869d52447ffe1e667e1`
(q4_K_M, `models/blobs`), llama.cpp b12199 Vulkan. Cutoff 2024-08-31 in
`data/cutoffs.yaml`. Node v24; the dataviz palette validator sits in a
version-specific Claude Code temp cache (path and sha256 in the receipt; 8.1 must
re-locate it). FMP key name is `FMP_API_TOKEN`. The real `.env` still carries
`SEC_USER_AGENT=memguard_alpha research ...`; `.env.sample` now says
`recall_guard`; change the real one by hand if wanted.

## Open follow-ups

- **Windows CI (fixed 2026-11-10 on the user's direction):** the POSIX operator
  tool is not ported to Windows; it is tested in the Linux container, PowerShell
  pieces on Windows. `tests/scripts/test_modal_operator.py` is marked Linux-only
  (Compose parity: 18 passed), and the FLM gateway test in
  `tests/scripts/test_npu_phase_a.py` skips without the untracked `models/flm`
  store and retries one `pwsh` launch refused by the Store alias. CI run
  42787332959 on `7260deb`: all five jobs green (Windows py3.12 and py3.14, Linux
  test py3.12 and py3.14, lint).
- **Host flake (2026-11-10):** Python launching the Store-installed `pwsh`
  app-execution alias (`%LOCALAPPDATA%\Microsoft\WindowsApps\...\pwsh.exe`)
  intermittently fails with `PermissionError: [WinError 5]` at CreateProcess
  (reproduced outside pytest: first of five launches denied, the rest fine;
  `powershell.exe` and `cmd` unaffected). It fails the FLM gateway test above
  locally. Options: install PowerShell 7 to Program Files ahead of WindowsApps on
  PATH, or let 6.2/6.3 tests prefer an installed `pwsh.exe` or retry one denied
  launch.
- Upstream, outside this spec (recorded in tasks.md notes):
  `recall_guard/core/providers.py` credential validation (newline and
  non-Latin-1 bearer leaks closed in `core/full_distribution.py` apply there);
  `consensus.py` `detect_multimodal` docstring vs `_search_split`;
  `intervention_metrics.py` dedup of `asdict(identity)` across the 3.2 change.
- Optional: put Git's `usr\bin` on the test PATH so the operator-script tests
  run natively; tighten `fred._API_KEY` to `[a-z0-9]{32}`; make `MacroRow`
  mappings read-only.

## Rules in effect

Credentials by name only, never printed or stored in receipts or provider stores;
`runs/` receipts and per-token scores are private; licensed raw data (BAA10Y, FMP,
ICE) is never published, only aggregates with citation; `.openidea/` stays
untracked; temp files under `D:\BuildArtifacts`; model hosts run natively on
Windows (iGPU, NPU), WSL has no hardware access; Modal must be clean after runs
(verified stop) and only runs after a user approval record (9.4); every research
result is a dated report on `docs/results/template.md`, registered in
`docs/research-register.md` and linked from MkDocs (strict build before committing
docs). Commit messages end with the Co-Authored-By line for the active model.

## Human decisions still ahead

- **Full-distribution evidence at study scale (raised 2026-11-10 after 6.1).** The
  host teacher-forces prompt plus response one token per decode at 15 to 17
  tokens/s on the iGPU: about 33 s per macro draw (534 decodes) and 26 s per EDGAR
  draw. The local sampling arm needs evidence for 8,064 draws, so per-draw
  requests would take about 67 hours, against the 13 to 15 hour estimate that
  covered generation only. Proposal: a follow-up task (6.4) adding a batch mode to
  the host, `POST /v1/full_distribution_batch {prompt_text, response_texts[]}`,
  that evaluates the prompt once per item and rendering (batched prompt
  processing), then teacher-forces each response against the cached prefix and
  rolls the KV cache back to the prompt length after each (`llama_memory_seq_rm`),
  recording only response positions. Estimate: about 100 s per item and rendering
  (252 of them), about 7 hours for the evidence pass; generation stays separate
  and sequential on the GPU. The exactness check must be re-run in batch mode
  before use. Alternatives: fewer draws for the full-distribution variants (breaks
  the pairing of 8.1) or evidence for a subset of items (weakens 9.x). Needs the
  user's decision and a tasks.md amendment.
- 9.2: the power-gate outcome on the local pilot (continue, widen the window, or
  more draws).
- 9.4: the serving-arm approval record against the frozen stage limits (USD
  ceiling 34.34; staging, smoke run, calibration, full run, negative control).
- Phase B of release-1.0 (about USD 7) and the parked items P1 (Mac GPT-OSS),
  P6 (Mistral), P7 (Dependabot) are unchanged.
