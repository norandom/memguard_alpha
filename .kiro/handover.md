# Handover: memorization-adjusted finance simulation, implementation in progress

Updated: 2026-11-09 (end of day). Workspace: `D:\Source\memguard_alpha`, branch `main`.
GitHub: `norandom/recall_guard` (the repository was renamed on 2026-11-10; the
local folder name and the Dagger `/mnt/d/Source/memguard_alpha` path are unchanged).
Last commit: `3148843` (task 4.3). Working tree: task 5.1 in progress, uncommitted
(see "Task 5.1 state" below).

## Start here

Active spec: [memorization-adjusted-finance-sim](specs/memorization-adjusted-finance-sim/)
(`spec.json` phase `tasks-generated`, all approvals true). Progress lives in its
[tasks.md](specs/memorization-adjusted-finance-sim/tasks.md): 15 of 41 sub-tasks
are `[x]` (groups 1 to 4 complete, 3.5 included). The `## Implementation Notes`
section at the bottom of tasks.md is the running memory for later tasks; read it
before dispatching anything.

Resume with `/kiro-impl memorization-adjusted-finance-sim` after settling the 5.1
working tree (next section). The earlier [1.0 release plan](specs/release-1.0/tasks.md)
remains the tracker for Phase B and the parked items (P1 to P7); nothing there
changed today.

## Task 5.1 state (uncommitted, unreviewed)

An Opus implementer was still running when the session ended. It had written:

- `recall_guard/harness/guarded_decision.py` (new)
- `tests/harness/test_guarded_decision.py` (new)
- `recall_guard/harness/scorer.py` and `recall_guard/harness/scorer_client.py`
  (modified: intended to add only two read-only properties `mcs` and
  `control_baseline` on `MemoryGuardedScorer`, and `identity_settings()` in
  scorer_client; nothing else may change there)

Its status report was not received before the handover, so treat the files as a
draft. Inspect before deciding:

1. `git status` and `git diff recall_guard/harness/scorer.py recall_guard/harness/scorer_client.py`.
2. `uv run --no-sync pytest -q tests/harness/test_guarded_decision.py tests/harness/test_scorer.py tests/harness/test_profile_calibration.py -p no:cacheprovider`
   and look for a leftover feature flag (`grep -n "_ENABLED\|_require_flag" recall_guard/harness/guarded_decision.py`).
3. If the tests pass, the flag is gone and the diff to scorer*.py is the two
   properties plus the helper only: dispatch a kiro-review reviewer for 5.1 against
   the design's GuardedDecisionApi bullets, then commit on APPROVED (the parent
   commits; stage explicit paths; never `git add -A`).
4. Otherwise discard the draft and re-run 5.1 through kiro-impl: restore the two
   modified files with `git checkout -- recall_guard/harness/scorer.py recall_guard/harness/scorer_client.py`
   and delete the two untracked files, after looking at them.

Logs for the draft, if any, are under `runs/finance-sim/validation/task-5.1-*.log`.

## Decisions taken today (user)

- **Validation route:** tests, ruff and the architecture check run natively on
  Windows with `uv run --no-sync ...` from PowerShell. Dagger (WSL Debian,
  `/home/mc/.local/bin/dagger -m ci call ...`) is the Linux CI gate only, for
  parity when a Linux-specific behaviour needs confirming; never the native
  `dagger.exe` (no engine). Recorded in `.kiro/steering/execution-environment.md`
  (2026-11-09 paragraph), README "Windows", `docs/development.md`, tasks.md notes.
  Native baseline: 3684 passed, 20 skipped; after 4.3: 3918 passed, 20 skipped.
  Windows-only skips: `tests/scripts/test_modal_operator.py` (18, needs a POSIX
  `sh` on PATH; Git for Windows ships one under `usr\bin`),
  `tests/deploy/test_modal_sdk_serialization.py` (modal SDK not in the root env),
  one symlink test in `tests/harness/test_intervention_report.py`.
- **Mixing platforms is fine**; statistical soundness is the invariant. Handle
  Linux-only parts specifically; do not containerise everything.
- **Implementer subagents run on Opus at xhigh effort**; reviewers stay on the
  session default.
- **Serving arm model:** Gemma 3 4B on Modal (`google/gemma-3-4b-it`, revision
  `093f9f388b31de276ce2de164bdc2081324b9767`, gated, HF_TOKEN by name), generic
  mapping, no pilot or power gate (it reports achieved power), variants
  no-adjustment and top-k only. Reviewed ceiling USD 34.34 for the whole arm
  (`configs/modal/gemma-3-4b-measurement.json`); nothing has been spent.
- **Negative control calibrator:** an EDGAR calibrator per model and source on the
  EDGAR calibration split (40 memorized, 100 control); the macro calibrator is
  never applied to EDGAR prompts. The serving arm also runs the negative control.
- **Scored probability for Brier and log loss:** share of draws with a positive
  defensive spread (0 or 1 for a single draw), clipped to [0.01, 0.99], shrunk
  toward 0.5 by `1 - p_memorized`.
- **Modal serving route is in scope** (3.5 done): per-manifest weight volume,
  CPU-only staging, `stage | serve | stop-serve` in `scripts/modal_operator.sh`,
  stage limits from a JSON file (7.3 writes them from the frozen plan).

## Committed today (main)

`7b16ca3` Dagger upload excludes `tools/` (6m31s -> 3.6s per call) ·
`8df3c62` `--paths` for focused Dagger tests · `820b365` 1.1 · `828b275` 1.2 ·
`9c795ca` 2.1 · `c2626ae` 2.2 · `993d573` 2.3 · `ecd81b2` 2.4 · `fcf1108` 2.4
follow-up (unbounded half-ensemble ratios counted against the ensemble) ·
`4c0cf9a` 2.5 · `7346ee7` 3.1 · `5616081` 3.2 · `2eea39b` 3.3 · `6d80c7a` 3.4 ·
`a2538fd` native dev loop · `aba69c6` 3.5 · `0c247e7` 4.1 · `af1e209` research
note on logits as model fingerprints (R42) · `5d515c9` 4.2 · `3148843` 4.3.

Every task went through a fresh implementer plus an independent kiro-review
reviewer; remediation rounds and the reasons are in the commit messages.

## Design preparation already done

design.md was amended, per task, with concrete contracts so implementers do not
invent APIs: DecisionDefinition (parse_response, parse_json_components),
DistributionSummary (RecordedDraw, three-valued `unresolved`, one-sided 95 percent
agreement interval), Significance (Interval, TestResult, dropped_resamples,
half-ensemble unbounded rule), FullDistributionClient (RequestError codes, ASCII
credentials), EstimateSources (SourcedFeatures, scope of "full distribution" =
kept top-100 alternatives), CalibrationIdentity (estimate_source, corpus_sha256,
mismatches/require_match), profiles (FullDistributionHostConfig, ports
11570/11571 and 11580/11581, gateway = loopback + 1, Modal fallback), Serving
route (StageLimits, stage/serve/stop-serve), FredLoader, MacroRegimeExample
(adopter-identical prompt and EXPOSURE matrix, nine-decimal sign rule),
EDGAR outlook, GuardedDecisionApi (SourceCalibrator, GuardIdentity, GuardedRun,
decide_from_draws, identity_settings without seed keys), FullDistributionHost
(6.1: reuse `scripts/score_documents_llamacpp.py` bindings via a shared
`scripts/llamacpp_native.py`; exactness receipt against llama-server, one model
resident at a time) and the 6.2/6.3 lifecycle scripts (siblings of
`scripts/serve_local_flm.ps1`, two-load RAM gate, Modal alternative on refusal).
Tasks 5.2, 6.x and 7.x onward have no further design gaps known; 7.x contracts
build on 5.1's final shape.

## Prerequisites and pins (from the 1.2 receipt, private)

`runs/finance-sim/prerequisites/receipt.json`: 18 of 18 pass. FRED_API_KEY is set
in the root `.env` (added today). FMP covers SWDA.L, XLK, IAU 2016-01 to 2026-07;
DTB3 the same span; CPIAUCSL point in time; T10Y2Y and BAA10Y vintages from
2014-01-27. Gemma 3 4B GGUF
`sha256:f06e91e18fc74295d0c781bf91622508c75655e2c18ed869d52447ffe1e667e1`
(q4_K_M, `models/blobs`), llama.cpp b12199 Vulkan. Cutoff 2024-08-31 in
`data/cutoffs.yaml`. Node v24; the dataviz palette validator sits in a
version-specific Claude Code temp cache (path and sha256 in the receipt; 8.1 must
re-locate it). FMP key name is `FMP_API_TOKEN`.

## Open follow-ups outside this spec (recorded in tasks.md notes)

- `recall_guard/core/providers.py` validates credentials with `.strip()` only; the
  newline and non-Latin-1 bearer leaks closed in `core/full_distribution.py` apply
  there too.
- `recall_guard/core/consensus.py` `detect_multimodal` docstring says thin clusters
  return None; `_search_split` returns `_unseparated()` (separated=False).
- `intervention_metrics.py` dedups `provenance.scoring` by `asdict(identity)`; a run
  resumed across the 3.2 change lists the same calibrator twice.
- Optional: put Git's `usr\bin` on the test PATH so the operator-script tests run
  natively; tighten `fred._API_KEY` to `[a-z0-9]{32}`; make `MacroRow` mappings
  read-only.

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

- 9.2: the power-gate outcome on the local pilot (continue, widen the window, or
  more draws).
- 9.4: the serving-arm approval record against the frozen stage limits (USD
  ceiling 34.34; staging, smoke run, calibration, full run, negative control).
- Phase B of release-1.0 (about USD 7) and the parked items P1 (Mac GPT-OSS),
  P6 (Mistral), P7 (Dependabot) are unchanged.
