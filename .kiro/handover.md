# Handover: memorization-adjusted-finance-sim

State as of 2026-11-11, about 07:00 (system clock). Repository `norandom/recall_guard` (local folder
`D:\Source\memguard_alpha`, branch `main`). The overnight live run authorized on 2026-11-10 ("Yes, run it
all except Modal") is complete. CI has been green on every commit since the 8.3c fix (`9f8a75a`); check the
latest pushes with `gh run list --limit 5` (strip ANSI codes before printing).

## Where the spec stands

- 40 of 42 sub-tasks are `[x]`: groups 1 to 8, 9.1 (runner), 9.2 and 9.3 (the live probe arm, accepted on
  the receipts by an independent read-only review on 2026-11-11), plus the follow-ups 7.3b, 8.3c, 9.2a
  (parser fence fix) and 9.3a (DTB3 advisory in the licensing guard).
- Open: 9.4 (the Modal serving arm; runs only with the user's approval record and the manifest endpoint set)
  and 9.5 (the dated report). 9.5 exists as a DRAFT labelled "serving arm pending": report R43 at
  `docs/results/2026-11-11-memorization-adjusted-finance-sim.md`, registered, in the nav and the reports
  index, with the 26 published assets under `docs/assets/finance-sim/`. The final 9.5 after the serving arm
  must also update the synthesis `docs/research-report.md` (the task bullet names it; the draft left it).
- `tasks.md` ends with "## Implementation Notes": one entry per task and per decision. Read the entries from
  "Overnight authorization" onward before touching the live outputs. `design.md` carries dated amendment
  paragraphs ("decision 2026-11-10", "Amendment after ...", "fix 9.3a") that override older text.

## What ran overnight (probe arm, native, one host at a time)

- Plan frozen 2026-11-10 21:35 under `runs/finance-sim/plan/` (hash
  `f7603eeb8efdbf74645d658f2bd733a35b3834ba7c90b74d00c6ac2e6b5332fe`, seed 20261110, 126 months, probe
  subset 20 months: 15 pre and 5 post cutoff, keys by name, receipt without values).
- Calibration attempt 1 (71 min) failed: every draw a parse error because Gemma 3 4B fences its JSON. Fix
  9.2a (`73cb8e4`) unwraps one fence. The attempt is archived under
  `records/sampling/calibration.attempt1-parse-error-20261111-0001/` with a NOTE.md: its receipt wrongly
  says "complete" (runner follow-up 9.1b).
- Calibration attempt 2 complete 02:56 (944 of 944 requests; 181 usable prompts in the evidence pass).
  Calibrators under `runs/finance-sim/calibrators/sampling/`: held-out AUC macro top_k 0.72 (0.59 to 0.86),
  macro full_distribution 1.00 (the features separate the two renderings outright), EDGAR top_k 0.59 and
  full 0.53 (both flagged weak).
- Pilot complete 03:32 (192 of 192 requests, 23 evidence batches, 5 of 12 months usable). Gate record under
  `runs/finance-sim/gates/<hash>.json`: expected power 0.111, decision stop_for_decision, policy inform (the
  frozen `pilot.gate_policy`; the plan's `pilot.rule` text is stale and the policy governs). The full run
  printed the gate text before its first request.
- Full run complete 04:25 (320 of 320 requests, 33 evidence batches, 40 of 40 results). Negative control
  complete 04:56 (320 of 320, 40 batches, 40 results). Every receipt: host checks ok, teardown verified
  against the captured stop outputs, no unknown outcomes. Host time about 4 h 40 min in all; no stage
  needed a restart, so the live resume path was not exercised.
- Parse failures concentrate on the macro prompts: about 28 percent of macro draws in calibration and the
  pilot, 42 percent in the full run (186 of 320 parsed), 0 percent on EDGAR. They cost 11 of the 20 probe
  months (a value missing in one rendering) and 7 of the 12 pilot months (6 below the minimum, 1 with an
  unresolved component).
- Analyses (`run_study analyses --arm sampling --plan-sha256 <hash>`, 12 files under
  `runs/finance-sim/analysis/`): accuracy gap difference 0 (p 1.0) on the 9 usable months, ensemble
  variants missing below the block length of 6; Brier and log loss lower for all four adjusted variants
  (Holm p 0.004 each) with the caveat that shrinking an over-confident forecast toward 0.5 lowers these
  losses when raw accuracy is near chance; achieved power missing; half-ensemble variance ratio 0.72 for the
  top_k decision (Holm p 0.003); first-draw rate 0.41; random-answerer null p 0.90; negative control "no
  material change" by construction (both EDGAR calibrators weak, adjustment skipped 40 of 40); convergence
  40 items and 186 draws, loss, min_k and zlib exact at every k, min_k_pp never agreeing, the estimate
  agreement improving with k.
- Replay with `--require-guard` first refused on 10 DTB3 matches (runs of short public rates coinciding
  with SVG style numbers, glyph scale factors and reliability bin edges; no licensed probe matched). Fix
  9.3a (`6d9927d`) made DTB3 advisory (`guard.warnings` in replay.json); the re-run published 26 files with
  0 licensed leaks and 6 warnings (`runs/finance-sim/validation/live-9.3-replay-2.log`), the committed copy
  compares byte for byte (`live-9.5-replay-compare.log`), and the strict docs build passes.

## Do this first in a new session

1. `git status` (nothing of the spec should be uncommitted; `.openidea/` stays untracked); check CI with
   `gh run list --limit 6` (strip ANSI codes before printing).
2. Reproduce the published assets at any time, no host needed:
   `uv run --no-sync python -m scripts.finance_sim.run_study replay --plan-sha256 <hash> --out-root D:\BuildArtifacts\memguard_alpha\replay-<date> --require-guard --compare docs\assets\finance-sim`
   (exit 0 and a byte-for-byte comparison are the pass; the 6 DTB3 warnings are expected).
3. 9.4 only with the user's written approval record (task 7.5 format, matching the frozen plan) and the
   manifest endpoint set before presenting the limits; frozen serving-arm stages USD 32.54 within the 34.34
   ceiling; stop with the operator script and verify no Modal app remains.
4. Then the final 9.5: replay on both arms, replace the draft's "serving arm pending" labels, add the
   achieved power per arm, update `docs/research-report.md`, re-run the strict build.

## Follow-ups recorded by the reviewers (none blocks acceptance)

- 9.1b is done (`2c357ae`): a refused or failed pass is recorded as such and the receipt counts the evidence
  batches; the generation session's own counters still stay 0 on the runner's path (the runner line carries
  them), and a failed exception still propagates with its traceback.
- `dtb3_probes` keeps duplicate runs, which inflates `guard.warnings` counts; apply `_distinct` as
  `fmp_probes` does (changes the DTB3 probe count that one test pins).
- Keep a bounded head of the raw reply for failed draws (today a failed draw keeps no reply text, which is
  why attempt 1 had to be re-run from scratch to diagnose it); a public alias for `_probe_after_failure`;
  a sampling-arm partial-path test; a symbol-set check in `check_guard_inputs`; promote the operator driver
  (`scratchpad\live\run_stage.ps1`) into `scripts/finance_sim` if the probe arm is ever re-run.
- The register index has no row for the existing R42 entry (pre-existing gap); the synthesis Evidence line
  still says R01 to R41.
- Line endings: the repository has core.autocrlf without a .gitattributes, so a fresh Windows checkout
  rewrites the published assets with CRLF and the replay `--compare` and the page hashes no longer match
  the working files (the blobs are right); add a .gitattributes rule for docs/assets/finance-sim before the
  final 9.5.
- If the probe arm is re-run: the parse failures (bounds and extra keys in the model's JSON) are the lever;
  a stricter response schema or a larger model would recover most of the lost months.
- The real `.env` `SEC_USER_AGENT` still says memguard_alpha (user to change by hand).

## Decisions (user; all in design.md and the notes)

- Execution: PowerShell for host commands (never Git Bash); native uv for the unit-test loop; Compose for
  containerised runs (unusable for the live stages because the dev image bakes the code, so the stages ran
  natively, recorded as a decision); Dagger as the CI gate; never stop the Podman machine or WSL; one model
  load at a time, gate on one load plus headroom (6 GiB), batch memory-heavy steps.
- Study sizing: the Modal serving arm is primary (none and top-k on the full evaluation set), approved only
  after the local probe results; the local arm is a probe arm (8 draws, calibration 944 requests, 20-month
  subset, negative control 40 rows). "We don't have 44h."
- Frozen in the plan: each arm's GGUF digest, fit settings (0.6, 50), evidence settings (600 s, 100,
  5 retries, 60 s), arm roles, `pilot.gate_policy = "inform"` (the gate cannot pass at effect 0.10 with
  month-level accuracy differences; it informs and the probe arm proceeds; the serving arm is ungated).
- Research defaults kept: effect size 0.10, negative-control tolerance 0.10 with the three-outcome rule;
  correctness scored on the raw sign when shrinking rounds a spread to zero.
- Licensing: BAA10Y levels, realized returns, FMP closes and reply text never published and always refused
  by the guard; DTB3 (public FRED H.15) probed but advisory since 9.3a.
- Subagents: implementers on Fable, reviewers on Opus at medium effort; waves of host-free tasks, one
  review each, commit per task, CI per push.

## Environment facts

- `docker` and `docker compose` are Podman 5.8.3 (machine `podman-machine-default`, WSL2 backend);
  `dagger` v0.21.11 with its engine in the Podman VM.
- Hosts and ports: llama-server 11570 (gateway 11571, `scripts/serve_local_llamacpp.ps1`),
  full-distribution host 11580 (gateway 11581, `scripts/serve_full_distribution.ps1`,
  `POST /v1/full_distribution_batch`), reference llama-server 11560, Ollama 11434/11435, FLM 11540/11541;
  gateways take `LOCAL_MODEL_API_KEY` from the root `.env` by name. No host is running at handover time.
- llama.cpp b12199 under WinGet, Vulkan iGPU, pinned GGUF blob `sha256-f06e91e1...e667e1` under
  `models/blobs/`; one load is about 6 GiB. The `uv run` launcher exits with code 0 after some minutes while
  its python child keeps serving; the lifecycle scripts track the listener itself.
- Known flakes: the Store `pwsh` alias (WinError 5 on launch) and slow CI runners on timing-sensitive tests.
- Private data: `runs/finance-sim/` (gitignored; plan, records, calibrators, gates, results, analysis,
  validation logs); replay out-roots under `D:\BuildArtifacts\memguard_alpha\replay-20261111*`.

## Rules that still apply

One root `.env` (template `.env.sample`); credentials only by name, never printed, never in receipts;
`runs/` private; licensed raw data (BAA10Y, FMP prices, ICE) never published, only aggregates with
citation; `.openidea/` untracked; temp files under `D:\BuildArtifacts`; never delete user data or session
history as cleanup; Modal only after the approval record (9.4) and with a verified stop; work on `main`;
stage explicit paths, never `git add -A`; commit messages end with
"Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"; strip ANSI codes from gh output.

## Commit index (this spec)

`d7991d7` 5.1, `9a6d781` 5.2, `7260deb` Windows CI fix, `5d6444e` 6.1, `1277cc2` 6.2, `9cd14a5` 6.3,
`0efa54d` upstream chore, `645fb5a` 7.2, `6fae278` 7.1, `0d386d1` 7.5, `340cc7f` 8.1, `b2009c5` 7.3,
`3e78d53` 6.4, `2273ea7` 7.6, `c50495e` 7.4, `c3b0a64` 7.7, `66a9694` 7.9, `233ef89` 7.8, `ef04406` 7.12,
`507efd3` 7.3b, `bd33e81` 7.10, `1c4035c` 7.11, `d023b17` 8.2, `c0b3abb` 8.3, `9f8a75a` 8.3c, `cb177f2` 8.4,
`657a015` 9.1, `73cb8e4` 9.2a, `6d9927d` 9.3a, `724c966` 8.3d (F4 labels), `086ab5a` 9.5 draft, `2c357ae` 9.1b; live-run notes `b550e88`, `1ff1591`, `69f78ea`; design
`35764dc`; test-threshold fixes `8dc3f0f`, `79b3909`. Notes commits `85e16d6` and the one carrying this handover follow.
