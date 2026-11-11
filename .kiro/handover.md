# Handover: qualified-model-catalogue (active), memorization-adjusted-finance-sim (closed)

State as of 2026-11-11, end of day (system clock). Repository `norandom/recall_guard` (local folder
`D:\Source\memguard_alpha`, branch `main`). The owner ended the session while task 4.1 was in review. This
first part is the current state and overrides the older sections below where they differ.

## The objective and the active spec

The owner's objective: recall-aware answers. An answer carries a calibrated recall estimate and is corrected
by it. A forecast at 50:50 is the design condition, never a finding. Delivery: a front where people run a
point-in-time prompt and get the best answer the catalogue allows; integration tests and demos go through it.
Models and serving routes are data; what matters is that a model on a route can be qualified from a catalogue,
in the logprobs mode or the logits mode.

Spec `.kiro/specs/qualified-model-catalogue/` (requirements 1 to 11, design with "Amendment 2026-11-11: the
front", tasks in groups 1 to 7, all phases approved). `tasks.md` ends with "## Implementation Notes": one
entry per task, with the follow-ups each review routed to a later task. Read them before starting a task.

| Group | Done | Open |
|---|---|---|
| 1 Foundation | 1.1 to 1.4 | none |
| 2 Core | 2.1 to 2.4 | none |
| 3 Analyses and figures | 3.1, 3.1b, 3.2 to 3.5 | none |
| 4 Integration | none | 4.1 implemented, uncommitted, in review; 4.2, 4.3 |
| 5 Research record | 5.1 | 5.2 to 5.6 |
| 7 Front | 7.1 to 7.5, 7.3b | 7.6 documentation page |
| 6 Validation | none | 6.1, last |

## Do this first in the next session

1. `git status`. Expected uncommitted: the task 4.1 files listed below, and `.openidea/` untracked (never add
   it). Everything else is committed and pushed (last commit of the day: this handover).
2. Task 4.1 has no recorded review verdict. The review was started at session end; if no verdict is recorded
   in `tasks.md`, dispatch a fresh review (brief below), then commit on APPROVED or remediate.
3. Check CI: `gh run list --workflow CI --limit 6` (strip ANSI codes). See "CI" below for what to expect.

## Task 4.1 (replay integration): implemented, not reviewed, not committed

Uncommitted files: `scripts/finance_sim/replay.py`, `published_text.py` (new), `figures.py`,
`figures_draws.py`, `figures_convergence.py`, `figures_catalogue.py`, `repeat_baseline.py`; under
`tests/scripts/finance_sim/`: `test_published_text.py` (new), `test_replay.py`, `test_run_study.py`,
`test_figures.py`, `test_figures_draws.py`, `test_figures_convergence.py`, `test_figures_catalogue.py`,
`test_repeat_baseline.py`.

Implementer's report (not independently verified): `test_replay.py` 30 passed, `test_run_study.py` 52 passed,
the figure and analysis tests 390 passed, ruff and the architecture check clean. Only the test fixture study
was replayed; `runs/` and `docs/` are untouched. A fixture replay's 38 published files are under
`D:\BuildArtifacts\memguard_alpha\task-4.1-published\`.

What it does: the guarded replay writes the cutoff check, the mode comparison (which receives the logits
cutoff result for the identifying prompt) and the repeat baseline, and publishes Figures 5 to 7 beside
Figures 1 to 4 and 8. The replay reads draw logs only through the read-only `recording.read_draw_log`.
`published_text.published_reason` turns stored reason strings into plain sentences and refuses an unknown
one. Published tables drop `arm` and `role` and carry plain variant labels. The repeat baseline's route label
comes from the frozen plan.

Open questions the review was asked to judge (decide them before 4.2):

- The 14 published analysis copies (pre-registered, ensemble, negative-control, convergence, cutoff-check,
  mode-comparison, repeat-baseline-analysis) are copied unchanged in the study's working vocabulary, with
  requirement numbers and identifiers. This is against the owner's rule for published text. Either stop
  publishing the copies (the eight figures with their tables may satisfy Requirement 10) or rewrite the
  Markdown writers of the seven analysis modules.
- Published figure tables still carry internal values in other columns (`draw_mode`, `adjustment`, `status`,
  `stage`, `rendering`, `quantity`, `evidence`, the `statistic` identifiers of Figures 5 to 7).
- A Wilson lower bound of 0 of 40 prints as "0.0000000000000000069" in a caption on the fixture.

Review brief: the licensing guard must be unchanged in logic and scan every new file; no path may modify a
private record; two fresh processes must give identical bytes; no published sentence may be untrue relative
to its stored reason.

## Remaining tasks, in order

- 4.2 Republish from the recorded data and complete the catalogue: run the replay on the real records with
  `--require-guard`, fill the permutation p, the cutoff checks and the calibrator fingerprints for the
  entries of report R43, repoint R43 to the new asset names (it links `-light`/`-dark` SVG files and uses the
  old variant keys). A stored reason without a rule refuses the replay by design: add the rule with a test.
- 4.3 Catalogue page: caveat the serving EDGAR AUC of 0.000, show valid-draw counts, do not print a profile
  display name that contains "RAM".
- 5.2 methods page (turn the template's methods-page path into a link, tighten the head matcher, name the
  figure style in the template), 5.3 synthesis from `blueprint.md` with a humanizer pass, 5.4 R43 rewrite,
  5.5 register, 5.6 summary blocks for the 19 other dated reports.
- 7.6 Front documentation page (`docs/front.md`), with the wording suggestions of the 7.4 review: point the
  demo's calibrator sentence to the new page; "qualified ... can be corrected when its recorded calibrator
  is supplied"; one sentence each in the script on the recall estimate and the two modes; take the draw
  count from the result; add "or run with --fixture" to a client failure. Also: the mode failure remedy in
  `front.py` prints "not_tested" with an underscore; one catalogue reason cites "(R41)"; update the design's
  allowed-dependency list for the front.
- 6.1 Whole-feature validation: full suite, ruff, architecture check, strict docs build, replay compare, CI.

## The front today

`from recall_guard import ask, front, FixedReplies, worked_example, load_catalogue, ModelConfig`.
`ask(prompt, definition, *, entry, mode=None, client=None, calibrator=None, ...)` returns a `FrontAnswer`
(entry, mode, mode choice, status, reason, answer, `p_memorized`, corrected answer, limits, failure, draws
asked and used). Limit and mode-choice phrases are `front.LIMIT_*` and `front.CHOICE_*`.

- Demo, offline: `uv run --no-sync python examples/pit_front.py --fixture` (also `--entry`, `--mode`,
  `--json`); notebook `notebooks/pit_front.ipynb`.
- Integration tests: `tests/integration/` (`test_front_offline.py`, `test_pit_front_demo.py`,
  `test_front_live.py`).
- Live test, not yet run against a real route. With the local llama.cpp route up and `LOCAL_MODEL_API_KEY`
  set: `$env:RECALL_GUARD_LIVE_ENTRY = "llamacpp-gemma3-4b"`, then
  `uv run --no-sync pytest -q -rs tests/integration/test_front_live.py -k declared_route`. It starts no host.
- The shipped catalogue has 14 entries and no qualified mode: `llamacpp-gemma3-4b` is not tested in both
  modes (no permutation p recorded), `modal-gemma3-4b` logprobs is not qualified (AUC 0.583), six
  document-text entries are not qualified, the rest are not tested. Task 4.2 fills the missing inputs.
- Recorded for a later spec: a person cannot obtain a calibrator through the front (Requirement 11 has the
  caller supply it). Candidates: a calibrate step in the front, or a shipped calibrator for the worked example.

## Results on the recorded data (from tasks 3.1 to 3.3; no new run)

- Repeat baseline: llama.cpp at temperature 0.7, 3 of 29 prompts gave one answer on every draw (0.103, Wilson
  0.036 to 0.264); Modal at temperature 0, 31 of 35 (0.886, 0.740 to 0.955).
- Cutoff check, identifying prompt, 12 months inside the training range and 5 after: logprobs difference
  +0.096 (-0.00033 to +0.192, p 0.102); logits +0.044 (-0.166 to +0.313, p 0.335). Neither passes.
- Mode comparison, macro, 11 held-out prompts: AUC logprobs 0.722 (0.586 to 0.855), logits 1.000; paired
  difference +0.2783 (0.1403 to 0.4106); limits of agreement -0.7763 to +0.6693; correction factor
  difference 0.0414 (-0.0571 to 0.1466). Answer on whether logits help: undecided.

## CI

Four of the last 30 runs failed. Two (`0eedff5`, `ae6b9cd`) failed on
`ImportError: cannot import name 'FIGURE_WIDTH_IN'`: the committed 3.4 tests needed a constant that was only
in the then-uncommitted 3.5 work; fixed by `90f7f07`, which the end-of-day push carries. One (`caaf818`)
failed on a dropped local connection in `tests/scripts/test_serve_full_distribution_batch.py` on Windows
(5,575 other tests passed). One (`33b47d0`) gave no readable failure lines. The CI result of the end-of-day
push was not awaited.

## Working rules added today

- Research documents are quant documents: summary first, each fact once, models and routes as data rows, no
  operating-system or tooling narrative, never a retelling of the spec (link it under references). Figures in
  academic journal style. Statistics with the calculation shown.
- Commit per task, push per group of tasks. Stage explicit paths.
- Private draw logs must never be modified by an analysis; read them with `recording.read_draw_log`.
- Implementers on Fable, reviewers on Opus at medium effort; one independent review per task.

# Earlier spec: memorization-adjusted-finance-sim (closed 2026-11-11)

The overnight live run authorized on 2026-11-10 ("Yes, run it all except Modal") is complete.

## Where the spec stands

- All 42 sub-tasks are `[x]`: groups 1 to 8, 9.1 (runner), 9.2 and 9.3 (the live probe arm, accepted on
  the receipts by an independent read-only review on 2026-11-11), plus the follow-ups 7.3b, 8.3c, 9.2a
  (parser fence fix) and 9.3a (DTB3 advisory in the licensing guard).
- 9.4 is closed as a stopped run and 9.5 is final (report R43 at
  `docs/results/2026-11-11-memorization-adjusted-finance-sim.md`, registered, in the nav and the reports
  index, 26 published assets under `docs/assets/finance-sim/`); see "State at close" below.
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

## State at close (2026-11-11, evening): spec complete, serving arm stopped

The user's decision: "proceed with what we have and cut". No Modal rerun. 9.4 and 9.5 are `[x]`.

### The objective, in the user's words

"The objective is to produce recall aware answers." A forecast at 50:50 every month is expected and is not
the objective: "50:50 means that we ask the right question to derive the recall and to calibrate this back".
Lead every report with the validity of the recall estimate (calibrated logits and logprobs), never with
forecast skill.

### Working rules from this day

Treat a failure of the runner as a defect of this repository's code first. Do not investigate the operating
system, antivirus, the harness or the provider unless the user asks. Do not ask the same approval question
twice. Report results, not diagnostics. Explain results in plain terms before proposing work.

### Serving arm, final facts

- Smoke run passed on TRITON_ATTN. Calibration complete: 6,016 of 6,016 requests, USD 8.31. Macro 1,504 of
  1,536 draws parsed; EDGAR 609 of 4,480, the other 3,871 came back with an empty reply (not investigated).
- Both top-k calibrators are weak: macro held-out AUC 0.583 (prompt-clustered 0.249 to 0.917), EDGAR 0.000 on
  a holdout with one memorized prompt (not interpretable). The guard applies no adjustment on this arm.
- Full run stopped at 1,139 of 5,760 requests (17 months complete, 2016-02 to 2018-02, all before the
  cutoff). Runner exit code -536870911, cause not established. Ledger and receipt stay `open`; no item result
  was derived; the draws stay under `runs/finance-sim/records/serving/full_run/`.
- Negative control not started (excluded). About USD 10 of the approved 32.86 spent. Every serve receipt is
  stopped with verified cleanup.

### What the calibrated logits showed (both arms)

- Probe arm (local, full logits and top-k): macro calibrators AUC 1.00 (full distribution) and 0.72 (top-k);
  EDGAR 0.53 and 0.59, weak. The macro labels are the renderings (identifying against anonymised), and the
  estimate follows the rendering before and after the cutoff alike (full distribution, identifying: mean 0.75
  before, 0.82 after; 5 results per cell). It marks the prompt as recall-exposed; it does not show that a
  month was memorized.
- A calibrator labelled by exposure (before against after the cutoff in one rendering) is the open research
  step. It can be fitted on the recorded draws, but the calibration months are all before the cutoff, so the
  after-cutoff class would come from the 5 post-cutoff probe months.

### Final 9.5

Report R43 is final ("probe arm complete, serving arm stopped"). The guarded replay republished the 4
reliability files (they now carry the serving calibrators) and the second run compares byte for byte:
`runs/finance-sim/validation/live-9.5-final-replay-compare.log` and `-2.log`.

## Do this first in a new session

1. `git status` (nothing of the spec should be uncommitted; `.openidea/` stays untracked); check CI with
   `gh run list --limit 6` (strip ANSI codes before printing).
2. Reproduce the published assets at any time, no host needed:
   `uv run --no-sync python -m scripts.finance_sim.run_study replay --plan-sha256 <hash> --out-root D:\BuildArtifacts\memguard_alpha\replay-<date> --require-guard --compare docs\assets\finance-sim`
   (exit 0 and a byte-for-byte comparison are the pass; the 12 advisory DTB3 matches are expected).
3. No Modal work is authorized. A new paid run needs a new approval record.

## Open follow-ups from the cut

- The runner's accumulation in `scripts/finance_sim/execute.py` (`execute_items` and `_generate` hold every
  draw of a stage and derive nothing until the stage ends): derive and write each item when its ensemble is
  complete.
- A host-free close-and-derive path for an open serving stage (the partial path needs an `incomplete` ledger;
  the stop reasons are limit names only; the serving arm has no pass without a live host).
- The empty EDGAR replies on vLLM (86 percent of the EDGAR calibration draws).
- The exposure-labelled macro calibrator described above.

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

- Models folder cleaned on 2026-11-11 (owner's choice "unused and disqualified only", 35.2 GiB freed, 48.9 GiB left): removed the local weights of GPT-OSS 20B (Ollama blob and the Hugging Face safetensors with a broken partial download), LLaVA and Phi-4 mini. Kept: Gemma 3 4B (the pinned GGUF), Llama 3.1 8B, Llama 3.2 3B, the FastFlowLM and Ryzen AI stores, and the GPT-OSS tokenizer and config files that the Modal profile reads. The profiles `ollama-gpt-oss-20b`, `ollama-llava-llama3-ctx2048` and `ollama-phi4-mini` have no local weights now; they were never measured (not eligible on this machine, or disqualified).

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
