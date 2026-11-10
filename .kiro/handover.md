# Handover: memorization-adjusted-finance-sim

State as of 2026-11-10, 21:12 local time (system clock). Repository `norandom/recall_guard`
(local folder `D:\Source\memguard_alpha`, branch `main`). Last code commit: `cb177f2` (task 8.4). CI is green on every commit since the 8.3c fix (`9f8a75a`); the 8.4 push is being watched.

## Where the spec stands

- 38 of 42 sub-tasks are `[x]`: groups 1 to 8 complete (the whole study pipeline on fakes, the four figures and the offline replay with its licensing guard), plus the follow-ups 7.3b and 8.3c.
- Open: 9.1 (end-to-end offline validation through one runner, in progress on Fable), 9.2 to 9.5 (the live study: plan build, probe-arm calibration and gate, probe stages, Modal, report) and 10.x (docs).
- `tasks.md` ends with "## Implementation Notes": one entry per task with the
  reviewer's forward notes. Read 7.3b, 7.4, 7.7, 7.8, 7.10, 7.11, 7.12, 8.2 before
  writing the 9.1 runner. `design.md` carries dated amendment paragraphs
  ("decision 2026-11-10", "Amendment after ...") that override older text.
- Last local full suite (wave 3, before 8.2 and 8.3): 5,071 passed, 20 skipped,
  one known pwsh launch flake (`runs/finance-sim/validation/wave-3-full.log`).

## Working tree

Task 9.1 (the study runner, probably `scripts/finance_sim/run_study.py`, tests `test_run_study*.py`) may be present, written by a Fable implementer and not yet reviewed: treat it as a draft (inspect, review with kiro-review on Opus medium, then commit, or discard). Nothing else is uncommitted; no host is running.

## Decisions of 2026-11-10 (user; all in design.md and the notes)

- Execution: PowerShell for host commands (never Git Bash); Compose for runs,
  native uv for the unit-test loop, Dagger as the CI gate; never stop the Podman
  machine or WSL (single containers with `docker stop`); one model load at a
  time, gate on one load plus headroom, batch memory-heavy steps.
- Study sizing: the Modal serving arm is primary (none and top-k on the full
  evaluation set; frozen stages USD 32.54 of GPU time within the 34.34 ceiling),
  approved only after the local probe results. The local arm is a probe arm:
  8 draws, calibration 944 requests, a seeded 20-month evaluation subset (15 pre,
  5 post cutoff), negative control 40 rows; stages 7,380 / 1,740 / 2,700 /
  2,700 s (6 s per generation, 1.25 headroom, 300 s client timeout), about four
  hours of iGPU time in all plus the batched evidence passes. "We don't have 44h."
- Frozen in the plan: each arm's GGUF digest, the fit settings (0.6, 50), the
  evidence settings (600 s, 100, 5 retries, 60 s), the arm roles, and
  `pilot.gate_policy = "inform"`: the power gate cannot pass at effect 0.10 with
  month-level accuracy differences (verified: power at most 0.40), so it informs
  and the probe arm proceeds; the serving arm is ungated.
- Research defaults kept: effect size 0.10, negative-control tolerance 0.10 with
  the three-outcome rule. Correctness is scored on the raw sign when shrinking
  rounds a spread to zero.
- Reviewers run on Opus at medium effort, implementers at xhigh; two or three
  host-free implementers per wave, one review each, commit per task, CI as the
  gate per push plus one local full suite per wave.

## Do this first

1. `git status`; if 9.1 is present, finish its review and commit it (see "Working tree").
2. 9.1 is the runner that 9.2 to 9.5 drive live; its report lists the exact operator commands for 9.2.
3. Then 9.1, on a quiet machine (`podman ps` shows no other session's job; free
   RAM for one load plus headroom): the runner that threads the 7.4 recorder
   through `on_request`, `on_draw` and `on_evidence` (writing evidence files in
   the EvidenceSession format), runs the host checks before admission, loads the
   hosts one at a time (generation with llama-server, then the evidence pass with
   the full-distribution host), reads `gate_policy`, builds EvidencePolicy from
   the plan, passes no calibration overrides, refuses a client whose timeout
   differs from the plan's 300 s, captures every `stop` output to a file, and
   verifies the Compose-container route to the gateways. Then 9.2: build the plan
   with the keys by name through Compose (about 380 FRED requests on a cold
   cache, at least 3 minutes), the probe-arm calibration (about two hours) and
   the freeze. Then the probe stages, the report, and Modal (9.4) only with the
   user's approval record and the manifest endpoint set (it changes the manifest
   digest; set it before presenting).

## Environment facts

- `docker` and `docker compose` are Podman 5.8.3 (machine `podman-machine-default`,
  WSL2 backend); `dagger` v0.21.11 with its engine in the Podman VM; other
  sessions' Dagger engines also live there and can hold several GB.
- Hosts and ports: llama-server 11570 (gateway 11571), full-distribution 11580
  (gateway 11581), reference llama-server 11560, Ollama 11434/11435, FLM
  11540/11541; gateways take `LOCAL_MODEL_API_KEY` from the root `.env` by name.
- llama.cpp b12199 under WinGet, Vulkan iGPU, pinned GGUF blob
  `sha256-f06e91e1...e667e1` under `models/blobs/`; one load is about 6 GiB. The
  `uv run` launcher exits with code 0 after some minutes while its python child
  keeps serving; the lifecycle scripts track the listener itself (root cause open;
  launching `.venv\Scripts\python.exe` directly is the noted follow-up).
- Known flakes: the Store `pwsh` alias (WinError 5 on launch) and slow CI runners
  on timing-sensitive tests; two thresholds were loosened today.
- Validation logs: `runs/finance-sim/validation/`; host receipts and all study
  data: `runs/finance-sim/` (private, gitignored).

## Rules that still apply

One root `.env` (template `.env.sample`); credentials only by name, never printed,
never in receipts; `runs/` private; licensed raw data (BAA10Y, FMP prices, ICE)
never published, only aggregates with citation; `.openidea/` untracked; temp files
under `D:\BuildArtifacts`; never delete user data or session history as cleanup;
Modal only after the approval record (9.4) and with a verified stop; work on
`main`; stage explicit paths, never `git add -A`; commit messages end with
"Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>". The real `.env`
`SEC_USER_AGENT` still says memguard_alpha (user to change by hand).

## Commit index (this spec, 2026-11-10)

`d7991d7` 5.1, `9a6d781` 5.2, `7260deb` Windows CI fix, `5d6444e` 6.1, `1277cc2`
6.2, `9cd14a5` 6.3, `0efa54d` upstream chore, `645fb5a` 7.2, `6fae278` 7.1,
`0d386d1` 7.5, `340cc7f` 8.1, `b2009c5` 7.3, `3e78d53` 6.4, `2273ea7` 7.6,
`c50495e` 7.4, `c3b0a64` 7.7, `66a9694` 7.9, `233ef89` 7.8, `ef04406` 7.12,
`507efd3` 7.3b, `bd33e81` 7.10, `1c4035c` 7.11, `d023b17` 8.2; decision and
design commits `f8e6429`, `2ff9ec2`, `83d85cd`, `0baf659`, `971158f`, `09285dc`,
`d66f4a9`; test-threshold fixes `8dc3f0f`, `79b3909`.
