# Handover: memorization-adjusted-finance-sim

Written 2026-11-10 at the end of the session. The repository is
`norandom/recall_guard` (local folder `D:\Source\memguard_alpha`, branch `main`).
Last code commit: `0efa54d`. CI is green on every commit since `7260deb` (five
jobs, Windows included).

## State

- Spec: `.kiro/specs/memorization-adjusted-finance-sim/`. 21 of 42 sub-tasks are
  `[x]`: groups 1 to 5 complete, 6.1 to 6.3 done. Task 6.4 (batch evidence and
  memory guard) was added on 2026-11-10 and is half done (see below). Tasks 7.1
  onward are untouched.
- `tasks.md` ends with "## Implementation Notes", the running memory of every
  decision, pin, port, flake and follow-up. Read it before dispatching anything.
- `design.md` carries concrete contracts through 8.4, so 7.x and 8.x should
  dispatch without context rounds.
- Full-suite baseline on `main` (native uv): 4092 passed, 20 skipped, plus the
  chore's 13 tests (`0efa54d`), so 4105 expected before 6.4's new tests.

## Working tree: task 6.4, phase 1 done, phase 2 not run

Uncommitted, written by an Opus implementer, not yet reviewed. Treat it as a
draft: inspect, review, then commit, or discard with `git checkout --` and `rm`.

Modified: `recall_guard/core/full_distribution.py`, `recall_guard/core/profiles.py`,
`recall_guard/core/provider_profiles/llamacpp-gemma3-4b.json`,
`scripts/llamacpp_native.py`, `scripts/serve_full_distribution.py`,
`scripts/serve_full_distribution.ps1`.
New: `scripts/soak_full_distribution.py`, `tests/core/test_full_distribution_batch.py`,
`tests/core/test_host_memory_floor.py`,
`tests/scripts/test_full_distribution_batch_exactness.py`,
`tests/scripts/test_serve_full_distribution_batch.py`,
`tests/scripts/test_serve_full_distribution_gateway_guard.py`,
`tests/scripts/test_serve_full_distribution_memory_guard.py`,
`tests/scripts/test_soak_full_distribution.py`.

Phase-1 evidence (logs under `runs/finance-sim/validation/task-6.4-*.log`): RED
and GREEN logs per behavioural change, focused set 361 passed (`task-6.4-test.log`),
ruff clean, architecture check OK, soak-sampler timing flake fixed
(`task-6.4-stability-soak.log`), one WinError 5 pwsh launch flake seen once and
not reproducible (the known Store alias flake). The implementer's final Status
Report was never produced because the session ended; the facts above come from
its interim reports.

Phase 2 never started: the RAM gate cleared at 12:32Z and the session ended.
Steps, in order, on a quiet machine:

1. Preflight: nothing listening on 11560/11570/11571/11580/11581, no model host
   process, free RAM at least 12 GiB (two loads), no other session's Dagger or
   Compose job in the Podman VM (`podman ps`).
2. Batch exactness against the 11560 reference llama-server (two single pairs plus
   one prompt with two responses; per-item timing; agreement within 1e-4 and
   identical ranks), receipt under `runs/finance-sim/hosts/full-distribution/exactness/`.
   See `scripts/serve_full_distribution.py --help` for the batch flag the
   implementer added and `tests/scripts/test_full_distribution_batch_exactness.py`.
3. Stop the reference server; verify it is gone.
4. Start the host through `scripts/serve_full_distribution.ps1` (faulthandler
   launch, gateway memory logging); run the 30-minute soak with
   `scripts/soak_full_distribution.py` through the gateway (bearer by name);
   receipt under `runs/finance-sim/hosts/full-distribution/soak/`; then `stop` and
   verify with Get-NetTCPConnection and Get-Process that nothing remains.
5. Only after the host is down: full suite `uv run --no-sync pytest -q -p no:cacheprovider`
   (expected about 4105 plus 6.4's new tests, 20 skipped).
6. Independent review (kiro-review, Opus medium), mark 6.4 `[x]`, notes in
   `tasks.md`, commit the files listed above with `tasks.md`, push.

## Do this first

Corrections from the user at session end, to apply before anything else:

- PowerShell for host commands, never Git Bash.
- Never stop the Podman machine or WSL; Compose works, and single containers are
  controlled with `docker ps` and `docker stop <name>`.
- The two-load RAM gate (12 GiB) was wrong: half the memory was free and processes
  waited for an hour. Gate on one load plus headroom (6 GiB) in both lifecycle
  scripts; keep the 6.3 drift test green.
- Never schedule two large memory consumers together; split into batches. The
  exactness check records llama-server outputs first, stops it, then runs the host
  against the recording; the soak runs alone; the full suite runs after the host is
  down. Rewrite 6.4 phase 2 (steps 1 to 5 above) to that order.


1. `git status`; confirm the tree matches the list above and nothing else moved.
2. Check the machine is quiet: `podman ps` (other sessions' Dagger engines live in
   the Podman VM, process `vmmemWSL`, and held 6 GB for an hour today), free RAM,
   no listeners on the host ports.
3. Finish 6.4 phase 2 as listed, either by re-dispatching an Opus xhigh
   implementer with that list and the file list, or by hand.
4. Then 7.1 onward (study pipeline on fakes, no hosts, no network): one Opus
   xhigh implementer per task, kiro-review on Opus medium, the parent verifies the
   evidence, marks the task, commits explicit paths, pushes. 7.1 was dispatched
   today and stopped before it wrote any file.
5. Tooling on this workstation: use PowerShell for shell work, not Git Bash (the
   user's instruction at the end of this session; Git Bash also showed fork and
   heredoc failures all day).

## Decisions taken 2026-11-10 (recorded in the spec, steering and memory)

- Task 6.4 added: `POST /v1/full_distribution_batch` (prompt evaluated once, KV
  rollback per response), continuous memory floor with 503 `memory_low`,
  faulthandler launch, batch exactness mode, 30-minute soak. Reason: per-draw
  evidence would take about 67 hours for the 8,064 local draws; batch mode about
  7 hours.
- The 6.2 host self-exit is attributed to memory contention (another session's
  build held free RAM near 4 GiB); hosts are never started without their free RAM
  plus headroom and never beside another heavy job
  (`.kiro/steering/execution-environment.md`).
- Research defaults kept: effect size 0.10, negative-control tolerance 0.10.
- Execution split: Compose for runs (study scripts, the runner, anything touching
  data or hosts), native uv for the unit-test loop, Dagger as the CI gate; model
  hosts native (iGPU, NPU); containers reach them through the gateways on an
  adapter address (`-BindAddress`), to be exercised from a Compose container in 9.1.
- Subagents: implementers Opus xhigh, reviewers Opus medium; the parent handles
  evidence checks, marks, commits and agent hand-offs directly.
- Upstream chore committed (`0efa54d`): providers.py credential validation,
  consensus.py docstring, calibration dedup, FLM pipe inheritance. Follow-ups are
  in the Implementation Notes (provider_anthropic.py keeps the old check, and more).

## Why today was slow, and what to change

- Every task went through implementer, independent review and verification in
  sequence, with live host checks (6.2, 6.3) and a 30-minute soak planned for 6.4.
  The gates caught real defects (the bearer echo in providers.py, the pipe
  inheritance, the dedup), but each cycle costs 30 to 60 minutes of wall time.
- The 6.4 live phase waited an hour for RAM because another session's Dagger
  engine filled the Podman VM. Live phases need a quiet machine: schedule them
  when no other agent session runs, and run the host-free tasks (7.1 to 7.12,
  8.x) meanwhile.
- Faster from here: tasks 7.1 to 7.3 touch disjoint modules under
  `scripts/finance_sim/` and need no host; two implementers at once are safe when
  each keeps to new files and the parent runs the full suite once per wave, not
  per task. Keep reviews on Opus medium.
- Known flake: the Store `pwsh` alias intermittently fails to launch with
  WinError 5 (retry helper `_pwsh` in the test files); not a code defect.

## Environment facts

- `docker` and `docker compose` are Podman 5.8.3 (machine `podman-machine-default`,
  WSL2 backend, `.wslconfig` memory 8 GB); `dagger` v0.21.11 with its engine in the
  Podman VM; the other WSL distributions are unused.
- Hosts and ports: llama-server 11570 (gateway 11571), full-distribution 11580
  (gateway 11581), reference llama-server 11560, Ollama 11434/11435, FLM
  11540/11541; gateways take the bearer `LOCAL_MODEL_API_KEY` from the root `.env`
  by name.
- llama.cpp b12199 under WinGet, Vulkan iGPU, pinned GGUF blob
  `sha256-f06e91e1...e667e1` under `models/blobs/`; one model load is about 6 GiB.
- Validation logs: `runs/finance-sim/validation/`; host receipts:
  `runs/finance-sim/hosts/`; all private.

## Rules that still apply

One root `.env` (template `.env.sample`); credentials only by name, never printed,
never in receipts; `runs/` receipts and per-token scores private; licensed raw data
(BAA10Y, FMP prices, ICE) never published, only aggregates with citation;
`.openidea/` untracked; temp files under `D:\BuildArtifacts`; Linux dependencies
stay in Linux; never delete user data or session history as cleanup; Modal only
after the approval record (9.4) and with a verified stop; work on `main`; stage
explicit paths, never `git add -A`; commit messages end with
"Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>". The real `.env`
`SEC_USER_AGENT` still says memguard_alpha (user to change by hand).

## Commit index (this spec)

`cd7a0f4` repository rename, `d7991d7` 5.1, `761b9c7` Podman environment docs,
`9a6d781` 5.2, `7260deb` Windows CI fix, `5d6444e` 6.1, `1277cc2` 6.2,
`f8e6429` spec: 6.4 and execution split, `9cd14a5` 6.3, `0efa54d` upstream chore.
