# Current restart checkpoint - 2026-11-05

Latest follow-up: investigate NIM, not local hosting. User reports the model works
and suspected a missing API key. The actual prepared Authorization header matches
the mounted dotenv key. Two direct authenticated NVIDIA-example requests, outside
Recall Guard, returned HTTP 504 after 302.458s (Docker Python) and 302.845s (host
HttpClient). Both diagnostics finished; no inference remains running. See
[diagnostic](docs/results/2026-11-05-nim-request-diagnostic.md). Root cause unresolved.
Pending user clarification: was the successful run in the web playground, or an
API call with this same project key? Do not ask for pasted credentials.
No WSL changes, model installs, or local server setup occurred.

Prepared a single-model runner option: `--model openai/gpt-oss-20b` requires
`--approved-max-requests 25` and a fresh output directory. It does not run Opus or
change frozen prompts. No follow-up measurement was started. Forty-one focused
Docker tests, Ruff, strict MkDocs and diff checks passed after this change.
Earlier test-launch attempts failed at the WSL service boundary before Python
started; the final container run succeeded without changing WSL configuration.

The approved S&P 500 endpoint pilot has finished. No model calls remain running.
It used 26/50 physical requests: Opus completed 25; NIM returned HTTP 504 on its
first eligibility call after 302.25 seconds. Zero retries or status polls.
No GPT-OSS measurements ran. The unused 24 requests were not transferred.

Read [results](docs/results/2026-11-05-sp500-pilot.md), the
[aggregate receipt](docs/results/2026-11-05-sp500-pilot.json), and the
[experiment track](docs/experiment-track.md).

Opus: 16/16 continuation answers parsed, 8/8 direct-recall abstentions. Four usable
pairs per cohort; cutoff contrast +0.0356553. Both pre-cutoff arms performed worse
than unchanged close. No established memorization or mitigation effect. Opus
reported 6,523 input/748 output tokens, zero cache tokens; estimated cost $0.0411.
NIM usage/cost unknown. The two-model coverage gate was not met.

Ignored artifacts:

- runs/container/2026-11-05-sp500-source/ (FRED snapshot through November 3)
- runs/container/2026-11-05-sp500-pilot-v3/ (active frozen plan, items and prompts)
- runs/container/2026-11-05-sp500-live/ (exclusive output; never silently resume)
- runs/container/2026-11-05-sp500-export.py (offline replay/export helper)

Next decision: compare the user's working API path with the direct reproductions
before another batch. User offered local GPT-OSS hosting or fmp, then explicitly
redirected attention to NIM. Do not infer that local hosting is required, that
the key is invalid, or that missing results are a null effect. No automatic retry
was run. Diagnostic calls are separate from the original pilot ledger.

Preparation: 39 focused Docker tests and Ruff passed. Live response replay matched
every score and cohort summary; physical accounting and minimum pacing (2.004s)
verified offline. Full divergence spec approvals remain unchanged. Preserve
unrelated uncommitted provider/readiness/harness WIP. Earlier drafts v1/v2 and the
Nov 1 detailed handoff are historical; use the v3 bundle for this experiment.
