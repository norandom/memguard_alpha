# S&P 500 preparation checkpoint — 2026-11-05

User resumed the planned work and approved the concrete endpoint pilot. Source
snapshot, 16 dates, 48 measurement prompts and offline scoring were frozen.

Review [the concrete pilot](../../../docs/experiments/2026-11-05-sp500-pilot.md).
The 50-request envelope was approved and execution finished using 26 calls.
Opus completed 25 calls; NIM returned 504 at eligibility after 302.25 seconds.
See [results](../../../docs/results/2026-11-05-sp500-pilot.md). This is an isolated discovery
measurement; requirements/design/task approvals for the full divergence
implementation remain unchanged.

The minimal runner uses existing clients, persists intent and request ledgers,
and passed offline failure checks (39 focused tests total). Offline replay of the
live scores and summary passed. Next: resolve GPT-OSS access before a larger
study; combined coverage gate failed. Do not silently retry. Use -v3 and verify its bundle hash;
earlier drafts are superseded. The final plan reflects actual adapter timeout semantics.
No new hosting, native Python diagnostics or provider infrastructure is needed.
