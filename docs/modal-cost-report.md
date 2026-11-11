# Modal pilot receipt and cost accounting

This describes the offline accounting contract for `scripts/probe_modal_model.py`.
It authorizes no paid work. Staging weights, building, deploying and verifying
live prices/account access are still separately reviewed prerequisites. The USD 5
target is not a hard ceiling. Monthly provider budgets include prior usage;
environment budgets cover compute, while workspace budgets cover broader usage.
The runner never changes shared account policy.

The default plan prints artifact/resource configuration, frozen prompts and
request/token/time bounds, dated pricing sources, and the configured-resource
estimate for the maximum GPU phase. The total estimate stays null because
staging, build, storage, expiry control, egress and cleanup tails are excluded.
Review these additional costs before paid approval; the subtotal alone is not enough
to establish a complete pilot allowance.

The existing reviewed execution approval must also contain `monthly_budget`:

```json
{
  "scope": "environment_compute",
  "workspace": "YOUR_WORKSPACE",
  "environment": "recall-guard-pilot",
  "period_month": "2026-11",
  "checked_at": "2026-11-06T08:00:00Z",
  "source": "operator_budget_review",
  "monthly_limit_usd": 20,
  "prior_usage_usd": 10,
  "requested_allowance_usd": 5,
  "accounting_available": true,
  "operator_confirmed": true
}
```

Use actual current, timezone-aware timestamps and current monthly period; numbers
above are examples. `scope` is `environment_compute` or `workspace`. The runner
checks target ownership, finite nonnegative USD, prior usage and remaining
allowance before any cloud operation. The requested allowance must cover at least
the configured compute subtotal. With unavailable accounting, set
`accounting_available` false and provide `operator_confirmed: true` and the
operator's allowance. Unknown `prior_usage_usd` is null; remaining allowance
then stays unknown. A known insufficient remaining allowance always fails.
The record is a dated snapshot and does not guarantee enforcement or reserve
funds. Keep the external source record outside published receipts; `source` is
a short identifier, never an invoice, raw provider error or credential.

Each `attempts` row describes one physical inference intent/outcome. `position`
is the zero-based index into frozen prompts, not a generated-token index.
`input_tokens` and `usage` are counts, not token arrays. Token evidence stays
in the existing validator; generated text, token strings and logprob arrays are
not exported. `elapsed_seconds` measures request wall time, including transport
and validation; it is not model-only token generation time.

`control_events` describe cache inventory and readiness without incrementing
inference counts. `staging_events` explicitly marks external weight staging as
a separate prerequisite with unknown cost. Cleanup and expiry outcomes stay
in `lifecycle`, whether or not inference succeeded. `cost_report` is one summary
row per run: cold startup runs from deployment admission to readiness; generation
seconds sum inference elapsed time. HTTP success counts any 2xx result; usable
measurements require eligible evidence and the measurement stage. The eligibility
probe is excluded from the usable denominator. All timestamps ending in
`_unix_s` use Unix seconds, durations use seconds, and money uses USD.

The resource subtotal uses configured GPU/core/GiB rates times runner elapsed
time through cleanup. It is an estimate, not observed billing or verified remote
resource duration. Unverified cleanup may continue accruing charges. Gross
observed cost and credits remain separate and null until available. A missing
whole-pilot cost or zero usable measurements produces null cost per usable
measurement. A documented observed gross cost of zero may produce zero only
when there is usable evidence.

Append reviewed local billing without rerunning inference. Keep the receipt and
billing file under `runs/container`, which the development container sees as
`/outputs`:

```text
docker compose run --rm dev uv run --no-sync python scripts/probe_modal_model.py --receipt /outputs/pilot.json --reconcile-billing /outputs/billing.json
```

The billing JSON contains `observed_gross_usd`, `credits_usd` (both may be null),
`source`, `checked_at` and `coverage` (`whole_pilot` or `partial`). Only these
fields are published. Entries append to `billing_reconciliations`; they are
snapshots, not additive charges. The latest snapshot supplies the summary;
partial coverage cannot establish a whole-pilot observed total. Existing attempts
and frozen price assumptions remain unchanged. Reconciliation uses atomic local
replacement; run one writer per receipt. It does not access provider billing.
