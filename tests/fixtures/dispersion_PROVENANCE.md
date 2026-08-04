# Vendored dispersion corpus — provenance

## What this is

`dispersion_draws.csv` and `dispersion_guard.csv` are the numeric columns of a measurement
study that asked one model the **same prompt** many times at the production setting
(`temperature=0`), to characterise how much the serving stack varies its own answer.

| file | rows | contents |
|---|---|---|
| `dispersion_draws.csv` | 977 | five component values per draw, from 1000 attempts (23 did not parse) |
| `dispersion_guard.csv` | 100 | contamination score per draw, one identical prompt |

Only numeric columns were copied. No prompt text, reply text, model identifier, or timing is
vendored. Rows that failed to parse are excluded, so `n` here is the *parsed* subset.

## Where it came from

The `appendix_i_factor_dispersion` study in the sibling `Global_Macro_AI_Factors` project,
at a single rebalance date: **2020-03-02**, the COVID onset. The draws were collected at
`max_tokens=2048`; the contamination scores went through the library default of 512.

## Why it is vendored rather than referenced

`recall_guard` cannot import a sibling project, and the `ensemble-consensus` design's central
claims are all properties of *this* data — that the robust scale estimate is undefined on one
component, that another splits into two separated clusters no symmetric trim can reduce, that
the emission lattice the source proposal assumed is not the real one, and that the
contamination score is continuous and right-skewed. Without the data those tests cannot be
written and the claims go unguarded.

## What it does not establish

**One date is not a distribution over regimes.** This is a crisis onset, chosen because it is
the hard case. Whether the disagreeing component is bimodal in calm conditions is unmeasured.

Every constant tuned against this corpus — cluster mass thresholds, trough width, density
ratio — is therefore **provisional**, and ships as configuration rather than as a literal.
Treat a passing test here as evidence that an implementation behaves correctly *on the
measured pathology*, not as evidence that the pathology generalises.

The draws are also not independent: they were collected as a burst against one serving stack,
where batching, cache reuse, and node affinity all induce positive correlation. Any interval
computed from them is narrower than its nominal label.
