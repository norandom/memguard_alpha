# FOMC statistical source: older/newer lexical signature, 2026-11-01

> Register: R04 · Status: feasibility · Track: Divergence

A dated corpus produces a measurable lexical signature: the two rate probes favor
the later zero-rate continuation more strongly after adding the 2020 statements,
including against a matched-token-count old-only control. This supports further
investigation of the statistical source. It does not establish target-model
mitigation or justify building out the endpoint correction workflow.

## Question

Do two statistical references fitted from older and newer Federal Reserve
statements give a measurable older/newer signature over fixed diagnostic prompts?

## Setup

This CPU-only prototype fits two statistical references from Federal Reserve
statements and compares their continuation probabilities over 12 fixed
diagnostic prompts. It makes zero model API calls.

The existing sparse trigram scorer estimates each next word/punctuation token
from the previous two, with fixed interpolation and smoothing. Both references
use the same vocabulary and score the same candidate continuation.

For a continuation, we sum its conditional log probabilities. The correction
signal is older score minus newer score. The diagnostic below uses the opposite
sign to show which candidate became more associated with later data:

    exposure shift = log(P_newer(candidate 1) / P_newer(candidate 2))
                   - log(P_older(candidate 1) / P_older(candidate 2))

These are natural-log units, not confidence percentages or contamination scores.
Long candidates accumulate multiple token-level effects. Large shifts can arise
from near-zero smoothed probabilities in a tiny corpus.

## Data

| Reference | Documents | Word/punctuation tokens |
| --- | --- | --- |
| Older | Four statements, January–June 2019 | 1,570 |
| Newer | Same four plus four statements, March–July 2020 | 3,729 |
| Old-only volume control | Deterministic repetitions of older documents, final copy truncated | 3,729 |

The control contains ten document segments, not ten independent historical
documents. It controls total token volume but not every effect of repetition,
length, composition or smoothing.

Each real statement has a source URL, publication date, displayed last-update
date, retrieval time and content hash. Dates are evidence supplied by the current
pages, not independent proof of historical byte-for-byte snapshots. The
[Board's public-domain policy](https://www.federalreserve.gov/disclaimer.htm)
covers these Board statements unless otherwise indicated.

For example, the [January 2019 statement](https://www.federalreserve.gov/newsevents/pressreleases/monetary20190130a.htm)
and [March 2020 statement](https://www.federalreserve.gov/newsevents/pressreleases/monetary20200315a.htm)
document different policy-rate regimes. This source contrast was deliberately
selected to make a small lexical feasibility check possible.

## Results

| Probe | Exposure shift | Shift relative to old-only volume control | Interpretation |
| --- | ---: | ---: | --- |
| Rate wording ending in “rate at” | +30.58 | +34.54 | Later zero-rate candidate gains relative support |
| Rate wording ending in “rate to” | +32.71 | +35.16 | Same direction under a second context |
| Coronavirus wording | +18.67 | +18.67 | Positive, but comparison candidate contains an unknown token |
| Risk wording | +11.04 | +11.04 | Positive, but comparison candidate contains an unknown token |

- The two rate candidates are both fully represented in the shared vocabulary.
- All four predeclared later-exposure probes shift positively beyond the volume
  control, but the two unknown-token comparisons are weaker evidence.
- Across all 12 probes, full next-token probabilities were finite and normalized.
  Identical-reference differentials were exactly zero. Fitted artifacts reloaded
  with matching scores.

We now have actual older/newer fitted references, source provenance and a
replayable prompt-level signature. The statistical-source gap is partially filled.

## Limits

- These are deliberate lexical diagnostics, not 12 independent blinded test cases.
  Prompts were frozen before fitting; source inspection informed their selection.
  No development/held-out split, alpha fit or efficacy test was performed.
- The tokenizer is a small local word/punctuation vocabulary. It is not verified
  compatible with GPT-OSS or Anthropic. This is not a deployable endpoint adapter.
- Unknown words share one unknown-token bucket. Several alternative candidates
  contain unknown tokens, so their probabilities cannot support semantic claims.
- The unseen-prefix probe still shows a +7.63 exposure shift. Unigram backoff and
  candidate-internal transitions carry corpus associations even with an unrelated
  prefix. A surviving paraphrase signal therefore does not establish understanding.
- Shared-language controls also move. For example, the mandate probe shifts +3.68,
  reduced to +0.43 against the volume control. The contrast is not exclusively
  future-knowledge removal.
- Only four later documents were added. Period, subject, crisis language and
  corpus composition are confounded.
- The strict efficacy-study human semantic audit has not been performed. This
  isolated prototype constructs diagnostic count tables and reuses the existing
  scorer; it does not claim formal study admission.

Before calibration for an endpoint, validate a broader reference corpus against
matched, previously unused questions with known vocabulary and utility controls.
Separately establish the endpoint mechanism that applies a correction. Neither
the earlier repeated-answer prototype nor this source-only test establishes that
mechanism. GPT-OSS-20B remains in the planned target study; no target ran here.

This prototype follows the auxiliary-contrast idea in
[Jane Street's study](https://blog.janestreet.com/mitigating-memorization-in-llms/)
and the [divergence-decoding formulation](https://arxiv.org/html/2605.31293v1#S3).
It is limited to lexical statistical sourcing, not a replication of their results.

## Evidence

- [Source manifest](2026-11-01-fomc-statistical-signature/sources/sources.json)
- [Frozen sources, tokenizer and prompts](2026-11-01-fomc-statistical-signature/plan.json)
- [All 12 probe results](2026-11-01-fomc-statistical-signature/probe-results.json)
- [Summary](2026-11-01-fomc-statistical-signature/summary.json)
- Fitted references: [older](2026-11-01-fomc-statistical-signature/models/older.json),
  [newer](2026-11-01-fomc-statistical-signature/models/newer.json),
  [volume control](2026-11-01-fomc-statistical-signature/models/old_volume_control.json).

From the repository root in the existing development container:

    uv run --no-sync python -m scripts.probe_temporal_signature measure --sources docs/results/2026-11-01-fomc-statistical-signature/sources --output /tmp/fomc-signature-replay

The output directory must not already exist. This command makes no network calls.
Raw HTML is retained locally under runs/container/2026-11-01-fomc-signature-sources-02.
The initial acquisition stopped on omitted HTML paragraph end tags; extraction
was corrected before the eight-document acquisition and fitting. No model requests
were spent on this repair.

Validation: Ruff passed; 37 focused tests passed in Docker, covering the probe
and existing trigram implementation. This establishes extraction/math behavior,
not scientific efficacy.
