# Experiment track: establish a usable signal before building mitigation

Updated 2026-11-05. We have a measured contamination-score baseline and a small
source-backed lexical signature. We have not demonstrated memorization removal
in a target model. The S&P 500 pilot now has usable Opus continuations and a small,
inconclusive cutoff contrast. GPT-OSS remained unavailable after NIM eligibility
returned HTTP 504.

This record keeps three questions separate: what the target reveals, what the
auxiliary sources encode, and whether a correction improves the outcome. A
successful calculation or a changed answer does not settle all three.

## What we can use and what we cannot conclude

| Track and date | Approach | Observed evidence | Decision and limit |
| --- | --- | --- | --- |
| Detection, Oct 30 | GPT-OSS-20B through NIM; fit existing contamination-score calibrator | ROC-AUC 0.876; 312/330 answers parsed; 170/312 correct | Retain the scoring baseline. AUC separates supplied exposure-proxy labels; it does not prove training membership or improved probability calibration. |
| Aggregation, Oct 31 | Opus single answer versus three-draw ensemble | 11/16 versus 12/16 correct overall; both 11/14 on common usable items | Keep aggregation separate from mitigation. No paired common-usable gain was observed in this small sample. |
| Completion mitigation, Oct 31 | Synthetic rewrites with a frozen fidelity gate | All 16 rejected; zero valid intervention pairs | No rewrite-efficacy estimate. Review fidelity policy separately; do not relabel rejected outputs as accepted. |
| Provider access, Nov 1 | NIM refresh with eligibility and calibration preflight | Three physical requests: 504, success, 504 | Access was observed once. No fresh calibration or ensemble measurement completed. |
| Mathematical implementation | Divergence equation and compatible callback decoder | Offline mathematical and callback checks; optional environment, corpus and trigram helpers implemented | Keep the reusable mathematics. No target-model efficacy result. Mandatory local hosting was superseded by endpoint-first investigation. |
| Completion-sample workaround, Nov 1 | Eight Opus answers reweighted using synthetic auxiliary counts | Eligibility plus eight answers used nine requests; all eight answers A; zero effect | This tested a workaround, not published token-level divergence decoding. Discard it as evidence for or against divergence feasibility; retain the receipt. |
| Statistical sourcing, Nov 1 | Four 2019 FOMC statements versus those four plus four from 2020; 12 diagnostic prompts | Both fully in-vocabulary rate probes shifted toward later wording beyond an equal-token old-only control; all sham differences zero | A real lexical reference signature exists. Unknown-token, wording and corpus-composition effects limit interpretation. No target model ran. |

The [November 5 S&P 500 pilot](results/2026-11-05-sp500-pilot.md) adds 16/16 usable
Opus continuations, 8/8 direct-recall abstentions and a cutoff contrast of +0.03566.
NIM returned 504 at eligibility after 302 seconds. Opus mechanics worked, but the
combined coverage gate failed. No divergence correction was applied.
The [follow-up diagnostic](results/2026-11-05-nim-request-diagnostic.md) verified
the outgoing API key and reproduced HTTP 504 in two direct requests using NVIDIA's
example, outside Recall Guard. This narrows the investigation without proving
general model unavailability or a credential problem.

Evidence: [NIM baseline](results/2026-10-30-nim-measurement.md),
[single-answer view](results/2026-11-01-recall-guard-single.md),
[ensemble view](results/2026-11-01-recall-guard-ensemble.md),
[completion pilot](results/2026-10-31-completion-api-pilot.md),
[NIM refresh](results/2026-11-01-nim-refresh.md),
[completion-reweighting prototype](results/2026-11-01-anthropic-reweighting-prototype.md),
[dated-source prototype](results/2026-11-01-fomc-statistical-signature.md).

## Why the source prototype is progress, with limits

The FOMC prototype replaces invented auxiliary ratios with fitted statistics from
documented text. It fills part of the statistical-source gap. Its signal belongs
to those reference corpora: adding later documents changes continuation scores.

Calling that a target model's memorization imprint would overstate the result.
Shared language and unrelated contexts also move. Some comparisons include
unknown words, and a trigram only uses the last two tokens of context. The
observed difference combines exposure, wording, smoothing, volume and composition.

The original [Jane Street study](https://blog.janestreet.com/mitigating-memorization-in-llms/)
motivates temporal auxiliary contrast. The
[divergence-decoding paper](https://arxiv.org/html/2605.31293v1#S3) supplies the
score adjustment. Neither makes our lexical diagnostic a demonstrated mitigation.

## Why repeated answers were the wrong first feasibility test

The nine-request Anthropic prototype used invented categorical counts and never
steered token selection. It could not establish whether real auxiliary statistics
are useful. More identical completions would not have filled that gap.

All recorded cache creation/read counters were zero, and output-token counts
varied. The receipts do not support attributing identical final answers to
prompt-cache hits. Anthropic documents these
[cache usage fields](https://platform.claude.com/docs/en/build-with-claude/prompt-caching).
This still does not establish independent or well-calibrated sampling.

The experiment's first report conservatively stopped expansion. The narrower
conclusion is the one retained here: insufficient sampled support for that
workaround, no verdict on divergence decoding.

## Next: measure a historical identity advantage on S&P 500

The user's time-series proposal gives us numerical ground truth and many dated
items. We can ask whether a model reconstructs old prices or predicts historical
returns unusually well when it can identify the series and date.

The [S&P 500 proposal](experiments/sp500-cutoff-proposal.md) separates direct
close recall from one-step continuation. Its main diagnostic compares identified
and anonymized versions of the same numeric window, then compares that advantage
before and after each model's documented cutoff.

This is a screening experiment for contamination-related behavior. It needs no
logprobs and does not use repeated answers as a probability estimator. It does
not itself supply the older/newer auxiliary likelihood ratio or solve endpoint
correction. Those remain separate implementation questions.

## Evidence that would change the decision

The [November 5 preparation](experiments/2026-11-05-sp500-pilot.md) now freezes
16 dates and 48 prompts from a local FRED snapshot, with verified offline scoring.
The approved live cap was 50 requests including eligibility. The completed run
used 26: 25 successful Opus calls and one NIM 504. Opus's small positive contrast
is descriptive; GPT-OSS did not reach measurement.

- If no historical identity advantage appears, retain that null/inconclusive
  result. Do not search prompts until one looks successful.
- If the difference disappears after volatility matching, normalization or
  coverage accounting, treat the original difference as confounded.
- If the same advantage appears after cutoff or around placebo dates, temporal
  memorization is a weaker explanation.
- If a pre-cutoff advantage survives those checks on held-out dates, proceed to
  a separately specified mitigation test. It remains evidence consistent with
  contamination, not verified membership in training data.
- A mitigation must reduce the unwanted historical advantage while preserving
  post-cutoff task utility. Merely worsening all answers is not success.

The November 5 endpoint pilot is complete; no automatic retry is authorized by
this record. Full divergence implementation remains gated by the amended specification.
