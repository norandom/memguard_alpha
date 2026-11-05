# Dated-source statistical signature spike

Authorized by the user's "ok, do that" after agreeing to fit older/newer
references and probe their differential. Uses public source downloads and CPU
calculations only. No target model requests, GPU or full implementation approval.

## Frozen before fitting

Source: Federal Reserve Board FOMC statements, with the Board's public-domain
policy at https://www.federalreserve.gov/disclaimer.htm.
Older: 2019-01-30, 2019-03-20, 2019-05-01, 2019-06-19.
Added later: 2020-03-15, 2020-04-29, 2020-06-10, 2020-07-29.
Newer includes all older documents. Current downloads and hashes are retained;
page publication/last-update dates are evidence, not independently archived
proof that today's exact HTML bytes existed at the historical date.

Fit shared-vocabulary word/punctuation trigrams with the existing scoring
implementation, fixed beta=1 and add-one unigram smoothing. Vocabulary is derived
from the combined fitting texts with an explicit unknown token. This is a local
diagnostic tokenizer, not a claim of GPT-OSS or Anthropic token alignment.
No alpha tuning or target decoding.

Twelve fixed diagnostic prompts in the script cover rate language, pandemic
language, unchanged boilerplate, paraphrases and unseen contexts. They are
selected for source inspection, not random independent test items or a blinded
held-out efficacy set. Score identical candidate continuations under each model.

Controls: identical-model cancellation; larger old-only corpus matched to the
newer token count by deterministic repetition and final truncation. Repetition is
an artificial volume control, never represented as additional historical data.
Report its changed document count and construction.

Success is limited to reproducible finite conditional probabilities, zero sham
difference, and an interpretable lexical contrast beyond the volume control on
at least one predeclared later-exposure probe. No causal temporal-unlearning claim
or statistical generalization follows from this diagnostic sample. If paraphrases
lose the signal, report that limitation rather than tune prompts after fitting.

The older strict study fitter requires a human semantic audit. This isolated
source diagnostic does not claim that audit, study admission or held-out
validation: it builds count tables for the existing SparseTrigram scorer directly.
It leaves the approved study admission checks unchanged. No artifacts from this
spike may be promoted into an efficacy study without that study's provenance and
split review.

## Observed outcome

Completed: four older and eight cumulative newer documents fitted, 12 fixed
prompts scored, zero target API calls. The two fully in-vocabulary rate probes
showed positive later-exposure shifts beyond the equal-token old-only control.
All four later probes had positive shifts, but two used unknown-token alternatives.
Sham differences were zero. Shared and unseen-context probes also moved, limiting
specificity. No calibration fit, endpoint correction or mitigation efficacy claim.

Artifacts and limitations:
[dated source-signature report](../../../docs/results/2026-11-01-fomc-statistical-signature.md).
The diagnostic source gap is partially filled; production temporal provenance,
held-out validation and endpoint application remain unresolved.
