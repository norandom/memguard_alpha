> SUPERSEDED evaluation proposal (2026-11-01): the mandatory local runtime and fixed auxiliary selection below are historical. GPT-OSS-20B remains included. The [endpoint-based amendment](../divergence-decoding-evaluation/scope-amendment.md) and its requirements govern the follow-up. The completed mathematical MVP is unchanged.

# Real-model divergence evaluation: follow-up brief

Date: 2026-11-01. Language: English. Status: discovery for a separate evaluation
specification; existing MVP requirements, approvals and completion remain intact.

Recall Guard adopters have mathematical correctness tests and an unadjusted
`openai/gpt-oss-20b` scoring baseline, but no real-model divergence mitigation
result. The follow-up will test controlled token steering and recall/utility
tradeoffs with this required target, initially without ensembles.

The selected configuration is the official `openai/gpt-oss-20b` open weights
plus two proposed trigram language models fitted on dated corpora using its exact
tokenizer. See [model selection](model-selection.md) for rationale, evidence,
proposed windows, excluded alternatives and limitations. These small auxiliaries
make the first experiment possible without assuming that an off-the-shelf
older/newer neural pair exists for the GPT-OSS vocabulary.

Boundary candidates: reuse the completed decoder; add optional experimental
model execution, auxiliary corpus fitting and paired evaluation. Keep the main
model's weights unchanged. Keep completion-only rewriting and sampled ensembles
separate. Do not claim broad semantic or historical efficacy from lexical recall
suppression. Default library dependencies and hosted scoring contracts stay lean.

Next phase: generate and review requirements using this selection, then design
the runtime and tasks. No new implementation or live compute run is approved by
this brief. Requirements must include feasibility gates and a concrete resource
budget before execution.

