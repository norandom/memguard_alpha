# Design research and decisions

Reviewed 2026-11-01.

- Existing `core/divergence_decode.py` accepts full-prefix callbacks through
  `ModelSource`, validates finite same-length vectors and wraps callback errors
  in role/step `DecodeError`. It has no cancellation parameter or partial-success
  result. Cancellation therefore belongs in the callback/worker wrapper, with
  the parent recording incomplete work.
- `core/divergence.py` rejects infinities. Auxiliary smoothing must return finite
  scores; output-slot/header treatment must not introduce infinite masking.
- Current Dockerfile already centralizes the uv environment. Extend its optional
  targets and existing lock rather than create another dependency recipe.
- [Transformers GPT-OSS documentation](https://huggingface.co/docs/transformers/model_doc/gpt_oss)
  provides the direct model interface. Actual kernel/quantization support must
  be verified on the selected device and pinned dependency versions. A documented
  interface is not a successful local runtime measurement.
- [OpenAI reference implementation](https://github.com/openai/gpt-oss) establishes
  Harmony as a required model format. Ordinary JSON answer parsing cannot safely
  assume all decoded tokens are final-answer text.
- [Auxiliary-model research](https://arxiv.org/html/2605.31293v1) motivates a cheap
  trigram baseline but limits semantic efficacy expectations. The proposed
  interpolated estimator is an explicit implementation choice, not a reproduction
  of that paper's estimator.

The missing local design template is replaced by the existing repository's
boundary-first design structure. No new library API, serving stack or neural
training system is needed. Corpus/hardware/budget specifics remain execution
gates and must not be filled with invented provenance or performance claims.

## S&P 500 screening proposal — 2026-11-01

The user requested a COBESY experiment record including prior approaches and
suggested daily closes before/after model cutoff. The proposed next measurement
separates direct historical recall from paired identified/anonymized continuation.
It measures normalized forecast errors and an identity-advantage cutoff contrast,
not raw price variance or repeated-answer variance. This is target screening,
not auxiliary probability sourcing or a validated correction method.

See [experiment track](../../../docs/research/experiment-track.md) and
[S&P 500 research proposal](../../../docs/experiments/sp500-cutoff-proposal.md).
GPT-OSS-20B remains included; Opus is completion-only with no logprob requirement
for this measurement. Pilot budgets are proposals only. No live S&P requests
were run or newly authorized; amended requirements/design approvals are unchanged.
