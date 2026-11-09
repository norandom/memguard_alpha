# GPT-OSS response handling retest: reasoning and final tokens were mixed, 2026-11-06

> Register: R14 · Status: diagnostic · Track: Modal

The earlier text-alignment failure is a response-handling integration bug, not
evidence that GPT-OSS lacks usable log probabilities. The diagnostic response
confirms that vLLM returned probabilities for the full Harmony generation while
the adapter compared them with final-answer text alone. The corrected opt-in
adapter passes strict validation on replay of the captured API response.
Sampling across different requests has no bearing on this same-response check.

## Question

Did the earlier eligibility rejection come from the model's log probabilities or
from how the adapter mapped the Harmony response to final-answer text?

## Setup

The authorized diagnostic made one additional eligibility request with the same
model, pinned tokenizer, temperature 0, top_p 1, five alternatives and 512-token
output allowance. `return_token_ids: true` added original position metadata;
generation settings stayed fixed. No measurement prompts or retries followed.

Hosting and cost observations: this retest observed an A100-SXM4-40GB, unlike
attempt 08's provider-assigned 80 GB device. Startup took 100.249 seconds and the
request took 3.032 seconds including transport/validation. Model shutdown was
verified; a subsequent inventory showed all twelve remaining owned app entries
stopped. Older stopped entries may disappear from provider inventory, so this
count does not rewrite the earlier fifteen-entry observation.

The subsequent workspace snapshot reported provisional gross USD 0.29792825 and
billed USD 0. Retained weight storage remains separately chargeable.

## Data

One captured eligibility response (response 09). The private response body
enables offline replay without another API call. Generated text is absent from
this publication.

## Results

| Captured response structure | Observed value |
| --- | ---: |
| Generated token IDs and probability entries | 30 each |
| Final-answer probability positions | 1 |
| Final-answer source span, zero-based and half-open | `[28, 29)` |
| Final-answer characters | 2 |
| Reasoning characters | 70 |
| Alternatives for the final token | 5 |
| Full-generation text matches final answer | False |
| Explicit final-channel text matches final answer | True |

The final channel is identified through the pinned tokenizer's control IDs and
Harmony headers. The adapter must select its original probability entry, not
search for answer text inside the generated stream. Probabilities and alternatives
remain unchanged; reasoning/control positions are excluded from final-answer
evidence.

The repaired production adapter was replayed against that same captured response
with HTTP disabled. The original adapter reproduced the failure; the explicit
Harmony adapter passed strict validation with all five alternatives. Original
selected logprobs and alternatives matched exactly. This is offline replay of
one live response, not another generation after the fix.

Why the previous check failed: in pinned vLLM 0.17.0,
[chat response construction](https://github.com/vllm-project/vllm/blob/v0.17.0/vllm/entrypoints/openai/chat_completion/serving.py)
constructs logprobs before
[Harmony parsing](https://github.com/vllm-project/vllm/blob/v0.17.0/vllm/entrypoints/openai/parser/harmony_utils.py)
extracts the visible final message. Recall Guard's generic adapter attached the
full probability sequence to that final message. Its strict validator correctly
found a mismatch, but the result was framed too broadly as an ineligible model
configuration. The integration needed an explicit response mapping.

The explicit profile keeps strict validation. `HarmonyOpenAICompatibleClient`
opts into GPT-OSS-20B response mapping over the existing transport.
`HarmonyTokenMap.from_tokenizer_json` loads control IDs and records the revision
and SHA-256 without a tokenizer runtime dependency. The generic
OpenAI-compatible client keeps its existing behavior.

The profile requires original `token_ids` metadata, one complete final message
and an exact final-text match. Tool messages, ambiguous/multiple visible messages,
truncation and unsupported byte representations fail closed. It does not trim
probabilities by substring search, fabricate values or retokenize the answer.
The normal evidence validator still checks probability values and alternatives.

## Limits

- The original attempt-08 receipt remains unchanged. Its exact boundary layout
  cannot be recovered because that response body was not retained. Response 09
  provides the direct evidence above; it is a separate generation, not a claim
  about attempt 08's unseen text.
- Use a fresh calibration for this evidence profile before making scoring claims.
  This diagnostic establishes response structure and mapping behavior; it measures
  no calibration quality, ensemble benefit or
  [divergence decoding efficacy](https://blog.janestreet.com/mitigating-memorization-in-llms/).
- The startup and request timings are not a hardware performance ranking.
- The cost snapshot is neither a reconciled experiment invoice nor a final
  free-cost claim.

## Evidence

The [numeric receipt](2026-11-06-gpt-oss-response-mapping.json) records the body
hash, tokenizer identity, selected span and verification outcomes without output
text.

Validation: independent review approved the repair and publication. The focused
suite passed 159 tests; the full offline suite passed 2888 tests with two skips.
Those skips cover the optional Modal SDK absent from the root environment and an
existing symlink test requiring host privileges. Ruff, architecture and strict
docs checks passed. The live-response replay made zero HTTP requests.

See the [original pilot track](2026-11-06-modal-pilot.md) for startup repairs and
its historical outcomes.
