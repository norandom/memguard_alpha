# GPT-OSS-20B alignment diagnostic: proposed next decision

Date: 2026-11-06. Status: proposal only; no additional inference authorized or run.

## Decision and evidence

Keep GPT-OSS-20B in scope. Modal attempt 08 reached model readiness but its single
eligibility response failed `token_text_alignment_failed`. Stop-on-failure worked:
zero measurement prompts followed. The private receipt intentionally omits raw
response text. It therefore cannot distinguish a reasoning/final-channel mismatch,
control-token rendering, byte decoding or a different provider/parser discrepancy.
These are hypotheses, not findings.

The five-model Ollama smoke remains separate: all five supplied aligned evidence;
LLaVA alone had an environment context mismatch. No local GPT inference was run.
Opus and other completion-only frontier APIs remain outside this diagnostic.

## Prepare offline before requesting another call

1. Review the pinned vLLM 0.17.0 response/parser contracts and the existing adapter.
   Its reasoning API exposes reasoning separately from final content; that fact
   does not establish which field caused our failure. Use
   [the versioned primary documentation](https://github.com/vllm-project/vllm/blob/v0.17.0/docs/features/reasoning_outputs.md).
2. Add synthetic fixtures for final-only, reasoning-plus-final, control tokens,
   Unicode/byte fragments, missing bytes, truncation and inconsistent arrays.
   Keep token positions, channel ownership and array alignment explicit in names.
3. Prepare a structural diagnostic that compares response fields before parsing,
   without publishing generated text: field presence, lengths, token counts,
   byte availability, exact reconstruction booleans and finish reason. It must
   preserve raw evidence semantics and distinguish unavailable evidence from a
   parser defect. Never trim or retokenize until strings happen to match.
4. Resolve requested versus observed hardware before making memory-fit or cost
   claims. Attempt 08 requested A100-40GB but observed an A100-80GB. Any new run
   must state its accepted hardware policy explicitly and record actual hardware.

## Proposed bounded live check, subject to explicit approval

After the offline diagnostic is reviewable, propose at most **one additional
inference POST**, zero retries, the same eligibility prompt, pinned model/tokenizer,
512 output tokens, temperature 0, top_p 1 and five alternatives. Keep the original
120-second request, 600-second startup, 1200-second GPU-phase and 120-second cleanup
bounds, remote expiry, root `.env`, and retained cache policy. Recheck the existing
USD 5 allowance against current provisional usage; do not reset spent activity or
claim the operator-confirmed USD 30 monthly limit is API-verified.

This is a separate decision because the completed pilot's failure gate forbids
automatic continuation. Seven unsent measurement prompts do not authorize a
diagnostic replay. No calibration, ensemble, model swap or vLLM upgrade is bundled
with this proposal.

## What would count as progress

If structural evidence identifies a parser defect, implement an explicit,
documented mapping and validate it against fixtures before a separately bounded
eligibility rerun. If the endpoint lacks aligned final-token probabilities, mark
that runtime configuration ineligible and document the provider limitation.
Successful alignment establishes evidence eligibility only; it does not establish
calibration quality or [Jane Street divergence decoding](https://blog.janestreet.com/mitigating-memorization-in-llms/)
efficacy. A useful negative result is a bounded, reproducible explanation that
prevents a larger invalid measurement batch.
