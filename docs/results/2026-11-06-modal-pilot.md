# Modal GPT-OSS-20B evidence pilot, 2026-11-06

> Register: R13 · Status: failed · Track: Modal

The approved GPT-OSS-20B pilot reached model readiness and sent one eligibility
request. Strict evidence validation rejected the response with
`token_text_alignment_failed`; the runner stopped without sending the seven
measurement prompts. Hosting works for this observed configuration, but it is
blocked by an integration bug, not a model capability failure.

The [subsequent response-handling retest](2026-11-06-gpt-oss-response-mapping.md)
confirmed the defect: full-generation Harmony probabilities were attached to
final-answer text. An explicit channel mapper passed strict validation on replay
of that new diagnostic response. This report preserves attempts 01-08 and their
original validation outcomes; it does not classify GPT-OSS as lacking logprobs.

## Question

Can the approved Modal pilot host GPT-OSS-20B to model readiness and pass strict
evidence eligibility before its measurement prompts?

## Setup

What was verified:

- Root `.env` account authentication and proxy-token registration in `wishinet`.
- Registered endpoint identity and unauthenticated HTTP 401 before GPU readiness
  in attempts 07 and 08. Attempt 08 reached authenticated generation.
- Dedicated `recall-guard-pilot` environment and retained weight volume.
- Ten selected files: 13,789,255,711 bytes. Three safetensors shards matched
  primary repository SHA-256 values; remote inventory matched selected paths/sizes.
- The corrected vLLM registry image built successfully; the model app deployed.
- The independent CPU expiry input acknowledged the model app ID and deadline
  before the readiness request. Client-loss expiry has not been live exercised.
- Attempt 08 observed vLLM 0.17.0, the pinned model/tokenizer revision and stable
  before/after identity. Startup took 103.631 seconds; the eligibility request
  took 2.969 seconds including transport/validation, not generation alone.
- The request specified A100-40GB, but the observed device was A100-SXM4-80GB
  with 81920 MiB. Provider placement does not prove 40 GB fit or validate 40 GB
  pricing for actual usage. The client did not change its GPU request.
- Model-app cleanup was verified in the runner. Separate final inventory
  confirmed all fifteen owned model, expiry and diagnostic entries stopped.

Budget and cleanup: the user approved a USD 5 planning allowance and confirmed a
USD 30 monthly workspace usage limit. The latter is operator-confirmed: budget
API access was denied. The approval was not derived from Starter's advertised
monthly credits.

The post-attempt-08 workspace snapshot reported gross USD 0.15792723 and billed
USD 0. These are provisional workspace-wide observations, not a reconciled pilot
invoice; no final attributable pilot cost or cost per usable measurement is
claimed.

Model weights remain stored locally under ignored `models/` and remotely in the
retained volume. List storage for 12.842 GiB is approximately USD 1.16/month
before allowances. Stopping apps does not delete that storage. The approved scope
retains it.

## Data

Eight allowed requests: one eligibility request and seven measurement prompts.
Pinned tokenizer and chat template: eight inputs of 73–76 tokens, each within the
4096-token context including its 512-token output allowance. Generated text was
not retained.

## Results

| Attempt | Observed outcome | Inference POSTs | Change or limitation |
| --- | --- | ---: | --- |
| 01 | Controller deployment rejected with `InvalidError` | 0 | Serialized Python 3.14 tooling differed from Python 3.12 image; tooling now pinned to 3.12 with local guards |
| CPU diagnostic | Same rejection reproduced with sanitized traceback | 0 | Controller-only; no model app or GPU invocation |
| 02 | Registry image build failed with `ImageBuildError` | 0 | Modal bootstrap required `python`; pinned image exposed `python3`. Scoped alias added without replacing its interpreter |
| 03 | Image built, later deployment rejected with `InvalidError` | 0 | The runner retained only exception type; exact remaining rejection needs diagnostic review before another paid attempt |
| 04, diagnostic follow-up | SDK rejected image-build ordering with `InvalidError` at `model_deploy` | 0 | Sanitized message now identifies `image.env(...)` after deferred local files; moved image configuration before all local-file additions and added an ordering regression |
| 05 | Unhydrated SDK object captured by serialized serving class | 0 | Removed App/Dict closure captures; runtime identity and Dict creation now explicit; actual SDK serialization regression added |
| 06 | Deployed and armed expiry, but guessed endpoint returned HTTP 404 | 0 | Manifest now uses the provider-registered regional URL; identity check moved before HTTP and GPU readiness |
| 07 | HTTP 401 confirmed, then repeated container startup failures | 0 | Operator stopped app. Logs showed a `Sentinel` import failure through the inherited vLLM CLI entrypoint; pinned image configuration verified that entrypoint |
| CPU bootstrap 01 | Bootstrap ran; dependency import failed | 0 | Observed typing_extensions 4.12.2 without Sentinel; no GPU |
| CPU bootstrap 02 | Bootstrap and dependency import passed | 0 | Observed typing_extensions 4.16.0 and pydantic_core 2.41.5; no GPU |
| 08 | Model ready; eligibility response failed text alignment | 1 | Stable observed A100-80GB runtime; stopped immediately with verified cleanup; zero measurement prompts |

Attempts 02 and 03 reserved 302.345 and 206.183 seconds respectively between
deployment start and failure. These are deployment-phase wall times, not
observed GPU generation times. Attempts 01–07 sent no inference. Attempt 08
used one of the eight allowed requests; the failure gate prevented the remaining
seven. No usable token measurement, calibration, single-answer score or ensemble
measurement completed. Unused allowance does not bypass the stop gate.

How startup was repaired: the pinned image configuration contains
`ENTRYPOINT ["vllm", "serve"]`. Modal's registry-image construction preserves
it. The remote traceback starts in that CLI, not in the serving hook's
`python -m` subprocess. Clearing the inherited entrypoint is therefore a verified
configuration repair; it allows Modal's bootstrap arguments to run. See
[Modal's image guidance](https://modal.com/docs/guide/existing-images).

The immediate exception was `ImportError` for `Sentinel` from `typing_extensions`
while importing `pydantic_core`. A CPU-only diagnostic then observed effective
version 4.12.2, missing `Sentinel`, and reproduced that import failure. The recipe
now pins 4.16.0 in the derived image. A second CPU check observed that version,
found `Sentinel`, and successfully imported `pydantic_core` 2.41.5 under Python
3.12.13. Both diagnostic apps were stopped; neither used a GPU or generated tokens.
This established the import repair before attempt 08 tested model readiness.

Attempt 07's original runner receipt records cleanup as unverified because the
operator had already stopped its app. A separate provider inventory subsequently
confirmed the model app stopped; it does not overwrite the original receipt.
Separate provider inventory after the first CPU check confirmed all fourteen
experiment entries stopped. The second CPU check also confirmed its own app
stopped. Final inventory after attempt 08 confirmed all fifteen owned app entries
stopped. Earlier receipts remain unchanged.

## Limits

- Generated text was not retained, so this receipt cannot establish the exact
  alignment discrepancy.
- This pilot provides no measurement of divergence decoding efficacy. Hosting,
  token evidence, calibration quality and mitigation effects remain separate
  gates; see [Jane Street's method](https://blog.janestreet.com/mitigating-memorization-in-llms/).
  The frontier completion-only path remains separate.
- Startup failures also consumed compute/control activity despite sending zero
  inference requests.

Diagnostic handling: the runner now retains bounded, sanitized
deployment/control error messages, stage names and stack-frame metadata in new
receipts. It preserves the primary reason when cleanup or final provenance
collection also fails. Inference errors retain metadata without response text.
These changes are verified with offline synthetic failures and were exercised
during later controlled attempts. Earlier receipts remain unchanged.

Task 10.2 has a historical bounded outcome: ready host, strict evidence rejection
and verified shutdown. That rejection is now classified as a response-handling
integration defect. Task 10.3 records the repair and same-response replay; neither
task establishes calibrated scoring quality.
The next diagnostic proposal is recorded in
`.kiro/specs/local-ai-provider/modal-alignment-diagnostic-proposal.md`; the user
subsequently authorized its single diagnostic request. Calibration and ensemble
work remain separately budgeted.

## Evidence

The [sanitized numeric track](2026-11-06-modal-pilot.json) preserves source hashes
and separates inference, evidence, environment and cleanup outcomes.

The serialization and endpoint remediations passed independent review. The latest
integrated provider suite passed 153 tests with one optional SDK skip. This
establishes offline behavior; live readiness evidence comes from attempt 08.
The five-model local Ollama smoke is
[reported separately](2026-11-06-local-model-smoke.md).

Operator instructions: [Modal provider](../guides/modal-provider.md).
Private, ignored receipts: `runs/modal-preflight/remote-staging.json`,
`local-staging.json`, `controller-debug.json`, `pilot-2026-11-06-01.json` through
`pilot-2026-11-06-08.json`, both CPU import checks, registered endpoint metadata
and cleanup inventories. Credentials and generated response text are not
published.
