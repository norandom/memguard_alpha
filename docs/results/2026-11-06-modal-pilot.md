# Modal pilot: staging succeeded, inference remains unverified

The approved GPT-OSS-20B pilot did not reach inference. Model staging succeeded;
deployment failed before readiness. This is a **deployment integration failure**,
not evidence that GPT-OSS cannot fit the selected GPU or cannot supply logprobs.
All experiment apps were subsequently confirmed stopped.

## What was actually verified

- Root `.env` account authentication and proxy-token registration in `wishinet`.
  The proxy secret was not exercised against a running endpoint.
- Dedicated `recall-guard-pilot` environment and retained weight volume.
- Ten selected files: 13,789,255,711 bytes. Three safetensors shards matched
  primary repository SHA-256 values; remote inventory matched selected paths/sizes.
- Pinned tokenizer and chat template: eight inputs of 73–76 tokens, each within
  the 4096-token context including its 512-token output allowance.
- The corrected vLLM registry image built successfully in the final attempt.
- Final read-only inventory: five experiment app entries, all `APP_STATE_STOPPED`.

## Experiment track

| Attempt | Observed outcome | Inference POSTs | Change or limitation |
| --- | --- | ---: | --- |
| 01 | Controller deployment rejected with `InvalidError` | 0 | Serialized Python 3.14 tooling differed from Python 3.12 image; tooling now pinned to 3.12 with local guards |
| CPU diagnostic | Same rejection reproduced with sanitized traceback | 0 | Controller-only; no model app or GPU invocation |
| 02 | Registry image build failed with `ImageBuildError` | 0 | Modal bootstrap required `python`; pinned image exposed `python3`. Scoped alias added without replacing its interpreter |
| 03 | Image built, later deployment rejected with `InvalidError` | 0 | The runner retained only exception type; exact remaining rejection needs diagnostic review before another paid attempt |
| 04, diagnostic follow-up | SDK rejected image-build ordering with `InvalidError` at `model_deploy` | 0 | Sanitized message now identifies `image.env(...)` after deferred local files; moved image configuration before all local-file additions and added an ordering regression |

Attempts 02 and 03 reserved 302.345 and 206.183 seconds respectively between
deployment start and failure. These are deployment-phase wall times, **not
observed GPU generation times**. No readiness call, token evidence, calibration,
single-answer score or ensemble measurement completed. The eight-request
inference allowance was not consumed.

The two focused remediations passed independent review. Latest focused suite:
103 deployment, runner and accounting tests passed after the diagnostic follow-up. This establishes offline
behavior only; live Modal support remains unverified.

## Budget and cleanup

The user approved a USD 5 planning allowance and confirmed a USD 30 monthly
workspace usage limit. The latter is operator-confirmed: budget API access was
denied. The approval was not derived from Starter's advertised monthly credits.

Latest workspace billing snapshot reported gross USD 0.02792672 and billed USD
0, with an Ephemeral Apps breakdown of USD 0.00362059. These are provisional
workspace-wide observations, not a reconciled pilot invoice; no final attributable
pilot cost or cost per usable measurement is claimed.

Model weights remain stored locally under ignored `models/` and remotely in the
retained volume. List storage for 12.842 GiB is approximately USD 1.16/month before
allowances. Stopping apps does not delete that storage. The approved scope retains it.

## Next decision

The separately initiated diagnostic follow-up captured the actionable rejection:
Modal requires deferred `add_local_*` operations to follow all image-build
operations, including `image.env(...)`. The recipe now configures its environment
before adding the three allowlisted source/configuration files. The regression
failed before the fix; the focused deployment/runner/accounting suite passes
103 tests afterward. This is an offline correction awaiting a controlled live
retest, not proof of a working endpoint. The original 03 receipt's missing
message has not been reconstructed or overwritten.

After attempt 04, fresh provider inventory confirmed all seven experiment app
entries stopped. No further paid attempt was made automatically. Preserve all
four receipts; keep the model, GPU and eight-request allowance explicit.

This pilot provides **no measurement of divergence decoding efficacy**. Hosting,
token evidence, calibration quality and mitigation effects remain separate gates;
see [Jane Street's method](https://blog.janestreet.com/mitigating-memorization-in-llms/).
The frontier completion-only path remains separate.

## Local diagnostic preparation — 2026-11-06

The runner now retains bounded, sanitized deployment/control error messages,
stage names and stack-frame metadata in new receipts. It preserves the primary
reason when cleanup or final provenance collection also fails. Inference errors
retain metadata without response text. These changes are verified only with
offline synthetic failures; no further deployment, control-plane read, GPU or
inference call was made for this preparation. Earlier receipts remain unchanged.

Task 10.2 remains blocked. The next step is independent review of this diagnostic
change and human review of a concrete proposal for obtaining the remaining
deployment rejection detail within an explicitly approved scope. The change does
not itself establish that rejection's cause or authorize another paid attempt.

Operator instructions: [Modal provider](../modal-provider.md).
Private, ignored receipts: `runs/modal-preflight/remote-staging.json`,
`local-staging.json`, `controller-debug.json`, `pilot-2026-11-06-01.json` through
`pilot-2026-11-06-03.json`, and `final-observations.json`. Credentials and generated
response text are not published.
