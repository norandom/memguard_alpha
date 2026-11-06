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

Attempts 02 and 03 reserved 302.345 and 206.183 seconds respectively between
deployment start and failure. These are deployment-phase wall times, **not
observed GPU generation times**. No readiness call, token evidence, calibration,
single-answer score or ensemble measurement completed. The eight-request
inference allowance was not consumed.

The two focused remediations passed independent review. Latest focused suite:
87 deployment, runner and accounting tests passed. This establishes offline
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

Live acceptance is blocked pending diagnostic review of the final deployment
rejection. Further paid retries were stopped after the two debugging/remediation
rounds. Preserve the three receipts and improve sanitized deployment-error detail
before proposing another attempt. Do not change models, GPU type or inference
limits to obscure the deployment failure.

This pilot provides **no measurement of divergence decoding efficacy**. Hosting,
token evidence, calibration quality and mitigation effects remain separate gates;
see [Jane Street's method](https://blog.janestreet.com/mitigating-memorization-in-llms/).
The frontier completion-only path remains separate.

Operator instructions: [Modal provider](../modal-provider.md).
Private, ignored receipts: `runs/modal-preflight/remote-staging.json`,
`local-staging.json`, `controller-debug.json`, `pilot-2026-11-06-01.json` through
`pilot-2026-11-06-03.json`, and `final-observations.json`. Credentials and generated
response text are not published.
