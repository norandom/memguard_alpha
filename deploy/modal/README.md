# Optional Modal deployment setup

This directory isolates Modal from the core library. The Python tooling lock
pins Modal 1.6.3 and its transitive packages; the serving image and model/tokenizer
commits are frozen in `configs/modal/gpt-oss-20b-pilot.json`. Primary registry,
license, resource-unit and pricing provenance is included in that manifest.
Registry metadata and the Apache-2.0 license were accessed publicly on
2026-11-06; no weights were downloaded. Retain the license, attribution and
applicable notices if redistributing artifacts. Public artifact access does not
verify the operator's Modal account permissions or paid allowance.

Run this local-only plan from the checkout in PowerShell:

```powershell
uv run --no-sync python deploy/modal/manifest.py configs/modal/gpt-oss-20b-pilot.json
```

It does not import Modal, read credentials, contact a network, build images,
stage weights, deploy, or call inference. The output reports unresolved workspace,
endpoint and provider-budget gates. Those facts remain unknown until the
separately approved execution workflow verifies them. The proposed environment
and app names are identifiers, not evidence that cloud resources exist.
`MODAL_PROXY_BEARER` is only an environment-variable reference: the eventual
operator supplies a proxy token ID and secret joined with a period outside
tracked files. Never put token values in the manifest.

Tooling can later be installed explicitly with `uv sync --project deploy/modal
--locked`; this is separate from library installation and is not needed for
planning. The lock was resolved without installing the SDK or building an image.
The released Linux AMD64 image is pinned by its platform manifest digest; its
multi-platform index digest is retained in provenance. Published vLLM recipes
support this A100/MXFP4 candidate; actual hardware fit, startup and token
evidence remain unverified until the separately approved pilot.

The frozen candidate requests four physical CPU cores and 32 GiB RAM (32768 MiB
for the Modal API). These are resource requests; CPU/RAM usage above requests
can affect charges. No hardware escalation or model substitution is planned.
One eligibility request plus seven measurements is eight physical POSTs,
zero retries and at most 4096 generated tokens. The 1200-second GPU phase includes
600 seconds startup, followed by a separate 120-second cleanup allowance.

At the recorded rates, the 20-minute GPU/CPU/RAM resource estimate is
`1200 * (0.000583 + 4 * 0.0000131 + 32 * 0.00000222) = USD 0.847728`.
It excludes separately costed staging, builds, storage, expiry controller,
cleanup, egress and excess resource use. USD 5 is a target, not a guaranteed
ceiling; monthly provider budgets require prior-usage and coverage checks.
There is no verified configured provider budget in this setup.

Future serving code must stage the pinned artifacts into the dedicated remote
volume using a separate CPU-only operation and retain that volume after cleanup.
No staging operation is implemented or invoked here. Never upload the checkout
recursively. The manifest's exact upload allowlist permits only serving/planning
source and this configuration; it excludes `models/`, `localmodels/`, `.env`,
credential files and all unspecified files. Upload code must reject symlinks and
verify resolved paths stay within the checkout before applying the allowlist.
This setup supplies no uploader or serving recipe; those belong to task 8.2.
