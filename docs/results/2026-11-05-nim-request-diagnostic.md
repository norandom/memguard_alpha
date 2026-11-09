# NIM request diagnostic: API key attachment and the 504 outside Recall Guard, 2026-11-05

> Register: R09 · Status: diagnostic · Track: NIM

The API key is attached, and the HTTP 504 reproduces outside Recall Guard. An
outgoing-request inspection confirmed that the Bearer token exactly matches
`NVIDIA_API_KEY` loaded from the mounted project `.env`, and two direct requests
using NVIDIA's own example returned HTTP 504 without Recall Guard's adapter. The
root cause remains unresolved; an omitted key or a Recall Guard-only HTTP failure
does not explain these reproductions.

## Question

The user reports GPT-OSS-20B working on NVIDIA and questioned whether Recall
Guard sends the API key. Does Recall Guard attach the key, and does the 504
reproduce without Recall Guard?

## Setup

Run on November 5, 2026. One direct request used Python requests in the existing Docker environment; the
other used HttpClient on the host, bypassing Docker and WSL entirely. Both used
the example published on [NVIDIA's model page](https://build.nvidia.com/openai/gpt-oss-20b):

- Endpoint: `https://integrate.api.nvidia.com/v1/chat/completions`
- Model: `openai/gpt-oss-20b`
- Prompt: `Which number is larger, 9.11 or 9.8?`
- Settings: temperature 1, top_p 1, max_tokens 4096, stream false.
- Authentication: Bearer token from the project `.env`.
- No retries, redirects or status polling; one request per diagnostic.

No credential value was printed or included in the receipts.

## Data

NVIDIA's example prompt above. These were two additional diagnostic requests,
separate from the original [26-call pilot](2026-11-05-sp500-pilot.md).

## Results

| Check | Observed result |
| --- | --- |
| Recall Guard transport, intercepted before network I/O | Prepared Authorization header exactly matched the loaded key |
| Direct Python HTTP request in Docker | HTTP 504 after 302.458 seconds |
| Direct HttpClient request on host | HTTP 504 after 302.845 seconds |

Both returned before their local 600-second timeout. Neither reported usage;
monetary cost is unknown.

What the evidence rules out: the current request path does not omit
authentication. The failure is reproducible without Recall Guard's parsing,
pacing or readiness controls, and without Docker. The pilot's temperature 0,
smaller token cap and JSON eligibility prompt are also not necessary to reproduce
it: NVIDIA's example settings failed too.

## Limits

This does not prove that the configured key has working inference access, that
the model is generally unavailable, or that every account/network sees the same
result. No 401/403 was observed. A 504 alone cannot distinguish account routing,
service/backend behavior or other upstream causes.

The remaining comparison is the user's successful path: web playground versus
an API call, and whether it uses this same configured key. Do not ask the user to
paste credentials. Do not replace this investigation with local hosting or change
WSL configuration. No further inference attempts were started after these results.

## Evidence

Ignored receipts are under:

- `runs/container/2026-11-05-nim-diagnostic/`: sanitized header check, request intent and Docker result.
- `runs/container/2026-11-05-nim-direct-host/`: sanitized request intent and host result.
