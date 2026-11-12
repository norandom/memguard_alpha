<#
.SYNOPSIS
Start the local logits host for the online logits example: Gemma 3 4B, next-token vectors (Windows).

.DESCRIPTION
For docs/notebooks/recall_adjusted_answers.ipynb. The logits host serves the same model file as the
logprobs host and returns the next-token vector for a given reply. Only one of the two runs at a
time, so the notebook draws its replies with the logprobs host first; stop that host, then start
this one.

1. model: the pinned GGUF file under models/blobs (see start_model_online_logprobs.ps1 for the
   download and the Gemma Terms of Use).
2. runtime: llama.cpp of the pinned build, and uv, which runs the host.
3. start: scripts/serve_full_distribution.ps1 loads the model and waits until it listens.

There is no Linux script for this host: scripts/serve_full_distribution.py starts the Windows build
of llama-server.

.EXAMPLE
pwsh -File scripts/start_model_online_logits.ps1            # check, then start
pwsh -File scripts/start_model_online_logits.ps1 check
pwsh -File scripts/start_model_online_logits.ps1 status
pwsh -File scripts/start_model_online_logits.ps1 stop
#>
param(
    [Parameter(Position = 0)][ValidateSet('start', 'check', 'status', 'stop')][string]$Action = 'start',
    [switch]$AcceptModelTerms,
    [string]$HfRepo,
    [string]$HfFile,
    [string]$LlamaDir = ''
)

. (Join-Path $PSScriptRoot 'model_online_common.ps1')
Assert-RepositoryRoot | Out-Null
$hostScript = 'serve_full_distribution.ps1'

if ($Action -in 'status', 'stop') { exit (Invoke-HostScript $hostScript @($Action)) }

$hostProfile = Read-HostProfile 'llamacpp-gemma3-4b'
$declared = $hostProfile.model_config.local_identity.declared
Confirm-ModelFile -Digest $declared.artifact_digest -Model $hostProfile.model_config.model -RegistryPath 'library/gemma3' `
    -AcceptModelTerms:$AcceptModelTerms -HfRepo $HfRepo -HfFile $HfFile -Rerun 'pwsh -File scripts/start_model_online_logits.ps1'
Confirm-LlamaCpp -RuntimeVersion $declared.runtime_version -LlamaDir $LlamaDir
if (-not (Get-Command uv -CommandType Application -ErrorAction SilentlyContinue)) {
    Write-Todo @('runtime: uv was not found. Install it (https://docs.astral.sh/uv/) and run `uv sync` in the repository.')
}
Write-Output 'runtime: uv is present.'
if ($Action -eq 'check') { Write-Output 'check: model and runtime are ready. Nothing was started.'; exit 0 }

$code = Invoke-HostScript $hostScript @('start')
if ($code -eq 0) {
    Write-Output 'start: the logits host is up. Run the "Online, logits" cells of the notebook now.'
    Write-Output 'When done: pwsh -File scripts/start_model_online_logits.ps1 stop'
}
exit $code
