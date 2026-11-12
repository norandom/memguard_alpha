<#
.SYNOPSIS
Start the local model for the online logprobs example: Gemma 3 4B on llama.cpp (Windows).

.DESCRIPTION
For docs/notebooks/recall_adjusted_answers.ipynb. Three steps, each reported in one line:

1. model: the GGUF file the llamacpp-gemma3-4b profile pins must be under models/blobs. If it is
   not, the script names the Gemma Terms of Use and stops; with -AcceptModelTerms it downloads the
   file and keeps it only if its SHA-256 is the pinned one. The default source is the Ollama
   registry, which serves the pinned file by its digest. With -HfRepo and -HfFile it downloads from
   Hugging Face with HF_TOKEN (root .env); the SHA-256 must still be the pinned one.
2. runtime: llama-server of the pinned build must be installed.
3. start: scripts/serve_local_llamacpp.ps1 loads the model and waits until it answers.

Only one model host runs at a time: stop this one before starting the logits or the NPU host.

.EXAMPLE
pwsh -File scripts/start_model_online_logprobs.ps1            # check, then start
pwsh -File scripts/start_model_online_logprobs.ps1 check      # check only, start nothing
pwsh -File scripts/start_model_online_logprobs.ps1 status
pwsh -File scripts/start_model_online_logprobs.ps1 stop
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
$hostScript = 'serve_local_llamacpp.ps1'
$passLlama = if ($LlamaDir) { @('-LlamaDir', $LlamaDir) } else { @() }

if ($Action -in 'status', 'stop') { exit (Invoke-HostScript $hostScript (@($Action) + $passLlama)) }

$hostProfile = Read-HostProfile 'llamacpp-gemma3-4b'
$declared = $hostProfile.model_config.local_identity.declared
Confirm-ModelFile -Digest $declared.artifact_digest -Model $hostProfile.model_config.model -RegistryPath 'library/gemma3' `
    -AcceptModelTerms:$AcceptModelTerms -HfRepo $HfRepo -HfFile $HfFile -Rerun 'pwsh -File scripts/start_model_online_logprobs.ps1'
Confirm-LlamaCpp -RuntimeVersion $declared.runtime_version -LlamaDir $LlamaDir
if ($Action -eq 'check') { Write-Output 'check: model and runtime are ready. Nothing was started.'; exit 0 }

$code = Invoke-HostScript $hostScript (@('start') + $passLlama)
if ($code -eq 0) {
    Write-Output 'start: the logprobs host is up. Run the "Online, logprobs" cells of the notebook now.'
    Write-Output 'When done: pwsh -File scripts/start_model_online_logprobs.ps1 stop'
}
exit $code
