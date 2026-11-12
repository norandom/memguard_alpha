<#
.SYNOPSIS
Start a local model on the NPU for the online NPU example: FastFlowLM (Windows only, logprobs only).

.DESCRIPTION
For docs/notebooks/recall_adjusted_answers.ipynb. FastFlowLM runs on the AMD NPU natively on
Windows, so there is no Linux script. It returns logprobs with a reply. No logits host serves its
NPU weights, so the logits mode is not available on this route.

1. runtime: flm.exe must be installed (FastFlowLM).
2. model: the tag must be installed in the repository's FLM store, models/flm. If it is not, the
   script stops; with -AcceptModelTerms it runs `flm pull <tag>`, which downloads the model under
   that model's license (Gemma Terms of Use for the Gemma tags). HF_TOKEN from the root .env is
   passed to flm when it is set.
3. start: scripts/serve_local_flm.ps1 serves the tag.

Only one model host runs at a time: stop the llama.cpp hosts first.

.EXAMPLE
pwsh -File scripts/start_model_online_npu.ps1                     # gemma3:4b: check, then start
pwsh -File scripts/start_model_online_npu.ps1 start llama3.2:3b
pwsh -File scripts/start_model_online_npu.ps1 check
pwsh -File scripts/start_model_online_npu.ps1 stop
#>
param(
    [Parameter(Position = 0)][ValidateSet('start', 'check', 'status', 'stop')][string]$Action = 'start',
    [Parameter(Position = 1)][ValidatePattern('^[a-z0-9][a-z0-9.\-]*:[a-z0-9][a-z0-9.\-]*$')][string]$Model = 'gemma3:4b',
    [switch]$AcceptModelTerms,
    [string]$FlmPath
)

. (Join-Path $PSScriptRoot 'model_online_common.ps1')
$root = Assert-RepositoryRoot
$hostScript = 'serve_local_flm.ps1'
$passFlm = if ($FlmPath) { @('-FlmPath', $FlmPath) } else { @() }

if ($Action -in 'status', 'stop') { exit (Invoke-HostScript $hostScript (@($Action) + $passFlm)) }

# 1. runtime
$flm = if ($FlmPath) { $FlmPath } else {
    $command = Get-Command flm -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($command) { $command.Source } else {
        # Not on PATH: the installer's folder under the user's programs, newest version first.
        Get-ChildItem -Path (Join-Path $env:LOCALAPPDATA 'Programs') -Filter 'FastFlowLM*' -Directory -ErrorAction SilentlyContinue |
            Sort-Object Name -Descending | ForEach-Object { Join-Path $_.FullName 'flm.exe' } |
            Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
    }
}
if ($flm -and -not $FlmPath) { $passFlm = @('-FlmPath', $flm) }
if (-not $flm -or -not (Test-Path -LiteralPath $flm -PathType Leaf)) {
    Write-Todo @('runtime: flm.exe was not found.',
                 'Install FastFlowLM for Windows (it needs an AMD Ryzen AI NPU), or pass -FlmPath with the path of flm.exe.')
}
Write-Output 'runtime: FastFlowLM is present.'

# 2. model, in the repository's FLM store
$store = Join-Path $root 'models/flm'
New-Item -ItemType Directory -Force -Path $store | Out-Null
$env:FLM_MODEL_PATH = (Resolve-Path -LiteralPath $store).Path
$env:FLM_DISABLE_UPDATE_CHECK = '1'
function Test-Installed {
    $listing = (& $flm list --filter installed --json 2>$null | Out-String)
    if ($LASTEXITCODE -ne 0) { throw 'flm list --filter installed --json failed.' }
    return [bool](@(($listing | ConvertFrom-Json -AsHashtable).models) | Where-Object { $_.installed -and $_.name -eq $Model })
}
if (-not (Test-Installed)) {
    if (-not $AcceptModelTerms) {
        Write-Todo @("model: $Model is not installed in models/flm.",
                     "FastFlowLM downloads it under that model's license (Gemma tags: $script:GemmaTerms).",
                     "Read it. To accept it and download, run: pwsh -File scripts/start_model_online_npu.ps1 start $Model -AcceptModelTerms")
    }
    $token = Read-EnvValue 'HF_TOKEN'
    if ($token) { $env:HF_TOKEN = $token }
    Write-Output "model: flm pull $Model ..."
    & $flm pull $Model
    if ($LASTEXITCODE -ne 0 -or -not (Test-Installed)) {
        throw "flm pull $Model did not install the model. If the download was refused, accept the model's license on its Hugging Face page and set HF_TOKEN in the root .env."
    }
}
Write-Output "model: $Model is installed in models/flm."
if ($Action -eq 'check') { Write-Output 'check: model and runtime are ready. Nothing was started.'; exit 0 }

# 3. start
$code = Invoke-HostScript $hostScript (@('start', $Model) + $passFlm)
if ($code -eq 0) {
    Write-Output 'start: the NPU host is up. Run the "Online, NPU" cells of the notebook now.'
    Write-Output 'When done: pwsh -File scripts/start_model_online_npu.ps1 stop'
}
exit $code
