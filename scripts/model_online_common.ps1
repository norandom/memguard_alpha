<#
.SYNOPSIS
Shared checks of the start_model_online_*.ps1 scripts: repository root, pinned model file, runtime.

.DESCRIPTION
Dot-sourced, never run. The start scripts check that the model and the runtime are there, fetch a
missing model file after the person has accepted its terms, and then hand over to the host scripts
(serve_local_llamacpp.ps1, serve_full_distribution.ps1, serve_local_flm.ps1), which load the model.
No credential value is printed: HF_TOKEN is read from the root .env or the environment by name.
#>

$ErrorActionPreference = 'Stop'
$script:OnlineRoot = Split-Path -Parent $PSScriptRoot
$script:GemmaTerms = 'https://ai.google.dev/gemma/terms'
#: Exit codes: 0 ready or started, 1 a step failed, 2 the person has something to do first.
$script:ExitTodo = 2

function Assert-RepositoryRoot {
    $root = [IO.Path]::GetFullPath($script:OnlineRoot)
    if ($root -eq [IO.Path]::GetPathRoot($root)) { throw "The repository must not be a drive root; got $root." }
    if (-not (Test-Path -LiteralPath (Join-Path $root 'pyproject.toml'))) { throw "$root is not the repository root." }
    return $root
}

function Read-HostProfile([string]$ProfileId) {
    $path = Join-Path $script:OnlineRoot "recall_guard/core/provider_profiles/$ProfileId.json"
    return Get-Content -LiteralPath $path -Raw | ConvertFrom-Json
}

function Read-EnvValue([string]$Name) {
    # The process environment first, then the root .env; the value is returned, never written to output.
    $value = [Environment]::GetEnvironmentVariable($Name)
    if ($value) { return $value }
    $envPath = Join-Path $script:OnlineRoot '.env'
    if (-not (Test-Path -LiteralPath $envPath)) { return $null }
    foreach ($line in Get-Content -LiteralPath $envPath) {
        if ($line -match "^\s*$([regex]::Escape($Name))\s*=\s*(.*)$") {
            $found = $Matches[1].Trim().Trim('"').Trim("'")
            if ($found) { return $found }
        }
    }
    return $null
}

function Write-Todo([string[]]$Lines) {
    foreach ($line in $Lines) { Write-Output $line }
    exit $script:ExitTodo
}

function Confirm-ModelFile {
    <# The pinned model file under models/blobs, fetched when it is missing. Returns its path. #>
    param(
        [Parameter(Mandatory)][string]$Digest,          # "sha256:<64 hex>", from the provider profile
        [Parameter(Mandatory)][string]$Model,           # for messages
        [Parameter(Mandatory)][string]$RegistryPath,    # for example library/gemma3
        [switch]$AcceptModelTerms,
        [string]$HfRepo, [string]$HfFile,
        [string]$Rerun
    )
    if ($Digest -notmatch '^sha256:([0-9a-f]{64})$') { throw "The profile's artifact digest is not a SHA-256: $Digest" }
    $hex = $Matches[1]
    $blobs = Join-Path $script:OnlineRoot 'models/blobs'
    $target = Join-Path $blobs "sha256-$hex"
    if (Test-Path -LiteralPath $target -PathType Leaf) {
        Write-Output "model: $Model is present ($([math]::Round((Get-Item -LiteralPath $target).Length / 1GB, 2)) GiB, models/blobs/sha256-$($hex.Substring(0, 12))...)."
        return
    }
    if (-not $AcceptModelTerms) {
        Write-Todo @(
            "model: $Model is not under models/blobs.",
            "It is distributed under the Gemma Terms of Use: $script:GemmaTerms",
            "Read them. To accept them and download the file (about 2.5 GiB), run: $Rerun -AcceptModelTerms")
    }
    New-Item -ItemType Directory -Force -Path $blobs | Out-Null
    $partial = "$target.partial"
    $headers = @{}
    if ($HfRepo -or $HfFile) {
        if (-not ($HfRepo -and $HfFile)) { throw 'Pass -HfRepo and -HfFile together.' }
        $token = Read-EnvValue 'HF_TOKEN'
        if (-not $token) {
            Write-Todo @('model: HF_TOKEN is not set. Put a Hugging Face access token in the root .env as HF_TOKEN=...',
                         "and accept the model's license on https://huggingface.co/$HfRepo first.")
        }
        $headers['Authorization'] = "Bearer $token"
        $uri = "https://huggingface.co/$HfRepo/resolve/main/$HfFile"
        $source = "Hugging Face ($HfRepo)"
    } else {
        $uri = "https://registry.ollama.ai/v2/$RegistryPath/blobs/sha256:$hex"
        $source = "the Ollama registry ($RegistryPath)"
    }
    Write-Output "model: downloading $Model from $source ..."
    try {
        Invoke-WebRequest -Uri $uri -Headers $headers -OutFile $partial -Resume -MaximumRedirection 5
    } catch {
        $status = $_.Exception.Response.StatusCode.value__
        if ($status -in 401, 403) {
            Write-Todo @("model: the download was refused ($status).",
                         "Accept the model's license on its page and check HF_TOKEN in the root .env, then run this again.")
        }
        throw "The download failed: $($_.Exception.Message)"
    }
    $actual = (Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $hex) {
        Remove-Item -LiteralPath $partial -Confirm:$false
        throw "The downloaded file is not the pinned model: its SHA-256 is $actual, the profile pins $hex. Nothing was kept."
    }
    Move-Item -LiteralPath $partial -Destination $target
    Write-Output "model: $Model downloaded and verified."
}

function Confirm-LlamaCpp {
    <# llama-server of the build the profile pins. Returns nothing; stops with what to install otherwise. #>
    param([Parameter(Mandatory)][string]$RuntimeVersion, [string]$LlamaDir)
    $directory = if ($LlamaDir) { $LlamaDir } else {
        Join-Path $env:LOCALAPPDATA 'Microsoft/WinGet/Packages/ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe' }
    $exe = Join-Path $directory 'llama-server.exe'
    if (-not (Test-Path -LiteralPath $exe -PathType Leaf)) {
        Write-Todo @("runtime: llama-server.exe was not found in $directory.",
                     "Install llama.cpp $RuntimeVersion (winget install ggml.llamacpp), or pass -LlamaDir with the folder that holds it.")
    }
    $text = (& $exe --version 2>&1 | ForEach-Object { "$_" }) -join "`n"
    $build = if ($text -match '\(build (\d+)\b') { $Matches[1] } elseif ($text -match '(?m)^version: (\d+) \(') { $Matches[1] } else { $null }
    $wanted = $RuntimeVersion.TrimStart('b')
    if ($build -ne $wanted) {
        Write-Todo @("runtime: llama-server reports build $build; the profile pins $RuntimeVersion.",
                     "Install that build, or pass -LlamaDir with the folder that holds it. The host script refuses another build.")
    }
    Write-Output "runtime: llama.cpp $RuntimeVersion is present."
}

function Invoke-HostScript([string]$Name, [string[]]$Arguments) {
    # The pwsh that runs this script, by path: the Store alias of pwsh fails to launch on some machines.
    $pwsh = (Get-Process -Id $PID).Path
    # The host script's lines go to the console; only the exit code is returned to the caller.
    & $pwsh -NoProfile -File (Join-Path $PSScriptRoot $Name) @Arguments | Out-Host
    return $LASTEXITCODE
}
