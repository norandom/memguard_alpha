<#
.SYNOPSIS
Task 1.2 prerequisite receipt for the memorization-adjusted finance simulation.

.DESCRIPTION
Host-side tooling (PowerShell 7, no admin, no Windows Python). Two phases:

1. Host facts, checked natively on Windows and written to
   runs/finance-sim/prerequisites/host/host_facts.json:
   - llama.cpp b12199 (winget ggml.llamacpp): llama.dll, ggml.dll, ggml-vulkan.dll
     and llama-server.exe present, build number from `llama-server --version`.
   - Gemma 3 4B GGUF: the model layer of the Ollama manifest gemma3/4b under the
     project models/ folder, with size, GGUF magic and a recomputed sha256 that
     must equal the manifest digest (the digest the 3.3 profile will pin).
   - Node on PATH, and the dataviz skill's validate_palette.js located (newest
     match), hashed and smoke-run with its documented example palette.
   - The environment template (.env.sample) lists MODAL_ENDPOINT_URL and
     MODAL_PROXY_BEARER by name.

2. Live checks in the dev container: loads only the known credential names from
   the root .env into THIS process (never printed, never written elsewhere),
   forwards them to WSL through WSLENV for this one call, and runs
   `dagger -m ci call run-with-secrets` with `--<name>=env://<NAME>` for each set
   name. The container runs scripts/finance_sim/check_prerequisites.py, folds in
   the host facts and exports receipt.json and summary.md to
   runs/finance-sim/prerequisites/. Names that are empty in .env are not passed,
   so the receipt shows them as not set.

The Dagger log goes to runs/finance-sim/validation/. After the run, the log and
the receipt files are scanned in-process for every loaded secret value; only the
result (clean, or the offending names) is printed.

.EXAMPLE
pwsh -NoProfile -File scripts/finance_sim/check_prerequisites.ps1
pwsh -NoProfile -File scripts/finance_sim/check_prerequisites.ps1 -HostOnly
#>
param(
    [switch]$HostOnly,
    [string]$Distro = 'Debian',
    [string]$DaggerPath = '/home/mc/.local/bin/dagger',
    [string]$LlamaDir = (Join-Path $env:LOCALAPPDATA 'Microsoft\WinGet\Packages\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe'),
    [int]$ExpectedLlamaBuild = 12199,
    [string[]]$ValidatorSearchRoots = @('D:\BuildArtifacts\Temp\claude\bundled-skills',
                                        (Join-Path $env:TEMP 'claude\bundled-skills'),
                                        (Join-Path $HOME '.claude'))
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$OutDir = Join-Path $Root 'runs/finance-sim/prerequisites'
$HostDir = Join-Path $OutDir 'host'
$LogDir = Join-Path $Root 'runs/finance-sim/validation'
$HostFactsSchema = 'finance-sim-host-facts/1'
$SecretNames = @('FMP_API_TOKEN', 'FRED_API_KEY', 'HF_TOKEN', 'MODAL_TOKEN_ID', 'MODAL_TOKEN_SECRET',
                 'MODAL_PROXY_BEARER', 'MODAL_ENDPOINT_URL', 'LOCAL_MODEL_API_KEY')
$ExamplePalette = '#2a78d6,#eb6834,#1baf7a,#eda100,#e87ba4,#008300,#6250d6,#e34948'

function New-Check([string]$Id, [string]$Group, [bool]$Passed, [System.Collections.IDictionary]$Detail) {
    [ordered]@{ id = $Id; group = $Group; status = $(if ($Passed) { 'pass' } else { 'fail' }); required = $true; detail = $Detail }
}

function Get-Sha256([string]$Path) { (Get-FileHash -Algorithm SHA256 -LiteralPath $Path).Hash.ToLowerInvariant() }

function Test-LlamaCpp {
    $files = [ordered]@{}
    foreach ($name in 'llama.dll', 'ggml.dll', 'ggml-vulkan.dll', 'llama-server.exe') {
        $files[$name] = Test-Path -LiteralPath (Join-Path $LlamaDir $name)
    }
    $detail = [ordered]@{ path = $LlamaDir; files = $files; expected_build = $ExpectedLlamaBuild }
    if (-not $files['llama-server.exe']) { return New-Check 'llama_cpp_b12199' 'host' $false $detail }
    $versionText = (& (Join-Path $LlamaDir 'llama-server.exe') --version 2>&1 | Out-String)
    $build = if ($versionText -match 'build (\d+), commit ([0-9a-f]+)') { [int]$Matches[1] } else { $null }
    $detail.build = $build
    $detail.commit = if ($build) { $Matches[2] } else { $null }
    if ($files['llama.dll']) { $detail.llama_dll_sha256 = Get-Sha256 (Join-Path $LlamaDir 'llama.dll') }
    $allFiles = -not ($files.Values -contains $false)
    New-Check 'llama_cpp_b12199' 'host' ($allFiles -and $build -eq $ExpectedLlamaBuild) $detail
}

function Test-Gguf {
    $manifestRel = 'models/manifests/registry.ollama.ai/library/gemma3/4b'
    $manifest = Join-Path $Root $manifestRel
    $detail = [ordered]@{ manifest = $manifestRel }
    if (-not (Test-Path -LiteralPath $manifest)) { return New-Check 'gemma3_4b_gguf' 'host' $false $detail }
    $layer = (Get-Content -Raw -LiteralPath $manifest | ConvertFrom-Json).layers |
        Where-Object mediaType -eq 'application/vnd.ollama.image.model' | Select-Object -First 1
    $hex = $layer.digest -replace '^sha256:', ''
    $blobRel = "models/blobs/sha256-$hex"
    $blob = Join-Path $Root $blobRel
    $detail.digest = $layer.digest
    $detail.from = $layer.from
    $detail.path = $blobRel
    $detail.manifest_size = [int64]$layer.size
    if (-not (Test-Path -LiteralPath $blob)) { return New-Check 'gemma3_4b_gguf' 'host' $false $detail }
    $detail.size = (Get-Item -LiteralPath $blob).Length
    $stream = [System.IO.File]::OpenRead($blob)
    try { $magic = New-Object byte[] 4; [void]$stream.Read($magic, 0, 4) } finally { $stream.Dispose() }
    $detail.gguf_magic = [System.Text.Encoding]::ASCII.GetString($magic) -eq 'GGUF'
    $detail.sha256 = Get-Sha256 $blob
    $ok = $detail.gguf_magic -and $detail.sha256 -eq $hex -and $detail.size -eq $detail.manifest_size
    New-Check 'gemma3_4b_gguf' 'host' $ok $detail
}

function Test-Node {
    $cmd = Get-Command node -ErrorAction SilentlyContinue
    if (-not $cmd) { return New-Check 'node' 'tooling' $false ([ordered]@{ reason = 'node not on PATH' }) }
    $version = (& $cmd.Source --version 2>&1 | Out-String).Trim()
    New-Check 'node' 'tooling' ($LASTEXITCODE -eq 0 -and $version -match '^v\d+') ([ordered]@{ path = $cmd.Source; version = $version })
}

function Test-Validator {
    $roots = $ValidatorSearchRoots | Where-Object { $_ -and (Test-Path -LiteralPath $_) } | Select-Object -Unique
    $found = foreach ($r in $roots) {
        Get-ChildItem -LiteralPath $r -Recurse -Force -Filter 'validate_palette.js' -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -match '[\\/]dataviz[\\/]scripts[\\/]validate_palette\.js$' }
    }
    $found = @($found | Sort-Object FullName -Unique | Sort-Object LastWriteTime -Descending)
    $detail = [ordered]@{ search_roots = @($roots); candidates = @($found | ForEach-Object FullName) }
    if ($found.Count -eq 0) { return New-Check 'palette_validator' 'tooling' $false $detail }
    $path = $found[0].FullName
    $detail.path = $path
    $detail.sha256 = Get-Sha256 $path
    $detail.fragility = 'Claude Code bundled-skills cache under a temp directory; the path embeds the Claude Code ' +
        'version and a bundle hash, changes on every Claude Code update and may be cleaned. Re-locate it before ' +
        'task 8.1 with this script and compare the sha256; copy it into the repository only with a decision.'
    $node = Get-Command node -ErrorAction SilentlyContinue
    if ($node) {
        $null = & $node.Source $path $ExamplePalette --mode light 2>&1
        $detail.smoke_exit_code = $LASTEXITCODE
        $detail.smoke_args = "<documented example palette> --mode light"
    }
    New-Check 'palette_validator' 'tooling' ($null -ne $node -and $detail.smoke_exit_code -eq 0) $detail
}

function Test-EnvTemplate {
    $template = Join-Path $Root '.env.sample'
    $names = [ordered]@{}
    $lines = if (Test-Path -LiteralPath $template) { Get-Content -LiteralPath $template } else { @() }
    foreach ($name in 'MODAL_ENDPOINT_URL', 'MODAL_PROXY_BEARER') {
        $names[$name] = [bool]($lines | Where-Object { $_ -match "^\s*$name\s*=" })
    }
    New-Check 'env_template_names' 'credentials' (-not ($names.Values -contains $false)) ([ordered]@{ template = '.env.sample'; names = $names })
}

function Read-DotEnvSecrets {
    # Returns name -> value for the known names only. Values stay in memory.
    $values = @{}
    $envPath = Join-Path $Root '.env'
    if (-not (Test-Path -LiteralPath $envPath)) { return $values }
    foreach ($line in Get-Content -LiteralPath $envPath) {
        if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)$' -and $SecretNames -contains $Matches[1]) {
            $value = $Matches[2].Trim()
            $quoted = ($value.Length -ge 2) -and ($value[0] -eq $value[-1]) -and ([string]$value[0] -in @('"', "'"))
            if ($quoted) {
                $value = $value.Substring(1, $value.Length - 2)
            }
            if ($value) { $values[$Matches[1]] = $value }
        }
    }
    $values
}

New-Item -ItemType Directory -Force -Path $HostDir, $LogDir | Out-Null
$checks = @(Test-LlamaCpp; Test-Gguf; Test-Node; Test-Validator; Test-EnvTemplate)
$facts = [ordered]@{ schema = $HostFactsSchema; generated_at = (Get-Date).ToUniversalTime().ToString('s') + 'Z'; checks = $checks }
$factsPath = Join-Path $HostDir 'host_facts.json'
$facts | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $factsPath -Encoding utf8
foreach ($c in $checks) { '{0,-4}  {1}' -f $c.status, $c.id }
if ($HostOnly) { return }

$secrets = Read-DotEnvSecrets
$flags = @()
$forward = @()
foreach ($name in $SecretNames) {
    if ($secrets.ContainsKey($name)) {
        Set-Item -Path "Env:$name" -Value $secrets[$name]
        $forward += "$name/u"
        $flags += "--$($name.ToLowerInvariant().Replace('_', '-'))=env://$name"
    }
}
$env:WSLENV = (@($env:WSLENV -split ':' | Where-Object { $_ }) + $forward) -join ':'
'credentials passed by name: ' + (($SecretNames | ForEach-Object { "$_=$(if ($secrets.ContainsKey($_)) { 'set' } else { 'not set' })" }) -join ', ')

$stamp = (Get-Date).ToString('yyyyMMdd-HHmmss')
$logPath = Join-Path $LogDir "task-1.2-live-$stamp.log"
$command = 'uv run --no-sync python -m scripts.finance_sim.check_prerequisites --host-facts /inputs/host_facts.json --out /outputs'
$wslRoot = '/mnt/' + $Root.Substring(0, 1).ToLowerInvariant() + ($Root.Substring(2) -replace '\\', '/')
$daggerArgs = @('-d', $Distro, '--cd', $wslRoot, '--exec', $DaggerPath, '-m', 'ci', 'call',
    'run-with-secrets', '--source=.', '--inputs=runs/finance-sim/prerequisites/host', "--command=$command") + $flags +
    @('export', '--path=runs/finance-sim/prerequisites')
try {
    & wsl @daggerArgs *> $logPath
    $daggerExit = $LASTEXITCODE
} finally {
    foreach ($name in $SecretNames) { Remove-Item -Path "Env:$name" -ErrorAction SilentlyContinue }
}
"dagger exit code: $daggerExit (log: $([System.IO.Path]::GetRelativePath($Root, $logPath)))"

$scanned = @($logPath, $factsPath, (Join-Path $OutDir 'receipt.json'), (Join-Path $OutDir 'summary.md')) |
    Where-Object { Test-Path -LiteralPath $_ }
$leaks = foreach ($name in $secrets.Keys) {
    foreach ($file in $scanned) {
        $text = Get-Content -Raw -LiteralPath $file
        if ($text -and $text.Contains($secrets[$name])) { $name; break }
    }
}
$secrets.Clear()
'secret scan of log and receipt: ' + $(if ($leaks) { 'FOUND ' + (($leaks | Sort-Object -Unique) -join ', ') } else { 'clean' })
if ($daggerExit -ne 0 -or $leaks) { exit 1 }
