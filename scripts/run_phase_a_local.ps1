<#
.SYNOPSIS
Run release-1.0 Phase A for the five local Ollama profiles, one model at a time.

.DESCRIPTION
Host-side orchestration (PowerShell 7, no admin). It performs inference only
without -DryRun. Inputs are the frozen files written by
scripts/build_phase_a_inputs.py under runs/phase-a/<profile>/; the container
sees them read-only at /phase-a. Journals and reports go to
runs/container/phase-a/<profile>/ (= /outputs/phase-a/<profile>/ in the container).

Execution:
  1. Refuse to start if any journal, report or summary output already exists, if
     an input is missing, or if a manifest endpoint differs from the current
     gateway address (the endpoint is hashed into the plan; rebuild the inputs).
  2. serve_local_ollama.ps1 start (Windows Ollama behind the authenticated gateway).
  3. Per profile, sequentially: fresh Windows memory snapshot ->
     measure_local_models.py --execute --observer ollama --host-snapshot in the dev
     container (LOCAL_MODEL_API_KEY passed by name) -> unload the model through
     loopback Ollama and verify /api/ps is empty. A failing profile is recorded
     and the next profile still runs.
  4. finally: serve_local_ollama.ps1 stop, which fails unless nothing remains.
  A summary is written to runs/container/phase-a/summary-<UTC>.json.

-DryRun runs measure_local_models.py --dry-run per profile (no gateway, no
inference) and saves each safe plan export under runs/container/phase-a/dry-run/.

.EXAMPLE
pwsh -File scripts/run_phase_a_local.ps1 -DryRun
pwsh -File scripts/run_phase_a_local.ps1
#>
param(
    [switch]$DryRun,
    [string[]]$Profiles = @('ollama-phi4-mini', 'ollama-llama3.2-3b', 'ollama-gemma3-4b', 'ollama-llama3.1-8b',
                            'ollama-llava-llama3-ctx2048'),
    [string]$Distro = 'Debian',
    [string]$WslRepo = '/mnt/d/Source/memguard_alpha',
    [ValidateRange(10, 600)][int]$UnloadTimeoutSeconds = 60
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$InputDir = Join-Path $Root 'runs/phase-a'
$Outputs = Join-Path $Root 'runs/container/phase-a'
$Serve = Join-Path $PSScriptRoot 'serve_local_ollama.ps1'
$Snapshot = Join-Path $PSScriptRoot 'host_memory_snapshot.ps1'
$Ollama = 'http://127.0.0.1:11434'

function Invoke-Container([string]$ProfileId, [string[]]$Arguments) {
    $in = "/phase-a/$ProfileId"
    $command = "cd '$WslRepo' && docker compose run --rm -e LOCAL_MODEL_API_KEY " +
        "-v '$WslRepo/runs/phase-a:/phase-a:ro' dev uv run --no-sync python scripts/measure_local_models.py " +
        "--plan $in/plan.json --manifest $in/manifest.json --corpus $in/corpus.json " + ($Arguments -join ' ')
    $output = & wsl.exe -d $Distro -- sh -c $command 2>&1
    return @{ exit_code = $LASTEXITCODE; output = ($output | Out-String) }
}

function Get-LoadedModels {
    $ps = Invoke-RestMethod -Uri "$Ollama/api/ps" -TimeoutSec 10
    return @($ps.models | ForEach-Object { $_.name })
}

function Invoke-Unload([string]$Model) {
    # Loopback Ollama on this host; the container path uses the gateway instead.
    $body = @{ model = $Model; keep_alive = 0 } | ConvertTo-Json -Compress
    Invoke-RestMethod -Uri "$Ollama/api/generate" -Method Post -Body $body -ContentType 'application/json' -TimeoutSec 60 | Out-Null
    $deadline = [DateTime]::UtcNow.AddSeconds($UnloadTimeoutSeconds)
    do {
        $loaded = Get-LoadedModels
        if (-not $loaded) { return $true }
        Start-Sleep -Seconds 2
    } until ([DateTime]::UtcNow -gt $deadline)
    return $false
}

# ---------- refuse before any side effect ----------
foreach ($ProfileId in $Profiles) {
    foreach ($name in 'plan.json', 'manifest.json', 'corpus.json') {
        if (-not (Test-Path -LiteralPath (Join-Path $InputDir "$ProfileId/$name") -PathType Leaf)) {
            throw "Missing input runs/phase-a/$ProfileId/$name. Run scripts/build_phase_a_inputs.py first."
        }
    }
}

if ($DryRun) {
    $dryDir = Join-Path $Outputs 'dry-run'
    New-Item -ItemType Directory -Force -Path $dryDir | Out-Null
    $failed = @()
    foreach ($ProfileId in $Profiles) {
        $result = Invoke-Container $ProfileId @('--dry-run')
        [IO.File]::WriteAllText((Join-Path $dryDir "$ProfileId.json"), $result.output, [Text.UTF8Encoding]::new($false))
        $plan = try { ($result.output | ConvertFrom-Json).plan } catch { $null }
        if ($result.exit_code -ne 0 -or -not $plan) { $failed += $ProfileId; Write-Output "$ProfileId dry run FAILED (exit $($result.exit_code))"; continue }
        Write-Output ("{0}: {1} logical / {2} physical requests, max input {3} + output {4} tokens, context {5}" -f
            $ProfileId, $plan.logical_requests, $plan.physical_requests, $plan.limits.max_input_tokens,
            $plan.limits.max_output_tokens, $plan.limits.context_tokens)
    }
    if ($failed) { Write-Output "Dry run failed for: $($failed -join ', ')"; exit 1 }
    exit 0
}

$existing = @()
foreach ($ProfileId in $Profiles) {
    foreach ($name in 'journal.json', 'journal.json.report.json', 'journal.json.report.md') {
        $path = Join-Path $Outputs "$ProfileId/$name"
        if (Test-Path -LiteralPath $path) { $existing += "runs/container/phase-a/$ProfileId/$name" }
    }
}
if ($existing) {
    throw "Refusing to start: outputs already exist and are never overwritten:`n  $($existing -join "`n  ")`nMove them away deliberately (and reconcile any open journal) before a new run."
}

$gateway = (Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
    Where-Object InterfaceAlias -like 'vEthernet (WSL*' | Select-Object -First 1).IPAddress
if (-not $gateway) { throw 'No "vEthernet (WSL...)" adapter found. Start WSL (wsl -d Debian) first.' }
$expected = "http://${gateway}:11435/v1/chat/completions"
foreach ($ProfileId in $Profiles) {
    $manifest = Get-Content -LiteralPath (Join-Path $InputDir "$ProfileId/manifest.json") -Raw | ConvertFrom-Json
    if ($manifest.config.endpoint -ne $expected) {
        throw "runs/phase-a/$ProfileId/manifest.json endpoint $($manifest.config.endpoint) differs from the current gateway $expected. Rebuild the inputs with --local-endpoint $expected."
    }
}

$stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')
$summaryPath = Join-Path $Outputs "summary-$stamp.json"
New-Item -ItemType Directory -Force -Path $Outputs | Out-Null
$summary = [ordered]@{ started_at = [DateTime]::UtcNow.ToString('o'); endpoint = $expected; profiles = @(); stop = $null }
function Save-Summary { [IO.File]::WriteAllText($summaryPath, ($summary | ConvertTo-Json -Depth 6), [Text.UTF8Encoding]::new($false)) }
Save-Summary

$stopExit = $null
try {
    & pwsh -NoProfile -File $Serve start
    if ($LASTEXITCODE -ne 0) { throw 'serve_local_ollama.ps1 start failed.' }
    foreach ($ProfileId in $Profiles) {
        $dir = Join-Path $Outputs $ProfileId
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
        $model = (Get-Content -LiteralPath (Join-Path $InputDir "$ProfileId/manifest.json") -Raw | ConvertFrom-Json).config.model
        $record = [ordered]@{ profile = $ProfileId; model = $model; started_at = [DateTime]::UtcNow.ToString('o');
                              exit_code = $null; status = $null; unloaded = $null; error = $null }
        try {
            & pwsh -NoProfile -File $Snapshot -Path (Join-Path $dir 'host-memory.json') | Out-Null
            if ($LASTEXITCODE -ne 0) { throw 'host memory snapshot failed' }
            $result = Invoke-Container $ProfileId @('--execute', '--journal', "/outputs/phase-a/$ProfileId/journal.json",
                '--observer', 'ollama', '--host-snapshot', "/outputs/phase-a/$ProfileId/host-memory.json")
            $record.exit_code = $result.exit_code
            [IO.File]::WriteAllText((Join-Path $dir "console-$stamp.txt"), $result.output, [Text.UTF8Encoding]::new($false))
            $report = Join-Path $dir 'journal.json.report.json'
            $record.status = if (Test-Path -LiteralPath $report) { (Get-Content -LiteralPath $report -Raw | ConvertFrom-Json).status } else { 'no_report' }
        } catch {
            $record.error = $_.Exception.Message
        } finally {
            try { $record.unloaded = Invoke-Unload $model } catch { $record.unloaded = $false; $record.error = ($record.error ?? 'unload_failed') }
            $record.finished_at = [DateTime]::UtcNow.ToString('o')
            $summary.profiles += $record
            Save-Summary
            Write-Output ("{0}: exit {1}, status {2}, unloaded {3}{4}" -f $ProfileId, $record.exit_code, $record.status,
                $record.unloaded, $(if ($record.error) { ", error: $($record.error)" } else { '' }))
        }
    }
} finally {
    & pwsh -NoProfile -File $Serve stop
    $stopExit = $LASTEXITCODE
    $leftover = @(Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Where-Object LocalPort -in 11434, 11435)
    $summary.stop = @{ exit_code = $stopExit; listeners_left = $leftover.Count; finished_at = [DateTime]::UtcNow.ToString('o') }
    Save-Summary
    if ($stopExit -ne 0 -or $leftover) { Write-Output 'WARNING: Ollama or the gateway may still be running; check serve_local_ollama.ps1 status.' }
}
Write-Output "Summary: $summaryPath"
$bad = @($summary.profiles | Where-Object { $_.status -ne 'completed' -or -not $_.unloaded })
if ($bad -or $stopExit -ne 0) { exit 1 }
