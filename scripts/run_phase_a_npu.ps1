<#
.SYNOPSIS
Run release-1.0 Phase A for the three FastFlowLM (AMD NPU) profiles, one FLM process per model.

.DESCRIPTION
Host-side orchestration (PowerShell 7, no admin); the NPU sibling of
scripts/run_phase_a_local.ps1. FastFlowLM runs natively on Windows (WSL and
Docker have no NPU access); only the dev container talks to it, through the
authenticated FLM gateway (scripts/serve_local_flm.ps1, port 11541). It performs
inference only without -DryRun. Inputs are the frozen files written by
scripts/build_phase_a_inputs.py --profiles npu under runs/phase-a/<profile>/; the
container sees them read-only at /phase-a. Journals and reports go to
runs/container/phase-a/<profile>/ (= /outputs/phase-a/<profile>/ in the container).

Execution:
  1. Refuse to start if any journal, report or summary output already exists, if
     an input is missing, if FLM or its gateway already listens, or if a manifest
     endpoint differs from the current FLM gateway address (the endpoint is hashed
     into the plan; rebuild the inputs).
  2. Per profile, sequentially: fresh Windows memory snapshot (before the model is
     loaded) -> serve_local_flm.ps1 start <tag> (one flm serve process) ->
     measure_local_models.py --execute --observer fastflowlm --host-snapshot in the
     dev container (LOCAL_MODEL_API_KEY passed by name) -> serve_local_flm.ps1 stop,
     which fails unless no FLM process or listener remains. Unloading is stopping
     the process. A failing profile is recorded and the next profile still runs.
  3. finally: serve_local_flm.ps1 stop once more and a listener check on 11540/11541.
  A summary is written to runs/container/phase-a/summary-npu-<UTC>.json.

-DryRun runs measure_local_models.py --dry-run per profile (no FLM, no gateway, no
inference) and saves each safe plan export under runs/container/phase-a/dry-run/.

.EXAMPLE
pwsh -File scripts/run_phase_a_npu.ps1 -DryRun
pwsh -File scripts/run_phase_a_npu.ps1
#>
param(
    [switch]$DryRun,
    [string[]]$Profiles = @('flm-llama3.2-3b', 'flm-gemma3-4b', 'flm-gemma4-e4b'),
    [string]$Distro = 'Debian',
    [string]$WslRepo = '/mnt/d/Source/memguard_alpha'
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$InputDir = Join-Path $Root 'runs/phase-a'
$Outputs = Join-Path $Root 'runs/container/phase-a'
$Serve = Join-Path $PSScriptRoot 'serve_local_flm.ps1'
$Snapshot = Join-Path $PSScriptRoot 'host_memory_snapshot.ps1'
$Ports = 11540, 11541

function Invoke-Container([string]$ProfileId, [string[]]$Arguments) {
    $in = "/phase-a/$ProfileId"
    $command = "cd '$WslRepo' && docker compose run --rm -e LOCAL_MODEL_API_KEY " +
        "-v '$WslRepo/runs/phase-a:/phase-a:ro' dev uv run --no-sync python scripts/measure_local_models.py " +
        "--plan $in/plan.json --manifest $in/manifest.json --corpus $in/corpus.json " + ($Arguments -join ' ')
    # Keep stdout (the script's JSON) apart from Compose's progress lines on stderr.
    $errors = New-TemporaryFile
    try {
        $output = & wsl.exe -d $Distro -- sh -c $command 2> $errors.FullName
        $exitCode = $LASTEXITCODE
        return @{ exit_code = $exitCode; output = ($output | Out-String); errors = (Get-Content -Raw $errors.FullName) }
    } finally {
        Remove-Item -LiteralPath $errors.FullName -Force -ErrorAction SilentlyContinue
    }
}

function Get-FlmListeners { @(Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue | Where-Object LocalPort -in $Ports) }

# ---------- refuse before any side effect ----------
foreach ($ProfileId in $Profiles) {
    if ($ProfileId -notlike 'flm-*') { throw "$ProfileId is not a FastFlowLM profile; use run_phase_a_local.ps1." }
    foreach ($name in 'plan.json', 'manifest.json', 'corpus.json') {
        if (-not (Test-Path -LiteralPath (Join-Path $InputDir "$ProfileId/$name") -PathType Leaf)) {
            throw "Missing input runs/phase-a/$ProfileId/$name. Run scripts/build_phase_a_inputs.py --profiles npu first."
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
if (Get-FlmListeners) { throw 'Port 11540 or 11541 already listens. Run serve_local_flm.ps1 stop first.' }

$gateway = (Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
    Where-Object InterfaceAlias -like 'vEthernet (WSL*' | Select-Object -First 1).IPAddress
if (-not $gateway) { throw 'No "vEthernet (WSL...)" adapter found. Start WSL (wsl -d Debian) first.' }
$expected = "http://${gateway}:11541/v1/chat/completions"
foreach ($ProfileId in $Profiles) {
    $manifest = Get-Content -LiteralPath (Join-Path $InputDir "$ProfileId/manifest.json") -Raw | ConvertFrom-Json
    if ($manifest.config.endpoint -ne $expected) {
        throw "runs/phase-a/$ProfileId/manifest.json endpoint $($manifest.config.endpoint) differs from the current FLM gateway $expected. Rebuild the inputs with --npu-endpoint $expected."
    }
}

$stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')
$summaryPath = Join-Path $Outputs "summary-npu-$stamp.json"
New-Item -ItemType Directory -Force -Path $Outputs | Out-Null
$summary = [ordered]@{ started_at = [DateTime]::UtcNow.ToString('o'); endpoint = $expected; profiles = @(); stop = $null }
function Save-Summary { [IO.File]::WriteAllText($summaryPath, ($summary | ConvertTo-Json -Depth 6), [Text.UTF8Encoding]::new($false)) }
Save-Summary

$stopExit = $null
try {
    foreach ($ProfileId in $Profiles) {
        $dir = Join-Path $Outputs $ProfileId
        New-Item -ItemType Directory -Force -Path $dir | Out-Null
        $model = (Get-Content -LiteralPath (Join-Path $InputDir "$ProfileId/manifest.json") -Raw | ConvertFrom-Json).config.model
        $record = [ordered]@{ profile = $ProfileId; model = $model; started_at = [DateTime]::UtcNow.ToString('o');
                              exit_code = $null; status = $null; stopped = $null; error = $null }
        try {
            # Free RAM before this model's FLM process loads it (the RAM gate's question).
            & pwsh -NoProfile -File $Snapshot -Path (Join-Path $dir 'host-memory.json') | Out-Null
            if ($LASTEXITCODE -ne 0) { throw 'host memory snapshot failed' }
            & pwsh -NoProfile -File $Serve start $model
            if ($LASTEXITCODE -ne 0) { throw "serve_local_flm.ps1 start $model failed" }
            $result = Invoke-Container $ProfileId @('--execute', '--journal', "/outputs/phase-a/$ProfileId/journal.json",
                '--observer', 'fastflowlm', '--host-snapshot', "/outputs/phase-a/$ProfileId/host-memory.json")
            $record.exit_code = $result.exit_code
            [IO.File]::WriteAllText((Join-Path $dir "console-$stamp.txt"), $result.output, [Text.UTF8Encoding]::new($false))
            [IO.File]::WriteAllText((Join-Path $dir "stderr-$stamp.txt"), [string]$result.errors, [Text.UTF8Encoding]::new($false))
            $report = Join-Path $dir 'journal.json.report.json'
            $record.status = if (Test-Path -LiteralPath $report) { (Get-Content -LiteralPath $report -Raw | ConvertFrom-Json).status } else { 'no_report' }
        } catch {
            $record.error = $_.Exception.Message
        } finally {
            # Unload = stop this model's FLM process; stop verifies nothing remains.
            & pwsh -NoProfile -File $Serve stop
            $record.stopped = ($LASTEXITCODE -eq 0) -and -not (Get-FlmListeners)
            $record.finished_at = [DateTime]::UtcNow.ToString('o')
            $summary.profiles += $record
            Save-Summary
            Write-Output ("{0}: exit {1}, status {2}, stopped {3}{4}" -f $ProfileId, $record.exit_code, $record.status,
                $record.stopped, $(if ($record.error) { ", error: $($record.error)" } else { '' }))
            if (-not $record.stopped) { throw "FLM for $ProfileId did not stop; not starting another model." }
        }
    }
} finally {
    & pwsh -NoProfile -File $Serve stop
    $stopExit = $LASTEXITCODE
    $leftover = Get-FlmListeners
    $summary.stop = @{ exit_code = $stopExit; listeners_left = $leftover.Count; finished_at = [DateTime]::UtcNow.ToString('o') }
    Save-Summary
    if ($stopExit -ne 0 -or $leftover) { Write-Output 'WARNING: FLM or the gateway may still be running; check serve_local_flm.ps1 status.' }
}
Write-Output "Summary: $summaryPath"
$bad = @($summary.profiles | Where-Object { $_.status -ne 'completed' -or -not $_.stopped })
if ($bad -or $stopExit -ne 0 -or $summary.profiles.Count -ne $Profiles.Count) { exit 1 }
