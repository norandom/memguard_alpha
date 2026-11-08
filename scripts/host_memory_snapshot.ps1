<#
.SYNOPSIS
Write a timestamped free-RAM snapshot of THIS Windows machine (the Ollama model host).

.DESCRIPTION
Host-side tooling, not Recall Guard execution: run it natively on the Windows
host that serves Ollama, right before a containerized preflight. It reads only
local memory figures (no admin rights, no network) and writes JSON:
{available_bytes, total_bytes, observed_at (UTC ISO-8601), host, source}.
The default path runs/container/host-memory.json is what the container sees as
/outputs/host-memory.json; pass that to --host-snapshot. Snapshots older than
10 minutes are refused, so take a fresh one before each run.

.EXAMPLE
pwsh -File scripts/host_memory_snapshot.ps1
#>
param([string]$Path = (Join-Path $PSScriptRoot '..\runs\container\host-memory.json'))

$ErrorActionPreference = 'Stop'
$os = Get-CimInstance -ClassName Win32_OperatingSystem
$snapshot = [ordered]@{
    available_bytes = [int64]$os.FreePhysicalMemory * 1024      # KiB -> bytes
    total_bytes     = [int64]$os.TotalVisibleMemorySize * 1024  # KiB -> bytes
    observed_at     = [DateTime]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ')
    host            = $env:COMPUTERNAME
    source          = 'windows_host_snapshot'
}
$full = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Path)
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $full) | Out-Null
[IO.File]::WriteAllText($full, ($snapshot | ConvertTo-Json), [Text.UTF8Encoding]::new($false))
Write-Output "Wrote $full (container: /outputs/host-memory.json when under runs/container)"
