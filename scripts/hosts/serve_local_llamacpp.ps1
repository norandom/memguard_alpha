<#
.SYNOPSIS
Serve llama-server b12199 on the pinned Gemma 3 4B GGUF (top-k logprobs) behind an authenticated gateway, with idle stop and verified teardown.

.DESCRIPTION
Host-side tooling (PowerShell 7, no admin), the sibling of scripts/hosts/serve_full_distribution.ps1
(same state, token, gateway and teardown structure; its shared helpers are copied here verbatim
and a test keeps them identical; that script and scripts/hosts/serve_local_flm.ps1 are unchanged).
`start` launches ONE llama-server natively on Windows (the model-host exception) with the
reference server's command line of scripts/hosts/llamacpp_native.py:

    llama-server.exe -m <pinned GGUF> --host 127.0.0.1 --port 11570 -ngl 99 -dev Vulkan0 -fa off
                     -c 2048 -np 1 --swa-full --no-webui

All layers go to the Vulkan iGPU with a 2048-token context (the profile's context_length). The
b12199 build has no server switch for logprobs: each chat request asks for them with
`logprobs: true, top_logprobs: N` (the llamacpp-gemma3-4b profile sends 5). Before loading, the
script checks that llama-server reports the profile's runtime_version (b12199) and that the
model file is the blob named by the profile's artifact_digest with that SHA-256. llama-server
listens at once and answers GET /health with 503 while it loads; readiness is /health returning
{"status":"ok"} from a listener in the launched process tree, whose process id is recorded.

`start` then launches a hidden pwsh gateway on 127.0.0.1:11571 (the host's loopback port plus
one, the repository convention). The gateway binds 127.0.0.1 by default, which serves the native
study runner. When a container (Podman: host.containers.internal) or another machine must reach
the host, pass the address of that adapter with -BindAddress. It never binds all interfaces. It
requires `Authorization: Bearer <LOCAL_MODEL_API_KEY>` (constant-time compare; 401 otherwise),
strips that header and forwards ONLY `POST /v1/chat/completions`, `GET /v1/models`,
`GET /health` and `GET /props` (whose model_path names the GGUF blob for the runner's digest
check); every other method or path gets 404 and never reaches llama-server.

One model host at a time (user correction 2026-11-10): while the sibling full-distribution host
(task 6.1, a python process listening on the profile's full_distribution port) is resident,
`start` refuses before loading anything, names the sibling's stop command, and exits 2.

RAM gate before loading: free physical memory (Get-CimInstance Win32_OperatingSystem) must
reach the llamacpp-gemma3-4b profile's min_free_ram_bytes, which is one load of the GGUF plus
headroom and is never doubled. The gate waits up to -RamWaitSeconds (default 600) for three
readings in a row at or above the requirement and prints a progress line while it waits.
Otherwise it loads nothing, prints the Modal alternative and the "another machine" option in
preflight_host's wording, and exits 2.

The token is LOCAL_MODEL_API_KEY from the root .env (shared with the other local gateways). If
it is missing, a random 32-byte token is generated and appended there; its value is never
printed. After -IdleMinutes (default 30) without a gateway request, or when llama-server exits,
the gateway stops itself and the host and logs the same leftover check `stop` runs. `stop`
stops the process trees and verifies that no host or gateway process remains and that nothing
listens on either port; otherwise it names what remains and exits 1. State and logs:
runs/finance-sim/hosts/llama-server/ (private).

.EXAMPLE
pwsh -File scripts/hosts/serve_local_llamacpp.ps1 start
pwsh -File scripts/hosts/serve_local_llamacpp.ps1 status
pwsh -File scripts/hosts/serve_local_llamacpp.ps1 stop
pwsh -File scripts/hosts/serve_local_llamacpp.ps1 start -BindAddress 192.168.1.20   # adapter address for a container or another machine
#>
param(
    [Parameter(Position = 0, Mandatory)]
    [ValidateSet('start', 'stop', 'status', 'gateway')]
    [string]$Action,
    [string]$BindAddress = '127.0.0.1',     # Gateway address; never all interfaces.
    [ValidateRange(1024, 65535)][int]$Port = 11571,       # Gateway: the host's loopback port plus one.
    [ValidateRange(1024, 65535)][int]$HostPort = 11570,   # = the profile's model_config.endpoint port.
    [ValidateRange(1, 1440)][int]$IdleMinutes = 30,
    [ValidateRange(10, 3600)][int]$StartTimeoutSeconds = 600,
    [ValidateRange(0, 3600)][int]$RamWaitSeconds = 600,
    [string]$StateDir = 'runs/finance-sim/hosts/llama-server',   # Relative paths resolve against the repository root.
    [string]$EnvPath = '.env',
    [string]$ProfilePath = 'recall_guard/core/provider_profiles/llamacpp-gemma3-4b.json',
    [string]$ModelPath = 'models/blobs/sha256-f06e91e18fc74295d0c781bf91622508c75655e2c18ed869d52447ffe1e667e1',   # = llamacpp_native.MODEL_PATH
    [string]$LlamaDir = ''   # Empty: the winget llama.cpp package (= llamacpp_native.LLAMA_DIR).
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)

function Resolve-RootPath([string]$Path) {
    $full = if ([IO.Path]::IsPathRooted($Path)) { [IO.Path]::GetFullPath($Path) } else { [IO.Path]::GetFullPath((Join-Path $Root $Path)) }
    return $full.TrimEnd([char]'\', [char]'/')   # A trailing backslash would escape the closing quote of a launch argument.
}

$StateFolder = Resolve-RootPath $StateDir
$StatePath = Join-Path $StateFolder 'state.json'
$EnvFile = Resolve-RootPath $EnvPath
$ProfileFile = Resolve-RootPath $ProfilePath
$ModelFile = Resolve-RootPath $ModelPath
$MaxBodyBytes = 1MB              # A 2048-token prompt is far smaller.
$MaxResponseBytes = 64MB
$TokenName = 'LOCAL_MODEL_API_KEY'
# The study runner's routes: generation, and what it reads before admitting requests.
$Routes = @('POST /v1/chat/completions', 'GET /v1/models', 'GET /health', 'GET /props')
# Command lines of a host (llama-server on one of this script's ports) and of a gateway (pwsh) on this machine.
$HostCommand = '(?<![\w.-])llama-server(?:\.exe)?"?\s(?:.*\s)?--port[=\s]"?(\d+)\b'
$GatewayCommand = 'serve_local_llamacpp\.ps1"?\s+gateway\b'

# ---------- shared helpers (verbatim from serve_full_distribution.ps1; a test keeps them identical) ----------

function Read-State {
    if (Test-Path -LiteralPath $StatePath) { return Get-Content -LiteralPath $StatePath -Raw | ConvertFrom-Json -AsHashtable -DateKind String }
    return $null
}

function Write-State([hashtable]$State) {
    New-Item -ItemType Directory -Force -Path $StateFolder | Out-Null
    $temporary = "$StatePath.tmp"
    [IO.File]::WriteAllText($temporary, ($State | ConvertTo-Json -Depth 4), [Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath $temporary -Destination $StatePath -Force
}

function Read-Token {
    if (-not (Test-Path -LiteralPath $EnvFile)) { return $null }
    foreach ($line in [IO.File]::ReadAllLines($EnvFile)) {
        if ($line -match "^\s*$TokenName\s*=\s*(.*?)\s*$") {
            $value = $Matches[1].Trim('"', "'")
            if ($value) { return $value }
        }
    }
    return $null
}

function Initialize-Token {
    if (Read-Token) { return 'existing' }
    $bytes = [Security.Cryptography.RandomNumberGenerator]::GetBytes(32)
    $token = [Convert]::ToBase64String($bytes).TrimEnd('=').Replace('+', '-').Replace('/', '_')
    $text = if (Test-Path -LiteralPath $EnvFile) { [IO.File]::ReadAllText($EnvFile) } else { '' }
    $newline = if ($text -match "`r`n" -or -not $text) { "`r`n" } else { "`n" }
    $line = "$TokenName=$token"
    if ($text -match "(?m)^\s*$TokenName\s*=\s*$") {
        $text = [regex]::Replace($text, "(?m)^\s*$TokenName\s*=[ \t]*(?=\r?$)", $line)  # Fill an empty entry.
    } else {
        if ($text -and -not $text.EndsWith("`n")) { $text += $newline }
        $text += "# Bearer token for the local model gateways (scripts/hosts/serve_local_ollama.ps1, serve_local_flm.ps1, serve_full_distribution.ps1).$newline$line$newline"
    }
    [IO.File]::WriteAllText($EnvFile, $text, [Text.UTF8Encoding]::new($false))
    return 'generated'
}

function Get-ProcessTree([int]$ProcessId) {
    $ids = [Collections.Generic.List[int]]::new()
    $queue = [Collections.Generic.Queue[int]]::new()
    $queue.Enqueue($ProcessId)
    while ($queue.Count) {
        $id = $queue.Dequeue()
        $ids.Add($id)
        Get-CimInstance Win32_Process -Filter "ParentProcessId=$id" -ErrorAction SilentlyContinue |
            ForEach-Object { $queue.Enqueue([int]$_.ProcessId) }
    }
    return $ids
}

function Test-Tracked([int]$ProcessId, [string]$ExpectedName) {
    # The name check avoids touching an unrelated process that reused a stale PID.
    if (-not $ProcessId) { return $false }
    $process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    return [bool]($process -and $process.ProcessName -like $ExpectedName)
}

function Stop-Tree([int]$ProcessId, [string]$ExpectedName) {
    # Returns the ids it stopped, children first (python children of uv, conhost).
    if (-not (Test-Tracked $ProcessId $ExpectedName)) { return }
    $ids = [int[]]@(Get-ProcessTree $ProcessId)
    [array]::Reverse($ids)
    foreach ($id in $ids) { Stop-Process -Id $id -Force -ErrorAction SilentlyContinue }
    return $ids
}

function Get-Listeners([int[]]$Ports) {
    Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
        Where-Object { $_.LocalPort -in $Ports } |
        ForEach-Object { [pscustomobject]@{ Address = $_.LocalAddress; Port = [int]$_.LocalPort; Pid = [int]$_.OwningProcess } }
}

function Get-FreeRamBytes {
    return [long](Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory * 1KB
}

function Format-Gib([long]$Bytes) {
    return [string]::Format([Globalization.CultureInfo]::InvariantCulture, '{0:0.0} GiB', $Bytes / 1GB)
}

function Wait-FreeRam([long]$Required, [int]$Seconds) {
    # Three readings in a row at or above the requirement, as in the 6.1 exactness check: other
    # workloads move free RAM by gigabytes within seconds, so one reading is not evidence that a
    # load fits. Refused once the deadline has passed while the latest reading is below.
    $started = [DateTime]::UtcNow
    $deadline = $started.AddSeconds($Seconds)
    $readings = [Collections.Generic.List[long]]::new()
    $streak = 0
    $noted = [DateTime]::MinValue
    while ($true) {
        $free = Get-FreeRamBytes
        $readings.Add($free)
        $streak = if ($free -ge $Required) { $streak + 1 } else { 0 }
        $now = [DateTime]::UtcNow
        $result = [ordered]@{ passed = $false; free_ram_bytes = $free; required_bytes = $Required
                              samples = $readings.Count; waited_seconds = [Math]::Round(($now - $started).TotalSeconds, 1) }
        if ($streak -ge 3) {
            $result.passed = $true
            $result.free_ram_bytes = [long]($readings.GetRange($readings.Count - 3, 3) | Measure-Object -Minimum).Minimum
            return $result
        }
        if ($streak -eq 0) {
            if ($now -ge $deadline) { return $result }
            if (($now - $noted).TotalSeconds -ge 10) {
                Write-Host ("Waiting for free RAM: {0} free, {1} required, {2:0} s of {3} s" -f (Format-Gib $free), (Format-Gib $Required), ($now - $started).TotalSeconds, $Seconds)
                $noted = $now
            }
        }
        Start-Sleep -Seconds 2
    }
}

function Send-Response([IO.Stream]$Stream, [int]$Status, [byte[]]$Body, [string]$ContentType = 'application/json') {
    $reason = [Net.HttpStatusCode]$Status
    $head = "HTTP/1.1 $Status $reason`r`nContent-Type: $ContentType`r`nContent-Length: $($Body.Length)`r`nConnection: close`r`n`r`n"
    $bytes = [Text.Encoding]::ASCII.GetBytes($head)
    $Stream.Write($bytes, 0, $bytes.Length)
    if ($Body.Length) { $Stream.Write($Body, 0, $Body.Length) }
    $Stream.Flush()
}

function Send-Error([IO.Stream]$Stream, [int]$Status, [string]$Message) {
    Send-Response $Stream $Status ([Text.Encoding]::UTF8.GetBytes((@{ error = $Message } | ConvertTo-Json -Compress)))
}

function Read-Request([IO.Stream]$Stream) {
    # Returns @{Method; Path; Headers; Body} or an int HTTP error status.
    $buffer = [byte[]]::new(16384)
    $filled = 0
    $end = -1
    while ($end -lt 0) {
        if ($filled -eq $buffer.Length) { return 431 }
        $read = $Stream.Read($buffer, $filled, $buffer.Length - $filled)
        if ($read -le 0) { return 400 }
        $filled += $read
        $end = [Text.Encoding]::ASCII.GetString($buffer, 0, $filled).IndexOf("`r`n`r`n")
    }
    $lines = [Text.Encoding]::ASCII.GetString($buffer, 0, $end) -split "`r`n"
    $parts = $lines[0] -split ' '
    if ($parts.Count -ne 3 -or $parts[2] -notlike 'HTTP/1.*') { return 400 }
    $headers = @{}
    foreach ($line in ($lines | Select-Object -Skip 1)) {
        $colon = $line.IndexOf(':')
        if ($colon -gt 0) { $headers[$line.Substring(0, $colon).Trim().ToLowerInvariant()] = $line.Substring($colon + 1).Trim() }
    }
    if ($headers.ContainsKey('transfer-encoding')) { return 411 }
    $length = 0
    if ($headers.ContainsKey('content-length') -and -not [int]::TryParse($headers['content-length'], [ref]$length)) { return 400 }
    if ($length -lt 0) { return 400 }
    if ($length -gt $MaxBodyBytes) { return 413 }
    $body = [byte[]]::new($length)
    $already = [Math]::Min($filled - ($end + 4), $length)
    [Array]::Copy($buffer, $end + 4, $body, 0, $already)
    while ($already -lt $length) {
        $read = $Stream.Read($body, $already, $length - $already)
        if ($read -le 0) { return 400 }
        $already += $read
    }
    return @{ Method = $parts[0]; Path = $parts[1]; Headers = $headers; Body = $body }
}

function Test-Token([string]$Header, [byte[]]$Expected) {
    if (-not $Header -or -not $Header.StartsWith('Bearer ')) { return $false }
    $given = [Text.Encoding]::UTF8.GetBytes($Header.Substring(7))
    return [Security.Cryptography.CryptographicOperations]::FixedTimeEquals($given, $Expected)
}

function Disable-OutputInheritance {
    # Start-Process with redirection creates the child with handle inheritance on, so a caller's
    # stdout and stderr pipes would leak into the long-lived host and gateway, and a caller that
    # captures this script's output would wait until both exit. Clear HANDLE_FLAG_INHERIT on them.
    if (-not ('RecallGuard.StdHandles' -as [type])) {
        Add-Type -Namespace RecallGuard -Name StdHandles -MemberDefinition @'
[DllImport("kernel32.dll", SetLastError = true)] public static extern IntPtr GetStdHandle(int nStdHandle);
[DllImport("kernel32.dll", SetLastError = true)] public static extern bool SetHandleInformation(IntPtr hObject, uint dwMask, uint dwFlags);
'@
    }
    foreach ($id in -11, -12) {   # STD_OUTPUT_HANDLE, STD_ERROR_HANDLE
        $handle = [RecallGuard.StdHandles]::GetStdHandle($id)
        if ($handle -ne [IntPtr]::Zero -and $handle -ne [IntPtr]::new(-1)) { [void][RecallGuard.StdHandles]::SetHandleInformation($handle, 1, 0) }
    }
}

# ---------- llama-server processes, stop and the leftover check ----------

function Get-ServingProcesses([int[]]$HostPorts) {
    # Any llama-server on one of the given ports, and any gateway of this script, found by command line.
    $found = Get-CimInstance Win32_Process -Filter "CommandLine LIKE '%llama-server%' OR CommandLine LIKE '%serve[_]local[_]llamacpp.ps1%'" -ErrorAction SilentlyContinue
    foreach ($item in @($found)) {
        if (-not $item -or [int]$item.ProcessId -eq $PID) { continue }
        $kind = $null
        if ($item.CommandLine -match $GatewayCommand) { $kind = 'gateway' }
        elseif ($item.CommandLine -match $HostCommand -and [int]$Matches[1] -in $HostPorts) { $kind = 'host' }
        if ($kind) { [pscustomobject]@{ Kind = $kind; Pid = [int]$item.ProcessId; Name = $item.Name } }
    }
}

function Stop-All([hashtable]$State, [string]$Reason) {
    # Stops the recorded trees and returns the ids it stopped; the first stop's time and reason are kept.
    # The listener recorded at readiness is stopped too, in case it is not (or no longer) in the host tree.
    $stopped = @(Stop-Tree ([int]$State.host_pid) ([string]($State.host_name ?? 'llama-server')))
    if ($State.listener_pid -and $State.listener_pid -ne $State.host_pid) {
        $stopped += @(Stop-Tree ([int]$State.listener_pid) ([string]($State.listener_name ?? 'llama-server')))
    }
    if ($State.gateway_pid -and $State.gateway_pid -ne $PID) { $stopped += @(Stop-Tree ([int]$State.gateway_pid) 'pwsh') }
    if (-not $State.stopped_at) {
        $State.stopped_at = [DateTime]::UtcNow.ToString('o')
        $State.stop_reason = $Reason
    }
    Write-State $State
    return $stopped
}

function Wait-Exited([int[]]$Ids) {
    # Bounded wait until every stopped process has exited.
    $deadline = [DateTime]::UtcNow.AddSeconds(15)
    while ($Ids | Where-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue }) {
        if ([DateTime]::UtcNow -gt $deadline) { break }
        Start-Sleep -Milliseconds 250
    }
}

function Get-Remaining([hashtable]$State, [int[]]$Ports) {
    # Lines naming every listener and process that remains; empty means fully stopped.
    foreach ($item in @(Get-Listeners $Ports)) { "STILL LISTENING $($item.Address):$($item.Port) pid $($item.Pid)" }
    if ($State) {
        if (Test-Tracked ([int]$State.host_pid) ([string]($State.host_name ?? 'llama-server'))) { "STILL RUNNING recorded host pid $($State.host_pid)" }
        if ($State.listener_pid -and $State.listener_pid -ne $State.host_pid -and
            (Test-Tracked ([int]$State.listener_pid) ([string]($State.listener_name ?? 'llama-server')))) { "STILL RUNNING recorded listener pid $($State.listener_pid)" }
        if ($State.gateway_pid -ne $PID -and (Test-Tracked ([int]$State.gateway_pid) 'pwsh')) { "STILL RUNNING recorded gateway pid $($State.gateway_pid)" }
    }
    foreach ($item in @(Get-ServingProcesses $Ports)) { "STILL RUNNING $($item.Kind) pid $($item.Pid) ($($item.Name))" }
}

function Write-Leftover([string]$Log, [hashtable]$State, [int[]]$Ports, [int[]]$Stopped) {
    # The gateway's own stop (idle or host exit) runs the same check as `stop` and logs its result.
    Wait-Exited $Stopped
    $remaining = @(Get-Remaining $State $Ports)
    $stamp = [DateTime]::UtcNow.ToString('o')
    if (-not $remaining) { $remaining = @("no host or gateway process remains; nothing listens on $($Ports -join ' or ')") }
    Add-Content -LiteralPath $Log -Value ($remaining | ForEach-Object { "$stamp leftover check: $_" })
}

# ---------- one model host at a time; RAM gate for one load ----------

function Read-HostProfile {
    if (-not (Test-Path -LiteralPath $ProfileFile -PathType Leaf)) { throw "Profile $ProfileFile not found." }
    $data = Get-Content -LiteralPath $ProfileFile -Raw | ConvertFrom-Json -AsHashtable
    $minimum = [long]($data.host_requirements.min_free_ram_bytes ?? 0)
    if ($minimum -le 0) { throw "Profile $ProfileFile declares no host_requirements.min_free_ram_bytes." }
    $declared = $data.model_config.local_identity.declared
    $digest = [string]$declared.artifact_digest
    if ($digest -cnotmatch '^sha256:[0-9a-f]{64}$') { throw "Profile $ProfileFile declares no sha256 artifact_digest." }
    if (-not $data.full_distribution.endpoint) { throw "Profile $ProfileFile declares no full_distribution.endpoint." }
    return @{ model = [string]$data.model_config.model; min_free_ram_bytes = $minimum
              fallbacks = @($data.host_requirements.fallback_profile_ids | Where-Object { $_ })
              digest = $digest; runtime_version = [string]$declared.runtime_version
              sibling_port = ([Uri]$data.full_distribution.endpoint).Port }
}

function Test-SiblingResident([int]$SiblingPort) {
    # The full-distribution host (python, task 6.1) holds its load of the same GGUF once it listens.
    foreach ($item in @(Get-Listeners @($SiblingPort))) {
        $process = Get-Process -Id $item.Pid -ErrorAction SilentlyContinue
        if ($process -and $process.ProcessName -like 'python*') { return $true }
    }
    return $false
}

function Write-SiblingRefusal([int]$SiblingPort) {
    Write-Output ("Refusing to start: the full-distribution host (task 6.1) is resident on port $SiblingPort, and only one " +
                  "model host runs at a time. Stop it first (pwsh -File scripts/hosts/serve_full_distribution.ps1 stop). Nothing " +
                  "was loaded and no request was sent.")
}

function Get-RamRequirement([long]$PerLoad) {
    # One load of the GGUF plus headroom (the profile's min_free_ram_bytes), never doubled: only
    # one model host runs at a time, so no second load can coexist with this one.
    return [ordered]@{ per_load_bytes = $PerLoad; loads = 1; required_bytes = $PerLoad }
}

function Write-RamRefusal([hashtable]$HostProfile, [System.Collections.IDictionary]$Need, [System.Collections.IDictionary]$Gate) {
    Write-Output ("RAM gate: one load of the GGUF at $(Format-Gib $Need.per_load_bytes) (the profile's min_free_ram_bytes); " +
                  "free RAM stayed below it for $($Gate.waited_seconds) s.")
    # Same sentence as recall_guard.core.profiles.preflight_host (_ram_recommendation).
    $fallbacks = ($HostProfile.fallbacks | ForEach-Object { "--profile $_" }) -join ' or '
    $options = if ($fallbacks) { "Try $fallbacks, or run" } else { 'Run' }
    $required = Format-Gib $Need.required_bytes
    Write-Output ("$(Format-Gib $Gate.free_ram_bytes) free, $required required for $($HostProfile.model). Nothing was loaded and no " +
                  "request was sent. $options on a machine with at least $required of free RAM.")
}

# ---------- llama-server: command line, build, model file, readiness ----------

function Get-ServerCommand([string]$Directory, [string]$Model, [int]$ServerPort) {
    # = scripts/hosts/llamacpp_native.py: llama-server.exe in LLAMA_DIR, with the reference server's command line
    # (-m MODEL_PATH --host 127.0.0.1 --port PORT, then SERVER_FLAGS). b12199 has no server switch for
    # logprobs: each request asks for them (logprobs, top_logprobs).
    if (-not $Directory) { $Directory = Join-Path $env:LOCALAPPDATA 'Microsoft/WinGet/Packages/ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe' }
    $flags = @('-ngl', '99', '-dev', 'Vulkan0', '-fa', 'off', '-c', '2048', '-np', '1', '--swa-full', '--no-webui')
    return @{ exe = Join-Path $Directory 'llama-server.exe'
              arguments = @('-m', $Model, '--host', '127.0.0.1', '--port', "$ServerPort") + $flags }
}

function Get-ServerBuild([string]$Text) {
    # llama-server --version prints "version: 0.9.0-dev (build 12199, commit 372dc5ca1)" (older: "version: 12199 (...)").
    if ($Text -match '\(build (\d+)\b') { return [int]$Matches[1] }
    if ($Text -match '(?m)^version: (\d+) \(') { return [int]$Matches[1] }
    return $null
}

function Read-ServerVersion([string]$Exe) {
    $ErrorActionPreference = 'Continue'   # llama-server prints its version on stderr.
    return (& $Exe --version 2>&1 | ForEach-Object { "$_" }) -join "`n"
}

function Test-ModelFile([string]$Path, [string]$Digest, [switch]$Hash) {
    # $null when the file is the pinned blob (and, with -Hash, has its SHA-256); otherwise the reason.
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return "Model file $Path not found." }
    $blob = $Digest.Replace(':', '-')
    $name = Split-Path -Leaf $Path
    if ($name -cne $blob) { return "Model file name $name is not the pinned blob $blob." }
    if ($Hash) {
        $actual = 'sha256:' + (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actual -cne $Digest) { return "Model file $Path has $actual, not the pinned $Digest." }
    }
    return $null
}

function Test-HostHealth([int]$ServerPort) {
    # llama-server answers /health with 503 while it loads the model and 200 {"status":"ok"} once ready.
    $handler = [Net.Http.SocketsHttpHandler]::new()
    $handler.UseProxy = $false
    $handler.AllowAutoRedirect = $false
    $client = [Net.Http.HttpClient]::new($handler)
    $client.Timeout = [TimeSpan]::FromSeconds(5)
    try {
        $response = $client.GetAsync("http://127.0.0.1:$ServerPort/health").GetAwaiter().GetResult()
        if ([int]$response.StatusCode -ne 200) { return $false }
        $body = $response.Content.ReadAsStringAsync().GetAwaiter().GetResult() | ConvertFrom-Json -AsHashtable
        return ($body -is [Collections.IDictionary]) -and ($body.status -ceq 'ok')
    } catch {
        return $false
    } finally {
        $client.Dispose()
    }
}

function ConvertTo-ArgumentLine([string[]]$Arguments) {
    # Start-Process joins arguments with spaces, so an argument with a space is quoted here.
    return ($Arguments | ForEach-Object { if ($_ -match '\s') { "`"$_`"" } else { $_ } }) -join ' '
}

# ---------- gateway (runs in its own hidden pwsh process) ----------

function Invoke-Upstream([Net.Http.HttpClient]$Client, [string]$Upstream, [hashtable]$Request) {
    $message = [Net.Http.HttpRequestMessage]::new([Net.Http.HttpMethod]::new($Request.Method), "$Upstream$($Request.Path)")
    if ($Request.Method -ceq 'POST') {
        $message.Content = [Net.Http.ByteArrayContent]::new($Request.Body)
        $message.Content.Headers.TryAddWithoutValidation('Content-Type', ($Request.Headers['content-type'] ?? 'application/json')) | Out-Null
    }
    if ($Request.Headers['accept']) { $message.Headers.TryAddWithoutValidation('Accept', $Request.Headers['accept']) | Out-Null }
    # Authorization is never copied: llama-server sees an unauthenticated loopback request.
    $response = $Client.SendAsync($message).GetAwaiter().GetResult()
    $body = $response.Content.ReadAsByteArrayAsync().GetAwaiter().GetResult()
    $type = if ($response.Content.Headers.ContentType) { $response.Content.Headers.ContentType.ToString() } else { 'application/json' }
    return [int]$response.StatusCode, $body, $type
}

function Start-Gateway {
    $state = Read-State
    if (-not $state) { throw 'No host state; use start.' }
    $hostProcess = Get-Process -Id ([int]$state.host_pid) -ErrorAction SilentlyContinue
    if (-not (Test-Tracked ([int]$state.host_pid) ([string]($state.host_name ?? 'llama-server')))) { throw "The recorded host (pid $($state.host_pid)) is not running." }
    $null = $hostProcess.Handle   # Keep a handle: the watch below cannot mistake a reused PID, and the exit code stays readable.
    $token = Read-Token
    if (-not $token) { throw "$TokenName missing from $EnvFile." }
    $expected = [Text.Encoding]::UTF8.GetBytes($token)
    $token = $null
    $upstream = "http://127.0.0.1:$([int]$state.host_port)"
    $ports = @(@([int]$state.host_port, [int]$state.gateway_port) | Sort-Object -Unique)
    $log = Join-Path $StateFolder 'gateway.log'
    $handler = [Net.Http.SocketsHttpHandler]::new()
    $handler.UseProxy = $false
    $handler.AllowAutoRedirect = $false
    $client = [Net.Http.HttpClient]::new($handler)
    $client.Timeout = [TimeSpan]::FromSeconds(330)   # The profile's timeout_s (300) plus a margin.
    $client.MaxResponseContentBufferSize = $MaxResponseBytes
    $listener = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Parse($state.bind_address), [int]$state.gateway_port)
    $listener.ExclusiveAddressUse = $true   # SO_EXCLUSIVEADDRUSE: no second socket can share the gateway port.
    $listener.Start()
    $idle = [TimeSpan]::FromMinutes([double]$state.idle_minutes)
    $last = [DateTime]::UtcNow
    Add-Content -LiteralPath $log -Value "$([DateTime]::UtcNow.ToString('o')) listening $($state.bind_address):$($state.gateway_port) upstream=$upstream idle=$($state.idle_minutes)m"
    try {
        while ($true) {
            $accept = $listener.AcceptTcpClientAsync()
            while (-not $accept.Wait(5000)) {
                $reason = $null
                if ([DateTime]::UtcNow - $last -gt $idle) {
                    $reason = 'idle_timeout'
                    Add-Content -LiteralPath $log -Value "$([DateTime]::UtcNow.ToString('o')) idle timeout, stopping"
                } elseif ($hostProcess.HasExited) {
                    $reason = 'host_exited'
                    $code = try { $hostProcess.ExitCode } catch { 'unknown' }
                    Add-Content -LiteralPath $log -Value "$([DateTime]::UtcNow.ToString('o')) host exited (exit code $code), stopping"
                }
                if ($reason) {
                    $listener.Stop()
                    $stopped = @(Stop-All (Read-State) $reason)
                    Write-Leftover $log (Read-State) $ports $stopped
                    return
                }
            }
            $connection = $accept.Result
            $status = 0
            $route = '-'
            try {
                $connection.ReceiveTimeout = 30000
                $connection.SendTimeout = 30000
                $stream = $connection.GetStream()
                $request = Read-Request $stream
                if ($request -is [int]) {
                    $status = $request
                    Send-Error $stream $status 'bad request'
                } else {
                    $route = "$($request.Method) $($request.Path)"
                    if (-not (Test-Token $request.Headers['authorization'] $expected)) {
                        $status = 401
                        Send-Error $stream 401 'unauthorized'
                    } elseif (-not ($Routes -ccontains $route)) {
                        $status = 404
                        Send-Error $stream 404 'route not served by the gateway'
                    } else {
                        try {
                            $status, $body, $type = Invoke-Upstream $client $upstream $request
                            Send-Response $stream $status $body $type
                        } catch {
                            $status = 502
                            Send-Error $stream 502 'upstream unavailable'
                        }
                    }
                }
            } catch {
                if (-not $status) { $status = -1 }  # Client went away or timed out.
            } finally {
                # Half-close, then discard unread input (for example a refused oversized
                # body) so closing does not reset the connection before the client reads.
                try {
                    $connection.Client.Shutdown([Net.Sockets.SocketShutdown]::Send)
                    $connection.ReceiveTimeout = 2000
                    $sink = [byte[]]::new(65536)
                    $drained = 0
                    while ($drained -lt 8MB -and ($n = $connection.Client.Receive($sink)) -gt 0) { $drained += $n }
                } catch { }
                $connection.Dispose()
            }
            $last = [DateTime]::UtcNow
            Add-Content -LiteralPath $log -Value "$($last.ToString('o')) $route $status"
        }
    } finally {
        $listener.Stop()
    }
}

# ---------- actions ----------

function Resolve-BindAddress([string]$Address) {
    $parsed = $null
    if (-not [Net.IPAddress]::TryParse($Address, [ref]$parsed) -or
        $parsed.Equals([Net.IPAddress]::Any) -or $parsed.Equals([Net.IPAddress]::IPv6Any) -or
        ($parsed.IsIPv4MappedToIPv6 -and $parsed.MapToIPv4().Equals([Net.IPAddress]::Any))) {
        throw "Refusing bind address $Address. Give 127.0.0.1 or the IP address of one adapter, never all interfaces."
    }
    return $parsed
}

function Start-Serving {
    $state = Read-State
    if ($state -and -not $state.stopped_at) { throw 'A host state is active. Run stop first.' }
    $address = Resolve-BindAddress $BindAddress
    $bind = $address.ToString()
    if (Get-Listeners @($HostPort, $Port)) { throw "Port $HostPort or $Port already listens. Stop the other host or gateway first." }
    $running = @(Get-ServingProcesses @($HostPort, $Port))
    if ($running) { throw "A llama-server $($running[0].Kind) already runs (pid $($running.Pid -join ', ')). Run stop first." }
    $hostProfile = Read-HostProfile
    $server = Get-ServerCommand $LlamaDir $ModelFile $HostPort
    if (-not (Test-Path -LiteralPath $server.exe -PathType Leaf)) {
        throw "llama-server not found at $($server.exe). Install llama.cpp $($hostProfile.runtime_version) (winget ggml.llamacpp) or pass -LlamaDir."
    }
    $problem = Test-ModelFile $ModelFile $hostProfile.digest
    if ($problem) { throw $problem }
    if (Test-SiblingResident $hostProfile.sibling_port) {
        Write-SiblingRefusal $hostProfile.sibling_port
        exit 2
    }
    $need = Get-RamRequirement $hostProfile.min_free_ram_bytes
    $gate = Wait-FreeRam $need.required_bytes $RamWaitSeconds
    if (-not $gate.passed) {
        Write-RamRefusal $hostProfile $need $gate
        exit 2
    }
    $build = Get-ServerBuild (Read-ServerVersion $server.exe)
    if ("b$build" -cne $hostProfile.runtime_version) {
        throw "llama-server at $($server.exe) reports build '$build', not the profile's $($hostProfile.runtime_version)."
    }
    $problem = Test-ModelFile $ModelFile $hostProfile.digest -Hash
    if ($problem) { throw $problem }
    New-Item -ItemType Directory -Force -Path $StateFolder | Out-Null
    $tokenState = Initialize-Token
    Disable-OutputInheritance

    $hostProcess = Start-Process -FilePath $server.exe -WorkingDirectory $Root -WindowStyle Hidden -PassThru `
        -ArgumentList (ConvertTo-ArgumentLine $server.arguments) `
        -RedirectStandardOutput (Join-Path $StateFolder 'host.out.log') -RedirectStandardError (Join-Path $StateFolder 'host.log')
    $started = [DateTime]::UtcNow
    $origin = if ($address.AddressFamily -eq [Net.Sockets.AddressFamily]::InterNetworkV6) { "[$bind]" } else { $bind }
    $state = @{ host_pid = $hostProcess.Id; host_name = $hostProcess.ProcessName; listener_pid = $null; listener_name = $null
                gateway_pid = $null; bind_address = $bind; gateway_port = $Port; host_port = $HostPort
                host_address = "127.0.0.1:$HostPort"; idle_minutes = $IdleMinutes; started_at = $started.ToString('o')
                server_build = "b$build"; model_path = $ModelFile; model_digest = $hostProfile.digest; model_sha256_verified = $true
                server_arguments = $server.arguments
                ram_gate = [ordered]@{ per_load_bytes = $need.per_load_bytes; loads = $need.loads
                                       required_bytes = $need.required_bytes; sibling_resident = $false
                                       free_ram_bytes = $gate.free_ram_bytes; samples = $gate.samples
                                       waited_seconds = $gate.waited_seconds }
                ready_at = $null; readiness_seconds = $null; stopped_at = $null; stop_reason = $null
                endpoint = "http://${origin}:$Port/v1/chat/completions" }
    Write-State $state
    # llama-server listens at once and answers /health with 503 until the model is loaded.
    $deadline = $started.AddSeconds($StartTimeoutSeconds)
    $noted = $started
    $listener = $null
    do {
        Start-Sleep -Milliseconds 500
        $alive = -not $hostProcess.HasExited
        if ($alive -and (Test-HostHealth $HostPort)) {
            $tree = @(Get-ProcessTree $hostProcess.Id)
            $listening = @(Get-Listeners @($HostPort))
            $listener = $listening | Where-Object { $_.Pid -in $tree } | Select-Object -First 1
            if (-not $listener) {
                $null = Stop-All $state 'start_failed'
                throw "Port $HostPort answers /health, but not from the launched llama-server (listener pid $($listening.Pid -join ', '))."
            }
        }
        if (-not $listener -and ([DateTime]::UtcNow - $noted).TotalSeconds -ge 30) {
            $noted = [DateTime]::UtcNow
            Write-Output ("Waiting for llama-server /health on 127.0.0.1:{0} (model load): {1:0} s" -f $HostPort, ($noted - $started).TotalSeconds)
        }
    } until ($listener -or -not $alive -or [DateTime]::UtcNow -gt $deadline)
    if (-not $listener) {
        $null = Stop-All $state 'start_failed'
        $hostLog = Join-Path $StateFolder 'host.log'
        $reason = if (Test-Path -LiteralPath $hostLog) { (Get-Content -LiteralPath $hostLog -Tail 1) } else { '' }
        throw "llama-server did not become ready (alive: $alive). $reason See $hostLog."
    }
    $state.listener_pid = $listener.Pid
    $state.listener_name = (Get-Process -Id $listener.Pid -ErrorAction SilentlyContinue).ProcessName
    $state.ready_at = [DateTime]::UtcNow.ToString('o')
    $state.readiness_seconds = [Math]::Round(([DateTime]::UtcNow - $started).TotalSeconds, 1)
    Write-State $state
    # Start-Process joins arguments with spaces, so paths are quoted here.
    $gateway = Start-Process -FilePath (Get-Process -Id $PID).Path -WindowStyle Hidden -PassThru `
        -ArgumentList '-NoProfile', '-NonInteractive', '-File', "`"$PSCommandPath`"", 'gateway', '-Port', "$Port", '-HostPort', "$HostPort",
                      '-StateDir', "`"$StateFolder`"", '-EnvPath', "`"$EnvFile`"" `
        -RedirectStandardOutput (Join-Path $StateFolder 'gateway.out.log') -RedirectStandardError (Join-Path $StateFolder 'gateway.err.log')
    $state.gateway_pid = $gateway.Id
    Write-State $state
    $deadline = [DateTime]::UtcNow.AddSeconds(30)
    do {
        Start-Sleep -Milliseconds 500
        $ports = @(Get-Listeners @($HostPort, $Port) | ForEach-Object Port)
    } until (($ports -contains $HostPort -and $ports -contains $Port) -or [DateTime]::UtcNow -gt $deadline)
    if (-not ($ports -contains $HostPort -and $ports -contains $Port)) {
        $null = Stop-All $state 'start_failed'
        throw "Start failed (listening: $($ports -join ', ')). See $StateFolder\host.log and gateway.err.log."
    }
    Write-Output "Host llama-server b$build on 127.0.0.1:$HostPort (pid $($hostProcess.Id), listener pid $($listener.Pid)), /health ok after $($state.readiness_seconds) s"
    Write-Output "Model $(Split-Path -Leaf $ModelFile) (SHA-256 verified, = $($hostProfile.digest))"
    Write-Output "Gateway ${bind}:$Port (pid $($gateway.Id)), $TokenName $tokenState in $EnvFile (value not shown)"
    Write-Output "Gateway endpoint (LLAMACPP_ENDPOINT_URL): $($state.endpoint); routes $($Routes -join ', ')"
    Write-Output "RAM gate: $(Format-Gib $gate.free_ram_bytes) free, $(Format-Gib $need.required_bytes) required (one load; the full-distribution host not resident)"
    Write-Output "Auto-stop after $IdleMinutes min without gateway requests."
}

function Stop-Serving {
    $state = Read-State
    $ports = @($HostPort, $Port)
    $stopped = @()
    if ($state) {
        $ports += @($state.host_port, $state.gateway_port) | Where-Object { $_ } | ForEach-Object { [int]$_ }
        $stopped = @(Stop-All $state 'stop_command')
    }
    $ports = @($ports | Sort-Object -Unique)
    Wait-Exited $stopped
    $remaining = @(Get-Remaining $state $ports)
    foreach ($line in $remaining) { Write-Output $line }
    if ($remaining) { exit 1 }
    Write-Output "Stopped: no host or gateway process remains; nothing listens on $($ports -join ' or ')."
}

function Show-Status {
    $state = Read-State
    if (-not $state) { Write-Output 'No host state (never started).' }
    else {
        foreach ($pair in @(@('host_pid', ($state.host_name ?? 'llama-server')), @('listener_pid', ($state.listener_name ?? 'llama-server')), @('gateway_pid', 'pwsh'))) {
            $running = Test-Tracked ([int]$state[$pair[0]]) $pair[1]
            Write-Output ("{0} {1}: {2}" -f $pair[0], $state[$pair[0]], $(if ($running) { 'running' } else { 'not running' }))
        }
        Write-Output "host $($state.host_address) ($($state.server_build)), gateway $($state.bind_address):$($state.gateway_port), idle $($state.idle_minutes) min"
        Write-Output "started $($state.started_at); ready $($state.ready_at); stopped $($state.stopped_at) ($($state.stop_reason))"
        $log = Join-Path $StateFolder 'gateway.log'
        if (Test-Path -LiteralPath $log) { Write-Output "last gateway log: $(Get-Content -LiteralPath $log -Tail 1)" }
    }
    $ports = @($HostPort, $Port)
    if ($state) { $ports = @($ports + @($state.host_port, $state.gateway_port) | Where-Object { $_ } | ForEach-Object { [int]$_ } | Sort-Object -Unique) }
    $listeners = @(Get-Listeners $ports)
    if (-not $listeners) { Write-Output "No listener on $($ports -join ' or ')." }
    foreach ($item in $listeners) { Write-Output "listening $($item.Address):$($item.Port) pid $($item.Pid)" }
}

switch ($Action) {
    'start' { Start-Serving }
    'stop' { Stop-Serving }
    'status' { Show-Status }
    'gateway' { Start-Gateway }
}
