<#
.SYNOPSIS
Serve the teacher-forcing full-distribution host (task 6.1) behind an authenticated gateway, with idle stop and verified teardown.

.DESCRIPTION
Host-side tooling (PowerShell 7, no admin), the sibling of scripts/hosts/serve_local_flm.ps1
(same state, token, gateway and teardown structure; that script is unchanged). `start`
launches ONE host, `uv run --no-sync python scripts/hosts/serve_full_distribution.py --port 11580`,
natively on Windows (the model-host exception). The host hashes the pinned GGUF, loads one
llama.cpp instance, decodes a warm-up text and only then listens on 127.0.0.1:11580, so a
listening port means ready. `start` then launches a hidden pwsh gateway on 127.0.0.1:11581
(the host's loopback port plus one, the repository convention).

The gateway binds 127.0.0.1 by default, which serves the native study runner. When a
container (Podman: host.containers.internal) or another machine must reach the host, pass
the address of that adapter with -BindAddress. It never binds all interfaces. It requires
`Authorization: Bearer <LOCAL_MODEL_API_KEY>` (constant-time compare; 401 otherwise), strips
that header and forwards ONLY `POST /v1/full_distribution` and `POST /v1/full_distribution_batch`
(task 6.4) to the host; every other method or path gets 404 and never reaches the host.

Memory guard (task 6.4): the host is launched as `python -X faulthandler`, so a native death
leaves a Python stack in host.log, with --memory-floor-bytes set to the profile's
host_requirements.memory_floor_bytes (one quarter of min_free_ram_bytes when absent). The host
samples free physical memory every 5 s into host.out.log and refuses new requests with 503
memory_low below that floor; the gateway samples free physical memory every -MemorySampleSeconds
(default 5), also while it waits for a long host request, into gateway.log and logs each
memory_low refusal it passes through. Nothing restarts a host that died; the gateway then stops
and logs its exit code.

One model host at a time (user correction 2026-11-10): while the sibling llama-server (task
6.3, a llama-server process listening on the profile's generation port) is resident, `start`
refuses before loading anything, names the sibling's stop command, and exits 2.

RAM gate before loading: free physical memory (Get-CimInstance Win32_OperatingSystem) must
reach the llamacpp-gemma3-4b profile's min_free_ram_bytes, which is one load of the GGUF plus
headroom and is never doubled. The gate waits up to -RamWaitSeconds (default 600) for three
readings in a row at or above the requirement and prints a progress line while it waits.
Otherwise it loads nothing, prints the Modal alternative and the "another machine" option in
preflight_host's wording, and exits 2.

The token is LOCAL_MODEL_API_KEY from the root .env (shared with the Ollama and FastFlowLM
gateways). If it is missing, a random 32-byte token is generated and appended there; its
value is never printed. After -IdleMinutes (default 30) without a gateway request, the
gateway stops itself and the host. `stop` stops both process trees and verifies that no host
or gateway process remains and that nothing listens on either port; otherwise it names what
remains and exits 1. At readiness `start` records the python process that listens (uv's
grandchild through the venv launcher): the uv launcher can exit while that process still serves
(seen live in tasks 6.2 and 6.4), so the gateway watches the listener, logs the launcher's exit,
stops only when the listener exits, and every stop also stops the recorded listener. State and
logs: runs/finance-sim/hosts/full-distribution/ (private).

.EXAMPLE
pwsh -File scripts/hosts/serve_full_distribution.ps1 start
pwsh -File scripts/hosts/serve_full_distribution.ps1 status
pwsh -File scripts/hosts/serve_full_distribution.ps1 stop
pwsh -File scripts/hosts/serve_full_distribution.ps1 start -BindAddress 192.168.1.20   # adapter address for a container or another machine
#>
param(
    [Parameter(Position = 0, Mandatory)]
    [ValidateSet('start', 'stop', 'status', 'gateway')]
    [string]$Action,
    [string]$BindAddress = '127.0.0.1',     # Gateway address; never all interfaces.
    [ValidateRange(1024, 65535)][int]$Port = 11581,       # Gateway: the host's loopback port plus one.
    [ValidateRange(1024, 65535)][int]$HostPort = 11580,   # = the profile's full_distribution.endpoint port.
    [ValidateRange(1, 1440)][int]$IdleMinutes = 30,
    [ValidateRange(10, 3600)][int]$StartTimeoutSeconds = 600,
    [ValidateRange(0, 3600)][int]$RamWaitSeconds = 600,
    [ValidateRange(1, 60)][int]$MemorySampleSeconds = 5,   # Gateway free-memory sampling interval (task 6.4).
    [string]$StateDir = 'runs/finance-sim/hosts/full-distribution',   # Relative paths resolve against the repository root.
    [string]$EnvPath = '.env',
    [string]$ProfilePath = 'recall_guard/core/provider_profiles/llamacpp-gemma3-4b.json'
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
$MaxBodyBytes = 1MB              # = the host's own body limit.
$MaxResponseBytes = 64MB
$TokenName = 'LOCAL_MODEL_API_KEY'
# = ENDPOINT_PATH and BATCH_ENDPOINT_PATH in recall_guard.core.full_distribution
$EvidenceRoutes = @('POST /v1/full_distribution', 'POST /v1/full_distribution_batch')
$HostScript = 'scripts/hosts/serve_full_distribution.py'
# Command lines of a host (uv and its python children) and of a gateway (pwsh) on this machine.
$HostCommand = '(?<![\w.-])serve_full_distribution\.py\b'
$GatewayCommand = 'serve_full_distribution\.ps1"?\s+gateway\b'

# ---------- shared helpers (same behavior as serve_local_flm.ps1) ----------

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

function Get-ServingProcesses {
    # Any full-distribution host or gateway on this machine, found by its command line.
    $found = Get-CimInstance Win32_Process -Filter "CommandLine LIKE '%serve[_]full[_]distribution.p%'" -ErrorAction SilentlyContinue
    foreach ($item in @($found)) {
        if (-not $item -or [int]$item.ProcessId -eq $PID) { continue }
        $kind = if ($item.CommandLine -match $GatewayCommand) { 'gateway' } elseif ($item.CommandLine -match $HostCommand) { 'host' } else { $null }
        if ($kind) { [pscustomobject]@{ Kind = $kind; Pid = [int]$item.ProcessId; Name = $item.Name } }
    }
}

function Stop-All([hashtable]$State, [string]$Reason) {
    # Stops the recorded trees and returns the ids it stopped; the first stop's time and reason are kept.
    # The python listener recorded at readiness is stopped too: its uv launcher can exit while it still
    # serves (seen live in tasks 6.2 and 6.4), and then it is no longer in the launcher's tree.
    $stopped = @(Stop-Tree ([int]$State.host_pid) ([string]($State.host_name ?? 'uv')))
    if ($State.listener_pid -and $State.listener_pid -ne $State.host_pid) {
        $stopped += @(Stop-Tree ([int]$State.listener_pid) ([string]($State.listener_name ?? 'python')))
    }
    if ($State.gateway_pid -and $State.gateway_pid -ne $PID) { $stopped += @(Stop-Tree ([int]$State.gateway_pid) 'pwsh') }
    if (-not $State.stopped_at) {
        $State.stopped_at = [DateTime]::UtcNow.ToString('o')
        $State.stop_reason = $Reason
    }
    Write-State $State
    return $stopped
}

function Get-Remaining([hashtable]$State, [int[]]$Ports) {
    # Lines naming every listener and process that remains; empty means fully stopped.
    foreach ($item in @(Get-Listeners $Ports)) { "STILL LISTENING $($item.Address):$($item.Port) pid $($item.Pid)" }
    if ($State) {
        if (Test-Tracked ([int]$State.host_pid) ([string]($State.host_name ?? 'uv'))) { "STILL RUNNING recorded host pid $($State.host_pid)" }
        if ($State.listener_pid -and $State.listener_pid -ne $State.host_pid -and
            (Test-Tracked ([int]$State.listener_pid) ([string]($State.listener_name ?? 'python')))) { "STILL RUNNING recorded listener pid $($State.listener_pid)" }
        if ($State.gateway_pid -ne $PID -and (Test-Tracked ([int]$State.gateway_pid) 'pwsh')) { "STILL RUNNING recorded gateway pid $($State.gateway_pid)" }
    }
    foreach ($item in @(Get-ServingProcesses)) { "STILL RUNNING $($item.Kind) pid $($item.Pid) ($($item.Name))" }
}

# ---------- one model host at a time; RAM gate for one load ----------

function Read-HostProfile {
    if (-not (Test-Path -LiteralPath $ProfileFile -PathType Leaf)) { throw "Profile $ProfileFile not found." }
    $data = Get-Content -LiteralPath $ProfileFile -Raw | ConvertFrom-Json -AsHashtable
    $minimum = [long]($data.host_requirements.min_free_ram_bytes ?? 0)
    if ($minimum -le 0) { throw "Profile $ProfileFile declares no host_requirements.min_free_ram_bytes." }
    # = HostResourceRequirement.memory_floor(): the recorded floor, else one quarter of the load minimum.
    $floor = [long]($data.host_requirements.memory_floor_bytes ?? [Math]::Floor($minimum / 4))
    return @{ model = [string]$data.model_config.model; min_free_ram_bytes = $minimum
              memory_floor_bytes = $floor
              fallbacks = @($data.host_requirements.fallback_profile_ids | Where-Object { $_ })
              sibling_port = ([Uri]$data.model_config.endpoint).Port }
}

function Test-SiblingResident([int]$SiblingPort) {
    # The sibling llama-server holds its load of the same GGUF once it listens on the generation port.
    foreach ($item in @(Get-Listeners @($SiblingPort))) {
        $process = Get-Process -Id $item.Pid -ErrorAction SilentlyContinue
        if ($process -and $process.ProcessName -like 'llama-server*') { return $true }
    }
    return $false
}

function Write-SiblingRefusal([int]$SiblingPort) {
    Write-Output ("Refusing to start: llama-server (task 6.3) is resident on port $SiblingPort, and only one model host " +
                  "runs at a time. Stop it first (pwsh -File scripts/hosts/serve_local_llamacpp.ps1 stop). Nothing was loaded " +
                  "and no request was sent.")
}

function Get-RamRequirement([long]$PerLoad) {
    # One load of the GGUF plus headroom (the profile's min_free_ram_bytes), never doubled: only
    # one model host runs at a time, so no second load can coexist with this one.
    return [ordered]@{ per_load_bytes = $PerLoad; loads = 1; required_bytes = $PerLoad }
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

# ---------- memory guard and host launch (task 6.4) ----------

function Get-HostArguments([int]$ListenPort, [long]$FloorBytes) {
    # uv run arguments of the host: python -X faulthandler, so a native death (for example an
    # allocation failure under memory pressure) leaves a Python stack in host.log, and the floor
    # below which the host refuses new requests with 503 memory_low.
    return @('run', '--no-sync', 'python', '-X', 'faulthandler', $HostScript, '--port', "$ListenPort",
             '--memory-floor-bytes', "$FloorBytes")
}

function Write-MemorySample([string]$Log, [long]$FloorBytes) {
    # One reading of free physical memory, appended to the gateway log; returns the bytes.
    $free = Get-FreeRamBytes
    $below = if ($FloorBytes -gt 0 -and $free -lt $FloorBytes) { ' below_floor' } else { '' }
    Add-Content -LiteralPath $Log -Value "$([DateTime]::UtcNow.ToString('o')) memory free_ram_bytes=$free floor_bytes=$FloorBytes$below"
    return $free
}

function Update-MemorySample([hashtable]$Sampler) {
    # Samples when the next reading is due, on a fixed schedule of $Sampler.seconds; the accept
    # loop, each request and the wait for a host response call it at least every 0.5 s. A failed
    # reading is logged, never fatal.
    $now = [DateTime]::UtcNow
    if ($now -lt $Sampler.next) { return }
    $following = $Sampler.next.AddSeconds($Sampler.seconds)
    $Sampler.next = if ($Sampler.next -eq [DateTime]::MinValue -or $following -le $now) { $now.AddSeconds($Sampler.seconds) } else { $following }
    try { $null = Write-MemorySample $Sampler.log $Sampler.floor }
    catch { Add-Content -LiteralPath $Sampler.log -Value "$($now.ToString('o')) memory sample failed: $($_.Exception.GetType().Name)" }
}

function Test-MemoryLowRefusal([int]$Status, [byte[]]$Body) {
    # The host's refusal below its floor: 503 with {"error": {"code": "memory_low"}}.
    if ($Status -ne 503 -or -not $Body) { return $false }
    try {
        $data = [Text.Encoding]::UTF8.GetString($Body) | ConvertFrom-Json -AsHashtable
        return [bool]($data -is [hashtable] -and $data.error -is [hashtable] -and $data.error.code -ceq 'memory_low')
    } catch { return $false }
}

# ---------- gateway (runs in its own hidden pwsh process) ----------

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

function Invoke-Upstream([Net.Http.HttpClient]$Client, [string]$Upstream, [hashtable]$Request, [hashtable]$Sampler) {
    $message = [Net.Http.HttpRequestMessage]::new([Net.Http.HttpMethod]::Post, "$Upstream$($Request.Path)")
    $message.Content = [Net.Http.ByteArrayContent]::new($Request.Body)
    $message.Content.Headers.TryAddWithoutValidation('Content-Type', ($Request.Headers['content-type'] ?? 'application/json')) | Out-Null
    if ($Request.Headers['accept']) { $message.Headers.TryAddWithoutValidation('Accept', $Request.Headers['accept']) | Out-Null }
    # Authorization is never copied: the host sees an unauthenticated loopback request.
    # Wait in half-second steps so free memory is still sampled while a long batch runs.
    $pending = $Client.SendAsync($message)
    while (-not $pending.Wait(500)) { if ($Sampler) { Update-MemorySample $Sampler } }
    $response = $pending.GetAwaiter().GetResult()
    $body = $response.Content.ReadAsByteArrayAsync().GetAwaiter().GetResult()
    $type = if ($response.Content.Headers.ContentType) { $response.Content.Headers.ContentType.ToString() } else { 'application/json' }
    return [int]$response.StatusCode, $body, $type
}

function Start-Gateway {
    $state = Read-State
    if (-not $state) { throw 'No host state; use start.' }
    $hostProcess = Get-Process -Id ([int]$state.host_pid) -ErrorAction SilentlyContinue
    if (-not (Test-Tracked ([int]$state.host_pid) ([string]($state.host_name ?? 'uv')))) { throw "The recorded host (pid $($state.host_pid)) is not running." }
    $null = $hostProcess.Handle   # Keep a handle: the watch below cannot mistake a reused PID, and the exit code stays readable.
    # The process that serves is the python listener recorded at readiness; its uv launcher can exit (code 0)
    # while it still serves, so the gateway watches the listener when one is recorded and notes the launcher's exit.
    $listenerProcess = $null
    if ($state.listener_pid -and $state.listener_pid -ne $state.host_pid -and
        (Test-Tracked ([int]$state.listener_pid) ([string]($state.listener_name ?? 'python')))) {
        $listenerProcess = Get-Process -Id ([int]$state.listener_pid) -ErrorAction SilentlyContinue
        $null = $listenerProcess.Handle
    }
    $launcherNoted = $false
    $token = Read-Token
    if (-not $token) { throw "$TokenName missing from $EnvFile." }
    $expected = [Text.Encoding]::UTF8.GetBytes($token)
    $token = $null
    $upstream = "http://127.0.0.1:$([int]$state.host_port)"
    $log = Join-Path $StateFolder 'gateway.log'
    $handler = [Net.Http.SocketsHttpHandler]::new()
    $handler.UseProxy = $false
    $handler.AllowAutoRedirect = $false
    $client = [Net.Http.HttpClient]::new($handler)
    # Above any client timeout a plan sets: a long single response takes about 30 s on the iGPU, a
    # study batch several minutes (the plan gives about 600 s per batch, task 6.4).
    $client.Timeout = [TimeSpan]::FromSeconds(1800)
    $client.MaxResponseContentBufferSize = $MaxResponseBytes
    $listener = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Parse($state.bind_address), [int]$state.gateway_port)
    $listener.ExclusiveAddressUse = $true   # SO_EXCLUSIVEADDRUSE: no second socket can share the gateway port.
    $listener.Start()
    $idle = [TimeSpan]::FromMinutes([double]$state.idle_minutes)
    $last = [DateTime]::UtcNow
    $floor = [long]($state.memory_floor_bytes ?? 0)
    if ($floor -le 0) { $floor = try { [long](Read-HostProfile).memory_floor_bytes } catch { 0 } }
    $sampler = @{ log = $log; floor = $floor; seconds = $MemorySampleSeconds; next = [DateTime]::MinValue }
    Add-Content -LiteralPath $log -Value "$([DateTime]::UtcNow.ToString('o')) listening $($state.bind_address):$($state.gateway_port) upstream=$upstream idle=$($state.idle_minutes)m memory_floor=$floor memory_sample=$($MemorySampleSeconds)s"
    Update-MemorySample $sampler
    try {
        while ($true) {
            $accept = $listener.AcceptTcpClientAsync()
            while (-not $accept.Wait(500)) {
                Update-MemorySample $sampler
                if ([DateTime]::UtcNow - $last -gt $idle) {
                    Add-Content -LiteralPath $log -Value "$([DateTime]::UtcNow.ToString('o')) idle timeout, stopping"
                    $listener.Stop()
                    $null = Stop-All (Read-State) 'idle_timeout'
                    return
                }
                if ($listenerProcess -and -not $launcherNoted -and $hostProcess.HasExited) {
                    $code = try { $hostProcess.ExitCode } catch { 'unknown' }
                    Add-Content -LiteralPath $log -Value ("$([DateTime]::UtcNow.ToString('o')) host launcher pid $($state.host_pid) exited " +
                                                          "(exit code $code); the listener pid $($state.listener_pid) still serves")
                    $launcherNoted = $true
                }
                $watched = if ($listenerProcess) { $listenerProcess } else { $hostProcess }
                if ($watched.HasExited) {
                    $code = try { $watched.ExitCode } catch { 'unknown' }
                    Add-Content -LiteralPath $log -Value "$([DateTime]::UtcNow.ToString('o')) host exited (exit code $code), stopping"
                    $listener.Stop()
                    $null = Stop-All (Read-State) 'host_exited'
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
                    } elseif ($route -cnotin $EvidenceRoutes) {
                        $status = 404
                        Send-Error $stream 404 'route not served by the gateway'
                    } else {
                        try {
                            $status, $body, $type = Invoke-Upstream $client $upstream $request $sampler
                            Send-Response $stream $status $body $type
                            if (Test-MemoryLowRefusal $status $body) {
                                Add-Content -LiteralPath $log -Value "$([DateTime]::UtcNow.ToString('o')) host refused memory_low (503 on $route)"
                                $sampler.next = [DateTime]::MinValue   # A reading next to the refusal.
                                Update-MemorySample $sampler
                            }
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
            Update-MemorySample $sampler   # Also due readings while requests arrive back to back.
        }
    } finally {
        $listener.Stop()
    }
}

# ---------- actions ----------

function Resolve-BindAddress([string]$Address) {
    $parsed = $null
    if (-not [Net.IPAddress]::TryParse($Address, [ref]$parsed) -or
        $parsed.Equals([Net.IPAddress]::Any) -or $parsed.Equals([Net.IPAddress]::IPv6Any)) {
        throw "Refusing bind address $Address. Give 127.0.0.1 or the IP address of one adapter, never all interfaces."
    }
    return $parsed
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

function Resolve-Uv {
    $command = Get-Command uv -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $command) { throw 'uv not found on PATH; the host starts with uv run --no-sync python.' }
    return $command.Source
}

function Start-Serving {
    $state = Read-State
    if ($state -and -not $state.stopped_at) { throw 'A host state is active. Run stop first.' }
    $address = Resolve-BindAddress $BindAddress
    $bind = $address.ToString()
    if (Get-Listeners @($HostPort, $Port)) { throw "Port $HostPort or $Port already listens. Stop the other host or gateway first." }
    $running = @(Get-ServingProcesses)
    if ($running) { throw "A full-distribution $($running[0].Kind) already runs (pid $($running.Pid -join ', ')). Run stop first." }
    $uv = Resolve-Uv
    $hostProfile = Read-HostProfile
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
    New-Item -ItemType Directory -Force -Path $StateFolder | Out-Null
    $tokenState = Initialize-Token
    Disable-OutputInheritance

    $hostArguments = Get-HostArguments $HostPort $hostProfile.memory_floor_bytes
    $hostProcess = Start-Process -FilePath $uv -WorkingDirectory $Root -WindowStyle Hidden -PassThru `
        -ArgumentList $hostArguments `
        -RedirectStandardOutput (Join-Path $StateFolder 'host.out.log') -RedirectStandardError (Join-Path $StateFolder 'host.log')
    $started = [DateTime]::UtcNow
    $origin = if ($address.AddressFamily -eq [Net.Sockets.AddressFamily]::InterNetworkV6) { "[$bind]" } else { $bind }
    $state = @{ host_pid = $hostProcess.Id; host_name = $hostProcess.ProcessName; gateway_pid = $null
                listener_pid = $null; listener_name = $null
                bind_address = $bind; gateway_port = $Port; host_port = $HostPort; host_address = "127.0.0.1:$HostPort"
                idle_minutes = $IdleMinutes; started_at = $started.ToString('o')
                host_arguments = $hostArguments; memory_floor_bytes = $hostProfile.memory_floor_bytes
                memory_sample_seconds = $MemorySampleSeconds
                ram_gate = [ordered]@{ per_load_bytes = $need.per_load_bytes; loads = $need.loads
                                       required_bytes = $need.required_bytes; sibling_resident = $false
                                       free_ram_bytes = $gate.free_ram_bytes; samples = $gate.samples
                                       waited_seconds = $gate.waited_seconds }
                ready_at = $null; readiness_seconds = $null; stopped_at = $null; stop_reason = $null
                endpoint = "http://${origin}:$Port" }
    Write-State $state
    # The host listens only after hashing the GGUF, loading it and decoding the warm-up text.
    $deadline = $started.AddSeconds($StartTimeoutSeconds)
    $noted = $started
    do {
        Start-Sleep -Milliseconds 500
        $listening = @(Get-Listeners @($HostPort))
        $listener = if ($listening) { $tree = @(Get-ProcessTree $hostProcess.Id); $listening | Where-Object { $_.Pid -in $tree } | Select-Object -First 1 }
        $ready = [bool]$listener
        $alive = -not $hostProcess.HasExited
        if (-not $ready -and ([DateTime]::UtcNow - $noted).TotalSeconds -ge 30) {
            $noted = [DateTime]::UtcNow
            Write-Output ("Waiting for the host to listen on 127.0.0.1:{0} (hash, load, warm-up): {1:0} s" -f $HostPort, ($noted - $started).TotalSeconds)
        }
    } until ($ready -or -not $alive -or [DateTime]::UtcNow -gt $deadline)
    if (-not $ready) {
        $null = Stop-All $state 'start_failed'
        $hostLog = Join-Path $StateFolder 'host.log'
        $reason = if (Test-Path -LiteralPath $hostLog) { (Get-Content -LiteralPath $hostLog -Tail 1) } else { '' }
        throw "The host did not become ready (alive: $alive). $reason See $hostLog."
    }
    # The python process that listens (a grandchild of uv through the venv launcher): stop and the gateway
    # reach it even if uv exits while it serves.
    $state.listener_pid = $listener.Pid
    $state.listener_name = (Get-Process -Id $listener.Pid -ErrorAction SilentlyContinue).ProcessName
    $state.ready_at = [DateTime]::UtcNow.ToString('o')
    $state.readiness_seconds = [Math]::Round(([DateTime]::UtcNow - $started).TotalSeconds, 1)
    Write-State $state
    # Start-Process joins arguments with spaces, so paths are quoted here.
    $gateway = Start-Process -FilePath (Get-Process -Id $PID).Path -WindowStyle Hidden -PassThru `
        -ArgumentList '-NoProfile', '-NonInteractive', '-File', "`"$PSCommandPath`"", 'gateway', '-Port', "$Port", '-HostPort', "$HostPort",
                      '-StateDir', "`"$StateFolder`"", '-EnvPath', "`"$EnvFile`"", '-MemorySampleSeconds', "$MemorySampleSeconds" `
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
    Write-Output "Host 127.0.0.1:$HostPort (pid $($hostProcess.Id), listener pid $($listener.Pid)), ready after $($state.readiness_seconds) s"
    Write-Output "Gateway ${bind}:$Port (pid $($gateway.Id)), $TokenName $tokenState in $EnvFile (value not shown)"
    Write-Output "Gateway endpoint (FULL_DISTRIBUTION_ENDPOINT_URL): $($state.endpoint), routes $($EvidenceRoutes -join ' and ') only"
    Write-Output "RAM gate: $(Format-Gib $gate.free_ram_bytes) free, $(Format-Gib $need.required_bytes) required (one load; llama-server not resident)"
    Write-Output ("Memory guard: the host (python -X faulthandler) refuses new requests with 503 memory_low below " +
                  "$(Format-Gib $hostProfile.memory_floor_bytes) free; free memory is sampled every 5 s (host.out.log) " +
                  "and every $MemorySampleSeconds s (gateway.log).")
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
    # Wait (bounded) until every stopped process has exited, then check once.
    $deadline = [DateTime]::UtcNow.AddSeconds(15)
    while ($stopped | Where-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue }) {
        if ([DateTime]::UtcNow -gt $deadline) { break }
        Start-Sleep -Milliseconds 250
    }
    $remaining = @(Get-Remaining $state $ports)
    foreach ($line in $remaining) { Write-Output $line }
    if ($remaining) { exit 1 }
    Write-Output "Stopped: no host or gateway process remains; nothing listens on $($ports -join ' or ')."
}

function Show-Status {
    $state = Read-State
    if (-not $state) { Write-Output 'No host state (never started).' }
    else {
        foreach ($pair in @(@('host_pid', ($state.host_name ?? 'uv')), @('listener_pid', ($state.listener_name ?? 'python')), @('gateway_pid', 'pwsh'))) {
            $running = Test-Tracked ([int]$state[$pair[0]]) $pair[1]
            Write-Output ("{0} {1}: {2}" -f $pair[0], $state[$pair[0]], $(if ($running) { 'running' } else { 'not running' }))
        }
        Write-Output "host $($state.host_address), gateway $($state.bind_address):$($state.gateway_port), idle $($state.idle_minutes) min"
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
