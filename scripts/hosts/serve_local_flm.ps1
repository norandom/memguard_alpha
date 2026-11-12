<#
.SYNOPSIS
Serve native Windows FastFlowLM (AMD NPU) to Recall Guard containers in WSL through an authenticated gateway.

.DESCRIPTION
Host-side tooling (PowerShell 7, no admin). FastFlowLM must run natively on
Windows: WSL and Docker have no NPU access. `start <tag>` launches ONE
`flm serve <tag> --host 127.0.0.1 -p 11540 --quiet -c 2048` (FLM serves one
model per process; its default port 52625 lies in a Hyper-V excluded range),
with FLM_MODEL_PATH=<repo>/models/flm and FLM_DISABLE_UPDATE_CHECK=1. It then
launches a small gateway that listens ONLY on the Windows address of the WSL
virtual adapter (the WSL default gateway, for example 172.17.32.1:11541).

The gateway is the sibling of scripts/hosts/serve_local_ollama.ps1 (same request
parser, limits and logging; that script is unchanged). It requires
`Authorization: Bearer <LOCAL_MODEL_API_KEY>` (constant-time compare; 401
otherwise), strips that header, and forwards to 127.0.0.1:11540. It serves only
POST /v1/chat/completions (for the served model only), GET /api/version,
/api/ps, /api/npu/status, and a gateway-synthesized GET /api/tags. FLM's own
/api/tags and /v1/models list its whole catalog, so the synthesized answer lists
ONLY the installed models (`flm list --filter installed --json`, read at start),
marks itself with "source", and adds "serving" (served tag and the launch
context length). Each model's "digest" is the FLM catalog's LFS sha256 of its
weights (model_info.json in the FLM install directory), not a re-hash.
There is no unload route: unloading means stopping the FLM process.

The token is LOCAL_MODEL_API_KEY from the root .env (shared with the Ollama
gateway). If it is missing, a random 32-byte token is generated and appended
there; its value is never printed. After -IdleMinutes (default 30) without a
gateway request, the gateway stops itself and FLM. `stop` stops both process
trees and verifies that nothing remains. State and logs: runs/flm-gateway/ (ignored).

.EXAMPLE
pwsh -File scripts/hosts/serve_local_flm.ps1 start gemma3:4b
pwsh -File scripts/hosts/serve_local_flm.ps1 status
pwsh -File scripts/hosts/serve_local_flm.ps1 stop
pwsh -File scripts/hosts/serve_local_flm.ps1 tags gemma3:4b   # print the synthesized /api/tags; starts nothing
#>
param(
    [Parameter(Position = 0, Mandatory)]
    [ValidateSet('start', 'stop', 'status', 'tags', 'gateway')]
    [string]$Action,
    [Parameter(Position = 1)]
    [ValidatePattern('^[a-z0-9][a-z0-9.\-]*:[a-z0-9][a-z0-9.\-]*$')]
    [string]$Model,
    [string]$FlmPath,
    [string]$BindAddress,                 # Default: IPv4 of the "vEthernet (WSL...)" adapter.
    [ValidateRange(1024, 65535)][int]$Port = 11541,
    [ValidateRange(1024, 65535)][int]$FlmPort = 11540,
    [ValidateRange(256, 131072)][int]$ContextLength = 2048,
    [ValidateRange(1, 1440)][int]$IdleMinutes = 30,
    [ValidateRange(10, 1800)][int]$StartTimeoutSeconds = 600
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$StateDir = Join-Path $Root 'runs/flm-gateway'
$StatePath = Join-Path $StateDir 'state.json'
$TagsPath = Join-Path $StateDir 'tags.json'
$EnvPath = Join-Path $Root '.env'
$Upstream = "http://127.0.0.1:$FlmPort"
$MaxBodyBytes = 1MB
$MaxResponseBytes = 64MB
$TokenName = 'LOCAL_MODEL_API_KEY'
$TagsSource = 'flm list --filter installed --json'   # = recall_guard.core.profiles.FLM_INSTALLED_TAGS_SOURCE

# ---------- shared helpers (same behavior as serve_local_ollama.ps1) ----------

function Read-State {
    if (Test-Path -LiteralPath $StatePath) { return Get-Content -LiteralPath $StatePath -Raw | ConvertFrom-Json -AsHashtable -DateKind String }
    return $null
}

function Write-State([hashtable]$State) {
    New-Item -ItemType Directory -Force -Path $StateDir | Out-Null
    $temporary = "$StatePath.tmp"
    [IO.File]::WriteAllText($temporary, ($State | ConvertTo-Json), [Text.UTF8Encoding]::new($false))
    Move-Item -LiteralPath $temporary -Destination $StatePath -Force
}

function Get-WslAdapterAddress {
    $address = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
        Where-Object InterfaceAlias -like 'vEthernet (WSL*' | Select-Object -First 1
    if (-not $address) { throw 'No "vEthernet (WSL...)" adapter found. Start WSL (wsl -d Debian) first, or pass -BindAddress.' }
    return $address.IPAddress
}

function Read-Token {
    if (-not (Test-Path -LiteralPath $EnvPath)) { return $null }
    foreach ($line in [IO.File]::ReadAllLines($EnvPath)) {
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
    $text = if (Test-Path -LiteralPath $EnvPath) { [IO.File]::ReadAllText($EnvPath) } else { '' }
    $newline = if ($text -match "`r`n" -or -not $text) { "`r`n" } else { "`n" }
    $line = "$TokenName=$token"
    if ($text -match "(?m)^\s*$TokenName\s*=\s*$") {
        $text = [regex]::Replace($text, "(?m)^\s*$TokenName\s*=[ \t]*(?=\r?$)", $line)  # Fill an empty entry.
    } else {
        if ($text -and -not $text.EndsWith("`n")) { $text += $newline }
        $text += "# Bearer token for the local model gateways (scripts/hosts/serve_local_ollama.ps1, serve_local_flm.ps1).$newline$line$newline"
    }
    [IO.File]::WriteAllText($EnvPath, $text, [Text.UTF8Encoding]::new($false))
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

function Stop-Tree([int]$ProcessId, [string]$ExpectedName) {
    # The name check avoids killing an unrelated process that reused a stale PID.
    $process = if ($ProcessId) { Get-Process -Id $ProcessId -ErrorAction SilentlyContinue }
    if (-not $process -or $process.ProcessName -notlike $ExpectedName) { return }
    $ids = [int[]]@(Get-ProcessTree $ProcessId)
    [array]::Reverse($ids)  # Children first (conhost and any FLM helper).
    foreach ($id in $ids) { Stop-Process -Id $id -Force -ErrorAction SilentlyContinue }
}

function Get-Listeners {
    Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
        Where-Object { $_.LocalPort -in @($FlmPort, $Port) } |
        ForEach-Object { [pscustomobject]@{ Address = $_.LocalAddress; Port = $_.LocalPort; Pid = $_.OwningProcess } }
}

function Stop-All([hashtable]$State, [string]$Reason) {
    Stop-Tree ([int]$State.flm_pid) 'flm'
    if ($State.gateway_pid -and $State.gateway_pid -ne $PID) { Stop-Tree ([int]$State.gateway_pid) 'pwsh' }
    $State.stopped_at = [DateTime]::UtcNow.ToString('o')
    $State.stop_reason = $Reason
    Write-State $State
}

# ---------- FLM metadata ----------

function Resolve-Flm {
    if ($FlmPath) { $path = $FlmPath }
    else {
        $command = Get-Command flm -ErrorAction SilentlyContinue
        $path = if ($command) { $command.Source } else { Join-Path $env:LOCALAPPDATA 'Programs\FastFlowLM-1.0.12\flm.exe' }
    }
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw 'flm.exe not found. Supply -FlmPath.' }
    return (Resolve-Path -LiteralPath $path).Path
}

function Set-FlmEnvironment {
    $store = Join-Path $Root 'models/flm'
    if (-not (Test-Path -LiteralPath $store -PathType Container)) { throw "FLM model store $store not found." }
    # The user-level FLM_MODEL_PATH is not inherited by every process; set it explicitly.
    $env:FLM_MODEL_PATH = (Resolve-Path -LiteralPath $store).Path
    $env:FLM_DISABLE_UPDATE_CHECK = '1'
}

function New-InstalledTags([string]$Flm, [string]$Served) {
    # `flm list --filter installed` is the only installed-model source; the catalog is not.
    $listing = (& $Flm list --filter installed --json 2>$null | Out-String)
    if ($LASTEXITCODE -ne 0) { throw 'flm list --filter installed --json failed.' }
    $installed = @(($listing | ConvertFrom-Json -AsHashtable).models)
    $info = @{}
    $infoPath = Join-Path (Split-Path -Parent $Flm) 'model_info.json'
    if (Test-Path -LiteralPath $infoPath) { $info = Get-Content -LiteralPath $infoPath -Raw | ConvertFrom-Json -AsHashtable }
    $models = foreach ($entry in $installed) {
        if (-not $entry.installed) { continue }
        $weights = @($info[$entry.name]) | Where-Object { $_ -and $_.path -eq 'model.q4nx' } | Select-Object -First 1
        $oid = if ($weights -and $weights.lfs) { [string]$weights.lfs.oid } else { '' }
        [ordered]@{ name = $entry.name; model = $entry.model; size = $entry.size; details = $entry.details
                    digest = if ($oid -match '^[0-9a-f]{64}$') { "sha256:$oid" } else { $null }
                    footprint_gb = $entry.footprint; default_context_length = $entry.default_context_length }
    }
    $models = @($models)
    if (-not ($models | Where-Object { $_.name -eq $Served })) {
        $names = ($models | ForEach-Object { $_.name }) -join ', '
        throw "$Served is not installed for FLM (installed: $names). Nothing is downloaded automatically; run flm pull $Served yourself."
    }
    return [ordered]@{ source = $TagsSource; runtime = 'fastflowlm'; models = $models
                       serving = [ordered]@{ model = $Served; context_length = $ContextLength
                                             source = 'flm serve --ctx-len (process start)' } }
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

function Get-Route([hashtable]$Request, [string]$Served) {
    # 'tags' (synthesized), 'forward', or $null (404).
    $route = "$($Request.Method) $($Request.Path)"
    if ($route -eq 'GET /api/tags') { return 'tags' }
    if ($route -in 'GET /api/version', 'GET /api/ps', 'GET /api/npu/status') { return 'forward' }
    if ($route -ne 'POST /v1/chat/completions') { return $null }
    try { $json = [Text.Encoding]::UTF8.GetString($Request.Body) | ConvertFrom-Json -AsHashtable } catch { return $null }
    # One model per FLM process: a request for any other model never reaches FLM.
    if ($json -is [hashtable] -and $json.model -is [string] -and $json.model -eq $Served) { return 'forward' }
    return $null
}

function Invoke-Upstream([Net.Http.HttpClient]$Client, [hashtable]$Request) {
    $message = [Net.Http.HttpRequestMessage]::new([Net.Http.HttpMethod]::new($Request.Method), "$Upstream$($Request.Path)")
    if ($Request.Method -eq 'POST') {
        $message.Content = [Net.Http.ByteArrayContent]::new($Request.Body)
        $message.Content.Headers.TryAddWithoutValidation('Content-Type', ($Request.Headers['content-type'] ?? 'application/json')) | Out-Null
    }
    if ($Request.Headers['accept']) { $message.Headers.TryAddWithoutValidation('Accept', $Request.Headers['accept']) | Out-Null }
    # Authorization is never copied: FLM sees an unauthenticated loopback request.
    $response = $Client.SendAsync($message).GetAwaiter().GetResult()
    $body = $response.Content.ReadAsByteArrayAsync().GetAwaiter().GetResult()
    $type = if ($response.Content.Headers.ContentType) { $response.Content.Headers.ContentType.ToString() } else { 'application/json' }
    return [int]$response.StatusCode, $body, $type
}

function Start-Gateway {
    $state = Read-State
    if (-not $state) { throw 'No gateway state; use start.' }
    $token = Read-Token
    if (-not $token) { throw "$TokenName missing from .env." }
    $expected = [Text.Encoding]::UTF8.GetBytes($token)
    $tags = [IO.File]::ReadAllBytes($TagsPath)
    $served = [string]$state.model
    $log = Join-Path $StateDir 'gateway.log'
    $handler = [Net.Http.SocketsHttpHandler]::new()
    $handler.UseProxy = $false
    $handler.AllowAutoRedirect = $false
    $client = [Net.Http.HttpClient]::new($handler)
    $client.Timeout = [TimeSpan]::FromSeconds(330)
    $client.MaxResponseContentBufferSize = $MaxResponseBytes
    $listener = [Net.Sockets.TcpListener]::new([Net.IPAddress]::Parse($state.bind_address), [int]$state.gateway_port)
    $listener.Start()
    $idle = [TimeSpan]::FromMinutes([int]$state.idle_minutes)
    $last = [DateTime]::UtcNow
    Add-Content -LiteralPath $log -Value "$([DateTime]::UtcNow.ToString('o')) listening $($state.bind_address):$($state.gateway_port) model=$served idle=$($state.idle_minutes)m"
    try {
        while ($true) {
            $accept = $listener.AcceptTcpClientAsync()
            while (-not $accept.Wait(5000)) {
                if ([DateTime]::UtcNow - $last -gt $idle) {
                    Add-Content -LiteralPath $log -Value "$([DateTime]::UtcNow.ToString('o')) idle timeout, stopping"
                    $listener.Stop()
                    Stop-All (Read-State) 'idle_timeout'
                    return
                }
                if (-not (Get-Process -Id ([int]$state.flm_pid) -ErrorAction SilentlyContinue)) {
                    Add-Content -LiteralPath $log -Value "$([DateTime]::UtcNow.ToString('o')) flm exited, stopping"
                    $listener.Stop()
                    Stop-All (Read-State) 'flm_exited'
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
                    $kind = Get-Route $request $served
                    if (-not (Test-Token $request.Headers['authorization'] $expected)) {
                        $status = 401
                        Send-Error $stream 401 'unauthorized'
                    } elseif (-not $kind) {
                        $status = 404
                        Send-Error $stream 404 'route not served by the gateway'
                    } elseif ($kind -eq 'tags') {
                        $status = 200
                        Send-Response $stream 200 $tags
                    } else {
                        try {
                            $status, $body, $type = Invoke-Upstream $client $request
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

function Start-Serving {
    if (-not $Model) { throw 'start needs the FLM model tag, for example: start gemma3:4b' }
    $state = Read-State
    if ($state -and -not $state.stopped_at) { throw 'A gateway state is active. Run stop first.' }
    if (Get-Listeners) { throw "Port $FlmPort or $Port already listens. Stop the other FLM or gateway first." }
    $bind = if ($BindAddress) { $BindAddress } else { Get-WslAdapterAddress }
    if ($bind -in '0.0.0.0', '::', '*' -or -not [Net.IPAddress]::TryParse($bind, [ref]$null)) { throw "Refusing bind address $bind." }
    $flm = Resolve-Flm
    Set-FlmEnvironment
    New-Item -ItemType Directory -Force -Path $StateDir | Out-Null
    $tags = New-InstalledTags $flm $Model
    [IO.File]::WriteAllText($TagsPath, ($tags | ConvertTo-Json -Depth 8 -Compress), [Text.UTF8Encoding]::new($false))
    $tokenState = Initialize-Token
    $version = (& $flm version --json 2>$null | Out-String).Trim()
    Disable-OutputInheritance

    $flmProcess = Start-Process -FilePath $flm -WindowStyle Hidden -PassThru `
        -ArgumentList 'serve', $Model, '--host', '127.0.0.1', '-p', "$FlmPort", '--quiet', '-c', "$ContextLength" `
        -RedirectStandardOutput (Join-Path $StateDir 'flm.out.log') -RedirectStandardError (Join-Path $StateDir 'flm.log')

    $state = @{ flm_pid = $flmProcess.Id; gateway_pid = $null; model = $Model; bind_address = $bind; gateway_port = $Port
                flm_address = "127.0.0.1:$FlmPort"; context_length = $ContextLength; idle_minutes = $IdleMinutes
                flm_version = $version; started_at = [DateTime]::UtcNow.ToString('o'); stopped_at = $null
                stop_reason = $null; container_endpoint = "http://${bind}:$Port/v1/chat/completions" }
    Write-State $state
    # FLM listens once the model is loaded on the NPU; large models take a while.
    $deadline = [DateTime]::UtcNow.AddSeconds($StartTimeoutSeconds)
    do {
        Start-Sleep -Milliseconds 500
        $ports = @(Get-Listeners | ForEach-Object Port)
        $alive = [bool](Get-Process -Id $flmProcess.Id -ErrorAction SilentlyContinue)
    } until ($ports -contains $FlmPort -or -not $alive -or [DateTime]::UtcNow -gt $deadline)
    if (-not ($ports -contains $FlmPort)) {
        Stop-All $state 'start_failed'
        throw "FLM did not start (alive: $alive). See $StateDir\flm.log."
    }
    Disable-OutputInheritance
    $gateway = Start-Process -FilePath (Get-Process -Id $PID).Path -WindowStyle Hidden -PassThru `
        -ArgumentList '-NoProfile', '-NonInteractive', '-File', $PSCommandPath, 'gateway', '-Port', "$Port", '-FlmPort', "$FlmPort" `
        -RedirectStandardOutput (Join-Path $StateDir 'gateway.out.log') -RedirectStandardError (Join-Path $StateDir 'gateway.err.log')
    $state.gateway_pid = $gateway.Id
    Write-State $state
    $deadline = [DateTime]::UtcNow.AddSeconds(30)
    do {
        Start-Sleep -Milliseconds 500
        $ports = @(Get-Listeners | ForEach-Object Port)
    } until (($ports -contains $FlmPort -and $ports -contains $Port) -or [DateTime]::UtcNow -gt $deadline)
    if (-not ($ports -contains $FlmPort -and $ports -contains $Port)) {
        Stop-All $state 'start_failed'
        throw "Start failed (listening: $($ports -join ', ')). See $StateDir\flm.log and gateway.err.log."
    }
    Write-Output "FLM $Model 127.0.0.1:$FlmPort (pid $($flmProcess.Id), context $ContextLength)"
    Write-Output "Gateway ${bind}:$Port (pid $($gateway.Id)), $TokenName $tokenState in .env (value not shown)"
    Write-Output "Container endpoint: $($state.container_endpoint)"
    Write-Output "Auto-stop after $IdleMinutes min without gateway requests."
}

function Stop-Serving {
    $state = Read-State
    if ($state) {
        Stop-All $state ($state.stop_reason ?? 'stop_command')
        Start-Sleep -Seconds 1
    }
    $alive = @($state.flm_pid, $state.gateway_pid) | Where-Object { $_ } |
        Where-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue }
    # Any flm.exe on this host counts: one NPU, one served model at a time.
    $flms = @(Get-Process -Name flm -ErrorAction SilentlyContinue)
    $listeners = @(Get-Listeners)
    foreach ($item in $listeners) { Write-Output "STILL LISTENING $($item.Address):$($item.Port) pid $($item.Pid)" }
    foreach ($id in $alive) { Write-Output "STILL RUNNING pid $id" }
    foreach ($item in $flms) { Write-Output "STILL RUNNING flm pid $($item.Id)" }
    if ($listeners -or $alive -or $flms) { exit 1 }
    Write-Output "Stopped: no gateway or FLM process remains; nothing listens on $FlmPort or $Port."
}

function Show-Status {
    $state = Read-State
    if (-not $state) { Write-Output 'No gateway state (never started).' }
    else {
        foreach ($name in 'flm_pid', 'gateway_pid') {
            $running = $state[$name] -and (Get-Process -Id $state[$name] -ErrorAction SilentlyContinue)
            Write-Output ("{0} {1}: {2}" -f $name, $state[$name], $(if ($running) { 'running' } else { 'not running' }))
        }
        Write-Output "model $($state.model), context $($state.context_length), bind $($state.bind_address):$($state.gateway_port), idle $($state.idle_minutes) min"
        Write-Output "started $($state.started_at); stopped $($state.stopped_at) ($($state.stop_reason))"
        $log = Join-Path $StateDir 'gateway.log'
        if (Test-Path -LiteralPath $log) { Write-Output "last gateway log: $(Get-Content -LiteralPath $log -Tail 1)" }
    }
    $listeners = @(Get-Listeners)
    if (-not $listeners) { Write-Output "No listener on $FlmPort or $Port." }
    foreach ($item in $listeners) { Write-Output "listening $($item.Address):$($item.Port) pid $($item.Pid)" }
}

switch ($Action) {
    'start' { Start-Serving }
    'stop' { Stop-Serving }
    'status' { Show-Status }
    'tags' {
        if (-not $Model) { throw 'tags needs the FLM model tag that would be served.' }
        $flm = Resolve-Flm
        Set-FlmEnvironment
        Write-Output (New-InstalledTags $flm $Model | ConvertTo-Json -Depth 8 -Compress)
    }
    'gateway' { Start-Gateway }
}
