<#
.SYNOPSIS
Serve native Windows Ollama to Recall Guard containers in WSL through an authenticated gateway.

.DESCRIPTION
Host-side tooling (PowerShell 7, no admin). `start` launches Ollama exactly like
start_local_ollama.ps1 (project models/ store, context 2048, one model, one
request, no cloud) plus OLLAMA_IGPU_ENABLE=1, bound to 127.0.0.1:11434 only. It
then launches a small gateway that listens ONLY on the Windows address of the
WSL virtual adapter (the WSL default gateway, for example 172.17.32.1:11435).

The gateway requires `Authorization: Bearer <LOCAL_MODEL_API_KEY>` (constant-time
compare; 401 otherwise), strips that header, and forwards to 127.0.0.1:11434.
It serves only GET /api/version, /api/tags, /api/ps, POST /api/show,
/v1/chat/completions and an unload-only POST /api/generate
({"model": ..., "keep_alive": 0}). Bodies are limited to 1 MiB; responses are
buffered and returned with Content-Length.

The token is LOCAL_MODEL_API_KEY from the root .env. If it is missing, a random
32-byte token is generated and appended there; its value is never printed.
After -IdleMinutes (default 30) without a gateway request, the gateway stops
itself and Ollama. `stop` stops both and verifies that nothing remains.
State and logs: runs/ollama-gateway/ (ignored).

.EXAMPLE
pwsh -File scripts/hosts/serve_local_ollama.ps1 start
pwsh -File scripts/hosts/serve_local_ollama.ps1 status
pwsh -File scripts/hosts/serve_local_ollama.ps1 stop
#>
param(
    [Parameter(Position = 0, Mandatory)]
    [ValidateSet('start', 'stop', 'status', 'gateway')]
    [string]$Action,
    [string]$OllamaPath,
    [string]$BindAddress,                 # Default: IPv4 of the "vEthernet (WSL...)" adapter.
    [ValidateRange(1024, 65535)][int]$Port = 11435,
    [ValidateRange(1, 1440)][int]$IdleMinutes = 30,
    [switch]$NoIgpu
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$StateDir = Join-Path $Root 'runs/ollama-gateway'
$StatePath = Join-Path $StateDir 'state.json'
$EnvPath = Join-Path $Root '.env'
$OllamaPort = 11434
$Upstream = "http://127.0.0.1:$OllamaPort"
$MaxBodyBytes = 1MB
$MaxResponseBytes = 64MB
$TokenName = 'LOCAL_MODEL_API_KEY'

# ---------- shared helpers ----------

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
        $text += "# Bearer token for scripts/hosts/serve_local_ollama.ps1 (Windows Ollama gateway).$newline$line$newline"
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
    [array]::Reverse($ids)  # Children first (llama-server / ollama runner).
    foreach ($id in $ids) { Stop-Process -Id $id -Force -ErrorAction SilentlyContinue }
}

function Get-Listeners {
    Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
        Where-Object { $_.LocalPort -in @($OllamaPort, $Port) } |
        ForEach-Object { [pscustomobject]@{ Address = $_.LocalAddress; Port = $_.LocalPort; Pid = $_.OwningProcess } }
}

function Stop-All([hashtable]$State, [string]$Reason) {
    Stop-Tree ([int]$State.ollama_pid) 'ollama*'
    if ($State.gateway_pid -and $State.gateway_pid -ne $PID) { Stop-Tree ([int]$State.gateway_pid) 'pwsh' }
    $State.stopped_at = [DateTime]::UtcNow.ToString('o')
    $State.stop_reason = $Reason
    Write-State $State
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

function Test-Route([hashtable]$Request) {
    $route = "$($Request.Method) $($Request.Path)"
    if ($route -in 'GET /api/version', 'GET /api/tags', 'GET /api/ps', 'POST /api/show', 'POST /v1/chat/completions') { return $true }
    if ($route -ne 'POST /api/generate') { return $false }
    try { $json = [Text.Encoding]::UTF8.GetString($Request.Body) | ConvertFrom-Json -AsHashtable } catch { return $false }
    # Unload only: exactly {"model": "...", "keep_alive": 0}. No prompt, so no generation.
    return ($json -is [hashtable] -and $json.Count -eq 2 -and $json.model -is [string] -and
            $json.ContainsKey('keep_alive') -and "$($json.keep_alive)" -eq '0')
}

function Invoke-Upstream([Net.Http.HttpClient]$Client, [hashtable]$Request) {
    $message = [Net.Http.HttpRequestMessage]::new([Net.Http.HttpMethod]::new($Request.Method), "$Upstream$($Request.Path)")
    if ($Request.Method -eq 'POST') {
        $message.Content = [Net.Http.ByteArrayContent]::new($Request.Body)
        $message.Content.Headers.TryAddWithoutValidation('Content-Type', ($Request.Headers['content-type'] ?? 'application/json')) | Out-Null
    }
    if ($Request.Headers['accept']) { $message.Headers.TryAddWithoutValidation('Accept', $Request.Headers['accept']) | Out-Null }
    # Authorization is never copied: Ollama sees an unauthenticated loopback request.
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
    Add-Content -LiteralPath $log -Value "$([DateTime]::UtcNow.ToString('o')) listening $($state.bind_address):$($state.gateway_port) idle=$($state.idle_minutes)m"
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
                if (-not (Get-Process -Id ([int]$state.ollama_pid) -ErrorAction SilentlyContinue)) {
                    Add-Content -LiteralPath $log -Value "$([DateTime]::UtcNow.ToString('o')) ollama exited, stopping"
                    $listener.Stop()
                    Stop-All (Read-State) 'ollama_exited'
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
                    } elseif (-not (Test-Route $request)) {
                        $status = 404
                        Send-Error $stream 404 'route not served by the gateway'
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

function Start-Serving {
    $state = Read-State
    if ($state -and -not $state.stopped_at) { throw 'A gateway state is active. Run stop first.' }
    if (Get-Listeners) { throw "Port $OllamaPort or $Port already listens. Stop the other Ollama or gateway first." }
    $bind = if ($BindAddress) { $BindAddress } else { Get-WslAdapterAddress }
    if ($bind -in '0.0.0.0', '::', '*' -or -not [Net.IPAddress]::TryParse($bind, [ref]$null)) { throw "Refusing bind address $bind." }
    $tokenState = Initialize-Token

    if (-not $OllamaPath) {
        $command = Get-Command ollama -ErrorAction SilentlyContinue
        $script:OllamaPath = if ($command) { $command.Source } else { Join-Path $env:LOCALAPPDATA 'AMD\AI_Bundle\Ollama\ollama.exe' }
    }
    if (-not (Test-Path -LiteralPath $OllamaPath -PathType Leaf)) { throw 'Ollama executable not found. Supply -OllamaPath.' }
    New-Item -ItemType Directory -Force -Path $StateDir | Out-Null

    $env:OLLAMA_MODELS = (Resolve-Path -LiteralPath (Join-Path $Root 'models')).Path
    $env:OLLAMA_CONTEXT_LENGTH = '2048'
    $env:OLLAMA_NUM_PARALLEL = '1'
    $env:OLLAMA_MAX_LOADED_MODELS = '1'
    $env:OLLAMA_NO_CLOUD = '1'
    $env:OLLAMA_HOST = "127.0.0.1:$OllamaPort"
    $env:OLLAMA_IGPU_ENABLE = if ($NoIgpu) { '0' } else { '1' }
    $ollama = Start-Process -FilePath $OllamaPath -ArgumentList 'serve' -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $StateDir 'ollama.out.log') -RedirectStandardError (Join-Path $StateDir 'ollama.log')

    $state = @{ ollama_pid = $ollama.Id; gateway_pid = $null; bind_address = $bind; gateway_port = $Port
                ollama_address = "127.0.0.1:$OllamaPort"; idle_minutes = $IdleMinutes; igpu_enabled = -not $NoIgpu
                started_at = [DateTime]::UtcNow.ToString('o'); stopped_at = $null; stop_reason = $null
                container_endpoint = "http://${bind}:$Port/v1/chat/completions" }
    Write-State $state
    $gateway = Start-Process -FilePath (Get-Process -Id $PID).Path -WindowStyle Hidden -PassThru `
        -ArgumentList '-NoProfile', '-NonInteractive', '-File', $PSCommandPath, 'gateway' `
        -RedirectStandardOutput (Join-Path $StateDir 'gateway.out.log') -RedirectStandardError (Join-Path $StateDir 'gateway.err.log')
    $state.gateway_pid = $gateway.Id
    Write-State $state

    $deadline = [DateTime]::UtcNow.AddSeconds(30)
    do {
        Start-Sleep -Milliseconds 500
        $ports = @(Get-Listeners | ForEach-Object Port)
    } until (($ports -contains $OllamaPort -and $ports -contains $Port) -or [DateTime]::UtcNow -gt $deadline)
    if (-not ($ports -contains $OllamaPort -and $ports -contains $Port)) {
        Stop-All $state 'start_failed'
        throw "Start failed (listening: $($ports -join ', ')). See $StateDir\ollama.log and gateway.err.log."
    }
    Write-Output "Ollama 127.0.0.1:$OllamaPort (pid $($ollama.Id), OLLAMA_IGPU_ENABLE=$env:OLLAMA_IGPU_ENABLE)"
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
    $alive = @($state.ollama_pid, $state.gateway_pid) | Where-Object { $_ } |
        Where-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue }
    $runners = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^(ollama|llama-server)' -and $_.ParentProcessId -in @($state.ollama_pid) }
    $listeners = @(Get-Listeners)
    foreach ($item in $listeners) { Write-Output "STILL LISTENING $($item.Address):$($item.Port) pid $($item.Pid)" }
    foreach ($id in $alive) { Write-Output "STILL RUNNING pid $id" }
    foreach ($item in $runners) { Write-Output "STILL RUNNING runner pid $($item.ProcessId)" }
    if ($listeners -or $alive -or $runners) { exit 1 }
    Write-Output "Stopped: no gateway or Ollama process remains; nothing listens on $OllamaPort or $Port."
}

function Show-Status {
    $state = Read-State
    if (-not $state) { Write-Output 'No gateway state (never started).' }
    else {
        foreach ($name in 'ollama_pid', 'gateway_pid') {
            $running = $state[$name] -and (Get-Process -Id $state[$name] -ErrorAction SilentlyContinue)
            Write-Output ("{0} {1}: {2}" -f $name, $state[$name], $(if ($running) { 'running' } else { 'not running' }))
        }
        Write-Output "bind $($state.bind_address):$($state.gateway_port), idle $($state.idle_minutes) min, igpu $($state.igpu_enabled)"
        Write-Output "started $($state.started_at); stopped $($state.stopped_at) ($($state.stop_reason))"
        $log = Join-Path $StateDir 'gateway.log'
        if (Test-Path -LiteralPath $log) { Write-Output "last gateway log: $(Get-Content -LiteralPath $log -Tail 1)" }
    }
    $listeners = @(Get-Listeners)
    if (-not $listeners) { Write-Output "No listener on $OllamaPort or $Port." }
    foreach ($item in $listeners) { Write-Output "listening $($item.Address):$($item.Port) pid $($item.Pid)" }
}

switch ($Action) {
    'start' { Start-Serving }
    'stop' { Stop-Serving }
    'status' { Show-Status }
    'gateway' { Start-Gateway }
}
