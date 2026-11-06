# Start the optional model host from this checkout's ignored model store.
param([string]$OllamaPath)

$ErrorActionPreference = 'Stop'
$repositoryRoot = Split-Path -Parent $PSScriptRoot
$env:OLLAMA_MODELS = (Resolve-Path -LiteralPath (Join-Path $repositoryRoot 'models')).Path
$env:OLLAMA_CONTEXT_LENGTH = '2048'
$env:OLLAMA_NUM_PARALLEL = '1'
$env:OLLAMA_MAX_LOADED_MODELS = '1'
$env:OLLAMA_NO_CLOUD = '1'

if (-not $OllamaPath) {
    $command = Get-Command ollama -ErrorAction SilentlyContinue
    if ($command) {
        $OllamaPath = $command.Source
    } else {
        $OllamaPath = Join-Path $env:LOCALAPPDATA 'AMD\AI_Bundle\Ollama\ollama.exe'
    }
}
if (-not (Test-Path -LiteralPath $OllamaPath -PathType Leaf)) {
    throw 'Ollama executable not found. Supply -OllamaPath with its installed path.'
}
& $OllamaPath serve
exit $LASTEXITCODE
