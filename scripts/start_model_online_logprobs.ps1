# Gemma 3 4B on llama.cpp for the online logprobs example. Runs in the foreground: Ctrl+C stops it and unloads the model.
$root = Split-Path -Parent $PSScriptRoot
$model = Join-Path $root 'models/blobs/sha256-f06e91e18fc74295d0c781bf91622508c75655e2c18ed869d52447ffe1e667e1'
$server = Join-Path $env:LOCALAPPDATA 'Microsoft/WinGet/Packages/ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe/llama-server.exe'
if (-not (Test-Path $model)) {
    'Model file missing. Gemma Terms of Use: https://ai.google.dev/gemma/terms . If you accept them, download it with:'
    "curl.exe -L -o `"$model`" https://registry.ollama.ai/v2/library/gemma3/blobs/sha256:f06e91e18fc74295d0c781bf91622508c75655e2c18ed869d52447ffe1e667e1"
    exit 2
}
if (-not (Test-Path $server)) { 'llama-server not found. Install it: winget install ggml.llamacpp'; exit 2 }
$key = (Select-String -Path (Join-Path $root '.env') -Pattern '^LOCAL_MODEL_API_KEY=(.+)$').Matches[0].Groups[1].Value
& $server -m $model --host 127.0.0.1 --port 11571 --api-key $key -ngl 99 -dev Vulkan0 -fa off -c 2048 -np 1 --swa-full --no-webui
