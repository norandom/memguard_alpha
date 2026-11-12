# Gemma 3 4B on the NPU through FastFlowLM (Windows only, logprobs only). Runs in the foreground: Ctrl+C stops it and unloads the model.
param([string]$Model = 'gemma3:4b')
$root = Split-Path -Parent $PSScriptRoot
$flm = (Get-Command flm -ErrorAction SilentlyContinue).Source
if (-not $flm) { $flm = (Get-ChildItem "$env:LOCALAPPDATA/Programs/FastFlowLM*/flm.exe" | Select-Object -Last 1).FullName }
if (-not $flm) { 'flm.exe not found. Install FastFlowLM.'; exit 2 }
$env:FLM_MODEL_PATH = Join-Path $root 'models/flm'
$env:FLM_DISABLE_UPDATE_CHECK = '1'
# If the model is not installed yet: flm pull <model>
& $flm serve $Model --host 127.0.0.1 -p 11540 --quiet -c 2048
