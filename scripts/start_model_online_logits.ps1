# Logits host for the online logits example (same Gemma 3 4B file). Runs in the foreground: Ctrl+C stops it and unloads the model.
# Stop the logprobs model first: only one of the two fits in memory.
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv/Scripts/python.exe'
if (-not (Test-Path $python)) { 'Python environment missing. Run: uv sync'; exit 2 }
Set-Location $root
& $python scripts/serve_full_distribution.py --port 11580
