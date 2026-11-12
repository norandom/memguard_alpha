# Logits host for the online logits example (same Gemma 3 4B file). Runs in the foreground: Ctrl+C stops it and unloads the model.
# Stop the logprobs model first: only one of the two fits in memory.
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv/Scripts/python.exe'
if (-not (Test-Path $python)) { 'Python environment missing. Run: uv sync'; exit 2 }
# A model left loaded by an earlier start holds the ports or the memory: stop it.
Get-NetTCPConnection -State Listen -LocalPort 11570, 11571, 11580, 11581 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
Set-Location $root
& $python scripts/serve_full_distribution.py --port 11580
