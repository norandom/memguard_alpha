#!/usr/bin/env bash
# Gemma 3 4B on llama.cpp for the online logprobs example. Runs in the foreground: Ctrl+C stops it and unloads the model.
set -eu
root="$(cd "$(dirname "$0")/.." && pwd)"
digest=f06e91e18fc74295d0c781bf91622508c75655e2c18ed869d52447ffe1e667e1
model="$root/models/blobs/sha256-$digest"
if [ ! -f "$model" ]; then
  echo "Model file missing. Gemma Terms of Use: https://ai.google.dev/gemma/terms . If you accept them, download it with:"
  echo "curl -L -o \"$model\" https://registry.ollama.ai/v2/library/gemma3/blobs/sha256:$digest"
  exit 2
fi
command -v llama-server >/dev/null || { echo "llama-server not found. Install llama.cpp and put llama-server on PATH."; exit 2; }
key="$(sed -n 's/^LOCAL_MODEL_API_KEY=//p' "$root/.env" | tr -d '\r')"
exec llama-server -m "$model" --host 127.0.0.1 --port 11571 --api-key "$key" -ngl 99 -fa off -c 2048 -np 1 --swa-full --no-webui
