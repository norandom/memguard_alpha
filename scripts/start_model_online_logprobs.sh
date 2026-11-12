#!/usr/bin/env bash
# Start the local model for the online logprobs example: Gemma 3 4B on llama.cpp (Linux).
#
# For docs/notebooks/recall_adjusted_answers.ipynb. Three steps, each reported in one line:
#
# 1. model: the GGUF file the llamacpp-gemma3-4b profile pins must be under models/blobs. If it is
#    not, the script names the Gemma Terms of Use and stops; with --accept-model-terms it downloads
#    the file and keeps it only if its SHA-256 is the pinned one. The default source is the Ollama
#    registry, which serves the pinned file by its digest. With --hf-repo and --hf-file it downloads
#    from Hugging Face with HF_TOKEN (environment or root .env); the SHA-256 must still be the pinned
#    one.
# 2. runtime: llama-server must be on PATH (or --llama-server PATH) and report the pinned build.
#    --allow-other-build starts another build; recorded calibrators were fitted on the pinned one.
# 3. start: llama-server runs in the foreground on 127.0.0.1:11570 with the bearer
#    LOCAL_MODEL_API_KEY (root .env; generated there when missing, never printed). Ctrl+C stops it.
#
# Usage:
#   scripts/start_model_online_logprobs.sh            # check, then start
#   scripts/start_model_online_logprobs.sh check      # check only, start nothing
#   scripts/start_model_online_logprobs.sh start --accept-model-terms
#   scripts/start_model_online_logprobs.sh start -- -ngl 0     # extra llama-server arguments
#
# The logits host and the NPU route have no Linux script: the logits host starts the Windows build
# of llama-server, and FastFlowLM runs on Windows.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="$ROOT/recall_guard/core/provider_profiles/llamacpp-gemma3-4b.json"
GEMMA_TERMS="https://ai.google.dev/gemma/terms"
REGISTRY_PATH="library/gemma3"
PORT=11570
TODO=2

action="start"; accept=0; allow_other=0; hf_repo=""; hf_file=""; server="llama-server"; extra=()
if [[ $# -gt 0 && "$1" != -* ]]; then action="$1"; shift; fi
while [[ $# -gt 0 ]]; do
  case "$1" in
    --accept-model-terms) accept=1 ;;
    --allow-other-build) allow_other=1 ;;
    --hf-repo) hf_repo="${2:?--hf-repo needs a value}"; shift ;;
    --hf-file) hf_file="${2:?--hf-file needs a value}"; shift ;;
    --llama-server) server="${2:?--llama-server needs a path}"; shift ;;
    --) shift; extra=("$@"); break ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
  shift
done
case "$action" in start|check) ;; *) echo "unknown action: $action (start or check)" >&2; exit 1 ;; esac

todo() { printf '%s\n' "$@"; exit "$TODO"; }

if [[ "$ROOT" == "/" || ! -f "$ROOT/pyproject.toml" ]]; then
  echo "$ROOT is not the repository root." >&2; exit 1
fi

# A value by name from the environment, then from the root .env. Never echoed.
env_value() {
  local name="$1" value="${!1:-}"
  if [[ -z "$value" && -f "$ROOT/.env" ]]; then
    value="$(sed -n "s/^[[:space:]]*${name}[[:space:]]*=[[:space:]]*//p" "$ROOT/.env" | head -n 1 | tr -d '\r' \
             | sed -e 's/^"\(.*\)"$/\1/' -e "s/^'\(.*\)'\$/\1/")"
  fi
  printf '%s' "$value"
}

profile_field() {  # first occurrence of "name": "value" in the profile
  sed -n "s/.*\"$1\"[[:space:]]*:[[:space:]]*\"\([^\"]*\)\".*/\1/p" "$PROFILE" | head -n 1
}

digest="$(profile_field artifact_digest)"
runtime_version="$(profile_field runtime_version)"
model="$(profile_field model)"
if [[ ! "$digest" =~ ^sha256:[0-9a-f]{64}$ ]]; then echo "the profile's artifact digest is not a SHA-256: $digest" >&2; exit 1; fi
hex="${digest#sha256:}"
blobs="$ROOT/models/blobs"
target="$blobs/sha256-$hex"

# 1. model
if [[ -f "$target" ]]; then
  echo "model: $model is present (models/blobs/sha256-${hex:0:12}...)."
else
  if [[ $accept -ne 1 ]]; then
    todo "model: $model is not under models/blobs." \
         "It is distributed under the Gemma Terms of Use: $GEMMA_TERMS" \
         "Read them. To accept them and download the file (about 2.5 GiB), run: scripts/start_model_online_logprobs.sh $action --accept-model-terms"
  fi
  command -v curl >/dev/null || { echo "curl is needed for the download." >&2; exit 1; }
  mkdir -p "$blobs"
  partial="$target.partial"
  headers=()
  if [[ -n "$hf_repo" || -n "$hf_file" ]]; then
    [[ -n "$hf_repo" && -n "$hf_file" ]] || { echo "pass --hf-repo and --hf-file together." >&2; exit 1; }
    token="$(env_value HF_TOKEN)"
    [[ -n "$token" ]] || todo "model: HF_TOKEN is not set. Put a Hugging Face access token in the root .env as HF_TOKEN=..." \
                              "and accept the model's license on https://huggingface.co/$hf_repo first."
    headers=(-H "Authorization: Bearer $token")
    url="https://huggingface.co/$hf_repo/resolve/main/$hf_file"
    source_name="Hugging Face ($hf_repo)"
  else
    url="https://registry.ollama.ai/v2/$REGISTRY_PATH/blobs/sha256:$hex"
    source_name="the Ollama registry ($REGISTRY_PATH)"
  fi
  echo "model: downloading $model from $source_name ..."
  status="$(curl -L --fail-with-body -C - ${headers[@]+"${headers[@]}"} -o "$partial" -w '%{http_code}' "$url" || true)"
  if [[ "$status" == "401" || "$status" == "403" ]]; then
    rm -f "$partial"
    todo "model: the download was refused ($status)." \
         "Accept the model's license on its page and check HF_TOKEN in the root .env, then run this again."
  fi
  [[ "$status" == 2* ]] || { echo "the download failed (HTTP $status)." >&2; exit 1; }
  actual="$(sha256sum "$partial" | cut -d ' ' -f 1)"
  if [[ "$actual" != "$hex" ]]; then
    rm -f "$partial"
    echo "the downloaded file is not the pinned model: its SHA-256 is $actual, the profile pins $hex. Nothing was kept." >&2
    exit 1
  fi
  mv "$partial" "$target"
  echo "model: $model downloaded and verified."
fi

# 2. runtime
if ! command -v "$server" >/dev/null; then
  todo "runtime: llama-server was not found." \
       "Install llama.cpp $runtime_version (https://github.com/ggml-org/llama.cpp/releases) and put llama-server on PATH, or pass --llama-server PATH."
fi
version_text="$("$server" --version 2>&1 || true)"
build="$(printf '%s\n' "$version_text" | sed -n -e 's/.*(build \([0-9][0-9]*\).*/\1/p' -e 's/^version: \([0-9][0-9]*\) (.*/\1/p' | head -n 1)"
wanted="${runtime_version#b}"
if [[ "$build" != "$wanted" ]]; then
  if [[ $allow_other -ne 1 ]]; then
    todo "runtime: llama-server reports build ${build:-unknown}; the profile pins $runtime_version." \
         "Install that build, or pass --allow-other-build. Recorded calibrators were fitted on the pinned build."
  fi
  echo "runtime: llama-server build ${build:-unknown} (the profile pins $runtime_version; started on request)."
else
  echo "runtime: llama.cpp $runtime_version is present."
fi

if [[ "$action" == "check" ]]; then echo "check: model and runtime are ready. Nothing was started."; exit 0; fi

# 3. start
key="$(env_value LOCAL_MODEL_API_KEY)"
if [[ -z "$key" ]]; then
  command -v openssl >/dev/null || todo "start: LOCAL_MODEL_API_KEY is not set. Add LOCAL_MODEL_API_KEY=<a random token> to the root .env."
  key="$(openssl rand -hex 32)"
  printf '\nLOCAL_MODEL_API_KEY=%s\n' "$key" >> "$ROOT/.env"
  echo "start: LOCAL_MODEL_API_KEY was generated and appended to the root .env."
fi
echo "start: llama-server on 127.0.0.1:$PORT. When it reports that it is listening, run the \"Online, logprobs\" cells"
echo "of the notebook with LOCAL_MODEL_API_KEY set in that shell. Ctrl+C stops the host."
exec "$server" -m "$target" --host 127.0.0.1 --port "$PORT" -ngl 99 -fa off -c 2048 -np 1 --swa-full --no-webui \
     --api-key "$key" ${extra[@]+"${extra[@]}"}
