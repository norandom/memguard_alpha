#!/bin/sh
# Run the Modal operator tooling (deploy/modal, Python 3.12) inside a container.
# Usage (from WSL/Linux/macOS): sh scripts/modal_operator.sh plan | apps | inventory | lock | execute RECEIPT APPROVAL | stop APP_ID
#   | measure-plan PLAN MANIFEST CORPUS [APPROVAL] [--modal-manifest PATH]  (offline dry run of a measurement plan)
#   | execute-measurement PLAN MANIFEST CORPUS APPROVAL JOURNAL RECEIPT  (paid; measurement manifest)
#   | stage MANIFEST STAGE_LIMITS RECEIPT  (paid; CPU-only weight staging into the manifest's volume)
#   | serve MANIFEST STAGE_LIMITS RECEIPT  (paid; deploy, expiry at the stage limit, app left reachable)
#   | stop-serve MANIFEST RECEIPT  (stop the served app and its expiry controller, verified)
# plan, measure-plan and lock need no credentials; other modes pass the root .env to the container only.
# Credential values are whitespace-trimmed in the container; .env is never modified.
set -eu
REPO=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
MODE="$1"; shift
ENV_ARGS=""
case "$MODE" in
  plan|measure-plan|lock) ;;
  *) ENV_ARGS="--env-file $REPO/.env" ;;
esac
# shellcheck disable=SC2086
exec docker run --rm $ENV_ARGS \
  -v "$REPO:/app" -w /app \
  -v rg-modal-venv:/opt/modal-venv \
  -e UV_PROJECT_ENVIRONMENT=/opt/modal-venv -e UV_LINK_MODE=copy \
  ghcr.io/astral-sh/uv:0.9.7-python3.12-bookworm-slim \
  sh -c '
    set -eu
    for name in MODAL_TOKEN_ID MODAL_TOKEN_SECRET MODAL_PROXY_BEARER HF_TOKEN; do eval "value=\${$name:-}"; [ -z "$value" ] || export "$name=$(printf %s "$value" | tr -d "[:space:]")"; done
    [ "$0" = lock ] || uv sync --project deploy/modal --locked --quiet
    run() { uv run --project deploy/modal --no-sync python "$@"; }
    case "$0" in
      plan)
        run -I -c "import modal, transformers; print(\"modal\", modal.__version__, \"transformers\", transformers.__version__)"
        run scripts/probe_modal_model.py > /tmp/plan.json
        run -c "import json; d = json.load(open(\"/tmp/plan.json\")); print(d[\"mode\"], d[\"input_tokens\"], d[\"execution_blockers\"], round(d[\"resource_estimate_gross_usd\"], 3), d[\"inference_posts\"], d[\"manifest_sha256\"], d[\"prompts_sha256\"])"
        ;;
      execute)
        run scripts/probe_modal_model.py --execute --harmony-eligibility-only --receipt "$1" --approval "$2"
        ;;
      measure-plan)
        [ "$#" -ge 3 ] || { echo "measure-plan needs PLAN MANIFEST CORPUS [APPROVAL] [--modal-manifest PATH]" >&2; exit 2; }
        plan=$1 manifest=$2 corpus=$3; shift 3
        approval=""
        if [ "$#" -gt 0 ] && [ "$1" != --modal-manifest ]; then approval=$1; shift; fi
        modal_manifest=""
        if [ "$#" -gt 0 ]; then
          [ "$#" -eq 2 ] && [ "$1" = --modal-manifest ] || { echo "measure-plan takes only a trailing --modal-manifest PATH" >&2; exit 2; }
          modal_manifest=$2
        fi
        set -- --plan "$plan" --manifest "$manifest" --corpus "$corpus"
        [ -z "$approval" ] || set -- "$@" --approval "$approval"
        [ -z "$modal_manifest" ] || set -- "$@" --modal-manifest "$modal_manifest"
        run scripts/measure_modal_model.py "$@"
        ;;
      execute-measurement)
        [ "$#" -eq 6 ] || { echo "execute-measurement needs PLAN MANIFEST CORPUS APPROVAL JOURNAL RECEIPT" >&2; exit 2; }
        run scripts/measure_modal_model.py --execute --plan "$1" --manifest "$2" --corpus "$3"           --approval "$4" --journal "$5" --receipt "$6"
        ;;
      stage|serve)
        [ "$#" -eq 3 ] || { echo "$0 needs MANIFEST STAGE_LIMITS RECEIPT" >&2; exit 2; }
        run scripts/serve_modal_model.py "$0" --manifest "$1" --stage-limits "$2" --receipt "$3"
        ;;
      stop-serve)
        [ "$#" -eq 2 ] || { echo "stop-serve needs MANIFEST RECEIPT" >&2; exit 2; }
        run scripts/serve_modal_model.py stop-serve --manifest "$1" --receipt "$2"
        ;;
      apps)
        run -m modal app list --env recall-guard-pilot
        ;;
      inventory)
        run -m modal app list --env recall-guard-pilot
        run -m modal volume list --env recall-guard-pilot
        run -m modal dict list --env recall-guard-pilot
        ;;
      lock)
        uv lock --project deploy/modal
        ;;
      stop)
        run -m modal app stop "$1" --yes
        ;;
      *) echo "unknown mode $0" >&2; exit 2 ;;
    esac
  ' "$MODE" "$@"
