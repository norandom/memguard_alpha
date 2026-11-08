#!/bin/sh
# Run the Modal operator tooling (deploy/modal, Python 3.12) inside a container.
# Usage (from WSL/Linux/macOS): sh scripts/modal_operator.sh plan | apps | inventory | lock | execute RECEIPT APPROVAL | stop APP_ID
# plan and lock need no credentials; other modes pass the root .env to the container only.
# Credential values are whitespace-trimmed in the container; .env is never modified.
set -eu
REPO=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
MODE="$1"; shift
ENV_ARGS=""
case "$MODE" in
  plan|lock) ;;
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
    for name in MODAL_TOKEN_ID MODAL_TOKEN_SECRET MODAL_PROXY_BEARER; do eval "value=\${$name:-}"; [ -z "$value" ] || export "$name=$(printf %s "$value" | tr -d "[:space:]")"; done
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
