#!/usr/bin/env bash
set -euo pipefail

REMOTE_ROOT="${REMOTE_ROOT:-/home/dell/workspaces/allinone}"
SERVICE_HOST="${SERVICE_HOST:-127.0.0.1}"
SERVICE_PORT="${SERVICE_PORT:-8001}"
PID_FILE="${PID_FILE:-$REMOTE_ROOT/qwen_service.pid}"
LOG_FILE="${LOG_FILE:-$REMOTE_ROOT/qwen_service.log}"
RECIPE_PATH="${ALLINONE_QWEN_GATEWAY_RECIPE:-$REMOTE_ROOT/configs/model_recipes/qwen_gateway.yaml}"

cd "$REMOTE_ROOT"

if [ -f "$PID_FILE" ]; then
  existing_pid="$(cat "$PID_FILE")"
  if [ -n "$existing_pid" ] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "qwen_service=already_running pid=$existing_pid log=$LOG_FILE recipe=$RECIPE_PATH"
    exit 0
  fi
fi

. .venv/bin/activate
export PYTHONPATH="$REMOTE_ROOT:$REMOTE_ROOT/src"
export ALLINONE_QWEN_GATEWAY_RECIPE="$RECIPE_PATH"

nohup python3 -m allinone.interfaces.cli.main serve-qwen \
  --host "$SERVICE_HOST" \
  --port "$SERVICE_PORT" \
  >"$LOG_FILE" 2>&1 &

service_pid="$!"
echo "$service_pid" > "$PID_FILE"
echo "qwen_service=started pid=$service_pid log=$LOG_FILE recipe=$RECIPE_PATH"
