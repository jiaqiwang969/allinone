#!/usr/bin/env bash
set -euo pipefail

SERVICE_URL="${SERVICE_URL:-http://127.0.0.1:8001/health}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-20}"
SLEEP_SECONDS="${SLEEP_SECONDS:-3}"

attempt=1
while [ "$attempt" -le "$MAX_ATTEMPTS" ]; do
  if payload="$(curl --silent --show-error --fail "$SERVICE_URL")"; then
    if printf '%s' "$payload" | grep -q '"status"[[:space:]]*:[[:space:]]*"ready"'; then
      echo "$payload"
      exit 0
    fi
  fi
  sleep "$SLEEP_SECONDS"
  attempt=$((attempt + 1))
done

echo "qwen_service_status=unready url=$SERVICE_URL expected_status=ready"
exit 1
