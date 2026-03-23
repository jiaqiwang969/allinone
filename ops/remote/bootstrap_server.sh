#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-dell@192.168.1.104}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/dell/workspaces/allinone}"

ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_ROOT'"
rsync -av --delete \
  --exclude '.git' \
  --exclude '.worktrees' \
  --exclude '.venv' \
  --exclude '.pytest_cache' \
  --exclude '__pycache__' \
  ./ "$REMOTE_HOST:$REMOTE_ROOT/"
ssh "$REMOTE_HOST" "
  cd '$REMOTE_ROOT' && \
  python3 -m venv --clear .venv && \
  . .venv/bin/activate && \
  pip install --upgrade pip && \
  pip install -e . && \
  bash ops/remote/start_qwen_service.sh && \
  bash ops/remote/check_qwen_service.sh
"
