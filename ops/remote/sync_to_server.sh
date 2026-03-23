#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-dell@192.168.1.104}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/dell/workspaces/allinone}"

rsync -av --delete \
  --exclude '.git' \
  --exclude '.pytest_cache' \
  --exclude '__pycache__' \
  ./ "$REMOTE_HOST:$REMOTE_ROOT/"
