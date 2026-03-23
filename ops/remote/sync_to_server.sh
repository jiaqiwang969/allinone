#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-dell@192.168.1.104}"
REMOTE_ROOT="${REMOTE_ROOT:-/media/dell/eef6db82-afa8-4850-96b0-fdad3cf7025d/workspaces/allinone}"

rsync -av --delete \
  --exclude '.git' \
  --exclude '.pytest_cache' \
  --exclude '__pycache__' \
  ./ "$REMOTE_HOST:$REMOTE_ROOT/"
