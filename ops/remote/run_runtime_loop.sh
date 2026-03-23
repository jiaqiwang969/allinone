#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-dell@192.168.1.104}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/dell/workspaces/allinone}"

ssh "$REMOTE_HOST" "
  cd '$REMOTE_ROOT' && \
  . .venv/bin/activate && \
  export PYTHONPATH='$REMOTE_ROOT/src' && \
  python3 -m allinone.interfaces.cli.main guidance-smoke && \
  python3 -m allinone.interfaces.cli.main language-smoke && \
  python3 -m allinone.interfaces.cli.main research-smoke
"
