#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-dell@192.168.1.104}"
REMOTE_ROOT="${REMOTE_ROOT:-/home/dell/workspaces/allinone}"

ssh "$REMOTE_HOST" "
  cd '$REMOTE_ROOT' && \
  . .venv/bin/activate && \
  python3 -m pip install --upgrade pip && \
  python3 -m pip install \
    torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu126 && \
  python3 -m pip install \
    transformers \
    accelerate \
    sentencepiece \
    safetensors
"
