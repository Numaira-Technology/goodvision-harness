#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" != "--execute" ]]; then
  echo 'Plan: isolated YOLO26-S CUDA environment; no install/download/rental.'
  exit 0
fi
[[ "$(uname -s)" == Linux ]] || exit 1
cd "$(dirname "$0")/../.."
python3.12 -m venv .venv-vast-yolo
.venv-vast-yolo/bin/python -m pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
.venv-vast-yolo/bin/python -m pip install ultralytics==8.4.181 fastapi==0.142.1 uvicorn==0.54.0 httpx==0.28.1
.venv-vast-yolo/bin/python -m pip check
.venv-vast-yolo/bin/python -m pip freeze > .venv-vast-yolo/installed.txt
