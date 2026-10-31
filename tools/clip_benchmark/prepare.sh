#!/usr/bin/env bash
set -euo pipefail
task_root="$1"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_CACHE_DIR="$task_root/cache/pip"
mkdir -p "$PIP_CACHE_DIR" "$task_root/models/huggingface"
python3 -m venv --system-site-packages "$task_root/env"
"$task_root/env/bin/python" -m pip install \
  transformers==5.14.1 pillow==11.3.0 numpy==2.2.6 \
  opencv-python-headless==4.12.0.88
"$task_root/env/bin/python" -m pip list --format=json > "$task_root/installed-packages.json"

