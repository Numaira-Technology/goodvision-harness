#!/usr/bin/env bash
set -euo pipefail
task_root="$1"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_CACHE_DIR="$task_root/cache/pip"
mkdir -p "$PIP_CACHE_DIR" "$task_root/models/huggingface"
test ! -e "$task_root/env" || { echo 'Refusing to change an existing environment' >&2; exit 1; }
python3 -m venv "$task_root/env"
# Pins reused from the repository's neural extra; verify on the approved 4090 before scoring.
"$task_root/env/bin/python" -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
"$task_root/env/bin/python" -m pip install \
  transformers==5.14.1 pillow==11.3.0 numpy==2.2.6 \
  opencv-python-headless==4.12.0.88
"$task_root/env/bin/python" -m pip check
"$task_root/env/bin/python" -m pip list --format=json > "$task_root/installed-packages.json"
