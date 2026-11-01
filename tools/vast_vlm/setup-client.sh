#!/usr/bin/env bash
# Separate client environment: never replace the server's Torch/vLLM packages.
set -euo pipefail
if [[ "${1:-}" != "--execute" ]]; then
  echo "Plan: install a separate Linux Python 3.12 client and Chromium; no GPU rental."
  exit 0
fi
[[ "$(uname -s)" == "Linux" ]] || { echo "Linux required" >&2; exit 1; }
cd "$(dirname "$0")/../.."
python3.12 -m venv .venv-vast-client
.venv-vast-client/bin/python -m pip install -r tools/vlm_benchmark/requirements.txt -r tools/vlm_benchmark/monitor-requirements.txt
# Latest main adds websocket transport; the portable benchmark requirements
# predate that dependency. Match pyproject's >=15,<16 constraint explicitly.
.venv-vast-client/bin/python -m pip install websockets==15.0.1
.venv-vast-client/bin/python -m pip install --no-deps -e .
.venv-vast-client/bin/python -m pip check
.venv-vast-client/bin/python -m playwright install --with-deps chromium
apt-get install -y fonts-noto-cjk
.venv-vast-client/bin/python -m pip freeze > .venv-vast-client/installed.txt
