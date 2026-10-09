#!/usr/bin/env bash
# Run on the GPU server, except for the offline plan/package commands.
set -euo pipefail
VLM_MODEL_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$VLM_MODEL_REPO"
if command -v python3.12 >/dev/null 2>&1; then
  exec python3.12 -m tools.vlm_benchmark.local_model "$@"
fi
exec python3 -m tools.vlm_benchmark.local_model "$@"
