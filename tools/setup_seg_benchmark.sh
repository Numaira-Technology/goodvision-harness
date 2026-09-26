#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
export UV_CACHE_DIR="$PROJECT_DIR/.cache/uv"
if command -v uv >/dev/null 2>&1; then
  if [ ! -x .venv-seg-benchmark/bin/python ]; then
    uv venv --python 3.12 .venv-seg-benchmark
  fi
  uv pip install --python .venv-seg-benchmark/bin/python -r tools/seg_benchmark/requirements.txt
else
  python3 -m venv .venv-seg-benchmark
  .venv-seg-benchmark/bin/python -m pip install -r tools/seg_benchmark/requirements.txt
fi
.venv-seg-benchmark/bin/python -m tools.seg_benchmark doctor
echo 'Environment ready. See docs/guides/segmentation-benchmark.md.'
