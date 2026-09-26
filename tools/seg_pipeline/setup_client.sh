#!/usr/bin/env bash
# Run on the machine hosting Harness and Chrome (Mac is supported).
set -euo pipefail
BENCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$BENCH_ROOT"
BENCH_PYTHON="${BENCH_PYTHON:-python3.12}"
"$BENCH_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3,12)'
"$BENCH_PYTHON" -m venv .venv-harness-benchmark
.venv-harness-benchmark/bin/python -m pip install -e '.[video,vlm,face]'
.venv-harness-benchmark/bin/python -m pip install scipy==1.18.1
.venv-harness-benchmark/bin/python -m pip check
npm install --prefix .cache/seg-pipeline --save-exact playwright@1.62.1
echo 'Client ready. Use system Chrome for original HEVC playback.'
echo 'No benchmark, paid VLM request, or face-model download has been run.'
