#!/usr/bin/env bash
set -euo pipefail
TEST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$TEST_ROOT"
TEST_PYTHON="${BENCH_PYTHON:-python3.12}"
"$TEST_PYTHON" -m venv .venv-harness-benchmark
.venv-harness-benchmark/bin/python -m pip install -e '.[video,vlm,face]' psutil==7.2.2
.venv-harness-benchmark/bin/python -m pip check
echo 'Cloud CPU Harness installed. Start the real SAM worker separately.'
