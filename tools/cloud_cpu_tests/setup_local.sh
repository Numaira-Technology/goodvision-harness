#!/usr/bin/env bash
set -euo pipefail
TEST_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$TEST_ROOT"
TEST_PYTHON="${BENCH_PYTHON:-python3.12}"
"$TEST_PYTHON" -m venv .venv-cloud-tests
.venv-cloud-tests/bin/python -m pip install httpx==0.28.1 numpy==2.5.3 psutil==7.2.2
npm install --prefix .cache/seg-pipeline --save-exact playwright@1.62.1
.venv-cloud-tests/bin/python -m pip check
echo 'Local test runner installed. No models, paid API calls or screen recordings were started.'
