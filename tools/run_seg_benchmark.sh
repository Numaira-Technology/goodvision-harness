#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
if [ ! -x .venv-seg-benchmark/bin/python ]; then
  echo 'Run bash tools/setup_seg_benchmark.sh first.' >&2
  exit 2
fi
if [ "$#" -eq 0 ]; then
  set -- --help
fi
exec .venv-seg-benchmark/bin/python -m tools.seg_benchmark "$@"
