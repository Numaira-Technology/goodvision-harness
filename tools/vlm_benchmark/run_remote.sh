#!/usr/bin/env bash
# Run this on the user's computer, not inside the rented container.
set -euo pipefail
VLM_BENCH_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
exec python3 "$VLM_BENCH_ROOT/tools/vlm_benchmark/run_remote.py" "$@"
