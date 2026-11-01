#!/usr/bin/env bash
# Sample remote-SAM GPU pressure over SSH into a timestamped CSV.
#
# The harness runs on the Mac (no local GPU); SAM runs on a rented Vast GPU box.
# This streams nvidia-smi from that box to diag/logs/gpu-<stamp>.csv so
# parse_pipeline_log.py --gpu-csv can align GPU util/mem with the pipeline run.
#
# The GPU leg is the one resource the harness cannot see per stream (it only
# observes stage-4 HTTP latency). Run this in its own terminal for the whole run.
#
# Usage:
#   diag/gpu_sampler.sh root@ssh5.vast.ai 12345        # host then port
#   SSH_DEST=root@host SSH_PORT=12345 diag/gpu_sampler.sh
#   SSH_KEY=~/.ssh/id_ed25519_numaira diag/gpu_sampler.sh root@host 12345
#
# Get the SSH host/port for the current instance with:  vastai ssh-url <id>
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DEST="${1:-${SSH_DEST:-}}"
PORT="${2:-${SSH_PORT:-22}}"
KEY="${SSH_KEY:-}"
INTERVAL="${INTERVAL:-1}"   # seconds between samples

[ -n "$DEST" ] || { echo "!! usage: diag/gpu_sampler.sh <user@host> <port>  (or set SSH_DEST/SSH_PORT)" >&2; exit 1; }

mkdir -p diag/logs
STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="diag/logs/gpu-${STAMP}.csv"

SSH_OPTS=(-p "$PORT" -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15)
[ -n "$KEY" ] && SSH_OPTS+=(-i "$KEY")

QUERY="timestamp,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw"
REMOTE="nvidia-smi --query-gpu=${QUERY} --format=csv,nounits -l ${INTERVAL}"

echo ">> sampling GPU on ${DEST} (port ${PORT}) every ${INTERVAL}s -> ${OUT}"
echo ">> columns: ${QUERY}"
echo ">> Ctrl-C to stop."

# Write a header the parser tolerates, then stream remote nvidia-smi lines.
echo "$QUERY" > "$OUT"
# shellcheck disable=SC2029
exec ssh "${SSH_OPTS[@]}" "$DEST" "$REMOTE" | tee -a "$OUT"
