#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "Usage: $0 HOST SSH_PORT [IDENTITY] [REMOTE_ROOT]" >&2
  exit 2
fi

HOST="$1"
PORT="$2"
IDENTITY="${3:-$HOME/.ssh/id_ed25519_vast}"
REMOTE_ROOT="${4:-/workspace/goodvision-4090-benchmark}"
if [[ ! "$HOST" =~ ^[A-Za-z0-9.-]+$ || ! "$PORT" =~ ^[0-9]+$ || ! "$REMOTE_ROOT" =~ ^/[-A-Za-z0-9_./]+$ ]]; then
  echo "Invalid host, port, or remote directory" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUTPUT="outputs/cloud-decode/$RUN_ID"
SSH_OPTS=(-o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=10 -i "$IDENTITY" -p "$PORT")
SCP_OPTS=(-q -o BatchMode=yes -o IdentitiesOnly=yes -i "$IDENTITY" -P "$PORT")
REMOTE="root@$HOST"
VIDEO="outputs/cloud-decode/input-h264.mp4"

test -f "$VIDEO"
curl --fail --silent --show-error http://127.0.0.1:18094/benchmark/config >/dev/null
LOCAL_HASHES="$(shasum -a 256 tools/cloud_cpu_tests/decode_location.py tools/evaluate_pipeline_latency.cjs)"
REMOTE_HASHES="$(ssh "${SSH_OPTS[@]}" "$REMOTE" "cd $REMOTE_ROOT && sha256sum tools/cloud_cpu_tests/decode_location.py tools/evaluate_pipeline_latency.cjs")"
if [[ "$LOCAL_HASHES" != "$REMOTE_HASHES" ]]; then
  echo "The cloud and Mac test scripts differ; deploy the current bundle before running" >&2
  exit 1
fi
LOCAL_CONTROLLER_HASH="$(shasum -a 256 src/goodvision_harness/camera_display.js | cut -d ' ' -f 1)"
SERVER_CONTROLLER_HASH="$(curl --fail --silent --show-error http://127.0.0.1:18094/assets/camera_display.js | shasum -a 256 | cut -d ' ' -f 1)"
if [[ "$LOCAL_CONTROLLER_HASH" != "$SERVER_CONTROLLER_HASH" ]]; then
  echo "The cloud browser controller differs from the Mac source; deploy the current bundle before running" >&2
  exit 1
fi
mkdir -p "$OUTPUT"

run_mac() {
  echo "Running $1 on the Mac browser"
  .venv-cloud-tests/bin/python -m tools.cloud_cpu_tests.decode_location run \
    --location mac --video "$VIDEO" --output "$OUTPUT/$1" --delay 1 \
    --upload-window 4 --ack-timeout-ms 15000 --drain-timeout-ms 30000
}

run_cloud() {
  echo "Running $1 on the cloud browser"
  ssh "${SSH_OPTS[@]}" "$REMOTE" \
    "cd $REMOTE_ROOT && . /opt/nvm/nvm.sh && .venv-harness-benchmark/bin/python -m tools.cloud_cpu_tests.decode_location run --location cloud --video video_samples/decode_input.mp4 --output outputs/cloud-decode/$RUN_ID/$1 --delay 1 --upload-window 4 --ack-timeout-ms 15000 --drain-timeout-ms 30000"
  scp "${SCP_OPTS[@]}" -r \
    "$REMOTE:$REMOTE_ROOT/outputs/cloud-decode/$RUN_ID/$1" "$OUTPUT/"
}

run_mac mac-1
sleep 10
run_cloud cloud-1
sleep 10
run_cloud cloud-2
sleep 10
run_mac mac-2

.venv-cloud-tests/bin/python -m tools.cloud_cpu_tests.decode_location compare \
  --mac "$OUTPUT/mac-1" --cloud "$OUTPUT/cloud-1" \
  --mac "$OUTPUT/mac-2" --cloud "$OUTPUT/cloud-2" \
  --output "$OUTPUT/comparison.md"

echo "Comparison: $ROOT/$OUTPUT/comparison.md"
