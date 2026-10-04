#!/usr/bin/env bash
# Multi-stream pressure rig: loop N local videos over RTSP into N harness
# processes, so parse_pipeline_log.py can show where the marginal cost of each
# added stream lands (CPU / GPU / bandwidth / which stage).
#
# One harness process = one stream (each on its own port, its own log). All
# processes share the one remote SAM endpoint (localhost:8092 tunnel), so GPU
# contention shows up there while host CPU + ingest bandwidth grow per process.
#
# Prereqs:
#   - mediamtx + ffmpeg on PATH; venv at .venv (pip install -e ".[webrtc,diag]")
#   - SAM=on: the remote SAM tunnel up at localhost:8092, and for a real N-lane
#     GPU curve the remote worker started with SAM31_LIVE_CAMERAS>=N
#     (tools/start_sam.py SAM31_LIVE_CAMERAS=N ...). Without that the worker
#     serves one lane and streams 2..N serialize/reject — which itself shows the
#     1-lane ceiling.
#
# Usage:
#   STREAMS=2 VIDEO=testABvideos/0057__...fight...mpeg SAM=off diag/run_multistream_diag.sh   # free local dry-run
#   STREAMS=4 VIDEO=<file> SAM=on SECS=90 diag/run_multistream_diag.sh                         # full GPU run
#
# Knobs (env): STREAMS=2  VIDEO=<file>  SECS=60  SAM=off|on  BASE_PORT=8080
#              PROVIDER=openrouter  RICH_LABEL_MAX_FPS=1.1  SAM31_TRANSPORT_MAX_SIDE=512
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STREAMS="${STREAMS:-2}"
VIDEO="${VIDEO:?set VIDEO=<path to a local video file to loop>}"
SECS="${SECS:-60}"
SAM="${SAM:-off}"
BASE_PORT="${BASE_PORT:-8080}"
PROVIDER="${PROVIDER:-openrouter}"
PY="${PY:-$ROOT/.venv/bin/python}"
MTX_CONF="$ROOT/diag/mediamtx-multistream.yml"
LOG_DIR="$ROOT/diag/logs"
STAMP="$(date +%Y%m%d-%H%M%S)"
mkdir -p "$LOG_DIR"

[ -f "$VIDEO" ] || { echo "!! VIDEO not found: $VIDEO" >&2; exit 1; }
command -v mediamtx >/dev/null 2>&1 || { echo "!! mediamtx not found ('brew install mediamtx')." >&2; exit 1; }
command -v ffmpeg >/dev/null 2>&1 || { echo "!! ffmpeg not found ('brew install ffmpeg')." >&2; exit 1; }
[ -x "$PY" ] || { echo "!! venv python not found at $PY" >&2; exit 1; }

PIDS=()
STARTED_MTX=""
cleanup() {
  echo ""; echo ">> stopping ${#PIDS[@]} processes..."
  for pid in "${PIDS[@]:-}"; do kill -INT "$pid" 2>/dev/null || true; done
  sleep 1
  for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  [ -n "$STARTED_MTX" ] && kill "$STARTED_MTX" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# --- MediaMTX (RTSP hub) ---
if lsof -iTCP:8554 -sTCP:LISTEN >/dev/null 2>&1; then
  echo ">> MediaMTX already on :8554 — reusing."
else
  echo ">> starting MediaMTX ($MTX_CONF)..."
  mediamtx "$MTX_CONF" > "$LOG_DIR/mediamtx-multistream-$STAMP.log" 2>&1 &
  STARTED_MTX=$!
  sleep 2
fi

# --- publish N looped RTSP streams ---
echo ">> publishing $STREAMS looped RTSP stream(s) from $(basename "$VIDEO")..."
for i in $(seq 1 "$STREAMS"); do
  ffmpeg -nostdin -loglevel error -stream_loop -1 -re -i "$VIDEO" \
    -an -c:v libx264 -preset veryfast -tune zerolatency -pix_fmt yuv420p -g 30 \
    -f rtsp -rtsp_transport tcp "rtsp://127.0.0.1:8554/stream$i" \
    > "$LOG_DIR/ffmpeg-stream$i-$STAMP.log" 2>&1 &
  PIDS+=($!)
done
sleep 3  # let the publishers establish

# --- SAM env (shared by every harness; the per-camera ENDPOINT is added per i) ---
# Each harness must target its OWN lane on the multi-lane worker:
#   http://<host>/v1/cameras/camera-0N/perception/sam_live/sessions
# PerceptionSessionHTTPClient (perception.py) parses that path and binds the
# session to lane camera-0N. Pointing every harness at the single default endpoint
# (/v1/perception/sam_live or /v1/video/sessions) makes them collide on one native
# session -> native_session_failed. SAM_BASE overrides the tunnel origin.
SAM_BASE="${SAM_BASE:-http://127.0.0.1:8092}"
sam_env() {
  if [ "$SAM" = "on" ]; then
    printf '%s\n' \
      "SCENE_ENGINE_MODE=active" "PERSON_TRACKING_ENABLED=true" \
      "RICH_LABEL_ENABLED=true" "RICH_LABEL_BACKEND=sam31_stream" \
      "PERCEPTION_METHOD=sam_live" \
      "RICH_LABEL_MAX_FPS=${RICH_LABEL_MAX_FPS:-1.1}" \
      "SAM31_BATCH_FRAMES=4" "SAM31_BATCH_SECONDS=0.5" \
      "SAM31_CLIENT_TIMEOUT_SECONDS=75" "SAM31_TRANSPORT_CODEC=h264" \
      "SAM31_TRANSPORT_MAX_SIDE=${SAM31_TRANSPORT_MAX_SIDE:-512}" \
      "SAM31_TRANSPORT_CRF=${SAM31_TRANSPORT_CRF:-23}"
  fi
}

# --- launch N harness processes, one per stream ---
echo ">> launching $STREAMS harness process(es) (SAM=$SAM)..."
declare -a HARNESS_PIDS=()
declare -a HARNESS_PORTS=()
declare -a HARNESS_LOGS=()
for i in $(seq 1 "$STREAMS"); do
  port=$(( BASE_PORT + i - 1 ))
  log="$LOG_DIR/multistream-n${STREAMS}-cam${i}-${STAMP}.log"
  cam=$(printf 'camera-%02d' "$i")
  endpoint_env=()
  if [ "$SAM" = "on" ]; then
    endpoint_env=( "RICH_LABEL_ENDPOINT=${SAM_BASE}/v1/cameras/${cam}/perception/sam_live/sessions" )
  fi
  env \
    "INFERENCE_PROVIDER=$PROVIDER" "PYTHONPATH=$ROOT/src" \
    "HARNESS_HOST=127.0.0.1" "HARNESS_PORT=$port" \
    "HARNESS_AUTOSTART=true" \
    "HARNESS_STREAM_URL=rtsp://127.0.0.1:8554/stream$i" \
    "RESOURCE_SAMPLE_SECONDS=${RESOURCE_SAMPLE_SECONDS:-3}" \
    $( [ "$SAM" = off ] && echo "HARNESS_DRY_RUN=false" ) \
    "${endpoint_env[@]}" \
    $(sam_env | tr '\n' ' ') \
    "$PY" -m goodvision_harness.service > "$log" 2>&1 &
  pid=$!
  PIDS+=("$pid"); HARNESS_PIDS+=("$pid"); HARNESS_PORTS+=("$port"); HARNESS_LOGS+=("$log")
  lane=$( [ "$SAM" = on ] && echo " lane=$cam" || echo "" )
  echo "   cam$i -> port $port  (rtsp://127.0.0.1:8554/stream$i)${lane}  log: $(basename "$log")"
done

# --- wait for each to connect its preview, then start analysis ---
echo ">> waiting for previews to connect, then starting analysis..."
for idx in "${!HARNESS_PORTS[@]}"; do
  port="${HARNESS_PORTS[$idx]}"
  for _ in $(seq 1 25); do
    curl -s -m 3 "http://127.0.0.1:${port}/v1/status" | grep -q '"connected": *true' && break
    sleep 1
  done
  resp=$(curl -s -m 10 -X POST "http://127.0.0.1:${port}/v1/sessions/start" \
    -H "Content-Type: application/json" \
    -d '{"mode":"stream","analysis_mode":"multi","features":["fight","smoking","fall","weapon"]}' 2>&1)
  echo "   cam$((idx+1)) (port $port) analysis start -> $(echo "$resp" | head -c 160)"
done

echo ">> running ${STREAMS} stream(s) for ${SECS}s ..."
sleep "$SECS"

# --- stop harnesses (keep ffmpeg/mediamtx until trap) ---
for pid in "${HARNESS_PIDS[@]}"; do kill -INT "$pid" 2>/dev/null || true; done
sleep 2
for pid in "${HARNESS_PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done

echo ""
echo "=== per-stream pressure summary (N=$STREAMS, SAM=$SAM) ==="
GPU_ARGS=()
[ -n "${GPU_CSV:-}" ] && [ -f "${GPU_CSV}" ] && GPU_ARGS=(--gpu-csv "$GPU_CSV")
"$PY" "$ROOT/diag/parse_pipeline_log.py" "${HARNESS_LOGS[@]}" "${GPU_ARGS[@]}" || true

echo ""
echo "Logs: ${HARNESS_LOGS[*]}"
echo "Compare stream counts later, e.g.:"
echo "  $PY diag/parse_pipeline_log.py $LOG_DIR/multistream-n1-*.log $LOG_DIR/multistream-n2-*.log $LOG_DIR/multistream-n4-*.log --gpu-csv diag/logs/gpu-<stamp>.csv"
