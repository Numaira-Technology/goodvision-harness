#!/usr/bin/env bash
# One-command phone->harness diagnostic run.
#
# Starts MediaMTX (if not already running), prints the link to open on your
# phone, runs the harness against that live stream for a fixed duration while
# capturing the receive-diagnostics log, then prints the summary table.
#
# Usage (run each combination for the 2x2 comparison):
#   diag/run_diag_test.sh                        # webrtc + full (default)
#   MODE=light diag/run_diag_test.sh             # webrtc + annotation off (dry-run)
#   TRANSPORT=rtmp diag/run_diag_test.sh         # rtmp + full
#   TRANSPORT=rtmp MODE=light diag/run_diag_test.sh
#
# Knobs (env):
#   TRANSPORT=webrtc|rtmp   MODE=full|light   SECS=45   PATH_NAME=phonecam
#   IP=<PC LAN ip>          PROVIDER=openrouter
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TRANSPORT="${TRANSPORT:-webrtc}"
MODE="${MODE:-full}"
SECS="${SECS:-45}"
PATH_NAME="${PATH_NAME:-phonecam}"
PROVIDER="${PROVIDER:-openrouter}"
PY="${PY:-$ROOT/.venv/bin/python}"
MTX_CONF="$ROOT/diag/mediamtx.yml"
LOG_DIR="$ROOT/diag/logs"
mkdir -p "$LOG_DIR"

# --- detect the PC's LAN IP (override with IP=...) ---
IP="${IP:-$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || true)}"
if [ -z "$IP" ]; then
  echo "!! Could not auto-detect LAN IP. Re-run with  IP=192.168.x.x  diag/run_diag_test.sh" >&2
  exit 1
fi

# --- prerequisites ---
command -v mediamtx >/dev/null 2>&1 || { echo "!! mediamtx not found. Install it (e.g. 'brew install mediamtx') then retry." >&2; exit 1; }
command -v ffmpeg   >/dev/null 2>&1 || echo "(note: ffmpeg not found — only needed if you also do file replay)"
command -v openssl  >/dev/null 2>&1 || { echo "!! openssl not found (needed to make the HTTPS cert)." >&2; exit 1; }
[ -x "$PY" ] || { echo "!! venv python not found at $PY (run: python3 -m venv .venv && .venv/bin/pip install -e .)" >&2; exit 1; }
if [ "$TRANSPORT" = "webrtc" ] && ! "$PY" -c "import aiortc, aiohttp" >/dev/null 2>&1; then
  echo "!! WebRTC deps missing. Install them:  $PY -m pip install -e \".[webrtc]\"" >&2
  echo "   (the harness silently drops the WHEP session if aiortc/aiohttp are absent)" >&2
  exit 1
fi

# --- self-signed cert for MediaMTX WebRTC HTTPS (phone camera needs a secure
#     context). SAN includes the LAN IP so the harness can verify it too. ---
CERT_DIR="$ROOT/diag/certs"
CERT="$CERT_DIR/server.crt"
KEY="$CERT_DIR/server.key"
mkdir -p "$CERT_DIR"
if [ ! -f "$CERT" ] || ! openssl x509 -in "$CERT" -noout -text 2>/dev/null | grep -q "IP Address:$IP"; then
  echo "Generating self-signed cert for $IP ..."
  openssl req -x509 -newkey rsa:2048 -nodes -keyout "$KEY" -out "$CERT" -days 365 \
    -subj "/CN=$IP" -addext "subjectAltName=IP:$IP,DNS:localhost" >/dev/null 2>&1
fi

# --- start MediaMTX if not already up ---
STARTED_MTX=""
if lsof -iTCP:8889 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "MediaMTX already running on :8889 — reusing it."
else
  echo "Starting MediaMTX ($MTX_CONF) ..."
  mediamtx "$MTX_CONF" > "$LOG_DIR/mediamtx.log" 2>&1 &
  STARTED_MTX=$!
  sleep 2
fi

cleanup() {
  [ -n "${LOAD_PID:-}" ] && kill "$LOAD_PID" 2>/dev/null || true
  [ -n "${HARNESS_PID:-}" ] && kill -INT "$HARNESS_PID" 2>/dev/null || true
  [ -n "$STARTED_MTX" ] && kill "$STARTED_MTX" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# --- build the stream URL + the phone instruction ---
if [ "$TRANSPORT" = "webrtc" ]; then
  STREAM_URL="webrtc+https://$IP:8889/$PATH_NAME/whep"
  PHONE_LINE="📱 On your phone (same WiFi), open in the browser:\n      https://$IP:8889/$PATH_NAME/publish\n      1) Accept the security warning (self-signed cert: Advanced -> Proceed)\n      2) In the CODEC dropdown pick VP8  (H264-from-browser fails aiortc depacketize; VP9 unsupported)\n      3) Allow the camera and tap Start/Publish."
elif [ "$TRANSPORT" = "rtmp" ]; then
  STREAM_URL="rtmp://$IP:1935/$PATH_NAME"
  PHONE_LINE="📱 On your phone, use an RTMP app (Larix/OBS) and push to:\n      rtmp://$IP:1935/$PATH_NAME"
else
  echo "!! TRANSPORT must be webrtc or rtmp" >&2; exit 1
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="$LOG_DIR/${TRANSPORT}_${MODE}_${STAMP}.log"

echo ""
echo "=================================================================="
echo "  Diagnostic run:  TRANSPORT=$TRANSPORT   MODE=$MODE   ${SECS}s"
echo "  Harness will pull:  $STREAM_URL"
echo "  Log -> $LOG"
echo "------------------------------------------------------------------"
echo -e "  $PHONE_LINE"
echo "=================================================================="
read -r -p "  Press Enter once the phone is publishing... " _

# --- harness env: MODE=light disables model calls (isolates decode/transport) ---
# The `service` entry does NOT accept --stream-url; it reads HARNESS_STREAM_URL.
# load_dotenv does not override an already-set env var, so this beats .env's value.
# SSL_CERT_FILE lets the harness's aiohttp WHEP client trust the self-signed cert.
HARNESS_ENV=(
  "INFERENCE_PROVIDER=$PROVIDER" "PYTHONPATH=$ROOT/src"
  "HARNESS_AUTOSTART=true" "SSL_CERT_FILE=$CERT"
  "HARNESS_STREAM_URL=$STREAM_URL"
)
if [ "$MODE" = "light" ]; then
  HARNESS_ENV+=( "HARNESS_DRY_RUN=true" )
else
  HARNESS_ENV+=( "HARNESS_DRY_RUN=false" )
fi

# SAM=on -> real full annotation via the remote Vast SAM (sam31_stream). Mirrors
# compose.sam31-remote.yaml, but points the endpoint at the published tunnel
# (start the tunnel first: docker compose -f compose.yaml -f compose.sam31-remote.yaml
#  -f diag/compose.sam-tunnel.yaml up --build sam-remote  -> localhost:8092).
if [ "${SAM:-off}" = "on" ]; then
  SAM_ENDPOINT="${SAM_ENDPOINT:-http://127.0.0.1:8092/v1/video/sessions}"
  echo "  [SAM] full annotation via remote SAM at $SAM_ENDPOINT"
  HARNESS_ENV+=(
    "SCENE_ENGINE_MODE=active" "PERSON_TRACKING_ENABLED=true"
    "RICH_LABEL_ENABLED=true" "RICH_LABEL_BACKEND=sam31_stream"
    "PERCEPTION_METHOD=sam_live"
    "RICH_LABEL_ENDPOINT=$SAM_ENDPOINT"
    "RICH_LABEL_MAX_FPS=${RICH_LABEL_MAX_FPS:-1.1}"
    "FACE_SAMPLE_FPS=${FACE_SAMPLE_FPS:-1.25}"
    "SAM31_BATCH_FRAMES=4" "SAM31_BATCH_SECONDS=0.5"
    "SAM31_CLIENT_TIMEOUT_SECONDS=75"
    "SAM31_TRANSPORT_CODEC=h264"
    "SAM31_TRANSPORT_MAX_SIDE=${SAM31_TRANSPORT_MAX_SIDE:-512}"
    "SAM31_TRANSPORT_CRF=${SAM31_TRANSPORT_CRF:-23}"
  )
fi

echo "Running harness for ${SECS}s (Ctrl-C to stop early)..."
env "${HARNESS_ENV[@]}" "$PY" -m goodvision_harness.service > "$LOG" 2>&1 &
HARNESS_PID=$!

# CRITICAL: HARNESS_AUTOSTART only connects the PREVIEW. Analysis (safety VLM +
# SAM + face) must be started explicitly via POST /v1/analysis/start, or the run
# only measures the idle preview path. Wait for the preview to connect, then start.
HARNESS_PORT="${HARNESS_PORT:-8080}"
echo "  waiting for preview to connect, then starting analysis..."
for _ in $(seq 1 20); do
  if curl -s -m 3 "http://127.0.0.1:${HARNESS_PORT}/v1/status" | grep -q '"connected": *true'; then break; fi
  sleep 1
done
START_RESP=$(curl -s -m 10 -X POST "http://127.0.0.1:${HARNESS_PORT}/v1/analysis/start" 2>&1)
echo "  analysis start -> $(echo "$START_RESP" | python3 -c "import sys,json;d=json.load(sys.stdin);print('pipeline_state='+str(d.get('pipeline_state')))" 2>/dev/null || echo "$START_RESP" | head -c 120)"

# LOAD=on saturates CPU cores to simulate full-annotation pressure (the Mac has
# no GPU, so real SAM/rich-label load is absent; this reproduces the contention).
if [ "${LOAD:-off}" = "on" ]; then
  LOAD_CORES="${LOAD_CORES:-$(( $(sysctl -n hw.ncpu) - 2 ))}"
  echo "  [LOAD] pinning $LOAD_CORES cores for ${SECS}s to simulate full annotation"
  "$PY" "$ROOT/diag/cpu_load.py" --cores "$LOAD_CORES" --seconds "$SECS" >/dev/null 2>&1 &
  LOAD_PID=$!
fi

sleep "$SECS"
kill -INT "$HARNESS_PID" 2>/dev/null || true
wait "$HARNESS_PID" 2>/dev/null || true
HARNESS_PID=""

echo ""
echo "=== Summary ($TRANSPORT / $MODE) ==="
"$PY" "$ROOT/diag/parse_receive_log.py" "$LOG" || {
  echo "(no network_receive_timing windows found — did the phone actually publish? check $LOG)"; }
echo ""
echo "Log saved: $LOG"
echo "Compare runs later with:"
echo "  $PY $ROOT/diag/parse_receive_log.py $LOG_DIR/webrtc_full_*.log $LOG_DIR/webrtc_light_*.log $LOG_DIR/rtmp_full_*.log"
