#!/usr/bin/env bash
# Start MediaMTX PERSISTENTLY (foreground) for the phone->harness rig.
# Leave this running in its own terminal; the phone publishes to it and the
# harness / whep_probe pull from it. Ctrl-C stops it.
#
# Usage:  diag/start_mediamtx.sh        (auto-detects LAN IP for the cert)
#         IP=192.168.1.225 diag/start_mediamtx.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

IP="${IP:-$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || true)}"
[ -n "$IP" ] || { echo "!! Could not detect LAN IP. Re-run with IP=192.168.x.x" >&2; exit 1; }
command -v mediamtx >/dev/null 2>&1 || { echo "!! mediamtx not found ('brew install mediamtx')." >&2; exit 1; }
command -v openssl  >/dev/null 2>&1 || { echo "!! openssl not found." >&2; exit 1; }

CERT_DIR="$ROOT/diag/certs"; CERT="$CERT_DIR/server.crt"; KEY="$CERT_DIR/server.key"
mkdir -p "$CERT_DIR"
if [ ! -f "$CERT" ] || ! openssl x509 -in "$CERT" -noout -text 2>/dev/null | grep -q "IP Address:$IP"; then
  echo "Generating self-signed cert for $IP ..."
  openssl req -x509 -newkey rsa:2048 -nodes -keyout "$KEY" -out "$CERT" -days 365 \
    -subj "/CN=$IP" -addext "subjectAltName=IP:$IP,DNS:localhost" >/dev/null 2>&1
fi

if lsof -iTCP:8889 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "!! Something is already listening on :8889 (MediaMTX already up?). Stop it first." >&2
  exit 1
fi

echo "=================================================================="
echo "  MediaMTX starting (persistent). Keep this terminal open."
echo "  Phone WebRTC publish : https://$IP:8889/phonecam/publish  (pick VP8)"
echo "  Phone RTMP publish   : rtmp://$IP:1935/phonecam"
echo "  Harness/probe pull   : webrtc+https://$IP:8889/phonecam/whep"
echo "  Log also tee'd to    : diag/logs/mediamtx.log"
echo "=================================================================="
mkdir -p "$ROOT/diag/logs"
exec mediamtx "$ROOT/diag/mediamtx.yml" 2>&1 | tee "$ROOT/diag/logs/mediamtx.log"
