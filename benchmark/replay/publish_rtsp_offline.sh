#!/usr/bin/env bash
# Offline replay: publish a file to MediaMTX over RTSP, SINGLE PASS.
# The harness service accepts rtsp:// directly (no WHEP conversion needed), so
# this needs MediaMTX only as the RTSP hub, not the WebRTC layer.
#
#   Scheme B (offline): --stream-url rtsp://localhost:8554/<path-name>
#   Scheme A (live)   : use publish_rtsp.sh (adds -re + loop + WHEP via MediaMTX)
#
# Drop -re so frames are pushed as fast as the pipeline consumes them (faster
# than real time). No loop: one pass, then the path ends and the run drains.
#
# Usage: ./publish_rtsp_offline.sh <video-file> <path-name>
set -euo pipefail

VIDEO="${1:?usage: publish_rtsp_offline.sh <video-file> <path-name>}"
PATH_NAME="${2:?usage: publish_rtsp_offline.sh <video-file> <path-name>}"
RTSP="rtsp://localhost:8554/${PATH_NAME}"

exec ffmpeg -i "$VIDEO" \
  -an -c:v libx264 -preset veryfast -pix_fmt yuv420p \
  -f rtsp -rtsp_transport tcp "$RTSP"
