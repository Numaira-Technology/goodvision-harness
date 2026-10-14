#!/usr/bin/env bash
# Publish a local video file to MediaMTX over RTSP, in real time (-re), looping.
# MediaMTX then serves it as WHEP for the harness.
#
# Start MediaMTX first (mediamtx ./mediamtx.yml), then run this per video.
#
# Usage: ./publish_rtsp.sh <video-file> <path-name>
#   ./publish_rtsp.sh fight.mpeg  benchA
#   ./publish_rtsp.sh mot17_02.mp4 benchB
#
# Harness then reads:  --stream-url http://localhost:8889/<path-name>/whep
set -euo pipefail

VIDEO="${1:?usage: publish_rtsp.sh <video-file> <path-name>}"
PATH_NAME="${2:?usage: publish_rtsp.sh <video-file> <path-name>}"
RTSP="rtsp://localhost:8554/${PATH_NAME}"

# -re: pace at real time so the harness sees a live-like cadence.
# -stream_loop -1: keep the path alive; drop it for a single pass.
# Re-encode to H.264 baseline for broad WebRTC compatibility.
exec ffmpeg -re -stream_loop -1 -i "$VIDEO" \
  -an -c:v libx264 -preset veryfast -tune zerolatency -pix_fmt yuv420p \
  -f rtsp -rtsp_transport tcp "$RTSP"
