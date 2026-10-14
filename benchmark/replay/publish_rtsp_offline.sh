#!/usr/bin/env bash
# Offline replay: publish a file to MediaMTX over RTSP, SINGLE PASS, no pacing.
#
# ⚠️  NOT FOR TIME-SCORED QUESTIONS. Without -re, ffmpeg pushes faster than
# realtime; StreamVideoSource timestamps RTSP frames by wall-clock and drops
# excess frames by wall-clock cadence (it does not carry source PTS), so the
# capture timeline is compressed and tracks subsampled. Any question scored on
# source seconds — location time windows, temporal event end, count durations /
# count-at-time — will be WRONG. Use publish_rtsp_realtime_once.sh (-re, single
# pass) for those; the batch driver (run_batch.py) does.
#
# This fast path is only safe for fps-INDEPENDENT answers: counting peak
# (max simultaneous) and event yes/no. Even then, verify tracks are not so
# subsampled that the peak is missed.
#
# Usage: ./publish_rtsp_offline.sh <video-file> <path-name>
set -euo pipefail

VIDEO="${1:?usage: publish_rtsp_offline.sh <video-file> <path-name>}"
PATH_NAME="${2:?usage: publish_rtsp_offline.sh <video-file> <path-name>}"
RTSP="rtsp://localhost:8554/${PATH_NAME}"

exec ffmpeg -i "$VIDEO" \
  -an -c:v libx264 -preset veryfast -pix_fmt yuv420p \
  -f rtsp -rtsp_transport tcp "$RTSP"
