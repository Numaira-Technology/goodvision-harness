#!/usr/bin/env bash
# Realtime SINGLE-PASS replay: publish a file to MediaMTX over RTSP paced at the
# source frame rate (-re), exactly once (no -stream_loop).
#
# Why this and not publish_rtsp_offline.sh for scoring:
# StreamVideoSource timestamps RTSP frames by wall-clock (time.monotonic - origin)
# and drops excess frames at wall-clock cadence — it does NOT carry the source
# PTS. If ffmpeg pushes faster than realtime (the offline script, no -re), the
# capture timeline is COMPRESSED and tracks are subsampled, while questions still
# interpret seconds as SOURCE-video seconds (9-13s windows, event end times, count
# durations) -> wrong answers. Pacing at -re makes wall-clock ≈ source time, so
# the capture timeline aligns with the seconds the questions mean.
#
# Single pass (no loop): a timing question like "when does the first conflict end"
# needs one well-defined timeline; a looped stream repeats it.
#
# Verify alignment once: a known event/timestamp in the source should land at the
# same second in the capture (within a frame or two).
#
# Usage: ./publish_rtsp_realtime_once.sh <video-file> <path-name>
set -euo pipefail

VIDEO="${1:?usage: publish_rtsp_realtime_once.sh <video-file> <path-name>}"
PATH_NAME="${2:?usage: publish_rtsp_realtime_once.sh <video-file> <path-name>}"
RTSP="rtsp://localhost:8554/${PATH_NAME}"

# -re: pace at source rate so wall-clock timestamps track source seconds.
# No -stream_loop: single pass, then the path ends and the run drains.
exec ffmpeg -re -i "$VIDEO" \
  -an -c:v libx264 -preset veryfast -pix_fmt yuv420p \
  -f rtsp -rtsp_transport tcp "$RTSP"
