#!/usr/bin/env bash
# Assemble a MOT17 image_sequence into a 30 fps clip (media_type: image_sequence).
#
# The benchmark media row for crowd_mot17_mot17_02 is a folder of jpg frames
# (mot17/train/MOT17-02-FRCNN/img1/, 000001.jpg .. 000600.jpg), fps=30, 600
# frames, 20 s. how_to_feed.md requires assembling them at this fps before
# feeding, so "which second" maps as second = frame_number / 30.
#
# Usage: ./assemble_mot17.sh <img1_dir> <out.mp4>
set -euo pipefail

IMG_DIR="${1:?usage: assemble_mot17.sh <img1_dir> <out.mp4>}"
OUT="${2:?usage: assemble_mot17.sh <img1_dir> <out.mp4>}"

# MOT17 frames are 6-digit, 1-based. -start_number 1 is explicit for clarity.
ffmpeg -y \
  -framerate 30 -start_number 1 -i "${IMG_DIR%/}/%06d.jpg" \
  -c:v libx264 -pix_fmt yuv420p -r 30 \
  "$OUT"

echo "wrote $OUT (30 fps)"
