#!/usr/bin/env bash
# Assemble an image_sequence media item into a clip at a given fps.
#
# Generalizes assemble_mot17.sh to any Benchmark V1 image_sequence (mot17, mot20,
# vscrowd). fps VARIES per sequence (MOT17-02=30, MOT17-05=14, MOT20=25) and is
# taken from the fetch manifest's assemble_fps column (derived from each clip's
# end_frame_number / end_sec). Feed it explicitly — a wrong fps shifts every
# "which second" answer for location/temporal. vscrowd carries no clip and is
# counting-only (fps-agnostic); 25 is a safe default there.
#
# Frame filename pattern is auto-detected (6-digit .jpg like MOT, else a plain
# glob) so mixed datasets work without per-dataset flags.
#
# Usage: ./assemble_frames.sh <img_dir> <out.mp4> [fps]
set -euo pipefail

IMG_DIR="${1:?usage: assemble_frames.sh <img_dir> <out.mp4> [fps]}"
OUT="${2:?usage: assemble_frames.sh <img_dir> <out.mp4> [fps]}"
FPS="${3:-25}"
DIR="${IMG_DIR%/}"

if ls "$DIR"/000001.jpg >/dev/null 2>&1; then
  # Zero-padded 6-digit, 1-based sequence (MOT convention).
  ffmpeg -y -framerate "$FPS" -start_number 1 -i "$DIR/%06d.jpg" \
    -c:v libx264 -pix_fmt yuv420p -r "$FPS" "$OUT"
else
  # Fallback: lexthan-sorted glob of jpg frames.
  ffmpeg -y -framerate "$FPS" -pattern_type glob -i "$DIR/*.jpg" \
    -c:v libx264 -pix_fmt yuv420p -r "$FPS" "$OUT"
fi

echo "wrote $OUT (${FPS} fps)"
