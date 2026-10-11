#!/usr/bin/env python3
"""Offline tracking audit; reuses saved responses, never calls a paid model.

Known-identity synthetic crossings are regression tested separately. This tool
replays a real clip with original result arrival times and emits inspectable
video/JSON; its coverage is not an identity-accuracy score.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import av
import cv2
import numpy as np
from PIL import Image, ImageDraw

from tools.vlm_benchmark.preview_tracking import PreviewTracker, gray_frame
from tools.vlm_multistream_benchmark import decode


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--reports", type=Path, required=True)
    p.add_argument("--camera", type=int, default=1)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--sample-fps",
        type=float,
        default=2.5,
        help="Original benchmark sample rate (default observation 1.25 Hz)",
    )
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not math.isfinite(args.sample_fps) or args.sample_fps <= 0 or args.camera < 1:
        p.error("sample-fps must be positive and camera must be >= 1")
    samples = decode(args.video, args.sample_fps, 720, 90)
    seeds = []
    for line in args.reports.read_text(encoding="utf-8").splitlines():
        envelope = json.loads(line)
        if envelope["camera"] != args.camera:
            continue
        r = envelope["report"]
        if r.get("state") != "completed" or not r.get("analysis_complete"):
            continue
        tick = r["frame"]["source_frame_index"]
        if tick >= len(samples):
            continue
        at = samples.media_seconds[tick]
        seeds.append(
            {
                "id": r["message_id"],
                "time": at,
                "jpeg": samples[tick],
                "objects": r["objects"],
                "arrival": at + r["timing"]["source_to_publish_seconds"],
            }
        )
    seeds.sort(key=lambda s: s["arrival"])
    tracker = PreviewTracker()
    rows, contact, anchor = [], [], None
    index, due, snap = 0, 0.0, 0.0
    with (
        av.open(str(args.video)) as source,
        av.open(str(args.output / "tracking-replay.mp4"), "w") as writer,
    ):
        out = writer.add_stream("libx264", rate=15)
        out.width, out.height, out.pix_fmt = 960, 548, "yuv420p"
        out.options = {"crf": "23", "preset": "veryfast"}
        origin = float(
            (source.streams.video[0].start_time or 0)
            * source.streams.video[0].time_base
        )
        for frame in source.decode(video=0):
            at = max(0.0, frame.time - origin)
            if at + 1e-6 < due:
                continue
            due = (math.floor((at + 1e-6) * 15) + 1) / 15
            while index < len(seeds) and seeds[index]["arrival"] <= at:
                if anchor is None or seeds[index]["time"] > anchor["time"]:
                    anchor = seeds[index]
                index += 1
            rgb = frame.to_image()
            rgb.thumbnail((720, 410))
            bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
            jpeg = cv2.imencode(".jpg", bgr)[1].tobytes()
            result = tracker.advance(gray_frame(jpeg), at, anchor=anchor)
            rows.append(result)
            view = rgb.resize((960, 548))
            d = ImageDraw.Draw(view)
            d.rectangle((0, 0, 960, 34), fill="black")
            d.text(
                (10, 10),
                f"OFFLINE replay | {at:.2f}s | {result['status']} "
                f"| boxes={len(result['objects'])}",
                fill="white",
            )
            for obj in result["objects"]:
                b = obj["bbox"]
                coords = (b["x1"] * 960, b["y1"] * 548, b["x2"] * 960, b["y2"] * 548)
                d.rectangle(coords, outline="#57c7ff", width=3)
                number = int(obj["track_id"].rsplit(":", 1)[-1]) + 1
                d.text(coords[:2], f"T{number} estimate", fill="#57c7ff")
            for packet in out.encode(av.VideoFrame.from_image(view)):
                writer.mux(packet)
            if at >= snap:
                contact.append(view.copy().resize((480, 274)))
                snap += 2
        for packet in out.encode():
            writer.mux(packet)
    (args.output / "frames.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    times = [r["tracking_seconds"] for r in rows]
    active_times = [r["tracking_seconds"] for r in rows if r["objects"]]
    summary = {
        "kind": "offline replay with saved arrivals; not a new inference benchmark",
        "source": str(args.video),
        "reports": str(args.reports),
        "frames": len(rows),
        "frames_with_estimated_boxes": sum(bool(r["objects"]) for r in rows),
        "tracking_compute_p50_ms": float(np.percentile(times, 50) * 1000),
        "tracking_compute_p95_ms": float(np.percentile(times, 95) * 1000),
        "active_tracking_compute_p50_ms": float(np.percentile(active_times, 50) * 1000)
        if active_times
        else None,
        "active_tracking_compute_p95_ms": float(np.percentile(active_times, 95) * 1000)
        if active_times
        else None,
        "lost_tracks": rows[-1]["lost_tracks"] if rows else 0,
        "identity_accuracy": "not measured; VLM boxes are not independent ground truth",
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for page in range(math.ceil(len(contact) / 8)):
        sheet = Image.new("RGB", (960, 274 * 4), "black")
        for n, im in enumerate(contact[page * 8 : (page + 1) * 8]):
            sheet.paste(im, (n % 2 * 480, n // 2 * 274))
        sheet.save(args.output / f"contact-{page}.jpg")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
