import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from eval_ten_clips import CLIPS, VIDEOS, probe
from low_resolution_model_eval import prepare_frame

rows = []
for clip in CLIPS:
    path = VIDEOS / clip["file"]
    meta = probe(path)
    for sample in clip["samples"]:
        started = time.perf_counter()
        jpeg, width, height, original, actual_ts, actual_index = prepare_frame(
            path, sample["timestamp"], 0, 75
        )
        elapsed = time.perf_counter() - started
        rows.append(
            {
                "clip_id": clip["id"],
                "file": clip["file"],
                "source": clip["source"],
                "scene": clip["scene"],
                "bytes": path.stat().st_size,
                **meta,
                "timestamp": sample["timestamp"],
                "actual_timestamp": actual_ts,
                "decode_encode_seconds": elapsed,
                "jpeg_bytes": len(jpeg),
                "width": width,
                "height": height,
                "expected": sample["expected"],
            }
        )
        print(clip["id"], sample["timestamp"], f"{elapsed:.3f}s", f"{len(jpeg)}B")

out = Path(__file__).with_name("person-eval-local.json")
out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
print("wrote", out, "n=", len(rows))
