"""Local (Mac CPU/MPS) YOLO boxes for the 翻包 demo overlay.

Reuses the repo's frozen COCO weight `workers/detector_rich_label/yolo26s.pt` but widens
the class filter beyond person to the bag/contraband COCO classes. This is a standalone
demo helper and does NOT touch the benchmark's person-only detector worker.
"""

import base64
import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = REPO_ROOT / "workers" / "detector_rich_label" / "yolo26s.pt"

# COCO class ids we care about for a bag search. person kept for context.
CLASSES = {
    0: "person",
    24: "backpack",
    26: "handbag",
    28: "suitcase",
    39: "bottle",
    40: "wine glass",
    43: "knife",
}
CONTRABAND = {"bottle", "wine glass", "knife"}  # drawn as alerts
DEFAULT_CONF = 0.25
SAMPLE_FPS = 5.0  # overlay boxes sampling rate (local, cheap)


def _device():
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


class BagYolo:
    def __init__(self, weights=WEIGHTS, conf=DEFAULT_CONF):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.conf = conf
        self.device = _device()

    def detect(self, image_bgr):
        """Return boxes as dicts with normalized xyxy (0..1), class name and confidence."""
        h, w = image_bgr.shape[:2]
        results = self.model.predict(
            image_bgr, classes=list(CLASSES), conf=self.conf,
            device=self.device, verbose=False,
        )
        out = []
        if not results:
            return out
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            name = CLASSES.get(cls_id, str(cls_id))
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            out.append({
                "cls": name,
                "conf": round(float(box.conf[0]), 3),
                "xyxy": [round(x1 / w, 4), round(y1 / h, 4),
                         round(x2 / w, 4), round(y2 / h, 4)],
                "alert": name in CONTRABAND,
            })
        return out


def _frame_bgr_at(video_path, t):
    import cv2
    import numpy as np

    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-ss", f"{t:.3f}", "-i", str(video_path),
         "-vframes", "1", "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1"],
        capture_output=True, check=True,
    )
    arr = np.frombuffer(proc.stdout, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _duration(video_path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nokey=1:noprint_wrappers=1", str(video_path)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    return float(out)


def precompute(video_path, fps=SAMPLE_FPS, conf=DEFAULT_CONF):
    """Per-timestamp boxes for the whole clip, for synced client-side overlay."""
    video_path = Path(video_path)
    detector = BagYolo(conf=conf)
    duration = _duration(video_path)
    n = max(1, int(duration * fps))
    frames = []
    for i in range(n):
        t = (i + 0.5) / fps
        if t >= duration:
            break
        frames.append({"t": round(t, 3), "boxes": detector.detect(_frame_bgr_at(video_path, t))})
    return {"clip": video_path.name, "duration": round(duration, 3), "fps": fps, "frames": frames}


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="Precompute YOLO boxes for bag-demo clips.")
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--out", type=Path, help="write boxes JSON here (default: <video>.boxes.json)")
    ap.add_argument("--fps", type=float, default=SAMPLE_FPS)
    ap.add_argument("--conf", type=float, default=DEFAULT_CONF)
    args = ap.parse_args(argv)
    data = precompute(args.video, fps=args.fps, conf=args.conf)
    out = args.out or args.video.with_suffix(args.video.suffix + ".boxes.json")
    out.write_text(json.dumps(data))
    print(out)


if __name__ == "__main__":
    main()
