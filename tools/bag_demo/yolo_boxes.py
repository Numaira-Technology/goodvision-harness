"""Local (Mac CPU/MPS) YOLO boxes for the 翻包 demo overlay.

Reuses the repo's frozen COCO weight `workers/detector_rich_label/yolo26s.pt` but widens
the class filter beyond person to the bag/contraband COCO classes. This is a standalone
demo helper and does NOT touch the benchmark's person-only detector worker.
"""

import json
import math
from pathlib import Path

from tools.bag_demo.media import duration as _duration
from tools.bag_demo.media import frame_jpeg

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
        # Ultralytics otherwise downloads missing weights implicitly.
        if not Path(weights).is_file():
            raise FileNotFoundError("YOLO weights must already exist locally")
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.conf = conf
        self.device = _device()

    def detect(self, image_bgr):
        """Return normalized xyxy (0..1), class name and detector confidence."""
        h, w = image_bgr.shape[:2]
        results = self.model.predict(
            image_bgr,
            classes=list(CLASSES),
            conf=self.conf,
            device=self.device,
            verbose=False,
        )
        out = []
        if not results:
            return out
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            name = CLASSES.get(cls_id, str(cls_id))
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            out.append(
                {
                    "cls": name,
                    "conf": round(float(box.conf[0]), 3),
                    "xyxy": [
                        round(x1 / w, 4),
                        round(y1 / h, 4),
                        round(x2 / w, 4),
                        round(y2 / h, 4),
                    ],
                    "alert": name in CONTRABAND,
                }
            )
        return out


def _frame_bgr_at(video_path, t):
    import cv2
    import numpy as np

    arr = np.frombuffer(frame_jpeg(video_path, t), dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("could not decode frame")
    return image


def precompute(video_path, fps=SAMPLE_FPS, conf=DEFAULT_CONF):
    """Per-timestamp boxes for the whole clip, for synced client-side overlay."""
    if not math.isfinite(fps) or not 0 < fps <= 60:
        raise ValueError("fps must be finite and within (0, 60]")
    if not math.isfinite(conf) or not 0 <= conf <= 1:
        raise ValueError("confidence must be finite and within 0..1")
    video_path = Path(video_path).resolve(strict=True)
    duration = _duration(video_path)
    detector = BagYolo(conf=conf)
    n = max(1, math.ceil(duration * fps))
    frames = []
    for i in range(n):
        t = min((i + 0.5) / fps, (i / fps + duration) / 2)
        frames.append(
            {"t": round(t, 3), "boxes": detector.detect(_frame_bgr_at(video_path, t))}
        )
    return {
        "clip": video_path.name,
        "duration": round(duration, 3),
        "fps": fps,
        "frames": frames,
    }


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(
        description="Precompute YOLO boxes for bag-demo clips."
    )
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument(
        "--out", type=Path, help="write boxes JSON here (default: <video>.boxes.json)"
    )
    ap.add_argument("--fps", type=float, default=SAMPLE_FPS)
    ap.add_argument("--conf", type=float, default=DEFAULT_CONF)
    args = ap.parse_args(argv)
    data = precompute(args.video, fps=args.fps, conf=args.conf)
    out = args.out or args.video.with_suffix(args.video.suffix + ".boxes.json")
    out.write_text(json.dumps(data))
    print(out)


if __name__ == "__main__":
    main()
