"""Pinned-package adapters. Imports and weight downloads happen only on demand."""

from pathlib import Path

import cv2
import numpy as np

MODELS = {
    **{
        f"rfdetr-{s}": ("rf", f"RFDETRSeg{n}", size)
        for s, n, size in [
            ("n", "Nano", 312),
            ("s", "Small", 384),
            ("m", "Medium", 432),
            ("l", "Large", 504),
        ]
    },
    **{
        f"{family}-{s}": ("yolo", f"{family}{s}-seg.pt", 640)
        for family in ("yolov8", "yolo26")
        for s in "nsml"
    },
}


def resolve_device(requested):
    import torch

    if requested == "auto":
        requested = (
            "cuda:0"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "Apple MPS GPU unavailable. Run doctor in a normal terminal."
        )
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable on this machine.")
    return requested


def synchronize(device):
    import torch

    if device == "mps":
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize(device)


class Adapter:
    def __init__(self, name, device, weights_dir, confidence=0.25, size=None):
        family, entry, default_size = MODELS[name]
        self.family, self.device = family, device
        self.size, self.confidence = size or default_size, confidence
        weights_dir = Path(weights_dir)
        weights_dir.mkdir(parents=True, exist_ok=True)
        if family == "yolo":
            if self.size % 32:
                raise ValueError("YOLO size must be a multiple of 32")
            from ultralytics import YOLO

            self.weight = weights_dir / entry
            self.model = YOLO(str(self.weight))
            if self.model.names.get(0) != "person":
                raise ValueError("Expected official COCO person class at index 0")
        else:
            import rfdetr
            from rfdetr import config as rf_config

            cls = getattr(rfdetr, entry)
            config = getattr(rf_config, entry + "Config")(device=device)
            divisor = config.patch_size * config.num_windows
            if self.size % divisor:
                raise ValueError(f"RF-DETR size must be a multiple of {divisor}")
            # RF_HOME is set before the library is imported by the CLI.
            self.model = cls(device=device, resolution=self.size)
            self.weight = Path(self.model.model_config.pretrain_weights)

    def predict(self, bgr):
        return self.predict_scored(bgr)[0]

    def predict_scored(self, bgr):
        """Return masks and their real detector confidences in matching order."""
        if self.family == "yolo":
            result = self.model.predict(
                source=bgr,
                device=self.device,
                imgsz=self.size,
                conf=self.confidence,
                classes=[0],
                retina_masks=True,
                rect=False,
                verbose=False,
                save=False,
                half=False,
            )[0]
            if result.masks is None:
                return np.zeros((0, *bgr.shape[:2]), dtype=bool), np.zeros(0)
            masks = result.masks.data.cpu().numpy() > 0.5
            scores = result.boxes.conf.cpu().numpy()
        else:
            result = self.model.predict(
                cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
                threshold=self.confidence,
                include_source_image=False,
            )
            if result.mask is None or not len(result):
                return np.zeros((0, *bgr.shape[:2]), dtype=bool), np.zeros(0)
            # RF's COCO IDs are sparse/1-based. Resolve names, never assume YOLO IDs.
            names = result.data.get("class_name")
            if names is None:
                raise ValueError("RF-DETR did not supply class names")
            selected = np.asarray(names) == "person"
            masks = result.mask[selected]
            scores = result.confidence[selected]
        if len(masks) and masks.shape[1:] != bgr.shape[:2]:
            raise ValueError("Model masks are not in source image coordinates")
        return np.asarray(masks, dtype=bool), np.asarray(scores, dtype=float)
