"""Pluggable single-frame person detectors for the rich-label worker.

Each detector takes a BGR image and returns pixel-space person boxes as
tracker.Detection. The IoU tracker then assigns stable track_ids, so these
detectors need no temporal state of their own. Selected at startup by the
DETECTOR_MODEL env var (surfaced in the harness console):

    yolov8-m  -> YOLOv8-M   (accuracy pick; ultralytics)
    yolo26-s  -> YOLO26-S   (speed pick;    ultralytics)
    rfdetr-l  -> RF-DETR-L  (DETR pick;     rfdetr package)
    dummy     -> fixed box  (tests / no model deps)

Weights resolve from DETECTOR_WEIGHTS (explicit path) or a per-model default
filename under DETECTOR_WEIGHTS_DIR (default /models). COCO "person" is class 0.
"""

from __future__ import annotations

import os
from typing import Protocol

import numpy as np
from tracker import Detection, valid_pixel_bbox

# model key -> (engine, default weight filename)
MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "yolov8-m": ("yolo", "yolov8m.pt"),
    "yolo26-s": ("yolo", "yolo26s.pt"),
    "rfdetr-l": ("rfdetr", "rfdetr-l.pth"),
    "dummy": ("dummy", ""),
}

_PERSON = "person"


class Detector(Protocol):
    name: str
    def detect(self, image_bgr: np.ndarray) -> tuple[Detection, ...]: ...
    def warmup(self) -> None: ...
    @property
    def gpu_ready(self) -> bool: ...


def _weights_path(model_key: str, default_filename: str) -> str:
    explicit = os.environ.get("DETECTOR_WEIGHTS", "").strip()
    if explicit:
        return explicit
    base = os.environ.get("DETECTOR_WEIGHTS_DIR", "/models").strip() or "/models"
    return os.path.join(base, default_filename)


def _conf_threshold() -> float:
    try:
        return float(os.environ.get("DETECTOR_CONF", "0.35"))
    except ValueError:
        return 0.35


def _device() -> str:
    # "cuda"/"cuda:0"/"cpu"; ultralytics and torch both accept these.
    return os.environ.get("DETECTOR_DEVICE", "cuda").strip() or "cuda"


def build_detector(model_key: str) -> Detector:
    """Construct the detector selected by DETECTOR_MODEL / model_key."""
    key = (model_key or "").strip().lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"unknown DETECTOR_MODEL {model_key!r}; "
            f"choose one of {sorted(MODEL_REGISTRY)}"
        )
    engine, default_file = MODEL_REGISTRY[key]
    if engine == "dummy":
        return DummyDetector(name=key)
    if engine == "yolo":
        return YoloDetector(key, _weights_path(key, default_file), _device(),
                            _conf_threshold())
    if engine == "rfdetr":
        return RFDetrDetector(key, _weights_path(key, default_file), _device(),
                             _conf_threshold())
    raise ValueError(f"unsupported engine {engine!r}")


class DummyDetector:
    """Returns one fixed centre box; for contract tests without model deps."""

    def __init__(self, name: str = "dummy") -> None:
        self.name = name
        self._ready = True

    def detect(self, image_bgr: np.ndarray) -> tuple[Detection, ...]:
        h, w = image_bgr.shape[:2]
        return (
            Detection(_PERSON, 0.9, (w * 0.4, h * 0.3, w * 0.6, h * 0.9)),
        )

    def warmup(self) -> None:
        return None

    @property
    def gpu_ready(self) -> bool:
        return self._ready


class YoloDetector:
    """Ultralytics YOLO (v8 / 26) person detector."""

    def __init__(self, name: str, weights: str, device: str, conf: float) -> None:
        self.name = name
        self._weights = weights
        self._device = device
        self._conf = conf
        self._model = None
        self._gpu_ready = False

    def _ensure(self) -> None:
        if self._model is not None:
            return
        from ultralytics import YOLO  # lazy: only inside the worker image

        # ultralytics auto-downloads a known model NAME (e.g. "yolov8m.pt"); a
        # non-existent full path would just fail, so fall back to the bare
        # filename when the staged weights path is absent.
        weights = self._weights
        if not os.path.exists(weights):
            weights = os.path.basename(weights)
        self._model = YOLO(weights)

    def detect(self, image_bgr: np.ndarray) -> tuple[Detection, ...]:
        self._ensure()
        results = self._model.predict(
            image_bgr, imgsz=self._infer_size(), conf=self._conf,
            classes=[0], device=self._device, verbose=False,
        )
        h, w = image_bgr.shape[:2]
        out: list[Detection] = []
        first = next(iter(results), None)
        if first is None or first.boxes is None:
            return ()
        for box in first.boxes:
            xyxy = [float(v) for v in box.xyxy[0].tolist()]
            conf = float(box.conf[0])
            if conf < self._conf or not valid_pixel_bbox(xyxy, w, h):
                continue
            out.append(Detection(_PERSON, conf, tuple(xyxy)))
        return tuple(out)

    @staticmethod
    def _infer_size() -> int:
        try:
            return int(os.environ.get("DETECTOR_IMGSZ", "640"))
        except ValueError:
            return 640

    def warmup(self) -> None:
        try:
            self.detect(np.zeros((self._infer_size(), self._infer_size(), 3),
                                 dtype=np.uint8))
            self._gpu_ready = "cuda" in self._device
        except Exception:
            self._gpu_ready = False

    @property
    def gpu_ready(self) -> bool:
        return self._gpu_ready


class RFDetrDetector:
    """RF-DETR person detector (rfdetr package)."""

    def __init__(self, name: str, weights: str, device: str, conf: float) -> None:
        self.name = name
        self._weights = weights
        self._device = device
        self._conf = conf
        self._model = None
        self._gpu_ready = False

    def _ensure(self) -> None:
        if self._model is not None:
            return
        from rfdetr import RFDETRLarge  # lazy

        # Use staged weights when present, else let rfdetr fetch its pretrained
        # COCO checkpoint (RFDETRLarge() with no pretrain_weights).
        if os.path.exists(self._weights):
            self._model = RFDETRLarge(pretrain_weights=self._weights)
        else:
            self._model = RFDETRLarge()

    def detect(self, image_bgr: np.ndarray) -> tuple[Detection, ...]:
        self._ensure()
        # rfdetr predict wants RGB; returns a supervision.Detections-like object.
        # ascontiguousarray: the ::-1 view has a negative stride, which
        # torch.from_numpy rejects, so force a contiguous copy.
        image_rgb = np.ascontiguousarray(image_bgr[:, :, ::-1])
        det = self._model.predict(image_rgb, threshold=self._conf)
        h, w = image_bgr.shape[:2]
        out: list[Detection] = []
        xyxy = getattr(det, "xyxy", None)
        cls = getattr(det, "class_id", None)
        conf = getattr(det, "confidence", None)
        if xyxy is None:
            return ()
        for i in range(len(xyxy)):
            # COCO person is class 1 in RF-DETR's default label map.
            if cls is not None and int(cls[i]) != 1:
                continue
            box = [float(v) for v in xyxy[i]]
            score = float(conf[i]) if conf is not None else 1.0
            if score < self._conf or not valid_pixel_bbox(box, w, h):
                continue
            out.append(Detection(_PERSON, score, tuple(box)))
        return tuple(out)

    def warmup(self) -> None:
        try:
            self.detect(np.zeros((640, 640, 3), dtype=np.uint8))
            self._gpu_ready = "cuda" in self._device
        except Exception:
            self._gpu_ready = False

    @property
    def gpu_ready(self) -> bool:
        return self._gpu_ready
