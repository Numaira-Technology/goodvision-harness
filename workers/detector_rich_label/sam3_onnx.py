"""Optional, offline SAM3 text detector using the pinned OSAM adapter.

Only the three text-inference graphs are loaded. Model files must be provisioned
separately under their canonical filenames, after reviewing the SAM License.
Nothing in this module downloads models or accepts license terms.
"""

from __future__ import annotations

import hashlib
import math
import os
import sys
import time
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np

if __package__:
    from .tracker import Detection, valid_pixel_bbox
else:
    from tracker import Detection, valid_pixel_bbox

OSAM_VERSION = "0.9.4"
SESSION_KEYS = frozenset({"image_encoder", "language_encoder", "decoder"})
PROVIDERS = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
}


def _local_model_type(base, blob_type, model_dir: Path, cache_dir: Path):
    """Isolate OSAM's download boundary without modifying global OSAM classes.

    These private adapter fields are intentionally pinned and covered by an
    installed-package contract test. Never call the upstream pull implementation.
    """
    class LocalBlob(blob_type):
        @property
        def path(self):
            return str(model_dir / self.filename)

        @property
        def coreml_cache_path(self):
            return str(cache_dir / self.hash.replace(":", "-"))

        def pull(self, *args, **kwargs):
            raise RuntimeError("SAM3 ONNX automatic downloads are disabled")

    blobs = {}
    for key in SESSION_KEYS:
        original = base._blobs[key]
        if original.attachments:
            raise RuntimeError("unsupported OSAM model layout; review the adapter")
        blobs[key] = LocalBlob(url=original.url, hash=original.hash)

    class LocalSam3(base):
        _blobs = blobs

        @classmethod
        def pull(cls, *args, **kwargs):
            # Check the complete bundle before hashing or opening an ORT session.
            for blob in cls._blobs.values():
                if not Path(blob.path).is_file():
                    raise RuntimeError(f"missing local SAM3 model: {blob.path}")
            for blob in cls._blobs.values():
                digest = hashlib.sha256()
                with open(blob.path, "rb") as source:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(chunk)
                if f"sha256:{digest.hexdigest()}" != blob.hash:
                    raise RuntimeError(f"SAM3 model SHA-256 mismatch: {blob.path}")

    return LocalSam3


def _load_model(model_dir: Path, cache_dir: Path, device: str):
    if version("osam") != OSAM_VERSION:
        raise RuntimeError(f"SAM3 ONNX requires osam=={OSAM_VERSION}")
    import osam

    base = osam.apis.get_model_type_by_name("sam3")
    local_type = _local_model_type(base, osam.types.Blob, model_dir, cache_dir)
    return local_type(device=device)


def _mask_polygons(mask, *, xmin: int, ymin: int, width: int, height: int):
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
    polygons = []
    remaining = 1024
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:16]:
        if remaining < 3 or cv2.contourArea(contour) <= 0:
            continue
        epsilon = 0.5
        points = cv2.approxPolyDP(contour, epsilon, True)
        while len(points) > min(128, remaining):
            epsilon *= 2
            points = cv2.approxPolyDP(contour, epsilon, True)
        if len(points) < 3 or cv2.contourArea(points) <= 0:
            continue
        polygon = tuple(
            ((float(x) + xmin) / width, (float(y) + ymin) / height)
            for x, y in points[:, 0]
        )
        polygons.append(polygon)
        remaining -= len(polygon)
    return tuple(polygons)


class Sam3OnnxDetector:
    name = "sam3-onnx"

    def __init__(self) -> None:
        self.device = os.environ.get("SAM3_ONNX_DEVICE", "cuda").strip().lower()
        self.model_dir = Path(
            os.environ.get("SAM3_ONNX_MODEL_DIR", ""),
        ).expanduser().resolve()
        self.cache_dir = Path(os.environ.get(
            "SAM3_ONNX_CACHE_DIR", str(self.model_dir / ".coreml"),
        )).expanduser().resolve()
        try:
            self.conf = float(os.environ.get("SAM3_ONNX_SCORE_THRESHOLD", "0.5"))
            self.max_annotations = int(os.environ.get("SAM3_ONNX_MAX_OBJECTS", "64"))
            masks = os.environ.get("SAM3_ONNX_MASKS", "true").strip().lower()
            if masks not in {"true", "false"}:
                raise ValueError("SAM3_ONNX_MASKS must be true or false")
            self.masks = masks == "true"
            if not os.environ.get("SAM3_ONNX_MODEL_DIR", "").strip():
                raise ValueError("SAM3_ONNX_MODEL_DIR must name a local directory")
            if self.device not in PROVIDERS:
                raise ValueError("SAM3_ONNX_DEVICE must be cpu, cuda or coreml")
            if self.device == "cuda" and sys.version_info < (3, 11):
                raise ValueError("SAM3 ONNX CUDA requires Python 3.11 or newer")
            if not math.isfinite(self.conf) or not 0 <= self.conf <= 1:
                raise ValueError("SAM3_ONNX_SCORE_THRESHOLD must be between 0 and 1")
            if not 1 <= self.max_annotations <= 256:
                raise ValueError("SAM3_ONNX_MAX_OBJECTS must be between 1 and 256")
        except ValueError as exc:
            raise RuntimeError(f"invalid SAM3 ONNX configuration: {exc}") from exc
        self._model = None
        self._ready = False
        self._error: str | None = None
        self._providers: dict[str, list[str]] = {}
        self._last_inference_ms: float | None = None

    def _check_providers(self) -> None:
        sessions = self._model._inference_sessions
        self._providers = {key: list(s.get_providers()) for key, s in sessions.items()}
        expected = PROVIDERS[self.device]
        if set(sessions) != SESSION_KEYS or any(
            not providers or providers[0] != expected
            for providers in self._providers.values()
        ):
            raise RuntimeError(f"SAM3 ONNX provider mismatch: expected {expected}")
        for session in sessions.values():
            session.disable_fallback()

    def detect(self, image_bgr: np.ndarray) -> tuple[Detection, ...]:
        started = time.monotonic()
        try:
            if self._model is None:
                self._model = _load_model(self.model_dir, self.cache_dir, self.device)
            self._check_providers()
            import osam

            # Use this instance directly; osam.apis.generate uses a global cache
            # and can load/download another model with automatic CPU fallback.
            response = self._model.generate(osam.types.GenerateRequest(
                model="sam3",
                image=np.ascontiguousarray(image_bgr[:, :, ::-1]),
                prompt=osam.types.Prompt(
                    texts=["person"], score_threshold=self.conf,
                    max_annotations=self.max_annotations,
                ),
            ))
            self._check_providers()
            detections = self._detections(response.annotations, image_bgr.shape[:2])
            self._ready = True
            self._error = None
            self._last_inference_ms = (time.monotonic() - started) * 1000
            return detections
        except Exception as exc:
            self._ready = False
            self._error = str(exc)
            raise RuntimeError(f"SAM3 ONNX inference unavailable: {exc}") from exc

    def _detections(self, annotations, shape) -> tuple[Detection, ...]:
        height, width = shape
        output = []
        for annotation in annotations[:self.max_annotations]:
            score = annotation.score
            bb = annotation.bounding_box
            if score is None or not math.isfinite(score) or not 0 <= score <= 1:
                raise RuntimeError("invalid SAM3 annotation confidence")
            if score < self.conf:
                continue
            if bb is None:
                raise RuntimeError("missing SAM3 annotation box")
            # OSAM coordinates are inclusive, while Harness boxes are half-open.
            box = (bb.xmin, bb.ymin, bb.xmax + 1, bb.ymax + 1)
            if not valid_pixel_bbox(box, width, height):
                raise RuntimeError("invalid SAM3 annotation box")
            polygons = ()
            if self.masks and annotation.mask is not None:
                mask = annotation.mask
                if (mask.dtype != np.bool_
                        or mask.shape != (box[3] - box[1], box[2] - box[0])):
                    raise RuntimeError("SAM3 mask does not match its source box")
                polygons = _mask_polygons(
                    mask, xmin=bb.xmin, ymin=bb.ymin, width=width, height=height,
                )
            output.append(Detection("person", float(score), box, polygons))
        return tuple(output)

    def warmup(self) -> None:
        self.detect(np.zeros((64, 64, 3), dtype=np.uint8))

    @property
    def gpu_ready(self) -> bool:
        # Preserve the legacy field's CUDA meaning; CoreML/CPU use provider_ready.
        return self._ready and self.device == "cuda"

    def health(self) -> dict:
        return {
            "model": self.name, "ready": self._ready,
            "gpu_ready": self.gpu_ready, "provider_ready": self._ready,
            "device": self.device, "execution_providers": self._providers,
            "error": self._error, "last_inference_ms": self._last_inference_ms,
        }
