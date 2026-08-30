"""Run a real InsightFace CUDA warmup inside the GPU harness image."""

from __future__ import annotations

import os
from pathlib import Path

from goodvision_harness.face_recognition import (
    FaceRecognitionRuntime,
    FaceRecognizerConfig,
)


def main() -> None:
    model_root = Path(os.environ.get("FACE_MODEL_ROOT", "/models"))
    runtime = FaceRecognitionRuntime(
        FaceRecognizerConfig(
            model_name=os.environ.get("FACE_MODEL_NAME", "buffalo_l"),
            model_root=model_root,
            allow_model_download=os.environ.get(
                "FACE_ALLOW_MODEL_DOWNLOAD", "false"
            ).strip().lower()
            in {"1", "true", "yes", "on"},
            provider="CUDAExecutionProvider",
            det_size=640,
            gallery_path=Path("/tmp/face-gpu-smoke-gallery.json"),
        )
    )
    runtime.warmup()
    print("face_gpu_smoke=ready")


if __name__ == "__main__":
    main()
