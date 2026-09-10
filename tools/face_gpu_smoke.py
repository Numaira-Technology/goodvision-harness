"""Check face runtime packages, or run the real accelerated model warmup."""

from __future__ import annotations

import importlib.metadata as metadata
import json
import os
import subprocess
import sys
from pathlib import Path

EXPECTED_VERSIONS = {
    "insightface": "1.0.1",
    "onnxruntime-gpu": "1.26.0",
    "opencv-python-headless": "4.14.0.94",
}
EXPECTED_METADATA_EXCEPTIONS = {
    "insightface 1.0.1 requires onnxruntime, which is not installed.",
    "insightface 1.0.1 requires opencv-python, which is not installed.",
}


def validate_dependency_audit(returncode: int, output: str) -> None:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if (
        returncode != 1
        or len(lines) != len(EXPECTED_METADATA_EXCEPTIONS)
        or set(lines) != EXPECTED_METADATA_EXCEPTIONS
    ):
        raise RuntimeError(
            f"Unexpected dependency audit result ({returncode}): {output}"
        )


def check_runtime() -> None:
    """Validate installed modules without creating model or device sessions."""

    for package, version in EXPECTED_VERSIONS.items():
        actual = metadata.version(package)
        if actual != version:
            raise RuntimeError(f"{package}: expected {version}, got {actual}")
    for package in (
        "onnxruntime", "opencv-python", "opencv-contrib-python",
        "opencv-contrib-python-headless", "onnxruntime-directml",
    ):
        try:
            metadata.distribution(package)
        except metadata.PackageNotFoundError:
            continue
        raise RuntimeError(f"Conflicting distribution is installed: {package}")

    import cv2
    import insightface
    import numpy as np
    import onnxruntime as ort

    if (
        ort.__version__ != "1.26.0"
        or "CUDAExecutionProvider" not in ort.get_available_providers()
    ):
        raise RuntimeError("The imported ORT is not the expected CUDA-capable build")
    if cv2.__version__ != "4.14.0":
        raise RuntimeError(f"Unexpected imported OpenCV version: {cv2.__version__}")
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok or cv2.imdecode(encoded, cv2.IMREAD_COLOR).shape != frame.shape:
        raise RuntimeError("OpenCV JPEG encode/decode failed")
    if not hasattr(insightface.app, "FaceAnalysis"):
        raise RuntimeError("InsightFace FaceAnalysis could not be imported")

    import httpx

    from goodvision_harness.overlay import _unicode_font, draw_overlay

    with httpx.Client(http2=True):
        pass
    font = _unicode_font(20)
    if bytes(font.getmask("中")) == bytes(font.getmask("\u0378")):
        raise RuntimeError("Chinese video font is missing its glyphs")
    draw_overlay(
        np.zeros((100, 320, 3), dtype=np.uint8), event_label="fall",
        detected=True, confidence=0.9, description="人员跌倒 / Person fell",
        objects=(),
    )

    result = subprocess.run(
        [sys.executable, "-m", "pip", "--no-cache-dir", "check"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    validate_dependency_audit(result.returncode, result.stdout + result.stderr)
    print(json.dumps({
        "runtime_package_audit": "passed_with_declared_metadata_exceptions",
        "versions": EXPECTED_VERSIONS,
        "compiled_providers": ort.get_available_providers(),
        "metadata_exceptions": sorted(EXPECTED_METADATA_EXCEPTIONS),
        "real_gpu_inference_tested": False,
    }, sort_keys=True))


def check_models() -> None:
    from goodvision_harness.face_recognition import (
        FaceRecognitionRuntime,
        FaceRecognizerConfig,
    )

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


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args not in ([], ["--runtime"]):
        print("ERROR invalid face validation request")
        return 2
    try:
        if args:
            check_runtime()
        else:
            check_models()
    except Exception:
        print("ERROR face validation failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
