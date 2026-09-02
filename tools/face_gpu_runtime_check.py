"""Audit GPU-image packages without downloading models or requiring a GPU."""

from __future__ import annotations

import importlib.metadata as metadata
import json
import subprocess
import sys

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
        raise RuntimeError(f"Unexpected dependency audit result ({returncode}): {output}")


def main() -> None:
    for package, version in EXPECTED_VERSIONS.items():
        actual = metadata.version(package)
        if actual != version:
            raise RuntimeError(f"{package}: expected {version}, got {actual}")
    for package in ("onnxruntime", "opencv-python", "opencv-contrib-python",
                    "opencv-contrib-python-headless", "onnxruntime-directml"):
        try:
            metadata.distribution(package)
        except metadata.PackageNotFoundError:
            continue
        raise RuntimeError(f"Conflicting distribution is installed: {package}")

    import cv2
    import insightface
    import numpy as np
    import onnxruntime as ort

    if ort.__version__ != "1.26.0" or "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("The imported ORT is not the expected CUDA-capable build")
    if cv2.__version__ != "4.14.0":
        raise RuntimeError(f"Unexpected imported OpenCV version: {cv2.__version__}")
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok or cv2.imdecode(encoded, cv2.IMREAD_COLOR).shape != frame.shape:
        raise RuntimeError("OpenCV JPEG encode/decode failed")
    if not hasattr(insightface.app, "FaceAnalysis"):
        raise RuntimeError("InsightFace FaceAnalysis could not be imported")

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


if __name__ == "__main__":
    main()
