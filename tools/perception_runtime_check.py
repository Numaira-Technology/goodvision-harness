"""Internal package audit; customer wrappers expose only its success/failure."""

from __future__ import annotations

import importlib.metadata as metadata
import subprocess
import sys


def main() -> None:
    for name, expected in {"paddlepaddle-gpu": "3.3.0", "paddlex": "3.7.2"}.items():
        if metadata.version(name) != expected:
            raise RuntimeError("unexpected perception runtime dependency version")
    try:
        metadata.distribution("paddlepaddle")
    except metadata.PackageNotFoundError:
        pass
    else:
        raise RuntimeError("conflicting perception runtime distributions")
    import cv2

    if not cv2.__version__.startswith("4.10."):
        raise RuntimeError("unexpected image runtime version")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "--no-cache-dir", "check"],
        capture_output=True, text=True, timeout=60, check=False,
    )
    if result.returncode or result.stderr.strip():
        raise RuntimeError("perception runtime dependency validation failed")
    print("perception runtime dependency validation passed")


if __name__ == "__main__":
    main()
