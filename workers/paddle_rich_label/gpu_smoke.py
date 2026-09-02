"""Validate accelerated object and attribute inference before deployment."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any


def _run_checks() -> dict[str, Any]:
    # Resolve optional dependencies inside the public error boundary in main.
    import numpy as np
    import paddle
    from app import PaddleRichModels

    device = os.environ.get("RICH_DEVICE", "gpu:0").strip() or "gpu:0"
    if device != "gpu" and not (
        device.startswith("gpu:") and device.removeprefix("gpu:").isdigit()
    ):
        raise RuntimeError("accelerated device selection is invalid")
    if not paddle.device.is_compiled_with_cuda():
        raise RuntimeError("accelerated execution is unavailable")
    device_count = paddle.device.cuda.device_count()
    if device_count < 1:
        raise RuntimeError(
            "no accelerator is visible inside the container; check the host "
            "driver, container integration, and device allocation"
        )

    paddle.device.set_device(device)
    paddle.device.cuda.get_device_properties(device)
    paddle.version.cuda_archs()
    probe = paddle.ones([1], dtype="float32") + 1.0
    if float(probe.numpy()[0]) != 2.0:
        raise RuntimeError("accelerated calculation returned an unexpected result")
    paddle.utils.run_check()

    models = PaddleRichModels()
    # The standalone validation command must use the device just checked even
    # when the image did not set a worker-wide device default.
    models.device = device
    image = np.zeros((640, 640, 3), dtype=np.uint8)
    detections = models.detect(image)
    attributes = models.attributes((image,))
    if (
        len(attributes) != 1
        or not isinstance(attributes[0], Mapping)
        or not attributes[0]
    ):
        raise RuntimeError("pedestrian attribute validation failed")
    return {
        "ok": True,
        "acceleration": "ready",
        "object_detection": "ready",
        "pedestrian_attributes": "ready",
        "blank_detections": len(detections),
        "attribute_outputs": len(attributes),
    }


def main() -> int:
    try:
        result = _run_checks()
    except Exception:
        print(json.dumps({
            "ok": False,
            "error": "object and attribute engine validation failed",
        }, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
