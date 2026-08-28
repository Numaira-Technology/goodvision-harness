"""Exercise CUDA and both rich-label models before enabling a GPU deployment."""

from __future__ import annotations

import json
import os

import numpy as np
import paddle
from app import PaddleRichModels


def main() -> None:
    device = os.environ.get("RICH_DEVICE", "gpu:0").strip() or "gpu:0"
    if not paddle.device.is_compiled_with_cuda():
        raise RuntimeError("installed Paddle wheel was not compiled with CUDA")
    device_count = paddle.device.cuda.device_count()
    if device_count < 1:
        raise RuntimeError(
            "no NVIDIA GPU is visible inside the container; check the driver, "
            "NVIDIA Container Toolkit, and Compose GPU reservation"
        )

    paddle.device.set_device(device)
    properties = paddle.device.cuda.get_device_properties(device)
    probe = paddle.ones([1], dtype="float32") + 1.0
    if float(probe.numpy()[0]) != 2.0:
        raise RuntimeError("GPU tensor probe returned an unexpected result")
    paddle.utils.run_check()

    models = PaddleRichModels()
    image = np.zeros((640, 640, 3), dtype=np.uint8)
    detections = models.detect(image)
    attributes = models.attributes((image,))
    print(
        json.dumps(
            {
                "ok": True,
                "paddle_version": paddle.__version__,
                "device": device,
                "device_name": properties.name,
                "compute_capability": f"{properties.major}.{properties.minor}",
                "compiled_cuda_archs": paddle.version.cuda_archs(),
                "detector_model": models.detector_name,
                "attribute_model": models.attribute_name,
                "blank_detections": len(detections),
                "attribute_outputs": len(attributes),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
