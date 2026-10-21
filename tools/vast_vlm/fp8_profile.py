"""Separate, unvalidated 48 GiB Ada FP8 + YOLO deployment contract."""

from .profile import BASE_URL as BASE_URL
from .profile import ENGINE_VERSION as ENGINE_VERSION
from .profile import FROZEN as BF16_FROZEN
from .profile import IMAGE as IMAGE
from .profile import SERVED_MODEL as SERVED_MODEL
from .profile import policy_check as policy_check

MODEL_REPO = "Qwen/Qwen3.5-35B-A3B-FP8"
MODEL_REVISION = "9d1823d2dee688a6b25e77009dc727688c44936e"
VERSION = "vast-qwen35-fp8-yolo26s-v3-isolated"
QUANTIZATION = "fp8"
# Server batching is independent of the unchanged 48 client admission slots.
SERVER_SEQUENCES = 16
GPU_UTILIZATION = 0.85
YOLO_SHA256 = "646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b"
YOLO_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt"
DETECTOR_URL = "http://127.0.0.1:8092"


def validate_gpu(gpu):
    if (
        gpu.total_memory < 46 * 1024**3
        or "4090" not in gpu.name
        or (gpu.major, gpu.minor) != (8, 9)
    ):
        raise ValueError("requires actual >=46 GiB RTX 4090 Ada; no CPU offload")


FROZEN = {
    **BF16_FROZEN,
    "self_hosted_qwen35": False,
    "self_hosted_fp8_yolo": True,
    "model_revision": MODEL_REVISION,
    "quantization": "FP8",
    "person_ids_only": True,
    "backend_person_tracks": True,
    "monitor": True,
}


def validate(args):
    if getattr(args, "fp8_experiment", None) is not None:
        from .fp8_sweep import validate_arguments

        return validate_arguments(args)
    changed = [
        key for key, value in FROZEN.items() if getattr(args, key, None) != value
    ]
    if changed:
        raise ValueError("FP8 tracked contract changed: " + ", ".join(changed))
    policy_check()
