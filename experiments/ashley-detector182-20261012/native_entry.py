"""Ashley detector-stream service entry for this isolated experiment.

The controller selects exactly one detector model and one question mode before
starting this process. This module performs configuration only; it does not
probe the worker, run a model, or start a test.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> None:
    model = os.environ.get("DETECTOR_MODEL", "").strip()
    endpoint = os.environ.get("DETECTOR_ENDPOINT", "").strip()
    geometry = os.environ.get("ASHLEY_GEOMETRY_ONLY", "")
    if model not in {"yolo26-s", "rfdetr-l"}:
        raise ValueError("DETECTOR_MODEL must be yolo26-s or rfdetr-l")
    if not endpoint.startswith(("http://", "https://")):
        raise ValueError("DETECTOR_ENDPOINT must be an HTTP(S) worker URL")
    if geometry not in {"0", "1"}:
        raise ValueError("ASHLEY_GEOMETRY_ONLY must be 0 or 1")
    os.environ["RICH_LABEL_BACKEND"] = "detector_stream"
    os.environ["RICH_LABEL_ENDPOINT"] = endpoint
    os.environ["RICH_LABEL_ENABLED"] = "true"
    os.environ["SCENE_ENGINE_MODE"] = "active"
    source = Path(__file__).resolve().parent / "source" / "src"
    os.environ.setdefault(
        "ASHLEY_EVENT_DEFINITIONS",
        str(source.parent / "event-definitions.json"),
    )
    sys.path.insert(0, str(source))
    # 本评测禁止独立HTTP连通性探针。连接池仍由首个真实窗口按原路径建立。
    from goodvision_harness.scene_model import TemporalVLMInspector
    TemporalVLMInspector.warmup = lambda self: False
    from goodvision_harness.native_event_audit import install
    install()
    from goodvision_harness.offline_benchmark import install_capture_hooks
    from goodvision_harness.service import HarnessRuntime
    install_capture_hooks(HarnessRuntime)
    from goodvision_harness.service import main as serve

    serve()


if __name__ == "__main__":
    main()
