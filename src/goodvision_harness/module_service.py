"""Run an existing Bbox, System 1 or neural worker on any deployment host."""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import nullcontext
from pathlib import Path

from .module_services import module_directory, worker_lock
from .workflow_modules import LOCAL_PORTS


def serve(engine, host, port, directory):
    cache = Path(
        os.environ.get("GOODVISION_MODULE_MODEL_DIR", ".models/modules")
    ).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    directory.mkdir(parents=True, exist_ok=True)
    if engine == "bbox":
        # Reuse the detector worker and its session-isolated tracker unchanged.
        root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(root))
        os.environ["DETECTOR_MODEL"] = "yolo26-s"
        os.environ.setdefault("DETECTOR_WEIGHTS_DIR", str(cache / "yolo"))
        os.environ.setdefault("YOLO_CONFIG_DIR", str(directory / "ultralytics"))
        # Explicit host configuration wins; auto-selection is only for this
        # standalone lightweight module, not the legacy GPU-only workers.
        if "DETECTOR_DEVICE" not in os.environ:
            import torch

            os.environ["DETECTOR_DEVICE"] = (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        import uvicorn

        uvicorn.run(
            "workers.detector_rich_label.app:app", host=host, port=port, workers=1
        )
    elif engine == "system1":
        from .videojev_service import main

        sys.argv = [
            "goodvision-videojev",
            "--cache-dir",
            str(cache / "huggingface"),
            "--result-log",
            str(directory / "system1-results.jsonl"),
            "--host",
            host,
            "--port",
            str(port),
        ]
        main()
    else:
        from .neural_service import main

        sys.argv = [
            "goodvision-neural",
            "--cache-dir",
            str(cache / "huggingface"),
            "--host",
            host,
            "--port",
            str(port),
        ]
        main()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", choices=tuple(LOCAL_PORTS))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int)
    parser.add_argument("--data-dir", type=Path, default=module_directory())
    parser.add_argument("--singleton", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    port = args.port or LOCAL_PORTS[args.engine]
    if not 1 <= port <= 65535:
        parser.error("port must be within 1..65535")
    lock = (
        worker_lock(args.data_dir / f"{args.engine}.lock")
        if args.singleton
        else nullcontext(True)
    )
    with lock as acquired:
        if acquired:
            serve(args.engine, args.host, port, args.data_dir)


if __name__ == "__main__":
    main()
