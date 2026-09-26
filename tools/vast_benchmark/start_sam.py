"""Start the bundled worker on an existing instance, with an explicit checkpoint."""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--worker-env",
        type=Path,
        default=ROOT / "tools/vast_benchmark/repository-worker-env.json",
    )
    parser.add_argument("--port", type=int, default=8092)
    args = parser.parse_args()
    checkpoint = args.checkpoint.expanduser().resolve(strict=True)
    if not checkpoint.is_file():
        parser.error("Checkpoint must be a regular file")
    if checkpoint.stat().st_size < 1_000_000:
        parser.error("Checkpoint is too small to be a SAM model")
    if not 1 <= args.port <= 65535:
        parser.error("Invalid port")
    settings = json.loads(args.worker_env.read_text())
    for key, value in settings.items():
        if not key.startswith("SAM") or not isinstance(value, str):
            parser.error("Worker config may contain only SAM-prefixed string values")
        os.environ[key] = value
    os.environ["SAM31_CHECKPOINT_PATH"] = str(checkpoint)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["PYTHONUNBUFFERED"] = "1"
    python = ROOT / ".venv-sam31-benchmark/bin/python"
    if not python.is_file():
        parser.error("Run tools/vast_benchmark/setup.sh sam first")
    os.chdir(ROOT / "workers/sam_scene")
    print("Starting SAM3.1 on loopback only. Config:", args.worker_env, flush=True)
    os.execv(
        str(python),
        [
            str(python),
            "-m",
            "uvicorn",
            "app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(args.port),
            "--workers",
            "1",
        ],
    )


if __name__ == "__main__":
    sys.exit(main())
