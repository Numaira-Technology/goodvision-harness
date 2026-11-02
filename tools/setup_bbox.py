"""Install YOLO in its own environment; never installs into the main service."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cu128"), default="cpu")
    args = parser.parse_args()
    environment = ROOT / ".venv-module-bbox"
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.exists():
        subprocess.run([sys.executable, "-m", "venv", str(environment)], check=True)
    subprocess.run([
        str(python), "-m", "pip", "install",
        f"torch==2.10.0+{args.device}", f"torchvision==0.25.0+{args.device}",
        "--index-url", "https://download.pytorch.org/whl/" + args.device,
    ], check=True)
    subprocess.run([
        str(python), "-m", "pip", "install", "-r",
        str(ROOT / "workers/bbox/requirements.txt"),
    ], check=True)
    subprocess.run([str(python), "-m", "pip", "check"], check=True)
    # Stage the official checkpoint at a fixed cache path, outside app source.
    models = ROOT / ".models/modules/yolo"
    models.mkdir(parents=True, exist_ok=True)
    settings = ROOT / ".goodvision/modules/ultralytics"
    settings.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(python), "-c",
                    "from ultralytics import YOLO; YOLO('yolo26s.pt')"],
                   cwd=models, env={**os.environ, "YOLO_CONFIG_DIR": str(settings)},
                   check=True)
    print(f"Bbox environment ready: {python}")


if __name__ == "__main__":
    main()
