"""Build the shared CUDA layer and ABI-specific torch layers. No GPU rental."""

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSIONS = {"contour": ("2.10.0", "0.25.0", "2.10"),
            "bbox": ("2.10.0", "0.25.0", "2.10"),
            "system1": ("2.8.0", "0.23.0", "2.8")}


def build_commands(modules):
    seen, commands = set(), []
    for module in modules:
        torch, vision, tag = VERSIONS[module]
        if tag in seen:
            continue
        seen.add(tag)
        commands.append([
            "docker", "build", "-f", "workers/gpu_runtime/Dockerfile",
            "--build-arg", f"TORCH_VERSION={torch}",
            "--build-arg", f"TORCHVISION_VERSION={vision}",
            "-t", f"goodvision-gpu-runtime:torch{tag}-cu128", ".",
        ])
    return commands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("modules", nargs="+", choices=VERSIONS)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    for command in build_commands(args.modules):
        print(" ".join(command), flush=True)
        if not args.plan:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
