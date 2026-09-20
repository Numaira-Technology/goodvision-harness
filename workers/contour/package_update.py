"""Build a small offline COPY-only update using an explicitly verified base."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE_IMAGES = (
    {
        "service": "engine",
        "image": "goodvision-contour-runtime:0.1.0",
        "base_id": (
            "sha256:cf0c9ddde49279f0cff86e22811d1fc2770f7d43650ca3451a1625a9fe0915d8"
        ),
    },
    {
        "service": "contour",
        "image": "goodvision-contour:0.1.0",
        "base_id": (
            "sha256:ae13370f0bc7816c3c8d79d7b87c6445ddc8f635f85af4a10bd6f57d73d4ff47"
        ),
    },
)
ENGINE = (
    "app.py",
    "sam31.py",
    "continuous.py",
    "streaming_state.py",
    "h264_transport.py",
    "perception_routes.py",
    "sam_live.py",
    "multistream.py",
    "gpu_memory.py",
)
GATEWAY = (
    "__init__.py",
    "settings.py",
    "contracts.py",
    "engine_adapter.py",
    "app.py",
    "static/index.html",
    "static/console.css",
    "static/console.js",
    "static/favicon.svg",
)


def package(output: Path, release="20260920-memory1"):
    if not re.fullmatch("[a-z0-9-]+", release):
        raise ValueError("Invalid release name")
    files = {}
    for directory, names, source in (
        ("engine", ENGINE, ROOT.parent / "sam_scene"),
        ("contour", GATEWAY, ROOT),
    ):
        for name in names:
            path = source / name
            if not path.is_file() or path.is_symlink():
                raise ValueError(f"Missing regular update file: {path}")
            files[f"payload/{directory}/{name}"] = path.read_bytes()
    files["update_contour.ps1"] = (ROOT / "update_contour.ps1").read_bytes()
    files["Dockerfile.engine"] = (
        b"ARG BASE_IMAGE=goodvision-contour-runtime:0.1.0\n"
        b"FROM ${BASE_IMAGE}\nCOPY engine/ /engine/\n"
    )
    files["Dockerfile.contour"] = (
        b"ARG BASE_IMAGE=goodvision-contour:0.1.0\n"
        b"FROM ${BASE_IMAGE}\nCOPY contour/ /opt/contour/\n"
    )
    files["UPDATE.cmd"] = (
        b"@echo off\r\npowershell.exe -NoProfile -ExecutionPolicy Bypass "
        b'-File "%~dp0update_contour.ps1" -PackageDir "%~dp0."\r\n'
        b"set UPDATE_EXIT=%ERRORLEVEL%\r\n"
        b"pause\r\nexit /b %UPDATE_EXIT%\r\n"
    )
    files["README.md"] = (
        (ROOT / "CODE_UPDATE.md")
        .read_text(encoding="utf-8")
        .replace("{release}", release)
        .encode()
    )
    manifest = {
        "product": "GoodVision Contour Code Update",
        "release": release,
        "images": list(BASE_IMAGES),
        "includes_model": False,
        "files": [
            {
                "path": name,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            for name, data in sorted(files.items())
        ],
    }
    files["update-manifest.json"] = json.dumps(manifest, indent=2).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(".zip.partial")
    folder = "goodvision-contour-update-" + release
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(files.items()):
            archive.writestr(folder + "/" + name, data)
    partial.replace(output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".zip.sha256").write_text(digest + "  " + output.name + "\n")
    return {"path": str(output), "bytes": output.stat().st_size, "sha256": digest}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--release", default="20260920-memory1")
    args = parser.parse_args()
    print(json.dumps(package(args.output, args.release), indent=2))
