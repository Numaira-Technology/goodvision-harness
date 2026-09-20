"""Explicit allowlist exporter: no credentials, private media or unrelated code."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

CONTOUR_FILES = (
    "__init__.py",
    "app.py",
    "contracts.py",
    "engine_adapter.py",
    "settings.py",
    "requirements.txt",
    "Dockerfile",
    "Dockerfile.engine",
    "Dockerfile.engine.dockerignore",
    "repair_decord_metadata.py",
    "Dockerfile.jetson",
    "Dockerfile.jetson.dockerignore",
    "prepare_jetson.py",
    "run_jetson.py",
    "jetson_runtime.py",
    "compose.jetson.yaml",
    "JETSON.md",
    ".dockerignore",
    ".gitignore",
    ".env.example",
    "compose.yaml",
    "compose.gpu.yaml",
    "initialize.py",
    "deploy_contour.py",
    "deploy_contour.ps1",
    "download_model.py",
    "package_release.py",
    "package_update.py",
    "update_contour.ps1",
    "CODE_UPDATE.md",
    "export_images.py",
    "example_client.py",
    "README.md",
    "API.md",
    "THIRD_PARTY_NOTICES.md",
    "static/index.html",
    "static/console.css",
    "static/console.js",
    "static/favicon.svg",
)
ENGINE_FILES = (
    "requirements.txt",
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


def checksum(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def package(output: Path, *, include_model=False):
    contour = Path(__file__).resolve().parent
    workers = contour.parent
    files = [(contour / name, "workers/contour/" + name) for name in CONTOUR_FILES]
    files += [
        (workers / "sam_scene" / name, "workers/sam_scene/" + name)
        for name in ENGINE_FILES
    ]
    if include_model:
        files += [
            (contour / "models" / name, "workers/contour/models/" + name)
            for name in ("contour.pt", "MODEL_LICENSE.txt", "model-manifest.json")
        ]
    # Validate the complete selection before opening or replacing an output bundle.
    manifest = []
    for source, name in files:
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"Missing regular release file: {name}")
        manifest.append(
            {"path": name, "bytes": source.stat().st_size, "sha256": checksum(source)}
        )
    if include_model:
        origin = json.loads((contour / "models/model-manifest.json").read_text())
        model = next(row for row in manifest if row["path"].endswith("/contour.pt"))
        if model["sha256"] != origin["sha256"] or model["bytes"] != origin["bytes"]:
            raise ValueError("Model differs from verified download manifest")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(".zip.partial")
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source, name in files:
            archive.write(
                source,
                name,
                compress_type=(
                    zipfile.ZIP_STORED
                    if source.suffix == ".pt"
                    else zipfile.ZIP_DEFLATED
                ),
            )
        archive.writestr(
            "START_HERE.md",
            (
                "# GoodVision Contour\n\n"
                "See workers/contour/README.md for deployment "
                "and workers/contour/API.md "
                "for the API. Run initialize.py to generate a new node key.\n\n"
                "This bundle includes source, not prebuilt container images.\n"
            ),
        )
        archive.writestr(
            "release-manifest.json",
            json.dumps(
                {
                    "product": "GoodVision Contour",
                    "version": "0.1.0",
                    "includes_model": include_model,
                    "files": manifest,
                },
                indent=2,
            ),
        )
    partial.replace(output)
    output.with_suffix(".zip.sha256").write_text(
        checksum(output) + "  " + output.name + "\n", encoding="utf-8"
    )
    print(f"Created {output} ({output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package a standalone Contour release")
    parser.add_argument("--include-model", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    package(args.output.resolve(), include_model=args.include_model)
