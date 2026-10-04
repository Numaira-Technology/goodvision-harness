"""Build a Windows one-click update ZIP from explicitly selected Docker images.

Maintainers run this on Windows with PyInstaller installed. Customers receive
Update.exe and verified, prepared images; they need only their existing Docker.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

if __package__:
    from .customer_update import PRODUCT
    from .update_docker import Docker, UpdateError
else:
    from customer_update import PRODUCT
    from update_docker import Docker, UpdateError

ROOT = Path(__file__).resolve().parent


def build_executable(output_directory):
    if os.name != "nt":
        raise ValueError(
            "Build Update.exe on Windows, or supply a previously built --executable."
        )
    output_directory = Path(output_directory).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "PyInstaller",
                "--noconfirm",
                "--clean",
                "--onefile",
                "--windowed",
                "--name",
                "Update",
                "--distpath",
                str(output_directory),
                "--workpath",
                str(output_directory / "build"),
                "--specpath",
                str(output_directory / "build"),
                "--paths",
                str(ROOT),
                str(ROOT / "customer_update_gui.py"),
            ],
            check=True,
        )
    except subprocess.CalledProcessError as error:
        raise ValueError(
            "Install PyInstaller in the Windows build environment and retry."
        ) from error
    executable = output_directory / "Update.exe"
    if not executable.is_file():
        raise ValueError("The Windows updater build did not produce Update.exe.")
    return executable


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package(output, release, component, images, executable, *, docker=None):
    roles = {"harness"} if component == "harness" else {"contour", "engine"}
    if (
        component not in {"harness", "contour"}
        or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}", release)
        or set(images) != roles
    ):
        raise ValueError("Choose a release and every required component image.")
    executable = Path(executable).resolve()
    if not executable.is_file() or executable.is_symlink():
        raise ValueError("A built Windows Update.exe is required.")
    with executable.open("rb") as stream:
        if stream.read(2) != b"MZ":
            raise ValueError("The updater must be a Windows executable.")
    docker = docker or Docker()
    metadata = {}
    for role, reference in sorted(images.items()):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:/@-]*", reference):
            raise ValueError("Invalid Docker image reference.")
        image = docker.json("image", "inspect", reference)[0]
        if image.get("Os") != "linux" or image.get("Architecture") != "amd64":
            raise ValueError("Customer packages require Linux amd64 Docker images.")
        if not re.fullmatch(r"sha256:[a-f0-9]{64}", image["Id"]):
            raise ValueError("Docker returned an invalid image ID.")
        metadata[role] = {"reference": reference, "id": image["Id"]}
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError(
            "Choose a new output filename; existing releases are never replaced."
        )
    partial = output.with_suffix(".partial")
    if partial.exists():
        raise ValueError("A partial package already exists; use a new output filename.")
    prefix = f"GoodVision-{component}-update-{release}/"
    try:
        with tempfile.TemporaryDirectory(
            prefix="goodvision-update-", dir=output.parent
        ) as temp:
            temp = Path(temp)
            raw = temp / "images.tar"
            docker.run(
                "image",
                "save",
                "--output",
                str(raw),
                *sorted({image["id"] for image in metadata.values()}),
                timeout=7200,
            )
            archive = temp / "images.tar.gz"
            with (
                raw.open("rb") as source,
                gzip.open(archive, "wb", compresslevel=1) as target,
            ):
                shutil.copyfileobj(source, target, 4 * 1024 * 1024)
            manifest = {
                "product": PRODUCT,
                "schema_version": 1,
                "release": release,
                "component": component,
                "archive": {"file": "images.tar.gz", "sha256": file_digest(archive)},
                "images": metadata,
            }
            with zipfile.ZipFile(
                partial, "x", compression=zipfile.ZIP_STORED, allowZip64=True
            ) as bundle:
                bundle.write(executable, prefix + "Update.exe")
                bundle.write(archive, prefix + "images.tar.gz")
                bundle.writestr(
                    prefix + "update-package.json",
                    json.dumps(manifest, indent=2) + "\n",
                )
                instructions = (
                    "Extract this ZIP into a new folder. Double-click Update.exe "
                    "and wait for Update complete.\n"
                )
                if component == "contour":
                    instructions += (
                        "First stop analysis in the main GoodVision console.\n"
                    )
                instructions += (
                    "Keep Docker Desktop open. Your settings, models and data "
                    "are kept.\n"
                )
                bundle.writestr(prefix + "Read me.txt", instructions)
        partial.replace(output)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    digest = file_digest(output)
    output.with_suffix(output.suffix + ".sha256").write_text(
        digest + "  " + output.name + "\n", encoding="utf-8"
    )
    return {"path": str(output), "sha256": digest, "bytes": output.stat().st_size}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--component", choices=("harness", "contour"), required=True)
    parser.add_argument(
        "--image", action="append", required=True, metavar="SERVICE=IMAGE"
    )
    parser.add_argument("--executable", type=Path)
    args = parser.parse_args(argv)
    try:
        images = {}
        for assignment in args.image:
            role, separator, reference = assignment.partition("=")
            if not separator or not role or not reference or role in images:
                raise ValueError("Use unique --image SERVICE=IMAGE values.")
            images[role] = reference
        executable = args.executable or build_executable(
            args.output.resolve().parent / "updater-runtime"
        )
        print(
            json.dumps(
                package(args.output, args.release, args.component, images, executable),
                indent=2,
            )
        )
        return 0
    except (ValueError, OSError, UpdateError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
