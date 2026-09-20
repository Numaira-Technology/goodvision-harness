"""Export both prebuilt images as one compressed, checksummed offline archive."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import shutil
import subprocess
from pathlib import Path

IMAGES = ("goodvision-contour:0.1.0", "goodvision-contour-runtime:0.1.0")


def export(output: Path, context: str | None):
    docker = ["docker"] + (["--context", context] if context else [])
    # Fail before writing a large archive if either required image is missing.
    subprocess.run(
        [*docker, "image", "inspect", *IMAGES],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    with subprocess.Popen(
        [*docker, "save", *IMAGES], stdout=subprocess.PIPE
    ) as process:
        try:
            with partial.open("wb") as raw:
                with gzip.GzipFile(
                    fileobj=raw, mode="wb", compresslevel=1, mtime=0
                ) as gz:
                    shutil.copyfileobj(process.stdout, gz, length=4 * 1024 * 1024)
            if process.wait() != 0:
                raise RuntimeError("Docker image export failed; partial file retained")
        except BaseException:
            process.terminate()
            process.wait()
            raise
    partial.replace(output)
    with output.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    output.with_name(output.name + ".sha256").write_text(
        digest + "  " + output.name + "\n", encoding="utf-8"
    )
    print(f"Created {output} ({output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--context", help="Optional Docker context on the build machine"
    )
    args = parser.parse_args()
    export(args.output.resolve(), args.context)
