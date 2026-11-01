"""Export and verify both current release images as one local archive."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

IMAGE_TAGS = (
    "goodvision-harness:20260923-full1",
    "goodvision-rich-label-worker:20260923-full1",
)
ARCHIVE_NAME = "goodvision-full-20260923-full1.tar.gz"


def export(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    directory = directory.resolve()
    archive = directory / ARCHIVE_NAME
    partial = directory / (ARCHIVE_NAME + ".partial")
    if archive.exists() or partial.exists():
        raise RuntimeError("The archive destination already exists")
    images = json.loads(subprocess.check_output(
        ["docker", "image", "inspect", *IMAGE_TAGS], text=True,
    ))
    if len(images) != len(IMAGE_TAGS):
        raise RuntimeError("Both current release images must be installed")
    for tag, item in zip(IMAGE_TAGS, images, strict=True):
        if tag not in item["RepoTags"]:
            raise RuntimeError("Unexpected release image tag")
        if (item["Os"], item["Architecture"]) != ("linux", "amd64"):
            raise RuntimeError("Both images must target linux/amd64")
    started = time.monotonic()
    print("Exporting both current release images", flush=True)
    owned_partial = False
    try:
        with partial.open("xb") as output:
            owned_partial = True
            # Keep diagnostics separate from the binary stream on Windows.
            with (directory / "export.log").open("wb") as errors:
                process = subprocess.Popen(
                    ["docker", "image", "save", *IMAGE_TAGS],
                    stdout=subprocess.PIPE, stderr=errors,
                )
                try:
                    assert process.stdout is not None
                    with gzip.GzipFile(
                        fileobj=output, mode="wb", compresslevel=1, mtime=0,
                    ) as compressed:
                        shutil.copyfileobj(
                            process.stdout, compressed, 4 * 1024 * 1024,
                        )
                    process.stdout.close()
                    if process.wait() != 0:
                        raise RuntimeError("Docker image export failed; see export.log")
                finally:
                    if process.poll() is None:
                        process.terminate()
                        process.wait(timeout=30)
        print("Verifying the complete archive and both image tags", flush=True)
        manifests = []
        with partial.open("rb") as stream:
            with tarfile.open(fileobj=stream, mode="r|*") as contents:
                for member in contents:
                    if member.name == "manifest.json":
                        if not member.isfile() or member.size > 1024 * 1024:
                            raise RuntimeError("Invalid image manifest")
                        payload = contents.extractfile(member)
                        assert payload is not None
                        manifests.append(json.load(payload))
        if len(manifests) != 1 or len(manifests[0]) != 2:
            raise RuntimeError("Archive must contain exactly two image entries")
        tags = [tag for item in manifests[0] for tag in item.get("RepoTags", [])]
        if len(tags) != 2 or set(tags) != set(IMAGE_TAGS):
            raise RuntimeError("Archive contains unexpected or missing image tags")
        digest = hashlib.sha256()
        with partial.open("rb") as source:
            for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
                digest.update(block)
        partial.rename(archive)
        metadata = {
            "platform": "linux/amd64",
            "images": [
                {
                    "tag": tag, "id": item["Id"], "created": item["Created"],
                    "source_revision": item["Config"].get("Labels", {}).get(
                        "org.opencontainers.image.revision",
                    ),
                }
                for tag, item in zip(IMAGE_TAGS, images, strict=True)
            ],
            "archive": archive.name,
            "archive_bytes": archive.stat().st_size,
            "sha256": digest.hexdigest(),
        }
        (directory / "IMAGES.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8",
        )
        (directory / "SHA256SUMS.txt").write_text(
            f"{digest.hexdigest()}  {archive.name}\n", encoding="utf-8",
        )
        elapsed = round(time.monotonic() - started, 1)
        print(json.dumps({**metadata, "elapsed_seconds": elapsed}), flush=True)
    finally:
        if owned_partial and partial.exists():
            partial.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    export(parser.parse_args().directory)
