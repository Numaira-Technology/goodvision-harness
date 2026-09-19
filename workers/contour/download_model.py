"""Explicit, resumable model download; never invoked by service startup.

Requires authorized model access. Credentials are read from HF_TOKEN or the
normal local Hugging Face token file, never put in URLs, output, or manifests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import httpx

REPOSITORY = "facebook/sam3.1"
REVISION = "daa63191845a41281374e725f4c9e51c7a824460"
FILENAME = "sam3.1_multiplex.pt"
ORIGIN = "https://huggingface.co"


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def download(destination: Path):
    token = os.environ.get("HF_TOKEN")
    token_file = Path(os.getenv("HF_HOME", str(Path.home() / ".cache/huggingface")))
    if not token and (token_file / "token").is_file():
        token = (token_file / "token").read_text(encoding="utf-8").strip()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / "contour.pt"
    partial = destination / "contour.pt.partial"
    with httpx.Client(timeout=60, follow_redirects=True, headers=headers) as client:
        metadata = client.get(
            f"{ORIGIN}/api/models/{REPOSITORY}/revision/{REVISION}?blobs=true"
        )
        metadata.raise_for_status()
        entry = next(
            item
            for item in metadata.json()["siblings"]
            if item["rfilename"] == FILENAME
        )
        expected = entry["lfs"]["sha256"]
        total = entry["lfs"]["size"]
        if target.exists():
            if target.stat().st_size != total or digest(target) != expected:
                raise ValueError("Existing checkpoint differs; left untouched")
            print("Existing checkpoint verified.", flush=True)
        else:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > total:
                raise ValueError("Partial checkpoint exceeds expected length")
            if shutil.disk_usage(destination).free < total - offset + 128 * 1024 * 1024:
                raise ValueError("Insufficient disk space")
            url = f"{ORIGIN}/{REPOSITORY}/resolve/{REVISION}/{FILENAME}"
            for attempt in range(3):
                offset = partial.stat().st_size if partial.exists() else 0
                if offset == total:
                    break
                try:
                    with client.stream(
                        "GET",
                        url,
                        headers={"Range": f"bytes={offset}-"} if offset else {},
                    ) as response:
                        response.raise_for_status()
                        if response.status_code == 206:
                            if not response.headers.get("content-range", "").startswith(
                                f"bytes {offset}-"
                            ):
                                raise ValueError("Invalid resume range")
                        elif response.status_code == 200:
                            offset = 0
                        else:
                            raise ValueError("Unexpected download response")
                        mode = "ab" if offset else "wb"
                        updated = 0.0
                        with partial.open(mode) as handle:
                            for chunk in response.iter_bytes(1024 * 1024):
                                if offset + len(chunk) > total:
                                    raise ValueError("Download exceeds expected size")
                                handle.write(chunk)
                                offset += len(chunk)
                                if time.monotonic() - updated > 10:
                                    print(
                                        f"Checkpoint: {offset / total:.1%} "
                                        f"({offset // 1048576}/{total // 1048576} MiB)",
                                        flush=True,
                                    )
                                    updated = time.monotonic()
                    if offset != total:
                        raise ValueError("Incomplete download; resume by rerunning")
                    break
                except httpx.TransportError:
                    if attempt == 2:
                        raise
                    print(
                        "Connection interrupted; resuming partial download.", flush=True
                    )
            print("Verifying official SHA-256 checksum...", flush=True)
            if partial.stat().st_size != total or digest(partial) != expected:
                raise ValueError("Checkpoint checksum mismatch; partial file retained")
            partial.replace(target)
        license_response = client.get(
            f"{ORIGIN}/{REPOSITORY}/resolve/{REVISION}/LICENSE"
        )
        license_response.raise_for_status()
        (destination / "MODEL_LICENSE.txt").write_bytes(license_response.content)
        manifest = {
            "source_repository": REPOSITORY,
            "revision": REVISION,
            "source_filename": FILENAME,
            "local_filename": target.name,
            "bytes": total,
            "sha256": expected,
            "license_file": "MODEL_LICENSE.txt",
        }
        (destination / "model-manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        print(
            "Checkpoint, original license and provenance manifest are ready.",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Download authorized Contour model assets"
    )
    parser.add_argument(
        "--destination", type=Path, default=Path(__file__).parent / "models"
    )
    args = parser.parse_args()
    try:
        download(args.destination)
    except httpx.HTTPStatusError as error:
        print(f"Download refused (HTTP {error.response.status_code}).")
        if error.response.status_code in {401, 403}:
            print("Request access to facebook/sam3.1 and configure HF_TOKEN locally.")
        raise SystemExit(1) from None
    except (httpx.HTTPError, OSError, ValueError, KeyError, StopIteration) as error:
        print(f"Download incomplete ({type(error).__name__}); existing files retained.")
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
