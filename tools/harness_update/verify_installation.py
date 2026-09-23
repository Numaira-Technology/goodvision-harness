"""Recognize source-only upgrades without relying on machine-specific image IDs."""

import hashlib
import io
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import PurePosixPath
from urllib.parse import quote


def verify_installation(engine, manifest, base_id, targets, verification_base_id=None):
    if base_id in manifest["accepted_base_ids"]:
        return
    if verification_base_id not in manifest["accepted_base_ids"]:
        raise RuntimeError(
            "Unsupported installed version: a verified original base image is required."
        )
    current = engine.request("GET", f"/images/{quote(base_id, safe='')}/json")
    trusted = engine.request(
        "GET", f"/images/{quote(verification_base_id, safe='')}/json"
    )
    layers = trusted.get("RootFS", {}).get("Layers", [])
    if (
        not layers
        or current.get("RootFS", {}).get("Layers", [])[: len(layers)] != layers
    ):
        raise RuntimeError(
            "Installed image does not descend from the verified base image."
        )
    # Container labels can be inherited from a previous release; inspect the image.
    release = (current.get("Config", {}).get("Labels") or {}).get(
        "goodvision.ui.release"
    )
    sources = manifest.get("published_sources", {}).get(release)
    if not sources:
        raise RuntimeError(
            f"Unsupported installed release: {release or '(missing release label)'}. "
            "No containers were changed."
        )
    for name in sources:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name:
            raise RuntimeError("Invalid published source path.")

    def verify_file(item):
        target, root, name, expected = item
        path = root + "/" + name
        data = engine.request(
            "GET",
            f"/containers/{quote(target['Id'], safe='')}/archive"
            f"?path={quote(path, safe='')}",
            raw=True,
        )
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            members = archive.getmembers()
            if len(members) != 1 or not members[0].isfile():
                raise RuntimeError(f"Installed source is not a regular file: {name}")
            with archive.extractfile(members[0]) as source:
                digest = hashlib.file_digest(source, "sha256").hexdigest()
        if digest != expected:
            raise RuntimeError(
                f"Installed source checksum mismatch: {target['Name']} {path}"
            )

    jobs = (
        (target, root, name, digest)
        for target in targets
        for root in (
            "/app/src/goodvision_harness",
            "/usr/local/lib/python3.11/site-packages/goodvision_harness",
        )
        for name, digest in sources.items()
    )
    # Read files through Docker, without executing code from the unverified image.
    with ThreadPoolExecutor(max_workers=8) as pool:
        for _ in pool.map(verify_file, jobs):
            pass
    print(
        f"Verified published release {release}: {len(sources)} files "
        "in both application paths per container.",
        flush=True,
    )
