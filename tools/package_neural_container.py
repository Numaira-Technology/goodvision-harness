"""Package the first standalone CLIP image without resending installed CUDA layers.

The delta targets Docker's containerd image store and requires the exact V2.0
System1 image. The supplied source also documents the offline image build.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path

if __package__:
    from . import module_updates as updates
    from .package_customer_docker_update import add_config_ids
    from .package_image_update import archive_images
else:
    import module_updates as updates
    from package_customer_docker_update import add_config_ids
    from package_image_update import archive_images

ROOT = Path(__file__).resolve().parents[1]
BASE_NAME = "goodvision-system1:pr75-603f20d"
BASE_ID = "sha256:b7e2098b9aa4e6dcf98810797d490df5993027db31e476ad22d326e73750f779"
BASE_CONFIG_ID = (
    "sha256:e5685ef8ac313ef5455e9ac29c9ab1546d97b9b74cf552c45da26f0cd0d71ff0"
)
APPLICATION_REVISION = "603f20dc0089cc0b0f21429d8432f3c32aa4b2a6"
SOURCE_FILES = (
    "workers/neural/Dockerfile",
    "workers/neural/Dockerfile.dockerignore",
    "workers/neural/healthcheck.py",
    "workers/neural/README.md",
    "tools/neural_container.py",
    "tools/package_neural_container.py",
)


def read_baseline(path):
    """Use the original release's complete blob inventory, not a guessed tag."""
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            metadata = archive.getinfo("update.json")
            if metadata.file_size > 1024 * 1024:
                raise ValueError("Baseline metadata is too large")
            raw = archive.read(metadata)
    else:
        raw = path.read_bytes()
    value = json.loads(raw)
    images = value["images"]
    images = images.values() if isinstance(images, dict) else images
    found = [entry for entry in images if entry.get("name") == BASE_NAME]
    if len(found) != 1 or found[0].get("id") != BASE_ID:
        raise ValueError("Use the exact delivered V2.0 System1 baseline")
    image = dict(found[0])
    if image.get("os") != "linux" or image.get("architecture") != "amd64":
        raise ValueError("Baseline must be Linux/amd64")
    blobs = image.get("blobs")
    if not isinstance(blobs, dict) or BASE_ID not in blobs:
        raise ValueError("Baseline must include its complete OCI blob inventory")
    if any(
        not re.fullmatch(r"sha256:[a-f0-9]{64}", key)
        or type(size) is not int or size < 0
        for key, size in blobs.items()
    ):
        raise ValueError("Invalid baseline OCI blob inventory")
    image["config_id"] = BASE_CONFIG_ID
    return image, hashlib.sha256(raw).hexdigest()


def inspect_image(docker, name, run):
    result = run(
        [docker, "image", "inspect", name], check=True,
        capture_output=True, text=True,
    )
    return json.loads(result.stdout)[0]


def copy_source(root, output):
    sources = [*SOURCE_FILES, *(
        path.relative_to(root).as_posix()
        for path in sorted((root / "tools/neural_delivery").glob("*.cmd"))
    )]
    hashes = {}
    for relative in sources:
        source = root / relative
        if (source.is_symlink() or not source.is_file()
                or not source.resolve().is_relative_to(root.resolve())):
            raise ValueError("Missing or unsafe release source: " + relative)
        target = output / "source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        hashes[relative] = updates.digest(target)
    shutil.copyfile(root / "tools/neural_container.py", output / "neural_container.py")
    shutil.copyfile(root / "workers/neural/README.md", output / "README.md")
    for path in sorted((root / "tools/neural_delivery").glob("*.cmd")):
        shutil.copyfile(path, output / path.name)
    return hashes


def delta_archive(full, target, image, baseline):
    """Keep only declared new-image blobs; exclude every legacy save alias."""
    wanted = set(image["blobs"]) - set(baseline["blobs"])
    omitted = 0
    with (
        target.open("xb") as raw,
        gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=1, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w|") as delta,
        tarfile.open(full, "r:") as archive,
    ):
        for member in archive:
            if not member.isfile():
                continue
            if member.name not in {"index.json", "oci-layout"}:
                if not re.fullmatch(r"blobs/sha256/[a-f0-9]{64}", member.name):
                    continue
                key = "sha256:" + member.name.rsplit("/", 1)[1]
                if key not in wanted:
                    omitted += member.size
                    continue
            delta.addfile(member, archive.extractfile(member))
    updates.validate_archive(target, {"neural": image})
    return omitted


def package(image_name, baseline_path, output, *, docker="docker", root=ROOT,
            run=subprocess.run, validation=None):
    if not re.fullmatch(r"goodvision-neural:[a-zA-Z0-9_][a-zA-Z0-9_.-]{0,127}",
                        image_name):
        raise ValueError("Select an explicitly tagged goodvision-neural image")
    output, root = Path(output).resolve(), Path(root).resolve()
    zipped = output.parent / (output.name + ".zip")
    if output.exists() or zipped.exists():
        raise ValueError("Refusing to overwrite an existing delivery")
    validation_bytes = None
    if validation is not None:
        try:
            validation_bytes = Path(validation).read_bytes()
            validation_document = json.loads(validation_bytes)
        except (OSError, ValueError) as error:
            raise ValueError("Validation must be a readable JSON object") from error
        if not isinstance(validation_document, dict):
            raise ValueError("Validation must be a JSON object")
    baseline, baseline_metadata_sha256 = read_baseline(baseline_path)
    installed_base = inspect_image(docker, BASE_NAME, run)
    if (installed_base.get("Id") not in {BASE_ID, BASE_CONFIG_ID}
            or installed_base.get("Os") != "linux"
            or installed_base.get("Architecture") != "amd64"):
        raise ValueError("Build machine baseline identity differs from V2.0")
    built = inspect_image(docker, image_name, run)
    layers = built.get("RootFS", {}).get("Layers", [])
    base_layers = installed_base.get("RootFS", {}).get("Layers", [])
    if not base_layers or layers[:len(base_layers)] != base_layers:
        raise ValueError("Neural image does not inherit the verified V2.0 layers")
    if (built.get("Os") != "linux" or built.get("Architecture") != "amd64"
            or built.get("Config", {}).get("Labels", {}).get("com.goodvision.module")
            != "goodvision-neural"):
        raise ValueError("Select the standalone Linux/amd64 neural image")
    config = built["Config"]
    if (
        config.get("Entrypoint") != [
            "python", "-m", "goodvision_harness.neural_service",
        ]
        or config.get("Cmd") != ["--host", "0.0.0.0", "--port", "8098",
                                 "--cache-dir", "/models/huggingface"]
        or config.get("Healthcheck", {}).get("Test") != [
            "CMD", "python", "/opt/goodvision-neural/healthcheck.py",
        ]
    ):
        raise ValueError("Neural image needs its dedicated entrypoint and healthcheck")
    revision = run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="goodvision-neural-export-") as temporary:
        full = Path(temporary) / "full.tar"
        run([docker, "image", "save", "--output", str(full), image_name], check=True)
        images = archive_images(full)
        if len(images) != 1 or images[0]["name"] != image_name:
            raise ValueError("Export must contain only the selected neural image tag")
        image = images[0]
        add_config_ids(full, {"neural": image})
        if built["Id"] not in {image["id"], image["config_id"]}:
            raise ValueError("Image changed during export")
        with tempfile.TemporaryDirectory(prefix=".neural-delivery-",
                                         dir=output.parent) as staging:
            stage = Path(staging) / output.name
            stage.mkdir()
            archive = stage / "images.tar.gz"
            omitted = delta_archive(full, archive, image, baseline)
            hashes = copy_source(root, stage)
            release = {
                "schema": 1,
                "module": "goodvision-neural",
                "version": "v2.0",
                "kind": "initial-standalone-container",
                "platform": "linux/amd64",
                "image": {key: value for key, value in image.items() if key != "blobs"},
                "image_blobs": image["blobs"],
                "requires": [{key: value for key, value in baseline.items()
                              if key != "blobs"}],
                "image_archive": {
                    "file": archive.name,
                    "bytes": archive.stat().st_size,
                    "sha256": updates.digest(archive),
                    "kind": "oci-delta",
                    "requires_containerd_image_store": True,
                },
                "source": {
                    "repository_revision": revision,
                    "application_revision": APPLICATION_REVISION,
                    "application_update": False,
                    "baseline_metadata_sha256": baseline_metadata_sha256,
                    "files_sha256": hashes,
                },
                "source_files": [
                    {"file": "source/" + relative,
                     "bytes": (stage / "source" / relative).stat().st_size,
                     "sha256": checksum}
                    for relative, checksum in hashes.items()
                ],
                "build": {
                    "dockerfile": "source/workers/neural/Dockerfile",
                    "context": "source",
                    "base_image": BASE_NAME,
                    "base_image_id": BASE_ID,
                    "network": "none",
                    "pull": False,
                },
                "omitted_existing_blob_bytes": omitted,
                "model_payload_included": False,
                "customer_deployment_status": "unknown",
            }
            if validation_bytes is not None:
                (stage / "VALIDATION.json").write_bytes(validation_bytes)
                release["validation"] = {
                    "file": "VALIDATION.json",
                    "bytes": len(validation_bytes),
                    "sha256": hashlib.sha256(validation_bytes).hexdigest(),
                }
            (stage / "RELEASE.json").write_text(
                json.dumps(release, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            stage.rename(output)
    # Zip creation is exclusive; a previous delivery is never silently replaced.
    with zipfile.ZipFile(zipped, "x", compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(output).as_posix())
    return release


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--docker", default="docker")
    parser.add_argument("--validation", type=Path)
    args = parser.parse_args()
    release = package(args.image, args.baseline, args.output, docker=args.docker,
                      validation=args.validation)
    print(json.dumps({
        "module": release["module"], "version": release["version"],
        "output": str(args.output.resolve()),
        "archive_bytes": release["image_archive"]["bytes"],
        "omitted_existing_blob_bytes": release["omitted_existing_blob_bytes"],
    }, indent=2))


if __name__ == "__main__":
    main()
