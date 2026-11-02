"""Export one module, omitting image blobs already in its installed release."""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path

if __package__:
    from . import docker_modules as spec
    from . import module_updates as updates
else:
    import docker_modules as spec
    import module_updates as updates


def archive_images(path):
    objects, sizes, index = {}, {}, None
    with tarfile.open(path, "r:*") as archive:
        for member in archive:
            if member.name == "index.json":
                index = json.load(archive.extractfile(member))
            elif member.name.startswith("blobs/sha256/") and member.isfile():
                key = "sha256:" + member.name.rsplit("/", 1)[1]
                sizes[key] = member.size
                if member.size <= 2 * 1024 * 1024:
                    data = archive.extractfile(member).read()
                    try:
                        value = json.loads(data)
                        if isinstance(value, dict):
                            objects[key] = value
                    except (ValueError, UnicodeError):
                        pass
    if index is None:
        raise ValueError("An OCI image archive is required")

    def walk(key, result):
        if key in result:
            return
        result[key] = sizes[key]
        node = objects.get(key, {})
        if "rootfs" in node:
            return
        children = node.get("manifests", []) + node.get("layers", [])
        if "digest" in node.get("config", {}):
            children += [node["config"]]
        for child in children:
            walk(child["digest"], result)

    images = []
    for entry in index["manifests"]:
        name = entry["annotations"]["io.containerd.image.name"]
        blobs = {}
        walk(entry["digest"], blobs)
        configs = [objects[key] for key in blobs if "rootfs" in objects.get(key, {})]
        config = next(value for value in configs if value.get("os") != "unknown")
        images.append(
            {
                "name": name.removeprefix("docker.io/library/"),
                "id": entry["digest"],
                "os": config["os"],
                "architecture": config["architecture"],
                "blobs": blobs,
            }
        )
    return images


def role(module, image):
    if not image["name"].startswith(f"goodvision-{module}:"):
        raise ValueError("Image belongs to another module")
    if module == "contour":
        if image["name"].endswith("-engine"):
            return "engine"
        if image["name"].endswith("-api"):
            return "contour"
        raise ValueError("Contour image must use an explicit -api or -engine role")
    if module == "gpu-runtime":
        return next(
            "torch" + version
            for version in spec.TORCH
            if image["name"].endswith(f"-torch{version}-cu128")
        )
    return module


def baseline_images(path, module):
    if path is None:
        return []
    value = json.loads(path.read_text())
    images = value["images"]
    images = list(images.values()) if isinstance(images, dict) else images
    images = [
        image for image in images if image["name"].startswith(f"goodvision-{module}:")
    ]
    if not images:
        raise ValueError("Baseline does not contain the selected module")
    if any("blobs" not in image for image in images):
        archive = value.get("archive", {}).get("file", value.get("file"))
        if not archive:
            raise ValueError("Baseline has no image blob inventory")
        actual = {
            image["name"]: image
            for image in archive_images(updates.relative_file(path.parent, archive))
        }
        for image in images:
            found = actual[image["name"]]
            if found["id"] != image["id"]:
                raise ValueError("Baseline image identity mismatch")
            image.update(found)
    return images


def package(module, version, names, output, baseline=None, assets=()):
    spec.name(version)
    old = baseline_images(baseline, module)
    known = {digest for image in old for digest in image["blobs"]}
    output.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory(prefix="goodvision-export-") as temporary:
        full = Path(temporary) / "images.tar"
        subprocess.run(
            ["docker", "image", "save", "--output", str(full), *names], check=True
        )
        images = {role(module, image): image for image in archive_images(full)}
        if set(images) != updates.ROLES[module]:
            raise ValueError("Selected images do not match the module roles")
        wanted = {key for image in images.values() for key in image["blobs"]} - known
        target = output / "images.tar.gz"
        omitted = 0
        with (
            target.open("wb") as raw,
            gzip.GzipFile(
                fileobj=raw, mode="wb", compresslevel=1, mtime=0
            ) as compressed,
            tarfile.open(fileobj=compressed, mode="w|") as delta,
        ):
            with tarfile.open(full, "r:") as archive:
                for member in archive:
                    if not member.isfile():
                        continue
                    if member.name not in {"oci-layout", "index.json"}:
                        if not member.name.startswith("blobs/sha256/"):
                            continue
                        key = "sha256:" + member.name.rsplit("/", 1)[1]
                        if key not in wanted:
                            omitted += member.size
                            continue
                    delta.addfile(member, archive.extractfile(member))
        additions = []
        for target_name, source in assets:
            parts = Path(target_name).parts
            if (
                Path(target_name).is_absolute()
                or ".." in parts
                or "\\" in target_name
                or ":" in target_name
            ):
                raise ValueError("Invalid model addition target")
            target = output / "assets" / target_name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            additions.append(
                {
                    "target": target_name,
                    "file": target.relative_to(output).as_posix(),
                    "bytes": target.stat().st_size,
                    "sha256": updates.digest(target),
                }
            )
        manifest = {
            "schema": 1,
            "module": module,
            "version": version,
            "kind": "delta" if old else "full",
            "images": images,
            "requires": [
                {key: value for key, value in image.items() if key != "blobs"}
                for image in old
            ],
            "archive": {
                "file": "images.tar.gz",
                "bytes": (output / "images.tar.gz").stat().st_size,
                "sha256": updates.digest(output / "images.tar.gz"),
            },
            "assets": additions,
            "omitted_existing_blob_bytes": omitted,
        }
        (output / "update.json").write_text(json.dumps(manifest, indent=2) + "\n")
    updates.read_bundle(output / "update.json")
    tools = Path(__file__).resolve().parent
    helper_hashes = {}
    for filename in (
        "deploy_modules.py", "docker_modules.py", "module_updates.py",
        "docker_target.py",
    ):
        target_name = "install.py" if filename == "deploy_modules.py" else filename
        shutil.copyfile(tools / filename, output / target_name)
        helper_hashes[target_name] = updates.digest(output / target_name)
    (output / "module-package.json").write_text(
        json.dumps(
            {
                "module": module,
                "version": version,
                "delivery": "image-update",
                "bbox_device": (
                    "cuda" if names[0].endswith(("-cuda", "-cuda-amd64")) else "cpu"
                ),
                "files": helper_hashes,
            },
            indent=2,
        )
        + "\n"
    )
    (output / "UPDATE.txt").write_text(
        "更新本模块\n"
        "python install.py update --bundle update.json\n"
        "回滚\npython install.py rollback\n"
        "Windows 可直接运行 update.cmd / rollback.cmd。\n"
        "沿用首次安装的 --namespace 和 --state-dir。默认安装无需附加参数。\n",
        encoding="utf-8",
    )
    for action in ("update", "rollback"):
        arguments = (
            'update --bundle "%~dp0update.json"' if action == "update" else action
        )
        (output / (action + ".cmd")).write_text(
            "@echo off\r\nsetlocal\r\n"
            "where py >nul 2>nul\r\n"
            "if %errorlevel%==0 (\r\n"
            f'  py -3 "%~dp0install.py" {arguments} %*\r\n'
            ") else (\r\n"
            f'  python "%~dp0install.py" {arguments} %*\r\n'
            ")\r\nexit /b %errorlevel%\r\n",
            encoding="ascii",
            newline="",
        )
    zipped = output.parent / (output.name + ".zip")
    partial = zipped.with_suffix(".zip.partial")
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(output).as_posix())
    partial.replace(zipped)
    print(
        json.dumps(
            {
                "module": module,
                "package": str(zipped),
                "bytes": zipped.stat().st_size,
                "omitted_existing_blob_bytes": omitted,
            }
        ),
        flush=True,
    )
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module", choices=spec.MODULES)
    parser.add_argument("--version", required=True)
    parser.add_argument("--image", action="append", required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asset", action="append", default=[], metavar="TARGET=FILE")
    args = parser.parse_args()
    assets = [
        (value.split("=", 1)[0], Path(value.split("=", 1)[1])) for value in args.asset
    ]
    package(args.module, args.version, args.image, args.output, args.baseline, assets)


if __name__ == "__main__":
    main()
