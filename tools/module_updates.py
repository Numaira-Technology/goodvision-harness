"""Apply one prebuilt module release without builds, downloads or data resets."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from contextlib import contextmanager
from functools import wraps
from pathlib import Path, PurePosixPath

if __package__:
    from . import docker_modules as spec
else:
    import docker_modules as spec

ROLES = spec.IMAGE_ROLES


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(
                    lambda stream=stream: stream.read(8 * 1024 * 1024), b"",
                ):
            result.update(chunk)
    return result.hexdigest()


def relative_file(root, name):
    value = PurePosixPath(name)
    if value.is_absolute() or ".." in value.parts or "\\" in name or ":" in name:
        raise ValueError("Update package contains an invalid path")
    path = root.joinpath(*value.parts)
    if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Update package is incomplete")
    return path


def read_bundle(path):
    if path.suffix.lower() == ".zip":
        try:
            with zipfile.ZipFile(path) as archive:
                if archive.getinfo("update.json").file_size > 1024 * 1024:
                    raise ValueError("Update manifest is too large")
                bundle = json.loads(archive.read("update.json"))
        except (zipfile.BadZipFile, KeyError) as error:
            raise ValueError("Invalid module update ZIP") from error
    else:
        bundle = json.loads(path.read_text(encoding="utf-8"))
    module = bundle["module"]
    if bundle.get("schema") == 2:
        if module != "bbox" or not isinstance(bundle.get("variants"), dict):
            raise ValueError("Unsupported module variant package")
        spec.name(bundle["version"])
        supported = {"linux/amd64/cpu", "linux/arm64/cpu", "linux/amd64/cuda"}
        if not bundle["variants"] or set(bundle["variants"]) - supported:
            raise ValueError("Unsupported Bbox variant")
        for entry in bundle["variants"].values():
            if (type(entry.get("bytes")) is not int or entry["bytes"] <= 0
                    or not re.fullmatch(r"[a-f0-9]{64}", entry.get("sha256", ""))):
                raise ValueError("Invalid variant archive metadata")
        for ref in bundle.get("gpu_probe_images", []):
            if not re.fullmatch(
                r"goodvision-(bbox|gpu-runtime):[a-zA-Z0-9_][a-zA-Z0-9_.-]{0,127}", ref,
            ):
                raise ValueError("Invalid GPU probe image")
        return bundle
    if module not in ROLES or bundle.get("schema") != 1:
        raise ValueError("Unsupported module update package")
    spec.name(bundle["version"])
    if set(bundle["images"]) != ROLES[module]:
        raise ValueError("Update package does not match its module roles")
    for role, entry in bundle["images"].items():
        if spec.image_role(module, entry) != role:
            raise ValueError("Image name does not match its module role")
        if not entry["name"].startswith(f"goodvision-{module}:") or not re.fullmatch(
            r"sha256:[a-f0-9]{64}", entry["id"]
        ):
            raise ValueError("Update image has an invalid module name or ID")
    for entry in bundle.get("requires", []):
        if not entry["name"].startswith(f"goodvision-{module}:"):
            raise ValueError("Update baseline belongs to another module")
    return bundle


@contextmanager
def bundle_files(path):
    if path.suffix.lower() != ".zip":
        yield path
        return
    bundle = read_bundle(path)
    needed = [bundle["archive"], *bundle.get("assets", [])]
    with tempfile.TemporaryDirectory(prefix="goodvision-update-") as directory:
        root = Path(directory)
        if (
            shutil.disk_usage(root).free
            < sum(x["bytes"] for x in needed) + 128 * 1024**2
        ):
            raise ValueError("Insufficient space to unpack the module update")
        try:
            with zipfile.ZipFile(path) as archive:
                (root / "update.json").write_bytes(archive.read("update.json"))
                for entry in needed:
                    name = entry["file"]
                    value = PurePosixPath(name)
                    if (
                        value.is_absolute()
                        or ".." in value.parts
                        or "\\" in name
                        or ":" in name
                    ):
                        raise ValueError("Invalid path in update ZIP")
                    info = archive.getinfo(name)
                    if (
                        info.file_size != entry["bytes"]
                        or (info.external_attr >> 16) & 0o170000 == 0o120000
                    ):
                        raise ValueError("Invalid file in update ZIP")
                    target = root.joinpath(*value.parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(info) as source, target.open("wb") as destination:
                        shutil.copyfileobj(source, destination)
            yield root / "update.json"
        except (zipfile.BadZipFile, KeyError) as error:
            raise ValueError("Incomplete or corrupt module update ZIP") from error


def state_transaction(function):
    @wraps(function)
    def locked(state, subject, settings, *args, **kwargs):
        with spec.state_lock(state) as outermost:
            if outermost:
                # Direct API callers can hold a snapshot predating another update.
                # CLI overrides were already applied inside the same lock.
                saved = json.loads((state / "settings.json").read_text())
                settings.clear()
                settings.update(saved)
            return function(state, subject, settings, *args, **kwargs)
    return locked


def validate_archive(path, images):
    """Reject undeclared import names before Docker can mutate any local tag.

    Incremental bundles may omit existing blobs, but every included blob is
    content-addressed and the complete import index must match the manifest.
    Legacy Docker manifests are not part of our OCI-only update format.
    """
    expected = {image["name"]: image["id"] for image in images.values()}
    if len(expected) != len(images):
        raise ValueError("Duplicate image name in update manifest")
    index, seen = None, set()
    try:
        with tarfile.open(path, "r:*") as archive:
            for member in archive:
                name = member.name
                if name in seen or not member.isfile():
                    raise ValueError("Duplicate or non-file OCI archive member")
                seen.add(name)
                stream = archive.extractfile(member)
                if name in ("index.json", "oci-layout"):
                    if member.size > 1024 * 1024:
                        raise ValueError("OCI metadata is too large")
                    value = json.load(stream)
                    if name == "index.json":
                        index = value
                    continue
                if not re.fullmatch(r"blobs/sha256/[a-f0-9]{64}", name):
                    raise ValueError("Unexpected OCI archive member")
                checksum, small = hashlib.sha256(), bytearray()
                for chunk in iter(
                    lambda stream=stream: stream.read(8 * 1024 * 1024), b"",
                ):
                    checksum.update(chunk)
                    if member.size <= 2 * 1024 * 1024:
                        small.extend(chunk)
                if checksum.hexdigest() != name.rsplit("/", 1)[1]:
                    raise ValueError("OCI blob digest mismatch")
                if small:
                    try:
                        value = json.loads(small)
                    except (ValueError, UnicodeError):
                        continue
                    if isinstance(value, dict):
                        for child in value.get("manifests", []):
                            annotations = child.get("annotations", {})
                            if any(key in annotations for key in (
                                "io.containerd.image.name",
                                "org.opencontainers.image.ref.name",
                            )):
                                raise ValueError(
                                    "Nested OCI image alias is not allowed"
                                )
    except (tarfile.TarError, ValueError, KeyError, TypeError) as error:
        raise ValueError("Invalid OCI update archive: " + str(error)) from error
    if index is None or "oci-layout" not in seen:
        raise ValueError("An OCI image archive is required")
    found = {}
    for entry in index.get("manifests", []):
        annotations = entry.get("annotations", {})
        name = annotations.get("io.containerd.image.name", "").removeprefix(
            "docker.io/library/",
        )
        alias = annotations.get("org.opencontainers.image.ref.name", name)
        if (name not in expected or name in found
                or alias not in (name, "docker.io/library/" + name,
                                 name.split(":", 1)[-1])):
            raise ValueError("Undeclared or duplicate OCI image alias")
        found[name] = entry["digest"]
    if found != expected:
        raise ValueError("OCI archive identities differ from update manifest")


def inspect_image(name, run):
    value = run(["docker", "image", "inspect", name], capture_output=True, text=True)
    return json.loads(value.stdout)[0]


def check_identity(entry, run):
    info = inspect_image(entry["name"], run)
    if info["Id"] != entry["id"]:
        raise ValueError(
            "Required image differs from the verified release: " + entry["name"]
        )
    for key, field in (("os", "Os"), ("architecture", "Architecture")):
        if key in entry and info[field] != entry[key]:
            raise ValueError("Update image platform mismatch")


def up(state, module, path, timeout, run):
    command = ["docker", "compose", "--project-directory", str(state), "-f", str(path)]
    run([*command, "config", "--quiet"])
    run(
        [
            *command,
            "up",
            "-d",
            "--no-build",
            "--pull",
            "never",
            "--wait",
            "--wait-timeout",
            str(timeout),
        ]
    )


def apply_config(
    state,
    module,
    document,
    version,
    settings,
    timeout,
    run,
    write_json,
    runtime_images=None,
    module_settings=None,
    previous_images=None,
):
    active = state / f"compose.{module}.json"
    saved_settings = json.loads((state / "settings.json").read_text())
    before = json.loads(active.read_text()) if module != "gpu-runtime" else None
    old_version = settings.get("module_versions", {}).get(module, settings["version"])
    if previous_images is None:
        previous_images = []
        if before:
            for service in before["services"].values():
                info = inspect_image(service["image"], run)
                previous_images.append({"name": service["image"], "id": info["Id"]})
        else:
            for torch in spec.TORCH:
                image = spec.runtime_image(settings, torch)
                info = inspect_image(image, run)
                previous_images.append({"name": image, "id": info["Id"]})
    old_pins = (
        {key: service["image"] for key, service in before["services"].items()}
        if before
        else {"torch" + key: spec.runtime_image(settings, key) for key in spec.TORCH}
    )
    backup = {
        "module": module,
        "version": old_version,
        "compose": before,
        "images": previous_images,
        "module_images": old_pins,
    }
    if module == "bbox":
        backup["module_settings"] = {
            key: saved_settings.get(key) for key in
            ("bbox_device", "bbox_runtime_device", "bbox_platform")
        }
    backup_path = state / f"rollback.{module}.json"
    previous_backup = (
        json.loads(backup_path.read_text()) if backup_path.is_file() else None
    )
    versions = {item: settings["version"] for item in spec.MODULES}
    versions.update(settings.get("module_versions", {}))
    versions[module] = version
    pins = copy.deepcopy(settings.get("module_images", {}))
    pins[module] = (
        {key: service["image"] for key, service in document["services"].items()}
        if document
        else runtime_images
    )
    updated = {**settings, "module_versions": versions, "module_images": pins}
    for key, value in (module_settings or {}).items():
        if value is None:
            updated.pop(key, None)
        else:
            updated[key] = value
    write_json(backup_path, backup)
    pending = state / f"compose.{module}.pending.json"
    if document is not None:
        write_json(pending, document)
        try:
            up(state, module, pending, timeout, run)
            write_json(state / "settings.json", updated)
            pending.replace(active)
        except (subprocess.SubprocessError, OSError) as error:
            try:
                write_json(state / "settings.json", saved_settings)
                up(state, module, active, timeout, run)
                if previous_backup is None:
                    backup_path.unlink(missing_ok=True)
                else:
                    write_json(backup_path, previous_backup)
                settings.clear()
                settings.update(saved_settings)
            except (subprocess.SubprocessError, OSError) as recovery:
                raise ValueError(
                    "Module update and automatic recovery failed; "
                    "the previous configuration is retained"
                ) from recovery
            raise ValueError(
                "Module update failed; the previous module was restored"
            ) from error
    else:
        write_json(state / "settings.json", updated)
    settings.clear()
    settings.update(updated)


def stage_assets(state, module, folder, assets):
    for asset in assets:
        source = relative_file(folder, asset["file"])
        if source.stat().st_size != asset["bytes"] or digest(source) != asset["sha256"]:
            raise ValueError("Model addition checksum mismatch")
        relative = PurePosixPath(asset["target"])
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or "\\" in asset["target"]
            or ":" in asset["target"]
        ):
            raise ValueError("Invalid model addition path")
        base = (state / "models" / module).resolve()
        target = base.joinpath(*relative.parts)
        if not target.resolve().is_relative_to(base):
            raise ValueError("Model addition escapes the module data directory")
        if target.exists():
            if target.is_file() and digest(target) == asset["sha256"]:
                continue
            raise ValueError("Existing model differs; it was left unchanged")
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(dir=target.parent)
        temporary = Path(temporary)
        try:
            with os.fdopen(descriptor, "wb") as stream, source.open("rb") as original:
                shutil.copyfileobj(original, stream)
            temporary.chmod(0o644)
            os.link(temporary, target)  # Atomically publish without replacing a file.
        finally:
            temporary.unlink(missing_ok=True)


def update_files(state, path, settings, timeout, run, write_json, target=None):
    bundle = read_bundle(path)
    module, folder = bundle["module"], path.parent
    if not (state / "settings.json").is_file():
        raise ValueError("Install this module before applying its update package")
    active = state / f"compose.{module}.json"
    if module != "gpu-runtime" and not active.is_file():
        raise ValueError("Install this module before applying its update package")
    archive = relative_file(folder, bundle["archive"]["file"])
    if (
        archive.stat().st_size != bundle["archive"]["bytes"]
        or digest(archive) != bundle["archive"]["sha256"]
    ):
        raise ValueError("Update image archive checksum mismatch")
    validate_archive(archive, bundle["images"])
    run(["docker", "info"], stdout=subprocess.DEVNULL)
    # Release tags are immutable, so rollback never loses its old image bytes.
    present = True
    for entry in bundle["images"].values():
        try:
            info = inspect_image(entry["name"], run)
        except subprocess.CalledProcessError:
            present = False
            continue
        if info["Id"] != entry["id"]:
            raise ValueError("Update tag already names another image; left unchanged")
    desired = {role: entry["name"] for role, entry in bundle["images"].items()}
    current = (
        {"torch" + key: spec.runtime_image(settings, key) for key in spec.TORCH}
        if module == "gpu-runtime"
        else {
            role: service["image"]
            for role, service in json.loads(active.read_text())["services"].items()
        }
    )
    if present and desired == current:
        for entry in bundle["images"].values():
            check_identity(entry, run)
        stage_assets(state, module, folder, bundle.get("assets", []))
        if module != "gpu-runtime":
            up(state, module, active, timeout, run)
        versions = {key: settings["version"] for key in spec.MODULES}
        versions.update(settings.get("module_versions", {}))
        versions[module] = bundle["version"]
        settings["module_versions"] = versions
        settings.setdefault("module_images", {})[module] = desired
        if target:
            settings.update(target)
        write_json(state / "settings.json", settings)
        print(f"goodvision-{module} is already current.")
        return
    for entry in bundle.get("requires", []):
        check_identity(entry, run)
    previous_images = [
        {"name": name, "id": inspect_image(name, run)["Id"]}
        for name in current.values()
    ]
    stage_assets(state, module, folder, bundle.get("assets", []))
    run(["docker", "image", "load", "--input", str(archive)])
    for entry in previous_images:
        check_identity(entry, run)
    for entry in bundle["images"].values():
        check_identity(entry, run)
    document = None
    if module != "gpu-runtime":
        document = copy.deepcopy(json.loads(active.read_text()))
        if set(document["services"]) != ROLES[module]:
            raise ValueError("Saved Compose services do not match the update module")
        for role, entry in bundle["images"].items():
            document["services"][role]["image"] = entry["name"]
        if target:
            service = document["services"]["bbox"]
            service["platform"] = target["bbox_platform"]
            service["healthcheck"] = spec.bbox_health(target["bbox_runtime_device"])
            service.setdefault("environment", {})["DETECTOR_DEVICE"] = (
                target["bbox_runtime_device"]
            )
            if target["bbox_runtime_device"] == "cuda":
                devices = spec.gpu(settings)["resources"]["reservations"]["devices"]
                service.setdefault("deploy", {}).setdefault("resources", {}).setdefault(
                    "reservations", {},
                )["devices"] = devices
            else:
                reservations = service.get("deploy", {}).get("resources", {}).get(
                    "reservations", {},
                )
                reservations.pop("devices", None)
    apply_config(
        state,
        module,
        document,
        bundle["version"],
        settings,
        timeout,
        run,
        write_json,
        {key: image["name"] for key, image in bundle["images"].items()},
        target,
        previous_images,
    )
    print(f"Updated goodvision-{module}; other modules and persisted data retained.")


@state_transaction
def update(state, path, settings, timeout, run, write_json):
    bundle = read_bundle(path)
    if bundle["schema"] == 2:
        if __package__:
            from .docker_target import select_bbox
        else:
            from docker_target import select_bbox
        selected = select_bbox(settings, run, probes=bundle.get("gpu_probe_images", []))
        key = selected["platform"] + "/" + selected["device"]
        if key not in bundle["variants"]:
            raise ValueError("This release does not contain the selected Bbox variant")
        entry = bundle["variants"][key]
        target = {"bbox_device": selected["policy"],
                  "bbox_runtime_device": selected["device"],
                  "bbox_platform": selected["platform"]}
        print("Bbox target: " + json.dumps(selected))
        with tempfile.TemporaryDirectory(prefix="goodvision-variant-") as directory:
            root = Path(directory)
            if shutil.disk_usage(root).free < entry["bytes"] * 2 + 128 * 1024**2:
                raise ValueError("Insufficient space to unpack the selected variant")
            if path.suffix.lower() == ".zip":
                relative = PurePosixPath(entry["file"])
                if (relative.is_absolute() or ".." in relative.parts
                        or "\\" in entry["file"] or ":" in entry["file"]):
                    raise ValueError("Invalid variant path")
                with zipfile.ZipFile(path) as archive:
                    info = archive.getinfo(entry["file"])
                    if info.file_size != entry["bytes"]:
                        raise ValueError("Variant archive size mismatch")
                    selected_zip = root / "selected.zip"
                    with archive.open(info) as original, selected_zip.open("wb") as out:
                        shutil.copyfileobj(original, out)
            else:
                selected_zip = relative_file(path.parent, entry["file"])
            if (selected_zip.stat().st_size != entry["bytes"]
                    or digest(selected_zip) != entry["sha256"]):
                raise ValueError("Variant archive checksum mismatch")
            inner = read_bundle(selected_zip)
            if (inner["schema"] != 1 or inner["module"] != "bbox"
                    or inner["version"] != bundle["version"]):
                raise ValueError("Selected variant belongs to another release")
            for image in inner["images"].values():
                if f"{image['os']}/{image['architecture']}" != selected["platform"]:
                    raise ValueError("Selected variant image platform mismatch")
            with bundle_files(selected_zip) as manifest:
                update_files(
                    state, manifest, settings, timeout, run, write_json, target,
                )
        return
    with bundle_files(path) as manifest:
        update_files(state, manifest, settings, timeout, run, write_json)


@state_transaction
def rollback(state, module, settings, timeout, run, write_json):
    path = state / f"rollback.{module}.json"
    before = json.loads(path.read_text())
    if before["module"] != module:
        raise ValueError("Rollback record belongs to another module")
    for entry in before["images"]:
        check_identity(entry, run)
    apply_config(
        state,
        module,
        before["compose"],
        before["version"],
        settings,
        timeout,
        run,
        write_json,
        before["module_images"],
        before.get("module_settings"),
    )
    print(f"Rolled back goodvision-{module}; persisted data retained.")
