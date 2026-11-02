"""Apply one prebuilt module release without builds, downloads or data resets."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

if __package__:
    from . import docker_modules as spec
else:
    import docker_modules as spec

ROLES = spec.IMAGE_ROLES


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
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
    if module not in ROLES or bundle.get("schema") != 1:
        raise ValueError("Unsupported module update package")
    spec.name(bundle["version"])
    if set(bundle["images"]) != ROLES[module]:
        raise ValueError("Update package does not match its module roles")
    for entry in bundle["images"].values():
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
):
    active = state / f"compose.{module}.json"
    before = json.loads(active.read_text()) if module != "gpu-runtime" else None
    old_version = settings.get("module_versions", {}).get(module, settings["version"])
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
                write_json(state / "settings.json", settings)
                up(state, module, active, timeout, run)
                if previous_backup is None:
                    backup_path.unlink(missing_ok=True)
                else:
                    write_json(backup_path, previous_backup)
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


def update_files(state, path, settings, timeout, run, write_json):
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
        write_json(state / "settings.json", settings)
        print(f"goodvision-{module} is already current.")
        return
    for entry in bundle.get("requires", []):
        check_identity(entry, run)
    for name in current.values():
        inspect_image(name, run)
    stage_assets(state, module, folder, bundle.get("assets", []))
    run(["docker", "image", "load", "--input", str(archive)])
    for entry in bundle["images"].values():
        check_identity(entry, run)
    document = None
    if module != "gpu-runtime":
        document = copy.deepcopy(json.loads(active.read_text()))
        if set(document["services"]) != ROLES[module]:
            raise ValueError("Saved Compose services do not match the update module")
        for role, entry in bundle["images"].items():
            document["services"][role]["image"] = entry["name"]
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
    )
    print(f"Updated goodvision-{module}; other modules and persisted data retained.")


def update(state, path, settings, timeout, run, write_json):
    with bundle_files(path) as manifest:
        update_files(state, manifest, settings, timeout, run, write_json)


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
    )
    print(f"Rolled back goodvision-{module}; persisted data retained.")
