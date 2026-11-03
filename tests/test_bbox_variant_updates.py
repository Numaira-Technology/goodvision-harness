import copy
import hashlib
import io
import json
import subprocess
import tarfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_module_updates import fixture, oci_archive

from tools import deploy_modules as deploy
from tools import module_updates as updates


def release(tmp_path, arch="arm64", gpu=False):
    state, settings, old, _, basic, docker = fixture(tmp_path, "bbox")
    settings["bbox_device"] = "auto"
    deploy.write_json(state / "settings.json", settings)
    old["services"]["bbox"]["deploy"] = {"resources": {"limits": {"cpus": "1"}}}
    deploy.write_json(state / "compose.bbox.json", old)
    other = state / "compose.main.json"
    other.write_bytes(b"unrelated Main configuration")
    catalog = {"schema": 2, "module": "bbox", "version": "v2", "variants": {},
               "gpu_probe_images": ["goodvision-bbox:v1-cuda"]}
    docker.images["goodvision-bbox:v1-cuda"] = "sha256:" + "7" * 64
    images, payloads = {}, {}
    for i, key in enumerate(["linux/arm64/cpu", "linux/amd64/cpu", "linux/amd64/cuda"]):
        _, machine, device = key.split("/")
        image = {"name": f"goodvision-bbox:v2-{device}-{machine}",
                 "id": "sha256:" + str(i + 3) * 64,
                 "os": "linux", "architecture": machine}
        images[key] = image
        inner = copy.deepcopy(basic)
        inner["images"] = {"bbox": image}
        marker = oci_archive(inner["images"])
        inner["archive"].update(
            bytes=len(marker), sha256=hashlib.sha256(marker).hexdigest(),
        )
        raw = io.BytesIO()
        with zipfile.ZipFile(raw, "w") as archive:
            archive.writestr("update.json", json.dumps(inner))
            archive.writestr("images.tar.gz", marker)
        data = raw.getvalue()
        filename = f"variants/{device}-{machine}.zip"
        payloads[filename] = data
        catalog["variants"][key] = {"file": filename, "bytes": len(data),
                                    "sha256": hashlib.sha256(data).hexdigest()}
    path = tmp_path / "bbox.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("update.json", json.dumps(catalog))
        for name, data in payloads.items():
            archive.writestr(name, data)
    loaded = []

    def run(command, **options):
        if command[1] == "info" and "--format" in command:
            docker.calls.append(command)
            return SimpleNamespace(stdout=json.dumps({"OSType": "linux",
                                                       "Architecture": arch}))
        if command[1] == "run":
            docker.calls.append(command)
            if not gpu:
                raise subprocess.CalledProcessError(1, command)
            return SimpleNamespace(stdout="")
        if command[1:3] == ["image", "load"]:
            with tarfile.open(Path(command[-1])) as archive:
                index = json.load(archive.extractfile("index.json"))
            ref = index["manifests"][0]["annotations"]["io.containerd.image.name"]
            selected = next(key for key, image in images.items()
                            if ref.endswith(image["name"]))
            loaded.append(selected)
            docker.release = {"bbox": images[selected]}
        result = docker(command, **options)
        if command[1:3] == ["image", "inspect"]:
            value = json.loads(result.stdout)
            value[0]["Architecture"] = (
                "amd64" if "-cuda" in command[-1] else arch
            )
            result.stdout = json.dumps(value)
        return result

    return state, settings, old, path, catalog, run, docker, loaded


@pytest.mark.parametrize("arch,gpu,selected", [
    ("arm64", False, "linux/arm64/cpu"),
    ("amd64", False, "linux/amd64/cpu"),
    ("amd64", True, "linux/amd64/cuda"),
])
def test_one_zip_loads_only_the_selected_variant_and_rolls_back_hardware_settings(
    tmp_path, arch, gpu, selected,
):
    state, settings, old, path, _, run, _, loaded = release(tmp_path, arch, gpu)
    updates.update(state, path, settings, 30, run, deploy.write_json)
    assert loaded == [selected]
    changed = json.loads((state / "compose.bbox.json").read_text())["services"]["bbox"]
    assert changed["platform"] == "linux/" + arch
    assert changed["environment"]["DETECTOR_DEVICE"] == selected.split("/")[-1]
    assert changed["deploy"]["resources"]["limits"] == {"cpus": "1"}
    assert settings["bbox_platform"] == "linux/" + arch
    assert settings["bbox_runtime_device"] == selected.split("/")[-1]
    assert (state / "compose.main.json").read_bytes() == b"unrelated Main configuration"
    assert settings["module_versions"]["main"] == "v1"
    checkpoint = (state / "rollback.bbox.json").read_bytes()
    updates.update(state, path, settings, 30, run, deploy.write_json)
    assert loaded == [selected]
    assert (state / "rollback.bbox.json").read_bytes() == checkpoint
    updates.rollback(state, "bbox", settings, 30, run, deploy.write_json)
    assert json.loads((state / "compose.bbox.json").read_text()) == old
    assert "bbox_platform" not in settings and "bbox_runtime_device" not in settings
    assert settings["bbox_device"] == "auto"


def test_failed_variant_health_restores_previous_module_and_metadata(tmp_path):
    state, settings, old, path, _, run, docker, _ = release(tmp_path)
    previous = copy.deepcopy(settings)
    docker.fail_next_up = True
    with pytest.raises(ValueError, match="previous module was restored"):
        updates.update(state, path, settings, 30, run, deploy.write_json)
    assert settings == previous
    assert json.loads((state / "compose.bbox.json").read_text()) == old
    assert json.loads((state / "settings.json").read_text()) == previous


def test_selected_variant_checksum_failure_never_loads_or_changes_module(tmp_path):
    state, settings, old, path, catalog, run, _, loaded = release(tmp_path)
    catalog["variants"]["linux/arm64/cpu"]["sha256"] = "0" * 64
    folder = tmp_path / "catalog"
    with zipfile.ZipFile(path) as archive:
        archive.extractall(folder)
    deploy.write_json(folder / "update.json", catalog)
    with pytest.raises(ValueError, match="checksum"):
        updates.update(
            state, folder / "update.json", settings, 30, run, deploy.write_json,
        )
    assert not loaded
    assert json.loads((state / "compose.bbox.json").read_text()) == old
    assert not (state / "rollback.bbox.json").exists()
