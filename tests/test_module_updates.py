import copy
import io
import json
import subprocess
import tarfile
import zipfile
from types import SimpleNamespace

import pytest

from tools import deploy_modules as deploy
from tools import docker_modules as spec
from tools import module_updates as updates


def oci_archive(images, *, extra=()):
    # Metadata-only fixture: image load/inspect is supplied by the Docker double.
    index = {"schemaVersion": 2, "manifests": [
        {"digest": image["id"], "annotations": {
            "io.containerd.image.name": "docker.io/library/" + image["name"],
        }} for image in [*images.values(), *extra]
    ]}
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w:gz") as archive:
        for name, value in [("index.json", index),
                            ("oci-layout", {"imageLayoutVersion": "1.0.0"})]:
            data = json.dumps(value).encode()
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    return raw.getvalue()


def replace_archive(path, manifest, data):
    archive = path.parent / manifest["archive"]["file"]
    archive.write_bytes(data)
    manifest["archive"].update(bytes=len(data), sha256=updates.digest(archive))
    deploy.write_json(path, manifest)


class Docker:
    def __init__(self, images, release):
        self.images = dict(images)
        self.release = release
        self.calls = []
        self.fail_pending = False

    def __call__(self, command, **options):
        self.calls.append(command)
        if command[1:3] == ["image", "inspect"]:
            if command[-1] not in self.images:
                raise subprocess.CalledProcessError(1, command)
            value = [
                {"Id": self.images[command[-1]], "Os": "linux", "Architecture": "amd64"}
            ]
            return SimpleNamespace(stdout=json.dumps(value), returncode=0)
        if command[1:3] == ["image", "load"]:
            self.images.update({v["name"]: v["id"] for v in self.release.values()})
        if command[1] == "compose" and "up" in command:
            compose = command[command.index("-f") + 1]
            if self.fail_pending and compose.endswith(".pending.json"):
                raise subprocess.CalledProcessError(1, command)
        return SimpleNamespace(stdout="", returncode=0)


def fixture(tmp_path, module="main"):
    state = tmp_path / "state"
    settings = spec.defaults("site", "v1")
    deploy.write_json(state / "settings.json", settings)
    old = (
        spec.compose(module, settings, tmp_path, state)
        if module != "gpu-runtime"
        else None
    )
    if old:
        deploy.write_json(state / f"compose.{module}.json", old)
        previous = {
            service["image"]: "sha256:" + str(i + 1) * 64
            for i, service in enumerate(old["services"].values())
        }
    else:
        previous = {
            spec.runtime_image(settings, version): "sha256:" + str(i + 1) * 64
            for i, version in enumerate(spec.TORCH)
        }
    settings2 = spec.defaults("site", "v2")
    if module == "gpu-runtime":
        release = {
            "torch" + version: {
                "name": spec.runtime_image(settings2, version),
                "id": "sha256:" + str(i + 3) * 64,
            }
            for i, version in enumerate(spec.TORCH)
        }
    else:
        next_doc = spec.compose(module, settings2, tmp_path, state)
        release = {
            role: {"name": service["image"], "id": "sha256:" + str(i + 3) * 64}
            for i, (role, service) in enumerate(next_doc["services"].items())
        }
    folder = tmp_path / "release"
    folder.mkdir()
    image = folder / "images.tar.gz"
    image.write_bytes(oci_archive(release))
    manifest = {
        "schema": 1,
        "module": module,
        "version": "v2",
        "images": release,
        "requires": [{"name": name, "id": value} for name, value in previous.items()],
        "archive": {
            "file": image.name,
            "bytes": image.stat().st_size,
            "sha256": updates.digest(image),
        },
    }
    path = folder / "update.json"
    deploy.write_json(path, manifest)
    return state, settings, old, path, manifest, Docker(previous, release)


def test_update_and_rollback_change_only_the_selected_module(tmp_path):
    state, settings, old, path, _, docker = fixture(tmp_path)
    other = state / "compose.bbox.json"
    other.write_bytes(b"other module bytes stay unchanged")
    key = state / "secrets/node-key"
    key.parent.mkdir()
    key.write_bytes(b"existing private fixture")
    updates.update(state, path, settings, 30, docker, deploy.write_json)
    changed = json.loads((state / "compose.main.json").read_text())
    expected = copy.deepcopy(old)
    expected["services"]["main"]["image"] = "goodvision-main:v2"
    assert changed == expected
    assert settings["module_versions"] == {
        **spec.defaults("site", "v1")["module_versions"],
        "main": "v2",
    }
    assert (
        spec.compose("main", settings, tmp_path, state)["services"]["main"]["image"]
        == "goodvision-main:v2"
    )
    assert other.read_bytes() == b"other module bytes stay unchanged"
    assert key.read_bytes() == b"existing private fixture"
    assert all("build" not in call and "down" not in call for call in docker.calls)
    updates.rollback(state, "main", settings, 30, docker, deploy.write_json)
    assert json.loads((state / "compose.main.json").read_text()) == old
    assert settings["module_versions"]["main"] == "v1"
    assert other.read_bytes() == b"other module bytes stay unchanged"


def test_health_failure_automatically_restores_previous_module(tmp_path):
    state, settings, old, path, _, docker = fixture(tmp_path)
    before = (state / "settings.json").read_bytes()
    docker.fail_pending = True
    with pytest.raises(ValueError, match="previous module was restored"):
        updates.update(state, path, settings, 30, docker, deploy.write_json)
    assert json.loads((state / "compose.main.json").read_text()) == old
    assert (state / "settings.json").read_bytes() == before
    ups = [call for call in docker.calls if call[1] == "compose" and "up" in call]
    assert len(ups) == 2
    assert ups[-1][ups[-1].index("-f") + 1] == str(state / "compose.main.json")


def test_wrong_baseline_and_corrupt_archive_do_not_restart_anything(tmp_path):
    state, settings, _, path, manifest, docker = fixture(tmp_path)
    manifest["requires"][0]["id"] = "sha256:" + "f" * 64
    deploy.write_json(path, manifest)
    with pytest.raises(ValueError, match="Required image differs"):
        updates.update(state, path, settings, 30, docker, deploy.write_json)
    assert not any(call[1] == "compose" for call in docker.calls)
    docker.calls.clear()
    (path.parent / "images.tar.gz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        updates.update(state, path, settings, 30, docker, deploy.write_json)
    assert docker.calls == []


def test_existing_release_tag_cannot_be_overwritten(tmp_path):
    state, settings, _, path, _, docker = fixture(tmp_path)
    docker.images["goodvision-main:v2"] = "sha256:" + "e" * 64
    with pytest.raises(ValueError, match="left unchanged"):
        updates.update(state, path, settings, 30, docker, deploy.write_json)
    assert not any(call[1:3] == ["image", "load"] for call in docker.calls)


def test_repeating_current_update_keeps_previous_rollback_even_without_old_cache(
    tmp_path,
):
    state, settings, _, path, manifest, docker = fixture(tmp_path)
    updates.update(state, path, settings, 30, docker, deploy.write_json)
    before = (state / "rollback.main.json").read_bytes()
    del docker.images[manifest["requires"][0]["name"]]
    updates.update(state, path, settings, 30, docker, deploy.write_json)
    assert (state / "rollback.main.json").read_bytes() == before


def test_contour_update_can_reuse_unchanged_engine_image(tmp_path):
    state, settings, old, path, manifest, docker = fixture(tmp_path, "contour")
    name = old["services"]["engine"]["image"]
    manifest["images"]["engine"] = {"name": name, "id": docker.images[name]}
    docker.release = manifest["images"]
    replace_archive(path, manifest, oci_archive(manifest["images"]))
    updates.update(state, path, settings, 30, docker, deploy.write_json)
    saved = spec.compose("contour", settings, tmp_path, state)
    assert saved["services"]["engine"]["image"] == name
    assert saved["services"]["contour"]["image"] == "goodvision-contour:v2-api"


def test_gpu_runtime_update_and_rollback_start_no_containers(tmp_path):
    state, settings, _, path, _, docker = fixture(tmp_path, "gpu-runtime")
    updates.update(state, path, settings, 30, docker, deploy.write_json)
    assert (
        spec.runtime_image(settings, "2.8")
        == "goodvision-gpu-runtime:v2-torch2.8-cu128"
    )
    updates.rollback(state, "gpu-runtime", settings, 30, docker, deploy.write_json)
    assert (
        spec.runtime_image(settings, "2.8")
        == "goodvision-gpu-runtime:v1-torch2.8-cu128"
    )
    assert not any(call[1] == "compose" for call in docker.calls)


def test_model_additions_never_replace_existing_weights(tmp_path):
    state, settings, _, path, manifest, docker = fixture(tmp_path, "system1")
    source = path.parent / "person.pt"
    source.write_bytes(b"new model fixture")
    manifest["assets"] = [
        {
            "file": "person.pt",
            "target": "yolo26s.pt",
            "bytes": source.stat().st_size,
            "sha256": updates.digest(source),
        }
    ]
    deploy.write_json(path, manifest)
    updates.update(state, path, settings, 30, docker, deploy.write_json)
    target = state / "models/system1/yolo26s.pt"
    assert target.read_bytes() == source.read_bytes()
    target.write_bytes(b"user's existing custom model")
    with pytest.raises(ValueError, match="left unchanged"):
        updates.update(state, path, settings, 30, docker, deploy.write_json)
    assert target.read_bytes() == b"user's existing custom model"


def test_zip_update_and_cli_dry_run_are_independent_of_source_checkout(
    tmp_path, monkeypatch
):
    state, _, _, path, _, docker = fixture(tmp_path)
    zipped = tmp_path / "goodvision-main-v2.zip"
    with zipfile.ZipFile(zipped, "w") as archive:
        archive.write(path, "update.json")
        archive.write(path.parent / "images.tar.gz", "images.tar.gz")
    monkeypatch.setattr(
        deploy, "location", lambda: (tmp_path, tmp_path / "missing-source")
    )
    monkeypatch.setattr(deploy, "run", docker)
    before = (state / "settings.json").read_bytes()
    args = [
        "update",
        "--bundle",
        str(zipped),
        "--namespace",
        "site",
        "--state-dir",
        str(state),
    ]
    assert deploy.main([*args, "--dry-run"]) == 0
    assert docker.calls == [] and (state / "settings.json").read_bytes() == before
    assert deploy.main(args) == 0
    assert (
        json.loads((state / "settings.json").read_text())["module_versions"]["main"]
        == "v2"
    )
