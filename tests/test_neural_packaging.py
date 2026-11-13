import hashlib
import io
import json
import shutil
import tarfile
import zipfile
from types import SimpleNamespace

import pytest

from tools import package_neural_container as pack


def image_archive(path, name, layers):
    objects = {}

    def blob(data, media):
        checksum = hashlib.sha256(data).hexdigest()
        objects["blobs/sha256/" + checksum] = data
        return {"digest": "sha256:" + checksum, "size": len(data), "mediaType": media}

    config = blob(json.dumps({
        "os": "linux", "architecture": "amd64",
        "rootfs": {"type": "layers", "diff_ids": [str(len(x)) for x in layers]},
    }).encode(), "application/vnd.oci.image.config.v1+json")
    descriptors = [
        blob(x, "application/vnd.oci.image.layer.v1.tar+gzip") for x in layers
    ]
    manifest = blob(json.dumps({
        "schemaVersion": 2, "config": config, "layers": descriptors,
    }).encode(), "application/vnd.oci.image.manifest.v1+json")
    index = blob(json.dumps({
        "schemaVersion": 2, "manifests": [{
            **manifest, "platform": {"os": "linux", "architecture": "amd64"},
        }],
    }).encode(), "application/vnd.oci.image.index.v1+json")
    index["annotations"] = {"io.containerd.image.name": "docker.io/library/" + name}
    objects["index.json"] = json.dumps({
        "schemaVersion": 2, "manifests": [index],
    }).encode()
    objects["oci-layout"] = b'{"imageLayoutVersion":"1.0.0"}'
    with tarfile.open(path, "w") as archive:
        for filename, data in objects.items():
            member = tarfile.TarInfo(filename)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    return pack.archive_images(path)[0], config["digest"]


def fixture(tmp_path, monkeypatch):
    shared = b"existing cuda dependency"
    old, old_config = image_archive(tmp_path / "base.tar", pack.BASE_NAME, [shared])
    new_tar = tmp_path / "neural.tar"
    new_name = "goodvision-neural:v2.0-test"
    new, _ = image_archive(new_tar, new_name, [shared, b"dedicated healthcheck"])
    monkeypatch.setattr(pack, "BASE_ID", old["id"])
    monkeypatch.setattr(pack, "BASE_CONFIG_ID", old_config)
    baseline = tmp_path / "system1-v2.0.zip"
    with zipfile.ZipFile(baseline, "w") as archive:
        archive.writestr("update.json", json.dumps({"images": {"system1": old}}))
    root = tmp_path / "repo"
    for relative in pack.SOURCE_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("release source " + relative)
    cmd = root / "tools/neural_delivery/Install-Neural.cmd"
    cmd.parent.mkdir(parents=True, exist_ok=True)
    cmd.write_text("@echo off\n")
    config = {
        "Labels": {"com.goodvision.module": "goodvision-neural"},
        "Entrypoint": ["python", "-m", "goodvision_harness.neural_service"],
        "Cmd": ["--host", "0.0.0.0", "--port", "8098", "--cache-dir",
                "/models/huggingface"],
        "Healthcheck": {"Test": ["CMD", "python",
                                 "/opt/goodvision-neural/healthcheck.py"]},
    }
    base_info = {"Id": old["id"], "Os": "linux", "Architecture": "amd64",
                 "RootFS": {"Layers": ["base"]}}
    new_info = {"Id": new["id"], "Os": "linux", "Architecture": "amd64",
                "RootFS": {"Layers": ["base", "healthcheck"]}, "Config": config}
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[0] == "git":
            return SimpleNamespace(stdout="8f52692\n")
        if command[1:3] == ["image", "inspect"]:
            value = base_info if command[-1] == pack.BASE_NAME else new_info
            return SimpleNamespace(stdout=json.dumps([value]))
        assert command[1:3] == ["image", "save"]
        shutil.copyfile(new_tar, command[command.index("--output") + 1])
        return SimpleNamespace(stdout="")

    return SimpleNamespace(
        root=root, baseline=baseline, name=new_name, new=new, base=old,
        new_info=new_info, new_tar=new_tar, run=run, calls=calls, shared=shared,
    )


def test_delivery_omits_only_verified_baseline_blobs_and_records_sources(
    tmp_path, monkeypatch,
):
    f = fixture(tmp_path, monkeypatch)
    output = tmp_path / "delivery"
    validation = tmp_path / "validation.json"
    validation.write_text('{"gpu_inference_tested": false}\n')
    release = pack.package(f.name, f.baseline, output, root=f.root, run=f.run,
                           validation=validation)
    with tarfile.open(output / "images.tar.gz") as archive:
        names = archive.getnames()
    assert "blobs/sha256/" + hashlib.sha256(f.shared).hexdigest() not in names
    assert ("blobs/sha256/" + hashlib.sha256(b"dedicated healthcheck").hexdigest()
            in names)
    assert release["requires"][0]["id"] == f.base["id"]
    assert release["image"]["id"] == f.new["id"]
    assert release["image_archive"]["requires_containerd_image_store"] is True
    assert release["source"]["application_update"] is False
    assert release["customer_deployment_status"] == "unknown"
    assert (output / "VALIDATION.json").read_bytes() == validation.read_bytes()
    assert release["validation"] == {
        "file": "VALIDATION.json", "bytes": validation.stat().st_size,
        "sha256": pack.updates.digest(validation),
    }
    for entry in release["source_files"]:
        source = output / entry["file"]
        assert source.stat().st_size == entry["bytes"]
        assert pack.updates.digest(source) == entry["sha256"]
    with zipfile.ZipFile(tmp_path / "delivery.zip") as archive:
        assert {"Install-Neural.cmd", "README.md", "RELEASE.json",
                "neural_container.py", "VALIDATION.json"}.issubset(archive.namelist())
        assert "source/workers/neural/Dockerfile" in archive.namelist()
    assert all("load" not in call and "run" not in call for call in f.calls)


@pytest.mark.parametrize("wrong", ["layers", "entrypoint", "platform"])
def test_rejects_a_different_application_base_or_service(tmp_path, monkeypatch, wrong):
    f = fixture(tmp_path, monkeypatch)
    if wrong == "layers":
        f.new_info["RootFS"]["Layers"] = ["upgraded base", "healthcheck"]
    elif wrong == "entrypoint":
        f.new_info["Config"]["Entrypoint"] = ["python", "videojev_service"]
    else:
        f.new_info["Architecture"] = "arm64"
    with pytest.raises(ValueError):
        pack.package(f.name, f.baseline, tmp_path / "delivery", root=f.root, run=f.run)
    assert not (tmp_path / "delivery").exists()
    assert all(call[1:3] != ["image", "save"] for call in f.calls)


def test_rejects_an_export_with_an_extra_import_tag(tmp_path, monkeypatch):
    f = fixture(tmp_path, monkeypatch)
    with tarfile.open(f.new_tar) as archive:
        objects = {m.name: archive.extractfile(m).read() for m in archive}
    index = json.loads(objects["index.json"])
    extra = json.loads(json.dumps(index["manifests"][0]))
    extra["annotations"]["io.containerd.image.name"] = "docker.io/library/other:tag"
    index["manifests"].append(extra)
    objects["index.json"] = json.dumps(index).encode()
    with tarfile.open(f.new_tar, "w") as archive:
        for filename, data in objects.items():
            member = tarfile.TarInfo(filename)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    with pytest.raises(ValueError, match="only the selected neural image tag"):
        pack.package(f.name, f.baseline, tmp_path / "delivery", root=f.root, run=f.run)
    assert not (tmp_path / "delivery").exists()


@pytest.mark.parametrize("existing", ["folder", "zip"])
def test_existing_delivery_is_never_replaced(tmp_path, monkeypatch, existing):
    f = fixture(tmp_path, monkeypatch)
    output = tmp_path / "delivery"
    if existing == "folder":
        output.mkdir()
    else:
        (tmp_path / "delivery.zip").write_bytes(b"delivered zip")
    with pytest.raises(ValueError, match="overwrite"):
        pack.package(f.name, f.baseline, output, root=f.root, run=f.run)
    assert f.calls == []


@pytest.mark.parametrize("content", ["not json", "[]"])
def test_invalid_validation_is_rejected_before_export(tmp_path, monkeypatch, content):
    f = fixture(tmp_path, monkeypatch)
    validation = tmp_path / "validation.json"
    validation.write_text(content)
    with pytest.raises(ValueError, match="JSON object"):
        pack.package(f.name, f.baseline, tmp_path / "delivery", root=f.root,
                     run=f.run, validation=validation)
    assert f.calls == []
    assert not (tmp_path / "delivery").exists()
