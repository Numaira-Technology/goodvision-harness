import hashlib
import io
import json
import shutil
import tarfile
import zipfile

from tools import package_image_update as export


def image_archive(path, tag, layers):
    objects = {}

    def blob(data, media):
        digest = hashlib.sha256(data).hexdigest()
        objects["blobs/sha256/" + digest] = data
        return {"digest": "sha256:" + digest, "size": len(data), "mediaType": media}

    config = blob(
        json.dumps(
            {
                "os": "linux",
                "architecture": "amd64",
                "config": {"Env": ["A=B"]},
                "rootfs": {"type": "layers", "diff_ids": []},
            }
        ).encode(),
        "application/vnd.oci.image.config.v1+json",
    )
    descriptors = [
        blob(value, "application/vnd.oci.image.layer.v1.tar+gzip") for value in layers
    ]
    manifest = blob(
        json.dumps(
            {"schemaVersion": 2, "config": config, "layers": descriptors}
        ).encode(),
        "application/vnd.oci.image.manifest.v1+json",
    )
    image = blob(
        json.dumps({"schemaVersion": 2, "manifests": [manifest]}).encode(),
        "application/vnd.oci.image.index.v1+json",
    )
    image["annotations"] = {"io.containerd.image.name": "docker.io/library/" + tag}
    objects["index.json"] = json.dumps(
        {"schemaVersion": 2, "manifests": [image]}
    ).encode()
    objects["oci-layout"] = b'{"imageLayoutVersion":"1.0.0"}'
    with tarfile.open(path, "w") as archive:
        for name, data in objects.items():
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            archive.addfile(entry, io.BytesIO(data))
    return export.archive_images(path)


def test_image_graph_does_not_treat_runtime_configuration_as_a_descriptor(tmp_path):
    path = tmp_path / "image.tar"
    images = image_archive(path, "goodvision-main:v1", [b"base layer"])
    assert images[0]["name"] == "goodvision-main:v1"
    assert images[0]["os"] == "linux" and images[0]["architecture"] == "amd64"
    assert len(images[0]["blobs"]) == 4


def test_module_package_omits_only_blobs_in_its_selected_baseline(
    tmp_path, monkeypatch
):
    common = b"unchanged dependency layer"
    old = image_archive(tmp_path / "old.tar", "goodvision-main:v1", [common])
    new = tmp_path / "new.tar"
    image_archive(new, "goodvision-main:v2", [common, b"new application code"])
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"images": old}))

    def save(command, **options):
        assert command[:3] == ["docker", "image", "save"]
        shutil.copyfile(new, command[command.index("--output") + 1])

    monkeypatch.setattr(export.subprocess, "run", save)
    output = tmp_path / "goodvision-main-v2"
    result = export.package("main", "v2", ["goodvision-main:v2"], output, baseline)
    base_blob = "blobs/sha256/" + hashlib.sha256(common).hexdigest()
    code_blob = "blobs/sha256/" + hashlib.sha256(b"new application code").hexdigest()
    with tarfile.open(output / "images.tar.gz") as archive:
        assert base_blob not in archive.getnames()
        assert code_blob in archive.getnames()
    assert result["requires"][0]["name"] == "goodvision-main:v1"
    assert result["omitted_existing_blob_bytes"] >= len(common)
    assert output.with_suffix(".zip").is_file()
    with zipfile.ZipFile(output.with_suffix(".zip")) as archive:
        assert {"install.py", "update.cmd", "rollback.cmd", "update.json"}.issubset(
            archive.namelist()
        )
