import hashlib
import json
import zipfile

import pytest

from workers.contour.package_update import package


def test_code_package_has_only_allowlisted_code_and_valid_checksums(tmp_path):
    path = tmp_path / "update.zip"
    report = package(path)
    assert report["bytes"] < 200_000
    assert report["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    with zipfile.ZipFile(path) as archive:
        prefix = "goodvision-contour-update-20260920-memory1/"
        manifest = json.loads(archive.read(prefix + "update-manifest.json"))
        assert manifest["includes_model"] is False
        assert {row["service"] for row in manifest["images"]} == {"engine", "contour"}
        names = {row["path"] for row in manifest["files"]}
        assert "payload/engine/gpu_memory.py" in names
        assert all(
            not name.endswith((".pt", ".tar", ".gz", ".env", ".pyc")) for name in names
        )
        assert not any("secret" in name or "video_samples" in name for name in names)
        assert set(archive.namelist()) == {prefix + name for name in names} | {
            prefix + "update-manifest.json"
        }
        for row in manifest["files"]:
            data = archive.read(prefix + row["path"])
            assert len(data) == row["bytes"]
            assert hashlib.sha256(data).hexdigest() == row["sha256"]
        for role in ("engine", "contour"):
            dockerfile = archive.read(prefix + "Dockerfile." + role).decode()
            assert dockerfile.splitlines()[0].startswith(
                "ARG BASE_IMAGE=goodvision-contour"
            )
            assert dockerfile.splitlines()[1] == "FROM ${BASE_IMAGE}"
            assert dockerfile.splitlines()[2].startswith("COPY ")
            assert len(dockerfile.splitlines()) == 3  # No dependency installation.
        launcher = archive.read(prefix + "UPDATE.cmd")
        assert b'-File "%~dp0update_contour.ps1"' in launcher
        assert b'-PackageDir "%~dp0."' in launcher


@pytest.mark.parametrize("release", ["../escape", "/absolute", "with space"])
def test_code_package_rejects_invalid_release_names(tmp_path, release):
    with pytest.raises(ValueError):
        package(tmp_path / "update.zip", release)
    assert not list(tmp_path.iterdir())
