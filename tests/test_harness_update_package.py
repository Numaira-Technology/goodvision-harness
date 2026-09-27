"""Published source inventories must survive future packaging attempts."""

import hashlib
import json
import zipfile

import pytest

from tools.harness_update import package_update as packager


@pytest.fixture
def package_source(tmp_path, monkeypatch):
    source = tmp_path / "src/goodvision_harness"
    source.mkdir(parents=True)
    (source / "service.py").write_bytes(b"application v1\n")
    tool_root = tmp_path / "updater"
    tool_root.mkdir()
    for name in (
        "apply_update.py",
        "release_metadata.py",
        "verify_installation.py",
        "update_harness.ps1",
        "README.md",
    ):
        (tool_root / name).write_text("test fixture", encoding="utf-8")
    history = tool_root / "published_sources.json"
    history.write_text('{"older": {"service.py": "older-hash"}}', encoding="utf-8")
    monkeypatch.setattr(packager, "ROOT", tool_root)
    monkeypatch.setattr(packager, "REPO", tmp_path)
    monkeypatch.setattr(packager, "application_files", lambda: ("service.py",))
    monkeypatch.setattr(packager.subprocess, "check_output", lambda *a, **k: b"")
    return source / "service.py", history, tmp_path / "update.zip"


def test_release_inventory_matches_payload_and_cannot_be_redefined(package_source):
    source, history, output = package_source
    packager.package(output, "new-release", record_release=True)
    saved = history.read_bytes()
    with zipfile.ZipFile(output) as archive:
        manifest = json.loads(
            archive.read("goodvision-harness-update-new-release/manifest.json")
        )
        payload = archive.read(
            "goodvision-harness-update-new-release/payload/service.py"
        )
    assert manifest["published_sources"] == json.loads(saved)
    assert (
        manifest["published_sources"]["new-release"]["service.py"]
        == hashlib.sha256(payload).hexdigest()
    )
    assert manifest["published_sources"]["older"] == {"service.py": "older-hash"}
    # Rebuilding the same content is allowed; different content needs a new name.
    packager.package(output, "new-release", record_release=True)
    before = output.read_bytes()
    source.write_bytes(b"application v2\n")
    with pytest.raises(ValueError, match="different application sources"):
        packager.package(output, "new-release", record_release=True)
    assert history.read_bytes() == saved
    assert output.read_bytes() == before


def test_failed_package_does_not_register_release(package_source, monkeypatch):
    _, history, output = package_source
    before = history.read_bytes()

    def fail_write(*args, **kwargs):
        raise OSError("injected archive write failure")

    monkeypatch.setattr(packager.zipfile.ZipFile, "writestr", fail_write)
    with pytest.raises(OSError, match="archive write failure"):
        packager.package(output, "new-release", record_release=True)
    assert history.read_bytes() == before


@pytest.mark.parametrize("release", ["../release", "a/b", "", "bad tag"])
def test_invalid_release_never_writes_package(package_source, release):
    _, history, output = package_source
    before = history.read_bytes()
    with pytest.raises(ValueError, match="Docker image tag"):
        packager.package(output, release, record_release=True)
    assert not output.exists()
    assert history.read_bytes() == before
