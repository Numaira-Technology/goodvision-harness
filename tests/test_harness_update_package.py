"""Published source inventories must survive future packaging attempts."""

import hashlib
import json
import shlex
import subprocess
import sys
import zipfile
from pathlib import Path

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


@pytest.mark.parametrize("kind", ["source", "image"])
def test_update_recipes_retire_legacy_demo_without_removing_live_files(
    package_source, tmp_path, monkeypatch, kind,
):
    source, history, output = package_source
    if kind == "source":
        packager.package(output, "new-release")
        with zipfile.ZipFile(output) as archive:
            dockerfile = archive.read(
                "goodvision-harness-update-new-release/Dockerfile",
            ).decode()
    else:
        monkeypatch.syspath_prepend(
            str(Path(__file__).resolve().parents[1] / "tools/harness_update"),
        )
        from tools.harness_update import package_image_update as image_packager

        history.write_text(json.dumps({
            image_packager.RELEASE: {"service.py": "existing-release-hash"},
        }), encoding="utf-8")
        monkeypatch.setattr(image_packager, "ROOT", history.parent)
        monkeypatch.setattr(image_packager, "REPO", tmp_path)
        monkeypatch.setattr(
            image_packager, "application_files", lambda: ("service.py",),
        )
        context = tmp_path / "image-context"
        image_packager.prepare(context)
        dockerfile = (context / "Dockerfile").read_text(encoding="utf-8")
    # Execute the generated image-layer cleanup against an inherited filesystem.
    # COPY alone would leave both old Python modules and the console script.
    root = tmp_path / "inherited-image"
    retired = [
        "/app/src/goodvision_harness/demo",
        "/usr/local/lib/python3.11/site-packages/goodvision_harness/demo",
        "/usr/local/bin/goodvision-harness-demo",
    ]
    live_files = []
    for name in retired[:2]:
        directory = root / name.lstrip("/")
        directory.mkdir(parents=True)
        (directory / "server.py").write_text("legacy synthetic launcher")
        (directory / "__init__.py").touch()
        current = directory.parent / "service.py"
        current.write_bytes(source.read_bytes())
        live_files.append(current)
    launcher = root / retired[2].lstrip("/")
    launcher.parent.mkdir(parents=True)
    launcher.write_text("legacy entrypoint")
    data = root / "data/deployment-settings.json"
    data.parent.mkdir()
    data.write_text("preserved customer settings")
    live_files.append(data)
    before = {path: path.read_bytes() for path in live_files}
    cleanup_lines = [
        line for line in dockerfile.splitlines() if line.startswith("RUN ")
    ]
    assert len(cleanup_lines) == 1
    instruction, python, flag, cleanup = shlex.split(cleanup_lines[0])
    assert (instruction, python, flag) == ("RUN", "python", "-c")
    for name in retired:
        cleanup = cleanup.replace(repr(name), repr(str(root / name.lstrip("/"))))
    assert dockerfile.index("USER root") < dockerfile.index(cleanup_lines[0])
    assert dockerfile.rindex("COPY --chown") < dockerfile.index(cleanup_lines[0])
    assert dockerfile.rstrip().endswith("USER 10001:10001")
    # A second invocation also succeeds for already-upgraded installations.
    for _ in range(2):
        subprocess.run([sys.executable, "-c", cleanup], check=True, timeout=10)
    assert all(not (root / name.lstrip("/")).exists() for name in retired)
    assert {path: path.read_bytes() for path in live_files} == before


@pytest.mark.parametrize("release", ["../release", "a/b", "", "bad tag"])
def test_invalid_release_never_writes_package(package_source, release):
    _, history, output = package_source
    before = history.read_bytes()
    with pytest.raises(ValueError, match="Docker image tag"):
        packager.package(output, release, record_release=True)
    assert not output.exists()
    assert history.read_bytes() == before
