"""Deployment failure boundaries; these tests do not validate GPU inference."""

import hashlib
import json
import subprocess
import zipfile
from types import SimpleNamespace

import pytest

from workers.contour import deploy_contour as deploy


def make_bundle(directory, files):
    manifest = {
        "product": "GoodVision Contour",
        "files": [
            {
                "path": name,
                "bytes": len(body),
                "sha256": hashlib.sha256(body).hexdigest(),
            }
            for name, body in files.items()
        ],
    }
    with zipfile.ZipFile(
        directory / "goodvision-contour-framework.zip", "w"
    ) as archive:
        archive.writestr("release-manifest.json", json.dumps(manifest))
        for name, body in files.items():
            archive.writestr(name, body)


def test_deploy_repeated_extraction_preserves_key_and_detects_changed_source(tmp_path):
    make_bundle(tmp_path, {"workers/contour/compose.yaml": b"services: {}\n"})
    root = deploy.unpack(tmp_path, tmp_path / "installed")
    node = root / "workers/contour"
    key = deploy.initialize_key(node)
    original = key.read_text()
    deploy.write_env(node, {"CONTOUR_PORT": "8093"})
    assert deploy.unpack(tmp_path, root) == root
    deploy.initialize_key(node)
    assert key.read_text() == original
    deploy.write_env(node, {"CONTOUR_PORT": "18093"})
    backup = list(node.glob(".env.before-deploy-*"))
    assert len(backup) == 1 and "8093" in backup[0].read_text()
    (node / "compose.yaml").write_text("changed")
    with pytest.raises(RuntimeError, match="被修改或缺失"):
        deploy.unpack(tmp_path, root)
    assert key.read_text() == original


@pytest.mark.parametrize(
    "path", ["workers/../../outside", "/tmp/outside", "workers/..\\outside"]
)
def test_deploy_rejects_zip_path_escape_before_creating_installation(tmp_path, path):
    make_bundle(tmp_path, {path: b"unexpected"})
    with pytest.raises(RuntimeError, match="不安全"):
        deploy.unpack(tmp_path, tmp_path / "installed")
    assert not (tmp_path / "installed").exists()


def test_deploy_rejects_corrupt_archive_before_extracting(tmp_path):
    make_bundle(tmp_path, {"workers/contour/compose.yaml": b"services: {}"})
    (tmp_path / "goodvision-contour-framework.zip.sha256").write_text("0" * 64)
    with pytest.raises(RuntimeError, match="校验失败"):
        deploy.unpack(tmp_path, tmp_path / "installed")
    assert not (tmp_path / "installed").exists()


def test_deploy_readiness_waits_for_authenticated_gpu_ready(monkeypatch):
    calls = []
    probes = 0

    def run(command, **kwargs):
        nonlocal probes
        calls.append(command)
        if "exec" in command:
            probes += 1
            assert "Bearer " in command[-1] and "gpu_ready" in command[-1]
            return subprocess.CompletedProcess(command, 1 if probes == 1 else 0)
        if "inspect" in command:
            return SimpleNamespace(stdout=json.dumps({"Status": "running"}))
        return SimpleNamespace(stdout="abc123")

    monkeypatch.setattr(deploy, "run", run)
    monkeypatch.setattr(deploy.time, "sleep", lambda _: None)
    deploy.await_ready(["docker", "compose"], {}, 30)
    assert probes == 2


def test_deploy_readiness_fails_when_engine_restarts(monkeypatch):
    def run(command, **kwargs):
        if "exec" in command:
            return subprocess.CompletedProcess(command, 1)
        if "inspect" in command:
            return SimpleNamespace(
                stdout=json.dumps({"Status": "restarting", "OOMKilled": True})
            )
        return SimpleNamespace(stdout="abc123")

    monkeypatch.setattr(deploy, "run", run)
    with pytest.raises(RuntimeError, match="推理容器已退出或重启"):
        deploy.await_ready(["docker", "compose"], {}, 30)


def test_deploy_rejects_arm_before_docker_or_files_are_changed(monkeypatch):
    monkeypatch.setattr(deploy.platform, "system", lambda: "Linux")
    monkeypatch.setattr(deploy.platform, "machine", lambda: "aarch64")
    with pytest.raises(RuntimeError, match="不适用于 Jetson"):
        deploy.deploy(SimpleNamespace())
