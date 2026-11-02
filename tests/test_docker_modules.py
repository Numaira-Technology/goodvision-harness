"""Offline installer contracts; no Docker daemon, real package build or GPU."""

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import deploy_modules as deploy
from tools import docker_modules as spec
from tools import package_modules as export


def test_separate_projects_share_only_network_and_use_service_addresses(tmp_path):
    settings = spec.defaults("site1", "release1")
    main = spec.compose("main", settings, tmp_path, tmp_path)
    bbox = spec.compose("bbox", settings, tmp_path, tmp_path)
    assert main["name"] != bbox["name"]
    assert main["networks"]["modules"] == bbox["networks"]["modules"]
    assert main["networks"]["modules"]["external"] is True
    environment = main["services"]["main"]["environment"]
    assert environment["GOODVISION_BBOX_LOCAL_URL"] == "http://gv-bbox:8096"
    assert environment["GOODVISION_SYSTEM1_LOCAL_URL"] == "http://gv-system1:8097"
    assert environment["GOODVISION_MODULE_MANAGER"] == "compose"
    assert environment["HARNESS_AUTOSTART"] == "false"
    assert "deploy" not in main["services"]["main"]
    assert "deploy" not in bbox["services"]["bbox"]
    assert "gpu_ready'] is False" in bbox["services"]["bbox"]["healthcheck"]["test"][-1]
    assert "docker.sock" not in json.dumps(main)


def test_gpu_layers_are_reused_without_sharing_contour_engine_port(tmp_path):
    settings = spec.defaults()
    settings["bbox_device"] = "cuda"
    assert spec.runtime_versions(["bbox", "system1", "contour"], settings) == [
        "2.10",
        "2.8",
    ]
    bbox = spec.compose("bbox", settings, tmp_path, tmp_path)["services"]["bbox"]
    contour = spec.compose("contour", settings, tmp_path, tmp_path)
    engine = contour["services"]["engine"]
    system1 = spec.compose("system1", settings, tmp_path, tmp_path)["services"][
        "system1"
    ]
    assert (
        bbox["build"]["args"]["GPU_RUNTIME_IMAGE"]
        == engine["build"]["args"]["GPU_RUNTIME_IMAGE"]
    )
    assert (
        system1["build"]["args"]["GPU_RUNTIME_IMAGE"]
        != engine["build"]["args"]["GPU_RUNTIME_IMAGE"]
    )
    assert "ports" not in engine and engine["networks"] == ["engine"]
    assert contour["networks"]["engine"]["internal"] is True
    assert "Authorization" in contour["services"]["contour"]["healthcheck"]["test"][-1]


@pytest.mark.parametrize(
    "url",
    [
        "http://user:pass@host",
        "http://host:0",
        "http://host:99999",
        "http://host/?key=secret",
        "file:///data",
        "http://host\n",
    ],
)
def test_invalid_remote_addresses_are_rejected(url):
    with pytest.raises(ValueError):
        spec.endpoint(url)


def test_configure_persists_addresses_and_plan_has_no_side_effects(
    tmp_path, monkeypatch, capsys
):
    root, state = tmp_path / "source", tmp_path / "state"
    root.mkdir()
    monkeypatch.setattr(deploy, "location", lambda: (root, root))
    monkeypatch.setattr(
        deploy.subprocess, "run", lambda *a, **k: pytest.fail("Docker call")
    )
    arguments = ["--state-dir", str(state), "--modules", "main"]
    assert deploy.main(["plan", *arguments]) == 0
    assert not state.exists()
    assert (
        deploy.main(
            [
                "configure",
                *arguments,
                "--bbox-url",
                "http://worker:8096",
                "--main-port",
                "18080",
            ]
        )
        == 0
    )
    assert deploy.main(["plan", *arguments]) == 0
    settings = json.loads((state / "settings.json").read_text())
    assert settings["urls"]["bbox"] == "http://worker:8096"
    assert settings["ports"]["main"] == 18080
    rendered = json.loads((state / "compose.main.json").read_text())
    assert (
        rendered["services"]["main"]["environment"]["GOODVISION_BBOX_LOCAL_URL"]
        == "http://worker:8096"
    )
    assert "18080:8080" in rendered["services"]["main"]["ports"][0]


def test_group_start_waits_for_workers_and_stop_preserves_other_projects(
    tmp_path, monkeypatch
):
    calls = []
    monkeypatch.setattr(deploy, "location", lambda: (tmp_path, tmp_path))
    monkeypatch.setattr(
        deploy, "ensure_network", lambda settings: calls.append(["network"])
    )
    monkeypatch.setattr(deploy, "run", lambda command, **kwargs: calls.append(command))
    state = tmp_path / "state"
    arguments = ["--state-dir", str(state), "--modules", "main", "bbox"]
    assert deploy.main(["up", *arguments]) == 0
    startups = [command for command in calls if "up" in command]
    assert len(startups) == 2
    assert startups[0][5].endswith("compose.bbox.json")
    assert startups[1][5].endswith("compose.main.json")
    assert all("--wait" in command and "--no-build" in command for command in startups)
    calls.clear()
    assert deploy.main(["stop", "--state-dir", str(state), "--modules", "bbox"]) == 0
    assert len(calls) == 1 and calls[0][-1] == "stop"
    assert (state / "compose.main.json").is_file()
    assert not any("down" in command or "rm" in command for command in calls)


def test_missing_gpu_models_prevent_partial_group_start(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(deploy, "location", lambda: (tmp_path, tmp_path))
    monkeypatch.setattr(deploy, "run", lambda command, **kwargs: calls.append(command))
    monkeypatch.setattr(
        deploy, "ensure_network", lambda *a: pytest.fail("early network mutation")
    )
    assert (
        deploy.main(
            [
                "up",
                "--state-dir",
                str(tmp_path / "state"),
                "--modules",
                "bbox",
                "system1",
            ]
        )
        == 1
    )
    assert not any("up" in command for command in calls)


def test_existing_network_must_belong_to_our_installer(monkeypatch):
    monkeypatch.setattr(
        deploy.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            returncode=0, stdout='[{"Labels":{"owner":"another-app"}}]'
        ),
    )
    with pytest.raises(ValueError, match="not a GoodVision"):
        deploy.ensure_network(spec.defaults())


def test_model_download_is_explicit_and_tokens_are_not_in_command(tmp_path):
    settings = spec.defaults()
    document = spec.compose("system1", settings, tmp_path, tmp_path)
    command = deploy.prepare_command("system1", document, tmp_path, tmp_path)
    assert "--gpus" not in command
    assert "HF_TOKEN" in command and "HF_HUB_OFFLINE=0" in command
    assert "revision=REVISION" in command[-1]


def test_contour_key_is_created_once_without_changing_existing_credentials(tmp_path):
    deploy.initialize(tmp_path, ["contour"])
    path = tmp_path / "secrets/contour-key"
    original = path.read_bytes()
    assert len(original.strip()) >= 32
    if os.name != "nt":
        assert path.stat().st_mode & 0o777 == 0o644
    deploy.initialize(tmp_path, ["main", "bbox", "contour"])
    assert path.read_bytes() == original


def test_tiny_fixture_package_is_portable_and_detects_modified_files(
    tmp_path, monkeypatch
):
    # Only fixture text is archived; this never packages the actual application.
    root = tmp_path / "fixture"
    names = ["workers/gpu_runtime/Dockerfile", ".dockerignore"]
    for relative in names:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n")
    for filename in ("deploy_modules.py", "docker_modules.py"):
        destination = root / "tools" / filename
        destination.parent.mkdir(exist_ok=True)
        destination.write_text(
            (Path(deploy.__file__).parent / filename).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    monkeypatch.setattr(export, "tracked_files", lambda *args: names)
    archive = tmp_path / "fixture.zip"
    manifest = export.package(root, "gpu-runtime", "release1", "cpu", archive)
    installed = tmp_path / "installed"
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(installed)
    deploy.verify_package(installed, manifest)
    result = subprocess.run(
        [
            sys.executable,
            str(installed / "install.py"),
            "plan",
            "--state-dir",
            str(tmp_path / "unused"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    plan = json.loads(result.stdout)
    assert len(plan["base_builds"]) == 2 and not plan["compose_projects"]
    assert not (tmp_path / "unused").exists()
    assert not manifest["includes_images"] and not manifest["includes_models"]
    (installed / "source/.dockerignore").write_text("changed")
    with pytest.raises(ValueError, match="has changed"):
        deploy.verify_package(installed, manifest)


def test_packager_rejects_forced_tracked_secrets_and_model_files(tmp_path):
    for relative in (
        "workers/contour/secrets/node-key",
        "workers/bbox/.env",
        "workers/bbox/yolo26s.pt",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("private")
        with pytest.raises(ValueError):
            export.safe_source(tmp_path, relative)


def test_compose_documents_render_without_an_engine(tmp_path):
    docker = shutil.which("docker")
    if not docker:
        pytest.skip("Docker Compose client is not installed; no daemon is needed")
    environment = {
        **os.environ,
        "DOCKER_CONFIG": str(tmp_path / "docker-config"),
        "CACHE": "UNEXPECTED_EXPANSION",
    }
    version = subprocess.run(
        [docker, "compose", "version"], env=environment, capture_output=True
    )
    if version.returncode:
        pytest.skip("Docker Compose plugin is not installed")
    for module, device in (
        ("main", "cpu"),
        ("bbox", "cpu"),
        ("bbox", "cuda"),
        ("contour", "cuda"),
        ("system1", "cuda"),
    ):
        settings = spec.defaults()
        settings["bbox_device"] = device
        # A dollar in a real directory name must survive Compose interpolation.
        document = spec.compose(module, settings, tmp_path / "source$CACHE", tmp_path)
        result = subprocess.run(
            [docker, "compose", "-f", "-", "config", "--format", "json"],
            input=json.dumps(document),
            text=True,
            env=environment,
            capture_output=True,
            check=True,
        )
        rendered = json.loads(result.stdout)
        # `config` escapes dollars again so its output is valid Compose input.
        context = rendered["services"][module]["build"]["context"]
        assert "source$CACHE" in context.replace("$$", "$")
        assert "UNEXPECTED_EXPANSION" not in context
