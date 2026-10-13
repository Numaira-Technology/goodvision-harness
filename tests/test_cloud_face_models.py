"""Offline fixture checks for GPU-host face-model provisioning and restoration."""

from __future__ import annotations

import shutil
import tarfile
from pathlib import Path

import pytest
import yaml

from goodvision_harness.config import load_config
from tools import cloud_analysis, gpu_analysis_host

ROOT = Path(__file__).resolve().parents[1]
PROFILE = "operator-profile"
DETECTION = "models/face/models/" + PROFILE + "/detector.onnx"
RECOGNITION = "models/face/models/" + PROFILE + "/recognition.onnx"


@pytest.fixture
def provisioning(tmp_path, monkeypatch):
    application = tmp_path / "application"
    application.mkdir()
    (application / "src").mkdir()
    (application / "tools").mkdir()
    for name in ("pyproject.toml", "tools/gpu_analysis_host.py"):
        (application / name).write_text("fixture source", "utf-8")
    config = yaml.safe_load((ROOT / "config.yaml").read_text("utf-8"))
    config["face"].pop("model_profile", None)
    config["face"].update(
        model_name=PROFILE, model_root="ignored-original-root",
        allow_model_download=False,
    )
    (application / "config.yaml").write_text(yaml.safe_dump(config), "utf-8")
    controller = tmp_path / "controller"
    controller.mkdir()
    face_root = tmp_path / "operator-models"
    face_root.mkdir()
    monkeypatch.setenv("FACE_MODEL_ROOT", str(face_root))
    monkeypatch.setattr(cloud_analysis, "state_directory", lambda: controller)
    monkeypatch.setattr("goodvision_harness.config.find_dotenv", lambda **_: "")
    remote = tmp_path / "remote"
    (remote / "data").mkdir(parents=True)
    (remote / "source").mkdir()
    transfers = []
    setup_timeouts = []

    def prepare():
        def run(command, **kwargs):
            if command[0] == "scp":
                sources = [Path(path) for path in command[1:-1]]
                transfers.append((sources, kwargs["timeout"]))
                for source in sources:
                    assert source.is_relative_to(controller)
                    target = command[-1].split(":", 1)[1]
                    destination = Path(target)
                    if target.endswith("/"):
                        destination /= source.name
                    shutil.copyfile(source, destination)
            elif command[-1] == "bash -s":
                setup_timeouts.append(kwargs["timeout"])
                with tarfile.open(remote / "source.tar.gz") as archive:
                    archive.extractall(remote / "source", filter="data")
                # Execute the actual remote restore program, without running
                # installers, SSH, downloads, licensed models or inference.
                program = kwargs["input"].partition(
                    "<<'RESTORE_ANALYSIS'\n"
                )[2].partition("\nRESTORE_ANALYSIS")[0]
                assert program
                exec(compile(program, "<remote-model-restore>", "exec"), {})
        cloud_analysis.prepare(
            application, str(remote), ["ssh", "root@fixture"], ["scp"], run,
        )

    return (
        application, controller, face_root, remote, transfers, setup_timeouts, prepare,
    )


def install_fixture(root, name, content):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def snapshot(controller, tmp_path, files):
    source = tmp_path / "snapshot-source"
    for name, content in files.items():
        install_fixture(source, name, content)
    with tarfile.open(controller / "analysis-data.tar.gz", "w:gz") as archive:
        for child in source.iterdir():
            archive.add(child, arcname=child.name)


def test_controller_mounts_existing_operator_models_read_only():
    compose = yaml.safe_load((ROOT / "compose.sam31-remote.yaml").read_text("utf-8"))
    controller = compose["services"]["sam-remote"]
    assert "face-models:/legacy/face-models:ro" in controller["volumes"]
    assert controller["environment"]["FACE_MODEL_ROOT"] == "/legacy/face-models"


def test_operator_weights_reach_configured_profile_with_download_disabled(
    provisioning, monkeypatch,
):
    _, controller, supplied, remote, transfers, setup_timeouts, prepare = provisioning
    install_fixture(supplied, f"models/{PROFILE}/detector.onnx", b"operator-detection")
    install_fixture(
        supplied, f"models/{PROFILE}/recognition.onnx", b"operator-recognition",
    )
    before = {
        path.relative_to(supplied): path.read_bytes()
        for path in supplied.rglob("*") if path.is_file()
    }
    prepare()
    assert (remote / DETECTION).read_bytes() == b"operator-detection"
    assert (remote / RECOGNITION).read_bytes() == b"operator-recognition"
    assert before == {
        path.relative_to(supplied): path.read_bytes()
        for path in supplied.rglob("*") if path.is_file()
    }
    assert any(paths[0].name == "face-models.tar.gz" for paths, _ in transfers)
    assert all(timeout >= 120 for _, timeout in transfers)
    assert setup_timeouts[0] > 30 * 60
    assert not list(controller.glob("analysis-upload-*"))
    lane = gpu_analysis_host.lane_environment({}, remote, 1, 2)
    for key, value in lane.items():
        monkeypatch.setenv(key, value)
    config = load_config(remote / "source/config.yaml")
    assert config.face.model_name == PROFILE
    assert config.face.allow_model_download is False
    assert config.face.model_root == remote / "models/face"
    assert (
        config.face.model_root / "models" / config.face.model_name / "detector.onnx"
    ).read_bytes() == b"operator-detection"


def test_saved_models_restore_when_remote_data_is_already_populated(
    provisioning, tmp_path,
):
    _, controller, _, remote, _, _, prepare = provisioning
    snapshot(controller, tmp_path, {
        "data/camera-01/marker": b"old-data",
        DETECTION: b"saved-detection",
        RECOGNITION: b"saved-recognition",
    })
    install_fixture(remote, "data/camera-01/marker", b"live-data")
    prepare()
    assert (remote / "data/camera-01/marker").read_bytes() == b"live-data"
    assert (remote / DETECTION).read_bytes() == b"saved-detection"
    assert (remote / RECOGNITION).read_bytes() == b"saved-recognition"


def test_operator_files_override_saved_models_and_legacy_data_still_restores(
    provisioning, tmp_path,
):
    _, controller, supplied, remote, _, _, prepare = provisioning
    snapshot(controller, tmp_path, {
        "data/camera-01/marker": b"saved-data",
        DETECTION: b"old-detection",
        RECOGNITION: b"saved-recognition",
    })
    install_fixture(supplied, f"models/{PROFILE}/detector.onnx", b"new-detection")
    prepare()
    assert (remote / "data/camera-01/marker").read_bytes() == b"saved-data"
    assert (remote / DETECTION).read_bytes() == b"new-detection"
    assert (remote / RECOGNITION).read_bytes() == b"saved-recognition"


def test_existing_remote_models_are_preserved_without_operator_input(
    provisioning, tmp_path,
):
    _, controller, _, remote, _, _, prepare = provisioning
    snapshot(controller, tmp_path, {
        "data/camera-01/marker": b"saved-data",
        DETECTION: b"older-detection",
        RECOGNITION: b"missing-recognition",
    })
    install_fixture(remote, DETECTION, b"current-detection")
    prepare()
    assert (remote / DETECTION).read_bytes() == b"current-detection"
    assert (remote / RECOGNITION).read_bytes() == b"missing-recognition"


def test_data_only_snapshot_and_empty_model_volume_remain_supported(
    provisioning, tmp_path,
):
    _, controller, _, remote, transfers, _, prepare = provisioning
    snapshot(controller, tmp_path, {"data/camera-01/marker": b"old-data"})
    prepare()
    assert (remote / "data/camera-01/marker").read_bytes() == b"old-data"
    assert not any(paths[0].name == "face-models.tar.gz" for paths, _ in transfers)


def test_failed_model_upload_keeps_durable_snapshot_and_cleans_staging(
    provisioning, tmp_path,
):
    application, controller, supplied, _, _, _, _ = provisioning
    snapshot(controller, tmp_path, {"data/camera-01/marker": b"saved-data"})
    previous = (controller / "analysis-data.tar.gz").read_bytes()
    install_fixture(supplied, f"models/{PROFILE}/detector.onnx", b"operator-detection")

    def failed_upload(command, **kwargs):
        if command[0] == "scp" and command[-1].endswith("/face-models.tar.gz"):
            raise OSError("fixture upload failure")

    with pytest.raises(OSError, match="fixture upload failure"):
        cloud_analysis.prepare(
            application, "/remote", ["ssh", "root@fixture"], ["scp"], failed_upload,
        )
    assert (controller / "analysis-data.tar.gz").read_bytes() == previous
    assert not list(controller.glob("analysis-upload-*"))



def test_retained_host_preserves_models_and_only_fills_missing_operator_files(
    provisioning, tmp_path,
):
    _, controller, supplied, remote, _, _, prepare = provisioning
    snapshot(controller, tmp_path, {
        "data/camera-01/marker": b"old-data",
        DETECTION: b"snapshot-detection",
        RECOGNITION: b"snapshot-recognition",
    })
    install_fixture(remote, "data/camera-01/marker", b"running-data")
    install_fixture(remote, DETECTION, b"running-detection")
    install_fixture(supplied, f"models/{PROFILE}/detector.onnx", b"new-detection")
    install_fixture(supplied, "models/new-profile/new.onnx", b"new-profile")
    prepare()
    assert (remote / "data/camera-01/marker").read_bytes() == b"running-data"
    assert (remote / DETECTION).read_bytes() == b"running-detection"
    assert (remote / RECOGNITION).read_bytes() == b"snapshot-recognition"
    assert (remote / "models/face/models/new-profile/new.onnx").read_bytes() == (
        b"new-profile"
    )


@pytest.mark.parametrize("recipe", ["cloud", "plain"])
@pytest.mark.parametrize("selection", ["default", "project", "override"])
def test_rendered_compose_imports_the_correct_existing_model_volume(
    tmp_path, recipe, selection,
):
    import json
    import os
    import subprocess

    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("Docker Compose is unavailable")
    available = subprocess.run(
        [docker, "compose", "version"], capture_output=True, timeout=30,
    )
    if available.returncode:
        pytest.skip("Docker Compose is unavailable")
    key = tmp_path / "fixture-ssh-key"
    key.write_text("fixture-only", "utf-8")
    env_file = tmp_path / "fixture.env"
    env_file.write_text("", "utf-8")
    environment = dict(os.environ)
    environment.update(
        VAST_API_KEY="fixture", HF_TOKEN="fixture", SAM_SSH_KEY=str(key),
    )
    environment.pop("COMPOSE_PROJECT_NAME", None)
    environment.pop("HARNESS_MODELS_VOLUME", None)
    expected = (
        "goodvision-cloud_face-models" if recipe == "cloud"
        else "goodvision-harness_face-models"
    )
    if selection == "project":
        environment["COMPOSE_PROJECT_NAME"] = "existing-customer-project"
        expected = "existing-customer-project_face-models"
    elif selection == "override":
        environment["HARNESS_MODELS_VOLUME"] = "existing-operator-weights"
        expected = "existing-operator-weights"
    files = ["compose.cloud-gpu.yaml"] if recipe == "cloud" else [
        "compose.yaml", "compose.sam31-remote.yaml",
    ]
    command = [
        docker, "compose", "--project-directory", str(ROOT),
        "--env-file", str(env_file),
    ]
    for filename in files:
        command += ["-f", str(ROOT / filename)]
    result = subprocess.run(
        [*command, "config", "--format", "json"], env=environment,
        capture_output=True, timeout=30, text=True, check=True,
    )
    rendered = json.loads(result.stdout)
    assert rendered["volumes"]["face-models"]["name"] == expected
    imports = [
        volume for volume in rendered["services"]["sam-remote"]["volumes"]
        if volume["target"] == "/legacy/face-models"
    ]
    assert len(imports) == 1
    assert imports[0]["source"] == "face-models" and imports[0]["read_only"] is True
