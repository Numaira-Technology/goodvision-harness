"""Real archive regression checks for legacy cloud analysis state migration."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from goodvision_harness.deployment_settings import (
    DeploymentSettings,
    DeploymentSettingsStore,
)
from goodvision_harness.route_probe import ROUTES_BY_ID
from tools import cloud_analysis


def populate_lane(lane: Path, label: str) -> None:
    store = DeploymentSettingsStore(lane / "deployment-settings.json")
    store.save(DeploymentSettings(
        stream_url=f"rtsp://{label}.example/live",
        dry_run=False,
        autostart=False,
        buffer_delay_seconds=3.5,
        camera_layout="stereo",
        processing_method="sam_live",
        inference_mode="server",
        inference_server_url="https://vision.example",
        inference_server_model="saved-model",
        inference_server_protocol="goodvision",
        tracking_mode="server",
        tracking_server_url="https://tracking.example",
        tracking_server_capacity=2,
        tracking_server_protocol="contour",
        rich_label_model="contour",
    ))
    for provider in ("aiclub", "openrouter", "dashscope", "local", "tracking"):
        store.save_api_key(f"fixture-{label}-{provider}", provider=provider)
    store.save_model_route(next(iter(ROUTES_BY_ID)))
    store.save_analysis_settings({
        "features": ["smoking"], "event_sensitivities": {"smoking": "high"},
    })
    (lane / "face-gallery.json").write_text('{"people":[]}', encoding="utf-8")
    evidence = lane / "event-evidence"
    evidence.mkdir()
    (evidence / "migration-marker").write_text(label, encoding="utf-8")


def uploaded_restore(tmp_path: Path, monkeypatch, legacy: Path) -> bytes:
    source = tmp_path / "application"
    for name in (
        "src/goodvision_harness/marker",
        "pyproject.toml",
        "config.yaml",
        "tools/gpu_analysis_host.py",
    ):
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("source fixture", encoding="utf-8")
    controller = tmp_path / "controller"
    monkeypatch.setattr(cloud_analysis, "state_directory", lambda: controller)

    def relocated_path(value):
        return legacy if str(value) == "/legacy" else Path(value)

    monkeypatch.setattr(cloud_analysis, "Path", relocated_path)
    uploaded = []

    def run(command, **kwargs):
        if command[0] == "scp" and command[-1].endswith("/restore.tar.gz"):
            uploaded.append(Path(command[-2]).read_bytes())

    cloud_analysis.prepare(
        source, "/workspace/analysis", ["ssh", "root@fixture"], ["scp"], run,
    )
    assert len(uploaded) == 1
    return uploaded[0]


def restore_archive(snapshot: bytes, destination: Path) -> set[str]:
    files = set()
    with tarfile.open(fileobj=io.BytesIO(snapshot), mode="r:gz") as archive:
        for member in archive:
            if not member.isfile():
                continue
            assert member.name.startswith("data/")
            assert ".." not in member.name.split("/")
            files.add(member.name)
            path = destination / member.name
            path.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            assert source is not None
            with source:
                path.write_bytes(source.read())
    return files


def assert_lane_restored(lane: Path, label: str) -> None:
    store = DeploymentSettingsStore(lane / "deployment-settings.json")
    saved = store.load()
    assert saved is not None
    assert saved.stream_url == f"rtsp://{label}.example/live"
    assert saved.buffer_delay_seconds == 3.5 and saved.camera_layout == "stereo"
    assert saved.inference_server_protocol == "goodvision"
    assert saved.tracking_server_protocol == "contour"
    assert saved.inference_server_model == "saved-model"
    assert saved.rich_label_model == "contour"
    for provider in ("aiclub", "openrouter", "dashscope", "local", "tracking"):
        assert store.load_api_key(provider=provider) == f"fixture-{label}-{provider}"
    assert store.load_model_route() == next(iter(ROUTES_BY_ID))
    assert store.load_analysis_settings() == {
        "features": ["smoking"], "event_sensitivities": {"smoking": "high"},
    }
    assert (lane / "face-gallery.json").read_text("utf-8") == '{"people":[]}'
    assert (lane / "event-evidence/migration-marker").read_text("utf-8") == label


@pytest.mark.parametrize("empty_primary", [False, True])
@pytest.mark.parametrize("additional_camera", [False, True])
def test_plain_compose_state_migrates_past_empty_or_missing_camera_volume(
    tmp_path, monkeypatch, empty_primary, additional_camera,
):
    legacy = tmp_path / "legacy"
    populate_lane(legacy / "default", "old-default")
    if empty_primary:
        (legacy / "camera-01").mkdir()
    if additional_camera:
        populate_lane(legacy / "camera-02", "second")
    snapshot = uploaded_restore(tmp_path, monkeypatch, legacy)
    files = restore_archive(snapshot, tmp_path / "restored")
    assert "data/camera-01/model-api-key-2" in files
    assert_lane_restored(tmp_path / "restored/data/camera-01", "old-default")
    if additional_camera:
        assert_lane_restored(tmp_path / "restored/data/camera-02", "second")
    assert not any("/default/" in path for path in files)


def test_populated_camera_volume_takes_precedence_over_default(tmp_path, monkeypatch):
    legacy = tmp_path / "legacy"
    populate_lane(legacy / "default", "old-default")
    populate_lane(legacy / "camera-01", "current-primary")
    populate_lane(legacy / "camera-02", "current-second")
    snapshot = uploaded_restore(tmp_path, monkeypatch, legacy)
    restore_archive(snapshot, tmp_path / "restored")
    assert_lane_restored(tmp_path / "restored/data/camera-01", "current-primary")
    assert_lane_restored(tmp_path / "restored/data/camera-02", "current-second")


def test_empty_default_does_not_displace_populated_additional_camera(
    tmp_path, monkeypatch,
):
    legacy = tmp_path / "legacy"
    (legacy / "default").mkdir(parents=True)
    (legacy / "camera-01").mkdir()
    populate_lane(legacy / "camera-02", "second")
    snapshot = uploaded_restore(tmp_path, monkeypatch, legacy)
    files = restore_archive(snapshot, tmp_path / "restored")
    assert not any(path.startswith("data/camera-01/") for path in files)
    assert_lane_restored(tmp_path / "restored/data/camera-02", "second")


def test_saved_gpu_snapshot_remains_authoritative_over_legacy_volumes(
    tmp_path, monkeypatch,
):
    legacy = tmp_path / "legacy"
    populate_lane(legacy / "default", "old-default")
    populate_lane(legacy / "camera-01", "old-camera")
    current = tmp_path / "current/data/camera-01"
    populate_lane(current, "backed-up")
    controller = tmp_path / "controller"
    controller.mkdir()
    with tarfile.open(controller / "analysis-data.tar.gz", "w:gz") as archive:
        archive.add(current, arcname="data/camera-01")
    snapshot = uploaded_restore(tmp_path, monkeypatch, legacy)
    restore_archive(snapshot, tmp_path / "restored")
    assert_lane_restored(tmp_path / "restored/data/camera-01", "backed-up")
