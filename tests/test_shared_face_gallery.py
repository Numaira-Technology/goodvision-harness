"""Shared gallery edits refresh existing workers without restarting inference."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from test_face_observation_pipeline import runtime, wait_for, window

from goodvision_harness.config import load_config
from goodvision_harness.face_fast import FaceFastResults
from goodvision_harness.face_recognition import FaceGallery
from goodvision_harness.service import HarnessRuntime, create_app


@pytest.mark.parametrize("remove_file", [False, True])
def test_separate_gallery_instances_observe_atomic_edits(tmp_path, remove_file):
    path = tmp_path / "gallery.json"
    primary = FaceGallery(path, model_name="buffalo_l")
    worker = FaceGallery(path, model_name="buffalo_l")
    assert worker.refresh() is None
    primary.enroll("alice", "Alice", [1, 0])
    before, after = worker.refresh()
    assert before.people == {} and after.people == {"alice": "Alice"}
    assert worker.match([1, 0]).person_id == "alice"
    assert worker.refresh() is None
    assert primary.refresh() is None
    if remove_file:
        path.unlink()
    else:
        primary.delete("alice")
    old, empty = worker.refresh()
    assert old.people == {"alice": "Alice"} and not empty.people
    assert worker.match([1, 0]) is None
    assert old.version != empty.version


def test_invalid_external_edit_can_be_retried_without_losing_cached_snapshot(tmp_path):
    path = tmp_path / "gallery.json"
    primary = FaceGallery(path, model_name="buffalo_l")
    primary.enroll("alice", "Alice", [1, 0])
    worker = FaceGallery(path, model_name="buffalo_l")
    previous = worker.private_snapshot()
    path.write_text(json.dumps({"schema_version": -1}))
    with pytest.raises(RuntimeError, match="could not be loaded"):
        worker.refresh()
    assert worker.private_snapshot() is previous
    primary.enroll("bob", "Bob", [0, 1])
    assert worker.refresh()[1].people == {"alice": "Alice", "bob": "Bob"}


def test_running_app_refreshes_gallery_and_rejects_stale_fast_results(
    tmp_path, monkeypatch,
):
    engine, _, _, _ = runtime(tmp_path, monkeypatch)
    primary = FaceGallery(engine.gallery.path, model_name=engine.config.model_name)
    harness = HarnessRuntime(load_config(), face_runtime=engine)
    monkeypatch.setattr(harness, "start", lambda: None)
    harness.face_fast_results = FaceFastResults(stream_session_id="shared-test")
    with TestClient(create_app(lambda: harness)):
        primary.enroll("alice", "Alice", [1, 0])
        wait_for(lambda: engine.gallery.private_snapshot().people == {"alice": "Alice"})
        known = engine.observe_window(window(0))
        harness.face_fast_results.publish(known)
        face = harness.face_fast_results.snapshot()["result"]["faces"][0]
        assert face["person_id"] == "alice"
        primary.delete("alice")
        wait_for(lambda: not engine.gallery.private_snapshot().people)
        wait_for(lambda: not harness.face_fast_results.snapshot()["available"])
        harness.face_fast_results.publish(known)
        assert not harness.face_fast_results.snapshot()["available"]
        result = engine.complete_observation(known)
        assert all(item.attributes.get("person_id") != "alice"
                   for item in result.classification.face_observations)
        primary.enroll("alice", "Alice again", [1, 0])
        wait_for(lambda: engine.gallery.private_snapshot().people == {
            "alice": "Alice again",
        })
        assert harness._face_runtime is engine


def test_worker_invalidates_live_and_retired_identities_on_external_deletion(
    tmp_path, monkeypatch,
):
    engine, _, _, _ = runtime(tmp_path, monkeypatch)
    primary = FaceGallery(engine.gallery.path, model_name=engine.config.model_name)
    primary.enroll("alice", "Alice", [1, 0])
    engine.gallery.refresh()
    harness = HarnessRuntime(load_config(), face_runtime=engine)
    live, retired, scene, old_scene, display = (Mock() for _ in range(5))
    harness._session = SimpleNamespace(person_registry=live)
    harness._scene = scene
    harness._camera_display_metadata = display
    harness._run_postprocess["old"] = SimpleNamespace(
        state="running", session=SimpleNamespace(person_registry=retired),
        scene=old_scene,
    )
    try:
        primary.delete("alice")
        primary.enroll("bob", "Bob", [0, 1])
        harness.refresh_face_gallery()
        live.forget_person.assert_called_once_with("alice")
        retired.forget_person.assert_called_once_with("alice")
        scene.forget_identity.assert_called_once_with("alice")
        old_scene.forget_identity.assert_called_once_with("alice")
        display.forget_identity.assert_called_once_with("alice")
        live.allow_person.assert_called_once_with("bob")
        scene.allow_identity.assert_called_once_with("bob")
        display.allow_identity.assert_called_once_with("bob")
    finally:
        harness._session = harness._scene = None
        harness._run_postprocess.clear()
        harness.stop()
