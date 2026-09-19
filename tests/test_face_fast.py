"""Fast face delivery must bypass blocked durable/ordered/main delivery."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from test_face_observation_pipeline import runtime, wait_for, window

from goodvision_harness.config import load_config
from goodvision_harness.face_fast import FaceFastResults
from goodvision_harness.face_recognition import FaceRecognitionSession
from goodvision_harness.service import HarnessRuntime, create_app


def cache(run="run-1"):
    return FaceFastResults(stream_session_id=run, source_id="camera", generation=1)


def test_fast_result_overtakes_slow_previous_frame_without_rewinding(
    tmp_path, monkeypatch,
):
    gate = threading.Event()
    engine, _, _, _ = runtime(tmp_path, monkeypatch, gate=gate)
    face = FaceRecognitionSession(engine, observation_store_path=tmp_path / "private")
    face.fast_results = cache()
    try:
        first = face.submit_window(window(0))
        second = face.submit_window(window(1))
        wait_for(lambda: face.fast_results.snapshot()["available"])
        result = face.fast_results.snapshot()
        assert result["result"]["frame"]["source_frame_index"] == 1
        assert not first.done() and not second.done()
        assert "identity_embedding" not in json.dumps(result)
        assert "jpeg" not in json.dumps(result)
        gate.set()
        face.close(wait=True)
        assert first.result().classification is not None
        assert second.result().classification is not None
        latest = face.fast_results.snapshot()["result"]
        assert latest["frame"]["source_frame_index"] == 1
    finally:
        gate.set()
        face.close(wait=True)


@pytest.mark.parametrize("block", ["record_raw", "callback"])
def test_fast_delivery_bypasses_disk_write_and_ordered_callback(
    tmp_path, monkeypatch, block,
):
    gate, entered = threading.Event(), threading.Event()
    engine, _, _, _ = runtime(tmp_path, monkeypatch)
    face = FaceRecognitionSession(engine, observation_store_path=tmp_path / "private")
    face.fast_results = cache()

    def blocked(*args):
        entered.set()
        assert gate.wait(8)

    if block == "record_raw":
        from goodvision_harness.face_observation_store import FaceObservationStore

        original = FaceObservationStore.record_raw

        def write(store, *args):
            blocked()
            return original(store, *args)

        monkeypatch.setattr(FaceObservationStore, "record_raw", write)
    try:
        face.submit_window(window(0), on_admitted=(
            lambda future: future.add_done_callback(blocked)
        ) if block == "callback" else None)
        assert entered.wait(8)
        latest = face.fast_results.snapshot()["result"]
        assert latest["frame"]["source_frame_index"] == 0
        if block == "callback":
            face.submit_window(window(1))
            wait_for(lambda: face.fast_results.snapshot()["revision"] == 2)
            assert face.stats()["completed"] == 0
    finally:
        gate.set()
        face.close(wait=True)


def test_empty_error_gallery_changes_and_session_isolation(tmp_path, monkeypatch):
    engine, _, _, _ = runtime(tmp_path, monkeypatch)
    engine.gallery.enroll("alice", "Alice", [1.0, 0.0])
    old = engine.observe_window(window(0))
    fast = cache()
    fast.publish(old)
    assert fast.snapshot()["result"]["faces"][0]["person_id"] == "alice"
    engine.delete_person("alice")
    fast.require_gallery(engine.gallery.private_snapshot().version)
    fast.publish(replace(old, window=window(1)))
    assert not fast.snapshot()["available"]
    current = engine.observe_window(window(2))
    fast.publish(replace(current, objects=()))
    assert fast.snapshot()["result"]["faces"] == []
    fast.publish(replace(current, window=window(3), error="private path /secret"))
    result = fast.snapshot()
    assert result["result"]["error"]
    assert "secret" not in json.dumps(result)
    result["result"]["faces"].append({"fake": True})
    assert fast.snapshot()["result"]["faces"] == []
    new = cache("run-2")
    fast.stop()
    fast.publish(replace(current, window=window(4)))
    assert not fast.snapshot()["active"]
    assert fast.snapshot()["result"]["frame"]["source_frame_index"] == 3
    assert not new.snapshot()["available"]


def test_failed_inference_publishes_error_and_fast_failure_does_not_break_main(
    tmp_path, monkeypatch,
):
    engine, _, _, _ = runtime(tmp_path, monkeypatch)
    face = FaceRecognitionSession(engine, observation_store_path=tmp_path / "private")
    face.fast_results = cache()
    original = engine.observe_window

    def fail(*args, **kwargs):
        raise RuntimeError("secret")

    try:
        monkeypatch.setattr(engine, "observe_window", fail)
        face.submit_window(window(0)).result(timeout=8)
        assert face.fast_results.snapshot()["result"]["error"]
        monkeypatch.setattr(engine, "observe_window", original)
        monkeypatch.setattr(face.fast_results, "publish", fail)
        response = face.submit_window(window(1)).result(timeout=8)
        assert response.classification is not None
    finally:
        face.close(wait=True)


def test_http_fast_endpoint_does_not_wait_for_service_lock(monkeypatch):
    harness = HarnessRuntime(load_config())
    monkeypatch.setattr(harness, "start", lambda: None)
    # No inference or external services are started by this HTTP test.
    with TestClient(create_app(lambda: harness)) as client:
        with ThreadPoolExecutor(max_workers=1) as executor:
            with harness._state_lock:
                pending = executor.submit(client.get, "/v1/faces/fast")
                response = pending.result(timeout=3)
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-store"
        assert response.json()["available"] is False
        assert response.json()["active"] is False
