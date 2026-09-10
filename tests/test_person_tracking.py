"""Independent detector tracking, real-observation TTL, and shared host delivery."""

from dataclasses import replace
from threading import Event

import cv2
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from goodvision_harness.contracts import NormalizedBBox
from goodvision_harness.rich_labeling import (
    RichLabelCoordinator,
    RichLabelHTTPClient,
    RichLabelObject,
    RichLabelProtocolError,
    RichLabelSnapshot,
    select_rich_labels,
)
from workers.paddle_rich_label.app import (
    Detection,
    IoUTracker,
    RichLabelEngine,
    create_app,
)


def _jpeg():
    ok, data = cv2.imencode(".jpg", np.zeros((120, 160, 3), dtype=np.uint8))
    assert ok
    return data.tobytes()


def _headers(sequence=1, timestamp=1.0):
    return {
        "Content-Type": "image/jpeg",
        "X-GoodVision-Stream-Session-Id": "session",
        "X-GoodVision-Frame-Sequence": str(sequence),
        "X-GoodVision-Timestamp-Seconds": str(timestamp),
    }


class _Models:
    ready = True
    gpu_ready = False
    tracking_gpu_ready = True

    def __init__(self):
        self.attribute_calls = 0
        self.detect_calls = 0
        self.missing = False

    def detect(self, image):
        self.detect_calls += 1
        return (
            ()
            if self.missing
            else (
                Detection("person", 0.9, (10, 10, 50, 100)),
                Detection("bottle", 0.8, (80, 30, 100, 80)),
            )
        )

    def attributes(self, crops):
        self.attribute_calls += 1
        raise RuntimeError("attributes unavailable")


def test_tracks_are_independent_of_attributes_and_keep_object_overlays():
    models = _Models()
    engine = RichLabelEngine(models)
    with TestClient(create_app(engine)) as client:
        health = client.get("/health/tracking")
        assert health.status_code == 200
        assert health.json() == {"ready": True, "gpu_ready": True}
        first = client.post("/v1/tracks", content=_jpeg(), headers=_headers()).json()
        assert {item["label"] for item in first["objects"]} == {"person", "bottle"}
        assert all(item["observation_state"] == "observed" for item in first["objects"])
        assert all(
            item["last_observed_timestamp_seconds"] == 1.0 for item in first["objects"]
        )
        assert models.attribute_calls == 0
        models.missing = True
        second = client.post(
            "/v1/tracks", content=_jpeg(), headers=_headers(2, 1.2)
        ).json()
        assert all(
            item["observation_state"] == "predicted" for item in second["objects"]
        )
        assert all(
            item["last_observed_frame_sequence"] == 1 for item in second["objects"]
        )
        assert all(
            item["last_observed_timestamp_seconds"] == 1.0 for item in second["objects"]
        )
        models.missing = False
        response = client.post(
            "/v1/tracks?include_attributes=true",
            content=_jpeg(),
            headers=_headers(3, 1.4),
        )
        assert response.status_code == 200
        assert engine._attribute_idle.wait(1.0)
        assert models.attribute_calls == 1
        assert client.get("/health/tracking").json()["ready"] is True
        assert models.detect_calls == 3
        legacy = client.post(
            "/v1/labels", content=_jpeg(), headers=_headers(4, 1.6)
        ).json()
        assert all("observation_state" not in item for item in legacy["objects"])


def test_real_time_expiry_survives_dense_ticks_but_not_three_second_gap():
    tracker = IoUTracker(max_idle_frames=2, visible_idle_frames=100)
    person = Detection("person", 0.9, (10, 10, 50, 100))
    first = tracker.update((person,), frame_sequence=1, timestamp_seconds=0)[0]
    for sequence in range(2, 30):
        held = tracker.update(
            (), frame_sequence=sequence, timestamp_seconds=sequence / 10
        )
        assert held[0].track_id == first.track_id
        assert held[0].last_observed_timestamp_seconds == 0
        assert held[0].last_frame_sequence == 1
    assert tracker.update((), frame_sequence=30, timestamp_seconds=3.01) == ()
    next_person = tracker.update((person,), frame_sequence=31, timestamp_seconds=3.02)[
        0
    ]
    assert next_person.track_id != first.track_id


def test_slow_attribute_warmup_does_not_delay_tracking_http(monkeypatch):
    started, release = Event(), Event()
    models = _Models()
    engine = RichLabelEngine(models)

    def attribute_probe():
        started.set()
        assert release.wait(3)

    engine.warmup_attributes = attribute_probe
    monkeypatch.setattr("workers.paddle_rich_label.app.RichLabelEngine", lambda: engine)
    try:
        with TestClient(create_app()) as client:
            assert started.wait(1)
            assert client.get("/health/tracking").status_code == 200
            assert client.get("/health/ready").status_code == 503
            assert (
                client.post(
                    "/v1/tracks", content=_jpeg(), headers=_headers()
                ).status_code
                == 200
            )
            assert (
                client.post(
                    "/v1/labels", content=_jpeg(), headers=_headers(2, 1.2)
                ).status_code
                == 503
            )
            release.set()
    finally:
        release.set()


def _snapshot(sequence=1, timestamp=1.0):
    return RichLabelSnapshot(
        "session",
        sequence,
        timestamp,
        (
            RichLabelObject(
                "P1",
                "person",
                NormalizedBBox(0.1, 0.1, 0.5, 0.9),
                observation_state="observed",
                last_observed_timestamp_seconds=timestamp,
                last_observed_frame_sequence=sequence,
            ),
        ),
    )


def test_host_tracks_mode_uses_independent_readiness_and_one_request():
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(
            200,
            json=(
                {"ready": True, "gpu_ready": True}
                if request.method == "GET"
                else _snapshot().as_dict()
            ),
        )

    with RichLabelHTTPClient(
        "http://worker:8091/v1/labels",
        mode="tracks",
        include_attributes=False,
        transport=httpx.MockTransport(handle),
    ) as client:
        assert client.gpu_ready()
        snapshot = client.infer(
            _jpeg(), stream_session_id="session", frame_sequence=1, timestamp_seconds=1
        )
    assert str(requests[0].url) == "http://worker:8091/health/tracking"
    assert (
        str(requests[1].url) == "http://worker:8091/v1/tracks?include_attributes=false"
    )
    assert snapshot == _snapshot()


def test_tracking_client_rejects_missing_and_future_observation_metadata():
    payload = _snapshot().as_dict()
    payload["objects"][0].pop("observation_state")
    with pytest.raises(RichLabelProtocolError):
        RichLabelSnapshot.from_dict(payload)
    with pytest.raises(ValueError, match="future"):
        replace(_snapshot(), timestamp_seconds=0.5)
    payload = _snapshot().as_dict()
    for field in (
        "observation_state",
        "last_observed_timestamp_seconds",
        "last_observed_frame_sequence",
    ):
        payload["objects"][0].pop(field)
    with RichLabelHTTPClient(
        "http://worker/v1/labels",
        mode="tracks",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json=payload),
        ),
    ) as client:
        with pytest.raises(RichLabelProtocolError, match="observation metadata"):
            client.infer(
                _jpeg(),
                stream_session_id="session",
                frame_sequence=1,
                timestamp_seconds=1,
            )


def test_interpolation_keeps_last_real_observation_and_does_not_promote_prediction():
    selected = select_rich_labels(
        (_snapshot(), _snapshot(3, 1.4)),
        1.2,
        frame_sequence=2,
        max_gap_seconds=1,
        max_hold_seconds=1,
    )
    assert selected.mode == "interpolated"
    item = selected.objects[0]
    assert item.observation_state == "predicted"
    assert item.last_observed_timestamp_seconds == 1
    assert item.last_observed_frame_sequence == 1


def test_shared_coordinator_delivers_snapshot_outside_its_condition_lock():
    received = Event()
    records = []

    class Client:
        def infer(self, jpeg, **metadata):
            return _snapshot(metadata["frame_sequence"], metadata["timestamp_seconds"])

    def consume(snapshot):
        # A separate thread must be able to inspect history while the callback runs.
        from threading import Thread

        done = Event()
        thread = Thread(
            target=lambda: (records.extend(coordinator.history()), done.set())
        )
        thread.start()
        assert done.wait(1)
        thread.join()
        received.set()

    coordinator = RichLabelCoordinator(Client(), on_snapshot=consume)
    try:
        coordinator.reset("session")
        assert coordinator.submit(
            _jpeg(), stream_session_id="session", frame_sequence=1, timestamp_seconds=1
        )
        assert received.wait(2)
        assert records == [_snapshot()]
        assert coordinator.stats()["started"] == 1
    finally:
        coordinator.close()
