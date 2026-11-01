"""The detector worker's /v1/labels output must satisfy the harness contract.

Uses DETECTOR_MODEL=dummy so no model weights / GPU are needed. Proves:
  1. the response parses into goodvision_harness.rich_labeling.RichLabelSnapshot;
  2. track_ids and source ordering stay isolated across interleaved sessions;
  3. health endpoints respond.
Run: PYTHONPATH=src python -m pytest workers/detector_rich_label/test_contract.py
"""

import importlib
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_HERE))  # tracker / detectors / app
sys.path.insert(0, str(_ROOT / "src"))  # goodvision_harness
sys.path.insert(0, str(_ROOT))  # worker package

os.environ["DETECTOR_MODEL"] = "dummy"
from fastapi.testclient import TestClient  # noqa: E402

from goodvision_harness.rich_labeling import RichLabelSnapshot  # noqa: E402
from workers.detector_rich_label import app as worker_app  # noqa: E402


def _jpeg(w=320, h=240) -> bytes:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def _headers(seq: int, ts: float) -> dict:
    return {
        "x-goodvision-stream-session-id": "sess-1",
        "x-goodvision-frame-sequence": str(seq),
        "x-goodvision-timestamp-seconds": str(ts),
        "Content-Type": "image/jpeg",
    }


@pytest.fixture()
def client():
    importlib.reload(worker_app)  # rebuild app with DETECTOR_MODEL=dummy
    with TestClient(worker_app.app) as c:
        yield c


def test_response_parses_into_harness_snapshot(client):
    r = client.post("/v1/labels", headers=_headers(1, 0.10), content=_jpeg())
    assert r.status_code == 200, r.text
    snap = RichLabelSnapshot.from_dict(r.json())  # the real harness parser
    assert snap.stream_session_id == "sess-1"
    assert snap.frame_sequence == 1
    assert len(snap.objects) == 1
    obj = snap.objects[0]
    assert obj.label == "person"
    assert obj.track_id.startswith("P")
    assert 0.0 <= obj.bbox.x1 < obj.bbox.x2 <= 1.0


def test_track_id_is_stable_across_frames(client):
    ids = []
    for i in range(1, 4):
        r = client.post("/v1/labels", headers=_headers(i, i * 0.1), content=_jpeg())
        assert r.status_code == 200, r.text
        snap = RichLabelSnapshot.from_dict(r.json())
        ids.append(snap.objects[0].track_id)
    assert len(set(ids)) == 1  # same person -> one stable track_id


def test_tracks_route_carries_observation_metadata(client):
    # The harness RichLabelHTTPClient uses mode="tracks" -> POST /v1/tracks and
    # rejects any object missing observation_state. This worker must serve that
    # route and populate the tracking metadata (the /v1/labels-only gap that
    # made every harness rich-label call fail).
    r = client.post("/v1/tracks", headers=_headers(1, 0.10), content=_jpeg())
    assert r.status_code == 200, r.text
    snap = RichLabelSnapshot.from_dict(r.json())
    assert len(snap.objects) == 1
    for obj in snap.objects:
        assert obj.observation_state is not None
        assert obj.track_id.startswith("P")


@pytest.fixture()
def fake_engine(monkeypatch):
    from tracker import Detection

    built: list[str] = []
    used: list[str] = []
    warmed: list[str] = []
    left = Detection("person", 0.9, (5.0, 10.0, 15.0, 30.0))
    right = Detection("person", 0.9, (75.0, 10.0, 85.0, 30.0))

    class _Fake:
        def __init__(self, key: str) -> None:
            self.name = key
            self.gpu_ready = True

        def warmup(self) -> None:
            warmed.append(self.name)

        def detect(self, image):
            used.append(self.name)
            marker = int(image[0, 0, 0])
            if marker == 3:
                raise RuntimeError("inference failed")
            return {0: (left, right), 1: (right,), 2: (left,)}[marker]

    def _factory(key: str):
        built.append(key)
        return _Fake(key)

    monkeypatch.setattr(worker_app, "build_detector", _factory)
    return worker_app.DetectorEngine(), built, used, warmed


def _infer(engine, session, seq, ts, *, model="yolov8-m", marker=0):
    image = np.zeros((100, 100, 3), np.uint8)
    image[0, 0, 0] = marker
    return engine.infer(
        image,
        stream_session_id=session,
        frame_sequence=seq,
        timestamp_seconds=ts,
        model_key=model,
    )


def _observed_ids(response):
    return [
        obj["track_id"]
        for obj in response["objects"]
        if obj["observation_state"] == "observed"
    ]


def test_console_selection_switches_model_live(fake_engine):
    # Model switches reset only the selected session; detectors stay cached.
    eng, built, used, warmed = fake_engine

    r1 = _infer(eng, "a", 1, 0.1)
    assert used[-1] == "yolov8-m"
    assert _observed_ids(r1) == ["P0001", "P0002"]
    _infer(eng, "b", 1, 10.0)
    r2 = _infer(eng, "a", 2, 0.2, marker=1)
    assert _observed_ids(r2) == ["P0002"]

    r3 = _infer(eng, "a", 3, 0.3, model="rfdetr-l", marker=1)
    assert used[-1] == "rfdetr-l"
    assert _observed_ids(r3) == ["P0001"]
    assert len(r3["objects"]) == 1
    assert _observed_ids(_infer(eng, "b", 2, 10.1, marker=1)) == ["P0002"]
    assert _observed_ids(_infer(eng, "a", 4, 0.4, marker=1)) == ["P0001"]
    assert built == warmed == ["yolov8-m", "rfdetr-l"]

    with pytest.raises(ValueError):
        _infer(eng, "a", 5, 0.5, model="no-such-model")


def test_interleaved_sessions_preserve_tracks_and_observation_metadata(fake_engine):
    eng, built, _, warmed = fake_engine
    _infer(eng, "a", 1, 0.1)
    _infer(eng, "b", 40, 10.0, marker=2)
    response = _infer(eng, "a", 2, 0.2, marker=1)
    snapshot = RichLabelSnapshot.from_dict(response)
    # A's right person was P0002. A global tracker reset would return P0001.
    assert _observed_ids(response) == ["P0002"]
    assert len(snapshot.objects) == 2
    assert response["objects"][0]["observation_state"] == "predicted"
    assert response["objects"][0]["last_observed_frame_sequence"] == 1
    assert response["objects"][0]["last_observed_timestamp_seconds"] == 0.1
    assert built == warmed == ["yolov8-m"]


@pytest.mark.parametrize("seq, ts", [(5, 0.6), (6, 0.4)])
@pytest.mark.parametrize("model", ["yolov8-m", "rfdetr-l"])
def test_source_order_is_per_session_even_when_model_changes(
    fake_engine, seq, ts, model
):
    eng, _, used, _ = fake_engine
    _infer(eng, "a", 5, 0.5)
    _infer(eng, "b", 1, 0.1)
    with pytest.raises(ValueError, match="source frame metadata must increase"):
        _infer(eng, "a", seq, ts, model=model, marker=1)
    assert len(used) == 2  # reject stale frames before inference or model switch
    assert _observed_ids(_infer(eng, "a", 6, 0.5, marker=1)) == ["P0002"]


def test_failed_model_switch_preserves_session_state(fake_engine):
    eng, _, _, _ = fake_engine
    _infer(eng, "a", 1, 0.1)
    with pytest.raises(RuntimeError, match="inference failed"):
        _infer(eng, "a", 2, 0.2, model="rfdetr-l", marker=3)
    assert _observed_ids(_infer(eng, "a", 2, 0.2, marker=1)) == ["P0002"]


def test_session_state_is_bounded_and_evicts_least_recently_used(
    fake_engine, monkeypatch
):
    eng, _, _, _ = fake_engine
    monkeypatch.setattr(worker_app, "MAX_TRACKING_SESSIONS", 2)
    _infer(eng, "a", 1, 0.1)
    _infer(eng, "b", 1, 0.1)
    _infer(eng, "a", 2, 0.2, marker=1)  # a stays active; b becomes oldest
    _infer(eng, "c", 1, 0.1)
    assert len(eng._sessions) == 2
    assert _observed_ids(_infer(eng, "a", 3, 0.3, marker=1)) == ["P0002"]
    assert _observed_ids(_infer(eng, "b", 1, 0.1, marker=1)) == ["P0001"]
    assert len(eng._sessions) == 2


def test_bad_metadata_and_health(client):
    # non-increasing frame metadata is rejected
    client.post("/v1/labels", headers=_headers(5, 0.5), content=_jpeg())
    r = client.post("/v1/labels", headers=_headers(5, 0.5), content=_jpeg())
    assert r.status_code == 400
    assert client.get("/health/ready").json()["ready"] is True
