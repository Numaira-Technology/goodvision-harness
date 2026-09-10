from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import cv2
import jsonschema
import numpy as np
from fastapi.testclient import TestClient

from goodvision_harness.config import load_config
from goodvision_harness.contracts import (
    AnalysisResult,
    EventJudgment,
    FrameReference,
    GroundedObject,
    NormalizedBBox,
)
from goodvision_harness.event_archive import EventArchiveSettings
from goodvision_harness.service import HarnessRuntime, create_app

ROOT = Path(__file__).resolve().parents[1]


def observation(
    timestamp: float, message: str, detected: bool = True
) -> AnalysisResult:
    return AnalysisResult(
        message_id=message,
        source_id="camera",
        stream_session_id="session",
        sequence=int(timestamp * 5) + 1,
        emitted_at="2026-09-10T00:00:00Z",
        model_name="standard",
        model_latency_seconds=0.01,
        frame=FrameReference(
            int(timestamp * 5),
            timestamp,
            "2026-09-10T00:00:00Z",
            160,
            90,
        ),
        judgments=(
            EventJudgment(
                "smoking",
                detected,
                0.9,
                "可见烟具。 / A smoking item is visible.",
                ("item",) if detected else (),
            ),
        ),
        objects=(
            GroundedObject(
                "item",
                "cigarette",
                NormalizedBBox(0.2, 0.2, 0.3, 0.3),
                0.9,
                {"subject_id": "subject-test", "person_id": "Alice"},
            ),
        )
        if detected
        else (),
    )


def make_runtime(tmp_path: Path, monkeypatch) -> HarnessRuntime:
    monkeypatch.setenv("HARNESS_AUTOSTART", "false")
    config = load_config(ROOT / "config.yaml")
    config = replace(
        config,
        events=EventArchiveSettings(directory=tmp_path / "evidence"),
        rich_labeling=replace(
            config.rich_labeling, enabled=False, tracking_enabled=False
        ),
    )
    return HarnessRuntime(config)


def test_retired_photos_do_not_initialize_faces(tmp_path, monkeypatch) -> None:
    runtime = make_runtime(tmp_path, monkeypatch)

    def forbidden():
        raise AssertionError("retired route must not load recognition")

    monkeypatch.setattr(runtime, "_ensure_face_runtime", forbidden)
    with TestClient(create_app(runtime_factory=lambda: runtime)) as client:
        for path in ("/v1/faces/strangers", "/v1/faces/strangers/example/photo"):
            response = client.get(path)
            assert response.status_code == 410
            assert response.json()["detail"]["events_url"] == "/v1/events"
        assert client.get("/v1/events/missing").status_code == 404
        assert client.get("/v1/events?since=5&until=1").status_code == 400
        assert client.get("/v1/events?limit=201").status_code == 422
        paths = client.get("/openapi.json").json()["paths"]
        video = paths["/v1/events/{event_id}/video"]["get"]["responses"]
        assert "video/mp4" in video["200"]["content"]
        assert {"202", "206", "404", "416", "503"} <= video.keys()
        stream = paths["/v1/events/stream"]["get"]["responses"]
        assert "text/event-stream" in stream["200"]["content"]


def test_persistent_event_real_video_range_and_immediate_negative(
    tmp_path, monkeypatch
):
    runtime = make_runtime(tmp_path, monkeypatch)
    with TestClient(create_app(runtime_factory=lambda: runtime)) as client:
        archive = runtime.event_archive()
        archive.start_session("session", "camera")
        for index in range(21):
            ok, jpeg = cv2.imencode(
                ".jpg",
                np.full((90, 160, 3), index * 10, dtype=np.uint8),
            )
            assert ok
            archive.submit_frame(jpeg.tobytes(), index / 5, index, "session")
        positive = observation(1, "first")
        archive.submit_result(positive)
        runtime._on_analysis_result(positive)
        negative = observation(2, "pause", False)
        runtime._on_analysis_result(negative)
        archive.submit_result(negative)
        # A live negative is observable immediately while the archive is active.
        assert (
            client.get("/v1/results/latest").json()["judgments"][0]["detected"] is False
        )
        deadline = time.monotonic() + 10
        while True:
            assert archive.flush()
            page = client.get("/v1/events?message_id=first").json()
            if page["events"] and page["events"][0]["evidence"].get("available"):
                break
            assert time.monotonic() < deadline, archive.status()
            time.sleep(0.02)
        event = page["events"][0]
        assert event["status"] == "active"
        assert event["subject_id"] == "subject-test"
        schema = json.loads((ROOT / "schemas/event-v1.schema.json").read_text())
        jsonschema.validate(event, schema)
        url = event["evidence"]["available"]["url"]
        response = client.get(url)
        assert response.status_code == 200
        assert response.headers["content-type"] == "video/mp4"
        assert response.content[4:8] == b"ftyp"
        ranged = client.get(url, headers={"Range": "bytes=0-31"})
        assert ranged.status_code == 206
        assert ranged.content == response.content[:32]
        assert client.get(url + "999").status_code == 404
        archive.end_session("session")
        assert archive.flush()
    restarted = make_runtime(tmp_path, monkeypatch)
    with TestClient(create_app(runtime_factory=lambda: restarted)) as client:
        restarted.event_archive().flush()
        restored = client.get("/v1/events?message_id=first").json()["events"]
        assert len(restored) == 1
        assert restored[0]["event_id"] == event["event_id"]
        assert client.get(url).content == response.content


def test_identity_gallery_changes_reach_active_registry(tmp_path, monkeypatch):
    runtime = make_runtime(tmp_path, monkeypatch)
    changes = []
    runtime._session = SimpleNamespace(
        person_registry=SimpleNamespace(
            forget_person=lambda person_id: changes.append(("delete", person_id)),
            allow_person=lambda person_id: changes.append(("enroll", person_id)),
        )
    )
    face_runtime = SimpleNamespace(
        delete_person=lambda _: True,
        enroll=lambda person_id, name, _: SimpleNamespace(
            person_id=person_id,
            name=name,
            sample_count=1,
        ),
    )
    monkeypatch.setattr(runtime, "_ensure_face_runtime", lambda: face_runtime)
    assert runtime.delete_face("Alice")
    runtime.enroll_face("Alice", "Alice", b"fixture")
    assert changes == [("delete", "Alice"), ("enroll", "Alice")]
    runtime._session = None
    runtime.stop()
