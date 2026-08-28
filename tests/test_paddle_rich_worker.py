from __future__ import annotations

import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from workers.paddle_rich_label.app import (
    Detection,
    IoUTracker,
    PaddleRichModels,
    RichLabelBusyError,
    RichLabelEngine,
    _pedestrian_attributes,
    create_app,
)

ROOT = Path(__file__).resolve().parents[1]


class _Models:
    ready = True

    def detect(self, image: np.ndarray[Any, Any]) -> tuple[Detection, ...]:
        return (
            Detection("person", 0.94, (10.0, 10.0, 70.0, 95.0)),
            Detection("bottle", 0.82, (80.0, 25.0, 105.0, 85.0)),
        )

    def attributes(
        self, crops: list[np.ndarray[Any, Any]]
    ) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "estimated_age_range": "18-60",
                "estimated_gender": "female-presenting",
            }
            for _ in crops
        )


def _jpeg(width: int = 120, height: int = 100) -> bytes:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def test_iou_tracker_keeps_ids_and_resets_cleanly() -> None:
    tracker = IoUTracker(match_iou=0.2, max_idle_frames=4)
    first = tracker.update(
        (Detection("person", 0.9, (10.0, 10.0, 50.0, 80.0)),),
        frame_sequence=1,
    )
    second = tracker.update(
        (Detection("person", 0.92, (13.0, 10.0, 53.0, 80.0)),),
        frame_sequence=2,
    )

    assert first[0].track_id == second[0].track_id == "P0001"

    tracker.reset()
    reset = tracker.update(
        (Detection("bottle", 0.8, (1.0, 1.0, 10.0, 20.0)),),
        frame_sequence=3,
    )
    assert reset[0].track_id == "O0001"


def test_engine_returns_normalized_boxes_and_estimated_attributes() -> None:
    engine = RichLabelEngine(_Models())

    result = engine.infer(
        np.zeros((100, 120, 3), dtype=np.uint8),
        stream_session_id="session-a",
        frame_sequence=7,
        timestamp_seconds=1.5,
    )

    assert result["stream_session_id"] == "session-a"
    assert result["frame_sequence"] == 7
    assert [item["track_id"] for item in result["objects"]] == ["O0002", "P0001"]
    person = result["objects"][1]
    assert person["bbox"] == {
        "x1": 10 / 120,
        "y1": 0.1,
        "x2": 70 / 120,
        "y2": 0.95,
    }
    assert person["attributes"]["estimated_age_range"] == "18-60"


def test_attribute_result_uses_estimate_names() -> None:
    result = _pedestrian_attributes(
        {
            "label_names": [
                "Female(女)",
                "AgeOver60(年龄大于60岁)",
                "Backpack(双肩包)",
            ],
            "scores": [0.91, 0.88, 0.77],
        }
    )

    assert result["estimated_gender"] == "female-presenting"
    assert result["estimated_age_range"] == ">60"
    assert result["appearance_attributes"] == ["Backpack"]


def test_attribute_model_failure_keeps_detector_boxes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    attribute_attempted = threading.Event()

    class _Detector:
        def predict(self, image: Any, **kwargs: Any) -> list[dict[str, Any]]:
            del image, kwargs
            return [
                {
                    "res": {
                        "boxes": [
                            {
                                "label": "person",
                                "score": 0.91,
                                "coordinate": [10, 10, 70, 95],
                            }
                        ]
                    }
                }
            ]

    def create_model(**kwargs: Any) -> Any:
        model_name = str(kwargs["model_name"])
        calls.append(model_name)
        if "pedestrian_attribute" in model_name:
            attribute_attempted.set()
            raise RuntimeError("attribute model unavailable")
        return _Detector()

    monkeypatch.setitem(
        sys.modules,
        "paddlex",
        SimpleNamespace(create_model=create_model),
    )
    models = PaddleRichModels()
    engine = RichLabelEngine(models)

    result = engine.infer(
        np.zeros((100, 120, 3), dtype=np.uint8),
        stream_session_id="session-degraded",
        frame_sequence=1,
        timestamp_seconds=0.1,
    )

    assert len(result["objects"]) == 1
    assert result["objects"][0]["label"] == "person"
    assert result["objects"][0]["attributes"] == {}
    assert attribute_attempted.wait(1.0)

    second = engine.infer(
        np.zeros((100, 120, 3), dtype=np.uint8),
        stream_session_id="session-degraded",
        frame_sequence=2,
        timestamp_seconds=0.2,
    )

    assert len(second["objects"]) == 1
    assert calls.count("PP-YOLOE_plus-S") == 1
    assert calls.count("PP-LCNet_x1_0_pedestrian_attribute") == 1


def test_worker_http_contract_accepts_raw_jpeg() -> None:
    client = TestClient(create_app(RichLabelEngine(_Models())))

    response = client.post(
        "/v1/labels",
        content=_jpeg(),
        headers={
            "Content-Type": "image/jpeg",
            "X-Goodvision-Stream-Session-Id": "session-http",
            "X-Goodvision-Frame-Sequence": "3",
            "X-Goodvision-Timestamp-Seconds": "0.2",
        },
    )

    assert response.status_code == 200
    assert response.json()["stream_session_id"] == "session-http"
    assert len(response.json()["objects"]) == 2


def test_worker_rejects_missing_metadata_without_loading_models() -> None:
    client = TestClient(create_app(RichLabelEngine(_Models())))

    response = client.post(
        "/v1/labels",
        content=_jpeg(),
        headers={"Content-Type": "image/jpeg"},
    )

    assert response.status_code == 400


def test_worker_rejects_concurrent_inference_without_queueing() -> None:
    engine = RichLabelEngine(_Models())
    assert engine._lock.acquire(blocking=False)
    try:
        with pytest.raises(RichLabelBusyError):
            engine.infer(
                np.zeros((100, 120, 3), dtype=np.uint8),
                stream_session_id="busy-session",
                frame_sequence=1,
                timestamp_seconds=0.1,
            )

        client = TestClient(create_app(engine))
        response = client.post(
            "/v1/labels",
            content=_jpeg(),
            headers={
                "Content-Type": "image/jpeg",
                "X-Goodvision-Stream-Session-Id": "busy-session",
                "X-Goodvision-Frame-Sequence": "2",
                "X-Goodvision-Timestamp-Seconds": "0.2",
            },
        )
    finally:
        engine._lock.release()

    assert response.status_code == 429
    assert response.headers["Retry-After"] == "1"


def test_worker_image_uses_the_writable_paddlex_cache_volume() -> None:
    dockerfile = (
        ROOT / "workers" / "paddle_rich_label" / "Dockerfile"
    ).read_text(encoding="utf-8")

    assert "PADDLE_PDX_CACHE_HOME=/models" in dockerfile
    assert "libgl1" in dockerfile
    assert "import cv2, paddlex" in dockerfile
    assert "USER 10002:10002" in dockerfile

    requirements = (
        ROOT / "workers" / "paddle_rich_label" / "requirements.txt"
    ).read_text(encoding="utf-8")
    assert "opencv-python-headless" not in requirements
