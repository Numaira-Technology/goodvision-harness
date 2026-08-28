from __future__ import annotations

import sys
import threading
import time
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
    _model_kwargs,
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


def test_cpu_models_keep_paddlex_optimized_defaults() -> None:
    assert _model_kwargs("PicoDet-S", "", "cpu") == {
        "model_name": "PicoDet-S",
        "device": "cpu",
    }


def test_non_cpu_models_keep_paddlex_device_defaults() -> None:
    assert _model_kwargs("PP-YOLOE_plus-S", "/models/detector", "gpu:0") == {
        "model_name": "PP-YOLOE_plus-S",
        "device": "gpu:0",
        "model_dir": "/models/detector",
    }


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


def test_iou_tracker_keeps_id_during_fast_non_overlapping_motion() -> None:
    tracker = IoUTracker(match_iou=0.3, max_idle_frames=5)
    first = tracker.update(
        (Detection("person", 0.9, (0.0, 0.0, 20.0, 40.0)),),
        frame_sequence=0,
    )
    second = tracker.update(
        (Detection("person", 0.91, (18.0, 0.0, 38.0, 40.0)),),
        frame_sequence=1,
    )
    third = tracker.update(
        (Detection("person", 0.92, (37.0, 0.0, 57.0, 40.0)),),
        frame_sequence=2,
    )

    assert first[0].track_id == second[0].track_id == third[0].track_id
    assert tracker._next_id == 2


def test_iou_tracker_recovers_same_id_after_short_missed_detection() -> None:
    tracker = IoUTracker(match_iou=0.3, max_idle_frames=5)
    first = tracker.update(
        (Detection("person", 0.9, (0.0, 0.0, 20.0, 40.0)),),
        frame_sequence=0,
    )
    moving = tracker.update(
        (Detection("person", 0.91, (12.0, 0.0, 32.0, 40.0)),),
        frame_sequence=1,
    )
    missed = tracker.update((), frame_sequence=2)
    recovered = tracker.update(
        (Detection("person", 0.92, (36.0, 0.0, 56.0, 40.0)),),
        frame_sequence=3,
    )

    assert missed == ()
    assert first[0].track_id == moving[0].track_id == recovered[0].track_id
    assert tracker._next_id == 2


def test_iou_tracker_uses_motion_to_preserve_ids_through_crossing() -> None:
    tracker = IoUTracker(match_iou=0.3, max_idle_frames=5)

    initial = tracker.update(
        (
            Detection("person", 0.9, (0.0, 0.0, 20.0, 40.0)),
            Detection("person", 0.9, (80.0, 0.0, 100.0, 40.0)),
        ),
        frame_sequence=0,
    )
    left_to_right_id = _track_at_x(initial, 10.0).track_id
    right_to_left_id = _track_at_x(initial, 90.0).track_id

    tracker.update(
        (
            # Reverse detector order on every step so input order cannot carry IDs.
            Detection("person", 0.91, (55.0, 0.0, 75.0, 40.0)),
            Detection("person", 0.92, (25.0, 0.0, 45.0, 40.0)),
        ),
        frame_sequence=1,
    )
    crossed = tracker.update(
        (
            Detection("person", 0.93, (25.0, 0.0, 45.0, 40.0)),
            Detection("person", 0.94, (55.0, 0.0, 75.0, 40.0)),
        ),
        frame_sequence=2,
    )
    separated = tracker.update(
        (
            Detection("person", 0.95, (80.0, 0.0, 100.0, 40.0)),
            Detection("person", 0.95, (0.0, 0.0, 20.0, 40.0)),
        ),
        frame_sequence=3,
    )

    assert _track_at_x(crossed, 65.0).track_id == left_to_right_id
    assert _track_at_x(crossed, 35.0).track_id == right_to_left_id
    assert _track_at_x(separated, 90.0).track_id == left_to_right_id
    assert _track_at_x(separated, 10.0).track_id == right_to_left_id
    assert tracker._next_id == 3


def _track_at_x(tracks: tuple[Any, ...], center_x: float) -> Any:
    return min(
        tracks,
        key=lambda track: abs(((track.bbox[0] + track.bbox[2]) / 2) - center_x),
    )


def test_engine_returns_normalized_boxes_and_estimated_attributes() -> None:
    engine = RichLabelEngine(_Models())

    first = engine.infer(
        np.zeros((100, 120, 3), dtype=np.uint8),
        stream_session_id="session-a",
        frame_sequence=7,
        timestamp_seconds=1.5,
    )

    assert first["stream_session_id"] == "session-a"
    assert first["frame_sequence"] == 7
    assert [item["track_id"] for item in first["objects"]] == ["O0002", "P0001"]
    person = first["objects"][1]
    assert person["bbox"] == {
        "x1": 10 / 120,
        "y1": 0.1,
        "x2": 70 / 120,
        "y2": 0.95,
    }
    assert person["attributes"] == {}
    assert engine._attribute_idle.wait(1.0)

    second = engine.infer(
        np.zeros((100, 120, 3), dtype=np.uint8),
        stream_session_id="session-a",
        frame_sequence=8,
        timestamp_seconds=1.6,
    )
    person = second["objects"][1]
    assert person["attributes"]["estimated_age_range"] == "18-60"


def test_attribute_inference_never_blocks_detector_or_queues_tasks() -> None:
    attribute_started = threading.Event()
    release_attribute = threading.Event()
    calls = 0

    class _BlockingModels(_Models):
        def attributes(
            self,
            crops: list[np.ndarray[Any, Any]],
        ) -> tuple[dict[str, Any], ...]:
            nonlocal calls
            calls += 1
            attribute_started.set()
            assert release_attribute.wait(2.0)
            return super().attributes(crops)

    engine = RichLabelEngine(_BlockingModels())
    started = time.perf_counter()
    first = engine.infer(
        np.zeros((100, 120, 3), dtype=np.uint8),
        stream_session_id="non-blocking-session",
        frame_sequence=1,
        timestamp_seconds=0.1,
    )
    first_latency = time.perf_counter() - started

    try:
        assert first_latency < 0.5
        assert first["objects"][1]["attributes"] == {}
        assert attribute_started.wait(1.0)

        for frame_sequence in range(2, 7):
            started = time.perf_counter()
            result = engine.infer(
                np.zeros((100, 120, 3), dtype=np.uint8),
                stream_session_id="non-blocking-session",
                frame_sequence=frame_sequence,
                timestamp_seconds=frame_sequence / 10,
            )
            assert time.perf_counter() - started < 0.5
            assert len(result["objects"]) == 2
        assert calls == 1
    finally:
        release_attribute.set()

    assert engine._attribute_idle.wait(1.0)
    cached = engine.infer(
        np.zeros((100, 120, 3), dtype=np.uint8),
        stream_session_id="non-blocking-session",
        frame_sequence=7,
        timestamp_seconds=0.7,
    )
    assert cached["objects"][1]["attributes"]["estimated_age_range"] == "18-60"
    assert calls == 1


def test_old_attribute_task_cannot_pollute_reset_session_or_reused_track_id() -> None:
    old_started = threading.Event()
    release_old = threading.Event()
    new_finished = threading.Event()
    calls = 0

    class _SessionModels(_Models):
        def attributes(
            self,
            crops: list[np.ndarray[Any, Any]],
        ) -> tuple[dict[str, Any], ...]:
            nonlocal calls
            calls += 1
            if calls == 1:
                old_started.set()
                assert release_old.wait(2.0)
                marker = "old-session"
            else:
                marker = "new-session"
                new_finished.set()
            return tuple({"marker": marker} for _ in crops)

    engine = RichLabelEngine(_SessionModels())
    engine.infer(
        np.zeros((100, 120, 3), dtype=np.uint8),
        stream_session_id="session-old",
        frame_sequence=1,
        timestamp_seconds=0.1,
    )
    assert old_started.wait(1.0)

    reset = engine.infer(
        np.zeros((100, 120, 3), dtype=np.uint8),
        stream_session_id="session-new",
        frame_sequence=1,
        timestamp_seconds=0.1,
    )
    new_person = reset["objects"][1]
    assert new_person["track_id"] == "P0001"
    assert new_person["attributes"] == {}

    release_old.set()
    assert engine._attribute_idle.wait(1.0)
    after_old = engine.infer(
        np.zeros((100, 120, 3), dtype=np.uint8),
        stream_session_id="session-new",
        frame_sequence=2,
        timestamp_seconds=0.2,
    )
    assert after_old["objects"][1]["attributes"] == {}
    assert new_finished.wait(1.0)
    assert engine._attribute_idle.wait(1.0)

    cached = engine.infer(
        np.zeros((100, 120, 3), dtype=np.uint8),
        stream_session_id="session-new",
        frame_sequence=3,
        timestamp_seconds=0.3,
    )
    assert cached["objects"][1]["track_id"] == "P0001"
    assert cached["objects"][1]["attributes"] == {"marker": "new-session"}
    assert calls == 2


def test_old_attribute_task_cannot_update_an_expired_track() -> None:
    attribute_started = threading.Event()
    release_attribute = threading.Event()

    class _ExpiringModels(_Models):
        show_person = True

        def detect(self, image: np.ndarray[Any, Any]) -> tuple[Detection, ...]:
            del image
            if not self.show_person:
                return ()
            return (Detection("person", 0.94, (10.0, 10.0, 70.0, 95.0)),)

        def attributes(
            self,
            crops: list[np.ndarray[Any, Any]],
        ) -> tuple[dict[str, Any], ...]:
            attribute_started.set()
            assert release_attribute.wait(2.0)
            return tuple({"marker": "expired"} for _ in crops)

    models = _ExpiringModels()
    engine = RichLabelEngine(models)
    first = engine.infer(
        np.zeros((100, 120, 3), dtype=np.uint8),
        stream_session_id="same-session",
        frame_sequence=1,
        timestamp_seconds=0.1,
    )
    assert first["objects"][0]["track_id"] == "P0001"
    assert attribute_started.wait(1.0)

    models.show_person = False
    empty = engine.infer(
        np.zeros((100, 120, 3), dtype=np.uint8),
        stream_session_id="same-session",
        frame_sequence=22,
        timestamp_seconds=2.2,
    )
    assert empty["objects"] == []

    release_attribute.set()
    assert engine._attribute_idle.wait(1.0)
    assert "P0001" not in engine.tracker._tracks

    models.show_person = True
    replacement = engine.infer(
        np.zeros((100, 120, 3), dtype=np.uint8),
        stream_session_id="same-session",
        frame_sequence=23,
        timestamp_seconds=2.3,
    )
    assert replacement["objects"][0]["track_id"] == "P0002"
    assert replacement["objects"][0]["attributes"] == {}
    assert engine._attribute_idle.wait(1.0)


def test_shutdown_invalidates_in_flight_attributes_and_stops_new_tasks() -> None:
    attribute_started = threading.Event()
    release_attribute = threading.Event()
    calls = 0

    class _ShutdownModels(_Models):
        def attributes(
            self,
            crops: list[np.ndarray[Any, Any]],
        ) -> tuple[dict[str, Any], ...]:
            nonlocal calls
            calls += 1
            attribute_started.set()
            assert release_attribute.wait(2.0)
            return tuple({"marker": "too-late"} for _ in crops)

    engine = RichLabelEngine(_ShutdownModels())
    engine.attribute_shutdown_timeout_seconds = 0.01
    engine.infer(
        np.zeros((100, 120, 3), dtype=np.uint8),
        stream_session_id="shutdown-session",
        frame_sequence=1,
        timestamp_seconds=0.1,
    )
    assert attribute_started.wait(1.0)

    with TestClient(create_app(engine)) as client:
        assert client.get("/health/live").status_code == 200
    assert engine._closed

    release_attribute.set()
    assert engine._attribute_idle.wait(1.0)
    after_shutdown = engine.infer(
        np.zeros((100, 120, 3), dtype=np.uint8),
        stream_session_id="shutdown-session",
        frame_sequence=2,
        timestamp_seconds=0.2,
    )
    assert after_shutdown["objects"][1]["attributes"] == {}
    assert calls == 1


def test_malformed_attribute_results_back_off_without_losing_boxes() -> None:
    calls = 0

    class _MalformedModels(_Models):
        def attributes(
            self,
            crops: list[np.ndarray[Any, Any]],
        ) -> tuple[dict[str, Any], ...]:
            nonlocal calls
            del crops
            calls += 1
            return ()

    engine = RichLabelEngine(_MalformedModels())
    first = engine.infer(
        np.zeros((100, 120, 3), dtype=np.uint8),
        stream_session_id="malformed-session",
        frame_sequence=1,
        timestamp_seconds=0.1,
    )
    assert len(first["objects"]) == 2
    assert engine._attribute_idle.wait(1.0)

    second = engine.infer(
        np.zeros((100, 120, 3), dtype=np.uint8),
        stream_session_id="malformed-session",
        frame_sequence=2,
        timestamp_seconds=0.2,
    )
    assert len(second["objects"]) == 2
    assert calls == 1


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
    assert engine._attribute_idle.wait(1.0)

    second = engine.infer(
        np.zeros((100, 120, 3), dtype=np.uint8),
        stream_session_id="session-degraded",
        frame_sequence=2,
        timestamp_seconds=0.2,
    )

    assert len(second["objects"]) == 1
    assert calls.count("PicoDet-S") == 1
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


def test_gpu_worker_is_isolated_and_targets_blackwell_supported_paddle() -> None:
    gpu_dockerfile = (
        ROOT / "workers" / "paddle_rich_label" / "Dockerfile.gpu"
    ).read_text(encoding="utf-8")
    cpu_dockerfile = (
        ROOT / "workers" / "paddle_rich_label" / "Dockerfile"
    ).read_text(encoding="utf-8")
    gpu_compose = (ROOT / "compose.gpu.yaml").read_text(encoding="utf-8")
    delivery_compose = (
        ROOT / "compose.rich-gpu.delivery.yaml"
    ).read_text(encoding="utf-8")

    assert "paddlepaddle-gpu==3.3.0" in gpu_dockerfile
    assert "packages/stable/cu129" in gpu_dockerfile
    assert "paddle/version/__init__.py" in gpu_dockerfile
    assert "assert '120' in metadata" in gpu_dockerfile
    assert "RICH_DEVICE=gpu:0" in gpu_dockerfile
    assert "USER 10002:10002" in gpu_dockerfile
    assert "paddlepaddle==3.2.2" in cpu_dockerfile
    assert "paddlepaddle-gpu" not in cpu_dockerfile

    for compose_text in (gpu_compose, delivery_compose):
        assert "PP-YOLOE_plus-L" in compose_text
        assert "driver: nvidia" in compose_text
        assert "count: 1" in compose_text
        assert "capabilities: [gpu]" in compose_text

    gpu_smoke = (
        ROOT / "workers" / "paddle_rich_label" / "gpu_smoke.py"
    ).read_text(encoding="utf-8")
    assert "paddle.device.is_compiled_with_cuda()" in gpu_smoke
    assert "paddle.version.cuda_archs()" in gpu_smoke
    assert "paddle.utils.run_check()" in gpu_smoke
    assert "models.detect(image)" in gpu_smoke
    assert "models.attributes((image,))" in gpu_smoke
