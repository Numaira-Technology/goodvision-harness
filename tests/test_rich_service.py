from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi.testclient import TestClient

from goodvision_harness.config import load_config
from goodvision_harness.contracts import NormalizedBBox
from goodvision_harness.rich_labeling import (
    RichLabelObject,
    RichLabelSelection,
    RichLabelSnapshot,
)
from goodvision_harness.service import HarnessRuntime, create_app

ROOT = Path(__file__).resolve().parents[1]


def _jpeg() -> bytes:
    image = np.zeros((160, 240, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def _snapshot() -> RichLabelSnapshot:
    return RichLabelSnapshot(
        stream_session_id="stream-rich",
        frame_sequence=4,
        timestamp_seconds=1.0,
        objects=(
            RichLabelObject(
                track_id="P0001",
                label="person",
                bbox=NormalizedBBox(0.2, 0.15, 0.75, 0.9),
                confidence=0.94,
                attributes={
                    "estimated_age_range": "18-60",
                    "estimated_gender": "female-presenting",
                },
            ),
        ),
    )


class _RichCoordinator:
    def __init__(self) -> None:
        self.snapshot = _snapshot()
        self.submissions: list[dict[str, Any]] = []
        self.resets: list[str | None] = []
        self.closed = False

    def submit(self, jpeg: bytes, **metadata: Any) -> bool:
        self.submissions.append({"jpeg": jpeg, **metadata})
        return True

    def reset(self, stream_session_id: str | None = None) -> None:
        self.resets.append(stream_session_id)

    def select(
        self,
        stream_session_id: str,
        timestamp_seconds: float,
        *,
        frame_sequence: int | None = None,
    ) -> RichLabelSelection:
        assert stream_session_id == "stream-rich"
        assert timestamp_seconds == 1.0
        assert frame_sequence == 4
        return RichLabelSelection(self.snapshot, "exact", 4, 4)

    def latest(self, stream_session_id: str | None = None) -> RichLabelSnapshot:
        assert stream_session_id == "stream-rich"
        return self.snapshot

    def stats(self) -> dict[str, Any]:
        return {"state": "ready", "submitted": len(self.submissions)}

    def close(self, *, timeout_seconds: float = 5.0) -> None:
        assert timeout_seconds == 5.0
        self.closed = True


def _runtime(coordinator: _RichCoordinator) -> HarnessRuntime:
    runtime = HarnessRuntime(
        load_config(ROOT / "config.yaml"),
        rich_label_coordinator=coordinator,  # type: ignore[arg-type]
    )
    runtime.autostart = False
    runtime._active_rich_session_id = "stream-rich"
    return runtime


def test_runtime_queues_display_frames_and_renders_rich_labels() -> None:
    coordinator = _RichCoordinator()
    runtime = _runtime(coordinator)
    source = _jpeg()
    try:
        runtime._on_display_frame(source, 1.0, 4)

        assert coordinator.submissions == [
            {
                "jpeg": source,
                "stream_session_id": "stream-rich",
                "frame_sequence": 4,
                "timestamp_seconds": 1.0,
            }
        ]
        _, preview = runtime.preview_frames.latest() or (0, None)
        assert preview is not None
        assert preview.jpeg == source

        rendered = runtime._render_rich_label_jpeg(source, 1.0, 4)
        image = cv2.imdecode(np.frombuffer(rendered, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert image is not None
        assert np.count_nonzero(image) > 0
        assert runtime.status()["rich_labeling"]["state"] == "ready"
    finally:
        runtime.stop()
    assert coordinator.closed


def test_latest_rich_labels_endpoint_returns_worker_snapshot() -> None:
    coordinator = _RichCoordinator()
    runtime = _runtime(coordinator)
    app = create_app(lambda: runtime)

    with TestClient(app) as client:
        response = client.get("/v1/rich-labels/latest")

    assert response.status_code == 200
    assert response.headers["Cache-Control"] == "no-store"
    assert response.json()["objects"][0]["track_id"] == "P0001"
    assert response.json()["objects"][0]["attributes"] == {
        "estimated_age_range": "18-60",
        "estimated_gender": "female-presenting",
    }


def test_rich_label_queue_failure_does_not_interrupt_live_preview() -> None:
    class _FailingCoordinator(_RichCoordinator):
        def submit(self, jpeg: bytes, **metadata: Any) -> bool:
            del jpeg, metadata
            raise RuntimeError("worker unavailable")

    coordinator = _FailingCoordinator()
    runtime = _runtime(coordinator)
    source = _jpeg()
    try:
        runtime._on_display_frame(source, 1.0, 4)
        _, preview = runtime.preview_frames.latest() or (0, None)
        assert preview is not None
        assert preview.jpeg == source
    finally:
        runtime.stop()
