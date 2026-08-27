from __future__ import annotations

from concurrent.futures import Future
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from goodvision_harness.config import load_config
from goodvision_harness.contracts import NormalizedBBox
from goodvision_harness.grounding import (
    GroundedClassification,
    GroundingResponse,
    ModelObject,
)
from goodvision_harness.pipeline import SessionCallbacks, StreamingSession
from goodvision_harness.sliding_window import FrameWindow, SampledFrame
from goodvision_harness.visual_memory import NullVisualMemorySink, VisualMemoryRecord


ROOT = Path(__file__).resolve().parents[1]


class _Client:
    def __init__(self, settings: Any) -> None:
        self.settings = settings


def test_rich_grounding_selects_event_objects_and_preserves_attributes() -> None:
    config = load_config(ROOT / "config.yaml")
    emitted = []
    session = StreamingSession.for_camera(
        config,
        _Client(config.inference),  # type: ignore[arg-type]
        SessionCallbacks(on_analysis_result=emitted.append),
    )
    frame = SampledFrame(
        sample_index=0,
        source_frame_index=7,
        timestamp_seconds=1.25,
        jpeg_bytes=b"jpeg",
        width=320,
        height=180,
    )
    window = FrameWindow(index=0, frames=(frame,))
    objects = (
        ModelObject(
            label="known_face",
            bbox=NormalizedBBox(0.1, 0.1, 0.3, 0.4),
            attributes={"identity": "alice", "scores": [0.91]},
        ),
        ModelObject(
            label="unknown_face",
            bbox=NormalizedBBox(0.5, 0.1, 0.8, 0.5),
            attributes={"best_match": None, "is_unknown": True},
        ),
    )
    classification = GroundedClassification(
        detected=True,
        confidence=0.93,
        description="One unknown face is visible.",
        objects=objects,
        analysis_kind="face_recognition",
        event_object_indexes=(1,),
    )
    future: Future[Any] = Future()
    future.set_result(
        GroundingResponse(
            window=window,
            classification=classification,
            latency_seconds=0.05,
        )
    )

    session._on_window_done(future, window)

    assert len(emitted) == 1
    payload = emitted[0].as_dict()
    assert payload["analysis_kind"] == "face_recognition"
    assert payload["observation_id"] == payload["message_id"]
    assert payload["judgments"][0]["object_ids"] == ["object-2"]
    assert payload["objects"][0]["attributes"] == {
        "identity": "alice",
        "scores": [0.91],
    }
    assert payload["objects"][1]["attributes"]["is_unknown"] is True

    with pytest.raises(TypeError):
        objects[0].attributes["identity"] = "mallory"  # type: ignore[index]


def test_visual_memory_record_isolated_copy_and_null_sink() -> None:
    observation = {"observation_id": "message-1", "nested": {"values": [1]}}
    record = VisualMemoryRecord(observation=observation, evidence_jpeg=b"jpeg")
    observation["nested"]["values"].append(2)

    assert record.observation == {
        "observation_id": "message-1",
        "nested": {"values": [1]},
    }
    assert NullVisualMemorySink().submit(record) is True
    with pytest.raises(FrozenInstanceError):
        record.evidence_jpeg = None  # type: ignore[misc]


def test_event_object_indexes_are_validated() -> None:
    item = ModelObject(
        label="face",
        bbox=NormalizedBBox(0.1, 0.1, 0.9, 0.9),
    )
    with pytest.raises(ValueError, match="out-of-range"):
        GroundedClassification(
            detected=True,
            confidence=0.9,
            description="visible",
            objects=(item,),
            event_object_indexes=(1,),
        )
