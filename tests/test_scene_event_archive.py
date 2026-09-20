"""Scene inspection episodes retain independent, source-timed video evidence."""

from __future__ import annotations

import copy
import json
import threading
from pathlib import Path
from typing import Any

import cv2
import jsonschema
import numpy as np
import pytest

from goodvision_harness.event_archive import EventArchive, EventArchiveSettings

_SCHEMA = json.loads(
    (Path(__file__).resolve().parents[1] / "schemas/event-v1.schema.json").read_text(
        encoding="utf-8"
    )
)


def episode(
    event_id: str,
    label: str,
    start: float,
    last: float,
    *,
    subject: str = "person-1",
    phase: str = "confirmed",
) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "event_label": label,
        "subject_ids": [subject],
        "phase": phase,
        "first_evidence_timestamp_seconds": start,
        "last_evidence_timestamp_seconds": last,
        "frame_sequence": int(last * 5),
        "confidence": 0.95,
        "positive_observations": 2,
        "negative_observations": 0,
        "observation_state": "observed",
        "history": [
            {
                "timestamp_seconds": timestamp,
                "frame_sequence": int(timestamp * 5),
                "evidence_id": f"{event_id}:{index}",
                "evidence_ids": [f"{event_id}:{index}"],
                "revision": 1,
                "detected": True,
                "confidence": 0.95,
                "phase": "suspected" if index == 0 else "confirmed",
            }
            for index, timestamp in enumerate((start, last))
        ],
    }


def scene(*events: dict[str, Any], timestamp: float = 14) -> dict[str, Any]:
    return {
        "type": "scene_snapshot",
        "scene": {
            "schema_version": "1",
            "stream_session_id": "session",
            "source_id": "camera",
            "timestamp_seconds": timestamp,
            "events": list(events),
            "entities": [],
            "transitions": [],
        },
    }


def detection(timestamp: float, message: str, *, label: str = "stranger") -> dict:
    return {
        "message_id": message,
        "source_id": "camera",
        "stream_session_id": "session",
        "frame": {
            "timestamp_seconds": timestamp,
            "source_frame_index": int(timestamp * 5),
            "observed_at": "2026-09-16T00:00:00+00:00",
        },
        "judgments": [
            {
                "event_label": label,
                "detected": True,
                "object_ids": ["body"],
                "confidence": 0.9,
                "description": "Visible event",
            }
        ],
        "objects": [
            {
                "object_id": "body",
                "label": "person",
                "attributes": {"subject_id": "person-1", "subject_type": "person"},
                "bbox": {"x1": 0.1, "y1": 0.1, "x2": 0.4, "y2": 0.8},
            }
        ],
    }


class RecordingEncoder:
    def __init__(self) -> None:
        self.manifests: list[dict[str, Any]] = []

    def __call__(self, manifest: Path, destination: Path, **_: Any) -> dict[str, Any]:
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        assert metadata["frames"]
        for frame in metadata["frames"]:
            image = cv2.imdecode(
                np.frombuffer(
                    (manifest.parent / frame["filename"]).read_bytes(), np.uint8
                ),
                cv2.IMREAD_COLOR,
            )
            assert image is not None and image.shape == (90, 160, 3)
            assert (
                metadata["requested_start_seconds"]
                <= frame["timestamp_seconds"]
                < metadata["requested_end_seconds"]
            )
        self.manifests.append(metadata)
        destination.write_bytes(f"video-{metadata['revision']}".encode())
        return {"duration_seconds": 3.0, "fps": 5.0, "partial": False}


def frames(archive: EventArchive, *, end: float = 14) -> None:
    ok, encoded = cv2.imencode(".jpg", np.zeros((90, 160, 3), np.uint8))
    assert ok
    jpeg = encoded.tobytes()
    assert archive.start_session("session", "camera")
    for index in range(int(end * 5) + 1):
        assert archive.submit_frame(jpeg, index / 5, index, "session")
    assert archive.flush()


def rows(archive: EventArchive) -> list[dict[str, Any]]:
    values = archive.list_events()["events"]
    for value in values:
        jsonschema.Draft202012Validator(_SCHEMA).validate(value)
    return values


def finish(archive: EventArchive) -> list[dict[str, Any]]:
    assert archive.end_session("session")
    assert archive.wait_until_idle(3), archive.status()
    assert archive.status()["encoding_failures"] == 0
    return rows(archive)


def test_same_person_retains_smoking_and_fall_beside_one_stranger_clip(
    tmp_path: Path,
) -> None:
    encoder = RecordingEncoder()
    archive = EventArchive(
        EventArchiveSettings(directory=tmp_path), encoder=encoder, clock=lambda: 0.0
    )
    try:
        frames(archive)
        assert archive.submit_result(detection(1, "stranger-first"))
        assert archive.submit_result(detection(10, "stranger-returned"))
        assert archive.submit_scene(
            scene(
                episode("scene-stranger", "stranger", 1, 10),
                episode("scene-smoking", "smoking", 3, 4),
                episode("scene-fall", "fall", 7, 8),
            )
        )
        events = finish(archive)
        assert len(events) == 3
        by_label = {event["event_label"]: event for event in events}
        assert set(by_label) == {"stranger", "smoking", "fall"}
        assert {event["subject_id"] for event in events} == {"person-1"}
        assert by_label["smoking"]["start_seconds"] == 3
        assert by_label["fall"]["start_seconds"] == 7
        for label, expected_start in (("smoking", 2), ("fall", 6)):
            event = by_label[label]
            assert event["state"] == "ended"
            assert event["evidence"]["status"] == "ready"
            assert archive.video_path(event["event_id"]).is_file()
            manifest = next(
                item
                for item in encoder.manifests
                if item["event_id"] == event["event_id"]
            )
            assert manifest["requested_start_seconds"] == expected_start
            assert manifest["requested_end_seconds"] == expected_start + 3
            assert manifest["frames"][0]["timestamp_seconds"] == expected_start
        assert len(encoder.manifests) == 3
    finally:
        archive.close()


def test_distinct_scene_episode_ids_do_not_join_close_overlapping_same_label(
    tmp_path: Path,
) -> None:
    archive = EventArchive(
        EventArchiveSettings(directory=tmp_path),
        encoder=RecordingEncoder(),
        clock=lambda: 0.0,
    )
    try:
        frames(archive)
        assert archive.submit_scene(
            scene(
                episode("smoking-first", "smoking", 3, 4),
                episode("smoking-second", "smoking", 3.4, 4.4),
            )
        )
        events = finish(archive)
        assert len(events) == 2
        assert len({event["event_id"] for event in events}) == 2
        assert {event["start_seconds"] for event in events} == {3, 3.4}
        assert all(event["evidence"]["status"] == "ready" for event in events)
    finally:
        archive.close()


def test_identity_reconciliation_preserves_scene_episode_boundaries_and_videos(
    tmp_path: Path,
) -> None:
    archive = EventArchive(
        EventArchiveSettings(directory=tmp_path),
        encoder=RecordingEncoder(),
        clock=lambda: 0.0,
    )
    try:
        frames(archive)
        assert archive.submit_scene(
            scene(
                episode("first-track-event", "smoking", 3, 4),
                episode("second-track-event", "smoking", 3.4, 4.4, subject="person-2"),
            )
        )
        before = finish(archive)
        assert len(before) == 2
        previous = {event["event_id"]: event["evidence"] for event in before}
        assert archive.reconcile_subjects("session", {"person-2": "person-1"})
        assert archive.wait_until_idle(3)
        after = rows(archive)
        assert len(after) == 2
        assert {event["subject_id"] for event in after} == {"person-1"}
        assert {tuple(event["subject_ids"]) for event in after} == {("person-1",)}
        assert {event["event_id"]: event["evidence"] for event in after} == previous
        for event in after:
            assert archive.video_path(event["event_id"]).is_file()
    finally:
        archive.close()


def test_repeated_scene_snapshot_does_not_duplicate_events_or_encode_again(
    tmp_path: Path,
) -> None:
    encoder = RecordingEncoder()
    archive = EventArchive(
        EventArchiveSettings(directory=tmp_path), encoder=encoder, clock=lambda: 0.0
    )
    try:
        frames(archive)
        snapshot = scene(episode("stable-event", "smoking", 3, 4))
        assert archive.submit_scene(snapshot)
        assert archive.wait_until_idle(3)
        before = rows(archive)
        assert len(before) == 1
        for _ in range(5):
            assert archive.submit_scene(copy.deepcopy(snapshot))
        assert archive.wait_until_idle(3)
        after = rows(archive)
        assert after == before
        assert len(encoder.manifests) == 1
    finally:
        archive.close()


@pytest.mark.parametrize("transition_only", [False, True])
def test_scene_retraction_cannot_remain_a_playable_positive(
    tmp_path: Path, transition_only: bool
) -> None:
    archive = EventArchive(
        EventArchiveSettings(directory=tmp_path),
        encoder=RecordingEncoder(),
        clock=lambda: 0.0,
    )
    try:
        frames(archive)
        original = episode("corrected-event", "fall", 3, 4)
        assert archive.submit_scene(scene(original))
        assert archive.wait_until_idle(3)
        before = rows(archive)[0]
        assert before["evidence"]["status"] == "ready"
        corrected = copy.deepcopy(original)
        corrected["phase"] = "retracted"
        corrected["positive_observations"] = 0
        corrected["history"] = [
            {**item, "detected": False, "revision": 2, "phase": "retracted"}
            for item in corrected["history"]
        ]
        snapshot = scene() if transition_only else scene(corrected)
        snapshot["scene"]["transitions"] = [
            {
                "type": "retract",
                "event_id": original["event_id"],
                "event_label": "fall",
                "subject_ids": ["person-1"],
                "timestamp_seconds": 4,
                "frame_sequence": 20,
            }
        ]
        assert archive.submit_scene(snapshot)
        assert archive.wait_until_idle(3)
        after = rows(archive)
        assert len(after) == 1 and after[0]["event_id"] == before["event_id"]
        assert after[0]["state"] == "ended"
        assert after[0]["end_reason"] == "retracted"
        assert after[0]["evidence"]["status"] == "expired"
        assert after[0]["evidence"]["available"] is None
        # Revocation changes the current record, not an already published URL.
        assert archive.video_path(before["event_id"], 1).is_file()
    finally:
        archive.close()


def test_ended_scene_episodes_and_aliases_survive_restart_without_reencoding(
    tmp_path: Path,
) -> None:
    encoder = RecordingEncoder()
    settings = EventArchiveSettings(directory=tmp_path)
    snapshot = scene(
        episode("smoking-event", "smoking", 3, 4, subject="person-2"),
        episode("fall-event", "fall", 7, 8, subject="person-2"),
    )
    archive = EventArchive(settings, encoder=encoder, clock=lambda: 0.0)
    try:
        frames(archive)
        assert archive.submit_scene(snapshot)
        assert archive.reconcile_subjects("session", {"person-2": "person-1"})
        before = finish(archive)
        assert len(before) == 2
        assert all(event["finalized"] for event in before)
        assert all(event["state"] == "ended" for event in before)
    finally:
        archive.close()
    reopened = EventArchive(settings, encoder=encoder, clock=lambda: 0.0)
    try:
        assert reopened.flush()
        assert rows(reopened) == before
        assert reopened.submit_scene(snapshot)
        assert reopened.wait_until_idle(3)
        assert rows(reopened) == before
        assert len(encoder.manifests) == 2
        assert all(event["subject_id"] == "person-1" for event in rows(reopened))
        for event in rows(reopened):
            assert reopened.video_path(event["event_id"]).is_file()
    finally:
        reopened.close()


def test_encoding_completion_cannot_revive_retracted_scene_evidence(
    tmp_path: Path,
) -> None:
    entered, release = threading.Event(), threading.Event()
    encoder = RecordingEncoder()

    def delayed_encode(
        manifest: Path, destination: Path, **kwargs: Any
    ) -> dict[str, Any]:
        entered.set()
        assert release.wait(3)
        return encoder(manifest, destination, **kwargs)

    archive = EventArchive(
        EventArchiveSettings(directory=tmp_path),
        encoder=delayed_encode,
        clock=lambda: 0.0,
    )
    try:
        frames(archive)
        original = episode("withdrawn-while-encoding", "fall", 3, 4)
        assert archive.submit_scene(scene(original))
        assert entered.wait(3)
        corrected = copy.deepcopy(original)
        corrected["phase"] = "retracted"
        corrected["positive_observations"] = 0
        assert archive.submit_scene(scene(corrected))
        assert archive.flush()
        withdrawn = rows(archive)[0]
        assert withdrawn["evidence"]["status"] == "expired"
        assert not archive.wait_until_idle(0.01)
        release.set()
        assert archive.wait_until_idle(3)
        finished = rows(archive)[0]
        assert finished["end_reason"] == "retracted"
        assert finished["evidence"]["status"] == "expired"
        assert finished["evidence"]["available"] is None
    finally:
        release.set()
        archive.close()


def test_scene_revision_retains_explicit_zero_confidence(tmp_path: Path) -> None:
    archive = EventArchive(
        EventArchiveSettings(directory=tmp_path),
        encoder=RecordingEncoder(),
        clock=lambda: 0.0,
    )
    try:
        frames(archive)
        original = episode("revised-confidence", "smoking", 3, 4)
        assert archive.submit_scene(scene(original))
        assert archive.wait_until_idle(3)
        before = rows(archive)[0]
        assert before["confidence"] == 0.95
        corrected = copy.deepcopy(original)
        corrected["confidence"] = 0.0
        corrected["history"] = [
            {**item, "confidence": 0.0, "revision": 2} for item in corrected["history"]
        ]
        assert archive.submit_scene(scene(corrected))
        assert archive.wait_until_idle(3)
        after = rows(archive)[0]
        assert after["event_id"] == before["event_id"]
        assert after["confidence"] == 0.0
        assert after["evidence"] == before["evidence"]
    finally:
        archive.close()


@pytest.mark.parametrize("latest_phase", ["confirmed", "ended"])
def test_latest_historical_transition_overrides_older_retraction(
    tmp_path: Path, latest_phase: str
) -> None:
    encoder = RecordingEncoder()
    archive = EventArchive(
        EventArchiveSettings(directory=tmp_path), encoder=encoder, clock=lambda: 0.0
    )
    try:
        frames(archive)
        original = episode("historical-event", "smoking", 3, 4)
        assert archive.submit_scene(scene(original))
        assert archive.wait_until_idle(3)
        before = rows(archive)[0]
        snapshot = scene()
        snapshot["scene"]["transitions"] = [
            {
                "type": "retract",
                "event_id": original["event_id"],
                "event_label": "smoking",
                "subject_ids": ["person-1"],
                "timestamp_seconds": 4,
                "frame_sequence": 20,
            },
            {
                "type": "update",
                "event_id": original["event_id"],
                "event_label": "smoking",
                "subject_ids": ["person-1"],
                "timestamp_seconds": 6,
                "frame_sequence": 30,
                "phase": latest_phase,
            },
        ]
        assert archive.submit_scene(snapshot)
        assert archive.wait_until_idle(3)
        after = rows(archive)
        assert len(after) == 1
        event = after[0]
        assert event["event_id"] == before["event_id"]
        assert event["start_seconds"] == 3
        assert event["last_detected_seconds"] == 4
        assert event["evidence"] == before["evidence"]
        assert event["evidence"]["status"] == "ready"
        assert len(encoder.manifests) == 1
        if latest_phase == "ended":
            assert event["state"] == "ended"
            assert event["end_reason"] == "negative"
            assert event["end_seconds"] == 6
        else:
            assert event == before
            assert event["state"] == "active"
            assert event["end_reason"] is None
    finally:
        archive.close()


def test_scene_review_uses_original_source_time_after_detector_watermark(
    tmp_path: Path,
) -> None:
    encoder = RecordingEncoder()
    archive = EventArchive(
        EventArchiveSettings(directory=tmp_path), encoder=encoder, clock=lambda: 0.0
    )
    try:
        frames(archive, end=25)
        assert archive.note_window("session", "new-detector-window", 20, ("smoking",))
        assert archive.submit_result(detection(3, "late-detector", label="smoking"))
        assert archive.flush()
        assert rows(archive) == []
        assert archive.status()["late_results"] == 1
        assert archive.submit_scene(
            scene(episode("reviewed-source-event", "smoking", 3, 4), timestamp=25)
        )
        events = finish(archive)
        assert len(events) == 1
        assert events[0]["start_seconds"] == 3
        assert events[0]["last_detected_seconds"] == 4
        assert events[0]["evidence"]["status"] == "ready"
        assert encoder.manifests[0]["requested_start_seconds"] == 2
        assert encoder.manifests[0]["requested_end_seconds"] == 5
        assert archive.status()["late_results"] == 1
    finally:
        archive.close()
