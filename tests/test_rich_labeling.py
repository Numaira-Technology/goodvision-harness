from __future__ import annotations

import json
import threading
import time
from collections.abc import Mapping
from typing import Any

import httpx
import pytest

from goodvision_harness.contracts import BBOX_COORDINATE_SYSTEM, NormalizedBBox
from goodvision_harness.privacy import contains_private_implementation_detail
from goodvision_harness.rich_labeling import (
    FRAME_SEQUENCE_HEADER,
    STREAM_SESSION_HEADER,
    TIMESTAMP_HEADER,
    RichLabelCoordinator,
    RichLabelHTTPClient,
    RichLabelObject,
    RichLabelProtocolError,
    RichLabelSnapshot,
    select_rich_labels,
)


def _object(
    track_id: str = "person-1",
    bbox: tuple[float, float, float, float] = (0.1, 0.2, 0.3, 0.6),
    *,
    confidence: float | None = 0.9,
    label: str = "person",
    attributes: Mapping[str, Any] | None = None,
) -> RichLabelObject:
    return RichLabelObject(
        track_id=track_id,
        label=label,
        bbox=NormalizedBBox(*bbox),
        confidence=confidence,
        attributes={"gender": "female", "age": 31}
        if attributes is None
        else attributes,
    )


def _snapshot(
    sequence: int,
    timestamp: float,
    *objects: RichLabelObject,
    session_id: str = "session-1",
) -> RichLabelSnapshot:
    return RichLabelSnapshot(
        stream_session_id=session_id,
        frame_sequence=sequence,
        timestamp_seconds=timestamp,
        objects=tuple(objects),
    )


def _response_payload(
    *,
    session_id: str = "session-1",
    sequence: int = 7,
    timestamp: float = 1.25,
) -> dict[str, Any]:
    return _snapshot(sequence, timestamp, _object(), session_id=session_id).as_dict()


def test_rich_label_records_validate_freeze_and_serialize() -> None:
    attributes = {"profile": {"age": 31}, "tags": ["staff"]}
    item = _object(attributes=attributes)
    attributes["profile"]["age"] = 99
    attributes["tags"].append("changed")
    snapshot = _snapshot(7, 1.25, item)

    assert item.attributes["profile"]["age"] == 31
    assert item.attributes["tags"] == ("staff",)
    assert snapshot.as_dict() == {
        "stream_session_id": "session-1",
        "frame_sequence": 7,
        "timestamp_seconds": 1.25,
        "objects": [
            {
                "track_id": "person-1",
                "label": "person",
                "bbox": {
                    "coordinate_system": BBOX_COORDINATE_SYSTEM,
                    "x1": 0.1,
                    "y1": 0.2,
                    "x2": 0.3,
                    "y2": 0.6,
                },
                "confidence": 0.9,
                "attributes": {"profile": {"age": 31}, "tags": ["staff"]},
            }
        ],
    }


@pytest.mark.parametrize(
    ("factory", "error"),
    [
        (lambda: _object(track_id=""), ValueError),
        (lambda: _object(confidence=1.1), ValueError),
        (
            lambda: RichLabelObject(
                "track",
                "person",
                "not-a-box",  # type: ignore[arg-type]
            ),
            TypeError,
        ),
        (
            lambda: _snapshot(0, 0.0, _object("duplicate"), _object("duplicate")),
            ValueError,
        ),
        (lambda: _snapshot(-1, 0.0), ValueError),
        (lambda: _snapshot(0, float("nan")), ValueError),
    ],
)
def test_rich_label_records_reject_invalid_values(
    factory: Any,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        factory()


def test_http_client_posts_raw_jpeg_metadata_and_parses_response() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=_response_payload())

    client = RichLabelHTTPClient(
        "http://detector.local/v1/rich-labels",
        headers={"Authorization": "Bearer private-token"},
        transport=httpx.MockTransport(handler),
    )
    try:
        result = client.infer(
            b"raw-jpeg",
            stream_session_id="session-1",
            frame_sequence=7,
            timestamp_seconds=1.25,
        )
    finally:
        client.close()

    assert result == _snapshot(7, 1.25, _object())
    assert len(requests) == 1
    request = requests[0]
    assert request.method == "POST"
    assert request.content == b"raw-jpeg"
    assert request.headers["content-type"] == "image/jpeg"
    assert request.headers["accept"] == "application/json"
    assert request.headers[STREAM_SESSION_HEADER] == "session-1"
    assert request.headers[FRAME_SEQUENCE_HEADER] == "7"
    assert request.headers[TIMESTAMP_HEADER] == "1.25"
    assert request.headers["authorization"] == "Bearer private-token"


def test_http_client_round_trips_long_stream_timestamps() -> None:
    timestamp = 12_345.678901234567
    seen_header: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        rendered = request.headers[TIMESTAMP_HEADER]
        seen_header.append(rendered)
        return httpx.Response(
            200,
            json=_response_payload(timestamp=float(rendered)),
        )

    client = RichLabelHTTPClient(
        "http://detector.local/v1/rich-labels",
        transport=httpx.MockTransport(handler),
    )
    try:
        result = client.infer(
            b"raw-jpeg",
            stream_session_id="session-1",
            frame_sequence=7,
            timestamp_seconds=timestamp,
        )
    finally:
        client.close()

    assert seen_header == [format(timestamp, ".17g")]
    assert result.timestamp_seconds == timestamp


@pytest.mark.parametrize(
    "mutate",
    [
        lambda payload: payload.update({"unexpected": True}),
        lambda payload: payload.pop("objects"),
        lambda payload: payload.update({"frame_sequence": True}),
        lambda payload: payload.update({"stream_session_id": "other-session"}),
        lambda payload: payload.update({"timestamp_seconds": 1.5}),
        lambda payload: payload["objects"][0].update({"confidence": True}),
        lambda payload: payload["objects"][0]["bbox"].update({"x2": 2.0}),
        lambda payload: payload["objects"].append(payload["objects"][0].copy()),
    ],
)
def test_http_client_rejects_invalid_or_mismatched_json(mutate: Any) -> None:
    payload = _response_payload()
    mutate(payload)

    client = RichLabelHTTPClient(
        "http://detector.local/infer",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, json=payload)
        ),
    )
    try:
        with pytest.raises(RichLabelProtocolError):
            client.infer(
                b"jpeg",
                stream_session_id="session-1",
                frame_sequence=7,
                timestamp_seconds=1.25,
            )
    finally:
        client.close()


@pytest.mark.parametrize(
    ("content", "content_type"),
    [
        (b"not-json", "application/json"),
        (
            b'{"stream_session_id":"session-1","stream_session_id":"again"}',
            "application/json",
        ),
        (b'{"timestamp_seconds":NaN}', "application/json"),
        (json.dumps(_response_payload()).encode(), "text/plain"),
    ],
)
def test_http_client_rejects_non_strict_json_or_wrong_content_type(
    content: bytes,
    content_type: str,
) -> None:
    client = RichLabelHTTPClient(
        "http://detector.local/infer",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(
                200,
                content=content,
                headers={"Content-Type": content_type},
            )
        ),
    )
    try:
        with pytest.raises(RichLabelProtocolError):
            client.infer(
                b"jpeg",
                stream_session_id="session-1",
                frame_sequence=7,
                timestamp_seconds=1.25,
            )
    finally:
        client.close()


def test_http_client_propagates_http_failure_without_parsing_body() -> None:
    client = RichLabelHTTPClient(
        "http://detector.local/infer",
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(503, text="unavailable")
        ),
    )
    try:
        with pytest.raises(httpx.HTTPStatusError):
            client.infer(
                b"jpeg",
                stream_session_id="session-1",
                frame_sequence=7,
                timestamp_seconds=1.25,
            )
    finally:
        client.close()


def test_selection_interpolates_only_matching_stable_track_ids() -> None:
    left = _snapshot(
        0,
        0.0,
        _object("person-1", (0.1, 0.1, 0.3, 0.5), confidence=0.8),
        _object("left-only", (0.5, 0.1, 0.7, 0.5)),
    )
    right = _snapshot(
        10,
        1.0,
        _object("person-1", (0.3, 0.2, 0.5, 0.6), confidence=1.0),
        _object("new-on-right", (0.7, 0.1, 0.9, 0.5)),
    )

    selection = select_rich_labels(
        (right, left),
        0.5,
        max_gap_seconds=1.0,
        frame_sequence=5,
    )

    assert selection.mode == "interpolated"
    assert selection.left_frame_sequence == 0
    assert selection.right_frame_sequence == 10
    assert selection.snapshot is not None
    assert selection.snapshot.frame_sequence == 5
    assert selection.snapshot.timestamp_seconds == 0.5
    assert [item.track_id for item in selection.objects] == ["person-1", "left-only"]
    moving = selection.objects[0]
    assert (moving.bbox.x1, moving.bbox.y1, moving.bbox.x2, moving.bbox.y2) == (
        pytest.approx(0.2),
        pytest.approx(0.15),
        pytest.approx(0.4),
        pytest.approx(0.55),
    )
    assert moving.confidence == pytest.approx(0.9)
    assert selection.objects[1] is left.objects[1]


def test_selection_handles_exact_held_long_gap_and_pre_detection() -> None:
    left = _snapshot(2, 1.0, _object())
    right = _snapshot(8, 3.0, _object(bbox=(0.4, 0.2, 0.6, 0.6)))

    before = select_rich_labels((left, right), 0.5, max_gap_seconds=1.0)
    exact = select_rich_labels((left, right), 1.0, max_gap_seconds=1.0)
    long_gap = select_rich_labels((left, right), 2.5, max_gap_seconds=1.0)
    updated = select_rich_labels((left, right), 3.0, max_gap_seconds=1.0)
    after = select_rich_labels((left, right), 5.0, max_gap_seconds=1.0)

    assert before.mode == "none" and before.objects == ()
    assert exact.mode == "exact" and exact.snapshot is left
    assert long_gap.mode == "held" and long_gap.snapshot is left
    assert updated.mode == "exact" and updated.snapshot is right
    assert after.mode == "none" and after.objects == ()


def test_display_selection_holds_snapshot_until_next_keyframe() -> None:
    left = _snapshot(0, 0.0, _object(bbox=(0.1, 0.1, 0.3, 0.5)))
    right = _snapshot(12, 0.4, _object(bbox=(0.3, 0.2, 0.5, 0.6)))
    cleared = _snapshot(24, 0.8)
    history = (cleared, right, left)

    for sequence in range(30):
        selection = select_rich_labels(
            history, sequence / 30, max_gap_seconds=1.0,
            frame_sequence=sequence, interpolate=False,
        )
        expected = left if sequence < 12 else right if sequence < 24 else cleared
        assert selection.snapshot is expected
        assert selection.objects == expected.objects
        assert selection.right_frame_sequence is None
        assert selection.mode == ("exact" if sequence % 12 == 0 else "held")

    expired = select_rich_labels(
        (left,), 1.001, max_gap_seconds=1.0, interpolate=False,
    )
    assert expired.mode == "none"


def test_coordinator_display_hold_keeps_analysis_interpolation() -> None:
    left = _snapshot(0, 0.0, _object(bbox=(0.1, 0.1, 0.3, 0.5)))
    right = _snapshot(12, 0.4, _object(bbox=(0.3, 0.2, 0.5, 0.6)))
    snapshots = {0: left, 12: right}

    class Client:
        def infer(self, _jpeg, *, frame_sequence, **_metadata):
            return snapshots[frame_sequence]

    coordinator = RichLabelCoordinator(Client(), max_fps=10_000)
    try:
        for snapshot in (left, right):
            _complete_rich_sample(
                coordinator, sequence=snapshot.frame_sequence,
                timestamp=snapshot.timestamp_seconds,
            )
        display = coordinator.select("session-1", 0.2, interpolate=False)
        analysis = coordinator.select("session-1", 0.2)
        updated = coordinator.select("session-1", 0.4, interpolate=False)
        assert display.mode == "held" and display.snapshot is left
        assert analysis.mode == "interpolated"
        assert analysis.objects[0].bbox.x1 == pytest.approx(0.2)
        assert updated.mode == "exact" and updated.snapshot is right
    finally:
        coordinator.close()


def test_selection_expires_frozen_boxes_after_worker_outage() -> None:
    latest = _snapshot(20, 10.0, _object())

    briefly_held = select_rich_labels(
        (latest,),
        10.75,
        max_gap_seconds=0.25,
        max_hold_seconds=0.75,
    )
    expired = select_rich_labels(
        (latest,),
        10.751,
        max_gap_seconds=0.25,
        max_hold_seconds=0.75,
    )

    assert briefly_held.mode == "held"
    assert expired.mode == "none"
    assert expired.objects == ()


def test_selection_holds_long_gap_until_empty_keyframe_clears() -> None:
    detected = _snapshot(2, 1.0, _object())
    cleared = _snapshot(12, 3.0)

    held = select_rich_labels(
        (detected, cleared),
        2.999,
        max_gap_seconds=0.75,
        max_hold_seconds=0.25,
    )
    exact_clear = select_rich_labels(
        (detected, cleared),
        3.0,
        max_gap_seconds=0.75,
        max_hold_seconds=0.25,
    )

    assert held.mode == "held"
    assert held.snapshot is detected
    assert held.objects
    assert exact_clear.mode == "exact"
    assert exact_clear.snapshot is cleared
    assert exact_clear.objects == ()


class _BlockingClient:
    def __init__(self) -> None:
        self.calls: list[tuple[bytes, str, int, float]] = []
        self.active = 0
        self.max_active = 0
        self.first_started = threading.Event()
        self.release_first = threading.Event()
        self.closed = 0
        self._lock = threading.Lock()

    def infer(
        self,
        jpeg_bytes: bytes,
        *,
        stream_session_id: str,
        frame_sequence: int,
        timestamp_seconds: float,
    ) -> RichLabelSnapshot:
        with self._lock:
            self.calls.append(
                (jpeg_bytes, stream_session_id, frame_sequence, timestamp_seconds)
            )
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            call_number = len(self.calls)
        if call_number == 1:
            self.first_started.set()
            assert self.release_first.wait(1.0)
        with self._lock:
            self.active -= 1
        return _snapshot(
            frame_sequence,
            timestamp_seconds,
            _object(f"track-{frame_sequence}"),
            session_id=stream_session_id,
        )

    def close(self) -> None:
        self.closed += 1


def test_coordinator_is_single_worker_and_keeps_only_latest_waiting_frame() -> None:
    client = _BlockingClient()
    coordinator = RichLabelCoordinator(client, max_fps=10_000, history_size=4)
    try:
        assert coordinator.submit(
            b"one",
            stream_session_id="session-1",
            frame_sequence=1,
            timestamp_seconds=0.1,
        )
        assert client.first_started.wait(1.0)
        assert coordinator.submit(
            b"two",
            stream_session_id="session-1",
            frame_sequence=2,
            timestamp_seconds=0.2,
        )
        assert coordinator.submit(
            b"three",
            stream_session_id="session-1",
            frame_sequence=3,
            timestamp_seconds=0.3,
        )
        client.release_first.set()
        assert coordinator.wait_until_idle()

        assert [call[2] for call in client.calls] == [1, 3]
        assert client.max_active == 1
        assert coordinator.stats()["replaced"] == 1
        assert coordinator.stats()["succeeded"] == 2
        assert [item.frame_sequence for item in coordinator.history()] == [1, 3]
    finally:
        client.release_first.set()
        coordinator.close()
    assert client.closed == 1


class _ControlledClock:
    def __init__(self) -> None:
        self._now = 100.0
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self._now

    def advance(self, seconds: float) -> None:
        with self._lock:
            self._now += seconds


class _TimingClient:
    def __init__(self, call_count: int) -> None:
        self.calls: list[int] = []
        self.started = [threading.Event() for _ in range(call_count)]
        self.release = [threading.Event() for _ in range(call_count)]

    def infer(
        self,
        _jpeg_bytes: bytes,
        *,
        stream_session_id: str,
        frame_sequence: int,
        timestamp_seconds: float,
    ) -> RichLabelSnapshot:
        index = len(self.calls)
        self.calls.append(frame_sequence)
        self.started[index].set()
        assert self.release[index].wait(2.0)
        return _snapshot(
            frame_sequence, timestamp_seconds, session_id=stream_session_id,
        )


class _CadenceClient:
    def __init__(self, clock: _ControlledClock, *, duration: float) -> None:
        self.clock = clock
        self.duration = duration
        self.fail = False
        self.visible = True

    def infer(
        self,
        _jpeg_bytes: bytes,
        *,
        stream_session_id: str,
        frame_sequence: int,
        timestamp_seconds: float,
    ) -> RichLabelSnapshot:
        self.clock.advance(self.duration)
        if self.fail:
            raise httpx.ReadTimeout("unavailable")
        return _snapshot(
            frame_sequence,
            timestamp_seconds,
            *((_object(),) if self.visible else ()),
            session_id=stream_session_id,
        )


def _complete_rich_sample(
    coordinator: RichLabelCoordinator,
    *,
    sequence: int,
    timestamp: float,
    session_id: str = "session-1",
) -> None:
    assert coordinator.submit(
        b"frame", stream_session_id=session_id,
        frame_sequence=sequence, timestamp_seconds=timestamp,
    )
    assert coordinator.wait_until_idle()


def test_coordinator_holds_slow_detector_tail_without_changing_evidence() -> None:
    clock = _ControlledClock()
    client = _CadenceClient(clock, duration=4.0)
    coordinator = RichLabelCoordinator(
        client, clock=clock, max_fps=10_000, max_hold_seconds=2.0,
    )
    try:
        _complete_rich_sample(coordinator, sequence=1, timestamp=0.0)
        evidence = coordinator.latest()
        assert evidence is not None
        original = evidence.as_dict()

        # Two 4 s inferences with 2.5 s display delay leave the old source box
        # 5.5 s old just before the next result can arrive.
        for timestamp in (2.001, 4.0, 5.5, 10.0):
            display = coordinator.select("session-1", timestamp)
            assert display.mode == "held"
            assert display.objects == evidence.objects
        assert coordinator.select("session-1", 10.001).objects == ()
        assert coordinator.history() == (evidence,)
        assert coordinator.latest() is evidence
        assert evidence.as_dict() == original
        assert evidence.timestamp_seconds == 0.0
    finally:
        coordinator.close()


def test_coordinator_uses_successful_completion_gaps_for_display_continuity() -> None:
    clock = _ControlledClock()
    client = _CadenceClient(clock, duration=0.25)
    coordinator = RichLabelCoordinator(
        client, clock=clock, max_fps=10_000, max_hold_seconds=2.0,
    )
    try:
        _complete_rich_sample(coordinator, sequence=1, timestamp=0.0)
        clock.advance(4.0)
        _complete_rich_sample(coordinator, sequence=2, timestamp=4.25)
        latest = coordinator.latest()
        assert latest is not None

        # Computation is fast, but successive successful completions were
        # 4.25 s apart. The display must cover that actual update cadence.
        assert coordinator.select("session-1", 4.25 + 8.0).objects == latest.objects
        assert coordinator.select("session-1", 4.25 + 10.5).objects == latest.objects
        assert coordinator.select("session-1", 4.25 + 10.501).objects == ()
    finally:
        coordinator.close()


def test_coordinator_failures_do_not_extend_or_refresh_display_hold() -> None:
    clock = _ControlledClock()
    client = _CadenceClient(clock, duration=2.0)
    coordinator = RichLabelCoordinator(
        client, clock=clock, max_fps=10_000, max_hold_seconds=2.0,
    )
    try:
        _complete_rich_sample(coordinator, sequence=1, timestamp=0.0)
        evidence = coordinator.latest()
        assert evidence is not None
        assert coordinator.select("session-1", 5.9).objects == evidence.objects
        client.fail = True
        client.duration = 10.0

        for sequence in range(2, 5):
            _complete_rich_sample(
                coordinator, sequence=sequence, timestamp=float(sequence * 3),
            )
            assert coordinator.latest() is evidence
            assert coordinator.select("session-1", 6.001).objects == ()
            assert coordinator.history() == (evidence,)
        assert coordinator.stats()["failed"] == 3
    finally:
        coordinator.close()


def test_coordinator_adaptive_display_hold_has_fifteen_second_cap() -> None:
    clock = _ControlledClock()
    client = _CadenceClient(clock, duration=20.0)
    coordinator = RichLabelCoordinator(
        client, clock=clock, max_fps=10_000, max_hold_seconds=2.0,
    )
    try:
        _complete_rich_sample(coordinator, sequence=1, timestamp=0.0)

        assert coordinator.select("session-1", 15.0).objects
        assert coordinator.select("session-1", 15.001).objects == ()
        assert coordinator.select("session-1", 50.0).objects == ()
    finally:
        coordinator.close()


def test_coordinator_empty_success_clears_held_rich_labels_immediately() -> None:
    clock = _ControlledClock()
    client = _CadenceClient(clock, duration=4.0)
    coordinator = RichLabelCoordinator(
        client, clock=clock, max_fps=10_000, max_hold_seconds=2.0,
    )
    try:
        _complete_rich_sample(coordinator, sequence=1, timestamp=0.0)
        assert coordinator.select("session-1", 5.5).objects
        client.visible = False
        _complete_rich_sample(coordinator, sequence=2, timestamp=6.0)

        assert coordinator.select("session-1", 5.999).objects
        assert coordinator.select("session-1", 6.0).objects == ()
        assert coordinator.select("session-1", 7.0).objects == ()
        assert coordinator.latest() is not None
        assert coordinator.latest().objects == ()
    finally:
        coordinator.close()


def test_coordinator_reset_discards_previous_session_adaptive_display_hold() -> None:
    clock = _ControlledClock()
    client = _CadenceClient(clock, duration=20.0)
    coordinator = RichLabelCoordinator(
        client, clock=clock, max_fps=10_000, max_hold_seconds=2.0,
    )
    try:
        _complete_rich_sample(coordinator, sequence=1, timestamp=0.0)
        assert coordinator.select("session-1", 15.0).objects
        coordinator.reset("new-session")
        assert coordinator.history() == ()
        assert coordinator.select("session-1", 1.0).objects == ()
        assert coordinator.select("new-session", 1.0).objects == ()
        client.duration = 0.1
        _complete_rich_sample(
            coordinator, sequence=1, timestamp=0.0, session_id="new-session",
        )

        assert coordinator.select("new-session", 2.2).objects
        assert coordinator.select("new-session", 2.201).objects == ()
        assert coordinator.select("session-1", 2.0).objects == ()
    finally:
        coordinator.close()


def test_coordinator_display_hold_recovers_after_recent_fast_completions() -> None:
    clock = _ControlledClock()
    client = _CadenceClient(clock, duration=20.0)
    coordinator = RichLabelCoordinator(
        client, clock=clock, max_fps=10_000, max_hold_seconds=2.0,
    )
    try:
        _complete_rich_sample(coordinator, sequence=1, timestamp=0.0)
        assert coordinator.select("session-1", 15.0).objects
        client.duration = 0.1
        for sequence in range(2, 8):
            _complete_rich_sample(
                coordinator, sequence=sequence, timestamp=float(sequence),
            )

        assert coordinator.select("session-1", 7.0 + 2.2).objects
        assert coordinator.select("session-1", 7.0 + 2.201).objects == ()
    finally:
        coordinator.close()


def test_coordinator_stale_completion_cannot_expand_new_session_display_hold() -> None:
    class Client(_TimingClient):
        def infer(self, *args: Any, **kwargs: Any) -> RichLabelSnapshot:
            snapshot = super().infer(*args, **kwargs)
            return _snapshot(
                snapshot.frame_sequence, snapshot.timestamp_seconds, _object(),
                session_id=snapshot.stream_session_id,
            )

    clock = _ControlledClock()
    client = Client(2)
    coordinator = RichLabelCoordinator(
        client, clock=clock, max_fps=10_000, max_hold_seconds=2.0,
    )
    try:
        assert coordinator.submit(
            b"old", stream_session_id="old-session",
            frame_sequence=1, timestamp_seconds=0.0,
        )
        assert client.started[0].wait(1.0)
        coordinator.reset("new-session")
        clock.advance(20.0)
        client.release[0].set()
        assert coordinator.wait_until_idle()
        assert coordinator.history() == ()
        assert coordinator.stats()["stale_results"] == 1

        assert coordinator.submit(
            b"new", stream_session_id="new-session",
            frame_sequence=1, timestamp_seconds=0.0,
        )
        assert client.started[1].wait(1.0)
        clock.advance(0.1)
        client.release[1].set()
        assert coordinator.wait_until_idle()

        assert coordinator.select("new-session", 2.2).objects
        assert coordinator.select("new-session", 2.201).objects == ()
    finally:
        for release in client.release:
            release.set()
        coordinator.close()


def test_coordinator_times_local_queue_from_latest_replacement() -> None:
    clock = _ControlledClock()
    client = _TimingClient(2)
    coordinator = RichLabelCoordinator(client, max_fps=10_000, clock=clock)
    try:
        assert coordinator.submit(
            b"first", stream_session_id="session-1",
            frame_sequence=1, timestamp_seconds=0.1,
        )
        assert client.started[0].wait(1.0)
        clock.advance(1.0)
        assert coordinator.submit(
            b"replaced", stream_session_id="session-1",
            frame_sequence=2, timestamp_seconds=0.2,
        )
        clock.advance(2.0)
        assert coordinator.submit(
            b"latest", stream_session_id="session-1",
            frame_sequence=3, timestamp_seconds=0.3,
        )
        clock.advance(3.0)
        client.release[0].set()
        assert client.started[1].wait(1.0)

        first_stats = coordinator.stats()
        assert first_stats["last_queue_seconds"] == 0.0
        assert first_stats["last_latency_seconds"] == 6.0
        assert first_stats["last_total_latency_seconds"] == 6.0

        clock.advance(4.0)
        client.release[1].set()
        assert coordinator.wait_until_idle()
        stats = coordinator.stats()
        assert client.calls == [1, 3]
        assert stats["replaced"] == 1
        assert stats["last_frame_sequence"] == 3
        assert stats["last_queue_seconds"] == 3.0
        assert stats["last_latency_seconds"] == 4.0
        assert stats["last_total_latency_seconds"] == 7.0
    finally:
        for release in client.release:
            release.set()
        coordinator.close()


def test_coordinator_reset_clears_timing_and_ignores_old_completion() -> None:
    clock = _ControlledClock()
    client = _TimingClient(3)
    coordinator = RichLabelCoordinator(client, max_fps=10_000, clock=clock)
    timing_fields = (
        "last_queue_seconds", "last_latency_seconds", "last_total_latency_seconds",
        "last_attempt_frame_sequence", "last_attempt_succeeded",
    )
    try:
        assert coordinator.submit(
            b"completed", stream_session_id="old-session",
            frame_sequence=1, timestamp_seconds=0.1,
        )
        assert client.started[0].wait(1.0)
        clock.advance(2.0)
        client.release[0].set()
        assert coordinator.wait_until_idle()
        assert coordinator.stats()["last_total_latency_seconds"] == 2.0

        clock.advance(2.0)
        assert coordinator.submit(
            b"old", stream_session_id="old-session",
            frame_sequence=2, timestamp_seconds=0.2,
        )
        assert client.started[1].wait(1.0)
        clock.advance(1.0)
        coordinator.reset("new-session")
        assert all(coordinator.stats()[field] is None for field in timing_fields)

        clock.advance(1.0)
        assert coordinator.submit(
            b"new", stream_session_id="new-session",
            frame_sequence=3, timestamp_seconds=0.3,
        )
        clock.advance(4.0)
        client.release[1].set()
        assert client.started[2].wait(1.0)
        stats = coordinator.stats()
        assert stats["stale_results"] == 1
        assert all(stats[field] is None for field in timing_fields)

        clock.advance(3.0)
        client.release[2].set()
        assert coordinator.wait_until_idle()
        stats = coordinator.stats()
        assert stats["stream_session_id"] == "new-session"
        assert stats["last_frame_sequence"] == 3
        assert stats["last_queue_seconds"] == 4.0
        assert stats["last_latency_seconds"] == 3.0
        assert stats["last_total_latency_seconds"] == 7.0
        assert stats["last_attempt_frame_sequence"] == 3
        assert stats["last_attempt_succeeded"] is True
    finally:
        for release in client.release:
            release.set()
        coordinator.close()


def test_coordinator_failed_attempt_timing_keeps_last_successful_frame() -> None:
    class Client(_TimingClient):
        def infer(self, *args: Any, **kwargs: Any) -> RichLabelSnapshot:
            snapshot = super().infer(*args, **kwargs)
            if snapshot.frame_sequence == 2:
                raise httpx.ConnectError("offline")
            return snapshot

    clock = _ControlledClock()
    client = Client(2)
    coordinator = RichLabelCoordinator(client, max_fps=10_000, clock=clock)
    try:
        initial_stats = coordinator.stats()
        assert initial_stats["last_attempt_frame_sequence"] is None
        assert initial_stats["last_attempt_succeeded"] is None
        assert coordinator.submit(
            b"good", stream_session_id="session-1",
            frame_sequence=1, timestamp_seconds=0.1,
        )
        assert client.started[0].wait(1.0)
        clock.advance(1.0)
        assert coordinator.submit(
            b"bad", stream_session_id="session-1",
            frame_sequence=2, timestamp_seconds=0.2,
        )
        clock.advance(2.0)
        client.release[0].set()
        assert client.started[1].wait(1.0)
        good_stats = coordinator.stats()
        assert good_stats["last_attempt_frame_sequence"] == 1
        assert good_stats["last_attempt_succeeded"] is True
        assert good_stats["last_total_latency_seconds"] == 3.0

        clock.advance(4.0)
        client.release[1].set()
        assert coordinator.wait_until_idle()
        stats = coordinator.stats()
        assert stats["last_attempt_frame_sequence"] == 2
        assert stats["last_attempt_succeeded"] is False
        assert stats["last_queue_seconds"] == 2.0
        assert stats["last_latency_seconds"] == 4.0
        assert stats["last_total_latency_seconds"] == 6.0
        assert stats["last_frame_sequence"] == 1
        assert stats["succeeded"] == stats["failed"] == 1
        assert coordinator.latest().frame_sequence == 1
    finally:
        for release in client.release:
            release.set()
        coordinator.close()


def test_coordinator_rate_limits_request_start_times() -> None:
    started: list[float] = []

    class Client:
        def infer(
            self,
            _jpeg_bytes: bytes,
            *,
            stream_session_id: str,
            frame_sequence: int,
            timestamp_seconds: float,
        ) -> RichLabelSnapshot:
            started.append(time.monotonic())
            return _snapshot(
                frame_sequence,
                timestamp_seconds,
                session_id=stream_session_id,
            )

    coordinator = RichLabelCoordinator(Client(), max_fps=25.0)
    try:
        assert coordinator.submit(
            b"one",
            stream_session_id="session-1",
            frame_sequence=1,
            timestamp_seconds=0.1,
        )
        deadline = time.monotonic() + 1.0
        while len(started) < 1 and time.monotonic() < deadline:
            time.sleep(0.001)
        assert started
        assert coordinator.submit(
            b"two",
            stream_session_id="session-1",
            frame_sequence=2,
            timestamp_seconds=0.2,
        )
        assert coordinator.wait_until_idle()

        assert len(started) == 2
        assert started[1] - started[0] >= 0.035
    finally:
        coordinator.close()


def test_coordinator_reset_isolates_in_flight_old_session() -> None:
    client = _BlockingClient()
    coordinator = RichLabelCoordinator(client, max_fps=10_000)
    try:
        assert coordinator.submit(
            b"old",
            stream_session_id="old-session",
            frame_sequence=1,
            timestamp_seconds=0.1,
        )
        assert client.first_started.wait(1.0)
        coordinator.reset("new-session")
        assert not coordinator.submit(
            b"wrong",
            stream_session_id="old-session",
            frame_sequence=2,
            timestamp_seconds=0.2,
        )
        assert coordinator.submit(
            b"new",
            stream_session_id="new-session",
            frame_sequence=3,
            timestamp_seconds=0.3,
        )
        client.release_first.set()
        assert coordinator.wait_until_idle()

        assert [item.stream_session_id for item in coordinator.history()] == [
            "new-session"
        ]
        assert coordinator.latest("old-session") is None
        assert coordinator.select("old-session", 0.3).mode == "none"
        assert coordinator.stats()["stale_results"] == 1
        assert coordinator.stats()["rejected_session"] == 1
    finally:
        client.release_first.set()
        coordinator.close()


def test_coordinator_fails_open_and_recovers_without_losing_last_good_state() -> None:
    class FlakyClient:
        def __init__(self) -> None:
            self.fail = False

        def infer(
            self,
            _jpeg_bytes: bytes,
            *,
            stream_session_id: str,
            frame_sequence: int,
            timestamp_seconds: float,
        ) -> RichLabelSnapshot:
            if self.fail:
                raise httpx.ConnectError("offline")
            return _snapshot(
                frame_sequence,
                timestamp_seconds,
                _object(),
                session_id=stream_session_id,
            )

    client = FlakyClient()
    coordinator = RichLabelCoordinator(client, max_fps=10_000)
    try:
        assert coordinator.submit(
            b"good",
            stream_session_id="session-1",
            frame_sequence=1,
            timestamp_seconds=0.1,
        )
        assert coordinator.wait_until_idle()
        good = coordinator.latest()
        client.fail = True

        assert coordinator.submit(
            b"bad",
            stream_session_id="session-1",
            frame_sequence=2,
            timestamp_seconds=0.2,
        )
        assert coordinator.wait_until_idle()

        assert coordinator.latest() is good
        assert coordinator.select("session-1", 0.2).objects == good.objects
        stats = coordinator.stats()
        assert stats["succeeded"] == 1
        assert stats["failed"] == 1
        assert stats["last_error"] == "rich-label inference failed"
        assert stats["state"] == "degraded"

        client.fail = False
        assert coordinator.submit(
            b"recovered",
            stream_session_id="session-1",
            frame_sequence=3,
            timestamp_seconds=0.3,
        )
        assert coordinator.wait_until_idle()
        assert coordinator.stats()["state"] == "ready"
        assert coordinator.stats()["last_error"] is None
    finally:
        coordinator.close()


def test_coordinator_bounds_sorted_history_and_close_is_idempotent() -> None:
    class Client:
        def __init__(self) -> None:
            self.closed = 0

        def infer(
            self,
            _jpeg_bytes: bytes,
            *,
            stream_session_id: str,
            frame_sequence: int,
            timestamp_seconds: float,
        ) -> RichLabelSnapshot:
            return _snapshot(
                frame_sequence,
                timestamp_seconds,
                session_id=stream_session_id,
            )

        def close(self) -> None:
            self.closed += 1

    client = Client()
    coordinator = RichLabelCoordinator(client, max_fps=10_000, history_size=3)
    for sequence in (0, 2, 1, 3):
        assert coordinator.submit(
            b"jpeg",
            stream_session_id="session-1",
            frame_sequence=sequence,
            timestamp_seconds=sequence / 10,
        )
        assert coordinator.wait_until_idle()

    assert [item.frame_sequence for item in coordinator.history()] == [1, 2, 3]
    assert coordinator.stats()["history_size"] == 3
    coordinator.close()
    coordinator.close()
    assert client.closed == 1
    assert coordinator.stats()["state"] == "closed"
    assert not coordinator.submit(
        b"after-close",
        stream_session_id="session-1",
        frame_sequence=4,
        timestamp_seconds=0.4,
    )


def test_rich_failure_does_not_publish_exception_class_or_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class PaddleRuntimeError(RuntimeError):
        pass

    class Client:
        def infer(self, *_args: Any, **_kwargs: Any) -> None:
            raise PaddleRuntimeError("PaddleX OpenRouter inference failure")

    coordinator = RichLabelCoordinator(Client(), max_fps=10_000)
    try:
        assert coordinator.submit(
            b"fixture", stream_session_id="session-1",
            frame_sequence=1, timestamp_seconds=0,
        )
        assert coordinator.wait_until_idle()
        assert coordinator.stats()["failed"] == 1
        assert coordinator.stats()["state"] == "degraded"
        assert coordinator.stats()["last_error"] == "rich-label inference failed"
        assert not contains_private_implementation_detail(
            json.dumps(coordinator.stats())
        )
        assert not contains_private_implementation_detail(caplog.text)
    finally:
        coordinator.close()
