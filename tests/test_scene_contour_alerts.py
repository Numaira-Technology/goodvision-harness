"""Exact SAM outlines and current event alerts through the real scene pipeline."""

from __future__ import annotations

import io
import time
from copy import deepcopy

import pytest
from PIL import Image, ImageDraw

from goodvision_harness.contracts import NormalizedBBox
from goodvision_harness.result_bus import FramePacket
from goodvision_harness.rich_labeling import RichLabelObject, RichLabelSnapshot
from goodvision_harness.scene_delivery import (
    TrackedSceneDisplayWorker,
    render_scene_jpeg,
)
from goodvision_harness.scene_runtime import SceneRuntime, TemporalVerdict

SIZE = (800, 600)
POLYGON = ((0.12, 0.4), (0.2, 0.25), (0.35, 0.4), (0.28, 0.65), (0.13, 0.75))


def _jpeg():
    with Image.new("RGB", SIZE, "black") as image, io.BytesIO() as buffer:
        image.save(buffer, "JPEG", quality=95)
        return buffer.getvalue()


def _mask(sequence, timestamp, offset=0):
    return {
        "coordinate_system": "normalized",
        "source_frame_sequence": sequence,
        "source_timestamp_seconds": timestamp,
        "polygons": [[[x + offset, y] for x, y in POLYGON]],
    }


def _track(subject, sequence, timestamp, *, offset=0, mask=True):
    return RichLabelObject(
        subject,
        "person",
        NormalizedBBox(0.1 + offset, 0.2, 0.4 + offset, 0.9),
        attributes={
            "tracking_confirmation": "confirmed",
            **({"mask_contours": _mask(sequence, timestamp, offset)} if mask else {}),
        },
    )


def _observe(scene, sequence, timestamp=None, *, mask=True, offset=0, second=False):
    timestamp = float(sequence if timestamp is None else timestamp)
    packet = FramePacket(_jpeg(), timestamp, sequence, f"source-{sequence}")
    scene.observe_frame(packet.jpeg, timestamp, sequence, force=True)
    objects = [_track("person-a", sequence, timestamp, offset=offset, mask=mask)]
    if second:
        objects.append(_track("person-b", sequence, timestamp, offset=0.5, mask=mask))
    snapshot = RichLabelSnapshot("session", sequence, timestamp, tuple(objects))
    if scene.tracking_mode == "batch":
        assert scene.observe_tracking_batch(
            (snapshot,),
            completed_timestamp_seconds=timestamp,
            completed_frame_sequence=sequence,
        )
    else:
        scene.observe_tracks(snapshot)
    return packet


def _event(scene, label, sequence, *, detected=True, timestamp=None, historical=False):
    assert scene.scene.observe_event(
        label,
        ("person-a",),
        timestamp_seconds=float(sequence if timestamp is None else timestamp),
        frame_sequence=sequence,
        detected=detected,
        evidence_id=f"{label}-{sequence}",
        allow_historical=historical,
    )


def _phases(scene):
    return {
        event["event_label"]: event["phase"]
        for event in scene.scene.snapshot()["events"]
    }


def test_visible_person_without_face_stays_yellow_without_creating_an_event():
    scene = SceneRuntime("session", "camera", stranger_enabled=True)
    try:
        source = _observe(scene, 1)
        assert scene.snapshot()["scene"]["events"] == []
        assert _outline_color(_render(scene, source)) == "yellow"
    finally:
        scene.close()


def test_confirmed_safety_event_stays_red_while_stranger_identity_is_pending():
    scene = SceneRuntime("session", "camera", stranger_enabled=True)
    try:
        for sequence in (1, 2):
            source = _observe(scene, sequence)
            _event(scene, "stranger", sequence)
            _event(scene, "smoking", sequence)
        events = {
            event["event_label"]: event for event in scene.snapshot()["scene"]["events"]
        }
        assert events["stranger"]["phase"] == "suspected"
        assert events["smoking"]["phase"] == "confirmed"
        assert _outline_color(_render(scene, source)) == "red"
    finally:
        scene.close()


def _render(scene, packet):
    return render_scene_jpeg(
        packet.jpeg,
        scene.display_snapshot(
            packet.media_timestamp, frame_sequence=packet.frame_sequence
        ),
        packet.media_timestamp,
        quality=95,
    )


def _outline_color(jpeg, offset=0):
    # Sample a real mask vertex, away from labels, HUD text and bbox edges.
    x = round((POLYGON[0][0] + offset) * (SIZE[0] - 1))
    y = round(POLYGON[0][1] * (SIZE[1] - 1))
    with Image.open(io.BytesIO(jpeg)) as image:
        samples = [
            image.getpixel((x + dx, y + dy))
            for dx in range(-1, 2)
            for dy in range(-1, 2)
        ]
    red, green, blue = max(samples, key=lambda rgb: rgb[0] + rgb[1])
    if red > 110 and red > green + 35 and red > blue + 35:
        return "red"
    assert green > 110 and green > blue + 35
    return "yellow" if red > green - 20 else "green"


@pytest.fixture
def draw_calls(monkeypatch):
    calls = {"line": [], "rectangle": [], "text": []}
    for name in calls:
        original = getattr(ImageDraw.ImageDraw, name)

        def record(draw, *args, _name=name, _original=original, **kwargs):
            calls[_name].append((args, kwargs))
            return _original(draw, *args, **kwargs)

        monkeypatch.setattr(ImageDraw.ImageDraw, name, record)
    return calls


@pytest.mark.parametrize("tracking_mode", ["immediate", "batch"])
def test_runtime_preserves_each_source_mask_and_does_not_transplant_it(tracking_mode):
    scene = SceneRuntime("session", "camera", tracking_mode=tracking_mode)
    try:
        first = _observe(scene, 1)
        _observe(scene, 2, offset=0.1)
        first_view = scene.display_snapshot(1, frame_sequence=1)
        second_view = scene.display_snapshot(2, frame_sequence=2)
        assert first_view["entities"][0]["mask_contours"] == _mask(1, 1)
        assert second_view["entities"][0]["mask_contours"] == _mask(2, 2, 0.1)
        assert _outline_color(_render(scene, first)) == "green"

        _observe(scene, 3, mask=False)
        assert "mask_contours" not in scene.scene.snapshot()["entities"][0]
        assert "mask_contours" not in scene.display_snapshot(3)["entities"][0]
        assert scene.display_snapshot(1)["entities"][0]["mask_contours"] == _mask(1, 1)

        mismatched = scene.display_snapshot(1, frame_sequence=99)
        assert mismatched["entities"][0]["observation_state"] == "occluded"
        assert render_scene_jpeg(first.jpeg, mismatched, 1) == render_scene_jpeg(
            first.jpeg, {"entities": [], "events": []}, 1
        )
    finally:
        scene.close()


@pytest.mark.parametrize("confirmed", [False, True])
@pytest.mark.parametrize("event_label", ["smoking", "stranger"])
def test_actual_mask_replaces_box_and_only_event_subject_uses_phase_color(
    draw_calls, confirmed, event_label
):
    scene = SceneRuntime("session", "camera")
    try:
        source = _observe(scene, 1, second=True)
        _event(scene, event_label, 1)
        if confirmed:
            source = _observe(scene, 2, second=True)
            _event(scene, event_label, 2)
        if event_label == "stranger":
            scene.apply_identity_review({"person-a": "unknown"})
        result = _render(scene, source)
        assert _outline_color(result) == ("red" if confirmed else "yellow")
        assert _outline_color(result, offset=0.5) == "green"
        assert len(draw_calls["line"]) == 2
        assert all("outline" not in kwargs for _, kwargs in draw_calls["rectangle"])
        with Image.open(io.BytesIO(result)) as image:
            assert max(image.getpixel((80, 330))) < 25  # No bounding-box edge.
            assert max(image.getpixel((200, 320))) < 25  # No filled-in mask.
        phase = "confirmed" if confirmed else "suspected"
        color = (239, 68, 68) if confirmed else (255, 214, 64)
        assert any(
            args[0] == (7, 5)
            and f"{event_label}: {phase}" in args[1]
            and kwargs["fill"] == color
            for args, kwargs in draw_calls["text"]
        )
    finally:
        scene.close()


def test_outline_stays_red_until_every_confirmed_event_reaches_its_clear_threshold():
    scene = SceneRuntime("session", "camera")
    try:
        for sequence in (1, 2):
            source = _observe(scene, sequence)
            _event(scene, "smoking", sequence)
            _event(scene, "weapon", sequence)
        assert _outline_color(_render(scene, source)) == "red"
        count = scene.scene.clear_observations
        for sequence in range(3, 3 + count):
            source = _observe(scene, sequence)
            _event(scene, "smoking", sequence, detected=False)
            if sequence < 2 + count:
                assert _phases(scene)["smoking"] == "confirmed"
        assert _phases(scene) == {"smoking": "ended", "weapon": "confirmed"}
        assert _outline_color(_render(scene, source)) == "red"
        for sequence in range(3 + count, 3 + count * 2):
            source = _observe(scene, sequence)
            _event(scene, "weapon", sequence, detected=False)
            if sequence < 2 + count * 2:
                assert _outline_color(_render(scene, source)) == "red"
        assert _phases(scene) == {"smoking": "ended", "weapon": "ended"}
        assert _outline_color(_render(scene, source)) == "green"
    finally:
        scene.close()


@pytest.mark.parametrize("reverse", [False, True])
def test_confirmed_event_wins_then_clearing_restores_suspected_color(reverse):
    scene = SceneRuntime("session", "camera")
    try:
        _observe(scene, 1)
        _event(scene, "stranger", 1)
        source = _observe(scene, 2)
        _event(scene, "stranger", 2)
        scene.apply_identity_review({"person-a": "unknown"})
        _event(scene, "smoking", 2)
        assert _phases(scene) == {"stranger": "confirmed", "smoking": "suspected"}
        view = scene.display_snapshot(2, frame_sequence=2)
        if reverse:
            view["events"].reverse()
        rendered = render_scene_jpeg(source.jpeg, view, 2, quality=95)
        assert _outline_color(rendered) == "red"

        count = scene.scene.clear_observations
        for sequence in range(3, 3 + count):
            source = _observe(scene, sequence)
            _event(scene, "stranger", sequence, detected=False)
            if sequence < 2 + count:
                assert _outline_color(_render(scene, source)) == "red"
        assert _phases(scene) == {"stranger": "ended", "smoking": "suspected"}
        assert _outline_color(_render(scene, source)) == "yellow"

        for sequence in range(3 + count, 3 + count * 2):
            source = _observe(scene, sequence)
            _event(scene, "smoking", sequence, detected=False)
        assert _phases(scene) == {"stranger": "ended", "smoking": "ended"}
        assert _outline_color(_render(scene, source)) == "green"
    finally:
        scene.close()


def test_late_event_newer_than_delayed_source_uses_current_state_and_exact_mask(
    draw_calls,
):
    scene = SceneRuntime("session", "camera")
    try:
        displayed = _observe(scene, 1)
        _observe(scene, 2, offset=0.05)
        _observe(scene, 3, offset=0.1)
        assert _outline_color(_render(scene, displayed)) == "green"
        _event(scene, "fall", 2, historical=True)
        assert scene.scene.snapshot_at(1)["events"] == []
        view = scene.display_snapshot(1, frame_sequence=1)
        assert view["event_state_timestamp_seconds"] == 3
        assert view["events"][0]["last_evidence_timestamp_seconds"] == 2
        assert view["entities"][0]["mask_contours"] == _mask(1, 1)
        assert _outline_color(_render(scene, displayed)) == "yellow"
        assert any("fall: suspected" in args[1] for args, _ in draw_calls["text"])
        assert scene.scene.snapshot()["timestamp_seconds"] == 3
    finally:
        scene.close()


def test_held_source_uses_dashed_box_without_reusing_mask(draw_calls):
    scene = SceneRuntime("session", "camera")
    try:
        source = _observe(scene, 1)
        render_scene_jpeg(source.jpeg, scene.display_snapshot(1.5), 1.5)
        assert draw_calls["line"]
        assert all(len(args[0]) == 4 for args, _ in draw_calls["line"])
        assert any("hold 0.5s" in args[1] for args, _ in draw_calls["text"])
    finally:
        scene.close()


def test_late_event_refreshes_displayed_source_without_new_tracking_or_capture():
    scene = SceneRuntime("session", "camera", tracking_mode="batch")
    published = []
    worker = TrackedSceneDisplayWorker(scene, published.append, max_fps=60, quality=95)
    try:
        _observe(scene, 0)
        source = _observe(scene, 1)
        worker.submit(source)
        worker.start()

        def await_frames(count):
            deadline = time.monotonic() + 2
            while len(published) < count:
                assert time.monotonic() < deadline, worker.stats()
                time.sleep(0.005)

        await_frames(1)
        assert _outline_color(published[0].jpeg) == "green"
        assert worker.stats()["pending_frames"] == 0
        tracked_before = scene.snapshot()["stats"]["completed_tracking_frames"]
        _event(scene, "stranger", 0, historical=True)
        worker.notify_events()
        await_frames(2)
        assert _outline_color(published[1].jpeg) == "yellow"
        _event(scene, "stranger", 1)
        scene.apply_identity_review({"person-a": "unknown"})
        worker.notify_events()
        await_frames(3)
        assert _outline_color(published[2].jpeg) == "red"
        sources = [(frame.frame_sequence, frame.media_timestamp) for frame in published]
        assert sources == [(1, 1.0), (1, 1.0), (1, 1.0)]
        assert all(frame.observed_at == source.observed_at for frame in published)
        assert all(frame.annotation_mode == "exact" for frame in published)
        assert scene.snapshot()["stats"]["completed_tracking_frames"] == tracked_before
        assert worker.stats()["submitted_frames"] == 1
        assert worker.stats()["render_failures"] == 0
    finally:
        worker.close()
        scene.close()


def test_claimed_late_verdict_redraws_exact_mask_after_source_history_expires(
    draw_calls,
):
    scene = SceneRuntime(
        "session", "camera", tracking_mode="batch", tracking_history_seconds=32,
        event_labels=("smoking",),
    )
    published = []
    worker = TrackedSceneDisplayWorker(scene, published.append, max_fps=60, quality=95)

    def await_frames(count):
        deadline = time.monotonic() + 2
        while len(published) < count:
            assert time.monotonic() < deadline, worker.stats()
            time.sleep(0.005)

    try:
        # Both confirmed observations must come from the same completed batch
        # and span at least 0.75s, so this creates a real claimable temporal job.
        packets, observations = [], []
        for sequence in (0, 1):
            timestamp = float(sequence)
            packet = FramePacket(_jpeg(), timestamp, sequence, f"source-{sequence}")
            packets.append(packet)
            scene.observe_frame(packet.jpeg, timestamp, sequence, force=True)
            observations.append(RichLabelSnapshot(
                "session", sequence, timestamp,
                (_track("person-a", sequence, timestamp),),
            ))
        assert scene.observe_tracking_batch(
            tuple(observations), completed_timestamp_seconds=1.0,
            completed_frame_sequence=1,
        )
        jobs = scene.take_inspections()
        assert len(jobs) == 1
        job = jobs[0]
        assert [(frame.frame_sequence, frame.timestamp_seconds)
                for frame in job.frames] == [(0, 0.0), (1, 1.0)]
        for packet in packets:
            worker.submit(packet)
        worker.start()
        await_frames(1)
        displayed = published[0]
        assert (displayed.frame_sequence, displayed.media_timestamp) == (1, 1.0)
        assert _outline_color(displayed.jpeg) == "green"

        # Capture continues while SAM does not. Both ordinary geometry history
        # and cached source JPEGs expire; the claimed VLM evidence stays valid.
        scene.observe_frame(_jpeg(), 40.0, 40, retain_source=False)
        worker.submit(FramePacket(_jpeg(), 40.0, 40, "source-40"), retain_source=False)
        assert scene.scene.snapshot()["entities"][0]["history"] == []
        assert len(scene.evidence) == 0
        assert scene.tracked_display_frame(before_timestamp_seconds=1.0) is None
        assert scene.display_snapshot(1.0, frame_sequence=1)["entities"] == []
        assert scene.scene.snapshot()["source_pins"]["jobs"] == 1
        tracked_before = scene.snapshot()["stats"]["completed_tracking_frames"]
        for calls in draw_calls.values():
            calls.clear()

        counts = {}
        assert scene.complete_inspection(
            job.request.request_id,
            [TemporalVerdict("smoking", ("person-a",), True, 0.9)],
            completion_counts=counts,
        )
        assert counts == {"accepted_verdicts": 1, "rejected_verdicts": 0}
        worker.notify_events()
        await_frames(2)
        refreshed = published[1]
        assert (refreshed.frame_sequence, refreshed.media_timestamp) == (1, 1.0)
        assert refreshed.observed_at == displayed.observed_at
        assert refreshed.annotation_mode == "exact"
        assert _outline_color(refreshed.jpeg) == "yellow"
        expected = tuple(
            (round(x * (SIZE[0] - 1)), round(y * (SIZE[1] - 1))) for x, y in POLYGON
        )
        assert len(draw_calls["line"]) == 1
        assert draw_calls["line"][0][0][0] == (*expected, expected[0])
        assert all("outline" not in kwargs for _, kwargs in draw_calls["rectangle"])
        assert any(
            args[0] == (7, 5) and "smoking: suspected" in args[1]
            for args, _ in draw_calls["text"]
        )
        assert scene.snapshot()["stats"]["completed_tracking_frames"] == tracked_before
        assert worker.stats()["render_failures"] == 0
        assert worker.stats()["retained_display_bytes"] == len(packets[1].jpeg)
    finally:
        worker.close()
        scene.close()
    assert worker.stats()["retained_display_bytes"] == 0


def test_claimed_late_verdict_recolors_cached_source_after_canonical_subject_merge(
    draw_calls,
):
    scene = SceneRuntime("session", "camera", tracking_mode="batch")
    published = []
    worker = TrackedSceneDisplayWorker(scene, published.append, max_fps=60, quality=95)
    try:
        packets, observations = [], []
        for sequence in (0, 1):
            packet = FramePacket(
                _jpeg(), float(sequence), sequence, f"source-{sequence}",
            )
            packets.append(packet)
            scene.observe_frame(packet.jpeg, packet.media_timestamp, sequence)
            observations.append(RichLabelSnapshot(
                "session", sequence, float(sequence),
                (_track("person-a", sequence, float(sequence)),),
            ))
        assert scene.observe_tracking_batch(
            tuple(observations), completed_timestamp_seconds=1.0,
            completed_frame_sequence=1,
        )
        (job,) = scene.take_inspections()
        for packet in packets:
            worker.submit(packet)
        worker.start()
        assert worker.wait_until_idle(2)
        assert len(published) == 1
        assert _outline_color(published[0].jpeg) == "green"

        # Identity association can change while the original request is in flight.
        # Its new current body location must not replace the displayed source mask.
        assert scene.scene.observe_person(
            "native-person", timestamp_seconds=2, frame_sequence=2,
            bbox=NormalizedBBox(0.55, 0.2, 0.85, 0.9),
            identity={"status": "known", "person_id": "alice", "name": "Alice"},
        )
        assert scene.apply_subject_merges((("person-a", "native-person"),)) == 1
        assert scene.complete_inspection(
            job.request.request_id,
            [TemporalVerdict("smoking", ("person-a",), True, 0.9)],
        )
        assert scene.scene.snapshot()["events"][0]["subject_ids"] == ["native-person"]
        for calls in draw_calls.values():
            calls.clear()
        worker.notify_events(scene.scene.snapshot()["events"])
        assert worker.wait_until_idle(2)
        assert len(published) == 2
        refreshed = published[-1]
        assert (refreshed.frame_sequence, refreshed.media_timestamp) == (1, 1.0)
        assert _outline_color(refreshed.jpeg) == "yellow"
        expected = tuple(
            (round(x * (SIZE[0] - 1)), round(y * (SIZE[1] - 1))) for x, y in POLYGON
        )
        assert len(draw_calls["line"]) == 1
        assert draw_calls["line"][0][0][0] == (*expected, expected[0])
        assert any("Alice" in args[1] for args, _ in draw_calls["text"])
        assert any("native-person" in args[1] for args, _ in draw_calls["text"])
        assert not any("person-a" in args[1] for args, _ in draw_calls["text"])
    finally:
        worker.close()
        scene.close()


@pytest.mark.parametrize("reverse", [False, True])
def test_cached_merged_entities_deduplicate_to_native_exact_source_geometry(reverse):
    scene = SceneRuntime("session", "camera")
    native = {"backend": "sam31", "epoch": "session", "object_id": "7"}
    face_box = NormalizedBBox(0.15, 0.25, 0.25, 0.4)
    body_box = NormalizedBBox(0.1, 0.2, 0.4, 0.9)
    try:
        for subject, bbox, extra in (
            ("face", face_box, {}),
            (
                "native", body_box,
                {"native_track": native, "mask_contours": _mask(1, 1)},
            ),
        ):
            assert scene.scene.observe_person(
                subject, timestamp_seconds=1, frame_sequence=1, bbox=bbox, **extra,
            )
        cached = scene.display_snapshot(1, frame_sequence=1)["entities"]
        if reverse:
            cached.reverse()
        original = deepcopy(cached)
        assert scene.apply_subject_merges((("face", "native"),)) == 1
        assert scene.scene.observe_person(
            "native", timestamp_seconds=2, frame_sequence=2,
            bbox=NormalizedBBox(0.5, 0.2, 0.8, 0.9), native_track=native,
            identity={"status": "known", "person_id": "alice", "name": "Alice"},
        )
        view = scene.display_snapshot(1, frame_sequence=1, source_entities=cached)
        assert len(view["entities"]) == 1
        entity = view["entities"][0]
        assert entity["subject_id"] == entity["entity_id"] == "native"
        assert entity["bbox"] == body_box.as_dict()
        assert entity["mask_contours"] == _mask(1, 1)
        assert entity["frame_sequence"] == 1
        assert entity["last_observed_timestamp_seconds"] == 1
        assert entity["identity"]["name"] == "Alice"
        assert cached == original
    finally:
        scene.close()
