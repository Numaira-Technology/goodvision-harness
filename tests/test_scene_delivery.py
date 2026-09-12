from __future__ import annotations

import copy
import io
import time
from threading import Event, current_thread

import pytest
from PIL import Image, ImageDraw

from goodvision_harness import scene_delivery
from goodvision_harness.result_bus import FramePacket
from goodvision_harness.scene_delivery import SceneDisplayWorker, render_scene_jpeg


def _jpeg(*, size=(360, 240), metadata=None):
    with Image.new("RGB", size, "black") as image, io.BytesIO() as buffer:
        options = {}
        if metadata:
            exif = Image.Exif()
            exif[270] = metadata
            options["exif"] = exif
        image.save(buffer, "JPEG", **options)
        return buffer.getvalue()


def _person(**changes):
    return {
        "subject_id": "person-1",
        "bbox": {"x1": 0.2, "y1": 0.3, "x2": 0.7, "y2": 0.9},
        "last_observed_timestamp_seconds": 1.0,
        "observation_state": "observed",
        "identity": {"status": "known", "name": "Alice"},
        **changes,
    }


def _event(**changes):
    return {
        "event_label": "smoking",
        "subject_ids": ["person-1"],
        "phase": "confirmed",
        "last_evidence_timestamp_seconds": 1.0,
        "observation_state": "observed",
        **changes,
    }


def _snapshot(*, people=None, events=()):
    return {
        "state": "running",
        "scene": {
            "entities": [_person()] if people is None else people,
            "events": list(events),
        },
    }


def _packet(sequence=1, timestamp=None, jpeg=None):
    return FramePacket(
        jpeg=jpeg or _jpeg(),
        media_timestamp=float(sequence if timestamp is None else timestamp),
        frame_sequence=sequence,
        observed_at="2026-09-12T00:00:00Z",
    )


@pytest.fixture
def drawn_text(monkeypatch):
    seen = []
    original = ImageDraw.ImageDraw.text

    def text(draw, xy, text, *args, **kwargs):
        seen.append(text)
        return original(draw, xy, text, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "text", text)
    return seen


def test_renderer_preserves_geometry_and_metadata_privacy(drawn_text):
    jpeg = _jpeg(metadata="SECRET_TOKEN")
    snapshot = _snapshot(events=(_event(),))
    before = copy.deepcopy(snapshot)
    result = render_scene_jpeg(jpeg, snapshot, 1)
    with Image.open(io.BytesIO(result)) as image:
        assert image.size == (360, 240)
        assert image.format == "JPEG"
        assert not image.getexif()
        assert image.getpixel((72, 150))[1] > 100
    assert any("person-1 | Alice" in text for text in drawn_text)
    assert any("smoking: confirmed" in text for text in drawn_text)
    assert all("hold" not in text and "uncertain" not in text for text in drawn_text)
    assert b"SECRET_TOKEN" not in result
    assert snapshot == before


def test_short_hold_is_labelled_and_dashed(monkeypatch, drawn_text):
    segments = []
    original = ImageDraw.ImageDraw.line

    def line(draw, xy, *args, **kwargs):
        segments.append(xy)
        return original(draw, xy, *args, **kwargs)

    monkeypatch.setattr(ImageDraw.ImageDraw, "line", line)
    render_scene_jpeg(_jpeg(), _snapshot(), 1.5)
    assert any("hold 0.5s" in text for text in drawn_text)
    assert len(segments) > 4
    assert all(abs(segment[2] - segment[0]) <= 5 for segment in segments)


@pytest.mark.parametrize(
    "person,timestamp",
    [
        (_person(), 2.01),
        (_person(), 0.9),
        (_person(observation_state="predicted"), 1.0),
        (_person(observation_state="held"), 1.0),
        (_person(observation_state="occluded"), 1.0),
        (_person(last_observed_timestamp_seconds=float("nan")), 1.0),
        (_person(bbox={"x1": -0.1, "y1": 0, "x2": 1, "y2": 1}), 1.0),
        (_person(bbox={"x1": 0.9, "y1": 0, "x2": 0.1, "y2": 1}), 1.0),
    ],
)
def test_old_future_predicted_or_invalid_geometry_is_never_painted(person, timestamp):
    jpeg = _jpeg()
    assert render_scene_jpeg(jpeg, _snapshot(people=[person]), timestamp) == (
        render_scene_jpeg(jpeg, _snapshot(people=[]), timestamp)
    )


def test_lost_geometry_keeps_event_text_with_uncertain_visibility(drawn_text):
    render_scene_jpeg(_jpeg(), _snapshot(events=(_event(),)), 3)
    assert not any(text.startswith("person-1") for text in drawn_text)
    assert any(
        "smoking: confirmed" in text and "uncertain" in text for text in drawn_text
    )
    assert not any("ended" in text for text in drawn_text)


@pytest.mark.parametrize("phase", ["suspected", "confirmed", "ended", "retracted"])
def test_event_lifecycle_and_uncertainty_are_displayed_explicitly(drawn_text, phase):
    render_scene_jpeg(
        _jpeg(),
        _snapshot(events=(_event(phase=phase, observation_state="uncertain"),)),
        1,
    )
    assert any(
        f"smoking: {phase}" in text and "uncertain" in text for text in drawn_text
    )


def test_non_ascii_name_falls_back_to_subject_without_crashing(drawn_text):
    render_scene_jpeg(
        _jpeg(), _snapshot(people=[_person(identity={"name": "张三"})]), 1
    )
    assert drawn_text == ["person-1"]


def test_long_labels_keep_hold_and_uncertainty_before_clipped_identity(drawn_text):
    subject = "person-" + "a" * 200
    render_scene_jpeg(
        _jpeg(),
        _snapshot(
            people=[_person(subject_id=subject, identity={"name": "B" * 100})],
            events=[_event(subject_ids=[subject], observation_state="uncertain")],
        ),
        1.5,
    )
    assert drawn_text[0].startswith("hold 0.5s | ")
    assert drawn_text[1].startswith("smoking: confirmed | uncertain | ")


def test_future_event_state_is_not_painted_on_an_earlier_frame(drawn_text):
    render_scene_jpeg(
        _jpeg(), _snapshot(events=(_event(last_evidence_timestamp_seconds=2),)), 1
    )
    assert not any("smoking" in text for text in drawn_text)


@pytest.mark.parametrize("old_count", [8, 64])
def test_current_events_take_priority_over_ended_history(drawn_text, old_count):
    old = [
        _event(phase="ended" if index % 2 else "retracted")
        for index in range(old_count)
    ]
    events = [
        *old,
        _event(event_label="weapon", last_evidence_timestamp_seconds=2),
        _event(
            event_label="fall", phase="suspected", last_evidence_timestamp_seconds=3
        ),
    ]
    render_scene_jpeg(_jpeg(), _snapshot(people=[], events=events), 3)
    assert len(drawn_text) == 8
    assert drawn_text[0].startswith("fall: suspected")
    assert drawn_text[1].startswith("weapon: confirmed")
    assert all("ended" in text or "retracted" in text for text in drawn_text[2:])


def test_drawing_counts_and_text_lengths_are_bounded(drawn_text):
    people = [
        _person(subject_id=f"person-{index}", identity={"name": "A" * 10000})
        for index in range(1000)
    ]
    events = [_event() for _ in range(1000)]
    render_scene_jpeg(
        _jpeg(size=(800, 600)), _snapshot(people=people, events=events), 1
    )
    assert len(drawn_text) == 128 + 8
    assert max(len(text) for text in drawn_text) <= 160


def test_tiny_images_and_raw_scene_snapshot_are_supported():
    rendered = render_scene_jpeg(
        _jpeg(size=(1, 1)), _snapshot(events=(_event(),))["scene"], 1
    )
    with Image.open(io.BytesIO(rendered)) as image:
        assert image.size == (1, 1)


def test_invalid_or_oversized_jpeg_is_rejected():
    for jpeg in (b"invalid", b"a" * (16 * 1024 * 1024 + 1)):
        with pytest.raises(ValueError):
            render_scene_jpeg(jpeg, _snapshot(), 1)


class Scene:
    def __init__(self):
        self.value = _snapshot()
        self.read_threads = []

    def snapshot(self):
        self.read_threads.append(current_thread().name)
        return self.value


def _await(predicate, timeout=2):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            pytest.fail("display worker did not reach expected state")
        time.sleep(0.005)


def test_submit_is_nonblocking_and_keeps_only_latest_pending_frame(monkeypatch):
    started, release = Event(), Event()
    scene, published = Scene(), []
    original = scene_delivery.render_scene_jpeg

    def render(*args, **kwargs):
        started.set()
        assert release.wait(3)
        return original(*args, **kwargs)

    monkeypatch.setattr(scene_delivery, "render_scene_jpeg", render)
    worker = SceneDisplayWorker(scene, published.append, max_fps=60)
    worker.start()
    try:
        packet = _packet()
        worker.submit(packet)
        assert started.wait(1)
        before = time.monotonic()
        for sequence in range(2, 12):
            worker.submit(_packet(sequence, jpeg=packet.jpeg))
        assert time.monotonic() - before < 0.2
        stats = worker.stats()
        assert stats["pending_frames"] == stats["max_pending_frames"] == 1
        assert stats["pending_bytes"] == len(packet.jpeg)
        assert stats["replaced_frames"] == 9
        assert all(name == "scene-display" for name in scene.read_threads)
        release.set()
        _await(lambda: len(published) == 2)
        assert [frame.frame_sequence for frame in published] == [1, 11]
        assert published[0].media_timestamp == 1
        assert published[0].observed_at == packet.observed_at
        assert published[0].analysis_message_id is None
        assert published[0].annotation_mode == "held"
    finally:
        release.set()
        worker.close()


def test_close_is_bounded_and_discards_an_inflight_render(monkeypatch):
    started, release = Event(), Event()
    published = []

    def render(jpeg, *_args, **_kwargs):
        started.set()
        assert release.wait(3)
        return jpeg

    monkeypatch.setattr(scene_delivery, "render_scene_jpeg", render)
    worker = SceneDisplayWorker(Scene(), published.append)
    worker.start()
    worker.submit(_packet())
    assert started.wait(1)
    worker.submit(_packet(2))
    before = time.monotonic()
    worker.close(timeout_seconds=0.02)
    assert time.monotonic() - before < 0.3
    assert worker.stats()["pending_bytes"] == 0
    assert worker.stats()["state"] == "stopping"
    release.set()
    _await(lambda: worker.stats()["state"] == "stopped")
    assert not published
    assert worker.stats()["discarded_frames"] == 2


def test_rate_limit_replaces_waiting_frames_without_extra_paints():
    published = []
    worker = SceneDisplayWorker(Scene(), published.append, max_fps=2)
    worker.start()
    try:
        worker.submit(_packet())
        _await(lambda: len(published) == 1)
        worker.submit(_packet(2))
        worker.submit(_packet(3))
        time.sleep(0.05)
        assert len(published) == 1
        assert worker.stats()["pending_frames"] == 1
        _await(lambda: len(published) == 2)
        assert published[-1].frame_sequence == 3
    finally:
        worker.close()


def test_bad_and_out_of_order_inputs_are_rejected_with_bounded_memory():
    worker = SceneDisplayWorker(Scene(), lambda _: None)
    packet = _packet(4)
    worker.submit(packet)
    worker.submit(packet)
    worker.submit(_packet(3))
    worker.submit(_packet(5, timestamp=3))
    worker.submit(_packet(5, jpeg=b"a" * (16 * 1024 * 1024 + 1)))
    worker.submit(_packet(5, timestamp=float("nan")))
    assert worker.stats()["rejected_frames"] == 5
    assert worker.stats()["pending_bytes"] == len(packet.jpeg)
    worker.close()
    worker.start()
    worker.submit(_packet(6))
    assert worker.stats()["state"] == "stopped"
    assert worker.stats()["pending_frames"] == 0


def test_render_failure_does_not_kill_next_frame_or_expose_error(monkeypatch):
    original = scene_delivery.render_scene_jpeg
    calls = 0

    def render(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("SECRET camera URL")
        return original(*args, **kwargs)

    monkeypatch.setattr(scene_delivery, "render_scene_jpeg", render)
    published = []
    worker = SceneDisplayWorker(Scene(), published.append, max_fps=60)
    worker.start()
    try:
        worker.submit(_packet())
        _await(lambda: worker.stats()["render_failures"] == 1)
        worker.submit(_packet(2))
        _await(lambda: len(published) == 1)
        assert "SECRET" not in str(worker.stats())
    finally:
        worker.close()


def test_stopped_scene_does_not_publish_frames():
    scene = Scene()
    scene.value["state"] = "stopped"
    published = []
    worker = SceneDisplayWorker(scene, published.append)
    worker.start()
    try:
        worker.submit(_packet())
        _await(lambda: worker.stats()["discarded_frames"] == 1)
        assert not published
    finally:
        worker.close()


def test_publish_callback_runs_outside_lock_and_can_close_worker():
    closed = Event()

    def publish(_packet):
        worker.close()
        closed.set()

    worker = SceneDisplayWorker(Scene(), publish)
    worker.start()
    worker.submit(_packet())
    assert closed.wait(1)
    worker.close()
    assert worker.stats()["published_frames"] == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"quality": 0},
        {"quality": 96},
        {"max_fps": float("inf")},
        {"max_fps": 0},
        {"max_fps": True},
    ],
)
def test_invalid_render_budgets_are_rejected(kwargs):
    with pytest.raises(ValueError):
        SceneDisplayWorker(Scene(), lambda _: None, **kwargs)
