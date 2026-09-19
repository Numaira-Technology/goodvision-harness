from __future__ import annotations

import base64
import io
import json
from dataclasses import replace
from threading import Thread

import pytest
from PIL import Image

from goodvision_harness.contracts import NormalizedBBox
from goodvision_harness.scene_evidence import EvidenceFrame, InspectionRequest
from goodvision_harness.scene_inspection import (
    build_temporal_messages,
    run_pending_inspections,
)
from goodvision_harness.scene_runtime import (
    SceneRuntime,
    TemporalInspection,
    TemporalVerdict,
)

BOX = NormalizedBBox(0.1, 0.1, 0.4, 0.9)


def _jpeg(color="red", size=(1200, 600), metadata=None):
    with Image.new("RGB", size, color) as image, io.BytesIO() as encoded:
        options = {}
        if metadata:
            exif = Image.Exif()
            exif[270] = metadata
            options["exif"] = exif
        image.save(encoded, format="JPEG", **options)
        return encoded.getvalue()


def _job(*, targeted=True, boxes=True):
    return TemporalInspection(
        "session-with-private-token",
        "https://private.test/?token=SECRET",
        InspectionRequest(
            "request-1",
            ("person-1",) if targeted else (),
            "new_person",
            ("smoking", "fight"),
            0,
            1,
        ),
        (
            EvidenceFrame("session-with-private-token", 10, 0, _jpeg()),
            EvidenceFrame("session-with-private-token", 11, 1, _jpeg("blue")),
        ),
        ((10, "person-1", BOX), (11, "person-1", BOX)) if boxes else (),
    )


class _Runtime:
    """An acknowledgement stub; real runtime coverage below checks state acceptance."""

    def __init__(self, jobs, *, acknowledge=True):
        self.jobs = tuple(jobs)
        self.completed = []
        self.acknowledge = acknowledge

    def take_inspections(self):
        jobs, self.jobs = self.jobs, ()
        return jobs

    def complete_inspection(
        self,
        request_id,
        verdicts,
        *,
        success=True,
        completion_counts=None,
    ):
        self.completed.append((request_id, tuple(verdicts), success))
        if completion_counts is not None:
            completion_counts.update(
                accepted_verdicts=len(verdicts), rejected_verdicts=0
            )
        return self.acknowledge


class _Inspector:
    def __init__(self, verdicts=None, *, error=None):
        self.verdicts = (
            verdicts
            if verdicts is not None
            else (TemporalVerdict("smoking", ("person-1",), True, 0.9),)
        )
        self.error = error
        self.jobs = []
        self.messages = []

    def inspect(self, job):
        self.jobs.append(job)
        if self.error:
            raise self.error
        self.messages.append(build_temporal_messages(job))
        return self.verdicts


def test_messages_preserve_order_times_all_images_and_source_boxes():
    job = _job()
    messages = build_temporal_messages(job)
    content = messages[1]["content"]
    assert [item["type"] for item in content] == [
        "text",
        "text",
        "image_url",
        "text",
        "image_url",
    ]
    assert json.loads(content[0]["text"])["allowed_subject_ids"] == ["person-1"]
    assert json.loads(content[1]["text"])["frame_sequence"] == 10
    assert json.loads(content[3]["text"])["source_timestamp_seconds"] == 1
    assert json.loads(content[1]["text"])["subject_boxes"] == [
        {"subject_id": "person-1", "bbox": [0.1, 0.1, 0.4, 0.9]},
    ]
    for part in (content[2], content[4]):
        data = base64.b64decode(part["image_url"]["url"].split(",", 1)[1])
        with Image.open(io.BytesIO(data)) as image:
            assert image.size == (720, 360)
    assert "every supplied frame" in messages[0]["content"]
    assert "detected=null" in messages[0]["content"]


def test_compact_metadata_preserves_each_frames_own_people_boxes_and_time():
    job = _job(targeted=False)
    second = NormalizedBBox(0.312345678, 0.2, 0.712345678, 0.8)
    job = replace(job, subject_boxes=(
        (10, "person-1", BOX), (11, "person-2", second),
    ))
    messages = build_temporal_messages(job)
    context, first, last = [
        json.loads(part["text"]) for part in messages[1]["content"]
        if part["type"] == "text"
    ]
    assert context == {
        "allowed_event_labels": ["smoking", "fight"],
        "allowed_subject_ids": ["person-1", "person-2"],
    }
    assert first == {
        "frame_sequence": 10, "source_timestamp_seconds": 0,
        "subject_boxes": [{"subject_id": "person-1", "bbox": [0.1, 0.1, 0.4, 0.9]}],
    }
    assert last == {
        "frame_sequence": 11, "source_timestamp_seconds": 1,
        "subject_boxes": [{
            "subject_id": "person-2", "bbox": [0.312346, 0.2, 0.712346, 0.8],
        }],
    }
    assert job.subject_boxes[1][2] == second


@pytest.mark.parametrize("level", ["standard", "high"])
def test_temporal_policy_has_no_grounding_output_instructions(level):
    job = _job()
    labels = ("smoking", "fight", "fall", "weapon")
    job = replace(job, request=replace(job.request, event_labels=labels),
                  event_sensitivities=tuple((label, level) for label in labels))
    prompt = build_temporal_messages(job)[0]["content"]
    assert "Return only a JSON array" in prompt
    assert "Do not output new\nbounding boxes, descriptions" in prompt
    assert "Box the actual item" not in prompt
    assert "box the visible" not in prompt
    assert "box the actual smoking item" not in prompt
    assert "Report suspected weapon" not in prompt
    assert "describe a suspected smoking action" not in prompt


def test_message_builder_strips_jpeg_metadata_and_source_secrets():
    job = _job()
    job = replace(
        job,
        frames=(
            replace(job.frames[0], jpeg_bytes=_jpeg(metadata="SECRET-EXIF-CREDENTIAL")),
            job.frames[1],
        ),
    )
    messages = build_temporal_messages(job)
    assert "private.test" not in str(messages)
    assert "private-token" not in str(messages)
    assert "SECRET" not in str(messages)
    encoded = messages[1]["content"][2]["image_url"]["url"].split(",", 1)[1]
    raw = base64.b64decode(encoded)
    assert b"SECRET-EXIF-CREDENTIAL" not in raw
    with Image.open(io.BytesIO(raw)) as image:
        assert not image.getexif()


def test_configured_quality_changes_only_jpeg_encoding_and_preserves_default():
    job = _job()
    default = build_temporal_messages(job)
    assert default == build_temporal_messages(job, jpeg_quality=90)
    reduced = build_temporal_messages(job, jpeg_quality=70)
    assert reduced[0] == default[0]  # Exact prompt and event criteria are unchanged.
    for original, changed in zip(
        default[1]["content"], reduced[1]["content"], strict=True,
    ):
        if original["type"] != "image_url":
            assert changed == original
            continue
        data = base64.b64decode(changed["image_url"]["url"].split(",", 1)[1])
        with Image.open(io.BytesIO(data)) as image:
            assert image.size == (720, 360)
            assert image.quantization[0][0] == 10
            assert not image.getexif()


@pytest.mark.parametrize("quality", [0, 101, True, 70.5, "70"])
def test_message_builder_rejects_invalid_jpeg_quality(quality):
    with pytest.raises(ValueError, match="jpeg_quality"):
        build_temporal_messages(_job(), jpeg_quality=quality)


def test_full_scene_without_boxes_never_invents_subject_ids():
    messages = build_temporal_messages(_job(targeted=False, boxes=False))
    assert json.loads(messages[1]["content"][0]["text"])["allowed_subject_ids"] == []
    assert "Return []" in messages[0]["content"]
    with pytest.raises(ValueError, match="source-frame location"):
        build_temporal_messages(_job(boxes=False))


@pytest.mark.parametrize(
    "change", ["future", "unordered", "foreign", "one_frame", "box"]
)
def test_invalid_temporal_associations_are_rejected(change):
    job = _job()
    if change == "future":
        job = replace(
            job, frames=(job.frames[0], replace(job.frames[1], timestamp_seconds=2))
        )
    elif change == "unordered":
        job = replace(job, frames=tuple(reversed(job.frames)))
    elif change == "foreign":
        job = replace(
            job,
            frames=(job.frames[0], replace(job.frames[1], stream_session_id="other")),
        )
    elif change == "one_frame":
        job = replace(job, frames=job.frames[:1])
    else:
        job = replace(job, subject_boxes=((99, "person-1", BOX),))
    with pytest.raises(ValueError):
        build_temporal_messages(job)


def test_explicit_worker_consumes_temporal_images_and_completes_claim():
    runtime, inspector = _Runtime((_job(),)), _Inspector()
    counts = run_pending_inspections(runtime, inspector)
    assert counts.claimed_requests == counts.succeeded_requests == 1
    assert counts.submitted_verdicts == counts.accepted_verdicts == 1
    assert counts.failed_requests == counts.rejected_verdicts == 0
    assert len(inspector.jobs[0].frames) == 2
    assert len(inspector.messages[0][1]["content"]) == 5
    assert runtime.completed[0][2] is True
    assert run_pending_inspections(runtime, inspector).claimed_requests == 0


def test_invalid_jpeg_fails_before_invoking_model():
    job = _job()
    job = replace(
        job, frames=(replace(job.frames[0], jpeg_bytes=b"not-jpeg"), job.frames[1])
    )
    runtime, inspector = _Runtime((job,)), _Inspector()
    counts = run_pending_inspections(runtime, inspector)
    assert counts.failed_requests == 1
    assert not inspector.jobs
    assert runtime.completed == [("request-1", (), False)]


@pytest.mark.parametrize(
    "verdicts",
    [
        (TemporalVerdict("smoking", ("foreign-person",), True),),
        (TemporalVerdict("stranger", ("person-1",), True),),
        (TemporalVerdict("smoking", ("person-1",), "yes"),),
        (TemporalVerdict("smoking", ("person-1",), True, float("nan")),),
        (TemporalVerdict("smoking", ("person-1",), True),) * 2,
        ({"event_label": "smoking"},),
    ],
)
def test_bad_verdict_batches_fail_without_submitting_partial_evidence(verdicts):
    runtime = _Runtime((_job(),))
    counts = run_pending_inspections(runtime, _Inspector(verdicts))
    assert counts.failed_requests == 1
    assert counts.submitted_verdicts == 0
    assert runtime.completed == [("request-1", (), False)]


def test_model_exception_releases_claim_without_exposing_secret_error(capsys):
    runtime = _Runtime((_job(),))
    counts = run_pending_inspections(
        runtime,
        _Inspector(error=RuntimeError("API_KEY=SECRET_TOKEN")),
    )
    assert counts.failed_requests == 1
    assert runtime.completed == [("request-1", (), False)]
    assert "SECRET" not in repr(counts)
    assert not capsys.readouterr().out
    assert not capsys.readouterr().err


def test_expired_completion_is_discarded_not_reported_success():
    counts = run_pending_inspections(
        _Runtime((_job(),), acknowledge=False), _Inspector()
    )
    assert counts.discarded_requests == 1
    assert counts.succeeded_requests == counts.accepted_verdicts == 0


def test_worker_call_budget_is_bounded():
    inspector = _Inspector()
    counts = run_pending_inspections(_Runtime((_job(),) * 6), inspector)
    assert len(inspector.jobs) == 4
    assert counts.succeeded_requests == 4
    assert counts.failed_requests == 2


def test_distinct_inspections_of_one_source_frame_count_only_one_positive():
    runtime = SceneRuntime("session", "camera")
    for sequence, timestamp in enumerate((0.0, 1.0)):
        runtime.scene.observe_person(
            "person-1",
            timestamp_seconds=timestamp,
            frame_sequence=sequence,
            bbox=BOX,
            observation_state="observed",
        )
        runtime.observe_frame(
            _jpeg("red" if sequence == 0 else "blue"), timestamp, sequence
        )

    class Inspector(_Inspector):
        def inspect(self, job):
            # Another thread can read the runtime while the worker is inspecting.
            reader = Thread(target=runtime.snapshot)
            reader.start()
            reader.join(timeout=1)
            assert not reader.is_alive()
            return super().inspect(job)

    counts = run_pending_inspections(runtime, Inspector())
    assert counts.succeeded_requests == 2
    assert counts.submitted_verdicts == 2
    assert counts.accepted_verdicts == 2
    assert counts.rejected_verdicts == 0
    events = runtime.snapshot()["scene"]["events"]
    assert len(events) == 1
    assert events[0]["positive_observations"] == 1
    assert events[0]["phase"] == "suspected"
    assert len(events[0]["history"]) == 1
    assert len(events[0]["history"][0]["evidence_ids"]) == 2
