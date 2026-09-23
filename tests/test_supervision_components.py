"""Exercise the pinned library and its boundaries, without models or service calls."""

import io
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from goodvision_harness.contracts import NormalizedBBox
from goodvision_harness.evaluation.detection import convert_dataset, detection_metrics
from goodvision_harness.observations.normalization import from_snapshot
from goodvision_harness.perception.adapters.http import SupervisionLabelClient
from goodvision_harness.perception.adapters.supervision import (
    from_detections,
)
from goodvision_harness.perception.components import (
    SupervisionPipeline,
    SupervisionRenderer,
)
from goodvision_harness.perception.settings import SupervisionSettings, Zone
from goodvision_harness.perception.tracking import ByteTrackBaseline
from goodvision_harness.rich_labeling import RichLabelObject, RichLabelSnapshot
from goodvision_harness.runtime.scene import SceneRuntime
from goodvision_harness.video_tracking import BACKEND

sv = pytest.importorskip("supervision")


def snapshot(sequence=0, at=None, objects=None, session="session"):
    return RichLabelSnapshot(
        session,
        sequence,
        float(sequence) if at is None else at,
        objects if objects is not None else (person(),),
    )


def person(track="person-a", box=(0.1, 0.1, 0.3, 0.3), confidence=0.9, **kwargs):
    return RichLabelObject(track, "person", NormalizedBBox(*box), confidence, **kwargs)


def observation(value):
    return from_snapshot(value, source_id="camera", model="synthetic")


def pipeline(**kwargs):
    return SupervisionPipeline(
        source_id="camera",
        stream_session_id="session",
        resolution_wh=(100, 100),
        settings=SupervisionSettings(enabled=True, **kwargs),
    )


def test_nms_selects_original_identity_provenance_and_unknown_confidence():
    original = snapshot(
        objects=(
            person("high", attributes={"identity": {"name": "Alice"}}),
            person("duplicate", confidence=0.4),
            person("unknown", box=(0.6, 0.2, 0.9, 0.7), confidence=None),
        )
    )
    result = pipeline(nms_threshold=0.5).process(observation(original))
    selected = result.observation.snapshot
    assert [o.track_id for o in selected.objects] == ["high", "unknown"]
    assert selected.objects[0] is original.objects[0]
    assert selected.objects[0].attributes["identity"]["name"] == "Alice"
    assert selected.objects[1].confidence is None
    assert len(original.objects) == 3
    filtered = pipeline(confidence_threshold=0.8).process(observation(original))
    assert [o.track_id for o in filtered.observation.snapshot.objects] == ["high"]


def test_native_tracks_are_never_suppressed_or_renumbered():
    attrs = {"native_track": {"backend": "sam", "epoch": "e", "object_id": "1"}}
    original = snapshot(
        objects=(person("a", attributes=attrs), person("b", attributes=attrs))
    )
    result = pipeline(nms_threshold=0, confidence_threshold=1).process(
        observation(original)
    )
    assert result.observation.snapshot == original


def test_masks_boxes_and_frame_local_untracked_ids_are_normalized():
    masks = np.zeros((2, 40, 80), dtype=bool)
    masks[0, 5:20, 10:30] = True
    masks[1, 20:35, 40:60] = True
    detections = sv.Detections(
        xyxy=sv.mask_to_xyxy(masks), mask=masks, class_id=np.zeros(2, dtype=int)
    )
    result = from_detections(
        detections,
        resolution_wh=(80, 40),
        stream_session_id="session",
        frame_sequence=7,
        timestamp_seconds=2.0,
        classes=("person",),
        track_namespace="sam-image",
    )
    assert len({obj.track_id for obj in result.objects}) == 2
    assert result.objects[0].track_id == "sam-image:frame:7:0"
    assert result.objects[0].bbox.x1 == 0.125
    contours = result.objects[0].attributes["mask_contours"]
    assert contours["source_frame_sequence"] == 7
    assert contours["source_timestamp_seconds"] == 2
    assert result.objects[0].confidence is None
    with pytest.raises(ValueError, match="mask resolution"):
        from_detections(
            detections,
            resolution_wh=(80, 41),
            stream_session_id="session",
            frame_sequence=7,
            timestamp_seconds=2,
            classes=("person",),
            track_namespace="sam",
        )


def test_zones_lines_empty_frames_and_source_order():
    processor = pipeline(
        zones=(Zone("lower", ((0.0, 0.5), (1.0, 0.5), (1.0, 1.0), (0.0, 1.0))),),
        lines=(Zone("gate", ((0.0, 0.5), (1.0, 0.5))),),
    )
    first = processor.process(observation(snapshot()))
    assert first.geometry["zones"]["lower"]["count"] == 0
    moved = snapshot(1, objects=(person(box=(0.1, 0.6, 0.3, 0.8)),))
    result = processor.process(observation(moved))
    assert result.geometry["zones"]["lower"] == {"count": 1, "track_ids": ["person-a"]}
    line = result.geometry["lines"]["gate"]
    assert line["in_count"] + line["out_count"] == 1
    assert line["crossed_in"] + line["crossed_out"] == ["person-a"]
    with pytest.raises(ValueError, match="source-ordered"):
        processor.process(observation(moved))
    empty = processor.process(observation(snapshot(2, objects=())))
    assert empty.geometry["zones"]["lower"]["count"] == 0
    assert len(empty.display_detections) == 0
    with pytest.raises(ValueError, match="another source/session"):
        processor.process(observation(snapshot(3, session="other")))


def test_smoothing_does_not_modify_evidence_or_create_missing_people():
    processor = pipeline(smoothing_length=3)
    processor.process(observation(snapshot()))
    moved = snapshot(1, objects=(person(box=(0.3, 0.3, 0.5, 0.5)),))
    result = processor.process(observation(moved))
    assert result.observation.snapshot.objects[0].bbox == moved.objects[0].bbox
    assert result.display_detections.xyxy[0, 0] == pytest.approx(20)
    predicted = snapshot(
        2,
        objects=(
            person(
                observation_state="predicted",
                last_observed_frame_sequence=1,
                last_observed_timestamp_seconds=1.0,
            ),
        ),
    )
    assert len(processor.process(observation(predicted)).detections) == 0


def test_gaps_epochs_and_capacity_are_explicit():
    processor = pipeline(max_tracks=1, max_gap_seconds=2)
    processor.process(observation(snapshot()))
    with pytest.raises(ValueError, match="capacity"):
        processor.process(observation(snapshot(1, objects=(person("another"),))))
    restarted = processor.process(
        observation(snapshot(4, objects=(person("another"),)))
    )
    assert restarted.geometry["reset_reason"] == "source_gap"
    for sequence, epoch in ((5, "one"), (6, "two")):
        result = processor.process(
            observation(
                snapshot(
                    sequence,
                    objects=(
                        person(
                            "another",
                            attributes={
                                "native_track": {
                                    "backend": "sam",
                                    "epoch": epoch,
                                    "object_id": "1",
                                }
                            },
                        ),
                    ),
                )
            )
        )
    assert result.geometry["reset_reason"] == "native_epoch"


def test_renderer_changes_copy_without_touching_source():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = pipeline().process(observation(snapshot()))
    image = SupervisionRenderer(heatmap=True).annotate(frame, result)
    assert np.any(image)
    assert not np.any(frame)


def test_bytetrack_baseline_keeps_ids_and_does_not_emit_predicted_evidence():
    tracker = ByteTrackBaseline(frame_rate=30)
    ids = []
    for sequence in range(3):
        detections = sv.Detections(
            xyxy=np.array([[10 + sequence, 10, 30 + sequence, 50.0]]),
            confidence=np.array([0.9]),
            class_id=np.array([0]),
        )
        result = tracker.update(detections, frame_sequence=sequence)
        ids.append(result.tracker_id.tolist())
    assert ids[0] and ids[0] == ids[1] == ids[2]
    assert len(tracker.update(sv.Detections.empty(), frame_sequence=3)) == 0
    with pytest.raises(ValueError, match="retracked"):
        tracker.update(result, frame_sequence=4)
    tracker.reset()
    assert len(tracker.update(detections, frame_sequence=0)) == 1


def test_detection_metrics_and_yolo_coco_roundtrip(tmp_path):
    images, annotations = tmp_path / "images", tmp_path / "annotations"
    images.mkdir()
    annotations.mkdir()
    Image.new("RGB", (100, 100)).save(images / "one.png")
    (annotations / "one.txt").write_text("0 0.5 0.5 0.4 0.4\n")
    data = tmp_path / "data.yaml"
    data.write_text("names: ['person']\nnc: 1\n")
    coco = tmp_path / "coco.json"
    convert_dataset(
        source_format="yolo",
        target_format="coco",
        source={
            "images_directory_path": str(images),
            "annotations_directory_path": str(annotations),
            "data_yaml_path": str(data),
        },
        destination={"annotations_path": str(coco)},
    )
    dataset = sv.DetectionDataset.from_coco(str(images), str(coco))
    target = next(iter(dataset.annotations.values()))
    prediction = replace(target, confidence=np.ones(len(target)))
    metrics = detection_metrics([prediction], [target])
    assert metrics.map50 == pytest.approx(1)
    assert dataset.classes == ["person"]


def test_live_runtime_exposes_geometry_without_safety_verdicts_and_isolates_failure():
    runtime = SceneRuntime(
        "session",
        "camera",
        supervision_settings=SupervisionSettings(
            enabled=True,
            max_tracks=1,
            zones=(Zone("all", ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),),
        ),
    )
    runtime.observe_tracks(snapshot())
    state = runtime.snapshot()
    assert state["supervision"]["observation"]["zones"]["all"]["count"] == 1
    assert state["scene"]["events"] == []
    runtime.observe_tracks(snapshot(1, objects=(person("other"),)))
    state = runtime.snapshot()
    assert state["supervision"]["failures"] == 1
    assert state["supervision"]["observation"]["state"] == "unavailable"
    assert len(state["scene"]["entities"]) == 2


def test_non_native_client_filters_and_resets_on_session_change():
    class Client:
        def infer(self, jpeg, **source):
            return RichLabelSnapshot(
                **source, objects=(person(), person("duplicate", confidence=0.4))
            )

    raw = io.BytesIO()
    Image.new("RGB", (100, 100)).save(raw, format="JPEG")
    client = SupervisionLabelClient(
        Client(), source_id="camera", settings=SupervisionSettings(nms_threshold=0.5)
    )
    for session in ("one", "two"):
        value = client.infer(
            raw.getvalue(),
            stream_session_id=session,
            frame_sequence=0,
            timestamp_seconds=0.0,
        )
        assert len(value.objects) == 1
        assert value.stream_session_id == session


def test_native_batch_geometry_runs_only_after_source_commit():
    runtime = SceneRuntime(
        "session",
        "camera",
        tracking_mode="batch",
        native_memory_scope="session",
        event_labels=(),
        supervision_settings=SupervisionSettings(enabled=True),
    )
    value = snapshot(
        objects=(
            person(
                attributes={
                    "native_track": {
                        "backend": BACKEND,
                        "epoch": "e",
                        "object_id": "1",
                    }
                }
            ),
        )
    )
    assert not runtime.observe_tracking_batch(
        (value,),
        completed_timestamp_seconds=0.0,
        completed_frame_sequence=0,
    )
    assert runtime.snapshot()["supervision"]["observation"] is None
    runtime.observe_frame(b"original", 0.0, 0)
    assert runtime.observe_tracking_batch(
        (value,),
        completed_timestamp_seconds=0.0,
        completed_frame_sequence=0,
    )
    assert runtime.snapshot()["supervision"]["observation"]["observed_count"] == 1
    assert not runtime.observe_tracking_batch(
        (value,),
        completed_timestamp_seconds=0.0,
        completed_frame_sequence=0,
    )
    assert runtime.snapshot()["supervision"]["failures"] == 0
