"""Mask transport/geometry contract checks without model or GPU startup."""

import json
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from goodvision_harness.contracts import NormalizedBBox
from goodvision_harness.person_registry import PersonRegistry
from goodvision_harness.rich_labeling import (
    RichLabelProtocolError,
    RichLabelSnapshot,
    select_rich_labels,
)
from goodvision_harness.scene_state import SceneState
from goodvision_harness.video_tracking import MAX_RESPONSE_BYTES
from workers.sam_scene.sam31 import (
    SourceFrame,
    VideoBatch,
    native_mask_geometry,
    native_snapshot,
)


def snapshot_payload():
    mask = np.zeros((60, 80), dtype=bool)
    # A nonrectangular body and a disconnected limb, plus a rejected speckle.
    mask[10:45, 20:30] = True
    mask[35:45, 20:55] = True
    mask[10:20, 45:50] = True
    mask[1, 1] = True
    before = mask.copy()
    payload = native_snapshot(
        {"out_obj_ids": np.array([7]), "out_probs": np.array([0.9]),
         "out_binary_masks": mask[None]},
        batch=VideoBatch("session", "batch", (SourceFrame(12, 1.5, b""),)),
        index=0, epoch="epoch", image_size=(80, 60), max_objects=64,
    )
    assert np.array_equal(mask, before)
    return payload


def test_native_mask_contours_follow_shape_components_and_source():
    payload = snapshot_payload()
    item = payload["objects"][0]
    outline = item["attributes"]["mask_contours"]
    assert outline["source_frame_sequence"] == 12
    assert outline["source_timestamp_seconds"] == 1.5
    assert len(outline["polygons"]) == 2
    assert len(outline["polygons"][0]) > 4  # Preserves the L-shaped boundary.
    assert item["attributes"]["native_track"]["object_id"] == "7"
    assert item["attributes"]["mask_geometry"]["ignored_component_pixels"] == 1
    for polygon in outline["polygons"]:
        for x, y in polygon:
            assert item["bbox"]["x1"] <= x <= item["bbox"]["x2"]
            assert item["bbox"]["y1"] <= y <= item["bbox"]["y2"]
    parsed = RichLabelSnapshot.from_dict(payload)
    assert parsed.as_dict()["objects"][0]["attributes"] == item["attributes"]


def test_native_mask_outline_payload_is_bounded_for_fragmented_masks():
    mask = np.zeros((400, 400), dtype=bool)
    for y in range(0, 400, 40):
        for x in range(0, 400, 40):
            mask[y:y + 20, x:x + 20] = True
    _, polygons, _, _ = native_mask_geometry(mask)
    assert len(polygons) == 16
    assert sum(map(len, polygons)) <= 1024
    assert all(3 <= len(polygon) <= 128 for polygon in polygons)
    # A connected high-frequency border also needs to respect per-ring bounds.
    mask[:, :150] = True
    for y in range(0, 400, 4):
        mask[y:y + 2, 150:350] = True
    _, polygons, _, _ = native_mask_geometry(mask)
    assert sum(map(len, polygons)) <= 1024
    assert all(3 <= len(polygon) <= 128 for polygon in polygons)


def test_response_bound_accepts_full_worker_contours_and_pending_source_release():
    # 14 pending sources + 16 appended sources, each with the worker's maximum
    # 16 people. Every outline reaches its 1024-point ceiling.
    payload = snapshot_payload()
    ring = [[round(float(0.5 + 0.4 * np.cos(angle)), 6),
             round(float(0.5 + 0.4 * np.sin(angle)), 6)]
            for angle in np.linspace(0, 2 * np.pi, 128, endpoint=False)]
    payload["objects"][0]["attributes"]["mask_contours"]["polygons"] = [ring] * 8
    payload["objects"] *= 16
    encoded = json.dumps({"frames": [payload] * 30}).encode()
    assert 2 * 1024 * 1024 < len(encoded) <= MAX_RESPONSE_BYTES


@pytest.mark.parametrize("change", [
    lambda outline: outline.update(coordinate_system="pixels"),
    lambda outline: outline.update(source_frame_sequence=13),
    lambda outline: outline.update(source_frame_sequence=True),
    lambda outline: outline.update(source_timestamp_seconds=1.6),
    lambda outline: outline.update(source_timestamp_seconds=float("inf")),
    lambda outline: outline.update(source_timestamp_seconds=10 ** 400),
    lambda outline: outline.update(polygons=[]),
    lambda outline: outline.update(polygons=[[[0, 0], [1, 0]]]),
    lambda outline: outline.update(polygons=[[[0, 0], [1, 0], [2, 1]]]),
    lambda outline: outline.update(polygons=[[[0, 0], [1, 0], [10 ** 400, 1]]]),
    lambda outline: outline.update(polygons=[[[0, 0], [1, 0], [True, 1]]]),
    lambda outline: outline.update(polygons=[[[0, 0], [1, 0], [0.5, 0]]]),
    lambda outline: outline.update(polygons=[[[0, 0], [1, 0], [0, float("nan")]]]),
    lambda outline: outline.update(polygons=[[[0, 0], [1, 0], [0, 1, 2]]]),
    lambda outline: outline.update(polygons=outline["polygons"] * 9),
    lambda outline: outline.update(polygons=[[[0, 0], [1, 0], [0, 1]] * 43]),
    lambda outline: outline.update(polygons=[[[0, 0], [1, 0], [0, 1]] * 40] * 9),
    lambda outline: outline.pop("source_frame_sequence"),
])
def test_malformed_or_misattributed_mask_contours_rejected(change):
    payload = snapshot_payload()
    change(payload["objects"][0]["attributes"]["mask_contours"])
    with pytest.raises(RichLabelProtocolError):
        RichLabelSnapshot.from_dict(payload)


@pytest.mark.parametrize("native", [True, False])
def test_registry_preserves_contours_while_assigning_public_identity(native):
    payload = snapshot_payload()
    if not native:
        del payload["objects"][0]["attributes"]["native_track"]
    original = RichLabelSnapshot.from_dict(payload)
    registry = PersonRegistry("session", "camera")
    registry.observe_snapshot(original)
    canonical = registry.snapshot_at(1.5, 12)
    assert canonical.objects[0].track_id != original.objects[0].track_id
    assert canonical.objects[0].attributes["mask_contours"] == (
        original.objects[0].attributes["mask_contours"]
    )


def test_interpolation_does_not_invent_masks_for_unobserved_source():
    payload = snapshot_payload()
    left = RichLabelSnapshot.from_dict(payload)
    right_payload = deepcopy(payload)
    right_payload.update(frame_sequence=14, timestamp_seconds=2.0)
    right_item = right_payload["objects"][0]
    right_item.update(last_observed_frame_sequence=14,
                      last_observed_timestamp_seconds=2.0)
    right_item["attributes"]["mask_contours"].update(
        source_frame_sequence=14, source_timestamp_seconds=2.0,
    )
    right = RichLabelSnapshot.from_dict(right_payload)
    selected = select_rich_labels(
        (left, right), 1.75, frame_sequence=13, max_gap_seconds=1.0
    )
    assert selected.mode == "interpolated"
    assert "mask_contours" not in selected.snapshot.objects[0].attributes
    with pytest.raises(ValueError, match="predicted"):
        replace(left.objects[0], observation_state="predicted")


def observe_outline(scene, sequence, subject="a", **kwargs):
    return scene.observe_person(
        subject, timestamp_seconds=float(sequence), frame_sequence=sequence,
        bbox=NormalizedBBox(0.1, 0.1, 0.9, 0.9),
        mask_contours={
            "coordinate_system": "normalized", "source_frame_sequence": sequence,
            "source_timestamp_seconds": float(sequence),
            "polygons": [[[0.1, 0.1], [0.9, 0.1], [0.5, 0.9]]],
        }, **kwargs,
    )


def assert_contour_accounting(scene, expected):
    actual = sum(
        len(polygon)
        for person in scene.snapshot()["entities"]
        for record in person["history"]
        for polygon in record.get("mask_contours", {}).get("polygons", ())
    )
    assert actual == scene._contour_points == expected
    assert actual <= scene.max_contour_history_points
    assert len(scene._contour_order) <= 2 * len(scene._contour_records) + 64


def test_scene_contour_budget_is_global_and_evicts_only_oldest_geometry():
    scene = SceneState("session", "camera", max_contour_history_points=6)
    assert observe_outline(scene, 0, identity={"status": "known", "name": "A"})
    assert scene.observe_event(
        "smoking", ("a",), timestamp_seconds=0, frame_sequence=0,
        detected=True, evidence_id="e-0", confidence=0.9,
    )
    original = scene.snapshot()
    assert observe_outline(scene, 1, "b")
    assert observe_outline(scene, 2)
    snapshot = scene.snapshot()
    assert_contour_accounting(scene, 6)
    person = next(item for item in snapshot["entities"] if item["subject_id"] == "a")
    assert [item["frame_sequence"] for item in person["history"]] == [0, 2]
    assert "mask_contours" not in person["history"][0]
    assert person["mask_contours"]["source_frame_sequence"] == 2
    assert person["identity"] == original["entities"][0]["identity"]
    assert snapshot["events"] == original["events"]
    assert "mask_contours" in original["entities"][0]["history"][0]
    assert "mask_contours" not in scene.snapshot_at(0)["entities"][0]
    assert scene.snapshot_at(0)["entities"][0]["bbox"] == person["history"][0]["bbox"]


def test_late_source_contour_evicts_itself_without_replacing_current_geometry():
    scene = SceneState("session", "camera", max_contour_history_points=3)
    assert observe_outline(scene, 3)
    assert observe_outline(scene, 1, allow_historical=True)
    assert_contour_accounting(scene, 3)
    person = scene.snapshot()["entities"][0]
    assert [item["frame_sequence"] for item in person["history"]] == [1, 3]
    assert "mask_contours" not in person["history"][0]
    assert person["mask_contours"]["source_frame_sequence"] == 3
    assert "mask_contours" not in scene.snapshot_at(1)["entities"][0]


def test_same_source_replacement_and_history_pruning_reclaim_budget_and_index():
    scene = SceneState(
        "session", "camera", max_contour_history_points=6, person_history_size=2,
        person_history_seconds=4,
    )
    assert observe_outline(scene, 0)
    for _ in range(200):
        assert observe_outline(scene, 0)
    assert_contour_accounting(scene, 3)
    for source in range(1, 5):
        assert observe_outline(scene, source)
    assert_contour_accounting(scene, 6)
    history = scene.snapshot()["entities"][0]["history"]
    assert [item["frame_sequence"] for item in history] == [3, 4]
    # Historical replacement must release the prior record, not double-charge.
    assert observe_outline(scene, 3, allow_historical=True)
    assert_contour_accounting(scene, 6)
    scene.advance(9)
    assert_contour_accounting(scene, 0)
    assert "mask_contours" not in scene.snapshot()["entities"][0]
    assert not scene._contour_records


def test_contour_budget_accounting_handles_subject_merge_and_expiration():
    scene = SceneState("session", "camera", max_contour_history_points=12)
    assert observe_outline(scene, 0, "old")
    assert observe_outline(scene, 0, "new")
    assert observe_outline(scene, 1, "old")
    assert_contour_accounting(scene, 9)
    assert scene.merge_subject("old", "new")
    assert_contour_accounting(scene, 6)
    assert len(scene.snapshot()["entities"]) == 1
    scene.advance(100)
    assert_contour_accounting(scene, 0)
    assert not scene._contour_records


@pytest.mark.parametrize("budget", [0, -1, True, 131073])
def test_scene_contour_point_budget_cannot_exceed_hard_bound(budget):
    with pytest.raises(ValueError, match="max_contour_history_points"):
        SceneState("session", "camera", max_contour_history_points=budget)
