"""Delayed semantics board the next SAM batch without delaying raw publication."""

import copy
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_camera_display_protocol import RUN, batch, header
from test_face_observation_pipeline import runtime, window
from test_vlm_independent import config, report, verdict

from goodvision_harness.camera_display import CameraDisplayMetadata
from goodvision_harness.contracts import NormalizedBBox
from goodvision_harness.face_fast import FaceFastResults
from goodvision_harness.sam_display_fusion import SAMDisplayFusion
from goodvision_harness.scene_runtime import SceneRuntime
from goodvision_harness.vlm_independent import IndependentEventState

LEFT = NormalizedBBox(0.1, 0.1, 0.4, 0.9)
RIGHT = NormalizedBBox(0.6, 0.1, 0.9, 0.9)


def frame(sequence, *, box=LEFT, tracks=("a",), epoch="epoch-one"):
    objects = []
    for track in tracks:
        contours = {
            "coordinate_system": "normalized",
            "source_frame_sequence": sequence,
            "source_timestamp_seconds": sequence / 10,
            "polygons": [[[box.x1, box.y1], [box.x2, box.y1],
                          [box.x2, box.y2], [box.x1, box.y2]]],
        }
        attrs = {"native_track": {"backend": "sam", "epoch": epoch,
                                  "object_id": track}, "mask_contours": contours}
        objects.append(SimpleNamespace(
            bbox=box, confidence=0.9, observation_state="observed",
            as_dict=lambda attrs=attrs: {"attributes": attrs},
        ))
    return SimpleNamespace(frame_sequence=sequence, timestamp_seconds=sequence / 10,
                           objects=tuple(objects))


def setup():
    fusion = SAMDisplayFusion(RUN, "camera01")
    metadata = CameraDisplayMetadata()
    metadata.reset(RUN, fusion=fusion)
    metadata.accept_source(header())
    return metadata, fusion, IndependentEventState(config(RUN))


def face(sequence, *, person="zhang", name="张三", box=None, gallery=1):
    return {
        "frame": {"source_frame_index": sequence, "timestamp_seconds": sequence / 10},
        "gallery_version": gallery, "error": None,
        "faces": [{"bbox": (box or NormalizedBBox(0.2, 0.15, 0.3, 0.3)).as_dict(),
                   "status": "known", "person_id": person, "name": name}],
    }


def vlm(state, sequence, detected=True, *, box=LEFT):
    return report(state, sequence / 10, sequence, (verdict(detected, box=box),))


@pytest.mark.parametrize("order", [("face", "vlm"), ("vlm", "face"),
                                   ("face",), ("vlm",)])
def test_frame_five_semantics_board_frame_nine_at_moved_sam_person(order):
    metadata, fusion, state = setup()
    metadata.observe_batch(batch(*(frame(i) for i in range(1, 6))))
    metadata.observe_batch(batch(*(frame(i, box=RIGHT) for i in range(6, 9))))
    before = metadata.delta(None, stream_session_id=RUN)
    for kind in order:
        if kind == "face":
            fusion.offer_face(face(5))
        else:
            # Keep the established two-source confirmation rule.
            fusion.offer_vlm(vlm(state, 4))
            fusion.offer_vlm(vlm(state, 5))
    assert metadata.delta(before["metadata_revision"], stream_session_id=RUN) is None
    metadata.observe_batch(batch(frame(9, box=RIGHT)))
    update = metadata.delta(before["metadata_revision"], stream_session_id=RUN)
    assert [anchor["source_sequence"] for anchor in update["anchors"]] == [9]
    assert bool(update["identities"]) is ("face" in order)
    assert bool(update["events"]) is ("vlm" in order)
    for record in update["identities"] + update["events"]:
        assert record["track_id"] == "a"
        assert record["valid_from_timestamp"] == 0.9
    if "vlm" in order:
        assert update["events"][0]["phase"] == "confirmed"
    # Same boarded state drives server rendering; historical video is not relabeled.
    old, current = metadata.display_snapshot(0.8), metadata.display_snapshot(0.9)
    assert old["events"] == [] and not old["entities"][0]["identity"]
    assert bool(current["events"]) is ("vlm" in order)
    assert current["entities"][0]["bbox"] == {
        key: value for key, value in RIGHT.as_dict().items()
        if key != "coordinate_system"
    }
    if "face" in order:
        assert current["entities"][0]["identity"]["name"] == "张三"


def test_next_batch_is_required_and_duplicate_native_delivery_cannot_board():
    metadata, fusion, _ = setup()
    metadata.observe_batch(batch(frame(5)))
    before = metadata.delta(None, stream_session_id=RUN)
    fusion.offer_face(face(5))
    metadata.observe_batch(batch(frame(5)))
    assert metadata.delta(before["metadata_revision"], stream_session_id=RUN) is None
    metadata.observe_batch(batch(frame(6)))
    assert metadata.snapshot()["identities"][0]["valid_from_timestamp"] == 0.6


@pytest.mark.parametrize("tracks", [(), ("a", "b")])
def test_empty_or_ambiguous_exact_source_never_borrows_another_frame(tracks):
    metadata, fusion, state = setup()
    metadata.observe_batch(batch(frame(4), frame(5, tracks=tracks)))
    fusion.offer_face(face(5))
    fusion.offer_vlm(vlm(state, 5))
    metadata.observe_batch(batch(frame(6)))
    assert not metadata.snapshot()["identities"]
    assert not metadata.snapshot()["events"]


def test_nearest_source_can_follow_observation_and_new_epoch_cannot_inherit():
    metadata, fusion, state = setup()
    # Inference is ahead of SAM; no result or lock dependency on the other model.
    fusion.offer_face(face(5))
    fusion.offer_vlm(vlm(state, 5))
    metadata.observe_batch(batch(frame(1, tracks=())))
    assert not metadata.snapshot()["identities"]
    metadata.observe_batch(batch(frame(6)))
    assert metadata.snapshot()["identities"][0]["name"] == "张三"
    assert metadata.snapshot()["events"][0]["phase"] == "suspected"
    new_epoch = batch(frame(9, epoch="epoch-two"))
    new_epoch.predictor_epoch = "epoch-two"
    metadata.observe_batch(new_epoch)
    fusion.offer_face(face(6, person="stale", name="旧结果"))
    next_batch = batch(frame(10, epoch="epoch-two"))
    next_batch.predictor_epoch = "epoch-two"
    metadata.observe_batch(next_batch)
    assert not metadata.display_snapshot(1.0)["entities"][0]["identity"]
    assert not metadata.display_snapshot(1.0)["events"]


def test_out_of_order_clear_is_replayed_without_resurrecting_old_alert():
    metadata, fusion, state = setup()
    metadata.observe_batch(batch(*(frame(i) for i in range(1, 9))))
    for seq, detected in ((6, False), (7, False), (4, True), (5, True)):
        fusion.offer_vlm(vlm(state, seq, detected))
    metadata.observe_batch(batch(frame(9)))
    assert metadata.display_snapshot(0.9)["events"] == []
    intervals = copy.deepcopy(metadata.snapshot()["events"])
    metadata.observe_batch(batch(frame(10)))
    assert [(e["valid_from_timestamp"], e["valid_until_timestamp"])
            for e in metadata.snapshot()["events"]] == [
                (e["valid_from_timestamp"], e["valid_until_timestamp"])
                for e in intervals
            ]
    assert not metadata.display_snapshot(1.0)["events"]


def test_names_ignore_stale_matches_gallery_revocation_and_old_sessions():
    metadata, fusion, _ = setup()
    metadata.observe_batch(batch(*(frame(i) for i in range(1, 9))))
    fusion.offer_face(face(5))
    metadata.observe_batch(batch(frame(9)))
    fusion.offer_face(face(4, person="old", name="旧名字"))
    metadata.observe_batch(batch(frame(10)))
    assert metadata.display_snapshot(1.0)["entities"][0]["identity"]["name"] == "张三"
    fusion.offer_face(face(8))
    metadata.forget_identity("zhang")
    metadata.allow_identity("zhang")
    metadata.observe_batch(batch(frame(11)))
    assert not metadata.display_snapshot(1.1)["entities"][0]["identity"]
    fusion.require_gallery(2)
    fusion.offer_face(face(11))
    metadata.observe_batch(batch(frame(12)))
    assert not metadata.display_snapshot(1.2)["entities"][0]["identity"]
    metadata.reset("next-run", fusion=SAMDisplayFusion("next-run", "camera01"))
    fusion.offer_face(face(12, gallery=2))
    assert not metadata.observe_batch(batch(frame(13)))
    assert not metadata.snapshot()["identities"]


def test_clear_boards_later_and_missing_person_does_not_label_a_replacement():
    metadata, fusion, state = setup()
    metadata.observe_batch(batch(*(frame(i) for i in range(1, 9))))
    fusion.offer_vlm(vlm(state, 4))
    fusion.offer_vlm(vlm(state, 5))
    metadata.observe_batch(batch(frame(9)))
    assert metadata.display_snapshot(0.9)["events"]
    fusion.offer_vlm(vlm(state, 6, False))
    fusion.offer_vlm(vlm(state, 7, False))
    assert metadata.display_snapshot(0.9)["events"]
    metadata.observe_batch(batch(frame(10)))
    assert not metadata.display_snapshot(1.0)["events"]
    # Delayed identity for a departed person cannot name the new occupant.
    fusion.offer_face(face(5))
    metadata.observe_batch(batch(frame(11, tracks=("b",))))
    assert not metadata.display_snapshot(1.1)["entities"][0]["identity"]
    assert not metadata.display_snapshot(1.1)["events"]


def test_semantic_failure_cannot_suppress_native_masks(monkeypatch):
    metadata, fusion, _ = setup()

    def failed(_):
        raise RuntimeError("association unavailable")

    monkeypatch.setattr(fusion, "advance", failed)
    assert metadata.observe_batch(batch(frame(9)))
    payload = metadata.delta(None, stream_session_id=RUN)
    assert payload["anchors"][0]["source_sequence"] == 9
    assert payload["anchors"][0]["subjects"]


def test_fast_face_publication_and_display_mailbox_bypass_display_and_scene_locks(
    tmp_path, monkeypatch,
):
    metadata, fusion, _ = setup()
    engine, _, _, _ = runtime(tmp_path, monkeypatch)
    fast = FaceFastResults(stream_session_id=RUN, display_fusion=fusion)
    observation = engine.observe_window(window(5))
    scene = SceneRuntime(RUN, "camera01")
    scene.display_metadata = metadata
    with ThreadPoolExecutor(max_workers=1) as pool:
        with metadata._lock, scene._lock:
            pool.submit(fast.publish, observation).result(timeout=2)
            assert fast.snapshot()["available"]
        # Server rendering no longer takes the archival scene lock either.
        with scene._lock:
            assert pool.submit(scene.display_snapshot, 0.5).result(timeout=2)
    # Late faces are offered for historical binding without rewinding raw latest.
    fast.publish(replace(observation, window=window(4)))
    assert fast.snapshot()["result"]["frame"]["source_frame_index"] == 5
    assert len(fusion._inbox["face"]) == 2


def test_browser_receives_name_and_color_only_at_boarding_source():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for the display protocol")
    metadata, fusion, state = setup()
    metadata.observe_batch(batch(*(frame(i) for i in range(1, 9))))
    fusion.offer_face(face(5))
    fusion.offer_vlm(vlm(state, 4))
    fusion.offer_vlm(vlm(state, 5))
    metadata.observe_batch(batch(frame(9)))
    script = r"""
const assert=require('node:assert/strict');
const {ContourHistory,DisplayIdentities}=require(process.argv[1]);
let input='';process.stdin.on('data',chunk=>input+=chunk);process.stdin.on('end',()=>{
 const data=JSON.parse(input),config={
  stream_session_id:'camera-run',generation:3,crop_version:0};
 const history=new ContourHistory(config),names=new DisplayIdentities(config);
 for(let i=1;i<=9;i++)history.appendFrame({sequence:i,timestamp:i/10,
  width:16,height:16,pixels:new Uint8Array(256)});
 history.acceptMetadata(data);names.accept(data);
 const before=history.select(8,.8).subjects[0],after=history.select(9,.9).subjects[0];
 assert.equal(before.alert,false);assert.equal(names.nameAt(before,.8),null);
 assert.equal(after.alert,true);assert.equal(names.nameAt(after,.9),'张三');
});
"""
    module = Path(__file__).parents[1] / "src/goodvision_harness/camera_display.js"
    result = subprocess.run([node, "-e", script, str(module)],
                            input=json.dumps(metadata.snapshot()), text=True,
                            capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr
