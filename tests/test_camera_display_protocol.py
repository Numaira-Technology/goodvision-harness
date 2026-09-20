from __future__ import annotations

import copy
import json
import shutil
import struct
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from goodvision_harness.camera_display import (
    CameraDisplayMetadata,
    decode_camera_packet,
)

RUN = "camera-run"
NATIVE = {"backend": "sam31_stream", "epoch": "epoch-one", "object_id": "native-1"}
POLYGONS = [[[0.2, 0.2], [0.7, 0.2], [0.7, 0.7], [0.2, 0.7]]]


def header(sequence=0, timestamp=0.0):
    return {
        "stream_session_id": RUN,
        "generation": 3,
        "crop_version": 0,
        "source_sequence": sequence,
        "media_timestamp": timestamp,
        "width": 640,
        "height": 480,
    }


def bus():
    result = CameraDisplayMetadata()
    result.reset(RUN)
    assert result.accept_source(header())
    return result


def source(sequence=0, timestamp=0.0, **changes):
    contours = {
        "coordinate_system": "normalized",
        "source_frame_sequence": sequence,
        "source_timestamp_seconds": timestamp,
        "polygons": copy.deepcopy(POLYGONS),
    }
    contours.update(changes)
    attrs = {"native_track": dict(NATIVE), "mask_contours": contours}
    item = SimpleNamespace(
        observation_state="observed",
        confidence=0.9,
        as_dict=lambda: {"attributes": attrs},
    )
    return SimpleNamespace(
        frame_sequence=sequence, timestamp_seconds=timestamp, objects=(item,)
    )


def batch(*frames):
    return SimpleNamespace(
        stream_session_id=RUN, predictor_epoch="epoch-one", frames=frames
    )


def event(phase="confirmed", **changes):
    result = {
        "event_id": "event-one",
        "event_label": "smoking",
        "subject_ids": ["subject-one"],
        "phase": phase,
        "first_evidence_timestamp_seconds": 1.0,
        "last_evidence_timestamp_seconds": 2.0,
        "history": [
            {"timestamp_seconds": 1.0, "phase": "suspected"},
            {"timestamp_seconds": 2.0, "phase": "confirmed"},
        ],
    }
    result.update(changes)
    return result


def scene(now, *events, transitions=()):
    return {
        "scene": {
            "stream_session_id": RUN,
            "timestamp_seconds": now,
            "entities": [{"subject_id": "subject-one", "native_track": NATIVE}],
            "events": list(events),
            "transitions": list(transitions),
        }
    }


def named_scene(now, *, name="测试姓名", person_id="gallery-person", **changes):
    snapshot = scene(now)
    entity = snapshot["scene"]["entities"][0]
    entity.update(
        identity={"status": "known", "person_id": person_id, "name": name,
                  "identity_source": "face_match"},
        first_observed_timestamp_seconds=0,
        last_observed_timestamp_seconds=0,
        tracking_confirmation="confirmed",
    )
    entity.update(changes)
    return snapshot


def test_gallery_names_use_native_epoch_and_conservative_source_intervals():
    display = bus()
    display.observe_scene(named_scene(2))
    first = display.snapshot()["identities"][0]
    assert first == {
        "epoch": "epoch-one", "track_id": "native-1", "person_id": "gallery-person",
        "name": "测试姓名", "valid_from_timestamp": 2,
        "valid_until_timestamp": None, "revision": 1,
    }
    display.observe_scene(named_scene(3))
    assert display.snapshot()["identities"] == [first]
    assert display.snapshot()["identity_snapshot_revision"] == 1
    display.observe_scene(named_scene(4, name="另一个名字", person_id="another-person"))
    prior, current = display.snapshot()["identities"]
    assert prior["valid_from_timestamp"] == 2
    assert prior["valid_until_timestamp"] == current["valid_from_timestamp"] == 4
    assert current["name"] == "另一个名字"
    assert display.snapshot()["identity_snapshot_revision"] == 2
    display.observe_scene(named_scene(3, name="stale"))
    assert display.snapshot()["identities"][-1] == current
    display.observe_scene(scene(5))
    assert display.snapshot()["identities"][-1]["valid_until_timestamp"] == 5
    assert display.snapshot()["identity_snapshot_revision"] == 3
    display.observe_scene(scene(6))
    assert display.snapshot()["identity_snapshot_revision"] == 3


@pytest.mark.parametrize("change", [
    {"identity": {"status": "unknown", "person_id": "x", "name": "Not known"}},
    {"identity": {"status": "known", "matched_memory_id": "x", "name": "Reference"}},
    {"identity": {"status": "known", "person_id": "x", "name": "  "}},
    {"identity": {"status": "known", "person_id": "x", "name": "bad\nname"}},
    {"tracking_confirmation": "provisional"},
    {"native_track": {}},
])
def test_only_confirmed_gallery_names_can_create_a_display_identity(change):
    display = bus()
    display.observe_scene(named_scene(1, **change))
    assert display.snapshot()["identities"] == []
    assert display.snapshot()["identity_snapshot_revision"] == 0


def test_omission_and_conflicting_aliases_end_names_instead_of_guessing():
    display = bus()
    display.observe_scene(named_scene(1))
    missing = scene(2)
    missing["scene"]["entities"] = []
    display.observe_scene(missing)
    assert display.snapshot()["identities"][0]["valid_until_timestamp"] == 2
    conflicting = named_scene(3)
    conflicting["scene"]["entities"].extend([
        named_scene(3, name="Other", person_id="other")["scene"]["entities"][0],
        named_scene(3)["scene"]["entities"][0],
    ])
    display.observe_scene(conflicting)
    assert len(display.snapshot()["identities"]) == 1


def test_gallery_revocation_clears_history_and_reenrollment_needs_new_evidence():
    display = bus()
    display.observe_batch(batch(source()))
    display.observe_scene(named_scene(1))
    cursor = display.metadata_revision
    display.forget_identity("gallery-person")
    revoked = display.delta(cursor, stream_session_id=RUN)
    assert revoked["identities"] == []
    assert revoked["anchors"] == []
    assert revoked["identity_snapshot_revision"] == 2
    display.observe_scene(named_scene(2))
    assert display.snapshot()["identities"] == []
    display.allow_identity("gallery-person")
    assert display.snapshot()["identities"] == []
    display.observe_scene(named_scene(3))
    assert display.snapshot()["identities"][0]["valid_from_timestamp"] == 3
    display.reset("another-run")
    display.observe_scene(named_scene(4))
    assert display.snapshot()["identities"] == []


def test_identity_interval_and_revocation_protocol_matches_javascript():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for cross-language display protocol testing")
    display = bus()
    display.observe_batch(batch(*(source(i, float(i)) for i in range(5))))
    display.observe_scene(named_scene(2))
    known = display.snapshot()
    display.observe_scene(scene(4))
    ended = display.snapshot()
    display.forget_identity("gallery-person")
    revoked = display.snapshot()
    script = r"""
const assert=require('node:assert/strict'),{DisplayIdentities}=require(process.argv[1]);
let input='';process.stdin.on('data',chunk=>input+=chunk);process.stdin.on('end',()=>{
 const cases=JSON.parse(input),names=new DisplayIdentities({
  stream_session_id:'camera-run',
  generation:3,crop_version:0});
 const subject={epoch:'epoch-one',track_id:'native-1',
  tracking_confirmation:'confirmed'};
 const name=t=>names.nameAt(subject,t);
 names.accept(cases.known);assert.equal(name(1),null);assert.equal(name(2),'测试姓名');
 names.accept(cases.ended);assert.equal(name(3),'测试姓名');assert.equal(name(4),null);
 names.accept(cases.revoked);assert.equal(name(2),null);
 names.accept(cases.known);assert.equal(name(2),null);
});
"""
    module = Path(__file__).parents[1] / "src/goodvision_harness/camera_display.js"
    result = subprocess.run(
        [node, "-e", script, str(module)],
        input=json.dumps({"known": known, "ended": ended, "revoked": revoked}),
        text=True, capture_output=True, check=False, timeout=20,
    )
    assert result.returncode == 0, result.stderr


def test_gvc2_and_legacy_source_header_roundtrip():
    value = header(42, 1.25)
    encoded = json.dumps(value).encode()
    packet = b"GVC2" + struct.pack(">I", len(encoded)) + encoded + b"jpeg"
    assert decode_camera_packet(packet) == (b"jpeg", value)
    assert decode_camera_packet(struct.pack(">HH", 640, 480) + b"jpeg") == (
        b"jpeg",
        {"width": 640, "height": 480},
    )
    for bad in (b"GVC2", b"GVC2" + struct.pack(">I", 4097) + b"{}x"):
        with pytest.raises(ValueError):
            decode_camera_packet(bad)


def test_capture_run_generation_crop_and_both_source_orderings_are_fixed():
    display = bus()
    for change in (
        {"stream_session_id": "other"},
        {"generation": 4},
        {"crop_version": 1},
        {"source_sequence": 0},
        {"media_timestamp": 0.0},
    ):
        assert not display.accept_source({**header(1, 0.1), **change})
    assert display.accept_source(header(10, 0.2))


def test_anchor_requires_exact_contour_source_and_original_observation():
    display = bus()
    display.observe_batch(
        batch(
            source(1, 1.0, source_frame_sequence=999),
            source(2, 2.0, source_timestamp_seconds=1.0),
            source(3, 3.0, polygons=[[[0.2, 0.2], [0.2, 0.2], [0.2, 0.2]]]),
            source(4, 4.0),
        )
    )
    anchors = display.snapshot()["anchors"]
    assert [len(item["subjects"]) for item in anchors] == [0, 0, 0, 1]
    assert anchors[-1]["subjects"][0]["track_id"] == "native-1"
    original = source(5, 5.0)
    display.observe_batch(batch(original))
    original.objects[0].as_dict()["attributes"]["mask_contours"]["polygons"][0][0][
        0
    ] = 0.9
    assert display.snapshot()["anchors"][-1]["subjects"][0]["polygons"][0][0][0] == 0.2


def test_confirmation_starts_at_confirmed_source_and_survives_history_rollover():
    display = bus()
    display.observe_scene(scene(2, event()))
    assert display.snapshot()["events"][0]["valid_from_timestamp"] == 2
    display.observe_scene(
        scene(5, event(history=[{"timestamp_seconds": 5, "phase": "confirmed"}]))
    )
    assert display.snapshot()["events"][0]["valid_from_timestamp"] == 2
    display.observe_scene(
        scene(6, event(confirmed_since_timestamp_seconds=2, history=[]))
    )
    assert display.snapshot()["events"][0]["valid_from_timestamp"] == 2


def test_observed_provisional_sam_mask_is_display_only_and_keeps_confirmation_state():
    display = bus()
    provisional = source(1, 1.0)
    provisional_attrs = provisional.objects[0].as_dict()["attributes"]
    provisional_attrs["tracking_confirmation"] = "provisional"
    predicted = source(2, 2.0)
    predicted.objects[0].observation_state = "predicted"
    invalid = source(3, 3.0)
    invalid.objects[0].as_dict()["attributes"]["tracking_confirmation"] = "unknown"
    display.observe_batch(batch(provisional, predicted, invalid))
    anchors = display.snapshot()["anchors"]
    assert [len(item["subjects"]) for item in anchors] == [1, 0, 0]
    assert anchors[0]["subjects"][0]["tracking_confirmation"] == "provisional"
    assert display.snapshot()["events"] == []


def test_compact_confirmation_uses_transition_but_never_first_suspicion():
    display = bus()
    display.observe_scene(scene(2, event(history=[])))
    assert display.snapshot()["events"] == []
    display.observe_scene(
        scene(
            2,
            event(history=[]),
            transitions=[
                {
                    "event_id": "event-one",
                    "phase": "confirmed",
                    "timestamp_seconds": 2,
                }
            ],
        )
    )
    assert display.snapshot()["events"][0]["valid_from_timestamp"] == 2


def test_ended_interval_preserves_confirmation_until_the_real_clear_source():
    display = bus()
    display.observe_scene(scene(2, event()))
    display.observe_scene(
        scene(
            7,
            event(
                "ended",
                history=[],
                confirmed_since_timestamp_seconds=2,
                ended_since_timestamp_seconds=4,
                last_evidence_timestamp_seconds=7,
            ),
        )
    )
    interval = display.snapshot()["events"][0]
    assert interval["phase"] == "confirmed"
    assert interval["valid_from_timestamp"] == 2
    assert interval["valid_until_timestamp"] == 4
    assert interval["event_state"] == "ended"


def test_suspected_event_and_missing_identity_publish_yellow_intervals():
    display = bus()
    pending = scene(1, event("suspected", history=[]))
    pending["scene"]["entities"][0]["identity_pending"] = True
    display.observe_scene(pending)
    records = display.snapshot()["events"]
    assert len(records) == 2
    assert {item["phase"] for item in records} == {"suspected"}
    assert {item["event_label"] for item in records} == {"smoking", "identity_pending"}
    display.observe_scene(scene(2, event(confirmed_since_timestamp_seconds=2)))
    records = display.snapshot()["events"]
    confirmed = next(item for item in records if item["event_label"] == "smoking")
    assert confirmed["phase"] == "confirmed"
    assert confirmed["valid_from_timestamp"] == 2
    warning = next(
        item for item in records if item["event_label"] == "identity_pending"
    )
    assert warning["valid_until_timestamp"] == 2


def test_revised_confirmed_event_becomes_yellow_at_revision_time():
    display = bus()
    display.observe_scene(scene(2, event()))
    display.observe_scene(scene(4, event("suspected", history=[])))
    interval = display.snapshot()["events"][0]
    assert interval["phase"] == "suspected"
    assert interval["valid_from_timestamp"] == 4
    assert interval["valid_until_timestamp"] is None


@pytest.mark.parametrize("phase", ["retracted", None])
def test_retraction_or_full_snapshot_omission_closes_at_scene_source_time(phase):
    display = bus()
    display.observe_scene(scene(2, event()))
    display.observe_scene(scene(4, *([event(phase)] if phase else [])))
    interval = display.snapshot()["events"][0]
    assert (interval["valid_from_timestamp"], interval["valid_until_timestamp"]) == (
        2,
        4,
    )
    assert interval["event_state"] == "withdrawn"
    prior = display.snapshot()
    display.observe_scene(scene(3, event()))
    assert display.snapshot() == prior


def test_python_metadata_drives_real_javascript_color_intervals_and_source_alignment():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the browser protocol contract")
    display = bus()
    display.observe_batch(batch(*(source(i, float(i)) for i in range(7))))
    display.observe_scene(scene(2, event(confirmed_since_timestamp_seconds=2)))
    confirmed = display.snapshot()
    display.observe_scene(
        scene(
            4,
            event(
                "ended",
                confirmed_since_timestamp_seconds=2,
                ended_since_timestamp_seconds=4,
                last_evidence_timestamp_seconds=4,
            ),
        )
    )
    ended = display.snapshot()
    display.observe_scene(
        scene(
            5,
            event(
                event_id="event-two",
                first_evidence_timestamp_seconds=4.5,
                confirmed_since_timestamp_seconds=5,
                last_evidence_timestamp_seconds=5,
            ),
        )
    )
    next_episode = display.snapshot()
    display.observe_scene(scene(6))
    withdrawn = display.snapshot()
    script = r"""
const assert=require('node:assert/strict');
const {ContourHistory}=require(process.argv[1]);
let input='';process.stdin.on('data',chunk=>input+=chunk);
process.stdin.on('end',()=>{
 const cases=JSON.parse(input);
 const state=new ContourHistory({stream_session_id:'camera-run',generation:3,
   crop_version:0,
   max_history_seconds:10,max_propagation_seconds:10});
 for(let i=0;i<=6;i++)state.appendFrame({sequence:i,timestamp:i,width:16,height:16,
   pixels:new Uint8Array(256)});
 const red=t=>state.select(t,t).subjects.some(subject=>subject.alert);
 state.acceptMetadata(cases.confirmed);assert.equal(red(1),false);assert.equal(red(2),true);
 state.acceptMetadata(cases.ended);assert.equal(red(3),true);assert.equal(red(4),false);
 state.acceptMetadata(cases.next_episode);assert.equal(red(3),true);assert.equal(red(5),true);
 state.acceptMetadata(cases.withdrawn);assert.equal(red(5),true);assert.equal(red(6),false);
 state.acceptMetadata(cases.confirmed);assert.equal(red(6),false);
 console.log('source-aligned Python/JavaScript display protocol passed');
});
"""
    module = Path(__file__).parents[1] / "src/goodvision_harness/camera_display.js"
    result = subprocess.run(
        [node, "-e", script, str(module)],
        input=json.dumps(
            {
                "confirmed": confirmed,
                "ended": ended,
                "next_episode": next_episode,
                "withdrawn": withdrawn,
            }
        ),
        text=True,
        capture_output=True,
        check=False,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
