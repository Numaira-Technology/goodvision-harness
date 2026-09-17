"""A later gallery match corrects the same person's earlier stranger episode."""

import copy
import threading

import pytest
from test_event_archive import fake_encoder, jpeg, result

from goodvision_harness.event_archive import EventArchive, EventArchiveSettings


def known(at, subject="face-fragment", *, source="face_match", state="observed"):
    value = result(at, f"known-{subject}-{at}", subject=subject, label="stranger")
    value["judgments"][0].update(detected=False, object_ids=[])
    value["objects"][0]["attributes"].update(
        status="known",
        person_id="frank",
        name="Frank",
        identity_source=source,
        face_observation_state=state,
    )
    return value


def test_later_match_retracts_only_same_person_stranger_and_survives_restart(tmp_path):
    settings = EventArchiveSettings(directory=tmp_path)
    archive = EventArchive(settings, clock=lambda: 0)
    try:
        archive.submit_result(
            result(52, "unknown-1", subject="face-fragment", label="stranger")
        )
        archive.submit_result(
            result(52.8, "unknown-2", subject="face-fragment", label="stranger")
        )
        archive.submit_result(result(52, "other", subject="visitor", label="stranger"))
        archive.submit_result(result(52, "smoking", subject="body"))
        assert archive.flush()
        before = archive.list_events(message_id="unknown-1")["events"][0]
        archive.submit_result(known(53.6))
        archive.reconcile_subjects("session", {"face-fragment": "body"})
        assert archive.flush()
        corrected = archive.get_event(before["event_id"])
        assert corrected["end_reason"] == "retracted"
        assert corrected["identity_state"] == "known"
        assert corrected["person_id"] == "frank"
        assert corrected["person_name"] == "Frank"
        assert corrected["subject_id"] == "body"
        assert corrected["evidence"]["failure_reason"] == "identity_confirmed"
        assert corrected["evidence"]["available"] is None
        assert corrected["start_seconds"] == 52
        assert corrected["last_detected_seconds"] == 52.8
        assert corrected["observation_count"] == 2
        assert (
            archive.list_events(subject_id="visitor")["events"][0]["end_reason"]
            != "retracted"
        )
        smoking = archive.list_events(event_label="smoking")["events"][0]
        assert smoking["end_reason"] != "retracted"
        assert smoking["person_name"] == "Frank"
    finally:
        archive.close()
    reopened = EventArchive(settings, clock=lambda: 0)
    try:
        assert reopened.get_event(before["event_id"])["end_reason"] == "retracted"
        assert reopened.get_event(before["event_id"])["person_name"] == "Frank"
    finally:
        reopened.close()


@pytest.mark.parametrize(
    "source,state", [("track_history", "observed"), ("face_match", "held")]
)
def test_held_identity_does_not_retract_stranger(tmp_path, source, state):
    archive = EventArchive(EventArchiveSettings(directory=tmp_path), clock=lambda: 0)
    try:
        archive.submit_result(
            result(1, "unknown", subject="face-fragment", label="stranger")
        )
        archive.submit_result(known(2, source=source, state=state))
        assert archive.flush()
        assert archive.list_events()["events"][0]["end_reason"] != "retracted"
    finally:
        archive.close()


def test_source_time_correction_handles_late_positive_without_excusing_future_faces(
    tmp_path,
):
    archive = EventArchive(EventArchiveSettings(directory=tmp_path), clock=lambda: 0)
    try:
        archive.submit_result(known(53.6))
        archive.submit_result(
            result(52, "late", subject="face-fragment", label="stranger")
        )
        archive.submit_result(
            result(54.4, "future", subject="face-fragment", label="stranger")
        )
        another_run = copy.deepcopy(
            result(52, "another-run", subject="face-fragment", label="stranger")
        )
        another_run["stream_session_id"] = "other-session"
        archive.submit_result(another_run)
        assert archive.flush()
        assert (
            archive.list_events(message_id="late")["events"][0]["end_reason"]
            == "retracted"
        )
        assert (
            archive.list_events(message_id="future")["events"][0]["end_reason"]
            != "retracted"
        )
        assert (
            archive.list_events(stream_session_id="other-session")["events"][0][
                "end_reason"
            ]
            != "retracted"
        )
    finally:
        archive.close()


def test_known_identity_survives_restart_before_alias_and_finalized_correction(
    tmp_path,
):
    settings = EventArchiveSettings(directory=tmp_path)
    clock = [0]
    archive = EventArchive(settings, clock=lambda: clock[0])
    try:
        archive.submit_result(result(1, "unknown", subject="body", label="stranger"))
        archive.submit_result(known(2))
        archive.end_session("session")
        assert archive.flush()
        clock[0] = 100
        assert archive.flush()
        original = archive.list_events()["events"][0]
        assert original["finalized"]
        assert original["end_reason"] != "retracted"
    finally:
        archive.close()
    reopened = EventArchive(settings, clock=lambda: 100)
    try:
        reopened.reconcile_subjects("session", {"face-fragment": "body"})
        assert reopened.flush()
        corrected = reopened.get_event(original["event_id"])
        assert corrected["end_reason"] == "retracted"
        assert corrected["person_name"] == "Frank"
        revision = corrected["revision"]
        reopened.reconcile_subjects("session", {"face-fragment": "body"})
        assert reopened.flush()
        assert reopened.get_event(original["event_id"])["revision"] == revision
    finally:
        reopened.close()


def test_encoder_completion_cannot_restore_retracted_stranger_video(tmp_path):
    gate, entered = threading.Event(), threading.Event()

    def encode(manifest, destination, **kwargs):
        entered.set()
        assert gate.wait(10)
        return fake_encoder(manifest, destination, **kwargs)

    archive = EventArchive(
        EventArchiveSettings(directory=tmp_path), encoder=encode, clock=lambda: 0
    )
    try:
        for index in range(21):
            archive.submit_frame(jpeg(), index / 5, index, "session")
        archive.submit_result(
            result(1, "unknown", subject="face-fragment", label="stranger")
        )
        assert entered.wait(3)
        archive.submit_result(known(2))
        assert archive.flush()
        corrected = archive.list_events()["events"][0]
        assert corrected["evidence"]["failure_reason"] == "identity_confirmed"
        gate.set()
        assert archive.wait_until_idle(3)
        corrected = archive.get_event(corrected["event_id"])
        assert corrected["end_reason"] == "retracted"
        assert corrected["evidence"]["available"] is None
        assert archive.video_path(corrected["event_id"]) is None
        # Published immutable revisions remain available for audit.
        assert archive.video_path(corrected["event_id"], 1).is_file()
    finally:
        gate.set()
        archive.close()
