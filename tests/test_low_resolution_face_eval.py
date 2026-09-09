from tools.low_resolution_face_eval import summarize


def _face(status="known", identity="identity-1"):
    return {
        "status": status,
        "identity": identity,
        "matches_enrollment": status == "known",
    }


def test_missing_detections_cannot_be_reported_as_stable_identity():
    known = _face()
    rows = [
        {"raw": [known], "clear_seed": [known], "inference_ms": 10},
        {"raw": [], "clear_seed": [known], "inference_ms": 10},
        {"raw": [known], "clear_seed": [known], "inference_ms": 10},
    ]
    summary = summarize(rows, "clear_seed")
    assert summary["frames_with_face_detection"] == 2
    assert summary["frames_without_face_detection"] == 1
    assert summary["frames_with_face_output"] == 3
    assert summary["adjacent_single_face_pairs"] == 0


def test_switch_metric_separates_unusable_unknown_and_different_known_identity():
    faces = [
        _face(),
        _face("unusable", None),
        _face("unknown", None),
        _face(),
        _face(identity="identity-2"),
    ]
    rows = [{"raw": [face], "inference_ms": 10} for face in faces]
    summary = summarize(rows, "raw")
    assert summary["adjacent_single_face_pairs"] == 4
    assert summary["adjacent_state_or_identity_switches"] == 4
    assert summary["adjacent_known_unknown_switches"] == 1
    assert summary["adjacent_known_identity_switches"] == 1
