"""Frozen inputs and strict choice decoding; no model imports or network calls."""

import json

from goodvision_harness.videojev_contract import (
    CONTEXT,
    MODEL,
    OPTIONS,
    REVISION,
    VALUES,
    choice_indices,
)

from tools.benchmark_runtime.event_sensitivity import EVENT_SENSITIVITY_DEFAULTS
from tools.benchmark_runtime.vlm_event_policy import (
    event_policy_snapshot,
    event_policy_text,
)

from .ppe import adapt, experiment, values_for_question

LABELS = tuple(EVENT_SENSITIVITY_DEFAULTS)
POLICY_SHA = "d9c487c37404df2f6d1f4d1f3da39559abffe28f59c76e7fef85f4a9f9e48a7b"
VIDEO_SHA = "2ac7a0f3271c4c9937b1796bdc43526617d9b1715de9bdf15713d8d3d80dacb9"
MAX_PEOPLE = 4


def plan():
    policy = event_policy_snapshot(LABELS)
    if policy["sha256"] != POLICY_SHA:
        raise ValueError("six-event criteria changed; do not silently update baseline")
    return {
        "profile": "videojev4b-candidate-v1",
        "model": MODEL,
        "model_revision": REVISION,
        "dtype": "BF16 backbone; official prepare_for_inference uses FP32 vision/head",
        "context": CONTEXT,
        "scope": "small sequential candidate probe; not twelve-stream acceptance",
        "replaces": "Qwen judgment only; existing YOLO26-S detector remains",
        "max_windows": 12,
        "scene_questions_per_window": 6,
        "person_questions": "optional: six scene questions plus five per last-frame ID",
        "max_people_per_probe_window": MAX_PEOPLE,
        "event_policy": policy,
        "source_video_sha256": VIDEO_SHA,
        "inference": "joint question groups; one visual encode per window",
        "input": "original JPEGs, in order, no frame selection or mosaics",
        "timestamp_note": (
            "Original source times in queries; native video positional grid uses "
            "the packet sampling_fps. Temporal patch padding is reported."
        ),
        "person_note": (
            "Fresh YOLO session per sparse window. IDs are window-scoped. "
            "Normalized locations are text evidence, not drawn onto inputs. "
            "Person judgments have no Astra person-level ground truth."
        ),
        "confidence": "option scores are not calibrated correctness probabilities",
        "execute": False,
    }


def select_windows(windows, count=12):
    if type(count) is not int or not 1 <= count <= 12 or count > len(windows):
        raise ValueError("choose 1..12 existing windows")
    # Stratify over time without looking at annotations or predictions.
    if count == 1:
        return [windows[0]]
    return [windows[round(i * (len(windows) - 1) / (count - 1))] for i in range(count)]


def questions(
    window,
    snapshots=None,
    *,
    ppe_variant="baseline",
    ppe_option_order="canonical",
    max_people=MAX_PEOPLE,
):
    plan()  # Check the policy before preparing any model input.
    experiment(ppe_variant, ppe_option_order)
    evidence = {"source_times": [f["media_seconds"] for f in window["frames"]]}
    people = []
    if snapshots is not None:
        if len(snapshots) != len(window["frames"]):
            raise ValueError("detector must cover every input frame")
        for i, (frame, snapshot) in enumerate(
            zip(window["frames"], snapshots, strict=True)
        ):
            if (
                snapshot["state"] != "ready"
                or snapshot["frame_sequence"] != i
                or snapshot["timestamp_seconds"] != frame["media_seconds"]
                or snapshot["stream_session_id"] != window["window_id"]
            ):
                raise ValueError("detector failed or returned mismatched evidence")
        evidence["per_frame_people"] = [s["people"] for s in snapshots]
        people = [p["track_id"] for p in snapshots[-1]["people"]]
        if len(people) > max_people:
            raise ValueError("probe person budget exceeded; no person silently dropped")
        if len(set(people)) != len(people):
            raise ValueError("duplicate person ID")
    prefix = event_policy_text(LABELS) + "\nEvidence metadata: " + json.dumps(evidence)
    prefix += (
        "\nJudge the last original frame using all earlier frames as context. "
        "If supplied, person positions are normalized 0..1, per frame only. "
        "IDs are local tracking references, never real identities. "
        "Missing IDs do not mean an event is absent. "
        "For an ambiguous person association select insufficient evidence."
    )
    result = []
    for person in [None, *people]:
        for label in LABELS:
            if person and label == "ground_litter":
                continue  # Litter stays scene-level.
            target = "the whole visible scene" if person is None else f"person {person}"
            result.append(
                {
                    "id": f"{person or 'scene'}/{label}",
                    "task": "choice",
                    "query": prefix
                    + f"\nFor {target}, is {label} present under that policy?",
                    "options": list(OPTIONS),
                }
            )
    return adapt(result, window, snapshots, ppe_variant, ppe_option_order)


def parse_choices(raw, expected):
    indices = choice_indices(raw, expected)
    # No confidence cutoff and no conversion of uncertain into negative.
    return {
        q["id"]: values_for_question(q, OPTIONS)[indices[q["id"]]] for q in expected
    }
