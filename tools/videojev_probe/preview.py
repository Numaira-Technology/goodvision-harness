"""Offline request inspection and checked replay of previously observed YOLO IDs."""

import json
from pathlib import Path

from .contract import OPTIONS, plan, questions
from .packet import digest, validate, write_json
from .ppe import experiment, values_for_question


def saved_people(packet, directory, manifest):
    directory = Path(directory)
    run = json.loads((directory / "manifest.json").read_text())
    if run.get("probe_scope") != "person" or run.get("state") != "finished":
        raise ValueError("saved person run must be a finished person probe")
    for name in ("input-manifest.json", "event-policy.json", "annotations.json"):
        if run.get("input_hashes", {}).get(name) != digest(Path(packet) / name):
            raise ValueError("saved person run uses different packet fingerprints")
    rows = {}
    for window in manifest["windows"]:
        wid = window["window_id"]
        row = json.loads((directory / f"{wid}-result.json").read_text())
        if row.get("window_id") != wid or row.get("complete") is not True:
            raise ValueError("saved person run contains a missing or failed window")
        snapshots = row.get("detector_snapshots")
        if not isinstance(snapshots, list):
            raise ValueError("saved run has no person snapshots")
        for snapshot in snapshots:
            people = snapshot.get("people")
            if not isinstance(people, list):
                raise ValueError("saved run has invalid person snapshots")
            # Validate coordinates/IDs with the existing production contract.
            from tools.benchmark_runtime.person_detector import validate_snapshot

            sequence, timestamp = (
                snapshot["frame_sequence"],
                snapshot["timestamp_seconds"],
            )
            objects = [
                dict(
                    **p,
                    label="person",
                    observation_state="observed",
                    last_observed_frame_sequence=sequence,
                    last_observed_timestamp_seconds=timestamp,
                )
                for p in people
            ]
            validate_snapshot(
                dict(
                    stream_session_id=wid,
                    frame_sequence=sequence,
                    timestamp_seconds=timestamp,
                    objects=objects,
                ),
                wid,
                sequence,
                timestamp,
            )
        questions(window, snapshots)  # Exact original-frame/session alignment.
        rows[wid] = snapshots
    return rows, {
        "mode": "saved_yolo_snapshots",
        "manifest_sha256": digest(directory / "manifest.json"),
        "result_sha256": {
            wid: digest(directory / f"{wid}-result.json") for wid in rows
        },
        "latency_note": "YOLO is not executed: timing excludes live detection; "
        "not full pipeline latency.",
    }


def preview(
    packet,
    output,
    *,
    ppe_variant="baseline",
    ppe_option_order="canonical",
    person_run=None,
):
    config = experiment(ppe_variant, ppe_option_order)
    manifest, _, _, _ = validate(packet)
    snapshots, replay = (
        saved_people(packet, person_run, manifest) if person_run else ({}, None)
    )
    requests = {
        w["window_id"]: questions(
            w,
            snapshots.get(w["window_id"]),
            ppe_variant=ppe_variant,
            ppe_option_order=ppe_option_order,
        )
        for w in manifest["windows"]
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "requests.json", requests)
    write_json(
        output / "option-semantics.json",
        {
            wid: {q["id"]: values_for_question(q, OPTIONS) for q in qs}
            for wid, qs in requests.items()
        },
    )
    result = {
        **plan(),
        "state": "offline_preview_only",
        "ppe_experiment": config,
        "scene_questions_per_window": 7 if ppe_variant == "equipment" else 6,
        "person_questions": "five event judgments; "
        "PPE uses two observations in equipment variant",
        "probe_scope": "person" if person_run else "scene",
        "detector_replay": replay,
        "input_hashes": {
            name: digest(Path(packet) / name)
            for name in ("input-manifest.json", "event-policy.json", "annotations.json")
        },
        "question_counts": {wid: len(qs) for wid, qs in requests.items()},
        "request_sha256": digest(output / "requests.json"),
    }
    write_json(output / "preview.json", result)
    return result
