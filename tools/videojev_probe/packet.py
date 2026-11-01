"""Export a portable subset of the independently annotated original evidence."""

import copy
import hashlib
import json
import math
from pathlib import Path

from tools.vlm_benchmark.behavior_reference import load_packet, packet_file

from .contract import POLICY_SHA, VIDEO_SHA, plan, questions, select_windows


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(packet):
    packet = Path(packet)
    data = load_packet(packet, packet / "annotations.json")
    manifest, policy, _reference, _images = data
    if manifest["source_video_sha256"] != VIDEO_SHA or policy["sha256"] != POLICY_SHA:
        raise ValueError("use the original video and frozen six-class reference")
    if not 1 <= len(manifest["windows"]) <= 12:
        raise ValueError("small probe only: at most 12 windows")
    fps = manifest.get("sampling_fps")
    if type(fps) not in (int, float) or not math.isfinite(fps) or fps != 2.5:
        raise ValueError("retain the original 2.5 fps evidence clock")
    return data


def prepare(source, output, count=12):
    source, output = Path(source).resolve(), Path(output).resolve()
    manifest, policy, reference, _images = load_packet(
        source, source / "annotations.json"
    )
    if manifest["source_video_sha256"] != VIDEO_SHA or policy["sha256"] != POLICY_SHA:
        raise ValueError("wrong original video or reference policy")
    if manifest.get("sampling_fps") != 2.5:
        raise ValueError("retain the original 2.5 fps evidence clock")
    selected = select_windows(manifest["windows"], count)
    ids = {w["window_id"] for w in selected}
    manifest = copy.deepcopy(manifest)
    manifest["windows"] = selected
    reference = copy.deepcopy(reference)
    reference["rows"] = [r for r in reference["rows"] if r["window_id"] in ids]
    reference["inspected_window_ids"] = [w["window_id"] for w in selected]
    reference["uninspected_window_ids"] = []
    reference.pop("inspected_contact_sheets", None)
    reference.pop("additional_individual_frames_inspected", None)
    reference["subset_note"] = "Time-stratified subset; reference verdicts unchanged."
    output.mkdir(parents=True, exist_ok=False)
    for w in selected:
        for f in w["frames"]:
            path = output / f["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(packet_file(source, f["path"]).read_bytes())
        w.pop("contact_sheet", None)
    write_json(output / "input-manifest.json", manifest)
    write_json(output / "event-policy.json", policy)
    write_json(output / "annotations.json", reference)
    write_json(
        output / "probe-plan.json",
        {
            **plan(),
            "selected_windows": [w["window_id"] for w in selected],
            "reference_sha256": digest(source / "annotations.json"),
            "source_manifest_sha256": digest(source / "input-manifest.json"),
            "selection": "uniform source-order index spacing; label-blind",
        },
    )
    # Predictions/references are never part of this request material.
    write_json(
        output / "scene-questions.json",
        {w["window_id"]: questions(w) for w in selected},
    )
    validate(output)
    return output
