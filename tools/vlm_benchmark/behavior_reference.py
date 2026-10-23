"""Validate offline Astra packets against original six-class evidence."""

import hashlib
import io
import json
import math
import re
from itertools import pairwise
from pathlib import Path, PurePosixPath, PureWindowsPath

from PIL import Image

from goodvision_harness.grounding import SAFETY_EVENT_LABELS
from tools.benchmark_runtime.vlm_event_policy import annotation_policy


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    def unique(pairs):
        result = {}
        for k, v in pairs:
            if k in result:
                raise ValueError("duplicate JSON key")
            result[k] = v
        return result

    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def packet_file(root, name):
    if not isinstance(name, str):
        raise ValueError("invalid packet path")
    relative = PurePosixPath(name)
    if (
        relative.is_absolute()
        or PureWindowsPath(name).drive
        or ".." in relative.parts
        or "\\" in name
    ):
        raise ValueError("packet paths must be portable relative paths")
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("packet files must exist inside packet")
    return path


def load_packet(packet, annotations):
    """Fail before requests if provenance, windows, policy or images disagree."""
    packet = Path(packet).resolve()
    manifest = read_json(packet / "input-manifest.json")
    policy = read_json(packet / "event-policy.json")
    reference = read_json(annotations)
    if set(policy["event_labels"]) != set(SAFETY_EVENT_LABELS):
        raise ValueError("this comparison requires all six event labels")
    expected = annotation_policy(policy["event_labels"], policy["sensitivities"])
    if any(
        policy.get(k) != expected[k]
        for k in (
            "version",
            "sha256",
            "event_labels",
            "sensitivities",
            "decision_rules",
            "comparison_field",
        )
    ):
        raise ValueError("packet policy differs from current production policy")
    digest = manifest.get("source_video_sha256", "")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("invalid source video fingerprint")
    if (
        reference.get("source_video_sha256") != digest
        or reference.get("policy_sha256") != expected["sha256"]
        or manifest.get("policy_sha256") != expected["sha256"]
        or manifest.get("policy_version") != expected["version"]
    ):
        raise ValueError("reference, packet and current policy fingerprints differ")
    if (
        reference.get("human_reviewed") is not False
        or manifest.get("prediction_data_included") is not False
        or manifest.get("previous_annotations_included") is not False
        or not reference.get("annotator_model")
    ):
        raise ValueError("independent AI reference provenance is required")
    if not 1 <= len(manifest["windows"]) <= 1000:
        raise ValueError("packet must have 1..1000 windows")
    if not finite(manifest.get("source_duration_seconds")):
        raise ValueError("invalid source duration")
    for key, low, high in (("image_side", 1, 2048), ("jpeg_quality", 1, 100)):
        if type(manifest.get(key)) is not int or not low <= manifest[key] <= high:
            raise ValueError("invalid image settings")
    windows, images = {}, {}
    fingerprints = set()
    for w in manifest["windows"]:
        wid = w["window_id"]
        if not isinstance(wid, str) or not re.fullmatch(r"W[0-9]{3,6}", wid):
            raise ValueError("invalid window id")
        if wid in windows or not 2 <= len(w["frames"]) <= 4:
            raise ValueError("duplicate window or invalid frame count")
        times, hashes, payloads = [], [], []
        for frame in w["frames"]:
            at = frame["media_seconds"]
            if not finite(at) or not 0 <= at < manifest["source_duration_seconds"]:
                raise ValueError("invalid evidence timestamp")
            path = packet_file(packet, frame["path"])
            if path.stat().st_size > 4 * 1024 * 1024:
                raise ValueError("frame exceeds memory limit")
            data = path.read_bytes()
            if sha(data) != frame["sha256"]:
                raise ValueError("evidence image fingerprint mismatch")
            with Image.open(io.BytesIO(data)) as im:
                if im.format != "JPEG" or max(im.size) > manifest["image_side"]:
                    raise ValueError("invalid evidence image format or dimensions")
                im.verify()
            times.append(at)
            hashes.append(frame["sha256"])
            payloads.append(data)
        if (
            any(b <= a for a, b in pairwise(times))
            or not finite(w["judgment_time_seconds"])
            or w["judgment_time_seconds"] != times[-1]
        ):
            raise ValueError("window must be continuous and judged at last frame")
        if tuple(hashes) in fingerprints:
            raise ValueError("deduplicate evidence windows before comparison")
        fingerprints.add(tuple(hashes))
        windows[wid], images[wid] = w, payloads
    references = {}
    for row in reference["rows"]:
        wid = row["window_id"]
        if wid not in windows or wid in references:
            raise ValueError("reference has unknown or duplicate window")
        w = windows[wid]
        reference_times = row.get("evidence_timestamps_seconds")
        if (
            not finite(row["judgment_time_seconds"])
            or not isinstance(reference_times, list)
            or not all(finite(t) for t in reference_times)
            or row["judgment_time_seconds"] != w["judgment_time_seconds"]
            or reference_times != [f["media_seconds"] for f in w["frames"]]
            or set(row["judgments"]) != set(policy["event_labels"])
        ):
            raise ValueError("reference labels or timestamps do not match evidence")
        for verdict in row["judgments"].values():
            if type(verdict["detected"]) not in (bool, type(None)):
                raise ValueError("reference labels must be boolean or null")
            if not isinstance(verdict["reason"], str) or not verdict["reason"].strip():
                raise ValueError("reference requires visible-evidence reasons")
        references[wid] = row
    if (
        set(references) != set(windows)
        or set(reference.get("inspected_window_ids", [])) != set(windows)
        or reference.get("uninspected_window_ids") != []
    ):
        raise ValueError("reference does not cover every inspected window")
    return manifest, policy, reference, images
