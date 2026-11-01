"""Shared offline evidence alignment and quality metrics."""

import hashlib
from collections import Counter
from pathlib import Path

from tools.vlm_benchmark.behavior_reference import load_packet
from tools.vlm_multistream_benchmark import decode


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ratio(n, d):
    return n / d if d else None


def safe_path(root, relative):
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError("report path escapes its run")
    return path


def load_reference(packet, video):
    """Verify policy/provenance and original JPEGs; no labels reach the runner."""
    packet = Path(packet)
    manifest, policy, reference, _ = load_packet(packet, packet / "annotations.json")
    if sha(video) != manifest["source_video_sha256"]:
        raise ValueError("video differs from Astra source")
    if (manifest["sampling_fps"], manifest["image_side"], manifest["jpeg_quality"]) != (
        2.5,
        720,
        90,
    ):
        raise ValueError("reference image settings differ from frozen experiment")
    frames = decode(video, 2.5, 720, 90)
    fingerprints = [
        {"sha256": hashlib.sha256(data).hexdigest(), "media_seconds": at}
        for data, at in zip(frames, frames.media_seconds, strict=True)
    ]
    positions = {
        (f["sha256"], f["media_seconds"]): i for i, f in enumerate(fingerprints)
    }
    lookup = {}
    for window in manifest["windows"]:
        try:
            ticks = tuple(
                positions[(f["sha256"], f["media_seconds"])] for f in window["frames"]
            )
        except KeyError as exc:
            raise ValueError("decoded evidence differs from Astra JPEGs/PTS") from exc
        if ticks in lookup:
            raise ValueError("duplicate reference evidence sequence")
        lookup[ticks] = window["window_id"]
    if len(lookup) != 69:
        raise ValueError("fixed experiment requires the original 69 windows")
    return {
        "packet": manifest,
        "policy": policy,
        "reference": reference,
        "reference_sha256": sha(packet / "annotations.json"),
        "decoded_evidence": fingerprints,
        "lookup": lookup,
        "refs": {r["window_id"]: r for r in reference["rows"]},
    }


def quality(rows):
    counts = Counter(row["outcome"] for row in rows)
    for key in ("tp", "tn", "fp", "fn", "model_uncertain", "reference_uncertain"):
        counts[key] += 0
    decided = sum(counts[k] for k in ("tp", "tn", "fp", "fn"))
    return {
        **counts,
        "aligned_complete_judgments": len(rows),
        "binary_comparable": decided,
        "agree": counts["tp"] + counts["tn"],
        "agreement": ratio(counts["tp"] + counts["tn"], decided),
        "decided_coverage_of_reference_certain": ratio(
            decided, decided + counts["model_uncertain"]
        ),
    }
