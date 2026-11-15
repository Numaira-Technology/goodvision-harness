"""Versioned native CLIP threshold artifact and wire-score contracts.

Artifact hashes identify externally reviewed evidence; parsing cannot establish
that a validation split was isolated or that a threshold is accurate.
"""

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

MODEL = "openai/clip-vit-large-patch14"
REVISION = "32bd64288804d66eefd0ccbe215aa642df71cc41"
CORE_EVENTS = ("fight", "smoking", "fall", "weapon")
CALIBRATED_PROMPTS = {
    "fight": (
        "Two people are physically fighting, punching or kicking each other.",
        "People are walking, standing or talking peacefully.",
    ),
    "smoking": (
        "A person is smoking a cigarette.",
        "A person is eating food or drinking from a cup.",
    ),
    "fall": (
        "A person is losing balance and falling to the ground.",
        "A person is standing, walking or sitting steadily.",
    ),
    "weapon": (
        "A person is holding a gun or a knife.",
        "A person is holding an ordinary object such as a phone.",
    ),
}
PREPROCESSING = "rgb-clip-checkpoint-processor-v1"
_MAX_ARTIFACT_BYTES = 64 * 1024


@dataclass(frozen=True)
class FrozenThresholds:
    values: Mapping[str, float | None]
    sha256: str
    sampling: Mapping[str, float | int | str]


def sampling_contract(config):
    return {
        "policy": "two-frame-target-spacing-v1",
        "candidate_fps": config.candidate_fps,
        "observation_window_frames": config.observation_window_frames,
        "observation_frame_delta_seconds": config.observation_frame_delta_seconds,
        "observation_fps": config.observation_fps,
    }


def valid_sampling(value):
    fixed = {
        "policy": "two-frame-target-spacing-v1",
        "candidate_fps": 4.0,
        "observation_window_frames": 2,
        "observation_frame_delta_seconds": 0.5,
    }
    return (
        isinstance(value, Mapping)
        and set(value) == set(fixed) | {"observation_fps"}
        and all(value[key] == expected for key, expected in fixed.items())
        and type(value["observation_window_frames"]) is int
        and all(
            type(value[key]) in (int, float)
            for key in ("candidate_fps", "observation_frame_delta_seconds")
        )
        and type(value["observation_fps"]) in (int, float)
        and 0 < value["observation_fps"] <= 3
        and math.isfinite(value["observation_fps"])
    )


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate threshold artifact key")
        result[key] = value
    return result


def _reject_constant(_value):
    raise ValueError("nonfinite threshold artifact number")


def _hex(value, length):
    return isinstance(value, str) and re.fullmatch(f"[0-9a-f]{{{length}}}", value)


def valid_margin(value):
    # Check bounds before isfinite so even unbounded JSON integers fail cleanly.
    return type(value) in (int, float) and -2 <= value <= 2 and math.isfinite(value)


def load_thresholds(path):
    """Reject ambiguous or incompatible artifacts before importing model libraries."""
    with open(path, "rb") as source:
        raw = source.read(_MAX_ARTIFACT_BYTES + 1)
    if len(raw) > _MAX_ARTIFACT_BYTES:
        raise ValueError("threshold artifact exceeds 64 KiB")
    artifact = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_object,
        parse_constant=_reject_constant,
    )
    expected = {
        "schema_version": 2,
        "model": MODEL,
        "revision": REVISION,
        "prompts": {key: list(value) for key, value in CALIBRATED_PROMPTS.items()},
        "preprocessing": PREPROCESSING,
        "aggregation": "max_all_images",
        "comparison": ">",
    }
    if (
        not isinstance(artifact, dict)
        or set(artifact) != set(expected) | {"thresholds", "provenance", "sampling"}
        or type(artifact["schema_version"]) is not int
        or any(artifact[key] != value for key, value in expected.items())
    ):
        raise ValueError("incompatible threshold artifact contract")
    if not valid_sampling(artifact["sampling"]):
        raise ValueError("incompatible threshold sampling contract")
    values = artifact["thresholds"]
    if (
        not isinstance(values, dict)
        or set(values) != set(CORE_EVENTS)
        or any(
            value is not None and not valid_margin(value) for value in values.values()
        )
    ):
        raise ValueError(
            "provide four finite cosine-margin thresholds in [-2, 2] or null"
        )
    provenance = artifact["provenance"]
    hashes = {
        "validation_manifest_sha256",
        "validation_scores_sha256",
        "selection_protocol_sha256",
    }
    if (
        not isinstance(provenance, dict)
        or set(provenance) != hashes | {"source_commit"}
        or any(not _hex(provenance[key], 64) for key in hashes)
        or not _hex(provenance["source_commit"], 40)
    ):
        raise ValueError("invalid threshold validation provenance")
    return FrozenThresholds(
        MappingProxyType(values),
        hashlib.sha256(raw).hexdigest(),
        MappingProxyType(artifact["sampling"]),
    )


def calibrated_choice(row, label, frame_count=None, *, sampling=None):
    """Validate complete window evidence and return the threshold choice index."""
    if (
        label not in CORE_EVENTS
        or row.get("score_kind") != "cosine_margin"
        or "threshold" not in row
        or not _hex(row.get("thresholds_sha256"), 64)
    ):
        raise ValueError("invalid calibrated score contract")
    if sampling is not None and (
        not valid_sampling(row.get("sampling"))
        or row["sampling"] != sampling
    ):
        raise ValueError("incompatible calibrated sampling contract")
    score, threshold = row.get("score"), row["threshold"]
    values, winner = row.get("frame_scores"), row.get("image_index")
    if (
        not valid_margin(score)
        or (threshold is not None and not valid_margin(threshold))
        or not isinstance(values, list)
        or not 1 <= len(values) <= 4
        or (frame_count is not None and len(values) != frame_count)
        or any(not valid_margin(value) for value in values)
        or type(winner) is not int
        or not 0 <= winner < len(values)
        or winner != max(range(len(values)), key=values.__getitem__)
        or score != values[winner]
    ):
        raise ValueError("invalid calibrated window margins")
    probabilities = row.get("probabilities")
    if (
        not isinstance(probabilities, list)
        or len(probabilities) != 3
        or any(
            type(value) not in (int, float) or not 0 <= value <= 1
            for value in probabilities
        )
        or probabilities[2] != 0
        or abs(sum(probabilities) - 1) > 1e-5
    ):
        raise ValueError("invalid calibrated pair probabilities")
    return 2 if threshold is None else 0 if score > threshold else 1
