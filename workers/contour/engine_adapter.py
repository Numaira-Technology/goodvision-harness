"""Private integration boundary for the existing, independently licensed worker.

Public branding is an API projection, not an assertion of model authorship.
Keep implementation identifiers here and retain upstream license files.
"""

from __future__ import annotations

import copy
import hashlib

from .contracts import SessionAppended, SessionCreated, SessionDeleted

HEALTH_PATH = "/health/perception/sam_live"
SESSION_PATH = "/v1/perception/sam_live/sessions"
BACKEND = "sam3.1_video"
PUBLIC_BACKEND = "goodvision_contour"


def lane_paths(index: int, capacity: int) -> tuple[str, str]:
    if capacity == 1:
        return SESSION_PATH, HEALTH_PATH
    camera = f"camera-{index:02d}"
    return (
        f"/v1/cameras/{camera}/perception/sam_live/sessions",
        f"/health/cameras/{camera}/perception/sam_live",
    )


def project_response(operation: str, payload: dict) -> dict:
    if payload.get("backend") != BACKEND:
        raise ValueError("Unexpected engine contract")
    result = copy.deepcopy(payload)
    result["backend"] = PUBLIC_BACKEND
    if operation == "append":
        metadata = result["route_metadata"]
        if metadata["route"] != "sam_live":
            raise ValueError("Unexpected route contract")
        metadata["route"] = "contour"
        for frame in result["frames"]:
            for item in frame["objects"]:
                native = item["attributes"]["native_track"]
                if native["backend"] != BACKEND:
                    raise ValueError("Unexpected object contract")
                native["backend"] = PUBLIC_BACKEND
                # Stable across retries/restarts; includes epoch, never joins people
                # across independent model sessions. Caller identifiers stay intact.
                identity = f"{native['epoch']}:{native['object_id']}".encode()
                item["track_id"] = "ct-" + hashlib.sha256(identity).hexdigest()[:32]
    contract = {
        "create": SessionCreated,
        "append": SessionAppended,
        "delete": SessionDeleted,
    }[operation]
    projected = contract.model_validate(result).model_dump(exclude_none=True)
    if operation == "append":
        # Null watermarks are part of the wire contract, unlike optional contours.
        for field in ("emitted_frame_sequence", "emitted_timestamp_seconds"):
            projected[field] = result[field]
    return projected
