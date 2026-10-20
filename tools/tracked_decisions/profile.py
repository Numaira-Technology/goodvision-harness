"""Same load and policy as tracked Qwen; Decisions changes the wire protocol."""

from tools.decisions_benchmark import profile as base

VERSION = "yolo26s-decisions-v3-isolated"
MAX_PEOPLE = 38  # Five person classes + six scene/unassigned questions <= 200.
FROZEN = {
    **base.FROZEN,
    "modes": ["events", "compact", "parallel"],
    "parallel_scheduling": "paired",
    "grounding_format": "lean-v1",
    "person_ids_only": True,
    "backend_person_tracks": True,
    "monitor": True,
    "decisions_person_tracking": True,
    "rate_limit_cooldown": 0,  # Fixed 60s between rounds; no window retries.
}
