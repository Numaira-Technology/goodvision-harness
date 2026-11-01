"""Public presentation adapter for the detector's supported event presets.

The detector owns membership and sensitivity rules. The console consumes this
versioned contract and never decides which detectors are installed.
"""

from __future__ import annotations

from copy import deepcopy

from . import grounding
from .event_sensitivity import sensitivity_ui_info

# Optional presentation metadata. A newly registered detector still appears
# using its stable ID when no display metadata has been supplied yet.
EVENT_PRESENTATION = {
    "fight": {"labels": {"zh": "打架", "en": "Fight"}, "aliases": ["fighting"]},
    "smoking": {"labels": {"zh": "吸烟", "en": "Smoking"}, "aliases": ["smoke"]},
    "fall": {"labels": {"zh": "跌倒", "en": "Fall"}, "aliases": ["falling"]},
    "weapon": {
        "labels": {"zh": "持械", "en": "Weapon"},
        "aliases": ["weapon_possession", "holding_weapon"],
    },
    "ppe_violation": {
        "labels": {"zh": "防护装备缺失", "en": "Missing protective equipment"}
    },
    "ground_litter": {
        "labels": {"zh": "地面垃圾", "en": "Ground litter"},
        "scope": "scene",
    },
    "stranger": {
        "labels": {"zh": "陌生人", "en": "Stranger"},
        "kind": "identity",
        "aliases": ["unknown_person"],
    },
}


def event_catalog() -> dict:
    """Build fresh values from the authoritative supported event list."""
    sensitivities = sensitivity_ui_info()
    events = []
    for key in grounding.DETECTION_EVENT_LABELS:
        metadata = EVENT_PRESENTATION.get(key, {})
        item = {
            "id": key,
            "labels": dict(metadata.get("labels", {"zh": key, "en": key})),
            "aliases": list(metadata.get("aliases", [])),
            "scope": metadata.get("scope", "person"),
            "kind": metadata.get("kind", "detection"),
            "default_enabled": metadata.get("default_enabled", True),
        }
        options = sensitivities["event_sensitivity_options"].get(key)
        if options:
            item["sensitivity"] = {
                "default": sensitivities["event_sensitivity_defaults"][key],
                "options": deepcopy(options),
            }
        events.append(item)
    return {"version": 1, "events": events}
