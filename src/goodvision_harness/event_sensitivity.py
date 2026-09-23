"""Per-session safety-event evidence policies exposed by the console."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any

EVENT_SENSITIVITY_DEFAULTS = {
    "fight": "standard",
    "smoking": "high",
    "fall": "standard",
    "weapon": "standard",
}
EVENT_SENSITIVITY_OPTIONS = {
    "fight": [
        {
            "value": "standard",
            "label": "标准敏感度",
            "description": "明确攻击动作或即将发生的攻击",
        },
        {
            "value": "high",
            "label": "高敏感度",
            "description": "也关注早期冲突和部分遮挡的攻击动作",
        },
    ],
    "smoking": [
        {
            "value": "standard",
            "label": "标准敏感度",
            "description": "必须明确识别烟具",
        },
        {
            "value": "high",
            "label": "高敏感度",
            "description": "允许结合动作和上下文判断，无需认清烟具",  # noqa: RUF001
        },
    ],
    "fall": [
        {
            "value": "standard",
            "label": "标准敏感度",
            "description": "需要可见的失衡、失去支撑或倒下证据",
        },
        {
            "value": "high",
            "label": "高敏感度",
            "description": "也关注姿态不稳和疑似即将跌倒的迹象",
        },
    ],
    "weapon": [
        {
            "value": "standard",
            "label": "标准敏感度",
            "description": "有可辨认武器特征，或明确被用于攻击的物品",  # noqa: RUF001
        },
        {
            "value": "high",
            "label": "高敏感度",
            "description": "也关注结合持握或威胁动作的部分遮挡物品",
        },
    ],
}


def normalize_event_sensitivities(
    value: object,
    event_labels: Collection[str],
) -> dict[str, str]:
    """Validate the complete request, then snapshot only active safety events."""

    if value is not None and not isinstance(value, Mapping):
        raise ValueError("event_sensitivities must be an object")
    supplied = {} if value is None else value
    for label, level in supplied.items():
        if label not in EVENT_SENSITIVITY_DEFAULTS:
            raise ValueError("event_sensitivities contains an unsupported event")
        if not isinstance(level, str) or level not in {"standard", "high"}:
            raise ValueError("event sensitivity must be standard or high")
    return {
        label: supplied.get(label, default)
        for label, default in EVENT_SENSITIVITY_DEFAULTS.items()
        if label in event_labels
    }


SENSITIVITY_SYSTEM_PROMPT = (
    "You are a live-camera safety-event perception module. Independently judge "
    "each requested event using the per-event sensitivity policy supplied below. "
    "Use visible evidence in the NEWEST image, with earlier supplied images only "
    "as temporal context. Do not invent motion from a single frame, hidden "
    "objects, people, smoke, or unseen details. Report uncertainty honestly in "
    "confidence and description. Return only the specified JSON object."
)

_SMOKING_HIGH_CRITERIA = (
    "Smoking sensitivity HIGH: an identifiable smoking item is NOT required. "
    "Set detected=true when visible evidence plausibly indicates smoking, "
    "including a smoking-like hand-to-mouth or puffing gesture, a small or "
    "partly obscured object near the mouth, or visible handling/posture/context "
    "consistent with smoking. Small size, blur, or partial occlusion alone must "
    "not force a negative. "
)
_SMOKING_HIGH_LIMITS = (
    "do not claim an identifiable cigarette, smoke, "
    "or a distinguishing feature that is not visible. A clearly unrelated "
    "activity such as eating, drinking, or wiping the face remains negative."
)
SMOKING_HIGH_RULE = (
    _SMOKING_HIGH_CRITERIA
    + "Lower confidence when uncertain and describe a suspected smoking action; "
    + _SMOKING_HIGH_LIMITS
    + " "
    "For a positive, box the actual smoking item when it is visible; otherwise "
    "box the visible hand/mouth action region (label smoking_action) or the "
    "involved visible person (label person). Never fabricate a cigarette box "
    "on fingers or a concealed object."
)

_WEAPON_STANDARD_CRITERIA = (
    "Weapon sensitivity STANDARD: require a visible firearm, knife, machete, "
    "or other weapon with distinguishable weapon-specific features, such as "
    "a blade with its handle or a firearm's barrel and grip. A recognizable "
    "weapon does not require an attack to be positive. An ordinary or "
    "unidentified object counts as an improvised weapon only when the visible "
    "object itself is being used to strike, stab, or explicitly threaten a "
    "person; mere possession or movement is insufficient. "
    "Small size, dark color, an elongated shape, a closed fist, or a gripping "
    "pose alone does not identify a weapon. Independently assess fight and "
    "weapon: punching or raised fists must not turn an unidentified handheld "
    "item into a weapon. Carrying, placing, dropping, picking up, or arranging "
    "ordinary objects without weapon-specific features or directed weapon use "
    "is negative. Do not infer a blade, barrel, or hidden weapon from context."
)
_WEAPON_HIGH_CRITERIA = (
    "Weapon sensitivity HIGH: also flag a small or partly obscured "
    "weapon-like item when a visible weapon-specific feature plus compatible "
    "holding or threat cues supports that interpretation, without requiring "
    "full item identification. Ordinary objects count only with visible "
    "directed use as a weapon, not from elongated shape or possession alone. "
)
_WEAPON_HIGH_LIMITS = (
    "Bare-hand gestures without a visible item and clearly unrelated objects "
    "are insufficient. Punching while holding an unidentified small object "
    "does not by itself establish a weapon. Ordinary carrying, placing, "
    "dropping, or picking up objects without weapon features or directed "
    "weapon use remains negative."
)

_EVIDENCE_POLICIES = {
    ("fight", "standard"): (
        "Fighting sensitivity STANDARD: flag striking, kicking, pushing, "
        "grabbing, wrestling, aggressive lunging/chasing, raised fists, an "
        "attack posture, or an apparently imminent attack. Impact is not required."
    ),
    ("fight", "high"): (
        "Fighting sensitivity HIGH: flag plausible early confrontation, "
        "threatening approach, forceful interaction, or partly obscured attack "
        "gestures as well as active fighting. Complete contact or certainty is "
        "not required; lower confidence for ambiguous aggression. Mere proximity "
        "or clearly friendly/ordinary interaction is insufficient."
    ),
    ("fall", "standard"): (
        "Falling sensitivity STANDARD: require visible loss of balance/support, "
        "uncontrolled descent, or collapse. A stable supported squat, crouch, "
        "kneel, bend, sit, or lie is negative. Low body height or being on the "
        "ground alone is insufficient. Full impact is not required."
    ),
    ("fall", "high"): (
        "Falling sensitivity HIGH: also flag a plausible imminent fall indicated "
        "by an unstable posture, an apparent attempt to catch balance, partial "
        "collapse, or loss-of-support cues, even if descent is not conclusive. "
        "Use lower confidence for ambiguity. A clearly stable supported crouch, "
        "sit, lie, or deliberate bend alone remains negative."
    ),
    ("smoking", "high"): (
        _SMOKING_HIGH_CRITERIA
        + "Lower confidence when uncertain; "
        + _SMOKING_HIGH_LIMITS
    ),
    ("weapon", "standard"): _WEAPON_STANDARD_CRITERIA,
    ("weapon", "high"): (
        _WEAPON_HIGH_CRITERIA
        + "Lower confidence if uncertain. "
        + _WEAPON_HIGH_LIMITS
        + " Never invent a hidden weapon."
    ),
}

# Grounding still needs descriptions and object boxes. Temporal verdicts use
# the same evidence criteria but have neither field in their output contract.
_GROUNDING_POLICIES = {
    ("smoking", "high"): SMOKING_HIGH_RULE,
    ("weapon", "standard"): _WEAPON_STANDARD_CRITERIA + " Box the actual item.",
    ("weapon", "high"): (
        _WEAPON_HIGH_CRITERIA
        + "Report suspected weapon and lower confidence if uncertain. "
        + _WEAPON_HIGH_LIMITS
        + " Box only the visible item, never an invented hidden weapon."
    ),
}


def sensitivity_policy_text(
    sensitivities: Mapping[str, str],
    *,
    smoking_standard_rule: str,
    include_output_guidance: bool = True,
) -> str:
    """Render evidence criteria, optionally with grounding output instructions.

    The caller supplies its standard-smoking policy; when requesting evidence
    only, that supplied policy must also omit output instructions.
    """
    rules = []
    for label, level in sensitivities.items():
        if label == "smoking" and level == "standard":
            rules.append(smoking_standard_rule)
        else:
            key = (label, level)
            rule = _EVIDENCE_POLICIES[key]
            if include_output_guidance:
                rule = _GROUNDING_POLICIES.get(key, rule)
            rules.append(rule)
    return "\n\n".join(rules)


def sensitivity_ui_info() -> dict[str, Any]:
    """Return independent JSON values so callers cannot change the defaults."""

    return {
        "event_sensitivity_defaults": dict(EVENT_SENSITIVITY_DEFAULTS),
        "event_sensitivity_options": {
            label: [dict(option) for option in options]
            for label, options in EVENT_SENSITIVITY_OPTIONS.items()
        },
    }
