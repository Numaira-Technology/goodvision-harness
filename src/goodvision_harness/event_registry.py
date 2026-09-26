"""Single source of truth for which detection events exist and what they mean.

Each built-in event keeps the exact label, order, Chinese name, sensitivity
policy and grounding target the pipeline used before this registry existed.
Modules that previously repeated one of those tuples or frozensets now derive
them from here, so a later change can extend the vocabulary in one place
instead of in every validator and prompt builder.

This module must not import other harness modules: the validators that consume
it live near the bottom of the import graph.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

# Which detector branch owns the event. Safety events are judged together by
# the multi-event visual request; the face branch runs its own client.
SAFETY_BRANCH = "safety"
FACE_BRANCH = "face"

# What a positive verdict must put a box around. "any" places no constraint,
# which is what overlay fusion already did for labels it did not recognize.
PERSON_TARGET = "person"
OBJECT_TARGET = "object"
ANY_TARGET = "any"

SENSITIVITY_LEVELS = ("standard", "high")

# The one face-branch event, named here because several modules reference it
# directly rather than iterating the branch.
STRANGER_LABEL = "stranger"


@dataclass(frozen=True, slots=True)
class SensitivityOption:
    """One console-selectable evidence policy for an event."""

    value: str
    label: str
    description: str

    def as_json(self) -> dict[str, str]:
        return {
            "value": self.value,
            "label": self.label,
            "description": self.description,
        }


@dataclass(frozen=True, slots=True)
class EventDefinition:
    """Everything the pipeline needs to know about one detectable event."""

    label: str
    zh_name: str
    branch: str = SAFETY_BRANCH
    target: str = PERSON_TARGET
    requires_subject: bool = True
    default_sensitivity: str | None = None
    sensitivity_options: tuple[SensitivityOption, ...] = ()
    # Alternate labels other subsystems use for the same event, for display
    # lookups only. They are not detectable labels in their own right.
    aliases: tuple[str, ...] = ()
    builtin: bool = False

    def __post_init__(self) -> None:
        if not self.label or self.label != self.label.strip().casefold():
            raise ValueError("event label must be nonempty lowercase text")
        if not self.zh_name:
            raise ValueError("event definition requires a display name")
        if self.branch not in {SAFETY_BRANCH, FACE_BRANCH}:
            raise ValueError("event branch must be safety or face")
        if self.target not in {PERSON_TARGET, OBJECT_TARGET, ANY_TARGET}:
            raise ValueError("event target must be person, object or any")
        if self.default_sensitivity is not None and (
            self.default_sensitivity not in SENSITIVITY_LEVELS
        ):
            raise ValueError("default sensitivity must be standard or high")
        if bool(self.sensitivity_options) != (self.default_sensitivity is not None):
            raise ValueError(
                "sensitivity options and a default must be supplied together"
            )


_STANDARD = "standard"
_HIGH = "high"


def _levels(standard: str, high: str) -> tuple[SensitivityOption, ...]:
    return (
        SensitivityOption(_STANDARD, "标准敏感度", standard),
        SensitivityOption(_HIGH, "高敏感度", high),
    )


BUILTIN_EVENTS: tuple[EventDefinition, ...] = (
    EventDefinition(
        label="fight",
        zh_name="打架",
        target=PERSON_TARGET,
        default_sensitivity=_STANDARD,
        sensitivity_options=_levels(
            "明确攻击动作或即将发生的攻击",
            "也关注早期冲突和部分遮挡的攻击动作",
        ),
        builtin=True,
    ),
    EventDefinition(
        label="smoking",
        zh_name="吸烟",
        target=OBJECT_TARGET,
        default_sensitivity=_HIGH,
        sensitivity_options=_levels(
            "必须明确识别烟具",
            "允许结合动作和上下文判断，无需认清烟具",  # noqa: RUF001
        ),
        builtin=True,
    ),
    EventDefinition(
        label="fall",
        zh_name="跌倒",
        target=PERSON_TARGET,
        default_sensitivity=_STANDARD,
        sensitivity_options=_levels(
            "需要可见的失衡、失去支撑或倒下证据",
            "也关注姿态不稳和疑似即将跌倒的迹象",
        ),
        builtin=True,
    ),
    EventDefinition(
        label="weapon",
        zh_name="武器",
        target=OBJECT_TARGET,
        default_sensitivity=_STANDARD,
        sensitivity_options=_levels(
            "可见武器或被持有、使用的疑似武器",
            "也关注结合持握或威胁动作的部分遮挡物品",
        ),
        builtin=True,
    ),
    EventDefinition(
        label=STRANGER_LABEL,
        zh_name="陌生人",
        branch=FACE_BRANCH,
        target=PERSON_TARGET,
        aliases=("unknown_person",),
        builtin=True,
    ),
)

def labels(definitions: Iterable[EventDefinition]) -> tuple[str, ...]:
    """Return the labels of the supplied definitions, in their given order."""

    return tuple(definition.label for definition in definitions)


def branch_events(
    branch: str, definitions: Iterable[EventDefinition] = BUILTIN_EVENTS
) -> tuple[EventDefinition, ...]:
    """Return the definitions owned by one detector branch, in order."""

    return tuple(
        definition for definition in definitions if definition.branch == branch
    )


def display_names(
    definitions: Iterable[EventDefinition] = BUILTIN_EVENTS,
) -> dict[str, str]:
    """Map every label and alias to its Chinese display name."""

    names: dict[str, str] = {}
    for definition in definitions:
        names[definition.label] = definition.zh_name
        for alias in definition.aliases:
            names[alias] = definition.zh_name
    return names


def sensitivity_defaults(
    definitions: Iterable[EventDefinition] = BUILTIN_EVENTS,
) -> dict[str, str]:
    """Map each event that has an evidence policy to its default level."""

    return {
        definition.label: definition.default_sensitivity
        for definition in definitions
        if definition.default_sensitivity is not None
    }


def sensitivity_options(
    definitions: Iterable[EventDefinition] = BUILTIN_EVENTS,
) -> dict[str, list[dict[str, str]]]:
    """Render the console's selectable evidence policies as plain JSON values."""

    return {
        definition.label: [
            option.as_json() for option in definition.sensitivity_options
        ]
        for definition in definitions
        if definition.sensitivity_options
    }


def targets(
    target: str, definitions: Iterable[EventDefinition] = BUILTIN_EVENTS
) -> frozenset[str]:
    """Return the labels whose positive boxes must be of one target kind."""

    return frozenset(
        definition.label
        for definition in definitions
        if definition.target == target
    )
