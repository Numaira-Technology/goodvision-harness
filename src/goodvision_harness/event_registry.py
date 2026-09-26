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

from collections.abc import Iterable, Mapping
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
    # Prompt text. Built-in events leave these empty because their criteria are
    # already written out in the configured safety prompt and the sensitivity
    # policies; an injected event has nowhere else to state what it means.
    definition: str = ""
    counter_examples: str = ""
    target_hint: str = ""

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
        if not self.builtin and not self.definition.strip():
            raise ValueError("an injected event requires a definition")
        if not self.builtin and self.branch != SAFETY_BRANCH:
            raise ValueError("an injected event must use the safety branch")


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


@dataclass(frozen=True, slots=True)
class EventVocabulary:
    """The events one session may request: the built-ins plus its injections.

    A vocabulary is per-session rather than global so two cameras can run
    different injected events without one widening the other's validation.
    """

    definitions: tuple[EventDefinition, ...] = BUILTIN_EVENTS

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for definition in self.definitions:
            if definition.label in seen:
                raise ValueError("event vocabulary contains a duplicate label")
            seen.add(definition.label)

    @classmethod
    def extend(cls, injected: Iterable[EventDefinition]) -> EventVocabulary:
        """Return the built-ins followed by injected events, in supplied order."""

        added = tuple(injected)
        builtin = {definition.label for definition in BUILTIN_EVENTS}
        for definition in added:
            if definition.builtin or definition.label in builtin:
                raise ValueError("an injected event cannot replace a built-in event")
        return cls(BUILTIN_EVENTS + added)

    def __contains__(self, label: object) -> bool:
        return any(definition.label == label for definition in self.definitions)

    def get(self, label: str) -> EventDefinition | None:
        for definition in self.definitions:
            if definition.label == label:
                return definition
        return None

    def labels(self) -> tuple[str, ...]:
        return labels(self.definitions)

    def branch(self, branch: str) -> tuple[EventDefinition, ...]:
        return branch_events(branch, self.definitions)

    def injected(self) -> tuple[EventDefinition, ...]:
        return tuple(
            definition for definition in self.definitions if not definition.builtin
        )

    def default_labels(self) -> tuple[str, ...]:
        """What a session detects when it names no features.

        Declaring an event only makes it selectable. Turning it on is the
        session's call, so the default stays the built-in set no matter how
        large a vocabulary the deployment declared.
        """

        return labels(
            definition for definition in self.definitions if definition.builtin
        )

    def display_names(self) -> dict[str, str]:
        return {
            definition.label: definition.zh_name for definition in self.definitions
        }

    def resolve(self, requested: Iterable[str]) -> tuple[EventDefinition, ...]:
        """Order requested labels as the vocabulary does, rejecting unknown ones.

        Built-ins keep their historical order and injected events follow in the
        order they were supplied, so one request always renders and parses its
        events in the same sequence.
        """

        wanted = tuple(requested)
        if not wanted:
            raise ValueError("at least one event must be requested")
        if len(set(wanted)) != len(wanted):
            raise ValueError("requested events must not contain duplicates")
        unknown = [label for label in wanted if label not in self]
        if unknown:
            raise ValueError("requested events contain an unsupported label")
        return tuple(
            definition
            for definition in self.definitions
            if definition.label in set(wanted)
        )


BUILTIN_VOCABULARY = EventVocabulary()

# Injected text is pasted into a model prompt, so it is bounded and restricted
# to plain descriptive fields. A label must stay a machine identifier because
# it also keys event state, archives and the response contract.
MAX_INJECTED_EVENTS = 32
_MAX_LABEL_LENGTH = 48
_MAX_NAME_LENGTH = 32
_MAX_TEXT_LENGTH = 600
_LABEL_CHARACTERS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_")
_INJECTED_FIELDS = frozenset(
    {
        "label",
        "zh_name",
        "definition",
        "counter_examples",
        "target",
        "target_hint",
    }
)


def _injected_text(value: object, name: str, *, limit: int, required: bool) -> str:
    if value is None and not required:
        return ""
    if not isinstance(value, str):
        raise ValueError(f"event definition {name} must be text")
    text = value.strip()
    if required and not text:
        raise ValueError(f"event definition {name} must not be empty")
    if len(text) > limit:
        raise ValueError(f"event definition {name} is too long")
    # The text becomes one bullet in the prompt, so it may not introduce its
    # own lines and pose as a separate instruction.
    collapsed = " ".join(text.split())
    if not collapsed.isprintable():
        raise ValueError(f"event definition {name} contains control characters")
    return collapsed


def definition_from_json(payload: object) -> EventDefinition:
    """Validate one untrusted event definition from config or a request."""

    if not isinstance(payload, Mapping):
        raise ValueError("each event definition must be an object")
    unexpected = set(payload) - _INJECTED_FIELDS
    if unexpected:
        raise ValueError("event definition contains an unsupported field")
    label = _injected_text(
        payload.get("label"), "label", limit=_MAX_LABEL_LENGTH, required=True
    ).casefold()
    if not set(label) <= _LABEL_CHARACTERS:
        raise ValueError("event label must use lowercase letters, digits or _")
    target = payload.get("target", PERSON_TARGET)
    if target not in {PERSON_TARGET, OBJECT_TARGET, ANY_TARGET}:
        raise ValueError("event target must be person, object or any")
    return EventDefinition(
        label=label,
        zh_name=_injected_text(
            payload.get("zh_name"), "zh_name", limit=_MAX_NAME_LENGTH, required=True
        ),
        target=target,
        definition=_injected_text(
            payload.get("definition"),
            "definition",
            limit=_MAX_TEXT_LENGTH,
            required=True,
        ),
        counter_examples=_injected_text(
            payload.get("counter_examples"),
            "counter_examples",
            limit=_MAX_TEXT_LENGTH,
            required=False,
        ),
        target_hint=_injected_text(
            payload.get("target_hint"),
            "target_hint",
            limit=_MAX_NAME_LENGTH * 4,
            required=False,
        ),
    )


def vocabulary_from_json(payload: object) -> EventVocabulary:
    """Build a session vocabulary from untrusted injected definitions."""

    if payload is None:
        return BUILTIN_VOCABULARY
    if not isinstance(payload, (list, tuple)):
        raise ValueError("event_definitions must be an array")
    if not payload:
        return BUILTIN_VOCABULARY
    if len(payload) > MAX_INJECTED_EVENTS:
        raise ValueError("too many injected event definitions")
    return EventVocabulary.extend(
        definition_from_json(row) for row in payload
    )
