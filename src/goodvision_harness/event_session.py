"""本批 Ashley 原生事件词表；进程内冻结，不能在会话中途换定义。"""
from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

from .event_registry import (
    EventVocabulary, SAFETY_BRANCH, branch_events, definition_from_json, labels,
)

_VEHICLE_SCENE_LABELS = frozenset({
    "vehicle_stops", "vehicle_turns_right", "vehicle_makes_u_turn",
    "vehicle_drops_off_person", "vehicle_turns_left", "vehicle_starts", "vehicle_reverses",
    "vehicle_picks_up_person",
})
LEGACY_FOUR_SAFETY_LABELS = ("fight", "smoking", "fall", "weapon")
BUILTIN_SAFETY_LABELS = labels(branch_events(SAFETY_BRANCH))
BUILTIN_DETECTION_LABELS = (*BUILTIN_SAFETY_LABELS, "stranger")


def _load_vocabulary():
    path = os.environ.get("ASHLEY_EVENT_DEFINITIONS")
    if not path:
        raise RuntimeError("ASHLEY_EVENT_DEFINITIONS must name the frozen public definitions")
    raw = json.loads(Path(path).read_text())
    rows = raw.get("definitions") if isinstance(raw, dict) else raw
    # 独立快照的51项固定配置由正式manifest冻结；API不可信注入的32项
    # 上限不变，每一项仍经同一字段校验及重复/内建标签保护。
    if not isinstance(rows, list) or not rows or len(rows) > 51:
        raise ValueError("frozen Ashley definitions must contain 1..51 entries")
    vocabulary = EventVocabulary.extend(definition_from_json(row) for row in rows)
    return EventVocabulary(tuple(
        replace(row, requires_subject=False)
        if row.label in _VEHICLE_SCENE_LABELS else row
        for row in vocabulary.definitions
    ))


VOCABULARY = _load_vocabulary()
SAFETY_LABELS = labels(VOCABULARY.branch(SAFETY_BRANCH))
DETECTION_LABELS = VOCABULARY.labels()
SCENE_EVENT_LABELS = frozenset(
    row.label for row in VOCABULARY.branch(SAFETY_BRANCH)
    if not row.requires_subject
)


def requires_subject(label):
    definition = VOCABULARY.get(label)
    return True if definition is None else definition.requires_subject


def valid_subject_binding(label, subject_ids):
    return bool(subject_ids) if requires_subject(label) else not subject_ids


def prompt_definitions(requested):
    selected = set(requested)
    return {
        definition.label: {
            "zh_name": definition.zh_name,
            "definition": definition.definition,
            "not_this_event": definition.counter_examples,
            "target": definition.target,
            "target_hint": definition.target_hint,
            "requires_subject": definition.requires_subject,
        }
        for definition in VOCABULARY.injected() if definition.label in selected
    }
