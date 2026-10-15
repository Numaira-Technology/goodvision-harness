"""固定评测词表；观察配置不能用题目或选项筛选此表。"""
from __future__ import annotations

import re

VOCAB_VERSION = "frozen-ashley-51-plus-native-fight-v1"
VOCAB_SOURCE = "detector-ablation182-resume6-20261013/source/event-definitions.json"

# 原冻结文件的51个标签逐项保留；同义词也不在此处合并。
FROZEN_EVENT_LABELS = (
    "browsing", "leaving_baggage", "drop_down", "shop_exit", "hugging",
    "leaving_object", "shaking_hands_with_somebody", "browsing_newspaper",
    "shaking_hands_with_someone", "shop_enter", "running", "lie_down",
    "meeting", "walking", "immobile", "drinking_coffee", "vehicle_stops",
    "person_picks_up_object", "person_opens_vehicle_door", "person_exits_vehicle",
    "person_talks_on_phone", "person_exits_scene_through_structure",
    "person_enters_scene_through_structure", "person_closes_facility_door",
    "vehicle_turns_right", "person_carries_heavy_object", "person_stands_up",
    "vehicle_makes_u_turn", "person_embraces_person", "person_closes_door",
    "vehicle_drops_off_person", "vehicle_turns_left", "vehicle_starts",
    "person_leaves_shop_window", "window_shopping", "vehicle_reverses",
    "person_talks_to_person", "person_texts_on_phone", "hand_interacts_with_person",
    "person_closes_trunk", "person_closes_vehicle_door", "person_loads_vehicle",
    "person_opens_facility_door", "person_opens_trunk", "person_purchases",
    "person_puts_down_object", "person_reads_document", "person_rides_bicycle",
    "person_sits_down", "person_talks_to_phone", "vehicle_picks_up_person",
)
OBSERVATION_EVENT_LABELS = FROZEN_EVENT_LABELS + ("fight",)
SUPPORTED_EVENT_LABELS = frozenset(OBSERVATION_EVENT_LABELS)


def canonical_event_label(value: str) -> str:
    """只做公开标签的空白/下划线规范化及明确的physical_conflict映射。"""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("event label must be non-empty text")
    label = re.sub(r"[\s_]+", "_", value.strip().lower())
    return "fight" if label == "physical_conflict" else label
