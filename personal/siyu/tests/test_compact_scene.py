"""Scene captions on compact VLM verdicts. Not part of the official CI suite."""

from goodvision_harness.grounding import parse_multi_grounded_classification


def test_compact_scene_rides_with_negative_and_alert_verdicts():
    clear = parse_multi_grounded_classification(
        '{"no_events":0.94,"scene":"两人沿走廊走过"}',
        event_labels=("fight", "fall"),
    )
    assert not clear.detected
    assert clear.scene_summary == "两人沿走廊走过"
    alert = parse_multi_grounded_classification(
        '{"alerts":{"smoking":0.88},"scene_summary":"男子站着抽烟"}',
        event_labels=("smoking", "weapon"),
    )
    assert alert.detected
    assert alert.scene_summary == "男子站着抽烟"
