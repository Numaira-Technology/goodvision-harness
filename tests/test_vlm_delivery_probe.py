"""The external receiver must not mistake a pending positive for a negative."""

from tools.vlm_delivery_probe import observation_alerts


def report():
    return {
        "type": "vlm_report", "state": "completed", "message_id": "m1",
        "stream_session_id": "run1", "frame": {"timestamp_seconds": 12.7},
        "judgments": [
            {"event_label": "fall", "observed_detected": True, "detected": None,
             "confirmation_state": "pending", "localization_state": "unavailable"},
            {"event_label": "fight", "observed_detected": False, "detected": False},
            {"event_label": "weapon", "observed_detected": None, "detected": None},
        ],
    }


def test_pending_unlocalized_positive_reaches_receiver_without_confirmation():
    alerts = observation_alerts(report())
    assert len(alerts) == 1
    assert alerts[0]["event_label"] == "fall"
    assert alerts[0]["confirmation_state"] == "pending"
    assert alerts[0]["localization_state"] == "unavailable"
    assert alerts[0]["frame"]["timestamp_seconds"] == 12.7
    assert "detected" not in alerts[0]


def test_failure_and_scene_state_do_not_become_raw_vlm_alerts():
    payload = report()
    payload["state"] = "failed"
    assert observation_alerts(payload) == []
    payload = report()
    payload["type"] = "analysis_result"
    payload["judgments"][0]["detected"] = True
    assert observation_alerts(payload) == []
