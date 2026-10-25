"""Offline selection validation and asynchronous preview lifecycle regressions."""

import base64
import copy
import json
from types import SimpleNamespace

import pytest
import test_vlm_detector_preview as detector

from tools.vlm_benchmark import monitor as monitor_module

detector_monitor = detector.monitor
worker = detector.worker
preview = detector.preview
report = detector.report


@pytest.fixture
def monitor(detector_monitor, monkeypatch):
    detector_monitor._select_events = True
    detector_monitor._select_key = "offline-test-only"
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(
        monitor_module, "time", SimpleNamespace(monotonic=lambda: clock.now)
    )
    detector_monitor.clock = clock
    return detector_monitor


@pytest.fixture
def selections(monkeypatch):
    pending = []

    class QueuedThread:
        def __init__(self, target, args, **_):
            self.target, self.args = target, args

        def start(self):
            pending.append(self)

        def run(self):
            self.target(*self.args)

    monkeypatch.setattr(
        monitor_module, "threading", SimpleNamespace(Thread=QueuedThread)
    )
    return pending


def selected(monitor, monkeypatch, result):
    monkeypatch.setattr(monitor, "_select_query", lambda *_: result)


def tags(result):
    return {
        obj["track_id"]: obj["event"] for obj in result["objects"] if "event" in obj
    }


@pytest.mark.parametrize("key_available", [False, True])
def test_selection_absent_or_pending_preserves_grounded_and_scene_alerts(
    monitor, worker, selections, key_available
):
    monitor._select_key = "offline-test-only" if key_available else ""
    monitor.result(0, report())
    monitor.result(0, report(lane="information", subject=False, scene_label="fight"))
    result = preview(monitor)
    assert tags(result) == {"actor": "smoking"}
    assert all(obj["event_state"] == "alert" for obj in result["objects"])
    assert len(selections) == int(key_available)


def test_failed_selection_keeps_baseline_evidence(
    monitor, worker, selections, monkeypatch
):
    def fail(*_):
        raise ValueError("invalid model output")

    monkeypatch.setattr(monitor, "_select_query", fail)
    monitor.result(0, report())
    preview(monitor)
    selections[0].run()
    assert tags(preview(monitor, 0.2)) == {"actor": "smoking"}
    assert not monitor._select_inflight


def test_valid_negative_replaces_positive_snapshot(
    monitor, worker, selections, monkeypatch
):
    selected(monitor, monkeypatch, ({"actor": ("smoking",)}, False))
    preview(monitor)
    selections[0].run()
    assert tags(preview(monitor, 0.2)) == {"actor": "smoking"}
    monitor.clock.now += 4.1
    selected(monitor, monkeypatch, ({}, False))
    preview(monitor, 0.3)
    selections[1].run()
    result = preview(monitor, 0.4)
    assert tags(result) == {}
    assert result["event_state"] == "normal"


@pytest.mark.parametrize("change", ["stage", "epoch", "generation", "rewind", "viewer"])
def test_late_selection_cannot_tag_a_new_detector_session(
    monitor, worker, selections, monkeypatch, change
):
    selected(monitor, monkeypatch, ({"actor": ("smoking",)}, False))
    preview(monitor, 0.1)
    old = selections[0]
    kwargs = {"at": 0.2}
    if change == "stage":
        monitor.state["stage"] = "stage-two"
    elif change == "epoch":
        kwargs.update(at=1.7, epoch=1)
    elif change == "generation":
        kwargs["generation"] = 1
    elif change == "rewind":
        kwargs["at"] = 0.05
    else:
        monitor.clock.now += 31
    result = preview(monitor, **kwargs)
    assert tags(result) == {}
    assert len(selections) == 2
    old.run()
    session = monitor._detector_sessions[("a" * 36, 0)]["id"]
    assert monitor._active_tags(("a" * 36, 0), session, {"actor"}) == ({}, False)
    assert len(monitor._select_inflight) == 1
    selections[1].run()
    assert monitor._active_tags(("a" * 36, 0), session, {"actor"})[0] == {
        "actor": ("smoking",)
    }


def test_evidence_expires_from_arrival_before_detector_wait(
    monitor, worker, selections, monkeypatch
):
    original = monitor_module.urllib.request.urlopen

    def delayed(*args, **kwargs):
        monitor.clock.now += 6.1
        return original(*args, **kwargs)

    monkeypatch.setattr(monitor_module.urllib.request, "urlopen", delayed)
    preview(monitor)
    assert not selections


def test_late_response_does_not_get_a_fresh_ttl(
    monitor, worker, selections, monkeypatch
):
    selected(monitor, monkeypatch, ({"actor": ("smoking",)}, False))
    preview(monitor)
    monitor.clock.now += 6.1
    selections[0].run()
    assert tags(preview(monitor, 0.2)) == {}
    assert len(selections) == 2


def test_predicted_actor_boxes_are_not_selected_or_named(
    monitor, worker, selections, monkeypatch
):
    selected(monitor, monkeypatch, ({"actor": ("smoking",)}, False))
    preview(monitor)
    selections[0].run()
    worker.snapshot["objects"][0]["observation_state"] = "predicted"
    result = preview(monitor, 0.2)
    assert tags(result) == {}
    assert result["event_state"] == "alert"
    assert selections[0].args[-1] == [
        (1, "actor", (0.1, 0.1, 0.4, 0.9)),
        (2, "bystander", (0.6, 0.1, 0.9, 0.9)),
    ]


def test_predicted_boxes_are_excluded_from_numbered_query(monitor, worker, selections):
    worker.snapshot["objects"][0]["observation_state"] = "predicted"
    preview(monitor)
    assert selections[0].args[-1] == [(1, "bystander", (0.6, 0.1, 0.9, 0.9))]


def test_selection_reservations_are_bounded_across_seeks(monitor, worker, selections):
    for generation in range(8):
        preview(monitor, 0.1 + generation / 100, generation=generation)
    assert len(selections) == monitor._select_max_inflight == 4
    assert len(monitor._select_inflight) == 4
    assert len(monitor._select_states) == 1


def test_thread_start_failure_releases_reservation(monitor, worker, monkeypatch):
    class BrokenThread:
        def __init__(self, **_):
            pass

        def start(self):
            raise RuntimeError("no thread available")

    monkeypatch.setattr(
        monitor_module, "threading", SimpleNamespace(Thread=BrokenThread)
    )
    assert preview(monitor)["status"] == "tracking"
    assert not monitor._select_inflight
    assert next(iter(monitor._select_states.values()))["pending"] is None


def test_shutdown_rejects_pending_results_and_new_queries(
    monitor, worker, selections, monkeypatch
):
    selected(monitor, monkeypatch, ({"actor": ("smoking",)}, False))
    preview(monitor)
    log = monitor.log
    monitor.__exit__()
    monitor.log = log  # The reused detector fixture closes this handle again.
    selections[0].run()
    assert not monitor._select_states
    assert not monitor._select_inflight
    preview(monitor, 0.2)
    assert len(selections) == 1


def test_stage_begin_discards_old_cache_and_completion(
    monitor, worker, selections, monkeypatch, tmp_path
):
    selected(monitor, monkeypatch, ({"actor": ("smoking",)}, False))
    preview(monitor)
    monitor.begin(
        {
            "id": "new-stage",
            "mode": "events",
            "phase": "measurement",
            "duration": 10,
            "repeat": 1,
            "streams": 1,
        },
        tmp_path,
    )
    selections[0].run()
    assert not monitor._select_states
    assert not monitor._select_inflight


def verdicts(monitor):
    return {
        label: {"happening": False, "person": None} for label in monitor._select_labels
    }


def completion(verdict, finish="stop"):
    return {
        "choices": [
            {"finish_reason": finish, "message": {"content": json.dumps(verdict)}}
        ]
    }


def provider(monkeypatch, payload):
    state = SimpleNamespace(read_size=None)

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def read(self, size):
            state.read_size = size
            return (
                payload[:size]
                if isinstance(payload, bytes)
                else json.dumps(payload).encode()[:size]
            )

    monkeypatch.setattr(
        monitor_module.urllib.request, "urlopen", lambda *_a, **_k: Response()
    )
    return state


def query(monitor):
    return monitor._select_query(
        monitor.framesets[0][0], [(1, "actor", (0.1, 0.1, 0.4, 0.9))]
    )


def test_complete_selection_keeps_multiple_labels_and_scene_positive(
    monitor, monkeypatch
):
    verdict = verdicts(monitor)
    verdict.update(
        smoking={"happening": True, "person": 1},
        fight={"happening": True, "person": 1},
        ground_litter={"happening": True, "person": None},
    )
    state = provider(monkeypatch, completion(verdict))
    assert query(monitor) == ({"actor": ("fight", "smoking")}, True)
    assert state.read_size == 64 * 1024 + 1


@pytest.mark.parametrize(
    "happening,person",
    [
        ("false", 1),
        (1, 1),
        (True, True),
        (True, 1.0),
        (True, "1"),
        (True, 2),
        (False, 1),
        (None, None),
    ],
)
def test_invalid_actor_verdict_is_rejected_whole(
    monitor, monkeypatch, happening, person
):
    verdict = verdicts(monitor)
    verdict["smoking"] = {"happening": happening, "person": person}
    provider(monkeypatch, completion(verdict))
    with pytest.raises(ValueError):
        query(monitor)


@pytest.mark.parametrize("mutation", ["missing", "extra", "field", "list", "truncated"])
def test_partial_or_malformed_completion_is_rejected(monitor, monkeypatch, mutation):
    verdict = verdicts(monitor)
    payload = completion(verdict)
    if mutation == "missing":
        del verdict["fight"]
        payload = completion(verdict)
    elif mutation == "extra":
        verdict["other"] = {"happening": False, "person": None}
        payload = completion(verdict)
    elif mutation == "field":
        verdict["fight"]["extra"] = 0
        payload = completion(verdict)
    elif mutation == "list":
        payload = completion([])
    else:
        payload = completion(verdict, finish="length")
    provider(monkeypatch, payload)
    with pytest.raises(ValueError):
        query(monitor)


def test_oversized_response_is_rejected(monitor, monkeypatch):
    provider(monkeypatch, b"x" * (64 * 1024 + 1))
    with pytest.raises(ValueError, match="oversized"):
        query(monitor)


def test_selected_scene_positive_is_kept_without_actor_label(
    monitor, worker, selections, monkeypatch
):
    selected(monitor, monkeypatch, ({}, True))
    preview(monitor)
    selections[0].run()
    result = preview(monitor, 0.2)
    assert result["event_state"] == "alert"
    assert tags(result) == {}
    assert all(obj["event_state"] == "alert" for obj in result["objects"])


@pytest.mark.parametrize("terminal", ["draining", "finished", "failed"])
def test_terminal_monitor_does_not_launch_or_publish_selection(
    monitor, worker, selections, monkeypatch, terminal
):
    def unexpected(*_):
        pytest.fail("selection transport ran after the monitor stopped")

    monkeypatch.setattr(monitor, "_select_query", unexpected)
    preview(monitor)
    monitor.state["state"] = terminal
    selections[0].run()
    assert tags(preview(monitor, 0.2)) == {}
    assert len(selections) == 1
    assert not monitor._select_inflight


def test_expired_viewers_do_not_accumulate_selection_state(
    monitor, worker, selections, monkeypatch
):
    selected(monitor, monkeypatch, ({}, False))
    for index in range(10):
        monitor.clock.now += 31
        monitor.track_preview(
            monitor.state["stage"],
            0,
            {
                "client": f"{index:036x}",
                "time": 0.1,
                "epoch": 0,
                "jpeg": base64.b64encode(monitor.framesets[0][0]).decode(),
            },
        )
        selections[-1].run()
        assert len(monitor._select_states) == 1
        assert not monitor._select_inflight


def test_complete_negative_is_a_valid_empty_snapshot(monitor, monkeypatch):
    provider(monkeypatch, completion(verdicts(monitor)))
    assert query(monitor) == ({}, False)


def test_duplicate_verdict_fields_are_rejected(monitor, monkeypatch):
    payload = completion(verdicts(monitor))
    payload["choices"][0]["message"]["content"] = (
        '{"smoking":{"happening":false,"happening":true,"person":1}}'
    )
    provider(monkeypatch, payload)
    with pytest.raises(ValueError, match="duplicate"):
        query(monitor)


def test_missing_finish_reason_is_rejected(monitor, monkeypatch):
    payload = completion(verdicts(monitor))
    del payload["choices"][0]["finish_reason"]
    provider(monkeypatch, payload)
    with pytest.raises(ValueError, match="incomplete"):
        query(monitor)


def test_response_finishing_after_seek_cannot_retag_the_new_session(
    monitor, worker, selections, monkeypatch
):
    def delayed_result(*_):
        preview(monitor, 0.2, generation=1)
        return {"actor": ("smoking",)}, False

    monkeypatch.setattr(monitor, "_select_query", delayed_result)
    preview(monitor)
    selections[0].run()
    assert len(selections) == 2
    assert tags(preview(monitor, 0.3, generation=1)) == {}
    assert len(monitor._select_inflight) == 1


def test_response_finishing_after_evidence_ttl_is_discarded(
    monitor, worker, selections, monkeypatch
):
    def delayed_result(*_):
        monitor.clock.now += 6.1
        return {"actor": ("smoking",)}, False

    monkeypatch.setattr(monitor, "_select_query", delayed_result)
    preview(monitor)
    selections[0].run()
    assert tags(preview(monitor, 0.2)) == {}


def test_late_detector_response_cannot_recreate_an_obsolete_selection_context(
    monitor, worker, selections
):
    preview(monitor)
    key = ("a" * 36, 0)
    old_session = monitor._detector_sessions[key]["id"]
    preview(monitor, 0.2, generation=1)
    current_session = monitor._detector_sessions[key]["id"]
    monitor._maybe_select(
        key,
        old_session,
        monitor.state["stage"],
        monitor.clock.now,
        monitor.framesets[0][0],
        [("actor", (0.1, 0.1, 0.4, 0.9))],
    )
    assert monitor._select_states[key]["session"] == current_session
    assert len(selections) == 2


def preview_lane(monitor, camera, at=0.1, generation=0):
    while len(monitor.state["cameras"]) <= camera:
        monitor.state["cameras"].append(copy.deepcopy(monitor.state["cameras"][0]))
    return monitor.track_preview(
        monitor.state["stage"],
        camera,
        {
            "client": "a" * 36,
            "time": at,
            "epoch": 0,
            "generation": generation,
            "jpeg": base64.b64encode(monitor.framesets[0][0]).decode(),
        },
    )


def test_twelve_select_lanes_get_a_turn_before_completed_lanes_reenter(
    monitor, worker, selections, monkeypatch
):
    selected(monitor, monkeypatch, ({}, False))
    for camera in range(12):
        preview_lane(monitor, camera)
    assert [query.args[0][1] for query in selections] == [0, 1, 2, 3]
    for cycle in (1, 2):
        # Fixed-order callbacks and four-second model latency used to let
        # lanes0..3 reacquire every slot before any other camera was admitted.
        monitor.clock.now += 4
        for query in selections[-4:]:
            query.run()
        for camera in range(12):
            preview_lane(monitor, camera, at=0.1 + cycle / 10)
    assert [query.args[0][1] for query in selections] == list(range(12))
    assert len(monitor._select_inflight) == 4
    assert len(monitor._select_waiters) == 8
    assert all(len(metadata) == 2 for metadata in monitor._select_waiters.values())


@pytest.mark.parametrize(
    "inactive", ["stalled", "no-person", "expired-frame", "missing-key"]
)
def test_ineligible_select_waiter_does_not_block_a_live_lane(
    monitor, worker, selections, monkeypatch, inactive
):
    monitor._select_max_inflight = 1
    selected(monitor, monkeypatch, ({}, False))
    preview_lane(monitor, 0)
    preview_lane(monitor, 1)
    monitor.clock.now += 4
    selections[0].run()
    preview_lane(monitor, 0, at=0.2)
    assert len(selections) == 1  # The live lane queues behind camera1.
    snapshot = worker.snapshot
    if inactive == "stalled":
        monitor.clock.now += 2.1
    elif inactive == "no-person":
        worker.snapshot = {"objects": []}
        preview_lane(monitor, 1, at=0.2)
        worker.snapshot = snapshot
    elif inactive == "expired-frame":
        key = ("a" * 36, 1)
        session = monitor._detector_sessions[key]["id"]
        monitor._maybe_select(
            key,
            session,
            monitor.state["stage"],
            monitor.clock.now - 6.1,
            monitor.framesets[0][0],
            [("actor", (0.1, 0.1, 0.4, 0.9))],
        )
    else:
        monitor._select_key = ""
        preview_lane(monitor, 1, at=0.2)
        monitor._select_key = "offline-test-only"
    preview_lane(monitor, 0, at=0.3)
    assert [query.args[0][1] for query in selections] == [0, 0]
    assert ("a" * 36, 1) not in monitor._select_waiters


def test_select_waiter_uses_the_new_session_after_seek(
    monitor, worker, selections, monkeypatch
):
    monitor._select_max_inflight = 1
    selected(monitor, monkeypatch, ({}, False))
    preview_lane(monitor, 0)
    preview_lane(monitor, 1)
    key = ("a" * 36, 1)
    old_session = monitor._select_waiters[key][0]
    preview_lane(monitor, 1, at=0.2, generation=1)
    new_session = monitor._select_waiters[key][0]
    assert old_session != new_session
    assert len(monitor._select_waiters) == 1
    monitor.clock.now += 4
    selections[0].run()
    preview_lane(monitor, 0, at=0.2)
    preview_lane(monitor, 1, at=0.3, generation=1)
    assert [query.args[0][1] for query in selections] == [0, 1]
    assert selections[1].args[1] == new_session


@pytest.mark.parametrize("name,value", [
    ("PREVIEW_SELECT_INTERVAL", "0"),
    ("PREVIEW_SELECT_INTERVAL", "-1"),
    ("PREVIEW_SELECT_INTERVAL", "nan"),
    ("PREVIEW_SELECT_INTERVAL", "inf"),
    ("PREVIEW_SELECT_TTL", "0"),
    ("PREVIEW_SELECT_TTL", "-1"),
    ("PREVIEW_SELECT_TTL", "nan"),
    ("PREVIEW_SELECT_TTL", "inf"),
    ("PREVIEW_SELECT_MAX_INFLIGHT", "0"),
    ("PREVIEW_SELECT_MAX_INFLIGHT", "-1"),
    ("PREVIEW_SELECT_MAX_INFLIGHT", "1.5"),
])
def test_invalid_select_budgets_fail_before_startup(tmp_path, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        monitor_module.LiveMonitor(tmp_path, [], SimpleNamespace())


def test_select_budgets_are_configurable_without_changing_defaults(
    tmp_path, monkeypatch
):
    for name in (
        "PREVIEW_SELECT_INTERVAL",
        "PREVIEW_SELECT_TTL",
        "PREVIEW_SELECT_MAX_INFLIGHT",
    ):
        monkeypatch.delenv(name, raising=False)
    default = monitor_module.LiveMonitor(tmp_path, [], SimpleNamespace())
    assert default._select_interval == 4
    assert default._select_ttl == 6
    assert default._select_max_inflight == 4
    monkeypatch.setenv("PREVIEW_SELECT_INTERVAL", "0.5")
    monkeypatch.setenv("PREVIEW_SELECT_TTL", "14")
    monkeypatch.setenv("PREVIEW_SELECT_MAX_INFLIGHT", "6")
    configured = monitor_module.LiveMonitor(tmp_path, [], SimpleNamespace())
    assert configured._select_interval == 0.5
    assert configured._select_ttl == 14
    assert configured._select_max_inflight == 6


@pytest.mark.parametrize(
    "key_available,detector_available",
    [(False, True), (True, False), (True, True)],
)
def test_real_select_badge_requires_provider_and_detector(
    monitor, key_available, detector_available
):
    monitor.previews = [None]
    monitor._select_key = "offline-test-only" if key_available else ""
    monitor.detector_url = "http://detector" if detector_available else ""
    assert monitor.snapshot()["select_active"] is (key_available and detector_available)
