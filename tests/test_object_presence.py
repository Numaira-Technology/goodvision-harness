"""Object presence node: contract, gate, hysteresis, graph, runtime and render.

VideoJev is replaced by a fake responder; the request body and question template
are checked against the real VideoJev service validators, without a model.
"""

from __future__ import annotations

import copy
import io
import itertools
import json

import pytest
from PIL import Image

from goodvision_harness import object_presence as op
from goodvision_harness import videojev_service
from goodvision_harness.videojev_core import OPTIONS
from goodvision_harness.workflow_modules import display_settings
from goodvision_harness.workflow_render import render_output
from goodvision_harness.workflows import (
    WorkflowValidationError,
    compile_graph,
    validate_graph,
)


def jpeg(color=(40, 40, 40), size=(64, 48)):
    out = io.BytesIO()
    Image.new("RGB", size, color).save(out, format="JPEG")
    return out.getvalue()


def row(qid, probabilities):
    index = max(range(3), key=probabilities.__getitem__)
    return {
        "id": qid,
        "task": "choice",
        "answer_index": index,
        "answer": OPTIONS[index],
        "probabilities": probabilities,
    }


KNIFE = copy.deepcopy(op.PRESETS["knife"])
BOTTLE = copy.deepcopy(op.PRESETS["alcohol_bottle"])


# --- validation ---------------------------------------------------------


def test_presets_validate_and_defaults_apply():
    specs = op.objects_config([KNIFE, BOTTLE])
    assert [s["gate"] for s in specs] == [0.95, 0.95]
    spec = op.object_spec({"id": "phone", "name": "phone", "criterion": "a phone"})
    assert spec["gate"] == op.DEFAULT_GATE and spec["zh_name"] == ""


@pytest.mark.parametrize(
    "bad",
    [
        {"id": "Knife", "name": "k", "criterion": "c"},
        {"id": "knife", "name": "", "criterion": "c"},
        {"id": "knife", "name": "k", "criterion": "c", "gate": 0.2},
        {"id": "knife", "name": "k", "criterion": "c", "gate": float("nan")},
        {"id": "knife", "name": "k", "criterion": "c", "extra": 1},
        {"id": "knife", "name": "k", "criterion": "x" * 601},
    ],
)
def test_object_spec_rejects_bad_values(bad):
    with pytest.raises(ValueError):
        op.object_spec(bad)


def test_objects_list_bounds_and_unique_ids():
    with pytest.raises(ValueError):
        op.objects_config([])
    with pytest.raises(ValueError):
        op.objects_config([KNIFE, KNIFE])
    with pytest.raises(ValueError):
        op.objects_config([dict(KNIFE, id=f"o{i}") for i in range(9)])


# --- contract with the real VideoJev service ----------------------------


def test_template_is_accepted_by_the_videojev_service():
    wire = json.loads(json.dumps(list(op.question_template([KNIFE, BOTTLE]))))
    rows = videojev_service._question_rows(wire)
    assert [r["id"] for r in rows] == ["object/knife", "object/alcohol_bottle"]
    assert all(r["options"] == list(OPTIONS) for r in rows)
    before, after = rows[0]["query_parts"]
    assert "source_times" in before and "last frame" in after
    assert "Not this:" in before  # counter examples reach the model


def test_request_body_is_accepted_by_the_videojev_service():
    bodies = []
    manager = op.ObjectPresenceManager(
        {"id": "objects", "objects": [KNIFE]},
        frame_provider=iter([(1, 10.0, jpeg()), (2, 11.0, jpeg())]).__next__,
        post=lambda body: bodies.append(body)
        or {"raw": [row("object/knife", [0.1, 0.8, 0.1])]},
        stream_session_id="s1",
    )
    manager.tick()
    manager.tick()
    body = bodies[0]
    parsed = videojev_service._request(json.loads(json.dumps(body)))  # as sent
    assert parsed["frame_times"] == [10.0, 11.0] and len(parsed["images"]) == 2
    assert body["fps"] == pytest.approx(1.0)
    # Own session: never replaces the event branch's pending work or template.
    assert body["session"] == "s1:objects:objects"


# --- gate and hysteresis --------------------------------------------------


@pytest.mark.parametrize(
    "probabilities,answer,detected",
    [
        ([0.97, 0.02, 0.01], True, True),
        ([0.95, 0.04, 0.01], True, True),
        ([0.60, 0.30, 0.10], True, False),  # argmax present but below gate
        ([0.20, 0.70, 0.10], False, False),
        ([0.30, 0.10, 0.60], None, False),
    ],
)
def test_gate_requires_argmax_present_and_probability(probabilities, answer, detected):
    result = op.decide([row("object/knife", probabilities)], [op.object_spec(KNIFE)])
    assert result["knife"]["answer"] is answer
    assert result["knife"]["detected"] is detected
    assert result["knife"]["present_probability"] == probabilities[0]


def test_decide_rejects_contract_violations():
    spec = [op.object_spec(KNIFE)]
    with pytest.raises(ValueError):
        op.decide([], spec)
    broken = row("object/knife", [0.97, 0.02, 0.01])
    broken["answer_index"] = 1
    with pytest.raises(ValueError):
        op.decide([broken], spec)


def test_tracker_latches_until_n_consecutive_misses():
    tracker = op.PresenceTracker(["knife"], 3)
    seq = [True, False, False, True, False, False, False, False]
    states = [tracker.update("knife", d, float(t))["present"]
              for t, d in enumerate(seq)]
    assert states == [True, True, True, True, True, True, False, False]
    assert tracker.update("knife", None, 9.0)["present"] is False


# --- manager ---------------------------------------------------------------


class FakeVideoJev:
    def __init__(self, script):
        self.script = list(script)
        self.templates = {}
        self.bodies = []

    def __call__(self, body):
        self.bodies.append(copy.deepcopy(body))
        if "question_template" in body:
            self.templates[body["session"]] = body["question_template"]
        step = self.script.pop(0)
        if isinstance(step, Exception):
            raise step
        if body["session"] not in self.templates:
            raise op.ObjectPresenceError("template_missing")
        return {"request_id": body["request_id"], "raw": step}


def frames(n):
    return iter([(i, float(i), jpeg()) for i in range(1, n + 1)]).__next__


def test_manager_sends_template_once_and_publishes_latched_results():
    fake = FakeVideoJev(
        [
            [row("object/knife", [0.97, 0.02, 0.01])],
            [row("object/knife", [0.10, 0.85, 0.05])],
        ]
    )
    manager = op.ObjectPresenceManager(
        {"id": "objects", "objects": [KNIFE], "clear_after": 2},
        frame_provider=frames(3),
        post=fake,
        stream_session_id="s",
        source_id="cam",
        generation=4,
    )
    assert manager.tick() is None  # first frame is context only
    first = manager.tick()
    second = manager.tick()
    assert "question_template" in fake.bodies[0]
    assert "question_template" not in fake.bodies[1]
    knife = first["objects"][0]
    assert knife["detected"] and knife["present"] and knife["changed"]
    assert first["source_id"] == "cam" and first["generation"] == 4
    held = second["objects"][0]
    assert held["detected"] is False and held["present"] is True  # 1 of 2 misses
    assert manager.latest() == second
    assert len(manager.history()) == 2
    assert manager.status()["succeeded"] == 2


def test_manager_resends_template_after_service_restart():
    fake = FakeVideoJev(
        [
            [row("object/knife", [0.1, 0.8, 0.1])],
            op.ObjectPresenceError("template_missing"),
            [row("object/knife", [0.1, 0.8, 0.1])],
        ]
    )
    manager = op.ObjectPresenceManager(
        {"id": "objects", "objects": [KNIFE]},
        frame_provider=frames(3),
        post=fake,
        stream_session_id="s",
    )
    manager.tick()
    manager.tick()
    record = manager.tick()
    assert record["error"] is None
    assert "question_template" in fake.bodies[-1]


def test_manager_records_errors_without_stopping_or_clearing():
    fake = FakeVideoJev(
        [
            [row("object/knife", [0.97, 0.02, 0.01])],
            op.ObjectPresenceError("queue_full"),
            [{"id": "object/knife"}],
        ]
    )
    manager = op.ObjectPresenceManager(
        {"id": "objects", "objects": [KNIFE], "clear_after": 3},
        frame_provider=frames(4),
        post=fake,
        stream_session_id="s",
    )
    manager.tick()
    manager.tick()
    failed = manager.tick()
    invalid = manager.tick()
    assert failed["error"] == "queue_full"
    assert invalid["error"] == "invalid_response"
    assert failed["objects"][0]["detected"] is None
    assert invalid["objects"][0]["present"] is True  # 2 misses < 3
    assert manager.status()["failed"] == 2


def test_transport_failures_cannot_clear_a_positive_detection():
    fake = FakeVideoJev([
        [row("object/knife", [0.97, 0.02, 0.01])],
        *[op.ObjectPresenceError("queue_full") for _ in range(4)],
        *[[row("object/knife", [0.01, 0.98, 0.01])] for _ in range(3)],
    ])
    manager = op.ObjectPresenceManager(
        {"id": "objects", "objects": [KNIFE], "clear_after": 3},
        frame_provider=frames(9), post=fake, stream_session_id="s",
    )
    manager.tick()
    assert manager.tick()["objects"][0]["present"] is True
    for _ in range(4):
        failed = manager.tick()
        assert failed["error"] == "queue_full"
        assert failed["objects"][0]["present"] is True
        assert failed["objects"][0]["changed"] is False
    assert manager.tick()["objects"][0]["present"] is True
    assert manager.tick()["objects"][0]["present"] is True
    assert manager.tick()["objects"][0]["present"] is False


def test_stop_closes_transport_after_an_inflight_request_drains():
    import threading

    entered, release, closed = threading.Event(), threading.Event(), []

    def post(body):
        entered.set()
        assert release.wait(5)
        return {"raw": [row("object/knife", [0, 1, 0])]}

    post.close = lambda: closed.append(True)
    manager = op.ObjectPresenceManager(
        {"id": "objects", "objects": [KNIFE], "sample_fps": 4},
        frame_provider=frames(2), post=post, stream_session_id="s",
    )
    manager.start()
    try:
        assert entered.wait(5)
        manager.stop(timeout=0)
        assert manager.busy and closed == []
    finally:
        release.set()
        manager._thread.join(5)
    assert not manager.busy
    assert closed == [True]
    manager.stop()
    assert closed == [True]


def test_manager_skips_repeated_or_missing_frames():
    calls = []
    same = (5, 1.0, jpeg())
    provided = iter([None, same, same]).__next__
    manager = op.ObjectPresenceManager(
        {"id": "objects", "objects": [KNIFE]},
        frame_provider=provided,
        post=calls.append,
        stream_session_id="s",
    )
    assert [manager.tick() for _ in range(3)] == [None, None, None]
    assert calls == [] and manager.status()["skipped_no_frame"] == 1


def test_synthetic_post_reports_unclear_without_a_model():
    manager = op.ObjectPresenceManager(
        {"id": "objects", "objects": [KNIFE, BOTTLE]},
        frame_provider=frames(2),
        post=op.synthetic_post(),
        stream_session_id="s",
    )
    manager.tick()
    record = manager.tick()
    assert record["error"] is None
    assert [o["answer"] for o in record["objects"]] == [None, None]
    assert not any(o["present"] for o in record["objects"])


def test_large_frames_are_downscaled_to_720():
    bodies = []
    big = jpeg(size=(1920, 1080))
    manager = op.ObjectPresenceManager(
        {"id": "objects", "objects": [KNIFE]},
        frame_provider=iter([(1, 0.0, big), (2, 1.0, big)]).__next__,
        post=lambda b: bodies.append(b) or {"raw": [row("object/knife", [0, 1, 0])]},
        stream_session_id="s",
    )
    manager.tick()
    manager.tick()
    import base64

    with Image.open(io.BytesIO(base64.b64decode(bodies[0]["images"][1]))) as image:
        assert max(image.size) == 720


# --- graph ---------------------------------------------------------------


def objects_graph(config=None, *, extra=()):
    nodes = [
        {"id": "s", "type": "source"},
        {"id": "o", "type": "objects", "config": config or {}},
        {"id": "out", "type": "output"},
        *extra,
    ]
    edges = [
        {"id": "e1", "from_node": "s", "from_port": "frames",
         "to_node": "o", "to_port": "frames"},
        {"id": "e2", "from_node": "o", "from_port": "objects",
         "to_node": "out", "to_port": "objects"},
    ]
    return {"nodes": nodes, "edges": edges}


def test_compile_graph_emits_objects_plan_on_local_videojev():
    plan = compile_graph(objects_graph({"objects": [KNIFE, BOTTLE], "sample_fps": 2}))
    assert plan["objects_node_id"] == "o"
    assert plan["objects"]["sample_fps"] == 2
    assert [o["id"] for o in plan["objects"]["objects"]] == ["knife", "alcohol_bottle"]
    assert plan["objects_engine"] == "system1"
    assert plan["objects_execution_mode"] == "local"
    assert plan["objects_service_url"].endswith(":8097")
    assert plan["features"] == [] and plan["enabled"] is True


def test_default_objects_node_is_knife_preset():
    node = validate_graph(objects_graph())["nodes"][1]
    assert [o["id"] for o in node["config"]["objects"]] == ["knife"]
    assert node["config"]["engine"] == "system1"


@pytest.mark.parametrize(
    "config",
    [
        {"objects": []},
        {"objects": [dict(KNIFE, gate=2)]},
        {"sample_fps": 10},
        {"clear_after": 0},
        {"engine": "neural"},
        {"features": ["smoking"]},
    ],
)
def test_graph_rejects_bad_object_config(config):
    with pytest.raises(WorkflowValidationError):
        validate_graph(objects_graph(config))


def test_only_one_objects_node_per_workflow():
    second = {"id": "o2", "type": "objects"}
    with pytest.raises(WorkflowValidationError, match="only one objects"):
        validate_graph(objects_graph(extra=(second,)))


def test_unconsumed_objects_node_is_not_planned():
    graph = objects_graph()
    graph["edges"] = [
        graph["edges"][0],
        {"id": "e3", "from_node": "s", "from_port": "frames",
         "to_node": "out", "to_port": "frames"},
    ]
    with pytest.raises(WorkflowValidationError):
        compile_graph(graph)  # enabled node without an output path


# --- render --------------------------------------------------------------


def test_output_layer_draws_only_present_objects():
    assert display_settings({})["objects"] == {
        "visible": True, "position": "bottom-left",
    }
    value = {
        "timestamp_seconds": 5.0,
        "error": None,
        "objects": [
            {"name": "knife", "zh_name": "刀具", "present": True},
            {"name": "alcohol bottle", "zh_name": "酒瓶", "present": False},
        ],
    }
    output = {"inputs": [{"port": "objects", "value": value}], "display": {}}
    base = jpeg(size=(320, 240))
    drawn = render_output(base, output, 5.0)
    assert drawn != render_output(
        base, {"inputs": [{"port": "objects", "value": None}], "display": {}}, 5.0
    )


# --- multi-frame window and per-tick log (knife protocol prerequisites) ---


def stream(fps=30.0, seconds=3.0, start=0.0):
    """Synthetic display history: (sequence, timestamp, jpeg)."""
    image = jpeg()
    return [(i, start + i / fps, image) for i in range(int(seconds * fps) + 1)]


def test_select_window_picks_frames_about_one_step_apart():
    history = stream()
    newest = history[-1]
    window = op.select_window(history, newest, 4, 0.4)
    times = [round(f[1], 3) for f in window]
    assert len(window) == 4 and window[-1] == newest
    gaps = [b - a for a, b in itertools.pairwise(times)]
    assert all(0.35 <= g <= 0.45 for g in gaps), gaps
    assert len({f[0] for f in window}) == 4


def test_select_window_waits_until_history_covers_the_span():
    history = stream(seconds=1.0)  # only 1.0 s; 4 x 0.4 needs 1.2 s
    assert op.select_window(history, history[-1], 4, 0.4) is None
    assert op.select_window([history[-1]], history[-1], 2, 0.4) is None


def test_select_window_ignores_frames_newer_than_the_newest():
    history = stream(seconds=3.0)
    newest = history[60]
    window = op.select_window(history, newest, 4, 0.4)
    assert window[-1] == newest and all(f[1] <= newest[1] for f in window)


@pytest.mark.parametrize("bad", [3, 1, 5, 2.0, "4"])
def test_window_frames_must_be_two_or_four(bad):
    with pytest.raises(ValueError):
        op.window_frames(bad)


@pytest.mark.parametrize("bad", [0.05, 1.5, float("inf"), "0.4"])
def test_frame_step_bounds(bad):
    with pytest.raises(ValueError):
        op.frame_step_seconds(bad)


def test_four_frame_window_requires_a_history_provider():
    with pytest.raises(ValueError, match="history provider"):
        op.ObjectPresenceManager(
            {"id": "o", "objects": [KNIFE], "window_frames": 4},
            frame_provider=lambda: None, post=lambda b: {}, stream_session_id="s",
        )


def four_frame_manager(post, frames, history):
    return op.ObjectPresenceManager(
        {"id": "objects", "objects": [KNIFE], "window_frames": 4,
         "frame_step_seconds": 0.4},
        frame_provider=iter(frames).__next__,
        history_provider=lambda seconds: history,
        post=post, stream_session_id="s",
    )


def test_four_frame_request_is_accepted_by_the_videojev_service():
    history = stream(seconds=3.0)
    bodies = []
    manager = four_frame_manager(
        lambda b: bodies.append(b) or {"raw": [row("object/knife", [0, 1, 0])]},
        [history[-1]], history,
    )
    record = manager.tick()
    assert record["error"] is None and len(record["frame_times"]) == 4
    parsed = videojev_service._request(json.loads(json.dumps(bodies[0])))
    assert len(parsed["images"]) == 4
    assert parsed["fps"] == pytest.approx(2.5)  # 1 / 0.4 s
    assert parsed["frame_times"] == record["frame_times"]


def test_tick_log_records_every_tick_with_a_reason():
    clock = iter(x / 10 for x in range(1000))
    image = jpeg()
    frames = iter([None, (1, 0.0, image), (1, 0.0, image), (2, 1.0, image),
                   (3, 2.0, image)])
    script = iter([{"raw": [row("object/knife", [0.1, 0.8, 0.1])]},
                   op.ObjectPresenceError("queue_full")])

    def post(body):
        step = next(script)
        if isinstance(step, Exception):
            raise step
        return step

    ticks = []
    manager = op.ObjectPresenceManager(
        {"id": "objects", "objects": [KNIFE]},
        frame_provider=frames.__next__, post=post, stream_session_id="s",
        clock=lambda: next(clock), on_tick=ticks.append,
    )
    for scheduled in (0.0, 1.0, 2.0, 3.0, 4.0):
        manager.tick(scheduled_at=scheduled)
    log = manager.tick_log()
    assert [t["reason"] for t in log] == [
        "no_frame", "waiting_for_history", "no_new_frame", "valid", "error",
    ]
    assert [t["eligible"] for t in log] == [False, False, False, True, True]
    assert [t["scheduled_at"] for t in log] == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert log[3]["valid"] and log[3]["frame_times"] == [0.0, 1.0]
    assert log[4]["error"] == "queue_full" and not log[4]["valid"]
    assert log[3]["sent_at"] <= log[3]["completed_at"]
    assert ticks == log  # callback sees the same entries
    status = manager.status()
    assert status["skipped_no_frame"] == 1 and status["skipped_no_new_frame"] == 1
    assert status["waiting_for_history"] == 1


def test_four_frame_tick_waits_for_history_then_judges():
    short, full = stream(seconds=0.9), stream(seconds=2.0)
    provided = iter([short, full])
    image_frames = iter([short[-1], full[-1]])
    manager = op.ObjectPresenceManager(
        {"id": "objects", "objects": [KNIFE], "window_frames": 4},
        frame_provider=image_frames.__next__,
        history_provider=lambda seconds: next(provided),
        post=lambda b: {"raw": [row("object/knife", [0.2, 0.7, 0.1])]},
        stream_session_id="s",
    )
    assert manager.tick() is None
    assert manager.tick()["error"] is None
    assert [t["reason"] for t in manager.tick_log()] == [
        "waiting_for_history", "valid",
    ]


def test_compile_graph_carries_window_settings():
    plan = compile_graph(objects_graph({"objects": [KNIFE], "window_frames": 4,
                                        "frame_step_seconds": 0.4}))
    assert plan["objects"]["window_frames"] == 4
    assert plan["objects"]["frame_step_seconds"] == 0.4
    default = compile_graph(objects_graph({"objects": [KNIFE]}))["objects"]
    assert default["window_frames"] == 2


@pytest.mark.parametrize("config", [{"window_frames": 3},
                                    {"frame_step_seconds": 2}])
def test_graph_rejects_bad_window_settings(config):
    with pytest.raises(WorkflowValidationError):
        validate_graph(objects_graph(config))


def test_recent_frames_reads_bounded_display_history():
    class Connection:
        def display_frames(self, after, max_age_seconds, limit):
            assert after is None and limit == 120 and max_age_seconds == 1.6
            return ((1, 0.5, 20, 10, b"a"), (2, 1.0, 20, 10, b"b"))

    assert op.recent_frames(Connection(), 1.6) == [(1, 0.5, b"a"), (2, 1.0, b"b")]


# --- fixed-grid schedule: slow requests and missing frames (review on #121) ---


class FakeTime:
    def __init__(self):
        self.now = 0.0

    def clock(self):
        return self.now

    def wait(self, seconds):
        self.now += seconds
        return False


def grid_manager(fake, post, provider):
    return op.ObjectPresenceManager(
        {"id": "objects", "objects": [KNIFE]},
        frame_provider=provider, post=post, stream_session_id="s",
        clock=fake.clock, wait=fake.wait,
    )


def test_slow_request_logs_missed_planned_ticks_without_backfill():
    fake = FakeTime()
    image = jpeg()
    calls = []

    def provider():
        return (len(calls) + int(fake.now * 10), fake.now, image)

    def post(body):
        calls.append(fake.now)
        fake.now += 2.5 if len(calls) == 1 else 0.1  # first request is slow
        return {"raw": [row("object/knife", [0.1, 0.8, 0.1])]}

    manager = grid_manager(fake, post, provider)
    manager._loop(max_planned=6)
    log = manager.tick_log()
    assert [t["planned_index"] for t in log] == [0, 1, 2, 3, 4, 5]
    assert [t["scheduled_at"] for t in log] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert [t["reason"] for t in log] == [
        "waiting_for_history", "valid", "missed_busy", "missed_busy", "valid", "valid",
    ]
    # No backfill or queue: only ticks 1, 4 and 5 sent, each at its grid time.
    assert calls == [1.0, 4.0, 5.0]
    assert manager.status()["missed_busy"] == 2


def test_missing_frames_after_start_stay_on_the_planned_grid():
    fake = FakeTime()
    image = jpeg()
    script = iter([(1, 0.0, image), (2, 1.0, image), None, None,
                   (3, 4.0, image), (4, 5.0, image)])
    manager = grid_manager(
        fake, lambda b: {"raw": [row("object/knife", [0.1, 0.8, 0.1])]},
        script.__next__,
    )
    manager._loop(max_planned=6)
    log = manager.tick_log()
    assert [(t["planned_index"], t["scheduled_at"], t["reason"]) for t in log] == [
        (0, 0.0, "waiting_for_history"), (1, 1.0, "valid"),
        (2, 2.0, "no_frame"), (3, 3.0, "no_frame"),
        (4, 4.0, "valid"), (5, 5.0, "valid"),
    ]
    assert manager.status()["skipped_no_frame"] == 2
