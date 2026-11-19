"""Object presence node: contract, gate, hysteresis, graph, runtime and render.

VideoJev is replaced by a fake responder; the request body and question template
are checked against the real VideoJev service validators, without a model.
"""

from __future__ import annotations

import copy
import io
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
