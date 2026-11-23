"""Benchmark-recipe presets match bag requests for the same full-frame inputs.

CPU only. The bag_knife / bag_alcohol_bottle presets use recipe bench_bag_v1:
the frozen six-question template (copied in object_bench_recipe and checked
against the benchmark profile here), two frames one second apart at fps 1.0,
and argmax. The same two frames also exercise the distinct Bbox-cropped path.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import io
import json
from pathlib import Path

import pytest
from PIL import Image
from test_workflow_composable import base_graph

from goodvision_harness import object_presence as op
from goodvision_harness import videojev_service
from goodvision_harness.object_bench_recipe import QUESTION_TEMPLATE, TEMPLATE_SHA256
from goodvision_harness.videojev_core import OPTIONS
from goodvision_harness.workflows import compile_graph
from tools.unified_benchmark.adapters import parse_response, request_body

PROFILE = json.loads(Path("tools/unified_benchmark/profiles/videojev.json").read_text())
BAG_KNIFE = copy.deepcopy(op.PRESETS["bag_knife"])
BAG_BOTTLE = copy.deepcopy(op.PRESETS["bag_alcohol_bottle"])


def jpeg(color, size=(720, 406)):
    out = io.BytesIO()
    Image.new("RGB", size, color).save(out, format="JPEG", quality=90)
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


def raw_for(probabilities_by_id):
    return [
        row(r["id"], probabilities_by_id.get(r["id"], [0.1, 0.8, 0.1]))
        for r in QUESTION_TEMPLATE
    ]


def test_copied_template_is_the_benchmark_profile_template():
    assert QUESTION_TEMPLATE == PROFILE["question_template"]
    digest = hashlib.sha256(
        json.dumps(QUESTION_TEMPLATE, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert digest == TEMPLATE_SHA256
    ids = [r["id"] for r in QUESTION_TEMPLATE]
    assert {"scene/bag_knife", "scene/bag_alcohol_bottle"} <= set(ids)


def test_recipe_presets_validate_with_argmax_and_stay_experimental():
    for preset in (BAG_KNIFE, BAG_BOTTLE):
        spec = op.object_spec(preset)
        assert spec["recipe"] == "bench_bag_v1" and spec["gate"] is None
        assert op.is_experimental(spec["id"])
        assert "not validated live" in op.preset_status(spec["id"])["note"]
    # A custom object never carries a recipe key, so existing graphs keep their shape.
    assert "recipe" not in op.object_spec(op.PRESETS["knife"])


@pytest.mark.parametrize(
    "bad",
    [
        {**BAG_KNIFE, "gate": 0.9},  # recipe objects decide by argmax
        {**BAG_KNIFE, "recipe": "unknown"},
        {**BAG_KNIFE, "id": "forklift"},  # not a question of this recipe
    ],
)
def test_recipe_objects_reject_bad_fields(bad):
    with pytest.raises(ValueError):
        op.object_spec(bad)


def test_a_node_cannot_mix_recipe_and_custom_objects():
    with pytest.raises(ValueError, match="mix"):
        op.objects_config([BAG_KNIFE, op.PRESETS["knife"]])
    assert len(op.objects_config([BAG_KNIFE, BAG_BOTTLE])) == 2


def test_recipe_node_sends_the_frozen_template_and_reads_its_bag_rows():
    objects = op.objects_config([BAG_KNIFE, BAG_BOTTLE])
    assert [r["id"] for r in op.question_template(objects)] == [
        r["id"] for r in QUESTION_TEMPLATE
    ]
    raw = raw_for(
        {
            "scene/bag_knife": [0.40, 0.35, 0.25],  # weak but argmax present
            "scene/bag_alcohol_bottle": [0.2, 0.7, 0.1],
        }
    )
    decided = op.decide(raw, objects)
    assert decided["bag_knife"]["detected"] is True  # argmax, no 0.95 gate
    assert decided["bag_knife"]["gate"] is None
    assert decided["bag_alcohol_bottle"]["detected"] is False
    with pytest.raises(ValueError):  # every frozen row is required
        op.decide(raw[:-1], objects)


def recipe_manager(post, history):
    return op.ObjectPresenceManager(
        # The node's own window settings are overridden by the recipe.
        {
            "id": "objects",
            "objects": [BAG_KNIFE, BAG_BOTTLE],
            "window_frames": 4,
            "frame_step_seconds": 0.4,
        },
        frame_provider=iter([history[-1]]).__next__,
        history_provider=lambda seconds: history,
        post=post,
        stream_session_id="s",
    )


def test_recipe_window_is_two_frames_about_one_second_apart_at_fps_one():
    history = [(i, i * 0.4, jpeg((i, i, i))) for i in range(12)]  # 2.5 fps display grid
    bodies = []
    manager = recipe_manager(
        lambda b: bodies.append(b) or {"raw": raw_for({})}, history
    )
    record = manager.tick()
    assert record["error"] is None
    t0, t1 = record["frame_times"]
    # select_window takes the latest frame at or before newest - 1 s (within half a
    # step): on a 0.4 s display grid the real gap is 1.0-1.5 s; fps stays 1.0.
    assert len(record["frame_times"]) == 2 and 1.0 - 1e-6 <= t1 - t0 < 1.5
    assert bodies[0]["fps"] == 1.0
    videojev_service._request(
        json.loads(json.dumps(bodies[0]))
    )  # real parser accepts it


def test_recipe_presets_need_a_history_provider():
    with pytest.raises(ValueError, match="history provider"):
        op.ObjectPresenceManager(
            {"id": "objects", "objects": [BAG_KNIFE]},
            frame_provider=lambda: None,
            post=lambda b: {},
            stream_session_id="s",
        )


@pytest.mark.parametrize("wire_tracks", [False, True], ids=["full-frame", "bbox-crop"])
def test_recipe_request_image_parity_requires_uncropped_frames(tmp_path, wire_tracks):
    # Two 720 px frames exactly 1 s apart, as preflight extracts them.
    images = [jpeg((10, 20, 30)), jpeg((200, 100, 50))]
    frames = []
    for i, data in enumerate(images):
        (tmp_path / f"f{i}.jpg").write_bytes(data)
        frames.append(
            {
                "path": f"f{i}.jpg",
                "sha256": hashlib.sha256(data).hexdigest(),
                "time": 7.0 + i,
            }
        )
    unit = {
        "id": "u",
        "source_id": "test/x",
        "frames": frames,
        "targets": ["bag_knife", "bag_alcohol_bottle"],
    }
    _, bench = request_body(PROFILE, unit, tmp_path)

    history = [(1, 7.0, images[0]), (2, 8.0, images[1])]
    bodies = []
    graph = base_graph(wire_tracks=wire_tracks, tracks_to_output=True)
    node = next(node for node in graph["nodes"] if node["type"] == "objects")
    node["config"]["objects"] = [BAG_KNIFE, BAG_BOTTLE]
    plan = compile_graph(graph, camera_id="camera01")
    manager = op.ObjectPresenceManager(
        plan["objects"],
        boxes_provider=op.synthetic_boxes if wire_tracks else None,
        frame_provider=lambda: history[-1],
        history_provider=lambda _seconds: history,
        post=lambda b: bodies.append(b) or {"raw": raw_for({})},
        stream_session_id="s",
    )
    record = manager.tick()
    assert record["error"] is None
    live = bodies[0]
    for key in ("fps", "frame_times", "question_template"):
        assert json.loads(json.dumps(live[key])) == json.loads(
            json.dumps(bench[key])
        ), key
    if wire_tracks:
        assert record["focus"]["mode"] == "person_crop"
        assert all(a != b for a, b in zip(live["images"], bench["images"], strict=True))
        for encoded in live["images"]:
            with Image.open(io.BytesIO(base64.b64decode(encoded))) as cropped:
                assert cropped.width < 720
    else:
        assert record["focus"]["mode"] == "full_frame"
        assert live["images"] == bench["images"]

    # And both read the same answer from the same raw response.
    raw = raw_for({"scene/bag_knife": [0.6, 0.3, 0.1]})
    bench_answers = parse_response(
        PROFILE,
        unit,
        {
            "request_id": "u",
            "model": PROFILE["model"],
            "revision": PROFILE["revision"],
            "raw": raw,
        },
    )
    live_answers = op.decide(raw, op.objects_config([BAG_KNIFE, BAG_BOTTLE]))
    assert bench_answers["bag_knife"]["answer"] == "yes"
    assert live_answers["bag_knife"]["detected"] is True
    assert bench_answers["bag_alcohol_bottle"]["answer"] == "no"
    assert live_answers["bag_alcohol_bottle"]["detected"] is False
