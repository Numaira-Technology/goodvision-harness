"""The live System 1 inspector and the benchmark's current-workbench track share
one VideoJev recipe: same template, same request body, same decision.

CPU only. The inspector's own _payload/_decode are compared with the recipe and
with the benchmark adapter on identical frames; nothing here loads a model.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import replace

import pytest
from PIL import Image
from test_vlm_independent import config

from goodvision_harness import videojev_service
from goodvision_harness.event_sensitivity import EVENT_SENSITIVITY_DEFAULTS
from goodvision_harness.scene_evidence import EvidenceFrame, InspectionRequest
from goodvision_harness.scene_runtime import TemporalInspection
from goodvision_harness.videojev_contract import scene_question_template
from goodvision_harness.videojev_core import OPTIONS
from goodvision_harness.videojev_model import VideoJevInspector
from goodvision_harness.videojev_recipe import (
    PRODUCTION_FRAME_STEP_SECONDS,
    PRODUCTION_WINDOW_FRAMES,
    VideoJevRecipe,
    production_recipe,
    recipe_from_json,
)
from tools.unified_benchmark import adapters

BAG_KNIFE = {
    "label": "bag_knife",
    "zh_name": "翻包刀具",
    "target": "object",
    "definition": "A knife is visibly exposed inside a bag or being taken out of it.",
    "counter_examples": "Metal edges, pens and rulers do not count.",
}


def jpeg(size=(320, 180), color=(60, 80, 100)):
    out = io.BytesIO()
    Image.new("RGB", size, color).save(out, format="JPEG", quality=90)
    return out.getvalue()


def inspector(labels=("smoking", "fall")):
    cfg = replace(
        config(),
        labels=labels,
        model=VideoJevInspector.model_name,
        output_mode="events",
        observation_frame_delta_seconds=PRODUCTION_FRAME_STEP_SECONDS,
        model_settings=replace(
            config().model_settings, provider="local",
            endpoint="http://127.0.0.1:9000/predict",
        ),
    )
    return VideoJevInspector(cfg)


def live_job(cfg, images, at=10.0):
    step = PRODUCTION_FRAME_STEP_SECONDS
    times = [at - step * (len(images) - 1 - i) for i in range(len(images))]
    frames = tuple(
        EvidenceFrame(cfg.stream_session_id, 100 + i, t, image)
        for i, (t, image) in enumerate(zip(times, images, strict=True))
    )
    request = InspectionRequest("req-1", (), "independent_source", cfg.labels,
                                times[0], times[-1])
    return TemporalInspection(cfg.stream_session_id, cfg.source_id, request, frames)


def raw_rows(ids, probabilities):
    rows = []
    for qid, probs in zip(ids, probabilities, strict=True):
        index = max(range(3), key=probs.__getitem__)
        rows.append({"id": qid, "task": "choice", "answer_index": index,
                     "answer": OPTIONS[index], "probabilities": probs})
    return rows


# --- production recipe reproduces the live workbench ---------------------------


def test_production_window_matches_service_constants():
    assert (PRODUCTION_WINDOW_FRAMES, PRODUCTION_FRAME_STEP_SECONDS) == (2, 0.5)
    recipe = production_recipe(("smoking",))
    assert recipe.fps == 2.0 and recipe.gates == ()


@pytest.mark.parametrize(
    "labels", [("smoking", "fall"), tuple(EVENT_SENSITIVITY_DEFAULTS)]
)
@pytest.mark.parametrize("sensitivities", [{}, {"smoking": "high"}])
def test_production_template_equals_the_direct_template(labels, sensitivities):
    sens = {k: v for k, v in sensitivities.items() if k in labels}
    assert production_recipe(labels, sens).question_template() == (
        scene_question_template(labels=labels, sensitivities=sens)
    )


@pytest.mark.parametrize("size", [(320, 180), (1920, 1080)])
def test_live_inspector_body_equals_recipe_body(size):
    live = inspector()
    try:
        images = [jpeg(size, (10, 20, 30)), jpeg(size, (200, 100, 50))]
        job = live_job(live.config, images)
        live_body = json.loads(live._payload(job)[0])
        recipe_body = live.recipe.request_body(
            [(f.timestamp_seconds, f.jpeg_bytes) for f in job.frames],
            session=job.stream_session_id, request_id=job.request.request_id,
        )
        assert json.loads(json.dumps(recipe_body)) == live_body
        if size[0] > 720:  # the >720 px re-encode is shared, not duplicated
            first = Image.open(io.BytesIO(__import__("base64").b64decode(
                live_body["images"][0])))
            assert max(first.size) == 720
    finally:
        live.close()


def test_live_and_recipe_decisions_agree():
    live = inspector()
    try:
        ids = [row["id"] for row in live._question_template]
        raw = raw_rows(ids, [[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
        verdicts = live._decode({"raw": raw}, live._expected, live.config.labels)
        decoded = live.recipe.decode(raw)
        assert {v.event_label: v.detected for v in verdicts} == {
            label: decoded[label]["detected"] for label in live.config.labels
        }
        assert decoded["smoking"]["probabilities"] == [0.7, 0.2, 0.1]
    finally:
        live.close()


# --- benchmark current-workbench track uses the same code path ----------------


def bundle_unit(tmp_path, images, step=PRODUCTION_FRAME_STEP_SECONDS, at=10.0):
    frames = []
    for i, image in enumerate(images):
        path = tmp_path / f"f{i}.jpg"
        path.write_bytes(image)
        frames.append({"path": path.name, "sha256": hashlib.sha256(image).hexdigest(),
                       "time": at - step * (len(images) - 1 - i)})
    return {"id": "req-1", "source_id": "vlm-run", "frames": frames,
            "targets": ["smoking", "fall"]}


PRODUCTION_PROFILE = {"backend": "harness_videojev", "recipe": {
    "name": "production", "labels": ["smoking", "fall"],
    "window_frames": 2, "frame_step_seconds": 0.5}}


def test_benchmark_request_equals_live_inspector_request(tmp_path):
    live = inspector()
    try:
        images = [jpeg((1280, 720), (1, 2, 3)), jpeg((1280, 720), (9, 8, 7))]
        job = live_job(live.config, images)
        live_body = json.loads(live._payload(job)[0])
        url, bench_body = adapters.request_body(
            PRODUCTION_PROFILE, bundle_unit(tmp_path, images), tmp_path
        )
        assert url is None
        assert json.loads(json.dumps(bench_body)) == live_body
        # And the real service accepts it as sent.
        videojev_service._request(json.loads(json.dumps(bench_body)))
    finally:
        live.close()


def test_benchmark_refuses_frames_that_do_not_match_the_recipe(tmp_path):
    images = [jpeg(), jpeg()]
    with pytest.raises(ValueError, match="sampling differs"):
        adapters.request_body(PRODUCTION_PROFILE,
                              bundle_unit(tmp_path, images, step=1.0), tmp_path)


def test_benchmark_answers_follow_the_live_decision_and_gate(tmp_path):
    recipe = recipe_from_json(PRODUCTION_PROFILE["recipe"])
    ids = [row["id"] for row in recipe.question_template()]
    raw = raw_rows(ids, [[0.7, 0.2, 0.1], [0.2, 0.7, 0.1]])
    unit = {"id": "req-1", "targets": ["smoking", "fall"]}
    answers = adapters.parse_response(
        PRODUCTION_PROFILE, unit, {"request_id": "req-1", "raw": raw})
    assert answers["smoking"]["answer"] == "yes" and answers["fall"]["answer"] == "no"
    gated = {"backend": "harness_videojev",
             "recipe": dict(PRODUCTION_PROFILE["recipe"], gates={"smoking": 0.9})}
    answers = adapters.parse_response(gated, unit, {"request_id": "req-1", "raw": raw})
    assert answers["smoking"]["answer"] == "no"  # present but under the gate


# --- recipes with injected events (bag) and other windows -----------------------


def test_bag_recipe_questions_are_accepted_by_the_service():
    recipe = recipe_from_json({
        "name": "bag-4f", "labels": ["weapon", "bag_knife"],
        "definitions": [BAG_KNIFE], "window_frames": 4, "frame_step_seconds": 0.4,
        "gates": {"bag_knife": 0.9},
    })
    template = json.loads(json.dumps([dict(r, query_parts=list(r["query_parts"]))
                                      for r in recipe.question_template()]))
    rows = videojev_service._question_rows(template)
    assert [r["id"] for r in rows] == ["scene/weapon", "scene/bag_knife"]
    assert "visibly exposed inside a bag" in json.dumps(template)
    body = recipe.request_body([(i * 0.4, jpeg()) for i in range(4)],
                               session="s", request_id="r")
    assert body["fps"] == pytest.approx(2.5) and len(body["images"]) == 4
    videojev_service._request(json.loads(json.dumps(body)))


@pytest.mark.parametrize("bad", [
    {"labels": []},
    {"labels": ["fight", "fight"]},
    {"labels": ["nonexistent_event"]},
    {"labels": ["fight"], "window_frames": 5},
    {"labels": ["fight"], "frame_step_seconds": 0},
    {"labels": ["fight"], "gates": {"fall": 0.9}},
    {"labels": ["fight"], "gates": {"fight": 0.1}},
    {"labels": ["fight"], "unknown_field": 1},
])
def test_recipe_validation(bad):
    with pytest.raises(ValueError):
        recipe_from_json({"name": "x", **bad})


def test_request_body_checks_window_shape():
    recipe = VideoJevRecipe(name="x", labels=("fight",))
    with pytest.raises(ValueError, match="frame count"):
        recipe.request_body([(0.0, jpeg())], session="s", request_id="r")
    with pytest.raises(ValueError, match="increase"):
        recipe.request_body([(1.0, jpeg()), (1.0, jpeg())], session="s",
                            request_id="r")
