"""Live-spacing (0.5 s) bundle: windows, request bodies, manifest, sweep wiring."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest
from PIL import Image

from tools.unified_benchmark.contract import digest, file_hash
from tools.videojev_probe import gpu_sweep, live_bundle

CONFIG = json.loads(Path("tools/unified_benchmark/profiles/videojev.json").read_text())


def jpeg(shade):
    out = io.BytesIO()
    Image.new("RGB", (64, 36), (shade, shade, shade)).save(out, format="JPEG")
    return out.getvalue()


def frames(tmp_path, source, count, step=0.5):
    result = []
    for k in range(count):
        data = jpeg(k * 7 % 255)
        rel = f"frames/test/{source}/{k:05}.jpg"
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(data)
        result.append({"path": rel, "time": round(k * step, 6),
                       "sha256": hashlib.sha256(data).hexdigest(), "size": [64, 36]})
    return result


def frozen_plan():
    units = [{"id": f"videojev/test/{s}/all/{i}", "source_id": f"test/{s}", "targets": ["fight"]}
             for s in ("a", "b") for i in range(2)]
    questions = [
        {"number": 1, "event": "fight", "reference": "yes", "filename": "a.mp4",
         "source_video_group": "g", "unit_ids": ["videojev/test/a/all/0", "videojev/test/a/all/1"]},
        {"number": 2, "event": "fight", "reference": "no", "filename": "b.mp4",
         "source_video_group": "g", "unit_ids": ["videojev/test/b/all/0"]},
    ]
    return {"split": "test", "units": units, "questions": questions}


@pytest.mark.parametrize(("duration", "last", "expected"), [
    (2.1, 2.05, [0.0, 0.5, 1.0, 1.5, 2.0]),
    (2.0, 1.97, [0.0, 0.5, 1.0, 1.5]),  # 2.0 is past the last decodable frame
    (1.2, 5.0, [0.0, 0.5, 1.0]),
])
def test_live_times_stay_inside_the_video(duration, last, expected):
    assert live_bundle.live_times(duration, last, 0.5) == expected


def test_request_body_uses_live_fps_and_refuses_other_spacing(tmp_path):
    fs = frames(tmp_path, "a", 3)
    unit = {"id": "u", "source_id": "test/a", "frames": fs[:2]}
    body = live_bundle.request_body(CONFIG, unit, tmp_path, 0.5)
    assert body["fps"] == 2.0 and body["frame_times"] == [0.0, 0.5]
    assert body["question_template"] == CONFIG["question_template"]
    with pytest.raises(ValueError, match="adjacent frames"):
        live_bundle.request_body(CONFIG, {**unit, "frames": [fs[0], fs[2]]}, tmp_path, 0.5)


def test_plan_keeps_questions_and_regroups_them_by_video(tmp_path):
    by_source = {"test/a": frames(tmp_path, "a", 5), "test/b": frames(tmp_path, "b", 3)}
    plan = live_bundle.plan_from_frames(frozen_plan(), CONFIG, tmp_path, by_source, 0.5)
    assert len(plan["units"]) == 4 + 2
    q1, q2 = plan["questions"]
    assert (q1["reference"], q2["reference"]) == ("yes", "no")
    assert len(q1["unit_ids"]) == 4 and all("/test/a/" in u for u in q1["unit_ids"])
    assert len(q2["unit_ids"]) == 2
    unit = plan["units"][0]
    assert unit["request_sha256"] == digest(live_bundle.request_body(CONFIG, unit, tmp_path, 0.5))


def write_live_bundle(tmp_path):
    by_source = {"test/a": frames(tmp_path, "a", 4), "test/b": frames(tmp_path, "b", 3)}
    plan = live_bundle.plan_from_frames(frozen_plan(), CONFIG, tmp_path, by_source, 0.5)
    files = {}
    for rel, value in (("plans/videojev-test.json", plan), ("configs/videojev.json", CONFIG)):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(json.dumps(value))
    for path in tmp_path.rglob("*"):
        if path.is_file():
            files[path.relative_to(tmp_path).as_posix()] = file_hash(path)
    (tmp_path / "live-manifest.json").write_text(json.dumps(
        {"kind": live_bundle.KIND, "step_seconds": 0.5, "split": "test",
         "parent_bundle_sha256": digest(frozen_plan()), "units": len(plan["units"]),
         "questions": len(plan["questions"]), "files": files}))
    return plan


def test_verify_detects_a_changed_file(tmp_path):
    write_live_bundle(tmp_path)
    _, step = live_bundle.verify(tmp_path)
    assert step == 0.5
    (tmp_path / "frames/test/a/00001.jpg").write_bytes(jpeg(200))
    with pytest.raises(ValueError, match="differs"):
        live_bundle.verify(tmp_path)


def test_sweep_sends_live_bodies_for_a_live_bundle(tmp_path):
    plan = write_live_bundle(tmp_path)
    assert gpu_sweep.live_step(tmp_path) == 0.5
    body = gpu_sweep.checked_body(CONFIG, plan["units"][0], tmp_path)
    assert body["fps"] == 2.0
    tampered = dict(plan["units"][0], request_sha256="0" * 64)
    with pytest.raises(ValueError, match="differs"):
        gpu_sweep.checked_body(CONFIG, tampered, tmp_path)


@pytest.mark.parametrize("damage", ["plan", "config", "frame", "extra-plan", "split", "count", "request", "parent"])
def test_run_rejects_unsealed_live_inputs_before_contacting_service(tmp_path, monkeypatch, damage):
    plan = write_live_bundle(tmp_path)
    path = tmp_path / "live-manifest.json"
    manifest = json.loads(path.read_text())
    if damage in ("plan", "config"):
        name = "plans/videojev-test.json" if damage == "plan" else "configs/videojev.json"
        manifest["files"].pop(name)
        if damage == "plan":
            # Reference edits do not change any request fingerprint.
            plan["questions"][0]["reference"] = "no"
            (tmp_path / name).write_text(json.dumps(plan))
    elif damage == "frame":
        manifest["files"].pop(plan["units"][0]["frames"][0]["path"])
    elif damage == "extra-plan":
        (tmp_path / "plans/videojev-validation.json").write_text(json.dumps(plan))
    elif damage == "split":
        manifest["split"] = "validation"
    elif damage == "count":
        manifest["questions"] += 1
    elif damage == "parent":
        manifest.pop("parent_bundle_sha256")
    else:
        plan["units"][-1]["request_sha256"] = "0" * 64
        name = "plans/videojev-test.json"
        (tmp_path / name).write_text(json.dumps(plan))
        manifest["files"][name] = file_hash(tmp_path / name)
    path.write_text(json.dumps(manifest))

    def never(*args):
        pytest.fail("invalid bundle reached service boundary")

    monkeypatch.setattr(gpu_sweep, "http_health", never)
    monkeypatch.setattr(gpu_sweep, "http_post", never)
    with pytest.raises(ValueError):
        gpu_sweep.main(["run", "--bundle", str(tmp_path), "--label", "invalid-live",
                        "--output", str(tmp_path / "run")])
    assert not (tmp_path / "run").exists()


def test_run_accepts_a_sealed_live_bundle(tmp_path, monkeypatch):
    from test_gpu_sweep import health, raw_row, YES, NO

    plan = write_live_bundle(tmp_path)
    calls = []
    monkeypatch.setattr(gpu_sweep, "http_health", lambda url: health("RTX 4090", "bf16"))

    def post(url, body):
        calls.append(body)
        probs = YES if body["session"] == "test/a" else NO
        return 200, {"model": CONFIG["model"], "revision": CONFIG["revision"],
                     "request_id": body["request_id"], "raw": [raw_row("fight", probs)],
                     "timings": {"server_elapsed_seconds": 0.1}}

    monkeypatch.setattr(gpu_sweep, "http_post", post)
    gpu_sweep.main(["run", "--bundle", str(tmp_path), "--label", "sealed-live",
                    "--output", str(tmp_path / "run"), "--warmup", "0"])
    assert {body["request_id"]: digest(body) for body in calls} == {
        unit["id"]: unit["request_sha256"] for unit in plan["units"]}
    assert all(body["fps"] == 2.0 for body in calls)
    result, rows = gpu_sweep.read_run(tmp_path / "run")
    assert result["window_step_seconds"] == 0.5
    assert result["summary"]["accuracy"] == 1.0 and len(rows) == len(plan["units"])
