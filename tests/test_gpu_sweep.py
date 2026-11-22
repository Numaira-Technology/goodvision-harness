"""GPU / precision sweep of the current VideoJev service: run and baseline compare.

Synthetic bundle (two sources, three units) and an in-process fake service; no
model, no network, no torch.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest
from PIL import Image

from tools.unified_benchmark import gpu_sweep
from tools.unified_benchmark.adapters import request_body
from tools.unified_benchmark.contract import digest, file_hash

CONFIG = json.loads(Path("tools/unified_benchmark/profiles/videojev.json").read_text())
EVENTS = ("fight", "smoking")


def jpeg(shade):
    out = io.BytesIO()
    Image.new("RGB", (64, 36), (shade, shade, shade)).save(out, format="JPEG")
    return out.getvalue()


def make_bundle(tmp_path):
    units = []
    for i, source in enumerate(("v1", "v1", "v2")):
        frames = []
        for k in range(2):
            data = jpeg(10 * i + k)
            name = f"f{i}{k}.jpg"
            (tmp_path / name).write_bytes(data)
            frames.append({"path": name, "sha256": hashlib.sha256(data).hexdigest(),
                           "time": float(k)})
        unit = {"id": f"u{i}", "source_id": source, "frames": frames, "targets": list(EVENTS)}
        unit["request_sha256"] = digest(request_body(CONFIG, unit, tmp_path)[1])
        units.append(unit)
    questions = [
        {"number": 1, "event": "fight", "reference": "yes", "unit_ids": ["u0", "u1"]},
        {"number": 2, "event": "smoking", "reference": "no", "unit_ids": ["u0", "u1"]},
        {"number": 3, "event": "fight", "reference": "no", "unit_ids": ["u2"]},
        {"number": 4, "event": "smoking", "reference": "no", "unit_ids": ["u2"]},
    ]
    return {"units": units, "questions": questions}


def raw_row(event, probs):
    index = max(range(3), key=probs.__getitem__)
    options = {q["id"]: q["options"] for q in CONFIG["question_template"]}["scene/" + event]
    return {"id": "scene/" + event, "task": "choice", "answer_index": index,
            "answer": options[index], "probabilities": probs}


def fake_service(probabilities, *, fail=(), stage=0.2):
    """probabilities[unit][event] -> [yes, no, unknown]."""
    def post(body):
        unit = body["request_id"]
        if unit in fail:
            raise TimeoutError("timed out")
        return 200, {
            "model": CONFIG["model"], "revision": CONFIG["revision"], "request_id": unit,
            "raw": [raw_row(e, p) for e, p in probabilities[unit].items()],
            "timings": {"server_elapsed_seconds": stage * 2, "visual_encode_seconds": stage,
                        "decision_seconds": stage / 2, "unrelated": 1},
        }
    return post


YES, NO = [0.8, 0.15, 0.05], [0.1, 0.85, 0.05]
TRUTHY = {"u0": {"fight": YES, "smoking": NO}, "u1": {"fight": NO, "smoking": NO},
          "u2": {"fight": NO, "smoking": NO}}


def test_run_scores_with_the_frozen_scorer_and_keeps_failures(tmp_path):
    plan = make_bundle(tmp_path)
    records = gpu_sweep.run_units(plan, CONFIG, tmp_path, fake_service(TRUTHY))
    summary = gpu_sweep.summarize(plan, CONFIG, records, wall_seconds=3.0)
    assert summary["accuracy"] == 1.0 and summary["failed_units"] == 0
    assert summary["stages"]["visual_encode_seconds"]["p50"] == pytest.approx(0.2)
    assert "unrelated" not in records[0]["timings"]

    records = gpu_sweep.run_units(plan, CONFIG, tmp_path, fake_service(TRUTHY, fail={"u2"}))
    summary = gpu_sweep.summarize(plan, CONFIG, records, wall_seconds=3.0)
    assert summary["failed_units"] == 1 and summary["incomplete_questions"] == 2
    assert summary["accuracy"] == 0.5  # failures stay in the denominator
    assert summary["client_seconds"]["n"] == 2


def test_run_refuses_a_changed_request(tmp_path):
    plan = make_bundle(tmp_path)
    plan["units"][0]["request_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="frozen request differs"):
        gpu_sweep.run_units(plan, CONFIG, tmp_path, fake_service(TRUTHY))


def health(gpu, quantization, **settings):
    return {"ready": True, "queue_policy": "latest-per-stream", "context": 32768,
            "max_pixels": 409600, "gpu_batch_size": settings.get("gpu_batch_size", 4),
            "runtime": {"gpu": gpu, "model": CONFIG["model"], "revision": CONFIG["revision"],
                        "attention": "flash_attention_2", "prefix_cache": True,
                        "torch_cpu_threads": 4, "quantization": {"mode": quantization},
                        "gpu_allocated_bytes_after_load": 9 if quantization == "bf16" else 6}}


def saved_run(tmp_path, plan, label, probs, service, stage):
    records = gpu_sweep.run_units(plan, CONFIG, tmp_path, fake_service(probs, stage=stage))
    out = tmp_path / label
    out.mkdir()
    (out / "units.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    run = {"label": label, "bundle_sha256": "b", "split": "test", "plan_sha256": digest(plan),
           "config_sha256": digest(CONFIG), "service": gpu_sweep.describe(service),
           "summary": gpu_sweep.summarize(plan, CONFIG, records, wall_seconds=stage * 10),
           "units_sha256": file_hash(out / "units.jsonl")}
    (out / "run.json").write_text(json.dumps(run))
    return out


def test_compare_against_the_4090_bf16_baseline(tmp_path):
    plan = make_bundle(tmp_path)
    drifted = json.loads(json.dumps(TRUTHY))
    drifted["u2"]["fight"] = [0.5, 0.45, 0.05]  # FP8 flips a no to a yes
    base = saved_run(tmp_path, plan, "4090-bf16", TRUTHY,
                     health("NVIDIA GeForce RTX 4090", "bf16"), 0.2)
    cand = saved_run(tmp_path, plan, "4090-fp8", drifted,
                     health("NVIDIA GeForce RTX 4090", "fp8"), 0.1)
    out = tmp_path / "cmp.json"
    gpu_sweep.main(["compare", "--baseline", str(base), "--candidate", str(cand),
                    "--output", str(out)])
    result = json.loads(out.read_text())
    assert result["baseline"]["is_default_baseline"] is True
    assert result["accuracy"]["correct_delta"] == -1
    assert result["accuracy"]["changed"] == [
        {"number": 3, "event": "fight", "baseline": "no", "candidate": "yes"}]
    assert result["answers"]["unit_answers_changed"] == 1
    assert result["answers"]["max_probability_drift"]["n"] == 6
    assert result["latency"]["visual_encode_seconds"]["p50_ratio"] == pytest.approx(0.5)
    assert result["memory_after_load_bytes"] == {"baseline": 9, "candidate": 6}


def test_compare_refuses_without_a_proper_baseline(tmp_path):
    plan = make_bundle(tmp_path)
    l40s = saved_run(tmp_path, plan, "l40s-bf16", TRUTHY, health("NVIDIA L40S", "bf16"), 0.2)
    fp8 = saved_run(tmp_path, plan, "l40s-fp8", TRUTHY, health("NVIDIA L40S", "fp8"), 0.1)
    b, bu = gpu_sweep.read_run(l40s)
    c, cu = gpu_sweep.read_run(fp8)
    with pytest.raises(ValueError, match="baseline must be RTX 4090"):
        gpu_sweep.compare_runs(b, c, bu, cu)
    assert gpu_sweep.compare_runs(b, c, bu, cu, allow_other_baseline=True)[
        "baseline"]["is_default_baseline"] is False


def test_compare_refuses_different_service_settings(tmp_path):
    plan = make_bundle(tmp_path)
    base = saved_run(tmp_path, plan, "base", TRUTHY, health("RTX 4090", "bf16"), 0.2)
    other = saved_run(tmp_path, plan, "other", TRUTHY,
                      health("RTX 4090", "fp8", gpu_batch_size=2), 0.1)
    b, bu = gpu_sweep.read_run(base)
    c, cu = gpu_sweep.read_run(other)
    with pytest.raises(ValueError, match="settings differ"):
        gpu_sweep.compare_runs(b, c, bu, cu)


def test_read_run_detects_edited_units(tmp_path):
    plan = make_bundle(tmp_path)
    base = saved_run(tmp_path, plan, "base", TRUTHY, health("RTX 4090", "bf16"), 0.2)
    with (base / "units.jsonl").open("a") as f:
        f.write("{}\n")
    with pytest.raises(ValueError, match="differs"):
        gpu_sweep.read_run(base)
