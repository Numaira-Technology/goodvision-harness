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

from tools.videojev_probe import gpu_sweep
from tools.unified_benchmark.adapters import request_body
from tools.unified_benchmark.contract import digest, file_hash, source_files

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
    versions = {"torch": "2.8.0+cu128", "transformers": "5.14.1",
                "flash-attn": "2.8.3.post2"}
    if quantization == "fp8":
        versions["torchao"] = "0.13.0"
    return {
        "ready": True, "queue_policy": "latest-per-stream", "context": 32768,
        "max_pixels": 409600, "gpu_batch_size": 4, "ready_capacity": 4,
        "queue_age": 19.5, "queue_capacity": 48, "request_timeout": 20,
        "requests_received": 0, "model": CONFIG["model"], "revision": CONFIG["revision"],
        "runtime": {
            "gpu": gpu, "model": CONFIG["model"], "revision": CONFIG["revision"],
            "attention": "flash_attention_2", "prefix_cache": True, "cuda_runtime": "12.8",
            "torch_cpu_threads": 4, "torch_interop_threads": 4,
            "quantization": {"mode": quantization}, "versions": versions,
            "gpu_allocated_bytes_after_load": 9 if quantization == "bf16" else 6,
        },
        **settings,
    }


def saved_run(tmp_path, plan, label, probs, service, stage):
    records = gpu_sweep.run_units(plan, CONFIG, tmp_path, fake_service(probs, stage=stage))
    out = tmp_path / label
    out.mkdir()
    (out / "units.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    run = {"label": label, "bundle_sha256": "b", "split": "test", "plan_sha256": digest(plan),
           "config_sha256": digest(CONFIG), "sweep_sha256": "sweep-a",
           "service": gpu_sweep.describe(service), "warmup_seconds": [stage] * 3,
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


@pytest.mark.parametrize(("setting", "value"), [
    ("gpu_batch_size", 2), ("ready_capacity", 2), ("queue_age", 9.5),
    ("queue_capacity", 24), ("request_timeout", 10), ("context", 8192),
])
def test_compare_refuses_different_service_settings(tmp_path, setting, value):
    plan = make_bundle(tmp_path)
    base = saved_run(tmp_path, plan, "base", TRUTHY, health("RTX 4090", "bf16"), 0.2)
    other = saved_run(tmp_path, plan, "other", TRUTHY,
                      health("RTX 4090", "fp8", **{setting: value}), 0.1)
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


@pytest.mark.parametrize(("setting", "value", "message"), [
    ("versions", {"torch": "wrong"}, "dependency versions differ"),
    ("cuda_runtime", "12.9", "cuda_runtime differ"),
    ("torch_interop_threads", 2, "runtime differ"),
])
def test_compare_refuses_different_service_environment(tmp_path, setting, value, message):
    plan = make_bundle(tmp_path)
    base = saved_run(tmp_path, plan, "base", TRUTHY, health("RTX 4090", "bf16"), 0.2)
    candidate = health("RTX 4090", "fp8")
    candidate["runtime"][setting] = value
    other = saved_run(tmp_path, plan, "other", TRUTHY, candidate, 0.1)
    b, bu = gpu_sweep.read_run(base)
    c, cu = gpu_sweep.read_run(other)
    with pytest.raises(ValueError, match=message):
        gpu_sweep.compare_runs(b, c, bu, cu)


@pytest.mark.parametrize("damage", [
    lambda response: response.pop("raw"),
    lambda response: response.update(timings=[0.1]),
    lambda response: response["timings"].update(decision_seconds=float("nan")),
    lambda response: response["timings"].update(decision_seconds=True),
])
def test_malformed_response_does_not_abort_or_count_as_success(tmp_path, damage):
    plan = make_bundle(tmp_path)
    respond = fake_service(TRUTHY)

    def post(body):
        status, response = respond(body)
        if body["request_id"] == "u0":
            damage(response)
        return status, response

    rows = gpu_sweep.run_units(plan, CONFIG, tmp_path, post)
    assert len(rows) == 3 and rows[1]["complete"] and rows[2]["complete"]
    assert rows[0]["error"].startswith("invalid response:") and not rows[0]["complete"]
    summary = gpu_sweep.summarize(plan, CONFIG, rows, 1.0)
    assert summary["failed_units"] == 1 and summary["incomplete_questions"] == 2
    assert summary["planned_questions"] == 4 and summary["accuracy"] == 0.5


def run_args(tmp_path, *extra):
    return ["run", "--bundle", str(tmp_path), "--label", "synthetic",
            "--output", str(tmp_path / "run"), *extra]


def synthetic_sweep(tmp_path, monkeypatch, snapshot):
    plan = make_bundle(tmp_path)
    (tmp_path / "plans").mkdir()
    (tmp_path / "plans/videojev-test.json").write_text(json.dumps(plan))
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/videojev.json").write_text(json.dumps(CONFIG))
    monkeypatch.setattr(gpu_sweep, "verify_bundle", lambda bundle: "sealed-synthetic-bundle")
    monkeypatch.setattr(gpu_sweep, "http_health", lambda url: snapshot)
    calls = []
    respond = fake_service({**TRUTHY, **{f"warmup-{i}": TRUTHY["u0"] for i in range(3)}})

    def post(url, body):
        assert url == "http://127.0.0.1:8093"
        calls.append(body)
        return respond(body)

    monkeypatch.setattr(gpu_sweep, "http_post", post)
    return plan, calls


@pytest.mark.parametrize(("alter", "message"), [
    (lambda snapshot: snapshot.update(model="not-videojev"), "model/revision differs"),
    (lambda snapshot: snapshot["runtime"].update(revision="wrong"), "model/revision differs"),
    (lambda snapshot: snapshot.update(requests_received=1), "fresh, dedicated"),
    (lambda snapshot: snapshot.pop("requests_received"), "fresh, dedicated"),
    (lambda snapshot: snapshot.pop("ready_capacity"), "sweep settings"),
    (lambda snapshot: snapshot["runtime"].pop("quantization"), "sweep settings"),
])
def test_run_cli_refuses_before_sending_any_frames(tmp_path, monkeypatch, alter, message):
    snapshot = health("RTX 4090", "bf16")
    alter(snapshot)
    _, calls = synthetic_sweep(tmp_path, monkeypatch, snapshot)
    with pytest.raises(ValueError, match=message):
        gpu_sweep.main(run_args(tmp_path))
    assert calls == [] and not (tmp_path / "run").exists()


def test_run_cli_seals_sweep_and_sends_exact_frozen_requests(tmp_path, monkeypatch):
    plan, calls = synthetic_sweep(tmp_path, monkeypatch, health("RTX 4090", "bf16"))
    gpu_sweep.main(run_args(tmp_path))
    assert [(body["request_id"], body["session"]) for body in calls[:3]] == [
        (f"warmup-{i}", f"warmup-{i}") for i in range(3)
    ]
    assert {body["request_id"]: digest(body) for body in calls[3:]} == {
        unit["id"]: unit["request_sha256"] for unit in plan["units"]
    }
    result, rows = gpu_sweep.read_run(tmp_path / "run")
    assert len(result["warmup_seconds"]) == len(rows) == 3
    assert result["summary"]["accuracy"] == 1.0
    assert result["sweep_sha256"] == file_hash(gpu_sweep.__file__)


@pytest.mark.parametrize("url", [
    "http://192.0.2.10:8093", "http://127.0.0.1:8093/predict",
    "https://127.0.0.1:8093", "http://localhost:8093", "http://fake@127.0.0.1:8093",
])
def test_run_refuses_nonlocal_or_ambiguous_destination_before_network(tmp_path, monkeypatch, url):
    def never(*args):
        pytest.fail("invalid destination reached a network/bundle boundary")

    monkeypatch.setattr(gpu_sweep, "http_health", never)
    monkeypatch.setattr(gpu_sweep, "verify_bundle", never)
    with pytest.raises(ValueError, match="loopback"):
        gpu_sweep.main(run_args(tmp_path, "--url", url))


def test_sweep_is_not_part_of_historical_frozen_source_inventory():
    assert Path(gpu_sweep.__file__).resolve() not in set(source_files().values())


def test_http_opener_does_not_follow_redirects_or_use_proxies():
    from urllib.request import HTTPRedirectHandler, ProxyHandler

    redirects = [h for h in gpu_sweep._HTTP.handlers if isinstance(h, HTTPRedirectHandler)]
    assert len(redirects) == 1
    assert redirects[0].redirect_request(None, None, 307, "", {}, "outside") is None
    assert not any(isinstance(h, ProxyHandler) and h.proxies for h in gpu_sweep._HTTP.handlers)


def test_compare_rejects_runs_from_a_different_sweep_revision(tmp_path):
    plan = make_bundle(tmp_path)
    base = saved_run(tmp_path, plan, "base", TRUTHY, health("RTX 4090", "bf16"), 0.2)
    other = saved_run(tmp_path, plan, "other", TRUTHY, health("RTX 4090", "fp8"), 0.1)
    b, bu = gpu_sweep.read_run(base)
    c, cu = gpu_sweep.read_run(other)
    c["sweep_sha256"] = "sweep-b"
    with pytest.raises(ValueError, match="sweep_sha256"):
        gpu_sweep.compare_runs(b, c, bu, cu)


@pytest.mark.parametrize(("mutate", "message"), [
    (lambda u: u.update(request_sha256="0" * 64), "frozen request differs"),
    (lambda u: u.update(source_id="warmup-0"), "warm-up IDs collide"),
])
def test_warmup_checks_frozen_input_and_collision_before_sending(tmp_path, mutate, message):
    plan = make_bundle(tmp_path)
    unit = plan["units"][1] if message.endswith("collide") else plan["units"][0]
    mutate(unit)
    with pytest.raises(ValueError, match=message):
        gpu_sweep.warm_up(plan, CONFIG, tmp_path, lambda _: pytest.fail("sent data"), 1)


def test_compare_requires_same_warmup_count(tmp_path):
    plan = make_bundle(tmp_path)
    base = saved_run(tmp_path, plan, "base", TRUTHY, health("RTX 4090", "bf16"), 0.2)
    other = saved_run(tmp_path, plan, "other", TRUTHY, health("RTX 4090", "fp8"), 0.1)
    b, bu = gpu_sweep.read_run(base)
    c, cu = gpu_sweep.read_run(other)
    c["warmup_seconds"].pop()
    with pytest.raises(ValueError, match="warm-up counts differ"):
        gpu_sweep.compare_runs(b, c, bu, cu)


def test_zero_warmups_do_not_construct_or_send_a_request():
    assert gpu_sweep.warm_up({}, None, None, lambda _: pytest.fail("sent data"), 0) == []


def compiled(service, mode):
    service["runtime"]["compile"] = {"mode": mode}
    return service


def test_declared_batch_variation_is_allowed_and_recorded(tmp_path):
    plan = make_bundle(tmp_path)
    base = saved_run(tmp_path, plan, "b4", TRUTHY, health("RTX 4090", "bf16"), 0.2)
    big = saved_run(tmp_path, plan, "b8", TRUTHY,
                    health("RTX 4090", "bf16", gpu_batch_size=8, ready_capacity=8), 0.1)
    b, bu = gpu_sweep.read_run(base)
    c, cu = gpu_sweep.read_run(big)
    with pytest.raises(ValueError, match="settings differ"):
        gpu_sweep.compare_runs(b, c, bu, cu)
    result = gpu_sweep.compare_runs(b, c, bu, cu, vary=("gpu_batch_size", "ready_capacity"))
    assert result["declared_variations"] == ["gpu_batch_size", "ready_capacity"]
    assert result["candidate"]["settings"]["gpu_batch_size"] == 8
    with pytest.raises(ValueError, match="settings differ"):  # undeclared one still refused
        gpu_sweep.compare_runs(b, c, bu, cu, vary=("gpu_batch_size",))
    with pytest.raises(ValueError, match="may be varied"):
        gpu_sweep.compare_runs(b, c, bu, cu, vary=("queue_age",))


def test_compile_is_recorded_and_a_compiled_run_is_not_the_default_baseline(tmp_path):
    plan = make_bundle(tmp_path)
    base = saved_run(tmp_path, plan, "eager", TRUTHY, health("RTX 4090", "bf16"), 0.2)
    comp = saved_run(tmp_path, plan, "mlp", TRUTHY,
                     compiled(health("RTX 4090", "bf16"), "mlp"), 0.1)
    b, bu = gpu_sweep.read_run(base)
    c, cu = gpu_sweep.read_run(comp)
    result = gpu_sweep.compare_runs(b, c, bu, cu)
    assert (result["baseline"]["compile"], result["candidate"]["compile"]) == ("off", "mlp")
    with pytest.raises(ValueError, match="baseline must be RTX 4090"):
        gpu_sweep.compare_runs(c, b, cu, bu)
