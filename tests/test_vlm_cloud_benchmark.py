import io
import json
import os
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.vlm_benchmark import cloud, lock, run_remote
from tools.vlm_benchmark.plan import (
    availability_gate,
    concurrency_comparison,
    gate,
    make_plan,
    ramp_jobs,
    request_budget,
    soak_allowed,
)
from tools.vlm_benchmark.report import render

ROOT = Path(__file__).resolve().parents[1]
requires_posix = pytest.mark.skipif(
    os.name == "nt", reason="Linux cloud execution / POSIX SSH transport"
)


def test_windows_can_import_planning_but_cannot_bypass_suite_lock(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(lock, "fcntl", None)
    assert make_plan()["streams"] == [1, 4, 8, 12]
    with pytest.raises(RuntimeError, match="POSIX file locking"):
        lock.acquire(tmp_path)


def test_slow_valid_probe_is_available_but_not_performance_pass():
    data = stage()
    data["per_stream"][0].update(on_time_coverage=0, on_time_valid_results=0)
    assert availability_gate(data)["passed"]
    assert not gate(data, make_plan())["passed"]
    data["per_stream"][0]["complete_valid_results"] = 0
    assert not availability_gate(data)["passed"]


def stage(count=1):
    row = {
        "expected_observations": 10,
        "complete_valid_results": 10,
        "on_time_valid_results": 10,
        "on_time_coverage": 1,
        "uncertain_judgment_fraction": 0,
        "errors": {},
        "source_to_publish_seconds": {"p95": 1},
    }
    return {
        "streams": count,
        "all_drained": True,
        "input_timing_valid": True,
        "per_stream": [dict(row) for _ in range(count)],
    }


def test_plan_is_bounded_and_comparison_is_interleaved():
    plan = make_plan(compare=True)
    assert [j["variant"] for j in ramp_jobs(plan)[:4]] == [
        "unlimited",
        "shared",
        "shared",
        "unlimited",
    ]
    assert request_budget(make_plan()) == {
        "probe_upper": 1,
        "ramp_upper": 1900,
        "conditional_soak_upper": 13500,
        "paid_requests": True,
    }
    for overrides in (
        {"soak_seconds": 36000},
        {"streams": [12, 1]},
        {"duration": float("nan")},
        {"per_stream_limit": 8},
        {"streams": [1]},
        {"soak_seconds": 1},
    ):
        with pytest.raises(ValueError):
            make_plan(**overrides)


@pytest.mark.parametrize(
    "failure",
    ["missing_lane", "missing_records", "late", "error", "null", "input", "drain"],
)
def test_every_lane_and_missing_work_affect_gate(failure):
    data = stage(12)
    if failure == "missing_lane":
        data["per_stream"].pop()
    elif failure == "missing_records":
        data["result_records_missing"] = 1
    elif failure == "late":
        data["per_stream"][-1]["on_time_coverage"] = 0.5
    elif failure == "error":
        data["per_stream"][-1]["errors"] = {"timeout": 1}
    elif failure == "null":
        data["per_stream"][-1]["uncertain_judgment_fraction"] = 1
    elif failure == "input":
        data["input_timing_valid"] = False
    else:
        data["all_drained"] = False
    assert not gate(data, make_plan())["passed"]


def test_soak_requires_all_twelve_stream_shared_repeats():
    plan = make_plan()
    jobs = [{**j, "gate": {"passed": True}} for j in ramp_jobs(plan)]
    assert soak_allowed(jobs, plan)
    assert not soak_allowed(jobs[:-1], plan)
    jobs[-1]["gate"]["passed"] = False
    assert not soak_allowed(jobs, plan)


def test_archive_and_integrity_reject_unsafe_content(tmp_path):
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        member = tarfile.TarInfo("../escape")
        member.size = 1
        tar.addfile(member, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="unsafe"):
        run_remote.safe_extract(archive, tmp_path / "target")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "file").write_text("x", encoding="utf-8")
    app = tmp_path / "app"
    app.mkdir()
    (app / "linked").symlink_to(outside, target_is_directory=True)
    (app / "bundle-manifest.json").write_text(
        json.dumps({"files": {"linked/file": cloud.sha256(outside / "file")}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="integrity"):
        cloud.verify_bundle(app)


@requires_posix
def test_instance_lock_prevents_overlapping_suites(tmp_path):
    roots = [tmp_path / name / "app" for name in ("one", "two")]
    for root in roots:
        root.mkdir(parents=True)
    first = lock.acquire(roots[0])
    try:
        with pytest.raises(RuntimeError, match="another VLM"):
            lock.acquire(roots[1])
    finally:
        os.close(first)
    os.close(lock.acquire(roots[1]))


@requires_posix
def test_ssh_secret_only_uses_stdin_and_arguments_are_quoted(tmp_path, monkeypatch):
    identity = tmp_path / "test-key"
    identity.write_text("not-a-real-key", encoding="utf-8")
    args = SimpleNamespace(host="127.0.0.1", port=1234, identity=identity)
    ssh = run_remote.SSH(args)
    calls = []
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **kw: calls.append((a, kw)) or SimpleNamespace(stdout="done"),
    )
    try:
        secret = "private-test-sentinel"
        ssh.call("print('safe')", "a b; $(no)", data=secret)
        argv, kwargs = calls[0][0][0], calls[0][1]
        assert secret not in repr(argv)
        assert kwargs["input"] == secret
        assert shlex.split(argv[-1]) == ["python3", "-c", "print('safe')", "a b; $(no)"]
    finally:
        ssh.close()
    args.host = "host;unexpected"
    with pytest.raises(ValueError):
        run_remote.SSH.validate(args)


def test_html_escapes_untrusted_model_and_failure(tmp_path):
    render(
        tmp_path,
        {
            "plan": make_plan(simulate=True),
            "state": "failed",
            "jobs": [],
            "model": "<script>alert(1)</script>",
            "reason": "<img src=x>",
        },
    )
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "<script>" not in html and "&lt;script&gt;" in html
    assert "<img" not in html
    assert "SIMULATED" not in html  # Chinese explicit simulation heading


@pytest.fixture
def packaged(tmp_path):
    import av
    from PIL import Image

    video = tmp_path / "sample.mp4"
    with av.open(str(video), "w") as output:
        stream = output.add_stream("mpeg4", rate=10)
        stream.width, stream.height, stream.pix_fmt = 80, 64, "yuv420p"
        for index in range(30):
            frame = av.VideoFrame.from_image(
                Image.new("RGB", (80, 64), (index, 80, 90))
            )
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)
    archive = tmp_path / "bundle.tar.gz"
    manifest = run_remote.bundle(ROOT, [video], archive)
    app = tmp_path / "job/app"
    run_remote.safe_extract(archive, app)
    return app, manifest


def test_bundle_has_no_secrets_and_detects_modified_code(packaged):
    app, manifest = packaged
    assert "config.yaml" not in manifest["files"]
    assert not any(
        ".env" in p or "id_ed25519" in p or "checkpoint" in p for p in manifest["files"]
    )
    assert cloud.verify_bundle(app) == manifest
    (app / "tools/vlm_multistream_benchmark.py").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        cloud.verify_bundle(app)


def test_video_reuse_uploads_only_code_and_verifies_remote_copy(packaged, monkeypatch):
    app, manifest = packaged
    job = app.parent.parent / "reuse-job"
    job.mkdir()
    archive = job / "bundle.tar.gz"
    run_remote.bundle(ROOT, [app / manifest["videos"][0]], archive, include_video=False)
    with tarfile.open(archive) as tar:
        assert not any(n.startswith("media/") for n in tar.getnames())
    monkeypatch.setattr(sys, "argv", ["launch", str(job), str(app.parent)])
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"key":"test-only"}'))
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: SimpleNamespace(pid=123))
    old_mask = os.umask(0o077)
    try:
        exec(run_remote.LAUNCH, {})
    finally:
        os.umask(old_mask)
    copied = cloud.verify_bundle(job / "app")
    assert (
        copied["files"][manifest["videos"][0]]
        == manifest["files"][manifest["videos"][0]]
    )


@pytest.mark.parametrize("compare_concurrency", [False, True, "arrival", "routing"])
@requires_posix
def test_extracted_package_runs_production_branches_without_api(
    packaged, compare_concurrency
):
    app, _ = packaged
    job = app.parent
    plan = make_plan(
        simulate=True, streams=[1, 2], duration=3, repeats=1, soak_seconds=0
    )
    if compare_concurrency:
        plan = make_plan(
            simulate=True,
            provider="openrouter",
            event_json_mode="schema",
            streams=[1],
            duration=3,
            repeats=2,
            soak_seconds=0,
            compare_concurrency=True,
        )
    if compare_concurrency == "arrival":
        plan = make_plan(
            simulate=True,
            streams=[2],
            duration=3,
            repeats=2,
            soak_seconds=0,
            compare_arrival=True,
        )
    if compare_concurrency == "routing":
        plan = make_plan(
            simulate=True,
            provider="openrouter",
            model="qwen/qwen3.5-35b-a3b",
            event_json_mode="schema",
            streams=[2],
            duration=3,
            repeats=2,
            soak_seconds=0,
            compare_routing=True,
        )
    (job / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    secret = job / "runtime-key.json"
    secret.write_text(json.dumps({"key": "private-test-sentinel"}), encoding="utf-8")
    env = {
        **os.environ,
        "PYTHONPATH": str(app / "src") + os.pathsep + str(app),
        "PYTHON_DOTENV_DISABLED": "1",
    }
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.vlm_benchmark.cloud",
            "--root",
            str(app),
            "--plan",
            str(job / "plan.json"),
            "--output",
            str(job / "results"),
            "--secret",
            str(secret),
        ],
        cwd=app,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    summary = json.loads((job / "results/summary.json").read_text(encoding="utf-8"))
    assert result.returncode == 0, (result.stdout, result.stderr, summary)
    assert summary["state"] == "completed"
    assert len(summary["jobs"]) == (5 if compare_concurrency else 3)
    assert all(j["gate"]["passed"] for j in summary["jobs"])
    assert not summary["engineering_capacity_gate_passed"]
    assert not secret.exists()
    assert (job / "results/telemetry.jsonl").is_file()
    manifest = json.loads((job / "results/manifest.json").read_text(encoding="utf-8"))
    assert manifest["effective_inference"]["output_mode"] == "events"
    assert manifest["effective_inference"]["max_output_tokens"] == 160
    if compare_concurrency == "routing":
        assert [
            j["effective_request_parameters"]["provider"]["sort"]
            for j in summary["jobs"][1:]
        ] == ["throughput", "latency", "latency", "throughput"]
        assert all(
            j["effective_request_parameters"]["response_format"]["type"]
            == "json_schema"
            for j in summary["jobs"]
        )
    if compare_concurrency is True:
        assert [
            j["result"]["shared_admission"]["per_stream_limit"]
            for j in summary["jobs"][1:]
        ] == [2, 4, 4, 2]
        assert (
            manifest["effective_inference"]["request_parameters"]["response_format"][
                "type"
            ]
            == "json_schema"
        )
        assert all(a["complete"] for a in summary["concurrency_comparison"])
    if compare_concurrency == "arrival":
        trials = summary["jobs"][1:]
        assert [j["result"]["arrival_mode"] for j in trials] == [
            "aligned",
            "staggered",
            "staggered",
            "aligned",
        ]
        assert all(len(j["result"]["per_stream"]) == 2 for j in trials)
        assert (
            len(
                {
                    r["expected_observations"]
                    for j in trials
                    for r in j["result"]["per_stream"]
                }
            )
            == 1
        )
        assert trials[1]["result"]["source_phase_offsets_seconds"] == [0, 0.4]
        assert trials[1]["result"]["input_schedule_span_seconds"] == 3.4
    assert "private-test-sentinel" not in (
        result.stdout + result.stderr + json.dumps(summary)
    )


@requires_posix
def test_failed_probe_stops_before_paid_ramp_and_removes_key(packaged, monkeypatch):
    from tools import vlm_multistream_benchmark as bench

    app, _ = packaged
    secret = app.parent / "key.json"
    secret.write_text(json.dumps({"key": "private-test-sentinel"}), encoding="utf-8")
    monkeypatch.setattr(os, "environ", dict(os.environ))
    calls = []

    def failed(*args):
        calls.append(args)
        data = stage()
        data["per_stream"][0].update(
            complete_valid_results=0, on_time_coverage=0, errors={"model_error": 1}
        )
        return data

    monkeypatch.setattr(bench, "run_stage", failed)
    assert cloud.run(app, make_plan(), app.parent / "results", secret) == 2
    assert len(calls) == 1
    assert not secret.exists()
    summary = json.loads(
        (app.parent / "results/summary.json").read_text(encoding="utf-8")
    )
    assert summary["state"] == "failed"
    assert "private-test-sentinel" not in json.dumps(summary)


def test_remote_process_crash_persists_failed_report(packaged):
    app, _ = packaged
    job = app.parent
    (job / "plan.json").write_text(json.dumps(make_plan()), encoding="utf-8")
    (job / "control.json").write_text(json.dumps({"pid": 999999999}), encoding="utf-8")
    (job / "runtime-key.json").write_text("test-secret", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, "-c", run_remote.STATUS, str(job)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert '"state": "failed"' in result.stdout
    assert (
        json.loads((job / "status.json").read_text(encoding="utf-8"))["state"]
        == "failed"
    )
    assert not json.loads((job / "results/summary.json").read_text(encoding="utf-8"))[
        "engineering_capacity_gate_passed"
    ]
    assert not (job / "runtime-key.json").exists()


def concurrency_plan(**changes):
    return make_plan(
        **{
            "streams": [1],
            "soak_seconds": 0,
            "compare_concurrency": True,
            "provider": "openrouter",
            "event_json_mode": "schema",
            **changes,
        }
    )


def test_arrival_plan_freezes_abba_and_refuses_confounded_comparisons():
    plan = make_plan(
        streams=[8],
        soak_seconds=0,
        compare_arrival=True,
        per_stream_limit=4,
        global_limit=32,
    )
    jobs = ramp_jobs(plan)
    assert [(j["arrival_mode"], j["repeat"]) for j in jobs] == [
        ("aligned", 1),
        ("staggered", 1),
        ("staggered", 2),
        ("aligned", 2),
    ]
    assert request_budget(plan)["ramp_upper"] == 1216
    assert request_budget(plan)["conditional_soak_upper"] == 0
    for changes in (
        {"compare": True},
        {"compare_concurrency": True},
        {"streams": [1]},
        {"streams": [2, 4]},
        {"repeats": 1},
        {"arrival_mode": "invalid"},
    ):
        with pytest.raises(ValueError):
            make_plan(**{**plan, **changes})


def test_concurrency_experiment_is_bounded_abba_and_partial_arms_cannot_pass(tmp_path):
    plan = concurrency_plan()
    jobs = ramp_jobs(plan)
    assert [(j["per_stream_limit"], j["repeat"]) for j in jobs] == [
        (2, 1),
        (4, 1),
        (4, 2),
        (2, 2),
    ]
    assert request_budget(plan) == {
        "probe_upper": 1,
        "ramp_upper": 152,
        "conditional_soak_upper": 0,
        "paid_requests": True,
    }
    completed = [{**j, "result": stage(), "gate": gate(stage(), plan)} for j in jobs]
    comparison = concurrency_comparison(completed[:-1], plan)
    assert not comparison[0]["complete"] and not comparison[0]["all_trials_passed"]
    assert comparison[1]["complete"] and comparison[1]["all_trials_passed"]
    completed[-1]["result"]["per_stream"][0].update(
        complete_valid_results=8,
        on_time_valid_results=7,
        on_time_coverage=0.7,
        errors={"invalid_response": 1, "timeout": 1},
    )
    completed[-1]["gate"] = gate(completed[-1]["result"], plan)
    comparison = concurrency_comparison(completed, plan)
    assert comparison[0]["complete"] and not comparison[0]["all_trials_passed"]
    assert comparison[0]["on_time_coverage"] == 17 / 20
    assert comparison[0]["errors"] == {"invalid_response": 1, "timeout": 1}
    render(
        tmp_path,
        {
            "plan": plan,
            "state": "completed",
            "jobs": completed,
            "concurrency_comparison": comparison,
        },
    )
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "2 → 4 → 4 → 2" in html and "85.0%" in html
    assert "至少一轮不符合门槛" in html and "两轮均符合门槛" in html


@pytest.mark.parametrize(
    "override",
    [
        {"streams": [1, 4]},
        {"repeats": 1},
        {"soak_seconds": 900},
        {"compare": True},
        {"global_limit": 2},
        {"compare_concurrency": "yes"},
        {"event_json_mode": "anything"},
        {"provider": "aiclub"},
    ],
)
def test_concurrency_experiment_rejects_confounded_plan(override):
    with pytest.raises(ValueError):
        concurrency_plan(**override)


@requires_posix
def test_cloud_changes_only_per_stream_limit_between_abba_trials(packaged, monkeypatch):
    from tools import vlm_multistream_benchmark as bench

    app, _ = packaged
    monkeypatch.setattr(os, "environ", dict(os.environ))
    calls = []

    def capture(args, base, count, framesets, target, repeat):
        calls.append(
            (vars(args), dict(base.inference.request_parameters), count, framesets)
        )
        return stage()

    monkeypatch.setattr(bench, "run_stage", capture)
    assert cloud.run(app, concurrency_plan(simulate=True), app.parent / "results") == 0
    trials = calls[1:]
    assert [a[0]["per_stream_limit"] for a in trials] == [2, 4, 4, 2]
    frozen = [
        {k: v for k, v in a[0].items() if k != "per_stream_limit"} for a in trials
    ]
    assert all(a == frozen[0] for a in frozen)
    assert all(
        a[1] == trials[0][1] and a[2] == 1 and a[3] is trials[0][3] for a in trials
    )
    assert frozen[0]["observation_fps"] == 1.25
    assert frozen[0]["deadline"] == 3 and frozen[0]["local_concurrency"] == 4
    assert len(calls) == 5  # one probe, four trials; no soak or twelve-stream claim


@pytest.mark.parametrize("auth_failure", [False, True])
@requires_posix
def test_ab_measures_the_other_arm_after_zero_results_but_stops_on_auth(
    packaged, monkeypatch, auth_failure
):
    from tools import vlm_multistream_benchmark as bench

    app, _ = packaged
    monkeypatch.setattr(os, "environ", dict(os.environ))
    calls = []

    def measure(args, *unused):
        calls.append(args.per_stream_limit)
        result = stage()
        if len(calls) > 1 and args.per_stream_limit == 2:
            result["per_stream"][0].update(
                complete_valid_results=0,
                on_time_valid_results=0,
                on_time_coverage=0,
                errors={"http_error" if auth_failure else "timeout": 10},
                http_status_counts={"401": 10} if auth_failure else {},
            )
        return result

    monkeypatch.setattr(bench, "run_stage", measure)
    exit_code = cloud.run(app, concurrency_plan(simulate=True), app.parent / "results")
    summary = json.loads(
        (app.parent / "results/summary.json").read_text(encoding="utf-8")
    )
    arms = summary["concurrency_comparison"]
    assert not arms[0]["all_trials_passed"]
    if auth_failure:
        assert exit_code == 2 and calls == [2, 2]
        assert not arms[1]["complete"]
    else:
        assert exit_code == 0 and calls == [2, 2, 4, 4, 2]
        assert arms[0]["on_time_coverage"] == 0
        assert arms[1]["all_trials_passed"]
        assert not summary["engineering_capacity_gate_passed"]


def routing_plan(**overrides):
    return make_plan(
        **{
            "provider": "openrouter",
            "model": "qwen/qwen3.5-35b-a3b",
            "event_json_mode": "schema",
            "streams": [8],
            "global_limit": 32,
            "per_stream_limit": 4,
            "soak_seconds": 0,
            "compare_routing": True,
            **overrides,
        }
    )


@pytest.mark.parametrize(
    "override",
    [
        {"provider": "aiclub"},
        {"model": "qwen/qwen3.5-35b-a3b:nitro"},
        {"compare_arrival": True},
        {"compare_concurrency": True},
        {"compare": True},
        {"streams": [4, 8]},
        {"repeats": 1},
        {"provider_sort": "price"},
        {"compare_routing": "true"},
    ],
)
def test_routing_comparison_rejects_ambiguous_or_confounded_plans(override):
    with pytest.raises(ValueError):
        routing_plan(**override)


@requires_posix
def test_routing_comparison_changes_only_provider_sort(packaged, monkeypatch):
    from tools import vlm_multistream_benchmark as bench

    app, _ = packaged
    monkeypatch.setattr(os, "environ", dict(os.environ))
    calls = []

    def capture(args, base, count, framesets, target, repeat):
        calls.append(
            (
                vars(args),
                base.inference.model,
                dict(base.inference.request_parameters),
                count,
                framesets,
            )
        )
        return stage(count)

    monkeypatch.setattr(bench, "run_stage", capture)
    plan = routing_plan(simulate=True)
    assert request_budget(plan)["ramp_upper"] == 1216
    assert cloud.run(app, plan, app.parent / "results") == 0
    trials = calls[1:]
    assert len(trials) == 4
    assert [x[2]["provider"]["sort"] for x in trials] == [
        "throughput",
        "latency",
        "latency",
        "throughput",
    ]
    assert all(
        x[0] == trials[0][0]
        and x[1] == plan["model"]
        and x[3] == 8
        and x[4] is trials[0][4]
        for x in trials
    )
    for x in trials:
        x[2]["provider"].pop("sort")
    assert all(x[2] == trials[0][2] for x in trials)
    assert trials[0][2]["provider"]["require_parameters"] is True
    assert trials[0][0]["observation_fps"] == 1.25
    assert trials[0][0]["deadline"] == 3


@pytest.mark.parametrize(
    "model", ["qwen/qwen3.5-35b-a3b", "qwen/qwen3.5-35b-a3b:nitro"]
)
def test_model_alias_keeps_frozen_non_thinking_request_policy(monkeypatch, model):
    from tools import vlm_multistream_benchmark as bench

    monkeypatch.setenv("INFERENCE_PROVIDER", "openrouter")
    monkeypatch.setenv("INFERENCE_MODEL", model)
    base = cloud.configure_event_json(
        bench.load_config(ROOT / "tools/vlm_benchmark/config.json"), "schema"
    )
    parameters = base.inference.request_parameters
    assert parameters["reasoning"] == {"enabled": False}
    assert parameters["provider"] == {
        "sort": "throughput",
        "allow_fallbacks": True,
        "ignore": ["coreweave", "novita"],
        "require_parameters": True,
    }
    assert base.inference.model == model


@requires_posix
def test_fixed_provider_is_applied_to_probe_and_trials(packaged, monkeypatch):
    from tools import vlm_multistream_benchmark as bench

    app, _ = packaged
    monkeypatch.setattr(os, "environ", dict(os.environ))
    calls = []

    def capture(args, base, count, framesets, target, repeat):
        calls.append(dict(base.inference.request_parameters))
        return stage(count)

    monkeypatch.setattr(bench, "run_stage", capture)
    plan = make_plan(
        simulate=True,
        provider="openrouter",
        model="qwen/qwen3.5-35b-a3b",
        event_json_mode="schema",
        provider_only="deepinfra",
        streams=[1],
        duration=3,
        repeats=1,
        soak_seconds=0,
    )
    assert cloud.run(app, plan, app.parent / "results") == 0
    assert len(calls) == 2
    for p in calls:
        assert p["provider"]["only"] == ["deepinfra"]
        assert p["provider"]["allow_fallbacks"] is False
        assert p["provider"]["require_parameters"] is True
        assert p["reasoning"] == {"enabled": False}
        assert p["response_format"]["type"] == "json_schema"
    for overrides in [
        {"provider_only": "unknown"},
        {"provider_only": "deepinfra", "provider": "aiclub"},
        {"provider_only": "deepinfra", "compare_routing": True},
    ]:
        with pytest.raises(ValueError):
            make_plan(**{**plan, **overrides})


@requires_posix
def test_rate_limit_stops_future_trials_and_higher_load(packaged, monkeypatch):
    from tools import vlm_multistream_benchmark as bench

    app, _ = packaged
    monkeypatch.setattr(os, "environ", dict(os.environ))
    calls = []

    def capture(args, base, count, framesets, target, repeat):
        calls.append(count)
        result = stage(count)
        if count > 1:
            result["per_stream"][0]["http_status_counts"] = {"429": 1}
            result["per_stream"][0]["errors"] = {"http_error": 1}
        return result

    monkeypatch.setattr(bench, "run_stage", capture)
    plan = make_plan(simulate=True, streams=[8, 12], soak_seconds=900)
    assert cloud.run(app, plan, app.parent / "results") == 2
    assert calls == [1, 8]
    summary = json.loads(
        (app.parent / "results/summary.json").read_text(encoding="utf-8")
    )
    assert summary["state"] == "failed"
    assert "429" in summary["reason"]
    assert len(summary["jobs"]) == 2
    assert not summary.get("engineering_capacity_gate_passed", False)
