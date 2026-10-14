import json
import subprocess
import sys

import pytest
from test_vlm_mode_benchmark import lane_report

from tools.vlm_benchmark import modes, parallel_concurrency, qwen35b_alibaba
from tools.vlm_benchmark.parallel_sweep_report import analyze, render
from tools.vlm_multistream_benchmark import paired_summary, summarize


def test_cloud_failure_report_works_before_dependency_installation(tmp_path):
    args = modes.parser().parse_args(qwen35b_alibaba.parallel_arguments())
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"settings": vars(args), "event_labels": modes.LABELS}))
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "import json, sys; from pathlib import Path; "
            "from tools.vlm_benchmark.cloud_modes import failure_report; "
            "failure_report(Path(sys.argv[2]), "
            "json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')), "
            "'failed', 'dependency install failed')",
            str(plan),
            str(tmp_path / "result"),
        ],
        cwd=modes.ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "result/parallel-concurrency.html").is_file()


@pytest.mark.parametrize("cloud", [False, True])
def test_fixed_parallel_plan_same_locally_and_remotely(cloud, monkeypatch, capsys):
    def forbidden(*a, **kw):
        pytest.fail("plan must not read secrets or run inference")

    monkeypatch.setattr(qwen35b_alibaba, "read_api_key", forbidden)
    monkeypatch.setattr(modes, "run", forbidden)
    assert (
        parallel_concurrency.main(
            ["--real", "--plan-only", *(["--cloud"] if cloud else [])]
        )
        == 0
    )
    output = capsys.readouterr().out
    plan = json.loads(output.split("\n", 3)[3] if cloud else output)
    s = plan["settings"]
    assert s["model"] == "qwen/qwen3.5-35b-a3b"
    assert s["provider_prefer"] == "alibaba" and not s["provider_only"]
    assert s["modes"] == ["parallel"] and s["streams"] == [12]
    assert s["require_route_match"] and s["parallel_concurrency_sweep"] == [4, 6, 8]
    assert s["record"] and s["monitor_layout"] == "single-video"
    assert s["request_rate"] == 40 and s["observation_fps"] == 1.25
    assert s["queue_timeout"] == 1 and s["http_timeout"] == 20
    assert s["grounding_format"] == "lean-v1"
    assert plan["request_budget"] == {
        "measurement_upper": 13248,
        "preflight_and_warmup_upper": 216,
        "optional_soak_upper": 0,
        "total_upper": 13464,
    }


def test_sweep_has_independent_control_and_changes_all_three_limits():
    args = modes.parser().parse_args(qwen35b_alibaba.parallel_arguments())
    modes.validate(args)
    jobs = list(modes.jobs(args))
    assert [(j["parallel_scheduling"], j["limits"]["global_limit"]) for j in jobs] == [
        ("independent", 48),
        ("paired", 48),
        ("paired", 72),
        ("paired", 96),
        ("paired", 96),
        ("paired", 72),
        ("paired", 48),
        ("independent", 48),
    ]
    assert len({j["id"] for j in jobs}) == 8
    for job in jobs:
        c = job["limits"]["per_stream_limit"]
        assert job["limits"] == dict(
            per_stream_limit=c, local_concurrency=c, global_limit=12 * c
        )
    routing = modes.configure(args).inference.request_parameters["provider"]
    assert routing["order"] == ["alibaba"] and routing["allow_fallbacks"]
    assert "only" not in routing and "ignore" not in routing


@pytest.mark.parametrize(
    "extra",
    [
        ["--modes", "events"],
        ["--streams", "1"],
        ["--soak-seconds", "30"],
        ["--grounding-concurrency-sweep", "4"],
        ["--grounding-format", "legacy"],
        ["--parallel-concurrency-sweep", "6", "8"],
    ],
)
def test_incomparable_sweep_rejected(extra):
    with pytest.raises(ValueError, match="parallel sweep"):
        modes.validate(
            modes.parser().parse_args([*qwen35b_alibaba.parallel_arguments(), *extra])
        )


def records():
    def row(lane, window, attempted=True, error=None):
        r = lane_report(lane, window, error=error)
        r["timing"].update(
            http_request_attempted=attempted,
            inspection_attempted=attempted,
            admission_wait_seconds=0.2,
            http_timing={"total_seconds": 2} if attempted else {},
        )
        return r

    return [
        row("information", "a"),
        row("grounding", "a"),
        row("information", "b"),
        row("grounding", "b", False, "admission_timeout"),
        row("information", "c"),
        row("grounding", "c", True, "timeout"),
    ]


def test_paired_delivery_does_not_confuse_unsent_with_failed_or_legacy_unknown():
    rows = records()
    result = paired_summary([rows], 4)
    assert result["both_http_attempted"] == 2
    assert result["both_complete"] == 1
    assert result["both_complete_fraction"] == 0.25
    assert result["both_http_attempted_fraction"] == 0.5
    assert result["both_complete_fraction_of_both_http_attempted"] == 0.5
    assert (
        result["only_one_http_attempted"] == 1 and result["neither_http_attempted"] == 1
    )
    assert result["one_complete_peer_http_not_attempted"] == 1
    assert result["one_complete_peer_http_attempted_without_complete"] == 1
    rows[0]["timing"].pop("http_request_attempted")
    result = paired_summary([rows], 4)
    assert result["http_attempt_unknown_records"] == 1
    assert result["both_http_attempted_fraction"] is None
    assert result["neither_http_attempted"] is None
    assert not paired_summary([[*records(), records()[0]]], 4)[
        "http_pair_accounting_known"
    ]


def stage_fixture(location, count=12, repeat=1, *, fatal=None):
    rows = records()
    if fatal:
        rows[0]["timing"]["http_status"] = fatal
    lanes = {
        lane: summarize(rows, {}, 3, 4, 3, lane=lane)
        for lane in ["events", "grounding"]
    }
    folder = location / f"streams-{count}-repeat-{repeat}"
    folder.mkdir()
    path = folder / "reports.jsonl"
    path.write_text(
        "".join(json.dumps({"camera": 1, "report": r}) + "\n" for r in rows)
    )
    return dict(
        streams=count,
        repeat=repeat,
        lanes=lanes,
        per_stream=[{"lanes": lanes}],
        input_timing_valid=True,
        all_drained=True,
        result_records_missing=0,
        result_retention={"valid": True},
        shared_admission={"active": 0, "waiting": 0},
        paired_windows=paired_summary([rows], 4),
        input_schedule_span_seconds=3,
        drain_elapsed_seconds=0.1,
        nominal_requests_per_second=30,
        rate_limit_below_offered_load=False,
    ), path


def test_report_retains_funnel_queues_and_requires_retention(tmp_path):
    stage, path = stage_fixture(tmp_path)
    capacity = analyze(stage, path, real=True)
    assert capacity["valid"]
    assert capacity["lanes"]["events"]["http_attempt_fraction"] == 0.75
    assert capacity["lanes"]["grounding"]["api_all_attempts"]["count"] == 2
    assert capacity["lanes"]["grounding"]["queue_not_attempted"]["count"] == 1
    assert capacity["pairs"]["both_http_attempted_fraction"] == 0.5
    stage["result_retention"]["valid"] = False
    assert not analyze(stage, path, real=True)["valid"]
    args = modes.parser().parse_args(qwen35b_alibaba.parallel_arguments())
    job = next(iter(modes.jobs(args)))
    doc = {
        "state": "completed",
        "settings": vars(args),
        "jobs": [
            {
                **job,
                "capacity": capacity,
                "raw_reports": path.relative_to(tmp_path).as_posix(),
            }
        ],
    }
    render(tmp_path, doc)
    result = json.loads(
        (tmp_path / "parallel-concurrency.json").read_text(encoding="utf-8")
    )
    assert result["trials"][0]["pairs"]["both_complete"] == 1
    assert "events_queue_attempts_p95" in (
        tmp_path / "parallel-concurrency.csv"
    ).read_text(encoding="utf-8")
    assert "parallel-concurrency.csv" in (
        tmp_path / "parallel-concurrency.html"
    ).read_text(encoding="utf-8")


@pytest.mark.parametrize("fatal", [None, 429, 402, "retention"])
def test_execution_applies_job_scheduling_and_limits_and_stops_on_failure(
    tmp_path, monkeypatch, fatal
):
    args = modes.parser().parse_args(qwen35b_alibaba.parallel_arguments())
    args.record = args.monitor = False
    args.duration = args.warmup_seconds = 3
    args.repeats = 1
    args.output = tmp_path / "run"
    seen = []

    class NoTelemetry:
        def __init__(self, *a):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def summary(self):
            return {}

    def fake_stage(options, base, count, frames, location, repeat, **kwargs):
        if count == 12:
            seen.append(
                (
                    options.parallel_scheduling,
                    options.local_concurrency,
                    options.per_stream_limit,
                    options.global_limit,
                )
            )
        result, _ = stage_fixture(
            location,
            count,
            repeat,
            fatal=fatal if isinstance(fatal, int) and count == 12 else None,
        )
        if fatal == "retention" and count == 12:
            result["result_retention"]["valid"] = False
        return result

    monkeypatch.setattr(modes, "Telemetry", NoTelemetry)
    monkeypatch.setattr(modes, "run_stage", fake_stage)
    if fatal:
        with pytest.raises(RuntimeError):
            modes.run(args)
        assert seen == [("independent", 4, 4, 48)]
    else:
        modes.run(args)
        assert seen == [
            ("independent", 4, 4, 48),
            ("paired", 4, 4, 48),
            ("paired", 6, 6, 72),
            ("paired", 8, 8, 96),
        ]
    report = json.loads(
        (args.output / "parallel-concurrency.json").read_text(encoding="utf-8")
    )
    assert report["state"] == ("failed" if fatal else "completed")
    assert len(report["trials"]) == len(seen)
